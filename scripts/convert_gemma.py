#!/usr/bin/env python3
"""Export Gemma 4 E2B IT as a shared-weight Swev text/image package.

Fixed 384px image canvas, 64 image tokens, ten answer labels. No audio/video.
Use --check-only to generate source references without exporting weights.
"""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from package_models import bundle, check_output
from prepare_model import annotate
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4MultimodalEmbedder,
    Gemma4VisionModel,
    Gemma4VisionRotaryEmbedding,
)

SIZE = 384
TOKENS = 64
LABELS = list("ABCDEFGHIJ")
PREFIX = "Choose the best answer to the question using the state below. Reply with only the answer letter.\n\nState: "
BETWEEN = "\n\nQuestion: "
OPTIONS = "\n\nAnswers:\n"
END = "\nAnswer:"


def load_text(checkpoint):
    torch.set_num_threads(8)

    config = AutoConfig.from_pretrained(
        str(checkpoint), local_files_only=True
    ).text_config
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Gemma4ForCausalLM(config)
    state = {}
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as f:
        for name in model.state_dict():
            key = (
                "model.language_model." + name.removeprefix("model.")
                if name != "lm_head.weight"
                else "model.language_model.embed_tokens.weight"
            )
            state[name] = f.get_tensor(key)
    model.load_state_dict(state, strict=True, assign=True)
    model = model.float().eval()
    # Nonpersistent rotary buffers are not stored in the checkpoint.
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    model.model.rotary_emb = Gemma4TextRotaryEmbedding(config)
    for module in model.modules():
        if hasattr(module, "scalar_embed_scale"):
            module.embed_scale = torch.tensor(module.scalar_embed_scale)
    if any(x.is_meta for x in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Text checkpoint left uninitialized tensors")
    model.tie_weights()
    print("loaded exact checkpoint tensors", len(state), flush=True)
    return model


class TextDecision(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids, option_indices, position_ids, decision_indices, attention_bias
    ):
        out = self.model.model(
            input_ids=input_ids.long(),
            position_ids=position_ids.long(),
            attention_mask={
                "full_attention": attention_bias,
                "sliding_attention": attention_bias,
            },
            use_cache=False,
        ).last_hidden_state
        h = torch.index_select(out, 1, decision_indices.long()).squeeze(1)
        weights = self.model.lm_head.weight[option_indices.long()[0]]
        logits = torch.matmul(h, weights.transpose(0, 1))
        cap = self.model.config.final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits


def load_vision(config, checkpoint):
    config.vision_config._attn_implementation = "eager"
    with torch.device("meta"):
        vision = Gemma4VisionModel(config.vision_config)
        projector = Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt") as f:
        for module, prefix in [
            (vision, "model.vision_tower."),
            (projector, "model.embed_vision."),
        ]:
            state = {k: f.get_tensor(prefix + k) for k in module.state_dict()}
            module.load_state_dict(state, strict=True, assign=True)
    vision = vision.float().eval()
    projector = projector.float().eval()
    vision.encoder.rotary_emb = Gemma4VisionRotaryEmbedding(config.vision_config)
    if any(t.is_meta for t in list(vision.parameters()) + list(vision.buffers())):
        raise ValueError("Vision checkpoint left uninitialized tensors")
    return vision, projector


class VisionWrapper(torch.nn.Module):
    def __init__(self, vision, projector):
        super().__init__()
        self.vision = vision
        self.projector = projector
        yy, xx = torch.meshgrid(torch.arange(24), torch.arange(24), indexing="ij")
        pos = torch.stack([xx.flatten(), yy.flatten()], dim=-1)[None]
        self.register_buffer("positions", pos)
        self.register_buffer("padding", torch.zeros(1, 576, dtype=torch.bool))
        idx = (xx.flatten() // 3) + (yy.flatten() // 3) * 8
        self.register_buffer(
            "pool_weights",
            torch.nn.functional.one_hot(idx, 64).float().transpose(0, 1)[None] / 9,
        )

    def patches(self, pixels):
        return (
            pixels.reshape(1, 24, 16, 24, 16, 3)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(1, 576, 768)
        )

    def forward(self, pixels):
        h = self.vision.patch_embedder(
            self.patches(pixels), self.positions, self.padding
        )
        emb = self.vision.encoder.rotary_emb(h, self.positions)
        for layer in self.vision.encoder.layers:
            h = layer(
                h,
                position_embeddings=emb,
                attention_mask=None,
                position_ids=self.positions,
            )
        h = torch.matmul(self.pool_weights, h) * math.sqrt(768)
        return self.projector(h)


class ImageDecision(torch.nn.Module):
    def __init__(self, text, vision, image_id):
        super().__init__()
        self.text = text
        self.vision = vision
        self.image_id = image_id

    def forward(
        self,
        input_ids,
        option_indices,
        position_ids,
        decision_indices,
        attention_bias,
        image_pixels,
    ):
        mask = input_ids == self.image_id
        ids = torch.where(mask, 0, input_ids).long()
        embedding = self.text.model.embed_tokens(ids)
        per_layer = self.text.model.get_per_layer_inputs(ids, embedding)
        features = self.vision(image_pixels)
        ordinal = torch.clamp(torch.cumsum(mask.int(), dim=1) - 1, 0, TOKENS - 1).long()
        images = torch.index_select(features, 1, ordinal[0])
        embedding = torch.where(mask.unsqueeze(-1), images, embedding)
        h = self.text.model(
            inputs_embeds=embedding,
            per_layer_inputs=per_layer,
            position_ids=position_ids.long(),
            attention_mask={
                "full_attention": attention_bias,
                "sliding_attention": attention_bias,
            },
            use_cache=False,
        ).last_hidden_state
        last = torch.index_select(h, 1, decision_indices.long()).squeeze(1)
        weights = self.text.lm_head.weight[option_indices.long()[0]]
        logits = torch.matmul(last, weights.transpose(0, 1))
        cap = self.text.config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap if cap is not None else logits


def preprocess(path):
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    base = Image.new("RGBA", image.size, "white")
    base.alpha_composite(image)
    src = np.asarray(base.convert("RGB"))
    h, w = src.shape[:2]
    scale = min(SIZE / w, SIZE / h)
    nw = max(1, int(w * scale + 0.5))
    nh = max(1, int(h * scale + 0.5))
    xs = np.minimum(w - 1, np.floor((np.arange(nw) + 0.5) * w / nw).astype(int))
    ys = np.minimum(h - 1, np.floor((np.arange(nh) + 0.5) * h / nh).astype(int))
    out = np.full((SIZE, SIZE, 3), 255, np.uint8)
    x = (SIZE - nw) // 2
    y = (SIZE - nh) // 2
    out[y : y + nh, x : x + nw] = src[ys[:, None], xs[None, :]]
    return out.astype(np.float32) / 255


def text(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    )


def prompt(request, image_tokens=""):
    if len(request["questions"]) != 1:
        raise ValueError("Reference cases must each contain one question")
    question = next(iter(request["questions"].values()))
    if question["type"] == "noul":
        options = ["No", "Yes"]
    elif question["type"] == "score":
        options = [text(value) for value in question["criteria"]]
    else:
        options = [
            key + (": " + text(value) if value is not None else "")
            for key, value in question["criteria"].items()
        ]
    if not 2 <= len(options) <= len(LABELS):
        raise ValueError(f"This exporter supports 2–{len(LABELS)} candidates")
    content = (
        PREFIX
        + text(request["state"])
        + BETWEEN
        + text(question["instructions"])
        + OPTIONS
    )
    content += (
        "".join(f"{LABELS[i]}. {value}\n" for i, value in enumerate(options)) + END
    )
    return (
        "<bos><|turn>user\n" + image_tokens + content + "<turn|>\n<|turn>model\n",
        len(options),
    )


def inputs(ids, tokenizer, length, pixels=None):
    count = len(ids)
    if not 0 < count <= length:
        raise ValueError(f"Prompt requires {count} tokens, capacity is {length}")
    candidates = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(len(value) != 1 for value in candidates):
        raise ValueError("Candidate letters must be single tokens")
    query, key = np.arange(length)[:, None], np.arange(length)[None, :]
    result = {
        "input_ids": np.array(
            [ids + [tokenizer.pad_token_id] * (length - count)], np.int32
        ),
        "option_indices": np.array([[value[0] for value in candidates]], np.int32),
        "position_ids": np.array(
            [list(range(count)) + [0] * (length - count)], np.int32
        ),
        "decision_indices": np.array([count - 1], np.int32),
        "attention_bias": np.where(
            ((key <= query) & (key < count)) | (key == query), 0, -10000
        ).astype(np.float32)[None, None],
    }
    if pixels is not None:
        result["image_pixels"] = pixels[None].astype(np.float32)
    return result


def recipe(image=False):
    def literal(value):
        return {"op": "literal", "value": value}

    def field(value):
        return {"op": "field", "value": value}

    def expression(op, *args):
        return {"op": op, "args": list(args)}

    def formatted(value):
        return {"op": "format", "value": "text-or-json", "args": [value]}

    def segment(kind, value):
        return {"kind": kind, "value": value}

    label = expression("indexed", *(literal(value) for value in LABELS))
    description = field("description")
    suffix = expression(
        "present",
        description,
        expression("concat", literal(": "), formatted(description)),
        literal(""),
    )

    def option(value):
        return expression("concat", label, literal(". "), value, literal("\n"))

    segments = [
        segment("token", "<bos>"),
        segment("token", "<|turn>"),
        segment("text", "user\n"),
    ]
    if image:
        segments.append({"kind": "image"})
    segments += [
        segment("text", PREFIX),
        segment("group", "state"),
        segment("text", BETWEEN),
        segment("group", "instructions"),
        segment("text", OPTIONS),
        {"kind": "options", "segments": [{"kind": "option"}]},
        segment("text", END),
        segment("token", "<turn|>"),
        segment("text", "\n"),
        segment("token", "<|turn>"),
        segment("text", "model\n"),
    ]
    return {
        "tokenization": "joined",
        "candidateTokens": LABELS,
        "padToken": "<pad>",
        "state": formatted(field("state")),
        "instructions": formatted(field("instructions")),
        "options": {
            "choice": option(expression("concat", field("id"), suffix)),
            "score": option(formatted(description)),
            "noul": option(
                expression(
                    "concat",
                    expression("indexed", literal("No"), literal("Yes")),
                    suffix,
                )
            ),
        },
        "replacements": [],
        "segments": segments,
        "groupLimits": {},
        "scoreLegend": "json",
    }


def contract(args, tokenizer, image=False):
    length = args.image_length if image else args.text_length
    config = {
        "contractVersion": "1.0",
        "modelVersion": "0.2.0",
        "revision": args.revision,
        "id": args.output.stem,
        "architecture": "causal-language-model",
        "capabilities": {
            "modalities": ["text", "image"] if image else ["text"],
            "questionTypes": ["choice", "score", "noul"],
            "limits": {
                "maxQuestionsPerRequest": 64,
                "maxOptionsPerQuestion": len(LABELS),
                "maxSequenceTokens": length,
            },
        },
        "execution": {
            "profile": "vision-decision-v1" if image else "text-decision-v1",
            "inputAdapter": "text-recipe-v1",
        },
    }
    preprocessing = {
        "sequenceLength": length,
        "optionCapacity": len(LABELS),
        "tensors": "causal-labels",
        "recipe": recipe(image),
    }
    if image:
        preprocessing["image"] = {
            "width": SIZE,
            "height": SIZE,
            "resize": "fit-nearest",
            "background": [255, 255, 255],
            "tokenSequence": tokenizer.boi_token
            + tokenizer.image_token * TOKENS
            + tokenizer.eoi_token
            + "\n",
        }
    return {
        "swev.config": config,
        "swev.preprocessing": preprocessing,
        "swev.postprocessing": {
            "temperatures": [1, 1, 1],
            "temperaturesByOptions": {},
            "choiceConfidence": "normalized_entropy_v1",
            "scoreConfidence": "normalized_entropy_v1",
        },
        "swev.tokenizer.tokenizer.json": (
            args.checkpoint / "tokenizer.json"
        ).read_text(),
        "swev.provenance": {
            "checkpoint": "google/gemma-4-E2B-it",
            "revision": args.revision,
            "changes": "FP32 restricted next-token candidate logits. Audio omitted. Fixed 384px white nearest-fit image canvas and 64 image tokens.",
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def reference_case(case, ids, count, data, logits, wrapper, tolerance):
    actual = (
        wrapper(*(torch.from_numpy(value) for value in data.values()))
        .detach()
        .numpy()[0]
    )
    error = float(np.max(np.abs(logits - actual)))
    if error > tolerance or not np.isfinite(error):
        raise ValueError(f"Source/wrapper mismatch for {case['id']}: {error}")
    print(f"{case['id']}: source/wrapper max error {error:.7g}", flush=True)
    return {
        **case,
        "ids": ids,
        "options": data["option_indices"][0, :count].tolist(),
        "source_logits": logits.tolist(),
        "wrapper_max_error": error,
        "probabilities": torch.softmax(
            torch.from_numpy(logits[:count]).double(), -1
        ).tolist(),
    }


def export(wrapper, data, records, destination):
    import coremltools as ct

    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper,
            tuple(torch.from_numpy(value) for value in data.values()),
            check_trace=False,
        )
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name=key, shape=value.shape, dtype=value.dtype)
            for key, value in data.items()
        ],
        outputs=[ct.TensorType(name="option_logits", dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=True,
        package_dir=str(destination),
    )
    annotate(model, records)
    ct.utils.save_spec(
        model.get_spec(), str(destination / "Data/com.apple.CoreML/model.mlmodel")
    )


def run(args):
    check_output(args.output) if not args.check_only else None
    args.work_dir.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(str(args.checkpoint), local_files_only=True)
    expected = {
        "hidden_size": 1536,
        "num_hidden_layers": 35,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "vocab_size": 262144,
    }
    if config.model_type != "gemma4" or any(
        getattr(config.text_config, key) != value for key, value in expected.items()
    ):
        raise ValueError("This exporter supports Gemma 4 E2B IT only")
    if (
        config.vision_config.patch_size != 16
        or config.vision_config.hidden_size != 768
        or config.vision_config.pooling_kernel_size != 3
    ):
        raise ValueError("Unsupported vision configuration")
    if (
        not 8
        <= args.text_length
        <= args.image_length
        <= config.text_config.sliding_window
    ):
        raise ValueError(
            "Lengths must fit the sliding window; larger contexts require a separate sliding mask"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.checkpoint), local_files_only=True
    )
    text_model = load_text(args.checkpoint)
    text_wrapper = TextDecision(text_model).eval()
    text_records, image_records = (
        contract(args, tokenizer),
        contract(args, tokenizer, True),
    )
    write_json(
        args.work_dir / "recipe.json", text_records["swev.preprocessing"]["recipe"]
    )
    write_json(
        args.work_dir / "preprocessing.json", image_records["swev.preprocessing"]
    )
    references = []
    with torch.inference_mode():
        for case in json.loads(args.cases.read_text()):
            rendered, count = prompt(case["request"])
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            data = inputs(ids, tokenizer, args.text_length)
            source = (
                text_model(
                    input_ids=torch.tensor([ids]),
                    attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                    use_cache=False,
                    logits_to_keep=1,
                )
                .logits[0, -1, data["option_indices"][0]]
                .numpy()
            )
            references.append(
                reference_case(
                    case, ids, count, data, source, text_wrapper, args.tolerance
                )
            )
    if not references:
        raise ValueError("At least one text case is required")
    text_data = data
    write_json(args.work_dir / "text-reference.json", references)
    samples = [
        "apple",
        "你好",
        "e\u0301",
        "é",
        "🧑🏽‍🚀",
        " leading  spaces\n",
        "<bos>A<|turn>B",
    ]
    write_json(
        args.work_dir / "tokenizer-reference.json",
        [
            {"text": value, "ids": tokenizer.encode(value, add_special_tokens=False)}
            for value in samples
        ],
    )
    write_json(
        args.work_dir / "text-manifest.json",
        {
            "tokenizers": [
                {
                    "path": str(args.checkpoint / "tokenizer.json"),
                    "reference": str(args.work_dir / "tokenizer-reference.json"),
                }
            ],
            "adapters": [
                {
                    "tokenizer": str(args.checkpoint / "tokenizer.json"),
                    "recipe": str(args.work_dir / "recipe.json"),
                    "sequenceLength": args.text_length,
                    "optionCapacity": len(LABELS),
                    "reference": str(args.work_dir / "text-reference.json"),
                }
            ],
            "models": [
                {
                    "path": str(args.output),
                    "cases": str(args.cases.resolve()),
                    "reference": str(args.work_dir / "text-reference.json"),
                }
            ],
        },
    )
    config.audio_config = None
    config.text_config._attn_implementation = "eager"
    vision, projector = load_vision(config, args.checkpoint)
    vision_wrapper = VisionWrapper(vision, projector).eval()
    image_wrapper = ImageDecision(
        text_model, vision_wrapper, config.image_token_id
    ).eval()
    with torch.device("meta"):
        source_model = Gemma4ForConditionalGeneration(config)
    source_model.model.language_model = text_model.model
    source_model.lm_head = text_model.lm_head
    source_model.model.vision_tower = vision
    source_model.model.embed_vision = projector
    source_model.eval()
    if args.image_cases:
        image_cases = json.loads(args.image_cases.read_text())
        image_base = args.image_cases.parent
    else:
        from PIL import Image

        image_base = args.work_dir
        image_cases = []
        for color in ("red", "green", "blue"):
            Image.new("RGB", (96, 64), color).save(image_base / (color + ".png"))
            image_cases.append(
                {
                    "id": color,
                    "image": color + ".png",
                    "request": {
                        "state": "An image is attached.",
                        "questions": {
                            "color": {
                                "type": "choice",
                                "instructions": "What is the main color?",
                                "criteria": {
                                    "red": "red",
                                    "green": "green",
                                    "blue": "blue",
                                },
                            }
                        },
                    },
                }
            )
    image_references = []
    with torch.inference_mode():
        for index, case in enumerate(image_cases):
            image_path = (image_base / case["image"]).resolve()
            pixels = preprocess(image_path)
            rendered, count = prompt(
                case["request"],
                image_records["swev.preprocessing"]["image"]["tokenSequence"],
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            data = inputs(ids, tokenizer, args.image_length, pixels)
            source = (
                source_model(
                    input_ids=torch.tensor([ids]),
                    attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                    pixel_values=vision_wrapper.patches(torch.from_numpy(pixels[None])),
                    image_position_ids=vision_wrapper.positions,
                    use_cache=False,
                    logits_to_keep=1,
                )
                .logits[0, -1, data["option_indices"][0]]
                .numpy()
            )
            row = reference_case(
                case, ids, count, data, source, image_wrapper, args.tolerance
            )
            pixel_path = args.work_dir / f"image-{index}.f32"
            pixels.astype("<f4").tofile(pixel_path)
            row.update(
                image=os.path.relpath(image_path, args.work_dir), pixels=pixel_path.name
            )
            image_references.append(row)
    if not image_references:
        raise ValueError("At least one image case is required")
    write_json(args.work_dir / "image-reference.json", image_references)
    write_json(
        args.work_dir / "image-manifest.json",
        {
            "model": str(args.output),
            "preprocessing": str(args.work_dir / "preprocessing.json"),
            "cases": str(args.work_dir / "image-reference.json"),
            "textReference": str(args.work_dir / "text-reference.json"),
        },
    )
    if args.check_only:
        return
    with tempfile.TemporaryDirectory(
        prefix="swev-export-", dir=args.output.parent
    ) as temp:
        temp = Path(temp)
        text_path, image_path = temp / "text.mlpackage", temp / "image.mlpackage"
        export(text_wrapper, text_data, text_records, text_path)
        export(image_wrapper, data, image_records, image_path)
        bundle(text_path, image_path, args.output)
    print(args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local, unsharded model.safetensors checkpoint directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--revision",
        required=True,
        help="Pinned source commit, recorded in the package",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Generated references and integration manifests",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "fixtures/text-cases.json",
    )
    parser.add_argument(
        "--image-cases",
        type=Path,
        help="Optional JSON cases with id, image, request; paths relative to this file",
    )
    parser.add_argument("--text-length", type=int, default=128)
    parser.add_argument("--image-length", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    for name in ("checkpoint", "output", "work_dir", "cases", "image_cases"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if len(args.revision) != 40 or any(
        value not in "0123456789abcdef" for value in args.revision
    ):
        parser.error("--revision must be a full lowercase commit SHA")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
