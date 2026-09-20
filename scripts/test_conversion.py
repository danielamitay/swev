"""Synthetic conversion checks. Run with the pinned conversion environment."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from package_models import bundle, load, quantize, weight_references
from prepare_model import annotate


@unittest.skipUnless(
    importlib.util.find_spec("coremltools"), "Install conversion-requirements.txt"
)
class ConversionTests(unittest.TestCase):
    def test_bundle_and_quantize_preserve_routes_and_shared_weights(self):
        import base64

        import coremltools as ct
        import numpy as np
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types, get_new_symbol
        from coremltools.proto import Model_pb2
        from tokenizers import (
            AddedToken,
            Tokenizer,
            models,
            normalizers,
            pre_tokenizers,
        )

        tokenizer = Tokenizer(
            models.BPE(
                {
                    value: i
                    for i, value in enumerate(
                        sorted(pre_tokenizers.ByteLevel.alphabet())
                    )
                },
                [],
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.normalizer = normalizers.NFC()
        tokenizer.add_special_tokens([AddedToken("[PAD]", special=True)])
        weights = np.random.default_rng(0).normal(size=(64, 64)).astype(np.float32)

        def make(path, image):
            length = 512 if image else get_new_symbol()
            specs = [
                mb.TensorSpec((1, length), types.int32),
                mb.TensorSpec((1, 16), types.int32),
                mb.TensorSpec((1, length), types.int32),
                mb.TensorSpec((1,), types.int32),
                mb.TensorSpec((1, 1, length, length), types.fp32),
            ]

            def result(input_ids, image_pixels=None):
                ids = mb.cast(x=input_ids, dtype="fp32")
                value = mb.matmul(x=mb.slice_by_size(x=ids, begin=[0, 0], size=[1, 64]), y=weights)
                value = mb.reduce_mean(x=value, axes=[0, 1], keep_dims=True)
                if image_pixels is not None:
                    value = mb.add(
                        x=value,
                        y=mb.reduce_mean(
                            x=image_pixels, axes=[0, 1, 2, 3], keep_dims=False
                        ),
                    )
                return mb.tile(x=value, reps=[1, 16], name="option_logits")

            if image:

                @mb.program(
                    input_specs=specs + [mb.TensorSpec((1, 2, 2, 3), types.fp32)],
                    opset_version=ct.target.iOS18,
                )
                def program(
                    input_ids,
                    option_indices,
                    position_ids,
                    decision_indices,
                    attention_bias,
                    image_pixels,
                ):
                    return result(input_ids, image_pixels)
            else:

                @mb.program(input_specs=specs, opset_version=ct.target.iOS18)
                def program(
                    input_ids,
                    option_indices,
                    position_ids,
                    decision_indices,
                    attention_bias,
                ):
                    return result(input_ids)

            inputs = None
            if not image:
                inputs = [ct.TensorType(name=name, shape=ct.EnumeratedShapes(
                    shapes=[(1,1,n,n) if name == "attention_bias" else (1,n) for n in (128,512,4096)],
                    default=(1,1,128,128) if name == "attention_bias" else (1,128)))
                    for name in ("input_ids", "position_ids", "attention_bias")]
            model = ct.convert(
                program,
                inputs=inputs,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS18,
                compute_precision=ct.precision.FLOAT32,
                skip_model_load=True,
            )
            literal = lambda value: {"op": "literal", "value": value}
            recipe = {
                "tokenization": "joined",
                "candidateTokens": list("ABCDEFGHIJKLMNOP"),
                "padToken": "[PAD]",
                "state": {
                    "op": "format",
                    "value": "text-or-json",
                    "args": [{"op": "field", "value": "state"}],
                },
                "instructions": literal(""),
                "options": {
                    kind: literal("test") for kind in ("choice", "score", "noul")
                },
                "replacements": [],
                "segments": [
                    {"kind": "group", "value": "state"},
                    {"kind": "options", "segments": [{"kind": "option"}]},
                ],
                "groupLimits": {},
                "scoreLegend": "json",
            }
            if image:
                recipe["segments"].insert(0, {"kind": "image"})
            pre = {
                "sequenceLength": 512 if image else 4096,
                "optionCapacity": 16,
                "tensors": "causal-labels",
                "recipe": recipe,
            }
            if not image:
                pre["sequenceBuckets"] = [128,512,4096]
            if image:
                pre["image"] = {
                    "width": 2,
                    "height": 2,
                    "resize": "fit-nearest",
                    "background": [255] * 3,
                    "tokenSequence": "I",
                }
            records = {
                "swev.config": {
                    "contractVersion": "2.0",
                    "id": "test",
                    "modelVersion": "1",
                    "architecture": "test",
                    "execution": {
                        "profile": "vision-decision-v1"
                        if image
                        else "text-decision-v1",
                        "inputAdapter": "text-recipe-v1",
                    },
                    "capabilities": {
                        "modalities": ["text", "image"] if image else ["text"],
                        "questionTypes": ["choice", "score", "noul"],
                        "limits": {
                            "maxQuestionsPerRequest": 64,
                            "maxOptionsPerQuestion": 16,
                            "maxSequenceTokens": 512 if image else 4096,
                        },
                    },
                },
                "swev.preprocessing": pre,
                "swev.postprocessing": {
                    "temperatures": [1] * 3,
                    "temperaturesByOptions": {},
                    "choiceConfidence": "normalized_entropy_v1",
                    "scoreConfidence": "normalized_entropy_v1",
                },
                "swev.tokenizer.tokenizer.json": tokenizer.to_str(),
            }
            annotate(model, records).save(str(path))
            records["swev.config"]["contractVersion"] = "99.0"
            with self.assertRaisesRegex(ValueError, "contractVersion"):
                annotate(model, records)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text, image, output, compressed = [
                root / (name + ".mlpackage")
                for name in ("text", "image", "routed", "int4")
            ]
            make(text, False)
            make(image, True)
            bundle(text, image, output)
            quantize(output, compressed)
            for path in (output, compressed):
                model, config = load(path)
                self.assertEqual(
                    config["execution"]["profile"], "routed-vision-decision-v1"
                )
                self.assertEqual(config["id"], path.stem)
                record = json.loads(model.user_defined_metadata["swev.text-model"])
                spec = Model_pb2.Model()
                spec.ParseFromString(base64.b64decode(record["specification"]))
                self.assertEqual(set(spec.mlProgram.functions), {"main"})
                self.assertEqual(set(record["weights"]), weight_references(spec))
                self.assertEqual(
                    json.loads(record["metadata"]["swev.config"])["id"], path.stem
                )
                self.assertEqual(
                    len(list((path / "Data/com.apple.CoreML/weights").glob("*.bin"))), 1
                )
            self.assertLess(
                sum(p.stat().st_size for p in compressed.rglob("*.bin")),
                sum(p.stat().st_size for p in output.rglob("*.bin")),
            )
            # Exercise smaller requests and all sixteen candidate slots through the Swift loader.
            import subprocess

            cases = json.loads(
                (
                    Path(__file__).resolve().parents[1] / "fixtures/text-cases.json"
                ).read_text()
            )
            for kind, criteria in (
                ("choice", {letter: None for letter in "ABCDEFGHIJKLMNOP"}),
                ("score", list(map(str, range(16)))),
            ):
                cases.append({
                    "request": {
                        "state": "P" if kind == "choice" else "15",
                        "questions": {
                            "decision": {
                                "type": kind,
                                "instructions": "Choose the matching option.",
                                "criteria": criteria,
                            }
                        },
                    }
                })
            # Exercise different buckets and return to the resident short route.
            for repetitions in (150, 700, 1):
                cases.append({"request": {"state": "x " * repetitions, "questions": {
                    "q": {"type": "noul", "instructions": "Is this text?"}}}})
            case_path = root / "cases.json"
            case_path.write_text(json.dumps(cases))
            references = []
            for case in cases:
                question = next(iter(case["request"]["questions"].values()))
                count = 2 if question["type"] == "noul" else len(question["criteria"])
                references.append(
                    {"request": case["request"], "probabilities": [1 / count] * count}
                )
            reference = root / "reference.json"
            reference.write_text(json.dumps(references))
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "tokenizers": [],
                        "adapters": [],
                        "models": [
                            {"path": str(path), "reference": str(reference), "cases": str(case_path)}
                            for path in (output, compressed)
                        ],
                    }
                )
            )
            import os

            subprocess.run(
                ["swift", "test", "--filter", "endToEndModels"],
                cwd=Path(__file__).resolve().parents[1],
                env={**os.environ, "SWEV_TEST_MANIFEST": str(manifest)},
                check=True,
            )

            from PIL import Image

            Image.new("RGB", (2, 2), "red").save(root / "red.png")
            np.tile(np.array([1, 0, 0], dtype="<f4"), 4).tofile(root / "red.f32")
            image_cases = root / "images.json"
            image_cases.write_text(
                json.dumps(
                    [
                        {
                            "image": "red.png",
                            "pixels": "red.f32",
                            "request": cases[0]["request"],
                            "probabilities": [0.5, 0.5],
                        }
                    ]
                )
            )
            preprocessing = root / "preprocessing.json"
            preprocessing.write_text(
                load(output)[0].user_defined_metadata["swev.preprocessing"]
            )
            for path in (output, compressed):
                manifest.write_text(
                    json.dumps(
                        {
                            "model": str(path),
                            "preprocessing": str(preprocessing),
                            "cases": str(image_cases),
                            "textReference": str(reference),
                        }
                    )
                )
                subprocess.run(
                    ["swift", "test", "--filter", "imageModelReferenceParity"],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "SWEV_IMAGE_TEST_MANIFEST": str(manifest)},
                    check=True,
                )


if __name__ == "__main__":
    unittest.main()
