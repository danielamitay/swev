# Converting models

Conversion runs offline in Python on macOS; the resulting Swift runtime has no Python dependency. A Swev package includes its graph, weights, tokenizer, prompt recipe, calibration, and [schema 1.0 metadata](schema.md). Merely renaming a checkpoint or adding metadata does not make it compatible.

The included exporter targets **Gemma 4 E2B IT**. Other architectures need a wrapper that implements one of the [supported tensor contracts](models.md), followed by the generic packaging tools. Diffusion models are not supported by these scripts.

## Environment

Use Apple Silicon, Python 3.12, and the pinned conversion dependencies:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r scripts/conversion-requirements.txt
uv tool install huggingface_hub
```

The Gemma workflow was validated on a 128 GB Mac. Plan for at least 100 GB of free disk space for checkpoint, temporary exports, final packages, and Core ML compilation. FP32 export uses substantial RAM. Avoid concurrent large conversions or inference runs. Store downloads and reports under `.local/`; model assets and conversion environments are ignored by Git.

## Gemma 4 E2B IT

Accept the source model's access terms on Hugging Face, then authenticate with `hf auth login` if needed. Download the pinned source snapshot using the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```sh
hf download google/gemma-4-E2B-it \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --include '*.json' 'model.safetensors' \
  --local-dir .local/gemma/checkpoint
```

The exporter expects that unsharded `model.safetensors` layout. It strictly loads the text and vision tensors and restores nonpersistent rotary buffers. It does not download or execute remote Python code. `--revision` records the source identity; obtain the checkpoint with the matching pinned download command rather than labeling an arbitrary local directory with that revision.

```sh
.venv/bin/python scripts/convert_gemma.py \
  --checkpoint .local/gemma/checkpoint \
  --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 \
  --output .local/ready/gemma-4-e2b-it-swev-fp32-l256-k4.mlpackage \
  --work-dir .local/gemma-validation
```

This produces **one package** with shared weights: a 128-token text graph and a 256-token image graph. Requests without an image skip vision. Temporary individual exports are removed after bundling. Existing output paths are rejected. An unsuccessful export never replaces an existing final package.

Defaults are four candidate labels, FP32 computation, a 384×384 RGB image canvas, white nearest-neighbor aspect-fit letterboxing, and 64 image tokens. Audio/video are omitted. The model scores restricted next-token answer letters; it is not a generative chat interface or a separately trained decision head. Long requests throw context overflow. Length overrides cannot exceed the source sliding window because this wrapper uses one attention mask for both attention types.

Before export, the script checks the wrapper against source-model logits for the repository's 17 text cases and three generated color images. The image comparison uses the same fixed preprocessing, not the source processor's variable-resolution policy. Use `--image-cases path/to/cases.json` for a broader image suite; each entry has `id`, `image`, and a single-question `request`, with image paths relative to the JSON file. Generated references and integration manifests go into `--work-dir`.

Add `--check-only` to run source/wrapper checks and generate references without exporting. This can target an existing output package for subsequent Swift parity checks. The default maximum logit difference is `1e-4`; do not loosen it to conceal a conversion error.

## INT4 weights

```sh
.venv/bin/python scripts/package_models.py quantize \
  .local/ready/gemma-4-e2b-it-swev-fp32-l256-k4.mlpackage \
  .local/ready/gemma-4-e2b-it-swev-int4-l256-k4.mlpackage
```

This applies symmetric INT4 block quantization (32 weights per block) to eligible weights in both graphs, then deduplicates shared weights again. Core ML Tools' default minimum weight threshold is 2048 elements; small or unsupported constants remain uncompressed. Computation precision is unchanged. This is **INT4 weight compression, not NVIDIA NVFP4 or four-bit floating-point execution**. Smaller files do not guarantee lower runtime memory or latency; Core ML may expand weights when loading.

The validated Gemma artifacts were approximately 19.16 GB FP32 and 3.02 GB INT4. Their text smoke tests passed 17/17; the separate 15-image suite passed 14/15 for both. These are small smoke tests, not general accuracy claims. Every new checkpoint or quantization needs its own evaluation.

## Validate in Swift

Build and run source parity for FP32:

```sh
swift build -c release
SWEV_TEST_MANIFEST="$PWD/.local/gemma-validation/text-manifest.json" swift test
SWEV_IMAGE_TEST_MANIFEST="$PWD/.local/gemma-validation/image-manifest.json" \
  swift test --filter imageModelReferenceParity

.venv/bin/python scripts/evaluate.py \
  --model .local/ready/gemma-4-e2b-it-swev-fp32-l256-k4.mlpackage \
  --timeout 900 --driver .build/release/swev

.venv/bin/python scripts/evaluate.py \
  --model .local/ready/gemma-4-e2b-it-swev-int4-l256-k4.mlpackage \
  --timeout 900 --driver .build/release/swev
```

The source references check numerical parity, including tokenization, prompt assembly, image pixels, and returning to text after images. The labeled evaluator checks semantic correctness separately. INT4 changes probabilities, so strict FP32 parity is not an appropriate INT4 quality threshold; inspect probability drift and labeled results instead. The CLI evaluator currently accepts text only. For quantized image acceptance, use a Swift caller with the same labeled image cases and compare decisions separately from FP32 numerical parity.

Custom text cases can be supplied to the exporter with `--cases`. The existing `endToEndModels` Swift test uses the repository's fixed 17-case order; use the standard text cases for its generated manifest. The image integration test reads requests directly from its references.

Ordinary tests need no model downloads. Test the conversion tools with synthetic graphs (including native Swift loading of FP32 and INT4 outputs):

```sh
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
swift test
```

## Other compatible graphs

Export a single-function Core ML ML Program with one of the documented signatures. Write an explicit contract JSON containing `swev.config`, `swev.preprocessing`, `swev.postprocessing`, and the complete `swev.tokenizer.tokenizer.json` string. Tokenizer support is bounded to the formats described in [model support](models.md). Then validate signatures and embed the metadata:

```sh
.venv/bin/python scripts/prepare_model.py \
  .local/exported.mlpackage .local/prepared.mlpackage \
  --contract .local/contract.json
```

The tool derives actual tensor signatures and tokenizer hashes; it does not infer prompts, calibration, or architecture. Validate the prepared artifact with the Swift loader and source references before publishing.

If separate prepared text and image graphs share a tokenizer and postprocessing contract, combine them:

```sh
.venv/bin/python scripts/package_models.py bundle \
  --text .local/text.mlpackage --image .local/image.mlpackage \
  --output .local/ready/model.mlpackage
```

The bundler deduplicates weights at conversion time and embeds a single-function text specification in the root image package. It does not require Core ML multifunction selection at runtime. Matching source version/revision, question and option capacities, tokenizer, and postprocessing are required. The output filename stem becomes the model ID in both graphs. Test both routes after bundling; matching metadata alone cannot establish that two unrelated checkpoints behave identically.

Before sharing, keep only final packages outside Git, retain source revision and conversion settings in provenance, and publish the complete package directory with its applicable model license and evaluation results. Loading instructions are in [Hugging Face support](huggingface.md).
