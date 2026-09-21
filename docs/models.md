# Model contract

`SwevModel.load(from:)` accepts a local Core ML package, model, or compiled model
with the Swev model contract. The public API is independent of model identity.
Model packages supply tokenizer data, formatting recipes, tensor layout, limits,
and scoring configuration. The runtime does not infer these from model weights.

The execution profiles are `text-decision-v1`, `vision-decision-v1`, and
`routed-vision-decision-v1`, all with input adapter `text-recipe-v1`. They support static batch-one decisions and an
`option_logits` output. Text models use `masked-options`, `causal-pointer`, or
`causal-labels` tensors. Image-capable models use `causal-labels` plus an
`image_pixels` float tensor. New neural architectures may require a new execution
profile; arbitrary packages are not automatically compatible.

## Embedded metadata

- `swev.config`: identity, versions, capabilities, limits, and execution profile.
- `swev.preprocessing`: `sequenceLength`, optional `sequenceBuckets`, `optionCapacity`, `tensors`, and `recipe`.
- `swev.postprocessing`: temperatures, optional option-count temperature buckets,
  and confidence methods for choices and scores.
- `swev.signatures`: tensor names, default shapes, optional enumerated shapes, and data types.
- `swev.tokenizer.asset-index`: UTF-8 tokenizer payload keys, byte counts, and SHA-256 hashes.
- `swev.provenance`: optional source identity and conversion information.

The tokenizer reads BPE vocabulary, merges, special tokens, and pre-tokenization
settings from the embedded tokenizer document. It supports byte-level BPE with
NFC or no normalization, using standard ByteLevel splitting, individual Digits
followed by ByteLevel, or an isolated regex Split followed by ByteLevel without
regex. Pruned byte vocabularies are accepted, but unrepresentable input throws
rather than silently dropping characters. It also supports Unicode BPE with literal space-marker
replacement and UTF-8 byte fallback. Vocabulary keys preserve exact Unicode
scalar sequences. Unsupported tokenizer features fail explicitly.

## Text recipes

A recipe contains `padToken`, `state`, `instructions`, `options`, `replacements`,
`segments`, `groupLimits`, optional `prefixBudget`, and `scoreLegend`.

State, instructions, and each question type's option template are expression
objects. Every expression has `op` and optional `value` and `args`:

| Operation | Meaning |
| --- | --- |
| `literal` | Use the string in `value`. |
| `field` | Read state, instructions, type, option id, description, or index. |
| `concat` | Join string arguments. |
| `truthy` | Select argument 1 or 2 based on argument 0's truthiness. |
| `present` | Select argument 1 unless argument 0 is null or an empty string. |
| `indexed` | Select an argument using the option index. |
| `format` | Format argument 0 using `python`, `indented`, `text-or-json`, or strict `string`. |

Regex replacements apply to rendered text before tokenization. Segment kinds
assemble tokens: `token` appends an explicit special token, `group` appends state
or instructions, `options` repeats child `segments`, `option` appends the current
option text, and `mark` records an option position (`value: previous` records the
previous token). Each positional option must have exactly one valid position.

For `causal-labels`, set `candidateTokens` to an ordered list of distinct,
single-token labels, one per candidate slot, and `tokenization` to `joined`.
The runtime joins the recipe's text before tokenization, preserving BPE merges
across segment boundaries. A `text` segment appends a literal string. Label
recipes omit `mark`; `option_indices` contains candidate token IDs instead of
prompt positions. The model scores those vocabulary entries at the decision
position. Probabilities are conditional on the offered labels, not confidence
that an unconstrained text generator would emit one. This does not require a
trained classification head. Other layouts retain segmented tokenization.

`groupLimits` bounds state, instructions, or each option's token count.
`prefixBudget` reserves `minimumInstructionSlots` inside `maximum`, charging each
option its tokens plus `optionOverhead`. Oversized input is rejected rather than
truncated. `scoreLegend` is `json` or `indented`.

Recipes contain data, not executable Swift, Python, or downloaded scripts.
Model-specific recipes and reference fixtures belong with the model release or
local conversion workspace, not this repository.

## Preparing an export

With coremltools installed:

```sh
python3 scripts/prepare_model.py exported.mlpackage prepared.mlpackage --contract contract.json
```

The contract file maps metadata keys to their JSON values and can include tokenizer
payloads. The helper verifies tensor signatures and bundles the supplied contract;
it does not convert weights or prove recipe correctness. Loading and source-parity
tests remain necessary. The destination must be a new path.

## Runtime behavior

Probabilities retain full precision. Noul is the probability of true; score is the
expected zero-based level. Confidence identifies its method and is not a probability
of correctness. Caller metadata is echoed without entering the prompt.

Inference is serialized per model, off the main actor. Admission defaults to eight
requests total, including the active request. Cancellation is checked between stages; an active device call
must finish before cancellation returns. Temporary compiled assets live with the
model instance. CPU-only execution is validated locally; other compute units and
iOS devices need their own validation. Tokenizer hashes detect corruption, not
whether a model source is trustworthy.

## Image inputs

Image-capable packages declare modalities `["text", "image"]`, the
`vision-decision-v1` profile, and an `image` object in `swev.preprocessing`:

```json
{"width":384,"height":384,"resize":"fit-nearest","background":[255,255,255],"tokenSequence":"MODEL_IMAGE_TOKENS"}
```

The joined text recipe contains exactly one `image` segment, which inserts the
package's `tokenSequence` when an image is supplied. The model graph must match
that token sequence and its image tensor. Text-only requests omit those tokens
and receive a zero image tensor; the graph must ignore that unused tensor.

Pass owned PNG or JPEG bytes through the existing API:

```swift
let response = try await model.predict(
    state: "Use the attached image.",
    questions: [.noul(id: "red", instructions: "Is the image predominantly red?")],
    images: [.init(data: imageData, contentType: "image/png")]
)
```

The initial image processor handles EXIF orientation, converts to sRGB, composites
transparency over the configured background, and fits the image into a fixed
canvas using nearest-neighbor sampling and centered padding. It produces RGB
Float32 values in `[0,1]`, in `[1,height,width,3]` order. These choices are explicit
model-contract requirements, not inferred preprocessing for arbitrary models.

At most one image is accepted per request. Encoded data is limited to 32 MiB,
decoded images to 16 megapixels and 8192 pixels per side, and model input canvases
to 1024 pixels per side. Animated images, mismatched content types, and invalid
images fail. Preprocessing runs once per request off the main actor; the initial
combined graph executes for each question. Vision feature caching, audio, and
video are not implemented. The JSON Lines CLI remains text-only; image input is
available through the typed Swift API.

## Shared-weight text and image packages

`routed-vision-decision-v1` combines the image contract with an embedded text-only
graph in `swev.text-model`. Swift selects that graph when the request has no image,
so text requests do not preprocess pixels or execute the vision encoder. Both
graphs use the same public API, tokenizer, postprocessing and model identity.

`swev.text-model` contains:

- `specification`: a base64-encoded single-function Core ML model specification,
  without its own tokenizer metadata, at most 8 MiB decoded.
- `metadata`: string values for `swev.config`, `swev.preprocessing`, and
  `swev.signatures`, overriding the shared metadata for the text graph. Its
  execution profile must be `text-decision-v1`; identity, revision and option
  capacity must match the root model.
- `weights`: a mapping from blob references such as
  `@model_path/weights/weight.bin` to compiled-relative paths such as
  `weights/weight.bin`. At most 16 files directly inside `weights` are supported.

Both specifications must reference the same shared blob offsets. Exporters must
remap offsets when deduplicating weights; attaching a specification from an
unrelated export is invalid. The root graph remains a standard image-capable
Core ML model. Runtime routing is supplied by Swev.

During loading, Swift builds a temporary text package using the embedded
specification and hard links to the compiled weight files (copying if linking is
unavailable), compiles it, and validates its signatures. Temporary source files
are removed; compiled views are removed when the model is released. The text graph is loaded initially; the image graph loads on the first image
request and stays cached afterward. This avoids named-function and large-branch
compiler requirements. Once both routes have been used, runtime memory can
exceed a single graph.
No downloads or external model sidecars are required.

Text and image graphs have independent sequence budgets, each bounded
by the root `maxSequenceTokens`. Each graph applies its own recipe context limits; oversized requests are rejected. Exporters should check text → image → text transitions, probability
parity, and latency separately. Image features are not cached between questions.

See the [versioned schema](schema.md), [conversion workflow](conversion.md), and [Hugging Face loader](huggingface.md).
