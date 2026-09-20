# Model contract

`SwevModel.load(from:)` accepts a local Core ML package, model, or compiled model
with the Swev text contract. The public API is independent of model identity.
Model packages supply tokenizer data, formatting recipes, tensor layout, limits,
and scoring configuration. The runtime does not infer these from model weights.

The current execution profile is `text-decision-v1` with input adapter
`text-recipe-v1`. It supports static batch-one text decisions with either
`masked-options`, `causal-pointer`, or `causal-labels` tensors and an `option_logits` output.
Images are rejected. New neural architectures may require a new execution
profile; arbitrary packages are not automatically compatible.

## Embedded metadata

- `swev.config`: identity, versions, capabilities, limits, and execution profile.
- `swev.preprocessing`: `sequenceLength`, `optionCapacity`, `tensors`, and `recipe`.
- `swev.postprocessing`: temperatures, optional option-count temperature buckets,
  and confidence methods for choices and scores.
- `swev.signatures`: tensor names, static shapes, and data types.
- `swev.tokenizer.asset-index`: UTF-8 tokenizer payload keys, byte counts, and SHA-256 hashes.
- `swev.provenance`: optional source identity and conversion information.

The tokenizer reads BPE vocabulary, merges, special tokens, and pre-tokenization
settings from the embedded tokenizer document. It supports NFC byte-level BPE
with standard ByteLevel splitting or an isolated regex Split followed by
ByteLevel without regex. It also supports Unicode BPE with literal space-marker
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
pending requests. Cancellation is checked between stages; an active device call
must finish before cancellation returns. Temporary compiled assets live with the
model instance. CPU-only execution is validated locally; other compute units and
iOS devices need their own validation. Tokenizer hashes detect corruption, not
whether a model source is trustworthy.
