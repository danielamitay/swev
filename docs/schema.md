# Model schema 1.0

A Swev asset is a standard Core ML model with UTF-8 strings in its creator-defined metadata. JSON records are serialized as strings; tokenizer payloads are embedded verbatim. No external adapter files are needed. The tensor and recipe details are in [Model contracts](models.md).

## Versioning

`swev.config.contractVersion` is the **schema version**, currently exactly `"1.0"`. It covers all `swev.*` records, including the embedded text graph. The loader checks it before decoding the rest of the configuration. Missing or malformed versions throw `SwevError.invalidMetadata`; any other string, including `"1.1"`, throws `SwevError.unsupportedContractVersion`. There is no implicit forward compatibility or fallback.

`modelVersion` identifies the exported artifact, while optional `revision` identifies its source revision. Neither selects a schema. Execution profiles and input adapters also have explicit names/versions; unknown profiles throw `unsupportedProfile`. A future incompatible contract must get a new contract version and explicit runtime support. Existing packages already use this field and need no migration.

## Configuration

`swev.config` example for a text model:

```json
{
  "contractVersion": "1.0",
  "modelVersion": "0.1.0",
  "revision": "source-commit",
  "id": "example-model",
  "architecture": "causal-language-model",
  "capabilities": {
    "modalities": ["text"],
    "questionTypes": ["choice", "score", "noul"],
    "limits": {
      "maxQuestionsPerRequest": 64,
      "maxOptionsPerQuestion": 4,
      "maxSequenceTokens": 128
    }
  },
  "execution": {
    "profile": "text-decision-v1",
    "inputAdapter": "text-recipe-v1"
  }
}
```

All fields except `revision` are required. Identity/version/profile/adapter strings must be nonempty. Questions are bounded to 1–64, options to 2–32, and sequence length to 8–2048. Capabilities must match preprocessing and actual tensors. All three question types are required. Unknown JSON fields are ignored within this supported schema; they cannot change its required semantics.

## Metadata records

| Key | Contents |
| --- | --- |
| `swev.config` | Configuration above. |
| `swev.preprocessing` | `sequenceLength`, `optionCapacity`, `tensors`, declarative `recipe`, and optional `image`. |
| `swev.postprocessing` | Three positive finite `temperatures` in choice/score/noul order; `temperaturesByOptions` overrides; `choiceConfidence` and `scoreConfidence`. |
| `swev.signatures` | `inputs` and `outputs`, mapping names to `{ "shape": [...], "dtype": "int32" or "float32" }`. Must match the graph exactly. |
| `swev.tokenizer.asset-index` | Asset-name map of `{ "key": metadataKey, "bytes": utf8Length, "sha256": lowercaseHex }`. Must include `tokenizer.json`. |
| `swev.tokenizer.tokenizer.json` | Complete tokenizer JSON string, covered by the asset index. |
| `swev.text-model` | Required only for the routed profile; embedded text graph described below. |
| `swev.provenance` | Optional informational source/conversion record; does not control inference. |

Ordinary JSON records are limited to 1 MiB each. The tokenizer index permits at most 16 assets with at most 32 MiB of verified UTF-8 payload in total. Invalid hashes throw `metadataIntegrityFailure`. Output is `option_logits`, Float32 `[1, K]`. Scores are normalized over the supplied candidates, not the full vocabulary.

## Execution profiles

- `text-decision-v1`: modalities `["text"]`, no image preprocessing or embedded graph.
- `vision-decision-v1`: modalities `["text", "image"]`, image preprocessing and `causal-labels` tensors. One image recipe slot. The graph also accepts text requests with the configured blank canvas.
- `routed-vision-decision-v1`: same root image contract, plus `swev.text-model`. Image-free requests use the embedded text graph and skip vision computation.

All currently use `text-recipe-v1`. Its tensor layouts are `masked-options`, `causal-pointer`, and `causal-labels`. See [the recipe and image definitions](models.md) for their exact signatures, expression operations, formatting, pixel transforms, and context limits. Unsupported requests fail rather than silently changing modalities.

## Embedded text graph

`swev.text-model` is JSON with `specification` (base64 Core ML Model protobuf), `metadata` (string-valued overrides for exactly `swev.config`, `swev.preprocessing`, `swev.signatures`), and `weights` (map from `@model_path/weights/FILENAME` to `weights/FILENAME`). Other metadata, including tokenizer and postprocessing, is inherited from the root.

The record is bounded to 12 MiB, decoded specification to 8 MiB, and weight references to 1–16 safe filenames. The text configuration must use schema 1.0 and `text-decision-v1`; identity, artifact version, revision, question capacity, and option capacity must match the root. Its sequence length may be shorter. Weights are shared within the single package; the runtime compiles a temporary text view and loads the root image model lazily.
