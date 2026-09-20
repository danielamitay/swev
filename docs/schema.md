# Model schemas

A Swev asset is a standard Core ML model with UTF-8 strings in its creator-defined metadata. JSON records are serialized as strings; tokenizer payloads are embedded verbatim. No external adapter files are needed. The tensor and recipe details are in [Model contracts](models.md).

## Versioning

`swev.config.contractVersion` is the **schema version**, currently `"1.0"` and `"2.0"`. It covers all `swev.*` records, including the embedded text graph. The loader checks it before decoding the rest of the configuration. Missing or malformed versions throw `SwevError.invalidMetadata`; any unsupported string, including `"1.1"`, throws `SwevError.unsupportedContractVersion`. There is no implicit forward compatibility or fallback.

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

All fields except `revision` are required. Identity/version/profile/adapter strings must be nonempty. Questions are bounded to 1–64 and options to 2–32. Sequence length is 8–2048 in schema 1.0 and 8–4096 in schema 2.0. Capabilities must match preprocessing and actual tensors. All three question types are required. Unknown JSON fields are ignored within this supported schema; they cannot change its required semantics.

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

The record is bounded to 12 MiB, decoded specification to 8 MiB, and weight references to 1–16 safe filenames. The text configuration must use the same schema as its root and `text-decision-v1`; identity, artifact version, revision, question capacity, and option capacity must match the root. Its sequence length may be shorter. Weights are shared within the single package; the runtime compiles a temporary text view and loads the root image model lazily.

## Schema 2.0: sequence buckets

The runtime also accepts `"2.0"`. Schema 1.0 packages remain supported; every other version is rejected. Schema 2.0 raises the maximum sequence capacity to 4096 and adds optional `sequenceBuckets` to preprocessing. For example:

```json
{"sequenceLength":4096,"sequenceBuckets":[128,256,512,1024,2048,4096],"optionCapacity":16}
```

The full preprocessing record still requires `tensors` and `recipe`. Buckets must be unique, increasing integers, contain at most 16 entries, start at 8 or greater, and end at `sequenceLength`. Without this field, the graph has one fixed shape. The runtime encodes without truncation, selects the smallest bucket that fits the complete prompt, and pads to that length. The smallest bucket is the Core ML default shape.

Each sequence-dependent input signature includes `enumeratedShapes`, in bucket order, as well as its default `shape` and `dtype`. For token/position/mask vectors these are `[1,L]`; for causal attention they are `[1,1,L,L]`. All such inputs must declare matching Core ML enumerated shapes. Candidate arrays and logits remain fixed `[1,K]`. Export from a source graph with dynamic dimensions; changing a fixed model's declared input shapes does not make its computations dynamic. Verify source parity at every bucket, including prompts crossing any sliding attention boundary.

For a routed package, `maxSequenceTokens` is the larger route capacity; each route's preprocessing describes its own actual capacity. Thus a text graph can support 4096 tokens while the image graph retains a smaller independent limit. A prompt exceeding its selected modality's capacity throws `contextOverflow`. Root and embedded graph schema versions must match. These semantics require schema 2.0 so older runtimes reject the package instead of misinterpreting its limits.

The runtime keeps the smallest shape resident and caches at most one additional sequence size. Each model instance is used at a single shape to avoid invalid intermediate-buffer reuse during Core ML resizing. The first use of a larger size (or returning to an evicted size) includes its model loading and initialization. For steady-state benchmarks, group requests by sequence bucket and warm that bucket before starting inference timing; report this separately from cold or mixed-size performance.
