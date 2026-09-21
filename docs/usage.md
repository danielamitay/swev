# Using Swev in an app

Load a model once, retain the returned `SwevModel`, and reuse it across requests. Loading accepts a local compatible asset or a `HuggingFaceModel`; raw Transformers checkpoints are not loadable. See [loading and caching](huggingface.md) for downloads and [the runnable example](../Examples/README.md) for a complete program.

## Structured state

State and instructions accept strings, objects, or arrays. String literals work directly; wrap an existing Swift `String` as `.string(text)`. Object member order is preserved. Candidate IDs are application-defined; their optional descriptions provide additional context to the model.

```swift
let response = try await model.predict(
    state: .object([
        ("message", "Checkout fails when I submit payment."),
        ("affectedCustomers", .number(12)),
        ("workaroundAvailable", .bool(false))
    ]),
    questions: [.noul(id: "urgent", instructions: "Does this issue need immediate attention?")],
    metadata: .init(sourceID: "ticket-123", schemaRevision: "support-v1")
)
let probability = try response.noul("urgent").noul
```

Request metadata is echoed in the typed response and is not sent to the model. Question IDs must be unique within a request, and choice IDs must be unique within a question. Score levels run from lowest to highest; a three-level scale returns an expected value between 0 and 2, which may be fractional. `noul` returns a probability rather than a thresholded Boolean.

## Capabilities and limits

Inspect `model.descriptor.capabilities` after loading. It declares supported modalities, question types, maximum questions, candidate capacity, and maximum sequence length. Text and image routes can have different budgets; `maxSequenceTokens` is the maximum across routes, not a promise that every route accepts that length. Per-field token limits also apply.

Each question gets its own rendered prompt containing the shared state. Long input is rejected rather than silently truncated. Token usage sums the unpadded prompts across questions, so shared state is counted more than once. Output token usage is zero because predictions score candidates rather than generate text.

## Concurrency and cancellation

A model serializes inference off the main actor. The default admission limit is eight requests including the active one; excess calls throw `SwevError.queueFull`. You can set `RuntimeConfiguration.maxPendingRequests` to a value from 1 through 64. Submitting more concurrent work does not make one model execute questions in parallel.

Cancellation is checked between preprocessing, Core ML prediction, and questions. A device call already in progress finishes before cancellation is observed. A cancelled or failed request returns no partial response.

The default compute policy allows all Core ML devices. The examples select `.cpuOnly`, which is the locally validated configuration. Device compatibility, memory requirements, and latency depend on the package and compute policy.

## Handling errors

Handle errors at the call site; loading and prediction can throw underlying filesystem, network, image, and Core ML errors as well as Swev errors.

| Error | What to check |
| --- | --- |
| `contextOverflow` | Shorten the state, instructions, or option descriptions, or use a larger compatible export. |
| `tooManyOptions(limit:)` | Reduce the candidate count or choose a package with greater capacity. |
| `unsupportedModality` | Use an image-capable package or remove the image. |
| `queueFull` | Reduce concurrency or retry after an admitted request finishes. |
| `unsupportedContractVersion` / `unsupportedProfile` | Match the package to a runtime supporting its declared contract. |
| `metadataIntegrityFailure` / `signatureMismatch` | Check that the complete package was downloaded and that its metadata matches its graph. |
| `HuggingFaceError.cacheMiss` | Prefetch the package before requesting `.localOnly`. |
| `HuggingFaceError.httpStatus` | Check repository access, token permissions, revision, and network response. |

Typed answer accessors throw `missingAnswer` for an unknown question ID or `answerTypeMismatch` for the wrong accessor. `confidence` summarizes a distribution using the declared method; it is distinct from the selected answer's probability and is not a probability of correctness.

## JSON interoperability

`DecisionCodec.decodeRequest` and `encodeResponse` support the [text JSON Lines contract](evaluation.md). The codec retains question and option order and rejects duplicate object keys. It does not accept images or preserve every typed-only response field, such as caller metadata, model revision, or confidence method. Use the Swift API directly for those features.
