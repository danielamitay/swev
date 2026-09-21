# Swift API

Start with the runnable [README examples](../README.md#quickstart). Load a model once with `SwevModel.load(hf:)` or `SwevModel.load(url:)`, then call `predict` repeatedly.

## Structured state and metadata

State and instructions accept strings or ordered `JSONValue` objects/arrays. Application option IDs are returned unchanged. Metadata is echoed without entering the prompt.

```swift
let response = try await model.predict(
    state: .object([
        ("message", "My package arrived damaged."),
        ("orderValue", .number(85))
    ]),
    questions: [
        .choice(id: "action", instructions: "Choose the next support action.", options: [
            .init(id: "replace", description: "Offer a replacement"),
            .init(id: "investigate", description: "Ask for more information")
        ]),
        .score(id: "urgency", instructions: "Rate urgency.", levels: ["low", "medium", "high"]),
        .noul(id: "refund", instructions: "Does the customer explicitly request a refund?")
    ]
)
print(try response.choice("action").choice)
print(try response.score("urgency").score) // Expected level, from 0 through 2
print(try response.noul("refund").noul)   // Probability of true
```

Choice probabilities are keyed by your option IDs. Score probabilities follow the supplied level order; score is their expected zero-based index, not a rounded class. Noul uses false/true candidates and returns the true probability. Confidence statistics identify their method and do not guarantee correctness.

## Images, limits, and cancellation

Attach owned PNG/JPEG bytes through `images:`. Check `model.descriptor.capabilities.supportsImages` first. A text-only request skips vision processing. Multiple images, audio, and video are unsupported.

There are at most 64 questions per request, 26 options per question, and one image of at most 32 MiB. Context includes formatting and image tokens and applies independently to each question. Overlong requests fail explicitly rather than losing input. See [model limits](models.md#limits-and-errors).

Requests are serialized per model. `maxPendingRequests:` bounds admission, including the active request. Cancellation is cooperative between stages and prefill chunks; it cannot interrupt an active device operation. Keep inference work outside UI rendering callbacks and handle errors from the throwing API.
