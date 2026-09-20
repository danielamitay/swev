# Swev

Swev is a Swift/Core ML package for local, typed decisions, intended to mirror the API functionality recently popularized by Jev. Give it text or structured state and runtime-defined questions; receive choices, ordinal scores, and probabilities.

```swift
import Swev

let model = try await SwevModel.load(from: modelURL)
let response = try await model.predict(
    state: "apple",
    questions: [.noul(id: "edible", instructions: "Is this an edible food?")]
)
print(try response.noul("edible").noul)
```

Compatible model packages bundle their tokenizer, text formatting recipe, and inference settings. The runtime reads these settings without model-specific adapters. Images are not supported. Model accuracy depends on the checkpoint and task.

Requires Swift 6, macOS 15+ or iOS 18+. Tokenization and inference run locally without Python or third-party runtime dependencies. Model weights are not included.

Run `swift test`. See [model support](docs/models.md) and [text evaluation](docs/evaluation.md).
