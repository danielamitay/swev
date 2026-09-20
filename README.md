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

Compatible model packages bundle their tokenizer, text formatting recipe, and inference settings. The runtime reads these settings without model-specific adapters. Image-capable packages also accept a single PNG or JPEG. Packages with a bundled text graph skip vision computation for text-only requests. Model accuracy depends on the checkpoint and task.

Requires Swift 6, macOS 15+ or iOS 18+. Tokenization and inference run locally without Python or third-party runtime dependencies. Model weights are not included.

Run `swift test`. See [model support](docs/models.md) and [text evaluation](docs/evaluation.md).

Load and cache a hosted package with `SwevModel.load(from: HuggingFaceModel(repository: "owner/repo", package: "model.mlpackage"))`. See [Hugging Face loading](docs/huggingface.md) and the [versioned schema](docs/schema.md).
