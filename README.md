# Swev

Swev is a Swift package for local, typed decisions with Core ML. Its goal is to turn text or structured state and runtime-defined questions into choices, ordinal scores, and probabilities.

```swift
import Swev

let model = try await SwevModel.load(from: modelURL)
let response = try await model.predict(
    state: "apple",
    questions: [.noul(id: "edible", instructions: "Is this an edible food?")]
)
print(try response.noul("edible").noul)
```

**Status:** API scaffold. Request validation and scoring semantics are tested; model adapters and tokenization are not implemented. `load` currently reports an unsupported profile after reading model metadata. The example describes the intended API, not working inference.

Requires Swift 6, macOS 15+ or iOS 18+. No runtime dependencies beyond Apple frameworks. Model weights are distributed separately and are not included here.

Run `swift test` to build and test the package.
