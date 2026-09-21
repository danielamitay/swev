> **MLX branch:** SmolVLM 500M now loads directly from an ordinary MLX repository with `SwevModel.load(hf: "mlx-community/SmolVLM-500M-Instruct-bf16")`. See the [MLX quickstart and prototype limits](docs/mlx.md). Core ML loading remains temporarily available for migration comparisons.

![Swev — typed decisions, locally in Swift with Core ML. State and questions become choices, scores, and probabilities.](docs/assets/swev-header.png)

**Jev-style typed decisions, locally in Swift with MLX.**

Swev is a Swift package for running typed decision models locally with MLX—text or images in, choices, scores, and probabilities out.

Define the question and possible answers in your app. Swev scores those answers and returns typed Swift values, without generating text or asking a model to format JSON. Use it to route requests, classify content, rank urgency, or ask questions about an image.

- **Local inference.** Inputs stay on-device. No inference API or Python runtime. Model execution uses `mlx-swift-lm`.
- **Questions defined at runtime.** Choose among your own options, score an ordered scale, or request a probability of true.
- **Ordinary MLX models.** Load supported Hugging Face repositories or local model directories. No Swev export is required; text-only requests skip vision encoding.

**Swift 6.1 · macOS 15+ · iOS 18+ · MIT license**

[Quickstart](#quickstart) · [Models](#models) · [Image input](#image-input) · [Documentation](#documentation)

## Quickstart

Add `https://github.com/danielamitay/swev` in Xcode’s **Add Package Dependencies**, or use Swift Package Manager:

```swift
.package(url: "https://github.com/danielamitay/swev.git", branch: "mlx")
```

Add `.product(name: "Swev", package: "swev")` to your target’s dependencies.

Load a compatible model from Hugging Face and ask a question from an `async` throwing function:

```swift
import Swev

let model = try await SwevModel.load(
    hf: "mlx-community/SmolVLM-500M-Instruct-bf16"
)

let response = try await model.predict(
    state: "My order arrived with the wrong item. Can you help?",
    questions: [
        .choice(
            id: "route",
            instructions: "Which team should handle this message?",
            options: [
                .init(id: "support", description: "Help with existing orders"),
                .init(id: "sales", description: "Questions about buying a product"),
                .init(id: "other", description: "Anything else")
            ]
        )
    ]
)

let answer = try response.choice("route")
print(answer.choice)        // Selected option ID
print(answer.probabilities) // Probability for every option
```

The first load downloads the model; subsequent loads reuse the Hugging Face cache. Keep the model instance alive for repeated predictions. To load a local model directory without downloading:

```swift
import Foundation

let model = try await SwevModel.load(
    url: URL(fileURLWithPath: "/path/to/model-directory")
)
```

Build with Xcode so MLX's Metal shaders are compiled. See the [MLX setup and command-line quickstart](docs/mlx.md) for a runnable example.

## Three question types

| Question | Use it for | Result |
| --- | --- | --- |
| `choice` | Routing or classification with your own labels | Selected option and a probability for every candidate |
| `score` | An ordered scale such as low / medium / high | Expected zero-based level and its probability distribution |
| `noul` | A yes/no question | Probability of true, from 0 to 1 |

Ask several questions about the same state in one request:

```swift
let response = try await model.predict(
    state: "Checkout is failing for every customer. Please investigate immediately.",
    questions: [
        .score(id: "urgency", instructions: "How urgent is this issue?",
               levels: ["low", "medium", "high"]),
        .noul(id: "actionable", instructions: "Does this message request action?")
    ]
)

print(try response.score("urgency").score)     // Expected level between 0 and 2
print(try response.noul("actionable").noul)   // Probability of true
```

State and instructions accept strings or structured `JSONValue` data. Questions are evaluated independently. Probabilities describe the model’s predictions; a confidence statistic is not a guarantee of correctness.

## Models

These repositories contain MLX weights rather than Swev-specific exports. **SmolVLM 500M is currently validated with Swev.** The other entries are listed with their current compatibility status; an MLX checkpoint alone does not guarantee support in `mlx-swift-lm`.

| Model | Size (disk) | Precision | Latency (mean) | JevBench accuracy | Vision? | Swev status |
| --- | ---: | --- | ---: | ---: | :---: | --- |
| [Laya · 421M](https://huggingface.co/aac6fef/laya-mlx) | 0.85 GB | FP16 | — | — | No | Requires a dedicated backend |
| [SmolVLM 256M Instruct](https://huggingface.co/mlx-community/SmolVLM-256M-Instruct-bf16) | 0.52 GB | BF16 | — | — | Yes | Not yet validated |
| [SmolVLM 500M Instruct](https://huggingface.co/mlx-community/SmolVLM-500M-Instruct-bf16) | 1.02 GB | BF16 | 63 ms | 45.0% (104/231) | Yes | Validated |
| [SmolVLM 2.2B Instruct](https://huggingface.co/mlx-community/SmolVLM-Instruct-bf16) | 4.50 GB | BF16 | — | — | Yes | Not yet validated |

**Measurement notes:** latency is the mean over all 231 public JevBench text requests on an Apple M4 Max with 128 GiB memory, using Swift/MLX on the GPU. Requests run sequentially, with loading and per-request warmup excluded; tokenization and inference are included. Accuracy uses the [pinned public JevBench suite](https://github.com/fstandhartinger/jevbench/tree/275763201a29d6083d4ee1431d709c296ef81281), not the official full leaderboard or a vision benchmark. SmolVLM 500M returned valid answers for all 231 cases; its separate text and image checks passed 17/17 and 13/15 respectively.

Sizes are approximate decimal GB for the repository files reported by Hugging Face, not runtime memory requirements. Precision describes stored weights. A dash means no Swev measurement is available; results from other runtimes or model versions are not substituted. “Vision” describes the model's intended modality, not a validation claim for untested entries.

[Laya's model card](https://huggingface.co/aac6fef/laya-mlx) describes a bidirectional ModernBERT decision encoder with custom scoring heads and a dedicated Python MLX runtime. It is not a generative language model and cannot currently be loaded through Swev's `mlx-swift-lm` backend.

SmolVLM 500M supports an 8,192-token context, including prompt formatting and image tokens. Swev currently allows 26 answer options per question, 64 questions per request, and one image. Oversized input is rejected rather than truncated. See [MLX runtime compatibility and limits](docs/mlx.md) for the required processor compatibility shim and the current validation scope.

## Image input

The validated SmolVLM 500M model accepts text alone or text plus one PNG/JPEG. With the same `model` from the quickstart:

```swift
import Foundation

let imageURL = URL(fileURLWithPath: "/path/to/photo.png")
let imageData = try Data(contentsOf: imageURL)
let response = try await model.predict(
    state: "Look at the attached image.",
    questions: [
        .choice(id: "scene", instructions: "Where was this photo taken?",
                options: [.init(id: "indoors"), .init(id: "outdoors")])
    ],
    images: [.init(data: imageData, contentType: "image/png")]
)

print(try response.choice("scene").choice)
```

For JPEG bytes, use `image/jpeg`. You can check `model.descriptor.capabilities.supportsImages` before attaching an image. MLX handles model-specific resizing, normalization, and image tiling. Text-only requests skip the vision encoder; audio, video, and multiple images are not supported.

## Documentation

- [MLX usage, loading, and limits](docs/mlx.md) — Hub/local loading, images, runtime compatibility, and runnable commands.
- [Benchmark runner](Examples/Benchmark) — sequential evaluation with loading and warmup excluded from inference timing.
- [Contributing](CONTRIBUTING.md) — development setup, repository map, tests, and pull requests.

The Core ML conversion, packaging, and evaluation documents remain in `docs/` for migration comparisons; they do not describe MLX model requirements.

Run `swift test` for the package tests. Model weights and generated reports stay outside Git. Swev is [MIT-licensed](LICENSE); model licenses are listed separately in their releases.
