# MLX prototype

The `mlx` branch is migrating Swev to `mlx-swift-lm`. **SmolVLM 500M BF16 is the first validated model.** Core ML loading and tooling remain temporarily for before/after comparisons; the migration is not complete.

## Load once, predict repeatedly

Use an Apple silicon Mac. Build with Xcode: MLX needs Metal shaders that command-line `swift build` does not compile. The package currently requires Swift 6.1 or newer.

```swift
import Swev

let model = try await SwevModel.load(
    hf: "mlx-community/SmolVLM-500M-Instruct-bf16"
)
let result = try await model.predict(
    state: "apple",
    questions: [.noul(id: "edible", instructions: "Is this edible food for humans?")]
)
print(try result.noul("edible").noul)
```

For a local model directory, use `SwevModel.load(url: URL(fileURLWithPath: "/path/to/model"))`. Pass `revision:` to pin a Hub commit. Downloads use Hugging Face's normal cache; loading a cached model reuses the downloaded weights. Keep the returned instance resident. Local-directory loading needs no model download.

## Choices and images

The same resident model accepts text-only and image requests. A text-only request does not run its vision encoder.

```swift
import Foundation
import Swev

let image = ImageInput(
    data: try Data(contentsOf: URL(fileURLWithPath: "/path/to/photo.png")),
    contentType: "image/png"
)
let result = try await model.predict(
    state: "Examine the attached photograph.",
    questions: [.choice(id: "scene", instructions: "Where was this photographed?", options: [
        .init(id: "indoors", description: "Inside a building"),
        .init(id: "outdoors", description: "Outside a building")
    ])],
    images: [image]
)
print(try result.choice("scene").choice)
```

Structured state, `score`, `noul`, response probabilities, and metadata retain their decision semantics. `score` returns the expected zero-based level index; `noul` returns the true candidate's probability. These probabilities are conditional on the supplied options, not calibrated guarantees of correctness.

## Execution and limits

Swev formats one deterministic prompt per question, then continues the runtime's assistant prefix with ` Answer: `. Each question gets a fresh KV cache and one logical prefill. Swev reads only the candidate logits and normalizes them; it never generates an answer or parses generated JSON. The tokenizer must encode every answer label as one distinct token.

The current decision layer allows 26 options, 64 questions per request, and one PNG/JPEG image of at most 32 MiB. Requests are serialized; the default admission bound is eight pending requests, configurable with `maxPendingRequests:`. Excess requests fail with `queueFull`. Cancellation is checked between processing stages; it cannot interrupt a device operation already running.

The context bound comes from the model's `max_position_embeddings` (8,192 for this SmolVLM). Overlong input is rejected, never truncated. There are no exported sequence buckets on the MLX path.

## Runtime compatibility

`mlx-swift-lm` 3.31.4 routes this checkpoint's declared `Idefics3Processor` to a processor that omits chat framing and uses the wrong image dimensions. The isolated `ProcessorCompatibility` shim selects **MLX's existing SmolVLM processor** for configurations with `max_image_size`, supplying the otherwise-required video defaults for these image-only assets. Swev does not implement resizing, normalization, tiling, or vision encoding. This shim should move upstream before declaring broad model support; only the 500M checkpoint has been validated here.

The stock Transformers checkpoint is not interchangeable with the MLX repository: this runtime expects the MLX convolution weight layout. Use an ordinary supported MLX repository, not a Swev export. No special model metadata or modified checkpoint is needed.

## Build the command-line interface

```sh
xcodebuild -scheme swev -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .local/mlx-build -skipMacroValidation build

printf '%s\n' '{"state":"apple","questions":{"edible":{"type":"noul","instructions":"Is this edible food?"}}}' | \
  .local/mlx-build/Build/Products/Release/swev \
  --model mlx-community/SmolVLM-500M-Instruct-bf16
```

If Xcode reports a missing Metal compiler, install it with `xcodebuild -downloadComponent MetalToolchain`. The macro validation flag enables the pinned upstream Hugging Face integration macros for a command-line build. The CLI currently accepts text requests; images use the Swift API.

## Reproduce a benchmark

`Examples/Benchmark` loads one model, then evaluates JSON Lines cases sequentially. Each line contains `id`, a normal Swev `request`, and optionally an `image` file path. It emits a load event, one prediction event per case, and a completion event. Each case receives one excluded warmup before its timed prediction; timing covers `predict`, including tokenization and image processing, and excludes file I/O and response serialization.

```sh
cd Examples/Benchmark
xcodebuild -scheme SwevBenchmark -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath ../../.local/benchmark-build -skipMacroValidation build
../../.local/benchmark-build/Build/Products/Release/SwevBenchmark \
  mlx-community/SmolVLM-500M-Instruct-bf16 /path/to/cases.jsonl
```

The local migration experiment uses the pinned public JevBench dataset, its original option ordering, and upstream scoring functions. Full run receipts and the comparison report live in `.local/mlx-migration/`; they are not model assets or committed benchmark claims.
