# MLX runtime

Swev uses `mlx-swift-lm` to load ordinary MLX models and return typed decisions. See the [model table](../README.md#models) for measured compatibility and accuracy.

## Load once, predict repeatedly

Use an Apple silicon Mac. Build with Xcode so MLX’s Metal shaders are compiled and bundled. The package currently requires Swift 6.3 or newer.

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

Swev formats one deterministic prompt per question, then continues the runtime's assistant prefix with ` Answer: `. For tokenizers with channel-based chat control tokens, it first completes an open assistant header with the final channel. Recipient-based headers are completed from a rendered assistant message in the checkpoint’s own template. Unexpected headers fail explicitly. Selection uses tokenizer and template capabilities, not model names. Each question gets a fresh KV cache and one logical prefill. Swev reads only the candidate logits and normalizes them; it never generates an answer or parses generated JSON. The tokenizer must encode every answer label as one distinct token.

The current decision layer allows 26 options, 64 questions per request, and one PNG/JPEG image of at most 32 MiB, 8,192 pixels per side, and 16,777,216 total pixels. Image metadata and content type are checked before runtime processing. Requests are serialized; the default admission bound is eight pending requests, configurable with `maxPendingRequests:`. Excess requests fail with `queueFull`. Cancellation is checked between processing stages; it cannot interrupt a device operation already running.

The context bound comes from the model's `max_position_embeddings` (8,192 for this SmolVLM), preferring the nested text configuration. Both loading overloads accept `maxContextTokens:` to use a smaller bound. When the model omits its limit, supply this argument using the architecture's documented supported context. An override cannot exceed an explicitly declared limit. Invalid declarations and overlong input are rejected, never silently corrected or truncated. There are no exported sequence buckets on the MLX path.

```swift
let model = try await SwevModel.load(url: modelDirectory, maxContextTokens: 4096)
```

Swev inspects `config.json` before downloading weights and selects the VLM or language runtime from its registered `model_type`. It does not select a backend from a repository name or reinterpret an unknown architecture as a similar one. Quantization remains the runtime's responsibility, including per-layer settings. Registered architecture support is a prerequisite, not a guarantee of decision accuracy.

## Runtime compatibility

The upstream processor registry routes some checkpoints' declared `Idefics3Processor` to a processor that omits chat framing and uses the wrong image dimensions. The isolated `ProcessorCompatibility` shim selects **MLX's existing SmolVLM processor** when the declared processor settings include `max_image_size`. It merges split processor files, preserves the declared `image_seq_len`, and rejects conflicting counts. The registry is scoped to each load; no checkpoint files or global runtime registrations are modified. Image-only assets receive the runtime's required video defaults, but video requests remain unsupported.

A separate tokenizer compatibility loader preserves the upstream template-file precedence and converts complete `role.capitalize()` expressions to the equivalent `capitalize` filter, which the pinned Swift template engine supports. It leaves the checkpoint unchanged and delegates rendering and tokenization to the runtime. Source-engine comparisons confirm identical text and image prompt rendering for the affected template.

Swev does not implement resizing, normalization, tiling, or vision encoding. These compatibility bridges remain isolated so they can be removed when upstream support covers the affected configurations. The README reports measured checkpoints rather than claiming every model in a family works.

Two declared checkpoint formats have isolated Swift runtime extensions in `MLXDecisionModels`:

- `laya-mlx` format version 1 uses a bidirectional ModernBERT encoder and trained decision heads. Its own tokenizer, marker layout, option-count calibration, and declared context/head limits are preserved. It accepts text only and rejects inputs that would require truncation.
- `prism_hadamard_qwen35` schema version 2 uses the upstream Qwen 3.5 language/vision model with manifest-selected signed Hadamard transforms around packed 2-bit layers. The upstream runtime still owns the architecture and image processor.

Selection uses declared architecture/format identifiers, never repository names. Unknown formats and versions fail explicitly; no downloaded Python code is executed. Ordinary quantization remains upstream-owned.

The runtime is pinned to `mlx-swift-lm` commit `c6446cf7bfb7cea76408013b614d4b2c530eaa03` and `mlx-swift` 0.31.6. The language-model runtime adds Muse support; the pinned `mlx-swift` dependency requires Swift 6.3 (Xcode 26.4 or newer). The package disables optional runtime traits.
The stock Transformers checkpoint is not interchangeable with the MLX repository: this runtime expects the MLX convolution weight layout. Use an ordinary supported MLX repository, not a Swev export. No special model metadata or modified checkpoint is needed.

## Build the command-line interface

```sh
xcodebuild -scheme swev -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .local/mlx-build -skipMacroValidation -skipPackagePluginValidation build

printf '%s\n' '{"state":"apple","questions":{"edible":{"type":"noul","instructions":"Is this edible food?"}}}' | \
  .local/mlx-build/Build/Products/Release/swev \
  --model mlx-community/SmolVLM-500M-Instruct-bf16
```

Both the CLI and benchmark runner accept an optional trailing `--max-context-tokens N`, with the same bounds as the Swift loading API.

If Xcode reports a missing Metal compiler, install it with `xcodebuild -downloadComponent MetalToolchain`. The validation flags enable the pinned upstream macros and build plugins for a command-line build. The upstream CUDA plugin is inactive on macOS. The CLI currently accepts text requests; images use the Swift API.

## Reproduce a benchmark

`Examples/Benchmark` loads one model, then evaluates JSON Lines cases sequentially. Each line contains `id`, a normal Swev `request`, and optionally an `image` file path. It emits a load event, one prediction event per case, and a completion event. Each case receives one excluded warmup before its timed prediction; timing covers `predict`, including tokenization and image processing, and excludes file I/O and response serialization.

```sh
cd Examples/Benchmark
xcodebuild -scheme SwevBenchmark -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath ../../.local/benchmark-build -skipMacroValidation -skipPackagePluginValidation build
../../.local/benchmark-build/Build/Products/Release/SwevBenchmark \
  mlx-community/SmolVLM-500M-Instruct-bf16 /path/to/cases.jsonl
```

The local migration experiment uses the pinned public JevBench dataset, its original option ordering, and upstream scoring functions. Full run receipts and the comparison report live in `.local/full-mlx-bench/`; they are not model assets. Checkpoint revisions and measurement details are documented in [performance](performance.md).
