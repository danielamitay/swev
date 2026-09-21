# Runnable examples

The [Decisions example](Decisions/Sources/Decisions/Decisions.swift) is a small macOS command-line program using the public Swift API. It supports all three text question types and a single-image choice question. Its package depends on the repository checkout, so examples build against the code you are editing.

Requires Swift 6 and macOS 15+. From the repository root:

```sh
swift run --package-path Examples/Decisions -c release Decisions \
  danielamitay/gemma-4-e2b-it-lut4-g8-swev \
  gemma-4-e2b-it-lut4-g8-swev-l4096-k16.mlpackage
```

The first run downloads about 2.44 GB. Later runs reuse the downloaded package, but each new process still compiles/loads Core ML. The example uses CPU+GPU execution and prints a JSON response. The model chooses the answers; output is not hard-coded.

Append a PNG/JPEG path to ask about an image instead:

```sh
swift run --package-path Examples/Decisions -c release Decisions \
  danielamitay/gemma-4-e2b-it-lut4-g8-swev \
  gemma-4-e2b-it-lut4-g8-swev-l4096-k16.mlpackage \
  /path/to/photo.jpg
```

For Gemma FP32 or Kev 4B, add `--cpu-only` immediately after `Decisions`. Gemma FP32 crashes on GPU; Kev 4B has no meaningful speedup and excessive memory use at its largest GPU context. See [compute policy](../docs/performance.md) for tested configurations.

Replace the first two arguments with any [compatible model](../README.md#models). Text-only packages cannot accept the image argument. Edit the questions and state in the example source to try your own task.

To check the example without downloading a model:

```sh
swift build --package-path Examples/Decisions
swift run --package-path Examples/Decisions Decisions --help
```

For an already downloaded local package, use the repository's [JSON Lines driver](../docs/evaluation.md#native-swift-driver): `swift run swev --model /path/to/model.mlpackage`. It reads one text request per line from standard input.
