# Evaluating a model

Build the MLX driver with Xcode so its Metal shaders are available:

```sh
xcodebuild -scheme swev -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .local/mlx-build \
  -skipMacroValidation -skipPackagePluginValidation build

python3 scripts/evaluate.py \
  --model mlx-community/SmolVLM-500M-Instruct-bf16 \
  --driver .local/mlx-build/Build/Products/Release/swev
```

The driver loads once, reads JSON Lines requests from stdin, and emits one response per request. Use a local model directory in place of the Hub ID to avoid downloads. Logs go to stderr. The Swift API and benchmark example also accept images; the stdin CLI is text-only.

The default fixtures cover food recognition, classification, and sentiment. They are smoke tests, not food-safety advice or a comprehensive quality benchmark. Pass `--cases PATH` for a JSON array of fixtures in the format used by `fixtures/text-cases.json`.

Each fixture contains `id`, `request`, and `expected`. Expectations use `choice` for a selected option, or inclusive `min`/`max` bounds for noul and score. The harness checks response types, finite normalized probabilities, option order, and consistency between a score/choice and its distribution before checking correctness.

All cases must pass by default. Use `--min-accuracy 0.9` to choose another threshold. Exit codes are 0 for meeting the threshold, 1 for incorrect answers, and 2 for invalid input, driver failure, malformed output, or timeout. `--timeout 300` bounds the complete run. Pass `--max-context-tokens N` before `--driver` when a model needs an explicit context limit. Put the driver command last.

Run harness unit tests without model downloads:

```sh
python3 -m unittest discover -s scripts -p 'test_evaluate.py'
```

For full suites, `Examples/Benchmark` reads JSON Lines cases containing `id`, `request`, and an optional local `image` path. It records loading separately and excludes per-request warmup from inference latency. See [benchmark instructions](mlx.md#reproduce-a-benchmark) and [measurement guidance](performance.md).

Changes to model math or preprocessing need source-runtime parity at relevant lengths and modalities, as well as labeled checks. `RuntimeParityTests` is opt-in through `SWEV_TEST_MLX`, `SWEV_ENCODER_MODEL`, and `SWEV_ENCODER_PARITY`; use an existing checkpoint and a JSON array of source fixtures containing `state`, `questions`, `tokens`, `markers`, `type` (0/1/2), and `logits`. The test compares the Swift token layout and raw logits against these references. Ordinary unit tests require no weights or credentials. Keep generated fixtures and reports outside Git.
