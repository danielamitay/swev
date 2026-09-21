# Contributing to Swev

Use Swift 6.2 or newer with Xcode on macOS. Real MLX inference requires Apple silicon and compiled Metal shaders. The package targets macOS 15+ and iOS 18+; model performance and memory requirements need validation on the intended device. Python is used only for optional evaluation tooling, not inference.

## Get started

```sh
git clone https://github.com/danielamitay/swev.git
cd swev
git switch mlx
swift test
swift build --package-path Examples/Decisions
python3 -m unittest discover -s scripts -p 'test_evaluate.py'
```

Ordinary tests need no weights or credentials. They cover request validation, ordering, probabilities, decision prompts, admission bounds, and configuration compatibility. Real-model parity tests are opt-in; see [evaluation](docs/evaluation.md). Build runnable inference tools with Xcode as shown in the [MLX guide](docs/mlx.md).

## Repository map

| Path | Purpose |
| --- | --- |
| `Sources/Swev/` | Public model API, capabilities, errors, and admission bounds |
| `Sources/Swev/Decisions/` | Requests, responses, structured JSON, prompts, and scoring |
| `Sources/Swev/MLX/` | Model loading, runtime selection, and isolated compatibility adapters |
| `Sources/MLXDecisionModels/` | Runtime extensions for declared decision-head and packed-weight formats |
| `Sources/SwevCLI/` | Text-only JSON Lines driver |
| `Tests/SwevTests/` | Unit tests and opt-in source parity |
| `Examples/` | Text/image API consumer and sequential benchmark runner |
| `scripts/` | Model-independent evaluation harness and its tests |
| `fixtures/` | Small labeled text requests |
| `docs/` | Loading, configuration, usage, and evaluation guides |
| `.local/` | Ignored checkpoints, environments, and generated experiment reports |

Start with `Request.swift` and `Response.swift`, then `SwevModel.swift`. The upstream MLX runtime owns architectures, tokenizers, chat templates, caches, and image preprocessing. Swev owns the decision layer. Keep exceptional runtime extensions isolated and select them from declared configuration, never repository names.

## Making a change

Keep commits and pull requests focused on completed, useful changes. Explain the resulting behavior and relevant validation. Public API changes need concise documentation; behavior changes need a regression test. Check that examples still build.

Preserve option order, probability precision, bounded requests, cancellation, and explicit errors. Never silently truncate input. Changes to model math or processing require source parity at relevant lengths and modalities, not just a successful build. Do not run competing large inference jobs when measuring latency.

Do not commit weights, caches, access tokens, or generated reports. Report bugs with the toolchain, model ID and revision, minimal request, and exact error, removing private input and credentials. Swev code is MIT-licensed; model releases have their own licenses.
