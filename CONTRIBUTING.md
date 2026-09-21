# Contributing to Swev

Use Swift 6 with Xcode's command-line tools on macOS 15 or newer. The library also targets iOS 18+. Core ML is an Apple framework, so the Swift runtime cannot be built on Linux. Ordinary development needs no model download, Hugging Face account, or Python ML environment.

## Get started

```sh
git clone https://github.com/danielamitay/swev.git
cd swev
swift test
swift build --package-path Examples/Decisions
python3 -m unittest discover -s scripts -p 'test_evaluate.py'
```

The Swift suite covers requests, tokenization, postprocessing, package metadata, images, and Hub caching with local fixtures or mocked transports. Four source/model integration tests are opt-in and skipped unless their manifests are configured. The Python evaluation-harness tests use only the standard library.

Conversion tests require the separate [conversion environment](docs/conversion.md#environment). With those dependencies installed, run `python3 -m unittest discover -s scripts -p 'test_*.py'`. See [evaluation](docs/evaluation.md) for real-model manifests and labeled checks. Public CI runs the checks above without downloading weights or using credentials.

## Repository map

| Path | Purpose |
| --- | --- |
| `Sources/Swev/` | Public request/response types, model loading, native tokenization, and Core ML inference |
| `Sources/SwevCLI/` | Text-only JSON Lines driver for local evaluation |
| `Tests/SwevTests/` | Swift unit tests and opt-in model integration tests |
| `Examples/` | A runnable consumer of the public API |
| `scripts/` | Conversion, packaging, and evaluation tools; Python is not a runtime dependency |
| `fixtures/` | Small, labeled text requests used by the evaluation harness |
| `docs/` | Loading, model contracts, schema versions, conversion, and evaluation guides |
| `.local/` | Ignored space for checkpoints, exports, environments, and experiment reports |

The runtime is small enough to keep in one source directory. Start with `Request.swift` and `Response.swift` for the public types, then `SwevModel.swift` for execution. `ModelAssets.swift` validates the package contract; `TextAdapter.swift` and `BPETokenizer.swift` turn requests into model inputs. `HuggingFaceModel.swift` owns downloading and caching; `BundledTextModel.swift` and `ImagePreprocessing.swift` support image-capable packages.

## Making a change

Keep pull requests focused on one useful change. Explain the problem, resulting behavior, and relevant validation. Public API changes should include concise documentation and an example when needed; behavior changes should include a test that would catch a regression. Check that README examples still build.

Keep model-specific tokenizers, formatting recipes, and calibration inside exported packages, not Swift model-name branches. Schema changes need explicit version handling and updates to [the schema document](docs/schema.md). Unknown versions must fail explicitly. Conversion changes need source parity at supported shapes, not just successful serialization.

Do not commit model weights, compiled assets, access tokens, or generated reports. Use `.local/` for those files. Report a bug with the OS/toolchain, model repository and revision, minimal request, and exact error; remove private input and credentials first.

Swev code is MIT-licensed. Model releases have their own licenses and provenance.
