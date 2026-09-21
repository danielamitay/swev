# Working on Swev

- Swift 6.3 package for local typed decisions using MLX; macOS 15+ and iOS 18+. Read `README.md`, `CONTRIBUTING.md`, and `docs/mlx.md` before changing runtime behavior.
- Keep the public API model-independent. MLX owns architectures, tokenizers, chat templates, caches, and image processing. Swev owns decision formatting, candidate mapping, probabilities, and scoring.
- Select compatibility paths from declared configuration or tokenizer capabilities, never repository names. Keep runtime extensions isolated. Read `docs/models.md` and `docs/schema.md` before changing loading or format handling; reject unknown versions explicitly.
- Preserve option order, probability precision, bounded requests, cancellation, and explicit errors. Never silently truncate input. Keep inference off the main actor.
- Document public behavior concisely, especially limits, units, and errors. Keep examples runnable and do not claim untested model or platform support.
- Keep weights, credentials, caches, and generated reports under ignored paths. Do not download large models for routine checks. Make small atomic commits for completed changes.

## Validation

```sh
swift test
swift build --package-path Examples/Decisions
python3 -m unittest discover -s scripts -p 'test_evaluate.py'
```

Use Xcode for real inference and its Metal shaders. Model math or processing changes need source parity at affected lengths and modalities; see `docs/evaluation.md`. Run latency benchmarks sequentially without competing inference jobs.
