# Working on Swev

- Swift 6.1 package for local typed decisions; macOS 15+ and iOS 18+. The `mlx` branch is migrating to `mlx-swift-lm`, starting with SmolVLM 500M. Read `docs/mlx.md`; retain the Core ML baseline until comparisons justify removal.
- Read `README.md` and `CONTRIBUTING.md` for usage and the repository map. Read `docs/models.md` and `docs/schema.md` before changing model loading, tensor contracts, or package metadata.
- Keep the public decision API model-independent. MLX owns architectures, tokenizers, chat templates, caches, and image processing. Swev owns candidate prompts and scoring. Keep temporary upstream compatibility fixes isolated and documented.
- Preserve candidate order, probability precision, bounded requests, and explicit errors. Never silently truncate input or accept unknown schema versions. Keep preprocessing and inference off the main actor.
- Document public API behavior concisely, especially units, limits, cancellation, and errors. Keep README examples executable and avoid claiming untested platform or model support.
- Store checkpoints, model exports, credentials, generated reports, and experiments under ignored paths such as `.local/`. Never commit weights or access tokens. Do not download large models for routine checks.
- Make small, atomic commits for completed changes. Add meaningful regression tests for behavior changes; do not add tests that merely restate implementation details.

## Validation

Run the checks relevant to the change:

```sh
swift test # Unit tests; use Xcode for real MLX inference and its Metal shaders.
swift build --package-path Examples/Decisions
python3 -m unittest discover -s scripts -p 'test_evaluate.py'
```

Real-model tests are opt-in. Conversion tests need the separate dependencies in `scripts/conversion-requirements.txt`; see `docs/conversion.md` and `docs/evaluation.md`. Changes to conversion or inference require source parity at affected shapes and modalities, not just successful compilation. Do not run competing large conversions or benchmarks.
