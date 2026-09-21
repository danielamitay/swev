# Examples

`Decisions` loads an ordinary MLX model from Hugging Face or a local directory. It asks text questions by default, or a scene question when given an image. Model output is not hard-coded.

Build with Xcode so MLX's Metal shaders are available:

```sh
cd Examples/Decisions
xcodebuild -scheme SwevExamples -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath ../../.local/example-build \
  -skipMacroValidation -skipPackagePluginValidation build

../../.local/example-build/Build/Products/Release/Decisions \
  mlx-community/SmolVLM-500M-Instruct-bf16

../../.local/example-build/Build/Products/Release/Decisions \
  mlx-community/SmolVLM-500M-Instruct-bf16 /path/to/photo.jpg
```

Use a local model directory in place of the Hub ID to avoid downloads. Append `--max-context-tokens N` to supply a missing context declaration or use a smaller bound. Repeated Hub loads reuse the download cache; keep the model resident in your app to avoid repeated initialization. Images must be PNG or JPEG, and the selected model must support vision.

`Benchmark` evaluates JSON Lines cases sequentially and reports loading separately from inference. See [benchmark instructions](../docs/mlx.md#reproduce-a-benchmark).

For stdin JSON Lines requests, use the repository's [evaluation driver](../docs/evaluation.md).
