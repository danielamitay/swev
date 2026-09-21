# Measuring decisions

Keep one model loaded and evaluate requests sequentially on an Apple silicon Mac. Do not run competing model inference when comparing latency. Report the model revision, stored precision, hardware, runtime version, and input suite together with the result.

`Examples/Benchmark` measures the public `predict` call, including tokenization, prompt construction, model-specific processing, inference, and decision scoring. It excludes input-file reads and output serialization. Model loading is reported separately. Each request gets one excluded warmup before its timed prediction.

Loading time measures the loading API on an existing cache. Downloads and first-prediction warmup are separate costs. Image requests include vision processing; text-only requests do not run the vision encoder. Larger models and longer inputs can have very different latency from short smoke tests.

Probabilities are specific to the pinned runtime: matching answers across Swift and Python does not imply identical distributions. For GPT-OSS, five short/long reference cases chose the same answers, but the largest candidate-probability difference was 5.7 percentage points with matching MLX core versions. The table reports measured Swift results.

Accuracy must include the complete suite denominator. Report rejected requests separately and count them as incorrect, rather than comparing only the easy inputs that fit. A full run of the pinned public JevBench dataset is 231 cases, not the official full leaderboard. Text scores do not establish image accuracy.

See the [README model table](../README.md#models) for measurements and [MLX benchmark instructions](mlx.md#reproduce-a-benchmark) for the runner. Generated predictions, source-parity fixtures, and local reports belong under `.local/`, outside version control.

## Measured snapshots

Model revisions for the September 21, 2026 evaluation:

| Repository | Revision |
| --- | --- |
| [mlx-community/Muse-Glimmer-30B-4bit](https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit) | [3e7677d7a40d](https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit/tree/3e7677d7a40d348a3daba263a2b1c0aa41910710) |
| [mlx-community/Qwen2-VL-2B-mlx](https://huggingface.co/mlx-community/Qwen2-VL-2B-mlx) | [d8c7c767e2e2](https://huggingface.co/mlx-community/Qwen2-VL-2B-mlx/tree/d8c7c767e2e2c62cda8a51943276458ea6ad43bc) |
| [mlx-community/gemma-3-12b-it-4bit](https://huggingface.co/mlx-community/gemma-3-12b-it-4bit) | [86cc6a8dedbc](https://huggingface.co/mlx-community/gemma-3-12b-it-4bit/tree/86cc6a8dedbc456dd0e4af01a9d09f396f77e558) |
| [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | [10c35caafbb8](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/10c35caafbb80f7dc6a7a432cdd11af10a6d4818) |
| [mlx-community/gemma-4-31b-it-4bit](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit) | [696d436c4047](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit/tree/696d436c404745a59f30e4939a658162b0a9e57f) |
| [mlx-community/gpt-oss-20b-MXFP4-Q8](https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q8) | [773a7da77e56](https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q8/tree/773a7da77e569019bb0fd17a554b263738d669a3) |
| [prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) | [3f926b415992](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/tree/3f926b415992eaa2ae9dd7b573706494d6bbf787) |
| [mlx-community/SmolVLM-Instruct-bf16](https://huggingface.co/mlx-community/SmolVLM-Instruct-bf16) | [cae61cdedd06](https://huggingface.co/mlx-community/SmolVLM-Instruct-bf16/tree/cae61cdedd0602419b43b6102dc33cd9f1e929a6) |
| [mlx-community/SmolVLM-500M-Instruct-bf16](https://huggingface.co/mlx-community/SmolVLM-500M-Instruct-bf16) | [436121330f36](https://huggingface.co/mlx-community/SmolVLM-500M-Instruct-bf16/tree/436121330f361cc3ccde4546fcccc0ec8711bc01) |
| [mlx-community/SmolVLM-256M-Instruct-bf16](https://huggingface.co/mlx-community/SmolVLM-256M-Instruct-bf16) | [cfe179333719](https://huggingface.co/mlx-community/SmolVLM-256M-Instruct-bf16/tree/cfe17933371986666073f450510d9e40d6157b13) |
| [aac6fef/laya-mlx](https://huggingface.co/aac6fef/laya-mlx) | [20aed815fc6a](https://huggingface.co/aac6fef/laya-mlx/tree/20aed815fc6acde75733882e7ec0e3f28aeb9717) |
