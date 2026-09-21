# Configuration formats

Ordinary MLX models use their upstream `config.json`, tokenizer, and processor formats. Swev does not add a model metadata schema. The loader checks `model_type` against registered implementations and reads a positive integer `max_position_embeddings`, preferring the nested text configuration. A missing context declaration requires an explicit `maxContextTokens:` argument.

Unknown architectures are rejected before weight download. An ordinary model's quantization configuration is interpreted by the pinned MLX runtime. Swev never infers a different architecture from a filename or runs downloaded Python code.

## Decision-encoder format

A directory with `mlx_config.json` declaring `format: "laya-mlx"` and `format_version: 1` uses the decision-encoder backend. Every other format/version fails explicitly. Required assets are:

- `encoder/config.json`: a supported ModernBERT configuration.
- `rl_agent_config.json`: positive `max_len`, `head_max_len`, and `head_layers`; the head budget must be smaller than the context budget.
- `model.safetensors`: encoder, type embeddings, decision Transformer, and marker-scoring weights.
- `tokenizer/`: the checkpoint's tokenizer assets, including CLS, SEP, and mask tokens.

Optional `temperature` contains three positive finite values in choice/score/noul order. `temperature_by_options` may override them with source-defined option-count buckets. Tensor shapes are validated before inference. The trained marker layout and calibration are preserved; inputs exceeding the source format's truncation boundaries are rejected instead of truncated. Images are unsupported.

## Rotated packed-weight format

`model_type: "prism_hadamard_qwen35"` requires `schema_version: 2`, `base_model_type: "qwen3_5"`, the `mlx-vlm-qwen3_5` tensor namespace, grouped GDN activations, and declared vision support. Its quantization is 2-bit affine with group size 128.

The `modules` manifest identifies each packed layer by path, embedding flag, FP16 dtype, and transform block size. Supported blocks are 0, 512, 1024, 2048, and 4096. Duplicate or missing module paths, invalid shapes, and sign vectors containing values other than −1/+1 are rejected. Unknown schemas fail explicitly.

The extension supplies packed linear/embedding operations and their signed Hadamard transforms. The upstream Qwen 3.5 runtime continues to own the model, caches, vision encoder, and image processor. Registries are scoped to each load; checkpoint files and global registrations are not modified.
