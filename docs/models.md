# Model loading and compatibility

Swev loads ordinary supported MLX repositories or local directories. A model supplies its weights, tokenizer, chat template, and processor configuration. No Swev export, embedded metadata, sequence buckets, or separate adapter file is required.

```swift
let remote = try await SwevModel.load(hf: "mlx-community/SmolVLM-500M-Instruct-bf16")
let local = try await SwevModel.load(url: modelDirectory)
```

The loader inspects configuration before downloading weights. A registered `model_type` selects the upstream language or vision-language runtime. The inspected Hugging Face snapshot is pinned for subsequent downloads. Architecture support is necessary but does not guarantee decision accuracy; see the measured [model table](../README.md#models).

## Responsibility boundaries

`mlx-swift-lm` owns architectures, tokenizers, chat templates, KV caches, and image preprocessing. Swev renders a deterministic decision prompt, maps options to distinct single-token labels, reads their logits, and computes typed probabilities. There is no generated text or JSON parsing. Each question has a fresh cache and one logical prefill, which the runtime may split into chunks.

Two explicit checkpoint formats use isolated Swift extensions: a bidirectional decision encoder and signed-Hadamard packed layers. Their declared formats and supported versions are documented in [configuration formats](schema.md). Selection never uses repository names. Ordinary quantization remains upstream-owned.

## Limits and errors

- At most 64 questions per request and 26 options per question.
- At most one PNG/JPEG image, with at most 32 MiB of encoded data, 8,192 pixels per side, and 16,777,216 total pixels. The declared content type must match the bytes. Check `descriptor.capabilities.supportsImages` before attaching one.
- The context limit applies per question and includes chat framing and image tokens. Decision encoders can also impose smaller instruction/option budgets.
- The default admission bound is eight requests, including the active request. Configure `maxPendingRequests:` from 1 through 64. Excess admission throws `queueFull`.

Both loading methods accept `maxContextTokens:`. It may reduce a declared limit or supply an omitted one; it cannot exceed a declared limit. Oversized requests throw `contextOverflow` rather than losing state, instructions, or options. Unknown architectures and unsupported tokenizers fail explicitly.

Inference is serialized per model. Cancellation is checked between stages and prefill chunks; an active device operation must finish before cancellation returns. Returned probabilities preserve precision and are conditional on the offered candidates. Score is the expected zero-based level; noul is the probability of true.
