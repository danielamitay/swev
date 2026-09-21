# Hugging Face and local loading

```swift
import Foundation
import Swev

let model = try await SwevModel.load(
    hf: "mlx-community/SmolVLM-500M-Instruct-bf16"
)
```

The first load downloads the supported model's files into the normal Hugging Face cache. Repeated loads reuse those files. Keep the returned model resident to avoid repeated tokenizer and model initialization.

Use `revision:` with a Hub commit hash for reproducibility. The loader binds weight downloads to the configuration snapshot it inspected. Authentication for gated/private repositories uses the upstream Hugging Face client's configured credentials; do not put access tokens in source code.

For an existing local directory:

```swift
let model = try await SwevModel.load(
    url: URL(fileURLWithPath: "/path/to/model")
)
```

Local loading requires configuration, tokenizer/processor assets, and MLX weights. It performs no model download or conversion. A path to a single weights file is insufficient.

Both overloads accept `maxPendingRequests:` and `maxContextTokens:`. A supplied context bound may reduce a declared limit or provide a missing one, but cannot exceed a declared limit. See [model compatibility](models.md).
