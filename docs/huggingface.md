# Loading from Hugging Face

Upload the complete Swev-compatible `.mlpackage` directory to a model repository. The Swift loader downloads its files directly from the [Hub API](https://huggingface.co/docs/hub/api); Python and the Hugging Face CLI are not runtime dependencies.

```swift
let source = HuggingFaceModel(
    repository: "your-account/your-model",
    package: "model.mlpackage",
    revision: "main"
)
let model = try await SwevModel.load(from: source)
```

`package` is the exact directory path inside the repository, including any parent directories. A raw Transformers checkpoint is not a compatible package. Use a full commit SHA for reproducible releases. All files in one download resolve to the same commit, even if a branch changes while downloading.

For gated/private repositories, pass `token:` with a read token supplied by your app (for example, from Keychain). Access must already be granted on Hugging Face. Tokens are not persisted by Swev or forwarded to download hosts outside huggingface.co. HTTP authorization errors surface as `HuggingFaceError.httpStatus(401)` or `(403)`.

## Cache behavior

Packages are cached under the app's user caches directory, in `Swev/HuggingFace`. This is separate from the Python/CLI Hugging Face cache. You can provide `cacheDirectory:` to choose another location, including an Application Support directory if the download must survive OS cache eviction.

- `.useCache` (default): reuse a complete cached package without a network request. A cached `main` stays on its previous commit until explicitly refreshed.
- `.refresh`: check the revision on the Hub; reuse its snapshot if unchanged, otherwise download a new one. Network failures are reported, not silently hidden by stale content.
- `.localOnly`: never contact the Hub; throw `HuggingFaceError.cacheMiss` if no complete cached package exists.

```swift
var source = HuggingFaceModel(repository: "your-account/your-model", package: "model.mlpackage")
source.cachePolicy = .refresh
let localURL = try await source.download() // Also useful for prefetching without loading Core ML.
```

Downloads stream to temporary files, check file lengths and supplied LFS SHA-256 hashes, and publish a complete snapshot atomically. Interrupted downloads are discarded and restarted on retry. Cache hits check file presence and length. Concurrent callers are safe but may download the same files before one publishes the snapshot; prefetch once if coordinating many consumers.

The cache stores source packages, **not compiled Core ML models**. Loading still compiles the package and initializes its runtime; keep the returned `SwevModel` resident for repeated inference. Old snapshots remain until you remove the cache directory. Delete it only when downloads are idle. Cache files are read-only inputs to the loader; do not edit them.
