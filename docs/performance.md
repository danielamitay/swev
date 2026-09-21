# Compute policy

Select the execution devices when loading a model:

```swift
let model = try await SwevModel.load(
    from: modelURL,
    configuration: .init(computeUnits: .cpuAndGPU)
)
```

The README quickstart and runnable example use CPU+GPU. The library default remains `.all`, which allows Core ML to choose devices; it does not guarantee the fastest path. The evaluation CLI uses `.cpuOnly` to preserve its existing baseline.

**Exception: use `.cpuOnly` for the current Gemma FP32 export.** Its first GPU text prediction aborts inside Apple's Metal graph runtime with a shape/stride assertion. This terminates the process rather than throwing a catchable Swift error. Also prefer `.cpuOnly` for Kev 4B: it showed no meaningful short-request speedup, and its 4,096-token GPU check was stopped after severe memory pressure. The runnable example accepts `--cpu-only` before the repository argument.

Retain the loaded model between requests. Download caching avoids downloading weights again, but a fresh model instance still has compilation and initialization costs. GPU acceleration does not imply Neural Engine execution, and a smaller weight file does not guarantee faster inference.

## Paired measurements

Measured September 20–21, 2026 on Apple M4 Max, 128 GiB RAM. Swift release build, public API, serial execution. Each of the 17 text fixtures was warmed and measured three times (51 predictions per model/policy); loading and warmup are excluded. These are short-request timings, not the full JevBench latency or accuracy.

| Model | CPU-only mean | CPU+GPU mean | Speedup | Text correct CPU → GPU |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 E2B IT · FP32 | 846.9 ms | Process abort | — | 17/17 → — |
| Gemma 4 E2B IT · LUT4 | 843.6 ms | 144.2 ms | 5.8× | 17/17 → 17/17 |
| Gemma 4 E2B IT · LUT8 | 352.5 ms | 150.2 ms | 2.3× | 17/17 → 17/17 |
| Kev 0.5B · FP32 | 97.8 ms | 13.2 ms | 7.4× | 12/17 → 12/17 |
| Kev 0.6B · FP32 | 213.6 ms | 16.9 ms | 12.6× | 15/17 → 15/17 |
| Kev 4B · FP32 | 875.8 ms | 863.6 ms | 1.0× | 13/17 → 13/17 |
| Laya EN · FP32 | 75.4 ms | 14.6 ms | 5.2× | 14/17 → 14/17 |
| SmolVLM 500M · FP32 | 121.2 ms | 13.9 ms | 8.7× | 17/17 → 17/17 |

## Image requests

15 image fixtures, one warmed measurement each. Image preprocessing is included; initial image-route loading is excluded.

| Model | CPU-only mean | CPU+GPU mean | Correct CPU → GPU |
| --- | ---: | ---: | ---: |
| Gemma 4 E2B IT · FP32 | 2227.0 ms | Not completed | 14/15 → — |
| Gemma 4 E2B IT · LUT4 | 532.7 ms | 221.8 ms | 12/15 → 13/15 |
| Gemma 4 E2B IT · LUT8 | 530.6 ms | 231.7 ms | 14/15 → 14/15 |
| SmolVLM 500M · FP32 | 218.2 ms | 44.2 ms | 12/15 → 12/15 |

The six models recommended for CPU+GPU completed checks at all six text context sizes (128–4,096), 16-option requests, and image routes where supported.

These checks used serial execution on a working desktop, with normal background activity. They are not a full JevBench rerun or a guarantee for other hardware. Keep the README’s CPU-only JevBench results separate from these shorter requests. A fresh instance or an unseen context bucket can add substantial setup time.

LUT4 changed one image-position answer from incorrect to correct; its maximum image probability difference was 0.320. Tested text probabilities differed by less than 0.000009 across the completed pairs, with no changed choices. Validate representative inputs when changing compute policy, especially image inputs.

Kev 4B exhausted practical memory headroom while traversing all context buckets in one process under both policies; those maximum-context runs were stopped. A fresh CPU-only instance passed the 4,096-token case and returned to short text. A fresh CPU+GPU retry again caused severe memory pressure and was stopped before producing a result. Prefer `.cpuOnly` for Kev 4B; the largest GPU context remains unvalidated. Treat repeated large-bucket transitions as a separate memory concern.

The separate Gemma FP32 GPU image-only attempt could not complete because Core ML compilation ran out of temporary disk space. Its image GPU behavior is unverified; the text-route abort alone rules out recommending CPU+GPU for that package.
