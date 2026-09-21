# Text evaluation

Run the labeled smoke tests against a local model with a driver:

```sh
python3 scripts/evaluate.py --model /path/to/model.mlpackage \
  --driver /path/to/model-driver
```

A driver receives `--model PATH`, loads that model once, then reads JSON Lines
from stdin and writes one response per request to stdout, in order. Send logs
to stderr. The command can include arguments, for example
`--driver python3 /path/to/driver.py`. No network or model downloads are performed
by the harness. The driver is responsible for its own inference implementation.
The package includes a native Swift driver; see below.

Requests contain `state` and a `questions` object. Responses identify `model`
and contain an `answers` object with matching question IDs. Each answer has
`type` and the corresponding fields:

- `choice`: selected candidate ID and a `probabilities` object keyed by candidate ID.
- `noul`: a `noul` probability between zero and one.
- `score`: expected zero-based `score` and `probabilities` keyed by level index (`"0"`, `"1"`, …).

Use unrounded probabilities. This is a small evaluation protocol; it does not
establish full compatibility with any hosted API.

The default fixtures test food recognition, classification, and sentiment.
They are smoke tests, not food-safety advice or a model quality benchmark.
Use `--cases PATH` for a JSON array of custom fixtures in the same format as
`fixtures/text-cases.json`. Candidate order is retained. Each question needs an
expectation: `choice` for classification, or inclusive `min`/`max` for a probability
or ordinal score.

The runner checks response types, finite probabilities, normalization, selection,
and score consistency before checking labels. It prints a JSON report and exits
with `0` when the accuracy target is met, `1` for incorrect answers, or `2` for
invalid inputs, driver failures, malformed output, or timeouts. All cases must pass
by default; `--min-accuracy 0.9` sets an explicit alternative. `--timeout 300`
limits the complete run in seconds. Put driver arguments last.

Run harness tests with `python3 -m unittest discover -s scripts -p 'test_*.py'`.
Model conversion parity and runtime performance require separate evaluation.
Keep weights, conversion environments, and generated reports under `.local/`
or the ignored `models/` and `reports/` directories.

## Native Swift driver

Build the actual runtime driver and run the same labeled cases:

```sh
swift build -c release
python3 scripts/evaluate.py --model /path/to/prepared.mlpackage \
  --driver .build/release/swev
```

The driver uses CPU-only Core ML and native tokenization. It accepts a text-only
decision API subset, preserves object order, rejects duplicate keys, and checks any
supplied model ID. Score wire requests allow 2–32 levels, further limited by the
asset. Images and remote URLs are not fetched. Hosted API parity, extended wire
metadata, and legacy rounded answers are outside this codec's scope.

## Opt-in model integration tests

Set `SWEV_TEST_MANIFEST` to a local JSON file:

```json
{
  "tokenizers": [{"path": "/path/to/tokenizer.json", "reference": "/path/to/token-tests.json"}],
  "adapters": [{"tokenizer": "/path/to/tokenizer.json", "recipe": "/path/to/recipe.json", "reference": "/path/to/adapter-tests.json"}],
  "models": [{"path": "/path/to/model.mlpackage", "reference": "/path/to/probabilities.json"}]
}
```

Run `SWEV_TEST_MANIFEST=/path/to/manifest.json swift test`. All source-model fixtures
are external to the repository. Tokenizer references contain `text` and `ids`;
adapter references contain `request`, `ids`, and `options`. Model references are an
array containing `probabilities` per row, in the order of the 17 text cases.
Probabilities must already include the source model's calibration.

Ordinary `swift test` skips the three model-dependent tests and uses synthetic
unit fixtures. Full release parity needs a larger, independently frozen suite.

## Opt-in image integration tests

Set `SWEV_IMAGE_TEST_MANIFEST` to a local JSON file containing `model`,
`preprocessing`, `cases`, and `textReference` paths, plus an optional output
`report` path. `preprocessing` is the package's preprocessing JSON. Each image
case contains a request, an image filename, a `pixels` filename, and source
`probabilities`. Image and pixel paths are relative to the cases file. Pixel
references are little-endian Float32 RGB tensors in NHWC order. Text references
contain requests and probabilities for checking requests without an image.

Run `SWEV_IMAGE_TEST_MANIFEST=/path/to/manifest.json swift test --filter image`.
These checks compare preprocessing tensors and native inference to source
references. Semantic accuracy is evaluated separately; source-model mistakes
must not be hidden by changing parity expectations. Default tests generate tiny
synthetic images to check padding, transparency, content types, and EXIF rotation.
