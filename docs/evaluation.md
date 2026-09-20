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
The Swift package does not yet include an executable model driver.

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
