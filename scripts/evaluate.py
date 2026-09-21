#!/usr/bin/env python3
"""Run labeled text cases through a model driver using JSON Lines."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def distribution(answer, keys):
    values = answer.get("probabilities")
    if not isinstance(values, dict) or set(values) != set(keys):
        raise ValueError("Missing or mismatched probability keys")
    result = [values[key] for key in keys]
    if not all(number(p) and 0 <= p <= 1 for p in result):
        raise ValueError("Invalid probabilities")
    if abs(sum(result) - 1) > 1e-6:
        raise ValueError("Probabilities do not sum to one")
    return result


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def check(case, response):
    if not isinstance(response, dict) or not isinstance(response.get("model"), str) or not response["model"]:
        raise ValueError("Response must identify its model")
    questions = case["request"]["questions"]
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("Missing or unexpected answer IDs")
    checks = []
    for qid, question in questions.items():
        answer = answers[qid]
        expected = case["expected"][qid]
        kind = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ValueError("Answer type mismatch")
        if kind == "choice":
            keys = list(question["criteria"])
            p = distribution(answer, keys)
            selected = keys[max(range(len(p)), key=p.__getitem__)]
            if answer.get("choice") != selected:
                raise ValueError("Choice does not match distribution")
            passed = selected == expected["choice"]
            actual = selected
        elif kind == "noul":
            actual = answer.get("noul")
            if not number(actual) or not 0 <= actual <= 1:
                raise ValueError("Noul must be a probability")
            passed = expected["min"] <= actual <= expected["max"]
        elif kind == "score":
            keys = [str(i) for i in range(len(question["criteria"]))]
            p = distribution(answer, keys)
            actual = answer.get("score")
            if not number(actual) or abs(actual - sum(i * value for i, value in enumerate(p))) > 1e-6:
                raise ValueError("Score does not match ordinal expectation")
            passed = expected["min"] <= actual <= expected["max"]
        else:
            raise ValueError("Unsupported question type")
        checks.append({"case": case["id"], "question": qid, "passed": passed, "actual": actual})
    return checks


def load_cases(path):
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not cases or len(cases) > 10000:
        raise ValueError("Expected 1 to 10000 cases")
    seen = set()
    for case in cases:
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in seen:
            raise ValueError("Empty or duplicate case ID")
        seen.add(case["id"])
        request = case["request"]
        questions = request["questions"]
        if not isinstance(request["state"], (str, dict, list)) or not questions or set(questions) != set(case["expected"]):
            raise ValueError("Invalid fixture request or expectations")
        for qid, q in questions.items():
            expected = case["expected"][qid]
            if q["type"] == "choice":
                if not isinstance(q["criteria"], dict) or len(q["criteria"]) < 2 or expected["choice"] not in q["criteria"]:
                    raise ValueError("Invalid choice fixture")
            elif q["type"] in ("noul", "score"):
                upper = 1
                if q["type"] == "score":
                    if not isinstance(q["criteria"], list) or len(q["criteria"]) < 2:
                        raise ValueError("Invalid score fixture")
                    upper = len(q["criteria"]) - 1
                if not all(number(expected[k]) for k in ("min", "max")) or not 0 <= expected["min"] <= expected["max"] <= upper:
                    raise ValueError("Invalid expected range")
            else:
                raise ValueError("Unsupported fixture question type")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face owner/name or local model directory")
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[1] / "fixtures/text-cases.json")
    parser.add_argument("--min-accuracy", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--driver", nargs=argparse.REMAINDER, required=True,
                        help="Command receiving --model LOCATION; reads and writes one JSON object per line")
    args = parser.parse_args()
    try:
        if not args.driver:
            raise ValueError("A driver command is required")
        location = Path(args.model).expanduser()
        if location.is_dir():
            model = str(location.resolve())
        elif not args.model.startswith(("/", ".", "~")) and len(args.model.split("/")) == 2 and all(
            part and part not in (".", "..") for part in args.model.split("/")
        ):
            model = args.model
        else:
            raise ValueError("Model must be a local directory or Hugging Face owner/name")
        if args.max_context_tokens is not None and args.max_context_tokens <= 0:
            raise ValueError("Context limit must be positive")
        if not math.isfinite(args.min_accuracy) or not 0 <= args.min_accuracy <= 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("Invalid accuracy or timeout")
        cases = load_cases(args.cases)
        command = [*args.driver, "--model", model]
        if args.max_context_tokens is not None:
            command += ["--max-context-tokens", str(args.max_context_tokens)]
        result = subprocess.run(command,
                                input="".join(json.dumps(c["request"], ensure_ascii=False) + "\n" for c in cases),
                                text=True, capture_output=True, timeout=args.timeout, check=True)
        lines = result.stdout.splitlines()
        if len(lines) != len(cases):
            raise ValueError("Driver must return exactly one response per request")
        responses = [json.loads(line) for line in lines]
        checks = [check_item for case, response in zip(cases, responses) for check_item in check(case, response)]
        models = {response["model"] for response in responses}
        if len(models) != 1:
            raise ValueError("Driver changed model identity between requests")
        passed = sum(item["passed"] for item in checks)
        report = {"model": models.pop(), "passed": passed, "total": len(checks),
                  "accuracy": passed / len(checks), "checks": checks}
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0 if report["accuracy"] >= args.min_accuracy else 1
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr[-4000:], file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
