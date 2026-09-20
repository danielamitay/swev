import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluate import check, load_cases


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = {"id": "food", "request": {"state": "apple", "questions": {
            "food": {"type": "choice", "instructions": "Edible?", "criteria": {"yes": "Food", "no": "Not food"}}
        }}, "expected": {"food": {"choice": "yes"}}}
        self.response = {"model": "test", "answers": {"food": {"type": "choice", "choice": "yes", "probabilities": {"yes": .8, "no": .2}}}}

    def test_success_and_wrong_answer(self):
        self.assertTrue(check(self.case, self.response)[0]["passed"])
        self.response["answers"]["food"].update(choice="no", probabilities={"yes": .2, "no": .8})
        self.assertFalse(check(self.case, self.response)[0]["passed"])

    def test_bad_distributions(self):
        for probabilities in ({"yes": float("nan"), "no": 0}, {"yes": .8, "no": .8}, {"yes": 1}, {"yes": True, "no": False}):
            with self.subTest(probabilities=probabilities):
                self.response["answers"]["food"]["probabilities"] = probabilities
                with self.assertRaises(ValueError): check(self.case, self.response)

    def test_missing_and_mismatched_answers(self):
        for answers in ({}, {"food": {"type": "noul", "noul": .8}}):
            with self.assertRaises(ValueError): check(self.case, {"model": "test", "answers": answers})

    def test_noul_and_score(self):
        for kind, q, expected, answer in [
            ("noul", {}, {"min": .5, "max": 1}, {"noul": .8}),
            ("score", {"criteria": ["low", "high"]}, {"min": .5, "max": 1}, {"score": .8, "probabilities": {"0": .2, "1": .8}}),
        ]:
            case = copy.deepcopy(self.case)
            case["request"]["questions"]["food"] = dict(q, type=kind)
            case["expected"]["food"] = expected
            response = {"model": "test", "answers": {"food": dict(answer, type=kind)}}
            self.assertTrue(check(case, response)[0]["passed"])
            response["answers"]["food"][kind] = float("inf")
            with self.assertRaises(ValueError): check(case, response)

    def test_shipped_cases(self):
        self.assertEqual(len(load_cases(Path(__file__).resolve().parents[1] / "fixtures/text-cases.json")), 17)

    def test_driver_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.json"
            cases.write_text(json.dumps([self.case]))
            driver = root / "driver.py"
            command = [sys.executable, str(Path(__file__).with_name("evaluate.py")), "--model", str(root),
                       "--cases", str(cases), "--timeout", "2", "--driver", sys.executable, str(driver)]
            outputs = [(json.dumps(self.response), 0), ("{}", 2), ("not json", 2)]
            wrong = copy.deepcopy(self.response)
            wrong["answers"]["food"].update(choice="no", probabilities={"yes": .1, "no": .9})
            outputs.append((json.dumps(wrong), 1))
            for output, expected in outputs:
                driver.write_text("import sys\nfor line in sys.stdin:\n    print(" + repr(output) + ")\n")
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)
            driver.write_text("raise SystemExit(3)\n")
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            driver.write_text("import time\ntime.sleep(5)\n")
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)


if __name__ == "__main__":
    unittest.main()
