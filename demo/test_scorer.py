"""Hand-picked scorer cases from data/scorer_cases.jsonl.

  python demo/test_scorer.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scorer  # noqa: E402

bad = 0
for line in open(os.path.join(ROOT, "data", "scorer_cases.jsonl"), encoding="utf-8"):
    c = json.loads(line)
    got = scorer.correct(c["reference"], c["answer"])
    bad += got != c["expected"]
    print("ok  " if got == c["expected"] else "FAIL", c["reference"], "|", c["answer"], "|", got, scorer.words(c["reference"]), scorer.words(c["answer"]))
sys.exit(1 if bad else 0)
