"""The fixed rule for underspecified eval questions. Set before any full-run score was seen.

  python demo/flag_eval.py

A question is flagged when it has a deictic word or pronoun from data/scorer_lexicon.json (deictic_words) and no named
subject. A named subject means a capitalized word after the first word of the question. The list is written to
data/eval/underspecified.jsonl. Scores are reported on the full set and on the cleaned set (flagged questions removed).
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEX = json.load(open(os.path.join(ROOT, "data", "scorer_lexicon.json"), encoding="utf-8"))
DEICTIC = set(LEX["deictic_words"])


def flag(question):
    raw = re.findall(r"[^\W_]+", question)
    hit = sorted({w.lower().replace("\u0451", "\u0435") for w in raw} & DEICTIC)
    named = any(w[0].isupper() for w in raw[1:])
    return hit if hit and not named else []


def main():
    out, n = [], {}
    for book in "ab":
        items = [json.loads(line) for line in open(os.path.join(ROOT, "data", "eval", f"book_{book}.jsonl"), encoding="utf-8")]
        got = [{"id": r["id"], "matched": flag(r["question"]), "question": r["question"]} for r in items if flag(r["question"])]
        out += got
        n[book] = {"flagged": len(got), "of": len(items)}
    with open(os.path.join(ROOT, "data", "eval", "underspecified.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(n))


if __name__ == "__main__":
    main()
