#!/usr/bin/env python3
"""Deterministic scorer: normalized key-fact match, no LLM judge.

  python scorer.py <eval.jsonl> <answers.jsonl> [--out scored.jsonl]

An answer is correct when every content word of the reference answer appears in it. Words match when equal or when
they share a stem (common prefix of at least 4 letters, at most 3 letters left on each side, for Russian endings).
Numbers match exactly: digits, digit groups and number words in any case become one value before matching. A negation word within two words before a matched key word
makes the answer wrong, unless the reference has that negation. Word lists live in data/scorer_lexicon.json.
"""
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
LEX = json.load(open(os.path.join(HERE, "data", "scorer_lexicon.json"), encoding="utf-8"))
STOP, NEG, NUMW, MULT = set(LEX["stopwords"]), set(LEX["negation"]), LEX["number_words"], LEX["number_multipliers"]


def words(s):
    s = unicodedata.normalize("NFKC", s or "").lower().replace("\u0451", "\u0435")
    s = re.sub(r"(?<!\d)(?<!\d )\b\d{1,3}(?: \d{3})+\b", lambda m: m.group(0).replace(" ", ""), s)   # 2 000 -> 2000
    return numbers(re.findall(r"[^\W_]+", s))


def numbers(toks):
    """Number phrases become one digit token: digits, digit groups (2 000), words in any case, and multipliers."""
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if not (t.isdigit() or t in NUMW or t in MULT):
            out.append(t)
            i += 1
            continue
        total, group, last = 0, 0, None
        while i < len(toks):
            t = toks[i]
            if t.isdigit() and last is None:
                group, last = int(t), int(t)
                i += 1
            elif t in NUMW and (last is None or (last >= 20 and last % 10 == 0 and NUMW[t] < 10 ** (len(str(last)) - 1))):
                group, last = group + NUMW[t], NUMW[t]
                i += 1
            elif t in MULT:
                total, group, last = total + max(group, 1) * MULT[t], 0, None
                i += 1
                if i < len(toks) and not (toks[i] in NUMW or toks[i] in MULT):
                    break
            else:
                break
        out.append(str(total + group))
    return out


def same(a, b):
    if a == b:
        return True
    if a.isdigit() or b.isdigit():
        return False
    n = len(os.path.commonprefix([a, b]))
    return n >= 4 and len(a) - n <= 3 and len(b) - n <= 3


def key_words(ref):
    w = [x for x in words(ref) if x not in STOP]
    return w or words(ref)


def correct(ref, answer):
    aw, ref_w = words(answer), set(words(ref))
    for k in key_words(ref):
        hits = [i for i, x in enumerate(aw) if same(k, x)]
        if not hits:
            return False
        if all(any(w in NEG and w not in ref_w for w in aw[max(0, i - 2):i]) for i in hits):
            return False
    return True


def score(eval_path, answers_path):
    items = {}
    for line in open(eval_path, encoding="utf-8"):
        r = json.loads(line)
        items[r["id"]] = r
    out = []
    for line in open(answers_path, encoding="utf-8"):
        a = json.loads(line)
        if a["id"] in items:
            out.append({"id": a["id"], "correct": correct(items[a["id"]]["answer"], a["answer"])})
    return out


def main(argv):
    res = score(argv[0], argv[1])
    if "--out" in argv:
        with open(argv[argv.index("--out") + 1], "w", encoding="utf-8") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")
    print(json.dumps({"correct": sum(r["correct"] for r in res), "of": len(res)}))


if __name__ == "__main__":
    main(sys.argv[1:])
