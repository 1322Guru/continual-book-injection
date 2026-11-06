"""Row sets of the demo. Book rows = knowledge + verbatim; everyday rows are each book's own small sample.

  python demo/mix_demo.py t1 <out.jsonl>                              book A rows
  python demo/mix_demo.py arm <none|share|fixed> <seed> <out.jsonl>   book B rows + replay of book A

Replay definitions:
  share  the contest's share rule: replay = round(0.30 / 0.70 * book B rows, everyday excluded), so replay is 30% of
         book rows + replay.
  fixed  a fixed count, 4,700 book-A rows.
Both are stratified over knowledge and verbatim in book A's own proportions and spread one row per unit per round.
Replay never includes everyday rows.
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.mix import alloc, knowledge_per_unit  # noqa: E402

SHARE, FIXED = 0.30, 4700
TYPES = ("knowledge", "verbatim")


def rows(book, kind):
    p = os.path.join(ROOT, "data", "rows", f"book_{book}_{kind}.jsonl")
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def replay_count(arm, n_book_rows):
    return {"none": 0, "share": round(SHARE / (1 - SHARE) * n_book_rows), "fixed": FIXED}[arm]


def book_rows(book):
    return {t: rows(book, t) for t in TYPES}


def replay(arm, seed):
    by, n_b = book_rows("a"), sum(len(v) for v in book_rows("b").values())
    want = alloc({t: len(v) for t, v in by.items()}, replay_count(arm, n_b))
    rng = random.Random(f"replay|{arm}|{seed}")
    out = []
    for t in TYPES:
        out += [{**r, "replay": True} for r in knowledge_per_unit(sorted(by[t], key=lambda r: (r["unit_id"], r["question"])), want[t], rng)]
    return out, want


def build(cmd, arm=None, seed=None):
    if cmd == "t1":
        own = [r for t in TYPES for r in rows("a", t)] + rows("a", "everyday")
        return own, {"own": len(own), "replay": 0, "rows": len(own)}
    own = [r for t in TYPES for r in rows("b", t)] + rows("b", "everyday")
    rp, want = replay(arm, seed)
    return own + rp, {"arm": arm, "seed": seed, "own": len(own), "replay": len(rp), "replay_by_type": want, "rows": len(own) + len(rp)}


def main(a):
    rs, rep = build("t1") if a[0] == "t1" else build("arm", a[1], int(a[2]))
    out = a[-1]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(rep))


if __name__ == "__main__":
    main(sys.argv[1:])
