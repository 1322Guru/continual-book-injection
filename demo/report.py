"""Score every stage in <out> and write results.json and results.md.

  python demo/report.py <out_dir>
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import scorer  # noqa: E402

EVAL = {b: os.path.join(ROOT, "data", "eval", f"book_{b}.jsonl") for b in "ab"}
FLAGGED = {json.loads(line)["id"] for line in open(os.path.join(ROOT, "data", "eval", "underspecified.jsonl"), encoding="utf-8")}
ORDER = ["t0", "t1", "arm_none_s42", "arm_share_s42", "arm_share_s43", "arm_fixed_s42", "arm_fixed_s43"]


def book_score(ans, book):
    """(correct, of) on the full set and on the cleaned set (flagged questions removed)."""
    res = scorer.score(EVAL[book], ans)
    clean = [r for r in res if r["id"] not in FLAGGED]
    return sum(r["correct"] for r in res), len(res), sum(r["correct"] for r in clean), len(clean)


def main(out):
    res = {}
    for name in ORDER:
        d = os.path.join(out, name, "answers.jsonl")
        if not os.path.exists(d):
            continue
        a, na, ac, nac = book_score(d, "a")
        b, nb, bc, nbc = book_score(d, "b")
        res[name] = {"A": a, "A_of": na, "B": b, "B_of": nb, "A_clean": ac, "A_clean_of": nac, "B_clean": bc, "B_clean_of": nbc}
        meta = os.path.join(out, name, "train", "train_meta.json")
        if os.path.exists(meta):
            m = json.load(open(meta))
            res[name].update({k: m.get(k) for k in ("steps", "train_s", "s_per_step", "peak_vram_gb", "mean_loss")})
    t1 = res.get("t1", {})
    for v in res.values():
        v["ratio"] = round(v["A"] / t1["A"], 3) if t1.get("A") else None
        v["ratio_clean"] = round(v["A_clean"] / t1["A_clean"], 3) if t1.get("A_clean") else None
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    lines = ["| Stage | A full | A ratio | B full | A cleaned | A ratio cleaned | B cleaned | Steps | s/step | Peak VRAM GB (per GPU) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        vram = " / ".join(str(x) for x in (v.get("peak_vram_gb") or {}).values()) or "-"
        lines.append(f"| {k} | {v['A']}/{v['A_of']} | {v['ratio']} | {v['B']}/{v['B_of']} | {v['A_clean']}/{v['A_clean_of']} | {v['ratio_clean']} | "
                     f"{v['B_clean']}/{v['B_clean_of']} | {v.get('steps', '-')} | {v.get('s_per_step', '-')} | {vram} |")
    for arm in ("share", "fixed"):
        for key in ("A", "A_clean"):
            xs = [v[key] for k, v in res.items() if k.startswith(f"arm_{arm}_")]
            if len(xs) > 1:
                lines.append(f"\n{arm} {key}: {xs}, spread {max(xs) - min(xs)}, mean {statistics.mean(xs):.1f}")
    open(os.path.join(out, "results.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
