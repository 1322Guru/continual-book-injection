"""Training row sets of a book.

Stage 1 (six sources): this book's knowledge rows (after the 8-gram filter) + verbatim rows +
everyday rows + diverse trap/twin rows + unanswerable rows + presupposition stage-1 rows (the base trap pairs are only
the input of the diverse rewrite, not a training source); books 2 and 3 add a REPLAY of the earlier books.
Stage 2: this book's presupposition v2 rows + 1,000 everyday + 2,000 knowledge rows sampled per unit (both
per 1,280 units, all from this book) + the stage-2 replay.
Replay by FIXED COUNTS (config recipe.replay): stage 1 takes stage1_book1_rows of book 1 in book 1's
own stage-1 type proportions (everyday excluded) + stage1_other_frac of each other earlier book's own rows per type;
stage 2 takes stage2_book1_rows of book 1's own stage-2 rows (presupposition / everyday / knowledge, same proportions)
+ stage2_other_frac of each other earlier book's own stage-2 rows. Every replay row is RE-SAMPLED, seeded, from the
earlier book's stored rows (no generation calls): knowledge and verbatim one row per unit per round (spread over the book),
trap/twin, unanswerable and presupposition rows as whole pairs, everyday at random.

  python -I pipeline/mix.py replay-plan          replay rows per earlier book and type -> replay/plan.json
  python -I pipeline/mix.py replay-sample        the seeded re-sample -> replay/<book>/sampled.jsonl, sampled_s2.jsonl
  python -I pipeline/mix.py stage1               -> mix1/rows.jsonl + mix1/mix_report.json
  python -I pipeline/mix.py stage2               -> mix2/rows.jsonl + mix2/mix_report.json
"""
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import ctx  # noqa: E402
from pipeline import units as U  # noqa: E402

SEED = 20261015
# this book's own stage-1 sources (folder, file)
OWN = [("knowledge", "knowledge/book_rows.jsonl"), ("verbatim", "verbatim/verbatim_rows.jsonl"), ("everyday", None),
       ("traps_diverse", "traps_diverse/rows.jsonl"),
       ("unanswerable", "unanswerable/rows.jsonl"), ("presup1", "presup1/rows.jsonl")]
REPLAYED = ("knowledge", "verbatim", "traps_diverse", "unanswerable", "presup1")   # stage-1 types replayed (never everyday)
PAIRED = ("traps_diverse", "unanswerable", "presup1", "presup2")                      # sampled as whole pairs / variants
S2_TYPE = {"trap": "presup2", "twin": "presup2", "everyday": "everyday", "book": "knowledge"}   # stage-2 row origin -> type


def scale():
    """The units this run trains over book 1's 1,280: the recipe's everyday and stage-2 counts are per 1,280 units
    (book 1 full = 1.0; a 145-unit book = 0.113; a smoke slice scales the same way)."""
    return len(ctx.load_units()) / 1280


def everyday_rows_for_book(cfg):
    return max(1, round(cfg["recipe"]["everyday_rows"] * scale()))


def everyday_file():
    """Book 1 makes the everyday rows; later books reuse book 1's (same rows, book-independent)."""
    return ctx.path("everyday", "everyday_rows.jsonl", base=ctx.chain()[0])


def own_sources(base=None):
    out = {}
    for name, rel in OWN:
        p = everyday_file() if name == "everyday" else ctx.path(rel, base=base)
        if not os.path.exists(p):
            raise SystemExit(f"missing stage-1 source {name}: {p}")
        out[name] = p
    return out


def plan_path():
    return ctx.path("replay", "plan.json")


def alloc(counts, total):
    """total rows split over the types in the proportions of counts (largest remainder), capped at what exists."""
    tot = sum(counts.values())
    if tot <= 0 or total <= 0:
        return {k: 0 for k in counts}
    raw = {k: total * v / tot for k, v in counts.items()}
    out = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: (out[k] - raw[k], k))[:max(0, round(total) - sum(out.values()))]:
        out[k] += 1
    return {k: min(v, counts[k]) for k, v in out.items()}


def s2_own_rows(base):
    """An earlier book's own stage-2 rows (its replay rows excluded), by type."""
    by = {}
    for r in ctx.jl(ctx.path("mix2", "rows.jsonl", base=base)):
        if not r.get("replay_of"):
            by.setdefault(r.get("s2_src") or S2_TYPE.get(r.get("origin"), "knowledge"), []).append(r)
    return by


def replay_counts(cfg, books=None):
    """{earlier book: {"stage1": {type: rows}, "stage2": {type: rows}}} by config recipe.replay (fixed counts)."""
    RP = cfg["recipe"]["replay"]
    books = books or ctx.chain()
    k = ctx.slice_k()
    out = {}
    for i, b in enumerate(books[:-1]):
        own1 = {t: v for t, v in json.load(open(ctx.path("mix1", "mix_report.json", base=b)))["own_rows"].items() if t in REPLAYED}
        own2 = {t: len(v) for t, v in sorted(s2_own_rows(b).items())}
        if i == 0:
            s1, s2 = alloc(own1, RP["stage1_book1_rows"] / k), alloc(own2, RP["stage2_book1_rows"] / k)
        else:
            s1 = {t: round(v * RP["stage1_other_frac"]) for t, v in own1.items()}
            s2 = {t: round(v * RP["stage2_other_frac"]) for t, v in own2.items()}
        out[b] = {"stage1": s1, "stage2": s2}
    return out


def replay_plan():
    cfg, _ = U.load_config()
    books = ctx.chain()
    idx = len(books)
    os.makedirs(ctx.path("replay"), exist_ok=True)
    plan = {"book_index": idx, "replay": cfg["recipe"]["replay"], "earlier": []}
    for b, c in replay_counts(cfg, books).items():
        plan["earlier"].append({"book": b, "book_id": os.path.basename(b), "stage1": c["stage1"], "stage2": c["stage2"],
                                "stage1_rows": sum(c["stage1"].values()), "stage2_rows": sum(c["stage2"].values())})
    plan["stage1_rows"] = sum(e["stage1_rows"] for e in plan["earlier"])
    plan["stage2_rows"] = sum(e["stage2_rows"] for e in plan["earlier"])
    json.dump(plan, open(plan_path(), "w"), indent=1)
    print(json.dumps(plan))
    return plan


def sample_pairs(rows, n, rng):
    """About n rows, whole pairs/variants kept together (trap + twin rows share pair and variant)."""
    groups = {}
    for r in rows:
        groups.setdefault((r.get("pair") or r.get("unit_ids", [""])[0], r.get("variant"), r.get("origin") == "unanswerable"), []).append(r)
    keys = sorted(groups, key=str)
    rng.shuffle(keys)
    out = []
    for k in keys:
        if len(out) >= n:
            break
        out += groups[k]
    return out


def sample_type(t, rows, n, rng):
    if n <= 0:
        return []
    if t in PAIRED:
        return sample_pairs(rows, n, rng)
    if t in ("knowledge", "verbatim"):
        return knowledge_per_unit(sorted(rows, key=lambda r: (r["unit_id"], r["question"])), n, rng)
    return rng.sample(sorted(rows, key=lambda r: r["question"]), min(n, len(rows)))


def replay_sample():
    plan = json.load(open(plan_path()))
    for e in plan["earlier"]:
        rng = random.Random(f"{SEED}|{plan['book_index']}|{e['book_id']}")
        rows, got = [], {}
        for t, n in e["stage1"].items():
            xs = sample_type(t, ctx.jl(ctx.path(dict(OWN)[t], base=e["book"])), n, rng)
            got[t] = len(xs)
            rows += [{**r, "replay_of": e["book_id"], "replay_src": t} for r in xs]
        ctx.write_jl(ctx.path("replay", e["book_id"], "sampled.jsonl"), rows)
        rows2, got2, own2 = [], {}, s2_own_rows(e["book"])
        for t, n in e["stage2"].items():
            xs = sample_type(t, own2.get(t, []), n, rng)
            got2[t] = len(xs)
            rows2 += [{**r, "replay_of": e["book_id"], "s2_src": t} for r in xs]
        ctx.write_jl(ctx.path("replay", e["book_id"], "sampled_s2.jsonl"), rows2)
        print(json.dumps({"book": e["book_id"], "stage1": got, "stage2": got2}))


def stage1():
    cfg, _ = U.load_config()
    srcs = own_sources()
    rows, own = [], {}
    for name, p in srcs.items():
        xs = ctx.jl(p)
        if name == "everyday":   # the same share of everyday rows as book 1 (5,042 per 1,280 units), seeded
            n = min(len(xs), everyday_rows_for_book(cfg))
            xs = random.Random(f"{SEED}|everyday|{ctx.book_id()}").sample(xs, n) if n < len(xs) else xs
        own[name] = len(xs)
        rows += xs
    replay = {}
    if os.path.exists(plan_path()):
        plan = json.load(open(plan_path()))
        for e in plan["earlier"]:
            got = ctx.jl(ctx.path("replay", e["book_id"], "sampled.jsonl"))
            by = {}
            for r in got:
                by[r["replay_src"]] = by.get(r["replay_src"], 0) + 1
            replay[e["book_id"]] = {"rows": len(got), "by_type": by, "want": e["stage1"]}
            rows += got
    ctx.write_jl(ctx.path("mix1", "rows.jsonl"), rows)
    rep = {"own_rows": own, "own_total": sum(own.values()), "replay": replay,
           "replay_total": sum(v["rows"] for v in replay.values()),
           "rows": len(rows), "units": len(ctx.load_units()), "everyday_file": everyday_file(),
           "note": "row order is shuffled by the packer (seed 42)"}
    book_specific = sum(v for k, v in own.items() if k != "everyday")
    rep["replay_share_actual"] = round(rep["replay_total"] / max(1, len(rows)), 4)
    rep["replay_share_of_book_rows"] = round(rep["replay_total"] / max(1, rep["replay_total"] + book_specific), 4)
    json.dump(rep, open(ctx.path("mix1", "mix_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("own_total", "replay_total", "rows", "replay_share_actual")}))


def knowledge_per_unit(rows, n, rng):
    """Units in seeded order, rows shuffled inside each unit, one row per unit per round."""
    by = {}
    for r in rows:
        by.setdefault(r["unit_id"], []).append(r)
    units = sorted(by)
    rng.shuffle(units)
    for u in units:
        rng.shuffle(by[u])
    picked, k = [], 0
    while len(picked) < n and any(by[u] for u in units):
        u = units[k % len(units)]
        if by[u]:
            picked.append(by[u].pop())
        k += 1
    return picked


def stage2():
    cfg, _ = U.load_config()
    R = cfg["recipe"]["stage2"]
    rng = random.Random(SEED)
    traps = [{**r, "s2_src": "presup2"} for r in ctx.jl(ctx.path("presup2", "rows.jsonl"))]
    every = ctx.jl(everyday_file())
    sc = scale()   # book 1 full = 1.0 (1,000 everyday + 2,000 knowledge); a 145-unit book takes 145/1,280 of both
    n_every, n_know = max(1, round(int(R["everyday"]) * sc)), max(1, round(int(R["knowledge"]) * sc))
    everyday = [{**r, "s2_src": "everyday"} for r in rng.sample(every, min(n_every, len(every)))]
    know = [{**r, "s2_src": "knowledge"} for r in knowledge_per_unit(ctx.jl(ctx.path("knowledge", "book_rows.jsonl")), n_know, rng)]
    replay, by_book = [], {}
    if os.path.exists(plan_path()):
        for e in json.load(open(plan_path()))["earlier"]:
            got = ctx.jl(ctx.path("replay", e["book_id"], "sampled_s2.jsonl"))
            by_book[e["book_id"]] = {t: sum(1 for r in got if r["s2_src"] == t) for t in sorted(e["stage2"])}
            replay += got
    base = traps + everyday + know + replay
    rng.shuffle(base)
    ctx.write_jl(ctx.path("mix2", "rows.jsonl"), base)
    rep = {"presup_rows": len(traps), "trap_rows": sum(1 for r in traps if r.get("origin") == "trap"),
           "twin_rows": sum(1 for r in traps if r.get("origin") == "twin"), "everyday": len(everyday),
           "knowledge": len(know), "own": {"presup2": len(traps), "everyday": len(everyday), "knowledge": len(know)},
           "replay_by_book": by_book, "replay_total": len(replay), "rows": len(base), "seed": SEED}
    json.dump(rep, open(ctx.path("mix2", "mix_report.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(rep))


if __name__ == "__main__":
    {"replay-plan": replay_plan, "replay-sample": replay_sample, "stage1": stage1, "stage2": stage2}[sys.argv[1]]()
