"""GPU-h governor of a book: project this book's GPU-h on one A6000 from its
unit count and the recipe, and lower the knowledge rows per unit until the projection fits config governor.cap_gpu_h.
It never touches a filter, the trap / presupposition rows or the replay (fixed counts, recipe.replay; counted in the
projection). Writes <book>/plan.json (read by the
knowledge and replay phases) and logs the decision in decisions.jsonl.

  python -I pipeline/governor.py plan
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import ctx  # noqa: E402
from pipeline import units as U  # noqa: E402

TRAP_ROWS_PER_UNIT = (4862 + 610 + 814) / 1280   # diverse + unanswerable + presup1 rows of book 1


def project(n_units, rpu, book_index, cfg, replay=(0, 0)):
    """replay = (stage-1 rows, stage-2 rows) re-sampled from the earlier books (mix.replay_counts)."""
    G, R = cfg["governor"], cfg["recipe"]
    scale = n_units / 1280
    know = n_units * G["knowledge_rows_per_unit"] * rpu / 40
    everyday = R["everyday_rows"] * scale
    book_rows = know + n_units * G["verbatim_rows_per_unit"] + n_units * TRAP_ROWS_PER_UNIT
    new_rows = book_rows + everyday
    replay, replay2 = (replay if book_index > 1 else (0, 0))
    steps1 = 2 * math.ceil((new_rows + replay) * G["tokens_per_row"] / 512 / 32)
    steps2 = math.ceil(G["stage2_steps_per_1280_units"] * scale + replay2 / G["stage2_rows_per_step"])
    train_h = (steps1 + steps2) * G["s_per_step"] / 3600 + 2 * G["train_load_h"]
    trap_h = G["trap_chain_h_per_1280_units"] * scale
    know_h = know / 1000 * G["knowledge_h_per_1000_rows"]
    every_h = G["everyday_h"] if book_index == 1 else 0.0
    side = know_h + every_h
    gen_h = max(trap_h, side) if G.get("parallel", True) else trap_h + side
    total = G["fixed_h"] + gen_h + train_h
    return {"units": n_units, "rows_per_unit": rpu, "book_rows": round(book_rows), "everyday_rows": round(everyday),
            "stage1_rows": round(new_rows + replay), "replay_rows": round(replay), "stage2_replay_rows": round(replay2),
            "stage1_steps": steps1, "stage2_steps": steps2, "train_h": round(train_h, 2),
            "trap_chain_h": round(trap_h, 2), "knowledge_h": round(know_h, 2),
            "everyday_h": round(every_h, 2), "generation_h": round(gen_h, 2), "fixed_h": G["fixed_h"], "total_gpu_h": round(total, 2)}


def plan():
    cfg, _ = U.load_config()
    G = cfg["governor"]
    n = len(ctx.load_units())
    if ctx.slice_k() > 1:   # smoke slice: project the FULL book too
        n_full = len(ctx.load_units()) * ctx.slice_k()
    else:
        n_full = n
    idx = len(ctx.chain())
    rpu = int(cfg["book"]["rows_per_unit"])
    from pipeline import mix as MX
    rc = MX.replay_counts(cfg) if idx > 1 else {}
    rp_run = (sum(sum(c["stage1"].values()) for c in rc.values()), sum(sum(c["stage2"].values()) for c in rc.values()))
    rp = tuple(x * ctx.slice_k() for x in rp_run) if ctx.slice_k() > 1 else rp_run   # full-book projection of a slice
    proj = lambda units, r, rpl=rp: project(units, r, idx, cfg, rpl)
    p = proj(n_full, rpu)
    tried = [p]
    while p["total_gpu_h"] > G["cap_gpu_h"] and rpu - 5 >= G["min_rows_per_unit"]:
        rpu -= 5
        p = proj(n_full, rpu)
        tried.append(p)
    over = p["total_gpu_h"] > G["cap_gpu_h"]
    run_p = proj(n, rpu, rp_run)   # the units this run really writes (differs from the full book only in a smoke slice)
    out = {"book_index": idx, "units": n, "units_full_book": n_full, "rows_per_unit": rpu, "replay": {os.path.basename(b): c for b, c in rc.items()},
           "cap_gpu_h": G["cap_gpu_h"], "projection": p, "projection_run": run_p, "tried": tried, "over_cap_after_trim": over, "rates": G}
    json.dump(out, open(ctx.path("plan.json"), "w"), indent=1)
    ctx.decision("governor", {"units": n_full, "projection_at_40": tried[0]["total_gpu_h"], "cap_gpu_h": G["cap_gpu_h"]},
                 f"rows_per_unit {rpu}, projected {p['total_gpu_h']} GPU-h"
                 + (" (GPU-h STILL OVER THE CAP: logged)" if over else ""))
    print(json.dumps({k: out[k] for k in ("book_index", "units", "units_full_book", "rows_per_unit", "over_cap_after_trim")}
                     | {"replay_rows": p["replay_rows"], "stage2_replay_rows": p["stage2_replay_rows"]}
                     | {"projected_gpu_h": p["total_gpu_h"]}))


if __name__ == "__main__":
    plan()
