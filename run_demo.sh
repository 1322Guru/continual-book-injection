#!/usr/bin/env bash
# Re-runnable demo: two public-domain books in sequence, forgetting of book A measured with three replay arms.
#   ./run_demo.sh            full run: T0, T1, arms none / share / fixed (seed 42), share / fixed (seed 43)
#   ./run_demo.sh --smoke    20 steps per stage, 10 eval questions per book, seed 42 only, into <out>_smoke
#   ./run_demo.sh --plan     print exact step counts and a time estimate, then exit
# Optional: CBI_OUT (output folder), CBI_MODELS (model folder), CBI_PYTHON (an existing Python with the lock installed).
# A stage with a .done marker is skipped, so re-running resumes.
set -euo pipefail
cd "$(dirname "$0")"
MODE=full
[ "${1:-}" = "--smoke" ] && MODE=smoke
[ "${1:-}" = "--plan" ] && MODE=plan
OUT="${CBI_OUT:-$PWD/out}"
[ "$MODE" = smoke ] && OUT="${OUT}_smoke"
export CBI_MODELS="${CBI_MODELS:-$PWD/models}"
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
N=0
if [ "$MODE" = smoke ]; then export CBI_MAX_STEPS=20; N=10; fi

if [ -z "${CBI_PYTHON:-}" ]; then
  if [ ! -x .venv/bin/python ]; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
    [ "${DRIVER%%.*}" -ge 580 ] || { echo "NVIDIA driver $DRIVER: requirements.lock needs >= 580" >&2; exit 1; }
    python3.12 -m venv .venv
    .venv/bin/pip install -q --upgrade "pip<26"
    .venv/bin/pip install -q -r requirements.lock
  fi
  CBI_PYTHON="$PWD/.venv/bin/python"
fi
PY="$CBI_PYTHON"

(cd data && sha256sum -c --quiet SHA256SUMS)
"$PY" demo/fetch_base.py "$CBI_MODELS"
BASE="$CBI_MODELS/Qwen2.5-7B-Instruct"
if [ "$MODE" = plan ]; then "$PY" demo/plan.py "$BASE"; exit 0; fi
mkdir -p "$OUT"

stage() {   # stage <name> <command...>: skip when done, log wall seconds
  local name=$1 t0; shift
  [ -f "$OUT/.done.$name" ] && { echo "skip $name"; return 0; }
  t0=$(date +%s); echo "$(date +%T) start $name"
  "$@"
  echo "{\"stage\": \"$name\", \"wall_s\": $(( $(date +%s) - t0 ))}" >> "$OUT/stages.jsonl"
  touch "$OUT/.done.$name"
}
answers() { "$PY" demo/answer.py "$1" "$OUT/$2/answers.jsonl" "$N" data/eval/book_a.jsonl data/eval/book_b.jsonl; }
train() {   # train <dir> <rows> <seed>
  "$PY" pipeline/train_bf16.py --epochs 2 --lr 3e-4 --r 128 --alpha 128 --micro 4 --seed "$3" "$OUT/$1/train" "$2"
}
book() {    # book <dir> <prev dir or empty>: environment of one training step
  export CBI_BOOK_OUT="$OUT/$1" CBI_BOOK_ID="$1" CBI_PREV="$2"
  mkdir -p "$OUT/$1"
  if [ -n "$2" ]; then echo "$2" > "$OUT/$1/prev.txt"; fi
}

stage t0_score answers "$BASE" t0
book t1 ""
stage t1_mix "$PY" demo/mix_demo.py t1 "$OUT/t1/rows.jsonl"
stage t1_train train t1 "$OUT/t1/rows.jsonl" 42
stage t1_merge "$PY" pipeline/merge_export.py merge "$OUT/t1/train/adapter" "$OUT/t1/merged"
stage t1_score answers "$OUT/t1/merged" t1

ARMS="none:42 share:42 fixed:42 share:43 fixed:43"
[ "$MODE" = smoke ] && ARMS="none:42 share:42 fixed:42"
for spec in $ARMS; do
  arm=${spec%%:*}; seed=${spec##*:}; d="arm_${arm}_s${seed}"
  book "$d" "$OUT/t1"
  stage "${d}_mix" "$PY" demo/mix_demo.py arm "$arm" "$seed" "$OUT/$d/rows.jsonl"
  stage "${d}_train" train "$d" "$OUT/$d/rows.jsonl" "$seed"
  stage "${d}_merge" "$PY" pipeline/merge_export.py merge "$OUT/$d/train/adapter" "$OUT/$d/merged"
  stage "${d}_score" answers "$OUT/$d/merged" "$d"
  rm -rf "${OUT:?}/${d:?}/merged" "${OUT:?}/${d:?}/train"/adapter_epoch*
done
"$PY" demo/report.py "$OUT"
echo "results: $OUT/results.md"
