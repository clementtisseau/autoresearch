#!/usr/bin/env bash
# Gate 1 on a fresh GPU pod, in one command: install, data, smoke test, baseline x 3 seeds, noise report.
# Usage (from the repo root):  bash scripts/gate1.sh
# Everything is logged to gate1.log; the final summary is in gate1_summary.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
exec > >(tee -a gate1.log) 2>&1
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s) - T0) / 60 )) min] $*"; }

step "1/6 GPU check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

step "2/6 install uv + dependencies"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

step "3/6 data (20 shards) + tokenizer"
uv run prepare.py --num-shards 20

step "4/6 smoke test (1 nano run)"
rm -rf runs/smoke
uv run ladder.py run smoke --rungs nano
if ! grep -q '"status": "ok"' runs/smoke/nano_s0.json; then
  echo "SMOKE TEST FAILED -- last lines of the training log:"
  tail -40 runs/smoke/nano_s0.log
  exit 1
fi
cat runs/smoke/nano_s0.json
rm -rf runs/smoke

step "5/6 gate 1: baseline, 3 small rungs x 3 seeds"
uv run ladder.py run baseline --seeds 0 1 2

step "6/6 noise report"
{
  uv run ladder.py noise
  echo
  echo "rung   seed  val_bpb   train_s  wall_s  mfu%  steps  epoch"
  uv run python - <<'EOF'
import json, glob, os
for p in sorted(glob.glob("runs/baseline/*.json")):
    r = json.load(open(p)); name = os.path.basename(p)[:-5]
    if r.get("status") != "ok":
        print(f"{name:12s} CRASH {r.get('reason')}"); continue
    rung, seed = name.rsplit("_s", 1)
    print(f"{rung:6s} {seed:>4s}  {r['val_bpb']:.6f}  {r['training_seconds']:7.0f}  {r['wall_seconds']:6.0f}"
          f"  {r['mfu_percent']:4.1f}  {r['num_steps']:5d}  {r['final_epoch']}")
EOF
} | tee gate1_summary.txt

step "DONE in $(( ($(date +%s) - T0) / 60 )) min. Copy gate1_summary.txt, then TERMINATE the pod."
