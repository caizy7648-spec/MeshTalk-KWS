#!/usr/bin/env bash
# run_spk_d.sh - Phase D: speaker model that resists pitch imitation. Every step writes logs/spk_d_step<N>.log.
# PLAN_SPK_D.md explains why and what each number means; HANDOFF_SPK_D.md how to supervise.
#
#   0  checks: GPU, cache files, the old model, voice_mod, smoke test (synthetic)
#   1  "before": the old model (spk-20261023) and big120 (if present) on val + the owner's recordings
#   2  train d0 (the recipe alone)            ~ GPU, see the log for minutes per epoch
#   3  train d1 (+ speaker augmentation)
#   4  train d2 (+ pitch invariance)
#   5  all models on val -> models/spk_d_eval_val.json "best_variant_val" (chosen on val only)
#   6  all models on test + own recordings (report only)
#   7  export the chosen variant:  bash scripts/run_spk_d.sh 7 d2 [threshold]   (only after the owner agrees)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
STEP="${1:?step number 0-7}"
LOG="logs/spk_d_step${STEP}.log"
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }
echo "=== step $STEP  $(date)" >> "$LOG"

case "$STEP" in
0)
  run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
  run python - <<'PY'
import os
need = ["data/cache/clips.npy", "data/cache/spk_extra.npy", "data/cache/spk_extra.csv", "data/cache/spk_gender.csv",
        "data/cache/noise_train.npy", "data/cache/noise_test.npy", "data/cache/own.npy", "data/cache/chat.npy",
        "models/spk_best_v1.pt"]
miss = [p for p in need if not os.path.exists(p)]
for p in need:
    print("%-32s %s" % (p, "ok" if p not in miss else "MISSING"))
print("big120:", "models/spk_big120.pt", "present" if os.path.exists("models/spk_big120.pt") else "not present (skipped)")
import pandas as pd
o = pd.read_csv("data/cache/own.csv", keep_default_na=False)
print("own takes per speaker / style:"); print(o.groupby(["speaker", "word"]).size().to_string())
raise SystemExit(1 if miss else 0)
PY
  run python tools/smoke_spk_d.py
  run df -h .
  ;;
1)
  run python scripts/48_eval_spk_d.py --split val --models old big120
  ;;
2|3|4)
  V=$([ "$STEP" = 2 ] && echo d0 || ([ "$STEP" = 3 ] && echo d1 || echo d2))
  run python scripts/47_train_spk_d.py --variant "$V"
  ;;
5)
  run python scripts/48_eval_spk_d.py --split val
  ;;
6)
  run python scripts/48_eval_spk_d.py --split test
  ;;
7)
  V="${2:?variant, e.g. d2}"
  THR="${3:-}"
  if [ -n "$THR" ]; then
    run python scripts/42_export_spk.py --model "models/spk_${V}.pt" --name "spk-${V}-$(date +%Y%m%d)" --threshold "$THR"
  else
    run python scripts/42_export_spk.py --model "models/spk_${V}.pt" --name "spk-${V}-$(date +%Y%m%d)"
  fi
  ;;
*)
  echo "unknown step $STEP"; exit 1;;
esac
echo "=== step $STEP done $(date)" | tee -a "$LOG"
