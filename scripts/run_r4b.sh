#!/usr/bin/env bash
# run_r4b.sh - r4b: r4 + "help" with the end of the word cut off ("hel", no final /p/) as _other.
# Every step writes logs/r4b_step<N>.log. PLAN_R4B.md explains why, HANDOFF_R4B.md how to supervise.
#
#   0  smoke test (tools/smoke_r4.py now also runs the r4b steps on synthetic audio)
#   1  "before": r4 on val + test, PyTorch and the int8 C engine (adds the new cut_* sets: "hel" heard as help)
#   2  10_train_kw.py --stage r4b (warm start from r4, 8 epochs)
#   3  r4b threshold sweep on val (rule 0) -> models/eval_r4_sweep_val_r4b.json (reported; the owner decides)
#   4  "after": r4b on test with export_r4b + comparison figures r4 vs r4b
#   5  export r4b (out/kws_model.c) + the int8 C engine on test + r4b on every one of the owner's takes
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
STEP="${1:?step number 0-5}"
LOG="logs/r4b_step${STEP}.log"
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }
echo "=== step $STEP  $(date)" >> "$LOG"

case "$STEP" in
0)
  run python -m kws.fwdet --build --check
  run python tools/smoke_r4.py
  run df -h .
  ;;
1)
  run python scripts/30_export.py --section export_r4          # out/kws_model.c = r4 again (step 5 overwrites it with r4b)
  run python scripts/21_eval_r4.py --model models/kw_r4.pt --split val --tag r4 --from-export export_r4
  run python scripts/21_eval_r4.py --model models/kw_r4.pt --split test --tag r4 --from-export export_r4
  run python scripts/21_eval_r4.py --native out/kws_model.c --split test --tag r4_int8 --from-export export_r4
  ;;
2)
  run python scripts/10_train_kw.py --stage r4b
  ;;
3)
  run python scripts/21_eval_r4.py --model models/kw_r4b.pt --split val --sweep --tag r4b
  ;;
4)
  run python scripts/21_eval_r4.py --model models/kw_r4b.pt --split test --tag r4b --from-export export_r4b
  run python scripts/21_eval_r4.py --plot r4 r4b
  ;;
5)
  run python scripts/30_export.py --section export_r4b
  run python scripts/21_eval_r4.py --native out/kws_model.c --split test --tag r4b_int8 --from-export export_r4b
  run python scripts/21_eval_r4.py --plot r4_int8 r4b_int8
  run python tools/check_own.py --model models/kw_r4b.pt --from-export export_r4b
  ;;
*)
  echo "unknown step $STEP"; exit 1;;
esac
echo "=== step $STEP done $(date)" | tee -a "$LOG"
