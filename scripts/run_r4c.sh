#!/usr/bin/env bash
# run_r4c.sh - r4c: from r4, "help" with the end cut off as _other from PUBLIC clips only (r4b also used the
# owner's: his help collapsed), owner share as r4. Every step writes logs/r4c_step<N>.log. See HANDOFF_R4C.md.
# The "before" numbers (r4 with the cut sets) are already in models/eval_r4_r4_{val,test}.json, eval_r4_r4_int8_test.json.
#
#   0  smoke test (synthetic audio, includes r4c)
#   1  10_train_kw.py --stage r4c (warm start from r4, 8 epochs)
#   2  r4c threshold sweep on val (rule 0) -> models/eval_r4_sweep_val_r4c.json (reported; the owner decides)
#   3  r4c on test with export_r4c + figures r4 vs r4c
#   4  export r4c (out/kws_model.c) + int8 C engine on test + figures r4_int8 vs r4c_int8 + r4c on the owner's takes
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
STEP="${1:?step number 0-4}"
LOG="logs/r4c_step${STEP}.log"
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }
echo "=== step $STEP  $(date)" >> "$LOG"

case "$STEP" in
0)
  run python tools/smoke_r4.py
  ;;
1)
  run python scripts/10_train_kw.py --stage r4c
  ;;
2)
  run python scripts/21_eval_r4.py --model models/kw_r4c.pt --split val --sweep --tag r4c
  ;;
3)
  run python scripts/21_eval_r4.py --model models/kw_r4c.pt --split test --tag r4c --from-export export_r4c
  run python scripts/21_eval_r4.py --plot r4 r4c
  ;;
4)
  run python scripts/30_export.py --section export_r4c
  run python scripts/21_eval_r4.py --native out/kws_model.c --split test --tag r4c_int8 --from-export export_r4c
  run python scripts/21_eval_r4.py --plot r4_int8 r4c_int8
  run python tools/check_own.py --model models/kw_r4c.pt --from-export export_r4c
  ;;
*)
  echo "unknown step $STEP"; exit 1;;
esac
echo "=== step $STEP done $(date)" | tee -a "$LOG"
