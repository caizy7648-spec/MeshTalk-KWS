#!/usr/bin/env bash
# run_r4.sh - Phase C (r4), one step at a time. Every step writes logs/r4_<step>.log.
#
#   bash scripts/run_r4.sh 0          # environment + smoke test (minutes)
#   bash scripts/run_r4.sh 1 <DATASET path>   # the owner's recordings + r3 check
#   bash scripts/run_r4.sh 2 ... 10   # see below
#
# Steps (PLAN_R4.md explains why; HANDOFF_R4.md how to supervise):
#   0  venv check, kws.check_c, firmware detector build/check, tools/smoke_r4.py
#   1  04_import_own.py <DATASET> + tools/check_own.py (r3 on every take)
#   2  Step-1 data again on this machine: 01_download, 03_tts, 02_prepare   (long: ~75 GB streamed)
#   3  old hard negatives: 12_mine.py (r1 model) + 12_mine.py --stage r3 (r2 model)   (long)
#   4  new public data: 05_download_r4.py + 06_prepare_r4.py
#   5  "before": r3 on val and test with the firmware detector (21_eval_r4.py)
#   6  13_mine_r4.py (where r3 fires on talk / vocal sounds)
#   7  10_train_kw.py --stage r4 (warm start from r3)
#   8  threshold sweep of r4 on val -> models/eval_r4_sweep_val.json (then set export_r4 in the config)
#   9  "after": r4 on test with export_r4 settings + comparison figure
#   10 export (30_export --section export_r4) + the int8 C engine on test
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
STEP="${1:?step number 0-10}"
LOG="logs/r4_step${STEP}.log"
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }
echo "=== step $STEP  $(date)" >> "$LOG"

case "$STEP" in
0)
  run python -c "import torch, numpy, scipy, soundfile, pandas, pyarrow, yaml, matplotlib, kokoro_onnx; print('python packages ok, torch', torch.__version__)"
  run gcc --version
  run python -m kws.check_c
  run python -m kws.fwdet --build --check
  run python tools/smoke_r4.py
  run df -h .
  ;;
1)
  SRC="${2:?path of the DATASET folder (or zip)}"
  run python scripts/04_import_own.py --src "$SRC"
  run python tools/check_own.py
  ;;
2)
  run python scripts/01_download.py
  run python scripts/03_tts.py
  run python scripts/02_prepare.py
  ;;
3)
  run python scripts/12_mine.py
  run python scripts/12_mine.py --stage r3
  ;;
4)
  run python scripts/05_download_r4.py
  run python scripts/06_prepare_r4.py
  ;;
5)
  run python scripts/21_eval_r4.py --model models/kw_r3.pt --split val --tag r3
  run python scripts/21_eval_r4.py --model models/kw_r3.pt --split test --tag r3
  ;;
6)
  run python scripts/13_mine_r4.py
  ;;
7)
  run python scripts/10_train_kw.py --stage r4
  ;;
8)
  run python scripts/21_eval_r4.py --model models/kw_r4.pt --split val --sweep
  echo "now copy 'recommended' (thr, help_extra) of models/eval_r4_sweep_val.json into export_r4: of configs/train.yaml" | tee -a "$LOG"
  ;;
9)
  run python scripts/21_eval_r4.py --model models/kw_r4.pt --split test --tag r4 --from-export export_r4
  run python scripts/21_eval_r4.py --plot r3 r4
  ;;
10)
  run python scripts/30_export.py --section export_r4
  run python scripts/21_eval_r4.py --native out/kws_model.c --split test --tag r4_int8 --from-export export_r4
  run python tools/check_own.py --model models/kw_r4.pt
  ;;
*)
  echo "unknown step $STEP"; exit 1;;
esac
echo "=== step $STEP done $(date)" | tee -a "$LOG"
