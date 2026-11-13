# MeshTalk-KWS - training package for MeshTalk voice SOS (firmware v1.4.1)

| file | what |
|---|---|
| `REPORT.md` | results, numbers first (round 1 and round 2) |
| `TRAINING.md` | (Chinese) every parameter and why, commands to reproduce, environment, time per step |
| `CONTRACT.md`, `CONTRACT_SPK.md` | what the firmware expects (keyword model; separate speaker model) |
| `PLAN.md`, `HANDOFF.md`, `update_v1.4.1/` | the owner's plan and hand-over files |
| `PLAN_R4.md`, `HANDOFF_R4.md` | **Phase C (r4)**: fewer false calls from coughs / talk - steps, targets, the prompt for the supervising session |
| `scripts/04-06, 13, 21, run_r4.sh` | Phase C: own recordings (MeshTalk DataRec), VocalSound / AMI / AISHELL-1, r4 hard negatives, evaluation with the firmware v1.6.7 detector, step runner |
| `c_ref_v167/`, `kws/fwdet.py` | the firmware v1.6.7 keyword detector in C, called from Python (`--native`: the int8 C engine) |
| `tools/smoke_r4.py`, `tools/check_own.py` | Phase C smoke test on synthetic audio; r3 / r4 on every one of the owner's takes |
| `configs/train.yaml`, `configs/spk.yaml` | EVERY training parameter (keyword model / speaker model) |
| `scripts/` | 01 download, 02 prepare, 03 TTS, 10 train keyword model, 12 hard-negative mining, 20 system-level evaluation, 30 export; 40-42 speaker model |
| `kws/` | features / model / detect / export / check_c (firmware twins, unchanged) + cfg, audio, augment, dataset, evaluate |
| `spk/` | separate speaker model: network, data, evaluation, int8 export, C engine `spk/c/`, `check_c` |
| `models/` | `best.pt` (keyword model v1), `best_int8sim.pt` (its int8 equivalent), `kw_r1/r2/r3.pt`, `spk_best.pt`, logs, evaluation JSON |
| `out/` | for the firmware: `kws_model.c` (-> `Ai/`), `kws_model_test.c` (-> `Tests/`), `spk_engine.c` + `spk_model.c` + `spk_model.h` (-> `Ai/`), `spk_model_test.c` + `spk_test.c` (-> `Tests/`) |
| `docs/` | training curves, confusion matrices, trigger-rate vs false-alarm curves, EER curves |
| `c_ref/` | copy of the firmware C code (v1.4.1) |

Checks: `python -m kws.check_c out` -> `20 checks, 0 failed`; `python -m spk.check_c out` -> `10 checks, 0 failed`.
