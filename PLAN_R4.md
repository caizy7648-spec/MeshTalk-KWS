# PLAN_R4 - Phase C: keyword model r4 (fewer false SOS from coughs and talk)

**The code is written and tested** (`python tools/smoke_r4.py` runs every step on synthetic
audio: 14/14 PASS). The job of this phase is to **run it on the real data, watch it, check the
numbers and report** - not to write new training code. Read `CONTRACT.md` (firmware limits,
unchanged) and `TRAINING.md` section 8 (Phase C, every parameter and why).

## Why
Firmware v1.6.7 with model r3: single **"help"** is triggered by **coughs** (a hard cough scores
~0.65 for "help") and sometimes by people talking nearby. Causes:
1. `_other` never contained a cough, throat-clear, sneeze, laugh or breath as a clip of its own
   (ESC-50 was only background noise under speech) - a cough is out of distribution and the
   softmax puts it on the shortest, breathiest class, "help";
2. hard negatives came only from LibriSpeech (read audiobooks): no conversation, no Mandarin;
3. no recording through the device microphone was ever used.
v1.6.7 already raised single "help" by +0.05 and removed its confirm-drop / word-slack
(`c_ref_v167/svc_kws.c`, `HELP_EXTRA`). Everything is measured with that firmware logic.

## What is new (all in the repository)
| file | what |
|---|---|
| `scripts/04_import_own.py` | the owner's DataRec recordings -> `data/own/` + `manifest.csv` (split fixed) |
| `tools/check_own.py` | r3 (or r4) listens to every take with the device logic: hit rate per word / style, possible wrong takes |
| `scripts/05_download_r4.py` | VocalSound (cough, throat clearing, sneeze, sniff, sigh, laughter; 3 365 speakers), AMI meetings (English conversation) + word transcripts, AISHELL-1 (Mandarin, streamed) |
| `scripts/06_prepare_r4.py` | caches `own`, `vocal`, `chat`, false-alarm streams `fa4_val/test` |
| `scripts/13_mine_r4.py` | where r3 fires on the new training audio -> `hardneg_r4` |
| `kws/dataset.py` | loads the new caches; `own_share` of keyword windows from the owner (lighter `augment_own`) |
| `scripts/10_train_kw.py --stage r4` | warm start from r3, `train_kw_r4` in the config |
| `kws/fwdet.py` + `c_ref_v167/harness.c` | the **firmware v1.6.7 detector in C** (front end, sound segments, words mode, 2 calls in 10 s, help +0.05) called from Python; `--native` = the int8 C engine of an exported `kws_model.c` |
| `scripts/21_eval_r4.py` | r3 vs r4: calls (own / public), talk (AMI, AISHELL, LibriSpeech), vocal sounds; threshold sweep on val; comparison figure |
| `scripts/30_export.py --section export_r4` | export with the r4 settings |
| `scripts/run_r4.sh <step>` | the steps below, each logged to `logs/r4_step<N>.log` |
| `tools/smoke_r4.py` | the whole pipeline on synthetic data in `build/smoke_r4/` (a few minutes) |

## Steps (`bash scripts/run_r4.sh N`)
| N | what | time (24 cores) | check before going on |
|---|---|---|---|
| 0 | packages, gcc, `kws.check_c`, firmware detector check, smoke test, disk | 5-10 min | `0 failed` everywhere; >= 100 GB free |
| 1 | import the owner's DATASET + r3 check of every take | 2 min | counts per label / style; the "may be wrong takes" list -> **ask the owner** |
| 2 | Step-1 data on this machine: 01_download (~75 GB streamed), 03_tts, 02_prepare | 2-6 h (network) | `data/raw/MANIFEST.json` checksums; `data/cache/counts.json` close to REPORT.md (r1-r3) |
| 3 | old hard negatives: 12_mine (r1 model), 12_mine --stage r3 (r2 model) | 2-5 h (network) | `mine_stats_r2.json` / `_r3.json` similar to REPORT.md |
| 4 | new data: 05_download_r4, 06_prepare_r4 | 1-2 h | `counts_r4.json`: vocal clips per type / split, chat, fa4 hours |
| 5 | **before**: r3 on val and test with the firmware detector | 10-20 min | `models/eval_r4_r3_{val,test}.json` |
| 6 | 13_mine_r4 | 10-30 min | `mine_stats_r4.json`: share of coughs etc. where r3 fires (the problem, in numbers) |
| 7 | train r4 (20 epochs from r3) | 20-60 min | `models/kw_r4_log.csv`: loss falls, val accuracy does not collapse |
| 8 | sweep thresholds of r4 on val | 30-60 min | `eval_r4_sweep_val.json` "recommended" -> copy `thr` and `help_extra` into `export_r4:` of the config |
| 9 | **after**: r4 on test + `docs/kw_r4_compare_test.png` | 10-20 min | compare with step 5 (targets below) |
| 10 | export + the int8 C engine on test + r4 check of the owner's takes | 5-10 min | `check_c` 0 failed; int8 numbers ~= step 9 |

If step 2 or 3 cannot be done (disk / time): r4 still trains - groups without data are left out
of the mix automatically - but say so in REPORT.md (the old hard negatives keep r3's low false
alarms on LibriSpeech; without them check the `libri` line of step 9 carefully).

## Targets (test, medium sensitivity, 2 calls in 10 s, firmware v1.6.7 logic)
- coughs: false hits per 100 at least **5x lower** than r3 (`vocal.per_type.cough`, `coughing`);
- talk (AMI + AISHELL + LibriSpeech): false SOS **<= 1 per 24 h**;
- the owner's calls (`own_*`): hit rate **not lower than r3**, aim >= 90 % for help;
- public calls (`real_help`, `real_help_me`, `tts_sos`): not more than 2 points below r3.
A missed target is reported plainly - **never tune on test**; thresholds come from val (step 8).

## Rules
- Do not change the training code or the network. If a step crashes, show the error and the
  smallest fix, and apply it only after the owner agrees; note it in REPORT.md.
- `data/` is never committed (public repository; `data/own` is the owner's real voice).
- After each step: a short note in REPORT.md (numbers first), commit + push (code, configs,
  `models/*.json`, `models/kw_r4.pt`, `docs/*.png`, `out/`). Never commit `data/`, `build/`, `logs/`.
- At the end: REPORT.md Phase C section (r3 vs r4 table from steps 5 / 9 / 10, data counts and
  licences, chosen thresholds, HELP_EXTRA recommendation), TRAINING.md section 8 results table
  filled in, `out/kws_model.c` for the firmware.
