# PLAN_R4B - keyword model r4b ("hel" without the final /p/ should not be "help")

**The code is written and tested** (`python tools/smoke_r4.py` now also runs the r4b steps on
synthetic audio). The job is to **run it on the real data, watch it, check the numbers and
report** - not to write training code. Same rules as `PLAN_R4.md`.

## Why
On the device with r4 (firmware v1.6.8), the owner saying **"hel"** (no final /p/) is heard as
"help"; r3 did not do this. The owner's real "help" now scores ~0.96. Likely causes:
1. r4 learned the owner's voice: 25 % of the "help" windows were his recordings
   (`own_share`), so his "hel" is very close to his "help" - only the weak, unvoiced /p/ differs;
2. the final /p/ is often nearly silent in real recordings (shouted / far takes, MSWC), so the
   model learned that "hel..." is enough;
3. look-alike words (hell, held, helm, hello ...) went from 0.11 (r3) to 0.07 of `_other` in r4,
   and no training window ever had "help" without its end.
Placement is not the cause: `kws/dataset.py` `place()` keeps the whole keyword (and its tail)
inside the window.

## What is new
| file | what |
|---|---|
| `kws/dataset.py` `cut_tail()` | a "help" clip cut at the /p/ closure (quietest 10 ms frame in the last part of the word; else 55-80 % of the word), 0-40 ms earlier, 5 ms fade |
| `kws/dataset.py` group `cut_help` | `_other` windows of cut "help": half from the owner's takes (lighter `augment_own`), half MSWC / LibriSpeech; placed like a keyword |
| `configs/train.yaml` `cut_help`, `train_kw_r4b`, `export_r4b` | from r4, 8 epochs, lr 3e-4; `cut_help` 0.08, `mswc_lookalike` back to 0.11; `own_share` 0.15 / 0.15 / 0.25 (r4: 0.25 / 0.25 / 0.35) |
| `scripts/21_eval_r4.py` | new sets `cut_own_help`, `cut_real_help` (the val / test help calls with the end cut off): **"right" = heard as help = false accept**, `@pair` = "hel ... hel" -> false SOS; per set `score_median` / `score_p90` (own vs public voice); `--sweep --tag` writes `eval_r4_sweep_val_<tag>.json` |
| `scripts/run_r4b.sh` | steps 0-5 below |

## Steps (`bash scripts/run_r4b.sh N`)
| N | what | time | check before going on |
|---|---|---|---|
| 0 | firmware detector check + smoke test (now with r4b) | 5-10 min | `0 failed` |
| 1 | **before**: r4 on val + test (PyTorch) and test (int8) with the new cut sets | 30-60 min | `eval_r4_r4_{val,test}.json`, `eval_r4_r4_int8_test.json`: `cut_own_help.right` should be high (the problem in numbers); the other sets should match the old r4 jsons if their thresholds were the same (say if not) |
| 2 | train r4b | 15-40 min | `models/kw_r4b_log.csv`: loss falls, val accuracy does not collapse |
| 3 | r4b sweep on val (rule 0) | 30-60 min | report the 0.65 / 0.70 / 0.75 rows (calls, cut, talk false SOS); thresholds stay as `export_r4b` unless the owner says otherwise |
| 4 | **after**: r4b on test + `docs/kw_r4_r4b_compare_test.png`, `docs/kw_r4_r4b_cut_help_test.png` | 10-20 min | compare with step 1 (targets) |
| 5 | export + int8 on test + r4b on the owner's takes | 10 min | `check_c` 0 failed; int8 ~= step 4; `out/kws_model.c` name `kws-r4b` |

## Targets (test, int8, medium 0.70, 2 calls in 10 s, help +0.05)
- `cut_own_help.right` (owner's "hel" heard as help): **at least 3x lower** than r4;
- `cut_own_help@pair.sos_rate` ("hel ... hel" -> SOS): clearly lower than r4;
- `own_help.heard` >= 90 %; `own_help_me`, `own_sos` not lower than r4;
- public calls (`real_help`, `real_help_me`, `tts_sos`): not lower than r4 (r4 lost points against r3 here - less own voice should help);
- talk false SOS / 24 h and coughs: not worse than r4.
If a target is missed: report it plainly, **never tune on test**.

## Rules
As `PLAN_R4.md`: no new training code; `data/`, `build/`, `logs/` never committed; errors ->
show the error and the smallest fix, apply only after the owner agrees, note it in REPORT.md.
