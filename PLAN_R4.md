# PLAN_R4 - Phase C: keyword model r4 (fewer false SOS from coughs and talk)

Read first: `CONTRACT.md` (unchanged - same network, same sizes limits, same
export), `TRAINING.md` (how r1-r3 were made), `REPORT.md` (r3 numbers).

## Why
Firmware v1.6.7 with model r3 (`models/best.pt`): on the device, single
**"help"** is triggered by **coughs** (a hard cough scores ~0.65 for "help")
and sometimes by people talking nearby. Causes found:
1. `_other` never contained a cough, throat-clear, sneeze, laugh or breath
   as a clip of its own - ESC-50 was only used as background noise under
   speech. A cough is out of distribution and the softmax puts it on the
   shortest, breathiest class: "help".
2. Hard negatives were mined on LibriSpeech (read audiobooks) only - no
   conversational English, no Chinese at all (the owner and the people around
   him speak Chinese: he / hei / hao / hai / ha are short h- syllables).
3. No real recording through the device microphone was ever used.

Firmware v1.6.7 already raised single "help" by +0.05 and removed the
confirm-drop / word-slack relaxations for it (`c_ref_v167/svc_kws.c`,
`HELP_EXTRA`, `kws_det_t.cls_extra / no_relax`). Evaluate with that logic.

## Data
### A. The owner's recordings (positives, the device microphone)
Recorded with MeshTalk DataRec v1.0 (16 kHz / 16 bit / mono WAV, 2.5 s per
take, one call per take, styles NORMAL / LOUD / SOFT / HOARSE / FAR,
speaker letters A = owner, B-E = friends). Expected ~150 HELP, ~100 HELPME,
~100 SOS. Import:

    python scripts/04_import_own.py --src <path to DATASET folder or zip>

-> `data/own/` + `data/own/manifest.csv` (split already decided: a speaker
with < 30 clips is test as a whole, otherwise 20 % test / 10 % val per file).
**`data/own` is never committed** (the repo is public, it is a real voice) -
no audio of it in `docs/` either; numbers and plots only.

Before training: run r3 on every own clip (keyword fully inside a 1.22 s
window, best position) and list clips where r3's top class is confidently a
*different keyword* or the take is silent / two calls - possible wrong takes.
Show the owner the list (file names) and continue without those clips only if
he agrees; otherwise keep them.

### B. Public negatives, as standalone `_other` clips (new)
Non-call vocal sounds - each clip placed in the window like a word, at full
level, also 2-3 of them in a row (coughing fits), also with background noise:
| need | candidates (licence - verify, write in REPORT) |
|---|---|
| cough | COUGHVID (Zenodo, CC-BY 4.0); FSD50K class Cough (Zenodo, per-clip CC licences, keep CC0/CC-BY); ESC-50 `coughing` (already downloaded, CC-BY-NC) |
| throat clearing, sneeze, sniff, breathing, gasp, laughter | FSD50K (Throat clearing, Sneeze, Sniff, Breathing, Gasp, Laughter ...); ESC-50 `sneezing`, `breathing`, `laughing`, `snoring` |
| conversational English | something spontaneous, not read: e.g. AMI meeting corpus (CC-BY 4.0) or People's Speech conversational parts (used in phase B2) |
| Chinese speech | AISHELL-1 (openslr 33, Apache-2.0) and/or Common Voice zh-CN (CC0) or MAGICDATA (openslr 68, check licence) |

Keep ESC-50 fold 5, a held-out part of every new source (by speaker /
recording id) and >= 2 h of the conversational sets for **test only**.
If a host is blocked: write exact URL / size into REPORT.md and stop, as before.

## Training
- **Warm start from r3** (`models/best.pt`), same network (CONTRACT), stage
  `r4` in `scripts/10_train_kw.py`; all new parameters in `configs/train.yaml`
  (sampling weights of the new sources, own-clip weight, epochs, lr).
- Own clips: oversample (they are few), with the usual augmentation but
  lighter (they are already the real microphone).
- One round of hard-negative mining with r4-pre on the new conversational /
  Chinese / vocal-sound training parts (`scripts/12_mine.py`), then the final
  fine-tune.
- Re-choose `thr` (low/medium/high) and `cls_off` on val; say whether the
  firmware's `HELP_EXTRA` (+0.05 on help) is still needed with r4 or can be 0.

## Evaluation (r3 vs r4, same test sets, same detector)
Use the **firmware v1.6.7 detector**: compile `c_ref_v167/` (kws_frontend.c,
kws_engine.c, kws_detect.c) into a small harness (CLI or ctypes) that feeds
16 kHz audio in 20 ms blocks exactly as `svc_kws.c` does, with medium
sensitivity, rule "2 calls in 10 s" (words mode on) and help
`cls_extra = 0.05, no_relax = 1`. Check it against `kws.detect` on the r3 test
set and explain any difference. Report, for r3 and r4:

| metric | test set |
|---|---|
| hit rate per keyword, per style (single call, and SOS with 2 calls) | own test split (device mic) |
| false hits per 100 coughs / throat clears / ...; false SOS per 100 coughing fits | held-out vocal sounds |
| false hits per hour, false SOS per 24 h | held-out conversational EN, Chinese, and the old r3 test hours |
| the old r3 numbers | old test sets (must not get worse by more than 2 points) |

Targets (medium): cough false hits at least **5x lower** than r3; false SOS
<= 1 / 24 h on talk + vocal sounds; own-recording hit rate for help / help me /
SOS **not lower than r3** (aim >= 90 % for help). If a target is missed, say
so plainly - do not tune on test.

Figures in `docs/` (prefix `kw_r4_`): training curves, confusion matrix,
r3-vs-r4 bar chart of the table above, histogram of the "help" score on coughs
(r3 vs r4), trigger-vs-false-alarm curves.

## Export and hand back
- `python scripts/30_export.py` -> `out/kws_model.c`, `out/kws_model_test.c`,
  `python -m kws.check_c out` 0 failed; also run the exported model through
  the `c_ref_v167` harness once (same numbers as Python, within rounding).
- `models/best.pt` = r4 (keep r3 as `models/kw_r3.pt`).
- `REPORT.md`: Phase C section, numbers first; data counts and licences;
  the r3-vs-r4 table; thresholds; whether HELP_EXTRA should stay.
- `TRAINING.md`: Phase C section in Chinese - every new parameter with its
  value and why, exact commands to reproduce (including 04_import_own.py with
  the owner's DATASET path), environment, time per step.
- Commit + push after every step. Never commit `data/`.
