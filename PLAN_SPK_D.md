# PLAN_SPK_D - Phase D: a speaker model that an imitator cannot pass by changing her pitch

**The code is written and tested** (`python tools/smoke_spk_d.py`: every step on synthetic voices, 6/6 PASS).
The job of this phase is to **run it, watch it, check the numbers and report** - not to write training code.

## Why
"Only My Voice" (spk-20261023, firmware v1.6.x): the owner scores 0.53-0.83 on the device, a woman
lowering her voice to imitate him 0.38-0.61; after about a dozen tries one call passed. Two causes:
1. **retries add up**: a 5 % chance per call is ~40 % within 10 tries; every call is judged alone;
2. **the model was never told what a voice is**: nothing in training separated *pitch* (which a person
   changes at will - shouting, imitating) from the *vocal tract* (formants, which a person cannot change).
   A lowered female voice may therefore look closer to a man's than it should.
Earlier attempts (REPORT.md B2, B3, the local big120 run) changed the training signal (distillation)
or the size; none reached the 15 % gain. Phase D changes **what the model must ignore and what it must use**.

## Idea (three variants, same network, data, seeds and schedule; only these switches differ)
| variant | speaker augmentation | pitch invariance | question it answers |
|---|---|---|---|
| d0 | - | - | what the recipe alone gives (GPU, from scratch, B2 data + device chain + AISHELL voices) |
| d1 | yes | - | does "another vocal tract = another speaker" help? |
| d2 | yes | yes | does "another pitch = the same speaker" stop the imitation? |

- **speaker augmentation**: speed-perturbed copies (x0.9 / x1.1: F0 *and* formants move) are labelled as
  **new speakers** (Yamamoto et al., Interspeech 2019; Kaldi speed perturbation). 3x the classes.
- **pitch invariance**: copies with only the pitch moved by -5...+5 semitones, formants kept
  (`spk/voice_mod.py`: LPC source-filter - inverse filter, pitch-shift the residual, filter again with
  the original envelope; tested: F0 ratio within 4 %, F1 / F2 within 5 %) keep **the same speaker label**.
  An imitator who only moves her pitch keeps her own formants; the owner shouting (higher pitch) stays himself.
- **Same SpkNet as spk-20261023** (40 log-mel x 81 frames -> 128, 12.8 M MAC, 229 KB): the firmware
  C engine, memory and speed stay the same; only `spk_model.c` changes (re-enrolment needed).
- Scoring methods compared on the same vectors (`spk/phase_d.py`): `max` (the firmware now),
  `asn` (AS-norm with a 300-speaker cohort, 38 KB on the device), `pair` (the two calls of an SOS averaged), `pasn`.

## Tests (`scripts/48_eval_spk_d.py`, identical audio for every model)
- public speakers (val 150 / test 600, MSWC + LibriSpeech + VCTK), 10 enrolment takes, conditions
  clean / far / shout / noisy; same- and cross-gender impostors;
- **imitation** `mimic_f2m`: a woman's takes with only the pitch lowered by 4-8 semitones, against men's
  templates (`plain_f2m` = the same takes unchanged; `mimic_m2f` the other way round);
- **the owner's real recordings** (DataRec, device microphone): own-A = owner (10 NORMAL takes enrolled,
  30 random enrolments), own-B = the woman who imitated him: EER, score ranges, owner acceptance per
  style, and **the imitator's chance of an SOS within 2 / 6 / 10 / 20 tries** at the threshold that
  rejects 5 % of the owner's calls. These recordings are never used for training or choosing.
- `firmware_thresholds`: new-model thresholds with the same same-gender false-accept rate as the old
  model at 0.46 / 0.52 / 0.60 (Loose / Normal / Strict of "help") - for the firmware table.

## Steps (`bash scripts/run_spk_d.sh N`)
| N | what | check before going on |
|---|---|---|
| 0 | GPU, cache files, old model, voice_mod, smoke test | `0 failed`; all files `ok`; own-A and own-B counts |
| 1 | **before**: old model (+ big120 if present) on val + own | `spk_d_eval_val.json`: old `select_metric`, `mimic_f2m` vs `plain_f2m`, own imitator SOS chance |
| 2-4 | train d0, d1, d2 | `models/spk_d*_log.csv`: loss falls, `val_select` falls then flattens; minutes per epoch |
| 5 | all models on val | `best_variant_val` (lowest mean of the 4 conditions + imitation, scoring `max`) |
| 6 | all models on test + own | table below; never change anything because of test |
| 7 | export the chosen variant (**only after the owner agrees**) | `check_c` 0 failed; id, threshold |

## Targets (test, scoring `max` unless said)
- imitation `mimic_f2m` EER at least **25 % lower** than the old model;
- mean EER of clean / far / shout / noisy **not worse** than the old model (+0.5 points at most);
- own recordings: imitator SOS chance within 10 tries **lower** than the old model, owner acceptance (NORMAL / LOUD / FAR) not lower;
- report how much `asn` and `pair` add for each model (they need a firmware change; decided afterwards).
A missed target is reported plainly. The variant is chosen on val; test and own are only reported.

## Rules
As `PLAN_R4.md`: no new training code; `data/`, `build/`, `logs/` never committed (`data/own` is the
owner's and the imitator's real voices); errors -> show the error and the smallest fix, apply only after
the owner agrees, note it in REPORT.md.
