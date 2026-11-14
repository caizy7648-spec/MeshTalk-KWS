# REPORT - MeshTalk voice SOS model

## Step 0 - environment: GO

- `python -m kws.check_c` -> **15 checks, 0 failed**
- Cloud container: 4 vCPU Intel Xeon @ 2.10 GHz, 15 GB RAM, no GPU; Python 3.13.16, torch 2.14.1+cpu, numpy 2.5.4, scipy 1.18.1, onnxruntime 1.31.0, gcc 13.3
- Download hosts (through the session proxy):

| host | reachable | used for |
|---|---|---|
| huggingface.co | yes (~68 MB/s) | LibriSpeech + word alignments (`gilkeyio/librispeech-alignments`), ESC-50 mirror (`ashraq/esc50`) |
| storage.googleapis.com | yes (~25-70 MB/s) | MSWC English (`public-datasets-mswc`), Speech Commands v2 (`download.tensorflow.org` bucket) |
| download.tensorflow.org | **no** (TLS name mismatch) | -> same file via storage.googleapis.com/download.tensorflow.org/... |
| www.openslr.org | yes | MUSAN |
| github.com releases (`/releases/download/...`) | yes | Kokoro model files |
| github.com pages, codeload.github.com, api.github.com | **no** (403) | ESC-50 GitHub zip -> HuggingFace mirror instead |

## Step 1 - public data: GO

**Real "help": 5759 clips from ~3650 speakers** (MSWC 4433 clips / 3026 speakers + LibriSpeech 1326 / 623) -> target >= 1000 / >= 300 met.

| class | train | val | test | sources |
|---|---|---|---|---|
| help | 7050 | 958 | 841 | MSWC 3501/506/416, LibriSpeech 1049/152/125, Kokoro 2500/300/300 |
| help me | 2579 | 309 | 310 | LibriSpeech real 79/9/10 + joined same-speaker help+me (496 train speakers, on the fly), Kokoro 2500/300/300 |
| sos | 3000 | 300 | 300 | **Kokoro only** (no public real "SOS" recordings) |
| _other | 182 493 | 23 864 | 22 292 | MSWC look-alikes 16 193 + random words 94 530, LibriSpeech words / "me" / 1.5 s speech chunks 54 839, Speech Commands 11 931, Kokoro negatives 5000 |

- Split by speaker hash 80/10/10 (Kokoro: 22 train voices, 2 val, 4 test voices held out).
- Speaker head (real only): 4302 train / 555 val / 528 test speakers, 150k / 20k / 18k clips.
- Noise: train 17.3 h (MUSAN noise 4.9 h + music 10.1 h, ESC-50 folds 1-4 2.2 h, SC background), test 3.7 h (MUSAN 3.2 h, ESC-50 fold 5 0.56 h).
- False-alarm speech: LibriSpeech dev 10.5 h (threshold choice), test 10.7 h (final numbers).
- **Findings that changed the plan:**
  1. MSWC has **no "me" and no "sos"** (only words of >= 3 letters). "help me" therefore comes from LibriSpeech (98 real + "help"+"me" of the same speaker joined with 0-80 ms gap) plus Kokoro; SOS is synthetic only.
  2. Kokoro reads "SOS"/"SOS!" as "sauce" -> only spelled forms ("S. O. S.", "S-O-S!", ...) are used.
  3. A spelled "S. O. S." lasts 0.6-1.8 s (median 0.95 s); the default window T = 61 (1.22 s) cannot hold a slow one. **T = 81 (1.62 s)** is used (CONTRACT allows <= 101): ~6.9 M multiply-adds per run instead of ~5.2 M (+33 %) - please check the run time on the device (runs every 240 ms).
- Licences: MSWC CC-BY 4.0, LibriSpeech CC-BY 4.0, Speech Commands CC-BY 4.0, MUSAN CC-BY 4.0 (mixed per file), ESC-50 CC-BY-NC 3.0, Kokoro Apache-2.0. Checksums of every download: `data/raw/MANIFEST.json` (MD5 from GCS / SHA-256 from HuggingFace verified).

## Step 2 - keyword head: **NO-GO** (stopped here, see "Decisions needed")

Model `models/kw.pt` (v2, epoch 38 of 50, 132 min on 4 CPU cores). Numbers on the **test** split
(held-out speakers / Kokoro voices; 10.7 h LibriSpeech test speech, half with added noise, + 3.7 h test noise),
measured frame by frame with `kws.detect.Detector`. Hit = any keyword class fires within the keyword
(+1.5 s); half the streams clean (mic floor only), half noisy (SNR 5-20 dB).

| thr | help | help me | SOS | false hits / h | false SOS (14.5 h) |
|---|---|---|---|---|---|
| 0.50 | 86.4 % | 93.3 % | 90.3 % | 103 | 172 |
| 0.70 | 51.4 % | 62.2 % | 70.5 % | 20.6 | 8 |
| 0.80 | 34.9 % | 47.1 % | 56.8 % | 6.2 | 1 |
| 0.90 | 10.9 % | 27.8 % | 41.8 % | **0.83** | **0** |
| target | >= 90 % | >= 85 % | >= 80 % | <= 1 | 0 |

At the threshold that meets the false-alarm target (0.90) the hit rates are far below target -> NO-GO.
Curves: `docs/kw_hit_vs_fa_test.png` (dev: `docs/kw_hit_vs_fa_val.png`), JSON: `models/eval_kw_*.json`.

**Finding 1 - the -55 dBFS gate in `kws_detect.c` costs about half the hits in a quiet room.**
The CNN only runs when something louder than -55 dBFS happened in the *last 240 ms*, and the
smoothing history is cleared when it does not. A short word in a quiet room therefore gets only
1-2 runs, and the first one sees the word only partly; the 3-run average rarely passes a high
threshold. Same model, quiet condition, thr 0.70 (300 val clips): **41 % with the gate, 90 % without**.
Proposed C change (not applied, `detect.py` / firmware unchanged; Python model of it:
`kws/evaluate.py: WindowGateDetector`, option `20_eval.py --detector window_gate`):

```c
/* kws_detect.h, kws_det_t: replace  float loud;  by */
    int      quiet;                 /* frames since the last frame louder than the gate */
/* kws_det_reset():  d->quiet = 1 << 20;   (instead of d->loud = -120.0f) */
/* kws_det_frame(): */
    d->quiet = (dbfs >= KWS_GATE_DBFS) ? 0 : d->quiet + 1;   /* instead of: if (dbfs > d->loud) ... */
    ...
    if (d->quiet >= m->T) {           /* nothing loud in the whole window (was: d->loud < KWS_GATE_DBFS) */
        d->n_hist = d->hist_pos = 0;
        return 0;
    }
```
Silence still never runs the CNN (power saving kept); a word keeps the CNN running while it is in the window.
With this change (test split):

| thr | help | help me | SOS | false hits / h | false SOS (14.5 h) |
|---|---|---|---|---|---|
| 0.70 | 73.4 % | 89.4 % | 95.5 % | 20.3 | 9 |
| 0.80 | 54.1 % | 78.0 % | 90.8 % | 6.0 | 1 |
| 0.85 | 40.2 % | 67.7 % | 86.0 % | 2.6 | 0 |
| 0.90 | 20.5 % | 56.9 % | 81.5 % | 0.90 | 0 |

**Finding 2 - single "help" is the weak class.** A 0.3 s word inside running speech is easily
confused (real LibriSpeech "help" 8-16 % at thr 0.90). SOS numbers are on synthetic voices only and
are optimistic for real people.

What was tried: v1 (40 epochs, 15.8k speech chunks) -> v2 (+train-other-500: 45k speech chunks,
_other share 0.5, keywords also placed at the window end, lr 0.003 -> 0.002): false hits / h at
thr 0.85 went from 35 to 2.6 (dev), hit rates similar. v1 is kept in `models/v1/`.

## Step 3 - speaker head (preliminary): **NO-GO for "Only My Voice"**

Short runs (4 epochs x 20k windows, `models/prelim_spk/`), v1 trunk, device enrolment protocol
(5 takes of 2 s, best window per class, per-class template, cosine), validation speakers:

| speaker head | EER text-independent (300 spk) | EER text-dependent (67 spk, help/help me/me) |
|---|---|---|
| frozen keyword trunk + AAM | 36.1 % | - |
| multitask (trunk trained too) | 34.1 % (noisy 35.2 %, shouted 34.1 %) | 33.8 % (noisy 38.5 %) |

Target EER <= 10 % -> clearly not reached; the 48-value pooled trunk of this small CNN carries
little speaker information (all embeddings lie within cos 0.99 of each other). **Recommendation:
keep "anyone triggers"; export with `spk_dim` kept but "Only My Voice" off**, or `spk_dim = 0`.

## Decisions needed (from the owner)

1. Firmware gate change above - yes / no? (largest single gain, no retraining needed.)
2. Accept lower targets for this model, e.g. thr 0.85-0.90 (~1-3 false hits / h, 0 false SOS
   in 14.5 h) with the 3-hits-in-20 s rule doing the rest - or keep training (next steps I would try:
   larger channels within CONTRACT limits c1 24 / c2 48 / c3 64, hard-negative mining on running
   speech, more epochs; and, most effective, the owner's own recordings in `data/own/`).
3. Speaker head: confirm "Only My Voice" stays disabled.
Steps 4 (export) has not been done yet, as agreed for a NO-GO.

---

# Round 2 (firmware v1.4.1)

## Step 0 - firmware v1.4.1 synced: GO
- `c_ref/`, `kws/detect.py`, `kws/export.py`, `CONTRACT.md` replaced from `update_v1.4.1/`, `CONTRACT_SPK.md` added
  (no local changes existed in these files, nothing to merge).
- `python -m kws.check_c` -> **19 checks, 0 failed**.
- `WindowGateDetector` removed; all evaluation uses `kws.detect.Detector` (window gate + `cls_off`).

## Phase A - keyword model "usable v1": exported, targets met at the border

**`out/kws_model.c` + `out/kws_model_test.c` (kws-20261023-r3), `python -m kws.check_c out` -> 20 checks, 0 failed.**
Model: T 81, c1/c2/c3/hid 16/32/48/64, 4 classes, spk_dim 0 -> **6.92 M multiply-adds per run, weights 22.9 KB,
work memory 22 440 floats = 88 KB** (+ 25 KB ring/window of `kws_det_t`) - inside the budget (64 KB / 160 KB / 15 M).
Settings: `thr = {0.65, 0.55, 0.50}` (low / **medium** / high), `cls_off: help -0.10, help me 0, sos +0.05`, 3 hits / 20 s.

Test split, measured with `kws.detect.Detector` on the int8 model (`models/best_int8sim.pt` = the C arithmetic,
max difference to `export.reference` 8.5e-7). Trigger = EV_SOS when one person calls 6 times in 20 s
(mean of clean and noisy SNR 5-20 dB; 200 scenarios per cell). False alarms: 38.7 h held-out running speech
(LibriSpeech test + test-split speakers, half with noise) + 1.35 h test noise.

| sensitivity | real help/help me mixed | Kokoro help/help me/SOS mixed | Kokoro SOS only | **real "help" only** | false hits / h | **false SOS / 24 h** |
|---|---|---|---|---|---|---|
| low 0.65 | 87.0 % | 97.5 % | 94.5 % | 81.0 % | 2.5 | **0** (0 in 40 h) |
| **medium 0.55** | **92.5 %** | **99.5 %** | **97.0 %** | **88.0 %** | 6.9 | **1.2** (2 in 40 h) |
| high 0.50 | 95.0 % | 99.5 % | 98.5 % | 90.5 % | 10.3 | 2.4 (4 in 40 h) |
| target | >= 90 % | >= 90 % | | (report) | | <= 1 |

Medium in detail (clean / noisy, %):

| calls in 20 s | real help only | real mixed | Kokoro mixed | Kokoro help only | Kokoro SOS only |
|---|---|---|---|---|---|
| 4 | 89 / 77 | 95 / 80 | 99 / 97 | 100 / 92 | 97 / 89 |
| 6 | 95 / 81 | 94 / 91 | 100 / 99 | 100 / 97 | 99 / 95 |
| 8 | 94 / 89 | 97 / 99 | 100 / 100 | 100 / 97 | 100 / 98 |

- **Trigger target met** (mixed calls 92.5 % real, 99.5 % synthetic). Real "help" only: 88 % (95 % clean, 81 % in noise).
- **False SOS at the border:** dev 1 in 32.9 h (0.73/24 h), test 2 in 40 h (1.2/24 h), together 3 in 72.9 h = **0.99 / 24 h**.
  On running speech alone: 0.62 / 24 h (1 in 38.7 h). The other test event comes from noise: one MUSAN file
  (`noise-free-sound-0155.wav`, 72 s, probably voices in it) gives 31 high windows; the test noise is only 1.35 h, so one file decides.
- False single hits come from noise (13/h on test noise) and from scattered words with h/p sounds in speech
  ("hope", "himself", "helpless", "doctor"...), no single dominant word (`models/eval_sys_test.json` fa_top).
- Sensitivity "low" meets the false-alarm target with margin (0 false SOS, 2.5 hits/h) but real mixed calls drop to 87 %.

How (all choices on dev, test only reported): round 1 (`kw_r1.pt`, round-1 v2) -> **hard-negative mining**
(`12_mine.py`: round-1 model over 242 h of training speakers' speech: 20 000 windows where it fired + 3206 in training noise)
-> round 2 (`kw_r2.pt`, fine-tuned 25 epochs) -> mining again with round 2 (new shards, threshold 0.4: only 7946 + 986 left)
-> round 3 (`kw_r3.pt`, 20 epochs) -> `cls_off` / `thr` sweep on dev (160 settings). Dev at the same trigger rate:
false hits/h r2 9-10 -> r3 4.7-5.4. False-alarm sets enlarged for this: dev 30.5 h + test 38.7 h of held-out speech.

Next most effective steps: (1) the owner's own "help / help me / SOS" recordings with the device microphone (real SOS
is still synthetic only, noisy real "help" is the weak case); (2) more real noise with voices as hard negatives
(the device's own environment recordings would be best); (3) once `kwsbench` is measured, c1/c2 24/48 (14.5 M MAC) is the allowed next size.

## Phase B - separate speaker model: delivered, **system target NOT met** -> keep "anyone triggers"

**Delivered:** `out/spk_engine.c` + `out/spk_model.h` + `out/spk_model.c` (-> firmware `Ai/`), `out/spk_model_test.c` +
`out/spk_test.c` (C test, -> `Tests/`). `python -m spk.check_c out` -> **10 checks, 0 failed, C vs Python max difference 1.9e-7**.
Compiled and linked together with the keyword C code (`-Wall -Wextra`): no warnings, `test_kws` still 20/0.

Model (`spk/model.py` SpkNet, `models/spk_best.pt`): 81 frames (= the keyword window handed to `verify()`), 4 x Conv1d
(128/128/128/256 channels, kernels 5/3/3/1, dilations 1/2/3/1) + mean/std pooling + Linear -> 128-dim vector.
**12.8 M multiply-adds per check, weights 229 KB (int8, one scale per channel), work memory 45 352 floats = 177 KB**
(budget 512 KB / 512 KB / ~300 ms). `spk_threshold() = 0.60`, `spk_enrol_takes() = 10`.
Training: real voices only - 5224 speakers (MSWC ~4000 + LibriSpeech ~1200 training-split speakers, max 20 clips each),
AAM-softmax (m 0.2, s 30), noise / reverb / gain / **shouting** augmentation, 40 epochs (130 min).
Not available here: VoxCeleb (needs credentials), Common Voice (HuggingFace copy needs a login), VCTK (one 11 GB zip, no disk).

Per call (device enrolment protocol: N takes, per-class templates, cosine; test split, 400 speakers, ~3900 target and
~116 000 impostor trials per row):

| enrolment takes | EER clean | EER noisy | EER shouted | EER, test word = help / help me (clean / noisy / shouted) | at 0.60: false reject / false accept (clean) |
|---|---|---|---|---|---|
| 5 | 8.6 % | 11.3 % | 9.1 % | 8.1 / 11.1 / 7.8 % | 37.6 % / 1.0 % |
| 10 | **6.8 %** | **9.8 %** | **7.2 %** | 7.3 / 9.0 / 7.5 % | 22.0 % / 1.6 % |

System (keyword model v1 medium + 3 hits / 20 s + Only My Voice at 0.60, `kws.detect.Detector(only_me=True, verify=...)`,
380 test speakers who have real "help" recordings, 10 enrolment takes, 200 scenarios per cell):

| | clean | noisy | shouted | target |
|---|---|---|---|---|
| owner calls 6 times -> SOS | 75.5 % | 58.5 % | 78.0 % | >= 90 % |
| another person calls 6 times -> SOS | 2.5 % | 1.5 % | 2.0 % | <= 3 % |

Trade-off on val (`docs/spk_eer_val.png`, right): threshold 0.50 -> owner 90.5 / 77.5 / 89.5 %, other 9 / 5.5 / 10.5 %;
threshold 0.40 -> owner 93.5 / 88.5 / 94 %, other 19 / 18.5 / 19.5 %. No threshold meets both targets.

Why the per-call EER (6.8 %) does not give the system numbers the CONTRACT_SPK guide expects: errors are **per pair of
voices, not per call** - a person whose voice is close to the owner's passes on all 6 calls, and an owner whose
enrolment does not match is rejected on all 6. Also pessimistic here: the public data has almost no person saying
"help" several times, so enrolment uses 10 OTHER words of the speaker (different recordings, often different session),
and the 6 calls repeat one "help" recording; on the device the owner enrols the real keywords with the same microphone.

**Recommendation:** keep "anyone triggers" (as planned). The speaker model and engine are ready; with 10 of the owner's
own takes recorded on the device (`data/own/`), threshold and possibly a short fine-tune can be re-checked with
`scripts/41_eval_spk.py` - this is the most effective next step.

---

# Phase B2 (2026-11) - speaker model warm-start fine-tuning with distillation: **NOT delivered** (not clearly better)

`out/spk_model.c` stays **spk-20261023 (id 0x4A5BDC6D)** - no new speaker files, no re-enrolment needed.

Test: 568 test speakers (140 female; gender predicted for 244, see below), 10 enrolment takes (close, quiet, normal voice),
6 test takes each (other recordings; LibriSpeech: other chapters = other session first) in 4 device conditions,
15 same-gender + 15 cross-gender impostor templates per take; identical audio for both models (fixed seeds).
Score = cosine to the template of the take's class (what the firmware does). Threshold "Normal" 0.52.

| condition | EER old -> new | FRR @0.52 old -> new | FAR @0.52 old -> new (same / cross gender) |
|---|---|---|---|
| clean (0.3-0.5 m) | 7.4 % -> 7.6 % | 12.8 % -> 20.5 % | 6.7 / 2.5 % -> 3.4 / 1.2 % |
| far (0.8-1 m, RT60 0.4-0.8 s, quieter) | 8.6 % -> 8.4 % | 16.6 % -> 26.6 % | 6.3 / 2.3 % -> 3.0 / 1.1 % |
| shouted (clipping) | 10.2 % -> 9.6 % | 20.0 % -> 30.1 % | 7.9 / 3.1 % -> 3.7 / 1.5 % |
| noisy (SNR 5-15 dB) | 10.8 % -> 11.1 % | 26.3 % -> 38.4 % | 6.2 / 2.1 % -> 2.6 / 0.9 % |
| **mean EER** | **9.25 % -> 9.17 %** | | |

Score distributions (5 / 25 / 50 / 75 / 95 %, old model; `docs/spk_compare_test.png`): owner clean 0.45 / 0.59 / 0.67 / 0.74 / 0.82,
far 0.42 / 0.56 / 0.65 / 0.72 / 0.80, shouted 0.40 / 0.54 / 0.62 / 0.69 / 0.78; impostor same gender 95 % at 0.54,
cross gender 95 % at 0.46 (the device numbers 0.65-0.84 own / 0.17-0.45 female imitation fit this).
Same-gender impostors are about twice as hard as cross-gender (EER 9.0 vs 5.4 % clean).

**Why not delivered:** the mean EER improves by 1 % (criterion: >= 15 %). The fine-tuned model only shifts all scores
down by ~0.04 (FAR halves, FRR almost doubles at 0.52) - that is a threshold change, not a better model.
Three attempts, all warm-started from spk-20261023 with the first layer frozen, chosen on val (150 speakers), test only reported:

| attempt | losses | lr | val mean EER (start 9.33 %) | test mean EER (old 9.25 %) |
|---|---|---|---|---|
| A | AAM 0.5 + cosine distillation 1.0 | 3e-4 | 8.93 % | 8.90 % |
| B | + relational distillation 10 + consistency (device view = enrolment view) 1.0, distillation 0.5 | 3e-4 | 9.30 % | - |
| C | as B, all BatchNorm statistics frozen | 1e-4 | 9.05 % | 9.17 % |

The student (fixed: 220 k parameters, 1.62 s input) cannot follow the teacher: after the 128 -> 192 projection
its cosine with ECAPA stays at ~0.37 (distillation loss 0.63), and the old model was already trained with noise /
reverb / shouting, so device robustness gains are small. More capacity or longer input would be needed (CONTRACT unchanged -> not done).

**What helps without a new model (firmware scoring, same spk-20261023):** compare with the template of ALL 10 takes
instead of the class template (each class has only 3-4 takes):

| condition | EER class -> all | FRR @0.52 class -> all | FAR @0.52 class -> all (same / cross) |
|---|---|---|---|
| clean | 7.4 -> **5.4 %** | 12.8 -> **4.6 %** | 6.7 / 2.5 -> 8.8 / 3.3 % |
| far | 8.6 -> **6.1 %** | 16.6 -> **6.7 %** | 6.3 / 2.3 -> 8.3 / 2.9 % |
| shouted | 10.2 -> **8.4 %** | 20.0 -> **9.9 %** | 7.9 / 3.1 -> 10.7 / 4.2 % |
| noisy | 10.8 -> **9.0 %** | 26.3 -> **15.4 %** | 6.2 / 2.1 -> 8.2 / 2.8 % |

(Caveat: the public data rarely has one person repeating the same keyword, so the class template here does not get
the "same word" advantage it has on the device. The large SOS spread on the device (0.41-0.77) points the same way:
few SOS takes -> unstable class template.) Suggested firmware option: score = max(class template, all template)
(here EER 5.5 / 6.2 / 8.6 / 9.3 %), or all template for SOS; at 0.52 keep an eye on same-gender false accepts (~+2 points).
Enrolling the 10 takes on different days / distances also makes the all-template more stable.

Data added for this (no login needed): VCTK 110 speakers (4400 utterances, read with HTTP ranges from a 10.7 GB zip),
People's Speech 10 439 segments (8.7 h, no speaker labels, distillation only); teacher SpeechBrain `spkrec-ecapa-voxceleb`
(Apache-2.0) on 33 145 clean segments. Gender: LibriSpeech SPEAKERS.TXT + MSWC labels; others predicted by a logistic
regression on teacher vectors (speaker-level accuracy 95.8 % on LibriSpeech; 83.5 % agreement with MSWC self-labels).
Gender-balanced sampling (50 / 50). Device augmentation: `spk/device.py` (INMP441 high-pass + shelves, 0.3-1 m with
RT60 0.2-0.8 s, -45...-3 dBFS with clipping, shouting, noise). LibriTTS not used (same speakers as LibriSpeech).
VoxCeleb / Common Voice not used (gated) - an HF token would add ~7000 in-the-wild speakers, but the student capacity is the
main limit, so I would not expect a large gain from data alone.

---

# Phase B3 (2026-11) - larger speaker model from scratch: **stopped after the trial** (not promising)

Structure: Conv1d 192/192/192/192/384, kernels 5/3/3/3/1, dilations 1/2/3/4/1, mean + std pooling -> 128, L2 (36.1 M MAC).
Trial = 10 epochs (1/4 of the planned 40), ECAPA distillation + AAM, B2 data / device chain / teacher vectors reused.

| | cosine student vs teacher | val mean EER (score = max(class, all), clean / far / shout / noisy) |
|---|---|---|
| spk-20261023 (old, 12.8 M MAC) | 0.37 (B2 fine-tuning) | **7.38 %** (5.7 / 6.3 / 8.1 / 9.4) |
| new, trial after 10 epochs | **0.30** | **15.0 %** (12.9 / 13.2 / 15.7 / 18.2) |

Decision: no full training, no engine. Note: with the firmware's max(class, all) scoring the OLD model is already at 7.4 %
(< the 8 % gate); a 15 % gain needs ~6.3 %. The trial converges slowly (-0.4 points/epoch at the end) and its teacher
similarity is below the old model's, so 4x more training is not expected to reach that.

# Phase C (2026-11) - keyword model r4: fewer false SOS from coughs and talk - **in progress**

Plan: `PLAN_R4.md`; parameters and reasons: `TRAINING.md` section 8. Code ready and smoke-tested
(`tools/smoke_r4.py` 14/14 PASS on synthetic audio; firmware detector check 0 failed; PyTorch vs
int8 C engine agree). Steps run on the owner's laptop (WSL Ubuntu); results are added below as
each step finishes (`bash scripts/run_r4.sh N`).

| step | result |
|---|---|
| 0 environment | **PASS**: `.venv` packages complete (torch 2.14.1, CPU); gcc 13.3 already installed (no apt needed); `kws.check_c` 19 checks **0 failed**; firmware v1.6.7 detector build + check **0 failed** (C front end vs Python max diff 9.1e-6; help/sos call logic matches v1.6.7); `tools/smoke_r4.py` **14/14 PASS**; disk **916 GB** free. |
| 1 own recordings (counts, r3 check) | Imported **240 takes** (help 80 = train 61 / val 6 / test 13; help me 80 = 55 / 5 / 20; sos 80 = 54 / 6 / 20; styles FAR/LOUD/NORMAL 20/20/40 each; speakers A 180, B 60; 37 clipped takes kept). r3 with the firmware logic (medium): **heard as the right keyword** help 95 % FAR / 35 % LOUD / 60 % NORMAL, help me 65 / 30 / 37.5, sos 85 / 75 / 45 %. Verdicts: help 50 ok / 12 missed / 18 two_calls; help me 34 ok / 33 heard as plain "help" / 13 two_calls; sos 50 ok / 30 missed. The 60 takes listed as "may be wrong" are r3's known confusion (help me -> help, two detections in one take), not obviously bad recordings; **nothing deleted**. |
| 2-3 Step-1 data, old hard negatives | **Step-1 data (verified, step 2 not re-run)**: `data/cache/counts.json` matches TRAINING.md §2 exactly - help 8263/1080/1012, help me 2701/328/334, sos 3000/300/300, _other 220 310/28 377/27 469 (train/val/test); noise 17.3 h train / 3.7 h test; false-alarm speech dev 10.5 h + test 10.7 h. `data/raw` 12.8 GB present (mswc 1.2, librispeech 6.3, musan 2.0, speech_commands 0.94, esc50 0.74, kokoro 0.34 GB). `MANIFEST.json` has 141 checksummed entries; the 137 entries not on disk are archives streamed and deleted by design (`note: streamed; only the cut clips are kept`), not corruption. **Old hard negatives:** attempt 1 (13 Nov 21:53) reached shard 39/40 then the host blue-screened (`0x1A MEMORY_MANAGEMENT`, `nt!MiResolveTransitionFault`); attempt 2 (23:14) stopped at shard 16/40 with a FLAC decoder error in a shard whose SHA-256 had passed right after download, but whose on-disk copy no longer matched it (fresh re-download decodes fine) - host-side memory/storage corruption. WinDbg: three dumps (Jun 5 `0x3B` nt, Nov 3 `0x1E` tcpip, Nov 13 `0x1A` nt) are all in Microsoft core modules, no third-party driver; Windows memory diagnostic is planned. No code changed. **Step 3 then completed on the 3rd attempt (14 Nov):** `mine_stats_r2.json` 70 385 utts / 241.7 h scanned, 20 000 shard + 3 206 noise hard negatives = **23 206** (68 min); `mine_stats_r3.json` 70 785 utts / 242.1 h, 7 946 + 986 = **8 932** (65 min) - both exactly the r2/r3 counts in TRAINING.md §2. Extra false-alarm speech: test 27.9 h / dev 20.0 h. |
| 4 new data (counts, licences) | |
| 5 r3 "before" (test) | |
| 6 r4 hard negatives | |
| 7 training | |
| 8 thresholds (val) | |
| 9 r4 "after" (test) | |
| 10 export, int8 | |
