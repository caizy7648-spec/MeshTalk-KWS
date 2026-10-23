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
