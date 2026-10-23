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
