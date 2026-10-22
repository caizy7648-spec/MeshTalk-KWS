# PLAN - training the MeshTalk voice SOS model (Claude Code cloud session)

Goal: a small model that hears **"Help"**, **"Help me"** and **"SOS"** (English)
from the device microphone, plus a speaker vector so the owner's enrolled
voice can be recognised ("Only My Voice"). Read `CONTRACT.md` first.

Data policy (owner's choice): **public data first**. The only real voice the
owner can add is his own (later, recorded with a separate recorder firmware
over the serial port, into `data/own/<label>/`). No other people are
recorded (no ethics approval needed).

Work in steps; at the end of each step write what was found into
`REPORT.md` (short, numbers first) and commit. Stop and report instead of
guessing when a step's go / no-go fails.

## Step 0 - environment (go: check_c passes)
- `pip install -r requirements.txt` (CPU torch is enough).
- `python -m kws.check_c` -> `15 checks, 0 failed`.
- Note in REPORT.md: CPU / RAM, which download hosts are reachable
  (huggingface.co, download.tensorflow.org, openslr.org, github.com releases).

## Step 1 - public data (go: >= 1000 real "help" clips from >= 300 speakers)
| need | source (licence) | notes |
|---|---|---|
| "help", real voices | **MSWC** Multilingual Spoken Words Corpus, English (CC-BY 4.0), HuggingFace `MLCommons/ml_spoken_words` | only the words `help` and `me` are needed; keep the speaker id (for step 3) |
| "help me", real | MSWC `help` + `me` of the **same speaker**, joined with 0-80 ms gap | |
| "SOS" | MSWC `sos` if it exists; otherwise synthetic only | |
| synthetic all three | **Kokoro TTS** (Apache-2.0; `kokoro-onnx`, model files from GitHub releases of thewh1teagle/kokoro-onnx) - all English voices, speed 0.8-1.3, also blended voices; calm and urgent spellings ("Help!", "HELP!", "Help me!", "S. O. S.", "SOS!") | synthetic voices are used **only** for the keyword head, never for the speaker head |
| `_other` words | Google Speech Commands v2 (CC-BY 4.0) + random other MSWC words, **especially look-alikes**: hello, helped, health, held, yelp, self, hell, ask, us, also, oh yes ... | |
| `_other` running speech / noise | LibriSpeech dev/test (CC-BY 4.0), ESC-50 (CC-BY-NC, GitHub), MUSAN if reachable | also for false alarms per hour |

If a host is blocked: write the exact URL / file / size needed into REPORT.md
and stop - the owner downloads it on his PC and adds it.

## Step 2 - keyword head (go: see targets)
- Clips -> 16 kHz -> random placement inside a T=61 frame (1.22 s) window,
  keyword fully inside; augment: gain -20..+6 dB, noise SNR 0-20 dB,
  reverb (synthetic RIR), speed 0.9-1.1, pitch +-2 semitones, shouting
  (gain + soft clipping), mic high-pass ~80 Hz.
- Features only with `kws.features.logmel`. Set `in_mean`/`in_std` from the
  training features. Class-balanced cross-entropy, a few epochs on CPU.
- Evaluate with `kws.detect.Detector` (the device logic) on held-out speakers:
  - hit rate per keyword at each sensitivity;
  - false hits per hour and **false SOS (3 hits / 20 s) per hour** on >= 2 h
    of running speech + noise.
- Targets (medium): hit rate "help" >= 90 %, "help me" >= 85 %, "SOS" >= 80 %
  (SOS may be lower with synthetic-only data - report it), false hits <= 1 / h,
  false SOS 0 in the test hours.

## Step 3 - speaker head (go / no-go for "Only My Voice")
- Train `spk` (trunk frozen or multi-task) on **real** MSWC clips grouped by
  speaker (AAM-softmax or GE2E), not on synthetic voices.
- Device protocol: enrol 5 clips of a speaker, test other clips of the same
  speaker vs. other speakers, cosine to the per-class / overall template.
- Report EER, and false accept / false reject at the chosen `spk_thr`;
  also false reject for "shouted" augmented clips of the same speaker.
- Recommendation: EER <= 10 % -> Only My Voice usable (still off by default);
  worse -> say so clearly; the owner then uses "anyone triggers".

## Step 4 - export and hand back
- Choose `thr` (low / medium / high) and `spk_thr` from the curves.
- `python -m kws.export ...` and `python -m kws.check_c out` (0 failed).
- REPORT.md: data used (counts, licences), the numbers of steps 2-3, the
  chosen thresholds, known weaknesses, what the owner's own recordings
  should cover next.
- Hand back `out/kws_model.c`, `out/kws_model_test.c`, `REPORT.md`, and
  everything in "Training code and parameters" below.

## Training code and parameters (required deliverable)
The owner must be able to read, explain (viva) and re-run the training.
Everything is committed to this repository:

    configs/train.yaml      EVERY parameter, the code reads them from here (no magic numbers in code)
    scripts/01_download.py  public datasets -> data/raw (URLs, checksums, licences)
    scripts/02_prepare.py   cut / resample / label / split by speaker -> data/cache
    scripts/03_tts.py       synthetic "help" / "help me" / "SOS" (voices, speeds, texts)
    kws/augment.py          augmentation (gain, noise SNR, reverb, speed, pitch, shouting, high-pass)
    kws/dataset.py          windows of T frames with kws.features.logmel
    scripts/10_train_kw.py  keyword head
    scripts/11_train_spk.py speaker head
    scripts/20_eval.py      hit rate / false alarms per hour with kws.detect, speaker EER
    scripts/30_export.py    thresholds -> kws.export -> kws.check_c
    models/best.pt          the trained weights (commit it, it is small)
    docs/*.png              training curves, confusion matrix, hit-rate vs false-alarm curve, EER curve
    TRAINING.md             in Chinese: every parameter with its value and WHY; the exact
                            commands to reproduce from zero; environment (Python / torch
                            versions, CPU, time per step); data counts per class and source

`configs/train.yaml` must at least contain: data sources and clip counts
per class, train / validation / test split by speaker, window T, network
sizes (c1, c2, c3, hid, spk_dim), every augmentation parameter and its
probability, optimiser, learning rate and schedule, weight decay, batch
size, epochs, loss (and class weights / margins), random seeds, chosen
thresholds `thr` and `spk_thr`. Fix all random seeds so a re-run gives the
same model.
