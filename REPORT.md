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
