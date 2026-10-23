# CONTRACT_SPK - the separate speaker (voice print) model, phase 2

Goal ("Only My Voice"): the owner records at most 10 short takes of his call
words on the device; afterwards only HIS calls for help trigger Voice SOS.

## How the firmware uses it (fixed, already implemented)
- Interface: `Ai/spk_engine.h` (copy in `c_ref/`). Phase 2 delivers
  **`spk_engine.c` + `spk_model.c`** implementing exactly these functions
  (the current `spk_engine.c` is a stub that says "no model").
- Input: log-mel frames of the SAME front-end (`kws/features.py`, 40 bands,
  20 ms hop), `spk_frames()` frames (<= 101, i.e. <= 2.02 s) centred on the
  keyword window that triggered. Output: `spk_dim()` floats (<= 256), length 1.
- It runs ONCE per keyword hit (not continuously). Owner-side decision rule
  (already in `kws_detect.c` / `kws/detect.py`, `verify` hook): with Only My
  Voice on, a hit counts only if cosine(vector, enrolled template of that
  keyword class, else of all classes) >= `spk_threshold()`; still
  `hits` hits within `hits_ms` for an SOS.
- Enrolment: `spk_enrol_takes()` takes (3..10, owner prefers <= 10) of 2.2 s,
  phrases in turn "Help", "Help me", "SOS", ...; the template per class is
  the sum of the take vectors (normalised by the cosine).
- Plain C99, no libraries, int8 weights + float scales (like `kws_engine.c`).
- **Resource budget** (the owner wants room left for later features):
  weights (flash) <= 512 KB, work memory `spk_work()` (PSRAM) <= 512 KB,
  no internal RAM, <= ~300 ms per check on the ESP32-S3 at 240 MHz (it runs
  only after a keyword hit; the 5 s cancel countdown hides it). Choose the
  smallest model that meets the targets and report its size / MACs. Estimate the multiply-adds
  and say so in the report; the owner measures with `kwsbench` (prints the
  speaker check time when a model is built in).
- `spk_id()` must change with every export (stored voices of another id are
  discarded). Deliver also `spk_model_test.c` (input + expected vector) and
  a C test like `test_kws.c` showing the C engine == Python (max diff < 1e-4).

## Training and evaluation
- Real voices only (no TTS) for the speaker training: LibriSpeech (~2.4k
  speakers, already downloaded), MSWC "help" clips (text-dependent), and
  VoxCeleb1/2 or Common Voice if reachable. Speaker-disjoint train/val/test.
- Text-dependent focus: the test is the device protocol - enrol N takes of
  help / help me / (SOS where real data exists) per speaker, test other takes
  of the same speaker vs. other speakers. Augment: device-like noise, reverb,
  gain, **shouting** (owner's calls in an emergency are louder / strained).
- Report per-call EER, FAR / FRR at the chosen `spk_threshold`, also for
  shouted and noisy target takes; and the SYSTEM numbers with the keyword
  model and the 3-hits-in-20-s rule: owner calling 6 times -> trigger rate
  (target >= 90 %), another person calling 6 times -> trigger rate
  (target <= 3 %).
- Rough guide: a per-call EER around 10-15 % already gives about those
  system numbers. If the best model stays clearly worse, say so - the owner
  then keeps "anyone triggers" and records his own voice later.
