# CONTRACT - what the firmware expects from a trained model

The firmware (MeshTalk_ESP32 v1.4) already contains the complete runtime:
`Ai/kws_frontend.c` (features), `Ai/kws_engine.c` (CNN), `Ai/kws_detect.c`
(decisions), `Svc/kws.c` (microphone task, enrolment), the UI pages. The
only file that changes after training is **`Ai/kws_model.c`**. Copies of the
C code are in `c_ref/`; `python -m kws.check_c out/` compiles them with a
new model and must report `0 failed`.

## 1. Audio and features (fixed - `kws/features.py` = `Ai/kws_frontend.c`)
- 16 kHz mono, int16. INMP441 microphone, no AGC.
- Frame 480 samples (30 ms), hop 320 (20 ms), frame mean removed, periodic Hann,
  512-point FFT, power spectrum, 40 HTK-mel triangles 60-7600 Hz (peak 1),
  `ln(energy + 1e-6)`.
- **Always** compute features with `kws.features.logmel()`; never librosa /
  torchaudio mel (they differ).

## 2. Network (fixed layer types - `kws/model.py` = `Ai/kws_engine.c`)
`KwsNet(ncls, T=61, c1=16, c2=32, c3=48, hid=64, spk_dim=32)`:
input window T frames x 40 bands (T=61 is 1.22 s) -> per-band normalisation
(`in_mean`, `in_std` buffers: set them from the training features!) ->
3 x [conv3x3 + BN + ReLU] (+ max-pool 2 after the first two) -> global average
`g` -> keyword head `fc1 -> ReLU -> fc2` (softmax) and speaker head
`spk(g)` L2-normalised.
Allowed changes: the sizes only, within T <= 101, c1 <= 32, c2 <= 64,
c3 <= 96, hid <= 128, ncls <= 8, spk_dim <= 64 (spk_dim = 0: no speaker
head; then "Only My Voice" is disabled on the device). Budget: about 5 M
multiply-adds per run (default sizes), the device runs it every 240 ms.

## 3. Classes
`--labels` gives the class names in output order, separated by `|`.
Names starting with `_` are "not a keyword". Planned:
`help|help me|sos|_other` (optionally `_noise` as a 5th class).

## 4. On the device (fixed - `kws/detect.py` = `Ai/kws_detect.c`)
- the CNN runs every 12 frames (240 ms) on the last T frames, only if
  something louder than -55 dBFS was heard in those 240 ms;
- the last 3 runs are averaged; a keyword class above `thr[sensitivity]`
  is one hit, then 1.2 s pause;
- "Only My Voice": the hit also needs cosine(speaker vector of the clearest
  of those runs, enrolled template) >= `spk_thr`;
- `hits` hits within `hits_ms` -> Voice SOS (the UI then gives 5 s to cancel).
- Enrolment on the device: 5 takes of 2 s ("Help", "Help me", "SOS",
  "Help", "Help me"); for each take the window where its phrase is most
  likely (`kws_best_window`, 40 ms steps) gives the speaker vector; the
  template per class = sum of those vectors (compared by cosine).

## 5. Export and hand-back
```
torch.save({"cfg": net.cfg, "state": net.state_dict()}, "best.pt")
python -m kws.export best.pt out --labels "help|help me|sos|_other" \
       --thr 0.90 0.80 0.65 --spk-thr 0.70 --hits 3 --hits-ms 20000 --name kws-YYYYMMDD
python -m kws.check_c out          # must end with "0 failed"
```
Hand back: `out/kws_model.c`, `out/kws_model_test.c`, `REPORT.md`, plus the
training code, `configs/train.yaml`, `models/best.pt` and `TRAINING.md`
(see PLAN.md "Training code and parameters").
On the firmware side: `kws_model.c` -> `Ai/`, `kws_model_test.c` -> `Tests/`.

Do not change `features.py`, the layer types in `model.py`, `detect.py` or
the export format. If something really has to change, write it in REPORT.md
together with the matching C change - the firmware side is done in the other
conversation.
