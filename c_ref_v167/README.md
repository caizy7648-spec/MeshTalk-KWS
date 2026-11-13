# c_ref_v167 - the keyword engine of firmware MeshTalk_ESP32_v1.6.7 (reference, do not edit)

Copied unchanged from the firmware (Ai/ and Svc/kws.c) for Phase C (`PLAN_R4.md`):
evaluate r3 and r4 with the detector the device really runs.

| file | what |
|---|---|
| `kws_frontend.c/.h` | log-mel front end (20 ms blocks) |
| `kws_engine.c/.h` | int8 CNN inference |
| `kws_detect.c/.h` | detector: smoothing, per-class thresholds, sound/word segmentation (words mode), 2-in-10 s / 3-in-20 s rules, v1.6.7 `cls_extra` / `no_relax` |
| `kws_model.c` | model r3 as built into v1.6.7 |
| `svc_kws.c` | the firmware service (`Svc/kws.c`): how the detector is set up - `apply_opts()` (sensitivity, rule, words mode, `HELP_EXTRA` = +0.05 and `no_relax` for single "help") and how audio is fed (`kws_task`) |

`c_ref/` is the older v1.4.1 copy that `kws.check_c` uses; leave it as it is.
