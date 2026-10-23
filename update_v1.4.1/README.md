# update_v1.4.1 - firmware-side changes for the training package

Copy these over the repository (see HANDOFF_v2.md, step 0):
- `c_ref/`        -> `c_ref/`  (firmware Ai/ code v1.4.1 + test_kws.c, spk_engine.h interface)
- `kws/detect.py` -> `kws/detect.py`  (window gate, per class offsets, verify hook)
- `kws/export.py` -> `kws/export.py`  (`--cls-off`)
- `CONTRACT.md`, `CONTRACT_SPK.md` -> repository root

Then `python -m kws.check_c` must print `19 checks, 0 failed`.
