# MeshTalk-KWS - training package for MeshTalk voice SOS (v1.4)

| file | what |
|---|---|
| `HANDOFF.md` | **the prompt to paste into the Claude Code cloud session** |
| `CONTRACT.md` | what the firmware expects (features, network, decisions, export) |
| `PLAN.md` | steps 0-4, data sources, targets, go / no-go |
| `kws/features.py` | log-mel front-end = firmware `Ai/kws_frontend.c` |
| `kws/model.py` | `KwsNet` = firmware `Ai/kws_engine.c` |
| `kws/detect.py` | device decision logic = firmware `Ai/kws_detect.c` |
| `kws/export.py` | model -> `kws_model.c` + `kws_model_test.c` |
| `kws/check_c.py` | compiles `c_ref/` (the firmware C code) with a model and compares |
| `tools/make_golden.py` | rebuilds the golden test files in `c_ref/` |
| `data/` | datasets (not in git), `data/own/<label>/` = the owner's recordings |

Quick check: `python -m kws.check_c` -> `15 checks, 0 failed`.
