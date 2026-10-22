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
