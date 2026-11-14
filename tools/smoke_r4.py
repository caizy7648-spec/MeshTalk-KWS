"""smoke_r4.py - runs the whole Phase C pipeline on a few minutes of made-up audio.

    python tools/smoke_r4.py            # about 3-6 min; prints PASS / FAIL per step

Builds a throw-away world in build/smoke_r4/ (its own config, raw data in the formats of
VocalSound / ESC-50 / AMI / AISHELL-1 / MeshTalk DataRec, a tiny Step-1 cache) and runs:
04_import_own -> 05 extraction functions -> 06_prepare_r4 -> 13_mine_r4 ->
10_train_kw --stage r4 (1 short epoch) -> 21_eval_r4 --quick (r3, sweep, r4) ->
30_export --section export_r4 -> 21_eval_r4 --native (int8 C engine) -> --plot.
Nothing in data/, models/, out/ or docs/ of the repository is touched.
Use it before the long downloads to check the environment (Python packages, gcc), and after
changing any r4 script. The numbers it prints mean nothing (the audio is synthetic).
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import wave
import zipfile

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
W = os.path.join(ROOT, "build", "smoke_r4")
RATE = 16000
rng = np.random.default_rng(7)


def wav_bytes(x):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(np.asarray(x, np.int16).tobytes())
    return b.getvalue()


def write_wav(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(wav_bytes(x))


def voice(sec, f0=140.0, level=0.2, kind="voiced"):
    """speech-like sound: harmonics with an envelope (voiced) or a noise burst (cough-like)"""
    n = int(sec * RATE)
    t = np.arange(n) / RATE
    env = np.sin(np.pi * np.arange(n) / n) ** 0.5
    if kind == "voiced":
        x = sum(np.sin(2 * np.pi * f0 * k * t * (1 + 0.05 * np.sin(2 * np.pi * 3 * t))) / k for k in range(1, 12))
    else:
        x = rng.standard_normal(n) * np.exp(-t * 8)
    x = x / (np.abs(x).max() + 1e-9) * env * level
    return (x * 32767).astype(np.int16)


def silence(sec, level_db=-70):
    return (rng.standard_normal(int(sec * RATE)) * 32768 * 10 ** (level_db / 20)).astype(np.int16)


def word(label):
    f0 = {"help": 150, "help me": 120, "sos": 200}.get(label, rng.uniform(90, 260))
    parts = [voice(0.35, f0)] if label != "help me" else [voice(0.3, f0), silence(0.05), voice(0.25, f0 * 1.2)]
    if label == "sos":
        parts = [voice(0.2, f0), silence(0.08), voice(0.25, f0 * 0.8), silence(0.08), voice(0.2, f0)]
    return np.concatenate([silence(0.3)] + parts + [silence(0.3)])


# ------------------------------------------------------------------ the world
def make_config():
    C = yaml.safe_load(open(os.path.join(ROOT, "configs", "train.yaml")))
    C["paths"] = {k: os.path.join(W, k) for k in ("raw", "cache", "models", "out", "docs")}
    C["own"]["dir"] = os.path.join(W, "own")
    C["own"]["min_clips_per_speaker"] = 4
    C["train_kw"].update(windows_per_epoch=512, batch=64, val_windows_per_class=8, norm_windows=64, workers=2)
    C["train_kw_r4"].update(epochs=1, workers=2, torch_threads=2, init=os.path.join(W, "models", "kw_r3.pt"))
    C["train_kw_r4b"].update(epochs=1, workers=2, torch_threads=2, init=os.path.join(W, "models", "kw_r4.pt"))
    C["export_r4b"]["check_windows_per_class"] = 4
    C["mine_r4"]["model"] = os.path.join(W, "models", "kw_r3.pt")
    C["mine_r4"]["threshold"] = 0.0                  # synthetic audio: make sure something is "mined"
    C["eval_r4"].update(procs=2, sweep_med_thr=[0.5, 0.95], sweep_help_extra=[0.0, 0.05], max_per_set=6, pairs_per_set=3)
    C["export_r4"]["check_windows_per_class"] = 4
    C["r4_prepare"]["fa_hours"] = {"val": {"ami": 1, "aishell": 1}, "test": {"ami": 1, "aishell": 1}}
    C["r4_sources"]["aishell"]["speakers"] = {"train": 2, "dev": 1, "test": 1}
    C["r4_sources"]["aishell"]["utts_per_speaker"] = {"train": 3, "dev": 3, "test": 3}
    for s in ("train", "val", "test"):
        C["r4_sources"]["ami"][s] = C["r4_sources"]["ami"][s][:2]
    path = os.path.join(W, "smoke.yaml")
    yaml.safe_dump(C, open(path, "w"), sort_keys=False)
    return path, C


def base_cache(C):
    """a tiny Step-1 cache: clips (all groups of r1-r3), noise banks, LibriSpeech false-alarm speech"""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    os.environ["KWS_CONFIG"] = os.path.join(W, "smoke.yaml")
    prep = __import__("02_prepare")
    st = prep.Store("clips")
    groups = [("help", "mswc_help"), ("help", "libri_help"), ("help me", "libri_help_me"), ("_other", "libri_me"),
              ("help", "tts"), ("help me", "tts"), ("sos", "tts"), ("_other", "tts_other"), ("_other", "mswc_lookalike"),
              ("_other", "mswc_random"), ("_other", "speech_commands"), ("_other", "libri_words"), ("_other", "libri_chunk")]
    for split in ("train", "val", "test"):
        for lab, g in groups:
            for k in range(6):
                spk = "%s-%s-%d" % (g, split, k % 3)
                x = word(lab) if lab != "_other" else word("other")
                st.add(x, source=g.split("_")[0], group=g, label=lab, word=lab, speaker=spk, gender="", split=split,
                       spk_use=False, text="", kw_start=int(0.3 * RATE), kw_end=len(x) - int(0.3 * RATE))
    st.save(prep.COLS)
    for name in ("hardneg", "hardneg_r3"):
        h = prep.Store(name)
        for k in range(10):
            x = word("other")
            h.add(x, source="librispeech", group="hard_neg" if name == "hardneg" else "hard_neg_r3", label="_other", word="x",
                  speaker="s", gender="", split="train", spk_use=False, text="", kw_start=0, kw_end=len(x))
        h.save(prep.COLS + ["prob"])
    for s in ("train", "test"):
        n = prep.Store("noise_" + s)
        for k in range(4):
            n.add((rng.standard_normal(RATE * 3) * 3000).astype(np.int16), source="musan_noise", file="n%d" % k)
        n.save()
    for s in ("dev", "test"):
        f, words, pos = prep.Store("fa_" + s), [], 0
        for k in range(3):
            x = np.concatenate([word("other"), silence(0.5), word("help"), silence(0.5), word("other")])
            words.append(dict(utt="u%d" % k, word="help", at=f.pos + len(word("other")) + 8000, end=f.pos + len(word("other")) + 8000 + 8000))
            f.add(x, utt="u%d" % k, speaker="ls-%d" % k, subset=s)
        f.save()
        pd.DataFrame(words, columns=["utt", "word", "at", "end"]).to_csv(os.path.join(C["paths"]["cache"], "fa_%s_words.csv" % s), index=False)


def raw_data(C):
    raw = C["paths"]["raw"]
    # VocalSound release zip
    types = ["cough", "laughter", "sigh", "sneeze", "sniff", "throatclearing"]
    zb = io.BytesIO()
    split = {"tr": [], "val": [], "te": []}
    with zipfile.ZipFile(zb, "w") as z:
        for spk in range(9):
            sp = ["tr", "val", "te"][spk % 3]
            for t in types:
                name = "f%04d_0_%s.wav" % (spk, t)
                x = np.concatenate([silence(0.5), voice(0.3, kind="burst"), silence(0.3), voice(0.25, kind="burst"), silence(0.8)])
                z.writestr("vs_release_16k/audio_16k/" + name, wav_bytes(x))
                split[sp].append({"wav": "/data/audio_16k/" + name, "labels": "/m/x"})
        for sp, rows in split.items():
            z.writestr("vs_release_16k/datafiles/%s.json" % sp, json.dumps({"data": rows}))
    d = os.path.join(raw, "vocalsound")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "vs_release_16k.zip"), "wb").write(zb.getvalue())
    # ESC-50 parquet (the Step-1 download format)
    rows = []
    for k in range(10):
        b = wav_bytes(np.concatenate([silence(1.0), voice(0.4, kind="burst"), silence(3.6)]))
        rows.append(dict(filename="%d-%d-A-24.wav" % (k % 5 + 1, 100 + k), fold=k % 5 + 1, target=24,
                         category=["coughing", "sneezing", "dog"][k % 3], audio={"bytes": b, "path": None}))
    os.makedirs(os.path.join(raw, "esc50"), exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(raw, "esc50", "train-0.parquet"))
    # AMI: meetings + the annotation zip
    d = os.path.join(raw, "ami")
    os.makedirs(d, exist_ok=True)
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w") as z:
        for s in ("train", "val", "test"):
            for mt in C["r4_sources"]["ami"][s]:
                parts = []
                for k in range(12):
                    parts += [word("other"), silence(rng.uniform(0.2, 1.0))]
                write_wav(os.path.join(d, mt + ".wav"), np.concatenate(parts))
                z.writestr("words/%s.A.words.xml" % mt,
                           '<nite:root><w nite:id="x" starttime="3.10" endtime="3.40">help</w>'
                           '<w nite:id="y" starttime="5.0" endtime="5.3">okay</w></nite:root>')
    open(os.path.join(d, "ami_public_manual_1.6.2.zip"), "wb").write(zb.getvalue())
    # AISHELL-1: data_aishell.tgz with wav/<SPK>.tar.gz inside
    tb = io.BytesIO()
    with tarfile.open(fileobj=tb, mode="w:gz") as t:
        for spk, sp in (("S0002", "train"), ("S0003", "train"), ("S0724", "dev"), ("S0764", "test")):
            ib = io.BytesIO()
            with tarfile.open(fileobj=ib, mode="w:gz") as it:
                for u in range(4):
                    b = wav_bytes(np.concatenate([silence(0.2), word("other"), word("other"), silence(0.2)]))
                    ti = tarfile.TarInfo("%s/%s/BAC009%sW%04d.wav" % (sp, spk, spk, u))
                    ti.size = len(b)
                    it.addfile(ti, io.BytesIO(b))
            data = ib.getvalue()
            ti = tarfile.TarInfo("data_aishell/wav/%s.tar.gz" % spk)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    open(os.path.join(W, "data_aishell.tgz"), "wb").write(tb.getvalue())
    # MeshTalk DataRec card
    ds = os.path.join(W, "card", "DATASET")
    lines = ["file,category,speaker,style,word,ms,peak_dbfs,clipped,result,uptime_s,utc"]
    for cat, lab in (("HELP", "help"), ("HELPME", "help me"), ("SOS", "sos")):
        for k in range(10):
            x = np.concatenate([silence(0.4), word(lab), silence(2.5)])[:40000]
            name = "A_%s_%04d.WAV" % (["NORMAL", "LOUD", "HOARSE"][k % 3], k + 1)
            write_wav(os.path.join(ds, cat, name), x)
            lines.append("%s/%s,%s,A,X,,2500,-15.0,0,ok,0,0" % (cat, name, cat))
    open(os.path.join(ds, "INDEX.CSV"), "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ steps
def run(name, args, env):
    t = time.time()
    r = subprocess.run([sys.executable] + args, cwd=ROOT, env=env, capture_output=True, text=True)
    ok = r.returncode == 0
    print("%-48s %s  (%.0f s)" % (name, "PASS" if ok else "FAIL", time.time() - t), flush=True)
    if not ok:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return ok


def main():
    if os.path.isdir(W):
        shutil.rmtree(W)
    os.makedirs(W)
    cfg, C = make_config()
    for p in C["paths"].values():
        os.makedirs(p, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "models", "kw_r3.pt"), os.path.join(C["paths"]["models"], "kw_r3.pt"))
    env = dict(os.environ, KWS_CONFIG=cfg, PYTHONHASHSEED="0")
    base_cache(C)
    raw_data(C)
    steps = [
        ("04_import_own (DataRec card)", ["scripts/04_import_own.py", "--src", os.path.join(W, "card"), "--config", cfg]),
        ("05 extraction: VocalSound zip", ["-c", "import sys; sys.argv=['x']; sys.path.insert(0,'scripts'); m=__import__('05_download_r4'); "
                                              "import os; d=os.path.join(m.RAW,'vocalsound'); print(m.vocalsound_extract(os.path.join(d,'vs_release_16k.zip'), d))"]),
        ("05 extraction: AMI words", ["-c", "import sys; sys.path.insert(0,'scripts'); m=__import__('05_download_r4'); "
                                         "import os; d=os.path.join(m.RAW,'ami'); n=m.ami_words_extract(os.path.join(d,'ami_public_manual_1.6.2.zip'), d); assert n>0, n"]),
        ("05 extraction: AISHELL stream", ["-c", "import sys,os; sys.path.insert(0,'scripts'); m=__import__('05_download_r4'); "
                                              "k,n=m.aishell_from_stream(open(%r,'rb'), os.path.join(m.RAW,'aishell'), m.R['aishell']); print(k,n); assert n>0"
                                              % os.path.join(W, "data_aishell.tgz")]),
        ("06_prepare_r4", ["scripts/06_prepare_r4.py"]),
        ("13_mine_r4 (r3 on the new audio)", ["scripts/13_mine_r4.py"]),
        ("10_train_kw --stage r4 (1 short epoch)", ["scripts/10_train_kw.py", "--stage", "r4"]),
        ("21_eval_r4 --quick, r3 (test)", ["scripts/21_eval_r4.py", "--model", os.path.join(C["paths"]["models"], "kw_r3.pt"),
                                          "--split", "test", "--tag", "r3", "--quick"]),
        ("21_eval_r4 --quick --sweep, r4 (val)", ["scripts/21_eval_r4.py", "--model", os.path.join(C["paths"]["models"], "kw_r4.pt"),
                                                 "--split", "val", "--sweep", "--quick"]),
        ("21_eval_r4 --quick --sweep --rule 1, r4 (val)", ["scripts/21_eval_r4.py", "--model", os.path.join(C["paths"]["models"], "kw_r4.pt"),
                                                         "--split", "val", "--sweep", "--quick", "--rule", "1"]),
        ("21_eval_r4 --quick, r4 (test)", ["scripts/21_eval_r4.py", "--model", os.path.join(C["paths"]["models"], "kw_r4.pt"),
                                          "--split", "test", "--tag", "r4", "--quick", "--from-export", "export_r4"]),
        ("30_export --section export_r4", ["scripts/30_export.py", "--section", "export_r4"]),
        ("21_eval_r4 --native (int8 C engine)", ["scripts/21_eval_r4.py", "--native", os.path.join(C["paths"]["out"], "kws_model.c"),
                                                "--split", "test", "--tag", "r4_int8", "--quick", "--from-export", "export_r4"]),
        ("21_eval_r4 --plot r3 r4", ["scripts/21_eval_r4.py", "--plot", "r3", "r4"]),
        ("check_own (r3 and r4 on every take)", ["tools/check_own.py", "--model", os.path.join(C["paths"]["models"], "kw_r4.pt")]),
        # r4b: from r4 + "help" with the end cut off as _other
        ("cut_tail on every own help clip", ["-c", "import numpy as np, kws.cfg as k; from kws.dataset import Data, KwWindows, cut_tail; "
                                                "C=k.load(); D=Data(C); C['train_kw']=dict(C['train_kw'], **C['train_kw_r4b']); "
                                                "ds=KwWindows(D, C, 'train', 64); P=ds.P['cut_help','own']; assert len(P), 'no own help'; "
                                                "r=[cut_tail(*D.clip(i), np.random.default_rng(1), C['cut_help']) for i in P]; "
                                                "assert all(0 < c - s < e - s for (y, s, c), (_, _, e) in zip(r, [D.clip(i) for i in P])); "
                                                "w=[ds.make(np.random.default_rng(i), 3) for i in range(40)]; print(len(P), 'cut ok')"]),
        ("10_train_kw --stage r4b (1 short epoch)", ["scripts/10_train_kw.py", "--stage", "r4b"]),
        ("21_eval_r4 --quick, r4b (test, cut sets)", ["scripts/21_eval_r4.py", "--model", os.path.join(C["paths"]["models"], "kw_r4b.pt"),
                                                     "--split", "test", "--tag", "r4b", "--quick", "--from-export", "export_r4b"]),
        ("21_eval_r4 --quick --sweep --tag r4b (val)", ["scripts/21_eval_r4.py", "--model", os.path.join(C["paths"]["models"], "kw_r4b.pt"),
                                                       "--split", "val", "--sweep", "--quick", "--tag", "r4b"]),
        ("30_export --section export_r4b", ["scripts/30_export.py", "--section", "export_r4b"]),
        ("21_eval_r4 --plot r4 r4b", ["scripts/21_eval_r4.py", "--plot", "r4", "r4b"]),
    ]
    fails = 0
    for name, args in steps:
        fails += 0 if run(name, args, env) else 1
    print("smoke test: %d failed" % fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
