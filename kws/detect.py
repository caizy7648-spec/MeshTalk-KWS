"""detect.py - Python copy of the firmware's decision logic (Ai/kws_detect.c),
so that false alarms per hour and detection rates are measured exactly the
way the device behaves.

    det = Detector(run_fn, ncls, keyword_mask, thr, hits=3, hits_ms=20000)
    for each 20 ms frame: events = det.frame(feat_row, dbfs, t_ms)

run_fn(window[T, 40]) -> (probs[ncls], emb or None). Use the PyTorch model
for speed, or export.reference() for the bit-exact int8 version.
"""
import numpy as np

STEP = 12            # frames between runs (240 ms)
SMOOTH = 3           # runs averaged
REFRACT_MS = 1200
GATE_DBFS = -55.0

EV_RAN, EV_HIT, EV_REJECT, EV_SOS = 1, 2, 4, 8


class Detector:
    def __init__(self, run_fn, T, keyword, thr, hits=3, hits_ms=20000, spk_thr=0.7, voice=None, only_me=False):
        self.run_fn, self.T, self.keyword = run_fn, T, np.asarray(keyword, bool)
        self.thr, self.hits_needed, self.hits_ms = thr, hits, hits_ms
        self.spk_thr, self.voice, self.only_me = spk_thr, voice, only_me   # voice: {class: template} or None
        self.ring = []
        self.step = 0
        self.loud = -120.0
        self.hist = []
        self.last_hit = 0
        self.hits = []

    def reset(self):
        self.ring, self.step, self.loud, self.hist = [], 0, -120.0, []

    def frame(self, feat, dbfs, t_ms):
        ev = 0
        self.ring.append(np.asarray(feat, np.float32))
        if len(self.ring) > self.T:
            self.ring.pop(0)
        self.loud = max(self.loud, dbfs)
        if len(self.ring) < self.T:
            return 0
        self.step += 1
        if self.step < STEP:
            return 0
        self.step = 0
        if self.loud < GATE_DBFS:
            self.loud = -120.0
            self.hist = []
            return 0
        self.loud = -120.0
        probs, emb = self.run_fn(np.stack(self.ring))
        ev |= EV_RAN
        self.hist = (self.hist + [(np.asarray(probs), emb)])[-SMOOTH:]
        smooth = np.mean([h[0] for h in self.hist], axis=0)
        cand = np.where(self.keyword & (smooth >= self.thr), smooth, -1.0)
        best = int(np.argmax(cand))
        if cand[best] < 0 or (self.last_hit and t_ms - self.last_hit < REFRACT_MS):
            return ev
        self.last_hit = t_ms if t_ms else 1
        self.hit_cls, self.hit_score, self.hit_spk = best, float(smooth[best]), -2.0
        if self.voice is not None and emb is not None:
            at = int(np.argmax([h[0][best] for h in self.hist]))
            tmpl = self.voice.get(best, self.voice.get("all"))
            e = self.hist[at][1]
            self.hit_spk = float(np.dot(tmpl, e) / (np.linalg.norm(tmpl) * np.linalg.norm(e) + 1e-12))
        if self.only_me and self.voice is not None and self.hit_spk > -2.0 and self.hit_spk < self.spk_thr:
            return ev | EV_REJECT
        ev |= EV_HIT
        self.hits = [h for h in self.hits if t_ms - h < self.hits_ms] + [t_ms]
        if len(self.hits) >= self.hits_needed:
            ev |= EV_SOS
            self.hits = []
        return ev
