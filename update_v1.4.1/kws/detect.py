"""detect.py - Python copy of the firmware's decision logic (Ai/kws_detect.c),
so that false alarms per hour and detection rates are measured exactly the
way the device behaves.

    det = Detector(run_fn, T, keyword_mask, thr, cls_off=None, hits=3, hits_ms=20000)
    for each 20 ms frame: events = det.frame(feat_row, dbfs, t_ms)

v1.4.1 (same as the firmware):
  - window gate: the CNN runs every 240 ms as long as any frame of the window
    was louder than -55 dBFS; only a completely quiet window skips the run;
  - per class threshold: class k hits when smooth[k] >= thr + cls_off[k]
    (the class with the largest margin wins);
  - verify(window, cls) -> similarity: optional separate speaker model.

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
    def __init__(self, run_fn, T, keyword, thr, cls_off=None, hits=3, hits_ms=20000, spk_thr=0.7, voice=None,
                 only_me=False, verify=None, verify_thr=0.5):
        self.run_fn, self.T, self.keyword = run_fn, T, np.asarray(keyword, bool)
        self.thr, self.hits_needed, self.hits_ms = thr, hits, hits_ms
        self.cls_off = np.zeros(len(self.keyword)) if cls_off is None else np.asarray(cls_off, float)
        self.verify, self.verify_thr = verify, verify_thr
        self.spk_thr, self.voice, self.only_me = spk_thr, voice, only_me   # voice: {class: template} or None
        self.ring = []
        self.step = 0
        self.quiet = 1 << 20
        self.hist = []
        self.last_hit = 0
        self.hits = []

    def reset(self):
        self.ring, self.step, self.quiet, self.hist = [], 0, 1 << 20, []

    def frame(self, feat, dbfs, t_ms):
        ev = 0
        self.ring.append(np.asarray(feat, np.float32))
        if len(self.ring) > self.T:
            self.ring.pop(0)
        self.quiet = 0 if dbfs >= GATE_DBFS else min(self.quiet + 1, 1 << 20)
        if len(self.ring) < self.T:
            return 0
        self.step += 1
        if self.step < STEP:
            return 0
        self.step = 0
        if self.quiet >= self.T:                 # nothing loud in the whole window
            self.hist = []
            return 0
        win = np.stack(self.ring)
        probs, emb = self.run_fn(win)
        ev |= EV_RAN
        self.hist = (self.hist + [(np.asarray(probs), emb)])[-SMOOTH:]
        smooth = np.mean([h[0] for h in self.hist], axis=0).astype(np.float32)
        margin = smooth - (np.float32(self.thr) + self.cls_off.astype(np.float32))
        cand = np.where(self.keyword & (margin >= 0), margin, -1e9)
        best = int(np.argmax(cand))
        if cand[best] <= -1e9 or (self.last_hit and t_ms - self.last_hit < REFRACT_MS):
            return ev
        self.last_hit = t_ms if t_ms else 1
        self.hit_cls, self.hit_score, self.hit_spk = best, float(smooth[best]), -2.0
        if self.verify is not None:
            self.hit_spk = float(self.verify(win, best))
        elif self.voice is not None and emb is not None:
            at = int(np.argmax([h[0][best] for h in self.hist]))
            tmpl = self.voice.get(best, self.voice.get("all"))
            e = self.hist[at][1]
            self.hit_spk = float(np.dot(tmpl, e) / (np.linalg.norm(tmpl) * np.linalg.norm(e) + 1e-12))
        lim = self.verify_thr if self.verify is not None else self.spk_thr
        if self.only_me and self.hit_spk > -2.0 and self.hit_spk < lim:
            return ev | EV_REJECT
        ev |= EV_HIT
        self.hits = [h for h in self.hits if t_ms - h < self.hits_ms] + [t_ms]
        if len(self.hits) >= self.hits_needed:
            ev |= EV_SOS
            self.hits = []
        return ev
