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

v1.5.1 (same as the firmware):
  - high=True (sensitivity "high"): the best 2 of the last 3 runs are averaged;
  - the separate speaker model is only asked when only_me; a voice that does
    not match is asked again on the next VERIFY_TRIES-1 runs (while the
    keyword is still >= its threshold - RETRY_DROP) before it is rejected;
  - the device default is now hits=2, hits_ms=10000 (setting "Calls Needed").

run_fn(window[T, 40]) -> (probs[ncls], emb or None). Use the PyTorch model
for speed, or export.reference() for the bit-exact int8 version.
"""
import numpy as np

STEP = 12            # frames between runs (240 ms)
SMOOTH = 3           # runs averaged
REFRACT_MS = 1200
GATE_DBFS = -55.0

VERIFY_TRIES = 3
RETRY_DROP = 0.15
SEG_MIN, SEG_END, GAP, NEW_MIN, NEW_AFTER = 5, 3, 7, 8, 20   # v1.6 sounds (frames), kws_detect.h
REARM_DROP, SUSTAIN_MS = 0.15, 2500
CONFIRM_DROP = 0.08
VOICE_CONFIRM = 0.03  # v1.6d (v1.6b 0.05): a confirming call needs this much less voice match
REJ_PENALTY, REJ_MS, REJ_MAX, REJ_CLEAR = 0.02, 10000, 2, 0.06   # v1.6f: per keyword, stricter after its CLEAR misses   # v1.6: after a hit, the next call within hits_ms needs this much less

EV_RAN, EV_HIT, EV_REJECT, EV_SOS = 1, 2, 4, 8
# v1.6.1: words one by one (kws_detect.h KWS_DIP_DB ...)
WORD_MIN, DIP_DB, DIP_RISE, WORD_PAD, WORD_SLACK, WORD_AFTER = 5, 8.0, 6.0, 6, 0.10, 15


class Detector:
    def __init__(self, run_fn, T, keyword, thr, cls_off=None, hits=3, hits_ms=20000, spk_thr=0.7, voice=None,
                 only_me=False, verify=None, verify_thr=0.5, high=False, sos_cls=None, verify_thr_k=None, rej_step=None, words_on=False):
        self.run_fn, self.T, self.keyword = run_fn, T, np.asarray(keyword, bool)
        self.thr, self.hits_needed, self.hits_ms = thr, hits, hits_ms
        self.cls_off = np.zeros(len(self.keyword)) if cls_off is None else np.asarray(cls_off, float)
        self.verify, self.verify_thr = verify, verify_thr
        self.verify_thr_k = verify_thr_k     # v1.6c: per class (> 0), else verify_thr
        self.rej_step = rej_step             # v1.6g: penalty per clear miss per class (> 0), else REJ_PENALTY
        self.spk_thr, self.voice, self.only_me = spk_thr, voice, only_me   # voice: {class: template} or None
        self.ring = []
        self.step = 0
        self.quiet = 1 << 20
        self.hist = []
        self.last_hit = 0
        self.hits = []
        self.rej = []
        self.high = high
        self.retry_cls, self.retry_left, self.retry_n = -1, 0, 0
        self.sos_cls = sos_cls
        self.hold_until = 0
        self.words_on = words_on             # v1.6.1: 2 calls straight after each other (hits == 2 only)
        self.qfeat = None
        self._seg_reset()

    # ---- v1.6: counting each call once (kws_detect.c: seg_frame, counted_start, new_sound)
    def _seg_reset(self):
        self.fno, self.floor_db, self.peak_db = 0, np.float32(-120), np.float32(-120)
        self.seg, self.seg_open, self.seg_quiet = [], False, 0   # seg: [s, e]
        self.armed, self.after_sos, self.cnt_start = True, False, 0
        self.cnt_fno, self.ep_from, self.ep_calls = 0, 0, 0
        self.word, self.word_open = [], False          # word: [s, e, cls, checked, p]
        self.w_s, self.w_max, self.w_min, self.w_minf = 0, np.float32(0), np.float32(0), 0
        self.smooth_last = None

    def _seg_frame(self, db):
        db = np.float32(db)
        self.fno += 1
        if self.floor_db <= np.float32(-119) or db < self.floor_db:
            self.floor_db = db
        else:
            self.floor_db = np.float32(self.floor_db + np.float32(0.02))
        dec = np.float32(self.peak_db - np.float32(0.5))
        self.peak_db = db if db > dec else dec
        lim = np.float32(self.floor_db + np.float32(12))
        p30 = np.float32(self.peak_db - np.float32(30))
        if p30 > lim:
            lim = p30
        if np.float32(GATE_DBFS) > lim:
            lim = np.float32(GATE_DBFS)
        if db >= lim:
            if not self.seg_open:
                if len(self.seg) == 12:
                    self.seg.pop(0)
                self.seg.append([self.fno, self.fno])
                self.seg_open = True
            else:
                self.seg[-1][1] = self.fno
            self.seg_quiet = 0
        elif self.seg_open:
            self.seg_quiet += 1
            if self.seg_quiet >= SEG_END:
                self.seg_open = False
                if self.word_open:
                    self._word_push(self.w_s, self.seg[-1][1])
                    self.word_open = False
                if self.seg[-1][1] - self.seg[-1][0] + 1 < SEG_MIN:
                    self.seg.pop()
        loud = db >= lim
        if self.seg_open:
            if not self.word_open:
                if loud:
                    self.word_open, self.w_s, self.w_max, self.w_min, self.w_minf = True, self.fno, db, db, self.fno
            else:
                if db > self.w_max:
                    self.w_max, self.w_min, self.w_minf = db, db, self.fno
                elif db < self.w_min:
                    self.w_min, self.w_minf = db, self.fno
                if (loud and np.float32(self.w_max - self.w_min) >= np.float32(DIP_DB)
                        and np.float32(db - self.w_min) >= np.float32(DIP_RISE) and self.w_minf >= self.w_s + WORD_MIN):
                    self._word_push(self.w_s, self.w_minf - 1)
                    self.w_s = self.w_minf + 1 if self.w_minf + 1 <= self.fno else self.fno
                    self.w_max = self.w_min = db
                    self.w_minf = self.fno
        i = 0
        while i < len(self.word) and self.word[i][1] + 2 * self.T < self.fno:
            i += 1
        del self.word[:i]
        i = 0
        while i < len(self.seg) and self.seg[i][1] + 3 * self.T < self.fno and not (self.seg_open and i == len(self.seg) - 1):
            i += 1
        del self.seg[:i]

    def _word_push(self, s, e):
        if e < s or e - s + 1 < WORD_MIN:
            return
        if len(self.word) == 12:
            self.word.pop(0)
        self.word.append([s, e, -1, False, 0.0])

    def _word_check(self, i, win):
        T = self.T
        base = self.fno + 1 - T
        s, e = self.word[i][0], self.word[i][1]
        if e - s + 1 > T - WORD_PAD - 4:
            s = e + 1 - (T - WORD_PAD - 4)
        L = e - s + 1
        dst = T - WORD_PAD - L
        w2 = np.stack([win[s - base + (r - dst)] if dst <= r < dst + L else self.qfeat for r in range(T)])
        probs, _ = self.run_fn(w2)
        probs = np.asarray(probs, np.float32)
        thr = np.float32(self.thr)
        self.word[i][3], self.word[i][2], self.word[i][4] = True, -1, 0.0
        bm = -1e9
        for k in range(len(probs)):
            if not self.keyword[k]:
                continue
            mg = float(np.float32(probs[k] - np.float32(thr + np.float32(self.cls_off[k]))))
            if mg > bm:
                bm = mg
                self.word[i][4] = float(probs[k])
                self.word[i][2] = k if mg >= float(np.float32(-WORD_SLACK)) else -1

    def _word_calls(self, win):
        n, checks, have, last = 0, 0, False, 0
        base = self.fno + 1 - self.T
        for i, w in enumerate(self.word):
            if w[0] < self.ep_from:
                continue
            if not w[3]:
                if w[0] < base or checks >= 3:
                    continue
                self._word_check(i, win)
                checks += 1
            if w[2] >= 0 and (not have or w[0] >= last + WORD_AFTER):
                n, have, last = n + 1, True, w[0]
        return n

    def _sounds(self):
        a = 0
        while a < len(self.seg):
            b = a + 1
            while b < len(self.seg) and self.seg[b][0] - self.seg[b - 1][1] - 1 < GAP:
                b += 1
            yield a, b
            a = b

    def _counted_start(self):
        pick = newest = self.fno
        for a, b in self._sounds():
            newest = self.seg[a][0]
            if self.seg[a][0] + 15 <= self.fno:
                pick = self.seg[a][0]
        return pick if pick != self.fno else newest

    def _new_sound(self):
        for a, b in self._sounds():
            if self.seg[a][0] >= self.cnt_start + NEW_AFTER and self.seg[b - 1][1] - self.seg[a][0] + 1 >= NEW_MIN:
                return self.seg[a][0]
        return None

    def reset(self):
        self.ring, self.step, self.quiet, self.hist = [], 0, 1 << 20, []
        self.retry_left = 0
        self._seg_reset()

    def _note_reject(self, t_ms, c, vneed):
        if self.verify_thr_k is not None and self.verify_thr_k[c] > 0:
            base = np.float32(self.verify_thr_k[c])     # the keyword's own value (no drop / penalty)
        else:
            base = np.float32(self.verify_thr)
        if self.hit_spk <= -2.0 or self.hit_spk >= np.float32(base - np.float32(REJ_CLEAR)):
            return
        self.rej = (self.rej + [(t_ms, c)])[-8:]

    def _pen(self, c):
        n = sum(1 for _, k in self.rej if k == c)
        step = self.rej_step[c] if self.rej_step is not None and self.rej_step[c] > 0 else REJ_PENALTY
        return np.float32(np.float32(step) * np.float32(min(n, REJ_MAX)))

    def _count(self, t_ms, ev, calls=1):
        ev |= EV_HIT
        self.rej = []                                  # v1.6f: an accepted call ends any penalty
        self.hits = [h for h in self.hits if t_ms - h < self.hits_ms]
        for _ in range(max(calls, 1)):
            self.hits = (self.hits + [t_ms])[:8]
            if len(self.hits) >= self.hits_needed:
                ev |= EV_SOS
                self.hits = []
                self.after_sos = True
                break
        return ev

    def frame(self, feat, dbfs, t_ms):
        ev = 0
        self.ring.append(np.asarray(feat, np.float32))
        if len(self.ring) > self.T:
            self.ring.pop(0)
        self._seg_frame(dbfs)
        if self.fno == 1 or np.float32(dbfs) < np.float32(self.floor_db + np.float32(6)):
            self.qfeat = np.asarray(feat, np.float32).copy()
        self.quiet = 0 if dbfs >= GATE_DBFS else min(self.quiet + 1, 1 << 20)
        if len(self.ring) < self.T:
            return 0
        self.step += 1
        if self.step < STEP:
            return 0
        self.step = 0
        if self.quiet >= self.T:                 # nothing loud in the whole window: re-armed (v1.6)
            self.hist = []
            self.armed, self.after_sos = True, False
            return 0
        win = np.stack(self.ring)
        probs, emb = self.run_fn(win)
        ev |= EV_RAN
        self.hist = (self.hist + [(np.asarray(probs), emb)])[-SMOOTH:]
        P = np.stack([h[0] for h in self.hist]).astype(np.float32)
        if self.high and len(self.hist) == SMOOTH:
            smooth = ((P.sum(axis=0) - P.min(axis=0)) / np.float32(SMOOTH - 1)).astype(np.float32)
        else:
            smooth = P.mean(axis=0).astype(np.float32)
        thr = np.float32(self.thr)
        vdrop = np.float32(0)
        if any(t_ms - h < self.hits_ms for h in self.hits):
            thr = np.float32(thr - np.float32(CONFIRM_DROP))
            vdrop = np.float32(VOICE_CONFIRM)
        self.rej = [r for r in self.rej if t_ms - r[0] < REJ_MS]

        def vneed(c):
            if self.verify_thr_k is not None and self.verify_thr_k[c] > 0:
                return np.float32(np.float32(np.float32(self.verify_thr_k[c]) - vdrop) + self._pen(c))
            return np.float32(np.float32(np.float32(self.verify_thr) - vdrop) + self._pen(c))
        lim_k = thr + self.cls_off.astype(np.float32)
        if self.retry_left > 0:                  # the voice did not match: try this newer window
            c = self.retry_cls
            self.retry_left -= 1
            if self.verify is not None and self.only_me and smooth[c] >= lim_k[c] - np.float32(RETRY_DROP):
                sim = float(self.verify(win, c))
                self.hit_spk = max(self.hit_spk, sim)
                if sim <= -2.0 or sim >= vneed(c):
                    self.retry_left = 0
                    self.hit_score = max(self.hit_score, float(smooth[c]))
                    return self._count(t_ms, ev, self.retry_n)
            else:
                self.retry_left = 0
            if self.retry_left == 0:
                self._note_reject(t_ms, c, vneed)
                return ev | EV_REJECT
            return ev
        margin = smooth - lim_k
        cand = np.where(self.keyword & (margin >= 0), margin, -1e9)
        best = int(np.argmax(cand))
        base = np.float32(self.thr) + self.cls_off.astype(np.float32)
        top = max(float(np.float32(smooth[k] - base[k])) for k in range(len(smooth)) if self.keyword[k])
        if top < float(np.float32(-REARM_DROP)):
            self.armed, self.after_sos = True, False
        if cand[best] <= -1e9:
            return ev
        raw = np.asarray(self.hist[-1][0], np.float32)
        if self.armed:
            start = self._counted_start()
            w0 = self.fno + 1 - self.T if self.fno >= self.T else 0
            self.ep_from, self.ep_calls = max(self.cnt_fno + 1, w0), 0
        elif (not self.after_sos and self.words_on and self.hits_needed == 2
              and self._word_calls(win) > self.ep_calls):
            start = self._counted_start()
        elif not self.after_sos and self._new_sound() is not None and raw[best] >= lim_k[best]:
            start = self._new_sound()
        elif not self.after_sos and self.last_hit and t_ms - self.last_hit >= SUSTAIN_MS:
            start = self._counted_start()
        else:
            return ev
        calls = 1
        self.armed = False
        self.ep_calls += 1
        self.cnt_fno = self.fno
        self.cnt_start = start
        self.last_hit = t_ms if t_ms else 1
        self.hit_cls, self.hit_score, self.hit_spk = best, float(smooth[best]), -2.0
        if self.verify is not None:
            if self.only_me:
                self.hit_spk = float(self.verify(win, best))
                if -2.0 < self.hit_spk < vneed(best) and VERIFY_TRIES > 1:
                    self.retry_cls, self.retry_left, self.retry_n = best, VERIFY_TRIES - 1, calls
                    return ev
        elif self.voice is not None and emb is not None:
            at = int(np.argmax([h[0][best] for h in self.hist]))
            tmpl = self.voice.get(best, self.voice.get("all"))
            e = self.hist[at][1]
            self.hit_spk = float(np.dot(tmpl, e) / (np.linalg.norm(tmpl) * np.linalg.norm(e) + 1e-12))
        lim = vneed(best) if self.verify is not None else self.spk_thr
        if self.only_me and self.hit_spk > -2.0 and self.hit_spk < lim:
            self._note_reject(t_ms, best, vneed)
            return ev | EV_REJECT
        return self._count(t_ms, ev, calls)
