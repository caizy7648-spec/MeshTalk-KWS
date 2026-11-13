/* kws_detect.c - voice SOS decision logic. See kws_detect.h. */
#include "kws_detect.h"
#include <math.h>
#include <string.h>

void kws_det_reset(kws_det_t *d)
{
    d->head = d->filled = d->step = 0;
    d->quiet = 1 << 20;
    d->n_hist = d->hist_pos = 0;
    d->since = -1;
    d->retry_left = 0;
    d->nseg = 0;
    d->seg_open = false;
    d->seg_quiet = 0;
    d->floor_db = d->peak_db = -120.0f;
    d->armed = true;
    d->after_sos = d->same_told = false;
    d->cnt_fno = d->cnt_start = 0;
    d->nword = 0;
    d->word_open = false;
    d->ep_from = 0;
    d->ep_calls = 0;
    memset(d->smooth, 0, sizeof(d->smooth));
}

void kws_det_init(kws_det_t *d, const kws_model_t *m, float *work, float *ring, float *win)
{
    memset(d, 0, sizeof(*d));
    d->m = m;
    d->work = work;
    d->ring = ring;
    d->win = win;
    d->sens = KWS_SENS_MED;
    d->step_frames = KWS_STEP;
    d->hit_cls = -1;
    kws_det_reset(d);
}

void kws_det_window(const kws_det_t *d, float *out)
{
    int T = d->m->T, i;
    for (i = 0; i < T; i++) {
        int r = (d->head + i) % T;                  /* head = oldest once full */
        memcpy(out + i * KWS_BANDS, d->ring + r * KWS_BANDS, sizeof(float) * KWS_BANDS);
    }
}

static float cosine(const float *a, const float *b, int n)
{
    float ab = 0.0f, aa = 0.0f, bb = 0.0f;
    int i;
    for (i = 0; i < n; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
    if (aa <= 0.0f || bb <= 0.0f) { return -2.0f; }
    return ab / sqrtf(aa * bb);
}

float kws_voice_similarity(const kws_voice_t *v, int dim, int k, const float *emb)
{
    if (!v || dim <= 0 || dim > KWS_VOICE_DIM) { return -2.0f; }
    if (k >= 0 && k < KWS_MAX_NCLS && v->have[k]) { return cosine(v->tmpl[k], emb, dim); }
    if (v->n_all) { return cosine(v->all, emb, dim); }
    return -2.0f;
}

void kws_voice_add(kws_voice_t *v, int dim, int k, const float *emb)
{
    int i;
    if (!v || dim <= 0 || dim > KWS_VOICE_DIM || k < 0 || k >= KWS_MAX_NCLS) { return; }
    for (i = 0; i < dim; i++) { v->tmpl[k][i] += emb[i]; v->all[i] += emb[i]; }
    if (v->have[k] < 255) { v->have[k]++; }
    if (v->n_all < 255) { v->n_all++; }
}

bool kws_voice_ready(const kws_model_t *m, const kws_voice_t *v, int takes)
{
    int k;
    if (!kws_model_ok(m) || !v) { return false; }
    if (v->n_all < takes) { return false; }
    for (k = 0; k < m->ncls; k++) {
        if (kws_is_keyword(m, k) && v->have[k] < 1) { return false; }
    }
    return true;
}

/* v1.6: the sounds = segments joined across pauses shorter than KWS_GAP.
 * Calls f(first, last) segment index for each, oldest first. */
#define FOR_SOUNDS(d, a, b) \
    for (a = 0, b = 0; a < (d)->nseg; a = b) \
        if ((b = sound_end(d, a)), 1)

static int sound_end(const kws_det_t *d, int i)
{
    int j = i + 1;
    while (j < d->nseg && (int)(d->seg[j].s - d->seg[j - 1].e - 1u) < KWS_GAP) { j++; }
    return j;
}

/* v1.6: the sound a fresh count stands for: the newest one that began at
 * least 300 ms ago (the keyword model needs that long to recognise a word;
 * a sound that just began is a possible next call), else the newest one */
static uint32_t counted_start(const kws_det_t *d)
{
    int a, b;
    uint32_t pick = d->fno, newest = d->fno;
    FOR_SOUNDS(d, a, b) {
        newest = d->seg[a].s;
        if (d->seg[a].s + 15u <= d->fno) { pick = d->seg[a].s; }
    }
    return pick != d->fno ? pick : newest;
}

/* v1.6: a new sound since the counted one (see KWS_NEW_AFTER); its start */
static bool new_sound(const kws_det_t *d, uint32_t *start)
{
    int a, b;
    FOR_SOUNDS(d, a, b) {
        uint32_t len = d->seg[b - 1].e - d->seg[a].s + 1u;
        if (d->seg[a].s >= d->cnt_start + KWS_NEW_AFTER && len >= KWS_NEW_MIN) {
            *start = d->seg[a].s;
            return true;
        }
    }
    return false;
}

/* an accepted hit: count it, SOS when there are enough in the time window */
static void count_hit(kws_det_t *d, uint32_t now, int *ev, int calls)
{
    const kws_model_t *m = d->m;
    uint32_t win = d->need_ms ? d->need_ms : m->hits_ms;
    int i, j, n = d->need_hits > 0 ? d->need_hits : m->hits > 0 ? m->hits : 3;
    *ev |= KWS_EV_HIT;
    d->n_rej = 0;                               /* v1.6f: an accepted call ends any penalty */
    d->sound_counted++;
    for (i = 0, j = 0; i < d->n_hits; i++) {
        if (now - d->hits[i] < win) { d->hits[j++] = d->hits[i]; }
    }
    d->n_hits = j;
    for (i = 0; i < (calls > 0 ? calls : 1); i++) {
        if (d->n_hits < KWS_MAX_HITS) { d->hits[d->n_hits++] = now; }
        d->hit_count = d->n_hits;
        if (d->n_hits >= n) {
            *ev |= KWS_EV_SOS;
            d->n_hits = 0;
            d->after_sos = true;                    /* v1.6: only a drop of the score re-arms */
            break;
        }
    }
}

/* v1.6e: only a CLEAR miss (KWS_REJ_CLEAR below the keyword's own value)
 * counts towards the penalty - the owner's near misses do not, so the owner
 * is never locked out by his own retries */
static void note_reject(kws_det_t *d, uint32_t now, int cls)
{
    if (d->hit_spk <= -2.0f || d->hit_spk >= kws_voice_need(d, cls) - KWS_REJ_CLEAR) { return; }
    if (d->n_rej == KWS_MAX_HITS) {
        memmove(&d->rej[0], &d->rej[1], sizeof(d->rej[0]) * (KWS_MAX_HITS - 1));
        memmove(&d->rej_cls[0], &d->rej_cls[1], sizeof(d->rej_cls[0]) * (KWS_MAX_HITS - 1));
        d->n_rej--;
    }
    d->rej_cls[d->n_rej] = (int8_t)cls;
    d->rej[d->n_rej++] = now;
}

/* v1.6f: the extra a check of keyword k needs: only that keyword's clear
 * misses count (an imitated "help" does not make the owner's "SOS" harder) */
static float rej_pen(const kws_det_t *d, int k)
{
    int i, n = 0;
    for (i = 0; i < d->n_rej; i++) { if (d->rej_cls[i] == k) { n++; } }
    return (k >= 0 && k < KWS_MAX_NCLS && d->rej_step[k] > 0.0f ? d->rej_step[k] : KWS_REJ_PENALTY) *
           (float)(n < KWS_REJ_MAX ? n : KWS_REJ_MAX);
}

/* v1.6.7: the probability keyword k needs (base = thr[sens]); drop = the
 * relaxation this call would get, ignored for a no_relax keyword */
static float kthr(const kws_det_t *d, int k, float base, float drop)
{
    float t = base + d->m->cls_off[k];
    if (k >= 0 && k < KWS_MAX_NCLS) {
        t += d->cls_extra[k];
        if (d->no_relax[k]) { drop = 0.0f; }
    }
    return t - drop;
}

/* ---- v1.6.1: words one by one (see KWS_DIP_DB) ---- */
static void word_push(kws_det_t *d, uint32_t s, uint32_t e)
{
    if (e < s || e - s + 1u < KWS_WORD_MIN) { return; }
    if (d->nword == KWS_MAX_WORD) { memmove(&d->word[0], &d->word[1], sizeof(d->word[0]) * (KWS_MAX_WORD - 1)); d->nword--; }
    d->word[d->nword].s = s;
    d->word[d->nword].e = e;
    d->word[d->nword].cls = -1;
    d->word[d->nword].checked = 0;
    d->word[d->nword].p = 0.0f;
    d->nword++;
}

/* checks word i alone: the word at the end of a window, quiet frames around
 * it (d->win holds the window of this run) */
static void word_check(kws_det_t *d, int i, uint32_t now)
{
    const kws_model_t *m = d->m;
    int T = m->T, L, dst, r, k;
    uint32_t base = d->fno + 1u - (uint32_t)T, s = d->word[i].s, e = d->word[i].e;
    float probs[KWS_MAX_NCLS], thr = m->thr[d->sens > 2 ? 2 : d->sens], bm = -1e9f;

    if (e - s + 1u > (uint32_t)(T - KWS_WORD_PAD - 4)) { s = e + 1u - (uint32_t)(T - KWS_WORD_PAD - 4); }
    L = (int)(e - s + 1u);
    dst = T - KWS_WORD_PAD - L;
    for (r = 0; r < T; r++) {
        const float *src = r >= dst && r < dst + L ? d->win + (size_t)(s - base + (uint32_t)(r - dst)) * KWS_BANDS : d->qfeat;
        memcpy(d->win2 + (size_t)r * KWS_BANDS, src, sizeof(float) * KWS_BANDS);
    }
    if (d->fake_run) { d->fake_run(d->fake_ctx, now, d->win2, true, probs); }
    else if (!kws_engine_run(m, d->win2, d->work2, probs, NULL)) { return; }
    d->word[i].checked = 1;
    d->word[i].cls = -1;
    d->word[i].p = 0.0f;
    for (k = 0; k < m->ncls; k++) {
        float mg;
        if (!kws_is_keyword(m, k)) { continue; }
        mg = probs[k] - kthr(d, k, thr, 0.0f);
        if (mg > bm) {
            float slack = k < KWS_MAX_NCLS && d->no_relax[k] ? 0.0f : KWS_WORD_SLACK;   /* v1.6.7 */
            bm = mg; d->word[i].p = probs[k];
            d->word[i].cls = mg >= -slack ? (int8_t)k : -1;
        }
    }
    if (d->wc_n < 4) {
        d->wc[d->wc_n].len = e - d->word[i].s + 1u;
        d->wc[d->wc_n].cls = d->word[i].cls;
        d->wc[d->wc_n].p = d->word[i].p;
        d->wc_n++;
    }
}

/* how many words since the fresh count are keywords on their own (checks the
 * new ones; at most 3 checks per run) */
static int word_calls(kws_det_t *d, uint32_t now)
{
    int i, n = 0, checks = 0;
    uint32_t base = d->fno + 1u - (uint32_t)d->m->T, last = 0;
    bool have = false;
    for (i = 0; i < d->nword; i++) {
        if (d->word[i].s < d->ep_from) { continue; }
        if (!d->word[i].checked) {
            if (d->word[i].s < base || checks >= 3) { continue; }
            word_check(d, i, now);
            checks++;
        }
        if (d->word[i].cls >= 0 && (!have || d->word[i].s >= last + KWS_WORD_AFTER)) {
            n++;
            have = true;
            last = d->word[i].s;
        }
    }
    return n;
}

float kws_voice_need(const kws_det_t *d, int k)
{
    return k >= 0 && k < KWS_MAX_NCLS && d->verify_thr_k[k] > 0.0f ? d->verify_thr_k[k] : d->verify_thr;
}

static void decide(kws_det_t *d, uint32_t now, int *ev)
{
    const kws_model_t *m = d->m;
    int k, best = -1, i, calls;
    uint32_t start = 0;
    float bestp = 0.0f, bestm = -1e9f, thr = m->thr[d->sens > 2 ? 2 : d->sens];
    float cdrop = 0.0f;                         /* v1.6.7: KWS_CONFIRM_DROP, per keyword in kthr() */
    uint32_t win = d->need_ms ? d->need_ms : m->hits_ms;
    float vdrop = 0.0f;                         /* v1.6: a confirming call needs less voice match */

    /* v1.6: a call was already counted and the time window is still open: the
     * next one is a confirmation and needs a little less */
    for (i = 0; i < d->n_hits; i++) {
        if (now - d->hits[i] < win) { cdrop = KWS_CONFIRM_DROP; vdrop = KWS_VOICE_CONFIRM; break; }
    }
    /* v1.6d: every voice rejected in the last KWS_REJ_MS makes the next check
     * stricter (someone imitating the owner tries again and again) */
    for (i = 0, k = 0; i < d->n_rej; i++) {
        if (now - d->rej[i] < KWS_REJ_MS) { d->rej_cls[k] = d->rej_cls[i]; d->rej[k++] = d->rej[i]; }
    }
    d->n_rej = k;
    for (k = 0; k < m->ncls; k++) {
        float s = 0.0f, lo = 1e9f, margin;
        for (i = 0; i < d->n_hist; i++) { s += d->hist[i][k]; if (d->hist[i][k] < lo) { lo = d->hist[i][k]; } }
        if (d->sens >= KWS_SENS_HIGH && d->n_hist == KWS_SMOOTH) {
            d->smooth[k] = (s - lo) / (float)(KWS_SMOOTH - 1);   /* v1.5.1: best 2 of the last 3 */
        } else {
            d->smooth[k] = d->n_hist ? s / (float)d->n_hist : 0.0f;
        }
        margin = d->smooth[k] - kthr(d, k, thr, cdrop);       /* v1.4.1: per class threshold */
        if (kws_is_keyword(m, k) && margin >= 0.0f && margin > bestm) { best = k; bestp = d->smooth[k]; bestm = margin; }
    }
    if (d->retry_left > 0) {
        /* v1.5.1: the voice did not match on the last window: try this newer one */
        int c = d->retry_cls;
        d->retry_left--;
        if (d->verify && d->only_me && d->smooth[c] >= kthr(d, c, thr, cdrop) - KWS_RETRY_DROP) {
            float sim = d->verify(d->verify_ctx, d->win, m->T, c);
            if (sim > d->hit_spk) { d->hit_spk = sim; }
            d->hit_need = kws_voice_need(d, c) - vdrop + rej_pen(d, c);
            if (sim <= -2.0f || sim >= d->hit_need) {
                d->retry_left = 0;
                if (d->smooth[c] > d->hit_score) { d->hit_score = d->smooth[c]; }
                count_hit(d, now, ev, d->retry_n);
                return;
            }
        } else {
            d->retry_left = 0;                      /* the word has gone */
        }
        if (d->retry_left == 0) { *ev |= KWS_EV_REJECT; note_reject(d, now, c); }
        return;
    }
    {   /* v1.6: re-armed when every keyword dropped clearly below its threshold */
        float base = m->thr[d->sens > 2 ? 2 : d->sens], top = -1e9f;
        for (k = 0; k < m->ncls; k++) {
            if (kws_is_keyword(m, k) && d->smooth[k] - kthr(d, k, base, 0.0f) > top) { top = d->smooth[k] - kthr(d, k, base, 0.0f); }
        }
        if (top < -KWS_REARM_DROP) { d->armed = true; d->after_sos = false; }
    }
    if (best < 0) { return; }
    /* v1.6 (v1.5: one hit, then 1.2 s pause - the end of the same call counted again) */
    if (d->armed) {
        uint32_t w0 = d->fno >= (uint32_t)m->T ? d->fno + 1u - (uint32_t)m->T : 0u;
        calls = 1;
        d->hit_how = 0;
        start = counted_start(d);
        d->ep_from = d->cnt_fno + 1u > w0 ? d->cnt_fno + 1u : w0;   /* v1.6.1 */
        d->ep_calls = 0;
    } else if (!d->after_sos && d->words_on && d->work2 && d->win2 && d->need_hits == 2 &&
               word_calls(d, now) > d->ep_calls) {
        calls = 1;                              /* v1.6.1: the next word alone is a keyword too */
        d->hit_how = 3;
        start = counted_start(d);
    } else if (!d->after_sos && new_sound(d, &start) && d->probs[best] >= kthr(d, best, thr, cdrop)) {
        calls = 1;                              /* a quick second call */
        d->hit_how = 1;
    } else if (!d->after_sos && d->last_hit && now - d->last_hit >= KWS_SUSTAIN_MS) {
        calls = 1;                              /* calling on without a pause */
        d->hit_how = 2;
        start = counted_start(d);
    } else {
        if (!d->same_told) { d->same_told = true; *ev |= KWS_EV_SAME; d->hit_cls = best; d->hit_score = bestp; }
        return;
    }
    d->armed = false;
    d->same_told = false;
    d->ep_calls++;
    d->cnt_fno = d->fno;
    d->cnt_start = start;
    d->last_hit = now ? now : 1;
    d->hit_cls = best;
    d->hit_score = bestp;
    d->hit_spk = -2.0f;
    if (d->verify) {
        /* separate speaker model - v1.5: only when Only My Voice is on. It takes
         * much longer than the keyword CNN; running it on every hit made the
         * listening pause and miss the next calls. */
        if (d->only_me) {
            d->hit_need = kws_voice_need(d, best) - vdrop + rej_pen(d, best);
            d->hit_spk = d->verify(d->verify_ctx, d->win, m->T, best);
            if (d->hit_spk > -2.0f && d->hit_spk < d->hit_need && KWS_VERIFY_TRIES > 1) {
                d->retry_cls = best;                /* not yet: check again on the next runs */
                d->retry_n = calls;
                d->retry_left = KWS_VERIFY_TRIES - 1;
                return;
            }
        }
    } else if (m->spk_dim > 0 && d->voice) {
        /* the run of the last few where this keyword was clearest */
        int at = 0;
        float p = -1.0f;
        for (i = 0; i < d->n_hist; i++) { if (d->hist[i][best] > p) { p = d->hist[i][best]; at = i; } }
        d->hit_spk = kws_voice_similarity(d->voice, m->spk_dim, best, d->emb_hist[at]);
    }
    if (d->only_me && d->hit_spk > -2.0f && d->hit_spk < (d->verify ? d->hit_need : m->spk_thr)) {
        *ev |= KWS_EV_REJECT;
        note_reject(d, now, best);
        return;
    }
    count_hit(d, now, ev, calls);
}

/* v1.6: splits the sound into segments by loudness. Loud = above the gate,
 * 12 dB above the background and within 30 dB of the recent loudest; a
 * segment ends after KWS_SEG_END quiet frames. */
static void seg_frame(kws_det_t *d, float db)
{
    const kws_model_t *m = d->m;
    float lim;
    int i;
    bool loud;

    d->fno++;
    if (d->floor_db <= -119.0f || db < d->floor_db) { d->floor_db = db; }
    else { d->floor_db += 0.02f; }                  /* follows a rising background at 1 dB/s */
    d->peak_db = db > d->peak_db - 0.5f ? db : d->peak_db - 0.5f;
    lim = d->floor_db + 12.0f;
    if (d->peak_db - 30.0f > lim) { lim = d->peak_db - 30.0f; }
    if (KWS_GATE_DBFS > lim) { lim = KWS_GATE_DBFS; }
    loud = db >= lim;
    if (loud) {
        if (!d->seg_open) {
            if (d->nseg == KWS_MAX_SEG) {
                memmove(&d->seg[0], &d->seg[1], sizeof(d->seg[0]) * (KWS_MAX_SEG - 1));
                d->nseg--;
            }
            d->seg[d->nseg].s = d->seg[d->nseg].e = d->fno;
            d->nseg++;
            d->seg_open = true;
        } else {
            d->seg[d->nseg - 1].e = d->fno;
        }
        d->seg_quiet = 0;
    } else if (d->seg_open && ++d->seg_quiet >= KWS_SEG_END) {
        d->seg_open = false;
        if (d->word_open) { word_push(d, d->w_s, d->seg[d->nseg - 1].e); d->word_open = false; }
        if (d->seg[d->nseg - 1].e - d->seg[d->nseg - 1].s + 1u < KWS_SEG_MIN) {
            d->nseg--;                              /* a click */
        }
    }
    /* v1.6.1: words inside the sound - a dip of the loudness between two words */
    if (d->seg_open) {
        if (!d->word_open) {
            if (loud) { d->word_open = true; d->w_s = d->fno; d->w_max = d->w_min = db; d->w_minf = d->fno; }
        } else {
            if (db > d->w_max) { d->w_max = db; d->w_min = db; d->w_minf = d->fno; }
            else if (db < d->w_min) { d->w_min = db; d->w_minf = d->fno; }
            if (loud && d->w_max - d->w_min >= KWS_DIP_DB && db - d->w_min >= KWS_DIP_RISE &&
                d->w_minf >= d->w_s + KWS_WORD_MIN) {
                word_push(d, d->w_s, d->w_minf - 1u);
                d->w_s = d->w_minf + 1u <= d->fno ? d->w_minf + 1u : d->fno;
                d->w_max = d->w_min = db;
                d->w_minf = d->fno;
            }
        }
    }
    for (i = 0; i < d->nword && d->word[i].e + 2u * (uint32_t)m->T < d->fno; i++) { }
    if (i > 0) {
        memmove(&d->word[0], &d->word[i], sizeof(d->word[0]) * (size_t)(d->nword - i));
        d->nword -= i;
    }
    /* forget segments long out of the window */
    for (i = 0; i < d->nseg && d->seg[i].e + 3u * (uint32_t)m->T < d->fno && !(d->seg_open && i == d->nseg - 1); i++) { }
    if (i > 0) {
        memmove(&d->seg[0], &d->seg[i], sizeof(d->seg[0]) * (size_t)(d->nseg - i));
        d->nseg -= i;
    }
}

int kws_det_frame(kws_det_t *d, const float *feat, float dbfs, uint32_t now_ms)
{
    const kws_model_t *m = d->m;
    int ev = 0;

    if (!kws_model_ok(m)) { return 0; }
    memcpy(d->ring + d->head * KWS_BANDS, feat, sizeof(float) * KWS_BANDS);
    d->head = (d->head + 1) % m->T;
    if (d->filled < m->T) { d->filled++; }
    if (d->since >= 0 && d->since < (1 << 20)) { d->since++; }
    seg_frame(d, dbfs);
    if (d->fno == 1u || dbfs < d->floor_db + 6.0f) { memcpy(d->qfeat, feat, sizeof(d->qfeat)); }
    if (d->sound_from && !d->seg_open && (d->nseg == 0 || d->seg[d->nseg - 1].e + 65u < d->fno)) {
        ev |= KWS_EV_SOUND;                         /* 1.3 s after its end: reported once */
        d->sound_from = 0;
    }
    d->quiet = dbfs >= KWS_GATE_DBFS ? 0 : (d->quiet < (1 << 20) ? d->quiet + 1 : d->quiet);
    if (d->filled < m->T || ++d->step < (d->step_frames > 0 ? d->step_frames : KWS_STEP)) { return ev; }
    d->step = 0;
    if (d->quiet >= m->T) {
        /* nothing loud in the whole window: no run; what was heard before fades
         * out - v1.6: and the next call is a fresh one (re-armed) */
        d->n_hist = d->hist_pos = 0;
        memset(d->smooth, 0, sizeof(d->smooth));
        d->armed = true;
        d->after_sos = false;
        return ev;
    }
    kws_det_window(d, d->win);
    /* v1.5.1: the engine reuses our last run when nobody else used the work memory since */
    if (d->fake_run) {
        d->fake_run(d->fake_ctx, now_ms, d->win, false, d->probs);   /* (tests: scripted probabilities) */
    } else {
        if (!kws_engine_run_shift(m, d->win, d->work, d->probs, m->spk_dim > 0 ? d->emb : NULL,
                                  d->since >= 0 && kws_engine_stamp(d->work) == d->eng_stamp ? d->since : -1)) { return ev; }
        d->eng_stamp = kws_engine_stamp(d->work);
    }
    d->since = 0;
    ev |= KWS_EV_RAN;
    memcpy(d->hist[d->hist_pos], d->probs, sizeof(float) * (size_t)m->ncls);
    if (m->spk_dim > 0) { memcpy(d->emb_hist[d->hist_pos], d->emb, sizeof(float) * (size_t)m->spk_dim); }
    d->hist_pos = (d->hist_pos + 1) % KWS_SMOOTH;
    if (d->n_hist < KWS_SMOOTH) { d->n_hist++; }
    d->wc_n = 0;
    decide(d, now_ms, &ev);
    if (d->wc_n) { ev |= KWS_EV_WORDS; }
    {   /* diagnostics: best scores while a sound goes on */
        int k;
        if (!d->sound_from && d->nseg && d->seg[d->nseg - 1].e + 2u >= d->fno) {
            d->sound_from = d->seg[d->nseg - 1].s;
            memset(d->sound_best, 0, sizeof(d->sound_best));
            d->sound_counted = 0;
        }
        if (d->sound_from) {
            for (k = 0; k < m->ncls; k++) { if (d->smooth[k] > d->sound_best[k]) { d->sound_best[k] = d->smooth[k]; } }
        }
    }
    return ev;
}

float kws_best_window(const kws_model_t *m, float *work, const float *feats, int n, int k,
                      float *best_emb, int *best_at)
{
    float probs[KWS_MAX_NCLS], emb[KWS_MAX_SPK], best = -1.0f;
    int s;

    if (!kws_model_ok(m) || n < m->T) { return -1.0f; }
    for (s = 0; s + m->T <= n; s += 2) {            /* 40 ms steps */
        if (!kws_engine_run(m, feats + (size_t)s * KWS_BANDS, work, probs, m->spk_dim > 0 ? emb : NULL)) { break; }
        if (probs[k] > best) {
            best = probs[k];
            if (best_at) { *best_at = s; }
            if (best_emb && m->spk_dim > 0) { memcpy(best_emb, emb, sizeof(float) * (size_t)m->spk_dim); }
        }
    }
    return best;
}

void kws_best_windows(const kws_model_t *m, float *work, const float *feats, int n, int step,
                      float *best, int *at, float *any, int *any_at)
{
    float probs[KWS_MAX_NCLS], sum;
    int s, k, first = 1;

    if (!kws_model_ok(m) || !best || !at) { return; }
    for (k = 0; k < m->ncls; k++) { best[k] = -1.0f; at[k] = 0; }
    if (any) { *any = -1.0f; }
    if (any_at) { *any_at = 0; }
    if (step < 1) { step = 1; }
    for (s = 0; s + m->T <= n; s += step) {
        if (!kws_engine_run_shift(m, feats + (size_t)s * KWS_BANDS, work, probs, NULL, first ? -1 : step)) { break; }
        first = 0;
        for (k = 0, sum = 0.0f; k < m->ncls; k++) {
            if (probs[k] > best[k]) { best[k] = probs[k]; at[k] = s; }
            if (kws_is_keyword(m, k)) { sum += probs[k]; }
        }
        if (any && sum > *any) { *any = sum; if (any_at) { *any_at = s; } }
    }
}

int kws_enrol_class(const kws_model_t *m, int take)
{
    int k, n = 0, kw[KWS_MAX_NCLS];
    if (!kws_model_ok(m)) { return -1; }
    for (k = 0; k < m->ncls; k++) { if (kws_is_keyword(m, k)) { kw[n++] = k; } }
    return n ? kw[take % n] : -1;
}

float kws_dbfs(const int16_t *pcm, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++) { s += (double)pcm[i] * pcm[i]; }
    if (n <= 0 || s <= 0.0) { return -120.0f; }
    return (float)(10.0 * log10(s / n / (32768.0 * 32768.0)));
}
