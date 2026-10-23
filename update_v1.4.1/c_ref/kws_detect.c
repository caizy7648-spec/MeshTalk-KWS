/* kws_detect.c - voice SOS decision logic. See kws_detect.h. */
#include "kws_detect.h"
#include <math.h>
#include <string.h>

void kws_det_reset(kws_det_t *d)
{
    d->head = d->filled = d->step = 0;
    d->quiet = 1 << 20;
    d->n_hist = d->hist_pos = 0;
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

static void decide(kws_det_t *d, uint32_t now, int *ev)
{
    const kws_model_t *m = d->m;
    int k, best = -1, i, j, n;
    float bestp = 0.0f, bestm = -1e9f, thr = m->thr[d->sens > 2 ? 2 : d->sens];

    for (k = 0; k < m->ncls; k++) {
        float s = 0.0f, margin;
        for (i = 0; i < d->n_hist; i++) { s += d->hist[i][k]; }
        d->smooth[k] = d->n_hist ? s / (float)d->n_hist : 0.0f;
        margin = d->smooth[k] - (thr + m->cls_off[k]);        /* v1.4.1: per class threshold */
        if (kws_is_keyword(m, k) && margin >= 0.0f && margin > bestm) { best = k; bestp = d->smooth[k]; bestm = margin; }
    }
    if (best < 0 || (d->last_hit && now - d->last_hit < KWS_REFRACT_MS)) { return; }
    d->last_hit = now ? now : 1;
    d->hit_cls = best;
    d->hit_score = bestp;
    d->hit_spk = -2.0f;
    if (d->verify) {
        d->hit_spk = d->verify(d->verify_ctx, d->win, m->T, best);       /* separate speaker model */
    } else if (m->spk_dim > 0 && d->voice) {
        /* the run of the last few where this keyword was clearest */
        int at = 0;
        float p = -1.0f;
        for (i = 0; i < d->n_hist; i++) { if (d->hist[i][best] > p) { p = d->hist[i][best]; at = i; } }
        d->hit_spk = kws_voice_similarity(d->voice, m->spk_dim, best, d->emb_hist[at]);
    }
    if (d->only_me && d->hit_spk > -2.0f && d->hit_spk < (d->verify ? d->verify_thr : m->spk_thr)) {
        *ev |= KWS_EV_REJECT;
        return;
    }
    *ev |= KWS_EV_HIT;
    /* hits inside the time window */
    for (i = 0, j = 0; i < d->n_hits; i++) {
        if (now - d->hits[i] < m->hits_ms) { d->hits[j++] = d->hits[i]; }
    }
    d->n_hits = j;
    if (d->n_hits < KWS_MAX_HITS) { d->hits[d->n_hits++] = now; }
    n = m->hits > 0 ? m->hits : 3;
    d->hit_count = d->n_hits;
    if (d->n_hits >= n) {
        *ev |= KWS_EV_SOS;
        d->n_hits = 0;
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
    d->quiet = dbfs >= KWS_GATE_DBFS ? 0 : (d->quiet < (1 << 20) ? d->quiet + 1 : d->quiet);
    if (d->filled < m->T || ++d->step < KWS_STEP) { return 0; }
    d->step = 0;
    if (d->quiet >= m->T) {
        /* nothing loud in the whole window: no run; what was heard before fades out */
        d->n_hist = d->hist_pos = 0;
        return 0;
    }
    kws_det_window(d, d->win);
    if (!kws_engine_run(m, d->win, d->work, d->probs, m->spk_dim > 0 ? d->emb : NULL)) { return 0; }
    ev |= KWS_EV_RAN;
    memcpy(d->hist[d->hist_pos], d->probs, sizeof(float) * (size_t)m->ncls);
    if (m->spk_dim > 0) { memcpy(d->emb_hist[d->hist_pos], d->emb, sizeof(float) * (size_t)m->spk_dim); }
    d->hist_pos = (d->hist_pos + 1) % KWS_SMOOTH;
    if (d->n_hist < KWS_SMOOTH) { d->n_hist++; }
    decide(d, now_ms, &ev);
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
