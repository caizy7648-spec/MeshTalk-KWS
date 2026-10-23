/* spk_engine.c - separate speaker (voice print) model, implements
 * spk_engine.h with the model in spk_model.c (MeshTalk-KWS phase B).
 * Plain C99, no libraries; runs once per keyword hit. */
#include "spk_engine.h"
#include "spk_model.h"
#include <math.h>

#define STD_EPS 1e-5f

static const spk_model_t *M = &spk_model;

static int max_ch(void)
{
    int i, m = SPK_BANDS;
    for (i = 0; i < M->nlayers; i++) { if (M->ch[i] > m) { m = M->ch[i]; } }
    return m;
}

bool        spk_available(void)   { return M->nlayers > 0 && M->dim > 0; }
const char *spk_name(void)        { return M->name; }
uint32_t    spk_id(void)          { return M->id; }
int         spk_dim(void)         { return M->dim; }
int         spk_frames(void)      { return M->frames; }
float       spk_threshold(void)   { return M->threshold; }
int         spk_enrol_takes(void) { return M->enrol_takes; }

/* input copy (40 x F) + two activation buffers (max_ch x F) + pooled (2 x last) + embedding */
size_t spk_work(void)
{
    int F = M->frames, last = M->ch[M->nlayers - 1];
    return (size_t)SPK_BANDS * F + 2u * (size_t)max_ch() * F + 2u * (size_t)last + (size_t)M->dim;
}

/* one Conv1d (+ folded BN) + ReLU: in [cin][F] -> out [cout][F] */
static void conv(const float *in, int cin, float *out, int cout, int F, int k, int d,
                 const int8_t *w, const float *s, const float *b)
{
    int o, c, j, t, pad = d * (k - 1) / 2;
    for (o = 0; o < cout; o++) {
        float *y = out + (size_t)o * F;
        for (t = 0; t < F; t++) { y[t] = 0.0f; }
        for (c = 0; c < cin; c++) {
            const float *x = in + (size_t)c * F;
            const int8_t *wk = w + ((size_t)o * cin + c) * k;
            for (j = 0; j < k; j++) {
                float wv = (float)wk[j];
                int sh = j * d - pad, t0 = sh < 0 ? -sh : 0, t1 = sh > 0 ? F - sh : F;
                if (wv == 0.0f) { continue; }
                for (t = t0; t < t1; t++) { y[t] += wv * x[t + sh]; }
            }
        }
        for (t = 0; t < F; t++) {
            float v = y[t] * s[o] + b[o];
            y[t] = v > 0.0f ? v : 0.0f;
        }
    }
}

bool spk_embed(const float *feat, int n, float *work, float *emb)
{
    int F = M->frames, i, c, t, cin = SPK_BANDS, last;
    float *x, *a, *bb, *pool, *e, len = 0.0f;

    if (!spk_available() || !feat || !work || !emb || n != F) { return false; }
    x = work;
    a = x + (size_t)SPK_BANDS * F;
    bb = a + (size_t)max_ch() * F;
    last = M->ch[M->nlayers - 1];
    pool = bb + (size_t)max_ch() * F;
    e = pool + 2 * last;
    /* normalise, transpose to [band][frame] */
    for (t = 0; t < F; t++) {
        for (c = 0; c < SPK_BANDS; c++) {
            x[(size_t)c * F + t] = (feat[(size_t)t * SPK_BANDS + c] - M->in_mean[c]) / M->in_std[c];
        }
    }
    {
        const float *in = x;
        float *out = a;
        for (i = 0; i < M->nlayers; i++) {
            conv(in, cin, out, M->ch[i], F, M->k[i], M->d[i], M->w[i], M->s[i], M->b[i]);
            cin = M->ch[i];
            in = out;
            out = (out == a) ? bb : a;
        }
        /* statistics pooling over time */
        for (c = 0; c < last; c++) {
            const float *y = in + (size_t)c * F;
            float mu = 0.0f, v = 0.0f;
            for (t = 0; t < F; t++) { mu += y[t]; }
            mu /= (float)F;
            for (t = 0; t < F; t++) { float dlt = y[t] - mu; v += dlt * dlt; }
            pool[c] = mu;
            pool[last + c] = sqrtf(v / (float)F + STD_EPS);
        }
    }
    for (i = 0; i < M->dim; i++) {
        const int8_t *w = M->fw + (size_t)i * 2 * last;
        float acc = 0.0f;
        for (c = 0; c < 2 * last; c++) { acc += (float)w[c] * pool[c]; }
        e[i] = acc * M->fs[i] + M->fb[i];
        len += e[i] * e[i];
    }
    len = sqrtf(len) + 1e-9f;
    for (i = 0; i < M->dim; i++) { emb[i] = e[i] / len; }
    return true;
}
