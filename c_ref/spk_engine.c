/* spk_engine.c - separate speaker (voice print) model, implements
 * spk_engine.h with the model in spk_model.c (MeshTalk-KWS phase B).
 * Plain C99, no libraries; runs once per keyword hit. */
#include "spk_engine.h"
#include "spk_model.h"
#include <math.h>
#include <string.h>

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

/* v1.5.1: every activation row has a zero margin of P frames on both sides
 * (P = the widest padding of the layers), so the convolution needs no bounds
 * checks; 2 output channels x 4 frames are summed in registers (each input
 * value and weight is loaded once per block). About 2-3x faster on the
 * ESP32-S3, same results (same order of the additions per output value).
 * Work: input (40 x S) + two activation buffers (max_ch x S), S = F + 2P,
 * + pooled (2 x last) + embedding */
static int pad_max(void)
{
    int i, p = 0;
    for (i = 0; i < M->nlayers; i++) { int q = M->d[i] * (M->k[i] - 1) / 2; if (q > p) { p = q; } }
    return p;
}

size_t spk_work(void)
{
    int S = M->frames + 2 * pad_max(), last = M->ch[M->nlayers - 1];
    return (size_t)SPK_BANDS * S + 2u * (size_t)max_ch() * S + 2u * (size_t)last + (size_t)M->dim;
}

/* one Conv1d (+ folded BN) + ReLU: in [cin][S] -> out [cout][S]; frame t of
 * a row is at index P + t, the margins stay 0 */
static void conv(const float *in, int cin, float *out, int cout, int F, int S, int P, int k, int d,
                 const int8_t *w, const float *s, const float *b)
{
    int o = 0, c, j, t, pad = d * (k - 1) / 2;
    for (; o + 2 <= cout; o += 2) {
        const int8_t *wa = w + (size_t)o * cin * k, *wb = wa + (size_t)cin * k;
        float *ya = out + (size_t)o * S + P, *yb = ya + S;
        for (t = 0; t + 4 <= F; t += 4) {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
            for (c = 0; c < cin; c++) {
                const float *x = in + (size_t)c * S + P + t - pad;
                const int8_t *ka = wa + c * k, *kb = wb + c * k;
                for (j = 0; j < k; j++) {
                    const float *xx = x + j * d;
                    float x0 = xx[0], x1 = xx[1], x2 = xx[2], x3 = xx[3], u = (float)ka[j], v = (float)kb[j];
                    a0 += u * x0; a1 += u * x1; a2 += u * x2; a3 += u * x3;
                    b0 += v * x0; b1 += v * x1; b2 += v * x2; b3 += v * x3;
                }
            }
            a0 = a0 * s[o] + b[o]; a1 = a1 * s[o] + b[o]; a2 = a2 * s[o] + b[o]; a3 = a3 * s[o] + b[o];
            b0 = b0 * s[o + 1] + b[o + 1]; b1 = b1 * s[o + 1] + b[o + 1];
            b2 = b2 * s[o + 1] + b[o + 1]; b3 = b3 * s[o + 1] + b[o + 1];
            ya[t] = a0 > 0.0f ? a0 : 0.0f; ya[t + 1] = a1 > 0.0f ? a1 : 0.0f;
            ya[t + 2] = a2 > 0.0f ? a2 : 0.0f; ya[t + 3] = a3 > 0.0f ? a3 : 0.0f;
            yb[t] = b0 > 0.0f ? b0 : 0.0f; yb[t + 1] = b1 > 0.0f ? b1 : 0.0f;
            yb[t + 2] = b2 > 0.0f ? b2 : 0.0f; yb[t + 3] = b3 > 0.0f ? b3 : 0.0f;
        }
        for (; t < F; t++) {
            float a0 = 0.0f, b0 = 0.0f;
            for (c = 0; c < cin; c++) {
                const float *x = in + (size_t)c * S + P + t - pad;
                for (j = 0; j < k; j++) { a0 += (float)wa[c * k + j] * x[j * d]; b0 += (float)wb[c * k + j] * x[j * d]; }
            }
            a0 = a0 * s[o] + b[o]; b0 = b0 * s[o + 1] + b[o + 1];
            ya[t] = a0 > 0.0f ? a0 : 0.0f; yb[t] = b0 > 0.0f ? b0 : 0.0f;
        }
    }
    for (; o < cout; o++) {                         /* odd channel count: the last one */
        const int8_t *wa = w + (size_t)o * cin * k;
        float *ya = out + (size_t)o * S + P;
        for (t = 0; t < F; t++) {
            float a0 = 0.0f;
            for (c = 0; c < cin; c++) {
                const float *x = in + (size_t)c * S + P + t - pad;
                for (j = 0; j < k; j++) { a0 += (float)wa[c * k + j] * x[j * d]; }
            }
            a0 = a0 * s[o] + b[o];
            ya[t] = a0 > 0.0f ? a0 : 0.0f;
        }
    }
}

bool spk_embed(const float *feat, int n, float *work, float *emb)
{
    int F = M->frames, P = pad_max(), S = F + 2 * P, i, c, t, cin = SPK_BANDS, last;
    float *x, *a, *bb, *pool, *e, len = 0.0f;

    if (!spk_available() || !feat || !work || !emb || n != F) { return false; }
    x = work;
    a = x + (size_t)SPK_BANDS * S;
    bb = a + (size_t)max_ch() * S;
    last = M->ch[M->nlayers - 1];
    pool = bb + (size_t)max_ch() * S;
    e = pool + 2 * last;
    memset(work, 0, sizeof(float) * (size_t)(SPK_BANDS + 2 * max_ch()) * S);     /* the zero margins */
    /* normalise, transpose to [band][frame] */
    for (t = 0; t < F; t++) {
        for (c = 0; c < SPK_BANDS; c++) {
            x[(size_t)c * S + P + t] = (feat[(size_t)t * SPK_BANDS + c] - M->in_mean[c]) / M->in_std[c];
        }
    }
    {
        const float *in = x;
        float *out = a;
        for (i = 0; i < M->nlayers; i++) {
            conv(in, cin, out, M->ch[i], F, S, P, M->k[i], M->d[i], M->w[i], M->s[i], M->b[i]);
            cin = M->ch[i];
            in = out;
            out = (out == a) ? bb : a;
        }
        /* statistics pooling over time */
        for (c = 0; c < last; c++) {
            const float *y = in + (size_t)c * S + P;
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
