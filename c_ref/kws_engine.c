/* kws_engine.c - the voice SOS CNN in plain C. See kws_engine.h. */
#include "kws_engine.h"
#include <math.h>
#include <string.h>

bool kws_model_ok(const kws_model_t *m)
{
    return m && m->ncls >= 2 && m->ncls <= KWS_MAX_NCLS && m->T >= 8 && m->T <= KWS_MAX_T && m->B == 40 &&
           m->c1 >= 1 && m->c1 <= KWS_MAX_C1 && m->c2 >= 1 && m->c2 <= KWS_MAX_C2 &&
           m->c3 >= 1 && m->c3 <= KWS_MAX_C3 && m->hid >= 1 && m->hid <= KWS_MAX_HID &&
           m->spk_dim >= 0 && m->spk_dim <= KWS_MAX_SPK && (m->spk_dim == 0 || m->w6) &&
           m->w1 && m->w2 && m->w3 && m->w4 && m->w5 && m->in_mean && m->in_std && m->labels;
}

size_t kws_engine_work(const kws_model_t *m)
{
    int h1 = m->T / 2, w1 = m->B / 2, h2 = h1 / 2, w2 = w1 / 2;
    return (size_t)m->T * m->B + (size_t)m->c1 * h1 * w1 + (size_t)m->c2 * h2 * w2;
}

/* 3x3 convolution (padding 1) of one output channel at (y, x) */
static inline float conv_at(const float *in, int cin, int h, int w, const int8_t *k, int y, int x)
{
    float acc = 0.0f;
    int c, dy, dx;
    for (c = 0; c < cin; c++) {
        const float *p = in + (size_t)c * h * w;
        const int8_t *kk = k + c * 9;
        for (dy = -1; dy <= 1; dy++) {
            int yy = y + dy;
            if (yy < 0 || yy >= h) { continue; }
            for (dx = -1; dx <= 1; dx++) {
                int xx = x + dx;
                if (xx < 0 || xx >= w) { continue; }
                acc += p[yy * w + xx] * (float)kk[(dy + 1) * 3 + dx + 1];
            }
        }
    }
    return acc;
}

/* conv 3x3 + ReLU + max-pool 2x2 in one go: out[cout][h/2][w/2] */
static void conv_pool(const float *in, int cin, int h, int w, const int8_t *k, float s, const float *b,
                      int cout, float *out)
{
    int ho = h / 2, wo = w / 2, o, y, x;
    for (o = 0; o < cout; o++) {
        const int8_t *ko = k + (size_t)o * cin * 9;
        float *po = out + (size_t)o * ho * wo;
        for (y = 0; y < ho; y++) {
            for (x = 0; x < wo; x++) {
                float best = 0.0f;                  /* ReLU: max(0, ...) */
                int i, j;
                for (i = 0; i < 2; i++) {
                    for (j = 0; j < 2; j++) {
                        float v = conv_at(in, cin, h, w, ko, 2 * y + i, 2 * x + j) * s + b[o];
                        if (v > best) { best = v; }
                    }
                }
                po[y * wo + x] = best;
            }
        }
    }
}

bool kws_engine_run(const kws_model_t *m, const float *feat, float *work, float *probs, float *emb)
{
    int T, B, h1, w1, h2, w2, i, j, o, y, x;
    float *in, *a1, *a2, g[KWS_MAX_C3], hd[KWS_MAX_HID], mx, sum;

    if (!kws_model_ok(m) || !feat || !work || !probs) { return false; }
    T = m->T; B = m->B;
    h1 = T / 2; w1 = B / 2; h2 = h1 / 2; w2 = w1 / 2;
    in = work;
    a1 = in + (size_t)T * B;
    a2 = a1 + (size_t)m->c1 * h1 * w1;

    for (i = 0; i < T; i++) {
        for (j = 0; j < B; j++) { in[i * B + j] = (feat[i * B + j] - m->in_mean[j]) / m->in_std[j]; }
    }
    conv_pool(in, 1, T, B, m->w1, m->s1, m->b1, m->c1, a1);
    conv_pool(a1, m->c1, h1, w1, m->w2, m->s2, m->b2, m->c2, a2);
    /* conv3 + ReLU + global average */
    for (o = 0; o < m->c3; o++) {
        const int8_t *ko = m->w3 + (size_t)o * m->c2 * 9;
        float acc = 0.0f;
        for (y = 0; y < h2; y++) {
            for (x = 0; x < w2; x++) {
                float v = conv_at(a2, m->c2, h2, w2, ko, y, x) * m->s3 + m->b3[o];
                if (v > 0.0f) { acc += v; }
            }
        }
        g[o] = acc / (float)(h2 * w2);
    }
    /* keyword head */
    for (o = 0; o < m->hid; o++) {
        const int8_t *r = m->w4 + (size_t)o * m->c3;
        float acc = 0.0f;
        for (i = 0; i < m->c3; i++) { acc += g[i] * (float)r[i]; }
        acc = acc * m->s4 + m->b4[o];
        hd[o] = acc > 0.0f ? acc : 0.0f;
    }
    mx = -1e30f;
    for (o = 0; o < m->ncls; o++) {
        const int8_t *r = m->w5 + (size_t)o * m->hid;
        float acc = 0.0f;
        for (i = 0; i < m->hid; i++) { acc += hd[i] * (float)r[i]; }
        probs[o] = acc * m->s5 + m->b5[o];
        if (probs[o] > mx) { mx = probs[o]; }
    }
    sum = 0.0f;
    for (o = 0; o < m->ncls; o++) { probs[o] = expf(probs[o] - mx); sum += probs[o]; }
    for (o = 0; o < m->ncls; o++) { probs[o] /= sum; }
    /* speaker head */
    if (emb && m->spk_dim > 0) {
        float len = 0.0f;
        for (o = 0; o < m->spk_dim; o++) {
            const int8_t *r = m->w6 + (size_t)o * m->c3;
            float acc = 0.0f;
            for (i = 0; i < m->c3; i++) { acc += g[i] * (float)r[i]; }
            emb[o] = acc * m->s6 + m->b6[o];
            len += emb[o] * emb[o];
        }
        len = sqrtf(len) + 1e-9f;
        for (o = 0; o < m->spk_dim; o++) { emb[o] /= len; }
    }
    return true;
}

const char *kws_label(const kws_model_t *m, int k, char *buf, size_t size)
{
    const char *p = m && m->labels ? m->labels : "";
    size_t n = 0;
    while (k > 0 && *p) { if (*p++ == '|') { k--; } }
    while (*p && *p != '|' && n + 1 < size) { buf[n++] = *p++; }
    if (size) { buf[n] = 0; }
    return buf;
}

bool kws_is_keyword(const kws_model_t *m, int k)
{
    char b[24];
    kws_label(m, k, b, sizeof(b));
    return b[0] && b[0] != '_';
}
