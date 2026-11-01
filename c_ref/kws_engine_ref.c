/* kws_engine_ref.c - (tests) the straightforward v1.5 CNN, to check the fast Ai/kws_engine.c */
#include "kws_engine.h"
#include <math.h>
#include <string.h>

static bool ref_model_ok(const kws_model_t *m)
{
    return m && m->ncls >= 2 && m->ncls <= KWS_MAX_NCLS && m->T >= 8 && m->T <= KWS_MAX_T && m->B == 40 &&
           m->c1 >= 1 && m->c1 <= KWS_MAX_C1 && m->c2 >= 1 && m->c2 <= KWS_MAX_C2 &&
           m->c3 >= 1 && m->c3 <= KWS_MAX_C3 && m->hid >= 1 && m->hid <= KWS_MAX_HID &&
           m->spk_dim >= 0 && m->spk_dim <= KWS_MAX_SPK && (m->spk_dim == 0 || m->w6) &&
           m->w1 && m->w2 && m->w3 && m->w4 && m->w5 && m->in_mean && m->in_std && m->labels;
}

/* v1.5: work memory laid out for the fast convolution below. Every
 * activation map has a zero border of 1 (the "padding 1"), so the inner loop
 * runs over a whole row without any bounds checks:
 *   in   (T+2) x (B+2)            normalised input
 *   a1   c1 x (h1+2) x (w1+2)     conv1 + ReLU + pool
 *   a2   c2 x (h2+2) x (w2+2)     conv2 + ReLU + pool
 *   acc  T x B                    one output plane being summed up */
static size_t ref_engine_work(const kws_model_t *m)
{
    int h1 = m->T / 2, w1 = m->B / 2, h2 = h1 / 2, w2 = w1 / 2;
    return (size_t)(m->T + 2) * (m->B + 2) + (size_t)m->c1 * (h1 + 2) * (w1 + 2) +
           (size_t)m->c2 * (h2 + 2) * (w2 + 2) + (size_t)m->T * m->B;
}

/* acc[h][w] = sum over input channels and the 3x3 taps (no scale, no bias).
 * in: cin padded planes of (h+2) x (w+2); k: this output channel's [cin][3][3].
 * The order of the additions per output value is the same as in v1.4
 * (channel, row, column), so the results are the same; v1.4 computed every
 * value separately with bounds checks and was ~5x slower (the ESP32-S3 could
 * not keep up with the microphone). */
static void conv_plane(const float *in, int cin, int h, int w, const int8_t *k, float *acc)
{
    int c, ky, kx, y, x, W = w + 2;
    memset(acc, 0, sizeof(float) * (size_t)h * w);
    for (c = 0; c < cin; c++) {
        const float *ip = in + (size_t)c * (h + 2) * W;
        const int8_t *kk = k + c * 9;
        for (ky = 0; ky < 3; ky++) {
            for (kx = 0; kx < 3; kx++) {
                float wv = (float)kk[ky * 3 + kx];
                if (wv == 0.0f) { continue; }
                for (y = 0; y < h; y++) {
                    const float *r = ip + (size_t)(y + ky) * W + kx;
                    float *a = acc + (size_t)y * w;
                    for (x = 0; x + 4 <= w; x += 4) {       /* unrolled: fewer loop steps on the Xtensa */
                        float a0 = a[x] + wv * r[x], a1 = a[x + 1] + wv * r[x + 1];
                        float a2 = a[x + 2] + wv * r[x + 2], a3 = a[x + 3] + wv * r[x + 3];
                        a[x] = a0; a[x + 1] = a1; a[x + 2] = a2; a[x + 3] = a3;
                    }
                    for (; x < w; x++) { a[x] += wv * r[x]; }
                }
            }
        }
    }
}

/* conv 3x3 + ReLU + max-pool 2x2: out = cout padded planes of (h/2+2) x (w/2+2) */
static void conv_pool(const float *in, int cin, int h, int w, const int8_t *k, float s, const float *b,
                      int cout, float *acc, float *out)
{
    int ho = h / 2, wo = w / 2, o, y, x, Wo = wo + 2;
    memset(out, 0, sizeof(float) * (size_t)cout * (ho + 2) * Wo);
    for (o = 0; o < cout; o++) {
        float *po = out + (size_t)o * (ho + 2) * Wo;
        conv_plane(in, cin, h, w, k + (size_t)o * cin * 9, acc);
        for (y = 0; y < ho; y++) {
            const float *r0 = acc + (size_t)(2 * y) * w, *r1 = r0 + w;
            float *q = po + (size_t)(y + 1) * Wo + 1;
            for (x = 0; x < wo; x++) {
                float m = r0[2 * x];
                if (r0[2 * x + 1] > m) { m = r0[2 * x + 1]; }
                if (r1[2 * x] > m) { m = r1[2 * x]; }
                if (r1[2 * x + 1] > m) { m = r1[2 * x + 1]; }
                m = m * s + b[o];                   /* (s > 0: max before scaling is the same) */
                q[x] = m > 0.0f ? m : 0.0f;         /* ReLU */
            }
        }
    }
}

static bool ref_engine_run(const kws_model_t *m, const float *feat, float *work, float *probs, float *emb)
{
    int T, B, h1, w1, h2, w2, i, j, o;
    float *in, *a1, *a2, *acc, g[KWS_MAX_C3], hd[KWS_MAX_HID], mx, sum;

    if (!ref_model_ok(m) || !feat || !work || !probs) { return false; }
    T = m->T; B = m->B;
    h1 = T / 2; w1 = B / 2; h2 = h1 / 2; w2 = w1 / 2;
    in = work;
    a1 = in + (size_t)(T + 2) * (B + 2);
    a2 = a1 + (size_t)m->c1 * (h1 + 2) * (w1 + 2);
    acc = a2 + (size_t)m->c2 * (h2 + 2) * (w2 + 2);

    memset(in, 0, sizeof(float) * (size_t)(T + 2) * (B + 2));
    for (i = 0; i < T; i++) {
        float *r = in + (size_t)(i + 1) * (B + 2) + 1;
        for (j = 0; j < B; j++) { r[j] = (feat[i * B + j] - m->in_mean[j]) / m->in_std[j]; }
    }
    conv_pool(in, 1, T, B, m->w1, m->s1, m->b1, m->c1, acc, a1);
    conv_pool(a1, m->c1, h1, w1, m->w2, m->s2, m->b2, m->c2, acc, a2);
    /* conv3 + ReLU + global average */
    for (o = 0; o < m->c3; o++) {
        float sacc = 0.0f;
        conv_plane(a2, m->c2, h2, w2, m->w3 + (size_t)o * m->c2 * 9, acc);
        for (i = 0; i < h2 * w2; i++) {
            float v = acc[i] * m->s3 + m->b3[o];
            if (v > 0.0f) { sacc += v; }
        }
        g[o] = sacc / (float)(h2 * w2);
    }
    /* keyword head */
    for (o = 0; o < m->hid; o++) {
        const int8_t *r = m->w4 + (size_t)o * m->c3;
        float a = 0.0f;
        for (i = 0; i < m->c3; i++) { a += g[i] * (float)r[i]; }
        a = a * m->s4 + m->b4[o];
        hd[o] = a > 0.0f ? a : 0.0f;
    }
    mx = -1e30f;
    for (o = 0; o < m->ncls; o++) {
        const int8_t *r = m->w5 + (size_t)o * m->hid;
        float a = 0.0f;
        for (i = 0; i < m->hid; i++) { a += hd[i] * (float)r[i]; }
        probs[o] = a * m->s5 + m->b5[o];
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
            float a = 0.0f;
            for (i = 0; i < m->c3; i++) { a += g[i] * (float)r[i]; }
            emb[o] = a * m->s6 + m->b6[o];
            len += emb[o] * emb[o];
        }
        len = sqrtf(len) + 1e-9f;
        for (o = 0; o < m->spk_dim; o++) { emb[o] /= len; }
    }
    return true;
}


