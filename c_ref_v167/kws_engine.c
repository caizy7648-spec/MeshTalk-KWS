/* kws_engine.c - the voice SOS CNN in plain C. See kws_engine.h.
 *
 * v1.5.1: about 3-5x faster than v1.5, same results:
 *   1. register blocking: 4 output channels x 2 positions are summed in
 *      registers; every input value and weight is loaded once per block
 *      (v1.5 read and wrote the sum array in memory for every tap);
 *   2. the weights are turned into floats once (kept in the work memory);
 *   3. the window only moves by 12 frames between two runs of the detector:
 *      the rows of every layer that are the same as in the last run (all
 *      except the newest and the ones next to the window edges) are moved,
 *      not computed again (kws_engine_run_shift). The rule which rows can be
 *      reused is exact, so the result is the same as a full run.
 * The additions for every output value keep the order of v1.4 / v1.5
 * (input channel, row, column), only the average of conv3 is summed per row. */
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

/* ---- work memory -------------------------------------------------------
 *   hdr                           (HDR_F floats)
 *   in   (T+2) x (B+2)            normalised input, zero border
 *   a1   c1 x (h1+2) x (w1+2)     conv1 + ReLU + pool, zero border
 *   a2   c2 x (h2+2) x (w2+2)     conv2 + ReLU + pool, zero border
 *   rs   c3 x h2                  conv3: sum of ReLU per output row
 *   ra, rb  R each                two output rows of one layer (all channels)
 *   wf1, wf2, wf3                 float weights in blocks of 4 output channels */
#define HDR_F   16
#define MAGIC   0x4B575331u        /* "KWS1" */

typedef struct {
    uint32_t magic, stamp, macs;
    const kws_model_t *m;
    const int8_t *w1, *w2, *w3;
} hdr_t;

typedef struct {
    int T, B, h1, w1, h2, w2, c1p, c2p, c3p;
    size_t R;
    float *in, *a1, *a2, *rs, *ra, *rb, *wf1, *wf2, *wf3;
} lay_t;

static int up4(int n) { return (n + 3) & ~3; }

static size_t layout(const kws_model_t *m, float *work, lay_t *L)
{
    size_t n = HDR_F, r1, r2, r3;
    float *b = work;
    L->T = m->T; L->B = m->B;
    L->h1 = L->T / 2; L->w1 = L->B / 2; L->h2 = L->h1 / 2; L->w2 = L->w1 / 2;
    L->c1p = up4(m->c1); L->c2p = up4(m->c2); L->c3p = up4(m->c3);
    r1 = (size_t)L->c1p * L->B; r2 = (size_t)L->c2p * L->w1; r3 = (size_t)L->c3p * L->w2;
    L->R = r1 > r2 ? r1 : r2;
    if (r3 > L->R) { L->R = r3; }
#define TAKE(p, cnt) do { L->p = b ? b + n : NULL; n += (cnt); } while (0)
    TAKE(in, (size_t)(L->T + 2) * (L->B + 2));
    TAKE(a1, (size_t)m->c1 * (L->h1 + 2) * (L->w1 + 2));
    TAKE(a2, (size_t)m->c2 * (L->h2 + 2) * (L->w2 + 2));
    TAKE(rs, (size_t)m->c3 * L->h2);
    TAKE(ra, L->R);
    TAKE(rb, L->R);
    TAKE(wf1, (size_t)L->c1p * 1 * 9);
    TAKE(wf2, (size_t)L->c2p * m->c1 * 9);
    TAKE(wf3, (size_t)L->c3p * m->c2 * 9);
#undef TAKE
    return n;
}

size_t kws_engine_work(const kws_model_t *m)
{
    lay_t L;
    return kws_model_ok(m) ? layout(m, NULL, &L) : 0;
}

uint32_t kws_engine_stamp(const float *work)
{
    const hdr_t *h = (const hdr_t *)(const void *)work;
    return work && h->magic == MAGIC ? h->stamp : 0;
}

uint32_t kws_engine_macs(const float *work)
{
    const hdr_t *h = (const hdr_t *)(const void *)work;
    return work && h->magic == MAGIC ? h->macs : 0;
}

/* int8 [cout][cin][3][3] -> float [cout/4][cin][9][4] (missing channels 0) */
static void prep(const int8_t *w, int cout, int cin, float *wf)
{
    int o, c, t;
    memset(wf, 0, sizeof(float) * (size_t)up4(cout) * cin * 9);
    for (o = 0; o < cout; o++) {
        for (c = 0; c < cin; c++) {
            for (t = 0; t < 9; t++) {
                wf[(((size_t)(o / 4) * cin + c) * 9 + t) * 4 + (o & 3)] = (float)w[((size_t)o * cin + c) * 9 + t];
            }
        }
    }
}

/* One output row y of a 3x3 conv (no scale / bias) for all coutp channels.
 * in: cin zero-bordered planes (plane floats apart, row stride W = w + 2);
 * the output row y reads the bordered rows y, y+1, y+2. out: [coutp][w]. */
static void conv_row(const float *in, int cin, size_t plane, int w, int y, const float *wf, int coutp, float *out)
{
    const int W = w + 2;
    int ob, x, c, ky;
    for (ob = 0; ob < coutp; ob += 4) {
        const float *wb = wf + (size_t)ob * cin * 9;
        float *o0 = out + (size_t)ob * w, *o1 = o0 + w, *o2 = o1 + w, *o3 = o2 + w;
        for (x = 0; x + 2 <= w; x += 2) {
            float s00 = 0.0f, s01 = 0.0f, s10 = 0.0f, s11 = 0.0f, s20 = 0.0f, s21 = 0.0f, s30 = 0.0f, s31 = 0.0f;
            const float *wp = wb, *p = in + (size_t)y * W + x;
            for (c = 0; c < cin; c++, p += plane) {
                const float *r = p;
                for (ky = 0; ky < 3; ky++, r += W, wp += 12) {
                    float v0 = r[0], v1 = r[1], v2 = r[2], v3 = r[3], k0, k1, k2, k3;
                    k0 = wp[0]; k1 = wp[1]; k2 = wp[2]; k3 = wp[3];
                    s00 += k0 * v0; s01 += k0 * v1; s10 += k1 * v0; s11 += k1 * v1;
                    s20 += k2 * v0; s21 += k2 * v1; s30 += k3 * v0; s31 += k3 * v1;
                    k0 = wp[4]; k1 = wp[5]; k2 = wp[6]; k3 = wp[7];
                    s00 += k0 * v1; s01 += k0 * v2; s10 += k1 * v1; s11 += k1 * v2;
                    s20 += k2 * v1; s21 += k2 * v2; s30 += k3 * v1; s31 += k3 * v2;
                    k0 = wp[8]; k1 = wp[9]; k2 = wp[10]; k3 = wp[11];
                    s00 += k0 * v2; s01 += k0 * v3; s10 += k1 * v2; s11 += k1 * v3;
                    s20 += k2 * v2; s21 += k2 * v3; s30 += k3 * v2; s31 += k3 * v3;
                }
            }
            o0[x] = s00; o0[x + 1] = s01; o1[x] = s10; o1[x + 1] = s11;
            o2[x] = s20; o2[x + 1] = s21; o3[x] = s30; o3[x + 1] = s31;
        }
        if (x < w) {                                /* odd width: last column */
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            const float *wp = wb, *p = in + (size_t)y * W + x;
            for (c = 0; c < cin; c++, p += plane) {
                const float *r = p;
                for (ky = 0; ky < 3; ky++, r += W, wp += 12) {
                    int kx;
                    for (kx = 0; kx < 3; kx++) {
                        float v = r[kx];
                        s0 += wp[kx * 4] * v; s1 += wp[kx * 4 + 1] * v; s2 += wp[kx * 4 + 2] * v; s3 += wp[kx * 4 + 3] * v;
                    }
                }
            }
            o0[x] = s0; o1[x] = s1; o2[x] = s2; o3[x] = s3;
        }
    }
}

/* pooled row p of a layer: conv rows 2p, 2p+1 -> max -> * s + b -> ReLU,
 * written into the bordered output planes (row p+1, columns 1..w/2) */
static void pool_row(const float *ra, const float *rb, int w, int cout, float s, const float *b,
                     float *out, size_t oplane, int p)
{
    int o, x, wo = w / 2, Wo = wo + 2;
    for (o = 0; o < cout; o++) {
        const float *r0 = ra + (size_t)o * w, *r1 = rb + (size_t)o * w;
        float *q = out + (size_t)o * oplane + (size_t)(p + 1) * Wo + 1;
        for (x = 0; x < wo; x++) {
            float m = r0[2 * x];
            if (r0[2 * x + 1] > m) { m = r0[2 * x + 1]; }
            if (r1[2 * x] > m) { m = r1[2 * x]; }
            if (r1[2 * x + 1] > m) { m = r1[2 * x + 1]; }
            m = m * s + b[o];                       /* (s > 0: max before scaling is the same) */
            q[x] = m > 0.0f ? m : 0.0f;             /* ReLU */
        }
    }
}

/* Which output rows of a padding-1 conv over n rows are the same as in the
 * last run, when the input rows moved by sh (old row = new row + sh) and
 * rin[j] says input row j is the same: all three input rows must be the same,
 * and a zero border row must be a border row in both runs. */
static void conv_reuse(const uint8_t *rin, int n, int sh, uint8_t *rout)
{
    int q, d;
    for (q = 0; q < n; q++) {
        uint8_t ok = 1;
        for (d = -1; d <= 1 && ok; d++) {
            int j = q + d, oj = j + sh;
            bool np = j < 0 || j >= n, op = oj < 0 || oj >= n;
            if (np != op || (!np && !rin[j])) { ok = 0; }
        }
        rout[q] = ok;
    }
}

/* moves the interior rows of c bordered planes up by sh rows */
static void shift_planes(float *a, int c, int h, int w, int sh)
{
    int i;
    size_t W = (size_t)w + 2, plane = (size_t)(h + 2) * W;
    if (sh <= 0 || sh >= h) { return; }
    for (i = 0; i < c; i++) {
        float *p = a + i * plane + W;               /* row 1 */
        memmove(p, p + (size_t)sh * W, sizeof(float) * (size_t)(h - sh) * W);
    }
}

bool kws_engine_run_shift(const kws_model_t *m, const float *feat, float *work, float *probs, float *emb, int shift)
{
    hdr_t *H = (hdr_t *)(void *)work;
    lay_t L;
    uint8_t rin[KWS_MAX_T], rc1[KWS_MAX_T], ra1[KWS_MAX_T / 2 + 1], rc2[KWS_MAX_T / 2 + 1],
            ra2[KWS_MAX_T / 4 + 1], rc3[KWS_MAX_T / 4 + 1];
    float g[KWS_MAX_C3], hd[KWS_MAX_HID], mx, sum;
    int T, B, i, j, o, p, s1, s2;
    size_t pl0, pl1, pl2;
    uint32_t macs = 0;
    bool reuse;

    if (!kws_model_ok(m) || !feat || !work || !probs) { return false; }
    layout(m, work, &L);
    T = L.T; B = L.B;
    if (H->magic != MAGIC || H->m != m || H->w1 != m->w1 || H->w2 != m->w2 || H->w3 != m->w3) {
        prep(m->w1, m->c1, 1, L.wf1);
        prep(m->w2, m->c2, m->c1, L.wf2);
        prep(m->w3, m->c3, m->c2, L.wf3);
        H->magic = MAGIC; H->m = m; H->w1 = m->w1; H->w2 = m->w2; H->w3 = m->w3;
        H->stamp = 0;
    }
    /* the rows can be reused when this work memory's last run was the
     * window shift frames ago (pooling twice: shift must be a multiple of 4) */
    reuse = H->stamp != 0 && shift >= 0 && shift < T && (shift & 3) == 0;
    if (!reuse) {
        memset(L.in, 0, sizeof(float) * (size_t)(T + 2) * (B + 2));
        memset(L.a1, 0, sizeof(float) * (size_t)m->c1 * (L.h1 + 2) * (L.w1 + 2));
        memset(L.a2, 0, sizeof(float) * (size_t)m->c2 * (L.h2 + 2) * (L.w2 + 2));
        shift = 0;
    }
    s1 = shift / 2; s2 = shift / 4;
    for (i = 0; i < T; i++) { rin[i] = reuse && i + shift < T; }
    conv_reuse(rin, T, shift, rc1);
    for (i = 0; i < L.h1; i++) { ra1[i] = rc1[2 * i] && rc1[2 * i + 1]; }
    conv_reuse(ra1, L.h1, s1, rc2);
    for (i = 0; i < L.h2; i++) { ra2[i] = rc2[2 * i] && rc2[2 * i + 1]; }
    conv_reuse(ra2, L.h2, s2, rc3);

    /* input (all of it: cheap) */
    for (i = 0; i < T; i++) {
        float *r = L.in + (size_t)(i + 1) * (B + 2) + 1;
        for (j = 0; j < B; j++) { r[j] = (feat[i * B + j] - m->in_mean[j]) / m->in_std[j]; }
    }
    pl0 = (size_t)(T + 2) * (B + 2);
    pl1 = (size_t)(L.h1 + 2) * (L.w1 + 2);
    pl2 = (size_t)(L.h2 + 2) * (L.w2 + 2);
    if (reuse) {
        shift_planes(L.a1, m->c1, L.h1, L.w1, s1);
        shift_planes(L.a2, m->c2, L.h2, L.w2, s2);
        if (s2 > 0 && s2 < L.h2) {
            for (o = 0; o < m->c3; o++) {
                float *r = L.rs + (size_t)o * L.h2;
                memmove(r, r + s2, sizeof(float) * (size_t)(L.h2 - s2));
            }
        }
    }
    /* conv1 + pool */
    for (p = 0; p < L.h1; p++) {
        if (ra1[p]) { continue; }
        conv_row(L.in, 1, pl0, B, 2 * p, L.wf1, L.c1p, L.ra);
        conv_row(L.in, 1, pl0, B, 2 * p + 1, L.wf1, L.c1p, L.rb);
        pool_row(L.ra, L.rb, B, m->c1, m->s1, m->b1, L.a1, pl1, p);
        macs += 2u * (uint32_t)(m->c1 * 9 * B);
    }
    /* conv2 + pool */
    for (p = 0; p < L.h2; p++) {
        if (ra2[p]) { continue; }
        conv_row(L.a1, m->c1, pl1, L.w1, 2 * p, L.wf2, L.c2p, L.ra);
        conv_row(L.a1, m->c1, pl1, L.w1, 2 * p + 1, L.wf2, L.c2p, L.rb);
        pool_row(L.ra, L.rb, L.w1, m->c2, m->s2, m->b2, L.a2, pl2, p);
        macs += 2u * (uint32_t)(m->c2 * m->c1 * 9 * L.w1);
    }
    /* conv3 + ReLU, summed per row */
    for (p = 0; p < L.h2; p++) {
        if (rc3[p]) { continue; }
        conv_row(L.a2, m->c2, pl2, L.w2, p, L.wf3, L.c3p, L.ra);
        for (o = 0; o < m->c3; o++) {
            const float *r = L.ra + (size_t)o * L.w2;
            float sacc = 0.0f;
            for (j = 0; j < L.w2; j++) {
                float v = r[j] * m->s3 + m->b3[o];
                if (v > 0.0f) { sacc += v; }
            }
            L.rs[(size_t)o * L.h2 + p] = sacc;
        }
        macs += (uint32_t)(m->c3 * m->c2 * 9 * L.w2);
    }
    for (o = 0; o < m->c3; o++) {
        const float *r = L.rs + (size_t)o * L.h2;
        float sacc = 0.0f;
        for (p = 0; p < L.h2; p++) { sacc += r[p]; }
        g[o] = sacc / (float)(L.h2 * L.w2);
    }
    H->stamp = H->stamp + 1u ? H->stamp + 1u : 1u;
    H->macs = macs;

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

bool kws_engine_run(const kws_model_t *m, const float *feat, float *work, float *probs, float *emb)
{
    return kws_engine_run_shift(m, feat, work, probs, emb, -1);
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
