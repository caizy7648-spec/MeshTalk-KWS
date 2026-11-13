/* kws_frontend.c - log-mel features for the voice SOS model. See kws_frontend.h. */
#include "kws_frontend.h"
#include <math.h>
#include <string.h>

#ifdef ESP_PLATFORM
#include "esp_attr.h"
#define KWS_EXT EXT_RAM_BSS_ATTR        /* v1.5: keep the internal RAM free (Wi-Fi, LoRa task) */
#else
#define KWS_EXT
#endif
#define NBINS (KWS_NFFT / 2 + 1)
#define PI_D  3.14159265358979323846

static bool  s_ready;
KWS_EXT static float s_win[KWS_FRAME];
KWS_EXT static float s_cos[KWS_NFFT / 2], s_sin[KWS_NFFT / 2];
KWS_EXT static uint16_t s_rev[KWS_NFFT];
/* mel triangles: band b covers bins s_lo[b] .. s_lo[b] + s_n[b] - 1, its
 * weights are s_wp[s_off[b] ...]. v1.5: packed (each bin is in at most two
 * triangles) - the full 40 x 257 table took 41 KB of internal RAM. */
static uint16_t s_lo[KWS_BANDS], s_n[KWS_BANDS], s_off[KWS_BANDS];
KWS_EXT static float s_wp[2 * NBINS];

static double hz_to_mel(double f) { return 2595.0 * log10(1.0 + f / 700.0); }
static double mel_to_hz(double m) { return 700.0 * (pow(10.0, m / 2595.0) - 1.0); }

void kws_fe_init(void)
{
    double edge[KWS_BANDS + 2], m0, m1;
    int i, b, k, bits = 0;

    if (s_ready) { return; }
    for (i = 0; i < KWS_FRAME; i++) {               /* periodic Hann */
        s_win[i] = (float)(0.5 - 0.5 * cos(2.0 * PI_D * i / KWS_FRAME));
    }
    for (i = 0; i < KWS_NFFT / 2; i++) {
        s_cos[i] = (float)cos(2.0 * PI_D * i / KWS_NFFT);
        s_sin[i] = (float)-sin(2.0 * PI_D * i / KWS_NFFT);
    }
    while ((1 << bits) < KWS_NFFT) { bits++; }
    for (i = 0; i < KWS_NFFT; i++) {
        int r = 0, x = i, j;
        for (j = 0; j < bits; j++) { r = (r << 1) | (x & 1); x >>= 1; }
        s_rev[i] = (uint16_t)r;
    }
    m0 = hz_to_mel(KWS_FMIN);
    m1 = hz_to_mel(KWS_FMAX);
    for (i = 0; i < KWS_BANDS + 2; i++) { edge[i] = mel_to_hz(m0 + (m1 - m0) * i / (KWS_BANDS + 1)); }
    {
        int used = 0;
        memset(s_wp, 0, sizeof(s_wp));
        for (b = 0; b < KWS_BANDS; b++) {
            double lo = edge[b], ce = edge[b + 1], hi = edge[b + 2];
            int first = -1, last = -1;
            for (k = 0; k < NBINS; k++) {               /* the bins of this triangle */
                double f = (double)k * KWS_RATE / KWS_NFFT, w = 0.0;
                if (f > lo && f < hi) { w = f <= ce ? (f - lo) / (ce - lo) : (hi - f) / (hi - ce); }
                if (w > 0.0) {
                    if (first < 0) { first = k; }
                    last = k;
                }
            }
            s_lo[b] = (uint16_t)(first < 0 ? 0 : first);
            s_n[b] = (uint16_t)(first < 0 ? 0 : last - first + 1);
            s_off[b] = (uint16_t)used;
            for (k = 0; k < s_n[b] && used < 2 * NBINS; k++, used++) {
                double f = (double)(s_lo[b] + k) * KWS_RATE / KWS_NFFT, w = 0.0;
                if (f > lo && f < hi) { w = f <= ce ? (f - lo) / (ce - lo) : (hi - f) / (hi - ce); }
                s_wp[used] = (float)w;
            }
        }
    }
    s_ready = true;
}

void kws_fe_reset(kws_fe_t *fe)
{
    memset(fe, 0, sizeof(*fe));
}

void kws_fe_frame(const int16_t *frame, float feat[KWS_BANDS])
{
    KWS_EXT static float re[KWS_NFFT], im[KWS_NFFT];   /* one caller at a time (the KWS task) */
    float x[KWS_FRAME], mean = 0.0f, p[NBINS];
    int i, b, len, half, step, k;

    if (!s_ready) { kws_fe_init(); }
    for (i = 0; i < KWS_FRAME; i++) { x[i] = (float)frame[i] * (1.0f / 32768.0f); mean += x[i]; }
    mean /= (float)KWS_FRAME;
    for (i = 0; i < KWS_NFFT; i++) {
        int s = s_rev[i];
        re[i] = s < KWS_FRAME ? (x[s] - mean) * s_win[s] : 0.0f;
        im[i] = 0.0f;
    }
    for (len = 2; len <= KWS_NFFT; len <<= 1) {     /* radix-2 butterflies */
        half = len >> 1;
        step = KWS_NFFT / len;
        for (i = 0; i < KWS_NFFT; i += len) {
            for (k = 0; k < half; k++) {
                float wr = s_cos[k * step], wi = s_sin[k * step];
                int a = i + k, c = a + half;
                float tr = re[c] * wr - im[c] * wi;
                float ti = re[c] * wi + im[c] * wr;
                re[c] = re[a] - tr; im[c] = im[a] - ti;
                re[a] += tr;        im[a] += ti;
            }
        }
    }
    for (k = 0; k < NBINS; k++) { p[k] = re[k] * re[k] + im[k] * im[k]; }
    for (b = 0; b < KWS_BANDS; b++) {
        float e = 0.0f;
        const float *w = &s_wp[s_off[b]];
        const float *q = &p[s_lo[b]];
        for (k = 0; k < s_n[b]; k++) { e += w[k] * q[k]; }
        feat[b] = logf(e + KWS_LOG_EPS);
    }
}

int kws_fe_push(kws_fe_t *fe, const int16_t *pcm, int n,
                void (*out)(void *ctx, const float *feat), void *ctx)
{
    float feat[KWS_BANDS];
    int made = 0;

    while (n > 0) {
        /* how many samples until the next frame is due */
        int need = fe->fill < KWS_FRAME ? KWS_FRAME - fe->fill : KWS_HOP - fe->since;
        int take = n < need ? n : need;
        if (fe->fill < KWS_FRAME) {
            memcpy(&fe->buf[fe->fill], pcm, (size_t)take * 2u);
            fe->fill += take;
        } else {
            memmove(fe->buf, fe->buf + take, (size_t)(KWS_FRAME - take) * 2u);
            memcpy(&fe->buf[KWS_FRAME - take], pcm, (size_t)take * 2u);
            fe->since += take;
        }
        pcm += take;
        n -= take;
        if (fe->fill == KWS_FRAME && (fe->since == KWS_HOP || (fe->since == 0 && take == need))) {
            fe->since = 0;
            kws_fe_frame(fe->buf, feat);
            if (out) { out(ctx, feat); }
            made++;
        }
    }
    return made;
}
