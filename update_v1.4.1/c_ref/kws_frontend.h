/* kws_frontend.h - v1.4 voice SOS: sound -> log-mel features (plain C).
 *
 * The SAME computation is in the training package (MeshTalk-KWS/kws/features.py);
 * a model only works when both agree, Tests/test_kws.c checks it.
 *
 *   16 kHz mono int16, scaled to -1..1
 *   frame 480 samples (30 ms), hop 320 (20 ms), frame mean removed,
 *   periodic Hann window, 512 point FFT, power |X|^2,
 *   40 mel bands (HTK mel scale, triangles 60 .. 7600 Hz, peak 1),
 *   feature = ln(band energy + 1e-6)
 *
 * Streaming: feed any number of samples; every 320 new samples give one frame. */
#ifndef KWS_FRONTEND_H
#define KWS_FRONTEND_H

#include <stdint.h>
#include <stdbool.h>

#define KWS_RATE        16000
#define KWS_FRAME       480
#define KWS_HOP         320
#define KWS_NFFT        512
#define KWS_BANDS       40
#define KWS_FMIN        60.0f
#define KWS_FMAX        7600.0f
#define KWS_LOG_EPS     1e-6f

typedef struct {
    int16_t buf[KWS_FRAME];         /* the last KWS_FRAME samples           */
    int     fill;                   /* samples in buf (until the first frame) */
    int     since;                  /* new samples since the last frame     */
} kws_fe_t;

void kws_fe_init(void);             /* tables (window, mel); safe to call again */
void kws_fe_reset(kws_fe_t *fe);
/* Feeds n samples. For every finished frame calls out(ctx, feat[KWS_BANDS]).
 * Returns the number of frames made. */
int  kws_fe_push(kws_fe_t *fe, const int16_t *pcm, int n,
                 void (*out)(void *ctx, const float *feat), void *ctx);
/* one frame of exactly KWS_FRAME samples (used by kws_fe_push and the tests) */
void kws_fe_frame(const int16_t *frame, float feat[KWS_BANDS]);

#endif
