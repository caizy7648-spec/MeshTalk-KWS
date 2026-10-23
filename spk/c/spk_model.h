/* spk_model.h - the trained speaker model (written into spk_model.c by
 * MeshTalk-KWS spk/export.py) as used by spk_engine.c.
 *
 * Network (spk/model.py SpkNet): log-mel [frames x 40] -> per-band
 * normalisation -> nlayers x [Conv1d over time (kernel k, dilation d, zero
 * padded to the same length) + ReLU] (BatchNorm folded in) -> mean and
 * standard deviation over time -> Linear (BatchNorm folded in) -> length 1.
 * Weights: int8, one float scale per output channel. */
#ifndef SPK_MODEL_H
#define SPK_MODEL_H

#include <stdint.h>

#define SPK_MAX_LAYERS  6
#define SPK_BANDS       40

typedef struct {
    const char   *name;
    uint32_t      id;
    int           frames, nlayers, dim;
    int           ch[SPK_MAX_LAYERS], k[SPK_MAX_LAYERS], d[SPK_MAX_LAYERS];
    const float  *in_mean, *in_std;                 /* SPK_BANDS each                  */
    const int8_t *w[SPK_MAX_LAYERS];                /* [ch][cin][k]                     */
    const float  *s[SPK_MAX_LAYERS], *b[SPK_MAX_LAYERS];   /* scale, bias per channel   */
    const int8_t *fw;                               /* [dim][2 * ch[nlayers-1]]         */
    const float  *fs, *fb;                          /* [dim]                            */
    float         threshold;                        /* default cosine threshold         */
    int           enrol_takes;
} spk_model_t;

extern const spk_model_t spk_model;

#endif
