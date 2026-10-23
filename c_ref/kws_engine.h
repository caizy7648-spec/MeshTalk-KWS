/* kws_engine.h - v1.4 voice SOS: the keyword + speaker CNN, plain C (no library).
 *
 * The model file (kws_model.c) is written by export.py of the training package
 * MeshTalk-KWS; the network is the KwsNet class of its kws/model.py:
 *
 *   input   T x 40 log-mel frames, normalised per band: (x - in_mean) / in_std
 *   conv1   3x3, 1 -> c1, padding 1, ReLU, max-pool 2x2      (BatchNorm folded in)
 *   conv2   3x3, c1 -> c2, padding 1, ReLU, max-pool 2x2
 *   conv3   3x3, c2 -> c3, padding 1, ReLU, average over all positions -> g[c3]
 *   keyword head:  FC c3 -> hid, ReLU, FC hid -> ncls, softmax -> probs
 *   speaker head:  FC c3 -> spk_dim, divided by its length -> emb  (optional)
 *
 * Weights are int8 with one float scale per layer (w_real = w * scale), biases
 * float. Max-pool rounds down (an odd last row / column is dropped), like
 * torch.nn.MaxPool2d(2). Classes whose label starts with '_' are "not a
 * keyword" (e.g. "_other", "_noise"). */
#ifndef KWS_ENGINE_H
#define KWS_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define KWS_MAX_NCLS_ 8                /* (= KWS_MAX_NCLS, needed inside the struct) */

typedef struct {
    const char   *name;             /* "kws-2026-11-01" ...                     */
    uint32_t      id;               /* changes with every export (templates!)   */
    int           T, B;             /* frames per window, bands (40)            */
    int           c1, c2, c3, hid, ncls, spk_dim;
    const char   *labels;           /* "help|help me|sos|_other"                */
    const float  *in_mean, *in_std; /* [B]                                      */
    const int8_t *w1; float s1; const float *b1;    /* [c1][1][3][3]            */
    const int8_t *w2; float s2; const float *b2;    /* [c2][c1][3][3]           */
    const int8_t *w3; float s3; const float *b3;    /* [c3][c2][3][3]           */
    const int8_t *w4; float s4; const float *b4;    /* [hid][c3]                */
    const int8_t *w5; float s5; const float *b5;    /* [ncls][hid]              */
    const int8_t *w6; float s6; const float *b6;    /* [spk_dim][c3], or NULL   */
    /* detection settings chosen in training (see the report of the export) */
    float         thr[3];           /* keyword probability: sensitivity low / medium / high */
    float         cls_off[KWS_MAX_NCLS_]; /* v1.4.1: per class offset, threshold = thr[s] + cls_off[k] */
    float         spk_thr;          /* speaker: cosine similarity to the enrolled voice */
    int           hits;             /* keyword heard this often ...             */
    uint32_t      hits_ms;          /* ... within this time -> voice SOS        */
} kws_model_t;

#define KWS_MAX_T     101
#define KWS_MAX_C1    32
#define KWS_MAX_C2    64
#define KWS_MAX_C3    96
#define KWS_MAX_HID   128
#define KWS_MAX_NCLS  KWS_MAX_NCLS_
#define KWS_MAX_SPK   64

/* the model built into the firmware (Ai/kws_model.c). ncls == 0: none yet */
extern const kws_model_t kws_model;

bool   kws_model_ok(const kws_model_t *m);         /* present and within the limits */
/* floats of work memory kws_engine_run needs (give it from PSRAM if short) */
size_t kws_engine_work(const kws_model_t *m);
/* feat: T x B raw log-mel values (row = frame). probs: ncls (softmax).
 * emb: spk_dim values of length 1, or NULL. */
bool   kws_engine_run(const kws_model_t *m, const float *feat, float *work, float *probs, float *emb);
/* label of class k ("help me"), copied into buf */
const char *kws_label(const kws_model_t *m, int k, char *buf, size_t size);
bool   kws_is_keyword(const kws_model_t *m, int k);

#endif
