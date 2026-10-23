/* kws_detect.h - v1.4 voice SOS: from feature frames to "call for help".
 *
 * Plain C (runs on the ESP32, in Sim/ and in Tests/test_kws.c):
 *   - keeps the last T frames (about 1.2 s),
 *   - every KWS_STEP frames (240 ms) runs the CNN, as long as something louder
 *     than KWS_GATE_DBFS is inside the window (v1.4.1: before, only the last
 *     240 ms counted, so a short word got just 1-2 runs); complete silence in
 *     the whole window: no run (saves power),
 *   - averages the last KWS_SMOOTH results; a keyword class k above
 *     thr[sensitivity] + cls_off[k] is one "hit" (then KWS_REFRACT_MS pause,
 *     so one word is not counted twice),
 *   - with "Only My Voice" a hit also needs the voice to be close enough to
 *     the enrolled one: either the verify() hook (a separate speaker model,
 *     Ai/spk_engine.h) or the model's own speaker head (cosine >= spk_thr),
 *   - m->hits hits within m->hits_ms -> SOS. */
#ifndef KWS_DETECT_H
#define KWS_DETECT_H

#include "kws_engine.h"
#include "kws_frontend.h"

#define KWS_STEP        12          /* frames between two runs (12 x 20 ms)      */
#define KWS_SMOOTH      3           /* runs averaged                              */
#define KWS_REFRACT_MS  1200u       /* after a hit                                */
#define KWS_GATE_DBFS   (-55.0f)    /* quieter than this: no run                  */
#define KWS_MAX_HITS    8

enum { KWS_SENS_LOW = 0, KWS_SENS_MED, KWS_SENS_HIGH };

/* events of kws_det_frame (bit mask) */
#define KWS_EV_RAN      1           /* the CNN ran: probs / emb are new           */
#define KWS_EV_HIT      2           /* a keyword was heard (and accepted)         */
#define KWS_EV_REJECT   4           /* a keyword, but not the enrolled voice      */
#define KWS_EV_SOS      8           /* enough hits: call for help                 */

#define KWS_VOICE_DIM   256         /* longest voice vector a template can hold   */
#define KWS_ENROL_MAX   10          /* most enrolment takes                       */

typedef struct {
    float    tmpl[KWS_MAX_NCLS][KWS_VOICE_DIM]; /* enrolled voice per class (sum of vectors) */
    float    all[KWS_VOICE_DIM];                /* all classes                          */
    uint8_t  have[KWS_MAX_NCLS];                /* samples behind tmpl[k]              */
    uint8_t  n_all;
} kws_voice_t;

typedef struct {
    const kws_model_t *m;
    float   *work;                  /* kws_engine_work(m) floats                  */
    float   *ring;                  /* T x 40 frames, a ring                      */
    float   *win;                   /* T x 40, the window handed to the CNN       */
    int      head, filled, step;
    int      quiet;                 /* frames since the last one louder than the gate */
    float    hist[KWS_SMOOTH][KWS_MAX_NCLS];
    float    emb_hist[KWS_SMOOTH][KWS_MAX_SPK];
    int      n_hist, hist_pos;
    uint32_t last_hit, hits[KWS_MAX_HITS];
    int      n_hits;
    /* settings */
    uint8_t  sens;
    bool     only_me;
    const kws_voice_t *voice;       /* NULL = not enrolled                        */
    /* v1.4.1: a separate speaker model. Called on a hit with the window just
     * run (T x 40) and the class; returns the similarity to the enrolled voice
     * (-2 = cannot tell). NULL = use the model's speaker head, if any. */
    float  (*verify)(void *ctx, const float *win, int T, int cls);
    void    *verify_ctx;
    float    verify_thr;            /* accept when similarity >= this             */
    /* results (valid after the event that names them) */
    float    probs[KWS_MAX_NCLS];   /* last run                                   */
    float    smooth[KWS_MAX_NCLS];  /* average of the last runs                   */
    float    emb[KWS_MAX_SPK];
    int      hit_cls;               /* class of the last hit / reject             */
    float    hit_score, hit_spk;    /* its probability, its speaker similarity (-2 = not checked) */
    int      hit_count;             /* hits in the current time window            */
    uint32_t run_us;                /* not set here: the caller may time a run    */
} kws_det_t;

/* work: kws_engine_work(m) floats; ring, win: KWS_MAX_T x 40 floats each
 * (all three may be in PSRAM) */
void kws_det_init(kws_det_t *d, const kws_model_t *m, float *work, float *ring, float *win);
void kws_det_reset(kws_det_t *d);   /* forget the sound so far (after a pause)   */
/* one feature frame; dbfs = loudness of its 320 new samples */
int  kws_det_frame(kws_det_t *d, const float *feat, float dbfs, uint32_t now_ms);
/* copies the window (T x 40, oldest first) into out */
void kws_det_window(const kws_det_t *d, float *out);

/* speaker similarity of a dim-long vector to the enrolled voice for class k
 * (cosine; the class template when there is one, else all; -2: nothing enrolled) */
float kws_voice_similarity(const kws_voice_t *v, int dim, int k, const float *emb);
/* adds one enrolment sample */
void  kws_voice_add(kws_voice_t *v, int dim, int k, const float *emb);
/* enough samples: takes in all and every keyword of m at least once */
bool  kws_voice_ready(const kws_model_t *m, const kws_voice_t *v, int takes);
#define KWS_ENROL_TAKES     5       /* default: "Help", "Help me", "SOS", "Help", "Help me" */
#define KWS_ENROL_PER_WORD  KWS_ENROL_TAKES     /* (tests) */

/* Whole-clip search (enrolment, tests): runs the CNN over every window of the
 * features (n frames) and returns the window where class k is most likely.
 * best_emb may be NULL. Returns the probability, -1 when the clip is too short. */
float kws_best_window(const kws_model_t *m, float *work, const float *feats, int n, int k,
                      float *best_emb, int *best_at);

/* class the enrolment take i asks for: the model's keywords in turn (-1: none) */
int   kws_enrol_class(const kws_model_t *m, int take);

/* dBFS of n samples */
float kws_dbfs(const int16_t *pcm, int n);

#endif
