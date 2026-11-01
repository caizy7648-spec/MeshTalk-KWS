/* kws_detect.h - v1.4 voice SOS: from feature frames to "call for help".
 *
 * Plain C (runs on the ESP32, in Sim/ and in Tests/test_kws.c):
 *   - keeps the last T frames (about 1.2 s),
 *   - every KWS_STEP frames (240 ms) runs the CNN, as long as something louder
 *     than KWS_GATE_DBFS is inside the window (v1.4.1: before, only the last
 *     240 ms counted, so a short word got just 1-2 runs); complete silence in
 *     the whole window: no run (saves power),
 *   - averages the last KWS_SMOOTH results; a keyword class k above
 *     thr[sensitivity] + cls_off[k] is one "hit" (v1.6: each call counted once,
 *     see KWS_REARM_DROP; v1.5 used a KWS_REFRACT_MS pause,
 *     so one word is not counted twice),
 *   - with "Only My Voice" a hit also needs the voice to be close enough to
 *     the enrolled one: either the verify() hook (a separate speaker model,
 *     Ai/spk_engine.h) or the model's own speaker head (cosine >= spk_thr),
 *   - m->hits hits within m->hits_ms -> SOS (v1.5.1: need_hits / need_ms
 *     override it, e.g. 2 within 10 s).
 * v1.5.1: the CNN reuses the rows of the last run (kws_engine_run_shift, ~3x
 * faster); sensitivity "high" averages the best 2 of the last 3 runs (a fast,
 * short call is caught); a voice that does not match is checked again on the
 * next 2 runs (the word is then better inside the window) before it is
 * rejected. */
#ifndef KWS_DETECT_H
#define KWS_DETECT_H

#include "kws_engine.h"
#include "kws_frontend.h"

#define KWS_STEP        12          /* frames between two runs (12 x 20 ms); a multiple of 4
                                       lets the engine reuse the last run (v1.5.1) */
#define KWS_SMOOTH      3           /* runs averaged                              */
#define KWS_REFRACT_MS  1200u       /* after a hit                                */
#define KWS_GATE_DBFS   (-55.0f)    /* quieter than this: no run                  */
#define KWS_MAX_HITS    8
#define KWS_VERIFY_TRIES 3          /* v1.5.1: a rejected voice is checked again on the next runs */
#define KWS_RETRY_DROP  0.15f       /* (while the keyword is still about that likely)         */
/* v1.6 counting: a call is counted when the keyword score rises above the
 * threshold after it had dropped (KWS_REARM_DROP below it). While the score
 * stays high (a call stays ~1.6 s in the window) it only counts again for a
 * NEW sound that started after the last count, after a pause of >= 140 ms,
 * >= 400 ms after the counted one began and >= 160 ms long (a quick "Help!
 * Help!"), or when the calling goes on without a pause for KWS_SUSTAIN_MS.
 * After an SOS only a drop of the score re-arms (the end of that call or the
 * device's own alarm sound never count). */
#define KWS_MAX_SEG     12          /* sound segments remembered                            */
#define KWS_SEG_MIN     5           /* a segment shorter than 100 ms is a click             */
#define KWS_SEG_END     3           /* 60 ms below the level ends a segment                 */
#define KWS_GAP         7           /* a pause of 140 ms separates two sounds (v1.6b, was 200) */
#define KWS_NEW_MIN     8           /* a new sound must be 160 ms long ...                  */
#define KWS_NEW_AFTER   20          /* ... and begin 400 ms after the counted one           */
#define KWS_VOICE_CONFIRM 0.03f     /* a confirming call needs this much less voice match too (v1.6d, was 0.05) */
#define KWS_REJ_PENALTY 0.02f       /* v1.6d: each voice CLEARLY rejected within KWS_REJ_MS for the same
                                       keyword: +0.02 for that keyword (v1.6f: was 0.03 for all) ... */
#define KWS_REJ_MS      10000u      /* (v1.6f: 10 s, was 15 / 20) */
#define KWS_REJ_MAX     2           /* ... at most +0.04 (v1.6f, was 0.06 / 0.09); an accepted call ends it */
#define KWS_REJ_CLEAR   0.06f       /* v1.6e: a miss by less than this (the owner's near misses) adds nothing */
#define KWS_REARM_DROP  0.15f
#define KWS_SUSTAIN_MS  2500u
#define KWS_EV_SAME     16          /* (event) keyword still heard, same call: not counted   */
#define KWS_EV_SOUND    32          /* (event) a sound has ended: sound_best[] = best scores during it */
#define KWS_EV_WORDS    64          /* (event) v1.6.1: words were checked one by one: wc_*[] */

/* v1.6.1 "2 calls said straight after each other" (only with need_hits == 2
 * and words_on): the loud sound is also cut into words at the dips of the
 * loudness (KWS_DIP_DB down, KWS_DIP_RISE up again); when a keyword is being
 * heard, every finished word is checked ALONE by the keyword model (the
 * word with quiet before and after, work2 / win2). The calls of one sound
 * are then the words that are a keyword on their own: "Help! Help!" said
 * without a pause = 2, "Help me" = 1 ("me" alone is no keyword), an echo or
 * the "p" of help = none. */
#define KWS_MAX_WORD    12
#define KWS_WORD_MIN    5           /* a word is at least 100 ms                            */
#define KWS_DIP_DB      8.0f        /* the loudness drops this much inside a sound ...      */
#define KWS_DIP_RISE    6.0f        /* ... and rises this much again: a new word            */
#define KWS_WORD_PAD    6           /* quiet frames after the word in its check window      */
#define KWS_WORD_SLACK  0.10f       /* a word alone may score this much below the threshold */
#define KWS_WORD_AFTER  15          /* the 2nd word starts >= 300 ms after the 1st          */
#define KWS_CONFIRM_DROP 0.08f      /* v1.6: after a first hit, the next call within the time
                                       window needs this much less (confirming is easier)    */

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
    int      step_frames;           /* v1.5: frames between runs (KWS_STEP; more on a slow chip) */
    int      quiet;                 /* frames since the last one louder than the gate */
    int      since;                 /* v1.5.1: frames since the last run, -1 = unknown */
    uint32_t eng_stamp;             /* kws_engine_stamp(work) after our last run  */
    int      need_hits;             /* v1.5.1: hits for an SOS / within need_ms; 0 = the model's */
    uint32_t need_ms;
    int      retry_cls, retry_left; /* v1.5.1: voice rejected: tries left on newer windows */
    int      retry_n;               /* v1.6: calls the pending hit counts      */
    /* v1.6: counting each call once (see KWS_REARM_DROP) */
    uint32_t fno;                   /* frames so far                               */
    float    floor_db, peak_db;     /* background level, recent loudest            */
    struct { uint32_t s, e; } seg[KWS_MAX_SEG];     /* loud segments (frames)      */
    int      nseg, seg_quiet;
    bool     seg_open;
    bool     armed;                 /* the score dropped since the last count      */
    bool     after_sos;             /* the last count made an SOS                  */
    bool     same_told;             /* KWS_EV_SAME given for this call             */
    uint32_t cnt_fno, cnt_start;    /* frame of the last count, start of its sound */
    uint8_t  hit_how;               /* last count: 0 fresh, 1 quick next call, 2 calling on */
    float    sound_best[KWS_MAX_NCLS];      /* diagnostics: best smoothed scores of the sound */
    int      sound_counted;         /* calls counted during it                     */
    uint32_t sound_from;            /* its first frame (0 = no sound being followed) */
    float    hist[KWS_SMOOTH][KWS_MAX_NCLS];
    float    emb_hist[KWS_SMOOTH][KWS_MAX_SPK];
    int      n_hist, hist_pos;
    uint32_t last_hit, hits[KWS_MAX_HITS];
    uint32_t rej[KWS_MAX_HITS];     /* v1.6d: times of rejected voices */
    int8_t   rej_cls[KWS_MAX_HITS]; /* v1.6f: and their keywords */
    int      n_rej;
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
    float    verify_thr_k[KWS_MAX_NCLS]; /* v1.6c: per keyword (> 0), else verify_thr */
    float    rej_step[KWS_MAX_NCLS];     /* v1.6g: penalty per clear miss for this keyword (> 0), else KWS_REJ_PENALTY */
    /* tests only: probabilities from a script instead of the CNN (NULL on the device) */
    void   (*fake_run)(void *ctx, uint32_t now_ms, const float *win, bool word, float *probs);
    /* v1.6.1: words one by one (NULL work2 / win2: off) */
    bool     words_on;
    float   *work2, *win2;          /* kws_engine_work(m) / T x 40 floats         */
    float    qfeat[KWS_BANDS];      /* a recent quiet frame (fills a word's window) */
    struct { uint32_t s, e; int8_t cls; uint8_t checked; float p; } word[KWS_MAX_WORD];
    int      nword;
    bool     word_open;
    uint32_t w_s;
    float    w_max, w_min;
    uint32_t w_minf;
    uint32_t ep_from;               /* the sound the last fresh count stood for (frame) */
    int      ep_calls;              /* calls counted since that fresh count       */
    int      wc_n;                  /* diagnostics: the last words checked        */
    struct { uint32_t len; int8_t cls; float p; } wc[4];
    void    *fake_ctx;
    /* results (valid after the event that names them) */
    float    probs[KWS_MAX_NCLS];   /* last run                                   */
    float    smooth[KWS_MAX_NCLS];  /* average of the last runs                   */
    float    emb[KWS_MAX_SPK];
    int      hit_cls;               /* class of the last hit / reject             */
    float    hit_need;              /* v1.6e: voice similarity the last check needed (penalty included) */
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

/* v1.5.1: the same for all classes in one pass: best[k] / at[k] for every
 * class (windows step frames apart; a multiple of 4 reuses the last run) */
void  kws_best_windows(const kws_model_t *m, float *work, const float *feats, int n, int step,
                       float *best, int *at, float *any, int *any_at);
/* (v1.6: any / any_at, may be NULL: the best sum of all keyword classes = "some
 * call for help was said", and where) */

/* class the enrolment take i asks for: the model's keywords in turn (-1: none) */
int   kws_enrol_class(const kws_model_t *m, int take);

/* v1.6c: voice similarity keyword k needs (verify_thr_k[k], else verify_thr) */
float kws_voice_need(const kws_det_t *d, int k);

/* dBFS of n samples */
float kws_dbfs(const int16_t *pcm, int n);

#endif
