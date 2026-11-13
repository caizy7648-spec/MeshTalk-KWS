/* harness.c - the firmware v1.6.7 keyword detector as a shared library for
 * Python (kws/fwdet.py). Audio goes in exactly as Svc/kws.c feeds it on the
 * device: 20 ms blocks of int16 -> kws_dbfs() of the block -> kws_fe_push()
 * (log-mel) -> kws_det_frame() for every frame, with the settings
 * apply_opts() makes (sensitivity, 2-in-10 s / 3-in-20 s, words mode,
 * cls_extra / no_relax).
 *
 * Two ways to get the CNN probabilities:
 *   native = 1: the int8 C engine with the model compiled in (kws_model.c of
 *               the build, e.g. out/kws_model.c) - exactly the device;
 *   native = 0: a callback (Python, a PyTorch model) gets every window the
 *               detector would give the CNN (normal runs and word checks).
 * Build: python -m kws.fwdet --build [--model out/kws_model.c] */
#include "kws_frontend.h"
#include "kws_engine.h"
#include "kws_detect.h"
#include <stdlib.h>
#include <string.h>

typedef void (*probs_cb)(uint32_t now, const float *win, int T, int word, float *probs);

typedef struct {
    kws_model_t m;
    kws_det_t   d;
    kws_fe_t    fe;
    float      *work, *ring, *win, *work2, *win2;
    probs_cb    cb;
    uint32_t    now;
    float       db;
    int32_t    *ev;                 /* events: t_ms, flags, class, score*1000 */
    char       *own_labels;
    int         n_ev, max_ev;
} h_t;

static void fake(void *ctx, uint32_t now, const float *win, bool word, float *probs)
{
    h_t *h = (h_t *)ctx;
    h->cb(now, win, h->m.T, word ? 1 : 0, probs);
}

static void on_frame(void *ctx, const float *feat)
{
    h_t *h = (h_t *)ctx;
    int ev = kws_det_frame(&h->d, feat, h->db, h->now);
    if ((ev & (KWS_EV_HIT | KWS_EV_SOS | KWS_EV_REJECT)) && h->n_ev < h->max_ev) {
        int32_t *e = h->ev + 4 * h->n_ev++;
        e[0] = (int32_t)h->now; e[1] = ev; e[2] = h->d.hit_cls; e[3] = (int32_t)(h->d.hit_score * 1000.0f);
    }
}

/* labels "help|help me|sos|_other"; thr3 / cls_off may be NULL (native: the model's) */
void *h_new(int native, int T, int ncls, const char *labels, const float *thr3, const float *cls_off,
            const float *cls_extra, const unsigned char *no_relax, int sens, int need_hits, unsigned need_ms,
            int words_on, probs_cb cb)
{
    h_t *h = (h_t *)calloc(1, sizeof(h_t));
    size_t wk;
    int k;
    if (!h) { return NULL; }
    kws_fe_init();
    /* the detector checks kws_model_ok(): start from the compiled-in model;
     * with a callback its weights are never used, only T / ncls / labels / thresholds */
    h->m = kws_model;
    if (!native) {
        h->m.name = "python";
        if (T > 0) { h->m.T = T; }
        if (ncls > 0) { h->m.ncls = ncls; }
    }
    if (labels) { h->own_labels = strdup(labels); h->m.labels = h->own_labels; }
    if (thr3) { memcpy(h->m.thr, thr3, sizeof(h->m.thr)); }
    if (cls_off) { for (k = 0; k < h->m.ncls && k < KWS_MAX_NCLS; k++) { h->m.cls_off[k] = cls_off[k]; } }
    wk = native ? kws_engine_work(&h->m) : 1;
    h->work = (float *)calloc(wk, sizeof(float));
    h->work2 = (float *)calloc(wk, sizeof(float));
    h->ring = (float *)calloc((size_t)KWS_MAX_T * KWS_BANDS, sizeof(float));
    h->win = (float *)calloc((size_t)KWS_MAX_T * KWS_BANDS, sizeof(float));
    h->win2 = (float *)calloc((size_t)KWS_MAX_T * KWS_BANDS, sizeof(float));
    kws_det_init(&h->d, &h->m, h->work, h->ring, h->win);
    h->d.work2 = h->work2;
    h->d.win2 = h->win2;
    h->d.sens = (uint8_t)sens;
    h->d.need_hits = need_hits;
    h->d.need_ms = need_ms;
    h->d.words_on = words_on != 0;
    for (k = 0; k < h->m.ncls && k < KWS_MAX_NCLS; k++) {
        h->d.cls_extra[k] = cls_extra ? cls_extra[k] : 0.0f;
        h->d.no_relax[k] = no_relax ? no_relax[k] : 0;
    }
    if (!native) { h->cb = cb; h->d.fake_run = fake; h->d.fake_ctx = h; }
    kws_fe_reset(&h->fe);
    h->now = 1000;                  /* 0 means "never" inside the detector */
    return h;
}

/* n samples (a multiple of 320 is used; the rest is dropped). Events into
 * ev[4 * max_ev]; returns how many. The clock goes on from the last call. */
int h_feed(void *hp, const int16_t *pcm, int n, int32_t *ev, int max_ev)
{
    h_t *h = (h_t *)hp;
    int i;
    h->ev = ev; h->n_ev = 0; h->max_ev = max_ev;
    for (i = 0; i + KWS_HOP <= n; i += KWS_HOP) {
        h->db = kws_dbfs(pcm + i, KWS_HOP);
        kws_fe_push(&h->fe, pcm + i, KWS_HOP, on_frame, h);
        h->now += 20u;
    }
    return h->n_ev;
}

/* forget everything (a new, independent stream); the clock goes on */
void h_reset(void *hp)
{
    h_t *h = (h_t *)hp;
    uint8_t sens = h->d.sens, nr[KWS_MAX_NCLS];
    int need = h->d.need_hits, k;
    uint32_t need_ms = h->d.need_ms;
    bool words = h->d.words_on;
    float ex[KWS_MAX_NCLS];
    memcpy(ex, h->d.cls_extra, sizeof(ex));
    memcpy(nr, h->d.no_relax, sizeof(nr));
    kws_det_init(&h->d, &h->m, h->work, h->ring, h->win);
    h->d.work2 = h->work2; h->d.win2 = h->win2;
    h->d.sens = sens; h->d.need_hits = need; h->d.need_ms = need_ms; h->d.words_on = words;
    for (k = 0; k < KWS_MAX_NCLS; k++) { h->d.cls_extra[k] = ex[k]; h->d.no_relax[k] = nr[k]; }
    if (h->cb) { h->d.fake_run = fake; h->d.fake_ctx = h; }
    kws_fe_reset(&h->fe);
    h->now += 60000u;
}

void h_free(void *hp)
{
    h_t *h = (h_t *)hp;
    if (!h) { return; }
    free(h->own_labels);
    free(h->work); free(h->work2); free(h->ring); free(h->win); free(h->win2);
    free(h);
}

/* the C front end alone (to check it against kws.features.logmel) */
typedef struct { float *o; int n, max; } fe_out_t;

static void fe_out(void *ctx, const float *f)
{
    fe_out_t *c = (fe_out_t *)ctx;
    if (c->n < c->max) { memcpy(c->o + (size_t)c->n * KWS_BANDS, f, sizeof(float) * KWS_BANDS); c->n++; }
}

int h_logmel(const int16_t *pcm, int n, float *out, int max_frames)
{
    kws_fe_t fe;
    fe_out_t c = { out, 0, max_frames };
    int i;
    kws_fe_init();
    kws_fe_reset(&fe);
    for (i = 0; i + KWS_HOP <= n; i += KWS_HOP) { kws_fe_push(&fe, pcm + i, KWS_HOP, fe_out, &c); }
    return c.n;
}

uint32_t h_now(void *hp) { return ((h_t *)hp)->now; }
