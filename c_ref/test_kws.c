/* test_kws.c - v1.4 voice SOS: the C code against the Python training package.
 *
 *   cd Tests && make test_kws && ./test_kws
 *
 * 1. front-end: Ai/kws_frontend.c gives the same log-mel features as
 *    MeshTalk-KWS/kws/features.py (kws_fe_golden.c, made by that script)
 * 2. engine: Ai/kws_engine.c gives the same probabilities / speaker vector as
 *    export.py's reference for a model (kws_test_model*.c, random weights)
 * 3. decision logic (kws_detect.c): gate, hits, refractory time, 3 hits -> SOS,
 *    "Only My Voice" rejects another voice
 * 4. the firmware's own Ai/kws_model.c: either "no model" or a valid model
 *    whose kws_model_test.c (Tests/kws_model_test.c, if present) matches. */
#include "kws_frontend.h"
#include "kws_engine.h"
#include "kws_detect.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const int16_t kws_fe_golden_pcm[16000];
extern const int kws_fe_golden_frames;
extern const float kws_fe_golden_feat[];
extern const kws_model_t kws_test_model;
extern const float kws_test_model_test_in[], kws_test_model_test_probs[], kws_test_model_test_emb[];
#ifdef HAVE_MODEL_TEST
extern const float kws_model_test_in[], kws_model_test_probs[], kws_model_test_emb[];
#endif

static int checks, failed;
#define CHECK(c, ...) do { checks++; if (!(c)) { failed++; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static float s_feats[100][KWS_BANDS];
static int   s_nf;
static void collect(void *ctx, const float *f)
{
    (void)ctx;
    if (s_nf < 100) { memcpy(s_feats[s_nf], f, sizeof(s_feats[0])); }
    s_nf++;
}

static float verify_fixed(void *ctx, const float *win, int T, int cls)
{
    (void)win; (void)T; (void)cls;
    return *(const float *)ctx;
}

/* v1.5.1: the plain engine of v1.5, to check the fast one (included here so
 * the Makefile stays the same) */
#include "kws_engine_ref.c"

/* v1.5.1: the fast engine (full and moved-window runs) against the plain one
 * over a long random stream; returns the largest difference, *inc_diff the
 * largest difference of the moved-window runs to full runs (should be 0) */
static float compare_fast(const kws_model_t *m, int step, float *inc_diff, double *mac_share)
{
    int n = 400, i, j, k, T = m->T;
    float *feats = malloc(sizeof(float) * (size_t)n * KWS_BANDS);
    float *w_ref = malloc(sizeof(float) * ref_engine_work(m)), *w_full = malloc(sizeof(float) * kws_engine_work(m));
    float *w_inc = malloc(sizeof(float) * kws_engine_work(m));
    float pr[KWS_MAX_NCLS], pf[KWS_MAX_NCLS], pi[KWS_MAX_NCLS], er[KWS_MAX_SPK], ef[KWS_MAX_SPK], ei[KWS_MAX_SPK], d = 0.0f;
    double macs_full = 0.0, macs_inc = 0.0;
    uint32_t seed = 12345u;
    *inc_diff = 0.0f;
    for (i = 0; i < n * KWS_BANDS; i++) {
        seed = seed * 1664525u + 1013904223u;
        feats[i] = -14.0f + 13.0f * (float)(seed >> 8) / 16777216.0f + (float)((i / KWS_BANDS) % 37) * 0.05f;
    }
    for (i = 0, k = 0; i + T <= n; i += step, k++) {
        const float *f = feats + (size_t)i * KWS_BANDS;
        ref_engine_run(m, f, w_ref, pr, er);
        kws_engine_run(m, f, w_full, pf, ef);
        macs_full += kws_engine_macs(w_full);
        kws_engine_run_shift(m, f, w_inc, pi, ei, k ? step : -1);
        if (k) { macs_inc += kws_engine_macs(w_inc); }
        for (j = 0; j < m->ncls; j++) { d = fmaxf(d, fabsf(pf[j] - pr[j])); *inc_diff = fmaxf(*inc_diff, fabsf(pi[j] - pf[j])); }
        for (j = 0; j < m->spk_dim; j++) { d = fmaxf(d, fabsf(ef[j] - er[j])); *inc_diff = fmaxf(*inc_diff, fabsf(ei[j] - ef[j])); }
    }
    *mac_share = k > 1 ? macs_inc / (k - 1) / (macs_full / k) : 1.0;
    free(feats); free(w_ref); free(w_full); free(w_inc);
    return d;
}

/* v1.6 tests: scripted keyword scores. A call {start, length, keyword} makes
 * class 0 likely from 300 ms after its start until 1.3 s after its end (the
 * time it stays in the model's window) */
static const uint32_t *s_script;
static int s_script_n;
static void fake_run(void *ctx, uint32_t now, const float *win, bool word, float *probs)
{
    int c;
    (void)ctx; (void)win; (void)word;
    probs[0] = 0.05f; probs[1] = 0.02f; probs[2] = 0.02f; probs[3] = 0.91f;
    for (c = 0; c < s_script_n; c++) {
        const uint32_t *k = s_script + c * 3;
        if (k[2] && now >= k[0] + 300u && now <= k[0] + k[1] + 1300u) {      /* k[2]: class + 1 */
            probs[0] = 0.05f; probs[k[2] - 1] = 0.9f; probs[3] = 0.06f;
        }
    }
}

static int s_vcalls;
static const float *s_vseq;
static float verify_seq(void *ctx, const float *win, int T, int cls)
{
    int n = s_vcalls++;
    (void)ctx; (void)win; (void)T; (void)cls;
    return s_vseq[n < 3 ? n : 3];
}

static int s_vmax = 11;
static float verify_seq12(void *ctx, const float *win, int T, int cls)
{
    int n = s_vcalls++;
    (void)ctx; (void)win; (void)T; (void)cls;
    return s_vseq[n < s_vmax ? n : s_vmax];
}

/* v1.6.1 tests: script {start, length, marker, dB}; the whole-window model
 * hears keyword marker-1 from 300 ms after its start until 1.3 s after its
 * end; a word checked alone is a keyword when its frames carry a keyword marker */
static const int *s_sc4;
static int s_sc4_n;
static void fake_run4(void *ctx, uint32_t now, const float *win, bool word, float *probs)
{
    int c, r, mk = 0;
    (void)ctx;
    probs[0] = 0.05f; probs[1] = 0.02f; probs[2] = 0.02f; probs[3] = 0.91f;
    if (word) {
        for (r = 0; r < KWS_MAX_T; r++) { if (win[r * KWS_BANDS] > 0.5f && win[r * KWS_BANDS] < 4.5f) { mk = (int)(win[r * KWS_BANDS] + 0.5f); } }
        if (mk >= 1 && mk <= 3) { probs[mk - 1] = 0.9f; probs[3] = 0.06f; }
        return;
    }
    for (c = 0; c < s_sc4_n; c++) {
        const int *k = s_sc4 + c * 4;
        if (k[2] >= 1 && k[2] <= 3 && (int)now >= k[0] + 300 && (int)now <= k[0] + k[1] + 1300) { probs[0] = 0.05f; probs[k[2] - 1] = 0.9f; probs[3] = 0.06f; }
    }
}

/* rejects the first call, accepts after that */
static float verify_second(void *ctx, const float *win, int T, int cls)
{
    (void)ctx; (void)win; (void)T; (void)cls;
    return s_vcalls++ == 0 ? 0.1f : 0.9f;
}

static float compare_model(const kws_model_t *m, const float *in, const float *probs, const float *emb)
{
    float *work = malloc(sizeof(float) * kws_engine_work(m));
    float p[KWS_MAX_NCLS], e[KWS_MAX_SPK], d = 0.0f;
    int i;
    if (!kws_engine_run(m, in, work, p, e)) { free(work); return 1e9f; }
    for (i = 0; i < m->ncls; i++) { d = fmaxf(d, fabsf(p[i] - probs[i])); }
    for (i = 0; i < m->spk_dim; i++) { d = fmaxf(d, fabsf(e[i] - emb[i])); }
    free(work);
    return d;
}

int main(void)
{
    kws_fe_t fe;
    int i, j, pos;
    float d = 0.0f;
    char b[24];

    printf("1. front-end against features.py\n");
    kws_fe_init();
    kws_fe_reset(&fe);
    s_nf = 0;
    for (pos = 0; pos < 16000; pos += 77) {          /* odd pieces, like the I2S reads */
        int n = 16000 - pos < 77 ? 16000 - pos : 77;
        kws_fe_push(&fe, kws_fe_golden_pcm + pos, n, collect, NULL);
    }
    CHECK(s_nf == kws_fe_golden_frames, "frames %d, expected %d", s_nf, kws_fe_golden_frames);
    for (i = 0; i < kws_fe_golden_frames && i < s_nf; i++) {
        for (j = 0; j < KWS_BANDS; j++) { d = fmaxf(d, fabsf(s_feats[i][j] - kws_fe_golden_feat[i * KWS_BANDS + j])); }
    }
    printf("   largest difference %.2e (log-mel)\n", d);
    CHECK(d < 2e-3f, "features differ by %g", d);

    printf("2. engine against export.py's reference\n");
    CHECK(kws_model_ok(&kws_test_model), "test model accepted");
    d = compare_model(&kws_test_model, kws_test_model_test_in, kws_test_model_test_probs, kws_test_model_test_emb);
    printf("   largest difference %.2e (probabilities, speaker vector)\n", d);
    CHECK(d < 1e-4f, "outputs differ by %g", d);
    CHECK(strcmp(kws_label(&kws_test_model, 1, b, sizeof(b)), "help me") == 0, "label 1 = %s", b);
    CHECK(kws_is_keyword(&kws_test_model, 2) && !kws_is_keyword(&kws_test_model, 3), "'_other' is not a keyword");
    {
        float *work = malloc(sizeof(float) * kws_engine_work(&kws_test_model));
        float p[KWS_MAX_NCLS];
        printf("   work memory %u floats (%u KB)\n", (unsigned)kws_engine_work(&kws_test_model),
               (unsigned)(kws_engine_work(&kws_test_model) * 4u / 1024u));
        CHECK(kws_engine_run(&kws_test_model, kws_test_model_test_in, work, p, NULL), "runs without a speaker vector");
        free(work);
    }

    printf("2b. v1.5.1 fast engine against the plain one\n");
    {
        float di;
        double share;
        int st;
        for (st = 4; st <= 24; st += 4) {
            d = compare_fast(&kws_test_model, st, &di, &share);
            if (st == 12 || st == 4) {
                printf("   test model, window moved by %2d frames: difference %.1e, moved runs %.1e, work %.0f %%\n",
                       st, d, di, share * 100.0);
            }
            CHECK(d < 1e-5f && di == 0.0f, "step %d: fast %g, moved %g", st, d, di);
        }
        if (kws_model.ncls) {
            d = compare_fast(&kws_model, 12, &di, &share);
            printf("   firmware model, moved by 12: difference %.1e, moved runs %.1e, work %.0f %%\n", d, di, share * 100.0);
            CHECK(d < 1e-5f && di == 0.0f, "firmware model: fast %g, moved %g", d, di);
        }
        d = compare_fast(&kws_test_model, 6, &di, &share);   /* not a multiple of 4: full runs */
        CHECK(d < 1e-5f && di == 0.0f && share > 0.99, "step 6: always full (%g %g %.2f)", d, di, share);
    }

    printf("3. decision logic\n");
    {
        kws_model_t m = kws_test_model;
        kws_det_t det;
        kws_voice_t voice;
        float *work = malloc(sizeof(float) * kws_engine_work(&m));
        static float ring[KWS_MAX_T * KWS_BANDS], win[KWS_MAX_T * KWS_BANDS];
        float frame[KWS_BANDS];
        int ran = 0, hits = 0, sos = 0, rej = 0, ev;
        uint32_t t, t_sos = 0;

        m.thr[0] = m.thr[1] = m.thr[2] = 0.0f;       /* every run is a "hit": test the counting */
        kws_det_init(&det, &m, work, ring, win);
        for (t = 0; t < 4000; t += 20) {
            for (j = 0; j < KWS_BANDS; j++) { frame[j] = kws_test_model_test_in[((t / 20) % m.T) * KWS_BANDS + j]; }
            ev = kws_det_frame(&det, frame, -80.0f, t);
            ran += (ev & KWS_EV_RAN) != 0;
        }
        CHECK(ran == 0, "quiet: the CNN never runs (%d)", ran);
        for (t = 4000; t < 12000; t += 20) {
            for (j = 0; j < KWS_BANDS; j++) { frame[j] = kws_test_model_test_in[((t / 20) % m.T) * KWS_BANDS + j]; }
            ev = kws_det_frame(&det, frame, -30.0f, t);
            ran += (ev & KWS_EV_RAN) != 0;
            hits += (ev & KWS_EV_HIT) != 0;
            if ((ev & KWS_EV_SOS) && !t_sos) { t_sos = t; sos++; }
        }
        printf("   8 s loud: %d runs, %d hits, SOS after %u ms\n", ran, hits, (unsigned)(t_sos - 4000));
        CHECK(ran >= 30 && ran <= 34, "a run every 240 ms (%d)", ran);
        /* v1.6: one sound without any pause = one call, counted again only every 2.5 s */
        CHECK(hits >= 3 && hits <= 4, "calling on without a pause: again every 2.5 s (%d)", hits);
        CHECK(sos == 1 && t_sos - 4000 < 6000, "3 hits -> SOS");

        /* Only My Voice: enrolled voice = the opposite of what is heard -> rejected */
        memset(&voice, 0, sizeof(voice));
        {
            float probs[KWS_MAX_NCLS], emb[KWS_MAX_SPK];
            kws_engine_run(&m, kws_test_model_test_in, work, probs, emb);
            for (j = 0; j < m.spk_dim; j++) { emb[j] = -emb[j]; }
            for (i = 0; i < KWS_ENROL_PER_WORD; i++) {
                kws_voice_add(&voice, m.spk_dim, 0, emb);
                kws_voice_add(&voice, m.spk_dim, 1, emb);
                kws_voice_add(&voice, m.spk_dim, 2, emb);
            }
        }
        CHECK(kws_voice_ready(&m, &voice, KWS_ENROL_TAKES), "enrolment complete");
        kws_det_init(&det, &m, work, ring, win);
        det.voice = &voice;
        det.only_me = true;
        hits = sos = 0;
        for (t = 0; t < 8000; t += 20) {
            for (j = 0; j < KWS_BANDS; j++) { frame[j] = kws_test_model_test_in[((t / 20) % m.T) * KWS_BANDS + j]; }
            ev = kws_det_frame(&det, frame, -30.0f, t);
            hits += (ev & KWS_EV_HIT) != 0;
            rej += (ev & KWS_EV_REJECT) != 0;
            sos += (ev & KWS_EV_SOS) != 0;
        }
        CHECK(hits == 0 && sos == 0 && rej >= 2, "another voice is rejected (hits %d, rejected %d)", hits, rej);
        det.only_me = false;
        hits = 0;
        for (; t < 12000; t += 20) {
            for (j = 0; j < KWS_BANDS; j++) { frame[j] = kws_test_model_test_in[((t / 20) % m.T) * KWS_BANDS + j]; }
            ev = kws_det_frame(&det, frame, -30.0f, t);
            hits += (ev & KWS_EV_HIT) != 0;
        }
        CHECK(hits >= 2, "without Only My Voice it counts again (%d)", hits);
        free(work);
    }

    printf("3b. v1.4.1: window gate, per class thresholds, speaker hook\n");
    {
        kws_model_t m = kws_test_model;
        kws_det_t det;
        float *work = malloc(sizeof(float) * kws_engine_work(&m));
        static float ring[KWS_MAX_T * KWS_BANDS], win[KWS_MAX_T * KWS_BANDS];
        float frame[KWS_BANDS];
        int ran = 0, ev, hitcls[KWS_MAX_NCLS] = { 0 }, rej = 0, ok = 0;
        uint32_t t;

        m.thr[0] = m.thr[1] = m.thr[2] = 0.0f;
        kws_det_init(&det, &m, work, ring, win);
        for (t = 0; t < 2000; t += 20) {             /* fill the window with silence */
            for (j = 0; j < KWS_BANDS; j++) { frame[j] = kws_test_model_test_in[((t / 20) % m.T) * KWS_BANDS + j]; }
            kws_det_frame(&det, frame, -80.0f, t);
        }
        for (t = 2000; t < 6000; t += 20) {          /* one short word (300 ms), then silence */
            for (j = 0; j < KWS_BANDS; j++) { frame[j] = kws_test_model_test_in[((t / 20) % m.T) * KWS_BANDS + j]; }
            ev = kws_det_frame(&det, frame, (t < 2300) ? -30.0f : -80.0f, t);
            ran += (ev & KWS_EV_RAN) != 0;
        }
        printf("   a 0.3 s word in silence: %d runs (window %d ms)\n", ran, m.T * 20);
        CHECK(ran >= m.T / KWS_STEP - 1 && ran <= m.T / KWS_STEP + 2, "the CNN keeps running while the word is in the window (%d)", ran);

        /* class 0 made impossible by its offset: only the other keywords are hit */
        m.cls_off[0] = 2.0f;
        kws_det_init(&det, &m, work, ring, win);
        for (t = 0; t < 12000; t += 20) {
            for (j = 0; j < KWS_BANDS; j++) { frame[j] = kws_test_model_test_in[((t / 20) % m.T) * KWS_BANDS + j]; }
            ev = kws_det_frame(&det, frame, -30.0f, t);
            if (ev & KWS_EV_HIT) { hitcls[det.hit_cls]++; }
        }
        printf("   cls_off[0] = +2: hits per class %d %d %d\n", hitcls[0], hitcls[1], hitcls[2]);
        CHECK(hitcls[0] == 0 && hitcls[1] + hitcls[2] > 0, "cls_off: class 0 never, others yes (%d %d %d)", hitcls[0], hitcls[1], hitcls[2]);
        m.cls_off[0] = 0.0f;

        /* speaker hook (a separate model): rejects, then accepts */
        {
            float c0 = 0.1f, c1 = 0.9f;
            kws_det_init(&det, &m, work, ring, win);
            det.only_me = true;
            det.verify_thr = 0.5f;
            det.verify = verify_fixed;
            det.verify_ctx = &c0;
            for (t = 0; t < 6000; t += 20) {
                for (j = 0; j < KWS_BANDS; j++) { frame[j] = kws_test_model_test_in[((t / 20) % m.T) * KWS_BANDS + j]; }
                ev = kws_det_frame(&det, frame, -30.0f, t);
                rej += (ev & KWS_EV_REJECT) != 0;
                ok += (ev & KWS_EV_HIT) != 0;
            }
            CHECK(rej > 0 && ok == 0, "voice 0.1 < 0.5: rejected (%d / %d)", rej, ok);
            det.verify_ctx = &c1;
            rej = ok = 0;
            for (; t < 12000; t += 20) {
                for (j = 0; j < KWS_BANDS; j++) { frame[j] = kws_test_model_test_in[((t / 20) % m.T) * KWS_BANDS + j]; }
                ev = kws_det_frame(&det, frame, -30.0f, t);
                rej += (ev & KWS_EV_REJECT) != 0;
                ok += (ev & KWS_EV_HIT) != 0;
            }
            CHECK(rej == 0 && ok > 0, "voice 0.9: accepted (%d / %d)", rej, ok);
        }
        free(work);
    }

    printf("3c. v1.5.1: moved-window runs in the detector, 2 calls / 10 s, voice checked again\n");
    {
        kws_model_t m = kws_model.ncls ? kws_model : kws_test_model;
        kws_det_t a, b;
        float *wa = malloc(sizeof(float) * kws_engine_work(&m)), *wb = malloc(sizeof(float) * kws_engine_work(&m));
        static float ra[KWS_MAX_T * KWS_BANDS], va[KWS_MAX_T * KWS_BANDS], rb[KWS_MAX_T * KWS_BANDS], vb[KWS_MAX_T * KWS_BANDS];
        float frame[KWS_BANDS], dmax = 0.0f;
        int ev, eb, ran = 0, hits = 0, sos = 0, rej = 0;
        uint32_t t, seed = 99u, t_sos = 0;

        kws_det_init(&a, &m, wa, ra, va);
        kws_det_init(&b, &m, wb, rb, vb);
        for (t = 0; t < 20000; t += 20) {
            for (j = 0; j < KWS_BANDS; j++) { seed = seed * 1664525u + 1013904223u; frame[j] = -12.0f + 10.0f * (float)(seed >> 8) / 16777216.0f; }
            ev = kws_det_frame(&a, frame, (t / 1000) % 4 == 3 ? -80.0f : -30.0f, t);   /* with silent gaps */
            b.since = -1;                              /* b: always full runs */
            eb = kws_det_frame(&b, frame, (t / 1000) % 4 == 3 ? -80.0f : -30.0f, t);
            if ((ev & KWS_EV_RAN) != (eb & KWS_EV_RAN)) { dmax = 1.0f; }
            if (ev & KWS_EV_RAN) {
                ran++;
                for (i = 0; i < m.ncls; i++) { dmax = fmaxf(dmax, fabsf(a.probs[i] - b.probs[i])); }
            }
            if ((ev & KWS_EV_RAN) && t == 10000) { kws_best_window(&m, wa, va, m.T, 0, NULL, NULL); }  /* someone else uses a's work */
        }
        printf("   %d runs, moved-window vs full runs: largest difference %.1e\n", ran, dmax);
        CHECK(ran > 40 && dmax == 0.0f, "detector: moved-window runs give the same (%d, %g)", ran, dmax);

        m.thr[0] = m.thr[1] = m.thr[2] = 0.0f;
        for (i = 0; i < KWS_MAX_NCLS; i++) { m.cls_off[i] = 0.0f; }
        kws_det_init(&a, &m, wa, ra, va);
        a.need_hits = 2;
        a.need_ms = 10000;
        for (t = 0; t < 6000; t += 20) {
            for (j = 0; j < KWS_BANDS; j++) { frame[j] = kws_test_model_test_in[((t / 20) % 50) * KWS_BANDS + j]; }
            ev = kws_det_frame(&a, frame, -30.0f, t);
            hits += (ev & KWS_EV_HIT) != 0;
            if ((ev & KWS_EV_SOS) && !t_sos) { t_sos = t; sos++; }
        }
        printf("   2 hits / 10 s: SOS after %u ms\n", (unsigned)t_sos);
        CHECK(sos == 1 && t_sos < 5000, "2 hits -> SOS (%u ms)", (unsigned)t_sos);

        kws_det_init(&a, &m, wa, ra, va);
        a.only_me = true;
        a.verify = verify_second;
        a.verify_thr = 0.5f;
        s_vcalls = 0;
        hits = rej = 0;
        for (t = 0; t < 2400; t += 20) {
            for (j = 0; j < KWS_BANDS; j++) { frame[j] = kws_test_model_test_in[((t / 20) % 50) * KWS_BANDS + j]; }
            ev = kws_det_frame(&a, frame, -30.0f, t);
            hits += (ev & KWS_EV_HIT) != 0;
            rej += (ev & KWS_EV_REJECT) != 0;
        }
        CHECK(hits == 1 && rej == 0 && s_vcalls == 2, "voice checked again on the next run (hits %d, rejected %d, checks %d)", hits, rej, s_vcalls);
        free(wa); free(wb);
    }

    printf("3d. v1.6: every call counted once (scripted keyword scores)\n");
    {
        static const struct { const char *what; int n; uint32_t c[8][3]; int sos, hits; uint32_t end; } sc[] = {
            /* {start ms, length ms, keyword?} */
            { "one call", 1, { { 2000, 400, 1 } }, 0, 1, 8000 },
            { "one long call (1.3 s)", 1, { { 2000, 1300, 1 } }, 0, 1, 8000 },
            { "one call with a pause inside (\"hel - p\")", 2, { { 2000, 250, 1 }, { 2370, 80, 0 } }, 0, 1, 8000 },
            { "two quick calls (250 ms pause)", 2, { { 2000, 400, 1 }, { 2650, 400, 1 } }, 1, 2, 8000 },
            { "two quick calls (160 ms pause)", 2, { { 2000, 350, 1 }, { 2510, 350, 1 } }, 1, 2, 8000 },
            { "two calls 1.5 s apart", 2, { { 2000, 400, 1 }, { 3500, 400, 1 } }, 1, 2, 8000 },
            { "two calls 4 s apart", 2, { { 2000, 400, 1 }, { 6000, 400, 1 } }, 1, 2, 10000 },
            { "a cough, then one call", 2, { { 2000, 300, 0 }, { 3000, 400, 1 } }, 0, 1, 8000 },
            { "SOS, alarm beeps, one call (1 of 2), one more call: SOS", 8,
              { { 2000, 400, 1 }, { 2650, 400, 1 }, { 4000, 200, 0 }, { 4500, 200, 0 }, { 5000, 200, 0 },
                { 5500, 200, 0 }, { 9000, 400, 1 }, { 13000, 400, 1 } }, 2, 4, 17000 },
            { "SOS, silence, one call, one more call: SOS", 4,
              { { 2000, 400, 1 }, { 2650, 400, 1 }, { 9000, 400, 1 }, { 13000, 400, 1 } }, 2, 4, 17000 },
        };
        kws_model_t m = kws_test_model;
        kws_det_t det;
        float *work = malloc(sizeof(float) * kws_engine_work(&m));
        static float ring[KWS_MAX_T * KWS_BANDS], win[KWS_MAX_T * KWS_BANDS];
        float frame[KWS_BANDS];
        unsigned q;
        m.thr[0] = m.thr[1] = m.thr[2] = 0.5f;
        for (i = 0; i < KWS_MAX_NCLS; i++) { m.cls_off[i] = 0.0f; }
        for (q = 0; q < 2 * sizeof(sc) / sizeof(sc[0]); q++) {
            int hits = 0, sos = 0, ev, c, mid_ok = 1;
            uint32_t t;
            bool me = q >= sizeof(sc) / sizeof(sc[0]);          /* second round: Only My Voice on */
            float ok = 0.8f;
            if (me) { q -= sizeof(sc) / sizeof(sc[0]); }
            s_script = &sc[q].c[0][0];
            s_script_n = sc[q].n;
            kws_det_init(&det, &m, work, ring, win);
            det.need_hits = 2;
            det.need_ms = 10000;
            det.fake_run = fake_run;
            if (me) { det.only_me = true; det.verify = verify_fixed; det.verify_ctx = &ok; det.verify_thr = 0.5f; }
            for (t = 0; t < sc[q].end; t += 20) {
                bool loud = false;
                for (c = 0; c < sc[q].n; c++) { if (t >= sc[q].c[c][0] && t < sc[q].c[c][0] + sc[q].c[c][1]) { loud = true; } }
                for (j = 0; j < KWS_BANDS; j++) { frame[j] = -8.0f; }
                ev = kws_det_frame(&det, frame, loud ? -20.0f : -60.0f, t);
                if (ev & KWS_EV_HIT) { hits++; }
                if (ev & KWS_EV_SOS) { sos++; }
                if ((q == 8 || q == 9) && t == 12000) { mid_ok = sos == 1 && det.n_hits == 1; }
            }
            printf("   %s%-58s %d counted, %d SOS\n", me ? "[my voice] " : "", sc[q].what, hits, sos);
            CHECK(sos == sc[q].sos && hits == sc[q].hits && mid_ok, "%s%s: %d counted, %d SOS", me ? "my voice: " : "", sc[q].what, hits, sos);
            if (me) { q += sizeof(sc) / sizeof(sc[0]); }
        }
        free(work);
    }

    printf("3e. v1.6b: the second call needs a little less voice match\n");
    {
        static const uint32_t two[] = { 2000, 400, 1, 3500, 400, 1 };
        kws_model_t m = kws_test_model;
        kws_det_t det;
        float *work = malloc(sizeof(float) * kws_engine_work(&m));
        static float ring[KWS_MAX_T * KWS_BANDS], win[KWS_MAX_T * KWS_BANDS];
        float frame[KWS_BANDS], vs[] = { 0.60f, 0.48f, 0.48f, 0.48f };
        int sos = 0, c;
        uint32_t t;
        m.thr[0] = m.thr[1] = m.thr[2] = 0.5f;
        for (i = 0; i < KWS_MAX_NCLS; i++) { m.cls_off[i] = 0.0f; }
        s_script = two;
        s_script_n = 2;
        s_vseq = vs;
        s_vcalls = 0;
        kws_det_init(&det, &m, work, ring, win);
        det.need_hits = 2; det.need_ms = 10000;
        det.fake_run = fake_run;
        det.only_me = true; det.verify = verify_seq; det.verify_thr = 0.5f;
        for (t = 0; t < 8000; t += 20) {
            bool loud = false;
            for (c = 0; c < 2; c++) { if (t >= two[c * 3] && t < two[c * 3] + two[c * 3 + 1]) { loud = true; } }
            for (j = 0; j < KWS_BANDS; j++) { frame[j] = -8.0f; }
            sos += (kws_det_frame(&det, frame, loud ? -20.0f : -60.0f, t) & KWS_EV_SOS) != 0;
        }
        CHECK(sos == 1, "voice 0.60 then 0.48 (needs 0.50, 2nd call 0.47): SOS (%d)", sos);
        /* v1.6d: an imitation: 0.40 on every try (clearly rejected), then 0.52 - which
         * would pass 0.50, but after the rejections it needs more */
        {
            static const uint32_t many[] = { 2000, 400, 1, 5000, 400, 1, 8000, 400, 1, 11000, 400, 1 };
            static const float iv[] = { 0.40f, 0.40f, 0.40f, 0.40f, 0.40f, 0.40f, 0.40f, 0.40f, 0.40f, 0.52f, 0.52f, 0.52f };
            int hits = 0, rej = 0, ev;
            s_script = many; s_script_n = 4;
            s_vseq = iv; s_vcalls = 0;
            kws_det_init(&det, &m, work, ring, win);
            det.need_hits = 2; det.need_ms = 10000;
            det.fake_run = fake_run;
            det.only_me = true; det.verify = verify_seq12; det.verify_thr = 0.5f;
            for (t = 0; t < 14000; t += 20) {
                bool loud = false;
                for (c = 0; c < 4; c++) { if (t >= many[c * 3] && t < many[c * 3] + many[c * 3 + 1]) { loud = true; } }
                ev = kws_det_frame(&det, frame, loud ? -20.0f : -60.0f, t);
                hits += (ev & KWS_EV_HIT) != 0;
                rej += (ev & KWS_EV_REJECT) != 0;
            }
            CHECK(hits == 0 && rej == 4, "after 3 clear misses 0.52 is not enough (needs 0.54) (hits %d, rejected %d)", hits, rej);
        }
        /* v1.6e: the owner after the imitation: near misses (0.47 of 0.50) add no
         * penalty, so they never lock him out; 15 s after the imitation's clear
         * misses his 0.55 counts again */
        {
            static const uint32_t seq[] = { 2000, 400, 1, 5000, 400, 1, 8000, 400, 1, 11000, 400, 1, 21000, 400, 1, 24000, 400, 1 };
            static const float iv[] = { 0.40f, 0.40f, 0.40f, 0.40f, 0.40f, 0.40f,      /* imitation: clear misses */
                                        0.47f, 0.47f, 0.47f, 0.47f, 0.47f, 0.47f,      /* owner, near misses      */
                                        0.55f, 0.55f, 0.55f, 0.55f, 0.55f, 0.55f };    /* owner later             */
            int hits = 0, sos = 0, ev;
            s_script = seq; s_script_n = 6;
            s_vseq = iv; s_vcalls = 0; s_vmax = 17;
            kws_det_init(&det, &m, work, ring, win);
            det.need_hits = 2; det.need_ms = 10000;
            det.fake_run = fake_run;
            det.only_me = true; det.verify = verify_seq12; det.verify_thr = 0.5f;
            for (t = 0; t < 28000; t += 20) {
                bool loud = false;
                for (c = 0; c < 6; c++) { if (t >= seq[c * 3] && t < seq[c * 3] + seq[c * 3 + 1]) { loud = true; } }
                ev = kws_det_frame(&det, frame, loud ? -20.0f : -60.0f, t);
                hits += (ev & KWS_EV_HIT) != 0;
                sos += (ev & KWS_EV_SOS) != 0;
            }
            s_vmax = 11;
            CHECK(hits == 2 && sos == 1, "the owner is not locked out by the imitation (hits %d, SOS %d)", hits, sos);
        }
        /* v1.6f: an imitated "help" makes only "help" stricter: the owner's SOS
         * right after it needs just its own value */
        {
            static const uint32_t seq[] = { 2000, 400, 1, 5000, 400, 1, 7500, 400, 3 };
            static const float iv[] = { 0.40f, 0.40f, 0.40f, 0.40f, 0.40f, 0.40f, 0.51f };
            int hits = 0, ev;
            s_script = seq; s_script_n = 3;
            s_vseq = iv; s_vcalls = 0; s_vmax = 6;
            kws_det_init(&det, &m, work, ring, win);
            det.need_hits = 2; det.need_ms = 10000;
            det.fake_run = fake_run;
            det.only_me = true; det.verify = verify_seq12; det.verify_thr = 0.5f;
            for (t = 0; t < 10000; t += 20) {
                bool loud = false;
                for (c = 0; c < 3; c++) { if (t >= seq[c * 3] && t < seq[c * 3] + seq[c * 3 + 1]) { loud = true; } }
                ev = kws_det_frame(&det, frame, loud ? -20.0f : -60.0f, t);
                hits += (ev & KWS_EV_HIT) != 0;
            }
            s_vmax = 11;
            CHECK(hits == 1 && det.hit_cls == 2, "imitated help, then the owner's SOS 0.51 counts (hits %d, class %d)", hits, det.hit_cls);
        }
        free(work);
    }

    printf("3f. v1.6.1: two calls said straight after each other (words one by one)\n");
    {
        /* {start ms, length ms, marker, dB}: marker 1..3 = keyword class + 1 (the
         * scripted model hears it), 9 = a word that is no keyword ("me", the "p"),
         * 0 = the dip between words (still loud, no pause) */
        static const int sc[][5][4] = {
            { { 2000, 300, 1, -20 }, { 2300, 80, 0, -32 }, { 2380, 300, 1, -20 } },              /* help-help */
            { { 2000, 300, 1, -20 }, { 2300, 80, 0, -32 }, { 2380, 250, 9, -20 } },              /* help me   */
            { { 2000, 250, 1, -20 }, { 2250, 50, 0, -40 }, { 2300, 60, 9, -22 } },               /* hel-p     */
            { { 2000, 300, 3, -20 }, { 2300, 80, 0, -32 }, { 2380, 300, 3, -20 } },              /* sos-sos   */
            { { 2000, 300, 1, -20 }, { 2300, 80, 0, -32 }, { 2380, 300, 1, -20 }, { 2680, 80, 0, -32 },
              { 2760, 300, 1, -20 } },                                                          /* 3 x, rule 3 */
        };
        static const int nsc[] = { 3, 3, 3, 3, 5 }, want_hits[] = { 2, 1, 1, 2, 1 }, want_sos[] = { 1, 0, 0, 1, 0 };
        static const char *what[] = { "\"Help-help\" without a pause -> 2 calls, SOS", "\"Help me\" -> 1 call",
                                      "\"Hel-p\" -> 1 call", "\"SOS-SOS\" without a pause -> SOS",
                                      "3 calls / 20 s: \"help-help-help\" without pauses -> 1 call" };
        kws_model_t m = kws_test_model;
        kws_det_t det;
        float *work = malloc(sizeof(float) * kws_engine_work(&m)), *work2 = malloc(sizeof(float) * kws_engine_work(&m));
        static float ring[KWS_MAX_T * KWS_BANDS], win[KWS_MAX_T * KWS_BANDS], win2[KWS_MAX_T * KWS_BANDS];
        float frame[KWS_BANDS], ok = 0.8f;
        int q, me;
        m.thr[0] = m.thr[1] = m.thr[2] = 0.5f;
        for (i = 0; i < KWS_MAX_NCLS; i++) { m.cls_off[i] = 0.0f; }
        for (me = 0; me < 2; me++) {
            for (q = 0; q < 5; q++) {
                int hits = 0, sos = 0, ev, c;
                uint32_t t;
                s_sc4 = &sc[q][0][0];
                s_sc4_n = nsc[q];
                kws_det_init(&det, &m, work, ring, win);
                det.need_hits = q == 4 ? 3 : 2;
                det.need_ms = q == 4 ? 20000 : 10000;
                det.words_on = q != 4;
                det.work2 = work2;
                det.win2 = win2;
                det.fake_run = fake_run4;
                if (me) { det.only_me = true; det.verify = verify_fixed; det.verify_ctx = &ok; det.verify_thr = 0.5f; }
                for (t = 0; t < 8000; t += 20) {
                    float db = -60.0f;
                    for (j = 0; j < KWS_BANDS; j++) { frame[j] = -8.0f; }
                    for (c = 0; c < nsc[q]; c++) {
                        if ((int)t >= sc[q][c][0] && (int)t < sc[q][c][0] + sc[q][c][1]) { db = (float)sc[q][c][3]; frame[0] = (float)sc[q][c][2]; }
                    }
                    ev = kws_det_frame(&det, frame, db, t);
                    hits += (ev & KWS_EV_HIT) != 0;
                    sos += (ev & KWS_EV_SOS) != 0;
                }
                printf("   %s%-60s %d counted, %d SOS\n", me ? "[my voice] " : "", what[q], hits, sos);
                CHECK(hits == want_hits[q] && sos == want_sos[q], "%s%s: %d counted, %d SOS", me ? "my voice: " : "", what[q], hits, sos);
            }
        }
        free(work); free(work2);
    }

    printf("4. the firmware's model (Ai/kws_model.c): %s\n", kws_model.ncls ? kws_model.name : "none yet");
    if (kws_model.ncls) {
        CHECK(kws_model_ok(&kws_model), "model within the limits");
#ifdef HAVE_MODEL_TEST
        d = compare_model(&kws_model, kws_model_test_in, kws_model_test_probs, kws_model_test_emb);
        printf("   largest difference %.2e\n", d);
        CHECK(d < 1e-4f, "kws_model differs from its test by %g", d);
#endif
    } else {
        CHECK(!kws_model_ok(&kws_model), "no model: Voice SOS stays off");
    }

    printf("%d checks, %d failed\n", checks, failed);
    return failed ? 1 : 0;
}
