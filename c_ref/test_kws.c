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
        CHECK(hits >= 5 && hits <= 7, "one hit per 1.2 s at most (%d)", hits);
        CHECK(sos == 1 && t_sos - 4000 < 4500, "3 hits -> SOS");

        /* Only My Voice: enrolled voice = the opposite of what is heard -> rejected */
        memset(&voice, 0, sizeof(voice));
        {
            float probs[KWS_MAX_NCLS], emb[KWS_MAX_SPK];
            kws_engine_run(&m, kws_test_model_test_in, work, probs, emb);
            for (j = 0; j < m.spk_dim; j++) { emb[j] = -emb[j]; }
            for (i = 0; i < KWS_ENROL_PER_WORD; i++) {
                kws_voice_add(&m, &voice, 0, emb);
                kws_voice_add(&m, &voice, 1, emb);
                kws_voice_add(&m, &voice, 2, emb);
            }
        }
        CHECK(kws_voice_ready(&m, &voice), "enrolment complete");
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
        CHECK(hits == 0 && sos == 0 && rej >= 4, "another voice is rejected (hits %d, rejected %d)", hits, rej);
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
