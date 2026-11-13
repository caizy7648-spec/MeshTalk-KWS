/* kws.c - v1.4 voice SOS on the ESP32. See kws.h. */
#include "kws.h"
#include "kws_frontend.h"
#include "kws_engine.h"
#include "kws_detect.h"
#include "spk_engine.h"
#include "spk_model.h"
#include "audio_i2s.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_attr.h"
#include "nvs.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/idf_additions.h"
#include "freertos/stream_buffer.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static const char *TAG = "kws";

#define CHUNK           KWS_HOP                 /* 20 ms */
#define ENROL_MS        2600u                   /* v1.5.1: 2.6 s (was 2.2): time to react */
#define ENROL_FRAMES    ((KWS_RATE * ENROL_MS / 1000u - KWS_FRAME) / KWS_HOP + 1u)
#define ENROL_PAD       20                      /* v1.5.1: quiet frames added on both sides, so a
                                                 * phrase right at the start / end still fits a window */
#define ENROL_MIN_DBFS  (-45.0f)                /* quieter: "speak louder"            */
#define ENROL_MIN_PROB  0.15f                   /* v1.6 (v1.5 0.30, v1.5.1 0.20): the asked phrase ... */
#define ENROL_ANY_PROB  0.35f                   /* ... or the keywords together ("a call for help",
                                                 * e.g. an SOS the model half takes for "help") */
#define VOICE_MAGIC     0x3356574Bu             /* "KWV3" (v1.5.1: voices of v1.5 are void - they
                                                 * were made from chopped sound / another cut) */
#define KWS_STACK       (8 * 1024)

typedef struct {
    uint32_t    magic, model_id, spk_id;        /* the voice is only valid for these models */
    int32_t     dim, ncls;
    kws_voice_t v;
    float       agree;                          /* v1.5.1: how well the takes matched each other */
} voice_rec_t;

/* v1.6c: voice match levels (Loose / Normal / Strict) per keyword: the
 * similarity a call needs. From the owner's calls on the device (Only My
 * Voice, after enrolment): help 0.65-0.70 (another day 0.49-0.59), help me
 * 0.60-0.67, SOS 0.73-0.77 - SOS matches best, "help me" least, so SOS can be
 * stricter and "help me" a little looser. Lowered a little for a voice whose
 * takes vary (see voice_thr). */
static const float s_match_tab[3][3] = {
    /*            help   help me   sos */
    /* v1.6d: all +0.06 - a woman imitating the owner's lower voice reached
     * 0.28-0.55 on "help" and got through Strict (0.54) after many tries.
     * v1.6g: SOS -0.06 again: another day the owner's own SOS scored only
     * 0.41-0.69 (most 0.41-0.58) - SOS varies most from day to day */
#if SPK_USE_BIG
    /* v1.6.4: the big model (spk_model.h) scores ~0.16 lower. Each level is the
     * old one moved to the SAME false-accept rate of other speakers (test set,
     * firmware scoring, 406 080 impostor trials); there the owner is rejected
     * 13-20 % less often (e.g. old 0.52: FAR 6.96 %, FRR 8.3 % -> 0.3575: FRR 6.6 %) */
    /* Loose  */ { 0.298f, 0.279f, 0.288f },
    /* Normal */ { 0.358f, 0.337f, 0.358f },
    /* Strict */ { 0.444f, 0.422f, 0.465f },
#else
    /* Loose  */ { 0.46f, 0.44f, 0.45f },
    /* Normal */ { 0.52f, 0.50f, 0.52f },
    /* Strict */ { 0.60f, 0.58f, 0.62f },
#endif
};
#if SPK_USE_BIG
#define VOICE_THR_FLOOR 0.14f                   /* 0.30 of the old model, same shift */
#else
#define VOICE_THR_FLOOR 0.30f
#endif


static SemaphoreHandle_t s_mx;
static bool      s_mic_ok, s_audio_ok, s_model_ok, s_task_ok, s_task_psram;
static volatile bool s_save_voice;           /* v1.5: written to NVS by kws_service() (UI task) */
static volatile bool s_on, s_mic_on;
static bool      s_want_on;
static volatile int  s_hold;
static uint8_t   s_sens = KWS_SENS_MED;
static bool      s_only_me;
static uint8_t   s_rule;                        /* v1.5.1: 0 = 2 calls within 10 s, 1 = 3 within 20 s */
static uint8_t   s_match = 1;                   /* v1.5.1: voice match 0 loose, 1 normal, 2 strict */
static float     s_agree = -2.0f;               /* of the enrolled voice, -2 = unknown */
static float     s_run_avg;                     /* v1.5.1: average check time (ms) */
static volatile bool s_verified;                /* the voice was checked in this run (slow) */
EXT_RAM_BSS_ATTR static float s_takes[KWS_ENROL_MAX][KWS_VOICE_DIM];   /* v1.5.1: every take's vector */
static int8_t    s_take_cls[KWS_ENROL_MAX];
/* v1.5: the big ones live in PSRAM (EXT_RAM_BSS_ATTR): together ~45 KB that
 * v1.4 took from the internal RAM - Wi-Fi and the LoRa task then failed */
EXT_RAM_BSS_ATTR static kws_voice_t s_voice, s_pending;
static bool      s_enrolled, s_stale;
static int       s_pending_n;
EXT_RAM_BSS_ATTR static kws_det_t s_det;
EXT_RAM_BSS_ATTR static kws_fe_t s_fe;
static float    *s_work, *s_ring, *s_win, *s_efeat, *s_swork, *s_sfeat;
static float    *s_work2, *s_win2;          /* v1.6.1: checking words one by one */
static volatile bool s_sos;
/* status for the UI */
static float     s_level = -120.0f, s_smooth[KWS_MAX_NCLS];
static float     s_peak[KWS_MAX_NCLS], s_burst_loud;      /* v1.5: best of the last / current sound */
static bool      s_burst;
static int       s_burst_runs;
static int       s_last_cls = -1;
static float     s_last_score, s_last_spk = -2.0f, s_last_need;
static bool      s_last_rej;
static int64_t   s_last_us;
static uint16_t  s_run_ms;
static uint32_t  s_runs, s_diag_runs, s_frames, s_short_reads;
static bool      s_ran;                     /* v1.5: a CNN run just happened: give the CPU away */
static int64_t   s_diag_t0;
/* enrolment */
static volatile int s_en_state = BK_EN_IDLE, s_en_take = -1, s_en_req = -1;
static float     s_en_score;
static int       s_en_frames;
static float     s_en_loud;

static void lock(void)   { xSemaphoreTake(s_mx, portMAX_DELAY); }
static void unlock(void) { xSemaphoreGive(s_mx); }

/* ------------------------------------------------- the voice print ---- */
/* v1.4.1: a separate speaker model (Ai/spk_engine.h) when one is built in,
 * else the keyword model's own speaker head (spk_dim > 0), else none. */
static bool s_spk;                              /* separate model present and memory for it */

static int voice_dim(void)   { return s_spk ? spk_dim() : (s_model_ok ? kws_model.spk_dim : 0); }
static bool voice_ok(void)   { return s_model_ok && voice_dim() > 0 && voice_dim() <= KWS_VOICE_DIM; }
static int enrol_takes(void);
/* a usable stored voice: the separate model only needs 3+ good takes (the
 * worst ones are left out, v1.5.1); the keyword head needs every keyword */
static bool voice_ready(const kws_voice_t *v)
{
    return s_spk ? v->n_all >= 3 : kws_voice_ready(&kws_model, v, enrol_takes());
}

/* row of the table for keyword k: help 0, help me 1, sos 2 (others: help) */
static int match_row(int k)
{
    char b[16];
    kws_label(&kws_model, k, b, sizeof(b));
    return strcmp(b, "sos") == 0 ? 2 : strcmp(b, "help me") == 0 ? 1 : 0;
}

/* v1.5.1: the similarity a call needs: the level, but up to 0.12 lower when
 * the owner's own takes agree less (a varying voice must still pass);
 * v1.6c: per keyword (k < 0: "help") */
static float voice_thr(int k)
{
    float t = s_match_tab[s_match > 2 ? 2 : s_match][k < 0 ? 0 : match_row(k)], a = s_agree - 0.10f;
    if (s_agree > -1.0f && a < t) { t = a > t - 0.06f ? a : t - 0.06f; }   /* v1.6d: at most 0.06 (was 0.12) */
    return t < VOICE_THR_FLOOR ? VOICE_THR_FLOOR : t;
}

/* v1.6.7: single "help" (the short, weak keyword) was the false-SOS source:
 * coughs and chatter. Its threshold goes up by HELP_EXTRA at every
 * sensitivity, and it no longer gets the confirm drop / word slack, so every
 * "help" needs the full value (medium: 0.55 - 0.10 + 0.05 = 0.50; before, a
 * second call needed only 0.35-0.37). "help me" and SOS are unchanged. */
#define HELP_EXTRA      0.05f

static void apply_opts(void)
{
    int k;
    for (k = 0; k < kws_model.ncls && k < KWS_MAX_NCLS; k++) {
        bool help = kws_is_keyword(&kws_model, k) && match_row(k) == 0;
        s_det.cls_extra[k] = help ? HELP_EXTRA : 0.0f;
        s_det.no_relax[k] = help ? 1u : 0u;
    }
    s_det.need_hits = s_rule ? 3 : 2;
    s_det.need_ms = s_rule ? 20000u : 10000u;
    s_det.words_on = !s_rule;                   /* v1.6.1: 2 calls may be said straight after each other */
    if (s_spk) {
        s_det.verify_thr = voice_thr(-1);
        for (k = 0; k < kws_model.ncls && k < KWS_MAX_NCLS; k++) {
            s_det.verify_thr_k[k] = kws_is_keyword(&kws_model, k) ? voice_thr(k) : 0.0f;
            /* v1.6g: the owner's own SOS often misses clearly, so a clear miss
             * of SOS makes the next SOS only a little stricter */
            s_det.rej_step[k] = match_row(k) == 2 ? 0.01f : 0.0f;
        }
    }
}

static int enrol_takes(void)
{
    int n = s_spk ? spk_enrol_takes() : KWS_ENROL_TAKES;
    return n < 3 ? 3 : n > KWS_ENROL_MAX ? KWS_ENROL_MAX : n;
}

/* spk_frames() frames of feats (n rows) centred on row c; edges repeated */
static void spk_cut(const float *feats, int n, int c, float *out)
{
    int f = spk_frames(), i, r;
    for (i = 0; i < f; i++) {
        r = c - f / 2 + i;
        r = r < 0 ? 0 : r >= n ? n - 1 : r;
        memcpy(out + (size_t)i * KWS_BANDS, feats + (size_t)r * KWS_BANDS, sizeof(float) * KWS_BANDS);
    }
}

/* kws_detect hook: voice similarity of the window that just triggered */
static float verify_cb(void *ctx, const float *win, int T, int cls)
{
    float emb[KWS_VOICE_DIM], sim;
    int64_t t0 = esp_timer_get_time();
    (void)ctx;
    if (!s_enrolled) { return -2.0f; }
    s_verified = true;
    spk_cut(win, T, T / 2, s_sfeat);
    if (!spk_embed(s_sfeat, spk_frames(), s_swork, emb)) { return -2.0f; }
    /* v1.5.1: the better of this word's template and all takes together (a
     * word's own template has only 3-4 takes and varies more) */
    sim = kws_voice_similarity(&s_voice, spk_dim(), cls, emb);
    {
        float all = kws_voice_similarity(&s_voice, spk_dim(), -1, emb);
        if (all > sim) { sim = all; }
    }
    {   /* v1.6b: the checks of one call (the retries 240 ms apart) are also
         * averaged - like enrolment, which averages 3 cuts of every take */
        EXT_RAM_BSS_ATTR static float vsum[KWS_VOICE_DIM];   /* v1.6.5: PSRAM */
        static int64_t vlast;
        int i, dim = spk_dim();
        float a, b;
        if (t0 - vlast > 800000) { memset(vsum, 0, sizeof(vsum)); }
        vlast = t0;
        for (i = 0; i < dim; i++) { vsum[i] += emb[i]; }
        /* v1.6d: the average of the checks decides (not the best single one:
         * a lucky single check let an imitation through) */
        a = kws_voice_similarity(&s_voice, dim, cls, vsum);
        b = kws_voice_similarity(&s_voice, dim, -1, vsum);
        sim = a > b ? a : b;
    }
    ESP_LOGI(TAG, "voice check: %.2f (needs %.2f%s) in %u ms", (double)sim,
             (double)s_det.hit_need,
             s_det.hit_need > kws_voice_need(&s_det, cls) + 0.001f ? ", raised after clear rejections" :
             s_det.hit_need < kws_voice_need(&s_det, cls) - 0.001f ? ", 2nd call" : "", (unsigned)((esp_timer_get_time() - t0) / 1000));
    return sim;
}

/* ------------------------------------------------------------------ NVS ---- */
static void cfg_save(void)
{
    nvs_handle_t h;
    uint8_t c[4] = { s_sens, s_only_me ? 1 : 0, s_rule, s_match };
    if (nvs_open("kws", NVS_READWRITE, &h) == ESP_OK) {
        nvs_set_blob(h, "cfg", c, sizeof(c));
        nvs_commit(h);
        nvs_close(h);
    }
}

/* v1.6.4: the voice is stored compact - only the classes and vector length in
 * use (4 x 128 + 128 floats = ~2.6 KB). The old record always held the full
 * 8 x 256 + 256 floats (~9.3 KB); NVS (24 KB) needs room for the new copy
 * before the old one is dropped, and with the mesh keys / replay table /
 * Wi-Fi settings beside it the save could fail without a word - the voice was
 * then gone after a reset. Errors are logged now. */
#define VOICE_MAGIC2    0x3456574Bu             /* "KWV4": compact record */
typedef struct {
    uint32_t magic, model_id, spk_id;
    int32_t  dim, ncls;
    float    agree;
    uint8_t  have[KWS_MAX_NCLS];
    uint8_t  n_all, pad[3];
} voice_hdr_t;

static size_t voice_blob_size(int ncls, int dim)
{
    return sizeof(voice_hdr_t) + sizeof(float) * (size_t)(ncls + 1) * (size_t)dim;
}

static void voice_save(bool keep)
{
    nvs_handle_t h;
    esp_err_t e1 = ESP_OK, e2;
    if (nvs_open("kws", NVS_READWRITE, &h) != ESP_OK) { ESP_LOGE(TAG, "voice: cannot open NVS"); return; }
    nvs_erase_key(h, "voice");                  /* frees the old copy first (the new one is in RAM) */
    if (keep) {
        int ncls = kws_model.ncls, dim = voice_dim(), k;
        size_t n = voice_blob_size(ncls, dim);
        uint8_t *b = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
        if (!b) { nvs_close(h); ESP_LOGE(TAG, "voice: no memory to save"); return; }
        {
            voice_hdr_t hd;
            float *f = (float *)(b + sizeof(hd));
            memset(&hd, 0, sizeof(hd));
            hd.magic = VOICE_MAGIC2;
            hd.model_id = kws_model.id;
            hd.spk_id = s_spk ? spk_id() : 0;
            hd.dim = dim;
            hd.ncls = ncls;
            hd.agree = s_agree;
            memcpy(hd.have, s_voice.have, sizeof(hd.have));
            hd.n_all = s_voice.n_all;
            memcpy(b, &hd, sizeof(hd));
            for (k = 0; k < ncls; k++) { memcpy(f + (size_t)k * dim, s_voice.tmpl[k], sizeof(float) * (size_t)dim); }
            memcpy(f + (size_t)ncls * dim, s_voice.all, sizeof(float) * (size_t)dim);
        }
        e1 = nvs_set_blob(h, "voice", b, n);
        heap_caps_free(b);
    }
    e2 = nvs_commit(h);
    nvs_close(h);
    if (e1 != ESP_OK || e2 != ESP_OK) {
        ESP_LOGE(TAG, "voice NOT saved: %s / %s", esp_err_to_name(e1), esp_err_to_name(e2));
    } else if (keep) {
        ESP_LOGI(TAG, "voice saved (%u bytes)", (unsigned)voice_blob_size(kws_model.ncls, voice_dim()));
    }
}

static void nvs_load(void)
{
    nvs_handle_t h;
    uint8_t c[4];
    size_t n = sizeof(c);
    if (nvs_open("kws", NVS_READONLY, &h) != ESP_OK) { return; }
    if (nvs_get_blob(h, "cfg", c, &n) == ESP_OK && (n == 2 || n == 4)) {
        s_sens = c[0] <= KWS_SENS_HIGH ? c[0] : KWS_SENS_MED;
        s_only_me = c[1] != 0;
        if (n == 4) {                           /* v1.5.1 */
            s_rule = c[2] ? 1 : 0;
            s_match = c[3] <= 2 ? c[3] : 1;
        }
    }
    n = 0;
    if (nvs_get_blob(h, "voice", NULL, &n) == ESP_OK && n > 0) {
        uint8_t *b = heap_caps_malloc(n, MALLOC_CAP_SPIRAM);
        voice_hdr_t hd;
        if (b && n >= sizeof(hd) && nvs_get_blob(h, "voice", b, &n) == ESP_OK) {
            memcpy(&hd, b, sizeof(hd));
            if (hd.magic == VOICE_MAGIC2 && hd.model_id == kws_model.id && hd.spk_id == (s_spk ? spk_id() : 0) &&
                hd.dim == voice_dim() && hd.ncls == kws_model.ncls && voice_ok() &&
                hd.ncls <= KWS_MAX_NCLS && hd.dim <= KWS_VOICE_DIM && n == voice_blob_size(hd.ncls, hd.dim)) {
                const float *f = (const float *)(b + sizeof(hd));
                int k;
                memset(&s_voice, 0, sizeof(s_voice));
                for (k = 0; k < hd.ncls; k++) { memcpy(s_voice.tmpl[k], f + (size_t)k * hd.dim, sizeof(float) * (size_t)hd.dim); }
                memcpy(s_voice.all, f + (size_t)hd.ncls * hd.dim, sizeof(float) * (size_t)hd.dim);
                memcpy(s_voice.have, hd.have, sizeof(hd.have));
                s_voice.n_all = hd.n_all;
                s_agree = hd.agree;
                s_enrolled = voice_ready(&s_voice);
            } else {
                s_stale = true;                 /* another model / older format: enrol again */
            }
        } else {
            s_stale = true;
        }
        if (b) { heap_caps_free(b); }
    }
    nvs_close(h);
}

/* -------------------------------------------------------------- frames ---- */
static float s_frame_db;

static void on_frame(void *ctx, const float *feat)
{
    int ev;
    (void)ctx;
    if (s_en_state == BK_EN_REC) {              /* enrolment: keep the frames of the take */
        if (s_en_req >= 0) { return; }          /* (not started yet) */
        if (s_en_frames < (int)ENROL_FRAMES) {
            memcpy(s_efeat + (size_t)(ENROL_PAD + s_en_frames) * KWS_BANDS, feat, sizeof(float) * KWS_BANDS);
            s_en_frames++;
        }
        if (s_frame_db > s_en_loud) { s_en_loud = s_frame_db; }
        return;
    }
    if (!s_on) { return; }
    {
        int64_t t0 = esp_timer_get_time();
        s_verified = false;
        ev = kws_det_frame(&s_det, feat, s_frame_db, (uint32_t)(t0 / 1000));
        if (ev & KWS_EV_RAN) {
            int k, step;
            uint16_t ms = (uint16_t)((esp_timer_get_time() - t0) / 1000);
            s_runs++;
            /* v1.5: a check must take clearly less than the time between two
             * checks, or the microphone overruns and the task never sleeps
             * (watchdog). If the chip is too slow, check less often.
             * v1.5.1: on the average (the first run after a pause is a full one,
             * a run with a voice check is slow - those do not count), and in
             * steps of 4 frames so the engine can reuse the last run. */
            if (!s_verified) {
                s_run_avg = s_run_avg <= 0.0f ? (float)ms : s_run_avg * 0.8f + (float)ms * 0.2f;
                s_run_ms = (uint16_t)(s_run_avg + 0.5f);
                step = s_run_avg > 200.0f ? 24 : s_run_avg > 120.0f ? 16 : KWS_STEP;
                if (step != s_det.step_frames) {
                    ESP_LOGW(TAG, "a check takes %u ms: now every %d ms", (unsigned)s_run_ms, step * 20);
                    s_det.step_frames = step;
                }
            }
            s_ran = true;
            lock();
            memcpy(s_smooth, s_det.smooth, sizeof(s_smooth));
            if (!s_burst) {                             /* a new sound: peaks start again */
                memset(s_peak, 0, sizeof(s_peak));
                s_burst_loud = -120.0f;
                s_burst_runs = 0;
                s_burst = true;
            }
            for (k = 0; k < kws_model.ncls; k++) { if (s_det.smooth[k] > s_peak[k]) { s_peak[k] = s_det.smooth[k]; } }
            s_burst_runs++;
            unlock();
        }
        if (s_burst && s_frame_db > s_burst_loud) { s_burst_loud = s_frame_db; }
        if (ev & KWS_EV_SOUND) {
            /* v1.6 diagnostics: a sound has ended - one line with the best score of
             * every keyword during it, its threshold, and how many calls counted */
            char line[200], b[16];
            int k, n = 0;
            float thr = kws_model.thr[s_sens > 2 ? 2 : s_sens];
            s_burst = false;
            for (k = 0; k < kws_model.ncls && n < (int)sizeof(line) - 40; k++) {
                if (!kws_is_keyword(&kws_model, k)) { continue; }
                n += snprintf(line + n, sizeof(line) - (size_t)n, "%s%s %.2f/%.2f", n ? ", " : "",
                              kws_label(&kws_model, k, b, sizeof(b)), (double)s_det.sound_best[k],
                              (double)(thr + kws_model.cls_off[k] + s_det.cls_extra[k]));
            }
            ESP_LOGI(TAG, "sound ended (loudest %.0f dBFS): best/needed %s -> %d call(s) counted, %d of %d now",
                     (double)s_burst_loud, line, s_det.sound_counted, s_det.n_hits, s_det.need_hits);
        }
    }
    if (ev & KWS_EV_WORDS) {
        /* v1.6.1 diagnostics: the words checked one by one in this run */
        char line[160], b[16];
        int i, n = 0;
        for (i = 0; i < s_det.wc_n && n < (int)sizeof(line) - 40; i++) {
            n += snprintf(line + n, sizeof(line) - (size_t)n, "%s%u ms: %s %.2f", i ? ", " : "",
                          (unsigned)(s_det.wc[i].len * 20u),
                          s_det.wc[i].cls >= 0 ? kws_label(&kws_model, s_det.wc[i].cls, b, sizeof(b)) : "no keyword",
                          (double)s_det.wc[i].p);
        }
        ESP_LOGI(TAG, "words alone: %s", line);
    }
    if (ev & KWS_EV_SAME) {
        char b[16];
        ESP_LOGI(TAG, "\"%s\" p=%.2f: still the call counted before - not counted again",
                 kws_label(&kws_model, s_det.hit_cls, b, sizeof(b)), (double)s_det.hit_score);
    }
    if (ev & (KWS_EV_HIT | KWS_EV_REJECT)) {
        char b[16];
        lock();
        s_last_cls = s_det.hit_cls;
        s_last_score = s_det.hit_score;
        s_last_spk = s_det.hit_spk;
        s_last_need = s_det.hit_need;
        s_last_rej = (ev & KWS_EV_REJECT) != 0;
        s_last_us = esp_timer_get_time();
        unlock();
        ESP_LOGI(TAG, "heard \"%s\" p=%.2f voice=%.2f (needs %.2f)%s (%d/%d)%s", kws_label(&kws_model, s_det.hit_cls, b, sizeof(b)),
                 (double)s_det.hit_score, (double)s_det.hit_spk, (double)(s_det.hit_spk > -1.5f ? s_det.hit_need : 0.0f), (ev & KWS_EV_REJECT) ? " - not the enrolled voice" : "",
                 (ev & KWS_EV_REJECT) ? s_det.n_hits : s_det.hit_count, s_det.need_hits,
                 (ev & KWS_EV_REJECT) ? "" : s_det.hit_how == 1 ? " - a quick next call" :
                 s_det.hit_how == 2 ? " - calling on without a pause" :
                 s_det.hit_how == 3 ? " - the next word right after" : "");
    }
    if (ev & KWS_EV_SOS) {
        ESP_LOGW(TAG, "voice SOS: keyword heard %d times", s_det.need_hits);
        s_sos = true;
    }
}

/* v1.5.1: the takes so far -> the stored voice. Every take is compared with
 * the others (leave one out); takes that clearly do not match the rest (a
 * cough, another sound, a bad take) are left out, the rest is the template. */
static void enrol_complete(int n)
{
    float sum[KWS_VOICE_DIM], sim[KWS_ENROL_MAX], mean = 0.0f, agree = 0.0f;
    int dim = voice_dim(), i, j, kept = 0;
    bool keep[KWS_ENROL_MAX];

    memset(sum, 0, sizeof(sum));
    for (i = 0; i < n; i++) { for (j = 0; j < dim; j++) { sum[j] += s_takes[i][j]; } }
    for (i = 0; i < n; i++) {
        float ab = 0.0f, aa = 0.0f, bb = 0.0f, o;
        for (j = 0; j < dim; j++) {
            o = sum[j] - s_takes[i][j];
            ab += o * s_takes[i][j]; aa += o * o; bb += s_takes[i][j] * s_takes[i][j];
        }
        sim[i] = aa > 0.0f && bb > 0.0f ? ab / sqrtf(aa * bb) : 0.0f;
        mean += sim[i] / (float)n;
    }
    for (i = 0; i < n; i++) { keep[i] = sim[i] >= mean - 0.20f && sim[i] >= 0.25f; kept += keep[i]; }
    if (kept < 3 || kept * 10 < n * 6) {        /* most takes disagree: keep them all */
        for (i = 0; i < n; i++) { keep[i] = true; }
        kept = n;
    }
    lock();
    memset(&s_voice, 0, sizeof(s_voice));
    for (i = 0; i < n; i++) {
        if (!keep[i]) { continue; }
        kws_voice_add(&s_voice, dim, s_take_cls[i], s_takes[i]);
        agree += sim[i] / (float)kept;
    }
    s_agree = agree;
    s_enrolled = true;
    s_det.voice = &s_voice;
    s_stale = false;
    memset(&s_pending, 0, sizeof(s_pending));
    s_pending_n = 0;
    apply_opts();
    unlock();
    s_save_voice = true;                        /* NVS is written from the UI task: this task's
                                                 * stack may be in PSRAM (no flash writes there) */
    ESP_LOGI(TAG, "voice enrolled: %d of %d takes kept, they agree %.2f, a call needs %.2f",
             kept, n, (double)agree, (double)s_det.verify_thr);
}

/* one finished enrolment take */
static void enrol_finish(void)
{
    float emb[KWS_VOICE_DIM], best[KWS_MAX_NCLS], any = -1.0f;
    int any_at = 0, at[KWS_MAX_NCLS], k = kws_enrol_class(&kws_model, s_en_take), use = k, n, i, q = 0, j, dim = voice_dim();
    float qmin = 1e30f;
    int64_t t0 = esp_timer_get_time();

    if (s_en_loud < ENROL_MIN_DBFS) { s_en_state = BK_EN_QUIET; return; }
    if (k < 0 || s_pending_n >= KWS_ENROL_MAX) { s_en_state = BK_EN_UNCLEAR; return; }
    /* quiet frames around the take: the quietest 8 frames of it, repeated */
    for (i = 0; i + 8 <= s_en_frames; i++) {
        float e = 0.0f;
        for (j = 0; j < 8 * KWS_BANDS; j++) { e += s_efeat[(size_t)(ENROL_PAD + i) * KWS_BANDS + j]; }
        if (e < qmin) { qmin = e; q = i; }
    }
    for (i = 0; i < ENROL_PAD; i++) {
        const float *src = s_efeat + (size_t)(ENROL_PAD + q + i % 8) * KWS_BANDS;
        memcpy(s_efeat + (size_t)i * KWS_BANDS, src, sizeof(float) * KWS_BANDS);
        memcpy(s_efeat + (size_t)(ENROL_PAD + s_en_frames + i) * KWS_BANDS, src, sizeof(float) * KWS_BANDS);
    }
    n = s_en_frames + 2 * ENROL_PAD;
    /* one pass over the take (80 ms steps, the engine reuses the rows) */
    kws_best_windows(&kws_model, s_work, s_efeat, n, 4, best, at, &any, &any_at);
    s_en_score = best[k];
    if (best[k] < ENROL_MIN_PROB) {
        /* v1.6: the keywords together clearly heard (the model is unsure which
         * one, SOS is often split between "sos" and "help"): it is a call for
         * help in the owner's voice - kept for the asked phrase */
        int c = -1;
        for (i = 0; i < kws_model.ncls; i++) {
            if (kws_is_keyword(&kws_model, i) && (c < 0 || best[i] > best[c])) { c = i; }
        }
        if (any >= ENROL_ANY_PROB) { at[k] = any_at; s_en_score = any; }
        else if (c >= 0 && best[c] >= ENROL_ANY_PROB) { use = c; s_en_score = best[c]; }
        else {
            if (any > s_en_score) { s_en_score = any; }
            ESP_LOGI(TAG, "enrol take %d: not clear (asked p=%.2f, all keywords %.2f), loudest %.0f dBFS",
                     s_en_take, (double)best[k], (double)any, (double)s_en_loud);
            s_en_state = BK_EN_UNCLEAR;
            return;
        }
    }
    if (s_spk) {
        /* the separate model, centred on the phrase; v1.5.1: the average of 3
         * cuts 80 ms apart (one cut alone depends on where exactly it falls) */
        float e3[KWS_VOICE_DIM], len = 0.0f;
        int sh;
        memset(emb, 0, sizeof(emb));
        for (sh = -4; sh <= 4; sh += 4) {
            spk_cut(s_efeat, n, at[use] + kws_model.T / 2 + sh, s_sfeat);
            if (!spk_embed(s_sfeat, spk_frames(), s_swork, e3)) { s_en_state = BK_EN_UNCLEAR; return; }
            for (j = 0; j < dim; j++) { emb[j] += e3[j]; }
        }
        for (j = 0; j < dim; j++) { len += emb[j] * emb[j]; }
        len = sqrtf(len) + 1e-9f;
        for (j = 0; j < dim; j++) { emb[j] /= len; }
    } else {
        float p[KWS_MAX_NCLS];
        kws_engine_run(&kws_model, s_efeat + (size_t)at[use] * KWS_BANDS, s_work, p, emb);
    }
    ESP_LOGI(TAG, "enrol take %d: \"%s\" p=%.2f (asked %.2f) at %d ms, loudest %.0f dBFS, %u ms", s_en_take,
             kws_label(&kws_model, use, (char[16]){ 0 }, 16), (double)best[use], (double)best[k],
             (at[use] - ENROL_PAD) * 20, (double)s_en_loud, (unsigned)((esp_timer_get_time() - t0) / 1000));
    lock();
    memcpy(s_takes[s_pending_n], emb, sizeof(float) * (size_t)dim);
    s_take_cls[s_pending_n] = (int8_t)use;
    kws_voice_add(&s_pending, dim, use, emb);
    s_pending_n++;
    i = s_pending_n;
    unlock();
    if (i >= enrol_takes()) { enrol_complete(i); }
    s_en_state = BK_EN_OK;
}

/* ---------------------------------------------------------------- task ---- */
static void mic(bool on)
{
    if (on == s_mic_on) { return; }
    audio_mic_enable(on);
    s_mic_on = on;
    s_level = -120.0f;
}

/* v1.5: a small capture task moves the sound from the I2S DMA (90 ms deep)
 * into a 1 s buffer in PSRAM, so nothing is lost while a CNN check runs
 * (~200 ms on the ESP32-S3). Before, every check dropped ~110 ms of sound. */
#define CAP_BYTES       (KWS_RATE * 2 * 4)      /* v1.6.4: 4 s of 16 bit samples (v1.6 2 s): the big voice
                                                 * model's checks (up to 3 per call) must not lose sound */
static StreamBufferHandle_t s_sb;
static StaticStreamBuffer_t s_sb_ctl;
static uint32_t s_cap_lost;                     /* samples dropped: the 1 s buffer was full */
static volatile bool s_cap_busy;                /* inside audio_mic_read (kws_hold waits for it) */
static volatile bool s_cap_pause;               /* v1.6: drop the sound (enrolment take being worked out) */

static void cap_task(void *arg)
{
    int16_t buf[CHUNK];
    (void)arg;
    for (;;) {
        size_t n;
        s_cap_busy = true;
        if (!s_mic_on) { s_cap_busy = false; vTaskDelay(pdMS_TO_TICKS(10)); continue; }
        n = audio_mic_read(buf, CHUNK, 100);
        s_cap_busy = false;
        if (n == 0) { vTaskDelay(1); continue; }
        /* v1.6 fix: whole blocks only. Before, a full buffer took only the
         * bytes that still fitted - possibly an odd number - and from then on
         * every 16 bit sample was read one byte off: loud noise (~-5 dBFS),
         * nothing recognised, enrolment "Did not catch" - until a reset. It
         * happened when the buffer overflowed (enrolment work, Speed Test). */
        if (s_cap_pause || xStreamBufferSpacesAvailable(s_sb) < n * 2u ||
            xStreamBufferSend(s_sb, buf, n * 2u, 0) != n * 2u) {
            s_cap_lost += (uint32_t)n;
        }
    }
}

/* CHUNK samples from the capture buffer (false: none within 200 ms) */
static bool mic_get(int16_t *buf)
{
    size_t got = 0;
    while (got < CHUNK * 2u) {
        size_t r = xStreamBufferReceive(s_sb, (uint8_t *)buf + got, CHUNK * 2u - got, pdMS_TO_TICKS(200));
        if (r == 0) { return false; }
        got += r;
    }
    return true;
}

static void kws_task(void *arg)
{
    int16_t buf[CHUNK];
    (void)arg;
    for (;;) {
        bool want = s_hold == 0 && (s_on || s_en_state == BK_EN_REC);
        if (!want) {
            mic(false);
            if (s_en_state == BK_EN_REC && s_hold) { s_en_state = BK_EN_NOAUDIO; }
            vTaskDelay(pdMS_TO_TICKS(50));
            continue;
        }
        if (!s_mic_on) {
            mic(true);
            vTaskDelay(pdMS_TO_TICKS(40));      /* the first blocks after start-up */
            xStreamBufferReset(s_sb);
            kws_fe_reset(&s_fe);
            kws_det_reset(&s_det);
        }
        if (s_en_req >= 0) {                    /* a new enrolment take begins */
            s_en_take = s_en_req;
            s_en_req = -1;
            s_en_frames = 0;
            s_en_loud = -120.0f;
            kws_fe_reset(&s_fe);
        }
        if (!mic_get(buf)) { s_short_reads++; continue; }
        s_frames++;
        {   /* v1.5 diagnostics: every 10 s while listening, one line - shows a stalled
             * microphone or a CNN that cannot keep up (should be ~500 frames, up to ~41 checks) */
            int64_t now = esp_timer_get_time();
            if (now - s_diag_t0 >= 10000000) {
                if (s_on) {
                    ESP_LOGI(TAG, "listening: %u frames (500 = real time), %u checks of %u ms (every %d ms), "
                             "mic %.0f dBFS, %u ms of sound lost, hits in window %d, only-my-voice %s",
                             (unsigned)s_frames, (unsigned)(s_runs - s_diag_runs), (unsigned)s_run_ms,
                             s_det.step_frames * 20, (double)s_level, (unsigned)(s_cap_lost / 16u),
                             s_det.n_hits, s_only_me ? "on" : "off");
                    s_cap_lost = 0;
                }
                s_diag_t0 = now;
                s_diag_runs = s_runs;
                s_frames = s_short_reads = 0;
            }
        }
        s_frame_db = kws_dbfs(buf, CHUNK);
        s_level = s_level * 0.8f + s_frame_db * 0.2f;
        kws_fe_push(&s_fe, buf, CHUNK, on_frame, NULL);
        if (s_ran) {
            /* v1.5: let the lower priority tasks (and the idle task with its
             * watchdog) run after every check; the I2S buffer holds 90 ms */
            s_ran = false;
            vTaskDelay(1);
        }
        if (s_en_state == BK_EN_REC && s_en_req < 0 && s_en_frames >= (int)ENROL_FRAMES) {
            uint32_t lost = s_cap_lost;
            s_cap_pause = true;                 /* v1.6: the work takes ~1.2 s: don't fill the buffer meanwhile */
            s_en_state = BK_EN_BUSY;
            enrol_finish();
            xStreamBufferReset(s_sb);           /* listening goes on from fresh */
            s_cap_lost = lost;
            s_cap_pause = false;
            kws_fe_reset(&s_fe);
            kws_det_reset(&s_det);
        }
    }
}

/* ----------------------------------------------------------------- API ---- */
static float *alloc_f(size_t n)
{
    /* v1.6.5: zeroed. The engines keep a "weights prepared" stamp at the start of
     * their work memory; a reused heap block (Speed Test runs again, new random
     * test weights at the same address) could carry an old valid-looking stamp */
    float *p = heap_caps_calloc(n, sizeof(float), MALLOC_CAP_SPIRAM);
    return p ? p : heap_caps_calloc(n, sizeof(float), MALLOC_CAP_8BIT);
}

void kws_init(bool audio_ok, bool mic_ok)
{
    s_mx = xSemaphoreCreateMutex();
    s_audio_ok = audio_ok;
    s_mic_ok = mic_ok;
    s_model_ok = kws_model_ok(&kws_model);
    if (s_model_ok && spk_available() && spk_dim() > 0 && spk_dim() <= KWS_VOICE_DIM &&
        spk_frames() > 0 && spk_frames() <= KWS_MAX_T) {
        s_swork = alloc_f(spk_work() ? spk_work() : 1);
        s_sfeat = alloc_f((size_t)spk_frames() * KWS_BANDS);
        s_spk = s_swork && s_sfeat;
        if (s_spk) {                            /* v1.6.4b: the engine turns its weights into floats once
                                                 * (kept in s_swork): do it now, not at the first call */
            float *e = alloc_f(KWS_VOICE_DIM);  /* (not on the caller's stack) */
            if (e) {
                memset(s_sfeat, 0, sizeof(float) * (size_t)spk_frames() * KWS_BANDS);
                spk_embed(s_sfeat, spk_frames(), s_swork, e);
                free(e);
            }
        }
    }
    nvs_load();
    kws_fe_init();
    if (!s_model_ok) {
        ESP_LOGW(TAG, "no voice SOS model built in (Ai/kws_model.c): Voice SOS cannot listen");
        return;
    }
    s_work = alloc_f(kws_engine_work(&kws_model));
    s_ring = alloc_f((size_t)KWS_MAX_T * KWS_BANDS);
    s_win = alloc_f((size_t)KWS_MAX_T * KWS_BANDS);
    s_efeat = alloc_f((size_t)(ENROL_FRAMES + 2 * ENROL_PAD) * KWS_BANDS);
    if (!s_work || !s_ring || !s_win || !s_efeat) { ESP_LOGE(TAG, "no memory"); s_model_ok = false; return; }
    kws_det_init(&s_det, &kws_model, s_work, s_ring, s_win);
    /* v1.6.1: own memory for the word checks (~200 KB PSRAM), so the normal
     * checks keep reusing their last run */
    s_work2 = alloc_f(kws_engine_work(&kws_model));
    s_win2 = alloc_f((size_t)KWS_MAX_T * KWS_BANDS);
    if (s_work2 && s_win2) { s_det.work2 = s_work2; s_det.win2 = s_win2; }
    else { ESP_LOGW(TAG, "no memory for word checks: two calls need a short pause between them"); }
    s_det.sens = s_sens;
    s_det.only_me = s_only_me;
    s_det.voice = s_enrolled ? &s_voice : NULL;
    if (s_spk) {
        s_det.verify = verify_cb;
        s_det.verify_thr = spk_threshold();
    }
    apply_opts();
    /* v1.5: the task starts whenever the I2S audio works (v1.4 also needed
     * the start-up microphone check). Its stack is in PSRAM - the task never
     * writes the flash itself (kws_service() does that). */
    if (s_audio_ok) {
        uint8_t *store = heap_caps_malloc(CAP_BYTES + 1, MALLOC_CAP_SPIRAM);
        s_sb = store ? xStreamBufferCreateStatic(CAP_BYTES, CHUNK * 2u, store, &s_sb_ctl) : NULL;
        if (!s_sb || xTaskCreatePinnedToCoreWithCaps(cap_task, "kws_cap", 3072, NULL, 6, NULL, 0,
                                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT) != pdPASS) {
            ESP_LOGE(TAG, "cannot start the capture task");
            s_audio_ok = false;
        }
    }
    if (s_audio_ok) {
        /* stack in PSRAM first: the internal RAM is needed by Wi-Fi and the LoRa task */
        s_task_ok = xTaskCreatePinnedToCoreWithCaps(kws_task, "kws", KWS_STACK, NULL, 3, NULL, 0,
                                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT) == pdPASS;
        s_task_psram = s_task_ok;
        if (!s_task_ok) {
            s_task_ok = xTaskCreatePinnedToCore(kws_task, "kws", KWS_STACK, NULL, 3, NULL, 0) == pdPASS;
        }
        if (!s_task_ok) { ESP_LOGE(TAG, "cannot start the listening task (no memory)"); }
    }
    ESP_LOGI(TAG, "model \"%s\" (%d classes, T %d), voice print: %s, %s, voice %s", kws_model.name, kws_model.ncls,
             kws_model.T, s_spk ? spk_name() : kws_model.spk_dim > 0 ? "keyword model head" : "none",
             s_task_ok ? (s_task_psram ? "ready (stack in PSRAM)" : "ready") : s_audio_ok ? "TASK NOT STARTED" : "no audio",
             s_enrolled ? "enrolled" : s_stale ? "must be enrolled again" : "not enrolled");
    ESP_LOGI(TAG, "%s within %u s, voice match %s: help %.2f, help me %.2f, sos %.2f", s_rule ? "3 calls" : "2 calls",
             (unsigned)(s_det.need_ms / 1000u), s_match == 0 ? "loose" : s_match == 2 ? "strict" : "normal",
             (double)s_match_tab[s_match][0], (double)s_match_tab[s_match][1], (double)s_match_tab[s_match][2]);
}

void kws_options(uint8_t rule, uint8_t match)
{
    bool changed = (rule ? 1 : 0) != s_rule || match != s_match;
    s_rule = rule ? 1 : 0;
    s_match = match <= 2 ? match : 1;
    apply_opts();
    if (changed) { cfg_save(); }
}

void kws_config(bool on, uint8_t sens, bool only_me)
{
    bool changed = sens != s_sens || only_me != s_only_me;
    s_sens = sens <= KWS_SENS_HIGH ? sens : KWS_SENS_MED;
    s_only_me = only_me;
    s_det.sens = s_sens;
    s_det.only_me = s_only_me;
    s_want_on = on;
    s_on = on && s_model_ok && s_task_ok;
    ESP_LOGI(TAG, "Voice SOS %s%s", on ? "on" : "off",
             on && !s_on ? (!s_model_ok ? " - but no model" : " - but the listening task is not running") : "");
    if (changed) { cfg_save(); }
}

void kws_hold(bool hold)
{
    int i;
    if (!s_task_ok) { return; }
    if (hold) {
        s_hold++;
        for (i = 0; i < 30 && (s_mic_on || s_cap_busy); i++) { vTaskDelay(pdMS_TO_TICKS(10)); }   /* let go of the microphone */
    } else if (s_hold > 0) {
        s_hold--;
    }
}

bool kws_status(bk_vsos_t *o)
{
    int k;
    memset(o, 0, sizeof(*o));
    o->audio = s_audio_ok;
    o->mic_signal = s_mic_ok;
    o->task_ok = s_task_ok;
    o->want_on = s_want_on;
    o->model = s_model_ok;
    snprintf(o->model_name, sizeof(o->model_name), "%s", kws_model.name ? kws_model.name : "none");
    o->voice_ok = voice_ok();
    o->enrol_takes = (uint8_t)enrol_takes();
    o->listening = s_on && s_mic_on && s_en_state != BK_EN_REC;
    o->paused = s_on && s_hold > 0;
    o->sens = s_sens;
    o->only_me = s_only_me;
    o->enrolled = s_enrolled;
    o->enrol_stale = s_stale;
    o->ncls = s_model_ok ? kws_model.ncls : 0;
    lock();
    for (k = 0; k < o->ncls && k < BK_VS_MAXCLS; k++) {
        kws_label(&kws_model, k, o->labels[k], sizeof(o->labels[k]));
        o->keyword[k] = kws_is_keyword(&kws_model, k);
        o->probs[k] = s_smooth[k];
        o->peak[k] = s_peak[k];
        o->thr_k[k] = kws_model.thr[s_sens > 2 ? 2 : s_sens] + kws_model.cls_off[k] + s_det.cls_extra[k];   /* v1.6.7 */
    }
    o->last_cls = s_last_cls;
    o->last_score = s_last_score;
    o->last_spk = s_last_spk;
    o->last_need = s_last_need;
    o->last_rejected = s_last_rej;
    o->last_age_ms = s_last_cls >= 0 ? (uint32_t)((esp_timer_get_time() - s_last_us) / 1000) : 0;
    unlock();
    o->level_dbfs = s_mic_on ? s_level : -120.0f;
    o->hits = (uint8_t)s_det.n_hits;
    o->hits_needed = (uint8_t)s_det.need_hits;
    o->hits_ms = s_det.need_ms;
    o->rule = s_rule;
    o->match = s_match;
    o->voice_thr = s_det.verify_thr;
    for (k = 0; k < o->ncls && k < BK_VS_MAXCLS; k++) { o->vthr_k[k] = kws_voice_need(&s_det, k); }
    o->agree = s_enrolled ? s_agree : -2.0f;
    o->run_ms = s_run_ms;
    o->step_ms = (uint16_t)(s_det.step_frames * 20);
    o->runs = s_runs;
    return true;
}

bool kws_poll_sos(void)
{
    if (!s_sos) { return false; }
    s_sos = false;
    return true;
}

void kws_enrol_take(int take)
{
    if (!s_model_ok || !s_task_ok || !voice_ok()) { s_en_state = BK_EN_NOAUDIO; return; }
    if (s_en_state == BK_EN_REC || s_en_state == BK_EN_BUSY) { return; }
    s_en_score = 0.0f;
    s_en_frames = 0;
    s_en_req = take;                            /* the kws task starts it (it owns the front-end) */
    s_en_state = BK_EN_REC;
}

int kws_enrol_state(float *score)
{
    if (score) { *score = s_en_score; }
    return s_en_state;
}

void kws_enrol_restart(void)
{
    lock();
    memset(&s_pending, 0, sizeof(s_pending));
    s_pending_n = 0;
    unlock();
    if (s_en_state != BK_EN_REC && s_en_state != BK_EN_BUSY) { s_en_state = BK_EN_IDLE; }
}

int kws_enrol_done(void) { return s_pending_n; }

void kws_forget(void)
{
    lock();
    memset(&s_voice, 0, sizeof(s_voice));
    s_enrolled = false;
    s_agree = -2.0f;
    s_det.voice = NULL;
    s_stale = false;
    apply_opts();
    unlock();
    voice_save(false);
}

/* v1.5: things the kws task must not do itself (flash writes) - called from hw_task() */
void kws_service(void)
{
    if (s_save_voice) {
        s_save_voice = false;
        voice_save(true);
    }
}

/* ---------------------------------------------------- serial: kwsbench ---- */
/* "kwsbench [T c1 c2 c3]": how long one run of a keyword CNN of that size
 * takes on this chip (random weights), plus the built-in models. */
#include <stdarg.h>
static char  *s_bout;                           /* v1.5: also collect the results for the screen */
static size_t s_bout_n;

static void bout(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    if (s_bout && s_bout_n > 1) {
        size_t l = strlen(s_bout);
        va_start(ap, fmt);
        vsnprintf(s_bout + l, s_bout_n - l, fmt, ap);
        va_end(ap);
    }
}

static float s_bench_check;

static float bench_one(const kws_model_t *m, const char *what)
{
    float *work = alloc_f(kws_engine_work(m)), *in = alloc_f((size_t)m->T * KWS_BANDS), p[KWS_MAX_NCLS];
    float emb[KWS_MAX_SPK], ms;
    int i, n = 5;
    int64_t t0, busy = 0;
    if (!work || !in) { bout("kwsbench: no memory\n"); free(work); free(in); return -1.0f; }
    for (i = 0; i < m->T * KWS_BANDS; i++) { in[i] = (float)((i * 37) % 23) * 0.3f - 8.0f; }
    kws_engine_run(m, in, work, p, m->spk_dim > 0 ? emb : NULL);        /* warm up (caches) */
    vTaskDelay(1);
    for (i = 0; i < n; i++) {
        t0 = esp_timer_get_time();
        kws_engine_run(m, in, work, p, m->spk_dim > 0 ? emb : NULL);
        busy += esp_timer_get_time() - t0;
        vTaskDelay(1);                          /* v1.5: never starve the idle task (watchdog) */
    }
    ms = (float)busy / (float)n / 1000.0f;
    printf("kwsbench %s: T %d, c1 %d, c2 %d, c3 %d, hid %d: %.1f ms per full run\n",
           what, m->T, m->c1, m->c2, m->c3, m->hid, (double)ms);
    /* v1.5.1: what a check really costs while listening: the window moved by
     * 12 frames, the rows of the last run reused */
    busy = 0;
    for (i = 0; i < n; i++) {
        t0 = esp_timer_get_time();
        kws_engine_run_shift(m, in, work, p, m->spk_dim > 0 ? emb : NULL, KWS_STEP);
        busy += esp_timer_get_time() - t0;
        vTaskDelay(1);
    }
    s_bench_check = (float)busy / (float)n / 1000.0f;
    printf("kwsbench %s: %.1f ms per check while listening (every 240 ms: %.0f %% of one core)\n",
           what, (double)s_bench_check, (double)s_bench_check / 2.4);
    free(work);
    free(in);
    return ms;
}

void kws_bench(int T, int c1, int c2, int c3)
{
    EXT_RAM_BSS_ATTR static float mean[KWS_BANDS], sd[KWS_BANDS], bias[KWS_MAX_HID > KWS_MAX_C3 ? KWS_MAX_HID : KWS_MAX_C3];
    kws_model_t m;
    int8_t *w;
    size_t n1, n2, n3, n4, n5, i;
    float ms;

    if (kws_model_ok(&kws_model)) {
        ms = bench_one(&kws_model, "built-in model");
        if (ms >= 0.0f) {
            bout("Keyword model: %.0f ms per check (%.0f %% CPU), full run %.0f ms\n", (double)s_bench_check,
                 (double)s_bench_check / 2.4, (double)ms);
        }
    }
    memset(&m, 0, sizeof(m));
    m.name = "bench"; m.labels = "a|b|c|_d";
    m.T = T > 0 ? T : 81; m.B = 40;
    m.c1 = c1 > 0 ? c1 : 24; m.c2 = c2 > 0 ? c2 : 48; m.c3 = c3 > 0 ? c3 : 64; m.hid = 64; m.ncls = 4;
    for (i = 0; i < KWS_BANDS; i++) { mean[i] = -6.0f; sd[i] = 2.0f; }
    n1 = (size_t)m.c1 * 9; n2 = (size_t)m.c2 * m.c1 * 9; n3 = (size_t)m.c3 * m.c2 * 9;
    n4 = (size_t)m.hid * m.c3; n5 = (size_t)m.ncls * m.hid;
    w = heap_caps_malloc(n1 + n2 + n3 + n4 + n5, MALLOC_CAP_SPIRAM);
    if (!w) { bout("kwsbench: no memory\n"); return; }
    for (i = 0; i < n1 + n2 + n3 + n4 + n5; i++) { w[i] = (int8_t)((i * 73) % 255 - 127); }
    m.in_mean = mean; m.in_std = sd;
    m.w1 = w; m.w2 = w + n1; m.w3 = w + n1 + n2; m.w4 = w + n1 + n2 + n3; m.w5 = w + n1 + n2 + n3 + n4;
    m.s1 = m.s2 = m.s3 = m.s4 = m.s5 = 0.01f;
    m.b1 = m.b2 = m.b3 = m.b4 = m.b5 = bias;
    if (!kws_model_ok(&m)) { bout("kwsbench: sizes outside the limits (T<=101, c1<=32, c2<=64, c3<=96)\n"); }
    else {
        ms = bench_one(&m, "random weights (weights in PSRAM)");
        if (ms >= 0.0f) { bout("Size %d/%d/%d/%d: %.0f ms per check, full %.0f ms\n", m.T, m.c1, m.c2, m.c3,
                               (double)s_bench_check, (double)ms); }
    }
    free(w);
    if (s_spk) {
        /* own buffers: the kws task may use s_swork / s_sfeat at the same time */
        float *sw = alloc_f(spk_work() ? spk_work() : 1), *sf = alloc_f((size_t)spk_frames() * KWS_BANDS);
        float *emb = alloc_f(KWS_VOICE_DIM);
        int64_t t0;
        if (sw && sf && emb) {
            for (i = 0; i < (size_t)spk_frames() * KWS_BANDS; i++) { sf[i] = (float)((i * 37) % 23) * 0.3f - 8.0f; }
            spk_embed(sf, spk_frames(), sw, emb);   /* v1.6.4b: first call prepares the float weights */
            t0 = esp_timer_get_time();
            spk_embed(sf, spk_frames(), sw, emb);
            bout("Voice print: %.0f ms per check\n", (double)(esp_timer_get_time() - t0) / 1000.0);
        }
        free(sw);
        free(sf);
        free(emb);
    }
}

void kws_bench_text(char *out, size_t n)
{
    if (!out || !n) { return; }
    out[0] = 0;
    s_bout = out;
    s_bout_n = n;
    kws_bench(0, 0, 0, 0);
    s_bout = NULL;
}
