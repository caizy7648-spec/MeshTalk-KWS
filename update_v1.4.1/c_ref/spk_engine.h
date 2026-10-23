/* spk_engine.h - v1.4.1: interface for a SEPARATE speaker (voice print) model.
 *
 * "Only My Voice" needs to tell the owner's voice from others. The keyword CNN
 * (kws_engine) cannot do it well (it is trained to ignore who speaks), so a
 * second, larger model may be added later. It only runs ONCE per keyword hit
 * (not every 240 ms), so it may be slower (up to ~0.5 s) without costing power.
 *
 * Phase 2 of the training package (MeshTalk-KWS, CONTRACT_SPK.md) delivers
 * Ai/spk_engine.c + Ai/spk_model.c implementing these functions. Until then
 * the stub (spk_engine.c) reports "no speaker model" and Only My Voice is
 * hidden on the device.
 *
 * Input: log-mel frames from the same front-end (Ai/kws_frontend.c, 40 bands,
 * 20 ms hop), row = frame. Output: a voice vector of spk_dim() floats with
 * length 1; two voices are compared by cosine similarity. */
#ifndef SPK_ENGINE_H
#define SPK_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

bool        spk_available(void);    /* a trained speaker model is built in */
const char *spk_name(void);
uint32_t    spk_id(void);           /* changes with every export: stored voices of another id are void */
int         spk_dim(void);          /* length of the voice vector, <= 256 (KWS_VOICE_DIM) */
int         spk_frames(void);       /* frames it wants (centred on the keyword), <= 101 */
size_t      spk_work(void);         /* floats of work memory for spk_embed() */
float       spk_threshold(void);    /* default: accept when cosine >= this */
int         spk_enrol_takes(void);  /* enrolment takes it wants, 3..10 */
/* feat: n x 40 log-mel (n == spk_frames()), emb: spk_dim() floats, length 1 */
bool        spk_embed(const float *feat, int n, float *work, float *emb);

#endif
