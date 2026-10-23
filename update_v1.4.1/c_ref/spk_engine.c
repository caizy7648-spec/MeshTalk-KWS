/* spk_engine.c - STUB: no separate speaker model yet (see spk_engine.h).
 * Replaced by the spk_engine.c / spk_model.c of training phase 2. */
#include "spk_engine.h"

bool        spk_available(void)   { return false; }
const char *spk_name(void)        { return "none"; }
uint32_t    spk_id(void)          { return 0; }
int         spk_dim(void)         { return 0; }
int         spk_frames(void)      { return 0; }
size_t      spk_work(void)        { return 0; }
float       spk_threshold(void)   { return 1.0f; }
int         spk_enrol_takes(void) { return 5; }
bool        spk_embed(const float *feat, int n, float *work, float *emb)
{
    (void)feat; (void)n; (void)work; (void)emb;
    return false;
}
