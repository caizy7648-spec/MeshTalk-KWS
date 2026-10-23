/* spk_test.c - C test of the speaker engine against Python (spk/export.py):
 * spk_model_test.c holds an input and the vector reference() gives for it.
 * Exit code 0 = all checks passed. */
#include "spk_engine.h"
#include "spk_model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

extern const float spk_test_in[];
extern const float spk_test_emb[];

static int fails, checks;

static void check(int ok, const char *what)
{
    checks++;
    if (!ok) { fails++; }
    printf("   %-52s %s\n", what, ok ? "ok" : "FAILED");
}

int main(void)
{
    int n = spk_frames(), d = spk_dim(), i;
    float *work, *emb, diff = 0.0f, len = 0.0f;

    printf("speaker model \"%s\", id %08X\n", spk_name(), (unsigned)spk_id());
    check(spk_available(), "spk_available()");
    check(d > 0 && d <= 256, "spk_dim() in 1..256");
    check(n > 0 && n <= 101, "spk_frames() in 1..101");
    check(spk_enrol_takes() >= 3 && spk_enrol_takes() <= 10, "spk_enrol_takes() in 3..10");
    check(spk_threshold() > -1.0f && spk_threshold() < 1.0f, "spk_threshold() in -1..1");
    check(spk_work() * sizeof(float) <= 512u * 1024u, "spk_work() <= 512 KB");
    work = (float *)malloc(sizeof(float) * spk_work());
    emb = (float *)malloc(sizeof(float) * (size_t)d);
    check(work && emb && spk_embed(spk_test_in, n, work, emb), "spk_embed() runs");
    check(!spk_embed(spk_test_in, n - 1, work, emb) || n == 1, "spk_embed() refuses a wrong length");
    spk_embed(spk_test_in, n, work, emb);
    for (i = 0; i < d; i++) {
        float x = fabsf(emb[i] - spk_test_emb[i]);
        if (x > diff) { diff = x; }
        len += emb[i] * emb[i];
    }
    printf("   largest difference to Python %.2e, length %.6f\n", diff, sqrtf(len));
    check(diff < 1e-4f, "C == Python (max difference < 1e-4)");
    check(fabsf(sqrtf(len) - 1.0f) < 1e-4f, "vector has length 1");
    printf("   work memory %u floats (%u KB)\n", (unsigned)spk_work(), (unsigned)(spk_work() * 4 / 1024));
    printf("%d checks, %d failed\n", checks, fails);
    free(work);
    free(emb);
    return fails ? 1 : 0;
}
