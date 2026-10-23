/* kws_model.c - the voice SOS model built into the firmware.
 *
 * NO MODEL YET: this placeholder makes Voice SOS report "no model installed"
 * (it never triggers). Replace this file with the kws_model.c written by
 * export.py of the training package MeshTalk-KWS and rebuild. */
#include "kws_engine.h"

const kws_model_t kws_model = {
    .name = "none",
    .ncls = 0,
};
