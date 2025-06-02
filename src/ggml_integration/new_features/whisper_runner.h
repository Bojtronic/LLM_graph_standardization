#ifndef WHISPER_RUNNER_H
#define WHISPER_RUNNER_H

#include <ggml.h>
#include "model_modules.h"
#include <ggml-backend.h>
#include "gguf_loader.h"
#include "model_modules.h"


bool run_whisper_model(ggml_context* ctx, ggml_backend_t backend, GraphData graph_data);

#endif // WHISPER_RUNNER_H