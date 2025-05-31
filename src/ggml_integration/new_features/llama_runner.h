#ifndef LLAMA_RUNNER_H
#define LLAMA_RUNNER_H

#include <ggml.h>
#include "model_modules.h"
#include <ggml-backend.h>
#include "gguf_loader.h"
#include "model_modules.h"


ggml_tensor* run_llama_model(ggml_context* ctx, ggml_backend_t backend, const ModelParams& params);


#endif // LLAMA_RUNNER_H