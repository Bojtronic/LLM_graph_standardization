#ifndef MODEL_RUNNER_H
#define MODEL_RUNNER_H

#include <ggml.h>
#include "model_modules.h"
#include <ggml-backend.h>
#include "gguf_loader.h"
#include "model_modules.h"


// Declaraciones de funciones para ejecutar los modelos
bool run_llama_model(ggml_context* ctx, ggml_backend_t backend, const ModelParams& params);
bool run_vit_model(ggml_context* ctx, ggml_backend_t backend, const ModelParams& params);
bool run_whisper_model(ggml_context* ctx, ggml_backend_t backend, const ModelParams& params);

#endif // MODEL_RUNNER_H