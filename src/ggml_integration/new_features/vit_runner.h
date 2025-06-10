#ifndef VIT_RUNNER_H
#define VIT_RUNNER_H

#include <ggml.h>
#include "model_modules.h"
#include <ggml-backend.h>
#include "gguf_loader.h"
#include "model_modules.h"


//bool run_vit_model(ggml_context* ctx, ggml_backend_t backend, GraphData graph_data);
bool run_vit_model(ggml_context* ctx, ggml_backend_t backend, const std::string& model_filename);


#endif // VIT_RUNNER_H
