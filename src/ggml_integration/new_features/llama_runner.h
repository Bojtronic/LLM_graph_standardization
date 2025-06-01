#ifndef LLAMA_RUNNER_H
#define LLAMA_RUNNER_H

#include <ggml.h>
#include "model_modules.h"
#include <ggml-backend.h>
#include "gguf_loader.h"
#include "model_modules.h"


ggml_tensor* run_llama_model(ggml_context* ctx, 
                            ggml_backend_t backend,
                            const ModelParams& params,
                            const GraphData& graph_data,
                            int n_embd,
                            int n_head,
                            int n_layers,
                            float norm_eps,
                            int n_ctx,
                            int n_vocab,
                            const std::vector<int>& input_tokens);


#endif // LLAMA_RUNNER_H