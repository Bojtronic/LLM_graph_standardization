#ifndef LLAMA_RUNNER_H
#define LLAMA_RUNNER_H

#include <ggml.h>
#include "model_modules.h"
#include <ggml-backend.h>
#include "gguf_loader.h"
#include "model_modules.h"


std::vector<int> tokenize_input(const std::string& input, const std::string& model_filename) ;
std::vector<int> tokenize_basic(const std::string& input);
std::string decode_output(const std::vector<int>& tokens, const std::string& model_filename);
std::string decode_basic(const std::vector<int>& tokens);
bool run_interactive_chat(ggml_backend_t backend, const std::string& model_filename);
int sample_next_token(const float* logits, int n_vocab, 
                     float temperature, float top_p, int top_k);
                     
ggml_tensor* run_llama_model(ggml_context* ctx, 
                            ggml_backend_t backend,
                            const std::string& model_filename,
                            int n_embd,
                            int n_head,
                            int n_layers,
                            float norm_eps,
                            int n_ctx,
                            int n_vocab,
                            const std::vector<int>& input_tokens);


#endif // LLAMA_RUNNER_H