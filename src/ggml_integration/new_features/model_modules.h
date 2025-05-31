#ifndef MODEL_MODULES_H
#define MODEL_MODULES_H

#include "ggml.h"
#include <cmath>
#include <cstring>
#include <string>

// Model type definitions
enum ModelType {
    MODEL_TYPE_UNKNOWN,
    MODEL_TYPE_LLAMA,
    MODEL_TYPE_VIT,
    MODEL_TYPE_WHISPER
};

// Model parameters structure
struct ModelParams {
    ModelType type;
    std::string model_path;
    std::string input_path;
    std::string output_path;
    int n_threads;
    int n_gpu_layers;
    bool use_gpu;
    
    // Common parameters
    int seed;
    float temperature;
    int top_k;
    float top_p;
    
    // LLaMA specific
    int n_ctx;
    int n_batch;
    
    // ViT specific
    int image_size;
    
    // Whisper specific
    int n_mels;
    int n_audio_ctx;
};

// Módulos comunes
ggml_tensor * multi_head_attention(ggml_context* ctx, ggml_tensor* Q, ggml_tensor* K, ggml_tensor* V, bool is_causal, ggml_tensor* attention_mask = nullptr, float scale_factor = 0.0f);
ggml_tensor * layer_norm(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias, bool use_rmsnorm, float eps);
ggml_tensor * ggml_swiglu(ggml_context * ctx, ggml_tensor * x);
ggml_tensor * feed_forward(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias, const char * activation);
void ggml_sin_f32(int n, float * dest, const float * src);
void ggml_cos_f32(int n, float * dest, const float * src);
ggml_tensor * ggml_pow(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b);
ggml_tensor * positional_encoding(ggml_context * ctx, ggml_tensor * input, const char * type, int n_dims, int mode, float base, int n_ctx);

// Módulos específicos
ggml_tensor* llama_ffn(ggml_context* ctx, ggml_tensor* input, ggml_tensor* gate_proj, ggml_tensor* up_proj, ggml_tensor* down_proj);
ggml_tensor * cross_attention(ggml_context * ctx, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V);
ggml_tensor * class_token(ggml_context * ctx, ggml_tensor * input);
ggml_tensor * multi_head_latent_attention(ggml_context * ctx, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V, 
                                          ggml_tensor * W_Q, ggml_tensor * W_K, ggml_tensor * W_V, 
                                          int latent_dim, bool is_causal);

#endif // MODEL_MODULES_H
