#include "whisper_runner.h"
#include <iostream>
#include <ggml-cuda.h>
#include <ggml-cpu.h>

#include "model_modules.h"
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <sstream>


bool run_whisper_model(ggml_context* ctx, ggml_backend_t backend, const ModelParams& params) {
    std::cout << "Initializing Whisper model..." << std::endl;
    
    GraphData graph_data = gguf_graph_data(gguf_init_from_file(params.model_path.c_str(), {}), params.model_path.c_str());
    
    ggml_tensor* input_audio = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, params.n_mels, params.n_audio_ctx, 1);
    
    ggml_tensor* audio_emb = ggml_conv_1d(ctx, /* conv1 */, input_audio, 1, 1, 0);
    audio_emb = ggml_gelu(ctx, audio_emb);
    audio_emb = ggml_conv_1d(ctx, /* conv2 */, audio_emb, 1, 1, 0);
    audio_emb = ggml_gelu(ctx, audio_emb);
    
    audio_emb = positional_encoding(ctx, audio_emb, "sinusoidal", audio_emb->ne[0], 0, 10000.0f);
    
    for (int i = 0; i < /* num_encoder_layers */; ++i) {
        ggml_tensor* norm1 = layer_norm(ctx, audio_emb, false, 1e-5f);
        ggml_tensor* q = ggml_mul_mat(ctx, /* Wq */, norm1);
        ggml_tensor* k = ggml_mul_mat(ctx, /* Wk */, norm1);
        ggml_tensor* v = ggml_mul_mat(ctx, /* Wv */, norm1);
        
        ggml_tensor* self_attn = multi_head_attention(ctx, q, k, v, false);
        self_attn = ggml_mul_mat(ctx, /* out_proj */, self_attn);
        audio_emb = ggml_add(ctx, audio_emb, self_attn);
        
        ggml_tensor* norm2 = layer_norm(ctx, audio_emb, false, 1e-5f);
        ggml_tensor* mlp = feed_forward(ctx, norm2, /* fc1 */, /* b1 */, "gelu");
        mlp = feed_forward(ctx, mlp, /* fc2 */, /* b2 */, "linear");
        audio_emb = ggml_add(ctx, audio_emb, mlp);
    }
    
    ggml_tensor* tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, params.n_audio_ctx);
    ggml_tensor* token_emb = ggml_get_rows(ctx, /* token_emb */, tokens);
    
    for (int i = 0; i < /* num_decoder_layers */; ++i) {
        ggml_tensor* norm1 = layer_norm(ctx, token_emb, false, 1e-5f);
        ggml_tensor* q = ggml_mul_mat(ctx, /* Wq */, norm1);
        ggml_tensor* k = ggml_mul_mat(ctx, /* Wk */, norm1);
        ggml_tensor* v = ggml_mul_mat(ctx, /* Wv */, norm1);
        
        ggml_tensor* self_attn = multi_head_attention(ctx, q, k, v, true);
        token_emb = ggml_add(ctx, token_emb, self_attn);
        
        ggml_tensor* norm2 = layer_norm(ctx, token_emb, false, 1e-5f);
        ggml_tensor* cross_attn = cross_attention(ctx, norm2, audio_emb, audio_emb);
        token_emb = ggml_add(ctx, token_emb, cross_attn);
        
        ggml_tensor* norm3 = layer_norm(ctx, token_emb, false, 1e-5f);
        ggml_tensor* mlp = feed_forward(ctx, norm3, /* fc1 */, /* b1 */, "gelu");
        mlp = feed_forward(ctx, mlp, /* fc2 */, /* b2 */, "linear");
        token_emb = ggml_add(ctx, token_emb, mlp);
    }
    
    ggml_tensor* output = ggml_mul_mat(ctx, /* lm_head */, token_emb);
    
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    ggml_backend_graph_compute(backend, gf);
    
    std::cout << "Whisper model execution completed" << std::endl;
    return true;
}

