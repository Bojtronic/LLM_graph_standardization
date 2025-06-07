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


bool run_whisper_model(ggml_context* ctx, ggml_backend_t backend, GraphData& graph_data) {
    std::cout << "Initializing Whisper model..." << std::endl;
    
    // Obtener parámetros del modelo desde los metadatos GGUF
    int n_mels = graph_data.find_metadata("whisper.n_mels")->value.i32;
    int n_audio_ctx = graph_data.find_metadata("whisper.n_audio_ctx")->value.i32;
    int n_audio_state = graph_data.find_metadata("whisper.n_audio_state")->value.i32;
    int n_audio_head = graph_data.find_metadata("whisper.n_audio_head")->value.i32;
    int n_audio_layer = graph_data.find_metadata("whisper.n_audio_layer")->value.i32;
    int n_text_ctx = graph_data.find_metadata("whisper.n_text_ctx")->value.i32;
    int n_text_state = graph_data.find_metadata("whisper.n_text_state")->value.i32;
    int n_text_head = graph_data.find_metadata("whisper.n_text_head")->value.i32;
    int n_text_layer = graph_data.find_metadata("whisper.n_text_layer")->value.i32;
    float eps = 1e-5f;

    // 1. Preparar entrada de audio (log-Mel spectrogram)
    ggml_tensor* input_audio = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_mels, n_audio_ctx, 1);
    
    // 2. Procesamiento inicial del audio
    ggml_tensor* conv1_weight = get_layer_tensor(ctx, graph_data, "encoder.conv1.weight");
    ggml_tensor* conv1_bias = get_layer_tensor(ctx, graph_data, "encoder.conv1.bias");
    ggml_tensor* conv2_weight = get_layer_tensor(ctx, graph_data, "encoder.conv2.weight");
    ggml_tensor* conv2_bias = get_layer_tensor(ctx, graph_data, "encoder.conv2.bias");
    
    // Primera capa convolucional
    ggml_tensor* audio_emb = ggml_conv_1d(ctx, input_audio, conv1_weight, 1, 1, 0);
    audio_emb = ggml_add(ctx, audio_emb, conv1_bias);
    audio_emb = ggml_gelu(ctx, audio_emb);
    
    // Segunda capa convolucional
    audio_emb = ggml_conv_1d(ctx, audio_emb, conv2_weight, 2, 2, 0); // stride=2 para downsampling
    audio_emb = ggml_add(ctx, audio_emb, conv2_bias);
    audio_emb = ggml_gelu(ctx, audio_emb);
    
    // 3. Codificación posicional
    audio_emb = positional_encoding(ctx, audio_emb, "sinusoidal", n_audio_state, 0, 10000.0f);
    
    // 4. Capas del encoder
    for (int i = 0; i < n_audio_layer; ++i) {
        std::string layer_prefix = "encoder.blocks." + std::to_string(i) + ".";
        
        // Atención
        ggml_tensor* norm1 = get_layer_tensor(ctx, graph_data, layer_prefix + "attn_ln.weight");
        ggml_tensor* attn_norm = layer_norm(ctx, audio_emb, norm1, nullptr, false, eps);
        
        // Proyecciones Q, K, V
        ggml_tensor* q_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.query.weight");
        ggml_tensor* k_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.key.weight");
        ggml_tensor* v_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.value.weight");
        
        ggml_tensor* q = ggml_mul_mat(ctx, q_proj, attn_norm);
        ggml_tensor* k = ggml_mul_mat(ctx, k_proj, attn_norm);
        ggml_tensor* v = ggml_mul_mat(ctx, v_proj, attn_norm);
        
        // Atención multi-cabeza
        int head_dim = n_audio_state / n_audio_head;
        q = ggml_reshape_3d(ctx, q, head_dim, n_audio_head, n_audio_ctx/2); // /2 por el stride=2
        k = ggml_reshape_3d(ctx, k, head_dim, n_audio_head, n_audio_ctx/2);
        v = ggml_reshape_3d(ctx, v, head_dim, n_audio_head, n_audio_ctx/2);
        
        ggml_tensor* self_attn = multi_head_attention(ctx, q, k, v, false);
        self_attn = ggml_reshape_2d(ctx, self_attn, n_audio_state, n_audio_ctx/2);
        
        // Proyección de salida
        ggml_tensor* out_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.out.weight");
        self_attn = ggml_mul_mat(ctx, out_proj, self_attn);
        audio_emb = ggml_add(ctx, audio_emb, self_attn);
        
        // MLP
        ggml_tensor* norm2 = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp_ln.weight");
        ggml_tensor* mlp_norm = layer_norm(ctx, audio_emb, norm2, nullptr, false, eps);
        
        ggml_tensor* fc1 = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.0.weight");
        ggml_tensor* fc1_bias = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.0.bias");
        ggml_tensor* fc2 = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.2.weight");
        ggml_tensor* fc2_bias = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.2.bias");
        
        ggml_tensor* mlp = feed_forward(ctx, mlp_norm, fc1, fc1_bias, "gelu");
        mlp = feed_forward(ctx, mlp, fc2, fc2_bias, "linear");
        audio_emb = ggml_add(ctx, audio_emb, mlp);
    }
    
    // 5. Inicializar tokens del decoder
    ggml_tensor* tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_ctx);
    ggml_tensor* token_emb = get_layer_tensor(ctx, graph_data, "decoder.token_embedding.weight");
    token_emb = ggml_get_rows(ctx, token_emb, tokens);
    
    // 6. Posición embeddings para el decoder
    ggml_tensor* pos_emb = get_layer_tensor(ctx, graph_data, "decoder.positional_embedding");
    token_emb = ggml_add(ctx, token_emb, pos_emb);
    
    // 7. Capas del decoder
    for (int i = 0; i < n_text_layer; ++i) {
        std::string layer_prefix = "decoder.blocks." + std::to_string(i) + ".";
        
        // Self-attention
        ggml_tensor* norm1 = get_layer_tensor(ctx, graph_data, layer_prefix + "attn_ln.weight");
        ggml_tensor* attn_norm = layer_norm(ctx, token_emb, norm1, nullptr, false, eps);
        
        ggml_tensor* q_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.query.weight");
        ggml_tensor* k_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.key.weight");
        ggml_tensor* v_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.value.weight");
        
        ggml_tensor* q = ggml_mul_mat(ctx, q_proj, attn_norm);
        ggml_tensor* k = ggml_mul_mat(ctx, k_proj, attn_norm);
        ggml_tensor* v = ggml_mul_mat(ctx, v_proj, attn_norm);
        
        int head_dim = n_text_state / n_text_head;
        q = ggml_reshape_3d(ctx, q, head_dim, n_text_head, n_text_ctx);
        k = ggml_reshape_3d(ctx, k, head_dim, n_text_head, n_text_ctx);
        v = ggml_reshape_3d(ctx, v, head_dim, n_text_head, n_text_ctx);
        
        ggml_tensor* self_attn = multi_head_attention(ctx, q, k, v, true); // causal=true para decoder
        self_attn = ggml_reshape_2d(ctx, self_attn, n_text_state, n_text_ctx);
        
        ggml_tensor* out_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.out.weight");
        self_attn = ggml_mul_mat(ctx, out_proj, self_attn);
        token_emb = ggml_add(ctx, token_emb, self_attn);
        
        // Cross-attention
        ggml_tensor* norm2 = get_layer_tensor(ctx, graph_data, layer_prefix + "cross_attn_ln.weight");
        ggml_tensor* cross_norm = layer_norm(ctx, token_emb, norm2, nullptr, false, eps);
        
        ggml_tensor* cross_q = get_layer_tensor(ctx, graph_data, layer_prefix + "cross_attn.query.weight");
        ggml_tensor* cross_k = get_layer_tensor(ctx, graph_data, layer_prefix + "cross_attn.key.weight");
        ggml_tensor* cross_v = get_layer_tensor(ctx, graph_data, layer_prefix + "cross_attn.value.weight");
        
        q = ggml_mul_mat(ctx, cross_q, cross_norm);
        k = ggml_mul_mat(ctx, cross_k, audio_emb); // Keys/Values del encoder
        v = ggml_mul_mat(ctx, cross_v, audio_emb);
        
        q = ggml_reshape_3d(ctx, q, head_dim, n_text_head, n_text_ctx);
        k = ggml_reshape_3d(ctx, k, head_dim, n_text_head, n_audio_ctx/2);
        v = ggml_reshape_3d(ctx, v, head_dim, n_text_head, n_audio_ctx/2);
        
        ggml_tensor* cross_attn = cross_attention(ctx, q, k, v);
        cross_attn = ggml_reshape_2d(ctx, cross_attn, n_text_state, n_text_ctx);
        
        ggml_tensor* cross_out = get_layer_tensor(ctx, graph_data, layer_prefix + "cross_attn.out.weight");
        cross_attn = ggml_mul_mat(ctx, cross_out, cross_attn);
        token_emb = ggml_add(ctx, token_emb, cross_attn);
        
        // MLP
        ggml_tensor* norm3 = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp_ln.weight");
        ggml_tensor* mlp_norm = layer_norm(ctx, token_emb, norm3, nullptr, false, eps);
        
        ggml_tensor* fc1 = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.0.weight");
        ggml_tensor* fc1_bias = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.0.bias");
        ggml_tensor* fc2 = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.2.weight");
        ggml_tensor* fc2_bias = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.2.bias");
        
        ggml_tensor* mlp = feed_forward(ctx, mlp_norm, fc1, fc1_bias, "gelu");
        mlp = feed_forward(ctx, mlp, fc2, fc2_bias, "linear");
        token_emb = ggml_add(ctx, token_emb, mlp);
    }
    
    // 8. Normalización final y proyección
    ggml_tensor* norm_out = get_layer_tensor(ctx, graph_data, "decoder.ln.weight");
    token_emb = layer_norm(ctx, token_emb, norm_out, nullptr, false, eps);
    
    ggml_tensor* lm_head = get_layer_tensor(ctx, graph_data, "decoder.token_embedding.weight"); // weight tying
    ggml_tensor* output = ggml_mul_mat(ctx, lm_head, token_emb);
    
    // 9. Construir y ejecutar el grafo computacional
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    ggml_backend_graph_compute(backend, gf);
    
    std::cout << "Whisper model execution completed" << std::endl;
    return true;
}
