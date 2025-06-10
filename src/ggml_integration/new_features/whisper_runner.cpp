#include "whisper_runner.h"
#include <iostream>
#include <ggml-cuda.h>
#include <ggml-cpu.h>

#include "model_modules.h"
#include "graph_file.h" 
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <sstream>


bool run_whisper_model(ggml_context* ctx, ggml_backend_t backend, const std::string& model_filename) {
    std::cout << "Initializing Whisper model..." << std::endl;
    
    // Función auxiliar para cargar tensores
    auto load_tensor = [&](const std::string& name) -> ggml_tensor* {
        GGUFTensor tensor_data = read_tensor(model_filename, name);
        if (tensor_data.name.empty()) {
            std::cerr << "Error: Failed to load tensor " << name << std::endl;
            return nullptr;
        }

        // Crear tensor GGML basado en los datos leídos
        ggml_tensor* tensor = nullptr;
        if (tensor_data.n_dims == 1) {
            tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tensor_data.dims[0]);
        } else if (tensor_data.n_dims == 2) {
            tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tensor_data.dims[0], tensor_data.dims[1]);
        } else if (tensor_data.n_dims == 3) {
            tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 
                                      tensor_data.dims[0], tensor_data.dims[1],
                                      tensor_data.dims[2]);
        }

        if (!tensor) {
            std::cerr << "Error: Failed to create tensor for " << name << std::endl;
            return nullptr;
        }

        // Copiar datos al tensor GGML
        if (std::holds_alternative<std::vector<float>>(tensor_data.data)) {
            const auto& data = std::get<std::vector<float>>(tensor_data.data);
            memcpy(tensor->data, data.data(), data.size() * sizeof(float));
        } else {
            std::cerr << "Error: Unexpected tensor data type for " << name << std::endl;
            return nullptr;
        }

        return tensor;
    };

    // Obtener parámetros del modelo desde los metadatos GGUF
    auto get_metadata_int = [&](const std::string& key) -> int {
        GGUFMetadata md = read_metadata(model_filename, key);
        if (md.type == GGUF_TYPE_COUNT) {
            std::cerr << "Error: Missing metadata " << key << std::endl;
            return 0;
        }
        return md.value.i32;
    };

    int n_mels = get_metadata_int("whisper.n_mels");
    int n_audio_ctx = get_metadata_int("whisper.n_audio_ctx");
    int n_audio_state = get_metadata_int("whisper.n_audio_state");
    int n_audio_head = get_metadata_int("whisper.n_audio_head");
    int n_audio_layer = get_metadata_int("whisper.n_audio_layer");
    int n_text_ctx = get_metadata_int("whisper.n_text_ctx");
    int n_text_state = get_metadata_int("whisper.n_text_state");
    int n_text_head = get_metadata_int("whisper.n_text_head");
    int n_text_layer = get_metadata_int("whisper.n_text_layer");
    float eps = 1e-5f;

    // 1. Preparar entrada de audio (log-Mel spectrogram)
    ggml_tensor* input_audio = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_mels, n_audio_ctx, 1);
    
    // 2. Procesamiento inicial del audio
    ggml_tensor* conv1_weight = load_tensor("encoder.conv1.weight");
    ggml_tensor* conv1_bias = load_tensor("encoder.conv1.bias");
    ggml_tensor* conv2_weight = load_tensor("encoder.conv2.weight");
    ggml_tensor* conv2_bias = load_tensor("encoder.conv2.bias");
    
    if (!conv1_weight || !conv1_bias || !conv2_weight || !conv2_bias) return false;

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
        ggml_tensor* norm1 = load_tensor(layer_prefix + "attn_ln.weight");
        if (!norm1) return false;
        ggml_tensor* attn_norm = layer_norm(ctx, audio_emb, norm1, nullptr, false, eps);
        
        // Proyecciones Q, K, V
        ggml_tensor* q_proj = load_tensor(layer_prefix + "attn.query.weight");
        ggml_tensor* k_proj = load_tensor(layer_prefix + "attn.key.weight");
        ggml_tensor* v_proj = load_tensor(layer_prefix + "attn.value.weight");
        if (!q_proj || !k_proj || !v_proj) return false;
        
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
        ggml_tensor* out_proj = load_tensor(layer_prefix + "attn.out.weight");
        if (!out_proj) return false;
        self_attn = ggml_mul_mat(ctx, out_proj, self_attn);
        audio_emb = ggml_add(ctx, audio_emb, self_attn);
        
        // MLP
        ggml_tensor* norm2 = load_tensor(layer_prefix + "mlp_ln.weight");
        if (!norm2) return false;
        ggml_tensor* mlp_norm = layer_norm(ctx, audio_emb, norm2, nullptr, false, eps);
        
        ggml_tensor* fc1 = load_tensor(layer_prefix + "mlp.0.weight");
        ggml_tensor* fc1_bias = load_tensor(layer_prefix + "mlp.0.bias");
        ggml_tensor* fc2 = load_tensor(layer_prefix + "mlp.2.weight");
        ggml_tensor* fc2_bias = load_tensor(layer_prefix + "mlp.2.bias");
        if (!fc1 || !fc1_bias || !fc2 || !fc2_bias) return false;
        
        ggml_tensor* mlp = feed_forward(ctx, mlp_norm, fc1, fc1_bias, "gelu");
        mlp = feed_forward(ctx, mlp, fc2, fc2_bias, "linear");
        audio_emb = ggml_add(ctx, audio_emb, mlp);
    }
    
    // 5. Inicializar tokens del decoder
    ggml_tensor* tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_text_ctx);
    ggml_tensor* token_emb = load_tensor("decoder.token_embedding.weight");
    if (!token_emb) return false;
    token_emb = ggml_get_rows(ctx, token_emb, tokens);
    
    // 6. Posición embeddings para el decoder
    ggml_tensor* pos_emb = load_tensor("decoder.positional_embedding");
    if (!pos_emb) return false;
    token_emb = ggml_add(ctx, token_emb, pos_emb);
    
    // 7. Capas del decoder
    for (int i = 0; i < n_text_layer; ++i) {
        std::string layer_prefix = "decoder.blocks." + std::to_string(i) + ".";
        
        // Self-attention
        ggml_tensor* norm1 = load_tensor(layer_prefix + "attn_ln.weight");
        if (!norm1) return false;
        ggml_tensor* attn_norm = layer_norm(ctx, token_emb, norm1, nullptr, false, eps);
        
        ggml_tensor* q_proj = load_tensor(layer_prefix + "attn.query.weight");
        ggml_tensor* k_proj = load_tensor(layer_prefix + "attn.key.weight");
        ggml_tensor* v_proj = load_tensor(layer_prefix + "attn.value.weight");
        if (!q_proj || !k_proj || !v_proj) return false;
        
        ggml_tensor* q = ggml_mul_mat(ctx, q_proj, attn_norm);
        ggml_tensor* k = ggml_mul_mat(ctx, k_proj, attn_norm);
        ggml_tensor* v = ggml_mul_mat(ctx, v_proj, attn_norm);
        
        int head_dim = n_text_state / n_text_head;
        q = ggml_reshape_3d(ctx, q, head_dim, n_text_head, n_text_ctx);
        k = ggml_reshape_3d(ctx, k, head_dim, n_text_head, n_text_ctx);
        v = ggml_reshape_3d(ctx, v, head_dim, n_text_head, n_text_ctx);
        
        ggml_tensor* self_attn = multi_head_attention(ctx, q, k, v, true); // causal=true para decoder
        self_attn = ggml_reshape_2d(ctx, self_attn, n_text_state, n_text_ctx);
        
        ggml_tensor* out_proj = load_tensor(layer_prefix + "attn.out.weight");
        if (!out_proj) return false;
        self_attn = ggml_mul_mat(ctx, out_proj, self_attn);
        token_emb = ggml_add(ctx, token_emb, self_attn);
        
        // Cross-attention
        ggml_tensor* norm2 = load_tensor(layer_prefix + "cross_attn_ln.weight");
        if (!norm2) return false;
        ggml_tensor* cross_norm = layer_norm(ctx, token_emb, norm2, nullptr, false, eps);
        
        ggml_tensor* cross_q = load_tensor(layer_prefix + "cross_attn.query.weight");
        ggml_tensor* cross_k = load_tensor(layer_prefix + "cross_attn.key.weight");
        ggml_tensor* cross_v = load_tensor(layer_prefix + "cross_attn.value.weight");
        if (!cross_q || !cross_k || !cross_v) return false;
        
        q = ggml_mul_mat(ctx, cross_q, cross_norm);
        k = ggml_mul_mat(ctx, cross_k, audio_emb); // Keys/Values del encoder
        v = ggml_mul_mat(ctx, cross_v, audio_emb);
        
        q = ggml_reshape_3d(ctx, q, head_dim, n_text_head, n_text_ctx);
        k = ggml_reshape_3d(ctx, k, head_dim, n_text_head, n_audio_ctx/2);
        v = ggml_reshape_3d(ctx, v, head_dim, n_text_head, n_audio_ctx/2);
        
        ggml_tensor* cross_attn = cross_attention(ctx, q, k, v);
        cross_attn = ggml_reshape_2d(ctx, cross_attn, n_text_state, n_text_ctx);
        
        ggml_tensor* cross_out = load_tensor(layer_prefix + "cross_attn.out.weight");
        if (!cross_out) return false;
        cross_attn = ggml_mul_mat(ctx, cross_out, cross_attn);
        token_emb = ggml_add(ctx, token_emb, cross_attn);
        
        // MLP
        ggml_tensor* norm3 = load_tensor(layer_prefix + "mlp_ln.weight");
        if (!norm3) return false;
        ggml_tensor* mlp_norm = layer_norm(ctx, token_emb, norm3, nullptr, false, eps);
        
        ggml_tensor* fc1 = load_tensor(layer_prefix + "mlp.0.weight");
        ggml_tensor* fc1_bias = load_tensor(layer_prefix + "mlp.0.bias");
        ggml_tensor* fc2 = load_tensor(layer_prefix + "mlp.2.weight");
        ggml_tensor* fc2_bias = load_tensor(layer_prefix + "mlp.2.bias");
        if (!fc1 || !fc1_bias || !fc2 || !fc2_bias) return false;
        
        ggml_tensor* mlp = feed_forward(ctx, mlp_norm, fc1, fc1_bias, "gelu");
        mlp = feed_forward(ctx, mlp, fc2, fc2_bias, "linear");
        token_emb = ggml_add(ctx, token_emb, mlp);
    }
    
    // 8. Normalización final y proyección
    ggml_tensor* norm_out = load_tensor("decoder.ln.weight");
    if (!norm_out) return false;
    token_emb = layer_norm(ctx, token_emb, norm_out, nullptr, false, eps);
    
    ggml_tensor* lm_head = load_tensor("decoder.token_embedding.weight"); // weight tying
    if (!lm_head) return false;
    ggml_tensor* output = ggml_mul_mat(ctx, lm_head, token_emb);
    
    // 9. Construir y ejecutar el grafo computacional
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    ggml_backend_graph_compute(backend, gf);
    
    std::cout << "Whisper model execution completed" << std::endl;
    return true;
}