#include "vit_runner.h"
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


void copy_tensor_data(ggml_tensor* dst, ggml_tensor* src, size_t offset) {
    if (dst->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32) {
        std::cerr << "Solo se soporta copia de tensores F32" << std::endl;
        return;
    }
    
    size_t src_size = ggml_element_size(src) * ggml_nelements(src);
    size_t dst_offset = offset * ggml_element_size(dst);
    
    if (dst_offset + src_size > ggml_nbytes(dst)) {
        std::cerr << "Desbordamiento al copiar tensor" << std::endl;
        return;
    }
    
    memcpy((char*)dst->data + dst_offset, src->data, src_size);
}


bool run_vit_model(ggml_context* ctx, ggml_backend_t backend, const std::string& model_filename) {
    std::cout << "Initializing ViT model..." << std::endl;
    
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
        } else if (tensor_data.n_dims == 4) {
            tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                      tensor_data.dims[0], tensor_data.dims[1],
                                      tensor_data.dims[2], tensor_data.dims[3]);
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

    auto get_metadata_float = [&](const std::string& key) -> float {
        GGUFMetadata md = read_metadata(model_filename, key);
        if (md.type == GGUF_TYPE_COUNT) {
            std::cerr << "Error: Missing metadata " << key << std::endl;
            return 0.0f;
        }
        return md.value.f32;
    };

    int image_size = get_metadata_int("vit.image_size");
    int patch_size = get_metadata_int("vit.patch_size");
    int num_layers = get_metadata_int("vit.block_count");
    int hidden_dim = get_metadata_int("vit.embedding_length");
    int num_heads = get_metadata_int("vit.attention.head_count");
    float norm_eps = get_metadata_float("vit.attention.layer_norm_epsilon");
    
    // 1. Preparar entrada (imagen)
    ggml_tensor* input_image = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                               image_size, image_size, 3, 1);
    
    // 2. Proyección de parches (patch embedding)
    ggml_tensor* patch_proj = load_tensor("patch_embed.proj.weight");
    if (!patch_proj) return false;
    
    ggml_tensor* patch_emb = ggml_conv_2d(ctx, input_image, patch_proj, 
                                     patch_size, patch_size,  // stride
                                     0, 0,                   // padding
                                     1, 1);                  // dilation
    
    // Reformar a [num_patches, hidden_dim]
    int num_patches = (image_size / patch_size) * (image_size / patch_size);
    patch_emb = ggml_reshape_2d(ctx, patch_emb, hidden_dim, num_patches);
    
    // 3. Añadir token de clase [CLS]
    ggml_tensor* cls_token = load_tensor("cls_token");
    if (!cls_token) return false;
    
    ggml_tensor* embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, num_patches + 1);
    
    // Copiar cls_token y patch_emb a embeddings
    ggml_set_zero(embeddings);
    // Copiar cls_token al inicio (primera columna)
    copy_tensor_data(embeddings, cls_token, 0);
    // Copiar patch_emb después del cls_token
    copy_tensor_data(embeddings, patch_emb, hidden_dim * sizeof(float));
    
    // 4. Añadir positional embeddings
    ggml_tensor* pos_embed = load_tensor("pos_embed");
    if (!pos_embed) return false;
    embeddings = ggml_add(ctx, embeddings, pos_embed);
    
    // 5. Capas del Transformer
    for (int i = 0; i < num_layers; ++i) {
        std::string layer_prefix = "blocks." + std::to_string(i) + ".";
        
        // Atención
        ggml_tensor* norm1 = load_tensor(layer_prefix + "norm1.weight");
        if (!norm1) return false;
        ggml_tensor* attn_norm = layer_norm(ctx, embeddings, norm1, nullptr, false, norm_eps);
        
        // Proyecciones Q, K, V
        ggml_tensor* q_proj = load_tensor(layer_prefix + "attn.q_proj.weight");
        ggml_tensor* k_proj = load_tensor(layer_prefix + "attn.k_proj.weight");
        ggml_tensor* v_proj = load_tensor(layer_prefix + "attn.v_proj.weight");
        if (!q_proj || !k_proj || !v_proj) return false;
        
        ggml_tensor* q = ggml_mul_mat(ctx, q_proj, attn_norm);
        ggml_tensor* k = ggml_mul_mat(ctx, k_proj, attn_norm);
        ggml_tensor* v = ggml_mul_mat(ctx, v_proj, attn_norm);
        
        // Atención multi-cabeza
        int head_dim = hidden_dim / num_heads;
        q = ggml_reshape_3d(ctx, q, head_dim, num_heads, num_patches + 1);
        k = ggml_reshape_3d(ctx, k, head_dim, num_heads, num_patches + 1);
        v = ggml_reshape_3d(ctx, v, head_dim, num_heads, num_patches + 1);
        
        ggml_tensor* attention = multi_head_attention(ctx, q, k, v, false);
        attention = ggml_reshape_2d(ctx, attention, hidden_dim, num_patches + 1);
        
        // Proyección de salida y conexión residual
        ggml_tensor* out_proj = load_tensor(layer_prefix + "attn.out_proj.weight");
        if (!out_proj) return false;
        attention = ggml_mul_mat(ctx, out_proj, attention);
        embeddings = ggml_add(ctx, embeddings, attention);
        
        // MLP (Feed Forward Network)
        ggml_tensor* norm2 = load_tensor(layer_prefix + "norm2.weight");
        if (!norm2) return false;
        ggml_tensor* mlp_norm = layer_norm(ctx, embeddings, norm2, nullptr, false, norm_eps);
        
        ggml_tensor* fc1 = load_tensor(layer_prefix + "mlp.fc1.weight");
        ggml_tensor* fc1_bias = load_tensor(layer_prefix + "mlp.fc1.bias");
        ggml_tensor* fc2 = load_tensor(layer_prefix + "mlp.fc2.weight");
        ggml_tensor* fc2_bias = load_tensor(layer_prefix + "mlp.fc2.bias");
        if (!fc1 || !fc1_bias || !fc2 || !fc2_bias) return false;
        
        ggml_tensor* mlp = feed_forward(ctx, mlp_norm, fc1, fc1_bias, "gelu");
        mlp = feed_forward(ctx, mlp, fc2, fc2_bias, "linear");
        embeddings = ggml_add(ctx, embeddings, mlp);
    }
    
    // 6. Extraer token [CLS] y normalización final
    ggml_tensor* cls_output = ggml_view_1d(ctx, embeddings, hidden_dim, 0);
    ggml_tensor* norm = load_tensor("norm.weight");
    if (!norm) return false;
    cls_output = layer_norm(ctx, cls_output, norm, nullptr, false, norm_eps);
    
    // 7. Clasificador
    ggml_tensor* classifier = load_tensor("head.weight");
    if (!classifier) return false;
    ggml_tensor* output = ggml_mul_mat(ctx, classifier, cls_output);
    
    // 8. Construir y ejecutar el grafo computacional
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    ggml_backend_graph_compute(backend, gf);
    
    std::cout << "ViT model execution completed" << std::endl;
    return true;
}