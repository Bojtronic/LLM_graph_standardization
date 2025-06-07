#include "vit_runner.h"
#include <iostream>
#include <ggml-cuda.h>
#include <ggml-cpu.h>

#include "model_modules.h"
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


bool run_vit_model(ggml_context* ctx, ggml_backend_t backend, GraphData& graph_data) {
    std::cout << "Initializing ViT model..." << std::endl;
    
    // Obtener parámetros del modelo desde los metadatos GGUF
    int image_size = graph_data.find_metadata("vit.image_size")->value.i32;
    int patch_size = graph_data.find_metadata("vit.patch_size")->value.i32;
    int num_layers = graph_data.find_metadata("vit.block_count")->value.i32;
    int hidden_dim = graph_data.find_metadata("vit.embedding_length")->value.i32;
    int num_heads = graph_data.find_metadata("vit.attention.head_count")->value.i32;
    float norm_eps = graph_data.find_metadata("vit.attention.layer_norm_epsilon")->value.f32;
    
    // 1. Preparar entrada (imagen)
    ggml_tensor* input_image = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                               image_size, image_size, 3, 1);
    
    // 2. Proyección de parches (patch embedding)
    ggml_tensor* patch_proj = get_layer_tensor(ctx, graph_data, "patch_embed.proj.weight");
    ggml_tensor* patch_emb = ggml_conv_2d(ctx, input_image, patch_proj, 
                                     patch_size, patch_size,  // stride
                                     0, 0,                   // padding
                                     1, 1);                  // dilation
    
    // Reformar a [num_patches, hidden_dim]
    int num_patches = (image_size / patch_size) * (image_size / patch_size);
    patch_emb = ggml_reshape_2d(ctx, patch_emb, hidden_dim, num_patches);
    
    // 3. Añadir token de clase [CLS]
    ggml_tensor* cls_token = get_layer_tensor(ctx, graph_data, "cls_token");
    ggml_tensor* embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, num_patches + 1);
    
    // Copiar cls_token y patch_emb a embeddings
    // Copiar cls_token y patch_emb a embeddings
    ggml_set_zero(embeddings);
    // Copiar cls_token al inicio (primera columna)
    copy_tensor_data(embeddings, cls_token, 0);
    // Copiar patch_emb después del cls_token
    copy_tensor_data(embeddings, patch_emb, hidden_dim * sizeof(float));
    
    // 4. Añadir positional embeddings
    ggml_tensor* pos_embed = get_layer_tensor(ctx, graph_data, "pos_embed");
    embeddings = ggml_add(ctx, embeddings, pos_embed);
    
    // 5. Capas del Transformer
    for (int i = 0; i < num_layers; ++i) {
        std::string layer_prefix = "blocks." + std::to_string(i) + ".";
        
        // Atención
        ggml_tensor* norm1 = get_layer_tensor(ctx, graph_data, layer_prefix + "norm1.weight");
        ggml_tensor* attn_norm = layer_norm(ctx, embeddings, norm1, nullptr, false, norm_eps);
        
        // Proyecciones Q, K, V
        ggml_tensor* q_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.q_proj.weight");
        ggml_tensor* k_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.k_proj.weight");
        ggml_tensor* v_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.v_proj.weight");
        
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
        ggml_tensor* out_proj = get_layer_tensor(ctx, graph_data, layer_prefix + "attn.out_proj.weight");
        attention = ggml_mul_mat(ctx, out_proj, attention);
        embeddings = ggml_add(ctx, embeddings, attention);
        
        // MLP (Feed Forward Network)
        ggml_tensor* norm2 = get_layer_tensor(ctx, graph_data, layer_prefix + "norm2.weight");
        ggml_tensor* mlp_norm = layer_norm(ctx, embeddings, norm2, nullptr, false, norm_eps);
        
        ggml_tensor* fc1 = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.fc1.weight");
        ggml_tensor* fc1_bias = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.fc1.bias");
        ggml_tensor* fc2 = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.fc2.weight");
        ggml_tensor* fc2_bias = get_layer_tensor(ctx, graph_data, layer_prefix + "mlp.fc2.bias");
        
        ggml_tensor* mlp = feed_forward(ctx, mlp_norm, fc1, fc1_bias, "gelu");
        mlp = feed_forward(ctx, mlp, fc2, fc2_bias, "linear");
        embeddings = ggml_add(ctx, embeddings, mlp);
    }
    
    // 6. Extraer token [CLS] y normalización final
    ggml_tensor* cls_output = ggml_view_1d(ctx, embeddings, hidden_dim, 0);
    ggml_tensor* norm = get_layer_tensor(ctx, graph_data, "norm.weight");
    cls_output = layer_norm(ctx, cls_output, norm, nullptr, false, norm_eps);
    
    // 7. Clasificador
    ggml_tensor* classifier = get_layer_tensor(ctx, graph_data, "head.weight");
    ggml_tensor* output = ggml_mul_mat(ctx, classifier, cls_output);
    
    // 8. Construir y ejecutar el grafo computacional
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    ggml_backend_graph_compute(backend, gf);
    
    std::cout << "ViT model execution completed" << std::endl;
    return true;
}
