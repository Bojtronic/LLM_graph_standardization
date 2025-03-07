#include "model_modules.h"
#include <cstdio>
#include <cstring>
#include <cstdlib> // Para rand()

// Función para construir el grafo de cómputo del modelo
// Parámetros:
// - ctx: Contexto de GGML.
// - input: Tensor de entrada.
// - model_type: Tipo de modelo ("llama2", "vit", "whisper", "deepseek").
struct ggml_cgraph * build_model_graph(ggml_context * ctx, ggml_tensor * input, const char * model_type) {
    // Verificar que el contexto y la entrada sean válidos
    if (!ctx || !input) {
        fprintf(stderr, "Error: Contexto o entrada no válidos.\n");
        return nullptr;
    }

    // Crear un grafo de cómputo
    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // Módulo de embeddings
    ggml_tensor * embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input->ne[0], 128); // Ejemplo: 128 dimensiones
    if (!embeddings) {
        fprintf(stderr, "Error: No se pudo crear el tensor de embeddings.\n");
        return nullptr;
    }
    ggml_set_name(embeddings, "embeddings");

    // Módulo de codificaciones posicionales
    if (strcmp(model_type, "llama2") == 0 || strcmp(model_type, "deepseek") == 0) {
        embeddings = positional_encoding(ctx, embeddings, "rope", 64, 0, 10000.0f); // RoPE
    } else if (strcmp(model_type, "vit") == 0) {
        embeddings = positional_encoding(ctx, embeddings, "sinusoidal", 0, 0, 10000.0f); // Sinusoidal
    } else {
        fprintf(stderr, "Advertencia: No se aplicó codificación posicional para el modelo %s.\n", model_type);
    }

    // Crear tensores de Consulta, Clave y Valor
    ggml_tensor * Q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128); // Ejemplo: 128 dimensiones
    ggml_tensor * K = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);
    ggml_tensor * V = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128);

    if (!Q || !K || !V) {
        fprintf(stderr, "Error: No se pudieron crear los tensores Q, K o V.\n");
        return nullptr;
    }

    // Inicializar con valores aleatorios (simulación de pesos aprendidos)
    for (int i = 0; i < 128 * 128; i++) {
        ((float *)Q->data)[i] = (float)rand() / RAND_MAX;
        ((float *)K->data)[i] = (float)rand() / RAND_MAX;
        ((float *)V->data)[i] = (float)rand() / RAND_MAX;
    }

    // Módulo de atención
    ggml_tensor * attn_output;
    if (strcmp(model_type, "whisper") == 0) {
        attn_output = cross_attention(ctx, Q, K, V);
    } else if (strcmp(model_type, "deepseek") == 0) {
        // Cargar pesos aprendidos para MLA (W_Q, W_K, W_V)
        ggml_tensor * W_Q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 128); // Ejemplo: 64 dimensiones latentes
        ggml_tensor * W_K = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 128);
        ggml_tensor * W_V = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 128);

        if (!W_Q || !W_K || !W_V) {
            fprintf(stderr, "Error: No se pudieron crear los pesos para MLA.\n");
            return nullptr;
        }

        attn_output = multi_head_latent_attention(ctx, Q, K, V, W_Q, W_K, W_V, 64, true); // Con máscara causal
    } else {
        attn_output = multi_head_attention(ctx, Q, K, V, true); // Con máscara causal
    }

    if (!attn_output) {
        fprintf(stderr, "Error: No se pudo calcular la salida de atención.\n");
        return nullptr;
    }

    // Módulo de red feed-forward
    ggml_tensor * ff_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128); // Peso de la capa lineal
    ggml_tensor * ff_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128); // Sesgo de la capa lineal

    if (!ff_weight || !ff_bias) {
        fprintf(stderr, "Error: No se pudieron crear los tensores para la red feed-forward.\n");
        return nullptr;
    }

    ggml_tensor * ff_output = feed_forward(ctx, attn_output, ff_weight, ff_bias, "gelu");

    if (!ff_output) {
        fprintf(stderr, "Error: No se pudo calcular la salida de la red feed-forward.\n");
        return nullptr;
    }

    // Módulo de normalización
    bool use_rmsnorm = (strcmp(model_type, "llama2") == 0 || strcmp(model_type, "deepseek") == 0);
    ff_output = layer_norm(ctx, ff_output, use_rmsnorm, 1e-6f); // eps = 1e-6

    if (!ff_output) {
        fprintf(stderr, "Error: No se pudo aplicar la normalización.\n");
        return nullptr;
    }

    // Construir el grafo de cómputo
    ggml_build_forward_expand(gf, ff_output);

    return gf;
}
