#include "model_modules.h"

int main() {
    // Inicializar el contexto de GGML
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024, // 1 MB de memoria
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    // Crear tensor de entrada
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128); // Ejemplo: 128 dimensiones

    // Construir el grafo de cómputo para un modelo específico
    const char * model_type = "llama2"; // Cambiar a "vit", "whisper" o "deepseek" según sea necesario
    struct ggml_cgraph * gf = build_model_graph(ctx, input, model_type);

    if (!gf) {
        fprintf(stderr, "Error: No se pudo construir el grafo de cómputo.\n");
        return 1;
    }

    // Ejecutar el grafo de cómputo
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // Liberar memoria
    ggml_free(ctx);

    return 0;
}

