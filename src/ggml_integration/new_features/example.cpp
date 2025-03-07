#include "model_modules.h"
#include <cstdio>
#include <cstdlib> // Para rand() y srand()
#include <ctime>   // Para time()

int main() {
    // Inicializar la semilla para valores aleatorios
    srand(time(NULL));

    // Inicializar el contexto de GGML
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024, // 1 MB de memoria
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Error: No se pudo inicializar el contexto de GGML.\n");
        return 1;
    }

    // Crear tensor de entrada
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128); // Ejemplo: 128 dimensiones

    // Inicializar el tensor de entrada con valores aleatorios
    for (int i = 0; i < 128 * 128; i++) {
        ((float *)input->data)[i] = (float)rand() / RAND_MAX; // Valores aleatorios entre 0 y 1
    }

    // Construir el grafo de cómputo para un modelo específico
    const char * model_type = "llama2"; // Cambiar a "vit", "whisper" o "deepseek" según sea necesario
    struct ggml_cgraph * gf = build_model_graph(ctx, input, model_type);

    if (!gf) {
        fprintf(stderr, "Error: No se pudo construir el grafo de cómputo.\n");
        ggml_free(ctx); // Liberar memoria antes de salir
        return 1;
    }

    // Ejecutar el grafo de cómputo
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    // Imprimir algunos valores del tensor de salida (opcional)
    //ggml_tensor * output = gf->nodes[gf->n_nodes - 1]; 
    ggml_tensor * output = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1); // Último tensor en el grafo
    printf("Valores de salida (primeros 10 elementos):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", ((float *)output->data)[i]);
    }
    printf("\n");

    // Liberar memoria
    ggml_free(ctx);

    return 0;
}
