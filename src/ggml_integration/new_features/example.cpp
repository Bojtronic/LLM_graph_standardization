#include "build_model_graph.h" // Incluye la definición de la función build_model_graph
#include "ggml-cpu.h"         // Incluye las funciones de GGML para CPU
#include <ctime>              // Para usar time() en la inicialización de la semilla aleatoria

int main() {
    // Número de hilos para la ejecución del grafo de cómputo
    int n_threads = 1;

    // Inicializar la semilla para valores aleatorios (para el tensor de entrada)
    // time(NULL) devuelve el tiempo actual en segundos, lo que garantiza que la semilla sea diferente en cada ejecución.
    srand(time(NULL));

    // Inicializar el contexto de GGML
    // Un contexto en GGML es un entorno de memoria donde se almacenan los tensores y grafos de cómputo.
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024, // 1 MB de memoria reservada para el contexto
        /*.mem_buffer =*/ NULL,        // No se usa un buffer de memoria preasignado
        /*.no_alloc   =*/ false,       // Permitir la asignación de memoria dentro del contexto
    };

    // Crear el contexto de GGML
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        // Si no se pudo inicializar el contexto, mostrar un error y salir del programa
        fprintf(stderr, "Error: No se pudo inicializar el contexto de GGML.\n");
        return 1;
    }

    // Crear tensor de entrada
    // Un tensor es una estructura multidimensional que almacena datos (en este caso, de tipo float).
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 128); // Tensor 2D de 128x128

    // Inicializar el tensor de entrada con valores aleatorios
    // Los valores aleatorios se generan usando rand() y se normalizan al rango [0, 1].
    for (int i = 0; i < 128 * 128; i++) {
        ((float *)input->data)[i] = (float)rand() / RAND_MAX; // Valores aleatorios entre 0 y 1
    }

    // Construir el grafo de cómputo para un modelo específico
    // build_model_graph es una función que define la estructura del modelo (por ejemplo, atención, feed-forward, etc.).
    const char * model_type = "llama2"; // Tipo de modelo a construir (puede ser "vit", "whisper", "deepseek", etc.)
    struct ggml_cgraph * gf = build_model_graph(ctx, input, model_type);

    if (!gf) {
        // Si no se pudo construir el grafo, mostrar un error, liberar memoria y salir del programa
        fprintf(stderr, "Error: No se pudo construir el grafo de cómputo.\n");
        ggml_free(ctx); // Liberar memoria antes de salir
        return 1;
    }

    // Ejecutar el grafo de cómputo
    // ggml_graph_compute_with_ctx ejecuta el grafo en el contexto GGML usando n_threads hilos.
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    // Imprimir algunos valores del tensor de salida (opcional)
    // Obtener el último tensor en el grafo (la salida del modelo)
    ggml_tensor * output = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1); // Último tensor en el grafo
    printf("Valores de salida (primeros 10 elementos):\n");
    for (int i = 0; i < 10; i++) {
        // Imprimir los primeros 10 valores del tensor de salida
        printf("%f ", ((float *)output->data)[i]);
    }
    printf("\n");

    // Liberar memoria
    // ggml_free libera todo el contexto de GGML, incluyendo tensores y grafos.
    ggml_free(ctx);

    // Finalizar el programa
    return 0;
}
