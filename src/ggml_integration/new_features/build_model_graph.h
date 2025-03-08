#ifndef BUILD_MODEL_GRAPH_H
#define BUILD_MODEL_GRAPH_H

#include "model_modules.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

// Declaración de la función build_model_graph
// Parámetros:
// - ctx: Contexto de GGML.
// - input: Tensor de entrada.
// - model_type: Tipo de modelo ("llama2", "vit", "whisper", "deepseek").
// Retorna:
// - Un puntero al grafo de cómputo (ggml_cgraph) construido.
struct ggml_cgraph * build_model_graph(ggml_context * ctx, ggml_tensor * input, const char * model_type);

#endif // BUILD_MODEL_GRAPH_H
