#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include "graph_data_structs.h"
#include "gguf.h"

/**
 * @brief Parsea un contexto GGUF y almacena los datos en estructuras
 * @param ctx Contexto GGUF cargado
 * @param fname Nombre del archivo GGUF (para leer datos de tensores)
 * @return Estructura GraphData con todos los datos cargados
 */
GraphData gguf_parse_to_struct(const struct gguf_context *ctx, const char *fname);

#endif // GGUF_LOADER_H

