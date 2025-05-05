#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include "graph_data_structs.h"
#include <ggml.h>

/**
 * @file gguf_loader.h
 * @brief Functions for loading and parsing GGUF model files
 */

/**
 * @brief Parses a GGUF context and stores all data in a GraphData structure
 * @param ctx Loaded GGUF context
 * @param fname GGUF filename (for reading tensor data)
 * @return GraphData structure with all loaded data
 */
GraphData gguf_graph_data(const struct gguf_context *ctx, const char *fname);

/**
 * @brief Main function to load a GGUF file and get its configuration
 * @param fname GGUF filename
 * @return true if loading was successful, false otherwise
 */
bool get_gguf_config(const char *fname);


#endif // GGUF_LOADER_H

