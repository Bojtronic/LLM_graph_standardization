#ifndef GGUF_LOADER_H
#define GGUF_LOADER_H

#include "graph_data_structs.h"
#include "arch_info.h"
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
//GraphData gguf_graph_data(const struct gguf_context *ctx, const char *fname);
GraphData gguf_graph_data(const struct gguf_context *ctx, const char *file_gguf, const char *file_graph);

/**
 * @brief Main function to load a GGUF file and get its configuration
 * @param fname GGUF filename
 * @return true if loading was successful, false otherwise
 */
bool get_gguf_config(const char *fname);

/**
 * @brief Detects model architecture and configuration from metadata
 * @param metadata Model metadata
 * @return ModelConfig with detected architecture and features
 */
//ModelConfig detect_model_config(const std::vector<GGUFMetadata>& metadata);


enum ggml_op infer_operation(const std::string &tensor_name, llm_arch arch);

/**
 * @brief Infers the operation based on the tensor name
 * @param name tensor name
 * @return Inferred operation
 */
ggml_op infer_operation_fallback(const std::string& name);

/**
 * @brief Infers the source tensors based on the actual tensor name
 * @param name tensor name
 * @param tensors List of all tensors in the graph at the time of inference
 * @note This function is used to infer the source tensors for a given tensor
 * @return List of inferred source tensor names
 */
std::vector<std::string> infer_src_tensors(const std::string& tensor_name, llm_arch arch);
/**
 * @brief Infers the destination tensor based on the actual tensor name
 * @param name tensor name
 * @return Inferred destination tensor name
 */
std::string infer_dst_tensor(const std::string& tensor_name, llm_arch arch);

std::string generate_computational_graph(const GraphData& graph);

bool save_dot_to_file(const std::string& dot_content, const std::string& filename);

#endif // GGUF_LOADER_H

