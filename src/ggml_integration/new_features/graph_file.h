#ifndef GRAPH_FILE_H
#define GRAPH_FILE_H

#include <string>
#include <vector>
#include "graph_data_structs.h"

/**
 * @brief Writes graph data to a binary .graph file
 * @param filename Output filename (e.g., "model.graph")
 * @param graph_data GraphData structure to serialize
 * @return true if successful, false on error
 */
bool write_graph_data(const std::string& filename, const GraphData& graph_data);

/**
 * @brief Reads a metadata entry by key from a .graph file
 * @param filename Input .graph filename
 * @param key Metadata key to search for
 * @return GGUFMetadata structure if found, or empty metadata with type=GGUF_TYPE_COUNT if not found
 */
GGUFMetadata read_metadata(const std::string& filename, const std::string& key);

/**
 * @brief Reads a tensor by name from a .graph file
 * @param filename Input .graph filename
 * @param name Tensor name to search for
 * @return GGUFTensor structure if found, or empty tensor with name="" if not found
 */
GGUFTensor read_tensor(const std::string& filename, const std::string& name);

// Helper functions for graph file operations
void skip_metadata(std::ifstream& in);  // Skips metadata section in input stream
size_t type_size(enum gguf_type type);  // Returns byte size for GGUF data types
size_t ggml_type_size(enum ggml_type type);  // Returns byte size for GGML data types
    
/**
 * @brief Reads an array of specified type from input stream
 * @tparam T Data type to read
 * @param in Input file stream
 * @param size Number of elements to read
 * @return Vector containing the read data
 */
template<typename T>
std::vector<T> read_array(std::ifstream& in, size_t size);

// Debug/utility functions
void print_graph_data_from_struct(const GraphData& graph_data, const char *output_filename);  // Prints graph structure to file
void print_metadata_from_file(const std::string& input_filename, const std::string& output_filename);  // Prints metadata to file
void print_tensor_data_from_file(const std::string &input_filename, const std::string &output_filename);  // Prints tensor data to file

#endif // GRAPH_FILE_H

