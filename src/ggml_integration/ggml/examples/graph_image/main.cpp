#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include <cstdint>
#include <variant>
#include "gguf.h"
#include "ggml.h"


/**
 * @file graph_data_structs.h
 * @brief Definitions of structures to handle GGUF data
 */

/**
 * @brief Header of a GGUF file
 */
struct GGUFHeader {
    uint64_t n_tensors;     // Number of tensors in the file
    uint64_t n_kv;          // Number of key-value metadata pairs
};

/**
 * @brief GGUF metadata (key-value pairs)
 */
struct GGUFMetadata {
    std::string key;        // Metadata name/key
    enum gguf_type type;    // Type of the stored value
    
    // Value using a union for different possible types
    union {
        uint8_t u8;
        int8_t i8;
        uint16_t u16;
        int16_t i16;
        uint32_t u32;
        int32_t i32;
        float f32;
        uint64_t u64;
        int64_t i64;
        double f64;
        bool b;
    } value;
    
    // For array types
    struct {
        enum gguf_type type;
        size_t size;
        std::variant<
            std::vector<uint8_t>,   // For raw/bytes types
            std::vector<float>,     // F32
            std::vector<uint16_t>,  // F16
            std::vector<int8_t>,    // I8
            std::vector<int16_t>,   // I16
            std::vector<int32_t>,   // I32
            std::vector<uint32_t>,  // U32
            std::vector<int64_t>,   // I64
            std::vector<uint64_t>,  // U64
            std::vector<double>,    // F64
            std::vector<std::string> // Strings
        > data;
    } array;
    
    std::string str;        // For strings
    
    // Default constructor
    GGUFMetadata() : type(GGUF_TYPE_COUNT) {}
};

/**
 * @brief Representation of a GGUF tensor
 */
struct GGUFTensor {
    std::string name;           // Tensor name
    enum ggml_type type;        // Tensor data type
    size_t size;                // Tensor size in bytes
    int32_t n_dims;             // Number of tensor dimensions
    std::vector<int64_t> dims;  // Tensor dimensions

    enum ggml_op op;            // Operation that produces this tensor
    ggml_unary_op unary_op;     // Unary operation if applicable
    std::vector<std::string> src_tensors; ///< Names of input tensors
    std::string dst_tensor;     // Name of destination tensor that will use this result

    // Operation parameters (similar to ggml_tensor)
    //std::vector<int32_t> op_params;

    // Data storage using variant
    std::variant<
        std::vector<uint8_t>,    // For quantized types
        std::vector<int8_t>,     // For Int8
        std::vector<int16_t>,    // For Int16
        std::vector<float>,      // For F32
        std::vector<uint16_t>,   // For F16
        std::vector<int32_t>     // For I32
    > data;

    /**
     * @brief Gets tensor data as vector of specified type
     * @tparam T Requested data type
     * @return Pointer to data vector or nullptr if type doesn't match
     */
    template<typename T>
    const std::vector<T>* get_data() const {
        return std::get_if<std::vector<T>>(&data);
    }
};

/**
 * @brief Main container for GGUF data
 */
struct GraphData {
    GGUFHeader header;                      // GGUF header
    std::vector<GGUFMetadata> metadata;     // Metadata list
    std::vector<GGUFTensor> tensors;        // Tensor list
    
    /**
     * @brief Finds metadata by key
     * @param key Key to search for
     * @return Pointer to the metadata or nullptr if not found
     */
    const GGUFMetadata* find_metadata(const std::string& key) const;
    
    /**
     * @brief Finds a tensor by name
     * @param name Tensor name to search for
     * @return Pointer to the tensor or nullptr if not found
     */
    const GGUFTensor* find_tensor(const std::string& name) const;
};

/**
 * @brief Gets the size in bytes for a given GGUF type
 * @param type The GGUF type to check
 * @return Size of the type in bytes
 */
static inline size_t type_size(enum gguf_type type);

/**
 * @brief Reads array data from GGUF context into a vector of specified type
 * @tparam T Type of data to read (must match GGUF array type)
 * @param ctx GGUF context pointer
 * @param i Index of the array in GGUF context
 * @param size Number of elements to read
 * @return Vector containing the read data, or empty vector on error
 */
template <typename T>
static std::vector<T> read_array_data(const gguf_context *ctx, int64_t i, size_t size) {
    // Check for invalid input parameters
    if (!ctx || size == 0) {
        return {};
    }

    // Get raw array data pointer from GGUF context
    const void *src_data = gguf_get_arr_data(ctx, i);
    if (!src_data) {
        return {};
    }

    // Create destination vector with requested size
    std::vector<T> dest(size);
    
    // Special handling for byte arrays (use memcpy for efficiency)
    if constexpr (std::is_same_v<T, uint8_t>) {
        memcpy(dest.data(), src_data, size * sizeof(T));
    } else {
        // For typed arrays, use copy with proper type casting
        const T *typed_src = static_cast<const T *>(src_data);
        std::copy(typed_src, typed_src + size, dest.begin());
    }
    
    return dest;
}



std::string generate_computational_graph(const GraphData& graph) {
    std::string dot_output;
    
    // Inicio del gráfico DOT
    dot_output = "digraph ComputationalGraph {\n";
    dot_output += "    rankdir=LR;\n";
    dot_output += "    node [shape=box, style=filled, fillcolor=\"#E6E6FA\"];\n";
    dot_output += "    edge [arrowsize=0.8];\n\n";
    
    std::map<std::string, bool> processed_nodes;
    
    auto get_tensor_label = [](const GGUFTensor& tensor) {
        std::string label = tensor.name + "\\n";
        
        if (tensor.n_dims == 1) {
            label += "[" + std::to_string(tensor.dims[0]) + "]";
        } else if (tensor.n_dims == 2) {
            label += "[" + std::to_string(tensor.dims[0]) + "x" + 
                     std::to_string(tensor.dims[1]) + "]";
        }
        
        label += "\\nType: ";
        switch(tensor.type) {
            case GGML_TYPE_F32: label += "f32"; break;
            case GGML_TYPE_F16: label += "f16"; break;
            case GGML_TYPE_I32: label += "i32"; break;
            default: label += "unknown"; break;
        }
        
        return label;
    };
    
    for (const auto& tensor : graph.tensors) {
        if (processed_nodes.find(tensor.name) == processed_nodes.end()) {
            dot_output += "    \"" + tensor.name + "\" [label=\"" + 
                          get_tensor_label(tensor) + "\"";
            
            if (tensor.src_tensors.empty()) {
                dot_output += ", fillcolor=\"#98FB98\"";
            }
            
            dot_output += "];\n";
            processed_nodes[tensor.name] = true;
        }
        
        if (!tensor.src_tensors.empty()) {
            std::string op_node_name = "op_" + tensor.name;
            std::string op_label;
            
            switch(tensor.op) {
                case GGML_OP_ADD: op_label = "ADD"; break;
                case GGML_OP_MUL: op_label = "MUL"; break;
                case GGML_OP_MUL_MAT: op_label = "MATMUL"; break;
                case GGML_OP_UNARY: 
                    op_label = (tensor.unary_op == GGML_UNARY_OP_RELU) ? "RELU" : "UNARY";
                    break;
                case GGML_OP_SOFT_MAX: op_label = "SOFTMAX"; break;
                default: op_label = "OP"; break;
            }
            
            dot_output += "    \"" + op_node_name + "\" [shape=ellipse, label=\"" + 
                         op_label + "\", fillcolor=\"#FFD700\"];\n";
            
            for (const auto& src : tensor.src_tensors) {
                dot_output += "    \"" + src + "\" -> \"" + op_node_name + "\";\n";
            }
            
            dot_output += "    \"" + op_node_name + "\" -> \"" + tensor.name + "\";\n";
        }
    }
    
    dot_output += "}\n";
    return dot_output;
}

bool save_dot_to_file(const std::string& dot_content, const std::string& filename) {
    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
        return false;
    }
    out_file << dot_content;
    out_file.close();
    return true;
}




// Función auxiliar para crear tensores
GGUFTensor create_tensor(const std::string& name, ggml_type type, 
                        const std::vector<int64_t>& dims, ggml_op op,
                        const std::vector<std::string>& src_tensors = {}) {
    GGUFTensor tensor;
    tensor.name = name;
    tensor.type = type;
    tensor.n_dims = dims.size();
    tensor.dims = dims;
    tensor.op = op;
    tensor.src_tensors = src_tensors;
    return tensor;
}

GraphData create_sample_graph() {
    GraphData graph;
    graph.header.n_tensors = 10;
    graph.header.n_kv = 0;
    
    // Tensores de entrada
    graph.tensors.push_back(create_tensor("input1", GGML_TYPE_F32, {1024}, GGML_OP_NONE));
    graph.tensors.push_back(create_tensor("input2", GGML_TYPE_F32, {1024}, GGML_OP_NONE));
    graph.tensors.push_back(create_tensor("weight1", GGML_TYPE_F32, {4096, 1024}, GGML_OP_NONE));
    graph.tensors.push_back(create_tensor("weight2", GGML_TYPE_F32, {1024, 4096}, GGML_OP_NONE)); // Cambiado a [1024, 4096]
    graph.tensors.push_back(create_tensor("bias1", GGML_TYPE_F32, {4096}, GGML_OP_NONE));
    
    // Primera capa: input1 * weight1 + bias1
    graph.tensors.push_back(create_tensor("matmul_out", GGML_TYPE_F32, {4096}, GGML_OP_MUL_MAT, 
                                    {"input1", "weight1"}));
    
    graph.tensors.push_back(create_tensor("add_out", GGML_TYPE_F32, {4096}, GGML_OP_ADD, 
                                    {"matmul_out", "bias1"}));
    
    // ReLU
    GGUFTensor relu_tensor = create_tensor("relu_out", GGML_TYPE_F32, {4096}, GGML_OP_UNARY, {"add_out"});
    relu_tensor.unary_op = GGML_UNARY_OP_RELU;
    graph.tensors.push_back(relu_tensor);
    
    // Segunda capa: relu_out * weight2 (ahora weight2 tiene dimensiones [1024, 4096])
    graph.tensors.push_back(create_tensor("matmul2_out", GGML_TYPE_F32, {1024}, GGML_OP_MUL_MAT,
                                    {"relu_out", "weight2"}));
    
    // Multiplicación por input2 (element-wise)
    graph.tensors.push_back(create_tensor("mul_out", GGML_TYPE_F32, {1024}, GGML_OP_MUL, 
                                    {"matmul2_out", "input2"}));
    
    // Softmax final
    graph.tensors.push_back(create_tensor("output", GGML_TYPE_F32, {1024}, GGML_OP_SOFT_MAX, 
                                    {"mul_out"}));
    
    return graph;
}

int main() {
    // Crear el grafo de ejemplo
    GraphData sample_graph = create_sample_graph();
    
    // Generar el gráfico DOT
    std::string dot_graph = generate_computational_graph(sample_graph);
    
    // Guardar en archivo
    save_dot_to_file(dot_graph, "sample_graph.dot");
    
    std::cout << "Grafo de ejemplo creado y guardado en sample_graph.dot" << std::endl;
    return 0;
}

