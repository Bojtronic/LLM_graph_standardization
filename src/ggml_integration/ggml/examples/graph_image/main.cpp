#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include <cstdint>
#include <variant>
#include "gguf.h"
#include "ggml.h"
#include <set>
#include <algorithm>

/**
 * @file graph_data_structs.h
 * @brief Definitions of structures to handle GGUF data
 */

/**
 * @brief Header of a GGUF file
 */
struct GGUFHeader
{
    uint64_t n_tensors; // Number of tensors in the file
    uint64_t n_kv;      // Number of key-value metadata pairs
};

/**
 * @brief GGUF metadata (key-value pairs)
 */
struct GGUFMetadata
{
    std::string key;     // Metadata name/key
    enum gguf_type type; // Type of the stored value

    // Value using a union for different possible types
    union
    {
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
    struct
    {
        enum gguf_type type;
        size_t size;
        std::variant<
            std::vector<uint8_t>,    // For raw/bytes types
            std::vector<float>,      // F32
            std::vector<uint16_t>,   // F16
            std::vector<int8_t>,     // I8
            std::vector<int16_t>,    // I16
            std::vector<int32_t>,    // I32
            std::vector<uint32_t>,   // U32
            std::vector<int64_t>,    // I64
            std::vector<uint64_t>,   // U64
            std::vector<double>,     // F64
            std::vector<std::string> // Strings
            >
            data;
    } array;

    std::string str; // For strings

    // Default constructor
    GGUFMetadata() : type(GGUF_TYPE_COUNT) {}
};

/**
 * @brief Representation of a GGUF tensor
 */
struct GGUFTensor
{
    std::string name;          // Tensor name
    enum ggml_type type;       // Tensor data type
    size_t size;               // Tensor size in bytes
    int32_t n_dims;            // Number of tensor dimensions
    std::vector<int64_t> dims; // Tensor dimensions

    enum ggml_op op;                      // Operation that produces this tensor
    ggml_unary_op unary_op;               // Unary operation if applicable
    std::vector<std::string> src_tensors; ///< Names of input tensors
    std::string dst_tensor;               // Name of destination tensor that will use this result

    // Operation parameters (similar to ggml_tensor)
    // std::vector<int32_t> op_params;

    // Data storage using variant
    std::variant<
        std::vector<uint8_t>,  // For quantized types
        std::vector<int8_t>,   // For Int8
        std::vector<int16_t>,  // For Int16
        std::vector<float>,    // For F32
        std::vector<uint16_t>, // For F16
        std::vector<int32_t>   // For I32
        >
        data;

    /**
     * @brief Gets tensor data as vector of specified type
     * @tparam T Requested data type
     * @return Pointer to data vector or nullptr if type doesn't match
     */
    template <typename T>
    const std::vector<T> *get_data() const
    {
        return std::get_if<std::vector<T>>(&data);
    }
};

/**
 * @brief Main container for GGUF data
 */
struct GraphData
{
    GGUFHeader header;                  // GGUF header
    std::vector<GGUFMetadata> metadata; // Metadata list
    std::vector<GGUFTensor> tensors;    // Tensor list

    /**
     * @brief Finds metadata by key
     * @param key Key to search for
     * @return Pointer to the metadata or nullptr if not found
     */
    const GGUFMetadata *find_metadata(const std::string &key) const;

    /**
     * @brief Finds a tensor by name
     * @param name Tensor name to search for
     * @return Pointer to the tensor or nullptr if not found
     */
    const GGUFTensor *find_tensor(const std::string &name) const;
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
static std::vector<T> read_array_data(const gguf_context *ctx, int64_t i, size_t size)
{
    // Check for invalid input parameters
    if (!ctx || size == 0)
    {
        return {};
    }

    // Get raw array data pointer from GGUF context
    const void *src_data = gguf_get_arr_data(ctx, i);
    if (!src_data)
    {
        return {};
    }

    // Create destination vector with requested size
    std::vector<T> dest(size);

    // Special handling for byte arrays (use memcpy for efficiency)
    if constexpr (std::is_same_v<T, uint8_t>)
    {
        memcpy(dest.data(), src_data, size * sizeof(T));
    }
    else
    {
        // For typed arrays, use copy with proper type casting
        const T *typed_src = static_cast<const T *>(src_data);
        std::copy(typed_src, typed_src + size, dest.begin());
    }

    return dest;
}

std::string generate_computational_graph_test(const GraphData &graph)
{
    std::string dot_output;

    // Configuración inicial del gráfico DOT
    dot_output = "digraph ComputationalGraph {\n";
    dot_output += "    rankdir=LR;\n"; // Flujo izquierda a derecha
    dot_output += "    node [fontname=\"Helvetica\", fontsize=10];\n";
    dot_output += "    edge [arrowsize=0.8];\n";
    dot_output += "    compound=true;\n"; // Permite conectar bordes entre subgrafos\n\n";

    // Función para sanitizar nombres (reemplaza caracteres especiales)
    auto sanitize_name = [](const std::string &name) {
        std::string sanitized;
        for (char c : name) {
            if (c == '.') {
                sanitized += '_'; // Reemplazar puntos por guiones bajos
            }
            else if (std::isalnum(c) || c == '_') {
                sanitized += c;
            }
            else {
                sanitized += '_'; // Reemplazar otros caracteres especiales
            }
        }
        return sanitized;
    };

    // Función para generar la etiqueta completa del tensor
    auto get_tensor_label = [](const GGUFTensor &tensor) {
        std::string label = "{ " + tensor.name + " | ";

        // Dimensiones
        label += "Dims: ";
        if (tensor.dims.empty()) {
            label += "scalar";
        }
        else {
            for (size_t i = 0; i < tensor.dims.size(); ++i) {
                if (i > 0) label += "×";
                label += std::to_string(tensor.dims[i]);
            }
        }

        // Tipo de datos
        label += " | Type: ";
        switch (tensor.type) {
            case GGML_TYPE_F32: label += "f32"; break;
            case GGML_TYPE_F16: label += "f16"; break;
            case GGML_TYPE_I32: label += "i32"; break;
            case GGML_TYPE_I16: label += "i16"; break;
            case GGML_TYPE_I8:  label += "i8"; break;
            case GGML_TYPE_Q2_K: label += "Q2_K"; break;
            case GGML_TYPE_Q3_K: label += "Q3_K"; break;
            case GGML_TYPE_Q6_K: label += "Q6_K"; break;
            default: label += "unknown"; break;
        }

        // Operación (si aplica)
        if (!tensor.src_tensors.empty()) {
            label += " | Op: ";
            switch (tensor.op) {
                case GGML_OP_ADD: label += "ADD"; break;
                case GGML_OP_MUL: label += "MUL"; break;
                case GGML_OP_MUL_MAT: label += "MATMUL"; break;
                case GGML_OP_SOFT_MAX: label += "SOFTMAX"; break;
                case GGML_OP_NORM: label += "NORM"; break;
                case GGML_OP_UNARY: 
                    label += (tensor.unary_op == GGML_UNARY_OP_RELU) ? "RELU" : "UNARY"; 
                    break;
                default: label += "OP"; break;
            }
        }

        label += " }";
        return label;
    };

    // Identificar tensores de entrada y salida
    std::set<std::string> input_tensors;
    std::set<std::string> output_tensors;
    std::map<std::string, int> layer_map; // Mapea tensores a sus capas

    for (const auto &tensor : graph.tensors) {
        if (tensor.src_tensors.empty()) {
            input_tensors.insert(tensor.name);
        }

        // Determinar a qué capa pertenece cada tensor
        size_t blk_pos = tensor.name.find("blk.");
        if (blk_pos != std::string::npos) {
            size_t dot_pos = tensor.name.find('.', blk_pos + 4);
            if (dot_pos != std::string::npos) {
                std::string layer_str = tensor.name.substr(blk_pos + 4, dot_pos - (blk_pos + 4));
                try {
                    int layer = std::stoi(layer_str);
                    layer_map[tensor.name] = layer;
                } catch (...) {}
            }
        }

        bool is_output = true;
        for (const auto &other : graph.tensors) {
            if (std::find(other.src_tensors.begin(), other.src_tensors.end(), tensor.name) != other.src_tensors.end()) {
                is_output = false;
                break;
            }
        }
        if (is_output && !tensor.src_tensors.empty()) {
            output_tensors.insert(tensor.name);
        }
    }

    // Agrupar tensores por capa
    std::map<int, std::vector<GGUFTensor>> layer_tensors;
    for (const auto &tensor : graph.tensors) {
        int layer = -1; // -1 para capas especiales (entrada/salida)
        
        if (tensor.name.find("blk.") != std::string::npos) {
            auto it = layer_map.find(tensor.name);
            if (it != layer_map.end()) {
                layer = it->second;
            }
        }
        layer_tensors[layer].push_back(tensor);
    }

    // Generar subgrafos para cada capa
    for (const auto &layer_pair : layer_tensors) {
        int layer = layer_pair.first;
        const auto &tensors = layer_pair.second;

        if (layer == -1) {
            // Tensores de entrada/salida globales (no están en ninguna capa)
            continue;
        }

        
        dot_output += "    subgraph cluster_" + std::to_string(layer) + " {\n";
        dot_output += "        label=\"Layer " + std::to_string(layer) + "\";\n";
        //dot_output += "        label=\"\"; \n";
        dot_output += "        style=filled;\n";
        //dot_output += "        style=invis;\n";
        dot_output += "        color=lightgrey;\n";
        dot_output += "        fillcolor=\"#f8f8f8\";\n";
        //dot_output += "        fillcolor=none;\n"; // Sin relleno
        dot_output += "        node [style=filled, fillcolor=\"#E6E6FA\"];\n\n";
        

        // Generar nodos para esta capa
        for (const auto &tensor : tensors) {
            std::string safe_name = sanitize_name(tensor.name);
            dot_output += "        " + safe_name + " [label=\"" + get_tensor_label(tensor) + "\", shape=record];\n";
        }

        dot_output += "    }\n\n";
    }

    // Generar nodos especiales (entrada/salida) fuera de los subgrafos
    for (const auto &tensor : graph.tensors) {
        if (layer_map.find(tensor.name) == layer_map.end()) {
            std::string safe_name = sanitize_name(tensor.name);
            std::string fillcolor;

            if (input_tensors.count(tensor.name)) {
                // Tensor de entrada
                dot_output += "    " + safe_name + " [label=\"" + get_tensor_label(tensor) + 
                              "\\n(Input)\", shape=box, style=filled, fillcolor=\"#98FB98\"];\n";
            }
            else if (output_tensors.count(tensor.name)) {
                // Tensor de salida
                dot_output += "    " + safe_name + " [label=\"" + get_tensor_label(tensor) + 
                              "\", shape=box, style=filled, fillcolor=\"#FFA07A\"];\n";
            }
            else {
                // Otros tensores globales - gris intermedio
                fillcolor = "#C0C0C0";
                dot_output += "    " + safe_name + " [label=\"" + get_tensor_label(tensor) + 
                              "\", shape=record, style=filled, fillcolor=\"" + fillcolor + "\"];\n";
            }
        }
    }

    // Generar conexiones (operaciones)
    for (const auto &tensor : graph.tensors) {
        if (!tensor.src_tensors.empty()) {
            std::string safe_dest = sanitize_name(tensor.name);

            for (const auto &src : tensor.src_tensors) {
                std::string safe_src = sanitize_name(src);
                
                // Determinar si la conexión cruza capas
                int src_layer = -1;
                int dest_layer = -1;
                auto src_it = layer_map.find(src);
                auto dest_it = layer_map.find(tensor.name);
                
                if (src_it != layer_map.end()) src_layer = src_it->second;
                if (dest_it != layer_map.end()) dest_layer = dest_it->second;
                
                // Determinar si es la conexión de entrada al primer bloque (blk.0)
                bool is_first_block_input = (dest_layer == 0 && src_layer == -1);
                // Determinar si es la conexión de salida del último bloque
                bool is_last_block_output = (src_layer == (layer_tensors.size() - 1) && dest_layer == -1);

                bool is_consecutive_cluster_connection = (src_layer != -1 && dest_layer != -1 && abs(src_layer - dest_layer) == 1);

                if (!is_first_block_input && !is_last_block_output) {
                    if (src_layer != -1 && dest_layer != -1 && src_layer != dest_layer) {
                        dot_output += "    " + safe_src + " -> " + safe_dest;
                        dot_output += " [ltail=cluster_" + std::to_string(src_layer);
                        dot_output += ", lhead=cluster_" + std::to_string(dest_layer) + "];\n";
                    } 
                    else {
                        dot_output += "    " + safe_src + " -> " + safe_dest + ";\n"; 
                    }
                }
            }
        }
    }

    // Ordenar capas horizontalmente
    dot_output += "\n    // Ordenamiento de capas\n";
    for (int i = 0; i < static_cast<int>(layer_tensors.size()) - 2; i++) {
        if (layer_tensors.count(i) && layer_tensors.count(i+1)) {
            dot_output += "    cluster_" + std::to_string(i) + " -> cluster_" + std::to_string(i+1) + 
                         " [style=invis, weight=10];\n";
        }
    }

    // Conexión desde la entrada al primer bloque
    if (!layer_tensors.empty() && layer_tensors.count(0)) {
        dot_output += "    token_embd_weight -> blk_0_attn_norm_weight [lhead=cluster_0];\n";
    }

    // Conexión desde el último bloque a la salida
    if (!layer_tensors.empty()) {
        int last_layer = layer_tensors.rbegin()->first;
        if (last_layer != -1) {
            std::string last_block_output = "blk_" + std::to_string(last_layer) + "_block_output";
            dot_output += "    " + last_block_output + " -> output_norm_weight [ltail=cluster_" + 
                          std::to_string(last_layer) + "];\n";
        }
    }

    dot_output += "}\n";
    return dot_output;
}

bool save_dot_to_file_test(const std::string &dot_content, const std::string &filename)
{
    std::ofstream out_file(filename);
    if (!out_file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return false;
    }
    out_file << dot_content;
    out_file.close();
    return true;
}

// Función auxiliar para crear tensores
GGUFTensor create_tensor(const std::string &name, ggml_type type,
                         const std::vector<int64_t> &dims, ggml_op op,
                         const std::vector<std::string> &src_tensors = {})
{
    GGUFTensor tensor;
    tensor.name = name;
    tensor.type = type;
    tensor.n_dims = dims.size();
    tensor.dims = dims;
    tensor.op = op;
    tensor.src_tensors = src_tensors;
    return tensor;
}

GraphData create_sample_graph()
{
    GraphData graph;

    // ========== Input Tensors ==========
    // Input tensor (1D) - no weights or operation
    // 64 tokens, each is an integer index
    graph.tensors.push_back(create_tensor("tokens", GGML_TYPE_I32, {64}, GGML_OP_NONE));

    // ========== Embedding Layer ==========
    // Embedding tensor (contains weights internally)
    // Input: tokens (64,) → Output: (64, 512)
    GGUFTensor embeddings = create_tensor("embeddings", GGML_TYPE_F32, {64, 512}, GGML_OP_MUL_MAT, {"tokens"});
    // Embedding weights: vocab_size=32000, embedding_dim=512 → 32000*512 elements
    // embeddings.data = std::vector<float>(32000 * 512, 0.0f);
    graph.tensors.push_back(embeddings);

    // ========== Positional Encoding ==========
    // Positional encoding tensor (contains weights internally)
    // Input: embeddings (64,512) → Output: (64,512)
    GGUFTensor pos_emb = create_tensor("pos_embeddings", GGML_TYPE_F32, {64, 512}, GGML_OP_ADD_REL_POS, {"embeddings"});
    // Positional encoding weights: max_seq_len=1024, embedding_dim=512 → 1024*512 elements
    // pos_emb.data = std::vector<float>(1024 * 512, 0.0f);
    graph.tensors.push_back(pos_emb);

    // ========== Multi-Head Attention ==========
    // Q, K, V projections (parallel ops, each with internal weights)

    // Query projection
    // Input: pos_embeddings (64,512) → Output: (64,512)
    GGUFTensor q_proj = create_tensor("q_proj", GGML_TYPE_F32, {64, 512}, GGML_OP_MUL_MAT, {"pos_embeddings"});
    // Weights: embedding_dim=512 → 512*512 elements
    // q_proj.data = std::vector<float>(512 * 512, 0.0f);
    graph.tensors.push_back(q_proj);

    // Key projection
    GGUFTensor k_proj = create_tensor("k_proj", GGML_TYPE_F32, {64, 512}, GGML_OP_MUL_MAT, {"pos_embeddings"});
    // k_proj.data = std::vector<float>(512 * 512, 0.0f);
    graph.tensors.push_back(k_proj);

    // Value projection
    GGUFTensor v_proj = create_tensor("v_proj", GGML_TYPE_F32, {64, 512}, GGML_OP_MUL_MAT, {"pos_embeddings"});
    // v_proj.data = std::vector<float>(512 * 512, 0.0f);
    graph.tensors.push_back(v_proj);

    // Attention scores: Q*K^T
    // Input: q_proj (64,512), k_proj (64,512) → Output: (64,64)
    graph.tensors.push_back(create_tensor("scores", GGML_TYPE_F32, {64, 64}, GGML_OP_MUL_MAT, {"q_proj", "k_proj"}));

    // Softmax normalization
    // Input: scores (64,64) → Output: (64,64)
    graph.tensors.push_back(create_tensor("attn_weights", GGML_TYPE_F32, {64, 64}, GGML_OP_SOFT_MAX, {"scores"}));

    // Weighted sum: attention * V
    // Input: attn_weights (64,64), v_proj (64,512) → Output: (64,512)
    graph.tensors.push_back(create_tensor("attn_output", GGML_TYPE_F32, {64, 512}, GGML_OP_MUL_MAT, {"attn_weights", "v_proj"}));

    // ========== Feed-Forward Network ==========
    // First linear transformation
    // Input: attn_output (64,512) → Output: (64,2048)
    GGUFTensor ffn1 = create_tensor("ffn1", GGML_TYPE_F32, {64, 2048}, GGML_OP_MUL_MAT, {"attn_output"});
    // Weights: input_dim=512, hidden_dim=2048 → 512*2048 elements
    // ffn1.data = std::vector<float>(512 * 2048, 0.0f);
    graph.tensors.push_back(ffn1);

    // Bias add + ReLU activation
    // Input: ffn1 (64,2048) → Output: (64,2048)
    GGUFTensor ffn_relu = create_tensor("ffn_relu", GGML_TYPE_F32, {64, 2048}, GGML_OP_NORM, {"ffn1"});
    // Bias: hidden_dim=2048 → 2048 elements
    // ffn_relu.data = std::vector<float>(2048, 0.0f);
    graph.tensors.push_back(ffn_relu);

    // Second linear transformation
    // Input: ffn_relu (64,2048) → Output: (64,512)
    GGUFTensor ffn2 = create_tensor("ffn2", GGML_TYPE_F32, {64, 512}, GGML_OP_MUL_MAT, {"ffn_relu"});
    // Weights: hidden_dim=2048, output_dim=512 → 2048*512 elements
    // ffn2.data = std::vector<float>(2048 * 512, 0.0f);
    graph.tensors.push_back(ffn2);

    // ========== Output Layer ==========
    // Final projection to vocabulary size
    // Input: ffn2 (64,512) → Output: (64,32000)
    graph.tensors.push_back(create_tensor("logits", GGML_TYPE_F32, {64, 32000}, GGML_OP_MUL_MAT, {"ffn2"}));

    // Update header counts
    graph.header.n_tensors = graph.tensors.size();
    graph.header.n_kv = 0;

    return graph;
}

GraphData create_large_transformer_graph()
{
    GraphData graph;

    // ========== Input Tensor ==========
    // Input tokens (indices) - [batch_size, seq_len] = [1, 4096] as example
    graph.tensors.push_back(create_tensor("tokens", GGML_TYPE_I32, {1, 4096}, GGML_OP_NONE));

    // ========== Token Embeddings ==========
    // Embedding layer: [vocab_size, hidden_dim] = [32000, 4096]
    GGUFTensor token_embd = create_tensor("token_embd.weight", GGML_TYPE_Q2_K, {4096, 32000}, GGML_OP_MUL_MAT, {"tokens"});
    // token_embd.data = std::vector<float>(4096 * 32000, 0.0f); // Placeholder for weights
    graph.tensors.push_back(token_embd);

    // ========== Transformer Blocks (32 layers) ==========
    std::string prev_tensor = "token_embd.weight";

    //for (int i = 0; i < 32; ++i)
    for (int i = 0; i < 2; ++i)
    {
        std::string blk_prefix = "blk." + std::to_string(i) + ".";

        // Attention Norm - Operación de normalización
        GGUFTensor attn_norm = create_tensor(blk_prefix + "attn_norm.weight", GGML_TYPE_F32, {4096}, GGML_OP_NORM, {prev_tensor});
        graph.tensors.push_back(attn_norm);

        // Attention QKV Projections - Multiplicación de matrices
        GGUFTensor q_proj = create_tensor(blk_prefix + "attn_q.weight", GGML_TYPE_Q2_K, {4096, 4096}, GGML_OP_MUL_MAT, {blk_prefix + "attn_norm.weight"});
        graph.tensors.push_back(q_proj);

        GGUFTensor k_proj = create_tensor(blk_prefix + "attn_k.weight", GGML_TYPE_Q2_K, {4096, 4096}, GGML_OP_MUL_MAT, {blk_prefix + "attn_norm.weight"});
        graph.tensors.push_back(k_proj);

        GGUFTensor v_proj = create_tensor(blk_prefix + "attn_v.weight", GGML_TYPE_Q3_K, {4096, 4096}, GGML_OP_MUL_MAT, {blk_prefix + "attn_norm.weight"});
        graph.tensors.push_back(v_proj);

        // Attention Output Projection - Multiplicación de matrices
        // Nota: Para mantener solo los tensores especificados, asumimos que attn_output es el resultado final de atención
        GGUFTensor attn_output = create_tensor(blk_prefix + "attn_output.weight", GGML_TYPE_Q3_K, {4096, 4096}, GGML_OP_MUL_MAT,
                                               {blk_prefix + "attn_q.weight", blk_prefix + "attn_k.weight", blk_prefix + "attn_v.weight"});
        graph.tensors.push_back(attn_output);

        // FFN Norm - Operación de normalización
        GGUFTensor ffn_norm = create_tensor(blk_prefix + "ffn_norm.weight", GGML_TYPE_F32, {4096}, GGML_OP_NORM, {blk_prefix + "attn_output.weight"});
        graph.tensors.push_back(ffn_norm);

        // FFN Projections - Multiplicación de matrices
        GGUFTensor ffn_up = create_tensor(blk_prefix + "ffn_up.weight", GGML_TYPE_Q3_K, {4096, 11008}, GGML_OP_MUL_MAT, {blk_prefix + "ffn_norm.weight"});
        graph.tensors.push_back(ffn_up);

        GGUFTensor ffn_gate = create_tensor(blk_prefix + "ffn_gate.weight", GGML_TYPE_Q3_K, {4096, 11008}, GGML_OP_MUL_MAT, {blk_prefix + "ffn_norm.weight"});
        graph.tensors.push_back(ffn_gate);

        // FFN Down Projection - Multiplicación de matrices
        // Nota: Simplificamos la conexión directa a ffn_up y ffn_gate
        GGUFTensor ffn_down = create_tensor(blk_prefix + "ffn_down.weight", GGML_TYPE_Q3_K, {11008, 4096}, GGML_OP_MUL_MAT,
                                            {blk_prefix + "ffn_up.weight", blk_prefix + "ffn_gate.weight"});
        graph.tensors.push_back(ffn_down);

        // Salida del bloque - Suma residual
        prev_tensor = blk_prefix + "block_output";
        GGUFTensor block_output = create_tensor(prev_tensor, GGML_TYPE_F32, {4096, 4096}, GGML_OP_ADD,
                                                {blk_prefix + "attn_output.weight", blk_prefix + "ffn_down.weight"});
        graph.tensors.push_back(block_output);
    }
    // ========== Output Layer ==========
    // Final normalization
    GGUFTensor output_norm = create_tensor("output_norm.weight", GGML_TYPE_F32, {4096}, GGML_OP_NORM, {prev_tensor});
    // output_norm.data = std::vector<float>(4096, 0.0f);
    graph.tensors.push_back(output_norm);

    // Final projection to vocabulary
    GGUFTensor output_proj = create_tensor("output.weight", GGML_TYPE_Q6_K, {4096, 32000}, GGML_OP_MUL_MAT, {"output_norm.weight"});
    // output_proj.data = std::vector<float>(4096 * 32000, 0.0f);
    graph.tensors.push_back(output_proj);

    // Update header counts
    graph.header.n_tensors = graph.tensors.size();
    graph.header.n_kv = 0;

    return graph;
}

int main()
{
    // Crear el grafo de ejemplo
    GraphData sample_graph = create_large_transformer_graph();

    // Generar el gráfico DOT
    std::string dot_graph = generate_computational_graph_test(sample_graph);

    // Guardar en archivo
    save_dot_to_file_test(dot_graph, "sample_graph.dot");

    std::cout << "Grafo de ejemplo creado y guardado en sample_graph.dot" << std::endl;
    return 0;
}
