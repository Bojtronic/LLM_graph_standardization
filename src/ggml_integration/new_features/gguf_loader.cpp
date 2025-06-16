#include "gguf_loader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cctype>
#include <stdexcept>
#include <set>
#include <map>
#include <utility>
#include "arch_info.h"
#include "gguf.h"
#include "quantization_management.h"

/**
 * @brief Infers the operation type based on tensor name patterns
 * 
 * This function analyzes the tensor name to determine the most likely GGML operation
 * it represents (matrix multiplication, normalization, etc.) based on naming patterns.
 * 
 * @param tensor_name The name of the tensor to analyze
 * @return enum ggml_op The inferred GGML operation type
 */
enum ggml_op infer_operation(const std::string &tensor_name)
{
    // Patterns for attention
    if (tensor_name.find("attn_q.") != std::string::npos ||
        tensor_name.find("q_proj.") != std::string::npos ||
        tensor_name.find("attn_k.") != std::string::npos ||
        tensor_name.find("k_proj.") != std::string::npos ||
        tensor_name.find("attn_v.") != std::string::npos ||
        tensor_name.find("v_proj.") != std::string::npos ||
        tensor_name.find("attn_output.") != std::string::npos ||
        tensor_name.find("out_proj.") != std::string::npos)
    {
        return GGML_OP_MUL_MAT;
    }

    // Patterns for feed-forward
    if (tensor_name.find("ffn_up.") != std::string::npos ||
        tensor_name.find("ffn_down.") != std::string::npos ||
        tensor_name.find("ffn_gate.") != std::string::npos ||
        tensor_name.find("fc1.") != std::string::npos ||
        tensor_name.find("fc2.") != std::string::npos)
    {
        return GGML_OP_MUL_MAT;
    }

    // Patterns for embeddings
    if (tensor_name.find("token_embd.") != std::string::npos ||
        tensor_name.find("embed_tokens.") != std::string::npos ||
        tensor_name.find("position_embd.") != std::string::npos ||
        tensor_name.find("embed_positions.") != std::string::npos)
    {
        return GGML_OP_MUL_MAT;
    }

    // Patterns for normalization
    if (tensor_name.find("_norm.") != std::string::npos ||
        tensor_name.find("layer_norm.") != std::string::npos ||
        tensor_name.find("final_layer_norm.") != std::string::npos)
    {
        return GGML_OP_NORM;
    }

    // Default to matrix multiplication operation
    return GGML_OP_MUL_MAT;
}

/**
 * @brief Infers source tensors for a given tensor based on naming patterns
 * 
 * This function determines which other tensors are likely inputs (sources) for
 * the given tensor based on naming conventions and model architecture patterns.
 * 
 * @param tensor_name The name of the target tensor
 * @param tensors List of all available tensors in the model
 * @return std::vector<std::string> List of inferred source tensor names
 */
std::vector<std::string> infer_src_tensors(const std::string &tensor_name,
                                           const std::vector<GGUFTensor> &tensors)
{
    std::vector<std::string> src_tensors;

    try
    {
        // Extract block prefix (e.g. "blk.0.", "model.decoder.layers.0.")
        size_t block_end = tensor_name.find_last_of('.');
        std::string block_prefix = (block_end != std::string::npos) ? tensor_name.substr(0, block_end + 1) : "";

        // 1. For Q/K/V attention tensors
        if (tensor_name.find("attn_q.") != std::string::npos ||
            tensor_name.find("q_proj.") != std::string::npos ||
            tensor_name.find("attn_k.") != std::string::npos ||
            tensor_name.find("k_proj.") != std::string::npos ||
            tensor_name.find("attn_v.") != std::string::npos ||
            tensor_name.find("v_proj.") != std::string::npos)
        {
            // Find corresponding normalization tensor
            std::string norm_name = block_prefix + "attn_norm";
            for (const auto &t : tensors)
            {
                if (t.name.find(norm_name) != std::string::npos)
                {
                    src_tensors.push_back(t.name);
                    break;
                }
            }
        }
        // 2. For attention output tensors
        else if (tensor_name.find("attn_output.") != std::string::npos ||
                 tensor_name.find("out_proj.") != std::string::npos)
        {
            src_tensors.push_back(block_prefix + "attn_q");
            src_tensors.push_back(block_prefix + "attn_k");
            src_tensors.push_back(block_prefix + "attn_v");
        }
        // 3. For feed-forward tensors
        else if (tensor_name.find("ffn_down.") != std::string::npos ||
                 tensor_name.find("fc2.") != std::string::npos)
        {
            src_tensors.push_back(block_prefix + "ffn_up");
        }
        else if (tensor_name.find("ffn_up.") != std::string::npos ||
                 tensor_name.find("fc1.") != std::string::npos)
        {
            // Find FFN normalization tensor
            std::string norm_name = block_prefix + "ffn_norm";
            for (const auto &t : tensors)
            {
                if (t.name.find(norm_name) != std::string::npos)
                {
                    src_tensors.push_back(t.name);
                    break;
                }
            }
        }
        // 4. For normalization layers
        else if (tensor_name.find("_norm.") != std::string::npos)
        {
            if (!block_prefix.empty())
            {
                // Safely extract block number
                auto extract_block_num = [](const std::string &s) -> int
                {
                    try
                    {
                        size_t last_dot = s.find_last_of('.', s.length() - 2);
                        if (last_dot == std::string::npos)
                            return -1;

                        size_t prev_dot = s.find_last_of('.', last_dot - 1);
                        if (prev_dot == std::string::npos)
                            return -1;

                        std::string num_str = s.substr(prev_dot + 1, last_dot - prev_dot - 1);
                        if (num_str.empty() || !std::all_of(num_str.begin(), num_str.end(), ::isdigit))
                        {
                            return -1;
                        }
                        return std::stoi(num_str);
                    }
                    catch (...)
                    {
                        return -1;
                    }
                };

                int block_num = extract_block_num(block_prefix);
                if (block_num > 0)
                {
                    // Build previous block name
                    size_t block_start = block_prefix.find_last_of('.', block_prefix.length() - 2);
                    if (block_start != std::string::npos)
                    {
                        std::string prev_block = block_prefix.substr(0, block_start + 1) +
                                                 std::to_string(block_num - 1) + ".";
                        src_tensors.push_back(prev_block + "layer_output");
                    }
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error processing tensor '" << tensor_name << "': " << e.what() << std::endl;
    }

    // Remove possible duplicates
    std::sort(src_tensors.begin(), src_tensors.end());
    src_tensors.erase(std::unique(src_tensors.begin(), src_tensors.end()), src_tensors.end());

    return src_tensors;
}

/**
 * @brief Infers the destination tensor name based on the given tensor name
 * 
 * This function determines the most likely destination tensor for a given source tensor
 * by analyzing naming patterns and common transformer architecture conventions.
 * 
 * @param tensor_name The name of the source tensor to analyze
 * @return std::string The inferred destination tensor name, or empty string if cannot be determined
 */
std::string infer_dst_tensor(const std::string &tensor_name)
{
    // Extract the block prefix (e.g., "blk.3." or "layers.5.")
    size_t block_end = tensor_name.find_last_of('.');
    std::string block_prefix = (block_end != std::string::npos) ? tensor_name.substr(0, block_end + 1) : "";

    // Handle attention query/key/value projection tensors
    if (tensor_name.find("attn_q.") != std::string::npos ||
        tensor_name.find("q_proj.") != std::string::npos ||
        tensor_name.find("attn_k.") != std::string::npos ||
        tensor_name.find("k_proj.") != std::string::npos ||
        tensor_name.find("attn_v.") != std::string::npos ||
        tensor_name.find("v_proj.") != std::string::npos)
    {
        // These projections feed into the attention output
        return block_prefix + "attn_output";
    }
    // Handle feed-forward network up-projection or first fully-connected layer
    else if (tensor_name.find("ffn_up.") != std::string::npos ||
             tensor_name.find("fc1.") != std::string::npos)
    {
        // Up projection feeds into down projection
        return block_prefix + "ffn_down";
    }
    // Handle feed-forward network down-projection or second fully-connected layer
    else if (tensor_name.find("ffn_down.") != std::string::npos ||
             tensor_name.find("fc2.") != std::string::npos)
    {
        // Down projection feeds into the final layer output
        return block_prefix + "layer_output";
    }
    // Handle normalization layers
    else if (tensor_name.find("_norm.") != std::string::npos)
    {
        // Attention normalization feeds into query projection
        if (tensor_name.find("attn_norm.") != std::string::npos)
        {
            return block_prefix + "attn_q";
        }
        // FFN normalization feeds into up projection
        else if (tensor_name.find("ffn_norm.") != std::string::npos)
        {
            return block_prefix + "ffn_up";
        }
        // Other normalizations feed into layer output
        else
        {
            return block_prefix + "layer_output";
        }
    }

    return ""; // Could not infer destination tensor
}

/**
 * @brief Parses a GGUF context and stores all data in a GraphData structure
 * @param ctx Loaded GGUF context
 * @param fname GGUF filename (for reading tensor data)
 * @return GraphData structure with all loaded data
 */
GraphData gguf_graph_data(const struct gguf_context *ctx, const char *fname)
{
    GraphData graph_data;

    if (!ctx)
    {
        std::cerr << "Contexto GGUF es NULL." << std::endl;
        return graph_data;
    }

    // Fill in the header
    graph_data.header.n_tensors = gguf_get_n_tensors(ctx);
    graph_data.header.n_kv = gguf_get_n_kv(ctx);

    // Fill in metadata
    for (int64_t i = 0; i < gguf_get_n_kv(ctx); ++i)
    {
        GGUFMetadata md;
        md.key = gguf_get_key(ctx, i);
        md.type = gguf_get_kv_type(ctx, i);

        switch (md.type)
        {
        case GGUF_TYPE_UINT8:
            md.value.u8 = gguf_get_val_u8(ctx, i);
            break;
        case GGUF_TYPE_INT8:
            md.value.i8 = gguf_get_val_i8(ctx, i);
            break;
        case GGUF_TYPE_UINT16:
            md.value.u16 = gguf_get_val_u16(ctx, i);
            break;
        case GGUF_TYPE_INT16:
            md.value.i16 = gguf_get_val_i16(ctx, i);
            break;
        case GGUF_TYPE_UINT32:
            md.value.u32 = gguf_get_val_u32(ctx, i);
            break;
        case GGUF_TYPE_INT32:
            md.value.i32 = gguf_get_val_i32(ctx, i);
            break;
        case GGUF_TYPE_FLOAT32:
            md.value.f32 = gguf_get_val_f32(ctx, i);
            break;
        case GGUF_TYPE_BOOL:
            md.value.b = gguf_get_val_bool(ctx, i);
            break;
        case GGUF_TYPE_STRING:
            md.str = gguf_get_val_str(ctx, i);
            break;
        case GGUF_TYPE_UINT64:
            md.value.u64 = gguf_get_val_u64(ctx, i);
            break;
        case GGUF_TYPE_INT64:
            md.value.i64 = gguf_get_val_i64(ctx, i);
            break;
        case GGUF_TYPE_FLOAT64:
            md.value.f64 = gguf_get_val_f64(ctx, i);
            break;
        case GGUF_TYPE_ARRAY:
            md.array.type = gguf_get_arr_type(ctx, i);
            md.array.size = gguf_get_arr_n(ctx, i);

            switch (md.array.type)
            {
            case GGUF_TYPE_UINT8:
                md.array.data = read_array_data<uint8_t>(ctx, i, md.array.size);
                break;
            case GGUF_TYPE_INT8:
                md.array.data = read_array_data<int8_t>(ctx, i, md.array.size);
                break;
            case GGUF_TYPE_UINT16:
                md.array.data = read_array_data<uint16_t>(ctx, i, md.array.size);
                break;
            case GGUF_TYPE_INT16:
                md.array.data = read_array_data<int16_t>(ctx, i, md.array.size);
                break;
            case GGUF_TYPE_UINT32:
                md.array.data = read_array_data<uint32_t>(ctx, i, md.array.size);
                break;
            case GGUF_TYPE_INT32:
                md.array.data = read_array_data<int32_t>(ctx, i, md.array.size);
                break;
            case GGUF_TYPE_FLOAT32:
                md.array.data = read_array_data<float>(ctx, i, md.array.size);
                break;
            case GGUF_TYPE_UINT64:
                md.array.data = read_array_data<uint64_t>(ctx, i, md.array.size);
                break;
            case GGUF_TYPE_INT64:
                md.array.data = read_array_data<int64_t>(ctx, i, md.array.size);
                break;
            case GGUF_TYPE_FLOAT64:
                md.array.data = read_array_data<double>(ctx, i, md.array.size);
                break;
            case GGUF_TYPE_STRING:

            {
                std::vector<std::string> strings;
                strings.reserve(md.array.size);
                for (size_t j = 0; j < md.array.size; ++j) {
                    const char* str = gguf_get_arr_str(ctx, i, j);
                    strings.emplace_back(str ? str : "");
                }
                md.array.data = strings;
                
                break;
            }

            break;

            default:
                break;
            }
            break;
        default:
            break;
        }

        graph_data.metadata.push_back(md);
    }

    // Fill in tensor information
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i)
    //for (int64_t i = 0; i < 15; ++i)
    {
        GGUFTensor tensor;
        tensor.name = gguf_get_tensor_name(ctx, i);
        tensor.type = gguf_get_tensor_type(ctx, i);
        tensor.size = gguf_get_tensor_size(ctx, i);

        // Get dimensions
        const int32_t n_dims = gguf_get_tensor_n_dims(ctx, i);
        const int64_t *dims = gguf_get_tensor_dims(ctx, i);
        tensor.dims.assign(dims, dims + n_dims);
        tensor.n_dims = n_dims;

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Read data from the tensor
        std::ifstream file(fname, std::ios::binary);
        if (file)
        {
            size_t offset = gguf_get_tensor_offset(ctx, i);
            file.seekg(offset, std::ios::beg);

            // Assign the correct storage type
            if (!ggml_is_quantized(tensor.type))
            {
                switch (tensor.type)
                {
                case GGML_TYPE_F32:
                {
                    std::vector<float> float_data(tensor.size / sizeof(float));
                    file.read(reinterpret_cast<char *>(float_data.data()), tensor.size);
                    tensor.data = float_data;
                    break;
                }
                case GGML_TYPE_F16:
                {
                    std::vector<uint16_t> f16_data(tensor.size / sizeof(uint16_t));
                    file.read(reinterpret_cast<char *>(f16_data.data()), tensor.size);
                    tensor.data = f16_data;
                    break;
                }
                case GGML_TYPE_I32:
                {
                    std::vector<int32_t> i32_data(tensor.size / sizeof(int32_t));
                    file.read(reinterpret_cast<char *>(i32_data.data()), tensor.size);
                    tensor.data = i32_data;
                    break;
                }
                case GGML_TYPE_I16:
                {
                    std::vector<int16_t> i16_data(tensor.size / sizeof(int16_t));
                    file.read(reinterpret_cast<char *>(i16_data.data()), tensor.size);
                    tensor.data = i16_data;
                    break;
                }
                case GGML_TYPE_I8:
                {
                    std::vector<int8_t> i8_data(tensor.size / sizeof(int8_t));
                    file.read(reinterpret_cast<char *>(i8_data.data()), tensor.size);
                    tensor.data = i8_data;
                    break;
                }
                default:
                {
                    std::vector<uint8_t> raw_data(tensor.size);
                    file.read(reinterpret_cast<char *>(raw_data.data()), tensor.size);
                    tensor.data = raw_data;
                    break;
                }
                }
            }
            else
            {
                // Determine the block size according to the quantization type
                size_t block_size = 0;
                size_t values_per_block = 0;  // Dequantized securities per block
                switch (tensor.type) {
                    case GGML_TYPE_Q2_K:
                        block_size = sizeof(block_q2_k);
                        values_per_block = 256;  // Each Q2_K block contains 256 values
                        break;
                    case GGML_TYPE_Q3_K:
                        block_size = sizeof(block_q3_k);
                        values_per_block = 256;  // Each Q3_K block contains 256 values
                        break;
                    case GGML_TYPE_Q4_K:
                        block_size = sizeof(block_q4_k);
                        values_per_block = 256;
                        break;
                    case GGML_TYPE_Q5_K:
                        block_size = sizeof(block_q5_k);
                        values_per_block = 256;
                        break;
                    case GGML_TYPE_Q6_K:
                        block_size = sizeof(block_q6_k);
                        values_per_block = 256;
                        break;
                    case GGML_TYPE_Q8_K:
                        block_size = sizeof(block_q8_k);
                        values_per_block = 256;
                        break;
                    default:
                        throw std::runtime_error("Tipo de cuantización no soportado");
                }
                 
                // Calculate the number of blocks in the tensor
                size_t num_blocks = tensor.size / block_size;
                std::vector<uint8_t> quant_data(tensor.size);
                file.read(reinterpret_cast<char*>(quant_data.data()), tensor.size);

                // Dequantize to float
                std::vector<float> float_data;
                float_data.resize(num_blocks * values_per_block);  // Total dequantized values

                // Call the correct dequantization function (QX_K quantization only)
                dequantize_k_quant(tensor.type, quant_data.data(), float_data.data(), num_blocks * values_per_block);

                // Store dequantized data in the tensor
                tensor.data = float_data;
                            
            }
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////


        // Check that we do not load the data (for testing)
        //tensor.data = std::vector<uint8_t>();

        // Infer operation and connections
        tensor.op = infer_operation(tensor.name);
        tensor.src_tensors = infer_src_tensors(tensor.name, graph_data.tensors);
        tensor.dst_tensor = infer_dst_tensor(tensor.name);

        graph_data.tensors.push_back(tensor);
    }

    return graph_data;
}

/**
 * @brief Generates a DOT graph representation of the GGUF model architecture
 * 
 * This function creates a Graphviz DOT format string that visualizes the tensor
 * operations and their connections in the GGUF model. Each tensor is represented
 * as a node with operation type and dimensions, and connections show data flow.
 * 
 * @param graph_data The GraphData structure containing tensor information
 * @return std::string The DOT format graph as a string
 */
std::string generate_dot_graph(const GraphData &graph_data)
{
    std::ostringstream dot;

    // DOT file header
    dot << "digraph GGUF_Graph {\n";
    dot << "  rankdir=LR;\n";
    dot << "  node [shape=box, style=filled, fillcolor=\"#f0f0f0\", fontname=\"Helvetica\"];\n";
    dot << "  edge [fontname=\"Helvetica\", fontsize=10];\n\n";

    // Add nodes (tensors)
    for (const auto &tensor : graph_data.tensors)
    {
        std::string node_name = tensor.name;

        // Get operation name
        std::string op_str;
        switch (tensor.op)
        {
        case GGML_OP_MUL_MAT:
            op_str = "MUL_MAT";
            break;
        case GGML_OP_NORM:
            op_str = "NORM";
            break;
        case GGML_OP_SOFT_MAX:
            op_str = "SOFT_MAX";
            break;
        default:
            op_str = "OTHER";
            break;
        }

        // Create label with name, operation and dimensions
        std::string label = tensor.name + "\\n" +
                            "Op: " + op_str + "\\n" +
                            "Dims: [";

        for (size_t i = 0; i < tensor.dims.size(); ++i)
        {
            if (i > 0)
                label += ", ";
            label += std::to_string(tensor.dims[i]);
        }
        label += "]";

        // Different color based on operation type
        std::string color;
        switch (tensor.op)
        {
        case GGML_OP_MUL_MAT:
            color = "#d4f1f9"; // Light blue
            break;
        case GGML_OP_NORM:
            color = "#d5e8d4"; // Light green
            break;
        case GGML_OP_SOFT_MAX:
            color = "#f8cecc"; // Light red
            break;
        default:
            color = "#f0f0f0"; // Light gray
        }

        dot << "  \"" << node_name << "\" [label=\"" << label << "\", fillcolor=\"" << color << "\"];\n";
    }

    dot << "\n";

    // Add connections (edges)
    for (const auto &tensor : graph_data.tensors)
    {
        // Connections from source tensors
        for (const auto &src : tensor.src_tensors)
        {
            dot << "  \"" << src << "\" -> \"" << tensor.name << "\";\n";
        }

        // Connection to destination tensor (if exists)
        if (!tensor.dst_tensor.empty())
        {
            dot << "  \"" << tensor.name << "\" -> \"" << tensor.dst_tensor << "\";\n";
        }
    }

    dot << "}\n";

    return dot.str();
}


/**
 * @brief Saves the DOT graph representation to a file
 * 
 * This function generates a DOT format graph using the provided GraphData
 * and saves it to the specified file. The graph visualizes the tensor
 * operations and connections in the GGUF model.
 * 
 * @param graph_data The GraphData structure containing tensor information
 * @param filename The path to the output file where the DOT graph will be saved
 * @return true if the file was successfully saved, false otherwise
 */
bool save_dot_graph(const GraphData &graph_data, const std::string &filename)
{
    // Attempt to open the output file
    std::ofstream out_file(filename);
    if (!out_file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    // Generate the DOT graph content
    std::string dot_content = generate_dot_graph(graph_data);
    
    // Write the content to file
    out_file << dot_content;
    out_file.close();

    return true;
}

/**
 * @brief Main function to load a GGUF file and retrieve its configuration
 * @param fname Name of the GGUF file
 * @return true if loading was successful, false otherwise
 */
bool get_gguf_config(const char *fname)
{

    if (!fname)
    {
        std::cerr << "Nombre de archivo inválido (NULL)\n";
        return false;
    }

    struct ggml_context *ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    struct gguf_context *ctx_gguf = gguf_init_from_file(fname, params);
    if (!ctx_gguf)
    {
        std::cerr << "No se pudo cargar el archivo GGUF '" << fname << "'\n";
        return false;
    }

    GraphData graph_data = gguf_graph_data(ctx_gguf, fname);
    save_dot_graph(graph_data, "graph.dot");

    gguf_free(ctx_gguf);

    return true;
}
