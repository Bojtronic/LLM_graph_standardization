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
#include "arch_info.h"
//#include "quantization_management.h"
#include <regex>
//#include "ggml-common.h"
//#include "ggml.h"
//#include "ggml-impl.h"
#include "ggml-quants.h" // falta utilizar los recursos de este archivo, se estuvo utilizando la implementación propia de quantization_management.h


/**
 * @brief Infers the operation type based on tensor name patterns
 *
 * This function analyzes the tensor name to determine the most likely GGML operation
 * it represents (matrix multiplication, normalization, etc.) based on naming patterns.
 *
 * @param tensor_name The name of the tensor to analyze
 * @param arch The model architecture
 * @return enum ggml_op The inferred GGML operation type
 */
enum ggml_op infer_operation(const std::string &tensor_name, llm_arch arch) {
    llm_tensor tensor;

    if (get_tensor_by_name(tensor_name, arch, tensor)) {
        return get_tensor_operation(tensor);
    }
    return GGML_OP_NONE;
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
std::vector<std::string> infer_src_tensors(const std::string& tensor_name, llm_arch arch) {
    std::vector<std::string> src_tensors;
    llm_tensor current_tensor;
    
    if (!get_tensor_by_name(tensor_name, arch, current_tensor)) {
        return src_tensors;
    }

    // Extraer prefijo de bloque para tensores en bloques
    std::string block_prefix = "";
    if (tensor_name.find("blk.") != std::string::npos) {
        size_t blk_end = tensor_name.find(".", 4); // Después de "blk.X"
        if (blk_end != std::string::npos) {
            block_prefix = tensor_name.substr(0, blk_end + 1);
        }
    }

    switch(current_tensor) {
        // Capa de embedding
        case LLM_TENSOR_TOKEN_EMBD_NORM:
            src_tensors.push_back("token_embd.weight");
            break;
            
        // Atención dentro de bloques
        case LLM_TENSOR_ATTN_Q:
        case LLM_TENSOR_ATTN_K:
        case LLM_TENSOR_ATTN_V:
            src_tensors.push_back(block_prefix + "attn_norm.weight");
            break;
            
        case LLM_TENSOR_ATTN_OUT:
            src_tensors.push_back(block_prefix + "attn_q.weight");
            src_tensors.push_back(block_prefix + "attn_k.weight");
            src_tensors.push_back(block_prefix + "attn_v.weight");
            break;
            
        // Feed-Forward Network dentro de bloques
        case LLM_TENSOR_FFN_NORM:
            src_tensors.push_back(block_prefix + "attn_output.weight");
            break;
            
        case LLM_TENSOR_FFN_UP:
            src_tensors.push_back(block_prefix + "ffn_norm.weight");
            break;
            
        case LLM_TENSOR_FFN_GATE:
        case LLM_TENSOR_FFN_DOWN:
            src_tensors.push_back(block_prefix + "ffn_up.weight");
            break;
            
        // Capa de salida
        case LLM_TENSOR_OUTPUT_NORM:
            // Tomamos el último bloque (asumiendo blk.31)
            src_tensors.push_back("blk.31.ffn_down.weight");
            break;
            
        case LLM_TENSOR_OUTPUT:
            src_tensors.push_back("output_norm.weight");
            break;
            
        default:
            break;
    }
    
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
std::string infer_dst_tensor(const std::string& tensor_name, llm_arch arch) {
    llm_tensor current_tensor;
    
    if (!get_tensor_by_name(tensor_name, arch, current_tensor)) {
        return "";
    }

    // Extraer prefijo de bloque para tensores en bloques
    std::string block_prefix = "";
    if (tensor_name.find("blk.") != std::string::npos) {
        size_t blk_end = tensor_name.find(".", 4); // Después de "blk.X"
        if (blk_end != std::string::npos) {
            block_prefix = tensor_name.substr(0, blk_end + 1);
        }
    }

    switch(current_tensor) {
        // Capa de embedding
        case LLM_TENSOR_TOKEN_EMBD:
            return "token_embd_norm.weight";
            
        // Atención dentro de bloques
        case LLM_TENSOR_ATTN_NORM:
            return block_prefix + "attn_q.weight";
            
        case LLM_TENSOR_ATTN_Q:
        case LLM_TENSOR_ATTN_K:
        case LLM_TENSOR_ATTN_V:
            return block_prefix + "attn_output.weight";
            
        case LLM_TENSOR_ATTN_OUT:
            return block_prefix + "ffn_norm.weight";
            
        // Feed-Forward Network dentro de bloques
        case LLM_TENSOR_FFN_NORM:
            return block_prefix + "ffn_up.weight";
            
        case LLM_TENSOR_FFN_UP:
            return block_prefix + "ffn_gate.weight";
            
        case LLM_TENSOR_FFN_GATE:
        case LLM_TENSOR_FFN_DOWN:
            // Si es el último bloque (blk.31), va a output_norm
            if (block_prefix == "blk.31.") {
                return "output_norm.weight";
            }
            // Para otros bloques, iría al siguiente bloque (pero no lo implementamos aquí)
            return "";
            
        // Capa de salida
        case LLM_TENSOR_OUTPUT_NORM:
            return "output.weight";
            
        default:
            return "";
    }
}

/**
 * @brief Parses a GGUF context and stores all data in a GraphData structure
 * @param ctx Loaded GGUF context
 * @param file_gguf_name GGUF filename (for reading tensor data)
 * @return GraphData structure with all loaded data
 */

// crear el archivo.graph primero sin decuantizar los datos
// a la estructura GraphData no incluirle los datos de los tensores solo el nombre, tipo, dimensiones y operacion (de tensores y/o unaria)

GraphData gguf_graph_data(const struct gguf_context *ctx, const char *file_gguf, const char *file_graph)
{
    GraphData graph_data;
    std::string architecture = "";

    if (!ctx)
    {
        std::cerr << "Contexto GGUF es NULL." << std::endl;
        return graph_data;
    }

    std::ofstream out(file_graph, std::ios::binary);
    if (!out.is_open())
    {
        std::cerr << "Failed to open " << file_graph << " for writing\n";
    }

    // 1. Identification of the file
    const std::string magic = "GRAPH";
    out.write(magic.c_str(), magic.size());
    out.put('\n');

    // 2. Header
    uint64_t n_tensors = gguf_get_n_tensors(ctx);
    graph_data.header.n_tensors = n_tensors;
    out.write(reinterpret_cast<const char *>(&n_tensors), sizeof(uint64_t));
    out.put('\n');

    uint64_t n_kv = gguf_get_n_kv(ctx);
    graph_data.header.n_kv = n_kv;
    out.write(reinterpret_cast<const char *>(&n_kv), sizeof(uint64_t));
    out.put('\n');

    // 3. Metadata
    const std::string metadata_marker = "---METADATA---";

    out.write(metadata_marker.c_str(), metadata_marker.size());
    out.put('\n');

    for (int64_t i = 0; i < n_kv; ++i)
    {
        GGUFMetadata md;
        md.key = gguf_get_key(ctx, i);
        md.type = gguf_get_kv_type(ctx, i);

        // Key
        out << md.key;
        out.put('\n');

        // Type
        out.write(reinterpret_cast<const char *>(&md.type), sizeof(enum gguf_type));
        out.put('\n');

        // Value
        switch (md.type)
        {
        case GGUF_TYPE_UINT8:
            md.value.u8 = gguf_get_val_u8(ctx, i);
            out << static_cast<int>(md.value.u8);
            out.put('\n');
            break;
        case GGUF_TYPE_INT8:
            md.value.i8 = gguf_get_val_i8(ctx, i);
            out << static_cast<int>(md.value.i8);
            out.put('\n');
            break;
        case GGUF_TYPE_UINT16:
            md.value.u16 = gguf_get_val_u16(ctx, i);
            out << md.value.u16;
            out.put('\n');
            break;
        case GGUF_TYPE_INT16:
            md.value.i16 = gguf_get_val_i16(ctx, i);
            out << md.value.i16;
            out.put('\n');
            break;
        case GGUF_TYPE_UINT32:
            md.value.u32 = gguf_get_val_u32(ctx, i);
            out << md.value.u32;
            out.put('\n');
            break;
        case GGUF_TYPE_INT32:
            md.value.i32 = gguf_get_val_i32(ctx, i);
            out << md.value.i32;
            out.put('\n');
            break;
        case GGUF_TYPE_FLOAT32:
            md.value.f32 = gguf_get_val_f32(ctx, i);
            out << std::fixed << std::setprecision(6) << md.value.f32;
            out.put('\n');
            break;
        case GGUF_TYPE_BOOL:
            md.value.b = gguf_get_val_bool(ctx, i);
            out << (md.value.b ? "true" : "false");
            out.put('\n');
            break;
        case GGUF_TYPE_STRING:
            md.str = gguf_get_val_str(ctx, i);
            out << md.str;
            out.put('\n');
            break;
        case GGUF_TYPE_UINT64:
            md.value.u64 = gguf_get_val_u64(ctx, i);
            out << md.value.u64;
            out.put('\n');
            break;
        case GGUF_TYPE_INT64:
            md.value.i64 = gguf_get_val_i64(ctx, i);
            out << md.value.i64;
            out.put('\n');
            break;
        case GGUF_TYPE_FLOAT64:
            md.value.f64 = gguf_get_val_f64(ctx, i);
            out << std::fixed << std::setprecision(6) << md.value.f64;
            out.put('\n');
            break;
        case GGUF_TYPE_ARRAY:
            md.array.type = gguf_get_arr_type(ctx, i);
            md.array.size = gguf_get_arr_n(ctx, i);

            out.write(reinterpret_cast<const char *>(&md.array.type), sizeof(enum gguf_type));
            out.put('\n');
            out.write(reinterpret_cast<const char *>(&md.array.size), sizeof(size_t));
            out.put('\n');

            switch (md.array.type)
            {
            case GGUF_TYPE_UINT8:
                md.array.data = read_array_data<uint8_t>(ctx, i, md.array.size);
                out.write(reinterpret_cast<const char *>(std::get<std::vector<uint8_t>>(md.array.data).data()),
                          md.array.size * sizeof(uint8_t));
                out.put('\n');
                break;
            case GGUF_TYPE_INT8:
                md.array.data = read_array_data<int8_t>(ctx, i, md.array.size);
                out.write(reinterpret_cast<const char *>(std::get<std::vector<int8_t>>(md.array.data).data()),
                          md.array.size * sizeof(int8_t));
                out.put('\n');
                break;
            case GGUF_TYPE_UINT16:
                md.array.data = read_array_data<uint16_t>(ctx, i, md.array.size);
                out.write(reinterpret_cast<const char *>(std::get<std::vector<uint16_t>>(md.array.data).data()),
                          md.array.size * sizeof(uint16_t));
                out.put('\n');
                break;
            case GGUF_TYPE_INT16:
                md.array.data = read_array_data<int16_t>(ctx, i, md.array.size);
                out.write(reinterpret_cast<const char *>(std::get<std::vector<int16_t>>(md.array.data).data()),
                          md.array.size * sizeof(int16_t));
                out.put('\n');
                break;
            case GGUF_TYPE_UINT32:
                md.array.data = read_array_data<uint32_t>(ctx, i, md.array.size);
                out.write(reinterpret_cast<const char *>(std::get<std::vector<uint32_t>>(md.array.data).data()),
                          md.array.size * sizeof(uint32_t));
                out.put('\n');
                break;
            case GGUF_TYPE_INT32:
                md.array.data = read_array_data<int32_t>(ctx, i, md.array.size);
                out.write(reinterpret_cast<const char *>(std::get<std::vector<int32_t>>(md.array.data).data()),
                          md.array.size * sizeof(int32_t));
                out.put('\n');
                break;
            case GGUF_TYPE_FLOAT32:
                md.array.data = read_array_data<float>(ctx, i, md.array.size);
                out.write(reinterpret_cast<const char *>(std::get<std::vector<float>>(md.array.data).data()),
                          md.array.size * sizeof(float));
                out.put('\n');
                break;
            case GGUF_TYPE_UINT64:
                md.array.data = read_array_data<uint64_t>(ctx, i, md.array.size);
                out.write(reinterpret_cast<const char *>(std::get<std::vector<uint64_t>>(md.array.data).data()),
                          md.array.size * sizeof(uint64_t));
                out.put('\n');
                break;
            case GGUF_TYPE_INT64:
                md.array.data = read_array_data<int64_t>(ctx, i, md.array.size);
                out.write(reinterpret_cast<const char *>(std::get<std::vector<int64_t>>(md.array.data).data()),
                          md.array.size * sizeof(int64_t));
                out.put('\n');
                break;
            case GGUF_TYPE_FLOAT64:
                md.array.data = read_array_data<double>(ctx, i, md.array.size);
                out.write(reinterpret_cast<const char *>(std::get<std::vector<double>>(md.array.data).data()),
                          md.array.size * sizeof(double));
                out.put('\n');
                break;
            case GGUF_TYPE_STRING:
            {
                std::vector<std::string> strings;
                strings.reserve(md.array.size);
                for (size_t j = 0; j < md.array.size; ++j)
                {
                    const char *str = gguf_get_arr_str(ctx, i, j);
                    std::string safe_str = str ? str : "";
                    
                    // Codificar el string para evitar caracteres problemáticos
                    std::string encoded_str;
                    for (char c : safe_str) {
                        if (c == '\n') encoded_str += "\\n";
                        else if (c == '\0') encoded_str += "\\0";
                        else if (c == '\\') encoded_str += "\\\\";
                        else encoded_str += c;
                    }
                    
                    strings.emplace_back(safe_str);
                    
                    // Escribir tamaño primero, luego los datos
                    uint32_t len = encoded_str.size();
                    out.write(reinterpret_cast<const char*>(&len), sizeof(uint32_t));
                    out.write(encoded_str.c_str(), len);
                }
                out.put('\n'); // newline al final del array
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
        
        if(md.key == "general.architecture"){
            architecture = md.str;
        }
        graph_data.metadata.push_back(md);

        const std::string item_end_marker = "---END_ITEM---";
        out.write(item_end_marker.c_str(), item_end_marker.size());
        out.put('\n');
    }

    // 4. Tensors
    const std::string tensors_marker = "---TENSORS---";
    out.write(tensors_marker.c_str(), tensors_marker.size());
    out.put('\n');

    uint64_t tensors_count = n_tensors;
    out.write(reinterpret_cast<const char *>(&tensors_count), sizeof(uint64_t));
    out.put('\n');

    for (int64_t i = 0; i < n_tensors; ++i)
    // for (int64_t i = 0; i < 15; ++i)
    {
        // Start delimiter
        out << "---BEGIN_TENSOR---";
        out.put('\n');

        GGUFTensor tensor;
        tensor.name = gguf_get_tensor_name(ctx, i);
        out << "NAME: " << tensor.name;
        out.put('\n');

        tensor.type = gguf_get_tensor_type(ctx, i);
        out << "TYPE: " << tensor.type;
        out.put('\n');

        tensor.size = gguf_get_tensor_size(ctx, i);
        out << "SIZE: " << tensor.size;
        out.put('\n');

        // Get dimensions
        const int32_t n_dims = gguf_get_tensor_n_dims(ctx, i);
        const int64_t *dims = gguf_get_tensor_dims(ctx, i);
        tensor.dims.assign(dims, dims + n_dims);
        tensor.n_dims = n_dims;

        out << "NDIMS: " << tensor.n_dims;
        out.put('\n');

        out << "DIMS: ";
        if (!tensor.dims.empty())
        {
            // Write all dimensions except the last one
            for (size_t j = 0; j < tensor.dims.size() - 1; ++j)
            {
                out << tensor.dims[j] << ',';
            }
            // Write the last dimension without a comma
            out << tensor.dims.back();
        }
        out.put('\n');

        // Infer operation and connections
        llm_arch arch = llm_arch_from_string(architecture);
        tensor.op = infer_operation(tensor.name, arch);

        //out << "OPERATION: " << tensor.op;
        const std::string op_prefix = "OPERATION: ";
        std::string op_str = std::to_string(static_cast<int>(tensor.op));
        out.write(op_str.c_str(), op_str.size());
        out.put('\n');

        tensor.src_tensors = infer_src_tensors(tensor.name, arch);

        //std::string src_tensors_str;

        
        //out << "SOURCE: ";
        const std::string source_prefix = "SOURCE: ";
        out.write(source_prefix.c_str(), source_prefix.size());
        if (!tensor.src_tensors.empty()) {
            // Escribe todos los tensores excepto el último
            for (size_t j = 0; j < tensor.src_tensors.size() - 1; ++j) {
                out << tensor.src_tensors[j] << ',';
            }
            // Escribe el último tensor sin coma
            out << tensor.src_tensors.back();
        }
        out.put('\n');

        tensor.dst_tensor = infer_dst_tensor(tensor.name, arch);

        out << "DESTINATION: " << tensor.dst_tensor;
        out.put('\n');

        
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        out << "DATA_START:";
        out.put('\n');

        // Read data from the tensor
        std::ifstream file(file_gguf, std::ios::binary);
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
                        const size_t num_elements = tensor.size / sizeof(float);
                        std::vector<float> float_data(num_elements);

                        file.read(reinterpret_cast<char *>(float_data.data()), tensor.size);

                        // Verify that all bytes were read
                        // if(file.gcount() != static_cast<std::streamsize>(tensor.size)) {
                        //    throw std::runtime_error("Error reading data from tensor");
                        //}

                        //tensor.data = float_data;

                        out.write(reinterpret_cast<const char *>(float_data.data()), tensor.size);
                        break;
                    }
                    case GGML_TYPE_F16:
                    {
                        const size_t num_elements = tensor.size / sizeof(uint16_t);
                        std::vector<uint16_t> f16_data(num_elements);

                        file.read(reinterpret_cast<char *>(f16_data.data()), tensor.size);

                        //tensor.data = f16_data;

                        out.write(reinterpret_cast<const char *>(f16_data.data()), tensor.size);
                        break;
                    }
                    case GGML_TYPE_I32:
                    {
                        const size_t num_elements = tensor.size / sizeof(int32_t);
                        std::vector<int32_t> i32_data(num_elements);

                        file.read(reinterpret_cast<char *>(i32_data.data()), tensor.size);

                        //tensor.data = i32_data;

                        out.write(reinterpret_cast<const char *>(i32_data.data()), tensor.size);
                        break;
                    }
                    case GGML_TYPE_I16:
                    {
                        const size_t num_elements = tensor.size / sizeof(int16_t);
                        std::vector<int16_t> i16_data(num_elements);

                        file.read(reinterpret_cast<char *>(i16_data.data()), tensor.size);

                        //tensor.data = i16_data;

                        out.write(reinterpret_cast<const char *>(i16_data.data()), tensor.size);
                        break;
                    }
                    case GGML_TYPE_I8:
                    {
                        const size_t num_elements = tensor.size / sizeof(int8_t);
                        std::vector<int8_t> i8_data(num_elements);

                        file.read(reinterpret_cast<char *>(i8_data.data()), tensor.size);

                        //tensor.data = i8_data;

                        out.write(reinterpret_cast<const char *>(i8_data.data()), tensor.size);
                        break;
                    }
                    default:
                    {
                         std::vector<uint8_t> raw_data(tensor.size);
    
                        if (!file.read(reinterpret_cast<char*>(raw_data.data()), tensor.size)) {
                            throw std::runtime_error("Failed to read raw tensor data");
                        }
                                                
                        //tensor.data = raw_data;
                        
                        out.write(reinterpret_cast<const char*>(raw_data.data()), raw_data.size());
                        
                        std::cerr << "Warning: Unknown tensor type " << tensor.type 
                                << " stored as raw bytes (" << tensor.size << " bytes)" << std::endl;
                        break;
                    }
                }
            }
            else
            {
                // Determine the block size according to the quantization type
                size_t block_size = 0;
                switch (tensor.type) {
                case GGML_TYPE_Q2_K: block_size = sizeof(block_q2_K); break;
                case GGML_TYPE_Q3_K: block_size = sizeof(block_q3_K); break;
                case GGML_TYPE_Q4_K: block_size = sizeof(block_q4_K); break;
                case GGML_TYPE_Q5_K: block_size = sizeof(block_q5_K); break;
                case GGML_TYPE_Q6_K: block_size = sizeof(block_q6_K); break;
                case GGML_TYPE_Q8_K: block_size = sizeof(block_q8_K); break;
                default:
                    throw std::runtime_error("Quantization type " + std::to_string(tensor.type) + " not supported");
            }

                // esta parte lanza un error
                /*
                if (tensor.size % block_size != 0) {
                    throw std::runtime_error("Tensor size " + std::to_string(tensor.size) + 
                        " not aligned with block size " + std::to_string(block_size) +
                        " for type " + ggml_type_name(tensor.type));
                }
                */

                std::vector<uint8_t> quant_data(tensor.size);
                if (!file.read(reinterpret_cast<char*>(quant_data.data()), tensor.size)) {
                    throw std::runtime_error("Error reading quantized data");
                }

                //tensor.data = quant_data;

                out.write(reinterpret_cast<const char*>(quant_data.data()), quant_data.size());
            }
        }

        out.put('\n');
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////


        graph_data.tensors.push_back(tensor);

        const std::string end_tensors_marker = "---END_TENSOR---";
        out.write(end_tensors_marker.c_str(), end_tensors_marker.size());
        out.put('\n');
    }
    out.close();

    return graph_data;
}

/**
 * @brief Generates a DOT graph representation of the GGUF model architecture
 *
 * This function creates a Graphviz DOT format string that visualizes the tensor
 * operations and their connections in the GGUF model. Each tensor is represented
 * as a node with operation type and dimensions, and connections show data flow.
 *
 * @param graph The GraphData structure containing tensor information
 * @return std::string The DOT format graph as a string
 */
std::string generate_computational_graph(const GraphData& graph) {
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

/*
    // Conexión desde el último bloque a la salida
    if (!layer_tensors.empty()) {
        int last_layer = layer_tensors.rbegin()->first;
        if (last_layer != -1) {
            std::string last_block_output = "blk_" + std::to_string(last_layer) + "_block_output";
            dot_output += "    " + last_block_output + " -> output_norm_weight [ltail=cluster_" + 
                          std::to_string(last_layer) + "];\n";
        }
    }
*/

    dot_output += "}\n";
    return dot_output;
}

bool save_dot_to_file(const std::string& dot_content, const std::string& filename) {
    std::ofstream out_file(filename);
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return false;
    }
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

    GraphData graph_data = gguf_graph_data(ctx_gguf, fname, "graph.graph");

    std::string dot_content = generate_computational_graph(graph_data);
    save_dot_to_file(dot_content, "graph.dot");

    gguf_free(ctx_gguf);

    return true;
}
