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

// Función para inferir la operación basada en el nombre del tensor
enum ggml_op infer_operation(const std::string &tensor_name)
{
    // Patrones para atención
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

    // Patrones para feed-forward
    if (tensor_name.find("ffn_up.") != std::string::npos ||
        tensor_name.find("ffn_down.") != std::string::npos ||
        tensor_name.find("ffn_gate.") != std::string::npos ||
        tensor_name.find("fc1.") != std::string::npos ||
        tensor_name.find("fc2.") != std::string::npos)
    {
        return GGML_OP_MUL_MAT;
    }

    // Patrones para embeddings
    if (tensor_name.find("token_embd.") != std::string::npos ||
        tensor_name.find("embed_tokens.") != std::string::npos ||
        tensor_name.find("position_embd.") != std::string::npos ||
        tensor_name.find("embed_positions.") != std::string::npos)
    {
        return GGML_OP_MUL_MAT;
    }

    // Patrones para normalización
    if (tensor_name.find("_norm.") != std::string::npos ||
        tensor_name.find("layer_norm.") != std::string::npos ||
        tensor_name.find("final_layer_norm.") != std::string::npos)
    {
        return GGML_OP_NORM;
    }

    // Por defecto asumimos una operación de multiplicación de matrices
    return GGML_OP_MUL_MAT;
}

// Función para inferir tensores fuente
std::vector<std::string> infer_src_tensors(const std::string &tensor_name,
                                           const std::vector<GGUFTensor> &tensors)
{
    std::vector<std::string> src_tensors;

    try
    {
        // Extraer prefijo de bloque (ej. "blk.0.", "model.decoder.layers.0.")
        size_t block_end = tensor_name.find_last_of('.');
        std::string block_prefix = (block_end != std::string::npos) ? tensor_name.substr(0, block_end + 1) : "";

        // 1. Para tensores de atención Q/K/V
        if (tensor_name.find("attn_q.") != std::string::npos ||
            tensor_name.find("q_proj.") != std::string::npos ||
            tensor_name.find("attn_k.") != std::string::npos ||
            tensor_name.find("k_proj.") != std::string::npos ||
            tensor_name.find("attn_v.") != std::string::npos ||
            tensor_name.find("v_proj.") != std::string::npos)
        {

            // Buscar tensor de normalización correspondiente
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
        // 2. Para tensores de salida de atención
        else if (tensor_name.find("attn_output.") != std::string::npos ||
                 tensor_name.find("out_proj.") != std::string::npos)
        {
            src_tensors.push_back(block_prefix + "attn_q");
            src_tensors.push_back(block_prefix + "attn_k");
            src_tensors.push_back(block_prefix + "attn_v");
        }
        // 3. Para tensores feed-forward
        else if (tensor_name.find("ffn_down.") != std::string::npos ||
                 tensor_name.find("fc2.") != std::string::npos)
        {
            src_tensors.push_back(block_prefix + "ffn_up");
        }
        else if (tensor_name.find("ffn_up.") != std::string::npos ||
                 tensor_name.find("fc1.") != std::string::npos)
        {
            // Buscar tensor de normalización FFN
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
        // 4. Para capas de normalización
        else if (tensor_name.find("_norm.") != std::string::npos)
        {
            if (!block_prefix.empty())
            {
                // Extraer número de bloque de forma segura
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
                    // Construir nombre del bloque anterior
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
        std::cerr << "Error procesando tensor '" << tensor_name << "': " << e.what() << std::endl;
    }

    // Eliminar posibles duplicados
    std::sort(src_tensors.begin(), src_tensors.end());
    src_tensors.erase(std::unique(src_tensors.begin(), src_tensors.end()), src_tensors.end());

    return src_tensors;
}

// Función para inferir el tensor destino
std::string infer_dst_tensor(const std::string &tensor_name)
{
    // Extraer el prefijo del bloque
    size_t block_end = tensor_name.find_last_of('.');
    std::string block_prefix = (block_end != std::string::npos) ? tensor_name.substr(0, block_end + 1) : "";

    // Para tensores de atención Q/K/V
    if (tensor_name.find("attn_q.") != std::string::npos ||
        tensor_name.find("q_proj.") != std::string::npos ||
        tensor_name.find("attn_k.") != std::string::npos ||
        tensor_name.find("k_proj.") != std::string::npos ||
        tensor_name.find("attn_v.") != std::string::npos ||
        tensor_name.find("v_proj.") != std::string::npos)
    {
        return block_prefix + "attn_output";
    }
    // Para tensores feed-forward up/down
    else if (tensor_name.find("ffn_up.") != std::string::npos ||
             tensor_name.find("fc1.") != std::string::npos)
    {
        return block_prefix + "ffn_down";
    }
    else if (tensor_name.find("ffn_down.") != std::string::npos ||
             tensor_name.find("fc2.") != std::string::npos)
    {
        return block_prefix + "layer_output";
    }
    // Para tensores de normalización
    else if (tensor_name.find("_norm.") != std::string::npos)
    {
        if (tensor_name.find("attn_norm.") != std::string::npos)
        {
            return block_prefix + "attn_q";
        }
        else if (tensor_name.find("ffn_norm.") != std::string::npos)
        {
            return block_prefix + "ffn_up";
        }
        else
        {
            return block_prefix + "layer_output";
        }
    }

    return ""; // No se pudo inferir
}

/**
 * @brief Parsea un contexto GGUF y almacena todos los datos en una estructura GraphData
 * @param ctx Contexto GGUF cargado
 * @param fname Nombre del archivo GGUF (para leer datos de tensores)
 * @return Estructura GraphData con todos los datos cargados
 */
GraphData gguf_graph_data(const struct gguf_context *ctx, const char *fname)
{
    GraphData graph_data;

    if (!ctx)
    {
        std::cerr << "Contexto GGUF es NULL." << std::endl;
        return graph_data;
    }

    // Llenar el encabezado
    graph_data.header.n_tensors = gguf_get_n_tensors(ctx);
    graph_data.header.n_kv = gguf_get_n_kv(ctx);

    // Llenar metadatos
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
                // Marcar que hay un array de strings no procesado (analizar si se necesita o si se puede omitir)
                md.array.data = std::vector<std::string>(); // Vacío

                /*
                // Esto accede a datos internos de GGUF y puede ser inseguro.
                const auto &item = ctx->kv[i];
                std::vector<std::string> strings;
                strings.reserve(item.size);

                // Se asume que los strings están almacenados como punteros consecutivos
                const char **str_ptrs = reinterpret_cast<const char**>(item.data.data());
                for (size_t j = 0; j < item.size; ++j) {
                    strings.emplace_back(str_ptrs[j] ? str_ptrs[j] : "");
                }
                md.array.data = strings;
                */
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

    // Llenar información de tensores
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i)
    {
        GGUFTensor tensor;
        tensor.name = gguf_get_tensor_name(ctx, i);
        tensor.type = gguf_get_tensor_type(ctx, i);
        tensor.size = gguf_get_tensor_size(ctx, i);

        // Obtener dimensiones
        const int32_t n_dims = gguf_get_tensor_n_dims(ctx, i);
        const int64_t *dims = gguf_get_tensor_dims(ctx, i);
        tensor.dims.assign(dims, dims + n_dims);
        tensor.n_dims = n_dims;

        /////////////////////////////////////

        // Leer datos del tensor
        std::ifstream file(fname, std::ios::binary);
        if (file)
        {
            size_t offset = gguf_get_tensor_offset(ctx, i);
            file.seekg(offset, std::ios::beg);

            // Asignar el tipo de almacenamiento correcto
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
                // Para tipos cuantizados, usar vector<uint8_t>
                std::vector<uint8_t> quant_data(tensor.size);
                file.read(reinterpret_cast<char *>(quant_data.data()), tensor.size);
                tensor.data = quant_data;
            }
        }


        ////////////////////////////////////


        // Marcar que no cargamos los datos
        //tensor.data = std::vector<uint8_t>();

        // Inferir operación y conexiones
        tensor.op = infer_operation(tensor.name);
        tensor.src_tensors = infer_src_tensors(tensor.name, graph_data.tensors);
        tensor.dst_tensor = infer_dst_tensor(tensor.name);

        graph_data.tensors.push_back(tensor);
    }

    return graph_data;
}


std::string generate_dot_graph(const GraphData &graph_data)
{
    std::ostringstream dot;

    // Encabezado del archivo DOT
    dot << "digraph GGUF_Graph {\n";
    dot << "  rankdir=LR;\n";
    dot << "  node [shape=box, style=filled, fillcolor=\"#f0f0f0\", fontname=\"Helvetica\"];\n";
    dot << "  edge [fontname=\"Helvetica\", fontsize=10];\n\n";

    // Agregar nodos (tensores)
    for (const auto &tensor : graph_data.tensors)
    {
        std::string node_name = tensor.name;

        // Obtener nombre de la operación
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

        // Crear etiqueta con nombre, operación y dimensiones
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

        // Color diferente según el tipo de operación
        std::string color;
        switch (tensor.op)
        {
        case GGML_OP_MUL_MAT:
            color = "#d4f1f9"; // Azul claro
            break;
        case GGML_OP_NORM:
            color = "#d5e8d4"; // Verde claro
            break;
        case GGML_OP_SOFT_MAX:
            color = "#f8cecc"; // Rojo claro
            break;
        default:
            color = "#f0f0f0"; // Gris claro
        }

        dot << "  \"" << node_name << "\" [label=\"" << label << "\", fillcolor=\"" << color << "\"];\n";
    }

    dot << "\n";

    // Agregar conexiones (aristas)
    for (const auto &tensor : graph_data.tensors)
    {
        // Conexiones desde tensores fuente
        for (const auto &src : tensor.src_tensors)
        {
            dot << "  \"" << src << "\" -> \"" << tensor.name << "\";\n";
        }

        // Conexión al tensor destino (si existe)
        if (!tensor.dst_tensor.empty())
        {
            dot << "  \"" << tensor.name << "\" -> \"" << tensor.dst_tensor << "\";\n";
        }
    }

    dot << "}\n";

    return dot.str();
}


// Función para guardar el gráfico DOT en un archivo
bool save_dot_graph(const GraphData &graph_data, const std::string &filename)
{
    std::ofstream out_file(filename);
    if (!out_file.is_open())
    {
        std::cerr << "Error al abrir el archivo: " << filename << std::endl;
        return false;
    }

    std::string dot_content = generate_dot_graph(graph_data);
    out_file << dot_content;
    out_file.close();

    return true;
}

/**
 * @brief Función principal para cargar un archivo GGUF y obtener la configuración
 * @param fname Nombre del archivo GGUF
 * @return true si la carga fue exitosa, false en caso contrario
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
