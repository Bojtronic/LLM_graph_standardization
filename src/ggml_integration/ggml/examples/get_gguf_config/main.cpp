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

// Función mejorada para inferir la operación basada en el nombre del tensor
enum ggml_op infer_operation(const std::string &tensor_name) {
    // Mapa de patrones a operaciones para mayor flexibilidad y mantenibilidad
    static const std::vector<std::pair<std::vector<std::string>, ggml_op>> op_patterns = {
        // Atención y proyecciones
        {{"attn_q.", "q_proj.", "attn_k.", "k_proj.", "attn_v.", "v_proj.", 
          "attn_output.", "out_proj.", "wq.", "wk.", "wv.", "wo.", 
          "wq_a.", "wq_b.", "wkv_a_mqa.", "wkv_b.", "qkv_w."}, GGML_OP_MUL_MAT},
        
        // Feed-forward y MLP
        {{"ffn_up.", "ffn_down.", "ffn_gate.", "fc1.", "fc2.", 
          "mlp_lin1.", "mlp_lin2.", "ffn_up_exps.", "ffn_down_exps.", 
          "ffn_gate_exps.", "ffn_up_shexp.", "ffn_down_shexp.", 
          "ffn_gate_shexp."}, GGML_OP_MUL_MAT},
        
        // Capas de normalización
        {{"_norm.", "layer_norm.", "final_layer_norm.", "attn_norm.", 
          "ffn_norm.", "norm1.", "norm2.", "output_norm."}, GGML_OP_NORM},
        
        // Embebimientos
        {{"token_embd.", "embed_tokens.", "position_embd.", 
          "embed_positions.", "cls_token.", "pe."}, GGML_OP_MUL_MAT},
        
        // Operaciones de convolución (ViT)
        //{{"proj_w.", "proj_b.", "conv_"}, GGML_OP_CONV},
        
        // Operaciones de activación
        //{{"gelu.", "silu.", "relu.", "softmax."}, GGML_OP_GELU}, // GGML_OP_SILU, etc.
        
        // Operaciones de pooling y posprocesamiento
        {{"pool.", "head.", "classifier."}, GGML_OP_MUL_MAT},
        
        // Operaciones de RoPE
        {{"rope.", "rotary."}, GGML_OP_ROPE}
    };

    // Buscar coincidencias con los patrones definidos
    for (const auto &[patterns, op] : op_patterns) {
        for (const auto &pattern : patterns) {
            if (tensor_name.find(pattern) != std::string::npos) {
                return op;
            }
        }
    }

    // Patrones especiales para operaciones complejas
    if (tensor_name.find("softmax") != std::string::npos) {
        return GGML_OP_SOFT_MAX;
    }
    /*
    if (tensor_name.find("silu") != std::string::npos) {
        return GGML_OP_SILU;
    }
    if (tensor_name.find("gelu") != std::string::npos) {
        return GGML_OP_GELU;
    }
    */
    if (tensor_name.find("concat") != std::string::npos) {
        return GGML_OP_CONCAT;
    }
    if (tensor_name.find("reshape") != std::string::npos || 
        tensor_name.find("view") != std::string::npos) {
        return GGML_OP_RESHAPE;
    }

    // Para tensores de entrada/salida especiales
    if (tensor_name == "input" || tensor_name == "output") {
        return GGML_OP_NONE;
    }

    // Operación por defecto (más común en modelos)
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
                //md.array.data = std::vector<std::string>(); // Vacío

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


void print_graph_data(const GraphData& graph_data, const char *output_filename) {
    // Abrir el archivo de salida
    std::ofstream outfile(output_filename);
    if (!outfile) {
        std::cerr << "No se pudo abrir el archivo de salida: " << output_filename << "\n";
        return;
    }

    // Función helper para nombres de tipos GGUF
    auto gguf_type_name = [](enum gguf_type type) -> const char* {
        static const char* names[] = {
            "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32",
            "FLOAT32", "BOOL", "STRING", "UINT64", "INT64", "FLOAT64", "ARRAY"
        };
        return (type >= GGUF_TYPE_UINT8 && type <= GGUF_TYPE_ARRAY) ? names[type] : "UNKNOWN";
    };

    // Escribir el encabezado
    outfile << "\n**************************************************************\n";
    outfile << "**************************  HEADER  **************************\n";
    outfile << "**************************************************************\n";
    outfile << "Número de tensores: " << graph_data.header.n_tensors << "\n";
    outfile << "Número de pares clave-valor: " << graph_data.header.n_kv << "\n";

    // Escribir los metadatos (pares clave-valor)
    outfile << "\n**************************************************************\n";
    outfile << "*************************  METADATA  *************************\n";
    outfile << "**************************************************************\n";
    
    for (const auto &md : graph_data.metadata) {
        outfile << "Clave: " << md.key << "\n";
        outfile << "Tipo: " << gguf_type_name(md.type) << "\n";
        
        switch (md.type) {
            case GGUF_TYPE_UINT8:
                outfile << "Valor: " << static_cast<int>(md.value.u8) << " (uint8)\n";
                break;
            case GGUF_TYPE_INT8:
                outfile << "Valor: " << static_cast<int>(md.value.i8) << " (int8)\n";
                break;
            case GGUF_TYPE_UINT16:
                outfile << "Valor: " << md.value.u16 << " (uint16)\n";
                break;
            case GGUF_TYPE_INT16:
                outfile << "Valor: " << md.value.i16 << " (int16)\n";
                break;
            case GGUF_TYPE_UINT32:
                outfile << "Valor: " << md.value.u32 << " (uint32)\n";
                break;
            case GGUF_TYPE_INT32:
                outfile << "Valor: " << md.value.i32 << " (int32)\n";
                break;
            case GGUF_TYPE_FLOAT32:
                outfile << std::fixed << std::setprecision(6);
                outfile << "Valor: " << md.value.f32 << " (float32)\n";
                outfile.unsetf(std::ios::fixed);
                outfile.precision(6);
                break;
            case GGUF_TYPE_BOOL:
                outfile << "Valor: " << (md.value.b ? "true" : "false") << " (bool)\n";
                break;
            case GGUF_TYPE_STRING:
                outfile << "Valor: " << md.str << " (string)\n";
                break;
            case GGUF_TYPE_UINT64:
                outfile << "Valor: " << md.value.u64 << " (uint64)\n";
                break;
            case GGUF_TYPE_INT64:
                outfile << "Valor: " << md.value.i64 << " (int64)\n";
                break;
            case GGUF_TYPE_FLOAT64:
                outfile << std::fixed << std::setprecision(6);
                outfile << "Valor: " << md.value.f64 << " (float64)\n";
                outfile.unsetf(std::ios::fixed);
                outfile.precision(6);
                break;
                case GGUF_TYPE_ARRAY:
                outfile << "Tipo de array: " << gguf_type_name(md.array.type) << "\n";
                outfile << "Tamaño del array: " << md.array.size << "\n";
                
                // Mostrar primeros elementos para tipos conocidos
                if (md.array.type == GGUF_TYPE_STRING) {
                    if (const auto* strs = std::get_if<std::vector<std::string>>(&md.array.data)) {
                        outfile << "Primeros strings: [";
                        for (size_t i = 0; i < std::min(strs->size(), 5UL); ++i) {
                            outfile << "\"" << (*strs)[i] << "\" ";
                        }
                        outfile << "...]\n";
                    }
                }
                else if (md.array.type == GGUF_TYPE_FLOAT32) {
                    if (const auto* vals = std::get_if<std::vector<float>>(&md.array.data)) {
                        outfile << std::fixed << std::setprecision(6);
                        outfile << "Primeros valores: [";
                        for (size_t i = 0; i < std::min(vals->size(), 5UL); ++i) {
                            outfile << (*vals)[i] << " ";
                        }
                        outfile << "...]\n";
                        outfile.unsetf(std::ios::fixed);
                    }
                }
                else if (md.array.type == GGUF_TYPE_INT32) {
                    if (const auto* vals = std::get_if<std::vector<int32_t>>(&md.array.data)) {
                        outfile << "Primeros valores: [";
                        for (size_t i = 0; i < std::min(vals->size(), 5UL); ++i) {
                            outfile << (*vals)[i] << " ";
                        }
                        outfile << "...]\n";
                    }
                }
                else if (md.array.type == GGUF_TYPE_UINT8) {
                    if (const auto* vals = std::get_if<std::vector<uint8_t>>(&md.array.data)) {
                        outfile << "Primeros valores: [";
                        for (size_t i = 0; i < std::min(vals->size(), 5UL); ++i) {
                            outfile << static_cast<int>((*vals)[i]) << " ";
                        }
                        outfile << "...]\n";
                    }
                }
                else {
                    outfile << "[Datos binarios de tipo " << gguf_type_name(md.array.type) << "]\n";
                }
                break;
            default:
                outfile << "Valor: [tipo desconocido]\n";
                break;
        }
        outfile << "-----------------------------------\n";
    }

    // Escribir información de tensores
    outfile << "\n**************************************************************\n";
    outfile << "*************************  TENSORS  **************************\n";
    outfile << "**************************************************************\n";
    
    for (const auto &tensor : graph_data.tensors) {
        outfile << "Nombre: " << tensor.name << "\n";
        outfile << "Tipo: " << ggml_type_name(tensor.type) << "\n";
        outfile << "Tamaño: " << std::fixed << std::setprecision(2) 
               << tensor.size / 1024.0f / 1024.0f << " MB\n";
        outfile << "Número de dimensiones: " << tensor.n_dims << "\n";
        outfile << "Tamaño de cada dimensión: ";
        for (const auto &dim : tensor.dims) {
            outfile << dim << " ";
        }

        outfile << "\n";

        // Mostrar los primeros elementos del tensor según su tipo
        outfile << "Datos (primeros elementos): ";
        
        const size_t max_elements = 5; // Mostrar solo los primeros 5 elementos
        
        switch (tensor.type) {
            case GGML_TYPE_F32:
                if (const auto* data = std::get_if<std::vector<float>>(&tensor.data)) {
                    for (size_t i = 0; i < std::min(data->size(), max_elements); ++i) {
                        outfile << (*data)[i] << " ";
                    }
                }
                break;
                
            case GGML_TYPE_I32:
                if (const auto* data = std::get_if<std::vector<int32_t>>(&tensor.data)) {
                    for (size_t i = 0; i < std::min(data->size(), max_elements); ++i) {
                        outfile << (*data)[i] << " ";
                    }
                }
                break;
                
            case GGML_TYPE_F16:
                if (const auto* data = std::get_if<std::vector<uint16_t>>(&tensor.data)) {
                    for (size_t i = 0; i < std::min(data->size(), max_elements); ++i) {
                        outfile << (*data)[i] << " ";
                    }
                }
                break;
                
            case GGML_TYPE_I8:
                if (const auto* data = std::get_if<std::vector<int8_t>>(&tensor.data)) {
                    for (size_t i = 0; i < std::min(data->size(), max_elements); ++i) {
                        outfile << static_cast<int>((*data)[i]) << " "; 
                    }
                }
                break;
                
            // Tipos cuantizados
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
                if (const auto* data = std::get_if<std::vector<uint8_t>>(&tensor.data)) {
                    //outfile << "[Datos cuantizados - " << data->size() << " bytes]";
                    outfile << "[Datos cuantizados: se toman en grupos de 8 bits en este caso]  ";
                    for (size_t i = 0; i < std::min(data->size(), max_elements); ++i) {
                        outfile << static_cast<uint8_t>((*data)[i]) << " "; 
                    }
                }
                break;
                
            default:
                if (const auto* data = std::get_if<std::vector<uint8_t>>(&tensor.data)) {
                    outfile << "[Datos binarios - " << data->size() << " bytes]";
                }
                break;
        }

        // Indicar si hay más elementos
        if (std::visit([](const auto& v) { return v.size(); }, tensor.data) > max_elements) {
            outfile << "... [total: " 
                << std::visit([](const auto& v) { return v.size(); }, tensor.data) 
                << " elementos]";
        }
        
        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
    }

    // Cerrar el archivo de salida
    outfile.close();
    std::cout << "La información de GraphData se ha escrito en el archivo: " << output_filename << "\n";

}


int main()
{
    const char *fname = "llama-2-7b.Q2_K.gguf";
    const char *output_filename = "graph.dot";

    // Verifica si el nombre del archivo es válido
    if (!fname)
    {
        std::cerr << "Nombre de archivo inválido (NULL)\n";
        return 1; // Código de error
    }

    struct ggml_context *ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    // Intenta cargar el archivo GGUF
    struct gguf_context *ctx_gguf = gguf_init_from_file(fname, params);
    if (!ctx_gguf)
    {
        std::cerr << "No se pudo cargar el archivo GGUF '" << fname << "'\n";
        return 1; // Código de error
    }

    GraphData graph_data = gguf_graph_data(ctx_gguf, fname);
    
    //save_dot_graph(graph_data, output_filename);

    // Liberar el contexto GGUF cuando ya no sea necesario
    gguf_free(ctx_gguf);

    return 0; // Éxito
}
