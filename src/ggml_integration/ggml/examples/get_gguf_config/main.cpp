#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <string.h>
#include <variant>
#include "graph_data_structs.h"
#include "gguf.h"   

/**
 * @brief Parsea un contexto GGUF y almacena todos los datos en una estructura GraphData
 * @param ctx Contexto GGUF cargado
 * @param fname Nombre del archivo GGUF (para leer datos de tensores)
 * @return Estructura GraphData con todos los datos cargados
 */
GraphData gguf_graph_data(const struct gguf_context *ctx, const char *fname) {
    GraphData graph_data;
    
    if (!ctx) {
        std::cerr << "Contexto GGUF es NULL." << std::endl;
        return graph_data;
    }

    // Llenar el encabezado
    graph_data.header.n_tensors = gguf_get_n_tensors(ctx);
    graph_data.header.n_kv = gguf_get_n_kv(ctx);

    // Llenar metadatos
    for (int64_t i = 0; i < gguf_get_n_kv(ctx); ++i) {
        GGUFMetadata md;
        md.key = gguf_get_key(ctx, i);
        md.type = gguf_get_kv_type(ctx, i);

        switch (md.type) {
            case GGUF_TYPE_UINT8: md.value.u8 = gguf_get_val_u8(ctx, i); break;
            case GGUF_TYPE_INT8: md.value.i8 = gguf_get_val_i8(ctx, i); break;
            case GGUF_TYPE_UINT16: md.value.u16 = gguf_get_val_u16(ctx, i); break;
            case GGUF_TYPE_INT16: md.value.i16 = gguf_get_val_i16(ctx, i); break;
            case GGUF_TYPE_UINT32: md.value.u32 = gguf_get_val_u32(ctx, i); break;
            case GGUF_TYPE_INT32: md.value.i32 = gguf_get_val_i32(ctx, i); break;
            case GGUF_TYPE_FLOAT32: md.value.f32 = gguf_get_val_f32(ctx, i); break;
            case GGUF_TYPE_BOOL: md.value.b = gguf_get_val_bool(ctx, i); break;
            case GGUF_TYPE_STRING: md.str = gguf_get_val_str(ctx, i); break;
            case GGUF_TYPE_UINT64: md.value.u64 = gguf_get_val_u64(ctx, i); break;
            case GGUF_TYPE_INT64: md.value.i64 = gguf_get_val_i64(ctx, i); break;
            case GGUF_TYPE_FLOAT64: md.value.f64 = gguf_get_val_f64(ctx, i); break;
            case GGUF_TYPE_ARRAY:
                md.array.type = gguf_get_arr_type(ctx, i);
                md.array.size = gguf_get_arr_n(ctx, i);
                
                switch (md.array.type) {
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
                    case GGUF_TYPE_STRING: {
                        const char **str_array = (const char **)gguf_get_arr_data(ctx, i);
                        std::vector<std::string> strings;
                        strings.reserve(md.array.size);
                        for (size_t j = 0; j < md.array.size; j++) {
                            strings.emplace_back(str_array[j] ? str_array[j] : "");
                        }
                        md.array.data = strings;
                        break;
                    }
                    default:
                        break;
                }
                break;
            default: break;
        }

        graph_data.metadata.push_back(md);
    }

    // Llenar información de tensores
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        GGUFTensor tensor;
        tensor.name = gguf_get_tensor_name(ctx, i);
        tensor.type = gguf_get_tensor_type(ctx, i);
        tensor.size = gguf_get_tensor_size(ctx, i);

        // Obtener dimensiones
        const int32_t n_dims = gguf_get_tensor_n_dims(ctx, i);
        const int64_t *dims = gguf_get_tensor_dims(ctx, i);
        tensor.dims.assign(dims, dims + n_dims);
        tensor.n_dims = n_dims;

        // Leer datos del tensor
        std::ifstream file(fname, std::ios::binary);
        if (file) {
            size_t offset = gguf_get_tensor_offset(ctx, i);
            file.seekg(offset, std::ios::beg);

            // Asignar el tipo de almacenamiento correcto
            if (!ggml_is_quantized(tensor.type)) {
                switch (tensor.type) {
                    case GGML_TYPE_F32: {
                        std::vector<float> float_data(tensor.size / sizeof(float));
                        file.read(reinterpret_cast<char*>(float_data.data()), tensor.size);
                        tensor.data = float_data;
                        break;
                    }
                    case GGML_TYPE_F16: {
                        std::vector<uint16_t> f16_data(tensor.size / sizeof(uint16_t));
                        file.read(reinterpret_cast<char*>(f16_data.data()), tensor.size);
                        tensor.data = f16_data;
                        break;
                    }
                    case GGML_TYPE_I32: {
                        std::vector<int32_t> i32_data(tensor.size / sizeof(int32_t));
                        file.read(reinterpret_cast<char*>(i32_data.data()), tensor.size);
                        tensor.data = i32_data;
                        break;
                    }
                    case GGML_TYPE_I16: {
                        std::vector<int16_t> i16_data(tensor.size / sizeof(int16_t));
                        file.read(reinterpret_cast<char*>(i16_data.data()), tensor.size);
                        tensor.data = i16_data;
                        break;
                    }
                    case GGML_TYPE_I8: {
                        std::vector<int8_t> i8_data(tensor.size / sizeof(int8_t));
                        file.read(reinterpret_cast<char*>(i8_data.data()), tensor.size);
                        tensor.data = i8_data;
                        break;
                    }
                    default: {
                        std::vector<uint8_t> raw_data(tensor.size);
                        file.read(reinterpret_cast<char*>(raw_data.data()), tensor.size);
                        tensor.data = raw_data;
                        break;
                    }
                }
            } else {
                // Para tipos cuantizados, usar vector<uint8_t>
                std::vector<uint8_t> quant_data(tensor.size);
                file.read(reinterpret_cast<char*>(quant_data.data()), tensor.size);
                tensor.data = quant_data;
            }
        }

        graph_data.tensors.push_back(tensor);
    }

    return graph_data;
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
        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
    }

    // Cerrar el archivo de salida
    outfile.close();
}


// Función que obtiene la configuración de un archivo GGUF
int main() {
    const char *fname = "llama-2-7b.Q2_K.gguf";
    const char *output_filename = "output.txt";

    // Verifica si el nombre del archivo es válido
    if (!fname) {
        std::cerr << "Nombre de archivo inválido (NULL)\n";
        return 1;  // Código de error
    }

    struct ggml_context *ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    // Intenta cargar el archivo GGUF
    struct gguf_context *ctx_gguf = gguf_init_from_file(fname, params);
    if (!ctx_gguf) {
        std::cerr << "No se pudo cargar el archivo GGUF '" << fname << "'\n";
        return 1;  // Código de error
    }

    GraphData graph_data = gguf_graph_data(ctx_gguf, fname);

    print_graph_data(graph_data, output_filename);

    // Liberar el contexto GGUF cuando ya no sea necesario
    gguf_free(ctx_gguf);

    return 0;  // Éxito
}
