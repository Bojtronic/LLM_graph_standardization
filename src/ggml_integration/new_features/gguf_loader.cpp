#include "gguf_loader.h" 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
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


/**
 * @brief Función principal para cargar un archivo GGUF y obtener la configuración
 * @param fname Nombre del archivo GGUF
 * @return true si la carga fue exitosa, false en caso contrario
 */
bool get_gguf_config(const char *fname) {

    if (!fname) {
        std::cerr << "Nombre de archivo inválido (NULL)\n";
        return false;  
    }

    struct ggml_context *ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    struct gguf_context *ctx_gguf = gguf_init_from_file(fname, params);
    if (!ctx_gguf) {
        std::cerr << "No se pudo cargar el archivo GGUF '" << fname << "'\n";
        return false;  
    }

    GraphData graph_data = gguf_graph_data(ctx_gguf, fname);

    gguf_free(ctx_gguf);

    return true;
}
