#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include "gguf.h"  


void gguf_print_context(const struct gguf_context *ctx, const char *fname) {
    if (!ctx) {
        std::cerr << "Contexto GGUF es NULL." << std::endl;
        return;
    }

    // Imprimir el encabezado
    std::cout << "\n *************************  HEADER  ************************* \n";
    std::cout << "Version: " << gguf_get_version(ctx) << "\n";
    std::cout << "Number of tensors: " << gguf_get_n_tensors(ctx) << "\n";
    std::cout << "Number of key-value pairs: " << gguf_get_n_kv(ctx) << "\n";

    // Imprimir los metadatos (pares clave-valor)
    std::cout << "\n *************************  METADATA  ************************* \n";
    for (int64_t i = 0; i < gguf_get_n_kv(ctx); ++i) {
        const char *key = gguf_get_key(ctx, i);
        enum gguf_type type = gguf_get_kv_type(ctx, i);

        std::cout << "Key: " << key << "\n";

        // Imprimir el valor según su tipo
        switch (type) {
            case GGUF_TYPE_UINT8:
                std::cout << "Value: " << static_cast<int>(gguf_get_val_u8(ctx, i)) << " (uint8)\n";
                break;
            case GGUF_TYPE_INT8:
                std::cout << "Value: " << static_cast<int>(gguf_get_val_i8(ctx, i)) << " (int8)\n";
                break;
            case GGUF_TYPE_UINT16:
                std::cout << "Value: " << gguf_get_val_u16(ctx, i) << " (uint16)\n";
                break;
            case GGUF_TYPE_INT16:
                std::cout << "Value: " << gguf_get_val_i16(ctx, i) << " (int16)\n";
                break;
            case GGUF_TYPE_UINT32:
                std::cout << "Value: " << gguf_get_val_u32(ctx, i) << " (uint32)\n";
                break;
            case GGUF_TYPE_INT32:
                std::cout << "Value: " << gguf_get_val_i32(ctx, i) << " (int32)\n";
                break;
            case GGUF_TYPE_FLOAT32:
                std::cout << "Value: " << gguf_get_val_f32(ctx, i) << " (float32)\n";
                break;
            case GGUF_TYPE_BOOL:
                std::cout << "Value: " << (gguf_get_val_bool(ctx, i) ? "true" : "false") << " (bool)\n";
                break;
            case GGUF_TYPE_STRING:
                std::cout << "Value: " << gguf_get_val_str(ctx, i) << " (string)\n";
                break;
            case GGUF_TYPE_UINT64:
                std::cout << "Value: " << gguf_get_val_u64(ctx, i) << " (uint64)\n";
                break;
            case GGUF_TYPE_INT64:
                std::cout << "Value: " << gguf_get_val_i64(ctx, i) << " (int64)\n";
                break;
            case GGUF_TYPE_FLOAT64:
                std::cout << "Value: " << gguf_get_val_f64(ctx, i) << " (float64)\n";
                break;
            case GGUF_TYPE_ARRAY:
                std::cout << "Value: (array of type " << gguf_get_arr_type(ctx, i) << ", size " << gguf_get_arr_n(ctx, i) << ")\n";
                break;
            default:
                std::cout << "Value: Unknown type\n";
                break;
        }

        std::cout << "Type: " << gguf_type_name(type) << "\n";
        std::cout << " ----------------------------------- \n";
    }

    // Imprimir la información de los tensores
    std::cout << "\n *************************  TENSORS  ************************* \n";
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        size_t size = gguf_get_tensor_size(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);

        std::cout << "Tensor Name: " << name << "\n";
        std::cout << "Tensor Type: " << ggml_type_name(type) << "\n";
        std::cout << "Tensor Size: " << size << " bytes\n";
        std::cout << "Tensor Offset: " << offset << "\n";

        // Leer los datos del tensor desde el archivo
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            std::cerr << "No se pudo abrir el archivo: " << fname << "\n";
            return;
        }

        file.seekg(offset, std::ios::beg);
        std::vector<uint8_t> buffer(size);
        file.read(reinterpret_cast<char*>(buffer.data()), size);

        if (!file) {
            std::cerr << "Error al leer los datos del tensor.\n";
            return;
        }

        // Imprimir los datos del tensor según su tipo
        std::cout << "Tensor data: ";
        switch (type) {
            case GGML_TYPE_F32:
                for (size_t j = 0; j < size / sizeof(float); ++j) {
                    std::cout << reinterpret_cast<float*>(buffer.data())[j] << " ";
                }
                break;
            case GGML_TYPE_F16:
                for (size_t j = 0; j < size / sizeof(uint16_t); ++j) {
                    std::cout << reinterpret_cast<uint16_t*>(buffer.data())[j] << " ";
                }
                break;
            case GGML_TYPE_I32:
                for (size_t j = 0; j < size / sizeof(int32_t); ++j) {
                    std::cout << reinterpret_cast<int32_t*>(buffer.data())[j] << " ";
                }
                break;
            case GGML_TYPE_I16:
                for (size_t j = 0; j < size / sizeof(int16_t); ++j) {
                    std::cout << reinterpret_cast<int16_t*>(buffer.data())[j] << " ";
                }
                break;
            case GGML_TYPE_I8:
                for (size_t j = 0; j < size / sizeof(int8_t); ++j) {
                    std::cout << static_cast<int>(reinterpret_cast<int8_t*>(buffer.data())[j]) << " ";
                }
                break;
            default:
                std::cout << "Unsupported tensor type for printing.";
                break;
        }
        std::cout << "\n";

        std::cout << " ---------------------------------------------------------------------- \n";
    }
}


// Función que obtiene la configuración de un archivo GGUF
int main() {
    const char *fname = "llama-2-7b.Q2_K.gguf"; 

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

    // Imprimir el contexto
    gguf_print_context(ctx_gguf, fname);

    // Liberar el contexto GGUF cuando ya no sea necesario
    gguf_free(ctx_gguf);

    return 0;  // Éxito
}
