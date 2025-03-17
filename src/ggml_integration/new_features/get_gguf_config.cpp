#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include "gguf.h"  



void gguf_print_context(const struct gguf_context *ctx, const char *fname, const char *output_filename) {
    if (!ctx) {
        std::cerr << "Contexto GGUF es NULL." << std::endl;
        return;
    }

    // Abrir el archivo de salida
    std::ofstream outfile(output_filename);
    if (!outfile) {
        std::cerr << "No se pudo abrir el archivo de salida: " << output_filename << "\n";
        return;
    }

    // Escribir el encabezado
    outfile << "\n ************************************************************** \n";
    outfile << "\n **************************  HEADER  ************************** \n";
    outfile << "\n ************************************************************** \n";
    outfile << "Version: " << gguf_get_version(ctx) << "\n";
    outfile << "Number of tensors: " << gguf_get_n_tensors(ctx) << "\n";
    outfile << "Number of key-value pairs: " << gguf_get_n_kv(ctx) << "\n";

    // Escribir los metadatos (pares clave-valor)
    outfile << "\n ************************************************************** \n";
    outfile << "\n *************************  METADATA  ************************* \n";
    outfile << "\n ************************************************************** \n";
    for (int64_t i = 0; i < gguf_get_n_kv(ctx); ++i) {
        const char *key = gguf_get_key(ctx, i);
        enum gguf_type type = gguf_get_kv_type(ctx, i);

        outfile << "Key: " << key << "\n";

        // Escribir el valor según su tipo
        switch (type) {
            case GGUF_TYPE_UINT8:
                outfile << "Value: " << static_cast<int>(gguf_get_val_u8(ctx, i)) << " (uint8)\n";
                break;
            case GGUF_TYPE_INT8:
                outfile << "Value: " << static_cast<int>(gguf_get_val_i8(ctx, i)) << " (int8)\n";
                break;
            case GGUF_TYPE_UINT16:
                outfile << "Value: " << gguf_get_val_u16(ctx, i) << " (uint16)\n";
                break;
            case GGUF_TYPE_INT16:
                outfile << "Value: " << gguf_get_val_i16(ctx, i) << " (int16)\n";
                break;
            case GGUF_TYPE_UINT32:
                outfile << "Value: " << gguf_get_val_u32(ctx, i) << " (uint32)\n";
                break;
            case GGUF_TYPE_INT32:
                outfile << "Value: " << gguf_get_val_i32(ctx, i) << " (int32)\n";
                break;
            case GGUF_TYPE_FLOAT32:
                outfile << "Value: " << gguf_get_val_f32(ctx, i) << " (float32)\n";
                break;
            case GGUF_TYPE_BOOL:
                outfile << "Value: " << (gguf_get_val_bool(ctx, i) ? "true" : "false") << " (bool)\n";
                break;
            case GGUF_TYPE_STRING:
                outfile << "Value: " << gguf_get_val_str(ctx, i) << " (string)\n";
                break;
            case GGUF_TYPE_UINT64:
                outfile << "Value: " << gguf_get_val_u64(ctx, i) << " (uint64)\n";
                break;
            case GGUF_TYPE_INT64:
                outfile << "Value: " << gguf_get_val_i64(ctx, i) << " (int64)\n";
                break;
            case GGUF_TYPE_FLOAT64:
                outfile << "Value: " << gguf_get_val_f64(ctx, i) << " (float64)\n";
                break;
            case GGUF_TYPE_ARRAY:
                outfile << "Value: (array of type " << gguf_get_arr_type(ctx, i) << ", size " << gguf_get_arr_n(ctx, i) << ")\n";
                break;
            default:
                outfile << "Value: Unknown type\n";
                break;
        }

        outfile << "Type: " << gguf_type_name(type) << "\n";
        outfile << " ----------------------------------- \n";
    }

    // Escribir la información de los tensores
    outfile << "\n ************************************************************* \n";
    outfile << "\n *************************  TENSORS  ************************* \n";
    outfile << "\n ************************************************************* \n";
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx); ++i) {
        const char *name = gguf_get_tensor_name(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        size_t size = gguf_get_tensor_size(ctx, i);
        size_t offset = gguf_get_tensor_offset(ctx, i);

        outfile << "Tensor Name: " << name << "\n";
        outfile << "Tensor Type: " << ggml_type_name(type) << "\n";
        outfile << "Tensor Size: " << size << " bytes\n";
        outfile << "Tensor Offset: " << offset << "\n";

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

        // Escribir los datos del tensor según su tipo
        outfile << "Tensor data: ";
        switch (type) {
            // Tipos no cuantizados
            case GGML_TYPE_F32:
                for (size_t j = 0; j < size / sizeof(float); ++j) {
                    outfile << reinterpret_cast<float*>(buffer.data())[j] << " ";
                }
                break;
            case GGML_TYPE_F16:
                for (size_t j = 0; j < size / sizeof(uint16_t); ++j) {
                    outfile << reinterpret_cast<uint16_t*>(buffer.data())[j] << " ";
                }
                break;
            case GGML_TYPE_I32:
                for (size_t j = 0; j < size / sizeof(int32_t); ++j) {
                    outfile << reinterpret_cast<int32_t*>(buffer.data())[j] << " ";
                }
                break;
            case GGML_TYPE_I16:
                for (size_t j = 0; j < size / sizeof(int16_t); ++j) {
                    outfile << reinterpret_cast<int16_t*>(buffer.data())[j] << " ";
                }
                break;
            case GGML_TYPE_I8:
                for (size_t j = 0; j < size / sizeof(int8_t); ++j) {
                    outfile << static_cast<int>(reinterpret_cast<int8_t*>(buffer.data())[j]) << " ";
                }
                break;
        
            // Tipos cuantizados tradicionales
            case GGML_TYPE_Q4_0:
                // Descuantizar tensores q4_0
                outfile << "[Datos cuantizados (q4_0) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_Q4_1:
                // Descuantizar tensores q4_1
                outfile << "[Datos cuantizados (q4_1) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_Q5_0:
                // Descuantizar tensores q5_0
                outfile << "[Datos cuantizados (q5_0) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_Q5_1:
                // Descuantizar tensores q5_1
                outfile << "[Datos cuantizados (q5_1) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_Q8_0:
                // Descuantizar tensores q8_0
                outfile << "[Datos cuantizados (q8_0) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_Q8_1:
                // Descuantizar tensores q8_1
                outfile << "[Datos cuantizados (q8_1) - no se pueden imprimir directamente]";
                break;
        
            // Tipos cuantizados modernos (K-quants)
            case GGML_TYPE_Q2_K:
                // Descuantizar tensores q2_K
                outfile << "[Datos cuantizados (q2_K) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_Q3_K:
                // Descuantizar tensores q3_K
                outfile << "[Datos cuantizados (q3_K) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_Q4_K:
                // Descuantizar tensores q4_K
                outfile << "[Datos cuantizados (q4_K) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_Q5_K:
                // Descuantizar tensores q5_K
                outfile << "[Datos cuantizados (q5_K) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_Q6_K:
                // Descuantizar tensores q6_K
                outfile << "[Datos cuantizados (q6_K) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_Q8_K:
                // Descuantizar tensores q8_K
                outfile << "[Datos cuantizados (q8_K) - no se pueden imprimir directamente]";
                break;
        
            // Tipos cuantizados especiales (IQ-quants)
            case GGML_TYPE_IQ2_XXS:
                // Descuantizar tensores iq2_xxs
                outfile << "[Datos cuantizados (iq2_xxs) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_IQ2_XS:
                // Descuantizar tensores iq2_xs
                outfile << "[Datos cuantizados (iq2_xs) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_IQ3_XXS:
                // Descuantizar tensores iq3_xxs
                outfile << "[Datos cuantizados (iq3_xxs) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_IQ1_S:
                // Descuantizar tensores iq1_s
                outfile << "[Datos cuantizados (iq1_s) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_IQ4_NL:
                // Descuantizar tensores iq4_nl
                outfile << "[Datos cuantizados (iq4_nl) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_IQ3_S:
                // Descuantizar tensores iq3_s
                outfile << "[Datos cuantizados (iq3_s) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_IQ2_S:
                // Descuantizar tensores iq2_s
                outfile << "[Datos cuantizados (iq2_s) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_IQ4_XS:
                // Descuantizar tensores iq4_xs
                outfile << "[Datos cuantizados (iq4_xs) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_IQ1_M:
                // Descuantizar tensores iq1_m
                outfile << "[Datos cuantizados (iq1_m) - no se pueden imprimir directamente]";
                break;
        
            // Tipos experimentales u obsoletos
            case GGML_TYPE_TQ1_0:
                // Manejar tipos experimentales si es necesario
                outfile << "[Datos cuantizados (tq1_0) - no se pueden imprimir directamente]";
                break;
            case GGML_TYPE_TQ2_0:
                // Manejar tipos experimentales si es necesario
                outfile << "[Datos cuantizados (tq2_0) - no se pueden imprimir directamente]";
                break;
        
            default:
                outfile << "Unsupported tensor type for printing.";
                break;
        }
        outfile << "\n";

        outfile << " \n ---------------------------------------------------------------------- \n";
        outfile << " \n ---------------------------------------------------------------------- \n";
        outfile << " \n ---------------------------------------------------------------------- \n";
        outfile << " \n ---------------------------------------------------------------------- \n";
    }

    // Cerrar el archivo de salida
    outfile.close();
    std::cout << "La información se ha escrito en el archivo: " << output_filename << "\n";
}



// Función que obtiene la configuración de un archivo GGUF
bool get_gguf_config(const char *fname, const char *output_filename) {
    //const char *fname = "llama-2-7b.Q2_K.gguf";
    //const char *output_filename = "output.txt";

    // Verifica si el nombre del archivo es válido
    if (!fname) {
        std::cerr << "Nombre de archivo inválido (NULL)\n";
        return false;  // Código de error
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
        return false;  // Código de error
    }

    // Escribir el contexto en el archivo de salida
    gguf_print_context(ctx_gguf, fname, output_filename);

    // Liberar el contexto GGUF cuando ya no sea necesario
    gguf_free(ctx_gguf);

    return true;
}
