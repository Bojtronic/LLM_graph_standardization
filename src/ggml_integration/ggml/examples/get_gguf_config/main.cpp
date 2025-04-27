#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <cmath> 
#include "gguf.h"  


void read_tensor_data(const char *fname, std::ofstream &outfile, size_t offset, size_t size, enum ggml_type type) {
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
        case GGML_TYPE_Q4_0: {
            outfile << "[Datos cuantizados (" << ggml_type_name(type) << ") - no se pueden imprimir directamente]";
            break;
            /*
            const float scale = reinterpret_cast<const float*>(buffer.data())[0];
            const uint8_t *quantized_values = buffer.data() + sizeof(float);
            for (size_t j = 0; j < size; ++j) {  // Iterar sobre cada byte
                uint8_t byte_value = quantized_values[j];  // Obtener el byte actual
                uint8_t quantized_value_1 = byte_value & 0x0F;  // Primer nibble (4 bits inferiores)
                uint8_t quantized_value_2 = (byte_value >> 4) & 0x0F;  // Segundo nibble (4 bits superiores)
        
                // Descuantizar ambos valores
                float dequantized_value_1 = scale * (quantized_value_1 - 8);  // Ajustar el bias
                float dequantized_value_2 = scale * (quantized_value_2 - 8);  // Ajustar el bias
        
                // Imprimir ambos valores
                outfile << dequantized_value_1 << " " << dequantized_value_2 << " ";
            }
            break;
            */
        }
        
        case GGML_TYPE_Q4_1: {
            outfile << "[Datos cuantizados (" << ggml_type_name(type) << ") - no se pueden imprimir directamente]";
            break;
            /*
            const float scale = reinterpret_cast<const float*>(buffer.data())[0];
            const float bias = reinterpret_cast<const float*>(buffer.data() + sizeof(float))[0];
            const uint8_t *quantized_values = buffer.data() + 2 * sizeof(float);
            for (size_t j = 0; j < size; ++j) {  // Iterar sobre cada byte
                uint8_t byte_value = quantized_values[j];  // Obtener el byte actual
                uint8_t quantized_value_1 = byte_value & 0x0F;  // Primer nibble (4 bits inferiores)
                uint8_t quantized_value_2 = (byte_value >> 4) & 0x0F;  // Segundo nibble (4 bits superiores)
        
                // Descuantizar ambos valores
                float dequantized_value_1 = scale * quantized_value_1 + bias;
                float dequantized_value_2 = scale * quantized_value_2 + bias;
        
                // Imprimir ambos valores
                outfile << dequantized_value_1 << " " << dequantized_value_2 << " ";
            }
            break;
            */
        }

        case GGML_TYPE_Q8_0: {
            outfile << "[Datos cuantizados (" << ggml_type_name(type) << ") - no se pueden imprimir directamente]";
            break;
            /*
            const float scale = reinterpret_cast<const float*>(buffer.data())[0];
            const int8_t *quantized_values = reinterpret_cast<const int8_t*>(buffer.data() + sizeof(float));
            for (size_t j = 0; j < size; ++j) {  // Cada byte contiene 1 valor de 8 bits
                float dequantized_value = scale * quantized_values[j];
                outfile << dequantized_value << " ";
            }
            break;
            */
        }

        // Otros tipos cuantizados
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_1:
            outfile << "[Datos cuantizados (" << ggml_type_name(type) << ") - no se pueden imprimir directamente]";
            break;

        case GGML_TYPE_Q2_K: {
            outfile << "[Datos cuantizados (" << ggml_type_name(type) << ") - no se pueden imprimir directamente]";
            break;
            /*
            const float scale = reinterpret_cast<const float*>(buffer.data())[0];  // Escala global
            const uint8_t *quantized_values = buffer.data() + sizeof(float);  // Valores cuantizados
            for (size_t j = 0; j < size; ++j) {  // Iterar sobre cada byte
                uint8_t byte_value = quantized_values[j];  // Obtener el byte actual
        
                // Extraer los 4 valores de 2 bits
                uint8_t quantized_value_1 = byte_value & 0x03;  // Primer valor (2 bits inferiores)
                uint8_t quantized_value_2 = (byte_value >> 2) & 0x03;  // Segundo valor (siguientes 2 bits)
                uint8_t quantized_value_3 = (byte_value >> 4) & 0x03;  // Tercer valor (siguientes 2 bits)
                uint8_t quantized_value_4 = (byte_value >> 6) & 0x03;  // Cuarto valor (2 bits superiores)
        
                // Descuantizar los 4 valores
                float dequantized_value_1 = scale * (quantized_value_1 - 1);  // Ajustar el bias 
                float dequantized_value_2 = scale * (quantized_value_2 - 1);  // Ajustar el bias 
                float dequantized_value_3 = scale * (quantized_value_3 - 1);  // Ajustar el bias 
                float dequantized_value_4 = scale * (quantized_value_4 - 1);  // Ajustar el bias 
        
                // Imprimir los 4 valores
                outfile << dequantized_value_1 << " " << dequantized_value_2 << " "
                        << dequantized_value_3 << " " << dequantized_value_4 << " ";
            }
            break;
            */
        }

        case GGML_TYPE_Q3_K: {
            outfile << "[Datos cuantizados (" << ggml_type_name(type) << ") - no se pueden imprimir directamente]";
            break;
            /*
            const float scale = reinterpret_cast<const float*>(buffer.data())[0];  // Escala global
            const uint8_t *quantized_values = buffer.data() + sizeof(float);  // Valores cuantizados
        
            // Cada 3 bytes contiene 8 valores de 3 bits
            for (size_t j = 0; j < size; j += 3) {  // Iterar sobre cada grupo de 3 bytes
                uint8_t byte_value_1 = quantized_values[j];      // Primer byte
                uint8_t byte_value_2 = quantized_values[j + 1];  // Segundo byte
                uint8_t byte_value_3 = quantized_values[j + 2];  // Tercer byte
        
                // Extraer los 8 valores de 3 bits
                uint8_t quantized_values_array[8];
                quantized_values_array[0] = byte_value_1 & 0x07;  // Primeros 3 bits del primer byte
                quantized_values_array[1] = (byte_value_1 >> 3) & 0x07;  // Siguientes 3 bits del primer byte
                quantized_values_array[2] = ((byte_value_1 >> 6) | (byte_value_2 << 2)) & 0x07;  // Últimos 2 bits del primer byte + 1 bit del segundo byte
                quantized_values_array[3] = (byte_value_2 >> 1) & 0x07;  // Siguientes 3 bits del segundo byte
                quantized_values_array[4] = (byte_value_2 >> 4) & 0x07;  // Siguientes 3 bits del segundo byte
                quantized_values_array[5] = ((byte_value_2 >> 7) | (byte_value_3 << 1)) & 0x07;  // Último bit del segundo byte + 2 bits del tercer byte
                quantized_values_array[6] = (byte_value_3 >> 2) & 0x07;  // Siguientes 3 bits del tercer byte
                quantized_values_array[7] = (byte_value_3 >> 5) & 0x07;  // Últimos 3 bits del tercer byte
        
                // Descuantizar los 8 valores
                for (size_t k = 0; k < 8; ++k) {
                    float dequantized_value = scale * (quantized_values_array[k] - 3);  // Ajustar el bias 
                    outfile << dequantized_value << " ";
                }
            }
            break;
            */
        }

        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            outfile << "[Datos cuantizados (" << ggml_type_name(type) << ") - no se pueden imprimir directamente]";
            break;

        default:
            outfile << "Unsupported tensor type for printing.";
            break;
    }
    outfile << "\n";
}

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

        const int32_t n_dims = gguf_get_tensor_n_dims(ctx, i);
        const int64_t *dims = gguf_get_tensor_dims(ctx, i);

        outfile << "Tensor Name: " << name << "\n";
        outfile << "Tensor Type: " << ggml_type_name(type) << "\n";
        outfile << "Tensor Size: " << size << " bytes\n";
        outfile << "Tensor Offset: " << offset << "\n";

        outfile << "Number of dimensions: " << n_dims << "\n";
        outfile << "Dimensions: ";
        for (int32_t j = 0; j < n_dims; ++j) {
            outfile << dims[j] << " ";
        }
        outfile << "\n";

        read_tensor_data(fname, outfile, offset, size, type);

        outfile << "\n ---------------------------------------------------------------------- \n";
        outfile << "\n ---------------------------------------------------------------------- \n";
        outfile << "\n ---------------------------------------------------------------------- \n";
        outfile << "\n ---------------------------------------------------------------------- \n";
        outfile << "\n ---------------------------------------------------------------------- \n";
        outfile << "\n ---------------------------------------------------------------------- \n";
        outfile << "\n";
    }

    // Cerrar el archivo de salida
    outfile.close();
    std::cout << "La información se ha escrito en el archivo: " << output_filename << "\n";
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

    // Escribir el contexto en el archivo de salida
    gguf_print_context(ctx_gguf, fname, output_filename);

    // Liberar el contexto GGUF cuando ya no sea necesario
    gguf_free(ctx_gguf);

    return 0;  // Éxito
}
