#include <stdio.h>
#include <stdlib.h>
#include "gguf.h"  



// Función para imprimir el contenido de un contexto GGUF en un archivo
void gguf_print_context(const struct gguf_context *ctx) {
    // Abrir el archivo para escribir el contexto GGUF
    FILE *file = fopen("../gguf_context.txt", "w");
    if (file == NULL) {
        perror("Error al abrir el archivo");
        return;
    }

    // Imprimir el encabezado del archivo GGUF
    fprintf(file, "\n *************************  HEADER  ************************* \n \n");
    fprintf(file, "Magic: %.4s\n", ctx->header.magic);
    fprintf(file, "Version: %u\n", ctx->header.version);
    fprintf(file, "Number of tensors: %lu\n", ctx->header.n_tensors);
    fprintf(file, "Number of key-value pairs: %lu\n", ctx->header.n_kv);

    // Imprimir los metadatos (pares clave-valor)
    fprintf(file, "\n *************************  METADATA  ************************* \n \n");
    for (uint64_t i = 0; i < ctx->header.n_kv; ++i) {
        fprintf(file, "Key: %.*s\n", (int)ctx->kv[i].key.n, ctx->kv[i].key.data);

        // Imprimir el valor según su tipo
        switch (ctx->kv[i].type) {
            case GGUF_TYPE_UINT8:
                fprintf(file, "Value: %u (uint8)\n", ctx->kv[i].value.uint8);
                break;
            case GGUF_TYPE_INT8:
                fprintf(file, "Value: %d (int8)\n", ctx->kv[i].value.int8);
                break;
            case GGUF_TYPE_UINT16:
                fprintf(file, "Value: %u (uint16)\n", ctx->kv[i].value.uint16);
                break;
            case GGUF_TYPE_INT16:
                fprintf(file, "Value: %d (int16)\n", ctx->kv[i].value.int16);
                break;
            case GGUF_TYPE_UINT32:
                fprintf(file, "Value: %u (uint32)\n", ctx->kv[i].value.uint32);
                break;
            case GGUF_TYPE_INT32:
                fprintf(file, "Value: %d (int32)\n", ctx->kv[i].value.int32);
                break;
            case GGUF_TYPE_UINT64:
                fprintf(file, "Value: %lu (uint64)\n", ctx->kv[i].value.uint64);
                break;
            case GGUF_TYPE_INT64:
                fprintf(file, "Value: %ld (int64)\n", ctx->kv[i].value.int64);
                break;
            case GGUF_TYPE_FLOAT32:
                fprintf(file, "Value: %f (float32)\n", ctx->kv[i].value.float32);
                break;
            case GGUF_TYPE_FLOAT64:
                fprintf(file, "Value: %lf (float64)\n", ctx->kv[i].value.float64);
                break;
            case GGUF_TYPE_BOOL:
                fprintf(file, "Value: %s (bool)\n", ctx->kv[i].value.bool_ ? "true" : "false");
                break;
            case GGUF_TYPE_STRING:
                fprintf(file, "Value: %.*s (string)\n", (int)ctx->kv[i].value.str.n, ctx->kv[i].value.str.data);
                break;
            case GGUF_TYPE_ARRAY:
                fprintf(file, "Value: (array of type %d, size %lu)\n", ctx->kv[i].value.arr.type, ctx->kv[i].value.arr.n);
                break;
            default:
                fprintf(file, "Value: Unknown type\n");
                break;
        }

        fprintf(file, "Type: %d\n", ctx->kv[i].type);
        fprintf(file, " ----------------------------------- \n");
    }

    // Imprimir la información de los tensores
    fprintf(file, "\n *************************  TENSORS  ************************* \n \n");
    for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
        fprintf(file, "Tensor Name: %.*s\n", (int)ctx->infos[i].name.n, ctx->infos[i].name.data);
        fprintf(file, "Number of Dimensions: %u\n", ctx->infos[i].n_dims);
        fprintf(file, "Offsets: %lu\n", ctx->infos[i].offset);
        fprintf(file, "Data Size: %zu\n", ctx->infos[i].size);

        // Imprimir las dimensiones del tensor
        fprintf(file, "Dimensions: ");
        for (uint32_t j = 0; j < ctx->infos[i].n_dims; ++j) {
            fprintf(file, "%lu ", ctx->infos[i].ne[j]);
        }
        fprintf(file, "\n");

        // Crear un tensor temporal para imprimir detalles adicionales
        struct ggml_tensor temp_tensor;
        temp_tensor.type = ctx->infos[i].type;
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            temp_tensor.ne[j] = ctx->infos[i].ne[j];
        }
        temp_tensor.data = (void *)((char *)ctx->data + ctx->infos[i].offset);

        // Imprimir el tipo del tensor
        fprintf(file, "Tensor type: ");
        switch (temp_tensor.type) {
            case GGML_TYPE_F32:
                fprintf(file, "F32\n");
                break;
            case GGML_TYPE_F16:
                fprintf(file, "F16\n");
                break;
            case GGML_TYPE_Q4_0:
                fprintf(file, "Q4_0\n");
                break;
            case GGML_TYPE_Q4_1:
                fprintf(file, "Q4_1\n");
                break;
            case GGML_TYPE_Q5_0:
                fprintf(file, "Q5_0\n");
                break;
            case GGML_TYPE_Q5_1:
                fprintf(file, "Q5_1\n");
                break;
            case GGML_TYPE_Q8_0:
                fprintf(file, "Q8_0\n");
                break;
            case GGML_TYPE_Q8_1:
                fprintf(file, "Q8_1\n");
                break;
            case GGML_TYPE_Q2_K:
                fprintf(file, "Q2_K\n");
                break;
            case GGML_TYPE_Q3_K:
                fprintf(file, "Q3_K\n");
                break;
            case GGML_TYPE_Q4_K:
                fprintf(file, "Q4_K\n");
                break;
            case GGML_TYPE_Q5_K:
                fprintf(file, "Q5_K\n");
                break;
            case GGML_TYPE_Q6_K:
                fprintf(file, "Q6_K\n");
                break;
            case GGML_TYPE_Q8_K:
                fprintf(file, "Q8_K\n");
                break;
            case GGML_TYPE_IQ2_XXS:
                fprintf(file, "IQ2_XXS\n");
                break;
            case GGML_TYPE_IQ2_XS:
                fprintf(file, "IQ2_XS\n");
                break;
            case GGML_TYPE_IQ3_XXS:
                fprintf(file, "IQ3_XXS\n");
                break;
            case GGML_TYPE_IQ1_S:
                fprintf(file, "IQ1_S\n");
                break;
            case GGML_TYPE_IQ4_NL:
                fprintf(file, "IQ4_NL\n");
                break;
            case GGML_TYPE_IQ3_S:
                fprintf(file, "IQ3_S\n");
                break;
            case GGML_TYPE_IQ2_S:
                fprintf(file, "IQ2_S\n");
                break;
            case GGML_TYPE_IQ4_XS:
                fprintf(file, "IQ4_XS\n");
                break;
            case GGML_TYPE_I8:
                fprintf(file, "I8\n");
                break;
            case GGML_TYPE_I16:
                fprintf(file, "I16\n");
                break;
            case GGML_TYPE_I32:
                fprintf(file, "I32\n");
                break;
            case GGML_TYPE_I64:
                fprintf(file, "I64\n");
                break;
            case GGML_TYPE_F64:
                fprintf(file, "F64\n");
                break;
            case GGML_TYPE_IQ1_M:
                fprintf(file, "IQ1_M\n");
                break;
            case GGML_TYPE_BF16:
                fprintf(file, "BF16\n");
                break;
            default:
                fprintf(file, "Unknown type (%d)\n", temp_tensor.type);
                break;
        }

        fprintf(file, " ---------------------------------------------------------------------- \n");
    }

    // Imprimir el contenido de los datos en formato hexadecimal y como cadena
    fprintf(file, "\n==================== DATA CONTENT ====================\n");
    if (ctx->data == NULL) {
        fprintf(file, "Data is empty.\n");
    } else {
        fprintf(file, "\n==================== DATA CONTENT (HEX) ====================\n");
        uint8_t *data = (uint8_t *)ctx->data;
        for (size_t i = 0; i < ctx->size; i++) {
            fprintf(file, "%02X ", data[i]);
        }
        fprintf(file, "\n");

        fprintf(file, "\n==================== DATA CONTENT (STRING) ====================\n");
        if (ctx->data != NULL && ((char *)ctx->data)[0] != '\0') {
            fprintf(file, "%s\n", (char *)ctx->data);
        } else {
            fprintf(file, "Data is empty or not a valid string.\n");
        }
    }

    // Cerrar el archivo
    fclose(file);
}

// Función que obtiene la configuración de un archivo GGUF
bool get_gguf_config(const char *fname) {
    // Verifica si el nombre del archivo es válido
    if (!fname) {
        fprintf(stderr, "%s: Nombre de archivo inválido (NULL)\n", __func__);
        return false;
    }

    struct ggml_context *ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    // Intenta cargar el archivo GGUF
    struct gguf_context *ctx_gguf = gguf_init_from_file(fname, params);
    if (!ctx_gguf) {
        fprintf(stderr, "%s: No se pudo cargar el archivo GGUF '%s'\n", __func__, fname);
        return false;
    }

    // Imprimir el contexto
    gguf_print_context(ctx_gguf);

    // Liberar el contexto GGUF cuando ya no sea necesario
    gguf_free(ctx_gguf);

    return true;
}
