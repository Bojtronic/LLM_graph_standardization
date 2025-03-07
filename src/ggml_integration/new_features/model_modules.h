#ifndef MODEL_MODULES_H
#define MODEL_MODULES_H

#include "ggml.h"
#include <cmath>
#include <cstring>
#include <string>

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MÓDULOS COMUNES /////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Módulo de Atención Multi-Cabezal
// Parámetros:
// - ctx: Contexto de GGML.
// - Q, K, V: Tensores de Consulta, Clave y Valor.
// - is_causal: Indica si se debe aplicar una máscara causal (para modelos como LLAMA 2 y Whisper).
ggml_tensor * multi_head_attention(ggml_context * ctx, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V, bool is_causal) {
    // Calcular las matrices de atención
    ggml_tensor * scores = ggml_mul_mat(ctx, Q, K); // Q * K^T
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float)Q->ne[0])); // Escalar por sqrt(d_k)

    // Aplicar máscara causal (si es necesario)
    if (is_causal) {
        scores = ggml_diag_mask_inf(ctx, scores, 0); // Máscara causal
    }

    // Aplicar softmax para obtener los pesos de atención
    ggml_tensor * attn_weights = ggml_soft_max(ctx, scores);

    // Multiplicar por los valores (V)
    ggml_tensor * output = ggml_mul_mat(ctx, attn_weights, V);

    return output;
}

// Módulo de Conexiones Residuales
// Parámetros:
// - ctx: Contexto de GGML.
// - input: Tensor de entrada.
// - use_rmsnorm: Indica si se debe usar RMSNorm (para LLAMA 2 y DeepSeek) o LayerNorm (para ViT y Whisper).
// - eps: Valor pequeño para estabilidad numérica (evita divisiones por cero).
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * input, bool use_rmsnorm, float eps) {
    if (use_rmsnorm) {
        return ggml_rms_norm(ctx, input, eps); // RMSNorm
    } else {
        return ggml_norm(ctx, input, eps); // LayerNorm
    }
}

// Implementación de SWIGLU
ggml_tensor * ggml_swiglu(ggml_context * ctx, ggml_tensor * x) {
    // Calcular SiLU(x) = x * sigmoid(x)
    ggml_tensor * silu = ggml_mul(ctx, x, ggml_sigmoid(ctx, x));

    // Calcular SWIGLU(x) = SiLU(x) * x
    return ggml_mul(ctx, silu, x);
}

// Módulo de Red Feed-Forward
// Parámetros:
// - ctx: Contexto de GGML.
// - input: Tensor de entrada.
// - weight: Peso de la capa lineal.
// - bias: Sesgo de la capa lineal.
// - activation: Tipo de función de activación (ReLU, GELU o SWIGLU).
ggml_tensor * feed_forward(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight, ggml_tensor * bias, const char * activation) {
    ggml_tensor * output = ggml_add(ctx, ggml_mul_mat(ctx, weight, input), bias);

    // Aplicar función de activación
    if (strcmp(activation, "relu") == 0) {
        output = ggml_relu(ctx, output);
    } else if (strcmp(activation, "gelu") == 0) {
        output = ggml_gelu(ctx, output);
    } else if (strcmp(activation, "swiglu") == 0) {
        output = ggml_swiglu(ctx, output);
    }

    return output;
}

// Funciones personalizadas para seno y coseno
void ggml_sin_f32(int n, float * dest, const float * src) {
    for (int i = 0; i < n; i++) {
        dest[i] = sinf(src[i]);
    }
}

void ggml_cos_f32(int n, float * dest, const float * src) {
    for (int i = 0; i < n; i++) {
        dest[i] = cosf(src[i]);
    }
}

// Implementación manual de ggml_pow
ggml_tensor * ggml_pow(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    // Calcular log(a)
    ggml_tensor * log_a = ggml_log(ctx, a);

    // Calcular b * log(a)
    ggml_tensor * b_log_a = ggml_mul(ctx, b, log_a);

    // Calcular exp(b * log(a))
    return ggml_exp(ctx, b_log_a);
}

// Funciones personalizadas para seno y coseno
void ggml_sin_f32(int n, float * dest, const float * src) {
    for (int i = 0; i < n; i++) {
        dest[i] = sinf(src[i]);
    }
}

void ggml_cos_f32(int n, float * dest, const float * src) {
    for (int i = 0; i < n; i++) {
        dest[i] = cosf(src[i]);
    }
}

// Módulo de Codificaciones Posicionales
// Parámetros:
// - ctx: Contexto de GGML.
// - input: Tensor de entrada.
// - type: Tipo de codificación posicional ("rope" o "sinusoidal").
// - n_dims: Número de dimensiones rotatorias (solo para RoPE).
// - mode: Modo de RoPE (solo para RoPE).
// - base: Base para el cálculo de la codificación sinusoidal (solo para sinusoidal).
ggml_tensor * positional_encoding(
    ggml_context * ctx,
    ggml_tensor * input,
    const char * type,
    int n_dims,   // Número de dimensiones rotatorias (RoPE)
    int mode,     // Modo de RoPE (0 para RoPE estándar)
    float base    // Base para la codificación sinusoidal (sinusoidal)
) {
    if (strcmp(type, "rope") == 0) {
        // Rotary Positional Embeddings (RoPE)
        // Crear un tensor de posiciones (b) para RoPE
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input->ne[0]);
        for (int i = 0; i < input->ne[0]; i++) {
            ((float *)positions->data)[i] = (float)i;
        }

        // Aplicar RoPE
        return ggml_rope(ctx, input, positions, n_dims, mode);
    } else if (strcmp(type, "sinusoidal") == 0) {
        // Codificación posicional sinusoidal
        int n_pos = input->ne[0]; // Número de posiciones (longitud de la secuencia)
        int d_model = input->ne[1]; // Dimensionalidad del modelo

        // Crear un tensor para almacenar las posiciones (0, 1, 2, ..., n_pos-1)
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_pos);
        for (int pos = 0; pos < n_pos; pos++) {
            ((float *)positions->data)[pos] = (float)pos;
        }

        // Crear un tensor para almacenar las dimensiones (0, 1, 2, ..., d_model-1)
        ggml_tensor * dimensions = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);
        for (int i = 0; i < d_model; i++) {
            ((float *)dimensions->data)[i] = (float)i;
        }

        // Calcular los ángulos: pos / (base^(i / d_model))
        ggml_tensor * base_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ((float *)base_tensor->data)[0] = base;

        // Calcular base^(i / d_model) usando ggml_pow
        ggml_tensor * pow_result = ggml_pow(ctx, base_tensor, dimensions);

        // Calcular pos / (base^(i / d_model))
        ggml_tensor * angles = ggml_div(ctx, positions, pow_result);

        // Calcular las codificaciones posicionales: seno para índices pares, coseno para índices impares
        ggml_tensor * sin_enc = ggml_map_unary_f32(ctx, angles, ggml_sin_f32); // Seno
        ggml_tensor * cos_enc = ggml_map_unary_f32(ctx, angles, ggml_cos_f32); // Coseno

        // Combinar seno y coseno en un solo tensor
        ggml_tensor * pos_enc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_pos, d_model);
        for (int pos = 0; pos < n_pos; pos++) {
            for (int i = 0; i < d_model; i++) {
                float * ptr = (float *)pos_enc->data + pos * d_model + i;
                if (i % 2 == 0) {
                    *ptr = ((float *)sin_enc->data)[pos * d_model + i]; // Seno para índices pares
                } else {
                    *ptr = ((float *)cos_enc->data)[pos * d_model + i]; // Coseno para índices impares
                }
            }
        }

        // Sumar las codificaciones posicionales al tensor de entrada
        return ggml_add(ctx, input, pos_enc);
    } else {
        // Sin codificación posicional
        return input;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MÓDULOS ESPECÍFICOS /////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Módulo de Cross-Attention (Whisper)
// Parámetros:
// - ctx: Contexto de GGML.
// - Q, K, V: Tensores de Consulta, Clave y Valor.
ggml_tensor * cross_attention(ggml_context * ctx, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V) {
    return multi_head_attention(ctx, Q, K, V, false); // Sin máscara causal
}

// Módulo de Token de Clase (ViT)
// Parámetros:
// - ctx: Contexto de GGML.
// - input: Tensor de entrada.
ggml_tensor * class_token(ggml_context * ctx, ggml_tensor * input) {
    ggml_tensor * cls_token = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input->ne[0]);
    // Inicializar el token de clase (por ejemplo, con ceros)
    ggml_set_zero(cls_token);

    // Concatenar el token de clase con la entrada
    return ggml_concat(ctx, cls_token, input, 0);
}

// Módulo de MLA (DeepSeek)
// Parámetros:
// - ctx: Contexto de GGML.
// - Q, K, V: Tensores de Consulta, Clave y Valor.
// - W_Q, W_K, W_V: Pesos de proyección para Q, K y V.
// - latent_dim: Dimensión del espacio latente.
// - is_causal: Indica si se debe aplicar una máscara causal.
ggml_tensor * multi_head_latent_attention(ggml_context * ctx, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V, 
                                          ggml_tensor * W_Q, ggml_tensor * W_K, ggml_tensor * W_V, 
                                          int latent_dim, bool is_causal) {
    // Paso 1: Proyectar Q, K, V en un espacio latente de menor dimensión
    ggml_tensor * Q_proj = ggml_mul_mat(ctx, W_Q, Q); // Q_proj = W_Q * Q
    ggml_tensor * K_proj = ggml_mul_mat(ctx, W_K, K); // K_proj = W_K * K
    ggml_tensor * V_proj = ggml_mul_mat(ctx, W_V, V); // V_proj = W_V * V

    // Paso 2: Calcular las puntuaciones de atención en el espacio latente
    ggml_tensor * scores = ggml_mul_mat(ctx, Q_proj, K_proj); // Q_proj * K_proj^T
    scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float)latent_dim)); // Escalar por sqrt(latent_dim)

    // Paso 3: Aplicar máscara causal (si es necesario)
    if (is_causal) {
        scores = ggml_diag_mask_inf(ctx, scores, 0); // Máscara causal
    }

    // Paso 4: Aplicar softmax para obtener los pesos de atención
    ggml_tensor * attn_weights = ggml_soft_max(ctx, scores);

    // Paso 5: Multiplicar por los valores proyectados (V_proj)
    ggml_tensor * output = ggml_mul_mat(ctx, attn_weights, V_proj);

    // Paso 6: Proyectar de vuelta al espacio original (opcional)
    ggml_tensor * W_O = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Q->ne[1], latent_dim); // Peso de proyección de salida
    ggml_tensor * output_proj = ggml_mul_mat(ctx, W_O, output); // output_proj = W_O * output

    return output_proj;
}

#endif // MODEL_MODULES_H

