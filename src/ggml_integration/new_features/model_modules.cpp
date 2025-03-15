#include "model_modules.h"
#include <cmath>
#include <cstring>

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MÓDULOS COMUNES /////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Módulo de Atención Multi-Cabezal
// Parámetros:
// - ctx: Contexto de GGML.
// - Q, K, V: Tensores de Consulta, Clave y Valor.
// - is_causal: Indica si se debe aplicar una máscara causal (para modelos como LLAMA 2 y Whisper).
// Retorna:
// - Un tensor que representa el resultado de la atención multi-cabezal
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
// - eps:(épsilon) Valor pequeño añadido al denominador en la normalización para evitar divisiones por cero y garantizar estabilidad numérica. Típicamente un valor como 1e-5 o 1e-6.
// Retorna:
// - Un tensor normalizado utilizando RMSNorm o LayerNorm, dependiendo del valor de `use_rmsnorm`.
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * input, bool use_rmsnorm,  float eps) {
    if (use_rmsnorm) {
        return ggml_rms_norm(ctx, input, eps); // RMSNorm
    } else {
        return ggml_norm(ctx, input, eps); // LayerNorm
    }
}

// Función que implementa la activación SwiGLU (Swish-Gated Linear Unit).
// SwiGLU es una función de activación que combina la función Swish (Sigmoid-Weighted Linear Unit) con una compuerta (gating mechanism)
// Parámetros:
// - ctx: Contexto de GGML para manejar la memoria y los tensores.
// - x: Tensor de entrada.
// Retorna:
// - Un nuevo tensor que representa el resultado de la operación SwiGLU.
ggml_tensor * ggml_swiglu(ggml_context * ctx, ggml_tensor * x) {
    // Paso 1: Calcula la función Swish (x * sigmoid(x))
    // Combina una multiplicación por la entrada con la función sigmoide aplicada a la misma entrada.
    ggml_tensor * silu = ggml_mul(ctx, x, ggml_sigmoid(ctx, x));
    // Paso 2: Aplica la compuerta (gating mechanism) multiplicando el resultado de Swish por la entrada original (x)
    return ggml_mul(ctx, silu, x);
}



// Módulo de Red Feed-Forward
// Parámetros:
// - ctx: Contexto de GGML.
// - input: Tensor de entrada.
// - weight: Peso de la capa lineal.
// - bias: Sesgo de la capa lineal.
// - activation: Tipo de función de activación (ReLU, GELU o SWIGLU).
// Retorna:
// - Un tensor que representa la salida de la red feed-forward después de aplicar la transformación lineal y la función de activación especificada.
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

// Calcula el seno de cada elemento en un arreglo de números flotantes.
// Parámetros:
// - n: Número de elementos en el arreglo.
// - dest: Arreglo de destino donde se almacenarán los resultados.
// - src: Arreglo de origen que contiene los valores de entrada.
void ggml_sin_f32(int n, float * dest, const float * src) {
    for (int i = 0; i < n; i++) {
        // Calcula el seno de src[i] y lo almacena en dest[i].
        dest[i] = sinf(src[i]);
    }
}

// Calcula el coseno de cada elemento en un arreglo de números flotantes.
// Parámetros:
// - n: Número de elementos en el arreglo.
// - dest: Arreglo de destino donde se almacenarán los resultados.
// - src: Arreglo de origen que contiene los valores de entrada.
void ggml_cos_f32(int n, float * dest, const float * src) {
    for (int i = 0; i < n; i++) {
        // Calcula el coseno de src[i] y lo almacena en dest[i].
        dest[i] = cosf(src[i]);
    }
}

// Calcula la potencia de un tensor elevado a otro tensor (a^b).
// Parámetros:
// - ctx: Contexto de GGML para manejar la memoria y los tensores.
// - a: Tensor base.
// - b: Tensor exponente.
// Retorna:
// - Un nuevo tensor que representa el resultado de a^b.
ggml_tensor * ggml_pow(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    // Paso 1: Calcula el logaritmo natural del tensor base (a).
    ggml_tensor * log_a = ggml_log(ctx, a);

    // Paso 2: Multiplica el tensor exponente (b) por el logaritmo de a.
    // Esto es equivalente a b * log(a).
    ggml_tensor * b_log_a = ggml_mul(ctx, b, log_a);

    // Paso 3: Calcula la exponencial del resultado anterior.
    // Esto es equivalente a exp(b * log(a)), que es matemáticamente igual a a^b.
    return ggml_exp(ctx, b_log_a);
}

// Aplica codificación posicional a un tensor de entrada.
// La codificación posicional puede ser de tipo "rope" (Rotary Positional Embedding) o "sinusoidal".
// Parámetros:
// - ctx: Contexto de GGML para manejar la memoria y los tensores.
// - input: Tensor de entrada al que se aplicará la codificación posicional.
// - type: Tipo de codificación posicional ("rope" o "sinusoidal").
// - n_dims: Número de dimensiones para la codificación (usado en "rope").
// - mode: Modo de codificación (usado en "rope").
// - base: Base para el cálculo de frecuencias en la codificación sinusoidal.
// Retorna:
// - Un tensor con la codificación posicional aplicada.
ggml_tensor * positional_encoding(ggml_context * ctx, ggml_tensor * input, const char * type, int n_dims, int mode, float base) {
    // Codificación tipo "rope" (Rotary Positional Embedding).
    if (strcmp(type, "rope") == 0) {
        // Crea un tensor para almacenar las posiciones (índices de secuencia).
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input->ne[0]);
        for (int i = 0; i < input->ne[0]; i++) {
            ((float *)positions->data)[i] = (float)i; // Asigna valores secuenciales (0, 1, 2, ...).
        }

        // Aplica la codificación "rope" al tensor de entrada usando las posiciones.
        return ggml_rope(ctx, input, positions, n_dims, mode);
    }
    // Codificación tipo "sinusoidal".
    else if (strcmp(type, "sinusoidal") == 0) {
        // Obtiene el número de posiciones (longitud de la secuencia) y dimensiones del modelo.
        int n_pos = input->ne[0];
        int d_model = input->ne[1];

        // Crea un tensor para almacenar las posiciones (índices de secuencia).
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_pos);
        for (int pos = 0; pos < n_pos; pos++) {
            ((float *)positions->data)[pos] = (float)pos; // Asigna valores secuenciales (0, 1, 2, ...).
        }

        // Crea un tensor para almacenar las dimensiones (índices de características).
        ggml_tensor * dimensions = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);
        for (int i = 0; i < d_model; i++) {
            ((float *)dimensions->data)[i] = (float)i; // Asigna valores secuenciales (0, 1, 2, ...).
        }

        // Crea un tensor para almacenar la base de las frecuencias.
        ggml_tensor * base_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ((float *)base_tensor->data)[0] = base; // Asigna el valor de la base.

        // Calcula las frecuencias usando la fórmula: base^(-2i/d_model).
        ggml_tensor * pow_result = ggml_pow(ctx, base_tensor, dimensions);

        // Calcula los ángulos para la codificación sinusoidal: pos / frecuencias.
        ggml_tensor * angles = ggml_div(ctx, positions, pow_result);

        // Aplica la función seno a los ángulos para obtener la codificación sinusoidal.
        ggml_tensor * sin_enc = ggml_map_unary_f32(ctx, angles, ggml_sin_f32);

        // Aplica la función coseno a los ángulos para obtener la codificación cosinusoidal.
        ggml_tensor * cos_enc = ggml_map_unary_f32(ctx, angles, ggml_cos_f32);

        // Crea un tensor 2D para almacenar la codificación posicional completa.
        ggml_tensor * pos_enc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_pos, d_model);
        for (int pos = 0; pos < n_pos; pos++) {
            for (int i = 0; i < d_model; i++) {
                float * ptr = (float *)pos_enc->data + pos * d_model + i;
                // Alterna entre seno y coseno para cada dimensión.
                if (i % 2 == 0) {
                    *ptr = ((float *)sin_enc->data)[pos * d_model + i]; // Usa seno para índices pares.
                } else {
                    *ptr = ((float *)cos_enc->data)[pos * d_model + i]; // Usa coseno para índices impares.
                }
            }
        }

        // Suma la codificación posicional al tensor de entrada.
        return ggml_add(ctx, input, pos_enc);
    }
    // Si el tipo de codificación no es reconocido, devuelve el tensor de entrada sin cambios.
    else {
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
// Retorna:
// - Un tensor que representa el resultado de la atención cruzada.
ggml_tensor * cross_attention(ggml_context * ctx, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V) {
    // Llama a la función multi_head_attention para calcular la atención multi-cabeza.
    // El parámetro "false" indica que no se aplicará una máscara causal.
    // La máscara causal se usa en modelos autoregresivos para evitar que las posiciones futuras influyan en las actuales.
    // En este caso, no es necesaria porque la atención cruzada no es autoregresiva.
    return multi_head_attention(ctx, Q, K, V, false);
}

// Módulo de Token de Clase (ViT)
// Parámetros:
// - ctx: Contexto de GGML.
// - input: Tensor de entrada.
// Retorna:
// - Un nuevo tensor que contiene el token de clase concatenado con el tensor de entrada.
ggml_tensor * class_token(ggml_context * ctx, ggml_tensor * input) {
    // Crea un tensor 1D para representar el token de clase.
    // El tamaño del tensor es igual al número de características (dimensiones) del tensor de entrada.
    ggml_tensor * cls_token = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input->ne[0]);

    // Inicializa el token de clase con ceros, aunque también podría inicializarse con valores aprendidos.
    ggml_set_zero(cls_token);

    // Concatena el token de clase con el tensor de entrada a lo largo del eje 0 (primera dimensión).
    // Esto agrega el token de clase al inicio de la secuencia.
    return ggml_concat(ctx, cls_token, input, 0);
}


// Módulo de Atención Multi-Cabeza Latente
// Parámetros:
// - ctx: Contexto de GGML para manejar la memoria y los tensores.
// - Q: Tensor de Consulta (Query).
// - K: Tensor de Clave (Key).
// - V: Tensor de Valor (Value).
// - W_Q: Matriz de pesos para proyectar las consultas (Q) en el espacio latente.
// - W_K: Matriz de pesos para proyectar las claves (K) en el espacio latente.
// - W_V: Matriz de pesos para proyectar los valores (V) en el espacio latente.
// - latent_dim: Dimensión del espacio latente (número de características después de la proyección).
// - is_causal: Indica si se debe aplicar una máscara causal
// Retorna:
// - Un tensor que representa el resultado de la atención multi-cabeza latente.
ggml_tensor * multi_head_latent_attention(ggml_context * ctx, ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V, 
    ggml_tensor * W_Q, ggml_tensor * W_K, ggml_tensor * W_V, 
    int latent_dim, bool is_causal) {
// Proyecta las consultas (Q), claves (K) y valores (V) en el espacio latente.
ggml_tensor * Q_proj = ggml_mul_mat(ctx, W_Q, Q); // Q_proj = Q * W_Q
ggml_tensor * K_proj = ggml_mul_mat(ctx, W_K, K); // K_proj = K * W_K
ggml_tensor * V_proj = ggml_mul_mat(ctx, W_V, V); // V_proj = V * W_V

// Calcula las puntuaciones de atención (scores) como el producto escalar entre Q_proj y K_proj.
ggml_tensor * scores = ggml_mul_mat(ctx, Q_proj, K_proj); // scores = Q_proj * K_proj^T

// Escala las puntuaciones de atención para evitar valores demasiado grandes.
// La escala es 1 / sqrt(latent_dim), una técnica común para estabilizar el entrenamiento.
scores = ggml_scale(ctx, scores, 1.0f / sqrtf((float)latent_dim));

// Aplica una máscara causal si es necesario (para modelos autoregresivos).
// La máscara causal evita que las posiciones futuras influyan en las actuales.
if (is_causal) {
scores = ggml_diag_mask_inf(ctx, scores, 0); // Aplica máscara con valores infinitos en las posiciones futuras.
}

// Calcula los pesos de atención aplicando la función softmax a las puntuaciones escaladas.
ggml_tensor * attn_weights = ggml_soft_max(ctx, scores); // attn_weights = softmax(scores)

// Aplica los pesos de atención a los valores proyectados (V_proj).
ggml_tensor * output = ggml_mul_mat(ctx, attn_weights, V_proj); // output = attn_weights * V_proj

// Crea una matriz de pesos para proyectar la salida de la atención en el espacio final.
ggml_tensor * W_O = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Q->ne[1], latent_dim);

// Proyecta la salida de la atención usando la matriz de pesos W_O.
ggml_tensor * output_proj = ggml_mul_mat(ctx, W_O, output); // output_proj = output * W_O

// Retorna el resultado final de la atención multi-cabeza latente.
return output_proj;
}

