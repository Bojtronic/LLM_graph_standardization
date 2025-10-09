#include "model_modules.h"
#include <cmath>
#include <cstring>
#include <iostream>


ggml_tensor* get_layer_tensor(ggml_context* ctx, const GraphData& graph_data, const std::string& name) {
    const GGUFTensor* tensor_info = graph_data.find_tensor(name);
    if (!tensor_info) {
        std::cerr << "Tensor no encontrado: " << name << std::endl;
        return nullptr;
    }
    
    // Crear tensor GGML con las dimensiones correctas
    ggml_tensor* tensor = nullptr;
    switch (tensor_info->n_dims) {
        case 1:
            tensor = ggml_new_tensor_1d(ctx, tensor_info->type, tensor_info->dims[0]);
            break;
        case 2:
            tensor = ggml_new_tensor_2d(ctx, tensor_info->type, tensor_info->dims[0], tensor_info->dims[1]);
            break;
        default:
            std::cerr << "Dimensionalidad no soportada para " << name << std::endl;
            return nullptr;
    }
    
    // Copiar datos según el tipo
    switch (tensor_info->type) {
        case GGML_TYPE_F32: {
            if (const auto* data = tensor_info->get_data<float>()) {
                memcpy(tensor->data, data->data(), data->size() * sizeof(float));
            }
            break;
        }

/////////////////////////////////////////////////////////////////////////////////////
        //verificar si los datos cuantizados se pueden almacenar en 8 bits
        // en el link se explican las cuantizaciones
        // https://huggingface.co/docs/hub/gguf
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K: {
            if (const auto* data = tensor_info->get_data<uint8_t>()) {
                memcpy(tensor->data, data->data(), data->size());
            }
            break;
        }
/////////////////////////////////////////////////////////////////////////////////////

        default:
            std::cerr << "Tipo de tensor no soportado: " << ggml_type_name(tensor_info->type) << std::endl;
            return nullptr;
    }
    
    return tensor;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MÓDULOS COMUNES /////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Módulo de Atención Multi-Cabezal
 * 
 * @param ctx Contexto GGML para la asignación de memoria
 * @param Q Tensor de Consulta [seq_len, num_heads, head_dim]
 * @param K Tensor de Clave [seq_len, num_heads, head_dim]
 * @param V Tensor de Valor [seq_len, num_heads, head_dim]
 * @param is_causal Si es true, aplica máscara causal para evitar lookahead
 * @param attention_mask Tensor opcional para máscara de atención personalizada (NULL si no se usa)
 * @param scale_factor Factor de escalado opcional (si es 0, se usa 1/sqrt(d_k))
 * @return Tensor con el resultado de la atención [seq_len, num_heads, head_dim]
 */
ggml_tensor* multi_head_attention(ggml_context* ctx, ggml_tensor* Q, ggml_tensor* K, ggml_tensor* V, bool is_causal, ggml_tensor* attention_mask, float scale_factor) {
    // 1. Verificación de dimensiones
    if (Q->ne[0] != K->ne[0] || Q->ne[0] != V->ne[0] || 
        Q->ne[1] != K->ne[1] || Q->ne[1] != V->ne[1]) {
        std::cerr << "Error: Dimensiones incompatibles en Q, K o V" << std::endl;
        return nullptr;
    }

    printf("DEBUG - Input dimensions:\n");
    printf("  Q: [%ld, %ld, %ld, %ld]\n", Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3]);
    printf("  K: [%ld, %ld, %ld, %ld]\n", K->ne[0], K->ne[1], K->ne[2], K->ne[3]);
    printf("  V: [%ld, %ld, %ld, %ld]\n", V->ne[0], V->ne[1], V->ne[2], V->ne[3]);

    // 2. Calcular puntuaciones de atención QK^T
    // Transponer K para que la multiplicación sea correcta
    ggml_tensor* K_transposed = ggml_permute(ctx, K, 1, 0, 2, 3);   // Transpone los dos primeros ejes
    K_transposed = ggml_cont(ctx, K_transposed);                    // Asegurar que no quede como "transposed"

    printf("DEBUG - K_transposed: [%ld, %ld, %ld, %ld]\n", 
           K_transposed->ne[0], K_transposed->ne[1], K_transposed->ne[2], K_transposed->ne[3]);

    debug_mul_mat_detailed("scores", Q, K_transposed);
    ggml_tensor* scores = ggml_mul_mat(ctx, Q, K_transposed);

    printf("DEBUG - scores after QK^T: [%ld, %ld, %ld, %ld]\n", 
           scores->ne[0], scores->ne[1], scores->ne[2], scores->ne[3]);

    // 3. Escalar las puntuaciones
    const float scaling_factor = (scale_factor == 0.0f)
        ? 1.0f / sqrtf(static_cast<float>(Q->ne[0]))
        : scale_factor;
    scores = ggml_scale(ctx, scores, scaling_factor);

    // 4. Aplicar máscaras de atención
    if (is_causal) {
        scores = ggml_diag_mask_inf(ctx, scores, 0);  // Máscara causal estándar
    }

    if (attention_mask != nullptr) {
        scores = ggml_add(ctx, scores, attention_mask);  // Máscara adicional
    }

    // 5. Aplicar softmax para obtener pesos de atención
    ggml_tensor* attn_weights = ggml_soft_max(ctx, scores);

    printf("DEBUG - attn_weights: [%ld, %ld, %ld, %ld]\n", 
           attn_weights->ne[0], attn_weights->ne[1], attn_weights->ne[2], attn_weights->ne[3]);

    // 6. Transponer pesos de atención para la multiplicación con V
    ggml_tensor* attn_weights_transposed = ggml_permute(ctx, attn_weights, 1, 0, 2, 3);
    attn_weights_transposed = ggml_cont(ctx, attn_weights_transposed);

    printf("DEBUG - attn_weights_transposed: [%ld, %ld, %ld, %ld]\n", 
           attn_weights_transposed->ne[0], attn_weights_transposed->ne[1], 
           attn_weights_transposed->ne[2], attn_weights_transposed->ne[3]);

    // 7. Multiplicar por los valores V
    debug_mul_mat_detailed("output", attn_weights_transposed, V);
    ggml_tensor* output = ggml_mul_mat(ctx, attn_weights_transposed, V);

    printf("DEBUG - output final: [%ld, %ld, %ld, %ld]\n", 
           output->ne[0], output->ne[1], output->ne[2], output->ne[3]);

    return output;
}


/**
 * Normalización de capa mejorada para múltiples arquitecturas
 * 
 * @param ctx Contexto GGML
 * @param input Tensor de entrada
 * @param weight Tensor de pesos (scale) - puede ser NULL para normalización sin parámetros
 * @param bias Tensor de biases (shift) - puede ser NULL si no se usan biases
 * @param use_rmsnorm true para RMSNorm (LLaMA), false para LayerNorm (ViT, Whisper)
 * @param eps Valor épsilon para estabilidad numérica
 * @return Tensor normalizado
 */
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias, bool use_rmsnorm, float eps) {
    //Aplicar normalización base
    ggml_tensor* normalized;
    if (use_rmsnorm) {
        // RMSNorm (usado en LLaMA, DeepSeek)
        normalized = ggml_rms_norm(ctx, input, eps);
    } else {
        // LayerNorm clásico (usado en ViT, Whisper)
        normalized = ggml_norm(ctx, input, eps);
        
        // Para LayerNorm, añadir el centrado (restar media)
        ggml_tensor* mean = ggml_mean(ctx, input);
        normalized = ggml_sub(ctx, input, mean);
    }

    //Aplicar transformación affine (scale y shift) si hay parámetros
    if (weight) {
        normalized = ggml_mul(ctx, normalized, weight);
    }

    if (bias && !use_rmsnorm) { // RMSNorm normalmente no usa bias
        normalized = ggml_add(ctx, normalized, bias);
    }

    return normalized;
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
    
    debug_mul_mat_detailed("output_feed_fordward", weight, input);
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

        // Permute to [x, 1, seq_len, 1]
        input = ggml_permute(ctx, input, 0, 2, 1, 3);

        // Crea un tensor para almacenar las posiciones (índices de secuencia) como I32
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input->ne[2]);
        
        // COPIAR USANDO int32_t EXPLÍCITAMENTE (igual que en tokens_tensor)
        int32_t* positions_data = (int32_t*)positions->data;
        for (int i = 0; i < input->ne[2]; i++) {
            positions_data[i] = static_cast<int32_t>(i); // Convertir a int32
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

/**
 * Módulo Feed-Forward Network para LLaMA (SwiGLU)
 */
ggml_tensor* llama_ffn(ggml_context* ctx, ggml_tensor* input, ggml_tensor* gate_proj, ggml_tensor* up_proj, ggml_tensor* down_proj) {
    // Implementación SwiGLU
    ggml_tensor* gate = feed_forward(ctx, input, gate_proj, nullptr, "swiglu");

    debug_mul_mat_detailed("up", up_proj, input);
    ggml_tensor* up = ggml_mul_mat(ctx, up_proj, input);

    
    
    ggml_tensor* ffn_gate = ggml_mul(ctx, gate, up);


    debug_mul_mat_detailed("ffn_down_proj", down_proj, ffn_gate);
    ggml_tensor* ffn_down_proj = ggml_mul_mat(ctx, down_proj, ffn_gate);


    //return ggml_mul_mat(ctx, down_proj, ggml_mul(ctx, gate, up));
    return ffn_down_proj;
}


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
    return multi_head_attention(ctx, Q, K, V, false, nullptr, 0.0f);
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


// Módulo de Atención Multi-Cabeza Latente (DeepSeek)
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
debug_mul_mat_detailed("Q_proj", W_Q, Q);
ggml_tensor * Q_proj = ggml_mul_mat(ctx, W_Q, Q); // Q_proj = Q * W_Q

debug_mul_mat_detailed("K_proj", W_K, K);
ggml_tensor * K_proj = ggml_mul_mat(ctx, W_K, K); // K_proj = K * W_K

debug_mul_mat_detailed("V_proj", W_V, V);
ggml_tensor * V_proj = ggml_mul_mat(ctx, W_V, V); // V_proj = V * W_V

// Calcula las puntuaciones de atención (scores) como el producto escalar entre Q_proj y K_proj.
debug_mul_mat_detailed("scores", Q_proj, K_proj);
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
debug_mul_mat_detailed("output", attn_weights, V_proj);
ggml_tensor * output = ggml_mul_mat(ctx, attn_weights, V_proj); // output = attn_weights * V_proj

// Crea una matriz de pesos para proyectar la salida de la atención en el espacio final.
ggml_tensor * W_O = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Q->ne[1], latent_dim);

// Proyecta la salida de la atención usando la matriz de pesos W_O.
debug_mul_mat_detailed("output_proj", W_O, output);
ggml_tensor * output_proj = ggml_mul_mat(ctx, W_O, output); // output_proj = output * W_O

// Retorna el resultado final de la atención multi-cabeza latente.
return output_proj;
}






void debug_mul_mat_detailed(const char* name, ggml_tensor* A, ggml_tensor* B) {
    printf("DEBUG mul_mat %s:\n", name);
    printf("  A: [%ld, %ld, %ld, %ld]\n", A->ne[0], A->ne[1], A->ne[2], A->ne[3]);
    printf("  B: [%ld, %ld, %ld, %ld]\n", B->ne[0], B->ne[1], B->ne[2], B->ne[3]);
    
    bool dim0_ok = (A->ne[0] == B->ne[0]);
    bool dim2_ok = (B->ne[2] % A->ne[2] == 0);
    bool dim3_ok = (B->ne[3] % A->ne[3] == 0);
    
    printf("  Requirements:\n");
    printf("    A->ne[0] (%ld) == B->ne[0] (%ld) = %s\n", A->ne[0], B->ne[0], dim0_ok ? "OK" : "FAIL");
    printf("    B->ne[2] (%ld) %% A->ne[2] (%ld) = %ld = %s\n", B->ne[2], A->ne[2], B->ne[2] % A->ne[2], dim2_ok ? "OK" : "FAIL");
    printf("    B->ne[3] (%ld) %% A->ne[3] (%ld) = %ld = %s\n", B->ne[3], A->ne[3], B->ne[3] % A->ne[3], dim3_ok ? "OK" : "FAIL");
    printf("  ggml_can_mul_mat = %s\n", (dim0_ok && dim2_ok && dim3_ok) ? "true" : "false");
}
