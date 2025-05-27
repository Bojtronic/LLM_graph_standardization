#include "model_runner.h"
#include <iostream>
#include <ggml-cuda.h>
#include <ggml-cpu.h>

#include "model_modules.h"
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

// Variables globales para el estado del chat
static std::vector<int> g_context_tokens;
static gguf_context* g_tokenizer_ctx = nullptr;
static int g_eos_token = -1;

// Función para tokenizar entrada
std::vector<int> tokenize_input(const std::string& input, const gguf_context* ctx) {
    std::vector<int> tokens;
    
    // En una implementación real, usarías el tokenizador del modelo
    // Esto es un placeholder que simula tokenización básica
    const char* text = input.c_str();
    size_t text_len = input.length();
    
    // Buscar vocabulario en el contexto GGUF
    int vocab_key = gguf_find_key(ctx, "tokenizer.ggml.vocab");
    if (vocab_key == -1) {
        std::cerr << "No se encontró el vocabulario en el modelo\n";
        return tokens;
    }
    
    // Simulación de tokenización (dividir por espacios)
    std::string current_word;
    for (size_t i = 0; i <= text_len; ++i) {
        if (i == text_len || isspace(text[i])) {
            if (!current_word.empty()) {
                // Hash simple para simular ID de token
                int token_id = 0;
                for (char c : current_word) {
                    token_id = token_id * 256 + c;
                }
                token_id = abs(token_id) % 32000; // Limitamos al tamaño del vocabulario
                tokens.push_back(token_id);
                current_word.clear();
            }
        } else {
            current_word += text[i];
        }
    }
    
    return tokens;
}

// Función para decodificar tokens a texto
std::string decode_output(const std::vector<int>& tokens, const gguf_context* ctx) {
    std::string output;
    
    // En una implementación real, usarías el detokenizador del modelo
    for (int token : tokens) {
        // Simulación de detokenización
        char c = token % 256;
        if (isprint(c)) {
            output += c;
        } else {
            output += "[" + std::to_string(token) + "]";
        }
    }
    
    return output;
}

// Función para muestrear el siguiente token
int sample_next_token(const float* logits, int n_vocab, float temperature, float top_p, int top_k) {
    std::vector<float> probs(logits, logits + n_vocab);
    
    // Aplicar temperatura
    if (temperature != 1.0f) {
        for (float& p : probs) {
            p /= temperature;
        }
    }
    
    // Softmax
    float max_logit = *std::max_element(probs.begin(), probs.end());
    float sum_exp = 0.0f;
    for (float& p : probs) {
        p = expf(p - max_logit);
        sum_exp += p;
    }
    for (float& p : probs) {
        p /= sum_exp;
    }
    
    // Top-k sampling
    if (top_k > 0 && top_k < n_vocab) {
        std::vector<int> indices(n_vocab);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
            [&probs](int a, int b) { return probs[a] > probs[b]; });
        
        // Cero las probabilidades fuera del top-k
        for (int i = top_k; i < n_vocab; ++i) {
            probs[indices[i]] = 0.0f;
        }
    }
    
    // Nucleus sampling (top-p)
    if (top_p < 1.0f) {
        std::vector<int> indices(n_vocab);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&probs](int a, int b) { return probs[a] > probs[b]; });
        
        float cumulative = 0.0f;
        int last_idx = 0;
        for (int i = 0; i < n_vocab; ++i) {
            cumulative += probs[indices[i]];
            if (cumulative >= top_p) {
                last_idx = i;
                break;
            }
        }
        
        // Cero las probabilidades fuera del top-p
        for (int i = last_idx + 1; i < n_vocab; ++i) {
            probs[indices[i]] = 0.0f;
        }
    }
    
    // Remuestrear para normalizar
    sum_exp = std::accumulate(probs.begin(), probs.end(), 0.0f);
    for (float& p : probs) {
        p /= sum_exp;
    }
    
    // Muestrear según las probabilidades
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen);
}

// Función principal del chat interactivo
void run_interactive_chat(const ModelParams& params) {
    std::cout << "\n=== Modo Chat Interactivo ===\n";
    std::cout << "Escribe tu mensaje (o 'salir' para terminar):\n\n";

    // Inicializar el contexto del tokenizador
    g_tokenizer_ctx = gguf_init_from_file(params.model_path.c_str(), {});
    if (!g_tokenizer_ctx) {
        std::cerr << "Error al cargar el modelo\n";
        return;
    }

    // Obtener token EOS del modelo
    int eos_key = gguf_find_key(g_tokenizer_ctx, "tokenizer.eos_token_id");
    if (eos_key != -1) {
        g_eos_token = gguf_get_val_u32(g_tokenizer_ctx, eos_key);
    } else {
        g_eos_token = 2; // Valor por defecto común
    }

    // Inicializar backend
    ggml_backend_t backend = params.use_gpu ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    if (!backend) {
        std::cerr << "Error al inicializar el backend\n";
        gguf_free(g_tokenizer_ctx);
        return;
    }

    char input_buffer[1024];
    while (true) {
        std::cout << "> ";
        std::cin.getline(input_buffer, sizeof(input_buffer));
        std::string user_input(input_buffer);

        if (user_input == "salir" || user_input == "exit") {
            break;
        }

        // 1. Tokenizar entrada del usuario
        std::vector<int> input_tokens = tokenize_input(user_input, g_tokenizer_ctx);
        
        // Agregar al contexto (incluyendo tokens previos)
        g_context_tokens.insert(g_context_tokens.end(), input_tokens.begin(), input_tokens.end());
        
        // Limitar al contexto máximo
        if (g_context_tokens.size() > params.n_ctx) {
            int excess = g_context_tokens.size() - params.n_ctx;
            g_context_tokens.erase(g_context_tokens.begin(), g_context_tokens.begin() + excess);
        }

        // 2. Generar respuesta
        std::vector<int> response_tokens;
        bool generating = true;
        ggml_context* ctx = ggml_init({.mem_size = 16 * 1024 * 1024});

        std::cout << "Asistente: ";
        while (generating && response_tokens.size() < params.n_ctx) {
            // Ejecutar el modelo con el contexto actual
            ggml_tensor* logits_tensor = run_llama_model(ctx, backend, params, g_context_tokens);
            
            if (!logits_tensor) {
                std::cerr << "Error en la ejecución del modelo\n";
                break;
            }
            
            // Muestrear próximo token
            float* logits = ggml_get_data_f32(logits_tensor);
            int next_token = sample_next_token(logits, /* n_vocab */ 32000, 
                                            params.temperature, params.top_p, params.top_k);
            
            // Verificar fin de generación
            if (next_token == g_eos_token) {
                generating = false;
            } else {
                response_tokens.push_back(next_token);
                g_context_tokens.push_back(next_token);
                
                // Mostrar token decodificado
                std::cout << decode_output({next_token}, g_tokenizer_ctx) << std::flush;
            }
        }
        std::cout << "\n\n";

        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    gguf_free(g_tokenizer_ctx);
    g_tokenizer_ctx = nullptr;
    g_context_tokens.clear();
}

ggml_tensor* get_layer_tensor(ggml_context* ctx, const GraphData& graph_data, const std::string& name) {
    for (const auto& tensor : graph_data.tensors) {
        if (tensor.name.find(name) != std::string::npos) {
            ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tensor.dims[0], tensor.dims[1]);
            // Aquí deberías cargar los pesos reales del tensor
            return t;
        }
    }
    return nullptr;
}

ggml_tensor* run_llama_model(ggml_context* ctx, ggml_backend_t backend, const ModelParams& params, const std::vector<int>& input_tokens) {
    // 1. Convertir tokens de entrada a tensor GGML
    ggml_tensor* tokens_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_tokens.size());
    if (!tokens_tensor) {
        std::cerr << "Error al crear tensor de tokens" << std::endl;
        return nullptr;
    }
    memcpy(tokens_tensor->data, input_tokens.data(), input_tokens.size() * sizeof(int));

    // 2. Cargar pesos del modelo
    GraphData graph_data = gguf_graph_data(g_tokenizer_ctx, params.model_path.c_str());
    
    // 3. Obtener embeddings de tokens
    ggml_tensor* token_embd = nullptr;
    for (const auto& tensor : graph_data.tensors) {
        if (tensor.name.find("token_embd") != std::string::npos) {
            token_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tensor.dims[0], tensor.dims[1]);
            if (!token_embd) {
                std::cerr << "Error al crear tensor de embeddings" << std::endl;
                return nullptr;
            }
            // Aquí deberías cargar los pesos reales del embedding
            // ggml_set_tensor_data(token_embd, tensor.data);
            break;
        }
    }
    
    if (!token_embd) {
        std::cerr << "Error: Token embeddings not found in model" << std::endl;
        return nullptr;
    }
    
    // 4. Obtener embeddings para los tokens de entrada
    ggml_tensor* current = ggml_get_rows(ctx, token_embd, tokens_tensor);
    if (!current) {
        std::cerr << "Error en ggml_get_rows" << std::endl;
        return nullptr;
    }
    
    // 5. Aplicar codificación posicional
    current = positional_encoding(ctx, current, "rope", current->ne[0], 0, 10000.0f);
    if (!current) {
        std::cerr << "Error en positional_encoding" << std::endl;
        return nullptr;
    }
    
    // 6. Procesar a través de las capas del transformer
    int num_layers = 32; // Deberías obtener esto del modelo
    for (int i = 0; i < num_layers; ++i) {
        // Atención
        ggml_tensor* attn_norm = layer_norm(ctx, current, true, 1e-5f);
        if (!attn_norm) {
            std::cerr << "Error en layer_norm (attn)" << std::endl;
            return nullptr;
        }
        
        // Obtener pesos para Q, K, V de esta capa
        ggml_tensor* Wq = get_layer_tensor(ctx, graph_data, "blk." + std::to_string(i) + ".attn_q");
        ggml_tensor* Wk = get_layer_tensor(ctx, graph_data, "blk." + std::to_string(i) + ".attn_k");
        ggml_tensor* Wv = get_layer_tensor(ctx, graph_data, "blk." + std::to_string(i) + ".attn_v");
        
        if (!Wq || !Wk || !Wv) {
            std::cerr << "Error al obtener pesos de atención" << std::endl;
            return nullptr;
        }
        
        ggml_tensor* q = ggml_mul_mat(ctx, Wq, attn_norm);
        ggml_tensor* k = ggml_mul_mat(ctx, Wk, attn_norm);
        ggml_tensor* v = ggml_mul_mat(ctx, Wv, attn_norm);
        
        if (!q || !k || !v) {
            std::cerr << "Error en multiplicación de matrices (Q/K/V)" << std::endl;
            return nullptr;
        }
        
        ggml_tensor* attention = multi_head_attention(ctx, q, k, v, true);
        if (!attention) {
            std::cerr << "Error en multi_head_attention" << std::endl;
            return nullptr;
        }
        
        current = ggml_add(ctx, current, attention);
        if (!current) {
            std::cerr << "Error en suma residual (attention)" << std::endl;
            return nullptr;
        }
        
        // Feed Forward
        ggml_tensor* ffn_norm = layer_norm(ctx, current, true, 1e-5f);
        if (!ffn_norm) {
            std::cerr << "Error en layer_norm (ffn)" << std::endl;
            return nullptr;
        }
        
        ggml_tensor* W1 = get_layer_tensor(ctx, graph_data, "blk." + std::to_string(i) + ".ffn_up");
        ggml_tensor* W2 = get_layer_tensor(ctx, graph_data, "blk." + std::to_string(i) + ".ffn_gate");
        ggml_tensor* W3 = get_layer_tensor(ctx, graph_data, "blk." + std::to_string(i) + ".ffn_down");
        
        if (!W1 || !W2 || !W3) {
            std::cerr << "Error al obtener pesos FFN" << std::endl;
            return nullptr;
        }
        
        ggml_tensor* ffn_gate = feed_forward(ctx, ffn_norm, W2, nullptr, "swiglu");
        ggml_tensor* ffn_up = feed_forward(ctx, ffn_norm, W1, nullptr, "silu");
        ggml_tensor* ffn_down = feed_forward(ctx, ggml_mul(ctx, ffn_gate, ffn_up), W3, nullptr, "linear");
        
        if (!ffn_gate || !ffn_up || !ffn_down) {
            std::cerr << "Error en capas FFN" << std::endl;
            return nullptr;
        }
        
        current = ggml_add(ctx, current, ffn_down);
        if (!current) {
            std::cerr << "Error en suma residual (ffn)" << std::endl;
            return nullptr;
        }
    }
    
    // 7. Normalización final
    current = layer_norm(ctx, current, true, 1e-5f);
    if (!current) {
        std::cerr << "Error en layer_norm final" << std::endl;
        return nullptr;
    }
    
    // 8. Proyección a vocabulario
    ggml_tensor* output_proj = get_layer_tensor(ctx, graph_data, "output");
    if (!output_proj) {
        std::cerr << "Error al obtener proyección de salida" << std::endl;
        return nullptr;
    }
    
    ggml_tensor* output = ggml_mul_mat(ctx, output_proj, current);
    if (!output) {
        std::cerr << "Error en multiplicación de matrices final" << std::endl;
        return nullptr;
    }
    
    // 9. Ejecutar el grafo computacional
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    ggml_backend_graph_compute(backend, gf);
    
    return output;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


bool run_vit_model(ggml_context* ctx, ggml_backend_t backend, const ModelParams& params) {
    std::cout << "Initializing ViT model..." << std::endl;
    
    GraphData graph_data = gguf_graph_data(gguf_init_from_file(params.model_path.c_str(), {}), params.model_path.c_str());
    
    ggml_tensor* input_image = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 
                                               params.image_size, params.image_size, 3, 1);
    
    ggml_tensor* patch_emb = ggml_conv_2d(ctx, /* patch_proj */, input_image, 16, 16, 0, 0);
    patch_emb = ggml_reshape_3d(ctx, patch_emb, patch_emb->ne[0], patch_emb->ne[1], patch_emb->ne[2]*patch_emb->ne[3]);
    
    ggml_tensor* embeddings = class_token(ctx, patch_emb);
    embeddings = positional_encoding(ctx, embeddings, "sinusoidal", embeddings->ne[0], 0, 10000.0f);
    
    for (int i = 0; i < /* num_layers from metadata */; ++i) {
        ggml_tensor* norm1 = layer_norm(ctx, embeddings, false, 1e-6f);
        ggml_tensor* q = ggml_mul_mat(ctx, /* Wq */, norm1);
        ggml_tensor* k = ggml_mul_mat(ctx, /* Wk */, norm1);
        ggml_tensor* v = ggml_mul_mat(ctx, /* Wv */, norm1);
        
        ggml_tensor* attention = multi_head_attention(ctx, q, k, v, false);
        attention = ggml_mul_mat(ctx, /* out_proj */, attention);
        embeddings = ggml_add(ctx, embeddings, attention);
        
        ggml_tensor* norm2 = layer_norm(ctx, embeddings, false, 1e-6f);
        ggml_tensor* mlp = feed_forward(ctx, norm2, /* fc1 */, /* b1 */, "gelu");
        mlp = feed_forward(ctx, mlp, /* fc2 */, /* b2 */, "linear");
        embeddings = ggml_add(ctx, embeddings, mlp);
    }
    
    ggml_tensor* cls_output = ggml_view_1d(ctx, embeddings, embeddings->ne[0], 0);
    cls_output = layer_norm(ctx, cls_output, false, 1e-6f);
    ggml_tensor* output = ggml_mul_mat(ctx, /* classifier */, cls_output);
    
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    ggml_backend_graph_compute(backend, gf);
    
    std::cout << "ViT model execution completed" << std::endl;
    return true;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


bool run_whisper_model(ggml_context* ctx, ggml_backend_t backend, const ModelParams& params) {
    std::cout << "Initializing Whisper model..." << std::endl;
    
    GraphData graph_data = gguf_graph_data(gguf_init_from_file(params.model_path.c_str(), {}), params.model_path.c_str());
    
    ggml_tensor* input_audio = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, params.n_mels, params.n_audio_ctx, 1);
    
    ggml_tensor* audio_emb = ggml_conv_1d(ctx, /* conv1 */, input_audio, 1, 1, 0);
    audio_emb = ggml_gelu(ctx, audio_emb);
    audio_emb = ggml_conv_1d(ctx, /* conv2 */, audio_emb, 1, 1, 0);
    audio_emb = ggml_gelu(ctx, audio_emb);
    
    audio_emb = positional_encoding(ctx, audio_emb, "sinusoidal", audio_emb->ne[0], 0, 10000.0f);
    
    for (int i = 0; i < /* num_encoder_layers */; ++i) {
        ggml_tensor* norm1 = layer_norm(ctx, audio_emb, false, 1e-5f);
        ggml_tensor* q = ggml_mul_mat(ctx, /* Wq */, norm1);
        ggml_tensor* k = ggml_mul_mat(ctx, /* Wk */, norm1);
        ggml_tensor* v = ggml_mul_mat(ctx, /* Wv */, norm1);
        
        ggml_tensor* self_attn = multi_head_attention(ctx, q, k, v, false);
        self_attn = ggml_mul_mat(ctx, /* out_proj */, self_attn);
        audio_emb = ggml_add(ctx, audio_emb, self_attn);
        
        ggml_tensor* norm2 = layer_norm(ctx, audio_emb, false, 1e-5f);
        ggml_tensor* mlp = feed_forward(ctx, norm2, /* fc1 */, /* b1 */, "gelu");
        mlp = feed_forward(ctx, mlp, /* fc2 */, /* b2 */, "linear");
        audio_emb = ggml_add(ctx, audio_emb, mlp);
    }
    
    ggml_tensor* tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, params.n_audio_ctx);
    ggml_tensor* token_emb = ggml_get_rows(ctx, /* token_emb */, tokens);
    
    for (int i = 0; i < /* num_decoder_layers */; ++i) {
        ggml_tensor* norm1 = layer_norm(ctx, token_emb, false, 1e-5f);
        ggml_tensor* q = ggml_mul_mat(ctx, /* Wq */, norm1);
        ggml_tensor* k = ggml_mul_mat(ctx, /* Wk */, norm1);
        ggml_tensor* v = ggml_mul_mat(ctx, /* Wv */, norm1);
        
        ggml_tensor* self_attn = multi_head_attention(ctx, q, k, v, true);
        token_emb = ggml_add(ctx, token_emb, self_attn);
        
        ggml_tensor* norm2 = layer_norm(ctx, token_emb, false, 1e-5f);
        ggml_tensor* cross_attn = cross_attention(ctx, norm2, audio_emb, audio_emb);
        token_emb = ggml_add(ctx, token_emb, cross_attn);
        
        ggml_tensor* norm3 = layer_norm(ctx, token_emb, false, 1e-5f);
        ggml_tensor* mlp = feed_forward(ctx, norm3, /* fc1 */, /* b1 */, "gelu");
        mlp = feed_forward(ctx, mlp, /* fc2 */, /* b2 */, "linear");
        token_emb = ggml_add(ctx, token_emb, mlp);
    }
    
    ggml_tensor* output = ggml_mul_mat(ctx, /* lm_head */, token_emb);
    
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    ggml_backend_graph_compute(backend, gf);
    
    std::cout << "Whisper model execution completed" << std::endl;
    return true;
}

