#include "llama_runner.h"
#include "graph_file.h" 
#include <iostream>
#include <ggml-cuda.h>
#include <ggml-cpu.h>

#include "model_modules.h"
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <sstream>

// Variables globales para el estado del chat
static std::vector<int> g_context_tokens;
static int g_eos_token = -1;

std::vector<int> tokenize_input(const std::string& input, const std::string& model_filename) {
    // Buscar el vocabulario en los metadatos
    GGUFMetadata tokens_meta = read_metadata(model_filename, "tokenizer.ggml.tokens");
    
    if (tokens_meta.type != GGUF_TYPE_ARRAY || 
        !std::holds_alternative<std::vector<std::string>>(tokens_meta.array.data)) {
        std::cerr << "No se encontró el vocabulario, usando tokenización básica\n";
        return tokenize_basic(input);
    }

    const auto& vocab = std::get<std::vector<std::string>>(tokens_meta.array.data);
    std::vector<int> tokens;
    size_t pos = 0;

    while (pos < input.size()) {
        size_t longest_match = 0;
        int best_token = -1;

        // Buscar el token más largo que coincida
        for (int i = 0; i < vocab.size(); ++i) {
            const std::string& token = vocab[i];
            if (token.empty()) continue;

            if (input.compare(pos, token.size(), token) == 0 && token.size() > longest_match) {
                longest_match = token.size();
                best_token = i;
            }
        }

        if (best_token != -1) {
            tokens.push_back(best_token);
            pos += longest_match;
        } else {
            // Token desconocido - usar token UNK si existe
            GGUFMetadata unk_meta = read_metadata(model_filename, "tokenizer.ggml.unknown_token_id");
            int unk_token = unk_meta.type != GGUF_TYPE_COUNT ? unk_meta.value.i32 : 0;
            tokens.push_back(unk_token);
            pos++; // Avanzar un carácter
        }
    }

    return tokens;
}

// Función de respaldo para tokenización básica
std::vector<int> tokenize_basic(const std::string& input) {
    std::vector<int> tokens;
    std::istringstream iss(input);
    std::string word;
    
    while (iss >> word) {
        // Hash simple para generar un ID de token
        int token_id = 0;
        for (char c : word) {
            token_id = token_id * 256 + c;
        }
        token_id = abs(token_id) % 32000;
        tokens.push_back(token_id);
    }
    
    return tokens;
}

std::string decode_output(const std::vector<int>& tokens, const std::string& model_filename) {
    // Buscar el vocabulario en los metadatos
    GGUFMetadata tokens_meta = read_metadata(model_filename, "tokenizer.ggml.tokens");
    
    if (tokens_meta.type != GGUF_TYPE_ARRAY || 
        !std::holds_alternative<std::vector<std::string>>(tokens_meta.array.data)) {
        return decode_basic(tokens);
    }

    const auto& vocab = std::get<std::vector<std::string>>(tokens_meta.array.data);
    std::string output;
    
    for (int token : tokens) {
        if (token >= 0 && token < vocab.size()) {
            output += vocab[token];
        } else {
            output += "[UNK:" + std::to_string(token) + "]";
        }
    }
    
    return output;
}

// Función de respaldo para decodificación básica
std::string decode_basic(const std::vector<int>& tokens) {
    std::string output;
    for (int token : tokens) {
        char c = token % 256;
        if (isprint(c)) {
            output += c;
        } else {
            output += "[" + std::to_string(token) + "]";
        }
    }
    return output;
}

bool run_interactive_chat(ggml_backend_t backend, const std::string& model_filename) {
    try {
        float temperature = 1.0f;
        int top_k = 0;
        float top_p = 1.0f;

        // Validación inicial de metadatos requeridos
        const std::vector<std::string> required_metadata = {
            "llama.embedding_length",
            "llama.attention.head_count",
            "llama.block_count",
            "llama.attention.layer_norm_rms_epsilon",
            "llama.context_length"
        };


        for (const auto& meta : required_metadata) {
            GGUFMetadata md = read_metadata(model_filename, meta);
            if (md.type == GGUF_TYPE_COUNT) { // Not found
                std::cerr << "Error: Missing required metadata '" << meta << "'\n";
                return false;
            }
        }

        // Extracción de parámetros con verificación
        int n_embd = read_metadata(model_filename, "llama.embedding_length").value.i32;
        int n_head = read_metadata(model_filename, "llama.attention.head_count").value.i32;
        int n_layers = read_metadata(model_filename, "llama.block_count").value.i32;
        float norm_eps = read_metadata(model_filename, "llama.attention.layer_norm_rms_epsilon").value.f32;
        int n_ctx = read_metadata(model_filename, "llama.context_length").value.i32;


        // Configuración del vocabulario
        GGUFMetadata tokens_meta = read_metadata(model_filename, "tokenizer.ggml.tokens");
        if (tokens_meta.type != GGUF_TYPE_ARRAY || 
            !std::holds_alternative<std::vector<std::string>>(tokens_meta.array.data)) {
            std::cerr << "Error: Invalid or missing tokenizer vocabulary\n";
            return false;
        }
        int vocab_size = std::get<std::vector<std::string>>(tokens_meta.array.data).size();

        // Configurar tokens especiales
        GGUFMetadata eos_meta = read_metadata(model_filename, "tokenizer.ggml.eos_token_id");
        GGUFMetadata bos_meta = read_metadata(model_filename, "tokenizer.ggml.bos_token_id");
        
        g_eos_token = eos_meta.type != GGUF_TYPE_COUNT ? eos_meta.value.i32 : 2;
        int bos_token = bos_meta.type != GGUF_TYPE_COUNT ? bos_meta.value.i32 : 1;


        // Crear contexto GGML una sola vez
        ggml_context* ctx = ggml_init({.mem_size = 16 * 1024 * 1024});
        if (!ctx) {
            std::cerr << "Error: Failed to initialize GGML context\n";
            ggml_free(ctx);
            return false;
        }

        // Bucle principal del chat
        char input_buffer[1024];
        
        std::cout << "> ";
            std::cin.getline(input_buffer, sizeof(input_buffer));
            std::string user_input(input_buffer);

            if (!std::cin) {
                return false;  // Fin de entrada o error
            }
            
            if (user_input == "salir" || user_input == "exit") {
                return false; 
            }
            

            // Tokenización con manejo de errores
            std::vector<int> input_tokens;
            try {
                //input_tokens = tokenize_input(user_input, graph_data);
                input_tokens = tokenize_input(user_input, model_filename);
            } catch (const std::exception& e) {
                std::cerr << "Error tokenizing input: " << e.what() << "\n";
                return false;
            }
            
            if (bos_token != -1) {
                input_tokens.insert(input_tokens.begin(), bos_token);
            }

            g_context_tokens.insert(g_context_tokens.end(), input_tokens.begin(), input_tokens.end());
            
            if (g_context_tokens.size() > n_ctx) {
                int excess = g_context_tokens.size() - n_ctx;
                g_context_tokens.erase(g_context_tokens.begin(), g_context_tokens.begin() + excess);
            }

            // Generación de respuesta
            std::vector<int> response_tokens;
            bool generating = true;
            
            std::cout << "Asistente: ";
            
            while (generating && (response_tokens.size() < n_ctx)) {
                // Si el contexto excede n_ctx, truncamos manteniendo los más recientes
                if (g_context_tokens.size() >= n_ctx) {
                    g_context_tokens.erase(g_context_tokens.begin(), 
                                        g_context_tokens.end() - n_ctx + 1);
                }

                /*
                if (response_tokens.size() % 32 == 0) {
                    ggml_free(ctx);
                    ctx = ggml_init({.mem_size = 16 * 1024 * 1024}); // Recrear el contexto
                }
                */

                ggml_tensor* logits_tensor = run_llama_model(ctx, backend, model_filename, 
                    n_embd, n_head, n_layers, norm_eps, n_ctx, vocab_size, g_context_tokens);
                
                if (!logits_tensor) {
                    std::cerr << "\nError: Model execution failed\n";
                    ggml_free(ctx);
                    return false;
                }
                
                float* logits = ggml_get_data_f32(logits_tensor);
                int next_token = sample_next_token(logits, vocab_size, temperature, top_p, top_k);

                if (next_token == g_eos_token) {
                    generating = false;
                } else {
                    response_tokens.push_back(next_token);
                    g_context_tokens.push_back(next_token);
                    
                    try {
                        //std::cout << decode_output({next_token}, graph_data) << std::flush;
                        std::cout << decode_output({next_token}, model_filename) << std::flush;
                    } catch (const std::exception& e) {
                        std::cerr << "\nError decoding output: " << e.what() << "\n";
                        generating = false;
                    }
                }
            }
            std::cout << "\n\n";

        ggml_free(ctx);
        g_context_tokens.clear();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error in chat: " << e.what() << "\n";
        return false;
    }
}



int sample_next_token(const float* logits, int n_vocab, 
                     float temperature, float top_p, int top_k) {
    std::vector<float> probs(logits, logits + n_vocab);
    
    // 1. Aplicar temperatura
    if (temperature != 1.0f) {
        for (float& prob : probs) {
            prob /= temperature;
        }
    }

    // 2. Softmax para convertir logits a probabilidades
    float max_logit = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (float& prob : probs) {
        prob = expf(prob - max_logit);
        sum += prob;
    }
    for (float& prob : probs) {
        prob /= sum;
    }

    // 3. Filtrado top-k
    if (top_k > 0 && top_k < n_vocab) {
        std::vector<std::pair<float, int>> prob_index;
        for (int i = 0; i < n_vocab; ++i) {
            prob_index.emplace_back(probs[i], i);
        }

        // Ordenar descendente por probabilidad
        std::partial_sort(
            prob_index.begin(),
            prob_index.begin() + top_k,
            prob_index.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        // Cero las probabilidades fuera del top-k
        for (int i = top_k; i < n_vocab; ++i) {
            probs[prob_index[i].second] = 0.0f;
        }

        // Renormalizar
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (float& prob : probs) {
            prob /= sum;
        }
    }

    // 4. Muestreo top-p (nucleus sampling)
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<float, int>> prob_index;
        for (int i = 0; i < n_vocab; ++i) {
            prob_index.emplace_back(probs[i], i);
        }

        // Ordenar descendente por probabilidad
        std::sort(
            prob_index.begin(),
            prob_index.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });

        float cumulative_prob = 0.0f;
        int last_idx = n_vocab - 1;
        for (int i = 0; i < n_vocab; ++i) {
            cumulative_prob += prob_index[i].first;
            if (cumulative_prob >= top_p) {
                last_idx = i;
                break;
            }
        }

        // Cero las probabilidades fuera del top-p
        for (int i = last_idx + 1; i < n_vocab; ++i) {
            probs[prob_index[i].second] = 0.0f;
        }

        // Renormalizar
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        for (float& prob : probs) {
            prob /= sum;
        }
    }

    // 5. Muestreo de la distribución
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return dist(gen);
}

ggml_tensor* run_llama_model(ggml_context* ctx, 
                            ggml_backend_t backend,
                            const std::string& model_filename,
                            int n_embd,
                            int n_head,
                            int n_layers,
                            float norm_eps,
                            int n_ctx,
                            int n_vocab,
                            const std::vector<int>& input_tokens) {

    // 1. Convertir input_tokens a tensor GGML
    ggml_tensor* tokens_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_tokens.size());
    if (!tokens_tensor) {
        std::cerr << "Error al crear tensor de tokens" << std::endl;
        return nullptr;
    }
    memcpy(tokens_tensor->data, input_tokens.data(), input_tokens.size() * sizeof(int));

    // 2. Obtener embeddings de tokens
    GGUFTensor token_embd_tensor = read_tensor(model_filename, "token_embd.weight");
    if (token_embd_tensor.name.empty()) {
        std::cerr << "Error: No se pudo cargar token embeddings" << std::endl;
        return nullptr;
    }
    
    // Convertir GGUFTensor a ggml_tensor
    ggml_tensor* token_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, token_embd_tensor.dims[0], token_embd_tensor.dims[1]);
    if (!token_embd) {
        std::cerr << "Error: No se pudo asignar memoria para embeddings (" 
                << token_embd_tensor.dims[0] << "x" << token_embd_tensor.dims[1] 
                << ")" << std::endl;
        return nullptr;
    }

    memcpy(token_embd->data, std::get<std::vector<float>>(token_embd_tensor.data).data(), 
           token_embd_tensor.dims[0] * token_embd_tensor.dims[1] * sizeof(float));



    // 3. Aplicar embeddings

    
    if (*std::max_element(input_tokens.begin(), input_tokens.end()) >= token_embd->ne[1]) {
        std::cerr << "Índice de token excede el tamaño del vocabulario" << std::endl;
        return nullptr;
    }



    // a=token_embd   b=tokens_tensor
    // a->ne[2] == b->ne[1]:
    // La dimensión 2 de a (por ejemplo, número de "bloques" o canales) debe coincidir con la dimensión 1 de b.
    // b->ne[3] == 1:
    // La cuarta dimensión de b debe ser 1 (es decir, b es un tensor 3D o inferior).
    // b->type == GGML_TYPE_I32:
    // Los índices en b deben ser enteros de 32 bits (I32).
    ggml_tensor* current = ggml_get_rows(ctx, token_embd, tokens_tensor);

    // 4. Aplicar codificación posicional (RoPE)
    
    current = positional_encoding(
        ctx,
        current,
        "rope",                     // Tipo RoPE
        n_embd / n_head, // Dimensiones por cabeza
        0,                          // Modo (0 para implementación estándar)
        10000.0f                   // Base de frecuencia
    );

    if (!current) {
        std::cerr << "Error al aplicar codificación posicional" << std::endl;
        return nullptr;
    }

    // 5. Procesar cada capa del transformer
    for (int i = 0; i < n_layers; ++i) {
        std::string layer_prefix = "blk." + std::to_string(i) + ".";

        // Atención
        GGUFTensor attn_norm_weight_tensor = read_tensor(model_filename, layer_prefix + "attn_norm.weight");
        ggml_tensor* attn_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, attn_norm_weight_tensor.dims[0]);
        memcpy(attn_norm_weight->data, std::get<std::vector<float>>(attn_norm_weight_tensor.data).data(), 
               attn_norm_weight_tensor.dims[0] * sizeof(float));

        ggml_tensor* attn_norm_out = layer_norm(ctx, current, attn_norm_weight, nullptr, true, norm_eps);
    

        // Proyecciones Q, K, V
        auto load_proj = [&](const std::string& name) -> ggml_tensor* {
            GGUFTensor proj_tensor = read_tensor(model_filename, layer_prefix + name);
            ggml_tensor* proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, proj_tensor.dims[0], proj_tensor.dims[1]);
            memcpy(proj->data, std::get<std::vector<float>>(proj_tensor.data).data(), 
                   proj_tensor.dims[0] * proj_tensor.dims[1] * sizeof(float));
            return proj;
        };

        ggml_tensor* q_proj = load_proj("attn_q.weight");
        ggml_tensor* k_proj = load_proj("attn_k.weight");
        ggml_tensor* v_proj = load_proj("attn_v.weight");


        ggml_tensor* q = ggml_mul_mat(ctx, q_proj, attn_norm_out);
        ggml_tensor* k = ggml_mul_mat(ctx, k_proj, attn_norm_out);
        ggml_tensor* v = ggml_mul_mat(ctx, v_proj, attn_norm_out);

        // Aplicar RoPE a Q y K
        q = positional_encoding(ctx, q, "rope", n_embd / n_head, 0, 10000.0f);
        k = positional_encoding(ctx, k, "rope", n_embd / n_head, 0, 10000.0f);

        // Reorganizar tensores para atención multi-cabeza
        int head_dim = n_embd / n_head;
        q = ggml_reshape_3d(ctx, q, head_dim, n_head, input_tokens.size());
        k = ggml_reshape_3d(ctx, k, head_dim, n_head, input_tokens.size());
        v = ggml_reshape_3d(ctx, v, head_dim, n_head, input_tokens.size());

        // Aplicar atención multi-cabeza
        ggml_tensor* attn_output = multi_head_attention(
            ctx,
            ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)),  // [seq_len, n_head, head_dim]
            ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)),  // [seq_len, n_head, head_dim]
            ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)),  // [seq_len, n_head, head_dim]
            true  // is_causal para modelos autoregresivos
        );

        // Reorganizar la salida
        attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3);
        attn_output = ggml_reshape_2d(ctx, attn_output, n_embd, input_tokens.size());

        // Proyección de salida
        GGUFTensor attn_proj_tensor = read_tensor(model_filename, layer_prefix + "attn_proj.weight");
        ggml_tensor* attn_proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, attn_proj_tensor.dims[0], attn_proj_tensor.dims[1]);
        memcpy(attn_proj->data, std::get<std::vector<float>>(attn_proj_tensor.data).data(), 
               attn_proj_tensor.dims[0] * attn_proj_tensor.dims[1] * sizeof(float));
        
        attn_output = ggml_mul_mat(ctx, attn_proj, attn_output);

        // Conexión residual
        current = ggml_add(ctx, current, attn_output);

        // Feed Forward Network
        GGUFTensor ffn_norm_weight_tensor = read_tensor(model_filename, layer_prefix + "ffn_norm.weight");
        ggml_tensor* ffn_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ffn_norm_weight_tensor.dims[0]);
        memcpy(ffn_norm_weight->data, std::get<std::vector<float>>(ffn_norm_weight_tensor.data).data(), 
               ffn_norm_weight_tensor.dims[0] * sizeof(float));

        ggml_tensor* ffn_norm_out = layer_norm(ctx, current, ffn_norm_weight, nullptr, true, norm_eps);

        // Capas FFN (SwishGLU)
        ggml_tensor* ffn_gate = load_proj("ffn_gate.weight");
        ggml_tensor* ffn_up = load_proj("ffn_up.weight");
        ggml_tensor* ffn_down = load_proj("ffn_down.weight");

        ggml_tensor* ffn_out = llama_ffn(ctx, ffn_norm_out, ffn_gate, ffn_up, ffn_down);

        // Conexión residual final
        current = ggml_add(ctx, current, ffn_out);
    }

    // 6. Normalización final
    GGUFTensor output_norm_tensor = read_tensor(model_filename, "output_norm.weight");
    ggml_tensor* output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_norm_tensor.dims[0]);
    memcpy(output_norm->data, std::get<std::vector<float>>(output_norm_tensor.data).data(), 
           output_norm_tensor.dims[0] * sizeof(float));
    
    current = layer_norm(ctx, current, output_norm, nullptr, true, norm_eps);

    // 7. Capa de salida (LM head)
    GGUFTensor output_weight_tensor = read_tensor(model_filename, "output.weight");
    ggml_tensor* output_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, output_weight_tensor.dims[0], output_weight_tensor.dims[1]);
    memcpy(output_weight->data, std::get<std::vector<float>>(output_weight_tensor.data).data(), 
           output_weight_tensor.dims[0] * output_weight_tensor.dims[1] * sizeof(float));
    
    ggml_tensor* logits = ggml_mul_mat(ctx, output_weight, current);

    // 8. Construir y ejecutar el gráfico de computación
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);
    ggml_backend_graph_compute(backend, gf);

    // 9. Retornar solo los logits del último token
    ggml_tensor* last_logits = ggml_view_1d(ctx, logits, n_vocab, 
                                        (input_tokens.size() - 1) * n_vocab * sizeof(float));

    return last_logits;
}

