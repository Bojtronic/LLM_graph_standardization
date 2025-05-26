#include "model_runner.h"
#include <iostream>
#include <ggml-cuda.h>
#include <ggml-cpu.h>

// Variables globales para el estado del chat
static std::vector<int> g_context_tokens;
static gguf_context* g_tokenizer_ctx = nullptr;
static int g_eos_token = -1;

void run_interactive_chat(const ModelParams& params) {
    std::cout << "\n=== Modo Chat Interactivo ===\n";
    std::cout << "Escribe tu mensaje (o 'salir' para terminar):\n\n";

    //GraphData graph_data = gguf_graph_data(gguf_init_from_file(params.model_path.c_str(), {}), params.model_path.c_str());
    
    // Inicializar el contexto del tokenizador (simplificado)
    g_tokenizer_ctx = gguf_init_from_file(params.model_path.c_str(), {});
    if (!g_tokenizer_ctx) {
        std::cerr << "Error al cargar el tokenizador del modelo\n";
        return;
    }

    // Obtener token EOS del modelo
    int eos_key = gguf_find_key(g_tokenizer_ctx, "tokenizer.eos_token_id");
    if (eos_key != -1) {
        g_eos_token = gguf_get_val_u32(g_tokenizer_ctx, eos_key);
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
        ggml_backend_t backend = params.use_gpu ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
        ggml_context* ctx = ggml_init({.mem_size = 16 * 1024 * 1024});

        std::cout << "Asistente: ";
        while (generating && response_tokens.size() < params.n_ctx) {
            // Ejecutar el modelo con el contexto actual
            ggml_tensor* logits_tensor = run_llama_model(ctx, backend, params, g_context_tokens);
            
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
        ggml_backend_free(backend);
    }

    gguf_free(g_tokenizer_ctx);
    g_tokenizer_ctx = nullptr;
    g_context_tokens.clear();
}

std::vector<int> tokenize_input(const std::string& input, const gguf_context* ctx) {
    // Implementación simplificada - en realidad necesitarías el tokenizador real
    // Esto es solo un ejemplo conceptual
    std::vector<int> tokens;
    
    // Buscar el tokenizador en el contexto GGUF (simplificado)
    int tokenizer_key = gguf_find_key(ctx, "tokenizer");
    if (tokenizer_key != -1) {
        // En una implementación real, usarías la API del tokenizador aquí
        tokens.push_back(123); // Token ficticio para demostración
    }
    
    return tokens;
}

std::string decode_output(const std::vector<int>& tokens, const gguf_context* ctx) {
    // Implementación simplificada - deberías usar el tokenizador real
    std::string result;
    
    for (int token : tokens) {
        // En una implementación real, usarías el tokenizador para decodificar
        result += " palabra" + std::to_string(token); // Ejemplo ficticio
    }
    
    return result;
}

int sample_next_token(const float* logits, int n_vocab, float temperature, float top_p, int top_k) {
    // Implementación simplificada de muestreo
    // En una implementación real usarías softmax con temperatura, top-p, etc.
    return 0; // Token ficticio
}

bool run_llama_model(ggml_context* ctx, ggml_backend_t backend, const ModelParams& params, const std::vector<int>& input_tokens) {
    std::cout << "Initializing LLaMA model..." << std::endl;

    // Convertir input_tokens a tensor GGML
    ggml_tensor* tokens_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_tokens.size());
    memcpy(tokens_tensor->data, input_tokens.data(), input_tokens.size() * sizeof(int));

    
    GraphData graph_data = gguf_graph_data(gguf_init_from_file(params.model_path.c_str(), {}), params.model_path.c_str());
    
    ggml_tensor* input_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, params.n_ctx);
    ggml_tensor* token_embd = nullptr;
    
    for (const auto& tensor : graph_data.tensors) {
        if (tensor.name.find("token_embd") != std::string::npos) {
            token_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tensor.dims[0], tensor.dims[1]);
            break;
        }
    }
    
    if (!token_embd) {
        std::cerr << "Error: Token embeddings not found in model" << std::endl;
        return false;
    }
    
    ggml_tensor* current = ggml_get_rows(ctx, token_embd, input_tokens);
    current = positional_encoding(ctx, current, "rope", current->ne[0], 0, 10000.0f);
    
    for (int i = 0; i < /* num_layers from metadata */; ++i) {
        ggml_tensor* attn_norm = layer_norm(ctx, current, true, 1e-5f);
        ggml_tensor* q = ggml_mul_mat(ctx, /* Wq */, attn_norm);
        ggml_tensor* k = ggml_mul_mat(ctx, /* Wk */, attn_norm);
        ggml_tensor* v = ggml_mul_mat(ctx, /* Wv */, attn_norm);
        
        ggml_tensor* attention = multi_head_attention(ctx, q, k, v, true);
        current = ggml_add(ctx, current, attention);
        
        ggml_tensor* ffn_norm = layer_norm(ctx, current, true, 1e-5f);
        ggml_tensor* ffn_gate = feed_forward(ctx, ffn_norm, /* W1 */, /* b1 */, "swiglu");
        ggml_tensor* ffn_up = feed_forward(ctx, ffn_norm, /* W2 */, /* b2 */, "silu");
        ggml_tensor* ffn_down = feed_forward(ctx, ggml_mul(ctx, ffn_gate, ffn_up), /* W3 */, /* b3 */, "linear");
        
        current = ggml_add(ctx, current, ffn_down);
    }
    
    current = layer_norm(ctx, current, true, 1e-5f);
    ggml_tensor* output = ggml_mul_mat(ctx, /* output_proj */, current);
    
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);
    ggml_backend_graph_compute(backend, gf);
    

    // Retornar los logits del último token
    // return output; // ggml_tensor* con los logits
    
    std::cout << "LLaMA model execution completed" << std::endl;
    return true;
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

