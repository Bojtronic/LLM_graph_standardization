#include "model_runner.h"
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

std::vector<int> tokenize_input(const std::string& input, const GraphData& graph_data) {
    // Buscar el vocabulario en los metadatos
    const GGUFMetadata* tokens_meta = graph_data.find_metadata("tokenizer.ggml.tokens");
    
    if (!tokens_meta || tokens_meta->type != GGUF_TYPE_ARRAY || 
        !std::holds_alternative<std::vector<std::string>>(tokens_meta->array.data)) {
        std::cerr << "No se encontró el vocabulario, usando tokenización básica\n";
        return tokenize_basic(input);
    }

    const auto& vocab = std::get<std::vector<std::string>>(tokens_meta->array.data);
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
            const GGUFMetadata* unk_meta = graph_data.find_metadata("tokenizer.ggml.unknown_token_id");
            int unk_token = unk_meta ? unk_meta->value.i32 : 0;
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

std::string decode_output(const std::vector<int>& tokens, const GraphData& graph_data) {
    // Buscar el vocabulario en los metadatos
    const GGUFMetadata* tokens_meta = graph_data.find_metadata("tokenizer.ggml.tokens");
    
    if (!tokens_meta || tokens_meta->type != GGUF_TYPE_ARRAY || 
        !std::holds_alternative<std::vector<std::string>>(tokens_meta->array.data)) {
        return decode_basic(tokens);
    }

    const auto& vocab = std::get<std::vector<std::string>>(tokens_meta->array.data);
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

void run_interactive_chat(const ModelParams& params, GraphData graph_data) {
    
    /*
    // Cargar los datos del modelo GGUF en el grafo
    GraphData graph_data = gguf_graph_data(gguf_init_from_file(params.model_path.c_str(), {}), params.model_path.c_str());
    
    if (graph_data.tensors.empty()) {
        std::cerr << "Error: No se cargaron tensores del modelo\n";
        return;
    }
    */

    // Configurar tokens especiales desde los metadatos
    const GGUFMetadata* eos_meta = graph_data.find_metadata("tokenizer.ggml.eos_token_id");
    const GGUFMetadata* bos_meta = graph_data.find_metadata("tokenizer.ggml.bos_token_id");
    
    g_eos_token = eos_meta ? eos_meta->value.i32 : 2;
    int bos_token = bos_meta ? bos_meta->value.i32 : 1;

    // Inicializar backend
    ggml_backend_t backend = params.use_gpu ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    if (!backend) {
        std::cerr << "Error al inicializar el backend\n";
        return;
    }

    // Bucle principal del chat
    char input_buffer[1024];
    while (true) {
        std::cout << "> ";
        std::cin.getline(input_buffer, sizeof(input_buffer));
        std::string user_input(input_buffer);

        if (user_input == "salir" || user_input == "exit") break;

        // 1. Tokenizar la entrada del usuario
        std::vector<int> input_tokens = tokenize_input(user_input, graph_data);
        
        // Agregar token BOS al inicio si está configurado
        if (bos_token != -1) {
            input_tokens.insert(input_tokens.begin(), bos_token);
        }

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
            ggml_tensor* logits_tensor = run_llama_model(ctx, backend, params, graph_data, g_context_tokens);
            
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
                std::cout << decode_output({next_token}, graph_data) << std::flush;
            }
        }
        std::cout << "\n\n";

        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    g_context_tokens.clear();
}

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
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K: {
            if (const auto* data = tensor_info->get_data<uint8_t>()) {
                memcpy(tensor->data, data->data(), data->size());
            }
            break;
        }
        default:
            std::cerr << "Tipo de tensor no soportado: " << ggml_type_name(tensor_info->type) << std::endl;
            return nullptr;
    }
    
    return tensor;
}

ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight, ggml_tensor* bias, float eps) {
    if (!weight) {
        std::cerr << "Error: Peso de normalización es NULL\n";
        return nullptr;
    }
    
    // Normalización RMS (sin bias)
    ggml_tensor* rms = ggml_rms_norm(ctx, input, eps);
    return ggml_mul(ctx, rms, weight);
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

