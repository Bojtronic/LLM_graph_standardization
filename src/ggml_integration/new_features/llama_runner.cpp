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
#include "quantization_management.h"

// Variables globales para el estado del chat
// static std::vector<int> g_context_tokens;
// static int g_eos_token = -1;

std::vector<int> tokenize_input(const std::string &input, const std::string &model_filename)
{
    // Buscar el vocabulario en los metadatos
    GGUFMetadata tokens_meta = read_metadata(model_filename, "tokenizer.ggml.tokens");

    if (tokens_meta.type != GGUF_TYPE_ARRAY ||
        !std::holds_alternative<std::vector<std::string>>(tokens_meta.array.data))
    {
        std::cerr << "No se encontró el vocabulario, usando tokenización básica\n";
        return tokenize_basic(input);
    }

    const auto &vocab = std::get<std::vector<std::string>>(tokens_meta.array.data);
    std::vector<int> tokens;
    size_t pos = 0;

    while (pos < input.size())
    {
        size_t longest_match = 0;
        int best_token = -1;

        // Buscar el token más largo que coincida
        for (int i = 0; i < vocab.size(); ++i)
        {
            const std::string &token = vocab[i];
            if (token.empty())
                continue;

            if (input.compare(pos, token.size(), token) == 0 && token.size() > longest_match)
            {
                longest_match = token.size();
                best_token = i;
            }
        }

        if (best_token != -1)
        {
            tokens.push_back(best_token);
            pos += longest_match;
        }
        else
        {
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
std::vector<int> tokenize_basic(const std::string &input)
{
    std::vector<int> tokens;
    std::istringstream iss(input);
    std::string word;

    while (iss >> word)
    {
        // Hash simple para generar un ID de token
        int token_id = 0;
        for (char c : word)
        {
            token_id = token_id * 256 + c;
        }
        token_id = abs(token_id) % 32000;
        tokens.push_back(token_id);
    }

    return tokens;
}

/*
std::string decode_output_______(const std::vector<int>& tokens, const std::string& model_filename) {
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
*/

std::string decode_output(ggml_tensor *logits, const std::string &model_filename)
{
    // Verificar que el tensor es 2D [batch_size, vocab_size]
    /*
    if (logits->n_dims != 2) {
        std::cerr << "Error: Expected 2D logits tensor\n";
        return "[DECODE_ERROR]";
    }
    */

    // Obtener las dimensiones del tensor
    const int seq_len = logits->ne[0];
    const int vocab_size = logits->ne[1];

    // Convertir los logits a tokens (usando argmax)
    std::vector<int> tokens;
    float *logits_data = static_cast<float *>(logits->data);

    for (int i = 0; i < seq_len; ++i)
    {
        float *row = logits_data + i * vocab_size;
        int max_token = 0;
        float max_val = row[0];

        for (int j = 1; j < vocab_size; ++j)
        {
            if (row[j] > max_val)
            {
                max_val = row[j];
                max_token = j;
            }
        }
        tokens.push_back(max_token);
    }

    // Buscar el vocabulario en los metadatos
    GGUFMetadata tokens_meta = read_metadata(model_filename, "tokenizer.ggml.tokens");

    if (tokens_meta.type != GGUF_TYPE_ARRAY ||
        !std::holds_alternative<std::vector<std::string>>(tokens_meta.array.data))
    {
        return decode_basic(tokens);
    }

    const auto &vocab = std::get<std::vector<std::string>>(tokens_meta.array.data);
    std::string output;

    for (int token : tokens)
    {
        if (token >= 0 && token < vocab.size())
        {
            output += vocab[token];
        }
        else
        {
            output += "[UNK:" + std::to_string(token) + "]";
        }
    }

    return output;
}

// Función de respaldo para decodificación básica
std::string decode_basic(const std::vector<int> &tokens)
{
    std::string output;
    for (int token : tokens)
    {
        char c = token % 256;
        if (isprint(c))
        {
            output += c;
        }
        else
        {
            output += "[" + std::to_string(token) + "]";
        }
    }
    return output;
}

/**
 * Función para cargar y desquantizar un tensor a F32
 * Soporta solo tensores de 1D y 2D como se requiere
 *
 * @param ctx Contexto GGML
 * @param filename Ruta del archivo del modelo
 * @param tensor_name Nombre del tensor a cargar
 * @return ggml_tensor* Tensor desquantizado en F32, o nullptr en caso de error
 */
ggml_tensor *load_and_dequantize_to_f32(ggml_context *ctx,
                                        const std::string &filename,
                                        const std::string &tensor_name)
{

    // 1. Leer tensor GGUF
    GGUFTensor gguf_tensor = read_tensor(filename, tensor_name);
    
    if (gguf_tensor.name.empty())
    {
        std::cerr << "Error: Failed to load tensor '" << tensor_name << "'" << std::endl;
        return nullptr;
    }

    // 2. Validar dimensiones (solo 1D o 2D)
    size_t total_elements = 0;
    if (gguf_tensor.n_dims == 1)
    {
        total_elements = gguf_tensor.dims[0];
    }
    else if (gguf_tensor.n_dims == 2)
    {
        total_elements = gguf_tensor.dims[0] * gguf_tensor.dims[1];
    }
    else
    {
        std::cerr << "Error: Unsupported number of dimensions in tensor '" << tensor_name
                  << "': " << gguf_tensor.n_dims << " (only 1D and 2D supported)" << std::endl;
        return nullptr;
    }

    // 3. Verificar que las dimensiones sean válidas
    for (int i = 0; i < gguf_tensor.n_dims; i++)
    {
        if (gguf_tensor.dims[i] <= 0)
        {
            std::cerr << "Error: Tensor '" << tensor_name << "' has invalid dimension "
                      << i << ": " << gguf_tensor.dims[i] << std::endl;
            return nullptr;
        }
    }

    // 4. Crear tensor destino F32 con las mismas dimensiones
    ggml_tensor *tensor = nullptr;
    if (gguf_tensor.n_dims == 1)
    {
        tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, gguf_tensor.dims[0]);
    }
    else
    {
        tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, gguf_tensor.dims[0], gguf_tensor.dims[1]);
    }

    if (!tensor)
    {
        std::cerr << "Error: Could not allocate memory for tensor '" << tensor_name << "'" << std::endl;
        return nullptr;
    }

    

    // 5. Obtener datos crudos
    const void *raw_data = std::visit([](auto &&arg) -> const void *
                                      { return arg.empty() ? nullptr : arg.data(); }, gguf_tensor.data);

    if (!raw_data)
    {
        std::cerr << "Error: No data in tensor '" << tensor_name << "'" << std::endl;
        return nullptr;
    }

    // 6. Desquantizar según el tipo
    try
    {
        int k = 0;

        // Si es K-quant, calcular k desde el número de bloques
        if (ggml_is_quantized(gguf_tensor.type)) {

            size_t raw_size_bytes = gguf_tensor.size;
            size_t block_size = get_k_quant_super_block_size(gguf_tensor.type);

            if (raw_size_bytes % block_size != 0) {
                std::cerr << "Error: raw_data size is not aligned to block size in "
                        << tensor_name << std::endl;
                return nullptr;
            }

            size_t num_blocks = raw_size_bytes / block_size;
            k = num_blocks * QK_K;

            
        }
        else {
            // Para F32/F16 normales sí se usa dims[0] * dims[1]
            k = total_elements;
        }

        // Validar
        if (k != total_elements) {
            std::cerr << "Error: mismatch between dequantized size (k=" << k
                    << ") and expected tensor size (" << total_elements
                    << ") in tensor '" << tensor_name << "'" << std::endl;
            return nullptr;
        }

        // Ahora sí ejecutar dequantización
        dequantize_k_quant(gguf_tensor.type,
                        raw_data,
                        static_cast<float*>(tensor->data),
                        k);

        
  ////////////////////////////////////////////////////////////////////////      
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error dequantizing tensor '" << tensor_name << "': " << e.what() << std::endl;
        return nullptr;
    }

    return tensor;
}

ggml_tensor * load_weight_auto(
    ggml_context * ctx_weights,
    const std::string & model_filename,
    const std::string & name)
{
    GGUFTensor tinfo = read_tensor(model_filename, name);

    // Caso 1: ya es F32 → copiar directo
    if (tinfo.type == GGML_TYPE_F32) {

        ggml_tensor * t =
            ggml_new_tensor_1d(
                ctx_weights,
                GGML_TYPE_F32,
                tinfo.dims[0]
            );

        memcpy(
            t->data,
            std::get<std::vector<float>>(tinfo.data).data(),
            tinfo.dims[0] * sizeof(float)
        );

        return t;
    }

    // Caso 2: cuantizado → decuantizar
    return load_and_dequantize_to_f32(ctx_weights, model_filename, name);
}

bool run_llama_model(ggml_context *ctx, ggml_backend_t backend, const std::string &model_filename)
{

    int total_elements = 0;

    // Validación inicial de metadatos requeridos
    const std::vector<std::string> required_metadata = {
        "llama.embedding_length",
        "llama.attention.head_count",
        "llama.block_count",
        "llama.attention.layer_norm_rms_epsilon",
        "llama.context_length"};

    for (const auto &meta : required_metadata)
    {
        GGUFMetadata md = read_metadata(model_filename, meta);
        if (md.type == GGUF_TYPE_COUNT)
        { // Not found
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
        !std::holds_alternative<std::vector<std::string>>(tokens_meta.array.data))
    {
        std::cerr << "Error: Invalid or missing tokenizer vocabulary\n";
        return false;
    }
    int n_vocab = std::get<std::vector<std::string>>(tokens_meta.array.data).size();

    // std::vector<int> context_tokens;

    // Configurar tokens especiales
    GGUFMetadata eos_meta = read_metadata(model_filename, "tokenizer.ggml.eos_token_id");
    GGUFMetadata bos_meta = read_metadata(model_filename, "tokenizer.ggml.bos_token_id");

    int eos_token = eos_meta.type != GGUF_TYPE_COUNT ? eos_meta.value.i32 : 2;
    int bos_token = bos_meta.type != GGUF_TYPE_COUNT ? bos_meta.value.i32 : 1;

    char input_buffer[1024];
    std::cout << "> ";

    // Read input and check stream state
    if (!std::cin.getline(input_buffer, sizeof(input_buffer)))
    {
        if (std::cin.eof())
        {
            return false; // End of input (Ctrl+D/Ctrl+Z)
        }
        std::cerr << "Error: Failed to read input\n";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        return false;
    }

    // Check for empty input
    if (strlen(input_buffer) == 0)
    {
        std::cerr << "Error: Input cannot be empty\n";
        return false;
    }

    // Check for truncated input (buffer filled without finding '\n')
    if (strlen(input_buffer) == sizeof(input_buffer) - 1)
    {
        std::cerr << "Error: Input exceeds maximum length of "
                  << sizeof(input_buffer) - 1 << " characters\n";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        return false;
    }

    // Convert to std::string only if validation passed
    std::string user_input(input_buffer);

    // Exit commands
    if (user_input == "salir" || user_input == "exit")
    {
        return false;
    }

    // Tokenización
    std::vector<int> input_tokens;
    try
    {
        input_tokens = tokenize_input(user_input, model_filename);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error tokenizing input: " << e.what() << "\n";
        return false;
    }

    if (bos_token != -1)
    {
        input_tokens.insert(input_tokens.begin(), bos_token);
    }

    // Si se excede el tamaño del contexto n_ctx, se trunca manteniendo los tokens más recientes
    if (input_tokens.size() > n_ctx)
    {
        int excess = input_tokens.size() - n_ctx;
        input_tokens.erase(input_tokens.begin(), input_tokens.begin() + excess);
    }

    // Generación de respuesta
    std::vector<int> response_tokens;
    bool generating = true;

    std::cout << "Model response: \n";

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 1. Convertir input_tokens a tensor GGML
    ggml_tensor *tokens_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_tokens.size());
    if (!tokens_tensor)
    {
        std::cerr << "Error creating token tensor" << std::endl;
        return false;
    }

    // COPIAR USANDO int32_t EXPLÍCITAMENTE
    int32_t *tensor_data = (int32_t *)tokens_tensor->data;
    for (size_t i = 0; i < input_tokens.size(); i++)
    {
        tensor_data[i] = static_cast<int32_t>(input_tokens[i]);
    }

    // 2. Obtener embeddings de tokens
    ggml_tensor *token_embd = load_and_dequantize_to_f32(ctx, model_filename, "token_embd.weight");
    if (!token_embd)
    {
        std::cerr << "Error: Failed to load token embeddings" << std::endl;
        return false;
    }

    // 3. Aplicar embeddings
    if (*std::max_element(input_tokens.begin(), input_tokens.end()) >= token_embd->ne[1])
    {
        std::cerr << "Token index exceeds vocabulary size" << std::endl;
        return false;
    }

    // a=token_embd   b=tokens_tensor
    ggml_tensor *current = ggml_get_rows(ctx, token_embd, tokens_tensor);
    if (!current)
    {
        std::cerr << "Error in ggml_get_rows operation" << std::endl;
        return false;
    }

    // 4. Aplicar codificación posicional (RoPE)

    current = positional_encoding(
        ctx,
        current,
        "rope",          // Tipo RoPE
        n_embd / n_head, // Dimensiones por cabeza
        0,               // Modo (0 para implementación estándar)
        10000.0f         // Base de frecuencia
    );

    if (!current)
    {
        std::cerr << "Error applying positional encoding" << std::endl;
        return false;
    }

    // 5. Procesar cada capa del transformer
    for (int i = 0; i < n_layers; ++i)
    {
        printf("\n---- LAYER %d ----\n", i);

        std::string layer_prefix = "blk." + std::to_string(i) + ".";

        dbg_tensor("current (in)", current);

        // Atención - Norm Weight
        GGUFTensor attn_norm_weight_tensor = read_tensor(model_filename, layer_prefix + "attn_norm.weight");
        ggml_tensor *attn_norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, attn_norm_weight_tensor.dims[0]);
        memcpy(attn_norm_weight->data,
            std::get<std::vector<float>>(attn_norm_weight_tensor.data).data(),
            attn_norm_weight_tensor.dims[0] * sizeof(float));

        ggml_tensor *attn_norm_out = layer_norm(ctx, current, attn_norm_weight, nullptr, true, norm_eps);
        dbg_tensor("attn_norm_out", attn_norm_out);

        // Proyecciones Q, K, V
        auto load_proj = [&](const std::string &name) -> ggml_tensor *
        {
            ggml_tensor *tensor = load_and_dequantize_to_f32(ctx, model_filename, layer_prefix + name);
            if (!tensor)
                std::cerr << "Error loading " << name << " for layer " << i << std::endl;
            return tensor;
        };

        ggml_tensor *q_proj = load_proj("attn_q.weight");
        ggml_tensor *k_proj = load_proj("attn_k.weight");
        ggml_tensor *v_proj = load_proj("attn_v.weight");

        if (!q_proj || !k_proj || !v_proj)
            return false;

        dbg_tensor("q_proj", q_proj);
        dbg_tensor("k_proj", k_proj);
        dbg_tensor("v_proj", v_proj);

        ggml_tensor *q = ggml_mul_mat(ctx, q_proj, attn_norm_out);
        ggml_tensor *k = ggml_mul_mat(ctx, k_proj, attn_norm_out);
        ggml_tensor *v = ggml_mul_mat(ctx, v_proj, attn_norm_out);

        dbg_tensor("q (after mul)", q);
        dbg_tensor("k (after mul)", k);
        dbg_tensor("v (after mul)", v);

        // RoPE
        q = positional_encoding(ctx, q, "rope", n_embd / n_head, 0, 10000.0f);
        k = positional_encoding(ctx, k, "rope", n_embd / n_head, 0, 10000.0f);

        dbg_tensor("q (after rope)", q);
        dbg_tensor("k (after rope)", k);

        // Reorganizar tensores
        int head_dim = n_embd / n_head;
        q = ggml_reshape_3d(ctx, q, head_dim, n_head, input_tokens.size());
        k = ggml_reshape_3d(ctx, k, head_dim, n_head, input_tokens.size());
        v = ggml_reshape_3d(ctx, v, head_dim, n_head, input_tokens.size());

        dbg_tensor("q (3d)", q);
        dbg_tensor("k (3d)", k);
        dbg_tensor("v (3d)", v);

        // Atención multi-cabeza
        ggml_tensor *attn_output = multi_head_attention(
            ctx,
            q,
            k,
            v,
            true
        );

        dbg_tensor("attn_output (raw)", attn_output);

        attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3);
        attn_output = ggml_cont(ctx, attn_output);
        attn_output = ggml_reshape_2d(ctx, attn_output, n_embd, input_tokens.size());

        dbg_tensor("attn_output (2d)", attn_output);

        // Proyección de salida
        ggml_tensor *attn_weight =
            load_and_dequantize_to_f32(ctx, model_filename, layer_prefix + "attn_output.weight");

        if (!attn_weight) {
            std::cerr << "Error loading attn_output.weight for layer " << i << std::endl;
            return false;
        }

        dbg_tensor("attn_weight", attn_weight);

        attn_output = ggml_mul_mat(ctx, attn_weight, attn_output);
        dbg_tensor("attn_output (proj)", attn_output);

        current = ggml_permute(ctx, current, 0, 2, 1, 3);
        current = ggml_cont(ctx, current);
        dbg_tensor("current (before residual)", current);

        dbg_tensor("attn_output (before residual)", attn_output);

        printf("[DBG] trying residual add: current [%lld,%lld,%lld,%lld] + attn [%lld,%lld,%lld,%lld]\n",
            (long long)current->ne[0], (long long)current->ne[1],
            (long long)current->ne[2], (long long)current->ne[3],
            (long long)attn_output->ne[0], (long long)attn_output->ne[1],
            (long long)attn_output->ne[2], (long long)attn_output->ne[3]);

        // Conexión residual
        // Asegurar que current tenga la misma forma que attn_output
        if (current->ne[1] != attn_output->ne[1] ||
            current->ne[2] != attn_output->ne[2]) {

            printf("[FIX] aligning current to attn_output layout\n");
            current = ggml_permute(ctx, current, 0, 2, 1, 3);
            current = ggml_cont(ctx, current);
        }

        current = ggml_add(ctx, current, attn_output);
        dbg_tensor("current (after attn residual)", current);

        // Feed Forward Network
        GGUFTensor ffn_norm_weight_tensor =
            read_tensor(model_filename, layer_prefix + "ffn_norm.weight");

        ggml_tensor *ffn_norm_weight =
            ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ffn_norm_weight_tensor.dims[0]);

        memcpy(ffn_norm_weight->data,
            std::get<std::vector<float>>(ffn_norm_weight_tensor.data).data(),
            ffn_norm_weight_tensor.dims[0] * sizeof(float));

        ggml_tensor *ffn_norm_out =
            layer_norm(ctx, current, ffn_norm_weight, nullptr, true, norm_eps);

        dbg_tensor("ffn_norm_out", ffn_norm_out);

        ggml_tensor *ffn_gate = load_proj("ffn_gate.weight");
        ggml_tensor *ffn_up   = load_proj("ffn_up.weight");
        ggml_tensor *ffn_down = load_proj("ffn_down.weight");

        dbg_tensor("ffn_gate", ffn_gate);
        dbg_tensor("ffn_up",   ffn_up);
        dbg_tensor("ffn_down", ffn_down);

        if (!ffn_gate || !ffn_up || !ffn_down)
            return false;

        ggml_tensor *ffn_out =
            llama_ffn(ctx, ffn_norm_out, ffn_gate, ffn_up, ffn_down);

        dbg_tensor("ffn_out", ffn_out);

        // Conexión residual final
        dbg_tensor("current (before ffn residual)", current);
        dbg_tensor("ffn_out (before residual)", ffn_out);

        printf("[DBG] trying FFN residual add: current [%lld,%lld,%lld,%lld] + ffn [%lld,%lld,%lld,%lld]\n",
            (long long)current->ne[0], (long long)current->ne[1],
            (long long)current->ne[2], (long long)current->ne[3],
            (long long)ffn_out->ne[0], (long long)ffn_out->ne[1],
            (long long)ffn_out->ne[2], (long long)ffn_out->ne[3]);

        current = ggml_add(ctx, current, ffn_out);
        dbg_tensor("current (after ffn residual)", current);

        
    }

    
    

    // 6. Normalización final
    GGUFTensor output_norm_tensor = read_tensor(model_filename, "output_norm.weight");
    ggml_tensor *output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_norm_tensor.dims[0]);
    memcpy(output_norm->data,
        std::get<std::vector<float>>(output_norm_tensor.data).data(),
        output_norm_tensor.dims[0] * sizeof(float));


    
    current = layer_norm(ctx, current, output_norm, nullptr, true, norm_eps);

    // 7. Capa de salida (LM head)
    ggml_tensor *output_weight = load_and_dequantize_to_f32(ctx, model_filename, "output.weight");
    if (!output_weight)
    {
        std::cerr << "Error loading output.weight" << std::endl;
        ggml_free(ctx);
        return false;
    }


    debug_mul_mat_detailed_x("logits", output_weight, current);
    ggml_tensor *logits = ggml_mul_mat(ctx, output_weight, current);

    // 8. Construir y ejecutar el gráfico de computación
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);
    ggml_backend_graph_compute(backend, gf);

    // 9. Retornar solo los logits del último token
    // ggml_tensor* last_logits = ggml_view_1d(ctx, logits, n_vocab,
    //                                    (input_tokens.size() - 1) * n_vocab * sizeof(float));

    // return last_logits;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (!logits)
    {
        std::cerr << "\nError: Model execution failed\n";
        ggml_free(ctx);
        return false;
    }

    try
    {
        // std::cout << decode_output({next_token}, graph_data) << std::flush;
        std::cout << decode_output(logits, model_filename) << std::flush;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nError decoding output: " << e.what() << "\n";
        generating = false;
    }

    std::cout << "\n\n";

    ggml_free(ctx);
    input_tokens.clear();
    return true;
}


void dbg_tensor(const char *name, ggml_tensor *t) {
    if (!t) {
        printf("[DBG] %s = NULL\n", name);
        return;
    }
    printf("[DBG] %s: type=%d ne=[%lld,%lld,%lld,%lld]\n",
           name, t->type,
           (long long)t->ne[0],
           (long long)t->ne[1],
           (long long)t->ne[2],
           (long long)t->ne[3]);
}



ggml_tensor * ensure_compute_tensor(
    ggml_context * ctx_compute,
    ggml_tensor  * w)
{
    // Siempre duplicamos en ctx_compute para evitar mezclar contextos
    ggml_tensor * wc = ggml_dup_tensor(ctx_compute, w);
    ggml_set_name(wc, w->name);
    return wc;
}




ggml_tensor * ensure_2d(
    ggml_context * ctx,
    ggml_tensor  * t,
    int64_t        ne0,
    int64_t        ne1)
{
    if (t->ne[0] == ne0 &&
        t->ne[1] == ne1 &&
        t->ne[2] == 1 &&
        t->ne[3] == 1) {
        return t;
    }

    return ggml_reshape_2d(ctx, t, ne0, ne1);
}




void debug_mul_mat_detailed_x(const char* name, ggml_tensor* A, ggml_tensor* B) {
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
