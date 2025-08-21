bool run_llama_model(ggml_context *ctx, ggml_backend_t backend, const std::string &model_filename)
{
    // [Código inicial sin cambios...]
    
    // 1. Convertir input_tokens a tensor GGML (I32 - CORRECTO)
    ggml_tensor *tokens_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_tokens.size());
    if (!tokens_tensor)
    {
        std::cerr << "Error creating token tensor" << std::endl;
        return false;
    }
    memcpy(tokens_tensor->data, input_tokens.data(), input_tokens.size() * sizeof(int32_t));

    // 2. Obtener embeddings de tokens (CUANTIZADOS)
    GGUFTensor token_embd_tensor = read_tensor(model_filename, "token_embd.weight");
    if (token_embd_tensor.name.empty())
    {
        std::cerr << "Error: Failed to load token embeddings" << std::endl;
        return false;
    }

    // VERIFICAR: ¿El tensor ya está dequantizado o está cuantizado?
    if (ggml_is_quantized(token_embd_tensor.type)) {
        // 2a. DEQUANTIZAR los embeddings de tokens
        std::vector<float> dequantized_embeddings(token_embd_tensor.dims[0] * token_embd_tensor.dims[1]);
        
        dequantize_k_quant(token_embd_tensor.type,
                          std::get<std::vector<uint8_t>>(token_embd_tensor.data).data(),
                          dequantized_embeddings.data(),
                          token_embd_tensor.dims[0] * token_embd_tensor.dims[1]);

        // 2b. Crear tensor con datos dequantizados
        ggml_tensor *token_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                                   token_embd_tensor.dims[0], token_embd_tensor.dims[1]);
        memcpy(token_embd->data, dequantized_embeddings.data(), 
               dequantized_embeddings.size() * sizeof(float));
    } else {
        // 2c. Si ya es F32, copiar directamente (caso poco común con modelos cuantizados)
        ggml_tensor *token_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                                   token_embd_tensor.dims[0], token_embd_tensor.dims[1]);
        memcpy(token_embd->data, std::get<std::vector<float>>(token_embd_tensor.data).data(),
               token_embd_tensor.dims[0] * token_embd_tensor.dims[1] * sizeof(float));
    }

    // [Resto del código...]

    // 3. Modificar la función load_proj para manejar cuantización
    auto load_proj = [&](const std::string &name) -> ggml_tensor *
    {
        GGUFTensor proj_tensor = read_tensor(model_filename, layer_prefix + name);
        
        if (ggml_is_quantized(proj_tensor.type)) {
            // DEQUANTIZAR pesos cuantizados
            std::vector<float> dequantized_data(proj_tensor.dims[0] * proj_tensor.dims[1]);
            
            dequantize_k_quant(proj_tensor.type,
                              std::get<std::vector<uint8_t>>(proj_tensor.data).data(),
                              dequantized_data.data(),
                              proj_tensor.dims[0] * proj_tensor.dims[1]);

            ggml_tensor *proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                                 proj_tensor.dims[0], proj_tensor.dims[1]);
            memcpy(proj->data, dequantized_data.data(), 
                   dequantized_data.size() * sizeof(float));
            return proj;
        } else {
            // Ya está en float32
            ggml_tensor *proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                                 proj_tensor.dims[0], proj_tensor.dims[1]);
            memcpy(proj->data, std::get<std::vector<float>>(proj_tensor.data).data(),
                   proj_tensor.dims[0] * proj_tensor.dims[1] * sizeof(float));
            return proj;
        }
    };

    // 4. Modificar también para los otros tensores (norm weights, etc.)
    auto load_norm_weight = [&](const std::string &full_name) -> ggml_tensor *
    {
        GGUFTensor norm_tensor = read_tensor(model_filename, full_name);
        
        // Los pesos de normalización generalmente no están cuantizados, pero verificar
        if (ggml_is_quantized(norm_tensor.type)) {
            std::vector<float> dequantized_data(norm_tensor.dims[0]);
            dequantize_k_quant(norm_tensor.type,
                              std::get<std::vector<uint8_t>>(norm_tensor.data).data(),
                              dequantized_data.data(),
                              norm_tensor.dims[0]);
            
            ggml_tensor *norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, norm_tensor.dims[0]);
            memcpy(norm_weight->data, dequantized_data.data(), 
                   dequantized_data.size() * sizeof(float));
            return norm_weight;
        } else {
            ggml_tensor *norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, norm_tensor.dims[0]);
            memcpy(norm_weight->data, std::get<std::vector<float>>(norm_tensor.data).data(),
                   norm_tensor.dims[0] * sizeof(float));
            return norm_weight;
        }
    };

    // 5. Usar la nueva función para cargar pesos de normalización
    ggml_tensor *attn_norm_weight = load_norm_weight(layer_prefix + "attn_norm.weight");
    ggml_tensor *ffn_norm_weight = load_norm_weight(layer_prefix + "ffn_norm.weight");
    ggml_tensor *output_norm = load_norm_weight("output_norm.weight");

    // [Resto del código sin cambios...]
}









bool ggml_is_quantized(ggml_type type) {
    return type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K || 
           type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || 
           type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_K;
}

void dequantized_warning(const std::string& name, ggml_type type) {
    if (ggml_is_quantized(type)) {
        std::cout << "Tensor '" << name << "' type: " << ggml_type_name(type);
        std::cout << " (QUANTIZED - needs dequantization)";
    }
    std::cout << std::endl;
}

