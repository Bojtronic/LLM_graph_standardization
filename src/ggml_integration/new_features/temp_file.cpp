bool run_llama_model(ggml_context* ctx, ggml_backend_t backend, const std::string& model_filename) {
    // [Previous initialization code remains the same until tokenization...]

    // Generation loop
    std::vector<int> generated_tokens;
    int max_new_tokens = 256;  // Limit response length
    bool generating = true;
    int next_token = -1;

    std::cout << "Model response: ";

    while (generating && generated_tokens.size() < max_new_tokens) {
        // 1. Prepare input tokens (initial prompt + generated tokens)
        std::vector<int> current_tokens = input_tokens;
        if (!generated_tokens.empty()) {
            current_tokens.insert(current_tokens.end(), generated_tokens.begin(), generated_tokens.end());
        }

        // Truncate if exceeds context window
        if (current_tokens.size() > n_ctx) {
            current_tokens.erase(current_tokens.begin(), current_tokens.begin() + (current_tokens.size() - n_ctx));
        }

        // 2. Create input tensor
        ggml_tensor* tokens_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, current_tokens.size());
        memcpy(tokens_tensor->data, current_tokens.data(), current_tokens.size() * sizeof(int));

        // [Previous model processing code remains the same until logits...]

        // 3. Get logits and sample next token
        float* logits_data = static_cast<float*>(logits->data);
        int vocab_size = logits->ne[0];
        
        // Get last token logits
        float* last_logits = logits_data + (current_tokens.size() - 1) * vocab_size;
        
        // Simple argmax sampling (you might want to implement better sampling)
        next_token = std::max_element(last_logits, last_logits + vocab_size) - last_logits;

        // 4. Check for EOS or add to output
        if (next_token == eos_token) {
            generating = false;
        } else {
            generated_tokens.push_back(next_token);
            
            // Print token immediately (streaming output)
            try {
                std::string token_str = decode_output({next_token}, model_filename);
                std::cout << token_str << std::flush;
            } catch (...) {
                // Handle decoding error
            }
        }

        // Clean up for next iteration
        ggml_free(ctx);
        ctx = ggml_init({/* appropriate params */});
    }

    std::cout << "\n\n";
    ggml_free(ctx);
    return true;
}

std::string decode_output(const std::vector<int>& tokens, const std::string& model_filename) {
    // [Your existing decode_output implementation]
}




//-------------------------------------------------------------------------------------


case GGUF_TYPE_ARRAY:
    md.array.type = gguf_get_arr_type(ctx, i);
    md.array.size = gguf_get_arr_n(ctx, i);

    // Escribir tipo y tamaño del array
    out.write(reinterpret_cast<const char*>(&md.array.type), sizeof(md.array.type));
    out.put('\n');
    out.write(reinterpret_cast<const char*>(&md.array.size), sizeof(md.array.size));
    out.put('\n');

    // Manejar cada tipo de array
    switch (md.array.type) {
        case GGUF_TYPE_UINT8:
            md.array.data = read_array_data<uint8_t>(ctx, i, md.array.size);
            out.write(reinterpret_cast<const char*>(std::get<std::vector<uint8_t>>(md.array.data).data()), 
                     md.array.size * sizeof(uint8_t));
            break;
        case GGUF_TYPE_INT8:
            md.array.data = read_array_data<int8_t>(ctx, i, md.array.size);
            out.write(reinterpret_cast<const char*>(std::get<std::vector<int8_t>>(md.array.data).data()), 
                     md.array.size * sizeof(int8_t));
            break;
        case GGUF_TYPE_UINT16:
            md.array.data = read_array_data<uint16_t>(ctx, i, md.array.size);
            out.write(reinterpret_cast<const char*>(std::get<std::vector<uint16_t>>(md.array.data).data()), 
                     md.array.size * sizeof(uint16_t));
            break;
        case GGUF_TYPE_INT16:
            md.array.data = read_array_data<int16_t>(ctx, i, md.array.size);
            out.write(reinterpret_cast<const char*>(std::get<std::vector<int16_t>>(md.array.data).data()), 
                     md.array.size * sizeof(int16_t));
            break;
        case GGUF_TYPE_UINT32:
            md.array.data = read_array_data<uint32_t>(ctx, i, md.array.size);
            out.write(reinterpret_cast<const char*>(std::get<std::vector<uint32_t>>(md.array.data).data()), 
                     md.array.size * sizeof(uint32_t));
            break;
        case GGUF_TYPE_INT32:
            md.array.data = read_array_data<int32_t>(ctx, i, md.array.size);
            out.write(reinterpret_cast<const char*>(std::get<std::vector<int32_t>>(md.array.data).data()), 
                     md.array.size * sizeof(int32_t));
            break;
        case GGUF_TYPE_FLOAT32:
            md.array.data = read_array_data<float>(ctx, i, md.array.size);
            out.write(reinterpret_cast<const char*>(std::get<std::vector<float>>(md.array.data).data()), 
                     md.array.size * sizeof(float));
            break;
        case GGUF_TYPE_UINT64:
            md.array.data = read_array_data<uint64_t>(ctx, i, md.array.size);
            out.write(reinterpret_cast<const char*>(std::get<std::vector<uint64_t>>(md.array.data).data()), 
                     md.array.size * sizeof(uint64_t));
            break;
        case GGUF_TYPE_INT64:
            md.array.data = read_array_data<int64_t>(ctx, i, md.array.size);
            out.write(reinterpret_cast<const char*>(std::get<std::vector<int64_t>>(md.array.data).data()), 
                     md.array.size * sizeof(int64_t));
            break;
        case GGUF_TYPE_FLOAT64:
            md.array.data = read_array_data<double>(ctx, i, md.array.size);
            out.write(reinterpret_cast<const char*>(std::get<std::vector<double>>(md.array.data).data()), 
                     md.array.size * sizeof(double));
            break;
        case GGUF_TYPE_STRING: {
            std::vector<std::string> strings;
            strings.reserve(md.array.size);
            for (size_t j = 0; j < md.array.size; ++j) {
                const char* str = gguf_get_arr_str(ctx, i, j);
                std::string safe_str = str ? str : "";
                //strings.push_back(safe_str);
                strings.emplace_back(safe_str);

                out.write(safe_str.c_str(), safe_str.size());
                out.put('\0');
            }
            md.array.data = strings;
            
            
            break;
        }
        default:
            break;
    }
    out.put('\n'); // Separador final del array
    break;