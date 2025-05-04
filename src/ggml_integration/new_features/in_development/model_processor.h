#ifndef MODEL_PROCESSOR_FUNC_H
#define MODEL_PROCESSOR_FUNC_H

#include "graph_data_structs.h"
#include "model_modules.h"
#include "ggml.h"
#include <vector>
#include <string>
#include <stdexcept>

/**
 * @file model_processor_func.h
 * @brief Functional-style processor for GGUF model inference
 */

// Forward declarations
struct ModelMetadata {
    size_t n_ctx;
    size_t n_embd;
    size_t n_layer;
    size_t n_head;
    size_t n_vocab;
    float eps;
    bool use_rmsnorm;
    bool use_swiglu;
};

/**
 * @brief Validate model structure and extract metadata
 * @param model_data Loaded GGUF model data
 * @return Extracted metadata
 */
ModelMetadata model_processor_init(const GraphData& model_data);

/**
 * @brief Get a tensor by name and convert to ggml_tensor
 * @param ctx GGML context
 * @param model_data Model data
 * @param name Tensor name
 * @return ggml_tensor pointer
 */
ggml_tensor* model_processor_get_tensor(ggml_context* ctx, const GraphData& model_data, const std::string& name);

/**
 * @brief Process a single transformer layer
 * @param ctx GGML context
 * @param model_data Model data
 * @param metadata Model metadata
 * @param input Input tensor
 * @param layer_idx Layer index (0..n_layer-1)
 * @return Output tensor
 */
ggml_tensor* model_processor_layer(ggml_context* ctx, const GraphData& model_data, 
                                 const ModelMetadata& metadata, ggml_tensor* input, size_t layer_idx);

/**
 * @brief Process input tokens through the model
 * @param ctx GGML context
 * @param model_data Model data
 * @param metadata Model metadata
 * @param tokens Input token IDs
 * @return Output logits as float vector
 */
std::vector<float> model_processor_process(ggml_context* ctx, const GraphData& model_data,
                                        const ModelMetadata& metadata, const std::vector<int32_t>& tokens);

/**
 * @brief Convert ggml_tensor to float vector
 * @param tensor Input tensor
 * @return Vector of floats
 */
std::vector<float> model_processor_tensor_to_float(ggml_tensor* tensor);

// Implementation

ModelMetadata model_processor_init(const GraphData& model_data) {
    // Check required tensors exist
    const std::vector<std::string> required_tensors = {
        "token_embd.weight",
        "output.weight",
        "output_norm.weight"
    };

    for (const auto& name : required_tensors) {
        if (!model_data.find_tensor(name)) {
            throw std::runtime_error("Missing required tensor: " + name);
        }
    }

    // Helper function to get metadata values
    auto get_metadata = [&model_data](const std::string& key, auto default_val) {
        const GGUFMetadata* md = model_data.find_metadata(key);
        return md ? std::any_cast<decltype(default_val)>(md->value) : default_val;
    };

    return {
        .n_ctx = get_metadata("llama.context_length", 4096ul),
        .n_embd = get_metadata("llama.embedding_length", 4096ul),
        .n_layer = get_metadata("llama.block_count", 32ul),
        .n_head = get_metadata("llama.attention.head_count", 32ul),
        .n_vocab = get_metadata("tokenizer.ggml.tokens.size", 32000ul),
        .eps = get_metadata("llama.attention.layer_norm_rms_epsilon", 1e-5f),
        .use_rmsnorm = true,  // LLaMA uses RMSNorm
        .use_swiglu = true    // LLaMA uses SwiGLU
    };
}

ggml_tensor* model_processor_get_tensor(ggml_context* ctx, const GraphData& model_data, const std::string& name) {
    const GGUFTensor* tensor = model_data.find_tensor(name);
    if (!tensor) {
        throw std::runtime_error("Tensor not found: " + name);
    }

    // Convert to ggml_tensor
    ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, tensor->n_dims, tensor->dims.data());
    
    // Copy data based on quantization type
    if (const auto* data = tensor->get_data<float>()) {
        memcpy(result->data, data->data(), tensor->size);
    } else if (const auto* data = tensor->get_data<uint8_t>()) {
        // Handle quantized tensors (would need dequantization in real implementation)
        throw std::runtime_error("Quantized tensor support not implemented");
    } else {
        throw std::runtime_error("Unsupported tensor data type");
    }

    return result;
}

ggml_tensor* model_processor_layer(ggml_context* ctx, const GraphData& model_data, 
                                 const ModelMetadata& metadata, ggml_tensor* input, size_t layer_idx) {
    std::string prefix = "blk." + std::to_string(layer_idx) + ".";

    // Attention norm
    ggml_tensor* attn_norm = layer_norm(ctx, input, metadata.use_rmsnorm, metadata.eps);
    attn_norm = ggml_mul(ctx, attn_norm, 
                        model_processor_get_tensor(ctx, model_data, prefix + "attn_norm.weight"));

    // Attention
    ggml_tensor* Q = ggml_mul_mat(ctx, 
                                 model_processor_get_tensor(ctx, model_data, prefix + "attn_q.weight"), 
                                 attn_norm);
    ggml_tensor* K = ggml_mul_mat(ctx, 
                                 model_processor_get_tensor(ctx, model_data, prefix + "attn_k.weight"), 
                                 attn_norm);
    ggml_tensor* V = ggml_mul_mat(ctx, 
                                 model_processor_get_tensor(ctx, model_data, prefix + "attn_v.weight"), 
                                 attn_norm);
    
    ggml_tensor* attention_out = multi_head_attention(ctx, Q, K, V, true); // causal=true for LLaMA
    attention_out = ggml_mul_mat(ctx, 
                                model_processor_get_tensor(ctx, model_data, prefix + "attn_output.weight"), 
                                attention_out);
    
    // Residual connection
    ggml_tensor* attn_residual = ggml_add(ctx, input, attention_out);

    // FFN norm
    ggml_tensor* ffn_norm = layer_norm(ctx, attn_residual, metadata.use_rmsnorm, metadata.eps);
    ffn_norm = ggml_mul(ctx, ffn_norm, 
                       model_processor_get_tensor(ctx, model_data, prefix + "ffn_norm.weight"));

    // FFN (SwiGLU)
    ggml_tensor* gate = ggml_mul_mat(ctx, 
                                   model_processor_get_tensor(ctx, model_data, prefix + "ffn_gate.weight"), 
                                   ffn_norm);
    ggml_tensor* up = ggml_mul_mat(ctx, 
                                 model_processor_get_tensor(ctx, model_data, prefix + "ffn_up.weight"), 
                                 ffn_norm);
    ggml_tensor* ffn_out = ggml_swiglu(ctx, ggml_mul(ctx, gate, up));
    ffn_out = ggml_mul_mat(ctx, 
                          model_processor_get_tensor(ctx, model_data, prefix + "ffn_down.weight"), 
                          ffn_out);

    // Final residual
    return ggml_add(ctx, attn_residual, ffn_out);
}

std::vector<float> model_processor_process(ggml_context* ctx, const GraphData& model_data,
                                        const ModelMetadata& metadata, const std::vector<int32_t>& tokens) {
    // Get token embeddings
    ggml_tensor* token_embd = model_processor_get_tensor(ctx, model_data, "token_embd.weight");
    
    // Create tensor for input tokens
    ggml_tensor* tokens_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens.size());
    memcpy(tokens_tensor->data, tokens.data(), tokens.size() * sizeof(int32_t));
    
    // Get rows from embedding matrix
    ggml_tensor* input = ggml_get_rows(ctx, token_embd, tokens_tensor);

    // Process through all layers
    for (size_t i = 0; i < metadata.n_layer; i++) {
        input = model_processor_layer(ctx, model_data, metadata, input, i);
    }

    // Final norm
    input = layer_norm(ctx, input, metadata.use_rmsnorm, metadata.eps);
    input = ggml_mul(ctx, input, 
                    model_processor_get_tensor(ctx, model_data, "output_norm.weight"));

    // Output projection
    ggml_tensor* logits = ggml_mul_mat(ctx, 
                                      model_processor_get_tensor(ctx, model_data, "output.weight"), 
                                      input);

    // Convert to float vector
    return model_processor_tensor_to_float(logits);
}

std::vector<float> model_processor_tensor_to_float(ggml_tensor* tensor) {
    if (tensor->type != GGML_TYPE_F32) {
        throw std::runtime_error("Only F32 tensors can be converted");
    }

    size_t num_elements = 1;
    for (int i = 0; i < tensor->n_dims; i++) {
        num_elements *= tensor->ne[i];
    }

    const float* data = static_cast<const float*>(tensor->data);
    return std::vector<float>(data, data + num_elements);
}

#endif // MODEL_PROCESSOR_FUNC_H
