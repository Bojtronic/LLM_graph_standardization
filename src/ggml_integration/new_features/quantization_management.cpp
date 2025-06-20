#include "quantization_management.h"
#include <stdexcept>
#include <cstring>

// Returns the size of one super-block for each quantization type
size_t get_k_quant_super_block_size(ggml_type type) {
    switch(type) {
        case GGML_TYPE_Q2_K: return sizeof(block_q2_K);
        case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_Q8_K: return sizeof(block_q8_K);
        default: return 0;
    }
}

// Validates tensor dimensions and size for its quantization type
bool validate_quant_tensor(const GGUFTensor& tensor) {
    const size_t sb_size = get_k_quant_super_block_size(tensor.type);
    if (sb_size == 0) {
        throw std::runtime_error("Unsupported quantization type: " + 
                               std::string(ggml_type_name(tensor.type)));
    }
    
    if (tensor.size % sb_size != 0) {
        throw std::runtime_error("Invalid tensor size for " + 
                               std::string(ggml_type_name(tensor.type)) + 
                               ": " + std::to_string(tensor.size) + 
                               " not multiple of " + std::to_string(sb_size));
    }
    
    size_t total_dims = 1;
    for (auto dim : tensor.dims) total_dims *= dim;
    
    const size_t expected_weights = (tensor.size / sb_size) * QK_K;
    if (total_dims != expected_weights) {
        throw std::runtime_error("Dimension mismatch for " + 
                               std::string(ggml_type_name(tensor.type)) + 
                               ": expected " + std::to_string(expected_weights) + 
                               " weights, got " + std::to_string(total_dims));
    }
    
    return true;
}

// Creates and initializes a GGML tensor with K-type quantized data
ggml_tensor* create_k_quant_tensor(ggml_context* ctx, const GGUFTensor& tensor_info) {
    validate_quant_tensor(tensor_info);

    ggml_tensor* tensor = nullptr;
    switch (tensor_info.n_dims) {
        case 1:
            tensor = ggml_new_tensor_1d(ctx, tensor_info.type, tensor_info.dims[0]);
            break;
        case 2:
            tensor = ggml_new_tensor_2d(ctx, tensor_info.type, 
                                      tensor_info.dims[0], tensor_info.dims[1]);
            break;
        default:
            throw std::runtime_error("Unsupported dimensionality for K-quant tensor");
    }

    const auto* data = tensor_info.get_data<uint8_t>();
    if (!data || data->size() != tensor_info.size) {
        throw std::runtime_error("Data size mismatch for " + tensor_info.name + 
                               ": expected " + std::to_string(tensor_info.size) + 
                               ", got " + (data ? std::to_string(data->size()) : "0"));
    }
    
    memcpy(tensor->data, data->data(), data->size());
    return tensor;
}

// ==================== Optimized Dequantization Functions ====================

namespace {
    // Helper function for processing 4 values at once
    inline void process_4_values(float* dst, int base_idx, float scale, float min, 
                               uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3) {
        dst[base_idx + 0] = scale * v0 + min;
        dst[base_idx + 1] = scale * v1 + min;
        dst[base_idx + 2] = scale * v2 + min;
        dst[base_idx + 3] = scale * v3 + min;
    }

    // Helper function for processing 8 values (for 4-bit quantization)
    inline void process_8_values(float* dst, int base_idx, float scale, float min,
                               uint8_t q0, uint8_t q1, uint8_t scale_idx, 
                               const uint8_t* scales, const uint8_t* scale_mins) {
        const uint8_t scale_shift = scale_idx%8 < 4 ? 0 : 4;
        const float curr_scale = scale * ((scales[scale_idx/8] >> scale_shift) & 0xF);
        const float curr_min = min * ((scale_mins[scale_idx/8] >> scale_shift) & 0xF);
        
        dst[base_idx + 0] = curr_scale * (q0 & 0xF) + curr_min;
        dst[base_idx + 1] = curr_scale * (q0 >> 4)   + curr_min;
        dst[base_idx + 2] = curr_scale * (q1 & 0xF) + curr_min;
        dst[base_idx + 3] = curr_scale * (q1 >> 4)   + curr_min;
    }
}

void dequantize_q2_k(const void* src, float* dst, int k) {
    const int nb = k / QK_K;
    const block_q2_K* x = static_cast<const block_q2_K*>(src);
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const float min = ggml_fp16_to_fp32(x[i].dmin);
        
        for (int j = 0; j < QK_K/4; j++) {
            const uint8_t q = x[i].qs[j];
            process_4_values(dst, QK_K*i + 4*j, d, min, 
                           q & 0x3, (q >> 2) & 0x3, 
                           (q >> 4) & 0x3, q >> 6);
        }
    }
}

void dequantize_q3_k(const void* src, float* dst, int k) {
    const int nb = k / QK_K;
    const block_q3_K* x = static_cast<const block_q3_K*>(src);
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const uint8_t* scales = x[i].scales;
        
        for (int j = 0; j < QK_K/4; ) {
            const uint8_t scale_shift = 2*(j/8) + ((j%8)/4);
            const float block_scale = (scales[scale_shift >> 1] >> (4*(scale_shift & 1))) & 0x3F;
            const float scale = d * (block_scale - 32);
            
            const uint8_t q0 = x[i].qs[j];
            const uint8_t q1 = x[i].qs[j+1];
            
            process_4_values(dst, QK_K*i + 4*j, scale, 0,
                           (q0 & 0x7) - 4, ((q0 >> 3) & 0x7) - 4,
                           ((q0 >> 6) | ((q1 << 2) & 0x7)) - 4,
                           ((q1 >> 1) & 0x7) - 4);
            
            j += (j % 4 == 3) ? 2 : 1;
        }
    }
}

void dequantize_q4_k(const void* src, float* dst, int k) {
    const int nb = k / QK_K;
    const block_q4_K* x = static_cast<const block_q4_K*>(src);
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const float min = ggml_fp16_to_fp32(x[i].dmin);
        
        for (int j = 0; j < QK_K/8; j++) {
            process_8_values(dst, QK_K*i + 8*j, d, min,
                           x[i].qs[2*j], x[i].qs[2*j+1], j,
                           x[i].scales, x[i].scales + K_SCALE_SIZE/2);
        }
    }
}

void dequantize_q5_k(const void* src, float* dst, int k) {
    const int nb = k / QK_K;
    const block_q5_K* x = static_cast<const block_q5_K*>(src);
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const float min = ggml_fp16_to_fp32(x[i].dmin);
        
        for (int j = 0; j < QK_K/8; j++) {
            const uint8_t qh = x[i].qh[j/4] >> (2*(j%4));
            const uint8_t ql0 = x[i].qs[2*j];
            const uint8_t ql1 = x[i].qs[2*j+1];
            
            const uint8_t scale_shift = j%8 < 4 ? 0 : 4;
            const float curr_scale = d * ((x[i].scales[j/8] >> scale_shift) & 0xF);
            const float curr_min = min * ((x[i].scales[j/8+8] >> scale_shift) & 0xF);
            
            dst[QK_K*i + 8*j + 0] = curr_scale * ((ql0 & 0xF) | ((qh & 0x3) << 4)) + curr_min;
            dst[QK_K*i + 8*j + 1] = curr_scale * ((ql0 >> 4) | ((qh & 0xC) << 2)) + curr_min;
            dst[QK_K*i + 8*j + 2] = curr_scale * ((ql1 & 0xF) | ((qh & 0x30) >> 0)) + curr_min;
            dst[QK_K*i + 8*j + 3] = curr_scale * ((ql1 >> 4) | ((qh & 0xC0) >> 2)) + curr_min;
        }
    }
}

void dequantize_q6_k(const void* src, float* dst, int k) {
    const int nb = k / QK_K;
    const block_q6_K* x = static_cast<const block_q6_K*>(src);
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        
        for (int j = 0; j < QK_K/16; j++) {
            const float scale = d * x[i].scales[j];
            
            for (int l = 0; l < 8; l++) {
                const uint8_t ql = x[i].ql[16*j + l];
                const uint8_t qh = x[i].qh[8*j + l/2];
                const uint8_t shift = 4*(l%2);
                
                dst[QK_K*i + 16*j + 2*l + 0] = scale * ((ql & 0xF) | ((qh >> shift) & 0x30));
                dst[QK_K*i + 16*j + 2*l + 1] = scale * ((ql >> 4) | ((qh >> (shift+2)) & 0x30));
            }
        }
    }
}

void dequantize_q8_k(const void* src, float* dst, int k) {
    const int nb = k / QK_K;
    const block_q8_K* x = static_cast<const block_q8_K*>(src);
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        
        for (int j = 0; j < QK_K; j++) {
            dst[QK_K*i + j] = d * x[i].qs[j];
        }
    }
}

void dequantize_k_quant(ggml_type type, const void* src, float* dst, int k) {
    if (k % QK_K != 0) {
        throw std::runtime_error("Input size must be multiple of QK_K (" + 
                               std::to_string(QK_K) + ")");
    }

    switch(type) {
        case GGML_TYPE_Q2_K: dequantize_q2_k(src, dst, k); break;
        case GGML_TYPE_Q3_K: dequantize_q3_k(src, dst, k); break;
        case GGML_TYPE_Q4_K: dequantize_q4_k(src, dst, k); break;
        case GGML_TYPE_Q5_K: dequantize_q5_k(src, dst, k); break;
        case GGML_TYPE_Q6_K: dequantize_q6_k(src, dst, k); break;
        case GGML_TYPE_Q8_K: dequantize_q8_k(src, dst, k); break;
        default:
            throw std::runtime_error("Unsupported K-type quantization: " + 
                                   std::string(ggml_type_name(type)));
    }
}
