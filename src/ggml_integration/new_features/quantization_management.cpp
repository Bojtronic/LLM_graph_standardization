#include "quantization_management.h"

// Returns the size of one super-block for each quantization type
size_t get_k_quant_super_block_size(ggml_type type) {
    switch(type) {
        case GGML_TYPE_Q2_K: return sizeof(block_q2_k);
        case GGML_TYPE_Q3_K: return sizeof(block_q3_k);
        case GGML_TYPE_Q4_K: return sizeof(block_q4_k);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_k);
        case GGML_TYPE_Q6_K: return sizeof(block_q6_k);
        case GGML_TYPE_Q8_K: return sizeof(block_q8_k);
        default: return 0;
    }
}

// Validates that a tensor has correct dimensions and size for its quantization type
bool validate_quant_tensor(const GGUFTensor& tensor) {
    const size_t sb_size = get_k_quant_super_block_size(tensor.type);
    if (sb_size == 0) {
        std::cerr << "Unsupported quantization type: " << ggml_type_name(tensor.type) << std::endl;
        return false;
    }
    
    const size_t num_super_blocks = tensor.size / sb_size;
    const size_t expected_weights = num_super_blocks * 256; // All K-types use 256 weights/sb
    
    if (tensor.size % sb_size != 0) {
        std::cerr << "Invalid tensor size for " << ggml_type_name(tensor.type)
                  << ": " << tensor.size << " not multiple of " << sb_size << std::endl;
        return false;
    }
    
    size_t total_dims = 1;
    for (auto dim : tensor.dims) total_dims *= dim;
    
    if (total_dims != expected_weights) {
        std::cerr << "Dimension mismatch for " << ggml_type_name(tensor.type)
                  << ": expected " << expected_weights << " weights, got " << total_dims << std::endl;
        return false;
    }
    
    return true;
}

// Creates and initializes a GGML tensor with K-type quantized data
ggml_tensor* create_k_quant_tensor(ggml_context* ctx, const GGUFTensor& tensor_info) {
    if (!validate_quant_tensor(tensor_info)) {
        return nullptr;
    }

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
            std::cerr << "Unsupported dimensionality for K-quant tensor" << std::endl;
            return nullptr;
    }

    if (const auto* data = tensor_info.get_data<uint8_t>()) {
        if (data->size() != tensor_info.size) {
            std::cerr << "Data size mismatch for " << tensor_info.name 
                      << ": expected " << tensor_info.size 
                      << ", got " << data->size() << std::endl;
            return nullptr;
        }
        memcpy(tensor->data, data->data(), data->size());
    } else {
        std::cerr << "No quant data available for " << tensor_info.name << std::endl;
        return nullptr;
    }

    return tensor;
}

// ==================== Dequantization Functions ====================

void dequantize_q2_k(const void* src, float* dst, int k) {
    const int nb = k / 256;
    const block_q2_k* x = (const block_q2_k*)src;
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const float min = ggml_fp16_to_fp32(x[i].dmin);
        
        for (int j = 0; j < 128; j++) {
            const uint8_t q = x[i].qs[j];
            
            dst[256*i + 4*j + 0] = d * (q & 0x3) + min;
            dst[256*i + 4*j + 1] = d * ((q >> 2) & 0x3) + min;
            dst[256*i + 4*j + 2] = d * ((q >> 4) & 0x3) + min;
            dst[256*i + 4*j + 3] = d * (q >> 6) + min;
        }
    }
}

void dequantize_q3_k(const void* src, float* dst, int k) {
    const int nb = k / 256;
    const block_q3_k* x = (const block_q3_k*)src;
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const uint8_t * scales = x[i].scales;
        const uint8_t * q = x[i].qs;
        
        for (int j = 0; j < 128; j++) {
            const uint8_t packed = q[j];
            const uint8_t w0 =  packed       & 0x7;
            const uint8_t w1 = (packed >> 3) & 0x7;
            const uint8_t w2 = (packed >> 6) | ((q[j+1] << 2) & 0x7);
            const uint8_t w3 = (q[j+1] >> 1) & 0x7;
            
            const uint8_t scale_shift = 2*(j/8) + ((j%8)/4);
            const float block_scale = (scales[scale_shift >> 1] >> (4*(scale_shift & 1))) & 0x3F;
            const float scale = d * (block_scale - 32);
            
            dst[256*i + 4*j + 0] = scale * (w0 - 4);
            dst[256*i + 4*j + 1] = scale * (w1 - 4);
            dst[256*i + 4*j + 2] = scale * (w2 - 4);
            dst[256*i + 4*j + 3] = scale * (w3 - 4);
            
            if (j % 4 == 3) j++;
        }
    }
}

void dequantize_q4_k(const void* src, float* dst, int k) {
    const int nb = k / 256;
    const block_q4_k* x = (const block_q4_k*)src;
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const float min = ggml_fp16_to_fp32(x[i].dmin);
        
        for (int j = 0; j < 128; j++) {
            const uint8_t q0 = x[i].qs[2*j];
            const uint8_t q1 = x[i].qs[2*j+1];
            
            const uint8_t scale_shift = j%8 < 4 ? 0 : 4;
            const float scale = d * ((x[i].scales[j/8] >> scale_shift) & 0xF);
            const float min_val = min * ((x[i].scales[j/8+8] >> scale_shift) & 0xF);
            
            dst[256*i + 8*j + 0] = scale * (q0 & 0xF) + min_val;
            dst[256*i + 8*j + 1] = scale * (q0 >> 4)   + min_val;
            dst[256*i + 8*j + 2] = scale * (q1 & 0xF) + min_val;
            dst[256*i + 8*j + 3] = scale * (q1 >> 4)   + min_val;
        }
    }
}

void dequantize_q5_k(const void* src, float* dst, int k) {
    const int nb = k / 256;
    const block_q5_k* x = (const block_q5_k*)src;
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        const float min = ggml_fp16_to_fp32(x[i].dmin);
        
        for (int j = 0; j < 128; j++) {
            const uint8_t ql = x[i].qs[2*j];
            const uint8_t qh = x[i].qs[2*j+1];
            
            const uint8_t scale_shift = j%8 < 4 ? 0 : 4;
            const float scale = d * ((x[i].scales[j/8] >> scale_shift) & 0xF);
            const float min_val = min * ((x[i].scales[j/8+8] >> scale_shift) & 0xF);
            
            dst[256*i + 8*j + 0] = scale * ((ql & 0xF) | ((qh & 0x3) << 4)) + min_val;
            dst[256*i + 8*j + 1] = scale * ((ql >> 4) | ((qh & 0xC) << 2)) + min_val;
        }
    }
}

void dequantize_q6_k(const void* src, float* dst, int k) {
    const int nb = k / 256;
    const block_q6_k* x = (const block_q6_k*)src;
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        
        for (int j = 0; j < 128; j++) {
            const uint8_t ql = x[i].qs[j];
            const uint8_t qh = x[i].qs[j+128];
            
            const float scale = d * x[i].scales[j/16];
            
            dst[256*i + 2*j + 0] = scale * ((ql & 0xF) | ((qh & 0x3) << 4));
            dst[256*i + 2*j + 1] = scale * ((ql >> 4) | ((qh & 0xC) << 2));
        }
    }
}

void dequantize_q8_k(const void* src, float* dst, int k) {
    const int nb = k / 256;
    const block_q8_k* x = (const block_q8_k*)src;
    
    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);
        
        for (int j = 0; j < 256; j++) {
            dst[256*i + j] = d * x[i].qs[j];
        }
    }
}

// Unified dequantization interface
// k represents the total number of elements to be processed (dequantized)
void dequantize_k_quant(ggml_type type, const void* src, float* dst, int k) {
    switch(type) {
        case GGML_TYPE_Q2_K: dequantize_q2_k(src, dst, k); break;
        case GGML_TYPE_Q3_K: dequantize_q3_k(src, dst, k); break;
        case GGML_TYPE_Q4_K: dequantize_q4_k(src, dst, k); break;
        case GGML_TYPE_Q5_K: dequantize_q5_k(src, dst, k); break;
        case GGML_TYPE_Q6_K: dequantize_q6_k(src, dst, k); break;
        case GGML_TYPE_Q8_K: dequantize_q8_k(src, dst, k); break;
        default:
            std::cerr << "Unsupported K-type quantization: " << ggml_type_name(type) << std::endl;
    }
}