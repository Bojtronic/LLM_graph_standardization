#ifndef QUANTIZATION_MANAGEMENT_H
#define QUANTIZATION_MANAGEMENT_H

#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>
#include <cmath>
#include <ggml.h>
#include "graph_data_structs.h"

#define QK_K 256  // Super-block size
#define K_SCALE_SIZE 12  // Scale size for some blocks

typedef ggml_fp16_t ggml_half;

// Basic quantized block structures
typedef struct {
    uint8_t scales[QK_K/16];  // 16 scales (4 bits each)
    uint8_t qs[QK_K/4];       // 64 2-bit quantized values
    ggml_fp16_t d;            // Super-block scale
    ggml_fp16_t dmin;         // Super-block minimum
} block_q2_K;

typedef struct {
    uint8_t hmask[QK_K/8];    // 32 high bits (1 bit each)
    uint8_t qs[QK_K/4];       // 64 2-bit quantized values (low bits)
    uint8_t scales[QK_K/16];  // 16 scales (6 bits each)
    ggml_fp16_t d;            // Super-block scale
} block_q3_K;

typedef struct {
    uint8_t scales[K_SCALE_SIZE];  // Scales and minimums (6 bits)
    uint8_t qs[QK_K/2];            // 128 4-bit quantized values
    ggml_fp16_t d;                 // Super-block scale
    ggml_fp16_t dmin;              // Super-block minimum
} block_q4_K;

typedef struct {
    uint8_t scales[K_SCALE_SIZE];  // Scales and minimums (6 bits)
    uint8_t qh[QK_K/8];            // 32 high bits (1 bit each)
    uint8_t qs[QK_K/2];            // 128 4-bit quantized values (low bits)
    ggml_fp16_t d;                 // Super-block scale
    ggml_fp16_t dmin;              // Super-block minimum
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K/2];       // 128 4-bit quantized values (low bits)
    uint8_t qh[QK_K/4];       // 64 2-bit quantized values (high bits)
    int8_t scales[QK_K/16];   // 16 scales (8 bits)
    ggml_fp16_t d;            // Super-block scale
} block_q6_K;

typedef struct {
    int8_t qs[QK_K];          // 256 8-bit quantized values
    ggml_fp16_t d;            // Super-block scale
} block_q8_K;

// Function declarations
size_t get_k_quant_super_block_size(ggml_type type);
bool validate_quant_tensor(const GGUFTensor& tensor);
ggml_tensor* create_k_quant_tensor(ggml_context* ctx, const GGUFTensor& tensor_info);

// Dequantization functions
void dequantize_q2_k(const void* src, float* dst, int k);
void dequantize_q3_k(const void* src, float* dst, int k);
void dequantize_q4_k(const void* src, float* dst, int k);
void dequantize_q5_k(const void* src, float* dst, int k);
void dequantize_q6_k(const void* src, float* dst, int k);
void dequantize_q8_k(const void* src, float* dst, int k);
void dequantize_k_quant(ggml_type type, const void* src, float* dst, int k);
void convert_f32_to_i32(const float* src, int32_t* dst, int size, float scale = 1.0f, float offset = 0.0f);
bool ggml_is_quantized(ggml_type type);
void dequantized_warning(const std::string& name, ggml_type type);

#endif // QUANTIZATION_MANAGEMENT_H