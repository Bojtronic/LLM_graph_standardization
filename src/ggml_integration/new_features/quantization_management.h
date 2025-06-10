#ifndef QUANTIZATION_MANAGEMENT_H
#define QUANTIZATION_MANAGEMENT_H

#include <vector>
#include <cstdint>
#include <iostream>
#include <cstring>
#include <cmath>
#include <ggml.h>
#include "graph_data_structs.h"

// Define block structures for each quantization type
typedef struct {
    uint8_t qs[64];   // 2-bit quantized weights
    uint8_t scales[8]; // 4-bit scales
    ggml_fp16_t d;     // super-block scale
    ggml_fp16_t dmin;  // super-block minimum
} block_q2_k;

typedef struct {
    uint8_t qs[96];   // 3-bit quantized weights
    uint8_t scales[12]; // 6-bit scales
    ggml_fp16_t d;     // super-block scale
} block_q3_k;

typedef struct {
    uint8_t qs[128];  // 4-bit quantized weights
    uint8_t scales[12]; // 6-bit scales and mins
    ggml_fp16_t d;     // super-block scale
    ggml_fp16_t dmin;  // super-block minimum
} block_q4_k;

typedef struct {
    uint8_t qs[160];  // 5-bit quantized weights
    uint8_t scales[12]; // 6-bit scales and mins
    ggml_fp16_t d;     // super-block scale
    ggml_fp16_t dmin;  // super-block minimum
} block_q5_k;

typedef struct {
    uint8_t qs[192];  // 6-bit quantized weights
    uint8_t scales[16]; // 8-bit scales
    ggml_fp16_t d;     // super-block scale
} block_q6_k;

typedef struct {
    uint8_t qs[256];  // 8-bit quantized weights
    ggml_fp16_t d;     // super-block scale
} block_q8_k;

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

#endif // QUANTIZATION_MANAGEMENT_H