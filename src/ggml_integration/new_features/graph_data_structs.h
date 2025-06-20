#ifndef GRAPH_DATA_STRUCTS_H
#define GRAPH_DATA_STRUCTS_H

#include <cstdint>
#include <string>
#include <vector>
#include <variant>
#include <cstring>
#include "gguf.h"


/**
 * @file graph_data_structs.h
 * @brief Definitions of structures to handle GGUF data
 */

/**
 * @brief Header of a GGUF file
 */
struct GGUFHeader {
    uint64_t n_tensors;     // Number of tensors in the file
    uint64_t n_kv;          // Number of key-value metadata pairs
};

/**
 * @brief GGUF metadata (key-value pairs)
 */
struct GGUFMetadata {
    std::string key;        // Metadata name/key
    enum gguf_type type;    // Type of the stored value
    
    // Value using a union for different possible types
    union {
        uint8_t u8;
        int8_t i8;
        uint16_t u16;
        int16_t i16;
        uint32_t u32;
        int32_t i32;
        float f32;
        uint64_t u64;
        int64_t i64;
        double f64;
        bool b;
    } value;
    
    // For array types
    struct {
        enum gguf_type type;
        size_t size;
        std::variant<
            std::vector<uint8_t>,   // For raw/bytes types
            std::vector<float>,     // F32
            std::vector<uint16_t>,  // F16
            std::vector<int8_t>,    // I8
            std::vector<int16_t>,   // I16
            std::vector<int32_t>,   // I32
            std::vector<uint32_t>,  // U32
            std::vector<int64_t>,   // I64
            std::vector<uint64_t>,  // U64
            std::vector<double>,    // F64
            std::vector<std::string> // Strings
        > data;
    } array;
    
    std::string str;        // For strings
    
    // Default constructor
    GGUFMetadata() : type(GGUF_TYPE_COUNT) {}
};

/**
 * @brief Representation of a GGUF tensor
 */
struct GGUFTensor {
    std::string name;           // Tensor name
    enum ggml_type type;        // Tensor data type
    size_t size;                // Tensor size in bytes
    int32_t n_dims;             // Number of tensor dimensions
    std::vector<int64_t> dims;  // Tensor dimensions

    enum ggml_op op;            // Tensor operation if applicable
    ggml_unary_op unary_op;     // Unary operation if applicable
    std::vector<std::string> src_tensors; ///< Names of input tensors
    std::string dst_tensor;     // Name of destination tensor that will use this result

    // Operation parameters (similar to ggml_tensor)
    //std::vector<int32_t> op_params;

    // Data storage using variant
    std::variant<
        std::vector<uint8_t>,    // For quantized types
        std::vector<int8_t>,     // For Int8
        std::vector<int16_t>,    // For Int16
        std::vector<float>,      // For F32
        std::vector<uint16_t>,   // For F16
        std::vector<int32_t>     // For I32
    > data;

    /**
     * @brief Gets tensor data as vector of specified type
     * @tparam T Requested data type
     * @return Pointer to data vector or nullptr if type doesn't match
     */
    template<typename T>
    const std::vector<T>* get_data() const {
        return std::get_if<std::vector<T>>(&data);
    }
};

/**
 * @brief Main container for GGUF data
 */
struct GraphData {
    GGUFHeader header;                      // GGUF header
    std::vector<GGUFMetadata> metadata;     // Metadata list
    std::vector<GGUFTensor> tensors;        // Tensor list
    
    /**
     * @brief Finds metadata by key
     * @param key Key to search for
     * @return Pointer to the metadata or nullptr if not found
     */
    const GGUFMetadata* find_metadata(const std::string& key) const;
    
    /**
     * @brief Finds a tensor by name
     * @param name Tensor name to search for
     * @return Pointer to the tensor or nullptr if not found
     */
    const GGUFTensor* find_tensor(const std::string& name) const;
};

/**
 * @brief Gets the size in bytes for a given GGUF type
 * @param type The GGUF type to check
 * @return Size of the type in bytes
 */
static inline size_t type_size(enum gguf_type type);

/**
 * @brief Reads array data from GGUF context into a vector of specified type
 * @tparam T Type of data to read (must match GGUF array type)
 * @param ctx GGUF context pointer
 * @param i Index of the array in GGUF context
 * @param size Number of elements to read
 * @return Vector containing the read data, or empty vector on error
 */
template <typename T>
static std::vector<T> read_array_data(const gguf_context *ctx, int64_t i, size_t size) {
    // Check for invalid input parameters
    if (!ctx || size == 0) {
        return {};
    }

    // Get raw array data pointer from GGUF context
    const void *src_data = gguf_get_arr_data(ctx, i);
    if (!src_data) {
        return {};
    }

    // Create destination vector with requested size
    std::vector<T> dest(size);
    
    // Special handling for byte arrays (use memcpy for efficiency)
    if constexpr (std::is_same_v<T, uint8_t>) {
        memcpy(dest.data(), src_data, size * sizeof(T));
    } else {
        // For typed arrays, use copy with proper type casting
        const T *typed_src = static_cast<const T *>(src_data);
        std::copy(typed_src, typed_src + size, dest.begin());
    }
    
    return dest;
}

#endif // GRAPH_DATA_STRUCTS_H

