#include "graph_data_structs.h"

/**
 * @brief Finds metadata entry by its key
 * @param key The key to search for in metadata
 * @return Pointer to the metadata if found, nullptr otherwise
 */
const GGUFMetadata *GraphData::find_metadata(const std::string &key) const
{
    // Iterate through all metadata entries
    for (const auto &md : metadata)
    {
        // Check if current entry matches the search key
        if (md.key == key)
            return &md;  // Return pointer to matching metadata
    }
    return nullptr;  // Return null if not found
}

/**
 * @brief Finds tensor by its name
 * @param name The tensor name to search for
 * @return Pointer to the tensor if found, nullptr otherwise
 */
const GGUFTensor *GraphData::find_tensor(const std::string &name) const
{
    // Iterate through all tensors
    for (const auto &tensor : tensors)
    {
        // Check if current tensor matches the search name
        if (tensor.name == name)
            return &tensor;  // Return pointer to matching tensor
    }
    return nullptr;  // Return null if not found
}

/**
 * @brief Gets the size in bytes of a GGUF type
 * @param type The GGUF type to check
 * @return Size of the type in bytes (0 for types with variable size)
 */
static inline size_t type_size(enum gguf_type type)
{
    // Return size based on type
    switch (type)
    {
    case GGUF_TYPE_UINT8:
        return sizeof(uint8_t);
    case GGUF_TYPE_INT8:
        return sizeof(int8_t);
    case GGUF_TYPE_UINT16:
        return sizeof(uint16_t);
    case GGUF_TYPE_INT16:
        return sizeof(int16_t);
    case GGUF_TYPE_UINT32:
        return sizeof(uint32_t);
    case GGUF_TYPE_INT32:
        return sizeof(int32_t);
    case GGUF_TYPE_FLOAT32:
        return sizeof(float);
    case GGUF_TYPE_BOOL:
        return sizeof(bool);
    case GGUF_TYPE_STRING:
        return 0;  // Strings have variable size
    case GGUF_TYPE_UINT64:
        return sizeof(uint64_t);
    case GGUF_TYPE_INT64:
        return sizeof(int64_t);
    case GGUF_TYPE_FLOAT64:
        return sizeof(double);
    case GGUF_TYPE_ARRAY:
        return 0;  // Arrays have variable size
    default:
        return 0;  // Unknown types
    }
}



/**
 * @brief Explicit template instantiations for all supported data types
 * 
 * These instantiations ensure the read_array_data template function is compiled
 * for all supported numeric types, allowing efficient array reading from GGUF files.
 * The supported types cover standard integer and floating-point formats
 */
template std::vector<uint8_t> read_array_data<uint8_t>(const gguf_context*, int64_t, size_t);
template std::vector<int8_t> read_array_data<int8_t>(const gguf_context*, int64_t, size_t);
template std::vector<uint16_t> read_array_data<uint16_t>(const gguf_context*, int64_t, size_t);
template std::vector<int16_t> read_array_data<int16_t>(const gguf_context*, int64_t, size_t);
template std::vector<uint32_t> read_array_data<uint32_t>(const gguf_context*, int64_t, size_t);
template std::vector<int32_t> read_array_data<int32_t>(const gguf_context*, int64_t, size_t);
template std::vector<float> read_array_data<float>(const gguf_context*, int64_t, size_t);
template std::vector<uint64_t> read_array_data<uint64_t>(const gguf_context*, int64_t, size_t);
template std::vector<int64_t> read_array_data<int64_t>(const gguf_context*, int64_t, size_t);
template std::vector<double> read_array_data<double>(const gguf_context*, int64_t, size_t);

