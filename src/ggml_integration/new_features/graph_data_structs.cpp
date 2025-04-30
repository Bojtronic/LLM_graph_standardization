#include "graph_data_structs.h"

const GGUFMetadata *GraphData::find_metadata(const std::string &key) const
{
    for (const auto &md : metadata)
    {
        if (md.key == key)
            return &md;
    }
    return nullptr;
}

const GGUFTensor *GraphData::find_tensor(const std::string &name) const
{
    for (const auto &tensor : tensors)
    {
        if (tensor.name == name)
            return &tensor;
    }
    return nullptr;
}

static inline size_t type_size(enum gguf_type type)
{
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
        return 0;
    case GGUF_TYPE_UINT64:
        return sizeof(uint64_t);
    case GGUF_TYPE_INT64:
        return sizeof(int64_t);
    case GGUF_TYPE_FLOAT64:
        return sizeof(double);
    case GGUF_TYPE_ARRAY:
        return 0;
    default:
        return 0;
    }
}



/*
// Instanciaciones explícitas para todos los tipos soportados
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
*/