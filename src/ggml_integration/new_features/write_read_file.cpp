#include <fstream>
#include <iostream>
#include "graph_data_structs.h"


/**
 * @brief Writes GraphData to a binary .graph file
 * @param filename Output filename (e.g., "model.graph")
 * @param graph_data GraphData structure to serialize
 * @return true if successful, false on error
 */
bool write_graph_data(const std::string& filename, const GraphData& graph_data) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return false;
    }

    // Write header
    out.write(reinterpret_cast<const char*>(&graph_data.header), sizeof(GGUFHeader));

    // Write metadata count
    uint64_t metadata_count = graph_data.metadata.size();
    out.write(reinterpret_cast<const char*>(&metadata_count), sizeof(uint64_t));

    // Write each metadata
    for (const auto& md : graph_data.metadata) {
        // Write key
        uint64_t key_size = md.key.size();
        out.write(reinterpret_cast<const char*>(&key_size), sizeof(uint64_t));
        out.write(md.key.c_str(), key_size);

        // Write type
        out.write(reinterpret_cast<const char*>(&md.type), sizeof(enum gguf_type));

        // Write value based on type
        switch (md.type) {
            case GGUF_TYPE_UINT8: out.write(reinterpret_cast<const char*>(&md.value.u8), sizeof(uint8_t)); break;
            case GGUF_TYPE_INT8: out.write(reinterpret_cast<const char*>(&md.value.i8), sizeof(int8_t)); break;
            case GGUF_TYPE_UINT16: out.write(reinterpret_cast<const char*>(&md.value.u16), sizeof(uint16_t)); break;
            case GGUF_TYPE_INT16: out.write(reinterpret_cast<const char*>(&md.value.i16), sizeof(int16_t)); break;
            case GGUF_TYPE_UINT32: out.write(reinterpret_cast<const char*>(&md.value.u32), sizeof(uint32_t)); break;
            case GGUF_TYPE_INT32: out.write(reinterpret_cast<const char*>(&md.value.i32), sizeof(int32_t)); break;
            case GGUF_TYPE_FLOAT32: out.write(reinterpret_cast<const char*>(&md.value.f32), sizeof(float)); break;
            case GGUF_TYPE_UINT64: out.write(reinterpret_cast<const char*>(&md.value.u64), sizeof(uint64_t)); break;
            case GGUF_TYPE_INT64: out.write(reinterpret_cast<const char*>(&md.value.i64), sizeof(int64_t)); break;
            case GGUF_TYPE_FLOAT64: out.write(reinterpret_cast<const char*>(&md.value.f64), sizeof(double)); break;
            case GGUF_TYPE_BOOL: out.write(reinterpret_cast<const char*>(&md.value.b), sizeof(bool)); break;
            case GGUF_TYPE_STRING: {
                uint64_t str_size = md.str.size();
                out.write(reinterpret_cast<const char*>(&str_size), sizeof(uint64_t));
                out.write(md.str.c_str(), str_size);
                break;
            }
            case GGUF_TYPE_ARRAY: {
                out.write(reinterpret_cast<const char*>(&md.array.type), sizeof(enum gguf_type));
                out.write(reinterpret_cast<const char*>(&md.array.size), sizeof(size_t));

                // Handle array data
                if (std::holds_alternative<std::vector<std::string>>(md.array.data)) {
                    // String array case
                    const auto& strings = std::get<std::vector<std::string>>(md.array.data);
                    for (const auto& str : strings) {
                        uint64_t str_size = str.size();
                        out.write(reinterpret_cast<const char*>(&str_size), sizeof(uint64_t));
                        out.write(str.c_str(), str_size);
                    }
                } else {
                    // Numeric array case
                    std::visit([&out](const auto& vec) {
                        using T = std::decay_t<decltype(vec)>::value_type;
                        out.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
                    }, md.array.data);
                }
                break;
            }
            default:
                std::cerr << "Unsupported metadata type: " << md.type << "\n";
                return false;
        }
    }

    // Write tensors count
    uint64_t tensors_count = graph_data.tensors.size();
    out.write(reinterpret_cast<const char*>(&tensors_count), sizeof(uint64_t));

    // Write each tensor
    for (const auto& tensor : graph_data.tensors) {
        // Write name
        uint64_t name_size = tensor.name.size();
        out.write(reinterpret_cast<const char*>(&name_size), sizeof(uint64_t));
        out.write(tensor.name.c_str(), name_size);

        // Write tensor info
        out.write(reinterpret_cast<const char*>(&tensor.type), sizeof(enum ggml_type));
        out.write(reinterpret_cast<const char*>(&tensor.size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(&tensor.n_dims), sizeof(int32_t));

        // Write dimensions
        out.write(reinterpret_cast<const char*>(tensor.dims.data()), tensor.n_dims * sizeof(int64_t));

        // Write operation info
        out.write(reinterpret_cast<const char*>(&tensor.op), sizeof(enum ggml_op));

        // Write src tensors
        uint64_t src_count = tensor.src_tensors.size();
        out.write(reinterpret_cast<const char*>(&src_count), sizeof(uint64_t));
        for (const auto& src : tensor.src_tensors) {
            uint64_t src_size = src.size();
            out.write(reinterpret_cast<const char*>(&src_size), sizeof(uint64_t));
            out.write(src.c_str(), src_size);
        }

        // Write dst tensor
        uint64_t dst_size = tensor.dst_tensor.size();
        out.write(reinterpret_cast<const char*>(&dst_size), sizeof(uint64_t));
        out.write(tensor.dst_tensor.c_str(), dst_size);

        // Write tensor data
        std::visit([&out](const auto& vec) {
            using T = std::decay_t<decltype(vec)>::value_type;
            out.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
        }, tensor.data);
    }

    out.close();
    return true;
}

/**
 * @brief Reads a metadata entry by key from a .graph file
 * @param filename Input .graph filename
 * @param key Metadata key to search for
 * @return GGUFMetadata structure if found, or empty metadata with type=GGUF_TYPE_COUNT if not found
 */
GGUFMetadata read_metadata(const std::string& filename, const std::string& key) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << filename << " for reading\n";
        return GGUFMetadata();
    }

    // Read header
    GGUFHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(GGUFHeader));

    // Read metadata count
    uint64_t metadata_count;
    in.read(reinterpret_cast<char*>(&metadata_count), sizeof(uint64_t));

    // Search for the requested key
    for (uint64_t i = 0; i < metadata_count; ++i) {
        // Read key
        uint64_t key_size;
        in.read(reinterpret_cast<char*>(&key_size), sizeof(uint64_t));
        std::string current_key(key_size, '\0');
        in.read(&current_key[0], key_size);

        // Read type
        enum gguf_type type;
        in.read(reinterpret_cast<char*>(&type), sizeof(enum gguf_type));

        GGUFMetadata md;
        md.key = current_key;
        md.type = type;

        // Read value based on type
        switch (type) {
            case GGUF_TYPE_UINT8: in.read(reinterpret_cast<char*>(&md.value.u8), sizeof(uint8_t)); break;
            case GGUF_TYPE_INT8: in.read(reinterpret_cast<char*>(&md.value.i8), sizeof(int8_t)); break;
            case GGUF_TYPE_UINT16: in.read(reinterpret_cast<char*>(&md.value.u16), sizeof(uint16_t)); break;
            case GGUF_TYPE_INT16: in.read(reinterpret_cast<char*>(&md.value.i16), sizeof(int16_t)); break;
            case GGUF_TYPE_UINT32: in.read(reinterpret_cast<char*>(&md.value.u32), sizeof(uint32_t)); break;
            case GGUF_TYPE_INT32: in.read(reinterpret_cast<char*>(&md.value.i32), sizeof(int32_t)); break;
            case GGUF_TYPE_FLOAT32: in.read(reinterpret_cast<char*>(&md.value.f32), sizeof(float)); break;
            case GGUF_TYPE_UINT64: in.read(reinterpret_cast<char*>(&md.value.u64), sizeof(uint64_t)); break;
            case GGUF_TYPE_INT64: in.read(reinterpret_cast<char*>(&md.value.i64), sizeof(int64_t)); break;
            case GGUF_TYPE_FLOAT64: in.read(reinterpret_cast<char*>(&md.value.f64), sizeof(double)); break;
            case GGUF_TYPE_BOOL: in.read(reinterpret_cast<char*>(&md.value.b), sizeof(bool)); break;
            case GGUF_TYPE_STRING: {
                uint64_t str_size;
                in.read(reinterpret_cast<char*>(&str_size), sizeof(uint64_t));
                md.str.resize(str_size);
                in.read(&md.str[0], str_size);
                break;
            }
            case GGUF_TYPE_ARRAY: {
                in.read(reinterpret_cast<char*>(&md.array.type), sizeof(enum gguf_type));
                in.read(reinterpret_cast<char*>(&md.array.size), sizeof(size_t));

                // Read array data
                if (md.array.type == GGUF_TYPE_STRING) {
                    std::vector<std::string> strings(md.array.size);
                    for (size_t j = 0; j < md.array.size; ++j) {
                        uint64_t str_size;
                        in.read(reinterpret_cast<char*>(&str_size), sizeof(uint64_t));
                        strings[j].resize(str_size);
                        in.read(&strings[j][0], str_size);
                    }
                    md.array.data = strings;
                } else {
                    // Numeric array case
                    switch (md.array.type) {
                        case GGUF_TYPE_UINT8: md.array.data = read_array<uint8_t>(in, md.array.size); break;
                        case GGUF_TYPE_INT8: md.array.data = read_array<int8_t>(in, md.array.size); break;
                        case GGUF_TYPE_UINT16: md.array.data = read_array<uint16_t>(in, md.array.size); break;
                        case GGUF_TYPE_INT16: md.array.data = read_array<int16_t>(in, md.array.size); break;
                        case GGUF_TYPE_UINT32: md.array.data = read_array<uint32_t>(in, md.array.size); break;
                        case GGUF_TYPE_INT32: md.array.data = read_array<int32_t>(in, md.array.size); break;
                        case GGUF_TYPE_FLOAT32: md.array.data = read_array<float>(in, md.array.size); break;
                        case GGUF_TYPE_UINT64: md.array.data = read_array<uint64_t>(in, md.array.size); break;
                        case GGUF_TYPE_INT64: md.array.data = read_array<int64_t>(in, md.array.size); break;
                        case GGUF_TYPE_FLOAT64: md.array.data = read_array<double>(in, md.array.size); break;
                        default:
                            std::cerr << "Unsupported array type: " << md.array.type << "\n";
                            continue;
                    }
                }
                break;
            }
            default:
                std::cerr << "Unsupported metadata type: " << type << "\n";
                continue;
        }

        if (current_key == key) {
            in.close();
            return md;
        }
    }

    in.close();
    return GGUFMetadata(); // Not found
}

/**
 * @brief Reads a tensor by name from a .graph file
 * @param filename Input .graph filename
 * @param name Tensor name to search for
 * @return GGUFTensor structure if found, or empty tensor with name="" if not found
 */
GGUFTensor read_tensor(const std::string& filename, const std::string& name) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << filename << " for reading\n";
        return GGUFTensor();
    }

    // Read header
    GGUFHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(GGUFHeader));

    // Skip metadata section
    uint64_t metadata_count;
    in.read(reinterpret_cast<char*>(&metadata_count), sizeof(uint64_t));
    for (uint64_t i = 0; i < metadata_count; ++i) {
        skip_metadata(in);
    }

    // Read tensors count
    uint64_t tensors_count;
    in.read(reinterpret_cast<char*>(&tensors_count), sizeof(uint64_t));

    // Search for the requested tensor
    for (uint64_t i = 0; i < tensors_count; ++i) {
        // Read name
        uint64_t name_size;
        in.read(reinterpret_cast<char*>(&name_size), sizeof(uint64_t));
        std::string current_name(name_size, '\0');
        in.read(&current_name[0], name_size);

        // Read tensor info
        GGUFTensor tensor;
        tensor.name = current_name;
        in.read(reinterpret_cast<char*>(&tensor.type), sizeof(enum ggml_type));
        in.read(reinterpret_cast<char*>(&tensor.size), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&tensor.n_dims), sizeof(int32_t));

        // Read dimensions
        tensor.dims.resize(tensor.n_dims);
        in.read(reinterpret_cast<char*>(tensor.dims.data()), tensor.n_dims * sizeof(int64_t));

        // Read operation info
        in.read(reinterpret_cast<char*>(&tensor.op), sizeof(enum ggml_op));

        // Read src tensors
        uint64_t src_count;
        in.read(reinterpret_cast<char*>(&src_count), sizeof(uint64_t));
        tensor.src_tensors.resize(src_count);
        for (uint64_t j = 0; j < src_count; ++j) {
            uint64_t src_size;
            in.read(reinterpret_cast<char*>(&src_size), sizeof(uint64_t));
            tensor.src_tensors[j].resize(src_size);
            in.read(&tensor.src_tensors[j][0], src_size);
        }

        // Read dst tensor
        uint64_t dst_size;
        in.read(reinterpret_cast<char*>(&dst_size), sizeof(uint64_t));
        tensor.dst_tensor.resize(dst_size);
        in.read(&tensor.dst_tensor[0], dst_size);

        // Calculate number of elements based on tensor size and type
        size_t element_count = tensor.size / ggml_type_size(tensor.type);
        if (element_count == 0) element_count = tensor.size; // fallback for unknown types

        // Read tensor data
        switch (tensor.type) {
            case GGML_TYPE_F32: tensor.data = read_array<float>(in, element_count); break;
            case GGML_TYPE_F16: tensor.data = read_array<uint16_t>(in, element_count); break;
            case GGML_TYPE_I32: tensor.data = read_array<int32_t>(in, element_count); break;
            case GGML_TYPE_I16: tensor.data = read_array<int16_t>(in, element_count); break;
            case GGML_TYPE_I8: tensor.data = read_array<int8_t>(in, element_count); break;
            default:
                // For quantized or unknown types, read as uint8_t
                tensor.data = read_array<uint8_t>(in, tensor.size);
                break;
        }

        if (current_name == name) {
            in.close();
            return tensor;
        }
    }

    in.close();
    return GGUFTensor(); // Not found
}

// Helper function to skip metadata when searching for tensors
void skip_metadata(std::ifstream& in) {
    // Skip key
    uint64_t key_size;
    in.read(reinterpret_cast<char*>(&key_size), sizeof(uint64_t));
    in.seekg(key_size, std::ios::cur);

    // Skip type and value
    enum gguf_type type;
    in.read(reinterpret_cast<char*>(&type), sizeof(enum gguf_type));

    switch (type) {
        case GGUF_TYPE_UINT8: in.seekg(sizeof(uint8_t), std::ios::cur); break;
        case GGUF_TYPE_INT8: in.seekg(sizeof(int8_t), std::ios::cur); break;
        case GGUF_TYPE_UINT16: in.seekg(sizeof(uint16_t), std::ios::cur); break;
        case GGUF_TYPE_INT16: in.seekg(sizeof(int16_t), std::ios::cur); break;
        case GGUF_TYPE_UINT32: in.seekg(sizeof(uint32_t), std::ios::cur); break;
        case GGUF_TYPE_INT32: in.seekg(sizeof(int32_t), std::ios::cur); break;
        case GGUF_TYPE_FLOAT32: in.seekg(sizeof(float), std::ios::cur); break;
        case GGUF_TYPE_UINT64: in.seekg(sizeof(uint64_t), std::ios::cur); break;
        case GGUF_TYPE_INT64: in.seekg(sizeof(int64_t), std::ios::cur); break;
        case GGUF_TYPE_FLOAT64: in.seekg(sizeof(double), std::ios::cur); break;
        case GGUF_TYPE_BOOL: in.seekg(sizeof(bool), std::ios::cur); break;
        case GGUF_TYPE_STRING: {
            uint64_t str_size;
            in.read(reinterpret_cast<char*>(&str_size), sizeof(uint64_t));
            in.seekg(str_size, std::ios::cur);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            enum gguf_type arr_type;
            size_t arr_size;
            in.read(reinterpret_cast<char*>(&arr_type), sizeof(enum gguf_type));
            in.read(reinterpret_cast<char*>(&arr_size), sizeof(size_t));
            in.seekg(arr_size * type_size(arr_type), std::ios::cur);
            break;
        }
        default:
            break;
    }
}

// Helper function to read array data
template<typename T>
std::vector<T> read_array(std::ifstream& in, size_t size) {
    std::vector<T> data(size);
    in.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    return data;
}