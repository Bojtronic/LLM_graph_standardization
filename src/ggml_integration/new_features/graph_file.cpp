#include "graph_file.h"
#include "gguf.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>

size_t type_size(enum gguf_type type)
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
        return sizeof(uint32_t); // string size is stored as uint32_t
    case GGUF_TYPE_ARRAY:
        return sizeof(uint32_t); // array length is stored as uint32_t
    case GGUF_TYPE_UINT64:
        return sizeof(uint64_t);
    case GGUF_TYPE_INT64:
        return sizeof(int64_t);
    case GGUF_TYPE_FLOAT64:
        return sizeof(double);
    default:
        return 0; // unknown type
    }
}

/**
 * @brief Writes GraphData to a binary .graph file
 * @param filename Output filename (e.g., "model.graph")
 * @param graph_data GraphData structure to serialize
 * @return true if successful, false on error
 */
bool write_graph_data(const std::string &filename, const GraphData &graph_data)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
    {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return false;
    }

    // 1. Identification and version header
    const std::string magic = "GRAPH";
    out.write(magic.c_str(), magic.size());
    out.put('\n');

    // 2. Header
    out.write(reinterpret_cast<const char *>(&graph_data.header), sizeof(GGUFHeader));
    out.put('\n');

    const std::string metadata_marker = "---METADATA---";

    out.write(metadata_marker.c_str(), metadata_marker.size());
    out.put('\n');

    // 3. Metadata
    uint32_t metadata_count = graph_data.metadata.size();
    out.write(reinterpret_cast<const char *>(&metadata_count), sizeof(uint32_t));
    out.put('\n');

    for (const auto &md : graph_data.metadata)
    {
        // Key
        out << md.key;
        out.put('\n');

        // Type
        out.write(reinterpret_cast<const char *>(&md.type), sizeof(enum gguf_type));
        out.put('\n');

        // Value according to type
        switch (md.type)
        {
        case GGUF_TYPE_UINT8:
            out << static_cast<int>(md.value.u8);
            out.put('\n');
            break;
        case GGUF_TYPE_INT8:
            out << static_cast<int>(md.value.i8);
            out.put('\n');
            break;
        case GGUF_TYPE_UINT16:
            out << md.value.u16;
            out.put('\n');
            break;
        case GGUF_TYPE_INT16:
            out << md.value.i16;
            out.put('\n');
            break;
        case GGUF_TYPE_UINT32:
            out << md.value.u32;
            out.put('\n');
            break;
        case GGUF_TYPE_INT32:
            out << md.value.i32;
            out.put('\n');
            break;
        case GGUF_TYPE_FLOAT32:
            out << std::fixed << std::setprecision(6) << md.value.f32;
            out.put('\n');
            break;
        case GGUF_TYPE_UINT64:
            out << md.value.u64;
            out.put('\n');
            break;
        case GGUF_TYPE_INT64:
            out << md.value.i64;
            out.put('\n');
            break;
        case GGUF_TYPE_FLOAT64:
            out << std::fixed << std::setprecision(6) << md.value.f64;
            out.put('\n');
            break;
        case GGUF_TYPE_BOOL:
            out << (md.value.b ? "true" : "false");
            out.put('\n');
            break;
        case GGUF_TYPE_STRING:
            out << md.str;
            out.put('\n');
            break;
        case GGUF_TYPE_ARRAY:
            out.write(reinterpret_cast<const char *>(&md.array.type), sizeof(enum gguf_type));
            out.put('\n');
            out.write(reinterpret_cast<const char *>(&md.array.size), sizeof(size_t));
            out.put('\n');

            if (std::holds_alternative<std::vector<std::string>>(md.array.data))
            {
                for (const auto &str : std::get<std::vector<std::string>>(md.array.data))
                {
                    out << str;
                    out.put('\n');
                }
            }
            else
            {
                std::visit([&out](const auto &vec)
                           { using T = std::decay_t<decltype(vec)>::value_type; out.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T)); }, md.array.data);

                out.put('\n');
            }
            break;
        default:
            std::cerr << "Unsupported metadata type: " << md.type;
            out.put('\n');
            return false;
        }

        const std::string item_end_marker = "---END_ITEM---";
        out.write(item_end_marker.c_str(), item_end_marker.size());
        out.put('\n');
    }

    // 4. Tensors
    const std::string tensors_marker = "---TENSORS---";
    out.write(tensors_marker.c_str(), tensors_marker.size());
    out.put('\n');

    uint32_t tensors_count = graph_data.tensors.size(); // Número de tensores
    out.write(reinterpret_cast<const char *>(&tensors_count), sizeof(uint32_t));
    out.put('\n');

    for (const auto &tensor : graph_data.tensors)
    {
        // Start delimiter
        out << "---BEGIN_TENSOR---";
        out.put('\n');

        // Tensor info
        out << "NAME: " << tensor.name;
        out.put('\n');
        out << "ORIGINAL_TYPE(STORED_AS_F32): " << tensor.type; // Conservamos el tipo original (puede ser cuantizado)
        out.put('\n');
        out << "DATA_SIZE: " << tensor.size;
        out.put('\n');
        out << "NDIMS: " << tensor.n_dims;
        out.put('\n');
        out << "DIMS: ";
        for (const auto &dim : tensor.dims)
        {
            out << dim << " ";
        }
        out.put('\n');
        
        out << "OP:" << tensor.op;
        out.put('\n');

        // Tensor data (always float, even if the type indicates quantization)
        out << "DATA_START:";
        out.put('\n');

        // Direct access to float data with verification
        try
        {
            const auto &float_data = std::get<std::vector<float>>(tensor.data);
            out.write(reinterpret_cast<const char *>(float_data.data()), float_data.size() * sizeof(float));
        }
        catch (const std::bad_variant_access &)
        {
            std::cerr << "Error: Tensor " << tensor.name << " no contiene datos float\n";
            return false;
        }

        out.put('\n');
        out << "DATA_END";
        out.put('\n');

        // End delimiter
        out << "---END_TENSOR---";
        out.put('\n');
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
GGUFMetadata read_metadata(const std::string &filename, const std::string &key)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Failed to open " << filename << " for reading\n";
        return GGUFMetadata();
    }

    // 1. Verify magic string (5 chars "GRAPH" + 1 char '\n')
    char magic[6];
    in.read(magic, 6);
    if (std::string(magic, 5) != "GRAPH" || magic[5] != '\n')
    {
        std::cerr << "Invalid file format - bad magic string\n";
        return GGUFMetadata();
    }

    // 2. Read header
    //GGUFHeader header;
    //in.read(reinterpret_cast<char *>(&header), sizeof(GGUFHeader));

    uint64_t n_tensors;
    in.read(reinterpret_cast<char *>(&n_tensors), sizeof(uint64_t));

    in.ignore(1); // Skip \n

    uint64_t n_kv;
    in.read(reinterpret_cast<char *>(&n_kv), sizeof(uint64_t));

    // Read and verify header newline
    if (in.get() != '\n')
    {
        std::cerr << "Invalid header format - missing newline\n";
        return GGUFMetadata();
    }

    // 3. Read metadata marker
    std::string marker;
    std::getline(in, marker); // Reads until \n
    if (marker != "---METADATA---")
    {
        std::cerr << "Invalid metadata marker: " << marker << "\n";
        return GGUFMetadata();
    }

    // 4. Read metadata count (uint32_t)

    /*
    uint32_t metadata_count;
    in.read(reinterpret_cast<char *>(&metadata_count), sizeof(uint32_t));
    if (in.get() != '\n')
    {
        std::cerr << "Invalid metadata count format\n";
        return GGUFMetadata();
    }
    */

    // 5. Search for the requested key
    for (uint64_t i = 0; i < n_kv; ++i)
    {
        GGUFMetadata md;

        // Read key
        std::getline(in, md.key);
        if (in.fail())
            break;

        // Read type
        in.read(reinterpret_cast<char *>(&md.type), sizeof(enum gguf_type));
        if (in.get() != '\n')
        {
            std::cerr << "Invalid type format\n";
            return GGUFMetadata();
        }

        // Read value based on type
        switch (md.type)
        {
        case GGUF_TYPE_UINT8:
            in >> md.value.u8;
            in.ignore(1); // Skip \n
            break;
        case GGUF_TYPE_INT8:
            in >> md.value.i8;
            in.ignore(1);
            break;
        case GGUF_TYPE_UINT16:
            in >> md.value.u16;
            in.ignore(1);
            break;
        case GGUF_TYPE_INT16:
            in >> md.value.i16;
            in.ignore(1);
            break;
        case GGUF_TYPE_UINT32:
            in >> md.value.u32;
            in.ignore(1);
            break;
        case GGUF_TYPE_INT32:
            in >> md.value.i32;
            in.ignore(1);
            break;
        case GGUF_TYPE_FLOAT32:
            in >> md.value.f32;
            in.ignore(1);
            break;
        case GGUF_TYPE_UINT64:
            in >> md.value.u64;
            in.ignore(1);
            break;
        case GGUF_TYPE_INT64:
            in >> md.value.i64;
            in.ignore(1);
            break;
        case GGUF_TYPE_FLOAT64:
            in >> md.value.f64;
            in.ignore(1);
            break;
        case GGUF_TYPE_BOOL:
        {
            std::string val;
            in >> val;
            md.value.b = (val == "true");
            in.ignore(1); // Skip \n
            break;
        }
        case GGUF_TYPE_STRING:
            std::getline(in, md.str);
            break;
        case GGUF_TYPE_ARRAY:
        {
            // Read array type
            in.read(reinterpret_cast<char *>(&md.array.type), sizeof(enum gguf_type));
            if (in.get() != '\n')
            {
                std::cerr << "Invalid array type format\n";
                return GGUFMetadata();
            }

            // Read array size
            in.read(reinterpret_cast<char *>(&md.array.size), sizeof(size_t));
            if (in.get() != '\n')
            {
                std::cerr << "Invalid array size format\n";
                return GGUFMetadata();
            }

            if (md.array.type == GGUF_TYPE_STRING)
            {
                auto &arr = md.array.data.emplace<std::vector<std::string>>();
                arr.resize(md.array.size);
                for (auto &str : arr)
                {
                    // Leer longitud primero
                    uint32_t len;
                    in.read(reinterpret_cast<char*>(&len), sizeof(uint32_t));
                    
                    // Leer string codificado
                    std::vector<char> buffer(len);
                    in.read(buffer.data(), len);
                    
                    // Decodificar
                    std::string encoded_str(buffer.begin(), buffer.end());
                    str.clear();
                    for (size_t i = 0; i < encoded_str.size(); ++i) {
                        if (encoded_str[i] == '\\' && i+1 < encoded_str.size()) {
                            if (encoded_str[i+1] == 'n') { str += '\n'; i++; }
                            else if (encoded_str[i+1] == '0') { str += '\0'; i++; }
                            else if (encoded_str[i+1] == '\\') { str += '\\'; i++; }
                            else { str += encoded_str[i]; }
                        } else {
                            str += encoded_str[i];
                        }
                    }
                }
                if (in.get() != '\n') {
                    std::cerr << "Invalid string array format\n";
                    return GGUFMetadata();
                }
                break;
            }
            else
            {
                // Handle numeric arrays
                const size_t element_size = type_size(md.array.type);
                if (element_size == 0)
                {
                    std::cerr << "Unsupported array type: " << md.array.type << "\n";
                    return GGUFMetadata();
                }

                std::vector<char> buffer(md.array.size * element_size);
                in.read(buffer.data(), buffer.size());
                if (in.get() != '\n')
                { // Verify final newline
                    std::cerr << "Invalid array data format\n";
                    return GGUFMetadata();
                }

                // Convert buffer to appropriate type
                switch (md.array.type)
                {
                case GGUF_TYPE_UINT8:
                    md.array.data = std::vector<uint8_t>(
                        reinterpret_cast<uint8_t *>(buffer.data()),
                        reinterpret_cast<uint8_t *>(buffer.data()) + md.array.size);
                    break;
                case GGUF_TYPE_INT8:
                    md.array.data = std::vector<int8_t>(
                        reinterpret_cast<int8_t *>(buffer.data()),
                        reinterpret_cast<int8_t *>(buffer.data()) + md.array.size);
                    break;
                case GGUF_TYPE_UINT16:
                    md.array.data = std::vector<uint16_t>(
                        reinterpret_cast<uint16_t *>(buffer.data()),
                        reinterpret_cast<uint16_t *>(buffer.data()) + md.array.size);
                    break;
                case GGUF_TYPE_INT16:
                    md.array.data = std::vector<int16_t>(
                        reinterpret_cast<int16_t *>(buffer.data()),
                        reinterpret_cast<int16_t *>(buffer.data()) + md.array.size);
                    break;
                case GGUF_TYPE_UINT32:
                    md.array.data = std::vector<uint32_t>(
                        reinterpret_cast<uint32_t *>(buffer.data()),
                        reinterpret_cast<uint32_t *>(buffer.data()) + md.array.size);
                    break;
                case GGUF_TYPE_INT32:
                    md.array.data = std::vector<int32_t>(
                        reinterpret_cast<int32_t *>(buffer.data()),
                        reinterpret_cast<int32_t *>(buffer.data()) + md.array.size);
                    break;
                case GGUF_TYPE_FLOAT32:
                    md.array.data = std::vector<float>(
                        reinterpret_cast<float *>(buffer.data()),
                        reinterpret_cast<float *>(buffer.data()) + md.array.size);
                    break;
                case GGUF_TYPE_UINT64:
                    md.array.data = std::vector<uint64_t>(
                        reinterpret_cast<uint64_t *>(buffer.data()),
                        reinterpret_cast<uint64_t *>(buffer.data()) + md.array.size);
                    break;
                case GGUF_TYPE_INT64:
                    md.array.data = std::vector<int64_t>(
                        reinterpret_cast<int64_t *>(buffer.data()),
                        reinterpret_cast<int64_t *>(buffer.data()) + md.array.size);
                    break;
                case GGUF_TYPE_FLOAT64:
                    md.array.data = std::vector<double>(
                        reinterpret_cast<double *>(buffer.data()),
                        reinterpret_cast<double *>(buffer.data()) + md.array.size);
                    break;
                default:
                    std::cerr << "Unimplemented array type: " << md.array.type << "\n";
                    return GGUFMetadata();
                }
            }
            break;
        }
        default:
            std::cerr << "Unknown metadata type: " << md.type << "\n";
            return GGUFMetadata();
        }

        // Read end marker
        std::getline(in, marker);
        if (marker != "---END_ITEM---")
        {
            std::cerr << "Invalid item end marker: " << marker << "\n";
            return GGUFMetadata();
        }

        if (md.key == key)
        {
            in.close();
            return md;
        }
    }

    in.close();
    return GGUFMetadata(); // Not found
}



bool tensor_name_matches(const std::string &stored_name, const std::string &search_name) {
    // Si son exactamente iguales, coinciden
    if (stored_name == search_name) {
        return true;
    }
    
    // Verificar si search_name es stored_name + algún sufijo común
    if (search_name.size() > stored_name.size() &&
        search_name.substr(0, stored_name.size()) == stored_name &&
        search_name[stored_name.size()] == '.') {
        // Los sufijos comunes pueden ser: weight, bias, etc.
        std::string suffix = search_name.substr(stored_name.size() + 1);
        if (suffix == "weight" || suffix == "bias" || suffix == "scale" || suffix == "offset") {
            return true;
        }
    }
    
    return false;
}

/**
 * @brief Reads a tensor by name from a .graph file
 * @param filename Input .graph filename
 * @param name Tensor name to search for
 * @return GGUFTensor structure if found, or empty tensor with name="" if not found
 */
GGUFTensor read_tensor(const std::string &filename, const std::string &name)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Failed to open " << filename << " for reading\n";
        return GGUFTensor();
    }

    // Verify magic string
    char magic[6];
    in.read(magic, 6);
    if (std::string(magic, 5) != "GRAPH" || magic[5] != '\n')
    {
        std::cerr << "Invalid file format - bad magic string\n";
        return GGUFTensor();
    }

    // buscar la seccion de tensores
    std::string line;
    while (std::getline(in, line))
    {
        if (line == "---TENSORS---")
            break;
    }

    if (line != "---TENSORS---")
    {
        std::cerr << "Invalid file format - missing tensors section: " << line << "\n";
        return GGUFTensor();
    }

    uint64_t tensors_count;
    in.read(reinterpret_cast<char *>(&tensors_count), sizeof(uint64_t));

    // Verify that the reading was successful
    if (!in)
    {
        std::cerr << "Error reading the number of tensors\n";
        return GGUFTensor();
    }

    // Discard the following '\n' (if any)
    //if (in.peek() == '\n')
    //    in.ignore(1);

    in.ignore(1);

    // Search for the tensor
    for (uint64_t i = 0; i < tensors_count; ++i)
    {
        GGUFTensor tensor;

        // Find tensor start
        while (std::getline(in, line))
        {
            if (line == "---BEGIN_TENSOR---")
                break;
        }

        // Read tensor metadata
        while (std::getline(in, line))
        {
            if (line.empty())
                continue;
            if (line == "DATA_START:")
                break;

            if (line.find("NAME: ") == 0)
            {
                tensor.name = line.substr(6);
            }
            else if (line.find("TYPE: ") == 0)
            {
                tensor.type = static_cast<enum ggml_type>(std::stoi(line.substr(6)));
            }
            else if (line.find("SIZE: ") == 0)
            {
                tensor.size = std::stoull(line.substr(6));
            }
            else if (line.find("NDIMS: ") == 0)
            {
                tensor.n_dims = std::stoi(line.substr(7));
            }
            else if (line.find("DIMS: ") == 0)
            {

                std::string dims_str = line.substr(6);
                std::istringstream dims_stream(dims_str);
                tensor.dims.clear();
                
                int64_t dim;
                while (dims_stream >> dim) {
                    tensor.dims.push_back(dim);
                }
            }
        }

        // Check if this is the tensor we want
        //if (!tensor_name_matches(tensor.name, name))

        //std::cout << "Searching for tensor: " << name << "\n";
        //std::cout << "Found tensor: " << tensor.name << "\n";
        
        if (tensor.name != name)
        {
            // Skip binary data
            in.seekg(tensor.size, std::ios::cur);
            // Skip remaining tensor lines
            while (std::getline(in, line) && line != "---END_TENSOR---")
            {
            }
            continue;
        }

        // Read tensor data
        size_t element_count = tensor.size / sizeof(float);
        tensor.data = read_array<float>(in, element_count);

        // Skip remaining tensor lines
        while (std::getline(in, line) && line != "---END_TENSOR---")
        {
        }

        in.close();
        return tensor;
    }

    in.close();
    return GGUFTensor(); // Not found
}



/**
 * @brief Reads an array of binary data from an input stream
 * 
 * This helper function efficiently reads binary data of a specified type
 * from an input file stream into a std::vector. It handles proper type
 * conversion and memory management.
 * 
 * @tparam T The data type to read (must be a POD/trivially-copyable type)
 * @param in Input file stream (must be open and in good state)
 * @param size Number of elements to read
 * @return std::vector<T> containing the read data
 * 
 * @note The function will read exactly size*sizeof(T) bytes from the stream
 * @warning No bounds checking is performed - ensure stream has sufficient data
 * @warning The stream must be in binary mode for correct operation
 */
template <typename T>
std::vector<T> read_array(std::ifstream &in, size_t size)
{
    // Create output vector with requested size (initializes memory)
    std::vector<T> data(size);
    
    // Directly read binary data into vector's underlying storage
    // - Uses reinterpret_cast for type-safe pointer conversion
    // - Reads exactly size*sizeof(T) bytes from current stream position
    in.read(reinterpret_cast<char *>(data.data()), size * sizeof(T));
    
    // Return vector by move semantics (efficient transfer)
    return data;
}

////////////////////////////////////////////////////////////////////////////////////

void print_graph_data_from_struct(const GraphData &graph_data, const char *output_filename)
{
    std::ofstream outfile(output_filename);
    if (!outfile)
    {
        std::cerr << "The output file could not be opened: " << output_filename << "\n";
        return;
    }

    // Helper function for GGUF type names
    auto gguf_type_name = [](enum gguf_type type) -> const char *
    {
        static const char *names[] = {
            "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32",
            "FLOAT32", "BOOL", "STRING", "UINT64", "INT64", "FLOAT64", "ARRAY"};
        return (type >= GGUF_TYPE_UINT8 && type <= GGUF_TYPE_ARRAY) ? names[type] : "UNKNOWN";
    };

    const size_t max_elements = 20; // Show only the first 20 items

    outfile << "\n**************************************************************\n";
    outfile << "**************************  HEADER  **************************\n";
    outfile << "**************************************************************\n";
    outfile << "Number of tensors:" << graph_data.header.n_tensors << "\n";
    outfile << "Number of key-value pairs: " << graph_data.header.n_kv << "\n";

    outfile << "\n**************************************************************\n";
    outfile << "*************************  METADATA  *************************\n";
    outfile << "**************************************************************\n";

    for (const auto &md : graph_data.metadata)
    {
        outfile << "Key: " << md.key << "\n";
        outfile << "Value: " << gguf_type_name(md.type) << "\n";

        switch (md.type)
        {
        case GGUF_TYPE_UINT8:
            outfile << "Value: " << static_cast<int>(md.value.u8) << " (uint8)\n";
            break;
        case GGUF_TYPE_INT8:
            outfile << "Value: " << static_cast<int>(md.value.i8) << " (int8)\n";
            break;
        case GGUF_TYPE_UINT16:
            outfile << "Value: " << md.value.u16 << " (uint16)\n";
            break;
        case GGUF_TYPE_INT16:
            outfile << "Value: " << md.value.i16 << " (int16)\n";
            break;
        case GGUF_TYPE_UINT32:
            outfile << "Value: " << md.value.u32 << " (uint32)\n";
            break;
        case GGUF_TYPE_INT32:
            outfile << "Value: " << md.value.i32 << " (int32)\n";
            break;
        case GGUF_TYPE_FLOAT32:
            outfile << std::fixed << std::setprecision(6);
            outfile << "Value: " << md.value.f32 << " (float32)\n";
            outfile.unsetf(std::ios::fixed);
            outfile.precision(6);
            break;
        case GGUF_TYPE_BOOL:
            outfile << "Value: " << (md.value.b ? "true" : "false") << " (bool)\n";
            break;
        case GGUF_TYPE_STRING:
            outfile << "Value: " << md.str << " (string)\n";
            break;
        case GGUF_TYPE_UINT64:
            outfile << "Value: " << md.value.u64 << " (uint64)\n";
            break;
        case GGUF_TYPE_INT64:
            outfile << "Value: " << md.value.i64 << " (int64)\n";
            break;
        case GGUF_TYPE_FLOAT64:
            outfile << std::fixed << std::setprecision(6);
            outfile << "Value: " << md.value.f64 << " (float64)\n";
            outfile.unsetf(std::ios::fixed);
            outfile.precision(6);
            break;
        case GGUF_TYPE_ARRAY:
            outfile << "Array type: " << gguf_type_name(md.array.type) << "\n";
            outfile << "Array size: " << md.array.size << "\n";

            // Show first elements for known types
            if (md.array.type == GGUF_TYPE_STRING)
            {
                if (const auto *strs = std::get_if<std::vector<std::string>>(&md.array.data))
                {
                    outfile << "First strings: [";
                    for (size_t i = 0; i < std::min(strs->size(), max_elements); ++i)
                    {
                        outfile << "\"" << (*strs)[i] << "\" ";
                    }
                    outfile << "...]\n";
                }
            }
            else if (md.array.type == GGUF_TYPE_FLOAT32)
            {
                if (const auto *vals = std::get_if<std::vector<float>>(&md.array.data))
                {
                    outfile << std::fixed << std::setprecision(6);
                    outfile << "First values: [";
                    for (size_t i = 0; i < std::min(vals->size(), max_elements); ++i)
                    {
                        outfile << (*vals)[i] << " ";
                    }
                    outfile << "...]\n";
                    outfile.unsetf(std::ios::fixed);
                }
            }
            else if (md.array.type == GGUF_TYPE_INT32)
            {
                if (const auto *vals = std::get_if<std::vector<int32_t>>(&md.array.data))
                {
                    outfile << "First values: [";
                    for (size_t i = 0; i < std::min(vals->size(), max_elements); ++i)
                    {
                        outfile << (*vals)[i] << " ";
                    }
                    outfile << "...]\n";
                }
            }
            else if (md.array.type == GGUF_TYPE_UINT8)
            {
                if (const auto *vals = std::get_if<std::vector<uint8_t>>(&md.array.data))
                {
                    outfile << "First values: [";
                    for (size_t i = 0; i < std::min(vals->size(), max_elements); ++i)
                    {
                        outfile << static_cast<int>((*vals)[i]) << " ";
                    }
                    outfile << "...]\n";
                }
            }
            else
            {
                outfile << "[Binary data type " << gguf_type_name(md.array.type) << "]\n";
            }
            break;
        default:
            outfile << "Value: [unknown type]\n";
            break;
        }
        outfile << "-----------------------------------\n";
    }

    outfile << "\n**************************************************************\n";
    outfile << "*************************  TENSORS  **************************\n";
    outfile << "**************************************************************\n";

    for (const auto &tensor : graph_data.tensors)
    {
        outfile << "Name: " << tensor.name << "\n";
        outfile << "Type: " << ggml_type_name(tensor.type) << "\n";
        outfile << "Size: " << std::fixed << std::setprecision(2)
                << tensor.size / 1024.0f / 1024.0f << " MB\n";
        outfile << "Number of dimensions: " << tensor.n_dims << "\n";
        outfile << "Size of each dimension: ";
        for (const auto &dim : tensor.dims)
        {
            outfile << dim << " ";
        }

        outfile << "\n";

        // Display the first elements of the tensor according to its type
        outfile << "Data (first elements): ";

        switch (tensor.type)
        {
        case GGML_TYPE_F32:
            if (const auto *data = std::get_if<std::vector<float>>(&tensor.data))
            {
                for (size_t i = 0; i < std::min(data->size(), max_elements); ++i)
                {
                    outfile << (*data)[i] << " ";
                }
            }
            break;

        case GGML_TYPE_I32:
            if (const auto *data = std::get_if<std::vector<int32_t>>(&tensor.data))
            {
                for (size_t i = 0; i < std::min(data->size(), max_elements); ++i)
                {
                    outfile << (*data)[i] << " ";
                }
            }
            break;

        case GGML_TYPE_F16:
            if (const auto *data = std::get_if<std::vector<uint16_t>>(&tensor.data))
            {
                for (size_t i = 0; i < std::min(data->size(), max_elements); ++i)
                {
                    outfile << (*data)[i] << " ";
                }
            }
            break;

        case GGML_TYPE_I8:
            if (const auto *data = std::get_if<std::vector<int8_t>>(&tensor.data))
            {
                for (size_t i = 0; i < std::min(data->size(), max_elements); ++i)
                {
                    outfile << static_cast<int>((*data)[i]) << " ";
                }
            }
            break;

        // Quantized types
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            if (const auto *data = std::get_if<std::vector<uint8_t>>(&tensor.data))
            {
                // outfile << "[Datos cuantizados - " << data->size() << " bytes]";
                outfile << "[Quantized data: taken in groups of 8 bits in this case]  ";
                for (size_t i = 0; i < std::min(data->size(), max_elements); ++i)
                {
                    outfile << static_cast<uint8_t>((*data)[i]) << " ";
                }
            }
            break;

        default:
            if (const auto *data = std::get_if<std::vector<uint8_t>>(&tensor.data))
            {
                outfile << "[Binary data - " << data->size() << " bytes]";
            }
            break;
        }

        // Indicate if there are more items
        if (std::visit([](const auto &v)
                       { return v.size(); }, tensor.data) > max_elements)
        {
            outfile << "... [total: "
                    << std::visit([](const auto &v)
                                  { return v.size(); }, tensor.data)
                    << " elements]";
        }

        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
    }

    // Cerrar el archivo de salida
    outfile.close();
    std::cout << "GraphData information has been written to the file: " << output_filename << "\n";
}

void print_metadata_from_file(const std::string &input_filename, const std::string &output_filename)
{
    // Open output file first
    std::ofstream outfile(output_filename);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening output file: " << output_filename << std::endl;
        return;
    }

    const size_t max_elements = 50;

    // Read the metadata
    GGUFMetadata md = read_metadata(input_filename, "tokenizer.ggml.tokens");

    // Check if metadata was found
    if (md.type == GGUF_TYPE_COUNT)
    {
        outfile << "Metadata not found in file.\n";
        outfile.close();
        return;
    }

    // Write metadata to file
    outfile << "\nKey: " << md.key << "\n";
    outfile << "Type: " << gguf_type_name(md.type) << "\n";
    outfile << "Value: ";

    switch (md.type)
    {
    case GGUF_TYPE_UINT8:
        outfile << static_cast<int>(md.value.u8);
        break;
    case GGUF_TYPE_INT8:
        outfile << static_cast<int>(md.value.i8);
        break;
    case GGUF_TYPE_UINT16:
        outfile << md.value.u16;
        break;
    case GGUF_TYPE_INT16:
        outfile << md.value.i16;
        break;
    case GGUF_TYPE_UINT32:
        outfile << md.value.u32;
        break;
    case GGUF_TYPE_INT32:
        outfile << md.value.i32;
        break;
    case GGUF_TYPE_FLOAT32:
        outfile << std::fixed << std::setprecision(6) << md.value.f32;
        break;
    case GGUF_TYPE_UINT64:
        outfile << md.value.u64;
        break;
    case GGUF_TYPE_INT64:
        outfile << md.value.i64;
        break;
    case GGUF_TYPE_FLOAT64:
        outfile << std::fixed << std::setprecision(6) << md.value.f64;
        break;
    case GGUF_TYPE_BOOL:
        outfile << (md.value.b ? "true" : "false");
        break;
    case GGUF_TYPE_STRING:
        outfile << md.str;
        break;
    case GGUF_TYPE_ARRAY:
        outfile << "Array[" << md.array.size << "] of " << gguf_type_name(md.array.type) << "\n";

        if (auto v = std::get_if<std::vector<std::string>>(&md.array.data))
        {
            outfile << "  [";
            for (size_t j = 0; j < std::min(v->size(), max_elements); ++j)
            {
                if (j > 0)
                    outfile << ", ";
                outfile << "\"" << (*v)[j] << "\"";
            }
            if (v->size() > max_elements)
                outfile << ", ...";
            outfile << "]";
        }
        else
        {
            outfile << "  [";

            auto print_numeric_array = [&](const auto &vec)
            {
                for (size_t j = 0; j < std::min(vec.size(), max_elements); ++j)
                {
                    if (j > 0)
                        outfile << ", ";
                    outfile << vec[j];
                }
                if (vec.size() > max_elements)
                    outfile << ", ...";
            };

            if (auto v = std::get_if<std::vector<uint8_t>>(&md.array.data))
            {
                print_numeric_array(*v);
            }
            else if (auto v = std::get_if<std::vector<int8_t>>(&md.array.data))
            {
                print_numeric_array(*v);
            }
            else if (auto v = std::get_if<std::vector<uint16_t>>(&md.array.data))
            {
                print_numeric_array(*v);
            }
            else if (auto v = std::get_if<std::vector<int16_t>>(&md.array.data))
            {
                print_numeric_array(*v);
            }
            else if (auto v = std::get_if<std::vector<uint32_t>>(&md.array.data))
            {
                print_numeric_array(*v);
            }
            else if (auto v = std::get_if<std::vector<int32_t>>(&md.array.data))
            {
                print_numeric_array(*v);
            }
            else if (auto v = std::get_if<std::vector<float>>(&md.array.data))
            {
                outfile << std::fixed << std::setprecision(6);
                print_numeric_array(*v);
            }
            else if (auto v = std::get_if<std::vector<uint64_t>>(&md.array.data))
            {
                print_numeric_array(*v);
            }
            else if (auto v = std::get_if<std::vector<int64_t>>(&md.array.data))
            {
                print_numeric_array(*v);
            }
            else if (auto v = std::get_if<std::vector<double>>(&md.array.data))
            {
                outfile << std::fixed << std::setprecision(6);
                print_numeric_array(*v);
            }
            else
            {
                outfile << "<unsupported array type>";
            }

            outfile << "]";
        }
        break;
    default:
        outfile << "<unsupported type>";
    }
    outfile << "\n-----------------------------------\n";

    outfile.flush();
    outfile.close();

    std::cout << "File information " << input_filename
              << " has been written in: " << output_filename << "\n";
}

void print_tensor_data_from_file(const std::string &input_filename, const std::string &output_filename)
{
    // Open output file
    std::ofstream outfile(output_filename);
    if (!outfile.is_open())
    {
        std::cerr << "Error opening output file: " << output_filename << std::endl;
        return;
    }

    const size_t max_elements = 50; // Limit of items to display

    // Read the tensor
    GGUFTensor tensor = read_tensor(input_filename, "blk.0.ffn_down.weight");

    // Check if tensor was found
    if (tensor.name.empty())
    {
        std::cout << "Tensor not found in file.\n";
        outfile.close();
        return;
    }

    // Write tensor info to file
    outfile << "\nTensor: " << tensor.name << "\n";
    outfile << "Original type: " << ggml_type_name(tensor.type) << " (stored as F32)\n";
    outfile << "Size: " << tensor.size << " bytes\n";
    outfile << "Dimensions: " << tensor.n_dims << "\n";
    outfile << "Shape: [";
    for (size_t i = 0; i < tensor.dims.size(); ++i)
    {
        if (i > 0)
            outfile << " × ";
        outfile << tensor.dims[i];
    }
    outfile << "]\n";
    outfile << "Operation: " << tensor.op << "\n";

    // Handle tensor data
    outfile << "Data (first " << max_elements << " elements):\n";

    if (auto float_data = tensor.get_data<float>())
    {
        // Calculate total elements
        size_t total_elements = tensor.size / sizeof(float);
        size_t elements_to_show = std::min(total_elements, max_elements);

        outfile << "[";
        for (size_t j = 0; j < elements_to_show; ++j)
        {
            if (j > 0)
                outfile << ", ";
            outfile << std::fixed << std::setprecision(6) << (*float_data)[j];
        }
        if (total_elements > max_elements)
            outfile << ", ...";
        outfile << "]\n";
    }
    else
    {
        outfile << "<The tensor data could not be read>\n";
    }

    outfile << "\n-----------------------------------\n";

    outfile.flush();
    outfile.close();

    std::cout << "The tensor information has been written to: " << output_filename << "\n";
}
