#include "graph_file.h"
#include "gguf.h"
#include <fstream>
#include <iostream>
#include <iomanip>

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

    // 1. Cabecera de identificación y versión
    const std::string magic = "GRAPH";
    out.write(magic.c_str(), magic.size());
    out.put('\n');

    // 2. Encabezado
    out.write(reinterpret_cast<const char *>(&graph_data.header), sizeof(GGUFHeader));
    out.put('\n');

    const std::string metadata_marker = "---METADATA---";

    out.write(metadata_marker.c_str(), metadata_marker.size());
    out.put('\n');

    // 3. Metadatos
    uint32_t metadata_count = graph_data.metadata.size();
    out.write(reinterpret_cast<const char *>(&metadata_count), sizeof(uint32_t));
    out.put('\n');

    for (const auto &md : graph_data.metadata)
    {
        // Clave
        out << md.key;
        out.put('\n');

        // Tipo
        out.write(reinterpret_cast<const char *>(&md.type), sizeof(enum gguf_type));
        out.put('\n');

        // Valor según tipo
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

    // 4. Tensores
    const std::string tensors_marker = "---TENSORS---";
    out.write(tensors_marker.c_str(), tensors_marker.size());
    out.put('\n');

    uint32_t tensors_count = graph_data.tensors.size(); // Número de tensores
    out.write(reinterpret_cast<const char *>(&tensors_count), sizeof(uint32_t));
    out.put('\n');

    for (const auto &tensor : graph_data.tensors)
    {
        // Delimitador de inicio
        out << "---BEGIN_TENSOR---";
        out.put('\n');

        // 1. Información básica
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

        // 2. Datos del tensor (siempre float, aunque el tipo indique cuantización)
        out << "DATA_START:";
        out.put('\n');

        // Acceso directo a los datos float con verificación
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

        // Delimitador final
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

    // 2. Read GGUF header
    GGUFHeader header;
    in.read(reinterpret_cast<char *>(&header), sizeof(GGUFHeader));

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
    uint32_t metadata_count;
    in.read(reinterpret_cast<char *>(&metadata_count), sizeof(uint32_t));
    if (in.get() != '\n')
    {
        std::cerr << "Invalid metadata count format\n";
        return GGUFMetadata();
    }

    // 5. Search for the requested key
    for (uint32_t i = 0; i < metadata_count; ++i)
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
                    std::getline(in, str);
                }
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

    // 1. Verify magic string
    char magic[6];
    in.read(magic, 6);
    if (std::string(magic, 5) != "GRAPH" || magic[5] != '\n')
    {
        std::cerr << "Invalid file format\n";
        return GGUFTensor();
    }

    // 2. Read header
    GGUFHeader header;
    in.read(reinterpret_cast<char *>(&header), sizeof(GGUFHeader));
    if (in.get() != '\n')
    {
        std::cerr << "Invalid header format\n";
        return GGUFTensor();
    }

    // 3. Skip metadata section
    std::string marker;
    std::getline(in, marker); // Read metadata marker
    if (marker != "---METADATA---")
    {
        std::cerr << "Invalid metadata marker\n";
        return GGUFTensor();
    }

    // Read and skip metadata count
    uint32_t metadata_count;
    in.read(reinterpret_cast<char *>(&metadata_count), sizeof(uint32_t));
    in.ignore(1); // Skip \n

    // Skip each metadata item
    for (uint32_t i = 0; i < metadata_count; ++i)
    {
        std::getline(in, marker);              // Skip key
        in.ignore(sizeof(enum gguf_type) + 1); // Skip type and \n

        // Skip value based on type (simplified)
        std::getline(in, marker); // Skip value line
        std::getline(in, marker); // Skip end marker
    }

    // 4. Read tensors marker
    std::getline(in, marker);
    if (marker != "---TENSORS---")
    {
        std::cerr << "Invalid tensors marker\n";
        return GGUFTensor();
    }

    // 5. Read tensors count
    uint32_t tensors_count;
    in.read(reinterpret_cast<char *>(&tensors_count), sizeof(uint32_t));
    if (in.get() != '\n')
    {
        std::cerr << "Invalid tensors count format\n";
        return GGUFTensor();
    }

    // 6. Search for the requested tensor
    for (uint32_t i = 0; i < tensors_count; ++i)
    {
        GGUFTensor tensor;

        // Verify tensor start marker
        std::getline(in, marker);
        if (marker != "---BEGIN_TENSOR---")
        {
            std::cerr << "Invalid tensor start marker\n";
            return GGUFTensor();
        }

        // Read tensor metadata
        std::string line;
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
            else if (line.find("ORIGINAL_TYPE(STORED_AS_F32): ") == 0)
            {
                tensor.type = static_cast<enum ggml_type>(std::stoi(line.substr(29)));
            }
            else if (line.find("DATA_SIZE: ") == 0)
            {
                tensor.size = std::stoull(line.substr(11));
            }
            else if (line.find("NDIMS: ") == 0)
            {
                tensor.n_dims = std::stoi(line.substr(7));
            }
            else if (line.find("DIMS: ") == 0)
            {
                std::istringstream dims_stream(line.substr(6));
                int64_t dim;
                while (dims_stream >> dim)
                {
                    tensor.dims.push_back(dim);
                }
            }
            else if (line.find("OP:") == 0)
            {
                tensor.op = static_cast<enum ggml_op>(std::stoi(line.substr(3)));
            }
        }

        // Check if this is the tensor we want
        if (tensor.name != name)
        {
            // Skip to end of tensor
            while (std::getline(in, line) && line != "---END_TENSOR---")
            {
            }
            continue;
        }

        // Read tensor data
        size_t element_count = tensor.size / sizeof(float);
        tensor.data = read_array<float>(in, element_count);

        // Verify data end markers
        std::getline(in, line); // Should be empty line after binary data
        std::getline(in, line);
        if (line != "DATA_END")
        {
            std::cerr << "Invalid data end marker\n";
            return GGUFTensor();
        }

        // Verify tensor end marker
        std::getline(in, line);
        if (line != "---END_TENSOR---")
        {
            std::cerr << "Invalid tensor end marker\n";
            return GGUFTensor();
        }

        in.close();
        return tensor;
    }

    in.close();
    return GGUFTensor(); // Not found
}

// Helper function to skip metadata when searching for tensors
void skip_metadata(std::ifstream &in)
{
    // Saltar clave
    std::string line;
    std::getline(in, line);

    // Saltar tipo y valor
    enum gguf_type type;
    in.read(reinterpret_cast<char *>(&type), sizeof(enum gguf_type));
    in.ignore(1); // Saltar \n

    switch (type)
    {
    case GGUF_TYPE_UINT8:
        in.ignore(sizeof(uint8_t));
        break;
    case GGUF_TYPE_INT8:
        in.ignore(sizeof(int8_t));
        break;
    // ... otros tipos básicos
    case GGUF_TYPE_STRING:
    {
        std::getline(in, line); // Saltar el string
        break;
    }
    case GGUF_TYPE_ARRAY:
    {
        enum gguf_type arr_type;
        size_t arr_size;
        in.read(reinterpret_cast<char *>(&arr_type), sizeof(enum gguf_type));
        in.read(reinterpret_cast<char *>(&arr_size), sizeof(size_t));
        in.ignore(1);                              // Saltar \n
        in.ignore(arr_size * type_size(arr_type)); // Saltar datos
        break;
    }
    default:
        break;
    }

    // Saltar delimitador final
    std::getline(in, line); // ---END_ITEM---
}

// Helper function to read array data
template <typename T>
std::vector<T> read_array(std::ifstream &in, size_t size)
{
    std::vector<T> data(size);
    in.read(reinterpret_cast<char *>(data.data()), size * sizeof(T));
    return data;
}

////////////////////////////////////////////////////////////////////////////////

void print_graph_data_from_struct(const GraphData &graph_data, const char *output_filename)
{
    // Abrir el archivo de salida
    std::ofstream outfile(output_filename);
    if (!outfile)
    {
        std::cerr << "No se pudo abrir el archivo de salida: " << output_filename << "\n";
        return;
    }

    // Función helper para nombres de tipos GGUF
    auto gguf_type_name = [](enum gguf_type type) -> const char *
    {
        static const char *names[] = {
            "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32",
            "FLOAT32", "BOOL", "STRING", "UINT64", "INT64", "FLOAT64", "ARRAY"};
        return (type >= GGUF_TYPE_UINT8 && type <= GGUF_TYPE_ARRAY) ? names[type] : "UNKNOWN";
    };

    const size_t max_elements = 20; // Mostrar solo los primeros 20 elementos

    // Escribir el encabezado
    outfile << "\n**************************************************************\n";
    outfile << "**************************  HEADER  **************************\n";
    outfile << "**************************************************************\n";
    outfile << "Número de tensores: " << graph_data.header.n_tensors << "\n";
    outfile << "Número de pares clave-valor: " << graph_data.header.n_kv << "\n";

    // Escribir los metadatos (pares clave-valor)
    outfile << "\n**************************************************************\n";
    outfile << "*************************  METADATA  *************************\n";
    outfile << "**************************************************************\n";

    for (const auto &md : graph_data.metadata)
    {
        outfile << "Clave: " << md.key << "\n";
        outfile << "Tipo: " << gguf_type_name(md.type) << "\n";

        switch (md.type)
        {
        case GGUF_TYPE_UINT8:
            outfile << "Valor: " << static_cast<int>(md.value.u8) << " (uint8)\n";
            break;
        case GGUF_TYPE_INT8:
            outfile << "Valor: " << static_cast<int>(md.value.i8) << " (int8)\n";
            break;
        case GGUF_TYPE_UINT16:
            outfile << "Valor: " << md.value.u16 << " (uint16)\n";
            break;
        case GGUF_TYPE_INT16:
            outfile << "Valor: " << md.value.i16 << " (int16)\n";
            break;
        case GGUF_TYPE_UINT32:
            outfile << "Valor: " << md.value.u32 << " (uint32)\n";
            break;
        case GGUF_TYPE_INT32:
            outfile << "Valor: " << md.value.i32 << " (int32)\n";
            break;
        case GGUF_TYPE_FLOAT32:
            outfile << std::fixed << std::setprecision(6);
            outfile << "Valor: " << md.value.f32 << " (float32)\n";
            outfile.unsetf(std::ios::fixed);
            outfile.precision(6);
            break;
        case GGUF_TYPE_BOOL:
            outfile << "Valor: " << (md.value.b ? "true" : "false") << " (bool)\n";
            break;
        case GGUF_TYPE_STRING:
            outfile << "Valor: " << md.str << " (string)\n";
            break;
        case GGUF_TYPE_UINT64:
            outfile << "Valor: " << md.value.u64 << " (uint64)\n";
            break;
        case GGUF_TYPE_INT64:
            outfile << "Valor: " << md.value.i64 << " (int64)\n";
            break;
        case GGUF_TYPE_FLOAT64:
            outfile << std::fixed << std::setprecision(6);
            outfile << "Valor: " << md.value.f64 << " (float64)\n";
            outfile.unsetf(std::ios::fixed);
            outfile.precision(6);
            break;
        case GGUF_TYPE_ARRAY:
            outfile << "Tipo de array: " << gguf_type_name(md.array.type) << "\n";
            outfile << "Tamaño del array: " << md.array.size << "\n";

            // Mostrar primeros elementos para tipos conocidos
            if (md.array.type == GGUF_TYPE_STRING)
            {
                if (const auto *strs = std::get_if<std::vector<std::string>>(&md.array.data))
                {
                    outfile << "Primeros strings: [";
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
                    outfile << "Primeros valores: [";
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
                    outfile << "Primeros valores: [";
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
                    outfile << "Primeros valores: [";
                    for (size_t i = 0; i < std::min(vals->size(), max_elements); ++i)
                    {
                        outfile << static_cast<int>((*vals)[i]) << " ";
                    }
                    outfile << "...]\n";
                }
            }
            else
            {
                outfile << "[Datos binarios de tipo " << gguf_type_name(md.array.type) << "]\n";
            }
            break;
        default:
            outfile << "Valor: [tipo desconocido]\n";
            break;
        }
        outfile << "-----------------------------------\n";
    }

    // Escribir información de tensores
    outfile << "\n**************************************************************\n";
    outfile << "*************************  TENSORS  **************************\n";
    outfile << "**************************************************************\n";

    for (const auto &tensor : graph_data.tensors)
    {
        outfile << "Nombre: " << tensor.name << "\n";
        outfile << "Tipo: " << ggml_type_name(tensor.type) << "\n";
        outfile << "Tamaño: " << std::fixed << std::setprecision(2)
                << tensor.size / 1024.0f / 1024.0f << " MB\n";
        outfile << "Número de dimensiones: " << tensor.n_dims << "\n";
        outfile << "Tamaño de cada dimensión: ";
        for (const auto &dim : tensor.dims)
        {
            outfile << dim << " ";
        }

        outfile << "\n";

        // Mostrar los primeros elementos del tensor según su tipo
        outfile << "Datos (primeros elementos): ";

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

        // Tipos cuantizados
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            if (const auto *data = std::get_if<std::vector<uint8_t>>(&tensor.data))
            {
                // outfile << "[Datos cuantizados - " << data->size() << " bytes]";
                outfile << "[Datos cuantizados: se toman en grupos de 8 bits en este caso]  ";
                for (size_t i = 0; i < std::min(data->size(), max_elements); ++i)
                {
                    outfile << static_cast<uint8_t>((*data)[i]) << " ";
                }
            }
            break;

        default:
            if (const auto *data = std::get_if<std::vector<uint8_t>>(&tensor.data))
            {
                outfile << "[Datos binarios - " << data->size() << " bytes]";
            }
            break;
        }

        // Indicar si hay más elementos
        if (std::visit([](const auto &v)
                       { return v.size(); }, tensor.data) > max_elements)
        {
            outfile << "... [total: "
                    << std::visit([](const auto &v)
                                  { return v.size(); }, tensor.data)
                    << " elementos]";
        }

        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
        outfile << "\n --------------------------------------------------------------- \n";
    }

    // Cerrar el archivo de salida
    outfile.close();
    std::cout << "La información de GraphData se ha escrito en el archivo: " << output_filename << "\n";
}

// Función helper para obtener el tamaño en bytes de un tipo GGUF
size_t gguf_type_size(enum gguf_type type)
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
    case GGUF_TYPE_UINT64:
        return sizeof(uint64_t);
    case GGUF_TYPE_INT64:
        return sizeof(int64_t);
    case GGUF_TYPE_FLOAT64:
        return sizeof(double);
    case GGUF_TYPE_BOOL:
        return sizeof(bool);
    default:
        return 0; // Para tipos sin tamaño fijo (STRING, ARRAY)
    }
}


void print_graph_data_from_file(const std::string &input_filename, const std::string &output_filename) {
    // Open output file first
    std::ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        std::cerr << "Error al abrir archivo de salida: " << output_filename << std::endl;
        return;
    }

    const size_t max_elements = 50;

    // Read the metadata
    GGUFMetadata md = read_metadata(input_filename, "tokenizer.ggml.tokens");

    // Check if metadata was found
    if (md.type == GGUF_TYPE_COUNT) {
        outfile << "Metadata not found in file.\n";
        outfile.close();
        return;
    }

    // Write metadata to file
    outfile << "\nClave: " << md.key << "\n";
    outfile << "Tipo: " << gguf_type_name(md.type) << "\n";
    outfile << "Valor: ";

    switch (md.type) {
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
            
            if (auto v = std::get_if<std::vector<std::string>>(&md.array.data)) {
                outfile << "  [";
                for (size_t j = 0; j < std::min(v->size(), max_elements); ++j) {
                    if (j > 0) outfile << ", ";
                    outfile << "\"" << (*v)[j] << "\"";
                }
                if (v->size() > max_elements) outfile << ", ...";
                outfile << "]";
            }
            else {
                outfile << "  [";
                
                auto print_numeric_array = [&](const auto& vec) {
                    for (size_t j = 0; j < std::min(vec.size(), max_elements); ++j) {
                        if (j > 0) outfile << ", ";
                        outfile << vec[j];
                    }
                    if (vec.size() > max_elements) outfile << ", ...";
                };

                if (auto v = std::get_if<std::vector<uint8_t>>(&md.array.data)) {
                    print_numeric_array(*v);
                }
                else if (auto v = std::get_if<std::vector<int8_t>>(&md.array.data)) {
                    print_numeric_array(*v);
                }
                else if (auto v = std::get_if<std::vector<uint16_t>>(&md.array.data)) {
                    print_numeric_array(*v);
                }
                else if (auto v = std::get_if<std::vector<int16_t>>(&md.array.data)) {
                    print_numeric_array(*v);
                }
                else if (auto v = std::get_if<std::vector<uint32_t>>(&md.array.data)) {
                    print_numeric_array(*v);
                }
                else if (auto v = std::get_if<std::vector<int32_t>>(&md.array.data)) {
                    print_numeric_array(*v);
                }
                else if (auto v = std::get_if<std::vector<float>>(&md.array.data)) {
                    outfile << std::fixed << std::setprecision(6);
                    print_numeric_array(*v);
                }
                else if (auto v = std::get_if<std::vector<uint64_t>>(&md.array.data)) {
                    print_numeric_array(*v);
                }
                else if (auto v = std::get_if<std::vector<int64_t>>(&md.array.data)) {
                    print_numeric_array(*v);
                }
                else if (auto v = std::get_if<std::vector<double>>(&md.array.data)) {
                    outfile << std::fixed << std::setprecision(6);
                    print_numeric_array(*v);
                }
                else {
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

    std::cout << "La información del archivo " << input_filename
              << " se ha escrito en: " << output_filename << "\n";
}