#include "graph_file.h"
#include "gguf.h"
#include <fstream>
#include <iostream>
#include <iomanip>


size_t type_size(enum gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return sizeof(uint8_t);
        case GGUF_TYPE_INT8:    return sizeof(int8_t);
        case GGUF_TYPE_UINT16:  return sizeof(uint16_t);
        case GGUF_TYPE_INT16:   return sizeof(int16_t);
        case GGUF_TYPE_UINT32:  return sizeof(uint32_t);
        case GGUF_TYPE_INT32:   return sizeof(int32_t);
        case GGUF_TYPE_FLOAT32: return sizeof(float);
        case GGUF_TYPE_BOOL:    return sizeof(bool);
        case GGUF_TYPE_STRING:  return sizeof(uint32_t); // string size is stored as uint32_t
        case GGUF_TYPE_ARRAY:   return sizeof(uint32_t); // array length is stored as uint32_t
        case GGUF_TYPE_UINT64:  return sizeof(uint64_t);
        case GGUF_TYPE_INT64:   return sizeof(int64_t);
        case GGUF_TYPE_FLOAT64: return sizeof(double);
        default:                return 0; // unknown type
    }
}

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

    // 1. Cabecera de identificación y versión
    const std::string magic = "GRAPH\n";
    out.write(magic.c_str(), magic.size());

    // 2. Encabezado GGUF
    out.write(reinterpret_cast<const char*>(&graph_data.header), sizeof(GGUFHeader));
    
    const std::string metadata_marker = "\n---METADATA---\n";
    out.write(metadata_marker.c_str(), metadata_marker.size());

    // 3. Metadatos
    uint64_t metadata_count = graph_data.metadata.size();
    out.write(reinterpret_cast<const char*>(&metadata_count), sizeof(uint64_t));
    out.put('\n');

    for (const auto& md : graph_data.metadata) {
        // Clave
        out << md.key << "\n";
        
        // Tipo
        out.write(reinterpret_cast<const char*>(&md.type), sizeof(enum gguf_type));
        out.put('\n');

        // Valor según tipo
        switch (md.type) {
            case GGUF_TYPE_UINT8:
                out << static_cast<int>(md.value.u8) << "\n";
                break;
            case GGUF_TYPE_INT8:
                out << static_cast<int>(md.value.i8) << "\n";
                break;
            case GGUF_TYPE_UINT16:
                out << md.value.u16 << "\n";
                break;
            case GGUF_TYPE_INT16:
                out << md.value.i16 << "\n";
                break;
            case GGUF_TYPE_UINT32:
                out << md.value.u32 << "\n";
                break;
            case GGUF_TYPE_INT32:
                out << md.value.i32 << "\n";
                break;
            case GGUF_TYPE_FLOAT32:
                out << std::fixed << std::setprecision(6) << md.value.f32 << "\n";
                break;
            case GGUF_TYPE_UINT64:
                out << md.value.u64 << "\n";
                break;
            case GGUF_TYPE_INT64:
                out << md.value.i64 << "\n";
                break;
            case GGUF_TYPE_FLOAT64:
                out << std::fixed << std::setprecision(6) << md.value.f64 << "\n";
                break;
            case GGUF_TYPE_BOOL:
                out << (md.value.b ? "true" : "false") << "\n";
                break;
            case GGUF_TYPE_STRING:
                out << md.str << "\n";
                break;
            case GGUF_TYPE_ARRAY:
                out.write(reinterpret_cast<const char*>(&md.array.type), sizeof(enum gguf_type));
                out.write(reinterpret_cast<const char*>(&md.array.size), sizeof(size_t));
                out.put('\n');
                
                if (std::holds_alternative<std::vector<std::string>>(md.array.data)) {
                    for (const auto& str : std::get<std::vector<std::string>>(md.array.data)) {
                        out << str << "\n";
                    }
                } else {
                    std::visit([&out](const auto& vec) {
                        using T = std::decay_t<decltype(vec)>::value_type;
                        out.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(T));
                    }, md.array.data);
                    out.put('\n');
                }
                break;
            default:
                std::cerr << "Unsupported metadata type: " << md.type << "\n";
                return false;
        }
        
        const std::string item_end_marker = "---END_ITEM---\n";
        out.write(item_end_marker.c_str(), item_end_marker.size());
    }

    // 4. Tensores 
    const std::string tensors_marker = "\n---TENSORS---\n";
    out.write(tensors_marker.c_str(), tensors_marker.size());
    
    out << graph_data.tensors.size() << "\n";  // Número de tensores

    for (const auto& tensor : graph_data.tensors) {
        // Delimitador de inicio
        out << "---BEGIN_TENSOR---\n";
        
        // 1. Información básica
        out << "NAME:" << tensor.name << "\n";
        out << "TYPE:" << tensor.type << "\n";  // Conservamos el tipo original (puede ser cuantizado)
        out << "STORED_AS:F32\n";  // Indicamos que está almacenado como float
        out << "SIZE:" << tensor.size << "\n";
        out << "NDIMS:" << tensor.n_dims << "\n";
        out << "DIMS:";
        for (const auto& dim : tensor.dims) {
            out << dim << " ";
        }
        out << "\n";
        out << "OP:" << tensor.op << "\n";

        // 2. Datos del tensor (siempre float, aunque el tipo indique cuantización)
        out << "DATA_SIZE:" << tensor.size << "\n";
        out << "DATA_START:";
        
        // Acceso directo a los datos float con verificación
        try {
            const auto& float_data = std::get<std::vector<float>>(tensor.data);
            out.write(reinterpret_cast<const char*>(float_data.data()), float_data.size() * sizeof(float));
        } catch (const std::bad_variant_access&) {
            std::cerr << "Error: Tensor " << tensor.name << " no contiene datos float\n";
            return false;
        }
        
        out << "\nDATA_END\n";
        
        // Delimitador final
        out << "---END_TENSOR---\n";
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

    // Verificar cabecera mágica
    char magic[6];
    in.read(magic, 6);
    if (std::string(magic, 6) != "GRAPH") {
        std::cerr << "Invalid file format\n";
        return GGUFMetadata();
    }

    // Leer encabezado GGUF
    GGUFHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(GGUFHeader));
    
    // Buscar inicio de metadatos
    std::string marker;
    std::getline(in, marker); // Leer hasta el fin de línea después del header
    std::getline(in, marker); // Leer el marcador de metadatos
    
    if (marker != "---METADATA---") {
        std::cerr << "Invalid metadata marker\n";
        return GGUFMetadata();
    }

    // Leer cantidad de metadatos
    uint64_t metadata_count;
    in >> metadata_count;
    in.ignore(1); // Saltar el \n

    // Buscar la clave solicitada
    for (uint64_t i = 0; i < metadata_count; ++i) {
        GGUFMetadata md;
        
        // Leer clave
        std::getline(in, md.key);
        
        // Leer tipo
        in.read(reinterpret_cast<char*>(&md.type), sizeof(enum gguf_type));
        in.ignore(1); // Saltar \n

        // Leer valor según tipo
        switch (md.type) {
            case GGUF_TYPE_UINT8: in >> md.value.u8; break;
            case GGUF_TYPE_INT8: in >> md.value.i8; break;
            case GGUF_TYPE_UINT16: in >> md.value.u16; break;
            case GGUF_TYPE_INT16: in >> md.value.i16; break;
            case GGUF_TYPE_UINT32: in >> md.value.u32; break;
            case GGUF_TYPE_INT32: in >> md.value.i32; break;
            case GGUF_TYPE_FLOAT32: in >> md.value.f32; break;
            case GGUF_TYPE_UINT64: in >> md.value.u64; break;
            case GGUF_TYPE_INT64: in >> md.value.i64; break;
            case GGUF_TYPE_FLOAT64: in >> md.value.f64; break;
            case GGUF_TYPE_BOOL: {
                std::string val;
                in >> val;
                md.value.b = (val == "true");
                break;
            }
            case GGUF_TYPE_STRING:
                std::getline(in, md.str);
                break;
            case GGUF_TYPE_ARRAY:
                in.read(reinterpret_cast<char*>(&md.array.type), sizeof(enum gguf_type));
                in.read(reinterpret_cast<char*>(&md.array.size), sizeof(size_t));
                in.ignore(1); // Saltar \n
                
                if (md.array.type == GGUF_TYPE_STRING) {
                    auto& arr = md.array.data.emplace<std::vector<std::string>>();
                    arr.resize(md.array.size);
                    for (auto& str : arr) {
                        std::getline(in, str);
                    }
                } else {
                    switch (md.array.type) {
                        case GGUF_TYPE_UINT8: md.array.data = read_array<uint8_t>(in, md.array.size); break;
                        case GGUF_TYPE_INT8: md.array.data = read_array<int8_t>(in, md.array.size); break;
                        // ... otros tipos numéricos
                        default:
                            std::cerr << "Unsupported array type: " << md.array.type << "\n";
                            continue;
                    }
                    in.ignore(1); // Saltar \n después de los datos
                }
                break;
            default:
                std::cerr << "Unknown metadata type: " << md.type << "\n";
                continue;
        }
        
        // Verificar delimitador final
        std::getline(in, marker); // Saltar línea actual (valor)
        std::getline(in, marker); // Leer delimitador
        
        if (marker != "---END_ITEM---") {
            std::cerr << "Invalid item delimiter\n";
            return GGUFMetadata();
        }

        if (md.key == key) {
            in.close();
            return md;
        }
    }

    in.close();
    return GGUFMetadata(); // No encontrado
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

    // Verificar cabecera mágica
    char magic[6];
    in.read(magic, 6);
    if (std::string(magic, 6) != "GRAPH") {
        std::cerr << "Invalid file format\n";
        return GGUFTensor();
    }

    // Leer encabezado GGUF
    GGUFHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(GGUFHeader));
    
    // Saltar sección de metadatos
    std::string marker;
    std::getline(in, marker); // Leer hasta el fin de línea después del header
    std::getline(in, marker); // Leer el marcador de metadatos
    
    if (marker != "---METADATA---") {
        std::cerr << "Invalid metadata marker\n";
        return GGUFTensor();
    }

    // Saltar todos los metadatos
    skip_metadata(in);

    // Buscar inicio de tensores
    std::getline(in, marker); // Leer hasta el siguiente marcador
    if (marker != "---TENSORS---") {
        std::cerr << "Invalid tensors marker\n";
        return GGUFTensor();
    }

    // Leer cantidad de tensores
    uint64_t tensors_count;
    in >> tensors_count;
    in.ignore(1); // Saltar el \n

    // Buscar el tensor solicitado
    for (uint64_t i = 0; i < tensors_count; ++i) {
        GGUFTensor tensor;
        
        // Verificar delimitador de inicio
        std::getline(in, marker);
        if (marker != "---BEGIN_TENSOR---") {
            std::cerr << "Invalid tensor start marker\n";
            return GGUFTensor();
        }

        // Leer información básica
        std::string line;
        while (std::getline(in, line)) {
            if (line == "DATA_START:") break; // Fin de la sección de información
            
            if (line.find("NAME:") == 0) {
                tensor.name = line.substr(5);
            }
            else if (line.find("TYPE:") == 0) {
                tensor.type = static_cast<enum ggml_type>(std::stoi(line.substr(5)));
            }
            else if (line.find("SIZE:") == 0) {
                tensor.size = std::stoull(line.substr(5));
            }
            else if (line.find("NDIMS:") == 0) {
                tensor.n_dims = std::stoi(line.substr(6));
            }
            else if (line.find("DIMS:") == 0) {
                std::istringstream dims_stream(line.substr(5));
                int64_t dim;
                while (dims_stream >> dim) {
                    tensor.dims.push_back(dim);
                }
            }
            else if (line.find("OP:") == 0) {
                tensor.op = static_cast<enum ggml_op>(std::stoi(line.substr(3)));
            }
        }

        // Verificar que encontramos el tensor buscado
        if (tensor.name != name) {
            // Saltar los datos del tensor y continuar buscando
            while (std::getline(in, line) && line != "---END_TENSOR---") {}
            continue;
        }

        // Leer datos del tensor (siempre float32 según el formato)
        if (line != "DATA_START:") {
            std::cerr << "Invalid data start marker\n";
            return GGUFTensor();
        }

        // Leer datos binarios
        size_t element_count = tensor.size / sizeof(float);
        tensor.data = read_array<float>(in, element_count);

        // Verificar delimitador de fin de datos
        std::getline(in, line); // Saltar restos de datos binarios
        std::getline(in, line);
        if (line != "DATA_END") {
            std::cerr << "Invalid data end marker\n";
            return GGUFTensor();
        }

        // Verificar delimitador final de tensor
        std::getline(in, line);
        if (line != "---END_TENSOR---") {
            std::cerr << "Invalid tensor end marker\n";
            return GGUFTensor();
        }

        in.close();
        return tensor;
    }

    in.close();
    return GGUFTensor(); // No encontrado
}

// Helper function to skip metadata when searching for tensors
void skip_metadata(std::ifstream& in) {
    // Saltar clave
    std::string line;
    std::getline(in, line);
    
    // Saltar tipo y valor
    enum gguf_type type;
    in.read(reinterpret_cast<char*>(&type), sizeof(enum gguf_type));
    in.ignore(1); // Saltar \n

    switch (type) {
        case GGUF_TYPE_UINT8: in.ignore(sizeof(uint8_t)); break;
        case GGUF_TYPE_INT8: in.ignore(sizeof(int8_t)); break;
        // ... otros tipos básicos
        case GGUF_TYPE_STRING: {
            std::getline(in, line); // Saltar el string
            break;
        }
        case GGUF_TYPE_ARRAY: {
            enum gguf_type arr_type;
            size_t arr_size;
            in.read(reinterpret_cast<char*>(&arr_type), sizeof(enum gguf_type));
            in.read(reinterpret_cast<char*>(&arr_size), sizeof(size_t));
            in.ignore(1); // Saltar \n
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
template<typename T>
std::vector<T> read_array(std::ifstream& in, size_t size) {
    std::vector<T> data(size);
    in.read(reinterpret_cast<char*>(data.data()), size * sizeof(T));
    return data;
}


////////////////////////////////////////////////////////////////////////////////

void print_graph_data_from_struct(const GraphData& graph_data, const char *output_filename) {
    // Abrir el archivo de salida
    std::ofstream outfile(output_filename);
    if (!outfile) {
        std::cerr << "No se pudo abrir el archivo de salida: " << output_filename << "\n";
        return;
    }

    // Función helper para nombres de tipos GGUF
    auto gguf_type_name = [](enum gguf_type type) -> const char* {
        static const char* names[] = {
            "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32",
            "FLOAT32", "BOOL", "STRING", "UINT64", "INT64", "FLOAT64", "ARRAY"
        };
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
    
    for (const auto &md : graph_data.metadata) {
        outfile << "Clave: " << md.key << "\n";
        outfile << "Tipo: " << gguf_type_name(md.type) << "\n";
        
        switch (md.type) {
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
                if (md.array.type == GGUF_TYPE_STRING) {
                    if (const auto* strs = std::get_if<std::vector<std::string>>(&md.array.data)) {
                        outfile << "Primeros strings: [";
                        for (size_t i = 0; i < std::min(strs->size(), max_elements); ++i) {
                            outfile << "\"" << (*strs)[i] << "\" ";
                        }
                        outfile << "...]\n";
                    }
                }
                else if (md.array.type == GGUF_TYPE_FLOAT32) {
                    if (const auto* vals = std::get_if<std::vector<float>>(&md.array.data)) {
                        outfile << std::fixed << std::setprecision(6);
                        outfile << "Primeros valores: [";
                        for (size_t i = 0; i < std::min(vals->size(), max_elements); ++i) {
                            outfile << (*vals)[i] << " ";
                        }
                        outfile << "...]\n";
                        outfile.unsetf(std::ios::fixed);
                    }
                }
                else if (md.array.type == GGUF_TYPE_INT32) {
                    if (const auto* vals = std::get_if<std::vector<int32_t>>(&md.array.data)) {
                        outfile << "Primeros valores: [";
                        for (size_t i = 0; i < std::min(vals->size(), max_elements); ++i) {
                            outfile << (*vals)[i] << " ";
                        }
                        outfile << "...]\n";
                    }
                }
                else if (md.array.type == GGUF_TYPE_UINT8) {
                    if (const auto* vals = std::get_if<std::vector<uint8_t>>(&md.array.data)) {
                        outfile << "Primeros valores: [";
                        for (size_t i = 0; i < std::min(vals->size(), max_elements); ++i) {
                            outfile << static_cast<int>((*vals)[i]) << " ";
                        }
                        outfile << "...]\n";
                    }
                }
                else {
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
    
    for (const auto &tensor : graph_data.tensors) {
        outfile << "Nombre: " << tensor.name << "\n";
        outfile << "Tipo: " << ggml_type_name(tensor.type) << "\n";
        outfile << "Tamaño: " << std::fixed << std::setprecision(2) 
               << tensor.size / 1024.0f / 1024.0f << " MB\n";
        outfile << "Número de dimensiones: " << tensor.n_dims << "\n";
        outfile << "Tamaño de cada dimensión: ";
        for (const auto &dim : tensor.dims) {
            outfile << dim << " ";
        }

        outfile << "\n";

        // Mostrar los primeros elementos del tensor según su tipo
        outfile << "Datos (primeros elementos): ";
        
        
        
        switch (tensor.type) {
            case GGML_TYPE_F32:
                if (const auto* data = std::get_if<std::vector<float>>(&tensor.data)) {
                    for (size_t i = 0; i < std::min(data->size(), max_elements); ++i) {
                        outfile << (*data)[i] << " ";
                    }
                }
                break;
                
            case GGML_TYPE_I32:
                if (const auto* data = std::get_if<std::vector<int32_t>>(&tensor.data)) {
                    for (size_t i = 0; i < std::min(data->size(), max_elements); ++i) {
                        outfile << (*data)[i] << " ";
                    }
                }
                break;
                
            case GGML_TYPE_F16:
                if (const auto* data = std::get_if<std::vector<uint16_t>>(&tensor.data)) {
                    for (size_t i = 0; i < std::min(data->size(), max_elements); ++i) {
                        outfile << (*data)[i] << " ";
                    }
                }
                break;
                
            case GGML_TYPE_I8:
                if (const auto* data = std::get_if<std::vector<int8_t>>(&tensor.data)) {
                    for (size_t i = 0; i < std::min(data->size(), max_elements); ++i) {
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
                if (const auto* data = std::get_if<std::vector<uint8_t>>(&tensor.data)) {
                    //outfile << "[Datos cuantizados - " << data->size() << " bytes]";
                    outfile << "[Datos cuantizados: se toman en grupos de 8 bits en este caso]  ";
                    for (size_t i = 0; i < std::min(data->size(), max_elements); ++i) {
                        outfile << static_cast<uint8_t>((*data)[i]) << " "; 
                    }
                }
                break;
                
            default:
                if (const auto* data = std::get_if<std::vector<uint8_t>>(&tensor.data)) {
                    outfile << "[Datos binarios - " << data->size() << " bytes]";
                }
                break;
        }

        // Indicar si hay más elementos
        if (std::visit([](const auto& v) { return v.size(); }, tensor.data) > max_elements) {
            outfile << "... [total: " 
                << std::visit([](const auto& v) { return v.size(); }, tensor.data) 
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
size_t gguf_type_size(enum gguf_type type) {
    switch (type) {
        case GGUF_TYPE_UINT8:  return sizeof(uint8_t);
        case GGUF_TYPE_INT8:   return sizeof(int8_t);
        case GGUF_TYPE_UINT16: return sizeof(uint16_t);
        case GGUF_TYPE_INT16:  return sizeof(int16_t);
        case GGUF_TYPE_UINT32: return sizeof(uint32_t);
        case GGUF_TYPE_INT32:  return sizeof(int32_t);
        case GGUF_TYPE_FLOAT32:return sizeof(float);
        case GGUF_TYPE_UINT64: return sizeof(uint64_t);
        case GGUF_TYPE_INT64:  return sizeof(int64_t);
        case GGUF_TYPE_FLOAT64:return sizeof(double);
        case GGUF_TYPE_BOOL:   return sizeof(bool);
        default: return 0; // Para tipos sin tamaño fijo (STRING, ARRAY)
    }
}

void print_graph_data_from_file(const std::string& input_filename, const std::string& output_filename) {
    const size_t max_elements = 20; // Mostrar solo los primeros 20 elementos

    // Abrir archivos con verificación
    std::ifstream infile(input_filename, std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Error al abrir archivo de entrada: " << input_filename << std::endl;
        return;
    }

    std::ofstream outfile(output_filename);
    if (!outfile) {
        std::cerr << "Error al abrir archivo de salida: " << output_filename << std::endl;
        infile.close();
        return;
    }

    // Función helper para nombres de tipos GGUF
    auto gguf_type_name = [](enum gguf_type type) -> const char* {
        static const char* names[] = {
            "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32",
            "FLOAT32", "BOOL", "STRING", "UINT64", "INT64", "FLOAT64", "ARRAY"
        };
        return (type >= GGUF_TYPE_UINT8 && type <= GGUF_TYPE_ARRAY) ? names[type] : "UNKNOWN";
    };

    // Función helper para nombres de tipos GGML
    auto ggml_type_name = [](enum ggml_type type) -> const char* {
        static const char* names[] = {
            "F32", "F16", "I32", "I16", "I8", 
            "Q4_0", "Q4_1", "Q8_0", "Q2_K", "Q3_K"
        };
        return (type >= GGML_TYPE_F32 && type <= GGML_TYPE_Q3_K) ? names[type] : "UNKNOWN";
    };

    // Verificar cabecera mágica
    char magic[5];
    infile.read(magic, 5);
    if (std::string(magic, 5) != "GRAPH") {
        std::cerr << "Invalid file format\n";
        return;
    }

    // Leer encabezado GGUF
    GGUFHeader header;
    infile.read(reinterpret_cast<char*>(&header), sizeof(GGUFHeader));
    
    // Escribir el encabezado
    outfile << "\n**************************************************************\n";
    outfile << "**************************  HEADER  **************************\n";
    outfile << "**************************************************************\n";
    outfile << "Número de tensores: " << header.n_tensors << "\n";
    outfile << "Número de pares clave-valor: " << header.n_kv << "\n";

    // Saltar hasta la sección de metadatos
    std::string marker;
    std::getline(infile, marker); // Leer hasta el fin de línea después del header
    std::getline(infile, marker); // Leer el marcador de metadatos
    
    if (marker != "---METADATA---") {
        std::cerr << "Invalid metadata marker\n";
        return;
    }

    // Leer y escribir metadatos
    outfile << "\n**************************************************************\n";
    outfile << "*************************  METADATA  *************************\n";
    outfile << "**************************************************************\n";
    
    uint64_t metadata_count;
    infile.read(reinterpret_cast<char*>(&metadata_count), sizeof(uint64_t));
    infile.ignore(1); // Saltar el \n

    for (uint64_t i = 0; i < metadata_count; ++i) {
        GGUFMetadata md;
        
        // Leer clave
        std::getline(infile, md.key);
        
        // Leer tipo
        infile.read(reinterpret_cast<char*>(&md.type), sizeof(enum gguf_type));
        infile.ignore(1); // Saltar \n

        outfile << "\nClave: " << md.key << "\n";
        outfile << "Tipo: " << gguf_type_name(md.type) << "\n";
        outfile << "Valor: ";

        // Leer valor según tipo
        switch (md.type) {
            case GGUF_TYPE_UINT8: 
                infile >> md.value.u8;
                outfile << static_cast<int>(md.value.u8);
                break;
            case GGUF_TYPE_INT8:
                infile >> md.value.i8;
                outfile << static_cast<int>(md.value.i8);
                break;
            case GGUF_TYPE_UINT16:
                infile >> md.value.u16;
                outfile << md.value.u16;
                break;
            case GGUF_TYPE_INT16:
                infile >> md.value.i16;
                outfile << md.value.i16;
                break;
            case GGUF_TYPE_UINT32:
                infile >> md.value.u32;
                outfile << md.value.u32;
                break;
            case GGUF_TYPE_INT32:
                infile >> md.value.i32;
                outfile << md.value.i32;
                break;
            case GGUF_TYPE_FLOAT32:
                infile >> md.value.f32;
                outfile << std::fixed << std::setprecision(6) << md.value.f32;
                break;
            case GGUF_TYPE_UINT64:
                infile >> md.value.u64;
                outfile << md.value.u64;
                break;
            case GGUF_TYPE_INT64:
                infile >> md.value.i64;
                outfile << md.value.i64;
                break;
            case GGUF_TYPE_FLOAT64:
                infile >> md.value.f64;
                outfile << std::fixed << std::setprecision(6) << md.value.f64;
                break;
            case GGUF_TYPE_BOOL: {
                std::string val;
                infile >> val;
                md.value.b = (val == "true");
                outfile << (md.value.b ? "true" : "false");
                break;
            }
            case GGUF_TYPE_STRING:
                std::getline(infile, md.str);
                outfile << md.str;
                break;
            case GGUF_TYPE_ARRAY:
                infile.read(reinterpret_cast<char*>(&md.array.type), sizeof(enum gguf_type));
                infile.read(reinterpret_cast<char*>(&md.array.size), sizeof(size_t));
                infile.ignore(1); // Saltar \n
                
                outfile << "Array[" << md.array.size << "] of " << gguf_type_name(md.array.type) << "\n";
                
                if (md.array.type == GGUF_TYPE_STRING) {
                    outfile << "  [";
                    for (size_t j = 0; j < std::min(md.array.size, max_elements); ++j) {
                        std::string str;
                        std::getline(infile, str);
                        if (j > 0) outfile << ", ";
                        outfile << "\"" << str << "\"";
                    }
                    if (md.array.size > max_elements) outfile << ", ...";
                    outfile << "]";
                } else {
                    outfile << "  [";
                    const size_t element_size = gguf_type_size(md.array.type);
                    
                    if (element_size > 0) {
                        for (size_t j = 0; j < std::min(md.array.size, max_elements); ++j) {
                            if (j > 0) outfile << ", ";
                            
                            switch (md.array.type) {
                                case GGUF_TYPE_UINT8: { 
                                    uint8_t val; 
                                    infile.read(reinterpret_cast<char*>(&val), sizeof(uint8_t)); 
                                    outfile << static_cast<int>(val); 
                                    break; 
                                }
                                case GGUF_TYPE_INT8: { 
                                    int8_t val; 
                                    infile.read(reinterpret_cast<char*>(&val), sizeof(int8_t)); 
                                    outfile << static_cast<int>(val); 
                                    break; 
                                }
                                case GGUF_TYPE_UINT16: { 
                                    uint16_t val; 
                                    infile.read(reinterpret_cast<char*>(&val), sizeof(uint16_t)); 
                                    outfile << val; 
                                    break; 
                                }
                                case GGUF_TYPE_INT16: { 
                                    int16_t val; 
                                    infile.read(reinterpret_cast<char*>(&val), sizeof(int16_t)); 
                                    outfile << val; 
                                    break; 
                                }
                                case GGUF_TYPE_UINT32: { 
                                    uint32_t val; 
                                    infile.read(reinterpret_cast<char*>(&val), sizeof(uint32_t)); 
                                    outfile << val; 
                                    break; 
                                }
                                case GGUF_TYPE_INT32: { 
                                    int32_t val; 
                                    infile.read(reinterpret_cast<char*>(&val), sizeof(int32_t)); 
                                    outfile << val; 
                                    break; 
                                }
                                case GGUF_TYPE_FLOAT32: { 
                                    float val; 
                                    infile.read(reinterpret_cast<char*>(&val), sizeof(float)); 
                                    outfile << std::fixed << std::setprecision(6) << val; 
                                    break; 
                                }
                                case GGUF_TYPE_UINT64: { 
                                    uint64_t val; 
                                    infile.read(reinterpret_cast<char*>(&val), sizeof(uint64_t)); 
                                    outfile << val; 
                                    break; 
                                }
                                case GGUF_TYPE_INT64: { 
                                    int64_t val; 
                                    infile.read(reinterpret_cast<char*>(&val), sizeof(int64_t)); 
                                    outfile << val; 
                                    break; 
                                }
                                case GGUF_TYPE_FLOAT64: { 
                                    double val; 
                                    infile.read(reinterpret_cast<char*>(&val), sizeof(double)); 
                                    outfile << std::fixed << std::setprecision(6) << val; 
                                    break; 
                                }
                                case GGUF_TYPE_BOOL: { 
                                    bool val; 
                                    infile.read(reinterpret_cast<char*>(&val), sizeof(bool)); 
                                    outfile << (val ? "true" : "false"); 
                                    break; 
                                }
                                default:
                                    infile.ignore(md.array.size * element_size);
                                    outfile << "<binary data>";
                                    j = md.array.size; // Salir del bucle
                            }
                        }
                    } else {
                        infile.ignore(md.array.size);
                        outfile << "<unsupported array type>";
                    }
                    
                    if (md.array.size > max_elements) outfile << ", ...";
                    outfile << "]";
                }
                break;
            default:
                std::cerr << "Unknown metadata type: " << md.type << "\n";
                continue;
        }
        
        infile.ignore(1);

        // Leer el marcador de fin de item
        std::getline(infile, marker);
        

        // Agregar manualmente el \n para la comparación
        if (marker != "---END_ITEM---") {
            std::cerr << "Invalid item delimiter (end item). Found: '" << marker << "'\n";
            return;
        }

        outfile << "\n-----------------------------------\n";
    }

    // Buscar inicio de tensores
    std::getline(infile, marker); // Leer hasta el siguiente marcador
    if (marker != "---TENSORS---") {
        std::cerr << "Invalid tensors marker\n";
        return;
    }

    // Leer y escribir tensores
    outfile << "\n**************************************************************\n";
    outfile << "*************************  TENSORS  *************************\n";
    outfile << "**************************************************************\n";
    
    uint64_t tensors_count;
    infile >> tensors_count;
    infile.ignore(1); // Saltar el \n

    for (uint64_t i = 0; i < tensors_count; ++i) {
        // Verificar delimitador de inicio
        std::getline(infile, marker);
        if (marker != "---BEGIN_TENSOR---") {
            std::cerr << "Invalid tensor start marker\n";
            return;
        }

        GGUFTensor tensor;
        std::string line;
        
        // Leer información básica
        while (std::getline(infile, line)) {
            if (line == "DATA_START:") break; // Fin de la sección de información
            
            if (line.find("NAME:") == 0) {
                tensor.name = line.substr(5);
                outfile << "\nTensor: " << tensor.name << "\n";
            }
            else if (line.find("TYPE:") == 0) {
                tensor.type = static_cast<enum ggml_type>(std::stoi(line.substr(5)));
                outfile << "Tipo: " << ggml_type_name(tensor.type) << "\n";
            }
            else if (line.find("SIZE:") == 0) {
                tensor.size = std::stoull(line.substr(5));
                outfile << "Tamaño: " << tensor.size << " bytes\n";
            }
            else if (line.find("NDIMS:") == 0) {
                tensor.n_dims = std::stoi(line.substr(6));
                outfile << "Dimensiones: " << tensor.n_dims << "\n";
            }
            else if (line.find("DIMS:") == 0) {
                std::istringstream dims_stream(line.substr(5));
                int64_t dim;
                outfile << "Tamaños: [";
                while (dims_stream >> dim) {
                    tensor.dims.push_back(dim);
                    if (tensor.dims.size() > 1) outfile << " × ";
                    outfile << dim;
                }
                outfile << "]\n";
            }
            else if (line.find("OP:") == 0) {
                tensor.op = static_cast<enum ggml_op>(std::stoi(line.substr(3)));
                outfile << "Operación: " << tensor.op << "\n";
            }
        }

        // Leer datos del tensor (solo información básica, no los datos binarios)
        if (line != "DATA_START:") {
            std::cerr << "Invalid data start marker\n";
            return;
        }

        // Saltar datos binarios
        size_t element_count = tensor.size / sizeof(float);
        infile.ignore(element_count * sizeof(float));
        
        // Verificar delimitador de fin de datos
        std::getline(infile, line); // Saltar restos de datos binarios
        std::getline(infile, line);
        if (line != "DATA_END") {
            std::cerr << "Invalid data end marker\n";
            return;
        }

        // Verificar delimitador final de tensor
        std::getline(infile, line);
        if (line != "---END_TENSOR---") {
            std::cerr << "Invalid tensor end marker\n";
            return;
        }

        outfile << "-----------------------------------\n";
    }

    // Cerrar archivos
    infile.close();
    outfile.close();
    
    std::cout << "La información del archivo " << input_filename 
              << " se ha escrito en: " << output_filename << "\n";
}
