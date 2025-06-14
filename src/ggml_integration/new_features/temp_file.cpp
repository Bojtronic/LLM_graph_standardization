

void print_graph_data_from_file(const std::string& input_filename, const std::string& output_filename) {
    // Función helper para nombres de tipos GGUF
    auto gguf_type_name = [](enum gguf_type type) -> const char* {
        static const char* names[] = {
            "UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32",
            "FLOAT32", "BOOL", "STRING", "UINT64", "INT64", "FLOAT64", "ARRAY"
        };
        return (type >= GGUF_TYPE_UINT8 && type <= GGUF_TYPE_ARRAY) ? names[type] : "UNKNOWN";
    };

    // Función helper para nombres de tipos GGML (tensores)
    auto ggml_type_name = [](enum ggml_type type) -> const char* {
        static const char* names[] = {
            "F32", "F16", "Q4_0", "Q4_1", "Q8_0", "Q2_K", "Q3_K", "I8", "I16", "I32"
        };
        return (type >= GGML_TYPE_F32 && type <= GGML_TYPE_I32) ? names[type] : "UNKNOWN";
    };

    const size_t max_elements = 20; // Mostrar solo los primeros 20 elementos

    // Abrir archivo de salida
    std::ofstream outfile(output_filename);
    if (!outfile) {
        std::cerr << "No se pudo abrir el archivo de salida: " << output_filename << "\n";
        return;
    }

    // Abrir archivo de entrada
    std::ifstream infile(input_filename, std::ios::binary);
    if (!infile) {
        std::cerr << "No se pudo abrir el archivo de entrada: " << input_filename << "\n";
        outfile.close();
        return;
    }

    // Leer encabezado
    GGUFHeader header;
    infile.read(reinterpret_cast<char*>(&header), sizeof(GGUFHeader));

    // Escribir el encabezado
    outfile << "\n**************************************************************\n";
    outfile << "**************************  HEADER  **************************\n";
    outfile << "**************************************************************\n";
    outfile << "Número de tensores: " << header.n_tensors << "\n";
    outfile << "Número de pares clave-valor: " << header.n_kv << "\n";

    // Leer y escribir metadatos
    outfile << "\n**************************************************************\n";
    outfile << "*************************  METADATA  *************************\n";
    outfile << "**************************************************************\n";
    
    uint64_t metadata_count;
    infile.read(reinterpret_cast<char*>(&metadata_count), sizeof(uint64_t));

    for (uint64_t i = 0; i < metadata_count; ++i) {
        GGUFMetadata md;

        // Leer clave
        uint64_t key_size;
        infile.read(reinterpret_cast<char*>(&key_size), sizeof(uint64_t));
        md.key.resize(key_size);
        infile.read(&md.key[0], key_size);

        // Leer tipo
        infile.read(reinterpret_cast<char*>(&md.type), sizeof(enum gguf_type));

        // Escribir metadato
        outfile << "Clave: " << md.key << "\n";
        outfile << "Tipo: " << gguf_type_name(md.type) << "\n";
        
        // Leer y escribir valor según tipo
        switch (md.type) {
            case GGUF_TYPE_UINT8:
                infile.read(reinterpret_cast<char*>(&md.value.u8), sizeof(uint8_t));
                outfile << "Valor: " << static_cast<int>(md.value.u8) << " (uint8)\n";
                break;
            case GGUF_TYPE_INT8:
                infile.read(reinterpret_cast<char*>(&md.value.i8), sizeof(int8_t));
                outfile << "Valor: " << static_cast<int>(md.value.i8) << " (int8)\n";
                break;
            case GGUF_TYPE_UINT16:
                infile.read(reinterpret_cast<char*>(&md.value.u16), sizeof(uint16_t));
                outfile << "Valor: " << md.value.u16 << " (uint16)\n";
                break;
            case GGUF_TYPE_INT16:
                infile.read(reinterpret_cast<char*>(&md.value.i16), sizeof(int16_t));
                outfile << "Valor: " << md.value.i16 << " (int16)\n";
                break;
            case GGUF_TYPE_UINT32:
                infile.read(reinterpret_cast<char*>(&md.value.u32), sizeof(uint32_t));
                outfile << "Valor: " << md.value.u32 << " (uint32)\n";
                break;
            case GGUF_TYPE_INT32:
                infile.read(reinterpret_cast<char*>(&md.value.i32), sizeof(int32_t));
                outfile << "Valor: " << md.value.i32 << " (int32)\n";
                break;
            case GGUF_TYPE_FLOAT32:
                infile.read(reinterpret_cast<char*>(&md.value.f32), sizeof(float));
                outfile << std::fixed << std::setprecision(6);
                outfile << "Valor: " << md.value.f32 << " (float32)\n";
                outfile.unsetf(std::ios::fixed);
                outfile.precision(6);
                break;
            case GGUF_TYPE_BOOL:
                infile.read(reinterpret_cast<char*>(&md.value.b), sizeof(bool));
                outfile << "Valor: " << (md.value.b ? "true" : "false") << " (bool)\n";
                break;
            case GGUF_TYPE_STRING: {
                uint64_t str_size;
                infile.read(reinterpret_cast<char*>(&str_size), sizeof(uint64_t));
                md.str.resize(str_size);
                infile.read(&md.str[0], str_size);
                outfile << "Valor: " << md.str << " (string)\n";
                break;
            }
            case GGUF_TYPE_UINT64:
                infile.read(reinterpret_cast<char*>(&md.value.u64), sizeof(uint64_t));
                outfile << "Valor: " << md.value.u64 << " (uint64)\n";
                break;
            case GGUF_TYPE_INT64:
                infile.read(reinterpret_cast<char*>(&md.value.i64), sizeof(int64_t));
                outfile << "Valor: " << md.value.i64 << " (int64)\n";
                break;
            case GGUF_TYPE_FLOAT64:
                infile.read(reinterpret_cast<char*>(&md.value.f64), sizeof(double));
                outfile << std::fixed << std::setprecision(6);
                outfile << "Valor: " << md.value.f64 << " (float64)\n";
                outfile.unsetf(std::ios::fixed);
                outfile.precision(6);
                break;
            case GGUF_TYPE_ARRAY: {
                infile.read(reinterpret_cast<char*>(&md.array.type), sizeof(enum gguf_type));
                infile.read(reinterpret_cast<char*>(&md.array.size), sizeof(size_t));
                
                outfile << "Tipo de array: " << gguf_type_name(md.array.type) << "\n";
                outfile << "Tamaño del array: " << md.array.size << "\n";
                
                // Manejar arrays de strings
                if (md.array.type == GGUF_TYPE_STRING) {
                    std::vector<std::string> strings(md.array.size);
                    outfile << "Primeros strings: [";
                    for (size_t j = 0; j < std::min(md.array.size, max_elements); ++j) {
                        uint64_t str_size;
                        infile.read(reinterpret_cast<char*>(&str_size), sizeof(uint64_t));
                        strings[j].resize(str_size);
                        infile.read(&strings[j][0], str_size);
                        outfile << "\"" << strings[j] << "\" ";
                    }
                    outfile << "...]\n";
                } 
                // Manejar arrays numéricos
                else {
                    outfile << "Primeros valores: [";
                    size_t element_size = type_size(md.array.type);
                    for (size_t j = 0; j < std::min(md.array.size, max_elements); ++j) {
                        switch (md.array.type) {
                            case GGUF_TYPE_UINT8: {
                                uint8_t val;
                                infile.read(reinterpret_cast<char*>(&val), sizeof(uint8_t));
                                outfile << static_cast<int>(val) << " ";
                                break;
                            }
                            case GGUF_TYPE_INT8: {
                                int8_t val;
                                infile.read(reinterpret_cast<char*>(&val), sizeof(int8_t));
                                outfile << static_cast<int>(val) << " ";
                                break;
                            }
                            case GGUF_TYPE_UINT16: {
                                uint16_t val;
                                infile.read(reinterpret_cast<char*>(&val), sizeof(uint16_t));
                                outfile << val << " ";
                                break;
                            }
                            case GGUF_TYPE_INT16: {
                                int16_t val;
                                infile.read(reinterpret_cast<char*>(&val), sizeof(int16_t));
                                outfile << val << " ";
                                break;
                            }
                            case GGUF_TYPE_UINT32: {
                                uint32_t val;
                                infile.read(reinterpret_cast<char*>(&val), sizeof(uint32_t));
                                outfile << val << " ";
                                break;
                            }
                            case GGUF_TYPE_INT32: {
                                int32_t val;
                                infile.read(reinterpret_cast<char*>(&val), sizeof(int32_t));
                                outfile << val << " ";
                                break;
                            }
                            case GGUF_TYPE_FLOAT32: {
                                float val;
                                infile.read(reinterpret_cast<char*>(&val), sizeof(float));
                                outfile << std::fixed << std::setprecision(6) << val << " ";
                                outfile.unsetf(std::ios::fixed);
                                break;
                            }
                            case GGUF_TYPE_FLOAT64: {
                                double val;
                                infile.read(reinterpret_cast<char*>(&val), sizeof(double));
                                outfile << std::fixed << std::setprecision(6) << val << " ";
                                outfile.unsetf(std::ios::fixed);
                                break;
                            }
                            default:
                                // Para tipos no soportados, saltar los bytes
                                infile.seekg(element_size, std::ios::cur);
                                outfile << "[binary data] ";
                                break;
                        }
                    }
                    // Saltar los elementos restantes del array
                    if (md.array.size > max_elements) {
                        infile.seekg((md.array.size - max_elements) * element_size, std::ios::cur);
                    }
                    outfile << "...]\n";
                }
                break;
            }
            default:
                outfile << "Valor: [tipo desconocido]\n";
                break;
        }
        outfile << "-----------------------------------\n";
    }

    // Leer y escribir tensores
    outfile << "\n**************************************************************\n";
    outfile << "*************************  TENSORS  **************************\n";
    outfile << "**************************************************************\n";
    
    uint64_t tensors_count;
    infile.read(reinterpret_cast<char*>(&tensors_count), sizeof(uint64_t));

    for (uint64_t i = 0; i < tensors_count; ++i) {
        GGUFTensor tensor;

        // Leer nombre
        uint64_t name_size;
        infile.read(reinterpret_cast<char*>(&name_size), sizeof(uint64_t));
        tensor.name.resize(name_size);
        infile.read(&tensor.name[0], name_size);

        // Leer información del tensor
        infile.read(reinterpret_cast<char*>(&tensor.type), sizeof(enum ggml_type));
        infile.read(reinterpret_cast<char*>(&tensor.size), sizeof(size_t));
        infile.read(reinterpret_cast<char*>(&tensor.n_dims), sizeof(int32_t));

        // Leer dimensiones
        tensor.dims.resize(tensor.n_dims);
        infile.read(reinterpret_cast<char*>(tensor.dims.data()), tensor.n_dims * sizeof(int64_t));

        // Leer operación
        infile.read(reinterpret_cast<char*>(&tensor.op), sizeof(enum ggml_op));

        // Leer tensores fuente
        uint64_t src_count;
        infile.read(reinterpret_cast<char*>(&src_count), sizeof(uint64_t));
        tensor.src_tensors.resize(src_count);
        for (uint64_t j = 0; j < src_count; ++j) {
            uint64_t src_size;
            infile.read(reinterpret_cast<char*>(&src_size), sizeof(uint64_t));
            tensor.src_tensors[j].resize(src_size);
            infile.read(&tensor.src_tensors[j][0], src_size);
        }

        // Leer tensor destino
        uint64_t dst_size;
        infile.read(reinterpret_cast<char*>(&dst_size), sizeof(uint64_t));
        tensor.dst_tensor.resize(dst_size);
        infile.read(&tensor.dst_tensor[0], dst_size);

        // Escribir información del tensor
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

        // Leer y mostrar primeros elementos del tensor
        outfile << "Datos (primeros elementos): ";
        
        size_t element_count = tensor.size / type_size(static_cast<gguf_type>(tensor.type));
        if (element_count == 0) element_count = tensor.size; // fallback

        for (size_t j = 0; j < std::min(element_count, max_elements); ++j) {
            switch (tensor.type) {
                case GGML_TYPE_F32: {
                    float val;
                    infile.read(reinterpret_cast<char*>(&val), sizeof(float));
                    outfile << val << " ";
                    break;
                }
                case GGML_TYPE_I32: {
                    int32_t val;
                    infile.read(reinterpret_cast<char*>(&val), sizeof(int32_t));
                    outfile << val << " ";
                    break;
                }
                case GGML_TYPE_F16: {
                    uint16_t val;
                    infile.read(reinterpret_cast<char*>(&val), sizeof(uint16_t));
                    outfile << val << " ";
                    break;
                }
                case GGML_TYPE_I8: {
                    int8_t val;
                    infile.read(reinterpret_cast<char*>(&val), sizeof(int8_t));
                    outfile << static_cast<int>(val) << " ";
                    break;
                }
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K: {
                    uint8_t val;
                    infile.read(reinterpret_cast<char*>(&val), sizeof(uint8_t));
                    outfile << static_cast<int>(val) << " ";
                    break;
                }
                default: {
                    uint8_t val;
                    infile.read(reinterpret_cast<char*>(&val), sizeof(uint8_t));
                    outfile << static_cast<int>(val) << " ";
                    break;
                }
            }
        }

        // Saltar los elementos restantes del tensor
        if (element_count > max_elements) {
            size_t elements_to_skip = element_count - max_elements;
            size_t element_size = type_size(static_cast<gguf_type>(tensor.type));
            if (element_size == 0) element_size = 1; // fallback
            infile.seekg(elements_to_skip * element_size, std::ios::cur);
            outfile << "... [total: " << element_count << " elementos]";
        }
        
        outfile << "\n --------------------------------------------------------------- \n";
    }

    // Cerrar archivos
    infile.close();
    outfile.close();
    std::cout << "La información de GraphData se ha escrito en el archivo: " << output_filename << "\n";
}




/////////////////////////////////////////////////////////////////////////////////////








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

