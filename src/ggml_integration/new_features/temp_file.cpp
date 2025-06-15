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
        
        
        std::getline(infile, marker);
        
        infile.ignore(1);
        // Leer el marcador de fin de item
        std::getline(infile, marker);
        

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

