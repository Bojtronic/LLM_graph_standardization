#include <iostream>
#include <fstream>
#include <vector>
#include "graph_data_structs.h"

// Función para escribir strings en binario
void write_string(std::ofstream& file, const std::string& str) {
    size_t length = str.size();
    file.write(reinterpret_cast<const char*>(&length), sizeof(length));
    file.write(str.c_str(), length);
}

// Función para escribir tensores
void write_tensor(std::ofstream& file, const GGUFTensor& tensor) {
    write_string(file, tensor.name);
    file.write(reinterpret_cast<const char*>(&tensor.type), sizeof(tensor.type));
    file.write(reinterpret_cast<const char*>(&tensor.size), sizeof(tensor.size));
    file.write(reinterpret_cast<const char*>(&tensor.n_dims), sizeof(tensor.n_dims));
    
    // Escribir dimensiones
    for (const auto& dim : tensor.dims) {
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    
    // Escribir datos (ejemplo para float32)
    if (auto data = std::get_if<std::vector<float>>(&tensor.data)) {
        file.write(reinterpret_cast<const char*>(data->data()), data->size() * sizeof(float));
    }
    // Aquí añadir otros tipos según sea necesario
}

// Función principal para crear el archivo .graph
void create_graph_file(const std::string& filename) {
    GraphData graph;
    
    // 1. Configurar encabezado
    graph.header.n_tensors = 3;  // Ejemplo con 3 tensores
    graph.header.n_kv = 5;       // Ejemplo con 5 metadatos
    
    // 2. Añadir metadatos de ejemplo
    GGUFMetadata meta;
    meta.key = "model_name";
    meta.type = GGUF_TYPE_STRING;
    meta.str = "mi_modelo_ejemplo";
    graph.metadata.push_back(meta);
    
    meta.key = "n_layer";
    meta.type = GGUF_TYPE_UINT32;
    meta.value.u32 = 12;
    graph.metadata.push_back(meta);
    
    // 3. Añadir tensores de ejemplo
    GGUFTensor tensor;
    tensor.name = "embedding_weight";
    tensor.type = GGML_TYPE_F32;
    tensor.n_dims = 2;
    tensor.dims = {1024, 768};
    
    // Datos de ejemplo (en un caso real aquí irían los valores reales)
    std::vector<float> embedding_data(1024 * 768, 0.0f); // Tensor inicializado a ceros
    tensor.data = embedding_data;
    tensor.size = embedding_data.size() * sizeof(float);
    graph.tensors.push_back(tensor);
    
    // 4. Escribir archivo binario
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error al crear archivo" << std::endl;
        return;
    }
    
    // Escribir encabezado
    file.write(reinterpret_cast<const char*>(&graph.header), sizeof(graph.header));
    
    // Escribir metadatos
    size_t metadata_size = graph.metadata.size();
    file.write(reinterpret_cast<const char*>(&metadata_size), sizeof(metadata_size));
    for (const auto& m : graph.metadata) {
        write_string(file, m.key);
        file.write(reinterpret_cast<const char*>(&m.type), sizeof(m.type));
        if (m.type == GGUF_TYPE_STRING) {
            write_string(file, m.str);
        } else {
            file.write(reinterpret_cast<const char*>(&m.value), sizeof(m.value));
        }
    }
    
    // Escribir tensores
    size_t tensors_size = graph.tensors.size();
    file.write(reinterpret_cast<const char*>(&tensors_size), sizeof(tensors_size));
    for (const auto& t : graph.tensors) {
        write_tensor(file, t);
    }
    
    std::cout << "Archivo " << filename << " creado exitosamente" << std::endl;
}

/*
int main() {
    create_graph_file("modelo.graph");
    return 0;
}
*/


