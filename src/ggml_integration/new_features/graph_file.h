#ifndef GRAPH_FILE_H
#define GRAPH_FILE_H

#include <string>
#include <vector>
#include "graph_data_structs.h"

/**
 * @brief Escribe datos de grafo en un archivo binario .graph
 * @param filename Nombre del archivo de salida (ej. "model.graph")
 * @param graph_data Estructura GraphData a serializar
 * @return true si tiene éxito, false en caso de error
 */
bool write_graph_data(const std::string& filename, const GraphData& graph_data);

/**
 * @brief Lee una entrada de metadata por clave desde un archivo .graph
 * @param filename Nombre del archivo .graph de entrada
 * @param key Clave de metadata a buscar
 * @return Estructura GGUFMetadata si se encuentra, o metadata vacía con type=GGUF_TYPE_COUNT si no se encuentra
 */
GGUFMetadata read_metadata(const std::string& filename, const std::string& key);

/**
 * @brief Lee un tensor por nombre desde un archivo .graph
 * @param filename Nombre del archivo .graph de entrada
 * @param name Nombre del tensor a buscar
 * @return Estructura GGUFTensor si se encuentra, o tensor vacío con name="" si no se encuentra
 */
GGUFTensor read_tensor(const std::string& filename, const std::string& name);

// Funciones auxiliares 
void skip_metadata(std::ifstream& in);
size_t type_size(enum gguf_type type);
size_t ggml_type_size(enum ggml_type type);
    
template<typename T>
std::vector<T> read_array(std::ifstream& in, size_t size);

#endif // GRAPH_FILE_H
