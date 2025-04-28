#ifndef GRAPH_DATA_STRUCTS_H
#define GRAPH_DATA_STRUCTS_H

#include <cstdint>
#include <string>
#include <vector>
#include <variant>
#include "ggml.h"
#include "gguf.h"

/**
 * @file graph_data_structs.h
 * @brief Definiciones de estructuras para manejar datos GGUF
 */

/**
 * @brief Encabezado de un archivo GGUF
 */
struct GGUFHeader {
    uint64_t n_tensors;     ///< Número de tensores en el archivo
    uint64_t n_kv;          ///< Número de pares clave-valor de metadatos
};

/**
 * @brief Metadatos GGUF (pares clave-valor)
 */
struct GGUFMetadata {
    std::string key;        ///< Nombre/llave del metadato
    enum gguf_type type;    ///< Tipo del valor almacenado
    
    /// Valor usando una union para los diferentes tipos posibles
    union {
        uint8_t u8;
        int8_t i8;
        uint16_t u16;
        int16_t i16;
        uint32_t u32;
        int32_t i32;
        float f32;
        uint64_t u64;
        int64_t i64;
        double f64;
        bool b;
    } value;
    
    /// Para tipos array
    struct {
        enum gguf_type type; ///< Tipo de elementos del array
        size_t size;        ///< Número de elementos
        std::vector<uint8_t> data; ///< Datos del array
    } array;
    
    std::string str;        ///< Para strings
    
    /// Constructor por defecto
    GGUFMetadata() : type(GGUF_TYPE_COUNT) {}
};

/**
 * @brief Representación de un tensor GGUF
 */
struct GGUFTensor {
    std::string name;       ///< Nombre del tensor
    enum ggml_type type;    ///< Tipo de datos del tensor
    size_t size;            ///< Tamaño del tensor en bytes
    std::vector<int64_t> dims; ///< Dimensiones del tensor
    
    /// Almacenamiento de datos usando variant
    std::variant<
        std::vector<uint8_t>,    ///< Para tipos cuantizados
        std::vector<float>,      ///< Para F32
        std::vector<uint16_t>,   ///< Para F16
        std::vector<int32_t>     ///< Para I32
    > data;

    /**
     * @brief Obtiene los datos del tensor como vector del tipo especificado
     * @tparam T Tipo de datos solicitado
     * @return Puntero al vector de datos o nullptr si el tipo no coincide
     */
    template<typename T>
    const std::vector<T>* get_data() const {
        return std::get_if<std::vector<T>>(&data);
    }
};

/**
 * @brief Contenedor principal de datos GGUF
 */
struct GraphData {
    GGUFHeader header;                      ///< Encabezado GGUF
    std::vector<GGUFMetadata> metadata;     ///< Lista de metadatos
    std::vector<GGUFTensor> tensors;        ///< Lista de tensores
    
    /**
     * @brief Busca metadatos por clave
     * @param key Clave a buscar
     * @return Puntero al metadato o nullptr si no se encuentra
     */
    const GGUFMetadata* find_metadata(const std::string& key) const;
    
    /**
     * @brief Busca un tensor por nombre
     * @param name Nombre del tensor a buscar
     * @return Puntero al tensor o nullptr si no se encuentra
     */
    const GGUFTensor* find_tensor(const std::string& name) const;
};

#endif // GRAPH_DATA_STRUCTS_H

