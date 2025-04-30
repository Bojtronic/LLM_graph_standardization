#ifndef GRAPH_DATA_STRUCTS_H
#define GRAPH_DATA_STRUCTS_H

#include <cstdint>
#include <string>
#include <vector>
#include <variant>
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
        enum gguf_type type;
        size_t size;
        std::variant<
            std::vector<uint8_t>,   // Para tipos crudos/bytes
            std::vector<float>,     // F32
            std::vector<uint16_t>,  // F16
            std::vector<int8_t>,    // I8
            std::vector<int16_t>,   // I16
            std::vector<int32_t>,   // I32
            std::vector<uint32_t>,  // U32
            std::vector<int64_t>,   // I64
            std::vector<uint64_t>,  // U64
            std::vector<double>,    // F64
            std::vector<std::string> // Strings
        > data;
    } array;
    
    std::string str;        ///< Para strings
    
    /// Constructor por defecto
    GGUFMetadata() : type(GGUF_TYPE_COUNT) {}
};

/**
 * @brief Representación de un tensor GGUF
 */
struct GGUFTensor {
    std::string name;           ///< Nombre del tensor
    enum ggml_type type;        ///< Tipo de datos del tensor
    size_t size;                ///< Tamaño del tensor en bytes
    int32_t n_dims;             ///< Número de dimensiones del tensor
    std::vector<int64_t> dims;  ///< Dimensiones del tensor
    
    /// Almacenamiento de datos usando variant
    std::variant<
        std::vector<uint8_t>,    ///< Para tipos cuantizados
        std::vector<int8_t>,     ///< Para Int8
        std::vector<int16_t>,    ///< Para Int16
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


static inline size_t type_size(enum gguf_type type);

template <typename T>
static std::vector<T> read_array_data(const gguf_context *ctx, int64_t i, size_t size) {
    if (!ctx || size == 0) {
        return {};
    }

    const void *src_data = gguf_get_arr_data(ctx, i);
    if (!src_data) {
        return {};
    }

    std::vector<T> dest(size);
    
    if constexpr (std::is_same_v<T, uint8_t>) {
        memcpy(dest.data(), src_data, size * sizeof(T));
    } else {
        const T *typed_src = static_cast<const T *>(src_data);
        std::copy(typed_src, typed_src + size, dest.begin());
    }
    
    return dest;
}

#endif // GRAPH_DATA_STRUCTS_H

