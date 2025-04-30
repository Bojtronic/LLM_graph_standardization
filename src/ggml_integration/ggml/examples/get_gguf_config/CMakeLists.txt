# Nombre del ejecutable
set(TARGET gguf_reader)

# Requerir C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Archivo fuente
set(SOURCES main.cpp)

# Crear el ejecutable
add_executable(${TARGET} ${SOURCES})

# Enlazar con las bibliotecas necesarias
target_link_libraries(${TARGET} PRIVATE ggml)

# Opciones de compilación
target_compile_options(${TARGET} PRIVATE -Wall -Wextra -O2)

# Incluir directorios de cabecera
target_include_directories(${TARGET} PRIVATE 
    ../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../new_features
)

# Para GPU offloading (opcional)
if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()