# Nombre del ejecutable
set(TARGET graph_image)

# Requerir C++17
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Archivo fuente
set(SOURCES 
    main.cpp
    ../../../new_features/model_modules.cpp
    ../../../new_features/run_models.cpp
    ../../../new_features/gguf_loader.cpp
    ../../../new_features/graph_data_structs.cpp
    ../../../new_features/llama_runner.cpp
    ../../../new_features/vit_runner.cpp
    ../../../new_features/whisper_runner.cpp
    ../../../new_features/graph_file.cpp
    ../../../new_features/quantization_management.cpp
    ../../../new_features/arch_info.cpp
    
)

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