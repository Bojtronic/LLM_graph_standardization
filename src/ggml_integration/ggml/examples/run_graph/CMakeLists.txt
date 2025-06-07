# Nombre del ejecutable
set(TARGET run_graph)

# Requerir C++20 
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Archivos fuente
set(SOURCES
    main.cpp
    ../../../new_features/model_modules.cpp
    ../../../new_features/run_models.cpp
    ../../../new_features/gguf_loader.cpp
    ../../../new_features/graph_data_structs.cpp
    ../../../new_features/llama_runner.cpp
    ../../../new_features/vit_runner.cpp
    ../../../new_features/whisper_runner.cpp
)

# Crear el ejecutable
add_executable(${TARGET} ${SOURCES})

# Enlazar con las bibliotecas necesarias
target_link_libraries(${TARGET} PRIVATE 
    ggml
    $<$<BOOL:${GGML_CUDA}>:ggml-cuda>
    $<$<BOOL:${GGML_CUDA}>:CUDA::cudart>
)

# Opciones de compilación
target_compile_options(${TARGET} PRIVATE -Wall -Wextra -O2)

# Incluir directorios de cabecera
target_include_directories(${TARGET} PRIVATE 
    ../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../new_features
)

# Para GPU offloading 
if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(GGML_USE_CUDA)
    enable_language(CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
