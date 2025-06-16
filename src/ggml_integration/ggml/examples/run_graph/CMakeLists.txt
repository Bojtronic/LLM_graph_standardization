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
    ../../../new_features/graph_file.cpp
    ../../../new_features/quantization_management.cpp
)

# Crear el ejecutable
add_executable(${TARGET} ${SOURCES})

# Incluir directorios de cabecera
target_include_directories(${TARGET} PRIVATE 
    ../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../new_features
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ggml-cuda
)

# Configuración para CUDA
if(GGML_CUDA)
    message(STATUS "Configurando soporte CUDA...")
    
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    
    # Añadir definiciones
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_CUDA)
    
    # Archivos fuente CUDA específicos
    set(CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/ggml-cuda/ggml-cuda.cu)
    
    # Verificar que el archivo existe
    if(NOT EXISTS ${CUDA_SOURCES})
        message(FATAL_ERROR "Archivo ggml-cuda.cu no encontrado en: ${CUDA_SOURCES}")
    endif()
    
    # Añadir como fuente y marcar como CUDA
    target_sources(${TARGET} PRIVATE ${CUDA_SOURCES})
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
    
    # Configuración de arquitectura CUDA
    set(CMAKE_CUDA_ARCHITECTURES "native")
    
    # Enlazar bibliotecas CUDA necesarias
    target_link_libraries(${TARGET} PRIVATE 
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
    )
    
    message(STATUS "Fuente CUDA añadida: ${CUDA_SOURCES}")
endif()

# Enlazar con GGML base
target_link_libraries(${TARGET} PRIVATE ggml)

# Configuración para Metal
if(GGML_METAL AND APPLE)
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    target_compile_definitions(${TARGET} PRIVATE GGML_USE_METAL)
    target_link_libraries(${TARGET} PRIVATE ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})
endif()

# Opciones de compilación
target_compile_options(${TARGET} PRIVATE -Wall -Wextra -O2)

