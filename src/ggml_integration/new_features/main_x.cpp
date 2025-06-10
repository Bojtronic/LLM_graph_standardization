#include "model_modules.h"
#include "run_models.h"
#include "gguf_loader.h"
#include "graph_file.h"  
#include <iostream>
#include <filesystem> 

namespace fs = std::filesystem;


// ./programa --model ruta/al/modelo.gguf --gpu
// ./programa -m ruta/al/modelo.gguf --use-gpu

/**
 * @brief Punto de entrada principal
 */
int main(int argc, char** argv) {
    // Parsear línea de comandos
    ModelParams params;
    try {
        params = parse_command_line(argc, argv);
        if (params.model_path.empty()) {
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing command line: " << e.what() << std::endl;
        return 1;
    }

    // Cargar modelo GGUF
    ggml_context* ctx = nullptr;
    gguf_context* ctx_gguf = nullptr;
    
    try {
        gguf_init_params gguf_params = {
            .no_alloc = true,
            .ctx = &ctx,
        };
        
        ctx_gguf = gguf_init_from_file(params.model_path.c_str(), gguf_params);
        if (!ctx_gguf) {
            throw std::runtime_error("Failed to load GGUF file: " + params.model_path);
        }

        // Procesar datos del grafo
        GraphData graph_data = gguf_graph_data(ctx_gguf, params.model_path.c_str());

        // Generar nombre del archivo .graph
        fs::path graph_path = fs::path(params.model_path).replace_extension(".graph");
        
        // Escribir datos del grafo
        if (!write_graph_data(graph_path.string(), graph_data)) {
            throw std::runtime_error("Failed to write graph data to: " + graph_path.string());
        }
        std::cout << "Graph data successfully written to: " << graph_path << std::endl;


    
        // Ejecutar modelo
        run_model(params.use_gpu, graph_path);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (ctx_gguf) gguf_free(ctx_gguf);
        return 1;
    }

    // Limpieza
    if (ctx_gguf) gguf_free(ctx_gguf);
    return 0;
}


