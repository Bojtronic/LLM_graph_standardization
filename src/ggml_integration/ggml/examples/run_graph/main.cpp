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

        std::cout << "The GGUF context of the file was created " << std::endl; 

        // Generar nombre del archivo .graph
        fs::path graph_path = fs::path(params.model_path).replace_extension(".graph");

        // Procesar datos del grafo
        GraphData graph_data = gguf_graph_data(ctx_gguf, params.model_path.c_str(), graph_path.c_str());

        std::cout << "The GraphData structure and file.graph was created from the GGUF context " << std::endl;

        fs::path dot_path = fs::path(params.model_path).replace_extension(".dot");

        std::string dot_graph = generate_computational_graph(graph_data);
        
        if (!save_dot_to_file(dot_graph, dot_path.string())) {
            throw std::runtime_error("Failed to save DOT graph to file: " + dot_path.string());
        }
        std::cout << "The computational graph image was generated and saved to " << dot_path.string() << std::endl;

        /*
        GGUFMetadata md = read_metadata("llama-2-7b.Q2_K.graph", "llama.rope.dimension_count");

        if (md.type == GGUF_TYPE_COUNT) {
            std::cerr << "Metadata key not found in the graph file." << std::endl;
        } else {
            std::cout << "Key: " << md.key << std::endl;
            std::cout << "Value: " << md.value.u32 << std::endl;
        }
        */

        GGUFTensor tensor = read_tensor(graph_path.string(), "blk.5.attn_norm.weight");
        if (tensor.name.empty()) {
            std::cerr << "Tensor not found in the graph file." << std::endl;
        } else {
            std::cout << "Tensor Name: " << tensor.name << std::endl;
            std::cout << "Tensor Type: " << ggml_type_name(tensor.type) << std::endl;
            std::cout << "Tensor Size: " << tensor.size << " bytes" << std::endl;
            std::cout << "Tensor Dimensions: ";
            for (const auto& dim : tensor.dims) {
                std::cout << dim << " ";        
            
            }
        }


        // Ejecutar modelo
        //run_model(params.use_gpu, graph_path);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (ctx_gguf) gguf_free(ctx_gguf);
        return 1;
    }

    // Limpieza
    if (ctx_gguf) gguf_free(ctx_gguf);
    return 0;
}


