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
        std::cout << graph_path.c_str() << std::endl;
        std::cout << graph_path << std::endl;
        

        GGUFMetadata md = read_metadata(graph_path, "tokenizer.ggml.tokens");
        if (md.type == GGUF_TYPE_COUNT) {
            std::cerr << "No se encontró la metadata de tokens en el modelo GGUF." << std::endl;
        }
        std::cout << "Token metadata found: " << md.key << " of type " << gguf_type_name(md.type) << std::endl;
        std::cout << "Number of tokens: " << std::get<std::vector<std::string>>(md.array.data).size() << std::endl;
        */

        GGUFTensor tensor = read_tensor(graph_path, "token_embd.weight");
        //GGUFTensor tensor = read_tensor(graph_path, "blk.0.attn_norm.weight");
        
        if (tensor.name.empty()) {
            std::cerr << "Error: Failed to read tensor from GGUF file" << std::endl;
        }
        std::cout << "Tensor name: " << tensor.name << std::endl;
        std::cout << "Tensor n dims: " << tensor.n_dims << std::endl;
        std::cout << "dim 0: " << tensor.dims[0] << " dim 1: " << tensor.dims[1] << std::endl;



        
     
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


