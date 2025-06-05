#include "model_modules.h"
#include "run_models.h"
#include "gguf_loader.h"
#include <iostream>

int main(int argc, char** argv) {

    // ./programa --model ruta/al/modelo.gguf --gpu
    // ./programa -m ruta/al/modelo.gguf --use-gpu
    
    ModelParams params = parse_command_line(argc, argv);
    if (params.model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    struct ggml_context* ctx = NULL;
    struct gguf_init_params gguf_params = {
        .no_alloc = true,
        .ctx = &ctx,
    };
    
    struct gguf_context* ctx_gguf = gguf_init_from_file(params.model_path.c_str(), gguf_params);
    if (!ctx_gguf) {
        std::cerr << "Failed to load GGUF file: " << params.model_path << std::endl;
        return 1;
    }

    
    GraphData graph_data = gguf_graph_data(ctx_gguf, params.model_path.c_str());
    
    
    run_model(params.use_gpu, graph_data);

    gguf_free(ctx_gguf);
    
    return 0;
}
