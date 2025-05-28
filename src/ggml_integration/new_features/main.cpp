#include "model_modules.h"
#include "run_models.h"
#include <iostream>

int main(int argc, char** argv) {
    // Parse command line arguments
    ModelParams params = parse_command_line(argc, argv);
    if (params.model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Load GGUF file to detect model type
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
    

    // Detect model type and configure parameters
    params.type = detect_model_type(ctx_gguf);
    configure_model_specific_params(params, ctx_gguf);
    
    // Run the appropriate model
    run_model(params, graph_data);

    // Cleanup
    gguf_free(ctx_gguf);
    
    return 0;
}
