#include "run_models.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>

ModelType detect_model_type(const gguf_context* ctx) {
    int arch_key = gguf_find_key(ctx, "general.architecture");
    const char* arch = (arch_key != -1) ? gguf_get_val_str(ctx, arch_key) : nullptr;
    if (!arch) {
        return MODEL_TYPE_UNKNOWN;
    }

    if (strcmp(arch, "llama") == 0) {
        return MODEL_TYPE_LLAMA;
    }
    
    if (strstr(arch, "vit") != nullptr) {
        return MODEL_TYPE_VIT;
    }
    
    if (strcmp(arch, "whisper") == 0) {
        return MODEL_TYPE_WHISPER;
    }
    
    return MODEL_TYPE_UNKNOWN;
}

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help            Show this help message\n";
    std::cout << "  -m, --model FNAME     Path to GGUF model file (required)\n";
    std::cout << "  -i, --input FNAME     Input file path\n";
    std::cout << "  -o, --output FNAME    Output file path\n";
    std::cout << "  -t, --threads N       Number of threads to use (default: auto)\n";
    std::cout << "  -g, --gpu-layers N    Number of layers to offload to GPU (default: 0)\n";
    std::cout << "  -s, --seed N          RNG seed (default: -1, random)\n";
    std::cout << "\n";
    std::cout << "Model-specific parameters will be requested based on the detected model type.\n";
}

ModelParams parse_command_line(int argc, char** argv) {
    ModelParams params;
    params.type = MODEL_TYPE_UNKNOWN;
    params.n_threads = 0;
    params.n_gpu_layers = 0;
    params.use_gpu = false;
    params.seed = -1;
    params.temperature = 0.8f;
    params.top_k = 40;
    params.top_p = 0.9f;
    params.n_ctx = 2048;
    params.n_batch = 512;
    params.image_size = 224;
    params.n_mels = 80;
    params.n_audio_ctx = 1500;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                params.model_path = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for --model\n";
                exit(1);
            }
        } else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                params.input_path = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for --input\n";
                exit(1);
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                params.output_path = argv[++i];
            } else {
                std::cerr << "Error: Missing argument for --output\n";
                exit(1);
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                params.n_threads = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: Missing argument for --threads\n";
                exit(1);
            }
        } else if (arg == "-g" || arg == "--gpu-layers") {
            if (i + 1 < argc) {
                params.n_gpu_layers = std::atoi(argv[++i]);
                params.use_gpu = params.n_gpu_layers > 0;
            } else {
                std::cerr << "Error: Missing argument for --gpu-layers\n";
                exit(1);
            }
        } else if (arg == "-s" || arg == "--seed") {
            if (i + 1 < argc) {
                params.seed = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: Missing argument for --seed\n";
                exit(1);
            }
        } else {
            std::cerr << "Error: Unknown argument " << arg << "\n";
            print_usage(argv[0]);
            exit(1);
        }
    }

    return params;
}

void configure_model_specific_params(ModelParams& params, const gguf_context* ctx) {
    switch (params.type) {
        case MODEL_TYPE_LLAMA: {
            const int n_ctx = gguf_find_key(ctx, "llama.context_length");
            if (n_ctx != -1) {
                params.n_ctx = gguf_get_val_u32(ctx, n_ctx);
            }
            
            const int n_batch = gguf_find_key(ctx, "llama.batch_size");
            if (n_batch != -1) {
                params.n_batch = gguf_get_val_u32(ctx, n_batch);
            }
            
            std::cout << "LLaMA model detected. Configuring parameters...\n";
            std::cout << "Context length: " << params.n_ctx << "\n";
            std::cout << "Batch size: " << params.n_batch << "\n";
            break;
        }
        case MODEL_TYPE_VIT: {
            const int img_size = gguf_find_key(ctx, "vit.image_size");
            if (img_size != -1) {
                params.image_size = gguf_get_val_u32(ctx, img_size);
            }
            
            std::cout << "ViT model detected. Configuring parameters...\n";
            std::cout << "Image size: " << params.image_size << "\n";
            
            if (params.input_path.empty()) {
                std::cout << "Enter input image path: ";
                std::cin >> params.input_path;
            }
            break;
        }
        case MODEL_TYPE_WHISPER: {
            const int n_mels = gguf_find_key(ctx, "whisper.n_mels");
            if (n_mels != -1) {
                params.n_mels = gguf_get_val_u32(ctx, n_mels);
            }
            
            const int n_audio_ctx = gguf_find_key(ctx, "whisper.n_audio_ctx");
            if (n_audio_ctx != -1) {
                params.n_audio_ctx = gguf_get_val_u32(ctx, n_audio_ctx);
            }
            
            std::cout << "Whisper model detected. Configuring parameters...\n";
            std::cout << "Number of mel bins: " << params.n_mels << "\n";
            std::cout << "Audio context size: " << params.n_audio_ctx << "\n";
            
            if (params.input_path.empty()) {
                std::cout << "Enter input audio path: ";
                std::cin >> params.input_path;
            }
            break;
        }
        default: {
            std::cerr << "Warning: Unknown model type. Using default parameters.\n";
            break;
        }
    }
    
    if (params.input_path.empty() && params.type != MODEL_TYPE_LLAMA) {
        std::cout << "Enter input path: ";
        std::cin >> params.input_path;
    }
    
    if (params.output_path.empty()) {
        std::cout << "Enter output path (optional): ";
        std::cin >> params.output_path;
    }
}

void run_model(const ModelParams& params, GraphData graph_data) {
    ggml_backend_t backend = NULL;
    if (params.use_gpu) {
        backend = ggml_backend_cuda_init(0);
        if (!backend) {
            std::cerr << "Warning: Failed to initialize CUDA backend. Falling back to CPU.\n";
        }
    }
    
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
    
    struct ggml_init_params ggml_params = {
        .mem_size = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    
    struct ggml_context* ctx = ggml_init(ggml_params);
    bool success = false;


    switch (params.type) {
        case MODEL_TYPE_LLAMA:
            std::cout << "Running LLaMA model...\n";
            success = run_llama_model(ctx, backend, params, graph_data);
            break;
        case MODEL_TYPE_VIT:
            std::cout << "Running ViT model...\n";
            success = run_vit_model(ctx, backend, params);
            break;
        case MODEL_TYPE_WHISPER:
            std::cout << "Running Whisper model...\n";
            success = run_whisper_model(ctx, backend, params);
            break;
        default:
            std::cerr << "Error: Unknown model type\n";
    }
    
    if (!success) {
        std::cerr << "Model execution failed\n";
    }
    
    ggml_free(ctx);
    ggml_backend_free(backend);
}
