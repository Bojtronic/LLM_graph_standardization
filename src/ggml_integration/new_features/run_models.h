#ifndef RUN_MODELS_H
#define RUN_MODELS_H

#include <string>
#include "gguf.h"
#include "ggml.h"
#include "model_modules.h"
#include "graph_data_structs.h"
#include "llama_runner.h"
#include "vit_runner.h"
#include "whisper_runner.h"


// Function declarations
ModelType detect_model_type(const gguf_context* ctx);
void print_usage(const char* prog_name);
ModelParams parse_command_line(int argc, char** argv);
//void configure_model_specific_params(ModelParams& params, const gguf_context* ctx);
void run_model(bool use_gpu, GraphData graph_data);

#endif // RUN_MODELS_H
