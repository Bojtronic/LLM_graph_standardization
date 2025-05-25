#ifndef RUN_MODELS_H
#define RUN_MODELS_H

#include <string>
#include "gguf.h"
#include "ggml.h"
#include "model_modules.h"
#include "model_runner.h"

// Function declarations
ModelType detect_model_type(const gguf_context* ctx);
void print_usage(const char* prog_name);
ModelParams parse_command_line(int argc, char** argv);
void configure_model_specific_params(ModelParams& params, const gguf_context* ctx);
void run_model(const ModelParams& params);

#endif // RUN_MODELS_H
