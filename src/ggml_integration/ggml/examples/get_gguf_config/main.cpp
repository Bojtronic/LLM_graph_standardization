#include "gguf_loader.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cctype>
#include <stdexcept>
#include <set>
#include <map>
#include <utility>
#include "arch_info.h"
#include "gguf.h"
#include <filesystem>
#include <graph_file.h>

namespace fs = std::filesystem;



int main() {
    const char *fname = "llama-2-7b.Q2_K.gguf";
    const char *output_filename = "output.txt";

    // Verifica si el nombre del archivo es válido
    if (!fname) {
        std::cerr << "Nombre de archivo inválido (NULL)\n";
        return 1;  // Código de error
    }

    struct ggml_context *ctx = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx,
    };

    // Intenta cargar el archivo GGUF
    struct gguf_context *ctx_gguf = gguf_init_from_file(fname, params);
    if (!ctx_gguf) {
        std::cerr << "No se pudo cargar el archivo GGUF '" << fname << "'\n";
        return 1;  // Código de error
    }

    GraphData graph_data = gguf_graph_data(ctx_gguf, fname);

    //print_graph_data(graph_data, output_filename);

    // Generar nombre del archivo .graph
    fs::path graph_path = fs::path(fname).replace_extension(".graph");

    // Generar nombre del archivo .txt
    fs::path graph_info_path = fs::path(fname).replace_extension(".txt");

    // Escribir datos del grafo
    if (!write_graph_data(graph_path.string(), graph_data)) {
        throw std::runtime_error("Failed to write graph data to: " + graph_path.string());
    }
    std::cout << "Graph data successfully written to: " << graph_path << std::endl;

    print_tensor_data_from_file(graph_path.string(), graph_info_path.string());


    // Liberar el contexto GGUF cuando ya no sea necesario
    gguf_free(ctx_gguf);

    return 0;  // Éxito
}