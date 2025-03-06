#include "ggml.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <tinyxml2.h> // Librería para parsear XML

using namespace tinyxml2;

// Estructura para almacenar la información del modelo
struct model {
    std::map<std::string, ggml_tensor *> tensors; // Mapa de tensores por nombre
    ggml_context * ctx; // Contexto de GGML
};

// Función para cargar un tensor desde el XML
std::pair<std::string, ggml_tensor *> load_tensor(ggml_context * ctx, XMLElement * tensor_element) {
    const char * name = tensor_element->Attribute("name");
    const char * type = tensor_element->Attribute("type");
    int64_t ne[4] = {1, 1, 1, 1}; // Dimensiones del tensor

    // Obtener dimensiones del tensor (si están definidas)
    if (tensor_element->Attribute("ne0")) ne[0] = tensor_element->IntAttribute("ne0");
    if (tensor_element->Attribute("ne1")) ne[1] = tensor_element->IntAttribute("ne1");
    if (tensor_element->Attribute("ne2")) ne[2] = tensor_element->IntAttribute("ne2");
    if (tensor_element->Attribute("ne3")) ne[3] = tensor_element->IntAttribute("ne3");

    // Crear el tensor en el contexto de GGML
    ggml_tensor * tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);

    // Cargar datos del tensor (si están definidos)
    const char * data = tensor_element->Attribute("data");
    if (data) {
        std::istringstream iss(data);
        float value;
        int i = 0;
        while (iss >> value) {
            ((float *)tensor->data)[i++] = value;
        }
    }

    // Devolver el nombre y el tensor como un par
    return {name, tensor};
}

// Función para construir el grafo de cómputo desde el XML
ggml_cgraph * build_graph_from_xml(model & model, XMLElement * layers_element) {
    ggml_cgraph * gf = ggml_new_graph(model.ctx);

    // Recorrer las capas del modelo
    for (XMLElement * layer_element = layers_element->FirstChildElement("layer"); layer_element; layer_element = layer_element->NextSiblingElement("layer")) {
        // Recorrer las operaciones de la capa
        for (XMLElement * op_element = layer_element->FirstChildElement("operations")->FirstChildElement("operation"); op_element; op_element = op_element->NextSiblingElement("operation")) {
            const char * op_type = op_element->Attribute("type");

            // Manejar diferentes tipos de operaciones
            if (strcmp(op_type, "multi_head_attention") == 0) {
                // Implementar atención multi-cabezal
                // (Este es un ejemplo simplificado)
                ggml_tensor * q = model.tensors[op_element->FirstChildElement("inputs")->FirstChildElement("input")->Attribute("ref")];
                ggml_tensor * k = model.tensors[op_element->FirstChildElement("inputs")->FirstChildElement("input")->Attribute("ref")];
                ggml_tensor * v = model.tensors[op_element->FirstChildElement("inputs")->FirstChildElement("input")->Attribute("ref")];

                ggml_tensor * result = ggml_mul_mat(model.ctx, q, k);
                result = ggml_scale(model.ctx, result, 1.0f / sqrtf((float)q->ne[0]));
                result = ggml_soft_max(model.ctx, result);
                result = ggml_mul_mat(model.ctx, result, v);

                // Guardar el resultado en el mapa de tensores
                const char * output_name = op_element->FirstChildElement("outputs")->FirstChildElement("output")->Attribute("name");
                model.tensors[output_name] = result;
                ggml_build_forward_expand(gf, result);
            } else if (strcmp(op_type, "feed_forward") == 0) {
                // Implementar red feed-forward
                // (Este es un ejemplo simplificado)
                ggml_tensor * input = model.tensors[op_element->FirstChildElement("inputs")->FirstChildElement("input")->Attribute("ref")];
                ggml_tensor * result = ggml_add(model.ctx, input, ggml_mul(model.ctx, input, ggml_new_f32(model.ctx, 2.0f)));

                // Guardar el resultado en el mapa de tensores
                const char * output_name = op_element->FirstChildElement("outputs")->FirstChildElement("output")->Attribute("name");
                model.tensors[output_name] = result;
                ggml_build_forward_expand(gf, result);
            }
            // Agregar más operaciones según sea necesario
        }
    }

    return gf;
}

// Función principal
int main(void) {
    ggml_time_init();

    // Cargar el archivo XML
    XMLDocument doc;
    if (doc.LoadFile("model_graph.xml") != XML_SUCCESS) {
        std::cerr << "Error al cargar el archivo XML." << std::endl;
        return 1;
    }

    // Crear el modelo
    model model;
    size_t ctx_size = 1024 * 1024 * 10; // Tamaño inicial del contexto (ajustar según sea necesario)
    model.ctx = ggml_init({ctx_size, NULL, false});

    // Mapa para asociar nombres a tensores
    std::map<std::string, ggml_tensor *> tensor_map;

    // Cargar tensores desde el XML
    XMLElement * root = doc.RootElement();
    for (XMLElement * tensor_element = root->FirstChildElement("layers")->FirstChildElement("layer")->FirstChildElement("inputs")->FirstChildElement("tensor"); tensor_element; tensor_element = tensor_element->NextSiblingElement("tensor")) {
        auto [name, tensor] = load_tensor(model.ctx, tensor_element); // Cargar tensor y nombre
        tensor_map[name] = tensor; // Asociar nombre con tensor
    }

    // Construir el grafo de cómputo desde el XML
    ggml_cgraph * gf = build_graph_from_xml(model, root->FirstChildElement("layers"));

    // Ejecutar el grafo
    int n_threads = 1; // Número de hilos
    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

    // Obtener el tensor de salida
    ggml_tensor * output = ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1);

    // Imprimir el resultado
    std::vector<float> out_data(ggml_nelements(output));
    memcpy(out_data.data(), output->data, ggml_nbytes(output));
    printf("Resultado:\n");
    for (size_t i = 0; i < out_data.size(); i++) {
        printf("%.2f ", out_data[i]);
    }
    printf("\n");

    // Liberar memoria
    ggml_free(model.ctx);
    return 0;
}
