#include "graph_data_structs.h"

const GGUFMetadata* GraphData::find_metadata(const std::string& key) const {
    for (const auto& md : metadata) {
        if (md.key == key) return &md;
    }
    return nullptr;
}

const GGUFTensor* GraphData::find_tensor(const std::string& name) const {
    for (const auto& tensor : tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}