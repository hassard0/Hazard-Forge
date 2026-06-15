// Hazard Forge — JSON loader for material graphs (Slice AV).
//
// Parses an assets/materials/*.mat.json document into a material::Graph using the vendored
// single-header JSON parser (third_party/json/json.h), the same parser scene_io + introspect use.
// Pure CPU; no backend symbols.
//
// Schema (see assets/materials/showcase.mat.json):
//   {
//     "name": "...",
//     "nodes": [
//       { "id": 0, "type": "Constant", "value": [r,g,b,a], "outType": "float3" },
//       { "id": 1, "type": "UV" },
//       { "id": 2, "type": "TextureSample", "texture": "baseColorTex" },
//       { "id": 3, "type": "Fresnel", "power": 3.0 },
//       { "id": 4, "type": "Lerp" },
//       { "id": 5, "type": "Multiply" },
//       { "id": 6, "type": "Add" },
//       { "id": 99, "type": "PBROutput" }
//     ],
//     "edges": [
//       { "from": 1, "to": 2, "port": "uv" },
//       { "from": 2, "to": 4, "port": "a" },
//       { "from": 0, "to": 4, "port": "b" },
//       { "from": 3, "to": 4, "port": "t" },
//       { "from": 4, "to": 99, "port": "baseColor" }
//     ]
//   }
#pragma once

#include "material/shader_graph.h"

#include <string>

namespace hf::material {

struct LoadResult {
    bool ok = false;
    std::string error;
    Graph graph;
    std::string name;   // the material's "name" field (informational).
    explicit operator bool() const { return ok; }
};

// Parse from an in-memory JSON string.
LoadResult LoadGraphFromJson(const std::string& json);

// Read + parse a .mat.json file from disk.
LoadResult LoadGraphFromFile(const std::string& path);

}  // namespace hf::material
