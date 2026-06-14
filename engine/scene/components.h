#pragma once
// Hazard Forge — scene ECS components.
//
// The renderable scene is expressed as ECS entities (see engine/ecs/ecs.h). Each drawable object
// is an entity carrying these three POD-ish components — the SAME data the old scene::Renderable
// struct held, just split into separate component pools so the render graph can query them.
//
//   TransformC : the object's world transform (reuses scene::Transform).
//   MeshC      : a non-owning pointer to the geometry to draw.
//   MaterialC  : the PBR material — base-color + tangent-space normal map textures (opaque rhi
//                pointers held as values) plus the metallic/roughness factors.
//
// rhi pointers are stored as opaque values; the ECS core never dereferences them, so ecs.h stays
// free of any rhi/backend symbols.
#include "scene/mesh.h"
#include "scene/transform.h"
#include "rhi/rhi.h"

namespace hf::scene {

struct TransformC { Transform t; };
struct MeshC { Mesh* mesh = nullptr; };
struct MaterialC {
    rhi::ITexture* base = nullptr;    // base-color texture
    rhi::ITexture* normal = nullptr;  // tangent-space normal map
    float metallic = 0.0f;
    float roughness = 0.5f;
};

}  // namespace hf::scene
