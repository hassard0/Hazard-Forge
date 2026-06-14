#pragma once
#include "scene/mesh.h"
#include "rhi/rhi.h"
#include <memory>

namespace hf::asset {

// Load the first primitive of the first mesh of a glTF/glb file into a scene::Mesh.
//
// GEOMETRY ONLY (legacy): reads POSITION / NORMAL / TEXCOORD_0 into the engine's
// scene::Vertex layout and the index buffer (u16 or u32, widened to u32). glTF
// materials/textures/animation are ignored — the caller renders the returned mesh
// with a fixed material.
//
// The returned geometry is recentred so its bounding box centre sits at the origin,
// so the caller only needs a uniform scale + translate to place it. Vertex colour is
// set to a neutral tint ({0.8, 0.8, 0.85}); UV defaults to (0,0) if TEXCOORD_0 is absent.
//
// Throws std::runtime_error on parse/load/validation failure or missing POSITION.
scene::Mesh LoadGltfMesh(rhi::IRHIDevice& device, const char* path);

// Geometry + base-color material loaded from a glTF/glb file.
//
//  * mesh      — same geometry as LoadGltfMesh (first primitive, recentred to origin).
//  * baseColor — an RGBA8 texture decoded from the material's base-color texture. For a
//                .glb the image is embedded in a buffer_view; the bytes are decoded with
//                stb_image. If the material/texture is absent or cannot be decoded, this
//                falls back to a flat-white 1x1 texture (so the caller always has a valid
//                texture to bind).
//  * metallic  — pbr_metallic_roughness.metallic_factor (defaults to 1.0 per glTF spec).
//  * roughness — pbr_metallic_roughness.roughness_factor (defaults to 1.0 per glTF spec).
struct GltfModel {
    scene::Mesh mesh;
    std::unique_ptr<rhi::ITexture> baseColor;
    float metallic = 1.0f;
    float roughness = 1.0f;

    GltfModel(scene::Mesh m, std::unique_ptr<rhi::ITexture> tex, float met, float rough)
        : mesh(std::move(m)), baseColor(std::move(tex)), metallic(met), roughness(rough) {}
};

// Load geometry + base-color texture + metallic/roughness factors. See GltfModel.
// Throws std::runtime_error on parse/load/validation failure or missing POSITION.
GltfModel LoadGltfModel(rhi::IRHIDevice& device, const char* path);

} // namespace hf::asset
