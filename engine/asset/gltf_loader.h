#pragma once
#include "scene/mesh.h"
#include "rhi/rhi.h"
#include "anim/skeleton.h"
#include "anim/animation.h"
#include <memory>
#include <vector>

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

// A skinned glTF model: skinned-vertex geometry + base-color material + a skeleton + animations.
//
//  * mesh       — first primitive uploaded with the scene::SkinnedVertex layout (stride 88). Reads
//                 POSITION / TEXCOORD_0 / JOINTS_0 / WEIGHTS_0. JOINTS_0 (u8/u16) is widened to
//                 float and REMAPPED from glTF skin-joint order to the skeleton's topologically
//                 sorted order. WEIGHTS_0 is renormalized so each vertex's four weights sum to 1.
//                 NORMAL is computed as smooth per-vertex normals from the indexed positions when
//                 absent (the Fox has none); tangent defaults to (1,0,0). NOT recentred — the skin's
//                 inverse-bind matrices define the bind-space origin, so the caller places the mesh
//                 with an explicit model matrix (see `bbMin`/`bbMax`).
//  * baseColor  — RGBA8 base-color texture (same decode path as GltfModel; white fallback).
//  * metallic / roughness — material factors.
//  * skeleton   — joints (parent, inverse-bind, rest TRS) in topologically sorted order.
//  * animations — every animation clip in the file, channels remapped to skeleton joint indices.
//  * bbMin / bbMax — model-space bounding box of the (un-recentred) bind-pose positions, so the
//                 caller can ground-align + scale the placement.
struct SkinnedModel {
    scene::Mesh mesh;
    std::unique_ptr<rhi::ITexture> baseColor;
    float metallic = 1.0f;
    float roughness = 1.0f;
    anim::Skeleton skeleton;
    std::vector<anim::Animation> animations;
    float bbMin[3] = {0, 0, 0};
    float bbMax[3] = {0, 0, 0};

    SkinnedModel(scene::Mesh m, std::unique_ptr<rhi::ITexture> tex, float met, float rough,
                 anim::Skeleton skel, std::vector<anim::Animation> anims,
                 const float bbmin[3], const float bbmax[3])
        : mesh(std::move(m)), baseColor(std::move(tex)), metallic(met), roughness(rough),
          skeleton(std::move(skel)), animations(std::move(anims)) {
        for (int k = 0; k < 3; ++k) { bbMin[k] = bbmin[k]; bbMax[k] = bbmax[k]; }
    }

    // Find an animation by name; returns nullptr if not present.
    const anim::Animation* FindAnimation(const char* name) const;
};

// Load a skinned glTF/glb model (skin + first animation set). See SkinnedModel.
// Throws std::runtime_error on parse/load/validation failure, missing POSITION, or missing skin.
SkinnedModel LoadSkinnedGltfModel(rhi::IRHIDevice& device, const char* path);

} // namespace hf::asset
