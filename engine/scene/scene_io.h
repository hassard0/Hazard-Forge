#pragma once
// Hazard Forge — data-driven scenes (load/dump).
//
// A scene is DATA, not code. The engine loads a JSON scene description into the ECS and serializes
// the live ECS scene back out, so scenes are authorable + inspectable by an AI agent or a future
// editor. This is the agentic-development foundation: the same scene a sample built inline is now
// an asset (assets/scenes/default.json) that round-trips through the registry without changing a
// single pixel.
//
// Resources (meshes + textures) are GPU/backend objects, so they cannot live in JSON. Instead the
// sample creates them as it does today and registers them BY NAME in a SceneResources table; the
// scene file refers to them by those names. LoadScene resolves the names to the registered
// pointers; DumpScene reverses pointers back to names. The scene_io layer therefore touches no
// vk*/Metal symbols — it only ever handles scene::Mesh*/rhi::ITexture* as opaque, named values.
//
// Schema — a JSON array of objects, each:
//   {
//     "mesh":      "<mesh name>",                 // required; key into SceneResources::meshes
//     "baseColor": "<texture name>" | null,       // optional; key into SceneResources::textures
//     "normalMap": "<texture name>" | null,       // optional; key into SceneResources::textures
//     "metallic":  <float>,                       // default 0.0
//     "roughness": <float>,                        // default 0.5
//     "position":  [x, y, z],                      // default [0,0,0]
//     "euler":     [x, y, z],                      // radians; default [0,0,0]
//     "scale":     [x, y, z]                        // default [1,1,1]
//   }
// Objects become ECS entities IN FILE ORDER, each with TransformC + MeshC + MaterialC, so the
// draw order (view iteration is creation order) matches the file exactly.

#include "ecs/ecs.h"
#include "scene/components.h"
#include "scene/mesh.h"
#include "rhi/rhi.h"

#include <map>
#include <string>

namespace hf::scene {

// A named registry of the GPU/backend resources a scene file can refer to. The sample populates
// this (creating the meshes/textures exactly as it does today) BEFORE loading the scene; the
// scene file names entries here, and DumpScene reverses pointers back to these names.
struct SceneResources {
    std::map<std::string, Mesh*> meshes;
    std::map<std::string, rhi::ITexture*> textures;

    void AddMesh(std::string name, Mesh* mesh) { meshes.emplace(std::move(name), mesh); }
    void AddTexture(std::string name, rhi::ITexture* tex) { textures.emplace(std::move(name), tex); }

    // Resolve a mesh/texture name to its pointer (nullptr if absent / name is empty).
    Mesh* FindMesh(const std::string& name) const {
        auto it = meshes.find(name);
        return it == meshes.end() ? nullptr : it->second;
    }
    rhi::ITexture* FindTexture(const std::string& name) const {
        auto it = textures.find(name);
        return it == textures.end() ? nullptr : it->second;
    }

    // Reverse lookups for DumpScene (pointer -> name; empty string if not found).
    std::string NameOfMesh(const Mesh* mesh) const {
        for (const auto& [name, ptr] : meshes)
            if (ptr == mesh) return name;
        return {};
    }
    std::string NameOfTexture(const rhi::ITexture* tex) const {
        for (const auto& [name, ptr] : textures)
            if (ptr == tex) return name;
        return {};
    }
};

// Parse the JSON scene at `path` and create one ECS entity per object, IN FILE ORDER, each with
// TransformC/MeshC/MaterialC resolved from `resources` by name. Returns the entities created (in
// file order) so the caller can tag specific ones (e.g. the spinnable grid objects).
// Throws std::runtime_error on a missing file, a parse error, or an unknown mesh name.
std::vector<ecs::Entity> LoadScene(ecs::Registry& reg, const SceneResources& resources,
                                   const char* path);

// Serialize the live ECS scene (every entity holding TransformC + MeshC + MaterialC, in view /
// creation order) back to the same JSON schema, resolving resources to names via `resources`.
// This is the machine-readable scene state an agent can inspect. Pretty-printed.
std::string DumpScene(ecs::Registry& reg, const SceneResources& resources);

}  // namespace hf::scene
