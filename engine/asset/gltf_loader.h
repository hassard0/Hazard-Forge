#pragma once
#include "scene/mesh.h"
#include "rhi/rhi.h"

namespace hf::asset {

// Load the first primitive of the first mesh of a glTF/glb file into a scene::Mesh.
//
// GEOMETRY ONLY (Phase 1): reads POSITION / NORMAL / TEXCOORD_0 into the engine's
// scene::Vertex layout and the index buffer (u16 or u32, widened to u32). glTF
// materials/textures/animation are intentionally ignored — the caller renders the
// returned mesh with the engine's existing PBR (e.g. a fixed metallic material).
//
// The returned geometry is recentred so its bounding box centre sits at the origin,
// so the caller only needs a uniform scale + translate to place it. Vertex colour is
// set to a neutral tint ({0.8, 0.8, 0.85}); UV defaults to (0,0) if TEXCOORD_0 is absent.
//
// Throws std::runtime_error on parse/load/validation failure or missing POSITION.
scene::Mesh LoadGltfMesh(rhi::IRHIDevice& device, const char* path);

} // namespace hf::asset
