#pragma once
// Slice W — debug-draw convenience emitters that take ENGINE OBJECTS (scene vertices, the physics
// world, light directions) and append the right line primitives to a DebugDraw. Kept separate from
// the pure DebugDraw class so the core stays math-only; these add scene/ + physics/ deps but remain
// pure C++ (no RHI/graphics-backend symbols), so they live in hf_core alongside DebugDraw.
#include <span>

#include "debug/debug_draw.h"
#include "math/math.h"
#include "physics/world.h"
#include "scene/vertex.h"

namespace hf::debug {

// World-space AABB of a posed mesh: compute the [min,max] from the CPU vertices, transform all 8
// corners by `model`, and draw the AABB hugging the posed mesh (re-fit from the transformed
// corners so it stays axis-aligned in world space).
void MeshAabb(DebugDraw& dd, std::span<const scene::Vertex> verts, const Mat4& model, Vec3 color);

// World-space AABB given the mesh's object-space [min,max] (already computed at build time) and a
// model matrix. Same fit-from-transformed-corners logic as MeshAabb.
void AabbWorld(DebugDraw& dd, Vec3 localMin, Vec3 localMax, const Mat4& model, Vec3 color);

// Vertex normals as "hairs": one short segment per vertex, from the world-space position along the
// world-space normal, length `len`.
void MeshNormals(DebugDraw& dd, std::span<const scene::Vertex> verts, const Mat4& model, float len,
                 Vec3 color);

// A directional-light arrow: a shaft from `origin` along `dir` for `len`, plus four short head fins
// so the pointing direction is unmistakable.
void LightArrow(DebugDraw& dd, Vec3 origin, Vec3 dir, float len, Vec3 color);

// Physics contact markers: recompute the world's sphere/ground + sphere/sphere contacts (same fixed
// order the solver uses, read-only — the solver keeps contacts local), drawing a small 3-axis cross
// at each contact point and a short stub along the contact normal.
void PhysicsContacts(DebugDraw& dd, const physics::World& world, Vec3 pointColor, Vec3 normalColor);

}  // namespace hf::debug
