#pragma once
// Slice BF — procedural terrain / heightmap. PURE CPU (engine/math + the scene Vertex/index types +
// stdlib only); NO RHI or graphics-backend symbols. Compiled into hf_core (ASan-scoped) + hf_engine,
// and into the standalone metal_headless target so the Windows/Vulkan and Apple/Metal builds generate
// a BIT-IDENTICAL mesh from the SAME code (the cross-backend golden contract).
//
// Determinism: Height(x,z) is a fixed pure function — a sum of a few sines/cosines plus a deterministic
// integer value-noise lattice (bilinear). NO RNG, NO clock, NO float-time input. Same function => same
// mesh => same pixels => goldens match on both backends.
#include <cstdint>
#include <vector>

#include "scene/vertex.h"

namespace hf::terrain {

// --- Deterministic height field --------------------------------------------------------------------
// Height(x, z) in world units (pre-heightScale). The exact formula (constants are documented + locked
// in heightmap.cpp):
//
//   Height(x,z) = A1*sin(f1*x)*cos(f1*z)
//               + A2*sin(f2*x + P2x)*cos(f2*z + P2z)
//               + A3*ValueNoise(x*nf, z*nf)
//
// where ValueNoise is a hash-lattice value noise (integer-lattice corner values in [0,1) from a 2D
// integer hash, bilinearly interpolated). Pure, branch-light, locale/RNG/clock free. The two callers
// (Windows engine + metal_headless) compile this SAME TU, so the field — and therefore the generated
// mesh — is bit-identical across backends.
float Height(float x, float z);

// --- Generated terrain mesh ------------------------------------------------------------------------
// A CPU vertex/index grid ready to upload through the existing lit/PBR scene mesh path
// (scene::Vertex + uint32 indices). `verts.size() == n*n`, `indices.size() == (n-1)*(n-1)*6`.
struct TerrainMesh {
    std::vector<scene::Vertex> verts;
    std::vector<uint32_t>      indices;
    float peak = 0.0f;  // max world-space Y (= max Height*heightScale) over the grid — the deterministic stat.
};

// Build an `n x n` vertex grid over [-worldSize/2, +worldSize/2]^2 in the XZ plane, displacing each
// vertex by y = Height(x,z) * heightScale. UVs = grid coordinates in [0,1]^2. Per-vertex normals come
// from CENTRAL FINITE DIFFERENCES of Height (N = normalize(cross(dz, dx)); see the .cpp). Per-vertex
// COLOR carries a deterministic height-based tint (low=grass green -> high=rock/snow) so the relief
// reads clearly when drawn through the unchanged lit shader (which multiplies texture * vertex color)
// — no new shader required. Tangent = +X (dP/du), matching the Plane's convention. Indices wind two
// CCW (when viewed from above) triangles per quad, so the surface is front-facing under the engine's
// back-face culling. `n` must be >= 2.
TerrainMesh BuildTerrain(int n, float worldSize, float heightScale);

}  // namespace hf::terrain
