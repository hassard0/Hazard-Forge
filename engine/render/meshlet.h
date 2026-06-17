#pragma once
// Slice DS — Virtual-Geometry Slice 1: meshlet / cluster decomposition (the Nanite-style arc BEACHHEAD).
// Pure CPU (header-only, no device, no backend symbols). Same header-only pattern as
// engine/render/frustum.h / hiz.h. Namespace hf::render::vg (NOT `cluster` — that name is taken by
// the Forward+ LIGHT clustering in engine/render/cluster.h; this is GEOMETRY clustering).
//
// Shared by the --meshlet-viz showcase (which builds the cluster set over a sphere and draws each
// cluster as an index sub-range with a per-cluster color) AND tests/meshlet_test.cpp, so the unit
// test exercises the EXACT decomposition the renderer uses.
//
// WHAT THIS DOES: partition a mesh's index buffer into clusters of up to kMaxTrisPerCluster (128)
// triangles, each with a CONSERVATIVE per-cluster AABB + bounding sphere. The whole decomposition is
// integer Morton sort + index copy + min/max + one sqrt per cluster (the sqrt is host-only viz/cull
// metadata, never cross-backend GPU-shared), so the output is BIT-IDENTICAL on every platform and
// every run — the simplest possible proof surface for the virtual-geometry beachhead.
//
// CONSERVATIVE-BOUNDS CONTRACT: every referenced vertex of a cluster lies inside [boundMin,boundMax]
// AND within boundRadius of boundCenter. The later slices (DT per-cluster GPU cull, DU Hi-Z occlusion)
// rely on this exactly like frustum.h / hiz.h rely on their conservative contracts — do NOT under-bound.
//
// SEAM DISCIPLINE: this header has ZERO backend (vk*/MTL*/mtl::) symbols. Mentions of "Vulkan"/"Metal"
// here are doc-only; the decomposition emits a reordered index buffer + cluster metadata that the
// showcase uploads through the EXISTING draw + push-constant surface — no new RHI.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "math/math.h"
#include "scene/vertex.h"

namespace hf::render::vg {

// Max triangles per cluster. 128 is the conventional Nanite-style meshlet triangle budget.
static constexpr uint32_t kMaxTrisPerCluster = 128;

// One geometry cluster: a CONTIGUOUS run of triangles in MeshletSet::indices plus conservative bounds.
struct Meshlet {
    uint32_t   triOffset = 0;   // first triangle of this cluster (index into MeshletSet::indices, /3)
    uint32_t   triCount  = 0;   // triangles in this cluster (<= kMaxTrisPerCluster)
    math::Vec3 boundMin;        // object-space AABB min over the cluster's referenced vertices
    math::Vec3 boundMax;        // object-space AABB max over the cluster's referenced vertices
    math::Vec3 boundCenter;     // 0.5 * (boundMin + boundMax)
    float      boundRadius = 0; // max |v - boundCenter| over the cluster's verts (conservative sphere)
};

// The decomposition output: the cluster list + a REORDERED index buffer with the clusters stored
// contiguously. Cluster i covers indices[3*triOffset .. 3*(triOffset + triCount)). The vertex buffer
// is unchanged — DS reorders ONLY indices (shared-vertex clusters; per-cluster local vertex buffers
// are a later refinement, out of scope here).
struct MeshletSet {
    std::vector<Meshlet>  meshlets;
    std::vector<uint32_t> indices;
};

// Interleave three 10-bit lanes (x,y,z each in [0,1023]) into a 30-bit Morton code: bit i of x ->
// output bit 3i, y -> 3i+1, z -> 3i+2. Inputs above 10 bits are masked to 10. Documented +
// round-trip-tested helper (the round-trip recovers the three 10-bit lanes). This is the spatial-
// locality key: nearby centroids get nearby codes, so a sort by code groups spatially-adjacent
// triangles into the same cluster (the Morton coherence the viz makes visible).
inline uint32_t MortonPart10(uint32_t v) {
    v &= 0x3FFu;                       // keep 10 bits
    v = (v | (v << 16)) & 0x030000FFu; // spread bits with 2-bit gaps (standard 10-bit-into-30 spread)
    v = (v | (v << 8))  & 0x0300F00Fu;
    v = (v | (v << 4))  & 0x030C30C3u;
    v = (v | (v << 2))  & 0x09249249u;
    return v;
}
inline uint32_t MortonCode10(uint32_t x, uint32_t y, uint32_t z) {
    return MortonPart10(x) | (MortonPart10(y) << 1) | (MortonPart10(z) << 2);
}

// Deterministic integer hash of a cluster index -> [0,1] RGB. A fixed multiply-xor-shift (a Wang/
// xorshift-style avalanche) gives three well-spread bytes so ADJACENT cluster indices get clearly
// distinct colors. Computed CPU-side and pushed as a flat per-draw color so the fragment shader stays
// a trivial flat-color pass (no GPU-side hash to keep bit-exact). Unit-tested for determinism + spread.
inline math::Vec3 hashColor(uint32_t i) {
    uint32_t h = i * 2654435761u + 0x9E3779B9u; // Knuth multiplicative + golden-ratio offset
    h ^= h >> 15;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    uint32_t r = (h)       & 0xFFu;
    uint32_t g = (h >> 8)  & 0xFFu;
    uint32_t b = (h >> 16) & 0xFFu;
    return math::Vec3{r / 255.0f, g / 255.0f, b / 255.0f};
}

// Partition a mesh's index buffer into spatially-coherent clusters of <= kMaxTrisPerCluster triangles.
//
//   T = indices.size() / 3. Any TRAILING 1-2 indices (a non-multiple-of-3 length) are IGNORED — they
//   cannot form a whole triangle. T == 0 returns an empty MeshletSet{} (the no-op).
//
// Algorithm (all integer-deterministic):
//   1. Per triangle, an ordinary-FP centroid c = (p0+p1+p2)/3 (used ONLY as a sort key — never as
//      cross-backend GPU math). Mesh AABB over all referenced vertices.
//   2. Quantize each centroid into the mesh AABB to a 10-bit-per-axis integer (zero-extent axis -> 0)
//      and interleave to a 30-bit Morton code.
//   3. Sort triangles by (mortonCode, originalTriIndex). The secondary key makes this a TOTAL order
//      (no ties) -> fully deterministic. (This is the deliberate choice over float-greedy adjacency,
//      which risks tie-order nondeterminism.)
//   4. Walk the sorted triangles emitting clusters of kMaxTrisPerCluster consecutive triangles (the
//      last may be partial); append each cluster's 3*triCount indices contiguously into the output.
//   5. Per cluster, the AABB over its referenced verts -> boundMin/Max, boundCenter = 0.5(min+max),
//      boundRadius = max |v - boundCenter| (CONSERVATIVE — see the contract above).
inline MeshletSet BuildMeshlets(std::span<const scene::Vertex> verts,
                                std::span<const uint32_t> indices) {
    MeshletSet out;
    const uint32_t T = static_cast<uint32_t>(indices.size() / 3);  // trailing 1-2 indices ignored
    if (T == 0) return out;                                        // no-op

    auto vpos = [&](uint32_t idx) -> math::Vec3 {
        const scene::Vertex& v = verts[idx];
        return math::Vec3{v.pos[0], v.pos[1], v.pos[2]};
    };

    // --- Per-triangle centroid + mesh AABB over the referenced vertices. ---
    std::vector<math::Vec3> centroid(T);
    math::Vec3 aabbMin{ 1e30f,  1e30f,  1e30f};
    math::Vec3 aabbMax{-1e30f, -1e30f, -1e30f};
    for (uint32_t t = 0; t < T; ++t) {
        math::Vec3 p0 = vpos(indices[3 * t + 0]);
        math::Vec3 p1 = vpos(indices[3 * t + 1]);
        math::Vec3 p2 = vpos(indices[3 * t + 2]);
        centroid[t] = (p0 + p1 + p2) * (1.0f / 3.0f);
        for (const math::Vec3& p : {p0, p1, p2}) {
            aabbMin.x = std::min(aabbMin.x, p.x); aabbMax.x = std::max(aabbMax.x, p.x);
            aabbMin.y = std::min(aabbMin.y, p.y); aabbMax.y = std::max(aabbMax.y, p.y);
            aabbMin.z = std::min(aabbMin.z, p.z); aabbMax.z = std::max(aabbMax.z, p.z);
        }
    }
    const math::Vec3 extent = aabbMax - aabbMin;

    // --- Quantize each centroid into the AABB -> 10-bit/axis -> Morton code. ---
    auto quantize = [](float c, float lo, float ext) -> uint32_t {
        if (ext <= 0.0f) return 0u;                                  // zero-extent axis -> 0
        float f = (c - lo) / ext * 1024.0f;
        if (f < 0.0f) f = 0.0f;
        int q = static_cast<int>(f);                                 // floor (f >= 0)
        if (q > 1023) q = 1023;
        return static_cast<uint32_t>(q);
    };
    std::vector<uint32_t> morton(T);
    for (uint32_t t = 0; t < T; ++t) {
        uint32_t qx = quantize(centroid[t].x, aabbMin.x, extent.x);
        uint32_t qy = quantize(centroid[t].y, aabbMin.y, extent.y);
        uint32_t qz = quantize(centroid[t].z, aabbMin.z, extent.z);
        morton[t] = MortonCode10(qx, qy, qz);
    }

    // --- Sort triangle indices by (mortonCode, originalTriIndex) — a TOTAL order, no ties. ---
    std::vector<uint32_t> order(T);
    for (uint32_t t = 0; t < T; ++t) order[t] = t;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (morton[a] != morton[b]) return morton[a] < morton[b];
        return a < b;  // secondary key: original triangle index (guarantees determinism)
    });

    // --- Emit contiguous clusters of <= kMaxTrisPerCluster triangles. ---
    const uint32_t numClusters = (T + kMaxTrisPerCluster - 1) / kMaxTrisPerCluster;
    out.meshlets.reserve(numClusters);
    out.indices.reserve(static_cast<size_t>(T) * 3);

    uint32_t emittedTris = 0;
    for (uint32_t base = 0; base < T; base += kMaxTrisPerCluster) {
        uint32_t count = std::min(kMaxTrisPerCluster, T - base);
        Meshlet m;
        m.triOffset = emittedTris;
        m.triCount  = count;

        math::Vec3 cMin{ 1e30f,  1e30f,  1e30f};
        math::Vec3 cMax{-1e30f, -1e30f, -1e30f};
        for (uint32_t k = 0; k < count; ++k) {
            uint32_t srcTri = order[base + k];
            for (int e = 0; e < 3; ++e) {
                uint32_t idx = indices[3 * srcTri + e];
                out.indices.push_back(idx);
                math::Vec3 p = vpos(idx);
                cMin.x = std::min(cMin.x, p.x); cMax.x = std::max(cMax.x, p.x);
                cMin.y = std::min(cMin.y, p.y); cMax.y = std::max(cMax.y, p.y);
                cMin.z = std::min(cMin.z, p.z); cMax.z = std::max(cMax.z, p.z);
            }
        }
        m.boundMin = cMin;
        m.boundMax = cMax;
        m.boundCenter = (cMin + cMax) * 0.5f;
        // Conservative radius: max distance from the center to any referenced vertex (second pass over
        // the cluster's verts). Never under-bound.
        float r2max = 0.0f;
        for (uint32_t k = 0; k < count; ++k) {
            uint32_t srcTri = order[base + k];
            for (int e = 0; e < 3; ++e) {
                math::Vec3 d = vpos(indices[3 * srcTri + e]) - m.boundCenter;
                float r2 = math::dot(d, d);
                if (r2 > r2max) r2max = r2;
            }
        }
        m.boundRadius = std::sqrt(r2max);

        out.meshlets.push_back(m);
        emittedTris += count;
    }
    return out;
}

}  // namespace hf::render::vg
