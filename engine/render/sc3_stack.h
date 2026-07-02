#pragma once
// Slice SC3 — SPONZA THROUGH THE VIRTUAL-GEOMETRY STACK (docs/GAP_CLOSING_ROADMAP.md Tier 2). Pure
// CPU (header-only, no device, no backend symbols), namespace hf::render::vg — the same header-only
// pattern as meshlet.h / cluster_cull.h / lod_gen.h, which this header COMPOSES and does NOT modify.
//
// WHY: the whole virtual-geometry machinery (DS meshlet decomposition, DT per-cluster frustum cull,
// LOD1 automatic QEM decimation) had only ever run on SYNTHETIC fixtures (spheres <= 12k triangles).
// SC1 made the REAL Khronos PBR Sponza (103 meshes / 262k triangles of heterogeneous authored
// geometry: thin curtains, foliage cards, high-valence trim) load + render. SC3 pushes that REAL
// content through meshlet -> cluster-cull -> auto-LOD and PROVES (or honestly reports where) the
// pipeline holds at real-content scale:
//
//   1. Sc3DecomposeAll — BuildMeshlets over EVERY mesh; per-mesh + aggregate stats; the meshlet_test
//      PARTITION-COMPLETENESS invariant (every triangle exactly once — the reordered triangle
//      multiset == the original) applied to all meshes; honest per-mesh outlier reporting
//      (degenerate triangles, boundary edges, non-manifold edges — real assets have them; the
//      decomposition must carry them through VERBATIM, never drop/duplicate); and a DETERMINISTIC
//      integer digest (net::DigestBytes over the per-mesh cluster tables + reordered index buffers —
//      all uint32, so the digest is a cross-platform pin).
//   2. Sc3BuildAndCull — the DT cluster cull at (instance x cluster) scale over the REAL scene:
//      per-instance ClusterInstance records built by the byte-untouched BuildClusterInstances
//      (instance-major, cluster-minor — deterministic), culled + source-order-compacted by the
//      byte-untouched CullClusterInstances. firstInstance carries the GLOBAL record index.
//   3. Sc3CheckCullSoundness — a SOUNDNESS check on the real scene: any cluster with a WITNESS
//      vertex strictly inside the frustum (all six signed distances > eps) MUST be a survivor.
//      A vertex inside the frustum is a SUFFICIENT visibility witness (a cluster can also be visible
//      with all vertices outside — an edge crossing the frustum — so this is a NECESSARY-condition
//      check on the cull, not an exact-visibility oracle; the conservative sphere test keeps such
//      clusters anyway). violations MUST be 0: the conservative-bounds contract at scale.
//   4. Sc3BuildTopLods — LOD1's BuildAutoLods (integer QEM, boundary-preserving: collapses touching
//      a boundary vertex are forbidden outright) on the N largest meshes, 50%/25% targets. Real
//      open meshes (walls, curtains, foliage cards) are BOUNDARY-DOMINATED, so QEM v1 may run out
//      of legal collapses ABOVE the target — hit50/hit25 report that honestly per mesh (a real
//      finding about QEM v1 on real content, not a failure to hide).
//
// Shared by THREE call sites (the established vg discipline):
//   1. tests/sc3_stack_test.cpp — the committed-asset (DamagedHelmet/CesiumMilkTruck) always-on run
//      + the Sponza-gated at-scale run (skips gracefully when the fetched asset is absent).
//   2. samples/hello_triangle/main.cpp (--sc3-stack-shot, Vulkan BMP) — survivors-only cluster-color
//      render of Sponza from the SC1 hero camera + the pinned stat line.
//   3. metal_headless/visual_test.mm (--sc3-stack-shot, Metal PNG) — the same pipeline + render.
//
// SEAM DISCIPLINE: ZERO backend symbols. meshlet.h / cluster_cull.h / cluster_lod.h / lod_gen.h are
// byte-UNTOUCHED — SC3 is pure composition over their public entry points.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "math/math.h"
#include "net/session.h"        // hf::net::DigestBytes (FNV-1a-64, the pinned-golden currency)
#include "render/cluster_cull.h"
#include "render/frustum.h"
#include "render/lod_gen.h"
#include "render/meshlet.h"
#include "scene/mesh.h"
#include "scene/vertex.h"

namespace hf::render::vg {

// --- The SC3 input view: one CPU mesh (non-owning spans) + one placed instance. ---------------------
// Callers adapt whatever they loaded (asset::CpuScene, procedural geometry) into these views; SC3
// never owns geometry.
struct Sc3Mesh {
    std::span<const scene::Vertex> verts;
    std::span<const uint32_t>      indices;
};
struct Sc3Instance {
    uint32_t   meshIndex = 0;
    math::Mat4 world = math::Mat4::Identity();
};

// --- Per-mesh decomposition stats: the honest at-scale report. --------------------------------------
struct Sc3MeshStats {
    uint32_t tris = 0;               // whole triangles (indices/3, trailing 1-2 ignored — meshlet rule)
    uint32_t clusters = 0;
    uint32_t maxTrisPerCluster = 0;  // max observed triCount (must be <= kMaxTrisPerCluster)
    bool     indicesValid = true;    // every index of every whole triangle < verts.size()
    bool     complete = false;       // reordered triangle multiset == original (every tri exactly once)
    // Outliers (REPORTED, not repaired: the decomposition carries them through verbatim; these are
    // properties of the INPUT geometry — real content has them, synthetic fixtures did not):
    uint32_t degenerateTris = 0;     // repeated-index or exactly-zero float-cross-product triangles
    uint32_t boundaryEdges = 0;      // undirected edges with face-incidence 1 (OPEN meshes: rims)
    uint32_t nonManifoldEdges = 0;   // undirected edges with face-incidence > 2 (true non-manifold)
};

// --- The at-scale decomposition result. --------------------------------------------------------------
struct Sc3Decomposition {
    std::vector<MeshletSet>   sets;   // one per input mesh (empty set for invalid/empty meshes)
    std::vector<Sc3MeshStats> stats;  // parallel to sets
    uint64_t totalTris = 0;
    uint32_t totalClusters = 0;
    uint32_t maxTrisPerCluster = 0;   // max over all meshes
    uint32_t incompleteMeshes = 0;    // meshes failing the completeness invariant (MUST be 0)
    uint32_t invalidMeshes = 0;       // meshes with out-of-range indices (skipped, reported)
    uint64_t degenerateTris = 0;      // aggregates over all meshes
    uint64_t boundaryEdges = 0;
    uint64_t nonManifoldEdges = 0;
    uint64_t digest = 0;              // FNV-1a-64 over the full decomposition (integers only)
};

// Decompose EVERY mesh with the byte-untouched BuildMeshlets; verify the meshlet_test completeness
// invariant per mesh; gather the honest outlier stats; fold the whole decomposition into ONE
// deterministic integer digest. The digest stream is, per mesh (in order):
//   u32 meshIndex, u32 T, u32 clusterCount, then per cluster {u32 triOffset, u32 triCount}, then the
//   full reordered index buffer (u32 each) — all little-endian, all integers (the reordered index
//   buffer is the decomposition's complete integer fingerprint; cluster BOUNDS are float and stay
//   out of the digest so the pin is bit-stable everywhere).
inline Sc3Decomposition Sc3DecomposeAll(std::span<const Sc3Mesh> meshes) {
    Sc3Decomposition out;
    out.sets.reserve(meshes.size());
    out.stats.reserve(meshes.size());

    std::vector<unsigned char> stream;
    auto putU32 = [&](uint32_t v) {
        stream.push_back((unsigned char)(v & 0xFFu));
        stream.push_back((unsigned char)((v >> 8) & 0xFFu));
        stream.push_back((unsigned char)((v >> 16) & 0xFFu));
        stream.push_back((unsigned char)((v >> 24) & 0xFFu));
    };

    for (uint32_t mi = 0; mi < (uint32_t)meshes.size(); ++mi) {
        const Sc3Mesh& mesh = meshes[mi];
        Sc3MeshStats st;
        const uint32_t T = (uint32_t)(mesh.indices.size() / 3);
        st.tris = T;

        // Index validity over the whole triangles (BuildMeshlets reads verts[idx] unchecked; a real
        // asset should never trip this, but at-scale honesty means we CHECK, skip and REPORT).
        for (uint32_t k = 0; k < T * 3u; ++k)
            if (mesh.indices[k] >= mesh.verts.size()) { st.indicesValid = false; break; }
        if (!st.indicesValid) {
            ++out.invalidMeshes;
            out.sets.emplace_back();
            out.stats.push_back(st);
            continue;
        }

        MeshletSet ms = BuildMeshlets(mesh.verts, mesh.indices);
        st.clusters = (uint32_t)ms.meshlets.size();
        for (const Meshlet& m : ms.meshlets)
            if (m.triCount > st.maxTrisPerCluster) st.maxTrisPerCluster = m.triCount;

        // --- Completeness (the meshlet_test invariant): reordered triangle multiset == original. ---
        {
            auto sortedTris = [](std::span<const uint32_t> idx, uint32_t nTris) {
                std::vector<std::array<uint32_t, 3>> tris;
                tris.reserve(nTris);
                for (uint32_t t = 0; t < nTris; ++t)
                    tris.push_back({idx[3 * t + 0], idx[3 * t + 1], idx[3 * t + 2]});
                std::sort(tris.begin(), tris.end());
                return tris;
            };
            st.complete = (ms.indices.size() == (size_t)T * 3) &&
                          (sortedTris(mesh.indices, T) ==
                           sortedTris(std::span<const uint32_t>(ms.indices.data(),
                                                                ms.indices.size()), T));
        }
        if (!st.complete) ++out.incompleteMeshes;

        // --- Outliers: degenerate triangles + edge-incidence histogram (reported, never repaired). ---
        {
            std::vector<uint64_t> ekeys;
            ekeys.reserve((size_t)T * 3);
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t i0 = mesh.indices[3 * t + 0];
                const uint32_t i1 = mesh.indices[3 * t + 1];
                const uint32_t i2 = mesh.indices[3 * t + 2];
                bool degen = (i0 == i1 || i1 == i2 || i0 == i2);
                if (!degen) {
                    // Exactly-zero float cross product == geometrically-degenerate (zero-area) tri.
                    const float* p0 = mesh.verts[i0].pos;
                    const float* p1 = mesh.verts[i1].pos;
                    const float* p2 = mesh.verts[i2].pos;
                    const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
                    const float e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
                    const float cx = e1[1] * e2[2] - e1[2] * e2[1];
                    const float cy = e1[2] * e2[0] - e1[0] * e2[2];
                    const float cz = e1[0] * e2[1] - e1[1] * e2[0];
                    if (cx == 0.0f && cy == 0.0f && cz == 0.0f) degen = true;
                }
                if (degen) ++st.degenerateTris;
                // Edge keys over every whole triangle (degenerate self-edges x==y included: they show
                // up as odd-incidence edges, which is exactly the outlier signal we want to surface).
                for (int e = 0; e < 3; ++e) {
                    uint32_t x = mesh.indices[3 * t + e], y = mesh.indices[3 * t + (e + 1) % 3];
                    if (x > y) std::swap(x, y);
                    ekeys.push_back(((uint64_t)x << 32) | y);
                }
            }
            std::sort(ekeys.begin(), ekeys.end());
            for (size_t i = 0; i < ekeys.size();) {
                size_t j = i;
                while (j < ekeys.size() && ekeys[j] == ekeys[i]) ++j;
                const size_t inc = j - i;
                if (inc == 1) ++st.boundaryEdges;
                else if (inc > 2) ++st.nonManifoldEdges;
                i = j;
            }
        }

        // --- Fold this mesh into the digest stream. ---
        putU32(mi);
        putU32(T);
        putU32(st.clusters);
        for (const Meshlet& m : ms.meshlets) { putU32(m.triOffset); putU32(m.triCount); }
        for (uint32_t idx : ms.indices) putU32(idx);

        // --- Aggregates. ---
        out.totalTris += T;
        out.totalClusters += st.clusters;
        if (st.maxTrisPerCluster > out.maxTrisPerCluster)
            out.maxTrisPerCluster = st.maxTrisPerCluster;
        out.degenerateTris += st.degenerateTris;
        out.boundaryEdges += st.boundaryEdges;
        out.nonManifoldEdges += st.nonManifoldEdges;

        out.sets.push_back(std::move(ms));
        out.stats.push_back(st);
    }

    out.digest = net::DigestBytes(stream.data(), stream.size());
    return out;
}

// --- The at-scale cull result: the full (instance x cluster) record list + the survivors. -----------
// records[k] is the k-th global cluster-instance (instance-major, cluster-minor — the deterministic
// BuildClusterInstances order per instance). recordMesh/recordCluster are parallel: the owning mesh
// index (for draw-time vertex/index-buffer binding) and the cluster's index within its mesh's
// MeshletSet. survivors is the CullClusterInstances source-order compaction; each MdiCommand's
// firstInstance is the GLOBAL record index (the per-draw fetch key).
struct Sc3CullResult {
    std::vector<ClusterInstance>  records;
    std::vector<uint32_t>         recordMesh;
    std::vector<uint32_t>         recordCluster;
    std::vector<mdi::MdiCommand>  survivors;
    uint32_t total = 0;
    uint32_t survivorCount = 0;
    uint32_t culledCount = 0;
};

// Build the FULL (instance x cluster) record list over the heterogeneous scene and cull it. Reuses
// the byte-untouched BuildClusterInstances per instance (a one-element model span; the returned
// records' instanceIndex is then rebased to the GLOBAL instance index so the draw loop can fetch the
// owning instance's world transform) and the byte-untouched CullClusterInstances over the whole list.
inline Sc3CullResult Sc3BuildAndCull(const Sc3Decomposition& dec,
                                     std::span<const Sc3Instance> instances,
                                     const frustum::Frustum& f) {
    Sc3CullResult out;
    for (uint32_t ii = 0; ii < (uint32_t)instances.size(); ++ii) {
        const Sc3Instance& inst = instances[ii];
        if (inst.meshIndex >= dec.sets.size()) continue;  // bounds-checked (skip foreign instances)
        const MeshletSet& ms = dec.sets[inst.meshIndex];
        if (ms.meshlets.empty()) continue;
        std::vector<ClusterInstance> recs = BuildClusterInstances(
            std::span<const math::Mat4>(&inst.world, 1), ms);
        for (uint32_t c = 0; c < (uint32_t)recs.size(); ++c) {
            recs[c].instanceIndex = ii;  // rebase 0 (the one-element span) -> the global instance
            out.recordMesh.push_back(inst.meshIndex);
            out.recordCluster.push_back(c);
            out.records.push_back(recs[c]);
        }
    }
    out.total = (uint32_t)out.records.size();
    out.survivors = CullClusterInstances(
        std::span<const ClusterInstance>(out.records.data(), out.records.size()), f);
    out.survivorCount = (uint32_t)out.survivors.size();
    out.culledCount = out.total - out.survivorCount;
    return out;
}

// --- Cull soundness at scale (see the header banner, point 3). ---------------------------------------
// witnessClusters = records with at least one triangle vertex STRICTLY inside the frustum (all six
// signed distances > kSc3WitnessEps — the epsilon keeps float-boundary vertices out of the witness
// set, since a sphere-vs-plane and a point-vs-plane evaluation of the same boundary point can round
// differently). violations = witness records ABSENT from the survivor set — MUST be 0.
static constexpr float kSc3WitnessEps = 1e-4f;
struct Sc3Soundness {
    uint64_t witnessClusters = 0;
    uint64_t violations = 0;
};
inline Sc3Soundness Sc3CheckCullSoundness(std::span<const Sc3Mesh> meshes,
                                          const Sc3Decomposition& dec,
                                          std::span<const Sc3Instance> instances,
                                          const Sc3CullResult& cull,
                                          const frustum::Frustum& f) {
    Sc3Soundness out;
    std::vector<uint8_t> survived(cull.records.size(), 0);
    for (const mdi::MdiCommand& c : cull.survivors)
        if (c.firstInstance < survived.size()) survived[c.firstInstance] = 1;

    for (uint32_t r = 0; r < (uint32_t)cull.records.size(); ++r) {
        const uint32_t meshIdx = cull.recordMesh[r];
        const uint32_t clusterIdx = cull.recordCluster[r];
        const uint32_t instIdx = cull.records[r].instanceIndex;
        if (meshIdx >= meshes.size() || instIdx >= instances.size()) continue;
        const MeshletSet& ms = dec.sets[meshIdx];
        if (clusterIdx >= ms.meshlets.size()) continue;
        const Meshlet& m = ms.meshlets[clusterIdx];
        const Sc3Mesh& mesh = meshes[meshIdx];
        const math::Mat4& w = instances[instIdx].world;

        bool witness = false;
        for (uint32_t t = 0; t < m.triCount && !witness; ++t)
            for (int e = 0; e < 3 && !witness; ++e) {
                const uint32_t idx = ms.indices[3 * (m.triOffset + t) + e];
                const float* p = mesh.verts[idx].pos;
                // world * [p,1] (column-major: element(row,col) == m[col*4+row]).
                const math::Vec3 wp{
                    w.m[0] * p[0] + w.m[4] * p[1] + w.m[8] * p[2]  + w.m[12],
                    w.m[1] * p[0] + w.m[5] * p[1] + w.m[9] * p[2]  + w.m[13],
                    w.m[2] * p[0] + w.m[6] * p[1] + w.m[10] * p[2] + w.m[14]};
                bool inside = true;
                for (int pl = 0; pl < 6; ++pl)
                    if (frustum::SignedDistance(f.planes[pl], wp) <= kSc3WitnessEps) {
                        inside = false;
                        break;
                    }
                if (inside) witness = true;
            }
        if (witness) {
            ++out.witnessClusters;
            if (!survived[r]) ++out.violations;
        }
    }
    return out;
}

// --- Auto-LOD at scale: LOD1's BuildAutoLods on the N largest meshes. --------------------------------
// triCount[0] is the QUANTIZED WELDED input count (the decimation currency — see lod_gen.h; rawTris -
// triCount[0] == the lattice-degenerate weldDropped). hit50/hit25 report whether the boundary-
// preserving QEM actually REACHED the 50%/25% targets (targets are triCount[0]/2 and /4 — the
// BuildAutoLods integer targets); boundary-dominated open meshes may honestly stall above them.
// geometricError is lod_gen's conservative object-space heuristic bound (non-decreasing);
// maxQuadricError the integer QEM quality pin. invalidTris counts output triangles with out-of-range
// or repeated indices (MUST be 0 — DecimateMesh guards this by construction; we verify, not trust).
struct Sc3LodEntry {
    uint32_t meshIndex = 0;
    uint32_t rawTris = 0;
    uint32_t weldDropped = 0;
    std::array<uint32_t, kNumLods> triCount{};
    std::array<uint64_t, kNumLods> digest{};
    std::array<float, kNumLods>    geometricError{};
    std::array<uint64_t, kNumLods> maxQuadricError{};
    bool hit50 = false;
    bool hit25 = false;
    uint32_t invalidTris = 0;
};
inline std::vector<Sc3LodEntry> Sc3BuildTopLods(std::span<const Sc3Mesh> meshes, uint32_t topN) {
    // Rank meshes by whole-triangle count, descending; mesh index ascending breaks ties
    // (deterministic).
    std::vector<uint32_t> order;
    order.reserve(meshes.size());
    for (uint32_t i = 0; i < (uint32_t)meshes.size(); ++i)
        if (meshes[i].indices.size() / 3 > 0) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        const size_t ta = meshes[a].indices.size() / 3, tb = meshes[b].indices.size() / 3;
        if (ta != tb) return ta > tb;
        return a < b;
    });
    if (order.size() > topN) order.resize(topN);

    std::vector<Sc3LodEntry> out;
    out.reserve(order.size());
    for (uint32_t mi : order) {
        const Sc3Mesh& mesh = meshes[mi];
        scene::MeshGeometry g;
        g.verts.assign(mesh.verts.begin(), mesh.verts.end());
        g.indices.assign(mesh.indices.begin(), mesh.indices.end());
        const AutoLods al = BuildAutoLods(g);

        Sc3LodEntry e;
        e.meshIndex = mi;
        e.rawTris = (uint32_t)(mesh.indices.size() / 3);
        e.triCount = al.triCount;
        e.digest = al.digest;
        e.geometricError = al.geometricError;
        e.maxQuadricError = al.maxQuadricError;
        e.weldDropped = e.rawTris - al.triCount[0];
        e.hit50 = (al.triCount[1] <= al.triCount[0] / 2);
        e.hit25 = (al.triCount[2] <= al.triCount[0] / 4);
        for (uint32_t n = 1; n < kNumLods; ++n) {
            const scene::MeshGeometry& lg = al.geos[n];
            const size_t nv = lg.verts.size();
            for (size_t t = 0; t + 3 <= lg.indices.size(); t += 3) {
                const uint32_t i0 = lg.indices[t], i1 = lg.indices[t + 1], i2 = lg.indices[t + 2];
                if (i0 >= nv || i1 >= nv || i2 >= nv || i0 == i1 || i1 == i2 || i0 == i2)
                    ++e.invalidTris;
            }
        }
        out.push_back(e);
    }
    return out;
}

}  // namespace hf::render::vg
