#pragma once
// Slice DV — Virtual-Geometry Slice 4: discrete cluster-LOD selection by projected SCREEN-SPACE ERROR
// (the FINAL slice of the Nanite-style virtual-geometry CORE: DS decomposition -> DT frustum cull -> DU
// Hi-Z occlusion -> DV LOD). Pure CPU (header-only, no device, no backend symbols). Same header-only
// pattern as engine/render/meshlet.h / cluster_cull.h / frustum.h. Namespace hf::render::vg.
//
// CONCEPT: per instance, pick among kNumLods=3 PRE-BAKED discrete LOD meshes (different SphereGeometry
// tessellations, each DS-cluster-decomposed) by the projected screen-space error of the mesh's
// conservative object-space geometric error. Near geometry -> LOD0 (full detail, error 0); far geometry
// -> a coarser LOD (fewer, coarser clusters). The Nanite LOD primitive (screen-error-driven detail
// selection), kept DETERMINISTIC + bit-exact via pre-baked LODs (NO runtime simplifier — that is the
// deferred stretch). errorScale=0 (or forceLod0) selects LOD0 for every instance -> the disabled path,
// byte-identical to the full-detail render.
//
// AUTHORITATIVE CPU REFERENCE for shaders/cluster_lod_select.comp.hlsl: SelectLod (the squared-distance
// form) is copied VERBATIM into that shader so the SELECTED-LOD INTEGER is bit-exact CPU<->Vulkan<->Metal.
// Shared by THREE call sites:
//   1. tests/cluster_lod_test.cpp — pins BuildLodMeshes ranges/errors + SelectLod near/far/forceLod0/
//      monotonicity/threshold-boundary/clamp + ProjectionScaleForScreenError.
//   2. samples/hello_triangle/main.cpp (--cluster-lod-shot, Vulkan) — the CPU per-instance LOD the GPU
//      cluster_lod_select.comp output is asserted BIT-EXACT against (memcmp of the readback ints).
//   3. metal_headless/visual_test.mm (--cluster-lod, Metal) — same CPU reference (Metal renders the
//      selected set via the CPU bound path; the Vulkan path carries the GPU==CPU select proof).
//
// --- BIT-EXACT MAKE-OR-BREAK: the SQUARED-DISTANCE form (NO sqrt) ---
// The selected-LOD INTEGER must be bit-identical CPU<->GPU. A view-space distance `dist = length(...)`
// uses sqrt, which is NOT bit-identical CPU<->GPU (the DH lesson). So SelectLod NEVER takes a sqrt: it
// compares on the SQUARED view distance `dist2` directly.
//   projectedError[n] = geometricError[n] * projScale / dist        (the screen-space error, doc only)
//   allowed           = errorThreshold * errorScale                 (the per-frame screen-error budget)
// "LOD n is acceptable" == projectedError[n] <= allowed. With dist > 0, geometricError[n] >= 0 and
// allowed >= 0, multiply both sides by dist (positive, preserves the inequality):
//   geometricError[n] * projScale <= allowed * dist
// both sides are >= 0, so squaring preserves the inequality:
//   (geometricError[n] * projScale)^2 <= (allowed)^2 * dist2
// PURE multiply/compare, no sqrt, no division. The LHS lhs[n] = (geometricError[n]*projScale)^2 is a
// host scalar; the RHS rhs = allowed^2 * dist2 uses std::fma to accumulate dist2. We pick the COARSEST
// acceptable LOD: LOD0 (geometricError 0 -> lhs 0 <= rhs always) is the floor; walk n = 1..kNumLods-1
// in increasing-coarseness order and UPGRADE to n while n is acceptable. Clamp to [0, kNumLods-1].
// errorScale == 0 (or forceLod0) -> allowed = 0 -> rhs = 0; only LOD0 (lhs 0) is acceptable -> LOD0.
// The shader copy uses the IDENTICAL expression order (the same fma, the same compare direction).

#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include "math/math.h"
#include "render/meshlet.h"
#include "scene/mesh.h"
#include "scene/vertex.h"

namespace hf::render::vg {

// Number of pre-baked discrete LOD levels. 3 tessellations of the same mesh, LOD0 = finest (full detail).
static constexpr uint32_t kNumLods = 3;

// One discrete LOD level: a contiguous cluster range into the COMBINED MeshletSet (the concatenation of
// all LODs' clusters) + the LOD's CONSERVATIVE object-space geometric error (deviation from LOD0). LOD0
// has geometricError == 0 (full detail); coarser LODs have larger error + fewer clusters.
struct LodLevel {
    uint32_t firstCluster = 0;   // first cluster of this LOD in MeshletSet::meshlets
    uint32_t clusterCount = 0;   // clusters in this LOD
    float    geometricError = 0; // conservative object-space deviation from LOD0 (sphere sagitta; see below)
};

// The BuildLodMeshes output: ONE combined MeshletSet whose meshlets[] is the concatenation of each LOD's
// DS decomposition (LOD0 then LOD1 then LOD2), its indices[] the concatenation of each LOD's reordered
// index buffer (each LOD's cluster triOffsets already point into this combined buffer), plus the per-LOD
// cluster ranges + errors. One vertex buffer per LOD is NOT needed: the combined index buffer references
// the per-LOD vertices appended into one shared vertex array (vertexOffset handled below).
struct LodMeshes {
    MeshletSet                   combined;  // concatenated clusters + indices (all LODs, in LOD order)
    std::vector<scene::Vertex>   verts;     // concatenated vertices (all LODs); LOD n's indices are global
    std::array<LodLevel, kNumLods> lods{};
};

// The CONSERVATIVE geometric error of a SphereGeometry(segments, rings) tessellation: the sagitta (max
// chord deviation) of the coarsest arc, radius * (1 - cos(pi / min(segments, rings))). A coarser
// tessellation (fewer segments/rings) has a larger sagitta -> larger error. SphereGeometry uses unit
// radius (radius 1), so this is just (1 - cos(pi / min(seg,rings))). Documented + computed
// deterministically once on the host (a transcendental cos, host-only LOD metadata — never cross-backend
// GPU-shared, so a host cos is fine; only ProjectionScaleForScreenError's scalar crosses to the shader).
inline float SphereSagittaError(uint32_t segments, uint32_t rings) {
    const uint32_t n = (segments < rings) ? segments : rings;
    if (n == 0) return 0.0f;
    const float radius = 1.0f;  // SphereGeometry is unit-radius
    const float theta = 3.14159265358979323846f / static_cast<float>(n);  // the coarsest arc half-angle
    return radius * (1.0f - std::cos(theta));  // sagitta = radius*(1 - cos(pi/min(seg,rings)))
}

// Build the combined LOD mesh set from kNumLods tessellations (FINEST first; tess[0] is LOD0). For each
// LOD: append its vertices into the shared verts[] (recording the vertex base), DS-decompose its index
// buffer (offset by the vertex base so the combined index buffer is self-contained), append its clusters
// + reordered indices into `combined`, and record the LOD's cluster range + conservative geometric error
// (LOD0 -> 0, coarser -> the sphere sagitta). Pure, deterministic, no GPU.
//
// `geos[n]` is the n-th tessellation's geometry; `tess[n]` is its (segments, rings) pair (for the error).
inline LodMeshes BuildLodMeshes(const std::array<scene::MeshGeometry, kNumLods>& geos,
                                const std::array<std::pair<uint32_t, uint32_t>, kNumLods>& tess) {
    LodMeshes out;
    uint32_t clusterBase = 0;
    for (uint32_t n = 0; n < kNumLods; ++n) {
        const scene::MeshGeometry& geo = geos[n];
        const uint32_t vertBase = static_cast<uint32_t>(out.verts.size());

        // Append this LOD's vertices into the shared array.
        out.verts.insert(out.verts.end(), geo.verts.begin(), geo.verts.end());

        // Re-base this LOD's indices into the shared vertex array, then DS-decompose.
        std::vector<uint32_t> rebased(geo.indices.size());
        for (size_t k = 0; k < geo.indices.size(); ++k)
            rebased[k] = geo.indices[k] + vertBase;
        MeshletSet ms = BuildMeshlets(
            std::span<const scene::Vertex>(out.verts.data(), out.verts.size()),
            std::span<const uint32_t>(rebased.data(), rebased.size()));

        // Offset this LOD's cluster triOffsets so they index into the COMBINED index buffer, and append.
        const uint32_t triBase = static_cast<uint32_t>(out.combined.indices.size() / 3);
        for (Meshlet m : ms.meshlets) {
            m.triOffset += triBase;
            out.combined.meshlets.push_back(m);
        }
        out.combined.indices.insert(out.combined.indices.end(),
                                    ms.indices.begin(), ms.indices.end());

        // Record the LOD level: its cluster range + conservative error (LOD0 = 0; coarser = sagitta).
        LodLevel lvl;
        lvl.firstCluster = clusterBase;
        lvl.clusterCount = static_cast<uint32_t>(ms.meshlets.size());
        lvl.geometricError = (n == 0) ? 0.0f : SphereSagittaError(tess[n].first, tess[n].second);
        out.lods[n] = lvl;
        clusterBase += lvl.clusterCount;
    }
    return out;
}

// The HOST-PRECOMPUTED projection scalar for the screen-space error: screenH / (2 * tan(fovY/2)). The
// ONLY transcendental on the LOD path (tan), computed ONCE on the host and passed to BOTH the CPU
// SelectLod AND the shader as EXACT bits (so the per-instance math is pure multiply/compare). For a
// world-space deviation `e` at view distance `d`, e * projScale / d is the deviation projected to pixels.
inline float ProjectionScaleForScreenError(float fovYRadians, int screenH) {
    return static_cast<float>(screenH) / (2.0f * std::tan(fovYRadians * 0.5f));
}

// SQUARED view distance of `worldCenter` under the column-major `view` matrix: |(view * [c,1]).xyz|^2.
// std::fma accumulates each component squared (matches the shader's mad chain). Pure, no sqrt.
inline float ViewDistanceSquared(const math::Mat4& view, const math::Vec3& worldCenter) {
    const float* v = view.m;  // column-major: element(row,col) = v[col*4 + row]
    // view-space x,y,z (drop w; the view matrix is affine so w==1).
    float vx = std::fma(v[0], worldCenter.x, std::fma(v[4], worldCenter.y, std::fma(v[8],  worldCenter.z, v[12])));
    float vy = std::fma(v[1], worldCenter.x, std::fma(v[5], worldCenter.y, std::fma(v[9],  worldCenter.z, v[13])));
    float vz = std::fma(v[2], worldCenter.x, std::fma(v[6], worldCenter.y, std::fma(v[10], worldCenter.z, v[14])));
    return std::fma(vx, vx, std::fma(vy, vy, vz * vz));
}

// SELECT the discrete LOD for an instance whose world center is `worldCenter`, viewed under `view`, with
// the host-precomputed `projScale`, the per-LOD geometric errors `geometricError` (kNumLods entries;
// [0] == 0), the per-frame `errorThreshold` (the screen-error budget in projScale units) and `errorScale`
// (a dimensionless multiplier; 0 -> forceLod0). Returns the COARSEST LOD whose projected screen-space
// error <= errorThreshold*errorScale, via the SQUARED-DISTANCE form (see the header banner) — pure
// multiply/compare, no sqrt, bit-exact CPU<->GPU. Result clamped to [0, kNumLods-1].
inline uint32_t SelectLod(std::span<const float> geometricError, const math::Vec3& worldCenter,
                          const math::Mat4& view, float projScale, float errorThreshold,
                          float errorScale, bool forceLod0) {
    if (forceLod0 || errorScale == 0.0f) return 0u;  // disabled path: always LOD0 (full detail)

    const float dist2 = ViewDistanceSquared(view, worldCenter);
    const float allowed = errorThreshold * errorScale;  // the screen-error budget (>= 0)
    const float rhs = std::fma(allowed * allowed, dist2, 0.0f);  // (allowed)^2 * dist2 (the RHS, >= 0)

    // LOD0 (geometricError 0 -> lhs 0 <= rhs) is always acceptable; UPGRADE to a coarser LOD while it is
    // still acceptable (projectedError <= allowed, in the squared form). Walk increasing coarseness.
    uint32_t lod = 0u;
    for (uint32_t n = 1; n < kNumLods; ++n) {
        const float ge = geometricError[n] * projScale;  // geometricError[n] * projScale
        const float lhs = ge * ge;                       // (geometricError[n]*projScale)^2 (the LHS, >= 0)
        if (lhs <= rhs) lod = n;  // LOD n acceptable -> upgrade to the coarser level
        // (errors are non-decreasing in n, so once a LOD is rejected, coarser ones could still be
        //  acceptable only if their error were SMALLER — which it is not; but we keep the simple scan to
        //  mirror the shader VERBATIM, and the test pins monotonicity of the result.)
    }
    if (lod >= kNumLods) lod = kNumLods - 1u;  // clamp [0, kNumLods-1]
    return lod;
}

}  // namespace hf::render::vg
