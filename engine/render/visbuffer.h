#pragma once
// Slice DW — Virtual-Geometry Slice 1: Visibility-Buffer beachhead. Pure CPU (header-only, no device,
// no backend symbols). Namespace hf::render::vg (the same virtual-geometry namespace as meshlet.h /
// cluster_cull.h). Same header-only pattern as frustum.h / hiz.h / cluster_cull.h.
//
// WHAT THIS IS: the Nanite-style visibility buffer packs the IDENTITY of the front-most surface
// fragment — (clusterID, triangleID) — into a single screen-size INTEGER render target, DECOUPLING
// geometry rasterization from material shading (the deferred material resolve is the next slice DX).
// DW rasterizes the pair `(clusterID << kTriIdBits) | triID` into an R32_Uint RT via the existing
// DT/DV cluster MDI draw, reads it back, and proves the IDs are bit-exact + self-consistent + cover
// exactly the CPU cull's survivor set. The packed value is a FLAT integer (no interpolation / FP /
// blend), so it is inherently BIT-EXACT cross-vendor — the simplest possible proof surface.
//
// IDENTITY SPACE (DW convention): the "clusterID" packed here is the SURVIVOR-DRAW INDEX — the index
// of the survivor in the compacted CullClusterInstances command list (== gl_DrawID in the MDI draw),
// NOT the source (instance x cluster) index. This is exactly what the GPU vertex shader has cheaply
// available (gl_DrawID / PerDraw[gl_DrawID]) and is a dense [0, survivorCount) range, so packing
// stays tight. The mapping survivor-draw-index -> source cluster-instance is MdiCommand.firstInstance.
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/mtl::) symbols. The single additive RHI of this slice is
// Format::R32_Uint (rhi.h) + its two backend pixel-format mappings; this header touches none of it.

#include <cstdint>
#include <span>
#include <vector>

#include "math/math.h"
#include "render/cluster_cull.h"
#include "render/meshlet.h"

namespace hf::render::vg {

// triID bit budget. A cluster holds at most kMaxTrisPerCluster (128) triangles, so 7 bits index every
// triangle (0..127). static_assert pins that the budget actually covers the cluster triangle count.
static constexpr uint32_t kTriIdBits = 7;
static_assert((1u << kTriIdBits) >= kMaxTrisPerCluster,
              "kTriIdBits must index every triangle in a cluster (1<<kTriIdBits >= kMaxTrisPerCluster)");
static constexpr uint32_t kTriIdMask = (1u << kTriIdBits) - 1u;  // 0x7F

// The clear / background sentinel written to every texel no survivor covers. 0xFFFFFFFF can only be a
// VALID packed ID if clusterID == (0xFFFFFFFF >> 7) == 0x1FFFFFF AND triID == 0x7F — i.e. ~33.5M
// survivors, far beyond any feasible on-screen survivor count at this scale. The proofs additionally
// assert kVisBackground never equals a packed ID over the realized survivor range, so a collision is
// caught loudly rather than silently mistaken for coverage.
static constexpr uint32_t kVisBackground = 0xFFFFFFFFu;

// Pack a (clusterID = survivor-draw index, triID = SV_PrimitiveID within the cluster draw) pair into
// the 32-bit visibility ID: clusterID in the high bits, triID in the low kTriIdBits. Mirrors EXACTLY
// shaders/visbuffer.frag.hlsl's `(clusterID << 7) | (SV_PrimitiveID & 0x7F)` so the CPU reference and
// the GPU fragment agree bit-for-bit. triID is masked so an out-of-range triID can never bleed into
// the clusterID field (defensive; SV_PrimitiveID is always < triCount <= 128).
inline uint32_t PackVisId(uint32_t clusterID, uint32_t triID) {
    return (clusterID << kTriIdBits) | (triID & kTriIdMask);
}

// Inverse of PackVisId. UB-free for any 32-bit input (the background sentinel unpacks to its bit
// fields too; callers test against kVisBackground before unpacking a coverage ID).
inline void UnpackVisId(uint32_t v, uint32_t& clusterID, uint32_t& triID) {
    clusterID = v >> kTriIdBits;
    triID     = v & kTriIdMask;
}

// CPU COVERAGE REFERENCE (Proof B3 RHS) — INSTANCE PROVENANCE, the cross-vendor-robust oracle.
//
// A CLOSED convex mesh (the clustered sphere of the DT/DV showcase) occludes its own FAR hemisphere, and
// the Morton-coherent clusters are irregular surface patches that straddle the silhouette — so predicting
// the EXACT covering cluster per pixel is NOT cross-vendor-feasible (the scout's note). What IS rigorous:
// the front-most surface at a NON-OVERLAPPING instance's near pole must belong to THAT INSTANCE. The
// instances are spaced apart in X with no screen-space overlap, so at the pixel of instance i's nearest
// point to the camera, the GPU vis-buffer's clusterID — mapped back to its source cluster-instance via
// survivorCmds[cid].firstInstance — must name a cluster of INSTANCE i. This proves the visibility IDs
// carry the correct geometric PROVENANCE (not garbage, not a wrong instance) without a fragile per-cluster
// prediction. (The exact cluster IS additionally pinned bit-for-bit by B1's self-consistency + B2
// determinism + the read-back-equals-written guarantee.)
//
// `instanceNearPole` is the world point on the sphere nearest the camera: center + radius * normalize(
// cameraPos - center). Projected to a pixel, that pixel's front-most surface is unambiguously instance i.
struct InteriorSample {
    uint32_t expectedInstance;  // the SOURCE instance whose geometry must cover (px,py)
    uint32_t px;
    uint32_t py;
};

// One InteriorSample per VISIBLE instance (an instance with >=1 frustum survivor whose near pole projects
// in front of the camera + inside the viewport). Samples are in instance order. `instanceRadius` is the
// instance's world bounding radius (so the near pole sits on the surface, an interior point of the front
// cluster covering it).
inline std::vector<InteriorSample> InstanceInteriorSamples(
        std::span<const mdi::MdiCommand> survivorCmds,
        std::span<const ClusterInstance> clusters,
        std::span<const math::Vec3> instanceCenters,
        std::span<const float> instanceRadius,
        const math::Vec3& cameraPos,
        const math::Mat4& viewProj, uint32_t w, uint32_t h) {
    // Which source instances have a frustum survivor (so the GPU draws their geometry).
    std::vector<uint8_t> instanceHasSurvivor(instanceCenters.size(), 0);
    for (const mdi::MdiCommand& c : survivorCmds) {
        const uint32_t src = c.firstInstance;
        if (src < clusters.size() && clusters[src].instanceIndex < instanceHasSurvivor.size())
            instanceHasSurvivor[clusters[src].instanceIndex] = 1;
    }
    std::vector<InteriorSample> out;
    for (uint32_t inst = 0; inst < (uint32_t)instanceCenters.size(); ++inst) {
        if (!instanceHasSurvivor[inst]) continue;
        const math::Vec3 center = instanceCenters[inst];
        const float r = (inst < instanceRadius.size()) ? instanceRadius[inst] : 0.0f;
        math::Vec3 dir = math::normalize(cameraPos - center);
        math::Vec3 nearPole = center + dir * r;           // the surface point nearest the camera
        float clipW = 0.0f;
        math::Vec3 ndc = math::MulPointDivide(viewProj, nearPole, clipW);
        if (clipW <= 0.0f) continue;
        if (ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f) continue;
        // The engine's Perspective already applies the Vulkan Y-flip -> pixel = ((ndc+1)/2)*extent.
        int px = (int)((ndc.x * 0.5f + 0.5f) * (float)w);
        int py = (int)((ndc.y * 0.5f + 0.5f) * (float)h);
        if (px < 0 || px >= (int)w || py < 0 || py >= (int)h) continue;
        out.push_back(InteriorSample{inst, (uint32_t)px, (uint32_t)py});
    }
    return out;
}

}  // namespace hf::render::vg
