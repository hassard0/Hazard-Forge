#pragma once
// Slice DT — Virtual-Geometry Slice 2: GPU per-cluster frustum cull -> indirect cluster draw. Pure CPU
// (header-only, no device, no backend symbols). The per-CLUSTER analogue of the engine's per-OBJECT
// GPU cull (AR/CD): a DS cluster (engine/render/meshlet.h Meshlet) is just an object with a per-cluster
// bounding sphere, so this mirrors gpu_culled.h::CullAndCompact STRUCTURALLY over (instance x cluster)
// records. Namespace hf::render::vg (the same virtual-geometry namespace as meshlet.h).
//
// This header is the AUTHORITATIVE CPU reference for the EXACT same ordered-compaction + frustum-sphere
// logic shaders/cluster_cull.comp.hlsl runs on the GPU. It is shared by THREE call sites (like AR/CD):
//   1. tests/cluster_cull_test.cpp — pins BuildClusterInstances order + conservative bounds and
//      CullClusterInstances all-survive / none / half-cut == frustum.h subset IN SOURCE ORDER.
//   2. samples/hello_triangle/main.cpp (--cluster-cull-shot, Vulkan) — the CPU survivor count the GPU
//      draw-count is asserted against (the exact-count proof) + the CPU MdiCommand list the GPU-culled
//      image is asserted byte-identical to (render-invariance).
//   3. metal_headless/visual_test.mm (--cluster-cull, Metal) — same CPU reference on Metal.
//
// DETERMINISM (the crux of the golden): survivors are emitted in SOURCE-INDEX order. The GPU shader
// achieves this with a single-workgroup ORDERED prefix sum (the (instance x cluster) count is <=1024 =
// one workgroup), NOT an unordered atomicAdd append. This mirror walks the cluster-instances in source
// order and appends survivors, so the two orderings are identical by construction.
//
// BOUNDING SPHERE (must match the shader exactly): a cluster's world bounding sphere is
// render::gpu_cull::InstanceWorldSphere(instanceModel, meshlet.boundCenter, meshlet.boundRadius) — the
// SAME center = model*localCenter, radius = localRadius*|col0| (uniform scale) the AR cull uses — and the
// cull decision is the CONSERVATIVE render::frustum::SphereOutside test. The DS per-cluster boundRadius is
// conservative (never under-bounds), so this cull never drops a visible cluster (exactly like the
// per-object cull). A divergence surfaces loudly in the exact-count + byte-identical proofs.

#include <cstdint>
#include <span>
#include <vector>

#include "math/math.h"
#include "render/frustum.h"
#include "render/gpu_cull.h"
#include "render/meshlet.h"
#include "render/mdi.h"

namespace hf::render::vg {

// One (instance x cluster) record the per-cluster cull consumes: the cluster's index-buffer slice
// (triOffset/triCount, from the DS reordered index buffer) + its WORLD bounding sphere (the DS local
// per-cluster sphere transformed by the instance model matrix) + the owning instance index (so the
// vertex shader can fetch that instance's model matrix). instance-major, cluster-minor order. std430-
// friendly when uploaded as 2 float4 + 1 uint4 (see cluster_cull.comp.hlsl ClusterInstanceIn, 48 bytes).
struct ClusterInstance {
    uint32_t   triOffset = 0;     // first triangle of this cluster in the reordered index buffer (/3)
    uint32_t   triCount  = 0;     // triangles in this cluster (<= kMaxTrisPerCluster)
    math::Vec3 worldCenter{0, 0, 0}; // world bounding-sphere center (instanceModel * boundCenter)
    float      worldRadius = 0.0f;   // world bounding-sphere radius (boundRadius * |col0|)
    uint32_t   instanceIndex = 0;    // owning grid-instance index (firstInstance carries it to the shader)
};

// Build the FULL (instance x cluster) record list the cull consumes. For each instance i (in order), for
// each meshlet m of the shared DS decomposition (in order), push one ClusterInstance whose world sphere is
// render::gpu_cull::InstanceWorldSphere(instanceModels[i], m.boundCenter, m.boundRadius). Order is
// instance-major, cluster-minor (deterministic). Pure, no GPU. The result count is
// instanceModels.size() * mset.meshlets.size().
inline std::vector<ClusterInstance> BuildClusterInstances(std::span<const math::Mat4> instanceModels,
                                                          const MeshletSet& mset) {
    std::vector<ClusterInstance> out;
    out.reserve(instanceModels.size() * mset.meshlets.size());
    for (uint32_t i = 0; i < (uint32_t)instanceModels.size(); ++i) {
        const math::Mat4& model = instanceModels[i];
        for (const Meshlet& m : mset.meshlets) {
            ClusterInstance ci{};
            ci.triOffset = m.triOffset;
            ci.triCount  = m.triCount;
            gpu_cull::InstanceWorldSphere(model.m, m.boundCenter, m.boundRadius,
                                          ci.worldCenter, ci.worldRadius);
            ci.instanceIndex = i;
            out.push_back(ci);
        }
    }
    return out;
}

// CULL + ORDERED COMPACT (the mirror of shaders/cluster_cull.comp.hlsl). Walk `clusters` in SOURCE order;
// KEEP each cluster-instance whose world sphere is NOT fully outside the frustum (conservative
// render::frustum::SphereOutside), emitting in source order one MdiCommand per survivor:
//   { indexCount = triCount*3, instanceCount = 1, firstIndex = triOffset*3, vertexOffset = 0,
//     firstInstance = sourceClusterInstanceIndex }
// firstInstance carries the SOURCE cluster-instance index so the vertex shader fetches that
// cluster-instance's model matrix + hash color from an SSBO (gl_DrawID/firstInstance). The survivor order
// is the deterministic source order (matches the GPU prefix-sum compaction byte-for-byte). Pure, no GPU.
inline std::vector<mdi::MdiCommand> CullClusterInstances(std::span<const ClusterInstance> clusters,
                                                        const frustum::Frustum& f) {
    std::vector<mdi::MdiCommand> out;
    out.reserve(clusters.size());
    for (uint32_t i = 0; i < (uint32_t)clusters.size(); ++i) {
        const ClusterInstance& ci = clusters[i];
        if (frustum::SphereOutside(f, ci.worldCenter, ci.worldRadius)) continue;  // fully outside -> culled
        mdi::MdiCommand c{};
        c.indexCount    = ci.triCount * 3u;
        c.instanceCount = 1u;
        c.firstIndex    = ci.triOffset * 3u;
        c.vertexOffset  = 0u;
        c.firstInstance = i;            // SOURCE cluster-instance index (the per-draw fetch key)
        out.push_back(c);
    }
    return out;
}

// Convenience: just the survivor COUNT (the exact-count proof only needs the integer). Same scan.
inline uint32_t SurvivorClusterCount(std::span<const ClusterInstance> clusters,
                                     const frustum::Frustum& f) {
    uint32_t count = 0;
    for (const ClusterInstance& ci : clusters)
        if (!frustum::SphereOutside(f, ci.worldCenter, ci.worldRadius)) ++count;
    return count;
}

}  // namespace hf::render::vg
