#pragma once
// Slice CL — Clustered Light Culling (Forward+). Pure CPU (header-only, no device, no backend
// symbols). Same pattern as engine/render/frustum.h / hiz.h / gpu_culled.h. Namespace
// hf::render::cluster. Shared by the --clustered-lights-shot showcase, the cluster-assign compute
// (shaders/cluster_assign.comp.hlsl runs the SAME ClusterViewAABB + SphereAABBIntersect math) AND
// tests/cluster_test.cpp — so the unit test exercises the EXACT assignment the render path uses.
//
// (NOTE: this is a DISTINCT slice from the earlier Slice AG `clustered.h` / lit_clustered.frag.hlsl /
//  --clustered-shot. AG culls CPU-side with a SMOOTH falloff; THIS slice (CL) culls on the GPU in a
//  compute pass with a HARD-WINDOWED radius and proves the clustered shade is BYTE-IDENTICAL to a
//  brute-force all-lights render. Different namespace (cluster vs clustered), different shaders, and a
//  separate golden (clustered_lights.png) so AG's golden + path are untouched.)
//
// THE TECHNIQUE: partition the camera view frustum into a dimX*dimY*dimZ grid of clusters.
//   * XY follow the framebuffer's screen tiles (dimX columns, dimY rows).
//   * Z is split into dimZ EXPONENTIAL depth slices between zNear..zFar, so near clusters are thin and
//     far clusters deep (the standard cluster depth distribution that matches perspective).
// A compute pass assigns each point light to the clusters its (hard-radius) sphere of influence
// touches; the clustered lit fragment computes its own cluster index and iterates ONLY that cluster's
// light list.
//
// RENDER-INVARIANCE (the proof that makes it golden-safe): point lights use a WINDOWED attenuation
// with a HARD cutoff — a light contributes EXACTLY ZERO beyond `radius`. A cluster is a view-space
// AABB; a light is assigned to it IFF SphereAABBIntersect(lightPosView, radius, min, max). If the
// sphere misses the cluster AABB the light is farther than `radius` from EVERY point in the cluster ->
// zero contribution to every pixel there. So iterating only the cluster's assigned lights yields the
// IDENTICAL sum as iterating ALL lights. Byte-identical. AssignLights here is the CPU reference the
// showcase asserts the GPU assignment-total matches.
//
// CONVENTIONS (must match engine/math + the shaders EXACTLY):
//   * Mat4 is column-major; Mat4::Perspective produces Vulkan clip space (depth [0,1], Y-flip baked
//     into m[5] = -1/tan(fovY/2)). View space is RIGHT-HANDED, looking down -Z, so a point in front of
//     the camera has view-space z < 0. We use a POSITIVE view-distance vz = -viewPos.z for the slices.
//   * Cluster flat index: idx = cx + cy*dimX + cz*(dimX*dimY).
//   * Exponential Z slice boundary k in [0,dimZ]: SliceZ(k) = zNear * (zFar/zNear)^(k/dimZ).
//     The cluster (.,.,cz) spans view-distance [SliceZ(cz), SliceZ(cz+1)] -> view z in [-far,-near].
//   * Screen tile XY -> view-space XY: unproject the tile's NDC corners through invProj at the slice's
//     near & far view-distances and take the min/max over all 8 corners (a conservative view-space
//     AABB containing the cluster's frustum sub-volume). invProj == proj.Inverse().

#include "math/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace hf::render::cluster {

// The cluster grid: dimensions + the view depth range. (e.g. 16x9x24 over [0.5, 90].)
struct ClusterGrid {
    int   dimX = 16, dimY = 9, dimZ = 24;
    float zNear = 0.5f, zFar = 90.0f;

    int clusterCount() const { return dimX * dimY * dimZ; }
    int flatIndex(int cx, int cy, int cz) const { return cx + cy * dimX + cz * (dimX * dimY); }
};

// A point light as the CPU culler + the GPU `lights` SSBO see it. std430 packing (the shader struct
// is two float4): { posWorld.xyz, radius } as the first float4; { color.rgb, intensity } as the
// second. 32 bytes. posWorld is WORLD space; the assignment transforms it into view space with `view`
// (so a single light record serves both world-space shading and view-space culling).
struct PointLight {
    math::Vec3 posWorld{0, 0, 0};   // float4.xyz
    float      radius = 1.0f;       // float4.w   (hard cutoff: zero contribution beyond this)
    math::Vec3 color{1, 1, 1};      // float4.xyz
    float      intensity = 1.0f;    // float4.w
};

// Exponential z-slice boundary k in [0,dimZ]: positive view-space distance of slice plane k.
// SliceZ(0)==zNear, SliceZ(dimZ)==zFar. MIRRORED in the shaders.
inline float SliceZ(const ClusterGrid& g, int k) {
    return g.zNear * std::pow(g.zFar / g.zNear, (float)k / (float)g.dimZ);
}

// Positive view-distance -> z-slice index (inverse of SliceZ). MIRRORED in lit_clustered_cl.frag.
inline int SliceForViewZ(const ClusterGrid& g, float viewZ) {
    if (viewZ <= g.zNear) return 0;
    if (viewZ >= g.zFar)  return g.dimZ - 1;
    float t = std::log(viewZ / g.zNear) / std::log(g.zFar / g.zNear);  // [0,1)
    int s = (int)std::floor(t * (float)g.dimZ);
    return s < 0 ? 0 : (s > g.dimZ - 1 ? g.dimZ - 1 : s);
}

// Unproject an NDC point (ndcX,ndcY in [-1,1], post-divide) at a given POSITIVE view-distance vz to
// view space, using invProj == proj.Inverse(). The perspective proj maps view -> clip; for a point at
// view-z = -vz, invProj * (ndc, ndcZ, 1)*clipW recovers the view position. Because the cluster only
// needs the XY extents at a known vz, we form the NDC ray through invProj and scale it to the plane
// z = -vz. Standard cluster-grid corner unprojection.
inline math::Vec3 UnprojectToView(const math::Mat4& invProj, float ndcX, float ndcY, float vz) {
    // Take the NDC point at z=0 (any z works since we re-scale to the target plane); homogeneous w=1.
    float w = 0.0f;
    math::Vec3 vDir = math::MulPointDivide(invProj, math::Vec3{ndcX, ndcY, 0.0f}, w);
    // vDir is a view-space point on the ray; the ray passes through the origin (the eye), so the view
    // position at distance such that view-z == -vz is vDir * (-vz / vDir.z). vDir.z is negative.
    float s = (std::fabs(vDir.z) > 1e-9f) ? (-vz / vDir.z) : 0.0f;
    return math::Vec3{vDir.x * s, vDir.y * s, -vz};
}

// The VIEW-SPACE AABB of cluster (cx,cy,cz): unproject the cluster's screen-tile NDC corners at the
// slice's near & far view-distances and take the min/max over all 8 corners. invProj == proj.Inverse().
// Right-handed view space, -Z forward: outMin.z = -SliceZ(cz+1) (far), outMax.z = -SliceZ(cz) (near).
inline void ClusterViewAABB(const ClusterGrid& g, int cx, int cy, int cz, const math::Mat4& invProj,
                            int /*screenW*/, int /*screenH*/, math::Vec3& outMin, math::Vec3& outMax) {
    float zNearK = SliceZ(g, cz);       // positive near-plane view-distance of this slice
    float zFarK  = SliceZ(g, cz + 1);   // positive far-plane view-distance
    // The tile's NDC rectangle. NDC x grows right; NDC y in [-1,1]. (The Y-flip baked in proj is
    // undone by invProj, so unprojecting NDC corners gives the correct view-space XY regardless of the
    // backend's clip-space Y convention — the AABB is identical on Vulkan + Metal.)
    float nx0 = ((float)cx / (float)g.dimX) * 2.0f - 1.0f;
    float nx1 = ((float)(cx + 1) / (float)g.dimX) * 2.0f - 1.0f;
    float ny0 = ((float)cy / (float)g.dimY) * 2.0f - 1.0f;
    float ny1 = ((float)(cy + 1) / (float)g.dimY) * 2.0f - 1.0f;

    float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
    const float zs[2]  = {zNearK, zFarK};
    const float nxs[2] = {nx0, nx1};
    const float nys[2] = {ny0, ny1};
    for (float vz : zs)
        for (float nx : nxs)
            for (float ny : nys) {
                math::Vec3 v = UnprojectToView(invProj, nx, ny, vz);
                minX = std::min(minX, v.x); maxX = std::max(maxX, v.x);
                minY = std::min(minY, v.y); maxY = std::max(maxY, v.y);
            }
    outMin = math::Vec3{minX, minY, -zFarK};   // view z negative: [-far, -near]
    outMax = math::Vec3{maxX, maxY, -zNearK};
}

// Standard closest-point sphere/AABB intersection: distance²(clamp(center,min,max), center) <=
// radius². Exact; shared with the shader.
inline bool SphereAABBIntersect(const math::Vec3& center, float radius,
                                const math::Vec3& aabbMin, const math::Vec3& aabbMax) {
    float dx = center.x < aabbMin.x ? aabbMin.x - center.x
             : (center.x > aabbMax.x ? center.x - aabbMax.x : 0.0f);
    float dy = center.y < aabbMin.y ? aabbMin.y - center.y
             : (center.y > aabbMax.y ? center.y - aabbMax.y : 0.0f);
    float dz = center.z < aabbMin.z ? aabbMin.z - center.z
             : (center.z > aabbMax.z ? center.z - aabbMax.z : 0.0f);
    float d2 = dx * dx + dy * dy + dz * dz;
    return d2 <= radius * radius;
}

// CPU REFERENCE assignment: for each cluster, the list of light indices whose VIEW-space sphere
// intersects the cluster's view-space AABB. Deterministic + ORDERED: clusters iterated cx-major, and
// within a cluster lights are appended in ascending light index (identical order to the GPU's ordered
// fill, so the per-cluster lists match exactly). outPerCluster is sized clusterCount(); entry idx ==
// flatIndex(cx,cy,cz). `proj`/`view` are the camera matrices; lights carry WORLD positions (this
// transforms each into view space once for culling).
inline void AssignLights(const ClusterGrid& g, const math::Mat4& proj, const math::Mat4& view,
                         int screenW, int screenH, std::span<const PointLight> lights,
                         std::vector<std::vector<uint32_t>>& outPerCluster) {
    const int nClusters = g.clusterCount();
    outPerCluster.assign((size_t)nClusters, {});
    math::Mat4 invProj = proj.Inverse();

    // Pre-transform light positions into view space (affine: w stays 1).
    std::vector<math::Vec3> viewPos(lights.size());
    for (size_t li = 0; li < lights.size(); ++li)
        viewPos[li] = math::MulPoint(view, lights[li].posWorld);

    // Iterate clusters outer (cx-major then cy then cz) so the per-cluster AABB is computed once and
    // every light tested against it in ascending index order -> deterministic ordered lists.
    for (int cz = 0; cz < g.dimZ; ++cz)
        for (int cy = 0; cy < g.dimY; ++cy)
            for (int cx = 0; cx < g.dimX; ++cx) {
                math::Vec3 bmin, bmax;
                ClusterViewAABB(g, cx, cy, cz, invProj, screenW, screenH, bmin, bmax);
                std::vector<uint32_t>& dst = outPerCluster[(size_t)g.flatIndex(cx, cy, cz)];
                for (size_t li = 0; li < lights.size(); ++li)
                    if (SphereAABBIntersect(viewPos[li], lights[li].radius, bmin, bmax))
                        dst.push_back((uint32_t)li);
            }
}

// --- GPU-facing std430 records (16-byte-aligned float4s) the showcase uploads verbatim. -----------

// One light: { posView.xyz, radius } + { color.rgb, intensity }. 32 bytes. (The showcase fills
// posView with view-space positions so the shader culls + shades in view space.)
struct GpuLight {
    float posRadius[4];   // xyz = view-space position, w = radius
    float color[4];       // rgb = color, w = intensity
};

// One cluster's fixed-cap light slot array: count + MAX_LIGHTS_PER_CLUSTER u32 indices. The GPU fills
// these in ascending light index (ordered, no atomics) so the order matches AssignLights. The cap is
// sized so no cluster overflows for the showcase (the showcase asserts maxPerCluster < cap) AND so the
// brute-force single-cluster collapse (all 96 showcase lights in one cluster) fits one slot array — so
// the clustered + brute-force renders use the IDENTICAL ClusterList layout/shader.
static constexpr int kMaxLightsPerCluster = 96;

} // namespace hf::render::cluster
