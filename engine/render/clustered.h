#pragma once
// Clustered / Forward+ light culling — pure CPU (header-only, no device, no backend symbols).
// Shared by the --clustered-shot showcase AND tests/clustered_test.cpp so the GPU path and the
// unit test exercise the SAME cluster-grid math (slice formula + screen-tile mapping +
// sphere/AABB assignment) that shaders/lit_clustered.frag.hlsl mirrors.
//
// THE TECHNIQUE (Slice AG): partition the camera view frustum into a CX*CY*CZ grid of clusters.
//   * XY follow the framebuffer's screen tiles (CX columns, CY rows).
//   * Z is split into CZ EXPONENTIAL depth slices between znear..zfar, so near clusters are thin
//     and far clusters are deep (matches perspective foreshortening).
// For a set of point lights {viewSpacePos, radius, color} we test each light's bounding sphere
// against each cluster's view-space AABB and record, per cluster, the indices of the lights that
// overlap it. A clustered-lit fragment then computes its own cluster index and iterates ONLY that
// cluster's lights instead of all N — making HUNDREDS of point lights affordable.
//
// CONVENTIONS (must match engine/math + lit_clustered.frag.hlsl EXACTLY):
//   * Mat4 is column-major; Mat4::Perspective produces Vulkan clip space (depth [0,1], Y-flip
//     baked into m[5] = -1/tan(fovY/2)). View space looks down -Z, so a fragment in front of the
//     camera has view-space z < 0; we use vz = -viewSpacePos.z as a POSITIVE distance.
//   * Cluster flat index: idx = cx + cy*CX + cz*(CX*CY).
//   * Screen-tile XY:  cx = clamp(floor(px/W * CX), 0, CX-1),  cy = clamp(floor(py/H * CY), ...)
//     with (px,py) = SV_Position.xy (top-left origin, framebuffer pixels).
//   * Exponential Z:   sliceZ(k) = znear * (zfar/znear)^(k/CZ);
//     inverse:          cz = clamp(floor( log(vz/znear)/log(zfar/znear) * CZ ), 0, CZ-1).
//
// The cluster's view-space AABB:
//   * Z bounds: [ -sliceZ(cz+1), -sliceZ(cz) ]  (view z is negative; -far .. -near).
//   * XY bounds: the screen tile's NDC rectangle unprojected to BOTH the near & far z planes,
//     taking the min/max over all 8 corners (a conservative AABB containing the cluster's frustum
//     sub-volume). At a positive depth d the view-space half-extents are  x = ndcX * d * tanX,
//     y = ndcYtrue * d * tanY  where tanX = 1/proj.m[0], tanY = 1/|proj.m[5]|, and ndcYtrue maps
//     a TOP-origin screen row to view-up via the Y-flip already encoded in proj.m[5].

#include "math/math.h"
#include <cmath>
#include <cstdint>
#include <vector>

namespace hf::render::clustered {

// A point light as the CPU culler + the GPU `lights` buffer see it: VIEW-SPACE position + radius
// packed in posRadius (xyz=viewSpacePos, w=radius); color rgb + intensity in w.
struct Light {
    math::Vec3 viewPos;   // light position in VIEW space (caller applies view * worldPos)
    float      radius;    // attenuation/culling radius (lights beyond this contribute ~0)
    math::Vec3 color;     // linear rgb
    float      intensity; // scalar multiplier
};

// GPU-facing records (std430 / Metal-friendly: 16-byte aligned float4s). The showcase uploads
// these arrays verbatim into the three Storage buffers the fragment shader reads.
struct GpuLight {            // 32 bytes
    float posRadius[4];      // xyz = view-space position, w = radius
    float color[4];          // rgb = color, w = intensity
};
struct GpuCluster {          // 8 bytes
    uint32_t offset;         // first index into lightIndices for this cluster
    uint32_t count;          // number of light indices
};

struct Grid {
    int   cx = 16, cy = 9, cz = 24;
    float znear = 0.1f, zfar = 80.0f;
    float screenW = 1280.0f, screenH = 720.0f;
    // Per-unit-depth view-space half-extents (tan of half FOV), derived from the projection matrix.
    float tanX = 1.0f, tanY = 1.0f;

    int clusterCount() const { return cx * cy * cz; }
    int flatIndex(int icx, int icy, int icz) const { return icx + icy * cx + icz * (cx * cy); }
};

// Build the grid params from a camera. `proj` must be a Mat4::Perspective (the showcase + tests
// always use one). tanX/tanY are read straight off the projection so the CPU and the shader agree
// without re-deriving the FOV.
inline Grid MakeGrid(const math::Mat4& proj, float znear, float zfar, float screenW, float screenH,
                     int cx = 16, int cy = 9, int cz = 24) {
    Grid g;
    g.cx = cx; g.cy = cy; g.cz = cz;
    g.znear = znear; g.zfar = zfar;
    g.screenW = screenW; g.screenH = screenH;
    g.tanX = 1.0f / proj.m[0];          // proj.m[0] = 1/(aspect*tan(fovY/2))  -> tanX = aspect*tanY
    g.tanY = 1.0f / std::fabs(proj.m[5]); // proj.m[5] = -1/tan(fovY/2)
    return g;
}

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Exponential z-slice boundary k in [0,CZ]: positive view-space distance of slice plane k.
inline float SliceZ(const Grid& g, int k) {
    return g.znear * std::pow(g.zfar / g.znear, (float)k / (float)g.cz);
}

// View-space depth (positive distance) -> z-slice index. Inverse of SliceZ. MIRRORED in HLSL.
inline int SliceForViewZ(const Grid& g, float viewZ) {
    if (viewZ <= g.znear) return 0;
    if (viewZ >= g.zfar) return g.cz - 1;
    float t = std::log(viewZ / g.znear) / std::log(g.zfar / g.znear);  // [0,1)
    return clampi((int)std::floor(t * (float)g.cz), 0, g.cz - 1);
}

// Screen pixel (top-left origin) -> tile column/row.
inline int TileX(const Grid& g, float px) {
    return clampi((int)std::floor(px / g.screenW * (float)g.cx), 0, g.cx - 1);
}
inline int TileY(const Grid& g, float py) {
    return clampi((int)std::floor(py / g.screenH * (float)g.cy), 0, g.cy - 1);
}

// Cluster index for a VIEW-SPACE position. Projects to screen via tanX/tanY (matching the shader's
// reconstruction) then applies the tile + slice maps. Returns -1 if behind the camera.
inline int ClusterForViewPos(const Grid& g, const math::Vec3& vpos) {
    float vz = -vpos.z;                       // positive distance in front of camera
    if (vz <= 0.0f) return -1;
    // Project: ndcX = vx/(vz*tanX) in [-1,1]; screen px = (ndcX*0.5+0.5)*W.
    // View-up is +Y; screen row grows DOWNWARD, so the top-origin py uses (0.5 - ndcY*0.5).
    float ndcX = vpos.x / (vz * g.tanX);
    float ndcY = vpos.y / (vz * g.tanY);
    float px = (ndcX * 0.5f + 0.5f) * g.screenW;
    float py = (0.5f - ndcY * 0.5f) * g.screenH;
    int icx = TileX(g, px);
    int icy = TileY(g, py);
    int icz = SliceForViewZ(g, vz);
    return g.flatIndex(icx, icy, icz);
}

// The view-space AABB of cluster (icx,icy,icz): min/max corners.
inline void ClusterAABB(const Grid& g, int icx, int icy, int icz,
                        math::Vec3& outMin, math::Vec3& outMax) {
    float zNearK = SliceZ(g, icz);       // positive near-plane distance of this slice
    float zFarK  = SliceZ(g, icz + 1);   // positive far-plane distance
    // Screen-tile NDC rectangle (x grows right; for y, top-origin row maps to view-up via flip).
    float nx0 = ((float)icx / (float)g.cx) * 2.0f - 1.0f;       // left edge NDC x
    float nx1 = ((float)(icx + 1) / (float)g.cx) * 2.0f - 1.0f; // right edge NDC x
    // py top edge = icy/CY of screen -> ndcY = 1 - 2*(py/H); bottom edge = (icy+1)/CY.
    float nyTop = 1.0f - 2.0f * ((float)icy / (float)g.cy);
    float nyBot = 1.0f - 2.0f * ((float)(icy + 1) / (float)g.cy);
    float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
    const float zs[2] = {zNearK, zFarK};
    const float nxs[2] = {nx0, nx1};
    const float nys[2] = {nyTop, nyBot};
    for (float d : zs)
        for (float nx : nxs)
            for (float ny : nys) {
                float vx = nx * d * g.tanX;
                float vy = ny * d * g.tanY;
                minX = std::min(minX, vx); maxX = std::max(maxX, vx);
                minY = std::min(minY, vy); maxY = std::max(maxY, vy);
            }
    outMin = { minX, minY, -zFarK };   // view z is negative: [-far, -near]
    outMax = { maxX, maxY, -zNearK };
}

// Squared distance from a point to an AABB (0 inside). Standard clamped-distance test.
inline float SqDistPointAABB(const math::Vec3& p, const math::Vec3& bmin, const math::Vec3& bmax) {
    float d = 0.0f;
    float vx = p.x < bmin.x ? bmin.x - p.x : (p.x > bmax.x ? p.x - bmax.x : 0.0f);
    float vy = p.y < bmin.y ? bmin.y - p.y : (p.y > bmax.y ? p.y - bmax.y : 0.0f);
    float vz = p.z < bmin.z ? bmin.z - p.z : (p.z > bmax.z ? p.z - bmax.z : 0.0f);
    d = vx * vx + vy * vy + vz * vz;
    return d;
}

// True if a light's bounding sphere (view-space center, radius) overlaps cluster (icx,icy,icz).
inline bool LightOverlapsCluster(const Grid& g, const Light& light, int icx, int icy, int icz) {
    math::Vec3 bmin, bmax;
    ClusterAABB(g, icx, icy, icz, bmin, bmax);
    return SqDistPointAABB(light.viewPos, bmin, bmax) <= light.radius * light.radius;
}

// Output of a full cull: the three GPU buffers (clusters / lightIndices / lights).
struct ClusterBuffers {
    std::vector<GpuCluster> clusters;       // CX*CY*CZ entries
    std::vector<uint32_t>   lightIndices;   // flat, length == sum of counts
    std::vector<GpuLight>   lights;         // one per input light
};

// Build the cluster buffers for a set of VIEW-SPACE lights. Two passes: count -> prefix-sum
// offsets -> fill. Deterministic (iteration order is cluster-major then light-major).
inline ClusterBuffers BuildClusters(const Grid& g, const std::vector<Light>& lights) {
    ClusterBuffers out;
    const int nClusters = g.clusterCount();
    const int nLights = (int)lights.size();
    out.clusters.assign(nClusters, GpuCluster{0, 0});

    // Pass 1: count overlaps per cluster.
    // (Iterate clusters outer so per-cluster AABB is computed once per light test.)
    std::vector<uint32_t> counts(nClusters, 0);
    // For each light, mark which clusters it overlaps; we store a per-light hit list to avoid
    // recomputing the AABB twice. nClusters*nLights AABB tests — fine for the CPU showcase.
    std::vector<std::vector<uint32_t>> perCluster(nClusters);
    for (int li = 0; li < nLights; ++li) {
        const Light& L = lights[li];
        // Conservative screen-space slab the light could touch, to skip obviously-distant clusters
        // is possible, but for determinism + simplicity we test all clusters (grid is small).
        for (int icz = 0; icz < g.cz; ++icz)
            for (int icy = 0; icy < g.cy; ++icy)
                for (int icx = 0; icx < g.cx; ++icx) {
                    if (LightOverlapsCluster(g, L, icx, icy, icz)) {
                        perCluster[g.flatIndex(icx, icy, icz)].push_back((uint32_t)li);
                    }
                }
    }

    // Prefix-sum offsets + flatten.
    uint32_t running = 0;
    for (int c = 0; c < nClusters; ++c) {
        out.clusters[c].offset = running;
        out.clusters[c].count = (uint32_t)perCluster[c].size();
        running += out.clusters[c].count;
    }
    out.lightIndices.reserve(running);
    for (int c = 0; c < nClusters; ++c)
        for (uint32_t idx : perCluster[c]) out.lightIndices.push_back(idx);

    // GPU light records.
    out.lights.reserve(nLights);
    for (const Light& L : lights) {
        GpuLight gl{};
        gl.posRadius[0] = L.viewPos.x; gl.posRadius[1] = L.viewPos.y;
        gl.posRadius[2] = L.viewPos.z; gl.posRadius[3] = L.radius;
        gl.color[0] = L.color.x; gl.color[1] = L.color.y; gl.color[2] = L.color.z;
        gl.color[3] = L.intensity;
        out.lights.push_back(gl);
    }
    return out;
}

} // namespace hf::render::clustered
