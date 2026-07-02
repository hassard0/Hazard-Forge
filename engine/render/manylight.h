#pragma once
// Slice ML1 — MANY-LIGHT RENDERING (Track-S S7): the SSBO-backed path that lifts the
// HF_MAX_POINT_LIGHTS=8 FrameData cap to 100+ dynamic lights. Pure CPU (header-only, no device,
// no backend symbols) — shared by the --manylight-shot showcases (Vulkan + Metal) AND
// tests/manylight_test.cpp so all three exercise the SAME deterministic light field + the SAME
// cluster assignment (engine/render/clustered.h, reused AS-IS) and pin the SAME digest.
//
// THE DESIGN: FrameData's 8-slot point-light array stays byte-untouched (39 shaders include
// shaders/frame_data.hlsli; existing goldens depend on its layout). The many-light path instead
// carries N lights (N up to kMaxManyLights) in the three set-3 storage buffers the clustered
// machinery already binds via BindLightClusters (bindings 13/14/15 — proven by --probe-binding13):
//   binding 13: StructuredBuffer<Cluster>  { offset, count }   (CX*CY*CZ entries)
//   binding 14: StructuredBuffer<uint>     light indices        (flat, per-cluster runs)
//   binding 15: StructuredBuffer<GpuLight> { posRadius, color } (N lights, VIEW space)
// shaders/lit_manylight.frag.hlsl iterates ONLY its fragment's cluster's lights — the FrameData
// array is simply unused there.
//
// DETERMINISM: light positions/radii/colors derive from pcg::PcgHash (the engine's proven integer
// avalanche — NO RNG, NO clock). Each hashed 16-bit value becomes a float via an EXACT power-of-two
// scale ((h >> 16) * (1/65536)), so the light field is bit-identical on every compiler/vendor. The
// cluster assignment itself is float sphere-vs-AABB math (clustered.h) over a fixed camera; its
// digest (FNV-1a-64 over clusters||lightIndices, the net::DigestBytes currency) is pinned in
// tests/manylight_test.cpp. NOTE (honesty): the assignment uses std::pow/std::log slice math, so
// the digest is pinned per-toolchain-family (MSVC + clang-on-Windows verified identical; the Metal
// stat line pins whatever the Mac computes — a last-ULP libm divergence would surface there as a
// different digest, not silent corruption).

#include "render/clustered.h"
#include "pcg/pcg.h"
#include "net/session.h"
#include "math/math.h"
#include <cstdint>
#include <vector>

namespace hf::render::manylight {

// SSBO-path capacity (a soft authoring cap — buffers are sized to the actual N; the shader loop is
// bounded by each cluster's count, never this constant). 1024 = 32 KiB of GpuLight, trivially
// within storage-buffer budgets on both backends.
inline constexpr int kMaxManyLights = 1024;

// The ML1 showcase light count: 16x the old FrameData cap.
inline constexpr int kShowcaseLights = 128;
inline constexpr uint32_t kShowcaseSeed = 0x4D4C3101u;  // 'M','L','1',v1

// Hashed [0,1) float from (seed, stream index). The top 16 bits of the PcgHash avalanche scaled by
// an exact power of two — a pure integer-to-float conversion + multiply, bit-identical everywhere.
inline float Hash01(uint32_t seed, uint32_t idx) {
    return (float)(pcg::PcgHash(seed, idx) >> 16) * (1.0f / 65536.0f);
}

// N deterministic WORLD-space point lights scattered over the clustered-shot floor field
// (spanX x spanZ centered near the origin, hovering low so the pools spread on the ground).
// Every attribute is a pure function of (seed, light index) via PcgHash — replay-stable.
inline std::vector<clustered::Light> MakeManyLights(uint32_t seed, int count,
                                                    float spanX = 34.0f, float spanZ = 26.0f) {
    std::vector<clustered::Light> lights;
    lights.reserve((size_t)count);
    for (int i = 0; i < count; ++i) {
        uint32_t base = (uint32_t)i * 8u;  // 8 hash streams per light (x,z,y,radius,r,g,b,spare)
        clustered::Light L{};
        L.viewPos = {  // world position (the caller transforms into view space for the culler)
            (Hash01(seed, base + 0) - 0.5f) * spanX,
            0.8f + Hash01(seed, base + 2) * 1.6f,             // y in [0.8, 2.4) — just above floor
            (Hash01(seed, base + 1) - 0.5f) * spanZ - 2.0f,
        };
        L.radius = 3.0f + Hash01(seed, base + 3) * 3.0f;      // [3, 6) — pools overlap richly
        // Vivid hashed color: each channel in [0.15, 1.0), renormalized-ish by construction.
        L.color = {
            0.15f + Hash01(seed, base + 4) * 0.85f,
            0.15f + Hash01(seed, base + 5) * 0.85f,
            0.15f + Hash01(seed, base + 6) * 0.85f,
        };
        L.intensity = 2.6f;
        lights.push_back(L);
    }
    return lights;
}

// Per-cluster assignment statistics — the pinned showcase stat-line numbers.
struct AssignStats {
    uint32_t maxPerCluster = 0;    // densest cluster's light count
    uint64_t totalAssignments = 0; // sum of counts == lightIndices.size()
};
inline AssignStats ComputeAssignStats(const clustered::ClusterBuffers& cb) {
    AssignStats s;
    for (const auto& c : cb.clusters) {
        if (c.count > s.maxPerCluster) s.maxPerCluster = c.count;
        s.totalAssignments += c.count;
    }
    return s;
}

// FNV-1a-64 continuation (same constants as net::DigestBytes): folding more bytes into a running
// digest equals DigestBytes over the concatenation.
inline uint64_t FnvAppend(uint64_t h, const void* data, std::size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// The deterministic artifact: FNV-1a-64 over the whole CPU light-assignment
// (clusters {offset,count} array || flat lightIndices array).
inline uint64_t DigestAssignment(const clustered::ClusterBuffers& cb) {
    uint64_t h = net::DigestBytes(cb.clusters.data(),
                                  cb.clusters.size() * sizeof(clustered::GpuCluster));
    h = FnvAppend(h, cb.lightIndices.data(), cb.lightIndices.size() * sizeof(uint32_t));
    return h;
}

// The COMPLETE showcase assignment: the fixed --manylight-shot camera + grid (identical to the
// --clustered-shot vantage: eye {0,16,26} looking at {0,0,-2}, 60deg fovY, znear 0.5, zfar 90,
// 16x9x24 clusters), N hashed lights transformed into view space, culled via clustered.h. The
// Vulkan showcase, the Metal showcase, and manylight_test all call THIS, so the pinned digest is
// one artifact with three witnesses.
struct ShowcaseAssignment {
    clustered::Grid grid;
    math::Mat4 view, proj;                     // proj UNFLIPPED (Metal flips Y for render only)
    math::Vec3 eye{0.0f, 16.0f, 26.0f};
    std::vector<clustered::Light> viewLights;  // view-space (what the culler + GPU buffer consume)
    clustered::ClusterBuffers buffers;
    AssignStats stats;
    uint64_t digest = 0;
};
inline ShowcaseAssignment BuildShowcaseAssignment(uint32_t seed, int count,
                                                  float screenW, float screenH,
                                                  int cx = 16, int cy = 9, int cz = 24) {
    ShowcaseAssignment sa;
    const float kNear = 0.5f, kFar = 90.0f;
    const float fovY = 1.04719755f;  // 60deg — matches --clustered-shot
    const float aspect = (screenH > 0.0f) ? screenW / screenH : 1.0f;
    sa.view = math::Mat4::LookAt(sa.eye, {0.0f, 0.0f, -2.0f}, {0, 1, 0});
    sa.proj = math::Mat4::Perspective(fovY, aspect, kNear, kFar);
    sa.grid = clustered::MakeGrid(sa.proj, kNear, kFar, screenW, screenH, cx, cy, cz);

    std::vector<clustered::Light> worldLights = MakeManyLights(seed, count);
    sa.viewLights.reserve(worldLights.size());
    for (const clustered::Light& L : worldLights) {
        float vw = 0.0f;
        clustered::Light Lv = L;
        Lv.viewPos = math::MulPointDivide(sa.view, L.viewPos, vw);  // affine: w stays 1
        sa.viewLights.push_back(Lv);
    }
    sa.buffers = clustered::BuildClusters(sa.grid, sa.viewLights);
    if (sa.buffers.lightIndices.empty()) sa.buffers.lightIndices.push_back(0u);  // non-zero-size SSBO
    sa.stats = ComputeAssignStats(sa.buffers);
    sa.digest = DigestAssignment(sa.buffers);
    return sa;
}

} // namespace hf::render::manylight
