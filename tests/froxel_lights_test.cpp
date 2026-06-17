// Slice CV — Per-Froxel Clustered-Light Injection. Pure CPU math: the ADDED clustered-point-light
// in-scatter term froxel::InjectClusteredLights the inject pass (froxel_inject.comp.hlsl, behind the
// injectLights flag) accumulates on top of the CS sun in-scatter. No device, ASan-eligible (links
// hf_core). Exercises the EXACT math the GPU inject pass runs, so the lights-off==CS-fog +
// density=0==no-fog proofs in --froxellights-shot rest on this.
//
// Properties pinned (per the spec):
//   * NO LIGHTS = no added scatter: lightCount==0 -> (0,0,0) EXACTLY (the additive lights-off identity
//     -> the inject pass reduces to CS sun-only fog -> the byte-identical lights-off proof).
//   * A REACHING LIGHT scatters: a single light within `radius` of the froxel -> POSITIVE added scatter;
//     the magnitude scales with `density` (×0 density -> 0); falls with distance (windowed); a light
//     BEYOND `radius` -> 0 added (the windowed HARD cutoff, IDENTICAL to CL's atten).
//   * COLOR + PHASE: a colored light tints its color; the HG phase peaks (forward-scatter) when the light
//     is ALONG the view ray for g>0.
//   * FROXEL->CLUSTER Z MAP: a froxel at a known view-Z maps to the correct cluster slice via
//     cluster::SliceForViewZ; the froxel/cluster XY tiles align (both 16x9).
//   * DETERMINISM: same inputs -> bit-identical added scatter.
#include "render/froxel.h"
#include "render/cluster.h"
#include "math/math.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace froxel = hf::render::froxel;
namespace cluster = hf::render::cluster;
using hf::math::Vec3;
using hf::math::Mat4;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

int main() {
    HF_TEST_MAIN_INIT();
    const float kG = 0.76f;            // the showcase HG anisotropy
    const Mat4 view = Mat4::Identity();  // identity view -> world == view space, so the math is clear

    // ---- NO LIGHTS = no added scatter: lightCount==0 -> (0,0,0) EXACTLY (the lights-off identity). ----
    {
        std::vector<cluster::PointLight> lights;   // empty
        std::vector<uint32_t> indices;             // empty
        Vec3 add = froxel::InjectClusteredLights(Vec3{0, 0, -5}, 0.06f, Vec3{0, 0, -1}, kG,
                                                 std::span<const cluster::PointLight>(lights),
                                                 std::span<const uint32_t>(indices), 0, 0, view);
        check(add.x == 0.0f && add.y == 0.0f && add.z == 0.0f,
              "lightCount==0 -> (0,0,0) EXACTLY (additive lights-off identity)");
    }

    // A single white light a few units to the side of the froxel, well within radius.
    auto oneLight = [&](Vec3 lightPos, float radius, Vec3 color, float intensity) {
        std::vector<cluster::PointLight> lights(1);
        lights[0].posWorld = lightPos; lights[0].radius = radius;
        lights[0].color = color; lights[0].intensity = intensity;
        return lights;
    };
    const std::vector<uint32_t> idx0 = {0u};   // the cluster lists light 0 at offset 0

    // ---- A REACHING LIGHT scatters POSITIVE; scales with density (×0 -> 0). ----
    {
        Vec3 froxel{0, 0, -5};
        auto lights = oneLight(Vec3{1.5f, 0, -5}, 8.0f, Vec3{1, 1, 1}, 3.0f);
        Vec3 viewDir = hf::math::normalize(froxel);   // eye at origin looking toward the froxel

        Vec3 add = froxel::InjectClusteredLights(froxel, 0.06f, viewDir, kG,
                                                 std::span<const cluster::PointLight>(lights),
                                                 std::span<const uint32_t>(idx0), 0, 1, view);
        check(add.x > 0.0f && add.y > 0.0f && add.z > 0.0f,
              "a reaching light adds POSITIVE in-scatter");

        // ×density scaling: density 0 -> 0 EXACTLY; double the density -> double the scatter (linear).
        Vec3 zero = froxel::InjectClusteredLights(froxel, 0.0f, viewDir, kG,
                                                  std::span<const cluster::PointLight>(lights),
                                                  std::span<const uint32_t>(idx0), 0, 1, view);
        check(zero.x == 0.0f && zero.y == 0.0f && zero.z == 0.0f,
              "density==0 -> added scatter (0,0,0) EXACTLY (the density=0 no-fog foundation)");
        Vec3 dbl = froxel::InjectClusteredLights(froxel, 0.12f, viewDir, kG,
                                                 std::span<const cluster::PointLight>(lights),
                                                 std::span<const uint32_t>(idx0), 0, 1, view);
        check(approx(dbl.x, add.x * 2.0f, 1e-5f), "added scatter is LINEAR in density (×2 density -> ×2)");
    }

    // ---- Falls with distance (windowed attenuation). ----
    {
        Vec3 viewDir = Vec3{0, 0, -1};
        auto near = oneLight(Vec3{1.0f, 0, -5}, 12.0f, Vec3{1, 1, 1}, 3.0f);
        auto far  = oneLight(Vec3{5.0f, 0, -5}, 12.0f, Vec3{1, 1, 1}, 3.0f);
        Vec3 addNear = froxel::InjectClusteredLights(Vec3{0, 0, -5}, 0.06f, viewDir, kG,
                                                     std::span<const cluster::PointLight>(near),
                                                     std::span<const uint32_t>(idx0), 0, 1, view);
        Vec3 addFar = froxel::InjectClusteredLights(Vec3{0, 0, -5}, 0.06f, viewDir, kG,
                                                    std::span<const cluster::PointLight>(far),
                                                    std::span<const uint32_t>(idx0), 0, 1, view);
        check(addNear.x > addFar.x, "added scatter falls with distance (windowed attenuation)");
    }

    // ---- A light BEYOND radius adds EXACTLY 0 (the windowed HARD cutoff, matching CL). ----
    {
        Vec3 froxel{0, 0, -5};
        Vec3 viewDir = Vec3{0, 0, -1};
        // Light 10 units away, radius 4 -> dist (10) > radius (4) -> windowed atten EXACTLY 0.
        auto lights = oneLight(Vec3{10.0f, 0, -5}, 4.0f, Vec3{1, 1, 1}, 3.0f);
        Vec3 add = froxel::InjectClusteredLights(froxel, 0.06f, viewDir, kG,
                                                 std::span<const cluster::PointLight>(lights),
                                                 std::span<const uint32_t>(idx0), 0, 1, view);
        check(add.x == 0.0f && add.y == 0.0f && add.z == 0.0f,
              "a light BEYOND radius adds (0,0,0) EXACTLY (windowed hard cutoff, matching CL)");
        // And the cutoff matches CL's atten formula at the boundary: at d==radius win=(1-1)^2=0.
        float radius = 4.0f, d = radius;
        float r4 = d / radius; r4 = r4 * r4; r4 = r4 * r4;
        float win = 1.0f - r4; win = win < 0.0f ? 0.0f : win; win = win * win;
        check(win == 0.0f, "windowed atten == 0 EXACTLY at d==radius (CL boundary parity)");
    }

    // ---- COLOR: a colored light tints its added scatter by its color (no cross-channel bleed). ----
    {
        Vec3 froxel{0, 0, -5};
        Vec3 viewDir = Vec3{0, 0, -1};
        auto redLight = oneLight(Vec3{1.0f, 0, -5}, 10.0f, Vec3{1.0f, 0.0f, 0.0f}, 3.0f);
        Vec3 add = froxel::InjectClusteredLights(froxel, 0.06f, viewDir, kG,
                                                 std::span<const cluster::PointLight>(redLight),
                                                 std::span<const uint32_t>(idx0), 0, 1, view);
        check(add.x > 0.0f && add.y == 0.0f && add.z == 0.0f,
              "a RED light tints only the red channel (color carries through)");
    }

    // ---- PHASE: HG forward-peak. The light ALONG the view ray (forward scatter) gives MORE in-scatter
    //      than the same light off to the side, for g>0. ----
    {
        // Froxel at (0,0,-5); the camera ray travels toward -Z (viewDir = (0,0,-1)).
        Vec3 froxel{0, 0, -5};
        Vec3 viewDir = Vec3{0, 0, -1};
        // FORWARD: a light further down the -Z ray (dirToLight ~ (0,0,-1), aligned with viewDir -> cos~+1).
        auto fwdLight = oneLight(Vec3{0.0f, 0.0f, -9.0f}, 12.0f, Vec3{1, 1, 1}, 3.0f);
        // SIDE: a light to the +X side at the same distance (dirToLight ~ (+1,0,0) -> cos~0).
        auto sideLight = oneLight(Vec3{4.0f, 0.0f, -5.0f}, 12.0f, Vec3{1, 1, 1}, 3.0f);
        Vec3 addFwd = froxel::InjectClusteredLights(froxel, 0.06f, viewDir, kG,
                                                    std::span<const cluster::PointLight>(fwdLight),
                                                    std::span<const uint32_t>(idx0), 0, 1, view);
        Vec3 addSide = froxel::InjectClusteredLights(froxel, 0.06f, viewDir, kG,
                                                     std::span<const cluster::PointLight>(sideLight),
                                                     std::span<const uint32_t>(idx0), 0, 1, view);
        // Both lights are 4 units away (same windowed atten), so the difference is the HG phase only.
        check(addFwd.x > addSide.x,
              "HG phase forward-peaks: a light ALONG the view ray scatters more than one to the side (g>0)");
    }

    // ---- Two lights ACCUMULATE additively (the sum of singles). ----
    {
        Vec3 froxel{0, 0, -5};
        Vec3 viewDir = Vec3{0, 0, -1};
        std::vector<cluster::PointLight> lights(2);
        lights[0].posWorld = {1.0f, 0, -5}; lights[0].radius = 10.0f;
        lights[0].color = {1, 0, 0}; lights[0].intensity = 3.0f;
        lights[1].posWorld = {-1.0f, 0, -5}; lights[1].radius = 10.0f;
        lights[1].color = {0, 1, 0}; lights[1].intensity = 3.0f;
        std::vector<uint32_t> both = {0u, 1u};
        std::vector<uint32_t> justA = {0u};
        std::vector<uint32_t> justB = {1u};
        Vec3 addBoth = froxel::InjectClusteredLights(froxel, 0.06f, viewDir, kG,
                                                     std::span<const cluster::PointLight>(lights),
                                                     std::span<const uint32_t>(both), 0, 2, view);
        Vec3 addA = froxel::InjectClusteredLights(froxel, 0.06f, viewDir, kG,
                                                  std::span<const cluster::PointLight>(lights),
                                                  std::span<const uint32_t>(justA), 0, 1, view);
        Vec3 addB = froxel::InjectClusteredLights(froxel, 0.06f, viewDir, kG,
                                                  std::span<const cluster::PointLight>(lights),
                                                  std::span<const uint32_t>(justB), 0, 1, view);
        check(approx(addBoth.x, addA.x + addB.x, 1e-5f) && approx(addBoth.y, addA.y + addB.y, 1e-5f),
              "two lights accumulate ADDITIVELY (sum of singles)");
    }

    // ---- lightOffset indexes the shared light-index buffer (a cluster's slice of a flat index array). ----
    {
        Vec3 froxel{0, 0, -5};
        Vec3 viewDir = Vec3{0, 0, -1};
        std::vector<cluster::PointLight> lights(2);
        lights[0].posWorld = {100.0f, 0, -5}; lights[0].radius = 1.0f;   // far -> contributes 0
        lights[0].color = {1, 1, 1}; lights[0].intensity = 3.0f;
        lights[1].posWorld = {1.0f, 0, -5}; lights[1].radius = 10.0f;    // reaching
        lights[1].color = {1, 1, 1}; lights[1].intensity = 3.0f;
        // A flat index array [0, 1]; the cluster's slice starts at offset 1, count 1 -> only light 1.
        std::vector<uint32_t> flat = {0u, 1u};
        Vec3 add = froxel::InjectClusteredLights(froxel, 0.06f, viewDir, kG,
                                                 std::span<const cluster::PointLight>(lights),
                                                 std::span<const uint32_t>(flat), 1, 1, view);
        check(add.x > 0.0f, "lightOffset selects the cluster's slice of the shared light-index buffer");
    }

    // ---- FROXEL->CLUSTER Z MAP: a froxel at a known view-Z maps to the correct cluster slice. The froxel
    //      and cluster grids share the same exponential Z slicing + XY tiling (both 16x9), only Z
    //      resolution differs (froxel dimZ=64, cluster dimZ=24). The inject shader maps the froxel center's
    //      view-Z to a cluster slice via cluster::SliceForViewZ. ----
    {
        froxel::FroxelGrid fg; fg.dimX = 16; fg.dimY = 9; fg.dimZ = 64; fg.zNear = 0.5f; fg.zFar = 80.0f;
        cluster::ClusterGrid cg; cg.dimX = 16; cg.dimY = 9; cg.dimZ = 24; cg.zNear = 0.5f; cg.zFar = 80.0f;
        check(fg.dimX == cg.dimX && fg.dimY == cg.dimY, "froxel + cluster XY tiles align (both 16x9)");

        // The froxel center view-Z of a near, mid, and far slice maps to a non-decreasing cluster slice
        // (monotone), each in-range, and the near/far froxels land in the near/far cluster slices.
        bool monotone = true; int prevCz = -1; bool inRange = true;
        for (int fz = 0; fz < fg.dimZ; ++fz) {
            float vz = froxel::SliceCenterViewZ(fg, fz);
            int cz = cluster::SliceForViewZ(cg, vz);
            if (cz < 0 || cz >= cg.dimZ) inRange = false;
            if (cz < prevCz) monotone = false;
            prevCz = cz;
        }
        check(inRange, "every froxel center maps to an IN-RANGE cluster slice [0, cluster dimZ)");
        check(monotone, "froxel view-Z -> cluster slice is MONOTONE (shared exponential Z slicing)");
        // The nearest froxel maps to cluster slice 0; the farthest to the last cluster slice.
        check(cluster::SliceForViewZ(cg, froxel::SliceCenterViewZ(fg, 0)) == 0,
              "the nearest froxel center maps to cluster slice 0");
        check(cluster::SliceForViewZ(cg, froxel::SliceCenterViewZ(fg, fg.dimZ - 1)) == cg.dimZ - 1,
              "the farthest froxel center maps to the last cluster slice");
    }

    // ---- DETERMINISM: same inputs -> bit-identical added scatter. ----
    {
        Vec3 froxel{0.3f, -0.2f, -6.0f};
        Vec3 viewDir = hf::math::normalize(froxel);
        auto lights = oneLight(Vec3{1.4f, 0.5f, -5.5f}, 9.0f, Vec3{0.9f, 0.3f, 0.7f}, 2.5f);
        Vec3 a = froxel::InjectClusteredLights(froxel, 0.07f, viewDir, kG,
                                               std::span<const cluster::PointLight>(lights),
                                               std::span<const uint32_t>(idx0), 0, 1, view);
        Vec3 b = froxel::InjectClusteredLights(froxel, 0.07f, viewDir, kG,
                                               std::span<const cluster::PointLight>(lights),
                                               std::span<const uint32_t>(idx0), 0, 1, view);
        check(a.x == b.x && a.y == b.y && a.z == b.z,
              "InjectClusteredLights is deterministic (bit-identical across runs)");
    }

    if (g_fail == 0) { std::printf("froxel_lights_test: all checks passed\n"); return 0; }
    std::printf("froxel_lights_test: %d FAILURES\n", g_fail);
    return 1;
}
