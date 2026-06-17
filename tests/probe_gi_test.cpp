// Slice DH — DDGI Beachhead: Probe-Grid Ray-Trace. Pure CPU math: the world-space probe lattice, the
// deterministic Fibonacci-sphere ray set, the nearest-probe lookup, and the view-space depth-field
// march. No device, ASan-eligible (links hf_core). Exercises the EXACT math the --probegi-shot showcase
// + the probe_raytrace.comp shader copy VERBATIM (engine/render/probe_gi.h), so the GPU ray-hit SSBO is
// BIT-EXACT to this CPU reference.
//
// Properties pinned (per the spec):
//   * FibonacciSphere: unit-length for all i; deterministic; FibonacciSphere(0,1)==(0,0,1); even
//     distribution (|mean over i| -> 0); z monotone (decreasing) + evenly spaced ~2/N; no duplicate dirs.
//   * TraceRayToDepth — flat field: a ray aimed AWAY from the depth surface MISSES (w==kRayMiss); a ray
//     aimed AT the plane HITS at the analytic hit distance.
//   * TraceRayToDepth — synthetic occluder: a near-occluder block in the depth field -> a ray through the
//     occluder column hits at the OCCLUDER's distance, not the far plane.
//   * GetProbeGridIndex: round-trips probePos(px,py,pz) -> flatIndex(px,py,pz); far-outside positions
//     clamp to the boundary probe.
//   * Disabled path: dimX=0 / dimY=0 / dimZ=0 each -> probeCount()==0 -> ProbeDispatchGroups()==0.
//   * Determinism: the full per-probe ray set over a fixed field is bit-identical across two runs.
#include "render/probe_gi.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace probegi = hf::render::probegi;
using hf::math::Mat4;
using hf::math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

int main() {
    HF_TEST_MAIN_INIT();
    // ====================================================================================
    // FibonacciSphere
    // ====================================================================================
    {
        const int N = probegi::kRaysPerProbe;   // 16
        bool allUnit = true;
        Vec3 mean{0, 0, 0};
        std::vector<Vec3> dirs;
        for (int i = 0; i < N; ++i) {
            Vec3 d = probegi::FibonacciSphere(i, N);
            float len = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
            if (!approx(len, 1.0f, 1e-4f)) allUnit = false;
            mean = mean + d;
            dirs.push_back(d);
        }
        check(allUnit, "FibonacciSphere: every direction is unit-length");

        // Even distribution: the mean of a balanced full-sphere set is near the origin.
        mean = mean * (1.0f / (float)N);
        float meanMag = std::sqrt(mean.x * mean.x + mean.y * mean.y + mean.z * mean.z);
        check(meanMag < 0.15f, "FibonacciSphere: |mean over i| -> 0 (balanced full sphere)");

        // z monotone DECREASING (z = 1 - (2i+1)/N) + evenly spaced by ~2/N.
        bool zMonotone = true, zSpaced = true;
        for (int i = 1; i < N; ++i) {
            if (dirs[i].z >= dirs[i - 1].z) zMonotone = false;
            float dz = dirs[i - 1].z - dirs[i].z;   // positive step
            if (!approx(dz, 2.0f / (float)N, 1e-4f)) zSpaced = false;
        }
        check(zMonotone, "FibonacciSphere: z is monotone decreasing in i");
        check(zSpaced, "FibonacciSphere: z is evenly spaced (~2/N apart)");

        // No duplicate directions.
        bool noDup = true;
        for (int i = 0; i < N; ++i)
            for (int j = i + 1; j < N; ++j) {
                Vec3 a = dirs[i], b = dirs[j];
                if (approx(a.x, b.x, 1e-6f) && approx(a.y, b.y, 1e-6f) && approx(a.z, b.z, 1e-6f))
                    noDup = false;
            }
        check(noDup, "FibonacciSphere: no duplicate directions");

        // Determinism: a second call returns byte-identical directions.
        bool det = true;
        for (int i = 0; i < N; ++i) {
            Vec3 d = probegi::FibonacciSphere(i, N);
            if (std::memcmp(&d, &dirs[i], sizeof(Vec3)) != 0) det = false;
        }
        check(det, "FibonacciSphere: deterministic (byte-identical across calls)");

        // The canonical single-sample = the north pole.
        Vec3 one = probegi::FibonacciSphere(0, 1);
        check(approx(one.x, 0.0f) && approx(one.y, 0.0f) && approx(one.z, 1.0f),
              "FibonacciSphere(0,1) == (0,0,1)");

        // Clamp: out-of-range i/N are clamped to valid (no crash, finite, unit).
        Vec3 c = probegi::FibonacciSphere(999, N);
        float cl = std::sqrt(c.x * c.x + c.y * c.y + c.z * c.z);
        check(approx(cl, 1.0f, 1e-4f), "FibonacciSphere: out-of-range i clamps to a valid unit dir");
    }

    // ====================================================================================
    // TraceRayToDepth — flat field + synthetic occluder
    // ====================================================================================
    {
        // Camera at the origin looking down -Z (identity view) so world == view; a point at world
        // (x,y,-d) projects to the screen and has view-linear depth d. yFlip = -1 (Vulkan convention).
        Mat4 view = Mat4::Identity();   // world == view (camera at origin, looking down -Z)
        const float kFovY = 1.04719755f;   // 60 deg
        const float tanHalfFovY = std::tan(0.5f * kFovY);
        const float aspect = 16.0f / 9.0f;
        const float yFlip = -1.0f;
        const float kFarDepth = 10.0f;       // the flat "far plane" view-linear depth
        const int   kSteps = 64;
        const float kThickness = 0.25f;

        // --- FLAT depth field at a constant view-linear depth of kFarDepth everywhere. ---
        auto flatDepth = [&](float /*u*/, float /*v*/) { return kFarDepth; };

        // A ray straight down -Z from the origin: it marches toward the flat plane along the view axis;
        // its view-linear depth grows with t. It HITS when t crosses kFarDepth (within thickness). The
        // analytic hit distance is ~kFarDepth (the depth equals the distance for a -Z ray from origin).
        {
            probegi::ProbeRayHit hit{};
            bool hitOk = probegi::TraceRayToDepth(Vec3{0, 0, 0}, Vec3{0, 0, -1}, /*maxDist=*/20.0f,
                                                  kSteps, kThickness, view, tanHalfFovY, aspect, yFlip,
                                                  flatDepth, hit);
            check(hitOk, "TraceRayToDepth flat: a -Z ray HITS the constant-depth plane");
            check(approx(hit.hitPosDist[3], kFarDepth, /*eps=*/0.5f),
                  "TraceRayToDepth flat: hit distance ~= the plane's view-linear depth");
            // The hit world position is on the -Z axis at ~-kFarDepth.
            check(hit.hitPosDist[3] != probegi::kRayMiss,
                  "TraceRayToDepth flat: a hit's w is not the miss sentinel");
        }

        // A ray aimed AWAY from the plane (straight toward +Z, behind the camera) projects off-screen /
        // never crosses the surface in front -> MISS (w == kRayMiss).
        {
            probegi::ProbeRayHit hit{};
            bool hitOk = probegi::TraceRayToDepth(Vec3{0, 0, 0}, Vec3{0, 0, 1}, /*maxDist=*/20.0f,
                                                  kSteps, kThickness, view, tanHalfFovY, aspect, yFlip,
                                                  flatDepth, hit);
            check(!hitOk, "TraceRayToDepth flat: a +Z (away) ray MISSES");
            check(hit.hitPosDist[3] == probegi::kRayMiss,
                  "TraceRayToDepth flat: a miss writes w == kRayMiss");
        }

        // --- SYNTHETIC OCCLUDER: a near block. The depth field returns a NEAR depth (kNear) for screen
        // UVs left of center (u < 0.5) and the far plane (kFarDepth) elsewhere. A ray that marches into
        // the left (near-occluder) column must hit at the NEAR depth, not the far plane. ---
        const float kNear = 3.0f;
        auto occluderDepth = [&](float u, float /*v*/) { return (u < 0.5f) ? kNear : kFarDepth; };

        // Aim a ray that, projected, lands in the left half (u<0.5). A point at world (-x, 0, -z) with
        // x>0 projects to the LEFT of center on Vulkan. March a ray from the origin toward (-0.4,0,-1):
        // it crosses the near occluder's depth band (kNear) before it would reach kFarDepth.
        {
            Vec3 dir = hf::math::normalize(Vec3{-0.4f, 0.0f, -1.0f});
            probegi::ProbeRayHit hit{};
            bool hitOk = probegi::TraceRayToDepth(Vec3{0, 0, 0}, dir, /*maxDist=*/20.0f, kSteps,
                                                  kThickness, view, tanHalfFovY, aspect, yFlip,
                                                  occluderDepth, hit);
            check(hitOk, "TraceRayToDepth occluder: the ray through the near column HITS");
            // The hit's view-linear depth is ~kNear (the ray pierces the near occluder), well before the
            // far plane. The hit distance t corresponds to depth ~kNear; for a ray with -Z component
            // cos = |dir.z|, depth = t*|dir.z| ~= kNear -> t ~= kNear/|dir.z|.
            float expectedT = kNear / std::fabs(dir.z);
            check(approx(hit.hitPosDist[3], expectedT, /*eps=*/1.0f),
                  "TraceRayToDepth occluder: hits at the OCCLUDER distance, not the far plane");
            check(hit.hitPosDist[3] < kFarDepth / std::fabs(dir.z) - 1.0f,
                  "TraceRayToDepth occluder: the occluder hit is nearer than the far plane");
        }
    }

    // ====================================================================================
    // GetProbeGridIndex — round-trip + clamp
    // ====================================================================================
    {
        probegi::ProbeGrid grid;
        grid.origin = Vec3{-3.5f, 0.0f, -3.5f};
        grid.dimX = 8; grid.dimY = 4; grid.dimZ = 8; grid.spacing = 1.0f;

        check(grid.probeCount() == 8 * 4 * 8, "ProbeGrid: probeCount == dimX*dimY*dimZ (256)");

        // Round-trip: probePos(px,py,pz) -> GetProbeGridIndex -> flatIndex(px,py,pz).
        bool roundTrip = true;
        for (int pz = 0; pz < grid.dimZ; ++pz)
            for (int py = 0; py < grid.dimY; ++py)
                for (int px = 0; px < grid.dimX; ++px) {
                    Vec3 wp = grid.probePos(px, py, pz);
                    int idx = probegi::GetProbeGridIndex(wp, grid);
                    if (idx != grid.flatIndex(px, py, pz)) roundTrip = false;
                }
        check(roundTrip, "GetProbeGridIndex: round-trips probePos -> flatIndex at every lattice point");

        // Far-outside positions clamp to the boundary probe.
        int loIdx = probegi::GetProbeGridIndex(Vec3{-1000, -1000, -1000}, grid);
        check(loIdx == grid.flatIndex(0, 0, 0), "GetProbeGridIndex: far-negative clamps to probe (0,0,0)");
        int hiIdx = probegi::GetProbeGridIndex(Vec3{1000, 1000, 1000}, grid);
        check(hiIdx == grid.flatIndex(grid.dimX - 1, grid.dimY - 1, grid.dimZ - 1),
              "GetProbeGridIndex: far-positive clamps to the max-corner probe");
    }

    // ====================================================================================
    // Disabled path — dimX/dimY/dimZ == 0 -> probeCount 0 -> ProbeDispatchGroups 0
    // ====================================================================================
    {
        probegi::ProbeGrid gx; gx.dimX = 0;
        probegi::ProbeGrid gy; gy.dimY = 0;
        probegi::ProbeGrid gz; gz.dimZ = 0;
        check(gx.probeCount() == 0 && probegi::ProbeDispatchGroups(gx) == 0,
              "Disabled: dimX==0 -> probeCount 0 -> ProbeDispatchGroups 0");
        check(gy.probeCount() == 0 && probegi::ProbeDispatchGroups(gy) == 0,
              "Disabled: dimY==0 -> probeCount 0 -> ProbeDispatchGroups 0");
        check(gz.probeCount() == 0 && probegi::ProbeDispatchGroups(gz) == 0,
              "Disabled: dimZ==0 -> probeCount 0 -> ProbeDispatchGroups 0");

        // The enabled default grid dispatches ceil(256/64) == 4 groups.
        probegi::ProbeGrid def;
        check(probegi::ProbeDispatchGroups(def) == (256 + 63) / 64,
              "ProbeDispatchGroups: 256 probes -> ceil(256/64) == 4 groups");
    }

    // ====================================================================================
    // Determinism — the full per-probe ray set over a fixed field is bit-identical across two runs
    // ====================================================================================
    {
        probegi::ProbeGrid grid;
        grid.origin = Vec3{-1.5f, 0.5f, -1.5f};
        grid.dimX = 2; grid.dimY = 1; grid.dimZ = 2; grid.spacing = 1.0f;
        Mat4 view = Mat4::Identity();
        const float tanHalfFovY = std::tan(0.5f * 1.04719755f);
        const float aspect = 16.0f / 9.0f;
        const float yFlip = -1.0f;
        auto depthFn = [](float u, float v) { return 4.0f + 2.0f * u - 1.0f * v; };

        auto traceAll = [&]() {
            std::vector<probegi::ProbeRayHit> out((size_t)grid.probeCount() * probegi::kRaysPerProbe);
            for (int p = 0; p < grid.probeCount(); ++p) {
                int pz = p / (grid.dimX * grid.dimY);
                int rem = p % (grid.dimX * grid.dimY);
                int py = rem / grid.dimX;
                int px = rem % grid.dimX;
                Vec3 probeWorld = grid.probePos(px, py, pz);
                for (int r = 0; r < probegi::kRaysPerProbe; ++r) {
                    Vec3 dir = probegi::FibonacciSphere(r, probegi::kRaysPerProbe);
                    probegi::TraceRayToDepth(probeWorld, dir, 8.0f, 32, 0.3f, view, tanHalfFovY,
                                             aspect, yFlip, depthFn, out[(size_t)p * 16 + r]);
                }
            }
            return out;
        };
        auto a = traceAll();
        auto b = traceAll();
        check(a.size() == b.size() &&
                  std::memcmp(a.data(), b.data(), a.size() * sizeof(probegi::ProbeRayHit)) == 0,
              "Determinism: the full per-probe ray set is bit-identical across two runs");
    }

    if (g_fail == 0) { std::printf("probe_gi_test OK\n"); return 0; }
    std::printf("probe_gi_test: %d failures\n", g_fail);
    return 1;
}
