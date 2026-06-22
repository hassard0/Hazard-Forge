// Slice RT5 — Hardware Ray Tracing: DETERMINISM-ENVELOPE + LOCKSTEP TIE-IN (sim ⊗ RT). The pure-CPU
// invariants the --rt5-simrender-shot / --rt5-simrender showcases rest on (the HW==CPU memcmp lives in the
// Vulkan showcase, which needs a GPU). RT5 COMPOSES the determinism moat (the fpx deterministic Q16.16
// rigid-body sim, lockstep-replayable) with the RT4 render. This test pins the joint invariant:
//
//   * SIM LOCKSTEP: fpx::RunLockstep(init, stream, N) for the authority AND the replica (same inputs, no
//     shared state) -> BYTE-IDENTICAL final worlds (the deterministic sim re-derives bit-for-bit).
//   * CONVERTER PURE: BodiesToRtScene is a pure function of the world — two calls byte-equal; a B-body
//     world yields B+1 primitives (1 ground AABB + B spheres); the ground is primIndex 0 (reflective).
//   * RENDER CLOSURE: RenderSceneReflected(authScene) == RenderSceneReflected(replicaScene) BYTE-EQUAL (a
//     deterministic sim renders reproducibly — RT(authority) == RT(replica)).
//   * RENDER-ONLY: the RT render does NOT mutate the fpx world (the world is byte-identical pre vs post —
//     a pure read of the sim).
//
// Pure C++ (hf_core, ASan-eligible). rtrace.h #includes sim/fpx.h read-only; both are byte-frozen; the
// shared bridge tests/rt5_simrender_scene.h #includes BOTH read-only and is the SAME converter the
// showcases use.
#include "rt5_simrender_scene.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "test_main.h"

using namespace hf;
namespace rt = hf::render::rtrace;
namespace fpx = hf::sim::fpx;
namespace rt5 = hf::render::rt5;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static size_t BodiesBytes(const fpx::FxWorld& w) { return w.bodies.size() * sizeof(fpx::FxBody); }

static bool WorldsByteEqual(const fpx::FxWorld& a, const fpx::FxWorld& b) {
    return a.bodies.size() == b.bodies.size() &&
           a.groundY == b.groundY && a.gravity.x == b.gravity.x &&
           a.gravity.y == b.gravity.y && a.gravity.z == b.gravity.z &&
           std::memcmp(a.bodies.data(), b.bodies.data(), BodiesBytes(a)) == 0;
}

int main() {
    HF_TEST_MAIN_INIT();

    const fpx::FxWorld init = rt5::BuildRt5World();
    const std::vector<fpx::FxCommand> stream = rt5::BuildRt5Stream();

    // ================= SIM LOCKSTEP: authority == replica BIT-IDENTICAL ================================
    const fpx::FxWorld authority =
        fpx::RunLockstep(init, stream, rt5::kRt5Ticks, rt5::kRt5Dt, rt5::kRt5Iters);
    const fpx::FxWorld replicaClean =
        fpx::RunLockstep(init, stream, rt5::kRt5Ticks, rt5::kRt5Dt, rt5::kRt5Iters);
    check((int)authority.bodies.size() == rt5::kRt5Bodies,
          "RunLockstep: the authority world has the configured body count");
    check(WorldsByteEqual(authority, replicaClean),
          "SIM LOCKSTEP: authority == replica BYTE-IDENTICAL (inputs-only re-sim is deterministic)");

    // The pile actually settled (a resting heap — every body at/above the ground, not exploded). Sanity:
    // every dynamic body's y is finite-and-bounded near the ground (pos.y in [groundY-eps, a few units]).
    {
        bool settledBand = true;
        for (const fpx::FxBody& b : authority.bodies) {
            if (b.pos.y < -(fpx::kOne)              // below the ground by more than 1 unit -> exploded down
                || b.pos.y > (fpx::fx)(10 * (int)fpx::kOne)) settledBand = false;  // flew up -> diverged
        }
        check(settledBand, "SIM: the settled pile is in a sane band (no body exploded through/over the ground)");
    }

    // ================= CONVERTER PURE: two calls byte-equal; B bodies -> B+1 prims ======================
    {
        rt5::Rt5Scene s1 = rt5::BodiesToRtScene(authority);
        rt5::Rt5Scene s2 = rt5::BodiesToRtScene(authority);
        check(s1.spheres.size() == authority.bodies.size(),
              "BodiesToRtScene: one sphere per body");
        check(s1.aabbs.size() == 1, "BodiesToRtScene: exactly one ground AABB");
        check(s1.spheres.size() + s1.aabbs.size() == authority.bodies.size() + 1,
              "BodiesToRtScene: B bodies -> B+1 primitives");
        check(s1.aabbs[0].primIndex == 0,
              "BodiesToRtScene: the ground AABB is primIndex 0");
        check(rt::ReflectivityFor(s1.aabbs[0].primIndex) > 0,
              "BodiesToRtScene: the ground (primIndex 0) is reflective");
        // Provenance: two calls produce byte-equal primitive arrays (a pure function of the world).
        bool sphEq = s1.spheres.size() == s2.spheres.size() &&
                     std::memcmp(s1.spheres.data(), s2.spheres.data(),
                                 s1.spheres.size() * sizeof(rt::RtSphere)) == 0;
        bool aabbEq = s1.aabbs.size() == s2.aabbs.size() &&
                      std::memcmp(s1.aabbs.data(), s2.aabbs.data(),
                                  s1.aabbs.size() * sizeof(rt::RtAabb)) == 0;
        check(sphEq && aabbEq,
              "BodiesToRtScene: PURE — two calls produce byte-equal primitives (the provenance contract)");
        // Each sphere carries the body's exact pos/radius + the 1+i primIndex (the bridge mapping).
        bool mapped = true;
        for (size_t i = 0; i < authority.bodies.size(); ++i) {
            const fpx::FxBody& b = authority.bodies[i];
            const rt::RtSphere& sp = s1.spheres[i];
            if (sp.center.x != b.pos.x || sp.center.y != b.pos.y || sp.center.z != b.pos.z ||
                sp.radius != b.radius || sp.primIndex != (uint32_t)(1 + i)) mapped = false;
        }
        check(mapped,
              "BodiesToRtScene: each FxBody -> RtSphere{center=pos, radius=radius, primIndex=1+i}");
    }

    // ================= RENDER CLOSURE: RT(authority) == RT(replica) BYTE-EQUAL ==========================
    {
        rt5::Rt5Scene sa = rt5::BodiesToRtScene(authority);
        rt5::Rt5Scene sr = rt5::BodiesToRtScene(replicaClean);
        const uint32_t W = 160, H = 120;
        const size_t kPixels = (size_t)W * H;
        std::vector<uint32_t> imgA(kPixels, 0), imgR(kPixels, 0);
        rt::RenderSceneReflected(sa.scene, sa.camera, W, H, std::span<uint32_t>(imgA));
        rt::RenderSceneReflected(sr.scene, sr.camera, W, H, std::span<uint32_t>(imgR));
        check(std::memcmp(imgA.data(), imgR.data(), kPixels * sizeof(uint32_t)) == 0,
              "RENDER CLOSURE: RT(authority) == RT(replica) image BYTE-EQUAL (deterministic sim renders reproducibly)");

        // Two renders of the SAME scene are also byte-identical (the RT path itself is deterministic).
        std::vector<uint32_t> imgA2(kPixels, 0);
        rt::RenderSceneReflected(sa.scene, sa.camera, W, H, std::span<uint32_t>(imgA2));
        check(std::memcmp(imgA.data(), imgA2.data(), kPixels * sizeof(uint32_t)) == 0,
              "RENDER: two renders of the authority scene are BYTE-IDENTICAL (deterministic RT)");

        // The render actually drew the pile (some pixels HIT a primitive — not a blank frame). A hit pixel
        // differs from the pure background.
        bool drewSomething = false;
        for (size_t p = 0; p < kPixels; ++p) if (imgA[p] != sa.scene.background) { drewSomething = true; break; }
        check(drewSomething, "RENDER: the settled pile is visible (non-background pixels present)");
    }

    // ================= RENDER-ONLY: the RT render does NOT mutate the fpx world ========================
    {
        fpx::FxWorld world = fpx::RunLockstep(init, stream, rt5::kRt5Ticks, rt5::kRt5Dt, rt5::kRt5Iters);
        const fpx::FxWorld before = fpx::SnapshotWorld(world);   // deep copy
        rt5::Rt5Scene s = rt5::BodiesToRtScene(world);
        const uint32_t W = 96, H = 72;
        std::vector<uint32_t> img((size_t)W * H, 0);
        rt::RenderSceneReflected(s.scene, s.camera, W, H, std::span<uint32_t>(img));
        check(WorldsByteEqual(world, before),
              "RENDER-ONLY: the fpx world is byte-identical before vs after the RT render (a pure read)");
    }

    if (g_fail == 0) std::printf("rt_simrender_test: all RT5 sim<->RT envelope invariants PASS\n");
    return g_fail == 0 ? 0 : 1;
}
