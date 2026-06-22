#pragma once
// Slice RT5 — Hardware Ray Tracing: DETERMINISM-ENVELOPE + LOCKSTEP TIE-IN (sim ⊗ RT). The SHARED
// sim->RT bridge for the RT5 showcase, used by BOTH the Vulkan (--rt5-simrender-shot, samples/
// hello_triangle/main.cpp) and the Metal (--rt5-simrender, metal_headless/visual_test.mm) showcases AND
// the pure-CPU test (tests/rt_simrender_test.cpp), so the converter is ONE source of truth.
//
// THE BRIDGE: a deterministic fpx rigid-body sim (a handful of spheres dropping + settling on a ground)
// run through fpx::RunLockstep, its settled bodies converted -> an rtrace::RtScene (a reflective ground
// AABB at primIndex 0 + each FxBody bi -> an RtSphere{center=bi.pos, radius=bi.radius, primIndex=1+i}),
// rendered by the FROZEN RT4 path (rt_reflect.comp on Vulkan HW + rtrace::RenderSceneReflected on the CPU).
//
// SEAM DISCIPLINE (the RT5 design constraint): this header lives under tests/ (NOT engine/) so that
// rtrace.h keeps NO sim dependency and fpx.h keeps NO render dependency. It #includes BOTH read-only.
// rtrace::FxVec3 IS fpx::FxVec3 (the same Q16.16 type — both `using` the fpx alias), so body.pos assigns
// directly to RtSphere.center. PURE — no GPU, no backend symbols, deterministic, ASan-clean.

#include <cstdint>
#include <span>
#include <vector>

#include "render/rtrace.h"  // RtScene/RtSphere/RtAabb/RtCamera + RenderSceneReflected + ReflectivityFor (FROZEN)
#include "sim/fpx.h"        // FxWorld/FxBody/RunLockstep/FxCommand (READ-ONLY)

namespace hf::render::rt5 {

namespace fpx = hf::sim::fpx;
namespace rt = hf::render::rtrace;

// ----- The deterministic RT5 sim parameters (PINNED — the cross-vendor golden + the lockstep proof rest
// on this exact config; mirrors the --fpx-shot/--fpx-lockstep-shot gravity/dt/iters precedent). ----------
inline constexpr int kRt5Bodies = 8;          // 8 dynamic spheres dropped above the ground
inline constexpr int kRt5Ticks  = 90;         // N ticks so the pile SETTLES (a resting pile is the visual)
inline constexpr int kRt5Iters  = 6;          // SimTick solve iterations (the --fpx-lockstep precedent)

// kRt5GravY: -9.8 host-snapped to Q16.16 (the --fpx-shot rounding idiom — deterministic on every host).
inline constexpr fpx::fx kRt5GravY = (fpx::fx)(-9.8 * (double)fpx::kOne + (-9.8 < 0 ? -0.5 : 0.5));
inline constexpr fpx::fx kRt5Dt    = fpx::kOne / 60;                   // 1/60 s timestep
inline constexpr fpx::fx kRt5Radius = (fpx::fx)(fpx::kOne * 70 / 100); // 0.70 sphere radius

// BuildRt5World(): the FIXED initial fpx world — kRt5Bodies dynamic spheres at varied x/z above a ground
// at groundY=0, varied staggered heights so they fall + collide + settle into a pile. Pure: two calls
// return byte-equal worlds. The replica re-derives this exact world from the SAME inputs.
inline fpx::FxWorld BuildRt5World() {
    fpx::FxWorld w;
    w.gravity = {0, kRt5GravY, 0};
    w.groundY = 0;
    // A 4x2 footprint (x in {-1.5,-0.5,0.5,1.5} scaled, z in two rows) so the bodies overlap-collide as
    // they fall and settle into a small heap centered on the ground. Heights staggered 3..6 world units.
    for (int i = 0; i < kRt5Bodies; ++i) {
        fpx::FxBody b;
        const int gx = i % 4;            // 0..3 column
        const int gz = i / 4;            // 0..1 row
        // x: columns at -3,-1,+1,+3 half-units => F(2*gx-3, 2) world units; z: rows at +2,+4 half-units.
        b.pos = {(fpx::fx)(((2 * gx - 3) * (int)fpx::kOne) / 2),
                 (fpx::fx)((3 + (i % 4)) * (int)fpx::kOne),    // staggered heights 3..6
                 (fpx::fx)(((2 + 2 * gz) * (int)fpx::kOne) / 2)};
        b.vel = {0, 0, 0};
        b.invMass = fpx::kOne;           // unit mass
        b.flags = fpx::kFlagDynamic;
        b.radius = kRt5Radius;
        b.orient = fpx::FxQuat{0, 0, 0, fpx::kOne};
        b.angVel = {0, 0, 0};
        w.bodies.push_back(b);
    }
    return w;
}

// BuildRt5Stream(): the FIXED command stream the lockstep peers are fed (inputs ONLY). EMPTY here — the
// closure is proven by the deterministic free-fall + settle alone (an input stream is not needed for the
// envelope; RT5 is out-of-scope for rollback — see the spec). Returned by value for symmetry with the
// other lockstep showcases (a future variant could script impulses here without touching the bridge).
inline std::vector<fpx::FxCommand> BuildRt5Stream() {
    return std::vector<fpx::FxCommand>{};
}

// ----- The sim->RT scene bridge (the PURE function the spec pins) -----------------------------------------
// Rt5Scene OWNS the primitive storage; scene.spheres/aabbs span into it, so keep it alive while tracing
// (the RtScene1 owning-struct precedent in rtrace.h).
struct Rt5Scene {
    std::vector<rt::RtSphere> spheres;
    std::vector<rt::RtAabb>   aabbs;
    rt::RtScene  scene;
    rt::RtCamera camera;
};

// BodiesToRtScene(world): the PURE converter — maps the settled fpx world to an RtScene. A large reflective
// ground AABB is primIndex 0 (ReflectivityFor(0)=0.55 -> the floor MIRRORS the pile); each dynamic FxBody
// bi -> RtSphere{center=bi.pos, radius=bi.radius, primIndex=1+i} (so a B-body world yields B+1 primitives:
// 1 ground AABB + B spheres). The camera/light/background are the PINNED RT2 camera (eye (0,2,-9), world-
// axis basis, halfW/H 0.70, the upper-front-right light, the cool sky-grey background). PURE: a function of
// world ONLY (no clock/RNG), so two calls on the same world are byte-equal — the provenance contract.
inline Rt5Scene BodiesToRtScene(const fpx::FxWorld& world) {
    using rt::fx;
    using rt::FxVec3;
    using rt::F;
    Rt5Scene r;

    // Ground AABB -> primIndex 0 (reflective via ReflectivityFor(0)). A large thin slab at the groundY
    // plane (the settled spheres rest on top with their centers at radius above groundY).
    r.aabbs.push_back(rt::RtAabb{FxVec3{F(-20, 1), F(-3, 1), F(-20, 1)},
                                 FxVec3{F(20, 1),  F(0, 1),  F(20, 1)},
                                 /*primIndex*/ 0});

    // Each FxBody bi -> RtSphere{center=bi.pos, radius=bi.radius, primIndex=1+i}. rtrace::FxVec3 IS
    // fpx::FxVec3, so body.pos assigns directly. primIndex 1+i (the ground owns 0).
    for (size_t i = 0; i < world.bodies.size(); ++i) {
        const fpx::FxBody& b = world.bodies[i];
        r.spheres.push_back(rt::RtSphere{b.pos, b.radius, (uint32_t)(1 + i)});
    }

    r.scene.spheres = std::span<const rt::RtSphere>(r.spheres);
    r.scene.aabbs   = std::span<const rt::RtAabb>(r.aabbs);
    r.scene.lightDir = rt::RtNormalize(FxVec3{F(4, 10), F(8, 10), F(-3, 10)});
    r.scene.background = rt::PackRGBA8(34, 40, 56, 255);

    // The PINNED RT2 camera (eye behind/above looking +Z; world-axis orthonormal basis; ~53deg FOV).
    r.camera.eye     = FxVec3{F(0, 1), F(2, 1), F(-9, 1)};
    r.camera.right   = FxVec3{rt::kOne, 0, 0};
    r.camera.up      = FxVec3{0, rt::kOne, 0};
    r.camera.forward = FxVec3{0, 0, rt::kOne};
    r.camera.halfW   = F(7, 10);
    r.camera.halfH   = F(7, 10);
    return r;
}

}  // namespace hf::render::rt5
