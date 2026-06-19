#pragma once
// Slice VH1 — Deterministic Vehicle Physics: the SUSPENSION SPRING JOINT (the BEACHHEAD of FLAGSHIP #16:
// DETERMINISTIC VEHICLE PHYSICS, hf::sim::vehicle). A vehicle is the JT articulated mechanism specialized
// and the ONE genuinely-new primitive it needs is a SUSPENSION SPRING JOINT: a damped distance constraint
// that holds two bodies at a REST LENGTH (not coincident like the JT1 ball joint), the
// cloth::SolveDistanceConstraint mold with restLen != 0 + a stiffness scale + relative-velocity damping.
// This is the chassis<->wheel suspension. INTEGER-bit-exact (int64 -> shaders/vehicle_spring_solve.comp is
// Vulkan-only + the Metal showcase runs THIS CPU reference, the FPX3/CL3/JT1 split). Single-thread
// Gauss-Seidel (springs are order-dependent). Pure CPU, header-only, NO device, NO backend symbols, NO
// <cmath>. Namespace hf::sim::vehicle. #includes sim/joint.h + sim/fpx.h read-only (vehicle.h is the
// additive sibling).
//
// THE DESIGN CALL: a damped spring == joint::FxJoint (the JT1 ball joint) with restLen != 0 + a stiffness
// scale + velocity damping. A FxSpringJoint ties two fpx::FxBody's at a pair of body-local anchor points
// (anchorA/anchorB, Q16.16 offsets from each body's centre) but holds them at a REST LENGTH `restLen`
// along the anchor-to-anchor line, with a `stiffness` (the per-iteration correction fraction — a soft
// constraint, like the cloth PBD stiffness) and a `damping` (the relative-velocity damping along the
// spring normal). The world anchor of body b is WorldAnchor(b, anchorLocal) = b.pos +
// fpx::FxRotate(b.orient, anchorLocal) (REUSE joint::WorldAnchor). The solve is
// cloth::SolveDistanceConstraint (cloth.h:324) GENERALIZED:
//   pa = WorldAnchor(a, j.anchorA);  pb = WorldAnchor(b, j.anchorB)
//   d  = pb - pa;  len = FxLength(d);  if (len == 0) skip      // coincident -> no deterministic normal
//   n   = FxNormalize(d);  pen = len - j.restLen               // != 0: the spring REST LENGTH (JT1 used 0)
//   wsum = invMassA + invMassB;  if (wsum == 0) skip
//   wa  = fxdiv(invMassA, wsum);  wb = fxdiv(invMassB, wsum)
//   // (1) POSITIONAL spring restore (stiffness-scaled — a soft constraint; stiffness in [0,kOne]):
//   corr = fxmul(pen, j.stiffness)                              // partial correction toward restLen / iter
//   a.pos += FxScale(n, fxmul(corr, wa));  b.pos -= FxScale(n, fxmul(corr, wb))
//   // (2) NORMAL-velocity DAMPING (the spring damper — converts oscillation to heat deterministically):
//   vRelN = FxDot(FxSub(b.vel, a.vel), n)                      // relative velocity along the spring normal
//   dvel  = fxmul(vRelN, j.damping)                            // the damped impulse magnitude
//   a.vel += FxScale(n, fxmul(dvel, wa));  b.vel -= FxScale(n, fxmul(dvel, wb))
// All int64 (FxLength/FxNormalize/FxDot/fxmul/fxdiv). The vehicle_spring_solve.comp shader copies
// SolveSpringJoint VERBATIM so the GPU exercises the EXACT integer ops -> the GPU==CPU memcmp catches any
// divergence.
//
// THE int64 / glslc METAL LESSON (FPX3/CL3/JT1): SolveSpringJoint uses FxRotate's fxmul + fxdiv + FxLength
// (int64). DXC -spirv compiles int64 (the Int64 capability); glslc (the Metal HLSL->SPIR-V->MSL frontend)
// CANNOT parse int64_t in HLSL. So shaders/vehicle_spring_solve.comp is VULKAN-SPIR-V-ONLY (NOT in the
// Metal hf_gen_msl list); the Metal --vehicle-spring showcase runs THIS CPU SolveSpringJoint over the same
// scene -> byte-identical to the Vulkan GPU result BY CONSTRUCTION (the joint_ball_solve.comp /
// cloth_solve.comp convention), while the Vulkan side carries the GPU==CPU bit-identity proof.
//
// HONEST CAVEAT (cloth/JT-identical): the suspension stiffness is ∝ iterations — after `iters` Gauss-Seidel
// passes the spring length is a DETERMINISTIC band around restLen but NOT exactly restLen (a stiff spring
// under heavy load sags within a deterministic-but-nonzero band; tune stiffness/iters, NOT zero residual);
// fxdiv/FxISqrt truncation makes the solver bit-REPRODUCIBLE, not analytically exact. The headline is
// DETERMINISM + cross-platform bit-identity (the UE5-Chaos differentiator), NOT analytic spring mechanics.

#include <cstdint>
#include <vector>

#include "sim/fpx.h"     // read-only: fx / fxmul / fxdiv / FxVec3 / FxAdd / FxSub / FxScale / FxLength /
                         // FxNormalize / FxRotate / FxBody / FxWorld / IntegrateBodyFull / kOne / kFlagDynamic
#include "sim/joint.h"   // read-only: joint::WorldAnchor (the body-local anchor -> world) + joint::FxDot
                         // (the Q16.16 3-vector dot, added by JT2). vehicle.h is the additive sibling.

namespace hf::sim {
namespace vehicle {

// Reuse the fpx Q16.16 toolbox + the JT helpers verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::FxBody;
using fpx::FxWorld;
using fpx::fxmul;
using fpx::fxdiv;
using fpx::FxAdd;
using fpx::FxSub;
using fpx::FxScale;
using fpx::FxLength;
using fpx::FxNormalize;
using fpx::FxRotate;
using joint::WorldAnchor;   // worldAnchor = b.pos + FxRotate(b.orient, anchorLocal) (JT1, read-only)
using joint::FxDot;         // the Q16.16 3-vector dot (JT2, read-only)
inline constexpr int kFrac = fpx::kFrac;      // Q16.16 fractional bits (MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;       // 1.0 in Q16.16 (65536)

// ----- The spring-joint record (the std430 GPU mirror; the FxJoint packing discipline) -----------------
// A single spring ties body bodyA to body bodyB at the body-local anchor offsets anchorA/anchorB (Q16.16
// offsets from each body's centre, rotated into the body's current orientation by FxRotate), holding the
// two world anchors at distance `restLen` along the anchor-to-anchor line with a `stiffness` (the soft
// per-iteration correction fraction, stiffness in [0,kOne]) and a `damping` (the relative-velocity damping
// along the spring normal, in [0,kOne]). std430-packable as plain int32s: bodyA, bodyB (2 x uint32) +
// anchorA.xyz + anchorB.xyz (6 x int32) + restLen + stiffness + damping (3 x int32) = 11 x 4-byte = 44
// bytes, NO padding holes (memcmp-able; the GPU FxSpringJoint mirror).
struct FxSpringJoint {
    uint32_t bodyA = 0;       // index of the first body in FxWorld::bodies
    uint32_t bodyB = 0;       // index of the second body
    FxVec3   anchorA;         // Q16.16 body-local anchor offset on bodyA (rotated by bodyA.orient)
    FxVec3   anchorB;         // Q16.16 body-local anchor offset on bodyB (rotated by bodyB.orient)
    fx       restLen   = 0;   // Q16.16 spring rest length (the held anchor-to-anchor distance; != 0)
    fx       stiffness = 0;   // Q16.16 per-iteration positional correction fraction (soft constraint)
    fx       damping   = 0;   // Q16.16 normal-velocity damping fraction (the spring damper)
};

// ----- SpringLength: the current anchor-to-anchor length of one spring (a coherence metric / the proof) -
// The Q16.16 distance between the two world anchors. A settled spring keeps this within a small
// deterministic band of restLen (the Gauss-Seidel residual is nonzero-but-deterministic — the cloth/JT
// caveat). Pure integer FxLength -> bit-exact CPU<->GPU.
inline fx SpringLength(const FxWorld& world, const FxSpringJoint& j) {
    const size_t n = world.bodies.size();
    if (j.bodyA >= (uint32_t)n || j.bodyB >= (uint32_t)n) return 0;
    const FxVec3 pa = WorldAnchor(world.bodies[(size_t)j.bodyA], j.anchorA);
    const FxVec3 pb = WorldAnchor(world.bodies[(size_t)j.bodyB], j.anchorB);
    return FxLength(FxSub(pb, pa));
}

// ----- SolveSpringJoint: project ONE damped spring (the bit-exact core) -------------------------------
// VERBATIM the spec's positional rest-length restore + normal-velocity damping (vehicle_spring_solve.comp
// copies THIS body):
//   pa = WorldAnchor(a, j.anchorA); pb = WorldAnchor(b, j.anchorB)
//   d  = pb - pa; len = FxLength(d); if len == 0 -> skip (coincident; no deterministic normal)
//   n  = FxNormalize(d); pen = len - j.restLen           (restLen != 0: the spring deflection)
//   wsum = a.invMass + b.invMass; if wsum == 0 -> skip   (both pinned)
//   wa = fxdiv(a.invMass, wsum); wb = fxdiv(b.invMass, wsum)
//   corr = fxmul(pen, j.stiffness)                       (soft per-iteration restore)
//   a.pos += n*fxmul(corr, wa); b.pos -= n*fxmul(corr, wb)
//   vRelN = FxDot(b.vel - a.vel, n); dvel = fxmul(vRelN, j.damping)
//   a.vel += n*fxmul(dvel, wa); b.vel -= n*fxmul(dvel, wb)
// Pinned bodies (invMass 0) take share 0 and never move. Out-of-range body indices -> skip (deterministic).
// int64-backed (FxRotate fxmul + fxdiv + FxLength + FxDot). The shader copies THIS body VERBATIM.
inline void SolveSpringJoint(FxWorld& world, const FxSpringJoint& j) {
    const size_t n = world.bodies.size();
    if (j.bodyA >= (uint32_t)n || j.bodyB >= (uint32_t)n) return;   // out-of-range -> skip
    FxBody& a = world.bodies[(size_t)j.bodyA];
    FxBody& b = world.bodies[(size_t)j.bodyB];
    const fx wsum = a.invMass + b.invMass;
    if (wsum == 0) return;                              // both pinned -> skip
    const FxVec3 pa = WorldAnchor(a, j.anchorA);
    const FxVec3 pb = WorldAnchor(b, j.anchorB);
    const FxVec3 d = FxSub(pb, pa);
    const fx len = FxLength(d);
    if (len == 0) return;                               // coincident -> no deterministic normal -> skip
    const FxVec3 nrm = FxNormalize(d);
    const fx pen = len - j.restLen;                     // restLen != 0: the spring deflection
    const fx wa = fxdiv(a.invMass, wsum);
    const fx wb = fxdiv(b.invMass, wsum);
    // (1) POSITIONAL spring restore (stiffness-scaled soft constraint).
    const fx corr = fxmul(pen, j.stiffness);
    a.pos = FxAdd(a.pos, FxScale(nrm, fxmul(corr, wa)));   // A moves toward restLen
    b.pos = FxSub(b.pos, FxScale(nrm, fxmul(corr, wb)));   // B moves toward restLen
    // (2) NORMAL-velocity DAMPING (the spring damper).
    const fx vRelN = FxDot(FxSub(b.vel, a.vel), nrm);   // relative velocity along the spring normal
    const fx dvel = fxmul(vRelN, j.damping);
    a.vel = FxAdd(a.vel, FxScale(nrm, fxmul(dvel, wa)));
    b.vel = FxSub(b.vel, FxScale(nrm, fxmul(dvel, wb)));
}

// ----- StepSpringWorld: one full spring step (integrate + K Gauss-Seidel spring passes + ground) -------
// The joint::StepJointWorld mold over fpx::FxWorld:
//   (1) IntegrateBodyFull each body (the FPX4 6-DOF semi-implicit-Euler: dynamic bodies get vel +=
//       gravity*dt; pos += vel*dt; every body's orientation integrates from angVel). NO ground clamp here
//       (it is applied after the constraint passes).
//   (2) `iters` Gauss-Seidel spring passes, EACH iterating ALL springs in the FIXED spring-list order
//       applying SolveSpringJoint. SEQUENTIAL — each spring reads the body state the EARLIER springs THIS
//       pass already moved -> order-dependent -> single-thread on the GPU (the cloth CL3 / JT1 discipline).
//       Pinned bodies (invMass 0) never move.
//   (3) ground floor-clamp: every dynamic body with pos.y < groundY snaps pos.y = groundY (the
//       non-penetration floor; the constraint passes can pull a body below ground, so clamp AFTER them).
// The pinned anchor holds + the spring holds the body at ~restLen + the damper kills the oscillation -> the
// body BOBS then SETTLES at rest at ~restLen below the anchor. Pure integer, fixed op order, no RNG/clock
// -> two-run bit-identical AND bit-exact GPU==CPU. vehicle_spring_solve.comp runs THIS exact body (the
// integrate is the host's IntegrateBodyFull; the per-pass SolveSpringJoint loop is the dispatched shader).
inline void StepSpringWorld(FxWorld& world, const std::vector<FxSpringJoint>& springs, fx dt, int iters) {
    const size_t n = world.bodies.size();
    // (1) integrate one step (6-DOF: translation gated by kFlagDynamic, orientation for every body).
    for (size_t i = 0; i < n; ++i)
        fpx::IntegrateBodyFull(world.bodies[i], world.gravity, dt);
    // (2) `iters` Gauss-Seidel spring passes in the FIXED spring order (sequential -> order-dependent).
    for (int it = 0; it < iters; ++it)
        for (size_t e = 0; e < springs.size(); ++e)
            SolveSpringJoint(world, springs[e]);
    // (3) ground floor clamp AFTER the constraint passes (a spring may have pulled a body below ground).
    for (size_t i = 0; i < n; ++i) {
        FxBody& b = world.bodies[i];
        if ((b.flags & fpx::kFlagDynamic) && b.pos.y < world.groundY) b.pos.y = world.groundY;
    }
}

// ----- StepSpringWorldSteps: run K full spring steps (the showcase / GPU K-step driver) ----------------
// K successive StepSpringWorld steps. The GPU driver runs the integrate on the host then dispatches
// vehicle_spring_solve.comp once per Gauss-Seidel pass per step (the joint_ball_solve per-pass driver);
// the CPU reference here is the exact byte-for-byte target.
inline void StepSpringWorldSteps(FxWorld& world, const std::vector<FxSpringJoint>& springs, fx dt,
                                 int iters, int steps) {
    for (int s = 0; s < steps; ++s)
        StepSpringWorld(world, springs, dt, iters);
}

}  // namespace vehicle
}  // namespace hf::sim
