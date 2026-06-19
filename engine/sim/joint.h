#pragma once
// Slice JT1 — Deterministic Articulated-Body Ragdoll: the JOINT GRAPH + the BALL-JOINT CONSTRAINT
// (the BEACHHEAD of FLAGSHIP #15: DETERMINISTIC ARTICULATED-BODY RAGDOLL, hf::sim::joint — the rigid
// solver fpx.h is already 6-DOF but has NO bilateral constraint tying two bodies; JT1 adds the MISSING
// PRIMITIVE, a BALL JOINT pinning two bodies' anchors coincident, which turns "piles of independent
// spheres" into a CHAIN / mechanism). INTEGER-bit-exact (int64 -> the joint_ball_solve.comp shader is
// Vulkan-only + the Metal showcase runs THIS CPU reference, the FPX3/cloth CL3 split). Single-thread
// Gauss-Seidel (joints are order-dependent). Pure CPU, header-only, NO device, NO backend symbols,
// NO <cmath>. Namespace hf::sim::joint. #includes sim/fpx.h read-only (joint.h is the additive sibling).
//
// THE DESIGN CALL: a ball joint == cloth::SolveDistanceConstraint (cloth.h:324) over BODY anchors,
// restLen 0. A FxJoint ties two fpx::FxBody's at a pair of body-local anchor points (anchorA/anchorB,
// Q16.16 offsets from each body's centre). The world anchor of body b is
//   worldAnchor = b.pos + fpx::FxRotate(b.orient, anchorLocal)   (FxRotate exists, fpx.h:440)
// The BALL constraint pins the two world anchors COINCIDENT (a distance-0 constraint), resolved by the
// inverse-mass-weighted POSITIONAL correction of the two body centres — LITERALLY the cloth distance
// constraint with restLen = 0, the constraint endpoints being the world anchors, the correction applied
// to b.pos (the cloth "wsum, wi/wj via fxdiv, FxLength/FxNormalize direction, pinned-skip, pos += n·
// fxmul(pen,w)" projection). The joint_ball_solve.comp shader copies SolveBallJoint VERBATIM so the GPU
// exercises the EXACT integer ops -> the GPU==CPU memcmp catches any divergence.
//
// THE JT1 SIMPLIFICATION (honest, documented): JT1 is the POSITIONAL ball joint ONLY. It satisfies the
// anchor constraint by TRANSLATING the body centres (the cloth positional split). It does NOT yet rotate
// a body to align an OFF-centre anchor (no lever-arm / angular coupling — that is the JT2/JT3 angular
// work). With anchors at the link ends, a translation-only ball joint already produces a hanging,
// swinging CHAIN (each link's anchor pulled to its neighbour's); orientation is driven by
// IntegrateBodyFull's angular integrate, NOT yet by the joint. A translation-only ball joint still hangs
// a chain. Inertia tensor / lever-arm torque / hinge / cone-twist limits are out of scope (JT2+).
//
// THE int64 / glslc METAL LESSON (FPX3/CL3): SolveBallJoint uses FxRotate's fxmul + fxdiv + FxLength
// (int64). DXC -spirv compiles int64 (the Int64 capability); glslc (the Metal HLSL->SPIR-V->MSL frontend)
// CANNOT parse int64_t in HLSL. So shaders/joint_ball_solve.comp is VULKAN-SPIR-V-ONLY (NOT in the Metal
// hf_gen_msl list); the Metal --joint-ball showcase runs THIS CPU SolveBallJoint over the same chain ->
// byte-identical to the Vulkan GPU result BY CONSTRUCTION (the fpx_solve.comp / cloth_solve.comp
// convention), while the Vulkan side carries the GPU==CPU bit-identity proof.
//
// HONEST CAVEAT (cloth/FPX3-identical): Gauss-Seidel is ITERATIVE — after `iters` passes the residual
// anchor gap is DETERMINISTIC but NOT zero (stiffness ∝ iterations); fxdiv/FxISqrt truncation makes the
// solver bit-REPRODUCIBLE, not analytically exact. The headline is DETERMINISM + cross-platform
// bit-identity (the UE5-Chaos differentiator), NOT "more physically correct".

#include <cstdint>
#include <vector>

#include "sim/fpx.h"   // read-only: fx / fxmul / fxdiv / FxVec3 / FxAdd / FxSub / FxScale / FxLength /
                       // FxNormalize / FxRotate / FxBody / FxWorld / IntegrateBodyFull / kOne / kFlagDynamic

namespace hf::sim {
namespace joint {

// Reuse the fpx Q16.16 toolbox verbatim (NO new fixed-point primitives).
using fpx::fx;
using fpx::FxVec3;
using fpx::FxQuat;
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
inline constexpr int kFrac = fpx::kFrac;      // Q16.16 fractional bits (MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;       // 1.0 in Q16.16 (65536)

// kJointBall = the BALL joint kind (the two anchors pinned coincident). Hinge / cone-twist are JT2 (the
// FxJoint::kind discriminant + the FxJoint::limit field are carried now so the record layout is final
// from the beachhead; JT1 only solves kJointBall, ignores `limit`).
inline constexpr uint32_t kJointBall = 0u;

// ----- The joint record (the std430 GPU mirror; the FxBody packing discipline) --------------------
// A single joint ties body bodyA to body bodyB at the body-local anchor offsets anchorA/anchorB (Q16.16
// offsets from each body's centre, rotated into the body's current orientation by FxRotate). kind selects
// the constraint (kJointBall at JT1). `limit` is carried for JT2 (hinge/cone angular limit) and UNUSED at
// JT1. std430-packable as plain int32s: bodyA, bodyB (2 x uint32) + anchorA.xyz + anchorB.xyz (6 x int32)
// + kind (uint32) + limit (int32) = 10 x 4-byte = 40 bytes, NO padding holes (memcmp-able; the GPU
// FxJoint mirror).
struct FxJoint {
    uint32_t bodyA = 0;       // index of the first body in FxWorld::bodies
    uint32_t bodyB = 0;       // index of the second body
    FxVec3   anchorA;         // Q16.16 body-local anchor offset on bodyA (rotated by bodyA.orient)
    FxVec3   anchorB;         // Q16.16 body-local anchor offset on bodyB (rotated by bodyB.orient)
    uint32_t kind  = kJointBall;  // kJointBall (JT1); hinge / cone-twist are JT2
    fx       limit = 0;       // Q16.16 angular limit (UNUSED at JT1; carried for JT2)
};

// ----- WorldAnchor: the body-local anchor offset rotated into world space + the body centre -----------
// worldAnchor = body.pos + FxRotate(body.orient, anchorLocal). For an identity orientation this is
// body.pos + anchorLocal (FxRotate of identity is the identity map). Pure integer (FxRotate is fxmul-only),
// int64-backed. The deterministic world position the BALL constraint pins coincident.
inline FxVec3 WorldAnchor(const FxBody& body, const FxVec3& anchorLocal) {
    return FxAdd(body.pos, FxRotate(body.orient, anchorLocal));
}

// ----- SolveBallJoint: project ONE ball joint (the bit-exact core) ------------------------------------
// The VERBATIM generalization of cloth::SolveDistanceConstraint (cloth.h:324) to restLen 0 over the world
// anchors, the correction applied to the body CENTRES (the JT1 translation-only simplification):
//   pa   = WorldAnchor(a, j.anchorA);  pb = WorldAnchor(b, j.anchorB)
//   d    = pb - pa
//   len  = FxLength(d); if len == 0 -> skip (coincident; no deterministic normal)
//   wsum = a.invMass + b.invMass; if wsum == 0 -> skip (both pinned)
//   n    = FxNormalize(d); pen = len - 0 = len            (restLen 0: pull the anchors together)
//   wa   = fxdiv(a.invMass, wsum); wb = fxdiv(b.invMass, wsum)
//   a.pos += n * fxmul(pen, wa)    (A's centre moves toward the midpoint)
//   b.pos -= n * fxmul(pen, wb)    (B's centre moves toward the midpoint)
// Pinned bodies (invMass 0) take share 0 and never move. Out-of-range body indices -> skip (deterministic).
// int64-backed (FxRotate fxmul + fxdiv + FxLength). The shader copies THIS body VERBATIM.
inline void SolveBallJoint(FxWorld& world, const FxJoint& j) {
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
    const fx pen = len;                                 // restLen 0: the whole gap is the penetration
    const FxVec3 nrm = FxNormalize(d);
    const fx wa = fxdiv(a.invMass, wsum);
    const fx wb = fxdiv(b.invMass, wsum);
    const FxVec3 ca = FxScale(nrm, fxmul(pen, wa));     // A moves +ca (toward B's anchor)
    const FxVec3 cb = FxScale(nrm, fxmul(pen, wb));     // B moves -cb (toward A's anchor)
    a.pos = FxAdd(a.pos, ca);
    b.pos = FxSub(b.pos, cb);
}

// ----- AnchorGap: the deterministic world-anchor gap |pb - pa| of one joint (a coherence metric) -------
// The Q16.16 distance between the two world anchors. A satisfied (held) ball joint keeps this within a
// small deterministic band (the Gauss-Seidel residual is nonzero-but-deterministic — the cloth/FPX3
// caveat). Pure integer FxLength -> bit-exact CPU<->GPU.
inline fx AnchorGap(const FxWorld& world, const FxJoint& j) {
    const size_t n = world.bodies.size();
    if (j.bodyA >= (uint32_t)n || j.bodyB >= (uint32_t)n) return 0;
    const FxVec3 pa = WorldAnchor(world.bodies[(size_t)j.bodyA], j.anchorA);
    const FxVec3 pb = WorldAnchor(world.bodies[(size_t)j.bodyB], j.anchorB);
    return FxLength(FxSub(pb, pa));
}

// MaxAnchorGap(world, joints): the deterministic max world-anchor gap over all joints (the chain-connected
// proof metric — small after settling means the links stayed connected, NOT flying apart). Pure integer.
inline fx MaxAnchorGap(const FxWorld& world, const std::vector<FxJoint>& joints) {
    fx m = 0;
    for (const FxJoint& j : joints) {
        const fx g = AnchorGap(world, j);
        if (g > m) m = g;
    }
    return m;
}

// ----- StepJointWorld: one full articulated step (integrate + K Gauss-Seidel ball passes + ground) -----
// The StepCloth mold (cloth.h:353) over fpx::FxWorld:
//   (1) IntegrateBodyFull each body (the FPX4 6-DOF semi-implicit-Euler: dynamic bodies get vel += gravity
//       *dt; pos += vel*dt; every body's orientation integrates from angVel — so the chain links carry
//       orientation). NO ground clamp here (it is applied after the constraint passes).
//   (2) `iters` Gauss-Seidel ball passes, EACH iterating ALL joints in the FIXED joint-list order applying
//       SolveBallJoint. SEQUENTIAL — each joint reads the body centres the EARLIER joints THIS pass already
//       moved -> order-dependent -> single-thread on the GPU (the cloth CL3 / fpx FPX3 discipline). Pinned
//       bodies (invMass 0) never move.
//   (3) ground floor-clamp: every dynamic body with pos.y < groundY snaps pos.y = groundY (the
//       non-penetration floor; the constraint passes can pull a body below ground, so clamp AFTER them).
// The pinned root holds + the ball joints keep the links connected -> the chain HANGS / SWINGS under
// gravity (unlike independent free-fall). Pure integer, fixed op order, no RNG/clock -> two-run bit-
// identical AND bit-exact GPU==CPU. joint_ball_solve.comp runs THIS exact body (the integrate is the
// host's IntegrateBodyFull; the per-pass SolveBallJoint loop is the dispatched shader).
inline void StepJointWorld(FxWorld& world, const std::vector<FxJoint>& joints, fx dt, int iters) {
    // (1) integrate one step (6-DOF: translation gated by kFlagDynamic, orientation for every body).
    const size_t n = world.bodies.size();
    for (size_t i = 0; i < n; ++i)
        fpx::IntegrateBodyFull(world.bodies[i], world.gravity, dt);
    // (2) `iters` Gauss-Seidel ball passes in the FIXED joint order (sequential -> order-dependent).
    for (int it = 0; it < iters; ++it)
        for (size_t e = 0; e < joints.size(); ++e)
            SolveBallJoint(world, joints[e]);
    // (3) ground floor clamp AFTER the constraint passes (a joint may have pulled a body below ground).
    for (size_t i = 0; i < n; ++i) {
        FxBody& b = world.bodies[i];
        if ((b.flags & fpx::kFlagDynamic) && b.pos.y < world.groundY) b.pos.y = world.groundY;
    }
}

// ----- StepJointWorldSteps: run K full articulated steps (the showcase / GPU K-step driver) -------------
// K successive StepJointWorld steps over the chain. The GPU driver runs the integrate on the host then
// dispatches joint_ball_solve.comp once per Gauss-Seidel pass per step (the cloth_solve per-pass driver);
// the CPU reference here is the exact byte-for-byte target.
inline void StepJointWorldSteps(FxWorld& world, const std::vector<FxJoint>& joints, fx dt, int iters,
                                int steps) {
    for (int s = 0; s < steps; ++s)
        StepJointWorld(world, joints, dt, iters);
}

}  // namespace joint
}  // namespace hf::sim
