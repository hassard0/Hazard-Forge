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
#include "anim/skeleton.h"  // Slice JT4 (read-only): anim::Skeleton / anim::Joint {parent, inverseBind, t, r, s}
#include "math/math.h"      // Slice JT4 (read-only): math::Mat4 / math::Quat / math::Vec3 / FromTRS — the
                            // host float bind globals + the palette product. NO circular include: skeleton.h
                            // + math.h only pull math (no sim), and fpx.h already includes math.h.

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
using fpx::FxQuatMul;        // Slice JT2: the Hamilton product (the swing-twist recompose + qrel)
using fpx::FxQuatNormalize;  // Slice JT2: the int64 quaternion normalize (the nlerp + re-synth)
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

// ====================================================================================================
// Slice JT2 — ANGULAR LIMITS (hinge + cone): THE NEW-PHYSICS BEAT (the GR4-friction-equivalent).
// After the JT1 positional ball projection holds the anchors coincident, JT2 projects the RELATIVE
// orientation qrel = qA⁻¹·qB back into the joint's allowed cone/hinge via a quaternion SWING-TWIST
// decomposition + a host-cos CONE CLAMP — a quaternion analog of the GR4 Coulomb-cone clamp (clamp the
// off-axis rotation into a cone). A HINGE holds its axis (cone angle 0 -> swing forced to identity, only
// twist about `axis`); a CONE limits the swing to a host half-angle. INTEGER-bit-exact, int64 ->
// joint_angular_solve.comp Vulkan-only + the Metal showcase runs THIS CPU reference (the FPX4/CL3 split).
// Single-thread Gauss-Seidel. ADDITIVE — JT1's FxJoint/SolveBallJoint/StepJointWorld stay byte-frozen;
// FxAngularLimit is a SEPARATE record (a ragdoll edge carries BOTH a FxJoint and a FxAngularLimit).
//
// THE INTEGER-CLEAN DESIGN (ZERO runtime transcendentals): the cone half-angle is supplied as HOST
// CONSTANTS cosHalfLimit = cos(θ/2) + sinHalfLimit = sin(θ/2) (the caller computes them once in float at
// scene build; the SIM does ZERO acos/sin/cos). A HINGE is cosHalfLimit = kOne, sinHalfLimit = 0 (cone
// angle 0 -> swing forced to identity). A FREE control is cosHalfLimit = -kOne (a 180° cone -> never
// clamps). The swing re-synthesis at the limit is FxNormalize (int64 FxISqrt) + fxmul; the inverse-mass
// apply is NLERP (component lerp + FxQuatNormalize — the integer-friendly small-angle slerp approximation,
// the cloth/PBD precedent, NOT a slerp/quaternion-power). All int64 -> joint_angular_solve.comp copies
// SolveAngularLimit VERBATIM (Vulkan-only; glslc can't parse int64 in HLSL -> NOT in hf_gen_msl).
//
// HONEST CAVEAT (the GR4-angle-of-repose caveat shape): the swing-twist + host-cos clamp + nlerp apply is
// a DETERMINISTIC PROXY for an angular limit — exact-deterministic + bit-identical cross-backend (the
// headline), but NOT an analytic constraint-mechanics solution. The nlerp small-angle apply leaves a
// deterministic-but-nonzero residual (more Gauss-Seidel iters -> tighter); a within-band limit, not exact.

inline constexpr uint32_t kAngularHinge = 0u;  // cone limit 0 -> swing forced to identity (only twist about axis)
inline constexpr uint32_t kAngularCone  = 1u;  // swing limited to the cone half-angle (cosHalfLimit/sinHalfLimit)

// ----- The angular-limit record (the std430 GPU mirror; SEPARATE from FxJoint — JT1 frozen) -----------
// An FxAngularLimit limits the RELATIVE orientation of bodyB in bodyA's frame about a body-local `axis`
// (the hinge/cone axis, a UNIT vector in A's local frame). cosHalfLimit/sinHalfLimit are HOST-snapped
// Q16.16 constants (= cos(θ/2)/sin(θ/2); hinge = kOne/0; free = -kOne/0). kind selects hinge vs cone.
// std430-packable as plain int32s: bodyA, bodyB (2 x uint32) + axis.xyz (3 x int32) + cosHalfLimit +
// sinHalfLimit (2 x int32) + kind (uint32) = 8 x 4-byte = 32 bytes, NO padding holes (memcmp-able; the
// GPU FxAngularLimit mirror).
struct FxAngularLimit {
    uint32_t bodyA = 0;            // index of the first body (the frame body — A's frame defines `axis`)
    uint32_t bodyB = 0;            // index of the second body (the limited body)
    FxVec3   axis;                // UNIT body-local hinge/cone axis on bodyA
    fx       cosHalfLimit = kOne; // cos(θ/2) host constant (hinge=kOne, free=-kOne)
    fx       sinHalfLimit = 0;    // sin(θ/2) host constant (hinge=0)
    uint32_t kind = kAngularHinge;
};

// ----- The quaternion helpers JT2 adds in joint.h (do NOT modify fpx.h) -------------------------------
// QConj(q): the conjugate {-x,-y,-z,w} (the inverse of a UNIT quaternion). Pure negate, no int64.
inline FxQuat QConj(const FxQuat& q) { return FxQuat{-q.x, -q.y, -q.z, q.w}; }

// FxDot(a,b): the Q16.16 dot product of two 3-vectors (int64 fxmul terms, the FxLength discipline).
inline fx FxDot(const FxVec3& a, const FxVec3& b) {
    return fxmul(a.x, b.x) + fxmul(a.y, b.y) + fxmul(a.z, b.z);
}

// QNlerp(p, q, t): the component-wise normalized lerp p + t*(q - p) (NOT normalized here — the caller
// FxQuatNormalize's the result). The integer-friendly small-angle slerp approximation (the cloth/PBD
// precedent), each term an int64 fxmul. t in [0,kOne]; t=0 -> p, t=kOne -> q.
inline FxQuat QNlerp(const FxQuat& p, const FxQuat& q, fx t) {
    return FxQuat{p.x + fxmul(q.x - p.x, t), p.y + fxmul(q.y - p.y, t),
                  p.z + fxmul(q.z - p.z, t), p.w + fxmul(q.w - p.w, t)};
}

// ----- SolveAngularLimit: project ONE angular limit (the swing-twist + cone clamp + nlerp apply) -------
// VERBATIM the spec's swing-twist + host-cos cone clamp + inverse-mass nlerp apply (joint_angular_solve.comp
// copies THIS body). Reads the JT1-projected positions read-only; writes only the two bodies' orient. A
// pinned body (invMass 0 -> share 0) is NOT rotated. int64 (FxQuatMul/FxQuatNormalize/FxNormalize/fxdiv).
inline void SolveAngularLimit(FxWorld& world, const FxAngularLimit& lim) {
    const size_t n = world.bodies.size();
    if (lim.bodyA >= (uint32_t)n || lim.bodyB >= (uint32_t)n) return;   // out-of-range -> skip
    FxBody& a = world.bodies[(size_t)lim.bodyA];
    FxBody& b = world.bodies[(size_t)lim.bodyB];
    const fx wsum = a.invMass + b.invMass;
    if (wsum == 0) return;                              // both pinned -> skip

    const FxQuat qA = a.orient;
    const FxQuat qB = b.orient;
    const FxQuat qrel = FxQuatMul(QConj(qA), qB);       // B's orientation in A's frame

    // --- swing-twist decomposition about `axis` (the hinge/cone axis, body-local on A) ---
    const FxVec3 qrelXyz{qrel.x, qrel.y, qrel.z};
    const fx proj = FxDot(qrelXyz, lim.axis);           // qrel's rotation component along the axis
    FxQuat twist = FxQuatNormalize(
        FxQuat{fxmul(lim.axis.x, proj), fxmul(lim.axis.y, proj), fxmul(lim.axis.z, proj), qrel.w});
    FxQuat swing = FxQuatMul(qrel, QConj(twist));       // the off-axis part (qrel = swing * twist)

    // --- cone clamp the SWING (host cos/sin limit, NO acos) ---
    if (swing.w < lim.cosHalfLimit) {                   // swing half-angle exceeds the cone -> clamp
        const FxVec3 sxyz{swing.x, swing.y, swing.z};
        const fx slen = FxLength(sxyz);
        if (slen != 0) {                                // degenerate (swing≈identity) -> skip the re-synth
            const FxVec3 nhat = FxNormalize(sxyz);      // the swing rotation axis
            swing = FxQuat{fxmul(lim.sinHalfLimit, nhat.x), fxmul(lim.sinHalfLimit, nhat.y),
                           fxmul(lim.sinHalfLimit, nhat.z), lim.cosHalfLimit};   // re-synth at the limit
        }
    }                                                   // hinge: cosHalfLimit=kOne -> always clamps -> identity
    const FxQuat qrelClamped = FxQuatNormalize(FxQuatMul(swing, twist));   // recompose

    // --- the correction targets + the nlerp inverse-mass apply (NO slerp) ---
    const FxQuat qBtarget = FxQuatMul(qA, qrelClamped);           // qB such that qA⁻¹·qB == qrelClamped
    const FxQuat qAtarget = FxQuatMul(qB, QConj(qrelClamped));    // qA such that qA⁻¹·qB == qrelClamped
    const fx wA = fxdiv(a.invMass, wsum);
    const fx wB = fxdiv(b.invMass, wsum);
    b.orient = FxQuatNormalize(QNlerp(qB, qBtarget, wB));         // a pinned body (w 0) is NOT rotated
    a.orient = FxQuatNormalize(QNlerp(qA, qAtarget, wA));
}

// ----- SwingAngleCos: the swing .w (== cos(half-angle)) of one limit — the "within the cone" metric -----
// A held limit keeps this >= cosHalfLimit within an LSB band (the swing did NOT exceed the cone). For a
// hinge (cosHalfLimit=kOne) a held hinge drives this toward kOne (the door stayed in the hinge plane).
// Pure integer (the same swing-twist decompose as SolveAngularLimit, no clamp/apply). int64.
inline fx SwingAngleCos(const FxWorld& world, const FxAngularLimit& lim) {
    const size_t n = world.bodies.size();
    if (lim.bodyA >= (uint32_t)n || lim.bodyB >= (uint32_t)n) return kOne;
    const FxQuat qA = world.bodies[(size_t)lim.bodyA].orient;
    const FxQuat qB = world.bodies[(size_t)lim.bodyB].orient;
    const FxQuat qrel = FxQuatMul(QConj(qA), qB);
    const FxVec3 qrelXyz{qrel.x, qrel.y, qrel.z};
    const fx proj = FxDot(qrelXyz, lim.axis);
    const FxQuat twist = FxQuatNormalize(
        FxQuat{fxmul(lim.axis.x, proj), fxmul(lim.axis.y, proj), fxmul(lim.axis.z, proj), qrel.w});
    const FxQuat swing = FxQuatMul(qrel, QConj(twist));
    return swing.w;   // cos(half swing angle); >= cosHalfLimit means within the cone
}

// ----- StepArticulated: one full articulated step (JT1 ball + JT2 angular interleaved Gauss-Seidel) ----
// The StepJointWorld mold extended with the angular pass: (1) IntegrateBodyFull each body (FPX4 6-DOF) ->
// (2) K Gauss-Seidel passes EACH doing {all SolveBallJoint in FIXED order, then all SolveAngularLimit in
// FIXED order} -> (3) ground floor clamp. Each pass interleaves the positional (translate) + angular
// (rotate) projection; SEQUENTIAL single-thread (order-dependent — the cloth/JT1 discipline). Pure integer,
// fixed op order -> two-run bit-identical AND bit-exact GPU==CPU. joint_angular_solve.comp runs the angular
// pass (the ball pass reuses joint_ball_solve.comp's body); the showcase driver dispatches per pass.
inline void StepArticulated(FxWorld& world, const std::vector<FxJoint>& joints,
                            const std::vector<FxAngularLimit>& angularLimits, fx dt, int iters) {
    const size_t n = world.bodies.size();
    // (1) integrate one step (6-DOF: translation gated by kFlagDynamic, orientation for every body).
    for (size_t i = 0; i < n; ++i)
        fpx::IntegrateBodyFull(world.bodies[i], world.gravity, dt);
    // (2) `iters` Gauss-Seidel passes: all ball joints (position), then all angular limits (orientation).
    for (int it = 0; it < iters; ++it) {
        for (size_t e = 0; e < joints.size(); ++e)
            SolveBallJoint(world, joints[e]);
        for (size_t e = 0; e < angularLimits.size(); ++e)
            SolveAngularLimit(world, angularLimits[e]);
    }
    // (3) ground floor clamp AFTER the constraint passes (a joint may have pulled a body below ground).
    for (size_t i = 0; i < n; ++i) {
        FxBody& b = world.bodies[i];
        if ((b.flags & fpx::kFlagDynamic) && b.pos.y < world.groundY) b.pos.y = world.groundY;
    }
}

// ----- StepArticulatedSteps: run K full articulated steps (the showcase / GPU K-step driver) -----------
inline void StepArticulatedSteps(FxWorld& world, const std::vector<FxJoint>& joints,
                                 const std::vector<FxAngularLimit>& angularLimits, fx dt, int iters,
                                 int steps) {
    for (int s = 0; s < steps; ++s)
        StepArticulated(world, joints, angularLimits, dt, iters);
}

// ====================================================================================================
// Slice JT3 — ARTICULATED MULTI-BODY STEP: the JOINTS-MEET-CONTACTS tick (the coherent MECHANISM).
// JT2's StepArticulated already does integrate -> K {ball | angular} -> ground, so a jointed chain SWINGS
// and orients, but the links pass THROUGH each other and the ground (no rigid-body contacts). JT3 adds the
// MISSING piece — the fpx FPX2 broadphase + FPX3 sphere-sphere SolveContacts — so the links self-collide
// and rest on the floor as a COHERENT articulated structure. This is the fract.h FR4 StepFracture mold (a
// host-driven multi-pass driver over the EXISTING int64 shaders) with the two joint passes added; NO new
// shader, NO new RHI. ADDITIVE — JT1/JT2 (FxJoint/SolveBallJoint/SolveAngularLimit/StepArticulated/the
// joint shaders) stay BYTE-FROZEN; JT3 is only this step + its measure helper + the showcase.
//
// THE LOCKED TICK (StepArticulatedContacts):
//   (1) PREDICT:   IntegrateBodyFull(each body)                  // FPX4 6-DOF integrate (VERBATIM)
//   (2) JOINTS:    K Gauss-Seidel iters EACH {all SolveBallJoint | all SolveAngularLimit}  // JT1+JT2
//   (3) BROADPHASE:BuildPairs(world, off, pairs)                 // FPX2 integer broadphase, ONCE per tick
//   (4) CONTACTS:  fpx::StepWorld(world, pairs, dt=0, solveIters)// FPX3 ground + sphere non-penetration
// The chain swings (held by the ball joints, oriented by the angular limits) then the contact block parts
// the overlapping links + rests them on the ground -> a self-colliding settling mechanism. Pure integer,
// fixed op order -> two-run bit-identical AND bit-exact GPU==CPU.
//
// THE NO-NEW-SHADER GPU DECOMPOSITION (documented design call — the locked tick is structured so the
// EXISTING whole-step int64 shaders reproduce it BYTE-FOR-BYTE): the GPU showcase drives (per tick) the
// EXISTING joint_angular_solve.comp (steps=1, iters=K, groundY = a far-below SENTINEL so its internal floor
// clamp is a dead no-op) to do (1)+(2) integrate + K {ball | angular} with NO ground/contacts, then host-
// rebuilds the FPX2 pair list from the post-joint positions, then drives the EXISTING fpx_solve.comp (dt=0
// -> no translation integrate, the 9-int fpx body carries no orientation so NO renormalize side-effect,
// real groundY -> the ground clamp + K SolveContacts sweeps) for (3)+(4). The CPU StepArticulatedContacts
// here calls the EXACT SAME ops (IntegrateBodyFull + SolveBallJoint/SolveAngularLimit + fpx::StepWorld with
// dt=0), so the GPU body world memcmp's bit-exact vs it. WHY this structure (not a per-iteration {ball |
// angular | contacts} interleave): the existing shaders are WHOLE-STEP and integrate-suppression via dt=0
// is NOT bit-idempotent for orientation (FxQuatNormalize of an already-unit quaternion can drift 1 LSB), so
// a per-iteration GPU contact interleave over the whole-step joint shader would diverge from the CPU. The
// contacts-as-a-trailing-block tick is the SAME components (joints + FPX3 contacts) and settles a coherent
// chain identically, while staying bit-exact GPU==CPU with ZERO new shader (the JT3 hard constraint).
//
// HONEST CAVEATS (carried from fpx/fract): FPX3 SolveContacts is sphere-sphere + NO inertia tensor, so the
// links collide as SPHERES and a contact does NOT spin a body (angVel comes only from the integrate; the
// fract FR4 / fpx caveat). The Gauss-Seidel joint + contact residual is deterministic-but-nonzero (the
// cloth/JT1/JT2 within-band caveat). The headline is DETERMINISM + cross-platform bit-identity, NOT "more
// physically correct".

// ----- StepArticulatedContacts: ONE articulated multi-body tick (joints + fpx contacts) ----------------
// The (1)-(4) locked tick above. After the JT2 joint passes hold the chain together, BuildPairs broadphases
// the CURRENT positions ONCE and fpx::StepWorld(dt=0) resolves the ground + the self-collision contacts.
// Reuses JT1/JT2 (SolveBallJoint/SolveAngularLimit) + fpx (BuildPairs/StepWorld) VERBATIM. The contact
// block uses dt=0 so it does NOT re-integrate (the integrate already happened at (1)); fpx::StepWorld(dt=0)
// = IntegrateStep(dt=0) [the idempotent ground clamp + the vy-zero on a grounded body] + SolveContacts.
inline void StepArticulatedContacts(FxWorld& world, const std::vector<FxJoint>& joints,
                                    const std::vector<FxAngularLimit>& angularLimits, fx dt, int iters,
                                    int solveIters) {
    const size_t n = world.bodies.size();
    // (1) integrate one step (6-DOF: translation gated by kFlagDynamic, orientation for every body).
    for (size_t i = 0; i < n; ++i)
        fpx::IntegrateBodyFull(world.bodies[i], world.gravity, dt);
    // (2) K Gauss-Seidel joint passes: all ball joints (position), then all angular limits (orientation).
    for (int it = 0; it < iters; ++it) {
        for (size_t e = 0; e < joints.size(); ++e)
            SolveBallJoint(world, joints[e]);
        for (size_t e = 0; e < angularLimits.size(); ++e)
            SolveAngularLimit(world, angularLimits[e]);
    }
    // (3) FPX2 broadphase ONCE per tick over the post-joint positions.
    std::vector<uint32_t> perBodyOffset;
    std::vector<fpx::FxPair> pairs;
    fpx::BuildPairs(world, perBodyOffset, pairs);
    // (4) FPX3 contacts: ground + sphere-sphere non-penetration (dt=0 -> no re-integrate). VERBATIM fpx.
    fpx::StepWorld(world, std::span<const fpx::FxPair>(pairs), /*dt=*/0, solveIters);
}

// ----- StepArticulatedContactsSteps: run K full articulated-multi-body ticks (the showcase driver) ------
inline void StepArticulatedContactsSteps(FxWorld& world, const std::vector<FxJoint>& joints,
                                         const std::vector<FxAngularLimit>& angularLimits, fx dt, int iters,
                                         int solveIters, int steps) {
    for (int s = 0; s < steps; ++s)
        StepArticulatedContacts(world, joints, angularLimits, dt, iters, solveIters);
}

// ----- ArticulatedState + MeasureArticulated: the honest coherence metrics ------------------------------
// The deterministic state of a settled articulated mechanism (all pure integer -> bit-exact CPU<->GPU):
//   maxAnchorGap     : the max world-anchor gap over all joints (small -> the links stayed CONNECTED).
//   meanDynamicY     : the mean pos.y of the DYNAMIC bodies (the settle line; > groundY after resting).
//   minDynamicBottom : the lowest (pos.y - radius) over dynamic bodies (>= groundY -> nothing buried).
//   residualOverlaps : the FPX3 residual-overlap count over the CURRENT broadphase pairs (links + ground
//                      non-penetrating means the contact pass HELD; a coherence stat, NOT necessarily 0).
//   dynamic          : the dynamic-body count (the denominator for meanDynamicY).
struct ArticulatedState {
    fx       maxAnchorGap = 0;
    fx       meanDynamicY = 0;
    fx       minDynamicBottom = 0;
    uint32_t residualOverlaps = 0;
    uint32_t dynamic = 0;
};

inline ArticulatedState MeasureArticulated(const FxWorld& world, const std::vector<FxJoint>& joints) {
    ArticulatedState st;
    st.maxAnchorGap = MaxAnchorGap(world, joints);
    // Mean dynamic pos.y + the lowest dynamic bottom (pos.y - radius).
    int64_t sumY = 0;
    uint32_t dyn = 0;
    bool haveBottom = false;
    for (const FxBody& b : world.bodies) {
        if (!(b.flags & fpx::kFlagDynamic)) continue;
        sumY += (int64_t)b.pos.y;
        const fx bottom = b.pos.y - b.radius;
        if (!haveBottom || bottom < st.minDynamicBottom) { st.minDynamicBottom = bottom; haveBottom = true; }
        ++dyn;
    }
    st.dynamic = dyn;
    st.meanDynamicY = dyn > 0u ? (fx)(sumY / (int64_t)dyn) : 0;
    // Residual overlaps over the CURRENT broadphase pairs (the FPX3 non-penetration coherence stat).
    std::vector<uint32_t> off;
    std::vector<fpx::FxPair> pairs;
    fpx::BuildPairs(world, off, pairs);
    st.residualOverlaps = fpx::CountResidualOverlaps(world, std::span<const fpx::FxPair>(pairs));
    return st;
}

// ====================================================================================================
// Slice JT4 — SKELETON->RAGDOLL BIND: THE PILLAR-BRIDGE (the physics moat meets the anim pillar).
// JT1-JT3 built a generic jointed mechanism; JT4 maps an anim::Skeleton onto it — each bone becomes an
// fpx::FxBody, each parent-child edge a FxJoint (ball) + FxAngularLimit (cone) — so the float animation
// skeleton becomes a bit-exact RAGDOLL that collapses under gravity into a settled pose, and the pose
// reads back as a joint palette (for the JT6 skinning path). NO new shader: the bind (float->Q16.16) +
// the palette read-back (Q16.16->float) are HOST conversions; the COLLAPSE sim is the bit-exact JT3
// StepArticulatedContacts (that integer sim is the GPU==CPU + cross-vendor-zero claim — the golden
// renders the collapsed INTEGER body positions). ADDITIVE — JT1/JT2/JT3 stay byte-FROZEN.
//
// THE THREE PIECES (all host-float for the bind/read-back; the collapse is the JT3 integer driver):
//   (1) RagdollFromSkeleton(skeleton, cfg): the float->Q16.16 bind. Forward-accumulate each joint's
//       model-space bind global (global[j] = (parent<0?I:global[parent])·TRS(t,r,s) over the topological
//       order — the SAME single forward pass anim::PaletteFromLocalPose uses, MIRRORED) for translation,
//       AND the global ROTATION quaternion (gq[j] = (parent<0?jr:gq[parent]·jr), float Hamilton product).
//       One fpx::FxBody per joint (pos = global translation·worldScale snapped to Q16.16, orient = gq[j]
//       -> FxQuat, radius = cfg.boneRadius, invMass = root? cfg.rootStatic-gated : cfg.invMass), and per
//       NON-ROOT edge a FxJoint (ball, anchors = the child joint's bind pos in each body's LOCAL frame via
//       FxRotate(QConj(orient), worldJointPos − bodyPos)) + a FxAngularLimit (cone from cfg).
//   (2) The collapse: StepArticulatedContactsSteps (JT3, VERBATIM) — bit-exact integer SIM.
//   (3) PoseToPalette(skeleton, world): the Q16.16->float read-back. palette[j] =
//       FxBodyTransform(body[j]) · skeleton.joints[j].inverseBind (the standard global·inverseBind matrix).
//
// HONEST CAVEATS (documented): bones are capsule-as-SPHERE proxies (the fpx sphere-contact caveat — a
// contact does not spin a body; angVel comes only from the integrate). The bind/read-back float crossings
// are DETERMINISTIC (no RNG/clock) but are NOT part of the bit-exact loop — they are render-only-adjacent
// (the FPX6/FL6 host-float-bridge shape). The bind assumes ~unit joint scale for the orientation chain
// (the rotation quaternion ignores TRS scale/shear; translation honours the full TRS Mat4 forward pass).

// ----- The host quaternion helpers JT4 needs (float, deterministic; do NOT modify math.h) -------------
// QuatMulF(a,b): the float Hamilton product a·b (the global-rotation forward accumulate, mirroring the
// Mat4 global[j]=global[parent]·local[j] for the rotation component).
inline math::Quat QuatMulF(const math::Quat& a, const math::Quat& b) {
    return math::Quat{
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z};
}

// ----- RagdollConfig: the bind recipe (host constants — the documented single float crossing at bind) --
// worldScale  : Q16.16 scale applied to the skeleton's model-space translations (skeleton units -> world).
// boneRadius  : the capsule-as-sphere proxy radius (Q16.16) for every bone body.
// invMass     : the dynamic bone inverse mass (Q16.16); the root is invMass 0 iff rootStatic.
// coneCos/Sin : cos(θ/2)/sin(θ/2) HOST constants for each edge's FxAngularLimit cone (the JT2 convention;
//               free = -kOne/0). gravity/groundY seed the FxWorld for the collapse.
// rootStatic  : pin the root body (invMass 0) — a pinned-root ragdoll dangles; a free root collapses fully.
struct RagdollConfig {
    fx       worldScale = kOne;            // skeleton-unit -> world scale
    fx       boneRadius = kOne / 4;        // 0.25 capsule-as-sphere proxy radius
    fx       invMass    = kOne;            // dynamic bone inverse mass
    fx       coneCos    = -kOne;           // cos(θ/2): default a 180° free cone (never clamps)
    fx       coneSin    = 0;               // sin(θ/2)
    FxVec3   gravity{0, (fx)(-9 * (int)kOne - kOne * 8 / 10), 0};  // (0, -9.8, 0) Q16.16
    fx       groundY    = 0;               // the floor the ragdoll collapses onto
    bool     rootStatic = true;            // pin the root (dangle) vs free (full collapse)
};

// ----- Ragdoll: the bound articulation (an FxWorld + its joint/limit graph) ----------------------------
struct Ragdoll {
    fpx::FxWorld                world;     // one FxBody per skeleton joint (skeleton order)
    std::vector<FxJoint>        joints;    // one ball joint per non-root edge
    std::vector<FxAngularLimit> limits;    // one cone limit per non-root edge (parallel to `joints`)
};

// ----- RagdollFromSkeleton: the float->Q16.16 bind (host, outside the bit-exact loop) ------------------
// Builds the Ragdoll from the skeleton's bind pose. Mirrors anim::PaletteFromLocalPose's single forward
// pass for translation (global[j] = global[parent]·FromTRS(t,r,s)) and accumulates the global rotation
// quaternion in parallel. One body per joint; per non-root edge a ball joint pinning the child's bind
// pivot (its global translation) to the parent + a cone angular limit. Pure deterministic host float.
inline Ragdoll RagdollFromSkeleton(const anim::Skeleton& skeleton, const RagdollConfig& cfg) {
    Ragdoll rag;
    const size_t n = skeleton.joints.size();

    // (1) the bind globals: translation via the Mat4 forward pass (mirroring anim), rotation via the
    //     float quaternion forward accumulate (both topologically valid — parent precedes child).
    std::vector<math::Mat4> global(n);
    std::vector<math::Quat> gquat(n);
    for (size_t j = 0; j < n; ++j) {
        const anim::Joint& jt = skeleton.joints[j];
        const math::Mat4 local = math::FromTRS(jt.t, jt.r, jt.s);
        const int parent = jt.parent;
        global[j] = (parent >= 0) ? (global[(size_t)parent] * local) : local;
        gquat[j]  = (parent >= 0) ? QuatMulF(gquat[(size_t)parent], jt.r) : jt.r;
    }

    // The Q16.16 world position of joint j (its global translation · worldScale).
    auto worldPos = [&](size_t j) -> FxVec3 {
        return FxVec3{(fx)((int64_t)(global[j].m[12] * (float)kOne) * (int64_t)cfg.worldScale / (int64_t)kOne),
                      (fx)((int64_t)(global[j].m[13] * (float)kOne) * (int64_t)cfg.worldScale / (int64_t)kOne),
                      (fx)((int64_t)(global[j].m[14] * (float)kOne) * (int64_t)cfg.worldScale / (int64_t)kOne)};
    };
    // The Q16.16 orientation FxQuat of joint j (its global rotation quaternion, normalized in float).
    auto worldOrient = [&](size_t j) -> FxQuat {
        const math::Quat q = math::Normalize(gquat[j]);
        return FxQuatNormalize(FxQuat{(fx)(q.x * (float)kOne), (fx)(q.y * (float)kOne),
                                      (fx)(q.z * (float)kOne), (fx)(q.w * (float)kOne)});
    };

    // (2) one body per joint.
    rag.world.gravity = cfg.gravity;
    rag.world.groundY = cfg.groundY;
    rag.world.bodies.resize(n);
    for (size_t j = 0; j < n; ++j) {
        FxBody b;
        b.pos = worldPos(j);
        b.vel = FxVec3{0, 0, 0};
        const bool isRoot = (skeleton.joints[j].parent < 0);
        const bool pinned = isRoot && cfg.rootStatic;
        b.invMass = pinned ? 0 : cfg.invMass;
        b.flags   = pinned ? 0u : fpx::kFlagDynamic;
        b.radius  = cfg.boneRadius;
        b.orient  = worldOrient(j);
        b.angVel  = FxVec3{0, 0, 0};
        rag.world.bodies[j] = b;
    }

    // (3) per non-root edge: a ball joint (pin the child's bind pivot = its global translation) + a cone.
    for (size_t j = 0; j < n; ++j) {
        const int parent = skeleton.joints[j].parent;
        if (parent < 0) continue;                       // root: no incoming edge
        const FxVec3 jointWorld = worldPos(j);          // the child joint's bind pivot in world space
        const FxBody& pa = rag.world.bodies[(size_t)parent];
        const FxBody& pb = rag.world.bodies[j];
        // anchor in each body's LOCAL frame: FxRotate(QConj(orient), worldJointPos − bodyPos).
        FxJoint e;
        e.bodyA = (uint32_t)parent;
        e.bodyB = (uint32_t)j;
        e.anchorA = FxRotate(QConj(pa.orient), FxSub(jointWorld, pa.pos));
        e.anchorB = FxRotate(QConj(pb.orient), FxSub(jointWorld, pb.pos));
        e.kind = kJointBall;
        rag.joints.push_back(e);
        // the cone limit: axis = the bone's bind direction (child − parent) in the PARENT's local frame.
        const FxVec3 boneWorld = FxSub(pb.pos, pa.pos);
        FxVec3 axisLocal = FxRotate(QConj(pa.orient), boneWorld);
        const fx alen = FxLength(axisLocal);
        axisLocal = (alen != 0) ? FxNormalize(axisLocal) : FxVec3{0, kOne, 0};  // degenerate -> +Y
        FxAngularLimit lim;
        lim.bodyA = (uint32_t)parent;
        lim.bodyB = (uint32_t)j;
        lim.axis = axisLocal;
        lim.cosHalfLimit = cfg.coneCos;
        lim.sinHalfLimit = cfg.coneSin;
        lim.kind = kAngularCone;
        rag.limits.push_back(lim);
    }
    return rag;
}

// ----- PoseToPalette: the Q16.16->float read-back (host; the standard skinning palette) ----------------
// For each joint j, palette[j] = FxBodyTransform(body[j]) · skeleton.joints[j].inverseBind — the body's
// current world transform (the FPX6 Q16.16->float bridge) times the joint's inverse-bind matrix. A pure
// deterministic function of the bit-exact body state (the provenance proof: rebuild from the settled world
// reproduces the palette byte-for-byte). One Mat4 per joint, in skeleton order.
inline std::vector<math::Mat4> PoseToPalette(const anim::Skeleton& skeleton, const fpx::FxWorld& world) {
    const size_t n = skeleton.joints.size();
    std::vector<math::Mat4> palette(n);
    for (size_t j = 0; j < n; ++j) {
        const math::Mat4 bodyGlobal = (j < world.bodies.size())
                                          ? fpx::FxBodyTransform(world.bodies[j])
                                          : math::Mat4::Identity();
        palette[j] = bodyGlobal * skeleton.joints[j].inverseBind;
    }
    return palette;
}

// ====================================================================================================
// Slice JT6 — LIT 3D SKINNED RENDER CAPSTONE (the PILLAR-BRIDGE money-shot; COMPLETES FLAGSHIP #15). The
// bit-exact ragdoll pose drives the EXISTING GPU skinned render: RagdollFromSkeleton -> collapse (JT3
// StepArticulatedContacts) -> the pose reads back as a joint palette -> the SAME lit_skinned pipeline the
// anim FSM uses renders the character POSED BY PHYSICS. NO new shader, NO new RHI; the ONLY float crossing
// is the JT4 PoseToPalette read-back. JT6 is purely additive — JT1-JT5 stay byte-frozen.
//
// RagdollToPalette: a thin convenience that names the JT6 bridge — the settled ragdoll's joint palette,
// flattened for the skinning pipeline IS exactly PoseToPalette(skeleton, ragdoll.world) (the std::vector
// <math::Mat4> the showcase copies into the fixed-size JointPalette UBO in skeleton order). It exists only
// to spell out the bridge at the call site; the showcase may call PoseToPalette directly with no change.
inline std::vector<math::Mat4> RagdollToPalette(const anim::Skeleton& skeleton, const Ragdoll& ragdoll) {
    return PoseToPalette(skeleton, ragdoll.world);
}

// ----- RagdollState + MeasureRagdoll: the honest ragdoll-collapse metrics (deterministic) --------------
// maxAnchorGap : the max world-anchor gap over the ragdoll's joints (bones stayed CONNECTED — small).
// meanBodyY    : the mean pos.y over ALL bodies (the collapsed/settle line; drops from the bind pose).
// minBottom    : the lowest (pos.y − radius) over dynamic bodies (>= groundY -> nothing buried).
// bones/edges  : the body count (== joint count) + the joint/limit edge count.
struct RagdollState {
    fx       maxAnchorGap = 0;
    fx       meanBodyY    = 0;
    fx       minBottom    = 0;
    uint32_t bones = 0;
    uint32_t edges = 0;
};

inline RagdollState MeasureRagdoll(const anim::Skeleton& /*skeleton*/, const Ragdoll& rag) {
    RagdollState st;
    st.maxAnchorGap = MaxAnchorGap(rag.world, rag.joints);
    st.bones = (uint32_t)rag.world.bodies.size();
    st.edges = (uint32_t)rag.joints.size();
    int64_t sumY = 0;
    bool haveBottom = false;
    for (const FxBody& b : rag.world.bodies) {
        sumY += (int64_t)b.pos.y;
        if (b.flags & fpx::kFlagDynamic) {
            const fx bottom = b.pos.y - b.radius;
            if (!haveBottom || bottom < st.minBottom) { st.minBottom = bottom; haveBottom = true; }
        }
    }
    st.meanBodyY = st.bones > 0u ? (fx)(sumY / (int64_t)st.bones) : 0;
    return st;
}

// ====================================================================================================
// Slice JT5 — LOCKSTEP + ROLLBACK over the ragdoll collapse (THE NETCODE HEADLINE — pure CPU). Additive
// over JT1-JT4 (ALL code above is byte-unchanged). THE DESIGN: the ragdoll world IS an fpx::FxWorld
// (the Ragdoll.world bone bodies). fpx already ships the bit-exact FPX5 lockstep/rollback machinery over
// FxWorld — fpx::FxCommand (an input impulse/spin on a body), fpx::ApplyCommand, fpx::SnapshotWorld
// (deep-copy the world), fpx::RestoreWorld. JT5 REUSES ALL of it VERBATIM and changes ONE thing: the
// per-tick step is joint::StepArticulatedContacts (JT3: integrate -> K {ball|angular} joint passes ->
// broadphase -> contacts -> ground) instead of fpx's StepWorld. So JT5 is a THIN harness — the direct
// FR5 twin (fract.h::SimFractTick/RunFractLockstep/RunFractRollback over StepFracture), with the extra
// joints/angularLimits/iters/solveIters params threaded through. Pure integer, fixed op order ->
// bit-identical on every peer/platform (and cross-backend by StepArticulatedContacts's proven bit-
// exactness — JT3 proved it; JT5 only sequences it via inputs).
//
// WHAT IS REPLAYED: the ragdoll TOPOLOGY (bones + joints + cone limits) is the deterministic `init`
// (JT4 RagdollFromSkeleton, bit-reproducible) + the const joints/angularLimits (CONSTANT across ticks,
// passed by const-ref, NOT part of the snapshot). JT5 replays the COLLAPSE DYNAMICS — every bone's fall,
// swing, tumble, self-collision, and settle — from the input impact stream, bit-for-bit. The command is
// a body impulse (kCmdImpulse, "punch a bone") / spin (kCmdSetAngVel) on a ragdoll bone, re-simulated
// identically on two peers; a peer with only the impacts re-derives the EXACT slumped pose; a rollback
// corrects a mispredicted punch.
//
// SEAM DISCIPLINE: unchanged — ZERO backend symbols, header-only, PURE CPU. NO new shader, NO new RHI.
// JT5 only ADDS the three harness functions; it reuses fpx.h's FxCommand/ApplyCommand/SnapshotWorld/
// RestoreWorld (already #included read-only) — it does NOT re-implement them.

// Re-export the fpx FPX5 command + snapshot primitives (read-only — REUSED VERBATIM, not re-implemented).
using fpx::FxCommand;
using fpx::kCmdImpulse;
using fpx::kCmdSetAngVel;
using fpx::ApplyCommand;
using fpx::SnapshotWorld;
using fpx::RestoreWorld;

// SimRagdollTick(world, joints, angularLimits, stream, tick, dt, iters, solveIters): the deterministic
// per-tick step (the fpx::SimTick twin with StepArticulatedContacts substituted for StepWorld). (1) apply
// ALL `stream` commands whose .tick == `tick`, in ARRAY ORDER (the deterministic input-order contract —
// the same order on every peer/platform) via fpx::ApplyCommand; (2) StepArticulatedContacts(world, joints,
// angularLimits, dt, iters, solveIters) — the JT3 articulated tick, reused VERBATIM. Pure integer, fixed
// order -> bit-identical on every peer/platform. (No new command type — reuses fpx::FxCommand.)
inline void SimRagdollTick(fpx::FxWorld& world, const std::vector<FxJoint>& joints,
                           const std::vector<FxAngularLimit>& angularLimits,
                           const std::vector<fpx::FxCommand>& stream, uint32_t tick, fx dt, int iters,
                           int solveIters) {
    for (const fpx::FxCommand& c : stream)
        if (c.tick == tick) fpx::ApplyCommand(world, c);
    StepArticulatedContacts(world, joints, angularLimits, dt, iters, solveIters);
}

// RunRagdollLockstep(init, joints, angularLimits, stream, ticks, dt, iters, solveIters): THE peer entry
// point (the fpx::RunLockstep control flow over SimRagdollTick). Run `ticks` SimRagdollTicks from a COPY
// of `init`, applying the command stream -> the final collapsed ragdoll world. authority =
// RunRagdollLockstep(init, ..., stream, N); replica = RunRagdollLockstep(init, ..., stream, N) from the
// SAME init + stream (INPUTS ONLY — no state shared) -> BIT-IDENTICAL by determinism (the lockstep proof
// memcmps them). joints/angularLimits are the CONSTANT topology, passed by const-ref (not snapshotted).
inline fpx::FxWorld RunRagdollLockstep(const fpx::FxWorld& init, const std::vector<FxJoint>& joints,
                                       const std::vector<FxAngularLimit>& angularLimits,
                                       const std::vector<fpx::FxCommand>& stream, int ticks, fx dt,
                                       int iters, int solveIters) {
    fpx::FxWorld w = init;
    for (int t = 0; t < ticks; ++t)
        SimRagdollTick(w, joints, angularLimits, stream, (uint32_t)t, dt, iters, solveIters);
    return w;
}

// RunRagdollRollback(init, joints, angularLimits, authStream, mispredictStream, ticks, mispredictTick,
// dt, iters, solveIters): the rollback harness (the fpx::RunRollback control flow over SimRagdollTick).
// (1) run ticks 0..mispredictTick from `init` applying authStream; (2) SAVE a snapshot AT mispredictTick
// (fpx::SnapshotWorld — the FxWorld deep-copy); (2b) speculatively advance <=3 ticks with the MISPREDICTED
// stream (the wrong input — the client prediction that diverges); (3) ROLLBACK — fpx::RestoreWorld to the
// snapshot + RE-SIMULATE mispredictTick..ticks with the CORRECT authStream -> the corrected final world.
// The proof asserts this == RunRagdollLockstep(init, ..., authStream, ticks) (rollback corrected the
// misprediction EXACTLY) AND that the speculative pre-rollback state DIFFERED from authority (a real
// divergence was fixed). Reuses fpx::SnapshotWorld/RestoreWorld VERBATIM. joints/angularLimits const-ref.
inline fpx::FxWorld RunRagdollRollback(const fpx::FxWorld& init, const std::vector<FxJoint>& joints,
                                       const std::vector<FxAngularLimit>& angularLimits,
                                       const std::vector<fpx::FxCommand>& authStream,
                                       const std::vector<fpx::FxCommand>& mispredictStream, int ticks,
                                       int mispredictTick, fx dt, int iters, int solveIters) {
    fpx::FxWorld w = init;
    // (1) advance 0..mispredictTick with the authoritative stream.
    for (int t = 0; t < mispredictTick; ++t)
        SimRagdollTick(w, joints, angularLimits, authStream, (uint32_t)t, dt, iters, solveIters);
    // (2) SAVE the snapshot at mispredictTick (the rollback restore point).
    const fpx::FxWorld snap = fpx::SnapshotWorld(w);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (bounded to the remaining ticks).
    int specTicks = ticks - mispredictTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s)
        SimRagdollTick(w, joints, angularLimits, mispredictStream, (uint32_t)(mispredictTick + s), dt,
                       iters, solveIters);
    // (3) ROLLBACK: restore the snapshot + re-simulate mispredictTick..ticks with the CORRECT authStream.
    fpx::RestoreWorld(w, snap);
    for (int t = mispredictTick; t < ticks; ++t)
        SimRagdollTick(w, joints, angularLimits, authStream, (uint32_t)t, dt, iters, solveIters);
    return w;
}

}  // namespace joint
}  // namespace hf::sim
