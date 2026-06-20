#pragma once
// Slice AC1 — Deterministic Active Ragdoll / Physical-Animation Blending: the ANGULAR POSE-DRIVE PRIMITIVE
// (the BEACHHEAD of FLAGSHIP #17, hf::sim::active). Flagship #15 (joint) built a ragdoll; flagship #16
// (vehicle) built the spring soft-constraint. AC1 adds the ONE genuinely-new primitive active ragdoll
// needs: an angular DRIVE-TO-TARGET torque — a motor that drives a joint's RELATIVE orientation toward a
// target quaternion by a stiffness fraction. INTEGER-bit-exact (int64 -> the active_drive_solve.comp
// shader is Vulkan-only + the Metal showcase runs THIS CPU reference, the JT2/VH1/CL3/FPX3 split).
// Single-thread Gauss-Seidel (drives are order-dependent). Pure CPU, header-only, NO device, NO backend
// symbols, NO <cmath>. Namespace hf::sim::active. #includes sim/joint.h + sim/fpx.h read-only ONLY
// (active.h is the additive sibling; joint.h + fpx.h are byte-FROZEN).
//
// THE DESIGN CALL: the drive is joint.h::SolveAngularLimit's inverse-mass nlerp APPLY (joint.h:299-304)
// with the swing-twist CONE CLAMP replaced by an NLERP-TOWARD-TARGET. SolveAngularLimit already computes
// qrel = FxQuatMul(QConj(qA), qB) (B's orientation in A's frame), clamps the swing into the joint cone
// producing qrelClamped, then rotates BOTH bodies (inverse-mass-weighted nlerp) so qA⁻¹·qB -> qrelClamped.
// AC1 keeps the APPLY verbatim and replaces the clamp with a drive: qrelDriven = normalize(nlerp(qrel,
// qTarget, stiffness)) — nlerp the current relative orientation toward the target by stiffness in [0,kOne]
// (the VH1 spring `stiffness` soft-constraint pattern lifted to orientation; stiffness 0 -> no drive,
// stiffness kOne -> snap to target). QNlerp + FxQuatNormalize are the existing joint.h int64 helpers.
//
// THE SHORTEST-ARC SIGN FIX (documented, deterministic): a quaternion double-covers SO(3) (q and -q are
// the SAME rotation), so a raw component-wise nlerp toward qTarget can take the LONG way around when qrel
// and qTarget are in opposite hemispheres. The fix is a deterministic, fixed-order sign flip: if
// FxDot4(qrel, qTarget) < 0 negate qTarget's components before the nlerp -> the drive always takes the
// shortest arc. Pure integer (FxDot4 is int64 fxmul terms), no RNG/branch on float -> bit-reproducible.
//
// THE int64 / glslc METAL LESSON (FPX3/CL3/JT2/VH1): SolveAngularDrive uses FxQuatMul/FxQuatNormalize/fxdiv
// (int64). DXC -spirv compiles int64 (the Int64 capability); glslc (the Metal HLSL->SPIR-V->MSL frontend)
// CANNOT parse int64_t in HLSL. So shaders/active_drive_solve.comp is VULKAN-SPIR-V-ONLY (NOT in the Metal
// hf_gen_msl list); the Metal --active-drive showcase runs THIS CPU StepDriveWorld over the same chain ->
// byte-identical to the Vulkan GPU result BY CONSTRUCTION (the joint_angular_solve.comp / vehicle_spring_
// solve.comp convention), while the Vulkan side carries the GPU==CPU bit-identity proof.
//
// HONEST CAVEAT (the JT2/VH1 caveat shape): the drive is a stiffness-scaled NLERP toward target (a SOFT
// angular constraint), NOT analytic motor mechanics / N·m torque limits / critical damping. The held angle
// is a DETERMINISTIC Gauss-Seidel residual (near-target within a band, more iters -> tighter, NOT exact);
// bones have no inertia tensor (the inherited fpx/JT caveat — a drive rotates the orient, it does not
// torque the angVel). The headline is DETERMINISM + cross-platform bit-identity (the UE5-Chaos
// differentiator: UE5's physical-animation drive is float/non-deterministic), NOT "more physically correct".

#include <cstdint>
#include <vector>

#include "sim/joint.h"  // read-only: FxJoint / SolveBallJoint / FxAngularLimit / SolveAngularLimit /
                        // QConj / QNlerp / FxDot / kJointBall / kAngularHinge / kAngularCone — the JT
                        // toolbox AC1's drive is built from (SolveAngularLimit's apply, verbatim).
#include "sim/fpx.h"    // read-only: fx / fxmul / fxdiv / FxQuat / FxBody / FxWorld / FxQuatMul /
                        // FxQuatNormalize / IntegrateBodyFull / kOne / kFrac / kFlagDynamic.

namespace hf::sim {
namespace active {

// Reuse the fpx Q16.16 + joint.h toolbox verbatim (NO new fixed-point primitives, NO new quaternion math).
using fpx::fx;
using fpx::FxVec3;
using fpx::FxQuat;
using fpx::FxBody;
using fpx::FxWorld;
using fpx::fxmul;
using fpx::fxdiv;
using fpx::FxQuatMul;
using fpx::FxQuatNormalize;
using joint::FxJoint;
using joint::FxAngularLimit;
using joint::SolveBallJoint;
using joint::SolveAngularLimit;
using joint::QConj;        // the unit-quaternion conjugate {-x,-y,-z,w} (joint.h:248)
using joint::QNlerp;       // the component-wise normalized lerp p + t*(q - p) (joint.h:258)
inline constexpr int kFrac = fpx::kFrac;   // Q16.16 fractional bits (MUST match the shader)
inline constexpr fx  kOne  = fpx::kOne;    // 1.0 in Q16.16 (65536)

// ----- FxDot4: the Q16.16 dot product of two quaternions (the shortest-arc sign test) -----------------
// joint.h::FxDot is a 3-vector dot; the drive's double-cover sign fix needs the FULL 4-component dot
// (x,y,z,w). int64 fxmul terms (the FxLength / joint::FxDot discipline). Pure integer -> bit-exact CPU<->GPU.
inline fx FxDot4(const FxQuat& a, const FxQuat& b) {
    return fxmul(a.x, b.x) + fxmul(a.y, b.y) + fxmul(a.z, b.z) + fxmul(a.w, b.w);
}

// ----- The angular-drive record (the std430 GPU mirror; the FxAngularLimit packing discipline) --------
// An FxAngularDrive drives the RELATIVE orientation of bodyB in bodyA's frame toward qTarget (a UNIT
// target quaternion, B-in-A's-frame) by `stiffness` in [0, kOne] (the per-iteration nlerp fraction:
// stiffness 0 -> no drive; stiffness kOne -> snap to qTarget; the VH1 spring soft-constraint pattern on
// orientation). std430-packable as plain int32s: bodyA, bodyB (2 x uint32) + qTarget.xyzw (4 x int32) +
// stiffness (1 x int32) + driveWeight (1 x int32, AC2) = 8 x 4-byte = 32 bytes. The AC1 GPU mirror ALREADY
// padded the 28-byte logical record to a 16-byte-aligned 32-byte stride (one trailing int `_pad`); AC2
// REPURPOSES that reserved pad slot as `driveWeight` so the std430 STRIDE IS UNCHANGED (no new shader file)
// — the intended in-flagship extension AC1 reserved the pad for. The host FxAngularDriveGpu carries
// driveWeight where it carried the pad; the C++ record is now the 32-byte logical record.
//
// AC2 driveWeight (the PER-JOINT PHYSICAL BLEND WEIGHT): the physical blend alpha in [0, kOne] that scales
// the MAGNITUDE of the applied inverse-mass correction (NOT stiffness: stiffness is the per-iteration nlerp
// RATE — a low-stiffness drive still converges over K iters; driveWeight is HOW MUCH of the correction is
// applied at all). kOne -> the full AC1 correction (the joint tracks the target — ACTIVE); 0 -> ZERO
// correction (pure physics — LIMP ragdoll, the drive is a no-op); intermediate -> a partial pull competing
// with gravity (a soft physical blend). DEFAULTS to kOne so AC1 call-sites are UNCHANGED in behavior
// (fxmul(w, kOne) == w in Q16.16 -> the AC1 apply byte-for-byte — the render-invariance contract).
struct FxAngularDrive {
    uint32_t bodyA = 0;               // index of the first body (the frame body — A's frame defines qTarget)
    uint32_t bodyB = 0;               // index of the second body (the driven body)
    FxQuat   qTarget;                 // UNIT target relative orientation qA⁻¹·qB (default identity {0,0,0,kOne})
    fx       stiffness = 0;           // Q16.16 per-iteration nlerp fraction in [0,kOne] (0 -> no drive)
    fx       driveWeight = kOne;      // AC2: Q16.16 physical blend alpha in [0,kOne] (kOne -> active/AC1, 0 -> limp)
};

// ----- SolveAngularDrive: drive ONE joint's relative orientation toward qTarget (the bit-exact core) ---
// VERBATIM joint.h::SolveAngularLimit's inverse-mass nlerp APPLY (joint.h:299-304) with the swing-twist
// cone clamp replaced by an nlerp-toward-target (the design call). active_drive_solve.comp copies THIS
// body. Reads + writes only the two bodies' orient. A pinned body (invMass 0 -> share 0) is NOT rotated.
//   (1) skip if out-of-range or wsum = a.invMass + b.invMass == 0.
//   (2) qrel = FxQuatMul(QConj(qA), qB)                         (B in A's frame — the SolveAngularLimit line)
//   (3) the SHORTEST-ARC sign fix: q = qTarget; if FxDot4(qrel, q) < 0 negate q (double-cover, deterministic)
//   (4) qrelDriven = FxQuatNormalize(QNlerp(qrel, q, stiffness)) (the DRIVE — replaces the cone clamp)
//   (5) the inverse-mass nlerp apply (VERBATIM SolveAngularLimit): qBtarget = FxQuatMul(qA, qrelDriven);
//       qAtarget = FxQuatMul(qB, QConj(qrelDriven)); wA/wB = fxdiv(invMass, wsum); nlerp qB/qA toward them.
// int64 (FxQuatMul/FxQuatNormalize/fxdiv/FxDot4) -> the shader is Vulkan-only (glslc can't parse int64).
inline void SolveAngularDrive(FxWorld& world, const FxAngularDrive& drv) {
    const size_t n = world.bodies.size();
    if (drv.bodyA >= (uint32_t)n || drv.bodyB >= (uint32_t)n) return;   // out-of-range -> skip
    FxBody& a = world.bodies[(size_t)drv.bodyA];
    FxBody& b = world.bodies[(size_t)drv.bodyB];
    const fx wsum = a.invMass + b.invMass;
    if (wsum == 0) return;                              // both pinned -> skip

    const FxQuat qA = a.orient;
    const FxQuat qB = b.orient;
    const FxQuat qrel = FxQuatMul(QConj(qA), qB);       // B's orientation in A's frame

    // --- the shortest-arc sign fix (quaternion double-cover; deterministic, fixed-order) ---
    FxQuat tgt = drv.qTarget;
    if (FxDot4(qrel, tgt) < 0) { tgt.x = -tgt.x; tgt.y = -tgt.y; tgt.z = -tgt.z; tgt.w = -tgt.w; }

    // --- the drive (replaces the cone clamp): nlerp qrel toward the target by stiffness ---
    const FxQuat qrelDriven = FxQuatNormalize(QNlerp(qrel, tgt, drv.stiffness));

    // --- the correction targets + the nlerp inverse-mass apply (VERBATIM SolveAngularLimit) ---
    const FxQuat qBtarget = FxQuatMul(qA, qrelDriven);           // qB such that qA⁻¹·qB == qrelDriven
    const FxQuat qAtarget = FxQuatMul(qB, QConj(qrelDriven));    // qA such that qA⁻¹·qB == qrelDriven
    const fx wA = fxdiv(a.invMass, wsum);
    const fx wB = fxdiv(b.invMass, wsum);
    // AC2: scale each apply share by the per-joint physical blend weight (kOne -> the full AC1 correction
    // [fxmul(w,kOne)==w, render-invariant]; 0 -> QNlerp(q, target, 0)==q -> no rotation -> pure physics/limp).
    const fx sA = fxmul(wA, drv.driveWeight);
    const fx sB = fxmul(wB, drv.driveWeight);
    b.orient = FxQuatNormalize(QNlerp(qB, qBtarget, sB));        // a pinned body (w 0) is NOT rotated
    a.orient = FxQuatNormalize(QNlerp(qA, qAtarget, sA));
}

// ----- DriveAngleCos: the held-to-target metric (the .w of qrel·qTarget⁻¹) — the SwingAngleCos shape ----
// qrel·qTarget⁻¹ is the residual rotation FROM the target TO the current relative orientation; its .w ==
// cos(half the residual angle). A held drive keeps this near kOne (qrel reached/held qTarget within the
// deterministic Gauss-Seidel residual). With the shortest-arc convention this is computed with the
// SAME sign-fixed target so the metric is in [−kOne, kOne] with kOne == on-target. Pure integer
// (FxQuatMul/FxDot4), no clamp/apply. int64 -> bit-exact CPU<->GPU. (joint.h::SwingAngleCos shape.)
inline fx DriveAngleCos(const FxWorld& world, const FxAngularDrive& drv) {
    const size_t n = world.bodies.size();
    if (drv.bodyA >= (uint32_t)n || drv.bodyB >= (uint32_t)n) return kOne;
    const FxQuat qA = world.bodies[(size_t)drv.bodyA].orient;
    const FxQuat qB = world.bodies[(size_t)drv.bodyB].orient;
    const FxQuat qrel = FxQuatMul(QConj(qA), qB);
    // residual = qrel · qTarget⁻¹ (qTarget unit -> inverse == conjugate); .w == cos(half residual angle).
    const FxQuat residual = FxQuatMul(qrel, QConj(drv.qTarget));
    return residual.w;
}

// ----- StepDriveWorld: one full active-drive step (integrate + K Gauss-Seidel {ball|limit|drive} + ground)
// The joint.h::StepArticulated mold (joint.h:332) with the DRIVE pass added to the Gauss-Seidel interleave:
//   (1) IntegrateBodyFull each body (FPX4 6-DOF semi-implicit-Euler) — VERBATIM fpx.h.
//   (2) `iters` Gauss-Seidel passes, EACH doing in FIXED order:
//         all SolveBallJoint   (position — the JT1 ball anchors)
//         all SolveAngularLimit(orientation cap — the JT2 cone/hinge; AC1's scene leaves this empty)
//         all SolveAngularDrive(orientation drive — the AC1 motor toward the target pose)
//   (3) ground floor clamp AFTER the constraint passes.
// SEQUENTIAL single-thread, fixed op order, no RNG/clock -> two-run bit-identical AND bit-exact GPU==CPU.
// active_drive_solve.comp runs THIS exact whole step (integrate + K {ball | limit | drive} + ground) so the
// GPU body world memcmp's bit-exact vs it. (The ball pass keeps the chain CONNECTED while the drive bends
// each joint to the target — the L-pose held against gravity.)
inline void StepDriveWorld(FxWorld& world, const std::vector<FxJoint>& joints,
                           const std::vector<FxAngularLimit>& angularLimits,
                           const std::vector<FxAngularDrive>& drives, fx dt, int iters) {
    const size_t n = world.bodies.size();
    // (1) integrate one step (6-DOF: translation gated by kFlagDynamic, orientation for every body).
    for (size_t i = 0; i < n; ++i)
        fpx::IntegrateBodyFull(world.bodies[i], world.gravity, dt);
    // (2) `iters` Gauss-Seidel passes: all ball joints, then all angular limits, then all angular drives.
    for (int it = 0; it < iters; ++it) {
        for (size_t e = 0; e < joints.size(); ++e)
            SolveBallJoint(world, joints[e]);
        for (size_t e = 0; e < angularLimits.size(); ++e)
            SolveAngularLimit(world, angularLimits[e]);
        for (size_t e = 0; e < drives.size(); ++e)
            SolveAngularDrive(world, drives[e]);
    }
    // (3) ground floor clamp AFTER the constraint passes (a joint may have pulled a body below ground).
    for (size_t i = 0; i < n; ++i) {
        FxBody& b = world.bodies[i];
        if ((b.flags & fpx::kFlagDynamic) && b.pos.y < world.groundY) b.pos.y = world.groundY;
    }
}

// ----- StepDriveWorldSteps: run K full active-drive steps (the showcase / GPU K-step driver) ------------
inline void StepDriveWorldSteps(FxWorld& world, const std::vector<FxJoint>& joints,
                                const std::vector<FxAngularLimit>& angularLimits,
                                const std::vector<FxAngularDrive>& drives, fx dt, int iters, int steps) {
    for (int s = 0; s < steps; ++s)
        StepDriveWorld(world, joints, angularLimits, drives, dt, iters);
}

}  // namespace active
}  // namespace hf::sim
