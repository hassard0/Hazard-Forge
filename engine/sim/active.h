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
#include <cstring>   // std::memcmp (the AC5 lockstep bit-identity assert)
#include <vector>

#include "anim/animation.h"  // read-only (AC3 — the pillar bridge): SampleLocalPose / Animation / JointPose.r
#include "anim/skeleton.h"   // read-only (AC3): anim::Skeleton / anim::Joint (the source the ragdoll binds from)
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

// ====================================================================================================
// Slice AC3 — THE ANIM-TARGET STEP (THE PILLAR BRIDGE). AC1 built the angular drive-to-target primitive;
// AC2 the per-joint blend weight. AC3 unites the anim pillar (engine/anim/) and the physics moat: each tick
// SAMPLE an anim clip into its per-bone LOCAL rotations -> WRITE those into the joints' drive qTargets -> run
// the AC1/AC2 integer drive step, so a JT4 ragdoll TRACKS an animation clip via physics torques while still
// colliding/yielding. The clip-sample + the float->Q16.16 qTarget snap are a documented DETERMINISTIC FLOAT
// crossing OUTSIDE the bit-exact loop (the JT4 bind shape) — computed HOST-side IDENTICALLY for the GPU and
// CPU paths, so the only thing the GPU/CPU bit-exactly reproduce is the integer StepDriveWorld over those
// shared qTargets (the AC1 contract, now with per-tick-varying targets). AC3 APPENDS — AC1/AC2 byte-frozen.
// NO new shader (the drive solve is AC1's active_drive_solve.comp; the sample+snap is host C++), NO new RHI.

// ----- FxQuatFromFloat: round-to-nearest float->Q16.16 quaternion snap (the JT4/BuildPileWorld idiom) -----
// The drive's qTarget is a UNIT quaternion in Q16.16. SampleLocalPose returns a float math::Quat (the bone's
// local rotation = qTarget); snap each component round-to-nearest, ties away from zero — the SAME host snap
// convention RagdollFromSkeleton / fpx::BuildPileWorld use ((fx)(v*kOne + (v<0?-0.5:0.5))) so the AC3
// targets and the JT4 bind agree. Pure host float, deterministic (no RNG/clock) -> bit-reproducible.
inline FxQuat FxQuatFromFloat(const math::Quat& q) {
    auto snap = [](double v) -> fx { return (fx)(v * (double)kOne + (v < 0 ? -0.5 : 0.5)); };
    return FxQuat{snap(q.x), snap(q.y), snap(q.z), snap(q.w)};
}

// ----- ActiveRagdoll: a JT4 joint::Ragdoll + a parallel drive per non-root edge (the AC3 binding) ----------
// `ragdoll` is the verbatim JT4 bind (one body per skeleton joint + one ball joint + one cone limit per
// non-root edge); `drives` is ONE FxAngularDrive per `ragdoll.joints[e]` (same non-root-edge ordering),
// driving that joint's relative orientation toward the clip-sampled target each tick. drives.size() ==
// ragdoll.joints.size(). The child bone of edge e is ragdoll.joints[e].bodyB (RagdollFromSkeleton pushes one
// edge per non-root joint j with bodyA=parent, bodyB=j — so joints[e].bodyB IS the skeleton joint index whose
// LOCAL rotation is the qTarget for that edge). WriteClipTargets fills the qTargets; ActiveFromSkeleton builds.
struct ActiveRagdoll {
    joint::Ragdoll               ragdoll;   // the JT4 bodies + ball joints + cone limits (REUSED VERBATIM)
    std::vector<FxAngularDrive>  drives;    // one drive per ragdoll.joints[e] (same non-root-edge order)
};

// ----- ActiveFromSkeleton: bind the ragdoll + one drive per non-root edge ------------------------------
// Builds joint::RagdollFromSkeleton(skeleton, ragdollCfg) (the bodies + ball joints + cone limits — the host
// float->Q16.16 bind), then a parallel drive per `ragdoll.joints[e]` with bodyA/bodyB = joints[e].bodyA/bodyB
// (parent/child), the given per-iteration `stiffness`, and driveWeight = driveWeightFn(e) (AC2 — default kOne,
// so callers can make some bones limp). qTarget is left identity; WriteClipTargets fills it each tick.
// `driveWeightFn` is any callable e(size_t)->fx; the default (nullptr-equivalent) is kOne for every edge.
template <typename DriveWeightFn>
inline ActiveRagdoll ActiveFromSkeleton(const anim::Skeleton& skeleton, const joint::RagdollConfig& ragdollCfg,
                                        fx stiffness, DriveWeightFn driveWeightFn) {
    ActiveRagdoll act;
    act.ragdoll = joint::RagdollFromSkeleton(skeleton, ragdollCfg);
    act.drives.reserve(act.ragdoll.joints.size());
    for (size_t e = 0; e < act.ragdoll.joints.size(); ++e) {
        FxAngularDrive d;
        d.bodyA = act.ragdoll.joints[e].bodyA;    // the parent body (the frame body — A's frame defines qTarget)
        d.bodyB = act.ragdoll.joints[e].bodyB;    // the child body (the driven body); also the child bone index
        d.qTarget = FxQuat{0, 0, 0, kOne};        // identity until WriteClipTargets fills it from the clip
        d.stiffness = stiffness;
        d.driveWeight = driveWeightFn(e);
        act.drives.push_back(d);
    }
    return act;
}
// Overload: default driveWeight kOne for every edge (the all-active bind).
inline ActiveRagdoll ActiveFromSkeleton(const anim::Skeleton& skeleton, const joint::RagdollConfig& ragdollCfg,
                                        fx stiffness) {
    return ActiveFromSkeleton(skeleton, ragdollCfg, stiffness, [](size_t) -> fx { return kOne; });
}

// ----- WriteClipTargets: THE FLOAT CROSSING — sample the clip + snap into the qTargets (host, deterministic)
// pose = SampleLocalPose(skeleton, clip, time) (float; one local TRS per joint in skeleton order). For each
// non-root edge e, active.drives[e].qTarget = FxQuatFromFloat(pose[childBoneOf(e)].r), where childBoneOf(e) ==
// active.ragdoll.joints[e].bodyB (the SAME index RagdollFromSkeleton built joints[e] from) and JointPose.r is
// EXACTLY the bone's local rotation in its parent's frame = the drive's relative-orientation target. The
// snap is RNG/clock-free and `time` advances by a fixed dt -> the qTargets are bit-reproducible AND shared
// identically by the GPU + CPU paths (only the integer StepDriveWorld over them is the GPU==CPU memcmp).
inline void WriteClipTargets(ActiveRagdoll& active, const anim::Skeleton& skeleton,
                             const anim::Animation& clip, float time) {
    const std::vector<anim::JointPose> pose = anim::SampleLocalPose(skeleton, clip, time);
    for (size_t e = 0; e < active.drives.size(); ++e) {
        const uint32_t childBone = active.ragdoll.joints[e].bodyB;
        if ((size_t)childBone < pose.size())
            active.drives[e].qTarget = FxQuatFromFloat(pose[(size_t)childBone].r);
    }
}

// ----- StepActive: one anim-tracking tick = WriteClipTargets then the AC1/AC2 integer drive step ---------
// The per-tick host pre-pass (sample+snap the clip into the qTargets, OUTSIDE the bit-exact loop) then the
// bit-exact integer StepDriveWorld over active.ragdoll.world/joints/limits + the freshly-written drives. The
// GPU showcase runs the SAME host pre-pass (sample+snap+upload the drives) then the AC1 active_drive_solve.comp
// for the integer step -> memcmp the GPU body world vs this CPU StepActive.
inline void StepActive(ActiveRagdoll& active, const anim::Skeleton& skeleton, const anim::Animation& clip,
                       float time, fx dt, int iters) {
    WriteClipTargets(active, skeleton, clip, time);
    StepDriveWorld(active.ragdoll.world, active.ragdoll.joints, active.ragdoll.limits, active.drives, dt, iters);
}

// ----- StepActiveSteps: run `steps` anim-tracking ticks advancing time = startTime + s*dt each step --------
// Deterministic: the clip `time` advances by a fixed dt per step (the AC5 note: this time is per-tick state).
inline void StepActiveSteps(ActiveRagdoll& active, const anim::Skeleton& skeleton, const anim::Animation& clip,
                            fx dt, int iters, int steps, float startTime) {
    // dt in seconds (float, for the clip time advance) = the Q16.16 dt back to float (render/clip-domain only).
    const float dtSeconds = (float)dt / (float)kOne;
    for (int s = 0; s < steps; ++s)
        StepActive(active, skeleton, clip, startTime + (float)s * dtSeconds, dt, iters);
}

// ====================================================================================================
// Slice AC4 — ACTIVE -> LIMP -> RECOVER (THE HEADLINE BEHAVIOR). AC1 built the angular drive; AC2 the
// per-joint blend weight; AC3 the anim-clip tracking. AC4 adds THE MONEY BEHAVIOR: a GLOBAL physicality
// knob in [0, kOne] that scales EVERY drive's weight (the UE5 physical-blend master alpha), an IMPULSE
// (a host body-velocity kick — the hit), and a deterministic RECOVERY RAMP. The character ANIMATES
// (physicality kOne, tracks the clip) -> GETS HIT (impulse + physicality drop -> goes LIMP, ragdoll wins)
// -> RECOVERS (physicality ramps back -> returns to the clip pose). INTEGER-bit-exact. NO new shader, NO
// new RHI (the drive solve is AC1's active_drive_solve.comp; physicality + impulse are host C++). AC4
// APPENDS — AC1-AC3 byte-frozen.

// ----- ApplyImpulse: a host body-velocity kick (the hit; the fpx/command idiom, fpx.h frozen) -----------
// If `bodyIndex` is in range AND the body is dynamic (flags & kFlagDynamic), add `dv` to its velocity
// (world.bodies[bodyIndex].vel = FxAdd(vel, dv)); otherwise a no-op (out-of-range OR a static/pinned body
// is never kicked). Deterministic integer add (FxAdd is per-axis +) -> bit-reproducible CPU<->GPU. This is
// a HOST mutation of the world state (the JT5/VH5/FR5 scripted-impulse-per-tick shape).
inline void ApplyImpulse(FxWorld& world, uint32_t bodyIndex, const FxVec3& dv) {
    if (bodyIndex >= (uint32_t)world.bodies.size()) return;            // out-of-range -> no-op
    FxBody& b = world.bodies[(size_t)bodyIndex];
    if (!(b.flags & fpx::kFlagDynamic)) return;                        // static/pinned -> never kicked
    b.vel = fpx::FxAdd(b.vel, dv);                                     // the deterministic integer velocity kick
}

// ----- PhysicalityAtTick: the deterministic recovery ramp (pure integer, no float) ----------------------
// The global physicality alpha as a function of the tick index. The EXACT tick boundaries:
//   * tick <  struckTick                              -> kOne   (ACTIVE — fully anim-driven, tracks the clip)
//   * struckTick <= tick < struckTick + limpTicks     -> 0      (LIMP — no drive, pure physics ragdoll)
//   * the next `recoverTicks` ticks (struckTick+limpTicks <= tick < struckTick+limpTicks+recoverTicks):
//         a LINEAR INTEGER ramp 0 -> kOne. Let r = tick - (struckTick + limpTicks) in [0, recoverTicks);
//         physicality = (fx)((int64_t)kOne * (r + 1) / recoverTicks)  (a clamped linear interp: the FIRST
//         recover tick (r==0) is kOne/recoverTicks (already partially active), the LAST (r==recoverTicks-1)
//         is kOne — the ramp REACHES full physicality on the final recover tick). recoverTicks <= 0 -> snap
//         straight to kOne (no ramp window).
//   * tick >= struckTick + limpTicks + recoverTicks   -> kOne   (RE-TRACKED — fully anim-driven again)
// Pure integer (int64 numerator avoids overflow, integer divide) -> bit-reproducible. No RNG/clock/float.
inline fx PhysicalityAtTick(int tick, int struckTick, int limpTicks, int recoverTicks) {
    if (tick < struckTick) return kOne;                               // before the hit -> active
    const int sinceStruck = tick - struckTick;                        // >= 0
    if (sinceStruck < limpTicks) return 0;                            // the limp window -> pure physics
    if (recoverTicks <= 0) return kOne;                               // no ramp window -> snap to active
    const int r = sinceStruck - limpTicks;                            // 0-based recover index, >= 0
    if (r >= recoverTicks) return kOne;                               // past the ramp -> re-tracked
    // a clamped linear integer ramp 0 -> kOne over recoverTicks ticks (reaches kOne on the last tick).
    return (fx)(((int64_t)kOne * (int64_t)(r + 1)) / (int64_t)recoverTicks);
}

// ----- StepActivePhysicality: one tick = WriteClipTargets then the AC1/AC2 drive step with EVERY drive's
// weight scaled by the global physicality (into a SCRATCH set — NO persistent mutation) --------------------
// WriteClipTargets(active, skeleton, clip, time) (AC3 — fills the base qTargets) -> build a scratch copy of
// active.drives with eff[e].driveWeight = fxmul(active.drives[e].driveWeight, physicality) -> StepDriveWorld
// over the scratch drives. The base active.drives is NEVER overwritten (the scratch keeps AC1-AC3 records
// frozen + avoids compounding mutation). At physicality == kOne, fxmul(w, kOne) == w -> eff == active.drives
// -> byte-identical to the AC3 StepActive (the equivalence contract). At physicality == 0, every weight 0 ->
// no correction -> pure physics (limp). Deterministic integer (the scale + the step are fixed-order int ops).
inline void StepActivePhysicality(ActiveRagdoll& active, const anim::Skeleton& skeleton,
                                  const anim::Animation& clip, float time, fx physicality, fx dt, int iters) {
    WriteClipTargets(active, skeleton, clip, time);                   // AC3 host pre-pass (fills base qTargets)
    std::vector<FxAngularDrive> eff = active.drives;                  // SCRATCH (base drives never overwritten)
    for (size_t e = 0; e < eff.size(); ++e)
        eff[e].driveWeight = fxmul(active.drives[e].driveWeight, physicality);   // scale every weight
    StepDriveWorld(active.ragdoll.world, active.ragdoll.joints, active.ragdoll.limits, eff, dt, iters);
}

// ----- StepActiveRecover: the scripted active -> struck -> recover episode driver -----------------------
// For each tick t in [0, totalTicks): physicality = PhysicalityAtTick(t, struckTick, limpTicks, recoverTicks);
// if t == struckTick call ApplyImpulse(world, impulseBody, impulseDv) (the hit, BEFORE the step so the kick
// integrates this tick); then StepActivePhysicality(active, skeleton, clip, startTime + t*dtSeconds,
// physicality, dt, iters). dtSeconds = (float)dt / (float)kOne (the AC3 clip-time convention). Deterministic,
// fixed order (physicality from the tick index, the impulse a scripted event at struckTick, the clip time
// per-tick state). The GPU showcase runs this SAME host per-tick logic (physicality, impulse at struckTick,
// the scratch-scaled drives, RE-UPLOAD) + the AC1 active_drive_solve.comp for one step -> the GPU body world
// memcmp's bit-exact vs this CPU StepActiveRecover (only the integer StepDriveWorld is the memcmp).
inline void StepActiveRecover(ActiveRagdoll& active, const anim::Skeleton& skeleton, const anim::Animation& clip,
                              fx dt, int iters, int struckTick, uint32_t impulseBody, const FxVec3& impulseDv,
                              int limpTicks, int recoverTicks, int totalTicks, float startTime) {
    const float dtSeconds = (float)dt / (float)kOne;
    for (int t = 0; t < totalTicks; ++t) {
        const fx physicality = PhysicalityAtTick(t, struckTick, limpTicks, recoverTicks);
        if (t == struckTick) ApplyImpulse(active.ragdoll.world, impulseBody, impulseDv);   // the hit
        StepActivePhysicality(active, skeleton, clip, startTime + (float)t * dtSeconds, physicality, dt, iters);
    }
}

// ====================================================================================================
// Slice AC5 — LOCKSTEP + ROLLBACK (THE NETCODE HEADLINE). AC1-AC4 built a clip-tracking, blendable,
// hit-reacting active ragdoll. AC5 proves the bit-exact clip-driven tick (AC3 StepActive) is true
// cross-platform LOCKSTEP + ROLLBACK — the FPX5/FR5/GR5/CG5/GF5/JT5/VH5 twin: two peers fed ONLY the
// input HIT stream re-derive the exact clip-driven ragdoll trajectory bit-for-bit, and a mispredicted hit
// is corrected by rolling back to a saved snapshot + re-simulating. PURE CPU (NO GPU dispatch, NO new
// shader, NO new RHI). ADDITIVE — AC1-AC4 above is byte-FROZEN; AC5 only APPENDS the command + snapshot +
// the three harness functions.
//
// THE INPUT is a HIT stream. ActiveCommand {tick; body; dv} — a velocity-impulse event (the AC4
// ApplyImpulse, now an INPUT rather than a scripted constant). The clip-drive (AC3 StepActive) holds the
// ragdoll on the animation; the hits perturb it; the drive recovers it. (AC5 uses full physicality /
// StepActive clip-tracking; the AC4 physicality ramp is NOT replayed here — a hit is the only
// nondeterminism-free input event.)
//
// THE VH5 TWIST — snapshot the world AND the tick. AC3's StepActive recomputes the qTargets each tick from
// the clip at time = startTime + tick*dt (a host pre-pass), and the drives' base weights are constant — so
// the only MUTABLE replayable state is the fpx::FxWorld (bodies). BUT the clip time is DERIVED from the tick
// index, so a snapshot must record WHICH tick it was taken at, or the resumed sim samples the clip at the
// wrong time. Therefore ActiveSnapshot {fpx::FxWorld world; int tick;} (the VH5 hinge-axes / GF5 two-pool
// analog: snapshot the world + the per-tick anchor). SnapshotActive deep-copies the world (via
// fpx::SnapshotWorld) + stores the tick; RestoreActive restores the world + returns the tick so the harness
// resumes the clip-time from there. (fpx.h SnapshotWorld/RestoreWorld are reused VERBATIM, read-only.)

// ----- ActiveCommand: ONE hit-impulse input event (the AC4 ApplyImpulse lifted to a replayable input) -----
// A velocity-impulse event at a given tick on a given body. The harness applies every command whose
// cmd.tick == tick (in ARRAY ORDER) via ApplyImpulse before the StepActive of that tick. (The fpx FxCommand
// / VehicleCommand scripted-impulse-per-tick shape.) Pure integer (dv is a Q16.16 FxVec3 velocity delta).
struct ActiveCommand {
    uint32_t tick = 0;       // the tick this hit fires at
    uint32_t body = 0;       // the body index the impulse is applied to (a dynamic bone)
    fpx::FxVec3 dv;          // the Q16.16 velocity-impulse delta (the AC4 ApplyImpulse dv)
};

// ----- ActiveSnapshot: the captured mutable active-ragdoll state (the world AND the clip-time tick) --------
// The VH5 twist: the body world (deep-copied via fpx::SnapshotWorld — the std::vector<FxBody> + the scalar
// gravity/groundY) PLUS the `tick` the snapshot was taken at (the clip-time anchor — StepActive recomputes
// the qTargets from time = startTime + tick*dt, so the resumed sim needs the tick to sample the clip at the
// right time). The drives + joints + limits are immutable structure (their qTargets are re-derived from the
// clip each tick) so they are NOT snapshotted — EXACTLY the mutable state is captured (bodies + tick).
struct ActiveSnapshot {
    fpx::FxWorld world;      // the body world (fpx::SnapshotWorld deep-copy: bodies + gravity/groundY)
    int          tick = 0;   // the clip-time anchor — the tick this snapshot was taken at
};

// ----- SnapshotActive: deep-copy the mutable active-ragdoll state (the rollback restore point) ------------
// Reuses fpx::SnapshotWorld VERBATIM for the bodies (a value copy -> deep-copies the bodies vector) + stores
// the tick. Bit-exact round-trip with RestoreActive: RestoreActive(active, SnapshotActive(active0, t))
// leaves active's world == active0's byte-for-byte AND returns t.
inline ActiveSnapshot SnapshotActive(const ActiveRagdoll& active, int tick) {
    ActiveSnapshot snap;
    snap.world = fpx::SnapshotWorld(active.ragdoll.world);
    snap.tick = tick;
    return snap;
}

// ----- RestoreActive: restore the world from a snapshot + return the captured tick (the rollback) ---------
// Restores the body world (fpx::RestoreWorld) and returns snap.tick so the harness resumes the clip-time
// from the saved tick. Bit-exact round-trip with SnapshotActive. The drives/joints/limits are untouched
// (immutable structure — their qTargets are re-derived from the clip).
inline int RestoreActive(ActiveRagdoll& active, const ActiveSnapshot& snap) {
    fpx::RestoreWorld(active.ragdoll.world, snap.world);
    return snap.tick;
}

// ----- SimActiveTick: the deterministic per-tick step (apply this tick's hits + StepActive) ---------------
// (0) APPLY this tick's commands in ARRAY ORDER (every cmd with cmd.tick == tick) via ApplyImpulse — the
//     AC4 hit, now an input event — BEFORE the step so the kick integrates this tick.
// (1) StepActive(active, skeleton, clip, startTime + tick*dtSeconds, dt, iters) — the AC3 clip-driven tick
//     (WriteClipTargets then the integer StepDriveWorld). dtSeconds = (float)dt/(float)kOne (the AC3
//     clip-time convention). Pure integer sim + the deterministic host clip pre-pass -> bit-identical on
//     every peer/platform. The JT5 SimRagdollTick / VH5 SimVehicleTick twin.
inline void SimActiveTick(ActiveRagdoll& active, const anim::Skeleton& skeleton, const anim::Animation& clip,
                          const std::vector<ActiveCommand>& commands, int tick, float startTime, fx dt,
                          int iters) {
    for (size_t c = 0; c < commands.size(); ++c)
        if ((int)commands[c].tick == tick)
            ApplyImpulse(active.ragdoll.world, commands[c].body, commands[c].dv);
    const float dtSeconds = (float)dt / (float)kOne;
    StepActive(active, skeleton, clip, startTime + (float)tick * dtSeconds, dt, iters);
}

// ----- RunActiveLockstep: authority + replica from the SAME inputs, bit-identical every tick --------------
// THE peer entry point (the JT5 RunRagdollLockstep / VH5 RunVehicleLockstep control flow over SimActiveTick).
// Run `ticks` SimActiveTicks from a COPY of `initialActive`, applying the command stream -> the converged
// ragdoll. authority = RunActiveLockstep(skel, clip, init, commands, N, ...); replica = the SAME from the
// SAME init + stream (INPUTS ONLY — no state shared) -> BIT-IDENTICAL by determinism. This function ASSERTS
// authority == replica bit-for-bit every tick (memcmp the world bodies) via an internal replica run; the
// caller also memcmps two RunActiveLockstep returns for the determinism proof. Returns the converged
// authority ActiveRagdoll.
inline ActiveRagdoll RunActiveLockstep(const anim::Skeleton& skeleton, const anim::Animation& clip,
                                       const ActiveRagdoll& initialActive,
                                       const std::vector<ActiveCommand>& commands, int ticks, float startTime,
                                       fx dt, int iters) {
    ActiveRagdoll authority = initialActive;   // a fresh copy (world + drives)
    ActiveRagdoll replica   = initialActive;   // the second peer fed the SAME inputs
    for (int t = 0; t < ticks; ++t) {
        SimActiveTick(authority, skeleton, clip, commands, t, startTime, dt, iters);
        SimActiveTick(replica,   skeleton, clip, commands, t, startTime, dt, iters);
        // assert bit-identical every tick — two peers fed only the hit stream stay in lockstep.
        if (authority.ragdoll.world.bodies.size() != replica.ragdoll.world.bodies.size() ||
            std::memcmp(authority.ragdoll.world.bodies.data(), replica.ragdoll.world.bodies.data(),
                        authority.ragdoll.world.bodies.size() * sizeof(FxBody)) != 0) {
            // the lockstep invariant broke — a nondeterminism the showcase/test reports loudly. We leave the
            // authority as-is; the caller's memcmp proof catches the divergence (this branch is unreachable
            // for a deterministic sim — the fixed-order integer ops guarantee authority == replica).
            return authority;
        }
    }
    return authority;
}

// ----- RunActiveRollback: snapshot -> mispredict diverges -> rollback -> corrected == authority -----------
// The rollback harness (the JT5 RunRagdollRollback / VH5 RunVehicleRollback control flow over SimActiveTick).
// (1) advance ticks 0..divergeTick from `initialActive` applying authorityCmds; (2) SAVE an ActiveSnapshot
// AT divergeTick (SnapshotActive — the world AND the tick); (2b) speculatively advance a few ticks with the
// MISPREDICTED stream (a WRONG hit — different body/dv/tick, the client prediction that diverges); (3)
// ROLLBACK — RestoreActive to the snapshot (restoring the world AND resuming the clip-time from the saved
// tick) + RE-SIMULATE divergeTick..ticks with the CORRECT authorityCmds -> the corrected final ragdoll. The
// caller asserts this == RunActiveLockstep(skel, clip, init, authorityCmds, ticks, ...) (rollback corrected
// the misprediction EXACTLY) AND that the speculative pre-rollback state DIFFERED from authority (a real
// divergence was fixed). Reuses SnapshotActive/RestoreActive (which carry the tick). startTime/dt/iters are
// CONSTANT, NOT snapshotted.
inline ActiveRagdoll RunActiveRollback(const anim::Skeleton& skeleton, const anim::Animation& clip,
                                       const ActiveRagdoll& initialActive,
                                       const std::vector<ActiveCommand>& authorityCmds,
                                       const std::vector<ActiveCommand>& mispredictCmds, int divergeTick,
                                       int ticks, float startTime, fx dt, int iters) {
    ActiveRagdoll active = initialActive;
    // (1) advance 0..divergeTick with the authoritative stream.
    for (int t = 0; t < divergeTick; ++t)
        SimActiveTick(active, skeleton, clip, authorityCmds, t, startTime, dt, iters);
    // (2) SAVE the snapshot at divergeTick (the rollback restore point — the world AND the clip-time tick).
    const ActiveSnapshot snap = SnapshotActive(active, divergeTick);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (the wrong hit — the client
    // prediction that diverges). Bounded to the remaining ticks.
    int specTicks = ticks - divergeTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s)
        SimActiveTick(active, skeleton, clip, mispredictCmds, divergeTick + s, startTime, dt, iters);
    // (3) ROLLBACK: restore the snapshot (world + tick) + re-sim divergeTick..ticks with the authStream.
    const int resumeTick = RestoreActive(active, snap);   // == divergeTick (the clip-time anchor)
    for (int t = resumeTick; t < ticks; ++t)
        SimActiveTick(active, skeleton, clip, authorityCmds, t, startTime, dt, iters);
    return active;
}

// ====================================================================================================
// Slice AC6 — LIT 3D SKINNED RENDER CAPSTONE (THE MONEY-SHOT, COMPLETES FLAGSHIP #17). AC1-AC5 built +
// proved a deterministic, blendable, hit-reacting, lockstep-replayable active ragdoll. AC6 RENDERS it:
// the bit-exact active-ragdoll pose (a character TRACKING an animation clip via physics torques) drives
// the EXISTING Fox skinned lit render — the physics moat poses the animation mesh. JT6 VERBATIM with the
// palette source swapped (the JT6 RagdollToPalette precedent). The ONE float crossing on the render path,
// render-only, OUTSIDE the bit-exact sim loop (joint::PoseToPalette, Q16.16->float). NO new shader, NO new
// RHI. AC6 APPENDS ONLY this thin alias — AC1-AC5 above is byte-FROZEN.

// ----- ActiveToPalette: the JT6 RagdollToPalette alias over the active ragdoll's settled world ----------
// The active ragdoll's joint palette, flattened for the skinning pipeline, IS exactly
// joint::PoseToPalette(skeleton, active.ragdoll.world) (the std::vector<math::Mat4> the showcase copies into
// the fixed-size JointPalette UBO in skeleton order). A pure deterministic function of the bit-exact active
// body state (the provenance proof: rebuild from the settled world reproduces the palette byte-for-byte) —
// the SAME palette joint::RagdollToPalette names for a JT4 ragdoll, here for the AC3-clip-tracked active
// pose. It exists only to spell out the bridge at the call site; the showcase may call PoseToPalette
// directly with no change. One Mat4 per joint, in skeleton order.
inline std::vector<math::Mat4> ActiveToPalette(const anim::Skeleton& skeleton, const ActiveRagdoll& active) {
    return joint::PoseToPalette(skeleton, active.ragdoll.world);
}

}  // namespace active
}  // namespace hf::sim
