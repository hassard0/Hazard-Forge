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
#include <span>          // Slice VH3: std::span for the fpx::BuildPairs / fpx::StepWorld contact block
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

// ====================================================================================================
// Slice VH2 — THE VEHICLE RIG + WHEEL HINGE (the 2nd slice of FLAGSHIP #16: DETERMINISTIC VEHICLE
// PHYSICS, hf::sim::vehicle). VH1 built the suspension SPRING joint in isolation (a body bobbing on a
// spring); VH2 ASSEMBLES the car — 1 chassis fpx::FxBody + 4 wheel FxBodies tied by 4 VH1 FxSpringJoints
// (the suspension) + 4 joint::FxAngularLimit hinges (kAngularHinge — the wheels SPIN about their axle but
// the hinge clamps the off-axis swing so a wheel stays in its rolling plane, no flopping) — and SETTLES
// it on its springs so the chassis rests at ride height and the four wheels rest on the ground. The
// joint::RagdollFromSkeleton twin (a host-built body+constraint rig) over the vehicle. ADDITIVE — VH1's
// FxSpringJoint/SolveSpringJoint/SpringLength/StepSpringWorld + vehicle_spring_solve.comp stay
// BYTE-FROZEN. INTEGER-bit-exact. **NO new shader** — StepVehicleRig is structured as the JT3
// host-driven multi-pass mold so the GPU showcase drives the EXISTING VH1 vehicle_spring_solve.comp +
// JT2 joint_angular_solve.comp in the locked order, memcmp vs this CPU reference.
//
// THE STEP STRUCTURE (StepVehicleRig — the JT3 two-PHASE-block tick that the existing whole-step shaders
// reproduce byte-for-byte, NOT a per-iteration {spring | angular} interleave; the JT3 hard lesson is that
// dt=0 integrate-suppression is NOT bit-idempotent for orientation, so a per-iteration GPU interleave
// over a whole-step shader would diverge):
//   PHASE A (the spring phase — the EXISTING vehicle_spring_solve.comp, SENTINEL groundY so its floor
//            clamp is a DEAD no-op): IntegrateBodyFull(all) [the FPX4 6-DOF integrate, real dt] -> K
//            Gauss-Seidel passes EACH applying all SolveSpringJoint. Touches pos/vel.
//   PHASE B (the hinge phase — the EXISTING joint_angular_solve.comp, dt=0 + jointCount=0 + SENTINEL
//            groundY): IntegrateBodyFull(all) with dt=0 [the dt=0 translation integrate is a no-op; the
//            orientation integrate runs its FxQuatNormalize, exactly the shader's behaviour] -> K
//            Gauss-Seidel passes EACH applying all SolveAngularLimit (the JT2 swing-twist + hinge clamp).
//            With jointCount=0 the shader's ball pass is a no-op. Touches orient.
//   PHASE C (the ground clamp — HOST, NOT the shaders' generic pos.y<groundY clamp): each DYNAMIC body's
//            pos.y is lifted to >= groundY + radius (the wheels rest ON the floor at their radius; the
//            chassis is held up by the springs). Both shader phases use the SENTINEL groundY so their
//            internal clamp never fires; the WHEEL-radius clamp lives here.
// The CPU StepVehicleRig below calls exactly these ops in this order, so the GPU driver (Phase A via the
// spring shader, Phase B via the angular shader with dt=0, Phase C on the host) memcmps BIT-EXACT.
//
// HONEST CAVEATS (cloth/JT/VH1-identical): the suspension residual is deterministic-but-nonzero (the
// Gauss-Seidel spring residual + the nlerp angular residual; stiffness/limit ∝ iterations — within-band,
// not analytic). NO inter-body broadphase/contacts (that is VH3 — the wheels rest via the ground clamp,
// the chassis floats on the springs; ramps/obstacles + drive/traction are later VH slices). The hinge is
// the JT2 cone-limit-0 proxy. The headline is DETERMINISM + cross-platform bit-identity, NOT analytic
// vehicle mechanics.

// Re-export the JT2 angular-limit toolbox VERBATIM (read-only — REUSED, not re-implemented).
using joint::FxAngularLimit;     // the hinge/cone angular limit record (8 x int32, 32 bytes)
using joint::SolveAngularLimit;  // the swing-twist + host-cos cone clamp + nlerp inverse-mass apply
using joint::SwingAngleCos;      // the swing .w (== cos half off-axis angle) metric — the in-plane proof
using joint::QConj;              // the unit-quaternion conjugate (for the body-local frame transform)
using joint::kAngularHinge;      // cone limit 0 -> swing forced to identity (the wheel spin-axis hinge)

// ----- VehicleConfig: the host recipe for the car (all Q16.16; the documented integer scene constants) -
// rideHeight     : the chassis centre height above groundY at rest (Q16.16 world units).
// suspensionLen  : the spring rest length (chassis-corner anchor -> wheel centre) — the suspension travel.
// springStiffness/springDamping : the FxSpringJoint soft-restore + normal-velocity damper (in [0,kOne]).
// chassisHalfX/Y/Z : the chassis box half-extents (render-only side view; the body is sphere-bound).
// chassisRadius  : the chassis broadphase/ground sphere radius (Q16.16; VH2 has no broadphase — used by
//                  the Phase-C ground clamp + the render).
// wheelRadius    : the wheel sphere radius (the wheels rest at groundY + wheelRadius).
// wheelBaseX/wheelBaseZ : the 4 chassis-corner offsets (±wheelBaseX, ±wheelBaseZ) the wheels hang from.
// hingeCosHalf/hingeSinHalf : the JT2 host cos(θ/2)/sin(θ/2) hinge constants (a HINGE = kOne/0).
// gravity        : the Q16.16 acceleration (e.g. (0,-9.8,0)). groundY : the floor the wheels rest on.
// chassisInvMass/wheelInvMass : the dynamic inverse masses (Q16.16; a heavier chassis -> smaller invMass).
struct VehicleConfig {
    fx       rideHeight    = (fx)(3 * (int)kOne / 2); // chassis centre 1.5 above ground at rest
    fx       suspensionLen = kOne;                   // 1.0 spring rest length (suspension travel)
    fx       springStiffness = (fx)(kOne * 6 / 10);  // 0.6 stiff per-iteration positional restore (holds the chassis up)
    fx       springDamping   = kOne / 4;             // 0.25 normal-velocity damping
    fx       chassisHalfX  = (fx)(3 * (int)kOne / 2);// chassis box half-extents (render side-view)
    fx       chassisHalfY  = (fx)(kOne * 3 / 10);    // 0.3 half-height (a slim chassis -> a clear gap to the wheels)
    fx       chassisHalfZ  = kOne;
    fx       chassisRadius = (fx)(kOne / 4);         // 0.25 chassis ground sphere radius (slack vs the springs)
    fx       wheelRadius   = kOne / 2;               // 0.5 wheel sphere radius
    fx       wheelBaseX    = (fx)(3 * (int)kOne / 2);// ±1.5 corner offset along the car length (X)
    fx       wheelBaseZ    = kOne;                   // ±1.0 corner offset across the car width (Z)
    fx       hingeCosHalf  = kOne;                   // HINGE: cos(0/2) = 1 (swing forced to identity)
    fx       hingeSinHalf  = 0;                      // HINGE: sin(0/2) = 0
    FxVec3   gravity{0, (fx)(-9 * (int)kOne - kOne * 8 / 10), 0};  // (0, -9.8, 0) Q16.16
    fx       groundY       = 0;                      // the floor the wheels rest on
    fx       chassisInvMass = kOne / 2;              // a moderately heavy chassis
    fx       wheelInvMass   = kOne;                  // lighter wheels
};

// ----- Vehicle: the assembled rig (an fpx::FxWorld + its spring/hinge graph + the body indices) --------
// world.bodies[chassisIndex] = the chassis; world.bodies[wheelIndex[k]] = wheel k (k in 0..3). springs +
// hinges are PARALLEL per-corner (spring[k] + hinge[k] tie the chassis to wheel k). All host-fixed,
// integer — the deterministic init the StepVehicleRig settle runs over (the bit-reproducible scene).
struct Vehicle {
    fpx::FxWorld                world;       // 1 chassis + 4 wheel bodies (chassis first)
    std::vector<FxSpringJoint>  springs;     // 4 suspension springs (chassis <-> wheel k)
    std::vector<FxAngularLimit> hinges;      // 4 wheel hinges (kAngularHinge, axle = the car lateral axis)
    uint32_t chassisIndex = 0;               // index of the chassis body in world.bodies
    uint32_t wheelIndex[4] = {1, 2, 3, 4};   // indices of the 4 wheel bodies
};

// ----- VehicleFromConfig: host-assemble the car (the RagdollFromSkeleton twin — bodies + constraints) --
// Builds (pure integer, host-fixed): the chassis FxBody (box, sphere-bound chassisRadius, at ride height)
// + 4 wheel FxBodies (at the 4 chassis-corner offsets ±wheelBaseX/±wheelBaseZ, dropped by suspensionLen
// below the chassis), 4 FxSpringJoints (bodyA=chassis, bodyB=wheel; anchorA = the chassis corner in the
// chassis-LOCAL frame, anchorB = (0,0,0) the wheel centre; restLen=suspensionLen; stiffness/damping from
// cfg), and 4 FxAngularLimit hinges (kAngularHinge; axis = the wheel spin axis = the car LATERAL axis
// (0,0,1) on the chassis — body-local on the chassis A; a hinge limits the off-axis swing). Anchors are
// taken in the body-LOCAL frame via FxRotate(QConj(orient), worldPos − bodyPos) — the RagdollFromSkeleton
// convention (the bodies start at identity orientation so this is the identity map, but the form is
// future-proof + matches the JT4 mold). The 4 corners are laid out in a FIXED order:
//   0 = (+wheelBaseX, +wheelBaseZ)  (front-right)   2 = (−wheelBaseX, +wheelBaseZ)  (rear-right)
//   1 = (+wheelBaseX, −wheelBaseZ)  (front-left)    3 = (−wheelBaseX, −wheelBaseZ)  (rear-left)
inline Vehicle VehicleFromConfig(const VehicleConfig& cfg) {
    Vehicle v;
    v.world.gravity = cfg.gravity;
    v.world.groundY = cfg.groundY;

    const FxVec3 identityQxyz{0, 0, 0};  // (kept for clarity; orient is identity below)
    (void)identityQxyz;

    // (1) the chassis body at ride height (sphere-bound chassisRadius; the heavier dynamic body).
    FxBody chassis;
    chassis.pos = FxVec3{0, cfg.groundY + cfg.rideHeight, 0};
    chassis.vel = FxVec3{0, 0, 0};
    chassis.invMass = cfg.chassisInvMass;
    chassis.flags   = fpx::kFlagDynamic;
    chassis.radius  = cfg.chassisRadius;
    chassis.orient  = fpx::FxQuat{0, 0, 0, kOne};
    chassis.angVel  = FxVec3{0, 0, 0};
    v.chassisIndex = 0;
    v.world.bodies.push_back(chassis);

    // The 4 corner offsets (chassis-local), FIXED order. The wheels start dropped by suspensionLen.
    const FxVec3 corner[4] = {
        FxVec3{ cfg.wheelBaseX,  0,  cfg.wheelBaseZ},   // 0 front-right
        FxVec3{ cfg.wheelBaseX,  0, -cfg.wheelBaseZ},   // 1 front-left
        FxVec3{-cfg.wheelBaseX,  0,  cfg.wheelBaseZ},   // 2 rear-right
        FxVec3{-cfg.wheelBaseX,  0, -cfg.wheelBaseZ},   // 3 rear-left
    };

    // (2) the 4 wheel bodies — each at the chassis corner, dropped suspensionLen below the chassis centre.
    for (int k = 0; k < 4; ++k) {
        FxBody wheel;
        wheel.pos = FxVec3{chassis.pos.x + corner[k].x,
                           chassis.pos.y + corner[k].y - cfg.suspensionLen,
                           chassis.pos.z + corner[k].z};
        wheel.vel = FxVec3{0, 0, 0};
        wheel.invMass = cfg.wheelInvMass;
        wheel.flags   = fpx::kFlagDynamic;
        wheel.radius  = cfg.wheelRadius;
        wheel.orient  = fpx::FxQuat{0, 0, 0, kOne};
        wheel.angVel  = FxVec3{0, 0, 0};
        v.wheelIndex[k] = (uint32_t)v.world.bodies.size();
        v.world.bodies.push_back(wheel);
    }

    // (3) per corner: a suspension spring (chassis corner <-> wheel centre) + a wheel hinge.
    const FxBody& chassisRef = v.world.bodies[(size_t)v.chassisIndex];
    for (int k = 0; k < 4; ++k) {
        const uint32_t wk = v.wheelIndex[k];
        const FxBody& wheelRef = v.world.bodies[(size_t)wk];
        // The chassis-corner world position (where the spring's chassis end attaches).
        const FxVec3 cornerWorld = FxAdd(chassisRef.pos, corner[k]);

        // The spring: anchorA = the chassis corner in the chassis-LOCAL frame; anchorB = the wheel centre
        // in the wheel-LOCAL frame (== (0,0,0)). FxRotate(QConj(orient), worldPos − bodyPos) (JT4 mold).
        FxSpringJoint s;
        s.bodyA = v.chassisIndex;
        s.bodyB = wk;
        s.anchorA = FxRotate(QConj(chassisRef.orient), FxSub(cornerWorld, chassisRef.pos));
        s.anchorB = FxRotate(QConj(wheelRef.orient), FxSub(wheelRef.pos, wheelRef.pos));  // (0,0,0)
        s.restLen = cfg.suspensionLen;
        s.stiffness = cfg.springStiffness;
        s.damping = cfg.springDamping;
        v.springs.push_back(s);

        // The hinge: axis = the wheel spin axis = the car LATERAL axis (0,0,1) in the chassis-local frame
        // (the chassis starts at identity so the world lateral axis IS the local axis). A HINGE
        // (kAngularHinge, cos/sin = kOne/0 -> cone limit 0) clamps the off-axis swing -> the wheel may
        // spin about Z but does not flop out of its rolling plane.
        FxAngularLimit h;
        h.bodyA = v.chassisIndex;
        h.bodyB = wk;
        h.axis = FxRotate(QConj(chassisRef.orient), FxVec3{0, 0, kOne});  // lateral (Z) axle, local on A
        h.cosHalfLimit = cfg.hingeCosHalf;
        h.sinHalfLimit = cfg.hingeSinHalf;
        h.kind = kAngularHinge;
        v.hinges.push_back(h);
    }
    return v;
}

// ----- StepVehicleRig: ONE vehicle-rig settle tick (the JT3 two-phase-block + the WHEEL ground clamp) ---
// The PHASE A / PHASE B / PHASE C structure documented above — the exact op order the GPU driver
// reproduces with the EXISTING spring + angular shaders:
//   PHASE A (spring): IntegrateBodyFull(all, real dt) -> K Gauss-Seidel passes of all SolveSpringJoint.
//   PHASE B (hinge):  IntegrateBodyFull(all, dt=0) [the angular shader's per-step orientation re-normalize
//                     — a no-op for translation; mirrors the shader] -> K passes of all SolveAngularLimit.
//   PHASE C (ground): lift each DYNAMIC body to pos.y >= groundY + radius (the wheels rest ON the floor at
//                     their radius; the chassis, held by the springs, floats above — its radius clamp is
//                     slack because the springs keep it well above groundY + chassisRadius).
// Pure integer, fixed op order -> two-run bit-identical AND bit-exact GPU==CPU. NO inter-body contacts.
inline void StepVehicleRig(Vehicle& v, fx dt, int iters) {
    FxWorld& world = v.world;
    const size_t n = world.bodies.size();

    // PHASE A — the spring phase (== vehicle_spring_solve.comp, SENTINEL groundY so no floor clamp).
    for (size_t i = 0; i < n; ++i)
        fpx::IntegrateBodyFull(world.bodies[i], world.gravity, dt);
    for (int it = 0; it < iters; ++it)
        for (size_t e = 0; e < v.springs.size(); ++e)
            SolveSpringJoint(world, v.springs[e]);

    // PHASE B — the hinge phase (== joint_angular_solve.comp, dt=0 + jointCount=0 + SENTINEL groundY). The
    // dt=0 IntegrateBodyFull mirrors the angular shader's per-step integrate (its orientation FxQuatNormalize
    // runs; translation is a no-op at dt=0) so the CPU reference is byte-identical to the GPU phase.
    for (size_t i = 0; i < n; ++i)
        fpx::IntegrateBodyFull(world.bodies[i], world.gravity, /*dt=*/0);
    for (int it = 0; it < iters; ++it)
        for (size_t e = 0; e < v.hinges.size(); ++e)
            SolveAngularLimit(world, v.hinges[e]);

    // PHASE C — the WHEEL/CHASSIS ground clamp (each dynamic body rests at groundY + its radius). Like the
    // fpx IntegrateBody floor clamp, a clamped body's DOWNWARD velocity is zeroed (it stops falling — the
    // wheel rests; without this the clamped bodies keep accumulating gravity velocity and the spring damper
    // sees zero RELATIVE velocity, so the suspension never settles).
    for (size_t i = 0; i < n; ++i) {
        FxBody& b = world.bodies[i];
        if ((b.flags & fpx::kFlagDynamic) && b.pos.y < world.groundY + b.radius) {
            b.pos.y = world.groundY + b.radius;
            if (b.vel.y < 0) b.vel.y = 0;
        }
    }
}

// ----- StepVehicleRigSteps: run K full vehicle-rig settle ticks (the showcase / GPU K-step driver) ------
inline void StepVehicleRigSteps(Vehicle& v, fx dt, int iters, int steps) {
    for (int s = 0; s < steps; ++s)
        StepVehicleRig(v, dt, iters);
}

// ----- VehicleRigState + MeasureVehicleRig: the honest settled-rig metrics (deterministic Q16.16) -------
// chassisY        : the chassis centre pos.y (the ride-height proof — settles near groundY + rideHeight).
// meanSpringLen   : the mean of the 4 spring lengths (compression: < restLen means the springs hold the
//                   chassis up under its weight).
// minWheelBottom  : the lowest (wheel.pos.y − wheelRadius) over the 4 wheels (>= groundY -> on the ground,
//                   nothing buried).
// wheelsOnGround  : true iff every wheel rests AT the ground band (|wheel.pos.y − (groundY+wheelRadius)|
//                   within a small band — the wheels rest on the floor, not floating, not buried).
// maxHingeOffAxis : the max (kOne − SwingAngleCos) over the 4 hinges (small -> the wheels stayed IN their
//                   rolling plane; a held hinge keeps SwingAngleCos near kOne).
struct VehicleRigState {
    fx       chassisY        = 0;
    fx       meanSpringLen   = 0;
    fx       minWheelBottom  = 0;
    bool     wheelsOnGround  = false;
    fx       maxHingeOffAxis = 0;
};

inline VehicleRigState MeasureVehicleRig(const Vehicle& v, const VehicleConfig& cfg) {
    VehicleRigState st;
    const FxWorld& world = v.world;
    st.chassisY = world.bodies[(size_t)v.chassisIndex].pos.y;

    // Mean spring length over the 4 springs (the suspension compression metric).
    int64_t sumLen = 0;
    for (const FxSpringJoint& s : v.springs)
        sumLen += (int64_t)SpringLength(world, s);
    st.meanSpringLen = v.springs.empty() ? 0 : (fx)(sumLen / (int64_t)v.springs.size());

    // The wheels-on-ground stat: every wheel's bottom (pos.y − wheelRadius) within a band of groundY.
    const fx kGroundBand = kOne / 8;   // 0.125 unit band around the floor
    bool allOnGround = true;
    bool haveBottom = false;
    for (int k = 0; k < 4; ++k) {
        const FxBody& w = world.bodies[(size_t)v.wheelIndex[k]];
        const fx bottom = w.pos.y - w.radius;
        if (!haveBottom || bottom < st.minWheelBottom) { st.minWheelBottom = bottom; haveBottom = true; }
        const fx d = bottom - world.groundY;
        const fx ad = d < 0 ? -d : d;
        if (ad > kGroundBand) allOnGround = false;
    }
    st.wheelsOnGround = allOnGround;
    (void)cfg;

    // The max hinge off-axis swing (kOne − SwingAngleCos; small -> the wheels stayed in-plane).
    fx maxOff = 0;
    for (const FxAngularLimit& h : v.hinges) {
        const fx cosSwing = SwingAngleCos(world, h);
        const fx off = kOne - cosSwing;   // 0 at perfectly in-plane
        if (off > maxOff) maxOff = off;
    }
    st.maxHingeOffAxis = maxOff;
    return st;
}

// ====================================================================================================
// Slice VH3 — DRIVE + STEER COMMANDS + THE LOCKED VEHICLE TICK (the 3rd slice of FLAGSHIP #16:
// DETERMINISTIC VEHICLE PHYSICS, hf::sim::vehicle). VH1 built the suspension spring; VH2 assembled +
// settled the car (chassis + 4 wheels + 4 springs + 4 hinges). VH3 makes it RESPOND TO INPUT: two new
// integer command kinds — kCmdDriveTorque (spin a wheel about its axle) + kCmdSteer (rotate the front-
// wheel hinge axes) — fed through a single locked vehicle tick StepVehicle that adds the fpx broadphase +
// sphere-sphere CONTACT pass to VH2's spring+hinge solve (the JT3 contacts-as-a-trailing-block mold).
// INTEGER-bit-exact. NO new shader — the GPU showcase drives the EXISTING VH1 vehicle_spring_solve.comp +
// JT2 joint_angular_solve.comp + fpx fpx_solve.comp in the locked order. ADDITIVE — VH1/VH2 byte-frozen.
//
// THE DESIGN CALL (input as deterministic integer commands, contacts as a trailing block — the JT3 mold):
// drive + steer = 2 new integer command kinds. fpx::FxCommand{tick, kind, target, arg} already exists (the
// FPX5/lockstep substrate). VH3 adds two NEW `kind` values DEFINED HERE (NOT in fpx.h — fpx.h stays
// frozen; the values are vehicle-local constants ABOVE fpx's existing kind range, applied ONLY by
// ApplyVehicleCommand, never by fpx::ApplyCommand):
//   - kCmdDriveTorque: target = a wheel body index, arg = a signed Q16.16 magnitude. Adds an angular
//     impulse about the wheel's spin axis to the wheel's angVel (wheel.angVel += FxScale(spinAxis, arg)).
//     The throttle — spins the driven wheels. Out-of-range / static targets are deterministic no-ops.
//   - kCmdSteer: target = a front hinge index (0/1), arg = a signed Q16.16 steer angle. Rotates that
//     hinge's `axis` about the chassis up-axis by a HOST-SNAPPED steer quaternion (QFromAxisAngleSnapped —
//     a PURE-INTEGER fixed-point half-angle series so the snap is bit-reproducible on EVERY platform, the
//     BuildPileWorld host-snap idiom but with NO <cmath>), then re-normalizes the axis. Steering re-aims
//     the front wheels' rolling plane. Non-front targets (∉ {0,1}) are deterministic no-ops.
// Integer adds + a host-snapped quaternion only — NO runtime transcendentals, NO float in the sim path.
//
// HONEST CAVEAT (the FR4/JT3 caveat carried forward): fpx contacts are sphere-sphere with NO inertia
// tensor — a contact does NOT spin a body, so wheel rolling-from-ground-contact is NOT emergent; drive
// comes from the command/integrate path and VH4's traction is a deterministic tangential PROXY, not
// analytic tyre mechanics. The headline is DETERMINISM + cross-platform bit-identity, NOT physical
// correctness. The Gauss-Seidel spring/hinge + the contact residual is deterministic-but-nonzero.

// Re-export the fpx command primitive (read-only — REUSED, not re-implemented). VH3's command kinds are
// vehicle-local; we reuse the FxCommand RECORD shape so the showcase/test command stream is a plain
// std::vector<fpx::FxCommand>. (fpx::kCmdImpulse=0/kCmdSetAngVel=1 live in fpx.h; VH3's kinds are above.)
using fpx::FxCommand;
inline constexpr uint32_t kFlagDynamic = fpx::kFlagDynamic;

// VH3 vehicle-local command kinds. Chosen ABOVE fpx's existing FxCommand kind range (fpx uses 0,1) so a
// vehicle command stream is unambiguous and ApplyVehicleCommand is the ONLY applier (fpx::ApplyCommand
// ignores these — it only knows kCmdImpulse/kCmdSetAngVel). NOT added to fpx.h (frozen).
inline constexpr uint32_t kCmdDriveTorque = 100u;  // arg = Q16.16 axle spin added to wheel target's angVel
inline constexpr uint32_t kCmdSteer       = 101u;  // arg.x = Q16.16 steer angle for front hinge `target`

// The chassis UP-axis (the steer rotation axis) + the wheel SPIN axis (the axle, the lateral Z axis). These
// are the FIXED body-local frame constants VehicleFromConfig laid the hinges out about (the hinge axis is
// the lateral (0,0,1) Z axle; the up axis the chassis (0,1,0) Y). Steering rotates the hinge axle about Y.
inline const FxVec3 kChassisUpAxis{0, kOne, 0};    // (0,1,0) — the steer rotation axis
inline const FxVec3 kWheelSpinAxis{0, 0, kOne};    // (0,0,1) — the wheel axle / drive-torque axis

// The 2 FRONT hinge indices (VehicleFromConfig corner order: 0 front-right, 1 front-left, 2/3 rear). Only a
// front hinge accepts a kCmdSteer (the rears are unchanged — the steer re-aims the FRONT rolling plane).
inline bool IsFrontHinge(uint32_t hingeIndex) { return hingeIndex == 0u || hingeIndex == 1u; }

// ----- QFromAxisAngleSnapped: a PURE-INTEGER host-snapped quaternion about a unit axis ------------------
// q = {axis*sin(θ/2), cos(θ/2)} with cos/sin built from a fixed-point half-angle SERIES (NO <cmath>, NO
// runtime transcendentals): for the half-angle h = θ/2 (Q16.16), cos(h) = 1 - h²/2 + h⁴/24 - h⁶/720 and
// sin(h) = h - h³/6 + h⁵/120 (the Taylor series), each term an fxmul — bit-identical on EVERY compiler/
// vendor by construction (the steer angles are small, well within the series' accurate range). This is the
// "host round-to-nearest snap" of the BuildPileWorld idiom made cross-platform-EXACT by being integer-only
// (a std::cos/std::sin host snap is NOT guaranteed bit-identical cross-vendor; this series IS). The result
// is FxQuatNormalize'd so |q|≈kOne deterministically. axis MUST be unit (the caller passes kChassisUpAxis).
inline fpx::FxQuat QFromAxisAngleSnapped(const FxVec3& axis, fx angle) {
    const fx h = angle / 2;                         // the half-angle (Q16.16)
    const fx h2 = fxmul(h, h);                       // h²
    // cos(h) = 1 - h²/2 + h⁴/24 - h⁶/720 (Q16.16 integer Horner — fixed op order).
    const fx h4 = fxmul(h2, h2);
    const fx h6 = fxmul(h4, h2);
    const fx cosH = kOne - h2 / 2 + h4 / 24 - h6 / 720;
    // sin(h) = h - h³/6 + h⁵/120 (Q16.16 integer — fixed op order).
    const fx h3 = fxmul(h2, h);
    const fx h5 = fxmul(h3, h2);
    const fx sinH = h - h3 / 6 + h5 / 120;
    fpx::FxQuat q{fxmul(axis.x, sinH), fxmul(axis.y, sinH), fxmul(axis.z, sinH), cosH};
    return fpx::FxQuatNormalize(q);                  // deterministic |q|≈kOne
}

// ----- ApplyVehicleCommand: apply ONE vehicle input command (drive-torque / steer) ---------------------
// kCmdDriveTorque: add FxScale(kWheelSpinAxis, cmd.arg.x) to world.bodies[cmd.target].angVel (the axle
//   spin). Out-of-range target or a STATIC (!kFlagDynamic) body is a deterministic no-op.
// kCmdSteer: rotate hinges[cmd.target].axis by QFromAxisAngleSnapped(kChassisUpAxis, cmd.arg.x) about the
//   chassis up-axis (FxRotate), then re-normalize the axis. Non-front targets (∉ {0,1}) / out-of-range are
//   no-ops. (The hinge axis is body-local on the chassis A; the chassis starts at identity so the local up
//   axis IS world (0,1,0) — the VH2 layout convention.)
// Integer adds + the host-snapped quaternion only. The deterministic input event the stream is made of.
inline void ApplyVehicleCommand(Vehicle& v, const VehicleConfig& cfg, const FxCommand& cmd) {
    (void)cfg;
    // NOTE: fpx::FxCommand's target-index field is named `bodyId` (the FPX5 substrate). VH3 uses it as the
    // wheel-body / front-hinge index `target` (the spec's term) — same field, no fpx.h change.
    if (cmd.kind == kCmdDriveTorque) {
        if (cmd.bodyId >= (uint32_t)v.world.bodies.size()) return;        // out-of-range -> no-op
        FxBody& b = v.world.bodies[(size_t)cmd.bodyId];
        if (!(b.flags & kFlagDynamic)) return;                            // static -> no-op
        const FxVec3 dOmega = FxScale(kWheelSpinAxis, cmd.arg.x);         // axle spin (Q16.16)
        b.angVel = FxAdd(b.angVel, dOmega);
    } else if (cmd.kind == kCmdSteer) {
        if (cmd.bodyId >= (uint32_t)v.hinges.size()) return;            // out-of-range -> no-op
        if (!IsFrontHinge(cmd.bodyId)) return;                          // only front hinges steer
        const fpx::FxQuat q = QFromAxisAngleSnapped(kChassisUpAxis, cmd.arg.x);
        FxVec3 newAxis = FxRotate(q, v.hinges[(size_t)cmd.bodyId].axis);// re-aim the rolling plane
        const fx len = FxLength(newAxis);
        if (len != 0) newAxis = FxNormalize(newAxis);                    // keep the hinge axis unit
        v.hinges[(size_t)cmd.bodyId].axis = newAxis;
    }
    // unknown kind -> no-op (deterministic).
}

// ----- StepVehicle: ONE locked, command-driven vehicle tick (the JT3 contacts-as-a-trailing-block mold) -
// The VH2 StepVehicleRig TWO-PHASE-BLOCK tick (PHASE A spring / PHASE B hinge — the structure the EXISTING
// whole-step int64 shaders reproduce byte-for-byte) with a command-apply PROLOGUE + the JT3 contacts
// TRAILING block replacing VH2's PHASE-C ground clamp:
//   (0) APPLY this tick's commands in ARRAY ORDER (every cmd with cmd.tick == tick) via ApplyVehicleCommand.
//   PHASE A (spring, == vehicle_spring_solve.comp): IntegrateBodyFull(all, real dt) [the FPX4 6-DOF
//           integrate] -> K Gauss-Seidel passes of all SolveSpringJoint. Touches pos/vel + orientation.
//   PHASE B (hinge, == joint_angular_solve.comp, dt=0): IntegrateBodyFull(all, dt=0) [the dt=0 translation
//           integrate is a no-op; the orientation integrate runs its FxQuatNormalize — mirrors the angular
//           shader's per-step integrate] -> K Gauss-Seidel passes of all SolveAngularLimit. Touches orient.
//   PHASE D (contacts, == fpx_solve.comp, dt=0): fpx::BuildPairs ONCE over the world + fpx::StepWorld(dt=0,
//           solveIters) — the dt=0 integrate-suppressed ground + FPX3 sphere-sphere contacts so the wheels/
//           chassis rest on the ground and on each other as a coherent body. (REPLACES VH2's host PHASE-C
//           per-wheel ground clamp with real inter-body broadphase + contacts — the VH3 addition.)
// WHY the TWO-PHASE block (not a single integrate + interleaved {spring | hinge}): the existing shaders are
// WHOLE-STEP and each integrates internally; the dt=0 PHASE-B integrate is NOT bit-idempotent for
// orientation, so the CPU reference MUST call the EXACT same two integrate ops in the same order as the GPU
// composes the two whole-step shaders (the JT3/VH2 hard lesson) — otherwise GPU != CPU. Pure integer, fixed
// op order -> two-run bit-identical AND bit-exact GPU==CPU.
inline void StepVehicle(Vehicle& v, const VehicleConfig& cfg, const std::vector<FxCommand>& commands,
                        uint32_t tick, fx dt, int iters, int solveIters) {
    FxWorld& world = v.world;
    const size_t n = world.bodies.size();
    // (0) apply this tick's commands in ARRAY ORDER (deterministic input-order contract).
    for (const FxCommand& c : commands)
        if (c.tick == tick) ApplyVehicleCommand(v, cfg, c);
    // PHASE A — the spring phase (== vehicle_spring_solve.comp, SENTINEL groundY so no shader floor clamp).
    for (size_t i = 0; i < n; ++i)
        fpx::IntegrateBodyFull(world.bodies[i], world.gravity, dt);
    for (int it = 0; it < iters; ++it)
        for (size_t e = 0; e < v.springs.size(); ++e)
            SolveSpringJoint(world, v.springs[e]);
    // PHASE B — the hinge phase (== joint_angular_solve.comp, dt=0). The dt=0 IntegrateBodyFull mirrors the
    // angular shader's per-step integrate (orientation FxQuatNormalize runs; translation is a no-op).
    for (size_t i = 0; i < n; ++i)
        fpx::IntegrateBodyFull(world.bodies[i], world.gravity, /*dt=*/0);
    for (int it = 0; it < iters; ++it)
        for (size_t e = 0; e < v.hinges.size(); ++e)
            SolveAngularLimit(world, v.hinges[e]);
    // PHASE D — FPX2 broadphase ONCE + FPX3 ground + sphere contacts (dt=0 -> no re-integrate). VERBATIM fpx.
    std::vector<uint32_t> perBodyOffset;
    std::vector<fpx::FxPair> pairs;
    fpx::BuildPairs(world, perBodyOffset, pairs);
    fpx::StepWorld(world, std::span<const fpx::FxPair>(pairs), /*dt=*/0, solveIters);
}

// ----- StepVehicleSteps: run K command-driven vehicle ticks 0..K-1 (the showcase / GPU K-step driver) ---
// Each tick applies its OWN commands (cmd.tick == the tick index). The GPU driver reproduces this with the
// EXISTING shaders in the locked order; the CPU reference here is the exact byte-for-byte target.
inline void StepVehicleSteps(Vehicle& v, const VehicleConfig& cfg, const std::vector<FxCommand>& commands,
                             fx dt, int ticks, int iters, int solveIters) {
    for (int t = 0; t < ticks; ++t)
        StepVehicle(v, cfg, commands, (uint32_t)t, dt, iters, solveIters);
}

// ----- VehicleDriveState + MeasureVehicleDrive: the honest drive/steer metrics (deterministic Q16.16) ---
// chassisPos        : the chassis world position (the moved-under-drive proof).
// forwardDisp       : the chassis forward (X) displacement from a reference start-X (did it move).
// meanDrivenAngVel  : the mean |angVel| magnitude over the driven wheels (did the throttle take).
// frontHingeHeadingX: the front-hinge axis X component (the steer-re-aim proof — a steered hinge's lateral
//                     Z axle gains an X component as it rotates about the up axis; 0 at the rest heading).
// rearHingeHeadingX : the rear-hinge axis X component (UNCHANGED by a front steer — stays at rest heading).
// residualOverlaps  : the FPX3 residual-overlap count over the CURRENT broadphase pairs (non-penetration).
struct VehicleDriveState {
    FxVec3   chassisPos;
    fx       forwardDisp        = 0;
    fx       meanDrivenAngVel   = 0;
    fx       frontHingeHeadingX = 0;
    fx       rearHingeHeadingX  = 0;
    uint32_t residualOverlaps   = 0;
};

// MeasureVehicleDrive(v, cfg, drivenWheels, startChassisX): the deterministic drive/steer stats. drivenWheels
// is the list of wheel body indices the drive stream spun (the rear wheels in the showcase). startChassisX
// is the chassis X at tick 0 (the forward-displacement reference). Pure integer FxLength -> bit-exact.
inline VehicleDriveState MeasureVehicleDrive(const Vehicle& v, const VehicleConfig& cfg,
                                             const std::vector<uint32_t>& drivenWheels, fx startChassisX) {
    VehicleDriveState st;
    const FxWorld& world = v.world;
    const FxBody& chassis = world.bodies[(size_t)v.chassisIndex];
    st.chassisPos = chassis.pos;
    st.forwardDisp = chassis.pos.x - startChassisX;

    // Mean |angVel| over the driven wheels (the throttle-took metric).
    int64_t sumSpin = 0;
    uint32_t cnt = 0;
    for (uint32_t wi : drivenWheels) {
        if (wi >= (uint32_t)world.bodies.size()) continue;
        sumSpin += (int64_t)FxLength(world.bodies[(size_t)wi].angVel);
        ++cnt;
    }
    st.meanDrivenAngVel = cnt > 0u ? (fx)(sumSpin / (int64_t)cnt) : 0;

    // The front + rear hinge axis headings (the steer-re-aim proof). A front steer rotates the front axle
    // about Y so its X component grows; the rears stay at the rest lateral (0,0,1) heading (axis.x == 0).
    if (v.hinges.size() >= 4) {
        const fx f0 = v.hinges[0].axis.x, f1 = v.hinges[1].axis.x;
        st.frontHingeHeadingX = (f0 < 0 ? -f0 : f0) > (f1 < 0 ? -f1 : f1) ? f0 : f1;  // the larger-magnitude
        const fx r2 = v.hinges[2].axis.x, r3 = v.hinges[3].axis.x;
        st.rearHingeHeadingX = (r2 < 0 ? -r2 : r2) > (r3 < 0 ? -r3 : r3) ? r2 : r3;
    }

    // Residual overlaps over the CURRENT broadphase pairs (the FPX3 non-penetration coherence stat).
    std::vector<uint32_t> off;
    std::vector<fpx::FxPair> pairs;
    fpx::BuildPairs(world, off, pairs);
    st.residualOverlaps = fpx::CountResidualOverlaps(world, std::span<const fpx::FxPair>(pairs));
    (void)cfg;
    return st;
}

// ====================================================================================================
// Slice VH4 — WHEEL-GROUND TRACTION / FRICTION (the NEW-PHYSICS BEAT of FLAGSHIP #16: DETERMINISTIC
// VEHICLE PHYSICS, hf::sim::vehicle). VH3 drove the car forward with an honest velocity SEED (the
// integrate-path throttle) because fpx sphere contacts have no inertia tensor — a ground contact can
// neither spin a wheel nor be spun BY a wheel, so wheel-roll-drives-chassis is NOT emergent. VH4 replaces
// that seed with a REAL deterministic TRACTION model: at each grounded wheel a Coulomb-friction-cone-
// clamped tangential ground force converts the wheel's spin (from kCmdDriveTorque) into chassis forward
// motion — "the car actually drives". INTEGER-bit-exact. NO new shader — traction is a pure-integer HOST
// pass applied identically on the GPU and CPU paths (the VH3 command-apply precedent), folded into a NEW
// additive tick StepVehicleDriven (VH3's StepVehicle stays byte-FROZEN — VH4 is APPEND-ONLY). ADDITIVE:
// VH1/VH2/VH3 byte-unchanged. The GR4 dry-friction / joint.h SolveAngularLimit cone-clamp idiom: an
// integer min/max clamp of a slip-proportional correction to the friction cone ±kMuMax.
//
// THE PER-WHEEL TRACTION (ApplyWheelTraction, FIXED wheel-index order, Gauss-Seidel — chassis.vel updated
// in place + read by later wheels; deterministic because the order is FIXED):
//   For each WHEEL that is GROUNDED (its bottom wheel.pos.y - cfg.wheelRadius <= cfg.groundY + kContactEps):
//   1. fwd = FxNormalize(cross(groundNormal=(0,kOne,0), axleWorld)) where axleWorld = the wheel spin axle
//      (the hinge's CURRENT axis — STEERED front wheels push along their re-aimed heading; the hinge axis
//      lives on the chassis in VH2's body-local-at-identity frame, == the world axle). |fwd|==0 -> skip.
//   2. vTarget = fxmul(spinAboutAxle, cfg.wheelRadius) where spinAboutAxle = FxDot(wheel.angVel, axleWorld)
//      — the wheel's contact-patch surface speed (spin × radius), the no-slip ground speed.
//   3. slip = vTarget - FxDot(chassis.vel, fwd) (target minus the chassis's current forward ground speed).
//   4. j = clamp(fxmul(slip, kGripK), -kMuMax, +kMuMax) — a stiffness-scaled correction toward no-slip,
//      CLAMPED to the friction cone (the Coulomb limit: large slip saturates). Apply chassis.vel +=
//      FxScale(fwd, fxmul(j, kChassisShare)) (the chassis accelerates toward rolling) AND bleed the wheel
//      spin by FxScale(axleWorld, -fxmul(j, kWheelBleed)) (momentum leaves the spin as the tyre grips).
//   Pure integer (cross/FxDot/FxNormalize/fxmul/FxScale + integer clamp), NO transcendentals.
//
// StepVehicleDriven = VH3 StepVehicle's EXACT phase order (apply tick commands -> PHASE A spring -> PHASE B
// hinge) + NEW PHASE C ApplyWheelTraction (between hinge and contacts, after the wheels' grounded state is
// set by integrate+constraints, before the contact solve) + PHASE D fpx contacts. Keep it in LOCKSTEP with
// StepVehicle (a comment cross-refs it) — only PHASE C is inserted.
//
// HONEST CAVEAT (the GR4/JT2 caveat): traction is a deterministic tangential PROXY (a slip-proportional
// cone-clamped force), NOT analytic tyre mechanics (no Pacejka slip curves, no lateral/longitudinal slip
// separation, no load-dependent grip, no differential/ABS/rolling-resistance/aero). The chassis has no
// inertia tensor (fpx limitation) so the car does not pitch/roll under traction — planar drive only. The
// headline is DETERMINISM + cross-platform bit-identity + "it drives from spin, not a seed".

// ----- VH4 host-fixed Q16.16 traction constants (documented; the GR4/cone-clamp tuning) ----------------
// kContactEps : the ground-contact band — a wheel is "grounded" if its bottom is within this of groundY
//               (kOne/16 = 0.0625 world units — generous enough to catch the settled-suspension wheel,
//               tight enough not to engage an airborne wheel).
// kGripK      : the slip->force stiffness (the per-tick correction fraction toward no-slip; kOne/4 = 0.25
//               — a soft correction so the car accelerates over several ticks, not a single jolt).
// kMuMax      : the friction cone (the max tangential delta-velocity per tick a tyre can transmit; kOne/8
//               = 0.125 world units/s — the Coulomb limit that bounds the traction; a kMuMax==0 control
//               leaves the car idle DESPITE spinning wheels — the proof the cone, not a seed, drives it).
// kChassisShare : the fraction of the cone-clamped impulse the CHASSIS gains (3*kOne/4 = 0.75 — most of
//               the gripped force pushes the body forward).
// kWheelBleed : the fraction of the impulse bled OFF the wheel spin as the tyre grips (kOne/4 = 0.25 —
//               momentum leaves the spin into forward motion; a freely-spinning driven wheel loses spin
//               as it grips, the chassis gains speed — the momentum-transfer signal).
inline constexpr fx kContactEps   = kOne / 16;       // 0.0625 ground-contact band
inline constexpr fx kGripK        = kOne / 4;        // 0.25 slip->force stiffness
inline constexpr fx kMuMax        = kOne / 8;        // 0.125 friction-cone limit (the Coulomb cap)
inline constexpr fx kChassisShare = (fx)(3 * (int)kOne / 4);  // 0.75 chassis impulse share
inline constexpr fx kWheelBleed   = kOne / 4;        // 0.25 wheel-spin bleed fraction

// ----- FxCrossLocal: the Q16.16 cross product a×b (inline — fpx.h has none; the FxRotate idiom) ---------
// Each term is an int64 fxmul difference (the swraster/mc int64 discipline). NOT added to fpx.h (frozen).
inline FxVec3 FxCrossLocal(const FxVec3& a, const FxVec3& b) {
    return FxVec3{
        fxmul(a.y, b.z) - fxmul(a.z, b.y),
        fxmul(a.z, b.x) - fxmul(a.x, b.z),
        fxmul(a.x, b.y) - fxmul(a.y, b.x),
    };
}

// ----- FxClampCone: the integer Coulomb-cone clamp of a scalar to [-lim, +lim] (the GR4/JT2 idiom) ------
inline fx FxClampCone(fx v, fx lim) {
    if (v >  lim) return  lim;
    if (v < -lim) return -lim;
    return v;
}

// ----- ApplyWheelTraction: the per-wheel Coulomb-cone tangential ground-traction pass (the HOST beat) ---
// Runs the per-wheel traction documented above over all four wheels in FIXED index order (Gauss-Seidel:
// the chassis vel is updated in place, read by later wheels — deterministic because the order is fixed).
// Pure integer, fixed op order -> two-run bit-identical AND identical on the GPU host path (the VH3
// command-apply precedent). The chassis is world.bodies[chassisIndex]; the wheels are world.bodies[
// wheelIndex[k]] with the matching hinge axle hinges[k].axis (the steered front wheels' re-aimed heading).
inline void ApplyWheelTraction(Vehicle& v, const VehicleConfig& cfg) {
    FxWorld& world = v.world;
    if (v.chassisIndex >= (uint32_t)world.bodies.size()) return;
    FxBody& chassis = world.bodies[(size_t)v.chassisIndex];
    const FxVec3 groundNormal{0, kOne, 0};
    for (int k = 0; k < 4; ++k) {
        const uint32_t wi = v.wheelIndex[k];
        if (wi >= (uint32_t)world.bodies.size()) continue;          // out-of-range -> skip
        if ((size_t)k >= v.hinges.size()) continue;                // no axle -> skip
        FxBody& wheel = world.bodies[(size_t)wi];
        // GROUNDED test: the wheel bottom within the contact band of the ground.
        const fx bottom = wheel.pos.y - cfg.wheelRadius;
        if (bottom > cfg.groundY + kContactEps) continue;          // airborne -> no traction
        // (1) the rolling-forward direction = ground-up × axle (the hinge's CURRENT axis -> world).
        const FxVec3 axleWorld = v.hinges[(size_t)k].axis;         // VH2 body-local-at-identity == world axle
        const FxVec3 fwdRaw = FxCrossLocal(groundNormal, axleWorld);
        if (FxLength(fwdRaw) == 0) continue;                       // degenerate (axle ∥ up) -> skip
        const FxVec3 fwd = FxNormalize(fwdRaw);
        // (2) the no-slip target ground speed = (spin about the axle) × wheelRadius.
        const fx spinAboutAxle = FxDot(wheel.angVel, axleWorld);
        const fx vTarget = fxmul(spinAboutAxle, cfg.wheelRadius);
        // (3) the slip = target minus the chassis's current forward ground speed.
        const fx slip = vTarget - FxDot(chassis.vel, fwd);
        // (4) the cone-clamped traction impulse + the chassis accel + the wheel-spin bleed.
        const fx j = FxClampCone(fxmul(slip, kGripK), kMuMax);
        chassis.vel = FxAdd(chassis.vel, FxScale(fwd, fxmul(j, kChassisShare)));
        // bleed the wheel spin along the axle (momentum leaves the spin as the tyre grips).
        wheel.angVel = FxSub(wheel.angVel, FxScale(axleWorld, fxmul(j, kWheelBleed)));
    }
}

// ----- StepVehicleDriven: ONE traction-driven vehicle tick (VH3 StepVehicle + PHASE C ApplyWheelTraction)
// CROSS-REF: this is vehicle::StepVehicle (above) with ONE pass inserted — PHASE C ApplyWheelTraction
// between PHASE B (hinge) and PHASE D (contacts). KEEP THESE TWO IN LOCKSTEP: the phase order + the
// integrate calls (PHASE A integrate at dt, PHASE B integrate at dt=0) MUST match StepVehicle byte-for-byte
// so the GPU showcase's two-shader composition (vehicle_spring_solve + joint_angular_solve) stays bit-exact
// against this CPU reference; PHASE C is a pure HOST velocity adjustment applied between the GPU constraint
// dispatches and the GPU contact dispatch (the VH3 command-apply seam). Pure integer, fixed op order ->
// two-run bit-identical AND bit-exact GPU==CPU.
inline void StepVehicleDriven(Vehicle& v, const VehicleConfig& cfg, const std::vector<FxCommand>& commands,
                              uint32_t tick, fx dt, int iters, int solveIters) {
    FxWorld& world = v.world;
    const size_t n = world.bodies.size();
    // (0) apply this tick's commands in ARRAY ORDER (deterministic input-order contract).
    for (const FxCommand& c : commands)
        if (c.tick == tick) ApplyVehicleCommand(v, cfg, c);
    // PHASE A — the spring phase (== vehicle_spring_solve.comp, SENTINEL groundY so no shader floor clamp).
    for (size_t i = 0; i < n; ++i)
        fpx::IntegrateBodyFull(world.bodies[i], world.gravity, dt);
    for (int it = 0; it < iters; ++it)
        for (size_t e = 0; e < v.springs.size(); ++e)
            SolveSpringJoint(world, v.springs[e]);
    // PHASE B — the hinge phase (== joint_angular_solve.comp, dt=0). The dt=0 IntegrateBodyFull mirrors the
    // angular shader's per-step integrate (orientation FxQuatNormalize runs; translation is a no-op).
    for (size_t i = 0; i < n; ++i)
        fpx::IntegrateBodyFull(world.bodies[i], world.gravity, /*dt=*/0);
    for (int it = 0; it < iters; ++it)
        for (size_t e = 0; e < v.hinges.size(); ++e)
            SolveAngularLimit(world, v.hinges[e]);
    // PHASE C — THE VH4 BEAT: the Coulomb-cone wheel-ground traction (a pure HOST velocity adjustment, the
    // VH3 command-apply seam; applied after the wheels' grounded state is set, before the contact solve).
    ApplyWheelTraction(v, cfg);
    // PHASE D — FPX2 broadphase ONCE + FPX3 ground + sphere contacts (dt=0 -> no re-integrate). VERBATIM fpx.
    std::vector<uint32_t> perBodyOffset;
    std::vector<fpx::FxPair> pairs;
    fpx::BuildPairs(world, perBodyOffset, pairs);
    fpx::StepWorld(world, std::span<const fpx::FxPair>(pairs), /*dt=*/0, solveIters);
}

// ----- StepVehicleDrivenSteps: run K traction-driven vehicle ticks 0..K-1 (the showcase / GPU driver) ---
// Each tick applies its OWN commands (cmd.tick == the tick index). The GPU driver reproduces this with the
// EXISTING shaders in the locked order + the host PHASE-C traction; the CPU reference here is the exact
// byte-for-byte target.
inline void StepVehicleDrivenSteps(Vehicle& v, const VehicleConfig& cfg,
                                   const std::vector<FxCommand>& commands, fx dt, int ticks, int iters,
                                   int solveIters) {
    for (int t = 0; t < ticks; ++t)
        StepVehicleDriven(v, cfg, commands, (uint32_t)t, dt, iters, solveIters);
}

// ====================================================================================================
// Slice VH5 — LOCKSTEP + ROLLBACK (THE NETCODE HEADLINE of FLAGSHIP #16: DETERMINISTIC VEHICLE PHYSICS,
// hf::sim::vehicle). VH1-VH4 built a drivable car (suspension -> rig -> drive/steer -> traction). VH5
// proves the bit-exact driven tick (StepVehicleDriven) is true cross-platform LOCKSTEP + ROLLBACK — the
// FPX5/FR5/GR5/CG5/JT5 twin. Two peers fed ONLY the input command stream (NOT full state) re-derive the
// authority's exact car trajectory bit-for-bit; a mispredicted input is corrected by rolling back to a
// saved snapshot + re-simulating. PURE CPU (NO GPU dispatch, NO new shader, NO new RHI). ADDITIVE — the
// VH1-VH4 code above is byte-FROZEN; VH5 only APPENDS the snapshot + the three harness functions.
//
// THE ONE VH-SPECIFIC TWIST — the snapshot must include the HINGE AXES. The ragdoll JT5 snapshot was just
// the fpx::FxWorld (bodies); JT5 reused fpx::SnapshotWorld/RestoreWorld VERBATIM. But a vehicle's kCmdSteer
// MUTATES hinges[i].axis (the steered heading) and ApplyWheelTraction READS those axes — so the four hinge
// axes are LIVE replayable state. VehicleSnapshot therefore captures BOTH the body world (world.bodies, via
// fpx::SnapshotWorld) AND the four hinges[i].axis. RestoreVehicle restores both. The springs are immutable
// (restLen/stiffness/damping never change) so they are NOT snapshotted; the chassis/wheel indices are
// structural constants. (We capture EXACTLY what mutates: bodies + hinge axes.)
//
// SEAM DISCIPLINE: unchanged — ZERO backend symbols, header-only, PURE CPU. NO new shader, NO new RHI. VH5
// reuses fpx::SnapshotWorld/RestoreWorld read-only for the body half (fpx.h FROZEN) and adds the hinge axes
// alongside; the three harness functions are the JT5 SimRagdollTick/RunRagdollLockstep/RunRagdollRollback
// twins with StepVehicleDriven as the per-tick step.

// ----- VehicleSnapshot: the captured mutable vehicle state (bodies + the 4 steered hinge axes) ----------
// The body world (deep-copied via fpx::SnapshotWorld — the std::vector<FxBody> + gravity/groundY scalars)
// PLUS the four hinge axes (the steered headings kCmdSteer mutates + ApplyWheelTraction reads). Captures
// EXACTLY the mutable state — springs (immutable) + the structural indices are NOT part of the snapshot.
struct VehicleSnapshot {
    fpx::FxWorld bodies;        // the body world (fpx::SnapshotWorld deep-copy: bodies + gravity/groundY)
    FxVec3       hingeAxis[4];  // the 4 steered hinge axes (kCmdSteer-mutated, ApplyWheelTraction-read)
};

// ----- SnapshotVehicle: deep-copy the mutable vehicle state (the rollback restore point) ----------------
// Reuses fpx::SnapshotWorld VERBATIM for the bodies (a value copy -> deep-copies the bodies vector) + copies
// the four hinge axes. Bit-exact round-trip with RestoreVehicle. The four hinge slots are captured from the
// FIRST four hinges in the fixed VH2 corner order (0/1 front, 2/3 rear); a vehicle always has 4.
inline VehicleSnapshot SnapshotVehicle(const Vehicle& v) {
    VehicleSnapshot snap;
    snap.bodies = fpx::SnapshotWorld(v.world);
    for (int k = 0; k < 4; ++k)
        snap.hingeAxis[k] = ((size_t)k < v.hinges.size()) ? v.hinges[(size_t)k].axis : FxVec3{0, 0, 0};
    return snap;
}

// ----- RestoreVehicle: restore the mutable vehicle state from a snapshot (the rollback) -----------------
// Restores BOTH the body world (fpx::RestoreWorld) AND the four hinge axes. Bit-exact round-trip:
// RestoreVehicle(v, SnapshotVehicle(v0)) leaves v's bodies + hinge axes == v0's byte-for-byte. Springs +
// indices are untouched (immutable / structural).
inline void RestoreVehicle(Vehicle& v, const VehicleSnapshot& snap) {
    fpx::RestoreWorld(v.world, snap.bodies);
    for (int k = 0; k < 4; ++k)
        if ((size_t)k < v.hinges.size()) v.hinges[(size_t)k].axis = snap.hingeAxis[k];
}

// ----- SimVehicleTick: the deterministic per-tick step (the JT5 SimRagdollTick twin) -------------------
// One StepVehicleDriven step — which ALREADY applies every command with cmd.tick == tick in ARRAY ORDER at
// its prologue (the VH3/VH4 contract), then runs PHASE A spring / PHASE B hinge / PHASE C traction / PHASE D
// contacts. Pure integer, fixed op order -> bit-identical on every peer/platform. (No new command type —
// reuses the vehicle-local kCmdDriveTorque/kCmdSteer through ApplyVehicleCommand inside StepVehicleDriven.)
inline void SimVehicleTick(Vehicle& v, const VehicleConfig& cfg, const std::vector<FxCommand>& commands,
                           uint32_t tick, fx dt, int iters, int solveIters) {
    StepVehicleDriven(v, cfg, commands, tick, dt, iters, solveIters);
}

// ----- RunVehicleLockstep: authority + replica from the SAME inputs, bit-identical every tick ----------
// THE peer entry point (the JT5 RunRagdollLockstep / fpx::RunLockstep control flow over SimVehicleTick).
// Run `ticks` SimVehicleTicks from a COPY of `initialVehicle`, applying the command stream -> the converged
// car. authority = RunVehicleLockstep(cfg, init, commands, N, ...); replica = RunVehicleLockstep(cfg, init,
// commands, N, ...) from the SAME init + stream (INPUTS ONLY — no state shared) -> BIT-IDENTICAL by
// determinism (the lockstep proof memcmps the bodies + the hinge axes). cfg is the CONSTANT config (NOT
// snapshotted). Returns the converged authority Vehicle (the caller memcmps two runs for the proof).
inline Vehicle RunVehicleLockstep(const VehicleConfig& cfg, const Vehicle& initialVehicle,
                                  const std::vector<FxCommand>& commands, int ticks, fx dt, int iters,
                                  int solveIters) {
    Vehicle v = initialVehicle;
    for (int t = 0; t < ticks; ++t)
        SimVehicleTick(v, cfg, commands, (uint32_t)t, dt, iters, solveIters);
    return v;
}

// ----- RunVehicleRollback: snapshot -> mispredict diverges -> rollback -> corrected == authority --------
// The rollback harness (the JT5 RunRagdollRollback / fpx::RunRollback control flow over SimVehicleTick).
// (1) advance ticks 0..divergeTick from `initialVehicle` applying authorityCommands; (2) SAVE a
// VehicleSnapshot AT divergeTick (SnapshotVehicle — the bodies + the STEERED hinge axes); (2b) speculatively
// advance <=3 ticks with the MISPREDICTED stream (the wrong drive/steer — the client prediction that
// diverges); (3) ROLLBACK — RestoreVehicle to the snapshot (restoring the bodies AND the steered hinge
// axes) + RE-SIMULATE divergeTick..ticks with the CORRECT authorityCommands -> the corrected final car. The
// proof asserts this == RunVehicleLockstep(cfg, init, authorityCommands, ticks) (rollback corrected the
// misprediction EXACTLY) AND that the speculative pre-rollback state DIFFERED from authority (a real
// divergence — including a steered-hinge divergence — was fixed). NOTE: snapshotTick == divergeTick (the
// restore point IS the divergence point, the JT5 mispredictTick convention); we keep a single divergeTick
// parameter so the snapshot is taken at exactly the tick the misprediction begins. Reuses SnapshotVehicle/
// RestoreVehicle (which carry the hinge axes). cfg is CONSTANT, NOT snapshotted.
inline Vehicle RunVehicleRollback(const VehicleConfig& cfg, const Vehicle& initialVehicle,
                                  const std::vector<FxCommand>& authorityCommands,
                                  const std::vector<FxCommand>& mispredictCommands, int snapshotTick,
                                  int divergeTick, int ticks, fx dt, int iters, int solveIters) {
    (void)snapshotTick;   // the restore point is divergeTick (snapshotTick == divergeTick by contract)
    Vehicle v = initialVehicle;
    // (1) advance 0..divergeTick with the authoritative stream.
    for (int t = 0; t < divergeTick; ++t)
        SimVehicleTick(v, cfg, authorityCommands, (uint32_t)t, dt, iters, solveIters);
    // (2) SAVE the snapshot at divergeTick (the rollback restore point — bodies + STEERED hinge axes).
    const VehicleSnapshot snap = SnapshotVehicle(v);
    // (2b) speculatively advance a few ticks with the MISPREDICTED stream (bounded to the remaining ticks).
    int specTicks = ticks - divergeTick;
    if (specTicks > 3) specTicks = 3;
    for (int s = 0; s < specTicks; ++s)
        SimVehicleTick(v, cfg, mispredictCommands, (uint32_t)(divergeTick + s), dt, iters, solveIters);
    // (3) ROLLBACK: restore the snapshot (bodies + hinge axes) + re-sim divergeTick..ticks with authStream.
    RestoreVehicle(v, snap);
    for (int t = divergeTick; t < ticks; ++t)
        SimVehicleTick(v, cfg, authorityCommands, (uint32_t)t, dt, iters, solveIters);
    return v;
}

}  // namespace vehicle
}  // namespace hf::sim
