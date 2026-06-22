#pragma once
// Slice HF4 — Hull Friction + Joints: HULL JOINTS COMPOSED (the one-deterministic-step headline; the FOURTH slice of
// FLAGSHIP #30: HULL FRICTION + HULL JOINTS, hf::sim::hulljoint). HF1–HF3 brought tangential friction to the general
// convex-hull world (the tagged manifold → the warm cone solver → the friction-locked step). HF4 brings JOINTS: the
// body-agnostic joint.h solvers (ball / hinge / angular-limit), already proven bit-exact for the articulated-ragdoll
// flagship (#15), now operating over gjk::HullWorld.bodies AND composed with the hull friction contacts in ONE
// deterministic tick — a chain of jointed hulls, a hinged hull door, settling with friction. The flagship's real new
// value beyond reuse is the COMPOSITION (joints + friction + contacts in one pinned deterministic step), bit-identical
// CPU/Vulkan/Metal.
//
// THE DESIGN CALL (transplant the proven articulated step onto the hull friction step; pin the order): joint.h's
// SolveBallJoint (the anchor-coincidence positional projection) and SolveAngularLimit (the swing-twist cone clamp) are
// ALREADY body-index based over an fpx::FxWorld.bodies vector of fpx::FxBody — which is EXACTLY gjk::FxBody (a using
// alias), the hull body type. And joint::StepArticulatedContacts ALREADY composes those joint passes with rigid
// contacts in one tick (integrate → K {ball | angular} → broadphase → contacts → de-pen → orient). HF4 does NOT write
// new constraint math: it builds JointedHullWorld (the hull world + the joint lists + the HF3 friction cache) and
// StepJointedHullWorld that runs the HF3 StepWarmFrictionHullWorld body with the joint passes inserted where
// StepArticulatedContacts puts them — between the integrate and the friction-contact warm solve.
//
// THE COMPOSITION ORDERING (the crux — PINNED): (1) predict-integrate every dynamic body → (2) cfg.jointIters
// Gauss-Seidel sweeps EACH doing {all SolveBallJoint in FIXED list order, then all SolveAngularLimit in FIXED list
// order} → (3) the HF3 FULL inertia + the friction+contact warm solve over the all-pairs i<j manifolds (in place) +
// the cache rewrite → (4) position de-penetration → (5) orientation (integrated in (1)). Joints first, then the
// friction-contact velocity solve, then the de-pen — the joint::StepArticulatedContacts order verbatim, so the joint
// pass and the friction pass never write the same body field in an order-ambiguous way within a sweep. FIXED
// joint-list order + FIXED i<j contact order + in-place mutation → two-run + GPU==CPU bit-identity by construction
// (the cloth/JT discipline). With EMPTY joint/limit lists the joint passes are no-ops, so StepJointedHullWorld is
// BYTE-FOR-BYTE the frozen HF3 StepWarmFrictionHullWorld (the no-joints control — joints are purely additive).
//
// THE BODY-BRIDGE (no copy, byte-exact): the joint solvers want an fpx::FxWorld&; the hull world carries the same
// std::vector<fpx::FxBody>. We MOVE the bodies into a scratch fpx::FxWorld for the joint sweeps, run the passes, then
// MOVE them back — the same FxBody objects, mutated in place, no value drift. (The joint passes read ONLY .bodies; the
// scratch world's gravity/groundY are unused — joints do NOT integrate or ground-clamp here; HF4's integrate is the
// HF3 fpx::IntegrateBodyFull and there is NO ground clamp, a static floor hull provides the ground via friction
// contacts.)
//
// THE int64 REALITY (the HF1–HF3 + JT lesson): the whole chain is int64 (GJK/EPA + the SH clip + the full inertia +
// the cone solve + FxLength + the joint FxRotate/FxQuatMul/FxQuatNormalize). DXC -spirv compiles int64; glslc CANNOT,
// so shaders/hulljoint_step.comp is VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl), single-thread whole-step, chunked 1
// tick/dispatch (the Windows ~2s TDR rule). The Metal --hf4-joint runs THIS CPU StepJointedHullWorldN (byte-identical
// by construction), while the Vulkan --hf4-joint-shot carries the GPU==CPU memcmp.
//
// Header-only, namespace hf::sim::hulljoint, #include sim/hullfric.h + sim/joint.h + sim/warmhull.h + sim/gjk.h +
// sim/fpx.h READ-ONLY (ALL BYTE-FROZEN — hulljoint.h is a brand-new additive sibling that NEVER edits a frozen
// header).

#include <cstdint>
#include <utility>
#include <vector>

#include "sim/hullfric.h"   // read-only/BYTE-FROZEN: StepWarmFrictionHullWorld(N) (the HF3 step HF4 mirrors), the
                            // HullFrictionStepConfig / HullFrictionCache / HullFrictionConfig, BuildHullFrictionManifold,
                            // MatchHullFrictionCache / UpdateHullFrictionCache, SolveHullFrictionWarm, MeasureHullGrip.
#include "sim/joint.h"      // read-only/BYTE-FROZEN: FxJoint / FxAngularLimit, SolveBallJoint / SolveAngularLimit,
                            // AnchorGap / SwingAngleCos (the joint constraints + their coherence metrics).
#include "sim/warmhull.h"   // read-only/BYTE-FROZEN: the frozen narrowphase the HF3 step rides on.
#include "sim/gjk.h"        // read-only: HullWorld / FxBody (== fpx::FxBody) / FxHull.
#include "sim/fpx.h"        // read-only: FxWorld / FxBody / IntegrateBodyFull (the body-bridge scratch world).

namespace hf::sim {
namespace hulljoint {

// Pull the frozen helpers into this namespace (REUSE, do NOT redefine).
using gjk::fx;
using gjk::kOne;
using gjk::FxVec3;
using gjk::FxHull;
using gjk::FxBody;
using gjk::HullWorld;
using convex::FxDot;
using convex::FxMat3;
using convex::IsDynamic;
using fpx::FxScale;
using manifold::FxHullFaces;
using manifold::BuildCanonicalFaces;
using manifold::FxHullInertiaBodyFull;
using manifold::WorldInvInertiaFull;
using manifold::HullContactMulti;

// ----- JointedHullWorld: a gjk::HullWorld + the joint/limit constraint lists + the HF3 persistent friction cache ----
// hulls = the hull bodies + collision hulls (the HF3 world). joints = the ball-joint list (FIXED order); limits = the
// angular-limit (hinge/cone) list (FIXED order); both indexed into hulls.bodies. cache = the HF3 friction cache,
// PERSISTED across ticks. The joints are stateless constraints (re-solved each tick); the friction cache persists.
// They compose in the one step with no shared mutable state beyond the bodies.
struct JointedHullWorld {
    gjk::HullWorld                       hulls;
    std::vector<joint::FxJoint>          joints;
    std::vector<joint::FxAngularLimit>   limits;
    hullfric::HullFrictionCache          cache;
};

// ----- JointedHullStepConfig: the HF3 friction-step config + the joint-pass iteration count ----------------------
// fric = the HF3 HullFrictionStepConfig (mu / gravity / dt / solveIters / posIters / restitution / slop / beta /
// linDamp / angDamp). jointIters = the Gauss-Seidel joint-pass sweep count (the JT iters knob; more iters → a tighter
// chain, the cloth/JT residual caveat).
struct JointedHullStepConfig {
    hullfric::HullFrictionStepConfig fric;
    int                              jointIters = 8;
};

// ----- StepJointedHullWorld(w, cfg): ONE deterministic JOINTED + friction-locked WARM-started tick ----------------
// The frozen HF3 hullfric::StepWarmFrictionHullWorld body, with the joint constraint passes inserted in the
// joint::StepArticulatedContacts position — between the integrate (step 1) and the FULL-inertia friction-contact warm
// solve (step 2/3). Steps (1) predict-integrate + damp, (2) FULL inertia, (3) the warm cone solve over the persistent
// friction cache, (4) position de-pen are the HF3 passes VERBATIM (so the EMPTY-joints control is byte-identical to
// HF3). The joint passes (step J) run cfg.jointIters sweeps of {all SolveBallJoint, then all SolveAngularLimit} in
// FIXED list order over a scratch fpx::FxWorld that BORROWS the hull bodies (moved in/out — the same FxBody objects).
// PURE INTEGER, FIXED order → identical CPU/GPU. The shader copies THIS body VERBATIM.
inline void StepJointedHullWorld(JointedHullWorld& w, const JointedHullStepConfig& cfg) {
    gjk::HullWorld& world = w.hulls;
    hullfric::HullFrictionCache& cache = w.cache;
    const hullfric::HullFrictionStepConfig& fc = cfg.fric;
    const size_t n = world.bodies.size();

    // (1) predict-integrate dynamic bodies + per-tick damping (== StepWarmFrictionHullWorld step 1, VERBATIM).
    for (size_t i = 0; i < n; ++i) {
        if (convex::IsDynamic(world.bodies[i])) {
            fpx::IntegrateBodyFull(world.bodies[i], fc.gravity, fc.dt);
            if (fc.linDamp != kOne) world.bodies[i].vel    = FxScale(world.bodies[i].vel, fc.linDamp);
            if (fc.angDamp != kOne) world.bodies[i].angVel = FxScale(world.bodies[i].angVel, fc.angDamp);
        }
    }

    // (J — THE JOINT PASSES) cfg.jointIters Gauss-Seidel sweeps EACH doing {all SolveBallJoint (position) in FIXED
    // list order, then all SolveAngularLimit (orientation) in FIXED list order} (== joint::StepArticulatedContacts
    // step 2, VERBATIM). The joint solvers index an fpx::FxWorld.bodies — BORROW the hull bodies (move in/out, the
    // same FxBody objects, no value drift). With EMPTY lists this whole block is a no-op (the no-joints control).
    if (!w.joints.empty() || !w.limits.empty()) {
        fpx::FxWorld jw;
        jw.bodies = std::move(world.bodies);          // move the bodies in (no copy)
        for (int it = 0; it < cfg.jointIters; ++it) {
            for (size_t e = 0; e < w.joints.size(); ++e) joint::SolveBallJoint(jw, w.joints[e]);
            for (size_t e = 0; e < w.limits.size(); ++e) joint::SolveAngularLimit(jw, w.limits[e]);
        }
        world.bodies = std::move(jw.bodies);          // move them back (mutated in place)
    }

    // (2) world inverse inertias once/tick — the FULL tensor (== StepWarmFrictionHullWorld step 2, VERBATIM).
    std::vector<FxMat3> invIW(n);
    for (size_t i = 0; i < n; ++i) {
        const FxHullFaces faces = BuildCanonicalFaces(world.hulls[i]);
        const FxMat3 invIbody = FxHullInertiaBodyFull(world.hulls[i], faces, world.bodies[i].invMass);
        invIW[i] = WorldInvInertiaFull(world.bodies[i], invIbody);
    }

    // (3) the warm cone solve over the persistent friction cache, FIXED i<j order, IN PLACE (== HF3 step 3, VERBATIM).
    hullfric::HullFrictionConfig solveCfg;
    solveCfg.mu          = fc.mu;
    solveCfg.restitution = fc.restitution;
    solveCfg.iters       = fc.solveIters;
    std::vector<hullfric::HullFrictionManifold> solved;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (world.bodies[i].invMass == 0 && world.bodies[j].invMass == 0) continue;   // static-static
            hullfric::HullFrictionManifold m = hullfric::BuildHullFrictionManifold(
                (uint32_t)i, world.bodies[i], world.hulls[i],
                (uint32_t)j, world.bodies[j], world.hulls[j]);
            if (m.count == 0) continue;
            hullfric::MatchHullFrictionCache(cache, m);   // warm-seed (key + basis-axis)
            hullfric::SolveHullFrictionWarm(world.bodies[i], world.bodies[j], m, invIW[i], invIW[j], solveCfg);
            solved.push_back(m);
        }
    }

    // (3a) rewrite the cache to EXACTLY this tick's solved contacts (== HF3 step 5a, VERBATIM).
    cache.entries.clear();
    for (const hullfric::HullFrictionManifold& m : solved) hullfric::UpdateHullFrictionCache(cache, m);

    // (4) position de-penetration (== StepWarmFrictionHullWorld step 4, VERBATIM — re-derives HullContactMulti).
    for (uint32_t pit = 0; pit < fc.posIters; ++pit) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                const fx invSum = world.bodies[i].invMass + world.bodies[j].invMass;
                if (invSum == 0) continue;
                const convex::ContactManifold m = HullContactMulti(world.bodies[i], world.hulls[i],
                                                                   world.bodies[j], world.hulls[j]);
                if (m.count == 0) continue;
                FxVec3 nrm = m.normal;
                if (FxDot(nrm, fpx::FxSub(world.bodies[j].pos, world.bodies[i].pos)) < 0)
                    nrm = FxVec3{-nrm.x, -nrm.y, -nrm.z};
                const fx excess = m.depths[0] - fc.slop;
                if (excess <= 0) continue;
                const fx corrected = fpx::fxmul(excess, fc.beta);
                const fx wi = fpx::fxdiv(world.bodies[i].invMass, invSum);
                const fx wj = kOne - wi;
                const FxVec3 ci = FxScale(nrm, fpx::fxmul(corrected, wi));
                const FxVec3 cj = FxScale(nrm, fpx::fxmul(corrected, wj));
                world.bodies[i].pos = fpx::FxSub(world.bodies[i].pos, ci);
                world.bodies[j].pos = fpx::FxAdd(world.bodies[j].pos, cj);
            }
        }
    }
    // (5) orientation was already integrated in step (1) (+ adjusted by the angular-limit passes in step J).
}

// ----- StepJointedHullWorldN(w, cfg, ticks): run `ticks` jointed friction-locked steps -----------------------------
// The cache carries the accumulated friction impulses ACROSS ticks (the warm-start that locks a resting grip); the
// joint/limit lists are constant across ticks (re-solved each tick). Mirrors the chunked 1-tick/dispatch the GPU does
// (TDR-safe) — bit-identical to one big run by construction.
inline void StepJointedHullWorldN(JointedHullWorld& w, const JointedHullStepConfig& cfg, uint32_t ticks) {
    for (uint32_t t = 0; t < ticks; ++t) StepJointedHullWorld(w, cfg);
}

// ----- JointMeasure: the deterministic coherence summary of a settled jointed hull world ----------------------------
// maxAnchorGap = the largest ball-joint world-anchor separation over all joints (small after settling → the chain
// stayed CONNECTED, not flying apart — joint::MaxAnchorGap). swingAngleCos = the MINIMUM SwingAngleCos over all
// angular limits (the worst hinge/cone swing — >= the limit's cosHalfLimit means the swing stayed within the cone;
// for a hinge cosHalfLimit==kOne so a held hinge keeps this near kOne). Pure integer (the joint.h metrics) →
// bit-exact CPU<->GPU. (If there are no limits, swingAngleCos defaults to kOne — vacuously within any cone.)
struct JointMeasure {
    fx maxAnchorGap  = 0;
    fx swingAngleCos = kOne;
};

// MeasureJointedHull(w): the largest ball-joint anchor gap + the worst-case (minimum) hinge/cone swing cosine. The
// joint solvers read an fpx::FxWorld& — BORROW the hull bodies read-only via a scratch world (copied here, since the
// measure is read-only and small; a copy keeps `w` const-correct). Pure integer, fixed order → deterministic.
inline JointMeasure MeasureJointedHull(const JointedHullWorld& w) {
    JointMeasure jm;
    fpx::FxWorld jw;
    jw.bodies = w.hulls.bodies;   // read-only copy for the const metric (joint.h takes FxWorld&)
    jm.maxAnchorGap = joint::MaxAnchorGap(jw, w.joints);
    bool first = true;
    for (const joint::FxAngularLimit& lim : w.limits) {
        const fx c = joint::SwingAngleCos(jw, lim);
        if (first) { jm.swingAngleCos = c; first = false; }
        else if (c < jm.swingAngleCos) jm.swingAngleCos = c;
    }
    return jm;
}

}  // namespace hulljoint
}  // namespace hf::sim
