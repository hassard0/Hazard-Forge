// Slice HF2 — Hull Friction + Joints: THE WARM CONE SOLVER (the friction SOLVER of FLAGSHIP #30: HULL FRICTION +
// HULL JOINTS, hf::sim::hullfric). The integer core (engine/sim/hullfric.h, APPENDED after HF1) the GPU
// shaders/hullfric_warm.comp.hlsl copies VERBATIM + proves bit-identical. HF2 adds the accumulated, warm-started,
// cone-clamped Coulomb friction solve over the HF1 hull manifold — the persist::SolveFrictionWarm body with the hull
// manifold::WorldInvInertiaFull tensor (exactly how warmhull::SolveHullManifoldWarm adapted the normal-only warm
// solver). #includes warmhull/gjk/fric/persist/manifold READ-ONLY (ALL BYTE-FROZEN).
//
// What this test PINS (the contracts the GPU hullfric_warm.comp + the GPU==CPU proof build on, the spec proofs):
//   * SolveHullFrictionWarm CONVERGES: the post-solve residual DECREASES (monotone non-increasing) with iters.
//   * the WARM-START LEVER: a cache-PRIMED (warm) solve reaches a STRICTLY LOWER residual than a COLD (zero-seeded)
//     solve at equal iters — the headline the whole slice rests on.
//   * the CONE + clamp invariants: every accumulated normal impulse >= 0, every accumulated tangent impulse within
//     the friction cone |jt| <= mu*jn on its accumulated normal.
//   * mu = 0 -> the tangent impulses stay EXACTLY zero (the frictionless control — no spurious tangential force).
//   * a high-mu resting contact REMOVES the relative tangential velocity (friction BITES — the post-solve slip is
//     strictly smaller than a frictionless solve leaves).
//   * two-run determinism (a fixed world yields byte-identical solved bodies + accumulators on two solves).
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/hullfric.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace hullfric = hf::sim::hullfric;
namespace warmhull = hf::sim::warmhull;
namespace fric     = hf::sim::fric;
namespace convex   = hf::sim::convex;
namespace gjk      = hf::sim::gjk;
namespace fpx      = hf::sim::fpx;
using gjk::fx;
using gjk::kOne;
using gjk::FxVec3;
using gjk::FxHull;
using fpx::FxBody;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static fx absfx(fx v) { return v < 0 ? -v : v; }

// A dynamic box body at (px,py,pz) with an initial linear velocity (vx,vy,vz).
static FxBody DynBody(fx px, fx py, fx pz, fx vx, fx vy, fx vz) {
    FxBody b;
    b.pos = {px, py, pz};
    b.vel = {vx, vy, vz};
    b.orient = {0, 0, 0, kOne};
    b.invMass = kOne;                // mass 1
    b.flags = fpx::kFlagDynamic;
    return b;
}
static FxBody StaticBody(fx px, fx py, fx pz) {
    FxBody b;
    b.pos = {px, py, pz};
    b.orient = {0, 0, 0, kOne};
    b.invMass = 0;                   // static
    b.flags = 0;
    return b;
}

// The deterministic curated hull world: a DYNAMIC box resting on a TILTED static box (a non-cardinal EPA normal so
// the basis + the full-inertia tensor are non-trivial) with a sideways velocity so friction has tangential work to
// do, PLUS a second flat dynamic-on-static contact with a sideways velocity (the "couple in contact with non-zero
// relative tangential velocity so friction bites"). All bodies overlap their support by the standard 0.125.
static gjk::HullWorld MakeWorld(fx slideVel) {
    const fx overlap = kOne / 8;
    const FxHull boxH = gjk::MakeBox(kOne, kOne, kOne);
    gjk::HullWorld w;
    // A downward approach velocity (into the support) so the NORMAL block produces a real normal impulse -- the
    // load the friction cone clamps against. Without a normal load a purely-tangential rest contact has jn=0, so the
    // cone is +/-0 and no tangent impulse can grow (friction would have nothing to bite on -- the gravity-driven
    // normal load comes from the HF3 step's integrate; HF2 solves a fixed manifold so we seed the approach here).
    const fx approach = kOne / 8;   // 0.125 downward

    // pair (0,1): a dynamic box on a TILTED static base (~22.5deg/Z), approaching + sliding sideways (+x).
    FxBody tiltedBase = StaticBody(0, 0, 0);
    tiltedBase.orient = {0, 0, (fx)(0.19509032f * 65536.0f), (fx)(0.98078528f * 65536.0f)};  // ~22.5deg/Z
    FxBody topTilt = DynBody(0, (fx)((1.0 + 1.41421356 - 0.125) * 65536.0), 0, slideVel, -approach, 0);

    // pair (2,3): a dynamic box on a FLAT static base, approaching + sliding sideways (+z, the other tangent).
    FxBody flatBase = StaticBody(gjk::FromInt(4), 0, 0);
    FxBody topFlat  = DynBody(gjk::FromInt(4), gjk::FromInt(2) - overlap, 0, 0, -approach, slideVel);

    w.bodies = {tiltedBase, topTilt, flatBase, topFlat};
    w.hulls  = {boxH, boxH, boxH, boxH};
    return w;
}

// Solve a world and measure the max post-solve residual over EACH pair using its actual two bodies. We mirror
// StepHullFrictionSolveOnly's pair loop so we know which bodies each manifold couples (the residual is a function of
// the post-solve velocities at the contact points). This is the warm<cold lever metric.
static fx SolveAndResidual(gjk::HullWorld w, hullfric::HullFrictionCache cache,
                           const hullfric::HullFrictionConfig& cfg) {
    namespace mf = hf::sim::manifold;
    const size_t n = w.bodies.size();
    std::vector<convex::FxMat3> invIW(n);
    for (size_t i = 0; i < n; ++i) {
        const mf::FxHullFaces faces = mf::BuildCanonicalFaces(w.hulls[i]);
        const convex::FxMat3 invIbody = mf::FxHullInertiaBodyFull(w.hulls[i], faces, w.bodies[i].invMass);
        invIW[i] = mf::WorldInvInertiaFull(w.bodies[i], invIbody);
    }
    fx maxRes = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (w.bodies[i].invMass == 0 && w.bodies[j].invMass == 0) continue;
            hullfric::HullFrictionManifold m = hullfric::BuildHullFrictionManifold(
                (uint32_t)i, w.bodies[i], w.hulls[i], (uint32_t)j, w.bodies[j], w.hulls[j]);
            if (m.count == 0) continue;
            hullfric::MatchHullFrictionCache(cache, m);
            hullfric::SolveHullFrictionWarm(w.bodies[i], w.bodies[j], m, invIW[i], invIW[j], cfg);
            const fx r = hullfric::HullContactResidual(w.bodies[i], w.bodies[j], m);
            if (r > maxRes) maxRes = r;
        }
    }
    return maxRes;
}

int main() {
    HF_TEST_MAIN_INIT();

    const fx slide = kOne / 4;   // 0.25 sideways velocity so friction has tangential work
    hullfric::HullFrictionConfig cfg;
    cfg.mu = (fx)((int64_t)6 * kOne / 10);   // 0.6 Coulomb cone
    cfg.restitution = 0;

    // ---- (A) CONVERGENCE: the residual is non-increasing as iters grows (a COLD solve from zero). ----
    {
        fx prev = -1;
        bool monotone = true;
        for (uint32_t it : {1u, 2u, 4u, 8u, 16u}) {
            hullfric::HullFrictionConfig c = cfg; c.iters = it;
            hullfric::HullFrictionCache emptyCache;
            const fx r = SolveAndResidual(MakeWorld(slide), emptyCache, c);
            if (prev >= 0 && r > prev) monotone = false;
            prev = r;
        }
        check(monotone, "SolveHullFrictionWarm residual is non-increasing with iters (cold convergence)");
    }

    // ---- (B) THE WARM-START LEVER: warm-primed residual < cold residual at EQUAL iters. ----
    {
        const uint32_t kIters = 3;   // a LOW budget so the warm-start advantage is visible
        hullfric::HullFrictionConfig c = cfg; c.iters = kIters;

        // Build the WARM cache: run a FULL converged solve (many iters) over the world ONCE; its persisted
        // accumulators are the warm seed for the low-iters solve.
        hullfric::HullFrictionCache warmCache;
        {
            hullfric::HullFrictionConfig cConverge = cfg; cConverge.iters = 64;
            gjk::HullWorld wPrime = MakeWorld(slide);
            hullfric::StepHullFrictionSolveOnly(wPrime, warmCache, cConverge);   // warmCache now holds the converged impulses
        }

        // COLD: an empty cache (all accumulators start at zero) at kIters.
        hullfric::HullFrictionCache coldCache;
        const fx coldRes = SolveAndResidual(MakeWorld(slide), coldCache, c);

        // WARM: the primed cache at the SAME kIters.
        const fx warmRes = SolveAndResidual(MakeWorld(slide), warmCache, c);

        std::printf("hf2-warm test: warmRes=%d coldRes=%d at iters=%u\n", (int)warmRes, (int)coldRes, kIters);
        check(warmRes < coldRes, "warm-primed residual < cold residual at equal iters (the warm-start lever)");
    }

    // ---- (C) the CONE + clamp invariants: accumNormal >= 0 all, |jt| <= mu*jn all. ----
    {
        hullfric::HullFrictionConfig c = cfg; c.iters = 16;
        gjk::HullWorld w = MakeWorld(slide);
        hullfric::HullFrictionCache cache;
        std::vector<hullfric::HullFrictionManifold> solved =
            hullfric::StepHullFrictionSolveOnly(w, cache, c);
        bool coneOk = true;
        bool normalOk = true;
        uint32_t contacts = 0;
        for (const auto& m : solved) {
            for (uint32_t i = 0; i < m.count && i < 4u; ++i) {
                ++contacts;
                const fx jn = m.pts[i].normalImpulse;
                if (jn < 0) normalOk = false;
                const fx cone = fpx::fxmul(c.mu, jn);   // mu * the accumulated normal
                if (absfx(m.pts[i].tangentImpulse1) > cone + 1) coneOk = false;   // +1 LSB rounding slack
                if (absfx(m.pts[i].tangentImpulse2) > cone + 1) coneOk = false;
            }
        }
        check(contacts > 0, "the curated world produces contacts to solve");
        check(normalOk, "every accumulated normal impulse >= 0 (cone: accumNormal>=0 all)");
        check(coneOk, "every accumulated tangent impulse within the friction cone (|jt| <= mu*jn all)");
    }

    // ---- (D) mu = 0 -> the tangent impulses stay EXACTLY zero (the frictionless control). ----
    {
        hullfric::HullFrictionConfig c; c.mu = 0; c.restitution = 0; c.iters = 16;
        gjk::HullWorld w = MakeWorld(slide);
        hullfric::HullFrictionCache cache;
        std::vector<hullfric::HullFrictionManifold> solved =
            hullfric::StepHullFrictionSolveOnly(w, cache, c);
        bool allZero = true;
        for (const auto& m : solved)
            for (uint32_t i = 0; i < m.count && i < 4u; ++i)
                if (m.pts[i].tangentImpulse1 != 0 || m.pts[i].tangentImpulse2 != 0) allZero = false;
        check(allZero, "mu=0 -> tangent impulses stay EXACTLY zero (frictionless control)");
    }

    // ---- (E) friction BITES: a high-mu solve leaves a SMALLER post-solve tangential slip than mu=0. ----
    {
        const uint32_t kIters = 16;
        // The flat sliding pair (indices 2,3) is the clean tangential test. Measure the post-solve relative
        // tangential velocity at the contact for mu=0 vs a high mu.
        auto tangentSlip = [&](fx mu) -> fx {
            namespace mf = hf::sim::manifold;
            gjk::HullWorld w = MakeWorld(slide);
            hullfric::HullFrictionConfig c; c.mu = mu; c.restitution = 0; c.iters = kIters;
            const size_t n = w.bodies.size();
            std::vector<convex::FxMat3> invIW(n);
            for (size_t i = 0; i < n; ++i) {
                const mf::FxHullFaces faces = mf::BuildCanonicalFaces(w.hulls[i]);
                const convex::FxMat3 invIbody = mf::FxHullInertiaBodyFull(w.hulls[i], faces, w.bodies[i].invMass);
                invIW[i] = mf::WorldInvInertiaFull(w.bodies[i], invIbody);
            }
            // solve only the flat pair (2,3) in isolation so the slip is the clean signal.
            hullfric::HullFrictionManifold m = hullfric::BuildHullFrictionManifold(
                2u, w.bodies[2], w.hulls[2], 3u, w.bodies[3], w.hulls[3]);
            hullfric::HullFrictionCache empty;
            hullfric::MatchHullFrictionCache(empty, m);
            hullfric::SolveHullFrictionWarm(w.bodies[2], w.bodies[3], m, invIW[2], invIW[3], c);
            // the post-solve relative tangential speed at the first contact point.
            fx maxSlip = 0;
            for (uint32_t i = 0; i < m.count && i < 4u; ++i) {
                const FxVec3 p = m.pts[i].point;
                const FxVec3 rA = fpx::FxSub(p, w.bodies[2].pos);
                const FxVec3 rB = fpx::FxSub(p, w.bodies[3].pos);
                const FxVec3 vpA = fpx::FxAdd(w.bodies[2].vel, convex::FxCross(w.bodies[2].angVel, rA));
                const FxVec3 vpB = fpx::FxAdd(w.bodies[3].vel, convex::FxCross(w.bodies[3].angVel, rB));
                const FxVec3 dv = fpx::FxSub(vpB, vpA);
                const fx s1 = absfx(convex::FxDot(dv, m.t1));
                const fx s2 = absfx(convex::FxDot(dv, m.t2));
                if (s1 > maxSlip) maxSlip = s1;
                if (s2 > maxSlip) maxSlip = s2;
            }
            return maxSlip;
        };
        const fx slipNoFric = tangentSlip(0);
        const fx slipHiFric = tangentSlip((fx)((int64_t)9 * kOne / 10));   // mu = 0.9
        std::printf("hf2-warm test: slip mu=0 -> %d, mu=0.9 -> %d\n", (int)slipNoFric, (int)slipHiFric);
        check(slipHiFric < slipNoFric, "high-mu friction BITES (removes more tangential slip than frictionless)");
    }

    // ---- (F) two-run determinism: a fixed world yields byte-identical solved bodies + accumulators. ----
    {
        hullfric::HullFrictionConfig c = cfg; c.iters = 8;
        gjk::HullWorld w1 = MakeWorld(slide), w2 = MakeWorld(slide);
        hullfric::HullFrictionCache c1, c2;
        std::vector<hullfric::HullFrictionManifold> s1 = hullfric::StepHullFrictionSolveOnly(w1, c1, c);
        std::vector<hullfric::HullFrictionManifold> s2 = hullfric::StepHullFrictionSolveOnly(w2, c2, c);
        bool bodiesEq = (w1.bodies.size() == w2.bodies.size());
        if (bodiesEq)
            bodiesEq = (std::memcmp(w1.bodies.data(), w2.bodies.data(),
                                    w1.bodies.size() * sizeof(FxBody)) == 0);
        bool manEq = (s1.size() == s2.size());
        if (manEq)
            for (size_t k = 0; k < s1.size(); ++k)
                if (std::memcmp(&s1[k], &s2[k], sizeof(hullfric::HullFrictionManifold)) != 0) manEq = false;
        check(bodiesEq, "two-run determinism: solved bodies byte-identical");
        check(manEq, "two-run determinism: solved manifolds byte-identical");
    }

    if (g_fail == 0) std::printf("hullfric_warm_test: ALL PASS\n");
    else             std::printf("hullfric_warm_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
