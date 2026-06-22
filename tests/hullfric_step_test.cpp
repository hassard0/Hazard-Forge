// Slice HF3 — Hull Friction + Joints: THE FRICTION-LOCKED HULL WORLD STEP (the friction-locked tick of FLAGSHIP #30:
// HULL FRICTION + HULL JOINTS, hf::sim::hullfric). The integer core (engine/sim/hullfric.h, APPENDED after HF2) the
// GPU shaders/hullfric_step.comp.hlsl copies VERBATIM + proves bit-identical. HF3 assembles the FULL per-tick world
// step: the warmhull::StepWarmHullWorld 5-pass tick with the normal-only solve SWAPPED for the HF2 cone solver + the
// per-pair tangent basis built each tick + mu threaded + the friction cache persisted across ticks. #includes
// warmhull/gjk/fric/persist/manifold READ-ONLY (ALL BYTE-FROZEN).
//
// What this test PINS (the contracts the GPU hullfric_step.comp + the GPU==CPU proof build on, the spec proofs):
//   * StepWarmFrictionHullWorldN two-run determinism (a fixed world yields byte-identical final bodies + cache).
//   * THE gripped-on-ramp DELTA (the falsifiable friction proof): a hull released on a tilted static hull GRIPS and
//     RESTS at mu>0, but SLIDES off (past the ramp footprint) at mu=0 — friction is the thing holding it.
//   * the resting contact's tangent impulse stays within the cone (|jt| <= mu*jn) and the accumulated normal >= 0.
//   * the de-pen keeps the hull ABOVE the ramp (no sink-through: the dynamic body stays outside the ramp body).
//   * the cache PERSISTS + warm-starts: it is non-empty after the run (the resting contact warm-started each tick).
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
static fx F(double v) { return (fx)(v * 65536.0); }

// ----- The PINNED gripped-on-ramp scene: a TILTED static ramp (~22.5deg about Z, tan~0.414 < mu) + a dynamic box
// released aligned on its top face. At mu>0 (>= 0.6 here, mu > tan(theta)) static friction holds the box at rest;
// at mu=0 it slides downhill off the ramp. Bodies are 2x2x2 boxes (half-extent 1). The dynamic box center sits on
// the ramp up-axis (-sin th, cos th, 0) at distance (2 - 0.125) so a contact forms with a small overlap.
static constexpr double kThetaDeg = 22.5;
static gjk::HullWorld BuildRampWorld() {
    gjk::HullWorld w;
    const FxHull boxH = gjk::MakeBox(kOne, kOne, kOne);   // 2x2x2 box
    const double sinth = 0.38268343236508984;   // sin(22.5deg)
    const double costh = 0.9238795325112867;     // cos(22.5deg)
    FxBody ramp; ramp.pos = {0, 0, 0};
    ramp.orient = {0, 0, F(0.19509032201612825), F(0.9807852804032304)};   // half-angle quat ~22.5deg/Z
    ramp.invMass = 0; ramp.flags = 0;
    const double sep = 2.0 - 0.125;
    FxBody dyn; dyn.pos = {F(-sinth * sep), F(costh * sep), 0};
    dyn.orient = {0, 0, F(0.19509032201612825), F(0.9807852804032304)};
    dyn.vel = {0, 0, 0}; dyn.angVel = {0, 0, 0};
    dyn.invMass = kOne; dyn.flags = fpx::kFlagDynamic;
    w.bodies = {ramp, dyn};
    w.hulls  = {boxH, boxH};
    return w;
}

static hullfric::HullFrictionStepConfig StepCfg(fx mu) {
    hullfric::HullFrictionStepConfig cfg;
    cfg.mu = mu;
    cfg.solveIters = 12;
    cfg.posIters   = 4;
    return cfg;
}

static constexpr uint32_t kTicks = 240;
static constexpr fx kRestThreshold = kOne / 16;     // ~0.0625 unit/s — well above the resting jitter, below any slide
static constexpr fx kSlideLimit    = gjk::FromInt(4);   // a slid hull translates FAR past this ramp footprint

int main() {
    HF_TEST_MAIN_INIT();

    // (1) two-run determinism — a fixed world yields byte-identical final bodies + cache.
    {
        gjk::HullWorld w1 = BuildRampWorld(), w2 = BuildRampWorld();
        hullfric::HullFrictionCache c1, c2;
        const auto cfg = StepCfg(F(0.6));
        hullfric::StepWarmFrictionHullWorldN(w1, c1, cfg, kTicks);
        hullfric::StepWarmFrictionHullWorldN(w2, c2, cfg, kTicks);
        bool bodiesEq = (w1.bodies.size() == w2.bodies.size());
        if (bodiesEq)
            bodiesEq = (std::memcmp(w1.bodies.data(), w2.bodies.data(),
                                    w1.bodies.size() * sizeof(FxBody)) == 0);
        check(bodiesEq, "two-run determinism: final bodies byte-identical");
        bool cacheEq = (c1.entries.size() == c2.entries.size());
        if (cacheEq && !c1.entries.empty())
            cacheEq = (std::memcmp(c1.entries.data(), c2.entries.data(),
                                   c1.entries.size() * sizeof(hullfric::CachedHullFrictionContact)) == 0);
        check(cacheEq, "two-run determinism: cache byte-identical");
    }

    // (2) THE gripped-on-ramp delta — mu>0 RESTS, mu=0 SLIDES off. The falsifiable friction proof.
    {
        gjk::HullWorld wPos = BuildRampWorld();
        hullfric::HullFrictionCache cachePos;
        hullfric::StepWarmFrictionHullWorldN(wPos, cachePos, StepCfg(F(0.6)), kTicks);
        hullfric::HullGripMeasure gPos = hullfric::MeasureHullGrip(wPos, 1, 0, kRestThreshold, kSlideLimit);

        gjk::HullWorld wZero = BuildRampWorld();
        hullfric::HullFrictionCache cacheZero;
        hullfric::StepWarmFrictionHullWorldN(wZero, cacheZero, StepCfg(0), kTicks);
        hullfric::HullGripMeasure gZero = hullfric::MeasureHullGrip(wZero, 1, 0, kRestThreshold, kSlideLimit);

        std::printf("hf3-step grip: muPos{rested=%d speed=%d onRamp=%d}  muZero{rested=%d speed=%d onRamp=%d}\n",
                    gPos.rested ? 1 : 0, (int)gPos.speed, gPos.onRamp ? 1 : 0,
                    gZero.rested ? 1 : 0, (int)gZero.speed, gZero.onRamp ? 1 : 0);
        check(gPos.rested, "gripped-on-ramp: mu>0 the hull RESTS on the ramp");
        check(!gZero.rested, "gripped-on-ramp: mu=0 the hull SLIDES off (not rested)");
        check(!gZero.onRamp, "gripped-on-ramp: mu=0 the hull leaves the ramp footprint");
    }

    // (3) the resting contact's tangent impulse is within the cone, and the de-pen keeps the hull above the ramp.
    {
        gjk::HullWorld w = BuildRampWorld();
        hullfric::HullFrictionCache cache;
        const auto cfg = StepCfg(F(0.6));
        // step K-1 ticks, then capture the FINAL tick's solved manifold to read the converged accumulators.
        hullfric::StepWarmFrictionHullWorldN(w, cache, cfg, kTicks - 1);
        hullfric::HullFrictionConfig solveCfg;
        solveCfg.mu = cfg.mu; solveCfg.restitution = cfg.restitution; solveCfg.iters = cfg.solveIters;
        // mirror the final tick's solve to read the manifold accumulators (the step integrates first; we re-derive
        // the manifold from the settled positions and inspect the cone on the converged warm-seeded accumulators).
        std::vector<hullfric::HullFrictionManifold> solved =
            hullfric::StepHullFrictionSolveOnly(w, cache, solveCfg);
        bool coneOk = true;
        bool sawContact = false;
        for (const auto& m : solved) {
            if (m.count == 0) continue;
            sawContact = true;
            for (uint32_t i = 0; i < m.count; ++i) {
                const fx jn = m.pts[i].normalImpulse;
                const fx cone = fpx::fxmul(cfg.mu, jn < 0 ? 0 : jn);
                if (jn < 0) coneOk = false;
                if (absfx(m.pts[i].tangentImpulse1) > cone + 2) coneOk = false;   // +2 LSB fixed-point slack
                if (absfx(m.pts[i].tangentImpulse2) > cone + 2) coneOk = false;
            }
        }
        check(sawContact, "resting contact: a manifold exists at the settled rest pose");
        check(coneOk, "resting contact: tangent impulse within cone |jt|<=mu*jn, accum normal >= 0");

        // de-pen keeps the hull ABOVE the ramp: the dynamic body center is not buried inside the ramp (its distance
        // to the ramp center exceeds a sink floor — the boxes are not overlapping by more than the slop band).
        const FxBody& ramp = w.bodies[0];
        const FxBody& dyn  = w.bodies[1];
        const fx dy = dyn.pos.y - ramp.pos.y;
        check(dy > kOne, "de-pen: the dynamic hull stays above the ramp (no sink-through)");
    }

    // (4) the cache PERSISTS + warm-starts — non-empty after a resting run (the resting contact warm-started).
    {
        gjk::HullWorld w = BuildRampWorld();
        hullfric::HullFrictionCache cache;
        hullfric::StepWarmFrictionHullWorldN(w, cache, StepCfg(F(0.6)), kTicks);
        check(!cache.entries.empty(), "warm-start: the cache is non-empty after the resting run (warm-started)");
    }

    if (g_fail == 0) std::printf("hullfric_step_test: ALL PASS\n");
    else             std::printf("hullfric_step_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
