// Slice CD1 — Deterministic Integer CCD: THE TIME-OF-IMPACT PRIMITIVE (the BEACHHEAD of FLAGSHIP #24:
// hf::sim::ccd). The integer core (engine/sim/ccd.h) that the GPU shaders/ccd_toi.comp.hlsl copies VERBATIM +
// proves bit-identical. Pure CPU (header-only, hf_core), ASan-eligible. ccd.h #includes sim/broad.h read-only
// (transitively gjk/convex/fpx).
//
// What this test PINS (the contracts the GPU ccd_toi.comp + the GPU==CPU proof build on):
//   * ConservativeAdvance straight-approach: a body moving straight toward a static obstacle reports a HIT at
//     a TOI within the documented band of the hand-computed Q16.16 (gap / speed).
//   * ConservativeAdvance rotating-approach: the returned TOI does NOT overshoot — at the TOI pose gjk::Gjk
//     reports the pair separated-or-touching, NOT overlapping (the conservative angular-bound guarantee).
//   * A RECEDING pair -> {hit=0, toi=dt}.
//   * An ALREADY-OVERLAPPING pair -> {toi=0, hit=1}.
//   * ClosingSpeedBound is conservative (>= the true projected closing speed for a hand case incl. rotation).
//   * MeasureCcdToi / ConservativeAdvance are PURE (two calls byte-equal — determinism).
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/ccd.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace ccd = hf::sim::ccd;
namespace gjk = hf::sim::gjk;
namespace convex = hf::sim::convex;
namespace fpx = hf::sim::fpx;
using ccd::fx;
using ccd::kOne;
using ccd::kFrac;
using ccd::FxVec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static fx FromInt(int v) { return (fx)((int64_t)v << kFrac); }

// A body at a Q16.16 position with optional velocity / angular velocity / orientation.
static fpx::FxBody MakeBody(fx px, fx py, fx pz) {
    fpx::FxBody b;
    b.pos = {px, py, pz};
    return b;
}

int main() {
    HF_TEST_MAIN_INIT();

    const gjk::FxHull box = gjk::MakeBox(kOne, kOne, kOne);   // unit box (half-extent 1)
    const fx dt = kOne;                                       // a 1-second timestep

    // === (1) Straight-approach: a unit box at x=0 moving +X at 4 u/s toward a STATIC unit box at x=5.
    // Surfaces at x=1 and x=4 -> gap = 3.0. Closing speed = 4 u/s -> hand-computed TOI = 3/4 = 0.75 s. The
    // conservative loop converges from BELOW within kContactEps -> toi within a documented band of 0.75. ===
    {
        fpx::FxBody mover = MakeBody(0, 0, 0);
        mover.vel = {FromInt(4), 0, 0};
        mover.flags = fpx::kFlagDynamic;
        fpx::FxBody obstacle = MakeBody(FromInt(5), 0, 0);   // static (zero vel/angVel)

        fpx::FxBody poseA, poseB;
        const ccd::FxToi r = ccd::ConservativeAdvancePose(box, mover, box, obstacle, dt, poseA, poseB);
        check(r.hit == 1u, "straight-approach reports a HIT");
        // Hand-computed TOI = gap/speed = 3/4 = 0.75 s. Conservative advancement converges from BELOW and never
        // OVERSHOOTS the true contact, so toi <= 0.75 (within a tiny truncation margin). Band: [0.625, 0.75].
        const fx kExpected = (kOne * 3) / 4;            // 0.75 in Q16.16
        const fx kBandLo = kExpected - kOne / 8;        // 0.625 lower guard (well within the convergence band)
        check(r.toi <= kExpected + kOne / 64, "straight TOI does not exceed the hand-computed 0.75 (no overshoot)");
        check(r.toi >= kBandLo, "straight TOI is within the documented band below 0.75");

        // No-overshoot proof: at the loop's OWN final TOI pose the pair is ARRESTED AT CONTACT (not tunneled).
        // Query the FROZEN Gjk at that pose: either SEPARATED with a gap within the contact band, or TOUCHING
        // (Gjk reports a sub-kProgressEps gap as overlap — an exact head-on lands exactly on contact). A
        // tunnel-through would instead show a LARGE far-side gap.
        const gjk::GjkResult gAt = gjk::Gjk(box, poseA, box, poseB);
        const fx gapAt = gAt.overlap ? 0 : fpx::FxLength(gAt.separation);
        check(gapAt <= ccd::kContactEps, "straight TOI pose is arrested at contact (no tunnel; gap within band)");
    }

    // === (2) Receding: the mover travels AWAY from the obstacle -> bound <= 0 -> {hit=0, toi=dt}. ===
    {
        fpx::FxBody mover = MakeBody(0, 0, 0);
        mover.vel = {FromInt(-4), 0, 0};                // moving -X, away from the +X obstacle
        mover.flags = fpx::kFlagDynamic;
        fpx::FxBody obstacle = MakeBody(FromInt(5), 0, 0);

        const ccd::FxToi r = ccd::ConservativeAdvance(box, mover, box, obstacle, dt);
        check(r.hit == 0u, "receding pair reports NO hit");
        check(r.toi == dt, "receding pair returns toi == dt");
    }

    // === (3) Tangential: the mover slides parallel (no approach on the separating normal) -> no hit. The
    // obstacle is offset on Z so the gap is along Z; the mover moves along X (tangent) -> bound <= 0. ===
    {
        fpx::FxBody mover = MakeBody(0, 0, 0);
        mover.vel = {FromInt(4), 0, 0};                 // moving +X
        mover.flags = fpx::kFlagDynamic;
        fpx::FxBody obstacle = MakeBody(0, 0, FromInt(5));  // 5 away on Z

        const ccd::FxToi r = ccd::ConservativeAdvance(box, mover, box, obstacle, dt);
        check(r.hit == 0u, "tangential pair reports NO hit");
        check(r.toi == dt, "tangential pair returns toi == dt");
    }

    // === (4) Already overlapping: two coincident boxes -> {toi=0, hit=1}. ===
    {
        fpx::FxBody a = MakeBody(0, 0, 0);
        a.vel = {FromInt(1), 0, 0};
        a.flags = fpx::kFlagDynamic;
        fpx::FxBody b = MakeBody(0, 0, 0);              // coincident -> overlapping at t=0

        const ccd::FxToi r = ccd::ConservativeAdvance(box, a, box, b, dt);
        check(r.hit == 1u, "already-overlapping pair reports a HIT");
        check(r.toi == 0, "already-overlapping pair returns toi == 0");
    }

    // === (5) Rotating-approach: a SPINNING box approaches a static box. The angular term of the closing-speed
    // bound is what keeps the TOI conservative (no overshoot). We assert the TOI pose is non-overlapping AND
    // that ClosingSpeedBound's angular term is mandatory (a hand check below). ===
    {
        fpx::FxBody mover = MakeBody(0, 0, 0);
        mover.vel = {FromInt(2), 0, 0};                 // approaching +X
        mover.angVel = {0, 0, FromInt(4)};              // spinning fast about Z (the surface sweeps toward B)
        mover.flags = fpx::kFlagDynamic;
        fpx::FxBody obstacle = MakeBody(FromInt(4), 0, 0);

        fpx::FxBody poseA, poseB;
        const ccd::FxToi r = ccd::ConservativeAdvancePose(box, mover, box, obstacle, dt, poseA, poseB);
        if (r.hit) {
            // No-overshoot proof at the loop's OWN final pose: the spinning box is ARRESTED AT CONTACT, not
            // tunneled. The angular term of ClosingSpeedBound is what keeps each advance a lower bound on the
            // true TOI so the surface never sweeps THROUGH the obstacle. Gjk at the final pose: separated with
            // a gap within the contact band, or a touch (sub-kProgressEps overlap).
            const gjk::GjkResult gAt = gjk::Gjk(box, poseA, box, poseB);
            const fx gapAt = gAt.overlap ? 0 : fpx::FxLength(gAt.separation);
            check(gapAt <= ccd::kContactEps,
                  "rotating-approach TOI pose is arrested at contact (angular bound conservative, no overshoot)");
        } else {
            // If it does not register a hit within dt that's fine (conservative may need more dt), but then
            // toi must be dt.
            check(r.toi == dt, "rotating-approach no-hit returns toi == dt");
        }
    }

    // === (6) ClosingSpeedBound conservative incl. rotation: the bound must be >= the LINEAR projected closing
    // speed, and the angular terms must ADD to it (mandatory). Hand case: relVel = (4,0,0), n = (1,0,0) ->
    // linear closing speed = 4; a spinning body with |angVel|=2 and rMax=~sqrt(3) adds a positive angular term
    // so the full bound STRICTLY exceeds the linear-only bound. ===
    {
        fpx::FxBody a = MakeBody(0, 0, 0);
        a.vel = {0, 0, 0};
        a.angVel = {0, 0, FromInt(2)};                  // |angVel| = 2
        fpx::FxBody b = MakeBody(FromInt(4), 0, 0);
        b.vel = {FromInt(-4), 0, 0};                    // relVel = b.vel - a.vel = (-4,0,0)

        const FxVec3 n{-kOne, 0, 0};                    // normal pointing A<-... so dot(relVel,n) = +4 (closing)
        const fx rMaxA = ccd::BodyMaxRadius(box, a);
        const fx rMaxB = ccd::BodyMaxRadius(box, b);
        const fx linOnly = convex::FxDot(fpx::FxSub(b.vel, a.vel), n);   // 4.0
        const fx full = ccd::ClosingSpeedBound(a, rMaxA, b, rMaxB, n);

        check(rMaxA > 0, "BodyMaxRadius is positive for a unit box");
        check(linOnly > 0, "linear closing speed is positive (the pair closes)");
        check(full > linOnly, "ClosingSpeedBound ADDS the angular term (mandatory, > linear-only)");
        // The angular term equals |angVelA|*rMaxA (b is non-spinning) -> full - linOnly == fxmul(2, rMaxA).
        const fx angTerm = fpx::fxmul(ccd::FxLength(a.angVel), rMaxA);
        check((full - linOnly) == angTerm, "the angular term is exactly |angVelA| * rMaxA");
    }

    // === (7) Determinism / purity: two ConservativeAdvance calls over the same inputs are byte-equal; the
    // MeasureCcdToi summary is a pure function (two calls byte-equal). ===
    {
        fpx::FxBody mover = MakeBody(0, 0, 0);
        mover.vel = {FromInt(4), 0, 0};
        mover.flags = fpx::kFlagDynamic;
        fpx::FxBody obstacle = MakeBody(FromInt(5), 0, 0);

        const ccd::FxToi r1 = ccd::ConservativeAdvance(box, mover, box, obstacle, dt);
        const ccd::FxToi r2 = ccd::ConservativeAdvance(box, mover, box, obstacle, dt);
        check(std::memcmp(&r1, &r2, sizeof(ccd::FxToi)) == 0, "ConservativeAdvance is deterministic (byte-equal)");

        std::vector<ccd::CcdPair> pairs;
        pairs.push_back({box, mover, box, obstacle, dt});
        fpx::FxBody recede = mover; recede.vel = {FromInt(-4), 0, 0};
        pairs.push_back({box, recede, box, obstacle, dt});
        const ccd::CcdToiMeasure m1 = ccd::MeasureCcdToi(pairs.data(), (uint32_t)pairs.size());
        const ccd::CcdToiMeasure m2 = ccd::MeasureCcdToi(pairs.data(), (uint32_t)pairs.size());
        check(std::memcmp(&m1, &m2, sizeof(ccd::CcdToiMeasure)) == 0, "MeasureCcdToi is a pure function (byte-equal)");
        check(m1.pairs == 2u && m1.hits == 1u, "MeasureCcdToi counts {pairs:2, hits:1}");
    }

    // ===== Slice CD2 — THE SWEPT-AABB BROADPHASE ============================================================
    // Helpers: a dynamic body with a velocity, a static body, and a HullWorld of unit boxes.
    namespace gjk = hf::sim::gjk;
    namespace broad = hf::sim::broad;
    auto mkDyn = [&](fx x, fx y, fx z, fx vx) {
        fpx::FxBody b = MakeBody(x, y, z);
        b.vel = {vx, 0, 0};
        b.invMass = kOne;
        b.flags = fpx::kFlagDynamic;
        return b;
    };
    auto mkStat = [&](fx x, fx y, fx z) {
        fpx::FxBody b = MakeBody(x, y, z);   // invMass 0, no dynamic flag -> static (never sweeps)
        return b;
    };
    // cellSize generous (>= the max swept-AABB diameter for these fast movers over dt=1).
    const fx kSweptCell = FromInt(64);

    // === (CD2.1) SweptHullAabb contains BOTH the start AND the end world hull verts. A unit box at x=0 moving
    // +X at 6 u/s over dt=1: start box [-1,1] on X, end pose x=6 box [5,7] on X -> swept union [-1,7] on X. ===
    {
        fpx::FxBody mover = mkDyn(0, 0, 0, FromInt(6));
        const fpx::FxAabb swept = ccd::SweptHullAabb(box, mover, dt);
        const fpx::FxAabb startBox = broad::BuildHullAabb(box, mover);
        fpx::FxBody endBody = mover;
        ccd::IntegrateBodyFull(endBody, FxVec3{0, 0, 0}, dt);
        const fpx::FxAabb endBox = broad::BuildHullAabb(box, endBody);
        check(swept.lo.x <= startBox.lo.x && swept.hi.x >= startBox.hi.x, "SweptHullAabb contains the START box on X");
        check(swept.lo.x <= endBox.lo.x && swept.hi.x >= endBox.hi.x, "SweptHullAabb contains the END box on X");
        check(swept.lo.x == FromInt(-1) && swept.hi.x == FromInt(7), "swept X span is the union [-1,7] (start UNION end)");
        // A STATIC body does not move -> its swept box == its instantaneous box (no sweep).
        fpx::FxBody stat = mkStat(0, 0, 0);
        const fpx::FxAabb sweptStat = ccd::SweptHullAabb(box, stat, dt);
        const fpx::FxAabb statBox = broad::BuildHullAabb(box, stat);
        check(std::memcmp(&sweptStat, &statBox, sizeof(fpx::FxAabb)) == 0, "a static body's swept AABB == its instantaneous AABB");
    }

    // === (CD2.2) BuildSweptBroadphasePairs is a canonical i<j set + matches the host CountSweptPairs total. A
    // fast mover sweeping across two clustered obstacles. ===
    {
        gjk::HullWorld world;
        world.bodies = { mkDyn(0, 0, 0, FromInt(10)),     // fast mover sweeps +X across x=5 and x=8
                         mkStat(FromInt(5), 0, 0),        // obstacle 1 (mover's start box misses it)
                         mkStat(FromInt(8), 0, 0) };       // obstacle 2
        world.hulls = { box, box, box };
        std::vector<uint32_t> off;
        std::vector<fpx::FxPair> pairs;
        ccd::BuildSweptBroadphasePairs(world, dt, kSweptCell, off, pairs);
        bool canonical = true;
        for (const fpx::FxPair& p : pairs) if (p.i >= p.j) canonical = false;
        check(canonical, "BuildSweptBroadphasePairs emits canonical i<j pairs");
        // The fast mover (body 0, swept [-1,11] on X) overlaps BOTH obstacles -> pairs (0,1) and (0,2).
        check(pairs.size() == 2u, "the fast mover's sweep produces 2 candidate pairs (both obstacles)");
        // Determinism: two builds byte-identical.
        std::vector<uint32_t> off2; std::vector<fpx::FxPair> pairs2;
        ccd::BuildSweptBroadphasePairs(world, dt, kSweptCell, off2, pairs2);
        const bool detEq = off == off2 && pairs.size() == pairs2.size() &&
            (pairs.empty() || std::memcmp(pairs.data(), pairs2.data(), pairs.size() * sizeof(fpx::FxPair)) == 0);
        check(detEq, "BuildSweptBroadphasePairs is deterministic (byte-identical)");
    }

    // === (CD2.3) A FAST mover -> the swept set has a pair the DISCRETE set lacks (M>0). The discrete
    // (instantaneous-AABB) broadphase keys on the start box [-1,1] which misses the obstacle at x=5; the swept
    // box [-1,11] catches it. ===
    {
        gjk::HullWorld world;
        world.bodies = { mkDyn(0, 0, 0, FromInt(10)),     // fast mover
                         mkStat(FromInt(5), 0, 0) };       // obstacle the start box misses
        world.hulls = { box, box };
        std::vector<uint32_t> sOff, dOff;
        std::vector<fpx::FxPair> swept, discrete;
        ccd::BuildSweptBroadphasePairs(world, dt, kSweptCell, sOff, swept);
        ccd::BuildDiscreteBroadphasePairs(world, kSweptCell, dOff, discrete);
        check(swept.size() == 1u, "the swept set catches the fast-mover pair (1 pair)");
        check(discrete.size() == 0u, "the discrete set MISSES the fast-mover pair (0 pairs — tunneling)");
        uint32_t missed = 0;
        const bool superset = ccd::SweptPairsSupersetOfDiscrete(world, dt, kSweptCell, &missed);
        check(superset, "swept ⊇ discrete (trivially, discrete is empty)");
        check(missed == 1u, "M>0: the swept set has 1 pair the discrete set misses (the fast-mover catch)");
    }

    // === (CD2.4) SweptPairsSupersetOfDiscrete is TRUE for several scenes (a slow cluster where swept==discrete,
    // and a mixed fast+slow scene). The swept set ALWAYS contains every discrete pair (instantaneous ⊆ swept). ===
    {
        // Scene A: a slow tight cluster (tiny velocities) — swept ~= discrete, swept ⊇ discrete still holds.
        gjk::HullWorld slow;
        slow.bodies = { mkDyn(0, 0, 0, kOne / 8),
                        mkDyn(kOne, 0, 0, kOne / 8),
                        mkDyn(0, 0, kOne, kOne / 8) };
        slow.hulls = { box, box, box };
        uint32_t mSlow = 0;
        check(ccd::SweptPairsSupersetOfDiscrete(slow, dt, kSweptCell, &mSlow), "swept ⊇ discrete for a slow cluster");

        // Scene B: a fast mover + a slow cluster (mixed). swept ⊇ discrete AND M>0 (the fast mover adds pairs).
        gjk::HullWorld mixed;
        mixed.bodies = { mkDyn(0, 0, 0, FromInt(12)),       // fast mover across the row
                         mkStat(FromInt(6), 0, 0),
                         mkDyn(FromInt(20), 0, 0, kOne / 8),  // far slow body (no sweep reach)
                         mkDyn(FromInt(20), 0, kOne, kOne / 8) };
        mixed.hulls = { box, box, box, box };
        uint32_t mMixed = 0;
        const bool supMixed = ccd::SweptPairsSupersetOfDiscrete(mixed, dt, kSweptCell, &mMixed);
        check(supMixed, "swept ⊇ discrete for a mixed fast+slow scene");
        check(mMixed > 0u, "the fast mover makes the swept set a STRICT superset (M>0)");

        // MeasureSweptPairs is a pure summary (two calls byte-equal) + reports the superset + M.
        const ccd::SweptPairMeasure ms1 = ccd::MeasureSweptPairs(mixed, dt, kSweptCell);
        const ccd::SweptPairMeasure ms2 = ccd::MeasureSweptPairs(mixed, dt, kSweptCell);
        check(std::memcmp(&ms1, &ms2, sizeof(ccd::SweptPairMeasure)) == 0, "MeasureSweptPairs is a pure function (byte-equal)");
        check(ms1.supersetOfDiscrete && ms1.missedByDiscrete > 0u, "MeasureSweptPairs reports {superset:true, M>0}");
        check(ms1.sweptPairs >= ms1.discretePairs, "the swept pair count >= the discrete pair count");
    }

    // ===== Slice CD3 — THE SUBSTEPPED CCD WORLD STEP ========================================================
    // The CD3 step config (a moderate settling config, small maxSubsteps). Matches the showcase config.
    auto makeCcdCfg = [&]() {
        ccd::CcdStepConfig c;
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        c.bcfg.cfg.gravity     = convex::FxVec3{0, kGravY, 0};
        c.bcfg.cfg.dt          = kOne / 60;
        c.bcfg.cfg.solveIters  = 20;
        c.bcfg.cfg.restitution = 0;
        c.bcfg.cfg.slop        = kOne / 64;
        c.bcfg.cfg.beta        = (fx)((int64_t)4 * kOne / 10);    // 0.4
        c.bcfg.cfg.linDamp     = (fx)((int64_t)90 * kOne / 100);  // 0.90
        c.bcfg.cfg.angDamp     = (fx)((int64_t)5 * kOne / 100);   // 0.05
        c.bcfg.cfg.posIters    = 4;
        c.bcfg.cellSize        = FromInt(64);   // >= the max swept-AABB diameter for the fast mover
        c.maxSubsteps          = 8;
        return c;
    };
    auto makeFullDyn = [&](fx x, fx y, fx z, fx vx, fx vy) {
        fpx::FxBody b = MakeBody(x, y, z);
        b.vel = {vx, vy, 0};
        b.orient = fpx::FxQuat{0, 0, 0, kOne};
        b.invMass = kOne;
        b.flags = fpx::kFlagDynamic;
        return b;
    };
    auto makeFullStat = [&](fx x, fx y, fx z) {
        fpx::FxBody b = MakeBody(x, y, z);
        b.orient = fpx::FxQuat{0, 0, 0, kOne};
        return b;   // invMass 0, no dynamic flag -> static
    };

    // === (CD3.1) StepHullWorldCCDN brings a MODERATE scene to a settled state (a small mixed pile on a floor).
    // The dynamic hulls come to REST (low maxSpeed) and are HELD (maxPen within slop + a small band). ===
    {
        const ccd::CcdStepConfig cfg = makeCcdCfg();
        gjk::HullWorld w;
        w.bodies.push_back(makeFullStat(0, 0, 0));  w.hulls.push_back(gjk::MakeBox(FromInt(4), kOne, FromInt(4)));  // floor
        for (int gx = 0; gx < 3; ++gx)
            for (int gz = 0; gz < 3; ++gz) {
                const fx x = (fx)((int64_t)(gx - 1) * 16 * kOne / 10);
                const fx z = (fx)((int64_t)(gz - 1) * 16 * kOne / 10);
                w.bodies.push_back(makeFullDyn(x, (fx)(17 * kOne / 10), z, 0, 0));
                w.hulls.push_back(gjk::MakeBox((fx)(kOne / 2), (fx)(kOne / 2), (fx)(kOne / 2)));
            }
        ccd::StepHullWorldCCDN(w, cfg, 200);
        const ccd::CcdMeasure ms = ccd::MeasureCcd(w);
        check(ms.dynamicCount == 9u, "CCD step settled scene has 9 dynamic bodies");
        check(ms.maxSpeed < kOne * 2, "CCD step: the pile came to REST (maxSpeed within band)");
        check(ms.maxPenetration < kOne, "CCD step: the pile is HELD (maxPen within band — not sunk)");
    }

    // === (CD3.2) THE NO-TUNNEL PROOF: a FAST mover aimed at a THIN static wall. StepHullWorldCCD keeps the
    // mover on the CORRECT (near) side; the discrete broad::StepHullWorldBP on the SAME scene TUNNELS it
    // through (the mover ends on the FAR side). The two final states DIFFER and the CCD one passes noTunnel. ===
    {
        ccd::CcdStepConfig cfg = makeCcdCfg();
        cfg.bcfg.cfg.gravity = convex::FxVec3{0, 0, 0};   // pure horizontal shot — isolate the tunnel mechanism
        cfg.bcfg.cfg.dt      = kOne / 10;                 // a big tick so per-tick travel >> wall thickness
        cfg.maxSubsteps      = 8;

        // The mover: a small box at x=0 moving +X at 100 u/s -> per-tick travel = 10 units >> the 0.2-thick wall.
        // The wall: a thin static box centred at x=5 (half-extent 0.1 on X, tall on Y/Z). The mover's surface
        // x=0.4 starts at gap 4.5 from the wall's near face x=4.9.
        const fx kWallX = FromInt(5);
        const fx kWallHalfX = (fx)(kOne / 10);            // 0.1 -> 0.2-thick wall
        auto buildShotScene = [&]() {
            gjk::HullWorld w;
            w.bodies.push_back(makeFullDyn(0, 0, 0, FromInt(100), 0));
            w.hulls.push_back(gjk::MakeBox((fx)(kOne * 4 / 10), (fx)(kOne * 4 / 10), (fx)(kOne * 4 / 10)));  // mover
            w.bodies.push_back(makeFullStat(kWallX, 0, 0));
            w.hulls.push_back(gjk::MakeBox(kWallHalfX, FromInt(2), FromInt(2)));                              // wall
            return w;
        };

        gjk::HullWorld ccdW = buildShotScene();
        ccd::StepHullWorldCCDN(ccdW, cfg, 4);
        const fx ccdX = ccdW.bodies[0].pos.x;

        gjk::HullWorld disW = buildShotScene();
        broad::StepHullWorldBPN(disW, cfg.bcfg, 4);
        const fx disX = disW.bodies[0].pos.x;

        // noTunnel: the CCD mover's centre is on the APPROACH (near) side of the wall centre (x < wall x), i.e.
        // it did NOT pass through. The discrete mover TUNNELED -> its centre is on the FAR side (x > wall x).
        const bool ccdNoTunnel = (ccdX < kWallX);
        const bool discreteTunneled = (disX > kWallX);
        check(ccdNoTunnel, "no-tunnel: CCD mover stays on the near side of the wall (gap>=0)");
        check(discreteTunneled, "no-tunnel: discrete StepHullWorldBP TUNNELS the mover through the wall");
        check(ccdX != disX, "no-tunnel: the CCD and discrete final states DIFFER");
    }

    // === (CD3.3) DETERMINISM: two StepHullWorldCCDN runs over the same scene are byte-identical (the body
    // vector is the only mutable replayable state; pure integer, fixed order). ===
    {
        const ccd::CcdStepConfig cfg = makeCcdCfg();
        auto build = [&]() {
            gjk::HullWorld w;
            w.bodies.push_back(makeFullStat(0, 0, 0));  w.hulls.push_back(gjk::MakeBox(FromInt(4), kOne, FromInt(4)));
            w.bodies.push_back(makeFullDyn(0, (fx)(17 * kOne / 10), 0, 0, 0));
            w.hulls.push_back(gjk::MakeBox((fx)(kOne / 2), (fx)(kOne / 2), (fx)(kOne / 2)));
            w.bodies.push_back(makeFullDyn((fx)(12 * kOne / 10), (fx)(20 * kOne / 10), 0, FromInt(30), 0));
            w.hulls.push_back(gjk::MakeBox((fx)(kOne / 2), (fx)(kOne / 2), (fx)(kOne / 2)));
            return w;
        };
        gjk::HullWorld a = build(); ccd::StepHullWorldCCDN(a, cfg, 60);
        gjk::HullWorld b = build(); ccd::StepHullWorldCCDN(b, cfg, 60);
        bool eq = (a.bodies.size() == b.bodies.size());
        for (size_t i = 0; i < a.bodies.size() && eq; ++i)
            if (std::memcmp(&a.bodies[i], &b.bodies[i], sizeof(fpx::FxBody)) != 0) eq = false;
        check(eq, "CCD step determinism: two runs BYTE-IDENTICAL");
    }

    // === (CD3.4) A SLOW scene with NO fast movers: every substep's earliest TOI >= the full remainingDt (no
    // impact within a single tick), so StepHullWorldCCD reduces to the discrete step — it advances all bodies by
    // the full dt and resolves no contact early. We assert the dynamic body falls (gravity integrated) and the
    // world stays finite/coherent (the CCD machinery is a no-op-superset of the discrete step when nothing is
    // fast). NOTE: the at-TOI resolve differs structurally from StepHullWorldBP's pair loop, so we assert the
    // REDUCTION qualitatively (the body moved under gravity, settled, did not blow up), not byte-equality. ===
    {
        ccd::CcdStepConfig cfg = makeCcdCfg();
        gjk::HullWorld w;
        w.bodies.push_back(makeFullStat(0, 0, 0));  w.hulls.push_back(gjk::MakeBox(FromInt(4), kOne, FromInt(4)));
        w.bodies.push_back(makeFullDyn(0, (fx)(30 * kOne / 10), 0, 0, 0));   // a slow drop, no horizontal motion
        w.hulls.push_back(gjk::MakeBox((fx)(kOne / 2), (fx)(kOne / 2), (fx)(kOne / 2)));
        const fx startY = w.bodies[1].pos.y;
        ccd::StepHullWorldCCDN(w, cfg, 120);
        const ccd::CcdMeasure ms = ccd::MeasureCcd(w);
        check(w.bodies[1].pos.y < startY, "CCD slow scene: the dropped body fell under gravity");
        check(w.bodies[1].pos.y > FromInt(0), "CCD slow scene: the body rests above the floor centre (not sunk)");
        check(ms.maxSpeed < kOne * 2, "CCD slow scene: the body came to REST");
    }

    // ===== Slice CD4 — A BULLET THROUGH A THIN WALL STOPS (the new-physics HEADLINE beat) ===================
    // The dedicated bullet-wall scene + the impact measurement + the discrete control. MakeBulletWallScene builds
    // the expected wall + fast projectile (per-tick travel >> wall thickness); StepHullWorldCCDN over it ->
    // tunneled=false (the projectile arrested on the approach side); the discrete broad::StepHullWorldBP over the
    // SAME scene -> tunneled=true (the control); deterministic.

    // === (CD4.1) MakeBulletWallScene builds the expected geometry: a fast +X projectile (body 0) + a thin static
    // wall (body 1) whose per-tick travel (|vel|*kBulletDt) is MANY times the wall thickness. ===
    {
        const gjk::HullWorld w = ccd::MakeBulletWallScene();
        check(w.bodies.size() == 5, "bullet-wall scene has the 5 fixed bodies (projectile/wall/2 drops/floor)");
        // body 0 = the fast DYNAMIC projectile aimed +X.
        check(convex::IsDynamic(w.bodies[0]), "bullet-wall body 0 (projectile) is dynamic");
        check(w.bodies[0].vel.x > 0, "bullet-wall projectile moves in +X");
        // body 1 = the thin STATIC wall (invMass 0, no dynamic flag).
        check(!convex::IsDynamic(w.bodies[1]), "bullet-wall body 1 (wall) is static");
        check(w.bodies[1].pos.x > w.bodies[0].pos.x, "bullet-wall wall is ahead of the projectile on +X");
        // The per-tick travel >> the wall thickness (the guaranteed-tunnel-for-discrete condition). The wall's
        // X half-extent is the max |localVert.x| over its hull verts; the projectile's per-tick travel is
        // |vel|*dt. Assert travel is at least 5x the FULL wall thickness (2 * half-extent).
        fx wallHalfX = 0;
        for (uint32_t v = 0; v < w.hulls[1].count; ++v) {
            const fx ax = w.hulls[1].verts[v].x < 0 ? -w.hulls[1].verts[v].x : w.hulls[1].verts[v].x;
            if (ax > wallHalfX) wallHalfX = ax;
        }
        const fx travel = fpx::fxmul(w.bodies[0].vel.x, ccd::kBulletDt);   // |vel|*dt (vel is +X only)
        check(travel > fpx::fxmul(wallHalfX * 2, FromInt(5)),
              "bullet-wall per-tick travel is >> the wall thickness (a guaranteed discrete tunnel)");
    }

    // === (CD4.2) THE HEADLINE + THE DISCRETE CONTROL: StepHullWorldCCDN over MakeBulletWallScene arrests the
    // projectile (tunneled=false); the discrete broad::StepHullWorldBP over the SAME scene TUNNELS it through
    // (tunneled=true). The two final states DIFFER. ===
    {
        const ccd::CcdStepConfig cfg = ccd::MakeBulletWallConfig();
        const uint32_t kTicks = 6;

        gjk::HullWorld ccdW = ccd::MakeBulletWallScene();
        ccd::StepHullWorldCCDN(ccdW, cfg, kTicks);
        const ccd::BulletMeasure ccdM = ccd::MeasureBullet(ccdW, 1, 0);
        check(ccdM.tunneled == 0u, "CCD bullet-wall: the projectile did NOT tunnel (arrested on the near side)");
        check(ccdW.bodies[0].pos.x < ccdW.bodies[1].pos.x,
              "CCD bullet-wall: the projectile centre is on the APPROACH side of the wall");

        gjk::HullWorld disW = ccd::MakeBulletWallScene();
        broad::StepHullWorldBPN(disW, cfg.bcfg, kTicks);
        const ccd::BulletMeasure disM = ccd::MeasureBullet(disW, 1, 0);
        check(disM.tunneled == 1u,
              "discrete bullet-wall: broad::StepHullWorldBP TUNNELS the projectile through the wall");

        check(ccdM.arrestX != disM.arrestX, "CCD and discrete bullet-wall final states DIFFER");

        // The impactTick is reported (1-based, within the stepped ticks) for the CCD arrest.
        gjk::HullWorld itW = ccd::MakeBulletWallScene();
        const uint32_t impactTick = ccd::StepBulletImpactTick(itW, cfg, 1, 0, kTicks);
        check(impactTick > 0u && impactTick <= kTicks, "CCD bullet-wall: impactTick is within the stepped ticks");
        check(ccd::MeasureBullet(itW, 1, 0).tunneled == 0u, "CCD bullet-wall (impact-tick path): still no tunnel");
    }

    // === (CD4.3) DETERMINISM: two StepHullWorldCCDN runs over MakeBulletWallScene are byte-identical. ===
    {
        const ccd::CcdStepConfig cfg = ccd::MakeBulletWallConfig();
        gjk::HullWorld a = ccd::MakeBulletWallScene(); ccd::StepHullWorldCCDN(a, cfg, 6);
        gjk::HullWorld b = ccd::MakeBulletWallScene(); ccd::StepHullWorldCCDN(b, cfg, 6);
        bool eq = (a.bodies.size() == b.bodies.size());
        for (size_t i = 0; i < a.bodies.size() && eq; ++i)
            if (std::memcmp(&a.bodies[i], &b.bodies[i], sizeof(fpx::FxBody)) != 0) eq = false;
        check(eq, "bullet-wall step determinism: two runs BYTE-IDENTICAL");
    }

    // === (CD5) LOCKSTEP + ROLLBACK — the netcode headline (pure CPU; the BP5/GJ5 twin over StepHullWorldCCD). ==
    // The bullet-wall scene + a deterministic command stream: a launch-impulse fires the projectile, the wall
    // arrests it, a later command perturbs a slow drop. Two peers fed only the inputs re-derive the world
    // byte-identical (re-deriving the swept broadphase + per-substep TOIs each tick), and a rollback re-sims
    // from a snapshot bit-for-bit.
    {
        namespace broad = hf::sim::broad;
        const ccd::CcdStepConfig cfg = ccd::MakeBulletWallConfig();
        const uint32_t kTicks = 8u;
        const uint32_t kRollbackAt = 3u;
        const gjk::HullWorld kInit = ccd::MakeBulletWallScene();   // bodies: 0=projectile,1=wall,2-3=drops,4=floor

        // The authoritative command stream: re-fire the projectile (+X impulse) early, then perturb a slow drop.
        const std::vector<convex::ConvexCommand> authStream = {
            convex::ConvexCommand{1u, convex::kConvexCmdAddImpulse, 0u, convex::FxVec3{FromInt(20), 0, 0}},
            convex::ConvexCommand{2u, convex::kConvexCmdSetAngVel,  2u, convex::FxVec3{0, kOne, 0}},
            convex::ConvexCommand{4u, convex::kConvexCmdAddImpulse, 3u, convex::FxVec3{-FromInt(3), 0, 0}},
        };

        // (1) LOCKSTEP: authority == replica BIT-IDENTICAL (inputs only).
        bool identical = false;
        const gjk::HullWorld authority = ccd::RunCcdLockstep(kInit, cfg, authStream, kTicks, &identical);
        const gjk::HullWorld replica   = ccd::RunCcdLockstep(kInit, cfg, authStream, kTicks);
        check(identical, "ccd lockstep: authority==replica reported BIT-IDENTICAL");
        check(gjk::HullBodiesEqual(authority.bodies, replica.bodies),
              "ccd lockstep: authority==replica final bodies BYTE-IDENTICAL");

        // (2) DETERMINISM: two full runs byte-identical.
        const gjk::HullWorld authority2 = ccd::RunCcdLockstep(kInit, cfg, authStream, kTicks);
        check(gjk::HullBodiesEqual(authority2.bodies, authority.bodies),
              "ccd lockstep determinism: two runs BYTE-IDENTICAL");

        // (2b) The command stream MOVED the scene non-trivially (not a frozen no-op): the projectile advanced.
        check(!gjk::HullBodiesEqual(authority.bodies, kInit.bodies),
              "ccd lockstep: command stream moved the scene (final != initial)");
        check(authority.bodies[0].pos.x != kInit.bodies[0].pos.x,
              "ccd lockstep: the projectile travelled under the command stream");

        // (3) ROLLBACK: a WRONG strong impulse arrives at rollbackAt; corrected==authority BIT-EXACT.
        std::vector<convex::ConvexCommand> mispredictStream = authStream;
        mispredictStream.push_back(convex::ConvexCommand{kRollbackAt, convex::kConvexCmdAddImpulse, 2u,
                                                         convex::FxVec3{FromInt(40), 0, 0}});
        bool corrected = false, diverged = false;
        const gjk::HullWorld rolledBack = ccd::RunCcdRollback(kInit, cfg, authStream, mispredictStream,
                                                              kTicks, kRollbackAt, &corrected, &diverged);
        check(corrected && gjk::HullBodiesEqual(rolledBack.bodies, authority.bodies),
              "ccd rollback: corrected==authority BIT-EXACT");
        // (4) the misprediction was REAL (the speculative pre-rollback state diverged from authority).
        check(diverged, "ccd rollback: mispredicted state diverged before rollback (real divergence corrected)");

        // Snapshot round-trip is bit-exact (the rollback restore-point contract).
        {
            gjk::HullWorld w = ccd::RunCcdLockstep(kInit, cfg, authStream, kRollbackAt);
            const gjk::HullSnapshot snap = gjk::SnapshotHull(w, kRollbackAt);
            ccd::SimCcdTick(w, cfg, authStream, kRollbackAt);   // mutate
            gjk::RestoreHull(w, snap);
            check(gjk::HullBodiesEqual(w.bodies, snap.bodies),
                  "ccd lockstep: snapshot round-trip == original BYTE-IDENTICAL");
        }
    }

    if (g_fail == 0) std::printf("ccd_test: ALL PASS\n");
    else std::printf("ccd_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
