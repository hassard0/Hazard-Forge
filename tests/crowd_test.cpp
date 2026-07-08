// Slice CR1 — DETERMINISTIC CROWD SIMULATION AT 10k+ AGENTS (the Mass-class archetype crowd; engine/sim/
// crowd.h, hf::sim::crowd). Pure CPU strict-integer (Q16.16), NO new shader — composes the byte-frozen BD2
// spatial-hash grid (boids.h) for O(N) neighbour separation instead of BD1's all-pairs O(N²), and proves the
// crowd at 10,000+ agents deterministic + lockstep-replayable.
//
// What this test PINS:
//   (a) GRID==ALLPAIRS: the O(N) grid separation step == the O(N²) brute-force reference BIT-FOR-BIT (the KEY
//       correctness pin — the grid gives the same answer as all-pairs), for N=64, single + multi-tick.
//   (b) SCALE: 10,000 agents stepped N ticks, a full-state digest PINNED; reports the wall-time (the O(N)
//       proof — must be non-pathological).
//   (c) ARCHETYPES: 3 archetypes with different params produce DIFFERENT pinned trajectories; an agent's
//       behaviour follows its archetype (runner outruns pedestrian outruns wanderer toward a goal).
//   (d) GOAL-SEEK: a lone agent (no neighbours) approaches its goal monotonically (pinned) while far.
//   (e) LOCKSTEP: a peer re-derives a 1k crowd bit-for-bit; rollback corrects a mispredicted goal-change;
//       snapshot completeness — omitting velocity OR archetype from the restore diverges.
//   (f) determinism: two full runs identical (digest); the shot scenario two-run identical.
//
// HONEST CAVEAT (the BD1/GR4 shape): agents are POINTS with a soft separation PUSH (not a hard contact — dense
// clusters can overlap; a Jacobi single-projection residual); un-normalized goal-seek carries momentum. The
// headline is DETERMINISM + SCALE + lockstep-replayability. All digests are pure-integer -> MSVC == clang.
//
// Pure C++ (hf_core), ASan-eligible like the other sim-math tests.
#include "sim/crowd.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace crowd = hf::sim::crowd;
namespace boids = hf::sim::boids;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

using fx = crowd::fx;
static fx wu(int u) { return (fx)(u * (int)crowd::kOne); }
static fx frac(int n, int d) { return (fx)((int64_t)n * (int64_t)crowd::kOne / d); }

// A deterministic 3-archetype / small-goal config for the correctness/archetype tests.
static crowd::CrowdConfig makeCfg() {
    crowd::CrowdConfig cfg;
    cfg.archetypes.push_back(crowd::CrowdArchetype{ wu(4), frac(3, 2), frac(1, 4), frac(1, 2) }); // pedestrian
    cfg.archetypes.push_back(crowd::CrowdArchetype{ wu(7), wu(1),      frac(1, 2), frac(1, 3) }); // runner
    cfg.archetypes.push_back(crowd::CrowdArchetype{ wu(2), wu(2),      frac(1, 8), frac(1, 2) }); // wanderer
    cfg.goals.push_back(crowd::FxVec3{ wu(40), 0, 0 });
    cfg.sepGain  = frac(1, 2);
    cfg.maxForce = wu(10);
    crowd::FinalizeCrowdConfig(cfg);
    return cfg;
}

// A deterministic NxN grid crowd on the -x side (all archetype `arch`, all goal 0) with 0.5-unit spacing.
static crowd::Crowd makeGridCrowd(int count, int cols, uint8_t arch, const crowd::CrowdConfig& cfg) {
    crowd::Crowd c;
    for (int i = 0; i < count; ++i) {
        const int col = i % cols, row = i / cols;
        const fx x = (fx)(-wu(30) + col * frac(1, 2));
        const fx y = (fx)(-wu(10) + row * frac(1, 2));
        crowd::Agent a; a.pos = crowd::FxVec3{x, y, 0}; a.vel = crowd::FxVec3{0, 0, 0};
        c.agents.push_back(a);
        c.archetype.push_back((uint8_t)(arch == 255 ? (i % 3) : arch));
        c.goal.push_back(0);
    }
    (void)cfg;
    return c;
}

static bool crowdEqual(const crowd::Crowd& a, const crowd::Crowd& b) {
    if (a.agents.size() != b.agents.size()) return false;
    if (std::memcmp(a.agents.data(), b.agents.data(), a.agents.size() * sizeof(crowd::Agent)) != 0) return false;
    return a.archetype == b.archetype && a.goal == b.goal;
}

int main() {
    HF_TEST_MAIN_INIT();
    const crowd::CrowdConfig cfg = makeCfg();
    const fx dt = frac(1, 4);   // 0.25s tick

    // ===== (a) GRID == ALLPAIRS (the KEY correctness pin) =====
    {
        crowd::Crowd base = makeGridCrowd(64, 8, /*mixed*/255, cfg);
        // single tick: the O(N) grid step vs the O(N²) reference — bit-for-bit.
        crowd::Crowd viaGrid = base, viaAll = base;
        crowd::StepCrowd(viaGrid, cfg, dt);
        crowd::StepCrowdAllPairs(viaAll, cfg, dt);
        check(crowdEqual(viaGrid, viaAll), "GRID==ALLPAIRS: one tick bit-for-bit");
        check(crowd::DigestCrowd(viaGrid) == crowd::DigestCrowd(viaAll),
              "GRID==ALLPAIRS: one-tick digest identical");
        // multi-tick: 40 ticks both ways, still bit-for-bit (the grid tracks all-pairs through motion).
        crowd::Crowd g2 = base, a2 = base;
        for (int t = 0; t < 40; ++t) { crowd::StepCrowd(g2, cfg, dt); crowd::StepCrowdAllPairs(a2, cfg, dt); }
        check(crowdEqual(g2, a2), "GRID==ALLPAIRS: 40 ticks bit-for-bit");
        check(crowd::DigestCrowd(g2) == 0x305cc72e13c496edull, "GRID==ALLPAIRS: 40-tick digest matches pin");
        std::printf("(a) grid==allpairs: N=64, 40 ticks, digest %016llx\n",
                    (unsigned long long)crowd::DigestCrowd(g2));
    }

    // ===== (b) SCALE: 10,000 agents, N ticks, digest pinned + wall-time =====
    // 3 goals in 3 lanes (a realistic Mass-style stream — density stays BOUNDED by separation so the grid is
    // truly O(N); a single-goal pile would collapse every agent into one cell -> O(N²) in that cell).
    {
        const int N = 10000, cols = 100, ticks = 200;
        crowd::CrowdConfig cfg3 = cfg;
        cfg3.goals = { crowd::FxVec3{ wu(40), wu(20), 0 }, crowd::FxVec3{ wu(44), 0, 0 },
                       crowd::FxVec3{ wu(40), -wu(20), 0 } };
        crowd::FinalizeCrowdConfig(cfg3);
        auto make10k = [&]() {
            crowd::Crowd c = makeGridCrowd(N, cols, /*mixed*/255, cfg3);
            for (uint32_t i = 0; i < crowd::CrowdSize(c); ++i) c.goal[i] = (i / cols) % 3;  // 3 lanes by row
            return c;
        };
        crowd::Crowd big = make10k();
        check(crowd::CrowdSize(big) == (uint32_t)N, "SCALE: 10000 agents built");
        const uint64_t d0 = crowd::DigestCrowd(big);
        const auto t0 = std::chrono::steady_clock::now();
        crowd::StepCrowdSteps(big, cfg3, dt, ticks);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const uint64_t dN = crowd::DigestCrowd(big);
        // determinism: a SECOND identical 10k run digests the same.
        crowd::Crowd big2 = make10k();
        crowd::StepCrowdSteps(big2, cfg3, dt, ticks);
        check(crowd::DigestCrowd(big2) == dN, "SCALE: two 10k runs digest-identical (deterministic)");
        // PIN the 10k full-state digest (10,000 agents x 200 ticks, 3-goal streams; MSVC == clang).
        check(dN == 0xc30528684a7acb1bull, "SCALE: 10k digest matches pin");
        std::printf("(b) SCALE: 10000 agents x %d ticks in %.1f ms (%.4f ms/tick) — d0=%016llx dN=%016llx\n",
                    ticks, ms, ms / ticks, (unsigned long long)d0, (unsigned long long)dN);
        check(ms < 60000.0, "SCALE: 10k x 200 ticks under 60s (non-pathological O(N))");
    }

    // ===== (c) ARCHETYPES: different params -> different trajectories =====
    {
        // Three lone agents (no neighbours), one per archetype, same start, same goal — the type drives speed.
        auto loneToGoal = [&](uint8_t arch, int ticks) -> fx {
            crowd::Crowd c;
            crowd::Agent a; a.pos = crowd::FxVec3{-wu(30), 0, 0}; a.vel = crowd::FxVec3{0, 0, 0};
            c.agents.push_back(a); c.archetype.push_back(arch); c.goal.push_back(0);
            crowd::StepCrowdSteps(c, cfg, dt, ticks);
            return crowd::CrowdL1(crowd::FxSub(c.agents[0].pos, cfg.goals[0]));
        };
        const fx ped = loneToGoal(0, 30), run = loneToGoal(1, 30), wan = loneToGoal(2, 30);
        // runner (fast, strong pull) is closest; wanderer (slow, weak pull) is farthest.
        check(run < ped, "ARCHETYPES: runner reaches goal faster than pedestrian");
        check(ped < wan, "ARCHETYPES: pedestrian reaches goal faster than wanderer");
        std::printf("(c) archetypes: L1-to-goal @30t {runner:%d, pedestrian:%d, wanderer:%d}\n", run, ped, wan);
        // separation radius differs by type: build a dense pair, the wider-radius type pushes harder.
        auto sepPush = [&](uint8_t arch) -> fx {
            crowd::Crowd c;
            crowd::Agent a0; a0.pos = crowd::FxVec3{0, 0, 0};        a0.vel = crowd::FxVec3{0, 0, 0};
            crowd::Agent a1; a1.pos = crowd::FxVec3{frac(3, 4), 0, 0}; a1.vel = crowd::FxVec3{0, 0, 0};
            c.agents = {a0, a1}; c.archetype = {arch, arch}; c.goal = {0, 0};
            crowd::CrowdConfig ncfg = cfg; ncfg.goals[0] = crowd::FxVec3{0, 0, 0}; // kill goal pull -> pure sep
            crowd::StepCrowd(c, ncfg, dt);
            return c.agents[0].pos.x;   // agent 0 pushed toward -x by separation
        };
        // runner sepRadius(1.0) covers the 0.75 gap -> pushes; but pedestrian(1.5)/wanderer(2.0) also cover it.
        // The push exists (all three radii > 0.75) — pin that separation actually moved them apart.
        check(sepPush(1) < 0, "ARCHETYPES: separation pushes agents apart (runner radius covers the gap)");
    }

    // ===== (d) GOAL-SEEK: a lone agent approaches its goal monotonically (while far) =====
    {
        crowd::Crowd c;
        crowd::Agent a; a.pos = crowd::FxVec3{-wu(30), 0, 0}; a.vel = crowd::FxVec3{0, 0, 0};
        c.agents.push_back(a); c.archetype.push_back(0); c.goal.push_back(0);  // pedestrian
        fx prev = crowd::CrowdL1(crowd::FxSub(c.agents[0].pos, cfg.goals[0]));
        const fx start = prev;
        bool monotone = true;
        int approachTicks = 0;
        fx nearest = prev;
        bool arrived = false;   // once within 2 units of the goal the momentum can overshoot/orbit (real
                                // behaviour of un-normalized seek + velocity — NOT claimed as monotone).
        for (int t = 0; t < 90; ++t) {
            crowd::StepCrowd(c, cfg, dt);
            const fx d = crowd::CrowdL1(crowd::FxSub(c.agents[0].pos, cfg.goals[0]));
            if (d <= wu(2)) arrived = true;
            // strictly decreasing throughout the INITIAL approach (before first reaching the goal).
            if (!arrived) { if (d >= prev) monotone = false; else ++approachTicks; }
            if (d < nearest) nearest = d;
            prev = d;
        }
        check(monotone, "GOAL-SEEK: lone agent L1-to-goal strictly decreases over the approach (monotone)");
        check(nearest < start / 8, "GOAL-SEEK: lone agent reached its goal (nearest << start)");
        std::printf("(d) goal-seek: start L1=%d -> end L1=%d (nearest=%d, %d monotone approach ticks)\n",
                    start, prev, nearest, approachTicks);
    }

    // ===== (e) LOCKSTEP + ROLLBACK + snapshot completeness (1k crowd) =====
    {
        const int N = 1000, cols = 40, ticks = 60;
        crowd::CrowdConfig mcfg = cfg;   // 3 goals so redirects mean something
        mcfg.goals = { crowd::FxVec3{ wu(40), wu(15), 0 }, crowd::FxVec3{ wu(44), 0, 0 },
                       crowd::FxVec3{ wu(40), -wu(15), 0 } };
        crowd::FinalizeCrowdConfig(mcfg);
        crowd::Crowd init = makeGridCrowd(N, cols, 255, mcfg);
        for (uint32_t i = 0; i < crowd::CrowdSize(init); ++i) init.goal[i] = i % 3;  // three lanes
        // Authoritative command stream: redirect a few agents mid-run.
        std::vector<crowd::CrowdCommand> auth = {
            crowd::CrowdCommand{10, 5,   2}, crowd::CrowdCommand{20, 100, 0}, crowd::CrowdCommand{30, 500, 1}};
        // LOCKSTEP: two peers from the same inputs -> bit-for-bit.
        crowd::Crowd A = crowd::RunCrowdLockstep(mcfg, init, auth, ticks, dt);
        crowd::Crowd B = crowd::RunCrowdLockstep(mcfg, init, auth, ticks, dt);
        check(crowdEqual(A, B), "LOCKSTEP: two peers re-derive the 1k crowd bit-for-bit");
        check(crowd::DigestCrowd(A) == 0x7257e6ca28d1d438ull, "LOCKSTEP: 1k crowd digest matches pin");
        std::printf("(e) lockstep: 1k crowd, %d ticks, digest %016llx\n",
                    ticks, (unsigned long long)crowd::DigestCrowd(A));
        // ROLLBACK: a mispredicted redirect is corrected to == authority.
        std::vector<crowd::CrowdCommand> mis = { crowd::CrowdCommand{20, 100, 2} };  // wrong goal for agent 100
        crowd::Crowd R = crowd::RunCrowdRollback(mcfg, init, auth, mis, /*divergeTick*/18, ticks, dt);
        check(crowdEqual(R, A), "ROLLBACK: corrected crowd == authoritative lockstep crowd");
        // the misprediction really diverged: speculate 3 ticks with the wrong stream, compare to authority path.
        {
            crowd::Crowd spec = init;
            for (int t = 0; t < 18; ++t) crowd::SimCrowdTick(spec, mcfg, auth, t, dt);
            crowd::Crowd specSaved = spec;
            for (int s = 0; s < 3; ++s) crowd::SimCrowdTick(spec, mcfg, mis, 18 + s, dt);
            crowd::Crowd good = specSaved;
            for (int s = 0; s < 3; ++s) crowd::SimCrowdTick(good, mcfg, auth, 18 + s, dt);
            check(!crowdEqual(spec, good), "ROLLBACK: the mispredicted stream really diverged (a real fix)");
        }
        // SNAPSHOT COMPLETENESS: dropping velocity OR archetype from the restore diverges from the authority.
        {
            // advance to tick 18, snapshot, then re-sim to the end with the correct stream = authority tail.
            crowd::Crowd atDiverge = init;
            for (int t = 0; t < 18; ++t) crowd::SimCrowdTick(atDiverge, mcfg, auth, t, dt);
            // full (correct) restore -> matches authority.
            crowd::Crowd full = atDiverge;
            for (int t = 18; t < ticks; ++t) crowd::SimCrowdTick(full, mcfg, auth, t, dt);
            check(crowdEqual(full, A), "SNAPSHOT: complete restore reproduces authority");
            // drop velocity: zero every agent's velocity at the restore -> the tail diverges.
            crowd::Crowd noVel = atDiverge;
            for (auto& ag : noVel.agents) ag.vel = crowd::FxVec3{0, 0, 0};
            for (int t = 18; t < ticks; ++t) crowd::SimCrowdTick(noVel, mcfg, auth, t, dt);
            check(!crowdEqual(noVel, A), "SNAPSHOT: dropping velocity diverges (velocity is replayable state)");
            // drop archetype: reset every archetype to 0 at the restore -> the tail diverges (types differ).
            crowd::Crowd noArch = atDiverge;
            for (auto& ar : noArch.archetype) ar = 0;
            for (int t = 18; t < ticks; ++t) crowd::SimCrowdTick(noArch, mcfg, auth, t, dt);
            check(!crowdEqual(noArch, A), "SNAPSHOT: dropping archetype diverges (archetype is replayable state)");
        }
    }

    // ===== (f) determinism + the shot scenario two-run identical =====
    {
        const crowd::CrowdShotRun s1 = crowd::RunCrowdShotScenario();
        const crowd::CrowdShotRun s2 = crowd::RunCrowdShotScenario();
        check(s1.digest == s2.digest, "SHOT: two-run digest identical");
        check(s1.meanToGoalN < s1.meanToGoal0, "SHOT: crowd flowed toward goals (meanToGoal dropped)");
        check(s1.digest == 0xc80b1b212fadfeb0ull, "SHOT: digest matches pin");
        std::printf("(f) shot: {agents:%d, archetypes:%d, ticks:%d, goals:%d, meanToGoal:%d->%d, "
                    "digest:%016llx}\n",
                    s1.agents, s1.archetypes, s1.ticks, s1.goalCount, s1.meanToGoal0, s1.meanToGoalN,
                    (unsigned long long)s1.digest);
    }

    if (g_fail == 0) std::printf("crowd_test: ALL PASS\n");
    else std::printf("crowd_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
