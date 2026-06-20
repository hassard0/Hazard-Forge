// Slice BD1 — Deterministic GPU Crowds: the INTEGER STEERING PRIMITIVE integer core (engine/sim/boids.h) the
// GPU shaders/boids_steer.comp.hlsl copies VERBATIM + proves bit-identical. Pure CPU (header-only, no device,
// no backend symbols). Namespace hf::sim::boids.
//
// What this test PINS (the contracts the GPU boids_steer.comp + the GPU==CPU proof build on):
//   * SteerSeek: the force points target-ward (each component has the SAME SIGN as target - pos), is zero at
//     the target, and scales with seekGain.
//   * SteerSeparation: a too-close neighbor pushes the agent AWAY (the force points away from the neighbor);
//     a neighbor BEYOND sepRadius contributes nothing (zero force); two equidistant neighbors on opposite
//     sides cancel.
//   * StepBoids: a tight cluster seeking a target SPREADS + ADVANCES (meanToTarget drops, minSep stays above
//     a floor); a sepGain=0 control COLLAPSES (minSep -> ~0); the per-axis clamp caps force/speed; two runs
//     byte-identical.
//
// HONEST CAVEAT (the GR4-shape): boids are POINTS with steering FORCES (a soft separation push, NOT a hard
// non-penetration contact); the per-axis clamp is an axis-BOX, not a radial magnitude clamp. The headline is
// DETERMINISM + cross-platform bit-identity.
//
// Pure C++ (hf_core), ASan-eligible like the other sim-math tests.
#include "sim/boids.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace boids = hf::sim::boids;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static boids::fx fxabs(boids::fx v) { return v < 0 ? -v : v; }

// A Q16.16 scalar from an integer world unit.
static boids::fx wu(int u) { return (boids::fx)(u * (int)boids::kOne); }
// A Q16.16 scalar from a fraction n/d of a world unit.
static boids::fx frac(int n, int d) { return (boids::fx)((int64_t)n * (int64_t)boids::kOne / d); }

// An agent at integer world (x,y,z) with zero velocity.
static boids::Agent at(int x, int y, int z) {
    return boids::Agent{boids::FxVec3{wu(x), wu(y), wu(z)}, boids::FxVec3{0, 0, 0}};
}

// The shared steering tuning the StepBoids cases use (matches the showcase config).
static boids::BoidsConfig flockCfg(boids::fx sepGain) {
    boids::BoidsConfig c;
    c.seekGain  = frac(1, 4);    // 0.25 proportional seek
    c.sepGain   = sepGain;       // 0.5 (on) or 0 (control)
    c.sepRadius = wu(2);         // 2-world-unit neighbor radius
    c.maxForce  = wu(8);
    c.maxSpeed  = wu(6);
    c.target    = boids::FxVec3{wu(40), 0, 0};   // a target offset to one side (+x)
    c.gravity   = boids::FxVec3{0, 0, 0};
    return c;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ===== SteerSeek =====
    {
        boids::BoidsConfig c;
        c.seekGain = frac(1, 2);   // 0.5
        c.target   = boids::FxVec3{wu(10), wu(5), wu(-3)};
        boids::Agent a = at(0, 0, 0);
        boids::FxVec3 f = boids::SteerSeek(a, c.target, c);
        // force points target-ward: same sign as (target - pos) per axis, nonzero where target != pos.
        check(f.x > 0, "SteerSeek: +x target -> +x force");
        check(f.y > 0, "SteerSeek: +y target -> +y force");
        check(f.z < 0, "SteerSeek: -z target -> -z force");
        // scales with seekGain: 0.5 * (10,5,-3) world units.
        check(f.x == boids::fxmul(wu(10), c.seekGain), "SteerSeek: force.x == desired.x * seekGain");
        check(f.y == boids::fxmul(wu(5),  c.seekGain), "SteerSeek: force.y == desired.y * seekGain");
        // at the target -> zero force.
        boids::Agent atTarget{c.target, boids::FxVec3{0, 0, 0}};
        boids::FxVec3 f0 = boids::SteerSeek(atTarget, c.target, c);
        check(f0.x == 0 && f0.y == 0 && f0.z == 0, "SteerSeek: zero at the target");
        // larger gain -> larger force.
        boids::BoidsConfig c2 = c; c2.seekGain = boids::kOne;  // 1.0
        boids::FxVec3 f2 = boids::SteerSeek(a, c.target, c2);
        check(fxabs(f2.x) > fxabs(f.x), "SteerSeek: larger gain -> larger force");
    }

    // ===== SteerSeparation =====
    {
        boids::BoidsConfig c;
        c.sepGain   = boids::kOne;   // 1.0 so the sign survives
        c.sepRadius = wu(3);
        // a is at origin; a single close neighbor to the +x side -> push toward -x (away).
        boids::Agent a = at(0, 0, 0);
        std::vector<boids::Agent> others = {a, at(1, 0, 0)};   // index 0 = self, index 1 = +x neighbor
        boids::FxVec3 f = boids::SteerSeparation(a, std::span<const boids::Agent>(others), c, 0);
        check(f.x < 0, "SteerSeparation: a close +x neighbor pushes -x (away)");
        check(f.y == 0 && f.z == 0, "SteerSeparation: pure +x neighbor -> only x push");
        // a neighbor BEYOND sepRadius contributes nothing.
        std::vector<boids::Agent> far = {a, at(10, 0, 0)};     // 10 > sepRadius 3
        boids::FxVec3 ff = boids::SteerSeparation(a, std::span<const boids::Agent>(far), c, 0);
        check(ff.x == 0 && ff.y == 0 && ff.z == 0, "SteerSeparation: beyond sepRadius -> zero");
        // two equidistant neighbors on opposite sides cancel.
        std::vector<boids::Agent> both = {a, at(1, 0, 0), at(-1, 0, 0)};
        boids::FxVec3 fb = boids::SteerSeparation(a, std::span<const boids::Agent>(both), c, 0);
        check(fb.x == 0 && fb.y == 0 && fb.z == 0, "SteerSeparation: symmetric neighbors cancel");
        // exactly on the radius is NOT a neighbor (strict d² < r²): a neighbor at distance == sepRadius.
        std::vector<boids::Agent> onEdge = {a, at(3, 0, 0)};   // distance 3 == sepRadius 3
        boids::FxVec3 fe = boids::SteerSeparation(a, std::span<const boids::Agent>(onEdge), c, 0);
        check(fe.x == 0 && fe.y == 0 && fe.z == 0, "SteerSeparation: at exactly sepRadius -> excluded (d²<r²)");
    }

    // ===== StepBoids: a cluster seeks the target -> spreads + advances =====
    {
        boids::BoidsConfig c = flockCfg(frac(1, 2));   // sepGain 0.5 (separation ON)
        const boids::fx dt = boids::kOne / 60;
        // a tight 4x4 cluster near the origin (1-unit spacing < sepRadius 2 so they push apart).
        std::vector<boids::Agent> agents;
        for (int gx = 0; gx < 4; ++gx)
            for (int gz = 0; gz < 4; ++gz)
                agents.push_back(at(gx, 0, gz));

        boids::BoidsStats before = boids::MeasureBoids(agents, c);
        boids::StepBoidsSteps(agents, c, dt, 240);
        boids::BoidsStats after = boids::MeasureBoids(agents, c);

        // they sought the target: mean distance to target DROPPED.
        check(after.meanToTarget < before.meanToTarget, "StepBoids: meanToTarget drops (sought the target)");
        // they didn't collapse: min separation stays above a floor (separation kept them apart).
        check(after.minSep > frac(1, 4), "StepBoids: minSep above a floor (didn't collapse)");
        // they moved (nonzero mean speed).
        check(after.meanSpeed > 0, "StepBoids: nonzero mean speed (the flock is moving)");
    }

    // ===== StepBoids control: sepGain=0 collapses =====
    {
        boids::BoidsConfig on  = flockCfg(frac(1, 2));
        boids::BoidsConfig off = flockCfg(0);          // separation OFF
        const boids::fx dt = boids::kOne / 60;
        std::vector<boids::Agent> agOn, agOff;
        for (int gx = 0; gx < 4; ++gx)
            for (int gz = 0; gz < 4; ++gz) { agOn.push_back(at(gx, 0, gz)); agOff.push_back(at(gx, 0, gz)); }
        boids::StepBoidsSteps(agOn,  on,  dt, 600);
        boids::StepBoidsSteps(agOff, off, dt, 600);
        boids::BoidsStats sOn  = boids::MeasureBoids(agOn,  on);
        boids::BoidsStats sOff = boids::MeasureBoids(agOff, off);
        // the no-separation flock collapses much tighter than the separating one.
        check(sOff.minSep < sOn.minSep, "StepBoids control: sepGain=0 collapses tighter than sepGain>0");
        check(sOff.minSep <= frac(1, 8), "StepBoids control: sepGain=0 minSep -> ~0 (collapsed)");
    }

    // ===== per-axis clamp caps force + speed =====
    {
        // ClampAxis directly.
        check(boids::ClampAxis(wu(100), wu(8)) == wu(8),  "ClampAxis: positive over-limit clamped to +limit");
        check(boids::ClampAxis(wu(-100), wu(8)) == wu(-8),"ClampAxis: negative over-limit clamped to -limit");
        check(boids::ClampAxis(wu(3), wu(8)) == wu(3),    "ClampAxis: within limit unchanged");
        // a huge seek force + huge dt must NOT exceed maxSpeed per axis after one step.
        boids::BoidsConfig c;
        c.seekGain  = boids::kOne * 100;   // absurd gain -> huge raw force
        c.sepGain   = 0;
        c.sepRadius = 0;
        c.maxForce  = wu(8);
        c.maxSpeed  = wu(6);
        c.target    = boids::FxVec3{wu(1000), 0, 0};
        std::vector<boids::Agent> a = {at(0, 0, 0)};
        boids::StepBoids(a, c, boids::kOne / 60);
        check(fxabs(a[0].vel.x) <= c.maxSpeed, "StepBoids: vel.x capped to maxSpeed despite huge force");
        check(fxabs(a[0].vel.y) <= c.maxSpeed, "StepBoids: vel.y capped to maxSpeed");
    }

    // ===== two runs byte-identical (determinism) =====
    {
        boids::BoidsConfig c = flockCfg(frac(1, 2));
        const boids::fx dt = boids::kOne / 60;
        std::vector<boids::Agent> a1, a2;
        for (int gx = 0; gx < 6; ++gx)
            for (int gz = 0; gz < 6; ++gz) { a1.push_back(at(gx, 0, gz)); a2.push_back(at(gx, 0, gz)); }
        boids::StepBoidsSteps(a1, c, dt, 300);
        boids::StepBoidsSteps(a2, c, dt, 300);
        check(a1.size() == a2.size() &&
              std::memcmp(a1.data(), a2.data(), a1.size() * sizeof(boids::Agent)) == 0,
              "StepBoids: two runs BYTE-IDENTICAL");
        static_assert(sizeof(boids::Agent) == 24, "Agent std430 layout (6 x int32, 24 bytes)");
    }

    // ===== Slice BD2: the GRID-HASH NEIGHBOR LIST (engine/sim/boids.h, the GR2 grain engine cloned) =====
    // What this PINS (the contracts boids_cell_*/boids_neighbor_*.comp + the GPU==CPU memcmp build on):
    //   * BuildBoidsCellTable partitions EVERY agent into a cell exactly once (cellAgents.size()==N, cellStart
    //     monotone non-decreasing, last == N — a complete CSR partition).
    //   * BuildBoidsNeighborList: every emitted neighbor is within the radius box AND i != j; the grid result
    //     MATCHES a brute-force O(N²) box reference exactly (no misses, no extras) — the grid found ALL true
    //     neighbors and ONLY them.
    //   * Two runs byte-identical (deterministic).
    {
        const boids::fx radius = wu(2);   // perception radius == the cell size

        // A 5x1x5 agent grid at 1-unit spacing (so the radius-2 box neighborhood is non-trivial).
        std::vector<boids::Agent> agents;
        for (int gx = 0; gx < 5; ++gx)
            for (int gz = 0; gz < 5; ++gz)
                agents.push_back(at(gx, 0, gz));
        const uint32_t N = (uint32_t)agents.size();

        const boids::BoidsGrid grid = boids::MakeBoidsGrid(agents, radius);
        const boids::BoidsCellTable table = boids::BuildBoidsCellTable(agents, grid);

        // (1) partition complete: cellAgents has exactly N entries; cellStart monotone; last == N.
        check(table.cellAgents.size() == N, "BuildBoidsCellTable: cellAgents.size() == N (every agent bucketed)");
        check(table.cellStart.size() == (size_t)boids::BoidsCellCount(grid) + 1u,
              "BuildBoidsCellTable: cellStart has cellCount+1 entries");
        bool monotone = true;
        for (size_t c = 1; c < table.cellStart.size(); ++c)
            if (table.cellStart[c] < table.cellStart[c - 1]) monotone = false;
        check(monotone, "BuildBoidsCellTable: cellStart monotone non-decreasing");
        check(table.cellStart.back() == N, "BuildBoidsCellTable: cellStart last == N (the total)");
        // every agent index appears exactly once across cellAgents.
        {
            std::vector<int> seen((size_t)N, 0);
            for (uint32_t v : table.cellAgents) if (v < N) ++seen[(size_t)v];
            bool eachOnce = true;
            for (uint32_t i = 0; i < N; ++i) if (seen[(size_t)i] != 1) eachOnce = false;
            check(eachOnce, "BuildBoidsCellTable: each agent index appears exactly once");
        }

        // (2) neighbor list: every emitted neighbor within radius + matches the brute-force reference.
        const boids::BoidsNeighborList list = boids::BuildBoidsNeighborList(agents, grid, table, radius);
        check(list.neighborStart.size() == (size_t)N + 1u, "BuildBoidsNeighborList: neighborStart N+1 entries");
        check(list.neighborStart.back() == (uint32_t)list.neighbors.size(),
              "BuildBoidsNeighborList: neighborStart last == neighbors.size()");
        bool withinRadius = true;
        for (uint32_t i = 0; i < N && withinRadius; ++i)
            for (uint32_t s = list.neighborStart[i]; s < list.neighborStart[i + 1u]; ++s) {
                uint32_t j = list.neighbors[s];
                if (j == i) withinRadius = false;
                if (!boids::BoidsNeighborAccept(agents[i].pos, agents[j].pos, radius)) withinRadius = false;
            }
        check(withinRadius, "BuildBoidsNeighborList: every emitted neighbor within radius (and i!=j)");

        // brute-force O(N²) box reference: the grid result must match it set-for-set (no misses/extras).
        bool matchesBrute = true;
        for (uint32_t i = 0; i < N && matchesBrute; ++i) {
            // reference: the SORTED set of j!=i with BoidsNeighborAccept.
            std::vector<uint32_t> ref;
            for (uint32_t j = 0; j < N; ++j) {
                if (j == i) continue;
                if (boids::BoidsNeighborAccept(agents[i].pos, agents[j].pos, radius)) ref.push_back(j);
            }
            // the grid list for i, sorted (the emit order is stencil-then-ascending-j, so it may differ; the
            // SET must match — sort both then compare).
            std::vector<uint32_t> got(list.neighbors.begin() + list.neighborStart[i],
                                      list.neighbors.begin() + list.neighborStart[i + 1u]);
            std::sort(got.begin(), got.end());
            if (got.size() != ref.size() || std::memcmp(got.data(), ref.data(), ref.size() * sizeof(uint32_t)))
                matchesBrute = false;
        }
        check(matchesBrute, "BuildBoidsNeighborList: matches the brute-force O(N²) reference (no misses/extras)");

        // (3) two runs byte-identical (deterministic).
        const boids::BoidsCellTable t2 = boids::BuildBoidsCellTable(agents, grid);
        const boids::BoidsNeighborList l2 = boids::BuildBoidsNeighborList(agents, grid, t2, radius);
        check(t2.cellStart == table.cellStart && t2.cellAgents == table.cellAgents &&
              l2.neighborStart == list.neighborStart && l2.neighbors == list.neighbors,
              "BuildBoids{CellTable,NeighborList}: two runs BYTE-IDENTICAL");

        // (4) a single isolated agent -> ZERO neighbors (the sparse no-op).
        {
            std::vector<boids::Agent> one = {at(3, 5, 0)};
            boids::BoidsGrid g1 = boids::MakeBoidsGrid(one, radius);
            boids::BoidsCellTable c1 = boids::BuildBoidsCellTable(one, g1);
            boids::BoidsNeighborList n1 = boids::BuildBoidsNeighborList(one, g1, c1, radius);
            check(n1.neighbors.empty() && n1.neighborStart[1] == 0u,
                  "BuildBoidsNeighborList: single agent -> 0 neighbors");
        }
    }

    // ===== Slice BD3: THE FULL FLOCK STEP (separation + alignment + cohesion over the BD2 neighbor list) =====
    // What this PINS (the contracts boids_flock.comp + the GPU==CPU memcmp build on):
    //   * AccumFlock: separation points AWAY from a near neighbor; alignment points toward the neighbors' mean
    //     velocity; cohesion points toward the neighbors' mean position; zero-neighbors -> no crash, zero force.
    //   * StepFlock: a spread flock COHERES (the bounding diagonal shrinks) + ALIGNS (heading-alignment rises)
    //     while keeping min-separation above a floor; an alignGain=cohGain=0 (sep-only) control stays loose
    //     (does NOT cohere); two runs byte-identical.

    // A FlockConfig with the 3 gains + a perception radius (no seek -> free flocking). The recipe (perception
    // radius 3, weak separation 0.125, strong cohesion) is the showcase config: a dense flock CLUSTERS (diagonal
    // shrinks) + ALIGNS while weak separation keeps spacing above a floor — proven in the showcase + here.
    auto flockCfg3 = [&](boids::fx alignG, boids::fx cohG) {
        boids::FlockConfig c;
        c.seekGain         = 0;            // free flocking (no goal)
        c.sepGain          = frac(1, 8);   // 0.125 weak separation push (keeps spacing without dispersing)
        c.alignGain        = alignG;
        c.cohGain          = cohG;
        c.perceptionRadius = wu(3);        // 3-unit neighbor radius (the BD2 cell size; > the 2-unit spacing)
        c.maxForce         = wu(8);
        c.maxSpeed         = wu(6);
        c.target           = boids::FxVec3{0, 0, 0};
        c.gravity          = boids::FxVec3{0, 0, 0};
        return c;
    };

    // ===== AccumFlock: the 3 rules point the right way =====
    {
        // Build a tiny neighbor list by hand via the BD2 engine so AccumFlock reads a real slice.
        // agent 0 at origin (zero vel); agent 1 a close +x neighbor MOVING +z with velocity +6.
        std::vector<boids::Agent> ag = {
            boids::Agent{boids::FxVec3{wu(0), 0, wu(0)}, boids::FxVec3{0, 0, 0}},
            boids::Agent{boids::FxVec3{wu(2), 0, wu(0)}, boids::FxVec3{0, 0, wu(6)}},
        };
        const boids::fx radius = wu(4);
        boids::BoidsGrid grid = boids::MakeBoidsGrid(ag, radius);
        boids::BoidsCellTable tbl = boids::BuildBoidsCellTable(ag, grid);
        boids::BoidsNeighborList lst = boids::BuildBoidsNeighborList(ag, grid, tbl, radius);

        // separation only (align=coh=0): agent 0 pushed AWAY from the +x neighbor -> -x force.
        boids::FlockConfig sepOnly = flockCfg3(0, 0);
        boids::FxVec3 fsep = boids::AccumFlock(0, lst, ag, sepOnly);
        check(fsep.x < 0, "AccumFlock: separation pushes -x (away from a +x neighbor)");
        check(fsep.z == 0, "AccumFlock: separation has no z component (pure +x neighbor)");

        // alignment only (sep=coh=0): agent 0 steered toward the neighbor's mean velocity (+z) -> +z force.
        boids::FlockConfig alignOnly = flockCfg3(boids::kOne, 0); alignOnly.sepGain = 0;
        boids::FxVec3 fal = boids::AccumFlock(0, lst, ag, alignOnly);
        check(fal.z > 0, "AccumFlock: alignment steers toward the neighbors' mean velocity (+z)");

        // cohesion only (sep=align=0): agent 0 steered toward the neighbor's mean position (+x) -> +x force.
        boids::FlockConfig cohOnly = flockCfg3(0, boids::kOne); cohOnly.sepGain = 0;
        boids::FxVec3 fco = boids::AccumFlock(0, lst, ag, cohOnly);
        check(fco.x > 0, "AccumFlock: cohesion steers toward the neighbors' mean position (+x)");

        // zero neighbors -> no crash, zero force (an isolated agent far away).
        std::vector<boids::Agent> one = {boids::Agent{boids::FxVec3{wu(0), 0, wu(0)}, boids::FxVec3{0, 0, 0}}};
        boids::BoidsGrid g1 = boids::MakeBoidsGrid(one, radius);
        boids::BoidsCellTable t1 = boids::BuildBoidsCellTable(one, g1);
        boids::BoidsNeighborList n1 = boids::BuildBoidsNeighborList(one, g1, t1, radius);
        boids::FlockConfig full = flockCfg3(boids::kOne, boids::kOne);
        boids::FxVec3 f0 = boids::AccumFlock(0, n1, one, full);
        check(f0.x == 0 && f0.y == 0 && f0.z == 0, "AccumFlock: zero neighbors -> zero force (no div-by-zero)");
    }

    // ===== StepFlock: a spread flock coheres (diagonal shrinks) + aligns (alignment rises), min-sep floored ===
    {
        boids::FlockConfig c = flockCfg3(frac(1, 2), boids::kOne);   // align 0.5, coh 1.0
        const boids::fx dt = boids::kOne / 60;
        // A dense 8x8 flock at 2-unit spacing (within the radius-3 perception -> a connected neighbor graph),
        // each given a small deterministic initial velocity so headings start DISORDERED (then alignment aligns).
        std::vector<boids::Agent> agents;
        for (int gx = 0; gx < 8; ++gx)
            for (int gz = 0; gz < 8; ++gz) {
                // a deterministic per-agent jitter velocity (alternating signs) so initial alignment is LOW.
                const boids::fx vx = ((gx + gz) & 1) ? frac(1, 2) : -frac(1, 2);
                const boids::fx vz = ((gx) & 1) ? -frac(1, 3) : frac(1, 3);
                agents.push_back(boids::Agent{boids::FxVec3{wu(gx * 2), 0, wu(gz * 2)},
                                              boids::FxVec3{vx, 0, vz}});
            }
        boids::FlockStats before = boids::MeasureFlock(agents, c);
        boids::StepFlockSteps(agents, c, dt, 300);
        boids::FlockStats after = boids::MeasureFlock(agents, c);

        // cohesion pulled the flock together: the bounding diagonal SHRANK.
        check(after.diag < before.diag, "StepFlock: bounding diagonal shrinks (cohesion clustered the flock)");
        // alignment aligned the headings: the heading-alignment ROSE.
        check(after.alignment > before.alignment, "StepFlock: heading-alignment rises (alignment aligned them)");
        // separation kept spacing: min separation stays above a floor (didn't collapse to a point).
        check(after.minSep > frac(1, 8), "StepFlock: minSep above a floor (separation kept spacing)");
    }

    // ===== StepFlock control: a sep-only flock (align=coh=0) does NOT cohere as tightly =====
    {
        const boids::fx dt = boids::kOne / 60;
        auto makeFlock = [&]() {
            std::vector<boids::Agent> a;
            for (int gx = 0; gx < 8; ++gx)
                for (int gz = 0; gz < 8; ++gz) {
                    const boids::fx vx = ((gx + gz) & 1) ? frac(1, 2) : -frac(1, 2);
                    const boids::fx vz = ((gx) & 1) ? -frac(1, 3) : frac(1, 3);
                    a.push_back(boids::Agent{boids::FxVec3{wu(gx * 2), 0, wu(gz * 2)},
                                             boids::FxVec3{vx, 0, vz}});
                }
            return a;
        };
        boids::FlockConfig full    = flockCfg3(frac(1, 2), boids::kOne);
        boids::FlockConfig sepOnly = flockCfg3(0, 0);
        std::vector<boids::Agent> agFull = makeFlock(), agSep = makeFlock();
        boids::StepFlockSteps(agFull, full,    dt, 300);
        boids::StepFlockSteps(agSep,  sepOnly, dt, 300);
        boids::FlockStats sFull = boids::MeasureFlock(agFull, full);
        boids::FlockStats sSep  = boids::MeasureFlock(agSep,  sepOnly);
        // the full 3-rule flock coheres TIGHTER (smaller diagonal) than the sep-only control.
        check(sFull.diag < sSep.diag, "StepFlock control: sep-only stays looser (3 rules cohere tighter)");
    }

    // ===== StepFlock two runs byte-identical (determinism) =====
    {
        boids::FlockConfig c = flockCfg3(frac(1, 2), boids::kOne);
        const boids::fx dt = boids::kOne / 60;
        auto makeFlock = [&]() {
            std::vector<boids::Agent> a;
            for (int gx = 0; gx < 6; ++gx)
                for (int gz = 0; gz < 6; ++gz)
                    a.push_back(boids::Agent{boids::FxVec3{wu(gx * 2), 0, wu(gz * 2)},
                                             boids::FxVec3{frac(1, 4), 0, -frac(1, 4)}});
            return a;
        };
        std::vector<boids::Agent> a1 = makeFlock(), a2 = makeFlock();
        boids::StepFlockSteps(a1, c, dt, 200);
        boids::StepFlockSteps(a2, c, dt, 200);
        check(a1.size() == a2.size() &&
              std::memcmp(a1.data(), a2.data(), a1.size() * sizeof(boids::Agent)) == 0,
              "StepFlock: two runs BYTE-IDENTICAL");
    }

    // ===== Slice BD4: PATH-FOLLOWING THE A* CORRIDOR (THE NAV BRIDGE) =====
    // What this PINS (the contracts boids_flock.comp's path term + the GPU==CPU memcmp build on):
    //   * SteerPath: the force points along the corridor toward the NEXT waypoint ahead; an EMPTY path -> 0.
    //   * StepFlockPath with a corridor moves the flock centroid toward the goal while keeping min-separation.
    //   * EMPTY-path StepFlockPath == BD3 StepFlock byte-identical (the render-invariance equivalence contract).
    //   * Two runs byte-identical.
    {
        // A FlockConfig with a path gain (the BD4 showcase shape: gentle flocking + a corridor-follow).
        auto pathCfg = [&](boids::fx pathG) {
            boids::FlockConfig c;
            c.seekGain         = 0;            // free flocking (the corridor is the goal, not a point seek)
            c.sepGain          = frac(1, 8);   // 0.125 weak separation
            c.alignGain        = frac(1, 2);   // 0.5 alignment
            c.cohGain          = frac(1, 2);   // 0.5 cohesion
            c.perceptionRadius = wu(3);
            c.maxForce         = wu(8);
            c.maxSpeed         = wu(6);
            c.target           = boids::FxVec3{0, 0, 0};
            c.gravity          = boids::FxVec3{0, 0, 0};
            c.pathGain         = pathG;
            return c;
        };

        // ----- SteerPath: steers toward the NEXT waypoint ahead -----
        {
            boids::FlockConfig c = pathCfg(boids::kOne);   // pathGain 1.0 so the sign survives
            // a 3-waypoint corridor along +x: (0,0,0)->(10,0,0)->(20,0,0).
            boids::BoidsPath path;
            path.waypoints = {boids::FxVec3{wu(0), 0, 0}, boids::FxVec3{wu(10), 0, 0},
                              boids::FxVec3{wu(20), 0, 0}};
            // an agent AT the first waypoint: nearest is wp0, target is wp1 (+x) -> +x force.
            boids::Agent a0{boids::FxVec3{wu(0), 0, 0}, boids::FxVec3{0, 0, 0}};
            boids::FxVec3 f0 = boids::SteerPath(a0, path, c);
            check(f0.x > 0, "SteerPath: at wp0 steers toward wp1 (+x ahead)");
            check(f0.y == 0 && f0.z == 0, "SteerPath: pure +x corridor -> only x force");
            // an agent near the MIDDLE waypoint: nearest is wp1, target is wp2 (+x ahead) -> +x force.
            boids::Agent a1{boids::FxVec3{wu(10), 0, 0}, boids::FxVec3{0, 0, 0}};
            boids::FxVec3 f1 = boids::SteerPath(a1, path, c);
            check(f1.x > 0, "SteerPath: at wp1 steers toward wp2 (+x ahead, corridor progress)");
            // an agent AT the final waypoint: nearest is wp2, target clamps to wp2 -> zero force (arrived).
            boids::Agent a2{boids::FxVec3{wu(20), 0, 0}, boids::FxVec3{0, 0, 0}};
            boids::FxVec3 f2 = boids::SteerPath(a2, path, c);
            check(f2.x == 0 && f2.y == 0 && f2.z == 0, "SteerPath: at the final waypoint -> zero (arrived)");
            // EMPTY path -> zero force (no crash, the BD3 equivalence).
            boids::BoidsPath empty;
            boids::FxVec3 fe = boids::SteerPath(a0, empty, c);
            check(fe.x == 0 && fe.y == 0 && fe.z == 0, "SteerPath: empty path -> zero force");
            // pathGain 0 -> zero force regardless of the corridor (the render-invariant default).
            boids::FlockConfig c0 = pathCfg(0);
            boids::FxVec3 fz = boids::SteerPath(a0, path, c0);
            check(fz.x == 0 && fz.y == 0 && fz.z == 0, "SteerPath: pathGain 0 -> zero force");
        }

        // ----- StepFlockPath: a flock follows the corridor toward the goal, keeps min-separation -----
        {
            boids::FlockConfig c = pathCfg(frac(1, 2));   // pathGain 0.5
            const boids::fx dt = boids::kOne / 60;
            // a corridor from the spawn area toward +x+z (the goal far from the start).
            boids::BoidsPath path;
            path.waypoints = {boids::FxVec3{wu(0), 0, wu(0)}, boids::FxVec3{wu(20), 0, wu(8)},
                              boids::FxVec3{wu(40), 0, wu(16)}};
            // a dense 8x8 flock spawned near the corridor START (within the radius-3 perception -> connected).
            std::vector<boids::Agent> agents;
            for (int gx = 0; gx < 8; ++gx)
                for (int gz = 0; gz < 8; ++gz)
                    agents.push_back(boids::Agent{boids::FxVec3{wu(gx * 2), 0, wu(gz * 2)},
                                                  boids::FxVec3{0, 0, 0}});
            boids::FlockPathStats before = boids::MeasureFlockPath(agents, c, path);
            boids::StepFlockPathSteps(agents, c, path, dt, 300);
            boids::FlockPathStats after = boids::MeasureFlockPath(agents, c, path);
            // the crowd FOLLOWED the corridor: the flock centroid's L1 distance to the FINAL waypoint DROPPED.
            check(after.centroidToGoal < before.centroidToGoal,
                  "StepFlockPath: centroid->goal drops (the crowd followed the corridor)");
            // it stayed a flock: min separation stays above a small floor (separation keeps spacing even while
            // the whole crowd streams toward the SAME corridor — the agents pack but don't collapse to a point).
            check(after.flock.minSep > 0, "StepFlockPath: minSep above zero (didn't collapse to a point)");
        }

        // ----- EQUIVALENCE: an EMPTY-path StepFlockPath == BD3 StepFlock byte-identical (render-invariance) ---
        {
            boids::FlockConfig c = pathCfg(boids::kOne);   // nonzero pathGain, but an EMPTY path -> path term 0
            const boids::fx dt = boids::kOne / 60;
            auto makeFlock = [&]() {
                std::vector<boids::Agent> a;
                for (int gx = 0; gx < 8; ++gx)
                    for (int gz = 0; gz < 8; ++gz) {
                        const boids::fx vx = ((gx + gz) & 1) ? frac(1, 2) : -frac(1, 2);
                        const boids::fx vz = ((gx) & 1) ? -frac(1, 3) : frac(1, 3);
                        a.push_back(boids::Agent{boids::FxVec3{wu(gx * 2), 0, wu(gz * 2)},
                                                 boids::FxVec3{vx, 0, vz}});
                    }
                return a;
            };
            boids::BoidsPath empty;                         // no waypoints -> SteerPath == 0
            std::vector<boids::Agent> agPath = makeFlock();
            std::vector<boids::Agent> agBD3  = makeFlock();
            boids::StepFlockPathSteps(agPath, c, empty, dt, 200);   // BD4 with an empty corridor
            boids::StepFlockSteps(agBD3, c, dt, 200);               // BD3 StepFlock
            check(agPath.size() == agBD3.size() &&
                  std::memcmp(agPath.data(), agBD3.data(), agPath.size() * sizeof(boids::Agent)) == 0,
                  "StepFlockPath: EMPTY path == BD3 StepFlock BYTE-IDENTICAL (render-invariance)");
        }

        // ----- two runs byte-identical (determinism) -----
        {
            boids::FlockConfig c = pathCfg(frac(1, 2));
            const boids::fx dt = boids::kOne / 60;
            boids::BoidsPath path;
            path.waypoints = {boids::FxVec3{wu(0), 0, wu(0)}, boids::FxVec3{wu(20), 0, wu(10)},
                              boids::FxVec3{wu(40), 0, wu(20)}};
            auto makeFlock = [&]() {
                std::vector<boids::Agent> a;
                for (int gx = 0; gx < 6; ++gx)
                    for (int gz = 0; gz < 6; ++gz)
                        a.push_back(boids::Agent{boids::FxVec3{wu(gx * 2), 0, wu(gz * 2)},
                                                 boids::FxVec3{frac(1, 4), 0, -frac(1, 4)}});
                return a;
            };
            std::vector<boids::Agent> a1 = makeFlock(), a2 = makeFlock();
            boids::StepFlockPathSteps(a1, c, path, dt, 200);
            boids::StepFlockPathSteps(a2, c, path, dt, 200);
            check(a1.size() == a2.size() &&
                  std::memcmp(a1.data(), a2.data(), a1.size() * sizeof(boids::Agent)) == 0,
                  "StepFlockPath: two runs BYTE-IDENTICAL");
        }
    }

    // ===== Slice BD5: LOCKSTEP + ROLLBACK (the netcode headline; PURE CPU) =====
    // What this PINS (the contracts the --boids-lockstep showcase + the cross-platform golden build on):
    //   * SnapshotBoids/RestoreBoids: a deep-copy round-trip — snapshot -> mutate (a few perturb+step ticks)
    //     -> restore -> the agent vector is bit-identical to the snapshot.
    //   * ApplyBoidsCommand: kicks the target agent's velocity by dv (FxAdd); out-of-range agent -> no-op.
    //   * RunBoidsLockstep: an authority + a replica fed the SAME perturbation stream stay bit-identical.
    //   * RunBoidsRollback: a mispredicted kick DIVERGES (mispredicted != authority before rollback), then the
    //     rollback corrects to authority bit-for-bit.
    //   * Two runs byte-identical (determinism).
    {
        // The BD4 path-following flock recipe (a small flock + a short corridor so the test is fast).
        auto lockCfg = [&]() {
            boids::FlockConfig c;
            c.seekGain         = 0;
            c.sepGain          = frac(1, 8);   // 0.125 weak separation
            c.alignGain        = frac(1, 2);   // 0.5 alignment
            c.cohGain          = frac(1, 2);   // 0.5 cohesion
            c.perceptionRadius = wu(3);
            c.maxForce         = wu(8);
            c.maxSpeed         = wu(6);
            c.target           = boids::FxVec3{0, 0, 0};
            c.gravity          = boids::FxVec3{0, 0, 0};
            c.pathGain         = frac(1, 4);   // 0.25 corridor-follow
            return c;
        };
        const boids::FlockConfig cfg = lockCfg();
        const boids::fx dt = boids::kOne / 60;
        boids::BoidsPath path;
        path.waypoints = {boids::FxVec3{wu(0), 0, wu(0)}, boids::FxVec3{wu(20), 0, wu(8)},
                          boids::FxVec3{wu(40), 0, wu(16)}};
        // a 6x6 = 36-agent flock at the corridor start.
        auto makeFlock = [&]() {
            std::vector<boids::Agent> a;
            for (int gx = 0; gx < 6; ++gx)
                for (int gz = 0; gz < 6; ++gz)
                    a.push_back(boids::Agent{boids::FxVec3{wu(gx), 0, wu(gz)}, boids::FxVec3{0, 0, 0}});
            return a;
        };
        const std::vector<boids::Agent> flock0 = makeFlock();
        const int kTicks = 80;

        // ----- ApplyBoidsCommand: kicks the target agent's velocity; out-of-range -> no-op -----
        {
            std::vector<boids::Agent> a = makeFlock();
            const boids::FxVec3 v0 = a[5].vel;
            boids::ApplyBoidsCommand(a, boids::BoidsCommand{0u, 5u, boids::FxVec3{wu(3), 0, -wu(2)}});
            check(a[5].vel.x == v0.x + wu(3) && a[5].vel.z == v0.z - wu(2),
                  "ApplyBoidsCommand: kicks the target agent's velocity by dv");
            // out-of-range agent -> no-op (no crash, no mutation).
            std::vector<boids::Agent> b = makeFlock();
            const std::vector<boids::Agent> bRef = makeFlock();
            boids::ApplyBoidsCommand(b, boids::BoidsCommand{0u, 9999u, boids::FxVec3{wu(5), 0, 0}});
            check(b.size() == bRef.size() &&
                  std::memcmp(b.data(), bRef.data(), b.size() * sizeof(boids::Agent)) == 0,
                  "ApplyBoidsCommand: out-of-range agent -> no-op");
        }

        // ----- SnapshotBoids/RestoreBoids: deep-copy round-trip -----
        {
            std::vector<boids::Agent> a = flock0;
            // advance a few ticks so the world is non-trivial.
            std::vector<boids::BoidsCommand> empty;
            for (int t = 0; t < 5; ++t) boids::SimBoidsTick(a, cfg, path, empty, t, dt);
            const boids::BoidsSnapshot snap = boids::SnapshotBoids(a);
            // mutate: a few perturb + step ticks.
            std::vector<boids::BoidsCommand> kicks = {{5u, 3u, boids::FxVec3{wu(4), 0, 0}}};
            for (int t = 5; t < 12; ++t) boids::SimBoidsTick(a, cfg, path, kicks, t, dt);
            check(a.size() == snap.agents.size() &&
                  std::memcmp(a.data(), snap.agents.data(), a.size() * sizeof(boids::Agent)) != 0,
                  "SnapshotBoids: the world changed after the snapshot (sanity)");
            // restore -> bit-identical to the snapshot.
            boids::RestoreBoids(a, snap);
            check(a.size() == snap.agents.size() &&
                  std::memcmp(a.data(), snap.agents.data(), a.size() * sizeof(boids::Agent)) == 0,
                  "RestoreBoids: round-trip restores the snapshot BYTE-IDENTICAL");
        }

        // The scripted perturbation stream (a few velocity kicks scattering parts of the flock).
        const std::vector<boids::BoidsCommand> authStream = {
            {2u,  0u, boids::FxVec3{ wu(6), 0,  wu(4)}},
            {2u, 12u, boids::FxVec3{-wu(5), 0, -wu(3)}},
            {20u, 7u, boids::FxVec3{ wu(4), 0, -wu(5)}},
        };

        // ----- RunBoidsLockstep: authority == replica over the perturbation stream -----
        {
            std::vector<boids::Agent> authority =
                boids::RunBoidsLockstep(cfg, path, flock0, authStream, kTicks, dt);
            std::vector<boids::Agent> replica =
                boids::RunBoidsLockstep(cfg, path, flock0, authStream, kTicks, dt);
            check(authority.size() == replica.size() &&
                  std::memcmp(authority.data(), replica.data(), authority.size() * sizeof(boids::Agent)) == 0,
                  "RunBoidsLockstep: authority == replica BIT-IDENTICAL (inputs-only re-derivation)");
        }

        // ----- RunBoidsRollback: a mispredicted kick diverges, the rollback corrects to authority -----
        {
            const int divergeTick = 30;
            // a WRONG perturbation at divergeTick (a different agent + a strong-enough kick to truly diverge).
            const std::vector<boids::BoidsCommand> mispredictStream = {
                {30u, 20u, boids::FxVec3{wu(6), 0, wu(6)}},
                {31u, 25u, boids::FxVec3{-wu(6), 0, -wu(6)}},
            };
            std::vector<boids::Agent> authority =
                boids::RunBoidsLockstep(cfg, path, flock0, authStream, kTicks, dt);

            // the speculative (pre-rollback) state: advance to divergeTick with authStream, then a few ticks
            // with the WRONG stream -> it must DIVERGE from authority (a real divergence to fix).
            std::vector<boids::Agent> mispredicted = flock0;
            for (int t = 0; t < divergeTick; ++t) boids::SimBoidsTick(mispredicted, cfg, path, authStream, t, dt);
            for (int t = divergeTick; t < divergeTick + 3; ++t)
                boids::SimBoidsTick(mispredicted, cfg, path, mispredictStream, t, dt);
            // compare to authority at the SAME tick count (advance a fresh authority copy to divergeTick+3).
            std::vector<boids::Agent> authAtSpec = flock0;
            for (int t = 0; t < divergeTick + 3; ++t) boids::SimBoidsTick(authAtSpec, cfg, path, authStream, t, dt);
            check(mispredicted.size() == authAtSpec.size() &&
                  std::memcmp(mispredicted.data(), authAtSpec.data(),
                              mispredicted.size() * sizeof(boids::Agent)) != 0,
                  "RunBoidsRollback: mispredicted DIVERGED from authority before rollback (real divergence)");

            // the rollback corrects to authority bit-for-bit.
            std::vector<boids::Agent> corrected =
                boids::RunBoidsRollback(cfg, path, flock0, authStream, mispredictStream, divergeTick, kTicks, dt);
            check(corrected.size() == authority.size() &&
                  std::memcmp(corrected.data(), authority.data(),
                              corrected.size() * sizeof(boids::Agent)) == 0,
                  "RunBoidsRollback: corrected == authority BIT-EXACT (rollback fixed the misprediction)");
        }

        // ----- two runs byte-identical (determinism) -----
        {
            std::vector<boids::Agent> r1 =
                boids::RunBoidsLockstep(cfg, path, flock0, authStream, kTicks, dt);
            std::vector<boids::Agent> r2 =
                boids::RunBoidsLockstep(cfg, path, flock0, authStream, kTicks, dt);
            check(r1.size() == r2.size() &&
                  std::memcmp(r1.data(), r2.data(), r1.size() * sizeof(boids::Agent)) == 0,
                  "RunBoidsLockstep: two runs BYTE-IDENTICAL");
        }
    }

    if (g_fail == 0) std::printf("boids_test: ALL PASS\n");
    else             std::printf("boids_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
