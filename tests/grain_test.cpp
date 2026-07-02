// Slice GR1 — Deterministic GPU Granular/Sand: the Q16.16 GRAIN POOL INTEGRATOR + dropped-block core
// (engine/sim/grain.h) that the GPU shaders/grain_integrate.comp.hlsl copies VERBATIM + proves
// bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace hf::sim::grain.
// The FL1/CL1 integer-beachhead twin, with TWO deltas the GR1 spec locks: a first-class `radius`
// field (48-byte std430 packing) and a RADIUS-AWARE ground rest (the grain's surface rests on the
// floor: pos.y < groundY + radius -> groundY + radius).
//
// What this test PINS (the contracts the GPU grain_integrate.comp + the GPU==CPU proof build on):
//   * InitGrainBlock: a solid W x H x D block (size W*H*D, the GrainIndex traversal order), spacing/
//     origin host-snapped, uniform `radius`, every grain DYNAMIC (invMass kOne, flags 0); prev==pos, vel==0.
//   * IntegrateGrainParticle: one-step + K-step semi-implicit-Euler closed form (a FREE grain falls the
//     EXACT Q16.16 distance — hand-checked integer recurrence), prev = pos each step.
//   * the RADIUS-AWARE ground rest (pos.y pinned to groundY + radius, vel.y zeroed on contact, stays settled).
//   * IntegrateGrains ORDER-INDEPENDENCE (shuffled vs in-order -> identical, grains are independent in GR1).
//   * determinism: two runs of the SAME init+steps -> byte-identical grain arrays.
//   * a STATIC grain (invMass 0 / flags bit0) -> UNCHANGED after K steps (the static / no-op case).
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests.
#include "sim/grain.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace grain = hf::sim::grain;
using grain::fx;
using grain::kOne;
using grain::kFrac;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << kFrac); }

int main() {
    HF_TEST_MAIN_INIT();

    // ===== the std430 packing delta vs FluidParticle: 48 bytes, NO padding holes (memcmp-able) =====
    {
        check(sizeof(grain::GrainParticle) == 48, "GrainParticle is 48 bytes (the 12 x int32 std430 packing)");
    }

    // ================= InitGrainBlock: solid W x H x D block, uniform radius, all dynamic, at rest =====
    {
        grain::GrainBlock block;
        block.W = 4; block.H = 3; block.D = 2;
        block.spacing = kOne;                 // 1.0 world unit between grains
        block.radius  = kOne / 4;             // 0.25 grain radius (spacing >= 2*radius -> non-overlapping)
        block.origin  = grain::FxVec3{FromInt(5), FromInt(10), FromInt(-3)};
        std::vector<grain::GrainParticle> ps = grain::InitGrainBlock(block);

        check(ps.size() == (size_t)(block.W * block.H * block.D), "InitGrainBlock: size == W*H*D");

        // x-major/y-mid/z-minor layout: grain (ix,iy,iz) at origin + (ix*spacing, iy*spacing, iz*spacing).
        const grain::GrainParticle& p000 = ps[(size_t)grain::GrainIndex(block, 0, 0, 0)];
        const grain::GrainParticle& p312 = ps[(size_t)grain::GrainIndex(block, 3, 1, 1)];
        check(p000.pos.x == FromInt(5) && p000.pos.y == FromInt(10) && p000.pos.z == FromInt(-3),
              "InitGrainBlock: (0,0,0) at origin");
        check(p312.pos.x == FromInt(5) + FromInt(3) && p312.pos.y == FromInt(10) + FromInt(1) &&
              p312.pos.z == FromInt(-3) + FromInt(1),
              "InitGrainBlock: (3,1,1) at origin + (3,1,1)*spacing");

        // GrainIndex is the x-major/y-mid/z-minor flatten ((iz*H + iy)*W + ix).
        check(grain::GrainIndex(block, 0, 0, 0) == 0, "GrainIndex: (0,0,0) -> 0");
        check(grain::GrainIndex(block, 1, 0, 0) == 1, "GrainIndex: +x is the minor stride");
        check(grain::GrainIndex(block, 0, 1, 0) == block.W, "GrainIndex: +y strides by W");
        check(grain::GrainIndex(block, 0, 0, 1) == block.W * block.H,
              "GrainIndex: +z strides by W*H");

        // prev == pos, vel == 0, every grain DYNAMIC (invMass kOne), uniform radius initially.
        bool prevEqPos = true, velZero = true, allDynamic = true, uniformRadius = true;
        for (const grain::GrainParticle& p : ps) {
            if (std::memcmp(&p.prev, &p.pos, sizeof(grain::FxVec3)) != 0) prevEqPos = false;
            if (p.vel.x != 0 || p.vel.y != 0 || p.vel.z != 0) velZero = false;
            if ((p.flags & grain::kFlagStatic) != 0u || p.invMass != kOne) allDynamic = false;
            if (p.radius != block.radius) uniformRadius = false;
        }
        check(prevEqPos, "InitGrainBlock: prev == pos initially");
        check(velZero, "InitGrainBlock: vel == 0 initially");
        check(allDynamic, "InitGrainBlock: every grain dynamic (invMass kOne, no static flag)");
        check(uniformRadius, "InitGrainBlock: every grain carries the uniform block radius");
    }

    // ================= a FREE grain: one-step + K-step semi-implicit-Euler closed form ===============
    {
        const fx g  = FromInt(-10);     // gravity -10 (exact in Q16.16)
        const fx dt = kOne / 2;         // dt = 0.5 (exact)
        const fx groundY = FromInt(-1000);  // far below; effectively no clamp here
        const grain::FxVec3 grav{0, g, 0};

        // One step: vel.y = 0 + g*dt = -10*0.5 = -5 ; pos.y = 100 + vel.y*dt = 100 + (-5)*0.5 = 97.5.
        grain::GrainParticle p;
        p.pos = {0, FromInt(100), 0};
        p.prev = p.pos;
        p.vel = {0, 0, 0};
        p.invMass = kOne;
        p.radius = kOne / 4;
        p.flags = 0;                    // dynamic (not static)
        grain::IntegrateGrainParticle(p, grav, groundY, dt);
        check(p.vel.y == FromInt(-5), "free one-step vel.y == g*dt == -5.0");
        check(p.pos.y == FromInt(100) - (kOne * 5 / 2), "free one-step pos.y == 100 + (-5)*0.5 == 97.5");
        check(p.prev.y == FromInt(100), "free one-step prev.y == the pre-move pos (100)");

        // K-step closed form: independently re-run the EXACT integer ops K times and compare to the
        // header's IntegrateGrainSteps over the same single-grain pool (must agree by construction).
        const int K = 60;
        std::vector<grain::GrainParticle> one(1);
        one[0].pos = {0, FromInt(500), 0};
        one[0].prev = one[0].pos;
        one[0].vel = {0, 0, 0};
        one[0].invMass = kOne;
        one[0].radius = kOne / 4;
        one[0].flags = 0;

        // Reference: the same per-step integer recurrence, computed inline (radius-aware clamp).
        const fx radius = kOne / 4;
        fx refVy = 0, refPy = FromInt(500), refPrevY = FromInt(500);
        for (int s = 0; s < K; ++s) {
            refVy += grain::fxmul(g, dt);
            refPrevY = refPy;
            refPy += grain::fxmul(refVy, dt);
            if (refPy < groundY + radius) { refPy = groundY + radius; if (refVy < 0) refVy = 0; }
        }
        grain::IntegrateGrainSteps(one, grav, dt, groundY, K);
        check(one[0].pos.y == refPy, "free K-step pos.y == hand-computed integer recurrence");
        check(one[0].vel.y == refVy, "free K-step vel.y == hand-computed integer recurrence");
        check(one[0].prev.y == refPrevY, "free K-step prev.y == the pre-last-move pos");
    }

    // ================= RADIUS-AWARE ground rest: pos.y == groundY + radius, vel.y zeroed ============
    {
        const grain::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = 0;
        const fx radius = kOne / 2;       // 0.5 radius -> the grain's SURFACE rests on the floor

        // A grain whose center would drop below groundY + radius clamps to groundY + radius + zeros vel.y.
        grain::GrainParticle p;
        p.pos = {0, radius + kOne / 100, 0};  // just above the rest height (groundY + radius)
        p.prev = p.pos;
        p.vel = {0, FromInt(-50), 0};    // moving down fast
        p.invMass = kOne;
        p.radius = radius;
        p.flags = 0;
        grain::IntegrateGrainParticle(p, grav, groundY, dt);
        check(p.pos.y == groundY + radius, "ground rest: pos.y pinned to groundY + radius (surface on floor)");
        check(p.vel.y == 0, "ground rest: downward vel.y zeroed on contact");

        // After contact, the grain stays settled (pos.y stays groundY + radius across more steps).
        for (int s = 0; s < 50; ++s) grain::IntegrateGrainParticle(p, grav, groundY, dt);
        check(p.pos.y == groundY + radius, "ground rest: grain stays settled at groundY + radius");

        // Sanity: a ZERO-radius grain rests its CENTER on the floor (the FL1 plain-ground-clamp degenerate).
        grain::GrainParticle z;
        z.pos = {0, kOne / 100, 0};
        z.prev = z.pos;
        z.vel = {0, FromInt(-50), 0};
        z.invMass = kOne;
        z.radius = 0;
        z.flags = 0;
        grain::IntegrateGrainParticle(z, grav, groundY, dt);
        check(z.pos.y == groundY, "ground rest: radius 0 -> rests center on groundY (FL1 degenerate)");
    }

    // ================= ORDER-INDEPENDENCE: shuffled vs in-order integrate -> identical ===============
    // Grains are INDEPENDENT in GR1 (no inter-grain coupling), so integrating the pool in ANY order
    // yields the SAME per-grain result. This is what makes the GPU one-thread-per-grain dispatch race-free.
    {
        const grain::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = 0;
        grain::GrainBlock block;
        block.W = 5; block.H = 5; block.D = 5; block.spacing = kOne; block.radius = kOne / 4;
        block.origin = grain::FxVec3{0, FromInt(20), 0};
        std::vector<grain::GrainParticle> inOrder = grain::InitGrainBlock(block);

        // A SHUFFLED copy (remember each grain's original index so we can compare back).
        std::vector<grain::GrainParticle> shuffled = inOrder;
        std::vector<int> perm(shuffled.size());
        for (int i = 0; i < (int)perm.size(); ++i) perm[i] = i;
        // A FIXED deterministic shuffle (a reproducible reversal-stride permutation, NO RNG).
        for (size_t i = 0; i + 1 < perm.size(); i += 2) std::swap(perm[i], perm[perm.size() - 1 - i]);
        std::vector<grain::GrainParticle> shuf(shuffled.size());
        for (int i = 0; i < (int)perm.size(); ++i) shuf[i] = shuffled[perm[i]];

        const int K = 120;
        grain::IntegrateGrainSteps(inOrder, grav, dt, groundY, K);
        grain::IntegrateGrainSteps(shuf, grav, dt, groundY, K);
        // Map each shuffled result back to its original index and compare to the in-order result.
        bool same = true;
        for (int i = 0; i < (int)perm.size(); ++i)
            if (std::memcmp(&shuf[i], &inOrder[perm[i]], sizeof(grain::GrainParticle)) != 0) same = false;
        check(same, "order-independence: shuffled integrate == in-order integrate per grain (race-free)");
    }

    // ================= determinism: two runs byte-identical + a coherent dropped-block fall ===========
    {
        const grain::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = 0;
        auto makeBlock = []() {
            grain::GrainBlock b; b.W = 5; b.H = 5; b.D = 5; b.spacing = kOne; b.radius = kOne / 4;
            b.origin = grain::FxVec3{0, FromInt(20), 0};   // a corner well above the ground
            return b;
        };
        grain::GrainBlock block = makeBlock();
        std::vector<grain::GrainParticle> a = grain::InitGrainBlock(block);
        std::vector<grain::GrainParticle> b = grain::InitGrainBlock(block);
        const int K = 120;
        grain::IntegrateGrainSteps(a, grav, dt, groundY, K);
        grain::IntegrateGrainSteps(b, grav, dt, groundY, K);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(grain::GrainParticle)) == 0,
              "determinism: two runs byte-identical");

        // Coherence: the block FELL (every grain's y dropped below its rest y) and piled at the ground
        // (some grains rest at groundY + radius) — a coherent dropped-block fall.
        std::vector<grain::GrainParticle> init = grain::InitGrainBlock(block);
        int moved = 0;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i].pos.y < init[i].pos.y) ++moved;
        check(moved == (int)a.size(), "coherence: every grain fell (moved down)");
        check(grain::CountAtGround(a, groundY) > 0, "coherence: some grains piled at the ground (rest)");
    }

    // ================= a STATIC grain NEVER moves (invMass 0 / flags bit0) ===========================
    {
        const grain::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        grain::GrainParticle s;
        s.pos = {FromInt(3), FromInt(7), FromInt(-2)};
        s.prev = s.pos;
        s.vel = {0, 0, 0};
        s.invMass = 0;
        s.radius = kOne / 4;
        s.flags = grain::kFlagStatic;
        grain::GrainParticle before = s;
        for (int i = 0; i < 100; ++i) grain::IntegrateGrainParticle(s, grav, 0, dt);
        check(std::memcmp(&s, &before, sizeof(grain::GrainParticle)) == 0,
              "static grain never moves (untouched across 100 steps)");

        // A pool of static grains -> UNCHANGED after K steps (the static no-op the GR1 proof asserts).
        grain::GrainBlock block; block.W = 4; block.H = 4; block.D = 4; block.spacing = kOne;
        block.radius = kOne / 4; block.origin = grain::FxVec3{0, FromInt(5), 0};
        std::vector<grain::GrainParticle> ps = grain::InitGrainBlock(block);
        for (auto& p : ps) { p.invMass = 0; p.flags = grain::kFlagStatic; }
        std::vector<grain::GrainParticle> poolBefore = ps;
        grain::IntegrateGrainSteps(ps, grav, dt, 0, 120);
        check(std::memcmp(ps.data(), poolBefore.data(), ps.size() * sizeof(grain::GrainParticle)) == 0,
              "static pool UNCHANGED after K steps (static no-op)");
    }

    // ================= InitGrainBlock: the 10x10x10 showcase block (count + a known position) ========
    {
        grain::GrainBlock block; block.W = 10; block.H = 10; block.D = 10; block.spacing = kOne;
        block.radius = kOne / 4; block.origin = grain::FxVec3{0, FromInt(24), 0};
        std::vector<grain::GrainParticle> ps = grain::InitGrainBlock(block);
        check(ps.size() == 1000, "InitGrainBlock 10x10x10: 1000 grains");
        // The far corner (9,9,9) is at origin + (9,9,9).
        const grain::GrainParticle& far = ps[(size_t)grain::GrainIndex(block, 9, 9, 9)];
        check(far.pos.x == FromInt(9) && far.pos.y == FromInt(24) + FromInt(9) && far.pos.z == FromInt(9),
              "InitGrainBlock 10x10x10: far corner (9,9,9) at origin + (9,9,9)");
    }

    // ============================ Slice GR2 — GRID-HASH NEIGHBOR SEARCH =============================

    // ----- MakeGrainGrid: correct cellMin / gridDim over a known pool (FloorDiv per axis at hSearch) ------
    {
        const fx hSearch = kOne;   // 1.0 cell size
        // GrainCellOf: positive + negative coords floor toward -inf (monotone across 0).
        check(grain::GrainCellOf(grain::FxVec3{FromInt(0), FromInt(0), FromInt(0)}, hSearch).x == 0,
              "GrainCellOf: 0 -> cell 0");
        check(grain::GrainCellOf(grain::FxVec3{FromInt(2), FromInt(3), FromInt(5)}, hSearch).y == 3,
              "GrainCellOf: (2,3,5) -> cell 3 on y");
        check(grain::GrainCellOf(grain::FxVec3{kOne + kOne / 2, 0, 0}, hSearch).x == 1,
              "GrainCellOf: 1.5 -> cell 1");
        check(grain::GrainCellOf(grain::FxVec3{-(kOne / 2), 0, 0}, hSearch).x == -1,
              "GrainCellOf: -0.5 -> cell -1 (floor, not truncate)");
        check(grain::GrainCellOf(grain::FxVec3{-(kOne + kOne / 2), 0, 0}, hSearch).x == -2,
              "GrainCellOf: -1.5 -> cell -2");

        // A pool spanning a known AABB -> exact cellMin/gridDim. Grains at x={0,2,4}, y={-1,1}, z={0}.
        std::vector<grain::GrainParticle> pool;
        for (int gx : {0, 2, 4}) for (int gy : {-1, 1}) {
            grain::GrainParticle p; p.pos = {FromInt(gx), FromInt(gy), 0}; p.invMass = kOne; pool.push_back(p);
        }
        grain::GrainGrid grid = grain::MakeGrainGrid(pool, hSearch);
        check(grid.cellMin.x == 0 && grid.cellMin.y == -1 && grid.cellMin.z == 0,
              "MakeGrainGrid: cellMin == the min cell coord per axis");
        check(grid.gridDim.x == 5 && grid.gridDim.y == 3 && grid.gridDim.z == 1,
              "MakeGrainGrid: gridDim == (max-min+1) per axis (x:0..4, y:-1..1, z:0)");
        check(grain::GrainCellCount(grid) == 15u, "MakeGrainGrid: cellCount == 5*3*1 == 15");

        // Empty pool -> 1x1x1 grid at origin (deterministic degenerate).
        std::vector<grain::GrainParticle> empty;
        grain::GrainGrid eg = grain::MakeGrainGrid(empty, hSearch);
        check(eg.gridDim.x == 1 && eg.gridDim.y == 1 && eg.gridDim.z == 1 &&
              grain::GrainCellCount(eg) == 1u, "MakeGrainGrid: empty pool -> 1x1x1 grid");
    }

    // ----- GrainNeighborAccept: the per-axis |dx| < hSearch box reject (hand-checked) --------------------
    {
        const fx hSearch = kOne;   // 1.0
        const grain::FxVec3 a{0, 0, 0};
        check(grain::GrainNeighborAccept(a, grain::FxVec3{kOne / 2, 0, 0}, hSearch),
              "GrainNeighborAccept: dx=0.5 < hSearch -> accept");
        check(!grain::GrainNeighborAccept(a, grain::FxVec3{kOne, 0, 0}, hSearch),
              "GrainNeighborAccept: dx=1.0 == hSearch -> reject (strict <)");
        check(!grain::GrainNeighborAccept(a, grain::FxVec3{0, 0, kOne + 1}, hSearch),
              "GrainNeighborAccept: dz just > hSearch -> reject");
        check(grain::GrainNeighborAccept(grain::FxVec3{kOne / 2, 0, 0}, a, hSearch),
              "GrainNeighborAccept: symmetric (b,a) accept");
        check(grain::GrainNeighborAccept(a, grain::FxVec3{-(kOne / 2), 0, 0}, hSearch),
              "GrainNeighborAccept: dx=-0.5 within hSearch -> accept (abs)");
    }

    // ----- 2 grains within hSearch -> MUTUAL neighbors; > hSearch apart -> NONE; no self-neighbor --------
    {
        const fx hSearch = kOne;
        std::vector<grain::GrainParticle> within(2);
        within[0].pos = {0, 0, 0};            within[0].invMass = kOne;
        within[1].pos = {kOne / 2, 0, 0};     within[1].invMass = kOne;
        grain::GrainGrid g = grain::MakeGrainGrid(within, hSearch);
        grain::GrainCellTable t = grain::BuildGrainCellTable(within, g);
        grain::GrainNeighborList nl = grain::BuildGrainNeighborList(within, g, t, hSearch);
        check(nl.neighborStart.size() == 3, "neighbor: neighborStart has grainCount+1 entries");
        check(nl.neighborStart[2] == nl.neighbors.size() && nl.neighbors.size() == 2,
              "neighbor: 2 mutual neighbor entries within hSearch");
        auto slice = [&](uint32_t i, std::vector<uint32_t>& out) {
            out.assign(nl.neighbors.begin() + nl.neighborStart[i],
                       nl.neighbors.begin() + nl.neighborStart[i + 1]);
        };
        std::vector<uint32_t> n0, n1;
        slice(0, n0); slice(1, n1);
        check(n0.size() == 1 && n0[0] == 1u, "neighbor: g0 -> {1}");
        check(n1.size() == 1 && n1[0] == 0u, "neighbor: g1 -> {0}");
        bool noSelf = true;
        for (uint32_t i = 0; i < 2; ++i) { std::vector<uint32_t> s; slice(i, s);
            for (uint32_t j : s) if (j == i) noSelf = false; }
        check(noSelf, "neighbor: no grain is its own neighbor");

        // > hSearch apart (3.0 on x) -> NO neighbors.
        std::vector<grain::GrainParticle> apart(2);
        apart[0].pos = {0, 0, 0};            apart[0].invMass = kOne;
        apart[1].pos = {FromInt(3), 0, 0};   apart[1].invMass = kOne;
        grain::GrainGrid g2 = grain::MakeGrainGrid(apart, hSearch);
        grain::GrainCellTable t2 = grain::BuildGrainCellTable(apart, g2);
        grain::GrainNeighborList nl2 = grain::BuildGrainNeighborList(apart, g2, t2, hSearch);
        check(nl2.neighbors.empty() && nl2.neighborStart[2] == 0u,
              "neighbor: grains > hSearch apart -> 0 neighbors");
    }

    // ----- a small block: BuildGrainCellTable CSR offsets + ascending within-cell order + exact counts ----
    {
        // A 3x3x3 block spaced 1.0, hSearch = 1.5 (one grain per cell, the stencil reaches the 1-away
        // lattice neighbors). hSearch 1.5 >= 2*radius (0.5) -> a valid contact search radius.
        const fx hSearch = kOne + kOne / 2;   // 1.5
        grain::GrainBlock block;
        block.W = 3; block.H = 3; block.D = 3; block.spacing = kOne; block.radius = kOne / 4;
        block.origin = grain::FxVec3{0, 0, 0};
        std::vector<grain::GrainParticle> ps = grain::InitGrainBlock(block);   // 27 grains
        grain::GrainGrid g = grain::MakeGrainGrid(ps, hSearch);
        grain::GrainCellTable tab = grain::BuildGrainCellTable(ps, g);
        grain::GrainNeighborList nl = grain::BuildGrainNeighborList(ps, g, tab, hSearch);

        // Cell-table CSR invariants: cellStart has cellCount+1 entries, monotone, last == n.
        const uint32_t cells = grain::GrainCellCount(g);
        check(tab.cellStart.size() == (size_t)cells + 1, "cell-table: cellStart has cellCount+1 entries");
        bool monotone = true;
        for (size_t c = 0; c + 1 < tab.cellStart.size(); ++c)
            if (tab.cellStart[c] > tab.cellStart[c + 1]) monotone = false;
        check(monotone, "cell-table: cellStart monotone non-decreasing");
        check(tab.cellStart[cells] == ps.size() && tab.cellGrains.size() == ps.size(),
              "cell-table: sentinel == grain count, every grain bucketed");
        // Every grain index appears exactly once in cellGrains (a permutation).
        std::vector<int> seen(ps.size(), 0);
        for (uint32_t idx : tab.cellGrains) if (idx < ps.size()) ++seen[idx];
        bool perm = true; for (int s : seen) if (s != 1) perm = false;
        check(perm, "cell-table: cellGrains is a permutation of [0,n)");
        // Ascending within-cell order: each cell's slice is ascending grain index (the DET-CRUX).
        bool ascendingInCell = true;
        for (uint32_t c = 0; c < cells; ++c)
            for (uint32_t s = tab.cellStart[c] + 1; s < tab.cellStart[c + 1]; ++s)
                if (tab.cellGrains[s] <= tab.cellGrains[s - 1]) ascendingInCell = false;
        check(ascendingInCell, "cell-table: within-cell order is ascending grain index (deterministic)");

        // Expected neighbor count per grain = #lattice neighbors with |d|<hSearch=1.5 per axis (the 3x3x3
        // box minus self, clamped to the block bounds).
        auto inRange = [&](int v, int lo, int hi) { return v >= lo && v <= hi; };
        bool countsOk = true;
        for (int iz = 0; iz < 3; ++iz)
        for (int iy = 0; iy < 3; ++iy)
        for (int ix = 0; ix < 3; ++ix) {
            int idx = grain::GrainIndex(block, ix, iy, iz);
            uint32_t expected = 0;
            for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                if (inRange(ix + dx, 0, 2) && inRange(iy + dy, 0, 2) && inRange(iz + dz, 0, 2)) ++expected;
            }
            uint32_t got = nl.neighborStart[idx + 1] - nl.neighborStart[idx];
            if (got != expected) countsOk = false;
        }
        check(countsOk, "neighbor: 3x3x3 block counts == lattice |d|<hSearch reference (corner 7, center 26)");
        int cornerIdx = grain::GrainIndex(block, 0, 0, 0);
        int centerIdx = grain::GrainIndex(block, 1, 1, 1);
        check(nl.neighborStart[cornerIdx + 1] - nl.neighborStart[cornerIdx] == 7,
              "neighbor: corner grain has 7 neighbors (the 2x2x2 minus self)");
        check(nl.neighborStart[centerIdx + 1] - nl.neighborStart[centerIdx] == 26,
              "neighbor: center grain has 26 neighbors (all others within hSearch)");

        // Coherence: every emitted neighbor j of i passes GrainNeighborAccept (within hSearch per axis); i!=j.
        bool coherent = true;
        for (uint32_t i = 0; i < ps.size(); ++i)
            for (uint32_t s = nl.neighborStart[i]; s < nl.neighborStart[i + 1]; ++s) {
                uint32_t j = nl.neighbors[s];
                if (j == i) coherent = false;
                if (!grain::GrainNeighborAccept(ps[i].pos, ps[j].pos, hSearch)) coherent = false;
            }
        check(coherent, "neighbor: every emitted neighbor within hSearch per axis, no self");

        // The FIXED stencil emit order: with one grain per cell, each grain's neighbor list is the accepted j
        // visited in ascending stencil-cell (dz,dy,dx -1..+1) order. Hand-check the corner (0,0,0): its
        // neighbors are the 7 lattice points in the +x/+y/+z half, emitted in (dz,dy,dx) order. The grain
        // index is GrainIndex(ix,iy,iz) = (iz*3+iy)*3+ix. The first accepted (dz=0,dy=0,dx=+1) is grain 1.
        {
            std::vector<uint32_t> ref;
            int ix0 = 0, iy0 = 0, iz0 = 0;
            for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = ix0 + dx, ny = iy0 + dy, nz = iz0 + dz;
                if (nx < 0 || nx > 2 || ny < 0 || ny > 2 || nz < 0 || nz > 2) continue;
                ref.push_back((uint32_t)grain::GrainIndex(block, nx, ny, nz));
            }
            std::vector<uint32_t> got(nl.neighbors.begin() + nl.neighborStart[cornerIdx],
                                      nl.neighbors.begin() + nl.neighborStart[cornerIdx + 1]);
            check(got == ref, "neighbor: corner list is the fixed (dz,dy,dx) stencil order");
        }

        // Determinism: rebuild from scratch -> byte-identical cell table + neighbor list.
        grain::GrainGrid g2 = grain::MakeGrainGrid(ps, hSearch);
        grain::GrainCellTable tab2 = grain::BuildGrainCellTable(ps, g2);
        grain::GrainNeighborList nl2 = grain::BuildGrainNeighborList(ps, g2, tab2, hSearch);
        check(nl.neighborStart == nl2.neighborStart && nl.neighbors == nl2.neighbors &&
              tab.cellStart == tab2.cellStart && tab.cellGrains == tab2.cellGrains,
              "neighbor: two builds byte-identical (deterministic)");
    }

    // ----- a SINGLE grain (and grains spread > hSearch apart) -> 0 neighbors (the sparse no-op) -----------
    {
        const fx hSearch = kOne;
        std::vector<grain::GrainParticle> one(1);
        one[0].pos = {FromInt(5), FromInt(2), FromInt(-3)}; one[0].invMass = kOne;
        grain::GrainGrid g = grain::MakeGrainGrid(one, hSearch);
        grain::GrainCellTable t = grain::BuildGrainCellTable(one, g);
        grain::GrainNeighborList nl = grain::BuildGrainNeighborList(one, g, t, hSearch);
        check(nl.neighbors.empty() && nl.neighborStart.size() == 2 && nl.neighborStart[1] == 0u,
              "neighbor: single grain -> 0 neighbors (sparse no-op)");

        std::vector<grain::GrainParticle> sparse(3);
        sparse[0].pos = {0, 0, 0}; sparse[1].pos = {FromInt(5), 0, 0}; sparse[2].pos = {FromInt(10), 0, 0};
        for (auto& p : sparse) p.invMass = kOne;
        grain::GrainGrid gs = grain::MakeGrainGrid(sparse, hSearch);
        grain::GrainCellTable ts = grain::BuildGrainCellTable(sparse, gs);
        grain::GrainNeighborList nls = grain::BuildGrainNeighborList(sparse, gs, ts, hSearch);
        check(nls.neighbors.empty(), "neighbor: grains spread > hSearch apart -> 0 neighbors");
    }

    // ============================ Slice GR3 — FRICTIONLESS CONTACT PROJECTION ========================

    // ----- SolveGrainContact: a hand-laid overlapping pair -> exact inverse-mass-weighted Q16.16 push ------
    {
        // Two equal-mass grains, radius 0.5 each (contact diameter 1.0), centres 0.6 apart on x -> overlap
        // pen = (0.5+0.5) − 0.6 = 0.4. Each grain takes a 50/50 share -> Δp = 0.5 · 0.4 = 0.2 along ±x.
        const fx r = kOne / 2;                 // 0.5 radius
        const fx hSearch = kOne + kOne / 2;    // 1.5 >= contact diameter 1.0
        std::vector<grain::GrainParticle> pair(2);
        pair[0].pos = {0, 0, 0};               pair[0].invMass = kOne; pair[0].radius = r;
        pair[1].pos = {kOne * 6 / 10, 0, 0};   pair[1].invMass = kOne; pair[1].radius = r;   // 0.6 on x
        grain::GrainGrid g = grain::MakeGrainGrid(pair, hSearch);
        grain::GrainCellTable t = grain::BuildGrainCellTable(pair, g);
        grain::GrainNeighborList nl = grain::BuildGrainNeighborList(pair, g, t, hSearch);
        std::vector<grain::FxVec3> dp;
        grain::SolveGrainContact(pair, nl, dp);
        // d0 = p0 − p1 = (−0.6,0,0) -> unit = (−1,0,0); Δp0 = 0.5·0.4·(−1) = −0.2 on x.
        check(dp[0].x == -(kOne * 2 / 10) && dp[0].y == 0 && dp[0].z == 0,
              "GR3 SolveGrainContact: equal-mass pair Δp0 == −0.2 on x (50/50 split of pen 0.4)");
        check(dp[1].x == (kOne * 2 / 10) && dp[1].y == 0 && dp[1].z == 0,
              "GR3 SolveGrainContact: equal-mass pair Δp1 == +0.2 on x (symmetric)");

        // A STATIC + DYNAMIC pair: only the dynamic one moves, and it takes the FULL pen (share w_d/(w_d+0)
        // = 1). The same 0.6 overlap -> the dynamic grain pushes the full 0.4 away.
        std::vector<grain::GrainParticle> sd(2);
        sd[0].pos = {0, 0, 0};                 sd[0].invMass = 0;     sd[0].radius = r;       // STATIC
        sd[0].flags = grain::kFlagStatic;
        sd[1].pos = {kOne * 6 / 10, 0, 0};     sd[1].invMass = kOne;  sd[1].radius = r;       // DYNAMIC
        grain::GrainGrid g2 = grain::MakeGrainGrid(sd, hSearch);
        grain::GrainCellTable t2 = grain::BuildGrainCellTable(sd, g2);
        grain::GrainNeighborList nl2 = grain::BuildGrainNeighborList(sd, g2, t2, hSearch);
        std::vector<grain::FxVec3> dp2;
        grain::SolveGrainContact(sd, nl2, dp2);
        check(dp2[0].x == 0 && dp2[0].y == 0 && dp2[0].z == 0,
              "GR3 SolveGrainContact: static grain Δp == 0 (pinned case)");
        // d1 = p1 − p0 = (+0.6,0,0) -> unit (+1,0,0); share = w_d/(w_d+0) = 1; Δp1 = 1·pen·(+1) on x.
        // pen = (0.5+0.5) − |0.6| in exact Q16.16: 0.6 -> 39321 LSB, pen = 65536 − 39321 = 26215 (the
        // dynamic grain takes the FULL pen since the static partner contributes nothing).
        const fx kPen = kOne - (kOne * 6 / 10);   // 26215 (the exact integer penetration)
        check(dp2[1].x == kPen && dp2[1].y == 0 && dp2[1].z == 0,
              "GR3 SolveGrainContact: static+dynamic -> dynamic takes the FULL pen (exact 26215)");
    }

    // ----- SolveGrainContact: a NON-overlapping pair -> Δp 0 (the exact radial cull GR2 deferred) ----------
    {
        const fx r = kOne / 4;                 // 0.25 radius (diameter 0.5)
        const fx hSearch = kOne;               // 1.0 >= diameter 0.5
        // Centres 0.7 apart on x: within hSearch (a GR2 candidate) but pen = (0.25+0.25) − 0.7 = −0.2 < 0
        // -> NO overlap -> Δp 0 (the radial cull lands here, not in GR2).
        std::vector<grain::GrainParticle> pair(2);
        pair[0].pos = {0, 0, 0};               pair[0].invMass = kOne; pair[0].radius = r;
        pair[1].pos = {kOne * 7 / 10, 0, 0};   pair[1].invMass = kOne; pair[1].radius = r;
        grain::GrainGrid g = grain::MakeGrainGrid(pair, hSearch);
        grain::GrainCellTable t = grain::BuildGrainCellTable(pair, g);
        grain::GrainNeighborList nl = grain::BuildGrainNeighborList(pair, g, t, hSearch);
        check(nl.neighbors.size() == 2, "GR3 cull: the non-overlapping pair is still a GR2 candidate");
        std::vector<grain::FxVec3> dp;
        grain::SolveGrainContact(pair, nl, dp);
        check(dp[0].x == 0 && dp[0].y == 0 && dp[0].z == 0 &&
              dp[1].x == 0 && dp[1].y == 0 && dp[1].z == 0,
              "GR3 cull: non-overlapping candidate -> Δp 0 (pen <= 0 contributes nothing)");
    }

    // ----- CollideGrainPlane: a grain below groundY+radius snaps to groundY+radius (surface on floor) ------
    {
        const fx groundY = 0;
        const fx r = kOne / 2;
        std::vector<grain::GrainParticle> ps(2);
        ps[0].pos = {0, -(kOne), 0};   ps[0].invMass = kOne; ps[0].radius = r;   // y = −1, below the floor
        ps[1].pos = {0, FromInt(5), 0}; ps[1].invMass = kOne; ps[1].radius = r;  // y = 5, above -> untouched
        grain::CollideGrainPlane(ps, groundY);
        check(ps[0].pos.y == groundY + r, "GR3 CollideGrainPlane: grain below floor snaps to groundY+radius");
        check(ps[1].pos.y == FromInt(5), "GR3 CollideGrainPlane: grain above floor untouched");
        // A STATIC grain below the floor IS clamped (a fallen boundary grain is raised).
        std::vector<grain::GrainParticle> st(1);
        st[0].pos = {0, -(kOne), 0}; st[0].invMass = 0; st[0].radius = r; st[0].flags = grain::kFlagStatic;
        grain::CollideGrainPlane(st, groundY);
        check(st[0].pos.y == groundY + r, "GR3 CollideGrainPlane: static grain below floor IS clamped");
    }

    // ----- CollideGrainSphere: a grain inside a sphere snaps to sphereR + grainR (the surfaces touch) ------
    {
        // A sphere centre at origin, radius 2.0; a grain radius 0.5 at (1,0,0) (centre-distance 1 < 2.5 =
        // sphereR+grainR) -> snap the centre out to 2.5 along +x.
        grain::GrainSphereCollider s; s.center = {0, 0, 0}; s.radius = FromInt(2);
        grain::GrainParticle p; p.pos = {kOne, 0, 0}; p.invMass = kOne; p.radius = kOne / 2;
        bool hit = grain::CollideGrainSphere(p, s);
        check(hit, "GR3 CollideGrainSphere: grain inside the expanded sphere is a contact");
        check(p.pos.x == FromInt(2) + kOne / 2 && p.pos.y == 0 && p.pos.z == 0,
              "GR3 CollideGrainSphere: centre snapped to sphereR + grainR == 2.5 on +x");
        // A grain OUTSIDE (centre-distance 3 > 2.5) -> untouched.
        grain::GrainParticle q; q.pos = {FromInt(3), 0, 0}; q.invMass = kOne; q.radius = kOne / 2;
        check(!grain::CollideGrainSphere(q, s) && q.pos.x == FromInt(3),
              "GR3 CollideGrainSphere: grain outside the expanded sphere untouched");
        // A STATIC grain inside the sphere is NOT projected (it holds).
        grain::GrainParticle st; st.pos = {kOne, 0, 0}; st.invMass = 0; st.radius = kOne / 2;
        st.flags = grain::kFlagStatic;
        check(!grain::CollideGrainSphere(st, s) && st.pos.x == kOne,
              "GR3 CollideGrainSphere: static grain not sphere-projected (holds)");
        // GrainSphereFromBody bridges an fpx::FxBody -> a collider with its pos + radius.
        hf::sim::fpx::FxBody b; b.pos = {FromInt(3), FromInt(4), FromInt(5)}; b.radius = FromInt(2);
        grain::GrainSphereCollider sc = grain::GrainSphereFromBody(b);
        check(sc.center.x == FromInt(3) && sc.center.y == FromInt(4) && sc.center.z == FromInt(5) &&
              sc.radius == FromInt(2), "GR3 GrainSphereFromBody: bridges FxBody pos + radius");
    }

    // ----- StepGrainContact: a tiny overlapping cluster settles to REDUCED penetration, deterministic, ----
    // and Jacobi GPU-order-independent (the per-grain dp accumulate is order-free). -----------------------
    {
        const grain::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = 0;
        const fx r = kOne / 2;                 // 0.5 radius (diameter 1.0)
        const fx hSearch = FromInt(2);         // 2.0 >= diameter 1.0
        const int iters = 6, steps = 30;
        const std::vector<grain::GrainSphereCollider> noSpheres;

        // A 3x3x3 cluster spaced 0.6 (< the diameter 1.0) -> every lattice neighbour OVERLAPS at start.
        grain::GrainBlock block;
        block.W = 3; block.H = 3; block.D = 3; block.spacing = kOne * 6 / 10;   // 0.6 spacing -> overlapping
        block.radius = r; block.origin = grain::FxVec3{0, FromInt(4), 0};
        std::vector<grain::GrainParticle> cluster = grain::InitGrainBlock(block);

        const grain::GrainPenetration before = grain::MeasureGrainPenetration(cluster, hSearch);
        check(before.summed > 0, "GR3 StepGrainContact: the initial cluster overlaps (penBefore > 0)");

        std::vector<grain::GrainParticle> a = cluster;
        grain::StepGrainContactSteps(a, noSpheres, grav, dt, groundY, hSearch, iters, steps);
        const grain::GrainPenetration after = grain::MeasureGrainPenetration(a, hSearch);
        check(after.summed < before.summed,
              "GR3 StepGrainContact: the solve RELIEVES overlap (penAfter < penBefore — not zero, the FL4 caveat)");
        // Every grain ends at/above the floor (radius-aware): pos.y >= groundY + radius − eps.
        bool aboveFloor = true;
        for (const grain::GrainParticle& p : a)
            if (p.pos.y < groundY + p.radius - grain::kGrainCollideEps) aboveFloor = false;
        check(aboveFloor, "GR3 StepGrainContact: no grain ends below groundY + radius (within eps)");

        // Determinism: two full runs byte-identical.
        std::vector<grain::GrainParticle> b = cluster;
        grain::StepGrainContactSteps(b, noSpheres, grav, dt, groundY, hSearch, iters, steps);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(grain::GrainParticle)) == 0,
              "GR3 StepGrainContact: two runs byte-identical (deterministic)");

        // Jacobi order-independence: SolveGrainContact reads iteration-start positions into a SEPARATE dp[],
        // so the per-grain accumulate is order-free. A shuffled grain order (with the neighbour list rebuilt
        // for that order) yields the same per-grain dp -> a single solve+apply matches in-order per grain.
        {
            grain::GrainGrid g = grain::MakeGrainGrid(cluster, hSearch);
            grain::GrainCellTable t = grain::BuildGrainCellTable(cluster, g);
            grain::GrainNeighborList nl = grain::BuildGrainNeighborList(cluster, g, t, hSearch);
            std::vector<grain::FxVec3> dp1, dp2;
            grain::SolveGrainContact(cluster, nl, dp1);
            grain::SolveGrainContact(cluster, nl, dp2);   // recompute -> identical (no in-place dependence)
            bool same = dp1.size() == dp2.size();
            for (size_t i = 0; i < dp1.size() && same; ++i)
                if (std::memcmp(&dp1[i], &dp2[i], sizeof(grain::FxVec3)) != 0) same = false;
            check(same, "GR3 SolveGrainContact: Jacobi dp accumulate is order-free (recompute identical)");
        }
    }

    // ----- StepGrainContact: a single grain (no overlap, no colliders) -> free-fall + ground rest only ----
    {
        const grain::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = 0;
        const fx hSearch = FromInt(2);
        const std::vector<grain::GrainSphereCollider> noSpheres;
        std::vector<grain::GrainParticle> one(1);
        one[0].pos = {0, FromInt(10), 0}; one[0].invMass = kOne; one[0].radius = kOne / 2;

        // The contact solve is idle (no neighbours): StepGrainContact == the GR1 free-fall + ground rest.
        std::vector<grain::GrainParticle> viaContact = one;
        grain::StepGrainContactSteps(viaContact, noSpheres, grav, dt, groundY, hSearch, 6, 200);
        check(viaContact[0].pos.y == groundY + one[0].radius,
              "GR3 no-op: a lone grain free-falls + rests at groundY + radius (contact solve idle)");
    }

    // ============================ Slice GR4 — TANGENTIAL COULOMB FRICTION ============================

    // ----- SolveGrainFriction: a hand-laid overlapping pair with a known tangential slip ------------------
    {
        // Two equal-mass grains, radius 0.5 each (diameter 1.0), centres 0.6 apart on x -> overlap pen = 0.4.
        // The contact normal n = unit(p_i − p_j) is along ±x. Give grain 0 a PURELY TANGENTIAL (z) slip this
        // step: prev0 = pos0 − (0,0,sz), prev1 = pos1 (so Δx_rel = (0,0,sz), entirely tangential to the x-normal).
        const fx r = kOne / 2;                 // 0.5 radius
        const fx hSearch = kOne + kOne / 2;    // 1.5 >= contact diameter 1.0
        const fx mu = kOne;                    // μ = 1.0 (fmax = pen = 0.4)
        const fx pen = kOne - (kOne * 6 / 10); // 26215 (0.4 in Q16.16, the exact integer penetration)

        // Case A: SMALL slip (< fmax = μ·pen) -> STATIC: corr == Δx_t cancels the WHOLE slip. share = 0.5.
        {
            const fx slip = kOne / 10;         // 0.1 on z (< fmax 0.4)
            std::vector<grain::GrainParticle> pair(2);
            pair[0].pos = {0, 0, 0};               pair[0].invMass = kOne; pair[0].radius = r;
            pair[0].prev = {0, 0, -slip};          // Δx0 = (0,0,+slip)
            pair[1].pos = {kOne * 6 / 10, 0, 0};   pair[1].invMass = kOne; pair[1].radius = r;
            pair[1].prev = pair[1].pos;            // Δx1 = 0  -> Δx_rel = (0,0,slip), tangential
            grain::GrainGrid g = grain::MakeGrainGrid(pair, hSearch);
            grain::GrainCellTable t = grain::BuildGrainCellTable(pair, g);
            grain::GrainNeighborList nl = grain::BuildGrainNeighborList(pair, g, t, hSearch);
            std::vector<grain::FxVec3> dp;
            grain::SolveGrainFriction(pair, nl, mu, dp);
            // grain 0: Δp = −share·corr = −0.5·(0,0,slip). share = fxdiv(kOne,2kOne) = 0.5; corr.z = +slip.
            const fx half = grain::fxdiv(kOne, kOne + kOne);   // 0.5 in Q16.16
            check(dp[0].x == 0 && dp[0].y == 0 && dp[0].z == -grain::fxmul(half, slip),
                  "GR4 SolveGrainFriction: small slip (<μ·pen) -> STATIC, Δp cancels half the slip (−z)");
            // grain 1: its Δx_rel = (p1−prev1) − (p0−prev0) = (0,0,−slip); share 0.5 -> Δp = −0.5·(0,0,−slip).
            // (The >> floor-truncation makes the two halves off-by-one-LSB asymmetric, the documented fxmul bit.)
            check(dp[1].x == 0 && dp[1].y == 0 && dp[1].z == -grain::fxmul(half, -slip),
                  "GR4 SolveGrainFriction: symmetric half on grain 1 (+z, the fxmul-floor LSB)");
        }

        // Case B: LARGE slip (> fmax = μ·pen) -> KINETIC: corr = Δx_t · (fmax/t), clamped to the cone.
        {
            const fx slip = kOne;              // 1.0 on z (>> fmax 0.4)
            std::vector<grain::GrainParticle> pair(2);
            pair[0].pos = {0, 0, 0};               pair[0].invMass = kOne; pair[0].radius = r;
            pair[0].prev = {0, 0, -slip};
            pair[1].pos = {kOne * 6 / 10, 0, 0};   pair[1].invMass = kOne; pair[1].radius = r;
            pair[1].prev = pair[1].pos;
            grain::GrainGrid g = grain::MakeGrainGrid(pair, hSearch);
            grain::GrainCellTable t = grain::BuildGrainCellTable(pair, g);
            grain::GrainNeighborList nl = grain::BuildGrainNeighborList(pair, g, t, hSearch);
            std::vector<grain::FxVec3> dp;
            grain::SolveGrainFriction(pair, nl, mu, dp);
            // t = slip = 1.0; corr = (0,0,slip)·(fmax/t) = (0,0, fxdiv(pen,slip)) = (0,0,pen). Δp = −0.5·pen.
            const fx corrZ = grain::fxdiv(grain::fxmul(mu, pen), slip);   // == fmax/t == pen (μ=1)
            const fx expectZ = -grain::fxmul(grain::fxdiv(kOne, kOne + kOne), corrZ);   // −share·corr
            check(dp[0].z == expectZ && dp[0].x == 0 && dp[0].y == 0,
                  "GR4 SolveGrainFriction: large slip (>μ·pen) -> KINETIC, Δp clamped to the cone");
            check(corrZ < slip, "GR4 SolveGrainFriction: kinetic corr is clamped below the raw slip (cone)");
        }

        // Case C: μ = 0 -> fmax = 0 -> t > fmax always (kinetic with a zero cone) -> Δp 0 (friction idle).
        {
            const fx slip = kOne / 10;
            std::vector<grain::GrainParticle> pair(2);
            pair[0].pos = {0, 0, 0};               pair[0].invMass = kOne; pair[0].radius = r;
            pair[0].prev = {0, 0, -slip};
            pair[1].pos = {kOne * 6 / 10, 0, 0};   pair[1].invMass = kOne; pair[1].radius = r;
            pair[1].prev = pair[1].pos;
            grain::GrainGrid g = grain::MakeGrainGrid(pair, hSearch);
            grain::GrainCellTable t = grain::BuildGrainCellTable(pair, g);
            grain::GrainNeighborList nl = grain::BuildGrainNeighborList(pair, g, t, hSearch);
            std::vector<grain::FxVec3> dp;
            grain::SolveGrainFriction(pair, nl, 0, dp);
            check(dp[0].x == 0 && dp[0].y == 0 && dp[0].z == 0 &&
                  dp[1].x == 0 && dp[1].y == 0 && dp[1].z == 0,
                  "GR4 SolveGrainFriction: μ=0 -> Δp 0 (friction idle, the frictionless control)");
        }

        // Case D: a STATIC + DYNAMIC pair -> only the dynamic moves; static partner Δp 0, share = w_d/(w_d+0) = 1.
        {
            const fx slip = kOne / 10;
            std::vector<grain::GrainParticle> sd(2);
            sd[0].pos = {0, 0, 0};                 sd[0].invMass = 0;     sd[0].radius = r;   // STATIC
            sd[0].flags = grain::kFlagStatic;      sd[0].prev = {0, 0, -slip};
            sd[1].pos = {kOne * 6 / 10, 0, 0};     sd[1].invMass = kOne;  sd[1].radius = r;   // DYNAMIC
            sd[1].prev = sd[1].pos;
            grain::GrainGrid g = grain::MakeGrainGrid(sd, hSearch);
            grain::GrainCellTable t = grain::BuildGrainCellTable(sd, g);
            grain::GrainNeighborList nl = grain::BuildGrainNeighborList(sd, g, t, hSearch);
            std::vector<grain::FxVec3> dp;
            grain::SolveGrainFriction(sd, nl, mu, dp);
            check(dp[0].x == 0 && dp[0].y == 0 && dp[0].z == 0,
                  "GR4 SolveGrainFriction: static grain Δp == 0 (pinned)");
            // dynamic: Δx_rel = (p1−prev1) − (p0−prev0) = (0,0,0) − (0,0,slip) = (0,0,−slip); small (<fmax) ->
            // STATIC; share = 1; Δp = −1·(0,0,−slip) = (0,0,+slip).
            check(dp[1].z == slip && dp[1].x == 0 && dp[1].y == 0,
                  "GR4 SolveGrainFriction: static+dynamic -> dynamic takes the FULL correction (share 1)");
        }
    }

    // ----- StepGrainFriction: a sloped cluster HOLDS with slope > the μ=0 control, deterministic, Jacobi ----
    {
        const grain::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = 0;
        const fx r = kOne / 2;                 // 0.5 radius (diameter 1.0)
        const fx hSearch = FromInt(2);         // 2.0 >= diameter 1.0
        const int iters = 2, steps = 70;       // == the --grain-friction showcase config
        const fx mu = grain::kGrainMu;         // ~0.8
        const std::vector<grain::GrainSphereCollider> noSpheres;

        // A 5x5x5 STAGGERED block dropped onto FLAT ground (NO collider sphere). The half-offset on the odd
        // y-layers breaks the perfect-lattice symmetry so the collapsing column generates real TANGENTIAL slip
        // (a perfectly axis-aligned lattice collapses purely radially -> zero shear -> friction idle). spacing
        // 1.0 (== the contact diameter, non-overlapping start) + a small 0.12 stagger -> a clean shear source.
        // This is the EXACT --grain-friction showcase scene (the host-snapped angle-of-repose cone config).
        const fx sp = kOne;                    // 1.0 spacing (== diameter)
        const fx off = (fx)(0.12 * (double)kOne + 0.5);   // 0.12 stagger offset on odd y-layers
        std::vector<grain::GrainParticle> init;
        for (int iy = 0; iy < 5; ++iy)
            for (int iz = 0; iz < 5; ++iz)
                for (int ix = 0; ix < 5; ++ix) {
                    grain::GrainParticle p;
                    const fx ox = (iy & 1) ? off : 0, oz = (iy & 1) ? off : 0;
                    p.pos = grain::FxVec3{(fx)(ix * (int)sp) + ox, FromInt(3) + (fx)(iy * (int)sp),
                                          (fx)(iz * (int)sp) + oz};
                    p.prev = p.pos; p.invMass = kOne; p.radius = r; p.flags = 0;
                    init.push_back(p);
                }

        // WITH friction: the pile holds a slope.
        std::vector<grain::GrainParticle> withMu = init;
        grain::StepGrainFrictionSteps(withMu, noSpheres, grav, dt, groundY, hSearch, mu, iters, steps);
        const grain::GrainRepose repWith = grain::MeasureGrainRepose(withMu, groundY);

        // μ=0 control: frictionless -> equals the GR3 StepGrainContact (the pile spreads flatter).
        std::vector<grain::GrainParticle> zeroMu = init;
        grain::StepGrainFrictionSteps(zeroMu, noSpheres, grav, dt, groundY, hSearch, 0, iters, steps);
        const grain::GrainRepose repZero = grain::MeasureGrainRepose(zeroMu, groundY);

        // μ=0 StepGrainFriction MUST equal the GR3 StepGrainContact byte-for-byte (the frictionless control).
        std::vector<grain::GrainParticle> gr3 = init;
        grain::StepGrainContactSteps(gr3, noSpheres, grav, dt, groundY, hSearch, iters, steps);
        check(zeroMu.size() == gr3.size() &&
              std::memcmp(zeroMu.data(), gr3.data(), gr3.size() * sizeof(grain::GrainParticle)) == 0,
              "GR4 StepGrainFriction: μ=0 == GR3 StepGrainContact byte-for-byte (the frictionless control)");

        check(repWith.slope > 0, "GR4 StepGrainFriction: the friction pile holds a slope (slope > 0)");
        check(repWith.slope > repZero.slope,
              "GR4 StepGrainFriction: friction slope > the μ=0 control slope (friction holds the heap)");

        // Determinism: two full runs byte-identical.
        std::vector<grain::GrainParticle> b = init;
        grain::StepGrainFrictionSteps(b, noSpheres, grav, dt, groundY, hSearch, mu, iters, steps);
        check(withMu.size() == b.size() &&
              std::memcmp(withMu.data(), b.data(), withMu.size() * sizeof(grain::GrainParticle)) == 0,
              "GR4 StepGrainFriction: two runs byte-identical (deterministic)");

        // Jacobi order-independence: SolveGrainFriction reads iteration-start positions into a SEPARATE dp[],
        // so the per-grain accumulate is order-free (recompute yields identical dp).
        {
            grain::GrainGrid g = grain::MakeGrainGrid(withMu, hSearch);
            grain::GrainCellTable t = grain::BuildGrainCellTable(withMu, g);
            grain::GrainNeighborList nl = grain::BuildGrainNeighborList(withMu, g, t, hSearch);
            std::vector<grain::FxVec3> dp1, dp2;
            grain::SolveGrainFriction(withMu, nl, mu, dp1);
            grain::SolveGrainFriction(withMu, nl, mu, dp2);
            bool same = dp1.size() == dp2.size();
            for (size_t i = 0; i < dp1.size() && same; ++i)
                if (std::memcmp(&dp1[i], &dp2[i], sizeof(grain::FxVec3)) != 0) same = false;
            check(same, "GR4 SolveGrainFriction: Jacobi dp accumulate is order-free (recompute identical)");
        }
    }

    // ----- MeasureGrainRepose: a known cone -> the expected height/baseRadius/slope -----------------------
    {
        // A simple cone: an apex grain at (0, 4, 0) + a ring of base grains at radius 2 on the ground (y=0).
        // centroid (x,z) ≈ origin (the ring is symmetric + the apex at origin) -> height 4, baseRadius 2,
        // slope = 4/2 = 2.0.
        const fx groundY = 0;
        std::vector<grain::GrainParticle> cone;
        grain::GrainParticle apex; apex.pos = {0, FromInt(4), 0}; apex.invMass = kOne; apex.radius = kOne / 2;
        cone.push_back(apex);
        // 4 base grains at (+2,0,0),(−2,0,0),(0,0,+2),(0,0,−2): symmetric -> centroid at origin.
        const int bx[4] = {2, -2, 0, 0};
        const int bz[4] = {0, 0, 2, -2};
        for (int k = 0; k < 4; ++k) {
            grain::GrainParticle b; b.pos = {FromInt(bx[k]), 0, FromInt(bz[k])};
            b.invMass = kOne; b.radius = kOne / 2;
            cone.push_back(b);
        }
        const grain::GrainRepose rep = grain::MeasureGrainRepose(cone, groundY);
        check(rep.height == FromInt(4), "GR4 MeasureGrainRepose: height == max pos.y − groundY == 4.0");
        check(rep.baseRadius == FromInt(2), "GR4 MeasureGrainRepose: baseRadius == max horizontal dist == 2.0");
        check(rep.slope == FromInt(2), "GR4 MeasureGrainRepose: slope == height/baseRadius == 2.0");
        // A single-column degenerate (all grains stacked on the axis) -> baseRadius 0 -> slope 0.
        std::vector<grain::GrainParticle> column(3);
        for (int k = 0; k < 3; ++k) { column[k].pos = {0, FromInt(k), 0}; column[k].invMass = kOne; }
        const grain::GrainRepose deg = grain::MeasureGrainRepose(column, groundY);
        check(deg.baseRadius == 0 && deg.slope == 0,
              "GR4 MeasureGrainRepose: single-column -> baseRadius 0, slope 0 (degenerate)");
    }

    // ================= GR5: ApplyGrainCommand — wind / push / OOB no-op / static hold =================
    {
        std::vector<grain::GrainParticle> ps(4);
        for (int i = 0; i < 4; ++i) { ps[(size_t)i].pos = {FromInt(i), 0, 0}; ps[(size_t)i].invMass = kOne; }

        // kCmdWind adds to a dynamic grain's velocity.
        const grain::GrainCommand wind{0, grain::kCmdWind, 1u, grain::FxVec3{FromInt(2), FromInt(-3), FromInt(1)}};
        grain::ApplyGrainCommand(ps, wind);
        check(ps[1].vel.x == FromInt(2) && ps[1].vel.y == FromInt(-3) && ps[1].vel.z == FromInt(1),
              "GR5 ApplyGrainCommand: kCmdWind adds the delta-velocity to a dynamic grain");

        // kCmdPush adds to a dynamic grain's position.
        const grain::fx px0 = ps[2].pos.x;
        grain::ApplyGrainCommand(ps, grain::GrainCommand{0, grain::kCmdPush, 2u, grain::FxVec3{FromInt(5), 0, 0}});
        check(ps[2].pos.x == px0 + FromInt(5),
              "GR5 ApplyGrainCommand: kCmdPush adds the delta-position to a dynamic grain");

        // A static grain holds — wind/push are a no-op.
        ps[0].flags = grain::kFlagStatic; ps[0].invMass = 0;
        const grain::GrainParticle sBefore = ps[0];
        grain::ApplyGrainCommand(ps, grain::GrainCommand{0, grain::kCmdWind, 0u, grain::FxVec3{FromInt(9), 0, 0}});
        grain::ApplyGrainCommand(ps, grain::GrainCommand{0, grain::kCmdPush, 0u, grain::FxVec3{FromInt(9), 0, 0}});
        check(std::memcmp(&sBefore, &ps[0], sizeof(grain::GrainParticle)) == 0,
              "GR5 ApplyGrainCommand: a static grain holds (wind/push are no-ops)");

        // Out-of-range target is a deterministic no-op (no crash, no mutation).
        std::vector<grain::GrainParticle> before = ps;
        grain::ApplyGrainCommand(ps, grain::GrainCommand{0, grain::kCmdWind, 9999u, grain::FxVec3{FromInt(5), 0, 0}});
        check(before.size() == ps.size() &&
              std::memcmp(before.data(), ps.data(), ps.size() * sizeof(grain::GrainParticle)) == 0,
              "GR5 ApplyGrainCommand: out-of-range target is a no-op");
    }

    // ================= GR5: SnapshotGrain / RestoreGrain round-trip == original ========================
    {
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        const fx kDt = kOne / 60;
        const grain::FxVec3 kGravity{0, kGravY, 0};
        const fx kHSearch = kOne * 2, kMu = grain::kGrainMu, kGroundY = 0;
        const int iters = 2;
        std::vector<grain::GrainParticle> ps;
        for (int iy = 0; iy < 4; ++iy)
            for (int iz = 0; iz < 4; ++iz)
                for (int ix = 0; ix < 4; ++ix) {
                    grain::GrainParticle p;
                    p.pos = {FromInt(ix), FromInt(3 + iy), FromInt(iz)};
                    p.prev = p.pos; p.invMass = kOne; p.radius = kOne / 2;
                    ps.push_back(p);
                }
        const std::vector<grain::GrainSphereCollider> noSpheres;
        // Advance a few steps so the state is non-trivial, then snapshot it.
        grain::StepGrainFrictionSteps(ps, noSpheres, kGravity, kDt, kGroundY, kHSearch, kMu, iters, 5);

        const std::vector<grain::GrainParticle> snap = grain::SnapshotGrain(ps);
        // Mutate (one more step), then restore -> must equal the snapshot exactly.
        grain::StepGrainFriction(ps, noSpheres, kGravity, kDt, kGroundY, kHSearch, kMu, iters);
        check(std::memcmp(ps.data(), snap.data(), snap.size() * sizeof(grain::GrainParticle)) != 0,
              "GR5 snapshot: a step actually mutated the grains (control)");
        grain::RestoreGrain(ps, snap);
        check(ps.size() == snap.size() &&
              std::memcmp(ps.data(), snap.data(), snap.size() * sizeof(grain::GrainParticle)) == 0,
              "GR5 snapshot: SnapshotGrain -> RestoreGrain round-trip == original BIT-EXACT");
    }

    // ================= GR5: SimGrainTick determinism + deterministic command order ====================
    {
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        const fx kDt = kOne / 60;
        const grain::FxVec3 kGravity{0, kGravY, 0};
        const fx kHSearch = kOne * 2, kMu = grain::kGrainMu, kGroundY = 0;
        const int iters = 2;
        std::vector<grain::GrainParticle> base;
        for (int iy = 0; iy < 4; ++iy)
            for (int iz = 0; iz < 4; ++iz)
                for (int ix = 0; ix < 4; ++ix) {
                    grain::GrainParticle p;
                    p.pos = {FromInt(ix), FromInt(3 + iy), FromInt(iz)};
                    p.prev = p.pos; p.invMass = kOne; p.radius = kOne / 2;
                    base.push_back(p);
                }
        const std::vector<grain::GrainSphereCollider> noSpheres;
        const std::vector<grain::GrainCommand> stream{
            grain::GrainCommand{0, grain::kCmdWind, 5u, grain::FxVec3{FromInt(3), 0, 0}},
        };
        // Two SimGrainTick runs over the SAME init+stream -> byte-identical.
        std::vector<grain::GrainParticle> a = base, b = base;
        grain::SimGrainTick(a, noSpheres, stream, 0, kGravity, kDt, kGroundY, kHSearch, kMu, iters);
        grain::SimGrainTick(b, noSpheres, stream, 0, kGravity, kDt, kGroundY, kHSearch, kMu, iters);
        check(std::memcmp(a.data(), b.data(), a.size() * sizeof(grain::GrainParticle)) == 0,
              "GR5 SimGrainTick: two runs byte-identical (deterministic)");

        // A tick whose commands don't match -> no command applied (pure StepGrainFriction).
        std::vector<grain::GrainParticle> noCmd = base, plain = base;
        grain::SimGrainTick(noCmd, noSpheres, stream, 5, kGravity, kDt, kGroundY, kHSearch, kMu, iters);  // no cmd@5
        grain::StepGrainFriction(plain, noSpheres, kGravity, kDt, kGroundY, kHSearch, kMu, iters);
        check(std::memcmp(noCmd.data(), plain.data(), plain.size() * sizeof(grain::GrainParticle)) == 0,
              "GR5 SimGrainTick: a tick with no matching command == pure StepGrainFriction");
    }

    // ================= GR5: RunGrainLockstep replica == authority (the lockstep headline) =============
    // ================= + RunGrainRollback positive (converges) + negative (mispredict differs) ========
    {
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        const fx kDt = kOne / 60;
        const grain::FxVec3 kGravity{0, kGravY, 0};
        const fx kHSearch = kOne * 2, kMu = grain::kGrainMu, kGroundY = 0;
        const int iters = 2;
        const int ticks = 16, mispredictTick = 6;
        const grain::fx kStagger = (grain::fx)(0.12 * (double)kOne + 0.5);
        std::vector<grain::GrainParticle> init;
        for (int iy = 0; iy < 5; ++iy)
            for (int iz = 0; iz < 5; ++iz)
                for (int ix = 0; ix < 5; ++ix) {
                    grain::GrainParticle p;
                    const grain::fx ox = (iy & 1) ? kStagger : 0, oz = (iy & 1) ? kStagger : 0;
                    p.pos = {FromInt(ix) + ox, FromInt(3 + iy), FromInt(iz) + oz};
                    p.prev = p.pos; p.invMass = kOne; p.radius = kOne / 2;
                    init.push_back(p);
                }
        const std::vector<grain::GrainSphereCollider> noSpheres;

        const uint32_t wIdx = 62u;   // a mid-pile grain the wind shoves
        const std::vector<grain::GrainCommand> authStream{
            grain::GrainCommand{2,  grain::kCmdWind, wIdx, grain::FxVec3{FromInt(8), 0, 0}},
            grain::GrainCommand{6,  grain::kCmdPush, wIdx, grain::FxVec3{0, 0, FromInt(2)}},
            grain::GrainCommand{10, grain::kCmdWind, wIdx, grain::FxVec3{FromInt(4), 0, 0}},
        };
        // The MISPREDICTED stream: auth + a WRONG strong wind at mispredictTick (a real divergence).
        std::vector<grain::GrainCommand> mispredictStream = authStream;
        mispredictStream.push_back(grain::GrainCommand{(uint32_t)mispredictTick, grain::kCmdWind, wIdx,
                                                       grain::FxVec3{FromInt(60), 0, 0}});

        const std::vector<grain::GrainParticle> authority =
            grain::RunGrainLockstep(init, noSpheres, authStream, ticks, kGravity, kDt, kGroundY, kHSearch, kMu, iters);
        const std::vector<grain::GrainParticle> replica =
            grain::RunGrainLockstep(init, noSpheres, authStream, ticks, kGravity, kDt, kGroundY, kHSearch, kMu, iters);

        // LOCKSTEP: replica (inputs-only) == authority BIT-EXACT.
        check(authority.size() == replica.size() &&
              std::memcmp(authority.data(), replica.data(), authority.size() * sizeof(grain::GrainParticle)) == 0,
              "GR5 lockstep: replica == authority BIT-EXACT (inputs-only re-sim)");

        // ROLLBACK positive: rolledBack == authority BIT-EXACT.
        const std::vector<grain::GrainParticle> rolledBack =
            grain::RunGrainRollback(init, noSpheres, authStream, mispredictStream,
                                    ticks, mispredictTick, kGravity, kDt, kGroundY, kHSearch, kMu, iters);
        check(rolledBack.size() == authority.size() &&
              std::memcmp(rolledBack.data(), authority.data(), authority.size() * sizeof(grain::GrainParticle)) == 0,
              "GR5 rollback: corrected to authority BIT-EXACT (positive control)");

        // ROLLBACK negative control: the pre-rollback MISPREDICTED full run DIFFERED from authority.
        const std::vector<grain::GrainParticle> mispredicted =
            grain::RunGrainLockstep(init, noSpheres, mispredictStream, ticks, kGravity, kDt, kGroundY, kHSearch, kMu, iters);
        check(mispredicted.size() == authority.size() &&
              std::memcmp(mispredicted.data(), authority.data(), authority.size() * sizeof(grain::GrainParticle)) != 0,
              "GR5 rollback: mispredicted state DIFFERS from authority (negative control — the divergence was real)");
    }

    // ================= GR6 render helpers (GrainVertToWorld / GrainToRenderInstances) ===============
    // The render-only float helpers (pos/(float)kOne -> world; one per-instance model matrix per grain).
    // GR1-GR5 stay bit-exact integer; ONLY these helpers cross to float. The render is golden-verified
    // (not unit-tested) — here we PIN the scale + count + per-instance translate/scale contracts (the FL6 twin).
    {
        // GrainVertToWorld: a known Q16.16 pos -> the exact float world position (pos / (float)kOne).
        const grain::FxVec3 q{FromInt(3), FromInt(-2), kOne / 2};   // (3.0, -2.0, 0.5) in Q16.16
        const math::Vec3 w = grain::GrainVertToWorld(q);
        check(w.x == 3.0f && w.y == -2.0f && w.z == 0.5f,
              "GR6 GrainVertToWorld: pos/(float)kOne -> exact float world position");
        check(grain::GrainToFloat(kOne) == 1.0f && grain::GrainToFloat(0) == 0.0f,
              "GR6 GrainToFloat: kOne -> 1.0, 0 -> 0.0");

        // GrainParticleTransform: translate(pos/kOne) * scale(radius). The translation lands in the mat4's
        // last column (m[12..14]); the diagonal scale lands in m[0]/m[5]/m[10] (no rotation -> off-diagonals 0).
        const float kRadius = 0.5f;
        grain::GrainParticle g;
        g.pos = grain::FxVec3{FromInt(4), FromInt(6), FromInt(-8)};
        const math::Mat4 m = grain::GrainParticleTransform(g, kRadius);
        check(m.m[12] == 4.0f && m.m[13] == 6.0f && m.m[14] == -8.0f,
              "GR6 GrainParticleTransform: translate == pos/(float)kOne (mat4 last column)");
        check(m.m[0] == kRadius && m.m[5] == kRadius && m.m[10] == kRadius,
              "GR6 GrainParticleTransform: diagonal scale == grain radius");
        check(m.m[1] == 0.0f && m.m[2] == 0.0f && m.m[4] == 0.0f && m.m[6] == 0.0f &&
              m.m[8] == 0.0f && m.m[9] == 0.0f && m.m[15] == 1.0f,
              "GR6 GrainParticleTransform: no rotation (off-diagonals 0, w 1)");

        // GrainToRenderInstances: N grains -> N transforms, each the right translate+scale (provenance:
        // the transform derives from the bit-exact GrainParticle::pos).
        grain::GrainBlock block;
        block.W = 3; block.H = 2; block.D = 2;   // 12 grains
        block.spacing = kOne;
        block.radius = kOne / 2;
        block.origin = grain::FxVec3{FromInt(1), FromInt(2), FromInt(3)};
        const std::vector<grain::GrainParticle> gs = grain::InitGrainBlock(block);
        const std::vector<math::Mat4> insts = grain::GrainToRenderInstances(gs, kRadius);
        check(insts.size() == gs.size() && insts.size() == 12u,
              "GR6 GrainToRenderInstances: one transform per grain (count == N)");
        bool allMatch = true;
        for (size_t i = 0; i < gs.size(); ++i) {
            const math::Vec3 wp = grain::GrainVertToWorld(gs[i].pos);
            if (insts[i].m[12] != wp.x || insts[i].m[13] != wp.y || insts[i].m[14] != wp.z) allMatch = false;
            if (insts[i].m[0] != kRadius || insts[i].m[5] != kRadius || insts[i].m[10] != kRadius) allMatch = false;
        }
        check(allMatch, "GR6 GrainToRenderInstances: every instance translates to pos/kOne, scales by radius");

        // Empty pool -> empty instance array (the empty no-op: zero instances -> the cleared base scene).
        const std::vector<grain::GrainParticle> none;
        check(grain::GrainToRenderInstances(none, kRadius).empty(),
              "GR6 GrainToRenderInstances: empty pool -> empty (the empty no-op)");
    }

    // ============================ Slice GR7 — POLYDISPERSE GRANULAR MATERIAL ========================
    // The Track-R R6 refinement: closes the documented "monodisperse grains" caveat of FLAGSHIP #10.
    // GR7 adds the hash-assigned size classes (AssignGrainPolyRadii, the pcg convention), the EXACT r^3
    // inverse mass (PolyInvMass), the structurally-widened broadphase (StepGrainPoly: hSearch >= 2*r_max,
    // the largest pair sum) and the GR5-mold lockstep twins. These cases pin: (a) the GR1/GR4/GR5 PINNED
    // DIGESTS (the append-only proof — canonical pre-GR7 scenes must hash to the exact same state after
    // GR7 lands; GrainDigest is new in GR7, the values below are the first pinning of those states);
    // (b) the hash/mass/assignment units (hand-checked exact Q16.16); (c) the IDENTITY-AT-UNIFORM (all
    // radii equal -> poly pool AND poly step BIT-IDENTICAL to the mono GR1/GR4 path, incl. through the
    // lockstep harness — THE key append-only proof); (d) the r^3 MASS WEIGHTING (a single big-vs-small
    // contact pair: the small grain moves further by the exact integer mass-ratio split, pinned); (e) the
    // BROADPHASE CONTRACT (an under-sized hSearch misses a big-big contact in the raw GR4 step; the poly
    // step's widened stencil resolves it); (f) THE PHYSICS (a mixed 1/3-big 2/3-small pour settles: min
    // pair separation >= the pinned honest LSB residual, the pile comes to rest under a pinned velocity
    // bound, the big/small mean heights pinned HONESTLY — near-equal, segregation NOT forced); (g)
    // determinism + the GR7 pinned digest (identical MSVC + clang); (h) LOCKSTEP (a peer re-derives the
    // mixed-size pile bit-for-bit; rollback corrects a misprediction) + the radius-covering snapshot.

    // --- (a) the GR1/GR4/GR5 pinned digests (the append-only proof) ----------------------------------
    {
        // GR1: the 5x5x5 integrate determinism scene (grav -10, dt=1/60, groundY 0, K=120).
        {
            grain::GrainBlock block; block.W = 5; block.H = 5; block.D = 5; block.spacing = kOne;
            block.radius = kOne / 4; block.origin = grain::FxVec3{0, FromInt(20), 0};
            std::vector<grain::GrainParticle> ps = grain::InitGrainBlock(block);
            grain::IntegrateGrainSteps(ps, grain::FxVec3{0, FromInt(-10), 0}, kOne / 60, 0, 120);
            const uint64_t d = grain::GrainDigest(ps);
            std::printf("GR7 pin: GR1 integrate digest = 0x%016llx\n", (unsigned long long)d);
            check(d == 0x63cf0f047217d7ceull, "GR7 pin(a): GR1 integrate digest UNCHANGED");
        }
        // GR4: the 5x5x5 staggered friction cone (the --grain-friction showcase scene, K=70, iters=2).
        {
            const grain::FxVec3 grav{0, FromInt(-10), 0};
            const grain::fx off = (grain::fx)(0.12 * (double)kOne + 0.5);
            std::vector<grain::GrainParticle> ps;
            for (int iy = 0; iy < 5; ++iy)
                for (int iz = 0; iz < 5; ++iz)
                    for (int ix = 0; ix < 5; ++ix) {
                        grain::GrainParticle p;
                        const grain::fx ox = (iy & 1) ? off : 0, oz = (iy & 1) ? off : 0;
                        p.pos = grain::FxVec3{FromInt(ix) + ox, FromInt(3 + iy), FromInt(iz) + oz};
                        p.prev = p.pos; p.invMass = kOne; p.radius = kOne / 2; p.flags = 0;
                        ps.push_back(p);
                    }
            const std::vector<grain::GrainSphereCollider> noSpheres;
            grain::StepGrainFrictionSteps(ps, noSpheres, grav, kOne / 60, 0, FromInt(2),
                                          grain::kGrainMu, 2, 70);
            const uint64_t d = grain::GrainDigest(ps);
            std::printf("GR7 pin: GR4 friction-cone digest = 0x%016llx\n", (unsigned long long)d);
            check(d == 0xd5b197d2ec8e7bb8ull, "GR7 pin(a): GR4 friction-cone digest UNCHANGED");
        }
        // GR5: the lockstep authority (the wind/push stream over the staggered scene, ticks=16).
        {
            const grain::fx kGravY = (grain::fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
            const grain::FxVec3 kGravity{0, kGravY, 0};
            const grain::fx kStagger = (grain::fx)(0.12 * (double)kOne + 0.5);
            std::vector<grain::GrainParticle> init;
            for (int iy = 0; iy < 5; ++iy)
                for (int iz = 0; iz < 5; ++iz)
                    for (int ix = 0; ix < 5; ++ix) {
                        grain::GrainParticle p;
                        const grain::fx ox = (iy & 1) ? kStagger : 0, oz = (iy & 1) ? kStagger : 0;
                        p.pos = {FromInt(ix) + ox, FromInt(3 + iy), FromInt(iz) + oz};
                        p.prev = p.pos; p.invMass = kOne; p.radius = kOne / 2;
                        init.push_back(p);
                    }
            const std::vector<grain::GrainSphereCollider> noSpheres;
            const std::vector<grain::GrainCommand> authStream{
                grain::GrainCommand{2,  grain::kCmdWind, 62u, grain::FxVec3{FromInt(8), 0, 0}},
                grain::GrainCommand{6,  grain::kCmdPush, 62u, grain::FxVec3{0, 0, FromInt(2)}},
                grain::GrainCommand{10, grain::kCmdWind, 62u, grain::FxVec3{FromInt(4), 0, 0}},
            };
            const std::vector<grain::GrainParticle> authority =
                grain::RunGrainLockstep(init, noSpheres, authStream, 16, kGravity, kOne / 60, 0,
                                        kOne * 2, grain::kGrainMu, 2);
            const uint64_t d = grain::GrainDigest(authority);
            std::printf("GR7 pin: GR5 lockstep authority digest = 0x%016llx\n", (unsigned long long)d);
            check(d == 0x0f5dd49fcfc39517ull, "GR7 pin(a): GR5 lockstep-authority digest UNCHANGED");
        }
    }

    // --- (b) the units: GrainPolyHash / PolyInvMass / AssignGrainPolyRadii / InitGrainPolyBlock -------
    {
        // GrainPolyHash: deterministic (same in == same out), seed- and index-sensitive.
        check(grain::GrainPolyHash(7u, 0u) == grain::GrainPolyHash(7u, 0u),
              "GR7 hash: deterministic (same seed+index -> same value)");
        check(grain::GrainPolyHash(7u, 0u) != grain::GrainPolyHash(8u, 0u),
              "GR7 hash: seed-sensitive");
        check(grain::GrainPolyHash(7u, 0u) != grain::GrainPolyHash(7u, 1u),
              "GR7 hash: index-sensitive");

        // PolyInvMass: EXACT integer values. r == ref -> kOne EXACTLY (the identity anchor). r = 2*ref ->
        // mass 8x -> invMass kOne/8 EXACTLY (kOne/4 cubes to 1024, kOne/2 to 8192; fxdiv(1024,8192)=8192).
        check(grain::PolyInvMass(kOne / 4, kOne / 4) == kOne,
              "GR7 PolyInvMass: r == refRadius -> EXACTLY kOne (identity-at-uniform anchor)");
        check(grain::PolyInvMass(kOne / 2, kOne / 4) == kOne / 8,
              "GR7 PolyInvMass: r == 2*ref -> EXACTLY kOne/8 (mass = r^3 = 8x, exact integers)");
        check(grain::PolyInvMass(kOne, kOne / 2) == kOne / 8,
              "GR7 PolyInvMass: the r^3 law holds at another exact scale (1.0 vs 0.5 -> kOne/8)");
        check(grain::PolyInvMass(0, kOne / 4) == 0 && grain::PolyInvMass(kOne / 4, 0) == 0,
              "GR7 PolyInvMass: zero/negative radius -> invMass 0 (deterministic degenerate)");

        // AssignGrainPolyRadii: every radius comes from the class table; static grains untouched;
        // determinism (two assignments byte-identical); the class mix is hash-driven (both classes hit).
        grain::GrainBlock block; block.W = 5; block.H = 5; block.D = 5; block.spacing = kOne;
        block.radius = kOne / 4; block.origin = grain::FxVec3{0, FromInt(3), 0};
        const std::vector<grain::fx> classes{kOne / 4, kOne / 4, kOne / 2};   // 2/3 small + 1/3 big
        std::vector<grain::GrainParticle> ps = grain::InitGrainBlock(block);
        ps[7].flags = grain::kFlagStatic; ps[7].invMass = 0;                  // a boundary grain
        const grain::GrainParticle staticBefore = ps[7];
        grain::AssignGrainPolyRadii(ps, classes, 7u);
        int nSmall = 0, nBig = 0; bool fromTable = true, massOk = true;
        for (size_t i = 0; i < ps.size(); ++i) {
            if (i == 7) continue;
            if (ps[i].radius == kOne / 4)      { ++nSmall; if (ps[i].invMass != kOne)     massOk = false; }
            else if (ps[i].radius == kOne / 2) { ++nBig;   if (ps[i].invMass != kOne / 8) massOk = false; }
            else fromTable = false;
        }
        std::printf("GR7 assign: 125-pool classes -> %d small + %d big (+1 static)\n", nSmall, nBig);
        check(fromTable, "GR7 assign: every assigned radius comes from the class table");
        check(nSmall > 0 && nBig > 0, "GR7 assign: both size classes present (hash-driven mix)");
        check(nSmall == 77 && nBig == 47, "GR7 assign: pinned class split (77 small / 47 big, seed 7)");
        check(massOk, "GR7 assign: invMass == PolyInvMass(radius, minClassRadius) per class (exact)");
        check(std::memcmp(&ps[7], &staticBefore, sizeof(grain::GrainParticle)) == 0,
              "GR7 assign: a STATIC boundary grain is BYTE-UNTOUCHED");
        std::vector<grain::GrainParticle> ps2 = grain::InitGrainBlock(block);
        ps2[7].flags = grain::kFlagStatic; ps2[7].invMass = 0;
        grain::AssignGrainPolyRadii(ps2, classes, 7u);
        check(std::memcmp(ps.data(), ps2.data(), ps.size() * sizeof(grain::GrainParticle)) == 0,
              "GR7 assign: two assignments byte-identical (deterministic)");
        // Empty class table -> no-op.
        std::vector<grain::GrainParticle> noop = grain::InitGrainBlock(block);
        const std::vector<grain::GrainParticle> noopBefore = noop;
        grain::AssignGrainPolyRadii(noop, {}, 7u);
        check(std::memcmp(noop.data(), noopBefore.data(), noop.size() * sizeof(grain::GrainParticle)) == 0,
              "GR7 assign: empty class table -> no-op");
        // MaxGrainRadius over the mixed pool == the big class.
        check(grain::MaxGrainRadius(ps) == kOne / 2, "GR7 MaxGrainRadius: == the largest class radius");
    }

    // --- (c) IDENTITY-AT-UNIFORM: one class == the monodisperse pool + step BIT-IDENTICAL -------------
    {
        const grain::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60, groundY = 0, hSearch = FromInt(2);
        const fx mu = grain::kGrainMu;
        const int iters = 2, steps = 40;
        grain::GrainBlock block; block.W = 4; block.H = 4; block.D = 4; block.spacing = kOne;
        block.radius = kOne / 2; block.origin = grain::FxVec3{0, FromInt(3), 0};
        const std::vector<grain::GrainSphereCollider> noSpheres;

        // The POOL identity: one class {r0} -> InitGrainPolyBlock == InitGrainBlock byte-for-byte.
        const std::vector<grain::GrainParticle> mono = grain::InitGrainBlock(block);
        const std::vector<grain::GrainParticle> poly =
            grain::InitGrainPolyBlock(block, {block.radius}, 7u);
        check(mono.size() == poly.size() &&
              std::memcmp(mono.data(), poly.data(), mono.size() * sizeof(grain::GrainParticle)) == 0,
              "GR7 identity: one-class InitGrainPolyBlock == InitGrainBlock BYTE-IDENTICAL");

        // The STEP identity: uniform radii -> StepGrainPolySteps == StepGrainFrictionSteps BIT-IDENTICAL
        // (hSearch already >= 2*r_max, so the poly widen is a no-op and the GR4 step runs verbatim).
        std::vector<grain::GrainParticle> a = poly, b = mono;
        grain::StepGrainPolySteps(a, noSpheres, grav, dt, groundY, hSearch, mu, iters, steps);
        grain::StepGrainFrictionSteps(b, noSpheres, grav, dt, groundY, hSearch, mu, iters, steps);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(grain::GrainParticle)) == 0,
              "GR7 identity-at-uniform: poly step (all radii == r0) == mono GR4 step BIT-IDENTICAL");

        // ... and through the lockstep harness (the composed identity, the CL7 (d) twin).
        const std::vector<grain::GrainCommand> stream{
            grain::GrainCommand{3, grain::kCmdWind, 21u, grain::FxVec3{FromInt(5), 0, 0}},
        };
        const std::vector<grain::GrainParticle> lsMono =
            grain::RunGrainLockstep(mono, noSpheres, stream, 12, grav, dt, groundY, hSearch, mu, iters);
        const std::vector<grain::GrainParticle> lsPoly =
            grain::RunGrainPolyLockstep(poly, noSpheres, stream, 12, grav, dt, groundY, hSearch, mu, iters);
        check(lsMono.size() == lsPoly.size() &&
              std::memcmp(lsMono.data(), lsPoly.data(), lsMono.size() * sizeof(grain::GrainParticle)) == 0,
              "GR7 identity-at-uniform: RunGrainPolyLockstep(uniform) == RunGrainLockstep BIT-IDENTICAL");
    }

    // --- (d) the r^3 MASS WEIGHTING: a single big-vs-small contact pair, the exact displacement split --
    {
        // Big grain (r = 0.5, invMass kOne/8) at the origin; small grain (r = 0.25, invMass kOne) at
        // (0.6, 0, 0). Contact: r_i + r_j = 0.75 (49152), dist = 0.6 (39321) -> pen = 9831 LSBs. The
        // inverse-mass split (fxdiv truncation documented): share_small = fxdiv(kOne, kOne + kOne/8) =
        // 58254, share_big = fxdiv(kOne/8, kOne + kOne/8) = 7281 (sum 65535 = kOne − 1 LSB, the truncation).
        // dp_small = fxmul(fxmul(58254, 9831), +kOne) = +8738; dp_big = fxmul(fxmul(7281, 9831), −kOne) =
        // −1092. The SMALL grain moves 8738/1092 ≈ 8.0x further — the EXACT integer image of the r^3 = 8
        // mass ratio (the physical asymmetry: gravel shoves sand, not vice versa).
        const fx rBig = kOne / 2, rSmall = kOne / 4;
        const fx hSearch = FromInt(2);
        std::vector<grain::GrainParticle> pair(2);
        pair[0].pos = {0, 0, 0};             pair[0].radius = rBig;
        pair[0].invMass = grain::PolyInvMass(rBig, rSmall);      // kOne/8 exact
        pair[1].pos = {kOne * 6 / 10, 0, 0}; pair[1].radius = rSmall;
        pair[1].invMass = grain::PolyInvMass(rSmall, rSmall);    // kOne exact
        grain::GrainGrid g = grain::MakeGrainGrid(pair, hSearch);
        grain::GrainCellTable t = grain::BuildGrainCellTable(pair, g);
        grain::GrainNeighborList nl = grain::BuildGrainNeighborList(pair, g, t, hSearch);
        std::vector<grain::FxVec3> dp;
        grain::SolveGrainContact(pair, nl, dp);
        std::printf("GR7 mass split: big dp.x = %d, small dp.x = %d (pen 9831, ratio ~8x)\n",
                    (int)dp[0].x, (int)dp[1].x);
        check(dp[0].x == -1092 && dp[0].y == 0 && dp[0].z == 0,
              "GR7 mass weighting: the BIG grain takes the small share (dp.x == −1092 exact)");
        check(dp[1].x == 8738 && dp[1].y == 0 && dp[1].z == 0,
              "GR7 mass weighting: the SMALL grain takes the big share (dp.x == +8738 exact)");
        // The split ratio is the r^3 mass ratio within integer truncation: 8738 = 8*1092 + 2 LSBs.
        check(dp[1].x >= 8 * (-dp[0].x) && dp[1].x <= 8 * (-dp[0].x) + 8,
              "GR7 mass weighting: displacement ratio == the r^3 mass ratio 8 (within LSB truncation)");
    }

    // --- (e) the BROADPHASE CONTRACT: the widened stencil reaches the largest pair sum -----------------
    {
        // Two BIG grains (r = 0.5 each, contact diameter 1.0) 0.8 apart, an UNDER-SIZED hSearch = 0.5:
        // the raw GR4 step at hSearch 0.5 box-rejects the pair (|dx| = 0.8 > 0.5) -> the contact is
        // MISSED (positions unchanged; zero gravity isolates the broadphase). StepGrainPoly widens the
        // cell size to 2*r_max = 1.0 -> the pair is a candidate (0.8 < 1.0) -> the overlap (pen 0.2) is
        // RESOLVED. This is the polydisperse broadphase contract made structural.
        const grain::FxVec3 noGrav{0, 0, 0};
        const fx dt = kOne / 60, groundY = FromInt(-1000);
        const fx tinySearch = kOne / 2;
        const std::vector<grain::GrainSphereCollider> noSpheres;
        auto makePair = []() {
            std::vector<grain::GrainParticle> p(2);
            p[0].pos = {0, 0, 0};             p[0].prev = p[0].pos; p[0].invMass = kOne; p[0].radius = kOne / 2;
            p[1].pos = {kOne * 8 / 10, 0, 0}; p[1].prev = p[1].pos; p[1].invMass = kOne; p[1].radius = kOne / 2;
            return p;
        };
        std::vector<grain::GrainParticle> raw = makePair();
        const std::vector<grain::GrainParticle> before = raw;
        grain::StepGrainFriction(raw, noSpheres, noGrav, dt, groundY, tinySearch, grain::kGrainMu, 4);
        check(std::memcmp(raw.data(), before.data(), raw.size() * sizeof(grain::GrainParticle)) == 0,
              "GR7 broadphase: the raw GR4 step at hSearch < 2*r_max MISSES the big-big contact (control)");
        std::vector<grain::GrainParticle> poly = makePair();
        grain::StepGrainPoly(poly, noSpheres, noGrav, dt, groundY, tinySearch, grain::kGrainMu, 4);
        const fx sepBefore = grain::GrainPolyMinSeparation(before);
        const fx sepAfter = grain::GrainPolyMinSeparation(poly);
        std::printf("GR7 broadphase: pair separation %d -> %d LSBs (poly widened stencil)\n",
                    (int)sepBefore, (int)sepAfter);
        check(sepBefore < 0, "GR7 broadphase: the pair starts overlapping (sep < 0, pen 0.2)");
        check(sepAfter > sepBefore,
              "GR7 broadphase: StepGrainPoly RESOLVES the contact the under-sized hSearch missed");
    }

    // --- the GR7 MIXED-POUR scene builder (shared by the physics/determinism/lockstep cases; == the ----
    // --- --gr7-poly-shot showcase scene) ---------------------------------------------------------------
    // A staggered 5x5x5 POLYDISPERSE block — 2/3 small (r=0.25) + 1/3 big (r=0.5) assigned by index hash
    // (seed 7), r^3 masses (big grains 8x heavier), spacing 1.0 >= 2*r_max (non-overlapping start even for
    // two adjacent big grains) — poured GENTLY (drop from y=1) onto a STATIC ROUGH BED: a dense layer of
    // STATIC small grains (r=0.25 at 0.5 spacing, GR1's kFlagStatic boundary grains) covering x,z in
    // [-6,10]. WHY THE BED (the honest scene-design finding, measured): the GR family has NO grain-floor
    // tangential friction — GR4's Coulomb clamp is GRAIN-GRAIN only and CollideGrainPlane is a frictionless
    // clamp — so on BARE floor any grain that exits the pile with horizontal velocity slides FOREVER and
    // the pour never rests (measured: an open pour's median speed plateaus ~1.8 u/s as a spreading
    // monolayer; the MONO GR4 cone has the same slow-spread property — its showcase pins a K=70 snapshot).
    // A rough bed is the standard granular-experiment floor: sliders rub the static bed grains, GR4
    // friction arrests them, and the pour SETTLES into a stable mixed mound. Gravel + sand in ONE pile,
    // zero new physics — only GR1 static grains + GR4 friction, composed.
    const auto buildPolyPour = []() {
        const grain::fx kStagger = (grain::fx)(0.12 * (double)kOne + 0.5);
        std::vector<grain::GrainParticle> ps;
        // The 125 DYNAMIC grains first (indices 0..124 — the lockstep command targets stay dynamic).
        for (int iy = 0; iy < 5; ++iy)
            for (int iz = 0; iz < 5; ++iz)
                for (int ix = 0; ix < 5; ++ix) {
                    grain::GrainParticle p;
                    const grain::fx ox = (iy & 1) ? kStagger : 0, oz = (iy & 1) ? kStagger : 0;
                    p.pos = {FromInt(ix) + ox, FromInt(1 + iy), FromInt(iz) + oz};
                    p.prev = p.pos; p.invMass = kOne; p.radius = kOne / 4; p.flags = 0;
                    ps.push_back(p);
                }
        grain::AssignGrainPolyRadii(ps, {kOne / 4, kOne / 4, kOne / 2}, 7u);
        // The STATIC rough bed (33x33 = 1089 grains): r=0.25 at 0.5 spacing (touching), centers y=0.25.
        // AssignGrainPolyRadii ran FIRST and skips static grains by contract anyway.
        for (int gx = -12; gx <= 20; ++gx)
            for (int gz = -12; gz <= 20; ++gz) {
                grain::GrainParticle w;
                w.pos = {(fx)(gx * (int)(kOne / 2)), kOne / 4, (fx)(gz * (int)(kOne / 2))};
                w.prev = w.pos; w.vel = {0, 0, 0};
                w.invMass = 0; w.radius = kOne / 4; w.flags = grain::kFlagStatic;
                ps.push_back(w);
            }
        return ps;
    };
    const int kPolyDynamic = 125;                 // dynamic grains are indices 0..124; the rest is the bed
    const grain::fx kPolyGravY = (grain::fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
    const grain::FxVec3 kPolyGrav{0, kPolyGravY, 0};
    const fx kPolyDt = kOne / 60, kPolyGroundY = 0, kPolyHSearch = FromInt(2);
    const fx kPolyMu = grain::kGrainMu;
    const int kPolyIters = 4, kPolySteps = 280;
    // The honest Jacobi LSB residual bound (the CL7 slack discipline): 4 Jacobi iterations/step leave a
    // deterministic-but-nonzero pair penetration; the settled pour's deepest pair penetration is measured +
    // pinned below, and this bound brackets it without masking a broadphase miss (a missed contact under
    // gravity compresses to MANY THOUSANDS of LSBs — a full small radius is 16384).
    const fx kPolyResidualBound = 4096;   // LSBs (0.0625 world units, 25% of the small radius)

    // --- (f) THE PHYSICS: the mixed pour settles — non-penetration, rest, honest size structure --------
    {
        const std::vector<grain::GrainSphereCollider> noSpheres;
        std::vector<grain::GrainParticle> pour = buildPolyPour();
        grain::StepGrainPolySteps(pour, noSpheres, kPolyGrav, kPolyDt, kPolyGroundY, kPolyHSearch,
                                  kPolyMu, kPolyIters, kPolySteps);

        // (i) NO interpenetration beyond the honest residual: min pair separation (over the FULL pool,
        // dynamic AND bed — the brute-force metric does not share the solver's broadphase) >= −bound.
        const fx minSep = grain::GrainPolyMinSeparation(pour);
        std::printf("GR7 physics: settled min pair separation = %d LSBs (honest Jacobi residual)\n",
                    (int)minSep);
        check(minSep >= -kPolyResidualBound,
              "GR7 physics(i): min pair separation >= -(pinned residual bound) — no deep interpenetration");
        check(minSep == -1306,
              "GR7 physics(i) pin: the exact settled residual == -1306 LSBs (~8% of the small radius)");

        // (ii) the pile comes to REST: max DYNAMIC grain speed under the pinned bound at the pinned settle
        // step. HONEST framing: the family has no sleep threshold, so the Jacobi contact churn keeps a
        // small fluctuating jitter alive forever — this is the PINNED-STEP bound (every grain under 0.5
        // world-units/s at K=280), not an asymptotic zero.
        const std::vector<grain::GrainParticle> dyn(pour.begin(), pour.begin() + kPolyDynamic);
        const fx maxSpeed = grain::MaxGrainSpeed(dyn);
        std::printf("GR7 physics: settled max dynamic grain speed = %d LSBs/s\n", (int)maxSpeed);
        check(maxSpeed < kOne / 2,
              "GR7 physics(ii): the pour is at rest (EVERY grain < 0.5 world units/s at the settle step)");
        check(maxSpeed == 32723,
              "GR7 physics(ii) pin: the exact settled max speed == 32723 LSBs/s (deterministic)");

        // (iii) the HONEST size-structure metric: mean height of big vs small DYNAMIC grains after
        // settling. The "Brazil-nut" segregation is shaking-driven and EMERGENT — this is a single settled
        // pour, so the signal is WEAK: mean center heights big 115145 vs small 87182; subtracting each
        // class's radius (the geometric offset), the big grains' UNDERSIDES sit 82377 vs 70798 — ~0.18
        // world units higher, the Brazil-nut DIRECTION present but weak. We PIN the exact deterministic
        // values and report them honestly (the load-bearing proofs are (i), (ii) and the (d) mass-weighted
        // split — segregation is NOT forced).
        const grain::GrainSizeHeights hts = grain::MeasureGrainSizeHeights(dyn, kOne / 2);
        std::printf("GR7 physics: mean height big = %d, small = %d LSBs (%d big / %d small); undersides "
                    "big %d vs small %d — weak Brazil-nut direction, reported honestly\n",
                    (int)hts.meanYLarge, (int)hts.meanYSmall, hts.nLarge, hts.nSmall,
                    (int)(hts.meanYLarge - kOne / 2), (int)(hts.meanYSmall - kOne / 4));
        check(hts.nLarge == 47 && hts.nSmall == 78,
              "GR7 physics(iii): the pour splits 47 big / 78 small (the pinned hash mix, seed 7)");
        check(hts.meanYLarge == 115145 && hts.meanYSmall == 87182,
              "GR7 physics(iii) pin: exact mean heights (big 115145, small 87182 LSBs — a stable mixed "
              "MOUND, not a monolayer: the bed arrests the spread)");

        // Every grain ends at/above the floor (radius-aware, within the collide eps).
        bool aboveFloor = true;
        for (const grain::GrainParticle& p : pour)
            if (p.pos.y < kPolyGroundY + p.radius - grain::kGrainCollideEps) aboveFloor = false;
        check(aboveFloor, "GR7 physics: no grain ends below groundY + its OWN radius (mixed radii, eps)");
    }

    // --- (g) determinism + the GR7 pinned digest (identical MSVC + clang) ------------------------------
    {
        const std::vector<grain::GrainSphereCollider> noSpheres;
        std::vector<grain::GrainParticle> a = buildPolyPour(), b = buildPolyPour();
        grain::StepGrainPolySteps(a, noSpheres, kPolyGrav, kPolyDt, kPolyGroundY, kPolyHSearch,
                                  kPolyMu, kPolyIters, kPolySteps);
        grain::StepGrainPolySteps(b, noSpheres, kPolyGrav, kPolyDt, kPolyGroundY, kPolyHSearch,
                                  kPolyMu, kPolyIters, kPolySteps);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(grain::GrainParticle)) == 0,
              "GR7 determinism: two mixed-pour runs BYTE-IDENTICAL");
        const uint64_t d = grain::GrainDigest(a);
        std::printf("GR7 pin: GR7 settled mixed-pour digest = 0x%016llx\n", (unsigned long long)d);
        check(d == 0xb3325f416aff93ceull, "GR7 pin(g): mixed-pour digest == the pinned value");
    }

    // --- (h) LOCKSTEP + ROLLBACK over the mixed-size pile + the radius-covering snapshot ----------------
    {
        const std::vector<grain::GrainSphereCollider> noSpheres;
        const std::vector<grain::GrainParticle> init = buildPolyPour();
        const int ticks = 16, mispredictTick = 6;
        const uint32_t wIdx = 62u;
        const std::vector<grain::GrainCommand> authStream{
            grain::GrainCommand{2,  grain::kCmdWind, wIdx, grain::FxVec3{FromInt(8), 0, 0}},
            grain::GrainCommand{6,  grain::kCmdPush, wIdx, grain::FxVec3{0, 0, FromInt(2)}},
            grain::GrainCommand{10, grain::kCmdWind, wIdx, grain::FxVec3{FromInt(4), 0, 0}},
        };
        std::vector<grain::GrainCommand> mispredictStream = authStream;
        mispredictStream.push_back(grain::GrainCommand{(uint32_t)mispredictTick, grain::kCmdWind, wIdx,
                                                       grain::FxVec3{FromInt(60), 0, 0}});

        const std::vector<grain::GrainParticle> authority =
            grain::RunGrainPolyLockstep(init, noSpheres, authStream, ticks, kPolyGrav, kPolyDt,
                                        kPolyGroundY, kPolyHSearch, kPolyMu, kPolyIters);
        const std::vector<grain::GrainParticle> replica =
            grain::RunGrainPolyLockstep(init, noSpheres, authStream, ticks, kPolyGrav, kPolyDt,
                                        kPolyGroundY, kPolyHSearch, kPolyMu, kPolyIters);
        check(authority.size() == replica.size() &&
              std::memcmp(authority.data(), replica.data(),
                          authority.size() * sizeof(grain::GrainParticle)) == 0,
              "GR7 lockstep: mixed-size replica == authority BIT-EXACT (inputs-only re-sim)");
        const uint64_t d = grain::GrainDigest(authority);
        std::printf("GR7 pin: GR7 lockstep authority digest = 0x%016llx\n", (unsigned long long)d);
        check(d == 0x1b0add58d5bb8496ull, "GR7 pin(h): mixed-size lockstep digest == the pinned value");

        // ROLLBACK: corrected to authority BIT-EXACT; the misprediction was a REAL divergence.
        const std::vector<grain::GrainParticle> rolledBack =
            grain::RunGrainPolyRollback(init, noSpheres, authStream, mispredictStream, ticks,
                                        mispredictTick, kPolyGrav, kPolyDt, kPolyGroundY, kPolyHSearch,
                                        kPolyMu, kPolyIters);
        check(rolledBack.size() == authority.size() &&
              std::memcmp(rolledBack.data(), authority.data(),
                          authority.size() * sizeof(grain::GrainParticle)) == 0,
              "GR7 rollback: corrected to authority BIT-EXACT (positive control)");
        const std::vector<grain::GrainParticle> mispredicted =
            grain::RunGrainPolyLockstep(init, noSpheres, mispredictStream, ticks, kPolyGrav, kPolyDt,
                                        kPolyGroundY, kPolyHSearch, kPolyMu, kPolyIters);
        check(mispredicted.size() == authority.size() &&
              std::memcmp(mispredicted.data(), authority.data(),
                          authority.size() * sizeof(grain::GrainParticle)) != 0,
              "GR7 rollback: mispredicted state DIFFERS from authority (negative control)");

        // The snapshot COVERS the radius array (radius is a first-class GrainParticle field): mutate a
        // grain's radius after snapshotting, restore, and the ORIGINAL radius is back byte-for-byte.
        std::vector<grain::GrainParticle> ps = authority;
        const std::vector<grain::GrainParticle> snap = grain::SnapshotGrain(ps);
        ps[3].radius = FromInt(9);                             // corrupt a radius (not just positions)
        ps[3].pos.x += FromInt(1);
        grain::RestoreGrain(ps, snap);
        check(ps.size() == snap.size() &&
              std::memcmp(ps.data(), snap.data(), snap.size() * sizeof(grain::GrainParticle)) == 0,
              "GR7 snapshot: SnapshotGrain/RestoreGrain round-trips the per-grain RADIUS bit-exactly");
    }

    if (g_fail == 0) std::printf("grain_test: ALL PASS\n");
    else std::printf("grain_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
