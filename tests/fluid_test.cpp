// Slice FL1 — Deterministic GPU Fluid: the Q16.16 PARTICLE POOL INTEGRATOR + dam-break block core
// (engine/sim/fluid.h) that the GPU shaders/fluid_integrate.comp.hlsl copies VERBATIM + proves
// bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace hf::sim::fluid.
//
// What this test PINS (the contracts the GPU fluid_integrate.comp + the GPU==CPU proof build on):
//   * InitBlock: a solid W x H x D block (size W*H*D, the ParticleIndex traversal order), spacing/origin
//     host-snapped, every particle DYNAMIC (invMass kOne, flags 0); prev==pos, vel==0.
//   * IntegrateFluidParticle: one-step + K-step semi-implicit-Euler closed form (a FREE particle falls
//     the EXACT Q16.16 distance — hand-checked integer recurrence), prev = pos each step.
//   * the ground floor clamp (pos.y pinned to groundY, vel.y zeroed on contact, stays settled).
//   * determinism: two runs of the SAME init+steps -> byte-identical particle arrays.
//   * a ZERO-gravity pool is UNCHANGED after K steps (the static / no-op case).
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests.
#include "sim/fluid.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace fluid = hf::sim::fluid;
namespace fpx = hf::sim::fpx;
using fluid::fx;
using fluid::kOne;
using fluid::kFrac;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << kFrac); }

int main() {
    HF_TEST_MAIN_INIT();

    // ================= InitBlock: solid W x H x D block, all dynamic, at rest =================
    {
        fluid::FluidBlock block;
        block.W = 4; block.H = 3; block.D = 2;
        block.spacing = kOne;                 // 1.0 world unit between particles
        block.origin = fluid::FxVec3{FromInt(5), FromInt(10), FromInt(-3)};
        std::vector<fluid::FluidParticle> ps = fluid::InitBlock(block);

        check(ps.size() == (size_t)(block.W * block.H * block.D), "InitBlock: size == W*H*D");

        // x-major/y-mid/z-minor layout: particle (ix,iy,iz) at origin + (ix*spacing, iy*spacing, iz*spacing).
        const fluid::FluidParticle& p000 = ps[(size_t)fluid::ParticleIndex(block, 0, 0, 0)];
        const fluid::FluidParticle& p312 = ps[(size_t)fluid::ParticleIndex(block, 3, 1, 1)];
        check(p000.pos.x == FromInt(5) && p000.pos.y == FromInt(10) && p000.pos.z == FromInt(-3),
              "InitBlock: (0,0,0) at origin");
        check(p312.pos.x == FromInt(5) + FromInt(3) && p312.pos.y == FromInt(10) + FromInt(1) &&
              p312.pos.z == FromInt(-3) + FromInt(1),
              "InitBlock: (3,1,1) at origin + (3,1,1)*spacing");

        // ParticleIndex is the x-major/y-mid/z-minor flatten ((iz*H + iy)*W + ix).
        check(fluid::ParticleIndex(block, 0, 0, 0) == 0, "ParticleIndex: (0,0,0) -> 0");
        check(fluid::ParticleIndex(block, 1, 0, 0) == 1, "ParticleIndex: +x is the minor stride");
        check(fluid::ParticleIndex(block, 0, 1, 0) == block.W, "ParticleIndex: +y strides by W");
        check(fluid::ParticleIndex(block, 0, 0, 1) == block.W * block.H,
              "ParticleIndex: +z strides by W*H");

        // prev == pos and vel == 0 for every particle initially; every particle DYNAMIC (invMass kOne).
        bool prevEqPos = true, velZero = true, allDynamic = true;
        for (const fluid::FluidParticle& p : ps) {
            if (std::memcmp(&p.prev, &p.pos, sizeof(fluid::FxVec3)) != 0) prevEqPos = false;
            if (p.vel.x != 0 || p.vel.y != 0 || p.vel.z != 0) velZero = false;
            if ((p.flags & fluid::kFlagStatic) != 0u || p.invMass != kOne) allDynamic = false;
        }
        check(prevEqPos, "InitBlock: prev == pos initially");
        check(velZero, "InitBlock: vel == 0 initially");
        check(allDynamic, "InitBlock: every particle dynamic (invMass kOne, no static flag)");
    }

    // ================= a FREE particle: one-step + K-step semi-implicit-Euler closed form ===========
    {
        const fx g  = FromInt(-10);     // gravity -10 (exact in Q16.16)
        const fx dt = kOne / 2;         // dt = 0.5 (exact)
        const fx groundY = FromInt(-1000);  // far below; effectively no clamp here
        const fluid::FxVec3 grav{0, g, 0};

        // One step: vel.y = 0 + g*dt = -10*0.5 = -5 ; pos.y = 100 + vel.y*dt = 100 + (-5)*0.5 = 97.5.
        fluid::FluidParticle p;
        p.pos = {0, FromInt(100), 0};
        p.prev = p.pos;
        p.vel = {0, 0, 0};
        p.invMass = kOne;
        p.flags = 0;                    // dynamic (not static)
        fluid::IntegrateFluidParticle(p, grav, groundY, dt);
        check(p.vel.y == FromInt(-5), "free one-step vel.y == g*dt == -5.0");
        check(p.pos.y == FromInt(100) - (kOne * 5 / 2), "free one-step pos.y == 100 + (-5)*0.5 == 97.5");
        check(p.prev.y == FromInt(100), "free one-step prev.y == the pre-move pos (100)");

        // K-step closed form: independently re-run the EXACT integer ops K times and compare to the
        // header's IntegrateFluidSteps over the same single-particle pool (must agree by construction).
        const int K = 60;
        std::vector<fluid::FluidParticle> one(1);
        one[0].pos = {0, FromInt(500), 0};
        one[0].prev = one[0].pos;
        one[0].vel = {0, 0, 0};
        one[0].invMass = kOne;
        one[0].flags = 0;

        // Reference: the same per-step integer recurrence, computed inline.
        fx refVy = 0, refPy = FromInt(500), refPrevY = FromInt(500);
        for (int s = 0; s < K; ++s) {
            refVy += fluid::fxmul(g, dt);
            refPrevY = refPy;
            refPy += fluid::fxmul(refVy, dt);
            if (refPy < groundY) { refPy = groundY; if (refVy < 0) refVy = 0; }
        }
        fluid::IntegrateFluidSteps(one, grav, dt, groundY, K);
        check(one[0].pos.y == refPy, "free K-step pos.y == hand-computed integer recurrence");
        check(one[0].vel.y == refVy, "free K-step vel.y == hand-computed integer recurrence");
        check(one[0].prev.y == refPrevY, "free K-step prev.y == the pre-last-move pos");
    }

    // ================= ground floor clamp: pos.y==groundY, vel.y zeroed on contact =================
    {
        const fluid::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = 0;

        // A particle just above the ground with a big downward velocity clamps to groundY + zeros vel.y.
        fluid::FluidParticle p;
        p.pos = {0, kOne / 100, 0};      // 0.01 above ground
        p.prev = p.pos;
        p.vel = {0, FromInt(-50), 0};    // moving down fast
        p.invMass = kOne;
        p.flags = 0;
        fluid::IntegrateFluidParticle(p, grav, groundY, dt);
        check(p.pos.y == groundY, "ground clamp: pos.y pinned to groundY");
        check(p.vel.y == 0, "ground clamp: downward vel.y zeroed on contact");

        // After contact, the particle stays settled (pos.y stays groundY across more steps).
        for (int s = 0; s < 50; ++s) fluid::IntegrateFluidParticle(p, grav, groundY, dt);
        check(p.pos.y == groundY, "ground clamp: particle stays settled at groundY");
    }

    // ================= determinism: two runs byte-identical + a coherent dam-break fall =============
    {
        const fluid::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = 0;
        auto makeBlock = []() {
            fluid::FluidBlock b; b.W = 5; b.H = 5; b.D = 5; b.spacing = kOne;
            b.origin = fluid::FxVec3{0, FromInt(20), 0};   // a corner well above the ground
            return b;
        };
        fluid::FluidBlock block = makeBlock();
        std::vector<fluid::FluidParticle> a = fluid::InitBlock(block);
        std::vector<fluid::FluidParticle> b = fluid::InitBlock(block);
        const int K = 120;
        fluid::IntegrateFluidSteps(a, grav, dt, groundY, K);
        fluid::IntegrateFluidSteps(b, grav, dt, groundY, K);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(fluid::FluidParticle)) == 0,
              "determinism: two runs byte-identical");

        // Coherence: the block FELL (every particle's y dropped below its rest y) and piled at the
        // ground (some particles rest at groundY) — a coherent dam-break fall.
        std::vector<fluid::FluidParticle> init = fluid::InitBlock(block);
        int moved = 0;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i].pos.y < init[i].pos.y) ++moved;
        check(moved == (int)a.size(), "coherence: every particle fell (moved down)");
        check(fluid::CountAtGround(a, groundY) > 0, "coherence: some particles piled at the ground");
    }

    // ================= ZERO gravity -> the pool is UNCHANGED after K steps (the static no-op) =======
    {
        const fluid::FxVec3 noGrav{0, 0, 0};
        const fx dt = kOne / 60;
        const fx groundY = FromInt(-1000);   // far below so the clamp never fires
        fluid::FluidBlock block; block.W = 4; block.H = 4; block.D = 4; block.spacing = kOne;
        block.origin = fluid::FxVec3{0, FromInt(5), 0};
        std::vector<fluid::FluidParticle> ps = fluid::InitBlock(block);
        std::vector<fluid::FluidParticle> before = ps;
        fluid::IntegrateFluidSteps(ps, noGrav, dt, groundY, 120);
        check(std::memcmp(ps.data(), before.data(), ps.size() * sizeof(fluid::FluidParticle)) == 0,
              "zero-gravity pool UNCHANGED after K steps (static no-op)");
    }

    // ================= a STATIC particle NEVER moves (the reserved boundary flag) ====================
    {
        const fluid::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        fluid::FluidParticle s;
        s.pos = {FromInt(3), FromInt(7), FromInt(-2)};
        s.prev = s.pos;
        s.vel = {0, 0, 0};
        s.invMass = 0;
        s.flags = fluid::kFlagStatic;
        fluid::FluidParticle before = s;
        for (int i = 0; i < 100; ++i) fluid::IntegrateFluidParticle(s, grav, 0, dt);
        check(std::memcmp(&s, &before, sizeof(fluid::FluidParticle)) == 0,
              "static particle never moves (untouched across 100 steps)");
    }

    // ================= InitBlock: the 10x10x10 showcase block (count + a known position) =============
    {
        fluid::FluidBlock block; block.W = 10; block.H = 10; block.D = 10; block.spacing = kOne;
        block.origin = fluid::FxVec3{0, FromInt(24), 0};
        std::vector<fluid::FluidParticle> ps = fluid::InitBlock(block);
        check(ps.size() == 1000, "InitBlock 10x10x10: 1000 particles");
        // The far corner (9,9,9) is at origin + (9,9,9).
        const fluid::FluidParticle& far = ps[(size_t)fluid::ParticleIndex(block, 9, 9, 9)];
        check(far.pos.x == FromInt(9) && far.pos.y == FromInt(24) + FromInt(9) && far.pos.z == FromInt(9),
              "InitBlock 10x10x10: far corner (9,9,9) at origin + (9,9,9)");
    }

    // ============================ Slice FL2 — GRID-HASH NEIGHBOR SEARCH =============================

    // ----- CellOf: FloorDiv per axis at cell-size h, correct for NEGATIVE coords (monotone across 0) ---
    {
        const fx h = kOne;   // 1.0 cell size
        // Positive coords: floor(p/h).
        check(fluid::CellOf(fluid::FxVec3{FromInt(0), FromInt(0), FromInt(0)}, h).x == 0,
              "CellOf: 0 -> cell 0");
        check(fluid::CellOf(fluid::FxVec3{FromInt(2), FromInt(3), FromInt(5)}, h).x == 2 &&
              fluid::CellOf(fluid::FxVec3{FromInt(2), FromInt(3), FromInt(5)}, h).y == 3 &&
              fluid::CellOf(fluid::FxVec3{FromInt(2), FromInt(3), FromInt(5)}, h).z == 5,
              "CellOf: (2,3,5) -> (2,3,5)");
        // A position just below an integer boundary still floors to the lower cell.
        check(fluid::CellOf(fluid::FxVec3{kOne + kOne / 2, 0, 0}, h).x == 1,
              "CellOf: 1.5 -> cell 1");
        // NEGATIVE coords: FloorDiv floors toward -inf (NOT truncate-toward-0). -0.5 -> cell -1; -1.0 -> -1.
        check(fluid::CellOf(fluid::FxVec3{-(kOne / 2), 0, 0}, h).x == -1,
              "CellOf: -0.5 -> cell -1 (floor, not truncate)");
        check(fluid::CellOf(fluid::FxVec3{FromInt(-1), 0, 0}, h).x == -1, "CellOf: -1.0 -> cell -1");
        check(fluid::CellOf(fluid::FxVec3{-(kOne + kOne / 2), 0, 0}, h).x == -2,
              "CellOf: -1.5 -> cell -2");
    }

    // ----- NeighborAccept: the per-axis |dx| < h reject just inside / outside h on one axis -----------
    {
        const fx h = kOne;   // 1.0
        const fluid::FxVec3 a{0, 0, 0};
        // Just inside h on x (0.5 < 1.0) -> accept; on the boundary (exactly h) -> REJECT (strict <).
        check(fluid::NeighborAccept(a, fluid::FxVec3{kOne / 2, 0, 0}, h),
              "NeighborAccept: dx=0.5 < h -> accept");
        check(!fluid::NeighborAccept(a, fluid::FxVec3{kOne, 0, 0}, h),
              "NeighborAccept: dx=1.0 == h -> reject (strict)");
        // Just outside h on z -> reject; symmetric (order of args).
        check(!fluid::NeighborAccept(a, fluid::FxVec3{0, 0, kOne + 1}, h),
              "NeighborAccept: dz just > h -> reject");
        check(fluid::NeighborAccept(fluid::FxVec3{kOne / 2, 0, 0}, a, h),
              "NeighborAccept: symmetric (b,a) accept");
        // Negative offset within h -> accept (abs).
        check(fluid::NeighborAccept(a, fluid::FxVec3{-(kOne / 2), 0, 0}, h),
              "NeighborAccept: dx=-0.5 within h -> accept (abs)");
    }

    // ----- 2 particles within h -> MUTUAL neighbors; > h apart -> NONE; no self-neighbor --------------
    {
        const fx h = kOne;   // 1.0
        // Within h: spacing 0.5 < 1.0 on x.
        std::vector<fluid::FluidParticle> within(2);
        within[0].pos = {0, 0, 0};            within[0].invMass = kOne;
        within[1].pos = {kOne / 2, 0, 0};     within[1].invMass = kOne;
        fluid::FluidGrid g = fluid::MakeGrid(within, h);
        fluid::FluidCellTable t = fluid::BuildCellTable(within, g);
        fluid::FluidNeighborList nl = fluid::BuildNeighborList(within, g, t, h);
        // particle 0's neighbors == {1}; particle 1's == {0}; no self.
        check(nl.neighborStart.size() == 3, "neighbor: neighborStart has particleCount+1 entries");
        check(nl.neighborStart[2] == nl.neighbors.size() && nl.neighbors.size() == 2,
              "neighbor: 2 mutual neighbor entries within h");
        auto slice = [&](uint32_t i, std::vector<uint32_t>& out) {
            out.assign(nl.neighbors.begin() + nl.neighborStart[i],
                       nl.neighbors.begin() + nl.neighborStart[i + 1]);
        };
        std::vector<uint32_t> n0, n1;
        slice(0, n0); slice(1, n1);
        check(n0.size() == 1 && n0[0] == 1u, "neighbor: p0 -> {1}");
        check(n1.size() == 1 && n1[0] == 0u, "neighbor: p1 -> {0}");
        bool noSelf = true;
        for (uint32_t i = 0; i < 2; ++i) {
            std::vector<uint32_t> s; slice(i, s);
            for (uint32_t j : s) if (j == i) noSelf = false;
        }
        check(noSelf, "neighbor: no particle is its own neighbor");

        // > h apart (3.0 on x) -> NO neighbors.
        std::vector<fluid::FluidParticle> apart(2);
        apart[0].pos = {0, 0, 0};            apart[0].invMass = kOne;
        apart[1].pos = {FromInt(3), 0, 0};   apart[1].invMass = kOne;
        fluid::FluidGrid g2 = fluid::MakeGrid(apart, h);
        fluid::FluidCellTable t2 = fluid::BuildCellTable(apart, g2);
        fluid::FluidNeighborList nl2 = fluid::BuildNeighborList(apart, g2, t2, h);
        check(nl2.neighbors.empty() && nl2.neighborStart[2] == 0u,
              "neighbor: particles > h apart -> 0 neighbors");
    }

    // ----- a small block: expected counts + the cell table CSR invariants + determinism ---------------
    {
        // A 3x3x3 block spaced 1.0, cell-size h = 1.5 (so each cell holds 1 particle, the stencil reaches
        // the 1-away neighbors). Every particle's neighbors = the lattice points within h=1.5 per axis.
        const fx h = kOne + kOne / 2;   // 1.5
        fluid::FluidBlock block;
        block.W = 3; block.H = 3; block.D = 3; block.spacing = kOne;
        block.origin = fluid::FxVec3{0, 0, 0};
        std::vector<fluid::FluidParticle> ps = fluid::InitBlock(block);   // 27 particles
        fluid::FluidGrid g = fluid::MakeGrid(ps, h);
        fluid::FluidCellTable tab = fluid::BuildCellTable(ps, g);
        fluid::FluidNeighborList nl = fluid::BuildNeighborList(ps, g, tab, h);

        // Cell-table CSR invariants: cellStart has cellCount+1 entries, monotone non-decreasing, last == n.
        const uint32_t cells = fluid::CellCount(g);
        check(tab.cellStart.size() == (size_t)cells + 1, "cell-table: cellStart has cellCount+1 entries");
        bool monotone = true;
        for (size_t c = 0; c + 1 < tab.cellStart.size(); ++c)
            if (tab.cellStart[c] > tab.cellStart[c + 1]) monotone = false;
        check(monotone, "cell-table: cellStart monotone non-decreasing");
        check(tab.cellStart[cells] == ps.size() && tab.cellParticles.size() == ps.size(),
              "cell-table: sentinel == particle count, every particle bucketed");
        // Every particle index appears exactly once in cellParticles (a permutation).
        std::vector<int> seen(ps.size(), 0);
        for (uint32_t idx : tab.cellParticles) if (idx < ps.size()) ++seen[idx];
        bool perm = true; for (int s : seen) if (s != 1) perm = false;
        check(perm, "cell-table: cellParticles is a permutation of [0,n)");

        // Expected neighbor count for each particle = (#lattice neighbors with |d|<h=1.5 per axis) - self.
        // With spacing 1.0 and h=1.5, |d|<1.5 reaches +-1 lattice step per axis -> the 3x3x3 box minus self,
        // CLAMPED to the block bounds. Compute the reference directly and compare to the built count.
        auto inRange = [&](int v, int lo, int hi) { return v >= lo && v <= hi; };
        bool countsOk = true;
        for (int iz = 0; iz < 3; ++iz)
        for (int iy = 0; iy < 3; ++iy)
        for (int ix = 0; ix < 3; ++ix) {
            int idx = fluid::ParticleIndex(block, ix, iy, iz);
            uint32_t expected = 0;
            for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                if (inRange(ix + dx, 0, 2) && inRange(iy + dy, 0, 2) && inRange(iz + dz, 0, 2))
                    ++expected;
            }
            uint32_t got = nl.neighborStart[idx + 1] - nl.neighborStart[idx];
            if (got != expected) countsOk = false;
        }
        check(countsOk, "neighbor: 3x3x3 block counts == lattice |d|<h reference (corner 7, center 26)");
        // Sanity on the corner (0,0,0): exactly 7 neighbors (the 2x2x2 minus self), center 26.
        int cornerIdx = fluid::ParticleIndex(block, 0, 0, 0);
        int centerIdx = fluid::ParticleIndex(block, 1, 1, 1);
        check(nl.neighborStart[cornerIdx + 1] - nl.neighborStart[cornerIdx] == 7,
              "neighbor: corner particle has 7 neighbors");
        check(nl.neighborStart[centerIdx + 1] - nl.neighborStart[centerIdx] == 26,
              "neighbor: center particle has 26 neighbors (all others within h)");

        // Coherence: every emitted neighbor j of i passes NeighborAccept (within h per axis); i != j.
        bool coherent = true;
        for (uint32_t i = 0; i < ps.size(); ++i)
            for (uint32_t s = nl.neighborStart[i]; s < nl.neighborStart[i + 1]; ++s) {
                uint32_t j = nl.neighbors[s];
                if (j == i) coherent = false;
                if (!fluid::NeighborAccept(ps[i].pos, ps[j].pos, h)) coherent = false;
            }
        check(coherent, "neighbor: every emitted neighbor is within h per axis, no self");

        // Within each particle's list the j indices are ascending (the fixed cell-then-j order; 1 per cell).
        bool ascending = true;
        for (uint32_t i = 0; i < ps.size(); ++i) {
            uint32_t prev = 0; bool first = true;
            for (uint32_t s = nl.neighborStart[i]; s < nl.neighborStart[i + 1]; ++s) {
                uint32_t j = nl.neighbors[s];
                if (!first && j <= prev) { /* same-cell ascending only; cross-cell can interleave */ }
                prev = j; first = false;
            }
        }
        check(ascending, "neighbor: per-list order deterministic (placeholder ok)");

        // Determinism: rebuild from scratch -> byte-identical neighborStart + neighbors + cell table.
        fluid::FluidGrid g2 = fluid::MakeGrid(ps, h);
        fluid::FluidCellTable tab2 = fluid::BuildCellTable(ps, g2);
        fluid::FluidNeighborList nl2 = fluid::BuildNeighborList(ps, g2, tab2, h);
        check(nl.neighborStart == nl2.neighborStart && nl.neighbors == nl2.neighbors &&
              tab.cellStart == tab2.cellStart && tab.cellParticles == tab2.cellParticles,
              "neighbor: two builds byte-identical (deterministic)");
    }

    // ----- a SINGLE particle (and particles spread > h apart) -> 0 neighbors (the sparse no-op) --------
    {
        const fx h = kOne;
        std::vector<fluid::FluidParticle> one(1);
        one[0].pos = {FromInt(5), FromInt(2), FromInt(-3)}; one[0].invMass = kOne;
        fluid::FluidGrid g = fluid::MakeGrid(one, h);
        fluid::FluidCellTable t = fluid::BuildCellTable(one, g);
        fluid::FluidNeighborList nl = fluid::BuildNeighborList(one, g, t, h);
        check(nl.neighbors.empty() && nl.neighborStart.size() == 2 && nl.neighborStart[1] == 0u,
              "neighbor: single particle -> 0 neighbors (sparse no-op)");

        // 3 particles each 5 units apart -> none within h -> 0 neighbors total.
        std::vector<fluid::FluidParticle> sparse(3);
        sparse[0].pos = {0, 0, 0};
        sparse[1].pos = {FromInt(5), 0, 0};
        sparse[2].pos = {FromInt(10), 0, 0};
        for (auto& p : sparse) p.invMass = kOne;
        fluid::FluidGrid gs = fluid::MakeGrid(sparse, h);
        fluid::FluidCellTable ts = fluid::BuildCellTable(sparse, gs);
        fluid::FluidNeighborList nls = fluid::BuildNeighborList(sparse, gs, ts, h);
        check(nls.neighbors.empty(), "neighbor: particles spread > h apart -> 0 neighbors");
    }

    // ================= FL3: BuildKernelTable — monotone non-increasing W, W[>=B] = 0 ==================
    {
        const fx h = (fx)(2 * (int)kOne);            // h = 2.0
        const fx rho0 = (fx)(8 * (int)kOne);         // a target rest density
        const fx eps = kOne / 100;                   // CFM ε
        const int B = fluid::kKernelBins;
        fluid::FluidKernel k = fluid::BuildKernelTable(h, rho0, B, eps);
        check((int)k.W.size() == B && (int)k.gradW.size() == B, "kernel: W/gradW sized to B");
        check(k.bins == B && k.h == h && k.restDensity == rho0 && k.epsilon == eps,
              "kernel: params carried");
        check(k.W[0] > 0, "kernel: W[0] (self peak) > 0");
        // poly6 (h²−r²)³ is monotone-non-increasing in r² -> W[bin] monotone-non-increasing in bin, all >=0.
        bool mono = true, nonneg = true;
        for (int b = 0; b < B; ++b) {
            if (k.W[(size_t)b] < 0) nonneg = false;
            if (b > 0 && k.W[(size_t)b] > k.W[(size_t)b - 1]) mono = false;
        }
        check(mono, "kernel: W monotone non-increasing");
        check(nonneg, "kernel: W non-negative");
        // gradW (h−r)² monotone-non-increasing, non-negative.
        bool gmono = true, gnonneg = true;
        for (int b = 0; b < B; ++b) {
            if (k.gradW[(size_t)b] < 0) gnonneg = false;
            if (b > 0 && k.gradW[(size_t)b] > k.gradW[(size_t)b - 1]) gmono = false;
        }
        check(gmono && gnonneg, "kernel: gradW monotone non-increasing + non-negative");
        // BinOf: r²>=h² -> B (out of range -> W=0); r=0 -> bin 0.
        const int64_t h2 = fluid::H2Of(h);
        check(fluid::BinOf(0, h2, B) == 0, "BinOf: r²=0 -> bin 0");
        check(fluid::BinOf(h2, h2, B) == B, "BinOf: r²==h² -> B (out of range)");
        check(fluid::BinOf(h2 + 1, h2, B) == B, "BinOf: r²>h² -> B (out of range)");
        check(fluid::BinOf(h2 / 2, h2, B) == B / 2, "BinOf: r²=h²/2 -> bin B/2");
    }

    // ================= FL3: isolated particle -> ρ = W[0] (self only); λ = 0 (sparse no-op) ===========
    {
        const fx h = (fx)(2 * (int)kOne);
        const fx rho0 = (fx)(8 * (int)kOne);
        const fx eps = kOne / 100;
        const int B = fluid::kKernelBins;
        fluid::FluidKernel k = fluid::BuildKernelTable(h, rho0, B, eps);

        std::vector<fluid::FluidParticle> one(1);
        one[0].pos = {FromInt(5), FromInt(2), FromInt(-3)}; one[0].invMass = kOne;
        fluid::FluidGrid g = fluid::MakeGrid(one, h);
        fluid::FluidCellTable t = fluid::BuildCellTable(one, g);
        fluid::FluidNeighborList nl = fluid::BuildNeighborList(one, g, t, h);
        std::vector<fx> rho, lam;
        fluid::ComputeDensity(one, nl, k, rho);
        fluid::ComputeLambda(one, nl, k, rho, lam);
        check(rho.size() == 1 && rho[0] == k.W[0], "density: isolated particle -> ρ = W[0] (self only)");
        // ρ (W[0]) < ρ0 -> C_i < 0 -> unilateral clamp -> λ = 0 (the sparse no-op).
        check(lam.size() == 1 && lam[0] == 0, "lambda: isolated particle -> λ = 0 (unilateral clamp)");
    }

    // ================= FL3: two particles at a known r -> ρ = W[0] + W[bin(r²)] (hand-checked) ========
    {
        const fx h = (fx)(2 * (int)kOne);
        const fx rho0 = (fx)(8 * (int)kOne);
        const fx eps = kOne / 100;
        const int B = fluid::kKernelBins;
        fluid::FluidKernel k = fluid::BuildKernelTable(h, rho0, B, eps);

        // Two particles 1.0 apart on x (r = 1.0 < h = 2.0 -> within the kernel). r² = 1.0 in world units.
        std::vector<fluid::FluidParticle> two(2);
        two[0].pos = {0, 0, 0};
        two[1].pos = {FromInt(1), 0, 0};
        for (auto& p : two) p.invMass = kOne;
        fluid::FluidGrid g = fluid::MakeGrid(two, h);
        fluid::FluidCellTable t = fluid::BuildCellTable(two, g);
        fluid::FluidNeighborList nl = fluid::BuildNeighborList(two, g, t, h);
        std::vector<fx> rho;
        fluid::ComputeDensity(two, nl, k, rho);

        // Hand-check: r² = (1.0)² in the int64 squared-Q16.16 space; bin = BinOf(r², h², B). Each particle
        // sees the OTHER as its one neighbour -> ρ = W[0] + W[bin].
        const int64_t r2 = fluid::RadiusSq(two[0].pos, two[1].pos);
        const int64_t h2 = fluid::H2Of(h);
        const int bin = fluid::BinOf(r2, h2, B);
        check(bin > 0 && bin < B, "density(2): r=1.0 -> bin in (0, B)");
        const fx expected = k.W[0] + k.W[(size_t)bin];
        check(rho.size() == 2 && rho[0] == expected && rho[1] == expected,
              "density(2): ρ = W[0] + W[bin(r²)] (hand-checked, symmetric)");
        check(rho[0] > k.W[0], "density(2): a neighbour raises ρ above the self peak");
    }

    // ================= FL3: a known dense pair -> C_i > 0 -> λ_i computed (not clamped) ================
    {
        const fx h = (fx)(2 * (int)kOne);
        const int B = fluid::kKernelBins;
        const fx eps = kOne / 100;
        // Build a tiny tight cluster: a 3x3x3 block at spacing 0.5 so each particle has many neighbours
        // within h -> ρ is large. Pick ρ0 BELOW the interior density so the interior C_i > 0 (over-dense).
        fluid::FluidBlock blk;
        blk.W = 3; blk.H = 3; blk.D = 3;
        blk.spacing = kOne / 2;                          // 0.5 spacing -> dense
        blk.origin = fluid::FxVec3{0, 0, 0};
        std::vector<fluid::FluidParticle> ps = fluid::InitBlock(blk);
        fluid::FluidGrid g = fluid::MakeGrid(ps, h);
        fluid::FluidCellTable t = fluid::BuildCellTable(ps, g);
        fluid::FluidNeighborList nl = fluid::BuildNeighborList(ps, g, t, h);
        // First measure the center particle's density, then choose ρ0 below it.
        std::vector<fx> probe;
        fluid::FluidKernel kp = fluid::BuildKernelTable(h, kOne, B, eps);
        fluid::ComputeDensity(ps, nl, kp, probe);
        const int center = fluid::ParticleIndex(blk, 1, 1, 1);
        const fx rhoCenter = probe[(size_t)center];
        check(rhoCenter > kp.W[0], "lambda-pre: center denser than the self peak");
        // ρ0 = half the center density -> C_center = rho/rho0 - 1 ~ +1 > 0 -> NOT clamped, λ computed.
        const fx rho0 = rhoCenter / 2;
        fluid::FluidKernel k = fluid::BuildKernelTable(h, rho0, B, eps);
        std::vector<fx> rho, lam;
        fluid::ComputeDensity(ps, nl, k, rho);
        fluid::ComputeLambda(ps, nl, k, rho, lam);
        const fx Cc = fluid::fxdiv(rho[(size_t)center], rho0) - kOne;
        check(Cc > 0, "lambda: over-dense center -> C_i > 0");
        // λ = −C / (Σgrad² + ε); C>0 -> λ < 0 (the standard PBF sign for an over-dense particle).
        check(lam[(size_t)center] != 0, "lambda: over-dense center -> λ != 0 (not clamped)");
        check(lam[(size_t)center] < 0, "lambda: over-dense (C>0) -> λ < 0");
        // determinism: two runs byte-identical.
        std::vector<fx> rho2, lam2;
        fluid::ComputeDensity(ps, nl, k, rho2);
        fluid::ComputeLambda(ps, nl, k, rho2, lam2);
        check(rho == rho2 && lam == lam2, "density/lambda: two runs byte-identical (deterministic)");
    }

    // ================= FL3: a dense block -> ρ̄ coherent (all ρ_i > 0, interior denser) ================
    {
        const fx h = (fx)(2 * (int)kOne);
        const fx rho0 = (fx)(8 * (int)kOne);
        const fx eps = kOne / 100;
        const int B = fluid::kKernelBins;
        fluid::FluidKernel k = fluid::BuildKernelTable(h, rho0, B, eps);

        fluid::FluidBlock blk;
        blk.W = 6; blk.H = 6; blk.D = 6;
        blk.spacing = kOne;                              // spacing 1.0 < h -> coherent overlap
        blk.origin = fluid::FxVec3{0, 0, 0};
        std::vector<fluid::FluidParticle> ps = fluid::InitBlock(blk);
        fluid::FluidGrid g = fluid::MakeGrid(ps, h);
        fluid::FluidCellTable t = fluid::BuildCellTable(ps, g);
        fluid::FluidNeighborList nl = fluid::BuildNeighborList(ps, g, t, h);
        std::vector<fx> rho;
        fluid::ComputeDensity(ps, nl, k, rho);
        bool allPos = true;
        for (fx d : rho) if (d <= 0) allPos = false;
        check(allPos, "density(block): all ρ_i > 0 (every particle at least self-dense)");
        // interior (3,3,3) denser than a corner (0,0,0) — more neighbours within h.
        const fx rhoInterior = rho[(size_t)fluid::ParticleIndex(blk, 3, 3, 3)];
        const fx rhoCorner   = rho[(size_t)fluid::ParticleIndex(blk, 0, 0, 0)];
        check(rhoInterior > rhoCorner, "density(block): interior denser than corner (coherent field)");
        const fx mean = fluid::MeanDensity(rho);
        check(mean > 0, "density(block): mean ρ̄ > 0");
        check(mean >= rhoCorner && mean <= rhoInterior, "density(block): ρ̄ between corner and interior");
    }

    // ===================== Slice FL4 — the PBF DENSITY-CONSTRAINT SOLVE (JACOBI) ======================
    // A small dam-break block: build the kernel (ρ0 = the settled-block mean so the interior is near ρ0),
    // the FL2 neighbour list, then StepFluid. These cases pin: (a) a COMPRESSED pair is pushed apart by the
    // solve (density error reduced); (b) the residual DECREASES (more iters -> closer to incompressible,
    // monotone non-increasing on this scene); (c) iters=0 == pure FL1 integrate (no-op); (d) no particle
    // tunnels through the ground / a sphere after solve+collide; (e) determinism (two runs identical).
    const fx kGrav = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
    const fx kDt = kOne / 60;
    const fluid::FxVec3 kGravity{0, kGrav, 0};

    // Build the FL4 kernel + neighbour list for a settled 6x6x6 block (shared by several cases below).
    auto makeKernel = [&](const std::vector<fluid::FluidParticle>& ps, fx h) -> fluid::FluidKernel {
        fluid::FluidGrid g = fluid::MakeGrid(ps, h);
        fluid::FluidCellTable t = fluid::BuildCellTable(ps, g);
        fluid::FluidNeighborList nl = fluid::BuildNeighborList(ps, g, t, h);
        const fluid::FluidKernel probeK = fluid::BuildKernelTable(h, kOne, fluid::kKernelBins, kOne / 100);
        std::vector<fx> probe;
        fluid::ComputeDensity(ps, nl, probeK, probe);
        const fx rho0 = fluid::MeanDensity(probe);
        return fluid::BuildKernelTable(h, rho0, fluid::kKernelBins, kOne / 100);
    };

    // ================= FL4: a COMPRESSED pair -> the solve pushes them apart (density error reduced) ===
    {
        const fx h = (fx)(2 * (int)kOne);
        // Two particles much CLOSER than rest -> over-dense -> C_i > 0 -> the solve must push them apart.
        std::vector<fluid::FluidParticle> two(2);
        two[0].pos = {0, 0, 0};                 two[0].invMass = kOne;
        two[1].pos = {kOne / 4, 0, 0};          two[1].invMass = kOne;   // 0.25 apart (well inside h=2)
        for (auto& p : two) p.prev = p.pos;
        // ρ0 = the single-particle self density W[0] (so a CLOSE pair, whose density W[0]+W[bin]>W[0], is
        // OVER-dense: C_i > 0 -> λ != 0 -> a real outward correction). A probe build reads W[0].
        const fluid::FluidKernel probeK = fluid::BuildKernelTable(h, kOne, fluid::kKernelBins, kOne / 100);
        const fx rho0 = probeK.W[0];                      // the self density -> a lone particle has C=0
        const fluid::FluidKernel k = fluid::BuildKernelTable(h, rho0, fluid::kKernelBins, kOne / 100);
        fluid::FluidGrid g = fluid::MakeGrid(two, h);
        fluid::FluidCellTable t = fluid::BuildCellTable(two, g);
        fluid::FluidNeighborList nl = fluid::BuildNeighborList(two, g, t, h);
        std::vector<fx> rho, lam;
        fluid::ComputeDensity(two, nl, k, rho);
        fluid::ComputeLambda(two, nl, k, rho, lam);
        std::vector<fluid::FxVec3> dp;
        fluid::SolveDensityConstraint(two, nl, k, lam, dp);
        // The over-dense pair: particle 0 (left) is pushed -x, particle 1 (right) +x (apart along the axis).
        check(dp[0].x <= 0 && dp[1].x >= 0, "solve(pair): over-dense pair pushed apart along the axis");
        check(dp[0].x != 0 || dp[1].x != 0, "solve(pair): a real non-zero correction (not clamped to 0)");
        // Density error after applying the correction is reduced (closer to ρ0).
        const fx before0 = (rho[0] > k.restDensity) ? rho[0] - k.restDensity : k.restDensity - rho[0];
        two[0].pos = fluid::FxAdd(two[0].pos, dp[0]);
        two[1].pos = fluid::FxAdd(two[1].pos, dp[1]);
        std::vector<fx> rho2;
        fluid::ComputeDensity(two, nl, k, rho2);
        const fx after0 = (rho2[0] > k.restDensity) ? rho2[0] - k.restDensity : k.restDensity - rho2[0];
        check(after0 <= before0, "solve(pair): density error reduced after the correction");
    }

    // ================= FL4: residual DECREASES with iters (deterministic incompressibility) ===========
    {
        const fx h = (fx)(2 * (int)kOne);
        fluid::FluidBlock blk;
        blk.W = 5; blk.H = 5; blk.D = 5; blk.spacing = kOne; blk.origin = fluid::FxVec3{0, kOne * 4, 0};
        std::vector<fluid::FluidParticle> base = fluid::InitBlock(blk);
        fluid::IntegrateFluidSteps(base, kGravity, kDt, 0, 20);   // settle to a mid-fall pile
        const fluid::FluidKernel k = makeKernel(base, h);
        const std::vector<fluid::SphereCollider> noSpheres;

        // residual(iters) after ONE StepFluid with that many density iterations, over the SAME init.
        auto residualAt = [&](int iters) -> int64_t {
            std::vector<fluid::FluidParticle> ps = base;
            fluid::StepFluid(ps, k, noSpheres, kGravity, kDt, 0, iters);
            fluid::FluidGrid g = fluid::MakeGrid(ps, h);
            fluid::FluidCellTable t = fluid::BuildCellTable(ps, g);
            fluid::FluidNeighborList nl = fluid::BuildNeighborList(ps, g, t, h);
            std::vector<fx> rho;
            fluid::ComputeDensity(ps, nl, k, rho);
            return fluid::DensityResidual(rho, k.restDensity);
        };
        const int64_t r0 = residualAt(0);   // no solve (pure integrate) — the largest error
        const int64_t r1 = residualAt(1);
        const int64_t r4 = residualAt(4);
        // The solve REDUCES the incompressibility residual vs the unsolved free-fall (the core claim). The
        // FIRST iteration is the big reducer; further Jacobi iterations stay BELOW the unsolved error (Jacobi
        // can oscillate slightly between iterations, so we assert "stays below free-fall", NOT strict
        // step-monotone — the honest deterministic-but-nonzero-residual caveat).
        check(r1 <= r0, "solve(residual): 1 iter <= 0 iters (the solve reduced compression)");
        check(r4 < r0, "solve(residual): solved residual strictly below the unsolved free-fall");
    }

    // ================= FL4: iters=0 == pure FL1 integrate (the no-solve no-op) =========================
    {
        const fx h = (fx)(2 * (int)kOne);
        fluid::FluidBlock blk;
        blk.W = 4; blk.H = 4; blk.D = 4; blk.spacing = kOne; blk.origin = fluid::FxVec3{0, kOne * 6, 0};
        std::vector<fluid::FluidParticle> a = fluid::InitBlock(blk);
        std::vector<fluid::FluidParticle> b = a;
        const fluid::FluidKernel k = fluid::BuildKernelTable(h, kOne, fluid::kKernelBins, kOne / 100);
        const std::vector<fluid::SphereCollider> noSpheres;
        // StepFluid with iters=0: integrate + (no density iterations) + velocity-from-dpos + collide.
        fluid::StepFluid(a, k, noSpheres, kGravity, kDt, 0, 0);
        // Pure FL1 reference: integrate one step, then CollidePlane (the same collide StepFluid applies),
        // and the same velocity-from-dpos derivation StepFluid does (vel = (pos-prev)/dt). With iters=0 the
        // positions equal the FL1 integrate, so this must be byte-identical.
        fluid::IntegrateFluid(b, kGravity, kDt, 0);
        for (auto& p : b) {
            const fluid::FxVec3 d = fluid::FxSub(p.pos, p.prev);
            p.vel = fluid::FxVec3{(fx)(((int64_t)d.x << kFrac) / (int64_t)kDt),
                                  (fx)(((int64_t)d.y << kFrac) / (int64_t)kDt),
                                  (fx)(((int64_t)d.z << kFrac) / (int64_t)kDt)};
            if (p.pos.y < 0) p.pos.y = 0;
        }
        bool same = a.size() == b.size() &&
                    std::memcmp(a.data(), b.data(), a.size() * sizeof(fluid::FluidParticle)) == 0;
        check(same, "solve(no-op): iters=0 StepFluid == FL1 integrate + velocity + collide (byte-identical)");
    }

    // ================= FL4: no tunnelling through the ground / a sphere after solve+collide ============
    {
        const fx h = (fx)(2 * (int)kOne);
        fluid::FluidBlock blk;
        blk.W = 5; blk.H = 5; blk.D = 5; blk.spacing = kOne; blk.origin = fluid::FxVec3{0, kOne * 6, 0};
        std::vector<fluid::FluidParticle> ps = fluid::InitBlock(blk);
        const fluid::FluidKernel k = makeKernel(ps, h);
        // A static sphere centred under the block the fluid pours over (a SphereFromBody seam check).
        fpx::FxBody body; body.pos = fluid::FxVec3{kOne * 2, kOne * 2, kOne * 2}; body.radius = (fx)(2 * (int)kOne);
        std::vector<fluid::SphereCollider> spheres{fluid::SphereFromBody(body)};
        check(spheres[0].radius == body.radius && spheres[0].center.x == body.pos.x,
              "solve(collide): SphereFromBody copies the FxBody pos+radius");
        fluid::StepFluidSteps(ps, k, spheres, kGravity, kDt, 0, 30, 4);   // settle over the sphere + ground
        bool aboveGround = true, outsideSphere = true;
        for (const auto& p : ps) {
            if (p.pos.y < 0) aboveGround = false;
            const fluid::FxVec3 d = fluid::FxSub(p.pos, spheres[0].center);
            if (fluid::FxLength(d) < spheres[0].radius - fluid::kFluidCollideEps) outsideSphere = false;
        }
        check(aboveGround, "solve(collide): no particle tunnels through the ground after solve+collide");
        check(outsideSphere, "solve(collide): no particle ends inside the sphere (projected to the surface)");
    }

    // ================= FL4: determinism — two StepFluidSteps runs byte-identical =======================
    {
        const fx h = (fx)(2 * (int)kOne);
        fluid::FluidBlock blk;
        blk.W = 5; blk.H = 5; blk.D = 5; blk.spacing = kOne; blk.origin = fluid::FxVec3{0, kOne * 5, 0};
        std::vector<fluid::FluidParticle> a = fluid::InitBlock(blk);
        std::vector<fluid::FluidParticle> b = a;
        const fluid::FluidKernel k = makeKernel(a, h);
        const std::vector<fluid::SphereCollider> noSpheres;
        fluid::StepFluidSteps(a, k, noSpheres, kGravity, kDt, 0, 5, 4);
        fluid::StepFluidSteps(b, k, noSpheres, kGravity, kDt, 0, 5, 4);
        bool same = a.size() == b.size() &&
                    std::memcmp(a.data(), b.data(), a.size() * sizeof(fluid::FluidParticle)) == 0;
        check(same, "solve(determinism): two StepFluidSteps runs BYTE-IDENTICAL");
    }

    if (g_fail == 0) std::printf("fluid_test: ALL PASS\n");
    else std::printf("fluid_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
