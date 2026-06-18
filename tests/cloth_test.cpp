// Slice CL1 — Deterministic GPU Cloth: the Q16.16 PARTICLE LATTICE INTEGRATOR + grid build core
// (engine/sim/cloth.h) that the GPU shaders/cloth_integrate.comp.hlsl copies VERBATIM + proves
// bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace hf::sim::cloth.
//
// What this test PINS (the contracts the GPU cloth_integrate.comp + the GPU==CPU proof build on):
//   * InitGrid: a flat W x H sheet (size W*H, row-major), spacing/origin host-snapped, the two TOP
//     corners PINNED (invMass 0, kFlagPinned), every other particle dynamic (invMass kOne); prev==pos.
//   * IntegrateParticle: one-step + K-step semi-implicit-Euler closed form (a FREE particle falls the
//     EXACT Q16.16 distance — hand-checked integer recurrence), prev = pos each step.
//   * a PINNED particle NEVER moves (untouched across many steps).
//   * the ground floor clamp (pos.y pinned to groundY, vel.y zeroed on contact, stays settled).
//   * determinism: two runs of the SAME init+steps -> byte-identical particle arrays.
//   * an ALL-PINNED grid is UNCHANGED after K steps (the static no-op).
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests.
#include "sim/cloth.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace cloth = hf::sim::cloth;
namespace fpx = hf::sim::fpx;
using cloth::fx;
using cloth::kOne;
using cloth::kFrac;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << kFrac); }

int main() {
    HF_TEST_MAIN_INIT();

    // ================= InitGrid: flat W x H sheet, top corners pinned =================
    {
        cloth::ClothGrid grid;
        grid.W = 5; grid.H = 4;
        grid.spacing = kOne;                 // 1.0 world unit between particles
        grid.origin = cloth::FxVec3{0, FromInt(10), 0};
        std::vector<cloth::ClothParticle> ps = cloth::InitGrid(grid);

        check(ps.size() == (size_t)(grid.W * grid.H), "InitGrid: size == W*H");

        // Row-major layout: particle (r,c) at origin + (c*spacing, -r*spacing, 0).
        const cloth::ClothParticle& p00 = ps[(size_t)cloth::ParticleIndex(grid, 0, 0)];
        const cloth::ClothParticle& p23 = ps[(size_t)cloth::ParticleIndex(grid, 2, 3)];
        check(p00.pos.x == 0 && p00.pos.y == FromInt(10) && p00.pos.z == 0,
              "InitGrid: (0,0) at origin");
        check(p23.pos.x == FromInt(3) && p23.pos.y == FromInt(10) - FromInt(2) && p23.pos.z == 0,
              "InitGrid: (2,3) at origin + (3, -2, 0)");

        // prev == pos and vel == 0 for every particle initially.
        bool prevEqPos = true, velZero = true;
        for (const cloth::ClothParticle& p : ps) {
            if (std::memcmp(&p.prev, &p.pos, sizeof(cloth::FxVec3)) != 0) prevEqPos = false;
            if (p.vel.x != 0 || p.vel.y != 0 || p.vel.z != 0) velZero = false;
        }
        check(prevEqPos, "InitGrid: prev == pos initially");
        check(velZero, "InitGrid: vel == 0 initially");

        // The two TOP corners (row 0, col 0 + col W-1) are PINNED; everything else dynamic.
        const cloth::ClothParticle& topL = ps[(size_t)cloth::ParticleIndex(grid, 0, 0)];
        const cloth::ClothParticle& topR = ps[(size_t)cloth::ParticleIndex(grid, 0, grid.W - 1)];
        check((topL.flags & cloth::kFlagPinned) && topL.invMass == 0, "InitGrid: top-left pinned");
        check((topR.flags & cloth::kFlagPinned) && topR.invMass == 0, "InitGrid: top-right pinned");
        check(cloth::CountPinned(ps) == 2, "InitGrid: exactly 2 pinned (the two top corners)");
        const cloth::ClothParticle& mid = ps[(size_t)cloth::ParticleIndex(grid, 1, 2)];
        check(!(mid.flags & cloth::kFlagPinned) && mid.invMass == kOne, "InitGrid: interior dynamic");
    }

    // ================= a FREE particle: one-step + K-step semi-implicit-Euler closed form ===========
    {
        const fx g  = FromInt(-10);     // gravity -10 (exact in Q16.16)
        const fx dt = kOne / 2;         // dt = 0.5 (exact)
        const fx groundY = FromInt(-1000);  // far below; effectively no clamp here
        const cloth::FxVec3 grav{0, g, 0};

        // One step: vel.y = 0 + g*dt = -10*0.5 = -5 ; pos.y = 100 + vel.y*dt = 100 + (-5)*0.5 = 97.5.
        cloth::ClothParticle p;
        p.pos = {0, FromInt(100), 0};
        p.prev = p.pos;
        p.vel = {0, 0, 0};
        p.invMass = kOne;
        p.flags = 0;                    // free (not pinned)
        cloth::IntegrateParticle(p, grav, groundY, dt);
        check(p.vel.y == FromInt(-5), "free one-step vel.y == g*dt == -5.0");
        check(p.pos.y == FromInt(100) - (kOne * 5 / 2), "free one-step pos.y == 100 + (-5)*0.5 == 97.5");
        check(p.prev.y == FromInt(100), "free one-step prev.y == the pre-move pos (100)");

        // K-step closed form: independently re-run the EXACT integer ops K times and compare to the
        // header's IntegrateParticlesSteps over the same single-particle lattice (must agree by construction).
        const int K = 60;
        cloth::ClothGrid grid; grid.W = 1; grid.H = 1; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(500), 0};
        std::vector<cloth::ClothParticle> one(1);
        one[0].pos = {0, FromInt(500), 0};
        one[0].prev = one[0].pos;
        one[0].vel = {0, 0, 0};
        one[0].invMass = kOne;
        one[0].flags = 0;

        // Reference: the same per-step integer recurrence, computed inline.
        fx refVy = 0, refPy = FromInt(500), refPrevY = FromInt(500);
        for (int s = 0; s < K; ++s) {
            refVy += cloth::fxmul(g, dt);
            refPrevY = refPy;
            refPy += cloth::fxmul(refVy, dt);
            if (refPy < groundY) { refPy = groundY; if (refVy < 0) refVy = 0; }
        }
        cloth::IntegrateParticlesSteps(grid, one, grav, dt, groundY, K);
        check(one[0].pos.y == refPy, "free K-step pos.y == hand-computed integer recurrence");
        check(one[0].vel.y == refVy, "free K-step vel.y == hand-computed integer recurrence");
        check(one[0].prev.y == refPrevY, "free K-step prev.y == the pre-last-move pos");
    }

    // ================= a PINNED particle NEVER moves =================
    {
        const cloth::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        cloth::ClothParticle pin;
        pin.pos = {FromInt(3), FromInt(7), FromInt(-2)};
        pin.prev = pin.pos;
        pin.vel = {0, 0, 0};
        pin.invMass = 0;
        pin.flags = cloth::kFlagPinned;
        cloth::ClothParticle before = pin;
        for (int s = 0; s < 100; ++s) cloth::IntegrateParticle(pin, grav, 0, dt);
        check(std::memcmp(&pin, &before, sizeof(cloth::ClothParticle)) == 0,
              "pinned particle never moves (untouched across 100 steps)");
    }

    // ================= ground floor clamp: pos.y==groundY, vel.y zeroed on contact =================
    {
        const cloth::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = 0;

        // A particle just above the ground with a big downward velocity clamps to groundY + zeros vel.y.
        cloth::ClothParticle p;
        p.pos = {0, kOne / 100, 0};      // 0.01 above ground
        p.prev = p.pos;
        p.vel = {0, FromInt(-50), 0};    // moving down fast
        p.invMass = kOne;
        p.flags = 0;
        cloth::IntegrateParticle(p, grav, groundY, dt);
        check(p.pos.y == groundY, "ground clamp: pos.y pinned to groundY");
        check(p.vel.y == 0, "ground clamp: downward vel.y zeroed on contact");

        // After contact, the particle stays settled (pos.y stays groundY across more steps).
        for (int s = 0; s < 50; ++s) cloth::IntegrateParticle(p, grav, groundY, dt);
        check(p.pos.y == groundY, "ground clamp: particle stays settled at groundY");
    }

    // ================= determinism: two runs byte-identical =================
    {
        const cloth::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = 0;
        auto makeGrid = []() {
            cloth::ClothGrid g; g.W = 8; g.H = 8; g.spacing = kOne;
            g.origin = cloth::FxVec3{0, FromInt(12), 0};
            return g;
        };
        cloth::ClothGrid grid = makeGrid();
        std::vector<cloth::ClothParticle> a = cloth::InitGrid(grid);
        std::vector<cloth::ClothParticle> b = cloth::InitGrid(grid);
        const int K = 60;
        cloth::IntegrateParticlesSteps(grid, a, grav, dt, groundY, K);
        cloth::IntegrateParticlesSteps(grid, b, grav, dt, groundY, K);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(cloth::ClothParticle)) == 0,
              "determinism: two runs byte-identical");

        // Coherence: the non-pinned particles moved DOWN, the pinned corners held (a coherent lattice).
        std::vector<cloth::ClothParticle> init = cloth::InitGrid(grid);
        int moved = 0, pinnedHeld = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].flags & cloth::kFlagPinned) {
                if (std::memcmp(&a[i], &init[i], sizeof(cloth::ClothParticle)) == 0) ++pinnedHeld;
            } else if (a[i].pos.y < init[i].pos.y) {
                ++moved;
            }
        }
        check(moved > 0, "coherence: non-pinned particles fell (moved down)");
        check(pinnedHeld == cloth::CountPinned(init), "coherence: every pinned corner held");
    }

    // ================= an ALL-PINNED grid is UNCHANGED after K steps (the static no-op) =============
    {
        const cloth::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = 0;
        cloth::ClothGrid grid; grid.W = 4; grid.H = 4; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(5), 0};
        std::vector<cloth::ClothParticle> ps = cloth::InitGrid(grid);
        // Force EVERY particle pinned.
        for (cloth::ClothParticle& p : ps) { p.flags = cloth::kFlagPinned; p.invMass = 0; }
        std::vector<cloth::ClothParticle> before = ps;
        cloth::IntegrateParticlesSteps(grid, ps, grav, dt, groundY, 120);
        check(std::memcmp(ps.data(), before.data(), ps.size() * sizeof(cloth::ClothParticle)) == 0,
              "all-pinned grid UNCHANGED after K steps (static no-op)");
    }

    // ================= CL2: BuildConstraints — a 2x2 grid hand-enumerated =================
    // A 2x2 grid (indices 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)): STRUCTURAL = (0,1) right + (2,3) right +
    // (0,2) down + (1,3) down = 4; SHEAR = the cell's two diagonals (0,3) down-right + (1,2) down-left = 2;
    // BEND = none (no 2-away neighbour exists in a 2-wide/2-tall grid) = 0. Total = 6.
    {
        cloth::ClothGrid grid;
        grid.W = 2; grid.H = 2; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(4), 0};
        std::vector<cloth::ClothParticle> ps = cloth::InitGrid(grid);
        std::vector<cloth::Constraint> es = cloth::BuildConstraints(grid, ps);

        int s, h, b;
        cloth::CountConstraintsByKind(es, s, h, b);
        check(es.size() == 6, "CL2 2x2: 6 edges total");
        check(s == 4, "CL2 2x2: 4 structural");
        check(h == 2, "CL2 2x2: 2 shear");
        check(b == 0, "CL2 2x2: 0 bend");

        // Hand-enumerated edge set (i<j, by kind). Build a lookup over the produced edges.
        auto has = [&](uint32_t i, uint32_t j, uint32_t kind) {
            for (const cloth::Constraint& e : es)
                if (e.i == i && e.j == j && e.kind == kind) return true;
            return false;
        };
        check(has(0, 1, cloth::kConstraintStructural), "CL2 2x2: (0,1) structural");
        check(has(2, 3, cloth::kConstraintStructural), "CL2 2x2: (2,3) structural");
        check(has(0, 2, cloth::kConstraintStructural), "CL2 2x2: (0,2) structural");
        check(has(1, 3, cloth::kConstraintStructural), "CL2 2x2: (1,3) structural");
        check(has(0, 3, cloth::kConstraintShear),      "CL2 2x2: (0,3) shear (down-right)");
        check(has(1, 2, cloth::kConstraintShear),      "CL2 2x2: (1,2) shear (down-left)");

        // restLens on a FLAT unit sheet: structural == spacing (1.0); shear == FxLength(1,1,0) == sqrt(2).
        const fx structRest = cloth::kOne;
        const fx shearRest  = hf::sim::fpx::FxLength(cloth::FxVec3{kOne, kOne, 0});
        for (const cloth::Constraint& e : es) {
            if (e.kind == cloth::kConstraintStructural)
                check(e.restLen == structRest, "CL2 2x2: structural restLen == spacing");
            else if (e.kind == cloth::kConstraintShear)
                check(e.restLen == shearRest, "CL2 2x2: shear restLen == FxLength(spacing,spacing,0)");
        }
        // sqrt(2)*65536 ~= 92682; sanity-band the shear length (within ~1 LSB of the integer sqrt).
        check(shearRest >= 92680 && shearRest <= 92684, "CL2 2x2: shear restLen ~= sqrt(2) in Q16.16");
    }

    // ================= CL2: the W x H edge-count formula + invariants =================
    {
        cloth::ClothGrid grid;
        grid.W = 6; grid.H = 5; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(20), 0};
        std::vector<cloth::ClothParticle> ps = cloth::InitGrid(grid);
        std::vector<cloth::Constraint> es = cloth::BuildConstraints(grid, ps);

        const int W = grid.W, H = grid.H;
        // Closed-form counts: STRUCTURAL = W*(H-1) + H*(W-1); SHEAR = 2*(W-1)*(H-1) (two diagonals/cell);
        // BEND = H*(W-2) + W*(H-2) (right 2-away + down 2-away).
        const int expectStruct = W * (H - 1) + H * (W - 1);
        const int expectShear  = 2 * (W - 1) * (H - 1);
        const int expectBend   = H * (W - 2) + W * (H - 2);
        int s, h, b;
        cloth::CountConstraintsByKind(es, s, h, b);
        check(s == expectStruct, "CL2 formula: structural == W(H-1)+H(W-1)");
        check(h == expectShear,  "CL2 formula: shear == 2(W-1)(H-1)");
        check(b == expectBend,   "CL2 formula: bend == H(W-2)+W(H-2)");
        check((int)es.size() == expectStruct + expectShear + expectBend, "CL2 formula: total == sum");

        // Every edge: i<j, both in-bounds, restLen>0; no duplicate (i,j) pair.
        const int n = W * H;
        bool ok = true;
        for (const cloth::Constraint& e : es) {
            if (!(e.i < e.j)) ok = false;
            if ((int)e.j >= n) ok = false;
            if (e.restLen <= 0) ok = false;
        }
        check(ok, "CL2: every edge i<j, in-bounds, restLen>0");

        // No duplicate undirected edge.
        std::vector<uint64_t> keys;
        keys.reserve(es.size());
        for (const cloth::Constraint& e : es) keys.push_back(((uint64_t)e.i << 32) | (uint64_t)e.j);
        bool dup = false;
        for (size_t a = 0; a < keys.size() && !dup; ++a)
            for (size_t c = a + 1; c < keys.size(); ++c)
                if (keys[a] == keys[c]) { dup = true; break; }
        check(!dup, "CL2: no duplicate edges");

        // Determinism: a second build is byte-identical.
        std::vector<cloth::Constraint> es2 = cloth::BuildConstraints(grid, ps);
        check(es.size() == es2.size() &&
              std::memcmp(es.data(), es2.data(), es.size() * sizeof(cloth::Constraint)) == 0,
              "CL2: two builds byte-identical (deterministic)");
    }

    // ================= CL2: a 1x1 grid -> 0 edges (degenerate / empty) =================
    {
        cloth::ClothGrid grid;
        grid.W = 1; grid.H = 1; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(3), 0};
        std::vector<cloth::ClothParticle> ps = cloth::InitGrid(grid);
        std::vector<cloth::Constraint> es = cloth::BuildConstraints(grid, ps);
        check(es.empty(), "CL2 1x1: 0 edges (degenerate empty)");
    }

    // ================= CL3: a 2-particle 1-constraint, one pinned one free -> free ends at restLen ====
    // Two particles on the x-axis: particle 0 PINNED at x=0, particle 1 FREE at x=3, a single distance
    // constraint with restLen=2. The pinned takes share 0; the free takes the WHOLE correction -> it moves
    // to exactly x = restLen = 2 (the exact inverse-mass split, hand-checked Q16.16). One solve pass is
    // enough for a single constraint (the projection is exact when only one endpoint moves).
    {
        const fx restLen = FromInt(2);
        std::vector<cloth::ClothParticle> ps(2);
        ps[0].pos = {0, 0, 0}; ps[0].prev = ps[0].pos; ps[0].invMass = 0; ps[0].flags = cloth::kFlagPinned;
        ps[1].pos = {FromInt(3), 0, 0}; ps[1].prev = ps[1].pos; ps[1].invMass = kOne; ps[1].flags = 0;
        cloth::Constraint c{0u, 1u, restLen, cloth::kConstraintStructural};

        cloth::SolveDistanceConstraint(ps, c);
        // pen = |d| - restLen = 3 - 2 = 1; n = +x; wsum = invMass1 = kOne; wi (pinned) = 0; wj = kOne.
        // pos[0] (pinned) += n*fxmul(pen, 0) = unchanged; pos[1] -= n*fxmul(pen, kOne) = x: 3 - 1 = 2.
        check(ps[0].pos.x == 0 && ps[0].pos.y == 0 && ps[0].pos.z == 0, "CL3 2p: pinned endpoint unchanged");
        check(ps[1].pos.x == restLen, "CL3 2p: free endpoint pulled to exactly restLen (2.0)");
        check(ps[1].pos.y == 0 && ps[1].pos.z == 0, "CL3 2p: free endpoint stays on the axis");

        // Compressed case: free at x=1 (closer than restLen=2) -> pushed OUT to x=2 (pen<0, both resolved).
        std::vector<cloth::ClothParticle> ps2(2);
        ps2[0].pos = {0, 0, 0}; ps2[0].prev = ps2[0].pos; ps2[0].invMass = 0; ps2[0].flags = cloth::kFlagPinned;
        ps2[1].pos = {FromInt(1), 0, 0}; ps2[1].prev = ps2[1].pos; ps2[1].invMass = kOne; ps2[1].flags = 0;
        cloth::SolveDistanceConstraint(ps2, c);
        check(ps2[1].pos.x == restLen, "CL3 2p compressed: free endpoint pushed out to restLen (2.0)");
    }

    // ================= CL3: both pinned -> no movement (the all-pinned no-op / static) =================
    {
        std::vector<cloth::ClothParticle> ps(2);
        ps[0].pos = {0, 0, 0}; ps[0].prev = ps[0].pos; ps[0].invMass = 0; ps[0].flags = cloth::kFlagPinned;
        ps[1].pos = {FromInt(3), 0, 0}; ps[1].prev = ps[1].pos; ps[1].invMass = 0; ps[1].flags = cloth::kFlagPinned;
        cloth::Constraint c{0u, 1u, FromInt(2), cloth::kConstraintStructural};
        std::vector<cloth::ClothParticle> before = ps;
        cloth::SolveDistanceConstraint(ps, c);
        check(std::memcmp(ps.data(), before.data(), ps.size() * sizeof(cloth::ClothParticle)) == 0,
              "CL3 static: both-pinned constraint is a no-op (wsum==0)");
    }

    // ================= CL3: equal-mass split -> each moves pen/2 (the inverse-mass share) =============
    {
        // Two FREE unit-mass particles at x=0 and x=4, restLen=2: pen = 4-2 = 2, wi=wj=0.5 -> each moves
        // pen*0.5 = 1 toward the other: pos[0].x = 0 + 1 = 1; pos[1].x = 4 - 1 = 3.
        std::vector<cloth::ClothParticle> ps(2);
        ps[0].pos = {0, 0, 0}; ps[0].prev = ps[0].pos; ps[0].invMass = kOne; ps[0].flags = 0;
        ps[1].pos = {FromInt(4), 0, 0}; ps[1].prev = ps[1].pos; ps[1].invMass = kOne; ps[1].flags = 0;
        cloth::Constraint c{0u, 1u, FromInt(2), cloth::kConstraintStructural};
        cloth::SolveDistanceConstraint(ps, c);
        check(ps[0].pos.x == FromInt(1), "CL3 split: equal-mass endpoint 0 moves +pen/2 to x=1");
        check(ps[1].pos.x == FromInt(3), "CL3 split: equal-mass endpoint 1 moves -pen/2 to x=3");
    }

    // ================= CL3: a pinned-at-one-end ROW drapes deterministically + pinned never move =======
    {
        const cloth::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = FromInt(-1000);   // far below: the floor-clamp is not the focus here
        const int iters = 8, steps = 40;

        // A 1-row, 6-wide strip; pin ONLY the left end (col 0). The rest hang + drape under gravity.
        cloth::ClothGrid grid; grid.W = 6; grid.H = 1; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(20), 0};
        std::vector<cloth::ClothParticle> ps = cloth::InitGrid(grid);
        // InitGrid pins the two TOP corners; for a single row that is BOTH ends — re-pin to left-only.
        for (auto& p : ps) { p.flags = 0; p.invMass = kOne; }
        ps[0].flags = cloth::kFlagPinned; ps[0].invMass = 0;   // pin left end only
        std::vector<cloth::Constraint> es = cloth::BuildConstraints(grid, ps);
        const cloth::ClothParticle pinBefore = ps[0];

        cloth::StepClothSteps(grid, ps, es, grav, dt, groundY, iters, steps);

        // The pinned left end NEVER moved.
        check(std::memcmp(&ps[0], &pinBefore, sizeof(cloth::ClothParticle)) == 0,
              "CL3 drape: the pinned end never moves");
        // The free particles fell (every non-pinned particle's y dropped below its rest y).
        std::vector<cloth::ClothParticle> rest = cloth::InitGrid(grid);
        int fell = 0;
        for (size_t i = 1; i < ps.size(); ++i) if (ps[i].pos.y < rest[i].pos.y) ++fell;
        check(fell == (int)ps.size() - 1, "CL3 drape: every free particle fell (coherent drape)");

        // The drape is COHESIVE: the residual (summed |edge len - restLen|) is small + bounded, NOT the
        // free-fall scatter a constraintless integrate would produce.
        const int64_t residual = cloth::EdgeResidual(ps, es);
        check(residual >= 0, "CL3 drape: residual is a non-negative integer metric");

        // Determinism: a second identical run is byte-identical AND yields the SAME residual.
        std::vector<cloth::ClothParticle> ps2 = cloth::InitGrid(grid);
        for (auto& p : ps2) { p.flags = 0; p.invMass = kOne; }
        ps2[0].flags = cloth::kFlagPinned; ps2[0].invMass = 0;
        cloth::StepClothSteps(grid, ps2, es, grav, dt, groundY, iters, steps);
        check(ps.size() == ps2.size() &&
              std::memcmp(ps.data(), ps2.data(), ps.size() * sizeof(cloth::ClothParticle)) == 0,
              "CL3 drape: two runs byte-identical (deterministic)");
        check(cloth::EdgeResidual(ps2, es) == residual, "CL3 drape: residual deterministic across runs");
    }

    // ================= CL3: iters=0 == pure integrate (CL1) — the no-constraint equivalence =============
    {
        const cloth::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = FromInt(-1000);
        const int steps = 30;

        cloth::ClothGrid grid; grid.W = 8; grid.H = 8; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(12), 0};
        std::vector<cloth::ClothParticle> a = cloth::InitGrid(grid);
        std::vector<cloth::ClothParticle> b = cloth::InitGrid(grid);
        std::vector<cloth::Constraint> es = cloth::BuildConstraints(grid, a);

        // StepCloth with iters=0 -> only IntegrateParticles + floor clamp (no constraint passes).
        cloth::StepClothSteps(grid, a, es, grav, dt, groundY, /*iters*/0, steps);
        // Pure CL1 integrate over the same sheet (groundY far below so the clamp branch never fires in either).
        cloth::IntegrateParticlesSteps(grid, b, grav, dt, groundY, steps);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(cloth::ClothParticle)) == 0,
              "CL3 iters=0: StepCloth == pure CL1 IntegrateParticles (no-constraint equivalence)");
    }

    // ================= CL4: a particle INSIDE a sphere -> projected to EXACTLY the surface =============
    {
        // A sphere of radius 4 at the origin; a free particle at (1,0,0) is inside -> pushed out to the
        // surface along +X -> exactly (radius, 0, 0). FxLength(p-center) == radius (the on-surface property).
        cloth::SphereCollider s{cloth::FxVec3{0, 0, 0}, FromInt(4)};
        cloth::ClothParticle p;
        p.pos = cloth::FxVec3{FromInt(1), 0, 0}; p.prev = p.pos; p.invMass = kOne; p.flags = 0;
        const bool hit = cloth::CollideParticleSphere(p, s);
        check(hit, "CL4 inside: a particle inside the sphere is projected (a contact)");
        check(p.pos.x == FromInt(4) && p.pos.y == 0 && p.pos.z == 0,
              "CL4 inside: projected to exactly the surface along the axis (radius,0,0)");
        const fx len = cloth::FxLength(cloth::FxSub(p.pos, s.center));
        // On-surface to fixed-point tolerance (FxNormalize+FxScale truncation can land within a few units).
        const fx tol = 4;  // a handful of Q16.16 LSBs
        check(len >= s.radius - tol && len <= s.radius + tol,
              "CL4 inside: FxLength(p-center) == radius within fixed-point tolerance");
    }

    // ================= CL4: a particle OUTSIDE a sphere -> UNTOUCHED ===================================
    {
        cloth::SphereCollider s{cloth::FxVec3{0, 0, 0}, FromInt(4)};
        cloth::ClothParticle p;
        p.pos = cloth::FxVec3{FromInt(10), FromInt(3), 0}; p.prev = p.pos; p.invMass = kOne; p.flags = 0;
        const cloth::ClothParticle before = p;
        const bool hit = cloth::CollideParticleSphere(p, s);
        check(!hit, "CL4 outside: a particle outside the sphere is NOT a contact");
        check(std::memcmp(&p, &before, sizeof(cloth::ClothParticle)) == 0,
              "CL4 outside: the particle is byte-unchanged (no projection)");
    }

    // ================= CL4: a PINNED particle inside a sphere -> NEVER moves ===========================
    {
        cloth::SphereCollider s{cloth::FxVec3{0, 0, 0}, FromInt(4)};
        cloth::ClothParticle p;
        p.pos = cloth::FxVec3{FromInt(1), 0, 0}; p.prev = p.pos; p.invMass = 0; p.flags = cloth::kFlagPinned;
        const cloth::ClothParticle before = p;
        const bool hit = cloth::CollideParticleSphere(p, s);
        check(!hit && std::memcmp(&p, &before, sizeof(cloth::ClothParticle)) == 0,
              "CL4 pinned: a pinned particle inside a sphere is untouched");
    }

    // ================= CL4: dist==0 (particle AT the center) -> +Y default normal =====================
    {
        cloth::SphereCollider s{cloth::FxVec3{0, 0, 0}, FromInt(4)};
        cloth::ClothParticle p;
        p.pos = cloth::FxVec3{0, 0, 0}; p.prev = p.pos; p.invMass = kOne; p.flags = 0;
        const bool hit = cloth::CollideParticleSphere(p, s);
        check(hit && p.pos.x == 0 && p.pos.y == FromInt(4) && p.pos.z == 0,
              "CL4 center: a particle at the center projects to +Y surface (the deterministic fallback)");
    }

    // ================= CL4: the ground clamp (CollidePlane) ===========================================
    {
        std::vector<cloth::ClothParticle> ps(3);
        ps[0].pos = cloth::FxVec3{0, FromInt(-5), 0};   // below ground -> clamped
        ps[1].pos = cloth::FxVec3{0, FromInt(3), 0};    // above ground -> untouched
        ps[2].pos = cloth::FxVec3{0, 0, 0};             // exactly at ground -> untouched
        const fx groundY = 0;
        cloth::CollidePlane(ps, groundY);
        check(ps[0].pos.y == 0, "CL4 plane: a particle below ground is clamped to groundY");
        check(ps[1].pos.y == FromInt(3), "CL4 plane: a particle above ground is unchanged");
        check(ps[2].pos.y == 0, "CL4 plane: a particle at ground is unchanged");
    }

    // ================= CL4: a sheet over a sphere -> NO penetration after K steps + deterministic ======
    {
        const cloth::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = FromInt(-1000);
        const int iters = 6, steps = 40;

        cloth::ClothGrid grid; grid.W = 12; grid.H = 12; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{FromInt(-6), FromInt(8), 0};   // hang the sheet above a sphere
        std::vector<cloth::ClothParticle> ps = cloth::InitGrid(grid);
        std::vector<cloth::Constraint> es = cloth::BuildConstraints(grid, ps);
        // A static sphere centered under the sheet's middle (FromBody seam check below uses the same units).
        std::vector<cloth::SphereCollider> spheres{ cloth::SphereCollider{cloth::FxVec3{0, FromInt(2), 0}, FromInt(3)} };

        std::vector<cloth::ClothParticle> a = ps;
        cloth::StepClothCollideSteps(grid, a, es, spheres, grav, dt, groundY, iters, steps);
        // No particle ends strictly inside the sphere (every one projected to the surface, deterministically).
        check(cloth::CountPenetrating(a, spheres) == 0,
              "CL4 drape: no particle penetrates the sphere after K steps");

        // Determinism: a second identical run is byte-identical.
        std::vector<cloth::ClothParticle> b = ps;
        cloth::StepClothCollideSteps(grid, b, es, spheres, grav, dt, groundY, iters, steps);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(cloth::ClothParticle)) == 0,
              "CL4 drape: two runs byte-identical (deterministic)");

        // The FromBody seam: a SphereCollider built from an fpx::FxBody has the SAME center+radius.
        fpx::FxBody body; body.pos = cloth::FxVec3{0, FromInt(2), 0}; body.radius = FromInt(3);
        const cloth::SphereCollider fromBody = cloth::SphereFromBody(body);
        check(fromBody.center.x == spheres[0].center.x && fromBody.center.y == spheres[0].center.y &&
              fromBody.center.z == spheres[0].center.z && fromBody.radius == spheres[0].radius,
              "CL4 seam: SphereFromBody reuses fpx::FxBody pos+radius (same Q16.16 units)");
    }

    // ================= CL4: ZERO spheres -> byte-identical to the CL3 StepCloth (the no-op equivalence) =
    {
        const cloth::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        const fx groundY = FromInt(-1000);
        const int iters = 6, steps = 40;

        cloth::ClothGrid grid; grid.W = 16; grid.H = 16; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(16), 0};
        std::vector<cloth::ClothParticle> base = cloth::InitGrid(grid);
        std::vector<cloth::Constraint> es = cloth::BuildConstraints(grid, base);

        std::vector<cloth::ClothParticle> withCollide = base;
        std::vector<cloth::SphereCollider> noSpheres;   // EMPTY collider set
        cloth::StepClothCollideSteps(grid, withCollide, es, noSpheres, grav, dt, groundY, iters, steps);

        std::vector<cloth::ClothParticle> pureSolve = base;
        cloth::StepClothSteps(grid, pureSolve, es, grav, dt, groundY, iters, steps);

        check(withCollide.size() == pureSolve.size() &&
              std::memcmp(withCollide.data(), pureSolve.data(),
                          withCollide.size() * sizeof(cloth::ClothParticle)) == 0,
              "CL4 zero-sphere: StepClothCollide == CL3 StepCloth byte-identical (no-op equivalence)");
    }

    // ================= CL5: ApplyClothCommand — wind / pin / unpin / OOB no-op =========================
    {
        cloth::ClothGrid grid; grid.W = 4; grid.H = 4; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(4), 0};
        std::vector<cloth::ClothParticle> ps = cloth::InitGrid(grid);

        // kCmdWind adds to a dynamic particle's velocity.
        const int freeIdx = cloth::ParticleIndex(grid, 1, 1);   // an interior, dynamic particle
        const cloth::ClothCommand wind{0, cloth::kCmdWind, (uint32_t)freeIdx,
                                       cloth::FxVec3{FromInt(2), FromInt(-3), FromInt(1)}};
        cloth::ApplyClothCommand(ps, wind);
        check(ps[(size_t)freeIdx].vel.x == FromInt(2) && ps[(size_t)freeIdx].vel.y == FromInt(-3) &&
              ps[(size_t)freeIdx].vel.z == FromInt(1),
              "CL5 ApplyClothCommand: kCmdWind adds the delta-velocity to a dynamic particle");

        // kCmdWind on a PINNED particle is a no-op (pinned points hold).
        const int pinIdx = cloth::ParticleIndex(grid, 0, 0);    // a pinned top corner
        check((ps[(size_t)pinIdx].flags & cloth::kFlagPinned) != 0u, "CL5 setup: corner is pinned");
        cloth::ApplyClothCommand(ps, cloth::ClothCommand{0, cloth::kCmdWind, (uint32_t)pinIdx,
                                                         cloth::FxVec3{FromInt(9), 0, 0}});
        check(ps[(size_t)pinIdx].vel.x == 0 && ps[(size_t)pinIdx].vel.y == 0 && ps[(size_t)pinIdx].vel.z == 0,
              "CL5 ApplyClothCommand: kCmdWind on a pinned particle is a no-op");

        // kCmdPin: a dynamic particle becomes pinned (kFlagPinned set, invMass 0, vel zeroed).
        cloth::ApplyClothCommand(ps, cloth::ClothCommand{0, cloth::kCmdPin, (uint32_t)freeIdx, {}});
        check((ps[(size_t)freeIdx].flags & cloth::kFlagPinned) != 0u &&
              ps[(size_t)freeIdx].invMass == 0 &&
              ps[(size_t)freeIdx].vel.x == 0 && ps[(size_t)freeIdx].vel.y == 0 && ps[(size_t)freeIdx].vel.z == 0,
              "CL5 ApplyClothCommand: kCmdPin sets kFlagPinned + invMass 0 + zeroes velocity");

        // kCmdUnpin: a pinned particle becomes dynamic (kFlagPinned cleared, invMass kOne).
        cloth::ApplyClothCommand(ps, cloth::ClothCommand{0, cloth::kCmdUnpin, (uint32_t)pinIdx, {}});
        check((ps[(size_t)pinIdx].flags & cloth::kFlagPinned) == 0u && ps[(size_t)pinIdx].invMass == kOne,
              "CL5 ApplyClothCommand: kCmdUnpin clears kFlagPinned + restores invMass kOne");

        // Out-of-range target is a deterministic no-op (no crash, no mutation).
        std::vector<cloth::ClothParticle> before = ps;
        cloth::ApplyClothCommand(ps, cloth::ClothCommand{0, cloth::kCmdWind, 9999u,
                                                         cloth::FxVec3{FromInt(5), 0, 0}});
        check(before.size() == ps.size() &&
              std::memcmp(before.data(), ps.data(), ps.size() * sizeof(cloth::ClothParticle)) == 0,
              "CL5 ApplyClothCommand: out-of-range target is a no-op");
    }

    // ================= CL5: SnapshotCloth / RestoreCloth round-trip == original ========================
    {
        cloth::ClothGrid grid; grid.W = 8; grid.H = 8; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(8), 0};
        std::vector<cloth::ClothParticle> ps = cloth::InitGrid(grid);
        std::vector<cloth::Constraint> es = cloth::BuildConstraints(grid, ps);
        const cloth::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60; const fx groundY = FromInt(-1000); const int iters = 4;
        std::vector<cloth::SphereCollider> noSpheres;
        // Advance a few steps so the state is non-trivial, then snapshot it.
        cloth::StepClothCollideSteps(grid, ps, es, noSpheres, grav, dt, groundY, iters, 5);

        const std::vector<cloth::ClothParticle> snap = cloth::SnapshotCloth(ps);
        // Mutate (one more step), then restore -> must equal the snapshot exactly.
        cloth::StepClothCollide(grid, ps, es, noSpheres, grav, dt, groundY, iters);
        check(std::memcmp(ps.data(), snap.data(), snap.size() * sizeof(cloth::ClothParticle)) != 0,
              "CL5 snapshot: a step actually mutated the cloth (control)");
        cloth::RestoreCloth(ps, snap);
        check(ps.size() == snap.size() &&
              std::memcmp(ps.data(), snap.data(), snap.size() * sizeof(cloth::ClothParticle)) == 0,
              "CL5 snapshot: SnapshotCloth -> RestoreCloth round-trip == original BIT-EXACT");
    }

    // ================= CL5: SimClothTick determinism + deterministic command order ====================
    {
        cloth::ClothGrid grid; grid.W = 8; grid.H = 8; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(8), 0};
        std::vector<cloth::ClothParticle> base = cloth::InitGrid(grid);
        std::vector<cloth::Constraint> es = cloth::BuildConstraints(grid, base);
        const cloth::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60; const fx groundY = FromInt(-1000); const int iters = 4;
        std::vector<cloth::SphereCollider> noSpheres;

        const int t0 = cloth::ParticleIndex(grid, 4, 3);
        const std::vector<cloth::ClothCommand> stream{
            cloth::ClothCommand{0, cloth::kCmdWind, (uint32_t)t0, cloth::FxVec3{FromInt(3), 0, 0}},
        };
        // Two SimClothTick runs over the SAME init+stream -> byte-identical.
        std::vector<cloth::ClothParticle> a = base, b = base;
        cloth::SimClothTick(grid, a, es, noSpheres, stream, 0, grav, dt, groundY, iters);
        cloth::SimClothTick(grid, b, es, noSpheres, stream, 0, grav, dt, groundY, iters);
        check(std::memcmp(a.data(), b.data(), a.size() * sizeof(cloth::ClothParticle)) == 0,
              "CL5 SimClothTick: two runs byte-identical (deterministic)");

        // A tick whose commands don't match -> no command applied (pure StepClothCollide).
        std::vector<cloth::ClothParticle> noCmd = base, plain = base;
        cloth::SimClothTick(grid, noCmd, es, noSpheres, stream, 5, grav, dt, groundY, iters);  // tick 5: no cmd at 5
        cloth::StepClothCollide(grid, plain, es, noSpheres, grav, dt, groundY, iters);
        check(std::memcmp(noCmd.data(), plain.data(), plain.size() * sizeof(cloth::ClothParticle)) == 0,
              "CL5 SimClothTick: a tick with no matching command == pure StepClothCollide");
    }

    // ================= CL5: RunClothLockstep replica == authority (the lockstep headline) =============
    // ================= + RunClothRollback positive (converges) + negative (mispredict differs) ========
    {
        cloth::ClothGrid grid; grid.W = 12; grid.H = 12; grid.spacing = kOne;
        grid.origin = cloth::FxVec3{0, FromInt(12), 0};
        const std::vector<cloth::ClothParticle> init = cloth::InitGrid(grid);
        const std::vector<cloth::Constraint> es = cloth::BuildConstraints(grid, init);
        const cloth::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60; const fx groundY = FromInt(-1000); const int iters = 4;
        const int ticks = 16, mispredictTick = 6;
        std::vector<cloth::SphereCollider> noSpheres;

        const int wIdx = cloth::ParticleIndex(grid, 6, 6);
        const std::vector<cloth::ClothCommand> authStream{
            cloth::ClothCommand{2,  cloth::kCmdWind, (uint32_t)wIdx, cloth::FxVec3{FromInt(4), 0, 0}},
            cloth::ClothCommand{6,  cloth::kCmdWind, (uint32_t)wIdx, cloth::FxVec3{0, 0, FromInt(3)}},
            cloth::ClothCommand{10, cloth::kCmdPin,  (uint32_t)wIdx, {}},
        };
        // The MISPREDICTED stream: auth + a WRONG strong wind at mispredictTick (a real divergence).
        std::vector<cloth::ClothCommand> mispredictStream = authStream;
        mispredictStream.push_back(cloth::ClothCommand{(uint32_t)mispredictTick, cloth::kCmdWind,
                                                       (uint32_t)wIdx, cloth::FxVec3{FromInt(40), 0, 0}});

        const std::vector<cloth::ClothParticle> authority =
            cloth::RunClothLockstep(grid, init, es, noSpheres, authStream, ticks, grav, dt, groundY, iters);
        const std::vector<cloth::ClothParticle> replica =
            cloth::RunClothLockstep(grid, init, es, noSpheres, authStream, ticks, grav, dt, groundY, iters);

        // LOCKSTEP: replica (inputs-only) == authority BIT-EXACT.
        check(authority.size() == replica.size() &&
              std::memcmp(authority.data(), replica.data(), authority.size() * sizeof(cloth::ClothParticle)) == 0,
              "CL5 lockstep: replica == authority BIT-EXACT (inputs-only re-sim)");

        // ROLLBACK positive: rolledBack == authority BIT-EXACT.
        const std::vector<cloth::ClothParticle> rolledBack =
            cloth::RunClothRollback(grid, init, es, noSpheres, authStream, mispredictStream,
                                    ticks, mispredictTick, grav, dt, groundY, iters);
        check(rolledBack.size() == authority.size() &&
              std::memcmp(rolledBack.data(), authority.data(), authority.size() * sizeof(cloth::ClothParticle)) == 0,
              "CL5 rollback: corrected to authority BIT-EXACT (positive control)");

        // ROLLBACK negative control: the pre-rollback MISPREDICTED full run DIFFERED from authority.
        const std::vector<cloth::ClothParticle> mispredicted =
            cloth::RunClothLockstep(grid, init, es, noSpheres, mispredictStream, ticks, grav, dt, groundY, iters);
        check(mispredicted.size() == authority.size() &&
              std::memcmp(mispredicted.data(), authority.data(), authority.size() * sizeof(cloth::ClothParticle)) != 0,
              "CL5 rollback: mispredicted state DIFFERS from authority (negative control — the divergence was real)");
    }

    if (g_fail == 0) std::printf("cloth_test: ALL PASS\n");
    else std::printf("cloth_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
