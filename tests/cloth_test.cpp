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

    if (g_fail == 0) std::printf("cloth_test: ALL PASS\n");
    else std::printf("cloth_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
