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

    if (g_fail == 0) std::printf("fluid_test: ALL PASS\n");
    else std::printf("fluid_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
