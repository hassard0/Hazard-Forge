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

    if (g_fail == 0) std::printf("grain_test: ALL PASS\n");
    else std::printf("grain_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
