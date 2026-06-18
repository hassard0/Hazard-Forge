// Slice CP1 — Deterministic Rigid<->Fluid Coupling: the UNIFIED COUPLED WORLD + the BODY->FLUID grid-hash
// QUERY (engine/sim/couple.h) that the GPU couple_body_{count,scan,emit}.comp shaders copy VERBATIM + prove
// bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace hf::sim::couple.
// The GR2 grain-neighbor twin: per-BODY (not per-grain) it gathers the fluid particles inside the body's
// sphere, over the FL2 fluid grid + cell table, accepting iff the per-axis box reject
// |body.pos.axis - p.pos.axis| < body.radius passes (pure int32, the exact radial sphere cull deferred to
// CP2). The body spans a RANGE of cells (its BodyAabb in cell space), not a fixed 27-cell stencil.
//
// What this test PINS (the contracts the GPU couple_body_* + the GPU==CPU proof build on):
//   * GatherBodyParticles: a hand-laid pool + one body -> the EXACT gathered set (every particle inside the
//     box, none outside), CSR offsets correct, ascending particle-index order within the body slice.
//   * a body clear of the pool -> empty (bodyStart[i+1]==bodyStart[i], 0 gathered).
//   * two bodies -> disjoint per-body lists, each ascending; the CSR offsets partition bodyParticles.
//   * an empty world (no bodies / no particles) -> 0 gathered (the no-op).
//   * determinism: two builds of the SAME world -> byte-identical CSR.
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/couple.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace couple = hf::sim::couple;
namespace fluid  = hf::sim::fluid;
namespace fpx    = hf::sim::fpx;
using couple::fx;
using couple::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << fpx::kFrac); }

// Build a fluid particle at an integer world position (the rest defaulted/dynamic).
static fluid::FluidParticle ParticleAt(int x, int y, int z) {
    fluid::FluidParticle p;
    p.pos  = fpx::FxVec3{FromInt(x), FromInt(y), FromInt(z)};
    p.prev = p.pos;
    p.vel  = fpx::FxVec3{0, 0, 0};
    p.invMass = kOne;
    p.flags   = 0;
    return p;
}

// Build a body at an integer world position with an integer radius.
static fpx::FxBody BodyAt(int x, int y, int z, int radius) {
    fpx::FxBody b;
    b.pos     = fpx::FxVec3{FromInt(x), FromInt(y), FromInt(z)};
    b.invMass = kOne;
    b.flags   = fpx::kFlagDynamic;
    b.radius  = FromInt(radius);
    return b;
}

// A FluidKernel whose only CP1-relevant field is the cell-size h (the grid cell size).
static fluid::FluidKernel KernelWithH(fx h) {
    fluid::FluidKernel k;
    k.h = h;
    return k;
}

int main() {
    HF_TEST_MAIN_INIT();

    // The CP1 query uses ONLY the per-axis box reject |body.pos.axis - p.pos.axis| < body.radius. The cell
    // size h is the fluid grid quantization; it must NOT change which particles are accepted (it only
    // changes the cell RANGE the body iterates). We pick h = 1.0 so cells are 1 world unit.
    const fx h = kOne;

    // ============ GatherBodyParticles: one body over a hand-laid pool -> the exact gathered set ============
    // A 5x5x1 grid of particles on the z=0 plane at integer positions x,y in [0,4]. One body at (2,2,0)
    // with radius 2 accepts every particle with |x-2|<2 AND |y-2|<2 AND |0-0|<2 -> x in {1,2,3}, y in
    // {1,2,3} -> a 3x3 = 9-particle box (strict <, so x in {0,4} / y in {0,4} are EXCLUDED).
    {
        std::vector<fluid::FluidParticle> particles;
        for (int y = 0; y <= 4; ++y)
            for (int x = 0; x <= 4; ++x)
                particles.push_back(ParticleAt(x, y, 0));
        // particle index = y*5 + x (the push order).

        couple::CoupleWorld world;
        world.particles = particles;
        world.bodies    = {BodyAt(2, 2, 0, 2)};
        world.kernel    = KernelWithH(h);

        couple::CoupleQuery q = couple::GatherBodyParticles(world);

        check(q.bodyStart.size() == 2u, "one body -> bodyStart has bodyCount+1 == 2 entries");
        check(q.bodyStart[0] == 0u, "bodyStart[0] == 0");
        const uint32_t gathered = q.bodyStart[1];
        check(gathered == 9u, "body at (2,2) r=2 gathers the 3x3 box (9 particles, strict <)");
        check(q.bodyParticles.size() == 9u, "bodyParticles holds exactly the 9 gathered indices");

        // The EXACT expected set: x in {1,2,3}, y in {1,2,3} -> index y*5+x, ascending.
        std::vector<uint32_t> expected;
        for (int y = 1; y <= 3; ++y)
            for (int x = 1; x <= 3; ++x)
                expected.push_back((uint32_t)(y * 5 + x));
        std::sort(expected.begin(), expected.end());
        check(q.bodyParticles == expected, "the gathered set is EXACTLY the 3x3 box, ascending index");

        // Ascending order within the body slice (the EMIT-order discipline).
        bool ascending = true;
        for (size_t s = 1; s < q.bodyParticles.size(); ++s)
            if (q.bodyParticles[s] <= q.bodyParticles[s - 1]) ascending = false;
        check(ascending, "the per-body gathered indices are strictly ascending");

        // No particle OUTSIDE the box was gathered; every gathered particle IS inside the box.
        bool coherent = true;
        for (uint32_t idx : q.bodyParticles)
            if (!couple::BodyParticleAccept(world.bodies[0], particles[idx])) coherent = false;
        check(coherent, "every gathered particle passes BodyParticleAccept");
    }

    // ===================== a body CLEAR of the pool -> empty (0 gathered) =====================
    {
        std::vector<fluid::FluidParticle> particles;
        for (int x = 0; x <= 4; ++x) particles.push_back(ParticleAt(x, 0, 0));
        couple::CoupleWorld world;
        world.particles = particles;
        world.bodies    = {BodyAt(100, 100, 100, 2)};   // far from the pool
        world.kernel    = KernelWithH(h);

        couple::CoupleQuery q = couple::GatherBodyParticles(world);
        check(q.bodyStart.size() == 2u, "clear body -> bodyStart has 2 entries");
        check(q.bodyStart[0] == 0u && q.bodyStart[1] == 0u, "clear body gathers 0 (empty slice)");
        check(q.bodyParticles.empty(), "clear body -> bodyParticles empty");
    }

    // ============ two bodies -> disjoint per-body lists, each ascending, CSR partitions ============
    {
        // Pool: a 1D row of 10 particles at x=0..9, y=0, z=0 (index == x).
        std::vector<fluid::FluidParticle> particles;
        for (int x = 0; x <= 9; ++x) particles.push_back(ParticleAt(x, 0, 0));

        // Body A at x=1, r=2 -> |x-1|<2 -> x in {0,1,2} -> indices {0,1,2}.
        // Body B at x=8, r=2 -> |x-8|<2 -> x in {7,8,9} -> indices {7,8,9}. Disjoint from A.
        couple::CoupleWorld world;
        world.particles = particles;
        world.bodies    = {BodyAt(1, 0, 0, 2), BodyAt(8, 0, 0, 2)};
        world.kernel    = KernelWithH(h);

        couple::CoupleQuery q = couple::GatherBodyParticles(world);
        check(q.bodyStart.size() == 3u, "two bodies -> bodyStart has 3 entries");
        check(q.bodyStart[0] == 0u, "bodyStart[0]==0");

        const uint32_t a0 = q.bodyStart[0], a1 = q.bodyStart[1];
        const uint32_t b0 = q.bodyStart[1], b1 = q.bodyStart[2];
        check(a1 - a0 == 3u, "body A gathers 3 particles");
        check(b1 - b0 == 3u, "body B gathers 3 particles");
        check(q.bodyParticles.size() == 6u, "total gathered == 6");

        std::vector<uint32_t> sliceA(q.bodyParticles.begin() + a0, q.bodyParticles.begin() + a1);
        std::vector<uint32_t> sliceB(q.bodyParticles.begin() + b0, q.bodyParticles.begin() + b1);
        check((sliceA == std::vector<uint32_t>{0u, 1u, 2u}), "body A slice == {0,1,2} ascending");
        check((sliceB == std::vector<uint32_t>{7u, 8u, 9u}), "body B slice == {7,8,9} ascending");

        // Disjoint: no index appears in both bodies' slices (the per-body-disjoint property).
        bool disjoint = true;
        for (uint32_t ia : sliceA)
            for (uint32_t ib : sliceB)
                if (ia == ib) disjoint = false;
        check(disjoint, "the two bodies' gathered lists are disjoint");
    }

    // ===================== empty world (no bodies, no particles) -> 0 gathered =====================
    {
        couple::CoupleWorld world;
        world.kernel = KernelWithH(h);
        couple::CoupleQuery q = couple::GatherBodyParticles(world);
        check(q.bodyStart.size() == 1u, "no bodies -> bodyStart has 1 entry (the sentinel)");
        check(q.bodyStart[0] == 0u, "no bodies -> bodyStart[0]==0");
        check(q.bodyParticles.empty(), "no bodies -> bodyParticles empty (no-op)");
    }
    {
        // Bodies present but NO particles -> every body gathers 0.
        couple::CoupleWorld world;
        world.bodies = {BodyAt(0, 0, 0, 5), BodyAt(3, 3, 3, 5)};
        world.kernel = KernelWithH(h);
        couple::CoupleQuery q = couple::GatherBodyParticles(world);
        check(q.bodyStart.size() == 3u, "no particles -> bodyStart has bodyCount+1 entries");
        check(q.bodyStart[0] == 0u && q.bodyStart[1] == 0u && q.bodyStart[2] == 0u,
              "no particles -> every body gathers 0");
        check(q.bodyParticles.empty(), "no particles -> bodyParticles empty");
    }

    // ===================== determinism: two builds -> byte-identical CSR =====================
    {
        std::vector<fluid::FluidParticle> particles;
        for (int y = 0; y <= 6; ++y)
            for (int x = 0; x <= 6; ++x)
                particles.push_back(ParticleAt(x, y, 0));
        couple::CoupleWorld world;
        world.particles = particles;
        world.bodies    = {BodyAt(2, 2, 0, 3), BodyAt(4, 4, 0, 2)};
        world.kernel    = KernelWithH(h);

        couple::CoupleQuery q1 = couple::GatherBodyParticles(world);
        couple::CoupleQuery q2 = couple::GatherBodyParticles(world);
        check(q1.bodyStart == q2.bodyStart && q1.bodyParticles == q2.bodyParticles,
              "two GatherBodyParticles builds are byte-identical (deterministic)");
    }

    if (g_fail == 0) std::printf("couple_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
