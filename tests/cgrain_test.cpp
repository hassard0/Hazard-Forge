// Slice CG1 — Deterministic Rigid<->Grain Coupling: the UNIFIED bodies+grains WORLD + the BODY->GRAIN grid-hash
// QUERY (engine/sim/couple_grain.h) that the GPU cgrain_body_{count,scan,emit}.comp shaders copy VERBATIM +
// prove bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace hf::sim::cgrain.
// The CP1 twin (couple.h's body->fluid query) with the GRAIN grid instead of the fluid grid: per-BODY it
// gathers the grain particles inside the body's sphere, over the GR2 grain grid + cell table, accepting iff
// the per-axis box reject |body.pos.axis - g.pos.axis| < body.radius passes (pure int32, the exact radial
// sphere cull deferred to CG2/CG3). The body spans a RANGE of grain cells (its BodyAabb in cell space), not a
// fixed 27-cell stencil.
//
// What this test PINS (the contracts the GPU cgrain_body_* + the GPU==CPU proof build on):
//   * GatherBodyGrains: a hand-laid bed + one body -> the EXACT gathered set (every grain inside the box, none
//     outside), CSR offsets correct, ascending grain-index order within the body slice.
//   * a body clear of the bed -> empty (bodyStart[i+1]==bodyStart[i], 0 gathered).
//   * two bodies -> disjoint per-body lists, each ascending; the CSR offsets partition bodyGrains.
//   * an empty world (no bodies / no grains) -> 0 gathered (the no-op).
//   * determinism: two builds of the SAME world -> byte-identical CSR.
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/couple_grain.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace cgrain = hf::sim::cgrain;
namespace grain  = hf::sim::grain;
namespace fpx    = hf::sim::fpx;
using cgrain::fx;
using cgrain::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << fpx::kFrac); }

// Build a grain at an integer world position (the rest defaulted/dynamic, uniform radius).
static grain::GrainParticle GrainAt(int x, int y, int z, int radiusUnits) {
    grain::GrainParticle g;
    g.pos  = fpx::FxVec3{FromInt(x), FromInt(y), FromInt(z)};
    g.prev = g.pos;
    g.vel  = fpx::FxVec3{0, 0, 0};
    g.invMass = kOne;
    g.radius  = FromInt(radiusUnits);
    g.flags   = 0;
    return g;
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

int main() {
    HF_TEST_MAIN_INIT();

    // The CG1 query uses ONLY the per-axis box reject |body.pos.axis - g.pos.axis| < body.radius. The cell
    // size hSearch is the grain grid quantization; it must NOT change which grains are accepted (it only
    // changes the cell RANGE the body iterates). We pick hSearch = 1.0 so cells are 1 world unit.
    const fx h = kOne;

    // ============ GatherBodyGrains: one body over a hand-laid bed -> the exact gathered set ============
    // A 5x5x1 bed of grains on the z=0 plane at integer positions x,y in [0,4]. One body at (2,2,0) with
    // radius 2 accepts every grain with |x-2|<2 AND |y-2|<2 AND |0-0|<2 -> x in {1,2,3}, y in {1,2,3} -> a
    // 3x3 = 9-grain box (strict <, so x in {0,4} / y in {0,4} are EXCLUDED).
    {
        std::vector<grain::GrainParticle> grains;
        for (int y = 0; y <= 4; ++y)
            for (int x = 0; x <= 4; ++x)
                grains.push_back(GrainAt(x, y, 0, 0));   // radius irrelevant for the query (body-radius box)
        // grain index = y*5 + x (the push order).

        cgrain::CGrainWorld world;
        world.grains  = grains;
        world.bodies  = {BodyAt(2, 2, 0, 2)};
        world.hSearch = h;

        cgrain::CGrainQuery q = cgrain::GatherBodyGrains(world);

        check(q.bodyStart.size() == 2u, "one body -> bodyStart has bodyCount+1 == 2 entries");
        check(q.bodyStart[0] == 0u, "bodyStart[0] == 0");
        const uint32_t gathered = q.bodyStart[1];
        check(gathered == 9u, "body at (2,2) r=2 gathers the 3x3 box (9 grains, strict <)");
        check(q.bodyGrains.size() == 9u, "bodyGrains holds exactly the 9 gathered indices");

        // The EXACT expected set: x in {1,2,3}, y in {1,2,3} -> index y*5+x, ascending.
        std::vector<uint32_t> expected;
        for (int y = 1; y <= 3; ++y)
            for (int x = 1; x <= 3; ++x)
                expected.push_back((uint32_t)(y * 5 + x));
        std::sort(expected.begin(), expected.end());
        check(q.bodyGrains == expected, "the gathered set is EXACTLY the 3x3 box, ascending index");

        // Ascending order within the body slice (the EMIT-order discipline).
        bool ascending = true;
        for (size_t s = 1; s < q.bodyGrains.size(); ++s)
            if (q.bodyGrains[s] <= q.bodyGrains[s - 1]) ascending = false;
        check(ascending, "the per-body gathered indices are strictly ascending");

        // No grain OUTSIDE the box was gathered; every gathered grain IS inside the box.
        bool coherent = true;
        for (uint32_t idx : q.bodyGrains)
            if (!cgrain::BodyGrainAccept(world.bodies[0], grains[idx])) coherent = false;
        check(coherent, "every gathered grain passes BodyGrainAccept");
    }

    // ===================== a body CLEAR of the bed -> empty (0 gathered) =====================
    {
        std::vector<grain::GrainParticle> grains;
        for (int x = 0; x <= 4; ++x) grains.push_back(GrainAt(x, 0, 0, 0));
        cgrain::CGrainWorld world;
        world.grains  = grains;
        world.bodies  = {BodyAt(100, 100, 100, 2)};   // far from the bed
        world.hSearch = h;

        cgrain::CGrainQuery q = cgrain::GatherBodyGrains(world);
        check(q.bodyStart.size() == 2u, "clear body -> bodyStart has 2 entries");
        check(q.bodyStart[0] == 0u && q.bodyStart[1] == 0u, "clear body gathers 0 (empty slice)");
        check(q.bodyGrains.empty(), "clear body -> bodyGrains empty");
    }

    // ============ two bodies -> disjoint per-body lists, each ascending, CSR partitions ============
    {
        // Bed: a 1D row of 10 grains at x=0..9, y=0, z=0 (index == x).
        std::vector<grain::GrainParticle> grains;
        for (int x = 0; x <= 9; ++x) grains.push_back(GrainAt(x, 0, 0, 0));

        // Body A at x=1, r=2 -> |x-1|<2 -> x in {0,1,2} -> indices {0,1,2}.
        // Body B at x=8, r=2 -> |x-8|<2 -> x in {7,8,9} -> indices {7,8,9}. Disjoint from A.
        cgrain::CGrainWorld world;
        world.grains  = grains;
        world.bodies  = {BodyAt(1, 0, 0, 2), BodyAt(8, 0, 0, 2)};
        world.hSearch = h;

        cgrain::CGrainQuery q = cgrain::GatherBodyGrains(world);
        check(q.bodyStart.size() == 3u, "two bodies -> bodyStart has 3 entries");
        check(q.bodyStart[0] == 0u, "bodyStart[0]==0");

        const uint32_t a0 = q.bodyStart[0], a1 = q.bodyStart[1];
        const uint32_t b0 = q.bodyStart[1], b1 = q.bodyStart[2];
        check(a1 - a0 == 3u, "body A gathers 3 grains");
        check(b1 - b0 == 3u, "body B gathers 3 grains");
        check(q.bodyGrains.size() == 6u, "total gathered == 6");

        std::vector<uint32_t> sliceA(q.bodyGrains.begin() + a0, q.bodyGrains.begin() + a1);
        std::vector<uint32_t> sliceB(q.bodyGrains.begin() + b0, q.bodyGrains.begin() + b1);
        check((sliceA == std::vector<uint32_t>{0u, 1u, 2u}), "body A slice == {0,1,2} ascending");
        check((sliceB == std::vector<uint32_t>{7u, 8u, 9u}), "body B slice == {7,8,9} ascending");

        // Disjoint: no index appears in both bodies' slices (the per-body-disjoint property).
        bool disjoint = true;
        for (uint32_t ia : sliceA)
            for (uint32_t ib : sliceB)
                if (ia == ib) disjoint = false;
        check(disjoint, "the two bodies' gathered lists are disjoint");
    }

    // ===================== empty world (no bodies, no grains) -> 0 gathered =====================
    {
        cgrain::CGrainWorld world;
        world.hSearch = h;
        cgrain::CGrainQuery q = cgrain::GatherBodyGrains(world);
        check(q.bodyStart.size() == 1u, "no bodies -> bodyStart has 1 entry (the sentinel)");
        check(q.bodyStart[0] == 0u, "no bodies -> bodyStart[0]==0");
        check(q.bodyGrains.empty(), "no bodies -> bodyGrains empty (no-op)");
    }
    {
        // Bodies present but NO grains -> every body gathers 0.
        cgrain::CGrainWorld world;
        world.bodies  = {BodyAt(0, 0, 0, 5), BodyAt(3, 3, 3, 5)};
        world.hSearch = h;
        cgrain::CGrainQuery q = cgrain::GatherBodyGrains(world);
        check(q.bodyStart.size() == 3u, "no grains -> bodyStart has bodyCount+1 entries");
        check(q.bodyStart[0] == 0u && q.bodyStart[1] == 0u && q.bodyStart[2] == 0u,
              "no grains -> every body gathers 0");
        check(q.bodyGrains.empty(), "no grains -> bodyGrains empty");
    }

    // ===================== determinism: two builds -> byte-identical CSR =====================
    {
        std::vector<grain::GrainParticle> grains;
        for (int y = 0; y <= 6; ++y)
            for (int x = 0; x <= 6; ++x)
                grains.push_back(GrainAt(x, y, 0, 0));
        cgrain::CGrainWorld world;
        world.grains  = grains;
        world.bodies  = {BodyAt(2, 2, 0, 3), BodyAt(4, 4, 0, 2)};
        world.hSearch = h;

        cgrain::CGrainQuery q1 = cgrain::GatherBodyGrains(world);
        cgrain::CGrainQuery q2 = cgrain::GatherBodyGrains(world);
        check(q1.bodyStart == q2.bodyStart && q1.bodyGrains == q2.bodyGrains,
              "two GatherBodyGrains builds are byte-identical (deterministic)");
    }

    // ===================== hSearch independence: the cell size does NOT change the gathered set =====
    // The body-radius box reject is the ONLY acceptance test; a coarser/finer hSearch only changes the
    // iterated cell range, not which grains are accepted. Pick a 3D bed + a body and assert the gathered set
    // is identical for hSearch = 1.0 vs hSearch = 2.0.
    {
        std::vector<grain::GrainParticle> grains;
        for (int z = 0; z <= 4; ++z)
            for (int y = 0; y <= 4; ++y)
                for (int x = 0; x <= 4; ++x)
                    grains.push_back(GrainAt(x, y, z, 0));
        auto run = [&](fx hs) {
            cgrain::CGrainWorld world;
            world.grains  = grains;
            world.bodies  = {BodyAt(2, 2, 2, 2)};
            world.hSearch = hs;
            return cgrain::GatherBodyGrains(world);
        };
        cgrain::CGrainQuery qH1 = run(kOne);
        cgrain::CGrainQuery qH2 = run((fx)(2 * (int)kOne));
        // Sort each (the order is deterministic per hSearch but the cell traversal order can differ); the SET
        // of gathered grains must be identical.
        std::vector<uint32_t> s1 = qH1.bodyGrains, s2 = qH2.bodyGrains;
        std::sort(s1.begin(), s1.end());
        std::sort(s2.begin(), s2.end());
        check(s1 == s2, "the gathered SET is independent of hSearch (only the cell range changes)");
    }

    // ===================== CountGathered / MaxPerBody stat helpers =====================
    {
        std::vector<grain::GrainParticle> grains;
        for (int x = 0; x <= 9; ++x) grains.push_back(GrainAt(x, 0, 0, 0));
        cgrain::CGrainWorld world;
        world.grains  = grains;
        world.bodies  = {BodyAt(1, 0, 0, 2), BodyAt(8, 0, 0, 3)};   // A gathers 3, B gathers |x-8|<3 -> {6..9}=4? wait
        world.hSearch = h;
        // B at x=8 r=3 -> |x-8|<3 -> x in {6,7,8,9} -> 4 grains (clamped to the [0,9] bed). A gathers 3.
        cgrain::CGrainQuery q = cgrain::GatherBodyGrains(world);
        check(cgrain::CountGathered(q) == q.bodyGrains.size(), "CountGathered == bodyGrains.size()");
        check(cgrain::CountGathered(q) == 7u, "CountGathered == 3 (A) + 4 (B) == 7");
        check(cgrain::MaxPerBody(q) == 4u, "MaxPerBody == 4 (body B's gathered count)");
    }

    if (g_fail == 0) std::printf("cgrain_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
