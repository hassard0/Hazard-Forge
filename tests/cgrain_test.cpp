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

    // ============================================================================================
    // ===== Slice CG2 — CONTACT SUPPORT + DRAG (grain->body, the crux) ============================
    // ============================================================================================
    // AccumBodyGrainForces: a body over a hand-laid gathered list -> the EXACT Q16.16 vel delta (support
    // Σ pen·n along the contact normal; drag damps toward the grain avg; the fixed-order sum); a static body
    // untouched; an empty gather -> free-fall (no delta). StepCGrainSupport: a body settles to a rest line
    // above the bed floor, the support=0 control sinks, deterministic + body-order-independent.

    // ---- AccumBodyGrainForces: ONE grain directly BELOW the body -> a pure +Y support push (exact) ----
    // Body at (0, 1, 0) r=1; one grain at (0,0,0) r=1. d = body - grain = (0, kOne, 0); dist = kOne;
    // pen = (1+1) - 1 = 1 (in world units, kOne in Q16.16). n = normalize(d) = (0, kOne, 0). F_support =
    // n * fxmul(kSupport, pen) = (0, kSupport, 0). The grain is static (vel 0) so vGrainAvg = 0; F_drag =
    // kDrag*(0 - body.vel) = 0 (body.vel starts 0). dvel = (0, kSupport, 0) * invMass(kOne) * dt. So
    // body.vel.y == fxmul(fxmul(kSupport, dt? ) ...) — compute the exact expected with the same ops.
    {
        const fx dt = kOne / 60;
        cgrain::CGrainWorld world;
        world.hSearch = h;
        world.gravity = fpx::FxVec3{0, 0, 0};           // gravity handled by IntegrateBody (not here)
        world.dt = dt; world.groundY = 0;
        grain::GrainParticle g = GrainAt(0, 0, 0, 0);
        g.radius = FromInt(1);                           // r=1
        world.grains = {g};
        fpx::FxBody b = BodyAt(0, 1, 0, 1);              // body at y=1, r=1 -> overlaps the grain by pen=1
        world.bodies = {b};

        // The hand-laid gather: body 0 gathers grain 0.
        cgrain::CGrainQuery q;
        q.bodyStart = {0u, 1u};
        q.bodyGrains = {0u};

        cgrain::AccumBodyGrainForces(world, q, dt);

        // Exact expected: F_support = (0, fxmul(kSupport, kOne)==kSupport, 0); F_drag = 0; dvel.y =
        // fxmul(fxmul(kSupport, invMass=kOne), dt).
        const fx penFx = kOne;                           // pen = 1.0 in Q16.16
        const fx fSupportY = fpx::fxmul(cgrain::kSupport, penFx);
        const fx expVelY = fpx::fxmul(fpx::fxmul(fSupportY, kOne /*invMass*/), dt);
        check(world.bodies[0].vel.x == 0, "support: pure-below grain -> vel.x unchanged");
        check(world.bodies[0].vel.z == 0, "support: pure-below grain -> vel.z unchanged");
        check(world.bodies[0].vel.y == expVelY, "support: vel.y == fxmul(fxmul(kSupport*pen, invMass), dt)");
        check(world.bodies[0].vel.y > 0, "support pushes the body UP (positive +Y vel delta)");
    }

    // ---- AccumBodyGrainForces: DRAG damps toward a moving grain's velocity (exact) ----
    // A grain with a +X velocity, NON-overlapping (placed so pen <= 0) so ONLY drag acts. Body at (0,10,0)
    // r=1, grain at (0,0,0) r=1: dist=10, pen=(1+1)-10 = -8 < 0 -> NO support. The grain has vel=(kOne,0,0).
    // vGrainAvg=(kOne,0,0); body.vel=0 -> F_drag = kDrag*(kOne,0,0); dvel.x = fxmul(fxmul(kDrag*kOne, kOne), dt).
    {
        const fx dt = kOne / 60;
        cgrain::CGrainWorld world;
        world.hSearch = h; world.dt = dt; world.groundY = 0;
        grain::GrainParticle g = GrainAt(0, 0, 0, 0);
        g.radius = FromInt(1);
        g.vel = fpx::FxVec3{kOne, 0, 0};                 // grain moving +X at 1.0/s
        world.grains = {g};
        world.bodies = {BodyAt(0, 10, 0, 1)};            // far above -> no overlap -> pure drag
        cgrain::CGrainQuery q; q.bodyStart = {0u, 1u}; q.bodyGrains = {0u};

        cgrain::AccumBodyGrainForces(world, q, dt);

        const fx fDragX = fpx::fxmul(cgrain::kDrag, kOne);  // kDrag*(vGrainAvg.x - 0)
        const fx expVelX = fpx::fxmul(fpx::fxmul(fDragX, kOne /*invMass*/), dt);
        check(world.bodies[0].vel.x == expVelX, "drag: vel.x == fxmul(fxmul(kDrag*vGrainAvg, invMass), dt)");
        check(world.bodies[0].vel.x > 0, "drag pulls the body toward the grain's +X velocity");
        check(world.bodies[0].vel.y == 0, "drag: no support (non-overlapping) -> vel.y unchanged");
    }

    // ---- AccumBodyGrainForces: a STATIC (non-dynamic) body is untouched ----
    {
        const fx dt = kOne / 60;
        cgrain::CGrainWorld world;
        world.hSearch = h; world.dt = dt; world.groundY = 0;
        grain::GrainParticle g = GrainAt(0, 0, 0, 0); g.radius = FromInt(1);
        world.grains = {g};
        fpx::FxBody b = BodyAt(0, 1, 0, 1);
        b.flags = 0;                                     // NOT dynamic -> pinned
        b.vel = fpx::FxVec3{0, 0, 0};
        world.bodies = {b};
        cgrain::CGrainQuery q; q.bodyStart = {0u, 1u}; q.bodyGrains = {0u};

        cgrain::AccumBodyGrainForces(world, q, dt);
        check(world.bodies[0].vel.x == 0 && world.bodies[0].vel.y == 0 && world.bodies[0].vel.z == 0,
              "static body -> vel untouched (the pinned case)");
    }

    // ---- AccumBodyGrainForces: an EMPTY gather -> free-fall (no support/drag delta) ----
    {
        const fx dt = kOne / 60;
        cgrain::CGrainWorld world;
        world.hSearch = h; world.dt = dt; world.groundY = 0;
        world.grains = {GrainAt(0, 0, 0, 0)};
        fpx::FxBody b = BodyAt(0, 50, 0, 1);             // far above the bed
        b.vel = fpx::FxVec3{FromInt(7), 0, 0};           // some pre-existing velocity
        world.bodies = {b};
        cgrain::CGrainQuery q; q.bodyStart = {0u, 0u}; q.bodyGrains = {};   // gathers NOTHING

        cgrain::AccumBodyGrainForces(world, q, dt);
        check(world.bodies[0].vel.x == FromInt(7) && world.bodies[0].vel.y == 0 && world.bodies[0].vel.z == 0,
              "empty gather -> vel unchanged (free-fall, no support/drag)");
    }

    // ---- AccumBodyGrainForces: support Σ pen·n over MULTIPLE grains is the FIXED-ORDER sum ----
    // Two grains symmetric in X below the body so the X contributions cancel and the Y support doubles
    // (a hand-checkable Σ pen·n). Body at (0,1,0) r=2; grain A at (-1,0,0) r=1, grain B at (1,0,0) r=1.
    // For A: d=(1,1,0) (body-grain); for B: d=(-1,1,0). dist for each = sqrt(2). pen=(2+1)-sqrt(2)>0.
    // n_A=norm(1,1,0), n_B=norm(-1,1,0): x cancels, y doubles. The exact value is reproduced by replaying
    // the same ops in the same order.
    {
        const fx dt = kOne / 60;
        cgrain::CGrainWorld world;
        world.hSearch = h; world.dt = dt; world.groundY = 0;
        grain::GrainParticle gA = GrainAt(-1, 0, 0, 0); gA.radius = FromInt(1);
        grain::GrainParticle gB = GrainAt( 1, 0, 0, 0); gB.radius = FromInt(1);
        world.grains = {gA, gB};
        world.bodies = {BodyAt(0, 1, 0, 2)};
        cgrain::CGrainQuery q; q.bodyStart = {0u, 2u}; q.bodyGrains = {0u, 1u};

        cgrain::AccumBodyGrainForces(world, q, dt);

        // Replay the exact fixed-order reduction to get the expected delta.
        auto fxL = [](fpx::FxVec3 v) { return fpx::FxLength(v); };
        fpx::FxVec3 fSup{0, 0, 0};
        const fpx::FxBody bb = BodyAt(0, 1, 0, 2);
        for (uint32_t s = 0; s < 2; ++s) {
            const grain::GrainParticle& gg = world.grains[s];
            fpx::FxVec3 d = fpx::FxSub(bb.pos, gg.pos);
            fx dist = fxL(d);
            fx pen = (bb.radius + gg.radius) - dist;
            if (pen > 0) fSup = fpx::FxAdd(fSup, fpx::FxScale(fpx::FxNormalize(d), fpx::fxmul(cgrain::kSupport, pen)));
        }
        fpx::FxVec3 dvel = fpx::FxScale(fpx::FxScale(fSup, kOne), dt);
        check(world.bodies[0].vel.x == dvel.x, "multi-grain support: vel.x == the fixed-order Σ pen·n delta");
        check(world.bodies[0].vel.y == dvel.y, "multi-grain support: vel.y == the fixed-order Σ pen·n delta");
        // (The X contributions are equal-opposite normals scaled by the same magnitude; they NEARLY cancel
        // but the arithmetic-right-shift fxmul floors toward -inf so the net may be a 1-LSB residual — the
        // honest fixed-point reduction is what the GPU reproduces bit-for-bit, NOT an exact analytic zero.)
        check(world.bodies[0].vel.x >= -2 && world.bodies[0].vel.x <= 2,
              "symmetric grains: the X support contributions cancel to within a 1-LSB fixed-point residual");
        check(world.bodies[0].vel.y > 0, "symmetric grains: the Y support contributions add (net UP)");
    }

    // ---- StepCGrainSupport: a body dropped above a bed SETTLES to a rest line above the bed floor ----
    // A dense flat grain bed (a slab) + a body dropped above it. After K steps the body rests ON the bed:
    // restY > groundY + radius (it did NOT crash through) AND bounded above (it did NOT bounce out).
    {
        const fx dt = kOne / 60;
        const fx hSearch = kOne + kOne / 2;              // 1.5
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        // A flat bed slab: x,z in [0,8], y in [0,2], unit-spaced 0.25-radius grains.
        std::vector<grain::GrainParticle> grains;
        for (int gy = 0; gy <= 2; ++gy)
            for (int gz = 0; gz <= 8; ++gz)
                for (int gx = 0; gx <= 8; ++gx) {
                    grain::GrainParticle g = GrainAt(gx, gy, gz, 0);
                    g.radius = kOne / 4;
                    grains.push_back(g);
                }
        auto makeWorld = [&](int bx, int by, int bz) {
            cgrain::CGrainWorld w;
            w.grains = grains; w.hSearch = hSearch;
            w.gravity = fpx::FxVec3{0, kGravY, 0}; w.dt = dt; w.groundY = 0;
            w.bodies = {BodyAt(bx, by, bz, 2)};
            return w;
        };
        cgrain::CGrainWorld world = makeWorld(4, 12, 4);
        const fx kBedFloor = world.groundY + world.bodies[0].radius;   // groundY + radius (the bed floor)
        cgrain::StepCGrainSupportSteps(world, dt, 300);
        const fx restY = cgrain::MeasureRestLine(world);

        check(restY > kBedFloor + kOne / 4, "rests: restY above the bed floor by a clear margin (not crashed through)");
        check(restY < FromInt(12), "rests: restY bounded above (did not bounce out)");

        // determinism: two runs byte-identical.
        cgrain::CGrainWorld world2 = makeWorld(4, 12, 4);
        cgrain::StepCGrainSupportSteps(world2, dt, 300);
        check(std::memcmp(&world.bodies[0], &world2.bodies[0], sizeof(fpx::FxBody)) == 0,
              "StepCGrainSupport: two runs byte-identical (deterministic)");

        // ---- support=0 control: the body SINKS through to the bed floor (restY == groundY + radius) ----
        // With kSupport forced to 0 the body free-falls + only ResolveGround clamps it -> restY == bed floor.
        // (We emulate kSupport=0 by a body that gathers grains but whose support is zeroed: here we run the
        // SAME world but with grains placed so the body never overlaps -> equivalent free-fall to the floor.)
        cgrain::CGrainWorld ctrl;
        ctrl.grains = {};                                // NO grains -> no support, the body free-falls
        ctrl.hSearch = hSearch; ctrl.gravity = fpx::FxVec3{0, kGravY, 0}; ctrl.dt = dt; ctrl.groundY = 0;
        ctrl.bodies = {BodyAt(4, 12, 4, 2)};
        cgrain::StepCGrainSupportSteps(ctrl, dt, 300);
        const fx ctrlY = cgrain::MeasureRestLine(ctrl);
        check(ctrlY == kBedFloor, "control: support=0 sinks through to the bed floor (restY == groundY+radius)");
        check(restY > ctrlY, "rests: the supported body rests ABOVE the support=0 control (support does work)");
    }

    // ---- StepCGrainSupport: the rest line is body-ORDER-independent (a determinism property) ----
    // Two bodies dropped over a bed in order [A,B] vs [B,A]; each body's settled state is identical
    // regardless of the iteration order (the bodies are disjoint, no body-body coupling).
    {
        const fx dt = kOne / 60;
        const fx hSearch = kOne + kOne / 2;
        const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
        std::vector<grain::GrainParticle> grains;
        for (int gy = 0; gy <= 2; ++gy)
            for (int gz = 0; gz <= 12; ++gz)
                for (int gx = 0; gx <= 12; ++gx) {
                    grain::GrainParticle g = GrainAt(gx, gy, gz, 0);
                    g.radius = kOne / 4;
                    grains.push_back(g);
                }
        fpx::FxBody bodyA = BodyAt(3, 10, 3, 2);
        fpx::FxBody bodyB = BodyAt(9, 10, 9, 2);
        auto run = [&](std::vector<fpx::FxBody> bodies) {
            cgrain::CGrainWorld w;
            w.grains = grains; w.hSearch = hSearch;
            w.gravity = fpx::FxVec3{0, kGravY, 0}; w.dt = dt; w.groundY = 0;
            w.bodies = std::move(bodies);
            cgrain::StepCGrainSupportSteps(w, dt, 200);
            return w;
        };
        cgrain::CGrainWorld wAB = run({bodyA, bodyB});
        cgrain::CGrainWorld wBA = run({bodyB, bodyA});
        check(std::memcmp(&wAB.bodies[0], &wBA.bodies[1], sizeof(fpx::FxBody)) == 0,
              "body-order independence: A's settled state is identical in [A,B] vs [B,A]");
        check(std::memcmp(&wAB.bodies[1], &wBA.bodies[0], sizeof(fpx::FxBody)) == 0,
              "body-order independence: B's settled state is identical in [A,B] vs [B,A]");
    }

    if (g_fail == 0) std::printf("cgrain_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
