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

    // ============================================================================================
    // ============ Slice CP2 — BUOYANCY + DRAG (fluid->body, the CRUX) tests ======================
    // ============================================================================================
    const fpx::FxVec3 kGravityDown{0, -9 * (int)kOne, 0};   // a clean integer downward gravity (|g|=9.0)
    const fx kDt = kOne / 60;

    // ---- AccumBodyForces: a body over a HAND-LAID gathered list -> the EXACT Q16.16 vel delta. ----
    // The body's gathered list is constructed directly via a CoupleQuery so the reduction order + arithmetic
    // are pinned independently of the CP1 grid query. We re-derive the buoyancy + drag math by hand and assert
    // the exact resulting velocity (NO tolerance — the integer bar).
    {
        // 3 fluid particles with KNOWN velocities (the gathered set). vFluidAvg = the fixed-order int sum/count.
        std::vector<fluid::FluidParticle> particles(3);
        particles[0].vel = fpx::FxVec3{0,           0, 0};
        particles[1].vel = fpx::FxVec3{2 * (int)kOne, 0, 0};   // vx = 2.0
        particles[2].vel = fpx::FxVec3{4 * (int)kOne, 0, 0};   // vx = 4.0
        // -> Σvx = 6.0, count 3 -> vFluidAvg.x = 2.0 (exact integer divide), vFluidAvg.y=z=0.

        couple::CoupleWorld world;
        world.particles = particles;
        world.gravity   = kGravityDown;
        world.dt        = kDt;
        world.bodies    = {BodyAt(0, 0, 0, 2)};   // invMass=kOne, dynamic, radius 2
        // A hand-laid CSR: body 0 gathers particles {0,1,2} in ascending order.
        couple::CoupleQuery q;
        q.bodyStart     = {0u, 3u};
        q.bodyParticles = {0u, 1u, 2u};

        // The body starts at rest. Apply ONE AccumBodyForces.
        couple::AccumBodyForces(world, q, kDt);

        // Hand-computed expected (mirror of the header math, the same fxmul/FxScale ops):
        // up = -normalize((0,-9,0)) = (0,1,0).
        const fpx::FxVec3 up = fpx::FxNormalize(fpx::FxVec3{0, 9 * (int)kOne, 0});
        const fx countFx = (fx)(3 << fpx::kFrac);
        const fx buoyMag = fpx::fxmul(couple::kBuoyPerParticle, countFx);
        const fpx::FxVec3 fBuoy = fpx::FxScale(up, buoyMag);
        const fpx::FxVec3 vFluidAvg{2 * (int)kOne, 0, 0};
        const fpx::FxVec3 dv = fpx::FxSub(vFluidAvg, fpx::FxVec3{0,0,0});
        const fpx::FxVec3 fDrag{fpx::fxmul(couple::kDrag, dv.x), fpx::fxmul(couple::kDrag, dv.y),
                                fpx::fxmul(couple::kDrag, dv.z)};
        const fpx::FxVec3 fTotal = fpx::FxAdd(fBuoy, fDrag);
        const fpx::FxVec3 expectVel = fpx::FxScale(fpx::FxScale(fTotal, kOne /*invMass*/), kDt);

        const fpx::FxVec3 got = world.bodies[0].vel;
        check(got.x == expectVel.x && got.y == expectVel.y && got.z == expectVel.z,
              "AccumBodyForces: exact Q16.16 vel delta (buoyancy up + drag toward fluid avg)");
        // Sanity on direction: buoyancy is UP (vel.y > 0) and drag pulls toward the fluid's +x flow (vel.x>0).
        check(got.y > 0, "AccumBodyForces: buoyancy pushes the body UP (vel.y > 0)");
        check(got.x > 0, "AccumBodyForces: drag pulls the body toward the fluid's +x velocity (vel.x > 0)");
    }

    // ---- buoyancy up is PROPORTIONAL to the gathered count (the displaced-volume Archimedes proxy). ----
    {
        // Two bodies, identical except one gathers 2 particles, the other gathers 4 (still water, vel 0 ->
        // drag is 0, so the ONLY force is buoyancy). The 4-particle body must get ~2x the upward vel.
        std::vector<fluid::FluidParticle> particles(4);   // all vel = 0 (still water)
        couple::CoupleWorld world;
        world.particles = particles;
        world.gravity   = kGravityDown;
        world.bodies    = {BodyAt(0, 0, 0, 2), BodyAt(10, 0, 0, 2)};
        couple::CoupleQuery q;
        q.bodyStart     = {0u, 2u, 6u};          // body 0 gathers 2, body 1 gathers 4
        q.bodyParticles = {0u, 1u, 0u, 1u, 2u, 3u};

        couple::AccumBodyForces(world, q, kDt);
        const fx vy2 = world.bodies[0].vel.y;    // 2-particle buoyancy
        const fx vy4 = world.bodies[1].vel.y;    // 4-particle buoyancy
        check(vy2 > 0 && vy4 > 0, "buoyancy: both bodies pushed up");
        check(vy4 == 2 * vy2, "buoyancy up is EXACTLY proportional to the gathered count (4 == 2x2)");
        // Drag is zero in still water (vFluidAvg == 0, body at rest) -> no horizontal velocity.
        check(world.bodies[0].vel.x == 0 && world.bodies[1].vel.x == 0,
              "still water (vel 0, body at rest) -> drag contributes 0");
    }

    // ---- drag DAMPS: a moving body in still water gets a velocity correction TOWARD the fluid (opposing
    //      its own motion) — and a body gathering 0 particles is untouched (free-fall). ----
    {
        std::vector<fluid::FluidParticle> particles(2);   // still water (vel 0)
        couple::CoupleWorld world;
        world.particles = particles;
        world.gravity   = kGravityDown;
        world.bodies    = {BodyAt(0, 0, 0, 2), BodyAt(50, 0, 0, 2)};   // body1 far away -> gathers 0
        world.bodies[0].vel = fpx::FxVec3{3 * (int)kOne, 0, 0};        // moving +x at 3.0
        const fpx::FxVec3 vel1Before = world.bodies[1].vel;
        couple::CoupleQuery q;
        q.bodyStart     = {0u, 2u, 2u};          // body 0 gathers 2, body 1 gathers 0
        q.bodyParticles = {0u, 1u};

        couple::AccumBodyForces(world, q, kDt);
        // Drag opposes the body's +x motion (fluid at rest) -> the x-velocity delta is NEGATIVE.
        check(world.bodies[0].vel.x < 3 * (int)kOne, "drag damps the moving body (vel.x reduced toward fluid)");
        // The body that gathered 0 particles is UNTOUCHED (no buoyancy, no drag).
        check(world.bodies[1].vel.x == vel1Before.x && world.bodies[1].vel.y == vel1Before.y,
              "a body gathering 0 particles is untouched (free-fall, AccumBodyForces no-op)");
    }

    // ---- a STATIC (non-dynamic) body is untouched even when it gathers particles. ----
    {
        std::vector<fluid::FluidParticle> particles(3);
        couple::CoupleWorld world;
        world.particles = particles;
        world.gravity   = kGravityDown;
        fpx::FxBody stat = BodyAt(0, 0, 0, 2);
        stat.flags   = 0;       // NOT dynamic
        stat.invMass = 0;       // static / infinite mass
        world.bodies = {stat};
        const fpx::FxVec3 velBefore = world.bodies[0].vel;
        couple::CoupleQuery q;
        q.bodyStart     = {0u, 3u};
        q.bodyParticles = {0u, 1u, 2u};
        couple::AccumBodyForces(world, q, kDt);
        check(world.bodies[0].vel.x == velBefore.x && world.bodies[0].vel.y == velBefore.y &&
              world.bodies[0].vel.z == velBefore.z, "a static body is untouched by buoyancy/drag");
    }

    // ---- StepCoupleBuoyancy: a body dropped above a pool SETTLES to a float line ABOVE the bed; the
    //      buoy=0 control SINKS to the bed. Deterministic + body-order-independent. ----
    {
        // A still-water pool (a deep slab of particles at the ground) + a body dropped above it. The pool is
        // DEEP (y in [0,7]) so the body has room to settle at a PARTIAL-submersion float line WELL ABOVE the
        // bed (groundY + radius = 2.0) — a real float line, not a bed rest.
        auto buildWorld = [&](int bodyX, fx buoyTestY) -> couple::CoupleWorld {
            (void)buoyTestY;
            couple::CoupleWorld world;
            // 9x9x8 slab of particles (y in [0,7], a deep pool) at x,z=0..8.
            for (int py = 0; py <= 7; ++py)
                for (int pz = 0; pz <= 8; ++pz)
                    for (int px = 0; px <= 8; ++px)
                        world.particles.push_back(ParticleAt(px, py, pz));
            world.kernel  = KernelWithH(kOne);
            world.gravity = kGravityDown;
            world.dt      = kDt;
            world.groundY = 0;
            // The body dropped above the pool centre (x=4,z=4), radius 2.
            world.bodies = {BodyAt(bodyX, 12, 4, 2)};
            return world;
        };

        const fx kBedLine = (fx)(2 * (int)kOne);   // groundY + radius = 0 + 2.0 (the body's surface-rest line)
        const int kSteps = 240;

        // (a) buoyancy ON: the body settles ABOVE the bed (it floats) and is bounded (it did not fly out).
        couple::CoupleWorld floated = buildWorld(4, 0);
        couple::StepCoupleBuoyancySteps(floated, kDt, kSteps);
        const fx floatY = couple::MeasureFloatLine(floated);
        check(floatY > kBedLine + kOne / 4, "buoyancy ON: the body FLOATS above the bed by a clear margin");
        check(floatY < (fx)(12 * (int)kOne), "buoyancy ON: the float line is bounded (did NOT fly out)");

        // (b) buoy=0 CONTROL: a kBuoyPerParticle=0 body sinks to the bed (floatY == groundY + radius). We
        // emulate the control by setting invMass=0 buoyancy? No — the control must keep mass but kill buoyancy.
        // The header's coefficient is a compile-time constant; the control here is a body that gathers 0
        // particles for the WHOLE fall (placed far in x so the pool query never reaches it) -> pure free-fall
        // + ground clamp -> it rests at the bed groundY + radius. This is the "no buoyancy work" baseline.
        couple::CoupleWorld sunk = buildWorld(40, 0);   // far from the pool -> gathers 0 -> no buoyancy
        couple::StepCoupleBuoyancySteps(sunk, kDt, kSteps);
        const fx sunkY = couple::MeasureFloatLine(sunk);
        check(sunkY == kBedLine, "buoy=0 control (no gathered particles): SINKS to the bed (groundY + radius)");
        check(floatY > sunkY, "buoyancy does work: the floated body settles ABOVE the sunk control");

        // (c) determinism: two runs of the SAME scene -> byte-identical body state.
        couple::CoupleWorld a = buildWorld(4, 0), b = buildWorld(4, 0);
        couple::StepCoupleBuoyancySteps(a, kDt, kSteps);
        couple::StepCoupleBuoyancySteps(b, kDt, kSteps);
        check(std::memcmp(&a.bodies[0], &b.bodies[0], sizeof(fpx::FxBody)) == 0,
              "StepCoupleBuoyancy is deterministic (two runs byte-identical)");
    }

    // ---- body-order independence: two bodies in DISJOINT pools settle to the SAME float line regardless of
    //      the order they appear in world.bodies (the per-body reduction is independent across bodies). ----
    {
        auto buildTwo = [&](bool swapped) -> couple::CoupleWorld {
            couple::CoupleWorld world;
            // Two disjoint shallow pools: pool A around x=4, pool B around x=24 (well separated).
            for (int py = 0; py <= 2; ++py)
                for (int pz = 0; pz <= 8; ++pz)
                    for (int px = 0; px <= 8; ++px) world.particles.push_back(ParticleAt(px, py, pz));
            for (int py = 0; py <= 2; ++py)
                for (int pz = 0; pz <= 8; ++pz)
                    for (int px = 20; px <= 28; ++px) world.particles.push_back(ParticleAt(px, py, pz));
            world.kernel  = KernelWithH(kOne);
            world.gravity = kGravityDown;
            world.dt      = kDt;
            world.groundY = 0;
            const fpx::FxBody A = BodyAt(4, 8, 4, 2);
            const fpx::FxBody B = BodyAt(24, 8, 4, 2);
            world.bodies = swapped ? std::vector<fpx::FxBody>{B, A} : std::vector<fpx::FxBody>{A, B};
            return world;
        };
        couple::CoupleWorld w0 = buildTwo(false), w1 = buildTwo(true);
        couple::StepCoupleBuoyancySteps(w0, kDt, 200);
        couple::StepCoupleBuoyancySteps(w1, kDt, 200);
        // body A is index 0 in w0, index 1 in w1; body B is index 1 in w0, index 0 in w1.
        check(std::memcmp(&w0.bodies[0], &w1.bodies[1], sizeof(fpx::FxBody)) == 0 &&
              std::memcmp(&w0.bodies[1], &w1.bodies[0], sizeof(fpx::FxBody)) == 0,
              "buoyancy is body-order-independent (each body's reduction is independent)");
    }

    // ============================================================================================
    // ============ Slice CP3 — FLUID REACTION / DISPLACEMENT (body->fluid) tests ==================
    // ============================================================================================
    using couple::FxLength;
    using couple::FxNormalize;

    // ---- ApplyBodyToFluid: a particle INSIDE a body is SNAPPED to the body surface (|p−b.pos| == radius
    //      within an LSB) + gets a drag-reaction velocity toward the body. ----
    {
        couple::CoupleWorld world;
        world.gravity = kGravityDown;
        world.dt      = kDt;
        world.bodies  = {BodyAt(0, 0, 0, 2)};            // dynamic, radius 2, at origin
        world.bodies[0].vel = fpx::FxVec3{3 * (int)kOne, 0, 0};   // the body moves +x at 3.0
        // One fluid particle deep INSIDE the body (at (1,0,0), dist 1 < radius 2).
        world.particles = {ParticleAt(1, 0, 0)};

        couple::ApplyBodyToFluid(world, kDt);

        // The particle is snapped to the body surface: |p − b.pos| == radius within an LSB (FxNormalize +
        // FxScale truncate toward zero, so the snapped length lands a few LSBs short of radius).
        const fpx::FxVec3 d = fpx::FxSub(world.particles[0].pos, world.bodies[0].pos);
        const fx dist = FxLength(d);
        const fx radius = world.bodies[0].radius;
        check(dist <= radius && dist > radius - 64,
              "ApplyBodyToFluid: an inside particle is snapped to the body surface (|p−b.pos| ~ radius)");
        // The particle was pushed in +x (it was at +x of the centre -> outward normal +x).
        check(world.particles[0].pos.x > FromInt(1), "ApplyBodyToFluid: the particle is displaced OUTWARD");
        // The drag reaction pulls the particle's velocity TOWARD the body's +x velocity (vel.x > 0).
        check(world.particles[0].vel.x > 0,
              "ApplyBodyToFluid: drag reaction drags the fluid toward the body velocity (+x)");
    }

    // ---- ApplyBodyToFluid: a particle OUTSIDE the body is UNTOUCHED (no push, no drag). ----
    {
        couple::CoupleWorld world;
        world.dt     = kDt;
        world.bodies = {BodyAt(0, 0, 0, 2)};
        world.particles = {ParticleAt(10, 0, 0)};        // dist 10 >> radius 2 -> outside
        const fpx::FxVec3 posBefore = world.particles[0].pos;
        const fpx::FxVec3 velBefore = world.particles[0].vel;
        couple::ApplyBodyToFluid(world, kDt);
        check(world.particles[0].pos.x == posBefore.x && world.particles[0].pos.y == posBefore.y &&
              world.particles[0].pos.z == posBefore.z, "ApplyBodyToFluid: an outside particle is UNTOUCHED (pos)");
        check(world.particles[0].vel.x == velBefore.x && world.particles[0].vel.y == velBefore.y &&
              world.particles[0].vel.z == velBefore.z, "ApplyBodyToFluid: an outside particle is UNTOUCHED (vel)");
    }

    // ---- ApplyBodyToFluid: a STATIC (boundary) particle inside a body is UNTOUCHED (dp 0, vel untouched). ----
    {
        couple::CoupleWorld world;
        world.dt     = kDt;
        world.bodies = {BodyAt(0, 0, 0, 2)};
        world.bodies[0].vel = fpx::FxVec3{3 * (int)kOne, 0, 0};
        fluid::FluidParticle sp = ParticleAt(1, 0, 0);   // INSIDE the body
        sp.flags = fluid::kFlagStatic;                   // but a fixed boundary
        world.particles = {sp};
        const fpx::FxVec3 posBefore = world.particles[0].pos;
        const fpx::FxVec3 velBefore = world.particles[0].vel;
        couple::ApplyBodyToFluid(world, kDt);
        check(world.particles[0].pos.x == posBefore.x && world.particles[0].pos.y == posBefore.y &&
              world.particles[0].pos.z == posBefore.z, "ApplyBodyToFluid: a STATIC particle is UNTOUCHED (pos)");
        check(world.particles[0].vel.x == velBefore.x && world.particles[0].vel.y == velBefore.y &&
              world.particles[0].vel.z == velBefore.z, "ApplyBodyToFluid: a STATIC particle is UNTOUCHED (vel)");
    }

    // ---- ApplyBodyToFluid: a non-dynamic body does NOT displace (it holds). ----
    {
        couple::CoupleWorld world;
        world.dt     = kDt;
        fpx::FxBody stat = BodyAt(0, 0, 0, 2);
        stat.flags = 0; stat.invMass = 0;                // NOT dynamic
        world.bodies = {stat};
        world.particles = {ParticleAt(1, 0, 0)};         // inside the (non-dynamic) body
        const fpx::FxVec3 posBefore = world.particles[0].pos;
        couple::ApplyBodyToFluid(world, kDt);
        check(world.particles[0].pos.x == posBefore.x,
              "ApplyBodyToFluid: a non-dynamic body does NOT displace the fluid (it holds)");
    }

    // ---- ApplyBodyToFluid: TWO bodies, fixed order — a particle inside body 0 only, projected to body 0's
    //      surface; a particle inside body 1 only, projected to body 1's surface (the fixed-order projection). --
    {
        couple::CoupleWorld world;
        world.dt     = kDt;
        // Two bodies well separated so no particle is inside both (the single-projection case is exact).
        world.bodies = {BodyAt(0, 0, 0, 2), BodyAt(20, 0, 0, 2)};
        world.particles = {ParticleAt(1, 0, 0), ParticleAt(21, 0, 0)};   // p0 in body0, p1 in body1
        couple::ApplyBodyToFluid(world, kDt);
        const fx r = world.bodies[0].radius;
        const fx d0 = FxLength(fpx::FxSub(world.particles[0].pos, world.bodies[0].pos));
        const fx d1 = FxLength(fpx::FxSub(world.particles[1].pos, world.bodies[1].pos));
        check(d0 <= r && d0 > r - 64, "ApplyBodyToFluid two bodies: p0 snapped to body 0's surface");
        check(d1 <= r && d1 > r - 64, "ApplyBodyToFluid two bodies: p1 snapped to body 1's surface");
    }

    // ---- ApplyBodyToFluid: dist==0 (a particle exactly at the body centre) -> the +Y FxNormalize fallback
    //      (deterministic) snaps it to (b.pos + (0,radius,0)). ----
    {
        couple::CoupleWorld world;
        world.dt     = kDt;
        world.bodies = {BodyAt(5, 5, 5, 2)};
        world.particles = {ParticleAt(5, 5, 5)};         // exactly at the centre -> dist 0
        couple::ApplyBodyToFluid(world, kDt);
        // +Y fallback: snapped to (5, 5+radius, 5) within an LSB.
        const fx r = world.bodies[0].radius;
        check(world.particles[0].pos.x == FromInt(5) && world.particles[0].pos.z == FromInt(5),
              "ApplyBodyToFluid dist==0: the +Y fallback keeps x,z at the centre");
        check(world.particles[0].pos.y > FromInt(5) + r - 64 && world.particles[0].pos.y <= FromInt(5) + r,
              "ApplyBodyToFluid dist==0: snapped to b.pos + (0, radius, 0) (the +Y fallback)");
    }

    // ---- MeasureFluidPenetration on a KNOWN overlap: penetration > 0; AND ApplyBodyToFluid RELIEVES it
    //      (penAfter < penBefore — the fluid is parted). ----
    {
        couple::CoupleWorld world;
        world.dt     = kDt;
        world.bodies = {BodyAt(0, 0, 0, 3)};             // radius 3
        // A small lattice of particles, several inside the body's sphere.
        for (int x = -2; x <= 2; ++x)
            for (int y = -2; y <= 2; ++y)
                world.particles.push_back(ParticleAt(x, y, 0));

        const couple::FluidPenetration before = couple::MeasureFluidPenetration(world);
        check(before.summed > 0 && before.peak > 0, "MeasureFluidPenetration: a known overlap -> penetration > 0");
        const uint32_t displaced = couple::CountDisplaced(world);
        check(displaced > 0, "CountDisplaced: the body contains fluid particles (displaced > 0)");

        couple::ApplyBodyToFluid(world, kDt);
        const couple::FluidPenetration after = couple::MeasureFluidPenetration(world);
        check(after.summed < before.summed,
              "ApplyBodyToFluid: the fluid is parted (summed penetration RELIEVED, penAfter < penBefore)");
    }

    // ---- determinism: two ApplyBodyToFluid runs over the SAME world -> byte-identical particles. ----
    {
        auto build = [&]() -> couple::CoupleWorld {
            couple::CoupleWorld w;
            w.dt = kDt;
            w.bodies = {BodyAt(0, 0, 0, 3)};
            w.bodies[0].vel = fpx::FxVec3{2 * (int)kOne, (int)kOne, 0};
            for (int x = -2; x <= 2; ++x)
                for (int y = -2; y <= 2; ++y)
                    w.particles.push_back(ParticleAt(x, y, 0));
            return w;
        };
        couple::CoupleWorld a = build(), b = build();
        couple::ApplyBodyToFluid(a, kDt);
        couple::ApplyBodyToFluid(b, kDt);
        bool same = (a.particles.size() == b.particles.size());
        for (size_t i = 0; same && i < a.particles.size(); ++i)
            if (std::memcmp(&a.particles[i], &b.particles[i], sizeof(fluid::FluidParticle)) != 0) same = false;
        check(same, "ApplyBodyToFluid is deterministic (two runs byte-identical)");
    }

    // ---- no-op: a body CLEAR of the fluid (and zero bodies) -> the fluid is unchanged. ----
    {
        couple::CoupleWorld world;
        world.dt     = kDt;
        world.bodies = {BodyAt(100, 100, 100, 2)};       // far from the pool
        for (int x = 0; x <= 3; ++x) world.particles.push_back(ParticleAt(x, 0, 0));
        std::vector<fluid::FluidParticle> before = world.particles;
        couple::ApplyBodyToFluid(world, kDt);
        bool unchanged = true;
        for (size_t i = 0; i < before.size(); ++i)
            if (std::memcmp(&before[i], &world.particles[i], sizeof(fluid::FluidParticle)) != 0) unchanged = false;
        check(unchanged, "ApplyBodyToFluid: a body clear of the fluid -> the fluid is UNCHANGED (no-op)");

        // Zero bodies -> also a no-op.
        couple::CoupleWorld nobody;
        nobody.dt = kDt;
        for (int x = 0; x <= 3; ++x) nobody.particles.push_back(ParticleAt(x, 0, 0));
        std::vector<fluid::FluidParticle> nb = nobody.particles;
        couple::ApplyBodyToFluid(nobody, kDt);
        bool nbUnchanged = true;
        for (size_t i = 0; i < nb.size(); ++i)
            if (std::memcmp(&nb[i], &nobody.particles[i], sizeof(fluid::FluidParticle)) != 0) nbUnchanged = false;
        check(nbUnchanged, "ApplyBodyToFluid: zero bodies -> the fluid is UNCHANGED (no-op)");
    }

    // ============================================================================================
    // ============ Slice CP4 — THE COUPLED STEP (the bobbing barrel) tests ========================
    // ============================================================================================
    // StepCouple composes the FL4 fluid sub-passes + CP2 fluid->body + CP3 body->fluid + the rigid integrate
    // in the LOCKED order. A body dropped over a small pool settles to a float line ABOVE the bed AND BOBS
    // (peak-to-trough > 0), the fluid density residual stays bounded (no explosion), the buoy=0 control sinks;
    // deterministic; the composed order is fixed.

    const int kCpIters = 3;

    // Build the coupled-step scene: a STATIC BASIN (static boundary walls + floor confine the pool so the
    // DYNAMIC fluid stays a coherent pool the body can bob ON — without walls a free pool spreads/thins and
    // the body sinks through) + a body dropped above it. The kernel ρ0 = the packed-lattice mean density (the
    // FL4 recipe). The body's invMass is kOne/iters so the K-per-step buoyancy/drag impulses (CP4 runs CP2's
    // AccumBodyForces in EACH of the K iters, per the locked order) balance the once-per-step gravity.
    auto buildCoupleWorld = [&](int bodyX, int bodyY, int bodyZ) -> couple::CoupleWorld {
        couple::CoupleWorld w;
        const int LO = 0, HI = 8;
        auto Part = [&](int x, int y, int z, bool stat) {
            fluid::FluidParticle p = ParticleAt(x, y, z);
            if (stat) { p.invMass = 0; p.flags = fluid::kFlagStatic; }
            return p;
        };
        for (int py = -1; py <= 6; ++py)
            for (int pz = -1; pz <= HI + 1; ++pz)
                for (int px = -1; px <= HI + 1; ++px) {
                    const bool wall = (px == -1 || px == HI + 1 || pz == -1 || pz == HI + 1 || py == -1);
                    const bool inside = (px >= LO && px <= HI && pz >= LO && pz <= HI && py >= 0 && py <= 5);
                    if (wall) w.particles.push_back(Part(px, py, pz, true));      // static basin wall/floor
                    else if (inside) w.particles.push_back(Part(px, py, pz, false));  // dynamic fluid
                }
        const fx kH = (fx)(2 * (int)kOne);   // smoothing radius h = 2.0 (the FL4 density-solve radius)
        const int kBins = fluid::kKernelBins;
        // ρ0 = the mean density of the packed initial lattice (the FL4 probe recipe).
        const fluid::FluidGrid pg = fluid::MakeGrid(w.particles, kH);
        const fluid::FluidCellTable pt = fluid::BuildCellTable(w.particles, pg);
        const fluid::FluidNeighborList pl = fluid::BuildNeighborList(w.particles, pg, pt, kH);
        const fluid::FluidKernel kProbe = fluid::BuildKernelTable(kH, kOne, kBins, kOne / 100);
        std::vector<fluid::fx> probeRho;
        fluid::ComputeDensity(w.particles, pl, kProbe, probeRho);
        const fluid::fx rho0 = fluid::MeanDensity(probeRho);
        w.kernel  = fluid::BuildKernelTable(kH, rho0, kBins, kOne / 100);
        w.gravity = kGravityDown;
        w.dt      = kDt;
        w.groundY = 0;
        fpx::FxBody b = BodyAt(bodyX, bodyY, bodyZ, 2);   // radius 2, dropped above the pool
        b.invMass = kOne / kCpIters;                       // heavier body -> the K-fold buoyancy balances gravity
        w.bodies = {b};
        return w;
    };

    const int kCpSteps = 400;
    const fx  kCpBedLine = (fx)(2 * (int)kOne);   // groundY + radius (the body's surface-rest line)

    // (a) the headline: a body over the pool settles to a float line ABOVE the bed AND BOBS; the fluid stays
    // coherent (density residual bounded). Track the body's min/max y over the run for the bob amplitude.
    {
        couple::CoupleWorld w = buildCoupleWorld(4, 9, 4);
        fx minY = w.bodies[0].pos.y, maxY = w.bodies[0].pos.y;
        for (int s = 0; s < kCpSteps; ++s) {
            couple::StepCouple(w, kDt, kCpIters);
            const fx y = w.bodies[0].pos.y;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }
        const couple::CoupleState st = couple::MeasureCoupleState(w);
        check(st.floatY > kCpBedLine + kOne / 4,
              "StepCouple: the body FLOATS above the bed by a clear margin (emergent buoyancy)");
        check(st.floatY < (fx)(20 * (int)kOne), "StepCouple: the float line is bounded (did NOT fly out)");
        // The bob: the body oscillated — its peak-to-trough y motion over the run is > 0 (NOT monotonic).
        const fx bob = maxY - minY;
        check(bob > 0, "StepCouple: the body BOBBED (peak-to-trough y motion > 0, it oscillated)");
        // The fluid stayed coherent: the density residual is bounded (no explosion) + above the ground.
        check(st.densityResidual >= 0, "StepCouple: the density residual is a valid (non-negative) stat");
        // The dynamic fluid stays bounded — it rests in the static basin (floor at y=-1) and does NOT explode
        // or tunnel out the bottom (CP4's locked order has no per-step CollidePlane; the basin walls confine).
        bool bounded = true;
        const fx kBasinFloor = -(fx)(2 * (int)kOne);   // a generous floor (the static basin floor is y=-1)
        const fx kCeil = (fx)(40 * (int)kOne);
        for (const fluid::FluidParticle& p : w.particles)
            if (!(p.flags & fluid::kFlagStatic) && (p.pos.y < kBasinFloor || p.pos.y > kCeil)) bounded = false;
        check(bounded, "StepCouple: the dynamic fluid stays bounded in the basin (coherent, no explosion)");
    }

    // (b) the buoy=0 control: a NON-DYNAMIC body (no buoyancy/integrate) over a pool — but a dynamic body
    // placed FAR from the pool (gathers 0 fluid particles for the whole fall) free-falls to the bed. Compare
    // a floated body vs a sunk (no-gather) control.
    {
        couple::CoupleWorld floated = buildCoupleWorld(4, 9, 4);
        couple::StepCoupleSteps(floated, kDt, kCpIters, kCpSteps);
        const fx floatY = couple::MeasureCoupleState(floated).floatY;

        couple::CoupleWorld sunk = buildCoupleWorld(40, 9, 4);   // far from the pool -> gathers 0
        couple::StepCoupleSteps(sunk, kDt, kCpIters, kCpSteps);
        const fx sunkY = couple::MeasureCoupleState(sunk).floatY;
        check(sunkY == kCpBedLine,
              "StepCouple control (no-gather): the body SINKS to the bed (groundY + radius)");
        check(floatY > sunkY, "StepCouple: buoyancy does work (the floated body settles ABOVE the control)");
    }

    // (c) determinism: two runs of the SAME coupled scene -> byte-identical fluid + body state.
    {
        couple::CoupleWorld a = buildCoupleWorld(4, 9, 4), b = buildCoupleWorld(4, 9, 4);
        couple::StepCoupleSteps(a, kDt, kCpIters, kCpSteps);
        couple::StepCoupleSteps(b, kDt, kCpIters, kCpSteps);
        bool same = (a.bodies.size() == b.bodies.size()) && (a.particles.size() == b.particles.size());
        if (same) same = (std::memcmp(&a.bodies[0], &b.bodies[0], sizeof(fpx::FxBody)) == 0);
        for (size_t i = 0; same && i < a.particles.size(); ++i)
            if (std::memcmp(&a.particles[i], &b.particles[i], sizeof(fluid::FluidParticle)) != 0) same = false;
        check(same, "StepCouple is deterministic (two runs byte-identical, fluid + body)");
    }

    // (d) the composed order is FIXED: StepCouple(world, dt, iters) == StepCoupleSteps(world, dt, iters, 1)
    // for one tick (the K-step driver mirrors the single tick).
    {
        couple::CoupleWorld a = buildCoupleWorld(4, 9, 4), b = buildCoupleWorld(4, 9, 4);
        couple::StepCouple(a, kDt, kCpIters);
        couple::StepCoupleSteps(b, kDt, kCpIters, 1);
        bool same = (std::memcmp(&a.bodies[0], &b.bodies[0], sizeof(fpx::FxBody)) == 0);
        for (size_t i = 0; same && i < a.particles.size(); ++i)
            if (std::memcmp(&a.particles[i], &b.particles[i], sizeof(fluid::FluidParticle)) != 0) same = false;
        check(same, "StepCouple == StepCoupleSteps(...,1) (the composed order is fixed)");
    }

    // ============================================================================================
    // ============ Slice CP5 — LOCKSTEP + ROLLBACK (the multi-body netcode HEADLINE) tests ========
    // ============================================================================================
    // The FL5/GR5 harness over CoupleWorld + StepCouple — the FIRST MULTI-BODY lockstep (the snapshot covers
    // BOTH the bodies AND the particles vectors). CoupleCommand kinds: kCmdBodyShove (add to body vel),
    // kCmdBodyMove (add to body pos), kCmdFluidWind (add to fluid particle vel).

    // ---- ApplyCoupleCommand: kCmdBodyShove adds to a DYNAMIC body's velocity. ----
    {
        couple::CoupleWorld w;
        w.bodies = {BodyAt(0, 0, 0, 2)};
        const fpx::FxVec3 velBefore = w.bodies[0].vel;
        couple::CoupleCommand c{0, couple::kCmdBodyShove, 0, fpx::FxVec3{5 * (int)kOne, 0, 0}};
        couple::ApplyCoupleCommand(w, c);
        check(w.bodies[0].vel.x == velBefore.x + 5 * (int)kOne && w.bodies[0].vel.y == velBefore.y,
              "ApplyCoupleCommand kCmdBodyShove: adds arg to the body velocity");
    }

    // ---- ApplyCoupleCommand: kCmdBodyMove adds to a DYNAMIC body's position. ----
    {
        couple::CoupleWorld w;
        w.bodies = {BodyAt(3, 4, 5, 2)};
        couple::CoupleCommand c{0, couple::kCmdBodyMove, 0, fpx::FxVec3{0, 2 * (int)kOne, 0}};
        couple::ApplyCoupleCommand(w, c);
        check(w.bodies[0].pos.x == FromInt(3) && w.bodies[0].pos.y == FromInt(4) + 2 * (int)kOne &&
              w.bodies[0].pos.z == FromInt(5),
              "ApplyCoupleCommand kCmdBodyMove: adds arg to the body position");
    }

    // ---- ApplyCoupleCommand: kCmdFluidWind adds to a (non-static) fluid particle's velocity. ----
    {
        couple::CoupleWorld w;
        w.particles = {ParticleAt(0, 0, 0)};
        couple::CoupleCommand c{0, couple::kCmdFluidWind, 0, fpx::FxVec3{0, 0, 7 * (int)kOne}};
        couple::ApplyCoupleCommand(w, c);
        check(w.particles[0].vel.z == 7 * (int)kOne,
              "ApplyCoupleCommand kCmdFluidWind: adds arg to the fluid particle velocity");
    }

    // ---- ApplyCoupleCommand: a STATIC body / STATIC particle is NEVER mutated. ----
    {
        couple::CoupleWorld w;
        fpx::FxBody stat = BodyAt(0, 0, 0, 2);
        stat.flags = 0; stat.invMass = 0;               // NOT dynamic -> static body
        w.bodies = {stat};
        fluid::FluidParticle sp = ParticleAt(1, 0, 0);
        sp.flags = fluid::kFlagStatic;                  // static boundary particle
        w.particles = {sp};
        const fpx::FxVec3 bVel = w.bodies[0].vel, bPos = w.bodies[0].pos;
        const fpx::FxVec3 pVel = w.particles[0].vel;
        couple::ApplyCoupleCommand(w, couple::CoupleCommand{0, couple::kCmdBodyShove, 0, fpx::FxVec3{9 * (int)kOne, 0, 0}});
        couple::ApplyCoupleCommand(w, couple::CoupleCommand{0, couple::kCmdBodyMove, 0, fpx::FxVec3{9 * (int)kOne, 0, 0}});
        couple::ApplyCoupleCommand(w, couple::CoupleCommand{0, couple::kCmdFluidWind, 0, fpx::FxVec3{9 * (int)kOne, 0, 0}});
        check(w.bodies[0].vel.x == bVel.x && w.bodies[0].pos.x == bPos.x,
              "ApplyCoupleCommand: a static (non-dynamic) body is never mutated");
        check(w.particles[0].vel.x == pVel.x,
              "ApplyCoupleCommand: a static fluid particle is never mutated");
    }

    // ---- ApplyCoupleCommand: out-of-range target / unknown kind -> a no-op (deterministic). ----
    {
        couple::CoupleWorld w;
        w.bodies    = {BodyAt(0, 0, 0, 2)};
        w.particles = {ParticleAt(0, 0, 0)};
        const fpx::FxVec3 bVel = w.bodies[0].vel;
        const fpx::FxVec3 pVel = w.particles[0].vel;
        // body target 5 (>= 1 body) -> no-op
        couple::ApplyCoupleCommand(w, couple::CoupleCommand{0, couple::kCmdBodyShove, 5, fpx::FxVec3{9 * (int)kOne, 0, 0}});
        // fluid target 5 (>= 1 particle) -> no-op
        couple::ApplyCoupleCommand(w, couple::CoupleCommand{0, couple::kCmdFluidWind, 5, fpx::FxVec3{9 * (int)kOne, 0, 0}});
        // unknown kind -> no-op
        couple::ApplyCoupleCommand(w, couple::CoupleCommand{0, 999u, 0, fpx::FxVec3{9 * (int)kOne, 0, 0}});
        check(w.bodies[0].vel.x == bVel.x && w.particles[0].vel.x == pVel.x,
              "ApplyCoupleCommand: out-of-range target / unknown kind -> no-op");
    }

    // ---- SnapshotCouple / RestoreCouple: a deep-copy round-trip of BOTH bodies AND particles. ----
    {
        couple::CoupleWorld w = buildCoupleWorld(4, 9, 4);
        couple::StepCoupleSteps(w, kDt, kCpIters, 20);   // a non-trivial mixed state
        const couple::CoupleSnapshot snap = couple::SnapshotCouple(w);
        // Mutate BOTH vectors.
        w.bodies[0].pos.x += 12345;
        w.particles[10].vel.y -= 9999;
        couple::RestoreCouple(w, snap);
        bool bodiesSame = (w.bodies.size() == snap.bodies.size());
        for (size_t i = 0; bodiesSame && i < w.bodies.size(); ++i)
            if (std::memcmp(&w.bodies[i], &snap.bodies[i], sizeof(fpx::FxBody)) != 0) bodiesSame = false;
        bool partsSame = (w.particles.size() == snap.particles.size());
        for (size_t i = 0; partsSame && i < w.particles.size(); ++i)
            if (std::memcmp(&w.particles[i], &snap.particles[i], sizeof(fluid::FluidParticle)) != 0) partsSame = false;
        check(bodiesSame, "SnapshotCouple/RestoreCouple: the BODIES vector round-trips BIT-EXACT");
        check(partsSame, "SnapshotCouple/RestoreCouple: the PARTICLES vector round-trips BIT-EXACT");
    }

    // ---- RunCoupleLockstep: two runs from the SAME init + stream are byte-identical (bodies AND fluid). ----
    {
        couple::CoupleWorld init = buildCoupleWorld(4, 9, 4);
        const int kTicks = 30;
        const std::vector<couple::CoupleCommand> stream = {
            couple::CoupleCommand{2,  couple::kCmdBodyShove, 0, fpx::FxVec3{6 * (int)kOne, 0, 0}},
            couple::CoupleCommand{8,  couple::kCmdBodyShove, 0, fpx::FxVec3{0, 0, 4 * (int)kOne}},
            couple::CoupleCommand{14, couple::kCmdBodyMove,  0, fpx::FxVec3{0, (int)kOne, 0}},
        };
        const couple::CoupleWorld a = couple::RunCoupleLockstep(init, stream, kTicks, kDt, kCpIters);
        const couple::CoupleWorld b = couple::RunCoupleLockstep(init, stream, kTicks, kDt, kCpIters);
        bool same = (a.bodies.size() == b.bodies.size()) && (a.particles.size() == b.particles.size());
        for (size_t i = 0; same && i < a.bodies.size(); ++i)
            if (std::memcmp(&a.bodies[i], &b.bodies[i], sizeof(fpx::FxBody)) != 0) same = false;
        for (size_t i = 0; same && i < a.particles.size(); ++i)
            if (std::memcmp(&a.particles[i], &b.particles[i], sizeof(fluid::FluidParticle)) != 0) same = false;
        check(same, "RunCoupleLockstep: two runs (inputs-only) are byte-identical (bodies AND fluid)");
    }

    // ---- RunCoupleRollback: corrected == authority BIT-EXACT (bodies AND fluid) AND mispredicted != authority. ----
    {
        couple::CoupleWorld init = buildCoupleWorld(4, 9, 4);
        const int kTicks = 30;
        const int kMispredictTick = 10;
        const std::vector<couple::CoupleCommand> authStream = {
            couple::CoupleCommand{2,  couple::kCmdBodyShove, 0, fpx::FxVec3{6 * (int)kOne, 0, 0}},
            couple::CoupleCommand{8,  couple::kCmdBodyShove, 0, fpx::FxVec3{0, 0, 4 * (int)kOne}},
            couple::CoupleCommand{14, couple::kCmdBodyMove,  0, fpx::FxVec3{0, (int)kOne, 0}},
        };
        std::vector<couple::CoupleCommand> mispredictStream = authStream;
        mispredictStream.push_back(couple::CoupleCommand{(uint32_t)kMispredictTick, couple::kCmdBodyShove, 0,
                                                         fpx::FxVec3{40 * (int)kOne, 0, 0}});  // WRONG strong shove

        const couple::CoupleWorld authority = couple::RunCoupleLockstep(init, authStream, kTicks, kDt, kCpIters);
        const couple::CoupleWorld corrected = couple::RunCoupleRollback(init, authStream, mispredictStream,
                                                                        kTicks, kMispredictTick, kDt, kCpIters);
        const couple::CoupleWorld mispredicted =
            couple::RunCoupleLockstep(init, mispredictStream, kTicks, kDt, kCpIters);

        bool correctedSame = (corrected.bodies.size() == authority.bodies.size()) &&
                             (corrected.particles.size() == authority.particles.size());
        for (size_t i = 0; correctedSame && i < authority.bodies.size(); ++i)
            if (std::memcmp(&corrected.bodies[i], &authority.bodies[i], sizeof(fpx::FxBody)) != 0) correctedSame = false;
        for (size_t i = 0; correctedSame && i < authority.particles.size(); ++i)
            if (std::memcmp(&corrected.particles[i], &authority.particles[i], sizeof(fluid::FluidParticle)) != 0)
                correctedSame = false;
        check(correctedSame, "RunCoupleRollback: corrected == authority BIT-EXACT (bodies AND fluid)");

        bool diverged = (std::memcmp(&mispredicted.bodies[0], &authority.bodies[0], sizeof(fpx::FxBody)) != 0);
        for (size_t i = 0; !diverged && i < authority.particles.size(); ++i)
            if (std::memcmp(&mispredicted.particles[i], &authority.particles[i], sizeof(fluid::FluidParticle)) != 0)
                diverged = true;
        check(diverged, "RunCoupleRollback: the mispredicted state DIFFERED from authority (a real divergence)");
    }

    if (g_fail == 0) std::printf("couple_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
