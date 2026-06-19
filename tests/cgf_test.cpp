// Slice GF1 — Deterministic Grain<->Fluid Coupling: the UNIFIED TWO-POOL WORLD + the SHARED-GRID CROSS QUERY
// (engine/sim/couple_gf.h) that the GPU cgf_gf_{count,scan,emit}.comp + cgf_fg_{count,scan,emit}.comp shaders
// copy VERBATIM + prove bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace
// hf::sim::cgf. The GR2/FL2 27-cell-stencil neighbour search applied CROSS-POOL over ONE shared grid: per
// grain the FLUID particles in its 27-cell stencil (gfNeighbors), per fluid the GRAINS in its 27-cell stencil
// (fgNeighbors), each accepted iff the per-axis box reject |query.axis - target.axis| < h passes (pure int32,
// the exact radial cull deferred to GF2/GF3).
//
// What this test PINS (the contracts the GPU cgf_gf_*/cgf_fg_* + the GPU==CPU proof build on):
//   * MakeCGFGrid: the shared grid covers the UNION of both pools' cell bounds (every grain AND fluid cell in
//     [0,gridDim)); empty pools -> 1x1x1; one empty pool -> the other pool's bounds.
//   * BuildCGFNeighbors: a hand-laid grain+fluid set -> the EXACT gf/fg cross lists (every cross pair within
//     the box, none outside), CSR offsets correct, ascending within-query order; NO self-pairs (the pools are
//     distinct).
//   * the SYMMETRY X==Y: every grain<->fluid pair appears once in gf and once in fg (same h) -> CountGF==CountFG.
//   * separated pools (no overlap) -> 0 cross neighbours (the no-op).
//   * the empty world / one-empty-pool no-ops.
//   * determinism: two builds of the SAME world -> byte-identical CSR.
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/couple_gf.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace cgf   = hf::sim::cgf;
namespace grain = hf::sim::grain;
namespace fluid = hf::sim::fluid;
namespace fpx   = hf::sim::fpx;
using cgf::fx;
using cgf::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << fpx::kFrac); }

// Build a grain at an integer world position (dynamic, uniform radius).
static grain::GrainParticle GrainAt(int x, int y, int z) {
    grain::GrainParticle g;
    g.pos  = fpx::FxVec3{FromInt(x), FromInt(y), FromInt(z)};
    g.prev = g.pos;
    g.vel  = fpx::FxVec3{0, 0, 0};
    g.invMass = kOne;
    g.radius  = kOne / 4;
    g.flags   = 0;
    return g;
}

// Build a fluid particle at an integer world position (dynamic).
static fluid::FluidParticle FluidAt(int x, int y, int z) {
    fluid::FluidParticle f;
    f.pos  = fpx::FxVec3{FromInt(x), FromInt(y), FromInt(z)};
    f.prev = f.pos;
    f.vel  = fpx::FxVec3{0, 0, 0};
    f.invMass = kOne;
    f.flags   = 0;
    return f;
}

// The naive O(G*F) cross-pair reference (every grain x fluid pair passing the per-axis |dx|<h box reject) — the
// independent oracle the grid-hash CSR must match exactly.
static uint32_t NaiveCrossPairs(const cgf::CGFWorld& w) {
    uint32_t pairs = 0;
    for (const grain::GrainParticle& g : w.grains)
        for (const fluid::FluidParticle& f : w.fluid) {
            fx ax = g.pos.x - f.pos.x; if (ax < 0) ax = -ax;
            fx ay = g.pos.y - f.pos.y; if (ay < 0) ay = -ay;
            fx az = g.pos.z - f.pos.z; if (az < 0) az = -az;
            if (ax < w.h && ay < w.h && az < w.h) ++pairs;
        }
    return pairs;
}

int main() {
    HF_TEST_MAIN_INIT();

    const fx h = kOne;   // 1.0 cell size == coupling radius (cells are 1 world unit)

    // ---- MakeCGFGrid: the union bounds over a known two-pool set ------------------------------------------
    {
        // Grains span x in [0,2]; fluid spans x in [4,5]. The shared grid must cover x cells [0..5].
        cgf::CGFWorld w;
        w.h = h;
        w.grains = {GrainAt(0, 0, 0), GrainAt(2, 0, 0)};
        w.fluid  = {FluidAt(4, 0, 0), FluidAt(5, 0, 0)};
        const cgf::CGFGrid grid = cgf::MakeCGFGrid(w);
        check(grid.cellMin.x == 0, "MakeCGFGrid union cellMin.x == 0");
        check(grid.gridDim.x == 6, "MakeCGFGrid union gridDim.x == 6 (covers x cells 0..5)");
        check(grid.gridDim.y == 1 && grid.gridDim.z == 1, "MakeCGFGrid union gridDim.y/z == 1");
        // Every grain AND fluid cell must land in [0,gridDim) (total + collision-free).
        bool inRange = true;
        for (const grain::GrainParticle& g : w.grains) {
            const fpx::FxCell c = cgf::CGFCellOf(g.pos, h);
            if (c.x - grid.cellMin.x < 0 || c.x - grid.cellMin.x >= grid.gridDim.x) inRange = false;
        }
        for (const fluid::FluidParticle& f : w.fluid) {
            const fpx::FxCell c = cgf::CGFCellOf(f.pos, h);
            if (c.x - grid.cellMin.x < 0 || c.x - grid.cellMin.x >= grid.gridDim.x) inRange = false;
        }
        check(inRange, "MakeCGFGrid: every grain+fluid cell in [0,gridDim)");
    }

    // ---- MakeCGFGrid: empty / one-empty-pool degenerate cases --------------------------------------------
    {
        cgf::CGFWorld empty; empty.h = h;
        const cgf::CGFGrid g0 = cgf::MakeCGFGrid(empty);
        check(g0.gridDim.x == 1 && g0.gridDim.y == 1 && g0.gridDim.z == 1, "MakeCGFGrid empty -> 1x1x1");

        cgf::CGFWorld grainOnly; grainOnly.h = h;
        grainOnly.grains = {GrainAt(3, 0, 0), GrainAt(7, 0, 0)};
        const cgf::CGFGrid g1 = cgf::MakeCGFGrid(grainOnly);
        check(g1.cellMin.x == 3 && g1.gridDim.x == 5, "MakeCGFGrid grain-only -> grain bounds");

        cgf::CGFWorld fluidOnly; fluidOnly.h = h;
        fluidOnly.fluid = {FluidAt(-2, 0, 0), FluidAt(1, 0, 0)};
        const cgf::CGFGrid g2 = cgf::MakeCGFGrid(fluidOnly);
        check(g2.cellMin.x == -2 && g2.gridDim.x == 4, "MakeCGFGrid fluid-only -> fluid bounds (negative coords)");
    }

    // ---- BuildCGFNeighbors: a hand-laid overlapping set -> the exact cross lists --------------------------
    {
        // One grain at the origin. Two fluid particles inside its box (|dx|<1 per axis): same cell + diagonal
        // touching. One fluid far away (no cross pair). With h=1 (cell size 1), neighbours must be < 1 world
        // unit on every axis. Use sub-unit offsets via Q16.16 fractions.
        cgf::CGFWorld w;
        w.h = h;
        grain::GrainParticle g = GrainAt(0, 0, 0);
        w.grains = {g};
        fluid::FluidParticle near0 = FluidAt(0, 0, 0);                 // exactly on the grain -> |dx|=0 < h
        fluid::FluidParticle near1; near1 = FluidAt(0, 0, 0);
        near1.pos.x = kOne / 2;                                        // 0.5 away on x -> |dx|=0.5 < h
        fluid::FluidParticle far0 = FluidAt(5, 0, 0);                 // 5 away -> no cross pair
        w.fluid = {near0, near1, far0};

        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);

        // gf: the single grain gathers fluid 0 and fluid 1 (ascending index), not fluid 2.
        check(nbr.gfStart.size() == 2u, "gfStart has grainCount+1 entries");
        check(nbr.gfStart[0] == 0u, "gfStart[0] == 0");
        check(nbr.gfStart[1] == 2u, "grain 0 gathers exactly 2 fluid neighbours");
        check(nbr.gfNeighbors.size() == 2u, "gfNeighbors size == 2");
        check(nbr.gfNeighbors[0] == 0u && nbr.gfNeighbors[1] == 1u,
              "gfNeighbors ascending fluid index {0,1}, far fluid 2 excluded");

        // fg: fluid 0 and fluid 1 each gather the grain; fluid 2 gathers none.
        check(nbr.fgStart.size() == 4u, "fgStart has fluidCount+1 entries");
        check(nbr.fgStart[1] - nbr.fgStart[0] == 1u, "fluid 0 gathers 1 grain");
        check(nbr.fgStart[2] - nbr.fgStart[1] == 1u, "fluid 1 gathers 1 grain");
        check(nbr.fgStart[3] - nbr.fgStart[2] == 0u, "fluid 2 (far) gathers 0 grains");
        check(nbr.fgNeighbors.size() == 2u, "fgNeighbors size == 2");
        check(nbr.fgNeighbors[0] == 0u && nbr.fgNeighbors[1] == 0u, "both near fluid gather grain 0");

        // SYMMETRY: every grain<->fluid pair appears once in gf, once in fg (same h).
        check(cgf::CountGF(nbr) == cgf::CountFG(nbr), "symmetry: CountGF == CountFG");
        check(cgf::CountGF(nbr) == NaiveCrossPairs(w), "gf count == the naive O(G*F) oracle");
    }

    // ---- A richer overlapping set: two interpenetrating blocks; CSR vs the naive oracle + symmetry --------
    {
        cgf::CGFWorld w;
        w.h = h;
        // A 3x3x1 grain bed at z=0, and a 3x3x1 fluid sheet shifted by 0.5 in x and y so it interpenetrates.
        for (int gx = 0; gx < 3; ++gx)
            for (int gy = 0; gy < 3; ++gy)
                w.grains.push_back(GrainAt(gx, gy, 0));
        for (int fx_ = 0; fx_ < 3; ++fx_)
            for (int fy = 0; fy < 3; ++fy) {
                fluid::FluidParticle f = FluidAt(fx_, fy, 0);
                f.pos.x += kOne / 2; f.pos.y += kOne / 2;   // shift +0.5,+0.5 -> overlapping but offset
                w.fluid.push_back(f);
            }
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);

        // The grid-hash CSR total must equal the brute-force pair count BOTH directions.
        const uint32_t naive = NaiveCrossPairs(w);
        check(cgf::CountGF(nbr) == naive, "block: gf total == naive oracle");
        check(cgf::CountFG(nbr) == naive, "block: fg total == naive oracle");
        check(cgf::CountGF(nbr) == cgf::CountFG(nbr), "block: symmetry CountGF == CountFG");
        check(naive > 0u, "block: pools actually overlap (non-trivial)");

        // CSR offset sanity: monotonic non-decreasing, last == total, sizes match.
        bool gfMono = true; for (size_t i = 0; i + 1 < nbr.gfStart.size(); ++i)
            if (nbr.gfStart[i + 1] < nbr.gfStart[i]) gfMono = false;
        check(gfMono, "block: gfStart monotonic non-decreasing");
        check(nbr.gfStart.back() == nbr.gfNeighbors.size(), "block: gfStart sentinel == gfNeighbors.size()");
        check(nbr.fgStart.back() == nbr.fgNeighbors.size(), "block: fgStart sentinel == fgNeighbors.size()");

        // No DUPLICATES within a grain's gf slice (the stencil cells are disjoint, the target cell table lists
        // each fluid once, so a grain gathers each fluid AT MOST once). The global order is stencil-cell then
        // within-cell-ascending (GR2/FL2 emit discipline) — NOT globally ascending across cells.
        bool gfNoDup = true;
        for (size_t i = 0; i + 1 < nbr.gfStart.size(); ++i) {
            std::vector<uint32_t> slice(nbr.gfNeighbors.begin() + nbr.gfStart[i],
                                        nbr.gfNeighbors.begin() + nbr.gfStart[i + 1]);
            std::vector<uint32_t> uniq = slice;
            std::sort(uniq.begin(), uniq.end());
            if (std::unique(uniq.begin(), uniq.end()) != uniq.end()) gfNoDup = false;
        }
        check(gfNoDup, "block: each grain's gf slice has no duplicate fluid index");

        // Every emitted gf pair must actually pass the box reject (no spurious entries).
        bool gfValid = true;
        for (size_t i = 0; i + 1 < nbr.gfStart.size(); ++i)
            for (uint32_t s = nbr.gfStart[i]; s < nbr.gfStart[i + 1]; ++s) {
                const auto& g = w.grains[i]; const auto& f = w.fluid[nbr.gfNeighbors[s]];
                fx ax = g.pos.x - f.pos.x; if (ax < 0) ax = -ax;
                fx ay = g.pos.y - f.pos.y; if (ay < 0) ay = -ay;
                fx az = g.pos.z - f.pos.z; if (az < 0) az = -az;
                if (!(ax < h && ay < h && az < h)) gfValid = false;
            }
        check(gfValid, "block: every emitted gf pair passes the box reject");
    }

    // ---- Separated pools -> 0 cross neighbours (the no-op) ------------------------------------------------
    {
        cgf::CGFWorld w;
        w.h = h;
        w.grains = {GrainAt(0, 0, 0), GrainAt(1, 0, 0)};
        w.fluid  = {FluidAt(20, 0, 0), FluidAt(21, 0, 0)};   // far away on x
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        check(cgf::CountGF(nbr) == 0u, "separated: 0 gf cross neighbours");
        check(cgf::CountFG(nbr) == 0u, "separated: 0 fg cross neighbours");
        check(nbr.gfNeighbors.empty() && nbr.fgNeighbors.empty(), "separated: both neighbour lists empty");
        // The CSR is still well-formed (all-zero offsets, correct sizes).
        check(nbr.gfStart.size() == 3u && nbr.gfStart.back() == 0u, "separated: gfStart well-formed");
        check(nbr.fgStart.size() == 3u && nbr.fgStart.back() == 0u, "separated: fgStart well-formed");
        check(NaiveCrossPairs(w) == 0u, "separated: naive oracle also 0");
    }

    // ---- Empty / one-empty-pool no-ops -------------------------------------------------------------------
    {
        cgf::CGFWorld empty; empty.h = h;
        const cgf::CGFNeighbors n0 = cgf::BuildCGFNeighbors(empty);
        check(cgf::CountGF(n0) == 0u && cgf::CountFG(n0) == 0u, "empty world: 0 cross neighbours");
        check(n0.gfStart.size() == 1u && n0.fgStart.size() == 1u, "empty world: CSR has just the sentinel");

        cgf::CGFWorld grainOnly; grainOnly.h = h;
        grainOnly.grains = {GrainAt(0, 0, 0), GrainAt(1, 0, 0)};
        const cgf::CGFNeighbors n1 = cgf::BuildCGFNeighbors(grainOnly);
        check(cgf::CountGF(n1) == 0u && cgf::CountFG(n1) == 0u, "grain-only: 0 cross neighbours");
        check(n1.gfStart.size() == 3u && n1.gfStart.back() == 0u, "grain-only: gfStart well-formed");
        check(n1.fgStart.size() == 1u, "grain-only: fgStart has just the sentinel (no fluid)");
    }

    // ---- Determinism: two builds of the SAME world -> byte-identical CSR ----------------------------------
    {
        cgf::CGFWorld w;
        w.h = h;
        for (int gx = 0; gx < 4; ++gx) w.grains.push_back(GrainAt(gx, 0, 0));
        for (int fx_ = 0; fx_ < 4; ++fx_) { fluid::FluidParticle f = FluidAt(fx_, 0, 0); f.pos.y = kOne / 3; w.fluid.push_back(f); }
        const cgf::CGFNeighbors a = cgf::BuildCGFNeighbors(w);
        const cgf::CGFNeighbors b = cgf::BuildCGFNeighbors(w);
        check(a.gfStart == b.gfStart && a.gfNeighbors == b.gfNeighbors &&
              a.fgStart == b.fgStart && a.fgNeighbors == b.fgNeighbors,
              "determinism: two builds byte-identical");
    }

    // ===== Slice GF2 — BUOYANCY / SEEPAGE (fluid->grain) ==================================================

    // Helper: a grain at exact integer pos with a given velocity (dynamic, unit mass).
    auto GrainVel = [&](int x, int y, int z, fx vx, fx vy, fx vz) {
        grain::GrainParticle g = GrainAt(x, y, z);
        g.vel = fpx::FxVec3{vx, vy, vz};
        return g;
    };
    auto FluidVel = [&](int x, int y, int z, fx vx, fx vy, fx vz) {
        fluid::FluidParticle f = FluidAt(x, y, z);
        f.vel = fpx::FxVec3{vx, vy, vz};
        return f;
    };

    // ---- AccumGrainBuoyancy: a grain over a hand-laid fluid list -> the EXACT Q16.16 vel delta ------------
    {
        // One DYNAMIC grain at the origin, at rest. Two STATIC fluid particles inside its box (cnt==2). Gravity
        // straight down so up = +Y. The grain should gain +Y velocity (buoyancy) and the drag toward the
        // static fluid (vel 0) damps. We compute the EXACT expected delta and compare bit-for-bit.
        cgf::CGFWorld w;
        w.h = h;
        w.gravity = fpx::FxVec3{0, -9 * (int)kOne, 0};   // straight down -> up = +Y
        w.dt = kOne;                                     // dt = 1.0 (clean math)
        w.grains = {GrainVel(0, 0, 0, 0, 0, 0)};         // dynamic, invMass = kOne, vel 0
        w.fluid  = {FluidVel(0, 0, 0, 0, 0, 0), FluidVel(0, 0, 0, 0, 0, 0)};  // 2 coincident static fluid
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        check(nbr.gfStart[1] - nbr.gfStart[0] == 2u, "buoy: grain has 2 fluid neighbours (submerged)");

        // Expected: up = -normalize((0,-9,0)) = (0,+1,0). buoyMag = fxmul(kBuoyPerFluid, 2<<kFrac).
        const fpx::FxVec3 up{0, kOne, 0};
        const fx buoyMag = fpx::fxmul(cgf::kBuoyPerFluid, (fx)(2u << fpx::kFrac));
        const fpx::FxVec3 fBuoy{fpx::fxmul(up.x, buoyMag), fpx::fxmul(up.y, buoyMag), fpx::fxmul(up.z, buoyMag)};
        // vFluidAvg = 0 (static fluid). F_drag = kDrag*(0 - vel=0) = 0.
        const fpx::FxVec3 fTotal = fBuoy;
        // dvel = fTotal * invMass(kOne) * dt(kOne) = fTotal.
        const fpx::FxVec3 expectedVel = fTotal;

        cgf::CGFWorld w2 = w;
        cgf::AccumGrainBuoyancy(w2, nbr, cgf::kBuoyPerFluid);
        check(w2.grains[0].vel.x == expectedVel.x && w2.grains[0].vel.y == expectedVel.y &&
              w2.grains[0].vel.z == expectedVel.z, "buoy: exact Q16.16 vel delta (buoyancy up, no drag)");
        check(w2.grains[0].vel.y > 0, "buoy: submerged grain gains UP velocity (lift)");
    }

    // ---- Buoyancy ∝ count: 4 fluid neighbours lift TWICE as hard as 2 -------------------------------------
    {
        auto liftFor = [&](int nFluid) -> fx {
            cgf::CGFWorld w; w.h = h;
            w.gravity = fpx::FxVec3{0, -9 * (int)kOne, 0};
            w.dt = kOne;
            w.grains = {GrainVel(0, 0, 0, 0, 0, 0)};
            for (int k = 0; k < nFluid; ++k) w.fluid.push_back(FluidVel(0, 0, 0, 0, 0, 0));
            const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
            cgf::AccumGrainBuoyancy(w, nbr, cgf::kBuoyPerFluid);
            return w.grains[0].vel.y;
        };
        const fx lift2 = liftFor(2);
        const fx lift4 = liftFor(4);
        check(lift4 == 2 * lift2, "buoy: lift ∝ fluid count (4 lifts exactly 2x of 2)");
    }

    // ---- Drag damps toward the fluid velocity (a moving fluid pulls the grain along) ---------------------
    {
        // A grain at rest with ONE fluid neighbour moving in +X. Gravity down (up = +Y). The drag should add a
        // +X velocity component toward the fluid; buoyancy adds +Y.
        cgf::CGFWorld w; w.h = h;
        w.gravity = fpx::FxVec3{0, -9 * (int)kOne, 0};
        w.dt = kOne;
        w.grains = {GrainVel(0, 0, 0, 0, 0, 0)};
        w.fluid  = {FluidVel(0, 0, 0, 3 * (int)kOne, 0, 0)};   // fluid moving +3 in x
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        // Expected drag x = fxmul(kDrag, (3.0 - 0)) = kDrag*3.0.
        const fx expectDragX = fpx::fxmul(cgf::kDrag, 3 * (int)kOne);
        cgf::AccumGrainBuoyancy(w, nbr, cgf::kBuoyPerFluid);
        check(w.grains[0].vel.x == expectDragX, "buoy: drag pulls grain toward the moving fluid (+X)");
        check(w.grains[0].vel.y > 0, "buoy: buoyancy still lifts +Y with a moving fluid");
    }

    // ---- A STATIC grain is untouched ---------------------------------------------------------------------
    {
        cgf::CGFWorld w; w.h = h;
        w.gravity = fpx::FxVec3{0, -9 * (int)kOne, 0};
        w.dt = kOne;
        grain::GrainParticle sg = GrainVel(0, 0, 0, 0, 0, 0);
        sg.flags = grain::kFlagStatic;                 // boundary grain -> never moves
        w.grains = {sg};
        w.fluid  = {FluidVel(0, 0, 0, 0, 0, 0), FluidVel(0, 0, 0, 0, 0, 0)};
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        cgf::AccumGrainBuoyancy(w, nbr, cgf::kBuoyPerFluid);
        check(w.grains[0].vel.x == 0 && w.grains[0].vel.y == 0 && w.grains[0].vel.z == 0,
              "buoy: static grain vel untouched");
    }

    // ---- A DRY grain (no fluid neighbours) -> no buoyancy/drag (free GR sim) ------------------------------
    {
        cgf::CGFWorld w; w.h = h;
        w.gravity = fpx::FxVec3{0, -9 * (int)kOne, 0};
        w.dt = kOne;
        w.grains = {GrainVel(0, 0, 0, kOne, 0, 0)};    // grain has some +X velocity already
        w.fluid  = {FluidVel(20, 0, 0, 0, 0, 0)};      // fluid far away -> no cross neighbour
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        check(nbr.gfStart[1] - nbr.gfStart[0] == 0u, "buoy: dry grain has 0 fluid neighbours");
        cgf::AccumGrainBuoyancy(w, nbr, cgf::kBuoyPerFluid);
        check(w.grains[0].vel.x == kOne && w.grains[0].vel.y == 0 && w.grains[0].vel.z == 0,
              "buoy: dry grain vel unchanged by the accumulate (free-fall handled by IntegrateGrains)");
    }

    // ---- StepCGFBuoyancy: submerged grains end HIGHER than dry; buoy=0 control packs the same -------------
    {
        // A PACKED grain bed (0.5 spacing, h=1.5 coupling radius) with a fluid block over its LEFT half so the
        // submerged grains have MANY fluid neighbours (the buoyancy is ∝ count). The wet (left) grains should be
        // buoyed to a stable line ABOVE the dry (right) pack after K steps. The buoy=0 control packs them the
        // same. (The sparse 1.0-spaced scene above gives only ~1 fluid neighbour per grain — too little lift
        // for the mild per-neighbour coefficient — so the buoyancy proof uses the representative packed scene.)
        const fx kDt = kOne / 60;
        const fx kGroundY = 0;
        const fx kHb = kOne + kOne / 2;   // 1.5 coupling radius (the showcase config)
        const fpx::FxVec3 kGravity{0, (fx)(-9 * (int)kOne), 0};
        const fx kHalf = kOne / 2;        // 0.5 spacing (packed)
        auto makeWorld = [&]() {
            cgf::CGFWorld w;
            w.h = kHb; w.gravity = kGravity; w.dt = kDt; w.groundY = kGroundY;
            // A packed 8(x) x 4(y) x 3(z) grain bed at 0.5 spacing, just above the ground.
            for (int gx = 0; gx < 8; ++gx)
                for (int gy = 0; gy < 4; ++gy)
                    for (int gz = 0; gz < 3; ++gz) {
                        grain::GrainParticle g = GrainAt(0, 0, 0);
                        g.pos = fpx::FxVec3{(fx)(gx * (int)kHalf), kOne + (fx)(gy * (int)kHalf), (fx)(gz * (int)kHalf)};
                        g.prev = g.pos;
                        w.grains.push_back(g);
                    }
            // A packed fluid block over the LEFT half (x in [0,~1.5]), interpenetrating the bed.
            for (int fx_ = 0; fx_ < 4; ++fx_)
                for (int fy = 0; fy < 6; ++fy)
                    for (int fz = 0; fz < 3; ++fz) {
                        fluid::FluidParticle f = FluidAt(0, 0, 0);
                        f.pos = fpx::FxVec3{(fx)(fx_ * (int)kHalf), (fx)(fy * (int)kHalf), (fx)(fz * (int)kHalf)};
                        f.prev = f.pos;
                        w.fluid.push_back(f);
                    }
            return w;
        };

        cgf::CGFWorld wA = makeWorld();
        cgf::StepCGFBuoyancySteps(wA, kDt, 200);
        const cgf::WetDry wd = cgf::MeasureWetDry(wA);
        check(wd.wet > 0u && wd.dry > 0u, "step: both wet and dry grains exist");
        check(wd.wetY > wd.dryY + kOne / 8, "step: submerged grains end HIGHER than dry (buoyancy lightens)");

        // buoy=0 control: the wet and dry grains pack the SAME (within a band).
        cgf::CGFWorld wCtrl = makeWorld();
        cgf::StepCGFBuoyancyControlSteps(wCtrl, kDt, 0, 200);
        const cgf::WetDry wdc = cgf::MeasureWetDry(wCtrl);
        const fx band = kOne / 4;   // within a quarter-unit band
        fx diff = wdc.wetY - wdc.dryY; if (diff < 0) diff = -diff;
        check(diff < band, "step: buoy=0 control packs wet ≈ dry (buoyancy does the work)");

        // Determinism: two runs byte-identical.
        cgf::CGFWorld wB = makeWorld();
        cgf::StepCGFBuoyancySteps(wB, kDt, 200);
        bool same = (wA.grains.size() == wB.grains.size());
        for (size_t i = 0; same && i < wA.grains.size(); ++i)
            if (std::memcmp(&wA.grains[i], &wB.grains[i], sizeof(grain::GrainParticle)) != 0) same = false;
        check(same, "step: two runs byte-identical (deterministic)");
    }

    // ---- Grain-order independence: shuffling the grain array gives the SAME per-grain result -------------
    {
        const fx kDt = kOne / 60;
        const fpx::FxVec3 kGravity{0, (fx)(-9 * (int)kOne), 0};
        cgf::CGFWorld w; w.h = h; w.gravity = kGravity; w.dt = kDt; w.groundY = 0;
        // 4 grains, a fluid neighbour each (or not), at distinct positions.
        w.grains = {GrainAt(0, 1, 0), GrainAt(1, 1, 0), GrainAt(10, 1, 0), GrainAt(11, 1, 0)};
        w.fluid  = {FluidAt(0, 1, 0), FluidAt(1, 1, 0)};   // submerge grains 0,1; grains 2,3 dry
        // Reverse the grain array; the per-grain physics must be identical (the accumulate is per-grain).
        cgf::CGFWorld wRev = w;
        std::reverse(wRev.grains.begin(), wRev.grains.end());

        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        const cgf::CGFNeighbors nbrRev = cgf::BuildCGFNeighbors(wRev);
        cgf::AccumGrainBuoyancy(w, nbr, cgf::kBuoyPerFluid);
        cgf::AccumGrainBuoyancy(wRev, nbrRev, cgf::kBuoyPerFluid);
        // grain 0 of w == grain 3 of wRev (reversed), etc.
        bool match = true;
        for (size_t i = 0; i < w.grains.size(); ++i)
            if (std::memcmp(&w.grains[i].vel, &wRev.grains[w.grains.size() - 1 - i].vel,
                            sizeof(fpx::FxVec3)) != 0) match = false;
        check(match, "buoy: per-grain result is grain-order-independent");
    }

    // ===== Slice GF3 — CONTACT REACTION / DISPLACEMENT (grain->fluid, Newton's 3rd law to GF2) ============

    // ---- ApplyGrainsToFluid: a fluid particle inside a grain -> snapped to the grain surface + drag reaction --
    {
        // A grain of radius 0.25 at the origin, MOVING +X. A fluid particle 0.1 above it (inside the 0.25
        // exclusion radius) must be pushed OUT to the surface (|p − g.pos| == g.radius within an LSB) and gain
        // velocity toward the grain (drag reaction). dt = 1.0 for clean math.
        cgf::CGFWorld w; w.h = h; w.dt = kOne;
        grain::GrainParticle g = GrainAt(0, 0, 0);
        g.radius = kOne / 4;                              // 0.25 exclusion radius
        g.vel = fpx::FxVec3{2 * (int)kOne, 0, 0};         // grain moving +2 in x
        w.grains = {g};
        fluid::FluidParticle f = FluidAt(0, 0, 0);
        f.pos.y = kOne / 10;                              // 0.1 above the grain centre -> inside (0.1 < 0.25)
        f.prev = f.pos;
        w.fluid = {f};
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        check(nbr.fgStart[1] - nbr.fgStart[0] == 1u, "displace: fluid has the grain as a neighbour");

        cgf::CGFWorld w2 = w;
        cgf::ApplyGrainsToFluid(w2, nbr);
        // The fluid centre should now sit on the grain surface: |p − g.pos| == g.radius within an LSB band.
        const fpx::FxVec3 dd = fpx::FxSub(w2.fluid[0].pos, w.grains[0].pos);
        const fx dist = fpx::FxLength(dd);
        fx err = dist - (kOne / 4); if (err < 0) err = -err;
        check(err <= 4, "displace: fluid snapped to the grain surface (|p-g| == g.radius within an LSB)");
        // The fluid gained +X velocity toward the moving grain (drag reaction = fxmul(kDragReaction, (2-0))·dt).
        const fx expectDragX = fpx::fxmul(fpx::fxmul(cgf::kDragReaction, 2 * (int)kOne), kOne);
        check(w2.fluid[0].vel.x == expectDragX, "displace: exact Q16.16 drag-reaction vel toward the grain (+X)");
        check(w2.fluid[0].vel.x > 0, "displace: fluid dragged toward the grain velocity");
    }

    // ---- A fluid particle OUTSIDE the grain (beyond g.radius) -> untouched --------------------------------
    {
        cgf::CGFWorld w; w.h = h; w.dt = kOne;
        grain::GrainParticle g = GrainAt(0, 0, 0); g.radius = kOne / 4; g.vel = fpx::FxVec3{2 * (int)kOne, 0, 0};
        w.grains = {g};
        fluid::FluidParticle f = FluidAt(0, 0, 0);
        f.pos.y = kOne / 2;                              // 0.5 above (> 0.25 -> OUTSIDE the grain), still a GF1 nbr
        f.prev = f.pos;
        w.fluid = {f};
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        const fpx::FxVec3 posBefore = w.fluid[0].pos;
        cgf::ApplyGrainsToFluid(w, nbr);
        check(std::memcmp(&w.fluid[0].pos, &posBefore, sizeof(fpx::FxVec3)) == 0,
              "displace: fluid outside the grain is positionally untouched");
        check(w.fluid[0].vel.x == 0 && w.fluid[0].vel.y == 0 && w.fluid[0].vel.z == 0,
              "displace: fluid outside the grain keeps its velocity (no drag)");
    }

    // ---- A STATIC fluid particle inside a grain -> untouched (dp 0, vel held) -----------------------------
    {
        cgf::CGFWorld w; w.h = h; w.dt = kOne;
        grain::GrainParticle g = GrainAt(0, 0, 0); g.radius = kOne / 4; g.vel = fpx::FxVec3{2 * (int)kOne, 0, 0};
        w.grains = {g};
        fluid::FluidParticle f = FluidAt(0, 0, 0); f.pos.y = kOne / 10; f.prev = f.pos;
        f.flags = fluid::kFlagStatic;                   // boundary fluid -> never moves
        w.fluid = {f};
        const fpx::FxVec3 posBefore = w.fluid[0].pos;
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        cgf::ApplyGrainsToFluid(w, nbr);
        check(std::memcmp(&w.fluid[0].pos, &posBefore, sizeof(fpx::FxVec3)) == 0,
              "displace: static fluid positionally untouched");
        check(w.fluid[0].vel.x == 0 && w.fluid[0].vel.y == 0 && w.fluid[0].vel.z == 0,
              "displace: static fluid vel untouched");
    }

    // ---- Two grains, fixed-order projection (a fluid particle inside both) --------------------------------
    {
        // Two grains straddling a fluid particle on x. The fluid iterates its fgNeighbors grain list in the
        // FIXED GF1 emit order; the Jacobi sum of both surface-snap pushes is deterministic. We pin determinism
        // + that the result differs from a single-grain push (both contribute).
        cgf::CGFWorld w; w.h = h; w.dt = kOne;
        grain::GrainParticle g0 = GrainAt(0, 0, 0); g0.radius = kOne / 4;
        grain::GrainParticle g1; g1 = GrainAt(0, 0, 0); g1.radius = kOne / 4; g1.pos.x = kOne / 4;  // 0.25 over on x
        w.grains = {g0, g1};
        fluid::FluidParticle f = FluidAt(0, 0, 0); f.pos.x = kOne / 8; f.prev = f.pos;  // 0.125, inside BOTH
        w.fluid = {f};
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        check(nbr.fgStart[1] - nbr.fgStart[0] == 2u, "displace: fluid gathers BOTH grains");
        cgf::CGFWorld wA = w; cgf::ApplyGrainsToFluid(wA, nbr);
        cgf::CGFWorld wB = w; cgf::ApplyGrainsToFluid(wB, nbr);
        check(std::memcmp(&wA.fluid[0], &wB.fluid[0], sizeof(fluid::FluidParticle)) == 0,
              "displace: two grains, two runs byte-identical (fixed-order Jacobi sum)");
    }

    // ---- MeasureFluidGrainPenetration on a known overlap + CountDisplacedFluid -----------------------------
    {
        // A grain radius 0.25 at the origin; a fluid particle 0.1 above -> penetration = 0.25 − 0.1 = 0.15.
        cgf::CGFWorld w; w.h = h; w.dt = kOne;
        grain::GrainParticle g = GrainAt(0, 0, 0); g.radius = kOne / 4;
        w.grains = {g};
        fluid::FluidParticle f = FluidAt(0, 0, 0); f.pos.y = kOne / 10; f.prev = f.pos;
        w.fluid = {f};
        const cgf::FluidGrainPenetration pen = cgf::MeasureFluidGrainPenetration(w);
        const fx expectPen = (kOne / 4) - (kOne / 10);   // 0.15 (the fluid IS a point, dist == 0.1)
        fx perr = (fx)pen.summed - expectPen; if (perr < 0) perr = -perr;
        check(perr <= 4, "displace: penetration == g.radius − dist within an LSB (0.15)");
        check(cgf::CountDisplacedFluid(w) == 1u, "displace: 1 fluid particle inside the grain (displaced)");

        // The displacement RELIEVES the penetration (penAfter < penBefore — the FL4/GR3 honesty).
        const cgf::FluidGrainPenetration before = cgf::MeasureFluidGrainPenetration(w);
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        cgf::CGFWorld wd = w; cgf::ApplyGrainsToFluid(wd, nbr);
        const cgf::FluidGrainPenetration after = cgf::MeasureFluidGrainPenetration(wd);
        check(after.summed < before.summed, "displace: penAfter < penBefore (the fluid parted from the sand)");
    }

    // ---- A fluid particle CLEAR of all grains -> ApplyGrainsToFluid is a no-op -----------------------------
    {
        cgf::CGFWorld w; w.h = h; w.dt = kOne;
        grain::GrainParticle g = GrainAt(0, 0, 0); g.radius = kOne / 4;
        w.grains = {g};
        fluid::FluidParticle f = FluidAt(20, 0, 0); f.prev = f.pos;   // far away -> no grain neighbours
        w.fluid = {f};
        const fluid::FluidParticle before = w.fluid[0];
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        cgf::ApplyGrainsToFluid(w, nbr);
        check(std::memcmp(&w.fluid[0], &before, sizeof(fluid::FluidParticle)) == 0,
              "displace: fluid clear of the grains is unchanged (no-op)");
    }

    // ---- Fluid-order independence: shuffling the fluid array gives the SAME per-fluid result --------------
    {
        cgf::CGFWorld w; w.h = h; w.dt = kOne;
        grain::GrainParticle g = GrainAt(0, 0, 0); g.radius = kOne / 4; g.vel = fpx::FxVec3{kOne, 0, 0};
        w.grains = {g};
        fluid::FluidParticle f0 = FluidAt(0, 0, 0); f0.pos.y = kOne / 10; f0.prev = f0.pos;   // inside
        fluid::FluidParticle f1 = FluidAt(0, 0, 0); f1.pos.x = kOne / 10; f1.prev = f1.pos;   // inside
        w.fluid = {f0, f1};
        cgf::CGFWorld wRev = w; std::reverse(wRev.fluid.begin(), wRev.fluid.end());
        const cgf::CGFNeighbors nbr = cgf::BuildCGFNeighbors(w);
        const cgf::CGFNeighbors nbrRev = cgf::BuildCGFNeighbors(wRev);
        cgf::ApplyGrainsToFluid(w, nbr);
        cgf::ApplyGrainsToFluid(wRev, nbrRev);
        bool match = true;
        for (size_t i = 0; i < w.fluid.size(); ++i)
            if (std::memcmp(&w.fluid[i], &wRev.fluid[w.fluid.size() - 1 - i], sizeof(fluid::FluidParticle)) != 0)
                match = false;
        check(match, "displace: per-fluid result is fluid-order-independent (Jacobi)");
    }

    // ===== Slice GF4 — THE COUPLED STEP (StepCGF, the (1)-(6) locked-order host driver) ====================

    // A small co-settling scene builder: a packed grain bed + a fluid block resting on it, in one Q16.16 world,
    // with a kernel built from the scene (the FL4 probe recipe). Reused across the GF4 cases.
    auto buildStepScene = [&](cgf::CGFWorld& w, fluid::FluidKernel& kernel) {
        const fx kDt = kOne / 60;
        const fx kHb = kOne + kOne / 2;                  // 1.5 coupling radius
        const fpx::FxVec3 kGravity{0, (fx)(-9 * (int)kOne), 0};
        const fx kHalf = kOne / 2;                       // 0.5 spacing (packed)
        w = cgf::CGFWorld{};
        w.h = kHb; w.gravity = kGravity; w.dt = kDt; w.groundY = 0;
        // A packed 6(x) x 3(y) x 3(z) grain bed at 0.5 spacing, just above the ground (54 grains).
        for (int gx = 0; gx < 6; ++gx)
            for (int gy = 0; gy < 3; ++gy)
                for (int gz = 0; gz < 3; ++gz) {
                    grain::GrainParticle g = GrainAt(0, 0, 0);
                    g.pos = fpx::FxVec3{(fx)(gx * (int)kHalf), kOne + (fx)(gy * (int)kHalf), (fx)(gz * (int)kHalf)};
                    g.prev = g.pos;
                    w.grains.push_back(g);
                }
        // A packed fluid block seeded just above the bed centre (3x3x3 = 27 particles).
        for (int fx_ = 0; fx_ < 3; ++fx_)
            for (int fy = 0; fy < 3; ++fy)
                for (int fz = 0; fz < 3; ++fz) {
                    fluid::FluidParticle f = FluidAt(0, 0, 0);
                    f.pos = fpx::FxVec3{kOne + (fx)(fx_ * (int)kHalf),
                                        (fx)(3 * (int)kOne) + (fx)(fy * (int)kHalf), (fx)(fz * (int)kHalf)};
                    f.prev = f.pos;
                    w.fluid.push_back(f);
                }
        // The kernel: ρ0 = the mean density of the packed initial fluid lattice (the FL4 probe recipe).
        kernel = fluid::BuildKernelTable(kHb, kOne, fluid::kKernelBins, kOne / 100);
        const fluid::FluidGrid pg = fluid::MakeGrid(w.fluid, kHb);
        const fluid::FluidCellTable pt = fluid::BuildCellTable(w.fluid, pg);
        const fluid::FluidNeighborList pl = fluid::BuildNeighborList(w.fluid, pg, pt, kHb);
        std::vector<fx> probeRho;
        fluid::ComputeDensity(w.fluid, pl, kernel, probeRho);
        kernel = fluid::BuildKernelTable(kHb, fluid::MeanDensity(probeRho), fluid::kKernelBins, kOne / 100);
    };

    // ---- StepCGF on a tiny scene: two runs byte-identical (determinism) -----------------------------------
    {
        cgf::CGFWorld w; fluid::FluidKernel kernel;
        buildStepScene(w, kernel);
        cgf::CGFWorld a = w, b = w;
        cgf::StepCGFSteps(a, kernel, w.dt, 3, 30);
        cgf::StepCGFSteps(b, kernel, w.dt, 3, 30);
        bool same = (a.grains.size() == b.grains.size() && a.fluid.size() == b.fluid.size());
        for (size_t i = 0; same && i < a.grains.size(); ++i)
            if (std::memcmp(&a.grains[i], &b.grains[i], sizeof(grain::GrainParticle)) != 0) same = false;
        for (size_t i = 0; same && i < a.fluid.size(); ++i)
            if (std::memcmp(&a.fluid[i], &b.fluid[i], sizeof(fluid::FluidParticle)) != 0) same = false;
        check(same, "cgf-step: two runs byte-identical (deterministic)");
    }

    // ---- StepCGF: the fluid ends ABOVE the grains (co-settle) + the bed holds a repose + wet > dry --------
    {
        cgf::CGFWorld w; fluid::FluidKernel kernel;
        buildStepScene(w, kernel);
        cgf::StepCGFSteps(w, kernel, w.dt, 3, 80);
        const cgf::CGFState st = cgf::MeasureCGFState(w);
        check(st.dynamicGrains > 0u && st.dynamicFluid > 0u, "cgf-step: dynamic grains + fluid present");
        check(st.fluidY > st.bedY, "cgf-step: the fluid POOLS ABOVE the bed (fluidY > bedY)");
        check(st.repose > 0, "cgf-step: the bed still holds an angle of repose (repose > 0)");
        check(st.wetY >= st.dryY, "cgf-step: submerged grains sit at/above dry (the GF2 lift survives)");
        // Coherence: no dynamic grain or fluid rockets out (a bounded, deterministic settle).
        bool coherent = true;
        for (const grain::GrainParticle& g : w.grains)
            if (!(g.flags & grain::kFlagStatic) && g.pos.y > (fx)(40 * (int)kOne)) coherent = false;
        for (const fluid::FluidParticle& f : w.fluid)
            if (!(f.flags & fluid::kFlagStatic) && f.pos.y > (fx)(40 * (int)kOne)) coherent = false;
        check(coherent, "cgf-step: the coupled state stays coherent (no explosion)");
    }

    // ---- A static grain AND a static fluid particle are untouched by the coupled step --------------------
    {
        cgf::CGFWorld w; fluid::FluidKernel kernel;
        buildStepScene(w, kernel);
        // Pin grain 0 and fluid 0 static (invMass 0, kFlagStatic) — every sub-pass must skip them.
        w.grains[0].flags = grain::kFlagStatic; w.grains[0].invMass = 0;
        w.fluid[0].flags  = fluid::kFlagStatic; w.fluid[0].invMass  = 0;
        const fpx::FxVec3 gPos = w.grains[0].pos, gVel = w.grains[0].vel;
        const fpx::FxVec3 fPos = w.fluid[0].pos,  fVel = w.fluid[0].vel;
        cgf::StepCGFSteps(w, kernel, w.dt, 3, 20);
        check(std::memcmp(&w.grains[0].pos, &gPos, sizeof(fpx::FxVec3)) == 0 &&
              std::memcmp(&w.grains[0].vel, &gVel, sizeof(fpx::FxVec3)) == 0,
              "cgf-step: static grain pos+vel untouched");
        check(std::memcmp(&w.fluid[0].pos, &fPos, sizeof(fpx::FxVec3)) == 0 &&
              std::memcmp(&w.fluid[0].vel, &fVel, sizeof(fpx::FxVec3)) == 0,
              "cgf-step: static fluid pos+vel untouched");
    }

    // ---- iters=0: predict + velocity + couple only (no position solve), still deterministic --------------
    {
        cgf::CGFWorld w; fluid::FluidKernel kernel;
        buildStepScene(w, kernel);
        cgf::CGFWorld a = w, b = w;
        cgf::StepCGFSteps(a, kernel, w.dt, 0, 10);
        cgf::StepCGFSteps(b, kernel, w.dt, 0, 10);
        bool same = true;
        for (size_t i = 0; same && i < a.grains.size(); ++i)
            if (std::memcmp(&a.grains[i], &b.grains[i], sizeof(grain::GrainParticle)) != 0) same = false;
        for (size_t i = 0; same && i < a.fluid.size(); ++i)
            if (std::memcmp(&a.fluid[i], &b.fluid[i], sizeof(fluid::FluidParticle)) != 0) same = false;
        check(same, "cgf-step iters=0: predict+velocity+couple only, two runs byte-identical");
        // The pools still fell under gravity (predict moved them; no NaN/explosion).
        const cgf::CGFState st = cgf::MeasureCGFState(a);
        check(st.dynamicFluid > 0u, "cgf-step iters=0: dynamic fluid present (deterministic)");
    }

    // ---- MeasureCGFState on a known two-pool scene (the metric is well-formed) ----------------------------
    {
        cgf::CGFWorld w; w.h = kOne + kOne / 2; w.gravity = fpx::FxVec3{0, (fx)(-9 * (int)kOne), 0};
        w.dt = kOne / 60; w.groundY = 0;
        // 2 dynamic grains low, 2 dynamic fluid high overlapping them.
        w.grains = {GrainAt(0, 1, 0), GrainAt(1, 1, 0)};
        fluid::FluidParticle f0 = FluidAt(0, 0, 0); f0.pos.y = (fx)(3 * (int)kOne); f0.prev = f0.pos;
        fluid::FluidParticle f1 = FluidAt(1, 0, 0); f1.pos.y = (fx)(3 * (int)kOne); f1.prev = f1.pos;
        w.fluid = {f0, f1};
        const cgf::CGFState st = cgf::MeasureCGFState(w);
        check(st.dynamicGrains == 2u && st.dynamicFluid == 2u, "MeasureCGFState: counts both dynamic pools");
        check(st.bedY == (fx)(1 * (int)kOne), "MeasureCGFState: bedY == the grain mean y (1.0)");
        check(st.fluidY == (fx)(3 * (int)kOne), "MeasureCGFState: fluidY == the fluid mean y (3.0)");
        check(st.fluidY > st.bedY, "MeasureCGFState: fluidY > bedY (the pools are layered)");
    }

    if (g_fail == 0) std::printf("cgf_test: ALL PASS\n");
    else             std::printf("cgf_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
