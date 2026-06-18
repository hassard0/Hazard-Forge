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

    if (g_fail == 0) std::printf("cgf_test: ALL PASS\n");
    else             std::printf("cgf_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
