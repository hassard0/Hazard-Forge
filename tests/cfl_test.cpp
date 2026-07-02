// Slice CF1 — Deterministic Cloth<->Fluid Coupling: the WET-CLOTH CORE (engine/sim/couple_cf.h), Track-S S1
// of docs/SUPERIORITY_ROADMAP.md — the FOURTH material-interaction pairing (rigid<->fluid, rigid<->grain,
// grain<->fluid shipped). Pure CPU (header-only, no device, no backend symbols). Namespace hf::sim::cfl.
//
// What this test PINS (the CF1 contracts):
//   * the CROSS-QUERY (the GF1 twin over cloth verts + fluid): the shared union grid, the exact CSR cross
//     lists vs a naive O(C*F) oracle, the symmetry CountCF==CountFC, separated pools -> 0, empty no-ops,
//     two builds byte-identical.
//   * the CONTACT (the CL7 projection mold cross-pool): a hand-laid pair separates to rSum within LSBs,
//     the inverse-mass split (pinned vert -> fluid takes the full push; static fluid -> the vert does),
//     rSum <= 0 -> EXACT no-op.
//   * the TWO-WAY DRAG (the FL7 XSPH discipline cross-pool): a hand-laid pair -> the EXACT Q16.16
//     smoothed velocities both sides (cloth gains the fluid's direction, fluid slows), the prev
//     re-encode, kDrag == 0 -> EXACT no-op.
//   * (a) POOL-INVARIANCE / identity-at-zero: StepClothFluid(kDrag=0, radii=0) over K steps ==
//     StepFluid + StepClothSelf called independently, BIT-IDENTICAL both pools.
//   * (b) THE PHYSICS: a fluid block falling onto a pinned-border horizontal cloth sheet — WITHOUT
//     coupling the fluid passes through (below-count > 0) and the cloth sags only under its own weight;
//     WITH coupling the fluid is caught (below-count strictly lower; porosity may leak a few — reported
//     honestly) AND the cloth sags FURTHER under the fluid load (centerY drops vs the dry run — the
//     two-way proof).
//   * (c) MOMENTUM: one drag pass over an all-dynamic unit-mass scene -> total two-pool momentum drift
//     pinned within an honest fxmul-truncation bound (the FL7 <=256-LSB-scale precedent).
//   * (d) DETERMINISM: two coupled runs byte-identical; the ClothDigest + FluidDigest printed (pinned
//     identical MSVC + clang at verify time).
//   * (e) LOCKSTEP + ROLLBACK (the GF5 twin over TWO pools): authority==replica from inputs alone,
//     rollback corrects a real misprediction to authority bit-for-bit, snapshot round-trip.
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/couple_cf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace cfl   = hf::sim::cfl;
namespace cloth = hf::sim::cloth;
namespace fluid = hf::sim::fluid;
namespace fpx   = hf::sim::fpx;
using cfl::fx;
using cfl::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << fpx::kFrac); }

// A dynamic cloth vert at an integer world position (unit mass, unpinned).
static cloth::ClothParticle VertAt(int x, int y, int z) {
    cloth::ClothParticle p;
    p.pos  = fpx::FxVec3{FromInt(x), FromInt(y), FromInt(z)};
    p.prev = p.pos;
    p.vel  = fpx::FxVec3{0, 0, 0};
    p.invMass = kOne;
    p.flags   = 0;
    return p;
}

// A dynamic fluid particle at an integer world position (unit mass).
static fluid::FluidParticle FluidAt(int x, int y, int z) {
    fluid::FluidParticle f;
    f.pos  = fpx::FxVec3{FromInt(x), FromInt(y), FromInt(z)};
    f.prev = f.pos;
    f.vel  = fpx::FxVec3{0, 0, 0};
    f.invMass = kOne;
    f.flags   = 0;
    return f;
}

// The naive O(C*F) cross-pair oracle (every cloth x fluid pair passing the per-axis |dx|<h box reject).
static uint32_t NaiveCrossPairs(const cfl::CFLWorld& w) {
    uint32_t pairs = 0;
    for (const cloth::ClothParticle& c : w.cloth)
        for (const fluid::FluidParticle& f : w.fluid) {
            fx ax = c.pos.x - f.pos.x; if (ax < 0) ax = -ax;
            fx ay = c.pos.y - f.pos.y; if (ay < 0) ay = -ay;
            fx az = c.pos.z - f.pos.z; if (az < 0) az = -az;
            if (ax < w.h && ay < w.h && az < w.h) ++pairs;
        }
    return pairs;
}

// Both pools byte-equal (the two-pool memcmp).
static bool CoupledEqual(const cfl::CFLWorld& a, const cfl::CFLWorld& b) {
    return a.cloth.size() == b.cloth.size() && a.fluid.size() == b.fluid.size() &&
           std::memcmp(a.cloth.data(), b.cloth.data(),
                       a.cloth.size() * sizeof(cloth::ClothParticle)) == 0 &&
           std::memcmp(a.fluid.data(), b.fluid.data(),
                       a.fluid.size() * sizeof(fluid::FluidParticle)) == 0;
}

// ----- The shared fluid-on-pinned-cloth scene (the CF1 showcase config, reused across the proofs) ----------
// A 9x9 horizontal cloth HAMMOCK in the XZ plane at y=2 (spacing 0.5, the FOUR CORNERS pinned -> the sheet
// drapes into a pocket that both CATCHES and CONTAINS the pool), a 4x4x4 fluid block (64 particles) seeded
// just above its centre, gravity -9.8, ground y=0. clothRadius 0.35 + fluidRadius 0.25 -> rSum 0.6 > the
// 0.354 worst lattice-gap distance (the documented porosity geometry: the barrier covers the mesh holes at
// rest; a STRETCHED region can still open gaps — reported honestly). kernel h == coupling h == 1.0 (the
// FL4 h = 2x spacing convention; the first scene draft at h = 3x spacing burst apart — a PBF block's
// startup over-density scales with the neighbour count). rho0 = the probe MAX density (NOT the mean): the
// seeded block then starts with C_i <= 0 EVERYWHERE (the unilateral clamp -> ZERO startup repulsion), so
// the pool compacts gently instead of bursting off the sheet — the make-or-break scene lesson.
struct Scene {
    cfl::CFLWorld world;
    cfl::CFLScene scene;
    fx yCloth = 0;
};
static Scene BuildScene() {
    Scene s;
    const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));   // -9.8 host-snapped
    const fx kDt = kOne / 60;
    const fx kH = kOne;                               // 1.0 coupling radius == the kernel h (2x spacing)
    const fx kHalf = kOne / 2;                        // 0.5 spacing
    s.yCloth = FromInt(2);

    // The horizontal corner-pinned cloth hammock (9x9, XZ plane at y=2; cloth.h InitGrid is the vertical
    // curtain, so the horizontal sheet is hand-laid — the lattice TOPOLOGY (grid.W/H) is what CL2 uses).
    cloth::ClothGrid grid;
    grid.W = 9; grid.H = 9;
    grid.spacing = kHalf;
    grid.origin = fpx::FxVec3{0, s.yCloth, 0};
    std::vector<cloth::ClothParticle> sheet((size_t)(grid.W * grid.H));
    for (int r = 0; r < grid.H; ++r)
        for (int c = 0; c < grid.W; ++c) {
            cloth::ClothParticle p;
            p.pos = fpx::FxVec3{(fx)(c * (int)kHalf), s.yCloth, (fx)(r * (int)kHalf)};
            p.prev = p.pos;
            p.vel = fpx::FxVec3{0, 0, 0};
            const bool pinned = (r == 0 || r == grid.H - 1) && (c == 0 || c == grid.W - 1);
            p.invMass = pinned ? 0 : kOne;
            p.flags   = pinned ? cloth::kFlagPinned : 0u;
            sheet[(size_t)cloth::ParticleIndex(grid, r, c)] = p;
        }

    // The fluid block above the sheet centre: 4x4x4 at 0.5 spacing, x/z in [1, 2.5], y from 2.75.
    fluid::FluidBlock fblock;
    fblock.W = 4; fblock.H = 4; fblock.D = 4;
    fblock.spacing = kHalf;
    fblock.origin = fpx::FxVec3{FromInt(1), FromInt(2) + kOne * 3 / 4, FromInt(1)};
    std::vector<fluid::FluidParticle> fluidP = fluid::InitBlock(fblock);

    s.world.cloth = sheet;
    s.world.fluid = fluidP;
    s.world.gravity = fpx::FxVec3{0, kGravY, 0};
    s.world.dt = kDt;
    s.world.groundY = 0;
    s.world.h = kH;
    s.world.clothRadius = (fx)(kOne * 7 / 20);        // 0.35
    s.world.fluidRadius = kOne / 4;                   // 0.25 -> rSum 0.6 > the 0.354 lattice gap

    s.scene.grid = grid;
    s.scene.constraints = cloth::BuildConstraints(grid, sheet);
    s.scene.excl = cloth::BuildClothAdjacency(sheet.size(), s.scene.constraints);
    s.scene.iters = 6;                                // a stiffer sheet (less stretch -> smaller barrier gaps)
    s.scene.selfThickness = 0;                        // CL7 off (a taut sheet — the composed entry stays general)
    s.scene.selfIters = 0;
    s.scene.contactIters = 4;                         // a firmer cross-pool barrier (K Jacobi projections/step)
    s.scene.visc = kOne / 8;                          // FL7 XSPH (the FL7 showcase coefficient — damps the pile)
    // The kernel: ρ0 = the MAX density of the packed initial fluid lattice (C_i <= 0 at seed -> no
    // startup repulsive burst; the pool compacts gently — see the Scene banner).
    s.scene.kernel = fluid::BuildKernelTable(kH, kOne, fluid::kKernelBins, kOne / 100);
    {
        const fluid::FluidGrid pg = fluid::MakeGrid(s.world.fluid, kH);
        const fluid::FluidCellTable pt = fluid::BuildCellTable(s.world.fluid, pg);
        const fluid::FluidNeighborList pl = fluid::BuildNeighborList(s.world.fluid, pg, pt, kH);
        std::vector<fx> probeRho;
        fluid::ComputeDensity(s.world.fluid, pl, s.scene.kernel, probeRho);
        fx rho0 = 0;
        for (fx d : probeRho) if (d > rho0) rho0 = d;
        s.scene.kernel = fluid::BuildKernelTable(kH, rho0, fluid::kKernelBins, kOne / 100);
    }
    return s;
}

// The mean pos.y over the DYNAMIC cloth verts (a spatially-averaged sag stat, steadier than the single
// centre vert against the deterministic ringing of the pinned hammock).
static fx MeanDynamicClothY(const std::vector<cloth::ClothParticle>& verts) {
    int64_t sum = 0; int n = 0;
    for (const cloth::ClothParticle& p : verts)
        if (!(p.flags & cloth::kFlagPinned)) { sum += (int64_t)p.pos.y; ++n; }
    return n == 0 ? 0 : (fx)(sum / (int64_t)n);
}

int main() {
    HF_TEST_MAIN_INIT();

    const fx h = kOne + kOne / 2;   // 1.5 coupling radius (the scene config)
    const fx kDrag = kOne / 16;     // the drag coefficient (small: c*ΣW stays a contraction)
    const int kSteps = 120;         // 2 seconds at dt = 1/60

    // ===== The CROSS-QUERY (the GF1 twin over cloth verts + fluid) =========================================

    // ---- a hand-laid overlapping set -> the exact cross lists -------------------------------------------
    {
        cfl::CFLWorld w; w.h = h;
        w.cloth = {VertAt(0, 0, 0)};
        fluid::FluidParticle near0 = FluidAt(0, 0, 0);                 // |dx| = 0 < h
        fluid::FluidParticle near1 = FluidAt(0, 0, 0); near1.pos.x = kOne;   // 1.0 < 1.5
        fluid::FluidParticle far0  = FluidAt(9, 0, 0);                 // 9 away -> no pair
        w.fluid = {near0, near1, far0};
        const cfl::CFLNeighbors nbr = cfl::BuildCFLNeighbors(w);
        check(nbr.cfStart.size() == 2u && nbr.cfStart[1] == 2u, "cross: vert gathers exactly 2 fluid");
        check(nbr.cfNeighbors.size() == 2u && nbr.cfNeighbors[0] == 0u && nbr.cfNeighbors[1] == 1u,
              "cross: cfNeighbors ascending {0,1}, far fluid excluded");
        check(nbr.fcStart.size() == 4u && nbr.fcStart[1] - nbr.fcStart[0] == 1u &&
              nbr.fcStart[2] - nbr.fcStart[1] == 1u && nbr.fcStart[3] - nbr.fcStart[2] == 0u,
              "cross: fluid 0/1 gather the vert, far fluid gathers none");
        check(cfl::CountCF(nbr) == cfl::CountFC(nbr), "cross: symmetry CountCF == CountFC");
        check(cfl::CountCF(nbr) == NaiveCrossPairs(w), "cross: CSR total == the naive O(C*F) oracle");
    }

    // ---- two interpenetrating blocks: CSR vs the naive oracle + symmetry --------------------------------
    {
        cfl::CFLWorld w; w.h = h;
        for (int cx = 0; cx < 3; ++cx)
            for (int cz = 0; cz < 3; ++cz) w.cloth.push_back(VertAt(cx, 0, cz));
        for (int fx_ = 0; fx_ < 3; ++fx_)
            for (int fz = 0; fz < 3; ++fz) {
                fluid::FluidParticle f = FluidAt(fx_, 0, fz);
                f.pos.x += kOne / 2; f.pos.y += kOne / 2;   // shifted, overlapping
                w.fluid.push_back(f);
            }
        const cfl::CFLNeighbors nbr = cfl::BuildCFLNeighbors(w);
        const uint32_t naive = NaiveCrossPairs(w);
        check(naive > 0u, "cross-block: the pools actually overlap");
        check(cfl::CountCF(nbr) == naive && cfl::CountFC(nbr) == naive,
              "cross-block: both CSR totals == the naive oracle");
        check(nbr.cfStart.back() == nbr.cfNeighbors.size() && nbr.fcStart.back() == nbr.fcNeighbors.size(),
              "cross-block: CSR sentinels == list sizes");
        // Every emitted cf pair passes the box reject (no spurious entries).
        bool valid = true;
        for (size_t i = 0; i + 1 < nbr.cfStart.size(); ++i)
            for (uint32_t s = nbr.cfStart[i]; s < nbr.cfStart[i + 1]; ++s) {
                const auto& c = w.cloth[i]; const auto& f = w.fluid[nbr.cfNeighbors[s]];
                fx ax = c.pos.x - f.pos.x; if (ax < 0) ax = -ax;
                fx ay = c.pos.y - f.pos.y; if (ay < 0) ay = -ay;
                fx az = c.pos.z - f.pos.z; if (az < 0) az = -az;
                if (!(ax < h && ay < h && az < h)) valid = false;
            }
        check(valid, "cross-block: every emitted cf pair passes the box reject");
        // Determinism: two builds byte-identical.
        const cfl::CFLNeighbors nbr2 = cfl::BuildCFLNeighbors(w);
        check(nbr.cfStart == nbr2.cfStart && nbr.cfNeighbors == nbr2.cfNeighbors &&
              nbr.fcStart == nbr2.fcStart && nbr.fcNeighbors == nbr2.fcNeighbors,
              "cross-block: two builds byte-identical");
    }

    // ---- separated / empty pools -> 0 cross pairs (the no-ops) -------------------------------------------
    {
        cfl::CFLWorld w; w.h = h;
        w.cloth = {VertAt(0, 0, 0), VertAt(1, 0, 0)};
        w.fluid = {FluidAt(20, 0, 0)};
        const cfl::CFLNeighbors nbr = cfl::BuildCFLNeighbors(w);
        check(cfl::CountCF(nbr) == 0u && cfl::CountFC(nbr) == 0u, "cross: separated pools -> 0 pairs");

        cfl::CFLWorld empty; empty.h = h;
        const cfl::CFLNeighbors n0 = cfl::BuildCFLNeighbors(empty);
        check(cfl::CountCF(n0) == 0u && cfl::CountFC(n0) == 0u &&
              n0.cfStart.size() == 1u && n0.fcStart.size() == 1u,
              "cross: empty world -> sentinel-only CSR");

        cfl::CFLWorld clothOnly; clothOnly.h = h; clothOnly.cloth = w.cloth;
        const cfl::CFLNeighbors n1 = cfl::BuildCFLNeighbors(clothOnly);
        check(n1.cfStart.size() == 3u && n1.cfStart.back() == 0u && n1.fcStart.size() == 1u,
              "cross: cloth-only -> well-formed empty lists");
    }

    // ===== The CONTACT (the CL7 projection mold, cross-pool) ================================================

    // ---- a hand-laid dynamic pair separates to rSum within LSBs ------------------------------------------
    {
        cfl::CFLWorld w; w.h = h; w.dt = kOne; w.groundY = FromInt(-10);
        w.clothRadius = (fx)(kOne * 3 / 10);          // 0.3
        w.fluidRadius = kOne / 5;                     // 0.2 -> rSum 0.5
        w.cloth = {VertAt(0, 0, 0)};
        fluid::FluidParticle f = FluidAt(0, 0, 0); f.pos.x = kOne / 5; f.prev = f.pos;  // 0.2 apart on x
        w.fluid = {f};
        const cfl::CFLNeighbors nbr = cfl::BuildCFLNeighbors(w);
        const int proj = cfl::SolveClothFluidContact(w, nbr);
        check(proj == 1, "contact: one projection gathered");
        const fx dist = fpx::FxLength(fpx::FxSub(w.cloth[0].pos, w.fluid[0].pos));
        fx err = dist - (w.clothRadius + w.fluidRadius); if (err < 0) err = -err;
        check(err <= 8, "contact: pair separated to rSum within LSBs (both sides took half)");
        check(w.cloth[0].pos.x < 0 && w.fluid[0].pos.x > kOne / 5,
              "contact: vert pushed -x, fluid pushed +x (opposite)");
    }

    // ---- pinned vert -> untouched; the fluid takes the FULL push -----------------------------------------
    {
        cfl::CFLWorld w; w.h = h; w.dt = kOne; w.groundY = FromInt(-10);
        w.clothRadius = (fx)(kOne * 3 / 10); w.fluidRadius = kOne / 5;
        cloth::ClothParticle v = VertAt(0, 0, 0); v.invMass = 0; v.flags = cloth::kFlagPinned;
        w.cloth = {v};
        fluid::FluidParticle f = FluidAt(0, 0, 0); f.pos.x = kOne / 5; f.prev = f.pos;
        w.fluid = {f};
        const cfl::CFLNeighbors nbr = cfl::BuildCFLNeighbors(w);
        cfl::SolveClothFluidContact(w, nbr);
        check(w.cloth[0].pos.x == 0 && w.cloth[0].pos.y == 0, "contact: pinned vert untouched");
        const fx dist = fpx::FxLength(fpx::FxSub(w.cloth[0].pos, w.fluid[0].pos));
        fx err = dist - (w.clothRadius + w.fluidRadius); if (err < 0) err = -err;
        check(err <= 8, "contact: fluid took the FULL push off the pinned vert (dist == rSum)");
    }

    // ---- static fluid -> untouched; the vert takes the full push; rSum=0 -> exact no-op ------------------
    {
        cfl::CFLWorld w; w.h = h; w.dt = kOne; w.groundY = FromInt(-10);
        w.clothRadius = (fx)(kOne * 3 / 10); w.fluidRadius = kOne / 5;
        w.cloth = {VertAt(0, 0, 0)};
        fluid::FluidParticle f = FluidAt(0, 0, 0); f.pos.x = kOne / 5; f.prev = f.pos;
        f.flags = fluid::kFlagStatic; f.invMass = 0;
        w.fluid = {f};
        const cfl::CFLNeighbors nbr = cfl::BuildCFLNeighbors(w);
        cfl::SolveClothFluidContact(w, nbr);
        check(w.fluid[0].pos.x == kOne / 5, "contact: static fluid untouched");
        const fx dist = fpx::FxLength(fpx::FxSub(w.cloth[0].pos, w.fluid[0].pos));
        fx err = dist - (w.clothRadius + w.fluidRadius); if (err < 0) err = -err;
        check(err <= 8, "contact: the vert took the FULL push off the static fluid");

        // rSum = 0 -> exact no-op (byte-identical pools).
        cfl::CFLWorld z = w; z.clothRadius = 0; z.fluidRadius = 0;
        const cfl::CFLWorld before = z;
        const cfl::CFLNeighbors nz = cfl::BuildCFLNeighbors(z);
        check(cfl::SolveClothFluidContact(z, nz) == 0 && CoupledEqual(z, before),
              "contact: rSum <= 0 -> EXACT no-op");
    }

    // ===== The TWO-WAY DRAG (the FL7 XSPH discipline, cross-pool) ==========================================

    // ---- a hand-laid coincident pair -> the EXACT Q16.16 smoothed velocities both sides -------------------
    {
        const fx kD = kOne / 4;
        cfl::CFLWorld w; w.h = h; w.dt = kOne; w.groundY = FromInt(-10);
        fluid::FluidKernel kern = fluid::BuildKernelTable(h, kOne, fluid::kKernelBins, kOne / 100);
        // Cloth vert at rest at the origin (pos == prev -> v = 0). Fluid AT the vert moving +2 in x
        // (prev = pos - vel*dt with dt = 1.0 -> the derived v is exactly 2.0).
        w.cloth = {VertAt(0, 0, 0)};
        fluid::FluidParticle f = FluidAt(0, 0, 0);
        f.vel = fpx::FxVec3{FromInt(2), 0, 0};
        f.prev = fpx::FxVec3{f.pos.x - FromInt(2), f.pos.y, f.pos.z};
        w.fluid = {f};
        const cfl::CFLNeighbors nbr = cfl::BuildCFLNeighbors(w);
        cfl::ApplyClothFluidDrag(w, nbr, kern, kD);
        // Expected (r = 0 -> bin 0 -> W[0]): cloth v' = 0 + kD*fxmul(2.0, W0); fluid v' = 2.0 + kD*fxmul(-2.0, W0).
        const fx w0 = kern.W[0];
        const fx expC = fpx::fxmul(kOne, fpx::fxmul(kD, fpx::fxmul(FromInt(2), w0)));
        const fx expF = FromInt(2) + fpx::fxmul(kOne, fpx::fxmul(kD, fpx::fxmul(FromInt(-2), w0)));
        check(w.cloth[0].vel.x == expC, "drag: EXACT cloth smoothed velocity (gains the fluid direction)");
        check(w.fluid[0].vel.x == expF, "drag: EXACT fluid smoothed velocity (slowed by the cloth)");
        check(w.cloth[0].vel.x > 0 && w.fluid[0].vel.x < FromInt(2),
              "drag: momentum flowed fluid -> cloth (two-way exchange)");
        // The prev re-encode: prev = pos - v'*dt on both pools.
        check(w.cloth[0].prev.x == w.cloth[0].pos.x - fpx::fxmul(w.cloth[0].vel.x, kOne),
              "drag: cloth prev re-encoded (prev = pos - v'*dt)");
        check(w.fluid[0].prev.x == w.fluid[0].pos.x - fpx::fxmul(w.fluid[0].vel.x, kOne),
              "drag: fluid prev re-encoded (prev = pos - v'*dt)");
    }

    // ---- kDrag = 0 -> EXACT no-op; pinned vert never re-encoded -------------------------------------------
    {
        cfl::CFLWorld w; w.h = h; w.dt = kOne; w.groundY = FromInt(-10);
        fluid::FluidKernel kern = fluid::BuildKernelTable(h, kOne, fluid::kKernelBins, kOne / 100);
        cloth::ClothParticle v = VertAt(0, 0, 0); v.invMass = 0; v.flags = cloth::kFlagPinned;
        w.cloth = {v};
        fluid::FluidParticle f = FluidAt(0, 0, 0);
        f.vel = fpx::FxVec3{FromInt(2), 0, 0};
        f.prev = fpx::FxVec3{f.pos.x - FromInt(2), f.pos.y, f.pos.z};
        w.fluid = {f};
        const cfl::CFLNeighbors nbr = cfl::BuildCFLNeighbors(w);
        const cfl::CFLWorld before = w;
        cfl::ApplyClothFluidDrag(w, nbr, kern, 0);
        check(CoupledEqual(w, before), "drag: kDrag == 0 -> EXACT no-op (identity-at-zero)");
        cfl::ApplyClothFluidDrag(w, nbr, kern, kOne / 4);
        check(std::memcmp(&w.cloth[0], &before.cloth[0], sizeof(cloth::ClothParticle)) == 0,
              "drag: pinned vert never re-encoded (holds bit-for-bit)");
        check(w.fluid[0].vel.x < FromInt(2), "drag: the fluid still slows against the pinned vert");
    }

    // ===== (c) MOMENTUM: the drag drift bound over an all-dynamic unit-mass scene ==========================
    {
        cfl::CFLWorld w; w.h = h; w.dt = kOne / 4; w.groundY = FromInt(-10);
        fluid::FluidKernel kern = fluid::BuildKernelTable(h, kOne, fluid::kKernelBins, kOne / 100);
        // A 3x3 all-dynamic cloth patch (XZ at y=0) + a 2x2x2 fluid block overlapping it, falling -2 in y
        // (prev encoded so the derived v is exactly -2.0: prev = pos - fxmul(vel, dt), dt = 0.25).
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                cloth::ClothParticle p = VertAt(0, 0, 0);
                p.pos = fpx::FxVec3{(fx)(c * (int)(kOne / 2)), 0, (fx)(r * (int)(kOne / 2))};
                p.prev = p.pos;
                w.cloth.push_back(p);
            }
        for (int fx_ = 0; fx_ < 2; ++fx_)
            for (int fy = 0; fy < 2; ++fy)
                for (int fz = 0; fz < 2; ++fz) {
                    fluid::FluidParticle f = FluidAt(0, 0, 0);
                    f.pos = fpx::FxVec3{(fx)(fx_ * (int)(kOne / 2)), kOne / 4 + (fx)(fy * (int)(kOne / 2)),
                                        (fx)(fz * (int)(kOne / 2))};
                    f.vel = fpx::FxVec3{0, FromInt(-2), 0};
                    f.prev = fpx::FxVec3{f.pos.x - fpx::fxmul(f.vel.x, w.dt),
                                         f.pos.y - fpx::fxmul(f.vel.y, w.dt),
                                         f.pos.z - fpx::fxmul(f.vel.z, w.dt)};
                    w.fluid.push_back(f);
                }
        const cfl::CFLNeighbors nbr = cfl::BuildCFLNeighbors(w);
        check(cfl::CountCF(nbr) > 0u, "momentum: the pools actually couple (pairs > 0)");
        const cfl::CFLMomentum before = cfl::TotalDynamicMomentum(w);
        cfl::ApplyClothFluidDrag(w, nbr, kern, kOne / 4);
        const cfl::CFLMomentum after = cfl::TotalDynamicMomentum(w);
        const int64_t dx = after.x - before.x, dy = after.y - before.y, dz = after.z - before.z;
        const int64_t adx = dx < 0 ? -dx : dx, ady = dy < 0 ? -dy : dy, adz = dz < 0 ? -dz : dz;
        std::printf("momentum drift after one drag pass: {%lld, %lld, %lld} LSB (pairs: %u)\n",
                    (long long)dx, (long long)dy, (long long)dz, cfl::CountCF(nbr));
        check(adx <= 256 && ady <= 256 && adz <= 256,
              "momentum: total two-pool drift <= 256 LSB per axis (the FL7-scale honest bound)");
        // The exchange is real: the cloth gained downward momentum, the fluid lost some.
        int64_t clothY = 0;
        for (const cloth::ClothParticle& p : w.cloth) clothY += (int64_t)p.vel.y;
        check(clothY < 0, "momentum: the cloth gained the fluid's downward momentum (dragged)");
    }

    // ===== (a) POOL-INVARIANCE / identity-at-zero over the full scene ======================================
    {
        Scene s = BuildScene();
        // Coupled entry with the coupling OFF (kDrag = 0, radii = 0).
        cfl::CFLWorld off = s.world;
        off.clothRadius = 0; off.fluidRadius = 0;
        cfl::StepClothFluidSteps(off, s.scene, 0, 40);
        // The pools' own steps, called independently (the uncoupled reference).
        std::vector<fluid::FluidParticle> fRef = s.world.fluid;
        std::vector<cloth::ClothParticle> cRef = s.world.cloth;
        for (int k = 0; k < 40; ++k) {
            fluid::StepFluid(fRef, s.scene.kernel, s.scene.fluidSpheres, s.world.gravity, s.world.dt,
                             s.world.groundY, s.scene.iters);
            cloth::StepClothSelf(s.scene.grid, cRef, s.scene.constraints, s.scene.excl,
                                 s.scene.clothSpheres, s.world.gravity, s.world.dt, s.world.groundY,
                                 s.scene.iters, s.scene.selfThickness, s.scene.selfIters);
        }
        check(off.fluid.size() == fRef.size() &&
              std::memcmp(off.fluid.data(), fRef.data(), fRef.size() * sizeof(fluid::FluidParticle)) == 0,
              "identity-at-zero: the FLUID pool == its own StepFluid BIT-IDENTICAL");
        check(off.cloth.size() == cRef.size() &&
              std::memcmp(off.cloth.data(), cRef.data(), cRef.size() * sizeof(cloth::ClothParticle)) == 0,
              "identity-at-zero: the CLOTH pool == its own StepClothSelf BIT-IDENTICAL");
    }

    // ===== (b) THE PHYSICS: the fluid stream onto the pinned sheet — caught + sagged =======================
    {
        const fx kBelow = FromInt(1);   // the "passed through" plane (well under the y=2 sheet)

        // WITHOUT coupling: the fluid falls straight through the cloth plane (the barrier contrast).
        Scene dry = BuildScene();
        cfl::CFLWorld un = dry.world;
        un.clothRadius = 0; un.fluidRadius = 0;
        cfl::StepClothFluidSteps(un, dry.scene, 0, kSteps);
        const uint32_t belowWithout = cfl::CountFluidBelow(un.fluid, kBelow);
        check(belowWithout > 0u, "physics: WITHOUT coupling the fluid passes through the cloth plane");

        // The FAIR dry-sag baseline: the SAME coupled entry (same kDrag/radii -> same velocity re-encode
        // discipline) with an EMPTY fluid pool — the sag delta below isolates the fluid load (the
        // documented fair-baseline lesson in couple_cf.h).
        Scene noF = BuildScene();
        cfl::CFLWorld dryRun = noF.world;
        dryRun.fluid.clear();
        cfl::StepClothFluidSteps(dryRun, noF.scene, kDrag, kSteps);
        const fx dryCenterY = cfl::ClothCenterY(noF.scene.grid, dryRun.cloth);
        const fx dryMeanY   = MeanDynamicClothY(dryRun.cloth);

        // WITH coupling: the sheet catches the stream and sags under it.
        Scene wet = BuildScene();
        cfl::CFLWorld co = wet.world;
        cfl::StepClothFluidSteps(co, wet.scene, kDrag, kSteps);
        const uint32_t belowWith = cfl::CountFluidBelow(co.fluid, kBelow);
        const fx wetCenterY = cfl::ClothCenterY(wet.scene.grid, co.cloth);
        const fx wetMeanY   = MeanDynamicClothY(co.cloth);
        std::printf("physics: below-plane count without=%u with=%u (of %u fluid); "
                    "centerY dry=%d wet=%d (delta %d LSB); meanY dry=%d wet=%d (delta %d LSB)\n",
                    belowWithout, belowWith, (uint32_t)co.fluid.size(),
                    dryCenterY, wetCenterY, dryCenterY - wetCenterY,
                    dryMeanY, wetMeanY, dryMeanY - wetMeanY);
        check(belowWith < belowWithout,
              "physics: WITH coupling the caught count is strictly higher (below-count strictly lower)");
        check(wetMeanY < dryMeanY - kOne / 32,
              "physics: the cloth SAGS under the fluid load (mean dynamic-vert drop > 1/32 unit — two-way)");
        // Coherence: nothing rockets out (a bounded deterministic settle).
        bool coherent = true;
        for (const fluid::FluidParticle& f : co.fluid)
            if (f.pos.y > FromInt(40) || f.pos.y < 0) coherent = false;
        for (const cloth::ClothParticle& p : co.cloth)
            if (p.pos.y > FromInt(40) || p.pos.y < 0) coherent = false;
        check(coherent, "physics: the coupled state stays coherent (no explosion)");
    }

    // ===== (d) DETERMINISM: two coupled runs byte-identical + the digests ==================================
    {
        Scene sA = BuildScene(), sB = BuildScene();
        cfl::CFLWorld a = sA.world, b = sB.world;
        cfl::StepClothFluidSteps(a, sA.scene, kDrag, kSteps);
        cfl::StepClothFluidSteps(b, sB.scene, kDrag, kSteps);
        check(CoupledEqual(a, b), "determinism: two coupled runs byte-identical (BOTH pools)");
        std::printf("digests after %d coupled steps: cloth=0x%016llx fluid=0x%016llx\n", kSteps,
                    (unsigned long long)cloth::ClothDigest(a.cloth),
                    (unsigned long long)fluid::FluidDigest(a.fluid));
    }

    // ===== (e) LOCKSTEP + ROLLBACK over the coupled two-pool world (the GF5 twin) ==========================
    {
        Scene s = BuildScene();
        const int kTicks = 8, kMispredictTick = 3;
        const uint32_t cIdx = (uint32_t)cloth::ParticleIndex(s.scene.grid, s.scene.grid.H / 2,
                                                             s.scene.grid.W / 2);   // dynamic centre vert
        const uint32_t fIdx = (uint32_t)(s.world.fluid.size() / 2);

        // ---- ApplyCFLCommand: wind adds to a vert; jet adds to a fluid; pinned/static/OOR no-ops ----------
        {
            cfl::CFLWorld a = s.world;
            cfl::ApplyCFLCommand(a, cfl::CFLCommand{0, cfl::kCmdClothWind, cIdx, fpx::FxVec3{kOne, 0, 0}});
            cfl::ApplyCFLCommand(a, cfl::CFLCommand{0, cfl::kCmdFluidJet, fIdx, fpx::FxVec3{0, kOne, 0}});
            check(a.cloth[cIdx].vel.x == kOne, "ApplyCFLCommand: wind adds to the vert velocity");
            check(a.fluid[fIdx].vel.y == kOne, "ApplyCFLCommand: jet adds to the fluid velocity");
            cfl::CFLWorld b = s.world;
            cfl::ApplyCFLCommand(b, cfl::CFLCommand{0, cfl::kCmdClothWind, 999999u, fpx::FxVec3{kOne, 0, 0}});
            cfl::ApplyCFLCommand(b, cfl::CFLCommand{0, 42u, cIdx, fpx::FxVec3{kOne, 0, 0}});
            cfl::ApplyCFLCommand(b, cfl::CFLCommand{0, cfl::kCmdClothWind, 0u, fpx::FxVec3{kOne, 0, 0}});  // pinned corner
            check(CoupledEqual(b, s.world), "ApplyCFLCommand: OOR / unknown kind / pinned -> no-op");
        }

        // ---- Snapshot round-trip (both pools) --------------------------------------------------------------
        {
            cfl::CFLWorld a = s.world;
            const cfl::CFLSnapshot snap = cfl::SnapshotCFL(a);
            cfl::SimCFLTick(a, s.scene, kDrag, {}, 0);   // mutate both pools
            cfl::RestoreCFL(a, snap);
            check(a.cloth.size() == snap.cloth.size() && a.fluid.size() == snap.fluid.size() &&
                  std::memcmp(a.cloth.data(), snap.cloth.data(),
                              snap.cloth.size() * sizeof(cloth::ClothParticle)) == 0 &&
                  std::memcmp(a.fluid.data(), snap.fluid.data(),
                              snap.fluid.size() * sizeof(fluid::FluidParticle)) == 0,
                  "SnapshotCFL/RestoreCFL: bit-exact round-trip on BOTH pools");
        }

        const std::vector<cfl::CFLCommand> authStream = {
            cfl::CFLCommand{1, cfl::kCmdClothWind, cIdx, fpx::FxVec3{FromInt(2), 0, 0}},
            cfl::CFLCommand{2, cfl::kCmdFluidJet,  fIdx, fpx::FxVec3{0, 0, FromInt(2)}},
            cfl::CFLCommand{4, cfl::kCmdClothWind, cIdx, fpx::FxVec3{0, kOne, 0}},
        };
        std::vector<cfl::CFLCommand> mispredictStream = authStream;
        mispredictStream.push_back(cfl::CFLCommand{(uint32_t)kMispredictTick, cfl::kCmdFluidJet, fIdx,
                                                   fpx::FxVec3{FromInt(20), 0, 0}});

        const cfl::CFLWorld authority = cfl::RunCFLLockstep(s.world, s.scene, kDrag, authStream, kTicks);
        const cfl::CFLWorld replica   = cfl::RunCFLLockstep(s.world, s.scene, kDrag, authStream, kTicks);
        check(CoupledEqual(authority, replica), "lockstep: authority==replica BIT-EXACT (BOTH pools)");

        const cfl::CFLWorld rolledBack = cfl::RunCFLRollback(s.world, s.scene, kDrag, authStream,
                                                             mispredictStream, kTicks, kMispredictTick);
        const cfl::CFLWorld mispredicted =
            cfl::RunCFLLockstep(s.world, s.scene, kDrag, mispredictStream, kTicks);
        check(CoupledEqual(rolledBack, authority), "rollback: corrected==authority BIT-EXACT (BOTH pools)");
        check(!CoupledEqual(mispredicted, authority),
              "rollback: the mispredicted state DIFFERED (a real divergence was fixed)");
    }

    if (g_fail == 0) std::printf("cfl_test: ALL PASS\n");
    else             std::printf("cfl_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
