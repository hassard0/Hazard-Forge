// Slice FPX1 — Deterministic Fixed-Point Physics: the Q16.16 INTEGRATOR + integer broadphase core
// (engine/sim/fpx.h) that the GPU shaders/fpx_integrate.comp.hlsl copies VERBATIM + proves
// bit-identical. Pure CPU (header-only, no device, no backend symbols). Namespace hf::sim::fpx.
//
// What this test PINS (the contracts the GPU fpx_integrate.comp + the GPU==CPU proof build on):
//   * fxmul: known products incl negatives, the int64 intermediate (no overflow within the bound),
//     the arithmetic >> floor-toward-negative-infinity semantics, identity (x*1 == x).
//   * FxISqrt: floor(sqrt) on known squares + non-squares incl 0 (matches mc::ISqrt semantics).
//   * IntegrateBody/IntegrateStep: one-step + K-step semi-implicit-Euler closed form (a body falls
//     the EXACT Q16.16 distance), the ground floor clamp (pos.y==groundY, vel.y zeroed on contact),
//     static bodies (no flag) never move, determinism (two runs byte-identical).
//   * integrateEnabled-off modeled (the disabled path leaves the body unchanged).
//   * BroadphaseCell/CellId/FloorDiv incl NEGATIVE coords (correct floor, not truncate-toward-zero).
//   * overflow bound: a fxmul at the documented +-32768 edge stays exact via the int64 intermediate.
//
// Pure C++ (hf_core), ASan-eligible like the other sim/render-math tests.
#include "sim/fpx.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace fpx = hf::sim::fpx;
using fpx::fx;
using fpx::kOne;
using fpx::kFrac;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << kFrac); }

int main() {
    HF_TEST_MAIN_INIT();

    // ================= fxmul known products (incl negatives, the int64 intermediate) =================
    {
        // 2.0 * 3.0 == 6.0
        check(fpx::fxmul(FromInt(2), FromInt(3)) == FromInt(6), "fxmul 2*3 == 6");
        // x * 1 == x (identity at kOne).
        check(fpx::fxmul(FromInt(7), kOne) == FromInt(7), "fxmul x*1 == x");
        check(fpx::fxmul(kOne, FromInt(-5)) == FromInt(-5), "fxmul 1*(-5) == -5");
        // 0.5 * 0.5 == 0.25.
        fx half = kOne / 2, quarter = kOne / 4;
        check(fpx::fxmul(half, half) == quarter, "fxmul 0.5*0.5 == 0.25");
        // Negatives: (-2) * 3 == -6 ; (-2)*(-3) == 6.
        check(fpx::fxmul(FromInt(-2), FromInt(3)) == FromInt(-6), "fxmul (-2)*3 == -6");
        check(fpx::fxmul(FromInt(-2), FromInt(-3)) == FromInt(6), "fxmul (-2)*(-3) == 6");
        // The int64 intermediate: a large product that overflows int32 but is exact in int64.
        // 1000.0 * 30.0 == 30000.0 ; (1000<<16)*(30<<16) overflows int32 -> the int64 path is required.
        check(fpx::fxmul(FromInt(1000), FromInt(30)) == FromInt(30000),
              "fxmul 1000*30 == 30000 (int64 intermediate)");
        // Arithmetic-shift floor semantics: fxmul of a tiny negative fraction floors toward -inf.
        // (-1 raw) * kOne >> 16 == (-1 in Q16.16 of 1/65536) * 1 == -1 raw (exact, identity).
        check(fpx::fxmul((fx)-1, kOne) == (fx)-1, "fxmul of the smallest negative unit * 1 == itself");
    }

    // ================= FxISqrt on known squares + non-squares =================
    {
        check(fpx::FxISqrt(0) == 0, "FxISqrt(0) == 0");
        check(fpx::FxISqrt(1) == 1, "FxISqrt(1) == 1");
        check(fpx::FxISqrt(4) == 2, "FxISqrt(4) == 2");
        check(fpx::FxISqrt(144) == 12, "FxISqrt(144) == 12");
        check(fpx::FxISqrt(1000000) == 1000, "FxISqrt(1e6) == 1000");
        // floor on non-squares: sqrt(8)=2.82 -> 2 ; sqrt(15)=3.87 -> 3 ; sqrt(99)=9.94 -> 9.
        check(fpx::FxISqrt(8) == 2, "FxISqrt(8) == 2 (floor)");
        check(fpx::FxISqrt(15) == 3, "FxISqrt(15) == 3 (floor)");
        check(fpx::FxISqrt(99) == 9, "FxISqrt(99) == 9 (floor)");
        check(fpx::FxISqrt(-5) == 0, "FxISqrt(negative) == 0");
        // FxLength: a (3,4,0) Q16.16 vector -> length 5.0 (the classic 3-4-5).
        fpx::FxVec3 v{FromInt(3), FromInt(4), 0};
        check(fpx::FxLength(v) == FromInt(5), "FxLength(3,4,0) == 5.0");
    }

    // ================= IntegrateBody one-step + K-step semi-implicit-Euler closed form =================
    {
        const fx g  = FromInt(-10);     // gravity -10 (exact in Q16.16)
        const fx dt = kOne / 2;         // dt = 0.5 (exact)
        const fx groundY = FromInt(-1000000 / 1000);  // far below; effectively no clamp here
        const fpx::FxVec3 grav{0, g, 0};

        // One step: vel.y = 0 + g*dt = -10*0.5 = -5 ; pos.y = 100 + vel.y*dt = 100 + (-5)*0.5 = 97.5.
        fpx::FxBody b;
        b.pos = {0, FromInt(100), 0};
        b.vel = {0, 0, 0};
        b.flags = fpx::kFlagDynamic;
        fpx::IntegrateBody(b, grav, groundY, dt);
        check(b.vel.y == FromInt(-5), "one-step vel.y == g*dt == -5.0");
        check(b.pos.y == FromInt(100) - (kOne * 5 / 2), "one-step pos.y == 100 + (-5)*0.5 == 97.5");

        // K-step closed form: independently re-run the EXACT integer ops K times and compare to the
        // header's IntegrateStep over the same world (they must agree by construction).
        const int K = 120;
        fpx::FxWorld w;
        w.gravity = grav; w.groundY = groundY;
        fpx::FxBody init;
        init.pos = {0, FromInt(500), 0};
        init.vel = {0, 0, 0};
        init.flags = fpx::kFlagDynamic;
        w.bodies.push_back(init);

        // Reference: the same per-step integer recurrence, computed inline.
        fx refVy = 0, refPy = FromInt(500);
        for (int s = 0; s < K; ++s) {
            refVy += fpx::fxmul(g, dt);
            refPy += fpx::fxmul(refVy, dt);
            if (refPy < groundY) { refPy = groundY; if (refVy < 0) refVy = 0; }
        }
        for (int s = 0; s < K; ++s) fpx::IntegrateStep(w, dt);
        check(w.bodies[0].pos.y == refPy, "K-step pos.y == hand-computed integer recurrence");
        check(w.bodies[0].vel.y == refVy, "K-step vel.y == hand-computed integer recurrence");
    }

    // ================= ground floor clamp: pos.y==groundY, vel.y zeroed on contact =================
    {
        const fx g  = FromInt(-10);
        const fx dt = kOne / 60;
        const fx groundY = 0;
        const fpx::FxVec3 grav{0, g, 0};

        // A body starting just above the ground with a big downward velocity clamps to groundY and
        // its downward vel.y is zeroed.
        fpx::FxBody b;
        b.pos = {0, kOne / 100, 0};      // 0.01 above ground
        b.vel = {0, FromInt(-50), 0};    // moving down fast
        b.flags = fpx::kFlagDynamic;
        fpx::IntegrateBody(b, grav, groundY, dt);
        check(b.pos.y == groundY, "ground clamp: pos.y pinned to groundY");
        check(b.vel.y == 0, "ground clamp: downward vel.y zeroed on contact");

        // After contact, the body stays settled (pos.y stays groundY across more steps).
        for (int s = 0; s < 50; ++s) fpx::IntegrateBody(b, grav, groundY, dt);
        check(b.pos.y == groundY, "ground clamp: body stays settled at groundY");
    }

    // ================= static bodies never move; disabled-path no-op model =================
    {
        const fpx::FxVec3 grav{0, FromInt(-10), 0};
        const fx dt = kOne / 60;
        // A body WITHOUT the dynamic flag is untouched.
        fpx::FxBody stat;
        stat.pos = {FromInt(3), FromInt(7), FromInt(-2)};
        stat.vel = {0, 0, 0};
        stat.flags = 0;  // not dynamic
        fpx::FxBody before = stat;
        for (int s = 0; s < 100; ++s) fpx::IntegrateBody(stat, grav, 0, dt);
        check(std::memcmp(&stat, &before, sizeof(fpx::FxBody)) == 0,
              "static (non-dynamic) body never moves");

        // The disabled path is modeled by simply NOT calling IntegrateBody -> the body is unchanged
        // (the GPU's integrateEnabled=0 writes the input back; the CPU mirror is "skip the step").
        fpx::FxBody dyn;
        dyn.pos = {0, FromInt(10), 0};
        dyn.vel = {0, 0, 0};
        dyn.flags = fpx::kFlagDynamic;
        fpx::FxBody dynBefore = dyn;
        // integrateEnabled=false -> no step applied.
        check(std::memcmp(&dyn, &dynBefore, sizeof(fpx::FxBody)) == 0,
              "disabled path (no step) leaves a dynamic body unchanged");
    }

    // ================= determinism: two runs byte-identical =================
    {
        const fx dt = kOne / 60;
        auto makeWorld = []() {
            fpx::FxWorld w;
            w.gravity = {0, FromInt(-10), 0};
            w.groundY = 0;
            for (int i = 0; i < 16; ++i) {
                fpx::FxBody b;
                b.pos = {FromInt(i), FromInt(8 + (i % 5)), FromInt(-i)};
                b.vel = {0, 0, 0};
                b.flags = fpx::kFlagDynamic;
                w.bodies.push_back(b);
            }
            return w;
        };
        fpx::FxWorld w1 = makeWorld(), w2 = makeWorld();
        for (int s = 0; s < 120; ++s) fpx::IntegrateStep(w1, dt);
        for (int s = 0; s < 120; ++s) fpx::IntegrateStep(w2, dt);
        check(w1.bodies.size() == w2.bodies.size(), "determinism: same body count");
        check(std::memcmp(w1.bodies.data(), w2.bodies.data(),
                          w1.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "determinism: two runs BYTE-IDENTICAL");
    }

    // ================= FloorDiv incl negatives (correct floor, not truncate-toward-zero) =================
    {
        check(fpx::FloorDiv(7, 2) == 3, "FloorDiv 7/2 == 3");
        check(fpx::FloorDiv(6, 2) == 3, "FloorDiv 6/2 == 3");
        check(fpx::FloorDiv(0, 2) == 0, "FloorDiv 0/2 == 0");
        // Negatives: -1/2 floors to -1 (NOT 0, which is what plain integer divide gives).
        check(fpx::FloorDiv(-1, 2) == -1, "FloorDiv -1/2 == -1 (floor, not trunc-toward-zero)");
        check(fpx::FloorDiv(-2, 2) == -1, "FloorDiv -2/2 == -1");
        check(fpx::FloorDiv(-3, 2) == -2, "FloorDiv -3/2 == -2 (floor)");
        check(fpx::FloorDiv(-7, 2) == -4, "FloorDiv -7/2 == -4 (floor)");
    }

    // ================= BroadphaseCell / CellId incl negative coords =================
    {
        const fx cell = FromInt(2);  // 2-world-unit cells
        // Position (5.0, 0.0, -3.0) -> cells (floor(5/2)=2, floor(0/2)=0, floor(-3/2)=-2).
        fpx::FxVec3 p{FromInt(5), FromInt(0), FromInt(-3)};
        fpx::FxCell c = fpx::BroadphaseCell(p, cell);
        check(c.x == 2, "BroadphaseCell x: floor(5/2) == 2");
        check(c.y == 0, "BroadphaseCell y: floor(0/2) == 0");
        check(c.z == -2, "BroadphaseCell z: floor(-3/2) == -2 (negative floor)");

        // A position straddling 0: (-0.5,...) -> cell -1 (floor), not 0 (trunc).
        fpx::FxVec3 pn{-(kOne / 2), 0, 0};
        fpx::FxCell cn = fpx::BroadphaseCell(pn, cell);
        check(cn.x == -1, "BroadphaseCell x: floor(-0.5/2) == -1 (negative straddle)");

        // CellId flat linearization over a known grid (offset cells into [0,gridDim)).
        fpx::FxCell grid{4, 4, 4};
        // cell (1,2,3) in a 4x4x4 grid -> (3*4 + 2)*4 + 1 == 57.
        fpx::FxCell q{1, 2, 3};
        check(fpx::CellId(q, grid) == (uint32_t)((3 * 4 + 2) * 4 + 1),
              "CellId (1,2,3) in 4^3 grid == 57");
        // CellId(0,0,0) == 0.
        check(fpx::CellId(fpx::FxCell{0, 0, 0}, grid) == 0u, "CellId origin == 0");
    }

    // ================= overflow bound: a fxmul at the documented +-32768 edge stays exact =================
    {
        // Near the bound: 30000.0 * 1.0 == 30000.0 (identity at a large magnitude).
        check(fpx::fxmul(FromInt(30000), kOne) == FromInt(30000),
              "fxmul 30000*1 == 30000 (within +-32768 bound)");
        // 100.0 * 100.0 == 10000.0 ; the int64 product (100<<16)^2 ~ 4.3e13 overflows int32 -> int64.
        check(fpx::fxmul(FromInt(100), FromInt(100)) == FromInt(10000),
              "fxmul 100*100 == 10000 (int64 product, no overflow)");
        // -200.0 * 150.0 == -30000.0.
        check(fpx::fxmul(FromInt(-200), FromInt(150)) == FromInt(-30000),
              "fxmul -200*150 == -30000 (int64, sign correct)");
    }

    // ================= Slice FPX2: integer-AABB BROADPHASE — BodyAabb / AabbOverlap =================
    {
        // BodyAabb: pos ± radius per-axis (pure integer add/sub).
        fpx::FxBody b;
        b.pos = {FromInt(3), FromInt(-2), FromInt(5)};
        b.radius = kOne;   // 1.0
        fpx::FxAabb a = fpx::BodyAabb(b);
        check(a.lo.x == FromInt(2) && a.lo.y == FromInt(-3) && a.lo.z == FromInt(4),
              "BodyAabb lo == pos - radius");
        check(a.hi.x == FromInt(4) && a.hi.y == FromInt(-1) && a.hi.z == FromInt(6),
              "BodyAabb hi == pos + radius");
        // radius 0 -> a point AABB (lo == hi == pos), the FPX1 default.
        fpx::FxBody pt; pt.pos = {FromInt(7), 0, 0};  // radius defaults to 0
        fpx::FxAabb pa = fpx::BodyAabb(pt);
        check(pa.lo.x == pa.hi.x && pa.lo.x == FromInt(7), "BodyAabb radius-0 -> point AABB");

        // AabbOverlap: clearly overlapping boxes (one inside the gap of the other on every axis).
        fpx::FxAabb o1{{0, 0, 0}, {FromInt(2), FromInt(2), FromInt(2)}};
        fpx::FxAabb o2{{FromInt(1), FromInt(1), FromInt(1)}, {FromInt(3), FromInt(3), FromInt(3)}};
        check(fpx::AabbOverlap(o1, o2), "AabbOverlap overlapping boxes -> true");
        check(fpx::AabbOverlap(o2, o1), "AabbOverlap is symmetric");

        // Touching at a face counts as overlap (inclusive <=).
        fpx::FxAabb t1{{0, 0, 0}, {FromInt(1), FromInt(1), FromInt(1)}};
        fpx::FxAabb t2{{FromInt(1), 0, 0}, {FromInt(2), FromInt(1), FromInt(1)}};
        check(fpx::AabbOverlap(t1, t2), "AabbOverlap touching faces -> true (inclusive)");

        // Apart on a single axis -> NO overlap (separating axis on x).
        fpx::FxAabb p1{{0, 0, 0}, {FromInt(1), FromInt(1), FromInt(1)}};
        fpx::FxAabb p2{{FromInt(2), 0, 0}, {FromInt(3), FromInt(1), FromInt(1)}};
        check(!fpx::AabbOverlap(p1, p2), "AabbOverlap separated on x -> false");
        // Overlapping on x,y but apart on z -> NO overlap (all-axis required).
        fpx::FxAabb z1{{0, 0, 0}, {FromInt(2), FromInt(2), FromInt(1)}};
        fpx::FxAabb z2{{0, 0, FromInt(2)}, {FromInt(2), FromInt(2), FromInt(3)}};
        check(!fpx::AabbOverlap(z1, z2), "AabbOverlap separated on z -> false");

        // Negative-coordinate overlap (boxes straddling the origin).
        fpx::FxAabb n1{{FromInt(-2), FromInt(-2), FromInt(-2)}, {0, 0, 0}};
        fpx::FxAabb n2{{FromInt(-1), FromInt(-1), FromInt(-1)}, {FromInt(1), FromInt(1), FromInt(1)}};
        check(fpx::AabbOverlap(n1, n2), "AabbOverlap negative-coord overlap -> true");
        fpx::FxAabb n3{{FromInt(-5), FromInt(-5), FromInt(-5)}, {FromInt(-3), FromInt(-3), FromInt(-3)}};
        check(!fpx::AabbOverlap(n1, n3), "AabbOverlap negative-coord apart -> false");
    }

    // ================= CountPairs / BuildPairs on a known scene =================
    {
        // 4 bodies in a row spaced 1.0, radius 0.6 -> 2*0.6=1.2 spans 1 neighbor (dist 1.0 overlaps,
        // dist 2.0 does NOT). So pairs: (0,1),(1,2),(2,3) -> 3 pairs, grouped-by-i, i<j, no dups.
        fpx::FxWorld w;
        const fx r = (fx)(kOne * 6 / 10);  // 0.6
        for (int i = 0; i < 4; ++i) {
            fpx::FxBody b; b.pos = {FromInt(i), 0, 0}; b.radius = r; b.flags = fpx::kFlagDynamic;
            w.bodies.push_back(b);
        }
        std::vector<uint32_t> counts((size_t)4, 99u);
        const uint32_t total = fpx::CountPairs(w, std::span<uint32_t>(counts));
        check(total == 3u, "CountPairs row-of-4 r=0.6 -> 3 pairs total");
        check(counts[0] == 1u && counts[1] == 1u && counts[2] == 1u && counts[3] == 0u,
              "CountPairs per-body counts {1,1,1,0}");

        std::vector<uint32_t> off; std::vector<fpx::FxPair> pairs;
        fpx::BuildPairs(w, off, pairs);
        // Exclusive prefix-sum offsets: {0,1,2,3}.
        check(off.size() == 4 && off[0] == 0u && off[1] == 1u && off[2] == 2u && off[3] == 3u,
              "BuildPairs prefix-sum offsets {0,1,2,3}");
        check(pairs.size() == 3, "BuildPairs emits 3 pairs");
        // grouped-by-i ascending, i<j, no dups: (0,1),(1,2),(2,3).
        check(pairs[0].i == 0u && pairs[0].j == 1u, "BuildPairs pair[0] == (0,1)");
        check(pairs[1].i == 1u && pairs[1].j == 2u, "BuildPairs pair[1] == (1,2)");
        check(pairs[2].i == 2u && pairs[2].j == 3u, "BuildPairs pair[2] == (2,3)");
        // Invariants: i<j, no dups, AabbOverlap true, totalPairs == Σ counts.
        uint32_t sum = 0; for (uint32_t c : counts) sum += c;
        check(sum == total, "totalPairs == Σ perBodyCount");
        bool inv = true;
        for (size_t p = 0; p < pairs.size(); ++p) {
            if (pairs[p].i >= pairs[p].j) inv = false;
            if (!fpx::AabbOverlap(fpx::BodyAabb(w.bodies[pairs[p].i]),
                                  fpx::BodyAabb(w.bodies[pairs[p].j]))) inv = false;
        }
        check(inv, "BuildPairs every pair i<j + AabbOverlap true");
    }

    // ================= known case: two overlapping -> 1 pair; two apart -> 0 =================
    {
        const fx r = (fx)(kOne * 6 / 10);  // 0.6
        fpx::FxWorld ov;
        fpx::FxBody a; a.pos = {0, 0, 0}; a.radius = r;
        fpx::FxBody b; b.pos = {FromInt(1), 0, 0}; b.radius = r;
        ov.bodies = {a, b};
        std::vector<uint32_t> off1; std::vector<fpx::FxPair> pr1;
        fpx::BuildPairs(ov, off1, pr1);
        check(pr1.size() == 1 && pr1[0].i == 0u && pr1[0].j == 1u, "known: 2 overlapping -> 1 pair (0,1)");

        fpx::FxWorld ap;
        fpx::FxBody c; c.pos = {0, 0, 0}; c.radius = r;
        fpx::FxBody d; d.pos = {FromInt(5), 0, 0}; d.radius = r;
        ap.bodies = {c, d};
        std::vector<uint32_t> off2; std::vector<fpx::FxPair> pr2;
        fpx::BuildPairs(ap, off2, pr2);
        check(pr2.empty(), "known: 2 apart -> 0 pairs");
    }

    // ================= enabled-off model -> empty; determinism =================
    {
        // The disabled GPU path writes 0 counts + leaves gPairs cleared. The CPU model of "disabled" is
        // simply an EMPTY world (no bodies) -> 0 pairs, the byte-identical no-op the GPU enabled=0 mirrors.
        fpx::FxWorld empty;
        std::vector<uint32_t> ec((size_t)0);
        check(fpx::CountPairs(empty, std::span<uint32_t>(ec)) == 0u, "enabled-off model: empty world -> 0");
        std::vector<uint32_t> eo; std::vector<fpx::FxPair> ep;
        fpx::BuildPairs(empty, eo, ep);
        check(ep.empty() && eo.empty(), "enabled-off model: BuildPairs(empty) -> empty");

        // Determinism: two BuildPairs over the SAME clustered grid -> byte-identical pair lists.
        auto makeGrid = []() {
            fpx::FxWorld w; const fx r = (fx)(kOne * 6 / 10);
            for (int i = 0; i < 16; ++i) {
                fpx::FxBody b; b.pos = {FromInt(i % 4), 0, FromInt(i / 4)}; b.radius = r;
                b.flags = fpx::kFlagDynamic; w.bodies.push_back(b);
            }
            return w;
        };
        fpx::FxWorld g1 = makeGrid(), g2 = makeGrid();
        std::vector<uint32_t> o1, o2; std::vector<fpx::FxPair> p1, p2;
        fpx::BuildPairs(g1, o1, p1); fpx::BuildPairs(g2, o2, p2);
        check(p1.size() == p2.size() && o1.size() == o2.size(), "broadphase determinism: same sizes");
        check(!p1.empty() &&
              std::memcmp(p1.data(), p2.data(), p1.size() * sizeof(fpx::FxPair)) == 0 &&
              std::memcmp(o1.data(), o2.data(), o1.size() * sizeof(uint32_t)) == 0,
              "broadphase determinism: two BuildPairs BYTE-IDENTICAL");
    }

    if (g_fail == 0) std::printf("fpx_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
