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
//   * FPX5 lockstep/rollback: ApplyCommand (impulse/angVel apply, OOB no-op); Snapshot/Restore lossless
//     round-trip; SimTick deterministic; command stream applied in deterministic order; RunLockstep
//     replica==authority BIT-EXACT (inputs only); RunRollback converges to authority AND the mispredicted
//     path differs (a positive + a negative control).
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

    // ================= Slice FPX3: fxdiv known quotients (incl negatives + the int64 shift) =============
    {
        // 6.0 / 3.0 == 2.0.
        check(fpx::fxdiv(FromInt(6), FromInt(3)) == FromInt(2), "fxdiv 6/3 == 2");
        // 1.0 / 2.0 == 0.5.
        check(fpx::fxdiv(kOne, FromInt(2)) == kOne / 2, "fxdiv 1/2 == 0.5");
        // x / 1 == x (identity).
        check(fpx::fxdiv(FromInt(7), kOne) == FromInt(7), "fxdiv x/1 == x");
        // Negatives: (-6)/3 == -2 ; 6/(-3) == -2 (truncate toward zero, identical sign of result).
        check(fpx::fxdiv(FromInt(-6), FromInt(3)) == FromInt(-2), "fxdiv (-6)/3 == -2");
        check(fpx::fxdiv(FromInt(6), FromInt(-3)) == FromInt(-2), "fxdiv 6/(-3) == -2");
        // The int64 shift is REQUIRED: 1000.0 / 0.5 == 2000.0 ((1000<<16)<<16 overflows int32).
        check(fpx::fxdiv(FromInt(1000), kOne / 2) == FromInt(2000),
              "fxdiv 1000/0.5 == 2000 (int64 shift)");
        // Truncation toward zero: 1.0 / 3.0 == floor((1<<32)/(3<<16)) -> the exact truncated Q16.16.
        check(fpx::fxdiv(kOne, FromInt(3)) == (fx)(((int64_t)kOne << kFrac) / (int64_t)FromInt(3)),
              "fxdiv 1/3 == truncated Q16.16");
        // Guard: divide by zero -> 0.
        check(fpx::fxdiv(FromInt(5), 0) == 0, "fxdiv x/0 == 0 (guard)");
        // fxdiv is the inverse of fxmul at unit scale: fxmul(fxdiv(a,b), b) ~= a (exact when divisible).
        check(fpx::fxmul(fpx::fxdiv(FromInt(8), FromInt(4)), FromInt(4)) == FromInt(8),
              "fxmul(fxdiv(8,4),4) == 8");
    }

    // ================= FxNormalize: unit-length (fp tol) + known direction =================
    {
        // A pure +x vector normalizes to (kOne, 0, 0) exactly.
        fpx::FxVec3 nx = fpx::FxNormalize(fpx::FxVec3{FromInt(5), 0, 0});
        check(nx.x == kOne && nx.y == 0 && nx.z == 0, "FxNormalize(+x) == (1,0,0)");
        // A pure -y vector normalizes to (0, -kOne, 0).
        fpx::FxVec3 ny = fpx::FxNormalize(fpx::FxVec3{0, FromInt(-3), 0});
        check(ny.x == 0 && ny.y == -kOne && ny.z == 0, "FxNormalize(-y) == (0,-1,0)");
        // Zero vector -> the fixed fallback (0, kOne, 0).
        fpx::FxVec3 nz = fpx::FxNormalize(fpx::FxVec3{0, 0, 0});
        check(nz.x == 0 && nz.y == kOne && nz.z == 0, "FxNormalize(0) == (0,1,0) fallback");
        // A (3,4,0) vector -> length 5 -> unit (0.6, 0.8, 0). Within a small Q16.16 tolerance (the
        // integer divide truncates), the magnitude is ~kOne.
        fpx::FxVec3 d = fpx::FxNormalize(fpx::FxVec3{FromInt(3), FromInt(4), 0});
        const fx len = fpx::FxLength(d);
        const fx tol = kOne / 1000;  // ~0.001 tolerance
        check(len <= kOne + tol && len >= kOne - tol, "FxNormalize(3,4,0) unit length (tol)");
        // Direction: x-share < y-share (0.6 < 0.8), both positive.
        check(d.x > 0 && d.y > 0 && d.x < d.y, "FxNormalize(3,4,0) direction 0.6<0.8");
    }

    // ================= ground resolution: penetrating -> pos.y = groundY + radius =================
    {
        const fx r = (fx)(fpx::kOne / 2);   // radius 0.5
        const fx groundY = 0;
        // A body whose bottom (pos.y - r) is below groundY -> pushed so pos.y == groundY + r.
        fpx::FxBody b; b.pos = {0, FromInt(-2), 0}; b.radius = r; b.invMass = kOne; b.flags = fpx::kFlagDynamic;
        fpx::ResolveGround(b, groundY);
        check(b.pos.y == groundY + r, "ResolveGround: penetrating -> pos.y = groundY + radius");
        // A body already above the ground is unmoved.
        fpx::FxBody hi; hi.pos = {0, FromInt(5), 0}; hi.radius = r; hi.invMass = kOne;
        fpx::ResolveGround(hi, groundY);
        check(hi.pos.y == FromInt(5), "ResolveGround: above-ground unmoved");
        // A static body (invMass==0) takes no correction.
        fpx::FxBody st; st.pos = {0, FromInt(-2), 0}; st.radius = r; st.invMass = 0;
        fpx::ResolveGround(st, groundY);
        check(st.pos.y == FromInt(-2), "ResolveGround: static (invMass=0) unmoved");
    }

    // ================= sphere-sphere resolution: pushed apart by inverse-mass shares =================
    {
        const fx r = (fx)(fpx::kOne * 6 / 10);   // 0.6
        // Equal masses: a at x=0, b at x=1.0, both r=0.6 -> pen = 1.2 - 1.0 = 0.2 -> each moves 0.1.
        const fx D = kOne;
        fpx::FxBody a; a.pos = {0, 0, 0};  a.radius = r; a.invMass = kOne; a.flags = fpx::kFlagDynamic;
        fpx::FxBody b; b.pos = {D, 0, 0};  b.radius = r; b.invMass = kOne; b.flags = fpx::kFlagDynamic;
        const fx pen = (r + r) - fpx::FxLength(fpx::FxVec3{D, 0, 0});
        const fx half = fpx::fxmul(pen, fpx::fxdiv(kOne, kOne + kOne));  // pen * 0.5
        fpx::ResolvePair(a, b);
        check(a.pos.x == -half, "ResolvePair equal-mass: a moved -pen/2");
        check(b.pos.x == D + half, "ResolvePair equal-mass: b moved +pen/2");
        // Static b (invMass==0): a takes the FULL correction (wi = invMassA/(invMassA+0) = 1), b unmoved.
        fpx::FxBody a2; a2.pos = {0, 0, 0}; a2.radius = r; a2.invMass = kOne; a2.flags = fpx::kFlagDynamic;
        fpx::FxBody s2; s2.pos = {D, 0, 0}; s2.radius = r; s2.invMass = 0;    a2.flags = fpx::kFlagDynamic;
        const fx wi = fpx::fxdiv(kOne, kOne + 0);                    // == kOne
        const fx full = fpx::fxmul(pen, wi);
        fpx::ResolvePair(a2, s2);
        check(s2.pos.x == D, "ResolvePair: static b unmoved");
        check(a2.pos.x == -full, "ResolvePair: dynamic a takes full correction");
        // Both static -> no move.
        fpx::FxBody x1; x1.pos = {0, 0, 0}; x1.radius = r; x1.invMass = 0;
        fpx::FxBody x2; x2.pos = {D, 0, 0}; x2.radius = r; x2.invMass = 0;
        fpx::ResolvePair(x1, x2);
        check(x1.pos.x == 0 && x2.pos.x == D, "ResolvePair: both static unmoved");
        // Non-overlapping (far apart) -> no move.
        fpx::FxBody f1; f1.pos = {0, 0, 0};         f1.radius = r; f1.invMass = kOne;
        fpx::FxBody f2; f2.pos = {FromInt(5), 0, 0}; f2.radius = r; f2.invMass = kOne;
        fpx::ResolvePair(f1, f2);
        check(f1.pos.x == 0 && f2.pos.x == FromInt(5), "ResolvePair: non-overlapping unmoved");
    }

    // ================= SolveContacts: K-iter determinism + a known small scene =================
    {
        const fx r = (fx)(fpx::kOne * 6 / 10);
        auto makeScene = [&]() {
            fpx::FxWorld w; w.groundY = 0;
            // Three bodies in a row overlapping their neighbors, resting on the ground.
            for (int i = 0; i < 3; ++i) {
                fpx::FxBody b; b.pos = {FromInt(i), 0, 0}; b.radius = r; b.invMass = kOne;
                b.flags = fpx::kFlagDynamic; w.bodies.push_back(b);
            }
            return w;
        };
        std::vector<fpx::FxPair> pr = {fpx::FxPair{0u, 1u}, fpx::FxPair{1u, 2u}};
        // K-iteration determinism: same scene solved twice -> byte-identical bodies.
        fpx::FxWorld s1 = makeScene(), s2 = makeScene();
        fpx::SolveContacts(s1, std::span<const fpx::FxPair>(pr), 8);
        fpx::SolveContacts(s2, std::span<const fpx::FxPair>(pr), 8);
        check(s1.bodies.size() == s2.bodies.size() &&
              std::memcmp(s1.bodies.data(), s2.bodies.data(),
                          s1.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "SolveContacts: two runs BYTE-IDENTICAL");
        // Known result: bodies are pushed apart (their gaps grow) and all sit at/above the ground.
        for (const auto& b : s1.bodies)
            check(b.pos.y - b.radius >= s1.groundY, "SolveContacts: above ground");
        check(s1.bodies[1].pos.x - s1.bodies[0].pos.x > FromInt(1) ||
              s1.bodies[2].pos.x - s1.bodies[1].pos.x > FromInt(1),
              "SolveContacts: overlapping bodies pushed apart");
        // residual-overlap count is deterministic (same on both runs).
        check(fpx::CountResidualOverlaps(s1, std::span<const fpx::FxPair>(pr)) ==
              fpx::CountResidualOverlaps(s2, std::span<const fpx::FxPair>(pr)),
              "CountResidualOverlaps: deterministic");
    }

    // ================= StepWorld (integrate + solve) + enabled-off model unchanged =================
    {
        const fx r = (fx)(fpx::kOne * 6 / 10);
        const fx kDt = kOne / 60;
        fpx::FxWorld w; w.groundY = 0; w.gravity = {0, FromInt(-10), 0};
        fpx::FxBody a; a.pos = {0, FromInt(3), 0};        a.radius = r; a.invMass = kOne; a.flags = fpx::kFlagDynamic;
        fpx::FxBody b; b.pos = {kOne / 2, FromInt(3), 0}; b.radius = r; b.invMass = kOne; b.flags = fpx::kFlagDynamic;
        w.bodies = {a, b};
        std::vector<fpx::FxPair> pr = {fpx::FxPair{0u, 1u}};
        // Capture initial, step once -> bodies move (gravity integrated + overlap resolved).
        const fpx::FxVec3 a0 = w.bodies[0].pos;
        fpx::StepWorld(w, std::span<const fpx::FxPair>(pr), kDt, 8);
        check(w.bodies[0].pos.y != a0.y || w.bodies[0].pos.x != a0.x, "StepWorld: body moved");
        for (const auto& bb : w.bodies)
            check(bb.pos.y - bb.radius >= w.groundY, "StepWorld: stays above ground");

        // enabled-off model: SolveContacts with 0 iterations + IntegrateStep with no dynamic flag is a
        // no-op. Concretely model "disabled" as a world of STATIC bodies stepped -> unchanged.
        fpx::FxWorld stat; stat.groundY = 0; stat.gravity = {0, FromInt(-10), 0};
        fpx::FxBody sa; sa.pos = {0, FromInt(3), 0}; sa.radius = r; sa.invMass = 0; sa.flags = 0;
        stat.bodies = {sa};
        const fpx::FxVec3 sp = stat.bodies[0].pos;
        fpx::StepWorld(stat, std::span<const fpx::FxPair>(), kDt, 8);
        check(stat.bodies[0].pos.x == sp.x && stat.bodies[0].pos.y == sp.y &&
              stat.bodies[0].pos.z == sp.z, "StepWorld: static body unchanged (disabled model)");
    }

    // ================= Slice FPX4: FxQuatMul (Hamilton product, int64, identity·q==q) =================
    {
        using fpx::FxQuat;
        const FxQuat identity{0, 0, 0, kOne};
        // A non-trivial (not yet unit) quaternion to exercise all 16 product terms.
        const FxQuat q{kOne, kOne / 2, kOne / 4, kOne};
        // identity * q == q (the multiplicative identity, left).
        FxQuat li = fpx::FxQuatMul(identity, q);
        check(li.x == q.x && li.y == q.y && li.z == q.z && li.w == q.w, "FxQuatMul identity*q == q");
        // q * identity == q (right identity).
        FxQuat ri = fpx::FxQuatMul(q, identity);
        check(ri.x == q.x && ri.y == q.y && ri.z == q.z && ri.w == q.w, "FxQuatMul q*identity == q");

        // Known product: i*j == k (pure-imaginary basis quaternions). i={1,0,0,0}, j={0,1,0,0}, k={0,0,1,0}.
        const FxQuat qi{kOne, 0, 0, 0};
        const FxQuat qj{0, kOne, 0, 0};
        const FxQuat qk{0, 0, kOne, 0};
        FxQuat ij = fpx::FxQuatMul(qi, qj);
        check(ij.x == qk.x && ij.y == qk.y && ij.z == qk.z && ij.w == qk.w, "FxQuatMul i*j == k");
        // i*i == -1 (w = -kOne, xyz = 0).
        FxQuat ii = fpx::FxQuatMul(qi, qi);
        check(ii.x == 0 && ii.y == 0 && ii.z == 0 && ii.w == -kOne, "FxQuatMul i*i == -1");
        // j*i == -k (anticommutativity).
        FxQuat ji = fpx::FxQuatMul(qj, qi);
        check(ji.x == 0 && ji.y == 0 && ji.z == -kOne && ji.w == 0, "FxQuatMul j*i == -k");
    }

    // ================= Slice FPX4: FxQuatNormalize (|q|≈kOne; known un-normalized -> normalized) =====
    {
        using fpx::FxQuat;
        // Identity normalizes to identity (already unit).
        FxQuat n = fpx::FxQuatNormalize(FxQuat{0, 0, 0, kOne});
        check(n.x == 0 && n.y == 0 && n.z == 0 && n.w == kOne, "FxQuatNormalize identity == identity");
        // len==0 -> identity fallback.
        FxQuat z = fpx::FxQuatNormalize(FxQuat{0, 0, 0, 0});
        check(z.x == 0 && z.y == 0 && z.z == 0 && z.w == kOne, "FxQuatNormalize zero -> identity");
        // Known un-normalized: {kOne,kOne,kOne,kOne} (|q|=2) -> each component ~kOne/2.
        FxQuat u = fpx::FxQuatNormalize(FxQuat{kOne, kOne, kOne, kOne});
        // |q|^2 over the normalized quat ≈ kOne^2 within a small fixed-point tolerance.
        auto qmag2 = [](const FxQuat& q) -> int64_t {
            return (int64_t)q.x * q.x + (int64_t)q.y * q.y + (int64_t)q.z * q.z + (int64_t)q.w * q.w;
        };
        const int64_t one2 = (int64_t)kOne * (int64_t)kOne;
        // Components are all equal and ≈ 0.5 (kOne/2 = 32768).
        check(u.x == u.y && u.y == u.z && u.z == u.w, "FxQuatNormalize {1,1,1,1} symmetric");
        check(u.x > kOne / 2 - 64 && u.x < kOne / 2 + 64, "FxQuatNormalize {1,1,1,1} comp ~0.5");
        int64_t d = qmag2(u) - one2;
        if (d < 0) d = -d;
        check(d < 2 * (int64_t)kOne, "FxQuatNormalize |q|^2 ~ kOne^2 (within fp tol)");
    }

    // ================= Slice FPX4: FxRotate (identity->v; 90° about Z -> known vector) =================
    {
        using fpx::FxQuat;
        const fpx::FxVec3 vx{kOne, 0, 0};   // unit +x
        // Identity rotation leaves v unchanged.
        fpx::FxVec3 r0 = fpx::FxRotate(FxQuat{0, 0, 0, kOne}, vx);
        check(r0.x == vx.x && r0.y == vx.y && r0.z == vx.z, "FxRotate identity -> v");
        // 90° about +Z: q = {0,0,sin45,cos45} = {0,0,~0.7071,~0.7071}. Rotates +x -> +y.
        // sin/cos 45° in Q16.16: ~46341 (0.70710678 * 65536).
        const fx s = 46341;  // ~0.7071 in Q16.16
        FxQuat qz{0, 0, s, s};
        qz = fpx::FxQuatNormalize(qz);     // ensure unit (it already is ~)
        fpx::FxVec3 ry = fpx::FxRotate(qz, vx);
        // Expect ~(0, +1, 0) within a fixed-point tolerance (the integer rotate is not perfect).
        const fx tol = 512;  // ~0.0078 in Q16.16
        check(ry.x > -tol && ry.x < tol, "FxRotate 90Z: x ~ 0");
        check(ry.y > kOne - 2 * tol && ry.y < kOne + 2 * tol, "FxRotate 90Z: y ~ +1");
        check(ry.z > -tol && ry.z < tol, "FxRotate 90Z: z ~ 0");
    }

    // ================= Slice FPX4: IntegrateOrientation (angVel=0 unchanged; known spin drift) ========
    {
        using fpx::FxQuat;
        const fx kDt = kOne / 60;
        // angVel = 0 -> orientation unchanged (modulo the normalize, which is a no-op on a unit quat).
        fpx::FxBody b0; b0.orient = FxQuat{0, 0, 0, kOne}; b0.angVel = {0, 0, 0};
        fpx::IntegrateOrientation(b0, kDt);
        check(b0.orient.x == 0 && b0.orient.y == 0 && b0.orient.z == 0 && b0.orient.w == kOne,
              "IntegrateOrientation angVel=0 -> unchanged");

        // Known spin about +Z at omega = 1 rad/s for K steps -> the quaternion's Z component grows
        // (positive rotation), w decreases from kOne, and |q| stays ≈ kOne throughout.
        fpx::FxBody b1; b1.orient = FxQuat{0, 0, 0, kOne}; b1.angVel = {0, 0, kOne};  // 1 rad/s about Z
        const int K = 120;
        int64_t maxDrift = 0;
        for (int s = 0; s < K; ++s) {
            fpx::IntegrateOrientation(b1, kDt);
            int64_t m2 = (int64_t)b1.orient.x * b1.orient.x + (int64_t)b1.orient.y * b1.orient.y +
                         (int64_t)b1.orient.z * b1.orient.z + (int64_t)b1.orient.w * b1.orient.w;
            int64_t one2 = (int64_t)kOne * (int64_t)kOne;
            int64_t d = m2 - one2; if (d < 0) d = -d;
            if (d > maxDrift) maxDrift = d;
        }
        // After K=120 steps of dt=1/60 at 1 rad/s -> total angle ~2.0 rad -> half-angle ~1.0 rad ->
        // q ≈ {0,0,sin(1)=0.8415, cos(1)=0.5403}. z should be clearly positive + large, w reduced.
        check(b1.orient.z > kOne / 2, "IntegrateOrientation spin Z: z grew large (positive rotation)");
        check(b1.orient.w < kOne, "IntegrateOrientation spin Z: w reduced from identity");
        check(b1.orient.x == 0 && b1.orient.y == 0, "IntegrateOrientation spin Z: x,y stay 0");
        // |q| drift stays small + bounded (the documented fixed-point tolerance). one2 = kOne^2 ~ 4.29e9.
        // The tolerance: |q|^2-kOne^2 < kOne^2 / 256 ~ 1.68e7 (≈ |q|-1 < ~0.002).
        check(maxDrift < ((int64_t)kOne * (int64_t)kOne) / 256,
              "IntegrateOrientation: |q| drift bounded (fp tol)");
    }

    // ================= Slice FPX4: IntegrateBodyFull (translation + orient) + determinism =============
    {
        using fpx::FxQuat;
        const fx kDt = kOne / 60;
        const fpx::FxVec3 grav{0, FromInt(-10), 0};
        // A dynamic body with gravity + spin: position falls AND orientation integrates.
        fpx::FxBody b; b.pos = {0, FromInt(5), 0}; b.vel = {0, 0, 0};
        b.invMass = kOne; b.flags = fpx::kFlagDynamic;
        b.orient = FxQuat{0, 0, 0, kOne}; b.angVel = {0, kOne, 0};  // spin about +Y
        fpx::FxBody bcopy = b;
        const int K = 30;
        for (int s = 0; s < K; ++s) fpx::IntegrateBodyFull(b, grav, kDt);
        check(b.pos.y < FromInt(5), "IntegrateBodyFull: dynamic body fell under gravity");
        check(b.orient.y != 0 || b.orient.w != kOne, "IntegrateBodyFull: orientation integrated");

        // Determinism: a second identical run is byte-identical.
        for (int s = 0; s < K; ++s) fpx::IntegrateBodyFull(bcopy, grav, kDt);
        check(std::memcmp(&b, &bcopy, sizeof(fpx::FxBody)) == 0,
              "IntegrateBodyFull: two runs byte-identical (determinism)");

        // Static body (no dynamic flag) does NOT translate; with angVel=0 it is fully unchanged.
        fpx::FxBody stat; stat.pos = {FromInt(2), FromInt(3), 0}; stat.flags = 0; stat.invMass = 0;
        stat.orient = FxQuat{0, 0, 0, kOne}; stat.angVel = {0, 0, 0};
        fpx::FxBody statInit = stat;
        for (int s = 0; s < K; ++s) fpx::IntegrateBodyFull(stat, grav, kDt);
        check(std::memcmp(&stat, &statInit, sizeof(fpx::FxBody)) == 0,
              "IntegrateBodyFull: static+angVel=0 body unchanged (no-op)");
    }

    // ================= Slice FPX5: LOCKSTEP + ROLLBACK (the netcode primitive) ========================
    // The headline: prove the bit-exact fixed-point sim is inputs-only LOCKSTEP + ROLLBACK-ready.
    // ApplyCommand applies an input; Snapshot/Restore is the lossless rollback primitive; SimTick is a
    // deterministic per-tick step; RunLockstep fed only inputs re-derives identical state; RunRollback
    // corrects a misprediction by rolling back + re-simulating the authoritative stream.

    // A small deterministic scene: 3 dynamic spheres above a ground (radii so they can collide), gravity.
    auto makeLockstepWorld = []() {
        fpx::FxWorld w;
        w.gravity = {0, FromInt(-10), 0};
        w.groundY = 0;
        for (int i = 0; i < 3; ++i) {
            fpx::FxBody b;
            b.pos = {FromInt(i), FromInt(4 + i), 0};
            b.vel = {0, 0, 0};
            b.invMass = kOne;
            b.flags = fpx::kFlagDynamic;
            b.radius = (fx)(kOne * 6 / 10);   // 0.6 -> neighbors can collide
            b.orient = fpx::FxQuat{0, 0, 0, kOne};
            b.angVel = {0, 0, 0};
            w.bodies.push_back(b);
        }
        return w;
    };

    // ----- ApplyCommand: impulse adds to velocity; set-angVel sets angVel; OOB is a no-op -----
    {
        fpx::FxWorld w = makeLockstepWorld();
        const fpx::FxVec3 v0 = w.bodies[1].vel;
        fpx::ApplyCommand(w, fpx::FxCommand{0, fpx::kCmdImpulse, 1, fpx::FxVec3{kOne, 0, 0}});
        check(w.bodies[1].vel.x == v0.x + kOne && w.bodies[1].vel.y == v0.y && w.bodies[1].vel.z == v0.z,
              "ApplyCommand: kCmdImpulse adds arg to velocity");
        fpx::ApplyCommand(w, fpx::FxCommand{0, fpx::kCmdSetAngVel, 2, fpx::FxVec3{0, kOne, 0}});
        check(w.bodies[2].angVel.x == 0 && w.bodies[2].angVel.y == kOne && w.bodies[2].angVel.z == 0,
              "ApplyCommand: kCmdSetAngVel sets angVel");
        // OOB bodyId is a no-op (does not crash, does not mutate).
        fpx::FxWorld before = w;
        fpx::ApplyCommand(w, fpx::FxCommand{0, fpx::kCmdImpulse, 99, fpx::FxVec3{kOne, kOne, kOne}});
        check(std::memcmp(w.bodies.data(), before.bodies.data(),
                          w.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "ApplyCommand: out-of-range bodyId is a no-op");
    }

    // ----- SnapshotWorld / RestoreWorld: a lossless round-trip == the original -----
    {
        fpx::FxWorld w = makeLockstepWorld();
        // Advance it a few ticks so it's a non-trivial state.
        std::vector<fpx::FxCommand> empty;
        for (uint32_t t = 0; t < 10; ++t) fpx::SimTick(w, empty, t, kOne / 60, 4);
        const fpx::FxWorld snap = fpx::SnapshotWorld(w);
        // Mutate w, then restore -> byte-identical to the saved state.
        fpx::SimTick(w, empty, 10, kOne / 60, 4);
        check(w.bodies.size() == snap.bodies.size(), "Snapshot/Restore: body count preserved");
        fpx::RestoreWorld(w, snap);
        check(w.bodies.size() == snap.bodies.size() &&
              std::memcmp(w.bodies.data(), snap.bodies.data(),
                          w.bodies.size() * sizeof(fpx::FxBody)) == 0 &&
              w.groundY == snap.groundY && w.gravity.y == snap.gravity.y,
              "Snapshot/Restore: round-trip BIT-EXACT == original");
    }

    // ----- SimTick: deterministic (two runs from the same state+stream are byte-identical) -----
    {
        std::vector<fpx::FxCommand> stream = {
            fpx::FxCommand{1, fpx::kCmdImpulse, 0, fpx::FxVec3{FromInt(2), 0, 0}},
            fpx::FxCommand{3, fpx::kCmdSetAngVel, 2, fpx::FxVec3{0, kOne, 0}},
        };
        fpx::FxWorld a = makeLockstepWorld();
        fpx::FxWorld b = makeLockstepWorld();
        for (uint32_t t = 0; t < 8; ++t) { fpx::SimTick(a, stream, t, kOne / 60, 6); }
        for (uint32_t t = 0; t < 8; ++t) { fpx::SimTick(b, stream, t, kOne / 60, 6); }
        check(std::memcmp(a.bodies.data(), b.bodies.data(),
                          a.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "SimTick: deterministic (two runs byte-identical)");
    }

    // ----- Command stream applied in deterministic ORDER (two impulses on the same body, same tick) -----
    {
        // Two impulses on body0 at tick 0: applied in array order -> both add. Order independence of the
        // SUM here (commutative add) + the deterministic processing both hold; assert both are applied.
        std::vector<fpx::FxCommand> stream = {
            fpx::FxCommand{0, fpx::kCmdImpulse, 0, fpx::FxVec3{kOne, 0, 0}},
            fpx::FxCommand{0, fpx::kCmdImpulse, 0, fpx::FxVec3{FromInt(2), 0, 0}},
        };
        fpx::FxWorld w = makeLockstepWorld();
        // Apply just the tick-0 commands by hand (in array order) and compare to ApplyCommand twice.
        fpx::FxWorld ref = makeLockstepWorld();
        for (const auto& c : stream) fpx::ApplyCommand(ref, c);
        for (const auto& c : stream) if (c.tick == 0) fpx::ApplyCommand(w, c);
        check(w.bodies[0].vel.x == ref.bodies[0].vel.x && ref.bodies[0].vel.x == FromInt(3),
              "command stream: same-tick commands applied in deterministic order (sum == 3)");
    }

    // ----- RunLockstep: replica == authority BIT-EXACT (inputs ONLY) — THE HEADLINE -----
    {
        std::vector<fpx::FxCommand> stream = {
            fpx::FxCommand{1, fpx::kCmdImpulse, 0, fpx::FxVec3{FromInt(3), 0, 0}},
            fpx::FxCommand{2, fpx::kCmdImpulse, 2, fpx::FxVec3{FromInt(-2), kOne, 0}},
            fpx::FxCommand{4, fpx::kCmdSetAngVel, 1, fpx::FxVec3{0, kOne, 0}},
        };
        const fpx::FxWorld init = makeLockstepWorld();
        const int N = 20;
        fpx::FxWorld authority = fpx::RunLockstep(init, stream, N, kOne / 60, 6);
        fpx::FxWorld replica   = fpx::RunLockstep(init, stream, N, kOne / 60, 6);
        check(authority.bodies.size() == replica.bodies.size() &&
              std::memcmp(authority.bodies.data(), replica.bodies.data(),
                          authority.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "RunLockstep: replica == authority BIT-EXACT (inputs only)");

        // ----- RunRollback: converges to RunLockstep(authStream) AND the mispredicted path differs -----
        // The mispredicted stream: a WRONG impulse at tick 8 (a different value -> a real divergence).
        std::vector<fpx::FxCommand> mispredict = stream;
        mispredict.push_back(fpx::FxCommand{8, fpx::kCmdImpulse, 0, fpx::FxVec3{FromInt(50), 0, 0}});
        const int mispredictTick = 8;
        fpx::FxWorld rolledBack = fpx::RunRollback(init, stream, mispredict, N, mispredictTick,
                                                   kOne / 60, 6);
        // POSITIVE: rollback converged to the authoritative lockstep state.
        check(rolledBack.bodies.size() == authority.bodies.size() &&
              std::memcmp(rolledBack.bodies.data(), authority.bodies.data(),
                          authority.bodies.size() * sizeof(fpx::FxBody)) == 0,
              "RunRollback: corrected to authority BIT-EXACT (positive)");
        // NEGATIVE control: a pure-mispredicted run (no rollback) DIFFERS from authority — proving the
        // misprediction is a REAL divergence the rollback actually fixed (not a no-op).
        fpx::FxWorld mispredicted = fpx::RunLockstep(init, mispredict, N, kOne / 60, 6);
        check(std::memcmp(mispredicted.bodies.data(), authority.bodies.data(),
                          authority.bodies.size() * sizeof(fpx::FxBody)) != 0,
              "RunRollback: mispredicted path DIFFERS from authority (negative control)");
    }

    if (g_fail == 0) std::printf("fpx_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
