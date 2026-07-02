// Slice SB1 — DETERMINISTIC VOLUMETRIC SOFT BODY CORE (engine/sim/softbody.h): Q16.16 PBD volumetric
// lattice — an NxNxN particle cube with STRETCH (axis edges) + SHEAR (face diagonals) + VOLUME
// preservation (per-cell centroid-scale v1) that SQUASHES on impact and RECOVERS its shape —
// bit-identical + lockstep-replayable (Track-S S3). Pure CPU (header-only, no device, no backend
// symbols). Namespace hf::sim::softbody. The ~80% mold is engine/sim/cloth.h (READ-ONLY reuse:
// ClothParticle/IntegrateParticle/SolveDistanceConstraint/CollidePlane/ClothDigest).
//
// What this test PINS:
//   * InitLattice layout (flat x + y*N + z*N*N, exact integer multiples, pinBottom pins the y==0 layer).
//   * BuildSoftConstraints counts (3*N^2*(N-1) stretch, 6*N*(N-1)^2 shear) + hand-enumerated edges +
//     the host-snapped diagonal rest (FxLength(spacing, spacing, 0)).
//   * CellVolume EXACT integer scheme: the unit cube == kOne exactly; translation- and
//     SHEAR-invariant (the signed-tet decomposition is a real volume); a half-height cell == kOne/2.
//   * SolveVolumeConstraint hand-check: the clamped centroid-scale on a half-crushed cell (exact
//     integer positions), kVol==0 EXACT no-op, kVol==kOne/2 half projection, pinned corner holds.
//   * DETERMINISM: two runs byte-identical; net::DigestBytes-over-pos + SoftBodyDigest PINNED (must
//     be identical under MSVC and clang — the cross-compiler proof).
//   * THE PHYSICS: (i) a 4^3 cube slammed into the ground SQUASHES (min height 75% of rest, pinned)
//     then RECOVERS (settled height within a band of rest — the shape-memory proof); (ii) VOLUME:
//     with kVol==kOne the settled total volume error is ~569 LSB (0.26% of V0); with kVol==0 the SAME
//     impact folds cells and the body stays crushed at ~16669 LSB error (7.5% volume LOST) — the
//     volume constraint is LOAD-BEARING (both pinned); (iii) the settled body is at REST (MaxSpeed
//     bound + exact pin).
//   * Identity-at-zero: kVol==0 == the hand-composed pure stretch+shear lattice (bit-check);
//     damp==kOne == the hand-composed raw (undamped) re-encode (the HR1 fxmul-identity convention).
//   * Honest residuals: settled max stretch + shear error and per-cell volume error, pinned.
//   * LOCKSTEP: replica == authority BIT-EXACT from inputs alone; rollback corrects a REAL
//     misprediction to authority BIT-EXACT (negative control: the mispredicted run differed).
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/softbody.h"

#include "net/session.h"   // net::DigestBytes — the spec'd pos-digest currency

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"     // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace softbody = hf::sim::softbody;
namespace cloth    = hf::sim::cloth;
using softbody::fx;
using softbody::kOne;
using softbody::kFrac;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Q16.16 of an integer (exact).
static fx FromInt(int v) { return (fx)(v << kFrac); }

// net::DigestBytes over the packed pos array (x,y,z int32 words per vert, index order) — the spec'd
// determinism pin. Packed into an explicit int32 buffer so the digest is layout-independent.
static uint64_t PosDigest(const std::vector<softbody::SoftVert>& verts) {
    std::vector<int32_t> pos;
    pos.reserve(verts.size() * 3);
    for (const softbody::SoftVert& v : verts) {
        pos.push_back(v.pos.x); pos.push_back(v.pos.y); pos.push_back(v.pos.z);
    }
    return net::DigestBytes(pos.data(), pos.size() * sizeof(int32_t));
}

// The shared hard-impact drop scene (the physics/determinism currency): a free 4^3 cube, spacing 0.5
// (rest extent 1.5), starting 0.25 above the ground with a 28 u/s downward velocity (the tall-drop
// equivalent — ~40 units of free fall compressed into an initial velocity so the test stays fast),
// iters=2 (a soft jelly — deep visible squash), damp 0.875, 600 steps (settled; minH occurs @step 2).
struct DropScene {
    softbody::SoftLattice sl;
    softbody::SoftConstraints sc;
    std::vector<softbody::SoftCell> cells;
    std::vector<softbody::SoftVert> init;
    softbody::SoftParams p;
};
static DropScene MakeDropScene(fx kVol) {
    DropScene d;
    d.sl.N = 4; d.sl.spacing = kOne / 2;
    d.sc = softbody::BuildSoftConstraints(d.sl);
    d.init = softbody::InitLattice(d.sl, softbody::FxVec3{0, kOne / 4, 0}, false);
    for (softbody::SoftVert& v : d.init) v.vel.y = -FromInt(28);
    d.cells = softbody::BuildSoftCells(d.sl, d.init);
    d.p.gravity = {0, (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5)), 0};
    d.p.dt = kOne / 60; d.p.groundY = 0; d.p.iters = 2;
    d.p.kVol = kVol; d.p.damp = kOne - kOne / 8;   // 0.875 (exact Q16.16)
    return d;
}
inline constexpr int kDropSteps = 600;

int main() {
    HF_TEST_MAIN_INIT();

    // ================= InitLattice: flat x-minor layout, exact multiples, pinBottom ===================
    {
        softbody::SoftLattice sl; sl.N = 3; sl.spacing = kOne / 2;
        const softbody::FxVec3 origin{FromInt(1), FromInt(2), FromInt(3)};
        std::vector<softbody::SoftVert> vs = softbody::InitLattice(sl, origin, false);
        check(vs.size() == 27, "InitLattice: size == N^3");
        // Vert (2,1,2): origin + (1.0, 0.5, 1.0) — exact integer multiples of the Q16.16 spacing.
        const softbody::SoftVert& v = vs[(size_t)softbody::VertIndex(sl, 2, 1, 2)];
        check(softbody::VertIndex(sl, 2, 1, 2) == 2 + 1 * 3 + 2 * 9,
              "VertIndex: x + y*N + z*N*N (x-minor)");
        check(v.pos.x == FromInt(2) && v.pos.y == FromInt(2) + kOne / 2 && v.pos.z == FromInt(4),
              "InitLattice: vert (2,1,2) at origin + (x,y,z)*spacing (exact)");
        bool restOk = true; int pinned = 0;
        for (const softbody::SoftVert& p : vs) {
            if (p.flags & softbody::kFlagPinned) ++pinned;
            if (p.invMass != kOne) restOk = false;
            if (std::memcmp(&p.prev, &p.pos, sizeof(softbody::FxVec3)) != 0) restOk = false;
            if (p.vel.x != 0 || p.vel.y != 0 || p.vel.z != 0) restOk = false;
        }
        check(pinned == 0 && restOk, "InitLattice: free body — all dynamic kOne, at rest, prev==pos");
        // pinBottom pins exactly the y==0 layer (N^2 verts, invMass 0).
        std::vector<softbody::SoftVert> vp = softbody::InitLattice(sl, origin, true);
        int pinnedB = 0; bool layerOk = true;
        for (int z = 0; z < 3; ++z)
            for (int y = 0; y < 3; ++y)
                for (int x = 0; x < 3; ++x) {
                    const softbody::SoftVert& q = vp[(size_t)softbody::VertIndex(sl, x, y, z)];
                    const bool pin = (q.flags & softbody::kFlagPinned) != 0;
                    if (pin) { ++pinnedB; if (y != 0 || q.invMass != 0) layerOk = false; }
                }
        check(pinnedB == 9 && layerOk, "InitLattice: pinBottom pins exactly the y==0 layer (N^2)");
    }

    // ================= BuildSoftConstraints: counts + hand-enumerated edges ==========================
    {
        softbody::SoftLattice sl; sl.N = 4; sl.spacing = kOne / 2;
        softbody::SoftConstraints sc = softbody::BuildSoftConstraints(sl);
        check(sc.stretch.size() == 144, "BuildSoftConstraints: 3*N^2*(N-1) stretch edges (N=4: 144)");
        check(sc.shear.size() == 216, "BuildSoftConstraints: 6*N*(N-1)^2 shear diagonals (N=4: 216)");
        // First point (0,0,0): stretch (0,+x)=(0,1), (0,+y)=(0,4), (0,+z)=(0,16), rest = spacing.
        check(sc.stretch[0].i == 0 && sc.stretch[0].j == 1 && sc.stretch[0].restLen == kOne / 2 &&
              sc.stretch[0].kind == softbody::kConstraintStructural,
              "BuildSoftConstraints: stretch (0,0,0)-(1,0,0) rest=spacing kind=structural");
        check(sc.stretch[1].i == 0 && sc.stretch[1].j == 4 && sc.stretch[2].j == 16,
              "BuildSoftConstraints: stretch +y then +z in the fixed emit order");
        // The host-snapped diagonal rest: FxLength(spacing, spacing, 0) = floor(0.5*sqrt(2) * 65536).
        const fx restDiag = softbody::FxLength(softbody::FxVec3{kOne / 2, kOne / 2, 0});
        check(restDiag == 46340, "BuildSoftConstraints: diagonal rest == 46340 LSB (host-snapped)");
        // First shear pair: XY-face diagonals of the (0,0,0) cell face — (0,0,0)-(1,1,0) = (0,5) and
        // (1,0,0)-(0,1,0) = (1,4), both restDiag, kind=shear.
        check(sc.shear[0].i == 0 && sc.shear[0].j == 5 && sc.shear[0].restLen == restDiag &&
              sc.shear[0].kind == softbody::kConstraintShear,
              "BuildSoftConstraints: shear (0,0,0)-(1,1,0) restDiag kind=shear");
        check(sc.shear[1].i == 1 && sc.shear[1].j == 4,
              "BuildSoftConstraints: shear counter-diagonal (1,0,0)-(0,1,0)");
        // Determinism: a second build is byte-identical.
        softbody::SoftConstraints sc2 = softbody::BuildSoftConstraints(sl);
        check(sc.stretch.size() == sc2.stretch.size() && sc.shear.size() == sc2.shear.size() &&
              std::memcmp(sc.stretch.data(), sc2.stretch.data(),
                          sc.stretch.size() * sizeof(softbody::Constraint)) == 0 &&
              std::memcmp(sc.shear.data(), sc2.shear.data(),
                          sc.shear.size() * sizeof(softbody::Constraint)) == 0,
              "BuildSoftConstraints: two builds byte-identical");
        // N=1: no constraints, no cells (degenerate no-op).
        softbody::SoftLattice one; one.N = 1; one.spacing = kOne;
        softbody::SoftConstraints scOne = softbody::BuildSoftConstraints(one);
        std::vector<softbody::SoftVert> vOne = softbody::InitLattice(one, softbody::FxVec3{0, 0, 0});
        check(scOne.stretch.empty() && scOne.shear.empty() &&
              softbody::BuildSoftCells(one, vOne).empty(),
              "BuildSoftConstraints/Cells: N=1 -> no constraints, no cells (degenerate)");
    }

    // ================= CellVolume: the EXACT integer signed-tet decomposition =========================
    {
        // The unit cube (spacing kOne): V == kOne EXACTLY (6 path tets, each det = kOne^3).
        softbody::SoftLattice sl; sl.N = 2; sl.spacing = kOne;
        std::vector<softbody::SoftVert> vs = softbody::InitLattice(sl, softbody::FxVec3{0, 0, 0});
        std::vector<softbody::SoftCell> cells = softbody::BuildSoftCells(sl, vs);
        check(cells.size() == 1, "BuildSoftCells: (N-1)^3 cells (N=2: 1)");
        check(cells[0].restVol == kOne, "CellVolume: the unit cube == kOne EXACTLY");
        // TRANSLATION invariance (the volume is a function of edge deltas alone).
        std::vector<softbody::SoftVert> vt =
            softbody::InitLattice(sl, softbody::FxVec3{FromInt(-7), FromInt(3), FromInt(11)});
        check(softbody::CellVolume(vt, cells[0]) == kOne, "CellVolume: translation-invariant (exact)");
        // SHEAR invariance: slide the whole top face (+y corners) by +x — a sheared parallelepiped
        // has the SAME volume (the signed-tet sum is a real volume, not an AABB).
        std::vector<softbody::SoftVert> vsh = vs;
        for (int k = 0; k < 8; ++k)
            if (k & 2) vsh[(size_t)cells[0].v[k]].pos.x += kOne / 2;
        check(softbody::CellVolume(vsh, cells[0]) == kOne, "CellVolume: SHEAR-invariant (exact)");
        // A half-height cell (top face pushed down to y = 0.5): V == kOne/2 EXACTLY.
        std::vector<softbody::SoftVert> vhalf = vs;
        for (int k = 0; k < 8; ++k)
            if (k & 2) vhalf[(size_t)cells[0].v[k]].pos.y = kOne / 2;
        check(softbody::CellVolume(vhalf, cells[0]) == kOne / 2,
              "CellVolume: half-height cell == kOne/2 EXACTLY");
        // The N=4 spacing-0.5 lattice: 27 cells, each restVol 0.125 (8192 LSB), total 221184.
        softbody::SoftLattice sl4; sl4.N = 4; sl4.spacing = kOne / 2;
        std::vector<softbody::SoftVert> v4 = softbody::InitLattice(sl4, softbody::FxVec3{0, 0, 0});
        std::vector<softbody::SoftCell> c4 = softbody::BuildSoftCells(sl4, v4);
        bool volsOk = c4.size() == 27;
        for (const softbody::SoftCell& c : c4) if (c.restVol != 8192) volsOk = false;
        check(volsOk && softbody::TotalRestVolume(c4) == 221184,
              "BuildSoftCells: 27 cells x 8192 LSB rest volume (spacing 0.5), total 221184");
    }

    // ================= SolveVolumeConstraint: the exact clamped centroid-scale =======================
    {
        // A half-crushed unit cell (V = kOne/2, V0 = kOne): ds = (V0-V)<<16 / (3*V0) = 10922 -> CLAMPED
        // to kVolClamp = kOne/8 = 8192; kVol=kOne -> s = 8192 (the fxmul identity). Centroid of the
        // crushed cell = (32768, 16384, 32768); corner (0,0,0) moves by fxmul(offset, 8192) =
        // (-4096, -2048, -4096) EXACTLY.
        softbody::SoftLattice sl; sl.N = 2; sl.spacing = kOne;
        std::vector<softbody::SoftVert> rest = softbody::InitLattice(sl, softbody::FxVec3{0, 0, 0});
        std::vector<softbody::SoftCell> cells = softbody::BuildSoftCells(sl, rest);
        std::vector<softbody::SoftVert> crushed = rest;
        for (int k = 0; k < 8; ++k)
            if (k & 2) crushed[(size_t)cells[0].v[k]].pos.y = kOne / 2;
        {
            std::vector<softbody::SoftVert> vs = crushed;
            softbody::SolveVolumeConstraint(vs, cells[0], kOne);
            check(vs[0].pos.x == -4096 && vs[0].pos.y == -2048 && vs[0].pos.z == -4096,
                  "SolveVolumeConstraint: corner (0,0,0) -> (-4096, -2048, -4096) EXACT (clamped ds)");
            check(softbody::CellVolume(vs, cells[0]) > softbody::CellVolume(crushed, cells[0]),
                  "SolveVolumeConstraint: the crushed cell EXPANDS toward rest");
        }
        // kVol = kOne/2: s = fxmul(8192, kOne/2) = 4096 — corner (0,0,0) moves exactly half as far.
        {
            std::vector<softbody::SoftVert> vs = crushed;
            softbody::SolveVolumeConstraint(vs, cells[0], kOne / 2);
            check(vs[0].pos.x == -2048 && vs[0].pos.y == -1024 && vs[0].pos.z == -2048,
                  "SolveVolumeConstraint: kVol=kOne/2 halves the projection (exact)");
        }
        // kVol = 0: EXACT no-op (byte-untouched).
        {
            std::vector<softbody::SoftVert> vs = crushed;
            softbody::SolveVolumeConstraint(vs, cells[0], 0);
            check(std::memcmp(vs.data(), crushed.data(), vs.size() * sizeof(softbody::SoftVert)) == 0,
                  "SolveVolumeConstraint: kVol==0 is an EXACT no-op");
        }
        // A pinned corner holds (the inverse-mass discipline).
        {
            std::vector<softbody::SoftVert> vs = crushed;
            vs[0].invMass = 0; vs[0].flags = softbody::kFlagPinned;
            softbody::SolveVolumeConstraint(vs, cells[0], kOne);
            check(vs[0].pos.x == 0 && vs[0].pos.y == 0 && vs[0].pos.z == 0,
                  "SolveVolumeConstraint: a pinned corner never moves");
        }
        // An at-rest cell: V == V0 -> ds == 0 -> EXACT no-op (no drift at rest).
        {
            std::vector<softbody::SoftVert> vs = rest;
            softbody::SolveVolumeConstraint(vs, cells[0], kOne);
            check(std::memcmp(vs.data(), rest.data(), vs.size() * sizeof(softbody::SoftVert)) == 0,
                  "SolveVolumeConstraint: an at-rest cell is an EXACT no-op (no drift)");
        }
        // Out-of-range corner: deterministic no-op (bounds-checked).
        {
            std::vector<softbody::SoftVert> vs = crushed;
            softbody::SoftCell bad = cells[0];
            bad.v[3] = 999u;
            std::vector<softbody::SoftVert> before = vs;
            softbody::SolveVolumeConstraint(vs, bad, kOne);
            check(std::memcmp(vs.data(), before.data(), vs.size() * sizeof(softbody::SoftVert)) == 0,
                  "SolveVolumeConstraint: an out-of-range corner is a deterministic no-op");
            check(softbody::CellVolume(vs, bad) == 0, "CellVolume: out-of-range corner -> 0 (no-op)");
        }
    }

    // ================= DETERMINISM: two runs byte-identical + PINNED digests ==========================
    {
        DropScene d = MakeDropScene(kOne);
        std::vector<softbody::SoftVert> a = d.init, b = d.init;
        softbody::StepSoftBodySteps(d.sl, a, d.sc, d.cells, d.p, kDropSteps);
        softbody::StepSoftBodySteps(d.sl, b, d.sc, d.cells, d.p, kDropSteps);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(softbody::SoftVert)) == 0,
              "determinism: two runs byte-identical");

        const uint64_t dp = PosDigest(a);
        const uint64_t dh = softbody::SoftBodyDigest(a);
        std::printf("SB1 pin: determinism PosDigest = 0x%016llx  SoftBodyDigest = 0x%016llx\n",
                    (unsigned long long)dp, (unsigned long long)dh);
        check(dp == 0xec9f1418f8df9da1ull, "SB1 pin: net::DigestBytes-over-pos == the pinned value");
        check(dh == 0x28d0f723724b6f31ull, "SB1 pin: SoftBodyDigest == the pinned value");
    }

    // ================= THE PHYSICS (i): SQUASH on impact + RECOVER (the shape-memory proof) ===========
    // The hard-impact drop: rest extent (N-1)*spacing = 1.5 (98304 LSB). On impact the body SQUASHES
    // to 75% of its rest height, then the stretch+shear+volume constraints push it back — the settled
    // height recovers to within 1/16 world unit of rest. HONEST: the recovery is not exact (the
    // deterministic PBD residual), the settled height sits 304 LSB (~0.3%) under rest.
    {
        DropScene d = MakeDropScene(kOne);
        std::vector<softbody::SoftVert> vs = d.init;
        const fx restH = (fx)((int64_t)(d.sl.N - 1) * d.sl.spacing);   // 98304 LSB = 1.5 units
        fx minH = INT32_MAX; int minHStep = -1;
        for (int s = 0; s < kDropSteps; ++s) {
            softbody::StepSoftBody(d.sl, vs, d.sc, d.cells, d.p);
            const fx h = softbody::LatticeMaxY(vs) - softbody::LatticeMinY(vs);
            if (h < minH) { minH = h; minHStep = s; }
        }
        const fx endH = softbody::LatticeMaxY(vs) - softbody::LatticeMinY(vs);
        std::printf("SB1 physics(i): restH=%d minH=%d (@step %d, %d%% of rest) settledH=%d "
                    "(%d LSB under rest)\n",
                    restH, minH, minHStep, (int)((int64_t)minH * 100 / restH), endH, restH - endH);
        check(minH < restH - restH / 8,
              "physics(i): the body SQUASHES on impact (min height < 7/8 of rest)");
        check(endH > restH - kOne / 16 && endH < restH + kOne / 16,
              "physics(i): the settled height RECOVERS to within 1/16 unit of rest (shape memory)");
        check(minH == 73849 && minHStep == 2 && endH == 98000,
              "SB1 pin physics(i): minH == 73849 @step 2, settled height == 98000 (exact)");

        // (iii) the settled body is at REST: MaxSpeed bounded + pinned. HONEST: the volume pass +
        // ground clamp leave a deterministic micro-jitter (~0.04 u/s), an order below any visible
        // motion but NOT zero — reported, not hidden.
        const fx spd = softbody::MaxSpeed(vs);
        std::printf("SB1 physics(iii): settled MaxSpeed = %d LSB (~%d/65536 u/s)\n", spd, spd);
        check(spd < kOne / 16, "physics(iii): settled max speed < 1/16 u/s (at rest)");
        check(spd == 2564, "SB1 pin physics(iii): settled MaxSpeed == 2564 LSB (exact)");

        // (e) honest residuals at rest, pinned.
        const fx strErr = softbody::MaxDistanceError(vs, d.sc.stretch);
        const fx shrErr = softbody::MaxDistanceError(vs, d.sc.shear);
        const int64_t cellErr = softbody::MaxCellVolumeError(vs, d.cells);
        std::printf("SB1 residuals: maxStretchErr=%d maxShearErr=%d maxCellVolErr=%lld LSB\n",
                    strErr, shrErr, (long long)cellErr);
        check(strErr <= kOne / 64, "residuals: settled max stretch error <= 1/64 world unit");
        check(strErr == 294 && shrErr == 201 && cellErr == 65,
              "SB1 pin residuals: stretch/shear/cell-volume == 294/201/65 (exact)");
    }

    // ================= THE PHYSICS (ii): VOLUME preservation is LOAD-BEARING =========================
    // The SAME hard impact with kVol==kOne vs kVol==0. With the volume constraint the body recovers
    // to 99.7% of its rest volume; WITHOUT it the impact FOLDS cells through themselves (the
    // distance-constraint mirror ambiguity — a folded cell satisfies every stretch/shear length) and
    // the body stays permanently crushed at 92.5% (7.5% of the volume LOST). Both pinned — the
    // volume constraint is what makes the jelly recover instead of pancaking.
    {
        DropScene on = MakeDropScene(kOne);
        DropScene off = MakeDropScene(0);
        std::vector<softbody::SoftVert> von = on.init, voff = off.init;
        softbody::StepSoftBodySteps(on.sl, von, on.sc, on.cells, on.p, kDropSteps);
        softbody::StepSoftBodySteps(off.sl, voff, off.sc, off.cells, off.p, kDropSteps);
        const int64_t restV = softbody::TotalRestVolume(on.cells);
        int64_t errOn  = softbody::TotalVolume(von, on.cells) - restV;
        int64_t errOff = softbody::TotalVolume(voff, off.cells) - restV;
        if (errOn < 0) errOn = -errOn;
        if (errOff < 0) errOff = -errOff;
        std::printf("SB1 physics(ii): restV=%lld |volErr| kVol=kOne: %lld (%lld%%o)  kVol=0: %lld "
                    "(%lld%%o)\n", (long long)restV, (long long)errOn,
                    (long long)(errOn * 1000 / restV), (long long)errOff,
                    (long long)(errOff * 1000 / restV));
        check(errOn < restV / 128,
              "physics(ii): kVol==kOne settled volume within 1/128 of V0 (preserved)");
        check(errOff > restV / 16,
              "physics(ii): kVol==0 loses > 1/16 of V0 on the same impact (MEASURABLY worse)");
        check(errOff > 8 * errOn, "physics(ii): the volume constraint is LOAD-BEARING (>8x contrast)");
        check(errOn == 569 && errOff == 16669,
              "SB1 pin physics(ii): |volErr| == 569 (on) vs 16669 (off) LSB (exact)");
    }

    // ================= Identity-at-zero: kVol=0 == the pure stretch+shear lattice ====================
    // StepSoftBody with kVol==0 must be BIT-IDENTICAL to the hand-composed pure stretch+shear step
    // (integrate -> iters x (stretch -> shear -> ground) -> ground clamp -> damped re-encode) built
    // from the cloth.h primitives directly — the off-switch contract. AND damp==kOne must equal the
    // hand-composed RAW re-encode (no damp multiply) — the HR1 fxmul-identity convention.
    {
        DropScene d = MakeDropScene(0);
        const int steps = 90;
        std::vector<softbody::SoftVert> viaStep = d.init;
        softbody::StepSoftBodySteps(d.sl, viaStep, d.sc, d.cells, d.p, steps);

        std::vector<softbody::SoftVert> pure = d.init;
        for (int s = 0; s < steps; ++s) {
            for (size_t i = 0; i < pure.size(); ++i)
                cloth::IntegrateParticle(pure[i], d.p.gravity, d.p.groundY, d.p.dt);
            for (int it = 0; it < d.p.iters; ++it) {
                for (size_t e = 0; e < d.sc.stretch.size(); ++e)
                    cloth::SolveDistanceConstraint(pure, d.sc.stretch[e]);
                for (size_t e = 0; e < d.sc.shear.size(); ++e)
                    cloth::SolveDistanceConstraint(pure, d.sc.shear[e]);
                cloth::CollidePlane(pure, d.p.groundY);
            }
            cloth::CollidePlane(pure, d.p.groundY);
            for (size_t i = 0; i < pure.size(); ++i) {
                softbody::SoftVert& v = pure[i];
                if (v.flags & softbody::kFlagPinned) continue;
                v.vel.x = softbody::fxmul(softbody::fxdiv(v.pos.x - v.prev.x, d.p.dt), d.p.damp);
                v.vel.y = softbody::fxmul(softbody::fxdiv(v.pos.y - v.prev.y, d.p.dt), d.p.damp);
                v.vel.z = softbody::fxmul(softbody::fxdiv(v.pos.z - v.prev.z, d.p.dt), d.p.damp);
            }
        }
        check(viaStep.size() == pure.size() &&
              std::memcmp(viaStep.data(), pure.data(), pure.size() * sizeof(softbody::SoftVert)) == 0,
              "identity-at-zero: kVol=0 StepSoftBody == the pure stretch+shear lattice BIT-EXACT");

        // damp == kOne == the raw (no-multiply) re-encode, bit-exact (fxmul(v, kOne) == v).
        softbody::SoftParams pOne = d.p; pOne.damp = kOne;
        std::vector<softbody::SoftVert> viaOne = d.init;
        softbody::StepSoftBodySteps(d.sl, viaOne, d.sc, d.cells, pOne, 30);
        std::vector<softbody::SoftVert> raw = d.init;
        for (int s = 0; s < 30; ++s) {
            for (size_t i = 0; i < raw.size(); ++i)
                cloth::IntegrateParticle(raw[i], d.p.gravity, d.p.groundY, d.p.dt);
            for (int it = 0; it < d.p.iters; ++it) {
                for (size_t e = 0; e < d.sc.stretch.size(); ++e)
                    cloth::SolveDistanceConstraint(raw, d.sc.stretch[e]);
                for (size_t e = 0; e < d.sc.shear.size(); ++e)
                    cloth::SolveDistanceConstraint(raw, d.sc.shear[e]);
                cloth::CollidePlane(raw, d.p.groundY);
            }
            cloth::CollidePlane(raw, d.p.groundY);
            for (size_t i = 0; i < raw.size(); ++i) {
                softbody::SoftVert& v = raw[i];
                if (v.flags & softbody::kFlagPinned) continue;
                v.vel.x = softbody::fxdiv(v.pos.x - v.prev.x, d.p.dt);   // RAW — no damp multiply
                v.vel.y = softbody::fxdiv(v.pos.y - v.prev.y, d.p.dt);
                v.vel.z = softbody::fxdiv(v.pos.z - v.prev.z, d.p.dt);
            }
        }
        check(viaOne.size() == raw.size() &&
              std::memcmp(viaOne.data(), raw.data(), raw.size() * sizeof(softbody::SoftVert)) == 0,
              "identity-at-zero: damp==kOne == the raw re-encode BIT-EXACT (the HR1 convention)");
    }

    // ================= LOCKSTEP: replica == authority; rollback corrects a real divergence ============
    // An anchored jelly block (pinBottom — the y==0 layer holds) poked by an impulse command stream:
    // the peer re-derives the exact wobble from the inputs alone; a mispredicted (wrong) impulse is
    // rolled back and corrected to authority bit-for-bit.
    {
        softbody::SoftLattice sl; sl.N = 3; sl.spacing = kOne / 2;
        const softbody::SoftConstraints sc = softbody::BuildSoftConstraints(sl);
        const std::vector<softbody::SoftVert> init =
            softbody::InitLattice(sl, softbody::FxVec3{0, kOne, 0}, true);   // anchored above ground
        const std::vector<softbody::SoftCell> cells = softbody::BuildSoftCells(sl, init);
        softbody::SoftParams p;
        p.gravity = {0, (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5)), 0};
        p.dt = kOne / 60; p.groundY = 0; p.iters = 4;
        p.kVol = kOne / 2; p.damp = kOne - kOne / 16;
        const int ticks = 24, mispredictTick = 8;
        const uint32_t topCorner = (uint32_t)softbody::VertIndex(sl, 2, 2, 2);
        const uint32_t topMid    = (uint32_t)softbody::VertIndex(sl, 1, 2, 1);

        // The authoritative IMPULSE stream (the netcode input): two pokes on the top layer.
        const std::vector<softbody::SoftCommand> authStream{
            softbody::SoftCommand{3,  softbody::kCmdImpulse, topCorner,
                                  softbody::FxVec3{FromInt(2), 0, 0}},
            softbody::SoftCommand{14, softbody::kCmdImpulse, topMid,
                                  softbody::FxVec3{0, FromInt(-1), FromInt(1)}},
        };
        // The MISPREDICTED stream: auth + a WRONG large poke at mispredictTick (a real divergence).
        std::vector<softbody::SoftCommand> mispredictStream = authStream;
        mispredictStream.push_back(softbody::SoftCommand{(uint32_t)mispredictTick,
                                                         softbody::kCmdImpulse, topCorner,
                                                         softbody::FxVec3{0, FromInt(4), 0}});

        const std::vector<softbody::SoftVert> authority =
            softbody::RunSoftBodyLockstep(sl, init, sc, cells, p, authStream, ticks);
        const std::vector<softbody::SoftVert> replica =
            softbody::RunSoftBodyLockstep(sl, init, sc, cells, p, authStream, ticks);
        check(authority.size() == replica.size() &&
              std::memcmp(authority.data(), replica.data(),
                          authority.size() * sizeof(softbody::SoftVert)) == 0,
              "lockstep: replica == authority BIT-EXACT (inputs-only re-sim)");

        const std::vector<softbody::SoftVert> rolledBack =
            softbody::RunSoftBodyRollback(sl, init, sc, cells, p, authStream, mispredictStream,
                                          ticks, mispredictTick);
        check(rolledBack.size() == authority.size() &&
              std::memcmp(rolledBack.data(), authority.data(),
                          authority.size() * sizeof(softbody::SoftVert)) == 0,
              "rollback: corrected to authority BIT-EXACT (positive control)");

        const std::vector<softbody::SoftVert> mispredicted =
            softbody::RunSoftBodyLockstep(sl, init, sc, cells, p, mispredictStream, ticks);
        check(mispredicted.size() == authority.size() &&
              std::memcmp(mispredicted.data(), authority.data(),
                          authority.size() * sizeof(softbody::SoftVert)) != 0,
              "rollback: the mispredicted run DIFFERS from authority (the divergence was real)");

        // The impulse genuinely moved the body (not a no-op) + OOB target / pinned target no-ops.
        bool moved = false;
        for (size_t i = 0; i < authority.size(); ++i)
            if (!(authority[i].flags & softbody::kFlagPinned) &&
                std::memcmp(&authority[i].pos, &init[i].pos, sizeof(softbody::FxVec3)) != 0)
                moved = true;
        check(moved, "lockstep: the impulse stream genuinely moved the body");
        std::vector<softbody::SoftVert> oob = init;
        softbody::ApplySoftCommand(oob, softbody::SoftCommand{0, softbody::kCmdImpulse, 9999u,
                                                              softbody::FxVec3{kOne, 0, 0}});
        softbody::ApplySoftCommand(oob, softbody::SoftCommand{0, softbody::kCmdImpulse, 0u,
                                                              softbody::FxVec3{kOne, 0, 0}});
        check(std::memcmp(oob.data(), init.data(), init.size() * sizeof(softbody::SoftVert)) == 0,
              "lockstep: out-of-range + pinned-target commands are deterministic no-ops");

        const uint64_t d = softbody::SoftBodyDigest(authority);
        std::printf("SB1 pin: lockstep authority SoftBodyDigest = 0x%016llx\n", (unsigned long long)d);
        check(d == 0x8af74d2afde4fa08ull, "SB1 pin: lockstep authority digest == the pinned value");
    }

    if (g_fail == 0) std::printf("softbody_test: ALL PASS\n");
    else std::printf("softbody_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
