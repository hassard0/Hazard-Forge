#pragma once
// Slice SB1 — DETERMINISTIC VOLUMETRIC SOFT BODY CORE (Track-S slice S3 of
// docs/SUPERIORITY_ROADMAP.md): Q16.16 fixed-point PBD volumetric LATTICE — an NxNxN particle cube with
// STRETCH (axis edges) + SHEAR (face diagonals) + VOLUME-PRESERVATION (per-cell) constraints that
// SQUASHES on impact and RECOVERS its shape (jelly/flesh/rubber) — BIT-IDENTICAL on every platform AND
// lockstep/rollback-replayable. This is the deformable-VOLUME material family the engine was missing
// (rigid fpx / cloth / fluid / grain / fracture / hair are all shipped; the tet code referenced in
// ARCHITECTURE is the GJK narrowphase + inertia decomposition, NOT a deformable sim). UE5 has no
// deterministic soft body (Chaos Flesh is float / experimental / non-deterministic), so a deterministic
// volumetric soft body is a new headline. Namespace hf::sim::softbody, header-only, pure CPU, NO
// device / backend symbols, NO float on the sim path, NO RNG, NO clock.
//
// THE ~80% MOLD-REUSE (engine/sim/cloth.h #included READ-ONLY, byte-UNTOUCHED — softbody is the
// additive sibling; a soft cube is cloth's 3D generalization + volume):
//   * STATE     : SoftVert IS cloth::ClothParticle verbatim (pos/prev/vel/invMass/flags, the CL1
//                 std430-packable 44-byte POD). Flat index x + y*N + z*N*N.
//   * INTEGRATE : cloth::IntegrateParticle AS-IS (the CL1 semi-implicit Euler + prev snapshot + ground
//                 floor-clamp; the prev snapshot is the PBD/Verlet anchor the re-encode below reads).
//   * STRETCH   : cloth::Constraint + cloth::SolveDistanceConstraint AS-IS (the CL3 PBD projection,
//                 pinned share 0) over the 6-neighbourhood axis edges (rest = spacing).
//   * SHEAR     : the SAME CL3 projection over the FACE DIAGONALS (both diagonals of every lattice
//                 face, rest = FxLength(spacing, spacing, 0) — a build-time host-snapped constant).
//   * GROUND    : cloth::CollidePlane AS-IS (the CL4 y >= groundY clamp; frictionless — a clamped vert
//                 keeps its horizontal position only through the constraint graph, the FL/CL mold).
//   * DIGEST    : cloth::ClothDigest AS-IS (SoftVert == ClothParticle) — the golden currency.
//   * LOCKSTEP  : the CL5/FPX5/HR1 command+snapshot mold (SoftCommand impulse stream; snapshot = the
//                 deep vector copy; RunSoftBodyLockstep / RunSoftBodyRollback are the RunClothLockstep /
//                 RunClothRollback twins).
//
// THE NEW PHYSICS vs cloth/hair — the per-cell VOLUME-PRESERVATION constraint (the exact integer
// scheme, documented):
//   MEASURE: each lattice CELL (the cube of 8 particles) has a signed volume V computed over a FIXED
//   6-tetrahedron decomposition around the c000->c111 main diagonal (the local corner-bit order is
//   bit0=+x, bit1=+y, bit2=+z; the six path tets, odd permutations vertex-swapped so every tet of the
//   rest cube is POSITIVE, are {0,1,3,7} {0,3,2,7} {0,2,6,7} {0,6,4,7} {0,4,5,7} {0,5,1,7}). Each
//   tet's 6*volume is the int64 determinant det(b-a, c-a, d-a) of Q16.16 edge deltas — Q48.48 in an
//   int64 (the SAME signed-tet primitive shape as the GJK/inertia decomposition, made a deformable
//   constraint). V = (sum_of_dets / 6) >> 32 back to Q16.16 (truncating /6 toward zero, then an
//   arithmetic shift — one fixed op order, deterministic). INT64 BOUND (documented): an edge delta d
//   contributes |cross| <= 2*d^2 and |det| <= 6*d^3; d < 16 world units (2^20 LSB) keeps every
//   intermediate < 2^63 — cells are lattice-local (~spacing), far inside the bound.
//
//   PROJECT — THE SHIPPED v1 SCHEME IS THE SIMPLIFIED ISOTROPIC CENTROID-SCALE CORRECTION, NOT the
//   true per-particle volume gradient (the spec'd acceptable v1, documented honestly): the true PBD
//   volume gradient (grad_i V over the tet fan) needs per-particle int64 cross products PLUS a
//   normalization by the sum of squared gradient magnitudes — an int64/int128-heavy divide chain in
//   integer. v1 instead scales the 8 corners isotropically toward/away from the cell CENTROID:
//       V  = CellVolume(cell);  V0 = cell.restVol
//       ds = clamp( ((V0 - V) << kFrac) / (3 * V0), -kOne/8, +kOne/8 )   // the linearized (V0/V)^(1/3)
//       s  = fxmul((fx)ds, kVol)                                          // the Q16.16 stiffness scale
//       centroid = (sum of the 8 corner pos) >> 3 per axis (int64 sum, arithmetic shift)
//       for each corner (FIXED order 0..7): if pinned skip; pos += FxScale(pos - centroid, s)
//   ds is the first-order cube-root linearization ( (1+e)^(1/3) ~ 1 + e/3 with e = (V0-V)/V0 ): an
//   isotropic scale by (1+s) multiplies the volume by (1+s)^3 ~ 1 + 3s, so s = (V0-V)/(3*V0) restores
//   V -> V0 to first order. The denominator is the CONSTANT rest volume (not the current V), so a
//   crushed/inverted cell (V <= 0) still gets a bounded, well-defined EXPANSION push (no divide-by-
//   small-V blowup); the +-kOne/8 clamp bounds the per-application correction (overshoot control).
//   BEHAVIOR DIFFERENCE vs the true gradient (documented): the correction is ISOTROPIC — a cell
//   flattened along y re-inflates equally in x/y/z rather than preferentially along the compressed
//   axis; the composed stretch+shear projections supply the missing shape anisotropy, so the COMPOSITE
//   still recovers the cube. kVol == 0 -> the volume pass is SKIPPED entirely (identity-at-zero: a
//   zero-stiffness body is bit-identical to the pure stretch+shear lattice); kVol == kOne applies the
//   full clamped ds (fxmul(ds, kOne) == ds EXACTLY). Truncating integer division/fxmul give a
//   deterministic sub-LSB dead zone (a tiny |V0-V| rounds ds or s to 0 — the residual is NOT zero,
//   the tests PIN the real numbers).
//
//   HONEST CAVEATS (the CL3/FPX3 iterative-PBD reality): (1) the correction is a per-iteration
//   FRACTIONAL projection — it saturates like every PBD stiffness (effective stiffness grows with
//   iters, never rigid); (2) volume is enforced PER CELL by a clamped linearization, so large fast
//   compressions are recovered over multiple steps, not instantly, and the settled total volume
//   carries a deterministic nonzero residual (pinned by the tests, reported honestly); (3) the
//   centroid-scale scheme can trade volume error between neighbouring cells (a shared face moved by
//   one cell changes the next cell's V) — the Gauss-Seidel cell order is FIXED (ascending flat cell
//   index) so the trade is deterministic.
//
// THE STEP (StepSoftBody) — the HR1/FL7 velocity RE-ENCODE discipline: after the constraint passes the
// velocity is RE-DERIVED from the net position change, v = (pos - prev) / dt (per-axis fxdiv;
// fxdiv(x, 0) == 0 by the fpx contract), then scaled by the Q16.16 damp factor (damp == kOne is the
// EXACT identity — fxmul(v, kOne) == v for every int32). This is what lets the dropped cube SETTLE.
//   1. gravity integrate every vert (cloth::IntegrateParticle — pinned untouched)
//   2. `iters` rounds of (STRETCH Gauss-Seidel pass -> SHEAR Gauss-Seidel pass -> VOLUME per-cell pass
//      (skipped entirely when kVol <= 0) -> ground clamp), each list in its FIXED build order (the
//      CL3/HR1 sequential fixed-order contract — deterministic; a GPU port would need Jacobi
//      colouring, out of scope for the pure-CPU SB1)
//   3. ground clamp AFTER the loop (idempotent; covers iters == 0)
//   4. velocity re-encode: vel = fxmul((pos - prev) / dt, damp) per axis, non-pinned verts only
// Pure integer, fixed op order -> two runs byte-identical AND bit-identical on every platform.
//
// THE CPU/GPU CHOICE (documented, the CF1/FR8/HR1 precedent): SB1 ships PURE CPU on BOTH backends —
// the stretch/shear projections are int64-backed (FxLength/FxNormalize/fxdiv) and the volume measure
// is int64 determinants, which under the house convention would make any GPU kernel DXC/Vulkan-only
// with a Metal CPU reference anyway; CF1/FR8/HR1 established that a pure-CPU slice run IDENTICALLY on
// Vulkan-Windows and Metal-Mac is cross-backend bit-identical BY CONSTRUCTION and that this is an
// acceptable proof shape. NO new shader, NO new RHI; a GPU kernel is a future refinement, not SB1.
//
// BOUNDS: every constraint endpoint / cell corner / command target is range-checked (deterministic
// no-op on out-of-range); lattice sizes are showcase-scale (a few hundred verts) so all int32 sums
// stay far below 2^31 and every int64 volume intermediate far below 2^63 (the bound above).

#include <cstdint>
#include <vector>

#include "sim/cloth.h"   // READ-ONLY reuse: ClothParticle/Constraint/IntegrateParticle/
                         // SolveDistanceConstraint/CollidePlane/ClothDigest + the fpx Q16.16 toolbox

namespace hf::sim {
namespace softbody {

// The fpx Q16.16 toolbox through the cloth re-exports (NO new fixed-point primitives).
using cloth::fx;
using cloth::FxVec3;
using cloth::fxmul;
using cloth::fxdiv;
using cloth::FxAdd;
using cloth::FxSub;
using cloth::FxScale;
using cloth::FxLength;
using cloth::FxNormalize;
inline constexpr int kFrac = cloth::kFrac;
inline constexpr fx  kOne  = cloth::kOne;

// ----- The soft-body state: SoftVert IS the CL1 particle (the mold reused verbatim) -----------------
using SoftVert = cloth::ClothParticle;
using cloth::kFlagPinned;
using cloth::Constraint;
using cloth::kConstraintStructural;   // reused as the STRETCH kind (axis edges, rest = spacing)
using cloth::kConstraintShear;        // reused as the SHEAR kind (face diagonals, rest = spacing*sqrt2)

// The lattice layout: N particles per axis (N^3 total), spacing the Q16.16 rest edge length.
struct SoftLattice {
    int N = 0;         // particles per axis (>= 2 for any constraint/cell)
    fx  spacing = 0;   // Q16.16 rest distance between axis neighbours
};

// VertIndex(sl, x, y, z): the flat index of lattice point (x, y, z) — x + y*N + z*N*N (x-minor).
inline int VertIndex(const SoftLattice& sl, int x, int y, int z) {
    return x + y * sl.N + z * sl.N * sl.N;
}

// ----- InitLattice: the deterministic NxNxN rest cube ----------------------------------------------
// Particle (x,y,z) sits at origin + (x,y,z)*spacing (exact integer multiples of the host-snapped
// Q16.16 spacing — no float). prev == pos, vel == 0 at rest. pinBottom pins the whole y==0 layer
// (invMass 0, kFlagPinned — an anchored jelly block); pinBottom == false leaves EVERY vert dynamic
// with invMass = kOne (the free-fall drop scene). Returns the populated N^3 vert vector.
inline std::vector<SoftVert> InitLattice(const SoftLattice& sl, const FxVec3& origin,
                                         bool pinBottom = false) {
    std::vector<SoftVert> verts((size_t)(sl.N * sl.N * sl.N));
    for (int z = 0; z < sl.N; ++z)
        for (int y = 0; y < sl.N; ++y)
            for (int x = 0; x < sl.N; ++x) {
                SoftVert v;
                v.pos = FxVec3{origin.x + (fx)((int64_t)x * (int64_t)sl.spacing),
                               origin.y + (fx)((int64_t)y * (int64_t)sl.spacing),
                               origin.z + (fx)((int64_t)z * (int64_t)sl.spacing)};
                v.prev = v.pos;
                v.vel = FxVec3{0, 0, 0};
                if (pinBottom && y == 0) { v.invMass = 0;    v.flags = kFlagPinned; }
                else                     { v.invMass = kOne; v.flags = 0; }
                verts[(size_t)VertIndex(sl, x, y, z)] = v;
            }
    return verts;
}

// ----- BuildSoftConstraints: STRETCH (axis edges) + SHEAR (face diagonals), fixed order -------------
// Per lattice point (z,y,x ascending — the flat-index order), emit the FORWARD in-bounds edges:
//   STRETCH: (p, +x), (p, +y), (p, +z)                       rest = spacing        (3*N^2*(N-1) total)
//   SHEAR  : both diagonals of the three forward faces —
//            XY face: (x,y,z)-(x+1,y+1,z)   and (x+1,y,z)-(x,y+1,z)
//            XZ face: (x,y,z)-(x+1,y,z+1)   and (x+1,y,z)-(x,y,z+1)
//            YZ face: (x,y,z)-(x,y+1,z+1)   and (x,y+1,z)-(x,y,z+1)
//            rest = FxLength(spacing, spacing, 0) — ONE host-snapped build-time constant (the flat
//            lattice makes every face diagonal congruent), 6*N*(N-1)^2 total.
// Each unordered edge is emitted EXACTLY ONCE (all six diagonals above are owned by their minimal
// (x,y,z) face corner and both endpoints are in ascending flat-index i<j order). Two SEPARATE lists so
// the solver's stretch pass and shear pass are cleanly ordered. Deterministic (pure index arithmetic).
struct SoftConstraints {
    std::vector<Constraint> stretch;   // axis edges, rest = spacing — the CL3 distance chain in 3D
    std::vector<Constraint> shear;     // face diagonals, rest = spacing*sqrt(2) (host-snapped FxLength)
};

inline SoftConstraints BuildSoftConstraints(const SoftLattice& sl) {
    SoftConstraints sc;
    const int N = sl.N;
    if (N >= 2) {
        sc.stretch.reserve((size_t)(3 * N * N * (N - 1)));
        sc.shear.reserve((size_t)(6 * N * (N - 1) * (N - 1)));
    }
    // The ONE host-snapped diagonal rest length (every lattice face is congruent at rest).
    const fx restDiag = FxLength(FxVec3{sl.spacing, sl.spacing, 0});
    auto idx = [&](int x, int y, int z) { return (uint32_t)VertIndex(sl, x, y, z); };
    for (int z = 0; z < N; ++z)
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x) {
                // STRETCH: the three forward axis edges.
                if (x + 1 < N) sc.stretch.push_back(Constraint{idx(x, y, z), idx(x + 1, y, z),
                                                               sl.spacing, kConstraintStructural});
                if (y + 1 < N) sc.stretch.push_back(Constraint{idx(x, y, z), idx(x, y + 1, z),
                                                               sl.spacing, kConstraintStructural});
                if (z + 1 < N) sc.stretch.push_back(Constraint{idx(x, y, z), idx(x, y, z + 1),
                                                               sl.spacing, kConstraintStructural});
                // SHEAR: both diagonals of the three forward faces (fixed emit order).
                if (x + 1 < N && y + 1 < N) {
                    sc.shear.push_back(Constraint{idx(x, y, z), idx(x + 1, y + 1, z),
                                                  restDiag, kConstraintShear});
                    sc.shear.push_back(Constraint{idx(x + 1, y, z), idx(x, y + 1, z),
                                                  restDiag, kConstraintShear});
                }
                if (x + 1 < N && z + 1 < N) {
                    sc.shear.push_back(Constraint{idx(x, y, z), idx(x + 1, y, z + 1),
                                                  restDiag, kConstraintShear});
                    sc.shear.push_back(Constraint{idx(x + 1, y, z), idx(x, y, z + 1),
                                                  restDiag, kConstraintShear});
                }
                if (y + 1 < N && z + 1 < N) {
                    sc.shear.push_back(Constraint{idx(x, y, z), idx(x, y + 1, z + 1),
                                                  restDiag, kConstraintShear});
                    sc.shear.push_back(Constraint{idx(x, y + 1, z), idx(x, y, z + 1),
                                                  restDiag, kConstraintShear});
                }
            }
    return sc;
}

// SolveDistanceList(verts, list): one Gauss-Seidel pass over a constraint list in its FIXED build
// order — cloth::SolveDistanceConstraint per edge, with the SB1 bounds check in front (cloth's CL3
// solver indexes unchecked; softbody range-checks every endpoint, deterministic skip on out-of-range).
inline void SolveDistanceList(std::vector<SoftVert>& verts, const std::vector<Constraint>& list) {
    const size_t n = verts.size();
    for (size_t e = 0; e < list.size(); ++e) {
        const Constraint& c = list[e];
        if (c.i >= n || c.j >= n) continue;               // bounds-checked skip
        cloth::SolveDistanceConstraint(verts, c);
    }
}

// ----- The volume cells: the cube of 8 particles + its host-snapped rest volume ---------------------
// Corner order is the FIXED local bit code k in [0,8): bit0 = +x, bit1 = +y, bit2 = +z (c[0] = the
// cell's minimal corner, c[7] = the main-diagonal opposite). restVol is the Q16.16 cell volume at
// build (int64 — CellVolume of the rest lattice, host-snapped).
struct SoftCell {
    uint32_t v[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // corner vert indices in the fixed bit order
    int64_t  restVol = 0;                       // Q16.16 rest volume (host-snapped at build)
};

// Det3_48(a, b, c): the int64 determinant of three Q16.16 edge-delta columns — 6x the signed tet
// volume, in Q48.48. Every product is int64; the documented bound (|delta| < 16 world units) keeps
// the cross terms < 2^39 and the full det < 2^60 (cells are lattice-local, far inside it).
inline int64_t Det3_48(const FxVec3& a, const FxVec3& b, const FxVec3& c) {
    const int64_t cx = (int64_t)a.y * (int64_t)b.z - (int64_t)a.z * (int64_t)b.y;   // Q32.32
    const int64_t cy = (int64_t)a.z * (int64_t)b.x - (int64_t)a.x * (int64_t)b.z;
    const int64_t cz = (int64_t)a.x * (int64_t)b.y - (int64_t)a.y * (int64_t)b.x;
    return cx * (int64_t)c.x + cy * (int64_t)c.y + cz * (int64_t)c.z;               // Q48.48
}

// kCellTets: the FIXED 6-tet decomposition of a cell around the c0->c7 main diagonal (the header
// banner scheme — path tets over the axis permutations, odd permutations vertex-swapped so every tet
// of the REST cube is positive). Each row is (a, b, c, d) local corner codes; 6*V_tet =
// Det3_48(b-a, c-a, d-a).
inline constexpr int kCellTets[6][4] = {
    {0, 1, 3, 7}, {0, 3, 2, 7}, {0, 2, 6, 7}, {0, 6, 4, 7}, {0, 4, 5, 7}, {0, 5, 1, 7},
};

// CellVolume(verts, cell): the cell's signed volume in Q16.16 (int64). Sums the six tet determinants
// (Q48.48), divides by 6 (truncating toward zero), then arithmetic->>32 back to Q16.16 — ONE fixed op
// order, deterministic. Out-of-range corner -> 0 (bounds-checked, deterministic no-op).
inline int64_t CellVolume(const std::vector<SoftVert>& verts, const SoftCell& cell) {
    const size_t n = verts.size();
    for (int k = 0; k < 8; ++k)
        if (cell.v[k] >= n) return 0;                     // bounds-checked skip
    int64_t det6 = 0;                                     // sum of 6*V_tet, Q48.48
    for (int t = 0; t < 6; ++t) {
        const FxVec3& pa = verts[(size_t)cell.v[kCellTets[t][0]]].pos;
        const FxVec3& pb = verts[(size_t)cell.v[kCellTets[t][1]]].pos;
        const FxVec3& pc = verts[(size_t)cell.v[kCellTets[t][2]]].pos;
        const FxVec3& pd = verts[(size_t)cell.v[kCellTets[t][3]]].pos;
        det6 += Det3_48(FxSub(pb, pa), FxSub(pc, pa), FxSub(pd, pa));
    }
    return (det6 / 6) >> 32;                              // Q48.48 -> Q16.16 (truncate /6, then floor)
}

// BuildSoftCells(sl, verts): the (N-1)^3 cells in ascending flat cell order (cz, cy, cx ascending —
// the fixed Gauss-Seidel volume-pass order), each with restVol = CellVolume of the REST lattice
// (host-snapped at build, the BuildConstraints FxLength-at-build discipline).
inline std::vector<SoftCell> BuildSoftCells(const SoftLattice& sl,
                                            const std::vector<SoftVert>& verts) {
    std::vector<SoftCell> cells;
    const int C = sl.N - 1;
    if (C < 1) return cells;
    cells.reserve((size_t)(C * C * C));
    for (int cz = 0; cz < C; ++cz)
        for (int cy = 0; cy < C; ++cy)
            for (int cx = 0; cx < C; ++cx) {
                SoftCell cell;
                for (int k = 0; k < 8; ++k)
                    cell.v[k] = (uint32_t)VertIndex(sl, cx + (k & 1), cy + ((k >> 1) & 1),
                                                    cz + ((k >> 2) & 1));
                cell.restVol = CellVolume(verts, cell);
                cells.push_back(cell);
            }
    return cells;
}

// kVolClamp: the per-application bound on the linearized volume scale ds (+-kOne/8 = +-12.5% per
// projection) — the overshoot/oscillation control documented in the header banner.
inline constexpr fx kVolClamp = kOne / 8;

// ----- SolveVolumeConstraint: the v1 isotropic CENTROID-SCALE volume projection (the new physics) ----
// The exact integer scheme from the header banner: V = CellVolume; ds = clamp(((V0 - V) << kFrac) /
// (3*V0), +-kVolClamp) — the first-order (V0/V)^(1/3) linearization against the CONSTANT rest volume
// (no divide-by-current-V blowup; an inverted cell still gets a bounded expansion push); s =
// fxmul(ds, kVol); every corner (FIXED order 0..7, pinned skipped) moves pos += (pos - centroid) * s,
// centroid = (sum of the 8 corner pos) >> 3 per axis. kVol <= 0 -> EXACT no-op (identity-at-zero);
// kVol == kOne -> the full clamped ds (fxmul identity). Truncation gives a deterministic sub-LSB dead
// zone. Bounds-checked (out-of-range corner -> deterministic no-op).
inline void SolveVolumeConstraint(std::vector<SoftVert>& verts, const SoftCell& cell, fx kVol) {
    if (kVol <= 0) return;                                // identity-at-zero: EXACT no-op
    const size_t n = verts.size();
    for (int k = 0; k < 8; ++k)
        if (cell.v[k] >= n) return;                       // bounds-checked skip
    if (cell.restVol <= 0) return;                        // degenerate rest cell -> skip
    const int64_t V = CellVolume(verts, cell);
    int64_t ds64 = ((cell.restVol - V) << kFrac) / (3 * cell.restVol);   // truncating toward zero
    if (ds64 >  (int64_t)kVolClamp) ds64 =  (int64_t)kVolClamp;
    if (ds64 < -(int64_t)kVolClamp) ds64 = -(int64_t)kVolClamp;
    const fx s = fxmul((fx)ds64, kVol);
    if (s == 0) return;                                   // the deterministic sub-LSB dead zone
    // The cell centroid (int64 sums of 8 int32s, arithmetic >>3 per axis — deterministic floor).
    int64_t sx = 0, sy = 0, sz = 0;
    for (int k = 0; k < 8; ++k) {
        const FxVec3& p = verts[(size_t)cell.v[k]].pos;
        sx += p.x; sy += p.y; sz += p.z;
    }
    const FxVec3 centroid{(fx)(sx >> 3), (fx)(sy >> 3), (fx)(sz >> 3)};
    // Isotropic scale toward/away from the centroid (fixed corner order; pinned corners hold).
    for (int k = 0; k < 8; ++k) {
        SoftVert& p = verts[(size_t)cell.v[k]];
        if (p.flags & kFlagPinned) continue;
        p.pos = FxAdd(p.pos, FxScale(FxSub(p.pos, centroid), s));
    }
}

// ----- SoftParams: the per-step knobs (all Q16.16 / int, no float) -----------------------------------
struct SoftParams {
    FxVec3 gravity;         // Q16.16 gravity acceleration (world units / s^2)
    fx     dt = 0;          // Q16.16 timestep
    fx     groundY = 0;     // the ground plane (the CL4 y >= groundY clamp, frictionless)
    int    iters = 4;       // K rounds of (stretch -> shear -> volume -> ground) per step
    fx     kVol = 0;        // Q16.16 volume-preservation stiffness (0 = the pass is SKIPPED exactly)
    fx     damp = kOne;     // velocity damping at the re-encode (kOne = EXACT identity / no damping)
};

// ----- StepSoftBody: one full PBD volumetric step (the make-or-break reference) ----------------------
// integrate -> `iters` x (stretch -> shear -> volume -> ground) -> ground clamp -> velocity re-encode
// (the HR1/FL7 discipline; see the header banner for the exact order + why). kVol <= 0 never touches
// the volume pass (identity: the pure stretch+shear lattice). Pure integer, fixed order -> two-run
// byte-identical + cross-platform bit-identical.
inline void StepSoftBody(const SoftLattice& sl, std::vector<SoftVert>& verts,
                         const SoftConstraints& sc, const std::vector<SoftCell>& cells,
                         const SoftParams& p) {
    (void)sl;   // dims are implicit in verts/sc/cells; kept for a symmetric/extensible signature
    // (1) gravity integrate every vert (CL1 verbatim; pinned untouched; prev snapshotted for the
    //     re-encode below).
    const size_t n = verts.size();
    for (size_t i = 0; i < n; ++i)
        cloth::IntegrateParticle(verts[i], p.gravity, p.groundY, p.dt);

    // (2) `iters` rounds of (stretch -> shear -> volume -> ground), each list/cell set in its FIXED
    //     build order (the CL3/HR1 sequential deterministic-order contract).
    for (int it = 0; it < p.iters; ++it) {
        SolveDistanceList(verts, sc.stretch);
        SolveDistanceList(verts, sc.shear);
        if (p.kVol > 0)
            for (size_t c = 0; c < cells.size(); ++c)
                SolveVolumeConstraint(verts, cells[c], p.kVol);
        cloth::CollidePlane(verts, p.groundY);
    }

    // (3) ground clamp AFTER the loop (idempotent — covers iters == 0; a projection may have pushed a
    //     vert below).
    cloth::CollidePlane(verts, p.groundY);

    // (4) velocity RE-ENCODE (the HR1/FL7 discipline): v = (pos - prev) / dt, damped. Pinned verts
    //     skip (they never moved; vel stays 0). fxdiv(x, 0) == 0 by the fpx contract, and
    //     fxmul(v, kOne) == v EXACTLY, so damp == kOne is the exact identity.
    for (size_t i = 0; i < n; ++i) {
        SoftVert& v = verts[i];
        if (v.flags & kFlagPinned) continue;
        v.vel.x = fxmul(fxdiv(v.pos.x - v.prev.x, p.dt), p.damp);
        v.vel.y = fxmul(fxdiv(v.pos.y - v.prev.y, p.dt), p.damp);
        v.vel.z = fxmul(fxdiv(v.pos.z - v.prev.z, p.dt), p.damp);
    }
}

// StepSoftBodySteps: run K full soft-body steps (the showcase / test K-step driver).
inline void StepSoftBodySteps(const SoftLattice& sl, std::vector<SoftVert>& verts,
                              const SoftConstraints& sc, const std::vector<SoftCell>& cells,
                              const SoftParams& p, int steps) {
    for (int s = 0; s < steps; ++s)
        StepSoftBody(sl, verts, sc, cells, p);
}

// ----- The lockstep command stream (the CL5/FPX5/HR1 mold): IMPULSE ---------------------------------
// kCmdImpulse adds `arg` (a Q16.16 delta-velocity — a poke/kick on the jelly) to the target vert's
// velocity; pinned verts hold (the CL5 kCmdWind discipline). Out-of-range target is a deterministic
// no-op. Commands at the same tick apply in ARRAY ORDER (the deterministic input-order contract).
inline constexpr uint32_t kCmdImpulse = 0u;

struct SoftCommand {
    uint32_t tick   = 0;   // the tick this input applies on
    uint32_t kind   = 0;   // kCmdImpulse
    uint32_t target = 0;   // the target VERT index
    FxVec3   arg;          // the Q16.16 delta-velocity payload
};

inline void ApplySoftCommand(std::vector<SoftVert>& verts, const SoftCommand& c) {
    if (c.target >= (uint32_t)verts.size()) return;       // bounds-checked no-op
    SoftVert& v = verts[(size_t)c.target];
    if (c.kind == kCmdImpulse) {
        if (v.flags & kFlagPinned) return;                // pinned verts hold
        v.vel = FxAdd(v.vel, c.arg);
    }
}

// SimSoftTick: apply this tick's commands (array order) then StepSoftBody once — the fpx SimTick twin.
inline void SimSoftTick(const SoftLattice& sl, std::vector<SoftVert>& verts,
                        const SoftConstraints& sc, const std::vector<SoftCell>& cells,
                        const SoftParams& p, const std::vector<SoftCommand>& stream, uint32_t tick) {
    for (const SoftCommand& c : stream)
        if (c.tick == tick) ApplySoftCommand(verts, c);
    StepSoftBody(sl, verts, sc, cells, p);
}

// RunSoftBodyLockstep: THE peer entry point — `ticks` SimSoftTicks from a COPY of `init` fed the
// command stream ALONE (inputs, not state). authority == replica BIT-EXACT by determinism (the
// lockstep proof memcmps them). The RunClothLockstep/RunHairLockstep twin.
inline std::vector<SoftVert> RunSoftBodyLockstep(const SoftLattice& sl,
                                                 const std::vector<SoftVert>& init,
                                                 const SoftConstraints& sc,
                                                 const std::vector<SoftCell>& cells,
                                                 const SoftParams& p,
                                                 const std::vector<SoftCommand>& stream, int ticks) {
    std::vector<SoftVert> verts = init;
    for (int t = 0; t < ticks; ++t)
        SimSoftTick(sl, verts, sc, cells, p, stream, (uint32_t)t);
    return verts;
}

// RunSoftBodyRollback: the rollback harness (the RunClothRollback/RunHairRollback twin). Advance
// 0..mispredictTick with the authoritative stream, SNAPSHOT (the deep vector copy), speculatively
// advance a few ticks with the MISPREDICTED stream (the diverging client prediction), then RESTORE +
// re-simulate mispredictTick..ticks with the correct stream. The proof asserts the result ==
// RunSoftBodyLockstep(init, authStream, ticks) AND that the full mispredicted run DIFFERED (a real
// divergence was corrected).
inline std::vector<SoftVert> RunSoftBodyRollback(const SoftLattice& sl,
                                                 const std::vector<SoftVert>& init,
                                                 const SoftConstraints& sc,
                                                 const std::vector<SoftCell>& cells,
                                                 const SoftParams& p,
                                                 const std::vector<SoftCommand>& authStream,
                                                 const std::vector<SoftCommand>& mispredictStream,
                                                 int ticks, int mispredictTick) {
    std::vector<SoftVert> verts = init;
    for (int t = 0; t < mispredictTick; ++t)
        SimSoftTick(sl, verts, sc, cells, p, authStream, (uint32_t)t);
    const std::vector<SoftVert> snap = verts;             // SNAPSHOT (deep copy)
    int specTicks = ticks - mispredictTick;
    if (specTicks > 3) specTicks = 3;                     // bounded speculation (the CL5 shape)
    for (int s = 0; s < specTicks; ++s)
        SimSoftTick(sl, verts, sc, cells, p, mispredictStream, (uint32_t)(mispredictTick + s));
    verts = snap;                                         // ROLLBACK (restore)
    for (int t = mispredictTick; t < ticks; ++t)
        SimSoftTick(sl, verts, sc, cells, p, authStream, (uint32_t)t);
    return verts;
}

// ----- Deterministic diagnostics (test/showcase helpers, pure integer) -------------------------------

// MaxDistanceError: max |edge length - restLen| over a constraint list, Q16.16 LSBs — the honest PBD
// residual (iterative Gauss-Seidel leaves a deterministic, NON-zero error; the tests PIN it).
inline fx MaxDistanceError(const std::vector<SoftVert>& verts,
                           const std::vector<Constraint>& list) {
    fx worst = 0;
    for (const Constraint& c : list) {
        if (c.i >= verts.size() || c.j >= verts.size()) continue;
        const fx len = FxLength(FxSub(verts[(size_t)c.j].pos, verts[(size_t)c.i].pos));
        fx err = len - c.restLen;
        if (err < 0) err = -err;
        if (err > worst) worst = err;
    }
    return worst;
}

// TotalVolume / TotalRestVolume: the summed cell volume vs the summed rest volume, Q16.16 in int64 —
// the volume-preservation currency the tests band + pin (|Total - TotalRest| is the volume error).
inline int64_t TotalVolume(const std::vector<SoftVert>& verts, const std::vector<SoftCell>& cells) {
    int64_t v = 0;
    for (const SoftCell& c : cells) v += CellVolume(verts, c);
    return v;
}
inline int64_t TotalRestVolume(const std::vector<SoftCell>& cells) {
    int64_t v = 0;
    for (const SoftCell& c : cells) v += c.restVol;
    return v;
}

// MaxCellVolumeError: max |V - V0| over the cells, Q16.16 in int64 (the per-cell honest residual).
inline int64_t MaxCellVolumeError(const std::vector<SoftVert>& verts,
                                  const std::vector<SoftCell>& cells) {
    int64_t worst = 0;
    for (const SoftCell& c : cells) {
        int64_t err = CellVolume(verts, c) - c.restVol;
        if (err < 0) err = -err;
        if (err > worst) worst = err;
    }
    return worst;
}

// LatticeMinY / LatticeMaxY: the vertical extent probes (the squash/recover height currency —
// height = MaxY - MinY vs the rest extent (N-1)*spacing). Empty -> 0.
inline fx LatticeMinY(const std::vector<SoftVert>& verts) {
    if (verts.empty()) return 0;
    fx lo = verts[0].pos.y;
    for (const SoftVert& v : verts) if (v.pos.y < lo) lo = v.pos.y;
    return lo;
}
inline fx LatticeMaxY(const std::vector<SoftVert>& verts) {
    if (verts.empty()) return 0;
    fx hi = verts[0].pos.y;
    for (const SoftVert& v : verts) if (v.pos.y > hi) hi = v.pos.y;
    return hi;
}

// MaxSpeed: the max FxLength(vel) over all verts, Q16.16 (the settled-at-REST proof currency).
inline fx MaxSpeed(const std::vector<SoftVert>& verts) {
    fx worst = 0;
    for (const SoftVert& v : verts) {
        const fx s = FxLength(v.vel);
        if (s > worst) worst = s;
    }
    return worst;
}

// SoftBodyDigest: the FNV-1a-64 digest of the full vert state (SoftVert == ClothParticle -> the CL7
// ClothDigest reused verbatim; field-wise, layout/padding-independent — the golden currency).
inline uint64_t SoftBodyDigest(const std::vector<SoftVert>& verts) {
    return cloth::ClothDigest(verts);
}

}  // namespace softbody
}  // namespace hf::sim
