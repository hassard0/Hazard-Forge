#pragma once
// Slice DH1 — DETERMINISTIC CONVEX-CELL FRACTURE HULLS + DEBRIS + DUST (Track-R, upgrading a documented
// flagship fidelity gap). Namespace hf::sim::fhull. Header-only, PURE-CPU strict-integer core (a lit/wireframe
// render capstone is the ONLY float, and only in the showcase). This slice COMPOSES the frozen fracture
// flagship (engine/sim/fract.h, FR1-FR8) and the frozen convex-hull contact stack (gjk.h / manifold.h) and the
// frozen particle system (particles.h) READ-ONLY — it edits NONE of them, it only READS their output types.
//
// THE GAP IT CLOSES (CAPABILITIES.md L143 + fract.h's own FR4/FR8 banners): the fracture flagship shatters a
// mesh but its fragments are APPROXIMATED as colliders —
//   * FR4 (fract.h:628 SpawnFractWorld): each dislodged fragment is spawned as an fpx::FxBody BOUNDING SPHERE
//     (b.radius = boundRadius·cellSize) solved SPHERE-SPHERE (fpx::SolveContacts) -> the rubble is a pile of
//     ROLLING rounded chunks (fract.h:581 "each fragment collides as its BOUNDING SPHERE ... NOT interlocking
//     shards").
//   * FR8 (fract.h:1300 FragmentToBox): a first upgrade to an ORIENTED BOX == the AABB of the cell samples
//     about the centroid (fract.h:1275 "the AABB-BOX APPROXIMATION of the Voronoi cell, NOT the exact convex
//     cell hull"; "exact per-cell convex hulls (a gjk::FxHull per fragment through the warmhull solver) stay
//     FUTURE WORK").
// DH1 IS that future work: it builds each fragment's EXACT convex-cell hull from the cell's Voronoi VERTEX SET
// (the lattice samples fract.h's FR2 CSR already groups per cell), collides the shards as CONVEX bodies through
// the SHIPPED GJK/EPA support-based solver (gjk::StepHullWorld) so they rest on their FLAT FACES and interlock,
// adds small-fragment DEBRIS + an impact DUST-particle puff, and proves the whole shattered scene is bit-exact
// lockstep/rollback-replayable. Exact convex hulls of deterministic Voronoi cells replayable bit-for-bit is the
// "beyond UE5" moat improvement (UE5 Chaos destruction is float / non-deterministic).
//
// WHERE THE APPROXIMATION WAS + WHERE DH1 READS THE CELL VERTEX SET (the honest provenance):
//   fract.h FR2 (ExtractFragments, fract.h:239) already groups every lattice SAMPLE of a Voronoi cell into a
//   CSR slice: fragments.fragSamples[fragStart[c] .. fragStart[c+1]) are the sample indices of cell c, and
//   fract::SampleCoord(field, sample) maps each back to its (x,y,z) lattice coord. THAT set of integer lattice
//   points IS the cell's vertex set. FR2 then REDUCED it to a centroid + AABB + bounding-sphere radius (the
//   approximation). DH1 does NOT touch that reduction; it re-reads the SAME CSR sample set and builds the EXACT
//   convex hull of those points. So DH1 is purely additive: fract.h and its FR1-FR8 goldens are byte-untouched.
//
// THE HULL BUILDER (new, in this header — gjk.h/warmhull.h expose NO general hull builder; manifold.h's
// BuildCanonicalFaces is canonical-shapes-ONLY): a DETERMINISTIC INTEGER incremental 3D convex hull over the
// cell's lattice points. Predicates are EXACT int64 orient3d (lattice coords are tiny, |coord| < a few hundred,
// so a triple product fits int64 with vast headroom — NO float, NO overflow, identical on every compiler). The
// build processes points in ASCENDING sample order with strict-sign tie-breaks -> one deterministic answer.
// SAFETY NET (validity by construction): after building, EVERY input point is re-tested against EVERY final
// face; if any point lies strictly OUTSIDE (a degenerate/coplanar-extension miss) OR the hull exceeds the
// gjk::kMaxHullVerts=20 ceiling OR the point set is degenerate (coplanar/collinear — no seed tetrahedron), the
// builder FALLS BACK to the fragment's AABB box hull (the FR8 collider, honestly flagged exact=false). So a
// fragment ALWAYS gets a valid convex collider; it is the EXACT cell hull whenever the build succeeds (the
// common case, and EXACT for every axis-aligned box cell incl. the identity cube), else the documented AABB box.
//
// REUSE (read-only, nothing redefined): gjk::FxHull / HullWorld / StepHullWorldN / MeasureHullStack /
// HullContact / RunHullLockstep / RunHullRollback (the GJK/EPA convex solver + its lockstep, VERBATIM);
// gjk::MakeBox (the canonical box hull for the AABB fallback + the floor); manifold::BuildCanonicalFaces +
// FxHullInertiaBodyFull (the WH-flagship full-convex inertia — EXACT for the box-recognized hulls);
// particles::ParticlePool / EmitParticle (the dust puff). fpx.h fixed-point toolbox throughout.
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/Backend::) symbols, header-only, NO GPU/RHI, NO new shader. The
// collision solver (gjk::StepHullWorld) is int64 -> its GPU shader is Vulkan-only and the Metal path is CPU; DH1
// adds NO shader and runs the IDENTICAL CPU code on both backends (the FR5/FR7/FR8 pure-CPU-slice convention),
// so its golden is cross-backend bit-identical BY CONSTRUCTION. The dust + the wireframe render are cosmetic.

#include <cstdint>
#include <vector>

#include "sim/fract.h"       // read-only: the FR1-FR8 fracture pipeline output types (Field/Cells/Fragments/
                             // Bonds/BreakImpact/clusters + SampleCoord + FractAnchorPiece). BYTE-UNTOUCHED.
#include "sim/gjk.h"         // read-only: FxHull/HullWorld/StepHullWorldN/HullContact/MeasureHullStack/
                             // Run*Lockstep/Run*Rollback/MakeBox + the convex Q16.16 toolbox. BYTE-UNTOUCHED.
#include "sim/manifold.h"    // read-only: BuildCanonicalFaces + FxHullInertiaBodyFull (the WH full-convex
                             // inertia, EXACT for a box-recognized hull). BYTE-UNTOUCHED.
#include "sim/particles.h"   // read-only: ParticlePool/EmitParticle/InitParticlePool (the dust puff). UNTOUCHED.

namespace hf::sim {
namespace fhull {

// Re-export the fixed-point primitives + the convex helpers (READ-ONLY — the shared Q16.16 format).
using fpx::fx;
using fpx::FxVec3;
using fpx::kOne;
using fpx::kFrac;
using fpx::fxmul;
using gjk::FxHull;
using gjk::HullWorld;
using gjk::kMaxHullVerts;

// ==================================================================================================
// (A) The exact integer 3D convex-hull builder (the DH1 core primitive — deterministic, int64-exact).
// ==================================================================================================

// A hull triangle: three vertex indices into a point array, outward-wound. Used only during the build and for
// the volume / distinct-face-plane accounting (the collision uses gjk::FxHull verts, NOT these triangles).
struct HullTri { uint32_t a = 0, b = 0, c = 0; };

// Orient3D(a,b,c,p): the EXACT int64 signed volume (×6) of the tetra (a,b,c,p) = dot(cross(b-a,c-a), p-a).
// > 0  -> p is in FRONT of the outward face (a,b,c) (strictly outside that face's plane, on the normal side);
// == 0 -> p is COPLANAR with (a,b,c); < 0 -> p is BEHIND. Lattice coords are tiny so every product fits int64
// with huge headroom -> exact, deterministic, identical on every compiler/vendor. NO float.
inline int64_t Orient3D(const FxVec3& a, const FxVec3& b, const FxVec3& c, const FxVec3& p) {
    const int64_t bax = (int64_t)b.x - a.x, bay = (int64_t)b.y - a.y, baz = (int64_t)b.z - a.z;
    const int64_t cax = (int64_t)c.x - a.x, cay = (int64_t)c.y - a.y, caz = (int64_t)c.z - a.z;
    const int64_t pax = (int64_t)p.x - a.x, pay = (int64_t)p.y - a.y, paz = (int64_t)p.z - a.z;
    const int64_t nx = bay * caz - baz * cay;   // (b-a) x (c-a)
    const int64_t ny = baz * cax - bax * caz;
    const int64_t nz = bax * cay - bay * cax;
    return nx * pax + ny * pay + nz * paz;      // n · (p - a)
}

// TriNormal(a,b,c): the raw integer outward-candidate normal (b-a)x(c-a) as an int64 triple (NOT normalized).
struct HullNormal64 { int64_t x, y, z; };
inline HullNormal64 TriNormal(const FxVec3& a, const FxVec3& b, const FxVec3& c) {
    const int64_t bax = (int64_t)b.x - a.x, bay = (int64_t)b.y - a.y, baz = (int64_t)b.z - a.z;
    const int64_t cax = (int64_t)c.x - a.x, cay = (int64_t)c.y - a.y, caz = (int64_t)c.z - a.z;
    return HullNormal64{bay * caz - baz * cay, baz * cax - bax * caz, bax * cay - bay * cax};
}

// A canonical supporting plane of the hull: a reduced (gcd-divided, sign-normalized) integer normal + the plane
// offset n·p. Distinct planes are the hull's FACES; a true hull VERTEX lies on >= 3 of them with rank-3 normals.
struct HullPlane { int64_t nx, ny, nz, d; };

// The convex-hull build result over an integer point array: the hull VERTEX indices (a subset of the input, in
// ascending input order — the TRUE corners only, coplanar surface points removed via plane membership) + the
// outward-wound TRIANGLES + the distinct supporting PLANES. ok=false -> the point set was degenerate
// (coplanar/collinear — no non-degenerate seed tetrahedron) OR a containment/vertex-cap check failed; the caller
// then uses the AABB fallback. Deterministic.
struct HullBuild {
    bool                   ok = false;
    std::vector<uint32_t>  vertIdx;   // input indices that are TRUE hull corners (ascending)
    std::vector<HullTri>   tris;      // outward-wound triangles over INPUT indices (for volume/inertia)
    std::vector<HullPlane> planes;    // distinct supporting planes (the hull faces)
};

// DistinctFacePlanes(pts, tris): the distinct supporting planes of the triangulated hull (coplanar triangles
// collapse to one plane — a cube's 12 triangles -> its 6 planes). Each normal is reduced by its gcd and
// sign-normalized (first non-zero component positive, offset carried along). Pure integer, deterministic.
inline std::vector<HullPlane> DistinctFacePlanes(const std::vector<FxVec3>& pts, const std::vector<HullTri>& tris) {
    auto gcd64 = [](int64_t a, int64_t b) -> int64_t {
        a = a < 0 ? -a : a; b = b < 0 ? -b : b;
        while (b) { const int64_t t = a % b; a = b; b = t; }
        return a;   // raw gcd (gcd(0,0)==0) so gcd(0,0,k)==k reduces consistently
    };
    std::vector<HullPlane> planes;
    for (const HullTri& t : tris) {
        HullNormal64 nrm = TriNormal(pts[t.a], pts[t.b], pts[t.c]);
        if (nrm.x == 0 && nrm.y == 0 && nrm.z == 0) continue;   // degenerate sliver -> no plane
        int64_t g = gcd64(gcd64(nrm.x, nrm.y), nrm.z); if (g == 0) g = 1;
        int64_t nx = nrm.x / g, ny = nrm.y / g, nz = nrm.z / g;
        int64_t sign = (nx != 0) ? (nx < 0 ? -1 : 1) : (ny != 0) ? (ny < 0 ? -1 : 1) : (nz < 0 ? -1 : 1);
        nx *= sign; ny *= sign; nz *= sign;
        const int64_t d = nx * (int64_t)pts[t.a].x + ny * (int64_t)pts[t.a].y + nz * (int64_t)pts[t.a].z;
        bool seen = false;
        for (const HullPlane& pl : planes) if (pl.nx == nx && pl.ny == ny && pl.nz == nz && pl.d == d) { seen = true; break; }
        if (!seen) planes.push_back(HullPlane{nx, ny, nz, d});
    }
    return planes;
}

// PointIsHullVertex(p, planes): a point is a TRUE hull corner iff it lies exactly on >= 3 supporting planes
// whose normals span rank 3 (a non-zero scalar triple product for some triple). An edge point lies on 2 planes
// (rank 2); a face-interior point on 1 (rank 1). Pure integer, exact.
inline bool PointIsHullVertex(const FxVec3& p, const std::vector<HullPlane>& planes) {
    std::vector<const HullPlane*> on;
    for (const HullPlane& pl : planes)
        if (pl.nx * (int64_t)p.x + pl.ny * (int64_t)p.y + pl.nz * (int64_t)p.z == pl.d) on.push_back(&pl);
    if (on.size() < 3) return false;
    for (size_t i = 0; i < on.size(); ++i)
        for (size_t j = i + 1; j < on.size(); ++j)
            for (size_t k = j + 1; k < on.size(); ++k) {
                const HullPlane& A = *on[i]; const HullPlane& B = *on[j]; const HullPlane& C = *on[k];
                const int64_t det = A.nx * (B.ny * C.nz - B.nz * C.ny)
                                  - A.ny * (B.nx * C.nz - B.nz * C.nx)
                                  + A.nz * (B.nx * C.ny - B.ny * C.nx);
                if (det != 0) return true;   // three independent supporting planes -> a corner
            }
    return false;
}

// BuildConvexHull(pts): the deterministic integer incremental 3D convex hull of the (deduplicated) point set.
//   (1) Seed tetra: p0 = point 0; p1 = the point farthest from p0 (max squared distance, lowest-index tie);
//       p2 = the point maximizing |area| of triangle (p0,p1,·) (max |cross|², lowest-index tie); p3 = the point
//       maximizing |Orient3D(p0,p1,p2,·)| (lowest-index tie). Any stage failing to find a non-degenerate pick
//       -> ok=false (a flat/thin cell; the caller falls back to the AABB box).
//   (2) Four outward faces of the seed tetra (each wound so the tetra centroid is BEHIND it — the interior
//       reference, evaluated exactly via the ×4-scaled centroid trick so it stays pure integer).
//   (3) For each remaining point in ASCENDING index: gather the VISIBLE faces (Orient3D(face,p) > 0). None ->
//       p is inside/on the surface -> skip. Else find the HORIZON (directed edges of visible faces whose
//       reverse is not also a visible-face edge), delete the visible faces, and cone the horizon to p (one new
//       outward triangle per horizon edge). Fixed iteration order -> deterministic.
//   (4) Validity: re-test EVERY input point against EVERY surviving face; a strictly-outside point (a coplanar-
//       extension miss) -> ok=false. Collect the hull vertex set; > kMaxHullVerts -> ok=false.
inline HullBuild BuildConvexHull(const std::vector<FxVec3>& pts) {
    HullBuild out;
    const uint32_t n = (uint32_t)pts.size();
    if (n < 4) return out;   // need >= 4 points for a 3D hull (degenerate -> AABB fallback)

    // (1) seed tetra ----------------------------------------------------------------------------------
    auto sqDist = [](const FxVec3& a, const FxVec3& b) -> int64_t {
        const int64_t dx = (int64_t)a.x - b.x, dy = (int64_t)a.y - b.y, dz = (int64_t)a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    };
    const uint32_t i0 = 0u;
    uint32_t i1 = 0u; int64_t best = -1;
    for (uint32_t i = 0; i < n; ++i) { const int64_t d = sqDist(pts[i], pts[i0]); if (d > best) { best = d; i1 = i; } }
    if (i1 == i0) return out;   // all points coincident
    // p2: maximize |cross(p1-p0, p·-p0)|² (the farthest-from-line point).
    uint32_t i2 = 0u; int64_t bestA = -1;
    for (uint32_t i = 0; i < n; ++i) {
        const HullNormal64 nrm = TriNormal(pts[i0], pts[i1], pts[i]);
        const int64_t a2 = nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z;
        if (a2 > bestA) { bestA = a2; i2 = i; }
    }
    if (bestA <= 0) return out;   // all points collinear -> degenerate
    // p3: maximize |Orient3D(p0,p1,p2,·)| (the farthest-from-plane point).
    uint32_t i3 = 0u; int64_t bestV = -1;
    for (uint32_t i = 0; i < n; ++i) {
        int64_t v = Orient3D(pts[i0], pts[i1], pts[i2], pts[i]); if (v < 0) v = -v;
        if (v > bestV) { bestV = v; i3 = i; }
    }
    if (bestV <= 0) return out;   // all points coplanar -> degenerate (AABB fallback covers the flat cell)

    // (2) the four outward seed faces. The ×4 centroid (sum of the 4 seed verts) is a strictly-interior integer
    // reference: a face (a,b,c) is outward iff its centroid side is BEHIND, i.e. dot(n, cs - 4·a) < 0.
    const int64_t csx = (int64_t)pts[i0].x + pts[i1].x + pts[i2].x + pts[i3].x;
    const int64_t csy = (int64_t)pts[i0].y + pts[i1].y + pts[i2].y + pts[i3].y;
    const int64_t csz = (int64_t)pts[i0].z + pts[i1].z + pts[i2].z + pts[i3].z;
    auto centroidInFront = [&](const FxVec3& a, const FxVec3& b, const FxVec3& c) -> bool {
        const HullNormal64 nrm = TriNormal(a, b, c);
        const int64_t s = nrm.x * (csx - 4 * (int64_t)a.x) + nrm.y * (csy - 4 * (int64_t)a.y) +
                          nrm.z * (csz - 4 * (int64_t)a.z);
        return s > 0;   // centroid strictly in front -> the winding points INWARD -> must flip
    };
    auto makeOutward = [&](uint32_t a, uint32_t b, uint32_t c) -> HullTri {
        if (centroidInFront(pts[a], pts[b], pts[c])) return HullTri{a, c, b};   // flip to face outward
        return HullTri{a, b, c};
    };
    std::vector<HullTri> tris;
    tris.push_back(makeOutward(i0, i1, i2));
    tris.push_back(makeOutward(i0, i1, i3));
    tris.push_back(makeOutward(i0, i2, i3));
    tris.push_back(makeOutward(i1, i2, i3));

    // (3) incremental insertion of the remaining points in ASCENDING index order.
    for (uint32_t p = 0; p < n; ++p) {
        if (p == i0 || p == i1 || p == i2 || p == i3) continue;
        // visible faces: Orient3D(face, p) > 0.
        std::vector<uint8_t> vis(tris.size(), 0u);
        bool anyVis = false;
        for (size_t t = 0; t < tris.size(); ++t) {
            if (Orient3D(pts[tris[t].a], pts[tris[t].b], pts[tris[t].c], pts[p]) > 0) { vis[t] = 1u; anyVis = true; }
        }
        if (!anyVis) continue;   // inside or on the surface -> not a new vertex

        // horizon: directed edges of visible faces whose REVERSE is not also a visible-face edge. Collect all
        // directed edges of visible faces (fixed face+edge order), then keep the ones without a reverse partner.
        struct Edge { uint32_t u, v; };
        std::vector<Edge> vedges;
        for (size_t t = 0; t < tris.size(); ++t) {
            if (!vis[t]) continue;
            vedges.push_back(Edge{tris[t].a, tris[t].b});
            vedges.push_back(Edge{tris[t].b, tris[t].c});
            vedges.push_back(Edge{tris[t].c, tris[t].a});
        }
        auto hasReverse = [&](uint32_t u, uint32_t v) -> bool {
            for (const Edge& e : vedges) if (e.u == v && e.v == u) return true;
            return false;
        };
        std::vector<Edge> horizon;
        for (const Edge& e : vedges) if (!hasReverse(e.u, e.v)) horizon.push_back(e);

        // delete visible faces (keep the rest, order-stable), then cone the horizon to p.
        std::vector<HullTri> kept;
        kept.reserve(tris.size());
        for (size_t t = 0; t < tris.size(); ++t) if (!vis[t]) kept.push_back(tris[t]);
        for (const Edge& e : horizon) kept.push_back(makeOutward(e.u, e.v, p));
        tris.swap(kept);
    }

    // (4) validity: a strictly-outside point (a coplanar-extension miss) -> reject (the AABB fallback covers it).
    for (uint32_t p = 0; p < n; ++p) {
        for (const HullTri& t : tris) {
            if (Orient3D(pts[t.a], pts[t.b], pts[t.c], pts[p]) > 0) return HullBuild{};   // outside -> reject
        }
    }
    // (5) TRUE corners: the incremental build leaves coplanar surface points as vertices (extreme at insertion
    // time, non-extreme later). Re-derive the corners from the supporting planes (a point on >= 3 rank-3 planes)
    // so a solid box cell yields EXACTLY its 8 corners (not the 17 surface samples the raw build references).
    out.planes = DistinctFacePlanes(pts, tris);
    std::vector<uint32_t> verts;
    for (uint32_t i = 0; i < n; ++i) if (PointIsHullVertex(pts[i], out.planes)) verts.push_back(i);
    if (verts.size() < 4 || (uint32_t)verts.size() > kMaxHullVerts) return HullBuild{};   // degenerate/over-cap

    out.ok = true;
    out.vertIdx = std::move(verts);
    out.tris = std::move(tris);
    return out;
}

// ==================================================================================================
// (B) The per-fragment cell hull: build the EXACT hull, its metrics, and the gjk::FxHull collider.
// ==================================================================================================

// CountDistinctFacePlanes(pts, tris): the number of distinct SUPPORTING PLANES of the triangulated hull (the
// honest "faces" count — a cube's 12 triangles collapse to its 6 planes). Each triangle contributes its
// outward integer normal reduced by its gcd (a canonical direction) + the plane offset n·a; distinct (dir,
// offset) pairs are the faces. Pure integer, deterministic.
inline uint32_t CountDistinctFacePlanes(const std::vector<FxVec3>& pts, const std::vector<HullTri>& tris) {
    return (uint32_t)DistinctFacePlanes(pts, tris).size();
}

// HullLatticeVolume6(pts, tris): SIX times the hull volume in LATTICE units (exact int64) — the closed-mesh
// divergence sum Σ dot(a, cross(b,c)) over the outward-wound triangles (origin at the lattice origin). The true
// integer 6·volume; the world volume is (this/6)·worldCellSize³. Deterministic, exact.
inline int64_t HullLatticeVolume6(const std::vector<FxVec3>& pts, const std::vector<HullTri>& tris) {
    int64_t sum6 = 0;
    for (const HullTri& t : tris) {
        const FxVec3& a = pts[t.a]; const FxVec3& b = pts[t.b]; const FxVec3& c = pts[t.c];
        const int64_t cx = (int64_t)b.y * c.z - (int64_t)b.z * c.y;   // (b × c)
        const int64_t cy = (int64_t)b.z * c.x - (int64_t)b.x * c.z;
        const int64_t cz = (int64_t)b.x * c.y - (int64_t)b.y * c.x;
        sum6 += (int64_t)a.x * cx + (int64_t)a.y * cy + (int64_t)a.z * cz;   // a · (b × c)
    }
    return sum6 < 0 ? -sum6 : sum6;
}

// CellVolumeWorld(latticeVol6, worldCellSize): the Q16.16 world-unit hull volume = (latticeVol6/6)·cellSize³.
inline fx CellVolumeWorld(int64_t latticeVol6, fx worldCellSize) {
    const fx cellVol = fxmul(fxmul(worldCellSize, worldCellSize), worldCellSize);   // cellSize³ (Q16.16)
    const int64_t latVol = latticeVol6 / 6;                                          // integer cell count
    return (fx)((int64_t)latVol * (int64_t)cellVol);                                 // latVol · cellVol (Q16.16)
}

// The built collider for one fragment. `hull` is the gjk::FxHull collision hull (LOCAL verts in Q16.16 world
// units, centered on the fragment's integer lattice centroid, which becomes the body position). `exact` is true
// iff the exact cell hull was built (false == AABB-box fallback). `isBox` is true iff the exact hull is an
// axis-aligned box in gjk::MakeBox sign-sweep order (so manifold::BuildCanonicalFaces + FxHullInertiaBodyFull
// give it EXACT full-convex inertia + a 6-quad face table). `faceCount` is the distinct supporting-plane count
// (6 for a box). `worldVolume` is the Q16.16 hull volume; `aabbVolume` the Q16.16 point-AABB volume (>= it —
// the tightness win). `debris` marks a below-threshold small shard.
struct CellHull {
    FxHull   hull;                 // gjk collision hull (local verts, Q16.16 world units)
    bool     exact      = false;   // true = exact Voronoi-cell hull; false = AABB box fallback
    bool     isBox      = false;   // true = axis-aligned box in canonical order (WH-exact inertia)
    uint32_t hullVerts  = 0;       // hull.count (cached)
    uint32_t faceCount  = 0;       // distinct supporting planes (6 for a box; 0 if degenerate)
    fx       worldVolume = 0;      // Q16.16 hull volume in world units
    fx       aabbVolume  = 0;      // Q16.16 point-AABB volume (hull <= this — the tightness pin)
    bool     debris     = false;   // true = small sub-fragment (below the debris volume threshold)
    int32_t  cx = 0, cy = 0, cz = 0;   // the integer lattice centroid (the body position source)
};

// LocalWorldVert(coord, centroid, worldCellSize): a lattice point -> the gjk::FxHull LOCAL vert in Q16.16 world
// units, relative to the centroid: fxmul((coord - centroid) << kFrac, worldCellSize). Pure integer.
inline FxVec3 LocalWorldVert(int32_t x, int32_t y, int32_t z, int32_t cx, int32_t cy, int32_t cz, fx worldCellSize) {
    return FxVec3{fxmul((fx)((int64_t)(x - cx) << kFrac), worldCellSize),
                  fxmul((fx)((int64_t)(y - cy) << kFrac), worldCellSize),
                  fxmul((fx)((int64_t)(z - cz) << kFrac), worldCellSize)};
}

// IsAxisBox(pts, verts): true iff the hull vertex set is EXACTLY the 8 corners {minx,maxx}×{miny,maxy}×
// {minz,maxz} of its AABB (a full axis-aligned box). Fills the 8 corner lattice coords in gjk::MakeBox sign-
// sweep order (x outer, y mid, z inner) when true. Pure integer.
inline bool IsAxisBox(const std::vector<FxVec3>& pts, const std::vector<uint32_t>& verts, FxVec3 cornersOut[8]) {
    if (verts.size() != 8) return false;
    int32_t mnx = pts[verts[0]].x, mny = pts[verts[0]].y, mnz = pts[verts[0]].z;
    int32_t mxx = mnx, mxy = mny, mxz = mnz;
    for (uint32_t vi : verts) {
        const FxVec3& v = pts[vi];
        if (v.x < mnx) mnx = v.x; if (v.x > mxx) mxx = v.x;
        if (v.y < mny) mny = v.y; if (v.y > mxy) mxy = v.y;
        if (v.z < mnz) mnz = v.z; if (v.z > mxz) mxz = v.z;
    }
    if (mnx == mxx || mny == mxy || mnz == mxz) return false;   // flat -> not a 3D box
    // every hull vert must be a corner (each coord at its axis min or max).
    for (uint32_t vi : verts) {
        const FxVec3& v = pts[vi];
        const bool xok = (v.x == mnx || v.x == mxx);
        const bool yok = (v.y == mny || v.y == mxy);
        const bool zok = (v.z == mnz || v.z == mxz);
        if (!(xok && yok && zok)) return false;
    }
    // emit the 8 corners in gjk::MakeBox order (ix outer, iy mid, iz inner).
    const int32_t sx[2] = {mnx, mxx}, sy[2] = {mny, mxy}, sz[2] = {mnz, mxz};
    uint32_t k = 0;
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz)
                cornersOut[k++] = FxVec3{sx[ix], sy[iy], sz[iz]};
    return true;
}

// kDebrisVolumeCells: fragments whose hull world volume is below this many CELL-volumes are DEBRIS (small
// sub-shards). A host-fixed integer threshold (in lattice cell-volume units); the caller may override.
inline constexpr int64_t kDebrisVolumeCells = 8;

// BuildCellHull(field, cells, frags, f, worldCellSize): the EXACT convex-cell hull collider for fragment f. It
// re-reads fragment f's Voronoi VERTEX SET (fract.h FR2's CSR sample slice) and builds the hull; on any
// degeneracy/overflow it falls back to the AABB box. Pure integer, deterministic.
inline CellHull BuildCellHull(const fract::FractField& field, const fract::FractCells& cells,
                              const fract::FractFragments& frags, uint32_t f, fx worldCellSize) {
    (void)cells;
    CellHull out;
    if (f >= frags.fragments.size()) return out;
    const fract::FractFragment& fr = frags.fragments[(size_t)f];
    out.cx = fr.cx; out.cy = fr.cy; out.cz = fr.cz;

    // --- read the cell's lattice vertex set from the FR2 CSR (the SAME samples FR2 reduced to centroid/AABB) ---
    const uint32_t c = frags.fragmentToCell[(size_t)f];
    const uint32_t begin = frags.fragStart[(size_t)c];
    const uint32_t end   = frags.fragStart[(size_t)c + 1u];
    std::vector<FxVec3> pts;
    pts.reserve((size_t)(end - begin));
    for (uint32_t k = begin; k < end; ++k) {
        const fract::FractCoord p = fract::SampleCoord(field, (int)frags.fragSamples[(size_t)k]);
        pts.push_back(FxVec3{p.x, p.y, p.z});
    }
    // point-AABB volume (Q16.16) for the tightness pin: the span product · cellSize³.
    {
        const int64_t sx = (int64_t)fr.maxx - fr.minx, sy = (int64_t)fr.maxy - fr.miny, sz = (int64_t)fr.maxz - fr.minz;
        out.aabbVolume = CellVolumeWorld((sx * sy * sz) * 6, worldCellSize);   // ×6 because CellVolumeWorld /6's
    }

    // --- try the EXACT hull; on success emit a gjk::FxHull (box-canonicalized when it is an axis box) ---
    const HullBuild hb = BuildConvexHull(pts);
    if (hb.ok) {
        out.exact = true;
        out.faceCount = CountDistinctFacePlanes(pts, hb.tris);
        out.worldVolume = CellVolumeWorld(HullLatticeVolume6(pts, hb.tris), worldCellSize);
        FxVec3 boxCorners[8];
        if (IsAxisBox(pts, hb.vertIdx, boxCorners)) {
            out.isBox = true;
            out.hull.count = 8u;
            for (uint32_t k = 0; k < 8u; ++k)
                out.hull.verts[k] = LocalWorldVert(boxCorners[k].x, boxCorners[k].y, boxCorners[k].z,
                                                   fr.cx, fr.cy, fr.cz, worldCellSize);
        } else {
            out.hull.count = (uint32_t)hb.vertIdx.size();
            for (uint32_t k = 0; k < out.hull.count; ++k) {
                const FxVec3& v = pts[hb.vertIdx[k]];
                out.hull.verts[k] = LocalWorldVert(v.x, v.y, v.z, fr.cx, fr.cy, fr.cz, worldCellSize);
            }
        }
        out.hullVerts = out.hull.count;
    } else {
        // --- AABB-box fallback (the FR8 collider shape): the 8 AABB corners about the centroid, canonical order.
        out.exact = false;
        const int32_t sx[2] = {fr.minx, fr.maxx}, sy[2] = {fr.miny, fr.maxy}, sz[2] = {fr.minz, fr.maxz};
        uint32_t k = 0;
        for (int ix = 0; ix < 2; ++ix)
            for (int iy = 0; iy < 2; ++iy)
                for (int iz = 0; iz < 2; ++iz)
                    out.hull.verts[k++] = LocalWorldVert(sx[ix], sy[iy], sz[iz], fr.cx, fr.cy, fr.cz, worldCellSize);
        out.hull.count = 8u;
        out.hullVerts = 8u;
        out.isBox = true;   // the fallback IS a canonical axis box (exact box inertia), just not the cell hull
        out.faceCount = 6u;
        out.worldVolume = out.aabbVolume;
    }
    // debris flag: a small shard (hull volume below the threshold cell-volume count).
    out.debris = out.worldVolume < CellVolumeWorld(kDebrisVolumeCells * 6, worldCellSize);
    return out;
}

// CellHullInvInertiaBody(ch, invMass): the fragment's body-space inverse inertia. For a box-recognized hull
// this REUSES the WH full-convex inertia (manifold::FxHullInertiaBodyFull over BuildCanonicalFaces) — EXACT for
// a box. For a general (non-box) exact hull, the collision solver (gjk::StepHullWorld) uses gjk's own
// AABB-diagonal hull inertia; this helper returns that same diagonal tensor for a consistent stand-alone pin.
// Documented: the DH1 exact-hull win is the GEOMETRY + the support-based CONTACT; a full general-hull inertia
// tensor in the gjk step is a shipped-solver property (GJ4 uses the diagonal), not re-plumbed here.
inline convex::FxMat3 CellHullInvInertiaBody(const CellHull& ch, fx invMass) {
    if (ch.isBox) {
        const manifold::FxHullFaces faces = manifold::BuildCanonicalFaces(ch.hull);
        return manifold::FxHullInertiaBodyFull(ch.hull, faces, invMass);   // EXACT full-convex inertia (box)
    }
    return convex::FxMat3Diagonal(gjk::FxHullInvInertiaBody(ch.hull, invMass));   // the GJ4 solver's diagonal
}

// ==================================================================================================
// (C) The shattered-scene spawn: fragments (+debris) as convex hulls + a static floor hull.
// ==================================================================================================

// The DH1 spawn config — the lattice->world scale + gravity/ground + the impact seed + the floor extent. Mirrors
// fract::FractStepConfig (carried verbatim shape) plus the floor half-extents (gjk has no ground plane, so a big
// static box HULL is the floor, like FR8's SpawnFractHullWorld).
struct FractHullConfig {
    fx     worldCellSize = kOne;      // Q16.16 lattice->world scale
    FxVec3 gravity{};                 // Q16.16 acceleration
    fx     groundY = 0;               // Q16.16 ground plane height (the floor hull's top face)
    FxVec3 impactDir{};               // Q16.16 impact direction
    fx     impactSpeed = 0;           // Q16.16 impact speed (seeds the impacted dynamic fragment)
    FxVec3 floorHalfExtents{};        // Q16.16 floor box half-extents
};

// The result of a spawn: the gjk hull world + the per-body CellHull metadata (index-aligned to world.bodies for
// the fragment bodies; the LAST body is the static floor, which has no CellHull entry).
struct FractHullScene {
    HullWorld             world;      // gjk hull world (fragment hulls + a trailing static floor hull)
    std::vector<CellHull> cells;      // per-fragment CellHull (size == fragmentCount; floor excluded)
    uint32_t              floorIndex = 0;   // world.bodies index of the static floor hull
    uint32_t              anchorPiece = 0xFFFFFFFFu;  // the static anchor cluster label
    uint32_t              exactHulls = 0;   // # fragments given the EXACT cell hull (vs AABB fallback)
    uint32_t              debrisCount = 0;  // # fragments flagged as small debris
};

// SpawnFractureHullScene(field, cells, frags, bonds, severed, clusters, impact, cfg): build the shattered gjk
// hull world. One body+hull per fragment (ascending fragment index — the FR2/FR4 compact order): the LARGEST
// piece's fragments STATIC (FractAnchorPiece — the intact base), all others DYNAMIC; the impacted dynamic
// fragment seeded with impactDir·impactSpeed; each collider is BuildCellHull's EXACT convex cell hull (or its
// AABB fallback). A big STATIC FLOOR box hull (gjk::MakeBox) is appended LAST at groundY. Pure integer.
inline FractHullScene SpawnFractureHullScene(const fract::FractField& field, const fract::FractCells& cellsMap,
                                             const fract::FractFragments& frags, const fract::FractBonds& bonds,
                                             const std::vector<uint8_t>& severed,
                                             const std::vector<uint32_t>& clusters,
                                             const fract::BreakImpact& impact, const FractHullConfig& cfg) {
    (void)bonds; (void)severed;   // the clusters already encode the break connectivity (the FR4 contract).
    FractHullScene scene;
    const uint32_t M = (uint32_t)frags.fragments.size();
    if (M == 0u) return scene;

    scene.anchorPiece = fract::FractAnchorPiece(frags, clusters);
    const bool haveClusters = ((uint32_t)clusters.size() == M);
    scene.cells.reserve((size_t)M);
    scene.world.bodies.reserve((size_t)M + 1u);
    scene.world.hulls.reserve((size_t)M + 1u);
    int64_t sumCx = 0, sumCz = 0;

    for (uint32_t f = 0; f < M; ++f) {
        const fract::FractFragment& fr = frags.fragments[(size_t)f];
        CellHull ch = BuildCellHull(field, cellsMap, frags, f, cfg.worldCellSize);
        if (ch.exact) ++scene.exactHulls;
        if (ch.debris) ++scene.debrisCount;

        fpx::FxBody b;
        b.pos = FxVec3{fxmul((fx)((int64_t)fr.cx << kFrac), cfg.worldCellSize),
                       fxmul((fx)((int64_t)fr.cy << kFrac), cfg.worldCellSize),
                       fxmul((fx)((int64_t)fr.cz << kFrac), cfg.worldCellSize)};
        b.vel = FxVec3{0, 0, 0};
        b.radius = fxmul((fx)((int64_t)fr.boundRadius << kFrac), cfg.worldCellSize);   // provenance/debug only
        b.invMass = fr.invMass;
        b.orient = fpx::FxQuat{0, 0, 0, kOne};   // identity — every settled rotation is EARNED from the contacts
        b.angVel = FxVec3{0, 0, 0};

        const bool isAnchor = haveClusters && (clusters[(size_t)f] == scene.anchorPiece) &&
                              (scene.anchorPiece != 0xFFFFFFFFu);
        if (isAnchor) { b.invMass = 0; b.flags = 0; }             // the held base: static
        else          { b.flags = fpx::kFlagDynamic; }            // dislodged: dynamic convex shard
        scene.world.bodies.push_back(b);
        scene.world.hulls.push_back(ch.hull);
        scene.cells.push_back(ch);
        sumCx += fr.cx; sumCz += fr.cz;
    }

    // Seed the impacted fragment's body with the impact velocity (only if it is dynamic — the FR4 rule).
    if (impact.fragment < M) {
        fpx::FxBody& hit = scene.world.bodies[(size_t)impact.fragment];
        if (hit.flags & fpx::kFlagDynamic)
            hit.vel = FxVec3{fxmul(cfg.impactDir.x, cfg.impactSpeed), fxmul(cfg.impactDir.y, cfg.impactSpeed),
                             fxmul(cfg.impactDir.z, cfg.impactSpeed)};
    }

    // The static FLOOR box hull, appended LAST (top face at groundY, centered at the integer mean centroid).
    {
        const int32_t mx = (int32_t)(sumCx / (int64_t)M);
        const int32_t mz = (int32_t)(sumCz / (int64_t)M);
        fpx::FxBody floor;
        floor.pos = FxVec3{fxmul((fx)((int64_t)mx << kFrac), cfg.worldCellSize),
                           cfg.groundY - cfg.floorHalfExtents.y,
                           fxmul((fx)((int64_t)mz << kFrac), cfg.worldCellSize)};
        floor.vel = FxVec3{0, 0, 0};
        floor.invMass = 0;
        floor.flags = 0;   // static
        floor.orient = fpx::FxQuat{0, 0, 0, kOne};
        floor.angVel = FxVec3{0, 0, 0};
        scene.floorIndex = (uint32_t)scene.world.bodies.size();
        scene.world.bodies.push_back(floor);
        scene.world.hulls.push_back(gjk::MakeBox(cfg.floorHalfExtents.x, cfg.floorHalfExtents.y,
                                                 cfg.floorHalfExtents.z));
    }
    return scene;
}

// ==================================================================================================
// (D) The convex-shard settle step: the SHIPPED GJK/EPA hull solver, reused VERBATIM.
// ==================================================================================================

// StepFractureHullN(world, cfg, ticks): run `ticks` gjk::StepHullWorld ticks — the shards fall + collide as
// CONVEX bodies (GJK/EPA narrowphase + multi-body Gauss-Seidel impulse + position de-penetration + 6-DOF
// integrate), resting on their FLAT FACES. gjk.h is called AS-IS (byte-untouched). Pure integer, cross-backend
// bit-identical (pure CPU on both — the FR5/FR7/FR8 convention).
inline void StepFractureHullN(HullWorld& world, const convex::ConvexStepConfig& cfg, uint32_t ticks) {
    gjk::StepHullWorldN(world, cfg, ticks);
}

// ==================================================================================================
// (E) The honest settled-shard metrics (rest / rotation / face-to-face contact / pile floor).
// ==================================================================================================

// FaceToFaceContactNormal(world, i, j): the HullContact manifold normal between bodies i and j (A->B signed),
// or {0,0,0} if separated. The face-to-face pin reads this: two convex shards resting flush have an AXIS-like
// normal (dominant single component ~kOne — a face normal), NOT a radial sphere normal. Pure integer (reused
// gjk::HullContact).
inline FxVec3 FaceToFaceContactNormal(const HullWorld& world, uint32_t i, uint32_t j) {
    if (i >= world.bodies.size() || j >= world.bodies.size()) return FxVec3{0, 0, 0};
    const convex::ContactManifold m = gjk::HullContact(world.bodies[i], world.hulls[i],
                                                       world.bodies[j], world.hulls[j]);
    if (m.count == 0) return FxVec3{0, 0, 0};
    return m.normal;
}

// The settled-shard summary. dynamic/maxSpeed/maxPenetration come from gjk::MeasureHullStack (reused). rested =
// #dynamic shards at rest (|vel| below restBand). faceRestPairs = #dynamic shards whose contact with the FLOOR
// (or another shard) has a FACE-dominant normal (max component >= faceDom·total) — the interlocking proof.
struct FractHullState {
    uint32_t dynamic        = 0;
    uint32_t rested         = 0;
    uint32_t faceRestPairs  = 0;   // # dynamic shards contacting on a face-dominant normal
    fx       maxSpeed       = 0;
    fx       maxPenetration = 0;
    fx       minDynamicY    = 0;   // lowest dynamic center (the pile floor)
};

// MeasureFractureHull(scene, world, restBand): the deterministic settled metrics. faceDom is the face-dominance
// test — a contact normal is "face-like" if its largest |component| is at least 3/4 of |x|+|y|+|z| (an axis
// normal has ratio 1; a 45-degree edge normal ~0.5). Pure integer, fixed order.
inline FractHullState MeasureFractureHull(const FractHullScene& scene, const HullWorld& world, fx restBand) {
    auto absfx = [](fx v) { return v < 0 ? -v : v; };
    FractHullState st;
    const gjk::HullStackMeasure ms = gjk::MeasureHullStack(world);
    st.dynamic = ms.dynamicCount;
    st.maxSpeed = ms.maxSpeed;
    st.maxPenetration = ms.maxPenetration;
    fx minY = (fx)0x7FFFFFFF;
    bool any = false;
    for (size_t i = 0; i < world.bodies.size(); ++i) {
        if (i == scene.floorIndex) continue;
        const fpx::FxBody& b = world.bodies[i];
        if (!convex::IsDynamic(b)) continue;
        any = true;
        if (b.pos.y < minY) minY = b.pos.y;
        if (fpx::FxLength(b.vel) <= restBand) ++st.rested;
        // face-to-face: the contact normal with the floor (the guaranteed resting partner for a settled pile).
        const FxVec3 nrm = FaceToFaceContactNormal(world, (uint32_t)i, scene.floorIndex);
        const fx ax = absfx(nrm.x), ay = absfx(nrm.y), az = absfx(nrm.z);
        const fx tot = ax + ay + az;
        const fx mx = ax > ay ? (ax > az ? ax : az) : (ay > az ? ay : az);
        if (tot > 0 && (int64_t)mx * 4 >= (int64_t)tot * 3) ++st.faceRestPairs;   // dominant axis >= 3/4
    }
    st.minDynamicY = any ? minY : 0;
    return st;
}

// ==================================================================================================
// (F) Lockstep + rollback (the FR5/GJ5 pattern — REUSED VERBATIM via thin named delegates).
// ==================================================================================================
// The DH1 shattered world IS a gjk::HullWorld and its tick IS gjk::StepHullWorld, so the GJ5 lockstep/rollback
// machinery (convex::ConvexCommand + gjk::RunHullLockstep/RunHullRollback, themselves the CX5 machinery reused
// VERBATIM) applies UNCHANGED. These delegates expose DH1's own FR5-shaped entry points. Two peers fed the
// initial spawn world + the command stream re-derive the exact settled convex-shard rubble bit-for-bit; a
// mispredicted shove is corrected by snapshot-restore + re-sim. THE HEADLINE — exact-convex-hull deterministic
// destruction two peers re-derive bit-for-bit (UE5's float Chaos cannot).

using convex::ConvexCommand;
using convex::kConvexCmdAddImpulse;
using convex::kConvexCmdSetAngVel;

inline HullWorld RunFractureHullLockstep(const HullWorld& world0, const convex::ConvexStepConfig& cfg,
                                         const std::vector<convex::ConvexCommand>& commands, uint32_t ticks,
                                         bool* outIdentical = nullptr) {
    return gjk::RunHullLockstep(world0, cfg, commands, ticks, outIdentical);
}

inline HullWorld RunFractureHullRollback(const HullWorld& world0, const convex::ConvexStepConfig& cfg,
                                         const std::vector<convex::ConvexCommand>& authStream,
                                         const std::vector<convex::ConvexCommand>& mispredictStream,
                                         uint32_t ticks, uint32_t rollbackAt,
                                         bool* outCorrectedEqAuthority = nullptr,
                                         bool* outMispredictDiverged = nullptr) {
    return gjk::RunHullRollback(world0, cfg, authStream, mispredictStream, ticks, rollbackAt,
                                outCorrectedEqAuthority, outMispredictDiverged);
}

// ==================================================================================================
// (G) The impact DUST puff (cosmetic but deterministic — integer particles at the fracture surface).
// ==================================================================================================
// On the break, spawn dust PARTICLES at the fracture SURFACE — the midpoints of the SEVERED bonds (fract.h FR3
// bond.midpoint, the Q16.16 lattice-space mean of the two severed fragments' centroids, up-converted to world
// units) — with an impact-derived upward puff velocity (particles::EmitInitialVelocity's +Y-biased fountain
// table, scaled by the impact speed). Deterministic: a fixed ascending scan over the severed bonds, one spawn
// per bond into the SINGLE-THREAD host-ordered pool (particles::EmitParticle — a replay-stable slot + hash). The
// dust is COSMETIC (it does not perturb the sim); its determinism is what is pinned (count + a pool digest).
//
// SpawnFractureDust(bonds, severed, cfg, dustSpeed, capacity): returns a ParticlePool with one alive dust
// particle per severed bond, placed at the bond midpoint (world units) with a hash-seeded upward velocity.
inline particles::ParticlePool SpawnFractureDust(const fract::FractBonds& bonds,
                                                 const std::vector<uint8_t>& severed, const FractHullConfig& cfg,
                                                 fx dustSpeed, uint32_t capacity) {
    particles::ParticlePool pool = particles::InitParticlePool(capacity);
    particles::EmitterConfig ec;
    ec.speed = dustSpeed;
    ec.lifetime = (fx)((int64_t)2 * kOne);   // 2 seconds (cosmetic)
    ec.emitterId = 0xDCu;                     // a fixed dust salt
    for (size_t bi = 0; bi < bonds.bonds.size(); ++bi) {
        if (bi >= severed.size() || !severed[bi]) continue;   // only the severed (fractured) bonds emit dust
        const fract::FractBond& bd = bonds.bonds[bi];
        // bond.midpoint is a Q16.16 LATTICE-space point; up-convert to world units by the cell scale.
        ec.origin = FxVec3{fxmul(bd.midpoint.x, cfg.worldCellSize), fxmul(bd.midpoint.y, cfg.worldCellSize),
                           fxmul(bd.midpoint.z, cfg.worldCellSize)};
        particles::EmitParticle(pool, ec);   // one dust mote at this fracture-surface point (deterministic slot)
    }
    return pool;
}

// DustFnv64(pool): the FNV-1a-64 digest of the dust pool's particle array (the cross-compiler/backend dust pin).
inline uint64_t DustFnv64(const particles::ParticlePool& pool) {
    uint64_t digest = 1469598103934665603ull;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(pool.particles.data());
    const size_t n = pool.particles.size() * sizeof(particles::FxParticle);
    for (size_t i = 0; i < n; ++i) { digest ^= (uint64_t)p[i]; digest *= 1099511628211ull; }
    return digest;
}

// SceneBodiesFnv64(world): the FNV-1a-64 digest of the settled hull-world body array (the full-scene pin — the
// same body POD gjk's lockstep compares). Cross-compiler + cross-backend identical by construction.
inline uint64_t SceneBodiesFnv64(const HullWorld& world) {
    uint64_t digest = 1469598103934665603ull;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(world.bodies.data());
    const size_t n = world.bodies.size() * sizeof(fpx::FxBody);
    for (size_t i = 0; i < n; ++i) { digest ^= (uint64_t)p[i]; digest *= 1099511628211ull; }
    return digest;
}

}  // namespace fhull
}  // namespace hf::sim
