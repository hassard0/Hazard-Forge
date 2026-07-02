#pragma once
// Slice LOD1 — DETERMINISTIC AUTOMATIC LOD GENERATION (Track-S S6 of docs/SUPERIORITY_ROADMAP.md):
// integer quadric-error-metric (Garland-Heckbert) edge-collapse mesh decimation. Pure CPU (header-only,
// no device, no backend symbols), same header-only pattern as engine/render/meshlet.h / cluster_lod.h.
// Namespace hf::render::vg (the virtual-geometry namespace; this header ADDS to it, meshlet.h /
// cluster_lod.h are byte-untouched).
//
// WHY: the virtual-geometry pipeline (DS meshlets -> DV cluster-LOD SelectLod) ships with HAND-AUTHORED
// LODs (three fixed sphere tessellations). LOD1 generates LOD1/LOD2 meshes from ANY LOD0 automatically,
// with a PINNED cross-platform digest — a deterministic LOD build is something UE5's Nanite build
// pipeline explicitly is not (its results vary across versions/machines).
//
// --- THE DETERMINISM DISCIPLINE (the meshlet.h Morton precedent, extended) ---
// 1. QUANTIZE FIRST: input float positions are snapped to a per-mesh normalized integer lattice
//    (kLodGenLatticeBits = 12 bits/axis, coords in [0, 4095]) by ROUND-TO-NEAREST — (p-lo)/ext*4095
//    + 0.5, floored (NOT the meshlet floor-quantize: flooring splits the UV-sphere seam, whose
//    duplicated columns differ by ~1e-7 and land in ADJACENT cells exactly at a cell boundary;
//    rounding welds them and halves the quantization error). ALL decimation math after this point
//    is integer/fixed-point
//    (int64 accumulators), so the collapse sequence is bit-stable across compilers and platforms.
//    Vertices that land on the SAME OR AN ADJACENT lattice point (Chebyshev distance <= 1) are
//    WELDED to the earliest-seen representative (deterministic: input order + min-id tie-break;
//    welded keys ALIAS to their representative so epsilon-chains keep welding). The one-cell reach
//    is REQUIRED, not cosmetic: coincident-in-spirit vertices (the UV-sphere seam columns, the
//    bottom pole fan whose float sin(pi) != 0 scatters +-1e-7 around an EXACT half-cell rounding
//    boundary) can land in adjacent cells no matter how the quantizer rounds. Consequence
//    (documented): the lattice is the mesh's RESOLUTION FLOOR — features smaller than one cell
//    (extent/4095) weld away. Triangles that become degenerate on the lattice (repeated welded
//    vertex, or zero integer cross product) are dropped and counted.
// 2. INTEGER QUADRICS: per-face plane normals are the integer cross product of the lattice edge
//    vectors, NORMALIZED TO A FIXED-POINT SCALE (kLodGenNormalScale = 512 = 2^9) via an integer
//    sqrt + truncating divide — so every face contributes a comparably-weighted plane, and the
//    accumulated quadric (A = n n^T, b = d n, c = d^2; all int64) never sees a float.
//    OVERFLOW BUDGET (documented, not just hoped): lattice L <= 4095 (2^12), normal scale S = 2^9:
//      |n_fx| <= S, |d| <= 3*S*L < 2^23, per-face c = d^2 < 2^46. Quadric error evaluated at a
//      lattice point is < K * 2^47 where K is the number of face-planes accumulated into the quadric
//      (each face feeds 3 vertices, so K <= 3*F). K <= 2^15 (i.e. F <= ~10k triangles fully merged
//      into ONE vertex — far beyond any real collapse chain) keeps every intermediate < 2^62, inside
//      int64. All engine fixtures are <= 3k faces.
// 3. TOTAL-ORDER PRIORITY: collapse candidates are ordered by (cost, minVertexIndex, maxVertexIndex)
//    — integers only, no float compares, no hash iteration (NO std::unordered_*). The queue is a
//    binary heap (std::priority_queue) with that total-order comparator + LAZY invalidation via
//    per-vertex change stamps: a popped entry whose endpoint stamps are stale is discarded (a fresh
//    entry was pushed when the neighborhood changed). Deterministic by construction.
// 4. COLLAPSE TARGET (v1): the optimal-point 4x4 solve is SKIPPED (it needs a rational inverse);
//    instead the error is evaluated at the three candidates {v0, v1, midpoint} (midpoint = the
//    component-wise (a+b)>>1 on the non-negative lattice) and the FIRST strictly-smallest wins
//    (preference order v0, v1, mid — a documented deterministic tie-break).
// 5. VALIDITY GUARDS (all integer):
//      * boundary: collapses touching a BOUNDARY VERTEX are forbidden outright (v1 choice — boundary
//        edges are those with face-incidence != 2 on the welded input; open meshes keep their rim
//        EXACTLY, at the cost of decimation ratio near the rim).
//      * manifold link condition: the common vertex-neighbors of the edge's endpoints must be EXACTLY
//        the two opposite vertices of the two shared faces (rejects pinches/fold-throughs).
//      * flip test: every surviving incident face must keep sign(dot(crossBefore, crossAfter)) > 0
//        and a non-zero after-normal (integer orientation test — no float).
//      * duplicate faces: the remapped incident faces must stay pairwise distinct as vertex SETS.
// 6. ATTRIBUTE CARRY (v1, documented): decimation operates on positions+indices ONLY. Each welded
//    vertex remembers its LOWEST-index source vertex; the output scene::Vertex copies that source
//    vertex's color/uv/normal/tangent and overwrites pos with the dequantized lattice position (the
//    only float step, at the I/O boundary — it never feeds back into any decision). This is the
//    "nearest-source-vertex" carry; per-attribute quadrics are a later refinement.
// 7. GEOMETRIC ERROR (v1, documented): the conservative object-space error reported per LOD is
//    (maxAccumulatedVertexDisplacementInLatticeUnits + 1) * maxAxisStep, accumulated across the
//    LOD chain (LOD2 continues from LOD1, so errors are non-decreasing — the SelectLod contract).
//    Displacement is tracked per vertex in integer lattice units (isqrt, rounded UP). This is an
//    upper-estimate heuristic, NOT a certified Hausdorff bound; the +1 covers the quantization snap.
//
// DIGEST: LodGenDigest hand-LE-serializes (vertexCount, triCount, qpos[], indices[]) and feeds
// net::DigestBytes (the engine-wide FNV-1a-64) — a PURE-INTEGER pinned golden, identical on
// MSVC / Windows-clang / Mac-clang / any IEEE platform.
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/mtl::) symbols. No float in the collapse decision path,
// no RNG, no clock, no std::unordered_*. Bounds-checked at the public entry points.

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <map>
#include <queue>
#include <span>
#include <vector>

#include "math/math.h"
#include "net/session.h"        // hf::net::DigestBytes (FNV-1a-64, the pinned-golden currency)
#include "render/cluster_lod.h" // LodMeshes/BuildLodMeshes/kNumLods (the LOD consumer this feeds)
#include "scene/mesh.h"
#include "scene/vertex.h"

namespace hf::render::vg {

// --- Lattice + fixed-point scales (see the overflow budget in the header banner). -------------------
static constexpr uint32_t kLodGenLatticeBits = 12;                            // 4096 steps/axis
static constexpr int32_t  kLodGenLatticeMax  = (1 << kLodGenLatticeBits) - 1; // 4095
static constexpr int64_t  kLodGenNormalScale = 512;                           // 2^9 plane-normal scale

// --- The decimation I/O view: quantized lattice positions + triangle indices + the dequant frame. ---
// qpos is 3 int32 per vertex (each in [0, kLodGenLatticeMax]); indices is 3 uint32 per triangle.
// srcVert[i] is the LOWEST original source-vertex index that welded into vertex i (attribute carry).
// Object-space position of vertex i, axis k: origin[k] + qpos[3i+k] * step[k] (float, I/O only).
struct LodGenMesh {
    std::vector<int32_t>  qpos;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> srcVert;
    float origin[3] = {0, 0, 0};
    float step[3]   = {0, 0, 0};
    uint32_t weldDropped = 0;  // input triangles dropped as lattice-degenerate during QuantizeWeld
};

// --- Decimation statistics (the honest quality numbers; all integer, all pinnable). -----------------
struct DecimateStats {
    uint32_t collapses       = 0;  // committed collapses
    uint64_t maxQuadricError = 0;  // max committed collapse cost (integer QEM units — the quality pin)
    uint32_t maxDispLattice  = 0;  // max accumulated vertex displacement (lattice units, rounded up)
};

// --- Integer sqrt (uint64, floor). Deterministic bit-by-bit method, no float. -----------------------
inline uint64_t LodGenIsqrt(uint64_t x) {
    uint64_t r = 0, bit = 1ull << 62;
    while (bit > x) bit >>= 2;
    while (bit != 0) {
        if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; }
        else              { r >>= 1; }
        bit >>= 2;
    }
    return r;
}

// --- The integer quadric: E(v) = v^T A v + 2 b.v + c over the accumulated fixed-scale planes. -------
// A is symmetric (6 unique entries: xx,xy,xz,yy,yz,zz), b 3 entries, c scalar — all int64 (budgeted).
// Each face contributes the EXACT integer outer product of its scale-S normal, so Eval at a lattice
// point is the exact integer sum of (n.v + d)^2 over the accumulated planes: always >= 0.
struct LodQuadric {
    int64_t a[6] = {0, 0, 0, 0, 0, 0};  // xx, xy, xz, yy, yz, zz
    int64_t b[3] = {0, 0, 0};
    int64_t c    = 0;
    void Add(const LodQuadric& o) {
        for (int k = 0; k < 6; ++k) a[k] += o.a[k];
        for (int k = 0; k < 3; ++k) b[k] += o.b[k];
        c += o.c;
    }
    // Exact integer quadric error at lattice point (x,y,z). Non-negative by construction (see banner).
    uint64_t Eval(int64_t x, int64_t y, int64_t z) const {
        int64_t e = a[0] * x * x + a[3] * y * y + a[5] * z * z
                  + 2 * (a[1] * x * y + a[2] * x * z + a[4] * y * z)
                  + 2 * (b[0] * x + b[1] * y + b[2] * z) + c;
        return (e < 0) ? 0ull : (uint64_t)e;  // exact math is >= 0; clamp defends the pin anyway
    }
};

// The plane quadric of one lattice triangle (p0,p1,p2). Returns false if the integer cross product is
// zero (lattice-degenerate face — contributes nothing). The normal is normalized to kLodGenNormalScale
// by an integer isqrt + truncating divide (|n_fx| <= S), d = -(n_fx . p0).
inline bool LodFaceQuadric(const int32_t* p0, const int32_t* p1, const int32_t* p2, LodQuadric& out) {
    const int64_t e1[3] = {(int64_t)p1[0] - p0[0], (int64_t)p1[1] - p0[1], (int64_t)p1[2] - p0[2]};
    const int64_t e2[3] = {(int64_t)p2[0] - p0[0], (int64_t)p2[1] - p0[1], (int64_t)p2[2] - p0[2]};
    const int64_t cx = e1[1] * e2[2] - e1[2] * e2[1];
    const int64_t cy = e1[2] * e2[0] - e1[0] * e2[2];
    const int64_t cz = e1[0] * e2[1] - e1[1] * e2[0];
    const uint64_t n2 = (uint64_t)(cx * cx) + (uint64_t)(cy * cy) + (uint64_t)(cz * cz);
    if (n2 == 0) return false;
    const int64_t m = (int64_t)LodGenIsqrt(n2);  // >= 1
    const int64_t n[3] = {cx * kLodGenNormalScale / m, cy * kLodGenNormalScale / m,
                          cz * kLodGenNormalScale / m};
    const int64_t d = -(n[0] * p0[0] + n[1] * p0[1] + n[2] * p0[2]);
    out.a[0] = n[0] * n[0]; out.a[1] = n[0] * n[1]; out.a[2] = n[0] * n[2];
    out.a[3] = n[1] * n[1]; out.a[4] = n[1] * n[2]; out.a[5] = n[2] * n[2];
    out.b[0] = n[0] * d;    out.b[1] = n[1] * d;    out.b[2] = n[2] * d;
    out.c    = d * d;
    return true;
}

// --- QuantizeWeld: float positions -> the integer lattice, welded + degenerate-filtered. ------------
// AABB over ALL input vertices; per-axis ROUND-TO-NEAREST into [0, kLodGenLatticeMax] (see the banner:
// (p-lo)/ext * 4095 + 0.5, floored, clamped; zero-extent axis -> 0). Vertices on the same or an
// ADJACENT lattice point (Chebyshev <= 1; see the banner for why one-cell reach is required) weld to
// ONE vertex (the earliest-seen representative; first = lowest original index -> srcVert; the welded
// vertex keeps the REPRESENTATIVE's lattice point). Triangles with a repeated welded vertex OR a
// zero integer cross product are dropped (weldDropped counts them — for scene::SphereGeometry these
// are exactly the 2*segments pole-fan degenerates + nothing else). Only REFERENCED welded vertices
// are kept (compacted in first-appearance order of the weld scan, i.e. ascending lowest-source-index
// order). Deterministic: the weld map is an ORDERED std::map, the scan order is the input order, the
// neighbor probe order is fixed, ties break to the MINIMUM welded id.
inline LodGenMesh QuantizeWeld(std::span<const scene::Vertex> verts,
                               std::span<const uint32_t> indices) {
    LodGenMesh out;
    const size_t V = verts.size();
    const uint32_t T = (uint32_t)(indices.size() / 3);  // trailing 1-2 indices ignored (meshlet rule)
    if (V == 0 || T == 0) return out;

    // AABB over all vertices.
    float lo[3] = {verts[0].pos[0], verts[0].pos[1], verts[0].pos[2]};
    float hi[3] = {lo[0], lo[1], lo[2]};
    for (size_t i = 1; i < V; ++i)
        for (int k = 0; k < 3; ++k) {
            const float p = verts[i].pos[k];
            if (p < lo[k]) lo[k] = p;
            if (p > hi[k]) hi[k] = p;
        }
    float ext[3];
    for (int k = 0; k < 3; ++k) {
        ext[k] = hi[k] - lo[k];
        out.origin[k] = lo[k];
        out.step[k] = (ext[k] > 0.0f) ? ext[k] / (float)kLodGenLatticeMax : 0.0f;
    }
    auto quant = [&](float p, int k) -> int32_t {
        if (ext[k] <= 0.0f) return 0;
        float f = (p - lo[k]) / ext[k] * (float)kLodGenLatticeMax + 0.5f;
        if (f < 0.0f) f = 0.0f;
        int32_t q = (int32_t)f;  // floor of (scaled + 0.5) == round-to-nearest (f >= 0)
        if (q > kLodGenLatticeMax) q = kLodGenLatticeMax;
        return q;
    };

    // Weld: ordered map lattice-point -> welded index. Exact hit first; else the MINIMUM welded id
    // among the 26 adjacent cells (fixed probe order + min-id tie-break = deterministic); else a new
    // representative. A neighbor-welded key is INSERTED as an alias so epsilon-chains keep welding.
    std::vector<uint32_t> weldOf(V);
    std::map<std::array<int32_t, 3>, uint32_t> weld;
    std::vector<std::array<int32_t, 3>> weldedPos;
    std::vector<uint32_t> weldedSrc;
    for (size_t i = 0; i < V; ++i) {
        const std::array<int32_t, 3> q{quant(verts[i].pos[0], 0), quant(verts[i].pos[1], 1),
                                       quant(verts[i].pos[2], 2)};
        auto it = weld.find(q);
        if (it != weld.end()) { weldOf[i] = it->second; continue; }
        uint32_t hit = 0xFFFFFFFFu;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const auto n = weld.find({q[0] + dx, q[1] + dy, q[2] + dz});
                    if (n != weld.end() && n->second < hit) hit = n->second;
                }
        if (hit != 0xFFFFFFFFu) {
            weld.emplace(q, hit);  // alias key -> the representative
            weldOf[i] = hit;
            continue;
        }
        const uint32_t id = (uint32_t)weldedPos.size();
        weld.emplace(q, id);
        weldedPos.push_back(q);
        weldedSrc.push_back((uint32_t)i);
        weldOf[i] = id;
    }

    // Faces on welded ids: drop repeated-vertex + zero-lattice-area triangles.
    std::vector<uint32_t> faces;
    faces.reserve((size_t)T * 3);
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t i0 = indices[3 * t + 0], i1 = indices[3 * t + 1], i2 = indices[3 * t + 2];
        if (i0 >= V || i1 >= V || i2 >= V) { ++out.weldDropped; continue; }  // bounds-checked
        const uint32_t w0 = weldOf[i0], w1 = weldOf[i1], w2 = weldOf[i2];
        if (w0 == w1 || w1 == w2 || w0 == w2) { ++out.weldDropped; continue; }
        LodQuadric dummy;
        if (!LodFaceQuadric(weldedPos[w0].data(), weldedPos[w1].data(), weldedPos[w2].data(),
                            dummy)) { ++out.weldDropped; continue; }
        faces.push_back(w0); faces.push_back(w1); faces.push_back(w2);
    }

    // Compact to referenced welded vertices (ascending welded id = first-appearance order).
    std::vector<uint32_t> remap(weldedPos.size(), 0xFFFFFFFFu);
    for (uint32_t idx : faces) remap[idx] = 1u;
    uint32_t next = 0;
    for (size_t i = 0; i < remap.size(); ++i)
        if (remap[i] == 1u) remap[i] = next++;
        else remap[i] = 0xFFFFFFFFu;
    out.qpos.reserve((size_t)next * 3);
    out.srcVert.reserve(next);
    for (size_t i = 0; i < weldedPos.size(); ++i)
        if (remap[i] != 0xFFFFFFFFu) {
            out.qpos.push_back(weldedPos[i][0]);
            out.qpos.push_back(weldedPos[i][1]);
            out.qpos.push_back(weldedPos[i][2]);
            out.srcVert.push_back(weldedSrc[i]);
        }
    out.indices.reserve(faces.size());
    for (uint32_t idx : faces) out.indices.push_back(remap[idx]);
    return out;
}

// --- The pinned-golden digest: hand-LE-serialize (V, T, qpos[], indices[]) -> FNV-1a-64. ------------
inline uint64_t LodGenDigest(const LodGenMesh& m) {
    std::vector<unsigned char> b;
    b.reserve(8 + m.qpos.size() * 4 + m.indices.size() * 4);
    auto putU32 = [&](uint32_t v) {
        b.push_back((unsigned char)(v & 0xFFu));
        b.push_back((unsigned char)((v >> 8) & 0xFFu));
        b.push_back((unsigned char)((v >> 16) & 0xFFu));
        b.push_back((unsigned char)((v >> 24) & 0xFFu));
    };
    putU32((uint32_t)(m.qpos.size() / 3));
    putU32((uint32_t)(m.indices.size() / 3));
    for (int32_t q : m.qpos) putU32((uint32_t)q);
    for (uint32_t i : m.indices) putU32(i);
    return net::DigestBytes(b.data(), b.size());
}

// --- DecimateMesh: integer QEM edge-collapse to (at most) targetTris triangles. ---------------------
// Collapses stop when the alive-triangle count reaches targetTris, or when NO remaining collapse
// passes the validity guards (the result may then hold ABOVE the target — e.g. a welded cube resists
// most collapses; the tests pin whatever it honestly reaches). Every step of the decision path is
// integer (see the header banner). Deterministic: the same input always yields the byte-identical
// output. Returns the compacted mesh (surviving vertices in ascending pre-collapse order, faces in
// original order); the dequant frame is copied through, so chained calls stay in ONE lattice.
inline LodGenMesh DecimateMesh(const LodGenMesh& in, uint32_t targetTris,
                               DecimateStats* statsOut = nullptr) {
    DecimateStats stats;
    const uint32_t V = (uint32_t)(in.qpos.size() / 3);
    const uint32_t F = (uint32_t)(in.indices.size() / 3);
    LodGenMesh out = in;  // frame + weldDropped carried; geometry rebuilt below
    if (V == 0 || F == 0 || F <= targetTris) { if (statsOut) *statsOut = stats; return out; }

    // Working state.
    std::vector<std::array<int32_t, 3>> pos(V);
    for (uint32_t i = 0; i < V; ++i)
        pos[i] = {in.qpos[3 * i + 0], in.qpos[3 * i + 1], in.qpos[3 * i + 2]};
    std::vector<std::array<uint32_t, 3>> face(F);
    for (uint32_t f = 0; f < F; ++f) {
        face[f] = {in.indices[3 * f + 0], in.indices[3 * f + 1], in.indices[3 * f + 2]};
        for (int e = 0; e < 3; ++e)
            if (face[f][e] >= V) { if (statsOut) *statsOut = stats; return out; }  // bounds-checked
    }
    std::vector<uint8_t>  aliveV(V, 1), aliveF(F, 1);
    std::vector<uint32_t> stamp(V, 0);
    std::vector<uint64_t> disp(V, 0);  // accumulated displacement, lattice units (rounded up)

    // Per-vertex quadrics + adjacency.
    std::vector<LodQuadric> quad(V);
    std::vector<std::vector<uint32_t>> vfaces(V);
    for (uint32_t f = 0; f < F; ++f) {
        LodQuadric q;
        if (LodFaceQuadric(pos[face[f][0]].data(), pos[face[f][1]].data(), pos[face[f][2]].data(), q))
            for (int e = 0; e < 3; ++e) quad[face[f][e]].Add(q);
        for (int e = 0; e < 3; ++e) vfaces[face[f][e]].push_back(f);
    }

    // Boundary detection on the input: an edge with face-incidence != 2 marks both verts boundary.
    std::vector<uint8_t> boundary(V, 0);
    {
        std::vector<uint64_t> ekeys;
        ekeys.reserve((size_t)F * 3);
        for (uint32_t f = 0; f < F; ++f)
            for (int e = 0; e < 3; ++e) {
                uint32_t x = face[f][e], y = face[f][(e + 1) % 3];
                if (x > y) std::swap(x, y);
                ekeys.push_back(((uint64_t)x << 32) | y);
            }
        std::sort(ekeys.begin(), ekeys.end());
        for (size_t i = 0; i < ekeys.size();) {
            size_t j = i;
            while (j < ekeys.size() && ekeys[j] == ekeys[i]) ++j;
            if (j - i != 2) {
                boundary[(uint32_t)(ekeys[i] >> 32)] = 1;
                boundary[(uint32_t)(ekeys[i] & 0xFFFFFFFFu)] = 1;
            }
            i = j;
        }
    }

    // The lazy heap: min on (cost, a, b, sa, sb) — a TOTAL order over integers.
    struct Ent {
        uint64_t cost; uint32_t a, b, sa, sb; std::array<int32_t, 3> t;
        bool operator>(const Ent& o) const {
            if (cost != o.cost) return cost > o.cost;
            if (a != o.a) return a > o.a;
            if (b != o.b) return b > o.b;
            if (sa != o.sa) return sa > o.sa;
            return sb > o.sb;
        }
    };
    std::priority_queue<Ent, std::vector<Ent>, std::greater<Ent>> heap;

    // Cost of collapsing (a,b): quadric sum evaluated at {pos[a], pos[b], mid}; first strict min wins.
    auto edgeCost = [&](uint32_t a, uint32_t b, std::array<int32_t, 3>& tOut) -> uint64_t {
        LodQuadric q = quad[a];
        q.Add(quad[b]);
        const std::array<int32_t, 3> mid = {(pos[a][0] + pos[b][0]) >> 1,
                                            (pos[a][1] + pos[b][1]) >> 1,
                                            (pos[a][2] + pos[b][2]) >> 1};
        const std::array<int32_t, 3>* cand[3] = {&pos[a], &pos[b], &mid};
        uint64_t best = ~0ull;
        for (int k = 0; k < 3; ++k) {
            const uint64_t e = q.Eval((*cand[k])[0], (*cand[k])[1], (*cand[k])[2]);
            if (e < best) { best = e; tOut = *cand[k]; }
        }
        return best;
    };
    auto pushEdge = [&](uint32_t a, uint32_t b) {
        if (a > b) std::swap(a, b);
        if (boundary[a] || boundary[b]) return;  // v1: boundary verts never collapse
        Ent e;
        e.cost = edgeCost(a, b, e.t);
        e.a = a; e.b = b; e.sa = stamp[a]; e.sb = stamp[b];
        heap.push(e);
    };

    // Seed: every unique edge of the input.
    {
        std::vector<uint64_t> ekeys;
        ekeys.reserve((size_t)F * 3);
        for (uint32_t f = 0; f < F; ++f)
            for (int e = 0; e < 3; ++e) {
                uint32_t x = face[f][e], y = face[f][(e + 1) % 3];
                if (x > y) std::swap(x, y);
                ekeys.push_back(((uint64_t)x << 32) | y);
            }
        std::sort(ekeys.begin(), ekeys.end());
        ekeys.erase(std::unique(ekeys.begin(), ekeys.end()), ekeys.end());
        for (uint64_t k : ekeys) pushEdge((uint32_t)(k >> 32), (uint32_t)(k & 0xFFFFFFFFu));
    }

    // Helpers over the alive adjacency.
    auto aliveFacesOf = [&](uint32_t v, std::vector<uint32_t>& tmp) {
        tmp.clear();
        for (uint32_t f : vfaces[v]) if (aliveF[f]) tmp.push_back(f);
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    };
    auto crossOf = [&](const std::array<int32_t, 3>& p0, const std::array<int32_t, 3>& p1,
                       const std::array<int32_t, 3>& p2, int64_t c[3]) {
        const int64_t e1[3] = {(int64_t)p1[0] - p0[0], (int64_t)p1[1] - p0[1], (int64_t)p1[2] - p0[2]};
        const int64_t e2[3] = {(int64_t)p2[0] - p0[0], (int64_t)p2[1] - p0[1], (int64_t)p2[2] - p0[2]};
        c[0] = e1[1] * e2[2] - e1[2] * e2[1];
        c[1] = e1[2] * e2[0] - e1[0] * e2[2];
        c[2] = e1[0] * e2[1] - e1[1] * e2[0];
    };

    uint32_t aliveTris = F;
    std::vector<uint32_t> fa, fb, shared, other, nbrA, nbrB, common, expectOpp;
    while (aliveTris > targetTris && !heap.empty()) {
        const Ent e = heap.top();
        heap.pop();
        const uint32_t a = e.a, b = e.b;
        if (!aliveV[a] || !aliveV[b]) continue;
        if (stamp[a] != e.sa || stamp[b] != e.sb) continue;  // stale (a fresh entry exists)

        // Incident faces; shared = faces containing BOTH endpoints (must be exactly 2: interior edge).
        aliveFacesOf(a, fa);
        aliveFacesOf(b, fb);
        shared.clear(); other.clear();
        {
            std::vector<uint32_t> uni = fa;
            uni.insert(uni.end(), fb.begin(), fb.end());
            std::sort(uni.begin(), uni.end());
            uni.erase(std::unique(uni.begin(), uni.end()), uni.end());
            for (uint32_t f : uni) {
                const auto& fv = face[f];
                const bool hasA = (fv[0] == a || fv[1] == a || fv[2] == a);
                const bool hasB = (fv[0] == b || fv[1] == b || fv[2] == b);
                if (hasA && hasB) shared.push_back(f);
                else other.push_back(f);
            }
        }
        if (shared.size() != 2) continue;  // the edge no longer exists, or is non-manifold: skip

        // LINK CONDITION: common vertex-neighbors of a,b == the two opposite verts of the shared faces.
        auto nbrsOf = [&](uint32_t v, const std::vector<uint32_t>& fs, std::vector<uint32_t>& nb) {
            nb.clear();
            for (uint32_t f : fs)
                for (int k = 0; k < 3; ++k)
                    if (face[f][k] != v) nb.push_back(face[f][k]);
            std::sort(nb.begin(), nb.end());
            nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        };
        nbrsOf(a, fa, nbrA);
        nbrsOf(b, fb, nbrB);
        common.clear();
        std::set_intersection(nbrA.begin(), nbrA.end(), nbrB.begin(), nbrB.end(),
                              std::back_inserter(common));
        expectOpp.clear();
        for (uint32_t f : shared)
            for (int k = 0; k < 3; ++k)
                if (face[f][k] != a && face[f][k] != b) expectOpp.push_back(face[f][k]);
        std::sort(expectOpp.begin(), expectOpp.end());
        expectOpp.erase(std::unique(expectOpp.begin(), expectOpp.end()), expectOpp.end());
        if (common != expectOpp || expectOpp.size() != 2) continue;  // pinch: reject

        // FLIP / DEGENERATE / DUPLICATE guards over the surviving (non-shared) incident faces.
        const std::array<int32_t, 3>& t = e.t;
        bool ok = true;
        std::vector<std::array<uint32_t, 3>> newKeys;
        newKeys.reserve(other.size());
        for (uint32_t f : other) {
            std::array<std::array<int32_t, 3>, 3> before, after;
            std::array<uint32_t, 3> key{};
            for (int k = 0; k < 3; ++k) {
                const uint32_t vId = face[f][k];
                before[k] = pos[vId];
                const uint32_t nId = (vId == b) ? a : vId;
                after[k] = (vId == a || vId == b) ? t : pos[vId];
                key[k] = nId;
            }
            int64_t nB[3], nA[3];
            crossOf(before[0], before[1], before[2], nB);
            crossOf(after[0], after[1], after[2], nA);
            if (nA[0] == 0 && nA[1] == 0 && nA[2] == 0) { ok = false; break; }  // degenerate
            const int64_t d = nB[0] * nA[0] + nB[1] * nA[1] + nB[2] * nA[2];
            if (d <= 0) { ok = false; break; }  // orientation flip
            std::sort(key.begin(), key.end());
            newKeys.push_back(key);
        }
        if (ok) {
            std::sort(newKeys.begin(), newKeys.end());
            for (size_t k = 1; k < newKeys.size(); ++k)
                if (newKeys[k] == newKeys[k - 1]) { ok = false; break; }  // duplicate face (fold-over)
        }
        if (!ok) continue;

        // ---- COMMIT: merge b into a at target t. ----
        const std::array<int32_t, 3> pa = pos[a], pb = pos[b];
        pos[a] = t;
        quad[a].Add(quad[b]);
        aliveV[b] = 0;
        for (uint32_t f : shared) { aliveF[f] = 0; --aliveTris; }
        for (uint32_t f : fb)
            if (aliveF[f])
                for (int k = 0; k < 3; ++k)
                    if (face[f][k] == b) face[f][k] = a;
        {
            std::vector<uint32_t> merged;
            merged.reserve(fa.size() + fb.size());
            for (uint32_t f : fa) if (aliveF[f]) merged.push_back(f);
            for (uint32_t f : fb) if (aliveF[f]) merged.push_back(f);
            std::sort(merged.begin(), merged.end());
            merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            vfaces[a] = std::move(merged);
            vfaces[b].clear();
        }
        auto ceilDist = [&](const std::array<int32_t, 3>& p, const std::array<int32_t, 3>& q) {
            const int64_t dx = (int64_t)p[0] - q[0], dy = (int64_t)p[1] - q[1],
                          dz = (int64_t)p[2] - q[2];
            return LodGenIsqrt((uint64_t)(dx * dx + dy * dy + dz * dz)) + 1;  // round UP (conservative)
        };
        const uint64_t da = disp[a] + ceilDist(t, pa), db = disp[b] + ceilDist(t, pb);
        disp[a] = (da > db) ? da : db;
        if (disp[a] > stats.maxDispLattice)
            stats.maxDispLattice = (disp[a] > 0xFFFFFFFFull) ? 0xFFFFFFFFu : (uint32_t)disp[a];
        if (e.cost > stats.maxQuadricError) stats.maxQuadricError = e.cost;
        ++stats.collapses;
        ++stamp[a];
        // Re-push the edges around the kept vertex (fresh stamps; older entries become stale).
        {
            std::vector<uint64_t> around;
            for (uint32_t f : vfaces[a])
                for (int k = 0; k < 3; ++k) {
                    uint32_t x = face[f][k], y = face[f][(k + 1) % 3];
                    if (x != a && y != a) continue;
                    if (x > y) std::swap(x, y);
                    around.push_back(((uint64_t)x << 32) | y);
                }
            std::sort(around.begin(), around.end());
            around.erase(std::unique(around.begin(), around.end()), around.end());
            for (uint64_t k : around) pushEdge((uint32_t)(k >> 32), (uint32_t)(k & 0xFFFFFFFFu));
        }
    }

    // ---- Compact: surviving vertices ascending, faces in original order. ----
    std::vector<uint32_t> remap(V, 0xFFFFFFFFu);
    uint32_t next = 0;
    for (uint32_t i = 0; i < V; ++i) if (aliveV[i]) remap[i] = next++;
    out.qpos.clear(); out.srcVert.clear(); out.indices.clear();
    out.qpos.reserve((size_t)next * 3);
    out.srcVert.reserve(next);
    for (uint32_t i = 0; i < V; ++i)
        if (aliveV[i]) {
            out.qpos.push_back(pos[i][0]);
            out.qpos.push_back(pos[i][1]);
            out.qpos.push_back(pos[i][2]);
            out.srcVert.push_back(in.srcVert.empty() ? i : in.srcVert[i]);
        }
    out.indices.reserve((size_t)aliveTris * 3);
    for (uint32_t f = 0; f < F; ++f)
        if (aliveF[f])
            for (int k = 0; k < 3; ++k) out.indices.push_back(remap[face[f][k]]);
    if (statsOut) *statsOut = stats;
    return out;
}

// --- ToMeshGeometry: the decimated lattice mesh back to the engine vertex format. -------------------
// Position = the dequantized lattice point (origin + q*step — the ONLY float step, presentation/IO).
// All other attributes (color/uv/normal/tangent) are copied from the recorded lowest-index source
// vertex (the v1 nearest-source carry; see the banner). srcVerts must be the ORIGINAL vertex array
// the LodGenMesh chain started from.
inline scene::MeshGeometry ToMeshGeometry(const LodGenMesh& m,
                                          std::span<const scene::Vertex> srcVerts) {
    scene::MeshGeometry g;
    const uint32_t V = (uint32_t)(m.qpos.size() / 3);
    g.verts.reserve(V);
    for (uint32_t i = 0; i < V; ++i) {
        scene::Vertex v{};
        if (i < m.srcVert.size() && m.srcVert[i] < srcVerts.size()) v = srcVerts[m.srcVert[i]];
        for (int k = 0; k < 3; ++k)
            v.pos[k] = m.origin[k] + (float)m.qpos[3 * i + k] * m.step[k];
        g.verts.push_back(v);
    }
    g.indices = m.indices;
    return g;
}

// Convenience overload: MeshGeometry in -> quantize+weld -> decimate -> MeshGeometry out.
inline scene::MeshGeometry DecimateMesh(const scene::MeshGeometry& in, uint32_t targetTris,
                                        DecimateStats* statsOut = nullptr) {
    const LodGenMesh q = QuantizeWeld(
        std::span<const scene::Vertex>(in.verts.data(), in.verts.size()),
        std::span<const uint32_t>(in.indices.data(), in.indices.size()));
    const LodGenMesh d = DecimateMesh(q, targetTris, statsOut);
    return ToMeshGeometry(d, std::span<const scene::Vertex>(in.verts.data(), in.verts.size()));
}

// --- BuildAutoLods: LOD0 = the input; LOD1 ~50%, LOD2 ~25% of the WELDED triangle count. ------------
// The kNumLods=3 convention of cluster_lod.h. LOD2 continues decimating LOD1's lattice mesh (ONE
// quantization, errors accumulate monotonically). triCount[0]/digest[0] describe the QUANTIZED WELDED
// input (the decimation currency — for a UV sphere this is the seam-welded, pole-degenerate-free
// count, slightly below the raw index-buffer count; documented). geometricError is the v1 heuristic
// bound from the banner (LOD0 = 0; non-decreasing in n — the SelectLod contract).
struct AutoLods {
    std::array<scene::MeshGeometry, kNumLods> geos;   // [0] = the input, verbatim
    std::array<float, kNumLods>    geometricError{};  // conservative object-space deviation (LOD0 = 0)
    std::array<uint64_t, kNumLods> digest{};          // pinned integer digests ([0] = welded input)
    std::array<uint64_t, kNumLods> maxQuadricError{}; // the quality pins ([0] = 0)
    std::array<uint32_t, kNumLods> maxDispLattice{};  // per-pass displacement bounds ([0] = 0)
    std::array<uint32_t, kNumLods> triCount{};        // welded/decimated triangle counts
};
inline AutoLods BuildAutoLods(const scene::MeshGeometry& in) {
    AutoLods out;
    out.geos[0] = in;
    const std::span<const scene::Vertex> vs(in.verts.data(), in.verts.size());
    const LodGenMesh q0 = QuantizeWeld(vs, std::span<const uint32_t>(in.indices.data(),
                                                                     in.indices.size()));
    const uint32_t F0 = (uint32_t)(q0.indices.size() / 3);
    out.triCount[0] = F0;
    out.digest[0] = LodGenDigest(q0);

    DecimateStats s1{}, s2{};
    const LodGenMesh q1 = DecimateMesh(q0, F0 / 2, &s1);
    const LodGenMesh q2 = DecimateMesh(q1, F0 / 4, &s2);
    out.geos[1] = ToMeshGeometry(q1, vs);
    out.geos[2] = ToMeshGeometry(q2, vs);
    out.triCount[1] = (uint32_t)(q1.indices.size() / 3);
    out.triCount[2] = (uint32_t)(q2.indices.size() / 3);
    out.digest[1] = LodGenDigest(q1);
    out.digest[2] = LodGenDigest(q2);
    out.maxQuadricError[1] = s1.maxQuadricError;
    out.maxQuadricError[2] = s2.maxQuadricError;
    out.maxDispLattice[1] = s1.maxDispLattice;
    out.maxDispLattice[2] = s2.maxDispLattice;

    float stepMax = q0.step[0];
    if (q0.step[1] > stepMax) stepMax = q0.step[1];
    if (q0.step[2] > stepMax) stepMax = q0.step[2];
    out.geometricError[0] = 0.0f;
    out.geometricError[1] = (float)(s1.maxDispLattice + 1u) * stepMax;
    out.geometricError[2] = out.geometricError[1] + (float)(s2.maxDispLattice + 1u) * stepMax;
    return out;
}

// --- BuildAutoLodMeshes: plug the generated LODs into the EXISTING cluster_lod consumer. ------------
// Runs the byte-untouched BuildLodMeshes (DS decomposition + combined buffers) over the generated
// geometries, then overrides the per-LOD geometric errors with the generated conservative bounds
// (BuildLodMeshes' tess-pair sagitta formula is the HAND-AUTHORED-sphere path; passing {0,0} pairs
// yields 0, which we replace). The result feeds SelectLod exactly like the hand-authored LODs.
inline LodMeshes BuildAutoLodMeshes(const AutoLods& al) {
    const std::array<std::pair<uint32_t, uint32_t>, kNumLods> zeroTess{};  // errors overridden below
    LodMeshes lm = BuildLodMeshes(al.geos, zeroTess);
    for (uint32_t n = 0; n < kNumLods; ++n) lm.lods[n].geometricError = al.geometricError[n];
    return lm;
}

}  // namespace hf::render::vg
