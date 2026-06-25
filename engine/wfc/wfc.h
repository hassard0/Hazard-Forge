// Deterministic WAVE-FUNCTION-COLLAPSE / constraint-based procedural generation (Slice WFC-S1, flagship
// #29 WFC, beachhead). The first CONSTRAINT-SOLVER paradigm in the engine: a grid of superposed tiles
// (uint64 domain bitmasks) collapsed to a globally-consistent layout via integer AC-3 constraint
// propagation given an adjacency rule-set. Real WFC is infamous for non-reproducibility (mt19937 + float
// Shannon entropy + hash-set iteration -> different output per compiler/platform); this slice establishes
// the data model + the integer AC-3 propagation core (NO randomness, NO collapse yet — S2/S3) as PURE-CPU
// INTEGER set-logic over uint64 domain bitmasks, bit-identical across MSVC/clang.
//
// SELF-CONTAINED ON PURPOSE: this header includes ONLY <cstdint>/<cstddef>/<vector> plus net/session.h
// (itself self-contained, for hf::net::DigestBytes) so it compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/wfc_test.cpp` on the Mac — the cheapest cross-platform
// proof in the engine (NO render-bake, NO image). NO fpx / RHI / GPU / shader / <cmath> / float / clock /
// RNG / <random> / <unordered_*> / <functional> / std::hash — none of these appear: the determinism is in
// the FIXED iteration order (ascending cell id, fixed dir order, no hash-ordered containers anywhere).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "net/session.h"  // hf::net::DigestBytes (FNV-1a-64) — self-contained, the pinned-golden currency

namespace hf::wfc {

// --- Core types ------------------------------------------------------------------------------------
// A cell's superposition: bit t set => tile t is still allowed at that cell (tileCount <= 64).
using Domain = uint64_t;

// The 4 cardinal directions, FIXED order. Offsets (used by Propagate): kRight=(+1,0), kUp=(0,+1),
// kLeft=(-1,0), kDown=(0,-1). The opposite of a direction is the one pointing back.
enum Dir { kRight = 0, kUp = 1, kLeft = 2, kDown = 3 };

// Opposite(kRight)=kLeft, Opposite(kLeft)=kRight, Opposite(kUp)=kDown, Opposite(kDown)=kUp.
inline int Opposite(int dir) {
    switch (dir) {
        case kRight: return kLeft;
        case kLeft:  return kRight;
        case kUp:    return kDown;
        default:     return kUp;   // kDown
    }
}

// The adjacency rule-set. allowed[t*4 + dir] = the mask of tiles permitted on the `dir` side of tile t.
// weight[t] is an integer authoring weight (stored now; consumed by S2's entropy/collapse). tileCount<=64.
struct TileSet {
    uint32_t             tileCount = 0;
    std::vector<int32_t> weight;    // [tileCount]
    std::vector<Domain>  allowed;   // [tileCount*4]
};

// allowed[t*4+dir] — the tiles permitted on the `dir` side of tile t.
inline Domain AllowedMask(const TileSet& ts, uint32_t t, int dir) {
    return ts.allowed[static_cast<std::size_t>(t) * 4u + static_cast<std::size_t>(dir)];
}

// The superposed grid. cell[z*w + x] is that cell's current domain.
struct Grid {
    int32_t             w = 0, h = 0;
    std::vector<Domain> cell;   // [w*h]
    int32_t cellId(int32_t x, int32_t z) const { return z * w + x; }
};

// --- NeighborConstraint: the union over every set tile t of AllowedMask(ts, t, dir) ----------------
// The mask the neighbor on side `dir` of a cell with domain `srcDomain` must be AND-ed with. Iterate
// set bits in ASCENDING tile order (deterministic; the OR is order-independent anyway, but the discipline
// is the point). Pure bit ops — no popcount, no float, no hash.
inline Domain NeighborConstraint(const TileSet& ts, Domain srcDomain, int dir) {
    Domain out = 0;
    for (uint32_t t = 0; t < ts.tileCount; ++t)
        if ((srcDomain >> t) & Domain{1})
            out |= AllowedMask(ts, t, dir);
    return out;
}

// --- Propagate: the integer AC-3 worklist loop -----------------------------------------------------
// Process pending cells in PINNED order — the LOWEST pending cell id each step. Dedup via an in-grid
// `queued` flag vector; the pop order is a pure function of WHICH cells are pending (scan `queued`
// ascending), never of insertion / pointer / hash order. For the popped cell, for each of its 4 in-bounds
// neighbors in fixed order {kRight, kUp, kLeft, kDown} with offsets kRight=(+1,0)/kUp=(0,+1)/kLeft=(-1,0)/
// kDown=(0,-1): m = NeighborConstraint(ts, poppedDomain, dir); newDom = neighbor & m; if newDom != neighbor
// write it + enqueue the neighbor. Returns false IMMEDIATELY if any domain becomes 0 (a CONTRADICTION,
// consumed by S3). `worklist` seeds the initial pending set; it is consumed (left empty on return).
inline bool Propagate(const TileSet& ts, Grid& g, std::vector<int32_t>& worklist) {
    const int32_t n = g.w * g.h;
    std::vector<uint8_t> queued(static_cast<std::size_t>(n < 0 ? 0 : n), 0u);
    int32_t pending = 0;

    // Seed the pending set from the worklist (dedup via the flag).
    for (const int32_t c : worklist) {
        if (c >= 0 && c < n && !queued[static_cast<std::size_t>(c)]) {
            queued[static_cast<std::size_t>(c)] = 1u;
            ++pending;
        }
    }
    worklist.clear();

    // The 4 dirs in fixed order with their (dx,dz) offsets.
    static const int kDirs[4]  = { kRight, kUp, kLeft, kDown };
    static const int kDx[4]    = { +1, 0, -1, 0 };
    static const int kDz[4]    = {  0, +1, 0, -1 };

    while (pending > 0) {
        // Pop the LOWEST pending cell id (scan `queued` ascending — pure function of WHICH cells pend).
        int32_t c = -1;
        for (int32_t i = 0; i < n; ++i) {
            if (queued[static_cast<std::size_t>(i)]) { c = i; break; }
        }
        if (c < 0) break;  // defensive; pending>0 guarantees one exists
        queued[static_cast<std::size_t>(c)] = 0u;
        --pending;

        const int32_t cx = c % g.w;
        const int32_t cz = c / g.w;
        const Domain  cDom = g.cell[static_cast<std::size_t>(c)];

        for (int d = 0; d < 4; ++d) {
            const int   dir = kDirs[d];
            const int32_t nx = cx + kDx[d];
            const int32_t nz = cz + kDz[d];
            if (nx < 0 || nx >= g.w || nz < 0 || nz >= g.h) continue;  // out of bounds
            const int32_t nId = nz * g.w + nx;

            const Domain m      = NeighborConstraint(ts, cDom, dir);
            const Domain oldDom = g.cell[static_cast<std::size_t>(nId)];
            const Domain newDom = oldDom & m;
            if (newDom != oldDom) {
                g.cell[static_cast<std::size_t>(nId)] = newDom;
                if (newDom == 0) return false;  // CONTRADICTION — empty domain
                if (!queued[static_cast<std::size_t>(nId)]) {
                    queued[static_cast<std::size_t>(nId)] = 1u;
                    ++pending;
                }
            }
        }
    }
    return true;
}

// --- DigestGrid: the pinned-golden currency, reusing net/session.h verbatim ------------------------
inline uint64_t DigestGrid(const Grid& g) {
    return hf::net::DigestBytes(g.cell.data(), g.cell.size() * sizeof(Domain));
}

// --- IsSymmetric: the AC-3 soundness debug helper --------------------------------------------------
// For every tile t,u and every dir: tile u is allowed on the `dir` side of t IFF tile t is allowed on the
// Opposite(dir) side of u. (The showcase rule-set MUST satisfy this for AC-3 to be sound.)
inline bool IsSymmetric(const TileSet& ts) {
    for (uint32_t t = 0; t < ts.tileCount; ++t)
        for (uint32_t u = 0; u < ts.tileCount; ++u)
            for (int dir = 0; dir < 4; ++dir) {
                const Domain a = (AllowedMask(ts, t, dir) >> u) & Domain{1};
                const Domain b = (AllowedMask(ts, u, Opposite(dir)) >> t) & Domain{1};
                if (a != b) return false;
            }
    return true;
}

// --- Showcase fixture (FIXED, golden-stable — integer literals, no float) --------------------------
// A 4-tile terrain band: 0=water, 1=sand, 2=grass, 3=rock. Adjacency (a "no water-next-to-grass"
// gradient), applied ISOTROPICALLY (the same rule in all 4 dirs -> trivially symmetric):
//   water <-> {water, sand}
//   sand  <-> {water, sand, grass}
//   grass <-> {sand, grass, rock}
//   rock  <-> {grass, rock}
// All weights 1 (weights exercised in S2). FIXED FOREVER — the golden pins its propagation.
inline TileSet MakeShowcaseTileSet() {
    TileSet ts;
    ts.tileCount = 4;
    ts.weight.assign(4, 1);

    // Per-tile neighbor mask (the SAME mask in all 4 directions -> isotropic, symmetric by construction).
    // bit 0=water, bit 1=sand, bit 2=grass, bit 3=rock.
    const Domain kWater = (Domain{1} << 0) | (Domain{1} << 1);                    // water, sand
    const Domain kSand  = (Domain{1} << 0) | (Domain{1} << 1) | (Domain{1} << 2); // water, sand, grass
    const Domain kGrass = (Domain{1} << 1) | (Domain{1} << 2) | (Domain{1} << 3); // sand, grass, rock
    const Domain kRock  = (Domain{1} << 2) | (Domain{1} << 3);                    // grass, rock
    const Domain perTile[4] = { kWater, kSand, kGrass, kRock };

    ts.allowed.assign(static_cast<std::size_t>(ts.tileCount) * 4u, Domain{0});
    for (uint32_t t = 0; t < ts.tileCount; ++t)
        for (int dir = 0; dir < 4; ++dir)
            ts.allowed[static_cast<std::size_t>(t) * 4u + static_cast<std::size_t>(dir)] = perTile[t];
    return ts;
}

// A w*h grid with every cell's domain = all-tiles-set.
inline Grid MakeShowcaseGrid(int32_t w, int32_t h) {
    Grid g;
    g.w = w;
    g.h = h;
    const uint32_t tileCount = 4u;  // matches MakeShowcaseTileSet
    const Domain all = (tileCount < 64u) ? ((Domain{1} << tileCount) - Domain{1}) : ~Domain{0};
    g.cell.assign(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), all);
    return g;
}

}  // namespace hf::wfc
