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

#include <bit>      // std::popcount (S2: integer set-bit entropy surrogate — std, deterministic, cross-vendor)
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

// ===================================================================================================
// --- Slice WFC-S2: min-entropy cell selection + seeded weighted collapse (the OBSERVE half) ---------
// ===================================================================================================
// S1 built the domain grid + adjacency rule-set + integer AC-3 Propagate. S2 adds the OBSERVE step of
// wave-function-collapse: pick the next cell by INTEGER min-entropy (a popcount surrogate, NOT float
// Shannon), COLLAPSE it to one tile by a SEEDED WEIGHTED integer draw, then propagate. The three classic
// sources of WFC non-determinism (float entropy + RNG + hash-ordered iteration) are all neutralized:
// integer popcount entropy, the engine's proven pure-uint32 hash, and ascending-cell-id scans. APPEND-ONLY
// below S1 — S1's types/functions are untouched. Still PURE-CPU INTEGER, self-contained, NO float/RNG.

// --- WfcHash: the integer PRNG ----------------------------------------------------------------------
// Copied VERBATIM from engine/pcg/pcg.h:42-48 (which itself copies engine/sim/particles.h::ParticleHash
// "verbatim ops") — the SAME pure-uint32 ops + constants, so the hash STREAM is byte-identical. Kept inline
// here on purpose to preserve wfc.h's self-containment: pcg.h transitively pulls in fpx.h/particles.h (the
// Q16.16 / fx chain), which would break the cheap standalone-clang cross-platform proof. Pure uint32 wrapping
// arithmetic + shifts — NO float, NO clock, NO RNG — deterministic + cross-vendor identical.
inline uint32_t WfcHash(uint32_t seed, uint32_t index) {
    uint32_t h = seed * 2654435761u;               // Knuth multiplicative
    h ^= (index + 0x9E3779B9u + (h << 6) + (h >> 2));
    h += index * 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}
inline constexpr uint32_t kCollapseSalt = 0x57464301u;  // 'WFC\1' — the collapse stream's salt (distinct layer)

// --- PopCount: the integer entropy surrogate --------------------------------------------------------
// # of tiles still allowed in a domain. std::popcount is std (C++20 <bit>), deterministic + cross-vendor;
// it replaces float Shannon entropy. Bit-exact on MSVC + clang.
inline int PopCount(Domain d) { return std::popcount(d); }

// --- SumWeight: the entropy tie-breaker -------------------------------------------------------------
// Sum of weight[t] over the SET bits of d, iterated in ASCENDING tile order (the OR/sum is order-independent
// anyway, but the discipline is the point). Pure integer.
inline int SumWeight(const TileSet& ts, Domain d) {
    int s = 0;
    for (uint32_t t = 0; t < ts.tileCount; ++t)
        if ((d >> t) & Domain{1})
            s += ts.weight[t];
    return s;
}

// --- SelectCell: the integer min-entropy observer ---------------------------------------------------
// Among all UNDECIDED cells (PopCount(domain) > 1), return the cell id of MINIMUM integer entropy, with a
// FIXED tie-order: (1) min PopCount, (2) tie -> min SumWeight, (3) tie -> lowest cell id. Decided cells
// (PopCount == 1) and contradictions (PopCount == 0) are skipped. Returns -1 if no cell is undecided
// (fully collapsed). Scans cell ids ASCENDING — NO float, NO hash-set iteration.
inline int32_t SelectCell(const TileSet& ts, const Grid& g) {
    const int32_t n = g.w * g.h;
    int32_t bestId = -1;
    int     bestPop = 0;
    int     bestWeight = 0;
    for (int32_t c = 0; c < n; ++c) {
        const Domain d = g.cell[static_cast<std::size_t>(c)];
        const int pop = PopCount(d);
        if (pop <= 1) continue;  // skip decided (1) and contradictions (0)
        const int w = SumWeight(ts, d);
        // Ascending scan => lowest cell id wins the final tie automatically (only replace on STRICT improve).
        if (bestId < 0 || pop < bestPop || (pop == bestPop && w < bestWeight)) {
            bestId = c;
            bestPop = pop;
            bestWeight = w;
        }
    }
    return bestId;
}

// --- Collapse: a seeded weighted integer draw -------------------------------------------------------
// Decide one cell: among the tiles set in g.cell[cellId], pick one with probability proportional to
// weight[t], deterministically from the seed. total = SumWeight; r = WfcHash(seed ^ kCollapseSalt, cellId)
// % total; walk the SET tiles ASCENDING accumulating weight[t], choose the FIRST tile whose running sum > r.
// Set g.cell[cellId] = Domain{1} << chosen (a single-bit domain = decided). All weights positive (tileset
// guarantee) so total > 0. Returns the chosen tile index.
inline uint32_t Collapse(const TileSet& ts, Grid& g, int32_t cellId, uint32_t seed) {
    const Domain d = g.cell[static_cast<std::size_t>(cellId)];
    const uint32_t total = static_cast<uint32_t>(SumWeight(ts, d));
    const uint32_t r = WfcHash(seed ^ kCollapseSalt, static_cast<uint32_t>(cellId)) % total;
    uint32_t acc = 0;
    uint32_t chosen = 0;
    for (uint32_t t = 0; t < ts.tileCount; ++t) {
        if ((d >> t) & Domain{1}) {
            acc += static_cast<uint32_t>(ts.weight[t]);
            if (acc > r) { chosen = t; break; }
        }
    }
    g.cell[static_cast<std::size_t>(cellId)] = Domain{1} << chosen;
    return chosen;
}

// --- ObserveStep: one full observe iteration --------------------------------------------------------
// SelectCell -> if -1 the grid is fully collapsed (kDone). Otherwise Collapse that cell, seed the worklist
// with it, and Propagate. If Propagate returns false (an empty domain appeared) -> kContradiction; else
// kProgressed. S2 does NOT backtrack on contradiction — that is S3.
enum class StepResult { kProgressed, kDone, kContradiction };

inline StepResult ObserveStep(const TileSet& ts, Grid& g, uint32_t seed) {
    const int32_t c = SelectCell(ts, g);
    if (c < 0) return StepResult::kDone;  // fully collapsed — observer terminates
    Collapse(ts, g, c, seed);
    std::vector<int32_t> worklist{ c };
    if (!Propagate(ts, g, worklist)) return StepResult::kContradiction;
    return StepResult::kProgressed;
}

// ===================================================================================================
// --- Slice WFC-S3: the full deterministic BACKTRACKING solver (the make-or-break) ------------------
// ===================================================================================================
// S1 built Propagate; S2 built SelectCell/Collapse/ObserveStep (observe WITHOUT backtrack). S3 is the
// heart of the flagship: a COMPLETE WFC solver — observe → propagate, and on a CONTRADICTION (an empty
// domain) BACKTRACK to the last decision, remove the just-tried tile, and retry — until the grid is fully
// collapsed or proven unsolvable. Real WFC goes non-deterministic exactly here (ordered-set iteration,
// RNG re-seed, float entropy on retry). S3 makes the WHOLE backtracking search bit-identical CPU/Vulkan/
// Metal by pinning every ordering and snapshotting domains by byte-exact value-copy — the fpx
// SnapshotWorld/RestoreWorld discipline + the net::RollbackSession didRollback flag, applied to the WFC
// domain array. APPEND-ONLY below S2 — S1/S2 types/functions are untouched. Still PURE-CPU INTEGER,
// self-contained, NO float/RNG/hash-set in the solve path.

// --- Decision: one entry on the backtracking stack --------------------------------------------------
// snapshot is a byte-exact value-copy of the WHOLE g.cell vector AS OF just BEFORE this cell's collapse
// (the RestoreWorld discipline — restores bit-exact). triedMask records tiles ALREADY tried at this cell.
struct Decision {
    int32_t              cellId;      // the cell this decision collapsed
    Domain               triedMask;   // tiles already tried at this cell (bit set => tried)
    std::vector<Domain>  snapshot;    // value-copy of g.cell just BEFORE this collapse
};

// --- SolveResult: the outcome + provenance ----------------------------------------------------------
struct SolveResult {
    bool     solved       = false;  // true iff fully collapsed with no contradiction
    bool     didBacktrack = false;  // true iff >=1 backtrack fired (the didRollback twin)
    uint32_t steps        = 0;      // observe+backtrack iterations consumed (the maxSteps guard counter)
    uint32_t backtracks   = 0;      // how many times we popped/retried
};

// --- PickTileFromDomain: the deterministic weighted draw over a CANDIDATE set -----------------------
// total = SumWeight(ts, cand); if total <= 0 return -1 (empty candidate set). r = WfcHash(seed ^
// kCollapseSalt, idx) % total; walk cand's SET bits ASCENDING accumulating weight[t], return the FIRST
// tile whose running sum > r. The S2 Collapse draw, but restricted to an arbitrary candidate mask and
// indexed by a caller-supplied idx (so successive retries at the same cell draw DIFFERENT tiles). NEW
// append-only helper — S2's Collapse is NOT modified.
inline int PickTileFromDomain(const TileSet& ts, Domain cand, uint32_t seed, uint32_t idx) {
    const int total = SumWeight(ts, cand);
    if (total <= 0) return -1;
    const uint32_t r = WfcHash(seed ^ kCollapseSalt, idx) % static_cast<uint32_t>(total);
    uint32_t acc = 0;
    for (uint32_t t = 0; t < ts.tileCount; ++t) {
        if ((cand >> t) & Domain{1}) {
            acc += static_cast<uint32_t>(ts.weight[t]);
            if (acc > r) return static_cast<int>(t);
        }
    }
    return -1;  // unreachable when total > 0 (acc reaches total > r)
}

// --- Solve: the decision-stack backtracking solver --------------------------------------------------
// Loop up to maxSteps. Each iteration:
//   - c = SelectCell(ts, g). If c == -1 → fully collapsed → solved=true, return.
//   - DECIDE: push Decision{cellId=c, triedMask=0, snapshot=g.cell (whole-grid value-copy BEFORE collapse)}.
//     Pick a tile from the candidate set (snapshot[c] & ~triedMask) via PickTileFromDomain indexed by
//     (c*64 + PopCount(triedMask)) so retries draw different tiles. Set g.cell[c]=1<<tile, mark triedMask,
//     seed worklist with c, Propagate. If Propagate fails → contradiction path (next iteration).
//   - On a contradiction (the top decision's collapse failed): ++backtracks; didBacktrack=true. Restore
//     g.cell = top.snapshot (bit-exact). cand = top.snapshot[top.cellId] & ~top.triedMask (untried tiles).
//     If cand != 0: pick the next tile (same rule, indexed by triedCount), set+mark+propagate STAYING at
//     this decision (do NOT push). If cand == 0 (cell exhausted): pop this Decision and backtrack to the
//     PARENT (restore + try its next tile — same logic, re-evaluated from the new top next iteration). If
//     the stack empties with no candidate → solved=false (unsolvable), return.
// Each loop body re-evaluates the current top rather than recursing — clean "retry current" vs "pop to
// parent" handling. Hitting maxSteps without full collapse → solved=false (deterministic give-up).
// FIXED ORDER everywhere: SelectCell (S2), weighted ascending tile trial indexed by (cellId,triedCount),
// LIFO std::vector<Decision>, whole-grid value-copy snapshot/restore. NO unordered/hash/float.
inline SolveResult Solve(const TileSet& ts, Grid& g, uint32_t seed, uint32_t maxSteps) {
    SolveResult res;
    std::vector<Decision> stack;

    // pendingContradiction == true means the current top decision's last collapse contradicted and we must
    // back out / retry BEFORE selecting a new cell. false means we are in the "forward" decide phase.
    bool pendingContradiction = false;

    for (uint32_t step = 0; step < maxSteps; ++step) {
        res.steps = step + 1u;

        if (!pendingContradiction) {
            // --- FORWARD: select a cell and decide it. ---
            const int32_t c = SelectCell(ts, g);
            if (c < 0) {
                // SelectCell skips both decided (pop==1) AND empty (pop==0) cells, so a -1 means "no cell
                // with pop>1". Distinguish a genuine full collapse from a grid that already holds an empty
                // domain (e.g. a pre-pinned contradiction the search must back out of).
                bool hasEmpty = false;
                for (const Domain d : g.cell) if (d == 0) { hasEmpty = true; break; }
                if (!hasEmpty) { res.solved = true; return res; }  // fully collapsed, no contradiction
                pendingContradiction = true;  // an empty domain — back out (or prove unsolvable)
                continue;
            }

            // Push a fresh decision: snapshot the WHOLE grid BEFORE collapsing.
            Decision d;
            d.cellId    = c;
            d.triedMask = 0;
            d.snapshot  = g.cell;  // byte-exact value-copy (RestoreWorld discipline)
            stack.push_back(std::move(d));
            Decision& top = stack.back();

            const Domain cand = top.snapshot[static_cast<std::size_t>(c)] & ~top.triedMask;
            const uint32_t idx = static_cast<uint32_t>(c) * 64u +
                                 static_cast<uint32_t>(PopCount(top.triedMask));
            const int tile = PickTileFromDomain(ts, cand, seed, idx);
            if (tile < 0) { pendingContradiction = true; continue; }  // no candidate — back out

            g.cell[static_cast<std::size_t>(c)] = Domain{1} << static_cast<uint32_t>(tile);
            top.triedMask |= Domain{1} << static_cast<uint32_t>(tile);

            std::vector<int32_t> worklist{ c };
            if (!Propagate(ts, g, worklist)) { pendingContradiction = true; continue; }
            // else: progressed — next iteration selects the next cell.
        } else {
            // --- BACKTRACK: the current top's collapse contradicted. ---
            ++res.backtracks;
            res.didBacktrack = true;

            if (stack.empty()) { res.solved = false; return res; }  // nothing to back out to
            Decision& top = stack.back();

            // Restore the whole grid to this decision's pre-collapse state (bit-exact).
            g.cell = top.snapshot;

            const Domain cand = top.snapshot[static_cast<std::size_t>(top.cellId)] & ~top.triedMask;
            if (cand == 0) {
                // This cell is exhausted — pop it and back out to the parent (retry parent next iter).
                stack.pop_back();
                if (stack.empty()) { res.solved = false; return res; }  // unsolvable
                // pendingContradiction stays true: re-evaluate the new top next iteration.
                continue;
            }

            const uint32_t idx = static_cast<uint32_t>(top.cellId) * 64u +
                                 static_cast<uint32_t>(PopCount(top.triedMask));
            const int tile = PickTileFromDomain(ts, cand, seed, idx);
            if (tile < 0) {  // defensive — cand != 0 implies total > 0, but guard anyway
                stack.pop_back();
                if (stack.empty()) { res.solved = false; return res; }
                continue;
            }

            g.cell[static_cast<std::size_t>(top.cellId)] = Domain{1} << static_cast<uint32_t>(tile);
            top.triedMask |= Domain{1} << static_cast<uint32_t>(tile);

            std::vector<int32_t> worklist{ top.cellId };
            if (Propagate(ts, g, worklist)) {
                pendingContradiction = false;  // retry succeeded — back to forward decide
            }
            // else: STILL contradicts — stay pendingContradiction, retry/back out next iteration.
        }
    }

    // Hit maxSteps without fully collapsing — deterministic give-up.
    res.solved = false;
    return res;
}

}  // namespace hf::wfc
