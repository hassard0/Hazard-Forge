#pragma once
// Slice AI1 — Deterministic AI: THE BLACKBOARD + DECISION-TREE NODE GRAPH + DETERMINISTIC TICK (the
// beachhead of FLAGSHIP for GitHub issue #28: DETERMINISTIC AI — decision trees + environment queries,
// slice-tag AI, hf::ai). The engine ships a deterministic navmesh + A* (engine/nav/navmesh.h) but no NPC
// DECISION layer. This flagship extends the determinism moat into game AI — a bit-identical,
// cross-platform, lockstep-replayable decision layer that float engines run non-deterministically (so they
// cannot lockstep/replay AI).
//
// AI1 is the beachhead: a FLAT, data-driven decision-tree node graph + an INTEGER blackboard + a
// DETERMINISTIC tick. It pins the two load-bearing determinism decisions the whole flagship rides on:
//
//   * A FLAT INDEX GRAPH with a FIXED depth-first child order — the tree is a std::vector<BtNode>; children
//     are integer INDICES into the vector (NOT pointers), evaluated in a FIXED order. This is the
//     nav::Poly flat-index discipline (a pointer graph or an unbounded/allocator-dependent walk would be
//     non-deterministic across allocators/compilers; the flat index walk is bit-stable by construction).
//   * An INTEGER BLACKBOARD — a fixed std::array of int32_t/fx slots (a small key space), zero-initialized.
//     Conditions/actions read/write by integer key. NO float, NO std::map (its iteration/allocation could
//     vary). DigestBlackboard FNV-1a's the slot array in FIXED order -> the deterministic proof currency
//     every later slice reuses.
//
// THE TICK is depth-first in FIXED child order: a selector returns the first non-Failure child
// (short-circuits); a sequence returns the first non-Success child; an inverter flips Success<->Failure; a
// condition leaf tests bb.slot[bbKey] against param; an action leaf writes param into bb.slot[bbKey] and
// returns Success. Two TickTree runs over the same tree+blackboard are byte-identical BY CONSTRUCTION
// (fixed order, in-place integer mutation).
//
// TREE REPRESENTATION CHOICE: a BOUNDED-DEPTH RECURSION (TickNode recurses on child indices, guarded by
// kMaxDepth). This is simpler than an explicit integer stack and equally bit-stable — the recursion order
// is the FIXED child order, the depth is bounded by kMaxDepth (a canonical AI1 tree is depth 2-3), and a
// depth overrun is a deterministic kFailure (never UB). The graph itself is flat indices, so there is no
// allocator/pointer non-determinism; the recursion is purely the evaluation walk.
//
// Header-only, namespace hf::ai, PURE CPU. #includes game/verdict.h + sim/fpx.h READ-ONLY/BYTE-FROZEN; ai.h
// is a brand-new additive sibling that NEVER edits a frozen header or a shader. NO render RHI, NO shader,
// NO compute. Out of scope (later AI slices): environment queries (AI2), line-of-sight/perception (AI3),
// the verdict-agent composition (AI4), lockstep (AI5), the lit render (AI6 — AI1's render is the 2D
// node-status viz). Only the five core node kinds; no parallel/decorator-with-memory, no JSON tree loader.

#include <array>
#include <cstdint>
#include <vector>

#include "game/verdict.h"   // read-only: the FNV-1a digest discipline (DigestFnv constants
                            // 1469598103934665603 / 1099511628211) we mirror in a LOCAL helper — establish
                            // the dependency now (AI agents become verdict entities in AI4)
                            // [Slice AI2 also uses verdict::FxDist2 — the Q16.16 squared distance, NO sqrt]
#include "nav/navmesh.h"    // read-only (Slice AI2): the deterministic navmesh + INTEGER A* — FindPath
                            // (reachability cost), ConnectedComponents (component membership), ManhattanDist,
                            // ComputePolyCentroids (candidate->poly), QuantizeCoord (world->cell). AI2's
                            // ScoreNavReachable runs the candidate's nav poly through these. navmesh.h FROZEN.
#include "sim/fpx.h"        // read-only: fx / kOne / kFrac (the Q16.16 blackboard values + params)

namespace hf {
namespace ai {

// The frozen fixed-point toolbox (under hf::sim::fpx) — alias it locally so the blackboard values read like
// the sim headers.
namespace fpx = hf::sim::fpx;
using fpx::fx;     // a Q16.16 fixed-point scalar (== int32_t)
using fpx::kOne;   // 1.0 in Q16.16 (65536)
using fpx::kFrac;  // fractional bits (16)

// ----- Fixed capacities (the flat-graph / fixed-key-space determinism bounds) --------------------------
inline constexpr int kMaxBbKeys   = 16;  // the integer blackboard key space (slot[0..kMaxBbKeys-1])
inline constexpr int kMaxChildren = 8;   // the max children of a composite node (child[0..kMaxChildren-1])
inline constexpr int kMaxDepth    = 32;  // the bounded recursion depth (a depth overrun -> kFailure, no UB)

// ----- Blackboard: a fixed-capacity integer/Q16.16 key->value store (deterministic, byte-stable) -------
// A fixed std::array of int32_t slots, ZERO-initialized. Read/write by integer key. NO float, NO std::map.
// Condition leaves read slot[bbKey]; action leaves write slot[bbKey]. Values are plain int32_t — a Q16.16
// fx is just an int32_t, so the same store carries integer counters AND Q16.16 scalars/params. A key out
// of [0, kMaxBbKeys) is a deterministic no-op on write / reads as 0 (never UB).
struct Blackboard {
    std::array<int32_t, kMaxBbKeys> slot{};  // zero-initialized fixed slot array

    int32_t Get(int key) const {
        if (key < 0 || key >= kMaxBbKeys) return 0;   // out-of-range reads as 0 (deterministic)
        return slot[(size_t)key];
    }
    void Set(int key, int32_t value) {
        if (key < 0 || key >= kMaxBbKeys) return;     // out-of-range write is a no-op (deterministic)
        slot[(size_t)key] = value;
    }
};

// ----- NodeKind: the five core node kinds (no parallel/decorator-with-memory — YAGNI) ------------------
enum NodeKind : uint32_t {
    kSelector   = 0u,  // returns the FIRST non-Failure child (short-circuits on the first Success/Running)
    kSequence   = 1u,  // returns the FIRST non-Success child (short-circuits on the first Failure/Running)
    kInverter   = 2u,  // flips its single child's Success<->Failure (Running stays Running)
    kCondLeaf   = 3u,  // tests bb.slot[bbKey] >= param -> Success, else Failure (a read; no write)
    kActionLeaf = 4u,  // writes param into bb.slot[bbKey] -> Success (the side effect; always Success)
};

// ----- BtNode: a FLAT node (children are INDICES into the DecisionTree vector, NOT pointers) ------------
// kind = a NodeKind. child[0..childCount-1] are indices into the tree vector (the flat-graph discipline).
// param is the integer/Q16.16 payload (the condition threshold / the action value). bbKey is the
// blackboard slot a leaf reads/writes (ignored by composites). A leaf has childCount 0; an inverter has
// childCount 1; selector/sequence have childCount in [0, kMaxChildren].
struct BtNode {
    uint32_t kind     = kSelector;            // a NodeKind
    int      child[kMaxChildren] = {};        // child INDICES into the DecisionTree (NOT pointers)
    int      childCount = 0;                  // number of valid entries in child[]
    int32_t  param    = 0;                    // condition threshold / action value (integer or Q16.16)
    int      bbKey    = 0;                     // the blackboard slot a leaf reads/writes
};

// The decision tree is a flat vector; the ROOT is at index 0. Children referenced by index — NOT pointers.
using DecisionTree = std::vector<BtNode>;

// ----- Status: the tick result of a node (the standard behaviour-tree tri-state) -----------------------
enum Status : uint32_t {
    kRunning = 0u,
    kSuccess = 1u,
    kFailure = 2u,
};

// ----- TickNode: evaluate node `idx` depth-first in FIXED child order (bounded recursion) --------------
// The bounded-recursion walk (guarded by kMaxDepth). A bad index or a depth overrun is a deterministic
// kFailure (never UB). Pure integer, fixed order -> bit-identical by construction.
inline Status TickNode(const DecisionTree& tree, Blackboard& bb, int idx, int depth) {
    if (idx < 0 || (size_t)idx >= tree.size()) return kFailure;  // bad index -> deterministic Failure
    if (depth >= kMaxDepth) return kFailure;                     // depth overrun -> deterministic Failure
    const BtNode& n = tree[(size_t)idx];

    switch (n.kind) {
        case kSelector: {
            // The FIRST non-Failure child decides (short-circuits). All children Failure -> Failure.
            for (int c = 0; c < n.childCount && c < kMaxChildren; ++c) {
                const Status s = TickNode(tree, bb, n.child[c], depth + 1);
                if (s != kFailure) return s;   // first Success/Running wins; remaining children NOT visited
            }
            return kFailure;
        }
        case kSequence: {
            // The FIRST non-Success child decides (short-circuits). All children Success -> Success.
            for (int c = 0; c < n.childCount && c < kMaxChildren; ++c) {
                const Status s = TickNode(tree, bb, n.child[c], depth + 1);
                if (s != kSuccess) return s;   // first Failure/Running wins; remaining children NOT visited
            }
            return kSuccess;
        }
        case kInverter: {
            // Flip the single child's Success<->Failure (Running stays Running). No child -> Failure.
            if (n.childCount < 1) return kFailure;
            const Status s = TickNode(tree, bb, n.child[0], depth + 1);
            if (s == kSuccess) return kFailure;
            if (s == kFailure) return kSuccess;
            return kRunning;
        }
        case kCondLeaf:
            // A READ: bb.slot[bbKey] >= param -> Success, else Failure. No blackboard mutation.
            return (bb.Get(n.bbKey) >= n.param) ? kSuccess : kFailure;
        case kActionLeaf:
            // A WRITE: store param into bb.slot[bbKey], always Success (the side effect).
            bb.Set(n.bbKey, n.param);
            return kSuccess;
        default:
            return kFailure;   // unknown kind -> deterministic Failure
    }
}

// ----- TickTree(tree, bb): tick the whole tree from the root (index 0) ---------------------------------
// Depth-first in FIXED child order over the flat index graph, mutating the integer blackboard in place.
// An empty tree is a deterministic kFailure. Two TickTree runs over the same tree + blackboard are
// byte-identical (fixed order, in-place integer mutation) — the AI1 determinism contract.
inline Status TickTree(const DecisionTree& tree, Blackboard& bb) {
    if (tree.empty()) return kFailure;
    return TickNode(tree, bb, 0, 0);
}

// ----- TreeDepth: the max depth of the flat tree (root at index 0), for the {nodes, depth} proof -------
// A bounded DFS over the flat indices (the SAME fixed child order as the tick), returning the longest
// root-to-leaf path length (the root alone is depth 1). A bad index / a depth overrun stops that branch
// (deterministic). Pure read.
inline int TreeDepthAt(const DecisionTree& tree, int idx, int depth) {
    if (idx < 0 || (size_t)idx >= tree.size()) return depth;
    if (depth >= kMaxDepth) return depth;
    const BtNode& n = tree[(size_t)idx];
    int best = depth + 1;   // this node contributes one level
    for (int c = 0; c < n.childCount && c < kMaxChildren; ++c) {
        const int d = TreeDepthAt(tree, n.child[c], depth + 1);
        if (d > best) best = d;
    }
    return best;
}
inline int TreeDepth(const DecisionTree& tree) {
    if (tree.empty()) return 0;
    return TreeDepthAt(tree, 0, 0);
}

// ----- DigestBlackboard(bb): FNV-1a over slot[] in FIXED order (the deterministic proof currency) ------
// Mirrors verdict.h's DigestFnv constants (1469598103934665603 / 1099511628211 — the same engine-wide FNV
// offset/prime) in a LOCAL helper (we do NOT pull verdict's private helper into ai.h). Folds the slot
// array LSB-first (a fixed byte order -> endianness-independent + stable). Two equal blackboards hash
// IDENTICALLY; a single changed slot changes the digest.
inline uint64_t DigestBlackboard(const Blackboard& bb) {
    uint64_t h = 1469598103934665603ull;          // the FNV-1a 64-bit offset basis
    for (int k = 0; k < kMaxBbKeys; ++k) {
        const uint32_t v = (uint32_t)bb.slot[(size_t)k];
        for (int b = 0; b < 4; ++b) {              // fold the 4 bytes of the int32 slot, LSB-first
            h ^= (uint64_t)(uint8_t)(v >> (b * 8));
            h *= 1099511628211ull;                 // the FNV-1a 64-bit prime
        }
    }
    return h;
}

// ----- Canonical AI1 blackboard key layout (the showcase/test scenario keys) ---------------------------
// A small fixed key map. kBbEnemyClose carries a Q16.16 "enemy proximity" score the condition leaf tests;
// kBbState is the action output (the chosen NPC state). Plain integer keys into the fixed slot array.
inline constexpr int kBbEnemyClose = 0;   // input: a Q16.16 enemy-proximity score (the cond leaf reads it)
inline constexpr int kBbState      = 1;   // output: the chosen state (an action leaf writes it)

// State values an action leaf writes into kBbState (plain integers — the NPC state enum).
inline constexpr int32_t kStatePatrol = 1;
inline constexpr int32_t kStateChase  = 2;

// The condition threshold: enemyClose >= kEnemyThreshold (Q16.16 1.0) -> the enemy is "close".
inline constexpr int32_t kEnemyThreshold = kOne;   // 1.0 in Q16.16

// ----- BuildAi1Tree(): the fixed canonical decision tree for the showcase/test -------------------------
// selector[ sequence[ condLeaf(enemyClose >= 1.0), actionLeaf(state = CHASE) ], actionLeaf(state = PATROL) ]
//
// Flat layout (root at index 0):
//   [0] selector       child = {1, 4}
//   [1]   sequence     child = {2, 3}
//   [2]     condLeaf   bbKey = kBbEnemyClose, param = kEnemyThreshold   (is the enemy close?)
//   [3]     actionLeaf bbKey = kBbState,      param = kStateChase       (-> CHASE)
//   [4]   actionLeaf   bbKey = kBbState,      param = kStatePatrol      (the fallback -> PATROL)
//
// SEMANTICS: if enemyClose >= 1.0 the sequence runs to completion (cond Success -> action writes CHASE,
// Success), so the selector short-circuits at child 0 and never reaches the PATROL fallback. If the enemy
// is NOT close the condition Fails, the sequence Fails at child 0 (the action is NOT reached — no CHASE
// write), so the selector falls through to child 1 (the PATROL action). The selected branch is a pure,
// deterministic function of the kBbEnemyClose input -> flipping it flips the branch.
inline DecisionTree BuildAi1Tree() {
    DecisionTree t;
    t.resize(5);

    // [0] selector over { [1] sequence, [4] patrol action }
    t[0].kind = kSelector;
    t[0].child[0] = 1; t[0].child[1] = 4; t[0].childCount = 2;

    // [1] sequence over { [2] cond, [3] chase action }
    t[1].kind = kSequence;
    t[1].child[0] = 2; t[1].child[1] = 3; t[1].childCount = 2;

    // [2] condLeaf: enemyClose >= threshold ?
    t[2].kind = kCondLeaf;
    t[2].bbKey = kBbEnemyClose; t[2].param = kEnemyThreshold; t[2].childCount = 0;

    // [3] actionLeaf: state = CHASE
    t[3].kind = kActionLeaf;
    t[3].bbKey = kBbState; t[3].param = kStateChase; t[3].childCount = 0;

    // [4] actionLeaf: state = PATROL (the selector fallback)
    t[4].kind = kActionLeaf;
    t[4].bbKey = kBbState; t[4].param = kStatePatrol; t[4].childCount = 0;

    return t;
}

// =====================================================================================================
// Slice AI2 — ENVIRONMENT QUERIES: integer candidate generation + integer scoring over the navmesh.
// =====================================================================================================
// AI1 (above) is BYTE-FROZEN — everything below is APPEND-ONLY. AI2 adds the "where should I go?"
// layer: an NPC scores candidate positions around an anchor by PURE-INTEGER queries (Q16.16 distance²
// + navmesh-reachability) and picks the best, deterministically. NO float in the generator OR the
// scorers — the candidate order is FIXED, the scores are integer, the tie-break is lowest candidate
// index. Bit-identical CPU/Vulkan/Metal BY CONSTRUCTION (same integers, same fixed order). This is what
// a decision-tree "pick a destination" action will call (wired in AI4).
//
// The reachability scorer reuses the FROZEN nav primitives (nav::ConnectedComponents / nav::FindPath /
// nav::ComputePolyCentroids / nav::ManhattanDist) READ-ONLY, so it inherits nav::FindPath's documented
// scope: paths within the LARGEST connected component only (navmesh.h NAV scope). The AI2 scene keeps
// the anchor + candidates + target reachable within one component; an unreachable candidate (a different
// nav component, or no finite path) gets a large penalty so it loses.

namespace nav     = hf::nav;
namespace verdict = hf::game::verdict;

// ----- The candidate world position (Q16.16 / integer voxel coords — the SAME units the navmesh uses) -
// A candidate is a 2D top-down XZ position (y == 0); the query runs over the navmesh's XZ plane. Held as
// fpx::FxVec3 so ScoreDistanceSq can call verdict::FxDist2 directly (the Q16.16 squared distance).
using verdict::FxDist2;   // read-only: int64 Q16.16 squared distance (NO sqrt) — the AI2 distance scorer

// ----- Fixed AI2 capacities + scoring constants -------------------------------------------------------
inline constexpr int kMaxCandidates = 64;   // the bound on a query's candidate count (the EQS min-scan)
inline constexpr int kQueryScorers  = 2;    // the fixed scorer chain length: distance² + reachability

// kBigPenalty: the unreachable sentinel added to a candidate's combined score so it always loses to ANY
// reachable candidate. The distance² term is FxDist2 (a Q32.32 squared distance in int64) NARROWED by
// >> kFrac to a Q16.16-scaled squared distance (i.e. real_d² * 65536) that fits int32 for in-scene
// distances (a few world units: 25*65536 ≈ 1.6M << INT32_MAX) while keeping FULL sub-unit resolution —
// the AI1 int32 digest-currency tier. kBigPenalty (2^30) is far above any in-scene narrowed distance² yet
// far below INT32_MAX, so distance²+penalty never overflows int32. A reachable candidate's score is its
// narrowed distance² (penalty 0); an unreachable one's score is that + kBigPenalty (so even the NEAREST
// unreachable candidate loses to the FARTHEST reachable one).
inline constexpr int32_t kBigPenalty = 0x40000000;   // 2^30 — >> any in-scene distance², << INT32_MAX

// ----- The fixed precomputed integer ring direction table (NO float trig) -----------------------------
// 16 unit-ish integer offset vectors evenly around a ring, scaled by 64 (so a radius in Q16.16 multiplies
// them without losing the diagonal directions). The table is HAND-BAKED (the gi/ik host-baked angle-LUT
// precedent): dir k = (round(64*cos(2*pi*k/16)), round(64*sin(2*pi*k/16))) computed at AUTHORING time and
// frozen here as literals, so the runtime has NO float trig — the generator is pure integer. FIXED order
// k=0..15 → the candidate order is deterministic. (cos/sin baked: 64, 59, 45, 24, 0, -24, -45, -59, -64,
// -59, -45, -24, 0, 24, 45, 59 for x; the sin column is the same sequence rotated by 4.)
inline constexpr int kRingDirCount = 16;
struct RingDir { int32_t dx, dz; };   // integer offsets, scaled by 64 (kRingDirScale)
inline constexpr int32_t kRingDirScale = 64;
inline constexpr RingDir kRingDirs[kRingDirCount] = {
    { 64,   0}, { 59,  24}, { 45,  45}, { 24,  59},
    {  0,  64}, {-24,  59}, {-45,  45}, {-59,  24},
    {-64,   0}, {-59, -24}, {-45, -45}, {-24, -59},
    {  0, -64}, { 24, -59}, { 45, -45}, { 59, -24},
};

// ----- GenerateRing(anchor, radius, count): `count` candidates on a ring around `anchor` ---------------
// Places `count` candidates (clamped to [0, min(kRingDirCount, kMaxCandidates)]) on a ring of the given
// Q16.16 `radius` around `anchor`, in FIXED table order (k=0..count-1). Each candidate's offset =
// (kRingDirs[k] * radius) / kRingDirScale, applied to the anchor's XZ (y stays the anchor's y). PURE
// INTEGER: the offset is (dir * radius) >> 0 then / 64 — int64 intermediate (dir is small, radius is
// Q16.16) narrowed back to fx, NO float. Deterministic by construction (fixed table, fixed order).
inline std::vector<fpx::FxVec3> GenerateRing(const fpx::FxVec3& anchor, fpx::fx radius, int count) {
    std::vector<fpx::FxVec3> out;
    int n = count;
    if (n < 0) n = 0;
    if (n > kRingDirCount) n = kRingDirCount;
    if (n > kMaxCandidates) n = kMaxCandidates;
    out.reserve((size_t)n);
    for (int k = 0; k < n; ++k) {
        // offset axis = (dir * radius) / kRingDirScale (int64 intermediate, deterministic truncation).
        const int64_t ox = ((int64_t)kRingDirs[k].dx * (int64_t)radius) / (int64_t)kRingDirScale;
        const int64_t oz = ((int64_t)kRingDirs[k].dz * (int64_t)radius) / (int64_t)kRingDirScale;
        fpx::FxVec3 c;
        c.x = anchor.x + (fpx::fx)ox;
        c.y = anchor.y;                 // 2D top-down query: the ring lies in the XZ plane
        c.z = anchor.z + (fpx::fx)oz;
        out.push_back(c);
    }
    return out;
}

// ----- ScoreDistanceSq(cand, target): verdict::FxDist2 narrowed (>> kFrac) to a Q16.16-scaled d² --------
// = verdict::FxDist2(cand, target) (a pure-integer XYZ squared distance, Q32.32 in int64, NO sqrt) NARROWED
// by >> kFrac to a Q16.16-scaled squared distance (real_d² * 65536) that fits int32 for in-scene distances
// while keeping FULL sub-unit resolution (the Q32.32 raw would overflow int32 for any distance >= 1 world
// unit). Deterministic arithmetic shift (>> on a non-negative int64 is well-defined). Saturated to the
// int32 ceiling defensively (a hostile-scale query > ~180 world units would trip it; the AI2 scene never
// does). NO float. A LOWER score is closer.
inline int32_t ScoreDistanceSq(const fpx::FxVec3& cand, const fpx::FxVec3& target) {
    const int64_t d2raw = FxDist2(cand, target);     // Q32.32 squared distance in int64 (NO float, NO sqrt)
    const int64_t d2 = d2raw >> fpx::kFrac;          // -> Q16.16-scaled d² (real_d² * 65536), keeps resolution
    if (d2 > (int64_t)0x3FFFFFFF) return (int32_t)0x3FFFFFFF;   // defensive saturate (never trips in-scene)
    return (int32_t)d2;
}

// ----- NavScene: the canonical AI2 navmesh (the nav-path 32x32 showcase) + its centroids/components -----
// A small wrapper over the FROZEN nav primitives so the scorer + the showcase share ONE built navmesh.
// Built by BuildNavScene() (the --nav-path scene): polys + flat contour verts + per-poly vertex base →
// ComputePolyCentroids (cx/cz) → ConnectedComponents (comp). All integer; deterministic by construction.
struct NavScene {
    std::vector<nav::Poly>     polys;
    std::vector<int32_t>       flatVerts;
    std::vector<uint32_t>      polyVertBase;
    std::vector<int32_t>       cx, cz;        // per-poly integer centroids (ComputePolyCentroids)
    std::vector<uint32_t>      comp;          // per-poly connected-component id (ConnectedComponents)
    uint32_t                   nComp = 0;
    nav::Heightfield           hf{};          // the heightfield (for QuantizeCoord world->cell, if needed)
    int32_t                    navScale = 1;  // world(Q16.16)/scale → voxel: voxel = (world>>kFrac)*scale
};

// ----- PolyOfPoint(scene, wx, wz): the candidate's nav poly = nearest poly centroid (Manhattan) --------
// Maps a candidate's VOXEL-space (wx, wz) to a poly id by the nearest poly centroid in integer Manhattan
// distance (tie → LOWEST poly id, the ascending-scan deterministic tie-break — the nav::SelectStartGoal
// discipline). Returns nav::kNoCameFrom if there are no polys. PURE INTEGER. (A point-in-poly cover test
// is a future refinement; nearest-centroid is deterministic + sufficient for the within-component scene.)
inline uint32_t PolyOfPoint(const NavScene& scene, int32_t wx, int32_t wz) {
    const size_t nP = scene.polys.size();
    if (nP == 0) return nav::kNoCameFrom;
    uint32_t best = 0u;
    int32_t bestD = nav::ManhattanDist(scene.cx[0], scene.cz[0], wx, wz);
    for (uint32_t p = 1; p < (uint32_t)nP; ++p) {
        const int32_t d = nav::ManhattanDist(scene.cx[p], scene.cz[p], wx, wz);
        if (d < bestD) { bestD = d; best = p; }   // strict < → tie keeps the LOWEST poly id (ascending)
    }
    return best;
}

// ----- WorldToVoxel(scene, w): the candidate's Q16.16 world XZ → integer voxel XZ (the navmesh units) ---
// The candidate coords are Q16.16 world units; the navmesh centroids are integer voxel units. The showcase
// scale maps the 32-voxel grid to world via navScale (voxel = (world >> kFrac) * navScale). PURE INTEGER
// (an arithmetic shift + multiply — deterministic for negatives too; no FloorDiv needed as the scene
// candidates are non-negative). Returns the voxel (x, z).
inline void WorldToVoxel(const NavScene& scene, const fpx::FxVec3& w, int32_t& vx, int32_t& vz) {
    vx = (w.x >> fpx::kFrac) * scene.navScale;
    vz = (w.z >> fpx::kFrac) * scene.navScale;
}

// ----- ScoreNavReachable(cand, agentVoxel, scene): 0 if reachable, kBigPenalty if not ------------------
// The candidate is nav-reachable from the agent iff: the candidate's nav poly is in the SAME connected
// component as the agent's poly AND nav::FindPath returns a finite (non-empty corridor) cost between them.
// Reachable → 0 (no penalty); unreachable → kBigPenalty (so it loses). PURE INTEGER (the nav primitives
// are int32 single-thread serial → deterministic). agentVoxel = the agent's integer voxel XZ.
inline int32_t ScoreNavReachable(const fpx::FxVec3& cand, int32_t agentVx, int32_t agentVz,
                                 const NavScene& scene) {
    const size_t nP = scene.polys.size();
    if (nP == 0) return kBigPenalty;
    int32_t cvx, cvz; WorldToVoxel(scene, cand, cvx, cvz);
    const uint32_t candPoly  = PolyOfPoint(scene, cvx, cvz);
    const uint32_t agentPoly = PolyOfPoint(scene, agentVx, agentVz);
    if (candPoly == nav::kNoCameFrom || agentPoly == nav::kNoCameFrom) return kBigPenalty;
    // Component membership (the cheap reject) — different component → unreachable.
    if (scene.comp[candPoly] != scene.comp[agentPoly]) return kBigPenalty;
    // Finite path → the corridor is non-empty (FindPath returns an empty corridor for "no path").
    std::vector<uint32_t> corridor;
    nav::FindPath(scene.polys, scene.cx, scene.cz, agentPoly, candPoly, corridor);
    if (corridor.empty()) return kBigPenalty;   // no finite path → penalize
    return 0;                                    // reachable → no penalty
}

// ----- EqsQuery: a query = an anchor + ring params + the target + the agent's voxel cell ---------------
// The generator places `count` candidates on a ring of `radius` around `anchor`; the scorers score each
// against `target` (distance²) + the agent's nav cell (reachability). All Q16.16 / integer.
struct EqsQuery {
    fpx::FxVec3 anchor;          // the ring center (Q16.16 world)
    fpx::fx     radius = kOne;   // the ring radius (Q16.16 world)
    int         count  = kRingDirCount;  // candidate count (clamped in GenerateRing)
    fpx::FxVec3 target;          // the scoring target (Q16.16 world) — closest reachable wins
    int32_t     agentVx = 0;     // the agent's nav voxel x (the reachability source cell)
    int32_t     agentVz = 0;     // the agent's nav voxel z
};

// ----- EqsResult: the chosen best candidate + its combined integer score ------------------------------
struct EqsResult {
    int     bestIndex = -1;      // the winning candidate index (in the FIXED generated order); -1 if none
    int32_t bestScore = 0;       // its combined integer score (distance² + reachability penalty)
};

// ----- ScoreCandidate(cand, query, scene): the FIXED integer scorer chain → the combined score ---------
// total = ScoreDistanceSq(cand, target) + ScoreNavReachable(cand, agentCell, scene). Lower is better
// ("closest reachable wins"). PURE INTEGER. Two scorers (the kQueryScorers fixed chain); more scorers
// (AI3 line-of-sight) compose by adding more terms the SAME way.
inline int32_t ScoreCandidate(const fpx::FxVec3& cand, const EqsQuery& query, const NavScene& scene) {
    const int32_t dist = ScoreDistanceSq(cand, query.target);
    const int32_t reach = ScoreNavReachable(cand, query.agentVx, query.agentVz, scene);
    return dist + reach;   // distance² + (reachable ? 0 : kBigPenalty); int32, never overflows for AI2
}

// ----- RunQuery(query, scene): generate → score in FIXED order → lowest score wins (tie → lowest idx) --
// Scans the generated candidates in FIXED order (the generator's k=0..count-1), keeps the candidate with
// the LOWEST combined score; on a TIE the LOWEST candidate index wins (the strict-< update keeps the first
// — the deterministic tie-break, never a float compare / unordered container). Returns {bestIndex,
// bestScore}; bestIndex == -1 if there are no candidates. Bit-identical by construction.
inline EqsResult RunQuery(const EqsQuery& query, const NavScene& scene) {
    const std::vector<fpx::FxVec3> cands = GenerateRing(query.anchor, query.radius, query.count);
    EqsResult r;
    for (int i = 0; i < (int)cands.size(); ++i) {
        const int32_t score = ScoreCandidate(cands[(size_t)i], query, scene);
        if (r.bestIndex < 0 || score < r.bestScore) {   // strict < → ties keep the LOWEST index
            r.bestIndex = i;
            r.bestScore = score;
        }
    }
    return r;
}

// ----- CountReachable(query, scene): how many candidates are nav-reachable (the R/C proof) -------------
inline int CountReachable(const EqsQuery& query, const NavScene& scene) {
    const std::vector<fpx::FxVec3> cands = GenerateRing(query.anchor, query.radius, query.count);
    int reachable = 0;
    for (const auto& c : cands)
        if (ScoreNavReachable(c, query.agentVx, query.agentVz, scene) == 0) ++reachable;
    return reachable;
}

// ----- BuildNavScene(): the canonical AI2 navmesh = the --nav-path 32x32 showcase (FROZEN nav pipeline) -
// Runs the deterministic NAV1-NAV4 pipeline (rasterize → merge → filter → distance → regions → contour →
// simplify → polygonize) over the SAME 32x32 heightfield the --nav-path / nav-render showcases use, then
// computes the per-poly centroids + connected-component labels. PURE INTEGER, deterministic. The query +
// the showcase share this ONE built navmesh, so AI2's query runs over a REAL navmesh.
inline NavScene BuildNavScene() {
    NavScene s;
    const int kW = 32, kH = 32;
    s.hf.w = kW; s.hf.h = kH;
    s.hf.bminX = 0; s.hf.bminY = 0; s.hf.bminZ = 0;
    s.hf.bmaxX = kW; s.hf.bmaxY = 64; s.hf.bmaxZ = kH;
    s.hf.cs = 1; s.hf.ch = 1;
    s.navScale = 4;   // == the nav-render showcase kScale (8x8 world units across the 32-voxel grid)
    const int kColumnCount = s.hf.columnCount();
    nav::WalkableConfig cfg; cfg.walkableHeight = 2; cfg.walkableClimb = 1;
    const int32_t kMaxError = 0;

    std::vector<nav::NavTri> tris = nav::MakeShowcaseTriangles(s.hf);
    std::vector<uint32_t> rColCount, rColOffset;
    std::vector<nav::Span> rSpans;
    nav::RasterizeTriangleSpans(s.hf, std::span<const nav::NavTri>(tris), rColCount, rColOffset, rSpans);
    std::vector<std::vector<nav::Span>> mergedPerCol((size_t)kColumnCount);
    for (int c = 0; c < kColumnCount; ++c) {
        std::vector<nav::Span> raw(rSpans.begin() + rColOffset[(size_t)c],
                                   rSpans.begin() + rColOffset[(size_t)c] + rColCount[(size_t)c]);
        mergedPerCol[(size_t)c] = nav::MergeColumnSpans(std::move(raw));
    }
    std::vector<uint32_t> walkable; std::vector<int32_t> surfaceY;
    nav::FilterWalkableSpans(s.hf, cfg, mergedPerCol, walkable, surfaceY);
    std::vector<uint32_t> dist;
    nav::BuildDistanceField(s.hf, cfg, walkable, surfaceY, dist);
    const uint32_t maxDist = nav::MaxDistOf(dist);
    std::vector<uint32_t> region;
    const uint32_t regionCount = nav::BuildRegions(s.hf, cfg, walkable, surfaceY, dist, maxDist, region);
    std::vector<nav::Contour> contours;
    nav::TraceContours(s.hf, region, regionCount, contours);
    for (auto& cc : contours) {
        std::vector<nav::ContourVertex> sv;
        nav::SimplifyContour(cc.verts, kMaxError, sv);
        cc.verts = sv;
    }
    nav::BuildPolyMesh(contours, s.polys);

    // Flatten contour verts region-by-region + per-region vertex base (the --nav-path layout).
    std::vector<uint32_t> vOff((size_t)regionCount, 0u);
    {
        std::vector<int> contourOfRegion((size_t)regionCount + 1u, -1);
        for (size_t ci = 0; ci < contours.size(); ++ci)
            contourOfRegion[(size_t)contours[ci].region] = (int)ci;
        uint32_t base = 0u;
        for (uint32_t R = 1u; R <= regionCount; ++R) {
            vOff[(size_t)(R - 1u)] = base;
            const int ci = contourOfRegion[(size_t)R];
            if (ci < 0) continue;
            const auto& vv = contours[(size_t)ci].verts;
            for (const auto& v : vv) { s.flatVerts.push_back(v.x); s.flatVerts.push_back(v.z); }
            base += (uint32_t)vv.size();
        }
    }
    const uint32_t kPolyCount = (uint32_t)s.polys.size();
    s.polyVertBase.assign((size_t)kPolyCount, 0u);
    for (uint32_t pi = 0; pi < kPolyCount; ++pi) {
        const uint32_t R = s.polys[pi].region;
        s.polyVertBase[pi] = (R >= 1u && R <= regionCount) ? vOff[R - 1u] : 0u;
    }
    nav::ComputePolyCentroids(s.polys, s.flatVerts, s.polyVertBase, s.cx, s.cz);
    s.nComp = nav::ConnectedComponents(s.polys, s.comp);
    return s;
}

// ----- BuildAi2Scene(): the canonical scene = the navmesh + a fixed agent + target + the query ---------
// The deterministic AI2 query scenario (== the showcase + the test): build the canonical navmesh, place
// the agent at a poly in the LARGEST component, the target at a DIFFERENT poly in that component (so the
// closest reachable candidate is the headline), and build the ring query around the anchor. The agent's
// poly is the lowest-id poly of the largest component (nav::SelectStartGoal's start); the target poly is
// that component's farthest poly (its goal) — so the query asks "which ring point near the anchor is
// closest to the target AND nav-reachable from the agent?". All integer; deterministic.
//
// Returns the scene by out-params (the navmesh is large; avoid copying). The anchor sits at the agent's
// world position; the ring radius is a couple of world units; the target world position is the goal poly's
// centroid mapped back to Q16.16 world (voxel / navScale << kFrac).
inline EqsQuery BuildAi2Scene(const NavScene& scene) {
    EqsQuery q;
    // Deterministic agent + target polys: the largest component's start + goal (nav::SelectStartGoal).
    uint32_t startPoly = 0u, goalPoly = 0u;
    nav::SelectStartGoal(scene.polys, scene.cx, scene.cz, startPoly, goalPoly);

    // The agent's voxel cell = the start poly's centroid; its world anchor = that voxel / navScale.
    q.agentVx = scene.cx[startPoly];
    q.agentVz = scene.cz[startPoly];
    auto voxelToWorld = [&](int32_t v) -> fpx::fx {
        // voxel → world units (v / navScale) → Q16.16 (<< kFrac). Integer (truncating). y stays 0.
        return (fpx::fx)(((int64_t)v / (int64_t)scene.navScale) << fpx::kFrac);
    };
    q.anchor.x = voxelToWorld(scene.cx[startPoly]);
    q.anchor.y = 0;
    q.anchor.z = voxelToWorld(scene.cz[startPoly]);

    // The target = the goal poly's centroid in Q16.16 world.
    q.target.x = voxelToWorld(scene.cx[goalPoly]);
    q.target.y = 0;
    q.target.z = voxelToWorld(scene.cz[goalPoly]);

    q.radius = kOne;                 // a 1-world-unit ring around the agent
    q.count  = kRingDirCount;        // the full 16-point ring
    return q;
}

}  // namespace ai
}  // namespace hf
