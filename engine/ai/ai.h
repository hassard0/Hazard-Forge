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

// =====================================================================================================
// Slice AI3 — LINE-OF-SIGHT / PERCEPTION: a DETERMINISTIC INTEGER segment-vs-AABB visibility test.
// =====================================================================================================
// AI1 (decision tree) + AI2 (environment queries) above are BYTE-FROZEN — everything below is APPEND-ONLY.
// AI3 adds PERCEPTION: can the agent SEE the target? The engine has NO line-of-sight / raycast / segment-
// intersection on a deterministic path (the only ray test is the FLOAT GPU ray tracer, which is non-
// deterministic across vendors). LOS is INVENTED here as a PURE-INTEGER, bit-identical visibility test —
// because the textbook float DDA (tMax/tDelta) and any float parametric `t` diverge across compilers/
// backends. AI4 (perception->decision) and AI5 (rollback divergence) ride on this being EXACT.
//
// THE FORMULATION: segment-vs-AABB by an INTEGER SLAB TEST (no float `t`, no division). For a segment
// p0->p1 (direction d = p1-p0) vs an axis-aligned box [bmin, bmax], the slab test finds the parametric
// entry/exit `t` per axis. The float version DIVIDES (t = (bmin.a - p0.a)/d.a) and compares floats — BOTH
// non-deterministic. The INTEGER version keeps each `t` as a FRACTION num/den (NORMALIZED to den > 0) and
// compares two fractions by CROSS-MULTIPLICATION (a/b < c/d  <=>  a*d < c*b for b,d > 0 — int64 inter-
// mediates, the navmesh.h:126 PointInTriXZ integer edge-function discipline), so there is NO division and
// NO float. The segment hits the box iff max(entry_t over axes) <= min(exit_t over axes) AND that overlap
// intersects [0,1] (clip the segment to its endpoints). LineOfSight = NOT (the segment hits ANY blocker).
//
// THE PINNED TIE/BOUNDARY RULE: a segment that merely TOUCHES a box (grazes a face/edge/corner, or whose
// tEnter == tExit at a single point, or whose entry/exit coincides exactly with t==0 / t==1) is treated as
// NOT blocked. Concretely the slab overlap test uses STRICT `<` at the box faces (tEnter < tExit, not <=),
// so a measure-zero grazing contact does NOT occlude. Rationale: an agent standing flush against a wall,
// or a sight-line skimming a corner, should still SEE past it — the binary visible/occluded answer favors
// visibility at the boundary. This rule is applied IDENTICALLY on every axis + at the [0,1] clip, so two
// runs and both backends agree by construction (it is all integer comparison). A segment that PENETRATES a
// box interior (a positive-measure overlap) IS blocked. A segment whose endpoint lies strictly INSIDE a
// box is blocked (the interior overlap is non-degenerate). DEGENERATE p0==p1 (a zero-length segment): the
// "from" point only sees itself; it is blocked iff that point is strictly inside a box (the same interior
// rule). AXIS-PARALLEL degenerate (d.a == 0 on some axis): no slab fraction exists for that axis; instead
// the segment is inside that slab iff p0.a is STRICTLY between bmin.a and bmax.a (bmin.a < p0.a < bmax.a) —
// the same strict-interior rule, NO division. If p0.a is on/outside the slab face the segment misses the
// box on that axis (NOT blocked). PURE INTEGER, deterministic, bit-identical CPU/Vulkan/Metal.
//
// COORDINATE BOUND (overflow safety): coords are Q16.16 world units. A cross-multiply forms num*den where
// num = (boxFace - p0.a) is a difference of two coords and den = |d.a| is a difference of two coords. To
// keep num*den within int64 (|product| < 2^63 ≈ 9.22e18) with margin, the LOS scene keeps |coord| <= ~256
// world units (|coord| <= 256<<16 ≈ 1.68e7 -> |num|,|den| <= ~3.4e7 -> |num*den| <= ~1.2e15 << 2^63). The
// AI3 showcase scene (a ~16-unit arena) is far inside this; LineOfSight DEFENSIVELY notes the bound but
// does not clamp (the bound is a scene contract, like fpx.h's +-32768 position bound).

// ----- AiBlocker: an axis-aligned integer box obstacle (the LOS occluder) ------------------------------
// min/max are inclusive Q16.16 world corners (min.a <= max.a per axis). The 2D XZ horizontal-plane LOS for
// the NPC sample uses the X and Z extents; the Y extent is carried for provenance (a future 3D refinement)
// but the 2D test (LineOfSightXZ) reads only X/Z. A point is strictly inside iff min.a < p.a < max.a on
// BOTH tested axes (the pinned strict-interior rule). Structurally an fpx::FxAabb (lo/hi) — we name it
// AiBlocker for the AI vocabulary; it is trivially convertible.
struct AiBlocker {
    fpx::FxVec3 min, max;   // inclusive Q16.16 world corners (min.a <= max.a)
};

// ----- StrictlyBetween(v, lo, hi): the integer strict-interior predicate (lo < v < hi) -----------------
// Pure integer compare. Touching a face (v == lo or v == hi) is NOT inside (the pinned boundary rule).
inline bool StrictlyBetween(fpx::fx v, fpx::fx lo, fpx::fx hi) {
    return lo < v && v < hi;
}

// ----- PointInsideBlockerXZ(p, b): is p STRICTLY inside the blocker's XZ footprint? --------------------
// Used for the degenerate cases (a zero-length segment / an endpoint inside a box). Strict on BOTH X and Z
// (the pinned interior rule); a point on a face/edge is NOT inside. Pure integer.
inline bool PointInsideBlockerXZ(const fpx::FxVec3& p, const AiBlocker& b) {
    return StrictlyBetween(p.x, b.min.x, b.max.x) && StrictlyBetween(p.z, b.min.z, b.max.z);
}

// ----- SegmentHitsBlockerXZ(p0, p1, b): the INTEGER slab test on the XZ plane ---------------------------
// Returns true iff the segment p0->p1 PENETRATES the blocker's XZ footprint (a positive-measure overlap),
// per the pinned strict-boundary rule (a grazing/touching contact returns false). PURE INTEGER, NO float,
// NO division — every `t` is a fraction num/den (den > 0 after sign normalization), every comparison is a
// cross-multiplication in int64 (the PointInTriXZ edge-function discipline). The interval algebra:
//
//   For each tested axis (X, then Z) with direction component dc = p1.a - p0.a:
//     * dc == 0 (axis-parallel): the segment is inside the slab for ALL t iff p0.a is STRICTLY between
//       bmin.a and bmax.a; otherwise it misses the box -> return false (NOT blocked). No fraction added.
//     * dc != 0: the segment crosses the slab faces at entry_t = (entryFace - p0.a)/dc and
//       exit_t = (exitFace - p0.a)/dc, where (entryFace, exitFace) = (bmin.a, bmax.a) if dc > 0 else
//       (bmax.a, bmin.a) [so entry_t <= exit_t]. Each is normalized to a den > 0 fraction.
//   The box is hit iff the per-axis [entry_t, exit_t] intervals, intersected with [0, 1], have a
//   POSITIVE-measure overlap: max(entry_t, 0) < min(exit_t, 1) (STRICT — the grazing rule). All compares
//   are cross-multiplications; the running max-entry / min-exit are kept as fractions.
//
// Implementation keeps the running tEnter (a fraction, init 0/1 for the [0,1] clip) and tExit (init 1/1),
// updating tEnter = max(tEnter, entry_t) and tExit = min(tExit, exit_t) per crossing axis, then returns
// tEnter < tExit (STRICT). A fraction is (num, den) with den > 0.
struct AiFrac { int64_t num; int64_t den; };   // den > 0 invariant (sign normalized)

// FracLess(a, b): a.num/a.den < b.num/b.den, given a.den>0 && b.den>0. Cross-multiply in int64 (the scene
// coordinate bound keeps num*den < 2^63). Pure integer.
inline bool FracLess(const AiFrac& a, const AiFrac& b) {
    return a.num * b.den < b.num * a.den;
}

inline bool SegmentHitsBlockerXZ(const fpx::FxVec3& p0, const fpx::FxVec3& p1, const AiBlocker& b) {
    // The running hit interval, clipped to [0,1]: tEnter starts at 0/1, tExit at 1/1.
    AiFrac tEnter{0, 1};
    AiFrac tExit {1, 1};

    // Process X then Z (the 2D XZ footprint). A fixed-order, branch-deterministic loop.
    const fpx::fx p0a[2] = {p0.x, p0.z};
    const fpx::fx p1a[2] = {p1.x, p1.z};
    const fpx::fx bmin[2] = {b.min.x, b.min.z};
    const fpx::fx bmax[2] = {b.max.x, b.max.z};

    for (int ax = 0; ax < 2; ++ax) {
        const int64_t dc = (int64_t)p1a[ax] - (int64_t)p0a[ax];   // direction component (int64, no overflow)
        const int64_t lo = (int64_t)bmin[ax];
        const int64_t hi = (int64_t)bmax[ax];
        const int64_t p  = (int64_t)p0a[ax];
        if (dc == 0) {
            // Axis-parallel: inside the slab for all t iff p0.a STRICTLY between bmin/bmax; else MISS.
            if (!(lo < p && p < hi)) return false;   // outside/on the slab face -> not blocked (strict rule)
            continue;                                // inside the slab on this axis: no fraction constraint
        }
        // entry/exit faces ordered so entry_t <= exit_t (depends on the sign of dc).
        int64_t entryFace, exitFace;
        if (dc > 0) { entryFace = lo; exitFace = hi; }
        else        { entryFace = hi; exitFace = lo; }
        // entry_t = (entryFace - p)/dc, exit_t = (exitFace - p)/dc. Normalize to den > 0 (flip sign if dc<0).
        AiFrac entryT{entryFace - p, dc};
        AiFrac exitT { exitFace - p, dc};
        if (dc < 0) { entryT.num = -entryT.num; entryT.den = -entryT.den;
                      exitT.num  = -exitT.num;  exitT.den  = -exitT.den; }
        // tEnter = max(tEnter, entry_t); tExit = min(tExit, exit_t) — fraction compares (cross-multiply).
        if (FracLess(tEnter, entryT)) tEnter = entryT;
        if (FracLess(exitT,  tExit )) tExit  = exitT;
        // Early-out: an empty (or grazing) interval can never become a positive overlap (intervals only
        // shrink) -> not blocked. STRICT: tEnter >= tExit means no positive-measure overlap.
        if (!FracLess(tEnter, tExit)) return false;
    }
    // A positive-measure overlap of all axis slabs with [0,1] -> the segment penetrates the box.
    return FracLess(tEnter, tExit);
}

// ----- LineOfSight(from, to, blockers, count): true (visible) iff NO blocker occludes the segment -------
// The PERCEPTION primitive: returns false (occluded) iff the segment from->to PENETRATES any blocker's XZ
// footprint, true (visible) otherwise. Handles the degenerate from==to (sees itself unless strictly inside
// a box). PURE INTEGER, deterministic, bit-identical CPU/Vulkan/Metal — the strict-zero cross-vendor LOS.
inline bool LineOfSight(const fpx::FxVec3& from, const fpx::FxVec3& to,
                        const AiBlocker* blockers, int count) {
    const bool degenerate = (from.x == to.x && from.z == to.z);
    for (int i = 0; i < count; ++i) {
        const AiBlocker& b = blockers[i];
        if (degenerate) {
            if (PointInsideBlockerXZ(from, b)) return false;   // zero-length: blocked iff strictly inside
        } else {
            if (SegmentHitsBlockerXZ(from, to, b)) return false;
        }
    }
    return true;
}

// ----- The AI3 perception blackboard keys (the canonical perception write) -----------------------------
// kBbCanSeeTarget carries the binary visibility (1 visible / 0 occluded); kBbLastSeenCellX/Z carry the
// last-seen target cell (integer voxel) when visible (left UNCHANGED when occluded — a "last known
// position" memory). Distinct keys from AI1's kBbEnemyClose(0)/kBbState(1) and within [0,kMaxBbKeys).
inline constexpr int kBbCanSeeTarget = 2;   // output: 1 if the agent can see the target, else 0
inline constexpr int kBbLastSeenCellX = 3;  // output: the target's last-seen integer cell x (when visible)
inline constexpr int kBbLastSeenCellZ = 4;  // output: the target's last-seen integer cell z (when visible)

// ----- WritePerception(bb, canSee, lastSeenX, lastSeenZ): the perception->blackboard helper ------------
// Writes canSeeTarget (1/0). When canSee, ALSO updates the last-seen cell (a "I saw the target HERE"
// memory the decision tree can chase toward); when occluded, the last-seen cell is LEFT UNCHANGED (the
// agent remembers where it last saw the target). Pure integer mutation; deterministic.
inline void WritePerception(Blackboard& bb, bool canSee, int32_t lastSeenX, int32_t lastSeenZ) {
    bb.Set(kBbCanSeeTarget, canSee ? 1 : 0);
    if (canSee) {
        bb.Set(kBbLastSeenCellX, lastSeenX);
        bb.Set(kBbLastSeenCellZ, lastSeenZ);
    }
}

// ----- The canonical AI3 line-of-sight scene (the showcase + the test share ONE scene) -----------------
// A small 2D arena (XZ), Q16.16 world units. An agent at one corner, a target at the opposite corner, and
// a set of axis-aligned box blockers between them (so the direct lane to the target is occluded while side
// lanes are clear — the "shadow" the raster must show). A fixed raster grid spans the arena; the showcase
// runs LineOfSight(agent, cellCenter) for every cell. All integer; deterministic by construction. The
// arena is kArenaCells x kArenaCells world units (well inside the +-256 coordinate bound).
inline constexpr int kArenaCells = 16;            // the raster grid is kArenaCells x kArenaCells
inline constexpr int kMaxBlockers = 8;            // the fixed blocker count bound for the scene

struct Ai3Scene {
    fpx::FxVec3 agent;                            // the LOS source (Q16.16 world)
    fpx::FxVec3 target;                           // the LOS target (Q16.16 world)
    std::vector<AiBlocker> blockers;             // the axis-aligned box occluders
    int gridW = kArenaCells, gridH = kArenaCells; // the raster grid dims (cells)
};

// CellToWorld helpers: a cell index maps to the Q16.16 world coord of its CENTER (cell c -> (c + 0.5)
// world units). PURE INTEGER: (c*2 + 1) half-units, i.e. ((2c+1) * kOne) / 2 = (2c+1) << (kFrac-1).
inline fpx::fx CellCenterWorld(int c) {
    return (fpx::fx)(((int64_t)(2 * c + 1)) << (fpx::kFrac - 1));   // (c + 0.5) world units in Q16.16
}

// ----- BuildAi3Scene(): the canonical agent + target + blockers + grid ---------------------------------
// Agent at cell (1,1)'s center; target at cell (14,14)'s center (the diagonal). Three axis-aligned box
// blockers straddling the middle so the DIRECT diagonal lane is occluded but clear side lanes exist — the
// raster shows a coherent shadow behind each box. The blocker corners are EXACT integer world units (whole
// cells, no half-cell ambiguity at the showcase tie cases). Deterministic.
inline Ai3Scene BuildAi3Scene() {
    Ai3Scene s;
    s.gridW = kArenaCells; s.gridH = kArenaCells;
    s.agent.x  = CellCenterWorld(1);  s.agent.y  = 0; s.agent.z  = CellCenterWorld(1);
    s.target.x = CellCenterWorld(14); s.target.y = 0; s.target.z = CellCenterWorld(14);

    auto box = [](int x0, int z0, int x1, int z1) {
        // a box on WHOLE world-unit corners (xc..x1) x (zc..z1), Q16.16. y spans a nominal [0,2] (provenance).
        AiBlocker b;
        b.min.x = (fpx::fx)((int64_t)x0 << fpx::kFrac); b.min.z = (fpx::fx)((int64_t)z0 << fpx::kFrac);
        b.max.x = (fpx::fx)((int64_t)x1 << fpx::kFrac); b.max.z = (fpx::fx)((int64_t)z1 << fpx::kFrac);
        b.min.y = 0; b.max.y = (fpx::fx)((int64_t)2 << fpx::kFrac);
        return b;
    };
    // A central diagonal wall (broken into boxes) blocking the agent->target line; a clear side lane stays.
    s.blockers.push_back(box(6,  4, 8,  9));   // a vertical bar mid-arena (occludes the direct diagonal)
    s.blockers.push_back(box(9,  9, 12, 11));  // a horizontal bar lower-right (a second shadow caster)
    s.blockers.push_back(box(3,  10, 5, 12));  // a small box lower-left (an isolated shadow)
    return s;
}

// =====================================================================================================
// Slice AI4 — THE NPC AGENT IN THE GAMEPLAY TICK (sense -> decide -> query -> path -> act).
// =====================================================================================================
// AI1 (decision tree) + AI2 (environment queries) + AI3 (line-of-sight) above are BYTE-FROZEN —
// everything below is APPEND-ONLY. AI4 COMPOSES the three frozen primitives + the frozen navmesh A*
// into a working NPC: one deterministic per-agent update, run for every agent in a FIXED array order,
// that (1) PERCEIVES the player (AI3 LineOfSight + integer distance), (2) DECIDES patrol-vs-chase (the
// AI1 TickTree on the perception blackboard), (3) QUERIES a destination cell (AI2 RunQuery over the
// navmesh), (4) PATHS to it (nav::FindPath -> a poly corridor), and (5) ACTS by stepping its position
// toward the next corridor centroid. The whole pass is PURE INTEGER (Q16.16 + the frozen integer
// primitives) in a FIXED order, so two StepAiWorld sequences are byte-identical BY CONSTRUCTION, and
// (like AI1-AI3) the cross-backend bar is strict 0.0000.
//
// THE order[]-KEYED CONTRACT: the agents are a parallel std::vector<AiAgent> updated STRICTLY in array
// index order (the index IS the agent's pinned identity — the verdict order[] discipline). Each beat
// mutates the agent in place with integer math; no float, no allocator-dependent container drives the
// order, so every peer/platform re-derives the same agent states.
//
// COMPOSE, DON'T MODIFY: StepAiWorld runs StepAi (the AI pass) then steps the agent transforms — it
// does NOT edit verdict::StepWorld (a parallel composition, the IK4/IK5 "don't edit the frozen step"
// discipline). The agent IS conceptually a verdict entity (its pos is the gameplay Transform2D.pos);
// AI4 keeps the AI state in the parallel AiAgent array keyed by index and steps the pos directly, so
// the frozen verdict tick is untouched.

// ----- AgentState: the NPC behaviour state (the AI1 tree's kStatePatrol/kStateChase, named for AI4) -----
inline constexpr int kAgentPatrol = (int)kStatePatrol;   // 1 — patrol a ring point (player not visible/far)
inline constexpr int kAgentChase  = (int)kStateChase;    // 2 — chase toward the player's last-seen cell

// ----- The AI4 blackboard threshold + scene tuning (integer / Q16.16) ----------------------------------
// The DECISION beat writes a Q16.16 "enemy proximity" score into kBbEnemyClose that the AI1 tree's
// condLeaf tests against kEnemyThreshold (1.0). We map "player visible AND within kChaseRangeWorld WORLD
// units" -> a proximity >= 1.0 (CHASE), else < 1.0 (PATROL). The proximity is a pure-integer function of
// the visibility bit + the integer WORLD Manhattan distance (NO float, NO sqrt): visible && |dx|+|dz| <=
// range -> kOne, else 0. World units (not voxels) so the threshold reads intuitively in the agent space.
inline constexpr int32_t kChaseRangeWorld = 6;   // chase iff the player is within this many WORLD units (Manhattan)

// The AI4 destination-query ring radius (Q16.16 WORLD units). Sized a few world units so a ring candidate
// reaches a NEIGHBOURING nav poly (the navmesh polys are ~3-5 voxels = ~1 world unit apart at the scene's
// navScale=4) -> RunQuery picks a destination poly DIFFERENT from the agent's own poly -> nav::FindPath
// returns a real multi-poly corridor to steer along (a single-poly result would be a degenerate path).
inline constexpr fpx::fx kAiRingRadius = (fpx::fx)(3 * (1 << fpx::kFrac));   // 3.0 world units in Q16.16

// ----- AiAgent: the per-agent AI state (a parallel-array entry, index == the pinned agent identity) -----
// pos is the agent's Q16.16 world position (the gameplay Transform2D.pos); bb is its perception/decision
// blackboard; treeIndex selects the decision tree (all agents share the canonical BuildAi1Tree in the
// AI4 scene, but the field keeps the door open for per-agent trees); state is the last decided
// AgentState; navCell{X,Z} is the agent's current integer nav cell; navTarget is the chosen destination
// poly id; corridor + corridorStep are the FindPath corridor and the index of the next centroid to steer
// toward. All integer; the whole struct is a deterministic function of the inputs.
struct AiAgent {
    int          treeIndex = 0;                 // which decision tree (index into a tree table; 0 = canonical)
    Blackboard   bb{};                          // the perception/decision blackboard (AI1/AI3 keys)
    fpx::FxVec3  pos{};                          // Q16.16 world position (the agent's Transform2D.pos)
    int          state = kAgentPatrol;          // the last decided AgentState (kAgentPatrol / kAgentChase)
    int32_t      navCellX = 0, navCellZ = 0;    // the agent's current integer nav voxel cell
    uint32_t     navPoly  = nav::kNoCameFrom;   // the agent's current nav poly id (nearest centroid)
    int          navTarget = -1;                // the chosen destination poly id (-1 = none)
    uint32_t     patrolPoly = nav::kNoCameFrom; // the fixed PATROL objective poly (a reachable far poly in the agent's component)
    int32_t      destWorldX = 0, destWorldZ = 0;// the destination steer point in Q16.16 world (next centroid)
    std::vector<uint32_t> corridor{};           // the FindPath corridor (poly-id sequence), empty if none
    int          corridorStep = 0;              // index of the next corridor poly to steer toward
};

// ----- voxelToWorld / worldToVoxel for the agent (the AI2 NavScene mapping, integer) -------------------
// The navmesh centroids are integer voxel units; the agent world coords are Q16.16. The AI2 BuildAi2Scene
// used voxel/navScale -> world. AI4 reuses that SAME mapping so the agents live in the navmesh's space.
inline fpx::fx AiVoxelToWorld(const NavScene& scene, int32_t v) {
    // voxel -> world units (v / navScale) -> Q16.16 (<< kFrac). Integer (truncating); the AI2 convention.
    return (fpx::fx)(((int64_t)v / (int64_t)scene.navScale) << fpx::kFrac);
}
inline void AiWorldToVoxel(const NavScene& scene, const fpx::FxVec3& w, int32_t& vx, int32_t& vz) {
    vx = (w.x >> fpx::kFrac) * scene.navScale;
    vz = (w.z >> fpx::kFrac) * scene.navScale;
}

// ----- AiAgentStep: the agent's per-tick integer move toward its current steer point (Q16.16) ----------
// The ACT beat. The agent moves a fixed Q16.16 step toward (destWorldX, destWorldZ); the move is clamped
// so it never overshoots the steer point (integer min of the remaining delta and the step). Pure integer:
// the per-axis signed delta, its magnitude clamped to kStepLen, applied to pos. NO float, NO sqrt — an
// axis-independent (Manhattan-style) move so the whole act is integer + deterministic (a Euclidean
// normalize would need a divide; the axis-clamped move is exact + bit-stable).
inline constexpr fpx::fx kAgentStepLen = kOne / 4;   // 0.25 world units / tick (Q16.16)

inline void AiAgentMoveToward(AiAgent& a, int32_t destX, int32_t destZ, fpx::fx stepLen) {
    auto moveAxis = [stepLen](fpx::fx cur, fpx::fx dst) -> fpx::fx {
        const int64_t delta = (int64_t)dst - (int64_t)cur;
        if (delta == 0) return cur;
        const int64_t mag = delta < 0 ? -delta : delta;
        const int64_t step = mag < (int64_t)stepLen ? mag : (int64_t)stepLen;   // never overshoot
        return (fpx::fx)(cur + (delta < 0 ? -step : step));
    };
    a.pos.x = moveAxis(a.pos.x, destX);
    a.pos.z = moveAxis(a.pos.z, destZ);
}

// ----- StepAi(agents, scene, blockers, count, playerPos): the FIVE-BEAT per-agent pass, FIXED order ----
// For each agent in array index order: PERCEIVE (LineOfSight to the player + integer cell distance ->
// the blackboard) -> DECIDE (TickTree on the canonical AI1 tree -> kBbState = PATROL/CHASE) -> QUERY
// (RunQuery a ring around the agent toward the player's last-seen cell [chase] or a fixed patrol anchor
// [patrol] -> a destination poly) -> PATH (nav::FindPath agent poly -> destination poly -> corridor) ->
// (the ACT beat sets the next steer point; the move itself is StepAiWorld so the perceive/decide all run
// BEFORE any agent moves — a fixed read-then-write order). PURE INTEGER, deterministic. `trees` is the
// per-treeIndex decision-tree table (the AI4 scene passes {BuildAi1Tree()}).
inline void StepAi(std::vector<AiAgent>& agents, const NavScene& scene,
                   const AiBlocker* blockers, int blockerCount,
                   const fpx::FxVec3& playerPos, const std::vector<DecisionTree>& trees) {
    // The player's nav cell (integer voxel) — the chase target + the last-seen cell.
    int32_t playerVx = 0, playerVz = 0;
    AiWorldToVoxel(scene, playerPos, playerVx, playerVz);

    for (size_t i = 0; i < agents.size(); ++i) {
        AiAgent& a = agents[i];

        // --- (1) PERCEIVE: LineOfSight(agent, player) + the integer WORLD distance -> the blackboard ---
        const bool canSee = LineOfSight(a.pos, playerPos, blockers, blockerCount);
        AiWorldToVoxel(scene, a.pos, a.navCellX, a.navCellZ);
        // WritePerception: canSeeTarget + (when visible) the player's last-seen cell (a memory, AI3).
        WritePerception(a.bb, canSee, playerVx, playerVz);
        // The "enemy proximity" the AI1 condLeaf tests: visible AND within chase range -> 1.0, else 0.
        // The chase test is a WORLD-space Manhattan distance (|dx|+|dz|), pure integer (NO sqrt).
        const int64_t wdx = (int64_t)playerPos.x - (int64_t)a.pos.x;
        const int64_t wdz = (int64_t)playerPos.z - (int64_t)a.pos.z;
        const int64_t worldDist = (wdx < 0 ? -wdx : wdx) + (wdz < 0 ? -wdz : wdz);
        const int64_t chaseRange = (int64_t)kChaseRangeWorld << fpx::kFrac;   // world units -> Q16.16
        const int32_t proximity = (canSee && worldDist <= chaseRange) ? kEnemyThreshold : 0;
        a.bb.Set(kBbEnemyClose, proximity);

        // --- (2) DECIDE: the frozen AI1 TickTree -> kBbState (CHASE if proximity >= 1.0, else PATROL) ---
        const DecisionTree& tree = trees[(size_t)((a.treeIndex >= 0 && a.treeIndex < (int)trees.size())
                                                  ? a.treeIndex : 0)];
        TickTree(tree, a.bb);
        a.state = (int)a.bb.Get(kBbState);

        // --- (3) QUERY: a ring around the agent; the scoring target = chase (player last-seen) or -------
        //     patrol (a fixed ring anchor offset from the agent). RunQuery picks the closest reachable.
        a.navPoly = PolyOfPoint(scene, a.navCellX, a.navCellZ);
        EqsQuery q;
        q.anchor  = a.pos;
        q.radius  = kAiRingRadius;        // the AI4 ring radius (a few world units so candidates reach
                                          // neighbouring polys -> a real multi-poly corridor)
        q.count   = kRingDirCount;
        q.agentVx = a.navCellX;
        q.agentVz = a.navCellZ;
        if (a.state == kAgentChase) {
            // Chase: steer toward the player's LAST-SEEN cell (the AI3 memory in the blackboard).
            const int32_t lx = a.bb.Get(kBbLastSeenCellX);
            const int32_t lz = a.bb.Get(kBbLastSeenCellZ);
            q.target.x = AiVoxelToWorld(scene, lx); q.target.y = 0;
            q.target.z = AiVoxelToWorld(scene, lz);
        } else {
            // Patrol: steer toward the agent's FIXED patrol objective poly (a reachable far poly in its
            // own component — set at scene build). A deterministic ring objective that yields a real path.
            const uint32_t pp = (a.patrolPoly != nav::kNoCameFrom &&
                                 a.patrolPoly < (uint32_t)scene.polys.size())
                                ? a.patrolPoly : a.navPoly;
            if (pp != nav::kNoCameFrom && pp < (uint32_t)scene.polys.size()) {
                q.target.x = AiVoxelToWorld(scene, scene.cx[pp]); q.target.y = 0;
                q.target.z = AiVoxelToWorld(scene, scene.cz[pp]);
            } else {
                q.target = a.pos;   // no objective -> degenerate (hold position)
            }
        }
        const EqsResult r = RunQuery(q, scene);
        // Map the chosen ring candidate -> its nav poly = the destination poly.
        if (r.bestIndex >= 0) {
            const std::vector<fpx::FxVec3> cands = GenerateRing(q.anchor, q.radius, q.count);
            const fpx::FxVec3& dst = cands[(size_t)r.bestIndex];
            int32_t dvx, dvz; AiWorldToVoxel(scene, dst, dvx, dvz);
            a.navTarget = (int)PolyOfPoint(scene, dvx, dvz);
        } else {
            a.navTarget = (int)a.navPoly;
        }

        // --- (4) PATH: nav::FindPath agent poly -> destination poly -> the corridor --------------------
        a.corridor.clear();
        a.corridorStep = 0;
        if (a.navPoly != nav::kNoCameFrom && a.navTarget >= 0) {
            nav::FindPath(scene.polys, scene.cx, scene.cz, a.navPoly, (uint32_t)a.navTarget, a.corridor);
        }

        // --- (5) ACT (set the steer point): the next corridor centroid (or the destination directly) ---
        // The MOVE itself happens in StepAiWorld AFTER every agent has perceived/decided (read-then-write).
        if (a.corridor.size() >= 2) {
            // Steer toward the SECOND corridor poly's centroid (the next waypoint, not the current cell).
            const uint32_t nextPoly = a.corridor[1];
            a.corridorStep = 1;
            a.destWorldX = AiVoxelToWorld(scene, scene.cx[nextPoly]);
            a.destWorldZ = AiVoxelToWorld(scene, scene.cz[nextPoly]);
        } else if (a.state == kAgentChase) {
            // No corridor (already at the goal poly / single-poly path): steer straight toward the player.
            a.destWorldX = playerPos.x;
            a.destWorldZ = playerPos.z;
        } else {
            // Patrol with no corridor: hold position (steer to self -> a zero move).
            a.destWorldX = a.pos.x;
            a.destWorldZ = a.pos.z;
        }
    }
}

// ----- StepAiWorld(agents, scene, blockers, count, playerPos, trees): the parallel composition wrapper --
// Runs the AI pass (StepAi — perceive/decide/query/path/act, all agents in FIXED order) THEN steps every
// agent's transform toward its chosen steer point (a second FIXED-order pass — read-then-write so the
// perception is over the PRE-move positions). Does NOT touch verdict::StepWorld. Pure integer; two
// StepAiWorld sequences over the same inputs are byte-identical BY CONSTRUCTION.
inline void StepAiWorld(std::vector<AiAgent>& agents, const NavScene& scene,
                        const AiBlocker* blockers, int blockerCount,
                        const fpx::FxVec3& playerPos, const std::vector<DecisionTree>& trees) {
    StepAi(agents, scene, blockers, blockerCount, playerPos, trees);
    // ACT (move): step each agent toward its steer point, in FIXED array order.
    for (size_t i = 0; i < agents.size(); ++i) {
        AiAgent& a = agents[i];
        AiAgentMoveToward(a, a.destWorldX, a.destWorldZ, kAgentStepLen);
    }
}

// ----- DigestAgents(agents): FNV-1a over each agent's pos/state/nav/corridor in FIXED order -------------
// The AI4 proof currency: folds, for each agent in array index order, its pos (x,y,z), state, navTarget,
// corridorStep, and the corridor poly ids — all LSB-first, the DigestBlackboard byte order. Two equal
// agent arrays hash IDENTICALLY; any changed agent field changes the digest. Mirrors verdict.h's FNV
// constants. Pure integer.
inline uint64_t DigestAgents(const std::vector<AiAgent>& agents) {
    uint64_t h = 1469598103934665603ull;            // the FNV-1a 64-bit offset basis
    auto fold32 = [&h](uint32_t v) {
        for (int b = 0; b < 4; ++b) { h ^= (uint64_t)(uint8_t)(v >> (b * 8)); h *= 1099511628211ull; }
    };
    for (size_t i = 0; i < agents.size(); ++i) {
        const AiAgent& a = agents[i];
        fold32((uint32_t)a.pos.x);
        fold32((uint32_t)a.pos.y);
        fold32((uint32_t)a.pos.z);
        fold32((uint32_t)a.state);
        fold32((uint32_t)a.navTarget);
        fold32((uint32_t)a.corridorStep);
        fold32((uint32_t)a.corridor.size());
        for (size_t k = 0; k < a.corridor.size(); ++k) fold32((uint32_t)a.corridor[k]);
    }
    return h;
}

// ----- Ai4Scene: the canonical AI4 scenario (the navmesh + N agents + a player + the blockers) ---------
// Bundles the AI2 NavScene + a fixed set of agents (placed at deterministic poly centroids in the
// largest component) + a movable player + a set of AiBlocker occluders (so an agent's LOS to the player
// can be toggled by the blocker — the falsifiable perception->decision beat) + the decision-tree table.
struct Ai4Scene {
    NavScene                  nav;          // the canonical navmesh (BuildNavScene)
    std::vector<AiAgent>      agents;       // the NPC agents (parallel array, index == identity)
    fpx::FxVec3               player{};     // the player position (Q16.16 world) — the perception target
    std::vector<AiBlocker>    blockers;     // the LOS occluders (can hide the player from an agent)
    std::vector<DecisionTree> trees;        // the per-treeIndex decision-tree table (0 = BuildAi1Tree)
};

// ----- BuildAi4Scene(): the navmesh + agents + player + blockers ---------------------------------------
// Builds the canonical navmesh, places the player at the largest component's GOAL poly centroid, the
// agents at deterministic poly centroids in that component (the START poly + a couple of others), and a
// blocker between an agent and the player so toggling it flips that agent CHASE<->PATROL. All integer;
// deterministic by construction. (The blocker corners are EXACT integer world units in the agents'
// world space so the LOS slab test hits clean cases.)
inline Ai4Scene BuildAi4Scene() {
    Ai4Scene s;
    s.nav = BuildNavScene();
    s.trees.push_back(BuildAi1Tree());

    // The player at the largest component's GOAL poly centroid; agents seeded around the START poly.
    uint32_t startPoly = 0u, goalPoly = 0u;
    nav::SelectStartGoal(s.nav.polys, s.nav.cx, s.nav.cz, startPoly, goalPoly);

    auto polyWorld = [&](uint32_t p, fpx::FxVec3& out) {
        out.x = AiVoxelToWorld(s.nav, s.nav.cx[p]); out.y = 0;
        out.z = AiVoxelToWorld(s.nav, s.nav.cz[p]);
    };
    polyWorld(goalPoly, s.player);

    // Place up to 3 agents at distinct polys in the largest component (start + the next two by id).
    const uint32_t comp = s.nav.comp.empty() ? 0u : s.nav.comp[startPoly];
    std::vector<uint32_t> seedPolys;
    seedPolys.push_back(startPoly);
    for (uint32_t p = 0; p < (uint32_t)s.nav.polys.size() && seedPolys.size() < 3; ++p) {
        if (p == startPoly || p == goalPoly) continue;
        if (!s.nav.comp.empty() && s.nav.comp[p] != comp) continue;
        seedPolys.push_back(p);
    }
    // The fixed PATROL objective = the component's GOAL poly (a reachable far poly) — so a patrolling
    // agent paths along a real corridor (NOT toward an unreachable other-component poly).
    for (uint32_t sp : seedPolys) {
        AiAgent a;
        a.treeIndex = 0;
        polyWorld(sp, a.pos);
        a.state = kAgentPatrol;
        a.patrolPoly = (sp == goalPoly) ? startPoly : goalPoly;   // patrol toward the far end of the component
        s.agents.push_back(a);
    }

    return s;
}

}  // namespace ai
}  // namespace hf
