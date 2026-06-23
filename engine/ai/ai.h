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

}  // namespace ai
}  // namespace hf
