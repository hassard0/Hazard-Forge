#pragma once
// Slice BT1 — DETERMINISTIC BEHAVIOR-TREE DEPTH + UTILITY AI (parallel nodes, stateful decorators with
// memory, service nodes, utility scoring). Next-tier parity gap #3 AND the closing of engine/ai/ai.h's OWN
// self-declared L37 gap: ai.h ships kSelector/kSequence/kInverter composites + an integer blackboard + EQS,
// but its banner explicitly says "no parallel/decorator-with-memory, no JSON tree loader, no utility AI."
// BT1 adds the missing behavior-tree DEPTH — the reactive/stateful node kinds UE5's Behavior Tree + Utility
// AI plugins ship, made PURE-INTEGER + bit-identical + (by construction) lockstep-replayable.
//
// ================= ai.h INTEGRATION: A NEW HEADER COMPOSING ai.h READ-ONLY (byte-UNTOUCHED) =================
// ai.h's node dispatch (TickNode) is a CLOSED switch over the five node kinds, and — critically — the AI1
// tick is STATELESS: TickNode carries no per-node memory across ticks. Decorators-with-memory (cooldown,
// retry, loop), service nodes, and the observer-abort blackboard-condition ALL need per-node state that
// PERSISTS across ticks. That is a fundamentally different execution model, so BT1 is a NEW additive header
// that COMPOSES ai.h READ-ONLY (reusing its Blackboard, the Status tri-state kRunning/kSuccess/kFailure, the
// fixed key space kMaxBbKeys, the bounded-depth guard kMaxDepth, and the FNV digest constants) rather than
// editing the frozen AI1 switch. ai.h + every ai_* golden stay byte-identical. The new model:
//
//   * A FLAT INDEX GRAPH (the ai.h nav::Poly discipline): BtxTree is a std::vector<BtxNode>; children are
//     integer INDICES into the vector (NOT pointers), evaluated in a FIXED order -> bit-stable by construction.
//   * A PARALLEL PER-NODE STATE ARRAY (BtxState), sized to the tree, that carries the decorator/service
//     MEMORY across ticks (cooldown counters, service tick counters, loop/retry counts, the observer-abort
//     "child was running" latch). The state is plain integers, mutated in place in FIXED order -> two ticks
//     over the same tree+state+blackboard are byte-identical.
//   * The blackboard is ai.h's integer Blackboard (a fixed std::array of int32 slots). BT1 uses DISTINCT
//     keys (5..14) from AI1's kBbEnemyClose(0)/kBbState(1) and AI3's kBbCanSeeTarget(2)/last-seen(3,4).
//
// ================= THE OBSERVER-ABORT SEMANTICS (the subtle part — pinned exactly) =========================
// kBtxBlackboardCondition is a decorator gating a single child subtree on a blackboard predicate that is
// RE-CHECKED EVERY TICK. It is an "observer abort" (UE5's "Abort Self / Lower Priority" on a decorator):
//   * If the predicate is TRUE this tick: tick the child normally; return its status; LATCH whether the child
//     returned kRunning (so we know a subtree is in progress under this gate).
//   * If the predicate is FALSE this tick AND the child was Running from a previous tick: this is the ABORT.
//     The decorator returns kFailure THIS tick, does NOT tick the child (the running subtree is INTERRUPTED
//     — it does not get to run on the abort tick), resets the child subtree's in-progress memory (so a later
//     re-entry starts fresh), and increments the abort counter. THE PINNED TICK: if the child ran (Running)
//     at ticks 0..K-1 and the predicate flips false at tick K, the decorator returns kFailure at tick K (the
//     abort tick), and the child's memory is frozen/reset at K (it is NOT ticked at K).
//   * If the predicate is FALSE and the child was NOT running: the gate is simply closed -> kFailure, child
//     not ticked (no abort — there was nothing in progress to interrupt).
//
// ================= UTILITY AI: A SMALL AUTHORED SCORER SET (NOT an expression language) ====================
// kBtxUtilitySelector scores each child by an INTEGER utility function chosen from a SMALL AUTHORED SET
// (kScoreConst / kScoreDirect / kScoreInverse) reading one blackboard slot each, and picks the HIGHEST score
// (tie-break: LOWEST child index — the ai.h RunQuery strict-compare discipline, but max instead of min). This
// is NOT a general expression evaluator: the scorers are three hand-written integer transforms, exactly like
// AI2's fixed two-scorer chain. More scorers compose by adding more enum cases the SAME way. Deterministic.
//
// Header-only, namespace hf::ai, PURE CPU, PURE INTEGER (no float, no transcendentals, no division in the
// tick). #includes ai/ai.h READ-ONLY/BYTE-FROZEN. NO render RHI, NO shader, NO compute. The --bt1-behavior
// showcase raster is a strict-zero integer top-down/timeline viz (RenderBt1Shot) both backends call verbatim.

#include <array>
#include <cstdint>
#include <vector>

#include "ai/ai.h"   // READ-ONLY/BYTE-FROZEN: Blackboard, Status (kRunning/kSuccess/kFailure), kMaxBbKeys,
                     // kMaxChildren, kMaxDepth — the AI1 determinism primitives BT1 composes.

namespace hf {
namespace ai {

// ===== The BT1 FNV-1a digest word (mirrors ai.h's DigestBlackboard constants — the engine-wide currency) ===
inline constexpr uint64_t kBtxFnvOffset = 1469598103934665603ull;
inline constexpr uint64_t kBtxFnvPrime  = 1099511628211ull;
inline uint64_t BtxFnvWord(uint64_t h, uint32_t v) {
    for (int b = 0; b < 4; ++b) { h ^= (uint64_t)(uint8_t)(v >> (b * 8)); h *= kBtxFnvPrime; }
    return h;
}

// ===== NODE KINDS: the reactive composites + the BT1-new depth (parallel / stateful decorators / service /
// utility). Distinct identifiers from ai.h's kSelector/kSequence/... (which live in the same namespace). =====
enum BtxKind : uint32_t {
    kBtxSelector           = 0u,  // first non-Failure child (short-circuits) — the ai.h kSelector twin
    kBtxSequence           = 1u,  // first non-Success child (short-circuits) — the ai.h kSequence twin
    kBtxInverter           = 2u,  // flips its single child's Success<->Failure (Running stays Running)
    kBtxCondLeaf           = 3u,  // tests bb.Get(bbKey) OP param -> Success/Failure (a read; op in `op`)
    kBtxActionLeaf         = 4u,  // writes param into bb.Get(bbKey) -> always Success (the side effect)
    kBtxRunningLeaf        = 5u,  // returns Running for `param` ticks then Success (a controllable task)
    kBtxParallel           = 6u,  // ticks ALL children each tick; policy-decided success/failure
    kBtxCooldown           = 7u,  // DECORATOR w/ memory: after a child Success, blocks re-entry for `param` ticks
    kBtxBlackboardCondition= 8u,  // DECORATOR (observer abort): gate a subtree on a bb predicate re-checked each tick
    kBtxRetry              = 9u,  // DECORATOR w/ memory: re-tick the child up to `param` attempts on Failure
    kBtxLoop               = 10u, // DECORATOR w/ memory: run the child `param` times (counting Successes)
    kBtxService            = 11u, // runs serviceFn every `param` ticks while active, then ticks its child
    kBtxUtilitySelector    = 12u, // scores each child (authored scorer set), ticks the HIGHEST (tie -> lowest idx)
};

// ===== Comparison operators for kBtxCondLeaf / kBtxBlackboardCondition (integer, deterministic) =============
enum BtxOp : uint32_t { kOpGE = 0u, kOpLE = 1u, kOpGT = 2u, kOpLT = 3u, kOpEQ = 4u, kOpNE = 5u };
inline bool BtxEvalOp(int32_t lhs, uint32_t op, int32_t rhs) {
    switch (op) {
        case kOpGE: return lhs >= rhs;
        case kOpLE: return lhs <= rhs;
        case kOpGT: return lhs >  rhs;
        case kOpLT: return lhs <  rhs;
        case kOpEQ: return lhs == rhs;
        case kOpNE: return lhs != rhs;
        default:    return false;
    }
}

// ===== Parallel policies: succeed/fail when ONE child does, or when ALL children do ========================
enum BtxPolicy : uint32_t { kRequireOne = 0u, kRequireAll = 1u };

// ===== The authored utility scorer set (a SMALL fixed set, NOT an expression language) =====================
// Each utility child names a scorer + an argument. kScoreConst returns the arg verbatim (a fixed priority);
// kScoreDirect returns bb.Get(arg) (higher slot value -> higher score); kScoreInverse returns
// kUtilInverseBase - bb.Get(arg) (LOWER slot value -> higher score — e.g. "flee when health is low").
enum BtxScorer : uint32_t { kScoreConst = 0u, kScoreDirect = 1u, kScoreInverse = 2u };
inline constexpr int32_t kUtilInverseBase = 55;   // the fixed base kScoreInverse subtracts from (authored)
inline int32_t BtxScore(uint32_t scorer, int32_t arg, const Blackboard& bb) {
    switch (scorer) {
        case kScoreConst:   return arg;                              // a fixed authored priority
        case kScoreDirect:  return bb.Get((int)arg);                // higher slot value -> higher score
        case kScoreInverse: return kUtilInverseBase - bb.Get((int)arg);  // lower slot value -> higher score
        default:            return 0;
    }
}

// ===== The authored service routines (a SMALL fixed set — updates the blackboard from other slots) =========
enum BtxServiceFn : uint32_t { kSvcNone = 0u, kSvcRefreshThreat = 1u };

// ===== BtxNode: a FLAT node (children are INDICES into the BtxTree, NOT pointers) ==========================
struct BtxNode {
    uint32_t kind = kBtxSelector;              // a BtxKind
    int      child[kMaxChildren] = {};         // child INDICES into the BtxTree (NOT pointers)
    int      childCount = 0;                   // number of valid entries in child[]
    int32_t  param = 0;                        // threshold / value / interval / count / attempts / cooldown / runTicks
    int      bbKey = 0;                        // the blackboard slot a leaf/condition reads or writes
    uint32_t op = kOpGE;                       // condition operator (kBtxCondLeaf / kBtxBlackboardCondition)
    uint32_t successPolicy = kRequireAll;      // parallel success policy
    uint32_t failurePolicy = kRequireOne;      // parallel failure policy
    uint32_t serviceFn = kSvcNone;             // service update routine id (kBtxService)
    uint32_t scorer[kMaxChildren] = {};        // utility: per-child scorer id (kBtxUtilitySelector)
    int32_t  scorerArg[kMaxChildren] = {};     // utility: per-child scorer argument (a slot key or a constant)
};

using BtxTree = std::vector<BtxNode>;

// ===== BtxNodeState: the PER-NODE MEMORY carried across ticks (the decorator/service state) ================
// A parallel array sized to the tree; index == node index. runLeft uses -1 as "not started" (lazy init).
struct BtxNodeState {
    int32_t cooldown     = 0;    // kBtxCooldown: ticks left blocked after a Success (0 = ready)
    int32_t svcCounter   = 0;    // kBtxService: number of times ticked (drives the every-N update)
    int32_t loopCount    = 0;    // kBtxLoop: Successes counted so far
    int32_t attempts     = 0;    // kBtxRetry: Failures counted so far
    int32_t runLeft      = -1;   // kBtxRunningLeaf: Running ticks remaining (-1 = uninitialized)
    uint8_t childRunning = 0;    // kBtxBlackboardCondition: was the gated child Running last tick (the abort latch)
};

inline std::vector<BtxNodeState> MakeBtxState(const BtxTree& tree) {
    return std::vector<BtxNodeState>(tree.size());
}

// ===== BtxRun: the per-tick + cumulative trace accumulators (the utility choice, aborts, service runs) =====
struct BtxRun {
    int tick               = 0;   // the current tick (set by the caller before each TickBtxTree)
    int utilityChoice      = -1;  // the child index chosen by the LAST utility selector this tick (-1 = none)
    int serviceRanThisTick = 0;   // 1 if a service ran this tick (reset each TickBtxTree)
    int utilityChoices     = 0;   // cumulative count of utility evaluations
    int aborts             = 0;   // cumulative observer-aborts
    int serviceRuns        = 0;   // cumulative service updates
};

// ===== RunServiceFn: the authored blackboard-updating routine (deterministic integer) =====================
// kSvcRefreshThreat: read the raw enemy distance + ammo slots (the "sensed" world state the harness writes)
// and compute the derived threat + attack-score slots the utility selector scores. Pure integer.
inline int32_t kBt1ThreatRange = 50;   // enemy within this (integer) distance registers a threat
inline int32_t kBt1AttackRange = 10;   // enemy within this distance is in attack range
inline int32_t kBt1AttackBase  = 60;   // the attack-score baseline (so ATTACK outranks CHASE in range)

// ----- BT1 blackboard key layout (5..14; distinct from AI1's 0/1 + AI3's 2/3/4) -----
inline constexpr int kBbxHealth      = 5;   // raw: agent health (world-set each tick)
inline constexpr int kBbxAmmo        = 6;   // raw: agent ammo (world-set each tick)
inline constexpr int kBbxEnemyDist   = 7;   // raw: distance to the nearest enemy (world-set each tick)
inline constexpr int kBbxThreat      = 8;   // derived (service): closeness score, 0 if far
inline constexpr int kBbxAttackScore = 9;   // derived (service): attack desirability, 0 if out of range/no ammo
inline constexpr int kBbxChoice      = 10;  // output: the chosen utility branch (kChoice*)
inline constexpr int kBbxMoveX       = 11;  // output: the parallel "move" arm result
inline constexpr int kBbxAim         = 12;  // output: the parallel "aim" arm result
inline constexpr int kBbxAttackReady = 13;  // output: 1 on the tick the cooldown-gated attack fires
inline constexpr int kBbxGate        = 14;  // the observer-abort gate key (tests)

// The chosen-branch enum (written into kBbxChoice by the utility branches).
inline constexpr int32_t kChoicePatrol = 1;
inline constexpr int32_t kChoiceChase  = 2;
inline constexpr int32_t kChoiceAttack = 3;
inline constexpr int32_t kChoiceFlee   = 4;

inline void RunServiceFn(uint32_t fn, Blackboard& bb) {
    if (fn == kSvcRefreshThreat) {
        const int32_t dist = bb.Get(kBbxEnemyDist);
        const int32_t threat = (dist <= kBt1ThreatRange) ? (kBt1ThreatRange - dist) : 0;
        bb.Set(kBbxThreat, threat);
        const int32_t ammo = bb.Get(kBbxAmmo);
        const int32_t attackScore =
            (dist <= kBt1AttackRange && ammo > 0) ? (kBt1AttackBase + threat) : 0;
        bb.Set(kBbxAttackScore, attackScore);
    }
}

// ===== ResetSubtree: clear the IN-PROGRESS memory of a subtree (the observer-abort reset) ==================
// Resets the "in flight" fields (runLeft/attempts/loopCount/childRunning) of `idx` and every descendant so a
// re-entry starts fresh, but LEAVES the persistent timers (cooldown, svcCounter) — a cooldown or a service
// cadence is a wall-independent clock that an abort of an unrelated subtree should not rewind. Pure integer.
inline void ResetSubtree(const BtxTree& tree, std::vector<BtxNodeState>& st, int idx, int depth) {
    if (idx < 0 || (size_t)idx >= tree.size()) return;
    if (depth >= kMaxDepth) return;
    BtxNodeState& s = st[(size_t)idx];
    s.runLeft = -1; s.attempts = 0; s.loopCount = 0; s.childRunning = 0;
    const BtxNode& n = tree[(size_t)idx];
    for (int c = 0; c < n.childCount && c < kMaxChildren; ++c)
        ResetSubtree(tree, st, n.child[c], depth + 1);
}

// ===== TickBtx: evaluate node `idx` (bounded recursion, FIXED child order, per-node state carried) =========
// A bad index / a depth overrun is a deterministic kFailure (never UB), the ai.h TickNode discipline. All
// integer, fixed order, in-place state mutation -> bit-identical by construction.
inline Status TickBtx(const BtxTree& tree, std::vector<BtxNodeState>& st, Blackboard& bb,
                      int idx, int depth, BtxRun& run) {
    if (idx < 0 || (size_t)idx >= tree.size()) return kFailure;
    if (depth >= kMaxDepth) return kFailure;
    const BtxNode& n = tree[(size_t)idx];
    BtxNodeState& s = st[(size_t)idx];

    switch (n.kind) {
        case kBtxSelector: {
            for (int c = 0; c < n.childCount && c < kMaxChildren; ++c) {
                const Status r = TickBtx(tree, st, bb, n.child[c], depth + 1, run);
                if (r != kFailure) return r;           // first non-Failure wins (short-circuit)
            }
            return kFailure;
        }
        case kBtxSequence: {
            for (int c = 0; c < n.childCount && c < kMaxChildren; ++c) {
                const Status r = TickBtx(tree, st, bb, n.child[c], depth + 1, run);
                if (r != kSuccess) return r;           // first non-Success wins (short-circuit)
            }
            return kSuccess;
        }
        case kBtxInverter: {
            if (n.childCount < 1) return kFailure;
            const Status r = TickBtx(tree, st, bb, n.child[0], depth + 1, run);
            if (r == kSuccess) return kFailure;
            if (r == kFailure) return kSuccess;
            return kRunning;
        }
        case kBtxCondLeaf:
            return BtxEvalOp(bb.Get(n.bbKey), n.op, n.param) ? kSuccess : kFailure;
        case kBtxActionLeaf:
            bb.Set(n.bbKey, n.param);
            return kSuccess;
        case kBtxRunningLeaf: {
            if (s.runLeft < 0) s.runLeft = n.param;    // lazy init on first entry
            if (s.runLeft > 0) { --s.runLeft; return kRunning; }
            return kSuccess;                            // Running for `param` ticks, then Success
        }
        case kBtxParallel: {
            // Tick ALL children each tick (FIXED order, for their side effects); count Success/Failure.
            int nSucc = 0, nFail = 0, nRun = 0, nTotal = 0;
            for (int c = 0; c < n.childCount && c < kMaxChildren; ++c) {
                const Status r = TickBtx(tree, st, bb, n.child[c], depth + 1, run);
                ++nTotal;
                if (r == kSuccess) ++nSucc; else if (r == kFailure) ++nFail; else ++nRun;
            }
            if (nTotal == 0) return kFailure;
            // Success policy takes precedence (pinned), then failure policy, else Running.
            const bool succeed = (n.successPolicy == kRequireOne) ? (nSucc >= 1) : (nSucc == nTotal);
            if (succeed) return kSuccess;
            const bool fail = (n.failurePolicy == kRequireOne) ? (nFail >= 1) : (nFail == nTotal);
            if (fail) return kFailure;
            return kRunning;
        }
        case kBtxCooldown: {
            // After a child Success, block re-entry for exactly `param` ticks. While cooling, return Failure
            // and consume one tick of the cooldown. When ready (cooldown==0) tick the child; on its Success
            // start the cooldown.
            if (s.cooldown > 0) { --s.cooldown; return kFailure; }   // blocked (consumes one tick)
            if (n.childCount < 1) return kFailure;
            const Status r = TickBtx(tree, st, bb, n.child[0], depth + 1, run);
            if (r == kSuccess) s.cooldown = n.param;                 // start the cooldown on success
            return r;
        }
        case kBtxBlackboardCondition: {
            // Observer abort — see the header banner for the pinned tick semantics.
            const bool pred = BtxEvalOp(bb.Get(n.bbKey), n.op, n.param);
            if (pred) {
                if (n.childCount < 1) return kSuccess;
                const Status r = TickBtx(tree, st, bb, n.child[0], depth + 1, run);
                s.childRunning = (r == kRunning) ? 1u : 0u;          // latch the in-progress state
                return r;
            }
            // predicate false:
            if (s.childRunning) {                                    // a running subtree is INTERRUPTED
                s.childRunning = 0;
                ResetSubtree(tree, st, n.child[0], depth + 1);       // fresh re-entry later
                ++run.aborts;
                return kFailure;                                     // ABORT at the exact flip tick
            }
            return kFailure;                                         // gate simply closed (nothing to abort)
        }
        case kBtxRetry: {
            if (n.childCount < 1) return kFailure;
            const Status r = TickBtx(tree, st, bb, n.child[0], depth + 1, run);
            if (r == kSuccess) { s.attempts = 0; return kSuccess; }
            if (r == kRunning) return kRunning;                      // in progress (not an attempt yet)
            ++s.attempts;                                            // a Failure = one attempt used
            if (s.attempts < n.param) return kRunning;               // retry next tick
            s.attempts = 0; return kFailure;                         // attempts exhausted
        }
        case kBtxLoop: {
            if (n.childCount < 1) return kFailure;
            const Status r = TickBtx(tree, st, bb, n.child[0], depth + 1, run);
            if (r == kRunning) return kRunning;
            if (r == kFailure) { s.loopCount = 0; return kFailure; }
            ++s.loopCount;                                           // a Success = one iteration done
            if (s.loopCount >= n.param) { s.loopCount = 0; return kSuccess; }
            return kRunning;                                         // more iterations to go
        }
        case kBtxService: {
            // Run the service routine every `param` ticks while active (svcCounter counts ticks); then tick
            // the child. When the node stops being ticked, svcCounter freezes -> updates cease (deactivation).
            const int32_t interval = n.param > 0 ? n.param : 1;
            if ((s.svcCounter % interval) == 0) {
                RunServiceFn(n.serviceFn, bb);
                run.serviceRanThisTick = 1;
                ++run.serviceRuns;
            }
            ++s.svcCounter;
            if (n.childCount < 1) return kSuccess;
            return TickBtx(tree, st, bb, n.child[0], depth + 1, run);
        }
        case kBtxUtilitySelector: {
            // Score each child (authored scorer set); pick the HIGHEST (tie -> LOWEST index via strict >).
            int best = -1; int32_t bestScore = 0;
            for (int c = 0; c < n.childCount && c < kMaxChildren; ++c) {
                const int32_t sc = BtxScore(n.scorer[c], n.scorerArg[c], bb);
                if (best < 0 || sc > bestScore) { best = c; bestScore = sc; }
            }
            if (best < 0) return kFailure;                           // empty -> deterministic Failure
            run.utilityChoice = best;
            ++run.utilityChoices;
            return TickBtx(tree, st, bb, n.child[best], depth + 1, run);
        }
        default:
            return kFailure;                                         // unknown kind -> deterministic Failure
    }
}

// ===== TickBtxTree: tick the whole tree from the root (index 0), resetting the per-tick trace fields =======
// An empty tree is a deterministic kFailure. run.utilityChoice/serviceRanThisTick are reset each tick; the
// cumulative counters persist. Two TickBtxTree calls over the same tree+state+blackboard are byte-identical.
inline Status TickBtxTree(const BtxTree& tree, std::vector<BtxNodeState>& st, Blackboard& bb, BtxRun& run) {
    run.utilityChoice = -1;
    run.serviceRanThisTick = 0;
    if (tree.empty()) return kFailure;
    return TickBtx(tree, st, bb, 0, 0, run);
}

// ===== TreeNodeCount: the node count (the {nodes} proof) ==================================================
inline int BtxNodeCountOf(const BtxTree& tree) { return (int)tree.size(); }

// =========================================================================================================
// THE GUARD NPC BEHAVIOR — a tree combining every BT1 node kind, driven over a scripted world.
// =========================================================================================================
// A guard that: (SERVICE refreshes threat every 4 ticks) -> (UTILITY-selects between patrol / chase / attack
// / flee-when-low-health) -> the chosen branch uses a PARALLEL (move + aim simultaneously) and, on attack, a
// COOLDOWN decorator gating the attack pulse. The world writes raw (enemyDist, health, ammo) each tick; the
// service converts distance+ammo into the derived threat/attack-score slots the utility selector scores.
//
// Flat layout (root at index 0):
//   [0]  Service(interval=4, RefreshThreat)     -> {1}
//   [1]  UtilitySelector                        -> {2(patrol), 6(chase), 11(attack), 20(flee)}
//          scorers: [Const 15] [Direct threat] [Direct attackScore] [Inverse health]
//   [2]  Sequence patrol -> {3,4,5}
//   [3]    Action choice=PATROL   [4] Action moveX=0   [5] Action aim=0
//   [6]  Sequence chase -> {7,8}
//   [7]    Action choice=CHASE
//   [8]    Parallel(RequireAll) -> {9,10}   [9] Action moveX=1   [10] Action aim=1
//   [11] Sequence attack -> {12,13,16}
//   [12]   Action choice=ATTACK
//   [13]   Parallel(RequireAll) -> {14,15}  [14] Action moveX=1  [15] Action aim=1
//   [16]   Selector -> {17,19}   [17] Cooldown(3) -> {18} Action attackReady=1   [19] Action attackReady=0
//   [20] Sequence flee -> {21,22,23}
//   [21]   Action choice=FLEE   [22] Action moveX=-1   [23] Action aim=0
inline constexpr int32_t kBt1PatrolBase = 15;   // the patrol baseline priority (Const scorer)
inline constexpr int32_t kBt1AtkCooldown = 3;   // the attack cooldown (ticks)

inline BtxTree BuildGuardTree() {
    BtxTree t;
    t.resize(24);
    auto action = [&](int i, int key, int32_t val) {
        t[i].kind = kBtxActionLeaf; t[i].bbKey = key; t[i].param = val; t[i].childCount = 0;
    };

    // [0] Service
    t[0].kind = kBtxService; t[0].param = 4; t[0].serviceFn = kSvcRefreshThreat;
    t[0].child[0] = 1; t[0].childCount = 1;

    // [1] UtilitySelector over patrol/chase/attack/flee
    t[1].kind = kBtxUtilitySelector;
    t[1].child[0] = 2;  t[1].scorer[0] = kScoreConst;   t[1].scorerArg[0] = kBt1PatrolBase;
    t[1].child[1] = 6;  t[1].scorer[1] = kScoreDirect;  t[1].scorerArg[1] = kBbxThreat;
    t[1].child[2] = 11; t[1].scorer[2] = kScoreDirect;  t[1].scorerArg[2] = kBbxAttackScore;
    t[1].child[3] = 20; t[1].scorer[3] = kScoreInverse; t[1].scorerArg[3] = kBbxHealth;
    t[1].childCount = 4;

    // patrol
    t[2].kind = kBtxSequence; t[2].child[0] = 3; t[2].child[1] = 4; t[2].child[2] = 5; t[2].childCount = 3;
    action(3, kBbxChoice, kChoicePatrol); action(4, kBbxMoveX, 0); action(5, kBbxAim, 0);

    // chase
    t[6].kind = kBtxSequence; t[6].child[0] = 7; t[6].child[1] = 8; t[6].childCount = 2;
    action(7, kBbxChoice, kChoiceChase);
    t[8].kind = kBtxParallel; t[8].successPolicy = kRequireAll; t[8].failurePolicy = kRequireOne;
    t[8].child[0] = 9; t[8].child[1] = 10; t[8].childCount = 2;
    action(9, kBbxMoveX, 1); action(10, kBbxAim, 1);

    // attack
    t[11].kind = kBtxSequence; t[11].child[0] = 12; t[11].child[1] = 13; t[11].child[2] = 16; t[11].childCount = 3;
    action(12, kBbxChoice, kChoiceAttack);
    t[13].kind = kBtxParallel; t[13].successPolicy = kRequireAll; t[13].failurePolicy = kRequireOne;
    t[13].child[0] = 14; t[13].child[1] = 15; t[13].childCount = 2;
    action(14, kBbxMoveX, 1); action(15, kBbxAim, 1);
    // Selector { Cooldown(N) -> attackReady=1 ,  attackReady=0 } — cooldown Failure falls through to =0.
    t[16].kind = kBtxSelector; t[16].child[0] = 17; t[16].child[1] = 19; t[16].childCount = 2;
    t[17].kind = kBtxCooldown; t[17].param = kBt1AtkCooldown; t[17].child[0] = 18; t[17].childCount = 1;
    action(18, kBbxAttackReady, 1);
    action(19, kBbxAttackReady, 0);

    // flee
    t[20].kind = kBtxSequence; t[20].child[0] = 21; t[20].child[1] = 22; t[20].child[2] = 23; t[20].childCount = 3;
    action(21, kBbxChoice, kChoiceFlee); action(22, kBbxMoveX, -1); action(23, kBbxAim, 0);

    return t;
}

// ===== The guard scenario harness — per-tick trace over the scripted world ================================
struct Bt1Frame {
    int32_t tick;
    int32_t choice;        // kBbxChoice (kChoice*)
    int32_t threat;        // kBbxThreat (service-refreshed)
    int32_t attackScore;   // kBbxAttackScore
    int32_t health;
    int32_t ammo;
    int32_t enemyDist;
    int32_t attackReady;   // kBbxAttackReady (cooldown-gated)
    int32_t moveX;         // kBbxMoveX (parallel arm)
    int32_t aim;           // kBbxAim (parallel arm)
    int32_t utilityChoice; // the chosen utility child index
    int32_t serviceRan;    // 1 if the service updated this tick
    uint32_t rootStatus;   // the root tick Status
};

struct Bt1ShotRun {
    std::vector<Bt1Frame> frames;
    int nodes = 0;
    int ticks = 0;
    int utilityChoices = 0;   // cumulative
    int aborts = 0;           // cumulative (0 in the guard run; the abort demo is a separate test)
    int serviceRuns = 0;      // cumulative
    uint64_t digest = 0;
    int patrolToChaseTick = -1;
    int chaseToAttackTick = -1;
    int attackToFleeTick = -1;
};

// The scripted world (== behavior_tree_test): enemy approaches, health then drops + ammo runs out. The
// service (interval 4) refreshes threat/attackScore at ticks 0,4,8,12; the utility choice reacts on those.
inline void Bt1WorldInputs(int t, int32_t& dist, int32_t& health, int32_t& ammo) {
    dist   = (t < 4) ? 40 : (t < 8) ? 20 : 8;
    health = (t < 8) ? 50 : (t < 12) ? 45 : 10;
    ammo   = (t < 12) ? 3 : 0;
}

inline constexpr int kBt1Ticks = 16;

inline Bt1ShotRun RunBt1Scenario() {
    const BtxTree tree = BuildGuardTree();
    std::vector<BtxNodeState> st = MakeBtxState(tree);
    Blackboard bb;
    bb.Set(kBbxAttackReady, 0);
    BtxRun run;

    Bt1ShotRun out;
    out.nodes = BtxNodeCountOf(tree);
    out.ticks = kBt1Ticks;
    uint64_t dg = kBtxFnvOffset;
    int32_t prevChoice = 0;

    for (int t = 0; t < kBt1Ticks; ++t) {
        int32_t dist, health, ammo;
        Bt1WorldInputs(t, dist, health, ammo);
        bb.Set(kBbxEnemyDist, dist);
        bb.Set(kBbxHealth, health);
        bb.Set(kBbxAmmo, ammo);

        run.tick = t;
        const Status rs = TickBtxTree(tree, st, bb, run);

        Bt1Frame fr;
        fr.tick = t;
        fr.choice = bb.Get(kBbxChoice);
        fr.threat = bb.Get(kBbxThreat);
        fr.attackScore = bb.Get(kBbxAttackScore);
        fr.health = health; fr.ammo = ammo; fr.enemyDist = dist;
        fr.attackReady = bb.Get(kBbxAttackReady);
        fr.moveX = bb.Get(kBbxMoveX);
        fr.aim = bb.Get(kBbxAim);
        fr.utilityChoice = run.utilityChoice;
        fr.serviceRan = run.serviceRanThisTick;
        fr.rootStatus = (uint32_t)rs;

        if (prevChoice == kChoicePatrol && fr.choice == kChoiceChase && out.patrolToChaseTick < 0)
            out.patrolToChaseTick = t;
        if (prevChoice == kChoiceChase && fr.choice == kChoiceAttack && out.chaseToAttackTick < 0)
            out.chaseToAttackTick = t;
        if (prevChoice == kChoiceAttack && fr.choice == kChoiceFlee && out.attackToFleeTick < 0)
            out.attackToFleeTick = t;
        prevChoice = fr.choice;

        dg = BtxFnvWord(dg, (uint32_t)fr.choice);
        dg = BtxFnvWord(dg, (uint32_t)fr.threat);
        dg = BtxFnvWord(dg, (uint32_t)fr.attackScore);
        dg = BtxFnvWord(dg, (uint32_t)fr.attackReady);
        dg = BtxFnvWord(dg, (uint32_t)fr.moveX);
        dg = BtxFnvWord(dg, (uint32_t)fr.aim);
        dg = BtxFnvWord(dg, (uint32_t)fr.utilityChoice);
        dg = BtxFnvWord(dg, (uint32_t)fr.serviceRan);
        dg = BtxFnvWord(dg, fr.rootStatus);

        out.frames.push_back(fr);
    }

    out.utilityChoices = run.utilityChoices;
    out.aborts = run.aborts;
    out.serviceRuns = run.serviceRuns;
    out.digest = dg;
    return out;
}

// =========================================================================================================
// RenderBt1Shot — the strict-zero PURE-INTEGER top-down/timeline viz both backends call VERBATIM.
// =========================================================================================================
// A behavior timeline (per-tick column colored by the active utility choice) as the "colored trail", plus a
// side panel of stacked timelines: the threat + health graph, and the cooldown/service marker rows. Integer
// RGB only (no float), so both backends produce byte-identical BGRA BY CONSTRUCTION. NO shader.
inline void RenderBt1Shot(const Bt1ShotRun& run, std::vector<uint8_t>& bgra, uint32_t& outW, uint32_t& outH) {
    const int kMargin = 14, kCol = 24;
    const int nT = (int)run.frames.size();
    const int W = kMargin * 2 + nT * kCol;                 // 16 ticks -> 412
    const int trailY0 = kMargin, trailH = 84;              // the behavior "trail" band
    const int gap = 12;
    const int graphY0 = trailY0 + trailH + gap, graphH = 70;   // threat/health graph
    const int markY0 = graphY0 + graphH + gap, markH = 26;     // cooldown/service marker rows
    const int H = markY0 + markH + kMargin;
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {           // deep slate background
        bgra[p * 4 + 0] = 22; bgra[p * 4 + 1] = 18; bgra[p * 4 + 2] = 28; bgra[p * 4 + 3] = 255;
    }
    auto put = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        uint8_t* d = &bgra[((size_t)y * W + x) * 4];
        d[0] = b; d[1] = g; d[2] = r; d[3] = 255;
    };
    auto fill = [&](int x0, int y0, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y < y0 + h; ++y) for (int x = x0; x < x0 + w; ++x) put(x, y, r, g, b);
    };
    struct RGB { uint8_t r, g, b; };
    auto choiceColor = [&](int32_t c) -> RGB {
        if (c == kChoicePatrol) return RGB{40, 158, 148};   // teal
        if (c == kChoiceChase)  return RGB{230, 170, 60};   // amber
        if (c == kChoiceAttack) return RGB{230, 70, 40};    // red
        if (c == kChoiceFlee)   return RGB{160, 100, 210};  // purple
        return RGB{90, 90, 100};                            // neutral
    };

    // (1) The behavior trail: each tick a full-height column tinted by the active utility choice; a brighter
    // "agent dot" rides the column at a height set by the choice (patrol low ... attack high) so the trail
    // reads as a moving guard.
    for (int t = 0; t < nT; ++t) {
        const Bt1Frame& fr = run.frames[(size_t)t];
        const RGB c = choiceColor(fr.choice);
        const int x0 = kMargin + t * kCol;
        fill(x0, trailY0, kCol - 2, trailH, (uint8_t)(c.r / 3 + 12),
             (uint8_t)(c.g / 3 + 10), (uint8_t)(c.b / 3 + 14));   // dim column wash
        // agent dot height: patrol=1, chase=2, attack=3, flee=1 (returning) -> map to band rows.
        int lvl = (fr.choice == kChoiceAttack) ? 3 : (fr.choice == kChoiceChase) ? 2
                : (fr.choice == kChoiceFlee) ? 1 : 0;
        const int dotY = trailY0 + trailH - 10 - lvl * ((trailH - 20) / 3);
        const int dotX = x0 + (kCol - 2) / 2;
        for (int dy = -3; dy <= 3; ++dy) for (int dx = -3; dx <= 3; ++dx)
            if (dx * dx + dy * dy <= 9) put(dotX + dx, dotY + dy, c.r, c.g, c.b);
        // service tick: a white notch at the top of the column.
        if (fr.serviceRan) fill(x0, trailY0, kCol - 2, 4, 235, 235, 235);
    }

    // (2) The threat + health graph: per tick a threat bar (orange) and a health bar (green), scaled to graphH.
    for (int t = 0; t < nT; ++t) {
        const Bt1Frame& fr = run.frames[(size_t)t];
        const int x0 = kMargin + t * kCol;
        const int th = (int)(((int64_t)(fr.threat < 0 ? 0 : fr.threat) * graphH) / 60);
        const int hh = (int)(((int64_t)(fr.health < 0 ? 0 : fr.health) * graphH) / 60);
        fill(x0, graphY0 + graphH - th, (kCol - 2) / 2, th, 220, 120, 40);                 // threat
        fill(x0 + (kCol - 2) / 2, graphY0 + graphH - hh, (kCol - 2) / 2, hh, 90, 200, 110); // health
    }

    // (3) The cooldown/attackReady + service marker rows.
    for (int t = 0; t < nT; ++t) {
        const Bt1Frame& fr = run.frames[(size_t)t];
        const int x0 = kMargin + t * kCol;
        if (fr.attackReady) fill(x0, markY0, kCol - 2, markH / 2, 245, 205, 90);            // attack fired (gold)
        if (fr.serviceRan)  fill(x0, markY0 + markH / 2, kCol - 2, markH / 2, 200, 200, 235); // service (blue-white)
    }
    (void)fill;
}

}  // namespace ai
}  // namespace hf
