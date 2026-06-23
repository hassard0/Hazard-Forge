// Slice AI1 — Deterministic AI: THE BLACKBOARD + DECISION-TREE NODE GRAPH + DETERMINISTIC TICK (the
// beachhead of the DETERMINISTIC AI flagship, GitHub issue #28, hf::ai). ai.h is a NEW additive sibling
// that #includes game/verdict.h + sim/fpx.h READ-ONLY.
//
// What this test PINS (the determinism contract + the node semantics):
//   * TickTree TWO-RUN determinism: the same tree + blackboard yield the same Status AND the same
//     DigestBlackboard on two independent runs (fixed child order + in-place integer mutation).
//   * SELECTOR returns the first non-Failure child AND short-circuits — a later child's action side effect
//     is NOT applied once an earlier child succeeds (provable via the blackboard).
//   * SEQUENCE returns the first non-Success child (a Failure short-circuits; the trailing action is NOT
//     reached).
//   * INVERTER flips Success<->Failure.
//   * A condition leaf READS the blackboard; an action leaf WRITES it.
//   * The canonical BuildAi1Tree scenario produces the EXPECTED status + digest, and flipping the single
//     kBbEnemyClose input deterministically changes which branch the selector/sequence takes (CHASE vs
//     PATROL).
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "ai/ai.h"

#include <cstdint>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace ai = hf::ai;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- (1) Blackboard get/set round-trips + out-of-range guards (deterministic, never UB) ----------
    {
        ai::Blackboard bb;
        check(bb.Get(0) == 0 && bb.Get(ai::kMaxBbKeys - 1) == 0, "ai1: blackboard zero-initialized");
        bb.Set(3, 1234);
        check(bb.Get(3) == 1234, "ai1: blackboard set/get round-trips");
        bb.Set(-1, 999); bb.Set(ai::kMaxBbKeys, 999);            // out-of-range -> no-op
        check(bb.Get(-1) == 0 && bb.Get(ai::kMaxBbKeys) == 0, "ai1: out-of-range read is 0");
        check(bb.Get(3) == 1234, "ai1: out-of-range writes did not corrupt a valid slot");
    }

    // ---- (2) condLeaf READS the blackboard; actionLeaf WRITES it -------------------------------------
    {
        // A two-node tree: [0] cond(slot5 >= 10), used as a pure read.
        ai::DecisionTree t(1);
        t[0].kind = ai::kCondLeaf; t[0].bbKey = 5; t[0].param = 10; t[0].childCount = 0;
        ai::Blackboard bb;
        check(ai::TickTree(t, bb) == ai::kFailure, "ai1: condLeaf below threshold -> Failure");
        bb.Set(5, 10);
        check(ai::TickTree(t, bb) == ai::kSuccess, "ai1: condLeaf at threshold -> Success");
        const uint64_t digestBefore = ai::DigestBlackboard(bb);
        ai::TickTree(t, bb);   // a condLeaf must NOT mutate the blackboard
        check(ai::DigestBlackboard(bb) == digestBefore, "ai1: condLeaf does not mutate the blackboard");
    }
    {
        // An action tree: [0] action(slot7 = 42) -> Success + the write happened.
        ai::DecisionTree t(1);
        t[0].kind = ai::kActionLeaf; t[0].bbKey = 7; t[0].param = 42; t[0].childCount = 0;
        ai::Blackboard bb;
        check(ai::TickTree(t, bb) == ai::kSuccess, "ai1: actionLeaf returns Success");
        check(bb.Get(7) == 42, "ai1: actionLeaf wrote param into the blackboard");
    }

    // ---- (3) INVERTER flips Success<->Failure -------------------------------------------------------
    {
        // [0] inverter -> [1] cond(slot0 >= 1)
        ai::DecisionTree t(2);
        t[0].kind = ai::kInverter; t[0].child[0] = 1; t[0].childCount = 1;
        t[1].kind = ai::kCondLeaf; t[1].bbKey = 0; t[1].param = 1; t[1].childCount = 0;
        ai::Blackboard bb;                                  // slot0 = 0 -> cond Failure -> inverter Success
        check(ai::TickTree(t, bb) == ai::kSuccess, "ai1: inverter flips Failure -> Success");
        bb.Set(0, 1);                                       // cond Success -> inverter Failure
        check(ai::TickTree(t, bb) == ai::kFailure, "ai1: inverter flips Success -> Failure");
    }

    // ---- (4) SELECTOR returns the first non-Failure child AND short-circuits (no later side effect) ---
    {
        // [0] selector { [1] action(slotA = 100), [2] action(slotB = 200) }
        // The selector must run child[0] (Success) and NEVER reach child[1] -> slotB stays 0.
        ai::DecisionTree t(3);
        t[0].kind = ai::kSelector; t[0].child[0] = 1; t[0].child[1] = 2; t[0].childCount = 2;
        t[1].kind = ai::kActionLeaf; t[1].bbKey = 10; t[1].param = 100; t[1].childCount = 0;
        t[2].kind = ai::kActionLeaf; t[2].bbKey = 11; t[2].param = 200; t[2].childCount = 0;
        ai::Blackboard bb;
        check(ai::TickTree(t, bb) == ai::kSuccess, "ai1: selector returns child[0] Success");
        check(bb.Get(10) == 100, "ai1: selector ran child[0] (side effect applied)");
        check(bb.Get(11) == 0,
              "ai1: selector SHORT-CIRCUITS — child[1] side effect NOT applied");
    }
    {
        // Selector whose child[0] FAILS -> it must fall through to child[1].
        // [0] selector { [1] sequence(cond Fails), [2] action(slotB = 200) }
        ai::DecisionTree t(4);
        t[0].kind = ai::kSelector; t[0].child[0] = 1; t[0].child[1] = 2; t[0].childCount = 2;
        t[1].kind = ai::kSequence; t[1].child[0] = 3; t[1].childCount = 1;
        t[2].kind = ai::kActionLeaf; t[2].bbKey = 11; t[2].param = 200; t[2].childCount = 0;
        t[3].kind = ai::kCondLeaf; t[3].bbKey = 0; t[3].param = 1; t[3].childCount = 0;  // slot0=0 -> Fail
        ai::Blackboard bb;
        check(ai::TickTree(t, bb) == ai::kSuccess, "ai1: selector falls through to child[1] on child[0] Failure");
        check(bb.Get(11) == 200, "ai1: selector ran the fallback child[1] action");
    }

    // ---- (5) SEQUENCE returns the first non-Success child (Failure short-circuits the trailing action) -
    {
        // [0] sequence { [1] cond(slot0 >= 1), [2] action(slotC = 300) }
        // slot0 = 0 -> cond Fails -> sequence Fails at child[0]; the action is NOT reached -> slotC stays 0.
        ai::DecisionTree t(3);
        t[0].kind = ai::kSequence; t[0].child[0] = 1; t[0].child[1] = 2; t[0].childCount = 2;
        t[1].kind = ai::kCondLeaf; t[1].bbKey = 0; t[1].param = 1; t[1].childCount = 0;
        t[2].kind = ai::kActionLeaf; t[2].bbKey = 12; t[2].param = 300; t[2].childCount = 0;
        ai::Blackboard bb;
        check(ai::TickTree(t, bb) == ai::kFailure, "ai1: sequence Fails at the first failing child");
        check(bb.Get(12) == 0, "ai1: sequence SHORT-CIRCUITS — trailing action NOT reached");
        bb.Set(0, 1);                                       // now cond Succeeds -> sequence runs the action
        check(ai::TickTree(t, bb) == ai::kSuccess, "ai1: sequence Succeeds when every child Succeeds");
        check(bb.Get(12) == 300, "ai1: sequence ran the trailing action once unblocked");
    }

    // ---- (6) The canonical BuildAi1Tree scenario: expected status + digest, branch flips with input ---
    {
        const ai::DecisionTree tree = ai::BuildAi1Tree();
        check((int)tree.size() == 5, "ai1: canonical tree has 5 nodes");
        check(ai::TreeDepth(tree) == 3, "ai1: canonical tree depth is 3");

        // Scenario A — enemy NOT close: the sequence Fails at the cond, the selector falls through to the
        // PATROL fallback (CHASE is never written).
        {
            ai::Blackboard bb;                              // kBbEnemyClose = 0 (< threshold)
            const ai::Status s = ai::TickTree(tree, bb);
            check(s == ai::kSuccess, "ai1: canonical (enemy far) -> selector Success (PATROL fallback)");
            check(bb.Get(ai::kBbState) == ai::kStatePatrol, "ai1: canonical (enemy far) -> state PATROL");
        }

        // Scenario B — enemy close: the sequence runs (cond Success -> CHASE action), the selector
        // short-circuits at child 0 (PATROL never written).
        {
            ai::Blackboard bb;
            bb.Set(ai::kBbEnemyClose, ai::kEnemyThreshold);  // enemy is close (>= threshold)
            const ai::Status s = ai::TickTree(tree, bb);
            check(s == ai::kSuccess, "ai1: canonical (enemy close) -> selector Success (CHASE branch)");
            check(bb.Get(ai::kBbState) == ai::kStateChase, "ai1: canonical (enemy close) -> state CHASE");
        }

        // The branch is a deterministic pure function of the input — the two scenarios pick DIFFERENT
        // states, so the digests differ.
        {
            ai::Blackboard farBb; ai::TickTree(tree, farBb);
            ai::Blackboard closeBb; closeBb.Set(ai::kBbEnemyClose, ai::kEnemyThreshold); ai::TickTree(tree, closeBb);
            check(ai::DigestBlackboard(farBb) != ai::DigestBlackboard(closeBb),
                  "ai1: flipping the input deterministically changes the selected branch (distinct digests)");
        }
    }

    // ---- (7) TWO-RUN determinism of the canonical script (status + DigestBlackboard byte-identical) ---
    {
        const ai::DecisionTree tree = ai::BuildAi1Tree();
        // A fixed script: a few blackboard inputs, ticking after each (the showcase script shape).
        auto runScript = [&](ai::Status& outStatus) {
            ai::Blackboard bb;
            ai::Status s = ai::kFailure;
            s = ai::TickTree(tree, bb);                                   // enemy far  -> PATROL
            bb.Set(ai::kBbEnemyClose, ai::kEnemyThreshold);
            s = ai::TickTree(tree, bb);                                   // enemy close -> CHASE
            bb.Set(ai::kBbEnemyClose, 0);
            s = ai::TickTree(tree, bb);                                   // enemy far again -> PATROL
            outStatus = s;
            return ai::DigestBlackboard(bb);
        };
        ai::Status s1, s2;
        const uint64_t d1 = runScript(s1);
        const uint64_t d2 = runScript(s2);
        check(s1 == s2, "ai1: two-run script -> identical final Status");
        check(d1 == d2, "ai1: two-run script -> identical DigestBlackboard (byte-identical)");
    }

    if (g_fail == 0) std::printf("ai_tree_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
