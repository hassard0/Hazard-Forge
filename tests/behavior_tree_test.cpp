// Slice BT1 — DETERMINISTIC BEHAVIOR-TREE DEPTH + UTILITY AI (engine/ai/behavior_tree.h, hf::ai). A NEW
// additive header composing ai.h READ-ONLY (Blackboard / Status / the fixed key space) — it adds the node
// kinds ai.h's L37 banner explicitly disclaims: parallel, stateful decorators (cooldown/retry/loop), the
// observer-abort blackboard-condition, service nodes, and a utility (scoring) selector.
//
// What this test PINS:
//   (a) PARALLEL: RequireOne/RequireAll success + failure policies; deterministic child tick order.
//   (b) DECORATORS: cooldown blocks for EXACTLY N ticks (pinned re-allow tick); the observer-abort
//       blackboard-condition interrupts a Running subtree at the EXACT flip tick (child NOT ticked at abort);
//       retry/loop counts.
//   (c) SERVICE: updates the blackboard every interval ticks EXACTLY (pinned values at pinned ticks) + stops
//       when its subtree deactivates (gated off).
//   (d) UTILITY: per-child integer scores; highest wins; tie-break by lowest index; a score change flips the
//       selection at the exact tick.
//   (e) THE NPC TRACE: the guard behavior over 16 ticks — the digest + the pinned transitions
//       (patrol->chase->attack->flee) + attackReady on the cooldown tick.
//   (f) DETERMINISM: two independent RunBt1Scenario runs -> identical trace digest.
//   (g) EMPTY/ALL-FAILURE: deterministic Failure.
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "ai/behavior_tree.h"

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

    // ================= (a) PARALLEL =================
    {
        // Parallel(RequireOne success) over { action(always Success), cond(always Failure) } -> Success.
        ai::BtxTree t; t.resize(3);
        t[0].kind = ai::kBtxParallel; t[0].successPolicy = ai::kRequireOne; t[0].failurePolicy = ai::kRequireOne;
        t[0].child[0] = 1; t[0].child[1] = 2; t[0].childCount = 2;
        t[1].kind = ai::kBtxActionLeaf; t[1].bbKey = ai::kBbxMoveX; t[1].param = 7;   // always Success
        t[2].kind = ai::kBtxCondLeaf;   t[2].bbKey = ai::kBbxAim;   t[2].op = ai::kOpGE; t[2].param = 1; // slot 0 -> Fail
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        const ai::Status r = ai::TickBtxTree(t, st, bb, run);
        check(r == ai::kSuccess, "bt1(a): parallel RequireOne with one success child -> Success");
        // deterministic child tick order: BOTH children were ticked (both side effects observable — the
        // action wrote moveX; the cond is a read, so we prove ticking by re-running with a success cond).
        check(bb.Get(ai::kBbxMoveX) == 7, "bt1(a): parallel ticked child[0] (side effect applied)");
    }
    {
        // Parallel(RequireAll success, RequireOne failure) over { action(Success), cond(Failure) } -> Failure
        // (not all succeeded; one failed).
        ai::BtxTree t; t.resize(3);
        t[0].kind = ai::kBtxParallel; t[0].successPolicy = ai::kRequireAll; t[0].failurePolicy = ai::kRequireOne;
        t[0].child[0] = 1; t[0].child[1] = 2; t[0].childCount = 2;
        t[1].kind = ai::kBtxActionLeaf; t[1].bbKey = ai::kBbxMoveX; t[1].param = 1;
        t[2].kind = ai::kBtxCondLeaf;   t[2].bbKey = ai::kBbxAim;   t[2].op = ai::kOpGE; t[2].param = 1;
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        check(ai::TickBtxTree(t, st, bb, run) == ai::kFailure,
              "bt1(a): parallel RequireAll success with a failing child -> Failure");
    }
    {
        // RequireAll success with ALL-success children -> Success.
        ai::BtxTree t; t.resize(3);
        t[0].kind = ai::kBtxParallel; t[0].successPolicy = ai::kRequireAll; t[0].failurePolicy = ai::kRequireOne;
        t[0].child[0] = 1; t[0].child[1] = 2; t[0].childCount = 2;
        t[1].kind = ai::kBtxActionLeaf; t[1].bbKey = ai::kBbxMoveX; t[1].param = 3;
        t[2].kind = ai::kBtxActionLeaf; t[2].bbKey = ai::kBbxAim;   t[2].param = 5;
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        check(ai::TickBtxTree(t, st, bb, run) == ai::kSuccess, "bt1(a): parallel RequireAll all-success -> Success");
        check(bb.Get(ai::kBbxMoveX) == 3 && bb.Get(ai::kBbxAim) == 5,
              "bt1(a): parallel ticked BOTH children in order (both side effects)");
    }

    // ================= (b) DECORATORS =================
    {
        // Cooldown(3) over an action(Success). Blocks for EXACTLY 3 ticks, re-allows at tick 4.
        ai::BtxTree t; t.resize(2);
        t[0].kind = ai::kBtxCooldown; t[0].param = 3; t[0].child[0] = 1; t[0].childCount = 1;
        t[1].kind = ai::kBtxActionLeaf; t[1].bbKey = ai::kBbxMoveX; t[1].param = 1;
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        ai::Status seq[5];
        for (int i = 0; i < 5; ++i) seq[i] = ai::TickBtxTree(t, st, bb, run);
        check(seq[0] == ai::kSuccess, "bt1(b): cooldown tick0 -> Success (fires)");
        check(seq[1] == ai::kFailure && seq[2] == ai::kFailure && seq[3] == ai::kFailure,
              "bt1(b): cooldown blocks EXACTLY ticks 1,2,3 (N=3)");
        check(seq[4] == ai::kSuccess, "bt1(b): cooldown re-allows at tick 4 (N+1)");
    }
    {
        // Observer abort: BlackboardCondition(gate>=1) over a RunningLeaf(10). Flip gate=0 at tick 3 -> the
        // decorator returns Failure at tick 3 (the abort), and the child is NOT ticked (runLeft reset to -1).
        ai::BtxTree t; t.resize(2);
        t[0].kind = ai::kBtxBlackboardCondition; t[0].bbKey = ai::kBbxGate; t[0].op = ai::kOpGE; t[0].param = 1;
        t[0].child[0] = 1; t[0].childCount = 1;
        t[1].kind = ai::kBtxRunningLeaf; t[1].param = 10;
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        bb.Set(ai::kBbxGate, 1);
        ai::Status seq[4];
        for (int tk = 0; tk < 4; ++tk) {
            if (tk == 3) bb.Set(ai::kBbxGate, 0);   // FLIP the gate right before tick 3
            seq[tk] = ai::TickBtxTree(t, st, bb, run);
        }
        check(seq[0] == ai::kRunning && seq[1] == ai::kRunning && seq[2] == ai::kRunning,
              "bt1(b): observer-abort — child Running while the condition holds (ticks 0,1,2)");
        check(seq[3] == ai::kFailure, "bt1(b): observer-abort — Failure at the EXACT flip tick (3)");
        check(run.aborts == 1, "bt1(b): observer-abort counted exactly one abort");
        check(st[1].runLeft == -1,
              "bt1(b): observer-abort RESET the child (not ticked at the abort tick — runLeft cleared)");
        // Re-enabling the gate re-enters the child FRESH (runLeft back to the full 10 -> 9 after one tick).
        bb.Set(ai::kBbxGate, 1);
        const ai::Status re = ai::TickBtxTree(t, st, bb, run);
        check(re == ai::kRunning && st[1].runLeft == 9,
              "bt1(b): observer-abort — a re-entry after abort starts the child fresh");
    }
    {
        // Retry(3) over a cond that always Fails. Two attempts return Running, the 3rd exhausts -> Failure.
        ai::BtxTree t; t.resize(2);
        t[0].kind = ai::kBtxRetry; t[0].param = 3; t[0].child[0] = 1; t[0].childCount = 1;
        t[1].kind = ai::kBtxCondLeaf; t[1].bbKey = ai::kBbxGate; t[1].op = ai::kOpGE; t[1].param = 1; // slot0 -> Fail
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        ai::Status seq[3];
        for (int i = 0; i < 3; ++i) seq[i] = ai::TickBtxTree(t, st, bb, run);
        check(seq[0] == ai::kRunning && seq[1] == ai::kRunning,
              "bt1(b): retry — attempts 1,2 return Running");
        check(seq[2] == ai::kFailure, "bt1(b): retry — exhausts at attempt 3 -> Failure");
    }
    {
        // Loop(3) over an action(Success). Two Successes return Running, the 3rd completes the loop -> Success.
        ai::BtxTree t; t.resize(2);
        t[0].kind = ai::kBtxLoop; t[0].param = 3; t[0].child[0] = 1; t[0].childCount = 1;
        t[1].kind = ai::kBtxActionLeaf; t[1].bbKey = ai::kBbxMoveX; t[1].param = 1;
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        ai::Status seq[3];
        for (int i = 0; i < 3; ++i) seq[i] = ai::TickBtxTree(t, st, bb, run);
        check(seq[0] == ai::kRunning && seq[1] == ai::kRunning,
              "bt1(b): loop — iterations 1,2 return Running");
        check(seq[2] == ai::kSuccess, "bt1(b): loop — completes at count 3 -> Success");
    }

    // ================= (c) SERVICE =================
    {
        // BlackboardCondition(active>=1) over Service(interval=2, RefreshThreat) over action(Success). The
        // service refreshes threat every 2 ticks WHILE active; gating it off freezes the update.
        ai::BtxTree t; t.resize(3);
        t[0].kind = ai::kBtxBlackboardCondition; t[0].bbKey = ai::kBbxGate; t[0].op = ai::kOpGE; t[0].param = 1;
        t[0].child[0] = 1; t[0].childCount = 1;
        t[1].kind = ai::kBtxService; t[1].param = 2; t[1].serviceFn = ai::kSvcRefreshThreat;
        t[1].child[0] = 2; t[1].childCount = 1;
        t[2].kind = ai::kBtxActionLeaf; t[2].bbKey = ai::kBbxMoveX; t[2].param = 1;
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        bb.Set(ai::kBbxGate, 1);
        // t0: dist 40 -> threat 10 (service runs, counter 0%2==0).
        bb.Set(ai::kBbxEnemyDist, 40); ai::TickBtxTree(t, st, bb, run);
        check(bb.Get(ai::kBbxThreat) == 10, "bt1(c): service updated threat at tick 0 (dist40 -> 10)");
        // t1: dist 20, but service SKIPS (counter 1%2==1) -> threat stays 10.
        bb.Set(ai::kBbxEnemyDist, 20); ai::TickBtxTree(t, st, bb, run);
        check(bb.Get(ai::kBbxThreat) == 10, "bt1(c): service SKIPS the off-interval tick 1 (threat stays 10)");
        // t2: dist still 20, service runs (counter 2%2==0) -> threat 30.
        ai::TickBtxTree(t, st, bb, run);
        check(bb.Get(ai::kBbxThreat) == 30, "bt1(c): service updated threat at tick 2 (dist20 -> 30)");
        const int svcAtActive = run.serviceRuns;
        check(svcAtActive == 2, "bt1(c): exactly 2 service runs while active (ticks 0,2)");
        // t3: GATE OFF -> the service subtree deactivates. dist changes to 8 but threat is FROZEN at 30.
        bb.Set(ai::kBbxGate, 0); bb.Set(ai::kBbxEnemyDist, 8);
        const ai::Status s3 = ai::TickBtxTree(t, st, bb, run);
        check(s3 == ai::kFailure, "bt1(c): gated-off service subtree -> the condition returns Failure");
        check(bb.Get(ai::kBbxThreat) == 30, "bt1(c): service STOPS when the subtree deactivates (threat frozen)");
        check(run.serviceRuns == svcAtActive, "bt1(c): no service run while deactivated");
    }

    // ================= (d) UTILITY =================
    {
        // UtilitySelector over 3 actions writing distinct choice values; Const scorers {10,30,20} -> idx 1.
        ai::BtxTree t; t.resize(4);
        t[0].kind = ai::kBtxUtilitySelector;
        t[0].child[0] = 1; t[0].scorer[0] = ai::kScoreConst; t[0].scorerArg[0] = 10;
        t[0].child[1] = 2; t[0].scorer[1] = ai::kScoreConst; t[0].scorerArg[1] = 30;
        t[0].child[2] = 3; t[0].scorer[2] = ai::kScoreConst; t[0].scorerArg[2] = 20;
        t[0].childCount = 3;
        t[1].kind = ai::kBtxActionLeaf; t[1].bbKey = ai::kBbxChoice; t[1].param = 100;
        t[2].kind = ai::kBtxActionLeaf; t[2].bbKey = ai::kBbxChoice; t[2].param = 200;
        t[3].kind = ai::kBtxActionLeaf; t[3].bbKey = ai::kBbxChoice; t[3].param = 300;
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        ai::TickBtxTree(t, st, bb, run);
        check(run.utilityChoice == 1, "bt1(d): utility selector picks the HIGHEST scorer (idx 1, score 30)");
        check(bb.Get(ai::kBbxChoice) == 200, "bt1(d): utility ticked the winning child (its action ran)");
    }
    {
        // Tie-break by LOWEST index: Const {30,30,10} -> idx 0 wins the tie at 30.
        ai::BtxTree t; t.resize(4);
        t[0].kind = ai::kBtxUtilitySelector;
        t[0].child[0] = 1; t[0].scorer[0] = ai::kScoreConst; t[0].scorerArg[0] = 30;
        t[0].child[1] = 2; t[0].scorer[1] = ai::kScoreConst; t[0].scorerArg[1] = 30;
        t[0].child[2] = 3; t[0].scorer[2] = ai::kScoreConst; t[0].scorerArg[2] = 10;
        t[0].childCount = 3;
        for (int i = 1; i <= 3; ++i) { t[i].kind = ai::kBtxActionLeaf; t[i].bbKey = ai::kBbxChoice; t[i].param = i; }
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        ai::TickBtxTree(t, st, bb, run);
        check(run.utilityChoice == 0, "bt1(d): utility tie-break -> LOWEST index (idx 0)");
        check(bb.Get(ai::kBbxChoice) == 1, "bt1(d): tie-break ran child[0]");
    }
    {
        // A score CHANGE flips the selection at the EXACT tick: Direct(keyA) vs Direct(keyB).
        ai::BtxTree t; t.resize(3);
        t[0].kind = ai::kBtxUtilitySelector;
        t[0].child[0] = 1; t[0].scorer[0] = ai::kScoreDirect; t[0].scorerArg[0] = ai::kBbxHealth; // "A"
        t[0].child[1] = 2; t[0].scorer[1] = ai::kScoreDirect; t[0].scorerArg[1] = ai::kBbxAmmo;   // "B"
        t[0].childCount = 2;
        t[1].kind = ai::kBtxActionLeaf; t[1].bbKey = ai::kBbxChoice; t[1].param = 11;
        t[2].kind = ai::kBtxActionLeaf; t[2].bbKey = ai::kBbxChoice; t[2].param = 22;
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        bb.Set(ai::kBbxHealth, 50); bb.Set(ai::kBbxAmmo, 10);
        ai::TickBtxTree(t, st, bb, run);
        check(run.utilityChoice == 0, "bt1(d): score-flip — A wins initially (50 > 10)");
        bb.Set(ai::kBbxAmmo, 90);   // B now outscores A
        ai::TickBtxTree(t, st, bb, run);
        check(run.utilityChoice == 1, "bt1(d): score-flip — B wins after the change (90 > 50), at the exact tick");
    }

    // ================= (g) EMPTY / ALL-FAILURE (deterministic) =================
    {
        ai::BtxTree empty;
        auto st = ai::MakeBtxState(empty);
        ai::Blackboard bb; ai::BtxRun run;
        check(ai::TickBtxTree(empty, st, bb, run) == ai::kFailure, "bt1(g): empty tree -> deterministic Failure");
    }
    {
        // Selector over two always-Failure conds -> Failure, byte-identical over two runs.
        ai::BtxTree t; t.resize(3);
        t[0].kind = ai::kBtxSelector; t[0].child[0] = 1; t[0].child[1] = 2; t[0].childCount = 2;
        t[1].kind = ai::kBtxCondLeaf; t[1].bbKey = ai::kBbxGate; t[1].op = ai::kOpGE; t[1].param = 1;
        t[2].kind = ai::kBtxCondLeaf; t[2].bbKey = ai::kBbxGate; t[2].op = ai::kOpGE; t[2].param = 1;
        auto st = ai::MakeBtxState(t);
        ai::Blackboard bb; ai::BtxRun run;
        check(ai::TickBtxTree(t, st, bb, run) == ai::kFailure, "bt1(g): all-failure selector -> Failure (run 1)");
        auto st2 = ai::MakeBtxState(t); ai::Blackboard bb2; ai::BtxRun run2;
        check(ai::TickBtxTree(t, st2, bb2, run2) == ai::kFailure, "bt1(g): all-failure selector -> Failure (run 2)");
    }

    // ================= (e) THE NPC TRACE =================
    {
        const ai::Bt1ShotRun r = ai::RunBt1Scenario();
        check(r.nodes == 24, "bt1(e): guard tree has 24 nodes");
        check(r.ticks == ai::kBt1Ticks, "bt1(e): guard scenario ran the scripted ticks");
        check((int)r.frames.size() == ai::kBt1Ticks, "bt1(e): one trace frame per tick");

        // The pinned choice sequence: PATROL[0,4) -> CHASE[4,8) -> ATTACK[8,12) -> FLEE[12,16).
        bool story = true;
        for (int t = 0; t < 4;  ++t) if (r.frames[(size_t)t].choice != ai::kChoicePatrol) story = false;
        for (int t = 4; t < 8;  ++t) if (r.frames[(size_t)t].choice != ai::kChoiceChase)  story = false;
        for (int t = 8; t < 12; ++t) if (r.frames[(size_t)t].choice != ai::kChoiceAttack) story = false;
        for (int t = 12; t < 16; ++t) if (r.frames[(size_t)t].choice != ai::kChoiceFlee)  story = false;
        check(story, "bt1(e): the guard behavior story — PATROL -> CHASE -> ATTACK -> FLEE at the pinned bands");

        check(r.patrolToChaseTick == 4, "bt1(e): PATROL->CHASE at tick 4 (threat crosses patrol baseline)");
        check(r.chaseToAttackTick == 8, "bt1(e): CHASE->ATTACK at tick 8 (enemy enters attack range)");
        check(r.attackToFleeTick == 12, "bt1(e): ATTACK->FLEE at tick 12 (health low + ammo out)");

        // attackReady (the cooldown-gated attack pulse) fires ONLY at tick 8, then cools down.
        check(r.frames[8].attackReady == 1, "bt1(e): attack fires (attackReady=1) on entering ATTACK at tick 8");
        check(r.frames[9].attackReady == 0 && r.frames[10].attackReady == 0 && r.frames[11].attackReady == 0,
              "bt1(e): the attack COOLDOWN blocks re-fire at ticks 9,10,11");

        check(r.serviceRuns == 4, "bt1(e): the threat service ran exactly 4 times (ticks 0,4,8,12)");
        check(r.utilityChoices == 16, "bt1(e): the utility selector evaluated once per tick (16)");
        check(r.aborts == 0, "bt1(e): no observer-aborts in the guard run");
        std::printf("bt1: guard trace {nodes:%d, ticks:%d, utilityChoices:%d, serviceRuns:%d, aborts:%d, "
                    "transitions:%d/%d/%d, digest:0x%016llx}\n",
                    r.nodes, r.ticks, r.utilityChoices, r.serviceRuns, r.aborts,
                    r.patrolToChaseTick, r.chaseToAttackTick, r.attackToFleeTick,
                    (unsigned long long)r.digest);
    }

    // ================= (f) DETERMINISM — two independent runs -> identical trace digest =================
    {
        const ai::Bt1ShotRun a = ai::RunBt1Scenario();
        const ai::Bt1ShotRun b = ai::RunBt1Scenario();
        check(a.digest == b.digest, "bt1(f): two independent guard runs -> identical trace digest (bit-exact)");
        bool eq = a.frames.size() == b.frames.size();
        for (size_t i = 0; i < a.frames.size() && eq; ++i)
            if (a.frames[i].choice != b.frames[i].choice ||
                a.frames[i].attackReady != b.frames[i].attackReady ||
                a.frames[i].threat != b.frames[i].threat ||
                a.frames[i].utilityChoice != b.frames[i].utilityChoice) eq = false;
        check(eq, "bt1(f): two runs -> byte-identical per-tick trace");
    }

    // The shared shot raster is a pure function of the run (RenderBt1Shot) — two calls byte-equal.
    {
        const ai::Bt1ShotRun r = ai::RunBt1Scenario();
        std::vector<uint8_t> i1, i2; uint32_t w1 = 0, h1 = 0, w2 = 0, h2 = 0;
        ai::RenderBt1Shot(r, i1, w1, h1);
        ai::RenderBt1Shot(r, i2, w2, h2);
        check(w1 == w2 && h1 == h2 && i1.size() == i2.size() && i1 == i2,
              "bt1: RenderBt1Shot is a pure function (two calls byte-identical)");
        check(w1 > 0 && h1 > 0 && i1.size() == (size_t)w1 * h1 * 4, "bt1: shot raster is a well-formed BGRA image");
    }

    if (g_fail == 0) std::printf("behavior_tree_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
