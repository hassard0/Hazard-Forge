// Slice FK1 — WHAT-IF FORK REPLAY (counterfactual timelines), hf::replay::fork.
//
// Because the whole HF sim is bit-exact deterministic (VD1-VD6 whole-world lockstep from an input stream
// ALONE) AND seek/rollback-capable (replay.h RP1-6), you can Seek a recorded replay to ANY tick, MUTATE one
// input, and RE-SIMULATE a COUNTERFACTUAL timeline that is itself perfectly reproducible — a "what-if"
// replay. fork.h composes replay.h (RP4 Seek) + session.h (NS6 CatchUp) + verdict.h (VD1-VD6 world) READ-
// ONLY; it adds NO field, edits NO frozen function. UE5's float sim has no reproducible re-derivation and no
// inverse, so a bit-exact counterfactual is STRUCTURALLY impossible.
//
// What this test PINS (the spec's proofs):
//   (a) NULL FORK      — a fork with NO mutation == the original bit-exact (full-timeline digest equal).
//   (b) CAUSAL DIVERGE — mutate at T -> ticks [0,T) identical + ticks >= T diverge; first-divergence == T,
//                        even when forkTick < T (divergence is the MUTATION tick, not the fork tick).
//   (c) REPRODUCIBLE   — two independent re-sims of the SAME fork -> identical timeline digest (the moat).
//   (d) FORK@0         — fork at tick 0 + mutate == a fresh sim from the mutated initial (pinned equal).
//   (e) TIMELINE TREE  — original + 2 counterfactuals -> 3 distinct digests; shared prefixes byte-identical.
//   (f) OUTCOME CHANGE — a mutation flips the final outcome: original-final != fork-final (pinned).
//   (+) SEEK REWIND    — ForkAt's RP4 Seek reconstructs the EXACT forkTick state (== original digest there).
//   (+) VIZ            — the timeline-tree report image is two-run byte-identical + the stats pin.
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests. Cross-compiler proof: the printed digests are
// IDENTICAL on MSVC + local clang (the same pure-integer computation).
#include "replay/fork.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace fork    = hf::replay;
namespace verdict = hf::game::verdict;
using verdict::fx;
using verdict::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A small helper: build a fresh command carrying its own tick.
static verdict::Command Cmd(uint32_t tick, uint32_t kind, verdict::EntityId target, int ix, int iy, int iz) {
    verdict::Command c;
    c.tick = tick; c.kind = kind; c.target = target;
    c.arg = verdict::FxVec3{ (fx)((int64_t)ix * (int64_t)kOne), (fx)((int64_t)iy * (int64_t)kOne),
                             (fx)((int64_t)iz * (int64_t)kOne) };
    return c;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- Build the canonical fork scenario (the shared HEADLINE tree: original + branch A + branch B). ----
    verdict::VerdictWorld world0;
    const fork::Fk1Scenario sc = fork::BuildFk1Scenario(world0);
    check(sc.ticks > fork::kFk1ForkTick, "scenario: ticks span past the fork tick");
    check(sc.original.digests.size() == sc.ticks, "scenario: original timeline has one digest per tick");
    check(sc.resimA.digests.size() == sc.ticks && sc.resimB.digests.size() == sc.ticks,
          "scenario: both counterfactual timelines cover every tick");

    // ---- (+) SEEK REWIND — ForkAt's RP4 Seek reconstructs the EXACT world state AT forkTick. -------------
    {
        const fork::VerdictDigest dig{};
        // The world AS OF forkTick (before stepping forkTick) == the original digest AFTER tick forkTick-1.
        const uint64_t baseDig = dig(sc.forkA.baseWorld);
        check(baseDig == sc.original.digests[(std::size_t)fork::kFk1ForkTick - 1],
              "seek: ForkAt reconstructs the EXACT forkTick state (== original digest there)");
        check(sc.forkB.baseWorld.order.size() == sc.forkA.baseWorld.order.size(),
              "seek: both branches rewound to the SAME base state");
        std::printf("fk1: fork@%u seek {keyframeTick:%u, replayedTicks:%u}\n",
                    sc.forkA.forkTick, sc.forkA.seekKeyframeTick, sc.forkA.seekReplayedTicks);
    }

    // ===== (a) NULL FORK — a fork with NO mutation == the original bit-exact. =========================
    {
        fork::ForkedTimeline nullFork =
            fork::ForkAt(sc.demo, sc.original, sc.params, fork::kFk1ForkTick, sc.originalStream);
        check(!nullFork.mutated, "null: an unmutated fork reports mutated == false");
        const fork::Timeline nullTl = fork::ResimulateFork(nullFork);
        check(nullTl.fullDigest == sc.original.fullDigest,
              "null: an unmutated fork's full-timeline digest == the original (bit-exact)");
        const fork::TimelineDiff d = fork::DiffTimelines(sc.original, nullTl);
        check(d.identical && d.firstDivergence == -1,
              "null: the null fork is fully identical to the original (no divergence)");
        // The final world also matches bit-for-bit (the outcome is unchanged).
        check(fork::OutcomesEqual(fork::OutcomeOf(sc.original, sc.player),
                                  fork::OutcomeOf(nullTl, sc.player)),
              "null: the null fork's final outcome == the original");
    }

    // ===== (b) CAUSAL DIVERGENCE — mutate at T (with forkTick < T) -> [0,T) identical, >=T diverges. ===
    {
        const uint32_t kForkT = 6u;    // rewind BEFORE the mutation (mid-keyframe-interval -> Seek replays a tail)
        const uint32_t kMutT  = 10u;   // inject the counterfactual LATER
        check(kForkT < kMutT && kMutT < sc.ticks, "causal: fork tick < mutation tick < ticks");
        fork::ForkedTimeline fk =
            fork::ForkAt(sc.demo, sc.original, sc.params, kForkT, sc.originalStream);
        const bool applied = fork::MutateInput(fk, Cmd(kMutT, verdict::kCmdImpulse, sc.player, 4, 0, 0));
        check(applied && fk.firstMutationTick == kMutT, "causal: the mutation applies at tick T (>= forkTick)");
        // A mutation BEFORE the fork tick is REJECTED (cannot rewrite shared history — the causal guard).
        check(!fork::MutateInput(fk, Cmd(kForkT - 1u, verdict::kCmdImpulse, sc.player, 9, 0, 0)),
              "causal: a mutation before the fork tick is rejected (shared history is immutable)");

        const fork::Timeline tl = fork::ResimulateFork(fk);
        const fork::TimelineDiff d = fork::DiffTimelines(sc.original, tl);
        check(d.firstDivergence == (int)kMutT,
              "causal: the FIRST divergence is EXACTLY the mutation tick T (not the fork tick)");
        // Directly assert causality: every tick < T is byte-identical to the original.
        bool beforeMatch = true;
        for (uint32_t t = 0; t < kMutT; ++t)
            if (sc.original.digests[t] != tl.digests[t]) beforeMatch = false;
        check(beforeMatch, "causal: every tick < T matches the original (the shared prefix is intact)");
        // And at least one tick >= T diverges (the counterfactual actually changed the future).
        bool afterDiverges = false;
        for (uint32_t t = kMutT; t < sc.ticks; ++t)
            if (sc.original.digests[t] != tl.digests[t]) afterDiverges = true;
        check(afterDiverges, "causal: at least one tick >= T diverges (the counterfactual took effect)");
        check(tl.fullDigest != sc.original.fullDigest, "causal: the counterfactual full digest != the original");
    }

    // ===== (c) REPRODUCIBLE COUNTERFACTUAL — two independent re-sims of the SAME fork -> identical. ====
    {
        fork::ForkedTimeline fk =
            fork::ForkAt(sc.demo, sc.original, sc.params, fork::kFk1ForkTick, sc.originalStream);
        fork::MutateInput(fk, Cmd(fork::kFk1ForkTick, verdict::kCmdImpulse, sc.player, 3, 0, 0));
        const fork::Timeline a = fork::ResimulateFork(fk);
        const fork::Timeline b = fork::ResimulateFork(fk);   // re-sim the SAME fork again
        check(a.fullDigest == b.fullDigest, "reproducible: two re-sims of the same fork -> identical full digest");
        bool perTick = (a.digests.size() == b.digests.size());
        for (std::size_t i = 0; perTick && i < a.digests.size(); ++i) if (a.digests[i] != b.digests[i]) perTick = false;
        check(perTick, "reproducible: the two re-sims are per-tick byte-identical");
        check(fork::OutcomesEqual(fork::OutcomeOf(a, sc.player), fork::OutcomeOf(b, sc.player)),
              "reproducible: the two re-sims reach the identical final outcome");
        // A third party re-building the scenario from scratch re-derives the identical counterfactual.
        verdict::VerdictWorld world0b;
        const fork::Fk1Scenario sc2 = fork::BuildFk1Scenario(world0b);
        check(sc2.resimA.fullDigest == sc.resimA.fullDigest && sc2.resimB.fullDigest == sc.resimB.fullDigest,
              "reproducible: a from-scratch rebuild re-derives the identical branch digests (the moat)");
    }

    // ===== (d) FORK@0 — fork at tick 0 + mutate == a fresh sim from the mutated initial. ==============
    {
        fork::ForkedTimeline fk0 =
            fork::ForkAt(sc.demo, sc.original, sc.params, 0u, sc.originalStream);
        check(fk0.forkTick == 0u && fk0.prefixDigests.empty(), "fork@0: no shared prefix at tick 0");
        // The Seek@0 base IS the initial world (bit-for-bit).
        check(fork::VerdictDigest{}(fk0.baseWorld) == fork::VerdictDigest{}(sc.w0Snap),
              "fork@0: the Seek@0 base == the initial world");
        const verdict::Command spawn = Cmd(0u, verdict::kCmdSpawn, verdict::kNoEntity, -2, 2, 0);
        fork::MutateInput(fk0, spawn);
        const fork::Timeline forked = fork::ResimulateFork(fk0);
        // A fresh sim from tick 0 over the SAME mutated stream (original + the injected spawn).
        std::vector<verdict::Command> mutatedStream = sc.originalStream;
        mutatedStream.push_back(spawn);
        const fork::Timeline fresh = fork::SimulateTimeline(sc.w0Snap, sc.params, mutatedStream, sc.ticks);
        check(forked.fullDigest == fresh.fullDigest,
              "fork@0: fork-at-0 + mutate == a FRESH sim from the mutated initial (full digest equal)");
        check(fork::DiffTimelines(fresh, forked).identical, "fork@0: the two timelines are fully identical");
    }

    // ===== (e) TIMELINE TREE — original + 2 counterfactuals -> 3 distinct digests; shared prefix equal. =
    {
        const uint64_t o = sc.original.fullDigest, a = sc.resimA.fullDigest, b = sc.resimB.fullDigest;
        check(o != a && o != b && a != b, "tree: the original + 2 counterfactuals are 3 DISTINCT timelines");
        // Both branches share ticks [0, forkTick) byte-for-byte with the original AND with each other.
        bool prefixEq = true;
        for (uint32_t t = 0; t < fork::kFk1ForkTick; ++t)
            if (sc.original.digests[t] != sc.resimA.digests[t] || sc.original.digests[t] != sc.resimB.digests[t])
                prefixEq = false;
        check(prefixEq, "tree: the shared prefix [0, forkTick) is byte-identical across all 3 timelines");
        // Both branches diverge at EXACTLY the fork tick (the mutation is injected there).
        check(sc.diffA.firstDivergence == (int)fork::kFk1ForkTick &&
              sc.diffB.firstDivergence == (int)fork::kFk1ForkTick,
              "tree: both branches diverge at EXACTLY the fork tick");
        std::printf("fk1: tree {orig:0x%016llx, A:0x%016llx, B:0x%016llx}\n",
                    (unsigned long long)o, (unsigned long long)a, (unsigned long long)b);
    }

    // ===== (f) OUTCOME CHANGE — a mutation flips the final outcome (original-final != fork-final). =====
    {
        // Branch B (an ability +25 hp) flips a GAMEPLAY outcome: the player's final Health changes.
        check(sc.outB.playerHealth != sc.outOriginal.playerHealth,
              "outcome: the ability counterfactual flips the player's final Health");
        check(sc.outB.playerHealth == sc.outOriginal.playerHealth + 25,
              "outcome: the final Health changed by EXACTLY the counterfactual +25");
        // Branch A (a physics shove) flips the final WORLD digest (the bodies settle differently).
        check(sc.outA.finalDigest != sc.outOriginal.finalDigest,
              "outcome: the physics counterfactual flips the final world digest");
        check(!fork::OutcomesEqual(sc.outOriginal, sc.outA) && !fork::OutcomesEqual(sc.outOriginal, sc.outB),
              "outcome: BOTH counterfactuals reach a DIFFERENT final outcome than the original");
        std::printf("fk1: outcomes {origHP:%d, A.finalDigest:0x%016llx, B.HP:%d}\n",
                    sc.outOriginal.playerHealth, (unsigned long long)sc.outA.finalDigest, sc.outB.playerHealth);
    }

    // ===== (+) VIZ — the timeline-tree report image is two-run byte-identical + the stats pin. =========
    {
        std::vector<uint8_t> img1, img2;
        fork::Fk1VizStats s1{}, s2{};
        fork::RenderFk1ForkViz(img1, s1);
        fork::RenderFk1ForkViz(img2, s2);
        const bool twoRun = (img1.size() == img2.size()) &&
                            (std::memcmp(img1.data(), img2.data(), img1.size()) == 0) &&
                            (s1.pixDigest == s2.pixDigest);
        check(twoRun, "viz: the timeline-tree report image is two-run BYTE-IDENTICAL");
        check(s1.branches == 2u && s1.ticks == sc.ticks && s1.forkTick == fork::kFk1ForkTick,
              "viz: report covers 2 branches over all ticks at the fork tick");
        check(s1.firstDivergenceA == (int)fork::kFk1ForkTick && s1.firstDivergenceB == (int)fork::kFk1ForkTick,
              "viz: both branches' divergence markers are at the fork tick");
        check(s1.outcomeChanged, "viz: the report flags the outcome CHANGED");
        check(s1.origFullDigest == sc.original.fullDigest && s1.forkAFullDigest == sc.resimA.fullDigest &&
              s1.forkBFullDigest == sc.resimB.fullDigest, "viz: the report pins the three timeline digests");
        std::printf("fk1-fork: {ticks:%u, forkTick:%u, branches:%u, firstDivergence:%d, outcomeChanged:%s, "
                    "pixDigest:0x%016llx}\n",
                    s1.ticks, s1.forkTick, s1.branches, s1.firstDivergenceA,
                    s1.outcomeChanged ? "true" : "false", (unsigned long long)s1.pixDigest);
    }

    if (g_fail == 0) std::printf("fork_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
