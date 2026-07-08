// Slice GAME1 — A COMPLETE DETERMINISTIC ROLLBACK-PHYSICS GAME: a 2-player physics KNOCKOUT DUEL,
// hf::game::duel. The engine's BEAT-UE5 P0 — the first actual GAME (samples/ held only hello_triangle). A
// small-but-complete headless deterministic match SIMULATION that COMPOSES the whole determinism moat stack
// (bit-exact integer physics + whole-world lockstep/rollback + replay + what-if fork + provable anti-cheat)
// into ONE deliverable, plus the shipped gameplay-framework (GAS1 shove ability with cost+cooldown, GT1
// State.Stunned gate, GC1 cue event stream). A category UE5's float architecture is structurally disqualified
// from: a bit-exact, rollback-replayable, fork-able, provably-fair game.
//
// What this test PINS (the spec's proofs (a)-(i)):
//   (a) KNOCKOUT — a scripted shove sequence knocks the opponent past the ring edge at the EXACT tick;
//                  the round winner is pinned.
//   (b) MATCH    — the best-of-3 canonical match -> pinned final score + winner + full match digest.
//   (c) DETERMIN — two runs of the match are byte-identical (matchDigest); MSVC == clang (pure integer).
//   (d) LOCKSTEP — a peer re-derives the round bit-for-bit from the emitted stream ALONE (RunVerdictLockstep);
//                  a rollback corrects a mispredicted input (RunVerdictRollback, corrected + diverged).
//   (e) REPLAY   — the round records to a verdict ReplayFile + replays bit-identically (pinned demo hash).
//   (f) FORK     — fork the winning round + inject one counter-shove -> a DIFFERENT winner (both pinned).
//   (g) ANTICHEAT— the honest match verifies; a "I won" claim is rejected at the EXACT divergence tick.
//   (h) ABIL/TAG/CUE — shove costs stamina + goes on cooldown (mana pin); a stunned player's shove is BLOCKED
//                  (GT1 pin); impact cues fire at the exact hit ticks (GC1 pin).
//   (i) referee consistency + viz two-run byte-identical + the digests are printed for the MSVC==clang proof.
//
// Pure C++ (hf_core), ASan-eligible. Cross-compiler proof: every printed digest is IDENTICAL on MSVC + local
// clang (the same pure-integer computation).
#include "game/duel.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::game;
namespace duel    = hf::game::duel;
namespace verdict = hf::game::verdict;
namespace tags    = hf::game::tags;
using verdict::fx;
using verdict::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    const tags::TagRegistry reg = duel::MakeDuelRegistry();

    // ===== (a) KNOCKOUT — a scripted shove sequence rings-out the opponent at the EXACT tick. =============
    {
        verdict::VerdictWorld world0;
        const duel::DuelScene scene = duel::BuildDuelScene(world0);
        // Player 0 aggressive (shoves t2/t6/t10); player 1 idle.
        const std::vector<duel::DuelInput> sA = duel::MakeRoundScript(0u, 0);
        const std::vector<duel::DuelInput> sB = duel::MakeRoundScript(0u, 1);
        const duel::RoundResult rr = duel::RunDuelRound(scene, reg, sA, sB, duel::kRoundTicks);
        check(rr.verdictOut.knockoutTick == 14, "knockout: player 1 rings out at EXACTLY tick 14");
        check(rr.verdictOut.winner == 0 && rr.verdictOut.loser == 1, "knockout: player 0 wins the round");
        check(rr.shovesLanded[0] == 3 && rr.shovesLanded[1] == 0, "knockout: player 0 landed 3 shoves, player 1 none");
        check(rr.emittedStream.size() == 3, "knockout: exactly 3 impulses were lowered into the physics stream");
        check(rr.finalDigest == "d908251e954108c9", "knockout: round-0 final DigestSnapshot pinned");
        std::printf("game1 (a) KNOCKOUT: ko@%d winner=%d landed=%u digest=%s\n",
                    rr.verdictOut.knockoutTick, rr.verdictOut.winner, rr.shovesLanded[0], rr.finalDigest.c_str());
    }

    // ===== (b) MATCH + (c) DETERMINISM — best-of-3 -> pinned score/winner/match digest; two runs identical.
    const duel::MatchResult mr = duel::RunDuelMatch();
    {
        check(mr.roundsPlayed == 3, "match: best-of-3 played all three rounds (2-1)");
        check(mr.score[0] == 2 && mr.score[1] == 1, "match: final score is 2-1");
        check(mr.matchWinner == 0, "match: player 0 wins the match");
        check(mr.matchDigest == 0x78123003c3a55a37ull, "match: full match digest pinned");
        // Round-by-round pins (round 2 replays round 0's scene -> identical digest).
        check(mr.rounds[0].verdictOut.winner == 0 && mr.rounds[0].verdictOut.knockoutTick == 14, "match: round 0 -> P0 @14");
        check(mr.rounds[1].verdictOut.winner == 1 && mr.rounds[1].verdictOut.knockoutTick == 14, "match: round 1 -> P1 @14");
        check(mr.rounds[2].verdictOut.winner == 0 && mr.rounds[2].verdictOut.knockoutTick == 14, "match: round 2 -> P0 @14");
        check(mr.rounds[0].finalDigest == "d908251e954108c9" && mr.rounds[1].finalDigest == "409d354ea01f1ba6",
              "match: round-0 and round-1 final digests pinned");
        check(mr.rounds[2].finalDigest == mr.rounds[0].finalDigest, "match: round 2 == round 0 (deterministic reset)");
        check(mr.rounds[0].traceDigest == 0x93efee8a6dad5d31ull && mr.rounds[1].traceDigest == 0xce6a368091fe710bull,
              "match: round trace digests pinned");
        const duel::MatchResult mr2 = duel::RunDuelMatch();
        check(mr2.matchDigest == mr.matchDigest, "determinism: two runs of the match are byte-identical");
        std::printf("game1 (b) MATCH: score %u-%u winner=%d matchDigest=0x%016llx\n",
                    mr.score[0], mr.score[1], mr.matchWinner, (unsigned long long)mr.matchDigest);
    }

    // Build a canonical round-0 scene + stream reused by (d)-(g).
    verdict::VerdictWorld pw;
    const duel::DuelScene scene = duel::BuildDuelScene(pw);
    const std::vector<verdict::Command> stream = mr.rounds[0].emittedStream;
    const uint32_t ticks = mr.rounds[0].ticks;

    // ===== (d) LOCKSTEP + ROLLBACK — a peer re-derives bit-for-bit; a mispredict is corrected. ============
    {
        bool identical = false;
        const std::string lockDigest = duel::DuelLockstep(scene, stream, ticks, &identical);
        check(identical, "lockstep: two peers re-derive the round bit-for-bit from the emitted stream ALONE");
        check(lockDigest == mr.rounds[0].finalDigest, "lockstep: the peer digest == the played round's final digest");
        bool corrected = false, diverged = false;
        duel::DuelRollback(scene, stream, ticks, /*rollbackAt*/6u, &corrected, &diverged);
        check(corrected, "rollback: a mispredicted input is CORRECTED bit-exactly to the authority");
        check(diverged, "rollback: the injected misprediction ACTUALLY diverged (a non-vacuous control)");
        std::printf("game1 (d) LOCKSTEP: identical=%d rollback{corrected=%d diverged=%d} digest=%s\n",
                    identical, corrected, diverged, lockDigest.c_str());
    }

    // ===== (e) REPLAY — record to a verdict ReplayFile + replay bit-identically (pinned demo hash). =======
    {
        uint64_t demoHash = 0; bool replayOk = false;
        (void)duel::DuelReplayDemo(scene, stream, ticks, &demoHash, &replayOk);
        check(replayOk, "replay: the recorded demo replays to the IDENTICAL final digest (record == replay)");
        check(demoHash == 0x6b2b59cf6725212bull, "replay: the serialized demo-file byte-hash is pinned");
        std::printf("game1 (e) REPLAY: ok=%d demoHash=0x%016llx\n", replayOk, (unsigned long long)demoHash);
    }

    // ===== (f) FORK — inject one counter-shove at the fork tick -> a DIFFERENT winner (both pinned). ======
    {
        const duel::DuelForkProof fk =
            duel::DuelForkChangeWinner(scene, stream, ticks, duel::kForkTick, duel::MakeForkCounterShove(scene));
        check(fk.origWinner == 0, "fork: the ORIGINAL round winner is player 0");
        check(fk.forkWinner == 1, "fork: the COUNTERFACTUAL winner is player 1 (a DIFFERENT outcome)");
        check(fk.winnerChanged, "fork: the injected counter-shove flipped the winner");
        check(fk.firstDivergence == (int)duel::kForkTick, "fork: the timelines diverge at EXACTLY the injection tick");
        check(fk.forkKnockout == 6, "fork: in the counterfactual player 0 is rung out at tick 6 (earlier)");
        check(fk.origFullDigest == 0x9e08b21a023dc2c6ull && fk.forkFullDigest == 0x5cb62f292697883aull,
              "fork: both timeline-tree fingerprints pinned (original + counterfactual)");
        check(fk.origFullDigest != fk.forkFullDigest, "fork: the two timelines are distinct");
        std::printf("game1 (f) FORK: origWinner=%d(ko@%d) -> forkWinner=%d(ko@%d) origFull=0x%016llx forkFull=0x%016llx\n",
                    fk.origWinner, fk.origKnockout, fk.forkWinner, fk.forkKnockout,
                    (unsigned long long)fk.origFullDigest, (unsigned long long)fk.forkFullDigest);
    }

    // ===== (g) ANTI-CHEAT — the honest match verifies; a faked "I won" claim is rejected at the exact tick.
    {
        const duel::DuelAntiCheatProof ac = duel::DuelAntiCheat(scene, stream, ticks, /*cheatTick*/6u);
        check(ac.honestVerified, "anti-cheat: the HONEST client's claimed outcomes verify (firstDivergentTick == -1)");
        check(ac.cheaterCaughtTick == 6, "anti-cheat: the CHEATER is REJECTED at EXACTLY the tick of the faked outcome");
        check(ac.inputCommitment == "214bfe9492a33a9e", "anti-cheat: the input-stream integrity commitment is pinned");
        std::printf("game1 (g) ANTICHEAT: honest=%d cheaterCaught=%d commit=%s (server:%s != client:%s)\n",
                    ac.honestVerified, ac.cheaterCaughtTick, ac.inputCommitment.c_str(),
                    ac.serverDigest.c_str(), ac.clientDigest.c_str());
    }

    // ===== (h) ABILITIES / TAGS / CUES — shove costs stamina + cooldown; a stunned shove is BLOCKED; cues fire.
    {
        // Round 0: player 0 landed 3 shoves -> mana 100 - 3*20 = 40; player 1 never shoved -> mana 100.
        check((mr.rounds[0].finalMana[0] >> 16) == 40, "abilities: 3 shoves cost 3*20 mana -> player 0 at 40");
        check((mr.rounds[0].finalMana[1] >> 16) == 100, "abilities: player 1 never shoved -> mana unchanged at 100");
        check(mr.rounds[0].stunsApplied == 3, "tags: each landed shove applied the State.Stunned aura (3 total)");
        // Round 1: player 1 aggressive; the STUNNED player 0 tried to shove at t3 and was BLOCKED (GT1 gate).
        check(mr.rounds[1].shovesBlocked[0] == 1, "tags: the stunned player 0's shove was BLOCKED by State.Stunned");
        check(mr.rounds[1].shovesLanded[1] == 3, "tags: player 1 landed its 3 shoves");
        // Cues: 3 impact cues fire at the exact shove ticks + 1 knockout cue.
        check(mr.rounds[0].impactTicks.size() == 3 &&
              mr.rounds[0].impactTicks[0] == 2 && mr.rounds[0].impactTicks[1] == 6 && mr.rounds[0].impactTicks[2] == 10,
              "cues: impact cues fire at the EXACT hit ticks (2, 6, 10)");
        check(mr.rounds[0].cueLog.size() == 4, "cues: 3 impact cues + 1 knockout cue in the event stream");
        check(mr.rounds[0].cueLog.back().cueTag == duel::DuelCueKnockout(reg), "cues: the last cue is the KNOCKOUT cue");
        std::printf("game1 (h) ABIL/TAG/CUE: mana[%d,%d] stuns=%u blocked(P0 r1)=%u impacts@{2,6,10} cues=%zu\n",
                    (int)(mr.rounds[0].finalMana[0] >> 16), (int)(mr.rounds[0].finalMana[1] >> 16),
                    mr.rounds[0].stunsApplied, mr.rounds[1].shovesBlocked[0], mr.rounds[0].cueLog.size());
    }

    // ===== (i) REFEREE CONSISTENCY + VIZ two-run byte-identical + the pinned viz digests. =================
    {
        // The referee re-sim over the emitted stream reproduces the played round's knockout bit-exactly.
        const duel::RoundVerdict rv = duel::RefereeStream(scene, stream, ticks);
        check(rv.knockoutTick == mr.rounds[0].verdictOut.knockoutTick && rv.winner == mr.rounds[0].verdictOut.winner,
              "referee: the pure re-sim reproduces the played round's knockout/winner");

        std::vector<uint8_t> img1, img2;
        duel::Duel1VizStats s1{}, s2{};
        duel::RenderDuelViz(img1, s1);
        duel::RenderDuelViz(img2, s2);
        const bool twoRun = (img1.size() == img2.size()) &&
                            (std::memcmp(img1.data(), img2.data(), img1.size()) == 0) && (s1.pixDigest == s2.pixDigest);
        check(twoRun, "viz: two renders of the duel report are byte-identical");
        check(s1.width == 640 && s1.height == 400, "viz: the report image is 640x400");
        check(s1.deterministic && s1.lockstep && s1.replayOk && s1.forkChanged && s1.cheaterCaught == 6,
              "viz: all five moat proofs pass in the report");
        check(s1.matchWinner == 0 && s1.score0 == 2 && s1.score1 == 1 && s1.knockoutTick == 14,
              "viz: the scoreboard stats match the match");
        check(s1.pixDigest == 0x1cf6209a4ff4f413ull, "viz: the RGBA8 pixel digest is pinned (strict-zero cross-backend)");
        check(s1.matchDigest == mr.matchDigest, "viz: the report's match digest == the played match");
        std::printf("game1 (i) VIZ: %ux%u twoRun=%d pixDigest=0x%016llx checks{det=%d lock=%d replay=%d fork=%d cheat=%d}\n",
                    s1.width, s1.height, twoRun, (unsigned long long)s1.pixDigest,
                    s1.deterministic, s1.lockstep, s1.replayOk, s1.forkChanged, s1.cheaterCaught);
    }

    if (g_fail == 0) std::printf("duel_test: ALL PASS\n");
    else             std::printf("duel_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
