// Slice DX5 — THE RECORD->REPLAY->ASSERT-DETERMINISM HARNESS (FLAGSHIP #31 THE AGENT EXPERIENCE,
// hf::game::verdict). DX5 productizes the lockstep MOAT (#27) as a testable agent artifact: a command
// stream RECORDED against the FIXED canonical gameplay world, REPLAYED via RunVerdictLockstep, and the
// final world DIGESTED + asserted bit-identical against a golden replay file.
//
// What this test PINS (the spec's DX5 proofs):
//   * DigestSnapshot DETERMINISM — two digests of the SAME final world are equal.
//   * DigestSnapshot ORDER-INDEPENDENCE under ECS handle churn — a RestoreWorld'd world (whose
//     ecs::Entity handles are re-allocated) digests IDENTICALLY (the order[]-keyed contract).
//   * SerializeReplay -> ParseReplay ROUND-TRIP — the ReplayFile survives (ticks/stream/digest), and
//     re-serializing the parsed file reproduces the exact text.
//   * VerifyReplay TRUE for the recorded replay (replayed digest == recorded digest).
//   * TAMPER — flip one command arg -> VerifyReplay FALSE + the digest DIFFERS (the regression guard).
//   * LOCKSTEP authority==replica (RunVerdictLockstep's outIdentical over the WHOLE world).
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests. verdict.h is APPEND-ONLY (VD1-VD6
// byte-frozen); this exercises only the DX5 append.
#include "game/verdict.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace verdict = hf::game::verdict;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Build the canonical replay (world0 + params + stream + ticks) and RECORD: lockstep -> the final
// world digest. Returns the recorded ReplayFile.
static verdict::ReplayFile RecordCanonical() {
    verdict::VerdictWorld world0;
    const verdict::CanonicalReplay cr = verdict::BuildCanonicalReplay(world0);
    const verdict::VerdictSnapshot world0Snap = verdict::SnapshotWorld(world0);
    const verdict::VerdictSnapshot finalSnap =
        verdict::RunVerdictLockstep(world0Snap, cr.params, cr.stream, cr.ticks);
    verdict::ReplayFile rf;
    rf.ticks       = cr.ticks;
    rf.stream      = cr.stream;
    rf.finalDigest = verdict::DigestSnapshot(finalSnap);
    return rf;
}

int main() {
    HF_TEST_MAIN_INIT();

    // === DigestSnapshot determinism: two digests of the same final world are equal. ===
    verdict::VerdictWorld world0;
    const verdict::CanonicalReplay cr = verdict::BuildCanonicalReplay(world0);
    const verdict::VerdictSnapshot world0Snap = verdict::SnapshotWorld(world0);
    const verdict::VerdictSnapshot finalA =
        verdict::RunVerdictLockstep(world0Snap, cr.params, cr.stream, cr.ticks);
    const verdict::VerdictSnapshot finalB =
        verdict::RunVerdictLockstep(world0Snap, cr.params, cr.stream, cr.ticks);
    const std::string digA = verdict::DigestSnapshot(finalA);
    const std::string digB = verdict::DigestSnapshot(finalB);
    check(digA == digB, "DigestSnapshot determinism (two runs equal)");
    check(digA.size() == 16, "DigestSnapshot is 16 hex digits");

    // === DigestSnapshot order-independence under ECS handle churn: a RestoreWorld'd world (handles
    // re-allocated) digests identically. RestoreWorld into a FRESH world (seed hulls via ClonePeer) and
    // re-snapshot -> same digest. ===
    {
        verdict::VerdictWorld restored = verdict::ClonePeer(finalA, cr.params);
        // Churn the ecs handles further: restore the SAME snapshot AGAIN into the same world (a second
        // re-allocation pass) so the opaque handles certainly differ from finalA's.
        verdict::RestoreWorld(restored, finalA);
        const std::string digR = verdict::DigestSnapshot(verdict::SnapshotWorld(restored));
        check(digR == digA, "DigestSnapshot order-independent under ECS handle churn (restored == orig)");
    }

    // === SerializeReplay -> ParseReplay round-trip: the ReplayFile survives. ===
    const verdict::ReplayFile rf = RecordCanonical();
    const std::string text = verdict::SerializeReplay(rf);
    const verdict::ReplayFile parsed = verdict::ParseReplay(text);
    check(parsed.ticks == rf.ticks, "ParseReplay ticks round-trip");
    check(parsed.finalDigest == rf.finalDigest, "ParseReplay finalDigest round-trip");
    check(parsed.stream.size() == rf.stream.size(), "ParseReplay stream size round-trip");
    bool streamEq = (parsed.stream.size() == rf.stream.size());
    for (size_t i = 0; streamEq && i < rf.stream.size(); ++i) {
        const verdict::Command& a = rf.stream[i];
        const verdict::Command& b = parsed.stream[i];
        streamEq = a.tick == b.tick && a.kind == b.kind && a.target == b.target &&
                   a.arg.x == b.arg.x && a.arg.y == b.arg.y && a.arg.z == b.arg.z;
    }
    check(streamEq, "ParseReplay stream content round-trip (every command field exact)");
    // Re-serializing the parsed file reproduces the exact text (the canonical text is a fixed point).
    check(verdict::SerializeReplay(parsed) == text, "Serialize(Parse(Serialize(rf))) == text (fixed point)");

    // === VerifyReplay TRUE for the recorded replay (replayed digest == recorded digest). ===
    check(verdict::VerifyReplay(rf), "VerifyReplay true for the recorded replay");
    // And true via the parsed-from-text replay (the full round-trip through JSON).
    check(verdict::VerifyReplay(parsed), "VerifyReplay true for the parsed-from-text replay");

    // === TAMPER: flip ONE command's arg -> VerifyReplay FALSE + the digest DIFFERS. ===
    {
        verdict::ReplayFile tampered = rf;
        check(!tampered.stream.empty(), "tamper precondition: stream non-empty");
        // Flip the player nudge impulse (command 0's arg.x) — a physics-divergent change.
        tampered.stream[0].arg.x = -tampered.stream[0].arg.x - verdict::kOne;
        check(!verdict::VerifyReplay(tampered), "tamper-detect: VerifyReplay false on a flipped arg");
        // The replayed final digest of the tampered stream DIFFERS from the recorded digest.
        verdict::VerdictWorld tw0;
        const verdict::CanonicalReplay tcr = verdict::BuildCanonicalReplay(tw0);
        const verdict::VerdictSnapshot tSnap0 = verdict::SnapshotWorld(tw0);
        const verdict::VerdictSnapshot tFinal =
            verdict::RunVerdictLockstep(tSnap0, tcr.params, tampered.stream, tampered.ticks);
        check(verdict::DigestSnapshot(tFinal) != rf.finalDigest,
              "tamper-detect: tampered final digest DIFFERS from recorded");
    }

    // === LOCKSTEP authority==replica over the WHOLE world (RunVerdictLockstep's outIdentical). ===
    {
        bool identical = false;
        verdict::VerdictWorld lw0;
        const verdict::CanonicalReplay lcr = verdict::BuildCanonicalReplay(lw0);
        const verdict::VerdictSnapshot lSnap0 = verdict::SnapshotWorld(lw0);
        (void)verdict::RunVerdictLockstep(lSnap0, lcr.params, lcr.stream, lcr.ticks, &identical);
        check(identical, "lockstep authority==replica BIT-IDENTICAL (whole world)");
    }

    if (g_fail == 0) std::printf("agent_replay_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
