// Slice AC1 — SERVER-AUTHORITATIVE RE-SIMULATION VERIFIER (provable anti-cheat), hf::net::authority_verify.
//
// Because the whole HF sim is bit-exact deterministic (VD1-VD6 whole-world lockstep from an input stream
// ALONE), a server can INGEST a suspect client's inputs, RE-SIMULATE authoritatively, and PROVE via per-tick
// digest comparison whether the client faked its outcomes — rejecting the diverging peer at the EXACT tick.
// authority_verify.h composes verdict.h (RunVerdictLockstep/SimVerdictTick/DigestSnapshot) + session.h (the
// NS5 DesyncDetector located-divergence machinery) READ-ONLY; it adds NO field, edits NO frozen function.
//
// What this test PINS (the spec's proofs):
//   (a) HONEST     — an honest client's claim (the true trace) -> VERIFIED, firstDivergentTick == -1.
//   (b) CHEAT      — a single-tick tampered claim -> REJECTED at EXACTLY that tick; the server's true digest
//                    at that tick is pinned (== the authoritative re-sim's digest).
//   (c) CAUSAL     — tampering tick T does NOT diverge before T (ticks < T match the authoritative trace).
//   (d) COMMITMENT — the input-stream commitment pins; a changed input flips it (two-run stable).
//   (e) REPRODUCIBLE — two independent verifier runs -> the identical verdict + located tick + digest.
//   (f) MULTI-CLIENT — honest + cheater in one scenario -> honest OK, cheater caught at its exact tamper tick.
//   (+) EMPTY      — a zero-tick / empty stream verifies trivially; the viz report is two-run byte-identical.
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "net/authority_verify.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace verdict = hf::game::verdict;
namespace net     = hf::net;
using verdict::fx;
using verdict::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Return a 16-hex string GUARANTEED different from `h` (flip the last nibble) — a hand-tampered digest.
static std::string FlipLastHex(const std::string& h) {
    if (h.empty()) return std::string("0000000000000001");
    std::string t = h;
    char c = t[t.size() - 1];
    t[t.size() - 1] = (c == 'f') ? 'e' : (char)(c + 1);
    return t;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- Build the canonical adversarial scenario (the shared HEADLINE scene). ----------------------
    verdict::VerdictWorld world0;
    const net::Ac1Scenario sc = net::BuildAc1Scenario(world0);
    const verdict::VerdictSnapshot w0Snap = verdict::SnapshotWorld(world0);
    // The server's authoritative ground-truth trace over the SUBMITTED honest inputs.
    const std::vector<std::string> server =
        net::VerdictDigestTrace(w0Snap, sc.params, sc.honestInputs, sc.ticks);
    check(server.size() == sc.ticks, "trace: server digest trace has one entry per tick");
    check(sc.honestClaim.size() == sc.ticks, "trace: honest claim covers every tick");

    // Determinism of the authoritative re-sim: two traces byte-identical.
    {
        const std::vector<std::string> server2 =
            net::VerdictDigestTrace(w0Snap, sc.params, sc.honestInputs, sc.ticks);
        bool same = (server.size() == server2.size());
        for (size_t i = 0; same && i < server.size(); ++i) if (server[i] != server2[i]) same = false;
        check(same, "trace: the authoritative re-sim is deterministic (two runs byte-identical)");
        // The server trace IS the honest client's true claim (an honest client re-derives the real sim).
        check(same && server == sc.honestClaim, "trace: server trace == the honest client's true claim");
    }

    // ===== (a) HONEST — the honest claim verifies clean. =============================================
    {
        const net::VerifyResult r =
            net::Verify(w0Snap, sc.params, sc.honestInputs, sc.honestClaim, sc.ticks);
        check(r.ok, "honest: an honest client's claim is VERIFIED (ok)");
        check(r.firstDivergentTick == -1, "honest: firstDivergentTick == -1 (no divergence)");
        check(r.serverDigest.empty() && r.clientDigest.empty(), "honest: no divergence evidence recorded");
    }

    // ===== (b) CHEAT CAUGHT — a single-tick tampered claim is caught at EXACTLY that tick. ===========
    const uint32_t kTamperT = 10u;
    check(kTamperT < sc.ticks, "cheat: the tamper tick is within range");
    {
        std::vector<std::string> tampered = sc.honestClaim;   // start from the TRUE claim
        tampered[kTamperT] = FlipLastHex(tampered[kTamperT]); // fake ONE outcome digest
        const net::VerifyResult r =
            net::Verify(w0Snap, sc.params, sc.honestInputs, tampered, sc.ticks);
        check(!r.ok, "cheat: a tampered claim is REJECTED (!ok)");
        check(r.firstDivergentTick == (int)kTamperT, "cheat: caught at EXACTLY the tampered tick T");
        // The server's TRUE digest at T is pinned (== the authoritative re-sim's digest at T).
        check(r.serverDigest == server[kTamperT], "cheat: server's true digest at T pinned (== re-sim digest)");
        check(r.clientDigest == tampered[kTamperT], "cheat: the client's faked digest at T is reported");
        check(r.serverDigest != r.clientDigest, "cheat: server != client at the located tick");
    }

    // ===== (c) CAUSAL — a lie at T leaves ticks < T matching (the located tick is exactly T, not earlier). =
    {
        const uint32_t kLate = 15u;   // tamper only a LATER tick
        std::vector<std::string> tampered = sc.honestClaim;
        tampered[kLate] = FlipLastHex(tampered[kLate]);
        const net::VerifyResult r =
            net::Verify(w0Snap, sc.params, sc.honestInputs, tampered, sc.ticks);
        check(r.firstDivergentTick == (int)kLate, "causal: caught at the LATE tamper tick (not earlier)");
        // Directly assert causality: every tick BEFORE the tamper matches the authoritative trace.
        bool allBeforeMatch = true;
        for (uint32_t t = 0; t < kLate; ++t) if (server[t] != tampered[t]) allBeforeMatch = false;
        check(allBeforeMatch, "causal: every tick < T matches the authoritative digest (divergence is causal)");
        // And the tamper is the ONLY differing tick (so the FIRST divergence is genuinely at kLate).
        bool onlyAtLate = (server[kLate] != tampered[kLate]);
        for (uint32_t t = kLate + 1; t < sc.ticks; ++t) if (server[t] != tampered[t]) onlyAtLate = false;
        check(onlyAtLate, "causal: only tick T differs (the tamper is isolated)");
    }

    // ===== (d) INPUT COMMITMENT — the stream commitment pins; a changed input flips it. ==============
    {
        const std::string c0 = net::CommitInputStream(sc.honestInputs);
        check(c0 == sc.inputCommitment, "commitment: CommitInputStream pins the submitted stream");
        check(net::CommitInputStream(sc.honestInputs) == c0, "commitment: two runs byte-identical (deterministic)");
        // A changed input (flip one command's arg) flips the commitment -> the client can't retroactively
        // change what it 'played' without breaking the pin.
        std::vector<verdict::Command> changed = sc.honestInputs;
        check(!changed.empty(), "commitment: the canonical stream is non-empty");
        changed[0].arg.x = (fx)((int64_t)changed[0].arg.x + (int64_t)kOne);
        check(net::CommitInputStream(changed) != c0, "commitment: a changed input FLIPS the commitment");
        // An added command also flips it (a client can't inject an extra input undetected).
        std::vector<verdict::Command> added = sc.honestInputs;
        added.push_back(verdict::Command{3u, verdict::kCmdMove, sc.player, verdict::FxVec3{kOne, 0, 0}});
        check(net::CommitInputStream(added) != c0, "commitment: an ADDED input FLIPS the commitment");
    }

    // ===== (e) REPRODUCIBLE VERDICT — two independent runs -> the identical verdict + located tick. ==
    {
        const net::VerifyResult a =
            net::Verify(w0Snap, sc.params, sc.honestInputs, sc.cheaterClaim, sc.ticks);
        const net::VerifyResult b =
            net::Verify(w0Snap, sc.params, sc.honestInputs, sc.cheaterClaim, sc.ticks);
        check(net::VerifyResultsEqual(a, b), "reproducible: two verifier runs -> the IDENTICAL verdict");
        check(a.firstDivergentTick == b.firstDivergentTick && a.serverDigest == b.serverDigest,
              "reproducible: the located tick + server digest are identical run-to-run");
    }

    // ===== (f) MULTI-CLIENT — honest + cheater in ONE scenario: honest OK, cheater caught at its tick. =
    {
        const net::VerifyResult honest =
            net::Verify(w0Snap, sc.params, sc.honestInputs, sc.honestClaim, sc.ticks);
        const net::VerifyResult cheater =
            net::Verify(w0Snap, sc.params, sc.honestInputs, sc.cheaterClaim, sc.ticks);
        check(honest.ok && honest.firstDivergentTick == -1, "multi-client: the HONEST client is VERIFIED");
        check(!cheater.ok, "multi-client: the CHEATER client is REJECTED");
        check(cheater.firstDivergentTick == (int)net::kAc1CheatTick,
              "multi-client: the cheater caught at its EXACT impossible-outcome tick");
        // The cheat is an IMPOSSIBLE FINAL STATE (a claimed +50 health the submitted inputs never produce):
        // the server's true digest at the tick != the cheater's claim, but the digests match BEFORE it.
        check(server[net::kAc1CheatTick] == sc.honestClaim[net::kAc1CheatTick],
              "multi-client: the honest true outcome at the cheat tick is the server's digest");
        check(sc.cheaterClaim[net::kAc1CheatTick] != server[net::kAc1CheatTick],
              "multi-client: the cheater's claimed outcome diverges from the re-sim");
        bool matchBefore = true;
        for (uint32_t t = 0; t < net::kAc1CheatTick; ++t)
            if (sc.cheaterClaim[t] != server[t]) matchBefore = false;
        check(matchBefore, "multi-client: the cheat only shows AFTER the injected tick (causal)");
    }

    // ===== (+) EMPTY / TRIVIAL — a zero-tick stream verifies trivially; an empty-input honest run OK. ==
    {
        const net::VerifyResult trivial =
            net::Verify(w0Snap, sc.params, std::vector<verdict::Command>{}, std::vector<std::string>{}, 0u);
        check(trivial.ok && trivial.firstDivergentTick == -1, "empty: a zero-tick stream verifies trivially");
        // A NON-zero run over an EMPTY input stream, honestly claimed (the true trace), still verifies.
        const uint32_t kN = 8u;
        const std::vector<verdict::Command> noInputs;
        const std::vector<std::string> honestNoInput =
            net::VerdictDigestTrace(w0Snap, sc.params, noInputs, kN);
        const net::VerifyResult r = net::Verify(w0Snap, sc.params, noInputs, honestNoInput, kN);
        check(r.ok && r.firstDivergentTick == -1, "empty: an honest empty-input run verifies clean");
        // An UNDER-claim (fewer digests than ticks) is caught at the first unclaimed tick (unprovable).
        std::vector<std::string> shortClaim = honestNoInput;
        shortClaim.resize(kN - 3u);
        const net::VerifyResult under = net::Verify(w0Snap, sc.params, noInputs, shortClaim, kN);
        check(!under.ok && under.firstDivergentTick == (int)(kN - 3u),
              "empty: an under-claim is rejected at the first unclaimed tick");
    }

    // ===== VIZ — the verification-report image is two-run byte-identical + the verdict pins. =========
    {
        std::vector<uint8_t> img1, img2;
        net::Ac1VizStats s1{}, s2{};
        net::RenderAc1VerifyViz(img1, s1);
        net::RenderAc1VerifyViz(img2, s2);
        const bool twoRun = (img1.size() == img2.size()) &&
                            (std::memcmp(img1.data(), img2.data(), img1.size()) == 0) &&
                            (s1.pixDigest == s2.pixDigest);
        check(twoRun, "viz: the verification-report image is two-run BYTE-IDENTICAL");
        check(s1.clients == 2u && s1.ticks == sc.ticks, "viz: report covers 2 clients over all ticks");
        check(s1.honestVerdict == true, "viz: the honest client's banner is VERIFIED");
        check(s1.cheaterCaughtTick == (int)net::kAc1CheatTick, "viz: the cheater's banner is REJECTED @ the tamper tick");
        check(s1.inputCommitment == sc.inputCommitment, "viz: the report carries the input commitment");
        std::printf("ac1-verify: {ticks:%u, clients:%u, honestVerdict:%s, cheaterCaughtTick:%d, "
                    "commit:%s, pixDigest:0x%016llx}\n",
                    s1.ticks, s1.clients, s1.honestVerdict ? "VERIFIED" : "REJECTED",
                    s1.cheaterCaughtTick, s1.inputCommitment.c_str(), (unsigned long long)s1.pixDigest);
    }

    if (g_fail == 0) std::printf("authority_verify_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
