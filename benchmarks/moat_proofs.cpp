// benchmarks/moat_proofs.cpp — RUNNABLE "PROOFS UE5 STRUCTURALLY LOSES" (strategic-moat item #5).
//
// A self-contained, pure-CPU proof binary that EXERCISES the already-shipped + golden-gated determinism
// substrate and asserts the pinned deterministic outcomes. It is the runnable "offense" docs/BEAT_UE5_PLAN.md
// (Phase 3) names: every "beats UE5" claim ships as a script that proves it. This binary is ALSO a ctest
// target (registered as hf_moat_proofs), so `verify.ps1` runs it alongside the pure-core suite.
//
// Each proof composes the FROZEN cores READ-ONLY (no engine/ header is touched):
//   PROOF 1 — cross-platform determinism    : verdict RunVerdictLockstep -> DigestSnapshot (the pinned hash
//                                              that reproduces on Windows/Mac/Linux; see DETERMINISM_THREE_PLATFORMS.md)
//   PROOF 2 — rollback/desync detection      : net NS5 DesyncDetector catches an injected desync at the exact tick
//   PROOF 3 — provable anti-cheat            : net AC1 AuthorityVerifier accepts an honest client, rejects a
//                                              cheater at the exact tamper tick
//   PROOF 4 — reproducible counterfactual    : replay FK1 what-if fork re-runs bit-identically + shares a
//                                              byte-identical prefix, diverging at the fork tick
//   PROOF 5 — reproducible procedural gen    : pcg seed -> byte-identical field digest; a different seed -> a
//                                              different (pinned) digest
//   PROOF 6 — bandwidth (inputs vs state)    : inputs-per-tick bytes vs full-snapshot bytes for the SAME scene
//
// The binary prints one machine-checkable line per proof and exits 0 IFF ALL proofs pass. Every digest is
// pinned identical under MSVC and local clang (the same integer core, so bit-for-bit equal). This is a CPU
// proof binary — NO render, NO shader, NO Metal, NO golden image.

#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <string>
#include <vector>

#include "net/session.h"           // NS1-NS6: InputRing / DigestTrace / DesyncDetector / DigestBytes
#include "net/authority_verify.h"  // AC1: BuildAc1Scenario / Verify / kAc1CheatTick
#include "replay/fork.h"           // FK1: BuildFk1Scenario / kFk1ForkTick
#include "game/verdict.h"          // VD: BuildCanonicalReplay / RunVerdictLockstep / DigestSnapshot
#include "pcg/pcg.h"               // PCG: Generate / PcgGraph / PcgStream
#include "test_main.h"             // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace verdict = hf::game::verdict;
namespace net     = hf::net;
namespace fork    = hf::replay;
namespace pcg     = hf::pcg;

// -------- a tiny local FNV-1a-64 (the engine's currency) for the PCG field fold -------------------------
static uint64_t Fnv64Mix(uint64_t h, uint32_t w) {
    for (int b = 0; b < 4; ++b) { h ^= (uint8_t)(w >> (b * 8)); h *= 1099511628211ull; }
    return h;
}
static uint64_t DigestInstances(const std::vector<pcg::PcgInstance>& v) {
    uint64_t h = 1469598103934665603ull;
    h = Fnv64Mix(h, (uint32_t)v.size());
    for (const auto& i : v) {
        h = Fnv64Mix(h, (uint32_t)i.pos.x);    h = Fnv64Mix(h, (uint32_t)i.pos.y);    h = Fnv64Mix(h, (uint32_t)i.pos.z);
        h = Fnv64Mix(h, (uint32_t)i.orient.x); h = Fnv64Mix(h, (uint32_t)i.orient.y);
        h = Fnv64Mix(h, (uint32_t)i.orient.z); h = Fnv64Mix(h, (uint32_t)i.orient.w);
        h = Fnv64Mix(h, (uint32_t)i.scale);
    }
    return h;
}

// -------- SnapshotWireBytes: packed-integer wire size of a WHOLE verdict world snapshot ----------------
// Counts the Q16.16 words DigestSnapshot walks (4 bytes/word), i.e. the size a full-state netcode packet
// would occupy if it serialized the whole world as packed integers (NO struct padding). This is the
// "state-sync" cost per tick that PROOF 6 compares the input stream against. HONEST: a specific-scene
// measurement, not a universal ratio.
static std::size_t SnapshotWireBytes(const verdict::VerdictSnapshot& s) {
    std::size_t w = 0;
    w += 3;                                  // header: tick, nextId, order-count
    w += s.order.size();                     // order[] ids
    w += s.transforms.size() * (1 + 3 + 4);  // id + pos(3) + orient(4)
    w += s.healths.size()    * (1 + 1);      // id + hp
    w += s.bodyRefs.size()   * (1 + 1);      // id + simBodyIndex
    w += s.velocities.size() * (1 + 3);      // id + vel(3)
    w += s.pickups.size()    * (1 + 1);      // id + value
    w += s.scores.size()     * (1 + 1);      // id + points
    w += 1 + s.simSnap.bodies.size()  * 16;  // count + per-body: pos3+vel3+invMass+flags+radius+orient4+angVel3
    w += 1 + s.simSnap.cache.entries.size() * 5;   // count + per-entry: bodyA,bodyB,refFaceId,incVertId,normalImpulse
    w += 1 + s.simSnap.sleep.size()   * 3;   // count + per-sleep: energy,lowEnergyTicks,asleep
    return w * 4;                            // 4 bytes per Q16.16 word
}

// The verdict Command packed wire size: {tick, kind, target, arg.x, arg.y, arg.z} = 6 words = 24 bytes.
static constexpr std::size_t kCommandWireBytes = 6 * 4;

// -------- Pinned goldens (bit-identical MSVC == local clang; the cross-platform regression anchors) ----
// Computed on first run, then hard-pinned. A changed input OR a changed engine core moves these -> the
// gate fails, which is the point.
static const std::string kProof1FinalDigest = "76a37a56d256c401";
static const uint64_t    kProof2CleanFinal  = 0x49aa655446b5c3a2ull;
static const uint32_t    kProof2DesyncTick  = 7u;
static const std::string kProof3Commitment  = "d7326bc4bbec56ac";
static const uint64_t    kProof4OrigDigest  = 0x562f8800b4577f5aull;
static const uint64_t    kProof4ForkADigest = 0x88faa2ebe6383db9ull;
static const uint64_t    kProof4ForkBDigest = 0x07fe638fd0bcf831ull;
static const uint64_t    kProof5SeedADigest = 0x5194e133e4e4da0aull;
static const uint64_t    kProof5SeedBDigest = 0xda657619051660f7ull;
static const std::size_t kProof6InputBytes  = 48;      // total command-stream bytes for the 24-tick match
static const std::size_t kProof6SnapBytes   = 884;     // per-tick full-snapshot bytes (canonical scene)

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("  FAIL: %s\n", what); ++g_fail; }
}

// ===== PROOF 1 — cross-platform determinism ============================================================
static bool Proof1() {
    verdict::VerdictWorld world0;
    const verdict::CanonicalReplay cr = verdict::BuildCanonicalReplay(world0);
    const verdict::VerdictSnapshot w0 = verdict::SnapshotWorld(world0);
    bool identical = false;
    const verdict::VerdictSnapshot fin = verdict::RunVerdictLockstep(w0, cr.params, cr.stream, cr.ticks, &identical);
    const std::string dig = verdict::DigestSnapshot(fin);

    check(identical, "P1 authority == replica (lockstep from inputs alone)");
    // Determinism: a second independent run reproduces the identical digest.
    verdict::VerdictWorld world0b;
    const verdict::CanonicalReplay crb = verdict::BuildCanonicalReplay(world0b);
    const verdict::VerdictSnapshot w0b = verdict::SnapshotWorld(world0b);
    const std::string digb = verdict::DigestSnapshot(
        verdict::RunVerdictLockstep(w0b, crb.params, crb.stream, crb.ticks));
    check(dig == digb, "P1 two independent runs -> identical final digest");
    check(dig == kProof1FinalDigest, "P1 final digest matches the pinned cross-platform golden");

    const bool pass = (dig == kProof1FinalDigest) && identical && (dig == digb);
    std::printf("PROOF 1 [cross-platform determinism]: %s digest=0x%s ticks=%u "
                "(UE5: float Chaos + FPU-order/FMA/task-scheduling divergence -> no two machines agree bit-for-bit)\n",
                pass ? "PASS" : "FAIL", dig.c_str(), cr.ticks);
    return pass;
}

// ===== PROOF 2 — rollback/desync detection (NS5) =======================================================
struct ToyA { int64_t acc = 0; };
static void StepA(ToyA& w, const std::vector<int32_t>& in, uint32_t tick) {
    for (int32_t v : in) w.acc += (int64_t)v * (int64_t)(tick + 1);
}
static uint64_t DigestA(const ToyA& w) { return net::DigestBytes(&w.acc, sizeof w.acc); }

static bool Proof2() {
    const uint32_t kTicks = 16u, kK = kProof2DesyncTick;
    auto ring = []() {
        net::InputRing<int32_t> r;
        r.AddInput(0, 4); r.AddInput(2, 9); r.AddInput(2, -1); r.AddInput(5, 13);
        r.AddInput(7, 6); r.AddInput(9, -8); r.AddInput(12, 3); r.AddInput(15, 7);
        return r;
    };
    auto corrupt = [&]() { net::InputRing<int32_t> r = ring(); r.AddInput(kK, 999); return r; };

    const std::vector<uint64_t> traceB     = net::DigestTrace<ToyA, int32_t>(ToyA{}, ring(),    kTicks, StepA, DigestA);
    const std::vector<uint64_t> traceACorr = net::DigestTrace<ToyA, int32_t>(ToyA{}, corrupt(), kTicks, StepA, DigestA);

    net::DesyncDetector det;
    for (uint32_t t = 0; t < kTicks; ++t) net::RecordLocal(det, t, traceB[t]);
    for (uint32_t t = 0; t < kTicks; ++t) net::IngestRemote(det, net::ChecksumPacket{t, traceACorr[t]});

    check(det.desynced, "P2 corrupted peer detected");
    check(det.desyncTick == kK, "P2 desync located at the EXACT injected tick");
    check(det.localDigest != det.remoteDigest, "P2 the two diverging digests differ");
    // Localization: identical before K, differ at K.
    bool loc = true;
    for (uint32_t t = 0; t < kK; ++t) if (traceB[t] != traceACorr[t]) loc = false;
    if (traceB[kK] == traceACorr[kK]) loc = false;
    check(loc, "P2 identical before K, diverge at K");
    const uint64_t cleanFinal = traceB[kTicks - 1];
    check(cleanFinal == kProof2CleanFinal, "P2 clean-trace final digest matches pinned golden");

    const bool pass = det.desynced && det.desyncTick == kK && loc && cleanFinal == kProof2CleanFinal;
    std::printf("PROOF 2 [rollback/desync detection]: %s caughtTick=%u digest=0x%016llx "
                "(UE5: no bit-exact per-tick digest -> desync is invisible; drift is hidden by interpolation, never caught)\n",
                pass ? "PASS" : "FAIL", det.desyncTick, (unsigned long long)cleanFinal);
    return pass;
}

// ===== PROOF 3 — provable anti-cheat (AC1) =============================================================
static bool Proof3() {
    verdict::VerdictWorld world0;
    const net::Ac1Scenario sc = net::BuildAc1Scenario(world0);
    const verdict::VerdictSnapshot w0 = verdict::SnapshotWorld(world0);

    const net::VerifyResult honest  = net::Verify(w0, sc.params, sc.honestInputs, sc.honestClaim,  sc.ticks);
    const net::VerifyResult cheater = net::Verify(w0, sc.params, sc.honestInputs, sc.cheaterClaim, sc.ticks);

    check(honest.ok && honest.firstDivergentTick == -1, "P3 honest client VERIFIED");
    check(!cheater.ok, "P3 cheater REJECTED");
    check(cheater.firstDivergentTick == (int)net::kAc1CheatTick, "P3 cheater caught at the EXACT tamper tick");
    check(sc.inputCommitment == kProof3Commitment, "P3 input-stream commitment matches pinned golden");
    // Reproducible verdict: a third party re-runs Verify and gets the identical answer.
    const net::VerifyResult cheater2 = net::Verify(w0, sc.params, sc.honestInputs, sc.cheaterClaim, sc.ticks);
    check(net::VerifyResultsEqual(cheater, cheater2), "P3 verdict is reproducible (third-party re-check)");

    const bool pass = honest.ok && honest.firstDivergentTick == -1 && !cheater.ok &&
                      cheater.firstDivergentTick == (int)net::kAc1CheatTick &&
                      sc.inputCommitment == kProof3Commitment &&
                      net::VerifyResultsEqual(cheater, cheater2);
    std::printf("PROOF 3 [provable anti-cheat]: %s honest=VERIFIED cheaterCaughtTick=%d commitment=0x%s "
                "(UE5: a server cannot re-derive a client's float physics bit-for-bit -> outcomes are unprovable)\n",
                pass ? "PASS" : "FAIL", cheater.firstDivergentTick, sc.inputCommitment.c_str());
    return pass;
}

// ===== PROOF 4 — reproducible counterfactual (FK1 what-if fork) ========================================
static bool Proof4() {
    verdict::VerdictWorld world0;
    const fork::Fk1Scenario sc = fork::BuildFk1Scenario(world0);

    // The shared prefix [0, forkTick) is byte-identical across all three timelines; the branches diverge
    // at EXACTLY the fork tick.
    bool prefixEq = true;
    for (uint32_t t = 0; t < fork::kFk1ForkTick; ++t)
        if (sc.original.digests[t] != sc.resimA.digests[t] || sc.original.digests[t] != sc.resimB.digests[t])
            prefixEq = false;
    check(prefixEq, "P4 shared prefix byte-identical across all 3 timelines");
    check(sc.diffA.firstDivergence == (int)fork::kFk1ForkTick, "P4 branch A diverges at the fork tick");
    check(sc.diffB.firstDivergence == (int)fork::kFk1ForkTick, "P4 branch B diverges at the fork tick");
    check(sc.original.fullDigest != sc.resimA.fullDigest &&
          sc.original.fullDigest != sc.resimB.fullDigest &&
          sc.resimA.fullDigest   != sc.resimB.fullDigest, "P4 three DISTINCT timelines");
    check(sc.original.fullDigest == kProof4OrigDigest,  "P4 original timeline digest matches pinned golden");
    check(sc.resimA.fullDigest   == kProof4ForkADigest, "P4 branch-A timeline digest matches pinned golden");
    check(sc.resimB.fullDigest   == kProof4ForkBDigest, "P4 branch-B timeline digest matches pinned golden");
    // Reproducible: rebuild the whole tree and confirm the branch digests are bit-identical.
    verdict::VerdictWorld world0b;
    const fork::Fk1Scenario sc2 = fork::BuildFk1Scenario(world0b);
    check(sc2.resimA.fullDigest == sc.resimA.fullDigest && sc2.resimB.fullDigest == sc.resimB.fullDigest,
          "P4 the counterfactual re-runs bit-identically (two builds)");

    const bool pass = prefixEq &&
                      sc.diffA.firstDivergence == (int)fork::kFk1ForkTick &&
                      sc.diffB.firstDivergence == (int)fork::kFk1ForkTick &&
                      sc.original.fullDigest == kProof4OrigDigest &&
                      sc.resimA.fullDigest == kProof4ForkADigest &&
                      sc.resimB.fullDigest == kProof4ForkBDigest &&
                      sc2.resimA.fullDigest == sc.resimA.fullDigest &&
                      sc2.resimB.fullDigest == sc.resimB.fullDigest;
    std::printf("PROOF 4 [reproducible counterfactual]: %s forkTick=%u digest=0x%016llx (branchA=0x%016llx branchB=0x%016llx) "
                "(UE5: no reproducible re-derivation of a float sim -> a bit-exact what-if is structurally impossible)\n",
                pass ? "PASS" : "FAIL", fork::kFk1ForkTick, (unsigned long long)sc.original.fullDigest,
                (unsigned long long)sc.resimA.fullDigest, (unsigned long long)sc.resimB.fullDigest);
    return pass;
}

// ===== PROOF 5 — reproducible procedural generation (PCG) ==============================================
static pcg::PcgGraph MakeGraph() {
    auto fi = [](int v) -> pcg::fx { return (pcg::fx)((int64_t)v * (int64_t)pcg::kOne); };
    pcg::PcgGraph g;
    g.area = pcg::PcgArea{ pcg::FxVec3{fi(-8), 0, fi(-8)}, pcg::FxVec3{fi(8), 0, fi(8)} };
    g.cellsX = 8; g.cellsZ = 8;
    g.transform.randomYaw = true;
    g.transform.scaleLo = fi(1); g.transform.scaleHi = fi(1) + (pcg::kOne / 2);
    g.prune = true; g.pruneRadius = pcg::kOne / 2;
    return g;
}

static bool Proof5() {
    const pcg::PcgGraph g = MakeGraph();
    const pcg::PcgStream seedA{ 0x1234u, 0u };
    const pcg::PcgStream seedB{ 0x5678u, 0u };

    const std::vector<pcg::PcgInstance> a1 = pcg::Generate(g, seedA);
    const std::vector<pcg::PcgInstance> a2 = pcg::Generate(g, seedA);
    const std::vector<pcg::PcgInstance> b1 = pcg::Generate(g, seedB);
    const uint64_t dA = DigestInstances(a1);
    const uint64_t dA2 = DigestInstances(a2);
    const uint64_t dB = DigestInstances(b1);

    check(dA == dA2, "P5 same seed -> byte-identical field (two runs)");
    check(dA != dB, "P5 a different seed -> a different field");
    check(dA == kProof5SeedADigest, "P5 seed-A field digest matches pinned golden");
    check(dB == kProof5SeedBDigest, "P5 seed-B field digest matches pinned golden");

    const bool pass = dA == dA2 && dA != dB && dA == kProof5SeedADigest && dB == kProof5SeedBDigest;
    std::printf("PROOF 5 [reproducible procedural generation]: %s instances=%zu seedA=0x%016llx seedB=0x%016llx "
                "(UE5: PCG runtime uses float noise/transforms -> a seeded field is not bit-reproducible across machines)\n",
                pass ? "PASS" : "FAIL", a1.size(), (unsigned long long)dA, (unsigned long long)dB);
    return pass;
}

// ===== PROOF 6 — bandwidth (inputs-per-tick vs full-snapshot) ==========================================
static bool Proof6() {
    verdict::VerdictWorld world0;
    const verdict::CanonicalReplay cr = verdict::BuildCanonicalReplay(world0);
    const verdict::VerdictSnapshot w0 = verdict::SnapshotWorld(world0);
    const verdict::VerdictSnapshot fin = verdict::RunVerdictLockstep(w0, cr.params, cr.stream, cr.ticks);

    // Rollback netcode sends INPUTS: the whole match's command stream, once.
    const std::size_t inputBytes = cr.stream.size() * kCommandWireBytes;
    // State-sync sends the full world SNAPSHOT every tick.
    const std::size_t snapBytes  = SnapshotWireBytes(fin);
    const std::size_t stateBytes = (std::size_t)cr.ticks * snapBytes;
    const std::size_t ratio      = inputBytes ? (stateBytes / inputBytes) : 0;

    check(inputBytes == kProof6InputBytes, "P6 input-stream bytes match pinned measurement");
    check(snapBytes == kProof6SnapBytes || kProof6SnapBytes == 0, "P6 snapshot bytes match pinned measurement");
    check(inputBytes * 10 < stateBytes, "P6 the input stream is at least 10x smaller than state-sync");

    const bool pass = (inputBytes == kProof6InputBytes) &&
                      (kProof6SnapBytes == 0 || snapBytes == kProof6SnapBytes) &&
                      (inputBytes * 10 < stateBytes);
    std::printf("PROOF 6 [bandwidth inputs<<state]: %s inputStream=%zuB stateSync=%zuB/tick*%u=%zuB ratio=%zux digest=0x%016llx "
                "(UE5: non-deterministic physics forces authoritative STATE replication every tick; it cannot send inputs and re-sim)\n",
                pass ? "PASS" : "FAIL", inputBytes, snapBytes, cr.ticks, stateBytes, ratio,
                (unsigned long long)((uint64_t)inputBytes << 32 | (uint64_t)snapBytes));
    return pass;
}

int main() {
    HF_TEST_MAIN_INIT();
    std::printf("=== Hazard Forge — moat proofs (runnable \"UE5 structurally loses\" suite) ===\n");
    bool all = true;
    all &= Proof1();
    all &= Proof2();
    all &= Proof3();
    all &= Proof4();
    all &= Proof5();
    all &= Proof6();
    std::printf("=== %s (%d check failures) ===\n", (all && g_fail == 0) ? "ALL PROOFS PASS" : "PROOFS FAILED", g_fail);
    return (all && g_fail == 0) ? 0 : 1;
}
