// Unit test for the demo-file format + RECORDER (engine/replay/replay.h, Slice RP1, flagship #28
// REPLAY/DEMO beachhead). Pure CPU (hf_core), ASan-eligible like the other pure tests.
//
// SELF-CONTAINED: the ToyA toy world (ToyA/InA/StepA/DigestA) + makeRingA are COPIED verbatim from
// session_test.cpp (NOT included) so this test compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/replay_test.cpp` on the Mac — the cheap cross-platform
// proof (exactly like session_test / dsp_test). Everything is INTEGER, so the demo file bytes — and
// hence net::DigestBytes over them — are bit-identical run-to-run AND platform-to-platform (MSVC vs
// Apple clang). The golden is a PINNED FNV-1a-64 DigestBytes value over the demo file bytes.
//
// What this pins (the four RP1 assertions):
//   (a) net::DigestBytes(demoFileBytes) == a hard-pinned uint64_t (the cross-platform byte-layout proof);
//   (b) the header round-trips field-exact (magic/version/seed/tickCount/keyframeInterval/world+inputByteLen);
//   (c) digestTrace.size()==16 AND digestTrace.back() == the pinned ToyA final digest (== session_test's hToyA);
//   (d) re-encoding the same Recorder is byte-identical (deterministic, no clock/RNG/padding).

#include "replay/replay.h"
#include "net/session.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS: %s\n", what); }
}

// ---- ToyA: a scalar accumulator (COPIED verbatim from session_test.cpp). ---------------------------
struct ToyA { int64_t acc = 0; };
using InA = int32_t;
static void StepA(ToyA& w, const std::vector<InA>& inputs, uint32_t tick) {
    for (const InA in : inputs)
        w.acc += static_cast<int64_t>(in) * static_cast<int64_t>(tick + 1);
}
static uint64_t DigestA(const ToyA& w) { return net::DigestBytes(&w.acc, sizeof w.acc); }

// A fixed deterministic ToyA input ring (COPIED verbatim from session_test.cpp:60-69).
static net::InputRing<InA> makeRingA() {
    net::InputRing<InA> r;
    r.AddInput(0,  5);
    r.AddInput(1,  3); r.AddInput(1, -2);     // two inputs on tick 1 (insertion order matters)
    r.AddInput(3,  7);
    r.AddInput(7, 11); r.AddInput(7,  4); r.AddInput(7, -9);  // three on tick 7
    r.AddInput(10, 2);
    r.AddInput(15, 6);
    return r;
}

// ---- Hand-LE driver serializers (field-by-field, NO struct memcpy — the cross-platform key). -------
// ToyA: serialize `acc` (int64_t) as PutU64 of its two's-complement bit pattern reinterpreted as uint64_t.
static std::vector<uint8_t> serToyA(const ToyA& w) {
    std::vector<uint8_t> b;
    replay::PutU64(b, static_cast<uint64_t>(w.acc));
    return b;
}
// InputRing<InA>: tickCount(u32) = byTick.size(), then for each tick count(u32) then each input (int32_t)
// as PutU32 of its bits.
static std::vector<uint8_t> serRingA(const net::InputRing<InA>& ring) {
    std::vector<uint8_t> b;
    replay::PutU32(b, static_cast<uint32_t>(ring.byTick.size()));
    for (const std::vector<InA>& tickInputs : ring.byTick) {
        replay::PutU32(b, static_cast<uint32_t>(tickInputs.size()));
        for (const InA in : tickInputs)
            replay::PutU32(b, static_cast<uint32_t>(in));
    }
    return b;
}

// ---- RP2 inverse deserializers (the byte-exact inverse of serToyA / serRingA). ---------------------
// deserToyA: GetU64 -> reinterpret the bit pattern as int64_t acc (the inverse of serToyA's PutU64).
static ToyA deserToyA(const uint8_t* p, uint32_t len) {
    (void)len;
    ToyA w;
    w.acc = static_cast<int64_t>(replay::GetU64(p));
    return w;
}
// deserRingA: GetU32 tickCount, then per tick GetU32 count + each input GetU32 -> static_cast<int32_t>.
static net::InputRing<InA> deserRingA(const uint8_t* p, uint32_t len) {
    (void)len;
    net::InputRing<InA> r;
    std::size_t off = 0;
    const uint32_t tickCount = replay::GetU32(p + off); off += 4;
    r.byTick.resize(static_cast<std::size_t>(tickCount));
    for (uint32_t t = 0; t < tickCount; ++t) {
        const uint32_t count = replay::GetU32(p + off); off += 4;
        for (uint32_t i = 0; i < count; ++i) {
            const InA in = static_cast<int32_t>(replay::GetU32(p + off)); off += 4;
            r.byTick[static_cast<std::size_t>(t)].push_back(in);
        }
    }
    return r;
}

int main() {
    HF_TEST_MAIN_INIT();
    const uint32_t kTicks = 16;
    const uint32_t kSeed  = 0x5EED1234u;

    // Record THE fixed ToyA session, then encode it to demo file bytes.
    const net::InputRing<InA> ring = makeRingA();
    const replay::Recorder<ToyA, InA> rec =
        replay::RecordSession<ToyA, InA>(kSeed, ToyA{}, ring, kTicks, /*keyframeInterval=*/0u,
                                         StepA, DigestA, serToyA);
    const std::vector<uint8_t> demoFileBytes = replay::EncodeDemo<ToyA, InA>(rec, serToyA, serRingA);

    // Print the live values so the pinned hashes are visible in test output.
    const uint64_t fileHash = net::DigestBytes(demoFileBytes.data(), demoFileBytes.size());
    std::printf("rp1-record: demo file size = %zu bytes\n", demoFileBytes.size());
    std::printf("rp1-record: DigestBytes(demoFileBytes) = 0x%016llx\n",
                static_cast<unsigned long long>(fileHash));
    std::printf("rp1-record: digestTrace.back() = 0x%016llx (ToyA final digest)\n",
                static_cast<unsigned long long>(rec.digestTrace.empty() ? 0u : rec.digestTrace.back()));

    // The pinned goldens (the regression anchor / cross-platform bar).
    const uint64_t kPinnedFileHash = 0x92e4d491013137c4ull;  // RP3 RE-PIN: DigestBytes over the no-keyframe demo file (new 36-byte header)
    const uint64_t kPinnedToyA     = 0x6227bc7b4046d08aull;  // == session_test.cpp's hToyA (cross-check)

    // ---- (a) PINNED FILE HASH — the make-or-break cross-platform byte-layout proof. ----------------
    check(fileHash == kPinnedFileHash,
          "rp1-record: demo file bytes -> DigestBytes == pinned uint64");

    // Cross-check the pinned ToyA digest by an independent RunLockstep (the same number session_test pins).
    const uint64_t hToyA = net::RunLockstep<ToyA, InA>(ToyA{}, makeRingA(), kTicks, StepA, DigestA);
    check(hToyA == kPinnedToyA, "rp1-record: RunLockstep ToyA final digest == pinned (session_test cross-check)");

    // ---- (b) HEADER ROUND-TRIP — decode each header field back out and check it equals what went in. -
    {
        const uint8_t* p = demoFileBytes.data();
        bool magicOk = true;
        for (int i = 0; i < 8; ++i)
            if (static_cast<char>(p[i]) != replay::kDemoMagic[i]) magicOk = false;
        const uint32_t version          = replay::GetU32(p + 8);
        const uint32_t seed             = replay::GetU32(p + 12);
        const uint32_t tickCount        = replay::GetU32(p + 16);
        const uint32_t keyframeInterval = replay::GetU32(p + 20);
        const uint32_t worldByteLen     = replay::GetU32(p + 24);
        const uint32_t inputByteLen     = replay::GetU32(p + 28);
        const uint32_t keyframeByteLen  = replay::GetU32(p + 32);
        // Independently recompute the blob lengths to validate worldByteLen/inputByteLen.
        const uint32_t expectWorldLen = static_cast<uint32_t>(serToyA(rec.initial).size());
        const uint32_t expectInputLen = static_cast<uint32_t>(serRingA(rec.ring).size());
        const bool headerOk =
            magicOk && version == replay::kDemoVersion && seed == kSeed && tickCount == kTicks &&
            keyframeInterval == 0u && worldByteLen == expectWorldLen && inputByteLen == expectInputLen &&
            keyframeByteLen == 0u &&
            // The header is now exactly 36 bytes (no keyframe section here); the two blobs follow it.
            demoFileBytes.size() == static_cast<std::size_t>(36u + worldByteLen + inputByteLen);
        check(headerOk,
              "rp1-record: header round-trips field-exact (magic/version/seed/tickCount/keyframeInterval/world+input+keyframeByteLen)");
    }

    // ---- (c) DIGEST-TRACE PROVENANCE — length 16 and last == the pinned ToyA final digest. ----------
    check(rec.digestTrace.size() == static_cast<std::size_t>(kTicks) &&
              !rec.digestTrace.empty() && rec.digestTrace.back() == kPinnedToyA,
          "rp1-record: digestTrace.size() == 16 AND digestTrace.back() == pinned ToyA final digest");

    // ---- (d) RE-ENCODE DETERMINISM — encoding the same Recorder twice is byte-identical. ------------
    {
        const std::vector<uint8_t> again = replay::EncodeDemo<ToyA, InA>(rec, serToyA, serRingA);
        check(again == demoFileBytes, "rp1-record: re-encoding the same Recorder is byte-identical");
    }

    // ================================ RP2: PLAYBACK assertions =====================================
    // Decode the SAME demoFileBytes from RP1, then Replay it — proving byte-exact re-derivation.
    {
        const replay::Demo<ToyA, InA> demo =
            replay::DecodeDemo<ToyA, InA>(demoFileBytes, deserToyA, deserRingA);
        const replay::ReplayResult<ToyA, InA> rr = replay::Replay<ToyA, InA>(demo, StepA, DigestA);

        std::printf("rp2-playback: replay final digest = 0x%016llx (== live RunLockstep == pinned ToyA final)\n",
                    static_cast<unsigned long long>(rr.finalDigest));

        // ---- (1) HEADER ROUND-TRIP — decoded Demo.header fields equal what was recorded. ----------
        {
            bool magicOk = true;
            for (int i = 0; i < 8; ++i)
                if (demo.header.magic[i] != replay::kDemoMagic[i]) magicOk = false;
            const uint32_t expectWorldLen = static_cast<uint32_t>(serToyA(rec.initial).size());
            const uint32_t expectInputLen = static_cast<uint32_t>(serRingA(rec.ring).size());
            const bool headerOk =
                magicOk && demo.header.version == replay::kDemoVersion && demo.header.seed == kSeed &&
                demo.header.tickCount == kTicks && demo.header.keyframeInterval == 0u &&
                demo.header.worldByteLen == expectWorldLen && demo.header.inputByteLen == expectInputLen &&
                demo.header.keyframeByteLen == 0u && demo.keyframes.empty();
            check(headerOk,
                  "rp2-playback: DecodeDemo round-trips header field-exact (magic/version/seed/tickCount/world+input+keyframeByteLen)");
        }

        // ---- (2) WORLD + RING ROUND-TRIP — decoded initial == ToyA{} (acc==0) AND ring byte-exact. -
        {
            const std::vector<uint8_t> decodedRingBytes  = serRingA(demo.ring);
            const std::vector<uint8_t> originalRingBytes = serRingA(ring);
            check(demo.initial.acc == 0 && decodedRingBytes == originalRingBytes,
                  "rp2-playback: decoded initial world == original initial (ToyA acc == 0) AND decoded ring re-encodes byte-identical");
        }

        // ---- (3) REPLAY == LIVE == PINNED — the make-or-break re-derivation. ----------------------
        {
            const uint64_t live = net::RunLockstep<ToyA, InA>(ToyA{}, ring, kTicks, StepA, DigestA);
            check(rr.finalDigest == live && rr.finalDigest == kPinnedToyA,
                  "rp2-playback: Replay final digest == live net::RunLockstep == pinned 0x6227bc7b4046d08a");
        }

        // ---- (4) PER-TICK INTEGRITY — replay trace == recorded trace == fresh DigestTrace (every tick). -
        {
            const std::vector<uint64_t> fresh =
                net::DigestTrace<ToyA, InA>(ToyA{}, ring, kTicks, StepA, DigestA);
            bool traceOk = rr.trace.size() == static_cast<std::size_t>(kTicks) &&
                           rec.digestTrace.size() == static_cast<std::size_t>(kTicks) &&
                           fresh.size() == static_cast<std::size_t>(kTicks);
            for (std::size_t t = 0; traceOk && t < rr.trace.size(); ++t)
                if (rr.trace[t] != rec.digestTrace[t] || rr.trace[t] != fresh[t]) traceOk = false;
            check(traceOk,
                  "rp2-playback: replay per-tick trace == recorded digestTrace == fresh net::DigestTrace (every tick, integrity)");
        }

        // ---- (5) FULL BYTE ROUND-TRIP — rebuild a Recorder from the decoded Demo, re-encode == bytes. -
        {
            replay::Recorder<ToyA, InA> decodedRec;
            decodedRec.seed        = demo.header.seed;
            decodedRec.initial     = demo.initial;
            decodedRec.ring        = demo.ring;
            decodedRec.tickCount   = demo.header.tickCount;
            decodedRec.digestTrace =
                net::DigestTrace<ToyA, InA>(demo.initial, demo.ring, demo.header.tickCount, StepA, DigestA);
            const std::vector<uint8_t> reBytes =
                replay::EncodeDemo<ToyA, InA>(decodedRec, serToyA, serRingA);
            check(reBytes == demoFileBytes,
                  "rp2-playback: Decode(Encode(rec)) re-encoded bytes == demoFileBytes (full round-trip, byte-exact)");
        }
    }

    // ================================ RP3: KEYFRAME assertions =====================================
    // Record the SAME ToyA session at two keyframe intervals (4 and 8); prove each keyframe-at-tick-T is a
    // BIT-IDENTICAL mid-session restore point (== live RunLockstep-to-T), and pin distinct file hashes.
    {
        // The pinned RP3 goldens (re-pin from the printed values on first run).
        const uint64_t kPinnedKf4Hash = 0xf2ccf29305652a39ull;  // RP3: DigestBytes over the interval-4 demo file
        const uint64_t kPinnedKf8Hash = 0xe7fb940bd0c3cc41ull;  // RP3: DigestBytes over the interval-8 demo file

        // ---- interval K1 = 4 → keyframes at ticks 0,4,8,12 (ceil-free: 16/4 == 4). -------------------
        const replay::Recorder<ToyA, InA> recKf4 =
            replay::RecordSession<ToyA, InA>(kSeed, ToyA{}, ring, kTicks, /*keyframeInterval=*/4u,
                                             StepA, DigestA, serToyA);
        const std::vector<uint8_t> demoKf4 = replay::EncodeDemo<ToyA, InA>(recKf4, serToyA, serRingA);
        const uint64_t kf4Hash = net::DigestBytes(demoKf4.data(), demoKf4.size());

        // ---- interval K2 = 8 → keyframes at ticks 0,8 (16/8 == 2). ------------------------------------
        const replay::Recorder<ToyA, InA> recKf8 =
            replay::RecordSession<ToyA, InA>(kSeed, ToyA{}, ring, kTicks, /*keyframeInterval=*/8u,
                                             StepA, DigestA, serToyA);
        const std::vector<uint8_t> demoKf8 = replay::EncodeDemo<ToyA, InA>(recKf8, serToyA, serRingA);
        const uint64_t kf8Hash = net::DigestBytes(demoKf8.data(), demoKf8.size());

        std::printf("rp3-keyframe: demoKf4 size = %zu bytes, keyframes = %zu\n",
                    demoKf4.size(), recKf4.keyframes.size());
        std::printf("rp3-keyframe: DigestBytes(demoKf4) = 0x%016llx (pinned, distinct from no-keyframe)\n",
                    static_cast<unsigned long long>(kf4Hash));
        std::printf("rp3-keyframe: demoKf8 size = %zu bytes, keyframes = %zu\n",
                    demoKf8.size(), recKf8.keyframes.size());
        std::printf("rp3-keyframe: DigestBytes(demoKf8) = 0x%016llx (pinned, distinct + smaller)\n",
                    static_cast<unsigned long long>(kf8Hash));

        // ---- (a) pinned distinct file hashes (the tradeoff knob is real + deterministic). -------------
        check(kf4Hash == kPinnedKf4Hash, "rp3-keyframe: DigestBytes(demoKf4) == pinned uint64 (interval 4)");
        check(kf8Hash == kPinnedKf8Hash, "rp3-keyframe: DigestBytes(demoKf8) == pinned uint64 (interval 8)");
        check(kf4Hash != kf8Hash && kf4Hash != fileHash && kf8Hash != fileHash,
              "rp3-keyframe: Kf4 / Kf8 / no-keyframe hashes are all distinct (interval is a real knob)");

        // ---- (b) keyframeInterval header field round-trips + keyframeCount == ceil(16/K). -------------
        const replay::Demo<ToyA, InA> dKf4 =
            replay::DecodeDemo<ToyA, InA>(demoKf4, deserToyA, deserRingA);
        const replay::Demo<ToyA, InA> dKf8 =
            replay::DecodeDemo<ToyA, InA>(demoKf8, deserToyA, deserRingA);
        check(dKf4.header.keyframeInterval == 4u && recKf4.keyframes.size() == 4u &&
                  dKf4.keyframes.size() == 4u &&
                  dKf4.keyframes[0].tick == 0u && dKf4.keyframes[1].tick == 4u &&
                  dKf4.keyframes[2].tick == 8u && dKf4.keyframes[3].tick == 12u,
              "rp3-keyframe: keyframeInterval==4 header round-trips; keyframeCount == 4 (ticks 0,4,8,12)");
        check(dKf8.header.keyframeInterval == 8u && recKf8.keyframes.size() == 2u &&
                  dKf8.keyframes.size() == 2u &&
                  dKf8.keyframes[0].tick == 0u && dKf8.keyframes[1].tick == 8u,
              "rp3-keyframe: keyframeInterval==8 header round-trips; keyframeCount == 2 (ticks 0,8)");

        // ---- (c) THE make-or-break: each keyframe-at-tick-T world digest == live world digest at T. ---
        // Live reference: net::RunLockstep<ToyA,InA>(ToyA{}, ring, T, ...) (T==0 → DigestA(ToyA{})).
        {
            bool kf4Ok = true;
            for (const auto& dk : dKf4.keyframes) {
                const uint64_t live = (dk.tick == 0u)
                    ? DigestA(ToyA{})
                    : net::RunLockstep<ToyA, InA>(ToyA{}, ring, dk.tick, StepA, DigestA);
                const uint64_t kfDig = DigestA(dk.world);
                std::printf("rp3-keyframe: tick %2u  keyframe digest = 0x%016llx  live = 0x%016llx\n",
                            dk.tick, static_cast<unsigned long long>(kfDig),
                            static_cast<unsigned long long>(live));
                if (kfDig != live) kf4Ok = false;
            }
            check(kf4Ok,
                  "rp3-keyframe: each keyframe-at-tick-T world digest == live RunLockstep-to-T (bit-identical restore point)");
        }
        {
            bool kf8Ok = true;
            for (const auto& dk : dKf8.keyframes) {
                const uint64_t live = (dk.tick == 0u)
                    ? DigestA(ToyA{})
                    : net::RunLockstep<ToyA, InA>(ToyA{}, ring, dk.tick, StepA, DigestA);
                if (DigestA(dk.world) != live) kf8Ok = false;
            }
            check(kf8Ok, "rp3-keyframe: interval-8 keyframes also == live RunLockstep-to-T (0,8)");
        }

        // ---- (d) decoded keyframes round-trip (tick + world byte-state exact via re-serialize). -------
        {
            bool rtOk = recKf4.keyframes.size() == dKf4.keyframes.size();
            for (std::size_t i = 0; rtOk && i < dKf4.keyframes.size(); ++i) {
                if (dKf4.keyframes[i].tick != recKf4.keyframes[i].tick) rtOk = false;
                else if (serToyA(dKf4.keyframes[i].world) != recKf4.keyframes[i].worldBytes) rtOk = false;
            }
            check(rtOk, "rp3-keyframe: decode(demoKf4) keyframes round-trip (tick + world byte-state exact)");
        }

        // ---- (e) keyframes don't change playback — Replay final digest STILL == pinned ToyA digest. ---
        {
            const replay::ReplayResult<ToyA, InA> rr4 = replay::Replay<ToyA, InA>(dKf4, StepA, DigestA);
            const replay::ReplayResult<ToyA, InA> rr8 = replay::Replay<ToyA, InA>(dKf8, StepA, DigestA);
            check(rr4.finalDigest == kPinnedToyA && rr8.finalDigest == kPinnedToyA,
                  "rp3-keyframe: Replay(demoKf4)/Replay(demoKf8) final digest still == 0x6227bc7b4046d08a");
        }
    }

    // ================================ RP4: SEEK assertions =========================================
    // SEEK(N) = restore the nearest keyframe at-or-before N and replay the tail [keyframeTick, N) forward
    // (net::CatchUp). The make-or-break: Seek(demo, N) is BIT-IDENTICAL to the live session at N (exact
    // re-derivation, not interpolation), and the keyframe interval is purely a COST knob (invisible to the
    // result). Live reference at tick N: net::RunLockstep<ToyA,InA>(ToyA{}, ring, N, StepA, DigestA).
    {
        // Re-record the interval-4 / interval-8 demos (RP3 built these inside its own scope) and the
        // keyframeless no-keyframe demo (decode the RP1 demoFileBytes, keyframeInterval 0).
        const replay::Recorder<ToyA, InA> recKf4 =
            replay::RecordSession<ToyA, InA>(kSeed, ToyA{}, ring, kTicks, /*keyframeInterval=*/4u,
                                             StepA, DigestA, serToyA);
        const std::vector<uint8_t> demoKf4Bytes = replay::EncodeDemo<ToyA, InA>(recKf4, serToyA, serRingA);
        const replay::Demo<ToyA, InA> dKf4 =
            replay::DecodeDemo<ToyA, InA>(demoKf4Bytes, deserToyA, deserRingA);

        const replay::Recorder<ToyA, InA> recKf8 =
            replay::RecordSession<ToyA, InA>(kSeed, ToyA{}, ring, kTicks, /*keyframeInterval=*/8u,
                                             StepA, DigestA, serToyA);
        const std::vector<uint8_t> demoKf8Bytes = replay::EncodeDemo<ToyA, InA>(recKf8, serToyA, serRingA);
        const replay::Demo<ToyA, InA> dKf8 =
            replay::DecodeDemo<ToyA, InA>(demoKf8Bytes, deserToyA, deserRingA);

        const replay::Demo<ToyA, InA> dNoKf =
            replay::DecodeDemo<ToyA, InA>(demoFileBytes, deserToyA, deserRingA);

        // The live reference at tick N (N==0 -> DigestA(ToyA{}); the from-scratch lockstep otherwise).
        auto liveAt = [&](uint32_t n) -> uint64_t {
            return (n == 0u) ? DigestA(ToyA{})
                             : net::RunLockstep<ToyA, InA>(ToyA{}, ring, n, StepA, DigestA);
        };

        // A printed sample seek line (N=10: keyframeTick=8, replayed=2).
        {
            const replay::SeekResult<ToyA, InA> s10 = replay::Seek<ToyA, InA>(dKf4, 10u, StepA, DigestA);
            std::printf("rp4-seek: seek N=10 -> keyframeTick=%u replayed=%u digest=0x%016llx (== live RunLockstep(10) 0x%016llx)\n",
                        s10.keyframeTick, s10.replayedTicks,
                        static_cast<unsigned long long>(s10.digest),
                        static_cast<unsigned long long>(liveAt(10u)));
        }

        // ---- (1) SEEK == LIVE (make-or-break): Seek(demoKf4, N) digest == live RunLockstep(N). --------
        {
            const uint32_t Ns[] = {0u, 4u, 7u, 8u, 12u, 15u, 16u};
            bool ok = true;
            for (const uint32_t N : Ns) {
                const replay::SeekResult<ToyA, InA> s = replay::Seek<ToyA, InA>(dKf4, N, StepA, DigestA);
                if (s.digest != liveAt(N)) ok = false;
            }
            // N=16 must equal the pinned final digest specifically.
            const replay::SeekResult<ToyA, InA> s16 = replay::Seek<ToyA, InA>(dKf4, 16u, StepA, DigestA);
            if (s16.digest != kPinnedToyA) ok = false;
            check(ok,
                  "rp4-seek: Seek(demoKf4, N) digest == live RunLockstep(N) for N in {0,4,7,8,12,15,16} (bit-identical)");
        }

        // ---- (2) NEAREST KEYFRAME bookkeeping — keyframeTick/replayedTicks correct per N. -------------
        {
            const replay::SeekResult<ToyA, InA> s10 = replay::Seek<ToyA, InA>(dKf4, 10u, StepA, DigestA);
            const replay::SeekResult<ToyA, InA> s7  = replay::Seek<ToyA, InA>(dKf4,  7u, StepA, DigestA);
            const replay::SeekResult<ToyA, InA> s4  = replay::Seek<ToyA, InA>(dKf4,  4u, StepA, DigestA);
            const replay::SeekResult<ToyA, InA> s15 = replay::Seek<ToyA, InA>(dKf4, 15u, StepA, DigestA);
            const bool ok =
                s10.keyframeTick == 8u  && s10.replayedTicks == 2u &&
                s7.keyframeTick  == 4u  && s7.replayedTicks  == 3u &&
                s4.keyframeTick  == 4u  && s4.replayedTicks  == 0u &&
                s15.keyframeTick == 12u && s15.replayedTicks == 3u;
            check(ok,
                  "rp4-seek: Seek restores the NEAREST keyframe <= N (keyframeTick/replayedTicks correct per N)");
        }

        // ---- (3) ON-KEYFRAME ZERO REPLAY — N in {0,4,8,12} restored exactly (replayedTicks==0). -------
        {
            const uint32_t Ns[] = {0u, 4u, 8u, 12u};
            bool ok = true;
            for (const uint32_t N : Ns) {
                const replay::SeekResult<ToyA, InA> s = replay::Seek<ToyA, InA>(dKf4, N, StepA, DigestA);
                if (s.replayedTicks != 0u || s.keyframeTick != N) ok = false;
            }
            check(ok,
                  "rp4-seek: on-keyframe seek (N in {0,4,8,12}) replays ZERO ticks (keyframeTick==N, replayedTicks==0)");
        }

        // ---- (4) MAX-TAIL — N in {3,7,11} (just before the next keyframe) replays N-(N/4)*4 == 3. -----
        {
            const uint32_t Ns[] = {3u, 7u, 11u};
            bool ok = true;
            for (const uint32_t N : Ns) {
                const replay::SeekResult<ToyA, InA> s = replay::Seek<ToyA, InA>(dKf4, N, StepA, DigestA);
                if (s.replayedTicks != N - (N / 4u) * 4u || s.replayedTicks != 3u) ok = false;
            }
            check(ok,
                  "rp4-seek: just-before-next-keyframe seek (N in {3,7,11}) replays the MAX tail (replayedTicks==N-floor(N/4)*4)");
        }

        // ---- (5) KEYFRAMELESS FALLBACK — no-keyframe demo seeks from tick 0 (full replay), == live. ---
        {
            const uint32_t Ns[] = {0u, 4u, 7u, 8u, 12u, 15u, 16u};
            bool ok = dNoKf.keyframes.empty();
            for (const uint32_t N : Ns) {
                const replay::SeekResult<ToyA, InA> s = replay::Seek<ToyA, InA>(dNoKf, N, StepA, DigestA);
                if (s.digest != liveAt(N) || s.keyframeTick != 0u || s.replayedTicks != N) ok = false;
            }
            check(ok,
                  "rp4-seek: keyframeless demo (interval 0) seeks from tick 0 (full replay) and still == live RunLockstep(N)");
        }

        // ---- (6) INTERVAL-INVISIBLE — Seek(demoKf4,N) == Seek(demoKf8,N) == live for the same N. ------
        {
            const uint32_t Ns[] = {0u, 4u, 7u, 8u, 12u, 15u, 16u};
            bool ok = true;
            for (const uint32_t N : Ns) {
                const replay::SeekResult<ToyA, InA> s4 = replay::Seek<ToyA, InA>(dKf4, N, StepA, DigestA);
                const replay::SeekResult<ToyA, InA> s8 = replay::Seek<ToyA, InA>(dKf8, N, StepA, DigestA);
                if (s4.digest != s8.digest || s4.digest != liveAt(N)) ok = false;
            }
            check(ok,
                  "rp4-seek: Seek(demoKf4, N).digest == Seek(demoKf8, N).digest == live (keyframe interval is invisible to the result)");
        }
    }

    if (g_fail == 0) std::printf("replay_test: ALL PASS\n");
    else             std::printf("replay_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
