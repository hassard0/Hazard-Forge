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
        replay::RecordSession<ToyA, InA>(kSeed, ToyA{}, ring, kTicks, StepA, DigestA);
    const std::vector<uint8_t> demoFileBytes = replay::EncodeDemo<ToyA, InA>(rec, serToyA, serRingA);

    // Print the live values so the pinned hashes are visible in test output.
    const uint64_t fileHash = net::DigestBytes(demoFileBytes.data(), demoFileBytes.size());
    std::printf("rp1-record: demo file size = %zu bytes\n", demoFileBytes.size());
    std::printf("rp1-record: DigestBytes(demoFileBytes) = 0x%016llx\n",
                static_cast<unsigned long long>(fileHash));
    std::printf("rp1-record: digestTrace.back() = 0x%016llx (ToyA final digest)\n",
                static_cast<unsigned long long>(rec.digestTrace.empty() ? 0u : rec.digestTrace.back()));

    // The pinned goldens (the regression anchor / cross-platform bar).
    const uint64_t kPinnedFileHash = 0x2add2e0b07ffcce4ull;  // PINNED: DigestBytes over the demo file bytes
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
        // Independently recompute the blob lengths to validate worldByteLen/inputByteLen.
        const uint32_t expectWorldLen = static_cast<uint32_t>(serToyA(rec.initial).size());
        const uint32_t expectInputLen = static_cast<uint32_t>(serRingA(rec.ring).size());
        const bool headerOk =
            magicOk && version == replay::kDemoVersion && seed == kSeed && tickCount == kTicks &&
            keyframeInterval == 0u && worldByteLen == expectWorldLen && inputByteLen == expectInputLen &&
            // The header is exactly 32 bytes; the two blobs follow it.
            demoFileBytes.size() == static_cast<std::size_t>(32u + worldByteLen + inputByteLen);
        check(headerOk,
              "rp1-record: header round-trips field-exact (magic/version/seed/tickCount/keyframeInterval/world+inputByteLen)");
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
                demo.header.worldByteLen == expectWorldLen && demo.header.inputByteLen == expectInputLen;
            check(headerOk,
                  "rp2-playback: DecodeDemo round-trips header field-exact (magic/version/seed/tickCount/world+inputByteLen)");
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

    if (g_fail == 0) std::printf("replay_test: ALL PASS\n");
    else             std::printf("replay_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
