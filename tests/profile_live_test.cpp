// Unit test for the LIVE ScopedZone capstone (engine/profile/profile_live.h, Slice PROFILE-S6, issue #31
// — the FLAGSHIP #31 capstone). A ScopedZone RAII helper measures REAL wall-clock time into the timing
// OVERLAY while building structure via the SAME S1 emitters — proving the structure-vs-timing split holds
// with REAL measured timing (structure golden + deterministic; timing real + non-deterministic). The
// clock crossing (<chrono>) is ISOLATED in profile_live.h so profile.h stays <chrono>-free.
//
// SELF-CONTAINED: the test scaffolding (check() + HF_TEST_MAIN_INIT()) is copied from profile_test.cpp /
// seq_test.cpp (NOT included) so this compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/profile_live_test.cpp` — the cheap cross-platform proof.
//
// THE CAPSTONE PROOF: a live ScopedZone capture's StructuralDigest is BYTE-IDENTICAL to the scripted
// MakeShowcaseCapture digest (0xedc7791443141dfd) EVEN with real measured timing in the overlay, and TWO
// runs with DIFFERENT work yield the IDENTICAL structural digest (the split holds with measured timing).
// The STRUCTURAL assertions are deterministic goldens; the timing assertions are >0 liveness checks (NO
// assertion on timing VALUES / differences / ordering — non-flaky discipline).

#include "profile/profile_live.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::profile;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

// Find the "Lit" zone's measured cpuNanos in a capture (the Enter event whose nameId == lit, id 2).
static uint64_t LitZoneNanos(const Capture& c) {
    // "Lit" is interned id 2 in BuildLiveShowcase (Frame/Shadow/Lit -> 0/1/2). The duration lives on the
    // ENTER marker (the ScopedZone convention).
    for (std::size_t i = 0; i < c.events.size(); ++i) {
        if (c.events[i].kind == EvKind::ScopeEnter && c.events[i].nameId == 2u) {
            return c.timings[i].cpuNanos;
        }
    }
    return 0;
}

int main() {
    HF_TEST_MAIN_INIT();

    const uint64_t kPinnedDigest = 0xedc7791443141dfdull;  // PINNED (S1 MakeShowcaseCapture, MSVC == clang)

    // Two runs with DIFFERENT work → different real durations, IDENTICAL structure. work >= 1e7 so the
    // busy loop reliably measures > 0 on steady_clock.
    const uint64_t work1 = 10000000ull;   // 1e7
    const uint64_t work2 = 30000000ull;   // 3e7 (a large margin so timing visibly differs — printed only)

    const NameTable seed;  // an empty seed (the live-engine seam; canonical re-intern is authoritative)

    const Capture liveA = BuildLiveShowcase(seed, work1);
    const Capture liveB = BuildLiveShowcase(seed, work2);
    const Capture scripted = MakeShowcaseCapture();

    const uint64_t liveDigestA = StructuralDigest(liveA);
    const uint64_t liveDigestB = StructuralDigest(liveB);
    const uint64_t runANanos   = LitZoneNanos(liveA);
    const uint64_t runBNanos   = LitZoneNanos(liveB);

    std::printf("profile-s6: live structural digest = 0x%016llx  (run A nanos = %llu, run B nanos = %llu)\n",
                static_cast<unsigned long long>(liveDigestA),
                static_cast<unsigned long long>(runANanos),
                static_cast<unsigned long long>(runBNanos));

    // ---- (1) LIVE STRUCTURE == SCRIPTED — the RAII path produces the byte-identical structure. -------
    check(liveDigestA == StructuralDigest(scripted) && liveDigestA == kPinnedDigest,
          "profile-s6: a live ScopedZone capture's StructuralDigest == the scripted MakeShowcaseCapture digest (0xedc7791443141dfd)");

    // ---- (2) RAII PROVENANCE — BuildLiveShowcase events == MakeShowcaseCapture events field-for-field.
    {
        bool ok = (liveA.events.size() == scripted.events.size());
        for (std::size_t i = 0; ok && i < liveA.events.size(); ++i) {
            ok = ok && (liveA.events[i].kind   == scripted.events[i].kind)
                    && (liveA.events[i].nameId == scripted.events[i].nameId)
                    && (liveA.events[i].a      == scripted.events[i].a)
                    && (liveA.events[i].b      == scripted.events[i].b);
        }
        check(ok,
              "profile-s6: ScopedZone built the same structural events as the scripted EmitEnter/EmitExit (RAII provenance)");
    }

    // ---- (3) TIMING POPULATED — after a substantial busy loop, a measured zone's cpuNanos > 0. -------
    check(runANanos > 0ull,
          "profile-s6: real timing is populated — a measured zone's cpuNanos > 0 after actual work");

    // ---- (4) STRUCTURE GOLDEN UNDER REAL TIMING — two DIFFERENT-work runs yield the IDENTICAL digest.
    check(liveDigestA == liveDigestB
              && liveDigestA == kPinnedDigest
              && liveDigestB == kPinnedDigest,
          "profile-s6: STRUCTURE IS GOLDEN UNDER REAL TIMING — two runs with DIFFERENT work yield the IDENTICAL structural digest");

    // ---- (5) TIMINGS ARE REAL (informational, non-flaky) — each run's Lit nanos > 0 (NO diff/order). -
    check(runANanos > 0ull && runBNanos > 0ull,
          "profile-s6: ...and the two runs' measured timings are recorded (the non-deterministic overlay, printed)");

    if (g_fail == 0) { std::printf("profile_live_test: ALL PASS\n"); return 0; }
    std::printf("profile_live_test: %d FAIL\n", g_fail);
    return 1;
}
