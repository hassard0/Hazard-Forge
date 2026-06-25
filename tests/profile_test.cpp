// Unit test for the deterministic profiler-capture event model (engine/profile/profile.h, Slice
// PROFILE-S1, issue #31 — the DETERMINISTIC SCRUB-FRIENDLY PROFILER CAPTURE beachhead). Pure CPU
// (hf_core), ASan-eligible like the other pure tests.
//
// SELF-CONTAINED: the test scaffolding (check() + HF_TEST_MAIN_INIT()) is copied from seq_test.cpp /
// obj_loader_test.cpp (NOT included) so this compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/profile_test.cpp` on the Mac — the cheap cross-platform
// proof.
//
// THE MOAT: a profiler measures TIME (non-deterministic), so this is NOT "a deterministic profiler".
// The capture's STRUCTURE is golden + deterministic; the TIMING overlay is stored separately and NEVER
// fed to StructuralDigest. The load-bearing proof (assertion 3): filling the timings with arbitrary
// nonzero values leaves StructuralDigest UNCHANGED. The golden is a PINNED FNV-1a-64 StructuralDigest
// value IN the test (NO image, NO render-bake — identical MSVC + Apple clang).

#include "profile/profile.h"

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

int main() {
    HF_TEST_MAIN_INIT();

    // ---- Build the showcase capture + its structural digest. ----------------------------------------
    const Capture showcase = MakeShowcaseCapture();
    const uint64_t digest = StructuralDigest(showcase);

    std::printf("profile-s1: showcase structural digest = 0x%016llx\n",
                static_cast<unsigned long long>(digest));

    // The pinned golden (computed on first run, hardcoded — the cross-platform / regression anchor).
    const uint64_t kPinnedDigest = 0xedc7791443141dfdull;  // PINNED on first run (MSVC == clang)

    // ---- (1) PINNED STRUCTURAL DIGEST — the cross-platform make-or-break (identical MSVC + clang). ---
    check(digest == kPinnedDigest,
          "profile-s1: StructuralDigest(MakeShowcaseCapture()) == pinned uint64 (cross-platform structural anchor)");

    // ---- (2) DETERMINISTIC — a second encode of the same capture is byte-identical (+ digest equal). -
    {
        const std::vector<uint8_t> enc1 = EncodeStructural(showcase);
        const std::vector<uint8_t> enc2 = EncodeStructural(showcase);
        check(enc1 == enc2 && StructuralDigest(showcase) == digest,
              "profile-s1: re-encoding the same capture is byte-identical (deterministic)");
    }

    // ---- (3) TIMING IS EXCLUDED (the load-bearing moat proof) — fill timings nonzero, digest UNCHANGED.
    {
        Capture timed = MakeShowcaseCapture();
        for (std::size_t i = 0; i < timed.timings.size(); ++i) {
            timed.timings[i].cpuNanos = static_cast<uint64_t>(i) * 1000ull + 7ull;
            timed.timings[i].gpuNanos = static_cast<uint64_t>(i) * 9ull;
        }
        check(StructuralDigest(timed) == kPinnedDigest,
              "profile-s1: TIMING IS EXCLUDED — filling timings with arbitrary nonzero leaves StructuralDigest UNCHANGED");
    }

    // ---- (4) INTERN STABLE — first-seen ids 0/1/2; same name -> same id; a new name -> a fresh id. ---
    {
        NameTable t;
        const char kFrame[]  = { 'F', 'r', 'a', 'm', 'e' };
        const char kShadow[] = { 'S', 'h', 'a', 'd', 'o', 'w' };
        const char kLit[]    = { 'L', 'i', 't' };
        const char kNew[]    = { 'N', 'e', 'w' };
        const uint32_t f = Intern(t, kFrame,  sizeof(kFrame));
        const uint32_t s = Intern(t, kShadow, sizeof(kShadow));
        const uint32_t l = Intern(t, kLit,    sizeof(kLit));
        const uint32_t l2 = Intern(t, kLit,   sizeof(kLit));   // re-intern "Lit" -> same id
        const uint32_t nw = Intern(t, kNew,   sizeof(kNew));   // a fresh name -> a fresh id
        check(f == 0u && s == 1u && l == 2u && l2 == 2u && nw == 3u && t.names.size() == 4u,
              "profile-s1: interning is first-seen-stable — the same name returns the same id; a new name a new id");
    }

    // ---- (5) STRUCTURE LOAD-BEARING — change one DrawCall's draw count -> a DIFFERENT digest. --------
    {
        Capture mutated = MakeShowcaseCapture();
        // events[2] is Draw(Shadow, 2): change its `a` (the draw count).
        bool found = false;
        for (std::size_t i = 0; i < mutated.events.size(); ++i) {
            if (mutated.events[i].kind == EvKind::DrawCall) { mutated.events[i].a += 1; found = true; break; }
        }
        check(found && StructuralDigest(mutated) != digest,
              "profile-s1: a changed structural field (a draw count) changes the digest (structure is load-bearing)");
    }

    // ---- (6) NAME LOAD-BEARING — intern one scope under a different name byte -> a DIFFERENT digest. --
    // Build a capture identical to the showcase EXCEPT the first scope is "Frame" not "Frame".
    {
        Capture diffName;
        const char kFrame2[] = { 'F', 'r', 'a', 'm', 'e', 'X' };   // a different name byte-string
        const char kShadow[] = { 'S', 'h', 'a', 'd', 'o', 'w' };
        const char kLit[]    = { 'L', 'i', 't' };
        const uint32_t frame  = Intern(diffName.names, kFrame2, sizeof(kFrame2));
        const uint32_t shadow = Intern(diffName.names, kShadow, sizeof(kShadow));
        const uint32_t lit    = Intern(diffName.names, kLit,    sizeof(kLit));
        EmitEnter(diffName, frame);
        EmitEnter(diffName, shadow);
        EmitDraw (diffName, shadow, 2);
        EmitExit (diffName, shadow);
        EmitEnter(diffName, lit);
        EmitDraw (diffName, lit, 5);
        EmitExit (diffName, lit);
        EmitExit (diffName, frame);
        check(StructuralDigest(diffName) != digest,
              "profile-s1: a different scope NAME changes the digest (interned names are load-bearing)");
    }

    if (g_fail == 0) { std::printf("profile_test: ALL PASS\n"); return 0; }
    std::printf("profile_test: %d FAIL\n", g_fail);
    return 1;
}
