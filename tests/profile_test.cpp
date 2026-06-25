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

    // ================================ SLICE PROFILE-S2 — scope/zone TREE ==============================
    // S2 reconstructs the hierarchical scope tree from the flat Enter/Exit stream and aggregates draw
    // counts bottom-up. The tree has its own PINNED digest; an unbalanced stream resolves deterministically.

    // ---- Build the showcase tree + its tree digest. -------------------------------------------------
    const ScopeTree tree = BuildScopeTree(showcase);
    const uint64_t treeDigest = DigestTree(tree);
    std::printf("profile-s2: scope-tree digest = 0x%016llx  (%zu nodes)\n",
                static_cast<unsigned long long>(treeDigest), tree.nodes.size());

    // ---- (S2-1) S1 INVARIANT — the structural digest is UNCHANGED by S2 (append-only). --------------
    check(StructuralDigest(MakeShowcaseCapture()) == 0xedc7791443141dfdull,
          "profile-s2: S1 structural digest 0xedc7791443141dfd UNCHANGED (append-only)");

    // ---- (S2-2) PINNED TREE DIGEST — byte-stable cross-platform (identical MSVC + clang). -----------
    const uint64_t kPinnedTreeDigest = 0xb41eb67a1d13443eull;  // PINNED on first run (MSVC == clang)
    check(treeDigest == kPinnedTreeDigest,
          "profile-s2: DigestTree(BuildScopeTree(showcase)) == pinned uint64 (zone tree byte-stable cross-platform)");

    // ---- (S2-3) SHAPE — root -> Frame -> {Shadow, Lit} in emission order; self draws 2 and 5. -------
    {
        bool ok = tree.balanced;
        // root (index 0) has exactly one child: Frame.
        const ScopeNode& root = tree.nodes[0];
        ok = ok && (root.firstChild != kNoNode) && (tree.nodes[root.firstChild].nextSibling == kNoNode);
        const ScopeNode& frame = tree.nodes[root.firstChild];
        // Frame has exactly two children: Shadow then Lit.
        ok = ok && (frame.firstChild != kNoNode);
        const uint32_t c0 = frame.firstChild;
        ok = ok && (tree.nodes[c0].nextSibling != kNoNode);
        const uint32_t c1 = tree.nodes[c0].nextSibling;
        ok = ok && (tree.nodes[c1].nextSibling == kNoNode);
        // Shadow.selfDrawCount == 2, Lit.selfDrawCount == 5 (emission order).
        ok = ok && (tree.nodes[c0].selfDrawCount == 2u) && (tree.nodes[c1].selfDrawCount == 5u);
        check(ok,
              "profile-s2: the tree is balanced and has the expected shape (root -> Frame -> {Shadow, Lit})");
    }

    // ---- (S2-4) AGGREGATION EXACT — root & Frame subtreeDrawCount == total draws (7). ---------------
    {
        const ScopeNode& root  = tree.nodes[0];
        const ScopeNode& frame = tree.nodes[root.firstChild];
        check(root.subtreeDrawCount == 7u && frame.subtreeDrawCount == 7u,
              "profile-s2: subtree draw aggregation is exact -- root subtreeDrawCount == total draws (7)");
    }

    // ---- (S2-5) HIERARCHY LOAD-BEARING — nest the draw one level deeper -> a DIFFERENT tree digest. -
    {
        // Frame{ Shadow{Draw2} Lit{ Cull{Draw5} } } — same total draws (7), DIFFERENT shape.
        Capture deeper;
        const char kFrame[]  = { 'F', 'r', 'a', 'm', 'e' };
        const char kShadow[] = { 'S', 'h', 'a', 'd', 'o', 'w' };
        const char kLit[]    = { 'L', 'i', 't' };
        const char kCull[]   = { 'C', 'u', 'l', 'l' };
        const uint32_t frame  = Intern(deeper.names, kFrame,  sizeof(kFrame));
        const uint32_t shadow = Intern(deeper.names, kShadow, sizeof(kShadow));
        const uint32_t lit    = Intern(deeper.names, kLit,    sizeof(kLit));
        const uint32_t cull   = Intern(deeper.names, kCull,   sizeof(kCull));
        EmitEnter(deeper, frame);
        EmitEnter(deeper, shadow);
        EmitDraw (deeper, shadow, 2);
        EmitExit (deeper, shadow);
        EmitEnter(deeper, lit);
        EmitEnter(deeper, cull);
        EmitDraw (deeper, cull, 5);
        EmitExit (deeper, cull);
        EmitExit (deeper, lit);
        EmitExit (deeper, frame);
        const ScopeTree deeperTree = BuildScopeTree(deeper);
        check(DigestTree(deeperTree) != treeDigest && deeperTree.nodes[0].subtreeDrawCount == 7u,
              "profile-s2: a deeper-nested scope changes the tree digest (hierarchy is load-bearing)");
    }

    // ---- (S2-6) UNBALANCED DETERMINISTIC — a missing Exit -> balanced==false + a stable canonical tree.
    const uint64_t kPinnedUnbalancedDigest = 0xd37725f33c5a5a2full;  // PINNED on first run (MSVC == clang)
    {
        // Enter A; Draw 1;  (NO Exit A) -> open scope never closed.
        Capture unb;
        const char kA[] = { 'A' };
        const uint32_t a = Intern(unb.names, kA, sizeof(kA));
        EmitEnter(unb, a);
        EmitDraw (unb, a, 1);
        const ScopeTree ut1 = BuildScopeTree(unb);
        const ScopeTree ut2 = BuildScopeTree(unb);   // a second build of the same stream
        const uint64_t ud1 = DigestTree(ut1);
        const uint64_t ud2 = DigestTree(ut2);
        std::printf("profile-s2: unbalanced-tree digest = 0x%016llx\n",
                    static_cast<unsigned long long>(ud1));
        check(ut1.balanced == false && ud1 == ud2 && ud1 == kPinnedUnbalancedDigest,
              "profile-s2: an unbalanced stream (a missing Exit) -> balanced==false + a deterministic canonical tree");
    }

    // ---- (S2-7) TIMING STILL EXCLUDED — filling timings nonzero leaves BOTH digests unchanged. ------
    {
        Capture timed = MakeShowcaseCapture();
        for (std::size_t i = 0; i < timed.timings.size(); ++i) {
            timed.timings[i].cpuNanos = static_cast<uint64_t>(i) * 1000ull + 7ull;
            timed.timings[i].gpuNanos = static_cast<uint64_t>(i) * 9ull;
        }
        const bool structuralSame = (StructuralDigest(timed) == 0xedc7791443141dfdull);
        const bool treeSame       = (DigestTree(BuildScopeTree(timed)) == treeDigest);
        check(structuralSame && treeSame,
              "profile-s2: TIMING STILL EXCLUDED -- filling timings nonzero leaves BOTH the structural AND tree digest unchanged");
    }

    // ================================ SLICE PROFILE-S3 — frame boundaries + TIMELINE =================
    // S3 brackets each frame with FrameBegin/FrameEnd markers and builds a per-frame timeline index where
    // every frame carries its OWN structural sub-digest (events-only, position-independent — the seek prop).

    // ---- Build the 4-frame timeline + its whole-timeline digest. ------------------------------------
    const Capture timeline = MakeTimelineCapture();
    const std::vector<FrameIndex> frames = BuildFrameIndex(timeline);
    const uint64_t timelineDigest = DigestTimeline(frames);
    std::printf("profile-s3: timeline digest = 0x%016llx  (%zu frames)\n",
                static_cast<unsigned long long>(timelineDigest), frames.size());
    for (std::size_t i = 0; i < frames.size(); ++i) {
        std::printf("profile-s3:   frame %u  events[%u..%u]  structuralDigest = 0x%016llx\n",
                    frames[i].frameNumber, frames[i].firstEvent,
                    frames[i].firstEvent + frames[i].eventCount,
                    static_cast<unsigned long long>(frames[i].structuralDigest));
    }

    const uint64_t kPinnedTimelineDigest = 0xc68ff46e1ab25f37ull;  // PINNED on first run (MSVC == clang)

    // ---- (S3-1) PRIOR INVARIANT — S1 0xedc7791443141dfd and S2 0xb41eb67a1d13443e both UNCHANGED. ----
    {
        const bool s1Same = (StructuralDigest(MakeShowcaseCapture()) == 0xedc7791443141dfdull);
        const bool s2Same = (DigestTree(BuildScopeTree(MakeShowcaseCapture())) == 0xb41eb67a1d13443eull);
        check(s1Same && s2Same,
              "profile-s1/s2: prior digests 0xedc7791443141dfd + 0xb41eb67a1d13443e UNCHANGED (append-only)");
    }

    // ---- (S3-2) FRAME COUNT — exactly 4 frames, frameNumber 0,1,2,3, each eventCount == 5. ----------
    {
        bool ok = (frames.size() == 4u);
        for (std::size_t i = 0; ok && i < frames.size(); ++i) {
            ok = ok && (frames[i].frameNumber == static_cast<uint32_t>(i)) && (frames[i].eventCount == 5u);
            // Each frame is bracketed FrameBegin..FrameEnd.
            ok = ok && (timeline.events[frames[i].firstEvent].kind == EvKind::FrameBegin);
            ok = ok && (timeline.events[frames[i].firstEvent + frames[i].eventCount - 1u].kind == EvKind::FrameEnd);
        }
        check(ok,
              "profile-s3: BuildFrameIndex(timeline) has 4 frames, each bracketed FrameBegin..FrameEnd");
    }

    // ---- (S3-3) PINNED TIMELINE — DigestTimeline == the hard-pinned uint64 (identical MSVC + clang). -
    check(timelineDigest == kPinnedTimelineDigest,
          "profile-s3: DigestTimeline == pinned uint64 (the multi-frame timeline is byte-stable cross-platform)");

    // ---- (S3-4) PER-FRAME REPRODUCIBILITY — frame 0 and frame 1 (identical workload) match. ---------
    check(frames[0].structuralDigest == frames[1].structuralDigest,
          "profile-s3: per-frame reproducibility -- frame 0 and frame 1 (identical workload) have the SAME structuralDigest");

    // ---- (S3-5) FRAMES DISTINGUISH WORKLOADS — frame 2 (Lit) differs from frame 0 (Shadow). ---------
    check(frames[2].structuralDigest != frames[0].structuralDigest,
          "profile-s3: a different-workload frame (frame 2) has a DIFFERENT structuralDigest (frames distinguish workloads)");

    // ---- (S3-6) POSITION-INDEPENDENT — a second capture of frame 0's workload ALONE has the same cell.
    {
        Capture solo;
        // Intern names in the SAME first-seen order as MakeTimelineCapture so the interned ids match
        // (the per-frame digest encodes nameId; the seek property is over the SAME interned id space).
        const char kFrame[]  = { 'F', 'r', 'a', 'm', 'e' };
        const char kShadow[] = { 'S', 'h', 'a', 'd', 'o', 'w' };
        (void)Intern(solo.names, kFrame, sizeof(kFrame));   // id 0 (matches the timeline capture)
        const uint32_t shadow = Intern(solo.names, kShadow, sizeof(kShadow));  // id 1
        EmitFrameBegin(solo, 0u);
        EmitEnter(solo, shadow);
        EmitDraw (solo, shadow, 2);
        EmitExit (solo, shadow);
        EmitFrameEnd(solo);
        const std::vector<FrameIndex> soloFrames = BuildFrameIndex(solo);
        check(soloFrames.size() == 1u && soloFrames[0].structuralDigest == frames[0].structuralDigest,
              "profile-s3: a frame's structuralDigest depends only on its OWN events (position-independent -- the seek property)");
    }

    // ---- (S3-7) TIMING STILL EXCLUDED — fill timings nonzero, timeline + per-frame digests unchanged. -
    {
        Capture timed = MakeTimelineCapture();
        for (std::size_t i = 0; i < timed.timings.size(); ++i) {
            timed.timings[i].cpuNanos = static_cast<uint64_t>(i) * 1000ull + 7ull;
            timed.timings[i].gpuNanos = static_cast<uint64_t>(i) * 9ull;
        }
        const std::vector<FrameIndex> timedFrames = BuildFrameIndex(timed);
        bool perFrameSame = (timedFrames.size() == frames.size());
        for (std::size_t i = 0; perFrameSame && i < timedFrames.size(); ++i) {
            perFrameSame = perFrameSame && (timedFrames[i].structuralDigest == frames[i].structuralDigest);
        }
        const bool timelineSame = (DigestTimeline(timedFrames) == timelineDigest);
        check(perFrameSame && timelineSame,
              "profile-s3: TIMING STILL EXCLUDED -- filling timings nonzero leaves the timeline digest unchanged");
    }

    if (g_fail == 0) { std::printf("profile_test: ALL PASS\n"); return 0; }
    std::printf("profile_test: %d FAIL\n", g_fail);
    return 1;
}
