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

    // ================================ SLICE PROFILE-S4 — draw-call / GPU-pass inspection =============
    // S4 ingests an injected render structure (passes + draws, in execution order) into a deterministic
    // RenderStructure with its OWN pinned digest (counts + ids, NO timing). The MDI Lit pass's drawCount==64
    // is the draw-call-inspection headline.

    // ---- Build the showcase render structure + its render-structure digest. --------------------------
    const RenderStructure rstruct = IngestRenderStructure(MakeShowcaseRenderInput());
    const uint64_t renderDigest = DigestRenderStructure(rstruct);
    std::printf("profile-s4: render-structure digest = 0x%016llx  (%zu passes, %zu draws)\n",
                static_cast<unsigned long long>(renderDigest), rstruct.passes.size(), rstruct.draws.size());

    const uint64_t kPinnedRenderDigest = 0x9b75187d6a4c3bf1ull;  // PINNED on first run (MSVC == clang)

    // ---- (S4-1) PRIOR INVARIANT — S1/S2/S3 digests all UNCHANGED (append-only). ---------------------
    {
        const bool s1Same = (StructuralDigest(MakeShowcaseCapture()) == 0xedc7791443141dfdull);
        const bool s2Same = (DigestTree(BuildScopeTree(MakeShowcaseCapture())) == 0xb41eb67a1d13443eull);
        const bool s3Same = (DigestTimeline(BuildFrameIndex(MakeTimelineCapture())) == 0xc68ff46e1ab25f37ull);
        check(s1Same && s2Same && s3Same,
              "profile-s1/s2/s3: prior digests 0xedc7791443141dfd + 0xb41eb67a1d13443e + 0xc68ff46e1ab25f37 UNCHANGED (append-only)");
    }

    // ---- (S4-2) STRUCTURE — 3 passes, 3 draws, in execution order (passId 0/1/2; Lit firstDraw==1). --
    {
        bool ok = (rstruct.passes.size() == 3u) && (rstruct.draws.size() == 3u);
        ok = ok && (rstruct.draws[0].passId == 0u) && (rstruct.draws[1].passId == 1u) && (rstruct.draws[2].passId == 2u);
        ok = ok && (rstruct.passes[1].firstDraw == 1u);
        check(ok,
              "profile-s4: IngestRenderStructure(showcase) has 3 passes, 3 draws, in execution order");
    }

    // ---- (S4-3) PINNED DIGEST — DigestRenderStructure == the hard-pinned uint64 (identical MSVC + clang).
    check(renderDigest == kPinnedRenderDigest,
          "profile-s4: DigestRenderStructure == pinned uint64 (the render structure is byte-stable cross-platform)");

    // ---- (S4-4) MDI INSPECTION — the Lit pass's draw reports drawCount==64 (the MDI collapse count). -
    check(rstruct.draws[1].drawCount == 64u,
          "profile-s4: the Lit MDI pass's draw reports drawCount==64 (draw-call inspection -- the MDI collapse count)");

    // ---- (S4-5) AGGREGATION — the Lit pass's totalInstances aggregates exactly to 64. ---------------
    check(rstruct.passes[1].totalInstances == 64u,
          "profile-s4: pass totalInstances aggregates exactly (Lit pass totalInstances == 64)");

    // ---- (S4-6) COUNT LOAD-BEARING — change the MDI drawCount to 65 -> a DIFFERENT digest. -----------
    {
        RenderStructInput mutated = MakeShowcaseRenderInput();
        mutated.passes[1].draws[0].drawCount = 65u;
        check(DigestRenderStructure(IngestRenderStructure(mutated)) != renderDigest,
              "profile-s4: a changed draw count changes the render-structure digest (counts are load-bearing)");
    }

    // ---- (S4-7) PIPELINE LOAD-BEARING — change a pipelineId -> a DIFFERENT digest. -------------------
    {
        RenderStructInput mutated = MakeShowcaseRenderInput();
        mutated.passes[0].draws[0].pipelineId += 1u;
        check(DigestRenderStructure(IngestRenderStructure(mutated)) != renderDigest,
              "profile-s4: a changed pipelineId changes the digest (pipeline binding is load-bearing)");
    }

    // ================================ SLICE PROFILE-S5 — THE SCRUB: serializable .capture + seek =====
    // S5 serializes the capture to a `.capture` artifact (structural section FIRST, timing overlay LAST in a
    // separate length-prefixed section) and SCRUBS it: seek to frame N via net::CatchUp == from-0 playback
    // at N (bit-identical). The structural digest covers ONLY the structural section -> corrupting a timing
    // byte is harmless; corrupting a structural byte diverges at the exact frame.

    namespace net = hf::net;

    // ---- Build the timed timeline capture + encode it to .capture bytes. -----------------------------
    const Capture s5cap = MakeTimelineCaptureTimed();
    const std::vector<uint8_t> captureBytes = EncodeCapture(s5cap, 1u);
    const uint64_t captureStructuralDigest = CaptureStructuralDigest(captureBytes);
    std::printf("profile-s5: capture file bytes = %zu, structural-section digest = 0x%016llx\n",
                captureBytes.size(), static_cast<unsigned long long>(captureStructuralDigest));

    const uint64_t kPinnedCaptureStructuralDigest = 0x9830afc651699a70ull;  // PINNED on first run (== StructuralDigest(MakeTimelineCaptureTimed()))

    // ---- (S5-1) PRIOR INVARIANT — S1/S2/S3/S4 digests ALL UNCHANGED (append-only). -------------------
    {
        const bool s1Same = (StructuralDigest(MakeShowcaseCapture()) == 0xedc7791443141dfdull);
        const bool s2Same = (DigestTree(BuildScopeTree(MakeShowcaseCapture())) == 0xb41eb67a1d13443eull);
        const bool s3Same = (DigestTimeline(BuildFrameIndex(MakeTimelineCapture())) == 0xc68ff46e1ab25f37ull);
        const bool s4Same = (DigestRenderStructure(IngestRenderStructure(MakeShowcaseRenderInput())) == 0x9b75187d6a4c3bf1ull);
        check(s1Same && s2Same && s3Same && s4Same,
              "profile-s1/s2/s3/s4: prior digests 0xedc7791443141dfd + 0xb41eb67a1d13443e + 0xc68ff46e1ab25f37 + 0x9b75187d6a4c3bf1 UNCHANGED (append-only)");
    }

    // ---- (S5-2) ROUND-TRIP — DecodeCapture(EncodeCapture(c)) recovers structure + the timing overlay. -
    {
        Capture decoded;
        const bool ok = DecodeCapture(captureBytes, decoded);
        const bool structuralSame = ok && (StructuralDigest(decoded) == StructuralDigest(s5cap));
        const bool timingsSame    = ok && (decoded.timings.size() == s5cap.timings.size());
        bool tsame = timingsSame;
        for (std::size_t i = 0; tsame && i < decoded.timings.size(); ++i) {
            tsame = tsame && (decoded.timings[i].cpuNanos == s5cap.timings[i].cpuNanos)
                          && (decoded.timings[i].gpuNanos == s5cap.timings[i].gpuNanos);
        }
        check(ok && structuralSame && tsame,
              "profile-s5: DecodeCapture(EncodeCapture(c)) round-trips -- StructuralDigest + timings recovered");
    }

    // ---- (S5-3) STRUCTURAL SECTION == S1 — the .capture structural bytes ARE the S1 encoding. ---------
    check(captureStructuralDigest == StructuralDigest(s5cap)
              && captureStructuralDigest == kPinnedCaptureStructuralDigest,
          "profile-s5: the .capture structural-section digest == StructuralDigest(c) (the section IS S1's encoding)");

    // ---- (S5-4) SCRUB == SEEK (THE HEADLINE) — CatchUp(keyframe@K, N) == from-0 playback at N. --------
    // Build frames; wrap the per-frame fold as a net::Session StepFn capturing `frames`. The from-0 world at
    // N and the keyframe world at K are both computed by stepping a Session; SeekToFrame (== net::CatchUp)
    // from worldAtK to N must reach the BIT-IDENTICAL world (full-digest equality -- seek == play).
    {
        const std::vector<FrameIndex> s5frames = BuildFrameIndex(s5cap);

        // The deterministic per-frame fold: currentFrame = t; acc folds in frame t's structuralDigest.
        auto step = [&s5frames](CaptureWorld& w, const std::vector<uint32_t>& /*inputs*/, uint32_t t) {
            w.currentFrame = t;
            if (t < s5frames.size()) w.acc = Mix(w.acc, s5frames[static_cast<std::size_t>(t)].structuralDigest);
        };

        // The tail ring carries every frame's inputs (empty here -- the step folds from `s5frames` directly).
        net::InputRing<uint32_t> tail;   // At(t) returns an empty vector for every t (deterministic)

        // A helper that steps a fresh Session from frame 0 to `toFrame` and returns the world AS OF toFrame.
        auto worldAt = [&](uint32_t toFrame) -> CaptureWorld {
            net::Session<CaptureWorld, uint32_t> s;
            s.world = CaptureWorld{};
            s.ring  = tail;
            s.tick  = 0;
            for (uint32_t t = 0; t < toFrame; ++t) net::Advance(s, step);
            return s.world;
        };

        // Several (K, N) pairs: seek to a keyframe@K then play forward to N == play-from-0 at N.
        const uint32_t pairsK[3] = { 0u, 1u, 2u };
        const uint32_t pairsN[3] = { 3u, 3u, 2u };
        bool allOk = true;
        for (int i = 0; i < 3; ++i) {
            const uint32_t K = pairsK[i];
            const uint32_t N = pairsN[i];
            const CaptureWorld worldAtK   = worldAt(K);
            const CaptureWorld fromZeroAtN = worldAt(N);
            const CaptureWorld seeked      = SeekToFrame(s5frames, N, worldAtK, K, tail, step);
            // Full seek==play equality: the seeked world is BIT-IDENTICAL to the from-0 world at N (the whole
            // CaptureWorld -- both currentFrame and acc -- matches, the SCRUB==SEEK headline).
            allOk = allOk && (DigestCaptureWorld(seeked) == DigestCaptureWorld(fromZeroAtN))
                          && (seeked.currentFrame == fromZeroAtN.currentFrame)
                          && (seeked.acc == fromZeroAtN.acc);
            // Also drive net::CatchUp DIRECTLY (prove the composition is literally CatchUp).
            const net::JoinSnapshot<CaptureWorld> snap{ K, worldAtK };
            const CaptureWorld direct = net::CatchUp(snap, N, tail, step);
            allOk = allOk && (DigestCaptureWorld(direct) == DigestCaptureWorld(fromZeroAtN));
        }
        check(allOk,
              "profile-s5: SCRUB==SEEK -- CatchUp(keyframe@K, N) world == from-0 playback world at N (bit-identical), several (K,N)");
    }

    // ---- (S5-5) MOAT — TIMING CORRUPTION IS HARMLESS — flip the first timing byte; structure UNCHANGED.
    {
        std::vector<uint8_t> bytes = EncodeCapture(s5cap, 1u);
        const uint32_t structuralByteLen = GetU32(bytes.data() + 24);
        const std::size_t timingOff = kCaptureHeaderLen + structuralByteLen;  // the first timing byte
        bytes[timingOff] = static_cast<uint8_t>(bytes[timingOff] ^ 0xFFu);    // corrupt one TIMING byte

        const bool structuralUnchanged = (CaptureStructuralDigest(bytes) == kPinnedCaptureStructuralDigest);

        Capture decoded;
        const bool decodeOk = DecodeCapture(bytes, decoded);
        const bool structureSame = decodeOk && (StructuralDigest(decoded) == StructuralDigest(s5cap));

        // The expected per-frame digests are the AUTHORITY (from the clean capture).
        const std::vector<FrameIndex> expectedFrames = BuildFrameIndex(s5cap);
        std::vector<uint64_t> expected;
        for (std::size_t i = 0; i < expectedFrames.size(); ++i) expected.push_back(expectedFrames[i].structuralDigest);
        const VerifyResult vr = VerifyCapture(decoded, expected);

        check(structuralUnchanged && structureSame && vr.ok,
              "profile-s5: corrupting a TIMING byte leaves the structural digest UNCHANGED (the moat, made testable)");
    }

    // ---- (S5-6) STRUCTURAL CORRUPTION DIVERGES — flip a byte inside the structural section. -----------
    {
        std::vector<uint8_t> bytes = EncodeCapture(s5cap, 1u);
        // Flip a byte INSIDE the structural section, PAST the magic(8)+version(4)+nameCount(4)+name data so
        // it lands in the event records (which BuildFrameIndex reads). Choose an offset within frame 1's
        // events region so a specific frame diverges. We target the event stream by jumping well past the
        // header+names: corrupt a byte deep enough to mutate an event field but keep the stream parseable.
        const std::vector<uint8_t> structural = EncodeStructural(s5cap);
        // Locate the eventCount field: magic(8)+version(4)+nameCount(4)+[per name len(4)+bytes]. Then events.
        std::size_t p = 8 + 4;
        const uint32_t nameCount = GetU32(structural.data() + p); p += 4;
        for (uint32_t i = 0; i < nameCount; ++i) { const uint32_t nlen = GetU32(structural.data() + p); p += 4 + nlen; }
        p += 4;  // skip eventCount -> now p == offset of event[0] within the structural section
        // Each event is 16 bytes [kind,nameId,a,b]. Frame 1 = FrameBegin(1),Enter,Draw,Exit,FrameEnd =
        // events[5..9]. Corrupt the `a` field (offset +8) of event 7 (the Draw in frame 1) so frame 1 diverges.
        const std::size_t eventInStruct = p + 7u * 16u + 8u;          // event[7].a within the structural section
        const std::size_t fileOff = kCaptureHeaderLen + eventInStruct; // -> the file offset
        bytes[fileOff] = static_cast<uint8_t>(bytes[fileOff] ^ 0x01u); // flip one structural byte

        const bool structuralChanged = (CaptureStructuralDigest(bytes) != kPinnedCaptureStructuralDigest);

        Capture decoded;
        const bool decodeOk = DecodeCapture(bytes, decoded);

        const std::vector<FrameIndex> expectedFrames = BuildFrameIndex(s5cap);
        std::vector<uint64_t> expected;
        for (std::size_t i = 0; i < expectedFrames.size(); ++i) expected.push_back(expectedFrames[i].structuralDigest);
        const VerifyResult vr = VerifyCapture(decoded, expected);

        check(structuralChanged && decodeOk && !vr.ok && vr.firstBadFrame == 1u,
              "profile-s5: corrupting a STRUCTURAL byte changes the digest AND VerifyCapture diverges at the exact frame");
    }

    // ---- (S5-7) DETERMINISTIC — two EncodeCapture(c) calls are byte-identical. ------------------------
    {
        const std::vector<uint8_t> enc1 = EncodeCapture(s5cap, 1u);
        const std::vector<uint8_t> enc2 = EncodeCapture(s5cap, 1u);
        check(enc1 == enc2,
              "profile-s5: EncodeCapture is deterministic -- two encodes are byte-identical");
    }

    if (g_fail == 0) { std::printf("profile_test: ALL PASS\n"); return 0; }
    std::printf("profile_test: %d FAIL\n", g_fail);
    return 1;
}
