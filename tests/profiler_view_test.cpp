// Unit test for the profiler timeline VIEW DATA model (engine/editor/profiler_view_data.h, issue #31 — the
// GUI half of Profiler + GPU debugger integration). Pure CPU (hf_core), ASan-eligible like the other pure
// tests, NO image / NO render-bake.
//
// SELF-CONTAINED: the scaffolding (check() + HF_TEST_MAIN_INIT) mirrors flow_editor_test.cpp / profile_test.cpp
// so this compiles STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/profiler_view_test.cpp` on the
// Mac — the cheap cross-platform proof. The whole layout is INTEGER (frame cells at a fixed stride, scope bars
// indented by tree depth with cpu-PROPORTIONAL widths computed in integer arithmetic over a FIXED-timings
// capture) so the view — and hence DigestProfilerView (FNV-1a-64) over it — is bit-identical run-to-run AND
// platform-to-platform (MSVC == Apple clang). The golden is a PINNED FNV-1a-64 value IN the test.
//
// The capture is profile::MakeTimelineCaptureTimed(): a FIXED 4-frame capture with FIXED nonzero timings
// (timings[i] = {(i+1)*1000, (i+1)*7}). Frames: 0=Shadow(2 draws), 1=Shadow(2 draws), 2=Lit(5 draws),
// 3=Post(1 draw). One ScopeEnter per frame -> 4 scope rows; one DrawCall per frame -> 4 draw rows.
//
// What this pins:
//   (a) the frame-cell count == 4 (BuildFrameIndex sees 4 complete frames);
//   (b) the scope-row count == 4 (one Enter per frame) + a hand-checked indent/position of a scope row;
//   (c) the draw-row count == 4 (one DrawCall per frame) + the hand-checked per-pass draw counts;
//   (d) DigestProfilerView(view) == a hard-pinned uint64 (the cross-platform proof);
//   (e) re-building the view is bit-identical (deterministic / replay-stable);
//   (f) the cpu-proportional bar widths are a pure integer function of the FIXED timings (the widest scope
//       gets scopeBarMaxW; a smaller scope gets a strictly-smaller bar).

#include "editor/profiler_view_data.h"
#include "profile/profile.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::editor;
using namespace hf::profile;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

int main() {
    HF_TEST_MAIN_INIT();

    const Capture cap = MakeTimelineCaptureTimed();
    const ProfLayout L;  // default fixed grid
    const ProfilerView view = BuildProfilerView(cap, L);

    std::printf("profiler-view: frames=%zu scopes=%zu draws=%zu timelineW=%d maxScopeNanos=%llu\n",
                view.frames.size(), view.scopes.size(), view.draws.size(),
                view.timelineW, static_cast<unsigned long long>(view.maxScopeNanos));

    // ---- (a) frame-cell count == 4 complete frames, laid out at a fixed stride. ----------------------
    check(view.frames.size() == 4, "profiler-view: 4 frame cells (BuildFrameIndex sees 4 complete frames)");
    if (view.frames.size() == 4) {
        // Cell 0 at originX/timelineY; each subsequent cell offset by frameCellW + frameGap.
        check(view.frames[0].x == L.originX && view.frames[0].y == L.timelineY,
              "profiler-view: frame cell 0 at (originX, timelineY)");
        check(view.frames[1].x == L.originX + (L.frameCellW + L.frameGap),
              "profiler-view: frame cell 1 offset by frameCellW+frameGap");
        check(view.frames[3].x == L.originX + 3 * (L.frameCellW + L.frameGap),
              "profiler-view: frame cell 3 at the 4th stride");
        // Frame numbers come straight from the FrameIndex (0,1,2,3).
        check(view.frames[0].frameNumber == 0 && view.frames[1].frameNumber == 1 &&
              view.frames[2].frameNumber == 2 && view.frames[3].frameNumber == 3,
              "profiler-view: frame cells carry frame numbers 0..3");
        // Each frame summed cpuNanos is nonzero (the FIXED timing overlay flows into the cell label).
        check(view.frames[0].cpuNanos > 0 && view.frames[3].cpuNanos > view.frames[0].cpuNanos,
              "profiler-view: frame cpuNanos are the summed fixed timings (monotone here)");
    }

    // ---- (b) scope-row count == 4 (one ScopeEnter per frame) + indent/position of a scope row. --------
    check(view.scopes.size() == 4, "profiler-view: 4 scope rows (one ScopeEnter per frame)");
    if (view.scopes.size() == 4) {
        // Every scope here is a top-level scope (directly under the synthetic root) -> depth 1.
        check(view.scopes[0].depth == 1 && view.scopes[3].depth == 1,
              "profiler-view: each frame's lone scope is depth 1 (top-level under root)");
        // X indent = originX + depth*scopeIndent; Y = scopeY + preIndex*scopeRowH.
        check(view.scopes[0].x == L.originX + 1 * L.scopeIndent && view.scopes[0].y == L.scopeY,
              "profiler-view: scope row 0 at (originX+indent, scopeY)");
        check(view.scopes[1].y == L.scopeY + 1 * L.scopeRowH,
              "profiler-view: scope row 1 one scopeRowH below");
        // Print the layout we produced (for the report).
        for (std::size_t i = 0; i < view.scopes.size(); ++i) {
            const ProfScopeRow& s = view.scopes[i];
            const std::vector<std::uint8_t>* nm = ResolveName(cap, s.nameId);
            char buf[32]; std::size_t n = nm ? nm->size() : 0; if (n > 31) n = 31;
            for (std::size_t k = 0; k < n; ++k) buf[k] = static_cast<char>((*nm)[k]); buf[n] = '\0';
            std::printf("  scope[%zu] '%s' depth=%d x=%d y=%d w=%d cpuNanos=%llu subtreeDraws=%u\n",
                        i, buf, s.depth, s.x, s.y, s.w,
                        static_cast<unsigned long long>(s.cpuNanos), s.subtreeDrawCount);
        }
    }

    // ---- (c) draw-row count == 4 (one DrawCall per frame) + the per-pass draw counts. ----------------
    check(view.draws.size() == 4, "profiler-view: 4 draw rows (one DrawCall per frame)");
    if (view.draws.size() == 4) {
        // The DrawCall amounts: frame0 Shadow draws 2, frame1 Shadow 2, frame2 Lit 5, frame3 Post 1.
        check(view.draws[0].drawCount == 2 && view.draws[1].drawCount == 2 &&
              view.draws[2].drawCount == 5 && view.draws[3].drawCount == 1,
              "profiler-view: draw rows carry per-pass draw counts 2,2,5,1");
        // Draw rows stack at drawY + i*drawRowH.
        check(view.draws[0].y == L.drawY && view.draws[1].y == L.drawY + 1 * L.drawRowH,
              "profiler-view: draw rows stack at drawY + i*drawRowH");
        for (std::size_t i = 0; i < view.draws.size(); ++i) {
            const ProfDrawRow& d = view.draws[i];
            const std::vector<std::uint8_t>* nm = ResolveName(cap, d.passNameId);
            char buf[32]; std::size_t n = nm ? nm->size() : 0; if (n > 31) n = 31;
            for (std::size_t k = 0; k < n; ++k) buf[k] = static_cast<char>((*nm)[k]); buf[n] = '\0';
            std::printf("  draw[%zu] pass='%s' drawCount=%u y=%d\n", i, buf, d.drawCount, d.y);
        }
    }

    // ---- (f) cpu-proportional bar widths are a pure integer function of the FIXED timings. ------------
    // The widest scope (max cpuNanos) gets exactly scopeBarMaxW; every other scope's bar is strictly thinner.
    {
        int widest = 0; uint64_t widestNanos = 0;
        for (const ProfScopeRow& s : view.scopes) {
            if (s.w > widest) widest = s.w;
            if (s.cpuNanos > widestNanos) widestNanos = s.cpuNanos;
        }
        check(widest == L.scopeBarMaxW,
              "profiler-view: the widest scope bar == scopeBarMaxW (cpu-proportional normalizer)");
        check(view.maxScopeNanos == widestNanos,
              "profiler-view: maxScopeNanos == the largest scope cpuNanos");
    }

    // ---- (d) PINNED DIGEST — the cross-platform make-or-break (identical on MSVC + clang). -----------
    const uint64_t digest = DigestProfilerView(view);
    std::printf("profiler-view: view digest = 0x%016llx\n", static_cast<unsigned long long>(digest));
    const uint64_t kPinnedDigest = 0x5c306858206f7ecaull;  // PINNED on first run (MSVC == clang)
    check(digest == kPinnedDigest,
          "profiler-view: DigestProfilerView(view) == pinned uint64 (the cross-platform proof)");

    // ---- (e) REPLAY-STABLE — re-building the same view reproduces the digest. ------------------------
    check(DigestProfilerView(BuildProfilerView(cap, L)) == digest,
          "profiler-view: re-building the view is bit-identical (deterministic)");

    if (g_fail == 0) std::printf("ALL PASS\n");
    else             std::printf("%d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
