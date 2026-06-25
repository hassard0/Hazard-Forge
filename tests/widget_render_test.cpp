// Unit test for the deterministic UI render-capstone bridge (engine/ui/widget_render.h, Slice WIDGET-S6,
// issue #30 — the labeled-layout money-shot of the DETERMINISTIC UMG-CLASS UI flagship). The CHEAP golden
// (no Metal): the render labels (one per widget, placed at the widget's bit-exact computed Rect origin +
// a fixed inset) DERIVE byte-for-byte from the S2 integer layout, so their DigestLabels is pinned
// cross-platform — the float raster (LayoutText -> NDC glyph quads, in the --ui-shot showcase) is the
// documented visresolve-bar; the integer layout digest (S2 0x95da64c52733eb16) is the real golden.
//
// SELF-CONTAINED: the scaffolding (check() + HF_TEST_MAIN_INIT()) mirrors widget_test.cpp so this compiles
// STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/widget_render_test.cpp` — the cheap
// cross-platform proof (identical labels digest on MSVC + clang).
//
// The five WIDGET-S6 assertions:
//   (1) COUNT       — WidgetTreeToLabels yields exactly tree.widgets.size() labels (one per widget);
//   (2) PROVENANCE  — each label's pixel pos == the bit-exact Rect origin + inset (labels derive from layout);
//   (3) PINNED      — DigestLabels == a hard-pinned uint64 (byte-stable on MSVC + clang);
//   (4) S2 EXACT    — DigestRects(SolveLayout(MakeLayoutShowcase, viewport)) == 0x95da64c52733eb16 (unchanged);
//   (5) DETERMINISTIC — two builds -> identical DigestLabels.

#include "ui/widget_render.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::ui;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS %s\n", what); }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- Build the S2 showcase layout, then the render labels. --------------------------------------
    const Rect viewport{0, 0, 1280, 720};
    const Tree showcase = MakeLayoutShowcase();
    const std::vector<Rect> rects = SolveLayout(showcase, viewport);
    const int32_t kPxScale = 3;
    const std::vector<WidgetLabel> labels = WidgetTreeToLabels(showcase, rects, kPxScale);

    const uint64_t labelsDigest = DigestLabels(labels);
    std::printf("widget-render: labels digest = 0x%016llx  (%zu labels)\n",
                static_cast<unsigned long long>(labelsDigest), labels.size());

    // ---- (1) COUNT — one label per widget. ----------------------------------------------------------
    check(labels.size() == showcase.widgets.size(),
          "widget-render: WidgetTreeToLabels(MakeLayoutShowcase, SolveLayout) has one label per widget (provenance count)");

    // ---- (2) PROVENANCE — each label pos == the bit-exact Rect origin + inset. -----------------------
    {
        bool ok = (labels.size() == rects.size());
        for (std::size_t i = 0; ok && i < labels.size(); ++i) {
            if (labels[i].widgetId != static_cast<uint32_t>(i)) ok = false;
            else if (labels[i].pxX != rects[i].x + kLabelInset) ok = false;
            else if (labels[i].pxY != rects[i].y + kLabelInset) ok = false;
            else if (labels[i].pxScale != kPxScale)             ok = false;
        }
        check(ok,
              "widget-render: each label's pixel position == the bit-exact Rect origin (provenance — labels derive from layout)");
    }

    // ---- (3) PINNED LABELS DIGEST — identical on MSVC + clang. ---------------------------------------
    const uint64_t kPinnedLabels = 0xbb55dc9cda2dc9a3ull;  // PINNED on first run (MSVC == clang)
    check(labelsDigest == kPinnedLabels,
          "widget-render: DigestLabels == pinned uint64 (the placement is byte-stable cross-platform)");

    // ---- (4) S2 STILL EXACT — the render bridge did NOT perturb the integer layout. ------------------
    {
        const uint64_t rectsDigest = DigestRects(SolveLayout(MakeLayoutShowcase(), viewport));
        check(rectsDigest == 0x95da64c52733eb16ull,
              "widget-render: the layout is still bit-exact — DigestRects(SolveLayout(showcase)) == 0x95da64c52733eb16 (S2 unchanged)");
    }

    // ---- (5) DETERMINISTIC — two builds -> identical DigestLabels. -----------------------------------
    {
        const std::vector<WidgetLabel> a = WidgetTreeToLabels(showcase, rects, kPxScale);
        const std::vector<WidgetLabel> b = WidgetTreeToLabels(showcase, rects, kPxScale);
        check(DigestLabels(a) == DigestLabels(b) && DigestLabels(a) == labelsDigest,
              "widget-render: two builds are byte-identical (deterministic)");
    }

    if (g_fail) { std::printf("\n%d CHECK(S) FAILED\n", g_fail); return 1; }
    std::printf("\nALL WIDGET-S6 RENDER CHECKS PASSED\n");
    return 0;
}
