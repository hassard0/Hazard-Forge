// Unit test for the deterministic UI widget tree + INTEGER box model (engine/ui/widget.h, Slice
// WIDGET-S1, issue #30 — the DETERMINISTIC UMG-CLASS UI FRAMEWORK beachhead). Pure CPU (hf_core),
// ASan-eligible like the other pure tests.
//
// SELF-CONTAINED: the test scaffolding (check() + HF_TEST_MAIN_INIT()) is copied from seq_test.cpp (NOT
// included) so this compiles STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/widget_test.cpp`
// on the Mac — the cheap cross-platform proof. The box model is INTEGER PIXELS (int32_t), so the hand-LE
// EncodeTree — and hence DigestTree (FNV-1a-64) over it — is bit-identical run-to-run AND platform-to-
// platform (MSVC vs Apple clang). The golden is a PINNED FNV-1a-64 DigestTree value IN the test (NO image,
// NO render-bake — UE5's UMG cannot pin a tree digest cross-platform).
//
// What this pins (the six WIDGET-S1 assertions):
//   (1) DigestTree(MakeShowcaseTree()) == a hard-pinned uint64 (the cross-platform tree anchor);
//   (2) re-encoding the same tree is byte-identical (deterministic);
//   (3) the tree shape is correct (root has 3 children; body has 2; header has 1);
//   (4) child order is insertion order (header < body < footer; left < right);
//   (5) a changed style field (a margin) changes the digest (style is load-bearing);
//   (6) a different tree shape (an extra child) changes the digest (hierarchy is load-bearing).

#include "ui/widget.h"

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

    // ---- Build the showcase tree + digest it. -------------------------------------------------------
    const Tree showcase = MakeShowcaseTree();
    const uint64_t digest = DigestTree(showcase);

    std::printf("widget-s1: showcase tree digest = 0x%016llx  (%zu widgets)\n",
                static_cast<unsigned long long>(digest), showcase.widgets.size());

    // The pinned golden (computed on first run, hardcoded — the regression anchor / cross-platform bar).
    const uint64_t kPinnedDigest = 0x53da0581a48f615eull;  // PINNED on first run (MSVC == clang)

    // ---- (1) PINNED TREE DIGEST — the cross-platform make-or-break (identical on MSVC + clang). ------
    check(digest == kPinnedDigest,
          "widget-s1: DigestTree(MakeShowcaseTree()) == pinned uint64 (cross-platform tree anchor)");

    // ---- (2) DETERMINISTIC — a second EncodeTree of the same tree -> byte-identical (and digest equal).
    {
        const std::vector<uint8_t> a = EncodeTree(showcase);
        const std::vector<uint8_t> b = EncodeTree(showcase);
        check(a == b && DigestTree(showcase) == digest,
              "widget-s1: re-encoding the same tree is byte-identical (deterministic)");
    }

    // ---- (3) SHAPE — root has 3 children; body has 2; header has 1; root's children are header/body/
    // footer in that order. Locate body + header by walking from the root (NO map). ------------------
    {
        const WidgetId root   = showcase.root;
        const WidgetId header = showcase.widgets[root].firstChild;                 // 1st child
        const WidgetId body   = showcase.widgets[header].nextSibling;              // 2nd child
        const WidgetId footer = showcase.widgets[body].nextSibling;               // 3rd child
        const bool shape = ChildCount(showcase, root)   == 3
                        && ChildCount(showcase, body)   == 2
                        && ChildCount(showcase, header) == 1
                        && showcase.widgets[footer].nextSibling == kNoWidget;     // exactly 3, no 4th
        check(shape,
              "widget-s1: the tree shape is correct (root has 3 children; body has 2 children; header has 1)");
    }

    // ---- (4) CHILD ORDER — header before body before footer; body.left before body.right (insertion
    // order, not reversed). header is kind 0 with one child of kind 1 (title); footer has fixed height 32.
    {
        const WidgetId root   = showcase.root;
        const WidgetId header = showcase.widgets[root].firstChild;
        const WidgetId body   = showcase.widgets[header].nextSibling;
        const WidgetId footer = showcase.widgets[body].nextSibling;
        const WidgetId left   = showcase.widgets[body].firstChild;
        const WidgetId right  = showcase.widgets[left].nextSibling;
        const bool order =
            // root children in insertion order: header (fixed h 64) < body (flex) < footer (fixed h 32)
            (showcase.widgets[header].style.flags & kFixedH) != 0
         && showcase.widgets[header].style.height == 64
         && (showcase.widgets[body].style.flags & kFlexGrow) != 0
         && (showcase.widgets[footer].style.flags & kFixedH) != 0
         && showcase.widgets[footer].style.height == 32
            // body children in insertion order: left (weight 1) < right (weight 2)
         && showcase.widgets[left].style.flexWeight  == 1
         && showcase.widgets[right].style.flexWeight == 2
         && showcase.widgets[right].nextSibling == kNoWidget;
        check(order,
              "widget-s1: child order is insertion order (header before body before footer; left before right)");
    }

    // ---- (5) STYLE LOAD-BEARING — clone the showcase, change one widget's marginL -> a DIFFERENT digest.
    {
        Tree mutated = MakeShowcaseTree();
        const WidgetId header = mutated.widgets[mutated.root].firstChild;
        mutated.widgets[header].style.marginL += 1;   // a single-pixel margin nudge
        check(DigestTree(mutated) != digest,
              "widget-s1: a changed style field (a margin) changes the digest (style is load-bearing)");
    }

    // ---- (6) SHAPE LOAD-BEARING — add one extra child to a widget -> a DIFFERENT digest. -------------
    {
        Tree mutated = MakeShowcaseTree();
        const WidgetId footer = showcase.widgets[showcase.widgets[showcase.root].firstChild]
                                    .nextSibling;                                  // root.child[1] = body
        const WidgetId footerReal = showcase.widgets[footer].nextSibling;         // root.child[2] = footer
        AddWidget(mutated, footerReal, Style{}, /*kind=*/0);                      // give footer a child
        check(DigestTree(mutated) != digest,
              "widget-s1: a different tree shape (an extra child) changes the digest (hierarchy is load-bearing)");
    }

    if (g_fail == 0) { std::printf("widget_test: ALL PASS\n"); return 0; }
    std::printf("widget_test: %d FAIL\n", g_fail);
    return 1;
}
