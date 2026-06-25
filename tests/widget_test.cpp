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

    // =================================================================================================
    // Slice WIDGET-S2 — integer layout solver (the CRUX). Seven assertions (spec). The viewport is
    // {0,0,1280,720}. The body is horizontal (kStackH) so left/right lay out side-by-side.
    // =================================================================================================

    // Locate the showcase widgets by walking the chain (NO map). MakeLayoutShowcase == MakeShowcaseTree
    // with body |= kStackH, so the same topology / indices hold.
    const Tree layoutTree = MakeLayoutShowcase();
    const WidgetId L_root   = layoutTree.root;
    const WidgetId L_header = layoutTree.widgets[L_root].firstChild;
    const WidgetId L_body   = layoutTree.widgets[L_header].nextSibling;
    const WidgetId L_footer = layoutTree.widgets[L_body].nextSibling;
    const WidgetId L_left   = layoutTree.widgets[L_body].firstChild;
    const WidgetId L_right  = layoutTree.widgets[L_left].nextSibling;

    const Rect kViewport{0, 0, 1280, 720};
    const std::vector<Rect> solved = SolveLayout(layoutTree, kViewport);
    const uint64_t rectsDigest = DigestRects(solved);

    std::printf("widget-s2: layout rects digest = 0x%016llx  (%zu rects)\n",
                static_cast<unsigned long long>(rectsDigest), solved.size());
    {
        const Rect& rh = solved[L_header]; const Rect& rb = solved[L_body]; const Rect& rf = solved[L_footer];
        const Rect& rl = solved[L_left];   const Rect& rr = solved[L_right];
        std::printf("widget-s2:   header={%d,%d,%d,%d} body={%d,%d,%d,%d} footer={%d,%d,%d,%d}\n",
                    rh.x, rh.y, rh.w, rh.h, rb.x, rb.y, rb.w, rb.h, rf.x, rf.y, rf.w, rf.h);
        std::printf("widget-s2:   left={%d,%d,%d,%d} right={%d,%d,%d,%d}\n",
                    rl.x, rl.y, rl.w, rl.h, rr.x, rr.y, rr.w, rr.h);
    }

    // ---- (S2-1) S1 INVARIANT — the S1 tree digest is UNCHANGED (append-only: kStackH did not move it).
    check(DigestTree(MakeShowcaseTree()) == 0x53da0581a48f615eull,
          "widget-s2: DigestTree(MakeShowcaseTree()) == 0x53da0581a48f615e UNCHANGED (append-only)");

    // ---- (S2-2) PINNED RECTS — the cross-platform make-or-break (identical on MSVC + clang). ---------
    const uint64_t kPinnedRects = 0x95da64c52733eb16ull;  // PINNED on first run (MSVC == clang)
    check(rectsDigest == kPinnedRects,
          "widget-s2: DigestRects(SolveLayout(MakeLayoutShowcase(), {0,0,1280,720})) == pinned uint64 (cross-platform)");

    // ---- (S2-3) EXACT FILL — root is a vertical stack of header(64)+body(flex)+footer(32); their heights
    // + vertical margins exactly equal the root content height (720 - padT(8) - padB(8) = 704). ---------
    {
        const Style& hs = layoutTree.widgets[L_header].style;
        const Style& bs = layoutTree.widgets[L_body].style;
        const Style& fs = layoutTree.widgets[L_footer].style;
        const int32_t contentH = 720 - 8 - 8;   // root padT/padB = 8
        const int32_t sum = solved[L_header].h + solved[L_body].h + solved[L_footer].h
                          + (hs.marginT + hs.marginB) + (bs.marginT + bs.marginB) + (fs.marginT + fs.marginB);
        check(sum == contentH,
              "widget-s2: the viewport is exactly filled — the vertical stack's children heights + margins sum to 720");
    }

    // ---- (S2-4) FIXED PRESERVED — header height == 64, footer height == 32. --------------------------
    check(solved[L_header].h == 64 && solved[L_footer].h == 32,
          "widget-s2: fixed-size widgets keep their size (header height == 64, footer height == 32)");

    // ---- (S2-5) WEIGHTED FLEX — body's right (weight 2) is ~2x left (weight 1), within the ±1 remainder.
    {
        // right (weight 2) ~= 2x left (weight 1). The base shares are floor(1252*1/3)=417 and
        // floor(1252*2/3)=834; the 1px odd-remainder goes to the LOWEST-INDEX flex child (left, weight 1)
        // per the remainder rule, so left=418/right=834 and 2*left exceeds right by 2 (NOT 1). Assert the
        // weighted ~2:1 within the ±2px the floor()+low-index-remainder rule can produce. (See deviation note.)
        const int32_t lw = solved[L_left].w, rw = solved[L_right].w;
        check(rw >= 2 * lw - 2 && rw <= 2 * lw + 2,
              "widget-s2: flex distribution is weighted — body's right child (weight 2) is ~2x the left child (weight 1)");
    }

    // ---- (S2-6) REMAINDER RULE — a direct case: a horizontal container width W (ODD) with two weight-1
    // flex children → the LOWER-index child gets the extra pixel, and left.w + right.w == W (exact fill).
    {
        Tree dt;
        Style contS; contS.flags = kStackH;          // a horizontal container, no padding
        const WidgetId cont = AddWidget(dt, kNoWidget, contS, /*kind=*/0);
        dt.root = cont;
        Style flexS; flexS.flags = kFlexGrow; flexS.flexWeight = 1;
        const WidgetId a = AddWidget(dt, cont, flexS, 0);   // lower index
        const WidgetId bb = AddWidget(dt, cont, flexS, 0);  // higher index
        const int32_t W = 101;                         // ODD width -> leftover is odd
        const std::vector<Rect> dr = SolveLayout(dt, Rect{0, 0, W, 50});
        const bool ok = dr[a].w == dr[bb].w + 1 && (dr[a].w + dr[bb].w) == W;
        check(ok,
              "widget-s2: the remainder rule — leftover pixels go to the LOWEST-INDEX flex child (exact-fill, deterministic)");
    }

    // ---- (S2-7) DETERMINISTIC — a second SolveLayout -> identical DigestRects. -----------------------
    {
        const std::vector<Rect> again = SolveLayout(layoutTree, kViewport);
        check(DigestRects(again) == rectsDigest,
              "widget-s2: re-solving is bit-identical (deterministic)");
    }

    // =================================================================================================
    // Slice WIDGET-S3 — Data binding (model -> view). Seven assertions (spec). A model (vector of int32)
    // drives EXISTING integer Style fields (width/height/flexWeight) through bindings applied in ascending
    // binding-index order, last-write-wins. The bound tree = MakeLayoutShowcase() then Propagate.
    // =================================================================================================

    // Bound tree: MakeLayoutShowcase() then Propagate(model, bindings). Same topology / indices as S2.
    const std::vector<int32_t> model    = MakeShowcaseModel();
    const std::vector<Binding> bindings = MakeShowcaseBindings();
    Tree boundTree = MakeLayoutShowcase();
    Propagate(boundTree, model, bindings);

    const WidgetId B_header = boundTree.widgets[boundTree.root].firstChild;        // 1
    const WidgetId B_body   = boundTree.widgets[B_header].nextSibling;             // 2
    const WidgetId B_footer = boundTree.widgets[B_body].nextSibling;              // 3
    const WidgetId B_left   = boundTree.widgets[B_body].firstChild;               // 4
    const WidgetId B_right  = boundTree.widgets[B_left].nextSibling;              // 5

    const uint64_t boundTreeDigest = DigestTree(boundTree);
    const std::vector<Rect> boundSolved = SolveLayout(boundTree, kViewport);
    const uint64_t boundRectsDigest = DigestRects(boundSolved);

    std::printf("widget-s3: bound tree digest = 0x%016llx   bound rects digest = 0x%016llx\n",
                static_cast<unsigned long long>(boundTreeDigest),
                static_cast<unsigned long long>(boundRectsDigest));

    // ---- (S3-1) PRIOR INVARIANT — S1 tree digest + S2 rects digest UNCHANGED (append-only). ----------
    check(DigestTree(MakeShowcaseTree()) == 0x53da0581a48f615eull
       && DigestRects(SolveLayout(MakeLayoutShowcase(), kViewport)) == 0x95da64c52733eb16ull,
          "widget-s1/s2: prior digests STILL green — 0x53da0581a48f615e + 0x95da64c52733eb16 UNCHANGED");

    // ---- (S3-2) BOUND PROPERTIES — the model values landed in the widget properties. -----------------
    {
        const bool landed =
            GetProp(boundTree, B_left,   kPropFlexWeight) == model[1]   // left.flexWeight  == 3
         && GetProp(boundTree, B_right,  kPropFlexWeight) == model[3]   // right.flexWeight == 2
         && GetProp(boundTree, B_footer, kPropHeight)     == model[2]   // footer.height    == 48
         && GetProp(boundTree, B_header, kPropHeight)     == model[4];  // header.height    == 99 (LWW)
        check(landed,
              "widget-s3: after Propagate, the bound widget properties hold the model values (header.height == model[...])");
    }

    // ---- (S3-3) PINNED BOUND TREE — DigestTree AFTER Propagate == pinned uint64, DIFFERS from unbound.
    const uint64_t kPinnedBoundTree = 0xbb31678bf35c1a37ull;  // PINNED on first run (MSVC == clang)
    check(boundTreeDigest == kPinnedBoundTree && boundTreeDigest != 0x53da0581a48f615eull,
          "widget-s3: DigestTree after Propagate == pinned uint64 (binding changed the tree, byte-stable)");

    // ---- (S3-4) PINNED BOUND RECTS — the binding flowed through to the computed layout. ---------------
    const uint64_t kPinnedBoundRects = 0xc93bdcf2c0b473a3ull;  // PINNED on first run (MSVC == clang)
    check(boundRectsDigest == kPinnedBoundRects,
          "widget-s3: DigestRects(SolveLayout(bound tree)) == pinned uint64 (the binding flowed through to layout)");

    // ---- (S3-5) LAST-WRITE-WINS — the two bindings to header.kPropHeight resolve to the HIGHER index.
    {
        // binding[0] = model[0]=80 -> header.height; binding[4] = model[4]=99 -> header.height. The later
        // (index 4) must win, so header.height == 99 (NOT 80).
        check(GetProp(boundTree, B_header, kPropHeight) == model[4]
           && GetProp(boundTree, B_header, kPropHeight) != model[0],
              "widget-s3: last-write-wins — two bindings to the same (widget,prop) resolve to the HIGHER-index binding");
    }

    // ---- (S3-6) LOAD-BEARING — change one bound model value, re-Propagate (fresh tree) -> different digest.
    {
        std::vector<int32_t> model2 = MakeShowcaseModel();
        model2[1] += 1;                                  // nudge left's flex weight
        Tree boundTree2 = MakeLayoutShowcase();
        Propagate(boundTree2, model2, bindings);
        check(DigestTree(boundTree2) != boundTreeDigest,
              "widget-s3: binding is load-bearing — changing a model value changes the bound digest");
    }

    // ---- (S3-7) DETERMINISTIC — Propagate twice (fresh trees) -> identical bound digest. --------------
    {
        Tree boundTreeAgain = MakeLayoutShowcase();
        Propagate(boundTreeAgain, model, bindings);
        check(DigestTree(boundTreeAgain) == boundTreeDigest,
              "widget-s3: Propagate is deterministic — re-applying yields the identical digest");
    }

    if (g_fail == 0) { std::printf("widget_test: ALL PASS\n"); return 0; }
    std::printf("widget_test: %d FAIL\n", g_fail);
    return 1;
}
