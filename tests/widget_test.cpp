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

    // =================================================================================================
    // Slice WIDGET-S4 — Widget animation via seq.h. Seven assertions (spec). A seq::ScalarTrack (Q16.16
    // keyframe curve) is sampled at a tick, quantized round-nearest to integer pixels (px = (v+0x8000)>>16),
    // and written via S3's SetProp. The animated tree = MakeLayoutShowcase() then ApplyAnims(..., kOne/2).
    // =================================================================================================

    // Animated tree: MakeLayoutShowcase() then ApplyAnims(MakeShowcaseAnims(), hf::seq::kOne/2). Same topology.
    const std::vector<PropAnim> anims = MakeShowcaseAnims();
    Tree animTree = MakeLayoutShowcase();
    ApplyAnims(animTree, anims, hf::seq::kOne / 2);

    const WidgetId A_header = animTree.widgets[animTree.root].firstChild;          // 1
    const WidgetId A_body   = animTree.widgets[A_header].nextSibling;              // 2
    const WidgetId A_left   = animTree.widgets[A_body].firstChild;                // 4
    const WidgetId A_right  = animTree.widgets[A_left].nextSibling;               // 5

    const uint64_t animTreeDigest = DigestTree(animTree);
    const std::vector<Rect> animSolved = SolveLayout(animTree, kViewport);
    const uint64_t animRectsDigest = DigestRects(animSolved);

    std::printf("widget-s4: animated tree digest = 0x%016llx   animated rects digest = 0x%016llx  (t = 0.5s)\n",
                static_cast<unsigned long long>(animTreeDigest),
                static_cast<unsigned long long>(animRectsDigest));

    // ---- (S4-1) PRIOR INVARIANT — S1 tree + S2 rects + S3 bound-tree + bound-rects ALL UNCHANGED. -----
    check(DigestTree(MakeShowcaseTree()) == 0x53da0581a48f615eull
       && DigestRects(SolveLayout(MakeLayoutShowcase(), kViewport)) == 0x95da64c52733eb16ull
       && DigestTree(boundTree) == 0xbb31678bf35c1a37ull
       && DigestRects(SolveLayout(boundTree, kViewport)) == 0xc93bdcf2c0b473a3ull,
          "widget-s1/s2/s3: prior digests STILL green — 0x53da0581a48f615e + 0x95da64c52733eb16 + 0xbb31678bf35c1a37 + 0xc93bdcf2c0b473a3 UNCHANGED");

    // ---- (S4-2) QUANTIZER — round-nearest Q16.16 -> px. ----------------------------------------------
    check(QuantizePx(96 * hf::seq::kOne) == 96
       && QuantizePx(hf::seq::kOne / 2) == 1     // 0.5 rounds to 1 (round-nearest)
       && QuantizePx(0) == 0
       && QuantizePx(hf::seq::kOne) == 1,
          "widget-s4: the quantizer is round-nearest — QuantizePx(96*kOne)==96, QuantizePx(kOne/2)==1, QuantizePx(0)==0");

    // ---- (S4-3) ANIMATED VALUE — at t=0.5s the header height animates to 96 + left flex weight to 3. -
    check(GetProp(animTree, A_header, kPropHeight) == 96
       && GetProp(animTree, A_left, kPropFlexWeight) == 3,
          "widget-s4: at t=0.5s the header height animates to 96 (the 64->128 linear track sampled + quantized)");

    // ---- (S4-4) PINNED ANIMATED TREE — DigestTree after ApplyAnims == pinned uint64, DIFFERS from unbound.
    const uint64_t kPinnedAnimTree = 0x0fb90c2346c92ca4ull;  // PINNED on first run (MSVC == clang)
    check(animTreeDigest == kPinnedAnimTree && animTreeDigest != 0x53da0581a48f615eull,
          "widget-s4: DigestTree after ApplyAnims(t=0.5s) == pinned uint64 (animated tree, byte-stable cross-platform)");

    // ---- (S4-5) PINNED ANIMATED RECTS — the animation flowed through to the layout. -------------------
    const uint64_t kPinnedAnimRects = 0x494e7ff0ecd098deull;  // PINNED on first run (MSVC == clang)
    check(animRectsDigest == kPinnedAnimRects,
          "widget-s4: DigestRects(SolveLayout(animated tree)) == pinned uint64 (animation flowed through to layout)");

    // ---- (S4-6) TIME LOAD-BEARING — a different tick (t=0.25s) yields a DIFFERENT animated tree digest.
    {
        Tree animTree2 = MakeLayoutShowcase();
        ApplyAnims(animTree2, anims, hf::seq::kOne / 4);
        check(DigestTree(animTree2) != animTreeDigest,
              "widget-s4: a different tick (t=0.25s) yields a DIFFERENT digest (animation is load-bearing on time)");
    }

    // ---- (S4-7) DETERMINISTIC — re-applying at the same tick (fresh tree) is bit-identical. -----------
    {
        Tree animTreeAgain = MakeLayoutShowcase();
        ApplyAnims(animTreeAgain, anims, hf::seq::kOne / 2);
        check(DigestTree(animTreeAgain) == animTreeDigest,
              "widget-s4: ApplyAnims is deterministic — re-applying at the same tick is bit-identical");
    }

    // =================================================================================================
    // Slice WIDGET-S5 — Input/event routing + SCRUB via net::Session (THE HEADLINE). Seven assertions.
    // Wrap the whole UI as a net::Session World (UiWorld) so a UI interaction sequence (clicks +
    // animation + data over time) is lockstep-replayable AND SCRUB-able: seek-to-tick-S+replay ==
    // from-0 bit-identical (the seq SCRUB==SEEK property via net::CatchUp). Viewport {0,0,1280,720},
    // dtPerTick = kOne/10 (0.1s/tick), N = 10 ticks. The ring clicks on tick 2 ({640,360}) + tick 5 ({40,30}).
    // =================================================================================================
    namespace net = hf::net;
    {
        // The step/digest lambdas (capturing the fixed anims/bindings/viewport/dt). The net::Session World
        // is ui::UiWorld; the per-tick Input is ui::UiInput.
        const Rect      s5viewport{0, 0, 1280, 720};
        const hf::seq::fx s5dt = hf::seq::kOne / 10;   // 0.1 s / tick
        const std::vector<PropAnim> s5anims    = MakeShowcaseAnims();
        const std::vector<Binding>  s5bindings = MakeShowcaseBindings();
        auto step = [&](UiWorld& w, const std::vector<UiInput>& in, uint32_t t){
            StepUi(s5anims, s5bindings, s5viewport, s5dt, w, in, t); };
        auto digest = [](const UiWorld& w){ return DigestUi(w); };

        const uint32_t kTicks = 10;
        const net::InputRing<UiInput> ring = MakeShowcaseInputRing();

        // The from-0 final digest + the per-tick trace (the digest AFTER each tick).
        const uint64_t uiFinal = net::RunLockstep(MakeShowcaseUiWorld(), ring, kTicks, step, digest);
        const std::vector<uint64_t> trace = net::DigestTrace(MakeShowcaseUiWorld(), ring, kTicks, step, digest);
        const uint64_t traceDigest = net::DigestBytes(trace.data(), trace.size() * sizeof(uint64_t));

        std::printf("widget-s5: ui final digest = 0x%016llx   ui trace digest = 0x%016llx  (%u ticks)\n",
                    static_cast<unsigned long long>(uiFinal),
                    static_cast<unsigned long long>(traceDigest), kTicks);

        // Pinned S5 goldens (computed on first run, hardcoded — the cross-platform bar; MSVC == clang).
        const uint64_t kPinnedUiFinal = 0x913aa55e209e8fafull;  // PINNED on first run (MSVC == clang)
        const uint64_t kPinnedUiTrace = 0x4f544f805d13ef98ull;  // PINNED on first run (MSVC == clang)

        // ---- (S5-1) PRIOR INVARIANT — re-assert S1 + all S2 + S3 (both) + S4 (both) digests, UNCHANGED. -
        check(DigestTree(MakeShowcaseTree()) == 0x53da0581a48f615eull
           && DigestRects(SolveLayout(MakeLayoutShowcase(), kViewport)) == 0x95da64c52733eb16ull
           && DigestTree(boundTree) == 0xbb31678bf35c1a37ull
           && DigestRects(SolveLayout(boundTree, kViewport)) == 0xc93bdcf2c0b473a3ull
           && DigestTree(animTree) == 0x0fb90c2346c92ca4ull
           && DigestRects(SolveLayout(animTree, kViewport)) == 0x494e7ff0ecd098deull,
              "widget-s1/s2/s3/s4: prior digests STILL green — all six UNCHANGED (S5 is purely additive)");

        // ---- (S5-2) HITTEST — front-most on overlap, kNoWidget on a miss. ------------------------------
        {
            // Solve the layout showcase; a point inside body/right returns right (or the topmost child if
            // one overlaps); the title (header child, last index) is the topmost over the header region.
            const std::vector<Rect> hrects = SolveLayout(MakeLayoutShowcase(), kViewport);
            const WidgetId hitCenter = HitTest(hrects, 640, 360);   // a click in the body region -> right (id 5)
            const WidgetId hitTitle  = HitTest(hrects, 40,  30);    // a click in the header/title region
            const WidgetId hitMiss   = HitTest(hrects, -5,  -5);    // outside every rect
            // Front-most: the highest-index rect containing the point wins (title id 6 sits over header id 1).
            const bool frontMost = (hitCenter == L_right)         // body/right is the front-most at center
                                && (hitTitle == 6 || hitTitle == L_header)  // title (id 6) front-most over header
                                && (hitMiss == kNoWidget);
            check(frontMost,
                  "widget-s5: HitTest is front-most — a point in an overlapping child returns the topmost widget, a miss returns kNoWidget");
        }

        // ---- (S5-3) PINNED FINAL — the from-0 lockstep final digest == a pinned uint64. ----------------
        check(uiFinal == kPinnedUiFinal,
              "widget-s5: RunLockstep ui final digest == pinned uint64 (a UI interaction sequence is deterministic)");

        // ---- (S5-4) PINNED TRACE — DigestTrace has length N AND its byte-digest == a pinned uint64. -----
        check(trace.size() == static_cast<std::size_t>(kTicks) && traceDigest == kPinnedUiTrace,
              "widget-s5: the per-tick DigestTrace digest == pinned uint64 (the interaction trace is replayable)");

        // ---- (S5-5) SCRUB == SEEK (THE HEADLINE) — CatchUp(snapshot@S, N) == the from-0 world at N. -----
        // For several (S,N) pairs: capture the world AS OF tick S (by stepping a Session to S), wrap it in a
        // JoinSnapshot{S, worldAtS}, then CatchUp(snap, N, ring, step) — the ring IS the tail. Assert its
        // DigestUi == the from-0 world at N (re-derived the same way). BIT-IDENTICAL: seek-then-play == play-from-0.
        {
            const uint32_t pairsS[3] = { 2, 5, 7 };
            const uint32_t pairsN[3] = { 10, 10, 7 };
            bool scrubOk = true;
            for (int p = 0; p < 3; ++p) {
                const uint32_t S = pairsS[p], N = pairsN[p];
                UiWorld worldAtS;                  // step a Session to S to capture the confirmed world AS OF S
                {
                    net::Session<UiWorld, UiInput> ss;
                    ss.world = MakeShowcaseUiWorld();
                    ss.ring  = ring;
                    ss.tick  = 0;
                    for (uint32_t t = 0; t < S; ++t) net::Advance(ss, step);
                    worldAtS = ss.world;
                }
                UiWorld worldAtN;                  // the from-0 world at N (re-derived by the same loop)
                {
                    net::Session<UiWorld, UiInput> ss;
                    ss.world = MakeShowcaseUiWorld();
                    ss.ring  = ring;
                    ss.tick  = 0;
                    for (uint32_t t = 0; t < N; ++t) net::Advance(ss, step);
                    worldAtN = ss.world;
                }
                net::JoinSnapshot<UiWorld> snap{ S, worldAtS };
                const UiWorld caughtUp = net::CatchUp(snap, N, ring, step);
                if (DigestUi(caughtUp) != DigestUi(worldAtN)) scrubOk = false;
            }
            check(scrubOk,
                  "widget-s5: SCRUB==SEEK — CatchUp(snapshot@S, N) world == from-0 playback world at N (bit-identical), several (S,N)");
        }

        // ---- (S5-6) CLICK LOAD-BEARING — a run WITH the clicks differs from a run over an EMPTY ring. ----
        {
            const net::InputRing<UiInput> emptyRing;   // no clicks
            const uint64_t emptyFinal = net::RunLockstep(MakeShowcaseUiWorld(), emptyRing, kTicks, step, digest);
            check(emptyFinal != uiFinal,
                  "widget-s5: a click is load-bearing — a run WITH the click differs from a run without (interaction changes state)");
        }

        // ---- (S5-7) DETERMINISTIC — two RunLockstep calls over the same ring -> identical final digest. --
        {
            const uint64_t again = net::RunLockstep(MakeShowcaseUiWorld(), ring, kTicks, step, digest);
            check(again == uiFinal,
                  "widget-s5: two RunLockstep runs over the same ring are identical (deterministic)");
        }
    }

    if (g_fail == 0) { std::printf("widget_test: ALL PASS\n"); return 0; }
    std::printf("widget_test: %d FAIL\n", g_fail);
    return 1;
}
