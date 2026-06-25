// Deterministic UI widget tree + INTEGER box model (Slice WIDGET-S1, issue #30 — the DETERMINISTIC
// UMG-CLASS UI FRAMEWORK beachhead). A widget hierarchy (parent / first-child / next-sibling) with an
// integer box-model style (size / margins / padding) + a hard-pinned tree digest, byte-identical
// cross-platform + replayable. The float text-shaping + pixel raster are isolated later (widget_render.h).
//
// THE INTEGER-PIXEL DECISION (the banner): the box model is INTEGER PIXELS (int32_t), NOT Q16.16 — UI
// layout is pixel-quantized and integer constraint math is trivially deterministic (no fxmul rounding).
// Q16.16 enters ONLY in S4 (animation), quantized back to pixels on apply. So S1's style fields are plain
// int32_t pixels.
//
// SELF-CONTAINED ON PURPOSE: this header includes ONLY <cstddef>/<cstdint>/<vector> + net/session.h (for
// net::DigestBytes) so it compiles STANDALONE with `clang++ -std=c++20 -I engine -I tests
// tests/widget_test.cpp` on the Mac — the cheap cross-platform proof. NO <cmath>/float/clock/RNG/<random>/
// <unordered_*>/<map>/<functional>/std::hash/<algorithm>/<string>. The LE appenders are inlined here (NO
// replay.h/seq.h/flow.h). This is ONE growing header — every later slice (S2-S5) APPENDS a section below
// S1; do NOT modify S1's symbols once pinned.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "net/session.h"  // hf::net::DigestBytes (FNV-1a-64) — the tree-digest currency

namespace hf::ui {

// --- Inline little-endian appenders (self-contained — mirror flow.h:805) -----------------------------
inline void PutU32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(static_cast<uint8_t>( v        & 0xFFu));
    b.push_back(static_cast<uint8_t>((v >> 8)  & 0xFFu));
    b.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
    b.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
}
inline void PutI32(std::vector<uint8_t>& b, int32_t v) { PutU32(b, static_cast<uint32_t>(v)); }  // two's-complement bits, LE-stable

// --- Types (all in hf::ui) ---------------------------------------------------------------------------
using WidgetId = uint32_t;                  // a widget's id == its index into Tree::widgets (monotonic, never recycled — the flow.h NodeId discipline)
constexpr WidgetId kNoWidget = 0xFFFFFFFFu; // the "no widget" sentinel (parent/firstChild/nextSibling terminator)

// Box-model style flags — FROZEN bit values (the wire contract; S2 reads them). Do NOT renumber.
enum StyleFlags : uint32_t {
    kFixedW   = 1u << 0,   // width is a fixed pixel size (else derived by layout)
    kFixedH   = 1u << 1,   // height is a fixed pixel size
    kFlexGrow = 1u << 2,   // this widget grows to fill leftover main-axis space (S2)
    kStackH   = 1u << 3,   // S2: container stacks its children HORIZONTALLY (default = vertical). Append-only
                           //     enum value — does NOT change the Style struct layout or S1's EncodeTree, so
                           //     S1's pinned digest 0x53da0581a48f615e is unaffected.
};

struct Style {                              // ALL integer pixels
    int32_t  width = 0,  height = 0;        // fixed pixel size when kFixedW/kFixedH set
    int32_t  marginL = 0, marginT = 0, marginR = 0, marginB = 0;
    int32_t  padL = 0,    padT = 0,    padR = 0,    padB = 0;
    uint32_t flexWeight = 1;                // relative grow weight when kFlexGrow set (S2)
    uint32_t flags      = 0;                // StyleFlags bitmask
};

struct Widget {
    WidgetId parent      = kNoWidget;       // index of parent (kNoWidget for the root)
    uint32_t firstChild  = kNoWidget;       // index of the first child (kNoWidget if none)
    uint32_t nextSibling = kNoWidget;       // index of the next sibling (kNoWidget if last)
    Style    style;
    uint32_t kind        = 0;               // widget kind id (e.g. 0=panel, 1=text — caller-defined; S6 uses it)
};

struct Tree {
    std::vector<Widget> widgets;            // widgets[0] is the ROOT by convention
    WidgetId            root = 0;
};

// --- Functions (pure, deterministic) -----------------------------------------------------------------

// AddWidget: append a new widget (id = widgets.size()), set parent/style/kind, and LINK it as the LAST
// child of `parent` (insertion-ordered first-child/next-sibling chain — NEVER a set/map). parent ==
// kNoWidget just appends without linking (used once for the root). Returns the new id.
inline WidgetId AddWidget(Tree& t, WidgetId parent, const Style& s, uint32_t kind) {
    const WidgetId id = static_cast<WidgetId>(t.widgets.size());
    Widget w;
    w.parent = parent;
    w.style  = s;
    w.kind   = kind;
    t.widgets.push_back(w);

    if (parent != kNoWidget) {
        Widget& p = t.widgets[parent];
        if (p.firstChild == kNoWidget) {
            p.firstChild = id;              // first child
        } else {
            WidgetId cur = p.firstChild;    // walk the sibling chain to the last child
            while (t.widgets[cur].nextSibling != kNoWidget) cur = t.widgets[cur].nextSibling;
            t.widgets[cur].nextSibling = id;
        }
    }
    return id;
}

// ChildCount: walk the first-child/next-sibling chain, count.
inline uint32_t ChildCount(const Tree& t, WidgetId w) {
    uint32_t n = 0;
    WidgetId c = t.widgets[w].firstChild;
    while (c != kNoWidget) { ++n; c = t.widgets[c].nextSibling; }
    return n;
}

// EncodeTree: hand-LE, the GOLDEN bytes. FIXED order: widgetCount, root; then per widget (index order)
// parent, firstChild, nextSibling, the Style field-by-field, then kind. NEVER memcpy the struct (padding).
inline std::vector<uint8_t> EncodeTree(const Tree& t) {
    std::vector<uint8_t> b;
    PutU32(b, static_cast<uint32_t>(t.widgets.size()));
    PutU32(b, t.root);
    for (const Widget& w : t.widgets) {
        PutU32(b, w.parent);
        PutU32(b, w.firstChild);
        PutU32(b, w.nextSibling);
        PutI32(b, w.style.width);
        PutI32(b, w.style.height);
        PutI32(b, w.style.marginL);
        PutI32(b, w.style.marginT);
        PutI32(b, w.style.marginR);
        PutI32(b, w.style.marginB);
        PutI32(b, w.style.padL);
        PutI32(b, w.style.padT);
        PutI32(b, w.style.padR);
        PutI32(b, w.style.padB);
        PutU32(b, w.style.flexWeight);
        PutU32(b, w.style.flags);
        PutU32(b, w.kind);
    }
    return b;
}

// DigestTree: FNV-1a-64 over the hand-LE serialization (the pinned cross-platform anchor).
inline uint64_t DigestTree(const Tree& t) {
    const std::vector<uint8_t> bytes = EncodeTree(t);
    return net::DigestBytes(bytes.data(), bytes.size());
}

// --- Fixture (deterministic, FIXED forever — the golden pins it) -------------------------------------
// root (vertical stack, kind 0) -> header (kFixedH, height 64) + body (kFlexGrow, weight 1) + footer
// (kFixedH, height 32); body -> left (kFlexGrow, weight 1) + right (kFlexGrow, weight 2); header -> title
// (kind 1 = text). Each widget gets a distinct style (margins/padding) so the digest exercises the fields.
inline Tree MakeShowcaseTree() {
    Tree t;

    Style rootS;                            // root vertical stack
    rootS.padL = 8; rootS.padT = 8; rootS.padR = 8; rootS.padB = 8;
    const WidgetId root = AddWidget(t, kNoWidget, rootS, /*kind=*/0);
    t.root = root;

    Style headerS;                          // header: fixed height 64
    headerS.height = 64; headerS.flags = kFixedH;
    headerS.marginL = 1; headerS.marginT = 2; headerS.marginR = 3; headerS.marginB = 4;
    const WidgetId header = AddWidget(t, root, headerS, /*kind=*/0);

    Style bodyS;                            // body: flex-grow weight 1
    bodyS.flexWeight = 1; bodyS.flags = kFlexGrow;
    bodyS.padL = 5; bodyS.padT = 6; bodyS.padR = 7; bodyS.padB = 8;
    const WidgetId body = AddWidget(t, root, bodyS, /*kind=*/0);

    Style footerS;                          // footer: fixed height 32
    footerS.height = 32; footerS.flags = kFixedH;
    footerS.marginL = 9; footerS.marginT = 10; footerS.marginR = 11; footerS.marginB = 12;
    AddWidget(t, root, footerS, /*kind=*/0);

    Style leftS;                            // body.left: flex-grow weight 1
    leftS.flexWeight = 1; leftS.flags = kFlexGrow;
    leftS.padL = 2; leftS.padR = 2;
    AddWidget(t, body, leftS, /*kind=*/0);

    Style rightS;                           // body.right: flex-grow weight 2
    rightS.flexWeight = 2; rightS.flags = kFlexGrow;
    rightS.padL = 4; rightS.padR = 4;
    AddWidget(t, body, rightS, /*kind=*/0);

    Style titleS;                           // header.title: a text widget (kind 1)
    titleS.width = 120; titleS.height = 24; titleS.flags = kFixedW | kFixedH;
    titleS.marginL = 6; titleS.marginT = 6;
    AddWidget(t, header, titleS, /*kind=*/1);

    return t;
}

// =====================================================================================================
// Slice WIDGET-S2 — Integer layout solver (issue #30). APPEND-ONLY below S1. THE CRUX: a deterministic
// integer top-down layout pass that computes each widget's pixel Rect from the box model (stack direction
// + margins/padding + fixed sizes + flex-grow). The make-or-break detail is the integer flex distribution
// with a precise remainder rule: leftover pixels that don't divide evenly across flex weights are assigned
// +1 each to the LOWEST-INDEX flex children until exhausted, so the container is always EXACTLY filled and
// the result is bit-identical on every compiler. Pure integer (one int64 intermediate for the flex share),
// NO recursion (a flat ascending index scan + per-container child-chain walks), NO new include.
// =====================================================================================================

// --- The computed rect (integer pixels) --------------------------------------------------------------
struct Rect { int32_t x = 0, y = 0, w = 0, h = 0; };

// SolveLayout: ONE deterministic top-down pass (index order, NO recursion). Because AddWidget always
// appends a child AFTER its parent, a parent's index < its children's indices — so when we visit a parent
// (ascending) its own rect is already finalized, and it lays out its OWN children. Flat scan, no work-stack.
inline std::vector<Rect> SolveLayout(const Tree& t, Rect viewport) {
    const std::size_t n = t.widgets.size();
    std::vector<Rect> rects(n);                 // all zero
    if (n == 0) return rects;
    rects[t.root] = viewport;

    for (std::size_t pi = 0; pi < n; ++pi) {
        const Widget& p = t.widgets[pi];
        if (p.firstChild == kNoWidget) continue;        // leaf — rect already set by its parent

        // p's content box = rect minus padding (clamp negatives to 0).
        const Rect pr = rects[pi];
        int32_t cx = pr.x + p.style.padL;
        int32_t cy = pr.y + p.style.padT;
        int32_t cw = pr.w - p.style.padL - p.style.padR; if (cw < 0) cw = 0;
        int32_t ch = pr.h - p.style.padT - p.style.padB; if (ch < 0) ch = 0;

        const bool horizontal = (p.style.flags & kStackH) != 0;
        const int32_t mainSize = horizontal ? cw : ch;   // main axis = x/w when horizontal, y/h when vertical

        // --- Pass A: accumulate used main space + total flex weight over the child chain. -------------
        int32_t  usedMain    = 0;
        uint32_t totalWeight = 0;
        for (WidgetId c = p.firstChild; c != kNoWidget; c = t.widgets[c].nextSibling) {
            const Style& cs = t.widgets[c].style;
            usedMain += horizontal ? (cs.marginL + cs.marginR) : (cs.marginT + cs.marginB);
            const bool fixedMain = horizontal ? ((cs.flags & kFixedW) != 0) : ((cs.flags & kFixedH) != 0);
            const bool flex      = (cs.flags & kFlexGrow) != 0;
            if (fixedMain) {
                usedMain += horizontal ? cs.width : cs.height;   // fixed child main size
            } else if (flex) {
                totalWeight += cs.flexWeight;
            }
            // (a non-fixed non-flex child has main size 0 in v1 — documented)
        }
        int32_t leftover = mainSize - usedMain; if (leftover < 0) leftover = 0;

        // --- Flex distribution: integer share + remainder rule (lowest-index flex children get +1). ---
        // We compute each flex child's base share, sum them, then hand the remainder out in ASCENDING
        // child order. Because the children are walked first-child->next-sibling (== ascending index),
        // this is purely sequential — the leftover counter decrements as we place each flex child.
        int32_t remainder = 0;
        if (totalWeight > 0) {
            int32_t sumShares = 0;
            for (WidgetId c = p.firstChild; c != kNoWidget; c = t.widgets[c].nextSibling) {
                const Style& cs = t.widgets[c].style;
                if ((cs.flags & kFlexGrow) != 0) {
                    const int32_t share =
                        static_cast<int32_t>((static_cast<int64_t>(leftover) * cs.flexWeight) / totalWeight);
                    sumShares += share;
                }
            }
            remainder = leftover - sumShares;   // 0..(#flex children - 1) pixels still to hand out
        }

        // --- Pass B: place children along the main axis. ----------------------------------------------
        const int32_t mainStart  = horizontal ? cx : cy;   // content main-axis start
        const int32_t crossStart = horizontal ? cy : cx;   // content cross-axis start
        const int32_t crossSize  = horizontal ? ch : cw;   // content cross-axis size
        int32_t cursor = mainStart;
        for (WidgetId c = p.firstChild; c != kNoWidget; c = t.widgets[c].nextSibling) {
            const Style& cs = t.widgets[c].style;
            const int32_t leadMainMargin  = horizontal ? cs.marginL : cs.marginT;
            const int32_t trailMainMargin = horizontal ? cs.marginR : cs.marginB;
            const int32_t leadCrossMargin = horizontal ? cs.marginT : cs.marginL;
            const int32_t crossMarginSum  = horizontal ? (cs.marginT + cs.marginB) : (cs.marginL + cs.marginR);

            cursor += leadMainMargin;

            // main size: fixed -> style size; flex -> base share + (one remainder px if any left); else 0.
            const bool fixedMain = horizontal ? ((cs.flags & kFixedW) != 0) : ((cs.flags & kFixedH) != 0);
            const bool flex      = (cs.flags & kFlexGrow) != 0;
            int32_t mainSz = 0;
            if (fixedMain) {
                mainSz = horizontal ? cs.width : cs.height;
            } else if (flex && totalWeight > 0) {
                mainSz = static_cast<int32_t>((static_cast<int64_t>(leftover) * cs.flexWeight) / totalWeight);
                if (remainder > 0) { mainSz += 1; remainder -= 1; }   // lowest-index flex children get the extras
            }
            const int32_t mainPos = cursor;

            // cross size: stretch to content cross size minus cross margins, unless fixed on the cross axis.
            const bool fixedCross = horizontal ? ((cs.flags & kFixedH) != 0) : ((cs.flags & kFixedW) != 0);
            int32_t crossSz;
            if (fixedCross) {
                crossSz = horizontal ? cs.height : cs.width;
            } else {
                crossSz = crossSize - crossMarginSum; if (crossSz < 0) crossSz = 0;
            }
            const int32_t crossPos = crossStart + leadCrossMargin;

            // map main/cross back to x/y/w/h.
            Rect r;
            if (horizontal) { r.x = mainPos; r.w = mainSz; r.y = crossPos; r.h = crossSz; }
            else            { r.y = mainPos; r.h = mainSz; r.x = crossPos; r.w = crossSz; }
            rects[c] = r;

            cursor += mainSz + trailMainMargin;
        }
    }
    return rects;
}

// DigestRects: FNV-1a-64 over the rects (x,y,w,h per rect, hand-LE) — the pinned cross-platform anchor.
inline uint64_t DigestRects(const std::vector<Rect>& rects) {
    std::vector<uint8_t> b;
    for (const Rect& r : rects) { PutI32(b, r.x); PutI32(b, r.y); PutI32(b, r.w); PutI32(b, r.h); }
    return net::DigestBytes(b.data(), b.size());
}

// MakeLayoutShowcase: the S2 layout fixture (FIXED forever). The S1 MakeShowcaseTree() with the BODY
// widget's style.flags |= kStackH so its left/right children lay out side-by-side (exercising horizontal
// flex); the root stays vertical. Modifies a COPY — S1's MakeShowcaseTree is untouched/frozen.
inline Tree MakeLayoutShowcase() {
    Tree t = MakeShowcaseTree();
    const WidgetId root   = t.root;
    const WidgetId header = t.widgets[root].firstChild;     // root.child[0]
    const WidgetId body   = t.widgets[header].nextSibling;  // root.child[1] = body
    t.widgets[body].style.flags |= kStackH;                 // body stacks its left/right horizontally
    return t;
}

}  // namespace hf::ui
