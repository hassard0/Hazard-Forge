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
#include "seq/seq.h"       // S4: hf::seq::ScalarTrack + SampleScalar (the Q16.16 keyframe sampler) — the ONLY new include

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

// =====================================================================================================
// Slice WIDGET-S3 — Data binding (issue #30). APPEND-ONLY below S2. A model (a vector of integer values)
// drives widget properties through a set of bindings, applied in a DETERMINISTIC order (ascending binding
// index, last-write-wins), so the post-binding tree + layout are byte-identical cross-platform. This is
// UMG's "bind a property to a data source" made deterministic. Bindings target EXISTING integer Style
// fields (width/height/flexWeight) — all already in EncodeTree + all affecting layout — so NO new Style
// field is added (S1's struct + digest 0x53da0581a48f615e stay frozen; S2 rects 0x95da64c52733eb16 too).
// Pure integer, NO new include, NO recursion (a flat ascending-index scan; the InputRing discipline, no
// hash-map). Opacity/visibility are render-side concerns deferred to S6.
// =====================================================================================================

// --- Bindable properties + the binding ---------------------------------------------------------------
// Selects a writable integer Style field. FROZEN values (the wire contract; S4 animation reuses them).
enum WidgetProp : uint32_t {
    kPropWidth      = 0,   // -> Style::width      (a fixed-size widget's pixel width)
    kPropHeight     = 1,   // -> Style::height     (a fixed-size widget's pixel height)
    kPropFlexWeight = 2,   // -> Style::flexWeight (a flex child's grow weight)
};

struct Binding {
    uint32_t srcModelIdx = 0;   // index into the model vector (the source value)
    WidgetId dstWidget   = 0;   // the widget whose property is written
    uint32_t dstProp     = 0;   // a WidgetProp value (the destination field)
};

// --- SetProp / GetProp — write/read a widget property by id -------------------------------------------
// Write `value` into the WidgetProp-selected Style field of widget `w`. Out-of-range widget OR prop = a
// deterministic no-op. flexWeight is uint32 — clamp negative `value` to 0 before the cast.
inline void SetProp(Tree& t, WidgetId w, uint32_t prop, int32_t value) {
    if (w >= t.widgets.size()) return;                 // out-of-range widget = no-op
    Style& s = t.widgets[w].style;
    switch (prop) {
        case kPropWidth:      s.width  = value; break;
        case kPropHeight:     s.height = value; break;
        case kPropFlexWeight: s.flexWeight = static_cast<uint32_t>(value < 0 ? 0 : value); break;
        default: break;                                // out-of-range prop = no-op
    }
}

inline int32_t GetProp(const Tree& t, WidgetId w, uint32_t prop) {
    if (w >= t.widgets.size()) return 0;
    const Style& s = t.widgets[w].style;
    switch (prop) {
        case kPropWidth:      return s.width;
        case kPropHeight:     return s.height;
        case kPropFlexWeight: return static_cast<int32_t>(s.flexWeight);
        default:              return 0;
    }
}

// --- Propagate — apply the bindings in deterministic order --------------------------------------------
// Apply each binding IN ASCENDING BINDING-INDEX ORDER (a flat ordered scan — the InputRing insertion-order
// discipline; NO hash-map). For each binding: if srcModelIdx and dstWidget are in range,
// SetProp(t, b.dstWidget, b.dstProp, model[b.srcModelIdx]). Two bindings to the same (widget,prop) →
// LAST-WRITE-WINS (the higher binding index wins — deterministic). Pure integer.
inline void Propagate(Tree& t, const std::vector<int32_t>& model, const std::vector<Binding>& bindings) {
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        const Binding& b = bindings[i];
        if (b.srcModelIdx >= model.size()) continue;   // source out of range — skip
        if (b.dstWidget   >= t.widgets.size()) continue;
        SetProp(t, b.dstWidget, b.dstProp, model[b.srcModelIdx]);
    }
}

// --- Fixtures (FIXED forever — the golden pins them) -------------------------------------------------
// MakeShowcaseModel: a fixed model. index 0 = a header height, 1 = left flex weight, 2 = footer height,
// 3 = right flex weight, 4 = an unused value (shows some model entries can be unbound). Keep FIXED.
inline std::vector<int32_t> MakeShowcaseModel() {
    return std::vector<int32_t>{ 80, 3, 48, 2, 99 };
}

// MakeShowcaseBindings: a fixed set binding model values to MakeLayoutShowcase's widgets (root=0, header=1,
// body=2, footer=3, left=4, right=5, title=6). The LAST binding to header.kPropHeight (model[4]=99) must
// win the last-write-wins pair over the earlier one (model[0]=80). Keep FIXED.
inline std::vector<Binding> MakeShowcaseBindings() {
    return std::vector<Binding>{
        Binding{ /*src*/0, /*dst header*/1, kPropHeight     },  // header.height <- model[0]=80 (overwritten below)
        Binding{ /*src*/1, /*dst left*/  4, kPropFlexWeight },  // left.flexWeight  <- model[1]=3
        Binding{ /*src*/3, /*dst right*/ 5, kPropFlexWeight },  // right.flexWeight <- model[3]=2
        Binding{ /*src*/2, /*dst footer*/3, kPropHeight     },  // footer.height    <- model[2]=48
        Binding{ /*src*/4, /*dst header*/1, kPropHeight     },  // header.height <- model[4]=99 — LAST-WRITE-WINS
    };
}

// =====================================================================================================
// Slice WIDGET-S4 — Widget animation via seq.h (issue #30). APPEND-ONLY below S3. The composition: a
// seq::ScalarTrack (Q16.16 keyframe curve + easing, proven bit-exact in seq.h) is bound to a widget
// property; sampled at a tick and QUANTIZED to integer pixels, it drives the property — so the animated
// tree + layout are byte-identical cross-platform AND (S5) lockstep/scrub-able. This is UMG's float
// animation curves rebuilt on the seq Q16.16 sampler. THE Q16.16 -> PIXEL BOUNDARY (the one place
// fixed-point touches the integer box model): SampleScalar returns Q16.16; the box model is integer
// pixels; the conversion is a single documented round-nearest quantize px = (v + 0x8000) >> 16
// (arithmetic right shift, well-defined for signed in C++20). Track values are therefore Q16.16 PIXEL
// values (100px == 100*seq::kOne). Reuses seq::SampleScalar + S3's SetProp VERBATIM (zero new animation
// math, zero new property-write code). Pure integer on the bit-exact path, NO recursion, ONE new include
// (seq/seq.h, above). STILL NO <cmath>/float/RNG/clock/<unordered_*>/<map>/std::hash/<algorithm>/<string>.
// =====================================================================================================

// --- The property animation --------------------------------------------------------------------------
// Bind a seq Q16.16 ScalarTrack to a widget property (reuses S3's WidgetProp ids — kPropWidth/kPropHeight/
// kPropFlexWeight). The track's values are Q16.16 PIXEL values (100px == 100*seq::kOne).
struct PropAnim {
    WidgetId         widget = 0;
    uint32_t         prop   = 0;   // a WidgetProp value
    seq::ScalarTrack track;        // the Q16.16 keyframe curve (seq.h)
};

// --- The Q16.16 -> pixel quantizer + ApplyAnims ------------------------------------------------------
// Round-nearest Q16.16 -> integer pixels. The ONE documented quantization (arithmetic >> 16 is C++20-
// defined for signed). Used everywhere a sampled animation value writes the integer box model.
inline int32_t QuantizePx(seq::fx v) { return (int32_t)((v + (seq::fx)0x8000) >> 16); }

// ApplyAnims: sample each animation at time `tSeconds` (Q16.16 seconds) and write the quantized pixel
// value into its widget property via S3's SetProp. Applied in ASCENDING anim-index order (the Propagate
// discipline — last-write-wins for two anims on the same (widget,prop)). Pure of side effects beyond the
// tree. Reuses the proven seq sampler + S3's writer verbatim.
inline void ApplyAnims(Tree& t, const std::vector<PropAnim>& anims, seq::fx tSeconds) {
    for (std::size_t i = 0; i < anims.size(); ++i) {
        const PropAnim& a = anims[i];
        const seq::fx v = seq::SampleScalar(a.track, tSeconds);
        SetProp(t, a.widget, a.prop, QuantizePx(v));
    }
}

// --- Fixture (FIXED forever — the golden pins it) ----------------------------------------------------
// MakeShowcaseAnims: a fixed set over the MakeLayoutShowcase widgets (root=0, header=1, body=2, footer=3,
// left=4, right=5, title=6). At t = seq::kOne/2 (0.5s):
//   - header.kPropHeight     <- track times{0,kOne} values{64*kOne,128*kOne} Linear  -> 96*kOne  -> 96
//   - left.kPropFlexWeight   <- track times{0,kOne} values{1*kOne, 5*kOne}   Linear  ->  3*kOne  ->  3
//   - right.kPropFlexWeight  <- track times{0,kOne} values{2*kOne, 2*kOne}   Linear  ->  2*kOne  ->  2 (a
//     control that does not change). Keep FIXED forever — the golden pins the animated state at a fixed tick.
inline std::vector<PropAnim> MakeShowcaseAnims() {
    std::vector<PropAnim> anims;

    PropAnim header;                                   // header.height 64 -> 128 over [0,1]s (Linear)
    header.widget = 1; header.prop = kPropHeight;
    header.track.times  = {0, seq::kOne};
    header.track.values = {64 * seq::kOne, 128 * seq::kOne};
    header.track.easing = seq::Easing::Linear;
    anims.push_back(header);

    PropAnim left;                                     // left.flexWeight 1 -> 5 over [0,1]s (Linear)
    left.widget = 4; left.prop = kPropFlexWeight;
    left.track.times  = {0, seq::kOne};
    left.track.values = {1 * seq::kOne, 5 * seq::kOne};
    left.track.easing = seq::Easing::Linear;
    anims.push_back(left);

    PropAnim right;                                    // right.flexWeight constant 2 (a control)
    right.widget = 5; right.prop = kPropFlexWeight;
    right.track.times  = {0, seq::kOne};
    right.track.values = {2 * seq::kOne, 2 * seq::kOne};
    right.track.easing = seq::Easing::Linear;
    anims.push_back(right);

    return anims;
}

// =====================================================================================================
// Slice WIDGET-S5 — Input/event routing + SCRUB via net::Session (issue #30 — THE HEADLINE). APPEND-ONLY
// below S4. Deterministic pointer hit-testing + event routing, and wrapping the whole UI as a net::Session
// World so a UI INTERACTION SEQUENCE (clicks + animation + data over time) is lockstep-replayable AND
// SCRUB-able: seek to interaction-tick S then replay forward is BIT-IDENTICAL to replaying from tick 0
// (the seq SCRUB==SEEK property via net::CatchUp). A UI session replays bit-for-bit on any machine — UE5's
// UMG float playback timing cannot. Pure-integer; reuses SolveLayout/Propagate/ApplyAnims/HitTest verbatim.
// NO new include (net/session.h + seq/seq.h already present). STILL NO <cmath>/float-on-bit-exact-path/RNG/
// clock/<unordered_*>/<map>/std::hash/<algorithm>/<string>, NO recursion.
// =====================================================================================================

// --- Pointer events + deterministic hit-testing ------------------------------------------------------
enum PointerType : uint32_t { kPointerNone = 0, kPointerDown = 1 };  // FROZEN wire values
struct UiInput { int32_t x = 0, y = 0; uint32_t type = kPointerNone; };  // a net::Session Input (value-copyable)

// HitTest: the FRONT-MOST widget whose rect contains (x,y). Scan rects in REVERSE index order (later
// widgets = drawn on top = hit first), return the first containing rect, else kNoWidget. Integer compares
// only (no z-buffer). Contains = x >= r.x && x < r.x+r.w && y >= r.y && y < r.y+r.h. NO recursion (flat scan).
inline WidgetId HitTest(const std::vector<Rect>& rects, int32_t x, int32_t y) {
    for (std::size_t k = rects.size(); k-- > 0; ) {
        const Rect& r = rects[k];
        if (x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h)
            return static_cast<WidgetId>(k);
    }
    return kNoWidget;
}

// --- The UI world (a flat, value-copyable net::Session World) ----------------------------------------
// The complete interactive UI state. Flat + value-copyable so net::Session's value-copy snapshot is
// COMPLETE by construction (the seq SeqPlayhead discipline). `tree` carries the styles (mutated by
// binding/animation), `model` the data, `time` the animation clock, `focus` the last-hit widget.
struct UiWorld {
    Tree                 tree;
    std::vector<int32_t> model;
    seq::fx              time  = 0;
    uint32_t             focus = kNoWidget;
};

// DigestUi: hand-LE the WHOLE world — EncodeTree(tree) bytes, then model ints (PutI32), then time
// (PutI32/PutU32), then focus (PutU32) -> net::DigestBytes. Bit-identical run-to-run + cross-platform.
inline uint64_t DigestUi(const UiWorld& w) {
    std::vector<uint8_t> b = EncodeTree(w.tree);
    for (std::size_t i = 0; i < w.model.size(); ++i) PutI32(b, w.model[i]);
    PutI32(b, static_cast<int32_t>(w.time & 0xFFFFFFFFu));          // low 32 bits of the Q16.16 clock
    PutU32(b, static_cast<uint32_t>((static_cast<uint64_t>(w.time) >> 32) & 0xFFFFFFFFu));  // high 32 bits
    PutU32(b, w.focus);
    return net::DigestBytes(b.data(), b.size());
}

// --- StepUi — the deterministic UI transition (the net::Session StepFn) ------------------------------
// One tick: advance the clock, route this tick's pointer events, then re-propagate bindings + re-apply
// animations so the model/time changes flow into the tree. Static config (anims/bindings/viewport/dt) is
// passed in; the test wraps it in a lambda capturing them (the StepPlayhead pattern). Reuses S2/S3/S4
// verbatim. The click model mutation is EXACTLY model[hit % model.size()] += 1 (documented, deterministic).
inline void StepUi(const std::vector<PropAnim>& anims, const std::vector<Binding>& bindings,
                   Rect viewport, seq::fx dtPerTick,
                   UiWorld& w, const std::vector<UiInput>& inputs, uint32_t /*tick*/) {
    w.time += dtPerTick;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const UiInput& in = inputs[i];
        if (in.type != kPointerDown) continue;
        const std::vector<Rect> rects = SolveLayout(w.tree, viewport);
        const WidgetId hit = HitTest(rects, in.x, in.y);
        w.focus = hit;
        if (hit != kNoWidget && !w.model.empty())
            w.model[hit % w.model.size()] += 1;              // a click bumps the clicked widget's model slot
    }
    Propagate(w.tree, w.model, bindings);                    // the model flows into the tree (S3)
    ApplyAnims(w.tree, anims, w.time);                       // the new clock flows into the tree (S4)
}

// --- Fixtures (FIXED forever — the golden pins them) -------------------------------------------------
// MakeShowcaseUiWorld: the S2/S3 fixtures wrapped — { MakeLayoutShowcase(), MakeShowcaseModel(), 0, kNoWidget }.
inline UiWorld MakeShowcaseUiWorld() {
    return UiWorld{ MakeLayoutShowcase(), MakeShowcaseModel(), 0, kNoWidget };
}

// MakeShowcaseInputRing: a FIXED interaction schedule — a kPointerDown at {640,360} on tick 2 (a click in
// body/right) + another at {40,30} on tick 5 (a click in header), empty otherwise. Keep FIXED.
inline net::InputRing<UiInput> MakeShowcaseInputRing() {
    net::InputRing<UiInput> ring;
    ring.AddInput(2, UiInput{ 640, 360, kPointerDown });
    ring.AddInput(5, UiInput{ 40,  30,  kPointerDown });
    return ring;
}

}  // namespace hf::ui
