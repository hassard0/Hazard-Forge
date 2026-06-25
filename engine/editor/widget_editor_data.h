#pragma once
// Hazard Forge — UMG-Designer-class WIDGET-TREE EDITOR DATA model (pure CPU, ImGui-free, backend-free).
//
// Issue #30 (UMG-class UI framework — widget hierarchy + data binding + animations), the DESIGNER half:
// the deterministic SEAM between "how a hf::ui::Tree should be DRAWN as a UMG-Designer" and the Dear ImGui
// calls that draw it. This is the EXACT seq_editor_data.{h} / profiler_view_data.{h} / flow_editor_data.{h}
// discipline applied to the UI runtime (engine/ui/widget.h): BuildWidgetEditorView computes a byte-identical
// VIEW with three regions —
//   (1) a HIERARCHY: the widget tree as indented rows (pre-order walk via firstChild/nextSibling, depth ->
//       indent), each carrying its kind + a selection flag;
//   (2) a LAYOUT PREVIEW: SolveLayout(tree, viewport) rects scaled+offset into a preview pane (the ACTUAL
//       integer UI layout drawn as nested boxes), the selected widget flagged;
//   (3) a PROPERTY INSPECTOR: the selected widget's integer Style fields (size / margins / padding / flex /
//       flags) as labeled name+value rows.
// — from a Tree + a viewport + a selected WidgetId, with ZERO ImGui / rhi / backend symbols, so it is
// unit-tested HEADLESSLY (assert the layout + a pinned FNV-1a-64 digest, NOT pixels) and lives in hf_core.
//
// THE DETERMINISM CONTRACT: the UI runtime is already bit-exact integer pixels (the moat — UE5 UMG layout is
// editor-state / DPI / mouse-driven, no two layouts agree). This VIEW keeps that property end-to-end: every
// laid-out coordinate is a PURE INTEGER function of (tree, viewport, selected, layout params). The hierarchy
// row positions are integer (depth*indent), the preview boxes are SolveLayout's integer rects folded through
// an integer scale (an int64 numerator / a positive denominator, NO float, NO <cmath>), and the inspector
// values are the raw Style int32s. Same tree+viewport+selection -> a byte-identical WidgetEditorView -> a
// deterministic golden. DigestWidgetEditorView pins it (hand little-endian -> net::DigestBytes, the
// flow.h / seq.h discipline).
//
// THE NO-RECURSION DISCIPLINE (the flow_editor pre-order rule): the hierarchy is a pre-order tree walk built
// with an explicit integer WORK-STACK over the firstChild/nextSibling links (NEVER C++ recursion) — a child
// is pushed, then its siblings, so rows come out parent-before-children, siblings in insertion order. This is
// the same flat discipline widget.h's SolveLayout uses (it relies on parent-index < child-index; this walk
// does not assume that, so it survives an arbitrary editor-authored tree).
//
// Touches ONLY ui/widget.h (Tree / Widget / Style / Rect / SolveLayout / WidgetId / StyleFlags) and
// net/session.h (DigestBytes — pulled in transitively by widget.h, included explicitly for clarity). NO
// vk*/Metal/rhi rendering symbols, NO imgui.h, NO <cmath>/<algorithm>/float/clock/RNG/hash containers.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ui/widget.h"        // hf::ui::Tree / Widget / Style / Rect / SolveLayout / WidgetId / StyleFlags
#include "net/session.h"      // hf::net::DigestBytes — the pinned-golden FNV-1a-64 currency

namespace hf::editor {

// Bring the UI runtime types into this namespace (the exact names widget.h defines).
using hf::ui::Tree;
using hf::ui::Widget;
using hf::ui::Style;
using hf::ui::Rect;
using hf::ui::WidgetId;
using hf::ui::kNoWidget;
using hf::ui::SolveLayout;

// ---- Layout parameters (fixed integer grid; the layout is a deterministic VALUE) -------------------
// The designer has a left HIERARCHY column (indented rows) and a LAYOUT PREVIEW pane to its right. The
// preview shows the SolveLayout rects scaled by previewNum/previewDen and offset by (previewX, previewY).
// The inspector rows are listed at a fixed stride. All integers -> byte-stable cross-platform (no float).
struct WidgetEditorLayout {
    int hierX      = 16;    // left edge of the hierarchy rows
    int hierY      = 48;    // top of hierarchy row 0
    int rowH       = 22;    // vertical stride between hierarchy rows
    int indentW    = 18;    // horizontal indent per tree depth level

    int previewX   = 360;   // top-left of the layout-preview pane (where viewport (0,0) maps)
    int previewY   = 48;
    int previewNum = 1;     // preview scale numerator   (box px = rect px * previewNum / previewDen)
    int previewDen = 2;     // preview scale denominator (1/2 -> a 1280x720 viewport previews at 640x360)

    int inspX      = 16;    // left edge of the inspector field rows
    int inspY      = 430;   // top of inspector field 0
    int inspRowH   = 20;    // vertical stride between inspector field rows
};

// ---- HIERARCHY: one indented tree row -------------------------------------------------------------
// A pre-order row of the widget tree. (x,y) is the row's top-left on the integer grid (x already indented
// by depth). depth labels the indent level; id/kind annotate it; selected drives the highlight.
struct WidgetTreeRow {
    int      x = 0, y = 0;          // row top-left (x = hierX + depth*indentW)
    int      depth = 0;            // tree depth (0 = root)
    WidgetId id    = kNoWidget;    // the widget this row represents
    uint32_t kind  = 0;           // the widget's kind id (panel/text/... — caller-defined)
    bool     selected = false;    // == (id == the selected widget)
};

// ---- LAYOUT PREVIEW: one nested box = a SolveLayout rect scaled into the preview pane ---------------
// The ACTUAL computed UI rect for a widget, scaled+offset into the preview pane. Boxes nest exactly as the
// real layout nests (children inside parents) because they are the SAME integer rects, uniformly scaled.
struct WidgetLayoutBox {
    int      x = 0, y = 0, w = 0, h = 0;   // preview-space box (scaled SolveLayout rect)
    WidgetId id    = kNoWidget;            // the widget this box lays out
    bool     selected = false;             // == (id == the selected widget)
};

// ---- PROPERTY INSPECTOR: one Style field of the selected widget ------------------------------------
// A labeled integer property of the selected widget's Style. `value` is the raw int32 field value; `name`
// is a stable C-string label (a static literal — NOT serialized into the digest, only `value`/`prop` are).
// `prop` is a stable small code so the panel can group/edit; the golden pins the values, not the labels.
enum InspectorProp : uint32_t {
    kInspWidth = 0, kInspHeight, kInspMarginL, kInspMarginT, kInspMarginR, kInspMarginB,
    kInspPadL,  kInspPadT,  kInspPadR,  kInspPadB,  kInspFlexWeight, kInspFlags,
    kInspKind,                       // the widget's kind (handy in the inspector; not a Style field)
    kInspCount                       // the fixed number of inspector rows (the field order below)
};
struct InspectorField {
    const char* name = "";          // a static label (NOT digested — labels are presentation)
    uint32_t    prop = 0;          // an InspectorProp code (digested — identifies the row)
    int32_t     value = 0;         // the raw Style field value (digested — the data)
};

// ---- The complete, deterministic designer view. Two calls on the identical (tree, viewport, selected)
// yield BYTE-IDENTICAL views (determinism); DigestWidgetEditorView pins it as the golden. ------------
struct WidgetEditorView {
    std::vector<WidgetTreeRow>   rows;        // pre-order hierarchy rows (one per widget)
    std::vector<WidgetLayoutBox> boxes;       // scaled SolveLayout rects (one per widget, index order)
    std::vector<InspectorField>  inspector;   // the selected widget's Style fields (kInspCount rows; empty if no selection)
    WidgetId                     selected = kNoWidget;  // the selected widget id (echoed for the panel)
    int                          widgetCount = 0;       // tree widget count (== rows.size() == boxes.size())
};

// ---- ScaleRect: fold a SolveLayout rect into the preview pane. Pure integer. -----------------------
// box = (rect * num / den) offset by (ox, oy). The multiply/divide is int64 to avoid overflow; w/h round
// to NEAREST (+den/2) so a /2 preview of an odd pixel size is stable; x/y use floor-toward-the-origin via a
// plain integer divide of the non-negative scaled coordinate (rects from SolveLayout have x,y >= viewport
// origin >= 0 in practice; a degenerate den<=0 maps everything to the offset, deterministic).
inline WidgetLayoutBox ScaleRect(const Rect& r, WidgetId id, bool selected,
                                 int ox, int oy, int num, int den) {
    WidgetLayoutBox b;
    b.id = id;
    b.selected = selected;
    if (den <= 0) { b.x = ox; b.y = oy; b.w = 0; b.h = 0; return b; }
    const int64_t n = static_cast<int64_t>(num);
    const int64_t d = static_cast<int64_t>(den);
    const int64_t sx = (static_cast<int64_t>(r.x) * n) / d;            // floor-divide the position
    const int64_t sy = (static_cast<int64_t>(r.y) * n) / d;
    const int64_t sw = (static_cast<int64_t>(r.w) * n + d / 2) / d;    // round the size to nearest
    const int64_t sh = (static_cast<int64_t>(r.h) * n + d / 2) / d;
    b.x = ox + static_cast<int>(sx);
    b.y = oy + static_cast<int>(sy);
    b.w = static_cast<int>(sw);
    b.h = static_cast<int>(sh);
    return b;
}

// ---- BuildWidgetEditorView: lay a ui::Tree out as a UMG-Designer (THE pure deterministic builder) ----
//
// HIERARCHY (rows): a PRE-ORDER tree walk via an explicit integer work-stack over firstChild/nextSibling
// (NO recursion — the flow_editor discipline). Starting from t.root, push the root; on pop, emit a row at
// (hierX + depth*indentW, hierY + emittedSoFar*rowH), then push its children in REVERSE sibling order so
// they pop in insertion order (parent-before-children, siblings left-to-right). depth is carried on the
// stack. Each row carries kind + selected. The walk visits every reachable widget exactly once; for a
// well-formed Tree (every non-root widget reachable from root) rows.size() == widget count.
//
// LAYOUT PREVIEW (boxes): SolveLayout(tree, viewport) -> the integer rects, each folded through ScaleRect
// into the preview pane (one box per widget, in widget-index order). The selected widget's box is flagged.
//
// INSPECTOR (fields): if `selected` is a valid widget, emit kInspCount labeled rows of its Style (and kind);
// otherwise the inspector is empty. Values are the raw int32 Style fields.
inline WidgetEditorView BuildWidgetEditorView(const Tree& t, Rect viewport, WidgetId selected,
                                              const WidgetEditorLayout& L = WidgetEditorLayout{}) {
    WidgetEditorView view;
    const std::size_t n = t.widgets.size();
    view.selected    = selected;
    view.widgetCount = static_cast<int>(n);
    if (n == 0) return view;

    // --- HIERARCHY rows: pre-order via an explicit work-stack (id, depth). ---------------------------
    // The stack holds (widgetId, depth); we pop, emit, then push children reverse so insertion order pops
    // first. A small struct keeps it readable; the vector IS the work-stack (no std::stack, no recursion).
    struct StackItem { WidgetId id; int depth; };
    std::vector<StackItem> stack;
    stack.reserve(n);
    if (t.root < n) stack.push_back(StackItem{ t.root, 0 });
    int emitted = 0;
    while (!stack.empty()) {
        const StackItem cur = stack.back();
        stack.pop_back();

        WidgetTreeRow row;
        row.depth    = cur.depth;
        row.id       = cur.id;
        row.kind     = t.widgets[cur.id].kind;
        row.selected = (cur.id == selected);
        row.x = L.hierX + cur.depth * L.indentW;
        row.y = L.hierY + emitted * L.rowH;
        view.rows.push_back(row);
        ++emitted;

        // Collect this widget's children (insertion order via firstChild/nextSibling), then push them in
        // REVERSE so the first child pops next (pre-order, siblings left-to-right). Bounded by n -> no cycle
        // can loop forever for a well-formed tree; a stray cycle is naturally capped by the child collection
        // being finite per node (we don't track visited — a well-formed Tree has none; this matches widget.h).
        const std::size_t firstPushIdx = stack.size();
        for (WidgetId c = t.widgets[cur.id].firstChild; c != kNoWidget && c < n;
             c = t.widgets[c].nextSibling) {
            stack.push_back(StackItem{ c, cur.depth + 1 });
        }
        // reverse the just-pushed range so child[0] is on top of the stack
        std::size_t lo = firstPushIdx, hi = stack.size();
        while (lo + 1 < hi) { StackItem tmp = stack[lo]; stack[lo] = stack[hi - 1]; stack[hi - 1] = tmp; ++lo; --hi; }
    }

    // --- LAYOUT PREVIEW boxes: the real SolveLayout rects scaled into the preview pane. --------------
    const std::vector<Rect> rects = SolveLayout(t, viewport);
    for (std::size_t i = 0; i < n; ++i) {
        const WidgetId id = static_cast<WidgetId>(i);
        view.boxes.push_back(
            ScaleRect(rects[i], id, /*selected=*/(id == selected),
                      L.previewX, L.previewY, L.previewNum, L.previewDen));
    }

    // --- INSPECTOR: the selected widget's Style fields (kInspCount fixed rows). ----------------------
    if (selected < n) {
        const Style& s = t.widgets[selected].style;
        const uint32_t kind = t.widgets[selected].kind;
        auto add = [&](const char* nm, uint32_t prop, int32_t val) {
            InspectorField f; f.name = nm; f.prop = prop; f.value = val; view.inspector.push_back(f);
        };
        add("width",      kInspWidth,      s.width);
        add("height",     kInspHeight,     s.height);
        add("marginL",    kInspMarginL,    s.marginL);
        add("marginT",    kInspMarginT,    s.marginT);
        add("marginR",    kInspMarginR,    s.marginR);
        add("marginB",    kInspMarginB,    s.marginB);
        add("padL",       kInspPadL,       s.padL);
        add("padT",       kInspPadT,       s.padT);
        add("padR",       kInspPadR,       s.padR);
        add("padB",       kInspPadB,       s.padB);
        add("flexWeight", kInspFlexWeight, static_cast<int32_t>(s.flexWeight));
        add("flags",      kInspFlags,      static_cast<int32_t>(s.flags));
        add("kind",       kInspKind,       static_cast<int32_t>(kind));
    }

    return view;
}

// ---- DigestWidgetEditorView: FNV-1a-64 over a HAND little-endian serialization (the golden). --------
// HAND-LE field by field (NEVER memcpy the structs — padding/endianness-unsafe; the flow.h / seq.h
// discipline) so the digest is byte-stable cross-platform. Encodes selected + widgetCount, then every row
// (x,y,depth,id,kind,selected), every box (x,y,w,h,id,selected), and every inspector field (prop,value).
// Labels are NOT encoded (presentation only). A single add-child / set-prop / delete edit changes the view
// -> changes the digest (proven by the test).
inline uint64_t DigestWidgetEditorView(const WidgetEditorView& v) {
    std::vector<unsigned char> buf;
    buf.reserve(v.rows.size() * 24u + v.boxes.size() * 24u + v.inspector.size() * 8u + 16u);
    auto putU32 = [&](uint32_t x) {
        buf.push_back(static_cast<unsigned char>( x        & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 8)  & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 16) & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 24) & 0xFFu));
    };
    auto putI32 = [&](int x) { putU32(static_cast<uint32_t>(x)); };

    putU32(v.selected);
    putI32(v.widgetCount);

    putU32(static_cast<uint32_t>(v.rows.size()));
    for (const WidgetTreeRow& r : v.rows) {
        putI32(r.x); putI32(r.y); putI32(r.depth);
        putU32(r.id); putU32(r.kind);
        putU32(r.selected ? 1u : 0u);
    }
    putU32(static_cast<uint32_t>(v.boxes.size()));
    for (const WidgetLayoutBox& b : v.boxes) {
        putI32(b.x); putI32(b.y); putI32(b.w); putI32(b.h);
        putU32(b.id);
        putU32(b.selected ? 1u : 0u);
    }
    putU32(static_cast<uint32_t>(v.inspector.size()));
    for (const InspectorField& f : v.inspector) {
        putU32(f.prop);
        putI32(f.value);
    }
    return hf::net::DigestBytes(buf.data(), buf.size());
}

}  // namespace hf::editor
