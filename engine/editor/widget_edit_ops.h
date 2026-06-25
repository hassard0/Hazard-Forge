#pragma once
// Hazard Forge — UMG-Designer WIDGET-TREE EDIT OPERATIONS (pure CPU, ImGui-free, backend-free).
//
// Issue #30 (UMG-class UI framework), the designer's WRITE path: deterministic, programmatic widget-tree
// mutation ops the designer's add-child / set-property / delete actions call. Mirrors seq_edit_ops.{h}
// (the timeline editor's write path) and flow_edit_ops.{h} (the node graph's), but for a hf::ui::Tree
// instead of a seq::ScalarTrack / flow::Graph. Each op MUTATES the tree deterministically and PRESERVES THE
// INVARIANTS widget.h's SolveLayout relies on: the parent / firstChild / nextSibling links stay a valid
// forest (every non-root widget reachable from root through firstChild/nextSibling, no dangling ids, no
// cycles), and — the crux for DeleteWidget — widget INDICES are COMPACTED on removal so widgets[i].index ==
// i still holds (the WidgetId == array-index discipline widget.h documents), with every parent/firstChild/
// nextSibling reference RE-MAPPED to the new indices. No <algorithm> (hand integer loops, the widget.h
// discipline) so this compiles STANDALONE with clang exactly like widget.h / widget_editor_data.h.
//
// Header-only + SELF-CONTAINED (only ui/widget.h, which pulls <cstddef>/<cstdint>/<vector> + net/session.h +
// seq/seq.h). NO ImGui / rhi / float / hash containers. The view-digest after an edit is deterministic +
// CHANGES (proven by the golden test).

#include "ui/widget.h"   // hf::ui::Tree / Widget / Style / WidgetId / AddWidget / WidgetProp / SetProp

namespace hf::editor {

using hf::ui::Tree;
using hf::ui::Widget;
using hf::ui::Style;
using hf::ui::WidgetId;
using hf::ui::kNoWidget;

// ---- AddChildWidget: append a new child under `parent` with `style`/`kind`. ------------------------
// A thin deterministic wrapper over widget.h's AddWidget (which assigns id == widgets.size() and links the
// new widget as the LAST child of parent via the first-child/next-sibling chain). Returns the new id. An
// out-of-range parent is a deterministic no-op returning kNoWidget (the designer never authors under one).
inline WidgetId AddChildWidget(Tree& t, WidgetId parent, const Style& style, uint32_t kind) {
    if (parent != kNoWidget && parent >= t.widgets.size()) return kNoWidget;  // bad parent -> no-op
    return hf::ui::AddWidget(t, parent, style, kind);
}

// ---- The designer-editable Style properties. REUSES widget.h's S3 WidgetProp where it overlaps -----
// (width/height/flexWeight) and EXTENDS it with the remaining integer Style fields the inspector exposes
// (margins / padding / flags). FROZEN small codes — the designer's set-property action passes one of these.
// (We do NOT renumber widget.h's WidgetProp; these are designer-side codes, a superset, used only by
// SetWidgetStyleProp below.)
enum WidgetStyleProp : uint32_t {
    kWSPWidth = 0, kWSPHeight, kWSPMarginL, kWSPMarginT, kWSPMarginR, kWSPMarginB,
    kWSPPadL,  kWSPPadT,  kWSPPadR,  kWSPPadB,  kWSPFlexWeight, kWSPFlags,
};

// ---- SetWidgetStyleProp: write `value` into the WidgetStyleProp-selected Style field of widget `w`. ----
// Out-of-range widget OR prop = a deterministic no-op. flexWeight/flags are uint32 — clamp a negative
// `value` to 0 before the cast (the SetProp discipline in widget.h). Deterministic; same (tree,w,prop,value)
// -> same mutation. (For width/height/flexWeight this matches widget.h's SetProp exactly; it adds the
// margin/padding/flags fields the designer also edits.)
inline void SetWidgetStyleProp(Tree& t, WidgetId w, uint32_t prop, int32_t value) {
    if (w >= t.widgets.size()) return;                  // out-of-range widget = no-op
    Style& s = t.widgets[w].style;
    const uint32_t u = static_cast<uint32_t>(value < 0 ? 0 : value);
    switch (prop) {
        case kWSPWidth:      s.width   = value; break;
        case kWSPHeight:     s.height  = value; break;
        case kWSPMarginL:    s.marginL = value; break;
        case kWSPMarginT:    s.marginT = value; break;
        case kWSPMarginR:    s.marginR = value; break;
        case kWSPMarginB:    s.marginB = value; break;
        case kWSPPadL:       s.padL    = value; break;
        case kWSPPadT:       s.padT    = value; break;
        case kWSPPadR:       s.padR    = value; break;
        case kWSPPadB:       s.padB    = value; break;
        case kWSPFlexWeight: s.flexWeight = u; break;
        case kWSPFlags:      s.flags      = u; break;
        default: break;                                 // out-of-range prop = no-op
    }
}

// ---- DeleteWidget: remove a widget AND its whole subtree, COMPACTING indices + re-mapping references. ---
// The hard op. widget.h's WidgetId == array index, so a removal must (1) collect the subtree rooted at `w`
// (a flat work-stack walk over firstChild/nextSibling — NO recursion), (2) unlink `w` from its parent's
// child chain, (3) erase the collected widgets and REBUILD the array compacted, re-mapping every surviving
// parent/firstChild/nextSibling id to its new index, and (4) re-map t.root. Deleting the root clears the
// tree (the designer guards against it, but the op is total). Returns false for an out-of-range `w`.
//
// Pure integer, NO <algorithm>, NO recursion (an explicit work-stack), NO hash container (a plain
// old-index -> new-index remap vector). Deterministic: the surviving widgets keep their relative order, so
// the compacted tree is byte-stable; the post-delete view digest is reproducible.
inline bool DeleteWidget(Tree& t, WidgetId w) {
    const std::size_t n = t.widgets.size();
    if (w >= n) return false;

    // (0) Deleting the root empties the tree (total + deterministic).
    if (w == t.root) { t.widgets.clear(); t.root = 0; return true; }

    // (1) Mark the subtree rooted at `w` for removal (pre-order work-stack over the child links).
    std::vector<unsigned char> remove(n, 0u);
    std::vector<WidgetId> stack;
    stack.reserve(n);
    stack.push_back(w);
    while (!stack.empty()) {
        const WidgetId cur = stack.back();
        stack.pop_back();
        if (cur >= n || remove[cur]) continue;
        remove[cur] = 1u;
        for (WidgetId c = t.widgets[cur].firstChild; c != kNoWidget && c < n;
             c = t.widgets[c].nextSibling) {
            stack.push_back(c);
        }
    }

    // (2) Unlink `w` from its parent's first-child/next-sibling chain (so the surviving siblings re-link).
    const WidgetId parent = t.widgets[w].parent;
    if (parent != kNoWidget && parent < n) {
        Widget& p = t.widgets[parent];
        if (p.firstChild == w) {
            p.firstChild = t.widgets[w].nextSibling;           // w was the first child
        } else {
            WidgetId cur = p.firstChild;
            while (cur != kNoWidget && cur < n) {
                if (t.widgets[cur].nextSibling == w) {
                    t.widgets[cur].nextSibling = t.widgets[w].nextSibling;  // splice w out
                    break;
                }
                cur = t.widgets[cur].nextSibling;
            }
        }
    }

    // (3) Build the old-index -> new-index remap over the SURVIVORS (relative order preserved -> stable).
    std::vector<WidgetId> remap(n, kNoWidget);
    WidgetId next = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!remove[i]) remap[i] = next++;
    }
    auto mapId = [&](WidgetId id) -> WidgetId {
        if (id == kNoWidget) return kNoWidget;
        if (id >= n) return kNoWidget;
        return remap[id];                                  // a removed id maps to kNoWidget (never referenced post-unlink)
    };

    // (4) Emit the compacted widget array with every reference re-mapped.
    std::vector<Widget> out;
    out.reserve(static_cast<std::size_t>(next));
    for (std::size_t i = 0; i < n; ++i) {
        if (remove[i]) continue;
        Widget wi = t.widgets[i];
        wi.parent      = mapId(wi.parent);
        wi.firstChild  = mapId(wi.firstChild);
        wi.nextSibling = mapId(wi.nextSibling);
        out.push_back(wi);
    }
    t.widgets = out;
    t.root    = mapId(t.root);    // the root survives (we returned early if w == root)
    return true;
}

}  // namespace hf::editor
