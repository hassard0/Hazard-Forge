#pragma once
// Slice WIDGET-S6 — Render capstone bridge (the labeled-layout money-shot of FLAGSHIP #30, issue #30).
// THE ONE FLOAT CROSSING of the whole UI flagship lives in the --ui-shot SHOWCASE (the float NDC convert +
// the float ui::LayoutText glyph quads). This header itself is INTEGER placements — one render label per
// widget, positioned at the widget's bit-exact computed pixel Rect (S2 SolveLayout) plus a small fixed
// inset. The label PLACEMENTS are a pure deterministic function of (tree, rects), so DigestLabels is
// byte-identical run-to-run AND MSVC-vs-Apple-clang (the provenance: the labels DERIVE from the bit-exact
// S2 layout). The float raster (LayoutText -> NDC glyph quads, the EXISTING --hud text-overlay pipeline) is
// the documented visresolve-bar; the integer layout digest (S2 0x95da64c52733eb16) is the real golden.
//
// SEPARATE HEADER ON PURPOSE: widget.h's S1-S5 are SELF-CONTAINED (only <cstddef>/<cstdint>/<vector> +
// net/session.h + seq/seq.h) so its standalone-clang cross-platform proof stays cheap; this render bridge
// is allowed to include ui/text.h + math/math.h (the econ_render.h / wfc_render.h / seq_render.h split
// precedent). widget.h is UNMODIFIED — S6 appends ONLY this new header + a CPU provenance test + the two
// --ui-shot showcases. All S1-S5 pinned digests stay frozen.

#include <cstdint>
#include <vector>

#include "ui/widget.h"   // S1-S5 UI core (Tree / Rect / SolveLayout / PutU32 / PutI32) — UNMODIFIED
#include "ui/text.h"     // ui::LayoutText / TextVertex (the float text path, used by the --ui-shot showcase)
#include "math/math.h"   // render bridge: float math (the showcase NDC crossing; pulled in for the bridge layer)

namespace hf::ui {

// --- The small fixed inset (golden-stable) -----------------------------------------------------------
// Each label is placed at its widget's Rect top-left origin PLUS this inset (a few pixels of padding so the
// text doesn't kiss the rect edge). FIXED forever — the placement (and hence DigestLabels) pins it.
constexpr int32_t kLabelInset = 4;

// --- WidgetLabel: a widget's name placed at its computed rect (pixel-space) ---------------------------
// The text itself is supplied by the caller's name table (this header stays string-free / clang-light by
// emitting positions + ids, not text). pxX/pxY are integer pixels (the Rect origin + kLabelInset); the
// --ui-shot showcase maps widgetId -> a display string and calls ui::LayoutText to do the float NDC convert.
struct WidgetLabel { uint32_t widgetId; int32_t pxX, pxY; int32_t pxScale; };

// --- WidgetTreeToLabels: the placement bridge (PURE, deterministic, integer) --------------------------
// One label per widget, at the widget's Rect top-left origin + kLabelInset, with the supplied pxScale.
// PURE function of (tree, rects) — no RNG, no clock. The label pixel positions ARE the bit-exact S2 Rect
// origins (the provenance). Empty tree (or a rects vector shorter than the widget count) -> empty output.
inline std::vector<WidgetLabel> WidgetTreeToLabels(const Tree& t, const std::vector<Rect>& rects,
                                                   int32_t pxScale) {
    std::vector<WidgetLabel> out;
    if (t.widgets.empty() || rects.size() != t.widgets.size()) return out;  // empty / mismatch -> no-op
    out.reserve(t.widgets.size());
    for (std::size_t i = 0; i < t.widgets.size(); ++i) {
        WidgetLabel lbl;
        lbl.widgetId = static_cast<uint32_t>(i);
        lbl.pxX      = rects[i].x + kLabelInset;   // the bit-exact rect origin + the fixed inset (provenance)
        lbl.pxY      = rects[i].y + kLabelInset;
        lbl.pxScale  = pxScale;
        out.push_back(lbl);
    }
    return out;
}

// --- DigestLabels: the provenance digest (FNV-1a-64 over the hand-LE placements) -----------------------
// Hand-LE (widgetId, pxX, pxY, pxScale) per label -> net::DigestBytes. Reuses widget.h's PutU32/PutI32 so
// the byte layout matches the rest of the UI flagship. The placements derive byte-for-byte from the
// bit-exact rects, so this digest is identical on MSVC + clang. NEVER memcpy the struct (padding).
inline uint64_t DigestLabels(const std::vector<WidgetLabel>& labels) {
    std::vector<uint8_t> b;
    for (const WidgetLabel& l : labels) {
        PutU32(b, l.widgetId);
        PutI32(b, l.pxX);
        PutI32(b, l.pxY);
        PutI32(b, l.pxScale);
    }
    return net::DigestBytes(b.data(), b.size());
}

}  // namespace hf::ui
