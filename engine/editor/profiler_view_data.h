#pragma once
// Hazard Forge — profiler timeline VIEW DATA model (pure CPU, ImGui-free, backend-free).
//
// Issue #31 (Profiler + GPU debugger integration), the GUI half: the Insights-class profiler VIEW over the
// already-shipped DETERMINISTIC scrub-friendly profiler CAPTURE (engine/profile/profile.h). This is the EXACT
// flow_editor_data.{h} / editor_panel_data.{h} discipline applied to a profile::Capture: BuildProfilerView
// computes a byte-identical VIEW (a row of frame cells along a timeline, the BuildScopeTree hierarchy as
// indented bars, and per-pass draw-call rows) from a Capture, with ZERO ImGui / rhi / backend symbols, so it
// is unit-tested HEADLESSLY (assert the layout + a pinned FNV-1a-64 digest, NOT pixels) and lives in hf_core.
//
// THE DETERMINISM CONTRACT: a profiler measures TIME (non-deterministic) — so this VIEW is fed a FIXED-timings
// capture (profile::MakeTimelineCaptureTimed(), timings[i] = {(i+1)*1000, (i+1)*7}). cpuNanos drives ONLY a
// bar's pixel WIDTH (a proportion of a fixed max, computed in pure INTEGER arithmetic — no float on the layout
// path), so the same fixed capture yields the byte-identical view on any machine. DigestProfilerView pins it
// as the golden (hand little-endian -> net::DigestBytes, the flow.h / replay.h discipline). The cpuNanos
// values are carried in the structs for the panel to LABEL, but every laid-out integer coordinate is a pure
// function of (capture, layout params).
//
// Touches ONLY profile/profile.h (Capture / BuildScopeTree / BuildFrameIndex / the S4 render-structure) and
// net/session.h (DigestBytes). NO vk*/Metal/rhi rendering symbols, NO imgui.h, NO <cmath>/<algorithm>/float/
// clock/RNG/hash containers. NO recursion (the scope-tree pre-order walk uses an explicit integer work-stack,
// mirroring profile.h's DigestTree).

#include <cstddef>
#include <cstdint>
#include <vector>

#include "profile/profile.h"  // hf::profile::Capture / BuildScopeTree / BuildFrameIndex / RenderStructure
#include "net/session.h"      // hf::net::DigestBytes — the pinned-golden FNV-1a-64 currency

namespace hf::editor {

// ---- Layout parameters (fixed integer grid; the layout is a deterministic VALUE) -------------------
// Three stacked regions, top-to-bottom: a TIMELINE row of frame cells, the SCOPE TREE of indented bars,
// and a DRAW-CALL list. All integers -> byte-stable cross-platform (no float on the layout path).
struct ProfLayout {
    int originX        = 40;    // left margin of everything
    // --- Timeline (the row of frame cells) ---
    int timelineY      = 48;    // top Y of the frame-cell row
    int frameCellW     = 150;   // width of each frame cell
    int frameCellH     = 56;    // height of each frame cell
    int frameGap       = 12;    // horizontal gap between frame cells
    // --- Scope tree (indented rows of cpu-proportional bars) ---
    int scopeY         = 160;   // top Y of the first scope row
    int scopeRowH      = 26;    // vertical stride between scope rows
    int scopeIndent    = 22;    // X indent added per tree depth
    int scopeBarMaxW   = 360;   // the pixel width a bar of the MAX cpuNanos in the tree gets
    // --- Draw-call list ---
    int drawY          = 460;   // top Y of the first draw row
    int drawRowH       = 24;    // vertical stride between draw rows
};

// One frame cell on the timeline: a positioned box for one captured frame, carrying its frame number + its
// total cpuNanos (the sum of that frame's per-event cpuNanos overlay — labels the cell, drives no geometry).
struct ProfFrameCell {
    int      x = 0, y = 0, w = 0, h = 0;   // box top-left + size (integer grid)
    uint32_t frameNumber = 0;              // FrameIndex.frameNumber
    uint64_t cpuNanos    = 0;              // sum of cpuNanos over this frame's events (the timing overlay)
};

// One scope-tree row: an indented, cpu-proportional bar for one ScopeNode (pre-order, root excluded). depth
// drives the X indent; the pre-order index drives Y; cpuNanos drives the bar WIDTH (integer proportion of the
// tree's max). nameId resolves to a byte-string via the Capture's NameTable (the panel reads the label).
struct ProfScopeRow {
    int      x = 0, y = 0, w = 0, h = 0;   // bar top-left + size (w == cpu-proportional bar width)
    int      depth   = 0;                  // tree depth (1 for a top-level scope; root is not emitted)
    uint32_t nameId  = 0;                  // the interned scope name id (resolve via Capture.names)
    uint64_t cpuNanos = 0;                 // this scope's summed cpuNanos overlay (drives w)
    uint32_t subtreeDrawCount = 0;         // BuildScopeTree subtreeDrawCount (labels the row)
};

// One draw-call row: a pass name + its draw count (the draw-call inspection list). x/y position the row;
// passNameId resolves to a byte-string via the NameTable; drawCount is the aggregated draws in that pass.
struct ProfDrawRow {
    int      x = 0, y = 0;                  // row top-left (the panel lays out the label + count here)
    uint32_t passNameId = 0;               // the interned pass/scope name id (resolve via Capture.names)
    uint32_t drawCount  = 0;               // total draws attributed to this pass
};

// The complete, deterministic profiler view laid out for the editor. Two calls on the identical capture yield
// BYTE-IDENTICAL views (determinism); DigestProfilerView below pins it as the golden.
struct ProfilerView {
    std::vector<ProfFrameCell> frames;   // one per complete frame (BuildFrameIndex order)
    std::vector<ProfScopeRow>  scopes;   // one per non-root ScopeNode, in pre-order
    std::vector<ProfDrawRow>   draws;    // one per scope that emitted >=1 draw, in pre-order
    int timelineW = 0;                   // total pixel width consumed by the frame-cell row (for the panel)
    uint64_t maxScopeNanos = 0;          // the largest scope cpuNanos (the bar-width normalizer)
};

// ---- SumFrameCpuNanos: total cpuNanos over a frame's events [first, first+count). Pure integer. -----
inline uint64_t SumFrameCpuNanos(const profile::Capture& c, uint32_t first, uint32_t count) {
    uint64_t sum = 0;
    for (uint32_t k = 0; k < count; ++k) {
        const std::size_t i = static_cast<std::size_t>(first) + k;
        if (i < c.timings.size()) sum += c.timings[i].cpuNanos;
    }
    return sum;
}

// ---- BuildProfilerView: lay a profile::Capture out as an Insights-class profiler view (THE builder) ----
//
// TIMELINE: BuildFrameIndex(c) yields one FrameIndex per complete frame; each becomes a ProfFrameCell laid out
// left-to-right at a FIXED stride (originX + i*(frameCellW+frameGap)), carrying its frameNumber + its summed
// cpuNanos. Fixed-width cells (not nanos-proportional) keep the timeline readable + the layout a pure function
// of the frame COUNT; the per-cell cpuNanos labels the cell.
//
// SCOPE TREE: BuildScopeTree(c) yields the Enter/Exit hierarchy. We walk it in PRE-ORDER via an explicit
// integer work-stack (NO recursion — the DigestTree discipline), skipping the synthetic root (index 0). Each
// node emits a ProfScopeRow: y by pre-order index, x indented by depth*scopeIndent, and a bar WIDTH that is
// the integer proportion of the node's cpuNanos against the tree's max scope cpuNanos (w = cpu*barMax/max,
// clamped to >=1 for any nonzero scope so a tiny scope is still visible). A scope's cpuNanos is the sum of
// the cpuNanos overlay over its Enter..matching-Exit event span — computed in one pass off the open-scope
// stack while we re-walk events (the same balancing BuildScopeTree does), so the view needs NO extra profile.h
// surface. Pure integer; the fixed-timings capture makes every width deterministic.
//
// DRAW ROWS: every scope with selfDrawCount > 0 emits one ProfDrawRow (the pass + its draw count) in the same
// pre-order, stacked top-to-bottom — the draw-call inspection list.
inline ProfilerView BuildProfilerView(const profile::Capture& c, const ProfLayout& L = ProfLayout{}) {
    ProfilerView view;

    // --- Timeline: one frame cell per complete frame. ------------------------------------------------
    const std::vector<profile::FrameIndex> frames = profile::BuildFrameIndex(c);
    view.frames.reserve(frames.size());
    int fx = L.originX;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const profile::FrameIndex& fi = frames[i];
        ProfFrameCell cell;
        cell.x = fx;
        cell.y = L.timelineY;
        cell.w = L.frameCellW;
        cell.h = L.frameCellH;
        cell.frameNumber = fi.frameNumber;
        cell.cpuNanos = SumFrameCpuNanos(c, fi.firstEvent, fi.eventCount);
        view.frames.push_back(cell);
        fx += L.frameCellW + L.frameGap;
    }
    view.timelineW = static_cast<int>(view.frames.size()) *
                     (L.frameCellW + L.frameGap) - (view.frames.empty() ? 0 : L.frameGap);

    // --- Scope tree: build the hierarchy, then sum per-scope cpuNanos off an open-scope stack pass. ---
    const profile::ScopeTree tree = profile::BuildScopeTree(c);
    const std::size_t nodeCount = tree.nodes.size();

    // Per-node cpuNanos = sum of the cpuNanos overlay over its Enter..Exit span. Re-walk the event stream with
    // an integer open-scope stack mapping the CURRENT open scope to its tree node index (the SAME canonical
    // tree BuildScopeTree built, so the k-th ScopeEnter maps to node (1 + the running enter count) in build
    // order). We track each open node's accumulating nanos and fold its total into the node on Exit; every
    // event's cpuNanos lands on the innermost open scope (matching DrawCall self-attribution in BuildScopeTree).
    std::vector<uint64_t> nodeNanos(nodeCount, 0);
    {
        std::vector<uint32_t> openStack;    // tree node indices of currently-open scopes
        openStack.push_back(0u);            // the synthetic root is always open (index 0)
        uint32_t nextNode = 1u;             // the next ScopeEnter creates this node (matches BuildScopeTree)
        for (std::size_t i = 0; i < c.events.size(); ++i) {
            const profile::CaptureEvent& ev = c.events[i];
            const uint64_t evNanos = (i < c.timings.size()) ? c.timings[i].cpuNanos : 0ull;
            if (ev.kind == profile::EvKind::ScopeEnter) {
                const uint32_t node = (nextNode < nodeCount) ? nextNode : 0u;
                ++nextNode;
                if (node < nodeCount) nodeNanos[node] += evNanos;  // the Enter event's own nanos
                openStack.push_back(node);
            } else if (ev.kind == profile::EvKind::ScopeExit) {
                if (openStack.size() > 1) {
                    const uint32_t node = openStack.back();
                    if (node < nodeCount) nodeNanos[node] += evNanos;  // the Exit event's nanos
                    openStack.pop_back();
                }
            } else {
                // DrawCall / FrameBegin / FrameEnd: their nanos land on the innermost open scope.
                const uint32_t node = openStack.back();
                if (node < nodeCount) nodeNanos[node] += evNanos;
            }
        }
    }

    // The bar-width normalizer: the max cpuNanos over the NON-root scope nodes.
    uint64_t maxNanos = 0;
    for (std::size_t k = 1; k < nodeCount; ++k) if (nodeNanos[k] > maxNanos) maxNanos = nodeNanos[k];
    view.maxScopeNanos = maxNanos;

    // Per-node DEPTH (root depth 0; a child is parent depth + 1). A child always has a higher index than its
    // parent (BuildScopeTree appends), so a single forward pass settles depth (integer, no recursion).
    std::vector<int> depth(nodeCount, 0);
    for (std::size_t k = 1; k < nodeCount; ++k)
        depth[k] = depth[tree.nodes[k].parent] + 1;

    // Pre-order walk (root, then each child subtree in firstChild->nextSibling order) via an integer work-
    // stack — the DigestTree discipline. Skip the synthetic root (index 0); emit a row per real scope.
    int preIndex = 0;
    std::vector<uint32_t> work;
    work.push_back(0u);
    while (!work.empty()) {
        const uint32_t n = work.back();
        work.pop_back();
        if (n != 0u) {  // skip the synthetic root
            const profile::ScopeNode& node = tree.nodes[n];
            ProfScopeRow row;
            row.depth   = depth[n];
            row.nameId  = node.nameId;
            row.cpuNanos = nodeNanos[n];
            row.subtreeDrawCount = node.subtreeDrawCount;
            row.x = L.originX + depth[n] * L.scopeIndent;
            row.y = L.scopeY + preIndex * L.scopeRowH;
            row.h = L.scopeRowH - 6;   // a little vertical padding between bars
            // Bar width: integer proportion of cpuNanos against the tree max, >=1 for any nonzero scope.
            if (maxNanos > 0 && row.cpuNanos > 0) {
                uint64_t w = row.cpuNanos * static_cast<uint64_t>(L.scopeBarMaxW) / maxNanos;
                if (w < 1) w = 1;
                row.w = static_cast<int>(w);
            } else {
                row.w = 0;
            }
            view.scopes.push_back(row);
            ++preIndex;
        }
        // Push children in REVERSE so they pop in firstChild->nextSibling (emission) order — pre-order.
        std::vector<uint32_t> kids;
        for (uint32_t ch = tree.nodes[n].firstChild; ch != profile::kNoNode; ch = tree.nodes[ch].nextSibling)
            kids.push_back(ch);
        for (std::size_t i = kids.size(); i-- > 0; ) work.push_back(kids[i]);
    }

    // --- Draw rows: one per scope that emitted >=1 direct draw, in the SAME pre-order as the scope rows. -
    int drawIndex = 0;
    for (std::size_t s = 0; s < view.scopes.size(); ++s) {
        // The scope rows are in pre-order; re-derive each row's node by matching the pre-order again would be
        // redundant — instead carry the draw count alongside. We re-walk the tree pre-order once more (cheap)
        // pairing each emitted scope row with its node's selfDrawCount.
        (void)s;
    }
    // Re-walk pre-order to attach selfDrawCount-bearing scopes to draw rows (kept separate so ProfScopeRow
    // stays purely the bar; the draw list is its own inspection panel).
    {
        std::vector<uint32_t> w2;
        w2.push_back(0u);
        while (!w2.empty()) {
            const uint32_t n = w2.back();
            w2.pop_back();
            if (n != 0u) {
                const profile::ScopeNode& node = tree.nodes[n];
                if (node.selfDrawCount > 0u) {
                    ProfDrawRow dr;
                    dr.x = L.originX;
                    dr.y = L.drawY + drawIndex * L.drawRowH;
                    dr.passNameId = node.nameId;
                    dr.drawCount  = node.selfDrawCount;
                    view.draws.push_back(dr);
                    ++drawIndex;
                }
            }
            std::vector<uint32_t> kids;
            for (uint32_t ch = tree.nodes[n].firstChild; ch != profile::kNoNode;
                 ch = tree.nodes[ch].nextSibling)
                kids.push_back(ch);
            for (std::size_t i = kids.size(); i-- > 0; ) w2.push_back(kids[i]);
        }
    }

    return view;
}

// ---- DigestProfilerView: FNV-1a-64 over a HAND little-endian serialization of the view (the golden) --
// HAND-LE field by field (NEVER memcpy the structs — padding/endianness-unsafe; the flow.h / profile.h
// discipline) so the digest is byte-stable cross-platform. Encodes the frame cells (x,y,w,h,frameNumber,
// cpuNanos), the scope rows (x,y,w,h,depth,nameId,cpuNanos,subtreeDrawCount), and the draw rows (x,y,
// passNameId,drawCount). int values are encoded as their two's-complement uint32 LE bits; uint64 nanos via
// PutU64. The NameTable byte-strings are NOT hashed here (they are part of the Capture, not the layout).
inline uint64_t DigestProfilerView(const ProfilerView& v) {
    std::vector<std::uint8_t> buf;
    auto putU32 = [&](uint32_t x) { profile::PutU32(buf, x); };
    auto putI32 = [&](int x)      { profile::PutU32(buf, static_cast<uint32_t>(x)); };
    auto putU64 = [&](uint64_t x) { profile::PutU64(buf, x); };

    putI32(v.timelineW);
    putU64(v.maxScopeNanos);

    putU32(static_cast<uint32_t>(v.frames.size()));
    for (const ProfFrameCell& f : v.frames) {
        putI32(f.x); putI32(f.y); putI32(f.w); putI32(f.h);
        putU32(f.frameNumber);
        putU64(f.cpuNanos);
    }
    putU32(static_cast<uint32_t>(v.scopes.size()));
    for (const ProfScopeRow& s : v.scopes) {
        putI32(s.x); putI32(s.y); putI32(s.w); putI32(s.h);
        putI32(s.depth);
        putU32(s.nameId);
        putU64(s.cpuNanos);
        putU32(s.subtreeDrawCount);
    }
    putU32(static_cast<uint32_t>(v.draws.size()));
    for (const ProfDrawRow& d : v.draws) {
        putI32(d.x); putI32(d.y);
        putU32(d.passNameId);
        putU32(d.drawCount);
    }
    return hf::net::DigestBytes(buf.data(), buf.size());
}

// ---- ResolveName: a NameTable byte-string id -> a (ptr,len) view for the panel to print as a label. ----
// Returns an empty span for the synthetic root nameId (profile::kRootName) or any out-of-range id. Pure;
// no <string> (names are raw byte-strings — the profile.h discipline).
inline const std::vector<std::uint8_t>* ResolveName(const profile::Capture& c, uint32_t nameId) {
    if (nameId >= c.names.names.size()) return nullptr;   // includes kRootName (0xFFFFFFFF)
    return &c.names.names[nameId];
}

}  // namespace hf::editor
