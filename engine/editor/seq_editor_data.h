#pragma once
// Hazard Forge — cinematic-sequencer TIMELINE VIEW DATA model (pure CPU, ImGui-free, backend-free).
//
// Issue #25 (Cinematic Sequencer / timeline), the EDITOR half: the deterministic SEAM between "how a
// hf::seq::Sequence should be DRAWN as a timeline" and the Dear ImGui calls that draw it. This is the
// EXACT flow_editor_data.{h} / profiler_view_data.{h} discipline applied to the sequencer runtime
// (engine/seq/seq.h): BuildSeqTimelineView computes a byte-identical VIEW (one horizontal TRACK LANE per
// ScalarTrack on a shared time axis, KEYFRAME markers as positioned diamonds, the sampled interpolation
// CURVE per track as a polyline of SampleScalar, and a PLAYHEAD line) from a Sequence + a playhead time,
// with ZERO ImGui / rhi / backend symbols, so it is unit-tested HEADLESSLY (assert the layout + a pinned
// FNV-1a-64 digest, NOT pixels) and lives in hf_core.
//
// THE DETERMINISM CONTRACT: the sequencer runtime is already bit-exact Q16.16 (the moat — UE5 Sequencer is
// float to the bone). This VIEW keeps that property end-to-end: every laid-out coordinate is a PURE INTEGER
// function of (sequence, playhead, layout params). Time -> X and value -> Y are integer maps computed by a
// Q16.16-domain proportion folded into a pixel span (an int64 numerator / a positive span, rounded with
// +0x8000 >> 16 just like the widget S4 quantize) — NO float on the layout path, NO <cmath>. The sampled
// CURVE is itself SampleScalar (bit-exact integer keyframe interpolation), evaluated at N fixed ticks across
// the time range, each result mapped value->Y by the SAME integer map. Same Sequence+playhead -> a
// byte-identical SeqTimelineView -> a deterministic golden. DigestSeqTimelineView pins it (hand little-endian
// -> net::DigestBytes, the flow.h / profile.h discipline). UE5 Sequencer's timeline layout is editor-state /
// mouse-driven (no two layouts agree on a machine, let alone across machines); this is a code-driven VALUE.
//
// Touches ONLY seq/seq.h (Sequence / ScalarTrack / SampleScalar / fx-Q16.16) and net/session.h
// (DigestBytes — pulled in transitively by seq.h, included explicitly for clarity). NO vk*/Metal/rhi
// rendering symbols, NO imgui.h, NO <cmath>/<algorithm>/float/clock/RNG/hash containers.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "seq/seq.h"          // hf::seq::Sequence / ScalarTrack / SampleScalar / fx (Q16.16) / kOne
#include "net/session.h"      // hf::net::DigestBytes — the pinned-golden FNV-1a-64 currency

namespace hf::editor {

// Bring the sequencer Q16.16 scalar + the sample fn into this namespace (the exact names seq.h defines).
using hf::seq::fx;            // Q16.16 fixed-point scalar (int32_t)
using hf::seq::kOne;          // 1.0 in Q16.16 (65536)
using hf::seq::Sequence;      // multi-track timeline
using hf::seq::ScalarTrack;   // one channel: parallel (times, values) in Q16.16, strictly-ascending times
using hf::seq::SampleScalar;  // the deterministic integer keyframe interpolation

// ---- Layout parameters (fixed integer grid; the layout is a deterministic VALUE) -------------------
// A vertical stack of equal-height TRACK LANES sharing one horizontal TIME AXIS. timeAxisW pixels span the
// WHOLE time range [tMin, tMax] (the union of all tracks' key times); laneH pixels span each track's value
// range [vMin, vMax]. All integers -> byte-stable cross-platform (no float on the layout path).
struct SeqLayout {
    int originX     = 60;    // left margin: where t == tMin maps (the lane left edge / value axis)
    int originY     = 48;    // top margin: top of lane 0
    int timeAxisW   = 900;   // pixel width the full time range [tMin, tMax] spans (shared by every lane)
    int laneH       = 120;   // height of one track lane
    int laneGap     = 24;    // vertical gap between lanes
    int laneInsetY  = 12;    // top+bottom inset inside a lane so value extremes don't touch the lane edge
    uint32_t curveSteps = 96;// number of SampleScalar samples per track for the interpolation polyline
};

// One drawn track lane: its box on the integer grid + which track it renders. Lanes stack top-to-bottom.
struct SeqTrackLane {
    int      x = 0, y = 0, w = 0, h = 0;   // lane box top-left + size (integer grid)
    uint32_t trackIndex = 0;               // index into Sequence.tracks
};

// One drawn keyframe marker (a diamond): the pixel center (x,y) of a single keyframe, plus the source
// indices + the raw Q16.16 (time,value) it came from (the panel labels them; geometry is x,y).
struct SeqKeyMarker {
    int      x = 0, y = 0;                  // pixel center of the keyframe diamond
    uint32_t trackIndex = 0;               // index into Sequence.tracks
    uint32_t keyIndex   = 0;               // index into that track's times/values
    fx       timeFx  = 0;                  // the keyframe time  (Q16.16) — labels the marker
    fx       valueFx = 0;                  // the keyframe value (Q16.16) — labels the marker
};

// One drawn curve point: a vertex of a track's sampled interpolation polyline. The points for a track are
// emitted in ascending time order (curveSteps+1 of them, the last at t == tMax). trackIndex groups them.
struct SeqCurvePoint {
    int      x = 0, y = 0;                  // pixel position of the sampled point
    uint32_t trackIndex = 0;               // which track's polyline this vertex belongs to
};

// The complete, deterministic timeline view laid out for the editor. Two calls on the identical
// (sequence, playhead) yield BYTE-IDENTICAL views (determinism); DigestSeqTimelineView pins it as the golden.
struct SeqTimelineView {
    std::vector<SeqTrackLane>  lanes;   // one per track, in track order
    std::vector<SeqKeyMarker>  keys;    // every keyframe of every track, in (track asc, key asc) order
    std::vector<SeqCurvePoint> curve;   // every track's polyline vertices, (track asc, step asc) order
    int playheadX  = 0;                 // pixel X of the playhead (playheadTime mapped via the time axis)
    int timeAxisW  = 0;                 // pixel width the full time range spans (== layout.timeAxisW)
    fx  tMinFx = 0, tMaxFx = 0;         // the time range mapped onto the axis (Q16.16; labels the ruler)
};

// ---- MapTimeToX: map a Q16.16 time t in [tMin, tMax] to a pixel X in [originX, originX+timeAxisW]. -----
// Pure integer. The proportion (t - tMin) / (tMax - tMin) folded into the pixel span: an int64 numerator
// (t-tMin)*timeAxisW divided by the positive span (tMax-tMin), rounded to nearest (+half the denominator).
// A degenerate span (tMax <= tMin) maps everything to originX (deterministic). t is clamped to the range so
// a playhead before/after the keyed range pins to an axis end.
inline int MapTimeToX(fx t, fx tMin, fx tMax, int originX, int timeAxisW) {
    if (tMax <= tMin) return originX;                       // degenerate range -> left edge
    if (t < tMin) t = tMin;
    if (t > tMax) t = tMax;
    const int64_t span = static_cast<int64_t>(tMax) - static_cast<int64_t>(tMin);  // > 0
    const int64_t num  = (static_cast<int64_t>(t) - static_cast<int64_t>(tMin)) *
                         static_cast<int64_t>(timeAxisW);
    const int64_t px   = (num + span / 2) / span;           // round to nearest pixel
    return originX + static_cast<int>(px);
}

// ---- MapValueToY: map a Q16.16 value v in [vMin, vMax] to a pixel Y inside a lane's drawable band. ------
// Pure integer, AXIS-FLIPPED (larger value -> SMALLER Y, the screen-up convention). The drawable band is
// [laneTop+inset, laneTop+laneH-inset]; v==vMax maps to the band top, v==vMin to the band bottom. A
// degenerate value range (vMax <= vMin — a constant track) centers the line in the band. v is clamped.
inline int MapValueToY(fx v, fx vMin, fx vMax, int laneTop, int laneH, int inset) {
    const int bandTop = laneTop + inset;
    const int bandH   = laneH - 2 * inset;                  // drawable height
    if (bandH <= 0) return laneTop + laneH / 2;             // no room -> lane center
    if (vMax <= vMin) return bandTop + bandH / 2;           // constant track -> band center
    if (v < vMin) v = vMin;
    if (v > vMax) v = vMax;
    // proportion from the TOP = (vMax - v) / (vMax - vMin); fold into bandH (int64, rounded to nearest).
    const int64_t span = static_cast<int64_t>(vMax) - static_cast<int64_t>(vMin);  // > 0
    const int64_t num  = (static_cast<int64_t>(vMax) - static_cast<int64_t>(v)) *
                         static_cast<int64_t>(bandH);
    const int64_t off  = (num + span / 2) / span;           // round to nearest pixel
    return bandTop + static_cast<int>(off);
}

// ---- TrackValueRange: the [min,max] of a track's keyframe values (the per-lane vertical scale). --------
// A pure scan over values; an empty track yields [0,0] (degenerate -> MapValueToY centers). The curve
// stays inside this band because SampleScalar clamps to [front,back] and (for Step/Linear/the eased modes
// whose LUT endpoints are exact) never overshoots the keyed value extremes — a + w*(b-a) with w in [0,kOne]
// stays within [min(a,b), max(a,b)] for the monotone Linear/Step/Quad eases; EaseInOutSine's LUT is also
// monotone in [0,kOne] so it does not overshoot either. (Were a future ease to overshoot, the clamp in
// MapValueToY keeps every curve vertex inside the band — still deterministic, just visually clipped.)
inline void TrackValueRange(const ScalarTrack& tr, fx& vMin, fx& vMax) {
    if (tr.values.empty()) { vMin = 0; vMax = 0; return; }
    vMin = tr.values[0];
    vMax = tr.values[0];
    for (std::size_t i = 1; i < tr.values.size(); ++i) {
        const fx v = tr.values[i];
        if (v < vMin) vMin = v;
        if (v > vMax) vMax = v;
    }
}

// ---- SequenceTimeRange: the [min,max] of all tracks' key times (the shared horizontal axis). -----------
// The union across every track (each track's times is strictly ascending, so front()=min, back()=max for
// that track). An all-empty sequence yields [0, kOne] (a 1-second default axis, deterministic + non-
// degenerate so the playhead/ruler still lay out). A single time across the whole sequence widens to
// [t, t+kOne] so the axis is non-degenerate.
inline void SequenceTimeRange(const Sequence& s, fx& tMin, fx& tMax) {
    bool any = false;
    tMin = 0; tMax = 0;
    for (const ScalarTrack& tr : s.tracks) {
        if (tr.times.empty()) continue;
        const fx lo = tr.times.front();
        const fx hi = tr.times.back();
        if (!any) { tMin = lo; tMax = hi; any = true; }
        else {
            if (lo < tMin) tMin = lo;
            if (hi > tMax) tMax = hi;
        }
    }
    if (!any) { tMin = 0; tMax = kOne; return; }     // empty sequence -> a default 1s axis
    if (tMax <= tMin) tMax = tMin + kOne;            // single-instant sequence -> widen so the axis is real
}

// ---- BuildSeqTimelineView: lay a Sequence out as a cinematic timeline (THE pure deterministic builder) --
//
// LANES: one SeqTrackLane per track, stacked top-to-bottom at a FIXED stride (originY + i*(laneH+laneGap)),
// each spanning [originX, originX+timeAxisW] horizontally. The track count drives the layout; a track's
// vertical SCALE is its own value range (so a small-amplitude channel still fills its lane).
//
// KEYS: every keyframe of every track emits one SeqKeyMarker at (time->X via the shared time axis, value->Y
// within that track's lane band), in (track ascending, key ascending) order — a fixed, hash-free order.
//
// CURVE: each track emits curveSteps+1 SeqCurvePoint vertices in ascending time order — SampleScalar(tr, t)
// at t = tMin + i*(tMax-tMin)/curveSteps for i in [0, curveSteps] (the last at t == tMax). The sampled
// value maps Y via the SAME per-lane integer map as the keys, so the curve passes through the markers. This
// is the bit-exact integer interpolation drawn as a polyline — the runtime moat made visible.
//
// PLAYHEAD: playheadTime mapped to X via the shared time axis (clamped into the range). One X for all lanes.
inline SeqTimelineView BuildSeqTimelineView(const Sequence& s, fx playheadTime,
                                            const SeqLayout& L = SeqLayout{}) {
    SeqTimelineView view;
    view.timeAxisW = L.timeAxisW;

    // Shared horizontal time axis: the union of all tracks' key-time ranges.
    fx tMin, tMax;
    SequenceTimeRange(s, tMin, tMax);
    view.tMinFx = tMin;
    view.tMaxFx = tMax;

    const std::size_t nTracks = s.tracks.size();

    // --- Lanes + (per-lane) keys + curve, in track order. -------------------------------------------
    for (std::size_t ti = 0; ti < nTracks; ++ti) {
        const ScalarTrack& tr = s.tracks[ti];

        // Lane box.
        SeqTrackLane lane;
        lane.x = L.originX;
        lane.y = L.originY + static_cast<int>(ti) * (L.laneH + L.laneGap);
        lane.w = L.timeAxisW;
        lane.h = L.laneH;
        lane.trackIndex = static_cast<uint32_t>(ti);
        view.lanes.push_back(lane);

        // Per-lane vertical scale = this track's value range.
        fx vMin, vMax;
        TrackValueRange(tr, vMin, vMax);

        // Keyframe markers: one diamond per (time,value) pair.
        for (std::size_t ki = 0; ki < tr.times.size(); ++ki) {
            SeqKeyMarker m;
            m.trackIndex = static_cast<uint32_t>(ti);
            m.keyIndex   = static_cast<uint32_t>(ki);
            m.timeFx     = tr.times[ki];
            m.valueFx    = tr.values[ki];
            m.x = MapTimeToX(m.timeFx, tMin, tMax, L.originX, L.timeAxisW);
            m.y = MapValueToY(m.valueFx, vMin, vMax, lane.y, L.laneH, L.laneInsetY);
            view.keys.push_back(m);
        }

        // Interpolation curve: curveSteps+1 samples across the WHOLE shared axis (so every lane's curve
        // spans the full width; a track keyed on a sub-range holds its end values per SampleScalar's clamp).
        const uint32_t steps = (L.curveSteps == 0u) ? 1u : L.curveSteps;
        const int64_t  span  = static_cast<int64_t>(tMax) - static_cast<int64_t>(tMin);  // > 0 by SequenceTimeRange
        for (uint32_t i = 0; i <= steps; ++i) {
            // t = tMin + i*span/steps, formed in int64 (avoids overflow), exact at i==0 and i==steps.
            const fx t = static_cast<fx>(static_cast<int64_t>(tMin) +
                                         (static_cast<int64_t>(i) * span) / static_cast<int64_t>(steps));
            const fx v = SampleScalar(tr, t);
            SeqCurvePoint cp;
            cp.trackIndex = static_cast<uint32_t>(ti);
            cp.x = MapTimeToX(t, tMin, tMax, L.originX, L.timeAxisW);
            cp.y = MapValueToY(v, vMin, vMax, lane.y, L.laneH, L.laneInsetY);
            view.curve.push_back(cp);
        }
    }

    // --- Playhead: one vertical line across all lanes at playheadTime -> X. --------------------------
    view.playheadX = MapTimeToX(playheadTime, tMin, tMax, L.originX, L.timeAxisW);

    return view;
}

// ---- DigestSeqTimelineView: FNV-1a-64 over a HAND little-endian serialization of the view (the golden) --
// HAND-LE field by field (NEVER memcpy the structs — padding/endianness-unsafe; the flow.h / profile.h /
// seq.h discipline) so the digest is byte-stable cross-platform. Encodes the axis (playheadX, timeAxisW,
// tMin, tMax), then every lane (x,y,w,h,trackIndex), every key (x,y,trackIndex,keyIndex,timeFx,valueFx),
// and every curve point (x,y,trackIndex). int + fx values are encoded as their two's-complement uint32 LE
// bits. A single moved/added/deleted keyframe changes the view -> changes the digest (proven by the test).
inline uint64_t DigestSeqTimelineView(const SeqTimelineView& v) {
    std::vector<unsigned char> buf;
    buf.reserve(v.lanes.size() * 20u + v.keys.size() * 24u + v.curve.size() * 12u + 24u);
    auto putU32 = [&](uint32_t x) {
        buf.push_back(static_cast<unsigned char>( x        & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 8)  & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 16) & 0xFFu));
        buf.push_back(static_cast<unsigned char>((x >> 24) & 0xFFu));
    };
    auto putI32 = [&](int x)  { putU32(static_cast<uint32_t>(x)); };
    auto putFx  = [&](fx x)   { putU32(static_cast<uint32_t>(x)); };

    putI32(v.playheadX);
    putI32(v.timeAxisW);
    putFx(v.tMinFx);
    putFx(v.tMaxFx);

    putU32(static_cast<uint32_t>(v.lanes.size()));
    for (const SeqTrackLane& ln : v.lanes) {
        putI32(ln.x); putI32(ln.y); putI32(ln.w); putI32(ln.h);
        putU32(ln.trackIndex);
    }
    putU32(static_cast<uint32_t>(v.keys.size()));
    for (const SeqKeyMarker& k : v.keys) {
        putI32(k.x); putI32(k.y);
        putU32(k.trackIndex);
        putU32(k.keyIndex);
        putFx(k.timeFx);
        putFx(k.valueFx);
    }
    putU32(static_cast<uint32_t>(v.curve.size()));
    for (const SeqCurvePoint& c : v.curve) {
        putI32(c.x); putI32(c.y);
        putU32(c.trackIndex);
    }
    return hf::net::DigestBytes(buf.data(), buf.size());
}

}  // namespace hf::editor
