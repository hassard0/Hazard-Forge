#include "editor/seq_editor_panels.h"

#include "imgui.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace hf::editor {

// --- Docked layout, built PROGRAMMATICALLY (deterministic; no imgui.ini). Mirrors flow_editor_panels.cpp's
// BeginDocked: fixed rect, no move/resize/collapse, no bring-to-front shuffle, so neither cursor input nor a
// persisted .ini can perturb the frame — the timeline editor is byte-stable run-to-run + cross-backend.
namespace {

bool BeginDocked(const char* title, ImVec2 pos, ImVec2 size, ImGuiWindowFlags extra = 0) {
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus | extra;
    return ImGui::Begin(title, nullptr, flags);
}

// A per-track curve/lane accent color, cycled by track index (deterministic — no input/time dependence).
// Warm-to-cool palette so stacked lanes read distinctly (camera/translation/etc. channels).
ImU32 TrackColor(uint32_t trackIndex) {
    switch (trackIndex % 5u) {
        case 0u: return IM_COL32(230, 170, 90, 255);    // amber
        case 1u: return IM_COL32(110, 200, 130, 255);   // green
        case 2u: return IM_COL32(120, 175, 235, 255);   // blue
        case 3u: return IM_COL32(210, 130, 200, 255);   // magenta
        default: return IM_COL32(120, 210, 205, 255);   // teal
    }
}

// The easing label for a track (the panel annotates each lane with its interpolation mode).
const char* EasingLabel(hf::seq::Easing e) {
    switch (e) {
        case hf::seq::Easing::Step:          return "Step";
        case hf::seq::Easing::Linear:        return "Linear";
        case hf::seq::Easing::EaseInOutSine: return "EaseInOutSine";
        case hf::seq::Easing::EaseInQuad:    return "EaseInQuad";
        case hf::seq::Easing::EaseOutQuad:   return "EaseOutQuad";
        default:                             return "Linear";
    }
}

// Q16.16 -> a printable value with 2 decimals (label-only; NOT on the deterministic layout path). Uses the
// rounded integer part + a thousandths fraction so the panel reads e.g. "1.00", "-0.50".
void FormatFx(char* out, std::size_t cap, hf::seq::fx v) {
    const bool neg = v < 0;
    int64_t a = neg ? -static_cast<int64_t>(v) : static_cast<int64_t>(v);
    const int64_t whole = a >> 16;
    const int64_t milli = ((a & 0xFFFF) * 1000 + 32768) >> 16;   // 3-digit fraction, rounded
    std::snprintf(out, cap, "%s%lld.%03lld", neg ? "-" : "", static_cast<long long>(whole),
                  static_cast<long long>(milli));
}

}  // namespace

void BuildSeqEditorUI(const Sequence& seq, const SeqTimelineView& view,
                      uint32_t fbWidth, uint32_t fbHeight, const SeqLayout& layout) {
    const float fbW = static_cast<float>(fbWidth);
    const float fbH = static_cast<float>(fbHeight);

    // --- Menu bar (fixed; renders fine without input). Reserve its height for the tiling below. ---
    float menuH = 0.0f;
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Hazard Forge")) {
            ImGui::MenuItem("Cinematic Sequencer", nullptr, true);
            ImGui::EndMenu();
        }
        ImGui::TextDisabled("  |  timeline editor  |  deterministic Q16.16 keyframes");
        menuH = ImGui::GetWindowSize().y;
        ImGui::EndMainMenuBar();
    }

    const float top = menuH;
    const float bodyH = fbH - top;
    const float trackStripW = fbW * 0.18f;     // left track strip
    const float canvasX = trackStripW;
    const float canvasW = fbW - trackStripW;

    // --- TRACK strip (left): one row per track with its easing + key count. The headless capture lists
    // them; in an interactive editor this is where add-track / select-track affordances live. ---
    if (BeginDocked("Tracks", ImVec2(0.0f, top), ImVec2(trackStripW, bodyH))) {
        ImGui::TextUnformatted("Sequence Tracks");
        ImGui::Separator();
        for (std::size_t ti = 0; ti < seq.tracks.size(); ++ti) {
            const hf::seq::ScalarTrack& tr = seq.tracks[ti];
            ImGui::PushStyleColor(ImGuiCol_Text, TrackColor(static_cast<uint32_t>(ti)));
            char row[64];
            std::snprintf(row, sizeof(row), "Track %zu  (%s)", ti, EasingLabel(tr.easing));
            ImGui::Selectable(row, false);
            ImGui::PopStyleColor();
            ImGui::TextDisabled("    keys: %zu", tr.times.size());
        }
        ImGui::Separator();
        ImGui::TextDisabled("Tracks: %zu", view.lanes.size());
        ImGui::TextDisabled("Keys: %zu", view.keys.size());
        ImGui::TextDisabled("Curve pts: %zu", view.curve.size());
    }
    ImGui::End();

    // --- CANVAS: the timeline. Lanes (alternating bg) -> grid -> curves -> keyframe diamonds -> playhead
    // -> time ruler, via the window draw list so geometry comes straight from the deterministic view. ---
    if (BeginDocked("Timeline", ImVec2(canvasX, top), ImVec2(canvasW, bodyH),
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 o = ImGui::GetCursorScreenPos();
        auto P = [&](int x, int y) { return ImVec2(o.x + static_cast<float>(x),
                                                   o.y + static_cast<float>(y)); };

        const int axisL = layout.originX;
        const int axisR = layout.originX + view.timeAxisW;

        // Lanes: alternating background fill + a baseline midline + the track/easing label.
        for (const SeqTrackLane& ln : view.lanes) {
            const ImVec2 tl = P(ln.x, ln.y);
            const ImVec2 br = P(ln.x + ln.w, ln.y + ln.h);
            const ImU32 bg = (ln.trackIndex & 1u) ? IM_COL32(34, 38, 48, 235)
                                                  : IM_COL32(28, 31, 40, 235);
            dl->AddRectFilled(tl, br, bg, 4.0f);
            dl->AddRect(tl, br, IM_COL32(70, 78, 92, 255), 4.0f, 0, 1.0f);
            // Lane label (track index + easing) in the lane's accent color.
            const hf::seq::ScalarTrack& tr =
                (ln.trackIndex < seq.tracks.size()) ? seq.tracks[ln.trackIndex] : seq.tracks[0];
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "Track %u  %s", ln.trackIndex, EasingLabel(tr.easing));
            dl->AddText(ImVec2(tl.x + 6.0f, tl.y + 4.0f), TrackColor(ln.trackIndex), lbl);
        }

        // Per-track interpolation CURVE: connect consecutive SeqCurvePoint of the same trackIndex. The
        // points are emitted (track asc, step asc), so a run of equal trackIndex is one polyline.
        for (std::size_t i = 1; i < view.curve.size(); ++i) {
            const SeqCurvePoint& a = view.curve[i - 1];
            const SeqCurvePoint& b = view.curve[i];
            if (a.trackIndex != b.trackIndex) continue;   // don't bridge across lanes
            dl->AddLine(P(a.x, a.y), P(b.x, b.y), TrackColor(a.trackIndex), 2.0f);
        }

        // Keyframe DIAMONDS: a filled rotated square at each marker, outlined for contrast.
        for (const SeqKeyMarker& k : view.keys) {
            const ImVec2 c = P(k.x, k.y);
            const float r = 5.0f;
            const ImVec2 pts[4] = { ImVec2(c.x, c.y - r), ImVec2(c.x + r, c.y),
                                    ImVec2(c.x, c.y + r), ImVec2(c.x - r, c.y) };
            dl->AddConvexPolyFilled(pts, 4, TrackColor(k.trackIndex));
            dl->AddPolyline(pts, 4, IM_COL32(245, 245, 245, 255), ImDrawFlags_Closed, 1.5f);
        }

        // PLAYHEAD: a vertical line across every lane at the playhead X (bright, drawn on top).
        if (!view.lanes.empty()) {
            const SeqTrackLane& first = view.lanes.front();
            const SeqTrackLane& last  = view.lanes.back();
            const ImVec2 a = P(view.playheadX, first.y - 6);
            const ImVec2 b = P(view.playheadX, last.y + last.h + 6);
            dl->AddLine(a, b, IM_COL32(255, 90, 90, 255), 2.0f);
            // A small playhead handle at the top.
            const ImVec2 h0 = P(view.playheadX - 5, first.y - 14);
            const ImVec2 h1 = P(view.playheadX + 5, first.y - 14);
            const ImVec2 h2 = P(view.playheadX, first.y - 4);
            const ImVec2 tri[3] = { h0, h1, h2 };
            dl->AddConvexPolyFilled(tri, 3, IM_COL32(255, 90, 90, 255));
        }

        // TIME RULER: a baseline at the bottom + tick labels at tMin / midpoint / tMax (label-only — the
        // tick X positions come straight from the deterministic axis).
        {
            const int rulerY = view.lanes.empty()
                                   ? layout.originY
                                   : (view.lanes.back().y + view.lanes.back().h + 16);
            dl->AddLine(P(axisL, rulerY), P(axisR, rulerY), IM_COL32(150, 156, 168, 255), 1.5f);
            auto tick = [&](int x, hf::seq::fx tFx) {
                dl->AddLine(P(x, rulerY - 4), P(x, rulerY + 4), IM_COL32(180, 186, 198, 255), 1.5f);
                char buf[24]; FormatFx(buf, sizeof(buf), tFx);
                char lbl[32]; std::snprintf(lbl, sizeof(lbl), "%ss", buf);
                dl->AddText(ImVec2(P(x, rulerY + 6).x - 8.0f, P(x, rulerY + 6).y),
                            IM_COL32(190, 196, 208, 255), lbl);
            };
            const hf::seq::fx tMid = static_cast<hf::seq::fx>(
                (static_cast<int64_t>(view.tMinFx) + static_cast<int64_t>(view.tMaxFx)) / 2);
            tick(axisL, view.tMinFx);
            tick(axisL + view.timeAxisW / 2, tMid);
            tick(axisR, view.tMaxFx);
        }
    }
    ImGui::End();
}

}  // namespace hf::editor
