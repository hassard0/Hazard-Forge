#include "editor/profiler_view_panels.h"

#include "imgui.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace hf::editor {

// --- Docked layout, built PROGRAMMATICALLY (deterministic; no imgui.ini). Mirrors editor_panels.cpp /
// flow_editor_panels.cpp's BeginDocked: fixed rect, no move/resize/collapse, no bring-to-front shuffle, so
// neither cursor input nor a persisted .ini can perturb the frame — the profiler view is byte-stable run-to-
// run + cross-backend.
namespace {

bool BeginDocked(const char* title, ImVec2 pos, ImVec2 size, ImGuiWindowFlags extra = 0) {
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse |
                                   ImGuiWindowFlags_NoBringToFrontOnFocus | extra;
    return ImGui::Begin(title, nullptr, flags);
}

// Copy a NameTable byte-string into a NUL-terminated buffer for ImGui's %s. Names are raw bytes (no <string>);
// an empty / out-of-range name renders as "<scope>" so the row is still legible. Deterministic.
void NameToBuf(const profile::Capture& cap, uint32_t nameId, char* out, std::size_t cap_n) {
    const std::vector<std::uint8_t>* nm = ResolveName(cap, nameId);
    if (!nm || nm->empty()) { std::snprintf(out, cap_n, "<scope>"); return; }
    std::size_t n = nm->size();
    if (n > cap_n - 1) n = cap_n - 1;
    for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<char>((*nm)[i]);
    out[n] = '\0';
}

// A frame cell's fill color by index parity so adjacent cells read apart (deterministic — no time dependence).
ImU32 FrameColor(std::size_t i) {
    return (i & 1u) ? IM_COL32(58, 96, 120, 235) : IM_COL32(70, 110, 135, 235);
}

// A scope bar's fill color cycles by depth so the hierarchy reads (deterministic palette).
ImU32 ScopeColor(int depth) {
    static const ImU32 pal[4] = {
        IM_COL32(95, 130, 75, 235),    // green
        IM_COL32(120, 95, 60, 235),    // amber
        IM_COL32(80, 95, 135, 235),    // blue
        IM_COL32(110, 75, 120, 235),   // purple
    };
    return pal[(depth >= 0 ? depth : 0) & 3];
}

}  // namespace

void BuildProfilerViewUI(const profile::Capture& capture, const ProfilerView& view,
                         uint32_t fbWidth, uint32_t fbHeight, const ProfLayout& layout) {
    (void)layout;
    const float fbW = static_cast<float>(fbWidth);
    const float fbH = static_cast<float>(fbHeight);

    // --- Menu bar (fixed; renders fine without input). Reserve its height for the tiling below. ---
    float menuH = 0.0f;
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("Hazard Forge")) {
            ImGui::MenuItem("Profiler", nullptr, true);
            ImGui::EndMenu();
        }
        ImGui::TextDisabled("  |  frame timeline  |  scope tree  |  draw-call inspection");
        menuH = ImGui::GetWindowSize().y;
        ImGui::EndMainMenuBar();
    }

    const float top = menuH;
    const float bodyH = fbH - top;

    // --- The single docked profiler window filling the frame. The frame cells, scope bars, and draw rows
    // come straight from the deterministic ProfilerView (CPU-built); we draw via the window draw list. ---
    if (BeginDocked("Profiler", ImVec2(0.0f, top), ImVec2(fbW, bodyH),
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
        ImDrawList* dl = ImGui::GetWindowDrawList();
        const ImVec2 o = ImGui::GetCursorScreenPos();
        auto P = [&](int x, int y) { return ImVec2(o.x + static_cast<float>(x),
                                                   o.y + static_cast<float>(y)); };

        // ===== Section 1: FRAME TIMELINE (the row of frame cells). =====
        dl->AddText(P(layout.originX, layout.timelineY - 22), IM_COL32(225, 230, 240, 255),
                    "Frame Timeline");
        for (std::size_t i = 0; i < view.frames.size(); ++i) {
            const ProfFrameCell& f = view.frames[i];
            const ImVec2 tl = P(f.x, f.y);
            const ImVec2 br = P(f.x + f.w, f.y + f.h);
            dl->AddRectFilled(tl, br, FrameColor(i), 4.0f);
            dl->AddRect(tl, br, IM_COL32(225, 230, 240, 255), 4.0f, 0, 1.5f);
            char l0[48];
            std::snprintf(l0, sizeof(l0), "Frame %u", f.frameNumber);
            dl->AddText(ImVec2(tl.x + 8.0f, tl.y + 6.0f), IM_COL32(245, 245, 245, 255), l0);
            char l1[64];
            // cpu nanos -> microseconds for the label (integer divide; display only).
            std::snprintf(l1, sizeof(l1), "%llu us",
                          static_cast<unsigned long long>(f.cpuNanos / 1000ull));
            dl->AddText(ImVec2(tl.x + 8.0f, tl.y + 28.0f), IM_COL32(210, 220, 235, 255), l1);
        }

        // ===== Section 2: SCOPE TREE (indented cpu-proportional bars). =====
        dl->AddText(P(layout.originX, layout.scopeY - 22), IM_COL32(225, 230, 240, 255),
                    "Scope Tree (CPU time)");
        for (const ProfScopeRow& s : view.scopes) {
            const ImVec2 tl = P(s.x, s.y);
            // The bar (cpu-time proportional). A zero-cpu scope still shows a thin stub so it is visible.
            const int barW = s.w > 0 ? s.w : 4;
            const ImVec2 br = P(s.x + barW, s.y + s.h);
            dl->AddRectFilled(tl, br, ScopeColor(s.depth), 3.0f);
            dl->AddRect(tl, br, IM_COL32(215, 220, 230, 220), 3.0f, 0, 1.0f);
            char nm[64];
            NameToBuf(capture, s.nameId, nm, sizeof(nm));
            char label[128];
            std::snprintf(label, sizeof(label), "%s  (%llu us, %u draws)",
                          nm, static_cast<unsigned long long>(s.cpuNanos / 1000ull),
                          s.subtreeDrawCount);
            // Label to the RIGHT of the bar so a short bar's text is never clipped by the fill.
            dl->AddText(ImVec2(br.x + 8.0f, tl.y + 1.0f), IM_COL32(240, 242, 248, 255), label);
        }

        // ===== Section 3: DRAW-CALL inspection list. =====
        dl->AddText(P(layout.originX, layout.drawY - 22), IM_COL32(225, 230, 240, 255),
                    "Draw Calls (per pass)");
        for (const ProfDrawRow& d : view.draws) {
            char nm[64];
            NameToBuf(capture, d.passNameId, nm, sizeof(nm));
            char row[128];
            std::snprintf(row, sizeof(row), "%-12s  draws: %u", nm, d.drawCount);
            const ImVec2 p = P(d.x, d.y);
            // A small bullet pin + the pass/draw text.
            dl->AddCircleFilled(ImVec2(p.x + 4.0f, p.y + 8.0f), 3.0f, IM_COL32(210, 180, 120, 255));
            dl->AddText(ImVec2(p.x + 16.0f, p.y), IM_COL32(235, 230, 215, 255), row);
        }
    }
    ImGui::End();
}

}  // namespace hf::editor
