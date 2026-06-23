#pragma once
// Backend-agnostic input snapshot (NO SDL / NO GPU). Lives in engine/runtime so the pure-C++ core
// (hf_core: Camera + FlyCameraController) can consume it without dragging in any HAL/SDL types. The
// HAL (engine/hal/window.cpp) is the ONLY place that fills this from SDL each PumpEvents.

namespace hf::runtime {

// The platform-agnostic key set the fly camera, editor, AND downstream samples bind. The HAL
// (engine/hal/window.cpp, SDL) and mac_window (AppKit) map the corresponding platform scancodes into
// keyDown[] each pump. APPEND-ONLY: keyDown[] is indexed by this enum, so new keys go before Count and
// existing entries never move (issue #2 — the set now covers the full alphabet, digit row, arrows,
// function keys, and common editing keys so a sample never has to intercept raw platform keycodes).
enum class Key {
    W, A, S, D,
    Q, E,
    Space, Ctrl, Shift,
    Esc,
    P, O, G, R, T,   // editor: P=play/pause toggle, O=single step, G/R/T=gizmo mode, plus Ctrl+S save
    // --- extended binding set (issue #2) ---
    B, C, F, H, I, J, K, L, M, N, U, V, X, Y, Z,                 // the rest of A-Z
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,  // the digit row
    Left, Right, Up, Down,                                        // arrow keys
    Tab, Enter, Backspace,                                        // common editing keys
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,            // function keys
    Count
};

// One frame's worth of input. mouseDx/dy and wheel are ACCUMULATED deltas since the previous pump
// (reset at the start of each pump); keyDown[] / mouseButtons[] are the level state at pump time.
struct InputState {
    bool  keyDown[static_cast<int>(Key::Count)] = {};
    float mouseDx = 0.0f;      // relative mouse motion since last pump (pixels)
    float mouseDy = 0.0f;
    float mouseX = 0.0f;       // ABSOLUTE cursor position in framebuffer pixels (origin top-left,
    float mouseY = 0.0f;       // +x right, +y down). Meaningful only while NOT in relativeMouse
                               // (mouse-look) mode; the editor unprojects a left-click from these.
    bool  mouseButtons[3] = {};  // 0 = left, 1 = right, 2 = middle
    float wheel = 0.0f;          // accumulated wheel ticks since last pump
    bool  relativeMouse = false; // true while mouse-look (relative) mode is engaged

    bool Down(Key k) const { return keyDown[static_cast<int>(k)]; }
};

} // namespace hf::runtime
