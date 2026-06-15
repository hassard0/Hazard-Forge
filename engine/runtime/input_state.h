#pragma once
// Backend-agnostic input snapshot (NO SDL / NO GPU). Lives in engine/runtime so the pure-C++ core
// (hf_core: Camera + FlyCameraController) can consume it without dragging in any HAL/SDL types. The
// HAL (engine/hal/window.cpp) is the ONLY place that fills this from SDL each PumpEvents.

namespace hf::runtime {

// Small key set the fly camera + editor need. Extend as new bindings are required; the HAL maps the
// corresponding SDL scancodes into keyDown[] each pump.
enum class Key {
    W, A, S, D,
    Q, E,
    Space, Ctrl, Shift,
    Esc,
    Count
};

// One frame's worth of input. mouseDx/dy and wheel are ACCUMULATED deltas since the previous pump
// (reset at the start of each pump); keyDown[] / mouseButtons[] are the level state at pump time.
struct InputState {
    bool  keyDown[static_cast<int>(Key::Count)] = {};
    float mouseDx = 0.0f;      // relative mouse motion since last pump (pixels)
    float mouseDy = 0.0f;
    bool  mouseButtons[3] = {};  // 0 = left, 1 = right, 2 = middle
    float wheel = 0.0f;          // accumulated wheel ticks since last pump
    bool  relativeMouse = false; // true while mouse-look (relative) mode is engaged

    bool Down(Key k) const { return keyDown[static_cast<int>(k)]; }
};

} // namespace hf::runtime
