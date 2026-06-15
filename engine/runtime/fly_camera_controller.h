#pragma once
// Free-fly camera controller (hf_core: pure C++, NO SDL/GPU). Maps an InputState + dt onto a
// Camera: WASD for planar move, up/down keys for vertical, mouse delta for yaw/pitch (mouse-look),
// wheel to adjust move speed. Takes InputState by const ref so it is fully unit-testable with a
// synthetic input snapshot.

#include "runtime/camera.h"
#include "runtime/input_state.h"

namespace hf::runtime {

class FlyCameraController {
public:
    float moveSpeed = 6.0f;        // units/second (adjusted by the wheel)
    float sprintMultiplier = 4.0f; // Shift
    float lookSensitivity = 0.0025f; // radians per pixel of mouse motion
    float minSpeed = 0.5f;
    float maxSpeed = 80.0f;

    // Advance `cam` from one input snapshot over dt seconds.
    void Update(Camera& cam, const InputState& in, float dt);

private:
    bool prevRelative_ = false;  // swallow the first delta after entering relative mode
};

} // namespace hf::runtime
