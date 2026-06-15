#include "runtime/fly_camera_controller.h"

namespace hf::runtime {

void FlyCameraController::Update(Camera& cam, const InputState& in, float dt) {
    // --- Wheel adjusts the base move speed (multiplicative, clamped to a sane range). ---
    if (in.wheel != 0.0f) {
        moveSpeed *= std::pow(1.1f, in.wheel);
        if (moveSpeed < minSpeed) moveSpeed = minSpeed;
        if (moveSpeed > maxSpeed) moveSpeed = maxSpeed;
    }

    // --- Mouse-look: only while relative mouse mode is engaged. Swallow the first frame's delta
    //     right after entering relative mode (SDL can report a large jump on the transition). ---
    if (in.relativeMouse) {
        if (prevRelative_) {
            cam.AddYawPitch(in.mouseDx * lookSensitivity,
                            -in.mouseDy * lookSensitivity);  // up = look up (invert screen-y)
        }
        prevRelative_ = true;
    } else {
        prevRelative_ = false;
    }

    // --- Translation: forward/right from the camera basis, up is world-space. ---
    math::Vec3 fwd = cam.Forward();
    math::Vec3 right = cam.Right();
    math::Vec3 worldUp{0.0f, 1.0f, 0.0f};

    math::Vec3 move{0.0f, 0.0f, 0.0f};
    if (in.Down(Key::W)) move = move + fwd;
    if (in.Down(Key::S)) move = move - fwd;
    if (in.Down(Key::D)) move = move + right;
    if (in.Down(Key::A)) move = move - right;
    if (in.Down(Key::E) || in.Down(Key::Space)) move = move + worldUp;
    if (in.Down(Key::Q) || in.Down(Key::Ctrl))  move = move - worldUp;

    float len = math::length(move);
    if (len > 1e-6f) {
        move = move / len;  // normalize so diagonal isn't faster
        float speed = moveSpeed * (in.Down(Key::Shift) ? sprintMultiplier : 1.0f);
        cam.position = cam.position + move * (speed * dt);
    }
}

} // namespace hf::runtime
