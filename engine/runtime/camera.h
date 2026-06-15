#pragma once
// Backend-agnostic flyable camera (hf_core: pure math, no backend/SDL types). Centralizes the
// view/proj/basis math the showcases used to duplicate by hand. Yaw rotates around world +Y; pitch
// around the camera right axis. At yaw=0, pitch=0 forward = (0,0,-1) (RH, looking down -Z), which
// matches Mat4::LookAt and the existing fixed showcase cameras.

#include "math/math.h"

namespace hf::runtime {

// Camera-derived values the renderer's FrameData needs: the world-space basis + eye + sky params.
// The sample copies these straight into its FrameData UBO (camFwd/camRight/camUp/viewPos/skyParams),
// replacing the per-showcase hand-rolled basis computation.
struct CameraBasis {
    math::Vec3 forward;
    math::Vec3 right;
    math::Vec3 up;
    math::Vec3 position;
    float tanHalfFovY = 0.0f;
    float aspect = 1.0f;
};

class Camera {
public:
    math::Vec3 position{0.0f, 0.0f, 0.0f};
    float yaw = 0.0f;     // radians, around world +Y
    float pitch = 0.0f;   // radians, around camera right; clamped in SetPitch/AddPitch
    float fovY = 1.04719755f;  // 60 degrees
    float aspect = 16.0f / 9.0f;
    float znear = 0.1f;
    float zfar = 100.0f;

    Camera() = default;

    // World-space forward from yaw/pitch. yaw=0,pitch=0 -> (0,0,-1).
    math::Vec3 Forward() const {
        float cp = std::cos(pitch), sp = std::sin(pitch);
        float cy = std::cos(yaw),   sy = std::sin(yaw);
        return math::normalize(math::Vec3{cp * sy, sp, -cp * cy});
    }
    // Right is perpendicular to forward in the horizontal plane (world up = +Y).
    math::Vec3 Right() const {
        return math::normalize(math::cross(Forward(), math::Vec3{0.0f, 1.0f, 0.0f}));
    }
    math::Vec3 Up() const {
        return math::cross(Right(), Forward());
    }

    math::Mat4 View() const {
        return math::Mat4::LookAt(position, position + Forward(), math::Vec3{0.0f, 1.0f, 0.0f});
    }
    // Vulkan-clip projection (already bakes the Y-flip). The mac/MSL headless path applies its own
    // projection Y-flip on top, exactly as the showcases do.
    math::Mat4 Proj() const {
        return math::Mat4::Perspective(fovY, aspect, znear, zfar);
    }
    math::Mat4 ViewProj() const {
        return Proj() * View();
    }

    CameraBasis Basis() const {
        CameraBasis b;
        b.forward = Forward();
        b.right = Right();
        b.up = Up();
        b.position = position;
        b.tanHalfFovY = std::tan(0.5f * fovY);
        b.aspect = aspect;
        return b;
    }

    // Clamp pitch to just under straight up/down to avoid the LookAt up-vector degenerating.
    void SetPitch(float p) {
        const float kLimit = 1.55334303f;  // pi/2 - ~0.0174 rad (~89 degrees)
        pitch = (p > kLimit) ? kLimit : (p < -kLimit ? -kLimit : p);
    }
    void AddYawPitch(float dYaw, float dPitch) {
        yaw += dYaw;
        SetPitch(pitch + dPitch);
    }
};

} // namespace hf::runtime
