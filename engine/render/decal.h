#pragma once
// Screen-space projected-decal math — pure CPU (header-only, no device, no backend symbols).
// Shared by the --decal-shot showcase AND tests/decal_test.cpp so the unit test exercises the SAME
// decal-box projection math the in-shader decal composite uses (shaders/decal.frag.hlsl), exactly as
// engine/render/ssr.h is shared with the SSR pass + tests/ssr_test.cpp.
//
// Model: a decal is an oriented box volume. BuildDecalTransform builds its LOCAL->WORLD transform as
// TRS = Translate(center) * R(eulerRot) * Scale(2*halfExtents), where R = Rz*Ry*Rx (column-major,
// right-handed, matching engine/math). In local space the box is the UNIT CUBE [-0.5,0.5]^3 (so the
// 2*halfExtents scale makes a world point at center + R*halfExtents land at the local corner
// (0.5,0.5,0.5)). The INVERSE (world->decal) is uploaded to the shader.
//
// The decal PROJECTS TOP-DOWN along local -Y onto whatever geometry lies inside the box: a fragment's
// world position is transformed into decal-local space; if it is InsideUnitBox the decal texture is
// sampled at DecalUV (the local XZ plane mapped to [0,1]^2) and alpha-blended over the scene. The
// shader reconstructs the world position from the g-buffer EXACTLY as SSR/SSAO reconstruct view space
// (ReconstructViewPos + the yFlip convention) and then maps view->world with the camera (invView),
// so this CPU math and the shader stay mutually consistent and the unit test pins the CPU side.

#include "math/math.h"
#include <algorithm>
#include <cmath>

namespace hf::render::decal {

// Decal-box LOCAL -> WORLD transform. `center`/`halfExtents` are world-space; `eulerRot` is XYZ Euler
// angles (radians) applied as Rz*Ry*Rx. The local unit cube [-0.5,0.5]^3 is scaled by 2*halfExtents so
// its faces sit at +-halfExtents in the rotated/translated frame. Its inverse is the world->decal
// matrix the shader uses.
inline math::Mat4 BuildDecalTransform(const math::Vec3& center,
                                      const math::Vec3& halfExtents,
                                      const math::Vec3& eulerRot) {
    math::Mat4 R = math::Mat4::RotateZ(eulerRot.z)
                 * math::Mat4::RotateY(eulerRot.y)
                 * math::Mat4::RotateX(eulerRot.x);
    math::Mat4 S = math::Mat4::Scale(
        math::Vec3{2.0f * halfExtents.x, 2.0f * halfExtents.y, 2.0f * halfExtents.z});
    return math::Mat4::Translate(center) * R * S;
}

// Transform a WORLD position into DECAL-LOCAL space (the unit cube [-0.5,0.5]^3). `worldToDecal` is the
// inverse of BuildDecalTransform's result (uploaded to the shader as a mat4).
inline math::Vec3 WorldToDecalLocal(const math::Vec3& worldPos, const math::Mat4& worldToDecal) {
    return math::MulPoint(worldToDecal, worldPos);
}

// True if a decal-local position lies inside the unit box (boundary inclusive on every face).
inline bool InsideUnitBox(const math::Vec3& local) {
    return std::fabs(local.x) <= 0.5f
        && std::fabs(local.y) <= 0.5f
        && std::fabs(local.z) <= 0.5f;
}

// Top-down decal UV: the decal projects along local -Y, so the local XZ plane is the decal surface.
// uv = local.xz + 0.5 maps [-0.5,0.5]^2 -> [0,1]^2 (independent of the local Y/projection axis).
inline math::Vec2 DecalUV(const math::Vec3& local) {
    return math::Vec2{local.x + 0.5f, local.z + 0.5f};
}

// Smooth alpha falloff near the box faces so the decal edge isn't a hard cut. `dist` is the distance
// (in local units) from the nearest face: 0.5 at the box center, 0 at a face. The fade ramps from 0 at
// the face to 1 once `fade` units inside (a smoothstep over [0,fade]); `fade<=0` disables the fade.
// Deterministic and monotonic non-increasing toward any face.
inline float EdgeFade(const math::Vec3& local, float fade) {
    if (fade <= 0.0f) return 1.0f;
    float dist = 0.5f - std::max(std::fabs(local.x),
                        std::max(std::fabs(local.y), std::fabs(local.z)));
    float t = std::min(std::max(dist / fade, 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);   // smoothstep(0,1,t)
}

} // namespace hf::render::decal
