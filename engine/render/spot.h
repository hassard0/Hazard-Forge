#pragma once
// Spot light shadow math — pure CPU (header-only, no device, no backend symbols).
// Shared by the --spot-shot showcase AND tests/spot_test.cpp so the GPU path and the unit test
// exercise the SAME perspective-projection + cone math.
//
// Conventions match engine/math: column-major Mat4, Mat4::Perspective/LookAt produce Vulkan clip
// space (depth [0,1], Y-flip baked in). The spot light matrix is therefore directly comparable to
// the existing single-shadow lightViewProj — the depth-only caster doesn't care that it's a
// perspective projection instead of an ortho one. On Metal the caller flips the projection Y
// (FlipProjY) exactly like the directional/CSM showcases do.

#include "math/math.h"
#include <cmath>

namespace hf::render::spot {

// Spot light parameters (half-angles in radians).
struct SpotLight {
    math::Vec3 position;
    math::Vec3 direction;  // unit cone axis (light -> scene)
    float      innerCone;  // full brightness inside this half-angle
    float      outerCone;  // zero past this half-angle; soft edge inner->outer
    math::Vec3 color;
    float      range;      // distance falloff + shadow far plane
    float      intensity;
};

// Choose an up vector not parallel to the cone axis (LookAt requires this). World +Y unless the
// spot points near-vertical, in which case world +Z.
inline math::Vec3 SpotUp(const math::Vec3& dir) {
    return (std::fabs(math::normalize(dir).y) > 0.99f) ? math::Vec3{0, 0, 1} : math::Vec3{0, 1, 0};
}

// Perspective light view-proj that exactly covers the cone: FOV = 2*outerCone, square aspect.
// (Vulkan clip space — on Metal the caller wraps this in FlipProjY.)
inline math::Mat4 SpotViewProj(const math::Vec3& position, const math::Vec3& direction,
                               float outerCone, float nearZ, float range) {
    using math::Mat4; using math::Vec3;
    Vec3 dir = math::normalize(direction);
    Mat4 view = Mat4::LookAt(position, position + dir, SpotUp(dir));
    Mat4 proj = Mat4::Perspective(2.0f * outerCone, 1.0f, nearZ, range);
    return proj * view;
}

// Cone attenuation: 1 on-axis (inside innerCone), smooth 1->0 across inner..outer, 0 outside.
// `dirToFrag` is the unit vector from the light toward the fragment (== -L in the shader). This is
// the exact CPU mirror of the shader's smoothstep(cosOuter, cosInner, dot(axis, dirToFrag)).
inline float ConeAttenuation(const math::Vec3& coneAxis, const math::Vec3& dirToFrag,
                             float innerCone, float outerCone) {
    float cosInner = std::cos(innerCone);
    float cosOuter = std::cos(outerCone);
    float ca = math::dot(math::normalize(coneAxis), math::normalize(dirToFrag));
    // smoothstep(edge0=cosOuter, edge1=cosInner, x=ca).
    float denom = (cosInner - cosOuter);
    float t = (denom > 1e-6f) ? (ca - cosOuter) / denom : (ca >= cosInner ? 1.0f : 0.0f);
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    return t * t * (3.0f - 2.0f * t);
}

} // namespace hf::render::spot
