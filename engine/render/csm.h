#pragma once
// Cascaded Shadow Maps — pure CPU math (header-only, no device, no backend symbols).
// Shared by the --csm-shot showcase AND tests/csm_test.cpp so the GPU path and the unit test
// exercise the SAME split/fit code.
//
// Conventions match engine/math: column-major Mat4 (element(row,col)==m[col*4+row]),
// Mat4::Ortho/LookAt produce Vulkan clip space (depth [0,1], Y-flip baked in). The cascade light
// matrices are therefore directly comparable to the existing single-shadow lightViewProj.

#include "math/math.h"
#include <array>
#include <algorithm>
#include <cmath>

namespace hf::render::csm {

inline constexpr int kMaxCascades = 4;

// Practical split scheme (Zhang et al.): blend of logarithmic and uniform splits. Returns the N
// view-space FAR distances of cascades 0..N-1 (cascade i covers (prevSplit, splits[i]]). splits[N-1]
// == far. lambda in [0,1]: 0 = pure uniform, 1 = pure logarithmic. Splits are strictly increasing
// within (near, far].
inline std::array<float, kMaxCascades> CsmSplits(float nearZ, float farZ, int n, float lambda) {
    std::array<float, kMaxCascades> out{};
    n = std::clamp(n, 1, kMaxCascades);
    for (int i = 0; i < n; ++i) {
        float p = (float)(i + 1) / (float)n;
        float logSplit = nearZ * std::pow(farZ / nearZ, p);
        float uniSplit = nearZ + (farZ - nearZ) * p;
        out[i] = lambda * logSplit + (1.0f - lambda) * uniSplit;
    }
    return out;
}

// World-space corners (8) of the camera view-frustum slice between view-space distances
// [sliceNear, sliceFar]. invViewProj is the inverse of the camera's full perspective viewProj
// (built with the same near/far the perspective used). We reconstruct corners by unprojecting the
// 8 NDC cube corners then re-parameterizing along each edge by the slice's normalized depth range —
// but because perspective depth is non-linear, we instead rebuild corners from camera basis +
// the slice's linear view-space distances (robust + matches the GPU's linear-depth cascade pick).
//
// cam: eye, normalized forward f, right r, up u (right-handed, world space). tanHalfFovY, aspect.
inline std::array<math::Vec3, 8> FrustumSliceCornersWorld(
    const math::Vec3& eye, const math::Vec3& f, const math::Vec3& r, const math::Vec3& u,
    float tanHalfFovY, float aspect, float sliceNear, float sliceFar) {
    using math::Vec3;
    auto plane = [&](float dist) -> std::array<Vec3, 4> {
        float hh = tanHalfFovY * dist;     // half-height at this distance
        float hw = hh * aspect;            // half-width
        Vec3 c = eye + f * dist;
        return {
            c - r * hw - u * hh,  // bottom-left
            c + r * hw - u * hh,  // bottom-right
            c + r * hw + u * hh,  // top-right
            c - r * hw + u * hh,  // top-left
        };
    };
    auto np = plane(sliceNear);
    auto fp = plane(sliceFar);
    return {np[0], np[1], np[2], np[3], fp[0], fp[1], fp[2], fp[3]};
}

// Result of fitting a cascade: the light view*ortho (Vulkan clip space), plus the light view alone
// (for tests).
struct CascadeFit {
    math::Mat4 lightViewProj;  // Ortho * LookAt(light)
    math::Mat4 lightView;
};

// Fit a tight ortho light projection around the 8 world-space corners of a frustum slice.
//   lightDir: normalized direction the light travels (fragments lit from -lightDir).
//   zPadNear: extra depth pulled toward the light so off-slice casters in front still cast (world units).
// The light "eye" is placed at the slice centroid pushed back along -lightDir; we fit an AABB of the
// corners in light view space and build Ortho from it. Y-flip + [0,1] depth come from Mat4::Ortho.
inline CascadeFit FitCascadeLightMatrix(const std::array<math::Vec3, 8>& cornersWorld,
                                        const math::Vec3& lightDir, float zPadNear) {
    using math::Vec3; using math::Mat4;
    // Centroid of the slice.
    Vec3 centroid{0, 0, 0};
    for (const auto& c : cornersWorld) centroid = centroid + c;
    centroid = centroid * (1.0f / 8.0f);

    Vec3 L = math::normalize(lightDir);
    // Choose an up vector not parallel to L.
    Vec3 up = (std::fabs(L.y) > 0.99f) ? Vec3{0, 0, 1} : Vec3{0, 1, 0};

    // Place the light eye behind the centroid along -L by a provisional distance; refined by AABB.
    Vec3 lightEye = centroid - L * 50.0f;
    Mat4 lightView = Mat4::LookAt(lightEye, centroid, up);

    // Transform corners into light view space; fit AABB.
    float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
    float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
    for (const auto& c : cornersWorld) {
        // view = lightView * c (point)
        float x = lightView.m[0]*c.x + lightView.m[4]*c.y + lightView.m[8]*c.z  + lightView.m[12];
        float y = lightView.m[1]*c.x + lightView.m[5]*c.y + lightView.m[9]*c.z  + lightView.m[13];
        float z = lightView.m[2]*c.x + lightView.m[6]*c.y + lightView.m[10]*c.z + lightView.m[14];
        minX = std::min(minX, x); maxX = std::max(maxX, x);
        minY = std::min(minY, y); maxY = std::max(maxY, y);
        minZ = std::min(minZ, z); maxZ = std::max(maxZ, z);
    }

    // In RH view space the camera looks down -Z, so corners have NEGATIVE z (in front). The ortho
    // near plane is the closest-to-light surface (largest z, i.e. maxZ); far is minZ. Mat4::Ortho
    // takes (l,r,b,t,zn,zf) as POSITIVE distances along -view-z. distance d = -z. So:
    //   nearDist = -maxZ - zPadNear  (pull near toward the light so casters in front still cast)
    //   farDist  = -minZ
    float nearDist = -maxZ - zPadNear;
    float farDist  = -minZ;
    if (farDist - nearDist < 1e-3f) farDist = nearDist + 1e-3f;

    Mat4 ortho = Mat4::Ortho(minX, maxX, minY, maxY, nearDist, farDist);
    CascadeFit fit;
    fit.lightView = lightView;
    fit.lightViewProj = ortho * lightView;
    return fit;
}

} // namespace hf::render::csm
