#pragma once
// Screen-space reflections math — pure CPU (header-only, no device, no backend symbols).
// Shared by the --ssr-shot showcase reasoning AND tests/ssr_test.cpp so the unit test exercises the
// SAME view<->screen projection + reflection math the in-shader ray-march uses.
//
// Conventions match engine/math + the SSAO g-buffer (Slice Y): RH view space (camera at the origin,
// -Z forward), VIEW-SPACE LINEAR depth = -vpos.z (positive in front of the camera). The shader stores
// this in the g-buffer and reconstructs the view-space position from it + the screen UV + the
// projection params (tanHalfFovY, aspect) WITHOUT a matrix inverse, exactly like ssao.frag. The Y-flip
// sign (`yFlip`) maps screen UV.y <-> view-space Y: -1 on Vulkan (projection bakes a Y-flip and
// post.vert gives a V-down UV), +1 on Metal (FlipProjY + post.vert V-flip compose the other way).
// ViewToScreenUV and ReconstructViewPos below are mutual inverses for a given yFlip.

#include "math/math.h"
#include <algorithm>
#include <cmath>

namespace hf::render::ssr {

// Forward-project a VIEW-SPACE position to a screen UV in [0,1] (the inverse of ReconstructViewPos).
// Mirrors ssao.frag's ProjectToUV: ndc = vp.xy / (scale * -vp.z), uv = ndc*0.5+0.5, with the yFlip
// sign on the Y channel. `yFlip` is -1 on Vulkan, +1 on Metal.
inline math::Vec3 /*xy = uv, z = view linear depth*/ ViewToScreenUV(
    const math::Vec3& viewPos, float tanHalfFovY, float aspect, float yFlip) {
    float invZ = 1.0f / std::max(-viewPos.z, 1e-4f);
    float ndcx = viewPos.x / (aspect * tanHalfFovY) * invZ;
    float ndcy = yFlip * viewPos.y / tanHalfFovY * invZ;
    return math::Vec3{ndcx * 0.5f + 0.5f, ndcy * 0.5f + 0.5f, -viewPos.z};
}

// Reconstruct a VIEW-SPACE position from a screen UV + linear depth (the inverse of ViewToScreenUV).
// Mirrors ssao.frag's ReconstructViewPos.
inline math::Vec3 ReconstructViewPos(
    float u, float v, float linDepth, float tanHalfFovY, float aspect, float yFlip) {
    float ndcx = u * 2.0f - 1.0f;
    float ndcy = v * 2.0f - 1.0f;
    float vx = ndcx * (aspect * tanHalfFovY) * linDepth;
    float vy = yFlip * ndcy * (tanHalfFovY) * linDepth;
    float vz = -linDepth;
    return math::Vec3{vx, vy, vz};
}

// View-space reflection of an INCIDENT direction about a surface normal. Matches HLSL's
// reflect(I, N) = I - 2*dot(I, N)*N (I is the incident direction, pointing INTO the surface). For SSR
// the incident direction is V = normalize(P) (camera->fragment), so R reflects the view ray about N.
inline math::Vec3 ReflectView(const math::Vec3& incident, const math::Vec3& normal) {
    math::Vec3 n = math::normalize(normal);
    float d = math::dot(incident, n);
    return math::Vec3{incident.x - 2.0f * d * n.x,
                      incident.y - 2.0f * d * n.y,
                      incident.z - 2.0f * d * n.z};
}

} // namespace hf::render::ssr
