#pragma once
// Slice AQ — frustum culling math. Pure CPU (header-only, no device, no backend symbols). Same
// pattern as engine/render/ssr.h / taa.h. Namespace hf::render::frustum. Shared by the --cull-shot
// showcase (which builds the render camera's frustum and tests each renderable's world sphere) AND
// tests/frustum_test.cpp, so the unit test exercises the SAME extraction + tests the renderer uses.
//
// CORRECTNESS CONTRACT: frustum culling is RENDER-INVARIANT. It must NEVER drop an object that
// contributes a visible pixel. The render path therefore uses a CONSERVATIVE bounding-SPHERE test:
// an object is culled ONLY IF its world sphere is fully outside at least one plane. Conservative =
// we may KEEP some off-screen objects (correct, just less optimal) but NEVER drop a visible one.
//
// MATRIX CONVENTION (the crux — a transpose bug here silently culls visible geometry):
//   engine/math Mat4 is COLUMN-MAJOR: element(row, col) == m[col*4 + row]. Row r of the matrix is
//   therefore (m[r], m[r+4], m[r+8], m[r+12]). The engine's clip space is the Vulkan one built by
//   Mat4::Perspective: x,y in [-w, w] and z in [0, w] (reverse-free [0,1] depth, with a baked
//   Y-flip in m[5]). For a clip position q = M * [p,1] the six inside inequalities are
//       -w <= x,  x <= w,   -w <= y,  y <= w,   0 <= z,  z <= w
//   which, written as dot(plane, [p,1]) >= 0 with the matrix ROWS R0..R3, give the Gribb-Hartmann
//   planes (Vulkan [0,1]-depth variant — NEAR is R2 alone, not R3+R2):
//       left   = R3 + R0      ( x + w >= 0 )
//       right  = R3 - R0      ( w - x >= 0 )
//       bottom = R3 + R1      ( y + w >= 0 )
//       top    = R3 - R1      ( w - y >= 0 )
//       near   = R2           ( z      >= 0 )   <-- [0,1] depth: z>=0, NOT (z+w)
//       far    = R3 - R2      ( w - z >= 0 )
//   Each plane (a,b,c,d) is normalized by |(a,b,c)| so signedDistance(p) = dot(n,p)+d is a true
//   distance and "inside" is signedDistance >= 0. This is pinned by hand-checked planes in
//   frustum_test.cpp before it is trusted in the render path. (The same extraction is convention-
//   correct on Metal: Mat4::Perspective is the single shared builder; the showcase composes the
//   Metal FlipProjY into the matrix it culls with, so the rows already carry that backend's clip.)

#include "math/math.h"

#include <cmath>

namespace hf::render::frustum {

// A plane in the form dot(n, p) + d = 0 with `n` UNIT length. signedDistance(p) = dot(n,p)+d is
// positive on the inside (the half-space the frustum interior lives in).
struct Plane {
    math::Vec3 n{0, 0, 0};
    float d = 0.0f;
};

// The six bounding planes of a view frustum, inward-facing. Index order: 0=left 1=right 2=bottom
// 3=top 4=near 5=far (matches the comments above + the unit test).
struct Frustum {
    Plane planes[6];
};

// Signed distance from a point to a plane (positive = inside half-space). Plane n is unit length,
// so this is a true Euclidean distance.
inline float SignedDistance(const Plane& p, const math::Vec3& point) {
    return math::dot(p.n, point) + p.d;
}

// Normalize a raw (a,b,c,d) plane (divide all four by |(a,b,c)|) into a unit-normal Plane.
inline Plane NormalizePlane(float a, float b, float c, float d) {
    float len = std::sqrt(a * a + b * b + c * c);
    if (len > 0.0f) {
        float inv = 1.0f / len;
        return Plane{math::Vec3{a * inv, b * inv, c * inv}, d * inv};
    }
    return Plane{math::Vec3{a, b, c}, d};
}

// Gribb-Hartmann extraction of the six inward-facing planes from a COLUMN-MAJOR view-projection
// matrix targeting Vulkan clip space (x,y in [-w,w], z in [0,w]). See the file header for the full
// derivation + convention warning. Pass the UNJITTERED view-proj when culling for TAA.
inline Frustum FromViewProj(const math::Mat4& m) {
    // Rows of the column-major matrix: Rr = (m[r], m[r+4], m[r+8], m[r+12]).
    // R0
    const float r0x = m.m[0], r0y = m.m[4], r0z = m.m[8],  r0w = m.m[12];
    // R1
    const float r1x = m.m[1], r1y = m.m[5], r1z = m.m[9],  r1w = m.m[13];
    // R2
    const float r2x = m.m[2], r2y = m.m[6], r2z = m.m[10], r2w = m.m[14];
    // R3
    const float r3x = m.m[3], r3y = m.m[7], r3z = m.m[11], r3w = m.m[15];

    Frustum f;
    // left   = R3 + R0
    f.planes[0] = NormalizePlane(r3x + r0x, r3y + r0y, r3z + r0z, r3w + r0w);
    // right  = R3 - R0
    f.planes[1] = NormalizePlane(r3x - r0x, r3y - r0y, r3z - r0z, r3w - r0w);
    // bottom = R3 + R1
    f.planes[2] = NormalizePlane(r3x + r1x, r3y + r1y, r3z + r1z, r3w + r1w);
    // top    = R3 - R1
    f.planes[3] = NormalizePlane(r3x - r1x, r3y - r1y, r3z - r1z, r3w - r1w);
    // near   = R2  ([0,1] depth: z >= 0)
    f.planes[4] = NormalizePlane(r2x, r2y, r2z, r2w);
    // far    = R3 - R2
    f.planes[5] = NormalizePlane(r3x - r2x, r3y - r2y, r3z - r2z, r3w - r2w);
    return f;
}

// CONSERVATIVE bounding-sphere cull test. Returns true (cull) iff the sphere is FULLY outside at
// least one plane (signedDistance(center) < -radius). A sphere that straddles any plane is kept.
// This is the test the render path uses: it never drops a sphere that reaches the frustum interior.
inline bool SphereOutside(const Frustum& f, const math::Vec3& center, float radius) {
    for (const Plane& p : f.planes)
        if (SignedDistance(p, center) < -radius) return true;
    return false;
}

// CONSERVATIVE AABB cull test (positive-vertex / "p-vertex" test). For each plane, evaluate the box
// corner FARTHEST along the plane normal (the p-vertex): if even that corner is behind the plane,
// the whole box is outside that plane -> cull. Returns true iff the box is fully outside any plane.
// Slightly tighter than the sphere test for boxy bounds; unit-tested but the renderer uses the
// sphere test for simplicity (documented in the design).
inline bool AabbOutside(const Frustum& f, const math::Vec3& min, const math::Vec3& max) {
    for (const Plane& p : f.planes) {
        // p-vertex: pick min/max per axis to maximize dot(n, vertex).
        math::Vec3 pv{
            p.n.x >= 0.0f ? max.x : min.x,
            p.n.y >= 0.0f ? max.y : min.y,
            p.n.z >= 0.0f ? max.z : min.z,
        };
        if (SignedDistance(p, pv) < 0.0f) return true;  // even the best corner is outside -> cull
    }
    return false;
}

// The eight world-space corners of the frustum a view-proj defines, for debug visualization (the
// --cull-shot draws these as the 12 wireframe edges). Un-projects the NDC cube corners through
// inverse(viewProj): NDC x,y in [-1,1], z in [0,1] (Vulkan clip). Order: bit0=x(-/+), bit1=y(-/+),
// bit2=z(near/far). Pure math; no backend symbols. The renderer never needs this (it uses the plane
// tests) — it exists only for the overview-camera debug viz, so it lives here next to the planes.
inline void Corners(const math::Mat4& viewProj, math::Vec3 out[8]) {
    math::Mat4 inv = viewProj.Inverse();
    int i = 0;
    for (int zi = 0; zi < 2; ++zi)
        for (int yi = 0; yi < 2; ++yi)
            for (int xi = 0; xi < 2; ++xi) {
                math::Vec3 ndc{xi ? 1.0f : -1.0f, yi ? 1.0f : -1.0f, zi ? 1.0f : 0.0f};
                float w = 0.0f;
                out[i++] = math::MulPointDivide(inv, ndc, w);
            }
}

}  // namespace hf::render::frustum
