#pragma once
// Slice DD — Runtime Cubemap-Capture Reflection Probe — cube-face capture math. Pure CPU
// (header-only, no device, no backend symbols). Namespace hf::render::cubemap. Mirrors
// reflection_probe.h / gtao.h / probe.h: a tiny shared-math header ABOVE the RHI seam (ZERO
// vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only mentions of "vk"/"MTL" anywhere in this
// slice's above-seam files are seam-discipline doc comments + the [[vk::binding]] HLSL
// decorations). The capture path (samples/hello_triangle/main.cpp) AND the unit test
// (tests/cubemap_test.cpp) consume the SAME FaceView / FaceProj / DirToFaceUV, so the GPU capture and
// the CPU test exercise the IDENTICAL 6-face cube projection + cube-lookup math — which is what makes
// the captured face BYTE-IDENTICAL to the scene rendered directly with that face's view/proj.
//
// THE TECHNIQUE (runtime cubemap-capture reflection probe — the dynamic counterpart to DA's STATIC
// env reflection): from a fixed probe CENTER, render the actual scene into the 6 faces of a color
// cubemap, each face being the scene viewed through a 90°-FOV square frustum looking down one of the
// ±X/±Y/±Z axes. A reflective surface then samples that CAPTURED cubemap along its (box-projected,
// reusing DA) reflection direction, so it mirrors the real surrounding geometry — not a baked
// snapshot. Because the per-face capture uses the IDENTICAL scene + shaders + camera math a direct
// render would, the captured +X face equals a standalone render with the +X face's view/proj: a
// clean capture-correctness identity (asserted byte-for-byte by the showcase).
//
// THE CUBEMAP-SAMPLING CONVENTION (capture and sampling MUST agree — this is the whole point):
//   This header matches the HARDWARE TextureCube convention (D3D11 / Vulkan / Metal cube textures all
//   share it), so a captured cube can be sampled directly by a TextureCube in the reflection shader.
//   The face index + per-face up-vector + the face S/T (UV) parameterization below are the standard
//   cube-map layout:
//
//   face 0 = +X : look +X, up -Y   |  ma = x  | sc = -z, tc = -y
//   face 1 = -X : look -X, up -Y   |  ma = x  | sc = +z, tc = -y
//   face 2 = +Y : look +Y, up +Z   |  ma = y  | sc = +x, tc = +z
//   face 3 = -Y : look -Y, up -Z   |  ma = y  | sc = +x, tc = -z
//   face 4 = +Z : look +Z, up -Y   |  ma = z  | sc = +x, tc = -y
//   face 5 = -Z : look -Z, up -Y   |  ma = z  | sc = -x, tc = -y
//   with s = (sc/|ma| + 1)/2,  t = (tc/|ma| + 1)/2.
//
//   THE +Y/-Y FLIP PITFALL (the classic cubemap bug): the LATERAL faces (+X,-X,+Z,-Z) use up = -Y,
//   but the VERTICAL faces use up = +Z (+Y) and up = -Z (-Y) — NOT -Y (LookAt would be degenerate
//   with up parallel to the look axis, and the wrong sign rotates/flips the pole face). This header
//   bakes the correct ±Z up for the vertical faces and the unit test hand-checks +Y/-Y so a captured
//   pole face is neither flipped nor rotated relative to how a TextureCube samples it.
//
//   FaceView/FaceProj match the ENGINE'S MAIN projection handedness/depth: RH view space (LookAt) and
//   Mat4::Perspective (Vulkan clip space, depth [0,1], Y-flip baked in). So a face render produced
//   here is directly comparable to any other scene render the engine does — that is what makes the
//   capture==direct-render proof hold. On Metal the showcase wraps the projection exactly like the
//   other showcases and the reflection shader applies the matching V-flip on cube sample.
//
// Pure, deterministic functions: no RNG, no time.

#include "math/math.h"

#include <cmath>

namespace hf::render::cubemap {

inline constexpr int kFaces = 6;

// Forward (look) direction for each cube face, indexed +X,-X,+Y,-Y,+Z,-Z. Matches the TextureCube
// face order (and render::point_shadow's FaceDir).
inline math::Vec3 FaceDir(int face) {
    switch (face) {
        case 0: return { 1, 0, 0};
        case 1: return {-1, 0, 0};
        case 2: return { 0, 1, 0};
        case 3: return { 0,-1, 0};
        case 4: return { 0, 0, 1};
        default:return { 0, 0,-1};
    }
}

// Per-face up vector for the cubemap convention. Lateral + Z faces use -Y up; the +Y face uses +Z up
// and the -Y face uses -Z up (the vertical faces can't use ±Y — it would be parallel to the look
// axis). These are the standard hardware-cubemap up-vectors; the unit test hand-checks the ±Y signs.
inline math::Vec3 FaceUp(int face) {
    switch (face) {
        case 2: return {0, 0, 1};   // +Y up = +Z
        case 3: return {0, 0,-1};   // -Y up = -Z
        default:return {0,-1, 0};   // +X,-X,+Z,-Z up = -Y
    }
}

// View matrix for cube face `face` looking from `probeCenter` down that axis with the cubemap up
// (RH view space, matches Mat4::LookAt). This is the camera the face's scene render uses.
inline math::Mat4 FaceView(int face, const math::Vec3& probeCenter) {
    math::Vec3 dir = FaceDir(face);
    return math::Mat4::LookAt(probeCenter, probeCenter + dir, FaceUp(face));
}

// The per-face projection: 90°-FOV (square, aspect 1) perspective covering exactly one cube face.
// Mat4::Perspective is Vulkan clip space (depth [0,1], Y-flip baked in) — the SAME handedness/depth
// as the engine's main projection, so a face render is comparable to any other scene render.
inline math::Mat4 FaceProj(float zNear, float zFar) {
    return math::Mat4::Perspective(1.57079632679489661923f /* 90° */, 1.0f, zNear, zFar);
}

// Combined per-face view-projection: FaceProj * FaceView. Convenience for the capture path.
inline math::Mat4 FaceViewProj(int face, const math::Vec3& probeCenter, float zNear, float zFar) {
    return FaceProj(zNear, zFar) * FaceView(face, probeCenter);
}

// --- The hardware cube lookup: direction -> (face, faceUV) -------------------------------------
// DirToFaceUV(dir, outFace, outUV) returns the SELECTED face + the [0,1] face S/T (UV) for a sample
// direction, using the standard major-axis cube mapping (the EXACT mapping a TextureCube does in
// hardware). The major (largest-magnitude) axis + its sign select the face; the other two components
// (with the convention-specific signs in the table above) divided by |major| give sc,tc in [-1,1],
// remapped to [0,1] UV. `dir` need not be unit. Returns the major-axis component magnitude (|ma|),
// which the test uses as a sanity value (it is the depth along the face axis). Pure / deterministic.
//
// Consistency with FaceView/FaceProj (proved by the unit test): a direction straight down a face's
// axis maps to that face with UV (0.5,0.5) — the face center — because the perspective frustum is
// centered on the axis; off-axis directions move within the same face's [0,1] UV until the major
// axis changes at a 45° edge, where the lookup crosses to the neighboring face.
inline float DirToFaceUV(const math::Vec3& dir, int& outFace, math::Vec2& outUV) {
    float ax = std::fabs(dir.x), ay = std::fabs(dir.y), az = std::fabs(dir.z);
    float ma, sc, tc;
    if (ax >= ay && ax >= az) {
        if (dir.x >= 0.0f) { outFace = 0; sc = -dir.z; tc = -dir.y; }  // +X
        else               { outFace = 1; sc =  dir.z; tc = -dir.y; }  // -X
        ma = ax;
    } else if (ay >= ax && ay >= az) {
        if (dir.y >= 0.0f) { outFace = 2; sc =  dir.x; tc =  dir.z; }  // +Y
        else               { outFace = 3; sc =  dir.x; tc = -dir.z; }  // -Y
        ma = ay;
    } else {
        if (dir.z >= 0.0f) { outFace = 4; sc =  dir.x; tc = -dir.y; }  // +Z
        else               { outFace = 5; sc = -dir.x; tc = -dir.y; }  // -Z
        ma = az;
    }
    float inv = (ma > 1e-20f) ? 0.5f / ma : 0.0f;
    outUV = math::Vec2{sc * inv + 0.5f, tc * inv + 0.5f};
    return ma;
}

// Select just the face for a direction (major-axis sign). Equivalent to DirToFaceUV's face. Mirrors
// the HLSL SelectFace in the reflection shader / render::point_shadow::SelectFace.
inline int SelectFace(const math::Vec3& dir) {
    int face;
    math::Vec2 uv;
    DirToFaceUV(dir, face, uv);
    return face;
}

} // namespace hf::render::cubemap
