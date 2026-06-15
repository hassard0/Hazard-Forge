#pragma once
// Point-light omnidirectional shadow math — pure CPU (header-only, no device, no backend symbols).
// Shared by the --point-shadow-shot showcase AND tests/point_shadow_test.cpp so the GPU path and the
// unit test exercise the SAME 6-face cube projection + face-selection math.
//
// Phase-1 fidelity: a point light casts shadows in ALL directions by rendering the scene from the
// light through 6 perspective frustums (the 6 cube faces +X,-X,+Y,-Y,+Z,-Z), each FOV=90deg,
// aspect=1, into 6 tiles of one shadow ATLAS (3x2 grid of 1024 tiles in a 3072 map). No cubemap RHI.
//
// Conventions match engine/math: column-major Mat4, Mat4::Perspective/LookAt produce Vulkan clip
// space (depth [0,1], Y-flip baked in). Each face matrix is therefore directly comparable to the
// existing single-shadow lightViewProj — the depth-only caster doesn't care it's one of 6 frustums.
// On Metal the caller flips the projection Y (FlipProjY) exactly like the directional/CSM/spot
// showcases do, and lit_point.frag applies the matching HF_MSL_GEN V-flip on sample.
//
// FACE CONVENTION (this CPU header and shaders/lit_point.frag MUST agree exactly):
//   face 0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z.
// Face selection picks the dominant absolute axis of (wpos - lightPos); its sign selects +/-.

#include "math/math.h"
#include <array>
#include <cmath>

namespace hf::render::point_shadow {

inline constexpr int kFaces = 6;

// Forward (look) direction for each cube face, indexed +X,-X,+Y,-Y,+Z,-Z.
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

// Conventional cube-face up vectors. Lateral faces use -Y up; the vertical faces use +/-Z so LookAt
// has a valid (non-parallel) up.
inline math::Vec3 FaceUp(int face) {
    switch (face) {
        case 2: return {0, 0, 1};   // +Y
        case 3: return {0, 0,-1};   // -Y
        default:return {0,-1, 0};   // +X,-X,+Z,-Z
    }
}

// View-proj for one cube face: Perspective(90deg, 1.0, near, range) * LookAt(P, P+faceDir, up).
// (Vulkan clip space — on Metal the caller wraps the projection in FlipProjY.)
inline math::Mat4 FaceViewProj(const math::Vec3& position, int face, float nearZ, float range) {
    using math::Mat4; using math::Vec3;
    Vec3 dir = FaceDir(face);
    Mat4 view = Mat4::LookAt(position, position + dir, FaceUp(face));
    // 90deg FOV in radians = pi/2; aspect 1 -> exactly covers one cube face.
    Mat4 proj = Mat4::Perspective(1.57079632679489661923f, 1.0f, nearZ, range);
    return proj * view;
}

// Build all 6 face view-projs at once.
inline std::array<math::Mat4, kFaces> FaceViewProjs(const math::Vec3& position, float nearZ,
                                                     float range) {
    std::array<math::Mat4, kFaces> out{};
    for (int i = 0; i < kFaces; ++i) out[i] = FaceViewProj(position, i, nearZ, range);
    return out;
}

// Select the cube face for a direction from the light toward the fragment (wpos - lightPos). The
// dominant absolute axis chooses X/Y/Z; its sign chooses the +/- face. This is the EXACT mirror of
// the HLSL face-selection in lit_point.frag, so the render face and the sampled face always agree.
inline int SelectFace(const math::Vec3& dirLightToFrag) {
    float ax = std::fabs(dirLightToFrag.x);
    float ay = std::fabs(dirLightToFrag.y);
    float az = std::fabs(dirLightToFrag.z);
    if (ax >= ay && ax >= az) return dirLightToFrag.x >= 0.0f ? 0 : 1;  // +X / -X
    if (ay >= ax && ay >= az) return dirLightToFrag.y >= 0.0f ? 2 : 3;  // +Y / -Y
    return dirLightToFrag.z >= 0.0f ? 4 : 5;                            // +Z / -Z
}

// Atlas tile (col,row) for a face in the 3x2 grid: col = face%3, row = face/3.
struct TileRect { int col; int row; };
inline TileRect FaceTile(int face) { return TileRect{ face % 3, face / 3 }; }

} // namespace hf::render::point_shadow
