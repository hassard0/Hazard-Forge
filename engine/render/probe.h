#pragma once
// Slice AK — reflection + irradiance probe math. Pure CPU (header-only, no device, no backend
// symbols). Shared by the --probe-shot showcase AND tests/probe_test.cpp so the GPU bake/sample path
// and the unit test exercise the SAME 6-face cube projection, face-selection, and atlas-tile mapping.
//
// A single baked LOCAL cubemap GI probe: the scene (a Cornell-style colored room) is rendered from a
// fixed probe position P into ONE RGBA16F atlas with TWO blocks:
//   * REFLECTION block: 3x2 grid of kReflTile (512) tiles -> 1536x1024, at atlas origin (0,0).
//   * IRRADIANCE block: 3x2 grid of kIrrTile (64) tiles -> 192x128, placed BELOW the reflection
//     block at y = kReflTile*2 (1024). A diffuse-convolved version of the reflection block.
// Total atlas: kAtlasW x kAtlasH = 1536 x 1152. Both blocks live in one texture so the whole probe
// binds through the single environment set/slot (mirroring Slice R's BindEnvironment) — no second
// descriptor set. The shader (shaders/lit_probe.frag.hlsl) MIRRORS the mapping here EXACTLY.
//
// Face convention is inherited verbatim from render::point_shadow:
//   face 0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z.  Tile (col,row) = (face%3, face/3).
// FaceViewProj is Perspective(90deg, aspect 1, near, range) * LookAt(P, P+faceDir, up) — Vulkan clip
// space; on Metal the caller wraps the projection in FlipProjY exactly like the other showcases and
// the shader applies the matching HF_MSL_GEN V-flip on sample.

#include "math/math.h"
#include "render/point_shadow.h"
#include <array>
#include <cmath>

namespace hf::render::probe {

inline constexpr int kFaces = point_shadow::kFaces;  // 6

// A 2D UV pair (math.h has no Vec2). Plain aggregate so the test can read .u/.v.
struct UV { float u = 0, v = 0; };

// Atlas geometry (pixels). Reflection tiles are large (sharp reflections); irradiance tiles tiny
// (a coarse cosine convolution is all diffuse ambient needs).
inline constexpr int kReflTile = 512;
inline constexpr int kIrrTile  = 64;
inline constexpr int kTilesPerRow = 3;   // 3x2 grid -> 6 faces
inline constexpr int kTilesPerCol = 2;
inline constexpr int kAtlasW = kReflTile * kTilesPerRow;                 // 1536
inline constexpr int kReflBlockH = kReflTile * kTilesPerCol;            // 1024
inline constexpr int kIrrBlockH  = kIrrTile  * kTilesPerCol;           //  128
inline constexpr int kAtlasH = kReflBlockH + kIrrBlockH;               // 1152

inline constexpr float kProbeNear = 0.05f;
inline constexpr float kProbeRange = 60.0f;

// --- Re-export the cube-face math from point_shadow so there is exactly one source of truth. ---
inline math::Vec3 FaceDir(int face) { return point_shadow::FaceDir(face); }
inline math::Vec3 FaceUp(int face)  { return point_shadow::FaceUp(face); }
inline int SelectFace(const math::Vec3& dir) { return point_shadow::SelectFace(dir); }
inline point_shadow::TileRect FaceTile(int face) { return point_shadow::FaceTile(face); }

inline math::Mat4 FaceViewProj(const math::Vec3& position, int face,
                               float nearZ = kProbeNear, float range = kProbeRange) {
    return point_shadow::FaceViewProj(position, face, nearZ, range);
}
inline std::array<math::Mat4, kFaces> FaceViewProjs(const math::Vec3& position,
                                                    float nearZ = kProbeNear,
                                                    float range = kProbeRange) {
    return point_shadow::FaceViewProjs(position, nearZ, range);
}

// Project a world direction onto its selected cube face and return the [0,1] face UV. This is the
// SAME parameterization the GPU produces by projecting (P + dir) through faceVP[face] and taking
// proj.xy*0.5+0.5 — verified by the unit test. Pure planar cube-face S/T. Returns face via out param.
inline UV FaceUVFromDir(const math::Vec3& dir, int& outFace) {
    outFace = SelectFace(dir);
    // Project a point along `dir` from the origin-probe through that face's view-proj. Using a unit
    // probe at origin keeps this a pure direction->uv mapping (translation cancels for a direction).
    math::Mat4 vp = FaceViewProj(math::Vec3{0, 0, 0}, outFace);
    float w = 0.0f;
    math::Vec3 ndc = math::MulPointDivide(vp, dir, w);  // dir treated as a point 1 unit out
    return UV{ndc.x * 0.5f + 0.5f, ndc.y * 0.5f + 0.5f};
}

// Atlas-UV (in [0,1] over the WHOLE kAtlasW x kAtlasH texture) for a face's [0,1] face UV, in the
// REFLECTION block (top 1536x1024). col=face%3, row=face/3 over 512 tiles.
inline UV ReflTileUV(int face, const UV& faceUV) {
    auto t = FaceTile(face);
    float tileW = (float)kReflTile / (float)kAtlasW;
    float tileH = (float)kReflTile / (float)kAtlasH;
    float ox = (float)t.col * tileW;
    float oy = (float)t.row * tileH;
    return UV{ox + faceUV.u * tileW, oy + faceUV.v * tileH};
}

// Atlas-UV for a face's [0,1] face UV in the IRRADIANCE block (192x128 starting at y=1024 px).
inline UV IrrTileUV(int face, const UV& faceUV) {
    auto t = FaceTile(face);
    float tileW = (float)kIrrTile / (float)kAtlasW;
    float tileH = (float)kIrrTile / (float)kAtlasH;
    float blockOY = (float)kReflBlockH / (float)kAtlasH;  // irradiance block starts here
    float ox = (float)t.col * tileW;
    float oy = blockOY + (float)t.row * tileH;
    return UV{ox + faceUV.u * tileW, oy + faceUV.v * tileH};
}

} // namespace hf::render::probe
