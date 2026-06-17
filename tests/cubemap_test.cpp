// Slice DD — Runtime Cubemap-Capture Reflection Probe. Pure CPU math: the 6 cube-face view/proj
// matrices (FaceView / FaceProj) + the hardware cube lookup (DirToFaceUV). No device, ASan-eligible
// (links hf_core). Mirrors the math the --captureprobe-shot showcase + the reflection shader use
// (engine/render/cubemap.h), so this test pins the convention BEFORE any GPU work.
//
// Properties pinned (per the spec):
//   * Face-view/proj sanity: FaceView(face) looks down the correct ±axis with the documented up; a
//     point straight down a face's axis from the center projects to ~the center of that face; the 6
//     faces tile all directions.
//   * Direction<->face consistency: DirToFaceUV selects the major-axis face + the right UV; a
//     direction down each face axis -> that face, UV ≈ (0.5,0.5); the 6 standard axes map to the 6
//     faces; round-trip (FaceView/FaceProj projects the same axis point to NDC center).
//   * Up-vector convention: the per-face up-vectors match the cubemap sampling convention (so a
//     captured face isn't flipped/rotated vs how a TextureCube samples) — hand-checked for +Y/-Y,
//     the classic flip pitfall.
//   * Determinism: same inputs -> identical matrices.
#include "render/cubemap.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

namespace cm = hf::render::cubemap;
using hf::math::Vec2;
using hf::math::Vec3;
using hf::math::Mat4;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps) { return std::fabs(a - b) <= eps; }
static bool approx3(const Vec3& a, const Vec3& b, float eps) {
    return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
}
// Transform a direction by the rotation part of a view matrix (w=0; column-major).
static Vec3 mulDir(const Mat4& m, const Vec3& d) {
    return {m.m[0]*d.x + m.m[4]*d.y + m.m[8] *d.z,
            m.m[1]*d.x + m.m[5]*d.y + m.m[9] *d.z,
            m.m[2]*d.x + m.m[6]*d.y + m.m[10]*d.z};
}

int main() {
    HF_TEST_MAIN_INIT();
    const Vec3 center{2.0f, -1.0f, 3.0f};   // an off-origin probe center
    const float kNear = 0.05f, kFar = 60.0f;

    // ---- Face-view: FaceView looks down the correct ±axis. ----------------------------------------
    // In a RH LookAt view matrix the forward (look) axis maps to view-space -Z. So mulDir(view,
    // faceDir) == (0,0,-1) for the correct face, and the up vector maps to view-space +Y.
    {
        for (int f = 0; f < cm::kFaces; ++f) {
            Mat4 view = cm::FaceView(f, center);
            Vec3 vfwd = mulDir(view, cm::FaceDir(f));
            check(approx3(vfwd, Vec3{0, 0, -1}, 1e-5f), "FaceView: look dir maps to view -Z");
            Vec3 vup = mulDir(view, cm::FaceUp(f));
            check(approx3(vup, Vec3{0, 1, 0}, 1e-5f), "FaceView: up maps to view +Y");
        }
    }

    // ---- Face-proj: a point straight down the face axis projects to the face CENTER (NDC ~0,0). ----
    // FaceViewProj * (center + dir) -> clip; after divide xy ≈ (0,0) (the principal axis), and the
    // point is in front (w>0). This is the round-trip that ties FaceView/FaceProj to the cube center.
    {
        for (int f = 0; f < cm::kFaces; ++f) {
            Mat4 vp = cm::FaceViewProj(f, center, kNear, kFar);
            Vec3 p = center + cm::FaceDir(f) * 5.0f;  // 5 units down the face axis
            float w = 0.0f;
            Vec3 ndc = hf::math::MulPointDivide(vp, p, w);
            check(w > 0.0f, "FaceProj: axis point is in front of the face (w>0)");
            check(approx(ndc.x, 0.0f, 1e-4f) && approx(ndc.y, 0.0f, 1e-4f),
                  "FaceProj: axis point projects to the face center (NDC 0,0)");
            check(ndc.z >= 0.0f && ndc.z <= 1.0f, "FaceProj: depth in [0,1] (Vulkan clip)");
        }
        // A point on the +X axis from the center projects to ~face-0 center specifically.
        Mat4 vp0 = cm::FaceViewProj(0, center, kNear, kFar);
        float w0 = 0.0f;
        Vec3 ndc0 = hf::math::MulPointDivide(vp0, center + Vec3{4, 0, 0}, w0);
        check(approx(ndc0.x, 0.0f, 1e-4f) && approx(ndc0.y, 0.0f, 1e-4f),
              "FaceProj: +X-axis point -> face-0 center");
    }

    // ---- DirToFaceUV: the 6 standard axes map to the 6 faces, UV at the center. --------------------
    {
        struct Ax { Vec3 dir; int face; };
        Ax axes[] = {
            {{ 1, 0, 0}, 0}, {{-1, 0, 0}, 1}, {{0,  1, 0}, 2},
            {{ 0,-1, 0}, 3}, {{ 0, 0, 1}, 4}, {{0,  0,-1}, 5},
        };
        for (const auto& a : axes) {
            int face = -1; Vec2 uv;
            cm::DirToFaceUV(a.dir, face, uv);
            check(face == a.face, "DirToFaceUV: axis -> expected face");
            check(approx(uv.x, 0.5f, 1e-5f) && approx(uv.y, 0.5f, 1e-5f),
                  "DirToFaceUV: axis -> UV center (0.5,0.5)");
            check(cm::SelectFace(a.dir) == a.face, "SelectFace: axis -> expected face");
        }
    }

    // ---- DirToFaceUV: major axis selection + UV in [0,1] for off-axis directions. ------------------
    {
        // A direction leaning +X but tilted toward +Y/+Z stays on face 0 with UV pulled off-center
        // but inside [0,1]. (sc = -z, tc = -y for +X: +z pulls s down, +y pulls t down.)
        int face = -1; Vec2 uv;
        cm::DirToFaceUV(Vec3{1.0f, -0.3f, -0.2f}, face, uv);
        check(face == 0, "DirToFaceUV: dominant +X -> face 0");
        check(uv.x >= 0.0f && uv.x <= 1.0f && uv.y >= 0.0f && uv.y <= 1.0f,
              "DirToFaceUV: off-axis UV stays in [0,1]");
        // +X: sc=-z, tc=-y. A -z component raises s above 0.5; a -y component raises t above 0.5.
        check(uv.x > 0.5f, "DirToFaceUV: +X,-z component raises s>0.5 (sc=-z)");
        check(uv.y > 0.5f, "DirToFaceUV: +X,-y component raises t>0.5 (tc=-y)");
    }

    // ---- Up-vector convention: hand-check +Y / -Y (the classic flip pitfall). ----------------------
    {
        // +Y face (face 2): up MUST be +Z (not ±Y — parallel to look — and not -Z, which flips the
        // pole). -Y face (face 3): up MUST be -Z. These are the exact hardware-cubemap pole ups.
        check(approx3(cm::FaceUp(2), Vec3{0, 0, 1}, 0.0f), "FaceUp(+Y) == +Z (not flipped/rotated)");
        check(approx3(cm::FaceUp(3), Vec3{0, 0,-1}, 0.0f), "FaceUp(-Y) == -Z (not flipped/rotated)");
        check(approx3(cm::FaceUp(0), Vec3{0,-1, 0}, 0.0f), "FaceUp(+X) == -Y");
        check(approx3(cm::FaceUp(4), Vec3{0,-1, 0}, 0.0f), "FaceUp(+Z) == -Y");

        // Concrete +Y-face flip check: a world point slightly toward +Z from straight-up should land
        // in the UPPER half of the +Y face (tc = +z so +z -> t>0.5), and a +X offset to the RIGHT
        // half (sc = +x -> s>0.5). If the +Y up were flipped/rotated these signs would invert.
        int f = -1; Vec2 uv;
        cm::DirToFaceUV(Vec3{0.0f, 1.0f, 0.3f}, f, uv);
        check(f == 2 && uv.y > 0.5f, "DirToFaceUV: +Y face, +z -> upper half (t>0.5)");
        cm::DirToFaceUV(Vec3{0.3f, 1.0f, 0.0f}, f, uv);
        check(f == 2 && uv.x > 0.5f, "DirToFaceUV: +Y face, +x -> right half (s>0.5)");
        // -Y face: tc = -z, so +z -> lower half (t<0.5) — the OPPOSITE sign of +Y (un-flipped poles).
        cm::DirToFaceUV(Vec3{0.0f, -1.0f, 0.3f}, f, uv);
        check(f == 3 && uv.y < 0.5f, "DirToFaceUV: -Y face, +z -> lower half (t<0.5)");
    }

    // ---- The 6 faces tile all directions: every random direction selects exactly one valid face. ---
    {
        // A deterministic sweep over a fibonacci-ish set of directions; each must land on a face in
        // [0,6) with UV inside [0,1]. (No RNG — fixed angles.)
        bool faceSeen[6] = {false, false, false, false, false, false};
        for (int i = 0; i < 200; ++i) {
            float a = (float)i * 2.39996323f;   // golden-angle increment
            float z = 1.0f - 2.0f * ((float)i + 0.5f) / 200.0f;
            float r = std::sqrt(std::max(0.0f, 1.0f - z * z));  // NOLINT
            Vec3 d{r * std::cos(a), r * std::sin(a), z};
            int face = -1; Vec2 uv;
            cm::DirToFaceUV(d, face, uv);
            check(face >= 0 && face < 6, "DirToFaceUV: every direction selects a valid face");
            check(uv.x >= -1e-4f && uv.x <= 1.0f + 1e-4f && uv.y >= -1e-4f && uv.y <= 1.0f + 1e-4f,
                  "DirToFaceUV: tiled UV in [0,1]");
            if (face >= 0 && face < 6) faceSeen[face] = true;
        }
        bool all = true;
        for (bool s : faceSeen) all = all && s;
        check(all, "DirToFaceUV: the sweep covers all 6 faces (the cube tiles all directions)");
    }

    // ---- Determinism: same inputs -> identical matrices + lookup. ----------------------------------
    {
        for (int f = 0; f < cm::kFaces; ++f) {
            Mat4 a = cm::FaceViewProj(f, center, kNear, kFar);
            Mat4 b = cm::FaceViewProj(f, center, kNear, kFar);
            bool same = true;
            for (int k = 0; k < 16; ++k) same = same && (a.m[k] == b.m[k]);
            check(same, "FaceViewProj: deterministic (bit-identical re-eval)");
        }
        int f1 = -1, f2 = -1; Vec2 u1, u2;
        cm::DirToFaceUV(Vec3{0.4f, -0.7f, 0.2f}, f1, u1);
        cm::DirToFaceUV(Vec3{0.4f, -0.7f, 0.2f}, f2, u2);
        check(f1 == f2 && u1.x == u2.x && u1.y == u2.y, "DirToFaceUV: deterministic");
    }

    if (g_fail == 0) std::printf("cubemap_test: all checks passed\n");
    else std::printf("cubemap_test: %d checks FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
