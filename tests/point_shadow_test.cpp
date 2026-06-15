// Slice AF — omnidirectional point-light shadows. Pure CPU math: 6-face cube projection +
// face-selection. No device, ASan-eligible (links hf_core). Mirrors the math the --point-shadow-shot
// showcase and lit_point.frag use.
#include "render/point_shadow.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>

using namespace hf::math;
namespace ps = hf::render::point_shadow;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 2e-3f) { return std::fabs(a - b) < eps; }

static Vec3 project(const Mat4& m, const Vec3& p, float& outW) {
    return MulPointDivide(m, p, outW);
}

int main() {
    const Vec3  P{0.0f, 0.0f, 0.0f};   // point light at origin
    const float nearZ = 0.1f;
    const float range = 30.0f;

    // ---- SelectFace returns the correct face for clear directions along each +/- axis. ----
    {
        check(ps::SelectFace(Vec3{ 5.0f,  0.1f, -0.2f}) == 0, "SelectFace +X -> 0");
        check(ps::SelectFace(Vec3{-5.0f, -0.2f,  0.1f}) == 1, "SelectFace -X -> 1");
        check(ps::SelectFace(Vec3{ 0.1f,  5.0f,  0.2f}) == 2, "SelectFace +Y -> 2");
        check(ps::SelectFace(Vec3{-0.2f, -5.0f,  0.1f}) == 3, "SelectFace -Y -> 3");
        check(ps::SelectFace(Vec3{ 0.1f, -0.2f,  5.0f}) == 4, "SelectFace +Z -> 4");
        check(ps::SelectFace(Vec3{-0.2f,  0.1f, -5.0f}) == 5, "SelectFace -Z -> 5");
    }

    // ---- Each face's view-proj projects a point straight out along its axis to ~clip center,
    //      depth in [0,1]; the SAME point projects OUTSIDE the opposite face. ----
    {
        for (int face = 0; face < ps::kFaces; ++face) {
            Vec3 dir = ps::FaceDir(face);
            Vec3 onAxis = P + dir * 8.0f;   // 8 units straight out along this face's axis
            Mat4 vp = ps::FaceViewProj(P, face, nearZ, range);
            float w = 0.0f;
            Vec3 ndc = project(vp, onAxis, w);
            check(w > 0.0f, "on-axis point is in front of its face (w>0)");
            check(approx(ndc.x, 0.0f), "on-axis projects to clip center x~0");
            check(approx(ndc.y, 0.0f), "on-axis projects to clip center y~0");
            check(ndc.z >= -1e-3f && ndc.z <= 1.0f + 1e-3f, "on-axis depth in [0,1]");

            // The face whose direction is straight out should be the one SelectFace picks for that
            // direction (render face == sampled face).
            check(ps::SelectFace(onAxis - P) == face, "SelectFace agrees with FaceDir for on-axis");
        }
    }

    // ---- A point BEHIND a face (opposite axis) projects outside the [-1,1] frustum or w<=0. ----
    {
        for (int face = 0; face < ps::kFaces; ++face) {
            Vec3 dir = ps::FaceDir(face);
            Vec3 behind = P - dir * 8.0f;   // straight out the OPPOSITE axis
            Mat4 vp = ps::FaceViewProj(P, face, nearZ, range);
            float w = 0.0f;
            Vec3 ndc = project(vp, behind, w);
            bool outside = (w <= 0.0f) || (ndc.x < -1.0f || ndc.x > 1.0f ||
                                           ndc.y < -1.0f || ndc.y > 1.0f);
            check(outside, "point behind a face projects outside that face's frustum");
        }
    }

    // ---- Tile mapping: 3x2 grid, col=face%3, row=face/3. ----
    {
        check(ps::FaceTile(0).col == 0 && ps::FaceTile(0).row == 0, "face 0 tile (0,0)");
        check(ps::FaceTile(2).col == 2 && ps::FaceTile(2).row == 0, "face 2 tile (2,0)");
        check(ps::FaceTile(3).col == 0 && ps::FaceTile(3).row == 1, "face 3 tile (0,1)");
        check(ps::FaceTile(5).col == 2 && ps::FaceTile(5).row == 1, "face 5 tile (2,1)");
    }

    if (g_fail == 0) std::printf("point_shadow_test: all checks passed\n");
    else std::printf("point_shadow_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
