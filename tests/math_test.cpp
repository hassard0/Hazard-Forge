#include "math/math.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

int main() {
    HF_TEST_MAIN_INIT();
    // Identity * Identity == Identity
    Mat4 i = Mat4::Identity();
    Mat4 ii = i * i;
    for (int k = 0; k < 16; ++k) check(approx(ii.m[k], i.m[k]), "I*I==I");

    // Translate composes: T(1,2,3) applied to origin via column-major MVP-style multiply.
    Mat4 t = Mat4::Translate({1, 2, 3});
    check(approx(t.m[12], 1) && approx(t.m[13], 2) && approx(t.m[14], 3), "translate col3");

    // LookAt from (0,0,5) toward origin: eye maps to ~origin in view space (translation z = -5 along -f).
    Mat4 v = Mat4::LookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0});
    // For a point at world origin, view-space z should be -5 (in front of camera, RH -z forward).
    // p_view = V * [0,0,0,1]; z component = m[2]*0 + m[6]*0 + m[10]*0 + m[14].
    check(approx(v.m[14], -5.0f), "lookat translates eye to -5 view z");

    // Perspective basic sanity: finite, m[11] == -1, aspect scaling on x.
    Mat4 p = Mat4::Perspective(1.0472f /*60deg*/, 16.0f/9.0f, 0.1f, 100.0f);
    check(approx(p.m[11], -1.0f), "persp m[11]==-1");
    check(p.m[0] > 0 && p.m[5] < 0, "persp x>0, y<0 (Vulkan flip)");

    // Associativity: (P*V)*M == P*(V*M)
    Mat4 m = Mat4::RotateY(0.7f);
    Mat4 lhs = (p * v) * m;
    Mat4 rhs = p * (v * m);
    for (int k = 0; k < 16; ++k) check(approx(lhs.m[k], rhs.m[k], 1e-3f), "assoc");

    // Scale: diagonal s.x,s.y,s.z; m[15]==1.
    Mat4 sc = Mat4::Scale({2, 3, 4});
    check(approx(sc.m[0], 2) && approx(sc.m[5], 3) && approx(sc.m[10], 4), "scale diagonal");
    check(approx(sc.m[15], 1), "scale m[15]==1");

    // Ortho: center of the box maps to ~0 in x/y; zn->0 and zf->1 in z (Vulkan [0,1] depth).
    Mat4 o = Mat4::Ortho(-8, 8, -8, 8, 1, 25);
    {
        // Box center in x/y, at the near plane z=zn: transform (0,0,-1,1) (RH: -z forward).
        // Column-major apply: out.c = sum_k m[k*4+c] * v[k].
        auto applyZ = [&](float pz) {
            return o.m[2]*0 + o.m[6]*0 + o.m[10]*(-pz) + o.m[14];
        };
        auto applyW = [&](float pz) {
            return o.m[3]*0 + o.m[7]*0 + o.m[11]*(-pz) + o.m[15];
        };
        float xCenter = o.m[0]*0 + o.m[12];   // x at world x=0
        float yCenter = o.m[5]*0 + o.m[13];   // y at world y=0
        check(approx(xCenter, 0.0f), "ortho center x->0");
        check(approx(yCenter, 0.0f), "ortho center y->0");
        check(approx(applyZ(1.0f) / applyW(1.0f), 0.0f), "ortho zn->0");
        check(approx(applyZ(25.0f) / applyW(25.0f), 1.0f), "ortho zf->1");
    }

    // RotateZ(0) == Identity (all 16).
    Mat4 rz0 = Mat4::RotateZ(0.0f);
    Mat4 id = Mat4::Identity();
    for (int k = 0; k < 16; ++k) check(approx(rz0.m[k], id.m[k]), "RotateZ(0)==Identity");

    // --- Perspective known values ------------------------------------------------------------------
    // fov=90deg (tan(45)=1), aspect=2 -> m[0]=1/(2*1)=0.5, m[5]=-1/1=-1.
    // zn=1, zf=11 -> m[10]=zf/(zn-zf)=11/-10=-1.1; m[14]=(zn*zf)/(zn-zf)=11/-10=-1.1.
    {
        Mat4 pk = Mat4::Perspective(1.5707963f /*90deg*/, 2.0f, 1.0f, 11.0f);
        check(approx(pk.m[0], 0.5f), "persp m[0]=1/(aspect*tan) known value");
        check(approx(pk.m[5], -1.0f), "persp m[5]=-1/tan known value");
        check(approx(pk.m[10], -1.1f), "persp m[10] known value");
        check(approx(pk.m[14], -1.1f), "persp m[14] known value");
        // A point on the far plane (view z = -zf) maps to clip z/w = 1 (Vulkan [0,1] depth).
        float zf = 11.0f;
        float clipZ = pk.m[10] * (-zf) + pk.m[14];
        float clipW = pk.m[11] * (-zf);
        check(approx(clipZ / clipW, 1.0f), "persp far plane -> NDC z 1");
        // Near plane (view z = -zn) maps to clip z/w = 0.
        float zn = 1.0f;
        float nClipZ = pk.m[10] * (-zn) + pk.m[14];
        float nClipW = pk.m[11] * (-zn);
        check(approx(nClipZ / nClipW, 0.0f), "persp near plane -> NDC z 0");
    }

    // --- Vec3 ops (cross / dot / normalize) — previously untested ----------------------------------
    {
        Vec3 x{1, 0, 0}, y{0, 1, 0};
        Vec3 z = cross(x, y);                 // right-handed: x cross y = +z
        check(approx(z.x, 0) && approx(z.y, 0) && approx(z.z, 1), "cross(x,y)=z (right-handed)");
        check(approx(dot(x, y), 0.0f), "dot of orthogonal axes is 0");
        check(approx(dot(x, x), 1.0f), "dot of unit axis with itself is 1");
        Vec3 n = normalize(Vec3{0, 3, 4});    // length 5 -> (0, 0.6, 0.8)
        check(approx(n.x, 0) && approx(n.y, 0.6f) && approx(n.z, 0.8f), "normalize(3-4-5) -> unit");
        check(approx(std::sqrt(dot(n, n)), 1.0f), "normalized vector has unit length");
        Vec3 zero = normalize(Vec3{0, 0, 0}); // degenerate: returns input unchanged, no NaN/div0
        check(approx(zero.x, 0) && approx(zero.y, 0) && approx(zero.z, 0), "normalize(0) is safe");
    }

    // --- RotateX / RotateZ rotate basis vectors by 90deg correctly ---------------------------------
    {
        // RotateX(90): +y -> +z. Column-major apply: out.row = sum_k m[k*4+row]*v[k].
        Mat4 rx = Mat4::RotateX(1.5707963f);
        auto applyRow = [](const Mat4& M, const Vec3& v, int row) {
            return M.m[0*4+row]*v.x + M.m[1*4+row]*v.y + M.m[2*4+row]*v.z;
        };
        Vec3 yA{0, 1, 0};
        check(approx(applyRow(rx, yA, 0), 0.0f) && approx(applyRow(rx, yA, 1), 0.0f) &&
              approx(applyRow(rx, yA, 2), 1.0f), "RotateX(90) maps +y to +z");
        // RotateZ(90): +x -> +y.
        Mat4 rz = Mat4::RotateZ(1.5707963f);
        Vec3 xA{1, 0, 0};
        check(approx(applyRow(rz, xA, 0), 0.0f) && approx(applyRow(rz, xA, 1), 1.0f) &&
              approx(applyRow(rz, xA, 2), 0.0f), "RotateZ(90) maps +x to +y");
    }

    if (g_fail == 0) { std::printf("math_test OK\n"); return 0; }
    std::printf("math_test: %d failures\n", g_fail);
    return 1;
}
