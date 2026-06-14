#include "math/math.h"
#include <cmath>
#include <cstdio>

using namespace hf::math;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

int main() {
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

    if (g_fail == 0) { std::printf("math_test OK\n"); return 0; }
    std::printf("math_test: %d failures\n", g_fail);
    return 1;
}
