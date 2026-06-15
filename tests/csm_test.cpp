// Slice AD — cascaded shadow maps. Pure CPU math: frustum split scheme + per-cascade ortho fit.
// No device, ASan-eligible (links hf_core). Mirrors the math the --csm-shot showcase uses.
#include "render/csm.h"
#include "math/math.h"
#include <array>
#include <cmath>
#include <cstdio>

using namespace hf::math;
namespace csm = hf::render::csm;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) < eps; }

int main() {
    // ---- Split scheme: N increasing splits within (near, far], last == far. ----
    {
        const float nearZ = 0.5f, farZ = 60.0f;
        const int N = 4;
        auto s = csm::CsmSplits(nearZ, farZ, N, 0.5f);
        check(s[0] > nearZ, "split0 > near");
        for (int i = 1; i < N; ++i) check(s[i] > s[i - 1], "splits strictly increasing");
        check(s[N - 1] <= farZ + 1e-3f && s[N - 1] >= farZ - 1e-3f, "last split == far");
        for (int i = 0; i < N; ++i) check(s[i] > nearZ && s[i] <= farZ + 1e-3f, "split within (near,far]");
        // lambda=0 (pure uniform) gives evenly spaced splits.
        auto su = csm::CsmSplits(nearZ, farZ, N, 0.0f);
        float step = (farZ - nearZ) / N;
        for (int i = 0; i < N; ++i) check(approx(su[i], nearZ + step * (i + 1), 1e-2f), "uniform split spacing");
    }

    // ---- Per-cascade ortho fit: every one of the 8 slice corners lands inside the cascade's
    //      clip volume ([-1,1]^2 in x/y, [0,1] in z). Run for a few cascades of a real config. ----
    {
        // Camera: eye at (0,4,12) looking toward origin, 60deg fovY, 16:9.
        Vec3 eye{0.0f, 4.0f, 12.0f};
        Vec3 center{0.0f, 1.0f, 0.0f};
        Vec3 f = normalize(center - eye);
        Vec3 r = normalize(cross(f, Vec3{0, 1, 0}));
        Vec3 u = cross(r, f);
        float fovY = 1.04719755f;
        float tanHalf = std::tan(0.5f * fovY);
        float aspect = 16.0f / 9.0f;

        Vec3 lightDir = normalize(Vec3{-0.6f, -1.0f, -0.35f});

        const float nearZ = 0.5f, farZ = 60.0f;
        const int N = 4;
        auto splits = csm::CsmSplits(nearZ, farZ, N, 0.5f);

        float sliceNear = nearZ;
        for (int c = 0; c < N; ++c) {
            float sliceFar = splits[c];
            auto corners = csm::FrustumSliceCornersWorld(eye, f, r, u, tanHalf, aspect,
                                                         sliceNear, sliceFar);
            auto fit = csm::FitCascadeLightMatrix(corners, lightDir, 10.0f);
            const Mat4& VP = fit.lightViewProj;
            for (int k = 0; k < 8; ++k) {
                const Vec3& p = corners[k];
                float cx = VP.m[0]*p.x + VP.m[4]*p.y + VP.m[8]*p.z  + VP.m[12];
                float cy = VP.m[1]*p.x + VP.m[5]*p.y + VP.m[9]*p.z  + VP.m[13];
                float cz = VP.m[2]*p.x + VP.m[6]*p.y + VP.m[10]*p.z + VP.m[14];
                float cw = VP.m[3]*p.x + VP.m[7]*p.y + VP.m[11]*p.z + VP.m[15];
                check(approx(cw, 1.0f, 1e-3f), "ortho w==1");
                float ndcX = cx / cw, ndcY = cy / cw, ndcZ = cz / cw;
                // Tolerances: corners should be inside [-1,1]^2; the X/Y AABB fit is exact so the
                // extremes hit +-1. Z within [0,1] (zPadNear pushes near out so corners sit > 0).
                check(ndcX >= -1.0f - 1e-3f && ndcX <= 1.0f + 1e-3f, "corner ndc x in [-1,1]");
                check(ndcY >= -1.0f - 1e-3f && ndcY <= 1.0f + 1e-3f, "corner ndc y in [-1,1]");
                check(ndcZ >= 0.0f - 1e-3f && ndcZ <= 1.0f + 1e-3f, "corner ndc z in [0,1]");
            }
            sliceNear = sliceFar;
        }
    }

    if (g_fail == 0) std::printf("csm_test: all checks passed\n");
    else std::printf("csm_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
