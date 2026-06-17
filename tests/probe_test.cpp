// Slice AK — reflection + irradiance probes. Pure CPU math: the 6-face cube projection +
// face-selection (reused from point_shadow) and the COMBINED reflection/irradiance atlas-tile
// mapping the lit_probe shader mirrors. No device, ASan-eligible (links hf_core).
#include "render/probe.h"
#include "render/point_shadow.h"
#include "math/math.h"
#include <cmath>
#include <cstdio>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace pr = hf::render::probe;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 2e-3f) { return std::fabs(a - b) < eps; }

int main() {
    HF_TEST_MAIN_INIT();
    // ---- A direction straight along +X maps to the +X face, at the face CENTER (uv ~ 0.5,0.5). ----
    {
        int face = -1;
        pr::UV uv = pr::FaceUVFromDir(Vec3{1, 0, 0}, face);
        check(face == 0, "dir +X -> face 0 (+X)");
        check(approx(uv.u, 0.5f) && approx(uv.v, 0.5f), "on-axis +X projects to face center");

        pr::UV uvy = pr::FaceUVFromDir(Vec3{0, 1, 0}, face);
        check(face == 2, "dir +Y -> face 2 (+Y)");
        check(approx(uvy.u, 0.5f) && approx(uvy.v, 0.5f), "on-axis +Y projects to face center");

        pr::UV uvz = pr::FaceUVFromDir(Vec3{0, 0, -1}, face);
        check(face == 5, "dir -Z -> face 5 (-Z)");
        check(approx(uvz.u, 0.5f) && approx(uvz.v, 0.5f), "on-axis -Z projects to face center");
    }

    // ---- SelectFace agrees with FaceDir, and the round-trip face -> dir -> face is consistent. ----
    {
        for (int face = 0; face < pr::kFaces; ++face) {
            Vec3 dir = pr::FaceDir(face);
            check(pr::SelectFace(dir) == face, "FaceDir(face) selects back to face");
            int rf = -1;
            pr::UV uv = pr::FaceUVFromDir(dir, rf);
            check(rf == face, "FaceUVFromDir picks the same face as FaceDir");
            check(approx(uv.u, 0.5f) && approx(uv.v, 0.5f), "face-dir centers in its tile");
        }
    }

    // ---- A probe face view-proj projects an on-axis point to clip center, depth in [0,1]. ----
    {
        const Vec3 P{2.0f, 1.0f, -3.0f};   // arbitrary probe position
        for (int face = 0; face < pr::kFaces; ++face) {
            Vec3 dir = pr::FaceDir(face);
            Vec3 onAxis = P + dir * 7.0f;
            Mat4 vp = pr::FaceViewProj(P, face);
            float w = 0.0f;
            Vec3 ndc = MulPointDivide(vp, onAxis, w);
            check(w > 0.0f, "on-axis point in front of its face (w>0)");
            check(approx(ndc.x, 0.0f), "on-axis projects to clip center x~0");
            check(approx(ndc.y, 0.0f), "on-axis projects to clip center y~0");
            check(ndc.z >= -1e-3f && ndc.z <= 1.0f + 1e-3f, "on-axis depth in [0,1]");
        }
    }

    // ---- Reflection / irradiance tile UVs land INSIDE their block, in the right tile. ----
    {
        // Reflection block occupies the top kReflBlockH/kAtlasH of V; irradiance the rest.
        const float reflVMax = (float)pr::kReflBlockH / (float)pr::kAtlasH;  // ~0.889
        for (int face = 0; face < pr::kFaces; ++face) {
            pr::UV center{0.5f, 0.5f};
            pr::UV r = pr::ReflTileUV(face, center);
            check(r.u >= 0.0f && r.u <= 1.0f, "refl atlas u in [0,1]");
            check(r.v >= 0.0f && r.v <= reflVMax + 1e-4f, "refl atlas v in reflection block");
            // tile center: col=face%3, row=face/3 over a 3x2 grid of 512-tiles in 1536x1024.
            float expU = ((float)(face % 3) + 0.5f) * ((float)pr::kReflTile / (float)pr::kAtlasW);
            float expV = ((float)(face / 3) + 0.5f) * ((float)pr::kReflTile / (float)pr::kAtlasH);
            check(approx(r.u, expU, 1e-4f) && approx(r.v, expV, 1e-4f), "refl tile center exact");

            pr::UV ir = pr::IrrTileUV(face, center);
            check(ir.v >= reflVMax - 1e-4f && ir.v <= 1.0f, "irr atlas v in irradiance block");
            check(ir.u >= 0.0f && ir.u <= 1.0f, "irr atlas u in [0,1]");
        }
    }

    // ---- Atlas geometry sanity. ----
    {
        check(pr::kAtlasW == 1536, "atlas width 1536");
        check(pr::kAtlasH == 1152, "atlas height 1152 (1024 refl + 128 irr)");
    }

    if (g_fail == 0) std::printf("probe_test: all checks passed\n");
    else std::printf("probe_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
