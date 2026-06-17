// Slice DX — Virtual-Geometry Visibility-Buffer Slice 2: DEFERRED MATERIAL RESOLVE. Pure CPU
// (header-only, no device, no backend symbols). Mirrors engine/render/visresolve.h, the SAME flat-shade
// resolve math the --visresolve-shot (Vulkan) / --visresolve (Metal) showcases use AND the verbatim
// math shaders/visresolve.frag.hlsl copies. Namespace hf::render::vg.
//
// What this test PINS (the contracts the GPU visresolve.frag.hlsl + the GPU==CPU proof build on):
//   * FlatNormal of a known triangle is the UNIT outward geometric normal (correct direction + magnitude
//     1) and FLIPS sign when the winding reverses (orientation contract).
//   * LambertShade: a face-on normal gives albedo*(ambient + 1); a BACK-FACING normal CLAMPS the diffuse
//     to 0 (only ambient survives) — the max(0, ndotl) contract; the result tracks the albedo per channel.
//   * ResolveFlatShade over a known 1-triangle mesh == the hand-composed FlatNormal+LambertShade.
//   * ResolvePixel: the background sentinel (kVisBackground) AND an out-of-range clusterID both resolve to
//     the SKY color with NO geometry fetch; a valid packed ID resolves to the covering triangle's shade.
//   * EncodeBGRA8 saturates + rounds into B,G,R,A byte order (the golden-image encode).
//   * DETERMINISM: two resolves of the same input are bit-identical.
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/visresolve.h"
#include "render/visbuffer.h"
#include "render/cluster_cull.h"
#include "render/frustum.h"
#include "render/meshlet.h"
#include "scene/mesh.h"
#include "scene/vertex.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace vg = hf::render::vg;
namespace fr = hf::render::frustum;
namespace mdi = hf::render::mdi;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }
static bool vapprox(const math::Vec3& a, const math::Vec3& b, float eps = 1e-5f) {
    return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) && approx(a.z, b.z, eps);
}

int main() {
    HF_TEST_MAIN_INIT();
    using math::Vec3; using math::Mat4;

    // ================= FlatNormal: unit outward normal + orientation =================
    {
        // A triangle in the z=0 plane wound CCW looking from +Z: p0,p1,p2 -> cross(e1,e2) = +Z.
        Vec3 p0{0, 0, 0}, p1{1, 0, 0}, p2{0, 1, 0};
        Vec3 n = vg::FlatNormal(p0, p1, p2);
        check(vapprox(n, Vec3{0, 0, 1}), "FlatNormal CCW(+Z) -> +Z unit normal");
        check(approx(std::sqrt(math::dot(n, n)), 1.0f), "FlatNormal is unit length");
        // Reverse the winding -> the normal flips to -Z (orientation contract).
        Vec3 nFlip = vg::FlatNormal(p0, p2, p1);
        check(vapprox(nFlip, Vec3{0, 0, -1}), "FlatNormal reversed winding -> -Z (sign flip)");
        // A non-axis-aligned triangle: the normal is still unit + perpendicular to both edges.
        Vec3 q0{1, 2, 3}, q1{4, 0, 1}, q2{0, 5, 2};
        Vec3 nq = vg::FlatNormal(q0, q1, q2);
        check(approx(std::sqrt(math::dot(nq, nq)), 1.0f), "FlatNormal arbitrary tri is unit");
        check(approx(math::dot(nq, q1 - q0), 0.0f, 1e-4f), "FlatNormal perpendicular to e1");
        check(approx(math::dot(nq, q2 - q0), 0.0f, 1e-4f), "FlatNormal perpendicular to e2");
    }

    // ================= LambertShade: face-on, back-facing clamp, ambient =================
    {
        vg::ResolveMaterial mat;
        mat.lightDir = math::normalize(Vec3{0, 0, -1});  // light travels toward -Z
        mat.albedo = Vec3{0.8f, 0.5f, 0.2f};
        mat.ambient = 0.1f;
        // A normal pointing straight at the light source (+Z): ndotl = -dot(+Z, -Z) = 1 -> fully lit.
        Vec3 faceOn = vg::LambertShade(Vec3{0, 0, 1}, mat);
        check(vapprox(faceOn, Vec3{0.8f * 1.1f, 0.5f * 1.1f, 0.2f * 1.1f}),
              "LambertShade face-on -> albedo*(ambient+1)");
        // A normal pointing AWAY from the light (-Z): ndotl = -dot(-Z,-Z) = -1 -> clamped to 0 -> ambient only.
        Vec3 backFace = vg::LambertShade(Vec3{0, 0, -1}, mat);
        check(vapprox(backFace, Vec3{0.8f * 0.1f, 0.5f * 0.1f, 0.2f * 0.1f}),
              "LambertShade back-facing -> ambient only (max(0,ndotl)=0)");
        check(backFace.x >= 0.0f && backFace.y >= 0.0f && backFace.z >= 0.0f,
              "LambertShade back-facing is non-negative");
        // A grazing normal (perpendicular to the light dir): ndotl = 0 -> ambient only too.
        Vec3 grazing = vg::LambertShade(Vec3{1, 0, 0}, mat);
        check(vapprox(grazing, Vec3{0.8f * 0.1f, 0.5f * 0.1f, 0.2f * 0.1f}),
              "LambertShade grazing -> ambient only (ndotl=0)");
    }

    // ================= ResolveFlatShade over a known 1-triangle mesh =================
    {
        // One triangle in the z=0 plane (CCW from +Z) -> outward normal +Z. Identity model.
        std::vector<scene::Vertex> verts(3);
        auto setPos = [&](int i, float x, float y, float z) {
            verts[i] = scene::Vertex{};
            verts[i].pos[0] = x; verts[i].pos[1] = y; verts[i].pos[2] = z;
        };
        setPos(0, 0, 0, 0); setPos(1, 1, 0, 0); setPos(2, 0, 1, 0);
        std::vector<uint32_t> indices = {0, 1, 2};
        vg::ResolveMaterial mat;
        mat.lightDir = math::normalize(Vec3{0, 0, -1});
        mat.albedo = Vec3{0.6f, 0.6f, 0.6f};
        mat.ambient = 0.2f;
        Mat4 model = Mat4::Identity();
        Vec3 shade = vg::ResolveFlatShade(/*triID*/0, /*triOffset*/0, model,
                                          std::span<const uint32_t>(indices.data(), indices.size()),
                                          std::span<const scene::Vertex>(verts.data(), verts.size()), mat);
        // The normal is +Z, light from -Z -> ndotl=1 -> albedo*(0.2+1.0) = 0.6*1.2 = 0.72 per channel.
        Vec3 expectN = vg::FlatNormal(Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0});
        Vec3 expect = vg::LambertShade(expectN, mat);
        check(vapprox(shade, expect), "ResolveFlatShade == FlatNormal+LambertShade composition");
        check(vapprox(shade, Vec3{0.72f, 0.72f, 0.72f}), "ResolveFlatShade hand-computed value");

        // A NON-trivial triOffset: place the same triangle at triangle slot 5 of a longer index buffer
        // and pass triOffset so 3*(triOffset+triID) lands on it -> the same shade (offset addressing).
        std::vector<uint32_t> longIdx(3 * 6, 0);
        longIdx[3 * 5 + 0] = 0; longIdx[3 * 5 + 1] = 1; longIdx[3 * 5 + 2] = 2;
        Vec3 shadeOff = vg::ResolveFlatShade(/*triID*/0, /*triOffset*/5, model,
                                             std::span<const uint32_t>(longIdx.data(), longIdx.size()),
                                             std::span<const scene::Vertex>(verts.data(), verts.size()), mat);
        check(vapprox(shadeOff, expect), "ResolveFlatShade honors triOffset addressing");

        // A TRANSLATION model leaves the flat normal (edge-based) unchanged -> identical shade.
        Mat4 trans = Mat4::Translate({10.0f, -3.0f, 4.0f});
        Vec3 shadeTrans = vg::ResolveFlatShade(0, 0, trans,
                                               std::span<const uint32_t>(indices.data(), indices.size()),
                                               std::span<const scene::Vertex>(verts.data(), verts.size()), mat);
        check(vapprox(shadeTrans, expect), "ResolveFlatShade translation-invariant normal -> same shade");
    }

    // ================= ResolvePixel: sentinel + out-of-range -> sky; valid -> shade =================
    {
        std::vector<scene::Vertex> verts(3);
        for (int i = 0; i < 3; ++i) verts[i] = scene::Vertex{};
        verts[1].pos[0] = 1.0f; verts[2].pos[1] = 1.0f;  // (0,0,0),(1,0,0),(0,1,0)
        std::vector<uint32_t> indices = {0, 1, 2};
        std::vector<Mat4> models = {Mat4::Identity()};
        // One survivor cluster-instance covering this triangle; survivor cmd maps draw 0 -> source 0.
        vg::ClusterInstance ci{};
        ci.triOffset = 0; ci.triCount = 1; ci.instanceIndex = 0;
        std::vector<vg::ClusterInstance> clusters = {ci};
        mdi::MdiCommand cmd{}; cmd.firstInstance = 0;
        std::vector<mdi::MdiCommand> cmds = {cmd};
        vg::ResolveMaterial mat = vg::DefaultResolveMaterial();
        const uint32_t drawn = 1;

        auto idxSpan = std::span<const uint32_t>(indices.data(), indices.size());
        auto vtxSpan = std::span<const scene::Vertex>(verts.data(), verts.size());
        auto cmdSpan = std::span<const mdi::MdiCommand>(cmds.data(), cmds.size());
        auto cluSpan = std::span<const vg::ClusterInstance>(clusters.data(), clusters.size());
        auto mdlSpan = std::span<const Mat4>(models.data(), models.size());

        // The background sentinel resolves to the sky color, NO geometry fetch.
        Vec3 bg = vg::ResolvePixel(vg::kVisBackground, drawn, cmdSpan, cluSpan, mdlSpan, idxSpan, vtxSpan, mat);
        check(vapprox(bg, vg::ResolveSkyColor()), "ResolvePixel(sentinel) -> sky color");

        // An out-of-range clusterID (cid >= drawn) ALSO resolves to sky (the shader's guard branch).
        uint32_t oob = vg::PackVisId(/*cid*/5, /*tid*/0);  // 5 >= drawn(1)
        Vec3 oobShade = vg::ResolvePixel(oob, drawn, cmdSpan, cluSpan, mdlSpan, idxSpan, vtxSpan, mat);
        check(vapprox(oobShade, vg::ResolveSkyColor()), "ResolvePixel(cid>=drawn) -> sky color");

        // A VALID packed ID resolves to the covering triangle's flat shade (== ResolveFlatShade).
        uint32_t valid = vg::PackVisId(/*cid*/0, /*tid*/0);
        Vec3 got = vg::ResolvePixel(valid, drawn, cmdSpan, cluSpan, mdlSpan, idxSpan, vtxSpan, mat);
        Vec3 want = vg::ResolveFlatShade(0, ci.triOffset, models[0], idxSpan, vtxSpan, mat);
        check(vapprox(got, want), "ResolvePixel(valid ID) -> covering-triangle shade");
        check(!vapprox(got, vg::ResolveSkyColor()), "ResolvePixel(valid ID) != sky (geometry was shaded)");
    }

    // ================= EncodeBGRA8: saturate + round into B,G,R,A =================
    {
        uint8_t out[4];
        vg::EncodeBGRA8(Vec3{1.0f, 0.5f, 0.0f}, out);  // R=1, G=0.5, B=0
        check(out[0] == 0, "EncodeBGRA8 B channel (=0)");
        check(out[1] == 128, "EncodeBGRA8 G channel (0.5 -> 128 round)");
        check(out[2] == 255, "EncodeBGRA8 R channel (=255)");
        check(out[3] == 255, "EncodeBGRA8 A = 255");
        // Over-range saturates to 1.0 (255), negative clamps to 0.
        uint8_t sat[4];
        vg::EncodeBGRA8(Vec3{2.0f, -1.0f, 0.999f}, sat);
        check(sat[2] == 255, "EncodeBGRA8 saturates >1 to 255");
        check(sat[0] == 255, "EncodeBGRA8 0.999 -> 255 (B)");
        check(sat[1] == 0, "EncodeBGRA8 clamps <0 to 0 (G)");
    }

    // ================= DefaultResolveMaterial: light dir is pre-normalized =================
    {
        vg::ResolveMaterial m = vg::DefaultResolveMaterial();
        float len = std::sqrt(math::dot(m.lightDir, m.lightDir));
        check(approx(len, 1.0f), "DefaultResolveMaterial light dir is unit (pre-normalized)");
        check(m.ambient > 0.0f && m.ambient < 1.0f, "DefaultResolveMaterial ambient in (0,1)");
    }

    // ================= DETERMINISM: two resolves bit-identical =================
    {
        std::vector<scene::Vertex> verts(3);
        for (int i = 0; i < 3; ++i) verts[i] = scene::Vertex{};
        verts[1].pos[0] = 1.0f; verts[2].pos[1] = 1.0f;
        std::vector<uint32_t> indices = {0, 1, 2};
        vg::ResolveMaterial mat = vg::DefaultResolveMaterial();
        Mat4 model = Mat4::Translate({1.0f, 2.0f, -3.0f});
        auto idxSpan = std::span<const uint32_t>(indices.data(), indices.size());
        auto vtxSpan = std::span<const scene::Vertex>(verts.data(), verts.size());
        Vec3 a = vg::ResolveFlatShade(0, 0, model, idxSpan, vtxSpan, mat);
        Vec3 b = vg::ResolveFlatShade(0, 0, model, idxSpan, vtxSpan, mat);
        check(std::memcmp(&a, &b, sizeof(Vec3)) == 0, "ResolveFlatShade two calls bit-identical");
        uint8_t e1[4], e2[4];
        vg::EncodeBGRA8(a, e1); vg::EncodeBGRA8(b, e2);
        check(std::memcmp(e1, e2, 4) == 0, "EncodeBGRA8 two calls bit-identical");
    }

    if (g_fail == 0) std::printf("visresolve_test: ALL PASS\n");
    else std::printf("visresolve_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
