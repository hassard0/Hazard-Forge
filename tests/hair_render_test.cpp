// Slice HRR1 — THE HAIR STRAND RENDERER (engine/render/hair_render.h): tapered CAMERA-FACING ribbon
// geometry over the HR1 deterministic strand sim, carrying the strand TANGENT for Kajiya-Kay shading
// (shaders/hair_kajiya.frag.hlsl). Pure CPU (header-only, no device / backend symbols). Namespace
// hf::render::hairr.
//
// What this test PINS:
//   * MakeGroomScene shape: 64 strands x 12 verts, 2 pinned per strand, roots ON the scalp-sphere cap
//     (|root - center| == R within the fxmul truncation LSBs), the k_bend 0 -> kOne ramp.
//   * THE SIM PROVENANCE: the settled groom (kGroomSteps) two-run BYTE-IDENTICAL + HairDigest PINNED
//     (must be identical under MSVC and clang — the integer layer's cross-compiler proof).
//   * THE RIBBON FRAME, hand-checked: a synthetic straight-up strand with a +Z camera gives
//     side == +X, normal == +Z (toward the camera), left/right = P -/+ side*halfW EXACT, the
//     root->tip taper halfW = 0.5*lerp(wRoot, wTip, t) exact at t = 0 / 0.5 / 1, v = t.
//   * DEGENERATE GUARDS: tangent PARALLEL to viewDir -> the cross(T, up) fallback (never NaN, side
//     unit); tangent parallel to BOTH viewDir and up -> the +X fallback; M < 2 / size mismatch /
//     both widths <= 0 -> the EMPTY mesh.
//   * FRAME INVARIANTS over the full groom mesh: every tangent unit, normal unit, side implied
//     orthogonal (|dot(N, T)| ~ 0), v monotone root -> tip per strand.
//   * COUNT CONTRACT: verts == S*2M, indices == S*(M-1)*6, NO repeated index in any triple (the
//     pinned SweepStrip winding).
//   * DETERMINISM: two mesh builds from the same state BYTE-IDENTICAL (memcmp).
//   * THE FLOAT-LAYER PIN: HairRenderDigest over the QUANTIZED mesh from the pinned sim state +
//     the SHARED showcase camera/widths — PINNED, must match under MSVC and clang (the std::fma
//     discipline makes the mesh floats bit-stable).
//
// Pure C++ (hf_core), ASan-eligible like the other render/sim tests.
#include "render/hair_render.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"     // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace hairr = hf::render::hairr;
namespace hair  = hf::sim::hair;
using hairr::fx;
using hairr::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// ---- THE PINNED CONSTANTS (computed once, then locked; MSVC == clang or the pin fails) ------------
// The settled groom sim digest (hair::HairDigest after hairr::kGroomSteps) and the quantized ribbon
// mesh digest (hairr::HairRenderDigest with the SHARED showcase camera/widths).
static constexpr uint64_t kPinnedGroomDigest = 0x2fe41334235921fdull;
static constexpr uint64_t kPinnedMeshDigest  = 0x66c6858b52521bcfull;

static bool Unit(float x, float y, float z, float tol) {
    const float len2 = x * x + y * y + z * z;
    return std::fabs(len2 - 1.0f) <= tol;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= MakeGroomScene: the deterministic scalp-cap groom =========================
    hairr::GroomScene g = hairr::MakeGroomScene();
    {
        check(g.hs.S == 64 && g.hs.M == 12, "groom: 64 strands x 12 verts");
        check(g.verts.size() == 768, "groom: vert count == S*M");
        int pinned = 0;
        for (const hair::HairVert& v : g.verts)
            if (v.flags & hair::kFlagPinned) ++pinned;
        check(pinned == 128, "groom: 2 pinned verts per strand (direction-clamped roots)");
        // Roots ON the scalp cap: |root - center| == R within the fxmul truncation LSBs.
        const hairr::FxVec3 C{0, (fx)(6 * (int)kOne), 0};
        const fx R = (fx)(kOne * 3 / 2);
        bool onCap = true;
        for (int s = 0; s < g.hs.S; ++s) {
            const hair::HairVert& r = g.verts[(size_t)hair::VertIndex(g.hs, s, 0)];
            const fx d = hair::FxLength(hair::FxSub(r.pos, C));
            fx err = d - R; if (err < 0) err = -err;
            if (err > 8) onCap = false;   // 8 LSBs ~ 0.00012 wu (FxNormalize + fxmul truncation)
        }
        check(onCap, "groom: every root sits on the scalp sphere (|root-C| == R within 8 LSBs)");
        // The k_bend ramp: strand 0 limp, strand 63 exactly kOne, monotone.
        check(g.kBend.size() == 64 && g.kBend[0] == 0 && g.kBend[63] == kOne,
              "groom: k_bend ramp endpoints 0 and kOne");
        bool mono = true;
        for (int s = 1; s < 64; ++s) if (g.kBend[(size_t)s] < g.kBend[(size_t)s - 1]) mono = false;
        check(mono, "groom: k_bend ramp monotone");
    }

    // ================= The settled sim: two-run byte-identical + the PINNED digest ================
    std::vector<hair::HairVert> settled = g.verts;
    hair::StepHairSteps(g.hs, settled, g.hc, g.excl, g.kBend, g.params, hairr::kGroomSteps);
    {
        std::vector<hair::HairVert> b = g.verts;
        hair::StepHairSteps(g.hs, b, g.hc, g.excl, g.kBend, g.params, hairr::kGroomSteps);
        check(settled.size() == b.size() &&
              std::memcmp(settled.data(), b.data(), settled.size() * sizeof(hair::HairVert)) == 0,
              "sim: two groom runs BYTE-IDENTICAL");
        const uint64_t digest = hair::HairDigest(settled);
        std::printf("groom sim digest: 0x%016llx\n", (unsigned long long)digest);
        check(digest == kPinnedGroomDigest, "sim: HairDigest == PINNED (MSVC/clang cross-compiler)");
        // The strands actually draped (the sim did something): the mean tip is BELOW the mean root.
        int64_t rootY = 0, tipY = 0;
        for (int s = 0; s < g.hs.S; ++s) {
            rootY += settled[(size_t)hair::VertIndex(g.hs, s, 0)].pos.y;
            tipY  += settled[(size_t)hair::VertIndex(g.hs, s, g.hs.M - 1)].pos.y;
        }
        check(tipY < rootY - (int64_t)g.hs.S * (int64_t)kOne,
              "sim: mean tip hangs >= 1 wu below the mean root (the drape happened)");
    }

    // ================= The hand-checked ribbon frame (a synthetic straight-up strand) ============
    {
        hair::HairStrands hs; hs.S = 1; hs.M = 3; hs.restLen = kOne;
        std::vector<hair::HairVert> vs(3);
        for (int i = 0; i < 3; ++i) {
            vs[(size_t)i].pos = hairr::FxVec3{0, (fx)(i * (int)kOne), 0};   // straight up +Y
            vs[(size_t)i].prev = vs[(size_t)i].pos;
            vs[(size_t)i].invMass = kOne; vs[(size_t)i].flags = 0;
        }
        std::vector<hairr::HairRenderVertex> mv;
        std::vector<uint32_t> mi;
        const fx wR = kOne / 4, wT = kOne / 8;   // taper 0.25 -> 0.125 full width
        hairr::HairToRenderMesh(hs, vs, math::Vec3{0.0f, 0.0f, 10.0f}, wR, wT, mv, mi);
        check(mv.size() == 6 && mi.size() == 12, "frame: 1x3 strand -> 6 verts, 12 indices");
        if (mv.size() == 6) {
            // Vert 0 (root, at the origin): T = +Y, viewDir ~ +Z, side = cross(T,V) = +X,
            // N = cross(side,T) = +Z (toward the camera). halfW(root) = 0.125 EXACT.
            check(mv[0].tx == 0.0f && mv[0].ty == 1.0f && mv[0].tz == 0.0f,
                  "frame: root tangent == +Y exact");
            check(mv[0].nx == 0.0f && mv[0].ny == 0.0f && mv[0].nz == 1.0f,
                  "frame: root normal == +Z (toward the camera) exact");
            check(mv[0].px == -0.125f && mv[1].px == 0.125f,
                  "frame: root rails at -/+ halfW = wRoot/2 exact");
            check(mv[0].u == 0.0f && mv[1].u == 1.0f, "frame: rail u = 0/1");
            check(mv[0].v == 0.0f && mv[2].v == 0.5f && mv[4].v == 1.0f,
                  "frame: v = the along-strand parameter 0 / 0.5 / 1");
            // The taper at the midpoint: halfW = 0.5*lerp(0.25, 0.125, 0.5) = 0.09375. The mid/tip
            // view directions are NOT axis-aligned (the normalize rounds +-ulp), so these rails are
            // checked to a 1e-5 band; the ROOT rail above is the exact-arithmetic pin.
            check(std::fabs(mv[2].px + 0.09375f) < 1e-5f && std::fabs(mv[3].px - 0.09375f) < 1e-5f,
                  "frame: mid taper halfW = 0.09375 (1e-5 band)");
            // The tip: halfW = 0.0625; the LAST vert reuses the previous segment's tangent.
            check(std::fabs(mv[4].px + 0.0625f) < 1e-5f && std::fabs(mv[5].px - 0.0625f) < 1e-5f,
                  "frame: tip taper halfW = 0.0625 (1e-5 band)");
            check(mv[4].ty == 1.0f, "frame: last vert reuses the previous segment tangent");
        }
        // The pinned SweepStrip winding: (a,b,c)(c,b,d) per span, no repeated index in any triple.
        if (mi.size() == 12) {
            const uint32_t want[12] = {0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5};
            check(std::memcmp(mi.data(), want, sizeof(want)) == 0,
                  "frame: the pinned SweepStrip winding, hand-enumerated");
        }
    }

    // ================= The degenerate guards ======================================================
    {
        // Tangent PARALLEL to viewDir: strand along +Z, camera straight down the +Z axis. The
        // camera-facing cross degenerates -> the cross(T, worldUp) fallback: side == -X, unit.
        hair::HairStrands hs; hs.S = 1; hs.M = 2; hs.restLen = kOne;
        std::vector<hair::HairVert> vs(2);
        vs[0].pos = hairr::FxVec3{0, 0, 0};
        vs[1].pos = hairr::FxVec3{0, 0, (fx)kOne};
        vs[0].prev = vs[0].pos; vs[1].prev = vs[1].pos;
        vs[0].invMass = vs[1].invMass = kOne;
        std::vector<hairr::HairRenderVertex> mv;
        std::vector<uint32_t> mi;
        hairr::HairToRenderMesh(hs, vs, math::Vec3{0.0f, 0.0f, 10.0f}, kOne / 8, kOne / 8, mv, mi);
        check(mv.size() == 4, "degen: parallel-to-view strand still emits its ribbon");
        bool finite = true, unit = true;
        for (const hairr::HairRenderVertex& v : mv) {
            if (!std::isfinite(v.px) || !std::isfinite(v.nx) || !std::isfinite(v.tx)) finite = false;
            if (!Unit(v.tx, v.ty, v.tz, 1e-4f)) unit = false;
        }
        check(finite && unit, "degen: the up-fallback frame is finite + unit (never NaN)");
        check(mv[0].px == 0.0625f && mv[1].px == -0.0625f,
              "degen: side == cross(T, up) == -X (the documented fallback)");

        // Tangent parallel to viewDir AND up: strand straight up, camera straight above -> +X.
        vs[1].pos = hairr::FxVec3{0, (fx)kOne, 0};
        vs[1].prev = vs[1].pos;
        hairr::HairToRenderMesh(hs, vs, math::Vec3{0.0f, 10.0f, 0.0f}, kOne / 8, kOne / 8, mv, mi);
        check(mv.size() == 4 && mv[0].px == -0.0625f && mv[1].px == 0.0625f,
              "degen: vertical strand + overhead camera -> the +X final fallback");

        // The EMPTY guards: M < 2 / size mismatch / both widths <= 0 -> zero geometry.
        hair::HairStrands bad; bad.S = 1; bad.M = 1; bad.restLen = kOne;
        hairr::HairToRenderMesh(bad, vs, math::Vec3{0, 0, 10.0f}, kOne, kOne, mv, mi);
        check(mv.empty() && mi.empty(), "degen: M < 2 -> empty mesh");
        hairr::HairToRenderMesh(hs, std::vector<hair::HairVert>(5), math::Vec3{0, 0, 10.0f},
                                kOne, kOne, mv, mi);
        check(mv.empty() && mi.empty(), "degen: size mismatch -> empty mesh");
        hairr::HairToRenderMesh(hs, vs, math::Vec3{0, 0, 10.0f}, 0, 0, mv, mi);
        check(mv.empty() && mi.empty(), "degen: both widths <= 0 -> empty mesh");
    }

    // ================= The full groom mesh: counts, invariants, determinism, THE PIN =============
    {
        const math::Vec3 cam{hairr::kGroomEyeX, hairr::kGroomEyeY, hairr::kGroomEyeZ};
        std::vector<hairr::HairRenderVertex> mv;
        std::vector<uint32_t> mi;
        hairr::HairToRenderMesh(g.hs, settled, cam, hairr::kGroomWidthRootQ,
                                hairr::kGroomWidthTipQ, mv, mi);
        check(mv.size() == (size_t)(g.hs.S * g.hs.M * 2), "mesh: verts == S * 2M");
        check(mi.size() == (size_t)(g.hs.S * (g.hs.M - 1) * 6), "mesh: indices == S * (M-1) * 6");
        // No repeated index in any triple (the winding's degenerate-index-free construction).
        bool noRepeat = true;
        for (size_t k = 0; k + 2 < mi.size(); k += 3)
            if (mi[k] == mi[k + 1] || mi[k] == mi[k + 2] || mi[k + 1] == mi[k + 2]) noRepeat = false;
        check(noRepeat, "mesh: no repeated index in any triple");
        // Frame invariants: tangent unit, normal unit, |dot(N,T)| ~ 0, v in [0,1] monotone per rail.
        bool tUnit = true, nUnit = true, ortho = true, vMono = true;
        for (const hairr::HairRenderVertex& v : mv) {
            if (!Unit(v.tx, v.ty, v.tz, 1e-4f)) tUnit = false;
            if (!Unit(v.nx, v.ny, v.nz, 1e-4f)) nUnit = false;
            const float d = v.nx * v.tx + v.ny * v.ty + v.nz * v.tz;
            if (std::fabs(d) > 1e-4f) ortho = false;
        }
        for (int s = 0; s < g.hs.S; ++s)
            for (int i = 1; i < g.hs.M; ++i) {
                const size_t a = (size_t)(s * g.hs.M + i) * 2u;
                if (!(mv[a].v > mv[a - 2].v)) vMono = false;
            }
        check(tUnit, "mesh: every tangent unit length");
        check(nUnit, "mesh: every normal unit length");
        check(ortho, "mesh: normal orthogonal to tangent (the camera-facing frame)");
        check(vMono, "mesh: v strictly increases root -> tip per strand");
        // Determinism: a second build is BYTE-IDENTICAL.
        std::vector<hairr::HairRenderVertex> mv2;
        std::vector<uint32_t> mi2;
        hairr::HairToRenderMesh(g.hs, settled, cam, hairr::kGroomWidthRootQ,
                                hairr::kGroomWidthTipQ, mv2, mi2);
        check(mv2.size() == mv.size() && mi2 == mi &&
              std::memcmp(mv.data(), mv2.data(), mv.size() * sizeof(hairr::HairRenderVertex)) == 0,
              "mesh: two builds BYTE-IDENTICAL");
        // THE PIN: the quantized mesh digest from the pinned sim state + the shared framing.
        const uint64_t meshDigest = hairr::HairRenderDigest(mv, mi);
        std::printf("groom mesh digest: 0x%016llx (%zu verts, %zu tris)\n",
                    (unsigned long long)meshDigest, mv.size(), mi.size() / 3);
        check(meshDigest == kPinnedMeshDigest,
              "mesh: HairRenderDigest == PINNED (MSVC/clang cross-compiler, the float-layer proof)");
        // PROVENANCE: a FRESH scene re-simulated + re-meshed re-derives the exact digest.
        hairr::GroomScene g2 = hairr::MakeGroomScene();
        std::vector<hair::HairVert> settled2 = g2.verts;
        hair::StepHairSteps(g2.hs, settled2, g2.hc, g2.excl, g2.kBend, g2.params, hairr::kGroomSteps);
        std::vector<hairr::HairRenderVertex> mv3;
        std::vector<uint32_t> mi3;
        hairr::HairToRenderMesh(g2.hs, settled2, cam, hairr::kGroomWidthRootQ,
                                hairr::kGroomWidthTipQ, mv3, mi3);
        check(hairr::HairRenderDigest(mv3, mi3) == meshDigest,
              "mesh: PROVENANCE — a fresh sim + mesh re-derives the digest from inputs alone");
    }

    // ================= The quantizer (the asset_compiler FxQuantize truncation precedent) =========
    {
        check(hairr::QuantizeQ16(1.0f) == 65536, "quantize: 1.0 -> kOne");
        check(hairr::QuantizeQ16(-0.5f) == -32768, "quantize: -0.5 -> -kOne/2");
        check(hairr::QuantizeQ16(0.0f) == 0, "quantize: 0 -> 0");
    }

    if (g_fail == 0) std::printf("hair_render_test: ALL PASS\n");
    else             std::printf("hair_render_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
