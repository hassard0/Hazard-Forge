#pragma once
// Slice HRR1 — THE HAIR STRAND RENDERER: ribbon geometry for the HR1 deterministic strand sim
// (engine/sim/hair.h) + the groom showcase scene builder. Parity++ audit #7: HR1 ships the bit-exact
// Q16.16 PBD strand SIM, but its golden renders as debug DOTS — this header turns the settled strands
// into a LIT GROOM: tapered CAMERA-FACING ribbons whose vertices carry the strand TANGENT for
// Kajiya-Kay anisotropic shading (shaders/hair_kajiya.frag.hlsl, a lit.frag clone). Namespace
// hf::render::hairr, header-only, NO device / backend / RHI symbols.
//
// THE TWO LAYERS (the CL6/FPX6/FL6 float-capstone convention, cleanly split):
//   1. THE GROOM SCENE (MakeGroomScene, PURE INTEGER): a denser HR1 sim — 64 strands rooted on an 8x8
//      grid over a scalp-SPHERE CAP, each strand grown OUTWARD along its scalp normal (FxNormalize —
//      the sim's own integer toolbox; NO float on the sim path), verts 0+1 pinned (the HR1
//      direction-clamped root), k_bend ramped 0 -> kOne across the strand index ("some stiff, some
//      limp"). Every op is the hair.h Q16.16 arithmetic (component fxmul, exact integer multiples —
//      the InitStrands inner loop with a per-strand root + grow direction), so the settled state is
//      bit-identical CPU/Vulkan/Metal AND MSVC/clang — the render mesh's provenance. Shared
//      byte-for-byte by the Vulkan shot, the Metal shot and hair_render_test.
//   2. THE RIBBON MESH (HairToRenderMesh, HOST FLOAT — the ONE documented float crossing, render-only,
//      the ClothToRenderMesh twin): per strand vert, a two-rail strip sample —
//        tangent_i = normalize(P_{i+1} - P_i)   (forward difference; the LAST vert reuses the previous
//                    segment's tangent — the VR1 BuildRibbons convention, TANGENT-aligned here where
//                    VR1's trails are velocity-aligned)
//        side_i    = normalize(cross(tangent_i, viewDir_i))   (the standard hair-billboard camera-
//                    facing frame; viewDir_i = normalize(camPos - P_i))
//        normal_i  = normalize(cross(side_i, tangent_i))      (== the component of viewDir
//                    perpendicular to the tangent -> points TOWARD the camera; the shading normal
//                    slot of scene::Vertex)
//        halfW_i   = 0.5 * lerp(widthRoot, widthTip, t_i), t_i = i/(M-1)  (the root->tip TAPER)
//        left/right = P_i -/+ side_i * halfW_i; u = 0/1 (rail), v = t_i (the along-strand parameter,
//                    the fragment shader's root->tip albedo ramp)
//      Indices: the SP1/VR1 pinned SweepStrip winding per span — (a,b,c)(c,b,d) with a=left_i,
//      b=right_i, c=left_{i+1}, d=right_{i+1}; NO repeated index in any triple, by construction. The
//      winding's right-hand normal is cross(side, tangent) == the camera-facing normal, matching the
//      CL6 "front faces the camera" convention.
//      TANGENT DELIVERY (documented): scene::Vertex already carries a TANGENT slot (pos/color/uv/
//      normal/tangent) and lit.vert forwards it to the fragment stage at location 5 — so the strand
//      tangent rides the vertex TANGENT slot (its natural home; NO normal-slot smuggling needed) and
//      the camera-facing ribbon normal keeps the NORMAL slot (shadow/ambient terms stay sane).
//      DEGENERATE GUARDS: a zero-length segment reuses the previous tangent (first segment: straight
//      down, the gravity prior); tangent PARALLEL to viewDir (|cross|^2 < eps) falls back to
//      cross(tangent, worldUp), then to +X if the tangent is also vertical — deterministic, never NaN.
//   FLOAT DISCIPLINE (the cloth.h CL6 precedent): std::fma in every cross/dot/length accumulation +
//   IEEE sqrt/divide only, so MSVC and clang produce the EXACT same mesh floats (the test pins the
//   quantized digest under both compilers).
//
// HONEST CAVEATS (v1, documented): (1) camera-facing ribbons SHIMMER at grazing tangents — when a
// strand points nearly at the camera the side vector swings rapidly between frames/pixels (the
// standard hair-billboard artifact; a stable per-strand frame or tube geometry is the future
// refinement); (2) NO strand<->strand self-shadowing — the ribbons cast/receive only the scene's
// directional shadow map; deep-opacity maps are a future slice; (3) the taper is per-VERT linear in
// the strand parameter, not arc-length.
//
// DIGEST: HairRenderDigest — the mesh QUANTIZED to Q16.16 int32 (the asset_compiler FxQuantize
// truncation precedent) + the index list, FNV-1a-64 (net::DigestBytes) — the cross-compiler pin
// currency for the float layer.

#include <cmath>
#include <cstdint>
#include <vector>

#include "math/math.h"     // float bridge only: math::Vec3 (render-only; NOT on the bit-exact sim path)
#include "net/session.h"   // net::DigestBytes — the FNV-1a-64 digest currency
#include "sim/hair.h"      // READ-ONLY reuse: HairStrands/HairVert/InitStrands-style math/HairParams +
                           // the cloth.h Q16.16 toolbox re-exports (fx/FxVec3/fxmul/FxNormalize) and
                           // cloth::ClothVertToWorld (the ONE documented Q16.16 -> float crossing)

namespace hf::render::hairr {

using sim::hair::fx;
using sim::hair::FxVec3;
using sim::hair::HairVert;
using sim::hair::HairStrands;
using sim::hair::kOne;

// ===== 1. THE GROOM SCENE (pure integer — the bit-exact provenance) ================================

// The full groom-sim bundle: scene + constraints + exclusion + per-strand stiffness + step params.
// MakeGroomScene() fills every field deterministically; the caller runs sim::hair::StepHairSteps.
struct GroomScene {
    HairStrands                          hs;
    std::vector<HairVert>                verts;    // the initial state (roots on the scalp cap)
    sim::hair::HairConstraints           hc;
    sim::hair::ClothAdjacency            excl;
    std::vector<fx>                      kBend;    // per-strand bend stiffness (the 0 -> kOne ramp)
    sim::hair::HairParams                params;
};

// The showcase step count (10 s at dt = 1/60 — settled, the HR1 convention).
inline constexpr int kGroomSteps = 600;

// The SHARED showcase framing (Vulkan shot == Metal shot == hair_render_test, so the quantized mesh
// digest is ONE pinnable value everywhere): the groom stays in SIM world coordinates (its bounds are
// known by construction — no AABB re-centering, unlike CL6's varying drape), the camera sits at a
// fixed 3/4 eye looking at the hanging strands, and the ribbon widths taper root -> tip.
inline constexpr float kGroomEyeX = 4.5f, kGroomEyeY = 6.5f, kGroomEyeZ = 11.0f;
inline constexpr float kGroomCenterX = 0.0f, kGroomCenterY = 4.8f, kGroomCenterZ = 0.0f;
inline constexpr fx kGroomWidthRootQ = (fx)(kOne / 8);    // 0.125 wu full width at the root
inline constexpr fx kGroomWidthTipQ  = (fx)(kOne / 64);   // 0.015625 wu at the tip (the taper)

// MakeGroomScene: 64 strands rooted on an 8x8 grid over a scalp-sphere cap (center (0,6,0), radius
// 1.5), grown OUTWARD along the scalp normal, verts 0+1 pinned (direction-clamped), k_bend ramped
// s*kOne/63 across the strands. PURE INTEGER (fxmul/FxNormalize — the hair.h toolbox); the root/grow
// math is the InitStrands inner loop with a per-strand root + direction. Deterministic on every
// platform and compiler — the provenance root of the HRR1 render mesh.
inline GroomScene MakeGroomScene() {
    GroomScene g;
    g.hs.S = 64;
    g.hs.M = 12;
    g.hs.restLen = (fx)(kOne * 3 / 8);                       // 0.375 -> 4.125 units per strand

    const FxVec3 scalpCenter{0, (fx)(6 * (int)kOne), 0};
    const fx scalpR   = (fx)(kOne * 3 / 2);                  // 1.5
    const fx halfStep = (fx)(kOne * 5 / 32);                 // 0.15625 (exact Q16.16 fraction)
    const fx upBias   = (fx)(kOne * 5 / 4);                  // the cap's vertical bias (1.25)
    const int kPin = 2;                                      // the HR1 direction-clamped root

    g.verts.resize((size_t)(g.hs.S * g.hs.M));
    for (int gz = 0; gz < 8; ++gz)
        for (int gx = 0; gx < 8; ++gx) {
            const int s = gz * 8 + gx;
            // Grid offset (2*g - 7) * halfStep: exact odd multiples, symmetric about the cap apex.
            const fx ox = (fx)((int64_t)(2 * gx - 7) * (int64_t)halfStep);
            const fx oz = (fx)((int64_t)(2 * gz - 7) * (int64_t)halfStep);
            // The scalp normal at this grid cell (integer FxNormalize — the sim's own toolbox).
            const FxVec3 dir = sim::hair::FxNormalize(FxVec3{ox, upBias, oz});
            // Root ON the sphere cap: center + dir * R (component fxmul, the InitStrands math).
            const FxVec3 root{scalpCenter.x + sim::hair::fxmul(dir.x, scalpR),
                              scalpCenter.y + sim::hair::fxmul(dir.y, scalpR),
                              scalpCenter.z + sim::hair::fxmul(dir.z, scalpR)};
            for (int i = 0; i < g.hs.M; ++i) {
                HairVert v;
                const fx along = (fx)((int64_t)i * (int64_t)g.hs.restLen);   // exact integer multiple
                v.pos = FxVec3{root.x + sim::hair::fxmul(dir.x, along),
                               root.y + sim::hair::fxmul(dir.y, along),
                               root.z + sim::hair::fxmul(dir.z, along)};
                v.prev = v.pos;
                v.vel = FxVec3{0, 0, 0};
                if (i < kPin) { v.invMass = 0;    v.flags = sim::hair::kFlagPinned; }
                else          { v.invMass = kOne; v.flags = 0; }
                g.verts[(size_t)sim::hair::VertIndex(g.hs, s, i)] = v;
            }
        }

    g.hc   = sim::hair::BuildHairConstraints(g.hs);
    g.excl = sim::hair::BuildHairExclusion(g.verts.size(), g.hc);

    g.kBend.resize((size_t)g.hs.S, 0);
    for (int s = 0; s < g.hs.S; ++s)
        g.kBend[(size_t)s] = (fx)((int64_t)s * (int64_t)kOne / 63);   // limp -> stiff ramp

    const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));   // the HR1 rounding
    g.params.gravity = FxVec3{0, kGravY, 0};
    g.params.dt      = kOne / 60;
    g.params.groundY = 0;                                    // far below the hanging tips (never hit)
    g.params.iters   = 12;
    g.params.radius  = kOne / 16;                            // 2r = 0.125 < the 0.3125 root spacing
    g.params.damp    = kOne - kOne / 32;                     // 0.96875 (exact Q16.16)
    return g;
}

// ===== 2. THE RIBBON MESH (host float — the ONE documented crossing, render-only) ==================

// A render-ready ribbon vertex: POD float11, trivially copied into scene::Vertex by the showcase
// (pos -> pos, normal -> normal, TANGENT -> tangent, (u,v) -> uv; color chosen by the showcase).
struct HairRenderVertex {
    float px, py, pz;   // world position = strand vert pos / kOne (cloth::ClothVertToWorld)
    float nx, ny, nz;   // the CAMERA-FACING ribbon normal = normalize(cross(side, tangent))
    float tx, ty, tz;   // the strand TANGENT (unit; drives the Kajiya-Kay terms in the fragment)
    float u, v;         // u = rail (0 left / 1 right), v = along-strand root->tip parameter
};

namespace detail {
// The CL6 float discipline: std::fma in every accumulation so MSVC == clang bit-for-bit.
struct V3 { float x, y, z; };
inline V3 Cross(const V3& a, const V3& b) {
    return V3{std::fma(a.y, b.z, -a.z * b.y),
              std::fma(a.z, b.x, -a.x * b.z),
              std::fma(a.x, b.y, -a.y * b.x)};
}
inline float Len2(const V3& a) { return std::fma(a.x, a.x, std::fma(a.y, a.y, a.z * a.z)); }
// Normalize; len2 <= eps -> returns ok=false and the input unchanged (caller picks the fallback).
inline bool NormalizeSafe(V3& a, float eps) {
    const float len2 = Len2(a);
    if (!(len2 > eps)) return false;
    const float inv = 1.0f / std::sqrt(len2);
    a.x *= inv; a.y *= inv; a.z *= inv;
    return true;
}
}  // namespace detail

// HairToRenderMesh(hs, verts, camPos, widthRootQ, widthTipQ, outVerts, outIdx): the settled strand
// state -> tapered camera-facing ribbon strips (see the header banner for the exact per-vert frame,
// taper, winding and degenerate guards). widthRootQ/widthTipQ are Q16.16 world-unit FULL widths
// (converted by the same /kOne crossing as the positions). outVerts.size() == S * 2M;
// outIdx.size() == S * (M-1) * 6. A degenerate layout (M < 2, size mismatch, both widths <= 0)
// yields the empty mesh. Pure + deterministic host float (std::fma discipline); RENDER-ONLY — the
// bit-exact sim state is consumed const.
inline void HairToRenderMesh(const HairStrands& hs, const std::vector<HairVert>& verts,
                             const math::Vec3& camPos, fx widthRootQ, fx widthTipQ,
                             std::vector<HairRenderVertex>& outVerts,
                             std::vector<uint32_t>& outIdx) {
    outVerts.clear();
    outIdx.clear();
    if (hs.S < 1 || hs.M < 2 || (size_t)(hs.S * hs.M) != verts.size()) return;
    if (widthRootQ <= 0 && widthTipQ <= 0) return;

    const float invOne = 1.0f / (float)kOne;
    const float wRoot = (float)widthRootQ * invOne;          // the same /kOne crossing as positions
    const float wTip  = (float)widthTipQ * invOne;
    const float invSpan = 1.0f / (float)(hs.M - 1);
    const float kEps = 1e-12f;

    outVerts.reserve((size_t)(hs.S * hs.M * 2));
    outIdx.reserve((size_t)(hs.S * (hs.M - 1) * 6));

    for (int s = 0; s < hs.S; ++s) {
        const uint32_t baseVert = (uint32_t)outVerts.size();
        // The strand's float centerline (cloth::ClothVertToWorld — the ONE documented crossing).
        std::vector<detail::V3> P((size_t)hs.M);
        for (int i = 0; i < hs.M; ++i) {
            const math::Vec3 w = sim::cloth::ClothVertToWorld(
                verts[(size_t)sim::hair::VertIndex(hs, s, i)].pos);
            P[(size_t)i] = detail::V3{w.x, w.y, w.z};
        }
        detail::V3 prevT{0.0f, -1.0f, 0.0f};   // the gravity prior (a zero FIRST segment's fallback)
        for (int i = 0; i < hs.M; ++i) {
            // Tangent: forward difference; the last vert reuses the previous segment (VR1 convention).
            detail::V3 T = prevT;
            if (i + 1 < hs.M) {
                detail::V3 d{P[(size_t)i + 1].x - P[(size_t)i].x,
                             P[(size_t)i + 1].y - P[(size_t)i].y,
                             P[(size_t)i + 1].z - P[(size_t)i].z};
                if (detail::NormalizeSafe(d, kEps)) T = d;   // zero-length segment -> reuse prevT
            }
            prevT = T;
            // The camera-facing frame: side = normalize(cross(T, viewDir)); degenerate (T parallel to
            // viewDir) -> cross(T, worldUp); T also vertical -> +X. Deterministic, never NaN.
            detail::V3 viewDir{camPos.x - P[(size_t)i].x, camPos.y - P[(size_t)i].y,
                               camPos.z - P[(size_t)i].z};
            (void)detail::NormalizeSafe(viewDir, kEps);      // camPos == P: viewDir stays as-is (guarded below)
            detail::V3 side = detail::Cross(T, viewDir);
            if (!detail::NormalizeSafe(side, kEps)) {
                side = detail::Cross(T, detail::V3{0.0f, 1.0f, 0.0f});
                if (!detail::NormalizeSafe(side, kEps)) side = detail::V3{1.0f, 0.0f, 0.0f};
            }
            detail::V3 N = detail::Cross(side, T);           // toward the camera (unit by construction)
            (void)detail::NormalizeSafe(N, kEps);            // guard the near-degenerate frame
            // The root->tip TAPER: halfW = 0.5 * lerp(wRoot, wTip, t).
            const float t = (float)i * invSpan;
            const float halfW = 0.5f * std::fma(wTip - wRoot, t, wRoot);
            // The two rails (left u=0, right u=1) — both carry the SAME tangent/normal/v.
            HairRenderVertex L{}, R{};
            L.px = std::fma(-halfW, side.x, P[(size_t)i].x);
            L.py = std::fma(-halfW, side.y, P[(size_t)i].y);
            L.pz = std::fma(-halfW, side.z, P[(size_t)i].z);
            R.px = std::fma(halfW, side.x, P[(size_t)i].x);
            R.py = std::fma(halfW, side.y, P[(size_t)i].y);
            R.pz = std::fma(halfW, side.z, P[(size_t)i].z);
            L.nx = R.nx = N.x; L.ny = R.ny = N.y; L.nz = R.nz = N.z;
            L.tx = R.tx = T.x; L.ty = R.ty = T.y; L.tz = R.tz = T.z;
            L.u = 0.0f; R.u = 1.0f;
            L.v = R.v = t;
            outVerts.push_back(L);
            outVerts.push_back(R);
        }
        for (int i = 0; i + 1 < hs.M; ++i) {                 // the pinned SweepStrip winding per span
            const uint32_t a = baseVert + 2u * (uint32_t)i;  // left_i
            const uint32_t b = a + 1u;                       // right_i
            const uint32_t c = baseVert + 2u * (uint32_t)(i + 1);   // left_{i+1}
            const uint32_t d = c + 1u;                       // right_{i+1}
            outIdx.push_back(a); outIdx.push_back(b); outIdx.push_back(c);
            outIdx.push_back(c); outIdx.push_back(b); outIdx.push_back(d);
        }
    }
}

// ===== 3. THE QUANTIZED DIGEST (the float-layer pin currency) ======================================

// QuantizeQ16: float -> Q16.16 int32 by TRUNCATION — the asset_compiler.h FxQuantize precedent
// (bit-stable across compilers because the mesh floats themselves are bit-stable, see the fma note).
inline int32_t QuantizeQ16(float f) { return (int32_t)(f * 65536.0f); }

// HairRenderDigest: FNV-1a-64 over the QUANTIZED vertex set (11 int32 per vertex, field order) + the
// raw index list. Layout/padding-independent (explicit int32 packing). The provenance pin: the digest
// re-derives from the pinned sim state alone.
inline uint64_t HairRenderDigest(const std::vector<HairRenderVertex>& verts,
                                 const std::vector<uint32_t>& idx) {
    std::vector<int32_t> buf;
    buf.reserve(verts.size() * 11u + idx.size() + 2u);
    buf.push_back((int32_t)verts.size());
    for (const HairRenderVertex& v : verts) {
        buf.push_back(QuantizeQ16(v.px)); buf.push_back(QuantizeQ16(v.py)); buf.push_back(QuantizeQ16(v.pz));
        buf.push_back(QuantizeQ16(v.nx)); buf.push_back(QuantizeQ16(v.ny)); buf.push_back(QuantizeQ16(v.nz));
        buf.push_back(QuantizeQ16(v.tx)); buf.push_back(QuantizeQ16(v.ty)); buf.push_back(QuantizeQ16(v.tz));
        buf.push_back(QuantizeQ16(v.u));  buf.push_back(QuantizeQ16(v.v));
    }
    buf.push_back((int32_t)idx.size());
    for (const uint32_t i : idx) buf.push_back((int32_t)i);
    return hf::net::DigestBytes(buf.data(), buf.size() * sizeof(int32_t));
}

}  // namespace hf::render::hairr
