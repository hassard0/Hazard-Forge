#pragma once
// Slice DX — Virtual-Geometry Visibility-Buffer Slice 2: DEFERRED MATERIAL RESOLVE. Pure CPU
// (header-only, no device, no backend symbols). Namespace hf::render::vg (the same virtual-geometry
// namespace as meshlet.h / cluster_cull.h / visbuffer.h). Same header-only pattern as visbuffer.h.
//
// WHAT THIS IS: the Nanite-style DEFERRED RESOLVE that consumes the DW visibility buffer. DW
// rasterized the IDENTITY (clusterID, triID) of the front-most surface fragment into an R32_Uint
// render target, DECOUPLING geometry rasterization from material shading. DX adds the second half:
// a fullscreen pass texel-fetches that integer vis-buffer per pixel, looks up the covering triangle's
// 3 world positions, computes a FLAT (per-triangle geometric) normal + a Lambert shade, and outputs
// the lit image — shading cost is now per-pixel-VISIBLE, not per-triangle (the Nanite payoff).
//
// THIS HEADER is the AUTHORITATIVE CPU MIRROR of the resolve math (shaders/visresolve.frag.hlsl copies
// it VERBATIM) — shared by tests/visresolve_test.cpp AND the --visresolve-shot (Vulkan) / --visresolve
// (Metal) showcases, so the unit test, the GPU==CPU proof and the shader all agree bit-for-bit.
//
// FLAT SHADING (the deliberate choice): the normal + Lambert are a single deterministic per-triangle
// value (NO per-pixel barycentric interpolation), so the resolve math is GPU==CPU BIT-EXACT via the DH
// FP discipline (std::fma + a host-precomputed light direction) and sidesteps the cross-vendor
// perspective-correct-interpolation fragility. The barycentric-interpolated variant is a later
// refinement (out of scope — it is the cross-vendor-fragile part).
//
// HONEST PROOF SCOPE: a byte-identical "resolve == forward render" proof is NOT cross-vendor-feasible
// (hardware-rasterizer barycentric/fill-rule divergence — even flat shading depends on WHICH triangle
// covers each edge pixel, which differs cross-vendor at silhouettes). DX proves ID-provenance bit-exact
// (inherited from DW) + GPU==CPU resolve-MATH bit-exact (interior pixels) + cross-backend via the IMAGE
// golden + a resolve-vs-forward SMOKE bound. Same shape as DW's interior-only oracle.
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/mtl::) symbols. DX adds NO new RHI: the integer vis-buffer is
// bound via the existing BindTexture (a sampled uint texture, texel-fetched — no sampler) and the three
// cluster-meta/index/vertex SSBOs ride the existing usesLightClusters/BindLightClusters fragment-stage
// storage-buffer path. This header touches none of it.

#include <cmath>
#include <cstdint>
#include <span>

#include "math/math.h"
#include "render/cluster_cull.h"
#include "render/meshlet.h"
#include "render/visbuffer.h"
#include "scene/vertex.h"

namespace hf::render::vg {

// --- The FIXED deterministic resolve material/light (shared CPU<->shader; the showcase uploads the
// SAME bits in its UBO). The light direction is the world-space direction the light TRAVELS (so the
// surface is lit by -lightDir); it is HOST-PRECOMPUTED + already NORMALIZED here so the GPU never
// renormalizes it (the DH discipline — a normalize() on the GPU could round differently than the CPU).
// kLightDir = normalize(0.4, -1.0, -0.35). albedo + ambient are fixed deterministic constants. ---
struct ResolveMaterial {
    math::Vec3 lightDir;   // world-space light TRAVEL direction, PRE-NORMALIZED (surface lit by -lightDir)
    math::Vec3 albedo;     // diffuse albedo (flat material)
    float      ambient;    // constant ambient term added to the Lambert diffuse
};

// The single deterministic resolve material the showcase + the CPU mirror + the shader all use. The
// light direction is normalized HERE (host) to the exact float bits both sides read — never on the GPU.
inline ResolveMaterial DefaultResolveMaterial() {
    // Pre-normalized normalize({0.4, -1.0, -0.35}) computed with the engine's own normalize so the
    // bytes match math::normalize exactly. A light from the upper-front-right, traveling down/back.
    math::Vec3 ld = math::normalize(math::Vec3{0.4f, -1.0f, -0.35f});
    ResolveMaterial m;
    m.lightDir = ld;
    m.albedo   = math::Vec3{0.80f, 0.72f, 0.55f};  // warm off-white
    m.ambient  = 0.12f;
    return m;
}

// The sky / background color the resolve writes where the vis-buffer texel is the background sentinel
// (clusterID >= drawnClusterCount, i.e. no survivor covered the pixel). A fixed dark navy, matching
// DW's CPU-coloring background so the framing reads the same. Returned as linear RGB float bits.
inline math::Vec3 ResolveSkyColor() { return math::Vec3{5.0f / 255.0f, 13.0f / 255.0f, 13.0f / 255.0f}; }

// FLAT geometric normal of a triangle (p0,p1,p2), oriented by the winding via cross(p1-p0, p2-p0).
// The cross-product components are formed with std::fma (a * b - c * d -> fma(a,b,-c*d)) so the value
// is a SINGLE correctly-rounded fused multiply-add per term — matching the shader's `mad` lowering on
// Vulkan (DXC) and Metal (spirv-cross) bit-for-bit (the DH/DV FP discipline). The result is normalized
// with math::normalize (the same sqrt/divide both backends do). ORIENTATION: the engine's sphere is
// CCW-front-facing; cross(e1,e2) of a front face points OUTWARD (away from the sphere center), so the
// flat normal is the outward surface normal — exactly the normal a forward lit pass would use.
inline math::Vec3 FlatNormal(const math::Vec3& p0, const math::Vec3& p1, const math::Vec3& p2) {
    const math::Vec3 e1{p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
    const math::Vec3 e2{p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};
    // cross(e1,e2) with each component as one fused multiply-add: e1.a*e2.b - e1.c*e2.d.
    const math::Vec3 n{
        std::fma(e1.y, e2.z, -(e1.z * e2.y)),
        std::fma(e1.z, e2.x, -(e1.x * e2.z)),
        std::fma(e1.x, e2.y, -(e1.y * e2.x)),
    };
    return math::normalize(n);
}

// Lambert diffuse + ambient for a flat-shaded surface with normal `n`, lit by material `mat`. The
// surface is lit by the direction TOWARD the light = -mat.lightDir; ndotl = max(0, dot(n, -lightDir))
// clamps back-faces to 0. shade = albedo * (ambient + ndotl) per channel, computed with std::fma so
// the GPU's `mad(albedo, ndotl, albedo*ambient)`-equivalent matches bit-for-bit. Returns linear RGB.
inline math::Vec3 LambertShade(const math::Vec3& n, const ResolveMaterial& mat) {
    // dot(n, -lightDir) = -dot(n, lightDir). max(0, .) clamps the unlit back hemisphere.
    float ndl = -math::dot(n, mat.lightDir);
    if (ndl < 0.0f) ndl = 0.0f;
    // out = albedo*ambient + albedo*ndl, as one fma per channel (matches the shader's mad).
    return math::Vec3{
        std::fma(mat.albedo.x, ndl, mat.albedo.x * mat.ambient),
        std::fma(mat.albedo.y, ndl, mat.albedo.y * mat.ambient),
        std::fma(mat.albedo.z, ndl, mat.albedo.z * mat.ambient),
    };
}

// THE RESOLVE (the CPU mirror the shader copies verbatim). Given a covering (clusterID, triID) read
// from the vis-buffer + the cluster's source record (for triOffset + the owning instance's model
// matrix) + the shared reordered index buffer + the shared vertex buffer + the material, compute the
// FLAT-shaded linear RGB of that triangle:
//   1. world tri = model * verts[ indices[ 3*(triOffset+triID) + {0,1,2} ] ].pos
//   2. n = FlatNormal(p0,p1,p2)  (outward geometric normal, std::fma)
//   3. shade = LambertShade(n, material)
// `clusterID` is the SURVIVOR-DRAW index (DW convention); the caller maps it back to its source
// ClusterInstance via the survivor MdiCommand.firstInstance, and passes that ClusterInstance's
// triOffset + the instance model matrix here. Pure, no GPU, deterministic.
inline math::Vec3 ResolveFlatShade(uint32_t triID,
                                   uint32_t clusterTriOffset,
                                   const math::Mat4& model,
                                   std::span<const uint32_t> indices,
                                   std::span<const scene::Vertex> verts,
                                   const ResolveMaterial& mat) {
    const uint32_t base = 3u * (clusterTriOffset + triID);
    const uint32_t i0 = indices[base + 0];
    const uint32_t i1 = indices[base + 1];
    const uint32_t i2 = indices[base + 2];
    auto objPos = [&](uint32_t idx) -> math::Vec3 {
        const scene::Vertex& v = verts[idx];
        return math::Vec3{v.pos[0], v.pos[1], v.pos[2]};
    };
    const math::Vec3 p0 = math::MulPoint(model, objPos(i0));
    const math::Vec3 p1 = math::MulPoint(model, objPos(i1));
    const math::Vec3 p2 = math::MulPoint(model, objPos(i2));
    const math::Vec3 n = FlatNormal(p0, p1, p2);
    return LambertShade(n, mat);
}

// Convenience: the full per-pixel resolve over a raw packed vis-buffer value `v`. If v is the
// background sentinel OR its clusterID is out of the drawn-survivor range, returns the sky color;
// else unpacks (clusterID,triID), maps clusterID->source cluster via survivorCmds, and shades. This is
// EXACTLY what the GPU fragment computes (background-sentinel branch included), so it is the per-texel
// CPU oracle for the GPU==CPU resolve-math proof.
inline math::Vec3 ResolvePixel(uint32_t v, uint32_t drawnClusterCount,
                               std::span<const mdi::MdiCommand> survivorCmds,
                               std::span<const ClusterInstance> clusters,
                               std::span<const math::Mat4> instanceModels,
                               std::span<const uint32_t> indices,
                               std::span<const scene::Vertex> verts,
                               const ResolveMaterial& mat) {
    if (v == kVisBackground) return ResolveSkyColor();
    uint32_t cid, tid;
    UnpackVisId(v, cid, tid);
    if (cid >= drawnClusterCount) return ResolveSkyColor();
    const uint32_t src = survivorCmds[cid].firstInstance;
    const ClusterInstance& ci = clusters[src];
    const math::Mat4& model = instanceModels[ci.instanceIndex];
    return ResolveFlatShade(tid, ci.triOffset, model, indices, verts, mat);
}

// Pack a linear RGB float color into the SAME BGRA8 byte order DW's CPU-coloring uses (B,G,R,A with the
// +0.5 round), so the resolved image golden is produced identically from a read-back GPU value or the
// CPU mirror. saturate to [0,1] first. (The showcase writes the GPU-resolved RT; this helper is the
// CPU side of the GPU==CPU shade memcmp + the forward-smoke comparison.)
inline void EncodeBGRA8(const math::Vec3& linRGB, uint8_t* outBGRA) {
    auto enc = [](float c) -> uint8_t {
        if (c < 0.0f) c = 0.0f;
        if (c > 1.0f) c = 1.0f;
        return (uint8_t)(c * 255.0f + 0.5f);
    };
    outBGRA[0] = enc(linRGB.z);  // B
    outBGRA[1] = enc(linRGB.y);  // G
    outBGRA[2] = enc(linRGB.x);  // R
    outBGRA[3] = 255;
}

}  // namespace hf::render::vg
