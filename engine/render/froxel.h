#pragma once
// Slice CS — Froxel Volumetric Fog (sun single-scattering) math — pure CPU (header-only, no device, no
// backend symbols). Namespace hf::render::froxel. Mirrors cluster.h / clouds.h / gtao.h: a small shared-
// math header ABOVE the RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only mentions of
// "vk"/"MTL" anywhere in this slice's above-seam files are seam-discipline doc comments). The three
// froxel shaders (froxel_inject.comp.hlsl / froxel_integrate.comp.hlsl / froxel_apply.frag.hlsl) copy
// SliceZ / ViewZToSlice / Density / Phase / IntegrateStep VERBATIM, so tests/froxel_test.cpp exercises
// the EXACT math the GPU inject/integrate/apply runs — which is what makes the density=0 render
// byte-identical to the no-fog scene AND bit-identical cross-backend.
//
// THE TECHNIQUE (froxel = frustum-voxel volumetric fog; the modern UE5/Frostbite architecture): the view
// frustum is partitioned into a dimX*dimY*dimZ grid of view-space FROXELS (XY follow the framebuffer
// tiles, Z is split into EXPONENTIAL depth slices between zNear..zFar so near froxels are thin). Three
// passes operate on a FLAT SSBO volume of size dimX*dimY*dimZ:
//   1. INJECT (one thread per froxel): reconstruct the froxel's view/world center, evaluate the fog
//      Density there, and write per-froxel (scatterRGB = sunColor * Phase(view,sun,g) * density,
//      extinction = density).
//   2. INTEGRATE (one thread per (x,y) column): march Z 0->dimZ FRONT-TO-BACK accumulating single-
//      scattering via IntegrateStep, writing per-froxel (accumulated inScatterRGB, transmittance).
//   3. APPLY (fullscreen): per pixel reconstruct the view depth from the G-buffer, map it to a Z slice
//      via ViewZToSlice, sample the integrated volume (NEAREST — deterministic), and composite
//      out = scene * transmittance + inScatter.
//
// THE ZERO-DENSITY NO-OP PROOF (what makes this golden-safe — like CL clustered==brute-force,
// CR radius=0==no-AO, CP heightScale=0==plain): single-scattering compositing is out = scene*T + L with
// T = Pi exp(-sigma*dz) and L the accumulated in-scatter. With baseDensity == 0 every froxel's Density is
// 0 -> extinction 0 -> IntegrateStep is a NO-OP (T stays 1, L stays 0) for every step of every column ->
// the apply computes out = scene*1 + 0 = scene EXACTLY. So the showcase renders the SAME apply path at
// baseDensity=0 and asserts SHA-equality to the no-fog scene (no constant bias, no Z-slice off-by-one, no
// integration-normalization drift), then renders baseDensity>0 as the golden. The proof is the SAME
// shader at density 0 vs the no-fog scene (backend-portable), NOT a comparison across different shaders.
//
// CONVENTIONS (must match engine/math + the shaders EXACTLY, same as cluster.h):
//   * Mat4 is column-major; Mat4::Perspective produces Vulkan clip space (depth [0,1], Y-flip baked).
//     View space is RIGHT-HANDED, looking down -Z: a point in front of the camera has view-space z < 0.
//     We slice on the POSITIVE view-distance vz = -viewPos.z.
//   * Froxel flat index: idx = fx + fy*dimX + fz*(dimX*dimY) (same cx-major order as cluster.h).
//   * Exponential Z slice boundary k in [0,dimZ] (REAL-valued k for the apply's continuous mapping):
//     SliceZ(k) = zNear * (zFar/zNear)^(k/dimZ). SliceZ(0)==zNear, SliceZ(dimZ)==zFar; monotone
//     increasing. ViewZToSlice is its EXACT inverse (unit-tested ViewZToSlice(SliceZ(k)) == k).
//   * Density is HEIGHT-based exponential fog: baseDensity * exp(-heightFalloff*(y - heightRef)),
//     clamped >= 0, and exactly 0 when baseDensity == 0 (units: 1/world-distance optical density).
//   * Phase is Henyey-Greenstein (forward-scatter for g>0; isotropic at g=0). cosTheta is the cosine of
//     the angle between the view ray and the light direction.
//   * IntegrateStep is one FRONT-TO-BACK single-scatter step: inScatter += T*stepScatter*stepLen first
//     (the in-scatter at this froxel is attenuated by the transmittance ACCUMULATED IN FRONT of it), then
//     T *= exp(-stepExtinction*stepLen). Order matters: a near scatterer contributes at full T==1.

#include "cluster.h"
#include "math/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

namespace hf::render::froxel {

inline constexpr float kPi = 3.14159265358979323846f;

// The froxel grid: XY*Z dimensions + the view depth range. (e.g. 16x9x64 over [0.5, 80].)
struct FroxelGrid {
    int   dimX = 16, dimY = 9, dimZ = 64;
    float zNear = 0.5f, zFar = 80.0f;

    int froxelCount() const { return dimX * dimY * dimZ; }
    int flatIndex(int fx, int fy, int fz) const { return fx + fy * dimX + fz * (dimX * dimY); }
};

// Exponential z-slice boundary for REAL-valued k in [0,dimZ]: positive view-space distance of slice
// plane k. SliceZ(0)==zNear, SliceZ(dimZ)==zFar; monotone increasing in k. MIRRORED in the shaders.
// (Real-valued k — not just integer slice boundaries — because the apply maps a continuous view depth to
// a continuous slice coordinate; the inject uses integer froxel centers.)
inline float SliceZ(const FroxelGrid& g, float k) {
    return g.zNear * std::pow(g.zFar / g.zNear, k / (float)g.dimZ);
}

// Positive view-distance -> REAL-valued slice coordinate (the EXACT inverse of SliceZ): solve
// vz = zNear*(zFar/zNear)^(k/dimZ) for k => k = dimZ * log(vz/zNear) / log(zFar/zNear). Clamped to
// [0, dimZ]. Unit-tested: ViewZToSlice(SliceZ(k)) == k. MIRRORED in froxel_apply.frag (used to pick the
// integrated volume slice for a pixel's depth). The apply floors this to a NEAREST froxel index.
inline float ViewZToSlice(const FroxelGrid& g, float viewZ) {
    if (viewZ <= g.zNear) return 0.0f;
    if (viewZ >= g.zFar)  return (float)g.dimZ;
    return (float)g.dimZ * std::log(viewZ / g.zNear) / std::log(g.zFar / g.zNear);
}

// The view-space CENTER (positive view-distance) of froxel z-slice fz in [0,dimZ): the geometric mid of
// the slice's [SliceZ(fz), SliceZ(fz+1)] span, evaluated at the half-slice k = fz+0.5. MIRRORED in
// froxel_inject.comp (the inject reconstructs each froxel's view center to evaluate density there).
inline float SliceCenterViewZ(const FroxelGrid& g, int fz) {
    return SliceZ(g, (float)fz + 0.5f);
}

// Height-based exponential fog DENSITY at a world point: baseDensity * exp(-heightFalloff*(y - heightRef)),
// clamped to >= 0. Units: optical density (1/world-distance) — multiplied by a step length it gives an
// optical depth. With a POSITIVE heightFalloff the fog is densest at/below heightRef and thins with
// altitude (ground haze). Returns EXACTLY 0 when baseDensity == 0 (the no-op proof's foundation: no
// density anywhere). Never negative. MIRRORED in froxel_inject.comp.
inline float Density(const math::Vec3& worldPos, float baseDensity, float heightFalloff,
                     float heightRef) {
    if (baseDensity == 0.0f) return 0.0f;   // exact zero -> exact pass-through downstream
    float d = baseDensity * std::exp(-heightFalloff * (worldPos.y - heightRef));
    return d > 0.0f ? d : 0.0f;
}

// Henyey-Greenstein phase function. cosTheta is the cosine of the angle between the VIEW ray and the
// LIGHT direction; g in (-1,1) controls anisotropy: g==0 isotropic (constant 1/(4*pi)); g>0 FORWARD-
// peaked (Phase(1,g) > Phase(-1,g)) so looking toward the sun through the fog glows. Finite for all
// cosTheta in [-1,1] (denom clamped away from 0). Same form as clouds.h::HenyeyGreenstein. MIRRORED in
// froxel_inject.comp.
inline float Phase(float cosTheta, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cosTheta;
    if (denom < 1e-6f) denom = 1e-6f;
    return (1.0f - g2) / (4.0f * kPi * denom * std::sqrt(denom));
}

// One FRONT-TO-BACK single-scattering integration step over a froxel of length stepLen with the froxel's
// in-scattered radiance stepScatter (per unit length) and extinction stepExtinction (per unit length):
//   accumInScatter += transmittance * stepScatter * stepLen;   // attenuated by the T accumulated in front
//   transmittance  *= exp(-stepExtinction * stepLen);          // Beer-Lambert extinction through this froxel
// ORDER MATTERS: the in-scatter is added at the CURRENT (front-accumulated) transmittance BEFORE this
// froxel's own extinction is applied, so a near scatterer at full T==1 contributes fully and far froxels
// are progressively dimmed. With stepExtinction==0 && stepScatter==0 this is a NO-OP (transmittance stays,
// inScatter stays) — the per-step foundation of the zero-density==scene proof. MIRRORED in
// froxel_integrate.comp. A column of constant extinction sigma over total length L yields the analytic
// Beer-Lambert transmittance exp(-sigma*L) (product of per-step exp(-sigma*stepLen)).
inline void IntegrateStep(const math::Vec3& stepScatter, float stepExtinction, float stepLen,
                          math::Vec3& accumInScatter, float& transmittance) {
    accumInScatter = accumInScatter + (stepScatter * (transmittance * stepLen));
    transmittance *= std::exp(-stepExtinction * stepLen);
}

// --- Slice CV — Per-Froxel Clustered-Light Injection (the marquee CS+CL fusion) -----------------------
// The ADDED clustered-point-light in-scatter term for one froxel: iterate the lights this froxel's
// cluster lists and sum each light's in-scattered contribution into the fog. This is ADDED to the sun
// in-scatter computed in the inject pass (scatter = sunScatter + InjectClusteredLights(...)), so it is a
// PURELY ADDITIVE term: with lightCount==0 (or all lights off) it returns (0,0,0) EXACTLY and the inject
// pass reduces to the CS sun-only fog — the byte-identical lights-off proof. It is also multiplied by the
// froxel `density`, so at density==0 the whole scatter is 0 — the density=0==no-fog proof still holds.
//
// Per light i in [0,lightCount): light = lights[clusterLightIndices[lightOffset + i]]; the light's
// VIEW-space position is view * light.posWorld (the SAME world->view transform cluster::AssignLights uses
// for culling); d = |froxelViewPos - lightViewPos|; the WINDOWED HARD-RADIUS attenuation is IDENTICAL to
// CL's lit_clustered_cl.frag:  atten = (1/(d²+0.01)) * clamp(1 - (d/radius)⁴, 0, 1)²  — EXACTLY 0 at and
// beyond `radius`, so a froxel only ever gains light from a light whose sphere reaches it (the same hard
// cutoff that makes CL clustered==brute-force; here it bounds the per-froxel glow to the light's radius).
// The HG `Phase(cos(viewDir, dirToLight), g)` weights forward-scatter toward each light: the in-scatter
// peaks when the view ray points along the light direction (a colored volumetric shaft/glow toward the
// light) for g>0. The accumulated term is  Σ light.color * light.intensity * atten * phase  * density.
//
// `viewDir` is the (unit) VIEW-space direction the camera ray travels for this froxel — for the apply
// this is normalize(froxelViewPos) (eye at origin looking toward the froxel). `dirToLight` is the unit
// view-space direction from the froxel TOWARD the light. Both endpoints are in VIEW space, so the cosine
// is taken consistently in view space (the froxel and lights are all transformed to view space).
//
// MIRRORED VERBATIM in froxel_inject.comp.hlsl behind the `injectLights` flag, so the unit test exercises
// the EXACT math the GPU inject pass adds. Pure CPU, ZERO backend symbols (reuses cluster::PointLight).
inline math::Vec3 InjectClusteredLights(const math::Vec3& froxelViewPos, float density,
                                        const math::Vec3& viewDir, float g,
                                        std::span<const cluster::PointLight> lights,
                                        std::span<const uint32_t> clusterLightIndices,
                                        uint32_t lightOffset, uint32_t lightCount,
                                        const math::Mat4& view) {
    math::Vec3 accum{0.0f, 0.0f, 0.0f};
    if (lightCount == 0) return accum;   // lights-off additive identity -> CS sun-only fog
    for (uint32_t i = 0; i < lightCount; ++i) {
        uint32_t li = clusterLightIndices[(size_t)lightOffset + i];
        const cluster::PointLight& L = lights[(size_t)li];
        math::Vec3 lvpos = math::MulPoint(view, L.posWorld);   // world -> view (same as AssignLights)

        math::Vec3 toLight{lvpos.x - froxelViewPos.x, lvpos.y - froxelViewPos.y,
                           lvpos.z - froxelViewPos.z};
        float dist = std::sqrt(toLight.x * toLight.x + toLight.y * toLight.y + toLight.z * toLight.z);

        // WINDOWED HARD-RADIUS attenuation IDENTICAL to CL (lit_clustered_cl.frag): EXACTLY 0 at d>=radius.
        float r4 = dist / L.radius; r4 = r4 * r4; r4 = r4 * r4;        // (d/radius)^4
        float win = 1.0f - r4; win = win < 0.0f ? 0.0f : (win > 1.0f ? 1.0f : win);
        win = win * win;
        float atten = (1.0f / (dist * dist + 0.01f)) * win;

        // HG phase toward this light: cos(viewDir, dirToLight). dirToLight = toLight/dist (unit).
        float invD = (dist > 1e-9f) ? (1.0f / dist) : 0.0f;
        float cosTheta = viewDir.x * toLight.x * invD + viewDir.y * toLight.y * invD +
                         viewDir.z * toLight.z * invD;
        float phase = Phase(cosTheta, g);

        float w = L.intensity * atten * phase * density;
        accum.x += L.color.x * w;
        accum.y += L.color.y * w;
        accum.z += L.color.z * w;
    }
    return accum;
}

// --- GPU-facing std430 record the showcase uploads/reads. ----------------------------------------------
// One froxel cell: { scatter.rgb, extinction } as float4 #0 + { inScatter.rgb, transmittance } as float4
// #1. 32 bytes, 16-byte aligned. The inject writes float4 #0; the integrate reads #0 + writes #1; the
// apply reads #1. (A single flat SSBO of FroxelCell[dimX*dimY*dimZ] — NOT a 3D texture — written/read via
// the existing storage-buffer + compute-dispatch path, like cluster_assign / gpu-cull.)
struct FroxelCell {
    float scatterExt[4];   // xyz = injected in-scatter radiance, w = extinction (inject writes)
    float resultT[4];      // xyz = integrated in-scatter, w = transmittance (integrate writes)
};

} // namespace hf::render::froxel
