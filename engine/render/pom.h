#pragma once
// Slice CP — Parallax Occlusion Mapping (POM) math — pure CPU (header-only, no device, no backend
// symbols). Namespace hf::render::pom. Mirrors dof.h / motion_blur.h / oit.h: a tiny shared-math
// header ABOVE the RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only mentions of
// "vk"/"MTL" anywhere here are seam-discipline doc comments). The POM fragment shader
// (shaders/pom.frag.hlsl) copies ParallaxUV + SelfShadow VERBATIM so the CPU unit test
// (tests/pom_test.cpp) and the GPU lit pass agree EXACTLY on the marched UV + the self-occlusion —
// which is what makes the heightScale=0 render byte-identical cross-backend AND byte-identical to the
// plain normal-mapped lit render.
//
// THE TECHNIQUE (steep parallax + binary-search refine, a.k.a. Parallax Occlusion Mapping): a flat
// tangent-plane surface carries a HEIGHT FIELD h(uv) ∈ [0,1] (a texture or a procedural function). To
// give the surface apparent depth we walk the VIEW RAY (expressed in tangent space) through that height
// field: starting at the surface entry point we step DOWNWARD in depth, and at the first step where the
// ray has descended PAST the height field we have bracketed the intersection; a few binary-search
// refinements between the last two layers pin the crossing. The UV at that intersection is the texel the
// eye actually sees, so bricks recess and stones rise — real per-pixel depth from a flat quad.
//
// THE ZERO-HEIGHT EQUIVALENCE PROOF (what makes this golden-safe — like CN zero-velocity == scene and
// CO permuted == canonical): when heightScale == 0 the height field has ZERO amplitude, the ray never
// descends below the flat base plane, the very first layer already "intersects", and the returned UV is
// the base UV EXACTLY (no march offset, no off-by-one). Likewise a STRAIGHT-ON view (the tangent-space
// view direction has no xy component) walks straight down with no lateral UV travel, so the returned UV
// is again the base UV exactly. So at heightScale == 0 the POM surface samples albedo/normal at the
// SAME UV as the plain normal-mapped surface and shades IDENTICALLY — the showcase renders both and
// asserts SHA equality, failing loudly on any drift.
//
// HEIGHT CONVENTION (must match the shader EXACTLY):
//   * h(uv) ∈ [0,1]. h == 1 is the SURFACE TOP (the flat base plane, closest to the eye); h == 0 is the
//     DEEPEST point of the recess. We march a "ray DEPTH" variable that starts at 0 (at the top) and
//     increases toward 1 as the ray descends into the material. The ray has hit the surface when its
//     depth has reached/passed the LOCAL surface depth (1 - h(uv)) — i.e. when rayDepth >= (1 - h). We
//     work in this (1 - h) "depth from top" space so the comparison is a simple "ray sank below the
//     surface" test. (A flat region h == 1 has surface depth 0, so the ray intersects it immediately at
//     depth 0 — the entry point — which is the heightScale==0 / flat-field behaviour.)
//   * The lateral UV travel for a full descent of depth 1 is  (viewDirTangent.xy / viewDirTangent.z) *
//     heightScale  — the ray, marched in tangent space, slides across the surface by this much per unit
//     depth, scaled by the height amplitude. Larger heightScale OR a more grazing view (larger xy/z)
//     => larger UV offset (the parallax). This is the standard "offset limiting"-free tangent-space
//     march; viewDirTangent points FROM the surface TOWARD the eye (so we negate its xy to step the
//     SAMPLE point in the direction the surface appears to shift).
//
// Pure, deterministic functions: no RNG, no time, fixed march + refine counts.

#include "math/math.h"

#include <algorithm>
#include <cmath>

namespace hf::render::pom {

// --- Steep-parallax march + binary-search refine -------------------------------------------------
// ParallaxUV(baseUV, viewDirTangent, heightScale, numSteps, sampleHeight) -> the UV at the ray/height-
// field intersection. `viewDirTangent` is the (un-normalized OK) view direction in TANGENT space,
// pointing from the surface toward the EYE (its .z is the cosine of the view angle to the surface
// normal; .xy its lateral lean). `heightScale` is the depth amplitude of the height field in UV units.
// `numSteps` is the linear-march layer count (more layers = finer first bracket). `sampleHeight` is a
// callable float(math::Vec2 uv) returning h(uv) ∈ [0,1] (the test passes a procedural height; the
// shader samples a height texture).
//
// DEGENERATE EXITS (both return baseUV EXACTLY — the zero-height proof):
//   * heightScale == 0      : zero amplitude -> the surface is the flat base plane -> the eye sees the
//                             entry texel -> baseUV.
//   * viewDirTangent.xy == 0 : straight-on view -> the ray descends with NO lateral travel -> baseUV.
// (Both are tested for exact float equality so the early-out is bit-clean; we treat |z| ~ 0 as straight-
// down too, guarding the divide.)
//
// THE MARCH: maxOffset = (viewDirTangent.xy / viewDirTangent.z) * heightScale is the TOTAL lateral UV
// shift across a full depth-1 descent. We step `numSteps` equal layers of depth dt = 1/numSteps from
// depth 0 downward; at layer k the ray depth is k*dt and the sample UV is baseUV - maxOffset*(k*dt)
// (negated: the texel slides opposite the view lean). We compare the ray depth against the LOCAL
// surface depth (1 - sampleHeight(uv)). The first layer where rayDepth >= surfaceDepth is PAST the
// surface; the previous layer was ABOVE it — that pair brackets the crossing.
//
// THE REFINE: a FIXED number of binary subdivisions between the bracketing layers (deterministic, no
// data-dependent loop count) converging on the depth where rayDepth == surfaceDepth. The returned UV is
// at that refined depth — it lies ON the height field (rayDepth(uv) ≈ 1 - h(uv)) with no overshoot.
template <typename HeightFn>
inline math::Vec2 ParallaxUV(const math::Vec2& baseUV, const math::Vec3& viewDirTangent,
                             float heightScale, int numSteps, HeightFn sampleHeight) {
    // --- Degenerate exits: return baseUV EXACTLY (the zero-height + straight-on equivalence). ---
    if (heightScale == 0.0f) return baseUV;                 // zero amplitude -> flat base plane
    if (viewDirTangent.x == 0.0f && viewDirTangent.y == 0.0f) return baseUV;  // straight-on view
    // Guard the .z divide (a view grazing exactly the tangent plane): treat a vanishing cosine as a
    // floored small positive so the lateral shift is large-but-finite rather than a div-by-zero.
    float vz = viewDirTangent.z;
    if (std::fabs(vz) < 1e-4f) vz = (vz < 0.0f ? -1e-4f : 1e-4f);

    int steps = numSteps > 0 ? numSteps : 1;
    // Total lateral UV shift over a full depth-1 descent. NEGATED below so the sampled texel slides
    // opposite the view lean (the standard parallax-offset direction toward the eye's apparent shift).
    math::Vec2 maxOffset{(viewDirTangent.x / vz) * heightScale,
                         (viewDirTangent.y / vz) * heightScale};

    float dt = 1.0f / static_cast<float>(steps);            // per-layer depth increment

    // --- Linear steep-parallax march: descend until the ray sinks below the surface. ---
    float prevRayDepth = 0.0f;
    float prevSurfDepth = 1.0f - std::min(std::max(sampleHeight(baseUV), 0.0f), 1.0f);  // at the entry
    // If the entry point is ALREADY at/below the surface (flat top h==1 -> surfDepth 0 >= rayDepth 0),
    // the intersection is the entry point: baseUV. (This is the heightScale-flat-field fast path; with
    // heightScale!=0 but a locally-flat top texel it still correctly returns ~baseUV.)
    if (prevRayDepth >= prevSurfDepth) return baseUV;

    math::Vec2 curUV = baseUV;
    float curRayDepth = 0.0f;
    float curSurfDepth = prevSurfDepth;
    bool hit = false;
    for (int k = 1; k <= steps; ++k) {
        float rayDepth = static_cast<float>(k) * dt;        // descend
        math::Vec2 uv{baseUV.x - maxOffset.x * rayDepth, baseUV.y - maxOffset.y * rayDepth};
        float surfDepth = 1.0f - std::min(std::max(sampleHeight(uv), 0.0f), 1.0f);
        if (rayDepth >= surfDepth) {                        // ray has sunk below the surface -> bracket
            prevRayDepth = static_cast<float>(k - 1) * dt;
            prevSurfDepth = 1.0f - std::min(std::max(
                sampleHeight(math::Vec2{baseUV.x - maxOffset.x * prevRayDepth,
                                        baseUV.y - maxOffset.y * prevRayDepth}), 0.0f), 1.0f);
            curRayDepth = rayDepth;
            curSurfDepth = surfDepth;
            curUV = uv;
            hit = true;
            break;
        }
    }
    if (!hit) {
        // Never crossed within the march (the ray exits the bottom without dipping below the field):
        // clamp to the deepest sampled layer (no overshoot past depth 1).
        return curUV;
    }

    // --- Binary-search refine between [prev layer (above), cur layer (below)]. A FIXED count of
    // bisections, deterministic. We track the signed gap g(depth) = rayDepth - surfDepth: g(prev) < 0
    // (above), g(cur) >= 0 (below); we bisect toward g == 0. ---
    float loDepth = prevRayDepth;   // above the surface (gap < 0)
    float hiDepth = curRayDepth;    // below the surface (gap >= 0)
    const int kRefine = 8;          // fixed -> deterministic, no data-dependent loop length
    math::Vec2 refinedUV = curUV;
    for (int r = 0; r < kRefine; ++r) {
        float midDepth = 0.5f * (loDepth + hiDepth);
        math::Vec2 uv{baseUV.x - maxOffset.x * midDepth, baseUV.y - maxOffset.y * midDepth};
        float surfDepth = 1.0f - std::min(std::max(sampleHeight(uv), 0.0f), 1.0f);
        float gap = midDepth - surfDepth;
        if (gap >= 0.0f) { hiDepth = midDepth; refinedUV = uv; }  // still below -> tighten from below
        else             { loDepth = midDepth; }                  // above -> raise the floor
    }
    return refinedUV;
}

// --- Soft self-shadowing toward the light (the "occlusion" in POM) -------------------------------
// SelfShadow(uv, lightDirTangent, heightScale, numSteps, sampleHeight) -> a shadow factor ∈ [0,1]
// (1 = fully lit, < 1 = partially self-occluded by the height field between this texel and the light).
// `uv` is the (already parallax-offset) surface texel; `lightDirTangent` is the direction TOWARD the
// light in tangent space (.z > 0 means the light is above the tangent plane). We march FROM the texel's
// own surface point UP toward the light along the tangent-space light direction; if the height field
// rises ABOVE the marching ray at any step the light is blocked there, and we accumulate the deepest
// penetration as a soft occlusion (a ridge between the texel and the light darkens the crevice).
//
// DEGENERATE EXITS (return 1.0 EXACTLY — fully lit, no occlusion):
//   * heightScale == 0        : flat field, nothing can occlude.
//   * lightDirTangent.xy == 0 : light straight overhead -> no lateral march -> nothing blocks it.
//   * lightDirTangent.z <= 0  : light at/below the tangent plane -> the surface faces away; the N·L
//     term in the shader already kills it, so we return 1 here (no extra self-occlusion to apply).
// Monotone: a deeper height amplitude (larger heightScale) can only ADD occlusion (shadow factor
// non-increasing in heightScale) for a texel a ridge can reach.
template <typename HeightFn>
inline float SelfShadow(const math::Vec2& uv, const math::Vec3& lightDirTangent, float heightScale,
                        int numSteps, HeightFn sampleHeight) {
    if (heightScale == 0.0f) return 1.0f;                                    // flat -> no occlusion
    if (lightDirTangent.x == 0.0f && lightDirTangent.y == 0.0f) return 1.0f; // straight overhead
    if (lightDirTangent.z <= 0.0f) return 1.0f;                             // light below the plane

    int steps = numSteps > 0 ? numSteps : 1;
    float h0 = std::min(std::max(sampleHeight(uv), 0.0f), 1.0f);  // the texel's own surface height
    // Lateral UV travel per unit upward depth toward the light (mirror of the view march geometry).
    float lz = lightDirTangent.z;
    math::Vec2 lstep{(lightDirTangent.x / lz) * heightScale,
                     (lightDirTangent.y / lz) * heightScale};

    float dt = 1.0f / static_cast<float>(steps);
    // March from the surface point UP toward the light: at fraction t the ray's height ABOVE the base
    // is h0 + t*(1 - h0) (it rises from the texel's height toward the top as it travels to the light).
    // The texel is occluded where the height field along the path rises ABOVE the ray's height. We
    // accumulate the MAX positive penetration (deepest blocker) as a soft shadow.
    float maxBlock = 0.0f;
    for (int k = 1; k <= steps; ++k) {
        float t = static_cast<float>(k) * dt;
        math::Vec2 suv{uv.x + lstep.x * t, uv.y + lstep.y * t};
        float rayHeight = h0 + t * (1.0f - h0);                  // the light ray's height at this step
        float sh = std::min(std::max(sampleHeight(suv), 0.0f), 1.0f);
        float block = sh - rayHeight;                            // > 0 => field rises above the ray
        if (block > maxBlock) maxBlock = block;
    }
    // Soft factor: no blocker -> 1; a full-height blocker -> 0. Clamp to [0,1].
    float shadow = 1.0f - std::min(std::max(maxBlock, 0.0f), 1.0f);
    return shadow;
}

}  // namespace hf::render::pom
