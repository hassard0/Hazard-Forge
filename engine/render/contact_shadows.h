#pragma once
// Slice CT — Screen-Space Contact Shadows math — pure CPU (header-only, no device, no backend symbols).
// Namespace hf::render::contact. Mirrors ssr.h / gtao.h / froxel.h: a tiny shared-math header ABOVE the
// RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only mentions of "vk"/"MTL" anywhere
// in this slice's above-seam files are seam-discipline doc comments). The contact-shadow fragment
// shader (shaders/contact_shadows.frag.hlsl) copies RayMarchShadow + ProjectToUV VERBATIM so the CPU
// unit test (tests/contact_shadows_test.cpp) and the GPU pass agree EXACTLY on the screen-space
// depth-march — which is what makes the maxDist=0 render byte-identical to the no-contact-shadow render
// AND bit-identical cross-backend.
//
// THE TECHNIQUE (UE5 "Contact Shadows" / screen-space contact shadows): the cascaded shadow map (CSM)
// is too coarse to capture the fine occlusion in tight contacts — the dark line where a sphere rests on
// the ground, the crease where a thin object nearly touches a surface. For each shaded pixel we take its
// view-space position P (reconstructed from the G-buffer depth) and march a SHORT ray TOWARD the sun
// (lightDirView = the direction to the sun) in `steps` increments out to `maxDist` view-space units. At
// each step we project the marched view point to screen, sample the scene's stored depth there, and ask:
// is the real surface at that screen position CLOSER to the camera than the marched ray (i.e. the ray
// has passed BEHIND some geometry)? If so — and if the depth difference is within a `thickness` window
// (so a distant background wall does NOT cast a false contact shadow) — the pixel is occluded from the
// sun and we darken the DIRECT sun contribution. A clear march returns 1.
//
// THE maxDist=0 / ZERO-STEPS NO-OP PROOF (what makes this golden-safe — like CR radius=0==no-AO,
// CS density=0==no-fog, CP heightScale=0==plain): the contact-shadow factor multiplies ONLY the direct
// sun term (ambient/IBL/point lights untouched, exactly like the CSM shadow term). With maxDist == 0 the
// march step length is 0 → every marched point COINCIDES with the origin pixel → no scene depth is ever
// in FRONT of the ray (the ray never leaves the surface) → no occluder is found → the factor is 1 for
// every pixel → multiplying the sun term by 1 changes nothing → the contact-shadowed render equals the
// no-contact-shadow (plain lit) render EXACTLY. We also short-circuit maxDist<=0 || steps<=0 to return
// 1.0f to make the identity unconditional + branch-clean. The showcase renders the SAME contact-shadow
// shader at maxDist=0 and asserts SHA-equality to the plain lit render; the unit test pins
// RayMarchShadow(..., maxDist=0)==1, (..., steps=0)==1, a clear field → 1, an occluder within the
// thickness window → <1 (monotone toward a closer occluder), an occluder BEYOND the thickness window → 1
// (no false shadow / no haloing — the safety case), and a flat lit surface → 1 (the bias prevents
// self-occlusion acne).
//
// THE BIAS (self-occlusion acne): the surface the ray starts on shares the origin pixel's depth, and the
// first marched step is only slightly behind the camera-facing surface; without a guard the surface
// would shadow ITSELF (acne). We only count an occluder when the scene surface is closer than the
// marched ray by MORE THAN `bias` view-space units — a flat surface (scene depth ≈ ray depth along the
// march) never crosses the bias, so it stays fully lit.
//
// THE THICKNESS WINDOW (no false shadow / no haloing): an occluder is only counted when the scene
// surface is in front of the ray by between `bias` and `thickness` view-space units. A surface FARTHER
// than `thickness` in front of the ray (e.g. a distant background wall sampled along the march) is
// treated as a different object the ray passes safely in front of — NOT a contact occluder — so it casts
// no shadow. This is the standard screen-space "max thickness" guard that prevents long dark halos.
//
// CONVENTIONS (match the SSAO/SSR/GTAO G-buffer + ssr.h/gtao.h EXACTLY):
//   * RH view space: camera at the origin looking down -Z; VIEW-SPACE LINEAR depth = -vpos.z (positive
//     in front of the camera), the value gbuffer.frag stores in .w and the shader reconstructs from.
//   * `lightDirView` is the VIEW-SPACE direction TO the sun (normalized internally). The FrameData sun
//     `lightDir` is the sun's TRAVEL direction, so the shader passes -lightDir transformed into view
//     space. The march walks viewPos + lightDirView * t for t in (0, maxDist].
//   * ProjectToUV mirrors ssr.h::ViewToScreenUV / gtao.frag::ProjectToUV (no matrix inverse): the yFlip
//     sign maps screen UV.y <-> view-space Y, -1 on Vulkan, +1 on Metal. The CPU test passes a depth
//     field directly keyed off screen UV; the shader projects + reads the G-buffer .w.
//   * The returned factor is in [0,1]: 1 = fully lit (no contact occluder), 0 = fully shadowed. We use a
//     HARD step (factor 0 on the first occluder hit, distance-attenuated by march progress for a soft
//     edge — documented at the call site) so the contact line reads crisply.
//
// Pure, deterministic functions: no RNG, no time, fixed step count, fixed params.

#include "math/math.h"

#include <algorithm>
#include <cmath>

namespace hf::render::contact {

// Forward-project a VIEW-SPACE position to a screen UV in [0,1] (IDENTICAL to ssr.h::ViewToScreenUV /
// gtao.frag::ProjectToUV). ndc = vp.xy / (scale * -vp.z), uv = ndc*0.5+0.5, with the yFlip sign on Y.
// `yFlip` is -1 on Vulkan, +1 on Metal.
inline math::Vec2 ProjectToUV(const math::Vec3& vp, float tanHalfFovY, float aspect, float yFlip) {
    float invZ = 1.0f / std::max(-vp.z, 1e-4f);
    float ndcx = vp.x / (aspect * tanHalfFovY) * invZ;
    float ndcy = yFlip * vp.y / tanHalfFovY * invZ;
    return math::Vec2{ndcx * 0.5f + 0.5f, ndcy * 0.5f + 0.5f};
}

// --- The screen-space contact-shadow ray-march ---------------------------------------------------
// RayMarchShadow(sampleDepth, viewPos, lightDirView, maxDist, steps, thickness, bias, tanHalfFovY,
//                aspect, yFlip, ditherOffset) -> the contact-shadow factor ∈ [0,1] (1 = fully lit,
// < 1 = in contact shadow). `sampleDepth` is a callable float(float u, float v) returning the
// VIEW-SPACE LINEAR depth (the G-buffer .w) at screen UV (u,v); the shader passes a texture sample, the
// test passes a procedural field. March from `viewPos` toward the sun (`lightDirView`, the view-space
// direction TO the sun) in `steps` even increments out to `maxDist` view-space units. At each step
// project the marched view point to screen, sample the scene depth there, and compare the scene surface
// view-distance (= the sampled linear depth) against the marched ray's view-distance (= -marchPos.z):
//   diff = rayDist - sceneDist   (> 0 when the scene surface is CLOSER to the camera than the ray, i.e.
//                                 the ray has passed BEHIND that surface)
//   occluded if  bias < diff <= thickness  (closer than the ray by more than the bias, but within the
//                                            thickness window so a far background does not falsely shadow)
// On the FIRST occluded step we return a factor attenuated by how FAR along the march the hit was — a
// near hit gives a darker (smaller) factor, a hit at the very end of the march barely darkens — for a
// soft contact edge that fades with march distance. A clear march (no step occluded) → 1.
//
// maxDist <= 0 || steps <= 0 → 1.0f EXACTLY (no march → no occluder → fully lit), the no-op identity the
// byte-identical proof rests on. `ditherOffset` ∈ [0,1) shifts the start of the first step by a fraction
// of the step length (the shader passes a BAKED per-pixel dither — no RNG — to hide step banding); it
// scales the marched distances, so at maxDist=0 every step is still 0 and the identity holds for ANY
// ditherOffset. The default 0 is what the unit test uses.
template <typename DepthFn>
inline float RayMarchShadow(DepthFn sampleDepth, const math::Vec3& viewPos,
                            const math::Vec3& lightDirView, float maxDist, int steps,
                            float thickness, float bias, float tanHalfFovY, float aspect, float yFlip,
                            float ditherOffset = 0.0f) {
    // maxDist == 0 (or non-positive) / no steps -> no march -> no occluder -> fully lit (1) EXACTLY.
    if (maxDist <= 0.0f || steps <= 0) return 1.0f;

    math::Vec3 L = math::normalize(lightDirView);   // toward the sun, in view space
    float stepLen = maxDist / static_cast<float>(steps);

    for (int k = 1; k <= steps; ++k) {
        // March distance for step k, nudged by the per-pixel dither (a fraction of one step). The dither
        // scales with stepLen, so maxDist=0 -> stepLen=0 -> t=0 for every k (identity preserved).
        float t = stepLen * (static_cast<float>(k - 1) + ditherOffset + 1.0f);
        if (t > maxDist) t = maxDist;
        math::Vec3 marchPos = viewPos + L * t;

        // Behind the camera (or at it): cannot project meaningfully -> skip this step.
        float rayDist = -marchPos.z;                 // view-space distance of the marched ray point
        if (rayDist <= 1e-4f) continue;

        math::Vec2 uv = ProjectToUV(marchPos, tanHalfFovY, aspect, yFlip);
        // Off-screen samples carry no occluder information -> skip (no false shadow at the screen edge).
        if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) continue;

        float sceneDist = sampleDepth(uv.x, uv.y); // view-space linear depth of the real surface here
        // Background / no geometry (cleared depth == 0): nothing to occlude against -> skip.
        if (sceneDist <= 1e-4f) continue;

        // diff > 0 when the real surface is CLOSER to the camera than the marched ray (ray went behind it).
        float diff = rayDist - sceneDist;
        if (diff > bias && diff <= thickness) {
            // Occluded. Soft falloff by march progress: a near hit (small t) darkens fully toward 0, a
            // hit at the end of the march barely darkens. frac in (0,1]; factor = frac at the hit.
            float frac = t / maxDist;
            float factor = std::min(std::max(frac, 0.0f), 1.0f);
            return factor;
        }
    }
    return 1.0f;   // clear march -> fully lit
}

}  // namespace hf::render::contact
