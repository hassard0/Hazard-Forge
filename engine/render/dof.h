#pragma once
// Depth-of-field math — pure CPU (header-only, no device, no backend symbols). Shared by the
// --dof-shot showcase reasoning AND tests/dof_test.cpp so the unit test exercises the SAME
// circle-of-confusion (CoC) + scatter-as-gather weight the in-shader gather (dof.frag.hlsl) uses.
//
// Mirrors ssr.h/ssgi.h/water.h: a tiny shared-math header above the RHI seam (zero vk*/MTL*/mtl::/
// Backend::Metal symbols). The DoF shader copies these two functions verbatim so the CPU test and the
// GPU pass agree on the depth-to-blur mapping.
//
// CONVENTIONS (match the SSAO/SSR G-buffer): `depth` is VIEW-SPACE LINEAR depth = -vpos.z (positive in
// front of the camera, the value gbuffer.frag stores in .w and dof.frag reconstructs). All CoC sizes
// are in PIXELS (the gather kernel walks the screen by pixels, so the thin-lens CoC is converted to a
// pixel radius up front and clamped to a fixed maxCoCpx so the kernel cost is bounded).

#include <algorithm>
#include <cmath>

namespace hf::render::dof {

// --- Thin-lens circle of confusion -------------------------------------------------------------
// CircleOfConfusion(depth, focalDist, aperture, focalLength, maxCoCpx) -> CoC RADIUS in pixels.
//
// The thin-lens CoC diameter for a point at object distance `depth`, with the lens focused at
// `focalDist`, an entrance-pupil `aperture` (diameter; larger = shallower DoF), and `focalLength` f is
// the textbook formula:
//
//     coc = aperture * |focalLength * (depth - focalDist)| / (depth * (focalDist - focalLength))
//
// Key properties (the unit test pins them):
//   * coc == 0 EXACTLY at depth == focalDist (the (depth - focalDist) numerator vanishes) -> the focal
//     plane is perfectly sharp.
//   * coc grows with |depth - focalDist|, SEPARATELY on the near (depth < focalDist) and far
//     (depth > focalDist) sides (the formula handles both branches via the absolute value).
//   * the result is a CoC SIZE scaled into screen pixels by a fixed factor folded into `aperture`
//     (the showcase passes an aperture already expressed so the saturated far field reaches ~maxCoCpx),
//     then CLAMPED to [0, maxCoCpx] so the gather kernel radius is bounded.
//
// ROBUSTNESS: the denominator carries a `depth` factor (-> +inf as depth -> 0) and a constant
// (focalDist - focalLength) term. We guard `depth` with a small epsilon (so depth -> 0 and behind-camera
// depths do NOT divide by zero -> a finite, clamped CoC) and guard (focalDist - focalLength) likewise
// (a degenerate lens with focalDist == focalLength yields the clamped max rather than a NaN). depth ->
// focalLength is NOT singular here (focalLength only appears in the constant denominator term), but the
// guards keep every input the spec calls out (depth -> 0, depth -> focalLength, depth < 0) finite.
inline float CircleOfConfusion(float depth, float focalDist, float aperture,
                               float focalLength, float maxCoCpx) {
    // Guard the object distance: clamp to a small positive epsilon so depth -> 0 (or behind the camera)
    // is a huge-but-finite defocus that the final clamp caps at maxCoCpx (never a divide-by-zero NaN).
    float d = std::max(depth, 1e-3f);
    // Guard the (focalDist - focalLength) denominator term against a degenerate lens.
    float denomFocus = focalDist - focalLength;
    if (std::fabs(denomFocus) < 1e-4f) denomFocus = (denomFocus < 0.0f ? -1e-4f : 1e-4f);

    float numer = aperture * std::fabs(focalLength * (depth - focalDist));
    float denom = d * denomFocus;
    float coc = numer / denom;
    if (!std::isfinite(coc) || coc < 0.0f) coc = 0.0f;  // belt-and-suspenders for any residual edge
    return std::min(coc, maxCoCpx);
}

// --- Scatter-as-gather blur weight -------------------------------------------------------------
// BlurWeight(tapCoCpx, tapDistPx) -> the (un-normalized) contribution of a NEIGHBOR tap to the CENTER
// pixel under a scatter-as-gather DoF gather.
//
// Gather model (documented): a blurred neighbor "scatters" its color over a disk the size of ITS OWN
// circle of confusion. Gathering at the center, a neighbor at screen distance `tapDistPx` contributes
// to the center IFF its CoC disk REACHES the center, i.e. tapDistPx <= tapCoCpx. Within that disk the
// contribution is normalized by the disk AREA (~pi * r^2) so a large-CoC tap spreads its energy thinly
// (a physically-plausible bokeh falloff) while a tight-CoC tap deposits concentrated energy. The caller
// (dof.frag) accumulates sum += weight*color and wsum += weight over the disk and outputs sum/wsum.
//
// Consequences the test pins:
//   * a tap whose CoC covers the center (tapDistPx <= tapCoCpx) contributes (> 0);
//   * a tap outside its CoC (tapDistPx > tapCoCpx) contributes ~0 (sharp regions don't bleed onto
//     neighbors, and a near blurred object correctly spreads only as far as its CoC);
//   * a FOCAL neighbor (tapCoCpx ~ 0) contributes only to its OWN pixel (tapDistPx == 0) and nothing to
//     any neighbor -> the sharp focal subject stays crisp and never blurs the background.
inline float BlurWeight(float tapCoCpx, float tapDistPx) {
    // The neighbor only reaches the center if its CoC disk covers it (a tiny slack so the center tap of
    // a focal pixel, dist == CoC == 0, still counts itself).
    if (tapDistPx > tapCoCpx + 1e-4f) return 0.0f;
    // Normalize by the disk area so a wide-CoC tap spreads thinly. Use the CoC radius (in px) with a
    // 1px floor so a focal/near-focal tap (CoC ~ 0) gathering ITSELF (dist 0) still yields a finite,
    // bounded weight rather than blowing up.
    float r = std::max(tapCoCpx, 1.0f);
    return 1.0f / (3.14159265358979323846f * r * r);
}

}  // namespace hf::render::dof
