#pragma once
// Slice CR — Ground-Truth Ambient Occlusion (GTAO; Jimenez et al. 2016, "Practical Realtime
// Strategies for Accurate Indirect Occlusion") math — pure CPU (header-only, no device, no backend
// symbols). Namespace hf::render::gtao. Mirrors ssr.h / dof.h / pom.h: a tiny shared-math header
// ABOVE the RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only mentions of
// "vk"/"MTL" anywhere here are seam-discipline doc comments). The GTAO fragment shader
// (shaders/gtao.frag.hlsl) copies IntegrateArc + HorizonAngle + Visibility VERBATIM so the CPU unit
// test (tests/gtao_test.cpp) and the GPU pass agree EXACTLY on the horizon-search visibility integral —
// which is what makes the radius=0 render byte-identical to the no-AO scene AND bit-identical
// cross-backend.
//
// THE TECHNIQUE (horizon-based ground-truth AO): for a shaded pixel with view-space position P and
// normal N, the cosine-weighted ambient-occlusion visibility of the hemisphere is estimated by
// sweeping several SLICE directions across the screen. Each slice is a plane through the view
// direction; within that plane we MARCH the surrounding depth field both ways to find the two largest
// HORIZON elevation angles (h1 on one side, h2 on the other) — the angles above which the slice plane
// "sees" the sky past any occluders. Projecting N onto the slice plane gives the slice-normal angle n;
// the closed-form cosine-weighted arc integral between the horizons (IntegrateArc) is the per-slice
// visibility. Averaging over the slices yields the GTAO visibility V ∈ [0,1] (1 = fully open / no
// occlusion). The "ground truth" name is because, unlike a sample-counting SSAO, the per-slice integral
// is the EXACT cosine-weighted visibility for the found horizons.
//
// THE RADIUS=0 / FLAT-FIELD EQUIVALENCE PROOF (what makes this golden-safe — like CN zero-velocity ==
// scene, CO permuted == canonical, CP heightScale=0 == plain): with radius == 0 every marched sample
// COINCIDES with the center (zero march distance) so NO horizon is ever raised above its initial
// hemisphere bound — both horizons reach the full ±90° relative to the slice normal — and IntegrateArc
// for that fully-open arc evaluates to EXACTLY 1 on every slice → Visibility == 1 for every pixel.
// Likewise a perfectly FLAT depth field raises no horizon → Visibility == 1. Multiplying the ambient
// term by 1 changes nothing, so the GTAO-applied image equals the no-AO scene EXACTLY (no constant
// bias, no off-by-one). The showcase renders the scene with radius=0 and asserts SHA-equality to the
// no-AO render; the unit test pins IntegrateArc(full hemisphere)==1, Visibility(flat)==1 exactly, and
// Visibility(radius=0)==1 exactly.
//
// CONVENTIONS (match the SSAO/SSR G-buffer + ssr.h EXACTLY):
//   * RH view space: camera at the origin looking down -Z; VIEW-SPACE LINEAR depth = -vpos.z (positive
//     in front of the camera), the value gbuffer.frag stores in .w and the shader reconstructs from.
//   * Angles are in RADIANS. The slice plane is spanned by the view direction toward the camera and a
//     screen-space slice direction; an angle of 0 lies along the view direction (toward the eye), and
//     +π/2 / −π/2 are the two tangent directions in the slice plane. The two horizons h1,h2 are
//     measured as signed elevations in this plane (one per marching side).
//   * The projected-normal angle n is the elevation of N projected into the slice plane, measured from
//     the view direction (same zero as the horizons). For a fragment facing the camera n ≈ 0.
//
// Pure, deterministic functions: no RNG, no time, fixed slice/step counts, fixed baked rotation.

#include "math/math.h"

#include <algorithm>
#include <cmath>

namespace hf::render::gtao {

inline constexpr float kPi     = 3.14159265358979323846f;
inline constexpr float kHalfPi = 1.57079632679489661923f;

// --- The GTAO inner integral for one slice -------------------------------------------------------
// IntegrateArc(h1, h2, n) -> the cosine-weighted visibility ∈ [0,1] for ONE slice, given the two
// horizon angles h1,h2 (signed elevations in the slice plane, relative to the view direction) and the
// slice-projected normal angle n. The published Jimenez et al. closed form (their Eq. for the inner
// integral of cos-weighted visibility between two horizons, with the normal tilt n) is:
//
//   0.25 * ( (-cos(2*h1 - n) + cos(n) + 2*h1*sin(n))
//          + (-cos(2*h2 - n) + cos(n) + 2*h2*sin(n)) )
//
// Each side contributes the integral of the projected (cosine-weighted, normal-tilted) visibility from
// the normal direction out to its horizon. For an UNOCCLUDED slice the two horizons reach the full
// hemisphere edges in the slice plane: h1 = -kHalfPi + n on one side, h2 = +kHalfPi + n on the other
// (clamped so the arc spans the whole tilted hemisphere). Substituting those into the form:
//   side1 = -cos(2*(-π/2+n) - n) + cos(n) + 2*(-π/2+n)*sin(n)
//         = -cos(-π + n) + cos(n) + (2n - π)*sin(n)
//         =  cos(n)      + cos(n) + (2n - π)*sin(n)
//   side2 = -cos(2*( π/2+n) - n) + cos(n) + 2*( π/2+n)*sin(n)
//         = -cos( π + n) + cos(n) + (π + 2n)*sin(n)
//         =  cos(n)      + cos(n) + (π + 2n)*sin(n)
//   sum   = 4*cos(n) + (4n)*sin(n)  ... the n terms cancel between the two ±π/2 contributions? Let's be
// precise: (2n - π) + (π + 2n) = 4n, so the sin(n) terms sum to 4n*sin(n) and the cos terms sum to
// 4*cos(n); 0.25*(4*cos(n) + 4n*sin(n)) = cos(n) + n*sin(n). For n == 0 this is 1 EXACTLY, which is the
// fully-open / radius=0 identity the proof rests on. (For a tilted normal n the fully-open value is the
// area of the tilted hemisphere slice — the integral correctly normalizes per slice; the showcase /
// test exercise the camera-facing n≈0 case where it is exactly 1.)
//
// SYMMETRY: the form is symmetric in (h1,h2) — swapping the two horizons returns the identical value
// (IntegrateArc(h1,h2,n) == IntegrateArc(h2,h1,n)), so the slice march order does not matter.
inline float IntegrateArc(float h1, float h2, float n) {
    float side1 = -std::cos(2.0f * h1 - n) + std::cos(n) + 2.0f * h1 * std::sin(n);
    float side2 = -std::cos(2.0f * h2 - n) + std::cos(n) + 2.0f * h2 * std::sin(n);
    return 0.25f * (side1 + side2);
}

// --- Per-sample horizon angle (measured from the view direction) ---------------------------------
// HorizonAngle(viewPos, samplePos, sliceTangent, viewDir) -> the angle (radians, in (-π/2, π/2]) of
// the marched sample `samplePos`, measured in the slice plane FROM the view direction `viewDir`
// (fragment -> eye) toward the screen-tangent march direction `sliceTangent`. cos resolves to:
//   angle = atan2( d·sliceTangent , d·viewDir )
// where d = samplePos - viewPos. Conventions / consequences:
//   * A sample flush on the TANGENT plane (d ≈ along sliceTangent, no view-ward component) -> +π/2:
//     the maximally OPEN horizon (no occlusion). This is the unoccluded / radius=0 / flat-field value.
//   * A sample that pokes UP toward the camera (a near occluder: d gains a +viewDir component) ->
//     angle < π/2: the horizon is RAISED (pulled toward the view direction), reducing visibility.
//   * A coincident sample (radius=0, samplePos == viewPos) has no direction -> +π/2 (no horizon
//     raised), so the radius=0 case stays fully open and integrates to 1.
// The +tangent march side returns angles in (0, π/2]; the -tangent side (sliceTangent negated by the
// caller) returns angles in [-π/2, 0) — together they straddle the view direction.
inline float HorizonAngle(const math::Vec3& viewPos, const math::Vec3& samplePos,
                          const math::Vec3& sliceTangent, const math::Vec3& viewDir) {
    math::Vec3 d = samplePos - viewPos;
    float tan_ = math::dot(d, sliceTangent);    // along the screen-tangent march direction
    float perp = math::dot(d, viewDir);         // toward the eye (raises the horizon)
    if (std::fabs(tan_) < 1e-9f && std::fabs(perp) < 1e-9f) {
        // Coincident sample (radius=0): no horizon -> the maximally-open edge on the +tangent side.
        return (tan_ < 0.0f) ? -kHalfPi : kHalfPi;
    }
    return std::atan2(tan_, perp);              // angle from viewDir toward sliceTangent, in (-π, π]
}

// --- Full GTAO visibility estimate ---------------------------------------------------------------
// Visibility(sampleDepth, viewPos, viewNormal, radius, slices, stepsPerSlice, screenW, screenH) -> the
// GTAO AO factor ∈ [0,1] (1 = unoccluded). `sampleDepth` is a callable float(float u, float v)
// returning the VIEW-SPACE LINEAR depth stored in the G-buffer at screen UV (u,v) (the shader passes a
// texture sample; the test passes a procedural field). `viewPos` / `viewNormal` are the center
// fragment's view-space position + normal. The screen<->view projection mirrors ssr.h
// (ViewToScreenUV / ReconstructViewPos) but the test supplies the depth field directly, so this routine
// is parameterized only on the depth-sampler + a projector embedded via the sampler's own coordinates;
// to keep the header self-contained and shader-parallel, the per-slice march walks in VIEW SPACE around
// the fragment and the sampler resolves a view-space-consistent depth (the shader does the UV
// projection inside its sampler lambda; the CPU test's sampler models a flat/known field directly).
//
// ALGORITHM (deterministic; fixed pattern; no RNG):
//   * viewDir = normalize(-viewPos) (fragment -> eye).
//   * For each of `slices` slice directions, rotated evenly around the view axis by a BAKED angle
//     (sliceRot = π * sliceIndex / slices, deterministic — no jitter), build two screen-tangent march
//     directions in the slice plane. March `stepsPerSlice` steps each way out to `radius` (in view-
//     space units), sampling the depth field and updating the MAX horizon (the smallest elevation angle
//     = the highest occluder) on each side.
//   * Project the normal onto the slice plane to get n; IntegrateArc(h1, h2, n) is the slice visibility.
//   * Average the slice visibilities -> AO.
//
// RADIUS == 0: every march step has zero offset, every sample coincides with the center → HorizonAngle
// returns kHalfPi → both horizons stay at the hemisphere edge → IntegrateArc returns the fully-open
// visibility (1 for n==0) → AO == 1 EXACTLY. We also short-circuit radius<=0 to return 1.0f exactly to
// make the identity unconditional + branch-clean.
//
// FLAT FIELD: a depth field that places every marched sample at or below the tangent plane raises no
// horizon above the center → AO == 1.
// `rotationOffset` (radians, default 0) rotates the whole slice fan around the view axis — the shader
// passes a BAKED per-pixel rotation (a fixed function of the pixel coords, NO RNG) here to decorrelate
// neighboring pixels' slice directions and reduce banding; the CPU test uses the default 0. The
// radius=0 / flat-field identity holds for ANY rotationOffset (every horizon stays at its open edge),
// so the per-pixel rotation never breaks the byte-identical proof.
template <typename DepthFn>
inline float Visibility(DepthFn sampleDepth, const math::Vec3& viewPos, const math::Vec3& viewNormal,
                        float radius, int slices, int stepsPerSlice, int screenW, int screenH,
                        float rotationOffset = 0.0f) {
    (void)screenW; (void)screenH;  // reserved for the shader-side UV projection; CPU sampler is view-space.
    // RADIUS == 0 (or non-positive): no horizon search -> fully open -> AO == 1 EXACTLY.
    if (radius <= 0.0f) return 1.0f;
    if (slices < 1) slices = 1;
    if (stepsPerSlice < 1) stepsPerSlice = 1;

    math::Vec3 viewDir = math::normalize(viewPos * -1.0f);  // fragment -> eye
    math::Vec3 N = math::normalize(viewNormal);

    // A view-space basis (right, up) spanning the plane PERPENDICULAR to the view direction, used to
    // place the screen-rotated slice tangent directions. Built deterministically off a fixed up
    // reference (no RNG); for a camera-facing fragment viewDir ≈ +Z so right ≈ +X, up ≈ +Y.
    math::Vec3 up0 = (std::fabs(viewDir.y) < 0.99f) ? math::Vec3{0.0f, 1.0f, 0.0f}
                                                    : math::Vec3{1.0f, 0.0f, 0.0f};
    math::Vec3 right = math::normalize(math::cross(up0, viewDir));
    math::Vec3 up    = math::cross(viewDir, right);

    float visSum = 0.0f;
    for (int s = 0; s < slices; ++s) {
        // Baked slice rotation around the view axis (deterministic, evenly spaced over a half-turn —
        // a slice tangent and its opposite are the two march sides, so a full turn would be redundant),
        // plus the per-pixel rotationOffset (shader: baked; test: 0).
        float phi = rotationOffset + kPi * static_cast<float>(s) / static_cast<float>(slices);
        float cphi = std::cos(phi), sphi = std::sin(phi);
        // sliceTangent: the screen-tangent march direction for this slice (lies in the view-perp plane).
        math::Vec3 sliceTangent{right.x * cphi + up.x * sphi,
                                right.y * cphi + up.y * sphi,
                                right.z * cphi + up.z * sphi};

        // Project N onto the slice plane (spanned by sliceTangent + viewDir) and measure n, the
        // signed angle of the projected normal FROM the view direction toward +sliceTangent — the SAME
        // frame the horizons use (so IntegrateArc's tilt is consistent). For a fragment whose normal
        // faces the camera (N ≈ viewDir) n ≈ 0.
        float nTan  = math::dot(N, sliceTangent);
        float nPerp = math::dot(N, viewDir);
        float n = std::atan2(nTan, nPerp);  // projected-normal angle from viewDir, in (-π, π]

        // March both sides out to `radius`, finding each side's horizon angle FROM the view direction.
        // +tangent side horizon h2 starts at the fully-OPEN edge +π/2 (flush tangent plane) and an
        // occluder pulls it IN toward 0 (the view dir) -> track the MINIMUM angle on that side.
        // -tangent side horizon h1 starts at -π/2 and an occluder pulls it IN toward 0 -> track the
        // MAXIMUM (least-negative) angle. Unoccluded: h1 = -π/2, h2 = +π/2 -> IntegrateArc == 1 at n==0.
        float h2 =  kHalfPi;  // +sliceTangent horizon (min toward 0 as occluders rise)
        float h1 = -kHalfPi;  // -sliceTangent horizon (max toward 0 as occluders rise)
        float dt = radius / static_cast<float>(stepsPerSlice);
        for (int k = 1; k <= stepsPerSlice; ++k) {
            float t = dt * static_cast<float>(k);
            // +sliceTangent sample: the screen offset, placed at the sampled surface's view-space z
            // (-Z forward). The CPU test's sampler returns the occluder's view-space LINEAR depth at
            // that screen point; the shader's sampler projects to UV + reads the G-buffer .w.
            math::Vec3 sPos = viewPos + sliceTangent * t;
            sPos.z = -sampleDepth(sPos.x, sPos.y);
            float aPos = HorizonAngle(viewPos, sPos, sliceTangent, viewDir);   // in (0, π/2]
            if (aPos < h2) h2 = aPos;                                          // occluder pulls it in

            // -sliceTangent sample (the opposite march side); HorizonAngle returns a NEGATIVE angle.
            math::Vec3 sNeg = viewPos - sliceTangent * t;
            sNeg.z = -sampleDepth(sNeg.x, sNeg.y);
            float aNeg = HorizonAngle(viewPos, sNeg, sliceTangent, viewDir);   // in [-π/2, 0)
            if (aNeg > h1) h1 = aNeg;                                          // occluder pulls it in
        }
        // Clamp the horizons into the tilted hemisphere [n - π/2, n + π/2] (an occluder can never open
        // the arc beyond the unoccluded hemisphere) and integrate the cosine-weighted visibility.
        h2 = std::min(h2, n + kHalfPi);
        h1 = std::max(h1, n - kHalfPi);
        visSum += IntegrateArc(h1, h2, n);
    }
    float ao = visSum / static_cast<float>(slices);
    return std::min(std::max(ao, 0.0f), 1.0f);
}

}  // namespace hf::render::gtao
