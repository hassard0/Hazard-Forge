#pragma once
// Slice DA — Box-Projected Cubemap Reflections (local reflection probe) math — pure CPU (header-only,
// no device, no backend symbols). Namespace hf::render::reflprobe. Mirrors ssr.h / gtao.h / sss.h: a
// tiny shared-math header ABOVE the RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the
// only mentions of "vk"/"MTL" anywhere in this slice's above-seam files are seam-discipline doc
// comments + the [[vk::binding]] HLSL decorations). The reflection-probe fragment shader
// (shaders/reflprobe.frag.hlsl) copies RayBoxExitT + BoxProject VERBATIM, so
// tests/reflection_probe_test.cpp exercises the EXACT box-projection the GPU pass runs — which is what
// makes the parallaxStrength=0 (or infinite box) render byte-identical to the standard infinite-cubemap
// reflection render AND bit-identical cross-backend.
//
// THE TECHNIQUE (box-projected / parallax-corrected cubemap reflections — the hallmark of a LOCAL
// reflection probe): an environment cubemap baked from a probe origin represents the room as if it were
// infinitely far away, so a plain cubemap reflection samples along the mirror direction R and the
// reflected walls/floor appear at infinity — they do NOT line up with the actual room geometry as the
// camera or surface moves. Box projection fixes this by treating the cubemap as the surface of a finite
// BOX (the room volume the probe represents): from the shaded world point P we trace the reflection ray
// P + t*R until it EXITS that box AABB at a point `hit`, then sample the cubemap along the direction
// FROM THE PROBE CENTER to `hit` (not along R). Because the box is the room, `hit` lands on the actual
// wall the ray would strike, so the reflected wall lines up with the room geometry — a parallax-correct
// local reflection. (McGuire/Lagarde "box-projected cubemap environment mapping".)
//
// THE parallaxStrength=0 / INFINITE-BOX NO-OP PROOF (what makes this golden-safe — like CR radius=0==no-AO,
// CZ sssStrength=0==no-SSS, CP heightScale=0==plain): the corrected direction is
//     normalize( lerp(R, hit - center, parallaxStrength) ).
// With parallaxStrength == 0 the lerp returns R exactly, so the corrected direction is normalize(R) —
// the SAME direction a plain cubemap reflection samples → BYTE-IDENTICAL to the standard
// infinite-cubemap reflection (no constant bias, no direction drift, no normalization error). As the box
// grows to infinity the exit point `hit` runs off infinitely far along R (hit ≈ P + t*R with t → ∞), so
// (hit - center) is dominated by t*R and normalize(hit - center) → normalize(R): the correction VANISHES
// in the infinite-box limit. So the showcase INTERNALLY renders the reflection at parallaxStrength=0 (or
// an effectively-infinite box) and asserts it is BYTE-IDENTICAL (SHA) to the engine's standard
// cubemap-reflection render of the same scene — then renders the real finite-box version as the golden.
// The unit test additionally proves the ray-box intersection (RayBoxExitT) + the correction analytically.
//
// CONVENTIONS:
//   * WORLD space throughout (the reflection probe box + the shaded world position P + the world
//     reflection vector R = reflect(-viewDir, N) all live in world space, like lit_probe.frag's probe
//     sampling). No view-space / depth conventions here — this is a pure geometric ray-box correction.
//   * `reflDir` (R) is the mirror reflection direction (NOT required unit on input; BoxProject
//     normalizes). `worldPos` (P) is the shaded surface world position. The box AABB is [boxMin,boxMax]
//     with the probe sample origin at `center` (typically the box center, but kept separate so the
//     probe origin can differ from the geometric center — matches lit_probe's probePos).
//   * `parallaxStrength` ∈ [0,1] blends OFF (0 → plain infinite cubemap reflection) ↔ FULL (1 → fully
//     box-corrected). Values outside [0,1] are NOT clamped (the lerp is total); the showcase passes 0
//     and 1. RayBoxExitT returns the smallest POSITIVE t at which the ray leaves the box.
//
// Pure, deterministic functions: no RNG, no time.

#include "math/math.h"

#include <algorithm>
#include <cmath>

namespace hf::render::reflprobe {

// The reflection probe: the probe SAMPLE origin (`center`, the cubemap was baked from here) + the
// world-space box AABB [boxMin,boxMax] (the room volume the cubemap represents). An "infinite" box
// (very large boxMin/boxMax) makes box projection a no-op (the exit point runs to infinity → corrected
// dir == R), which is the basis of the no-op proof.
struct ProbeBox {
    math::Vec3 center;   // probe sample origin (where the cubemap was baked)
    math::Vec3 boxMin;   // world-space AABB min corner of the room volume
    math::Vec3 boxMax;   // world-space AABB max corner of the room volume
};

// --- Ray-box EXIT t (slab method) ----------------------------------------------------------------
// RayBoxExitT(origin, dir, boxMin, boxMax) -> the smallest POSITIVE parameter t at which the ray
// origin + t*dir LEAVES the box AABB [boxMin,boxMax] (the FAR slab intersection, tFar). `dir` need not
// be unit (t is in units of |dir|). For an origin INSIDE the box (the reflection case: the shaded point
// is on a wall of / inside the room) the entry tNear is negative and the exit tFar is the first
// POSITIVE crossing — exactly the point where the reflection ray strikes the far wall.
//
// SLAB METHOD (documented): for each axis i the ray crosses the two parallel planes x_i == boxMin_i and
// x_i == boxMax_i at parameters t1 = (boxMin_i - origin_i)/dir_i and t2 = (boxMax_i - origin_i)/dir_i;
// the slab is entered at min(t1,t2) and exited at max(t1,t2). The box interval is the INTERSECTION of
// the three slab intervals: tNear = max over axes of the per-slab entry, tFar = min over axes of the
// per-slab exit. The ray leaves the box at tFar.
//
// DEGENERATE GUARD (ray parallel to a slab): if dir_i ≈ 0 the ray never crosses that axis's planes; it
// only stays inside the box if origin_i is already within [boxMin_i, boxMax_i]. We then skip that axis
// (it imposes no finite bound — leave tFar untouched), matching math::RayAabb's inf-slope convention. A
// FULLY-degenerate / origin-outside case (no positive exit) returns a large positive sentinel so
// BoxProject's (hit - center) is dominated by the far-along-R offset (≈ keep R) rather than producing a
// NaN — the function stays total. (BoxProject ALSO short-circuits parallaxStrength==0 before ever
// calling this, so the no-op proof never depends on the degenerate path.)
inline float RayBoxExitT(const math::Vec3& origin, const math::Vec3& dir,
                         const math::Vec3& boxMin, const math::Vec3& boxMax) {
    const float kBig = 1e30f;
    const float o[3]  = {origin.x, origin.y, origin.z};
    const float d[3]  = {dir.x, dir.y, dir.z};
    const float mn[3] = {boxMin.x, boxMin.y, boxMin.z};
    const float mx[3] = {boxMax.x, boxMax.y, boxMax.z};
    float tFar = kBig;
    for (int i = 0; i < 3; ++i) {
        if (d[i] > -1e-9f && d[i] < 1e-9f) {
            // Ray parallel to this slab: it imposes no finite exit bound on this axis (the box is left
            // through one of the OTHER two slabs). Skip it. (If origin is outside this slab the ray
            // never re-enters, but the other slabs still bound tFar; the result is then a no-correction
            // fallback, handled by BoxProject's normalize.)
            continue;
        }
        float inv = 1.0f / d[i];
        float t1 = (mn[i] - o[i]) * inv;   // crossing of the boxMin plane
        float t2 = (mx[i] - o[i]) * inv;   // crossing of the boxMax plane
        float tExit = std::max(t1, t2);    // this slab is EXITED at the larger crossing
        if (tExit < tFar) tFar = tExit;    // the box is exited at the SMALLEST per-slab exit
    }
    // For a sane box + an origin in/near it, tFar is the positive exit. If everything was degenerate
    // (dir ≈ 0 on every axis) tFar stays the big sentinel. Guard against a non-positive tFar (origin
    // outside the box on the far side) by returning the big sentinel so BoxProject keeps ≈ R.
    if (tFar <= 0.0f) return kBig;
    return tFar;
}

// --- Box-projected (parallax-corrected) reflection direction -------------------------------------
// BoxProject(reflDir, worldPos, box, parallaxStrength) -> the corrected, UNIT cubemap sample direction.
// Trace the reflection ray worldPos + t*reflDir to its positive exit from the box, take the world exit
// point `hit = worldPos + tExit*reflDir`, and return
//     normalize( lerp(reflDir, hit - box.center, parallaxStrength) ).
// (lerp(a,b,t) = a + (b-a)*t, applied component-wise.)
//
// THE NO-OP PROOF (the byte-identical identity): with parallaxStrength == 0 the lerp is reflDir exactly,
// so the result is normalize(reflDir) — the SAME direction a plain (infinite) cubemap reflection samples.
// We short-circuit parallaxStrength == 0 to return normalize(reflDir) DIRECTLY (so the identity is exact
// + branch-clean regardless of the box/ray field — no dependence on the ray-box result, no chance of a
// tExit-driven rounding difference). The shader does the SAME early-out, so the parallaxStrength=0
// reflection is byte-identical to the standard reflection on every backend.
//
// INFINITE-BOX LIMIT: with a huge box, tExit is huge, so hit = worldPos + tExit*reflDir is dominated by
// tExit*reflDir; (hit - center) ≈ tExit*reflDir, and normalize(tExit*reflDir) == normalize(reflDir) (a
// positive scalar does not change a normalized direction). So even at parallaxStrength == 1 a huge box
// converges to normalize(reflDir): the correction vanishes as the room grows to infinity (quantified by
// the unit test).
//
// DEGENERATE GUARDS: reflDir == 0 → normalize returns 0 (the shader never reflects a zero vector). A
// degenerate / no-exit ray (RayBoxExitT returns the big sentinel) makes hit run far along reflDir, so
// (hit - center) ≈ huge*reflDir → normalize ≈ reflDir: a clean "keep R" fallback (no NaN).
inline math::Vec3 BoxProject(const math::Vec3& reflDir, const math::Vec3& worldPos,
                             const ProbeBox& box, float parallaxStrength) {
    // NO-OP early-out: zero parallax -> plain infinite-cubemap reflection direction EXACTLY (the
    // byte-identical proof). No dependence on the box / ray-box result.
    if (parallaxStrength == 0.0f) return math::normalize(reflDir);

    float tExit = RayBoxExitT(worldPos, reflDir, box.boxMin, box.boxMax);
    math::Vec3 hit = worldPos + reflDir * tExit;        // world exit point on the box wall
    math::Vec3 toHit = hit - box.center;                // probe-center -> hit (the parallax direction)
    // Blend between the original reflection direction and the parallax direction by parallaxStrength.
    math::Vec3 corrected = math::Vec3{
        math::lerp(reflDir.x, toHit.x, parallaxStrength),
        math::lerp(reflDir.y, toHit.y, parallaxStrength),
        math::lerp(reflDir.z, toHit.z, parallaxStrength),
    };
    return math::normalize(corrected);
}

}  // namespace hf::render::reflprobe
