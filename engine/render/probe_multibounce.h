#pragma once
// Slice DR — DDGI MULTI-BOUNCE (2nd light bounce) — pure CPU (header-only, no device, no backend
// symbols). Namespace hf::render::probemb. The EIGHTH slice of the GLOBAL ILLUMINATION (DDGI) flagship
// arc (after DN's single-bounce GI composite). A tiny shared-logic header ABOVE the RHI seam (ZERO
// vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only "vk"/"MTL" mentions in this slice's above-seam
// files are seam-discipline doc comments). Mirrors probe_gi.h / probe_sh.h style.
//
// THE TECHNIQUE (the DDGI 2nd bounce): single-bounce DDGI (the DH–DN arc) captures only DIRECT light
// into the probes, so indirect light bounces exactly ONCE. Multi-bounce feeds the 1st-bounce GI (SH0,
// the direct capture's irradiance) BACK into a SECOND probe capture (the NEW probe_bake_gi.frag adds the
// DN indirect term to the captured surface radiance), then SH-encodes that bounce-1 radiance into SH1 —
// so the final composite over SH1 gets a brighter, more filled-in 2nd bounce (the step from UE4-style
// DDGI toward Lumen's infinite-bounce). Done as a FIXED 2-iteration pass in one frame (deterministic,
// golden-able), NOT a temporal feedback loop.
//
// THE bounceCount=1 NO-OP (make-or-break): the composite selects SH(bounceCount-1): SH0 at bounceCount=1
// (bounce-1 capture SKIPPED entirely -> it IS the DN single-bounce render, BYTE-IDENTICAL), SH1 at
// bounceCount>=2. SelectBounceSH encodes EXACTLY that selection so the showcase + the test agree the
// disabled path uses SH0. THE 2nd-bounce MONOTONICITY: the bounce-1 capture ADDS a non-negative indirect
// (SH0 irradiance * albedo * giStrength, all >= 0) to a non-negative direct radiance, so SH1's
// reconstructed irradiance is >= SH0's for a positive-albedo scene — the bounce only ADDS light, never
// darkens (BounceAddsLight / the CPU mirror below). THE probeCount=0 / bounceCount clamp: probeCount=0 ->
// the indirect feed is {0,0,0} (the DN zero-SH fallback); bounceCount clamps to >= 1.
//
// Pure, deterministic functions: no RNG, no time, no device.

#include "math/math.h"
#include "render/probe_gi.h"   // SHARE the probe grid (ProbeGrid / probePos / probeCount / flatIndex)
#include "render/probe_sh.h"   // SHARE the ProbeSH record + InterpolateIrradiance (the DN/DJ math)

#include <algorithm>
#include <cstdint>

namespace hf::render::probemb {

using probegi::ProbeGrid;
using probesh::ProbeSH;

// --- Clamp the bounce count to the legal range. ------------------------------------------------------
// The showcase runs EXACTLY 2 captures max (the 2nd bounce is the demonstrable leap; N-bounce is a
// trivial loop extension later — YAGNI). bounceCount<1 clamps to 1 (the single-bounce DN render); >=2
// runs the bounce-1 capture. Pure / deterministic.
inline constexpr int kMaxBounces = 2;
inline int ClampBounceCount(int bounceCount) {
    if (bounceCount < 1) return 1;
    if (bounceCount > kMaxBounces) return kMaxBounces;
    return bounceCount;
}

// --- Select which SH buffer the final composite binds. -----------------------------------------------
// SelectBounceSH(bounceCount, sh0, sh1): the composite samples SH(bounceCount-1). bounceCount<=1 -> sh0
// (the DI direct capture -> the DN single-bounce path; the bounce-1 capture is SKIPPED so this IS the
// byte-identical no-op the make-or-break proof rests on); bounceCount>=2 -> sh1 (the bounce-1 capture ->
// the 2nd bounce). A POINTER select (no copy) so the showcase binds the chosen SSBO directly. sh1 may be
// null when bounceCount==1 (it is never dereferenced on that path). Pure / deterministic.
template <typename T>
inline const T* SelectBounceSH(int bounceCount, const T* sh0, const T* sh1) {
    return (ClampBounceCount(bounceCount) >= 2) ? sh1 : sh0;
}

// --- The CPU mirror of one capture-with-GI fragment's added indirect (probe_bake_gi.frag). ----------
// BounceIndirect(wpos, N, grid, sh0, probeCount, albedo, giStrength): the indirect radiance the bounce-1
// capture ADDS to a captured surface — the SAME term the NEW probe_bake_gi.frag computes:
//   indirect = InterpolateIrradianceSH(wpos, N) * (1 - metallic=0)   // captured walls are dielectric
//   added    = indirect * albedo * giStrength
// All inputs non-negative (a positive-albedo scene, giStrength>=0, the SH0 irradiance of a positive
// direct capture) -> the added radiance is component-wise >= 0: the bounce only ADDS light. On the
// disabled path (probeCount<=0 / degenerate grid) InterpolateIrradiance returns {0,0,0} so the add is
// the zero vector. Pure / deterministic.
inline math::Vec3 BounceIndirect(const math::Vec3& wpos, const math::Vec3& N, const ProbeGrid& grid,
                                 const ProbeSH* sh0, int probeCount, const math::Vec3& albedo,
                                 float giStrength) {
    math::Vec3 irr = probesh::InterpolateIrradiance(wpos, grid, sh0, probeCount, N);
    return math::Vec3{irr.x * albedo.x * giStrength,
                      irr.y * albedo.y * giStrength,
                      irr.z * albedo.z * giStrength};
}

// --- The 2nd-bounce monotonicity predicate (the CPU mirror's brighter-never-darker invariant). -------
// BounceAddsLight(added): true when every component of the bounce-1 added radiance is >= 0 (the bounce
// only adds light). Because SH-encode is LINEAR and the cosine-lobe reconstruction of a non-negative
// added DC radiance is non-negative, a SH1 built from (direct + added) reconstructs to an irradiance >=
// the SH0 (direct-only) irradiance for a positive-albedo scene. Pure / deterministic.
inline bool BounceAddsLight(const math::Vec3& added) {
    return added.x >= 0.0f && added.y >= 0.0f && added.z >= 0.0f;
}

}  // namespace hf::render::probemb
