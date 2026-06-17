#pragma once
// Slice DO — DDGI Visibility Slice 1: Per-Probe Distance-Moment Capture — pure CPU (header-only, no
// device, no backend symbols). Namespace hf::render::probedist. The FIRST slice of the DDGI VISIBILITY
// sub-arc (after the DI radiance-capture / DJ SH-encode / DL interp / DN composite irradiance chain).
// Mirrors probe_capture.h EXACTLY (DI:radiance :: DO:distance): a tiny shared-math header ABOVE the RHI
// seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only mentions of "vk"/"MTL" anywhere in
// this slice's above-seam files are seam-discipline doc comments). The showcase capture loop
// (samples/hello_triangle/main.cpp --probedist-shot, metal_headless/visual_test.mm --probedist) AND the
// unit test (tests/probe_dist_test.cpp) consume the SAME DistTexelCount / ProbeDistFaceIndex /
// MomentsFromDistance and the SAME cubemap::FaceView/FaceProj — so the GPU capture loop and the CPU test
// agree on the probe×face×texel slot layout and on the 6-face projection.
//
// THE TECHNIQUE (the DDGI VISIBILITY data layer — what the next slice DP, Chebyshev occlusion, consumes
// to KILL DDGI light-leak through walls): for each probe in a (small) world-space lattice, RENDER the
// scene geometry from the probe center into the 6 faces of the DD cube RT, but instead of LIT RADIANCE
// (DI) each fragment writes the LINEAR WORLD-DISTANCE from the probe center to the surface — d =
// length(worldPos - probeCentre) — packed as the two MOMENTS float2(d, d*d) (mean distance + mean
// distance², at single-sample granularity). The faces are read back into a flat per-probe distance-
// moment store. The DP slice will sample this moment cube in the probe→shading-point direction and run
// the Chebyshev/variance visibility test (mean=m[0], variance=m[1]-m[0]², p=variance/(variance+(d-mean)²))
// to weight probes by visibility — leak GONE. DO produces NO visible lighting change yet (it is the data
// layer only). To keep render counts tractable the DO capture grid is the SAME SMALL grid DI uses
// (ProbeGrid{dimX=2,dimY=2,dimZ=2} = 8 probes → 8×6 = 48 face renders). The captured distance is
// GEOMETRY-ONLY: no lighting, no shadows, no RNG, no clock → deterministic + (given the identical mesh)
// cross-backend-coherent, the engine's cross-backend golden contract.
//
// THE CAPTURE-CORRECTNESS PROOF (the DD/DI pattern, per-backend): because each face uses the IDENTICAL
// scene + distance pipeline (probe_dist.frag) + push constants + viewport a direct render would, the
// captured distance face equals a standalone distance render with that face's FaceView/FaceProj. The
// showcase renders probe-0's face 0 DIRECTLY into a 2D RT and asserts it is BYTE-IDENTICAL (SHA / memcmp)
// to ReadCubemapFace(probe-0, face 0). THE probeCount=0 NO-OP PROOF: dimX==0 → probeCount()==0 →
// DistTexelCount()==0 → the capture loop body never runs → the per-probe moment store stays at its
// cleared upload (a byte-identical skip-loop no-op, the probe_gi.h dispatch-0 analog).
//
// THE MOMENT GPU==CPU BIT-EXACT PROOF (the framing that makes DO correct):
//   `length(worldPos - centre)` is `sqrt(dot(...))`, and `sqrt` is NOT guaranteed bit-identical between
//   the CPU libm and the GPU (the DH cross-backend lesson). So the GPU==CPU BIT-EXACT proof is NOT on the
//   sqrt itself — it is on the MOMENT-FROM-DISTANCE step: the GPU produces the per-texel DISTANCE d (the
//   shader's R channel = fp16(length(...))); the host reads d back and DERIVES the moment store via
//   MomentsFromDistance(d) = { d, d*d }. The proof re-derives the moments a SECOND time, INDEPENDENTLY,
//   straight from the same read-back distance bytes and memcmp's bit-for-bit against the store — so GIVEN
//   the SAME distance bytes `d`, the moment-from-distance step is BIT-EXACT and deterministic. This holds
//   because the second moment is a SINGLE BARE MULTIPLY d*d (one IEEE-754 round-to-nearest). We
//   deliberately do NOT use std::fma / mad / a+b*c here: a bare multiply is already exactly correctly-
//   rounded for one product, and forcing an fma would risk a contraction divergence (Metal contracts a+b*c
//   to fma, others don't — the DH 1-ULP trap). The shader ALSO emits `d*d` as a bare multiply (G channel,
//   per the slice spec), but the host derives the second moment from the read-back DISTANCE so the store is
//   fp32-exact and free of the fp16-store tie ambiguity a raw G readback would carry (a half-way product
//   d*d rounds driver-dependently at the fp16 store). The capture-distance proof (proof 1) is a
//   RENDER-EQUIVALENCE proof per backend (same shader, same mesh, same view/proj — the captured face ==
//   a direct distance render, byte-identical); the moment proof (proof 2) is the GPU==CPU bit-exact proof
//   on the moment-from-distance step. Be explicit about which proof covers which step — do NOT claim a
//   GPU==CPU bit-exact on the sqrt.
//
// CONVENTIONS (must match cubemap.h + probe_gi.h + probe_capture.h EXACTLY):
//   * The probe grid + its probePos/flatIndex/probeCount come from probe_gi.h (re-used unchanged). The DO
//     capture grid is the same SMALL grid DI uses.
//   * The 6 faces use cubemap::FaceView(face, probeCenter) + cubemap::FaceProj(zNear,zFar) — the SAME
//     6-axis projection + the SAME +Y/-Y up-vector convention DD/DI's capture uses, centered on the probe.
//     Face order is +X,-X,+Y,-Y,+Z,-Z (cubemap.h).
//   * The moment store is a FLAT per-probe block of a SMALL kDistFace×kDistFace×6 distance cube. The
//     per-probe distance face is kDistFace = 8 → 8*8*6 = 384 texels/probe → ProbeDistMoments[probeCount*384].
//     Probe p, face f, texel (u,v) lives at slot p*384 + f*64 + v*8 + u (probe-major → face → row-major
//     texel). This covers every (p,f,u,v) exactly once with no overlap (the round-trip is exact, hand-
//     checked by the test). The per-face block START for probe p face f is ProbeDistFaceIndex(p,f) = the
//     analog of DI's ProbeFaceIndex (here a TEXEL-block index, since DO stores moments per texel).
//
// Pure, deterministic functions: no RNG, no time, no device.

#include "math/math.h"
#include "render/probe_gi.h"   // SHARE the probe grid (ProbeGrid / probePos / probeCount / flatIndex)
#include "render/cubemap.h"    // SHARE the 6-face cube projection (FaceView / FaceProj / FaceViewProj)

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace hf::render::probedist {

// Re-export the SHARED probe grid + cube-face count so the capture loop + the tests write
// probedist::ProbeGrid / probedist::kFaces and the reuse is explicit at the call site (these are the SAME
// types/values the DH ray-trace + the DD/DI capture use — DO does not redefine them).
using probegi::ProbeGrid;
inline constexpr int kFaces = cubemap::kFaces;   // 6 — the cube faces, +X,-X,+Y,-Y,+Z,-Z

// The per-face square edge of the DO distance cube: 8 → 8×8 = 64 texels/face. SMALL (the moment store is
// the visibility data layer; a coarse distance cube is plenty for the Chebyshev test the DP slice runs).
inline constexpr int kDistFace = 8;
inline constexpr int kFaceTexels = kDistFace * kDistFace;   // 64 texels per face
inline constexpr int kProbeTexels = kFaceTexels * kFaces;   // 384 texels per probe (8×8×6)

// The far / sky-miss distance the shader writes where no geometry is hit (kRayMiss-style). Documented
// here so the host viz + any reference agree with the shader's far value.
inline constexpr float kDistFar = 1.0e4f;

// --- The per-texel distance moments (meanDist, meanDist²) — the SSBO record. ------------------------
// ProbeDistMoments is ONE texel's two distance moments: m[0] = d (the centre-ray distance), m[1] = d*d
// (its square). std430-tight (sizeof == 8, two contiguous float). The GPU writes float2(d, d*d) into the
// RG/RGBA16F distance target per fragment; the CPU reference writes MomentsFromDistance(d) — bit-identical
// (see the header banner: one bare multiply each).
struct ProbeDistMoments {
    float m[2];   // m[0] = meanDist (d), m[1] = meanDist² (d*d)
};
static_assert(sizeof(ProbeDistMoments) == 8, "ProbeDistMoments must be std430-tight (two float == 8 bytes)");
static_assert(std::is_trivially_copyable<ProbeDistMoments>::value, "ProbeDistMoments must be trivially copyable");

// --- The total distance-texel count (the moment store size + the per-texel loop trip count). --------
// DistTexelCount(grid) = grid.probeCount()*kProbeTexels = the number of distance-moment texels the moment
// store holds (probeCount * 8×8×6). EXACTLY 0 when probeCount<=0 (dimX/dimY/dimZ == 0 → probeCount == 0 →
// 0 texels → the capture loop body never runs → the moment store is untouched == its cleared upload). This
// is the byte-identical no-op the probeCount=0 proof rests on (the probe_capture.h CaptureFaceCount→0 /
// probe_gi.h ProbeDispatchGroups→0 analog). Pure / deterministic.
inline int DistTexelCount(const ProbeGrid& grid) {
    int n = grid.probeCount();
    return (n <= 0) ? 0 : n * kProbeTexels;
}

// --- The per-probe×face TEXEL-BLOCK start index into the flat moment store. -------------------------
// ProbeDistFaceIndex(probe, face) = probe*kProbeTexels + face*kFaceTexels = the index of texel (u=0,v=0)
// of probe `probe`, face `face` in the flat moment store. The flat store is a ProbeDistMoments[
// probeCount*kProbeTexels] block; probe p face f texel (u,v) lives at ProbeDistFaceIndex(p,f) + v*kDistFace
// + u. This is the DI ProbeFaceIndex analog (here a TEXEL-block start, since DO stores per-texel moments).
// Pure / deterministic.
inline int ProbeDistFaceIndex(int probe, int face) {
    return probe * kProbeTexels + face * kFaceTexels;
}

// ProbeFaceIndexRaw(probe, face) = probe*kFaces + face — the probe-major face-minor slot for the showcase's
// RAW read-back FACE store (one entry per (probe,face), holding that face's read-back bytes for the
// capture-correctness SHA proof). DISTINCT from ProbeDistFaceIndex (which indexes the per-TEXEL moment
// store); this is the DI ProbeFaceIndex analog for the coarse per-face byte block. Pure / deterministic.
inline int ProbeFaceIndexRaw(int probe, int face) { return probe * kFaces + face; }

// The full flat texel index for probe `probe`, face `face`, texel (u,v): probe-major → face → row-major
// (v outer, u inner). slot = probe*384 + face*64 + v*8 + u. Covers [0, probeCount*384) exactly once
// (a bijection, hand-checked by the test). Pure / deterministic.
inline int ProbeDistTexelIndex(int probe, int face, int u, int v) {
    return ProbeDistFaceIndex(probe, face) + v * kDistFace + u;
}

// The inverse of ProbeDistTexelIndex: recover (probe,face,u,v) from a flat slot. Round-trips
// ProbeDistTexelIndex for every (probe,face,u,v) in the grid. Pure / deterministic.
inline void ProbeDistTexelFromIndex(int slot, int& outProbe, int& outFace, int& outU, int& outV) {
    outProbe = slot / kProbeTexels;
    int rem  = slot % kProbeTexels;
    outFace  = rem / kFaceTexels;
    int t    = rem % kFaceTexels;
    outV     = t / kDistFace;
    outU     = t % kDistFace;
}

// The per-face view-projection from the probe center (the camera the face's distance render uses). A thin
// reuse of cubemap::FaceViewProj at the probe's world position — the SAME 6-axis projection DD/DI's capture
// uses, here centered on the probe. The capture loop + the test both call this for the face render math.
inline math::Mat4 ProbeFaceViewProj(int face, const math::Vec3& probeCenter, float zNear, float zFar) {
    return cubemap::FaceViewProj(face, probeCenter, zNear, zFar);
}

// --- The per-texel moment from a distance (the GPU==CPU bit-exact reference). -----------------------
// MomentsFromDistance(d) = { d, d*d } — the two distance moments at single-sample granularity. The GPU
// emits float2(d, d*d) per fragment; this is the CPU reference the bit-exact proof memcmp's against the
// read-back moment store. The second moment is a SINGLE BARE MULTIPLY d*d (one IEEE-754 round): NOT
// std::fma, NOT a+b*c — a bare multiply is exactly correctly-rounded for one product and matches the
// shader's bare `d*d`, so GPU and CPU each do ONE multiply-rounding and agree to the bit (see the banner;
// this deliberately avoids the DH fma-contraction 1-ULP trap). MomentsFromDistance(0) == {0,0}. Pure /
// deterministic.
inline ProbeDistMoments MomentsFromDistance(float d) {
    ProbeDistMoments r;
    r.m[0] = d;
    r.m[1] = d * d;   // BARE multiply (one rounding) — matches the shader's bare d*d for the bit-exact proof
    return r;
}

// --- The per-probe MEAN captured distance (the swatch-viz value, deterministic). --------------------
// ProbeMeanDistance(moments, texelCount) folds `texelCount` per-texel moments' FIRST moment (m[0] = d)
// into a single mean distance (the per-probe debug-viz value: a colored swatch sphere at the probe — near
// warm / far cool over a normalized ramp). The accumulation is in a FIXED texel order and each fold is an
// explicit std::fma(d, w, acc) with the reciprocal weight w = 1/texelCount precomputed once — a SINGLE
// correctly-rounded multiply-add per texel (the DH cross-backend discipline; this is a HOST-only viz
// reduction, not a GPU==CPU bit-exact path — the bit-exact proof is on MomentsFromDistance). texelCount<=0
// / null → 0 (an empty probe has no distance). Pure / deterministic.
inline float ProbeMeanDistance(const ProbeDistMoments* moments, int texelCount) {
    if (texelCount <= 0 || moments == nullptr) return 0.0f;
    const float w = 1.0f / (float)texelCount;   // reciprocal weight, precomputed ONCE (host float32)
    float acc = 0.0f;
    for (int i = 0; i < texelCount; ++i) {
        acc = std::fma(moments[i].m[0], w, acc);   // single correctly-rounded multiply-add
    }
    return acc;
}

}  // namespace hf::render::probedist
