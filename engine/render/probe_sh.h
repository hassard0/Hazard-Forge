#pragma once
// Slice DJ — DDGI Slice 3: Probe SH-Encode — pure CPU (header-only, no device, no backend symbols).
// Namespace hf::render::probesh. The THIRD slice of the GLOBAL ILLUMINATION (DDGI) flagship arc (after
// DH's probe ray-trace + DI's radiance capture). Mirrors probe_gi.h / probe_capture.h style: a tiny
// shared-math header ABOVE the RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only
// mentions of "vk"/"MTL" anywhere in this slice's above-seam files are seam-discipline doc comments).
// The compute shader shaders/probe_sh_encode.comp.hlsl copies SHBasis9 / SHEncodeAccumulate /
// SHNormalize VERBATIM, so tests/probe_sh_test.cpp exercises the EXACT encode the GPU pass runs — which
// is what makes the GPU SH SSBO BIT-EXACT to the CPU SH-encode over the same captured cube radiance AND
// bit-identical cross-backend.
//
// THE TECHNIQUE (the DDGI SH-irradiance data layer): each captured probe cubemap (DI) is sampled in a
// FIXED deterministic order and the radiance accumulated into 3rd-order REAL spherical-harmonic
// coefficients (9 basis functions x 3 RGB channels = 27 floats/probe), producing a compact per-probe SH
// irradiance record — the data the GI composite (slices DK/DL) samples. This slice is ONLY the verified
// SH-encode layer: no probe relighting / neighbor gather / GI composite yet, no visible indirect light.
//
// THE BIT-EXACT GPU==CPU PROOF (what makes this verifiable): the showcase reads the captured-radiance
// store + the GPU per-probe ProbeSH SSBO back, and on the CPU runs the SAME SHEncodeAccumulate /
// SHNormalize over the read-back radiance, with the SAME host-precomputed sample directions + SHBasis9
// weights + solid-angle weights that the GPU read. The two SH buffers must be BIT-EXACT (memcmp) — the
// probe analog of the DH gpu==cpu ray-hit proof. THE zero-radiance==zero-SH PROOF: a probe whose cube is
// all-zero radiance accumulates 0 into every coeff -> its ProbeSH is exactly zero. THE probeCount=0
// NO-OP PROOF: dimX==0 -> probeCount()==0 -> EncodeDispatchGroups()==0 -> DispatchCompute(0) -> the SH
// SSBO is untouched (== the cleared upload, all zero).
//
// THE CROSS-BACKEND FP DISCIPLINE (from DH, MANDATORY for the bit-exact proof): cos/sin/sqrt/normalize
// are NOT bit-identical CPU-libm vs the GPU, and a plain a+b*c contracts to fma on Metal (fast-math) but
// not on Vulkan/DXC or the CPU -> a ~1-ULP divergence. THEREFORE: the SH SAMPLE DIRECTIONS (cube-texel
// directions, which need a normalize = sqrt) AND their SHBasis9 weights AND solid-angle weights are
// HOST-PRECOMPUTED ONCE on the CPU (SHSampleTable) and uploaded as exact float32 bits that BOTH the GPU
// encode shader AND the CPU reference read; the per-sample accumulation uses explicit std::fma (CPU) /
// mad (shader). SHBasis9 itself is POLYNOMIAL in the (already-normalized) direction (no transcendentals)
// — but because the table stores the basis weights directly, the encode never recomputes them on the
// GPU, so the encode is a pure (radiance * stored-weight) fma accumulation -> bit-exact + deterministic.
//
// CONVENTIONS (must match the shader + the showcase EXACTLY):
//   * 3rd-order REAL SH (bands 0,1,2 = 9 basis functions). SHBasis9 is the standard Cartesian-polynomial
//     form (documented constants below). The directions are UNIT vectors (the table stores normalized).
//   * Per-probe SH flat index == the probe's cx-major flat index (idx = px + py*dimX + pz*(dimX*dimY)),
//     the SAME order probe_gi.h / probe_capture.h use. The SH SSBO is a flat ProbeSH[probeCount].
//   * The encode normalizes by the total solid-angle weight (a full-sphere cube -> ~4*pi). The optional
//     cosine-lobe band scales (A0=pi, A1=2*pi/3, A2=pi/4) convert the radiance SH into an IRRADIANCE SH
//     that SHEvaluate reconstructs directly; they are applied in SHEvaluate (documented), NOT baked into
//     the stored coeffs, so the stored ProbeSH is the raw radiance projection (what the GPU==CPU proof
//     compares) and the band scales are a pure reconstruction-time convolution.
//
// Pure, deterministic functions: no RNG, no time, no device.

#include "math/math.h"
#include "render/probe_gi.h"   // SHARE the probe grid (ProbeGrid / probePos / probeCount / flatIndex)

#include <cmath>
#include <cstdint>

namespace hf::render::probesh {

// Re-export the SHARED probe grid so the encode loop + the tests write probesh::ProbeGrid and the reuse
// is explicit at the call site (the SAME type DH/DI use — DJ does not redefine it).
using probegi::ProbeGrid;

// --- The 3rd-order real-SH constants (documented; SHBasis9 is polynomial in the unit direction). -----
// Standard real spherical-harmonic basis constants (Y_l^m, condon-shortley convention folded into the
// polynomial form). Band 0 (l=0): Y00 = 0.5*sqrt(1/pi). Band 1 (l=1): Y1m = 0.5*sqrt(3/pi) * {y, z, x}.
// Band 2 (l=2): {sqrt(15/pi)/2 * xy, sqrt(15/pi)/2 * yz, sqrt(5/pi)/4 * (3z^2-1), sqrt(15/pi)/2 * xz,
// sqrt(15/pi)/4 * (x^2-y^2)}. These are the exact float32 constants the shader copies VERBATIM.
inline constexpr float kY00 = 0.282094791773878140f;   // 0.5 * sqrt(1/pi)
inline constexpr float kY1  = 0.488602511902919920f;   // 0.5 * sqrt(3/pi)            (band-1 scale)
inline constexpr float kY2a = 1.092548430592079200f;   // 0.5 * sqrt(15/pi)           (xy, yz, xz)
inline constexpr float kY2b = 0.315391565252520000f;   // 0.25 * sqrt(5/pi)           (3z^2-1)
inline constexpr float kY2c = 0.546274215296039600f;   // 0.25 * sqrt(15/pi)          (x^2-y^2)

// The cosine-lobe (Lambert) convolution band scales (Ramamoorthi & Hanrahan 2001): converting a RADIANCE
// SH projection into the IRRADIANCE reconstructed in a surface-normal direction multiplies band l by
// A_l = {pi, 2*pi/3, pi/4}. Applied at RECONSTRUCTION time in SHEvaluate (documented), NOT baked into the
// stored coeffs.
inline constexpr float kA0 = 3.14159265358979324f;             // pi
inline constexpr float kA1 = 2.09439510239319549f;             // 2*pi/3
inline constexpr float kA2 = 0.785398163397448310f;            // pi/4

// --- The per-probe SH record (std430, 108 bytes: 9 coeffs x 3 RGB channels). -------------------------
// ProbeSH stores the 3rd-order real SH projection of one probe's captured radiance, per RGB channel:
// coeff[i] = {r, g, b} for basis function i. sizeof == 9*3*4 == 108 bytes, std430-clean (a tight float
// array, 4-byte aligned). Mirrors the shader's RWStructuredBuffer<ProbeSH> element.
struct ProbeSH {
    float coeff[9][3];   // coeff[basis][channel]; channel 0=r,1=g,2=b
};
static_assert(sizeof(ProbeSH) == 108, "ProbeSH must be 108 bytes (9 SH coeffs x 3 channels, std430)");

// --- The 9 real SH basis functions at a UNIT direction (polynomial; no transcendentals). -------------
// SHBasis9(dir, out[9]) evaluates Y00, Y1{-1,0,1}, Y2{-2,-1,0,1,2} at the normalized direction `dir`.
// Polynomial in (x,y,z) -> bit-exact-friendly (no cos/sin/sqrt). The COPY in probe_sh_encode.comp.hlsl
// is VERBATIM; but the encode reads the PRECOMPUTED weights from the uploaded table (this is used to
// BUILD that table on the host + by the tests, not on the GPU hot path). `dir` MUST be unit length.
inline void SHBasis9(const math::Vec3& dir, float out[9]) {
    const float x = dir.x, y = dir.y, z = dir.z;
    // Band 0 (l=0).
    out[0] = kY00;
    // Band 1 (l=1): proportional to y, z, x.
    out[1] = kY1 * y;
    out[2] = kY1 * z;
    out[3] = kY1 * x;
    // Band 2 (l=2): xy, yz, 3z^2-1, xz, x^2-y^2.
    out[4] = kY2a * (x * y);
    out[5] = kY2a * (y * z);
    out[6] = kY2b * (3.0f * z * z - 1.0f);
    out[7] = kY2a * (x * z);
    out[8] = kY2c * (x * x - y * y);
}

// --- Accumulate one captured sample into the SH projection (the encode inner loop). ------------------
// SHEncodeAccumulate(sh, radiance, basis[9], solidAngleWeight): fold one cube-texel sample's radiance
// into every coefficient with a SINGLE correctly-rounded fused multiply-add (std::fma) per channel:
//   sh.coeff[i][c] += radiance[c] * basis[i] * solidAngleWeight.
// `basis` is the PRECOMPUTED SHBasis9 weights for this sample (from the uploaded table) and
// `solidAngleWeight` the precomputed per-texel solid-angle weight — both host float32, read identically
// by the GPU + CPU. We fold the (basis[i]*solidAngleWeight) product ONCE (host-stable) then fma the
// radiance, matching the shader's mad(radiance, w, acc). This is the cross-backend bit-exactness key.
inline void SHEncodeAccumulate(ProbeSH& sh, const math::Vec3& radiance, const float basis[9],
                               float solidAngleWeight) {
    for (int i = 0; i < 9; ++i) {
        const float w = basis[i] * solidAngleWeight;   // the per-sample per-basis weight (host float32)
        sh.coeff[i][0] = std::fma(radiance.x, w, sh.coeff[i][0]);
        sh.coeff[i][1] = std::fma(radiance.y, w, sh.coeff[i][1]);
        sh.coeff[i][2] = std::fma(radiance.z, w, sh.coeff[i][2]);
    }
}

// --- Normalize the accumulated projection by the total solid-angle weight. ---------------------------
// SHNormalize(sh, totalWeight): divide every coefficient by the summed solid-angle weight of the sample
// set (~4*pi for a full-sphere cube), turning the weighted sum into the Monte-Carlo SH projection
// integral. totalWeight<=0 -> leave the coeffs untouched (a degenerate/empty sample set keeps zero).
// Uses a reciprocal multiply (host float32) so the GPU's `* invTotal` matches to the bit.
inline void SHNormalize(ProbeSH& sh, float totalWeight) {
    if (totalWeight <= 0.0f) return;
    const float inv = 1.0f / totalWeight;
    for (int i = 0; i < 9; ++i) {
        sh.coeff[i][0] *= inv;
        sh.coeff[i][1] *= inv;
        sh.coeff[i][2] *= inv;
    }
}

// --- Reconstruct the irradiance in a direction from the SH projection. -------------------------------
// SHEvaluate(sh, dir, cosineLobe): reconstruct the radiance (cosineLobe=false) or the cosine-convolved
// IRRADIANCE (cosineLobe=true, the default — the diffuse term a surface with this normal receives) by
// dotting the per-channel coeffs with the SHBasis9 at `dir`, applying the band scales A_l when cosineLobe
// is set. `dir` MUST be unit length. Used by the GI composite + the showcase viz.
inline math::Vec3 SHEvaluate(const ProbeSH& sh, const math::Vec3& dir, bool cosineLobe = true) {
    float basis[9];
    SHBasis9(dir, basis);
    // Per-band reconstruction scale: identity for raw radiance, the cosine-lobe A_l for irradiance.
    const float s0 = cosineLobe ? kA0 : 1.0f;
    const float s1 = cosineLobe ? kA1 : 1.0f;
    const float s2 = cosineLobe ? kA2 : 1.0f;
    const float bw[9] = {basis[0] * s0,
                         basis[1] * s1, basis[2] * s1, basis[3] * s1,
                         basis[4] * s2, basis[5] * s2, basis[6] * s2, basis[7] * s2, basis[8] * s2};
    math::Vec3 r{0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 9; ++i) {
        r.x += sh.coeff[i][0] * bw[i];
        r.y += sh.coeff[i][1] * bw[i];
        r.z += sh.coeff[i][2] * bw[i];
    }
    return r;
}

// --- The disabled / dispatch-sizing path (one thread per probe). -------------------------------------
// kEncodeThreads = the compute workgroup size. EncodeDispatchGroups(grid) = the number of kEncodeThreads
// workgroups to cover probeCount probes, or EXACTLY 0 when probeCount<=0 (dimX/dimY/dimZ == 0 ->
// probeCount == 0 -> 0 groups -> DispatchCompute(0) -> the SH SSBO is untouched == the cleared upload,
// all zero). This is the byte-identical no-op the probeCount=0 proof rests on (the probe_gi.h
// ProbeDispatchGroups(...) -> 0 analog). Pure / deterministic.
inline constexpr int kEncodeThreads = 64;
inline int EncodeDispatchGroups(const ProbeGrid& grid) {
    int n = grid.probeCount();
    return (n <= 0) ? 0 : (n + kEncodeThreads - 1) / kEncodeThreads;
}

}  // namespace hf::render::probesh
