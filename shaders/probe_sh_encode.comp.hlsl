// Slice DJ — DDGI Slice 3: the PROBE SH-ENCODE compute pass. ONE thread per probe (dimX*dimY*dimZ),
// dispatched as enough HF_ENCODE_THREADS workgroups to cover the grid. Each thread loops the FIXED
// host-precomputed sample set (one entry per cube-texel sample of that probe's captured cubemap),
// reads the captured radiance at the sample, and accumulates it into 3rd-order real spherical-harmonic
// coefficients (9 per RGB channel = 27 floats), then normalizes by the total solid angle and writes the
// per-probe ProbeSH[probe] record into the flat SH SSBO.
//
// THE BIT-EXACT GPU==CPU PROOF: SHEncodeAccumulate + SHNormalize are copied VERBATIM from
// engine/render/probe_sh.h, so this pass and the CPU reference (probesh::SHEncodeAccumulate over the
// SAME flat radiance store, the SAME uploaded sample basis weights + solid-angle weights) produce a
// BIT-EXACT per-probe SH buffer (memcmp == 0). The cross-backend FP discipline (from DH): the sample
// DIRECTIONS (cube-texel dirs, which need normalize=sqrt) and their SHBasis9 weights + solid-angle
// weights are HOST-PRECOMPUTED ONCE (CPU) and uploaded as exact float32 bits both this shader + the CPU
// reference read — so the encode is a pure (radiance * stored-weight) mad accumulation with NO
// transcendentals on the GPU hot path -> bit-exact + deterministic. THE zero-radiance==zero-SH PROOF: a
// probe whose radiance slots are all zero accumulates 0 into every coeff -> ProbeSH all zero. THE
// probeCount=0 NO-OP PROOF: dimX==0 -> probeCount()==0 -> the host dispatches 0 groups -> this main()
// never runs -> the SH SSBO is untouched (== the cleared upload, all zero).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal they land at buffer(kCsStorage+0..2)):
//   b0 gParams   : EncodeParams — {probeCount, sampleCount, totalWeight} + the host-precomputed sample
//                  table (per sample: the 9 SHBasis9 weights + the solid-angle weight), READ.
//   b1 gRadiance : the FLAT captured-radiance store float4[probeCount*sampleCount] — radiance[probe*N+s]
//                  is the radiance the showcase read from probe `probe`'s captured cube at sample `s`'s
//                  face/texel (host-sampled so the GPU + CPU read the IDENTICAL float32 bits), READ.
//   b2 gProbeSH  : the flat ProbeSH[probeCount] output (9 SH coeffs x 3 channels), WRITE.
// (NO sampled textures — same style as froxel_inject / probe_raytrace. The radiance + the sample table
//  are flat SSBOs so the GPU and CPU read them identically.)
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL are the HF_MSL_GEN
// generation-path guard (none needed here) + [[vk::binding]] decorations (same as froxel_inject /
// probe_raytrace), not backend CODE symbols. spirv-cross maps the SPIR-V bindings to Metal's flat
// compute buffer indices so the SAME HLSL feeds Vulkan (DXC) and Metal (glslang->spirv-cross):
// bit-identical math.

#define HF_ENCODE_THREADS 64
#define HF_MAX_SH_SAMPLES 1536   // the fixed-cap sample table (6 faces x kEncodeFaceDim^2; see showcase)

// One precomputed sample's encode weights (std430): the 9 SHBasis9 weights + the solid-angle weight,
// computed ONCE on the host (probesh::SHBasis9 over the normalized cube-texel dir) and uploaded as exact
// float32 bits. Padded to a float4 boundary (10 floats -> 12) for std430 array stride stability.
struct EncodeSample {
    float basis[9];      // the 9 real-SH basis weights at this sample's (normalized) direction
    float saWeight;      // the per-sample solid-angle weight (4*pi / sampleCount for a full-sphere cube)
    float pad0, pad1;    // pad the per-element stride to 12 floats (48 bytes) for std430 cleanliness
};

// The encode params + the host-precomputed sample table (std430). probeCount/sampleCount/totalWeight
// drive the loop; `samples` is the fixed sample set EVERY probe shares (the cube geometry is the same
// per probe — only the radiance differs).
struct EncodeParams {
    uint4  counts;                          // x=probeCount, y=sampleCount, zw unused
    float4 norm;                            // x=totalWeight (sum of saWeight over the sample set), yzw unused
    EncodeSample samples[HF_MAX_SH_SAMPLES];
};

// One per-probe SH record (std430, 108 bytes): mirrors render::probesh::ProbeSH. coeff[basis][channel].
struct ProbeSH {
    float coeff[9][3];
};

[[vk::binding(0, 0)]] RWStructuredBuffer<EncodeParams> gParams   : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float4>       gRadiance : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<ProbeSH>      gProbeSH  : register(u2);

[numthreads(HF_ENCODE_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint probeCount  = gParams[0].counts.x;
    uint sampleCount = gParams[0].counts.y;
    float totalWeight = gParams[0].norm.x;

    uint p = gid.x;
    if (p >= probeCount) return;

    // Accumulate the captured radiance into the 9 SH coefficients (per RGB channel). Each sample folds
    // radiance * (basis[i] * saWeight) with a SINGLE correctly-rounded mad — VERBATIM the CPU
    // probesh::SHEncodeAccumulate (the w = basis[i]*saWeight host product, then mad(radiance, w, acc)).
    float acc[9][3];
    [unroll] for (int i = 0; i < 9; ++i) { acc[i][0] = 0.0; acc[i][1] = 0.0; acc[i][2] = 0.0; }

    uint base = p * sampleCount;   // probe p's radiance block start (radiance[probe*N + s])
    [loop] for (uint s = 0; s < sampleCount; ++s) {
        float3 radiance = gRadiance[base + s].rgb;
        float saWeight = gParams[0].samples[s].saWeight;
        [unroll] for (int j = 0; j < 9; ++j) {
            float w = gParams[0].samples[s].basis[j] * saWeight;   // host-stable per-sample per-basis weight
            acc[j][0] = mad(radiance.r, w, acc[j][0]);
            acc[j][1] = mad(radiance.g, w, acc[j][1]);
            acc[j][2] = mad(radiance.b, w, acc[j][2]);
        }
    }

    // Normalize by the total solid-angle weight (VERBATIM probesh::SHNormalize: a reciprocal multiply;
    // totalWeight<=0 -> leave the accumulation untouched, which stays zero for an empty sample set).
    if (totalWeight > 0.0) {
        float inv = 1.0 / totalWeight;
        [unroll] for (int k = 0; k < 9; ++k) {
            acc[k][0] *= inv; acc[k][1] *= inv; acc[k][2] *= inv;
        }
    }

    [unroll] for (int o = 0; o < 9; ++o) {
        gProbeSH[p].coeff[o][0] = acc[o][0];
        gProbeSH[p].coeff[o][1] = acc[o][1];
        gProbeSH[p].coeff[o][2] = acc[o][2];
    }
}
