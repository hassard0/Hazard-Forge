// Slice GI2 — Deterministic Lumen-class GI: the INTEGER 3rd-order SH ENCODE compute pass (THE CRUX of
// FLAGSHIP #29: DETERMINISTIC LUMEN-CLASS GI). ONE thread per PROBE (p < probeCount). Each thread reads its
// 16 GI1 GiRadiance rays (gRadiance[p*16 + d]) and runs FxSHEncodeProbe VERBATIM — the int64-accumulator
// SH projection: acc[i][c] (int64, Q32.32) += (int64)rad.ch * (int64)kGiSHBasis[r][i] over the 16 rays,
// STAYING WIDE, then the single normalize (/ N then >> kFrac) narrows to Q16.16 EXACTLY ONCE, and writes
// the 9x3 FxProbeSH to gSH[p]. Bit-identical to the CPU gi::FxSHEncodeProbe / EncodeAllProbes.
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it consumes
// the host-snapped Q16.16 int radiance + the host-precomputed Q16.16 kGiSHBasis table (NO sqrt/cos/pow on
// the hot path) and runs the SAME pure-integer int64 multiply-accumulate + the SAME single arithmetic
// >>kFrac narrow the header runs. A divergence vs engine/render/gi.h is exactly what the host's GPU==CPU
// memcmp catches.
//
// INTEGER WIDTH (the determinism crux): the encode accumulator is int64 — a single radiance(Q16.16) *
// basis(Q16.16) term is a Q32.32 product (~1.0*~1.0 = 65536^2 = 4.29e9 > 2^31, overflowing int32), so the
// 16-ray sum MUST be held in int64 and quantized to Q16.16 only once at the end (no per-sample re-quantize).
// HLSL SM6 supports int64_t (the grain_integrate.comp / fluid_integrate.comp / swraster.comp pattern — DXC
// -spirv with the Int64 capability). glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in
// HLSL, so this shader is VULKAN-SPIR-V-ONLY (NOT in metal_headless/CMakeLists.txt hf_gen_msl); on Metal the
// --gi2-shencode showcase runs the CPU gi::EncodeAllProbes (byte-identical by construction).
//
// NO RayQuery, NO accel structure — pure-integer compute over the GI1 radiance buffer (this is why GI2 is a
// strict-zero MSL-NATIVE-CLASS proof on the Vulkan side; only the int64 width keeps the shader Vulkan-only).
//
// Buffers (storage, compute bindings 0..2):
//   b0 gRadiance : the GI1 GiRadiance array (probeCount*16; r,g,b,_pad int4 == 16 B), READ.
//   b1 gSH       : the FxProbeSH array (probeCount; 9*3 coeffs + pad == 28 ints == 112 B), WRITE.
//   b2 gParams   : { probeCount, rayCount(16), 0, 0 }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// grain_integrate.comp / fluid_integrate.comp), not backend CODE symbols.

#define HF_GI_THREADS 64
#define HF_GI_FRAC 16          // MUST match gi.h::kFrac
#define HF_GI_RAYS 16          // MUST match gi.h::kGiRaysPerProbe

// std430 GiRadiance mirror (gi.h::GiRadiance): r,g,b,_pad Q16.16 ints, 16 bytes.
struct GiRad {
    int r, g, b, _pad;
};

// std430 FxProbeSH mirror (gi.h::FxProbeSH): 9*3 = 27 Q16.16 coeff ints + 1 pad int = 28 ints (112 B).
// Laid out flat as c[28]; c[i*3 + ch] == coeff[basis i][channel ch], c[27] == pad (kept 0).
struct FxProbeSH {
    int c[28];
};

struct GiSHParams {
    int4 cfg;   // x=probeCount, y=rayCount(16), z=0, w=0
};

[[vk::binding(0, 0)]] StructuredBuffer<GiRad>            gRadiance : register(t0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FxProbeSH>      gSH       : register(u0);
[[vk::binding(2, 0)]] StructuredBuffer<GiSHParams>       gParams   : register(t1);

// The host-precomputed integer SH basis table (Y_lm(dir) for the 16 baked kGiProbeDirs, Q16.16) — IDENTICAL
// bits to gi.h::kGiSHBasis (the generator comment lives in gi.h). 16 rays x 9 basis functions.
static const int kGiSHBasis[HF_GI_RAYS][9] = {
    {   18487,       0,   30020,   11143,       0,       0,   33830,   23359,    4335 },
    {   18487,   12609,   26017,  -13764,  -12120,   22909,   20266,  -25007,    1064 },
    {   18487,  -23164,   22014,    2033,   -3289,  -35610,    8639,    3125,  -18591 },
    {   18487,   21010,   18012,   16108,   23634,   26427,   -1050,   20261,   -6353 },
    {   18487,   -5016,   14009,  -28354,    9931,   -4907,   -8801,  -27738,   27191 },
    {   18487,  -16326,   10007,   25665,  -29259,  -11408,  -14614,   17934,   13692 },
    {   18487,   30375,    6004,   -8166,  -17320,   12735,  -18490,   -3424,  -29886 },
    {   18487,  -28361,    2001,  -14730,   29173,   -3964,  -20427,   -2059,  -20509 },
    {   18487,   10963,   -2001,   30019,   22981,   -1532,  -20427,   -4195,   27268 },
    {   18487,   12001,   -6004,  -29074,  -24365,   -5032,  -18490,   12190,   24485 },
    {   18487,  -27550,  -10007,   12892,  -24803,   19251,  -14614,   -9009,  -20697 },
    {   18487,   27474,  -14009,    8617,   16533,  -26877,   -8801,   -8430,  -23762 },
    {   18487,  -13275,  -18012,  -22906,   21234,   16697,   -1050,   28811,   12167 },
    {   18487,   -4993,  -22014,   22711,   -7919,    7676,    8639,  -34913,   17138 },
    {   18487,   15271,  -26017,  -10736,  -11449,  -27744,   20266,   19505,   -4118 },
    {   18487,  -11050,  -30020,   -1432,    1105,   23165,   33830,    3002,   -4192 },
};

[numthreads(HF_GI_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int probeCount = gParams[0].cfg.x;
    uint p = gid.x;
    if ((int)p >= probeCount) return;

    // int64 Q32.32 accumulator (acc[basis][channel]) — STAY WIDE (the crux), narrow ONCE at the end.
    int64_t acc[9][3];
    [unroll] for (int i = 0; i < 9; ++i) { acc[i][0] = 0; acc[i][1] = 0; acc[i][2] = 0; }

    for (int r = 0; r < HF_GI_RAYS; ++r) {
        GiRad rad = gRadiance[p * HF_GI_RAYS + r];
        int64_t cr = (int64_t)rad.r, cg = (int64_t)rad.g, cb = (int64_t)rad.b;
        [unroll] for (int i2 = 0; i2 < 9; ++i2) {
            int64_t b = (int64_t)kGiSHBasis[r][i2];   // Q16.16 basis weight
            acc[i2][0] += cr * b;                     // Q32.32 term — WIDE, no narrow yet
            acc[i2][1] += cg * b;
            acc[i2][2] += cb * b;
        }
    }

    FxProbeSH sh;
    [unroll] for (int j = 0; j < 28; ++j) sh.c[j] = 0;
    [unroll] for (int k = 0; k < 9; ++k) {
        // Quantize ONCE: average over N (Q32.32 / int stays Q32.32) then arithmetic narrow >> kFrac -> Q16.16.
        sh.c[k * 3 + 0] = (int)((acc[k][0] / (int64_t)HF_GI_RAYS) >> HF_GI_FRAC);
        sh.c[k * 3 + 1] = (int)((acc[k][1] / (int64_t)HF_GI_RAYS) >> HF_GI_FRAC);
        sh.c[k * 3 + 2] = (int)((acc[k][2] / (int64_t)HF_GI_RAYS) >> HF_GI_FRAC);
    }
    gSH[p] = sh;
}
