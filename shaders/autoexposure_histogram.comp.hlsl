// Slice CW — Auto-Exposure: the HISTOGRAM compute pass. ONE thread per scene pixel. Each thread reads
// the HDR scene color from the flat scene SSBO (float4 per pixel, row-major), computes its Rec.709
// Luminance, maps it to a log2-luminance BIN via LumToBin, and InterlockedAdd(gHistogram[bin], 1) into
// the `bins`-entry INTEGER histogram SSBO (cleared to 0 by the host before this pass). The result is the
// per-bin pixel COUNT of the scene's luminance distribution.
//
// WHY DETERMINISTIC (the integer histogram): each pixel does an atomicAdd of the integer 1. Integer
// addition is commutative + associative, so the final per-bin count is the number of pixels in that
// luminance range — INDEPENDENT of the atomic execution order -> bit-deterministic. No float atomics
// (which would be order-nondeterministic). The reduce + exposure downstream are pure functions of these
// (deterministic) counts. Two runs byte-identical.
//
// The math (Luminance Rec.709 + LumToBin log2 binning) is the VERBATIM mirror of
// engine/render/auto_exposure.h, so tests/auto_exposure_test.cpp exercises the EXACT mapping this pass
// runs. A mismatch would shift the metered average -> a different exposure -> the adaptation-off no-op
// proof (which depends only on the reduce writing E0, not on these counts) still holds, but the golden
// auto-exposed image would drift; the unit test pins the binning so it cannot.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gScene     : the flat HDR scene color (float4 per pixel, row-major width*height), READ.
//   b1 gHistogram : the `bins`-entry INTEGER histogram, atomicAdd WRITE (cleared to 0 by the host).
//   b2 gParams    : { width, height, bins, _ } + { minLogLum, logLumRange, keyValue, E0 }, READ.
//
// SEAM DISCIPLINE: this shader is ABOVE the RHI seam; the only mentions of vk/MSL are the [[vk::binding]]
// decorations (same as cluster_assign / froxel_inject), not backend CODE symbols. spirv-cross maps the
// SPIR-V bindings to Metal's flat compute buffer indices so the SAME HLSL feeds Vulkan (DXC) and Metal
// (glslang->spirv-cross): bit-identical math + bit-identical integer atomics.

#define HF_AE_THREADS 64

struct ScenePixel { float4 color; };   // xyz = linear HDR color, w unused (std430 float4)

// Auto-exposure params (std430): dims(width,height,bins,_) + lum(minLogLum,logLumRange,keyValue,E0) +
// flags(adaptationEnabled,_,_,_).
struct AeParams {
    uint4  dims;    // x=width, y=height, z=bins, w=adaptationEnabled (read by the reduce pass)
    float4 lum;     // x=minLogLum, y=logLumRange, z=keyValue, w=E0 (the fixed reference exposure)
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ScenePixel> gScene     : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>       gHistogram : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<AeParams>   gParams    : register(u2);

// Rec.709 relative luminance. Mirrors auto_exposure.h::Luminance EXACTLY.
float Luminance(float3 rgb) {
    return 0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z;
}

// Log2-luminance bin in [0,bins-1]. Mirrors auto_exposure.h::LumToBin EXACTLY (lum<=0 / below floor ->
// bin 0; above ceiling -> top bin).
int LumToBin(float lum, float minLogLum, float logLumRange, int bins) {
    if (bins < 1) bins = 1;
    if (lum <= 0.0) return 0;
    float logLum = log2(lum);
    float t = (logLum - minLogLum) / logLumRange;
    if (t < 0.0) return 0;
    if (t > 1.0) t = 1.0;
    int bin = (int)floor(t * (float)bins);
    if (bin < 0) bin = 0;
    if (bin > bins - 1) bin = bins - 1;
    return bin;
}

[numthreads(HF_AE_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint width  = gParams[0].dims.x;
    uint height = gParams[0].dims.y;
    int  bins   = (int)gParams[0].dims.z;
    float minLogLum   = gParams[0].lum.x;
    float logLumRange = gParams[0].lum.y;

    uint pixelCount = width * height;
    uint p = gid.x;
    if (p >= pixelCount) return;

    float3 color = gScene[p].color.rgb;
    float lum = Luminance(color);
    int bin = LumToBin(lum, minLogLum, logLumRange, bins);

    // INTEGER atomic add of 1 -> order-independent count -> deterministic.
    InterlockedAdd(gHistogram[bin], 1u);
}
