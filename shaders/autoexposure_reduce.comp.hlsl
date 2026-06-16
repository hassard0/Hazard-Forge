// Slice CW — Auto-Exposure: the REDUCE compute pass. ONE thread reduces the (deterministic) integer
// histogram into the weighted AverageLuminance (excluding bin 0 = black, the standard), computes the
// target exposure via the key-value ExposureFromAverage, and writes the single `exposure` float to the
// 1-entry exposure SSBO. The tonemap_autoexp fragment pass then reads that exposure and applies it.
//
// THE adaptationEnabled GATE (the no-op proof's foundation): when adaptationEnabled == 0 the pass writes
// the FIXED reference exposure E0 (the engine's existing default tonemap exposure) instead of the
// histogram-derived one. So with adaptation OFF the tonemap_autoexp applies EXACTLY E0 -> its output is
// byte-identical to the standard fixed-exposure post.frag (which also multiplies by E0). The showcase
// asserts this byte-equality (no constant bias, no exposure drift), proving the histogram/reduce/apply
// plumbing is a pure pass-through when adaptation is off.
//
// The math (BinToLum + AverageLuminance black-bin exclusion + the key-value ExposureFromAverage) is the
// VERBATIM mirror of engine/render/auto_exposure.h, so tests/auto_exposure_test.cpp exercises the EXACT
// reduce this pass runs. Single-frame INSTANT adaptation (no temporal exposure history) -> the golden is
// stable + two runs byte-identical.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gHistogram : the `bins`-entry INTEGER histogram the histogram pass filled, READ.
//   b1 gExposure  : the 1-entry exposure float, WRITE (the tonemap_autoexp reads it via BindLightClusters).
//   b2 gParams    : { width,height,bins,adaptationEnabled } + { minLogLum,logLumRange,keyValue,E0 }, READ.
//
// SEAM DISCIPLINE: above the RHI seam; the only vk mention is the [[vk::binding]] decoration. spirv-cross
// maps the bindings cross-backend -> bit-identical reduce on Vulkan + Metal.

struct AeParams {
    uint4  dims;    // x=width, y=height, z=bins, w=adaptationEnabled
    float4 lum;     // x=minLogLum, y=logLumRange, z=keyValue, w=E0
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>     gHistogram : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<float>    gExposure  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<AeParams> gParams    : register(u2);

// Bin center -> representative luminance. Mirrors auto_exposure.h::BinToLum EXACTLY.
float BinToLum(int bin, float minLogLum, float logLumRange, int bins) {
    if (bins < 1) bins = 1;
    if (bin < 0) bin = 0;
    if (bin > bins - 1) bin = bins - 1;
    float t = ((float)bin + 0.5) / (float)bins;
    float logLum = minLogLum + t * logLumRange;
    return exp2(logLum);
}

// Key-value exposure. Mirrors auto_exposure.h::ExposureFromAverage EXACTLY (eps-floored divisor).
float ExposureFromAverage(float avgLum, float keyValue) {
    float eps = 1e-4;
    float denom = (avgLum > eps) ? avgLum : eps;
    return keyValue / denom;
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0) return;   // a single reducer thread (the histogram is tiny: `bins` entries)

    int   bins        = (int)gParams[0].dims.z;
    uint  adaptation  = gParams[0].dims.w;
    float minLogLum   = gParams[0].lum.x;
    float logLumRange = gParams[0].lum.y;
    float keyValue    = gParams[0].lum.z;
    float E0          = gParams[0].lum.w;

    if (adaptation == 0u) {
        // ADAPTATION OFF: write the FIXED reference exposure E0 -> tonemap_autoexp(E0) == the standard
        // fixed-exposure tonemap (the byte-identical no-op proof). The histogram is ignored here.
        gExposure[0] = E0;
        return;
    }

    // Weighted average luminance, EXCLUDING bin 0 (black). Mirrors auto_exposure.h::AverageLuminance.
    float weightedSum = 0.0;   // Sum binCenterLum * count over bins 1..bins-1
    float countSum    = 0.0;   // Sum count        over bins 1..bins-1 (the contributing pixels)
    [loop] for (int b = 1; b < bins; ++b) {
        float c = (float)gHistogram[b];
        if (c <= 0.0) continue;
        weightedSum += BinToLum(b, minLogLum, logLumRange, bins) * c;
        countSum    += c;
    }

    float avgLum;
    if (countSum <= 0.0) {
        avgLum = BinToLum(0, minLogLum, logLumRange, bins);   // all-black floor (never divide-by-zero)
    } else {
        avgLum = weightedSum / countSum;
    }

    gExposure[0] = ExposureFromAverage(avgLum, keyValue);
}
