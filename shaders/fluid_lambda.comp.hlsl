// Slice FL3 — Deterministic GPU Fluid: the PBF λ (constraint scaling factor) compute pass (the 2nd half of
// the make-or-break; the 3rd slice of FLAGSHIP #9). ONE thread per PARTICLE i (i < particleCount). The
// thread computes the unilateral density constraint C_i = fxdiv(ρ_i, ρ0) − 1 (ρ_i from the FL3 density
// pass); if C_i < 0 (under-dense surface — fluid doesn't PULL together) it writes λ_i = 0 (the unilateral
// clamp, a fixed integer compare); else it sums Σ_k |∇_k C_i|² over the FL2 neighbours (each ∇_j C_i =
// (gradW[bin]/ρ0)·unit(p_i−p_j), plus the self gradient ∇_i C_i = −Σ_j ∇_j C_i) and writes
// λ_i = fxdiv(−C_i, Σgrad² + ε). NO atomics (per-particle independent — the FL1 pattern). enabled=0 -> 0.
// The kernel LUT (gradW) is HOST-SNAPPED ONCE (fluid.h::BuildKernelTable) + uploaded.
//
// WHY BIT-IDENTICAL to the CPU fluid.h::ComputeLambda (the make-or-break): the body below is copied
// VERBATIM from ComputeLambda — fxdiv (the int64 Q16.16 truncating divide), fxmul (the int64 multiply),
// RadiusSq/BinOf (int64), and FxNormalize (via the int64 FxISqrt length, NO in-shader float sqrt) are the
// SAME ops over the SAME fixed-order FL2 neighbour list. A divergence vs the header is exactly what the
// host's GPU==CPU memcmp (lambda) catches.
//
// INTEGER WIDTH: fxdiv/fxmul/FxISqrt + r² use int64_t (the cloth_solve.comp / fpx_solve.comp lesson). DXC
// compiles int64 (Vulkan); glslc CANNOT parse int64 in HLSL -> this shader is VULKAN-SPIR-V-ONLY (NOT in
// metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --fluid-density showcase runs the CPU
// fluid::ComputeLambda (byte-identical by construction).
//
// Buffers (storage, bound at compute bindings 0..7; Vulkan-only):
//   b0 gParticles    : the Q16.16 FluidParticle array, READ (pos).
//   b1 neighborStart : the FL2 neighbor-list prefix-sum (particleCount+1), READ.
//   b2 neighbors     : the FL2 candidate neighbor j indices grouped by i, READ.
//   b3 kernelGradW   : the host-snapped Q16.16 spiky |∇W| LUT gradW[bins], READ.
//   b4 density       : the FL3 density pass output ρ_i (Q16.16), READ.
//   b5 lambda        : one Q16.16 fx per particle (the scaling factor λ_i), WRITE.
//   b6 gParams       : the FluidKernelParams (h, restDensity, epsilon, bins, particleCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_FLUID_THREADS 64
#define HF_FLUID_FRAC 16
#define HF_FLUID_ONE 65536      // kOne == 1<<16

struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

struct FluidKernelParams {
    int4 ker;   // x=h, y=restDensity, z=epsilon, w=bins
    int4 cfg;   // x=particleCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle>     gParticles    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>              neighborStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>              neighbors     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<int>               kernelGradW   : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<int>               density       : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<int>               lambda        : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<FluidKernelParams> gParams       : register(u6);

// fxmul/fxdiv — VERBATIM fpx.h::fxmul / fpx.h::fxdiv (int64 intermediate).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_FLUID_FRAC);
}
int fxdiv(int a, int b) {
    if (b == 0) return 0;
    return (int)(((int64_t)a << HF_FLUID_FRAC) / (int64_t)b);
}

// FxISqrt — VERBATIM fpx.h::FxISqrt (int64 binary integer sqrt). Build-time on the CPU; here it powers the
// per-step FxNormalize direction (the SAME deterministic integer sqrt, NOT a float sqrt).
int64_t FxISqrt(int64_t v) {
    if (v <= 0) return 0;
    int64_t bit = (int64_t)1 << 62;
    while (bit > v) bit >>= 2;
    int64_t res = 0;
    while (bit != 0) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else { res >>= 1; }
        bit >>= 2;
    }
    return res;
}

// FxLength/FxNormalize — VERBATIM fpx.h. Q16.16 length = FxISqrt of the int64 Q32.32 sum-of-squares;
// normalize divides each axis by the length (len==0 -> the fixed +Y fallback, the fpx convention).
int FxLength3(int x, int y, int z) {
    int64_t sx = (int64_t)x * (int64_t)x;
    int64_t sy = (int64_t)y * (int64_t)y;
    int64_t sz = (int64_t)z * (int64_t)z;
    return (int)FxISqrt(sx + sy + sz);
}

// RadiusSq/BinOf — VERBATIM fluid.h (int64).
int64_t RadiusSq(FluidParticle a, FluidParticle b) {
    int64_t dx = (int64_t)a.px - (int64_t)b.px;
    int64_t dy = (int64_t)a.py - (int64_t)b.py;
    int64_t dz = (int64_t)a.pz - (int64_t)b.pz;
    return dx * dx + dy * dy + dz * dz;
}
int BinOf(int64_t r2, int64_t h2, int bins) {
    if (r2 >= h2) return bins;
    return (int)((r2 * (int64_t)bins) / h2);
}

[numthreads(HF_FLUID_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int h             = gParams[0].ker.x;
    int rho0          = gParams[0].ker.y;
    int eps           = gParams[0].ker.z;
    int bins          = gParams[0].ker.w;
    int particleCount = gParams[0].cfg.x;
    int enabled       = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    if (enabled == 0) { lambda[i] = 0; return; }

    int64_t h2 = (int64_t)h * (int64_t)h;

    // C_i = ρ_i / ρ0 − 1 (Q16.16).
    int Ci = fxdiv(density[i], rho0) - HF_FLUID_ONE;
    // Unilateral: under-dense (C_i < 0) -> λ = 0 (no pulling-together correction).
    if (Ci < 0) { lambda[i] = 0; return; }

    FluidParticle pi = gParticles[i];
    int gsx = 0, gsy = 0, gsz = 0;   // the self gradient ∇_i C_i = −Σ_j ∇_j C_i
    int sumGrad2 = 0;                // Σ_j |∇_j C_i|²
    uint s0 = neighborStart[i], s1 = neighborStart[i + 1u];
    for (uint s = s0; s < s1; ++s) {
        uint j = neighbors[s];
        FluidParticle pj = gParticles[j];
        int bin = BinOf(RadiusSq(pi, pj), h2, bins);
        if (bin >= bins) continue;                    // r >= h -> zero gradient
        int gradWScaled = fxdiv(kernelGradW[bin], rho0);
        // dir = unit(p_i − p_j) (FxNormalize: len==0 -> +Y fallback).
        int dx = pi.px - pj.px, dy = pi.py - pj.py, dz = pi.pz - pj.pz;
        int len = FxLength3(dx, dy, dz);
        int nx, ny, nz;
        if (len == 0) { nx = 0; ny = HF_FLUID_ONE; nz = 0; }
        else { nx = fxdiv(dx, len); ny = fxdiv(dy, len); nz = fxdiv(dz, len); }
        int gx = fxmul(gradWScaled, nx), gy = fxmul(gradWScaled, ny), gz = fxmul(gradWScaled, nz);
        sumGrad2 += fxmul(gx, gx) + fxmul(gy, gy) + fxmul(gz, gz);
        gsx -= gx; gsy -= gy; gsz -= gz;
    }
    // |∇_i C_i|² joins the sum.
    sumGrad2 += fxmul(gsx, gsx) + fxmul(gsy, gsy) + fxmul(gsz, gsz);
    // λ_i = −C_i / (Σgrad² + ε).
    lambda[i] = fxdiv(-Ci, sumGrad2 + eps);           // per-particle independent write (NO atomics)
}
