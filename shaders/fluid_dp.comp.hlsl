// Slice FL4 — Deterministic GPU Fluid: the PBF DENSITY-CONSTRAINT Δp pass (the incompressibility solver's
// position correction; the 4th slice of FLAGSHIP #9). ONE thread per PARTICLE i (i < particleCount) — the
// JACOBI per-particle-independent pattern (NOT the cloth CL3 single-thread): each thread reads its
// neighbours' ITERATION-START positions (read-only) + the just-computed λ array (read-only) and writes only
// its OWN dp[i] (a SEPARATE buffer from gParticles — the Jacobi double-buffer), so there is NO race and the
// result is DETERMINISTIC regardless of thread order. NO atomics, NO single-thread, NO TDR ceiling (the FL4
// design win over the cloth's order-dependent Gauss-Seidel).
//
// The body is copied VERBATIM from fluid.h::SolveDensityConstraint:
//   Δp_i = (1/ρ0) Σ_{j∈neighbors(i)} (λ_i + λ_j) gradW[bin(r_ij²)] · unit(p_j − p_i)
// where gradW is the spiky-gradient MAGNITUDE LUT (>= 0) and the unit(p_j − p_i) carries the spiky-gradient
// sign (the kernel decreases with r so ∇W points along (p_j − p_i)) — an OVER-dense pair (λ_i+λ_j < 0) is
// pushed APART. STATIC particles (flags & STATIC bit) -> Δp = 0. A divergence vs the header is exactly what
// the host's GPU==CPU memcmp (the settled particle array) catches.
//
// INTEGER WIDTH: fxmul/fxdiv/FxISqrt + r² use int64_t (the fluid_lambda.comp / cloth_solve.comp lesson). DXC
// compiles int64 (Vulkan); glslc CANNOT parse int64 in HLSL -> this shader is VULKAN-SPIR-V-ONLY (NOT in
// metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --fluid-solve showcase runs the CPU
// fluid::StepFluid (byte-identical by construction).
//
// Buffers (storage, bound at compute bindings 0..7; Vulkan-only):
//   b0 gParticles    : the Q16.16 FluidParticle array, READ (pos, flags).
//   b1 neighborStart : the FL2 neighbor-list prefix-sum (particleCount+1), READ.
//   b2 neighbors     : the FL2 candidate neighbor j indices grouped by i, READ.
//   b3 kernelGradW   : the host-snapped Q16.16 spiky |∇W| LUT gradW[bins], READ.
//   b4 lambda        : the FL3 λ pass output λ_i (Q16.16), READ.
//   b5 dp            : one FxVec3Gpu (3 x int32) per particle (the Jacobi correction Δp_i), WRITE.
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

struct FxVec3Gpu { int x, y, z; };   // the Δp output (std430 12 bytes; the FxVec3 mirror)

struct FluidKernelParams {
    int4 ker;   // x=h, y=restDensity, z=epsilon, w=bins
    int4 cfg;   // x=particleCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle>     gParticles    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>              neighborStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>              neighbors     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<int>               kernelGradW   : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<int>               lambda        : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<FxVec3Gpu>         dp            : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<FluidKernelParams> gParams       : register(u6);

static const uint HF_FLUID_STATIC = 1u;   // == fluid.h::kFlagStatic (bit0)

// fxmul/fxdiv — VERBATIM fpx.h::fxmul / fpx.h::fxdiv (int64 intermediate).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_FLUID_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_FLUID_FRAC) / (int64_t)b); }

// FxISqrt — VERBATIM fpx.h::FxISqrt (int64 binary integer sqrt).
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
    int bins          = gParams[0].ker.w;
    int particleCount = gParams[0].cfg.x;
    int enabled       = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    FxVec3Gpu zero; zero.x = 0; zero.y = 0; zero.z = 0;
    if (enabled == 0) { dp[i] = zero; return; }

    FluidParticle pi = gParticles[i];
    if (pi.flags & HF_FLUID_STATIC) { dp[i] = zero; return; }   // static -> Δp = 0 (the pinned case)

    int64_t h2 = (int64_t)h * (int64_t)h;
    int lami = lambda[i];
    int ax = 0, ay = 0, az = 0;   // Σ_j (λ_i+λ_j)·∇W (pre 1/ρ0 scale)
    uint s0 = neighborStart[i], s1 = neighborStart[i + 1u];
    for (uint s = s0; s < s1; ++s) {
        uint j = neighbors[s];
        FluidParticle pj = gParticles[j];
        int bin = BinOf(RadiusSq(pi, pj), h2, bins);
        if (bin >= bins) continue;                              // r >= h -> zero gradient
        int scale = fxmul(lami + lambda[j], kernelGradW[bin]);
        // dir = unit(p_j − p_i) (the spiky-gradient sign: ∇W points along (p_j − p_i)).
        int dx = pj.px - pi.px, dy = pj.py - pi.py, dz = pj.pz - pi.pz;
        int len = FxLength3(dx, dy, dz);
        int nx, ny, nz;
        if (len == 0) { nx = 0; ny = HF_FLUID_ONE; nz = 0; }    // FxNormalize +Y fallback
        else { nx = fxdiv(dx, len); ny = fxdiv(dy, len); nz = fxdiv(dz, len); }
        ax += fxmul(scale, nx); ay += fxmul(scale, ny); az += fxmul(scale, nz);
    }
    // Δp_i = accum / ρ0 (per axis fxdiv — the SAME truncating divide as the gradient scale).
    FxVec3Gpu d; d.x = fxdiv(ax, rho0); d.y = fxdiv(ay, rho0); d.z = fxdiv(az, rho0);
    dp[i] = d;                                                  // per-particle independent write (NO atomics)
}
