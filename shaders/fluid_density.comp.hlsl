// Slice FL3 — Deterministic GPU Fluid: the PBF DENSITY gather compute pass (the make-or-break, the only
// genuinely fluid-specific computation; the 3rd slice of FLAGSHIP #9). ONE thread per PARTICLE i
// (i < particleCount). The thread starts from the self-density W[0] (the r=0 kernel peak) and gathers
// W[bin(r_ij²)] over its FL2 candidate neighbours (neighborStart/neighbors CSR), where r_ij² is the int64
// squared Q16.16 distance and bin = floor((r²*bins)/h²) (a neighbour with r >= h -> bin >= bins -> the
// deferred FL2 radial cull lands HERE -> no contribution) -> density[i]. NO atomics (each thread writes
// its OWN density[i], a per-particle independent gather — the FL1 per-particle pattern, NOT CL3
// single-thread). enabled=0 -> write 0. The kernel LUT (W) is HOST-SNAPPED ONCE (fluid.h::BuildKernelTable)
// + uploaded -> the shader does ZERO float / sqrt; it only INDEXES the table + accumulates integers.
//
// WHY BIT-IDENTICAL to the CPU fluid.h::ComputeDensity (the make-or-break): the body below is copied
// VERBATIM from ComputeDensity — RadiusSq is the int64 sum of per-axis (a-b)² (bounded positions overflow
// int32 for dx², so int64 is REQUIRED), BinOf is the int64 (r²*bins)/h² truncating divide, and the gather
// is a fixed-order sum over the SAME FL2 neighbour list. A divergence vs the header is exactly what the
// host's GPU==CPU memcmp (density) catches.
//
// INTEGER WIDTH (the determinism crux, the FL1 lesson): r² (dx² over Q16.16 world-scale) + the (r²*bins)
// binning exceed int32, so this shader uses int64_t. HLSL SM6 supports int64_t (the fluid_integrate.comp /
// cloth_solve.comp pattern — DXC -spirv with the Int64 capability). glslc (the Metal HLSL->SPIR-V->MSL
// frontend) CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY (NOT in
// metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --fluid-density showcase runs the CPU
// fluid::ComputeDensity (byte-identical by construction, the fluid_integrate.comp/cloth_solve.comp
// convention).
//
// Buffers (storage, bound at compute bindings 0..5; on Metal these would land at buffer(0..5) — but this
// shader is Vulkan-only):
//   b0 gParticles    : the Q16.16 FluidParticle array (44-byte std430 ints), READ (pos).
//   b1 neighborStart : the FL2 neighbor-list exclusive prefix-sum (particleCount+1), READ.
//   b2 neighbors     : the FL2 candidate neighbor j indices grouped by i, READ.
//   b3 kernelW       : the host-snapped Q16.16 poly6 density-kernel LUT W[bins], READ.
//   b4 density       : one Q16.16 fx per particle (the gathered ρ_i), WRITE.
//   b5 gParams       : the FluidKernelParams (h, restDensity, epsilon, bins, particleCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_FLUID_THREADS 64
#define HF_FLUID_FRAC 16     // MUST match fluid.h::kFrac (== fpx.h::kFrac)

struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

// FluidKernelParams (std430). Mirrors the C++ upload struct.
//   ker : x=h (Q16.16), y=restDensity (Q16.16), z=epsilon (Q16.16), w=bins (B)
//   cfg : x=particleCount, y=enabled
struct FluidKernelParams {
    int4 ker;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle>     gParticles    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>              neighborStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>              neighbors     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<int>               kernelW       : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<int>               density       : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<FluidKernelParams> gParams       : register(u5);

// RadiusSq(a, b): |a-b|² as int64 — VERBATIM fluid.h::RadiusSq. int64 per axis (dx² overflows int32).
int64_t RadiusSq(FluidParticle a, FluidParticle b) {
    int64_t dx = (int64_t)a.px - (int64_t)b.px;
    int64_t dy = (int64_t)a.py - (int64_t)b.py;
    int64_t dz = (int64_t)a.pz - (int64_t)b.pz;
    return dx * dx + dy * dy + dz * dz;
}

// BinOf(r2, h2, bins): the LUT bin — VERBATIM fluid.h::BinOf. r2>=h2 -> bins (out of range). int64.
int BinOf(int64_t r2, int64_t h2, int bins) {
    if (r2 >= h2) return bins;
    return (int)((r2 * (int64_t)bins) / h2);
}

[numthreads(HF_FLUID_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int h             = gParams[0].ker.x;
    int bins          = gParams[0].ker.w;
    int particleCount = gParams[0].cfg.x;
    int enabled       = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    if (enabled == 0) { density[i] = 0; return; }

    int64_t h2 = (int64_t)h * (int64_t)h;
    int wSelf = (bins > 0) ? kernelW[0] : 0;

    FluidParticle pi = gParticles[i];
    int rho = wSelf;                                 // the self-density term (r = 0 -> W[0])
    uint s0 = neighborStart[i], s1 = neighborStart[i + 1u];
    for (uint s = s0; s < s1; ++s) {
        uint j = neighbors[s];
        FluidParticle pj = gParticles[j];
        int bin = BinOf(RadiusSq(pi, pj), h2, bins);
        if (bin < bins) rho += kernelW[bin];         // r >= h -> bin == bins -> no contribution
    }
    density[i] = rho;                                // per-particle independent write (NO atomics)
}
