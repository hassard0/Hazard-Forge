// Slice FL4 — Deterministic GPU Fluid: the JACOBI Δp APPLY pass (the 4th slice of FLAGSHIP #9). ONE thread
// per PARTICLE i: p_i.pos += dp_i (the Jacobi double-buffer apply — reads the Δp_i computed by fluid_dp.comp
// from the iteration-start positions, writes p_i.pos). Per-particle independent (each thread writes only its
// OWN particle) -> NO race, NO atomics, deterministic regardless of thread order. STATIC particles
// (flags & STATIC bit) are untouched. Copied VERBATIM from fluid.h::StepFluid's "p_i += dp_i" apply step.
//
// Pure int32 add (no int64), but kept VULKAN-SPIR-V-ONLY alongside fluid_dp.comp / fluid_collide.comp (the
// FL4 solve passes are driven as a host sequence on Vulkan; the Metal --fluid-solve showcase runs the CPU
// fluid::StepFluid — byte-identical by construction — so none of the solve passes are in hf_gen_msl).
//
// Buffers (storage, Vulkan-only):
//   b0 gParticles : the Q16.16 FluidParticle array, READ+WRITE (pos += dp; flags read).
//   b1 dp         : one FxVec3Gpu (3 x int32) per particle (the Jacobi correction Δp_i), READ.
//   b2 gParams    : the FluidKernelParams (..., particleCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_FLUID_THREADS 64

struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

struct FxVec3Gpu { int x, y, z; };

struct FluidKernelParams {
    int4 ker;   // x=h, y=restDensity, z=epsilon, w=bins
    int4 cfg;   // x=particleCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle>     gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FxVec3Gpu>         dp         : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FluidKernelParams> gParams    : register(u2);

static const uint HF_FLUID_STATIC = 1u;   // == fluid.h::kFlagStatic (bit0)

[numthreads(HF_FLUID_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int particleCount = gParams[0].cfg.x;
    int enabled       = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    if (enabled == 0) return;

    FluidParticle p = gParticles[i];
    if (p.flags & HF_FLUID_STATIC) return;   // static -> never moves
    FxVec3Gpu d = dp[i];
    p.px += d.x; p.py += d.y; p.pz += d.z;
    gParticles[i] = p;                       // per-particle independent write (NO atomics)
}
