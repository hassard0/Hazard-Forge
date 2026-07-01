// Slice FL7 — Deterministic GPU Fluid: the XSPH VISCOSITY pass 2 (the (vel, prev) re-encode; the Track-R
// R2 refinement of FLAGSHIP #9). ONE thread per PARTICLE i, dispatched ONLY AFTER fluid_visc.comp computed
// EVERY smoothed velocity v'_i into vout (a ComputeToComputeBarrier between — the Jacobi two-pass split, so
// no particle's prev is read by pass 1 after pass 2 starts). Per-particle independent (each thread reads
// vout[i] + writes only its OWN particle) -> NO race, NO atomics, deterministic regardless of thread order.
//
// The body is copied VERBATIM from fluid.h::ApplyXsphVelocity: for each NON-static particle,
//   vel  = v'_i
//   prev = pos − v'_i·dt        (per-axis fxmul; pos UNTOUCHED)
// so the particle state STAYS the (pos, prev, vel) triple (the FL5 snapshot/lockstep machinery applies
// unchanged) and vel ≈ (pos − prev)/dt is re-established. STATIC particles are untouched.
//
// INTEGER WIDTH: fxmul uses int64_t -> VULKAN-SPIR-V-ONLY (NOT in hf_gen_msl); the Metal --fl7-visc
// showcase runs the CPU fluid::StepFluidViscSteps (byte-identical by construction, the FL4 convention).
//
// Buffers (storage, Vulkan-only):
//   b0 gParticles : the Q16.16 FluidParticle array, READ+WRITE (vel, prev).
//   b1 vout       : the fluid_visc.comp smoothed velocities v'_i (3 x int32 per particle), READ.
//   b2 gParams    : the FluidViscParams (h, c, dt, bins, particleCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_FLUID_THREADS 64
#define HF_FLUID_FRAC 16

struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

struct FxVec3Gpu { int x, y, z; };

// FluidViscParams (std430). ker {h, c, dt, bins}, cfg {particleCount, enabled}.
struct FluidViscParams {
    int4 ker;   // x=h, y=c (viscosity coefficient), z=dt, w=bins
    int4 cfg;   // x=particleCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle>   gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FxVec3Gpu>       vout       : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FluidViscParams> gParams    : register(u2);

static const uint HF_FLUID_STATIC = 1u;   // == fluid.h::kFlagStatic (bit0)

// fxmul — VERBATIM fpx.h::fxmul (int64 intermediate).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_FLUID_FRAC); }

[numthreads(HF_FLUID_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int dt            = gParams[0].ker.z;
    int particleCount = gParams[0].cfg.x;
    int enabled       = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    if (enabled == 0) return;

    FluidParticle p = gParticles[i];
    if (p.flags & HF_FLUID_STATIC) return;   // static particles are untouched (they hold)

    FxVec3Gpu v = vout[i];
    p.vx = v.x; p.vy = v.y; p.vz = v.z;                     // vel = v'_i
    p.prx = p.px - fxmul(v.x, dt);                          // prev = pos − v'·dt (pos untouched)
    p.pry = p.py - fxmul(v.y, dt);
    p.prz = p.pz - fxmul(v.z, dt);

    gParticles[i] = p;                                      // per-particle independent write (NO atomics)
}
