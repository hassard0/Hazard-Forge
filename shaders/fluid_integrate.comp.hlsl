// Slice FL1 — Deterministic GPU Fluid: the Q16.16 PARTICLE POOL INTEGRATOR compute pass (the BEACHHEAD
// of FLAGSHIP #9: DETERMINISTIC GPU FLUID via Position-Based Fluids). ONE thread per particle
// (i < particleCount). Each thread runs `steps` iterations of the semi-implicit-Euler integrator (vel +=
// gravity*dt; prev = pos; pos += vel*dt; ground floor-clamp) on its OWN particle, the fxmul + integrate +
// prev-snap + floor-clamp copied VERBATIM from engine/sim/fluid.h::IntegrateFluidParticle, and writes
// gParticles[i] back. In FL1 each particle is INDEPENDENT (no neighbours/density/constraints until
// FL2-FL4), so the per-thread write is order-independent — race-free, NO atomics, bit-identical GPU==CPU
// + cross-backend (the CL1 cloth_integrate / FPX1 fpx_integrate argument applied to a fluid pool).
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it
// consumes the host-snapped Q16.16 int FluidParticle array and runs the SAME pure-integer fxmul
// ((int64)a*b >> 16, an ARITHMETIC right shift on int64) + integer add + integer compare the header runs.
// A divergence here vs engine/sim/fluid.h is exactly what the host's GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux, the CL1/FPX1 lesson): fxmul uses int64_t — IDENTICAL to
// fluid.h::fxmul / cloth.h::fxmul / fpx.h::fxmul. The fluid integrate has the SAME `vel += gravity*dt;
// pos += vel*dt` form as cloth_integrate.comp / fpx_integrate.comp, whose (int64)a*b product before the
// >>16 shift exceeds int32 for Q16.16 gravity*dt (gravity ≈ -9.8*65536; products of two Q16.16 world-
// scale values blow past 2^31). To stay bit-exact to that int64-intermediate reference WITHOUT any
// overflow fragility, this shader uses int64. HLSL SM6 supports int64_t (the swraster.comp::SwEdge /
// cloth_integrate.comp pattern — DXC -spirv with the Int64 capability). glslc (the Metal HLSL->SPIR-V->MSL
// frontend) CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY (NOT in
// metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --fluid-integrate showcase runs the CPU
// fluid::IntegrateFluid (byte-identical by construction).
//
// integrateEnabled push/param flag: 0 -> write the input particle back UNCHANGED (the disabled-path
// no-op; gParticles stays byte-identical to the upload).
//
// Buffers (storage, bound at compute bindings 0..1; on Metal these would land at buffer(0..1) — but this
// shader is Vulkan-only):
//   b0 gParticles : the Q16.16 FluidParticle array (pos.xyz, prev.xyz, vel.xyz, invMass, flags — std430
//                   ints, 44 bytes), READ+WRITE.
//   b1 gParams    : { gravity.xyz, dt, groundY, particleCount, steps, integrateEnabled }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// cloth_integrate.comp / fpx_integrate.comp / mc_classify.comp), not backend CODE symbols.

#define HF_FLUID_THREADS 64
#define HF_FLUID_FRAC 16     // MUST match fluid.h::kFrac (== fpx.h::kFrac)
#define HF_FLUID_FLAG_STATIC 1u

// std430 FluidParticle mirror (engine/sim/fluid.h::FluidParticle): pos.xyz, prev.xyz, vel.xyz, invMass
// (int), flags (uint). 11 x 4-byte = 44 bytes, no padding holes (memcmp-able).
struct FluidParticle {
    int  px, py, pz;     // Q16.16 current position
    int  prx, pry, prz;  // Q16.16 previous position
    int  vx, vy, vz;     // Q16.16 velocity
    int  invMass;        // Q16.16 inverse mass (unused by the FL1 integrate; carried for FL4)
    uint flags;          // bit0 = STATIC
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt   (all Q16.16)
//   cfg  : x=groundY (Q16.16), y=particleCount, z=steps, w=integrateEnabled
struct FluidParams {
    int4 grav;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle> gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FluidParams>   gParams    : register(u1);

// VERBATIM fluid.h::fxmul / cloth.h::fxmul / fpx.h::fxmul — (a*b) >> kFrac with an int64 intermediate
// (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_FLUID_FRAC);
}

[numthreads(HF_FLUID_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int gravx = gParams[0].grav.x;
    int gravy = gParams[0].grav.y;
    int gravz = gParams[0].grav.z;
    int dt    = gParams[0].grav.w;
    int groundY          = gParams[0].cfg.x;
    int particleCount    = gParams[0].cfg.y;
    int steps            = gParams[0].cfg.z;
    int integrateEnabled = gParams[0].cfg.w;

    uint i = gid.x;
    if ((int)i >= particleCount) return;

    FluidParticle p = gParticles[i];

    // Disabled -> write the input back UNCHANGED (the byte-identical no-op proof).
    if (integrateEnabled == 0) { gParticles[i] = p; return; }

    // Run `steps` iterations of IntegrateFluidParticle's per-particle math (VERBATIM
    // fluid.h::IntegrateFluidParticle).
    for (int s = 0; s < steps; ++s) {
        if ((p.flags & HF_FLUID_FLAG_STATIC) == 0u) {
            // (1) integrate velocity: vel += gravity * dt.
            p.vx += fxmul(gravx, dt);
            p.vy += fxmul(gravy, dt);
            p.vz += fxmul(gravz, dt);
            // (2) snapshot the previous position (the PBF predicted-position anchor FL3/FL4 reads).
            p.prx = p.px;
            p.pry = p.py;
            p.prz = p.pz;
            // (3) integrate position: pos += vel * dt.
            p.px += fxmul(p.vx, dt);
            p.py += fxmul(p.vy, dt);
            p.pz += fxmul(p.vz, dt);
            // (4) ground floor clamp (no restitution — FL1 is free-fall + ground only).
            if (p.py < groundY) {
                p.py = groundY;
                if (p.vy < 0) p.vy = 0;
            }
        }
    }

    gParticles[i] = p;
}
