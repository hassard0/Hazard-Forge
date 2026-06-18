// Slice GR1 — Deterministic GPU Granular/Sand: the Q16.16 GRAIN POOL INTEGRATOR compute pass (the BEACHHEAD
// of FLAGSHIP #10: DETERMINISTIC GPU GRANULAR / SAND via Position-Based granular dynamics). ONE thread per
// grain (i < grainCount). Each thread runs `steps` iterations of the semi-implicit-Euler integrator (vel +=
// gravity*dt; prev = pos; pos += vel*dt; RADIUS-AWARE ground rest) on its OWN grain, the fxmul + integrate +
// prev-snap + radius-aware floor-clamp copied VERBATIM from engine/sim/grain.h::IntegrateGrainParticle, and
// writes gGrains[i] back. In GR1 each grain is INDEPENDENT (no neighbours/contact/friction until GR2-GR4), so
// the per-thread write is order-independent — race-free, NO atomics, bit-identical GPU==CPU + cross-backend
// (the FL1 fluid_integrate / CL1 cloth_integrate / FPX1 fpx_integrate argument applied to a grain pool).
//
// TWO DELTAS vs fluid_integrate.comp the GR1 spec locks: (1) a first-class `radius` field -> 48-byte std430
// packing (12 x int32); (2) a RADIUS-AWARE ground rest — the grain's SURFACE rests on the floor (pos.y <
// groundY + radius -> groundY + radius), not its center. Everything else is fluid_integrate.comp verbatim.
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it consumes
// the host-snapped Q16.16 int GrainParticle array and runs the SAME pure-integer fxmul ((int64)a*b >> 16, an
// ARITHMETIC right shift on int64) + integer add + integer compare the header runs. A divergence here vs
// engine/sim/grain.h is exactly what the host's GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux, the FL1/CL1 lesson): fxmul uses int64_t — IDENTICAL to
// grain.h::fxmul / fluid.h::fxmul / fpx.h::fxmul. The grain integrate has the SAME `vel += gravity*dt;
// pos += vel*dt` form as fluid_integrate.comp / cloth_integrate.comp, whose (int64)a*b product before the
// >>16 shift exceeds int32 for Q16.16 gravity*dt (gravity ≈ -9.8*65536; products of two Q16.16 world-scale
// values blow past 2^31). To stay bit-exact to that int64-intermediate reference WITHOUT any overflow
// fragility, this shader uses int64. HLSL SM6 supports int64_t (the swraster.comp / fluid_integrate.comp
// pattern — DXC -spirv with the Int64 capability). glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse
// int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY (NOT in metal_headless/CMakeLists.txt hf_gen_msl);
// on Metal the --grain-integrate showcase runs the CPU grain::IntegrateGrains (byte-identical by construction).
//
// integrateEnabled push/param flag: 0 -> write the input grain back UNCHANGED (the disabled-path no-op;
// gGrains stays byte-identical to the upload).
//
// Buffers (storage, bound at compute bindings 0..1; on Metal these would land at buffer(0..1) — but this
// shader is Vulkan-only):
//   b0 gGrains : the Q16.16 GrainParticle array (pos.xyz, prev.xyz, vel.xyz, invMass, radius, flags — std430
//                ints, 48 bytes), READ+WRITE.
//   b1 gParams : { gravity.xyz, dt, groundY, grainCount, steps, integrateEnabled }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// fluid_integrate.comp / cloth_integrate.comp / fpx_integrate.comp / mc_classify.comp), not backend CODE
// symbols.

#define HF_GRAIN_THREADS 64
#define HF_GRAIN_FRAC 16     // MUST match grain.h::kFrac (== fpx.h::kFrac)
#define HF_GRAIN_FLAG_STATIC 1u

// std430 GrainParticle mirror (engine/sim/grain.h::GrainParticle): pos.xyz, prev.xyz, vel.xyz, invMass (int),
// radius (int), flags (uint). 12 x 4-byte = 48 bytes, no padding holes (memcmp-able). The ONE delta vs
// fluid_integrate.comp's FluidParticle is the `radius` field (44 -> 48 bytes).
struct GrainParticle {
    int  px, py, pz;     // Q16.16 current position
    int  prx, pry, prz;  // Q16.16 previous position
    int  vx, vy, vz;     // Q16.16 velocity
    int  invMass;        // Q16.16 inverse mass (unused by the GR1 integrate; carried for GR3/GR4)
    int  radius;         // Q16.16 grain radius (the GR1 radius-aware ground-rest input)
    uint flags;          // bit0 = STATIC
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt   (all Q16.16)
//   cfg  : x=groundY (Q16.16), y=grainCount, z=steps, w=integrateEnabled
struct GrainParams {
    int4 grav;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle> gGrains : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<GrainParams>   gParams : register(u1);

// VERBATIM grain.h::fxmul / fluid.h::fxmul / fpx.h::fxmul — (a*b) >> kFrac with an int64 intermediate
// (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_GRAIN_FRAC);
}

[numthreads(HF_GRAIN_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int gravx = gParams[0].grav.x;
    int gravy = gParams[0].grav.y;
    int gravz = gParams[0].grav.z;
    int dt    = gParams[0].grav.w;
    int groundY          = gParams[0].cfg.x;
    int grainCount       = gParams[0].cfg.y;
    int steps            = gParams[0].cfg.z;
    int integrateEnabled = gParams[0].cfg.w;

    uint i = gid.x;
    if ((int)i >= grainCount) return;

    GrainParticle p = gGrains[i];

    // Disabled -> write the input back UNCHANGED (the byte-identical no-op proof).
    if (integrateEnabled == 0) { gGrains[i] = p; return; }

    // Run `steps` iterations of IntegrateGrainParticle's per-grain math (VERBATIM
    // grain.h::IntegrateGrainParticle).
    for (int s = 0; s < steps; ++s) {
        if ((p.flags & HF_GRAIN_FLAG_STATIC) == 0u) {
            // (1) integrate velocity: vel += gravity * dt.
            p.vx += fxmul(gravx, dt);
            p.vy += fxmul(gravy, dt);
            p.vz += fxmul(gravz, dt);
            // (2) snapshot the previous position (the predicted-position anchor GR2-GR4 reads).
            p.prx = p.px;
            p.pry = p.py;
            p.prz = p.pz;
            // (3) integrate position: pos += vel * dt.
            p.px += fxmul(p.vx, dt);
            p.py += fxmul(p.vy, dt);
            p.pz += fxmul(p.vz, dt);
            // (4) RADIUS-AWARE ground rest (no restitution — GR1 is free-fall + ground only). The grain's
            // surface rests on the floor: clamp the CENTER to groundY + radius (radius 0 -> the FL1 clamp).
            int restY = groundY + p.radius;
            if (p.py < restY) {
                p.py = restY;
                if (p.vy < 0) p.vy = 0;
            }
        }
    }

    gGrains[i] = p;
}
