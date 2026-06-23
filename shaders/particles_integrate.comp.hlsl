// Slice PT1 — Deterministic GPU Particles: the Q16.16 PARTICLE POOL INTEGRATOR compute pass (the BEACHHEAD of
// FLAGSHIP #19: DETERMINISTIC GPU PARTICLES). ONE thread per particle slot (i < count). Each thread runs the
// semi-implicit-Euler integrator + linear drag + age/death on its OWN slot (vel += gravity*dt; vel -= vel*dragK;
// pos += vel*dt; age += dt; age >= lifetime -> clear ALIVE), the fxmul + integrate + drag + age copied VERBATIM
// from engine/sim/particles.h::IntegrateParticle, and writes gParticles[i] back. In PT1 each particle is
// INDEPENDENT (no neighbours/collision), so the per-thread write is order-independent — race-free, NO atomics,
// bit-identical GPU==CPU + cross-backend (the GR1 grain_integrate / FL1 fluid_integrate argument applied to a
// particle pool). The EMIT + RECYCLE are HOST-side (single-thread, between integrate dispatches) — the GPU
// dispatch is ONLY the per-particle-independent integrate. ONE integrate step per dispatch (the K-step tick
// loop is host-driven: Emit -> dispatch -> RecycleDead, K times — the spec's host-side deterministic free-list).
//
// DELTA vs grain_integrate.comp: the FxParticle packs pos.xyz, vel.xyz, age, lifetime, seed, flags (12 x int32
// = 48 bytes) instead of grain's pos/prev/vel/invMass/radius/flags; the integrate has a LINEAR-DRAG term + an
// age/death instead of a radius-aware ground rest. Everything else is grain_integrate.comp's int64 fxmul shape.
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it consumes
// the host-snapped Q16.16 int FxParticle array and runs the SAME pure-integer fxmul ((int64)a*b >> 16, an
// ARITHMETIC right shift on int64) + integer add + integer compare the header runs. A divergence here vs
// engine/sim/particles.h is exactly what the host's GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux, the GR1/FL1/CL1 lesson): fxmul uses int64_t — IDENTICAL to
// particles.h::fxmul / grain.h::fxmul / fpx.h::fxmul. The integrate has the SAME `vel += gravity*dt;
// pos += vel*dt` form as grain_integrate.comp, whose (int64)a*b product before the >>16 shift exceeds int32 for
// Q16.16 gravity*dt (gravity ≈ -9.8*65536; products of two Q16.16 world-scale values blow past 2^31). To stay
// bit-exact to that int64-intermediate reference WITHOUT any overflow fragility, this shader uses int64. HLSL
// SM6 supports int64_t (the grain_integrate.comp pattern — DXC -spirv with the Int64 capability). glslc (the
// Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY (NOT in
// metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --pt1-emit showcase runs the CPU
// particles::StepEmitIntegrate (byte-identical by construction).
//
// integrateEnabled push/param flag: 0 -> write the input particle back UNCHANGED (the disabled-path no-op;
// gParticles stays byte-identical to the upload).
//
// Buffers (storage, bound at compute bindings 0..1; this shader is Vulkan-only):
//   b0 gParticles : the Q16.16 FxParticle array (pos.xyz, vel.xyz, age, lifetime, seed, flags — std430 ints,
//                   48 bytes), READ+WRITE.
//   b1 gParams    : { gravity.xyz, dt, dragK, count, steps, integrateEnabled }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// grain_integrate.comp / fluid_integrate.comp / fpx_integrate.comp), not backend CODE symbols.

#define HF_PT_THREADS 64
#define HF_PT_FRAC 16     // MUST match particles.h::kFrac (== fpx.h::kFrac)
#define HF_PT_FLAG_ALIVE 1u

// std430 FxParticle mirror (engine/sim/particles.h::FxParticle): pos.xyz, vel.xyz, age (int), lifetime (int),
// seed (uint), flags (uint), rsv0, rsv1. 12 x 4-byte = 48 bytes, no padding holes (memcmp-able). The two
// reserved ints round the struct to the 48-byte std430 stride (GrainParticle's prev/invMass/radius space) and
// are carried VERBATIM by the integrate (untouched -> byte-identical in the memcmp).
struct FxParticle {
    int  px, py, pz;     // Q16.16 current position
    int  vx, vy, vz;     // Q16.16 velocity
    int  age;            // Q16.16 seconds since spawn
    int  lifetime;       // Q16.16 max age
    uint seed;           // spawn hash (0 == empty slot)
    uint flags;          // bit0 = ALIVE
    int  rsv0;           // reserved (0) — 48-byte std430 stride padding
    int  rsv1;           // reserved (0)
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt   (all Q16.16)
//   cfg  : x=dragK (Q16.16), y=count, z=steps, w=integrateEnabled
struct PtParams {
    int4 grav;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxParticle> gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<PtParams>   gParams    : register(u1);

// VERBATIM particles.h::fxmul / grain.h::fxmul / fpx.h::fxmul — (a*b) >> kFrac with an int64 intermediate
// (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_PT_FRAC);
}

[numthreads(HF_PT_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int gravx = gParams[0].grav.x;
    int gravy = gParams[0].grav.y;
    int gravz = gParams[0].grav.z;
    int dt    = gParams[0].grav.w;
    int dragK            = gParams[0].cfg.x;
    int count            = gParams[0].cfg.y;
    int steps            = gParams[0].cfg.z;
    int integrateEnabled = gParams[0].cfg.w;

    uint i = gid.x;
    if ((int)i >= count) return;

    FxParticle p = gParticles[i];

    // Disabled -> write the input back UNCHANGED (the byte-identical no-op proof).
    if (integrateEnabled == 0) { gParticles[i] = p; return; }

    // Run `steps` iterations of IntegrateParticle's per-particle math (VERBATIM
    // particles.h::IntegrateParticle). `steps` is normally 1 (the host runs Emit/RecycleDead between dispatches).
    for (int s = 0; s < steps; ++s) {
        if ((p.flags & HF_PT_FLAG_ALIVE) != 0u) {
            // (1) integrate velocity: vel += gravity * dt.
            p.vx += fxmul(gravx, dt);
            p.vy += fxmul(gravy, dt);
            p.vz += fxmul(gravz, dt);
            // (2) linear drag: vel -= vel * dragK.
            p.vx -= fxmul(p.vx, dragK);
            p.vy -= fxmul(p.vy, dragK);
            p.vz -= fxmul(p.vz, dragK);
            // (3) integrate position: pos += vel * dt.
            p.px += fxmul(p.vx, dt);
            p.py += fxmul(p.vy, dt);
            p.pz += fxmul(p.vz, dt);
            // (4) age + death (NO ground clamp — PT3). Recycle is host-side (RecycleDead, ascending).
            p.age += dt;
            if (p.age >= p.lifetime) p.flags &= ~HF_PT_FLAG_ALIVE;
        }
    }

    gParticles[i] = p;
}
