// Slice CL1 — Deterministic GPU Cloth: the Q16.16 PARTICLE LATTICE INTEGRATOR compute pass (the
// BEACHHEAD of FLAGSHIP #8: DETERMINISTIC GPU CLOTH). ONE thread per particle (i < particleCount). Each
// thread runs `steps` iterations of the semi-implicit-Euler integrator (vel += gravity*dt; prev = pos;
// pos += vel*dt; ground floor-clamp) on its OWN particle, the fxmul + integrate + prev-snap + floor-clamp
// copied VERBATIM from engine/sim/cloth.h::IntegrateParticle, and writes gParticles[i] back. In CL1 each
// particle is INDEPENDENT (no inter-particle constraints until CL2/CL3), so the per-thread write is
// order-independent — race-free, NO atomics, bit-identical GPU==CPU + cross-backend (the FPX1
// fpx_integrate / VT1 set-write / MC1 classify argument applied to a cloth integrator).
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it
// consumes the host-snapped Q16.16 int ClothParticle array and runs the SAME pure-integer fxmul
// ((int64)a*b >> 16, an ARITHMETIC right shift on int64) + integer add + integer compare the header
// runs. A divergence here vs engine/sim/cloth.h is exactly what the host's GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux, the FPX1 lesson): fxmul uses int64_t — IDENTICAL to
// cloth.h::fxmul / fpx.h::fxmul. The cloth integrate has the SAME `vel += gravity*dt; pos += vel*dt`
// form as fpx_integrate.comp, whose (int64)a*b product before the >>16 shift exceeds int32 for Q16.16
// gravity*dt (gravity ≈ -9.8*65536; products of two Q16.16 world-scale values blow past 2^31). To stay
// bit-exact to that int64-intermediate reference WITHOUT any overflow fragility, this shader uses int64.
// HLSL SM6 supports int64_t (the swraster.comp::SwEdge / fpx_integrate.comp pattern — DXC -spirv with
// the Int64 capability). glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so
// this shader is VULKAN-SPIR-V-ONLY (NOT in metal_headless/CMakeLists.txt hf_gen_msl); on Metal the
// --cloth-integrate showcase runs the CPU cloth::IntegrateParticles (byte-identical by construction).
//
// integrateEnabled push/param flag: 0 -> write the input particle back UNCHANGED (the disabled-path
// no-op; gParticles stays byte-identical to the upload).
//
// Buffers (storage, bound at compute bindings 0..1; on Metal these would land at buffer(0..1) — but this
// shader is Vulkan-only):
//   b0 gParticles : the Q16.16 ClothParticle array (pos.xyz, prev.xyz, vel.xyz, invMass, flags — std430
//                   ints, 44 bytes), READ+WRITE.
//   b1 gParams    : { gravity.xyz, dt, groundY, particleCount, steps, integrateEnabled }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// fpx_integrate.comp / mc_classify.comp), not backend CODE symbols.

#define HF_CLOTH_THREADS 64
#define HF_CLOTH_FRAC 16     // MUST match cloth.h::kFrac (== fpx.h::kFrac)
#define HF_CLOTH_FLAG_PINNED 1u

// std430 ClothParticle mirror (engine/sim/cloth.h::ClothParticle): pos.xyz, prev.xyz, vel.xyz, invMass
// (int), flags (uint). 11 x 4-byte = 44 bytes, no padding holes (memcmp-able).
struct ClothParticle {
    int  px, py, pz;     // Q16.16 current position
    int  prx, pry, prz;  // Q16.16 previous position
    int  vx, vy, vz;     // Q16.16 velocity
    int  invMass;        // Q16.16 inverse mass (unused by the CL1 integrate; carried for CL3)
    uint flags;          // bit0 = PINNED
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt   (all Q16.16)
//   cfg  : x=groundY (Q16.16), y=particleCount, z=steps, w=integrateEnabled
struct ClothParams {
    int4 grav;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ClothParticle> gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<ClothParams>   gParams    : register(u1);

// VERBATIM cloth.h::fxmul / fpx.h::fxmul — (a*b) >> kFrac with an int64 intermediate (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_CLOTH_FRAC);
}

[numthreads(HF_CLOTH_THREADS, 1, 1)]
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

    ClothParticle p = gParticles[i];

    // Disabled -> write the input back UNCHANGED (the byte-identical no-op proof).
    if (integrateEnabled == 0) { gParticles[i] = p; return; }

    // Run `steps` iterations of IntegrateParticle's per-particle math (VERBATIM cloth.h::IntegrateParticle).
    for (int s = 0; s < steps; ++s) {
        if ((p.flags & HF_CLOTH_FLAG_PINNED) == 0u) {
            // (1) integrate velocity: vel += gravity * dt.
            p.vx += fxmul(gravx, dt);
            p.vy += fxmul(gravy, dt);
            p.vz += fxmul(gravz, dt);
            // (2) snapshot the previous position (the PBD/verlet anchor CL2/CL3 reads).
            p.prx = p.px;
            p.pry = p.py;
            p.prz = p.pz;
            // (3) integrate position: pos += vel * dt.
            p.px += fxmul(p.vx, dt);
            p.py += fxmul(p.vy, dt);
            p.pz += fxmul(p.vz, dt);
            // (4) ground floor clamp (no restitution — CL1 is free-fall + pin + ground only).
            if (p.py < groundY) {
                p.py = groundY;
                if (p.vy < 0) p.vy = 0;
            }
        }
    }

    gParticles[i] = p;
}
