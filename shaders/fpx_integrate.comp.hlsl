// Slice FPX1 — Deterministic Fixed-Point Physics: the Q16.16 INTEGRATOR compute pass (the BEACHHEAD of
// FLAGSHIP #6). ONE thread per body (i < bodyCount). Each thread runs `steps` iterations of the
// semi-implicit-Euler integrator (gravity*dt -> pos+=vel*dt -> ground floor-clamp) on its OWN body,
// the fxmul + integrate + floor-clamp copied VERBATIM from engine/sim/fpx.h::IntegrateBody, and writes
// gBodies[i] back. Each body is INDEPENDENT (no inter-body coupling until FPX3), so the per-thread
// write is order-independent — race-free, NO atomics, bit-identical GPU==CPU + cross-backend (the
// VT1 set-write / MC1 classify / SW2 integer-replay argument applied to a physics integrator).
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it
// consumes the host-snapped Q16.16 int FxBody array and runs the SAME pure-integer fxmul
// ((int64)a*b >> 16, an ARITHMETIC right shift on int64) + integer add + integer compare the header
// runs. A divergence here vs engine/sim/fpx.h is exactly what the host's GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux): fxmul uses int64_t — IDENTICAL to fpx.h::fxmul. HLSL SM6
// supports int64_t (the SAME pattern shaders/swraster.comp.hlsl::SwEdge uses — DXC -spirv with the
// Int64 capability; spirv-cross lowers it to MSL `long`); the product of two Q16.16 values in the
// documented +-32768 bound is < 2^62, well inside int64. The single mul+shift needs NO MSL-2.2
// (NO --msl-version 20200), the mc_classify/swraster lesson.
//
// integrateEnabled push/param flag: 0 -> write the input body back UNCHANGED (the disabled-path no-op;
// gBodies stays byte-identical to the upload).
//
// Buffers (storage, bound at compute bindings 0..1; on Metal these land at buffer(0..1)):
//   b0 gBodies : the Q16.16 FxBody array (pos.xyz, vel.xyz, invMass, flags — std430 ints), READ+WRITE.
//   b1 gParams : { gravity.xyz, dt, groundY, bodyCount, steps, integrateEnabled }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations
// (same as vsm_mark.comp / mc_classify.comp / swraster.comp), not backend CODE symbols.

#define HF_FPX_THREADS 64
#define HF_FPX_FRAC 16   // MUST match fpx.h::kFrac
#define HF_FPX_FLAG_DYNAMIC 1u

// std430 FxBody mirror (engine/sim/fpx.h::FxBody): pos.xyz, vel.xyz, invMass (int), flags (uint).
// 8 x 4-byte = 32 bytes, no padding holes (memcmp-able).
struct FpxBody {
    int  px, py, pz;   // Q16.16 position
    int  vx, vy, vz;   // Q16.16 velocity
    int  invMass;      // Q16.16 inverse mass (unused in FPX1's integrate; carried for FPX3)
    uint flags;        // bit0 = dynamic
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt   (all Q16.16)
//   cfg  : x=groundY (Q16.16), y=bodyCount, z=steps, w=integrateEnabled
struct FpxParams {
    int4 grav;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FpxBody>   gBodies : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FpxParams> gParams : register(u1);

// VERBATIM fpx.h::fxmul — (a*b) >> kFrac with an int64 intermediate (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_FPX_FRAC);
}

[numthreads(HF_FPX_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int gravx = gParams[0].grav.x;
    int gravy = gParams[0].grav.y;
    int gravz = gParams[0].grav.z;
    int dt    = gParams[0].grav.w;
    int groundY          = gParams[0].cfg.x;
    int bodyCount        = gParams[0].cfg.y;
    int steps            = gParams[0].cfg.z;
    int integrateEnabled = gParams[0].cfg.w;

    uint i = gid.x;
    if ((int)i >= bodyCount) return;

    FpxBody b = gBodies[i];

    // Disabled -> write the input back UNCHANGED (the byte-identical no-op proof).
    if (integrateEnabled == 0) { gBodies[i] = b; return; }

    // Run `steps` iterations of IntegrateBody's per-body math (VERBATIM fpx.h::IntegrateBody).
    for (int s = 0; s < steps; ++s) {
        if ((b.flags & HF_FPX_FLAG_DYNAMIC) != 0u) {
            // (1) integrate velocity: vel += gravity * dt.
            b.vx += fxmul(gravx, dt);
            b.vy += fxmul(gravy, dt);
            b.vz += fxmul(gravz, dt);
            // (2) integrate position: pos += vel * dt.
            b.px += fxmul(b.vx, dt);
            b.py += fxmul(b.vy, dt);
            b.pz += fxmul(b.vz, dt);
            // (3) ground floor clamp (no restitution yet — FPX3).
            if (b.py < groundY) {
                b.py = groundY;
                if (b.vy < 0) b.vy = 0;
            }
        }
    }

    gBodies[i] = b;
}
