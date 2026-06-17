// Slice FPX4 — Deterministic Fixed-Point Physics: the integer QUATERNION ORIENTATION integrator compute
// pass (Phase 11 #4). ONE thread PER BODY (the orientation integrator is per-body INDEPENDENT — unlike
// FPX3's serial Gauss-Seidel solver — so this is one-thread-per-body, 64 threads/group, NO atomics, NO
// race). Each thread runs `steps` iterations of IntegrateBodyFull = the FPX1 translational integrate
// (vel += gravity*dt; pos += vel*dt) + IntegrateOrientation (the fixed-point quaternion angular
// integrator: dq = ω⊗q; orient += 0.5*dq*dt; orient = normalize) on its OWN body, the
// FxQuatMul/FxQuatNormalize/FxISqrt/fxdiv/fxmul math copied VERBATIM from engine/sim/fpx.h so
// tests/fpx_test.cpp + the GPU pass exercise the EXACT integer ops -> a divergence is exactly what the
// host GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux): fxmul/fxdiv/FxISqrt use int64_t (IDENTICAL to fpx.h). DXC -spirv
// compiles int64 (the Int64 capability, the fpx_solve.comp / swraster.comp pattern); glslc (the Metal
// HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY (in the
// Vulkan compile list, NOT in the Metal hf_gen_msl list). The Metal --fpx-orient showcase runs the CPU
// fpx::IntegrateBodyFull over the same bodies -> byte-identical to this GPU result by construction, while
// the Vulkan side carries the GPU==CPU bit-identity proof.
//
// enabled=0 -> write the input bodies back UNCHANGED (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..1):
//   b0 gBodies : the Q16.16 FxBody array — the FPX4 pack: pos.xyz, vel.xyz, invMass, flags, radius,
//                orient.xyzw, angVel.xyz (std430 ints, 16 x 4-byte = 64 bytes), R+W.
//   b1 gParams : { gravity.xyz, dt, bodyCount, steps, enabled }, READ.

#define HF_FPX_FRAC 16   // MUST match fpx.h::kFrac
#define HF_FPX_ONE  (1 << HF_FPX_FRAC)
#define HF_FPX_HALF (HF_FPX_ONE / 2)  // MUST match fpx.h::kHalf

// std430 FxBody mirror (engine/sim/fpx.h::FxBody, FPX4 pack): the FPX1-3 fields (pos/vel/invMass/flags/
// radius) THEN orient.xyzw + angVel.xyz. 16 x 4-byte = 64 bytes (memcmp-able against the host FpxBodyGpu).
struct FpxBody {
    int  px, py, pz;   // Q16.16 position
    int  vx, vy, vz;   // Q16.16 velocity
    int  invMass;      // Q16.16 inverse mass (0 => static)
    uint flags;        // bit0 = dynamic
    int  radius;       // Q16.16 sphere radius (FPX2)
    int  ox, oy, oz, ow;   // Q16.16 orientation quaternion (identity 0,0,0,kOne)
    int  ax, ay, az;       // Q16.16 angular velocity
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt   (all Q16.16)
//   cfg  : x=bodyCount, y=steps, z=enabled, w=_
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

// VERBATIM fpx.h::fxdiv — (a << kFrac) / b in Q16.16 (int64 shift + truncating divide; guard b==0).
int fxdiv(int a, int b) {
    if (b == 0) return 0;
    return (int)(((int64_t)a << HF_FPX_FRAC) / (int64_t)b);
}

// VERBATIM fpx.h::FxISqrt — floor(sqrt) of a non-negative int64 (binary digit-by-digit).
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

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int bodyCount = gParams[0].cfg.x;
    int steps     = gParams[0].cfg.y;
    int enabled   = gParams[0].cfg.z;

    int i = (int)gid.x;
    if (i >= bodyCount) return;        // one thread per body; guard the tail of the last group.

    // Disabled -> leave the body UNCHANGED (the byte-identical no-op; gBodies is already the input).
    if (enabled == 0) return;

    int gravx = gParams[0].grav.x;
    int gravy = gParams[0].grav.y;
    int gravz = gParams[0].grav.z;
    int dt    = gParams[0].grav.w;

    const uint kFlagDynamic = 1u;

    FpxBody b = gBodies[i];
    for (int s = 0; s < steps; ++s) {
        // (A) FPX1 translational integrate (dynamic bodies only; FPX4 showcase uses NO ground clamp).
        if ((b.flags & kFlagDynamic) != 0u) {
            b.vx += fxmul(gravx, dt);
            b.vy += fxmul(gravy, dt);
            b.vz += fxmul(gravz, dt);
            b.px += fxmul(b.vx, dt);
            b.py += fxmul(b.vy, dt);
            b.pz += fxmul(b.vz, dt);
        }

        // (B) IntegrateOrientation: dq = FxQuatMul(omega, orient) with omega = {angVel.xyz, 0}.
        // VERBATIM fpx.h::FxQuatMul (the Hamilton product, each term an int64 fxmul).
        int qx = b.ox, qy = b.oy, qz = b.oz, qw = b.ow;
        int wx = b.ax, wy = b.ay, wz = b.az, ww = 0;   // the pure-quaternion omega {angVel.xyz, 0}
        int dqw = fxmul(ww, qw) - fxmul(wx, qx) - fxmul(wy, qy) - fxmul(wz, qz);
        int dqx = fxmul(ww, qx) + fxmul(wx, qw) + fxmul(wy, qz) - fxmul(wz, qy);
        int dqy = fxmul(ww, qy) - fxmul(wx, qz) + fxmul(wy, qw) + fxmul(wz, qx);
        int dqz = fxmul(ww, qz) + fxmul(wx, qy) - fxmul(wy, qx) + fxmul(wz, qw);

        // orient += 0.5*dq*dt component-wise.
        qx += fxmul(fxmul(dqx, HF_FPX_HALF), dt);
        qy += fxmul(fxmul(dqy, HF_FPX_HALF), dt);
        qz += fxmul(fxmul(dqz, HF_FPX_HALF), dt);
        qw += fxmul(fxmul(dqw, HF_FPX_HALF), dt);

        // orient = FxQuatNormalize(orient). VERBATIM fpx.h::FxQuatNormalize (int64 sum-of-squares -> Q32.32
        // -> floor-sqrt -> Q16.16 len; fxdiv each component; len==0 -> identity).
        int64_t ssx = (int64_t)qx * (int64_t)qx;
        int64_t ssy = (int64_t)qy * (int64_t)qy;
        int64_t ssz = (int64_t)qz * (int64_t)qz;
        int64_t ssw = (int64_t)qw * (int64_t)qw;
        int len = (int)FxISqrt(ssx + ssy + ssz + ssw);
        if (len == 0) { qx = 0; qy = 0; qz = 0; qw = HF_FPX_ONE; }
        else { qx = fxdiv(qx, len); qy = fxdiv(qy, len); qz = fxdiv(qz, len); qw = fxdiv(qw, len); }

        b.ox = qx; b.oy = qy; b.oz = qz; b.ow = qw;
    }
    gBodies[i] = b;
}
