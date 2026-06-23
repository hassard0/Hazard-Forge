// Slice PT2 — Deterministic integer FORCE FIELDS: the Q16.16 per-particle FORCE-ACCUMULATE + force-integrate
// compute pass (the 2nd slice of FLAGSHIP #19: DETERMINISTIC GPU PARTICLES). ONE thread per particle slot
// (i < count). Each thread (1) accumulates the force on its OWN particle over the FIXED field list (point
// attractor/repeller, vortex swirl, uniform wind) in ASCENDING field index order — the deterministic
// associative-order contract — then (2) runs the PT1 integrate with that force ADDED to gravity (vel +=
// (gravity+force)*dt; vel -= vel*dragK; pos += vel*dt; age + death). The accumulate + integrate copied
// VERBATIM from engine/sim/particles.h::AccumulateForce + IntegrateParticleWithForce. Each particle is
// INDEPENDENT (it reads the SAME read-only field list, writes only its OWN slot) -> race-free, NO atomics,
// bit-identical GPU==CPU + cross-backend (the GR1/FL1/PT1 argument extended with a read-only field input).
// The EMIT + RECYCLE stay HOST-side (single-thread, between dispatches — the deterministic free-list); the
// GPU dispatch is ONLY the per-particle accumulate+integrate. forcesEnabled=0 -> force is forced to (0,0,0)
// so the integrate equals PT1's IntegrateParticle EXACTLY (the no-op control path) BUT the gravity-only
// integrate STILL runs (so disabled == a plain PT1 step, not a write-back no-op — matches the showcase's
// zero-fields==PT1 control which still integrates under gravity).
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it consumes
// the host-snapped Q16.16 int FxParticle + ForceField arrays and runs the SAME pure-integer fxmul/fxdiv/
// FxLength(FxISqrt)/FxNormalize/FxDot/FxCross the header runs, summing fields in the SAME ascending order.
// A divergence here vs engine/sim/particles.h is exactly what the host's GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux, the GR1/FL1/PT1 lesson): fxmul/fxdiv/FxISqrt use int64_t — IDENTICAL
// to particles.h. The falloff fxdiv + the FxLength sum-of-squares + the gravity*dt fxmul all exceed int32 for
// Q16.16 world-scale values, so this shader uses int64. HLSL SM6 supports int64_t (the grain/fluid pattern —
// DXC -spirv with the Int64 capability). glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in
// HLSL, so this shader is VULKAN-SPIR-V-ONLY (NOT in metal_headless/CMakeLists.txt hf_gen_msl); on Metal the
// --pt2-forces showcase runs the CPU particles::StepEmitForcesIntegrate (byte-identical by construction).
//
// Buffers (storage, bound at compute bindings 0..2; this shader is Vulkan-only):
//   b0 gParticles : the Q16.16 FxParticle array (pos.xyz, vel.xyz, age, lifetime, seed, flags — std430 ints,
//                   48 bytes), READ+WRITE.
//   b1 gParams    : { gravity.xyz, dt, dragK, count, fieldCount, forcesEnabled }, READ.
//   b2 gFields    : the Q16.16 ForceField array (kind, center.xyz, axis.xyz, strength, radius — std430), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// particles_integrate.comp / grain_integrate.comp), not backend CODE symbols.

#define HF_PT_THREADS 64
#define HF_PT_FRAC 16     // MUST match particles.h::kFrac (== fpx.h::kFrac)
#define HF_PT_FLAG_ALIVE 1u
#define HF_FIELD_POINT  0u
#define HF_FIELD_VORTEX 1u
#define HF_FIELD_WIND   2u

// std430 FxParticle mirror (engine/sim/particles.h::FxParticle): 12 x 4-byte = 48 bytes (memcmp-able).
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

// std430 ForceField mirror (engine/sim/particles.h::ForceField, host-padded to a 48-byte stride): kind (uint),
// center.xyz, axis.xyz, strength, radius + 2 pad ints. 12 x 4-byte = 48 bytes (the C++ upload pads to match).
struct FxForceField {
    uint kind;           // 0 point / 1 vortex / 2 wind
    int  cx, cy, cz;     // Q16.16 center
    int  ax, ay, az;     // Q16.16 axis (unit-ish)
    int  strength;       // Q16.16 signed magnitude
    int  radius;         // Q16.16 influence radius
    int  pad0, pad1, pad2; // padding -> 48-byte std430 stride (12 x int32; matches the C++ upload)
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt   (all Q16.16)
//   cfg  : x=dragK (Q16.16), y=count, z=fieldCount, w=forcesEnabled
struct PtParams {
    int4 grav;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxParticle>   gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<PtParams>     gParams    : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<FxForceField>   gFields    : register(t0);

// VERBATIM particles.h::fxmul / fpx.h::fxmul — (a*b) >> kFrac with an int64 intermediate (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_PT_FRAC);
}
// VERBATIM fpx.h::fxdiv — (a << kFrac) / b in Q16.16, int64 shift then truncating divide (b==0 -> 0).
int fxdiv(int a, int b) {
    if (b == 0) return 0;
    return (int)(((int64_t)a << HF_PT_FRAC) / (int64_t)b);
}
// VERBATIM fpx.h::FxISqrt — floor(sqrt(v)) for non-negative int64 v (binary digit-by-digit, NO float).
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
// VERBATIM fpx.h::FxLength — sqrt(x^2+y^2+z^2) in Q16.16 (FxISqrt of the int64 sum-of-squares).
int FxLength3(int x, int y, int z) {
    int64_t sx = (int64_t)x * (int64_t)x;
    int64_t sy = (int64_t)y * (int64_t)y;
    int64_t sz = (int64_t)z * (int64_t)z;
    return (int)FxISqrt(sx + sy + sz);
}
// VERBATIM particles.h::FxDot (== convex.h) — (ax*bx+ay*by+az*bz) >> kFrac, sum kept in int64.
int FxDot3(int ax, int ay, int az, int bx, int by, int bz) {
    int64_t d = (int64_t)ax * (int64_t)bx + (int64_t)ay * (int64_t)by + (int64_t)az * (int64_t)bz;
    return (int)(d >> HF_PT_FRAC);
}

[numthreads(HF_PT_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int gravx = gParams[0].grav.x;
    int gravy = gParams[0].grav.y;
    int gravz = gParams[0].grav.z;
    int dt    = gParams[0].grav.w;
    int dragK         = gParams[0].cfg.x;
    int count         = gParams[0].cfg.y;
    int fieldCount    = gParams[0].cfg.z;
    int forcesEnabled = gParams[0].cfg.w;

    uint i = gid.x;
    if ((int)i >= count) return;

    FxParticle p = gParticles[i];

    if ((p.flags & HF_PT_FLAG_ALIVE) == 0u) { gParticles[i] = p; return; }

    // (A) AccumulateForce over the fields in ASCENDING index order (VERBATIM particles.h::AccumulateForce).
    // forcesEnabled=0 -> skip the loop entirely so force stays (0,0,0) (the zero-fields==PT1 control path).
    int fx_ = 0, fy_ = 0, fz_ = 0;
    if (forcesEnabled != 0) {
        for (int f = 0; f < fieldCount; ++f) {
            FxForceField fld = gFields[f];
            if (fld.kind == HF_FIELD_POINT) {
                // d = center - pos.
                int dx = fld.cx - p.px;
                int dy = fld.cy - p.py;
                int dz = fld.cz - p.pz;
                int dist = FxLength3(dx, dy, dz);
                if (dist > 0 && dist < fld.radius) {
                    // dir = normalize(d).
                    int dirx = fxdiv(dx, dist);
                    int diry = fxdiv(dy, dist);
                    int dirz = fxdiv(dz, dist);
                    int falloff = fxdiv(fld.radius - dist, fld.radius);
                    int mag = fxmul(fld.strength, falloff);
                    fx_ += fxmul(dirx, mag);
                    fy_ += fxmul(diry, mag);
                    fz_ += fxmul(dirz, mag);
                }
            } else if (fld.kind == HF_FIELD_VORTEX) {
                // r = pos - center.
                int rx = p.px - fld.cx;
                int ry = p.py - fld.cy;
                int rz = p.pz - fld.cz;
                int along = FxDot3(rx, ry, rz, fld.ax, fld.ay, fld.az);
                // rPerp = r - axis*along.
                int rpx = rx - fxmul(fld.ax, along);
                int rpy = ry - fxmul(fld.ay, along);
                int rpz = rz - fxmul(fld.az, along);
                int dist = FxLength3(rpx, rpy, rpz);
                if (dist > 0 && dist < fld.radius) {
                    // tang = cross(axis, rPerp).
                    int tx = fxmul(fld.ay, rpz) - fxmul(fld.az, rpy);
                    int ty = fxmul(fld.az, rpx) - fxmul(fld.ax, rpz);
                    int tz = fxmul(fld.ax, rpy) - fxmul(fld.ay, rpx);
                    int tlen = FxLength3(tx, ty, tz);
                    if (tlen > 0) {
                        int dirx = fxdiv(tx, tlen);
                        int diry = fxdiv(ty, tlen);
                        int dirz = fxdiv(tz, tlen);
                        int falloff = fxdiv(fld.radius - dist, fld.radius);
                        int mag = fxmul(fld.strength, falloff);
                        fx_ += fxmul(dirx, mag);
                        fy_ += fxmul(diry, mag);
                        fz_ += fxmul(dirz, mag);
                    }
                }
            } else if (fld.kind == HF_FIELD_WIND) {
                fx_ += fxmul(fld.ax, fld.strength);
                fy_ += fxmul(fld.ay, fld.strength);
                fz_ += fxmul(fld.az, fld.strength);
            }
        }
    }

    // (B) IntegrateParticleWithForce (VERBATIM particles.h): vel += (gravity+force)*dt; drag; pos += vel*dt;
    // age + death. (force==0 -> EXACTLY IntegrateParticle.)
    p.vx += fxmul(gravx + fx_, dt);
    p.vy += fxmul(gravy + fy_, dt);
    p.vz += fxmul(gravz + fz_, dt);
    p.vx -= fxmul(p.vx, dragK);
    p.vy -= fxmul(p.vy, dragK);
    p.vz -= fxmul(p.vz, dragK);
    p.px += fxmul(p.vx, dt);
    p.py += fxmul(p.vy, dt);
    p.pz += fxmul(p.vz, dt);
    p.age += dt;
    if (p.age >= p.lifetime) p.flags &= ~HF_PT_FLAG_ALIVE;

    gParticles[i] = p;
}
