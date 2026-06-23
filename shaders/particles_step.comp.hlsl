// Slice PT4 — The composed StepParticles tick: the Q16.16 per-particle FORCE-ACCUMULATE + force-INTEGRATE
// THEN COLLIDE compute pass (the 4th slice of FLAGSHIP #19: DETERMINISTIC GPU PARTICLES). ONE thread per
// particle slot (i < count). Each thread (1) accumulates the force on its OWN particle over the FIXED field
// list (point attractor/repeller, vortex swirl, uniform wind) in ASCENDING field index order, (2) runs the
// integrate with that force ADDED to gravity (vel += (gravity+force)*dt; vel -= vel*dragK; pos += vel*dt; age
// + death) THEN (3) collides its OWN particle with the world: a ground PLANE (clamp surface + reflect the
// downward velocity) then a FIXED-ORDER loop over a static SPHERE collider list (project the centre out +
// reflect the inward velocity). The accumulate+integrate body is copied VERBATIM from particles_forces.comp
// (== engine/sim/particles.h::AccumulateForce + IntegrateParticleWithForce); the collide body is copied
// VERBATIM from particles_collide.comp (== engine/sim/particles.h::CollideParticlePlane + CollideParticle-
// Sphere). This single pass composes BOTH (NO double-integrate): it is the GPU half of particles.h::
// StepParticles, which is exactly Emit -> IntegrateParticlesWithForces -> CollideParticleWorld -> RecycleDead.
// Each particle is INDEPENDENT (it reads the SAME read-only field + collider lists, writes only its OWN slot)
// -> race-free, NO atomics, bit-identical GPU==CPU + cross-backend (the GR1/FL1/PT1/PT2/PT3 argument with TWO
// read-only inputs). The EMIT + RECYCLE stay HOST-side (single-thread, between dispatches — the deterministic
// free-list); the GPU dispatch is the per-particle force-integrate+collide.
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it consumes
// the host-snapped Q16.16 int FxParticle + ForceField + ParticleSphereCollider arrays and runs the SAME
// pure-integer fxmul/fxdiv/FxLength(FxISqrt)/FxNormalize/FxDot/FxCross the header runs, summing fields then
// colliding plane-then-spheres in the SAME fixed order. A divergence here vs engine/sim/particles.h is exactly
// what the host's GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux, the GR1/FL1/PT1/PT2/PT3 lesson): fxmul/fxdiv/FxISqrt use int64_t —
// IDENTICAL to particles.h. The falloff fxdiv, the FxLength sum-of-squares, the FxNormalize fxdiv, the FxDot,
// and the gravity*dt + (1+e)*vn fxmul all exceed int32 for Q16.16 world-scale values, so this shader uses
// int64. HLSL SM6 supports int64_t (the grain/fluid/PT2/PT3 pattern — DXC -spirv with the Int64 capability).
// glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-
// ONLY (NOT in metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --pt4-step showcase runs the CPU
// particles::StepParticles (byte-identical by construction).
//
// Buffers (storage, bound at compute bindings 0..3; this shader is Vulkan-only):
//   b0 gParticles : the Q16.16 FxParticle array (pos.xyz, vel.xyz, age, lifetime, seed, flags — std430 ints,
//                   48 bytes), READ+WRITE.
//   b1 gParams    : { gravity.xyz, dt, dragK, count, fieldCount, sphereCount, groundY, radius, restitution },
//                   READ.
//   b2 gFields    : the Q16.16 ForceField array (kind, center.xyz, axis.xyz, strength, radius — std430), READ.
//   b3 gSpheres   : the Q16.16 ParticleSphereCollider array (center.xyz, radius, +pad — std430), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// particles_forces.comp / particles_collide.comp / grain_integrate.comp), not backend CODE symbols.

#define HF_PT_THREADS 64
#define HF_PT_FRAC 16     // MUST match particles.h::kFrac (== fpx.h::kFrac)
#define HF_PT_FLAG_ALIVE 1u
#define HF_PT_ONE 65536   // kOne in Q16.16 (1 << 16)
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

// std430 ParticleSphereCollider mirror (engine/sim/particles.h::ParticleSphereCollider, host-padded to a
// 16-byte stride): center.xyz + radius. 4 x 4-byte = 16 bytes (the C++ upload matches).
struct FxSphere {
    int  cx, cy, cz;     // Q16.16 center
    int  radius;         // Q16.16 sphere radius
};

// Params (std430). Mirrors the C++ upload struct.
//   grav : x=gravity.x, y=gravity.y, z=gravity.z, w=dt                       (all Q16.16)
//   cfg  : x=dragK (Q16.16), y=count, z=fieldCount, w=sphereCount
//   col  : x=groundY (Q16.16), y=radius (Q16.16), z=restitution (Q16.16), w=pad
struct PtParams {
    int4 grav;
    int4 cfg;
    int4 col;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxParticle>   gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<PtParams>     gParams    : register(u1);
[[vk::binding(2, 0)]] StructuredBuffer<FxForceField>   gFields    : register(t0);
[[vk::binding(3, 0)]] StructuredBuffer<FxSphere>       gSpheres   : register(t1);

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
// VERBATIM particles.h::FxDot — (ax*bx+ay*by+az*bz) >> kFrac, sum kept in int64.
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
    int dragK       = gParams[0].cfg.x;
    int count       = gParams[0].cfg.y;
    int fieldCount  = gParams[0].cfg.z;
    int sphereCount = gParams[0].cfg.w;
    int groundY = gParams[0].col.x;
    int radius  = gParams[0].col.y;
    int e       = gParams[0].col.z;

    uint i = gid.x;
    if ((int)i >= count) return;

    FxParticle p = gParticles[i];

    if ((p.flags & HF_PT_FLAG_ALIVE) == 0u) { gParticles[i] = p; return; }

    // (A) AccumulateForce over the fields in ASCENDING index order (VERBATIM particles.h::AccumulateForce /
    // particles_forces.comp). fieldCount==0 -> the loop is skipped so force stays (0,0,0) (the gravity-only
    // control path == PT1 integrate).
    int fx_ = 0, fy_ = 0, fz_ = 0;
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

    // (B) IntegrateParticleWithForce (VERBATIM particles.h / particles_forces.comp): vel += (gravity+force)*dt;
    // drag; pos += vel*dt; age + death. (force==0 -> EXACTLY IntegrateParticle.)
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

    // (C) CollideParticleWorld: only if still ALIVE (a particle that died this step is NOT collided — matches
    // CollideParticleWorld's ALIVE gate after the integrate marks death). VERBATIM particles_collide.comp.
    if ((p.flags & HF_PT_FLAG_ALIVE) != 0u) {
        // CollideParticlePlane (VERBATIM particles.h): clamp surface to ground + reflect downward vel.
        int restY = groundY + radius;
        if (p.py < restY) {
            p.py = restY;
            if (p.vy < 0) p.vy = -fxmul(e, p.vy);
        }
        // CollideParticleSphere loop (VERBATIM particles.h), spheres in ASCENDING fixed order.
        for (int s = 0; s < sphereCount; ++s) {
            FxSphere sp = gSpheres[s];
            int surf = sp.radius + radius;
            int dx = p.px - sp.cx;
            int dy = p.py - sp.cy;
            int dz = p.pz - sp.cz;
            int ax = dx < 0 ? -dx : dx;
            int ay = dy < 0 ? -dy : dy;
            int az = dz < 0 ? -dz : dz;
            if (ax > surf || ay > surf || az > surf) continue;   // outside the AABB -> no overlap
            int dist = FxLength3(dx, dy, dz);
            if (dist >= surf) continue;                          // outside the (expanded) sphere -> untouched
            // nrm = FxNormalize(d): dist==0 -> {0,kOne,0} fallback.
            int nrmx, nrmy, nrmz;
            if (dist == 0) { nrmx = 0; nrmy = HF_PT_ONE; nrmz = 0; }
            else { nrmx = fxdiv(dx, dist); nrmy = fxdiv(dy, dist); nrmz = fxdiv(dz, dist); }
            // pos = center + nrm*surf (snap the centre to sphereR + radius).
            p.px = sp.cx + fxmul(nrmx, surf);
            p.py = sp.cy + fxmul(nrmy, surf);
            p.pz = sp.cz + fxmul(nrmz, surf);
            // vn = vel·nrm; if (vn < 0) vel -= nrm*(1+e)*vn.
            int vn = FxDot3(p.vx, p.vy, p.vz, nrmx, nrmy, nrmz);
            if (vn < 0) {
                int j = fxmul(HF_PT_ONE + e, vn);
                p.vx -= fxmul(nrmx, j);
                p.vy -= fxmul(nrmy, j);
                p.vz -= fxmul(nrmz, j);
            }
        }
    }

    gParticles[i] = p;
}
