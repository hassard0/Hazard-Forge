// Slice FL4 — Deterministic GPU Fluid: the PBF VELOCITY-UPDATE + COLLISION pass (the 4th slice of FLAGSHIP
// #9). ONE thread per PARTICLE i, run ONCE after the `iters` Jacobi density iterations. It (1) derives the
// velocity from the NET position change vel = (pos − prev)/dt (the PBF velocity update — prev was the
// predicted-position anchor snapshotted in the FL1 integrate), then (2) projects the particle out of the
// ground plane + the STATIC sphere colliders (CollidePlane + CollideSpheres). Per-particle independent (each
// thread reads the read-only sphere set + writes only its OWN particle) -> NO race, NO atomics, deterministic
// regardless of thread order. STATIC particles skip the velocity update + sphere projection (the plane clamp
// still raises a fallen static, matching CollidePlane). Copied VERBATIM from fluid.h::StepFluid's velocity
// derivation + CollidePlane + CollideParticleSphere.
//
// INTEGER WIDTH: the velocity fxdiv + FxLength/FxNormalize use int64_t -> VULKAN-SPIR-V-ONLY (NOT in
// hf_gen_msl); the Metal --fluid-solve showcase runs the CPU fluid::StepFluid (byte-identical by
// construction, the fluid_lambda.comp / cloth_collide.comp convention).
//
// Buffers (storage, Vulkan-only):
//   b0 gParticles : the Q16.16 FluidParticle array, READ+WRITE (pos, prev, vel, flags).
//   b1 spheres    : the static SphereCollider set (center.xyz, radius — 4 x int32), READ.
//   b2 gParams    : the FluidSolveParams (dt, groundY, particleCount, sphereCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_FLUID_THREADS 64
#define HF_FLUID_FRAC 16
#define HF_FLUID_ONE 65536

struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

struct SphereCollider { int cx, cy, cz; int radius; };   // std430 16 bytes (== fluid.h::SphereCollider)

// FluidSolveParams (std430). cfg0 {dt, groundY, particleCount, sphereCount}, cfg1 {enabled, _, _, _}.
struct FluidSolveParams {
    int4 cfg0;
    int4 cfg1;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle>    gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<SphereCollider>   spheres    : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FluidSolveParams> gParams    : register(u2);

static const uint HF_FLUID_STATIC = 1u;   // == fluid.h::kFlagStatic (bit0)

int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_FLUID_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_FLUID_FRAC) / (int64_t)b); }
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
int FxLength3(int x, int y, int z) {
    int64_t sx = (int64_t)x * (int64_t)x;
    int64_t sy = (int64_t)y * (int64_t)y;
    int64_t sz = (int64_t)z * (int64_t)z;
    return (int)FxISqrt(sx + sy + sz);
}

[numthreads(HF_FLUID_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int dt            = gParams[0].cfg0.x;
    int groundY       = gParams[0].cfg0.y;
    int particleCount = gParams[0].cfg0.z;
    int sphereCount   = gParams[0].cfg0.w;
    int enabled       = gParams[0].cfg1.x;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    if (enabled == 0) return;

    FluidParticle p = gParticles[i];
    bool isStatic = (p.flags & HF_FLUID_STATIC) != 0u;

    // (4) derive velocity from the net position change: vel = (pos − prev) / dt (static skips).
    if (!isStatic && dt != 0) {
        p.vx = fxdiv(p.px - p.prx, dt);
        p.vy = fxdiv(p.py - p.pry, dt);
        p.vz = fxdiv(p.pz - p.prz, dt);
    }

    // (5a) CollidePlane: clamp to the ground (static particles ARE clamped — a fallen static is raised).
    if (p.py < groundY) p.py = groundY;

    // (5b) CollideSpheres: project out of every static sphere (static particles skip — they hold).
    if (!isStatic) {
        for (int s = 0; s < sphereCount; ++s) {
            SphereCollider sc = spheres[s];
            int dx = p.px - sc.cx, dy = p.py - sc.cy, dz = p.pz - sc.cz;
            int ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy, az = dz < 0 ? -dz : dz;
            if (ax > sc.radius || ay > sc.radius || az > sc.radius) continue;   // AABB reject
            int dist = FxLength3(dx, dy, dz);
            if (dist >= sc.radius) continue;                                    // outside the sphere
            int nx, ny, nz;
            if (dist == 0) { nx = 0; ny = HF_FLUID_ONE; nz = 0; }               // FxNormalize +Y fallback
            else { nx = fxdiv(dx, dist); ny = fxdiv(dy, dist); nz = fxdiv(dz, dist); }
            p.px = sc.cx + fxmul(nx, sc.radius);                                // snap to the surface
            p.py = sc.cy + fxmul(ny, sc.radius);
            p.pz = sc.cz + fxmul(nz, sc.radius);
        }
    }

    gParticles[i] = p;   // per-particle independent write (NO atomics)
}
