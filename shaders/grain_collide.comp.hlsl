// Slice GR3 — Deterministic GPU Granular/Sand: the VELOCITY-UPDATE + RADIUS-AWARE COLLISION pass (the 3rd
// slice of FLAGSHIP #10, the FL4 fluid_collide.comp twin). ONE thread per GRAIN i, run ONCE after the
// `iters` Jacobi contact iterations. It (1) derives the velocity from the NET position change
// vel = (pos − prev)/dt (the FL4 PBF velocity update — prev was the predicted-position anchor snapshotted in
// the GR1 integrate), then (2) projects the grain out of the ground plane (RADIUS-AWARE: pos.y >= groundY +
// radius) + the STATIC sphere colliders (RADIUS-AWARE: the grain CENTRE snaps to sphereRadius + grainRadius,
// the surfaces touch). Per-grain independent (each thread reads the read-only sphere set + writes only its
// OWN grain) -> NO race, NO atomics, deterministic regardless of thread order. STATIC grains skip the
// velocity update + sphere projection (the plane clamp still raises a fallen static, matching
// CollideGrainPlane). Copied VERBATIM from grain.h::StepGrainContact's velocity derivation +
// CollideGrainPlane + CollideGrainSphere.
//
// INTEGER WIDTH: the velocity fxdiv + FxLength/FxNormalize use int64_t -> VULKAN-SPIR-V-ONLY (NOT in
// hf_gen_msl); the Metal --grain-contact showcase runs the CPU grain::StepGrainContact (byte-identical by
// construction, the fluid_collide.comp/cloth_collide.comp convention).
//
// Buffers (storage, Vulkan-only):
//   b0 gGrains : the Q16.16 GrainParticle array (48 bytes), READ+WRITE (pos, prev, vel, radius, flags).
//   b1 spheres : the static GrainSphereCollider set (center.xyz, radius — 4 x int32), READ.
//   b2 gParams : the GrainSolveParams (dt, groundY, grainCount, sphereCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_GRAIN_THREADS 64
#define HF_GRAIN_FRAC 16
#define HF_GRAIN_ONE 65536
#define HF_GRAIN_STATIC 1u   // == grain.h::kFlagStatic (bit0)

struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

struct SphereCollider { int cx, cy, cz; int radius; };   // std430 16 bytes (== grain.h::GrainSphereCollider)

// GrainSolveParams (std430). cfg0 {dt, groundY, grainCount, sphereCount}, cfg1 {enabled, _, _, _}.
struct GrainSolveParams {
    int4 cfg0;
    int4 cfg1;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle>   gGrains : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<SphereCollider>  spheres : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<GrainSolveParams> gParams : register(u2);

int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_GRAIN_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_GRAIN_FRAC) / (int64_t)b); }
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

[numthreads(HF_GRAIN_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int dt          = gParams[0].cfg0.x;
    int groundY     = gParams[0].cfg0.y;
    int grainCount  = gParams[0].cfg0.z;
    int sphereCount = gParams[0].cfg0.w;
    int enabled     = gParams[0].cfg1.x;

    uint i = gid.x;
    if ((int)i >= grainCount) return;
    if (enabled == 0) return;

    GrainParticle p = gGrains[i];
    bool isStatic = (p.flags & HF_GRAIN_STATIC) != 0u;

    // (4) derive velocity from the net position change: vel = (pos − prev) / dt (static skips).
    if (!isStatic && dt != 0) {
        p.vx = fxdiv(p.px - p.prx, dt);
        p.vy = fxdiv(p.py - p.pry, dt);
        p.vz = fxdiv(p.pz - p.prz, dt);
    }

    // (5a) CollideGrainPlane: clamp the SURFACE to the ground (pos.y >= groundY + radius; static ARE clamped).
    int restY = groundY + p.radius;
    if (p.py < restY) p.py = restY;

    // (5b) CollideGrainSpheres: project the CENTRE out to sphereR + grainR (static grains skip — they hold).
    if (!isStatic) {
        for (int s = 0; s < sphereCount; ++s) {
            SphereCollider sc = spheres[s];
            int surf = sc.radius + p.radius;                                    // surfaces-touch distance
            int dx = p.px - sc.cx, dy = p.py - sc.cy, dz = p.pz - sc.cz;
            int ax = dx < 0 ? -dx : dx, ay = dy < 0 ? -dy : dy, az = dz < 0 ? -dz : dz;
            if (ax > surf || ay > surf || az > surf) continue;                  // AABB reject
            int dist = FxLength3(dx, dy, dz);
            if (dist >= surf) continue;                                         // outside the expanded sphere
            int nx, ny, nz;
            if (dist == 0) { nx = 0; ny = HF_GRAIN_ONE; nz = 0; }               // FxNormalize +Y fallback
            else { nx = fxdiv(dx, dist); ny = fxdiv(dy, dist); nz = fxdiv(dz, dist); }
            p.px = sc.cx + fxmul(nx, surf);                                     // snap the centre to surf
            p.py = sc.cy + fxmul(ny, surf);
            p.pz = sc.cz + fxmul(nz, surf);
        }
    }

    gGrains[i] = p;   // per-grain independent write (NO atomics)
}
