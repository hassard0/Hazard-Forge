// Slice CP3 — Deterministic Rigid<->Fluid Coupling: the FLUID REACTION / DISPLACEMENT pass (body->fluid, the
// Newton's-3rd-law HALF of CP2; the 3rd slice of FLAGSHIP #11). ONE thread per FLUID PARTICLE i
// (i < particleCount) — multi-thread OVER particles, each thread SERIALLY iterates the TINY body list (fixed
// order). For each DYNAMIC body that CONTAINS the particle (dist < b.radius), it (1) accumulates the
// surface-snap push into a LOCAL accumulator (the per-thread Jacobi dp — each thread reads the iteration-start
// BODY state read-only + writes ONLY its own particle) and (2) applies the equal-opposite DRAG-REACTION
// velocity impulse, then writes pos += accum + the updated vel. Per-particle DISJOINT -> NO race, NO atomics,
// [numthreads(64,1,1)] MULTI-THREAD, NO single-thread/TDR (the FL4/GR3 win). This is the GR3
// grain_collide.comp mold (project a particle out of a sphere) with fpx::FxBody as the sphere + the
// drag-reaction term. Copied VERBATIM from couple.h::ApplyBodyToFluid.
//
// INTEGER WIDTH: the FxLength/FxNormalize (FxISqrt) + the drag-reaction fxmul use int64_t -> VULKAN-SPIR-V-ONLY
// (DXC compiles int64; glslc cannot parse int64 in HLSL) -> NOT in the Metal hf_gen_msl list; the Metal
// --couple-displace showcase runs the CPU couple::ApplyBodyToFluid (byte-identical by construction, the
// couple_buoyancy.comp/grain_collide.comp/fluid_dp.comp convention). The CP1 query passes stay int32 MSL-native.
//
// Buffers (storage, Vulkan-only):
//   b0 gParticles : the FluidParticleGpu array (44 bytes), READ+WRITE (pos, vel; prev/invMass/flags read).
//   b1 gBodies    : the CoupleBodyGpu array (pos.xyz, vel.xyz, invMass, flags, radius — 36 bytes), READ.
//   b2 gParams    : the CoupleDisplaceParams (particleCount, bodyCount, enabled, dt), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_CP_THREADS 64
#define HF_CP_FRAC 16
#define HF_CP_ONE 65536
#define HF_CP_DYNAMIC 1u        // == fpx::kFlagDynamic (bit0)
#define HF_CP_STATIC 1u         // == fluid::kFlagStatic (bit0)

// The fluid particle pack (std430, 44 bytes — matches FluidParticleGpu; pos + vel READ+WRITE).
struct FluidParticleGpu {
    int  px, py, pz, prx, pry, prz, vx, vy, vz, invMass;
    uint flags;
};

// The body pack (std430): pos.xyz, vel.xyz, invMass, flags, radius = 9 x 32-bit (36 bytes; == CoupleBodyGpu).
struct CoupleBodyGpu {
    int  px, py, pz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
    int  radius;
};

// CoupleDisplaceParams (std430). cfg {particleCount, bodyCount, enabled, dt}.
struct CoupleDisplaceParams { int4 cfg; };

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticleGpu>     gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<CoupleBodyGpu>        gBodies    : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<CoupleDisplaceParams> gParams    : register(u2);

// fxmul — VERBATIM fpx.h::fxmul (int64 intermediate, arithmetic right shift).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_CP_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_CP_FRAC) / (int64_t)b); }

// FxISqrt — VERBATIM fpx.h::FxISqrt (int64 binary integer sqrt).
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

[numthreads(HF_CP_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int particleCount = gParams[0].cfg.x;
    int bodyCount     = gParams[0].cfg.y;
    int enabled       = gParams[0].cfg.z;
    int dt            = gParams[0].cfg.w;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    if (enabled == 0) return;

    FluidParticleGpu p = gParticles[i];
    if ((p.flags & HF_CP_STATIC) != 0u) return;          // boundary particle -> dp 0, vel untouched

    // kDragReaction == couple.h::kDragReaction (host-snapped 1.5 in Q16.16 == 98304); a compile-time constant.
    const int kDragReaction = (int)(1.5 * (double)HF_CP_ONE + 0.5);

    // The per-thread Jacobi accumulator (this particle's Δp — reads the iteration-start body state read-only).
    int accumX = 0, accumY = 0, accumZ = 0;
    for (int bi = 0; bi < bodyCount; ++bi) {
        CoupleBodyGpu b = gBodies[bi];
        if ((b.flags & HF_CP_DYNAMIC) == 0u) continue;   // non-dynamic body -> holds (the pinned case)
        // d = p.pos − b.pos ; dist = |d| ; inside iff dist < b.radius.
        int dx = p.px - b.px, dy = p.py - b.py, dz = p.pz - b.pz;
        int dist = FxLength3(dx, dy, dz);
        if (dist >= b.radius) continue;                  // outside the body sphere -> no displacement

        // (1) POSITIONAL DISPLACEMENT: snap the particle to the body surface (FxNormalize -> +Y fallback).
        int nx, ny, nz;
        if (dist == 0) { nx = 0; ny = HF_CP_ONE; nz = 0; }                   // FxNormalize +Y fallback
        else { nx = fxdiv(dx, dist); ny = fxdiv(dy, dist); nz = fxdiv(dz, dist); }
        int surfX = b.px + fxmul(nx, b.radius);          // the surface point along the outward normal
        int surfY = b.py + fxmul(ny, b.radius);
        int surfZ = b.pz + fxmul(nz, b.radius);
        accumX += surfX - p.px;                          // into the Jacobi accumulator
        accumY += surfY - p.py;
        accumZ += surfZ - p.pz;

        // (2) DRAG REACTION: the body imparts momentum to the fluid (the equal-opposite of CP2's drag).
        p.vx += fxmul(fxmul(kDragReaction, b.vx - p.vx), dt);
        p.vy += fxmul(fxmul(kDragReaction, b.vy - p.vy), dt);
        p.vz += fxmul(fxmul(kDragReaction, b.vz - p.vz), dt);
    }
    // Apply pos += accum (Jacobi — per-particle disjoint write, NO atomics).
    p.px += accumX; p.py += accumY; p.pz += accumZ;

    gParticles[i] = p;
}
