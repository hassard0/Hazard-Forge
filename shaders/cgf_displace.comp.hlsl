// Slice GF3 — Deterministic Grain<->Fluid Coupling: the CONTACT REACTION / DISPLACEMENT pass (grain->fluid, the
// Newton's-3rd-law HALF of GF2; the 3rd slice of FLAGSHIP #13). ONE thread per FLUID particle i (i < fluidCount)
// — multi-thread OVER fluid particles, each thread SERIALLY iterates its GF1 fgNeighbors grain list (the FIXED
// ascending GF1 emit order). For each grain that CONTAINS the fluid particle (dist < g.radius), it (1)
// accumulates the surface-snap push into a LOCAL accumulator (the per-thread Jacobi dp — each thread reads the
// iteration-start GRAIN state read-only + writes ONLY its own fluid particle) and (2) applies the equal-opposite
// DRAG-REACTION velocity impulse, then writes pos += accum + the updated vel. Per-fluid DISJOINT -> NO race, NO
// atomics, [numthreads(64,1,1)] MULTI-THREAD, NO single-thread/TDR (the GR3/CG3/CP3 win). This is the GR3
// grain_collide.comp mold (project a point out of a sphere) with the GRAIN as the sphere (surf = g.radius; fluid
// particles are points) + the drag-reaction term. Copied VERBATIM from couple_gf.h::ApplyGrainsToFluid; the
// positional push == grain.h::CollideGrainSphere with the grain as the sphere.
//
// INTEGER WIDTH: the FxLength/FxNormalize (FxISqrt) + the drag-reaction fxmul use int64_t -> VULKAN-SPIR-V-ONLY
// (DXC compiles int64; glslc cannot parse int64 in HLSL) -> NOT in the Metal hf_gen_msl list; the Metal
// --cgf-displace showcase runs the CPU cgf::ApplyGrainsToFluid (byte-identical by construction, the
// cgf_buoyancy.comp/cgrain_displace.comp convention). The GF1 cross-query passes stay int32 MSL-native.
//
// Buffers (storage, Vulkan-only):
//   b0 gFluid  : the FluidParticle array (44 bytes), READ+WRITE (pos, vel; prev/invMass/flags read).
//   b1 fgStart : the GF1 CSR offsets (fluidCount+1), READ.
//   b2 fgNbr   : the GF1 gathered grain indices grouped by fluid (ascending), READ.
//   b3 gGrains : the GrainParticle array (48 bytes; pos/vel/radius read), READ.
//   b4 gParams : the CGFDisplaceParams (fluidCount, enabled, dt, _), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_CGF_THREADS 64
#define HF_CGF_FRAC 16
#define HF_CGF_ONE 65536
#define HF_CGF_STATIC 1u        // == fluid::kFlagStatic (bit0)

// The fluid pack (std430, 44 bytes — matches FluidParticleGpu; pos + vel READ+WRITE).
struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

// The grain pack (std430, 48 bytes — matches GrainParticleGpu; pos + vel + radius read).
struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

// CGFDisplaceParams (std430). cfg {fluidCount, enabled, dt, _}.
struct CGFDisplaceParams { int4 cfg; };

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle>      gFluid  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>              fgStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>              fgNbr   : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<GrainParticle>      gGrains : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<CGFDisplaceParams> gParams : register(u4);

// fxmul/fxdiv — VERBATIM fpx.h::fxmul / fpx.h::fxdiv (int64 intermediate, arithmetic right shift).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_CGF_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_CGF_FRAC) / (int64_t)b); }

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

[numthreads(HF_CGF_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int fluidCount = gParams[0].cfg.x;
    int enabled    = gParams[0].cfg.y;
    int dt         = gParams[0].cfg.z;

    uint i = gid.x;
    if ((int)i >= fluidCount) return;
    if (enabled == 0) return;

    FluidParticle p = gFluid[i];
    if ((p.flags & HF_CGF_STATIC) != 0u) return;          // boundary fluid -> dp 0, vel untouched

    // kDragReaction == couple_gf.h::kDragReaction (host-snapped 1.5 in Q16.16 == 98304); compile-time const.
    const int kDragReaction = (int)(1.5 * (double)HF_CGF_ONE + 0.5);

    uint s0 = fgStart[i], s1 = fgStart[i + 1u];

    // The per-thread Jacobi accumulator (this fluid particle's Δp — reads the iteration-start grain state).
    int accumX = 0, accumY = 0, accumZ = 0;
    for (uint s = s0; s < s1; ++s) {
        GrainParticle g = gGrains[fgNbr[s]];
        // d = p.pos − g.pos ; dist = |d| ; surf = g.radius ; inside iff dist < surf.
        int dx = p.px - g.px, dy = p.py - g.py, dz = p.pz - g.pz;   // fluid relative to the grain (outward)
        int dist = FxLength3(dx, dy, dz);
        int surf = g.radius;                              // the grain exclusion radius (fluid particles are points)
        if (dist >= surf) continue;                       // outside the grain sphere -> no push

        // (1) POSITIONAL DISPLACEMENT: snap the fluid particle to the grain surface (FxNormalize -> +Y fallback).
        int nx, ny, nz;
        if (dist == 0) { nx = 0; ny = HF_CGF_ONE; nz = 0; }                  // FxNormalize +Y fallback
        else { nx = fxdiv(dx, dist); ny = fxdiv(dy, dist); nz = fxdiv(dz, dist); }
        int surfX = g.px + fxmul(nx, surf);               // the surface point along the outward normal
        int surfY = g.py + fxmul(ny, surf);
        int surfZ = g.pz + fxmul(nz, surf);
        accumX += surfX - p.px;                           // into the Jacobi accumulator
        accumY += surfY - p.py;
        accumZ += surfZ - p.pz;

        // (2) DRAG REACTION: the grain imparts momentum to the fluid (the equal-opposite of GF2's drag).
        p.vx += fxmul(fxmul(kDragReaction, g.vx - p.vx), dt);
        p.vy += fxmul(fxmul(kDragReaction, g.vy - p.vy), dt);
        p.vz += fxmul(fxmul(kDragReaction, g.vz - p.vz), dt);
    }
    // Apply pos += accum (Jacobi — per-fluid disjoint write, NO atomics).
    p.px += accumX; p.py += accumY; p.pz += accumZ;

    gFluid[i] = p;
}
