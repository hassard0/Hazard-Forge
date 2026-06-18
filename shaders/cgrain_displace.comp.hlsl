// Slice CG3 — Deterministic Rigid<->Grain Coupling: the GRAIN REACTION / DISPLACEMENT pass (body->grain, the
// Newton's-3rd-law HALF of CG2; the 3rd slice of FLAGSHIP #12). ONE thread per GRAIN i (i < grainCount) —
// multi-thread OVER grains, each thread SERIALLY iterates the TINY body list (fixed order). For each DYNAMIC
// body that CONTAINS the grain (dist < b.radius + g.radius), it (1) accumulates the surface-snap push into a
// LOCAL accumulator (the per-thread Jacobi dp — each thread reads the iteration-start BODY state read-only +
// writes ONLY its own grain) and (2) applies the equal-opposite DRAG-REACTION velocity impulse, then writes
// pos += accum + the updated vel. Per-grain DISJOINT -> NO race, NO atomics, [numthreads(64,1,1)] MULTI-
// THREAD, NO single-thread/TDR (the GR3/CP3 win). This is the GR3 grain_collide.comp mold (project a grain
// out of a sphere) with fpx::FxBody as the sphere (surf = b.radius + g.radius) + the drag-reaction term.
// Copied VERBATIM from couple_grain.h::ApplyBodyToGrains; the positional push == grain.h::CollideGrainSphere(g,
// GrainSphereFromBody(b)).
//
// INTEGER WIDTH: the FxLength/FxNormalize (FxISqrt) + the drag-reaction fxmul use int64_t -> VULKAN-SPIR-V-ONLY
// (DXC compiles int64; glslc cannot parse int64 in HLSL) -> NOT in the Metal hf_gen_msl list; the Metal
// --cgrain-displace showcase runs the CPU cgrain::ApplyBodyToGrains (byte-identical by construction, the
// cgrain_support.comp/couple_displace.comp/grain_collide.comp convention). The CG1 query passes stay int32
// MSL-native.
//
// Buffers (storage, Vulkan-only):
//   b0 gGrains : the GrainParticleGpu array (48 bytes), READ+WRITE (pos, vel; prev/invMass/radius/flags read).
//   b1 gBodies : the CGrainSupportBody array (pos.xyz, vel.xyz, invMass, flags, radius — 36 bytes), READ.
//   b2 gParams : the CGrainDisplaceParams (grainCount, bodyCount, enabled, dt), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_CG_THREADS 64
#define HF_CG_FRAC 16
#define HF_CG_ONE 65536
#define HF_CG_DYNAMIC 1u        // == fpx::kFlagDynamic (bit0)
#define HF_CG_STATIC 1u         // == grain::kFlagStatic (bit0)

// The grain pack (std430, 48 bytes — matches GrainParticleGpu; pos + vel READ+WRITE).
struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

// The body pack (std430): pos.xyz, vel.xyz, invMass, flags, radius = 9 x 32-bit (36 bytes; == CGrainSupportBody).
struct CGrainSupportBody {
    int  px, py, pz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
    int  radius;
};

// CGrainDisplaceParams (std430). cfg {grainCount, bodyCount, enabled, dt}.
struct CGrainDisplaceParams { int4 cfg; };

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle>        gGrains : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<CGrainSupportBody>    gBodies : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<CGrainDisplaceParams> gParams : register(u2);

// fxmul/fxdiv — VERBATIM fpx.h::fxmul / fpx.h::fxdiv (int64 intermediate, arithmetic right shift).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_CG_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_CG_FRAC) / (int64_t)b); }

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

[numthreads(HF_CG_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int grainCount = gParams[0].cfg.x;
    int bodyCount  = gParams[0].cfg.y;
    int enabled    = gParams[0].cfg.z;
    int dt         = gParams[0].cfg.w;

    uint i = gid.x;
    if ((int)i >= grainCount) return;
    if (enabled == 0) return;

    GrainParticle g = gGrains[i];
    if ((g.flags & HF_CG_STATIC) != 0u) return;          // boundary grain -> dp 0, vel untouched

    // kDragReaction == couple_grain.h::kDragReaction (host-snapped 1.5 in Q16.16 == 98304); compile-time const.
    const int kDragReaction = (int)(1.5 * (double)HF_CG_ONE + 0.5);

    // The per-thread Jacobi accumulator (this grain's Δp — reads the iteration-start body state read-only).
    int accumX = 0, accumY = 0, accumZ = 0;
    for (int bi = 0; bi < bodyCount; ++bi) {
        CGrainSupportBody b = gBodies[bi];
        if ((b.flags & HF_CG_DYNAMIC) == 0u) continue;   // non-dynamic body -> holds (the pinned case)
        // d = g.pos − b.pos ; dist = |d| ; surf = b.radius + g.radius ; inside iff dist < surf.
        int dx = g.px - b.px, dy = g.py - b.py, dz = g.pz - b.pz;   // grain relative to the body (outward)
        int dist = FxLength3(dx, dy, dz);
        int surf = b.radius + g.radius;                  // the surfaces-touch distance (sphereR + grainR)
        if (dist >= surf) continue;                      // outside the (expanded) body sphere -> no push

        // (1) POSITIONAL DISPLACEMENT: snap the grain centre to the body surface (FxNormalize -> +Y fallback).
        int nx, ny, nz;
        if (dist == 0) { nx = 0; ny = HF_CG_ONE; nz = 0; }                  // FxNormalize +Y fallback
        else { nx = fxdiv(dx, dist); ny = fxdiv(dy, dist); nz = fxdiv(dz, dist); }
        int surfX = b.px + fxmul(nx, surf);              // the surface point along the outward normal
        int surfY = b.py + fxmul(ny, surf);
        int surfZ = b.pz + fxmul(nz, surf);
        accumX += surfX - g.px;                          // into the Jacobi accumulator
        accumY += surfY - g.py;
        accumZ += surfZ - g.pz;

        // (2) DRAG REACTION: the body imparts momentum to the grain (the equal-opposite of CG2's drag).
        g.vx += fxmul(fxmul(kDragReaction, b.vx - g.vx), dt);
        g.vy += fxmul(fxmul(kDragReaction, b.vy - g.vy), dt);
        g.vz += fxmul(fxmul(kDragReaction, b.vz - g.vz), dt);
    }
    // Apply pos += accum (Jacobi — per-grain disjoint write, NO atomics).
    g.px += accumX; g.py += accumY; g.pz += accumZ;

    gGrains[i] = g;
}
