// Slice CP1 — Deterministic Rigid<->Fluid Coupling: the BODY->FLUID grid-hash QUERY per-body GATHER-COUNT
// compute pass (the 1st of the CP1 query count->scan->emit; the GR2 grain_neighbor_count twin, per BODY
// instead of per grain). ONE thread per BODY i (i < bodyCount). The thread computes the body's BodyAabb cell
// RANGE ([CellOf(pos - radius) .. CellOf(pos + radius)] at cell-size h — NOT a fixed 27-cell stencil, since a
// body radius is typically MANY fluid cells wide), and for each fluid particle j in those cells (read from
// the reused FL2 cell table cellStart/cellParticles) counts it iff the PURE INT32 per-axis BodyParticleAccept
// (|body.pos.axis - p.pos.axis| < body.radius on every axis) passes -> perBodyCount[i]. NO atomics (each
// thread writes its OWN perBodyCount[i], a per-body independent count). enabled=0 -> write 0.
//
// WHY BIT-IDENTICAL to the CPU couple.h::CountBodyParticles (the make-or-break): BodyParticleAccept is a
// per-axis integer subtract + abs + compare — PURE INT32, NO products, NO int64, NO sqrt (the exact radial
// sphere cull is DEFERRED to CP2's force; the box-candidate set is correct since CP2's impulse is a no-op
// beyond the sphere). The cell-range walk is the same fixed (cz,cy,cx) ascending order over the same cell
// table. So this MSL-generates NATIVELY on Metal (unlike the int64 fxmul integrators) -> the host's GPU==CPU
// memcmp (perBodyCount) catches divergence.
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gParticles   : the Q16.16 FluidParticle array (44-byte std430 ints), READ (pos).
//   b1 cellStart    : the reused FL2 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellParticles: the reused FL2 cell-table particle indices grouped by cell, READ.
//   b3 gBodies      : the body array (pos.xyz, radius — 16-byte std430 ints), READ.
//   b4 perBodyCount : one uint per body (the gathered-particle count), WRITE.
//   b5 gParams      : the CoupleParams (h, cellMin.xyz, gridDim.xyz, bodyCount, cellCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_COUPLE_THREADS 64

// std430 FluidParticle mirror (engine/sim/fluid.h::FluidParticle): 11 x 4-byte = 44 bytes (memcmp-able).
struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

// std430 CoupleBody mirror (the host CoupleBodyGpu upload): pos.xyz + radius = 4 x int32 (16 bytes). The
// query reads ONLY pos + radius from the fpx::FxBody (CP2-CP6 will append vel/orient as needed).
struct CoupleBody {
    int  bx, by, bz;   // Q16.16 body position
    int  radius;       // Q16.16 body radius (the per-axis box half-extent)
};

// std430 couple params (the C++ CoupleParams upload mirror).
//   grid : x=h (Q16.16 cell size), y=cellMinX, z=cellMinY, w=cellMinZ
//   dim  : x=gridDimX, y=gridDimY, z=gridDimZ, w=bodyCount
//   cfg  : x=cellCount, y=enabled, z=unused, w=unused
struct CoupleParams {
    int4 grid;
    int4 dim;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle> gParticles    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          cellStart     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>          cellParticles : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<CoupleBody>    gBodies       : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>          perBodyCount  : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<CoupleParams>  gParams       : register(u5);

// FloorDiv(n, d): deterministic FLOOR division for positive divisor d, ANY-sign n — VERBATIM
// engine/sim/fpx.h::FloorDiv. Pure int32 (no int64).
int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

// BodyParticleAccept(bx,by,bz,radius, px,py,pz): the PURE INT32 per-axis |dx| < radius box test — VERBATIM
// couple.h::BodyParticleAccept. NO products, NO int64, NO sqrt.
bool BodyParticleAccept(int bx, int by, int bz, int radius, int px, int py, int pz) {
    int dx = bx - px; if (dx < 0) dx = -dx;
    int dy = by - py; if (dy < 0) dy = -dy;
    int dz = bz - pz; if (dz < 0) dz = -dz;
    return dx < radius && dy < radius && dz < radius;
}

[numthreads(HF_COUPLE_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gdx = gParams[0].dim.x, gdy = gParams[0].dim.y, gdz = gParams[0].dim.z;
    int bodyCount = gParams[0].dim.w;
    int enabled   = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= bodyCount) return;
    if (enabled == 0) { perBodyCount[i] = 0u; return; }

    CoupleBody b = gBodies[i];
    // The body's BodyAabb [pos - radius, pos + radius], quantised to the fluid cell range (FloorDiv per axis).
    int loX = FloorDiv(b.bx - b.radius, h), hiX = FloorDiv(b.bx + b.radius, h);
    int loY = FloorDiv(b.by - b.radius, h), hiY = FloorDiv(b.by + b.radius, h);
    int loZ = FloorDiv(b.bz - b.radius, h), hiZ = FloorDiv(b.bz + b.radius, h);

    uint c = 0u;
    for (int cz = loZ; cz <= hiZ; ++cz)
    for (int cy = loY; cy <= hiY; ++cy)
    for (int cx = loX; cx <= hiX; ++cx) {
        // Skip cells outside the bounded grid (the body may overhang the particle AABB).
        if (cx < cellMinX || cx >= cellMinX + gdx) continue;
        if (cy < cellMinY || cy >= cellMinY + gdy) continue;
        if (cz < cellMinZ || cz >= cellMinZ + gdz) continue;
        uint cell = (uint)(((cz - cellMinZ) * gdy + (cy - cellMinY)) * gdx + (cx - cellMinX));
        uint s0 = cellStart[cell], s1 = cellStart[cell + 1u];
        for (uint s = s0; s < s1; ++s) {
            uint j = cellParticles[s];
            FluidParticle p = gParticles[j];
            if (BodyParticleAccept(b.bx, b.by, b.bz, b.radius, p.px, p.py, p.pz)) c += 1u;
        }
    }
    perBodyCount[i] = c;   // per-body independent count (NO atomics)
}
