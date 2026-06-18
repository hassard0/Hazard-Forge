// Slice CG1 — Deterministic Rigid<->Grain Coupling: the BODY->GRAIN grid-hash QUERY per-body GATHER-COUNT
// compute pass (the 1st of the CG1 query count->scan->emit; the CP1 couple_body_count twin, over the GR2 GRAIN
// cell table instead of the FL2 fluid cell table). ONE thread per BODY i (i < bodyCount). The thread computes
// the body's BodyAabb cell RANGE ([CellOf(pos - radius) .. CellOf(pos + radius)] at cell-size hSearch — NOT a
// fixed 27-cell stencil, since a body radius is typically MANY grain cells wide), and for each grain j in
// those cells (read from the reused GR2 cell table cellStart/cellGrains) counts it iff the PURE INT32 per-axis
// BodyGrainAccept (|body.pos.axis - g.pos.axis| < body.radius on every axis) passes -> perBodyCount[i]. NO
// atomics (each thread writes its OWN perBodyCount[i], a per-body independent count). enabled=0 -> write 0.
//
// WHY BIT-IDENTICAL to the CPU couple_grain.h::CountBodyGrains (the make-or-break): BodyGrainAccept is a
// per-axis integer subtract + abs + compare — PURE INT32, NO products, NO int64, NO sqrt (the exact radial
// sphere cull is DEFERRED to CG2's support / CG3's displacement; the box-candidate set is correct since the
// support/displacement impulse is a no-op beyond the sphere). The cell-range walk is the same fixed (cz,cy,cx)
// ascending order over the same GR2 cell table. So this MSL-generates NATIVELY on Metal (unlike the int64
// fxmul integrators) -> the host's GPU==CPU memcmp (perBodyCount) catches divergence.
//
// Buffers (storage, bound at compute bindings 0..5; on Metal these land at buffer(0..5)):
//   b0 gGrains      : the Q16.16 GrainParticle array (48-byte std430 ints), READ (pos).
//   b1 cellStart    : the reused GR2 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellGrains   : the reused GR2 cell-table grain indices grouped by cell, READ.
//   b3 gBodies      : the body array (pos.xyz, radius — 16-byte std430 ints), READ.
//   b4 perBodyCount : one uint per body (the gathered-grain count), WRITE.
//   b5 gParams      : the CGrainParams (hSearch, cellMin.xyz, gridDim.xyz, bodyCount, cellCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_CGRAIN_THREADS 64

// std430 GrainParticle mirror (engine/sim/grain.h::GrainParticle): 12 x 4-byte = 48 bytes (memcmp-able).
struct GrainParticle {
    int  px, py, pz;     // Q16.16 current position
    int  prx, pry, prz;  // Q16.16 previous position
    int  vx, vy, vz;     // Q16.16 velocity
    int  invMass;        // Q16.16 inverse mass
    int  radius;         // Q16.16 grain radius
    uint flags;          // bit0 = STATIC
};

// std430 CGrainBody mirror (the host CGrainBodyGpu upload): pos.xyz + radius = 4 x int32 (16 bytes). The
// query reads ONLY pos + radius from the fpx::FxBody (CG2-CG6 will append vel/orient as needed).
struct CGrainBody {
    int  bx, by, bz;   // Q16.16 body position
    int  radius;       // Q16.16 body radius (the per-axis box half-extent)
};

// std430 cgrain params (the C++ CGrainParams upload mirror).
//   grid : x=hSearch (Q16.16 cell size), y=cellMinX, z=cellMinY, w=cellMinZ
//   dim  : x=gridDimX, y=gridDimY, z=gridDimZ, w=bodyCount
//   cfg  : x=cellCount, y=enabled, z=unused, w=unused
struct CGrainParams {
    int4 grid;
    int4 dim;
    int4 cfg;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle> gGrains      : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          cellStart    : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>          cellGrains   : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<CGrainBody>    gBodies      : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>          perBodyCount : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<CGrainParams>  gParams      : register(u5);

// FloorDiv(n, d): deterministic FLOOR division for positive divisor d, ANY-sign n — VERBATIM
// engine/sim/fpx.h::FloorDiv. Pure int32 (no int64).
int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

// BodyGrainAccept(bx,by,bz,radius, gx,gy,gz): the PURE INT32 per-axis |dx| < radius box test — VERBATIM
// couple_grain.h::BodyGrainAccept. NO products, NO int64, NO sqrt.
bool BodyGrainAccept(int bx, int by, int bz, int radius, int gx, int gy, int gz) {
    int dx = bx - gx; if (dx < 0) dx = -dx;
    int dy = by - gy; if (dy < 0) dy = -dy;
    int dz = bz - gz; if (dz < 0) dz = -dz;
    return dx < radius && dy < radius && dz < radius;
}

[numthreads(HF_CGRAIN_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gdx = gParams[0].dim.x, gdy = gParams[0].dim.y, gdz = gParams[0].dim.z;
    int bodyCount = gParams[0].dim.w;
    int enabled   = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= bodyCount) return;
    if (enabled == 0) { perBodyCount[i] = 0u; return; }

    CGrainBody b = gBodies[i];
    // The body's BodyAabb [pos - radius, pos + radius], quantised to the grain cell range (FloorDiv per axis).
    int loX = FloorDiv(b.bx - b.radius, h), hiX = FloorDiv(b.bx + b.radius, h);
    int loY = FloorDiv(b.by - b.radius, h), hiY = FloorDiv(b.by + b.radius, h);
    int loZ = FloorDiv(b.bz - b.radius, h), hiZ = FloorDiv(b.bz + b.radius, h);

    uint c = 0u;
    for (int cz = loZ; cz <= hiZ; ++cz)
    for (int cy = loY; cy <= hiY; ++cy)
    for (int cx = loX; cx <= hiX; ++cx) {
        // Skip cells outside the bounded grid (the body may overhang the grain AABB).
        if (cx < cellMinX || cx >= cellMinX + gdx) continue;
        if (cy < cellMinY || cy >= cellMinY + gdy) continue;
        if (cz < cellMinZ || cz >= cellMinZ + gdz) continue;
        uint cell = (uint)(((cz - cellMinZ) * gdy + (cy - cellMinY)) * gdx + (cx - cellMinX));
        uint s0 = cellStart[cell], s1 = cellStart[cell + 1u];
        for (uint s = s0; s < s1; ++s) {
            uint j = cellGrains[s];
            GrainParticle g = gGrains[j];
            if (BodyGrainAccept(b.bx, b.by, b.bz, b.radius, g.px, g.py, g.pz)) c += 1u;
        }
    }
    perBodyCount[i] = c;   // per-body independent count (NO atomics)
}
