// Slice CG1 — Deterministic Rigid<->Grain Coupling: the BODY->GRAIN grid-hash QUERY per-body GATHER-EMIT
// compute pass (the 3rd of the CG1 query count->scan->emit; the CP1 couple_body_emit twin, over the GR2 GRAIN
// cell table). ONE thread per BODY i (i < bodyCount). The thread re-scans the SAME BodyAabb cell RANGE in the
// SAME fixed order, reads its write base bodyStart[i] (the CG1 prefix-sum), and EMITS each accepted grain j
// into bodyGrains[] at bodyStart[i] + local++. Each body writes into its OWN DISJOINT [bodyStart[i],
// bodyStart[i]+count[i]) range -> race-free, NO atomics (the per-body-disjoint pattern). The order is ascending
// cell (cz,cy,cx) then ascending grain index j (cellGrains is ascending-index per cell from the GR2 cell-table
// emit) -> VERBATIM the CPU couple_grain.h::GatherBodyGrains emit order. enabled=0 -> emit nothing (bodyGrains
// stays the pre-cleared upload).
//
// WHY BIT-IDENTICAL to the CPU couple_grain.h::GatherBodyGrains (the make-or-break): every value written is
// PURE INT32 — the same per-axis BodyGrainAccept over the same host-snapped positions, the same cell-range walk
// in the same fixed order, and each body's range is disjoint so a thread race CANNOT change any byte. A
// divergence vs the header is exactly what the host's GPU==CPU memcmp (bodyGrains) catches. INT32 only ->
// MSL-gens natively on Metal.
//
// Buffers (storage, bound at compute bindings 0..6; on Metal these land at buffer(0..6)):
//   b0 gGrains    : the Q16.16 GrainParticle array (48-byte std430 ints), READ (pos).
//   b1 cellStart  : the reused GR2 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellGrains : the reused GR2 cell-table grain indices grouped by cell, READ.
//   b3 gBodies    : the body array (pos.xyz, radius — 16-byte std430 ints), READ.
//   b4 bodyStart  : the CG1 query prefix-sum write base (bodyCount+1), READ.
//   b5 bodyGrains : the output gathered-grain j-index array grouped by body, WRITE (pre-cleared).
//   b6 gParams    : the CGrainParams, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_CGRAIN_THREADS 64

struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

struct CGrainBody {
    int  bx, by, bz;
    int  radius;
};

struct CGrainParams {
    int4 grid;   // x=hSearch, y=cellMinX, z=cellMinY, w=cellMinZ
    int4 dim;    // x=gridDimX, y=gridDimY, z=gridDimZ, w=bodyCount
    int4 cfg;    // x=cellCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle> gGrains    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          cellStart  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>          cellGrains : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<CGrainBody>    gBodies    : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>          bodyStart  : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<uint>          bodyGrains : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<CGrainParams>  gParams    : register(u6);

int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

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
    if (enabled == 0) return;

    CGrainBody b = gBodies[i];
    int loX = FloorDiv(b.bx - b.radius, h), hiX = FloorDiv(b.bx + b.radius, h);
    int loY = FloorDiv(b.by - b.radius, h), hiY = FloorDiv(b.by + b.radius, h);
    int loZ = FloorDiv(b.bz - b.radius, h), hiZ = FloorDiv(b.bz + b.radius, h);

    uint base  = bodyStart[i];   // this body's disjoint write base (from the prefix-sum)
    uint local = 0u;
    for (int cz = loZ; cz <= hiZ; ++cz)
    for (int cy = loY; cy <= hiY; ++cy)
    for (int cx = loX; cx <= hiX; ++cx) {
        if (cx < cellMinX || cx >= cellMinX + gdx) continue;
        if (cy < cellMinY || cy >= cellMinY + gdy) continue;
        if (cz < cellMinZ || cz >= cellMinZ + gdz) continue;
        uint cell = (uint)(((cz - cellMinZ) * gdy + (cy - cellMinY)) * gdx + (cx - cellMinX));
        uint s0 = cellStart[cell], s1 = cellStart[cell + 1u];
        for (uint s = s0; s < s1; ++s) {
            uint j = cellGrains[s];
            GrainParticle g = gGrains[j];
            if (BodyGrainAccept(b.bx, b.by, b.bz, b.radius, g.px, g.py, g.pz)) {
                bodyGrains[base + local] = j;
                local += 1u;
            }
        }
    }
}
