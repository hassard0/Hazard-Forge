// Slice CP1 — Deterministic Rigid<->Fluid Coupling: the BODY->FLUID grid-hash QUERY per-body GATHER-EMIT
// compute pass (the 3rd of the CP1 query count->scan->emit; the GR2 grain_neighbor_emit twin, per BODY). ONE
// thread per BODY i (i < bodyCount). The thread re-scans the SAME BodyAabb cell RANGE in the SAME fixed order,
// reads its write base bodyStart[i] (the CP1 prefix-sum), and EMITS each accepted particle j into
// bodyParticles[] at bodyStart[i] + local++. Each body writes into its OWN DISJOINT [bodyStart[i],
// bodyStart[i]+count[i]) range -> race-free, NO atomics (the per-body-disjoint pattern). The order is
// ascending cell (cz,cy,cx) then ascending particle index j (cellParticles is ascending-index per cell from
// the FL2 cell-table emit) -> VERBATIM the CPU couple.h::GatherBodyParticles emit order. enabled=0 -> emit
// nothing (bodyParticles stays the pre-cleared upload).
//
// WHY BIT-IDENTICAL to the CPU couple.h::GatherBodyParticles (the make-or-break): every value written is PURE
// INT32 — the same per-axis BodyParticleAccept over the same host-snapped positions, the same cell-range walk
// in the same fixed order, and each body's range is disjoint so a thread race CANNOT change any byte. A
// divergence vs the header is exactly what the host's GPU==CPU memcmp (bodyParticles) catches. INT32 only ->
// MSL-gens natively on Metal.
//
// Buffers (storage, bound at compute bindings 0..6; on Metal these land at buffer(0..6)):
//   b0 gParticles   : the Q16.16 FluidParticle array (44-byte std430 ints), READ (pos).
//   b1 cellStart    : the reused FL2 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellParticles: the reused FL2 cell-table particle indices grouped by cell, READ.
//   b3 gBodies      : the body array (pos.xyz, radius — 16-byte std430 ints), READ.
//   b4 bodyStart    : the CP1 query prefix-sum write base (bodyCount+1), READ.
//   b5 bodyParticles: the output gathered-particle j-index array grouped by body, WRITE (pre-cleared).
//   b6 gParams      : the CoupleParams, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_COUPLE_THREADS 64

struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

struct CoupleBody {
    int  bx, by, bz;
    int  radius;
};

struct CoupleParams {
    int4 grid;   // x=h, y=cellMinX, z=cellMinY, w=cellMinZ
    int4 dim;    // x=gridDimX, y=gridDimY, z=gridDimZ, w=bodyCount
    int4 cfg;    // x=cellCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle> gParticles    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          cellStart     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>          cellParticles : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<CoupleBody>    gBodies       : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>          bodyStart     : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<uint>          bodyParticles : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<CoupleParams>  gParams       : register(u6);

int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

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
    if (enabled == 0) return;

    CoupleBody b = gBodies[i];
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
            uint j = cellParticles[s];
            FluidParticle p = gParticles[j];
            if (BodyParticleAccept(b.bx, b.by, b.bz, b.radius, p.px, p.py, p.pz)) {
                bodyParticles[base + local] = j;
                local += 1u;
            }
        }
    }
}
