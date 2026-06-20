// Slice BD2 — Deterministic GPU Crowds: the GRID-HASH NEIGHBOR LIST per-agent NEIGHBOR-EMIT compute pass (the
// 3rd of the BD2 neighbor-list count->scan->emit; the GR2 grain_neighbor_emit twin). ONE thread per AGENT i
// (i < agentCount). The thread re-scans the SAME 27-cell stencil in the SAME fixed order, reads its write base
// neighborStart[i] (the BD2 prefix-sum), and EMITS each accepted j into neighbors[] at neighborStart[i] +
// local++. Each agent writes into its OWN DISJOINT [neighborStart[i], neighborStart[i]+count[i]) range ->
// race-free, NO atomics (the per-agent-disjoint pattern). The order is ascending stencil-cell (dz,dy,dx) then
// ascending j (cellAgents is ascending-index per cell from the cell-table emit) -> VERBATIM the CPU
// boids.h::BuildBoidsNeighborList emit order. enabled=0 -> emit nothing (neighbors stays the pre-cleared
// upload).
//
// WHY BIT-IDENTICAL to the CPU boids.h::BuildBoidsNeighborList (the make-or-break): every value written is
// PURE INT32 — the same per-axis BoidsNeighborAccept over the same host-snapped positions, the same stencil
// walk in the same fixed order, and each agent's range is disjoint so a thread race CANNOT change any byte. A
// divergence vs the header is exactly what the host's GPU==CPU memcmp (neighbors) catches. INT32 only ->
// MSL-gens natively on Metal (a TRUE GPU pass on BOTH backends).
//
// Buffers (storage, bound at compute bindings 0..5; on Metal these land at buffer(0..5)):
//   b0 gAgents       : the Q16.16 Agent array (24-byte std430 ints), READ (pos).
//   b1 cellStart     : the BD2 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellAgents    : the BD2 cell-table agent indices grouped by cell, READ.
//   b3 neighborStart : the BD2 neighbor-list prefix-sum write base (agentCount+1), READ.
//   b4 neighbors     : the output neighbor j-index array grouped by i, WRITE (pre-cleared).
//   b5 gParams       : the BoidsGridParams, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_BOIDS_THREADS 64

struct Agent {
    int px, py, pz;
    int vx, vy, vz;
};

struct BoidsGridParams {
    int4 grid;   // x=radius, y=cellMinX, z=cellMinY, w=cellMinZ
    int4 dim;    // x=gridDimX, y=gridDimY, z=gridDimZ, w=agentCount
    int4 cfg;    // x=cellCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Agent>           gAgents       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            cellStart     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            cellAgents    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>            neighborStart : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>            neighbors     : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<BoidsGridParams> gParams       : register(u5);

int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

bool BoidsNeighborAccept(int ax, int ay, int az, int bx, int by, int bz, int radius) {
    int dx = ax - bx; if (dx < 0) dx = -dx;
    int dy = ay - by; if (dy < 0) dy = -dy;
    int dz = az - bz; if (dz < 0) dz = -dz;
    return dx < radius && dy < radius && dz < radius;
}

[numthreads(HF_BOIDS_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gdx = gParams[0].dim.x, gdy = gParams[0].dim.y, gdz = gParams[0].dim.z;
    int agentCount = gParams[0].dim.w;
    int enabled    = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= agentCount) return;
    if (enabled == 0) return;

    Agent pi = gAgents[i];
    int cix = FloorDiv(pi.px, h);
    int ciy = FloorDiv(pi.py, h);
    int ciz = FloorDiv(pi.pz, h);

    uint base  = neighborStart[i];   // this agent's disjoint write base (from the prefix-sum)
    uint local = 0u;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int ncx = cix + dx, ncy = ciy + dy, ncz = ciz + dz;
        if (ncx < cellMinX || ncx >= cellMinX + gdx) continue;
        if (ncy < cellMinY || ncy >= cellMinY + gdy) continue;
        if (ncz < cellMinZ || ncz >= cellMinZ + gdz) continue;
        uint cell = (uint)(((ncz - cellMinZ) * gdy + (ncy - cellMinY)) * gdx + (ncx - cellMinX));
        uint s0 = cellStart[cell], s1 = cellStart[cell + 1u];
        for (uint s = s0; s < s1; ++s) {
            uint j = cellAgents[s];
            if (j == i) continue;
            Agent pj = gAgents[j];
            if (BoidsNeighborAccept(pi.px, pi.py, pi.pz, pj.px, pj.py, pj.pz, h)) {
                neighbors[base + local] = j;
                local += 1u;
            }
        }
    }
}
