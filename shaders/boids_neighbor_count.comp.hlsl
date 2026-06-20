// Slice BD2 — Deterministic GPU Crowds: the GRID-HASH NEIGHBOR LIST per-agent NEIGHBOR-COUNT compute pass
// (the 1st of the BD2 neighbor-list count->scan->emit; the GR2 grain_neighbor_count twin). ONE thread per
// AGENT i (i < agentCount). The thread computes agent i's cell, scans the 27 cells of its 3x3x3 stencil (the
// cell + its 26 neighbors), and for each agent j != i in those cells (read from the BD2 cell table
// cellStart/cellAgents) counts it iff the PURE INT32 per-axis BoidsNeighborAccept (|pos_i.axis - pos_j.axis| <
// radius on every axis) passes -> perAgentCount[i]. NO atomics (each thread writes its OWN perAgentCount[i], a
// per-agent independent count). enabled=0 -> write 0.
//
// WHY BIT-IDENTICAL to the CPU boids.h::CountBoidsNeighbors (the make-or-break): BoidsNeighborAccept is a
// per-axis integer subtract + abs + compare — PURE INT32, NO products, NO int64, NO sqrt (the radial
// dist²<radius² compare is AVOIDED — it would need an int64 product glslc cannot parse; the box IS the
// accepted perception neighborhood, the spec's MSL-native escape hatch). The stencil walk is the same fixed
// (dz,dy,dx) order over the same cell table. So this MSL-generates NATIVELY on Metal (unlike BD1's int64
// boids_steer) — a TRUE GPU pass on BOTH backends -> the host's GPU==CPU memcmp (perAgentCount) catches
// divergence.
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gAgents       : the Q16.16 Agent array (24-byte std430 ints), READ (pos).
//   b1 cellStart     : the BD2 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellAgents    : the BD2 cell-table agent indices grouped by cell, READ.
//   b3 perAgentCount : one uint per agent (the neighbor count), WRITE.
//   b4 gParams       : the BoidsGridParams (radius, cellMin.xyz, gridDim.xyz, agentCount, cellCount, enabled), READ.
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
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>            perAgentCount : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<BoidsGridParams> gParams       : register(u4);

int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

// BoidsNeighborAccept(ax,ay,az, bx,by,bz, radius): the PURE INT32 per-axis |dx| < radius candidate test —
// VERBATIM boids.h::BoidsNeighborAccept. NO products, NO int64, NO sqrt.
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
    if (enabled == 0) { perAgentCount[i] = 0u; return; }

    Agent pi = gAgents[i];
    int cix = FloorDiv(pi.px, h);
    int ciy = FloorDiv(pi.py, h);
    int ciz = FloorDiv(pi.pz, h);

    uint c = 0u;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int ncx = cix + dx, ncy = ciy + dy, ncz = ciz + dz;
        // Skip stencil cells outside the bounded grid (clamp).
        if (ncx < cellMinX || ncx >= cellMinX + gdx) continue;
        if (ncy < cellMinY || ncy >= cellMinY + gdy) continue;
        if (ncz < cellMinZ || ncz >= cellMinZ + gdz) continue;
        uint cell = (uint)(((ncz - cellMinZ) * gdy + (ncy - cellMinY)) * gdx + (ncx - cellMinX));
        uint s0 = cellStart[cell], s1 = cellStart[cell + 1u];
        for (uint s = s0; s < s1; ++s) {
            uint j = cellAgents[s];
            if (j == i) continue;
            Agent pj = gAgents[j];
            if (BoidsNeighborAccept(pi.px, pi.py, pi.pz, pj.px, pj.py, pj.pz, h)) c += 1u;
        }
    }
    perAgentCount[i] = c;   // per-agent independent count (NO atomics)
}
