// Slice BD2 — Deterministic GPU Crowds: the GRID-HASH NEIGHBOR LIST cell-table AGENT-SCATTER compute pass
// (the 3rd of the BD2 cell-table count->scan->emit; the GR2 grain_cell_emit twin). SINGLE-THREAD
// ([numthreads(1,1,1)]): one thread walks agent 0..agentCount-1 ASCENDING and scatters each agent's index
// into its cell's slice at cellStart[cell] + cursor[cell]++. The within-cell order is ASCENDING AGENT INDEX —
// VERBATIM the CPU boids.h::BuildBoidsCellTable emit. THIS IS WHY IT IS SINGLE-THREAD (the determinism crux,
// NOT the per-agent-disjoint pattern): multiple agents scatter into the SAME cell, so a parallel atomic-cursor
// emit would order them by GPU SCHEDULING (non-deterministic). Walking agents ascending in ONE thread with a
// per-cell cursor reproduces the CPU's ascending-index within-cell order EXACTLY -> bit-identical. The cell
// COUNT (boids_cell_count) is an order-independent SUM so it can be parallel+atomic; the EMIT's within-cell
// ORDER is what forces single-thread (the grain_cell_emit precedent).
//
// The per-cell cursor is kept in the cellCursor buffer (pre-cleared to 0; one uint per cell), NOT a
// groupshared/local array (cellCount can exceed any fixed local size). enabled=0 -> emit nothing
// (cellAgents stays the cleared upload).
//
// WHY BIT-IDENTICAL to the CPU boids.h::BuildBoidsCellTable (the make-or-break): same FlatCellId integer math,
// same ascending-agent walk, same per-cell cursor -> the cellAgents array is byte-for-byte the CPU's. Pure
// int32 -> MSL-gens natively on Metal.
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gAgents    : the Q16.16 Agent array (24-byte std430 ints), READ (only pos).
//   b1 cellStart  : the BD2 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellCursor : one uint per cell, the per-cell write cursor, READ+WRITE (pre-cleared to 0).
//   b3 cellAgents : the output agent-index array grouped by cell, WRITE (pre-cleared).
//   b4 gParams    : the BoidsGridParams, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

// std430 Agent mirror (24 bytes).
struct Agent {
    int px, py, pz;
    int vx, vy, vz;
};

struct BoidsGridParams {
    int4 grid;   // x=radius, y=cellMinX, z=cellMinY, w=cellMinZ
    int4 dim;    // x=gridDimX, y=gridDimY, z=gridDimZ, w=agentCount
    int4 cfg;    // x=cellCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Agent>           gAgents    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            cellStart  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            cellCursor : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>            cellAgents : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<BoidsGridParams> gParams    : register(u4);

int FloorDiv(int n, int d) {
    int q = n / d, r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

uint FlatCellId(int px, int py, int pz) {
    int h        = gParams[0].grid.x;
    int cellMinX = gParams[0].grid.y, cellMinY = gParams[0].grid.z, cellMinZ = gParams[0].grid.w;
    int gx = gParams[0].dim.x, gy = gParams[0].dim.y;
    int cx = FloorDiv(px, h) - cellMinX;
    int cy = FloorDiv(py, h) - cellMinY;
    int cz = FloorDiv(pz, h) - cellMinZ;
    return (uint)((cz * gy + cy) * gx + cx);
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: ascending-agent scatter (the within-cell order crux)

    int agentCount = gParams[0].dim.w;
    int enabled    = gParams[0].cfg.y;
    if (enabled == 0) return;

    for (int i = 0; i < agentCount; ++i) {
        Agent p = gAgents[i];
        uint cell = FlatCellId(p.px, p.py, p.pz);
        uint local = cellCursor[cell];
        cellAgents[cellStart[cell] + local] = (uint)i;
        cellCursor[cell] = local + 1u;
    }
}
