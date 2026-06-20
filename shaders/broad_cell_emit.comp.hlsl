// Slice BP1 — Deterministic Integer Broadphase: THE BODY GRID + CELL TABLE BODY-SCATTER compute pass (the 3rd
// of the BP1 cell-table count->scan->emit; the grain_cell_emit / boids_cell_emit / fluid_cell_emit twin).
// SINGLE-THREAD ([numthreads(1,1,1)]): one thread walks body 0..bodyCount-1 ASCENDING and scatters each body's
// index into its cell's slice at cellStart[cell] + cursor[cell]++. The within-cell order is ASCENDING BODY
// INDEX — VERBATIM the CPU broad.h::BuildBodyCellTable emit. THIS IS WHY IT IS SINGLE-THREAD (the determinism
// crux, NOT the per-body-disjoint pattern): multiple bodies scatter into the SAME cell, so a parallel
// atomic-cursor emit would order them by GPU SCHEDULING (non-deterministic). Walking bodies ascending in ONE
// thread with a per-cell cursor reproduces the CPU's ascending-index within-cell order EXACTLY -> bit-
// identical. The cell COUNT (broad_cell_count) is an order-independent SUM so it can be parallel+atomic; the
// EMIT's within-cell ORDER is what forces single-thread (the grain_cell_emit precedent).
//
// The per-cell cursor is kept in the cellCursor buffer (pre-cleared to 0; one uint per cell), NOT a
// groupshared/local array (cellCount can exceed any fixed local size). enabled=0 -> emit nothing (cellBodies
// stays the cleared upload).
//
// WHY BIT-IDENTICAL to the CPU broad.h::BuildBodyCellTable (the make-or-break): same FlatCellId integer math,
// same ascending-body walk, same per-cell cursor -> the cellBodies array is byte-for-byte the CPU's. Pure
// int32 -> MSL-gens natively on Metal.
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gBodies    : the Q16.16 FxBody mirror array (16-byte std430 ints), READ (only pos).
//   b1 cellStart  : the BP1 cell-table exclusive prefix-sum (cellCount+1), READ.
//   b2 cellCursor : one uint per cell, the per-cell write cursor, READ+WRITE (pre-cleared to 0).
//   b3 cellBodies : the output body-index array grouped by cell, WRITE (pre-cleared).
//   b4 gParams    : the BodyGridParams, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

// std430 FxBody mirror (16 bytes; matches broad_cell_count).
struct BroadBody {
    int  px, py, pz;
    int  radius;
};

struct BodyGridParams {
    int4 grid;   // x=cellSize, y=cellMinX, z=cellMinY, w=cellMinZ
    int4 dim;    // x=gridDimX, y=gridDimY, z=gridDimZ, w=bodyCount
    int4 cfg;    // x=cellCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<BroadBody>      gBodies    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           cellStart  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>           cellCursor : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>           cellBodies : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<BodyGridParams> gParams    : register(u4);

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
    if (gid.x != 0u) return;   // SINGLE THREAD: ascending-body scatter (the within-cell order crux)

    int bodyCount = gParams[0].dim.w;
    int enabled   = gParams[0].cfg.y;
    if (enabled == 0) return;

    for (int i = 0; i < bodyCount; ++i) {
        BroadBody b = gBodies[i];
        uint cell = FlatCellId(b.px, b.py, b.pz);
        uint local = cellCursor[cell];
        cellBodies[cellStart[cell] + local] = (uint)i;
        cellCursor[cell] = local + 1u;
    }
}
