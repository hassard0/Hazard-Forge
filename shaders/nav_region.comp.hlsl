// Slice NAV3 — Deterministic GPU Navmesh: the INTEGER WATERSHED REGION-GENERATION compute pass (the
// THIRD slice of FLAGSHIP #7; the nav_distance / nav_raster_scan / mc_scan / fpx_solve single-thread
// SERIAL mirror, and the MAKE-OR-BREAK slice). A SINGLE-THREAD allocator ([numthreads(1,1,1)], one
// thread runs the WHOLE level-descending watershed in a FIXED order) partitions the NAV2 walkable
// distance field into REGIONS (connected walkable basins) so each gets a distinct deterministic id.
//
// The LOCKED algorithm (copied VERBATIM from navmesh.h::BuildRegions — EVERY ordering decision is
// PINNED so the converged assignment is identical CPU<->GPU<->both backends):
//   region[c] = 0 for all c; nextRegion = 1
//   for level = maxDist down to 1 (descending water level — ridge tops first):
//     (A) GROW: repeat a full ASCENDING-cellId scan until no change — an unassigned walkable cell at
//         dist==level adopts the LOWEST region id among its 4 neighbours (fixed order up,down,left,
//         right) that are assigned AND IsConnected (the NAV2 max-step predicate).
//     (B) SEED: any still-unassigned walkable cell AT this level (ASCENDING cellId) starts a NEW
//         region (nextRegion++), then is grown across this level (same fixed-point, restricted to
//         dist==level cells connected to a cell already in THIS seed's region).
// region 0 = none (non-walkable / dist 0); ids dense from 1.
//
// WHY BIT-IDENTICAL to navmesh.h::BuildRegions (the make-or-break): PURE INT32 (region ids/levels are
// small ints; IsConnected is integer abs+compare) — NO int64, NO float, NO sqrt -> MSL-generates
// NATIVELY on Metal. The single thread + the fixed ASCENDING scan order + the LOWEST-region-id grow
// tie-break + ASCENDING-cellId seed order make the result order-independent of any scheduling. A
// divergence vs the header is what the host's GPU==CPU memcmp (region[]) catches. enabled=0 -> region
// all 0 (the disabled no-op).
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gWalkable : one uint per column (1=walkable surface), READ.
//   b1 gSurfaceY : one int per column (the walkable surface y), READ.
//   b2 gDist     : one uint per column (the NAV2 chamfer distance), READ.
//   b3 gRegion   : one uint per column (the region id; 0=none), WRITE.
//   b4 gParams   : { w, h, walkableClimb, enabled } + { maxDist, _, _, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

struct NavRegionParams {
    int4 cfg;   // x=w, y=h, z=walkableClimb, w=enabled
    int4 ext;   // x=maxDist, y/z/w=pad
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>            gWalkable : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<int>             gSurfaceY : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            gDist     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>            gRegion   : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<NavRegionParams> gParams   : register(u4);

// IsConnected(walkA,surfA, walkB,surfB, climb): the 4-neighbour max-step connectivity predicate.
// VERBATIM navmesh.h::IsConnected (pure integer abs + compare).
bool IsConnected(uint walkA, int surfA, uint walkB, int surfB, int climb) {
    if (walkA == 0u || walkB == 0u) return false;
    int d = surfA - surfB;
    if (d < 0) d = -d;
    return d <= climb;
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    // SINGLE THREAD: only thread 0 runs the serial watershed (guard so a larger dispatch can't re-run).
    if (gid.x != 0u) return;

    int w       = gParams[0].cfg.x;
    int h       = gParams[0].cfg.y;
    int climb   = gParams[0].cfg.z;
    int enabled = gParams[0].cfg.w;
    int maxDist = gParams[0].ext.x;
    int columnCount = w * h;

    // region[c] = 0 for all c.
    for (int c = 0; c < columnCount; ++c) gRegion[c] = 0u;

    // Disabled -> all-zero regions (the byte-identical no-op).
    if (enabled == 0) return;

    uint nextRegion = 1u;

    // Descend level = maxDist..1 (signed counter so the >=1 guard terminates).
    for (int level = maxDist; level >= 1; --level) {
        // (A) GROW: existing regions expand into this level's unassigned cells (fixed-point).
        bool changed = true;
        while (changed) {
            changed = false;
            for (int z = 0; z < h; ++z)
                for (int x = 0; x < w; ++x) {
                    int c = z * w + x;
                    if (gRegion[c] != 0u || gWalkable[c] == 0u || gDist[c] != (uint)level) continue;
                    // Adopt the LOWEST region id among the 4 neighbours assigned AND connected
                    // (fixed neighbour order: up (z-1), down (z+1), left (x-1), right (x+1)).
                    uint best = 0u;
                    // up
                    if (z - 1 >= 0) {
                        int nc = (z - 1) * w + x;
                        if (IsConnected(gWalkable[c], gSurfaceY[c], gWalkable[nc], gSurfaceY[nc], climb) &&
                            gRegion[nc] != 0u) { if (best == 0u || gRegion[nc] < best) best = gRegion[nc]; }
                    }
                    // down
                    if (z + 1 < h) {
                        int nc = (z + 1) * w + x;
                        if (IsConnected(gWalkable[c], gSurfaceY[c], gWalkable[nc], gSurfaceY[nc], climb) &&
                            gRegion[nc] != 0u) { if (best == 0u || gRegion[nc] < best) best = gRegion[nc]; }
                    }
                    // left
                    if (x - 1 >= 0) {
                        int nc = z * w + (x - 1);
                        if (IsConnected(gWalkable[c], gSurfaceY[c], gWalkable[nc], gSurfaceY[nc], climb) &&
                            gRegion[nc] != 0u) { if (best == 0u || gRegion[nc] < best) best = gRegion[nc]; }
                    }
                    // right
                    if (x + 1 < w) {
                        int nc = z * w + (x + 1);
                        if (IsConnected(gWalkable[c], gSurfaceY[c], gWalkable[nc], gSurfaceY[nc], climb) &&
                            gRegion[nc] != 0u) { if (best == 0u || gRegion[nc] < best) best = gRegion[nc]; }
                    }
                    if (best != 0u) { gRegion[c] = best; changed = true; }
                }
        }
        // (B) SEED: any still-unassigned walkable cell AT this level starts a NEW region (ascending
        // cellId), then is grown across this level (same fixed-point, restricted to dist==level cells
        // connected to a cell already in THIS seed's region).
        for (int z = 0; z < h; ++z)
            for (int x = 0; x < w; ++x) {
                int c = z * w + x;
                if (gRegion[c] != 0u || gWalkable[c] == 0u || gDist[c] != (uint)level) continue;
                uint thisSeed = nextRegion;
                gRegion[c] = thisSeed;
                ++nextRegion;
                // Grow this seed across the current level.
                bool grew = true;
                while (grew) {
                    grew = false;
                    for (int gz = 0; gz < h; ++gz)
                        for (int gx = 0; gx < w; ++gx) {
                            int c2 = gz * w + gx;
                            if (gRegion[c2] != 0u || gWalkable[c2] == 0u || gDist[c2] != (uint)level) continue;
                            bool adopt = false;
                            // up
                            if (gz - 1 >= 0) {
                                int nc = (gz - 1) * w + gx;
                                if (IsConnected(gWalkable[c2], gSurfaceY[c2], gWalkable[nc], gSurfaceY[nc], climb) &&
                                    gRegion[nc] == thisSeed) adopt = true;
                            }
                            // down
                            if (gz + 1 < h) {
                                int nc = (gz + 1) * w + gx;
                                if (IsConnected(gWalkable[c2], gSurfaceY[c2], gWalkable[nc], gSurfaceY[nc], climb) &&
                                    gRegion[nc] == thisSeed) adopt = true;
                            }
                            // left
                            if (gx - 1 >= 0) {
                                int nc = gz * w + (gx - 1);
                                if (IsConnected(gWalkable[c2], gSurfaceY[c2], gWalkable[nc], gSurfaceY[nc], climb) &&
                                    gRegion[nc] == thisSeed) adopt = true;
                            }
                            // right
                            if (gx + 1 < w) {
                                int nc = gz * w + (gx + 1);
                                if (IsConnected(gWalkable[c2], gSurfaceY[c2], gWalkable[nc], gSurfaceY[nc], climb) &&
                                    gRegion[nc] == thisSeed) adopt = true;
                            }
                            if (adopt) { gRegion[c2] = thisSeed; grew = true; }
                        }
                }
            }
    }
}
