// Slice NAV2 — Deterministic GPU Navmesh: the INTEGER CHAMFER DISTANCE FIELD compute pass (the SECOND
// slice of FLAGSHIP #7; the nav_raster_scan / mc_scan / fpx_solve single-thread SERIAL mirror). A
// SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread walks all w*h cells in a FIXED order) runs
// the two-pass integer chamfer distance transform over the NAV2 walkable grid: seed non-walkable +
// border cells = 0, walkable interior = kDistInf, then FORWARD sweep row-major TL->BR (relax against
// the up/left neighbours) then BACKWARD sweep BR->TL (relax against the down/right neighbours), each
// relax = min(self, neighbour + weight) with cardinal weight 2 / diagonal weight 3 (the standard
// integer Recast chamfer — NO sqrt, NO int64). A neighbour is only traversed if it is walkable AND
// CONNECTED (abs(surfaceY[c]-surfaceY[nbr]) <= walkableClimb, the max-step test) -> the distance is
// GEODESIC over the walkable surface. The two sweeps are inherently SEQUENTIAL (each cell reads its
// already-updated neighbours), so a deterministic SERIAL single-thread sweep is correct + bit-exact.
//
// WHY BIT-IDENTICAL to the CPU navmesh.h::BuildDistanceField (the make-or-break): everything is PURE
// INT32 (integer add/compare/abs, weights 2/3, distances bounded ~3*(w+h) << kDistInf < INT32_MAX) ->
// NO int64, NO float -> MSL-generates NATIVELY on Metal. A divergence vs the header is what the host's
// GPU==CPU memcmp (dist[]) catches. enabled=0 -> dist all 0 (the disabled no-op).
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3)):
//   b0 gWalkable : one uint per column (1=walkable surface), READ.
//   b1 gSurfaceY : one int per column (the walkable surface y), READ.
//   b2 gDist     : one uint per column (the chamfer distance), WRITE.
//   b3 gParams   : { w, h, walkableClimb, enabled }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

// kDistInf VERBATIM navmesh.h::kDistInf (~1.07e9, < INT32_MAX; dist+3 stays in int32/uint range).
#define HF_NAV_DIST_INF 0x3FFFFFFFu

struct NavDistParams {
    int4 cfg;   // x=w, y=h, z=walkableClimb, w=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>          gWalkable : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<int>           gSurfaceY : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>          gDist     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<NavDistParams> gParams   : register(u3);

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
    // SINGLE THREAD: only thread 0 runs the serial chamfer (guard so a larger dispatch can't re-run it).
    if (gid.x != 0u) return;

    int w       = gParams[0].cfg.x;
    int h       = gParams[0].cfg.y;
    int climb   = gParams[0].cfg.z;
    int enabled = gParams[0].cfg.w;
    int columnCount = w * h;

    // Disabled -> all-zero distance (the byte-identical no-op).
    if (enabled == 0) {
        for (int c = 0; c < columnCount; ++c) gDist[c] = 0u;
        return;
    }

    // Seed: walkable interior = kDistInf, non-walkable OR border = 0.
    for (int z = 0; z < h; ++z)
        for (int x = 0; x < w; ++x) {
            int c = z * w + x;
            bool border = (x == 0 || z == 0 || x == w - 1 || z == h - 1);
            gDist[c] = (gWalkable[c] != 0u && !border) ? HF_NAV_DIST_INF : 0u;
        }

    const uint kCard = 2u;
    const uint kDiag = 3u;

    // FORWARD sweep TL->BR: relax against the up/left neighbours (W, NW, N, NE).
    for (int z = 0; z < h; ++z)
        for (int x = 0; x < w; ++x) {
            int c = z * w + x;
            if (gDist[c] == 0u) continue;
            uint wc = gWalkable[c]; int sc = gSurfaceY[c];
            // W
            if (x - 1 >= 0) {
                int nc = z * w + (x - 1);
                if (IsConnected(wc, sc, gWalkable[nc], gSurfaceY[nc], climb)) {
                    uint cand = gDist[nc] + kCard; if (cand < gDist[c]) gDist[c] = cand;
                }
            }
            // NW
            if (x - 1 >= 0 && z - 1 >= 0) {
                int nc = (z - 1) * w + (x - 1);
                if (IsConnected(wc, sc, gWalkable[nc], gSurfaceY[nc], climb)) {
                    uint cand = gDist[nc] + kDiag; if (cand < gDist[c]) gDist[c] = cand;
                }
            }
            // N
            if (z - 1 >= 0) {
                int nc = (z - 1) * w + x;
                if (IsConnected(wc, sc, gWalkable[nc], gSurfaceY[nc], climb)) {
                    uint cand = gDist[nc] + kCard; if (cand < gDist[c]) gDist[c] = cand;
                }
            }
            // NE
            if (x + 1 < w && z - 1 >= 0) {
                int nc = (z - 1) * w + (x + 1);
                if (IsConnected(wc, sc, gWalkable[nc], gSurfaceY[nc], climb)) {
                    uint cand = gDist[nc] + kDiag; if (cand < gDist[c]) gDist[c] = cand;
                }
            }
        }

    // BACKWARD sweep BR->TL: relax against the down/right neighbours (E, SE, S, SW).
    for (int z = h - 1; z >= 0; --z)
        for (int x = w - 1; x >= 0; --x) {
            int c = z * w + x;
            if (gDist[c] == 0u) continue;
            uint wc = gWalkable[c]; int sc = gSurfaceY[c];
            // E
            if (x + 1 < w) {
                int nc = z * w + (x + 1);
                if (IsConnected(wc, sc, gWalkable[nc], gSurfaceY[nc], climb)) {
                    uint cand = gDist[nc] + kCard; if (cand < gDist[c]) gDist[c] = cand;
                }
            }
            // SE
            if (x + 1 < w && z + 1 < h) {
                int nc = (z + 1) * w + (x + 1);
                if (IsConnected(wc, sc, gWalkable[nc], gSurfaceY[nc], climb)) {
                    uint cand = gDist[nc] + kDiag; if (cand < gDist[c]) gDist[c] = cand;
                }
            }
            // S
            if (z + 1 < h) {
                int nc = (z + 1) * w + x;
                if (IsConnected(wc, sc, gWalkable[nc], gSurfaceY[nc], climb)) {
                    uint cand = gDist[nc] + kCard; if (cand < gDist[c]) gDist[c] = cand;
                }
            }
            // SW
            if (x - 1 >= 0 && z + 1 < h) {
                int nc = (z + 1) * w + (x - 1);
                if (IsConnected(wc, sc, gWalkable[nc], gSurfaceY[nc], climb)) {
                    uint cand = gDist[nc] + kDiag; if (cand < gDist[c]) gDist[c] = cand;
                }
            }
        }

    // Any walkable cell still at kDistInf (an isolated island) -> 0 (never read back the sentinel).
    for (int c2 = 0; c2 < columnCount; ++c2)
        if (gDist[c2] == HF_NAV_DIST_INF) gDist[c2] = 0u;
}
