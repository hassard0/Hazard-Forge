// Slice NAV4 — Deterministic GPU Navmesh: the INTEGER CONTOUR TRACE + SIMPLIFY compute pass (the
// FOURTH slice of FLAGSHIP #7; the nav_region / nav_distance / nav_raster_scan / mc_scan / fpx_solve
// single-thread SERIAL mirror). A SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread traces +
// simplifies EVERY region's boundary in a FIXED order) turns the NAV3 region partition into a flat
// buffer of integer CONTOUR vertices + per-region vertex offsets/counts.
//
// The LOCKED algorithm (copied VERBATIM from navmesh.h::TraceContours + SimplifyContour — EVERY
// ordering decision is PINNED so the result is identical CPU<->GPU<->both backends):
//   for each region id R = 1..regionCount (ascending):
//     find R's LOWEST-cellId cell -> its TOP-LEFT corner is the start, heading +x;
//     walk the boundary keeping the region cell on the RIGHT, choosing the next heading by the FIXED
//     priority {turnRight, straight, turnLeft} (else turn back), emitting a corner vertex at each
//     heading change, until back at the start corner;
//     then Douglas-Peucker simplify (perpendicular-distance SQUARED vs maxError^2, explicit fixed-order
//     stack, split at max-deviation, tie -> lowest index, keep >= 3).
//   The simplified loop is appended to gContourVerts at the region's offset; gRegionVOffset[R-1] /
//   gRegionVCount[R-1] are the per-region base + count; gCounts[0] = total contour vertices.
//
// WHY BIT-IDENTICAL to navmesh.h (the make-or-break): PURE INT32 — corner coords are small voxel ints
// (showcase 32x32 -> corner coords in [0,32]); the Cross2 / perpendicular-distance-squared products
// fit int32 (the documented overflow bound in navmesh.h NAV4). NO int64, NO float, NO sqrt -> the
// shader MSL-generates NATIVELY on Metal. The single thread + the fixed start + fixed turn order +
// fixed DP stack order make the result order-independent of any scheduling. enabled=0 -> all counts 0
// (the disabled no-op).
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gRegion       : one uint per column (the NAV3 region id; 0=none), READ.
//   b1 gContourVerts : flat int2 (x,z) contour vertices, capacity gParams.cap, WRITE.
//   b2 gRegionVOff   : per-region (regionCount) vertex offset, WRITE.
//   b3 gRegionVCnt   : per-region (regionCount) vertex count, WRITE.
//   b4 gParams       : { w, h, maxError, enabled } + { regionCount, cap, _, _ }, READ.
//   gCounts is the FIRST element of gRegionVOff is NOT used; total is gRegionVOff[regionCount] via
//   a sentinel — INSTEAD we pack total into gRegionVCnt has its own; see host. (We expose total via
//   the host summing gRegionVCnt; no extra buffer needed.)
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

struct NavContourParams {
    int4 cfg;   // x=w, y=h, z=maxError, w=enabled
    int4 ext;   // x=regionCount, y=cap (vertex capacity), z/w=pad
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>             gRegion       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<int2>             gContourVerts : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>             gRegionVOff   : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>             gRegionVCnt   : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<NavContourParams> gParams       : register(u4);

// Cross2: twice the signed area of triangle (a,b,p) — the NAV1 PointInTriXZ edge function. Pure int32.
int Cross2(int ax, int az, int bx, int bz, int px, int pz) {
    return (bx - ax) * (pz - az) - (bz - az) * (px - ax);
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 runs the serial trace+simplify.

    int w           = gParams[0].cfg.x;
    int h           = gParams[0].cfg.y;
    int maxError    = gParams[0].cfg.z;
    int enabled     = gParams[0].cfg.w;
    int regionCount = gParams[0].ext.x;
    int cap         = gParams[0].ext.y;

    // Clear per-region offsets/counts.
    for (int r = 0; r < regionCount; ++r) { gRegionVOff[r] = 0u; gRegionVCnt[r] = 0u; }
    if (enabled == 0) return;   // disabled -> all counts 0 (the byte-identical no-op).

    const int dxh[4] = {1, 0, -1, 0};
    const int dzh[4] = {0, 1, 0, -1};

    uint writePos = 0u;   // running base into gContourVerts (the implicit prefix-sum).

    for (int R = 1; R <= regionCount; ++R) {
        // Find the lowest-cellId cell in region R (ascending z, then x).
        int sx = -1, sz = -1;
        for (int z = 0; z < h && sx < 0; ++z)
            for (int x = 0; x < w; ++x)
                if (gRegion[z * w + x] == (uint)R) { sx = x; sz = z; break; }
        if (sx < 0) { gRegionVOff[R - 1] = writePos; gRegionVCnt[R - 1] = 0u; continue; }

        // --- TRACE: build the raw corner loop into a small local scratch (capped). ---
        // We write traced verts directly into gContourVerts starting at writePos, then SIMPLIFY in
        // place (Douglas-Peucker keeps a subset, compacted forward).
        const int startX = sx, startZ = sz;
        int curX = startX, curZ = startZ;
        int dir = 0;            // +x
        int lastDir = -1;
        uint rawBase = writePos;
        uint rawCount = 0u;
        int maxSteps = 8 * (w * h) + 16;
        for (int step = 0; step < maxSteps; ++step) {
            if (dir != lastDir) {
                if (rawBase + rawCount < (uint)cap) {
                    gContourVerts[rawBase + rawCount] = int2(curX, curZ);
                    rawCount += 1u;
                }
                lastDir = dir;
            }
            curX += dxh[dir];
            curZ += dzh[dir];
            if (curX == startX && curZ == startZ) break;

            int turnRight = (dir + 1) & 3;
            int straight  = dir;
            int turnLeft  = (dir + 3) & 3;
            int candc[3]; candc[0] = turnRight; candc[1] = straight; candc[2] = turnLeft;
            int nextDir = (dir + 2) & 3;
            for (int k = 0; k < 3; ++k) {
                int nd = candc[k];
                int rcx, rcz, lcx, lcz;
                if (nd == 0)      { rcx = curX;     rcz = curZ;     lcx = curX;     lcz = curZ - 1; }
                else if (nd == 1) { rcx = curX - 1; rcz = curZ;     lcx = curX;     lcz = curZ;     }
                else if (nd == 2) { rcx = curX - 1; rcz = curZ - 1; lcx = curX - 1; lcz = curZ;     }
                else              { rcx = curX;     rcz = curZ - 1; lcx = curX - 1; lcz = curZ - 1; }
                bool rin = (rcx >= 0 && rcz >= 0 && rcx < w && rcz < h) && (gRegion[rcz * w + rcx] == (uint)R);
                bool lin = (lcx >= 0 && lcz >= 0 && lcx < w && lcz < h) && (gRegion[lcz * w + lcx] == (uint)R);
                if (rin && !lin) { nextDir = nd; break; }
            }
            dir = nextDir;
        }

        // --- SIMPLIFY (integer Douglas-Peucker) over the raw loop [rawBase, rawBase+rawCount). ---
        // keep[] in a local fixed-size scratch indexed 0..rawCount-1. rawCount is bounded by the
        // perimeter (<= 4*w*h); for the showcase it is tiny. We cap the DP scratch at 4096.
        int n = (int)rawCount;
        if (n <= 3) {
            // Already minimal: the raw verts ARE the simplified output (already at writePos..).
            gRegionVOff[R - 1] = writePos;
            gRegionVCnt[R - 1] = (uint)n;
            writePos += (uint)n;
            continue;
        }
        int err2 = maxError * maxError;
        // Read raw verts into local arrays for DP, then write the kept subset back compacted.
        const int kMaxN = 512;
        int2 loc[kMaxN];
        if (n > kMaxN) n = kMaxN;
        for (int i = 0; i < n; ++i) loc[i] = gContourVerts[rawBase + (uint)i];

        bool keepArr[kMaxN];
        for (int i2 = 0; i2 < n; ++i2) keepArr[i2] = false;

        // Anchor 0 + the farthest-from-0 vertex.
        int farIdx = 0; int farD = -1;
        for (int i3 = 1; i3 < n; ++i3) {
            int ex = loc[i3].x - loc[0].x, ez = loc[i3].y - loc[0].y;
            int d = ex * ex + ez * ez;
            if (d > farD) { farD = d; farIdx = i3; }
        }
        keepArr[0] = true; keepArr[farIdx] = true;

        // dpChain via explicit stack (pairs packed as lo*kMaxN+hi).
        int stackArr[kMaxN * 2];
        int sp;

        // Chain A: [0, farIdx]
        sp = 0; stackArr[sp++] = 0 * kMaxN + farIdx;
        while (sp > 0) {
            int seg = stackArr[--sp];
            int a = seg / kMaxN, b = seg % kMaxN;
            if (b <= a + 1) continue;
            int bestIdx = -1; int bestNum = 0; int bestDd = 1;
            for (int i = a + 1; i < b; ++i) {
                int dd = (loc[b].x - loc[a].x) * (loc[b].x - loc[a].x) +
                         (loc[b].y - loc[a].y) * (loc[b].y - loc[a].y);
                int num;
                if (dd == 0) { int ex2 = loc[i].x - loc[a].x, ez2 = loc[i].y - loc[a].y; num = ex2 * ex2 + ez2 * ez2; dd = 1; }
                else { int cr = Cross2(loc[a].x, loc[a].y, loc[b].x, loc[b].y, loc[i].x, loc[i].y); num = cr * cr; }
                if (bestIdx < 0 || num * bestDd > bestNum * dd) { bestIdx = i; bestNum = num; bestDd = dd; }
            }
            if (bestIdx < 0) continue;
            if (bestNum > err2 * bestDd) {
                keepArr[bestIdx] = true;
                stackArr[sp++] = bestIdx * kMaxN + b;
                stackArr[sp++] = a * kMaxN + bestIdx;
            }
        }
        // Chain B: [farIdx, n] (n maps to vertex 0 — emit-order handles the wrap).
        sp = 0; stackArr[sp++] = farIdx * kMaxN + n;
        while (sp > 0) {
            int seg = stackArr[--sp];
            int a = seg / kMaxN, b = seg % kMaxN;
            if (b <= a + 1) continue;
            int ax = loc[a].x, az = loc[a].y;
            int bx = (b >= n) ? loc[0].x : loc[b].x;
            int bz = (b >= n) ? loc[0].y : loc[b].y;
            int bestIdx = -1; int bestNum = 0; int bestDd = 1;
            for (int i = a + 1; i < b; ++i) {
                int dd = (bx - ax) * (bx - ax) + (bz - az) * (bz - az);
                int num;
                if (dd == 0) { int ex2 = loc[i].x - ax, ez2 = loc[i].y - az; num = ex2 * ex2 + ez2 * ez2; dd = 1; }
                else { int cr = Cross2(ax, az, bx, bz, loc[i].x, loc[i].y); num = cr * cr; }
                if (bestIdx < 0 || num * bestDd > bestNum * dd) { bestIdx = i; bestNum = num; bestDd = dd; }
            }
            if (bestIdx < 0) continue;
            if (bestNum > err2 * bestDd) {
                keepArr[bestIdx] = true;
                stackArr[sp++] = bestIdx * kMaxN + b;
                stackArr[sp++] = a * kMaxN + bestIdx;
            }
        }

        // Count kept; if < 3, fall back to 3 evenly-spaced anchors (deterministic).
        int kept = 0;
        for (int i4 = 0; i4 < n; ++i4) if (keepArr[i4]) kept += 1;
        if (kept < 3) {
            for (int i5 = 0; i5 < n; ++i5) keepArr[i5] = false;
            keepArr[0] = true; keepArr[n / 3] = true; keepArr[(2 * n) / 3] = true;
            kept = 0; for (int i6 = 0; i6 < n; ++i6) if (keepArr[i6]) kept += 1;
        }

        // Write the kept subset compacted at writePos.
        uint outBase = writePos;
        uint outCount = 0u;
        for (int i7 = 0; i7 < n; ++i7) {
            if (!keepArr[i7]) continue;
            if (outBase + outCount < (uint)cap) gContourVerts[outBase + outCount] = loc[i7];
            outCount += 1u;
        }
        gRegionVOff[R - 1] = outBase;
        gRegionVCnt[R - 1] = outCount;
        writePos = outBase + outCount;
    }
}
