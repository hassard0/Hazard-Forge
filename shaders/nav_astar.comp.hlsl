// Slice NAV5 — Deterministic GPU Navmesh: the INTEGER A* PATHFINDING compute pass (the FIFTH slice +
// the HEADLINE of FLAGSHIP #7; the nav_region / nav_polygonize single-thread SERIAL mirror). A
// SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread runs the WHOLE A* in a FIXED order) runs a
// fully INTEGER A* over the NAV4 poly adjacency graph (nodes = polys, edges = nbr[]) to produce a
// CORRIDOR (the poly-id sequence start->goal) + total integer cost.
//
// The LOCKED algorithm (copied VERBATIM from navmesh.h::ConnectedComponents/SelectStartGoal/FindPath —
// EVERY ordering decision PINNED so the result is identical CPU<->GPU<->both backends):
//   1) ConnectedComponents: deterministic flood over nbr[] (ascending seed order, fixed neighbour
//      order 0,1,2) -> per-poly component id.
//   2) SelectStartGoal: pick the LARGEST component (tie -> lowest min poly id); start = its lowest poly
//      id; goal = the poly in it with the MAX Manhattan centroid distance from start (tie -> lowest id).
//   3) FindPath: A* — g[]=INF, g[start]=0, open={start}; loop: POP the open node with the lowest
//      f=g+h, tie-break LOWEST poly id (a linear min-scan); if it is the goal STOP; else close it +
//      relax neighbours (fixed edge order). Reconstruct via came_from goal->start then reverse.
//   Cost g + heuristic h = the MANHATTAN distance |dx|+|dz| between integer poly centroids (cx/cz,
//   precomputed host-side by the VERBATIM ComputePolyCentroids — integer, so byte-exact to upload).
//
// WHY BIT-IDENTICAL to navmesh.h (the make-or-break): PURE INT32 — centroid coords <= 32 (showcase),
// a single edge cost <= 64, a whole-corridor cost << INT32_MAX (the navmesh.h NAV5 overflow bound).
// NO int64, NO float, NO sqrt -> MSL-gens NATIVELY on Metal (the NAV3/NAV4 convention, unlike fpx).
// The single thread + the fixed scan order + the lowest-id tie-break make A* order-independent. A
// divergence vs the header is what the host's GPU==CPU memcmp (corridor + cost + start/goal) catches.
// enabled=0 -> corridorLen 0 (the disabled no-op).
//
// A Poly is 8 uints (std430, matching nav_polygonize): poly p occupies gPolys[p*8 + 0..7] = idx[0..2],
// nbr[0..2], region, _pad. We only read idx (for nothing here) + nbr (the graph edges).
//
// Buffers (storage, bound at compute bindings 0..5; on Metal these land at buffer(0..5)):
//   b0 gPolys    : flat uint poly buffer (8 uints/poly), READ.
//   b1 gPolyCx   : one int per poly (centroid x), READ.
//   b2 gPolyCz   : one int per poly (centroid z), READ.
//   b3 gCorridor : one uint per corridor slot (the output poly-id sequence), WRITE.
//   b4 gMeta     : { corridorLen, totalCost, start, goal } (4 uints), WRITE.
//   b5 gParams   : { polyCount, polyCap, enabled, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_NAV_NO_NEIGHBOUR 0xFFFFFFFFu
#define HF_NAV_PATH_INF     0x3FFFFFFF
#define HF_NAV_NO_CAMEFROM  0xFFFFFFFFu
#define HF_NAV_MAX_POLYS    512    // the per-thread scratch bound (matches nav_polygonize's kMaxN; the
                                   // 32x32 showcase produces a handful of polys; 7*512*4B ~ 14KB private
                                   // mem, within the compute local-memory limit — 4096 overflowed it).

struct NavAstarParams {
    int4 cfg;   // x=polyCount, y=polyCap, z=enabled, w=pad
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>           gPolys    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<int>            gPolyCx   : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<int>            gPolyCz   : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>           gCorridor : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>           gMeta     : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<NavAstarParams> gParams   : register(u5);

int ManhattanDist(int ax, int az, int bx, int bz) {
    int dx = ax - bx; if (dx < 0) dx = -dx;
    int dz = az - bz; if (dz < 0) dz = -dz;
    return dx + dz;
}

uint PolyNbr(uint p, int e) { return gPolys[p * 8u + 3u + (uint)e]; }

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 runs the serial A*.

    // Per-thread scratch as FUNCTION-LOCAL arrays (the nav_polygonize convention -> MSL-gens natively).
    uint comp[HF_NAV_MAX_POLYS];      // component id per poly (HF_NAV_NO_CAMEFROM = unassigned)
    int  gcost[HF_NAV_MAX_POLYS];     // A* g[]
    uint sopen[HF_NAV_MAX_POLYS];     // 1 = in the open set
    uint sclosed[HF_NAV_MAX_POLYS];   // 1 = expanded
    uint cameFrom[HF_NAV_MAX_POLYS];  // A* predecessor
    uint stk[HF_NAV_MAX_POLYS];       // flood stack
    uint rev[HF_NAV_MAX_POLYS];       // reversed corridor scratch

    int nP      = gParams[0].cfg.x;
    int enabled = gParams[0].cfg.z;

    // Clear the meta (corridorLen, totalCost, start, goal) first (the disabled no-op writes 0s).
    gMeta[0] = 0u; gMeta[1] = 0u; gMeta[2] = 0u; gMeta[3] = 0u;
    if (enabled == 0 || nP <= 0) return;
    if (nP > HF_NAV_MAX_POLYS) nP = HF_NAV_MAX_POLYS;

    // ----- 1) ConnectedComponents (deterministic flood, ascending seed + fixed neighbour order) -----
    for (int ci = 0; ci < nP; ++ci) comp[ci] = HF_NAV_NO_CAMEFROM;
    uint nextComp = 0u;
    for (int s = 0; s < nP; ++s) {
        if (comp[s] != HF_NAV_NO_CAMEFROM) continue;
        uint c = nextComp; nextComp += 1u;
        comp[s] = c;
        int sp = 0; stk[sp] = (uint)s; sp += 1;
        while (sp > 0) {
            sp -= 1; uint p = stk[sp];
            for (int e = 0; e < 3; ++e) {
                uint q = PolyNbr(p, e);
                if (q == HF_NAV_NO_NEIGHBOUR || q >= (uint)nP) continue;
                if (comp[q] != HF_NAV_NO_CAMEFROM) continue;
                comp[q] = c;
                stk[sp] = q; sp += 1;
            }
        }
    }

    // ----- 2) SelectStartGoal (largest component, tie->lowest min id; start=lowest id; goal=farthest) -
    // Per-component size + lowest min poly id (reuse gcost/sopen as scratch arrays over [0,nextComp)).
    // sizeOf -> stk (reused), minIdOf -> rev (reused) for [0,nextComp).
    uint nComp = nextComp;
    for (uint cc = 0u; cc < nComp; ++cc) { stk[cc] = 0u; rev[cc] = HF_NAV_NO_CAMEFROM; }
    for (uint p = 0u; p < (uint)nP; ++p) {
        uint c = comp[p];
        stk[c] += 1u;
        if (rev[c] == HF_NAV_NO_CAMEFROM) rev[c] = p;
    }
    uint bestC = 0u;
    for (uint c = 1u; c < nComp; ++c) {
        if (stk[c] > stk[bestC] || (stk[c] == stk[bestC] && rev[c] < rev[bestC])) bestC = c;
    }
    uint start = rev[bestC];
    uint goal = start;
    int bestD = -1;
    for (uint p2 = 0u; p2 < (uint)nP; ++p2) {
        if (comp[p2] != bestC) continue;
        int d = ManhattanDist(gPolyCx[start], gPolyCz[start], gPolyCx[p2], gPolyCz[p2]);
        if (d > bestD) { bestD = d; goal = p2; }
    }
    gMeta[2] = start; gMeta[3] = goal;

    // start == goal -> single-node corridor (cost 0).
    if (start == goal) {
        gCorridor[0] = start;
        gMeta[0] = 1u; gMeta[1] = 0u;
        return;
    }

    // ----- 3) FindPath: the integer A* (linear min-scan frontier, lowest-f / lowest-id pop) ----------
    for (int i = 0; i < nP; ++i) {
        gcost[i] = HF_NAV_PATH_INF;
        sopen[i] = 0u;
        sclosed[i] = 0u;
        cameFrom[i] = HF_NAV_NO_CAMEFROM;
    }
    gcost[start] = 0;
    sopen[start] = 1u;

    while (true) {
        uint cur = HF_NAV_NO_CAMEFROM;
        int bestF = HF_NAV_PATH_INF;
        for (uint p = 0u; p < (uint)nP; ++p) {
            if (sopen[p] == 0u) continue;
            int h = ManhattanDist(gPolyCx[p], gPolyCz[p], gPolyCx[goal], gPolyCz[goal]);
            int f = gcost[p] + h;
            if (cur == HF_NAV_NO_CAMEFROM || f < bestF) { bestF = f; cur = p; }
        }
        if (cur == HF_NAV_NO_CAMEFROM) break;   // open empty -> unreachable.
        if (cur == goal) break;                 // goal popped -> done.

        sopen[cur] = 0u;
        sclosed[cur] = 1u;
        for (int e = 0; e < 3; ++e) {
            uint nb = PolyNbr(cur, e);
            if (nb == HF_NAV_NO_NEIGHBOUR || nb >= (uint)nP) continue;
            if (sclosed[nb] != 0u) continue;
            int step = ManhattanDist(gPolyCx[cur], gPolyCz[cur], gPolyCx[nb], gPolyCz[nb]);
            int tentative = gcost[cur] + step;
            if (tentative < gcost[nb]) {
                gcost[nb] = tentative;
                cameFrom[nb] = cur;
                sopen[nb] = 1u;
            }
        }
    }

    // Reconstruct (goal reached iff g[goal] < INF). Walk came_from goal->start, then reverse.
    if (gcost[goal] >= HF_NAV_PATH_INF) {
        gMeta[0] = 0u; gMeta[1] = 0u;   // unreachable -> empty corridor.
        return;
    }
    int rn = 0;
    uint node = goal;
    for (int guard = 0; guard <= nP; ++guard) {
        rev[rn] = node; rn += 1;
        if (node == start) break;
        node = cameFrom[node];
        if (node == HF_NAV_NO_CAMEFROM) { rn = 0; break; }
    }
    if (rn == 0 || rev[rn - 1] != start) {
        gMeta[0] = 0u; gMeta[1] = 0u;
        return;
    }
    for (int k = 0; k < rn; ++k) gCorridor[k] = rev[rn - 1 - k];
    gMeta[0] = (uint)rn;
    gMeta[1] = (uint)gcost[goal];
}
