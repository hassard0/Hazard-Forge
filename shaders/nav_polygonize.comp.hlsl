// Slice NAV4 — Deterministic GPU Navmesh: the INTEGER EAR-CLIP POLYGONIZATION + ADJACENCY compute
// pass (the FOURTH slice of FLAGSHIP #7; the nav_region / nav_distance single-thread SERIAL mirror).
// A SINGLE-THREAD allocator ([numthreads(1,1,1)], one thread triangulates EVERY contour in a FIXED
// order) turns the NAV4 simplified contours into convex polygons (triangles) + per-edge cross-poly
// ADJACENCY (the graph NAV5's A* runs over).
//
// The LOCKED algorithm (copied VERBATIM from navmesh.h::BuildPolyMesh — EVERY ordering decision PINNED
// so the result is identical CPU<->GPU<->both backends):
//   for each region's simplified contour (region order ascending, the nav_contour output order):
//     ear-clip: while >3 verts remain, clip the LOWEST-index valid EAR (a convex vertex — turn matches
//     the loop winding — whose triangle contains NO other remaining vertex, by integer Cross2
//     orientation + point-in-triangle); emit each clipped triangle's 3 contour-local vertex indices;
//     then build per-edge ADJACENCY within the contour (two triangles sharing the reversed edge {a,b}
//     are mutual neighbours; a contour-boundary edge -> kNoNeighbour).
//   Polys are laid out contour-by-contour; gRegionPOff[R-1]/gRegionPCnt[R-1] are the per-region poly
//   base + count.
//
// WHY BIT-IDENTICAL to navmesh.h (the make-or-break): PURE INT32 — corner coords <= 32 (showcase),
// the Cross2 products fit int32 (the navmesh.h NAV4 overflow bound). NO int64, NO float -> MSL-gens
// NATIVELY on Metal. The single thread + fixed lowest-index ear order make it order-independent.
// enabled=0 -> all counts 0 (the disabled no-op).
//
// A Poly is 8 uints (std430, 32 bytes, memcmp-able): idx[0..2], nbr[0..2], region, _pad. We store it
// as a flat uint buffer (8 uints per poly): poly p occupies gPolys[p*8 + 0..7].
//
// Buffers (storage, bound at compute bindings 0..6; on Metal these land at buffer(0..6)):
//   b0 gContourVerts : flat int2 (x,z) contour vertices (the nav_contour output), READ.
//   b1 gRegionVOff   : per-region (regionCount) contour-vertex offset, READ.
//   b2 gRegionVCnt   : per-region (regionCount) contour-vertex count, READ.
//   b3 gPolys        : flat uint poly buffer (8 uints/poly), capacity gParams.cap polys, WRITE.
//   b4 gRegionPOff   : per-region (regionCount) poly offset, WRITE.
//   b5 gRegionPCnt   : per-region (regionCount) poly count, WRITE.
//   b6 gParams       : { regionCount, polyCap, enabled, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_NAV_NO_NEIGHBOUR 0xFFFFFFFFu

struct NavPolyParams {
    int4 cfg;   // x=regionCount, y=polyCap, z=enabled, w=pad
};

[[vk::binding(0, 0)]] RWStructuredBuffer<int2>          gContourVerts : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>          gRegionVOff   : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>          gRegionVCnt   : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>          gPolys        : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<uint>          gRegionPOff   : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<uint>          gRegionPCnt   : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<NavPolyParams> gParams       : register(u6);

int Cross2(int ax, int az, int bx, int bz, int px, int pz) {
    return (bx - ax) * (pz - az) - (bz - az) * (px - ax);
}

bool PointInTri(int2 A, int2 B, int2 C, int2 P) {
    int d0 = Cross2(A.x, A.y, B.x, B.y, P.x, P.y);
    int d1 = Cross2(B.x, B.y, C.x, C.y, P.x, P.y);
    int d2 = Cross2(C.x, C.y, A.x, A.y, P.x, P.y);
    bool anyNeg = (d0 < 0) || (d1 < 0) || (d2 < 0);
    bool anyPos = (d0 > 0) || (d1 > 0) || (d2 > 0);
    return !(anyNeg && anyPos);
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // SINGLE THREAD: only thread 0 runs the serial triangulation.

    int regionCount = gParams[0].cfg.x;
    int polyCap     = gParams[0].cfg.y;
    int enabled     = gParams[0].cfg.z;

    for (int r = 0; r < regionCount; ++r) { gRegionPOff[r] = 0u; gRegionPCnt[r] = 0u; }
    if (enabled == 0) return;   // disabled -> all counts 0 (the byte-identical no-op).

    uint polyWrite = 0u;        // running base into gPolys (poly id).

    const int kMaxN = 512;      // max contour verts (matches nav_contour).
    int2  v[kMaxN];
    int   rem[kMaxN];

    for (int R = 1; R <= regionCount; ++R) {
        uint vOff = gRegionVOff[R - 1];
        int  n    = (int)gRegionVCnt[R - 1];
        gRegionPOff[R - 1] = polyWrite;
        if (n < 3) { gRegionPCnt[R - 1] = 0u; continue; }
        if (n > kMaxN) n = kMaxN;
        for (int i = 0; i < n; ++i) v[i] = gContourVerts[vOff + (uint)i];

        // Loop winding (shoelace via Cross2 from vertex 0).
        int area2 = 0;
        for (int i2 = 1; i2 + 1 < n; ++i2)
            area2 += Cross2(v[0].x, v[0].y, v[i2].x, v[i2].y, v[i2 + 1].x, v[i2 + 1].y);
        int windSign = (area2 > 0) ? 1 : -1;

        for (int i3 = 0; i3 < n; ++i3) rem[i3] = i3;
        int remN = n;
        uint firstPoly = polyWrite;

        int guard = 0;
        int guardMax = n * n + 16;
        while (remN > 3 && guard < guardMax) {
            guard += 1;
            int earAt = -1;
            for (int i = 0; i < remN; ++i) {
                int ip = rem[(i + remN - 1) % remN];
                int ic = rem[i];
                int in_ = rem[(i + 1) % remN];
                int2 A = v[ip]; int2 B = v[ic]; int2 C = v[in_];
                int cr = Cross2(A.x, A.y, B.x, B.y, C.x, C.y);
                if (cr == 0) continue;
                if ((cr > 0 ? 1 : -1) != windSign) continue;
                bool clean = true;
                for (int j = 0; j < remN && clean; ++j) {
                    int vj = rem[j];
                    if (vj == ip || vj == ic || vj == in_) continue;
                    if (PointInTri(A, B, C, v[vj])) clean = false;
                }
                if (clean) { earAt = i; break; }
            }
            if (earAt < 0) break;
            int ip2 = rem[(earAt + remN - 1) % remN];
            int ic2 = rem[earAt];
            int in2 = rem[(earAt + 1) % remN];
            if (polyWrite < (uint)polyCap) {
                gPolys[polyWrite * 8u + 0u] = (uint)ip2;
                gPolys[polyWrite * 8u + 1u] = (uint)ic2;
                gPolys[polyWrite * 8u + 2u] = (uint)in2;
                gPolys[polyWrite * 8u + 3u] = HF_NAV_NO_NEIGHBOUR;
                gPolys[polyWrite * 8u + 4u] = HF_NAV_NO_NEIGHBOUR;
                gPolys[polyWrite * 8u + 5u] = HF_NAV_NO_NEIGHBOUR;
                gPolys[polyWrite * 8u + 6u] = (uint)R;
                gPolys[polyWrite * 8u + 7u] = 0u;
            }
            polyWrite += 1u;
            // erase earAt from rem.
            for (int e = earAt; e < remN - 1; ++e) rem[e] = rem[e + 1];
            remN -= 1;
        }
        // Final triangle / fan leftover.
        if (remN == 3) {
            if (polyWrite < (uint)polyCap) {
                gPolys[polyWrite * 8u + 0u] = (uint)rem[0];
                gPolys[polyWrite * 8u + 1u] = (uint)rem[1];
                gPolys[polyWrite * 8u + 2u] = (uint)rem[2];
                gPolys[polyWrite * 8u + 3u] = HF_NAV_NO_NEIGHBOUR;
                gPolys[polyWrite * 8u + 4u] = HF_NAV_NO_NEIGHBOUR;
                gPolys[polyWrite * 8u + 5u] = HF_NAV_NO_NEIGHBOUR;
                gPolys[polyWrite * 8u + 6u] = (uint)R;
                gPolys[polyWrite * 8u + 7u] = 0u;
            }
            polyWrite += 1u;
        } else if (remN > 3) {
            for (int f = 1; f + 1 < remN; ++f) {
                if (polyWrite < (uint)polyCap) {
                    gPolys[polyWrite * 8u + 0u] = (uint)rem[0];
                    gPolys[polyWrite * 8u + 1u] = (uint)rem[f];
                    gPolys[polyWrite * 8u + 2u] = (uint)rem[f + 1];
                    gPolys[polyWrite * 8u + 3u] = HF_NAV_NO_NEIGHBOUR;
                    gPolys[polyWrite * 8u + 4u] = HF_NAV_NO_NEIGHBOUR;
                    gPolys[polyWrite * 8u + 5u] = HF_NAV_NO_NEIGHBOUR;
                    gPolys[polyWrite * 8u + 6u] = (uint)R;
                    gPolys[polyWrite * 8u + 7u] = 0u;
                }
                polyWrite += 1u;
            }
        }

        uint lastPoly = polyWrite;
        gRegionPCnt[R - 1] = lastPoly - firstPoly;

        // Per-edge adjacency within this contour.
        for (uint pi = firstPoly; pi < lastPoly; ++pi) {
            if (pi >= (uint)polyCap) break;
            for (int e2 = 0; e2 < 3; ++e2) {
                if (gPolys[pi * 8u + 3u + (uint)e2] != HF_NAV_NO_NEIGHBOUR) continue;
                uint a = gPolys[pi * 8u + (uint)e2];
                uint b = gPolys[pi * 8u + (uint)((e2 + 1) % 3)];
                for (uint qi = firstPoly; qi < lastPoly; ++qi) {
                    if (qi == pi || qi >= (uint)polyCap) continue;
                    if (gPolys[pi * 8u + 3u + (uint)e2] != HF_NAV_NO_NEIGHBOUR) break;
                    for (int f2 = 0; f2 < 3; ++f2) {
                        uint qa = gPolys[qi * 8u + (uint)f2];
                        uint qb = gPolys[qi * 8u + (uint)((f2 + 1) % 3)];
                        if (qa == b && qb == a) {
                            gPolys[pi * 8u + 3u + (uint)e2] = qi;
                            gPolys[qi * 8u + 3u + (uint)f2] = pi;
                            break;
                        }
                    }
                }
            }
        }
    }
}
