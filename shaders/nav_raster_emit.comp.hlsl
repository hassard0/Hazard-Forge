// Slice NAV1 — Deterministic GPU Navmesh BEACHHEAD: the per-COLUMN span-EMIT compute pass (the MC3
// mc_emit / FPX2 fpx_pair_emit analog on columns). ONE thread per column (col < columnCount). The
// thread decomposes col -> (cx,cz), reads its write base base = gColOffset[col] (the NAV1 prefix-sum),
// re-scans every input triangle in the FIXED triangle order, and EMITS each covering triangle's integer
// y-span {ymin, ymax, area} into gSpans at base + local++. Each column writes into its OWN DISJOINT
// [base, base+count) range -> race-free, NO atomics. enabled=0 -> emit nothing (gSpans stays the
// pre-cleared upload).
//
// WHY BIT-IDENTICAL to the CPU navmesh.h::RasterizeTriangleSpans (the make-or-break): every value
// written is PURE INT32 — the same TriColumnAabb clamp + three-edge PointInTriXZ cover test over the
// same host-snapped voxel coords the CPU uses (CoversColumn copied VERBATIM from nav_raster_count), the
// y-span is the integer min/max of the tri's 3 voxel-y verts (TriYSpan VERBATIM), the triangle scan is
// the SAME fixed order, and each column's range is disjoint so a thread race CANNOT change any byte.
// The grouped-by-column then triangle-ascending order matches RasterizeTriangleSpans exactly. INT32
// only -> MSL-gens natively on Metal. (Span MERGING is a column-local, variable-length CPU finalize
// step — navmesh.h::MergeColumnSpans — NOT part of this raw bit-exact buffer.)
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3)):
//   b0 gTris      : the int32 NavTri array (3 verts x int3, std430), READ.
//   b1 gColOffset : one uint per column (the NAV1 prefix-sum write base), READ.
//   b2 gSpans     : the output Span array {ymin, ymax, area} per slot, WRITE (pre-cleared).
//   b3 gParams    : { w, h, triCount, enabled }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations.

#define HF_NAV_THREADS 64

// std430 NavTri mirror — IDENTICAL to nav_raster_count.comp's NavTri (9 x int32, 36 bytes).
struct NavTri {
    int v0x, v0y, v0z;
    int v1x, v1y, v1z;
    int v2x, v2y, v2z;
};

// std430 Span mirror (engine/nav/navmesh.h::Span): 3 x uint = 12 bytes.
struct Span { uint ymin; uint ymax; uint area; };

// Params (std430). Mirrors the C++ upload struct.
//   cfg : x=w, y=h, z=triCount, w=enabled
struct NavParams { int4 cfg; };

[[vk::binding(0, 0)]] RWStructuredBuffer<NavTri>    gTris      : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>      gColOffset : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Span>      gSpans     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<NavParams> gParams    : register(u3);

static const uint kDefaultArea = 1u;   // VERBATIM navmesh.h::kDefaultArea (walkable-flag placeholder)

// PointInTriXZ — VERBATIM navmesh.h::PointInTriXZ (int32 products; see nav_raster_count's bound).
bool PointInTriXZ(int px, int pz, NavTri t) {
    int e0 = (t.v1x - t.v0x) * (pz - t.v0z) - (t.v1z - t.v0z) * (px - t.v0x);
    int e1 = (t.v2x - t.v1x) * (pz - t.v1z) - (t.v2z - t.v1z) * (px - t.v1x);
    int e2 = (t.v0x - t.v2x) * (pz - t.v2z) - (t.v0z - t.v2z) * (px - t.v2x);
    bool allNonNeg = (e0 >= 0) && (e1 >= 0) && (e2 >= 0);
    bool allNonPos = (e0 <= 0) && (e1 <= 0) && (e2 <= 0);
    return allNonNeg || allNonPos;
}

// CoversColumn — VERBATIM nav_raster_count.comp::CoversColumn (TriColumnAabb clamp + cover test).
bool CoversColumn(int cx, int cz, NavTri t, int w, int h) {
    int minX = t.v0x, maxX = t.v0x, minZ = t.v0z, maxZ = t.v0z;
    if (t.v1x < minX) minX = t.v1x; if (t.v1x > maxX) maxX = t.v1x;
    if (t.v2x < minX) minX = t.v2x; if (t.v2x > maxX) maxX = t.v2x;
    if (t.v1z < minZ) minZ = t.v1z; if (t.v1z > maxZ) maxZ = t.v1z;
    if (t.v2z < minZ) minZ = t.v2z; if (t.v2z > maxZ) maxZ = t.v2z;
    if (minX < 0) minX = 0;
    if (minZ < 0) minZ = 0;
    if (maxX > w - 1) maxX = w - 1;
    if (maxZ > h - 1) maxZ = h - 1;
    if (minX > maxX || minZ > maxZ) return false;
    if (cx < minX || cx > maxX || cz < minZ || cz > maxZ) return false;
    return PointInTriXZ(cx, cz, t);
}

// TriYSpan — VERBATIM navmesh.h::TriYSpan (integer min/max of the 3 voxel-y verts).
void TriYSpan(NavTri t, out int ymin, out int ymax) {
    ymin = t.v0y; ymax = t.v0y;
    if (t.v1y < ymin) ymin = t.v1y; if (t.v1y > ymax) ymax = t.v1y;
    if (t.v2y < ymin) ymin = t.v2y; if (t.v2y > ymax) ymax = t.v2y;
}

[numthreads(HF_NAV_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int w        = gParams[0].cfg.x;
    int h        = gParams[0].cfg.y;
    int triCount = gParams[0].cfg.z;
    int enabled  = gParams[0].cfg.w;

    uint col = gid.x;
    int columnCount = w * h;
    if ((int)col >= columnCount) return;

    // Disabled -> emit nothing (gSpans stays the pre-cleared upload; the byte-identical no-op).
    if (enabled == 0) return;

    // Decompose col -> (cx,cz). col = cz*w + cx (VERBATIM navmesh.h::columnId).
    int cx = (int)col % w;
    int cz = (int)col / w;

    uint base  = gColOffset[col];   // this column's disjoint write base (from the prefix-sum)
    uint local = 0u;
    for (int ti = 0; ti < triCount; ++ti) {
        NavTri t = gTris[(uint)ti];
        if (!CoversColumn(cx, cz, t, w, h)) continue;
        int ymin, ymax;
        TriYSpan(t, ymin, ymax);
        Span s;
        s.ymin = (uint)ymin;
        s.ymax = (uint)ymax;
        s.area = kDefaultArea;
        gSpans[base + local] = s;
        local += 1u;
    }
}
