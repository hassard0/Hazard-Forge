// Slice SW2 — Nanite Software-Raster Slice 2: the GPU COMPUTE software rasterizer. ONE thread per
// cluster-triangle. Each thread reads its 3 HOST-SNAPPED integer ScreenVerts + its packed visId, runs
// the VERBATIM SW1 (engine/render/swraster.h) integer edge / top-left fill-rule / flat-min-depth
// rasterizer over the triangle's pixel bbox, and per covered pixel does
// InterlockedMin(gVis[y*w + x], PackSw(depthQ, visId)) into the w*h-entry visibility-buffer SSBO.
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it
// consumes the SAME integer ScreenVerts the host snapped (ProjectToScreenVert is host-only) and runs the
// SAME int64 integer edge functions + the SAME top-left fill rule + the SAME flat min(z) the CPU
// swraster.h::RasterTriangle runs. The merge is a COMMUTATIVE InterlockedMin over the 32-bit depth|id
// key (the GPU mirror of the CPU's serial std::min) — so the thread-race order CANNOT change the result.
// Integer coverage + integer depth + a commutative atomic => the GPU vis-buffer is provably == the CPU
// reference at EVERY pixel, on any vendor (Vulkan DXC / Metal spirv-cross), full-frame (not interior-only:
// the integer edge test is exact at edges too, unlike a hardware rasterizer).
//
// INTEGER WIDTH (the determinism crux): the edge function uses int64_t — IDENTICAL to swraster.h::SwEdge
// (which uses int64_t). With w,h <= 4096 and kSub=16 the snapped coords are < 2^16, the edge differences
// < 2^17, the products < 2^34 (overflows int32) — so int64 is REQUIRED to match the CPU, and the bound
// keeps it well inside int64. HLSL SM6 supports int64_t (DXC -spirv with the Int64 capability; spirv-cross
// lowers it to MSL `long`). InterlockedMin on a device uint buffer is plain MSL (no MSL-2.2 needed).
//
// The math below is COPIED VERBATIM from engine/render/swraster.h (SwEdge / SwIsTopLeft / RasterTriangle's
// orient+bbox+top-left+flat-min loop). A divergence here vs the header is exactly what the host's
// GPU==CPU memcmp catches.
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3)):
//   b0 gScreenVerts : the host-snapped integer screen verts (int x, int y, uint z), std430, READ.
//   b1 gTriangles   : per-triangle { i0, i1, i2, visId }, std430, READ.
//   b2 gVis         : the w*h visibility buffer (packed depth|id), InterlockedMin WRITE.
//   b3 gParams      : { width, height, triCount, _ }, READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk/MSL mention is the [[vk::binding]] decorations
// (same as vsm_mark.comp / autoexposure_histogram.comp), not backend CODE symbols.

#define HF_SWR_THREADS 64

// kSub MUST match swraster.h::kSub (sub-pixel resolution). 16 -> 4 sub-pixel bits.
#define HF_SWR_KSUB 16

// kSwIdBits MUST match swraster.h::kSwIdBits (depth|id packing budget). PackSw shifts depth by this.
#define HF_SWR_ID_BITS 16
#define HF_SWR_ID_MASK 0xFFFFu

// std430 ScreenVert mirror (swraster.h::ScreenVert): int x, int y, uint z (12 bytes, 4-byte aligned).
struct SwScreenVert {
    int  x;   // fixed-point pixel-x in 1/kSub units
    int  y;   // fixed-point pixel-y in 1/kSub units
    uint z;   // quantized depth in [0,kSwDepthMax] (nearer = smaller)
};

// Per-triangle record: the 3 vertex indices into gScreenVerts + the packed visId (PackVisId(cluster,tri)).
struct SwTriangle {
    uint i0;
    uint i1;
    uint i2;
    uint visId;
};

struct SwParams {
    uint4 dims;   // x=width, y=height, z=triCount, w=unused
};

[[vk::binding(0, 0)]] RWStructuredBuffer<SwScreenVert> gScreenVerts : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<SwTriangle>   gTriangles   : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>         gVis         : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<SwParams>     gParams      : register(u3);

// VERBATIM swraster.h::PackSw — depthQ HIGH bits, visId LOW kSwIdBits.
uint PackSw(uint depthQ, uint visId) {
    return (depthQ << HF_SWR_ID_BITS) | (visId & HF_SWR_ID_MASK);
}

// VERBATIM swraster.h::SwEdge — E(p,q,r) with int64 intermediates (matches the CPU width EXACTLY).
int64_t SwEdge(int px, int py, int qx, int qy, int rx, int ry) {
    return (int64_t)(px - qx) * (int64_t)(ry - qy) - (int64_t)(py - qy) * (int64_t)(rx - qx);
}

// VERBATIM swraster.h::SwIsTopLeft — a left edge goes up (dy<0); a top edge is horizontal-left (dy==0&&dx<0).
bool SwIsTopLeft(int ax, int ay, int bx, int by) {
    int dy = by - ay;
    int dx = bx - ax;
    return (dy < 0) || (dy == 0 && dx < 0);
}

// VERBATIM swraster.h::RasterTriangle's floorDiv — floor division for positive d, any sign n.
int FloorDiv(int n, int d) {
    int q = n / d;
    int r = n % d;
    return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
}

[numthreads(HF_SWR_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint width    = gParams[0].dims.x;
    uint height   = gParams[0].dims.y;
    uint triCount = gParams[0].dims.z;

    uint t = gid.x;
    if (t >= triCount) return;

    SwTriangle tri = gTriangles[t];
    SwScreenVert a = gScreenVerts[tri.i0];
    SwScreenVert b = gScreenVerts[tri.i1];
    SwScreenVert c = gScreenVerts[tri.i2];
    uint visId = tri.visId;

    // Signed area*2 (the (a,b,c) edge). Zero -> degenerate, skip. (VERBATIM swraster.h::RasterTriangle.)
    int64_t area2 = SwEdge(a.x, a.y, b.x, b.y, c.x, c.y);
    if (area2 == 0) return;

    // Orient so the triangle is CCW (area2 > 0) — swap b,c if needed so E>0 means inside for all 3 edges.
    SwScreenVert v0 = a, v1 = b, v2 = c;
    if (area2 < 0) { SwScreenVert tmp = v1; v1 = v2; v2 = tmp; }

    // Integer pixel bounding box (verts in 1/kSub units; divide to pixel units, floor min, ceil max),
    // clamped to [0,w)x[0,h).
    int minXf = min(v0.x, min(v1.x, v2.x));
    int maxXf = max(v0.x, max(v1.x, v2.x));
    int minYf = min(v0.y, min(v1.y, v2.y));
    int maxYf = max(v0.y, max(v1.y, v2.y));
    int minX = FloorDiv(minXf, HF_SWR_KSUB);
    int maxX = FloorDiv(maxXf + (HF_SWR_KSUB - 1), HF_SWR_KSUB);  // ceil
    int minY = FloorDiv(minYf, HF_SWR_KSUB);
    int maxY = FloorDiv(maxYf + (HF_SWR_KSUB - 1), HF_SWR_KSUB);
    if (minX < 0) minX = 0;
    if (minY < 0) minY = 0;
    if (maxX > (int)width - 1)  maxX = (int)width - 1;
    if (maxY > (int)height - 1) maxY = (int)height - 1;
    if (minX > maxX || minY > maxY) return;

    // Flat per-triangle depth: the nearest (smallest) of the 3 quantized depths.
    uint depthQ = min(v0.z, min(v1.z, v2.z));
    uint key = PackSw(depthQ, visId);

    // The three edges, each E_i opposite vertex i (consistent CCW winding).
    bool tl0 = SwIsTopLeft(v1.x, v1.y, v2.x, v2.y);
    bool tl1 = SwIsTopLeft(v2.x, v2.y, v0.x, v0.y);
    bool tl2 = SwIsTopLeft(v0.x, v0.y, v1.x, v1.y);

    for (int y = minY; y <= maxY; ++y) {
        int cy = y * HF_SWR_KSUB + HF_SWR_KSUB / 2;  // pixel-center in fixed-point
        for (int x = minX; x <= maxX; ++x) {
            int cx = x * HF_SWR_KSUB + HF_SWR_KSUB / 2;
            int64_t e0 = SwEdge(cx, cy, v1.x, v1.y, v2.x, v2.y);
            int64_t e1 = SwEdge(cx, cy, v2.x, v2.y, v0.x, v0.y);
            int64_t e2 = SwEdge(cx, cy, v0.x, v0.y, v1.x, v1.y);
            bool in0 = (e0 > 0) || (e0 == 0 && tl0);
            bool in1 = (e1 > 0) || (e1 == 0 && tl1);
            bool in2 = (e2 > 0) || (e2 == 0 && tl2);
            if (in0 && in1 && in2) {
                // The GPU mirror of the CPU's serial min — commutative -> order-independent.
                InterlockedMin(gVis[(uint)y * width + (uint)x], key);
            }
        }
    }
}
