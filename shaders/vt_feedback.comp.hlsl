// Slice VT1 — Runtime Virtual Texturing Slice 1: the PAGE-NEEDED FEEDBACK/MARKING compute pass. ONE
// thread per sample-request. Each thread reads its HOST-SNAPPED integer page coords (mip,px,py) — the
// host already ran the UV->page quantization (vt.h::VtPageId / SnapRequest), so the GPU does ZERO
// floating point — computes pageId = mipPageOffset(mip) + py*pagesPerSide(mip) + px (PageId, copied
// VERBATIM from engine/render/vt.h over host-precomputed integer per-mip page offsets + pages-per-side
// in gParams), and sets feedback[pageId] = 1. The marked set is a pure INTEGER set (order-independent),
// so writes race-free to 1 and the result is bit-identical GPU==CPU + cross-backend.
//
// The page math (PageId) is the VERBATIM mirror of engine/render/vt.h. Because the host snaps each
// request to integer page coords (the swraster.h host-snap discipline), there is NO float on the GPU's
// bit-exact path at all — pageId is a pure integer multiply/add over the request's (mip,px,py) and the
// host-precomputed gParams.pagesPerSide / gParams.mipOffset tables. A mismatch shows up as a wrong page
// marked -> the host's GPU==CPU memcmp fails loudly.
//
// feedbackEnabled push flag (in gParams.dims.w): false -> every thread returns early, writing NOTHING
// -> the feedback set stays the cleared all-zero upload (the byte-identical disabled-path proof).
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gRequests : the host-snapped sample requests (int4: x=mip, y=px, z=py, w unused), READ.
//   b1 gFeedback : the virtual page-table feedback set, feedback[pageId] in {0,1}, WRITE.
//   b2 gParams   : vt config (mipLevels/requestCount/feedbackEnabled) + the per-mip pagesPerSide +
//                  mipPageOffset tables (host-precomputed integers; CPU + shader read the same), READ.

#define HF_VT_MAX_MIPS 16
#define HF_VT_THREADS  64

// VT config + the host-precomputed per-mip integer tables (std430). Mirrors the C++ upload struct.
//   dims          : x=mipLevels, y=requestCount, z=feedbackEnabled, w=unused
//   pagesPerSide  : pagesPerSide[m] (packed uint4[]) = vt.pagesPerSide(m)
//   mipOffset     : mipOffset[m]    (packed uint4[]) = vt.mipPageOffset(m)
struct Params {
    uint4 dims;
    uint4 pagesPerSide[HF_VT_MAX_MIPS / 4];   // 16 uints = HF_VT_MAX_MIPS entries
    uint4 mipOffset[HF_VT_MAX_MIPS / 4];      // 16 uints = HF_VT_MAX_MIPS entries
};

[[vk::binding(0, 0)]] RWStructuredBuffer<int4>   gRequests : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gFeedback : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params> gParams   : register(u2);

// Read pagesPerSide[m] / mipOffset[m] from the uint4[] packs (table[m/4][m%4]).
int PagesPerSideAt(int m) { return (int)gParams[0].pagesPerSide[m >> 2][m & 3]; }
int MipOffsetAt(int m)    { return (int)gParams[0].mipOffset[m >> 2][m & 3]; }

// Flat page-table index: idx = mipPageOffset(mip) + py*pagesPerSide(mip) + px. Mirrors vt.h::PageId
// VERBATIM (the per-mip offset + pagesPerSide come from the host-precomputed gParams tables).
int PageId(int mip, int px, int py) {
    int pps = PagesPerSideAt(mip);
    return MipOffsetAt(mip) + py * pps + px;
}

[numthreads(HF_VT_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint requestCount   = gParams[0].dims.y;
    uint feedbackEnabled = gParams[0].dims.z;   // 0 -> write nothing (disabled-path no-op)

    uint i = gid.x;
    if (i >= requestCount) return;
    if (feedbackEnabled == 0u) return;   // disabled -> feedback stays the cleared all-zero upload

    // The host-snapped integer page coords (the GPU does ZERO float). PageId copied VERBATIM from vt.h.
    int4 req = gRequests[i];
    int mip = req.x;
    int px  = req.y;
    int py  = req.z;
    int pageId = PageId(mip, px, py);

    // The set is order-independent: a plain write to 1 (multiple requests can map to the same page;
    // every write stores the same value). No atomics needed.
    gFeedback[pageId] = 1u;
}
