// Slice VT4 — Runtime Virtual Texturing Slice 4: material-pass SAMPLE through the virtual->physical
// INDIRECTION. ONE thread per VIRTUAL texel (a race-free parallel MAP, NO sequential dependency, NO
// atomics). Each thread owns one virtual texel (vx,vy) of the mip being reconstructed: it computes the
// virtual page (px=vx/pageSize, py=vy/pageSize), the pageId (PageId = mipOffset(mip)+py*pps+px), reads
// gIndirection[pageId] to recover the physical TILE, and if the tile is resident offsets into the physical
// atlas (PhysicalTileOrigin -> (ox,oy)) and reads gAtlas[(oy+ly)*atlasW + (ox+lx)] (NEAREST — a direct
// integer texel read, NO filtering); a non-resident page (tile==kNoTile) -> kVtMiss. The reconstructed
// virtual image is written to gImage (the virtual texture as the user ADDRESSES it — pages in virtual-UV
// layout, NOT the physical-atlas tile-alloc order).
//
// SampleVirtualTexel here is the VERBATIM mirror of engine/render/vt.h::SampleVirtualTexel (the per-mip
// pagesPerSide / mipOffset tables come from the host-precomputed gParams, exactly like vt_pagegen.comp).
// PURE INTEGER (no float) -> the reconstructed image is bit-identical CPU<->Vulkan<->Metal; the default MSL
// gen suffices (no atomics, no integer texture.read -> NO --msl-version 20200). A mismatch shows up as a
// wrong texel -> the host's GPU==CPU memcmp fails loudly.
//
// sampleEnabled push flag (gParams.dims.w): 0 -> EVERY thread writes kVtMiss (the disabled-path no-op),
// written explicitly so the read-back proves the no-op regardless of the upload.
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3)):
//   b0 gIndirection : the VT2 indirection table, indirection[pageId] = tileIndex (0xFFFFFFFF == kNoTile), READ.
//   b1 gAtlas       : the VT3 physical atlas, atlasW*atlasH RGBA8-packed uints, READ.
//   b2 gImage       : the reconstructed virtual image, virtualW*virtualH RGBA8-packed uints, WRITE.
//   b3 gParams      : { virtualW, virtualH, atlasW, sampleEnabled } + { mip, pageSize, tilesPerSide, mipLevels }
//                     + the per-mip pagesPerSide + mipOffset tables for PageId (host integers), READ.

#define HF_VT_MAX_MIPS 16
#define HF_VT_S_THREADS 64

static const uint kVtMiss     = 0xFFFF00FFu;  // == vt.h::kVtMiss (magenta miss tint)
static const uint kNoTileU32  = 0xFFFFFFFFu;  // == vt.h::kNoTileU32

// VT config + the host-precomputed per-mip integer tables (std430). Mirrors the C++ upload struct.
//   dims  : x=virtualW, y=virtualH, z=atlasW, w=sampleEnabled
//   dims2 : x=mip, y=pageSize, z=tilesPerSide, w=mipLevels
//   pagesPerSide[m] = vt.pagesPerSide(m); mipOffset[m] = vt.mipPageOffset(m)  (packed uint4[])
struct Params {
    uint4 dims;
    uint4 dims2;
    uint4 pagesPerSide[HF_VT_MAX_MIPS / 4];   // 16 uints = HF_VT_MAX_MIPS entries
    uint4 mipOffset[HF_VT_MAX_MIPS / 4];      // 16 uints = HF_VT_MAX_MIPS entries
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   gIndirection : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gAtlas       : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>   gImage       : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<Params> gParams      : register(u3);

int PagesPerSideAt(int m) { return (int)gParams[0].pagesPerSide[m >> 2][m & 3]; }
int MipOffsetAt(int m)    { return (int)gParams[0].mipOffset[m >> 2][m & 3]; }

// PageId (the VERBATIM mirror of vt.h::PageId): mipOffset(mip) + py*pagesPerSide(mip) + px.
int PageId(int mip, int px, int py) {
    int pps = PagesPerSideAt(mip);
    return MipOffsetAt(mip) + py * pps + px;
}

// THE sample-through-indirection round-trip — COPIED VERBATIM from engine/render/vt.h::SampleVirtualTexel.
// PURE INTEGER (no float) -> bit-identical CPU<->Vulkan<->Metal. Returns the RGBA8-packed sampled texel, or
// kVtMiss for a non-resident page.
uint SampleVirtualTexel(int vx, int vy, int mip, int pageSize, int tilesPerSide, int atlasW, int nPages) {
    int px = vx / pageSize;
    int py = vy / pageSize;
    int pageId = PageId(mip, px, py);
    if (pageId < 0 || pageId >= nPages) return kVtMiss;
    uint tile = gIndirection[(uint)pageId];
    if (tile == kNoTileU32) return kVtMiss;          // kNoTile (non-resident OR overflowed)
    int ox = ((int)tile % tilesPerSide) * pageSize;  // PhysicalTileOrigin
    int oy = ((int)tile / tilesPerSide) * pageSize;
    int lx = vx % pageSize;
    int ly = vy % pageSize;
    return gAtlas[(uint)((oy + ly) * atlasW + (ox + lx))];
}

// ONE thread per virtual texel, indexed LINEARLY (texelIdx = vy*virtualW + vx) over a 1D dispatch — matching
// the engine's 1D-dispatch / threadsPerGroupX convention (vt_feedback.comp / vt_pagegen.comp). divmod -> (vx,vy).
[numthreads(HF_VT_S_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint virtualW     = gParams[0].dims.x;
    uint virtualH     = gParams[0].dims.y;
    uint atlasW       = gParams[0].dims.z;
    uint sampleEnabled = gParams[0].dims.w;
    int  mip          = (int)gParams[0].dims2.x;
    int  pageSize     = (int)gParams[0].dims2.y;
    int  tilesPerSide = (int)gParams[0].dims2.z;
    int  mipLevels    = (int)gParams[0].dims2.w;

    uint texelIdx = gid.x;
    if (texelIdx >= virtualW * virtualH) return;
    uint vx = texelIdx % virtualW;
    uint vy = texelIdx / virtualW;

    uint outIdx = texelIdx;  // == vy * virtualW + vx
    if (sampleEnabled == 0u) { gImage[outIdx] = kVtMiss; return; }   // disabled-path no-op

    int nPages = MipOffsetAt(mipLevels - 1) + PagesPerSideAt(mipLevels - 1) * PagesPerSideAt(mipLevels - 1);
    gImage[outIdx] = SampleVirtualTexel((int)vx, (int)vy, mip, pageSize, tilesPerSide, (int)atlasW, nPages);
}
