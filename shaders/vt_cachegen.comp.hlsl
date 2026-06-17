// Slice VT5 — Runtime Virtual Texturing Slice 5: per-page CACHING across frames. The cache-aware sibling of
// VT3's vt_pagegen.comp (which is UNCHANGED). ONE thread per atlas texel (a race-free parallel MAP, NO
// atomics): each thread finds its physical TILE (tx=x/pageSize, ty=y/pageSize), and:
//   * if the tile is UNALLOCATED (gTilePageId == 0xFFFFFFFF) -> RETURN without writing (the atlas SSBO
//     PERSISTS — the texel keeps its prior value; the caller pre-clears to kAtlasClear on the first populate).
//   * if the tile is CACHED (gNeedsGen[tileIndex] == 0) -> RETURN without writing (the cache HIT: the tile's
//     prior generated texels carry over BYTE-IDENTICAL — the perf win of the runtime virtual texture).
//   * else (gNeedsGen[tileIndex] != 0, a cache MISS) -> UnpackPageId the owning pageId -> (mip,px,py) and
//     write PageTexelV(pageId,mip,localXY, gTileVersion[tileIndex]) (the content-version-aware generator,
//     copied VERBATIM from engine/render/vt.h::PageTexelV) into gAtlas.
//
// The atlas SSBO is NOT cleared between passes (the host persists it), so cached + unallocated texels survive
// untouched — that is what makes a partial (cache-aware) regen BYTE-IDENTICAL to a full regen. PageTexelV +
// UnpackPageId are the VERBATIM mirror of engine/render/vt.h (the per-mip pagesPerSide / mipOffset tables
// come from gParams, exactly like vt_pagegen.comp). PURE INTEGER (no float, no atomics, no integer
// texture.read) -> the atlas is bit-identical GPU==CPU AND cross-backend; the default MSL gen suffices (NO
// --msl-version 20200). A mismatch shows up as a wrong texel -> the host's GPU==CPU memcmp fails loudly.
//
// Buffers (storage, bound at compute bindings 0..3; on Metal these land at buffer(0..3)):
//   b0 gTilePageId : the host-built reverse table, tilePageId[tileIndex] = pageId (0xFFFFFFFF if free), READ.
//   b1 gNeedsGen   : per-tile flag, 1 = regenerate (cache miss / first populate), 0 = cached/skip, READ.
//   b2 gTileVersion: per-tile contentVersion (folds into PageTexelV's seed), READ.
//   b3 gAtlas      : the PERSISTENT physical atlas, atlasW*atlasH RGBA8-packed uints, READ-MODIFY-WRITE.
//   b4 gParams     : { atlasW, atlasH, tilesPerSide, pageSize, mipLevels } + the per-mip pagesPerSide +
//                    mipOffset tables for UnpackPageId (host-precomputed integers; CPU + shader read same), READ.

#define HF_VT_MAX_MIPS 16
#define HF_VT_CG_THREADS 64

// VT config + the host-precomputed per-mip integer tables (std430). Mirrors the C++ upload struct.
//   dims  : x=atlasW, y=atlasH, z=tilesPerSide, w=unused
//   dims2 : x=pageSize, y=mipLevels, z=unused, w=unused
//   pagesPerSide[m] = vt.pagesPerSide(m); mipOffset[m] = vt.mipPageOffset(m)  (packed uint4[])
struct Params {
    uint4 dims;
    uint4 dims2;
    uint4 pagesPerSide[HF_VT_MAX_MIPS / 4];   // 16 uints = HF_VT_MAX_MIPS entries
    uint4 mipOffset[HF_VT_MAX_MIPS / 4];      // 16 uints = HF_VT_MAX_MIPS entries
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   gTilePageId  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gNeedsGen    : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>   gTileVersion : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>   gAtlas       : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<Params> gParams      : register(u4);

int PagesPerSideAt(int m) { return (int)gParams[0].pagesPerSide[m >> 2][m & 3]; }
int MipOffsetAt(int m)    { return (int)gParams[0].mipOffset[m >> 2][m & 3]; }

// Inverse of PageId (the VERBATIM mirror of vt.h::UnpackPageId): walk the mips subtracting each mip's page
// span until `id` lands inside one, then (py,px) = divmod by that mip's pagesPerSide.
void UnpackPageId(int id, int mipLevels, out int mip, out int px, out int py) {
    int rem = id;
    int m = 0;
    for (; m < mipLevels; ++m) {
        int pps = PagesPerSideAt(m);
        int span = pps * pps;
        if (rem < span) break;
        rem -= span;
    }
    if (m >= mipLevels) m = mipLevels - 1;
    int pps = PagesPerSideAt(m);
    mip = m;
    py = rem / pps;
    px = rem % pps;
}

// THE content-version-aware per-texel generator — COPIED VERBATIM from engine/render/vt.h::PageTexelV. PURE
// INTEGER (no float) -> bit-identical CPU<->Vulkan<->Metal. Returns RGBA8 packed 0xAABBGGRR. At
// contentVersion==0 this reproduces the original VT3 PageTexel exactly (the `+ version*prime` term is 0).
uint PageTexelV(int pageId, int mip, int lx, int ly, uint contentVersion) {
    uint h = (uint)pageId * 2654435761u + (uint)(mip + 1) * 0x9E3779B9u
           + contentVersion * 0x27D4EB2Fu;
    h ^= h >> 15; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    uint br = (h)       & 0xFFu;
    uint bg = (h >> 8)  & 0xFFu;
    uint bb = (h >> 16) & 0xFFu;

    uint chk  = (((uint)lx >> 3) ^ ((uint)ly >> 3)) & 1u;
    uint grad = ((uint)lx >> 4) + ((uint)ly >> 4);

    uint r = chk != 0u ? br : (br * 5u) >> 3;  r += grad * 6u;  if (r > 0xFFu) r = 0xFFu;  if (r < 0x40u) r = 0x40u;
    uint g = chk != 0u ? bg : (bg * 5u) >> 3;  g += grad * 6u;  if (g > 0xFFu) g = 0xFFu;  if (g < 0x40u) g = 0x40u;
    uint b = chk != 0u ? bb : (bb * 5u) >> 3;  b += grad * 6u;  if (b > 0xFFu) b = 0xFFu;  if (b < 0x40u) b = 0x40u;
    return 0xFF000000u | (b << 16) | (g << 8) | r;
}

// ONE thread per atlas texel, indexed LINEARLY (texelIdx = y*atlasW + x) over a 1D dispatch — matching the
// engine's 1D-dispatch / threadsPerGroupX convention (vt_pagegen.comp etc.). texelIdx -> (x,y) by divmod.
[numthreads(HF_VT_CG_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint atlasW      = gParams[0].dims.x;
    uint atlasH      = gParams[0].dims.y;
    uint tilesPerSide = gParams[0].dims.z;
    uint pageSize    = gParams[0].dims2.x;
    int  mipLevels   = (int)gParams[0].dims2.y;

    uint texelIdx = gid.x;
    if (texelIdx >= atlasW * atlasH) return;
    uint x = texelIdx % atlasW;
    uint y = texelIdx / atlasW;

    uint tx = x / pageSize;
    uint ty = y / pageSize;
    uint tileIndex = ty * tilesPerSide + tx;
    uint pageId = gTilePageId[tileIndex];

    if (pageId == 0xFFFFFFFFu) return;          // unallocated tile -> persist (no write)
    if (gNeedsGen[tileIndex] == 0u) return;     // CACHED -> persist (the cache hit; no write)

    int mip, px, py;
    UnpackPageId((int)pageId, mipLevels, mip, px, py);
    int lx = (int)(x % pageSize);
    int ly = (int)(y % pageSize);
    gAtlas[texelIdx] = PageTexelV((int)pageId, mip, lx, ly, gTileVersion[tileIndex]);
}
