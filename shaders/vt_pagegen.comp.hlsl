// Slice VT3 — Runtime Virtual Texturing Slice 3: procedural PAGE GENERATION into the PHYSICAL ATLAS. ONE
// thread per atlas texel (a race-free parallel MAP, NO sequential dependency, NO atomics — unlike VT2's
// serial allocator). Each thread owns one atlas texel (x,y): it finds which physical TILE that texel falls
// in (tx=x/pageSize, ty=y/pageSize), reads the host-built reverse table gTilePageId[ty*tilesPerSide+tx] to
// recover the pageId owning that tile, and if the tile is allocated UnpackPageId's the pageId -> (mip,px,py)
// and writes PageTexel(pageId,mip,localXY) (the page-local texel = (x%pageSize,y%pageSize)) into gAtlas;
// unallocated tiles (gTilePageId == 0xFFFFFFFF) -> kAtlasClear.
//
// PageTexel + UnpackPageId are the VERBATIM mirror of engine/render/vt.h (the per-mip pagesPerSide /
// mipOffset tables come from the host-precomputed gParams, exactly like vt_feedback.comp). PURE INTEGER
// (no float) -> the atlas is bit-identical GPU==CPU AND cross-backend; the default MSL gen suffices (no
// atomics, no integer texture.read -> NO --msl-version 20200). A mismatch shows up as a wrong texel ->
// the host's GPU==CPU memcmp fails loudly.
//
// genEnabled push flag (gParams.dims.w): 0 -> EVERY thread writes kAtlasClear (the disabled-path no-op),
// written explicitly so the read-back proves the no-op regardless of the upload.
//
// Buffers (storage, bound at compute bindings 0..2; on Metal these land at buffer(0..2)):
//   b0 gTilePageId : the host-built reverse table, tilePageId[tileIndex] = pageId (0xFFFFFFFF if free), READ.
//   b1 gAtlas      : the physical atlas, atlasW*atlasH RGBA8-packed uints, WRITE.
//   b2 gParams     : { atlasW, atlasH, tilesPerSide, pageSize, genEnabled } + the per-mip pagesPerSide +
//                    mipOffset tables for UnpackPageId (host-precomputed integers; CPU + shader read same), READ.

#define HF_VT_MAX_MIPS 16
#define HF_VT_PG_THREADS 64

// VT config + the host-precomputed per-mip integer tables (std430). Mirrors the C++ upload struct.
//   dims  : x=atlasW, y=atlasH, z=tilesPerSide, w=genEnabled
//   dims2 : x=pageSize, y=mipLevels, z=unused, w=unused
//   pagesPerSide[m] = vt.pagesPerSide(m); mipOffset[m] = vt.mipPageOffset(m)  (packed uint4[])
struct Params {
    uint4 dims;
    uint4 dims2;
    uint4 pagesPerSide[HF_VT_MAX_MIPS / 4];   // 16 uints = HF_VT_MAX_MIPS entries
    uint4 mipOffset[HF_VT_MAX_MIPS / 4];      // 16 uints = HF_VT_MAX_MIPS entries
};

[[vk::binding(0, 0)]] RWStructuredBuffer<uint>   gTilePageId : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>   gAtlas      : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params> gParams     : register(u2);

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

// THE deterministic per-texel generator — COPIED VERBATIM from engine/render/vt.h::PageTexel. PURE
// INTEGER (no float) -> bit-identical CPU<->Vulkan<->Metal. Returns RGBA8 packed 0xAABBGGRR.
uint PageTexel(int pageId, int mip, int lx, int ly) {
    uint h = (uint)pageId * 2654435761u + (uint)(mip + 1) * 0x9E3779B9u;
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

static const uint kAtlasClear = 0xFF101010u;  // == vt.h::kAtlasClear

// ONE thread per atlas texel, indexed LINEARLY (texelIdx = y*atlasW + x) over a 1D dispatch — matching the
// engine's 1D-dispatch / threadsPerGroupX convention (vt_feedback.comp etc.). texelIdx -> (x,y) by divmod.
[numthreads(HF_VT_PG_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint atlasW      = gParams[0].dims.x;
    uint atlasH      = gParams[0].dims.y;
    uint tilesPerSide = gParams[0].dims.z;
    uint genEnabled  = gParams[0].dims.w;
    uint pageSize    = gParams[0].dims2.x;
    int  mipLevels   = (int)gParams[0].dims2.y;

    uint texelIdx = gid.x;
    if (texelIdx >= atlasW * atlasH) return;
    uint x = texelIdx % atlasW;
    uint y = texelIdx / atlasW;

    uint outIdx = texelIdx;  // == y * atlasW + x
    if (genEnabled == 0u) { gAtlas[outIdx] = kAtlasClear; return; }   // disabled-path no-op

    uint tx = x / pageSize;
    uint ty = y / pageSize;
    uint tileIndex = ty * tilesPerSide + tx;
    uint pageId = gTilePageId[tileIndex];

    if (pageId == 0xFFFFFFFFu) { gAtlas[outIdx] = kAtlasClear; return; }  // unallocated tile

    int mip, px, py;
    UnpackPageId((int)pageId, mipLevels, mip, px, py);
    int lx = (int)(x % pageSize);
    int ly = (int)(y % pageSize);
    gAtlas[outIdx] = PageTexel((int)pageId, mip, lx, ly);
}
