#pragma once
// Slice VT1 — Runtime Virtual Texturing Slice 1: VT page table + page-needed FEEDBACK/MARKING
// (BEACHHEAD of FLAGSHIP #4: UE5's literal Runtime Virtual Texturing). Pure CPU (header-only, no
// device, no backend symbols). Namespace hf::render::vt. The STRUCTURAL TWIN of VSM Slice VA
// (engine/render/vsm.h): a virtual page table -> a feedback/marking pass -> (later) physical tile
// allocation + indirection -> a sampled lookup -> per-page caching. This slice is the marking pass,
// the direct analog of vsm.h's MarkResidentPages, applied to TEXTURE pages instead of shadow pages.
//
// The marking shader shaders/vt_feedback.comp.hlsl copies PageId VERBATIM from this header (and
// consumes the HOST-SNAPPED integer (mip,px,py) page coords this header's VtPageId produces), so
// tests/vt_test.cpp exercises the EXACT math the GPU feedback pass runs — which is what makes the
// feedback page set bit-identical GPU==CPU AND cross-backend.
//
// THE TECHNIQUE (Runtime Virtual Texturing = Nanite-scale texturing): a virtual texture far larger
// than VRAM, sampled through a virtual->physical page indirection driven by per-pixel FEEDBACK. The
// virtual texture is a MIP PYRAMID: mip 0 is virtualPagesPerSideMip0² pages of pageSize texels, each
// finer mip halving the pages-per-side (max 1). This slice builds the virtual PAGE TABLE + MARKS
// which virtual texture pages a set of sample-requests NEED, as a pure INTEGER compute pass (no
// rendering, no new RHI). The marking output is a pure integer SET (feedback[pageId] in {0,1}) ->
// inherently bit-exact + cross-backend, proven GPU==CPU via ReadBuffer memcmp.
//
// THE DETERMINISM CRUX (the make-or-break for GPU==CPU, like vsm.h's threshold-ladder / swraster.h's
// host-snapped ScreenVerts): the UV->page quantization (floor(u*pagesPerSide(mip))) is the one
// float->int boundary. We pin it identically on both sides (plain multiply, NOT fma), so floor(u*pps)
// is bit-identical CPU<->Vulkan<->Metal for the host-supplied exact float32 UVs and the small
// power-of-two pps. For TOTAL cross-backend safety we go further (the swraster.h host-snap pattern):
// the showcase HOST-SNAPS each SampleRequest to its integer (mip,px,py) page coords and uploads THOSE
// to the GPU, so the GPU does ZERO floating point — it just calls PageId on integers, making GPU==CPU
// trivially exact. VtPageId(u,v,mip) stays here as the CPU reference + the unit-test oracle.
//
// The mip-level SELECTION (SelectMipLevel) is the integer THRESHOLD-LADDER copied from vsm.h's
// SelectClipmapLevel: count host-precomputed thresholds a texel-density value exceeds, clamp. NO log2,
// NO transcendental on the bit-exact path. For THIS beachhead the showcase supplies `mip` per request
// directly (the simplest deterministic form); SelectMipLevel is exercised + unit-tested as the general
// path so slice VT4 (the material-pass sample) can use it.
//
// CONVENTIONS:
//   * Flat page-table index: PageId(mip,px,py) = mipPageOffset(mip) + py*pagesPerSide(mip) + px. Unlike
//     VSM (a CONSTANT vpps per clipmap level, so offset = level*vpps²), VT mips have DIFFERENT
//     pagesPerSide, so the per-mip offset is a host PREFIX-SUM of pagesPerSide(k)² for k<mip — NOT
//     mip*vpps². UnpackPageId is its exact inverse — a bijection over [0, pageCount()) (unit-tested).
//   * UV->page: px = clamp(floor(u*pagesPerSide(mip)), 0, pps-1); plain subtract/multiply/floor.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace hf::render::vt {

// Hard cap on mip levels (sizes the on-stack threshold table; the shader uses the same fixed cap).
inline constexpr int kMaxMips = 16;

// The virtual texture page table: a MIP pyramid. Mip 0 is virtualPagesPerSideMip0² pages of pageSize
// texels each; each finer mip halves the pages-per-side (floored at 1). (e.g. mipLevels=4,
// pageSize=128, virtualPagesPerSideMip0=16 -> a 2048²-texel mip-0 virtual texture, pages per side
// 16/8/4/2 over 4 mips, 256+64+16+4 = 340 pages total.)
struct VtTexture {
    int mipLevels = 4;
    int pageSize = 128;
    int virtualPagesPerSideMip0 = 16;

    // Pages per side at `mip`: virtualPagesPerSideMip0 >> mip, floored at 1.
    int pagesPerSide(int mip) const {
        int p = virtualPagesPerSideMip0 >> mip;
        return p < 1 ? 1 : p;
    }

    // The host PREFIX-SUM page offset of `mip`: sum of pagesPerSide(k)² for k < mip. (NOT mip*vpps² —
    // mips have DIFFERENT pagesPerSide. This is the crux that distinguishes VT from VSM's flat layout.)
    int mipPageOffset(int mip) const {
        int off = 0;
        for (int k = 0; k < mip; ++k) {
            int pk = pagesPerSide(k);
            off += pk * pk;
        }
        return off;
    }

    // Total virtual page-table slots across all mips (the feedback SSBO length).
    int pageCount() const { return mipPageOffset(mipLevels); }
};

// Flat page-table index: mip-major (via the prefix-sum offset), then py-major, then px. MIRRORED in
// vt_feedback.comp. idx = mipPageOffset(mip) + py*pagesPerSide(mip) + px.
inline int PageId(int mip, int px, int py, const VtTexture& vt) {
    const int pps = vt.pagesPerSide(mip);
    return vt.mipPageOffset(mip) + py * pps + px;
}

// Inverse of PageId (a bijection over [0, pageCount())). Unit-tested PageId(UnpackPageId(id))==id.
// Walk the mips, subtracting each mip's page span, until `id` lands inside one.
inline void UnpackPageId(int id, const VtTexture& vt, int& mip, int& px, int& py) {
    int rem = id;
    int m = 0;
    for (; m < vt.mipLevels; ++m) {
        const int pps = vt.pagesPerSide(m);
        const int span = pps * pps;
        if (rem < span) break;
        rem -= span;
    }
    if (m >= vt.mipLevels) m = vt.mipLevels - 1;  // clamp a degenerate out-of-range id to the top mip
    const int pps = vt.pagesPerSide(m);
    mip = m;
    py = rem / pps;
    px = rem % pps;
}

// Host-precompute the mip threshold table: thresholds[m] = 2^m (the texel-density doubling per mip).
// The showcase uploads these exact float32 bits; the CPU + the shader read the SAME bits and count how
// many a density value exceeds. 2^m is an exact power-of-two -> bit-identical to the shader's
// (float)(1u<<m). We expose the builder so the showcase + the test agree. (For THIS beachhead the
// showcase supplies mip per request directly; SelectMipLevel is exercised + unit-tested as the general
// path so slice VT4's material-pass sample can use it.)
inline void BuildMipThresholds(const VtTexture& vt, float* thresholds /*[vt.mipLevels]*/) {
    for (int m = 0; m < vt.mipLevels; ++m)
        thresholds[m] = (float)(1u << (uint32_t)m);
}

// DETERMINISM CRUX — select the mip level by an INTEGER threshold-ladder (NO log2). The level is the
// number of thresholds `density` EXCEEDS, clamped to [0, mipLevels-1]. thresholds[m] = 2^m
// (BuildMipThresholds). density <= 1 -> mip 0 (the finest); each doubling of density climbs one mip;
// large -> the coarsest mip. Pure compare/count -> bit-identical CPU<->GPU. (Equivalent to
// clamp(floor(log2(density)), 0, mipLevels-1) but WITHOUT the transcendental.) Copied from
// vsm.h::SelectClipmapLevel.
inline int SelectMipLevel(float density, const VtTexture& vt) {
    float thresholds[kMaxMips];
    const int mips = vt.mipLevels;
    BuildMipThresholds(vt, thresholds);
    int mip = 0;
    for (int m = 0; m < mips; ++m)
        if (density > thresholds[m]) mip = m + 1;
    if (mip < 0) mip = 0;
    if (mip > mips - 1) mip = mips - 1;
    return mip;
}

// A virtual-texture sample-request: a UV in [0,1] at a given mip. (The showcase host-snaps each to its
// integer (mip,px,py) page coords before uploading to the GPU — see SnappedRequest.)
struct SampleRequest {
    float u = 0.0f;
    float v = 0.0f;
    int   mip = 0;
};

// THE UV->PAGE QUANTIZER. px = clamp(floor(u*pagesPerSide(mip)), 0, pps-1); same for py; return
// PageId(mip,px,py). Plain subtract/multiply/floor (NOT fma — pin the identical op vs the shader). The
// `u*pps` float multiply is immediately floored to an integer; with host-supplied exact float32 UVs +
// a small power-of-two pps, floor(u*pps) is bit-identical CPU<->Vulkan<->Metal. This is the CPU
// REFERENCE + the unit-test oracle; the showcase host-snaps via this and uploads the resulting integer
// (mip,px,py) so the GPU does ZERO float (the swraster.h host-snap discipline). MIRRORED in
// vt_feedback.comp (the GPU consumes the snapped integers + calls PageId).
inline int VtPageId(float u, float v, int mip, const VtTexture& vt) {
    const int pps = vt.pagesPerSide(mip);
    int px = (int)std::floor(u * (float)pps);
    int py = (int)std::floor(v * (float)pps);
    if (px < 0) px = 0; else if (px > pps - 1) px = pps - 1;
    if (py < 0) py = 0; else if (py > pps - 1) py = pps - 1;
    return PageId(mip, px, py, vt);
}

// The host-snapped integer page coords for a SampleRequest (what the showcase uploads to the GPU so the
// GPU does ZERO floating point). SnapRequest runs VtPageId's quantization on the host.
struct SnappedRequest {
    int mip = 0;
    int px = 0;
    int py = 0;
};

inline SnappedRequest SnapRequest(const SampleRequest& req, const VtTexture& vt) {
    const int pps = vt.pagesPerSide(req.mip);
    int px = (int)std::floor(req.u * (float)pps);
    int py = (int)std::floor(req.v * (float)pps);
    if (px < 0) px = 0; else if (px > pps - 1) px = pps - 1;
    if (py < 0) py = 0; else if (py > pps - 1) py = pps - 1;
    return SnappedRequest{req.mip, px, py};
}

// CPU REFERENCE marking: zero feedbackOut (sized pageCount()), then for each request VtPageId ->
// feedbackOut[pageId] = 1. This is the EXACT integer set the GPU vt_feedback.comp matches byte-for-byte
// (the set is order-independent — writes race-free to 1). feedbackEnabled=false is modeled by the
// caller passing an empty request span (or skipping this call), yielding the cleared all-zero set.
inline void MarkFeedbackPages(std::span<const SampleRequest> requests, const VtTexture& vt,
                              std::span<uint32_t> feedbackOut) {
    for (uint32_t& f : feedbackOut) f = 0u;
    for (const SampleRequest& r : requests) {
        int pageId = VtPageId(r.u, r.v, r.mip, vt);
        if (pageId >= 0 && pageId < (int)feedbackOut.size())
            feedbackOut[(size_t)pageId] = 1u;
    }
}

// ============================ Slice VT2 — physical tile pool + indirection (pure CPU) ==========
// VT1 produced the resident page SET (feedback[pageId] in {0,1}); VT2 assigns each resident virtual
// page a PHYSICAL tile in a finite tile pool (a deterministic integer virtual->physical indirection
// table the future sampling pass reads). ZERO backend symbols — the same above-seam discipline as VT1;
// the DIRECT analog of VSM Slice VB's AllocatePhysicalPages (vsm.h), applied to TEXTURE pages. The
// allocator's ascending-priority determinism (the make-or-break for GPU==CPU + cross-backend) carries
// over unchanged: ascending pageId == mip-major == finest-mip-first priority (the nearest/most-detailed
// pages win the pool when it overflows). This slice produces ONLY the tile ASSIGNMENT + indirection
// table — no pixels (procedural page CONTENT is VT3).

// Sentinel: a virtual page with NO physical tile (non-resident, OR overflowed past the pool capacity).
// kNoTile is -1 in the int32 CPU/return form; the GPU SSBO uses 0xFFFFFFFF, cast to -1 on read-back.
// (Mirrors VSM's kNoTile (vsm.h), expressed as a signed int32 to match AllocatePhysicalTiles' return.)
inline constexpr int32_t  kNoTile      = -1;
inline constexpr uint32_t kNoTileU32   = 0xFFFFFFFFu;  // the SSBO sentinel (== (uint32_t)kNoTile)

// The physical tile pool: a finite square of tilesPerSide² tiles, each holding one pageSize² page.
// (e.g. tilesPerSide=12 -> 144 physical tiles for VT1's 340 virtual pages — deliberately SMALLER than
// pageCount() so the overflow/kNoTile path is exercised.) Mirrors vsm.h::VsmAtlas.
struct VtTilePool {
    int tilesPerSide = 12;

    int tileCapacity() const { return tilesPerSide * tilesPerSide; }
};

// The physical-atlas pixel origin (top-left) of tile `tileIndex` in the pool. Tiles are row-major:
// ox = (tileIndex % tilesPerSide) * pageSize; oy = (tileIndex / tilesPerSide) * pageSize. Integer,
// exact. Mirrors vsm.h::PhysicalTileOrigin (texture pages, not shadow tiles).
inline void PhysicalTileOrigin(int tileIndex, const VtTilePool& pool, int pageSize, int& ox, int& oy) {
    const int tps = pool.tilesPerSide;
    ox = (tileIndex % tps) * pageSize;
    oy = (tileIndex / tps) * pageSize;
}

// THE DETERMINISTIC ALLOCATOR (the vsm.h::AllocatePhysicalPages analog). Walk pageId 0..feedback.size()-1
// in ASCENDING order; each resident page (feedback[pageId]==1) gets the NEXT sequential physical tile
// index while nextTile < pool.tileCapacity(); non-resident pages — and resident pages that overflow past
// the capacity — get kNoTile. Returns the indirection table (sized feedback.size() == pageCount()).
//
// PRIORITY = ascending pageId. PageId is mip-major (id = mipPageOffset(mip) + py*pps + px), so ascending
// pageId == mip 0 (the finest, highest-detail mip) first, then mip 1, ... — i.e. the FINEST/NEAREST mip
// pages get the lowest tile indices and are allocated FIRST. That is the sane VT priority for an overflow
// cap (keep the near, high-detail pages; drop the far, coarse ones) — the same ascending=finest-first
// priority VSM uses. Allocation is INHERENTLY sequential (nextTile depends on prior assignments), so the
// GPU side mirrors this with a single-thread serial scan (shaders/vt_alloc.comp). Pure integer +
// order-deterministic -> bit-exact two-run + a CPU reference the showcase memcmp's against.
inline std::vector<int32_t> AllocatePhysicalTiles(std::span<const uint32_t> feedback,
                                                  const VtTilePool& pool) {
    std::vector<int32_t> indirection(feedback.size(), kNoTile);
    const int cap = pool.tileCapacity();
    int nextTile = 0;
    for (size_t pageId = 0; pageId < feedback.size(); ++pageId) {
        if (feedback[pageId] != 0u && nextTile < cap) {
            indirection[pageId] = nextTile;
            ++nextTile;
        } else {
            indirection[pageId] = kNoTile;  // non-resident OR overflowed past the cap
        }
    }
    return indirection;
}

// Convenience alias — the indirection table IS the allocation. (Spec's BuildIndirection.)
inline std::vector<int32_t> BuildIndirection(std::span<const uint32_t> feedback, const VtTilePool& pool) {
    return AllocatePhysicalTiles(feedback, pool);
}

// ============================ Slice VT3 — procedural PAGE GENERATION into the atlas (pure CPU) =====
// VT2 produced the virtual->physical INDIRECTION table; VT3 GENERATES each allocated virtual page's
// deterministic procedural CONTENT into its physical tile in the PHYSICAL ATLAS — the analog of VSM
// Slice VB's physical-page depth render, but writing procedural texture COLOR instead of caster depth.
// Kept PURE INTEGER (no float) so the atlas is cross-backend BIT-IDENTICAL: the per-texel color is a
// pure integer function of (pageId, mip, localXY). The physical atlas is a flat array of atlasW*atlasH
// RGBA8-packed uints (NOT a sampled texture yet — that is VT4). ZERO backend symbols (the same above-
// seam discipline as VT1/VT2). PageTexel below is copied VERBATIM into shaders/vt_pagegen.comp.hlsl, so
// tests/vt_test.cpp exercises the EXACT math the GPU pagegen pass runs (the GPU==CPU bit-exact crux).

// The physical-atlas dimensions: a square of tilesPerSide² physical tiles, each pageSize² texels, so the
// atlas is (tilesPerSide*pageSize) on a side. (Derived from VtTilePool + VtTexture.pageSize; held as a
// small struct so the showcase + the shader params share one source of truth.) e.g. tilesPerSide=8,
// pageSize=64 -> a 512×512 atlas = 262144 RGBA8 texels.
struct VtAtlasDims {
    int tilesPerSide = 8;
    int pageSize     = 64;

    int atlasW()      const { return tilesPerSide * pageSize; }
    int atlasH()      const { return tilesPerSide * pageSize; }  // square atlas
    int atlasTexels() const { return atlasW() * atlasH(); }
};

// The cleared-texel sentinel for non-allocated atlas regions: opaque dark grey, RGBA8 packed
// 0xAABBGGRR == (A=0xFF, B=0x10, G=0x10, R=0x10). Chosen DISTINCT from any value PageTexel can emit:
// PageTexel's three color lanes are each forced into [0x40,0xFF] (a generated texel's min channel is
// 0x40 = 64 > 0x10 = 16), so kAtlasClear can never collide with a generated texel. Opaque so the golden
// shows the unused pool as a flat dark field.
inline constexpr uint32_t kAtlasClear = 0xFF101010u;

// THE deterministic per-texel generator, CONTENT-VERSION-AWARE (Slice VT5). PURE INTEGER — no float, so
// bit-identical CPU<->Vulkan<->Metal. Returns an RGBA8-packed uint (0xAABBGGRR, A=0xFF). A per-page BASE
// color from an FNV/avalanche hash of (pageId, mip) — with `contentVersion` folded into the hash seed so a
// bumped version VISIBLY changes the page's color — is MODULATED by an integer pattern over the page-local
// (lx,ly) in [0,pageSize): a coarse 8×8 CHECKERBOARD toggles brightness and a coarse 4-step GRADIENT
// across the page tints it, so ADJACENT pages, ADJACENT mips, ADJACENT texels AND DIFFERENT content
// versions are all visually DISTINCT. Every channel is clamped into [0x40,0xFF] (so no generated texel
// collides with kAtlasClear's 0x10 channels). Pure shifts/xor/add/mul.
//
// VERSION-0 INVARIANCE (the VT3/VT4 golden guard): the version is folded as `contentVersion * <prime>`
// ADDED into the seed, so at contentVersion==0 it contributes EXACTLY 0 and the seed reproduces the
// ORIGINAL VT3 FNV mix byte-for-byte. PageTexel(pageId,mip,lx,ly) == PageTexelV(pageId,mip,lx,ly,0u)
// (unit-tested + guarded by the unchanged VT3 vt_atlas + VT4 vt_sample goldens). COPIED VERBATIM into
// shaders/vt_cachegen.comp.hlsl (the cache-aware sibling of vt_pagegen.comp).
inline uint32_t PageTexelV(int pageId, int mip, int lx, int ly, uint32_t contentVersion) {
    // FNV/avalanche hash of (pageId,mip) + contentVersion. At version 0 the `+ version*prime` term is 0,
    // so the seed == the original VT3 PageTexel seed exactly. The prime (0x27D4EB2F) is large + odd so a
    // bumped version well-separates the seed (a visibly different page color).
    uint32_t h = (uint32_t)pageId * 2654435761u + (uint32_t)(mip + 1) * 0x9E3779B9u
               + contentVersion * 0x27D4EB2Fu;
    h ^= h >> 15; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    uint32_t br = (h)       & 0xFFu;
    uint32_t bg = (h >> 8)  & 0xFFu;
    uint32_t bb = (h >> 16) & 0xFFu;

    // Integer pattern over the page-local texel: an 8×8 checkerboard brightness toggle (full vs 5/8) and
    // a coarse 4-step horizontal+vertical gradient (a small additive tint). All integer.
    uint32_t check = (((uint32_t)lx >> 3) ^ ((uint32_t)ly >> 3)) & 1u;  // 0/1 over 8×8 blocks
    uint32_t grad  = ((uint32_t)lx >> 4) + ((uint32_t)ly >> 4);          // 0..(pageSize/8) coarse steps

    // Apply: checker scales the base (full on a light cell, 5/8 on a dark cell); gradient adds a tint.
    auto chan = [&](uint32_t base) -> uint32_t {
        uint32_t v = check ? base : (base * 5u) >> 3;  // dark checker cells = 5/8 brightness
        v += grad * 6u;                                 // coarse gradient tint
        if (v > 0xFFu) v = 0xFFu;
        if (v < 0x40u) v = 0x40u;                       // floor into [0x40,0xFF] (distinct from kAtlasClear)
        return v;
    };
    uint32_t r = chan(br), g = chan(bg), b = chan(bb);
    return 0xFF000000u | (b << 16) | (g << 8) | r;      // 0xAABBGGRR
}

// THE deterministic per-texel generator (PURE INTEGER — no float, so bit-identical CPU<->Vulkan<->Metal).
// VT3's generator, now expressed as PageTexelV(...,/*contentVersion=*/0) — BYTE-IDENTICAL to the original
// VT3 PageTexel (version 0 reproduces the original FNV mix exactly; the VT3 vt_atlas + VT4 vt_sample
// goldens are unchanged). COPIED VERBATIM into shaders/vt_pagegen.comp.hlsl (which is UNCHANGED).
inline uint32_t PageTexel(int pageId, int mip, int lx, int ly) {
    return PageTexelV(pageId, mip, lx, ly, 0u);
}

// Build the HOST reverse table tilePageId[tileIndex] = pageId (the GPU pagegen input). For each pageId
// with indirection[pageId] != kNoTile, tilePageId[indirection[pageId]] = pageId; unallocated tiles get
// kNoTileU32 (0xFFFFFFFF). Sized tileCapacity(). Since the allocator assigns each allocated page a UNIQUE
// tile, this is a correct inverse of the indirection over the allocated set (unit-tested).
inline std::vector<uint32_t> BuildTilePageId(std::span<const int32_t> indirection, const VtTilePool& pool) {
    std::vector<uint32_t> tilePageId((size_t)pool.tileCapacity(), kNoTileU32);
    for (size_t pageId = 0; pageId < indirection.size(); ++pageId) {
        int32_t tile = indirection[pageId];
        if (tile != kNoTile && tile >= 0 && tile < pool.tileCapacity())
            tilePageId[(size_t)tile] = (uint32_t)pageId;
    }
    return tilePageId;
}

// CPU REFERENCE atlas generation (the EXACT atlas the GPU vt_pagegen.comp produces). Fill atlasOut (sized
// atlasTexels()) with kAtlasClear; for each pageId with indirection[pageId] != kNoTile, find its physical
// tile origin (PhysicalTileOrigin), unpack the pageId -> (mip,px,py), and for each page-local (lx,ly) in
// [0,pageSize)² write atlasOut[(oy+ly)*atlasW + (ox+lx)] = PageTexel(pageId, mip, lx, ly). genEnabled=false
// leaves the atlas all-kAtlasClear (the disabled-path no-op). pageSize == dims.pageSize (the tile/page
// edge); PhysicalTileOrigin uses the SAME pageSize so tile origins land on the atlas grid.
inline void GeneratePhysicalAtlas(std::span<const int32_t> indirection, const VtTexture& vt,
                                  const VtTilePool& pool, const VtAtlasDims& dims,
                                  std::span<uint32_t> atlasOut, bool genEnabled = true) {
    const int atlasW  = dims.atlasW();
    const int pageSize = dims.pageSize;
    for (uint32_t& t : atlasOut) t = kAtlasClear;
    if (!genEnabled) return;
    for (size_t pageId = 0; pageId < indirection.size(); ++pageId) {
        const int32_t tile = indirection[pageId];
        if (tile == kNoTile) continue;
        int ox, oy;
        PhysicalTileOrigin(tile, pool, pageSize, ox, oy);
        int mip, px, py;
        UnpackPageId((int)pageId, vt, mip, px, py);
        for (int ly = 0; ly < pageSize; ++ly)
            for (int lx = 0; lx < pageSize; ++lx) {
                const size_t idx = (size_t)(oy + ly) * (size_t)atlasW + (size_t)(ox + lx);
                if (idx < atlasOut.size())
                    atlasOut[idx] = PageTexel((int)pageId, mip, lx, ly);
            }
    }
}

// ============================ Slice VT4 — material-pass SAMPLE through the INDIRECTION (pure CPU) ====
// VT3 generated each allocated page's CONTENT into the physical atlas; VT4 SAMPLES the virtual texture
// through the virtual->physical INDIRECTION — the round-trip that makes RVT a usable texture: per virtual
// texel (vx,vy), compute its pageId, look up indirection[pageId] -> the physical tile, offset into the
// physical atlas, and read the texel (NEAREST — a direct integer texel read, NO filtering). Kept PURE
// INTEGER so the sample path is cross-backend BIT-IDENTICAL (the zero-differing-pixel posture of VT1-VT3);
// bilinear filtering + perspective-mapped geometry are an explicit DEFERRED float refinement (VT5+).
// ZERO backend symbols (the same above-seam discipline as VT1/VT2/VT3). SampleVirtualTexel below is copied
// VERBATIM into shaders/vt_sample.comp.hlsl, so tests/vt_test.cpp exercises the EXACT math the GPU sample
// pass runs (the GPU==CPU bit-exact crux). The reconstructed image is the virtual texture as the user
// ADDRESSES it (pages in virtual-UV layout) — visually DISTINCT from VT3's physical-atlas (tile-alloc order).

// The sampled value for a virtual texel whose page is NON-RESIDENT (indirection[pageId]==kNoTile): a flat
// MISS color, magenta debug tint RGBA8-packed 0xAABBGGRR == (A=0xFF, B=0xFF, G=0x00, R=0xFF). Chosen
// DISTINCT from both kAtlasClear (0xFF101010, dark grey) and any value PageTexel can emit (PageTexel's
// green lane is forced into [0x40,0xFF] >= 0x40, so a generated texel can never have G==0x00 -> a generated
// texel can never collide with kVtMiss). Reads clearly (bright magenta) against the generated pages.
inline constexpr uint32_t kVtMiss = 0xFFFF00FFu;  // magenta miss tint (B=0xFF,G=0x00,R=0xFF,A=0xFF)

// THE SAMPLE-THROUGH-INDIRECTION round-trip (PURE INTEGER — no float, so bit-identical CPU<->Vulkan<->Metal).
// vx,vy = a virtual texel of the mip being reconstructed. px=vx/pageSize, py=vy/pageSize = the virtual page;
// pageId = PageId(mip,px,py); tile = indirection[pageId]; if tile==kNoTile -> kVtMiss; else PhysicalTileOrigin
// -> (ox,oy), local (lx,ly) = (vx%pageSize, vy%pageSize), return atlas[(oy+ly)*atlasW + (ox+lx)] (NEAREST — a
// direct integer texel read). pageSize comes from atlasDims (the same tile/page edge VT3 generated with). The
// SAME atlas VT3 generated + the SAME indirection VT2 built -> the read-back texel == PageTexel(pageId,mip,lx,ly)
// (a self-consistency the proof asserts). COPIED VERBATIM into shaders/vt_sample.comp.hlsl.
inline uint32_t SampleVirtualTexel(int vx, int vy, int mip, std::span<const int32_t> indirection,
                                   std::span<const uint32_t> atlas, const VtTexture& vt,
                                   const VtTilePool& pool, const VtAtlasDims& atlasDims) {
    const int pageSize = atlasDims.pageSize;
    const int atlasW   = atlasDims.atlasW();
    const int px = vx / pageSize;
    const int py = vy / pageSize;
    const int pageId = PageId(mip, px, py, vt);
    if (pageId < 0 || pageId >= (int)indirection.size()) return kVtMiss;
    const int32_t tile = indirection[(size_t)pageId];
    if (tile == kNoTile) return kVtMiss;
    int ox, oy;
    PhysicalTileOrigin(tile, pool, pageSize, ox, oy);
    const int lx = vx % pageSize;
    const int ly = vy % pageSize;
    const size_t idx = (size_t)(oy + ly) * (size_t)atlasW + (size_t)(ox + lx);
    if (idx >= atlas.size()) return kVtMiss;
    return atlas[idx];
}

// CPU REFERENCE reconstruction (the EXACT virtual image the GPU vt_sample.comp produces). For each virtual
// texel (vx,vy) in [0, pagesPerSide(mip)*pageSize)², imageOut[vy*W + vx] = SampleVirtualTexel(...). The
// reconstructed virtual-texture image at `mip`: resident pages show their VT3 content at their virtual-UV
// location, non-resident pages show kVtMiss. sampleEnabled=false leaves the image all-kVtMiss (the disabled
// no-op). imageOut sized W*W where W = pagesPerSide(mip)*pageSize.
inline void ReconstructVirtualImage(int mip, std::span<const int32_t> indirection,
                                    std::span<const uint32_t> atlas, const VtTexture& vt,
                                    const VtTilePool& pool, const VtAtlasDims& atlasDims,
                                    std::span<uint32_t> imageOut, bool sampleEnabled = true) {
    const int pageSize = atlasDims.pageSize;
    const int W = vt.pagesPerSide(mip) * pageSize;  // virtual image is square at this mip
    if (!sampleEnabled) {
        for (uint32_t& t : imageOut) t = kVtMiss;
        return;
    }
    for (int vy = 0; vy < W; ++vy)
        for (int vx = 0; vx < W; ++vx) {
            const size_t idx = (size_t)vy * (size_t)W + (size_t)vx;
            if (idx < imageOut.size())
                imageOut[idx] = SampleVirtualTexel(vx, vy, mip, indirection, atlas, vt, pool, atlasDims);
        }
}

// ============================ Slice VT5 — per-page CONTENT KEY + CACHE (pure CPU) ===============
// VT3 GENERATES every allocated page's procedural content into its physical tile each frame. But a page's
// content is FULLY determined by (pageId, mip, contentVersion) — PageTexelV is deterministic — so if those
// inputs are unchanged across frames, re-generating the tile reproduces byte-identical texels: wasted work.
// VT5 adds a per-page CONTENT KEY (a deterministic FNV hash of exactly those inputs) + a per-PHYSICAL-TILE
// CACHE; a tile re-generates iff its content key changed (a cache MISS), a HIT keeps the existing (already-
// correct) atlas tile untouched. This is UE5's Runtime-VT key performance property (generate/stream a page
// once, reuse it until its source changes) and the direct analog of VSM Slice VD's PageContentKey/
// VsmPageCache. It is a PURE optimization — proven SAFE because a hit SKIPS work that would produce
// identical bytes, so the cached atlas is BYTE-IDENTICAL to the fully-regenerated atlas (the froxel.h
// "an optimization must be byte-identical to the unoptimized path" SHA-equality discipline). ZERO backend
// symbols — the cache is pure host logic; "skip a tile" = the GPU vt_cachegen.comp's needsGen==0 early-out
// over a PERSISTENT (not re-cleared) atlas SSBO.

// FNV-1a 32-bit — the fixed, integer/bit-exact hash the content key is built from (mirrors vsm.h's FNV
// lineage, in 32-bit to match the uint32 page content keys VT carries). Pure integer mixing -> two
// identical inputs always hash identically, cross-run + cross-backend.
inline constexpr uint32_t kVtFnvOffset = 2166136261u;
inline constexpr uint32_t kVtFnvPrime  = 16777619u;
inline uint32_t VtFnvMix(uint32_t h, uint32_t v) {
    // Hash the 4 bytes of v (little-endian) into the FNV accumulator.
    for (int b = 0; b < 4; ++b) { h ^= (v & 0xFFu); h *= kVtFnvPrime; v >>= 8; }
    return h;
}

// THE CONTENT KEY: a deterministic 32-bit hash of EVERYTHING that determines page `pageId`'s generated tile
// — its pageId, its mip, and its contentVersion (PageTexelV is a pure function of exactly these, so the key
// captures "has this page's content changed"). An avalanche finalizer spreads the bits. Identical inputs ->
// identical key; a bumped contentVersion (or a different page/mip) -> a different key. FNV over integer
// inputs -> bit-exact, transcendental-free, cross-backend.
inline uint32_t PageContentKey(int pageId, int mip, uint32_t contentVersion) {
    uint32_t h = kVtFnvOffset;
    h = VtFnvMix(h, (uint32_t)pageId);
    h = VtFnvMix(h, (uint32_t)mip);
    h = VtFnvMix(h, contentVersion);
    // Avalanche finalizer (the meshlet/PageTexelV mix) so near-identical inputs scatter.
    h ^= h >> 15; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

// The per-PHYSICAL-TILE cache: the last content key + a valid bit, sized pool.tileCapacity(). A fresh cache
// is all-invalid (every tile a miss on first sight). Mirrors vsm.h::VsmPageCache (indexed by physical TILE
// rather than virtual page, since the atlas persistence is per-tile).
struct VtPageCache {
    std::vector<uint32_t> key;     // [tileCapacity()] last content key seen for the page resident in each tile
    std::vector<uint8_t>  valid;   // [tileCapacity()] 1 iff key[tileIndex] holds a populated key

    void Resize(int tileCapacity) {
        key.assign((size_t)tileCapacity, 0u);
        valid.assign((size_t)tileCapacity, (uint8_t)0u);
    }
};

// A cache HIT for physical tile `tileIndex` at `newKey` == the tile has a valid cached key AND it equals
// newKey (same content -> skip the re-generation, the existing atlas tile is already correct -> byte-
// identical). Mirrors vsm.h::PageCacheHit.
inline bool PageCacheHit(const VtPageCache& cache, int tileIndex, uint32_t newKey) {
    if (tileIndex < 0 || (size_t)tileIndex >= cache.valid.size()) return false;
    return cache.valid[(size_t)tileIndex] != 0u && cache.key[(size_t)tileIndex] == newKey;
}

// Record `newKey` as physical tile `tileIndex`'s current content (called after a (re-)generation populates
// its atlas region). Mirrors vsm.h::PageCacheUpdate.
inline void PageCacheUpdate(VtPageCache& cache, int tileIndex, uint32_t newKey) {
    if (tileIndex < 0 || (size_t)tileIndex >= cache.valid.size()) return;
    cache.key[(size_t)tileIndex]   = newKey;
    cache.valid[(size_t)tileIndex] = (uint8_t)1u;
}

// CPU REFERENCE cached atlas generation (the EXACT atlas the GPU vt_cachegen.comp produces given the host-
// built needsGen flags + per-tile versions). For each allocated physical tile: if needsGen[tile]==0 the
// tile's existing atlasInOut texels are LEFT UNTOUCHED (the cache hit / persistence path); else its region
// is (re-)generated with PageTexelV(pageId,mip,lx,ly,tileVersion[tile]). Unallocated tiles are left
// untouched too (the caller pre-clears the atlas to kAtlasClear on the very first populate). atlasInOut is
// sized atlasTexels() and PERSISTS across calls (NOT re-cleared here) — mirroring the GPU SSBO.
inline void GenerateCachedAtlas(std::span<const uint32_t> tilePageId,
                                std::span<const uint8_t>  needsGen,
                                std::span<const uint32_t> tileVersion,
                                const VtTexture& vt, const VtTilePool& pool, const VtAtlasDims& dims,
                                std::span<uint32_t> atlasInOut) {
    const int atlasW  = dims.atlasW();
    const int pageSize = dims.pageSize;
    const int cap = pool.tileCapacity();
    for (int tile = 0; tile < cap; ++tile) {
        if ((size_t)tile >= tilePageId.size()) break;
        const uint32_t pageId = tilePageId[(size_t)tile];
        if (pageId == kNoTileU32) continue;                  // unallocated tile -> untouched
        if ((size_t)tile >= needsGen.size() || needsGen[(size_t)tile] == 0u) continue;  // cached -> untouched
        int ox, oy;
        PhysicalTileOrigin(tile, pool, pageSize, ox, oy);
        int mip, px, py;
        UnpackPageId((int)pageId, vt, mip, px, py);
        const uint32_t ver = ((size_t)tile < tileVersion.size()) ? tileVersion[(size_t)tile] : 0u;
        for (int ly = 0; ly < pageSize; ++ly)
            for (int lx = 0; lx < pageSize; ++lx) {
                const size_t idx = (size_t)(oy + ly) * (size_t)atlasW + (size_t)(ox + lx);
                if (idx < atlasInOut.size())
                    atlasInOut[idx] = PageTexelV((int)pageId, mip, lx, ly, ver);
            }
    }
}

}  // namespace hf::render::vt
