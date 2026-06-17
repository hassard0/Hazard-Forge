#pragma once
// Slice VA — Virtual Shadow Maps Slice 1: Page Table + Page-Needed Marking (BEACHHEAD). Pure CPU
// (header-only, no device, no backend symbols). Namespace hf::render::vsm. Same flat-SSBO +
// mirrored-CPU-header + no-op-flag TEMPLATE as froxel.h / cluster.h: a small shared-math header ABOVE
// the RHI seam (ZERO vk*/MTL*/mtl::/Backend::Metal CODE symbols — the only mentions of "vk"/"MTL"
// anywhere in this slice's above-seam files are seam-discipline doc comments). The marking shader
// shaders/vsm_mark.comp.hlsl copies SelectClipmapLevel + MarkPage VERBATIM from this header, so
// tests/vsm_test.cpp exercises the EXACT math the GPU marking pass runs — which is what makes the
// resident-page set bit-identical GPU==CPU AND cross-backend.
//
// THE TECHNIQUE (VSM = Nanite-scale shadowing): a directional light's shadow has a HUGE virtual
// resolution, sparsely backed by physical pages. The virtual texture is a directional CLIPMAP — a
// stack of `levels` levels, each `virtualPagesPerSide²` pages, where level L covers
// `level0WorldExtent * 2^L` world units centered on `cameraPos` (level 0 the small central region,
// each level 2x larger). This slice builds the virtual PAGE TABLE + MARKS which virtual pages the
// visible receivers need, as a pure INTEGER compute pass (no rendering, no new RHI). The marking
// output is a pure integer SET (resident[pageId] in {0,1}) -> inherently bit-exact + cross-backend,
// proven GPU==CPU via ReadBuffer memcmp.
//
// THE DETERMINISM CRUX (the make-or-break for GPU==CPU, like cluster.h's squared-distance / visbuffer.h's
// flat-integer discipline): the clipmap LEVEL must be selected bit-identically CPU<->Vulkan<->Metal.
// A raw log2 is a TRANSCENDENTAL — it is NOT guaranteed bit-identical across CPU and GPU vendors. So we
// AVOID it entirely: SelectClipmapLevel selects the level by an INTEGER comparison ladder against
// host-precomputed thresholds level0WorldExtent*2^L (a small `levels`-entry table; the showcase uploads
// it as exact float32 bits and BOTH the CPU + the shader read the SAME bits). The level is the NUMBER of
// thresholds distToCamera exceeds, clamped to [0, levels-1]. Pure compare/count -> no transcendental on
// the bit-exact path -> integer-stable cross-backend. The XY projection is plain subtract/divide/floor
// (also transcendental-free). The output is a pure integer set -> bit-exact GPU==CPU + two-run identical.
//
// CONVENTIONS:
//   * Flat page-table index: PageId(level,px,py) = level*(vpps*vpps) + py*vpps + px (the cluster.h:57 /
//     froxel.h:67 flat-index discipline). UnpackPageId is its exact inverse — a bijection over
//     [0, pageCount()) (unit-tested like PackVisId/UnpackVisId).
//   * The level's clipmap ortho is a top-down axis-aligned XY map: origin = cameraPos snapped to the
//     level's page grid, extent = level0WorldExtent * 2^level over vpps pages. A world point projects to
//     (px,py) = floor((worldXZ - origin) / pageWorldSize). (Directional clipmaps map the ground XZ
//     plane; we use world X for px and world Z for py — the top-down axes.)

#include "math/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace hf::render::vsm {

// Hard cap on clipmap levels (sizes the on-stack threshold table; the shader uses the same fixed cap).
inline constexpr int kMaxLevels = 16;

// The virtual clipmap page table: a stack of `levels` levels, each virtualPagesPerSide² pages of
// pageSize texels. Level L covers level0WorldExtent*2^L world units, centered on cameraPos. (e.g.
// levels=4, vpps=8, pageSize=128, level0WorldExtent=16.)
struct VsmClipmap {
    int       levels = 4;
    int       pageSize = 128;
    int       virtualPagesPerSide = 8;
    float     level0WorldExtent = 16.0f;
    math::Vec3 cameraPos{0.0f, 0.0f, 0.0f};

    // Total virtual page-table slots across all levels.
    int pageCount() const { return levels * virtualPagesPerSide * virtualPagesPerSide; }
};

// Flat page-table index: level-major, then py-major, then px (the cluster.h flat-index discipline).
// idx = level*(vpps*vpps) + py*vpps + px. MIRRORED in vsm_mark.comp.
inline int PageId(int level, int px, int py, const VsmClipmap& cm) {
    const int vpps = cm.virtualPagesPerSide;
    return level * (vpps * vpps) + py * vpps + px;
}

// Inverse of PageId (a bijection over [0, pageCount())). Unit-tested PageId(UnpackPageId(id))==id.
inline void UnpackPageId(int id, const VsmClipmap& cm, int& level, int& px, int& py) {
    const int vpps = cm.virtualPagesPerSide;
    const int perLevel = vpps * vpps;
    level = id / perLevel;
    int rem = id % perLevel;
    py = rem / vpps;
    px = rem % vpps;
}

// Host-precompute the level threshold table: thresholds[L] = level0WorldExtent * 2^L for L in
// [0, levels). The showcase uploads these exact float32 bits; the CPU + the shader read the SAME bits
// and count how many distToCamera exceeds. NO 2^L on the bit-exact compare path is a transcendental —
// it's an exact power-of-two scale of an exact float32 input, so it's bit-identical to the shader's
// `level0WorldExtent * (float)(1u << L)`. We expose the builder so the showcase + the test agree.
inline void BuildLevelThresholds(const VsmClipmap& cm, float* thresholds /*[cm.levels]*/) {
    for (int L = 0; L < cm.levels; ++L)
        thresholds[L] = cm.level0WorldExtent * (float)(1u << (uint32_t)L);
}

// DETERMINISM CRUX — select the clipmap level by an INTEGER threshold-ladder (NO log2). The level is the
// number of thresholds distToCamera EXCEEDS, clamped to [0, levels-1]. thresholds[L] = level0WorldExtent
// * 2^L (BuildLevelThresholds). distToCamera <= thresholds[0] -> level 0 (the central region); each
// doubling of distance climbs one level; far -> the top level. Pure compare/count -> bit-identical
// CPU<->GPU. MIRRORED VERBATIM in vsm_mark.comp.
//
// (Equivalent to clamp(floor(log2(distToCamera/level0WorldExtent))+1, 0, levels-1) but WITHOUT the
// transcendental — the threshold compare is the bit-exact form.)
inline int SelectClipmapLevel(float distToCamera, const VsmClipmap& cm) {
    float thresholds[kMaxLevels];
    const int levels = cm.levels;
    BuildLevelThresholds(cm, thresholds);
    int level = 0;
    for (int L = 0; L < levels; ++L)
        if (distToCamera > thresholds[L]) level = L + 1;
    if (level < 0) level = 0;
    if (level > levels - 1) level = levels - 1;
    return level;
}

// Mark the virtual page a world receiver point needs: pick its clipmap level via SelectClipmapLevel(
// length(worldPos - cameraPos)), project worldPos into that level's top-down clipmap ortho (origin =
// cameraPos snapped to the level's page grid; extent = level0WorldExtent*2^level over vpps pages) ->
// (px,py) in [0,vpps); outPageId = PageId(level,px,py). Returns true iff (px,py) lands in range (it
// always does for a level the threshold-ladder selected, but a degenerate far point at the very edge is
// clamped). The XY projection uses world X for px and world Z for py (the directional clipmap's top-down
// ground-plane axes) — plain subtract/divide/floor, integer-stable. MIRRORED VERBATIM in vsm_mark.comp.
inline bool MarkPage(const math::Vec3& worldPos, const VsmClipmap& cm, int& outPageId) {
    const int vpps = cm.virtualPagesPerSide;
    math::Vec3 d = worldPos - cm.cameraPos;
    float dist = math::length(d);
    int level = SelectClipmapLevel(dist, cm);

    // The level's clipmap covers `levelExtent` world units over `vpps` pages, so each page is
    // pageWorldSize world units. levelExtent = level0WorldExtent * 2^level (exact float32 power-of-two
    // scale, transcendental-free; identical to the shader's level0WorldExtent * (float)(1u<<level)).
    float levelExtent = cm.level0WorldExtent * (float)(1u << (uint32_t)level);
    float pageWorldSize = levelExtent / (float)vpps;

    // Origin = cameraPos snapped DOWN to the level's page grid, so the clipmap is page-aligned (a
    // standard clipmap snap that keeps page boundaries stable as the camera moves). The clipmap spans
    // [origin, origin + levelExtent) centered near cameraPos: we center it by snapping (cameraPos -
    // levelExtent/2) to the page grid. Snap via floor(v/pageWorldSize)*pageWorldSize — floor is exact.
    float originX = std::floor((cm.cameraPos.x - levelExtent * 0.5f) / pageWorldSize) * pageWorldSize;
    float originZ = std::floor((cm.cameraPos.z - levelExtent * 0.5f) / pageWorldSize) * pageWorldSize;

    int px = (int)std::floor((worldPos.x - originX) / pageWorldSize);
    int py = (int)std::floor((worldPos.z - originZ) / pageWorldSize);

    // Clamp to the page grid (a far receiver at the very edge of the top level lands on the boundary
    // page; the snap centers the clipmap so in-extent points map inside [0,vpps)).
    bool inRange = (px >= 0 && px < vpps && py >= 0 && py < vpps);
    if (px < 0) px = 0; else if (px > vpps - 1) px = vpps - 1;
    if (py < 0) py = 0; else if (py > vpps - 1) py = vpps - 1;

    outPageId = PageId(level, px, py, cm);
    return inRange;
}

// CPU REFERENCE marking: zero residentOut (sized pageCount()), then for each receiver sample MarkPage ->
// residentOut[pageId] = 1. This is the EXACT integer set the GPU vsm_mark.comp matches byte-for-byte
// (the set is order-independent — writes race-free to 1). markingEnabled=false is modeled by the caller
// passing an empty receiver span (or skipping this call), yielding the cleared all-zero set.
inline void MarkResidentPages(std::span<const math::Vec3> receiverSamples, const VsmClipmap& cm,
                              std::span<uint32_t> residentOut) {
    for (uint32_t& r : residentOut) r = 0u;
    for (const math::Vec3& s : receiverSamples) {
        int pageId = 0;
        MarkPage(s, cm, pageId);   // clamped to a valid page; always in [0, pageCount())
        if (pageId >= 0 && pageId < (int)residentOut.size())
            residentOut[(size_t)pageId] = 1u;
    }
}

// ============================ Slice VB — physical-page atlas (pure CPU) =======================
// VA produced the resident SET (resident[pageId] in {0,1}); VB assigns each resident virtual page a
// PHYSICAL atlas tile (a deterministic integer virtual->physical indirection table) and renders that
// page's casters' depth into the tile. ZERO backend symbols — the same above-seam discipline as VA;
// the showcase drives the existing CSM shadow-atlas RHI (CreateShadowMap/SetViewport/BeginShadowPass)
// from this header's integer allocation + per-page ortho.

// Sentinel: a virtual page with NO physical tile (non-resident, or overflowed past the atlas cap).
inline constexpr uint32_t kNoTile = 0xFFFFFFFFu;

// The physical atlas: a CreateShadowMap(tilesPerSide*tileSize) square carved into tilesPerSide² tiles
// of tileSize texels each (e.g. tilesPerSide=16, tileSize=128 -> a 2048² atlas of 256 tiles).
struct VsmAtlas {
    int tilesPerSide = 16;
    int tileSize     = 128;

    int tileCount()  const { return tilesPerSide * tilesPerSide; }
    int atlasSize()  const { return tilesPerSide * tileSize; }
};

// The pixel origin (top-left) of physical tile `tileIndex` in the atlas. Tiles are laid out row-major:
// col = tileIndex % tilesPerSide, row = tileIndex / tilesPerSide; origin = (col*tileSize, row*tileSize)
// — the same row-major tile layout the CSM atlas uses (col*kTile, row*kTile). Integer, exact.
inline void PhysicalTileOrigin(uint32_t tileIndex, const VsmAtlas& atlas, int& outX, int& outY) {
    const int tps = atlas.tilesPerSide;
    int col = (int)tileIndex % tps;
    int row = (int)tileIndex / tps;
    outX = col * atlas.tileSize;
    outY = row * atlas.tileSize;
}

// THE DETERMINISTIC ALLOCATOR. Walk the resident[] set in ASCENDING pageId order; each resident page
// gets the NEXT sequential physical tile index; write indirectionOut[pageId] = tileIndex. Non-resident
// pages (and resident pages that overflow past the atlas's tileCount() cap) get kNoTile. Returns the
// number of tiles allocated (== min(residentCount, tileCount())).
//
// PRIORITY = ascending pageId. PageId is level-major (id = level*vpps² + py*vpps + px), so ascending
// pageId == level 0 first, then level 1, ... — i.e. the FINEST/NEAREST clipmap level (level 0, the
// small central region) gets the lowest tile indices and is allocated FIRST. That is the sane priority
// for an overflow cap (keep the near, high-detail pages; drop the far, coarse ones), so we keep the
// natural ascending-pageId walk — no separate level-then-pageId reorder needed (it would be identical
// for this level-major PageId layout). Pure integer + order-deterministic -> bit-exact two-run + a CPU
// reference the showcase memcmp's against.
inline int AllocatePhysicalPages(std::span<const uint32_t> resident, const VsmAtlas& atlas,
                                 std::span<uint32_t> indirectionOut) {
    const int cap = atlas.tileCount();
    uint32_t next = 0u;
    const size_t n = std::min(resident.size(), indirectionOut.size());
    for (size_t pageId = 0; pageId < indirectionOut.size(); ++pageId) {
        if (pageId < n && resident[pageId] != 0u && (int)next < cap) {
            indirectionOut[pageId] = next;
            ++next;
        } else {
            indirectionOut[pageId] = kNoTile;   // non-resident OR overflowed past the cap
        }
    }
    return (int)next;
}

// The page's world sub-region + its light-space ortho extent. Level L's clipmap covers
// level0WorldExtent*2^L world units over `vpps` pages, so each page is pageWorldSize = levelExtent/vpps
// world units square. The page (px,py)'s world CENTER lies on the level's snapped grid (the SAME snap
// MarkPage uses: origin = floor((cameraPos - levelExtent/2)/pageWorldSize)*pageWorldSize), so the
// rendered depth tile aligns with the marking. The returned halfExtent is the page's half-size (the
// ortho half-width for rendering that page's depth). center is on the y=cameraPos.y ground plane (the
// directional clipmap maps the X/Z ground; X->px, Z->py). DH discipline: std::fma for the grid math.
struct PageOrtho {
    math::Vec3 center;      // world center of the page (on the clipmap's ground plane)
    float      halfExtent;  // half the page's world size (ortho half-width == half-height)
};

inline PageOrtho PageWorldOrtho(int pageId, const VsmClipmap& cm) {
    int level, px, py;
    UnpackPageId(pageId, cm, level, px, py);
    const int vpps = cm.virtualPagesPerSide;

    // levelExtent = level0WorldExtent * 2^level (exact float32 power-of-two scale, transcendental-free —
    // identical bits to MarkPage's levelExtent). pageWorldSize = levelExtent / vpps.
    float levelExtent   = cm.level0WorldExtent * (float)(1u << (uint32_t)level);
    float pageWorldSize = levelExtent / (float)vpps;

    // The level's snapped origin (matches MarkPage's snap exactly).
    float originX = std::floor((cm.cameraPos.x - levelExtent * 0.5f) / pageWorldSize) * pageWorldSize;
    float originZ = std::floor((cm.cameraPos.z - levelExtent * 0.5f) / pageWorldSize) * pageWorldSize;

    // Page (px,py)'s world center = origin + (page + 0.5) * pageWorldSize. fma for the offset (DH).
    PageOrtho out;
    out.center.x = std::fma((float)px + 0.5f, pageWorldSize, originX);
    out.center.y = cm.cameraPos.y;
    out.center.z = std::fma((float)py + 0.5f, pageWorldSize, originZ);
    out.halfExtent = pageWorldSize * 0.5f;
    return out;
}

// ============================ Slice VD — per-page CONTENT KEY + CACHE (pure CPU) ===============
// VB renders EVERY resident page's caster depth into its tile each frame. But a page's rendered depth
// is fully determined by (its clipmap level + its world region + the casters that overlap that region);
// if NONE of those changed, re-rendering the page reproduces byte-identical tile depth — wasted work.
// VD adds a per-page CONTENT KEY (a deterministic hash of exactly those inputs) + a per-page CACHE; a
// page re-renders iff its key changed (a cache MISS), a HIT keeps the existing (already-correct) tile
// depth untouched. This is UE5's VSM key performance property (cache static pages, re-render only the
// invalidated ones). It is a PURE optimization — proven SAFE because a hit SKIPS work that would produce
// identical bytes, so the cached atlas is BYTE-IDENTICAL to the fully-re-rendered atlas (the same
// froxel.h density=0 "an optimization must be byte-identical to the unoptimized path" SHA-equality
// discipline). ZERO backend symbols — the cache is pure host logic; "skip a page render" = the showcase
// not issuing that page's SetViewport+draw.

// A minimal POD reference to a shadow CASTER, the unit a page's content key hashes over. The page's
// rendered depth depends on each overlapping caster's TRANSFORM (we hash its model matrix, QUANTIZED to
// an integer grid so a sub-ULP wobble doesn't spuriously invalidate while a real move does) + its mesh
// identity (meshId; different geometry -> different silhouette) + its world bounding sphere
// (boundsCenter/Radius), which is what PageOverlapsCaster tests against the page's world region. The
// showcase builds one CasterRef per scene caster from its model matrix + mesh + bounds.
struct CasterRef {
    uint32_t   meshId = 0u;            // mesh identity (distinct geometry -> distinct shadow silhouette)
    uint64_t   xformHash = 0ull;       // a deterministic hash of the QUANTIZED model matrix (see MakeCasterRef)
    math::Vec3 boundsCenter{0,0,0};    // world-space bounding-sphere center (overlap test vs the page region)
    float      boundsRadius = 0.0f;    // world-space bounding-sphere radius
};

// FNV-1a 64-bit — the fixed, integer/bit-exact hash the content key is built from (the same documented
// fixed-hash discipline the rest of the engine uses for deterministic keys). Pure integer mixing -> two
// identical inputs always hash identically, cross-run + cross-backend.
inline constexpr uint64_t kFnvOffset = 1469598103934665603ull;
inline constexpr uint64_t kFnvPrime  = 1099511628211ull;
inline uint64_t FnvMix(uint64_t h, uint64_t v) {
    // Hash the 8 bytes of v (little-endian) into the FNV accumulator.
    for (int b = 0; b < 8; ++b) { h ^= (v & 0xFFull); h *= kFnvPrime; v >>= 8; }
    return h;
}
// Quantize a float to a fixed integer grid (1/1024 world unit) then FNV-mix its INTEGER bits, so the key
// is a pure-integer function of the quantized transform (transcendental-free, bit-exact CPU<->any
// backend) AND a sub-quantum wobble does not change the key while a real move (>= ~1mm) does.
inline uint64_t FnvMixQuantF(uint64_t h, float f) {
    // round-to-nearest on a 1/1024 grid; the result is an exact integer -> bit-stable.
    int64_t q = (int64_t)std::floor((double)f * 1024.0 + 0.5);
    return FnvMix(h, (uint64_t)q);
}

// Build a CasterRef from a model matrix (16 floats, column-major hf::math::Mat4), mesh id, and world
// bounds. The xformHash FNV-folds the QUANTIZED 16 matrix elements -> a real transform change flips it,
// a sub-quantum wobble does not. Deterministic + integer -> identical scene yields identical refs.
inline CasterRef MakeCasterRef(uint32_t meshId, const float* model16,
                               const math::Vec3& boundsCenter, float boundsRadius) {
    CasterRef c;
    c.meshId = meshId;
    uint64_t h = kFnvOffset;
    h = FnvMix(h, (uint64_t)meshId);
    for (int k = 0; k < 16; ++k) h = FnvMixQuantF(h, model16[k]);
    c.xformHash = h;
    c.boundsCenter = boundsCenter;
    c.boundsRadius = boundsRadius;
    return c;
}

// Does caster `c`'s world bounding sphere overlap page `pageId`'s world region? The page's region is its
// PageWorldOrtho square (center +/- halfExtent in world X/Z, the directional clipmap's ground plane). We
// test the caster's bounding sphere (projected to the X/Z plane) against that square, expanded by the
// caster radius (a standard sphere-vs-AABB on the ground plane; Y is the clipmap's projection axis so the
// page covers the full column — only X/Z gate overlap). Pure compare/abs -> integer-stable, deterministic.
inline bool PageOverlapsCaster(int pageId, const VsmClipmap& cm, const CasterRef& c) {
    PageOrtho po = PageWorldOrtho(pageId, cm);
    float dx = std::abs(c.boundsCenter.x - po.center.x) - po.halfExtent;
    float dz = std::abs(c.boundsCenter.z - po.center.z) - po.halfExtent;
    if (dx < 0.0f) dx = 0.0f;
    if (dz < 0.0f) dz = 0.0f;
    // Nearest-point distance² from the sphere center to the page square (on X/Z) vs radius².
    return (dx * dx + dz * dz) <= (c.boundsRadius * c.boundsRadius);
}

// THE CONTENT KEY: a deterministic 64-bit hash of EVERYTHING that determines page `pageId`'s rendered
// depth tile — its clipmap level + its world region (PageWorldOrtho center/halfExtent, QUANTIZED) + the
// set of casters whose bounds OVERLAP that region (each overlapping caster's meshId + quantized xformHash
// folded in ASCENDING caster index order, so the key is order-deterministic over the fixed caster span).
// Two identical scenes -> identical keys for every page; a caster move changes the key ONLY for the
// pages its old/new bounds overlap. FNV over integer/quantized inputs -> bit-exact, transcendental-free,
// cross-backend. (A page with no overlapping casters still gets a well-defined key from its level+region,
// so an empty page is cached too.)
inline uint64_t PageContentKey(int pageId, const VsmClipmap& cm, std::span<const CasterRef> casters) {
    int level, px, py;
    UnpackPageId(pageId, cm, level, px, py);
    PageOrtho po = PageWorldOrtho(pageId, cm);

    uint64_t h = kFnvOffset;
    h = FnvMix(h, (uint64_t)(uint32_t)level);
    h = FnvMix(h, (uint64_t)(uint32_t)px);
    h = FnvMix(h, (uint64_t)(uint32_t)py);
    h = FnvMixQuantF(h, po.center.x);
    h = FnvMixQuantF(h, po.center.y);
    h = FnvMixQuantF(h, po.center.z);
    h = FnvMixQuantF(h, po.halfExtent);
    // The overlapping casters, in ascending index order (deterministic).
    for (size_t i = 0; i < casters.size(); ++i) {
        const CasterRef& c = casters[i];
        if (!PageOverlapsCaster(pageId, cm, c)) continue;
        h = FnvMix(h, (uint64_t)c.meshId);
        h = FnvMix(h, c.xformHash);
    }
    return h;
}

// The per-virtual-page cache: the last content key + a valid bit, sized clipmap.pageCount(). A fresh
// cache is all-invalid (every page a miss on first sight).
struct VsmPageCache {
    std::vector<uint64_t> key;     // [pageCount()] last content key seen for each page
    std::vector<uint8_t>  valid;   // [pageCount()] 1 iff key[pageId] holds a populated key

    void Resize(int pageCount) {
        key.assign((size_t)pageCount, 0ull);
        valid.assign((size_t)pageCount, (uint8_t)0u);
    }
};

// A cache HIT for `pageId` at `newKey` == the page has a valid cached key AND it equals newKey (same
// content -> skip the re-render, the existing tile depth is already correct -> byte-identical).
inline bool PageCacheHit(int pageId, uint64_t newKey, const VsmPageCache& cache) {
    if (pageId < 0 || (size_t)pageId >= cache.valid.size()) return false;
    return cache.valid[(size_t)pageId] != 0u && cache.key[(size_t)pageId] == newKey;
}

// Record `newKey` as page `pageId`'s current content (called after a (re-)render populates its tile).
inline void PageCacheUpdate(int pageId, uint64_t newKey, VsmPageCache& cache) {
    if (pageId < 0 || (size_t)pageId >= cache.valid.size()) return;
    cache.key[(size_t)pageId]   = newKey;
    cache.valid[(size_t)pageId] = (uint8_t)1u;
}

}  // namespace hf::render::vsm
