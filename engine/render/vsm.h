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
#include <cstdint>
#include <span>

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

}  // namespace hf::render::vsm
