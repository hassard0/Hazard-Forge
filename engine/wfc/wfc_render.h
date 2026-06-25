#pragma once
// Slice WFC-S6 — LIT 3D render bridge (FLOAT, render-only — the money-shot capstone of FLAGSHIP #29 WFC).
// THE ONE FLOAT CROSSING of the whole flagship. wfc.h's S1-S5 stay STRICT INTEGER + self-contained (the
// cheap clang-hash proof); here — and ONLY here — we cross to float to build per-tile render transforms for
// the rasterizer. This is the documented FLOAT visresolve-bar (the PCG6/FO6/PT6 precedent): the GENERATION
// (wfc::Generate) is bit-exact, the final raster/shade is float (cross-vendor ~the engine baseline, NOT held
// to the integer zero-diff bar). The provenance proof: every transform derives from the bit-exact DECIDED
// cells the integer Solve emits.
//
// SEPARATE HEADER ON PURPOSE: wfc.h is self-contained (only <bit>/<cstddef>/<cstdint>/<vector> + net/session.h)
// so its standalone-clang cross-platform proof stays cheap. The render bridge needs math::Mat4/FromTRS/Vec3 +
// fpx::FxToFloat, which would break that — so the bridge lives HERE, NOT in wfc.h (mirrors pcg.h's render
// bridge which is in pcg.h only because pcg.h already pulls in math+fpx; wfc.h does not, so the split). This
// header is NOT standalone-clang (it's the render layer, only used by the showcase) — that's fine.

#include <cstdint>
#include <vector>

#include "wfc/wfc.h"            // S1-S5 WFC core (Grid / Domain / PopCount) — UNMODIFIED, self-contained
#include "math/math.h"          // render bridge: math::Mat4 / Quat / Vec3 / FromTRS / Normalize (float)
#include "sim/fpx.h"            // fpx::FxToFloat (the single fixed-point->float convention; mirrors pcg.h)

namespace hf::wfc {

// --- TileOf: the single set-bit index of a DECIDED cell (-1 if undecided/contradiction) -------------
// A decided cell has PopCount(domain)==1 -> exactly one allowed tile. Returns that tile index; returns -1
// for any cell that is NOT decided (PopCount != 1), so a partial grid renders only its decided cells.
inline int TileOf(Domain d) {
    if (PopCount(d) != 1) return -1;
    for (int t = 0; t < 64; ++t)
        if ((d >> static_cast<uint32_t>(t)) & Domain{1}) return t;
    return -1;  // unreachable when PopCount==1
}

// --- WfcRenderStyle: the FIXED integer tile->relief table (golden-stable, no float in the data) ------
// tileHeight[tileKind] = integer height per tile kind (e.g. for the showcase gradient water/sand/grass/rock
// {0,1,2,4}). tileSize = world spacing between cell centers; heightUnit = world height per integer step.
// FIXED FOREVER so the render is deterministic of (grid, style) alone.
struct WfcRenderStyle {
    float tileSize   = 1.0f;            // world spacing between cell centers
    float heightUnit = 0.25f;           // world height per integer height step
    std::vector<int32_t> tileHeight;    // [tileCount] integer height per tile kind
};

// --- WfcToRenderInstances: the render bridge (the PcgToRenderInstances twin) -------------------------
// For each DECIDED cell (TileOf>=0), emit ONE column-major model matrix placing a unit CUBE tile at the
// cell's CENTERED world position:
//   x = (cellX - (g.w-1)*0.5) * tileSize         (centered so the map straddles the origin)
//   z = (cellZ - (g.h-1)*0.5) * tileSize
//   y = height * heightUnit                       (relief; the cube's top rises with the tile kind)
// scale = tileSize (the unit cube is half-extent 0.5 -> a tileSize-scaled cube exactly tiles the grid).
// The cube is lifted in Y by height*heightUnit so taller tiles (rock) stand above lower ones (water): the
// relief that reads as biome terrain. Identity orientation. The SINGLE float crossing is here (the grid
// cells + the style table are integer; positions/heights become float once via fpx::FxToFloat-style host
// math). Undecided cells (TileOf<0) are skipped. Empty grid -> empty output (the empty no-op). Pure
// deterministic host float (no RNG, no clock). This is the canonical provenance function — the showcase
// memcmp-compares its bytes against a recompute.
inline std::vector<math::Mat4> WfcToRenderInstances(const Grid& g, const WfcRenderStyle& style) {
    std::vector<math::Mat4> out;
    if (g.w <= 0 || g.h <= 0) return out;
    out.reserve(static_cast<std::size_t>(g.w) * static_cast<std::size_t>(g.h));
    const float halfW = (static_cast<float>(g.w) - 1.0f) * 0.5f;
    const float halfH = (static_cast<float>(g.h) - 1.0f) * 0.5f;
    const math::Quat ident = math::Quat::Identity();
    const math::Vec3 scale{style.tileSize, style.tileSize, style.tileSize};
    for (int32_t z = 0; z < g.h; ++z) {
        for (int32_t x = 0; x < g.w; ++x) {
            const Domain d = g.cell[static_cast<std::size_t>(z * g.w + x)];
            const int tile = TileOf(d);
            if (tile < 0) continue;  // undecided cell — skip (a partial grid renders only decided cells)
            const int32_t hgt = (tile < static_cast<int>(style.tileHeight.size()))
                                    ? style.tileHeight[static_cast<std::size_t>(tile)] : 0;
            const math::Vec3 t{(static_cast<float>(x) - halfW) * style.tileSize,
                               static_cast<float>(hgt) * style.heightUnit,
                               (static_cast<float>(z) - halfH) * style.tileSize};
            out.push_back(math::FromTRS(t, ident, scale));
        }
    }
    return out;
}

// --- WfcTileKinds: the parallel per-instance tile-kind list (for per-kind COLORED draws) -------------
// SAME iteration order as WfcToRenderInstances, so tileKind[i] is the tile kind of mats[i]. The showcase
// uses this to issue ONE draw per tile kind with a per-kind albedo (water=blue, sand=tan, grass=green,
// rock=grey) — a COLORED biome money-shot WITHIN "no new shader" (the existing instanced-lit pipeline
// binds a per-draw albedo texture + material). The provenance compare still uses the flat
// WfcToRenderInstances bytes; this is a derived presentation index.
inline std::vector<uint32_t> WfcTileKinds(const Grid& g) {
    std::vector<uint32_t> kinds;
    if (g.w <= 0 || g.h <= 0) return kinds;
    kinds.reserve(static_cast<std::size_t>(g.w) * static_cast<std::size_t>(g.h));
    for (int32_t z = 0; z < g.h; ++z)
        for (int32_t x = 0; x < g.w; ++x) {
            const int tile = TileOf(g.cell[static_cast<std::size_t>(z * g.w + x)]);
            if (tile < 0) continue;
            kinds.push_back(static_cast<uint32_t>(tile));
        }
    return kinds;
}

}  // namespace hf::wfc
