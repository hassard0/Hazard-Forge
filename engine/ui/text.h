#pragma once
// Slice BA — Text / HUD renderer (Phase 4 #5): a deterministic procedural bitmap-font layout +
// atlas builder.
//
// PURE CPU: this module has ZERO RHI / graphics-backend symbols (no vk*/MTL*/mtl::/Backend::Metal).
// It bakes a fixed 8x8 monospace bitmap font (kFont8x8) in code — NO external asset, NO clock, NO
// RNG — rasterizes it into a single RGBA atlas (BuildFontAtlas), and lays a string out into a batch
// of screen-space textured quads in NDC (LayoutText). Same table + same layout math => same pixels =>
// goldens match on every backend. It is compiled into BOTH hf_core (ASan-scoped, unit-tested) and
// hf_engine (the live --hud-shot / --game-hud-shot showcases + the Metal --hud / --game-hud path).
//
// The draw side reuses the EXISTING alpha-blend + sampled-texture + screen-space-overlay paths
// (shaders/text.vert.hlsl + text.frag.hlsl, an alphaBlend+cullNone pipeline): the atlas is a normal
// sampled texture and the quads are drawn over the scene after post, alpha-blended.
#include <cstdint>
#include <string>
#include <vector>

namespace hf::ui {

// --- Baked 8x8 monospace bitmap font ------------------------------------------------------------
// One uint8_t[8] per printable-ASCII glyph (0x20..0x7E => 95 glyphs). Row r is `kFont8x8[g][r]`;
// bit (1 << (7 - col)) set => that pixel is INK. A public-domain 8x8 font table; unknown / out-of-
// range chars render as a blank cell. Defined in text.cpp.
constexpr int kGlyphFirst = 0x20;  // ' '
constexpr int kGlyphLast  = 0x7E;  // '~'
constexpr int kGlyphCount = kGlyphLast - kGlyphFirst + 1;  // 95
constexpr int kGlyphPx    = 8;     // glyph cell is 8x8 pixels

extern const uint8_t kFont8x8[kGlyphCount][kGlyphPx];

// --- Atlas geometry -----------------------------------------------------------------------------
// The 95 glyphs are laid out in a 16-column x 6-row grid of 8x8 cells => 128x48 atlas. Glyph index
// g (= ch - kGlyphFirst) sits at cell (g % 16, g / 16).
constexpr int kAtlasCols = 16;
constexpr int kAtlasRows = 6;
constexpr int kAtlasW    = kAtlasCols * kGlyphPx;  // 128
constexpr int kAtlasH    = kAtlasRows * kGlyphPx;  // 48

// Rasterize the baked font into `rgbaOut` (must be atlasW*atlasH*4 bytes). Every texel is white
// (255,255,255) with alpha = coverage: 255 where the glyph bit is INK, 0 elsewhere. So the fragment
// shader can read the alpha as coverage and tint with a configurable text color, and the transparent
// background blends to nothing. Deterministic: identical bytes every call. atlasW/atlasH must be
// kAtlasW/kAtlasH (the function asserts the size by ignoring out-of-range writes).
void BuildFontAtlas(uint8_t* rgbaOut, int atlasW, int atlasH);

// --- Text layout --------------------------------------------------------------------------------
// One vertex of a laid-out text quad. `posPx` holds the NDC position (already converted from pixels
// via screenW/screenH inside LayoutText — the field name reflects that the INPUT origin/scale are in
// pixels). `uv` indexes the atlas cell for the glyph. text.vert.hlsl passes both straight through.
struct TextVertex {
    float posPx[2];  // NDC x,y in [-1,1] (Y down, matching the UI/clip conventions)
    float uv[2];     // atlas UV in [0,1]
};

// Lay `s` out starting at pixel (originX, originY) (top-left of the first glyph), each glyph scaled
// to kGlyphPx*pxScale pixels, into a screenW x screenH target. Appends 6 vertices (two triangles)
// per VISIBLE glyph to `outVerts` (NDC positions + atlas UVs) and returns the QUAD count. Spaces and
// newlines advance / wrap the cursor but emit NO quad; '\n' returns the cursor to originX and steps
// down one line (kGlyphPx*pxScale). Chars outside [kGlyphFirst,kGlyphLast] are treated as blank
// (advance, no quad). Pure math; deterministic. `outVerts` is appended to (not cleared).
int LayoutText(const std::string& s, float originX, float originY, float pxScale,
               int screenW, int screenH, std::vector<TextVertex>& outVerts);

}  // namespace hf::ui
