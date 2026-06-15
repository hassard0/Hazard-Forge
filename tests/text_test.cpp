// Unit test for the text / HUD layout + atlas (engine/ui/text.{h,cpp}, Slice BA). Pure CPU (hf_core),
// ASan-eligible like the other pure tests — NO GPU. Asserts:
//   * BuildFontAtlas is deterministic (two builds are byte-identical) + has the documented shape
//     (128x48 RGBA, white with coverage alpha, transparent background, at least one ink texel).
//   * LayoutText emits exactly 6 verts per VISIBLE glyph + the right quad count for a known string
//     including spaces and a newline (spaces + '\n' advance/wrap but emit no quad).
//   * the cursor advance matches kGlyphPx*pxScale (the second glyph starts one cell to the right).
//   * UVs map to the correct atlas cell for a sample char.
//   * the NDC conversion is correct at a known screen coord / scale.
//   * the empty string yields 0 quads / 0 verts.
//   * chars outside the printable range are treated as blank (no quad).
#include "ui/text.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b) { return std::fabs(a - b) < 1e-4f; }

int main() {
    // --- Atlas determinism + shape --------------------------------------------------------------
    std::vector<uint8_t> a(static_cast<size_t>(ui::kAtlasW) * ui::kAtlasH * 4, 0xAB);
    std::vector<uint8_t> b(static_cast<size_t>(ui::kAtlasW) * ui::kAtlasH * 4, 0xCD);
    ui::BuildFontAtlas(a.data(), ui::kAtlasW, ui::kAtlasH);
    ui::BuildFontAtlas(b.data(), ui::kAtlasW, ui::kAtlasH);
    check(a == b, "BuildFontAtlas is deterministic (byte-identical across runs)");
    check(ui::kAtlasW == 128 && ui::kAtlasH == 48, "atlas is the documented 128x48");

    // Every texel must be white with alpha in {0,255}; at least one ink texel and one transparent.
    bool sawInk = false, sawClear = false, rgbAlwaysWhite = true, alphaBinary = true;
    for (size_t i = 0; i < a.size(); i += 4) {
        if (a[i] != 255 || a[i + 1] != 255 || a[i + 2] != 255) rgbAlwaysWhite = false;
        if (a[i + 3] == 255) sawInk = true;
        else if (a[i + 3] == 0) sawClear = true;
        else alphaBinary = false;
    }
    check(rgbAlwaysWhite, "atlas RGB is always white (coverage lives in alpha)");
    check(alphaBinary, "atlas alpha is binary coverage (0 or 255)");
    check(sawInk, "atlas has at least one ink texel");
    check(sawClear, "atlas has at least one transparent texel");

    // The 'A' glyph (index 'A'-0x20 == 33) must have SOME ink in its 8x8 cell (legibility guard).
    {
        int g = 'A' - ui::kGlyphFirst;
        int cx = (g % ui::kAtlasCols) * ui::kGlyphPx;
        int cy = (g / ui::kAtlasCols) * ui::kGlyphPx;
        int ink = 0;
        for (int r = 0; r < ui::kGlyphPx; ++r)
            for (int c = 0; c < ui::kGlyphPx; ++c) {
                size_t idx = (static_cast<size_t>(cy + r) * ui::kAtlasW + (cx + c)) * 4 + 3;
                if (a[idx] == 255) ++ink;
            }
        check(ink >= 6, "the 'A' glyph cell has ink (font is legible)");
    }

    // --- Empty string => no geometry ------------------------------------------------------------
    {
        std::vector<ui::TextVertex> v;
        int q = ui::LayoutText("", 10.0f, 10.0f, 2.0f, 1280, 720, v);
        check(q == 0 && v.empty(), "empty string yields 0 quads / 0 verts");
    }

    // --- 6 verts per visible glyph + quad count with spaces + newline ---------------------------
    {
        // "AB CD\nE": 5 visible glyphs (A B C D E), one space, one newline.
        std::vector<ui::TextVertex> v;
        int q = ui::LayoutText("AB CD\nE", 0.0f, 0.0f, 1.0f, 1280, 720, v);
        check(q == 5, "AB CD\\nE => 5 quads (space + newline emit none)");
        check(v.size() == static_cast<size_t>(q) * 6, "6 verts per visible glyph");
    }

    // --- Cursor advance == kGlyphPx*pxScale; the second glyph is one cell to the right -----------
    {
        const float originX = 100.0f, originY = 50.0f, scale = 3.0f;
        const int SW = 1280, SH = 720;
        std::vector<ui::TextVertex> v;
        ui::LayoutText("AA", originX, originY, scale, SW, SH, v);
        check(v.size() == 12, "two glyphs => 12 verts");
        // First vertex of each quad is the top-left corner (px = origin + n*cell).
        // Convert the expected pixel x of glyph 0 and glyph 1 top-left to NDC and compare.
        auto ndcX = [&](float px) { return px / static_cast<float>(SW) * 2.0f - 1.0f; };
        float cell = static_cast<float>(ui::kGlyphPx) * scale;  // 24 px
        float x0 = v[0].posPx[0];
        float x1 = v[6].posPx[0];
        check(approx(x0, ndcX(originX)), "glyph 0 left edge at originX (NDC)");
        check(approx(x1, ndcX(originX + cell)), "glyph 1 advanced by one cell (kGlyphPx*pxScale)");
    }

    // --- NDC conversion correct at a known screen coord -----------------------------------------
    {
        const int SW = 200, SH = 100;
        const float scale = 1.0f;  // 8x8 px cell
        std::vector<ui::TextVertex> v;
        ui::LayoutText("A", 0.0f, 0.0f, scale, SW, SH, v);
        check(v.size() == 6, "one glyph => 6 verts");
        // Top-left pixel (0,0) -> NDC (-1,-1); bottom-right pixel (8,8) -> NDC computed from SW/SH.
        auto ndcX = [&](float px) { return px / static_cast<float>(SW) * 2.0f - 1.0f; };
        auto ndcY = [&](float px) { return px / static_cast<float>(SH) * 2.0f - 1.0f; };
        // Find min/max corners across the 6 verts (two tris over the same quad).
        float minx = v[0].posPx[0], maxx = v[0].posPx[0];
        float miny = v[0].posPx[1], maxy = v[0].posPx[1];
        for (const auto& tv : v) {
            minx = std::min(minx, tv.posPx[0]); maxx = std::max(maxx, tv.posPx[0]);
            miny = std::min(miny, tv.posPx[1]); maxy = std::max(maxy, tv.posPx[1]);
        }
        check(approx(minx, ndcX(0.0f)) && approx(maxx, ndcX(8.0f)), "quad x spans pixels [0,8] in NDC");
        check(approx(miny, ndcY(0.0f)) && approx(maxy, ndcY(8.0f)), "quad y spans pixels [0,8] in NDC");
    }

    // --- UVs map to the correct atlas cell for a sample char ------------------------------------
    {
        std::vector<ui::TextVertex> v;
        ui::LayoutText("A", 0.0f, 0.0f, 1.0f, 1280, 720, v);
        int g = 'A' - ui::kGlyphFirst;
        float u0 = static_cast<float>((g % ui::kAtlasCols) * ui::kGlyphPx) / static_cast<float>(ui::kAtlasW);
        float v0 = static_cast<float>((g / ui::kAtlasCols) * ui::kGlyphPx) / static_cast<float>(ui::kAtlasH);
        float u1 = static_cast<float>((g % ui::kAtlasCols) * ui::kGlyphPx + ui::kGlyphPx) / static_cast<float>(ui::kAtlasW);
        float v1 = static_cast<float>((g / ui::kAtlasCols) * ui::kGlyphPx + ui::kGlyphPx) / static_cast<float>(ui::kAtlasH);
        float minu = v[0].uv[0], maxu = v[0].uv[0], minv = v[0].uv[1], maxv = v[0].uv[1];
        for (const auto& tv : v) {
            minu = std::min(minu, tv.uv[0]); maxu = std::max(maxu, tv.uv[0]);
            minv = std::min(minv, tv.uv[1]); maxv = std::max(maxv, tv.uv[1]);
        }
        check(approx(minu, u0) && approx(maxu, u1), "'A' UVs span its atlas column cell");
        check(approx(minv, v0) && approx(maxv, v1), "'A' UVs span its atlas row cell");
    }

    // --- Out-of-range char is blank (no quad) ---------------------------------------------------
    {
        std::vector<ui::TextVertex> v;
        std::string s;
        s.push_back('A');
        s.push_back(static_cast<char>(0x01));  // control char, below printable range
        s.push_back('B');
        int q = ui::LayoutText(s, 0.0f, 0.0f, 1.0f, 1280, 720, v);
        check(q == 2, "control char emits no quad (only A + B)");
    }

    if (g_fail == 0) { std::printf("text_test OK\n"); return 0; }
    std::printf("text_test: %d failures\n", g_fail);
    return 1;
}
