// Unit test for the DETERMINISTIC SDF TEXT pipeline (engine/ui/sdf_text.h, Slice UF1). Pure CPU
// (hf_core), ASan-eligible. Asserts:
//   (a) SDF: a stroke centerline is INSIDE (negative), far points are OUTSIDE (positive), the zero-
//       contour sits near the stroke edge; the full glyph SDF digest is pinned + two-run stable.
//   (b) SCALE: one SDF sampled/rendered at 1x AND 2x is crisp at BOTH — the partial-coverage edge
//       band is bounded (~const width in texels), unlike a 2x-scaled bitmap; coverage digests pinned.
//   (c) LAYOUT: pen positions are pinned exact (PROPORTIONAL advances); a kerned pair ("AV") pulls
//       closer than unkerned by the pinned kern amount; a width wrap breaks at the pinned glyph.
//   (d) ATLAS: the shelf-packed atlas rects are pinned with NO overlaps.
//   (e) the whole-scenario digest is pinned + two-run identical (cross-compiler currency).
#include "ui/sdf_text.h"

#include <cstdio>
#include <string>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::ui;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // --- (a) SDF sign + zero contour + digest ---------------------------------------------------
    {
        const int gi = sdf::GlyphIndex('H');
        check(gi >= 0, "'H' is in the glyph subset");
        const std::vector<sdf::FlatSeg> segs = sdf::FlattenGlyph(sdf::GlyphSet()[(size_t)gi]);
        // 'H' left stroke centerline is at em x=14, mid height y=32 -> exactly on a stroke => sd == -radius.
        const int32_t sdCenter = sdf::SignedDistWork(segs, 14 * sdf::kSub, 32 * sdf::kSub);
        check(sdCenter == -sdf::kStrokeW, "stroke centerline SDF == -strokeRadius (fully inside)");
        // A far corner is outside (positive).
        const int32_t sdFar = sdf::SignedDistWork(segs, 2 * sdf::kSub, 2 * sdf::kSub);
        check(sdFar > 0, "far point SDF is positive (outside)");
        // A point one stroke-radius off the left stem (em x=17==14+3) at y=20 (below the crossbar, so
        // the nearest edge is the stem) is ~ the zero contour.
        const int32_t sdEdge = sdf::SignedDistWork(segs, 17 * sdf::kSub, 20 * sdf::kSub);
        check(sdEdge > -sdf::kSub && sdEdge < sdf::kSub, "stroke edge SDF is ~zero (the 0-contour)");

        const sdf::GlyphSDF gH = sdf::GenerateGlyphSDF('H', 48);
        const sdf::GlyphSDF gH2 = sdf::GenerateGlyphSDF('H', 48);
        const uint64_t dH = sdf::DigestGlyphSDF(gH);
        check(dH == sdf::DigestGlyphSDF(gH2), "GlyphSDF is deterministic (two builds identical)");
        std::printf("UF1-PIN sdfH=%016llx\n", (unsigned long long)dH);
        check(dH == 0xbe30f5e3ec6189cbull, "'H' SDF digest pinned");  // PIN-A
    }

    // --- (b) SCALE: crisp at 1x AND 2x ----------------------------------------------------------
    auto maxPartialRun = [](const std::vector<uint8_t>& cov, int w, int row) {
        int best = 0, run = 0;
        for (int x = 0; x < w; ++x) {
            const uint8_t a = cov[(size_t)row * w + x];
            if (a > 0 && a < 255) { ++run; if (run > best) best = run; }
            else run = 0;
        }
        return best;
    };
    {
        const sdf::GlyphSDF gH = sdf::GenerateGlyphSDF('H', 32);   // ONE field...
        const std::vector<uint8_t> cov1 = sdf::RenderCoverage(gH, 32, 32);   // ...sampled at 1x
        const std::vector<uint8_t> cov2 = sdf::RenderCoverage(gH, 64, 64);   // ...and 2x
        // Rows through the two vertical stems (above the crossbar) => clean vertical edges.
        const int r1 = maxPartialRun(cov1, 32, 8);
        const int r2 = maxPartialRun(cov2, 64, 16);
        std::printf("UF1-PIN edgeRun1=%d edgeRun2=%d\n", r1, r2);
        // The SDF win: a smooth AA band EXISTS at both scales (not the hard 0/255-only blocks a 2x-scaled
        // 8x8 bitmap gives), AND the band stays a bounded few texels wide (crisp) — its em-space width is
        // ~scale-invariant, so text is equally sharp at any size.
        check(r1 >= 1 && r1 <= 4, "1x edge band is a bounded few texels (crisp, AA present)");
        check(r2 >= 1 && r2 <= 4, "2x edge band is a bounded few texels (still crisp, no blockiness)");
        const int bandEm1 = r1 * sdf::kEm / 32, bandEm2 = r2 * sdf::kEm / 64;
        check(bandEm1 - bandEm2 <= 6 && bandEm2 - bandEm1 <= 6, "AA band ~scale-invariant in em (SDF win)");
        uint64_t d1 = sdf::detail::kFnvSeed, d2 = sdf::detail::kFnvSeed;
        for (uint8_t a : cov1) d1 = sdf::detail::Fnv(d1, a);
        for (uint8_t a : cov2) d2 = sdf::detail::Fnv(d2, a);
        std::printf("UF1-PIN cov1=%016llx cov2=%016llx\n", (unsigned long long)d1, (unsigned long long)d2);
        check(d1 == 0x5b42c4dfe308f7c5ull, "1x coverage digest pinned");  // PIN-B1
        check(d2 == 0x5117a1945bfa8641ull, "2x coverage digest pinned");  // PIN-B2
    }

    // --- (c) LAYOUT: proportional pens, kerning, wrap --------------------------------------------
    {
        // size == kEm so advancePx == advance(em) exactly (clean pins). H adv 56, I adv 28.
        const sdf::LayoutResult r = sdf::LayoutText("HI", sdf::kEm, /*track=*/0, /*maxW=*/0, /*kern=*/false);
        check(r.glyphs.size() == 2, "\"HI\" -> 2 glyphs");
        check(r.glyphs[0].penX == 0 && r.glyphs[0].advancePx == 56, "H at penX 0, advance 56 (proportional)");
        check(r.glyphs[1].penX == 56 && r.glyphs[1].advancePx == 28, "I advanced by H's 56, own advance 28");

        // Kerning: "AV" pulls V closer by |KernEm(A,V)| px (size==kEm => 1 em == 1 px).
        const sdf::LayoutResult noK = sdf::LayoutText("AV", sdf::kEm, 0, 0, /*kern=*/false);
        const sdf::LayoutResult yesK = sdf::LayoutText("AV", sdf::kEm, 0, 0, /*kern=*/true);
        const int delta = noK.glyphs[1].penX - yesK.glyphs[1].penX;
        std::printf("UF1-PIN kernAV=%d\n", delta);
        check(delta == 10, "kerned 'AV' pulls V 10px closer than unkerned (pinned kern)");

        // Empty string -> nothing.
        const sdf::LayoutResult e = sdf::LayoutText("", sdf::kEm, 0, 0, true);
        check(e.glyphs.empty() && e.lines == 1, "empty string lays out to nothing");

        // Single glyph advance == pinned metric.
        const sdf::LayoutResult m = sdf::LayoutText("M", sdf::kEm, 0, 0, false);
        check(m.glyphs.size() == 1 && m.glyphs[0].advancePx == 64, "'M' advance == 64 (pinned wide metric)");

        // Line break: "HHHH" at size 64 (adv 56) wraps the 3rd H (penX 112 + 56 = 168 > 140).
        const sdf::LayoutResult w = sdf::LayoutText("HHHH", sdf::kEm, 0, /*maxW=*/140, /*kern=*/false);
        check(w.lines == 2, "width wrap produced 2 lines");
        check(w.glyphs[2].penX == 0 && w.glyphs[2].penY == 64 * 5 / 4, "wrap breaks at the 3rd glyph (pinned)");
    }

    // --- (d) ATLAS: shelf pack rects pinned, no overlaps ----------------------------------------
    {
        const sdf::Atlas a = sdf::BuildAtlas(24, /*maxWidth=*/8 * 24);
        check(a.rects.size() == 40, "atlas packs all 40 subset glyphs");
        check(a.w == 192 && a.h == 120, "atlas is the pinned 192x120 (8 cols x 5 shelves of 24)");
        check(a.rects[0].x == 0 && a.rects[0].y == 0, "glyph 0 at (0,0)");
        check(a.rects[7].x == 168 && a.rects[7].y == 0, "glyph 7 at the row-7 slot (168,0)");
        check(a.rects[8].x == 0 && a.rects[8].y == 24, "glyph 8 wraps to the 2nd shelf (0,24)");
        // No two rects overlap.
        bool overlap = false;
        for (size_t i = 0; i < a.rects.size() && !overlap; ++i)
            for (size_t j = i + 1; j < a.rects.size(); ++j) {
                const sdf::AtlasRect& p = a.rects[i];
                const sdf::AtlasRect& q = a.rects[j];
                const bool sep = p.x + p.w <= q.x || q.x + q.w <= p.x ||
                                 p.y + p.h <= q.y || q.y + q.h <= p.y;
                if (!sep) { overlap = true; break; }
            }
        check(!overlap, "no atlas rects overlap");
        const uint64_t dA = sdf::DigestAtlas(a);
        std::printf("UF1-PIN atlas=%016llx\n", (unsigned long long)dA);
        check(dA == 0xd7457a447dddac3bull, "atlas layout digest pinned");  // PIN-D
    }

    // --- (e) whole-scenario digest pinned + two-run identical ------------------------------------
    {
        const sdf::Uf1Run r1 = sdf::RunUf1TextScenario();
        const sdf::Uf1Run r2 = sdf::RunUf1TextScenario();
        check(r1.digest == r2.digest, "scenario is deterministic (two runs identical)");
        check(r1.glyphCount == 40 && r1.stringLen == (int)(r1.s1.size() + r1.s2.size()),
              "scenario stats consistent");
        std::printf("UF1-PIN scenario=%016llx atlasW=%d atlasH=%d\n",
                    (unsigned long long)r1.digest, r1.atlas.w, r1.atlas.h);
        check(r1.digest == 0xbaca98f42661ce00ull, "scenario digest pinned");  // PIN-E
    }

    if (g_fail == 0) { std::printf("sdf_text_test OK\n"); return 0; }
    std::printf("sdf_text_test: %d failures\n", g_fail);
    return 1;
}
