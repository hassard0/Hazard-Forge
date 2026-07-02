// Slice SC1b (ALPHA-MASK CUTOUTS + TEXTURE MIPMAPS) — the PURE deterministic RGBA8 mip chain,
// unit-tested. hf::asset::BuildRgba8MipChain is the CPU seam the glTF loader uses to feed
// rhi::TextureDesc::mipData on BOTH backends (unlike a GPU blit, whose filtering is vendor-defined,
// the chain is pure int arithmetic -> bit-identical everywhere). This pins:
//   * FullMipCount for representative sizes (1x1, POT, NPOT, extreme aspect),
//   * the exact chain bytes for small hand-computed images (integer-rounded 4-sample averages),
//   * odd-dimension edge-clamp behaviour (each dst texel still averages exactly 4 samples),
//   * chain shape (level count + per-level sizes) for a POT texture,
//   * two-call determinism (byte-identical chains).
// Pure CPU (hf_core), no RHI/device, ASan-eligible.

#include "asset/gltf_loader.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::asset;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else       { std::printf("PASS: %s\n", what); }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- FullMipCount pins -----------------------------------------------------------------------
    check(FullMipCount(1, 1) == 1, "mipcount: 1x1 -> 1");
    check(FullMipCount(2, 2) == 2, "mipcount: 2x2 -> 2");
    check(FullMipCount(4, 4) == 3, "mipcount: 4x4 -> 3");
    check(FullMipCount(1024, 1024) == 11, "mipcount: 1024x1024 -> 11 (the Sponza map size)");
    check(FullMipCount(1280, 720) == 11, "mipcount: 1280x720 NPOT -> 11 (floor(log2(1280))+1)");
    check(FullMipCount(3, 3) == 2, "mipcount: 3x3 -> 2");
    check(FullMipCount(8, 1) == 4, "mipcount: 8x1 extreme aspect -> 4");
    check(FullMipCount(0, 0) == 1, "mipcount: degenerate 0x0 -> 1 (treated as 1x1)");

    // ---- 2x2 -> 1x1: one level, the exact integer-rounded average --------------------------------
    {
        // Texels: (10,20,30,40) (20,40,60,80) / (30,60,90,120) (40,80,120,160).
        // Sum per channel: 100,200,300,400 -> (sum+2)>>2 = 25,50,75,100 (exact, no rounding needed).
        const uint8_t px[16] = {10, 20, 30, 40,  20, 40, 60, 80,
                                30, 60, 90, 120, 40, 80, 120, 160};
        auto chain = BuildRgba8MipChain(px, 2, 2);
        check(chain.size() == 1, "2x2: chain has exactly 1 level");
        const uint8_t expect[4] = {25, 50, 75, 100};
        check(chain.size() == 1 && chain[0].size() == 4 &&
                  std::memcmp(chain[0].data(), expect, 4) == 0,
              "2x2: mip1 is the exact per-channel average");
    }

    // ---- Rounding pin: (a+b+c+d+2)>>2 rounds HALF UP ---------------------------------------------
    {
        // All four alpha=255; RGB = 0,1,1,1 -> sum 3 -> (3+2)>>2 = 1 (0.75 rounds to 1).
        const uint8_t px[16] = {0, 0, 0, 255,  1, 1, 1, 255,
                                1, 1, 1, 255,  1, 1, 1, 255};
        auto chain = BuildRgba8MipChain(px, 2, 2);
        const uint8_t expect[4] = {1, 1, 1, 255};
        check(chain.size() == 1 && chain[0].size() == 4 &&
                  std::memcmp(chain[0].data(), expect, 4) == 0,
              "rounding: 0.75 -> 1 ((sum+2)>>2, deterministic half-up)");
    }

    // ---- 3x3 odd dims: 2x2 source block clamps to the edge (still exactly 4 samples) -------------
    {
        // R channel = 10*(x + 3*y), G=B=0, A=255. 3x3 -> 1x1: samples (0,0)=(0),(1,0)=(10),
        // (0,1)=(30),(1,1)=(40) -> R=(0+10+30+40+2)>>2 = 20.
        uint8_t px[36] = {0};
        for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 3; ++x) {
                px[(y * 3 + x) * 4 + 0] = (uint8_t)(10 * (x + 3 * y));
                px[(y * 3 + x) * 4 + 3] = 255;
            }
        auto chain = BuildRgba8MipChain(px, 3, 3);
        check(chain.size() == 1, "3x3: chain has exactly 1 level (3>>1 = 1x1)");
        const uint8_t expect[4] = {20, 0, 0, 255};
        check(chain.size() == 1 && chain[0].size() == 4 &&
                  std::memcmp(chain[0].data(), expect, 4) == 0,
              "3x3: dst averages the top-left 2x2 block (odd edge never sampled past bounds)");
    }

    // ---- 1xN strip: the width-1 axis clamps (duplicated edge sample), height halves --------------
    {
        // 1x4 column, R = 0,40,80,120 (A=255). Level1 = 1x2: rows avg (0,0,40,40)->20 and
        // (80,80,120,120)->100. Level2 = 1x1: (20,20,100,100)->60.
        const uint8_t px[16] = {0, 0, 0, 255,  40, 0, 0, 255,
                                80, 0, 0, 255, 120, 0, 0, 255};
        auto chain = BuildRgba8MipChain(px, 1, 4);
        check(chain.size() == 2, "1x4: chain has 2 levels (1x2, 1x1)");
        const uint8_t e1[8] = {20, 0, 0, 255, 100, 0, 0, 255};
        const uint8_t e2[4] = {60, 0, 0, 255};
        check(chain.size() == 2 && chain[0].size() == 8 &&
                  std::memcmp(chain[0].data(), e1, 8) == 0,
              "1x4: level 1 rows are the clamped-x averages");
        check(chain.size() == 2 && chain[1].size() == 4 &&
                  std::memcmp(chain[1].data(), e2, 4) == 0,
              "1x4: level 2 collapses level 1");
    }

    // ---- Chain shape for a POT texture + two-call determinism ------------------------------------
    {
        // 8x4 deterministic pseudo-pattern.
        std::vector<uint8_t> px(8 * 4 * 4);
        for (size_t i = 0; i < px.size(); ++i) px[i] = (uint8_t)((i * 37 + 11) & 0xFF);
        auto a = BuildRgba8MipChain(px.data(), 8, 4);
        auto b = BuildRgba8MipChain(px.data(), 8, 4);
        check(a.size() == 3, "8x4: chain has 3 levels (4x2, 2x1, 1x1)");
        check(a.size() == 3 && a[0].size() == 4u * 2u * 4u && a[1].size() == 2u * 1u * 4u &&
                  a[2].size() == 1u * 1u * 4u,
              "8x4: per-level byte sizes are max(1,w>>i) x max(1,h>>i) x 4");
        bool same = a.size() == b.size();
        for (size_t i = 0; same && i < a.size(); ++i)
            same = a[i].size() == b[i].size() &&
                   std::memcmp(a[i].data(), b[i].data(), a[i].size()) == 0;
        check(same, "8x4: two calls produce byte-identical chains (determinism)");
    }

    // ---- Degenerate inputs ------------------------------------------------------------------------
    {
        const uint8_t px[4] = {1, 2, 3, 4};
        check(BuildRgba8MipChain(px, 1, 1).empty(), "1x1: no chain");
        check(BuildRgba8MipChain(nullptr, 4, 4).empty(), "null pixels: no chain (no crash)");
        check(BuildRgba8MipChain(px, 0, 4).empty(), "zero width: no chain (no crash)");
    }

    if (g_fail == 0) std::printf("gltf_mipchain_test: ALL PASS\n");
    else             std::printf("gltf_mipchain_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
