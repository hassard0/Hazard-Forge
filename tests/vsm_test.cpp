// Slice VA — Virtual Shadow Maps Slice 1: Page Table + Page-Needed Marking. Pure CPU (header-only, no
// device, no backend symbols). Mirrors engine/render/vsm.h, the SAME clipmap math + page marking the
// --vsm-mark-shot (Vulkan) / --vsm-mark (Metal) showcases + shaders/vsm_mark.comp.hlsl run. Namespace
// hf::render::vsm.
//
// What this test PINS (the contracts the GPU vsm_mark.comp + the GPU==CPU proof build on):
//   * PageId / UnpackPageId round-trip over the FULL [0, pageCount()) range (a bijection).
//   * PageId is INJECTIVE over (level,px,py) — no two distinct cells alias the same flat slot.
//   * SelectClipmapLevel: a receiver AT the camera -> level 0; a far receiver -> the top level; the 2^L
//     boundary distances select the expected level (the threshold-ladder edges, the DETERMINISM CRUX).
//   * MarkPage: a single known point -> exactly 1 resident page at the expected (level,px,py).
//   * MarkResidentPages over a known point-set -> the expected resident set (count + membership).
//   * markingEnabled-off semantics (empty receiver span) -> the cleared all-zero set.
//   * DETERMINISM: two MarkResidentPages passes bit-identical.
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/vsm.h"
#include "math/math.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace vsm = hf::render::vsm;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();
    using math::Vec3;

    // The fixed showcase clipmap: levels=4, vpps=8, pageSize=128, level0WorldExtent=16, camera at origin.
    vsm::VsmClipmap cm;
    cm.levels = 4;
    cm.pageSize = 128;
    cm.virtualPagesPerSide = 8;
    cm.level0WorldExtent = 16.0f;
    cm.cameraPos = {0.0f, 0.0f, 0.0f};

    const int vpps = cm.virtualPagesPerSide;
    const int pageCount = cm.pageCount();
    check(pageCount == 4 * 8 * 8, "pageCount() == levels*vpps*vpps");

    // ================= PageId / UnpackPageId bijection over [0, pageCount()) =================
    {
        bool roundTripOk = true;
        std::vector<uint8_t> seen((size_t)pageCount, 0);
        bool injective = true;
        for (int level = 0; level < cm.levels; ++level)
            for (int py = 0; py < vpps; ++py)
                for (int px = 0; px < vpps; ++px) {
                    int id = vsm::PageId(level, px, py, cm);
                    if (id < 0 || id >= pageCount) { injective = false; continue; }
                    if (seen[(size_t)id]) injective = false;
                    seen[(size_t)id] = 1;
                    int ul, upx, upy;
                    vsm::UnpackPageId(id, cm, ul, upx, upy);
                    if (ul != level || upx != px || upy != py) roundTripOk = false;
                }
        check(roundTripOk, "PageId/UnpackPageId round-trip over (level,px,py)");
        check(injective, "PageId injective (covers [0,pageCount()) exactly once — a bijection)");

        // The inverse over EVERY flat id is in range + re-packs to the same id.
        bool inverseOk = true;
        for (int id = 0; id < pageCount; ++id) {
            int l, px, py; vsm::UnpackPageId(id, cm, l, px, py);
            if (l < 0 || l >= cm.levels || px < 0 || px >= vpps || py < 0 || py >= vpps) inverseOk = false;
            if (vsm::PageId(l, px, py, cm) != id) inverseOk = false;
        }
        check(inverseOk, "UnpackPageId(id) in-range + PageId(UnpackPageId(id))==id for all id");
    }

    // ================= SelectClipmapLevel: threshold-ladder edges (the DETERMINISM CRUX) =============
    {
        // thresholds[L] = level0WorldExtent*2^L = {16, 32, 64, 128}. level = #thresholds dist exceeds.
        float thresholds[vsm::kMaxLevels];
        vsm::BuildLevelThresholds(cm, thresholds);
        check(thresholds[0] == 16.0f && thresholds[1] == 32.0f &&
              thresholds[2] == 64.0f && thresholds[3] == 128.0f, "level thresholds = level0*2^L");

        // A receiver AT the camera -> dist 0 -> level 0.
        check(vsm::SelectClipmapLevel(0.0f, cm) == 0, "dist 0 (at camera) -> level 0");
        // Inside level 0's extent -> level 0; just past 16 -> level 1; etc.
        check(vsm::SelectClipmapLevel(8.0f, cm) == 0, "dist 8 -> level 0");
        check(vsm::SelectClipmapLevel(16.0f, cm) == 0, "dist == threshold[0] (16) -> level 0 (not >)");
        check(vsm::SelectClipmapLevel(16.0001f, cm) == 1, "dist just past 16 -> level 1");
        check(vsm::SelectClipmapLevel(32.0001f, cm) == 2, "dist just past 32 -> level 2");
        check(vsm::SelectClipmapLevel(64.0001f, cm) == 3, "dist just past 64 -> level 3 (top)");
        // A far receiver -> the top level, clamped (no overflow past levels-1).
        check(vsm::SelectClipmapLevel(128.0f, cm) == 3, "dist 128 -> level 3 (top, clamped)");
        check(vsm::SelectClipmapLevel(100000.0f, cm) == 3, "very far -> top level (clamped, no overflow)");
    }

    // ================= MarkPage: a single known point -> exactly the expected (level,px,py) ===========
    {
        // A receiver AT the camera: level 0, levelExtent=16, pageWorldSize=2, origin=floor((0-8)/2)*2=-8.
        // px = floor((0-(-8))/2) = 4, py = 4. PageId(0,4,4) = 0*64 + 4*8 + 4 = 36.
        int id = -1;
        bool inRange = vsm::MarkPage(Vec3{0.0f, 0.0f, 0.0f}, cm, id);
        check(inRange, "MarkPage(at camera) lands in range");
        int l, px, py; vsm::UnpackPageId(id, cm, l, px, py);
        check(l == 0 && px == 4 && py == 4, "MarkPage(at camera) -> level0 center page (4,4)");
        check(id == 36, "MarkPage(at camera) -> PageId 36");

        // A point one page (+2 world X) right of center within level 0 -> px 5, same py.
        int id2 = -1; vsm::MarkPage(Vec3{2.0f, 0.0f, 0.0f}, cm, id2);
        int l2, px2, py2; vsm::UnpackPageId(id2, cm, l2, px2, py2);
        check(l2 == 0 && px2 == 5 && py2 == 4, "MarkPage(+2 X) -> level0 page (5,4)");

        // A point along +Z -> advances py (the top-down Z axis).
        int id3 = -1; vsm::MarkPage(Vec3{0.0f, 0.0f, 2.0f}, cm, id3);
        int l3, px3, py3; vsm::UnpackPageId(id3, cm, l3, px3, py3);
        check(l3 == 0 && px3 == 4 && py3 == 5, "MarkPage(+2 Z) -> level0 page (4,5)");

        // A far point (dist > 64) selects level 3; verify it lands on a valid level-3 page.
        int idFar = -1; vsm::MarkPage(Vec3{90.0f, 0.0f, 0.0f}, cm, idFar);
        int lf, pxf, pyf; vsm::UnpackPageId(idFar, cm, lf, pxf, pyf);
        check(lf == 3, "MarkPage(far, dist 90) -> level 3");
        check(pxf >= 0 && pxf < vpps && pyf >= 0 && pyf < vpps, "far page (px,py) in [0,vpps)");
    }

    // ================= MarkResidentPages over a known point-set -> the expected set ===================
    {
        // Two distinct points at the camera level -> 2 distinct pages; a duplicate point -> still 1 page.
        std::vector<Vec3> pts = {
            {0.0f, 0.0f, 0.0f},   // page (0,4,4) = 36
            {2.0f, 0.0f, 0.0f},   // page (0,5,4) = 4*8+5 = 37
            {0.0f, 0.0f, 0.0f},   // duplicate -> same page 36 (set, order-independent)
        };
        std::vector<uint32_t> resident((size_t)pageCount, 0xCDu);  // poison to prove zeroing
        vsm::MarkResidentPages(std::span<const Vec3>(pts.data(), pts.size()), cm,
                               std::span<uint32_t>(resident.data(), resident.size()));
        uint32_t setCount = 0;
        for (uint32_t r : resident) { check(r == 0u || r == 1u, "resident entry is 0/1 (zeroed first)"); setCount += r; }
        check(setCount == 2u, "MarkResidentPages over {a,b,a} -> 2 distinct resident pages");
        check(resident[36] == 1u, "page 36 resident");
        check(resident[37] == 1u, "page 37 resident (point +2 X -> (0,5,4))");

        // A single point -> exactly 1 resident page at the expected slot.
        std::vector<Vec3> one = {{0.0f, 0.0f, 0.0f}};
        std::vector<uint32_t> r1((size_t)pageCount, 0);
        vsm::MarkResidentPages(std::span<const Vec3>(one.data(), one.size()), cm,
                               std::span<uint32_t>(r1.data(), r1.size()));
        uint32_t c1 = 0; for (uint32_t r : r1) c1 += r;
        check(c1 == 1u && r1[36] == 1u, "single point -> exactly 1 resident page (36)");
    }

    // ================= markingEnabled-off semantics: empty receiver span -> cleared all-zero set =====
    {
        std::vector<uint32_t> resident((size_t)pageCount, 0x7Fu);  // poison
        std::vector<Vec3> none;
        vsm::MarkResidentPages(std::span<const Vec3>(none.data(), none.size()), cm,
                               std::span<uint32_t>(resident.data(), resident.size()));
        bool allZero = true;
        for (uint32_t r : resident) if (r != 0u) allZero = false;
        check(allZero, "markingEnabled-off (no receivers) -> cleared all-zero resident set");
    }

    // ================= DETERMINISM: two MarkResidentPages passes byte-identical =======================
    {
        // A ground grid spanning several clipmap levels (the showcase point-set shape).
        std::vector<Vec3> grid;
        for (int z = -10; z <= 10; ++z)
            for (int x = -10; x <= 10; ++x)
                grid.push_back(Vec3{(float)x * 5.0f, 0.0f, (float)z * 5.0f});
        std::vector<uint32_t> a((size_t)pageCount, 0), b((size_t)pageCount, 0);
        vsm::MarkResidentPages(std::span<const Vec3>(grid.data(), grid.size()), cm,
                               std::span<uint32_t>(a.data(), a.size()));
        vsm::MarkResidentPages(std::span<const Vec3>(grid.data(), grid.size()), cm,
                               std::span<uint32_t>(b.data(), b.size()));
        bool det = std::memcmp(a.data(), b.data(), a.size() * sizeof(uint32_t)) == 0;
        check(det, "MarkResidentPages deterministic (two passes byte-identical)");
        uint32_t setCount = 0; for (uint32_t r : a) setCount += r;
        check(setCount > 1u, "ground grid marks >1 resident page (a real set)");
    }

    if (g_fail == 0) std::printf("vsm_test: ALL PASS\n");
    else std::printf("vsm_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
