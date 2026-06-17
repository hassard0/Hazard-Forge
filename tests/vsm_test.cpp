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

#include <cmath>
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

    // =========================== Slice VB — physical-page allocator ==================================
    // AllocatePhysicalPages: resident pages -> SEQUENTIAL tile indices in ascending pageId order;
    // non-resident -> kNoTile; allocated count == resident count under the cap.
    {
        vsm::VsmAtlas atlas;        // tilesPerSide=16, tileSize=128 -> 256 tiles, 2048² atlas (defaults).
        check(atlas.tileCount() == 256, "VsmAtlas default tileCount == 256");
        check(atlas.atlasSize() == 2048, "VsmAtlas default atlasSize == 2048");

        // A small resident set at known pageIds: {3, 5, 36} -> tiles 0,1,2 in ascending pageId order.
        std::vector<uint32_t> resident((size_t)pageCount, 0u);
        resident[3] = 1u; resident[5] = 1u; resident[36] = 1u;
        std::vector<uint32_t> indir((size_t)pageCount, 7u);  // poison
        int allocated = vsm::AllocatePhysicalPages(
            std::span<const uint32_t>(resident.data(), resident.size()), atlas,
            std::span<uint32_t>(indir.data(), indir.size()));
        check(allocated == 3, "AllocatePhysicalPages: 3 resident -> 3 tiles allocated");
        check(indir[3] == 0u, "page 3 (lowest) -> tile 0");
        check(indir[5] == 1u, "page 5 -> tile 1 (ascending pageId order)");
        check(indir[36] == 2u, "page 36 -> tile 2");
        // Every non-resident page -> kNoTile.
        uint32_t residentCount = 0, noTileCount = 0;
        for (int p = 0; p < pageCount; ++p) {
            if (resident[p]) { residentCount++; check(indir[p] != vsm::kNoTile, "resident page has a tile"); }
            else { check(indir[p] == vsm::kNoTile, "non-resident page -> kNoTile"); noTileCount++; }
        }
        check((int)noTileCount == pageCount - 3, "all non-resident pages -> kNoTile");

        // Tile indices are a 0..N-1 contiguous range (a bijection resident->[0,allocated)).
        std::vector<uint8_t> tileSeen((size_t)allocated, 0);
        bool contiguous = true;
        for (int p = 0; p < pageCount; ++p)
            if (indir[p] != vsm::kNoTile) {
                if (indir[p] >= (uint32_t)allocated || tileSeen[indir[p]]) contiguous = false;
                else tileSeen[indir[p]] = 1;
            }
        check(contiguous, "allocated tile indices are a contiguous [0,N) bijection");
    }

    // AllocatePhysicalPages OVERFLOW: more resident pages than the atlas cap -> first `cap` pages (lowest
    // pageId) get tiles, the overflow gets kNoTile, allocated == cap.
    {
        vsm::VsmAtlas tiny; tiny.tilesPerSide = 2; tiny.tileSize = 64;  // 4 tiles only.
        check(tiny.tileCount() == 4, "tiny atlas tileCount == 4");
        std::vector<uint32_t> resident((size_t)pageCount, 0u);
        // Mark 6 resident pages -> only the 4 lowest-pageId get tiles.
        int marked[6] = {2, 7, 10, 11, 20, 33};
        for (int m : marked) resident[m] = 1u;
        std::vector<uint32_t> indir((size_t)pageCount, 9u);
        int allocated = vsm::AllocatePhysicalPages(
            std::span<const uint32_t>(resident.data(), resident.size()), tiny,
            std::span<uint32_t>(indir.data(), indir.size()));
        check(allocated == 4, "overflow: allocated clamped to cap (4)");
        check(indir[2] == 0u && indir[7] == 1u && indir[10] == 2u && indir[11] == 3u,
              "overflow: the 4 lowest-pageId resident pages got tiles 0..3");
        check(indir[20] == vsm::kNoTile && indir[33] == vsm::kNoTile,
              "overflow: the 2 highest-pageId resident pages -> kNoTile (priority = ascending pageId)");
    }

    // PhysicalTileOrigin: tile index <-> pixel origin round-trip (row-major layout).
    {
        vsm::VsmAtlas atlas;  // tilesPerSide=16, tileSize=128.
        for (uint32_t t = 0; t < (uint32_t)atlas.tileCount(); ++t) {
            int x, y; vsm::PhysicalTileOrigin(t, atlas, x, y);
            int col = (int)t % atlas.tilesPerSide, row = (int)t / atlas.tilesPerSide;
            check(x == col * atlas.tileSize && y == row * atlas.tileSize, "tile origin == (col,row)*tileSize");
            // Recover the tile index from the origin.
            uint32_t recovered = (uint32_t)((y / atlas.tileSize) * atlas.tilesPerSide + (x / atlas.tileSize));
            check(recovered == t, "origin -> tile index round-trip");
        }
        // Tile 0 -> (0,0); tile 16 (one full row) -> (0, tileSize); tile 17 -> (tileSize, tileSize).
        int x, y;
        vsm::PhysicalTileOrigin(0, atlas, x, y);  check(x == 0 && y == 0, "tile 0 -> (0,0)");
        vsm::PhysicalTileOrigin(16, atlas, x, y); check(x == 0 && y == 128, "tile 16 -> (0,128) (row 1)");
        vsm::PhysicalTileOrigin(17, atlas, x, y); check(x == 128 && y == 128, "tile 17 -> (128,128)");
    }

    // PageWorldOrtho: page world center/extent matches the clipmap level math (level 0 small/central,
    // higher levels larger). Must agree EXACTLY with MarkPage's snap/extent (the inverse mapping).
    {
        // Level 0, center page (4,4) (== MarkPage(at-camera) page 36): levelExtent=16, pageWorldSize=2,
        // origin=floor((0-8)/2)*2=-8. center.x = -8 + (4+0.5)*2 = 1.0; halfExtent = 1.0.
        vsm::PageOrtho o0 = vsm::PageWorldOrtho(36, cm);
        check(std::fabs(o0.halfExtent - 1.0f) < 1e-5f, "level 0 page halfExtent == pageWorldSize/2 == 1");
        check(std::fabs(o0.center.x - 1.0f) < 1e-5f, "level 0 page (4,4) center.x == 1.0");
        check(std::fabs(o0.center.z - 1.0f) < 1e-5f, "level 0 page (4,4) center.z == 1.0");
        check(std::fabs(o0.center.y - cm.cameraPos.y) < 1e-6f, "page center on the clipmap ground plane");

        // Level 3 (top): levelExtent=128, pageWorldSize=16 -> 8x larger pages than level 0.
        int id3 = vsm::PageId(3, 0, 0, cm);
        vsm::PageOrtho o3 = vsm::PageWorldOrtho(id3, cm);
        check(std::fabs(o3.halfExtent - 8.0f) < 1e-4f, "level 3 page halfExtent == 8 (8x level 0)");
        check(o3.halfExtent > o0.halfExtent, "higher levels cover larger world regions");

        // A page that MarkPage marks for a known receiver must contain that receiver inside its extent.
        int markedId = -1; vsm::MarkPage(Vec3{5.0f, 0.0f, 3.0f}, cm, markedId);
        vsm::PageOrtho om = vsm::PageWorldOrtho(markedId, cm);
        check(std::fabs(5.0f - om.center.x) <= om.halfExtent + 1e-4f &&
              std::fabs(3.0f - om.center.z) <= om.halfExtent + 1e-4f,
              "PageWorldOrtho of a MarkPage'd page contains the receiver");
    }

    // VB determinism: AllocatePhysicalPages two passes byte-identical.
    {
        std::vector<Vec3> grid;
        for (int z = -10; z <= 10; ++z)
            for (int x = -10; x <= 10; ++x)
                grid.push_back(Vec3{(float)x * 5.0f, 0.0f, (float)z * 5.0f});
        std::vector<uint32_t> resident((size_t)pageCount, 0u);
        vsm::MarkResidentPages(std::span<const Vec3>(grid.data(), grid.size()), cm,
                               std::span<uint32_t>(resident.data(), resident.size()));
        vsm::VsmAtlas atlas;
        std::vector<uint32_t> a((size_t)pageCount, 0), b((size_t)pageCount, 0);
        int na = vsm::AllocatePhysicalPages(std::span<const uint32_t>(resident.data(), resident.size()),
                                            atlas, std::span<uint32_t>(a.data(), a.size()));
        int nb = vsm::AllocatePhysicalPages(std::span<const uint32_t>(resident.data(), resident.size()),
                                            atlas, std::span<uint32_t>(b.data(), b.size()));
        check(na == nb && std::memcmp(a.data(), b.data(), a.size() * sizeof(uint32_t)) == 0,
              "AllocatePhysicalPages deterministic (two passes byte-identical)");
        check(na > 1, "ground grid allocates >1 physical tile");
    }

    if (g_fail == 0) std::printf("vsm_test: ALL PASS\n");
    else std::printf("vsm_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
