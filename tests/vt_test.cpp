// Slice VT1 — Runtime Virtual Texturing Slice 1: VT page table + page-needed FEEDBACK/MARKING. Pure
// CPU (header-only, no device, no backend symbols). Mirrors engine/render/vt.h, the SAME mip page-table
// math + page marking the --vt-feedback-shot (Vulkan) / --vt-feedback (Metal) showcases +
// shaders/vt_feedback.comp.hlsl run. Namespace hf::render::vt.
//
// What this test PINS (the contracts the GPU vt_feedback.comp + the GPU==CPU proof build on):
//   * pagesPerSide(mip) = max(1, vpps0 >> mip); mipPageOffset is the PREFIX-SUM of pagesPerSide(k)²
//     (NOT mip*vpps² — the VT-vs-VSM crux); pageCount() == the total over all mips.
//   * PageId / UnpackPageId round-trip over the FULL [0, pageCount()) range (a bijection); PageId is
//     INJECTIVE over (mip,px,py) — no two distinct cells alias the same flat slot.
//   * SelectMipLevel: density 0/1 -> mip 0; the 2^m boundary densities select the expected mip; far ->
//     the coarsest mip, clamped (the threshold-ladder edges, the DETERMINISM CRUX).
//   * VtPageId corners: u,v at 0 / just-below-1 / mid -> the expected page; clamp at the boundary.
//   * MarkFeedbackPages over a known request-set -> the expected page set (count + membership);
//     feedbackEnabled=false (empty request span) -> the cleared all-zero set.
//   * DETERMINISM: two MarkFeedbackPages passes bit-identical.
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/vt.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace vt = hf::render::vt;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // The fixed showcase VT: mipLevels=4, pageSize=128, virtualPagesPerSideMip0=16. Pages per side
    // 16/8/4/2 over 4 mips; pageCount = 256+64+16+4 = 340.
    vt::VtTexture vtx;
    vtx.mipLevels = 4;
    vtx.pageSize = 128;
    vtx.virtualPagesPerSideMip0 = 16;

    const int pageCount = vtx.pageCount();

    // ================= pagesPerSide / mipPageOffset prefix-sum =================
    {
        check(vtx.pagesPerSide(0) == 16, "pagesPerSide(0) == 16");
        check(vtx.pagesPerSide(1) == 8, "pagesPerSide(1) == 8");
        check(vtx.pagesPerSide(2) == 4, "pagesPerSide(2) == 4");
        check(vtx.pagesPerSide(3) == 2, "pagesPerSide(3) == 2");

        // mipPageOffset is the PREFIX-SUM of pagesPerSide(k)² for k<mip (NOT mip*vpps²).
        check(vtx.mipPageOffset(0) == 0, "mipPageOffset(0) == 0");
        check(vtx.mipPageOffset(1) == 256, "mipPageOffset(1) == 16²=256");
        check(vtx.mipPageOffset(2) == 256 + 64, "mipPageOffset(2) == 256+64=320");
        check(vtx.mipPageOffset(3) == 256 + 64 + 16, "mipPageOffset(3) == 256+64+16=336");
        check(pageCount == 256 + 64 + 16 + 4, "pageCount() == 340 (prefix-sum over all mips)");

        // Offsets are strictly monotonic (each mip adds >=1 page).
        bool monotonic = true;
        for (int m = 1; m < vtx.mipLevels; ++m)
            if (vtx.mipPageOffset(m) <= vtx.mipPageOffset(m - 1)) monotonic = false;
        check(monotonic, "mipPageOffset strictly monotonic increasing");

        // A floor-at-1 VT (more mips than pages allow): pagesPerSide never drops below 1.
        vt::VtTexture deep; deep.mipLevels = 8; deep.virtualPagesPerSideMip0 = 4;
        check(deep.pagesPerSide(2) == 1, "deep mip 2 -> pagesPerSide 1");
        check(deep.pagesPerSide(7) == 1, "deep mip 7 -> pagesPerSide floored at 1 (never 0)");
        // pageCount stays well-defined (4²+2²+1+1+1+1+1+1 = 16+4+6 = 26).
        check(deep.pageCount() == 16 + 4 + 1 + 1 + 1 + 1 + 1 + 1, "deep pageCount() prefix-sum with floor");
    }

    // ================= PageId / UnpackPageId bijection over [0, pageCount()) =================
    {
        bool roundTripOk = true;
        std::vector<uint8_t> seen((size_t)pageCount, 0);
        bool injective = true;
        for (int mip = 0; mip < vtx.mipLevels; ++mip) {
            const int pps = vtx.pagesPerSide(mip);
            for (int py = 0; py < pps; ++py)
                for (int px = 0; px < pps; ++px) {
                    int id = vt::PageId(mip, px, py, vtx);
                    if (id < 0 || id >= pageCount) { injective = false; continue; }
                    if (seen[(size_t)id]) injective = false;
                    seen[(size_t)id] = 1;
                    int um, upx, upy;
                    vt::UnpackPageId(id, vtx, um, upx, upy);
                    if (um != mip || upx != px || upy != py) roundTripOk = false;
                }
        }
        check(roundTripOk, "PageId/UnpackPageId round-trip over (mip,px,py)");
        check(injective, "PageId injective (covers [0,pageCount()) exactly once — a bijection)");

        // Every flat id covered exactly once.
        bool allSeen = true;
        for (int id = 0; id < pageCount; ++id) if (!seen[(size_t)id]) allSeen = false;
        check(allSeen, "PageId surjective over [0,pageCount()) (the set is fully covered)");

        // The inverse over EVERY flat id is in range + re-packs to the same id.
        bool inverseOk = true;
        for (int id = 0; id < pageCount; ++id) {
            int m, px, py; vt::UnpackPageId(id, vtx, m, px, py);
            const int pps = vtx.pagesPerSide(m);
            if (m < 0 || m >= vtx.mipLevels || px < 0 || px >= pps || py < 0 || py >= pps) inverseOk = false;
            if (vt::PageId(m, px, py, vtx) != id) inverseOk = false;
        }
        check(inverseOk, "UnpackPageId(id) in-range + PageId(UnpackPageId(id))==id for all id");

        // Spot-check the mip boundaries: id 0 -> (0,0,0); id 255 -> mip0 last page; id 256 -> mip1 (0,0).
        int m, px, py;
        vt::UnpackPageId(0, vtx, m, px, py);   check(m == 0 && px == 0 && py == 0, "id 0 -> (mip0,0,0)");
        vt::UnpackPageId(255, vtx, m, px, py); check(m == 0 && px == 15 && py == 15, "id 255 -> mip0 (15,15)");
        vt::UnpackPageId(256, vtx, m, px, py); check(m == 1 && px == 0 && py == 0, "id 256 -> mip1 (0,0)");
        vt::UnpackPageId(pageCount - 1, vtx, m, px, py);
        check(m == 3 && px == 1 && py == 1, "last id -> mip3 (1,1) (the coarsest page)");
    }

    // ================= SelectMipLevel: threshold-ladder edges (the DETERMINISM CRUX) =============
    {
        // thresholds[m] = 2^m = {1, 2, 4, 8}. mip = #thresholds density exceeds, clamped.
        float thresholds[vt::kMaxMips];
        vt::BuildMipThresholds(vtx, thresholds);
        check(thresholds[0] == 1.0f && thresholds[1] == 2.0f &&
              thresholds[2] == 4.0f && thresholds[3] == 8.0f, "mip thresholds = 2^m");

        check(vt::SelectMipLevel(0.0f, vtx) == 0, "density 0 -> mip 0");
        check(vt::SelectMipLevel(1.0f, vtx) == 0, "density == threshold[0] (1) -> mip 0 (not >)");
        check(vt::SelectMipLevel(1.0001f, vtx) == 1, "density just past 1 -> mip 1");
        check(vt::SelectMipLevel(2.0001f, vtx) == 2, "density just past 2 -> mip 2");
        check(vt::SelectMipLevel(4.0001f, vtx) == 3, "density just past 4 -> mip 3 (coarsest)");
        check(vt::SelectMipLevel(8.0f, vtx) == 3, "density 8 -> mip 3 (clamped)");
        check(vt::SelectMipLevel(100000.0f, vtx) == 3, "very large -> coarsest mip (clamped, no overflow)");
    }

    // ================= VtPageId corners: u,v at 0 / just-below-1 / mid -> expected page ===========
    {
        // Mip 0: pps=16. u=0 -> px 0; u just below 1 -> px 15; u=0.5 -> floor(0.5*16)=8.
        check(vt::VtPageId(0.0f, 0.0f, 0, vtx) == vt::PageId(0, 0, 0, vtx), "VtPageId (0,0) mip0 -> (0,0)");
        int idCorner = vt::VtPageId(0.99999f, 0.99999f, 0, vtx);
        int m, px, py; vt::UnpackPageId(idCorner, vtx, m, px, py);
        check(m == 0 && px == 15 && py == 15, "VtPageId (just<1) mip0 -> (15,15)");
        int idMid = vt::VtPageId(0.5f, 0.5f, 0, vtx);
        vt::UnpackPageId(idMid, vtx, m, px, py);
        check(m == 0 && px == 8 && py == 8, "VtPageId (0.5,0.5) mip0 -> (8,8)");

        // Clamp at the boundary: u==1.0 -> floor(16)==16 clamped to 15; u>1 / u<0 clamp too.
        int idAt1 = vt::VtPageId(1.0f, 1.0f, 0, vtx);
        vt::UnpackPageId(idAt1, vtx, m, px, py);
        check(m == 0 && px == 15 && py == 15, "VtPageId (1.0,1.0) mip0 -> clamped (15,15)");
        int idOver = vt::VtPageId(2.0f, -1.0f, 0, vtx);
        vt::UnpackPageId(idOver, vtx, m, px, py);
        check(m == 0 && px == 15 && py == 0, "VtPageId (2.0,-1.0) mip0 -> clamped (15,0)");

        // Mip 1: pps=8. u=0.5 -> floor(4)=4; just<1 -> px 7.
        int idM1 = vt::VtPageId(0.5f, 0.5f, 1, vtx);
        vt::UnpackPageId(idM1, vtx, m, px, py);
        check(m == 1 && px == 4 && py == 4, "VtPageId (0.5,0.5) mip1 -> (4,4)");

        // Mip 3 (coarsest): pps=2. u=0.4 -> px 0; u=0.6 -> px 1.
        vt::UnpackPageId(vt::VtPageId(0.4f, 0.4f, 3, vtx), vtx, m, px, py);
        check(m == 3 && px == 0 && py == 0, "VtPageId (0.4,0.4) mip3 -> (0,0)");
        vt::UnpackPageId(vt::VtPageId(0.6f, 0.6f, 3, vtx), vtx, m, px, py);
        check(m == 3 && px == 1 && py == 1, "VtPageId (0.6,0.6) mip3 -> (1,1)");

        // SnapRequest agrees with VtPageId (the host-snap path the GPU consumes).
        vt::SampleRequest sr{0.3f, 0.7f, 1};
        vt::SnappedRequest sn = vt::SnapRequest(sr, vtx);
        check(vt::PageId(sn.mip, sn.px, sn.py, vtx) == vt::VtPageId(sr.u, sr.v, sr.mip, vtx),
              "SnapRequest -> PageId == VtPageId (host-snap path matches the CPU ref)");
    }

    // ================= MarkFeedbackPages over a known request-set -> the expected set ===============
    {
        // Three requests: two distinct pages + a duplicate -> 2 distinct pages.
        std::vector<vt::SampleRequest> reqs = {
            {0.5f, 0.5f, 0},   // mip0 (8,8) = PageId(0,8,8) = 8*16+8 = 136
            {0.0f, 0.0f, 0},   // mip0 (0,0) = 0
            {0.5f, 0.5f, 0},   // duplicate -> same page 136 (set, order-independent)
        };
        const int p136 = vt::PageId(0, 8, 8, vtx);
        const int p0 = vt::PageId(0, 0, 0, vtx);
        check(p136 == 136 && p0 == 0, "known PageIds (0,8,8)=136 / (0,0,0)=0");

        std::vector<uint32_t> feedback((size_t)pageCount, 0xCDu);  // poison to prove zeroing
        vt::MarkFeedbackPages(std::span<const vt::SampleRequest>(reqs.data(), reqs.size()), vtx,
                              std::span<uint32_t>(feedback.data(), feedback.size()));
        uint32_t setCount = 0;
        for (uint32_t f : feedback) { check(f == 0u || f == 1u, "feedback entry is 0/1 (zeroed first)"); setCount += f; }
        check(setCount == 2u, "MarkFeedbackPages over {a,b,a} -> 2 distinct resident pages");
        check(feedback[(size_t)p136] == 1u, "page 136 resident");
        check(feedback[(size_t)p0] == 1u, "page 0 resident");

        // A multi-mip request set -> a page in each requested mip.
        std::vector<vt::SampleRequest> multi = {
            {0.5f, 0.5f, 0}, {0.5f, 0.5f, 1}, {0.5f, 0.5f, 2}, {0.5f, 0.5f, 3},
        };
        std::vector<uint32_t> fb2((size_t)pageCount, 0);
        vt::MarkFeedbackPages(std::span<const vt::SampleRequest>(multi.data(), multi.size()), vtx,
                              std::span<uint32_t>(fb2.data(), fb2.size()));
        uint32_t c2 = 0; for (uint32_t f : fb2) c2 += f;
        check(c2 == 4u, "multi-mip request set -> 4 distinct pages (one per mip)");
        bool perMip = true;
        for (int mip = 0; mip < vtx.mipLevels; ++mip) {
            int id = vt::VtPageId(0.5f, 0.5f, mip, vtx);
            if (fb2[(size_t)id] != 1u) perMip = false;
        }
        check(perMip, "each requested mip's page is resident");
    }

    // ================= feedbackEnabled=false semantics: empty request span -> cleared set =========
    {
        std::vector<uint32_t> feedback((size_t)pageCount, 0x7Fu);  // poison
        std::vector<vt::SampleRequest> none;
        vt::MarkFeedbackPages(std::span<const vt::SampleRequest>(none.data(), none.size()), vtx,
                              std::span<uint32_t>(feedback.data(), feedback.size()));
        bool allZero = true;
        for (uint32_t f : feedback) if (f != 0u) allZero = false;
        check(allZero, "feedbackEnabled-off (no requests) -> cleared all-zero feedback set");
    }

    // ================= DETERMINISM: two MarkFeedbackPages passes byte-identical ===================
    {
        // A UV grid sampled across several mips (the showcase request-set shape).
        std::vector<vt::SampleRequest> grid;
        for (int gy = 0; gy < 12; ++gy)
            for (int gx = 0; gx < 12; ++gx) {
                float u = (float)gx / 12.0f, v = (float)gy / 12.0f;
                int mip = (gx + gy) % vtx.mipLevels;
                grid.push_back({u, v, mip});
            }
        std::vector<uint32_t> a((size_t)pageCount, 0), b((size_t)pageCount, 0);
        vt::MarkFeedbackPages(std::span<const vt::SampleRequest>(grid.data(), grid.size()), vtx,
                              std::span<uint32_t>(a.data(), a.size()));
        vt::MarkFeedbackPages(std::span<const vt::SampleRequest>(grid.data(), grid.size()), vtx,
                              std::span<uint32_t>(b.data(), b.size()));
        bool det = std::memcmp(a.data(), b.data(), a.size() * sizeof(uint32_t)) == 0;
        check(det, "MarkFeedbackPages deterministic (two passes byte-identical)");
        uint32_t setCount = 0; for (uint32_t f : a) setCount += f;
        check(setCount > 1u, "UV grid marks >1 resident page (a real set)");
    }

    // ================= Slice VT2: AllocatePhysicalTiles + VtTilePool + PhysicalTileOrigin =========
    {
        // Hand-verified ascending assignment: a known small feedback set -> a known indirection. Pages
        // 0,2,5 resident in a 9-page set; pool cap 4 (>= 3 resident) -> all 3 allocated in ascending order:
        // page 0 -> tile 0, page 2 -> tile 1, page 5 -> tile 2; the rest kNoTile.
        std::vector<uint32_t> fb = {1, 0, 1, 0, 0, 1, 0, 0, 0};
        vt::VtTilePool pool; pool.tilesPerSide = 2;   // cap = 4
        check(pool.tileCapacity() == 4, "VtTilePool{2}.tileCapacity() == 4");
        std::vector<int32_t> ind =
            vt::AllocatePhysicalTiles(std::span<const uint32_t>(fb.data(), fb.size()), pool);
        check(ind.size() == fb.size(), "indirection sized feedback.size()");
        check(ind[0] == 0, "page 0 (first resident) -> tile 0 (ascending)");
        check(ind[1] == vt::kNoTile, "page 1 non-resident -> kNoTile");
        check(ind[2] == 1, "page 2 (2nd resident) -> tile 1");
        check(ind[3] == vt::kNoTile && ind[4] == vt::kNoTile, "pages 3,4 non-resident -> kNoTile");
        check(ind[5] == 2, "page 5 (3rd resident) -> tile 2");
        check(ind[6] == vt::kNoTile && ind[7] == vt::kNoTile && ind[8] == vt::kNoTile,
              "trailing non-resident -> kNoTile");
        check(vt::kNoTile == -1, "kNoTile == -1 (int32 CPU form)");
        check(vt::kNoTileU32 == 0xFFFFFFFFu && (uint32_t)vt::kNoTile == vt::kNoTileU32,
              "kNoTileU32 == 0xFFFFFFFF == (uint32_t)kNoTile (SSBO sentinel round-trip)");

        // BuildIndirection is the same table.
        std::vector<int32_t> ind2 =
            vt::BuildIndirection(std::span<const uint32_t>(fb.data(), fb.size()), pool);
        check(std::memcmp(ind.data(), ind2.data(), ind.size() * sizeof(int32_t)) == 0,
              "BuildIndirection == AllocatePhysicalTiles");
    }

    // overflow: cap < resident -> exactly cap allocated, the rest kNoTile, ascending wins.
    {
        // 6-page set all resident; pool cap 4 -> first 4 (pages 0..3) get tiles 0..3, pages 4,5 overflow.
        std::vector<uint32_t> fb = {1, 1, 1, 1, 1, 1};
        vt::VtTilePool pool; pool.tilesPerSide = 2;   // cap = 4
        std::vector<int32_t> ind =
            vt::AllocatePhysicalTiles(std::span<const uint32_t>(fb.data(), fb.size()), pool);
        int allocated = 0; for (int32_t t : ind) if (t != vt::kNoTile) ++allocated;
        check(allocated == 4, "overflow: exactly cap(4) allocated when resident(6) > cap");
        check(ind[0] == 0 && ind[1] == 1 && ind[2] == 2 && ind[3] == 3,
              "overflow: first cap pages (ascending) get tiles 0..3");
        check(ind[4] == vt::kNoTile && ind[5] == vt::kNoTile,
              "overflow: pages past cap -> kNoTile (ascending priority wins)");
    }

    // cap >= resident -> all resident allocated.
    {
        std::vector<uint32_t> fb = {0, 1, 0, 1, 0, 0, 1, 0};   // 3 resident
        vt::VtTilePool pool; pool.tilesPerSide = 3;   // cap = 9 >= 3
        std::vector<int32_t> ind =
            vt::AllocatePhysicalTiles(std::span<const uint32_t>(fb.data(), fb.size()), pool);
        int allocated = 0, resident = 0;
        for (size_t i = 0; i < fb.size(); ++i) {
            if (fb[i]) ++resident;
            if (ind[i] != vt::kNoTile) { ++allocated; check(fb[i] != 0u, "allocated tile implies resident"); }
            else check(fb[i] == 0u, "kNoTile implies non-resident (cap >= resident)");
        }
        check(allocated == resident && resident == 3, "cap >= resident -> all resident allocated");
        check(ind[1] == 0 && ind[3] == 1 && ind[6] == 2, "ascending: resident pages -> tiles 0,1,2");
    }

    // allocEnabled-off modeled as no-alloc -> all kNoTile. (The CPU models 'disabled' as an empty/zero
    // feedback set; the GPU writes all kNoTile explicitly. Both yield an all-kNoTile indirection.)
    {
        std::vector<uint32_t> empty(20, 0u);   // no resident pages
        vt::VtTilePool pool; pool.tilesPerSide = 4;
        std::vector<int32_t> ind =
            vt::AllocatePhysicalTiles(std::span<const uint32_t>(empty.data(), empty.size()), pool);
        bool allNoTile = true; for (int32_t t : ind) if (t != vt::kNoTile) allNoTile = false;
        check(allNoTile, "allocEnabled-off / empty feedback -> all kNoTile (no-op)");
    }

    // tile indices unique + in [0, cap) + ascending-priority over the VT1 showcase scene.
    {
        // The real showcase: VT1 scene + the SAME 576 requests -> feedback -> alloc with tilesPerSide=12.
        std::vector<vt::SampleRequest> requests;
        const int kGrid = 24;
        for (int gy = 0; gy < kGrid; ++gy)
            for (int gx = 0; gx < kGrid; ++gx) {
                float u = (float)gx / (float)kGrid, v = (float)gy / (float)kGrid;
                int mip = (gx + gy) % vtx.mipLevels;
                requests.push_back({u, v, mip});
            }
        std::vector<uint32_t> feedback((size_t)pageCount, 0u);
        vt::MarkFeedbackPages(std::span<const vt::SampleRequest>(requests.data(), requests.size()), vtx,
                              std::span<uint32_t>(feedback.data(), feedback.size()));
        int resident = 0; for (uint32_t f : feedback) resident += (f != 0u);

        vt::VtTilePool pool; pool.tilesPerSide = 12;   // cap = 144 < resident (overflow exercised)
        const int cap = pool.tileCapacity();
        check(cap == 144, "showcase pool cap == 144");
        check(resident > cap, "showcase resident > cap (overflow path exercised)");

        std::vector<int32_t> ind =
            vt::AllocatePhysicalTiles(std::span<const uint32_t>(feedback.data(), feedback.size()), pool);

        // Tile indices unique + in [0, cap); allocated count == min(resident, cap); ascending-priority:
        // the first `cap` resident pages by pageId are exactly the allocated ones, tiles assigned 0..cap-1
        // in pageId order.
        std::vector<uint8_t> tileSeen((size_t)cap, 0);
        int allocated = 0, expectTile = 0;
        bool uniqueInRange = true, ascending = true, overflowKNoTile = true;
        int residentSeen = 0;
        for (size_t pageId = 0; pageId < feedback.size(); ++pageId) {
            int32_t t = ind[pageId];
            if (feedback[pageId] != 0u) {
                ++residentSeen;
                if (residentSeen <= cap) {
                    // within capacity -> must be allocated, in order
                    if (t != expectTile) ascending = false;
                    ++expectTile;
                } else {
                    // overflow resident -> kNoTile
                    if (t != vt::kNoTile) overflowKNoTile = false;
                }
            } else {
                if (t != vt::kNoTile) uniqueInRange = false;  // non-resident must be kNoTile
            }
            if (t != vt::kNoTile) {
                ++allocated;
                if (t < 0 || t >= cap) uniqueInRange = false;
                else { if (tileSeen[(size_t)t]) uniqueInRange = false; tileSeen[(size_t)t] = 1; }
            }
        }
        check(allocated == cap, "showcase allocated == min(resident,cap) == cap (overflow)");
        check(uniqueInRange, "allocated tile indices unique + in [0,cap), non-resident kNoTile");
        check(ascending, "ascending-priority: first cap resident pages get tiles 0..cap-1 in pageId order");
        check(overflowKNoTile, "overflow resident pages (past cap) -> kNoTile");
    }

    // PhysicalTileOrigin corners (row-major layout).
    {
        vt::VtTilePool pool; pool.tilesPerSide = 12;
        const int ps = vtx.pageSize;  // 128
        int ox, oy;
        vt::PhysicalTileOrigin(0, pool, ps, ox, oy);
        check(ox == 0 && oy == 0, "tile 0 origin (0,0)");
        vt::PhysicalTileOrigin(1, pool, ps, ox, oy);
        check(ox == ps && oy == 0, "tile 1 origin (pageSize,0)");
        vt::PhysicalTileOrigin(11, pool, ps, ox, oy);
        check(ox == 11 * ps && oy == 0, "tile 11 (row 0 last) origin (11*ps,0)");
        vt::PhysicalTileOrigin(12, pool, ps, ox, oy);
        check(ox == 0 && oy == ps, "tile 12 (row 1 first) origin (0,pageSize)");
        vt::PhysicalTileOrigin(13, pool, ps, ox, oy);
        check(ox == ps && oy == ps, "tile 13 origin (pageSize,pageSize)");
        vt::PhysicalTileOrigin(143, pool, ps, ox, oy);
        check(ox == 11 * ps && oy == 11 * ps, "tile 143 (last) origin (11*ps,11*ps)");
    }

    // determinism: two AllocatePhysicalTiles passes byte-identical.
    {
        std::vector<uint32_t> fb((size_t)pageCount, 0u);
        for (int i = 0; i < pageCount; i += 3) fb[(size_t)i] = 1u;  // a deterministic resident subset
        vt::VtTilePool pool; pool.tilesPerSide = 12;
        std::vector<int32_t> a =
            vt::AllocatePhysicalTiles(std::span<const uint32_t>(fb.data(), fb.size()), pool);
        std::vector<int32_t> b =
            vt::AllocatePhysicalTiles(std::span<const uint32_t>(fb.data(), fb.size()), pool);
        check(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(int32_t)) == 0,
              "AllocatePhysicalTiles deterministic (two passes byte-identical)");
    }

    // ================= Slice VT3: PageTexel + GeneratePhysicalAtlas + BuildTilePageId ===============
    // PageTexel determinism + adjacent-page/adjacent-mip DIFFERENT base colors + pattern correct at known
    // (lx,ly). These pin the EXACT integer math shaders/vt_pagegen.comp copies VERBATIM.
    {
        // Determinism: same inputs -> same texel (twice).
        check(vt::PageTexel(42, 1, 7, 13) == vt::PageTexel(42, 1, 7, 13),
              "PageTexel deterministic (same inputs -> same texel)");

        // Alpha is always opaque 0xFF; channels always in [0x40,0xFF] (distinct from kAtlasClear's 0x10).
        bool alphaOpaque = true, chanInRange = true;
        for (int s = 0; s < 64; ++s) {
            uint32_t t = vt::PageTexel(s * 3, s % 4, (s * 7) & 63, (s * 5) & 63);
            if ((t >> 24) != 0xFFu) alphaOpaque = false;
            uint32_t r = t & 0xFFu, g = (t >> 8) & 0xFFu, b = (t >> 16) & 0xFFu;
            if (r < 0x40u || g < 0x40u || b < 0x40u) chanInRange = false;
            if (t == vt::kAtlasClear) chanInRange = false;  // never collide with the clear sentinel
        }
        check(alphaOpaque, "PageTexel alpha always opaque (0xFF)");
        check(chanInRange, "PageTexel channels in [0x40,0xFF] (never collides with kAtlasClear)");

        // Adjacent PAGES (different pageId) -> different base color (the same texel position differs).
        check(vt::PageTexel(10, 0, 0, 0) != vt::PageTexel(11, 0, 0, 0),
              "adjacent pages (10 vs 11) -> DIFFERENT texel at (0,0)");
        check(vt::PageTexel(100, 0, 32, 32) != vt::PageTexel(101, 0, 32, 32),
              "adjacent pages (100 vs 101) -> DIFFERENT texel at (32,32)");

        // Adjacent MIPS (same pageId, different mip) -> different base color.
        check(vt::PageTexel(7, 0, 0, 0) != vt::PageTexel(7, 1, 0, 0),
              "adjacent mips (0 vs 1) -> DIFFERENT texel at (0,0)");
        check(vt::PageTexel(7, 1, 16, 16) != vt::PageTexel(7, 2, 16, 16),
              "adjacent mips (1 vs 2) -> DIFFERENT texel at (16,16)");

        // The checkerboard pattern: toggling lx across an 8-block boundary flips the dark/light cell, so a
        // texel in a light cell (check==1) is brighter (or equal) than the same base in a dark cell. Pick a
        // page whose base channels are large enough that 5/8 scaling is a strict decrease (>= 8 so floor !=
        // base). The 8×8 checker: (lx>>3)^(ly>>3). At (0,0) check=0^0=0 (dark); at (8,0) check=1^0=1 (light).
        {
            uint32_t darkCell  = vt::PageTexel(3, 0, 0, 0);    // (0>>3)^(0>>3) = 0 -> dark (5/8)
            uint32_t lightCell = vt::PageTexel(3, 0, 8, 0);    // (8>>3)^(0>>3) = 1 -> light (full); same grad step 0
            // grad = (lx>>4)+(ly>>4): at lx=0 -> 0; at lx=8 -> 0 (8>>4==0). So ONLY the checker differs.
            uint32_t dr = darkCell & 0xFFu, lr = lightCell & 0xFFu;
            check(lr >= dr, "checkerboard: light cell channel >= dark cell channel (same gradient step)");
            check(darkCell != lightCell || dr == 0x40u,
                  "checkerboard: adjacent 8-blocks differ (unless both clamp to the floor)");
        }
    }

    // BuildTilePageId is a correct INVERSE of the indirection over the allocated set.
    {
        // indirection: page 0 -> tile 2, page 3 -> tile 0, page 5 -> tile 1; the rest kNoTile.
        std::vector<int32_t> indir(8, vt::kNoTile);
        indir[0] = 2; indir[3] = 0; indir[5] = 1;
        vt::VtTilePool pool; pool.tilesPerSide = 2;   // cap = 4
        std::vector<uint32_t> tpi =
            vt::BuildTilePageId(std::span<const int32_t>(indir.data(), indir.size()), pool);
        check(tpi.size() == 4u, "tilePageId sized tileCapacity (4)");
        check(tpi[2] == 0u, "tile 2 -> page 0 (inverse)");
        check(tpi[0] == 3u, "tile 0 -> page 3 (inverse)");
        check(tpi[1] == 5u, "tile 1 -> page 5 (inverse)");
        check(tpi[3] == vt::kNoTileU32, "unallocated tile 3 -> kNoTileU32");
        // Round-trip: for every allocated page, tpi[indir[page]] == page.
        bool inverseOk = true;
        for (size_t p = 0; p < indir.size(); ++p)
            if (indir[p] != vt::kNoTile && tpi[(size_t)indir[p]] != (uint32_t)p) inverseOk = false;
        check(inverseOk, "BuildTilePageId is the exact inverse of the indirection over the allocated set");
    }

    // GeneratePhysicalAtlas: known indirection -> known atlas (allocated tiles filled with the right page's
    // pattern, unallocated kAtlasClear). Use a tiny atlas: tilesPerSide=2, pageSize=4 -> 8×8 atlas.
    {
        vt::VtTexture small; small.mipLevels = 2; small.pageSize = 4; small.virtualPagesPerSideMip0 = 2;
        // pages: mip0 = 2×2 = 4 pages (0..3), mip1 = 1×1 = 1 page (4). pageCount = 5.
        check(small.pageCount() == 5, "small VT pageCount == 5");
        vt::VtTilePool pool; pool.tilesPerSide = 2;       // cap = 4
        vt::VtAtlasDims dims; dims.tilesPerSide = 2; dims.pageSize = 4;
        check(dims.atlasW() == 8 && dims.atlasH() == 8 && dims.atlasTexels() == 64,
              "small atlas 8×8 = 64 texels");

        // indirection: page 0 -> tile 0, page 1 -> tile 1, page 4 (mip1) -> tile 2; pages 2,3 kNoTile.
        std::vector<int32_t> indir(5, vt::kNoTile);
        indir[0] = 0; indir[1] = 1; indir[4] = 2;

        std::vector<uint32_t> atlas((size_t)dims.atlasTexels(), 0u);
        vt::GeneratePhysicalAtlas(std::span<const int32_t>(indir.data(), indir.size()), small, pool, dims,
                                  std::span<uint32_t>(atlas.data(), atlas.size()), true);

        // Tile 0 origin (0,0): every texel == PageTexel(page 0, mip 0, lx, ly).
        bool tile0Ok = true, tile1Ok = true, tile2Ok = true;
        int mip, px, py;
        vt::UnpackPageId(0, small, mip, px, py);
        for (int ly = 0; ly < 4; ++ly) for (int lx = 0; lx < 4; ++lx)
            if (atlas[(size_t)ly * 8 + lx] != vt::PageTexel(0, mip, lx, ly)) tile0Ok = false;
        check(tile0Ok, "atlas tile 0 == PageTexel(page 0, ...) over its 4×4 region");

        // Tile 1 origin (4,0): page 1, mip 0.
        vt::UnpackPageId(1, small, mip, px, py);
        for (int ly = 0; ly < 4; ++ly) for (int lx = 0; lx < 4; ++lx)
            if (atlas[(size_t)ly * 8 + (4 + lx)] != vt::PageTexel(1, mip, lx, ly)) tile1Ok = false;
        check(tile1Ok, "atlas tile 1 == PageTexel(page 1, ...) over its region");

        // Tile 2 origin (0,4): page 4 (mip 1).
        vt::UnpackPageId(4, small, mip, px, py);
        check(mip == 1, "page 4 unpacks to mip 1");
        for (int ly = 0; ly < 4; ++ly) for (int lx = 0; lx < 4; ++lx)
            if (atlas[(size_t)(4 + ly) * 8 + lx] != vt::PageTexel(4, mip, lx, ly)) tile2Ok = false;
        check(tile2Ok, "atlas tile 2 == PageTexel(page 4 / mip 1, ...) over its region");

        // Tile 3 origin (4,4) is unallocated -> all kAtlasClear.
        bool tile3Clear = true;
        for (int ly = 0; ly < 4; ++ly) for (int lx = 0; lx < 4; ++lx)
            if (atlas[(size_t)(4 + ly) * 8 + (4 + lx)] != vt::kAtlasClear) tile3Clear = false;
        check(tile3Clear, "unallocated tile 3 -> all kAtlasClear");

        // No allocated tile contains a kAtlasClear hole; exactly 3 tiles (48 texels) generated.
        uint32_t generated = 0, cleared = 0;
        for (uint32_t t : atlas) { if (t == vt::kAtlasClear) ++cleared; else ++generated; }
        check(generated == 48u && cleared == 16u,
              "3 allocated tiles -> 48 generated texels, 16 kAtlasClear (1 free tile)");
    }

    // genEnabled=false -> all kAtlasClear (the disabled-path no-op).
    {
        vt::VtTexture small; small.mipLevels = 2; small.pageSize = 4; small.virtualPagesPerSideMip0 = 2;
        vt::VtTilePool pool; pool.tilesPerSide = 2;
        vt::VtAtlasDims dims; dims.tilesPerSide = 2; dims.pageSize = 4;
        std::vector<int32_t> indir(5, vt::kNoTile);
        indir[0] = 0; indir[1] = 1;   // some allocated, but disabled -> ignored
        std::vector<uint32_t> atlas((size_t)dims.atlasTexels(), 0u);
        vt::GeneratePhysicalAtlas(std::span<const int32_t>(indir.data(), indir.size()), small, pool, dims,
                                  std::span<uint32_t>(atlas.data(), atlas.size()), false);
        bool allClear = true; for (uint32_t t : atlas) if (t != vt::kAtlasClear) allClear = false;
        check(allClear, "GeneratePhysicalAtlas genEnabled=false -> all kAtlasClear (no-op)");
    }

    // GeneratePhysicalAtlas determinism: two passes byte-identical (over the real showcase config).
    {
        vt::VtTexture vfull; vfull.mipLevels = 4; vfull.pageSize = 128; vfull.virtualPagesPerSideMip0 = 16;
        std::vector<vt::SampleRequest> requests;
        const int kg = 24;
        for (int gy = 0; gy < kg; ++gy) for (int gx = 0; gx < kg; ++gx)
            requests.push_back({(float)gx / (float)kg, (float)gy / (float)kg, (gx + gy) % vfull.mipLevels});
        std::vector<uint32_t> fb((size_t)vfull.pageCount(), 0u);
        vt::MarkFeedbackPages(std::span<const vt::SampleRequest>(requests.data(), requests.size()), vfull,
                              std::span<uint32_t>(fb.data(), fb.size()));
        vt::VtTilePool pool; pool.tilesPerSide = 16;   // 256 tiles > resident(212) -> some tiles stay free
        vt::VtAtlasDims dims; dims.tilesPerSide = 16; dims.pageSize = 64;
        int resident = 0; for (uint32_t f : fb) resident += (f != 0u);
        std::vector<int32_t> indir =
            vt::AllocatePhysicalTiles(std::span<const uint32_t>(fb.data(), fb.size()), pool);
        int allocated = 0; for (int32_t t : indir) if (t != vt::kNoTile) ++allocated;
        check(resident == 212, "showcase pagegen config: resident == 212");
        check(allocated == resident && allocated < pool.tileCapacity(),
              "showcase pagegen config: all 212 resident allocated, free tiles remain (clear regions shown)");

        std::vector<uint32_t> a((size_t)dims.atlasTexels(), 0u), b((size_t)dims.atlasTexels(), 0u);
        vt::GeneratePhysicalAtlas(std::span<const int32_t>(indir.data(), indir.size()), vfull, pool, dims,
                                  std::span<uint32_t>(a.data(), a.size()), true);
        vt::GeneratePhysicalAtlas(std::span<const int32_t>(indir.data(), indir.size()), vfull, pool, dims,
                                  std::span<uint32_t>(b.data(), b.size()), true);
        check(std::memcmp(a.data(), b.data(), a.size() * sizeof(uint32_t)) == 0,
              "GeneratePhysicalAtlas deterministic (two passes byte-identical)");

        // Every allocated tile fully written (no kAtlasClear hole); the rest kAtlasClear.
        std::vector<uint32_t> tpi = vt::BuildTilePageId(
            std::span<const int32_t>(indir.data(), indir.size()), pool);
        bool coverageOk = true;
        for (int ty = 0; ty < pool.tilesPerSide && coverageOk; ++ty)
            for (int tx = 0; tx < pool.tilesPerSide && coverageOk; ++tx) {
                bool alloc = (tpi[(size_t)(ty * pool.tilesPerSide + tx)] != vt::kNoTileU32);
                for (int ly = 0; ly < dims.pageSize && coverageOk; ++ly)
                    for (int lx = 0; lx < dims.pageSize; ++lx) {
                        bool clear = a[(size_t)(ty * dims.pageSize + ly) * dims.atlasW()
                                       + (tx * dims.pageSize + lx)] == vt::kAtlasClear;
                        if (alloc == clear) { coverageOk = false; break; }  // alloc must be non-clear, free must be clear
                    }
            }
        check(coverageOk, "every allocated tile fully written, every free tile all kAtlasClear");
    }

    // ================= Slice VT4: SampleVirtualTexel + ReconstructVirtualImage =====================
    // SampleVirtualTexel: a resident page -> the EXACT atlas texel (== PageTexel); a non-resident page ->
    // kVtMiss; the (vx,vy)->(page,local) decomposition correct at page boundaries.
    {
        // A small virtual texture + a tiny 2×2-tile atlas. mip 0 = 2 pages/side, pageSize=4 -> 8×8 atlas.
        vt::VtTexture small; small.mipLevels = 2; small.pageSize = 4; small.virtualPagesPerSideMip0 = 2;
        vt::VtTilePool pool; pool.tilesPerSide = 2;       // 4 tiles
        vt::VtAtlasDims dims; dims.tilesPerSide = 2; dims.pageSize = 4;  // 8×8 atlas

        // Feedback: mip-0 pages 0 and 3 resident, page 1 and 2 NON-resident; mip-1's single page non-resident.
        // pageCount = 4 (mip0) + 1 (mip1) = 5.
        std::vector<uint32_t> fb((size_t)small.pageCount(), 0u);
        fb[vt::PageId(0, 0, 0, small)] = 1u;  // page 0
        fb[vt::PageId(0, 1, 1, small)] = 1u;  // page 3 (px=1,py=1)
        std::vector<int32_t> indir =
            vt::AllocatePhysicalTiles(std::span<const uint32_t>(fb.data(), fb.size()), pool);
        std::vector<uint32_t> atlas((size_t)dims.atlasTexels(), vt::kAtlasClear);
        vt::GeneratePhysicalAtlas(std::span<const int32_t>(indir.data(), indir.size()), small, pool, dims,
                                  std::span<uint32_t>(atlas.data(), atlas.size()), true);

        // A virtual texel inside resident page 0 (px=0,py=0): local (lx,ly)=(vx,vy) -> PageTexel(page0,mip0,lx,ly).
        {
            const int pid0 = vt::PageId(0, 0, 0, small);
            bool page0Ok = true;
            for (int vy = 0; vy < 4 && page0Ok; ++vy)
                for (int vx = 0; vx < 4; ++vx) {
                    uint32_t got = vt::SampleVirtualTexel(vx, vy, 0,
                        std::span<const int32_t>(indir.data(), indir.size()),
                        std::span<const uint32_t>(atlas.data(), atlas.size()), small, pool, dims);
                    if (got != vt::PageTexel(pid0, 0, vx, vy)) { page0Ok = false; break; }
                }
            check(page0Ok, "SampleVirtualTexel resident page 0 -> exact PageTexel over its 4×4 region");
        }

        // A virtual texel inside resident page 3 (px=1,py=1): virtual (vx,vy) in [4,8)² -> local = vx%4,vy%4.
        {
            const int pid3 = vt::PageId(0, 1, 1, small);
            bool page3Ok = true;
            for (int vy = 4; vy < 8 && page3Ok; ++vy)
                for (int vx = 4; vx < 8; ++vx) {
                    uint32_t got = vt::SampleVirtualTexel(vx, vy, 0,
                        std::span<const int32_t>(indir.data(), indir.size()),
                        std::span<const uint32_t>(atlas.data(), atlas.size()), small, pool, dims);
                    if (got != vt::PageTexel(pid3, 0, vx % 4, vy % 4)) { page3Ok = false; break; }
                }
            check(page3Ok, "SampleVirtualTexel resident page 3 (px=1,py=1) -> exact PageTexel, page boundary correct");
        }

        // A virtual texel inside NON-resident page 1 (px=1,py=0): vx in [4,8), vy in [0,4) -> kVtMiss.
        {
            uint32_t got = vt::SampleVirtualTexel(5, 1, 0,
                std::span<const int32_t>(indir.data(), indir.size()),
                std::span<const uint32_t>(atlas.data(), atlas.size()), small, pool, dims);
            check(got == vt::kVtMiss, "SampleVirtualTexel non-resident page -> kVtMiss");
        }

        // kVtMiss is DISTINCT from kAtlasClear and from any PageTexel (its green lane is forced >= 0x40).
        check(vt::kVtMiss != vt::kAtlasClear, "kVtMiss distinct from kAtlasClear");
        {
            bool distinctFromGen = true;
            for (int s = 0; s < 4096 && distinctFromGen; ++s) {
                uint32_t t = vt::PageTexel(s * 3, s % 4, (s * 7) & 63, (s * 5) & 63);
                if (t == vt::kVtMiss) distinctFromGen = false;
            }
            check(distinctFromGen, "kVtMiss never collides with a generated PageTexel");
        }
    }

    // ReconstructVirtualImage: known indirection+atlas -> known image; resident pages == PageTexel, the rest
    // kVtMiss. sampleEnabled=false -> all kVtMiss. Two passes byte-identical.
    {
        vt::VtTexture small; small.mipLevels = 2; small.pageSize = 4; small.virtualPagesPerSideMip0 = 2;
        vt::VtTilePool pool; pool.tilesPerSide = 2;
        vt::VtAtlasDims dims; dims.tilesPerSide = 2; dims.pageSize = 4;
        std::vector<uint32_t> fb((size_t)small.pageCount(), 0u);
        fb[vt::PageId(0, 0, 0, small)] = 1u;  // page 0 resident
        fb[vt::PageId(0, 1, 1, small)] = 1u;  // page 3 resident
        std::vector<int32_t> indir =
            vt::AllocatePhysicalTiles(std::span<const uint32_t>(fb.data(), fb.size()), pool);
        std::vector<uint32_t> atlas((size_t)dims.atlasTexels(), vt::kAtlasClear);
        vt::GeneratePhysicalAtlas(std::span<const int32_t>(indir.data(), indir.size()), small, pool, dims,
                                  std::span<uint32_t>(atlas.data(), atlas.size()), true);

        const int W = small.pagesPerSide(0) * dims.pageSize;  // 2*4 = 8
        std::vector<uint32_t> img((size_t)W * W, 0u);
        vt::ReconstructVirtualImage(0, std::span<const int32_t>(indir.data(), indir.size()),
                                    std::span<const uint32_t>(atlas.data(), atlas.size()), small, pool, dims,
                                    std::span<uint32_t>(img.data(), img.size()), true);

        // Image == SampleVirtualTexel per texel (the definition).
        bool imageOk = true;
        for (int vy = 0; vy < W && imageOk; ++vy)
            for (int vx = 0; vx < W; ++vx) {
                uint32_t expect = vt::SampleVirtualTexel(vx, vy, 0,
                    std::span<const int32_t>(indir.data(), indir.size()),
                    std::span<const uint32_t>(atlas.data(), atlas.size()), small, pool, dims);
                if (img[(size_t)vy * W + vx] != expect) { imageOk = false; break; }
            }
        check(imageOk, "ReconstructVirtualImage == SampleVirtualTexel per virtual texel");

        // Resident page 0 quadrant textured, non-resident pages kVtMiss.
        check(img[0] == vt::PageTexel(vt::PageId(0, 0, 0, small), 0, 0, 0),
              "ReconstructVirtualImage resident page 0 origin == PageTexel");
        check(img[(size_t)1 * W + 5] == vt::kVtMiss,
              "ReconstructVirtualImage non-resident page 1 texel == kVtMiss");

        // sampleEnabled=false -> all kVtMiss.
        std::vector<uint32_t> off((size_t)W * W, 0u);
        vt::ReconstructVirtualImage(0, std::span<const int32_t>(indir.data(), indir.size()),
                                    std::span<const uint32_t>(atlas.data(), atlas.size()), small, pool, dims,
                                    std::span<uint32_t>(off.data(), off.size()), false);
        bool allMiss = true; for (uint32_t t : off) if (t != vt::kVtMiss) { allMiss = false; break; }
        check(allMiss, "ReconstructVirtualImage sampleEnabled=false -> all kVtMiss (no-op)");

        // Determinism: two passes byte-identical.
        std::vector<uint32_t> img2((size_t)W * W, 0u);
        vt::ReconstructVirtualImage(0, std::span<const int32_t>(indir.data(), indir.size()),
                                    std::span<const uint32_t>(atlas.data(), atlas.size()), small, pool, dims,
                                    std::span<uint32_t>(img2.data(), img2.size()), true);
        check(std::memcmp(img.data(), img2.data(), img.size() * sizeof(uint32_t)) == 0,
              "ReconstructVirtualImage deterministic (two passes byte-identical)");
    }

    // Round-trip self-consistency over the REAL showcase config (mip 0, 16 pages/side, pageSize 64): every
    // resident virtual texel == PageTexel(pageId,0,lx,ly); every non-resident == kVtMiss; counts sane.
    {
        vt::VtTexture vfull; vfull.mipLevels = 4; vfull.pageSize = 128; vfull.virtualPagesPerSideMip0 = 16;
        std::vector<vt::SampleRequest> requests;
        const int kg = 24;
        for (int gy = 0; gy < kg; ++gy) for (int gx = 0; gx < kg; ++gx)
            requests.push_back({(float)gx / (float)kg, (float)gy / (float)kg, (gx + gy) % vfull.mipLevels});
        std::vector<uint32_t> fb((size_t)vfull.pageCount(), 0u);
        vt::MarkFeedbackPages(std::span<const vt::SampleRequest>(requests.data(), requests.size()), vfull,
                              std::span<uint32_t>(fb.data(), fb.size()));
        vt::VtTilePool pool; pool.tilesPerSide = 16;
        vt::VtAtlasDims dims; dims.tilesPerSide = 16; dims.pageSize = 64;
        std::vector<int32_t> indir =
            vt::AllocatePhysicalTiles(std::span<const uint32_t>(fb.data(), fb.size()), pool);
        std::vector<uint32_t> atlas((size_t)dims.atlasTexels(), vt::kAtlasClear);
        vt::GeneratePhysicalAtlas(std::span<const int32_t>(indir.data(), indir.size()), vfull, pool, dims,
                                  std::span<uint32_t>(atlas.data(), atlas.size()), true);

        const int W = vfull.pagesPerSide(0) * dims.pageSize;  // 16*64 = 1024
        std::vector<uint32_t> img((size_t)W * W, 0u);
        vt::ReconstructVirtualImage(0, std::span<const int32_t>(indir.data(), indir.size()),
                                    std::span<const uint32_t>(atlas.data(), atlas.size()), vfull, pool, dims,
                                    std::span<uint32_t>(img.data(), img.size()), true);

        int residentPages = 0, missPages = 0;
        for (int py = 0; py < 16; ++py) for (int px = 0; px < 16; ++px) {
            int pid = vt::PageId(0, px, py, vfull);
            if (indir[(size_t)pid] != vt::kNoTile) ++residentPages; else ++missPages;
        }
        check(residentPages + missPages == 256, "mip-0 has 256 pages (16×16)");

        bool selfConsistent = true;
        long textured = 0;
        for (int vy = 0; vy < W && selfConsistent; ++vy)
            for (int vx = 0; vx < W; ++vx) {
                uint32_t got = img[(size_t)vy * W + vx];
                int pid = vt::PageId(0, vx / 64, vy / 64, vfull);
                if (indir[(size_t)pid] != vt::kNoTile) {
                    if (got != vt::PageTexel(pid, 0, vx % 64, vy % 64)) { selfConsistent = false; break; }
                    ++textured;
                } else if (got != vt::kVtMiss) { selfConsistent = false; break; }
            }
        check(selfConsistent,
              "showcase sample: resident texel == PageTexel + non-resident == kVtMiss (round-trip self-consistent)");
        check(textured == (long)residentPages * 64 * 64,
              "showcase sample: textured texels == residentPages * pageSize²");
    }

    if (g_fail == 0) std::printf("vt_test: ALL PASS\n");
    else std::printf("vt_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
