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

    if (g_fail == 0) std::printf("vt_test: ALL PASS\n");
    else std::printf("vt_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
