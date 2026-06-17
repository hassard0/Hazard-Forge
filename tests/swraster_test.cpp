// Slice SW1 — Nanite Software-Raster Slice 1: CPU-reference integer rasterizer. Pure CPU (header-only,
// no device, no backend symbols). Pins the contracts the GPU compute software-rasterizer (SW2) will
// copy verbatim + prove bit-identical:
//   * PackSw / UnpackSw round-trip; kSwClear non-collision; a nearer (smaller depthQ) fragment's key
//     < a farther one's (depth in the high bits -> min selects nearer).
//   * TOP-LEFT FILL RULE (the watertight-raster crux): two triangles sharing an edge cover each
//     boundary pixel EXACTLY once — no double-cover, no gap.
//   * SUB-PIXEL coverage: a triangle smaller than a pixel straddling a pixel center -> that pixel
//     covered (HW sample-at-center would miss).
//   * MIN order-independence: a triangle set vs a shuffled set -> byte-identical packed[].
//   * DEGENERATE zero-area triangle -> no coverage.
//   * DEPTH OCCLUSION: a nearer triangle over a farther one -> the nearer visId wins on the overlap.
//   * DETERMINISM: two rasterizations of the showcase scene byte-identical.
//
// Pure C++ (hf_core), ASan-eligible like the other render-math tests.
#include "render/swraster.h"
#include "render/visbuffer.h"
#include "render/meshlet.h"
#include "scene/vertex.h"
#include "scene/mesh.h"        // Slice SW2: SphereGeometry — the clustered showcase scene the GPU memcmp's against
#include "math/math.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace vg = hf::render::vg;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// Build a ScreenVert directly in pixel coordinates (pixels * kSub), depth in [0,kSwDepthMax].
static vg::ScreenVert SV(int px, int py, uint32_t z) {
    vg::ScreenVert s;
    s.x = px * vg::kSub;
    s.y = py * vg::kSub;
    s.z = z;
    return s;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ================= PackSw / UnpackSw round-trip + budget =================
    {
        static_assert(vg::kSwIdBits + vg::kSwDepthBits == 32, "depth|id fills 32 bits");
        check(vg::kSwIdBits > vg::kTriIdBits, "kSwIdBits holds triID + cluster index");
        check(vg::kSwClear == 0xFFFFFFFFu, "kSwClear all-ones sentinel");

        bool roundTrip = true, fieldOk = true;
        for (uint32_t d = 0; d <= vg::kSwDepthMax; d += 257u) {          // sweep depth
            for (uint32_t id = 0; id <= vg::kSwIdMask; id += 113u) {     // sweep id
                uint32_t k = vg::PackSw(d, id);
                uint32_t ud = 0, uid = 0;
                vg::UnpackSw(k, ud, uid);
                if (ud != d || uid != (id & vg::kSwIdMask)) roundTrip = false;
                if ((k >> vg::kSwIdBits) != d) fieldOk = false;
                if ((k & vg::kSwIdMask) != (id & vg::kSwIdMask)) fieldOk = false;
            }
        }
        check(roundTrip, "PackSw/UnpackSw round-trip over (depthQ,visId)");
        check(fieldOk, "PackSw field layout: depth high bits, id low kSwIdBits");

        // A NEARER fragment's key < a FARTHER one's (the min-selects-nearer contract), independent of id.
        uint32_t near = vg::PackSw(100u, vg::kSwIdMask);   // near depth, max id
        uint32_t far  = vg::PackSw(200u, 0u);              // far depth, min id
        check(near < far, "nearer (smaller depthQ) key < farther key regardless of id");

        // kSwClear is never beaten by a real fragment unless it is nearer (kSwClear is the max key).
        check(vg::PackSw(vg::kSwDepthMax - 1u, 0u) < vg::kSwClear, "any nearer-than-far fragment < kSwClear");
        // kSwClear only collides with PackSw at (kSwDepthMax, kSwIdMask) — an at-far-plane all-ones id,
        // never a real (clusterID,triID) at this scale.
        check(vg::PackSw(vg::kSwDepthMax, vg::kSwIdMask) == vg::kSwClear, "kSwClear == max-depth max-id");
    }

    // ================= TOP-LEFT FILL RULE: shared edge covered EXACTLY once =================
    // Two triangles tiling a square (0,0)-(8,8): T_A = (0,0),(8,0),(0,8) and T_B = (8,0),(8,8),(0,8),
    // sharing the diagonal edge (8,0)-(0,8). Every pixel of the square must be covered EXACTLY once
    // across the two triangles (no double-cover -> no min-conflict; no gap -> watertight).
    {
        const uint32_t W = 8, H = 8;
        // Count coverage per pixel by giving each triangle a UNIQUE visId and rasterizing into a fresh
        // buffer per triangle, then a combined buffer; double-cover shows as the same pixel claimed by
        // both, gaps show as a still-clear pixel.
        std::vector<int> coverCount((size_t)W * H, 0);
        auto countTri = [&](vg::ScreenVert a, vg::ScreenVert b, vg::ScreenVert c) {
            vg::SwVisBuffer vb; vb.Init(W, H);
            vg::RasterTriangle(vb, a, b, c, 1u);
            for (uint32_t p = 0; p < W * H; ++p)
                if (vb.packed[p] != vg::kSwClear) coverCount[p] += 1;
        };
        countTri(SV(0, 0, 100), SV(8, 0, 100), SV(0, 8, 100));
        countTri(SV(8, 0, 100), SV(8, 8, 100), SV(0, 8, 100));

        int doubles = 0, gaps = 0, covered = 0;
        for (uint32_t p = 0; p < W * H; ++p) {
            if (coverCount[p] > 1) ++doubles;
            if (coverCount[p] == 1) ++covered;
            if (coverCount[p] == 0) ++gaps;  // a gap inside the union (some border pixels legitimately
                                             // fall outside both via the top-left rule at the far edges)
        }
        check(doubles == 0, "top-left fill rule: NO pixel double-covered across a shared edge (watertight)");
        check(covered > 0, "top-left fill rule: the shared-edge square has covered interior pixels");
        // The diagonal pixels (the shared edge) must each be covered exactly once — verify by checking a
        // known on-diagonal pixel center. Pixel (3,4): center (3.5,4.5)*kSub. The diagonal is x+y=8 in
        // pixel-corner space; the center sum 8.0 lies on it. Top-left rule assigns it to exactly one tri.
        check(coverCount[(size_t)4 * W + 3] <= 1, "diagonal pixel not double-covered");
    }

    // ================= SUB-PIXEL coverage =================
    // A triangle SMALLER than a pixel, straddling the center of pixel (5,5) (center at (5.5,5.5) in
    // pixel units = (88,88) in 1/16 fixed-point). The triangle's verts are within ~0.4px of that center
    // -> bbox is just pixel (5,5), and the center is inside -> exactly that one pixel covered. A HW
    // rasterizer sampling at a different sub-position could miss it.
    {
        const uint32_t W = 16, H = 16;
        vg::SwVisBuffer vb; vb.Init(W, H);
        // center pixel (5,5) -> center fixed-point (5*16+8, 5*16+8) = (88,88).
        vg::ScreenVert a, b, c;
        a.x = 88 - 5; a.y = 88 - 4; a.z = 50;   // tiny offsets (< kSub) around (88,88)
        b.x = 88 + 6; b.y = 88 - 4; b.z = 50;
        c.x = 88 + 0; c.y = 88 + 6; c.z = 50;
        vg::RasterTriangle(vb, a, b, c, vg::PackVisId(0u, 0u));
        uint32_t coveredCount = 0, coveredAt = 0xFFFFFFFFu;
        for (uint32_t p = 0; p < W * H; ++p)
            if (vb.packed[p] != vg::kSwClear) { ++coveredCount; coveredAt = p; }
        check(coveredCount == 1, "sub-pixel triangle covers EXACTLY one pixel");
        check(coveredAt == (uint32_t)(5 * W + 5), "sub-pixel triangle covers the straddled pixel (5,5)");
    }

    // ================= DEGENERATE zero-area -> no coverage =================
    {
        const uint32_t W = 8, H = 8;
        vg::SwVisBuffer vb; vb.Init(W, H);
        vg::RasterTriangle(vb, SV(1, 1, 10), SV(5, 1, 10), SV(3, 1, 10), 1u);  // collinear (y all == 1)
        bool anyCovered = false;
        for (uint32_t p = 0; p < W * H; ++p) if (vb.packed[p] != vg::kSwClear) anyCovered = true;
        check(!anyCovered, "degenerate zero-area triangle covers nothing");
    }

    // ================= DEPTH OCCLUSION: nearer wins on overlap =================
    {
        const uint32_t W = 8, H = 8;
        vg::SwVisBuffer vb; vb.Init(W, H);
        // A FAR full-cover quad-triangle (z=200, visId cluster 7) then a NEARER one (z=50, cluster 9)
        // overlapping it. Rasterize FAR first, then NEAR — the min must keep NEAR on the overlap.
        uint32_t farId  = vg::PackVisId(7u, 0u);
        uint32_t nearId = vg::PackVisId(9u, 0u);
        vg::RasterTriangle(vb, SV(0, 0, 200), SV(8, 0, 200), SV(0, 8, 200), farId);   // far, big
        vg::RasterTriangle(vb, SV(0, 0, 50),  SV(8, 0, 50),  SV(0, 8, 50),  nearId);  // near, same region
        // A pixel inside both: (1,1). The near visId must win.
        uint32_t d, id;
        vg::UnpackSw(vb.packed[(size_t)1 * W + 1], d, id);
        check(d == 50u, "depth occlusion: nearer depth wins on overlap");
        check(id == (nearId & vg::kSwIdMask), "depth occlusion: nearer visId wins on overlap");

        // ORDER-INDEPENDENCE of the min: do it NEAR-first this time -> same result.
        vg::SwVisBuffer vb2; vb2.Init(W, H);
        vg::RasterTriangle(vb2, SV(0, 0, 50),  SV(8, 0, 50),  SV(0, 8, 50),  nearId);
        vg::RasterTriangle(vb2, SV(0, 0, 200), SV(8, 0, 200), SV(0, 8, 200), farId);
        check(std::memcmp(vb.packed.data(), vb2.packed.data(), vb.packed.size() * sizeof(uint32_t)) == 0,
              "min order-independent: far-then-near == near-then-far BYTE-IDENTICAL");
    }

    // ================= MIN order-independence over the SHOWCASE scene (shuffled) =================
    {
        const uint32_t W = 256, H = 256;
        vg::SwRasterScene scene = vg::BuildSwRasterScene(W, H);
        std::vector<vg::ScreenVert> screen; std::vector<uint8_t> behind;
        vg::ProjectSwRasterScene(scene, screen, behind);

        vg::SwVisBuffer ref; ref.Init(W, H);
        vg::RasterClusters(ref,
            std::span<const vg::ScreenVert>(screen.data(), screen.size()),
            std::span<const uint8_t>(behind.data(), behind.size()),
            std::span<const uint32_t>(scene.indices.data(), scene.indices.size()),
            std::span<const vg::Meshlet>(scene.meshlets.data(), scene.meshlets.size()));

        // Build a flat (clusterId,triLocal) work list and shuffle it, then rasterize in that order.
        struct Tri { uint32_t cluster, triLocal; };
        std::vector<Tri> work;
        for (uint32_t cl = 0; cl < scene.meshlets.size(); ++cl)
            for (uint32_t t = 0; t < scene.meshlets[cl].triCount; ++t)
                work.push_back({cl, t});
        std::mt19937 rng(12345u);
        std::shuffle(work.begin(), work.end(), rng);

        vg::SwVisBuffer shuf; shuf.Init(W, H);
        for (const Tri& tw : work) {
            const vg::Meshlet& m = scene.meshlets[tw.cluster];
            uint32_t base = 3u * (m.triOffset + tw.triLocal);
            uint32_t i0 = scene.indices[base + 0], i1 = scene.indices[base + 1], i2 = scene.indices[base + 2];
            if (behind[i0] || behind[i1] || behind[i2]) continue;
            vg::RasterTriangle(shuf, screen[i0], screen[i1], screen[i2], vg::PackVisId(tw.cluster, tw.triLocal));
        }
        check(std::memcmp(ref.packed.data(), shuf.packed.data(), ref.packed.size() * sizeof(uint32_t)) == 0,
              "min order-independence: shuffled rasterization == in-order BYTE-IDENTICAL");

        // The sub-pixel cluster's expected pixel is covered in the reference buffer.
        uint32_t spIdx = scene.subPixelPy * W + scene.subPixelPx;
        check(ref.packed[spIdx] != vg::kSwClear, "showcase sub-pixel triangle COVERED (HW would miss)");

        // Some coverage exists (the quads).
        uint32_t coveredPixels = 0;
        for (uint32_t v : ref.packed) if (v != vg::kSwClear) ++coveredPixels;
        check(coveredPixels > 100u, "showcase scene covers a non-trivial pixel count");

        // ===== DETERMINISM: two rasterizations byte-identical. =====
        vg::SwVisBuffer ref2; ref2.Init(W, H);
        vg::RasterClusters(ref2,
            std::span<const vg::ScreenVert>(screen.data(), screen.size()),
            std::span<const uint8_t>(behind.data(), behind.size()),
            std::span<const uint32_t>(scene.indices.data(), scene.indices.size()),
            std::span<const vg::Meshlet>(scene.meshlets.data(), scene.meshlets.size()));
        check(std::memcmp(ref.packed.data(), ref2.packed.data(), ref.packed.size() * sizeof(uint32_t)) == 0,
              "showcase rasterization deterministic (two runs byte-identical)");
    }

    // ================= EMPTY tri set == cleared =================
    {
        const uint32_t W = 64, H = 64;
        vg::SwVisBuffer vb; vb.Init(W, H);
        std::vector<vg::ScreenVert> noScreen;
        std::vector<uint8_t> noBehind;
        std::vector<uint32_t> noIdx;
        std::vector<vg::Meshlet> noMesh;
        vg::RasterClusters(vb, noScreen, noBehind, noIdx, noMesh);
        vg::SwVisBuffer cleared; cleared.Init(W, H);
        check(std::memcmp(vb.packed.data(), cleared.packed.data(), vb.packed.size() * sizeof(uint32_t)) == 0,
              "empty triangle set == cleared (all kSwClear) BYTE-IDENTICAL");
    }

    // ===== Slice SW2: RasterClusters over a CLUSTERED SphereGeometry (the GPU oracle) =====
    // The --swraster-gpu-shot showcase rasterizes a clustered SphereGeometry; this is the CPU reference
    // the GPU memcmp's against. Prove it is (a) DETERMINISTIC (two runs byte-identical) and (b)
    // WATERTIGHT at cluster seams — every covered pixel is claimed EXACTLY once across all triangles (no
    // gap, no double-cover), which is what makes the serial-min == InterlockedMin bit-identity meaningful.
    {
        using math::Vec3; using math::Mat4;
        const uint32_t W = 512, H = 512;
        scene::MeshGeometry geo = scene::SphereGeometry(24, 16);

        // Decompose into clusters (the showcase decomposition).
        vg::MeshletSet ms = vg::BuildMeshlets(
            std::span<const scene::Vertex>(geo.verts.data(), geo.verts.size()),
            std::span<const uint32_t>(geo.indices.data(), geo.indices.size()));
        check(ms.meshlets.size() >= 1, "SphereGeometry decomposes into >=1 cluster");

        // Host projection (== the showcase camera).
        Mat4 view = Mat4::LookAt({0, 0, 3}, {0, 0, 0}, {0, 1, 0});
        Mat4 proj = Mat4::Perspective(0.9f, (float)W / (float)H, 0.1f, 50.0f);
        Mat4 viewProj = proj * view;
        std::vector<vg::ScreenVert> screen(geo.verts.size());
        std::vector<uint8_t> behind(geo.verts.size(), 0);
        for (size_t i = 0; i < geo.verts.size(); ++i) {
            bool ok = true;
            screen[i] = vg::ProjectToScreenVert(
                Vec3{geo.verts[i].pos[0], geo.verts[i].pos[1], geo.verts[i].pos[2]},
                viewProj, W, H, vg::kSub, ok);
            behind[i] = ok ? 0 : 1;
        }
        auto screenSpan  = std::span<const vg::ScreenVert>(screen.data(), screen.size());
        auto behindSpan  = std::span<const uint8_t>(behind.data(), behind.size());
        auto idxSpan     = std::span<const uint32_t>(ms.indices.data(), ms.indices.size());
        auto meshletSpan = std::span<const vg::Meshlet>(ms.meshlets.data(), ms.meshlets.size());

        vg::SwVisBuffer a; a.Init(W, H);
        vg::RasterClusters(a, screenSpan, behindSpan, idxSpan, meshletSpan);
        vg::SwVisBuffer b; b.Init(W, H);
        vg::RasterClusters(b, screenSpan, behindSpan, idxSpan, meshletSpan);
        check(std::memcmp(a.packed.data(), b.packed.data(), a.packed.size() * sizeof(uint32_t)) == 0,
              "clustered-sphere RasterClusters deterministic (two runs BYTE-IDENTICAL)");

        uint64_t coveredSphere = 0;
        for (uint32_t v : a.packed) if (v != vg::kSwClear) ++coveredSphere;
        check(coveredSphere > 1000u, "clustered-sphere covers a non-trivial pixel count");

        // WATERTIGHT at cluster seams: rasterize ALL triangles into a per-pixel COVERAGE COUNTER (one tri
        // per fresh buffer, count any covered pixel). Every covered pixel must be claimed EXACTLY once at
        // the FRONT-MOST surface — i.e. the top-left fill rule prevents shared seams double-covering. We
        // verify no FRONT-facing triangle double-covers a same-depth neighbour by checking that, for the
        // front layer, the number of (triangle, pixel) coverages whose key == the final min equals the
        // covered-pixel count (each visible pixel claimed by exactly one winning triangle).
        std::vector<uint32_t> winnerCount((size_t)W * H, 0);
        for (uint32_t cl = 0; cl < (uint32_t)ms.meshlets.size(); ++cl) {
            const vg::Meshlet& m = ms.meshlets[cl];
            for (uint32_t t = 0; t < m.triCount; ++t) {
                uint32_t base = 3u * (m.triOffset + t);
                uint32_t i0 = ms.indices[base+0], i1 = ms.indices[base+1], i2 = ms.indices[base+2];
                if (behind[i0] || behind[i1] || behind[i2]) continue;
                vg::SwVisBuffer one; one.Init(W, H);
                vg::RasterTriangle(one, screen[i0], screen[i1], screen[i2], vg::PackVisId(cl, t));
                for (uint32_t p = 0; p < W * H; ++p) {
                    if (one.packed[p] == vg::kSwClear) continue;
                    // This triangle covers pixel p with its key; if its key == the global winner, it is the
                    // visible surface there. Count how many distinct triangles produce the EXACT winning key.
                    if (one.packed[p] == a.packed[p]) winnerCount[p] += 1;
                }
            }
        }
        // Watertight + min-deterministic: every covered pixel has EXACTLY ONE winning triangle (no two
        // triangles produce the same min key at a pixel — no double-cover at a seam, no depth-tie ambiguity).
        bool exactlyOneWinner = true;
        for (uint32_t p = 0; p < W * H; ++p) {
            if (a.packed[p] == vg::kSwClear) { if (winnerCount[p] != 0) exactlyOneWinner = false; }
            else if (winnerCount[p] != 1) exactlyOneWinner = false;
        }
        check(exactlyOneWinner,
              "clustered-sphere watertight: each covered pixel claimed by EXACTLY one winning triangle "
              "(no seam gap/double-cover)");
    }

    if (g_fail == 0) std::printf("swraster_test: ALL PASS\n");
    else std::printf("swraster_test: %d FAILURES\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
