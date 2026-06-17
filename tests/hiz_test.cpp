// Slice CJ — Hi-Z occlusion culling. Pure-CPU test of engine/render/hiz.h (namespace hf::render::hiz)
// that the GPU cull compute (shaders/hiz_cull.comp.hlsl) mirrors: a depth pre-pass feeds a Hi-Z
// max-depth pyramid; the cull pass tests each object's screen-space rect's NEAREST depth against the
// Hi-Z's FARTHEST depth there and drops the object iff it is FULLY behind everything already drawn.
//
// Pins, WITHOUT a GPU, the contracts the render-invariance proof rests on:
//   * Hi-Z build: each coarser mip texel == the MAX of its 2x2 children; dims halve; a known buffer
//     produces the expected pyramid.
//   * Occlusion TRUE POSITIVE: an object fully BEHIND a closer full-screen occluder is IsOccluded.
//   * Occlusion TRUE NEGATIVES (never false-cull): in-front, partially-visible, near-plane-straddle,
//     and partly-off-screen objects are NOT occluded (the safety cases — false-culling corrupts).
//   * CONSERVATIVE parity: over random AABBs + a known Hi-Z, IsOccluded NEVER reports occluded when a
//     brute-force per-covered-texel check finds ANY texel with the AABB's nearest depth in FRONT of
//     the Hi-Z (0 false-culls).
//   * Determinism: same inputs -> same build + same result.
//
// Pure C++ (hf_core), ASan-eligible like the other render-data tests.
#include "render/hiz.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf::math;
namespace hz = hf::render::hiz;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();
    // =====================================================================================
    // 1) Hi-Z BUILD: a known 4x4 depth buffer -> the expected 4x4/2x2/1x1 pyramid, each coarse texel
    //    the MAX of its 2x2 children, dims halving to the 1x1 top.
    // =====================================================================================
    {
        // 4x4 depth (row-major). Distinct values so the MAX is unambiguous.
        const int W = 4, H = 4;
        float d[16] = {
            0.10f, 0.20f, 0.30f, 0.40f,
            0.50f, 0.60f, 0.70f, 0.80f,
            0.15f, 0.25f, 0.35f, 0.45f,
            0.55f, 0.65f, 0.75f, 0.85f,
        };
        std::vector<hz::HiZMip> mips;
        hz::BuildHiZ(d, W, H, mips);

        check(mips.size() == 3, "4x4 depth -> 3 mips (4x4, 2x2, 1x1)");
        check(mips[0].width == 4 && mips[0].height == 4, "mip0 is 4x4");
        check(mips[1].width == 2 && mips[1].height == 2, "mip1 is 2x2 (dims halved)");
        check(mips[2].width == 1 && mips[2].height == 1, "mip2 is 1x1 (top)");

        // mip0 == the depth buffer verbatim.
        bool m0ok = true;
        for (int i = 0; i < 16; ++i) if (mips[0].depth[i] != d[i]) m0ok = false;
        check(m0ok, "mip0 == depth buffer verbatim");

        // mip1 texels = MAX of the 2x2 blocks of mip0.
        // block (0,0): max(0.10,0.20,0.50,0.60)=0.60; (1,0): max(0.30,0.40,0.70,0.80)=0.80
        // block (0,1): max(0.15,0.25,0.55,0.65)=0.65; (1,1): max(0.35,0.45,0.75,0.85)=0.85
        check(mips[1].At(0, 0) == 0.60f, "mip1(0,0) == MAX of its 2x2 children (0.60)");
        check(mips[1].At(1, 0) == 0.80f, "mip1(1,0) == MAX of its 2x2 children (0.80)");
        check(mips[1].At(0, 1) == 0.65f, "mip1(0,1) == MAX of its 2x2 children (0.65)");
        check(mips[1].At(1, 1) == 0.85f, "mip1(1,1) == MAX of its 2x2 children (0.85)");
        // mip2 = MAX of mip1 = 0.85 (== global farthest).
        check(mips[2].At(0, 0) == 0.85f, "mip2(0,0) == global MAX (farthest) 0.85");

        // Each coarse texel is the MAX of its children — verify generically too.
        for (size_t L = 1; L < mips.size(); ++L) {
            const hz::HiZMip& fine = mips[L - 1];
            const hz::HiZMip& coarse = mips[L];
            bool ok = true;
            for (int y = 0; y < coarse.height; ++y)
                for (int x = 0; x < coarse.width; ++x) {
                    int fx0 = x * 2, fy0 = y * 2;
                    float mx = fine.At(fx0, fy0);
                    if (fx0 + 1 < fine.width)  mx = std::max(mx, fine.At(fx0 + 1, fy0));
                    if (fy0 + 1 < fine.height) mx = std::max(mx, fine.At(fx0, fy0 + 1));
                    if (fx0 + 1 < fine.width && fy0 + 1 < fine.height) mx = std::max(mx, fine.At(fx0 + 1, fy0 + 1));
                    if (coarse.At(x, y) != mx) ok = false;
                }
            check(ok, "every coarse texel == MAX of its 2x2 children (generic)");
        }

        // Odd dimensions halve via ceil and the edge texels still MAX their in-range children.
        {
            const int OW = 3, OH = 3;
            float od[9] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
            std::vector<hz::HiZMip> om;
            hz::BuildHiZ(od, OW, OH, om);
            check(om.size() == 3, "3x3 -> 3 mips (3x3, 2x2, 1x1)");
            check(om[1].width == 2 && om[1].height == 2, "3x3 mip1 is 2x2 (ceil)");
            check(om[2].width == 1 && om[2].height == 1, "3x3 mip2 is 1x1 (top)");
            // block(1,1) of a 3x3 only has the single child (2,2)=0.9.
            check(om[1].At(1, 1) == 0.9f, "odd-dim edge texel == MAX of its single in-range child");
        }

        // DETERMINISM: a second build is bit-identical.
        std::vector<hz::HiZMip> mips2;
        hz::BuildHiZ(d, W, H, mips2);
        bool same = mips.size() == mips2.size();
        for (size_t L = 0; same && L < mips.size(); ++L)
            same = (mips[L].width == mips2[L].width) && (mips[L].height == mips2[L].height) &&
                   (mips[L].depth == mips2[L].depth);
        check(same, "two Hi-Z builds are bit-identical (deterministic)");
    }

    // =====================================================================================
    // 2) OCCLUSION TEST. A real camera + a FULL-SCREEN near occluder baked into the Hi-Z as a uniform
    //    depth wall: every texel == zWall. An object whose projected NEAREST depth > zWall everywhere
    //    it covers is occluded; nearer / partial / straddling / off-screen objects are kept.
    // =====================================================================================
    const int   SW = 256, SH = 256;
    const float aspect = (float)SW / (float)SH;
    Mat4 view = Mat4::LookAt(Vec3{0, 0, 6}, Vec3{0, 0, 0}, Vec3{0, 1, 0});
    Mat4 proj = Mat4::Perspective(0.9f, aspect, 0.5f, 100.0f);
    Mat4 vp   = proj * view;

    // A full-screen occluder wall at z=2 (world): compute its NDC depth and bake a UNIFORM Hi-Z so
    // every screen texel's farthest depth == the wall. Anything BEHIND the wall (smaller world z ->
    // larger NDC z) and on-screen is fully occluded.
    auto ndcZOfWorldZ = [&](float wz) {
        float w = 0.0f;
        Vec3 ndc = MulPointDivide(vp, Vec3{0, 0, wz}, w);
        return ndc.z;
    };
    const float zWallNdc = ndcZOfWorldZ(2.0f);   // the wall plane's NDC depth
    std::vector<float> depth((size_t)SW * SH, zWallNdc);
    std::vector<hz::HiZMip> mips;
    hz::BuildHiZ(depth.data(), SW, SH, mips);
    std::span<const hz::HiZMip> mipSpan(mips.data(), mips.size());

    // A small AABB centered on the view axis at world z = `cz`, half-size `hs`. At cz < 2 it is
    // BEHIND the wall (farther; larger NDC z) -> occluded; at cz > 2 it is in FRONT -> visible.
    auto boxAt = [&](float cx, float cy, float cz, float hs, Vec3& mn, Vec3& mx) {
        mn = Vec3{cx - hs, cy - hs, cz - hs};
        mx = Vec3{cx + hs, cy + hs, cz + hs};
    };

    // 2a) TRUE POSITIVE: a small box well behind the wall, centered, fully on-screen -> occluded.
    {
        Vec3 mn, mx; boxAt(0, 0, -2.0f, 0.4f, mn, mx);
        check(hz::IsOccluded(mn, mx, vp, SW, SH, mipSpan),
              "box fully BEHIND a closer full-screen occluder -> IsOccluded (true positive)");
    }

    // 2b) TRUE NEGATIVE — in FRONT of the wall (nearer than the Hi-Z) -> NOT occluded.
    {
        Vec3 mn, mx; boxAt(0, 0, 4.0f, 0.4f, mn, mx);   // world z=4 > 2 -> closer to the camera
        check(!hz::IsOccluded(mn, mx, vp, SW, SH, mipSpan),
              "box IN FRONT of the occluder -> NOT occluded (safety: never false-cull)");
    }

    // 2c) TRUE NEGATIVE — PARTIALLY visible: the box straddles the wall depth (its nearest corner is
    //     in front, its far corner behind) -> nearest depth < Hi-Z somewhere -> NOT occluded.
    {
        // A deep box centered at z=2 spanning z in [1.0, 3.0]: nearest face (z=3) is in front of the
        // wall (world z 3 > 2) so its nearest NDC z < zWall -> kept.
        Vec3 mn{-0.4f, -0.4f, 1.0f}, mx{0.4f, 0.4f, 3.0f};
        check(!hz::IsOccluded(mn, mx, vp, SW, SH, mipSpan),
              "box straddling the occluder depth (nearer somewhere) -> NOT occluded");
    }

    // 2d) TRUE NEGATIVE — near-plane straddle: a box spanning from behind the camera to in front has a
    //     corner with clip w<=0 -> conservative KEEP regardless of depth.
    {
        Vec3 mn{-0.4f, -0.4f, 5.0f}, mx{0.4f, 0.4f, 8.0f};  // camera at z=6 -> box spans across it
        check(!hz::IsOccluded(mn, mx, vp, SW, SH, mipSpan),
              "box straddling the near plane (corner behind camera) -> NOT occluded (conservative)");
    }

    // 2e) TRUE NEGATIVE — partly OFF-SCREEN: a box behind the wall but pushed far to the side so its
    //     screen rect leaves the view bounds -> conservative KEEP (we can't see its whole coverage).
    {
        Vec3 mn, mx; boxAt(-9.0f, 0, -2.0f, 0.6f, mn, mx);  // far left, behind the wall
        check(!hz::IsOccluded(mn, mx, vp, SW, SH, mipSpan),
              "box partly off-screen (behind the wall) -> NOT occluded (conservative)");
    }

    // 2f) DETERMINISM of the test: same inputs -> same answer.
    {
        Vec3 mn, mx; boxAt(0, 0, -2.0f, 0.4f, mn, mx);
        bool a = hz::IsOccluded(mn, mx, vp, SW, SH, mipSpan);
        bool b = hz::IsOccluded(mn, mx, vp, SW, SH, mipSpan);
        check(a == b, "IsOccluded is deterministic (same inputs -> same result)");
    }

    // 2g) An EMPTY pyramid / zero screen never culls.
    {
        Vec3 mn, mx; boxAt(0, 0, -2.0f, 0.4f, mn, mx);
        std::span<const hz::HiZMip> empty;
        check(!hz::IsOccluded(mn, mx, vp, SW, SH, empty), "empty Hi-Z -> never occluded (keep)");
    }

    // =====================================================================================
    // 3) CONSERVATIVE PARITY (the no-false-cull contract): over random AABBs + a random (non-uniform)
    //    Hi-Z, IsOccluded must NEVER report occluded when a BRUTE-FORCE per-covered-texel check finds
    //    ANY mip0 texel in the object's screen rect whose Hi-Z (mip0) depth is >= the object's nearest
    //    depth (i.e. something there is at or in front of the object's nearest point). 0 false-culls.
    // =====================================================================================
    {
        std::mt19937 rng(0x7110FFEEu);
        // A random but structured depth field (a tilted ramp + bumps) so the Hi-Z is non-uniform.
        std::vector<float> rd((size_t)SW * SH);
        std::uniform_real_distribution<float> jitter(-0.03f, 0.03f);
        for (int y = 0; y < SH; ++y)
            for (int x = 0; x < SW; ++x) {
                float base = 0.2f + 0.5f * ((float)x / SW) + 0.2f * ((float)y / SH);
                float v = base + jitter(rng);
                if (v < 0.0f) v = 0.0f; if (v > 1.0f) v = 1.0f;
                rd[(size_t)y * SW + x] = v;
            }
        std::vector<hz::HiZMip> rmips;
        hz::BuildHiZ(rd.data(), SW, SH, rmips);
        std::span<const hz::HiZMip> rspan(rmips.data(), rmips.size());

        std::uniform_real_distribution<float> cx(-3.0f, 3.0f), cy(-3.0f, 3.0f), cz(-6.0f, 5.0f);
        std::uniform_real_distribution<float> hs(0.1f, 1.2f);
        int trials = 0, occluded = 0, falseCulls = 0;
        for (int t = 0; t < 4000; ++t) {
            Vec3 mn{cx(rng), cy(rng), cz(rng)};
            float s = hs(rng);
            Vec3 mx{mn.x + s, mn.y + s, mn.z + s};
            bool culled = hz::IsOccluded(mn, mx, vp, SW, SH, rspan);
            ++trials;
            if (!culled) continue;
            ++occluded;

            // BRUTE FORCE: recompute the screen rect + nearest depth exactly, then scan EVERY mip0
            // texel in the rect. If any has depth >= nearest (something at/in front of our nearest
            // point), the object is NOT fully hidden there -> IsOccluded must NOT have culled it.
            float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f, nearestZ = 1e30f;
            bool behindNear = false, offscreen = false;
            for (int c = 0; c < 8; ++c) {
                Vec3 corner{(c & 1) ? mx.x : mn.x, (c & 2) ? mx.y : mn.y, (c & 4) ? mx.z : mn.z};
                float w = 0.0f;
                Vec3 ndc = MulPointDivide(vp, corner, w);
                if (w <= 1e-6f) { behindNear = true; break; }
                float px = (ndc.x * 0.5f + 0.5f) * SW;
                float py = (ndc.y * 0.5f + 0.5f) * SH;
                minX = std::min(minX, px); maxX = std::max(maxX, px);
                minY = std::min(minY, py); maxY = std::max(maxY, py);
                nearestZ = std::min(nearestZ, ndc.z);
            }
            if (behindNear) { ++falseCulls; continue; }       // should have been kept
            if (minX < 0 || minY < 0 || maxX > SW || maxY > SH) { offscreen = true; }
            if (offscreen) { ++falseCulls; continue; }        // should have been kept
            int x0 = std::max(0, (int)std::floor(minX)), x1 = std::min(SW - 1, (int)std::floor(maxX));
            int y0 = std::max(0, (int)std::floor(minY)), y1 = std::min(SH - 1, (int)std::floor(maxY));
            bool anyInFront = false;
            for (int y = y0; y <= y1 && !anyInFront; ++y)
                for (int x = x0; x <= x1; ++x)
                    if (rd[(size_t)y * SW + x] >= nearestZ) { anyInFront = true; break; }
            if (anyInFront) ++falseCulls;  // a covered texel is at/in front -> culling it is a FALSE cull
        }
        check(falseCulls == 0, "CONSERVATIVE: 0 false-culls over random AABBs (brute-force per-texel parity)");
        check(occluded > 0, "the random parity actually exercised some real occlusions (occluded>0)");
        std::printf("hiz parity: %d trials, %d occluded, %d false-culls\n", trials, occluded, falseCulls);
    }

    if (g_fail == 0) { std::printf("hiz_test: all checks passed\n"); return 0; }
    std::printf("hiz_test: %d FAILURES\n", g_fail);
    return 1;
}
