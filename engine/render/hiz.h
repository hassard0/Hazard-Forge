#pragma once
// Slice CJ — Hi-Z (hierarchical-Z) occlusion culling math. Pure CPU (header-only, no device, no
// backend symbols). Same pattern as engine/render/frustum.h / gpu_culled.h. Namespace
// hf::render::hiz. Shared by the --hiz-cull-shot showcase, the GPU cull compute (the shader runs the
// SAME IsOccluded math sampling the uploaded Hi-Z mips) AND tests/hiz_test.cpp — so the unit test
// exercises the EXACT build + occlusion test the render path uses.
//
// THE OCCLUSION CULL is the AAA-completion of the GPU-driven cull pipeline (frustum from CD + the
// OCCLUSION test here): an object is dropped ONLY when it is FULLY HIDDEN behind closer geometry, so
// the occlusion-culled image is BYTE-IDENTICAL to a frustum-only render (the culled objects
// contribute zero pixels). The proof lives in --hiz-cull-shot. That invariant rests entirely on the
// CONSERVATIVE contract below — a single false-cull (dropping a visible object) corrupts the image.
//
// DEPTH CONVENTION (the crux — get this wrong and you cull the wrong half):
//   We use the engine's clip-space depth: Vulkan [0,1] reverse-FREE depth (Mat4::Perspective +
//   render::frustum). After the perspective divide, NDC z in [0,1] with z=0 at the NEAR plane and
//   z=1 at the FAR plane, so LARGER z == FARTHER from the camera. The Hi-Z therefore stores this
//   post-divide NDC z, and a coarse mip texel = the MAX (largest z == FARTHEST surface) of its 2x2
//   children. "MAX = farthest" means: if your object's NEAREST (smallest) z is STILL LARGER than the
//   coarse texel's MAX, then EVERY surface already drawn in that whole block is in front of your
//   object's nearest point -> the block fully covers you -> you are occluded. (Metal's clip is the
//   same single Mat4::Perspective [0,1] depth; the showcase composes the per-backend matrix, so the
//   projected z carries that backend's convention and the test is convention-correct on both.)
//
// CONSERVATIVE CONTRACT (never false-cull):
//   * The Hi-Z mip is the FARTHEST depth in each block, so testing the object's NEAREST depth against
//     it is the safe direction: occluded iff nearest > Hi-Z-max across ALL covered texels.
//   * Any uncertainty keeps the object (returns NOT occluded): the screen rect spans more texels than
//     the chosen mip covers, the AABB straddles the near plane (any corner with clip w <= 0), the
//     rect is partially (or fully) off-screen, or the rect is degenerate. We would rather keep an
//     occluded object (a missed optimization) than drop a visible one (a corrupted image).

#include "math/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace hf::render::hiz {

// One mip level of the Hi-Z pyramid: a width x height grid of MAX-reduced depths (row-major, top row
// first — same layout as a tightly-packed depth image). mip 0 is the source depth buffer verbatim;
// each coarser level halves the dimensions (rounding up) and each texel is the MAX of its up-to-2x2
// children in the finer level.
struct HiZMip {
    int                width  = 0;
    int                height = 0;
    std::vector<float> depth;  // size == width*height, row-major

    float At(int x, int y) const { return depth[(size_t)y * (size_t)width + (size_t)x]; }
};

// Build the Hi-Z mip pyramid from a `w`x`h` depth buffer (`depth` is row-major, w*h floats, the
// post-divide NDC z described in the header: 0 near .. 1 far, LARGER == farther). `mips[0]` is the
// depth buffer copied verbatim; each subsequent level halves the dimensions (ceil) and sets each
// texel to the MAX (farthest) of its up-to-2x2 children. Stops at the 1x1 top. Deterministic (a pure
// max-reduction; no RNG/clock). Clears `mips` first.
inline void BuildHiZ(const float* depth, int w, int h, std::vector<HiZMip>& mips) {
    mips.clear();
    if (w <= 0 || h <= 0 || depth == nullptr) return;

    // mip 0 == the depth buffer verbatim.
    HiZMip m0;
    m0.width = w;
    m0.height = h;
    m0.depth.assign(depth, depth + (size_t)w * (size_t)h);
    mips.push_back(std::move(m0));

    // Coarser levels: dims halve (ceil), each texel = MAX of its 2x2 children (clamped at the edges
    // so an odd dimension's last texel just sees its single in-range child/children — still a MAX, so
    // still conservative).
    while (mips.back().width > 1 || mips.back().height > 1) {
        const HiZMip& fine = mips.back();
        HiZMip coarse;
        coarse.width  = std::max(1, (fine.width  + 1) / 2);
        coarse.height = std::max(1, (fine.height + 1) / 2);
        coarse.depth.resize((size_t)coarse.width * (size_t)coarse.height);
        for (int y = 0; y < coarse.height; ++y) {
            for (int x = 0; x < coarse.width; ++x) {
                int fx0 = x * 2, fy0 = y * 2;
                float mx = fine.At(fx0, fy0);
                if (fx0 + 1 < fine.width)                         mx = std::max(mx, fine.At(fx0 + 1, fy0));
                if (fy0 + 1 < fine.height)                        mx = std::max(mx, fine.At(fx0, fy0 + 1));
                if (fx0 + 1 < fine.width && fy0 + 1 < fine.height) mx = std::max(mx, fine.At(fx0 + 1, fy0 + 1));
                coarse.depth[(size_t)y * (size_t)coarse.width + (size_t)x] = mx;
            }
        }
        mips.push_back(std::move(coarse));
    }
}

// The conservative occlusion test. Projects the world AABB's 8 corners through `viewProj` (Vulkan
// clip), forms the screen-space pixel rect they cover + the object's NEAREST clip depth, selects the
// coarsest mip whose texel covers the rect in ~1-4 texels, and returns true (OCCLUDED) IFF the
// object's nearest depth is FARTHER (larger NDC z) than the Hi-Z MAX across every covered texel.
//
// Returns false (KEEP) on ANY uncertainty (the conservative fallbacks):
//   * any corner has clip w <= 0 (the AABB straddles/behind the near plane — projection unreliable),
//   * the screen rect touches or leaves the screen bounds (partially off-screen),
//   * the rect is degenerate / the pyramid is empty.
// `screenW`/`screenH` are the depth-buffer (mip 0) dimensions in pixels.
inline bool IsOccluded(const math::Vec3& aabbMin, const math::Vec3& aabbMax,
                       const math::Mat4& viewProj, int screenW, int screenH,
                       std::span<const HiZMip> mips) {
    if (mips.empty() || screenW <= 0 || screenH <= 0) return false;  // nothing to test against -> keep

    // Project the 8 AABB corners. Track the screen-pixel bounds + the nearest (smallest) NDC z.
    float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
    float nearestZ = 1e30f;
    for (int c = 0; c < 8; ++c) {
        math::Vec3 corner{
            (c & 1) ? aabbMax.x : aabbMin.x,
            (c & 2) ? aabbMax.y : aabbMin.y,
            (c & 4) ? aabbMax.z : aabbMin.z,
        };
        float w = 0.0f;
        math::Vec3 ndc = math::MulPointDivide(viewProj, corner, w);
        if (w <= 1e-6f) return false;  // near-plane straddle / behind camera -> conservative KEEP

        // NDC x,y in [-1,1] -> pixel coords. (The Y direction doesn't matter for a max-rect/min-z
        // test that is symmetric in the pixel grid, but we map consistently with a V-down image.)
        float px = (ndc.x * 0.5f + 0.5f) * (float)screenW;
        float py = (ndc.y * 0.5f + 0.5f) * (float)screenH;
        minX = std::min(minX, px); maxX = std::max(maxX, px);
        minY = std::min(minY, py); maxY = std::max(maxY, py);
        nearestZ = std::min(nearestZ, ndc.z);  // smallest NDC z == nearest surface of the AABB
    }

    // Off-screen (or touching the edge) -> conservative KEEP (we cannot see the whole coverage). A
    // strict inside-the-bounds requirement keeps any partially-clipped object.
    if (minX < 0.0f || minY < 0.0f || maxX > (float)screenW || maxY > (float)screenH) return false;
    if (!(maxX > minX) || !(maxY > minY)) return false;  // degenerate rect -> keep
    if (nearestZ < 0.0f || nearestZ > 1.0f) return false;  // outside the depth range -> keep

    // Integer pixel rect [x0,x1] x [y0,y1] inclusive on mip 0.
    int x0 = (int)std::floor(minX), x1 = (int)std::floor(maxX);
    int y0 = (int)std::floor(minY), y1 = (int)std::floor(maxY);
    x0 = std::max(0, x0); y0 = std::max(0, y0);
    x1 = std::min(screenW - 1, x1); y1 = std::min(screenH - 1, y1);
    if (x1 < x0 || y1 < y0) return false;

    // MIP SELECTION: pick the coarsest mip on which the rect spans at most 2 texels in each axis (so
    // the test reads ~1-4 texels). The rect width in mip-L texels is (x1>>L) - (x0>>L) + 1; choose the
    // smallest L making both axes <= 2. This guarantees a bounded, conservative read (we test EVERY
    // covered texel of that mip; the MAX-pyramid means a coarse texel already folds its block's
    // farthest depth, so reading few coarse texels stays conservative).
    int level = 0;
    const int maxLevel = (int)mips.size() - 1;
    while (level < maxLevel) {
        int tx0 = x0 >> (level + 1), tx1 = x1 >> (level + 1);
        int ty0 = y0 >> (level + 1), ty1 = y1 >> (level + 1);
        if ((tx1 - tx0) <= 1 && (ty1 - ty0) <= 1) { ++level; break; }
        ++level;
    }
    // Clamp to a level where the span is small; if even the coarsest leaves a big span (shouldn't,
    // the top is 1x1) we still test every covered texel below.
    const HiZMip& mip = mips[(size_t)level];
    int mx0 = x0 >> level, mx1 = x1 >> level;
    int my0 = y0 >> level, my1 = y1 >> level;
    mx0 = std::max(0, mx0); my0 = std::max(0, my0);
    mx1 = std::min(mip.width - 1, mx1); my1 = std::min(mip.height - 1, my1);

    // OCCLUDED iff the object's nearest depth is FARTHER (>) than the Hi-Z MAX across EVERY covered
    // texel. The Hi-Z texel is the FARTHEST surface in that block; if our nearest point is still
    // behind even the farthest already-drawn surface there, the block fully hides us. A single texel
    // whose Hi-Z max is >= our nearest means something there is at or behind us -> NOT occluded.
    for (int ty = my0; ty <= my1; ++ty) {
        for (int tx = mx0; tx <= mx1; ++tx) {
            float hiZMax = mip.At(tx, ty);
            if (nearestZ <= hiZMax) return false;  // not strictly in front everywhere -> KEEP
        }
    }
    return true;  // nearest is strictly nearer-than-nothing: farther than every covered Hi-Z max
}

}  // namespace hf::render::hiz
