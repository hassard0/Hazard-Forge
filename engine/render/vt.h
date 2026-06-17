#pragma once
// Slice VT1 — Runtime Virtual Texturing Slice 1: VT page table + page-needed FEEDBACK/MARKING
// (BEACHHEAD of FLAGSHIP #4: UE5's literal Runtime Virtual Texturing). Pure CPU (header-only, no
// device, no backend symbols). Namespace hf::render::vt. The STRUCTURAL TWIN of VSM Slice VA
// (engine/render/vsm.h): a virtual page table -> a feedback/marking pass -> (later) physical tile
// allocation + indirection -> a sampled lookup -> per-page caching. This slice is the marking pass,
// the direct analog of vsm.h's MarkResidentPages, applied to TEXTURE pages instead of shadow pages.
//
// The marking shader shaders/vt_feedback.comp.hlsl copies PageId VERBATIM from this header (and
// consumes the HOST-SNAPPED integer (mip,px,py) page coords this header's VtPageId produces), so
// tests/vt_test.cpp exercises the EXACT math the GPU feedback pass runs — which is what makes the
// feedback page set bit-identical GPU==CPU AND cross-backend.
//
// THE TECHNIQUE (Runtime Virtual Texturing = Nanite-scale texturing): a virtual texture far larger
// than VRAM, sampled through a virtual->physical page indirection driven by per-pixel FEEDBACK. The
// virtual texture is a MIP PYRAMID: mip 0 is virtualPagesPerSideMip0² pages of pageSize texels, each
// finer mip halving the pages-per-side (max 1). This slice builds the virtual PAGE TABLE + MARKS
// which virtual texture pages a set of sample-requests NEED, as a pure INTEGER compute pass (no
// rendering, no new RHI). The marking output is a pure integer SET (feedback[pageId] in {0,1}) ->
// inherently bit-exact + cross-backend, proven GPU==CPU via ReadBuffer memcmp.
//
// THE DETERMINISM CRUX (the make-or-break for GPU==CPU, like vsm.h's threshold-ladder / swraster.h's
// host-snapped ScreenVerts): the UV->page quantization (floor(u*pagesPerSide(mip))) is the one
// float->int boundary. We pin it identically on both sides (plain multiply, NOT fma), so floor(u*pps)
// is bit-identical CPU<->Vulkan<->Metal for the host-supplied exact float32 UVs and the small
// power-of-two pps. For TOTAL cross-backend safety we go further (the swraster.h host-snap pattern):
// the showcase HOST-SNAPS each SampleRequest to its integer (mip,px,py) page coords and uploads THOSE
// to the GPU, so the GPU does ZERO floating point — it just calls PageId on integers, making GPU==CPU
// trivially exact. VtPageId(u,v,mip) stays here as the CPU reference + the unit-test oracle.
//
// The mip-level SELECTION (SelectMipLevel) is the integer THRESHOLD-LADDER copied from vsm.h's
// SelectClipmapLevel: count host-precomputed thresholds a texel-density value exceeds, clamp. NO log2,
// NO transcendental on the bit-exact path. For THIS beachhead the showcase supplies `mip` per request
// directly (the simplest deterministic form); SelectMipLevel is exercised + unit-tested as the general
// path so slice VT4 (the material-pass sample) can use it.
//
// CONVENTIONS:
//   * Flat page-table index: PageId(mip,px,py) = mipPageOffset(mip) + py*pagesPerSide(mip) + px. Unlike
//     VSM (a CONSTANT vpps per clipmap level, so offset = level*vpps²), VT mips have DIFFERENT
//     pagesPerSide, so the per-mip offset is a host PREFIX-SUM of pagesPerSide(k)² for k<mip — NOT
//     mip*vpps². UnpackPageId is its exact inverse — a bijection over [0, pageCount()) (unit-tested).
//   * UV->page: px = clamp(floor(u*pagesPerSide(mip)), 0, pps-1); plain subtract/multiply/floor.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace hf::render::vt {

// Hard cap on mip levels (sizes the on-stack threshold table; the shader uses the same fixed cap).
inline constexpr int kMaxMips = 16;

// The virtual texture page table: a MIP pyramid. Mip 0 is virtualPagesPerSideMip0² pages of pageSize
// texels each; each finer mip halves the pages-per-side (floored at 1). (e.g. mipLevels=4,
// pageSize=128, virtualPagesPerSideMip0=16 -> a 2048²-texel mip-0 virtual texture, pages per side
// 16/8/4/2 over 4 mips, 256+64+16+4 = 340 pages total.)
struct VtTexture {
    int mipLevels = 4;
    int pageSize = 128;
    int virtualPagesPerSideMip0 = 16;

    // Pages per side at `mip`: virtualPagesPerSideMip0 >> mip, floored at 1.
    int pagesPerSide(int mip) const {
        int p = virtualPagesPerSideMip0 >> mip;
        return p < 1 ? 1 : p;
    }

    // The host PREFIX-SUM page offset of `mip`: sum of pagesPerSide(k)² for k < mip. (NOT mip*vpps² —
    // mips have DIFFERENT pagesPerSide. This is the crux that distinguishes VT from VSM's flat layout.)
    int mipPageOffset(int mip) const {
        int off = 0;
        for (int k = 0; k < mip; ++k) {
            int pk = pagesPerSide(k);
            off += pk * pk;
        }
        return off;
    }

    // Total virtual page-table slots across all mips (the feedback SSBO length).
    int pageCount() const { return mipPageOffset(mipLevels); }
};

// Flat page-table index: mip-major (via the prefix-sum offset), then py-major, then px. MIRRORED in
// vt_feedback.comp. idx = mipPageOffset(mip) + py*pagesPerSide(mip) + px.
inline int PageId(int mip, int px, int py, const VtTexture& vt) {
    const int pps = vt.pagesPerSide(mip);
    return vt.mipPageOffset(mip) + py * pps + px;
}

// Inverse of PageId (a bijection over [0, pageCount())). Unit-tested PageId(UnpackPageId(id))==id.
// Walk the mips, subtracting each mip's page span, until `id` lands inside one.
inline void UnpackPageId(int id, const VtTexture& vt, int& mip, int& px, int& py) {
    int rem = id;
    int m = 0;
    for (; m < vt.mipLevels; ++m) {
        const int pps = vt.pagesPerSide(m);
        const int span = pps * pps;
        if (rem < span) break;
        rem -= span;
    }
    if (m >= vt.mipLevels) m = vt.mipLevels - 1;  // clamp a degenerate out-of-range id to the top mip
    const int pps = vt.pagesPerSide(m);
    mip = m;
    py = rem / pps;
    px = rem % pps;
}

// Host-precompute the mip threshold table: thresholds[m] = 2^m (the texel-density doubling per mip).
// The showcase uploads these exact float32 bits; the CPU + the shader read the SAME bits and count how
// many a density value exceeds. 2^m is an exact power-of-two -> bit-identical to the shader's
// (float)(1u<<m). We expose the builder so the showcase + the test agree. (For THIS beachhead the
// showcase supplies mip per request directly; SelectMipLevel is exercised + unit-tested as the general
// path so slice VT4's material-pass sample can use it.)
inline void BuildMipThresholds(const VtTexture& vt, float* thresholds /*[vt.mipLevels]*/) {
    for (int m = 0; m < vt.mipLevels; ++m)
        thresholds[m] = (float)(1u << (uint32_t)m);
}

// DETERMINISM CRUX — select the mip level by an INTEGER threshold-ladder (NO log2). The level is the
// number of thresholds `density` EXCEEDS, clamped to [0, mipLevels-1]. thresholds[m] = 2^m
// (BuildMipThresholds). density <= 1 -> mip 0 (the finest); each doubling of density climbs one mip;
// large -> the coarsest mip. Pure compare/count -> bit-identical CPU<->GPU. (Equivalent to
// clamp(floor(log2(density)), 0, mipLevels-1) but WITHOUT the transcendental.) Copied from
// vsm.h::SelectClipmapLevel.
inline int SelectMipLevel(float density, const VtTexture& vt) {
    float thresholds[kMaxMips];
    const int mips = vt.mipLevels;
    BuildMipThresholds(vt, thresholds);
    int mip = 0;
    for (int m = 0; m < mips; ++m)
        if (density > thresholds[m]) mip = m + 1;
    if (mip < 0) mip = 0;
    if (mip > mips - 1) mip = mips - 1;
    return mip;
}

// A virtual-texture sample-request: a UV in [0,1] at a given mip. (The showcase host-snaps each to its
// integer (mip,px,py) page coords before uploading to the GPU — see SnappedRequest.)
struct SampleRequest {
    float u = 0.0f;
    float v = 0.0f;
    int   mip = 0;
};

// THE UV->PAGE QUANTIZER. px = clamp(floor(u*pagesPerSide(mip)), 0, pps-1); same for py; return
// PageId(mip,px,py). Plain subtract/multiply/floor (NOT fma — pin the identical op vs the shader). The
// `u*pps` float multiply is immediately floored to an integer; with host-supplied exact float32 UVs +
// a small power-of-two pps, floor(u*pps) is bit-identical CPU<->Vulkan<->Metal. This is the CPU
// REFERENCE + the unit-test oracle; the showcase host-snaps via this and uploads the resulting integer
// (mip,px,py) so the GPU does ZERO float (the swraster.h host-snap discipline). MIRRORED in
// vt_feedback.comp (the GPU consumes the snapped integers + calls PageId).
inline int VtPageId(float u, float v, int mip, const VtTexture& vt) {
    const int pps = vt.pagesPerSide(mip);
    int px = (int)std::floor(u * (float)pps);
    int py = (int)std::floor(v * (float)pps);
    if (px < 0) px = 0; else if (px > pps - 1) px = pps - 1;
    if (py < 0) py = 0; else if (py > pps - 1) py = pps - 1;
    return PageId(mip, px, py, vt);
}

// The host-snapped integer page coords for a SampleRequest (what the showcase uploads to the GPU so the
// GPU does ZERO floating point). SnapRequest runs VtPageId's quantization on the host.
struct SnappedRequest {
    int mip = 0;
    int px = 0;
    int py = 0;
};

inline SnappedRequest SnapRequest(const SampleRequest& req, const VtTexture& vt) {
    const int pps = vt.pagesPerSide(req.mip);
    int px = (int)std::floor(req.u * (float)pps);
    int py = (int)std::floor(req.v * (float)pps);
    if (px < 0) px = 0; else if (px > pps - 1) px = pps - 1;
    if (py < 0) py = 0; else if (py > pps - 1) py = pps - 1;
    return SnappedRequest{req.mip, px, py};
}

// CPU REFERENCE marking: zero feedbackOut (sized pageCount()), then for each request VtPageId ->
// feedbackOut[pageId] = 1. This is the EXACT integer set the GPU vt_feedback.comp matches byte-for-byte
// (the set is order-independent — writes race-free to 1). feedbackEnabled=false is modeled by the
// caller passing an empty request span (or skipping this call), yielding the cleared all-zero set.
inline void MarkFeedbackPages(std::span<const SampleRequest> requests, const VtTexture& vt,
                              std::span<uint32_t> feedbackOut) {
    for (uint32_t& f : feedbackOut) f = 0u;
    for (const SampleRequest& r : requests) {
        int pageId = VtPageId(r.u, r.v, r.mip, vt);
        if (pageId >= 0 && pageId < (int)feedbackOut.size())
            feedbackOut[(size_t)pageId] = 1u;
    }
}

}  // namespace hf::render::vt
