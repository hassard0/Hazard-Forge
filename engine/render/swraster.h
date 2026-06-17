#pragma once
// Slice SW1 — Nanite Software-Raster Slice 1: CPU-Reference Rasterizer + Integer Edge Math (the
// BEACHHEAD of the software-rasterization flagship). Pure CPU (header-only, no device, no backend
// symbols). Mirrors engine/render/visbuffer.h — the SAME (clusterID, triID) identity packing — but
// where DW rasterizes the visibility IDs on the GPU's HARDWARE rasterizer, this header is a
// deterministic INTEGER/FIXED-POINT SOFTWARE rasterizer that scan-converts cluster triangles into a
// packed `depth|id` visibility buffer entirely on the CPU. Namespace hf::render::vg (the same
// virtual-geometry namespace as meshlet.h / visbuffer.h / visresolve.h).
//
// WHAT THIS IS: the GPU compute software-rasterizer (SW2 — swraster.comp.hlsl with InterlockedMin into
// a vis-buffer SSBO) will copy this math VERBATIM and prove bit-identical to it. So this is the
// reference: clean, deterministic, integer. The cross-backend crux: the rasterizer consumes
// ALREADY-SNAPPED integer ScreenVerts (the ONE floating-point projection step is host-only,
// ProjectToScreenVert; SW2's GPU does ZERO FP — it gets the same integer ScreenVerts via SSBO, so
// coverage is trivially identical cross-vendor). The coverage test is pure integer edge functions
// with a top-left fill rule; the merge is a serial std::min over a 32-bit depth|id key (the CPU
// mirror of SW2's InterlockedMin — commutative, so the triangle order is irrelevant to the result).
//
// DEPTH SEMANTICS: a SMALLER quantized depth means NEARER. The depth occupies the HIGH bits of the
// packed key, the visId the LOW bits, so a smaller depthQ makes the whole key smaller -> min selects
// the front-most surface, ties broken toward the lower visId deterministically. The flat per-triangle
// depth = min(a.z, b.z, c.z) (the simplest deterministic beachhead — conservative; perspective-correct
// interpolated depth is a deferred refinement, see the spec's Out-of-scope).
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/mtl::) symbols. NO GPU, NO new RHI. Mentions of "GPU"/SW2
// here are doc-only.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "math/math.h"
#include "render/meshlet.h"
#include "render/visbuffer.h"  // PackVisId / UnpackVisId / kTriIdBits — the shared (clusterID,triID) identity
#include "scene/vertex.h"      // scene::Vertex (the showcase scene builder)

namespace hf::render::vg {

// ----- The 32-bit `depth|id` packing budget -----------------------------------------------------
// The packed key is [ depthQ : kSwDepthBits | visId : kSwIdBits ], depth in the HIGH bits so a smaller
// (nearer) depthQ makes the whole key smaller -> min picks the front-most surface.
static constexpr uint32_t kSwIdBits    = 16;
static constexpr uint32_t kSwDepthBits = 16;
static_assert(kSwIdBits + kSwDepthBits == 32, "depth|id key must fill exactly 32 bits");
// The visId field must cover the max visId at this scale: visId = (clusterID << kTriIdBits) | triID,
// so kSwIdBits must leave room for the cluster index above the 7 triID bits. With kSwIdBits=16 the id
// field holds clusterIDs in [0, 1<<(16-7)) = [0, 512) — far above any realized on-screen cluster count
// for this CPU-reference scale (the showcase scene has a few clusters). The static_assert pins that the
// id field at least spans a full cluster's triangles plus headroom for the cluster index.
static_assert(kSwIdBits > kTriIdBits, "the visId field must hold the triID bits PLUS a cluster index");
static constexpr uint32_t kSwIdMask = (kSwIdBits >= 32u) ? 0xFFFFFFFFu : ((1u << kSwIdBits) - 1u);

// The maximum quantized depth (the far plane / clear depth). kSwDepthBits bits -> [0, kSwDepthMax].
static constexpr uint32_t kSwDepthMax = (kSwDepthBits >= 32u) ? 0xFFFFFFFFu : ((1u << kSwDepthBits) - 1u);

// The clear / background sentinel written to every texel no triangle covers: max depth + max id, all
// 1s. Because depth is the high bits, kSwClear is the LARGEST possible key, so any real fragment
// (depthQ <= kSwDepthMax, but with a real id field) only beats it if it is nearer — correct background
// semantics. It collides with a valid PackSw only at depthQ==kSwDepthMax AND visId==kSwIdMask (a
// fragment at the exact far plane whose id is all-ones), which never happens for a real (clusterID,
// triID) at this scale (clusterID would have to be 0x1FF, ~511 clusters, with triID 0x7F, at far-clip).
static constexpr uint32_t kSwClear = 0xFFFFFFFFu;
static_assert(kSwClear == ((kSwDepthMax << kSwIdBits) | kSwIdMask),
              "kSwClear must be the max depth + max id key (all 1s) so min never picks it over a real "
              "nearer fragment");

// Pack a quantized depth (nearer = smaller, [0,kSwDepthMax]) + a visId (= PackVisId(clusterID,triID))
// into the 32-bit depth|id key. depthQ HIGH bits, visId LOW kSwIdBits. visId is masked so an
// out-of-range id can never bleed into the depth field (defensive). depthQ is assumed already clamped
// to [0,kSwDepthMax] by ProjectToScreenVert (the high-bit shift drops any overflow harmlessly).
inline uint32_t PackSw(uint32_t depthQ, uint32_t visId) {
    return (depthQ << kSwIdBits) | (visId & kSwIdMask);
}

// Inverse of PackSw. UB-free for any 32-bit input (the clear sentinel unpacks to its bit fields too;
// callers test against kSwClear before treating an unpacked value as coverage).
inline void UnpackSw(uint32_t v, uint32_t& depthQ, uint32_t& visId) {
    depthQ = v >> kSwIdBits;
    visId  = v & kSwIdMask;
}

// ----- The fixed-point screen vertex -------------------------------------------------------------
// x,y are fixed-point pixel coordinates in 1/kSub units (kSub sub-pixel subdivisions per pixel — 4
// fractional bits at kSub=16). z is the quantized depth in [0,kSwDepthMax]. The rasterizer consumes
// these integers directly (no FP), so coverage is bit-identical on every platform (SW2's GPU gets the
// SAME ScreenVerts via SSBO).
static constexpr int32_t kSub = 16;  // sub-pixel resolution (16 -> 4 sub-pixel bits)

struct ScreenVert {
    int32_t  x = 0;  // fixed-point pixel-x in 1/kSub units
    int32_t  y = 0;  // fixed-point pixel-y in 1/kSub units
    uint32_t z = 0;  // quantized depth in [0,kSwDepthMax] (nearer = smaller)
};

// ----- The visibility buffer ---------------------------------------------------------------------
// packed[y*w + x], every texel initialized to kSwClear. Same flat-integer layout as visbuffer.h's
// read-back uint32[w*h], so the same CPU-coloring (bg -> clear, else hashColor) applies verbatim.
struct SwVisBuffer {
    uint32_t w = 0;
    uint32_t h = 0;
    std::vector<uint32_t> packed;  // size w*h, init kSwClear

    void Init(uint32_t width, uint32_t height) {
        w = width; h = height;
        packed.assign((size_t)w * h, kSwClear);
    }
};

// ----- The ONE floating-point step (HOST-ONLY): world position -> snapped integer ScreenVert -------
// MulPointDivide -> NDC -> pixel (with the Vulkan Y-flip already baked into the projection's row 1, the
// SAME convention visbuffer.h:113-115 uses: pixel = ((ndc+1)/2)*extent) -> round to 1/sub fixed-point.
// The depth z (NDC z in [0,1] for the engine's [0,1] Vulkan depth range) is quantized to [0,kSwDepthMax]
// with round-to-nearest. THE SNAP RULE: round(px * sub) — a symmetric round-half-away-from-zero via
// floor(v + 0.5) for v>=0 (screen coords are >= 0 inside the viewport; the bbox clamp handles the rest).
// `ok` is set false for a vertex behind the camera (clipW <= 0); the caller skips such triangles.
// SW2's GPU does NOT run this — it receives the resulting integer ScreenVerts; this is the host snap.
inline ScreenVert ProjectToScreenVert(const math::Vec3& worldPos, const math::Mat4& viewProj,
                                      uint32_t w, uint32_t h, int32_t sub, bool& ok) {
    float clipW = 0.0f;
    math::Vec3 ndc = math::MulPointDivide(viewProj, worldPos, clipW);
    ok = (clipW > 0.0f);
    // pixel = ((ndc + 1)/2) * extent (Y-flip already in the projection matrix).
    float px = (ndc.x * 0.5f + 0.5f) * (float)w;
    float py = (ndc.y * 0.5f + 0.5f) * (float)h;
    // NDC z is already in [0,1] for the engine's Vulkan depth range; clamp + quantize.
    float z = ndc.z;
    if (z < 0.0f) z = 0.0f;
    if (z > 1.0f) z = 1.0f;
    auto snap = [](float v, int32_t s) -> int32_t {
        // round-half-away-from-zero
        return (v >= 0.0f) ? (int32_t)(v * (float)s + 0.5f)
                           : -(int32_t)(-v * (float)s + 0.5f);
    };
    ScreenVert sv;
    sv.x = snap(px, sub);
    sv.y = snap(py, sub);
    sv.z = (uint32_t)(z * (float)kSwDepthMax + 0.5f);
    return sv;
}

// ----- The integer edge function -----------------------------------------------------------------
// E(p,q,r) = (p.x - q.x)*(r.y - q.y) - (p.y - q.y)*(r.x - q.x). int64 intermediates: the fixed-point
// coordinate products (pixels * kSub, up to ~256*16 = 4096 per axis, squared ~1.7e7, times another such
// difference) stay well inside int64. Sign convention: with this orientation E(a,b,c) > 0 for a
// counter-clockwise triangle (a,b,c) in the pixel-Y-DOWN screen space the projection produces (the
// Y-flip is in the projection, so screen-space winding follows the projected vertex order). The fill
// rule below is consistent with E > 0 = inside.
inline int64_t SwEdge(int32_t px, int32_t py, int32_t qx, int32_t qy, int32_t rx, int32_t ry) {
    return (int64_t)(px - qx) * (int64_t)(ry - qy) - (int64_t)(py - qy) * (int64_t)(rx - qx);
}

// Top-left fill rule: a boundary pixel (E == 0 exactly on an edge) is covered ONLY if that edge is a
// top-or-left edge, so a shared edge between two triangles covers each boundary pixel EXACTLY once (no
// double-cover, no gap) — the watertight-raster contract. For the edge running (va -> vb):
//   - a LEFT edge goes "up" the screen: dy = vb.y - va.y < 0.
//   - a TOP edge is horizontal pointing left: dy == 0 && dx = vb.x - va.x < 0.
// (Winding: the three edges of a CCW triangle are taken as (b->c), (c->a), (a->b); the edge that
// produces E for vertex-opposite is the segment skipped by that vertex. Documented + unit-tested.)
inline bool SwIsTopLeft(int32_t ax, int32_t ay, int32_t bx, int32_t by) {
    int32_t dy = by - ay;
    int32_t dx = bx - ax;
    return (dy < 0) || (dy == 0 && dx < 0);
}

// ----- The deterministic integer scan-converter --------------------------------------------------
// Rasterize one triangle (a,b,c) with identity visId into the vis-buffer. Integer edge functions at
// each pixel CENTER ((x*kSub + kSub/2, y*kSub + kSub/2)); top-left fill rule; flat depth = min(z);
// packed[p] = min(packed[p], PackSw(depthQ, visId)) — the serial min = SW2's InterlockedMin mirror.
inline void RasterTriangle(SwVisBuffer& vb, const ScreenVert& a, const ScreenVert& b,
                           const ScreenVert& c, uint32_t visId) {
    // Signed area*2 (the (a,b,c) edge). Zero -> degenerate, skip.
    const int64_t area2 = SwEdge(a.x, a.y, b.x, b.y, c.x, c.y);
    if (area2 == 0) return;

    // Orient so the triangle is CCW (area2 > 0) — swap b,c if needed so E > 0 means inside for all
    // three edges below. The top-left rule is applied to the (possibly swapped) edges consistently.
    ScreenVert v0 = a, v1 = b, v2 = c;
    if (area2 < 0) std::swap(v1, v2);

    // Integer pixel bounding box (the verts are in 1/kSub units; divide to pixel units, floor the min,
    // ceil the max), clamped to [0,w)x[0,h).
    auto floorDiv = [](int32_t n, int32_t d) -> int32_t {
        // floor division for positive d, any sign n.
        int32_t q = n / d, r = n % d;
        return (r != 0 && ((r < 0) != (d < 0))) ? q - 1 : q;
    };
    int32_t minXf = std::min({v0.x, v1.x, v2.x});
    int32_t maxXf = std::max({v0.x, v1.x, v2.x});
    int32_t minYf = std::min({v0.y, v1.y, v2.y});
    int32_t maxYf = std::max({v0.y, v1.y, v2.y});
    int32_t minX = floorDiv(minXf, kSub);
    int32_t maxX = floorDiv(maxXf + (kSub - 1), kSub);  // ceil
    int32_t minY = floorDiv(minYf, kSub);
    int32_t maxY = floorDiv(maxYf + (kSub - 1), kSub);
    if (minX < 0) minX = 0;
    if (minY < 0) minY = 0;
    if (maxX > (int32_t)vb.w - 1) maxX = (int32_t)vb.w - 1;
    if (maxY > (int32_t)vb.h - 1) maxY = (int32_t)vb.h - 1;
    if (minX > maxX || minY > maxY) return;

    // Flat per-triangle depth: the nearest (smallest) of the 3 quantized depths.
    const uint32_t depthQ = std::min({v0.z, v1.z, v2.z});
    const uint32_t key = PackSw(depthQ, visId);

    // The three edges, taken so each E_i is the half-plane opposite vertex i (consistent CCW winding):
    //   e0 = (v1 -> v2) opposite v0;  e1 = (v2 -> v0) opposite v1;  e2 = (v0 -> v1) opposite v2.
    const bool tl0 = SwIsTopLeft(v1.x, v1.y, v2.x, v2.y);
    const bool tl1 = SwIsTopLeft(v2.x, v2.y, v0.x, v0.y);
    const bool tl2 = SwIsTopLeft(v0.x, v0.y, v1.x, v1.y);

    for (int32_t y = minY; y <= maxY; ++y) {
        const int32_t cy = y * kSub + kSub / 2;  // pixel-center in fixed-point
        for (int32_t x = minX; x <= maxX; ++x) {
            const int32_t cx = x * kSub + kSub / 2;
            const int64_t e0 = SwEdge(cx, cy, v1.x, v1.y, v2.x, v2.y);
            const int64_t e1 = SwEdge(cx, cy, v2.x, v2.y, v0.x, v0.y);
            const int64_t e2 = SwEdge(cx, cy, v0.x, v0.y, v1.x, v1.y);
            // Covered iff each edge is strictly inside (E>0) OR exactly-on (E==0) AND that edge is a
            // top-or-left edge.
            const bool in0 = (e0 > 0) || (e0 == 0 && tl0);
            const bool in1 = (e1 > 0) || (e1 == 0 && tl1);
            const bool in2 = (e2 > 0) || (e2 == 0 && tl2);
            if (in0 && in1 && in2) {
                uint32_t& dst = vb.packed[(size_t)y * vb.w + x];
                if (key < dst) dst = key;  // serial min (InterlockedMin mirror; commutative)
            }
        }
    }
}

// ----- The convenience: rasterize all clusters' triangles ----------------------------------------
// `screen` holds one ScreenVert per source vertex (already projected/snapped by the caller).
// `behind[i]` marks a vertex projected behind the camera (clipW<=0); a triangle with any behind vertex
// is skipped (no near-clip in this beachhead — the showcase scene is fully in front). For each meshlet
// `m`, each of its `triCount` triangles `t` (the 3 indices indices[3*(m.triOffset+t)+{0,1,2}]) is
// rasterized with visId = PackVisId(clusterId, t), where clusterId is the meshlet's index in `meshlets`.
// The serial min makes the rasterization order IRRELEVANT to the result (proven by the shuffled-order
// determinism test) — so this fixed order is a presentation choice, not a correctness one.
inline void RasterClusters(SwVisBuffer& vb,
                           std::span<const ScreenVert> screen,
                           std::span<const uint8_t> behind,
                           std::span<const uint32_t> indices,
                           std::span<const Meshlet> meshlets) {
    for (uint32_t clusterId = 0; clusterId < (uint32_t)meshlets.size(); ++clusterId) {
        const Meshlet& m = meshlets[clusterId];
        for (uint32_t t = 0; t < m.triCount; ++t) {
            const uint32_t base = 3u * (m.triOffset + t);
            const uint32_t i0 = indices[base + 0];
            const uint32_t i1 = indices[base + 1];
            const uint32_t i2 = indices[base + 2];
            if (behind[i0] || behind[i1] || behind[i2]) continue;  // skip cross-camera triangles
            RasterTriangle(vb, screen[i0], screen[i1], screen[i2], PackVisId(clusterId, t));
        }
    }
}

// ----- The showcase scene (shared VERBATIM by --swraster-shot (Vulkan) + --swraster (Metal)) ------
// A small fixed deterministic scene: two clustered quads at known different depths, overlapping in
// screen space so the nearer one occludes the farther on the overlap (the depth-occlusion proof), PLUS
// one sub-pixel triangle positioned so a hardware sample-at-pixel-center would MISS it but the SW
// raster (testing the exact pixel center it straddles) COVERS it (the raison d'etre). Everything is
// integer/fixed-point after the host projection -> identical bytes both backends BY CONSTRUCTION.
struct SwRasterScene {
    std::vector<scene::Vertex> verts;
    std::vector<uint32_t>      indices;
    std::vector<Meshlet>       meshlets;
    math::Mat4                 viewProj;
    uint32_t                   w = 0, h = 0;
    // Pixel coordinate of the sub-pixel triangle's expected single covered texel (for the proof).
    uint32_t subPixelPx = 0, subPixelPy = 0;
};

// Build the fixed showcase scene. `w,h` is the vis-buffer resolution (the showcase uses 256x256).
inline SwRasterScene BuildSwRasterScene(uint32_t w, uint32_t h) {
    using math::Vec3; using math::Mat4;
    SwRasterScene s;
    s.w = w; s.h = h;

    auto addVert = [&](float x, float y, float z) {
        scene::Vertex v{};
        v.pos[0] = x; v.pos[1] = y; v.pos[2] = z;
        v.normal[0] = 0.0f; v.normal[1] = 0.0f; v.normal[2] = 1.0f;
        s.verts.push_back(v);
        return (uint32_t)(s.verts.size() - 1);
    };
    // A z=const facing quad as 2 triangles (CCW front-facing in this RH/Vulkan setup).
    auto addQuad = [&](float cx, float cy, float half, float z) -> std::pair<uint32_t,uint32_t> {
        uint32_t a = addVert(cx - half, cy - half, z);
        uint32_t b = addVert(cx + half, cy - half, z);
        uint32_t c = addVert(cx + half, cy + half, z);
        uint32_t d = addVert(cx - half, cy + half, z);
        uint32_t t0 = (uint32_t)(s.indices.size() / 3);
        s.indices.push_back(a); s.indices.push_back(b); s.indices.push_back(c);
        s.indices.push_back(a); s.indices.push_back(c); s.indices.push_back(d);
        (void)t0;
        return {a, d};
    };

    // Cluster 0: a FAR quad (z=-6), left-of-center, large.
    uint32_t farTriStart = (uint32_t)(s.indices.size() / 3);
    addQuad(-0.6f, 0.0f, 1.3f, -6.0f);
    Meshlet m0; m0.triOffset = farTriStart; m0.triCount = 2; s.meshlets.push_back(m0);

    // Cluster 1: a NEAR quad (z=-4), right-of-center, large, OVERLAPPING cluster 0 in screen space so
    // the near quad wins on the overlap (depth-occlusion proof).
    uint32_t nearTriStart = (uint32_t)(s.indices.size() / 3);
    addQuad(0.6f, 0.0f, 1.3f, -4.0f);
    Meshlet m1; m1.triOffset = nearTriStart; m1.triCount = 1;  // first tri of the near quad
    s.meshlets.push_back(m1);
    Meshlet m1b; m1b.triOffset = nearTriStart + 1; m1b.triCount = 1;  // second tri (cluster 2)
    s.meshlets.push_back(m1b);

    // The camera + projection (the ONE FP context; everything downstream is integer).
    Mat4 view = Mat4::LookAt({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f});
    float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
    Mat4 proj = Mat4::Perspective(0.9f, aspect, 0.1f, 50.0f);
    s.viewProj = proj * view;

    // Cluster 3: a SUB-PIXEL triangle. Placed in screen space so it is smaller than one pixel and
    // straddles a single pixel's CENTER (so the SW raster covers exactly that pixel, while a HW sample
    // at a slightly different position would miss). We pick a target pixel near the top-left quadrant,
    // away from the quads, and synthesize a tiny world triangle that projects onto it. Easiest robust
    // construction: project a known world point to find its pixel, then build a tiny triangle in NDC
    // around the target pixel center and unproject is overkill — instead we add it in a region the
    // quads don't cover and assert coverage of the resulting pixel from the rasterized buffer.
    // Target pixel: a deterministic spot in the lower-left (in pixel coords).
    s.subPixelPx = w / 5;       // e.g. 51 for w=256
    s.subPixelPy = h / 5;       // e.g. 51 for h=256
    // Build a tiny world triangle that projects to ~ a third of a pixel around that target. We do this
    // by placing it at a near depth and a world XY chosen to land on the target pixel center. Find the
    // world plane z=-2; the pixel center maps back through the inverse view-proj at that depth.
    {
        // The pixel center in NDC.
        float ndcX = ((float)s.subPixelPx + 0.5f) / (float)w * 2.0f - 1.0f;
        // Account for the Y-flip baked into the projection: pixel->ndc is the inverse of
        // ndc->pixel = ((ndc+1)/2)*extent, so ndcY = (py+0.5)/h*2 - 1 (the flip is already in proj).
        float ndcY = ((float)s.subPixelPy + 0.5f) / (float)h * 2.0f - 1.0f;
        float zw = -2.0f;  // world depth of the tiny triangle
        // Forward-project zw to get its NDC z + w, then back out world XY from the NDC center.
        // worldCenter = invVP * (ndcX, ndcY, ndcZ, 1) * w. We have proj*view; invert it.
        Mat4 invVP = s.viewProj.Inverse();
        // First find ndcZ for world z=-2 along the view axis (x=y=0 ray).
        float wDummy = 0.0f;
        Vec3 onAxis = math::MulPointDivide(s.viewProj, Vec3{0.0f, 0.0f, zw}, wDummy);
        float ndcZ = onAxis.z;
        // Unproject the target NDC at that depth back to world.
        // Homogeneous unproject: p = invVP * (ndcX,ndcY,ndcZ,1); divide by p.w.
        auto unproj = [&](float nx, float ny, float nz) -> Vec3 {
            const float* m = invVP.m;
            float x = m[0]*nx + m[4]*ny + m[8]*nz + m[12];
            float y = m[1]*nx + m[5]*ny + m[9]*nz + m[13];
            float z = m[2]*nx + m[6]*ny + m[10]*nz + m[14];
            float wq = m[3]*nx + m[7]*ny + m[11]*nz + m[15];
            float inv = (wq != 0.0f) ? 1.0f / wq : 1.0f;
            return Vec3{x*inv, y*inv, z*inv};
        };
        Vec3 centerW = unproj(ndcX, ndcY, ndcZ);
        // A tiny triangle around centerW: offsets of ~ a quarter pixel in world units. One pixel in
        // world at z=-2 ~ (2*tan(fov/2)*|z| / h). Use 0.3px so it straddles the center but is < 1px.
        float worldPerPixel = (2.0f * std::tan(0.45f) * 2.0f) / (float)h;  // approx, fov=0.9 -> half 0.45
        float r = worldPerPixel * 0.30f;
        uint32_t triStart = (uint32_t)(s.indices.size() / 3);
        uint32_t a = addVert(centerW.x - r, centerW.y - r, zw);
        uint32_t b = addVert(centerW.x + r, centerW.y - r, zw);
        uint32_t c = addVert(centerW.x,     centerW.y + r, zw);
        s.indices.push_back(a); s.indices.push_back(b); s.indices.push_back(c);
        Meshlet m3; m3.triOffset = triStart; m3.triCount = 1; s.meshlets.push_back(m3);
    }
    return s;
}

// Project + snap every scene vertex to integer ScreenVerts (the host-only FP step) -> the parallel
// `screen` + `behind` arrays RasterClusters consumes.
inline void ProjectSwRasterScene(const SwRasterScene& s, std::vector<ScreenVert>& screen,
                                  std::vector<uint8_t>& behind) {
    screen.resize(s.verts.size());
    behind.assign(s.verts.size(), 0);
    for (size_t i = 0; i < s.verts.size(); ++i) {
        const scene::Vertex& v = s.verts[i];
        bool ok = true;
        screen[i] = ProjectToScreenVert(math::Vec3{v.pos[0], v.pos[1], v.pos[2]}, s.viewProj,
                                        s.w, s.h, kSub, ok);
        behind[i] = ok ? 0 : 1;
    }
}

// CPU-color a vis-buffer into BGRA8 (top row first), the SAME coloring visbuffer.h's showcase uses:
// background -> a fixed dark clear; else hashColor(visId >> kTriIdBits) (the cluster's color). Shared
// VERBATIM by both backends -> identical bytes by construction.
inline std::vector<uint8_t> ColorSwVisBuffer(const SwVisBuffer& vb) {
    std::vector<uint8_t> bgra((size_t)vb.w * vb.h * 4);
    const uint8_t bgB = 13, bgG = 13, bgR = 5;  // the same dark navy as visbuffer.h
    for (size_t p = 0; p < vb.packed.size(); ++p) {
        uint32_t v = vb.packed[p];
        uint8_t* px = &bgra[p * 4];
        if (v == kSwClear) {
            px[0] = bgB; px[1] = bgG; px[2] = bgR; px[3] = 255;
        } else {
            uint32_t depthQ, visId;
            UnpackSw(v, depthQ, visId);
            math::Vec3 col = hashColor(visId >> kTriIdBits);  // color by cluster
            px[0] = (uint8_t)(col.z * 255.0f + 0.5f);  // B
            px[1] = (uint8_t)(col.y * 255.0f + 0.5f);  // G
            px[2] = (uint8_t)(col.x * 255.0f + 0.5f);  // R
            px[3] = 255;
        }
    }
    return bgra;
}

}  // namespace hf::render::vg
