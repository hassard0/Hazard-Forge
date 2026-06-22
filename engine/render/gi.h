#pragma once
// Slice GI1 — Deterministic Lumen-class GI: INTEGER RT PROBE TRACE + SHADE (the beachhead) — pure CPU
// (header-only, NO device, NO backend symbols, NO <cmath> on the bit-exact path). Namespace
// hf::render::gi. The FIRST slice of FLAGSHIP #29 (DETERMINISTIC LUMEN-CLASS GI). Flagship #28 delivered
// deterministic hardware ray tracing (engine/render/rtrace.h: integer TraceClosest / TraceAnyHit /
// ShadeHitShadowed, the HW candidate-drain rt_query.comp, the accel-struct RHI seam — all strict-zero
// cross-vendor). GI1 builds the GI data layer on that foundation, kept STRICT INTEGER so the GI is
// byte-identical HW==SW==CPU on NVIDIA + Apple — the moat answer to UE5 Lumen (float / temporal /
// non-deterministic).
//
// WHAT THIS IS: a WORLD-SPACE lattice of irradiance PROBES; every probe casts kGiRaysPerProbe = 16
// deterministic Fibonacci-sphere rays via the RT CLOSEST-HIT (rtrace::TraceClosest over the procedural-AABB
// scene — world-space, off-screen-complete, not the float DDGI's screen-space depth march), shading each
// hit with direct light + an RT SHADOW ray (rtrace::ShadeHitShadowed, gated by rtrace::TraceAnyHit), and
// accumulates the per-ray radiance in integer Q16.16. A probe ray that MISSES sees the scene background /
// sky term. This is the BRUTE-FORCE no-cull REFERENCE — the completeness oracle the GI's HW path
// (shaders/gi_probe_trace.comp, Vulkan inline RayQuery) is memcmp'd against. GI1 produces ONLY the
// per-probe-per-ray radiance; GI2 encodes it to integer SH, GI3 interpolates, GI4 adds multi-bounce.
//
// THE CROSS-BACKEND CRUX (the make-or-break for HW==CPU): every probe position, ray direction, and shade
// is PURE-INTEGER Q16.16 — the SAME frozen rtrace:: fx helpers (TraceClosest/TraceAnyHit/ShadeHitShadowed),
// NO float / std::sqrt / <cmath> on the bit-exact path. The 16 probe-ray directions are a HOST-PRECOMPUTED
// Fibonacci sphere baked as EXACT Q16.16 literals (computed once in double at table-definition time — see
// the generator comment on kGiProbeDirs) so the GPU never evaluates a transcendental on the hot path and
// both backends read IDENTICAL bits. The per-ray radiance GiRadiance{fx r,g,b} is the UNPACK of the
// ShadeHitShadowed RGBA8 (channel*kOne/255), an integer divide identical CPU<->DXC. Each (probe,ray) is
// INDEPENDENT (one thread per probe in the shader, the 16 rays a register loop) so two runs are
// byte-identical.
//
// REUSE MAP (file:line): rtrace.h (READ-ONLY, BYTE-FROZEN) — TraceClosest (242), TraceAnyHit (417),
// ShadeHitShadowed (433), IntersectSphere/IntersectAabb, RtScene/RtSphere/RtAabb/RtRay/RtHit (92-132),
// kRtShadowEps/kRtShadowMinT (409-410), FxVec3/fxmul/FxDot/FxAdd/FxScale/RtNormalize/PackRGBA8,
// kRtMiss/kRtNoHit, F (354) — all #included read-only. probe_gi.h is the FLOAT DDGI lattice template GI1
// twins in integer (ProbeGrid 58-72 -> GiProbeGrid; probeCount/probePos; ProbeDispatchGroups->0 no-op 285
// -> ProbeDispatchGroups here). The shader shaders/gi_probe_trace.comp.hlsl copies the 16 directions + the
// per-ray TraceClosest/ShadeHitShadowed VERBATIM. NO sim header / sim shader / sim golden touched (GI is
// render-only).
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/mtl::) symbols. NO GPU, NO new RHI here. Mentions of "GPU"/HW
// are doc-only — GI1 reuses the RT2 accel-struct seam (CreateBlas/CreateTlas/BindAccelStructure) verbatim.

#include <cstdint>
#include <span>
#include <vector>

#include "render/rtrace.h"  // TraceClosest / TraceAnyHit / ShadeHitShadowed / RtScene / fx helpers (READ-ONLY, BYTE-FROZEN)

namespace hf::render::gi {

// ----- Reuse the Q16.16 fixed-point scalar + vector + the frozen RT helpers (rtrace.h, byte-frozen) ------
using rtrace::fx;
using rtrace::FxVec3;
using rtrace::kFrac;
using rtrace::kOne;

using rtrace::FxAdd;
using rtrace::FxScale;
using rtrace::fxmul;
using rtrace::RtScene;
using rtrace::RtRay;
using rtrace::RtHit;
using rtrace::TraceClosest;
using rtrace::TraceAnyHit;
using rtrace::ShadeHitShadowed;
using rtrace::kRtMiss;
using rtrace::kRtShadowEps;
using rtrace::kRtShadowMinT;
using rtrace::PackRGBA8;

// A Q16.16 fraction wholeNum/wholeDen (== rtrace::F; re-declared here so gi.h's scene builder reads cleanly
// without dragging rtrace::F into the namespace via a using that might collide). Integer, deterministic.
inline fx GiF(int wholeNum, int wholeDen) {
    return (fx)(((int64_t)wholeNum << kFrac) / wholeDen);
}

// ===== The probe grid: a WORLD-space integer lattice of irradiance probes ============================
// nx*ny*nz probes spaced `spacing` apart from `origin` (all Q16.16). The integer twin of the float DDGI
// probegi::ProbeGrid: probe (ix,iy,iz) sits at origin + (ix,iy,iz)*spacing. A dim == 0 -> ProbeCount() == 0
// -> the disabled / no-op path (ProbeDispatchGroups -> 0 -> the radiance buffer untouched).
struct GiProbeGrid {
    FxVec3  origin{0, 0, 0};
    fx      spacing = kOne;   // Q16.16 lattice step
    int32_t nx = 4, ny = 4, nz = 4;
};

// The probe count. nx/ny/nz == 0 -> 0 (the no-op).
inline int ProbeCount(const GiProbeGrid& grid) {
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) return 0;
    return grid.nx * grid.ny * grid.nz;
}

// cx-major flat index (the froxel/cluster/DDGI order): i = ix + iy*nx + iz*(nx*ny).
inline int ProbeFlatIndex(const GiProbeGrid& grid, int ix, int iy, int iz) {
    return ix + iy * grid.nx + iz * (grid.nx * grid.ny);
}

// The world position of probe (ix,iy,iz): origin + (ix,iy,iz)*spacing (pure integer fxmul scale).
inline FxVec3 ProbePos(const GiProbeGrid& grid, int ix, int iy, int iz) {
    return FxVec3{grid.origin.x + fxmul((fx)(ix * (int64_t)kOne), grid.spacing),
                  grid.origin.y + fxmul((fx)(iy * (int64_t)kOne), grid.spacing),
                  grid.origin.z + fxmul((fx)(iz * (int64_t)kOne), grid.spacing)};
}

// The world position of probe linearIndex (cx-major decode), for the per-probe loop / the shader thread.
inline FxVec3 ProbePos(const GiProbeGrid& grid, int linearIndex) {
    int nx = grid.nx, ny = grid.ny;
    int ix = linearIndex % nx;
    int iy = (linearIndex / nx) % ny;
    int iz = linearIndex / (nx * ny);
    return ProbePos(grid, ix, iy, iz);
}

// ===== The probe-ray direction table (host-precomputed Fibonacci sphere, baked Q16.16) ===============
// kGiRaysPerProbe = 16 deterministic full-sphere directions. The shader reads an IDENTICAL constant array.
inline constexpr int kGiRaysPerProbe = 16;

// kGiProbeDirs — the 16 Fibonacci-sphere (golden-spiral) UNIT directions, computed ONCE in double at
// table-definition time and baked as EXACT Q16.16 (round-to-nearest) literals so NO cos/sin runs on the
// hot path and both backends read identical bits (the probe_sh.h host-table discipline, in integer). The
// generator (the probegi::FibonacciSphere formula, golden angle GA = 2.39996322972865332):
//     for i in 0..15:
//         z   = 1 - (2*i + 1)/16          // evenly spaced z in (-1,1) at slab centers (DECREASING)
//         r   = sqrt(max(0, 1 - z*z))     // the circle radius at height z
//         phi = i * GA                    // the golden-angle azimuth spiral
//         dir = (r*cos(phi), r*sin(phi), z)
//         literal = round(dir * 65536)    // Q16.16, round-to-nearest
// Each baked vector is unit-length to within < 2e-5 (|q| in [0.99999, 1.00001]) — the rt::IntersectSphere /
// IntersectAabb math never requires the dir be exactly unit (t is in units of |dir|, consistent per ray).
static constexpr FxVec3 kGiProbeDirs[kGiRaysPerProbe] = {
    {   22806,       0,   61440 },  // i= 0  (+0.34799,+0.00000,+0.93750)
    {  -28171,   25807,   53248 },  // i= 1  (-0.42986,+0.39378,+0.81250)
    {    4161,  -47409,   45056 },  // i= 2  (+0.06349,-0.72340,+0.68750)
    {   32968,   43001,   36864 },  // i= 3  (+0.50306,+0.65615,+0.56250)
    {  -58030,  -10265,   28672 },  // i= 4  (-0.88547,-0.15663,+0.43750)
    {   52527,  -33413,   20480 },  // i= 5  (+0.80150,-0.50985,+0.31250)
    {  -16712,   62167,   12288 },  // i= 6  (-0.25500,+0.94859,+0.18750)
    {  -30147,  -58046,    4096 },  // i= 7  (-0.46001,-0.88571,+0.06250)
    {   61439,   22437,   -4096 },  // i= 8  (+0.93748,+0.34237,-0.06250)
    {  -59504,   24562,  -12288 },  // i= 9  (-0.90795,+0.37479,-0.18750)
    {   26386,  -56385,  -20480 },  // i=10  (+0.40262,-0.86037,-0.31250)
    {   17637,   56230,  -28672 },  // i=11  (+0.26912,+0.85800,-0.43750)
    {  -46881,  -27169,  -36864 },  // i=12  (-0.71535,-0.41456,-0.56250)
    {   46481,  -10219,  -45056 },  // i=13  (+0.70925,-0.15593,-0.68750)
    {  -21973,   31254,  -53248 },  // i=14  (-0.33528,+0.47690,-0.81250)
    {   -2931,  -22616,  -61440 },  // i=15  (-0.04472,-0.34510,-0.93750)
};

// ===== The per-(probe,ray) radiance output element ===================================================
// Q16.16 linear radiance per ray (the unpacked RGBA8 ShadeHitShadowed color). Padded to 16 B so the GPU
// std430 mirror (gRadiance[probe*16 + dir]) matches byte-for-byte (a 4th int slot, kept 0).
struct GiRadiance {
    fx r = 0, g = 0, b = 0;
    fx _pad = 0;   // std430 16-byte stride (matches the shader's int4)
};
static_assert(sizeof(GiRadiance) == 16, "GiRadiance must be 16 bytes (std430 int4)");

// Unpack a ShadeHitShadowed RGBA8 (0xAABBGGRR) to a GiRadiance (per-channel byte * kOne / 255 -> Q16.16 in
// [0,1]). A pure integer divide — identical CPU <-> DXC. This is the SAME unpack the shader runs, so the
// CPU reference and the HW path agree to the bit.
inline GiRadiance UnpackRadiance(uint32_t rgba) {
    auto ch = [&](int shift) -> fx {
        int32_t b = (int32_t)((rgba >> shift) & 0xFFu);
        return (fx)(((int64_t)b * (int64_t)kOne) / 255);
    };
    return GiRadiance{ch(0), ch(8), ch(16), 0};
}

// ===== TraceProbeRays — the brute-force no-cull GI completeness oracle ================================
// Per probe p, per direction d: build RtRay{ProbePos(grid,p), kGiProbeDirs[d]}; TraceClosest over the
// scene; if it HIT, cast a SHADOW ray {hit.pos + normal*kRtShadowEps, scene.lightDir} and occluded =
// TraceAnyHit(shadowRay, scene, kRtShadowMinT); shade = ShadeHitShadowed(hit, scene, occluded); a MISS
// shades to scene.background (ShadeHitShadowed returns it). Unpack the RGBA8 to GiRadiance, store at
// out[p*16 + d]. PURE INTEGER, deterministic. `out` must be ProbeCount(grid)*kGiRaysPerProbe long; a
// 0-probe grid writes nothing (the no-op). The shader copies this loop VERBATIM (one thread per probe).
inline void TraceProbeRays(const GiProbeGrid& grid, const RtScene& scene, std::span<GiRadiance> out) {
    int probes = ProbeCount(grid);
    for (int p = 0; p < probes; ++p) {
        FxVec3 origin = ProbePos(grid, p);
        for (int d = 0; d < kGiRaysPerProbe; ++d) {
            RtRay ray{origin, kGiProbeDirs[d]};
            RtHit hit = TraceClosest(ray, scene);
            bool occluded = false;
            if (hit.primIndex != kRtMiss) {
                RtRay shadowRay;
                shadowRay.origin = FxAdd(hit.pos, FxScale(hit.normal, kRtShadowEps));
                shadowRay.dir = scene.lightDir;
                occluded = TraceAnyHit(shadowRay, scene, kRtShadowMinT);
            }
            uint32_t rgba = ShadeHitShadowed(hit, scene, occluded);
            out[(size_t)p * kGiRaysPerProbe + d] = UnpackRadiance(rgba);
        }
    }
}

// MeanRadiance — the integer average of a probe's kGiRaysPerProbe radiances (the per-tile viz color +
// the "lit" sanity). Pure integer sum >> /count. `rays` is the probe's 16-element radiance slice.
inline GiRadiance MeanRadiance(std::span<const GiRadiance> rays) {
    int64_t sr = 0, sg = 0, sb = 0;
    int n = (int)rays.size();
    if (n <= 0) return GiRadiance{};
    for (const GiRadiance& g : rays) { sr += g.r; sg += g.g; sb += g.b; }
    return GiRadiance{(fx)(sr / n), (fx)(sg / n), (fx)(sb / n), 0};
}

// ===== GiProbesToImage — the deterministic probe-radiance grid viz (the golden) ======================
// A flat 2D TILE GRID: the ProbeCount probes are laid out left-to-right, top-to-bottom (cx-major linear
// order) into a near-square tile grid; each tile is filled with the probe's MEAN per-ray radiance quantized
// to RGBA8 (a (val*255)>>kFrac per channel). NOT a 3D render — GI6 is the lit capstone. Strict integer ->
// identical both backends. `img` is row-major RGBA8 (top row first), size == w*h. `out` is the radiance
// buffer (ProbeCount*kGiRaysPerProbe). A 0-probe grid -> the image is left as the background fill.
inline void GiProbesToImage(const GiProbeGrid& grid, std::span<const GiRadiance> out,
                            std::span<uint32_t> img, uint32_t w, uint32_t h) {
    // A neutral dark background (so empty tiles + padding read as "no probe").
    const uint32_t kBg = PackRGBA8(18, 18, 22, 255);
    for (uint32_t i = 0; i < w * h; ++i) img[i] = kBg;

    int probes = ProbeCount(grid);
    if (probes <= 0) return;

    // A near-square tile grid: cols = ceil(sqrt(probes)), rows = ceil(probes/cols).
    int cols = 1;
    while (cols * cols < probes) ++cols;
    int rows = (probes + cols - 1) / cols;

    // Each tile is tileW x tileH with a 1px gutter implicit (the background shows between tiles when the
    // tile pixels don't fully tile the image). Integer division -> deterministic.
    uint32_t tileW = w / (uint32_t)cols;
    uint32_t tileH = h / (uint32_t)rows;
    if (tileW == 0) tileW = 1;
    if (tileH == 0) tileH = 1;

    auto q = [](fx v) -> int32_t { return (int32_t)(((int64_t)v * 255) >> kFrac); };

    for (int p = 0; p < probes; ++p) {
        int col = p % cols;
        int row = p / cols;
        GiRadiance mean = MeanRadiance(out.subspan((size_t)p * kGiRaysPerProbe, kGiRaysPerProbe));
        uint32_t color = PackRGBA8(q(mean.r), q(mean.g), q(mean.b), 255);
        // Fill the tile interior, leaving a 1px gutter on the right/bottom so neighboring tiles are
        // visually distinct (the gutter shows the background).
        uint32_t x0 = (uint32_t)col * tileW;
        uint32_t y0 = (uint32_t)row * tileH;
        uint32_t x1 = x0 + (tileW > 1 ? tileW - 1 : tileW);
        uint32_t y1 = y0 + (tileH > 1 ? tileH - 1 : tileH);
        for (uint32_t y = y0; y < y1 && y < h; ++y)
            for (uint32_t x = x0; x < x1 && x < w; ++x)
                img[(size_t)y * w + x] = color;
    }
}

// ===== The pinned GI1 enclosure scene ================================================================
// A small Cornell-ish enclosure: a floor + a back wall + a red left wall + a green right wall + a ceiling
// (so GI4's multi-bounce later shows color bleed) + a sphere on the floor, lit by a directional light
// angled down into the box. PINNED (the cross-vendor golden rests on it). The owning storage lives in the
// returned struct (the spans in `scene` point into it — keep the GiScene1 alive while tracing).
struct GiScene1 {
    std::vector<rtrace::RtSphere> spheres;
    std::vector<rtrace::RtAabb>   aabbs;
    RtScene                       scene;
};

inline GiScene1 BuildGi1Scene() {
    GiScene1 r;
    uint32_t prim = 0;

    // The enclosure is roughly the cube [-4,4] x [-1,7] x [-4,4] (thin slabs as walls). primIndex picks the
    // rtrace albedo palette (mod 6): 0 warm-red, 1 cool-blue, 2 green, 3 amber, 4 violet, 5 grey.
    // Floor (primIndex 5 -> grey): a thin slab at y in [-1, 0].
    r.aabbs.push_back(rtrace::RtAabb{FxVec3{GiF(-4,1), GiF(-1,1), GiF(-4,1)},
                                     FxVec3{GiF(4,1),  GiF(0,1),  GiF(7,1)}, /*primIndex*/ 5});
    // Ceiling (grey): a thin slab at y in [6, 7].
    r.aabbs.push_back(rtrace::RtAabb{FxVec3{GiF(-4,1), GiF(6,1), GiF(-4,1)},
                                     FxVec3{GiF(4,1),  GiF(7,1), GiF(7,1)}, /*primIndex*/ 11});  // 11%6==5 grey
    // Back wall (cool-blue, primIndex 1): a thin slab at z in [6, 7].
    r.aabbs.push_back(rtrace::RtAabb{FxVec3{GiF(-4,1), GiF(-1,1), GiF(6,1)},
                                     FxVec3{GiF(4,1),  GiF(7,1),  GiF(7,1)}, /*primIndex*/ 1});
    // Left wall (warm-red, primIndex 0): a thin slab at x in [-4,-3].
    r.aabbs.push_back(rtrace::RtAabb{FxVec3{GiF(-4,1), GiF(-1,1), GiF(-4,1)},
                                     FxVec3{GiF(-3,1), GiF(7,1),  GiF(7,1)}, /*primIndex*/ 0});
    // Right wall (green, primIndex 2): a thin slab at x in [3,4].
    r.aabbs.push_back(rtrace::RtAabb{FxVec3{GiF(3,1),  GiF(-1,1), GiF(-4,1)},
                                     FxVec3{GiF(4,1),  GiF(7,1),  GiF(7,1)}, /*primIndex*/ 2});
    // A box occluder on the floor, off to the left (casts a probe-shadow — the per-ray shadow proof).
    r.aabbs.push_back(rtrace::RtAabb{FxVec3{GiF(-5,2), GiF(0,1), GiF(5,2)},
                                     FxVec3{GiF(-1,2), GiF(5,2), GiF(9,2)}, /*primIndex*/ 3});  // amber
    (void)prim;

    // A sphere resting on the floor near the box (amber, primIndex 4 -> violet).
    r.spheres.push_back(rtrace::RtSphere{FxVec3{GiF(3,2), GiF(1,1), GiF(7,2)}, GiF(1,1), /*primIndex*/ 4});

    r.scene.spheres = std::span<const rtrace::RtSphere>(r.spheres);
    r.scene.aabbs   = std::span<const rtrace::RtAabb>(r.aabbs);
    // Directional light angled DOWN into the box from the upper-front-right (TOWARD the light, unit).
    r.scene.lightDir = rtrace::RtNormalize(FxVec3{GiF(3,10), GiF(8,10), GiF(-2,10)});
    // Background (a probe ray escaping the box opening sees this dim sky-grey).
    r.scene.background = PackRGBA8(28, 32, 44, 255);
    return r;
}

// ===== The disabled / dispatch-sizing path (the probegi::ProbeDispatchGroups twin) ===================
// kGiProbeThreads = the compute workgroup size (one thread per probe). GiProbeDispatchGroups(grid) = the
// number of kGiProbeThreads workgroups to cover ProbeCount probes, or EXACTLY 0 when ProbeCount <= 0
// (nx/ny/nz == 0 -> ProbeCount == 0 -> 0 groups -> DispatchCompute(0) -> the radiance buffer untouched ==
// the cleared upload). The byte-identical no-op the GI1 proof rests on.
inline constexpr int kGiProbeThreads = 64;
inline int GiProbeDispatchGroups(const GiProbeGrid& grid) {
    int n = ProbeCount(grid);
    return (n <= 0) ? 0 : (n + kGiProbeThreads - 1) / kGiProbeThreads;
}

}  // namespace hf::render::gi
