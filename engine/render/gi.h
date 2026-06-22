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

// =====================================================================================================
// ===== Slice GI2 — INTEGER 3rd-order SH ENCODE (the crux of FLAGSHIP #29) =============================
// =====================================================================================================
// APPEND-ONLY (everything above this banner is GI1, BYTE-FROZEN). GI2 encodes the GI1 per-probe-per-ray
// radiance (GiRadiance[probe*16 + dir], TraceProbeRays) into a deterministic INTEGER 3rd-order spherical-
// harmonics record (9 coeffs x RGB, Q16.16) — the strict-zero twin of the float DDGI probe_sh.h. The
// later multi-bounce loop (GI4) re-injects these coeffs, so the dynamic range must be managed EXACTLY.
//
// THE CRUX (the "stay wide, narrow once" discipline, from rtrace::IntersectSphere's Q32.32 discriminant):
// each encode term is radiance(Q16.16) * basis(Q16.16) -> a Q32.32 product that OVERFLOWS int32 (a single
// ~1.0*~1.0 product is 65536^2 = 4.29e9 > 2^31). So we accumulate every term in an int64 (long), keep the
// 16-ray sum wide, and QUANTIZE TO Q16.16 EXACTLY ONCE at the final normalize (the / N then >> kFrac). No
// per-sample re-quantization. (This int64 accumulator is why gi_sh_encode.comp is Vulkan-SPIR-V-ONLY, the
// grain_integrate.comp/fluid_integrate.comp convention — glslc can't lower int64 to MSL; on Metal the
// --gi2-shencode showcase runs the CPU EncodeAllProbes, byte-identical by construction.)
//
// THE SCALE (the dynamic-range envelope, proven CPU-first in tests/gi_sh_test.cpp):
//   * Stored coeff convention: coeff_lm = (1/N) * Sum_j rad_j * Y_lm(dir_j)  -- a plain N-average of the
//     radiance-weighted basis. For rad in [0,1] this keeps every stored coeff in [-0.29, 0.29] (the DC
//     l=0 term is the largest at Y00=0.282 for a fully-lit probe) -- WELL within the [-2,2] Q16.16
//     headroom, AND the band-2 coeffs for a directional input stay ~0.003..0.03 (>>LSB -> NO underflow).
//   * Reconstruction (FxSHEvaluate, the cosine-lobe irradiance): irr(n) = Sum_l B_l * Sum_m coeff_lm *
//     Y_lm(n), with B_l = 4*A_l = {4*pi, 8*pi/3, pi} (A_l = {pi, 2pi/3, pi/4} the Ramamoorthi-Hanrahan
//     cosine-lobe band factors). The 4* folds the 4*pi full-sphere measure AND a 1/pi so a UNIFORM unit-
//     radiance field reconstructs to UNIT irradiance (irr(uniform R=1) ~= 1.002), and the worst-case
//     all-lit R=1 irradiance is ~1.02 -- both < 2. B_l exceed the +-32768 fx range (4*pi ~= 12.57), so
//     they are PLAIN int Q16.16 constants and the reconstruction multiply (coeff*basis*B_l, a triple
//     Q16.16 product) is done in int64 and narrowed once.
//
// THE BASIS TABLE is HOST-PRECOMPUTED (kGiSHBasis below): Y_lm evaluated in double at table-definition
// time on the SAME baked kGiProbeDirs the GPU reads, rounded to Q16.16 literals -- NO sqrt/cos/pow on the
// GPU hot path, both backends read identical bits (the probe_sh.h discipline, in integer). The directions
// are not exactly unit (|q| in [0.99999,1.00001]); the basis is evaluated on the decoded stored direction
// exactly, so CPU and GPU agree to the bit.

// ----- The 9-basis 3rd-order SH record (std430: 9 coeffs x 3 RGB channels). --------------------------
// FxProbeSH stores the integer SH projection of one probe's 16-ray radiance, per RGB channel. Laid out as
// a tight int32 array of 9*3 = 27 ints + 1 pad int -> 112 bytes, a clean std430 stride the GPU mirror
// (RWStructuredBuffer<FxProbeSH>) matches byte-for-byte. coeff[basis][channel]; channel 0=r,1=g,2=b.
struct FxProbeSH {
    fx coeff[9][3];   // Q16.16; coeff[basis][channel]
    fx _pad = 0;      // 27 ints -> pad to 28 ints (112 B) for a clean std430 stride
};
static_assert(sizeof(FxProbeSH) == 28 * 4, "FxProbeSH must be 112 bytes (9*3 coeffs + pad, std430)");

// ----- The host-precomputed integer SH basis table (Y_lm(dir) for the 16 baked dirs, Q16.16). --------
// GENERATOR (run once in double; see the GI2 design spec). For each baked kGiProbeDirs[k] decoded to
// double (x,y,z) = kGiProbeDirs[k]/65536:
//   Y[0] = kY00                          (kY00 = 0.5*sqrt(1/pi))
//   Y[1] = kY1*y, Y[2] = kY1*z, Y[3] = kY1*x   (kY1 = 0.5*sqrt(3/pi))
//   Y[4] = kY2a*x*y, Y[5] = kY2a*y*z, Y[6] = kY2b*(3z^2-1), Y[7] = kY2a*x*z, Y[8] = kY2c*(x^2-y^2)
//     (kY2a = 0.5*sqrt(15/pi), kY2b = 0.25*sqrt(5/pi), kY2c = 0.25*sqrt(15/pi))
//   literal = round(Y * 65536)           (Q16.16, round-to-nearest)
// These are the EXACT integer counterparts of probe_sh.h::SHBasis9; the shader reads an IDENTICAL array.
static constexpr fx kGiSHBasis[kGiRaysPerProbe][9] = {
    {   18487,       0,   30020,   11143,       0,       0,   33830,   23359,    4335 },  // i= 0
    {   18487,   12609,   26017,  -13764,  -12120,   22909,   20266,  -25007,    1064 },  // i= 1
    {   18487,  -23164,   22014,    2033,   -3289,  -35610,    8639,    3125,  -18591 },  // i= 2
    {   18487,   21010,   18012,   16108,   23634,   26427,   -1050,   20261,   -6353 },  // i= 3
    {   18487,   -5016,   14009,  -28354,    9931,   -4907,   -8801,  -27738,   27191 },  // i= 4
    {   18487,  -16326,   10007,   25665,  -29259,  -11408,  -14614,   17934,   13692 },  // i= 5
    {   18487,   30375,    6004,   -8166,  -17320,   12735,  -18490,   -3424,  -29886 },  // i= 6
    {   18487,  -28361,    2001,  -14730,   29173,   -3964,  -20427,   -2059,  -20509 },  // i= 7
    {   18487,   10963,   -2001,   30019,   22981,   -1532,  -20427,   -4195,   27268 },  // i= 8
    {   18487,   12001,   -6004,  -29074,  -24365,   -5032,  -18490,   12190,   24485 },  // i= 9
    {   18487,  -27550,  -10007,   12892,  -24803,   19251,  -14614,   -9009,  -20697 },  // i=10
    {   18487,   27474,  -14009,    8617,   16533,  -26877,   -8801,   -8430,  -23762 },  // i=11
    {   18487,  -13275,  -18012,  -22906,   21234,   16697,   -1050,   28811,   12167 },  // i=12
    {   18487,   -4993,  -22014,   22711,   -7919,    7676,    8639,  -34913,   17138 },  // i=13
    {   18487,   15271,  -26017,  -10736,  -11449,  -27744,   20266,   19505,   -4118 },  // i=14
    {   18487,  -11050,  -30020,   -1432,    1105,   23165,   33830,    3002,   -4192 },  // i=15
};

// The cosine-lobe RECONSTRUCTION band factors B_l = 4*A_l, baked Q16.16 (PLAIN int, NOT fx-bounded:
// 4*pi ~= 12.57 exceeds the +-32768 fx range, so the reconstruction multiply is widened to int64).
// A_l = {pi, 2pi/3, pi/4} (Ramamoorthi-Hanrahan); 4* folds the 4*pi measure + a 1/pi normalization so a
// uniform unit-radiance field reconstructs to unit irradiance.
inline constexpr int64_t kGiSHReconB0 = 823550;   // 4*pi    in Q16.16
inline constexpr int64_t kGiSHReconB1 = 549033;   // 8*pi/3  in Q16.16
inline constexpr int64_t kGiSHReconB2 = 205887;   // pi      in Q16.16  (== 4*(pi/4))

// ----- FxSHEncodeProbe — the int64-accumulator encode (THE CRUX), quantize once. ---------------------
// Encodes one probe's kGiRaysPerProbe radiances into an FxProbeSH. acc[i][c] (int64, Q32.32) accumulates
// (int64)rays[r].ch * (int64)kGiSHBasis[r][i] over the 16 rays -- STAYING WIDE; then the single normalize
// divides by N (the average) and narrows >> kFrac to Q16.16 EXACTLY ONCE. The shader copies this VERBATIM.
inline FxProbeSH FxSHEncodeProbe(const GiRadiance* rays, int rayCount) {
    int64_t acc[9][3] = {};
    for (int r = 0; r < rayCount; ++r) {
        const GiRadiance& rad = rays[r];
        int64_t ch[3] = {(int64_t)rad.r, (int64_t)rad.g, (int64_t)rad.b};
        for (int i = 0; i < 9; ++i) {
            int64_t b = (int64_t)kGiSHBasis[r][i];   // Q16.16 basis weight
            acc[i][0] += ch[0] * b;                  // Q32.32 term -- WIDE, no narrow yet
            acc[i][1] += ch[1] * b;
            acc[i][2] += ch[2] * b;
        }
    }
    FxProbeSH sh{};
    if (rayCount <= 0) return sh;
    for (int i = 0; i < 9; ++i)
        for (int c = 0; c < 3; ++c)
            // Quantize ONCE: average over N (Q32.32 / int stays Q32.32) then narrow >> kFrac -> Q16.16.
            // Arithmetic right shift on int64 (round-toward-negative-infinity, identical CPU<->DXC).
            sh.coeff[i][c] = (fx)((acc[i][c] / (int64_t)rayCount) >> kFrac);
    return sh;
}

// ----- FxSHEvaluate — the integer cosine-lobe irradiance reconstruction (probe_sh::SHEvaluate twin). --
// irr(n) = clamp(Sum_l B_l * Sum_m coeff_lm * Y_lm(n), >= 0), per RGB channel. The triple Q16.16 product
// coeff(Q16.16) * basis(Q16.16) * B_l(Q16.16) is accumulated in int64 and narrowed >> (2*kFrac) once.
// `normal` is a Q16.16 direction (need not be exactly unit -- the basis is evaluated on it directly).
inline GiRadiance FxSHEvaluate(const FxProbeSH& sh, const FxVec3& normal) {
    // SHBasis9 at `normal` (integer; the kGiSHBasis generator's formula, evaluated live in fxmul).
    // kY* constants baked Q16.16 (round-to-nearest): kY00=18487, kY1=32021, kY2a=71601, kY2b=20670, kY2c=35801.
    const fx x = normal.x, y = normal.y, z = normal.z;
    const fx kY00f = 18487, kY1f = 32021, kY2af = 71601, kY2bf = 20670, kY2cf = 35801;
    fx Y[9];
    Y[0] = kY00f;
    Y[1] = fxmul(kY1f, y);
    Y[2] = fxmul(kY1f, z);
    Y[3] = fxmul(kY1f, x);
    Y[4] = fxmul(kY2af, fxmul(x, y));
    Y[5] = fxmul(kY2af, fxmul(y, z));
    Y[6] = fxmul(kY2bf, (fx)(fxmul(fxmul((fx)(3 * (int64_t)kOne), z), z) - kOne));
    Y[7] = fxmul(kY2af, fxmul(x, z));
    Y[8] = fxmul(kY2cf, (fx)(fxmul(x, x) - fxmul(y, y)));

    const int64_t B[9] = {kGiSHReconB0,
                          kGiSHReconB1, kGiSHReconB1, kGiSHReconB1,
                          kGiSHReconB2, kGiSHReconB2, kGiSHReconB2, kGiSHReconB2, kGiSHReconB2};
    GiRadiance out{};
    fx* outc[3] = {&out.r, &out.g, &out.b};
    for (int c = 0; c < 3; ++c) {
        int64_t acc = 0;   // Q48.48-ish wide accumulator (coeff*basis*B, three Q16.16 -> Q48.48)
        for (int i = 0; i < 9; ++i)
            acc += (int64_t)sh.coeff[i][c] * (int64_t)Y[i] * B[i];   // WIDE triple product
        fx v = (fx)(acc >> (2 * kFrac));   // narrow ONCE: Q48.48 >> 32 -> Q16.16
        *outc[c] = (v < 0) ? 0 : v;        // clamp >= 0 (a surface receives no negative irradiance)
    }
    return out;
}

// ----- EncodeAllProbes — encode every probe's 16-ray radiance slice into out[probe]. -----------------
// `radiance` is the GI1 buffer (ProbeCount*kGiRaysPerProbe), `out` the SH buffer (ProbeCount). A 0-probe
// grid writes nothing (the no-op). One probe per output element, matching the shader's one-thread-per-probe.
inline void EncodeAllProbes(const GiProbeGrid& grid, std::span<const GiRadiance> radiance,
                            std::span<FxProbeSH> out) {
    int probes = ProbeCount(grid);
    for (int p = 0; p < probes; ++p)
        out[p] = FxSHEncodeProbe(radiance.data() + (size_t)p * kGiRaysPerProbe, kGiRaysPerProbe);
}

// ----- GiSHToImage — the deterministic SH-irradiance grid viz (the GI2 golden). ----------------------
// A flat 2D TILE GRID (the GiProbesToImage layout): each probe tile is colored by FxSHEvaluate(sh[p],
// evalDir) (e.g. evalDir = +Y) quantized to RGBA8 -- a SMOOTHER lighting grid than GI1's raw mean radiance.
// Strict integer -> identical both backends. `img` is row-major RGBA8 (top row first), size == w*h.
inline void GiSHToImage(const GiProbeGrid& grid, std::span<const FxProbeSH> sh, FxVec3 evalDir,
                        std::span<uint32_t> img, uint32_t w, uint32_t h) {
    const uint32_t kBg = PackRGBA8(18, 18, 22, 255);
    for (uint32_t i = 0; i < w * h; ++i) img[i] = kBg;

    int probes = ProbeCount(grid);
    if (probes <= 0) return;

    int cols = 1;
    while (cols * cols < probes) ++cols;
    int rows = (probes + cols - 1) / cols;
    uint32_t tileW = w / (uint32_t)cols;
    uint32_t tileH = h / (uint32_t)rows;
    if (tileW == 0) tileW = 1;
    if (tileH == 0) tileH = 1;

    auto q = [](fx v) -> int32_t {
        int64_t t = ((int64_t)v * 255) >> kFrac;   // Q16.16 [0,1+] -> [0,255]
        if (t < 0) t = 0; if (t > 255) t = 255;     // clamp (irradiance can be >1 within the headroom)
        return (int32_t)t;
    };

    for (int p = 0; p < probes; ++p) {
        int col = p % cols;
        int row = p / cols;
        GiRadiance irr = FxSHEvaluate(sh[p], evalDir);
        uint32_t color = PackRGBA8(q(irr.r), q(irr.g), q(irr.b), 255);
        uint32_t x0 = (uint32_t)col * tileW;
        uint32_t y0 = (uint32_t)row * tileH;
        uint32_t x1 = x0 + (tileW > 1 ? tileW - 1 : tileW);
        uint32_t y1 = y0 + (tileH > 1 ? tileH - 1 : tileH);
        for (uint32_t y = y0; y < y1 && y < h; ++y)
            for (uint32_t x = x0; x < x1 && x < w; ++x)
                img[(size_t)y * w + x] = color;
    }
}

// =====================================================================================================
// ===== Slice GI3 — INTEGER TRILINEAR SH INTERPOLATION (the continuous irradiance field) ==============
// =====================================================================================================
// APPEND-ONLY (everything above this banner is GI1+GI2, BYTE-FROZEN). GI3 makes the discrete probe volume
// CONTINUOUS: for any query point it trilinearly blends the 8 surrounding probes' integer SH (with
// partition-of-unity weights, Σ == kOne EXACTLY) and evaluates the cosine-lobe irradiance for a surface
// normal — a SMOOTH integer irradiance field, the piece a renderer samples for indirect light. The integer
// twin of the float DDGI probe_gi.h::NearestProbes (172) + probe_sh.h::InterpolateSH (180). Pure integer ->
// bit-exact GPU==CPU, strict-zero cross-vendor.
//
// THE PARTITION-OF-UNITY CRUX (a hard invariant the test pins, Σw == kOne EXACTLY): a query in the lattice
// falls in a cell with a per-axis fractional offset frac in [0,kOne]. The per-axis weights are wlo = kOne -
// frac and whi = frac — they sum to kOne EXACTLY by construction. The 8-corner weight is the triple product
// wx*wy*wz. Each raw product P_c = wx*wy*wz is a Q48.48 int64 (wx,wy,wz ≤ kOne, three Q16.16 -> Q48.48); the
// EXACT sum Σ_c P_c factorizes to (Σwx)(Σwy)(Σwz) = kOne^3 (each axis's lo+hi == kOne). We narrow the first
// 7 corners by >> (2*kFrac) (a FLOOR truncation, so the partial sum is always ≤ kOne) and assign the LAST
// corner the leftover (kOne - Σ_{c<7} w_c) — the remainder is provably ≥ 0 (each floor only LOSES bits) so
// the 8 weights sum to kOne EXACTLY (no rounding drift). Lattice-point identity: at a probe (all frac==0)
// corner 0 gets P=kOne^3 -> w=kOne, the rest P=0 -> w=0, remainder 0 -> corner 7 == 0; the blend is EXACTLY
// that probe's SH (a falsifiable proof the index/weight math is right).
//
// THE SH BLEND CRUX (the GI2 stay-wide-narrow-once discipline): the blended coeff is Σ_c weight_c *
// cornerSH_c. weight (≤kOne) * coeff (a Q16.16 in roughly [-2,2]) is a Q32.32 product that OVERFLOWS int32
// (a single ~1.0*~1.0 product is 65536^2 = 4.29e9 > 2^31), so we accumulate the 8-corner blend in an int64
// (Q32.32) and quantize to Q16.16 EXACTLY ONCE (>> kFrac) — NO per-corner re-quantize. This int64 product is
// why gi_interp.comp is VULKAN-SPIR-V-ONLY (the GI2 gi_sh_encode.comp convention — glslc can't lower int64
// to MSL); on Metal the --gi3-interp showcase runs the CPU InterpolateField (byte-identical by construction).
//
// OUT-OF-GRID queries CLAMP to the boundary cell (deterministic edge handling): the per-axis base is clamped
// to [0, dim-2] and frac to [0, kOne] (a query below the grid saturates frac->0 onto the low corner, above
// the grid frac->kOne onto the high corner). A dim==1 axis collapses to base 0, frac 0 (no second corner).

// ----- FxProbeWeights — the 8 corner probe linear-indices + integer trilinear weights (Σ w == kOne). --
struct FxProbeWeights {
    int32_t idx[8];   // the 8 cell-corner flat indices (cx-major; corner bit0=+x, bit1=+y, bit2=+z)
    fx      w[8];     // the 8 integer trilinear weights, Σ w == kOne EXACTLY (partition of unity)
};

// ----- FxNearestProbes — the floor-cell + 8-corner index + integer trilinear weights. ----------------
// The integer twin of probe_gi.h::NearestProbes (172). Per axis: g = (point - origin)/spacing (Q16.16);
// base = floor(g) = g >> kFrac (arithmetic shift, toward -inf), CLAMPED to [0, dim-2] so base & base+1 are
// BOTH valid probe indices; frac = g - (base<<kFrac), CLAMPED to [0, kOne] (out-of-grid saturates to the
// boundary). A dim==1 axis -> base 0, frac 0 (no second corner). The 8 corner weights are the wx*wy*wz
// triple products narrowed >> (2*kFrac), with the last corner taking the exact leftover so Σ w == kOne.
// PURE INTEGER, deterministic. A degenerate grid (spacing<=0 || any dim<=0) -> idx all 0, w[0]=kOne (the
// documented disabled fallback; FxInterpolateSH guards probeCount<=0 separately).
inline FxProbeWeights FxNearestProbes(const GiProbeGrid& grid, const FxVec3& point) {
    FxProbeWeights t{};
    if (grid.spacing <= 0 || grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        for (int c = 0; c < 8; ++c) { t.idx[c] = 0; t.w[c] = 0; }
        t.w[0] = kOne;
        return t;
    }
    // Per-axis FLOOR cell base + the [0,kOne] fractional position (the integer NearestProbes::axis).
    auto axis = [&](fx v, fx o, fx sp, int dim, int& base, fx& frac) {
        if (dim <= 1) { base = 0; frac = 0; return; }
        // g = (v - o) / sp in Q16.16 (the rtrace::fxdiv idiom, int64-widen + integer divide).
        fx g = (fx)((((int64_t)(v - o)) << kFrac) / (int64_t)sp);
        int b = (int)(g >> kFrac);                       // floor toward -inf (arithmetic shift)
        if (b < 0) b = 0;
        if (b > dim - 2) b = dim - 2;
        fx fr = (fx)(g - ((int64_t)b << kFrac));          // g - base, the fractional remainder
        if (fr < 0) fr = 0;
        if (fr > kOne) fr = kOne;
        base = b; frac = fr;
    };
    int bx, by, bz;
    fx fxr, fyr, fzr;
    axis(point.x, grid.origin.x, grid.spacing, grid.nx, bx, fxr);
    axis(point.y, grid.origin.y, grid.spacing, grid.ny, by, fyr);
    axis(point.z, grid.origin.z, grid.spacing, grid.nz, bz, fzr);

    // Per-axis lo/hi weights — wlo + whi == kOne EXACTLY (the partition-of-unity factor).
    const fx wlo[3] = {(fx)(kOne - fxr), (fx)(kOne - fyr), (fx)(kOne - fzr)};
    const fx whi[3] = {fxr, fyr, fzr};

    fx accum = 0;   // running sum of the first 7 corner weights (to take the exact leftover at corner 7)
    for (int c = 0; c < 8; ++c) {
        int sx = (c & 1), sy = ((c >> 1) & 1), sz = ((c >> 2) & 1);
        // A dim==1 axis (frac 0, no second probe) clamps the +offset corner back onto the single valid
        // index; its +offset weight is already 0 there (whi==0) so the blend is unchanged.
        int cx = bx + sx; if (cx > grid.nx - 1) cx = grid.nx - 1;
        int cy = by + sy; if (cy > grid.ny - 1) cy = grid.ny - 1;
        int cz = bz + sz; if (cz > grid.nz - 1) cz = grid.nz - 1;
        t.idx[c] = ProbeFlatIndex(grid, cx, cy, cz);
        fx wx = sx ? whi[0] : wlo[0];
        fx wy = sy ? whi[1] : wlo[1];
        fx wz = sz ? whi[2] : wlo[2];
        if (c < 7) {
            // P_c = wx*wy*wz (Q48.48 int64); narrow >> (2*kFrac) -> Q16.16 (a FLOOR truncation so the
            // partial sum never exceeds kOne — the leftover at corner 7 is provably >= 0).
            int64_t p = (int64_t)wx * (int64_t)wy * (int64_t)wz;
            fx wc = (fx)(p >> (2 * kFrac));
            t.w[c] = wc;
            accum += wc;
        } else {
            t.w[c] = (fx)(kOne - accum);   // the EXACT leftover -> Σ w == kOne (partition of unity)
        }
    }
    return t;
}

// ----- FxInterpolateSH — the integer 8-corner SH blend (the probe_sh.h::InterpolateSH integer twin). --
// out.coeff[i][c] = Σ_corner weight_corner * shBuffer[idx_corner].coeff[i][c]. SH projection is LINEAR, so
// blend-then-evaluate == evaluate-then-blend; we blend the compact 9x3 SH once. weight(≤kOne, Q16.16) *
// coeff(Q16.16) is a Q32.32 product that OVERFLOWS int32, so the 8-corner sum is accumulated in an int64
// (STAY WIDE) and quantized to Q16.16 EXACTLY ONCE (>> kFrac) — the GI2 narrow-once discipline. The corner
// indices are clamped in-range by FxNearestProbes so every read is valid for probeCount>0. The shader
// copies this VERBATIM. A degenerate/empty buffer -> a zeroed SH (the disabled fallback).
inline FxProbeSH FxInterpolateSH(const FxProbeWeights& wts, std::span<const FxProbeSH> shBuffer) {
    FxProbeSH out{};
    if (shBuffer.empty()) return out;
    int64_t acc[9][3] = {};
    for (int c = 0; c < 8; ++c) {
        int64_t wc = (int64_t)wts.w[c];                 // Q16.16 weight
        const FxProbeSH& src = shBuffer[(size_t)wts.idx[c]];
        for (int i = 0; i < 9; ++i) {
            acc[i][0] += wc * (int64_t)src.coeff[i][0];   // Q32.32 term — WIDE, no narrow yet
            acc[i][1] += wc * (int64_t)src.coeff[i][1];
            acc[i][2] += wc * (int64_t)src.coeff[i][2];
        }
    }
    for (int i = 0; i < 9; ++i)
        for (int c = 0; c < 3; ++c)
            out.coeff[i][c] = (fx)(acc[i][c] >> kFrac);   // narrow ONCE: Q32.32 >> 16 -> Q16.16
    return out;
}

// ----- FxInterpolateIrradiance — the continuous irradiance at `point` for `normal`. ------------------
// = FxSHEvaluate(FxInterpolateSH(FxNearestProbes(grid,point), shBuffer), normal): trilinearly blend the 8
// surrounding probes' SH, then reconstruct the cosine-lobe irradiance for the surface normal. The diffuse
// indirect a renderer samples. `normal` is a Q16.16 direction (need not be exactly unit — FxSHEvaluate
// evaluates the basis on it directly). Pure integer, deterministic.
inline GiRadiance FxInterpolateIrradiance(const GiProbeGrid& grid, std::span<const FxProbeSH> shBuffer,
                                          const FxVec3& point, const FxVec3& normal) {
    return FxSHEvaluate(FxInterpolateSH(FxNearestProbes(grid, point), shBuffer), normal);
}

// ----- InterpolateField — evaluate the irradiance field at a set of query points/normals. ------------
// One query per output element (matching the shader's one-thread-per-query). `points`/`normals`/`out` are
// parallel spans; out[i] = FxInterpolateIrradiance(grid, shBuffer, points[i], normals[i]). An empty query
// set writes nothing (the no-op — the shader's 0-dispatch). PURE INTEGER, deterministic.
inline void InterpolateField(const GiProbeGrid& grid, std::span<const FxProbeSH> shBuffer,
                             std::span<const FxVec3> points, std::span<const FxVec3> normals,
                             std::span<GiRadiance> out) {
    size_t n = points.size();
    for (size_t i = 0; i < n; ++i)
        out[i] = FxInterpolateIrradiance(grid, shBuffer, points[i], normals[i]);
}

// ----- GiInterpDispatchGroups — the dispatch-sizing / no-op path (one thread per query point). -------
// kGiInterpThreads = the compute workgroup size. GiInterpDispatchGroups(n) = the number of workgroups to
// cover `n` query points, or EXACTLY 0 when n<=0 (an empty query set -> 0 groups -> DispatchCompute(0) ->
// the output buffer untouched == the cleared upload). The byte-identical no-op the GI3 determinism proof
// rests on. Pure integer.
inline constexpr int kGiInterpThreads = 64;
inline int GiInterpDispatchGroups(int queryCount) {
    return (queryCount <= 0) ? 0 : (queryCount + kGiInterpThreads - 1) / kGiInterpThreads;
}

// ----- GiFieldToImage — the SMOOTH interpolated-irradiance-field viz (the GI3 golden). ---------------
// A 2D SLICE through the continuous probe volume: a horizontal plane at a fixed world y (the grid's vertical
// midplane), sampled on a w x h lattice spanning the grid's XZ footprint. Each pixel = FxInterpolateIrradiance
// at that world point for a fixed surface normal (+Y, the up-facing receiver) quantized to RGBA8 — a SMOOTH
// irradiance field, MUCH smoother than the discrete GI1/GI2 probe-tile grids (the continuous blend the
// renderer samples). PINNED slice + resolution (the cross-vendor golden rests on it). Strict integer ->
// identical both backends. `img` is row-major RGBA8 (top row first), size == w*h. A 0-probe / empty-SH grid
// -> the image is left as the background fill.
inline void GiFieldToImage(const GiProbeGrid& grid, std::span<const FxProbeSH> shBuffer,
                           std::span<uint32_t> img, uint32_t w, uint32_t h) {
    const uint32_t kBg = PackRGBA8(18, 18, 22, 255);
    for (uint32_t i = 0; i < w * h; ++i) img[i] = kBg;

    int probes = ProbeCount(grid);
    if (probes <= 0 || shBuffer.empty() || w == 0 || h == 0) return;

    // The XZ footprint of the lattice: x in [origin.x, origin.x + (nx-1)*spacing], z analogous. The slice
    // y is the vertical midplane: origin.y + ((ny-1)*spacing)/2 (integer halve). The receiver normal is +Y.
    const fx spanX = fxmul((fx)((int64_t)(grid.nx - 1) * kOne), grid.spacing);
    const fx spanZ = fxmul((fx)((int64_t)(grid.nz - 1) * kOne), grid.spacing);
    const fx sliceY = (fx)(grid.origin.y + (fxmul((fx)((int64_t)(grid.ny - 1) * kOne), grid.spacing) >> 1));
    const FxVec3 nrm{0, kOne, 0};

    auto q = [](fx v) -> int32_t {
        int64_t t = ((int64_t)v * 255) >> kFrac;   // Q16.16 [0,1+] -> [0,255]
        if (t < 0) t = 0; if (t > 255) t = 255;     // clamp (irradiance can exceed 1 within the headroom)
        return (int32_t)t;
    };

    for (uint32_t py = 0; py < h; ++py) {
        // u,v in [0,kOne] across the image (integer: pixel center / (dim-1) so the corners hit the lattice
        // bounds exactly; a 1px image collapses to u=0).
        fx vfr = (h > 1) ? (fx)(((int64_t)py * kOne) / (int64_t)(h - 1)) : 0;
        fx z = (fx)(grid.origin.z + (int64_t)(((int64_t)spanZ * vfr) >> kFrac));
        for (uint32_t px = 0; px < w; ++px) {
            fx ufr = (w > 1) ? (fx)(((int64_t)px * kOne) / (int64_t)(w - 1)) : 0;
            fx x = (fx)(grid.origin.x + (int64_t)(((int64_t)spanX * ufr) >> kFrac));
            GiRadiance irr = FxInterpolateIrradiance(grid, shBuffer, FxVec3{x, sliceY, z}, nrm);
            img[(size_t)py * w + px] = PackRGBA8(q(irr.r), q(irr.g), q(irr.b), 255);
        }
    }
}

// =====================================================================================================
// ===== Slice GI4 — INTEGER MULTI-BOUNCE FEEDBACK (light that bounces, deterministically) =============
// =====================================================================================================
// APPEND-ONLY (everything above this banner is GI1+GI2+GI3, BYTE-FROZEN). GI4 closes the loop: MULTI-BOUNCE.
// Each probe-ray hit's shade gains an INDIRECT term = the PREVIOUS iteration's GI3-interpolated irradiance
// at the hit point × the hit's albedo, fed back for a fixed K<=3 iterations — the deterministic integer twin
// of the float DDGI probe_multibounce.h (ClampBounceCount/BounceIndirect). This is the step that makes light
// BOUNCE (a red wall tints the floor). STRICT INTEGER -> byte-identical HW==CPU, strict-zero cross-vendor.
//
// THE BOUNDED-GAIN ENVELOPE (why the feedback CANNOT overflow the GI2 [-2,2] SH headroom): the bounce energy
// at iteration k is Σ albedo^k. Because every albedo is < kOne (energy-conserving, AlbedoFor channels are at
// most 0.82) and K is fixed (<=3), the series is bounded (< radiance/(1-maxAlbedo)). GI2 proved a [0,1]
// radiance encodes to coeffs in [-0.29,0.29] and reconstructs to irradiance < ~1.02; the bounded feedback
// keeps the re-injected radiance within [0,~2), so the dynamic-range guarantee carries through (the GI2 crux
// re-verified under feedback). The ONLY new ingredient is the indirect shade.
//
// THE NO-OP CONTRACT (the make-or-break, falsifiable): ShadeHitGI(hit, scene, occluded, indirectIrr) with
// indirectIrr == {0,0,0} is BYTE-IDENTICAL to UnpackRadiance(ShadeHitShadowed(hit, scene, occluded)) — the
// indirect term is a literal integer +0. So K==1 BounceProbes (iteration 1 traces with indirectIrr=0) is the
// GI1+GI2 single-bounce SH EXACTLY. THE MONOTONICITY GUARD: SH_2's reconstructed irradiance >= SH_1's
// component-wise (the bounce only ADDS non-negative light) AND is measurably greater for >=1 probe (so the
// integer indirect did NOT truncate to zero — the underflow guard).

// ----- kGiMaxBounces / ClampBounces — the fixed-K clamp (the probemb::ClampBounceCount integer twin). ----
// K is fixed (the float DDGI posture: a 2nd bounce is the demonstrable leap; N-bounce is a trivial loop
// extension). bounces<1 clamps to 1 (the single-bounce path); >kGiMaxBounces clamps to kGiMaxBounces.
inline constexpr int kGiMaxBounces = 3;
inline int ClampBounces(int bounces) {
    if (bounces < 1) return 1;
    if (bounces > kGiMaxBounces) return kGiMaxBounces;
    return bounces;
}

// ----- ShadeHitGI — the direct shadowed shade + an indirect (albedo × interpolated irradiance) term. -----
// = albedo(hit) × (directDiffuse + indirectIrr), per RGB channel, as a GiRadiance (the unpacked RGBA8, the
// GI1 radiance convention). directDiffuse = ambient + (occluded ? 0 : lambert) is the ShadeHitShadowed body
// VERBATIM (the SAME ambient/lambert/quantize integer math), and indirectIrr is the GI3-interpolated
// irradiance of the PREVIOUS SH at the hit (passed in; {0,0,0} on iteration 1). PINNED per-channel integer
// blend (the shader copies this VERBATIM):
//   lit_ch  = fxmul(alb.ch, diffuse) + fxmul(alb.ch, indirectIrr.ch)   // Q16.16, direct + indirect
//   byte_ch = (lit_ch * 255) >> kFrac                                   // [0,255], clamped
//   rad.ch  = byte_ch * kOne / 255                                      // UnpackRadiance (Q16.16 in [0,1])
// When indirectIrr == {0,0,0}: lit_ch == fxmul(alb.ch, diffuse) == ShadeHitShadowed's q, so byte_ch is
// identical and rad == UnpackRadiance(ShadeHitShadowed(...)) BYTE-FOR-BYTE (the no-op contract). A MISS ->
// UnpackRadiance(scene.background) (indirectIrr irrelevant — no surface to receive indirect light). PURE
// INTEGER, deterministic, cross-backend exact.
inline GiRadiance ShadeHitGI(const RtHit& hit, const RtScene& scene, bool occluded,
                             const GiRadiance& indirectIrr) {
    if (hit.primIndex == kRtMiss) return UnpackRadiance(scene.background);
    fx ndl = rtrace::FxDot(hit.normal, scene.lightDir);
    if (ndl < 0) ndl = 0;                          // clamp max(dot,0)
    const fx ambient = (fx)(kOne * 18 / 100);      // 0.18 ambient floor (== ShadeHitShadowed)
    fx lambert = occluded ? 0 : fxmul(kOne - ambient, ndl);  // the GATED diffuse term
    fx diffuse = ambient + lambert;                // [ambient, 1]; occluded -> exactly ambient
    FxVec3 alb = rtrace::AlbedoFor(hit.primIndex);
    fx albc[3] = {alb.x, alb.y, alb.z};
    fx indc[3] = {indirectIrr.r, indirectIrr.g, indirectIrr.b};
    auto unpackCh = [](int32_t byteVal) -> fx {
        if (byteVal < 0) byteVal = 0;
        if (byteVal > 255) byteVal = 255;
        return (fx)(((int64_t)byteVal * (int64_t)kOne) / 255);   // == UnpackRadiance per-channel
    };
    auto chOut = [&](int i) -> fx {
        // lit = albedo*diffuse (the ShadeHitShadowed term) + albedo*indirect (the GI4 bounce). Both Q16.16.
        fx lit = (fx)(fxmul(albc[i], diffuse) + fxmul(albc[i], indc[i]));
        int32_t byteVal = (int32_t)(((int64_t)lit * 255) >> kFrac);   // [0,255], unclamped (clamped in unpack)
        return unpackCh(byteVal);
    };
    return GiRadiance{chOut(0), chOut(1), chOut(2), 0};
}

// ----- BounceProbes — the fixed-K multi-bounce iteration (the pure-CPU reference, returns SH_K). ---------
// SH_0 is undefined-zero. Iteration 1: TraceProbeRays direct (indirectIrr = {0,0,0}) -> EncodeAllProbes ->
// SH_1 (== the GI1+GI2 single-bounce SH, the no-op contract). Iteration k>1: re-trace each probe-ray, but
// shade each hit with ShadeHitGI(..., FxInterpolateIrradiance(SH_{k-1}, hit.pos, hit.normal)) -> the previous
// iteration's interpolated indirect feeds the new radiance -> EncodeAllProbes -> SH_k. Returns SH_K (clamp
// K<=kGiMaxBounces). The HOST drives the SAME loop on the GPU (dispatch gi_bounce.comp reading SH_{k-1}, then
// gi_sh_encode.comp to make SH_k). `outSH` must be ProbeCount(grid) long; a 0-probe grid leaves it untouched.
// PURE INTEGER, deterministic.
inline void BounceProbes(const RtScene& scene, const GiProbeGrid& grid, int bounces,
                         std::span<FxProbeSH> outSH) {
    int K = ClampBounces(bounces);
    int probes = ProbeCount(grid);
    if (probes <= 0) return;

    std::vector<GiRadiance> radiance((size_t)probes * kGiRaysPerProbe);

    // ----- Iteration 1: the DIRECT capture (indirectIrr == {0,0,0}) -> SH_1. -----
    // Byte-identical to TraceProbeRays + EncodeAllProbes (ShadeHitGI with a zero indirect == the unpacked
    // ShadeHitShadowed) — the no-op contract the K==1 proof rests on.
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
            radiance[(size_t)p * kGiRaysPerProbe + d] = ShadeHitGI(hit, scene, occluded, GiRadiance{});
        }
    }
    EncodeAllProbes(grid, std::span<const GiRadiance>(radiance), outSH);

    // ----- Iterations 2..K: re-trace with the PREVIOUS iteration's GI3 interpolated indirect. -----
    // outSH holds SH_{k-1}; re-shade each hit with ShadeHitGI(..., FxInterpolateIrradiance(SH_{k-1}, hit)),
    // re-encode into outSH -> SH_k. The host GPU loop mirrors this exactly (gi_bounce.comp + gi_sh_encode.comp).
    for (int k = 2; k <= K; ++k) {
        std::span<const FxProbeSH> prevSH(outSH.data(), (size_t)probes);
        for (int p = 0; p < probes; ++p) {
            FxVec3 origin = ProbePos(grid, p);
            for (int d = 0; d < kGiRaysPerProbe; ++d) {
                RtRay ray{origin, kGiProbeDirs[d]};
                RtHit hit = TraceClosest(ray, scene);
                bool occluded = false;
                GiRadiance indirect{};
                if (hit.primIndex != kRtMiss) {
                    RtRay shadowRay;
                    shadowRay.origin = FxAdd(hit.pos, FxScale(hit.normal, kRtShadowEps));
                    shadowRay.dir = scene.lightDir;
                    occluded = TraceAnyHit(shadowRay, scene, kRtShadowMinT);
                    // The indirect = the previous SH's interpolated irradiance at the hit point for the hit
                    // normal (the GI3 continuous field). A MISS receives no indirect (the {} above).
                    indirect = FxInterpolateIrradiance(grid, prevSH, hit.pos, hit.normal);
                }
                radiance[(size_t)p * kGiRaysPerProbe + d] = ShadeHitGI(hit, scene, occluded, indirect);
            }
        }
        EncodeAllProbes(grid, std::span<const GiRadiance>(radiance), outSH);
    }
}

// ----- GiMaxIrradiance — the max reconstructed-irradiance magnitude across the SH buffer (the in-range
// proof helper). For each probe, reconstruct the +Y irradiance and track the largest channel (Q16.16). The
// GI2 headroom holds under feedback iff this stays < 2*kOne. Pure integer, deterministic.
inline fx GiMaxIrradiance(std::span<const FxProbeSH> sh, const FxVec3& normal) {
    fx maxIrr = 0;
    for (const FxProbeSH& s : sh) {
        GiRadiance irr = FxSHEvaluate(s, normal);
        if (irr.r > maxIrr) maxIrr = irr.r;
        if (irr.g > maxIrr) maxIrr = irr.g;
        if (irr.b > maxIrr) maxIrr = irr.b;
    }
    return maxIrr;
}

// =====================================================================================================
// ===== Slice GI5 — INTEGER CHEBYSHEV OCCLUSION WEIGHTING (the probe-volume light-leak fix) ============
// =====================================================================================================
// APPEND-ONLY (everything above this banner is GI1+GI2+GI3+GI4, BYTE-FROZEN). GI5 fixes the classic
// probe-volume artifact — LIGHT LEAK: the unoccluded GI3 trilinear blend (FxInterpolateIrradiance) lets a
// probe on the FAR side of a wall contribute to a query point in FRONT of it, so light bleeds through
// geometry. The fix is the variance-shadow-map CHEBYSHEV VISIBILITY (Majercik et al. 2019 / DDGI): capture
// per-probe distance MOMENTS (mean + mean-squared distance — which the GI1 RT probe trace yields FOR FREE
// from rtrace::TraceClosest's RtHit.t, NO separate distance-cube pass) and weight each of the 8 trilinear
// corners by the probability the query point is VISIBLE to that corner probe. STRICT INTEGER -> byte-exact
// GPU==CPU, strict-zero cross-vendor. The integer twin of probe_dist.h::MomentsFromDistance (167) +
// ChebyshevVisibility (294) — but in Q16.16, accumulated wide and quantized once (the GI2 discipline).
//
// THE MOMENTS COME FREE (the design win, vs the float DDGI's separate distance-cube render pass): the GI1
// probe trace already calls rtrace::TraceClosest, whose RtHit.t IS the per-ray hit distance (in units of
// |dir|; the 16 kGiProbeDirs are unit to < 2e-5, so t is the world distance to the bit). Per probe:
// meanDist = (1/16) Σ t_d, meanDist2 = (1/16) Σ t_d² — accumulate t and t² in int64 (t² is a Q16.16*Q16.16
// = Q32.32 product overflowing int32; meanDist2 = (Σ t²)/16 >> kFrac, the GI2 stay-wide-narrow-once). A
// ray that MISSES (primIndex == kRtMiss) sees no occluder -> its distance is a LARGE SENTINEL (kGiMomFar)
// so the probe reads as fully visible at any realistic query distance (no spurious occlusion from open
// directions). The means are quantized EXACTLY ONCE.
//
// THE CHEBYSHEV INEQUALITY (the integer VSM, FxChebyshevVisibility): for a query at distance d from probe p
// with moments {meanDist, meanDist2}:
//   * d <= meanDist                 -> vis = kOne (the query is at/closer than the average occluder -> fully
//                                       visible; the surface in front of the wall is NOT behind the probe's
//                                       average occluder).
//   * else: variance = meanDist2 - meanDist² (>= 0 by Jensen; a small bias kGiVarFloor avoids div-by-zero +
//     softens a near-flat face); vis = variance / (variance + (d - meanDist)²)  (Chebyshev's upper bound on
//     P(dist >= d)), clamped to [0,kOne]. The divide is fxdiv (int64-widen + integer divide, the rtrace
//     idiom). variance ↓ as the occluder distance is sharply defined -> vis ↓ for a query BEYOND the wall.
//
// THE OCCLUSION-WEIGHTED BLEND (FxInterpolateIrradianceOcc): take the GI3 FxNearestProbes 8 corners; scale
// each corner weight w_c *= lerp(kOne, FxChebyshevVisibility(moments[corner], distToCorner), occStrength);
// then RE-NORMALIZE so Σ w_c == kOne EXACTLY (partition of unity preserved — no energy gain/loss, the GI3
// invariant), then FxInterpolateSH + FxSHEvaluate VERBATIM. occStrength == 0 -> every lerp factor is kOne
// -> every weight unchanged -> Σ already kOne -> the re-normalize is the identity -> BYTE-IDENTICAL to
// FxInterpolateIrradiance (the falsifiable NO-OP contract). All integer -> strict-zero.

// ----- FxProbeMoments — the per-probe distance moments {meanDist, meanDist2} (Q16.16). ----------------
// The integer twin of probe_dist.h::ProbeDistMoments. meanDist = avg hit distance over the 16 probe rays;
// meanDist2 = avg of the squared hit distance (so variance = meanDist2 - meanDist² >= 0). std430-tight
// (two contiguous fx == 8 bytes), a clean GPU mirror. A miss contributes the kGiMomFar sentinel.
struct FxProbeMoments {
    fx meanDist = 0;    // (1/16) Σ_d hitDist_d   (Q16.16)
    fx meanDist2 = 0;   // (1/16) Σ_d hitDist_d²  (Q16.16)
};
static_assert(sizeof(FxProbeMoments) == 8, "FxProbeMoments must be std430-tight (two fx == 8 bytes)");

// The far / miss-sentinel distance a probe ray writes where it hits NO geometry (no occluder in that
// direction). Large enough that any realistic query distance d <= meanDist -> fully visible, but small
// enough that meanDist² + the int64 t² accumulation never overflow (16 * (256*kOne)² fits int64 easily).
// 256 world units in Q16.16 (256 << kFrac).
inline constexpr fx kGiMomFar = (fx)(256 * (int64_t)kOne);

// A small integer variance floor (the band-softener + div-by-zero guard, the probe_dist.h kVarFloorAbs
// integer analog): a near-flat face has ~0 variance, so var/(var+dd²) would collapse to a hard step; we
// clamp variance UP to at least this. kOne/4096 ~= 16 LSB of Q16.16 (a tiny absolute floor; the transition
// stays sharp but never divides by zero).
inline constexpr fx kGiVarFloor = (fx)(kOne / 4096);

// ----- FxProbeMoments_All — trace the 16 probe rays per probe, fold t/t² into the moments. ------------
// Per probe p: cast the 16 baked kGiProbeDirs via rtrace::TraceClosest; a HIT contributes its RtHit.t, a
// MISS contributes kGiMomFar (no occluder -> far). Accumulate t (int64, Q16.16) and t² (int64, Q32.32 —
// a single ~dist² product overflows int32 for dist>~256, so STAY WIDE) over the 16 rays; quantize the two
// means EXACTLY ONCE: meanDist = (Σt)/16, meanDist2 = ((Σt²)/16) >> kFrac (Q32.32 average -> Q16.16). PURE
// INTEGER, deterministic. `out` must be ProbeCount(grid) long; a 0-probe grid writes nothing (the no-op).
inline void FxProbeMoments_All(const GiProbeGrid& grid, const RtScene& scene, std::span<FxProbeMoments> out) {
    int probes = ProbeCount(grid);
    for (int p = 0; p < probes; ++p) {
        FxVec3 origin = ProbePos(grid, p);
        int64_t sumT = 0, sumT2 = 0;
        for (int d = 0; d < kGiRaysPerProbe; ++d) {
            RtRay ray{origin, kGiProbeDirs[d]};
            RtHit hit = TraceClosest(ray, scene);
            int64_t dist = (hit.primIndex != kRtMiss) ? (int64_t)hit.t : (int64_t)kGiMomFar;
            sumT  += dist;
            sumT2 += dist * dist;   // Q32.32 term — WIDE, no narrow yet
        }
        FxProbeMoments m{};
        m.meanDist  = (fx)(sumT / kGiRaysPerProbe);                       // Q16.16 average, quantize once
        m.meanDist2 = (fx)((sumT2 / kGiRaysPerProbe) >> kFrac);           // Q32.32 average -> Q16.16, once
        out[p] = m;
    }
}

// ----- FxChebyshevVisibility — the integer variance-shadow (Chebyshev) visibility weight. -------------
// vis(m, d) in [0,kOne]: the probability the surface at distance d from the probe is VISIBLE to it (not
// behind the probe's average occluder). d <= meanDist -> kOne (in front of / at the occluder). Else
// variance = meanDist2 - meanDist² (clamped >= kGiVarFloor — the bias avoids div0 + softens a flat face);
// vis = variance / (variance + (d-meanDist)²) via fxdiv (int64). The integer twin of
// probe_dist.h::ChebyshevVisibility. PURE INTEGER, deterministic, no div-by-zero (denom >= kGiVarFloor>0).
inline fx FxChebyshevVisibility(const FxProbeMoments& m, fx queryDist) {
    if (queryDist <= m.meanDist) return kOne;                 // at/closer than the average occluder
    fx variance = (fx)(m.meanDist2 - fxmul(m.meanDist, m.meanDist));   // meanDist2 - meanDist² (Q16.16)
    if (variance < kGiVarFloor) variance = kGiVarFloor;       // bias: avoid div0 + soften a near-flat face
    fx dd = (fx)(queryDist - m.meanDist);                     // > 0 here
    fx dd2 = fxmul(dd, dd);                                   // (d - meanDist)² (Q16.16)
    fx denom = (fx)(variance + dd2);                          // > 0 strictly (variance >= kGiVarFloor)
    fx vis = rtrace::fxdiv(variance, denom);                  // Chebyshev upper bound, Q16.16 in [0,1]
    if (vis < 0) vis = 0;
    if (vis > kOne) vis = kOne;
    return vis;
}

// ----- GiFxLerp — the integer lerp a + (b-a)*t, t in [0,kOne] (the occStrength blend). ----------------
// lerp(a, b, t) = a + fxmul(b - a, t). t == 0 -> a EXACTLY (a + fxmul(b-a, 0) == a + 0); t == kOne -> b
// (within Q16.16). PURE INTEGER, deterministic. (Named GiFxLerp to avoid colliding with sim lerp helpers.)
inline fx GiFxLerp(fx a, fx b, fx t) {
    return (fx)(a + fxmul((fx)(b - a), t));
}

// ----- FxInterpolateIrradianceOcc — the GI3 blend with Chebyshev occlusion weighting (the leak fix). --
// = FxNearestProbes(grid, point) corners, each weight scaled by lerp(kOne, FxChebyshevVisibility(moments
// [corner], distToCorner), occStrength), RE-NORMALIZED so Σ == kOne, then FxInterpolateSH + FxSHEvaluate
// (the GI3 path VERBATIM). distToCorner = the Q16.16 length from `point` to that corner PROBE's world
// position (fpx::FxLength via the sum-of-squares int64 sqrt — the SAME integer length the RT uses). When
// occStrength == 0: every visibility factor is lerp(kOne, vis, 0) == kOne -> every weight unchanged ->
// Σ w == kOne already -> the re-normalize is a pure identity -> the result is BYTE-IDENTICAL to
// FxInterpolateIrradiance (the no-op contract). PURE INTEGER, deterministic, cross-backend exact.
//
// THE RE-NORMALIZE (partition of unity preserved, no energy gain/loss): scale each weight, sum them (int64,
// Q16.16), then w_c <- fxdiv(w_c, sumW) for the first 7 corners and assign corner 7 the EXACT leftover
// (kOne - Σ_{c<7}) so Σ == kOne EXACTLY (the GI3 leftover discipline). A degenerate sumW <= 0 (all corners
// fully occluded — impossible since the corner AT/nearest the point has d<=meanDist -> vis kOne, but
// guarded) falls back to the unweighted GI3 weights.
inline GiRadiance FxInterpolateIrradianceOcc(const GiProbeGrid& grid, std::span<const FxProbeSH> shBuffer,
                                             std::span<const FxProbeMoments> moments, const FxVec3& point,
                                             const FxVec3& normal, fx occStrength) {
    FxProbeWeights t = FxNearestProbes(grid, point);
    // Scale each corner weight by its lerped Chebyshev visibility. occStrength==0 -> factor kOne -> unchanged.
    fx scaled[8];
    int64_t sumW = 0;
    bool haveMoments = !moments.empty() && !shBuffer.empty();
    for (int c = 0; c < 8; ++c) {
        fx w = t.w[c];
        if (haveMoments && occStrength != 0) {
            FxVec3 cornerPos = ProbePos(grid, t.idx[c]);
            FxVec3 delta{(fx)(point.x - cornerPos.x), (fx)(point.y - cornerPos.y),
                         (fx)(point.z - cornerPos.z)};
            int64_t sx = (int64_t)delta.x * (int64_t)delta.x;
            int64_t sy = (int64_t)delta.y * (int64_t)delta.y;
            int64_t sz = (int64_t)delta.z * (int64_t)delta.z;
            fx distToCorner = (fx)rtrace::FxISqrt(sx + sy + sz);   // Q16.16 length (the RT integer sqrt)
            fx vis = FxChebyshevVisibility(moments[(size_t)t.idx[c]], distToCorner);
            fx factor = GiFxLerp(kOne, vis, occStrength);          // lerp(kOne, vis, occStrength)
            w = fxmul(w, factor);
        }
        scaled[c] = w;
        sumW += (int64_t)w;
    }
    // Re-normalize so Σ == kOne EXACTLY (partition of unity). occStrength==0 -> scaled==t.w, sumW==kOne ->
    // fxdiv(w, kOne) == w and the leftover corner is the exact original -> a TRUE byte-identity no-op.
    FxProbeWeights norm = t;
    if (sumW > 0) {
        int64_t accum = 0;
        for (int c = 0; c < 7; ++c) {
            fx w = rtrace::fxdiv(scaled[c], (fx)sumW);   // w / sumW in Q16.16 (== w when sumW==kOne)
            norm.w[c] = w;
            accum += (int64_t)w;
        }
        norm.w[7] = (fx)(kOne - accum);                  // exact leftover -> Σ == kOne
    }
    // Blend + reconstruct (the GI3 path VERBATIM).
    return FxSHEvaluate(FxInterpolateSH(norm, shBuffer), normal);
}

// ----- InterpolateFieldOcc — the occlusion-weighted irradiance field over a set of query points. ------
// One query per output element (matching the shader's one-thread-per-query). `points`/`normals`/`out` are
// parallel spans; out[i] = FxInterpolateIrradianceOcc(grid, shBuffer, moments, points[i], normals[i],
// occStrength). An empty query set writes nothing (the no-op). occStrength==0 -> == InterpolateField
// BYTE-for-BYTE. PURE INTEGER, deterministic.
inline void InterpolateFieldOcc(const GiProbeGrid& grid, std::span<const FxProbeSH> shBuffer,
                                std::span<const FxProbeMoments> moments, std::span<const FxVec3> points,
                                std::span<const FxVec3> normals, fx occStrength,
                                std::span<GiRadiance> out) {
    size_t n = points.size();
    for (size_t i = 0; i < n; ++i)
        out[i] = FxInterpolateIrradianceOcc(grid, shBuffer, moments, points[i], normals[i], occStrength);
}

// ----- GiOccFieldToImage — the occlusion-fixed interpolated-irradiance-field viz (the GI5 golden). ----
// The GiFieldToImage 2D slice (a horizontal plane at the grid's vertical midplane, +Y receiver normal,
// PINNED slice + resolution) but evaluated with FxInterpolateIrradianceOcc — light behind an occluder wall
// is DARKER than the leaky GI3 blend. Strict integer -> identical both backends. `img` is row-major RGBA8
// (top row first), size == w*h. A 0-probe / empty-SH grid -> the image is left as the background fill.
inline void GiOccFieldToImage(const GiProbeGrid& grid, std::span<const FxProbeSH> shBuffer,
                              std::span<const FxProbeMoments> moments, fx occStrength,
                              std::span<uint32_t> img, uint32_t w, uint32_t h) {
    const uint32_t kBg = PackRGBA8(18, 18, 22, 255);
    for (uint32_t i = 0; i < w * h; ++i) img[i] = kBg;

    int probes = ProbeCount(grid);
    if (probes <= 0 || shBuffer.empty() || w == 0 || h == 0) return;

    const fx spanX = fxmul((fx)((int64_t)(grid.nx - 1) * kOne), grid.spacing);
    const fx spanZ = fxmul((fx)((int64_t)(grid.nz - 1) * kOne), grid.spacing);
    const fx sliceY = (fx)(grid.origin.y + (fxmul((fx)((int64_t)(grid.ny - 1) * kOne), grid.spacing) >> 1));
    const FxVec3 nrm{0, kOne, 0};

    auto q = [](fx v) -> int32_t {
        int64_t t = ((int64_t)v * 255) >> kFrac;
        if (t < 0) t = 0; if (t > 255) t = 255;
        return (int32_t)t;
    };

    for (uint32_t py = 0; py < h; ++py) {
        fx vfr = (h > 1) ? (fx)(((int64_t)py * kOne) / (int64_t)(h - 1)) : 0;
        fx z = (fx)(grid.origin.z + (int64_t)(((int64_t)spanZ * vfr) >> kFrac));
        for (uint32_t px = 0; px < w; ++px) {
            fx ufr = (w > 1) ? (fx)(((int64_t)px * kOne) / (int64_t)(w - 1)) : 0;
            fx x = (fx)(grid.origin.x + (int64_t)(((int64_t)spanX * ufr) >> kFrac));
            GiRadiance irr = FxInterpolateIrradianceOcc(grid, shBuffer, moments,
                                                        FxVec3{x, sliceY, z}, nrm, occStrength);
            img[(size_t)py * w + px] = PackRGBA8(q(irr.r), q(irr.g), q(irr.b), 255);
        }
    }
}

}  // namespace hf::render::gi
