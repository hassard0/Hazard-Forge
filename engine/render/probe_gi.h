#pragma once
// Slice DH — DDGI Beachhead: Probe-Grid Ray-Trace math — pure CPU (header-only, no device, no backend
// symbols). Namespace hf::render::probegi. The FIRST slice of the GLOBAL ILLUMINATION (DDGI — dynamic
// diffuse GI via irradiance probes) flagship arc. Mirrors froxel.h (a grid + a flat std430 SSBO record)
// + ssr.h / ssgi.h (view<->screen projection + a depth march) + gtao.h (a templated DepthFn + a
// disabled-path identity): a tiny shared-math header ABOVE the RHI seam (ZERO vk*/MTL*/mtl::/
// Backend::Metal CODE symbols — the only mentions of "vk"/"MTL" anywhere in this slice's above-seam
// files are seam-discipline doc comments). The compute shader shaders/probe_raytrace.comp.hlsl copies
// FibonacciSphere + ReconstructViewPos/ViewToScreenUV + TraceRayToDepth VERBATIM, so tests/probe_gi_test
// .cpp exercises the EXACT march the GPU pass runs — which is what makes the GPU ray-hit SSBO BIT-EXACT
// to the CPU reference over the same depth field AND bit-identical cross-backend.
//
// THE TECHNIQUE (the DDGI ray-trace data layer): a world-space lattice of irradiance PROBES; each probe
// traces kRaysPerProbe deterministic Fibonacci-sphere rays against the scene's VIEW-SPACE depth field
// (a flat float[w*h] SSBO read back from the rendered G-buffer's .w channel — NOT a sampled depth
// texture, mirroring froxel_inject which reconstructs analytically and never samples a depth texture in
// compute), recording per ray the world-space HIT position + the hit distance (or kRayMiss). This slice
// is ONLY the verified ray-trace data layer — radiance capture / SH encoding / the GI composite are
// later DDGI slices. No visible indirect lighting yet.
//
// THE BIT-EXACT GPU==CPU PROOF (what makes this verifiable): the showcase reads the rendered G-buffer
// depth into a flat float field, uploads it as the gDepth SSBO, and runs the probe-raytrace compute;
// the CPU reference runs probegi::TraceRayToDepth over the SAME flat field (the SAME nearest index
// math) for every probe x ray. The two ray-hit buffers must be BIT-EXACT (memcmp) — the probe analog of
// the gpu-cull gpu==cpuRef count proof. A mismatch means the shader-copied FibonacciSphere/projection/
// march diverged from the header. THE probeCount=0 NO-OP PROOF: dimX==0 -> probeCount()==0 ->
// ProbeDispatchGroups()==0 -> DispatchCompute(0) -> the ray-hit SSBO is untouched (== the cleared
// upload, all kRayMiss) — a byte-identical dispatch-0 no-op.
//
// CONVENTIONS (must match engine/math + ssr.h + the shader EXACTLY):
//   * Mat4 is column-major; RIGHT-HANDED view space looking down -Z (a point in front of the camera has
//     view-space z < 0). VIEW-SPACE LINEAR depth = -viewPos.z (positive in front), the value the
//     gbuffer.frag stores in .w and ssr.h's ViewToScreenUV returns as its .z.
//   * The screen UV <-> view-space projection is the SHARED ssr::ViewToScreenUV / ssr::ReconstructViewPos
//     (re-exported below, as ssgi.h does), so the probe march and the in-shader march use the SAME
//     projection. The yFlip sign maps screen UV.y <-> view Y: -1 on Vulkan, +1 on Metal.
//   * Probe flat index: idx = px + py*dimX + pz*(dimX*dimY) (cx-major, same as froxel/cluster).
//   * The ray-hit SSBO is a flat ProbeRayHit[probeCount*kRaysPerProbe]; probe p ray r is at
//     p*kRaysPerProbe + r (the froxel block layout).

#include "math/math.h"
#include "render/ssr.h"   // SHARE the view<->screen projection (ReconstructViewPos/ViewToScreenUV)

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace hf::render::probegi {

// Re-export the SHARED SSR reconstruction so the probe march + tests write probegi::ViewToScreenUV and
// the reuse is explicit at the call site (these are the SAME functions the shader's march copies).
using ssr::ReconstructViewPos;
using ssr::ViewToScreenUV;

// --- The probe grid: a WORLD-space lattice of irradiance probes. --------------------------------
// dimX*dimY*dimZ probes spaced `spacing` apart from `origin`. Defaults = 8x4x8 = 256 probes. A
// dim==0 -> probeCount()==0 -> the disabled / no-op path (ProbeDispatchGroups -> 0).
struct ProbeGrid {
    math::Vec3 origin{0.0f, 0.0f, 0.0f};
    int   dimX = 8, dimY = 4, dimZ = 8;
    float spacing = 1.0f;

    int probeCount() const { return dimX * dimY * dimZ; }
    // cx-major flat index (same order as froxel/cluster): idx = px + py*dimX + pz*(dimX*dimY).
    int flatIndex(int px, int py, int pz) const { return px + py * dimX + pz * (dimX * dimY); }
    // The world position of probe (px,py,pz): origin + (px,py,pz)*spacing.
    math::Vec3 probePos(int px, int py, int pz) const {
        return math::Vec3{origin.x + (float)px * spacing,
                          origin.y + (float)py * spacing,
                          origin.z + (float)pz * spacing};
    }
};

// kRaysPerProbe deterministic rays per probe (matches SsgiParams::K). The golden angle in radians
// (2*pi*(1 - 1/phi)) drives the Fibonacci spiral's azimuth.
inline constexpr int   kRaysPerProbe = 16;
inline constexpr float kGoldenAngle  = 2.39996322972865332f;

// --- Fibonacci-sphere direction --------------------------------------------------------------------
// FibonacciSphere(i,N) is the i-th of N points of the golden-spiral (Fibonacci-lattice) distribution on
// the UNIT SPHERE — a FULL-sphere, low-discrepancy, deterministic set (no RNG/time). Construction:
//   z   = 1 - (2*i + 1)/N          // evenly spaced z in (-1, 1) at the slab centers (monotone DECREASING)
//   r   = sqrt(max(0, 1 - z*z))    // the circle radius at height z
//   phi = i * kGoldenAngle         // the golden-angle azimuth spiral
//   dir = (r*cos phi, r*sin phi, z)
// Unit-length for all i. FibonacciSphere(0,1) == (0,0,1) (z = 1 - 1/1 = 0... -> actually z=0 for N=1;
// see below). N>=1; i is clamped to [0, N-1] and N to >=1 so the inputs are always valid. The shader
// copies this VERBATIM.
//
// NOTE on (0,1): with N==1, i==0: z = 1 - (2*0+1)/1 = 0, r = 1, phi = 0 -> dir = (1,0,0). The spec's
// "(0,1)->(0,0,1)" refers to the canonical single-point convention z=+1; to honor it AND keep the
// uniform formula we special-case N==1 -> (0,0,1) (the north pole), which is the natural single-sample
// representative and is what the shader's N==1 guard returns too. For the real kRaysPerProbe=16 path the
// uniform formula is used unmodified.
inline math::Vec3 FibonacciSphere(int i, int N) {
    if (N < 1) N = 1;
    if (i < 0) i = 0;
    if (i >= N) i = N - 1;
    if (N == 1) return math::Vec3{0.0f, 0.0f, 1.0f};   // canonical single-sample = the north pole
    float z   = 1.0f - (2.0f * (float)i + 1.0f) / (float)N;
    float r   = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = (float)i * kGoldenAngle;
    return math::Vec3{r * std::cos(phi), r * std::sin(phi), z};
}

// --- The flat std430 ray-hit record + the per-probe block. -----------------------------------------
// kRayMiss is the sentinel hit distance written for a ray that found no occluder (its .w). A huge finite
// value (not inf) so it survives the float SSBO round-trip + a memcmp byte-comparison cleanly.
inline constexpr float kRayMiss = 1e30f;

// One ray hit (std430, 16 bytes): xyz = world-space hit position, w = hit distance (or kRayMiss on a
// miss). Mirrors the shader's RWStructuredBuffer<ProbeRayHit> element.
struct ProbeRayHit {
    float hitPosDist[4];   // xyz = world hit position, w = hit distance or kRayMiss
};
static_assert(sizeof(ProbeRayHit) == 16, "ProbeRayHit must be 16 bytes (std430 float4)");

// One probe's full ray set (std430, 256 bytes): the kRaysPerProbe hits in ray-index order. The SSBO is a
// flat ProbeRayHit[probeCount*kRaysPerProbe]; this struct documents the per-probe block stride.
struct ProbeRayHits {
    ProbeRayHit rays[kRaysPerProbe];
};
static_assert(sizeof(ProbeRayHits) == 256, "ProbeRayHits must be 256 bytes (16 x float4)");

// --- Nearest-probe lookup (the round-trip + clamp). -----------------------------------------------
// GetProbeGridIndex(worldPos, grid) -> the flat index of the probe nearest `worldPos` (per-axis nearest
// cell): cell = lround((v - origin)/spacing), clamped to [0, dim-1], then flatIndex. Round-trips
// probePos at lattice points (probePos(px,py,pz) -> exactly flatIndex(px,py,pz)) and clamps far-outside
// positions to the boundary probe. Guards spacing<=0 (degenerate -> probe 0). Pure/deterministic.
inline int GetProbeGridIndex(const math::Vec3& worldPos, const ProbeGrid& grid) {
    if (grid.spacing <= 0.0f || grid.dimX <= 0 || grid.dimY <= 0 || grid.dimZ <= 0) return 0;
    auto axis = [](float v, float o, float sp, int dim) -> int {
        long c = std::lround((v - o) / sp);
        if (c < 0) c = 0;
        if (c > (long)dim - 1) c = (long)dim - 1;
        return (int)c;
    };
    int px = axis(worldPos.x, grid.origin.x, grid.spacing, grid.dimX);
    int py = axis(worldPos.y, grid.origin.y, grid.spacing, grid.dimY);
    int pz = axis(worldPos.z, grid.origin.z, grid.spacing, grid.dimZ);
    return grid.flatIndex(px, py, pz);
}

// --- Trilinear cell-corner lookup (the DDGI Slice 4 interpolation primitive). ----------------------
// ProbeTrilinear: the 8 surrounding probes of a FLOOR cell + their trilinear blend weights. idx[c] is
// the cx-major flat index of corner c (c in 0..7, with bit0=+x, bit1=+y, bit2=+z relative to the cell's
// base corner); w[c] is the trilinear weight (the 8 weights sum to 1 on the enabled path). valid==false
// on the degenerate / disabled path (spacing<=0 or any dim<=0): then idx is all 0 and w[0]==1 (a single
// harmless corner) so a downstream blend reads probe 0 only — but callers should test `valid` and treat
// it as the zeroed/identity fallback.
struct ProbeTrilinear {
    int   idx[8];   // the 8 cell-corner flat indices (cx-major; corner bit0=+x, bit1=+y, bit2=+z)
    float w[8];     // the 8 trilinear weights (sum == 1 on the enabled path)
    bool  valid;    // false on the degenerate/disabled path (idx all 0, w[0]=1)
};

// NearestProbes(worldPos, grid) -> the 8 trilinear cell corners + weights surrounding `worldPos`. Mirrors
// GetProbeGridIndex's per-axis projection but uses FLOOR (not round) to find the cell whose 8 corners
// bracket the position, then the fractional trilinear weights:
//   * degenerate guard: spacing<=0 || any dim<=0 -> {idx all 0, w[0]=1, valid=false}.
//   * per axis: g = (v - origin)/spacing; base = floor(g) clamped to [0, dim-2] (so base and base+1 are
//     BOTH valid probe indices); dim==1 -> base=0 and frac forced 0 on that axis (a 1-thick axis has no
//     second corner to blend toward); frac = clamp(g - base, 0, 1) (out-of-grid positions saturate to the
//     boundary cell — frac 0 below the grid, 1 above it -> a clamp to the nearest boundary corner).
//   * corner c in 0..7: (base+(c&1), base+((c>>1)&1), base+((c>>2)&1)) per axis, each coordinate clamped
//     to [0, dim-1] (so a dim==1 axis's +offset corner collapses onto the single valid index — its weight
//     is already 0 there) -> grid.flatIndex; the per-axis weight is
//     (selected? frac : 1-frac); w[c] = wx*wy*wz computed with std::fma so the per-corner product is a
//     single correctly-rounded fma matching the shader's mad (the cross-backend bit-exactness key — a
//     plain wx*wy*wz contracts to fma on Metal's fast-math but not on Vulkan/DXC or the CPU). Pure /
//     deterministic / no transcendentals (the weights are polynomial: floor + subtract + multiply).
inline ProbeTrilinear NearestProbes(const math::Vec3& worldPos, const ProbeGrid& grid) {
    ProbeTrilinear t;
    if (grid.spacing <= 0.0f || grid.dimX <= 0 || grid.dimY <= 0 || grid.dimZ <= 0) {
        for (int c = 0; c < 8; ++c) { t.idx[c] = 0; t.w[c] = 0.0f; }
        t.w[0] = 1.0f;
        t.valid = false;
        return t;
    }
    // Per-axis FLOOR cell base + fractional position. base in [0, dim-2] so base+1 is in range; a 1-thick
    // axis (dim==1) collapses to base 0, frac 0 (no second corner).
    auto axis = [](float v, float o, float sp, int dim, int& base, float& frac) {
        if (dim <= 1) { base = 0; frac = 0.0f; return; }
        float g = (v - o) / sp;
        float fb = std::floor(g);
        if (fb < 0.0f) fb = 0.0f;
        if (fb > (float)(dim - 2)) fb = (float)(dim - 2);
        base = (int)fb;
        float fr = g - fb;
        if (fr < 0.0f) fr = 0.0f;
        if (fr > 1.0f) fr = 1.0f;
        frac = fr;
    };
    int bx, by, bz;
    float fx, fy, fz;
    axis(worldPos.x, grid.origin.x, grid.spacing, grid.dimX, bx, fx);
    axis(worldPos.y, grid.origin.y, grid.spacing, grid.dimY, by, fy);
    axis(worldPos.z, grid.origin.z, grid.spacing, grid.dimZ, bz, fz);
    for (int c = 0; c < 8; ++c) {
        int sx = (c & 1), sy = ((c >> 1) & 1), sz = ((c >> 2) & 1);
        // A dim==1 axis has frac 0 and no second probe, so its +offset corner must clamp back onto the
        // single in-range index (base+1 would be out of range). Clamping the coordinate keeps every
        // corner index valid; the +offset weight is already 0 there (frac==0), so the blend is unchanged.
        int cx = bx + sx; if (cx > grid.dimX - 1) cx = grid.dimX - 1;
        int cy = by + sy; if (cy > grid.dimY - 1) cy = grid.dimY - 1;
        int cz = bz + sz; if (cz > grid.dimZ - 1) cz = grid.dimZ - 1;
        t.idx[c] = grid.flatIndex(cx, cy, cz);
        float wx = sx ? fx : (1.0f - fx);
        float wy = sy ? fy : (1.0f - fy);
        float wz = sz ? fz : (1.0f - fz);
        // w = wx*wy*wz as two single-rounded fma-able products: p = wx*wy (a single multiply), then
        // w = std::fma(p, wz, 0) — a SINGLE correctly-rounded fma, matching the shader's mad(p, wz, 0).
        float p = wx * wy;
        t.w[c] = std::fma(p, wz, 0.0f);
    }
    t.valid = true;
    return t;
}

// --- The per-ray view-space march against the depth field. -----------------------------------------
// TraceRayToDepth(originWorld, dirWorld, maxDist, steps, thickness, view, tanHalfFovY, aspect, yFlip,
// sampleDepth, outHit) marches the ray in WORLD space, projecting each marched point to a screen UV +
// view-linear depth via the SHARED ssr projection, and compares against the sampled scene depth. It
// mirrors gtao::Visibility's fixed-step march + ssr's view<->screen projection + thickness band.
//
// March (fixed steps, no RNG -> bit-identical cross-backend):
//   for k = 1..steps:
//     t      = (maxDist/steps) * k
//     pWorld = originWorld + dirWorld * t
//     pView  = view * pWorld                          // world -> view (RH, -Z forward)
//     uvd    = ssr::ViewToScreenUV(pView, ...)        // uvd.xy = screen UV, uvd.z = view-linear depth
//     if uvd.xy outside [0,1]  -> continue            // off-screen: no occluder information here
//     surf   = sampleDepth(uvd.x, uvd.y)              // the scene's view-linear depth at that UV
//     HIT iff uvd.z in [surf - thickness, surf + thickness]   // the front-to-back crossing band (SSR-style)
//        -> outHit = { pWorld.xyz, t }, return true
//   no hit after the loop -> outHit.w = kRayMiss, return false.
// The thickness band is the SAME front-to-back crossing test SSR uses: the ray HITS when its projected
// view depth lands within +/-thickness of the stored surface depth at that pixel (the ray pierces the
// surface). A point in FRONT of all surfaces (uvd.z << surf) keeps marching; one BEHIND (uvd.z >>
// surf+thickness, the surface occludes it) keeps marching too (it is hidden, not a hit) — only the
// crossing band registers. The CPU test feeds a procedural sampleDepth; the shader feeds the gDepth
// SSBO (nearest index) — the SAME march. `outHit` is always written (the hit pos+dist or {0,0,0,kRayMiss}).
template <typename DepthFn>
inline bool TraceRayToDepth(const math::Vec3& originWorld, const math::Vec3& dirWorld, float maxDist,
                            int steps, float thickness, const math::Mat4& view, float tanHalfFovY,
                            float aspect, float yFlip, DepthFn sampleDepth, ProbeRayHit& outHit) {
    if (steps < 1) steps = 1;
    float dt = maxDist / (float)steps;
    for (int k = 1; k <= steps; ++k) {
        float t = dt * (float)k;
        // The marched point, computed as a SINGLE correctly-rounded fused multiply-add per axis (std::fma).
        // This is the cross-backend bit-exactness key: an IEEE fma is correctly rounded (a SINGLE rounding)
        // on the CPU (std::fma, hardware FMA), on Vulkan (DXC mad -> SPIR-V the driver fuses) AND on Metal
        // (the MSL mad contracts to fma() under the default fast-math), so all three agree to the bit. A
        // plain `origin + dir*t` rounds TWICE and the result then depends on whether each backend contracts
        // it (Metal does, Vulkan/DXC + the CPU do not) -> a 1-ULP divergence that broke the GPU==CPU memcmp.
        math::Vec3 pWorld{std::fma(dirWorld.x, t, originWorld.x),
                          std::fma(dirWorld.y, t, originWorld.y),
                          std::fma(dirWorld.z, t, originWorld.z)};
        math::Vec3 pView = math::MulPoint(view, pWorld);
        math::Vec3 uvd = ViewToScreenUV(pView, tanHalfFovY, aspect, yFlip);
        if (uvd.x < 0.0f || uvd.x > 1.0f || uvd.y < 0.0f || uvd.y > 1.0f) continue;  // off-screen
        float surf = sampleDepth(uvd.x, uvd.y);
        if (uvd.z >= surf - thickness && uvd.z <= surf + thickness) {
            outHit.hitPosDist[0] = pWorld.x;
            outHit.hitPosDist[1] = pWorld.y;
            outHit.hitPosDist[2] = pWorld.z;
            outHit.hitPosDist[3] = t;
            return true;
        }
    }
    outHit.hitPosDist[0] = 0.0f;
    outHit.hitPosDist[1] = 0.0f;
    outHit.hitPosDist[2] = 0.0f;
    outHit.hitPosDist[3] = kRayMiss;
    return false;
}

// --- The disabled / dispatch-sizing path. ----------------------------------------------------------
// kProbeThreads = the compute workgroup size (one thread per probe). ProbeDispatchGroups(grid) = the
// number of HF_PROBE_THREADS workgroups to cover probeCount probes, or EXACTLY 0 when probeCount<=0
// (dimX/dimY/dimZ == 0 -> probeCount == 0 -> 0 groups -> DispatchCompute(0) -> the ray-hit SSBO is
// untouched == the cleared upload). This is the byte-identical no-op the proof rests on.
inline constexpr int kProbeThreads = 64;
inline int ProbeDispatchGroups(const ProbeGrid& grid) {
    int n = grid.probeCount();
    return (n <= 0) ? 0 : (n + kProbeThreads - 1) / kProbeThreads;
}

// --- The trilinear-interp dispatch sizing (one thread per QUERY point). ----------------------------
// kInterpThreads = the interp compute workgroup size (one thread per query point). InterpDispatchGroups(
// grid, nQueries) = the number of kInterpThreads workgroups to cover nQueries query points, or EXACTLY 0
// when nQueries<=0 OR the grid is disabled (probeCount<=0, i.e. dimX/dimY/dimZ == 0). probeCount==0 -> 0
// groups -> DispatchCompute(0) -> the blended-output SSBO is untouched (== the cleared upload) — the
// byte-identical probeCount=0 no-op the proof rests on (the ProbeDispatchGroups/EncodeDispatchGroups
// analog). Pure / deterministic.
inline constexpr int kInterpThreads = 64;
inline int InterpDispatchGroups(const ProbeGrid& grid, int nQueries) {
    if (nQueries <= 0 || grid.probeCount() <= 0) return 0;
    return (nQueries + kInterpThreads - 1) / kInterpThreads;
}

}  // namespace hf::render::probegi
