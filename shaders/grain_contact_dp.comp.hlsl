// Slice GR3 — Deterministic GPU Granular/Sand: the FRICTIONLESS CONTACT Δp pass (the contact projection's
// position correction; the 3rd slice of FLAGSHIP #10, the FL4 fluid_dp.comp twin). ONE thread per GRAIN i
// (i < grainCount) — the JACOBI per-grain-independent pattern (NOT the cloth CL3 single-thread): each thread
// reads its neighbours' ITERATION-START positions (read-only) and writes only its OWN dp[i] (a SEPARATE
// buffer from gGrains — the Jacobi double-buffer), so there is NO race and the result is DETERMINISTIC
// regardless of thread order. NO atomics, NO single-thread, NO TDR ceiling (the FL4 design win over the
// cloth's order-dependent Gauss-Seidel).
//
// The body is copied VERBATIM from grain.h::SolveGrainContact:
//   Δp_i = Σ_{j∈neighbors(i), pen>0} ( w_i / (w_i + w_j) ) · pen · unit(p_i − p_j)
// where pen = (r_i + r_j) − |p_i − p_j| (only pen > 0 contributes — the exact radial overlap cull GR2
// deferred), w = invMass (w_i+w_j==0 both-static -> skip the pair), d==0 -> the +Y FxNormalize fallback.
// STATIC grains (flags & STATIC bit) -> Δp = 0. A divergence vs the header is exactly what the host's
// GPU==CPU memcmp (the settled grain array) catches.
//
// INTEGER WIDTH: fxmul/fxdiv/FxISqrt + the centre distance use int64_t (the fluid_dp.comp lesson). DXC
// compiles int64 (Vulkan); glslc CANNOT parse int64 in HLSL -> this shader is VULKAN-SPIR-V-ONLY (NOT in
// metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --grain-contact showcase runs the CPU
// grain::StepGrainContact (byte-identical by construction).
//
// Buffers (storage, bound at compute bindings 0..3; Vulkan-only):
//   b0 gGrains       : the Q16.16 GrainParticle array (48 bytes), READ (pos, invMass, radius, flags).
//   b1 neighborStart : the GR2 neighbor-list prefix-sum (grainCount+1), READ.
//   b2 neighbors     : the GR2 candidate neighbor j indices grouped by i, READ.
//   b3 dp            : one FxVec3Gpu (3 x int32) per grain (the Jacobi correction Δp_i), WRITE.
//   b4 gParams       : the GrainContactParams (grainCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_GRAIN_THREADS 64
#define HF_GRAIN_FRAC 16
#define HF_GRAIN_ONE 65536      // kOne == 1<<16
#define HF_GRAIN_STATIC 1u      // == grain.h::kFlagStatic (bit0)

struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

struct FxVec3Gpu { int x, y, z; };   // the Δp output (std430 12 bytes; the FxVec3 mirror)

// GrainContactParams (std430). cfg {grainCount, enabled, _, _}.
struct GrainContactParams { int4 cfg; };

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle>      gGrains       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>              neighborStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>              neighbors     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<FxVec3Gpu>         dp            : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<GrainContactParams> gParams      : register(u4);

// fxmul/fxdiv — VERBATIM fpx.h::fxmul / fpx.h::fxdiv (int64 intermediate).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_GRAIN_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_GRAIN_FRAC) / (int64_t)b); }

// FxISqrt — VERBATIM fpx.h::FxISqrt (int64 binary integer sqrt).
int64_t FxISqrt(int64_t v) {
    if (v <= 0) return 0;
    int64_t bit = (int64_t)1 << 62;
    while (bit > v) bit >>= 2;
    int64_t res = 0;
    while (bit != 0) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else { res >>= 1; }
        bit >>= 2;
    }
    return res;
}
int FxLength3(int x, int y, int z) {
    int64_t sx = (int64_t)x * (int64_t)x;
    int64_t sy = (int64_t)y * (int64_t)y;
    int64_t sz = (int64_t)z * (int64_t)z;
    return (int)FxISqrt(sx + sy + sz);
}

[numthreads(HF_GRAIN_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int grainCount = gParams[0].cfg.x;
    int enabled    = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= grainCount) return;
    FxVec3Gpu zero; zero.x = 0; zero.y = 0; zero.z = 0;
    if (enabled == 0) { dp[i] = zero; return; }

    GrainParticle pi = gGrains[i];
    if (pi.flags & HF_GRAIN_STATIC) { dp[i] = zero; return; }   // static -> Δp = 0 (the pinned case)

    int wi = pi.invMass;
    int ax = 0, ay = 0, az = 0;   // Σ_j share·pen·unit(p_i − p_j)
    uint s0 = neighborStart[i], s1 = neighborStart[i + 1u];
    for (uint s = s0; s < s1; ++s) {
        uint j = neighbors[s];
        GrainParticle pj = gGrains[j];
        int wsum = wi + pj.invMass;
        if (wsum == 0) continue;                                // both static -> no push
        int dx = pi.px - pj.px, dy = pi.py - pj.py, dz = pi.pz - pj.pz;   // d = p_i − p_j
        int dist = FxLength3(dx, dy, dz);
        int pen = (pi.radius + pj.radius) - dist;
        if (pen <= 0) continue;                                 // non-overlapping candidate -> no-op
        int share = fxdiv(wi, wsum);
        int scale = fxmul(share, pen);
        int nx, ny, nz;
        if (dist == 0) { nx = 0; ny = HF_GRAIN_ONE; nz = 0; }   // FxNormalize +Y fallback
        else { nx = fxdiv(dx, dist); ny = fxdiv(dy, dist); nz = fxdiv(dz, dist); }
        ax += fxmul(scale, nx); ay += fxmul(scale, ny); az += fxmul(scale, nz);
    }
    FxVec3Gpu d; d.x = ax; d.y = ay; d.z = az;
    dp[i] = d;                                                  // per-grain independent write (NO atomics)
}
