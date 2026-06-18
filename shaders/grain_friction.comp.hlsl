// Slice GR4 — Deterministic GPU Granular/Sand: the TANGENTIAL COULOMB FRICTION Δp pass (the angle-of-repose;
// the SIGNATURE slice of FLAGSHIP #10, the grain_contact_dp.comp twin with the friction clamp). ONE thread
// per GRAIN i (i < grainCount) — the JACOBI per-grain-independent pattern (NOT single-thread): each thread
// reads its neighbours' positions/prev (read-only) and writes only its OWN dp[i] (a SEPARATE buffer from
// gGrains — the Jacobi double-buffer), so there is NO race and the result is DETERMINISTIC regardless of
// thread order. NO atomics, NO single-thread, NO TDR ceiling (the GR3/FL4 design win).
//
// The body is copied VERBATIM from grain.h::SolveGrainFriction (the standard Unified-Particle / PBD friction):
//   for each overlapping neighbour j (pen = (r_i+r_j) − |p_i−p_j| > 0):
//     n      = unit(p_i − p_j)
//     Δx_rel = (p_i − prev_i) − (p_j − prev_j)
//     Δx_t   = Δx_rel − (Δx_rel · n)·n                      // tangential (subtract the normal part)
//     t      = |Δx_t| ; fmax = fxmul(μ, pen)                // the Coulomb cone
//     if t <= eps:        no friction
//     else if t <= fmax:  corr = Δx_t                       // STATIC: cancel ALL tangential slip
//     else:               corr = FxScale(Δx_t, fxdiv(fmax,t))// KINETIC: clamp to the cone
//     Δp_i += −share·corr  (share = fxdiv(w_i, w_i+w_j))
// STATIC grains (flags & STATIC bit) -> Δp = 0. A divergence vs the header is exactly what the host's GPU==CPU
// memcmp (the settled grain array) catches.
//
// INTEGER WIDTH: fxmul/fxdiv/FxISqrt + the centre distance use int64_t (the grain_contact_dp.comp lesson). DXC
// compiles int64 (Vulkan); glslc CANNOT parse int64 in HLSL -> this shader is VULKAN-SPIR-V-ONLY (NOT in
// metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --grain-friction showcase runs the CPU
// grain::StepGrainFriction (byte-identical by construction). REUSES grain_contact_apply.comp + grain_collide.comp
// verbatim (the apply + the velocity/collide passes are shared with GR3).
//
// Buffers (storage, bound at compute bindings 0..4; Vulkan-only):
//   b0 gGrains       : the Q16.16 GrainParticle array (48 bytes), READ (pos, prev, invMass, radius, flags).
//   b1 neighborStart : the GR2 neighbor-list prefix-sum (grainCount+1), READ.
//   b2 neighbors     : the GR2 candidate neighbor j indices grouped by i, READ.
//   b3 dp            : one FxVec3Gpu (3 x int32) per grain (the Jacobi friction Δp_i), WRITE.
//   b4 gParams       : the GrainFrictionParams (grainCount, enabled, mu), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_GRAIN_THREADS 64
#define HF_GRAIN_FRAC 16
#define HF_GRAIN_ONE 65536      // kOne == 1<<16
#define HF_GRAIN_STATIC 1u      // == grain.h::kFlagStatic (bit0)
#define HF_GRAIN_FRICTION_EPS 16   // == grain.h::kGrainFrictionEps (the slip dead-band)

struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

struct FxVec3Gpu { int x, y, z; };   // the Δp output (std430 12 bytes; the FxVec3 mirror)

// GrainFrictionParams (std430). cfg {grainCount, enabled, mu, _}.
struct GrainFrictionParams { int4 cfg; };

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle>       gGrains       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>               neighborStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>               neighbors     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<FxVec3Gpu>          dp            : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<GrainFrictionParams> gParams      : register(u4);

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
    int mu         = gParams[0].cfg.z;

    uint i = gid.x;
    if ((int)i >= grainCount) return;
    FxVec3Gpu zero; zero.x = 0; zero.y = 0; zero.z = 0;
    if (enabled == 0) { dp[i] = zero; return; }

    GrainParticle pi = gGrains[i];
    if (pi.flags & HF_GRAIN_STATIC) { dp[i] = zero; return; }   // static -> Δp = 0 (the pinned case)

    int wi = pi.invMass;
    int ax = 0, ay = 0, az = 0;   // Σ_j −share·corr
    uint s0 = neighborStart[i], s1 = neighborStart[i + 1u];
    for (uint s = s0; s < s1; ++s) {
        uint j = neighbors[s];
        GrainParticle pj = gGrains[j];
        int wsum = wi + pj.invMass;
        if (wsum == 0) continue;                                // both static -> no friction
        int dx = pi.px - pj.px, dy = pi.py - pj.py, dz = pi.pz - pj.pz;   // d = p_i − p_j
        int dist = FxLength3(dx, dy, dz);
        int pen = (pi.radius + pj.radius) - dist;
        if (pen <= 0) continue;                                 // non-overlapping candidate -> no friction
        int nx, ny, nz;
        if (dist == 0) { nx = 0; ny = HF_GRAIN_ONE; nz = 0; }   // FxNormalize +Y fallback
        else { nx = fxdiv(dx, dist); ny = fxdiv(dy, dist); nz = fxdiv(dz, dist); }
        // Δx_rel = (p_i − prev_i) − (p_j − prev_j) — the relative displacement this step.
        int rx = (pi.px - pi.prx) - (pj.px - pj.prx);
        int ry = (pi.py - pi.pry) - (pj.py - pj.pry);
        int rz = (pi.pz - pi.prz) - (pj.pz - pj.prz);
        // Δx_t = Δx_rel − (Δx_rel · n)·n (subtract the normal component).
        int dotN = fxmul(rx, nx) + fxmul(ry, ny) + fxmul(rz, nz);
        int tx = rx - fxmul(nx, dotN);
        int ty = ry - fxmul(ny, dotN);
        int tz = rz - fxmul(nz, dotN);
        int t = FxLength3(tx, ty, tz);
        if (t <= HF_GRAIN_FRICTION_EPS) continue;               // no real slip (dead-band) -> no friction
        int fmax = fxmul(mu, pen);                              // the Coulomb cone radius
        int cx, cy, cz;
        if (t <= fmax) { cx = tx; cy = ty; cz = tz; }           // STATIC: cancel ALL tangential slip
        else {                                                  // KINETIC: clamp the slip to the cone
            int sc = fxdiv(fmax, t);
            cx = fxmul(tx, sc); cy = fxmul(ty, sc); cz = fxmul(tz, sc);
        }
        int share = fxdiv(wi, wsum);
        ax -= fxmul(share, cx); ay -= fxmul(share, cy); az -= fxmul(share, cz);
    }
    FxVec3Gpu d; d.x = ax; d.y = ay; d.z = az;
    dp[i] = d;                                                  // per-grain independent write (NO atomics)
}
