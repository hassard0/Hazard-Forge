// Slice CL7 — Deterministic GPU Cloth: the SELF-COLLISION pass 1 (the Jacobi correction gather; the
// Track-R R3 refinement of FLAGSHIP #8). ONE thread per cloth VERT i — the JACOBI per-vert-independent
// pattern (the fluid_visc.comp twin): each thread reads the particle positions READ-ONLY and writes only
// its OWN gCorr[i] (a SEPARATE scratch buffer from gParticles — the Jacobi double-buffer), so there is NO
// race and the result is DETERMINISTIC regardless of thread order. NO atomics, NO single-thread, NO TDR
// ceiling — unlike the Gauss-Seidel cloth_solve/cloth_collide constraint passes, this pass MULTI-THREADS.
//
// The body is copied VERBATIM from cloth.h::SolveSelfCollision PASS 1: for each candidate j in i's CSR
// slice (the host-built, exclusion-filtered FL2-grid candidate list — verts sharing a CL2 constraint were
// removed at build): d = pos_i - pos_j; dist = FxLength(d); dist < thickness ->
//   pen  = thickness - dist
//   axis = dist == 0 ? (i < j ? +Y : -Y) : FxNormalize(d)     (the deterministic coincident tie-break)
//   wi   = fxdiv(invMass_i, invMass_i + invMass_j)            (the CL3/ResolvePair inverse-mass split)
//   corr_i += axis * fxmul(pen, wi)
// PINNED verts (bit0) -> corr = 0 (pass 2 never moves them). A divergence vs the header is exactly what
// the host's GPU==CPU memcmp (the final particle array) catches.
//
// INTEGER WIDTH: FxLength/FxISqrt/fxdiv use int64_t (the cloth_solve.comp lesson). DXC compiles int64
// (Vulkan); glslc CANNOT parse int64 in HLSL -> this shader is VULKAN-SPIR-V-ONLY (NOT in
// metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --cl7-self showcase runs the CPU
// cloth::StepClothSelfSteps (byte-identical by construction, the cloth_solve/cloth_collide convention).
//
// Buffers (storage, Vulkan-only):
//   b0 gParticles : the Q16.16 ClothParticle array (44 bytes, the CL1 mirror), READ.
//   b1 gCandStart : the per-vert candidate-list prefix-sum (particleCount+1, CSR), READ.
//   b2 gCand      : the candidate j indices grouped by i (exclusion-filtered at host build), READ.
//   b3 gCorr      : one FxVec3Gpu (3 x int32) per vert (the gathered correction), WRITE.
//   b4 gParams    : the ClothSelfParams (thickness, groundY, particleCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_CLOTH_THREADS 64
#define HF_CLOTH_FRAC 16   // MUST match cloth.h::kFrac (== fpx.h::kFrac)
#define HF_CLOTH_ONE  (1 << HF_CLOTH_FRAC)
#define HF_CLOTH_FLAG_PINNED 1u

// std430 ClothParticle mirror (engine/sim/cloth.h::ClothParticle): 11 x 4-byte = 44 bytes (== CL1-CL4).
struct ClothParticle {
    int  px, py, pz;     // Q16.16 current position
    int  prx, pry, prz;  // Q16.16 previous position
    int  vx, vy, vz;     // Q16.16 velocity
    int  invMass;        // Q16.16 inverse mass (0 => pinned)
    uint flags;          // bit0 = PINNED
};

struct FxVec3Gpu { int x, y, z; };   // the correction output (std430 12 bytes; the FxVec3 mirror)

// ClothSelfParams (std430). cfg {thickness, groundY, particleCount, enabled}.
struct ClothSelfParams {
    int4 cfg;   // x=thickness (Q16.16), y=groundY (Q16.16), z=particleCount, w=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<ClothParticle>   gParticles : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            gCandStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            gCand      : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<FxVec3Gpu>       gCorr      : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<ClothSelfParams> gParams    : register(u4);

// fxmul/fxdiv — VERBATIM fpx.h::fxmul / fpx.h::fxdiv (int64 intermediate).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_CLOTH_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_CLOTH_FRAC) / (int64_t)b); }

// VERBATIM fpx.h::FxISqrt — floor(sqrt) of a non-negative int64 (binary digit-by-digit).
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

// VERBATIM fpx.h::FxLength — sqrt(x^2+y^2+z^2) in Q16.16 (sum of int64 squares -> floor-sqrt).
int FxLength(int vx, int vy, int vz) {
    int64_t sx = (int64_t)vx * (int64_t)vx;
    int64_t sy = (int64_t)vy * (int64_t)vy;
    int64_t sz = (int64_t)vz * (int64_t)vz;
    return (int)FxISqrt(sx + sy + sz);
}

[numthreads(HF_CLOTH_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int thickness     = gParams[0].cfg.x;
    int groundY       = gParams[0].cfg.y;   // unused in pass 1 (pass 2 clamps); kept for the shared params
    int particleCount = gParams[0].cfg.z;
    int enabled       = gParams[0].cfg.w;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    FxVec3Gpu zero; zero.x = 0; zero.y = 0; zero.z = 0;
    if (enabled == 0 || thickness <= 0) { gCorr[i] = zero; return; }

    ClothParticle pi = gParticles[i];
    if (pi.flags & HF_CLOTH_FLAG_PINNED) { gCorr[i] = zero; return; }   // pinned share 0 -> corr 0

    int ax = 0, ay = 0, az = 0;   // the per-vert correction accumulate (int32, the FL7 accumulate bound)
    uint s0 = gCandStart[i], s1 = gCandStart[i + 1u];
    for (uint s = s0; s < s1; ++s) {
        uint j = gCand[s];
        if ((int)j >= particleCount) continue;                          // bounds-checked skip
        ClothParticle pj = gParticles[j];
        int wsum = pi.invMass + pj.invMass;
        if (wsum == 0) continue;                                        // both pinned -> skip
        int dx = pi.px - pj.px;
        int dy = pi.py - pj.py;
        int dz = pi.pz - pj.pz;
        int dist = FxLength(dx, dy, dz);
        if (dist >= thickness) continue;                                // the exact radial cull
        int pen = thickness - dist;
        // The push-apart axis: coincident pair -> the deterministic INDEX tie-break (+Y for the lower
        // index, -Y for the higher); else FxNormalize(d) (fxdiv per axis).
        int nx, ny, nz;
        if (dist == 0) { nx = 0; ny = (i < j) ? HF_CLOTH_ONE : -HF_CLOTH_ONE; nz = 0; }
        else { nx = fxdiv(dx, dist); ny = fxdiv(dy, dist); nz = fxdiv(dz, dist); }
        int wi = fxdiv(pi.invMass, wsum);
        int mag = fxmul(pen, wi);
        ax += fxmul(nx, mag);
        ay += fxmul(ny, mag);
        az += fxmul(nz, mag);
    }
    FxVec3Gpu c; c.x = ax; c.y = ay; c.z = az;
    gCorr[i] = c;                                                       // per-vert independent write
}
