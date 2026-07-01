// Slice FL7 — Deterministic GPU Fluid: the XSPH VISCOSITY pass 1 (the smoothed-velocity gather; the
// Track-R R2 refinement of FLAGSHIP #9). ONE thread per PARTICLE i — the JACOBI per-particle-independent
// pattern (the fluid_dp.comp twin): each thread reads its neighbours' (pos, prev) state READ-ONLY and
// writes only its OWN vout[i] (a SEPARATE scratch buffer from gParticles — the Jacobi double-buffer), so
// there is NO race and the result is DETERMINISTIC regardless of thread order. NO atomics, NO
// single-thread, NO TDR ceiling.
//
// The body is copied VERBATIM from fluid.h::ComputeXsphVelocity:
//   v_i  = (pos_i − prev_i) / dt                                    (the implicit PBF velocity, fxdiv)
//   v'_i = v_i + c · Σ_{j∈neighbors(i)} (v_j − v_i) · W[bin(r_ij²)] (the XSPH smooth; the FL3 poly6 W LUT)
// STATIC particles (flags & STATIC bit) -> v' = 0 (pass 2 never re-encodes them). A divergence vs the
// header is exactly what the host's GPU==CPU memcmp (the final particle array) catches.
//
// INTEGER WIDTH: the velocity fxdiv + RadiusSq/BinOf use int64_t (the fluid_dp.comp lesson). DXC compiles
// int64 (Vulkan); glslc CANNOT parse int64 in HLSL -> this shader is VULKAN-SPIR-V-ONLY (NOT in
// metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --fl7-visc showcase runs the CPU
// fluid::StepFluidViscSteps (byte-identical by construction, the FL4 fluid_dp/fluid_collide convention).
//
// Buffers (storage, Vulkan-only):
//   b0 gParticles    : the Q16.16 FluidParticle array, READ (pos, prev, flags).
//   b1 neighborStart : the FL2 neighbor-list prefix-sum (particleCount+1), READ.
//   b2 neighbors     : the FL2 candidate neighbor j indices grouped by i, READ.
//   b3 kernelW       : the host-snapped Q16.16 poly6 W LUT W[bins] (the FL3 density LUT, REUSED), READ.
//   b4 vout          : one FxVec3Gpu (3 x int32) per particle (the smoothed v'_i), WRITE.
//   b5 gParams       : the FluidViscParams (h, c, dt, bins, particleCount, enabled), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_FLUID_THREADS 64
#define HF_FLUID_FRAC 16
#define HF_FLUID_ONE 65536      // kOne == 1<<16

struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

struct FxVec3Gpu { int x, y, z; };   // the v' output (std430 12 bytes; the FxVec3 mirror)

// FluidViscParams (std430). ker {h, c, dt, bins}, cfg {particleCount, enabled}.
struct FluidViscParams {
    int4 ker;   // x=h, y=c (viscosity coefficient), z=dt, w=bins
    int4 cfg;   // x=particleCount, y=enabled
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FluidParticle>   gParticles    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>            neighborStart : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>            neighbors     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<int>             kernelW       : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<FxVec3Gpu>       vout          : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<FluidViscParams> gParams       : register(u5);

static const uint HF_FLUID_STATIC = 1u;   // == fluid.h::kFlagStatic (bit0)

// fxmul/fxdiv — VERBATIM fpx.h::fxmul / fpx.h::fxdiv (int64 intermediate).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_FLUID_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_FLUID_FRAC) / (int64_t)b); }

// RadiusSq/BinOf — VERBATIM fluid.h (int64).
int64_t RadiusSq(FluidParticle a, FluidParticle b) {
    int64_t dx = (int64_t)a.px - (int64_t)b.px;
    int64_t dy = (int64_t)a.py - (int64_t)b.py;
    int64_t dz = (int64_t)a.pz - (int64_t)b.pz;
    return dx * dx + dy * dy + dz * dz;
}
int BinOf(int64_t r2, int64_t h2, int bins) {
    if (r2 >= h2) return bins;
    return (int)((r2 * (int64_t)bins) / h2);
}

[numthreads(HF_FLUID_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int h             = gParams[0].ker.x;
    int c             = gParams[0].ker.y;
    int dt            = gParams[0].ker.z;
    int bins          = gParams[0].ker.w;
    int particleCount = gParams[0].cfg.x;
    int enabled       = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= particleCount) return;
    FxVec3Gpu zero; zero.x = 0; zero.y = 0; zero.z = 0;
    if (enabled == 0) { vout[i] = zero; return; }

    FluidParticle pi = gParticles[i];
    if (pi.flags & HF_FLUID_STATIC) { vout[i] = zero; return; }   // static -> v' = 0 (never re-encoded)

    int64_t h2 = (int64_t)h * (int64_t)h;
    // v_i = (pos − prev) / dt (the implicit PBF velocity; fxdiv(x, 0) == 0 by the fpx contract).
    int vix = fxdiv(pi.px - pi.prx, dt);
    int viy = fxdiv(pi.py - pi.pry, dt);
    int viz = fxdiv(pi.pz - pi.prz, dt);
    int ax = 0, ay = 0, az = 0;   // Σ_j (v_j − v_i)·W[bin] (the pre-c accumulate, int32 like fluid_dp)
    uint s0 = neighborStart[i], s1 = neighborStart[i + 1u];
    for (uint s = s0; s < s1; ++s) {
        uint j = neighbors[s];
        FluidParticle pj = gParticles[j];
        int bin = BinOf(RadiusSq(pi, pj), h2, bins);
        if (bin >= bins) continue;                                // r >= h -> zero kernel weight
        int w = kernelW[bin];
        // v_j re-derived from (pos_j − prev_j)/dt (identical to i's derivation — a pure function).
        int vjx = fxdiv(pj.px - pj.prx, dt);
        int vjy = fxdiv(pj.py - pj.pry, dt);
        int vjz = fxdiv(pj.pz - pj.prz, dt);
        ax += fxmul(vjx - vix, w);
        ay += fxmul(vjy - viy, w);
        az += fxmul(vjz - viz, w);
    }
    // v'_i = v_i + c · accum (the XSPH blend toward the neighbourhood mean).
    FxVec3Gpu v; v.x = vix + fxmul(c, ax); v.y = viy + fxmul(c, ay); v.z = viz + fxmul(c, az);
    vout[i] = v;                                                  // per-particle independent write (NO atomics)
}
