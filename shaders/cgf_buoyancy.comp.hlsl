// Slice GF2 — Deterministic Grain<->Fluid Coupling: the BUOYANCY + DRAG fluid->grain pass (the CRUX; the 2nd
// slice of FLAGSHIP #13). ONE thread per GRAIN i (i < grainCount) — multi-thread OVER grains, each thread
// SERIALLY sums its grain's GF1 fluid-neighbour list (gfNeighbors[gfStart[i]..gfStart[i+1]), ascending — the
// FIXED GF1 emit order) into a buoyancy (∝ the fluid count, opposing gravity) + drag (toward the static
// fluid's mean velocity) velocity delta, then writes the grain's new vel. PER-GRAIN (each grain reads its OWN
// fluid list, writes ONLY its own vel) -> per-grain-disjoint, race-free, NO atomics, [numthreads(64,1,1)]
// MULTI-THREAD, NO single-thread TDR (the FL4/GR3 Jacobi pattern — cleaner than CG2's per-body reduction).
//
// The body is copied VERBATIM from couple_gf.h::AccumGrainBuoyancy (linear only — grains are points, no
// torque; the host driver runs IntegrateGrains AFTER this pass):
//   up = -FxNormalize(gravity)                              // the buoyant direction (computed per thread)
//   cnt = gfStart[i+1] - gfStart[i] ; if (cnt == 0) skip    // dry grain -> free GR sim
//   F_buoy = up * fxmul(kBuoyPerFluid, cnt<<kFrac)          // ∝ the submerging fluid count
//   vFluidAvg = (Σ fluid[j].vel) / cnt                      // fixed-order int64 sum / the integer count
//   F_drag = fxmul(kDrag, vFluidAvg - grain.vel)            // per axis
//   grain.vel += FxScale(F_buoy + F_drag, grain.invMass) · dt
// A STATIC grain (kFlagStatic) or a DRY grain (cnt==0) is untouched. A divergence vs the header is exactly
// what the host's GPU==CPU memcmp (the settled grain array) catches.
//
// INTEGER WIDTH: fxmul/fxdiv/FxISqrt/FxNormalize + the vFluidAvg sum use int64_t (the cgrain_support.comp /
// grain_contact_dp.comp lesson). DXC compiles int64 (Vulkan); glslc CANNOT parse int64 in HLSL -> this shader
// is VULKAN-SPIR-V-ONLY (NOT in metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --cgf-buoyancy
// showcase runs the CPU cgf::StepCGFBuoyancy (byte-identical by construction). The GF1 cross-query passes
// (re-run each step) stay int32 MSL-native.
//
// Buffers (storage, bound at compute bindings 0..5; Vulkan-only):
//   b0 gGrains   : the GrainParticle array (48 bytes; pos/vel/invMass/flags read, vel WRITTEN), READ+WRITE.
//   b1 gfStart   : the GF1 CSR offsets (grainCount+1), READ.
//   b2 gfNbr     : the GF1 gathered fluid indices grouped by grain (ascending), READ.
//   b3 gFluid    : the FluidParticle array (44 bytes; vel read), READ.
//   b4 gParams   : the CGFBuoyancyParams (grainCount, enabled, buoyPerFluid, kDrag, gravity, dt), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_CGF_THREADS 64
#define HF_CGF_FRAC 16
#define HF_CGF_STATIC 1u        // == grain::kFlagStatic (bit0)

// The grain pack (std430, 48 bytes — matches GrainParticleGpu; pos + vel + invMass + flags read/written).
struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

// The fluid pack (std430, 44 bytes — matches FluidParticleGpu; vel is read).
struct FluidParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
};

// CGFBuoyancyParams (std430). cfg {grainCount, enabled, _, _} + grav {gx, gy, gz, dt} + coef {kBuoyPerFluid, kDrag, _, _}.
struct CGFBuoyancyParams { int4 cfg; int4 grav; int4 coef; };

[[vk::binding(0, 0)]] RWStructuredBuffer<GrainParticle>     gGrains  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>              gfStart  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>              gfNbr    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<FluidParticle>     gFluid   : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<CGFBuoyancyParams> gParams  : register(u4);

// fxmul/fxdiv — VERBATIM fpx.h::fxmul / fpx.h::fxdiv (int64 intermediate).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_CGF_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_CGF_FRAC) / (int64_t)b); }

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

[numthreads(HF_CGF_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int grainCount = gParams[0].cfg.x;
    int enabled    = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= grainCount) return;
    if (enabled == 0) return;

    GrainParticle g = gGrains[i];
    if ((g.flags & HF_CGF_STATIC) != 0u) return;          // static boundary grain -> untouched (the pinned case)

    uint s0 = gfStart[i], s1 = gfStart[i + 1u];
    uint cnt = s1 - s0;
    if (cnt == 0u) return;                                // dry grain (no fluid neighbours) -> free GR sim

    int dt = gParams[0].grav.w;
    int gx = gParams[0].grav.x, gy = gParams[0].grav.y, gz = gParams[0].grav.z;
    int kBuoyPerFluid = gParams[0].coef.x, kDrag = gParams[0].coef.y;
    int oneQ = (int)1 << HF_CGF_FRAC;                     // kOne (the FxNormalize +Y fallback)

    // up = -FxNormalize(gravity), VERBATIM (len==0 -> the +Y fallback; up = -(0,kOne,0) = (0,-kOne,0)).
    int glen = FxLength3(gx, gy, gz);
    int upx, upy, upz;
    if (glen == 0) { upx = 0; upy = -oneQ; upz = 0; }
    else { upx = -fxdiv(gx, glen); upy = -fxdiv(gy, glen); upz = -fxdiv(gz, glen); }

    // BUOYANCY: F_buoy = up * fxmul(kBuoyPerFluid, cnt<<kFrac) — ∝ the fluid count, opposing gravity.
    int buoyMag = fxmul(kBuoyPerFluid, (int)(cnt << HF_CGF_FRAC));
    int fbx = fxmul(upx, buoyMag), fby = fxmul(upy, buoyMag), fbz = fxmul(upz, buoyMag);

    // DRAG: vFluidAvg = (Σ fluid[j].vel) / cnt, the FIXED-ORDER int64 sum / the integer count.
    int64_t sumX = 0, sumY = 0, sumZ = 0;
    for (uint s = s0; s < s1; ++s) {
        FluidParticle f = gFluid[gfNbr[s]];
        sumX += (int64_t)f.vx;
        sumY += (int64_t)f.vy;
        sumZ += (int64_t)f.vz;
    }
    int avgX = (int)(sumX / (int64_t)cnt);
    int avgY = (int)(sumY / (int64_t)cnt);
    int avgZ = (int)(sumZ / (int64_t)cnt);
    // F_drag = kDrag * (vFluidAvg - grain.vel) per axis.
    int fdx = fxmul(kDrag, avgX - g.vx);
    int fdy = fxmul(kDrag, avgY - g.vy);
    int fdz = fxmul(kDrag, avgZ - g.vz);

    // Apply the impulse: vel += (F_buoy + F_drag) * invMass * dt (linear only — grains are points).
    int ftx = fbx + fdx, fty = fby + fdy, ftz = fbz + fdz;
    int dvx = fxmul(fxmul(ftx, g.invMass), dt);
    int dvy = fxmul(fxmul(fty, g.invMass), dt);
    int dvz = fxmul(fxmul(ftz, g.invMass), dt);
    g.vx += dvx; g.vy += dvy; g.vz += dvz;

    gGrains[i] = g;   // per-grain independent write (NO atomics — the grains are disjoint)
}
