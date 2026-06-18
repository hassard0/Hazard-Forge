// Slice CP2 — Deterministic Rigid<->Fluid Coupling: the BUOYANCY + DRAG pass (fluid->body, the CRUX; the 2nd
// slice of FLAGSHIP #11). ONE thread per BODY i (i < bodyCount) — multi-thread OVER bodies, each thread
// SERIALLY sums its body's CP1 gathered fluid-particle list (bodyParticles[bodyStart[i]..bodyStart[i+1]),
// ascending — the FIXED CP1 emit order) into a buoyant + drag velocity delta, then writes the body's new vel.
// NOT per particle: the body count is TINY (1-few), so each body's short inner loop is far below the ~2s
// watchdog — NO single-thread [numthreads(1,1,1)] / TDR ceiling (the CL3/FPX3 limit does not apply). If body
// counts ever scale up, a deterministic integer atomic-add reduction would be needed (OUT of scope, CP1-CP6).
//
// The body is copied VERBATIM from couple.h::AccumBodyForces (linear only — a sphere body has no buoyancy
// torque, so angVel is untouched; the host driver runs IntegrateBody + ResolveGround AFTER this pass):
//   up = -FxNormalize(gravity)
//   F_buoy = fxmul(kBuoyPerParticle, count<<16) · up        // ∝ the gathered count (the displaced volume)
//   vFluidAvg = (Σ_j particle[j].vel) / count               // fixed-order int64 sum / the integer count
//   F_drag = fxmul(kDrag, vFluidAvg − body.vel)             // per axis
//   body.vel += FxScale(F_buoy + F_drag, body.invMass) · dt
// A body that is static (no kFlagDynamic) or gathers 0 particles is untouched. A divergence vs the header is
// exactly what the host's GPU==CPU memcmp (the settled body array) catches.
//
// INTEGER WIDTH: fxmul/fxdiv/FxISqrt/FxNormalize + the vFluidAvg sum use int64_t (the fluid_dp.comp /
// grain_contact_dp.comp lesson). DXC compiles int64 (Vulkan); glslc CANNOT parse int64 in HLSL -> this shader
// is VULKAN-SPIR-V-ONLY (NOT in metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --couple-buoyancy
// showcase runs the CPU couple::StepCoupleBuoyancy (byte-identical by construction). The CP1 query passes
// (re-run each step) stay int32 MSL-native.
//
// Buffers (storage, bound at compute bindings 0..4; Vulkan-only):
//   b0 gBodies        : the CoupleBodyGpu array (pos.xyz, vel.xyz, invMass, flags, radius), READ+WRITE (vel).
//   b1 bodyStart      : the CP1 CSR offsets (bodyCount+1), READ.
//   b2 bodyParticles  : the CP1 gathered fluid-particle indices grouped by body (ascending), READ.
//   b3 gParticles     : the FluidParticleGpu array (44 bytes; only vel is read), READ.
//   b4 gParams        : the CoupleBuoyParams (bodyCount, enabled, gravity, dt, kBuoy, kDrag), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_CP_THREADS 64
#define HF_CP_FRAC 16
#define HF_CP_DYNAMIC 1u        // == fpx::kFlagDynamic (bit0)

// The body pack (std430): pos.xyz, vel.xyz, invMass, flags, radius = 9 x 32-bit (36 bytes).
struct CoupleBodyGpu {
    int  px, py, pz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
    int  radius;
};

// The fluid particle pack (std430, 44 bytes — matches FluidParticleGpu; only vel is read here).
struct FluidParticleGpu {
    int  px, py, pz, prx, pry, prz, vx, vy, vz, invMass;
    uint flags;
};

// CoupleBuoyParams (std430). cfg {bodyCount, enabled, _, _} + grav {gx, gy, gz, dt} + coef {kBuoy, kDrag, _, _}.
struct CoupleBuoyParams { int4 cfg; int4 grav; int4 coef; };

[[vk::binding(0, 0)]] RWStructuredBuffer<CoupleBodyGpu>     gBodies       : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>             bodyStart     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>             bodyParticles : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<FluidParticleGpu> gParticles    : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<CoupleBuoyParams> gParams       : register(u4);

// fxmul/fxdiv — VERBATIM fpx.h::fxmul / fpx.h::fxdiv (int64 intermediate).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_CP_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_CP_FRAC) / (int64_t)b); }

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

[numthreads(HF_CP_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int bodyCount = gParams[0].cfg.x;
    int enabled   = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= bodyCount) return;
    if (enabled == 0) return;

    CoupleBodyGpu b = gBodies[i];
    if ((b.flags & HF_CP_DYNAMIC) == 0u) return;          // static/kinematic -> untouched (the pinned case)

    uint s0 = bodyStart[i], s1 = bodyStart[i + 1u];
    uint count = s1 - s0;
    if (count == 0u) return;                              // gathers nothing (out of the pool) -> free-fall

    int gx = gParams[0].grav.x, gy = gParams[0].grav.y, gz = gParams[0].grav.z;
    int dt = gParams[0].grav.w;
    int kBuoy = gParams[0].coef.x, kDrag = gParams[0].coef.y;

    // up = -FxNormalize(gravity). gravity 0 -> the +Y fallback (VERBATIM FxNormalize).
    int ngx = -gx, ngy = -gy, ngz = -gz;
    int glen = FxLength3(ngx, ngy, ngz);
    int upx, upy, upz;
    if (glen == 0) { upx = 0; upy = (int)1 << HF_CP_FRAC; upz = 0; }   // kOne == 1<<16
    else { upx = fxdiv(ngx, glen); upy = fxdiv(ngy, glen); upz = fxdiv(ngz, glen); }

    // BUOYANCY: F_buoy = kBuoy * (count<<16) along up (count<<16 promotes the integer count to Q16.16).
    int countFx = (int)(count << HF_CP_FRAC);
    int buoyMag = fxmul(kBuoy, countFx);
    int fbx = fxmul(upx, buoyMag), fby = fxmul(upy, buoyMag), fbz = fxmul(upz, buoyMag);

    // DRAG: vFluidAvg = (Σ_j particle[j].vel) / count, summed in the FIXED CP1 gathered (ascending) order.
    int64_t sumX = 0, sumY = 0, sumZ = 0;
    for (uint s = s0; s < s1; ++s) {
        FluidParticleGpu p = gParticles[bodyParticles[s]];
        sumX += (int64_t)p.vx;
        sumY += (int64_t)p.vy;
        sumZ += (int64_t)p.vz;
    }
    int avgX = (int)(sumX / (int64_t)count);
    int avgY = (int)(sumY / (int64_t)count);
    int avgZ = (int)(sumZ / (int64_t)count);
    // F_drag = kDrag * (vFluidAvg - body.vel) per axis.
    int fdx = fxmul(kDrag, avgX - b.vx);
    int fdy = fxmul(kDrag, avgY - b.vy);
    int fdz = fxmul(kDrag, avgZ - b.vz);

    // Apply the impulse: vel += (F_buoy + F_drag) * invMass * dt (linear only — angVel untouched).
    int ftx = fbx + fdx, fty = fby + fdy, ftz = fbz + fdz;
    int dvx = fxmul(fxmul(ftx, b.invMass), dt);
    int dvy = fxmul(fxmul(fty, b.invMass), dt);
    int dvz = fxmul(fxmul(ftz, b.invMass), dt);
    b.vx += dvx; b.vy += dvy; b.vz += dvz;

    gBodies[i] = b;   // per-body independent write (NO atomics — the bodies are disjoint)
}
