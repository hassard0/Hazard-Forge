// Slice CG2 — Deterministic Rigid<->Grain Coupling: the CONTACT SUPPORT + DRAG pass (grain->body, the CRUX;
// the 2nd slice of FLAGSHIP #12). ONE thread per BODY i (i < bodyCount) — multi-thread OVER bodies, each
// thread SERIALLY sums its body's CG1 gathered grain list (bodyGrains[bodyStart[i]..bodyStart[i+1]),
// ascending — the FIXED CG1 emit order) into a contact-support + drag velocity delta, then writes the body's
// new vel. NOT per grain: the body count is TINY (1-few), so each body's inner loop is far below the ~2s
// watchdog — NO single-thread [numthreads(1,1,1)] / TDR ceiling (the CL3/FPX3 limit does not apply). CAVEAT:
// a body resting ON a sand bed gathers MORE grains than a body floating in fluid (a bed supports through many
// simultaneous contacts), so the per-body inner loop is LONGER than CP2's — still bounded for a tiny body
// count, but if body counts ever scale up a deterministic integer atomic-add reduction is needed (OUT of
// scope, CG1-CG6 — the swraster 64-bit-atomics caveat).
//
// The body is copied VERBATIM from couple_grain.h::AccumBodyGrainForces (linear only — a sphere body has no
// support torque, so angVel is untouched; the host driver runs IntegrateBody + ResolveGround AFTER this pass):
//   for each gathered grain j (ascending):
//     d = body.pos − grain[j].pos ; dist = FxLength(d) ; pen = (body.radius + grain[j].radius) − dist
//     if (pen > 0) { n = FxNormalize(d); F_support += FxScale(n, fxmul(kSupport, pen)); }   // ∝ penetration
//     sum += grain[j].vel
//   vGrainAvg = sum / count                                   // fixed-order int64 sum / the integer count
//   F_drag = fxmul(kDrag, vGrainAvg − body.vel)               // per axis
//   body.vel += FxScale(F_support + F_drag, body.invMass) · dt
// A body that is static (no kFlagDynamic) or gathers 0 grains is untouched. A divergence vs the header is
// exactly what the host's GPU==CPU memcmp (the settled body array) catches.
//
// INTEGER WIDTH: fxmul/fxdiv/FxISqrt/FxNormalize + the vGrainAvg sum use int64_t (the couple_buoyancy.comp /
// grain_contact_dp.comp lesson). DXC compiles int64 (Vulkan); glslc CANNOT parse int64 in HLSL -> this shader
// is VULKAN-SPIR-V-ONLY (NOT in metal_headless/CMakeLists.txt hf_gen_msl); on Metal the --cgrain-support
// showcase runs the CPU cgrain::StepCGrainSupport (byte-identical by construction). The CG1 query passes
// (re-run each step) stay int32 MSL-native.
//
// Buffers (storage, bound at compute bindings 0..4; Vulkan-only):
//   b0 gBodies    : the CGrainSupportBody array (pos.xyz, vel.xyz, invMass, flags, radius), READ+WRITE (vel).
//   b1 bodyStart  : the CG1 CSR offsets (bodyCount+1), READ.
//   b2 bodyGrains : the CG1 gathered grain indices grouped by body (ascending), READ.
//   b3 gGrains    : the GrainParticle array (48 bytes; pos + vel + radius are read), READ.
//   b4 gParams    : the CGrainSupportParams (bodyCount, enabled, gravity, dt, kSupport, kDrag), READ.
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations.

#define HF_CG_THREADS 64
#define HF_CG_FRAC 16
#define HF_CG_DYNAMIC 1u        // == fpx::kFlagDynamic (bit0)

// The body pack (std430): pos.xyz, vel.xyz, invMass, flags, radius = 9 x 32-bit (36 bytes). (== CoupleBodyGpu.)
struct CGrainSupportBody {
    int  px, py, pz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
    int  radius;
};

// The grain pack (std430, 48 bytes — matches GrainParticleGpu; pos + vel + radius are read here).
struct GrainParticle {
    int  px, py, pz;
    int  prx, pry, prz;
    int  vx, vy, vz;
    int  invMass;
    int  radius;
    uint flags;
};

// CGrainSupportParams (std430). cfg {bodyCount, enabled, _, _} + grav {gx, gy, gz, dt} + coef {kSupport, kDrag, _, _}.
struct CGrainSupportParams { int4 cfg; int4 grav; int4 coef; };

[[vk::binding(0, 0)]] RWStructuredBuffer<CGrainSupportBody>   gBodies    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>               bodyStart  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>               bodyGrains : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<GrainParticle>      gGrains    : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<CGrainSupportParams> gParams   : register(u4);

// fxmul/fxdiv — VERBATIM fpx.h::fxmul / fpx.h::fxdiv (int64 intermediate).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_CG_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_CG_FRAC) / (int64_t)b); }

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

[numthreads(HF_CG_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int bodyCount = gParams[0].cfg.x;
    int enabled   = gParams[0].cfg.y;

    uint i = gid.x;
    if ((int)i >= bodyCount) return;
    if (enabled == 0) return;

    CGrainSupportBody b = gBodies[i];
    if ((b.flags & HF_CG_DYNAMIC) == 0u) return;          // static/kinematic -> untouched (the pinned case)

    uint s0 = bodyStart[i], s1 = bodyStart[i + 1u];
    uint count = s1 - s0;
    if (count == 0u) return;                              // gathers nothing (clear of the bed) -> free-fall

    int dt = gParams[0].grav.w;
    int kSupport = gParams[0].coef.x, kDrag = gParams[0].coef.y;
    int oneQ = (int)1 << HF_CG_FRAC;                      // kOne (the FxNormalize +Y fallback)

    // SUPPORT: Σ over the gathered grains of the contact push (∝ penetration along the contact normal), summed
    // in the FIXED CG1 gathered (ascending) order. DRAG: vGrainAvg = (Σ grain.vel)/count, the int64 sum.
    int fsx = 0, fsy = 0, fsz = 0;
    int64_t sumX = 0, sumY = 0, sumZ = 0;
    for (uint s = s0; s < s1; ++s) {
        GrainParticle g = gGrains[bodyGrains[s]];
        int dx = b.px - g.px, dy = b.py - g.py, dz = b.pz - g.pz;   // body relative to the grain
        int dist = FxLength3(dx, dy, dz);
        int pen = (b.radius + g.radius) - dist;
        if (pen > 0) {
            // n = FxNormalize(d) (dist==0 -> the +Y fallback), VERBATIM.
            int nx, ny, nz;
            if (dist == 0) { nx = 0; ny = oneQ; nz = 0; }
            else { nx = fxdiv(dx, dist); ny = fxdiv(dy, dist); nz = fxdiv(dz, dist); }
            int mag = fxmul(kSupport, pen);
            fsx += fxmul(nx, mag); fsy += fxmul(ny, mag); fsz += fxmul(nz, mag);
        }
        sumX += (int64_t)g.vx;
        sumY += (int64_t)g.vy;
        sumZ += (int64_t)g.vz;
    }
    int avgX = (int)(sumX / (int64_t)count);
    int avgY = (int)(sumY / (int64_t)count);
    int avgZ = (int)(sumZ / (int64_t)count);
    // F_drag = kDrag * (vGrainAvg - body.vel) per axis.
    int fdx = fxmul(kDrag, avgX - b.vx);
    int fdy = fxmul(kDrag, avgY - b.vy);
    int fdz = fxmul(kDrag, avgZ - b.vz);

    // Apply the impulse: vel += (F_support + F_drag) * invMass * dt (linear only — angVel untouched).
    int ftx = fsx + fdx, fty = fsy + fdy, ftz = fsz + fdz;
    int dvx = fxmul(fxmul(ftx, b.invMass), dt);
    int dvy = fxmul(fxmul(fty, b.invMass), dt);
    int dvz = fxmul(fxmul(ftz, b.invMass), dt);
    b.vx += dvx; b.vy += dvy; b.vz += dvz;

    gBodies[i] = b;   // per-body independent write (NO atomics — the bodies are disjoint)
}
