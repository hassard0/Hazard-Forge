// Slice HF2 — Hull Friction + Joints: THE WARM CONE SOLVER compute pass (the friction SOLVER of FLAGSHIP #30:
// HULL FRICTION + HULL JOINTS, hf::sim::hullfric). The GPU mirror of engine/sim/hullfric.h::StepHullFrictionSolveOnly
// — ONE THREAD ([numthreads(1,1,1)]) runs the WHOLE solve over the SMALL pair set: per pair in the FIXED i<j order,
// warm-seed the HOST-BUILT keyed friction manifold from the cache, then run SolveHullFrictionWarm (the accumulated,
// warm-started, cone-clamped Coulomb friction solve over the hull manifold — the persist::SolveFrictionWarm body with
// the hull full-inertia tensor) over cfg.iters Gauss-Seidel sweeps, mutating the bodies IN PLACE so later pairs see
// earlier updates. HF2 solves a FIXED manifold set in place — NO integrate, NO de-pen (that is HF3). The final
// gBodies + gManifolds (the solved accumulators) array is byte-identical to the CPU StepHullFrictionSolveOnly -> the
// host GPU==CPU memcmp catches any divergence.
//
// THE int64 REALITY (the WH3/HF1 lesson): the warm-solve delta math (fxdiv/fxmul/FxDot Q16.16 world-scale products)
// is int64. DXC -spirv compiles int64; glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so
// this shader is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal hf_gen_msl list — the
// warmhull_warm.comp convention). The Metal --hf2-warm showcase runs the CPU hullfric::StepHullFrictionSolveOnly over
// the same scene -> byte-identical to this GPU result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU
// memcmp proof.
//
// THE HOST-BUILT MANIFOLD (the HF1 architecture): the keyed friction manifold (positions/normal/t1/t2/basis/keys) +
// the warm-seeded accumulators + the per-body FULL world inverse-inertia tensors are built HOST-SIDE (the int64
// GJK/EPA narrowphase + the warmhull cache match + the full-tensor inertia run on the host), so this shader does ONLY
// the SOLVE — it consumes the manifold + the inertia + the cache-seeded accumulators verbatim and runs the cone
// sweeps. This keeps the SPIR-V small (no GJK/EPA) — the solve math is the divergence surface the memcmp guards.
//
// SINGLE THREAD: the solve is order-dependent (the per-pair Gauss-Seidel + the in-place body mutation), so the GPU
// mirror MUST be single-thread (the warmhull_warm.comp convention). The pair set is small.
//
// stepEnabled=0 -> write the input bodies + manifolds back UNCHANGED (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..3):
//   b0 gBodies    : the body array (N x FxBody), READ+WRITE (solved in place).
//   b1 gManifolds : the host-built keyed friction manifold per pair (count + normal + t1 + t2 + 4 points + the
//                   bodyA/bodyB GLOBAL indices + the warm-seeded accumulators), READ+WRITE (accumulators solved).
//   b2 gInertia   : the per-body FULL world inverse-inertia tensor (N x 9 int), READ.
//   b3 gParams    : { pairCount, bodyCount, stepEnabled, iters | mu, restitution, _, _ }, READ.

#define HF_FPX_FRAC 16          // MUST match fpx.h::kFrac
#define HF_FPX_ONE  (1 << HF_FPX_FRAC)
#define HF_MAX_BODIES 16        // the small-scene cap
#define HF_MAX_PAIRS  16        // the small pair-set cap

// std430 FxBody mirror (engine/sim/fpx.h::FxBody): 16 x int32 (64 bytes).
struct FxBody {
    int  px, py, pz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
    int  radius;
    int  ox, oy, oz, ow;
    int  ax, ay, az;
};

// std430 HullFrictionPoint mirror (engine/sim/hullfric.h::HullFrictionPoint): point.xyz + 3 accumulators = 6 int32.
struct HullFrictionPointGpu {
    int px, py, pz;
    int normalImpulse;
    int tangentImpulse1;
    int tangentImpulse2;
};

// std430 keyed friction manifold mirror — the SOLVE-relevant fields HF2 reads (count + the A->B normal + the per-pair
// tangent basis + the 4 contact points/accumulators) PLUS the pair's GLOBAL body indices (so the single thread knows
// which two bodies a manifold couples). 1 + 3 + 3 + 3 + 2 = 12 leading int + 4 x 6 accumulators = 36 int (144 bytes).
struct HullFrictionManifoldGpu {
    uint count;
    int  nx, ny, nz;
    int  t1x, t1y, t1z;
    int  t2x, t2y, t2z;
    int  bodyA;          // GLOBAL body index of the pair's A body
    int  bodyB;          // GLOBAL body index of the pair's B body
    HullFrictionPointGpu fpts[4];
};

// The per-body FULL world inverse-inertia tensor (9 row-major int) packed std430.
struct InertiaGpu { int m[9]; int _pad[3]; };   // 12 int (48 bytes), pad to keep 16-byte alignment friendly

struct HullfricWarmParams {
    int4 c0;   // x=pairCount, y=bodyCount, z=stepEnabled, w=iters
    int4 c1;   // x=mu, y=restitution, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxBody>                  gBodies    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<HullFrictionManifoldGpu> gManifolds : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<InertiaGpu>              gInertia   : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<HullfricWarmParams>      gParams    : register(u3);

// ===== VERBATIM Q16.16 toolbox (== warmhull_warm.comp / hullfric.h / fric.h / convex.h / fpx.h) =====
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_FPX_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_FPX_FRAC) / (int64_t)b); }
int FxDot(int3 a, int3 b) {
    int64_t d = (int64_t)a.x * (int64_t)b.x + (int64_t)a.y * (int64_t)b.y + (int64_t)a.z * (int64_t)b.z;
    return (int)(d >> HF_FPX_FRAC);
}
int3 FxCross(int3 a, int3 b) {
    return int3(fxmul(a.y, b.z) - fxmul(a.z, b.y),
                fxmul(a.z, b.x) - fxmul(a.x, b.z),
                fxmul(a.x, b.y) - fxmul(a.y, b.x));
}
int3 FxScale(int3 v, int s) { return int3(fxmul(v.x, s), fxmul(v.y, s), fxmul(v.z, s)); }
int3 FxSub(int3 a, int3 b) { return int3(a.x - b.x, a.y - b.y, a.z - b.z); }
int3 FxAdd(int3 a, int3 b) { return int3(a.x + b.x, a.y + b.y, a.z + b.z); }
int3 Mat3MulVec(int M[9], int3 v) {
    int3 r0 = int3(M[0], M[1], M[2]);
    int3 r1 = int3(M[3], M[4], M[5]);
    int3 r2 = int3(M[6], M[7], M[8]);
    return int3(FxDot(r0, v), FxDot(r1, v), FxDot(r2, v));
}

// ===== SolveHullFrictionWarm (== hullfric.h) — the accumulated warm cone solve, full tensor. =====
// One pair: bodies bA/bB (in/out), their FULL world inverse-inertia ia/ib, the manifold km (accumulators in/out).
void SolveHullFrictionWarm(inout FxBody bA, inout FxBody bB, int ia[9], int ib[9],
                           inout HullFrictionManifoldGpu km, int mu, int rest, uint iters) {
    if (km.count == 0u) return;
    int invMassA = bA.invMass;
    int invMassB = bB.invMass;
    uint cnt = km.count < 4u ? km.count : 4u;
    int3 n  = int3(km.nx, km.ny, km.nz);
    int3 t1 = int3(km.t1x, km.t1y, km.t1z);
    int3 t2 = int3(km.t2x, km.t2y, km.t2z);

    // (1) PRIME ONCE: re-inject the seeded TOTAL impulse at every point (the warm-start kick).
    for (uint pi = 0u; pi < cnt; ++pi) {
        int3 p  = int3(km.fpts[pi].px, km.fpts[pi].py, km.fpts[pi].pz);
        int3 rA = int3(p.x - bA.px, p.y - bA.py, p.z - bA.pz);
        int3 rB = int3(p.x - bB.px, p.y - bB.py, p.z - bB.pz);
        int3 J = FxAdd(FxScale(n, km.fpts[pi].normalImpulse),
                       FxAdd(FxScale(t1, km.fpts[pi].tangentImpulse1),
                             FxScale(t2, km.fpts[pi].tangentImpulse2)));
        bA.vx -= fxmul(J.x, invMassA); bA.vy -= fxmul(J.y, invMassA); bA.vz -= fxmul(J.z, invMassA);
        int3 dwA = Mat3MulVec(ia, FxCross(rA, J));
        bA.ax -= dwA.x; bA.ay -= dwA.y; bA.az -= dwA.z;
        bB.vx += fxmul(J.x, invMassB); bB.vy += fxmul(J.y, invMassB); bB.vz += fxmul(J.z, invMassB);
        int3 dwB = Mat3MulVec(ib, FxCross(rB, J));
        bB.ax += dwB.x; bB.ay += dwB.y; bB.az += dwB.z;
    }

    // (2) the accumulated Gauss-Seidel sweeps (apply only the DELTA each time).
    for (uint it = 0u; it < iters; ++it) {
        for (uint pi = 0u; pi < cnt; ++pi) {
            int3 p  = int3(km.fpts[pi].px, km.fpts[pi].py, km.fpts[pi].pz);
            int3 rA = int3(p.x - bA.px, p.y - bA.py, p.z - bA.pz);
            int3 rB = int3(p.x - bB.px, p.y - bB.py, p.z - bB.pz);

            // ---- NORMAL impulse (accumulated, clamp >= 0) ----
            {
                int3 vpA = int3(bA.vx, bA.vy, bA.vz) + FxCross(int3(bA.ax, bA.ay, bA.az), rA);
                int3 vpB = int3(bB.vx, bB.vy, bB.vz) + FxCross(int3(bB.ax, bB.ay, bB.az), rB);
                int vn = FxDot(int3(vpB.x - vpA.x, vpB.y - vpA.y, vpB.z - vpA.z), n);
                int3 raxn = FxCross(rA, n);
                int3 rbxn = FxCross(rB, n);
                int angA = FxDot(n, FxCross(Mat3MulVec(ia, raxn), rA));
                int angB = FxDot(n, FxCross(Mat3MulVec(ib, rbxn), rB));
                int kn = invMassA + invMassB + angA + angB;
                if (kn > 0) {
                    int djn = fxdiv(-fxmul(HF_FPX_ONE + rest, vn), kn);
                    int newTotal = km.fpts[pi].normalImpulse + djn;
                    if (newTotal < 0) newTotal = 0;
                    int applied = newTotal - km.fpts[pi].normalImpulse;
                    int3 J = FxScale(n, applied);
                    bA.vx -= fxmul(J.x, invMassA); bA.vy -= fxmul(J.y, invMassA); bA.vz -= fxmul(J.z, invMassA);
                    int3 dwA = Mat3MulVec(ia, FxCross(rA, J));
                    bA.ax -= dwA.x; bA.ay -= dwA.y; bA.az -= dwA.z;
                    bB.vx += fxmul(J.x, invMassB); bB.vy += fxmul(J.y, invMassB); bB.vz += fxmul(J.z, invMassB);
                    int3 dwB = Mat3MulVec(ib, FxCross(rB, J));
                    bB.ax += dwB.x; bB.ay += dwB.y; bB.az += dwB.z;
                    km.fpts[pi].normalImpulse = newTotal;
                }
            }

            // ---- TANGENT impulses (t1 then t2), the cone on the ACCUMULATED normal ----
            int coneLo = -fxmul(mu, km.fpts[pi].normalImpulse);
            int coneHi =  fxmul(mu, km.fpts[pi].normalImpulse);
            for (int ti = 0; ti < 2; ++ti) {
                int3 t = (ti == 0) ? t1 : t2;
                int3 vpA = int3(bA.vx, bA.vy, bA.vz) + FxCross(int3(bA.ax, bA.ay, bA.az), rA);
                int3 vpB = int3(bB.vx, bB.vy, bB.vz) + FxCross(int3(bB.ax, bB.ay, bB.az), rB);
                int vt = FxDot(int3(vpB.x - vpA.x, vpB.y - vpA.y, vpB.z - vpA.z), t);
                int3 raxt = FxCross(rA, t);
                int3 rbxt = FxCross(rB, t);
                int angA = FxDot(t, FxCross(Mat3MulVec(ia, raxt), rA));
                int angB = FxDot(t, FxCross(Mat3MulVec(ib, rbxt), rB));
                int kt = invMassA + invMassB + angA + angB;
                if (kt <= 0) continue;
                int djt = fxdiv(-vt, kt);
                int prev = (ti == 0) ? km.fpts[pi].tangentImpulse1 : km.fpts[pi].tangentImpulse2;
                int newTotal = prev + djt;
                if (newTotal < coneLo) newTotal = coneLo;
                else if (newTotal > coneHi) newTotal = coneHi;
                int applied = newTotal - prev;
                int3 Jt = FxScale(t, applied);
                bA.vx -= fxmul(Jt.x, invMassA); bA.vy -= fxmul(Jt.y, invMassA); bA.vz -= fxmul(Jt.z, invMassA);
                int3 dwA = Mat3MulVec(ia, FxCross(rA, Jt));
                bA.ax -= dwA.x; bA.ay -= dwA.y; bA.az -= dwA.z;
                bB.vx += fxmul(Jt.x, invMassB); bB.vy += fxmul(Jt.y, invMassB); bB.vz += fxmul(Jt.z, invMassB);
                int3 dwB = Mat3MulVec(ib, FxCross(rB, Jt));
                bB.ax += dwB.x; bB.ay += dwB.y; bB.az += dwB.z;
                if (ti == 0) km.fpts[pi].tangentImpulse1 = newTotal;
                else         km.fpts[pi].tangentImpulse2 = newTotal;
            }
        }
    }
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // single thread runs the WHOLE solve (order-dependent)

    int pairCount   = gParams[0].c0.x;
    int bodyCount   = gParams[0].c0.y;
    int stepEnabled = gParams[0].c0.z;
    int iters       = gParams[0].c0.w;
    int mu          = gParams[0].c1.x;
    int restitution = gParams[0].c1.y;

    int nb = bodyCount; if (nb > HF_MAX_BODIES) nb = HF_MAX_BODIES;
    int np = pairCount; if (np > HF_MAX_PAIRS)  np = HF_MAX_PAIRS;

    // Load the bodies into registers (solved in place over the pair loop).
    FxBody body[HF_MAX_BODIES];
    for (int li = 0; li < nb; ++li) body[li] = gBodies[li];

    if (stepEnabled == 0) {
        for (int wi0 = 0; wi0 < nb; ++wi0) gBodies[wi0] = body[wi0];
        return;   // manifolds untouched
    }

    // The FIXED pair order (the host already laid the manifolds out in i<j order). Each manifold names its two
    // GLOBAL body indices; the solve mutates those bodies IN PLACE so later pairs see earlier updates.
    for (int pp = 0; pp < np; ++pp) {
        HullFrictionManifoldGpu km = gManifolds[pp];
        if (km.count == 0u) continue;
        int ai = km.bodyA; int bi = km.bodyB;
        if (ai < 0 || ai >= nb || bi < 0 || bi >= nb) continue;
        int ia[9]; int ib[9];
        [unroll] for (int z = 0; z < 9; ++z) { ia[z] = gInertia[ai].m[z]; ib[z] = gInertia[bi].m[z]; }
        FxBody bA = body[ai]; FxBody bB = body[bi];
        SolveHullFrictionWarm(bA, bB, ia, ib, km, mu, restitution, (uint)iters);
        body[ai] = bA; body[bi] = bB;
        gManifolds[pp] = km;   // write back the solved accumulators
    }

    for (int wi = 0; wi < nb; ++wi) gBodies[wi] = body[wi];
}
