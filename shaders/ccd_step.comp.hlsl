// Slice CD3 — Deterministic Integer CCD: THE SUBSTEPPED CCD WORLD STEP (the mechanism beat) compute pass (the
// 3rd slice of FLAGSHIP #24: hf::sim::ccd). ONE THREAD ([numthreads(1,1,1)]) runs the WHOLE StepHullWorldCCDN
// over a MODERATE scene (a fast mover aimed at a thin wall + a few slow settling hulls) — each tick: a bounded
// substep loop that (1) swept-broadphases the candidate pairs, (2) per-pair conservative-advancement TOI ->
// the GLOBAL-MIN TOI, (3) advances ALL dynamic bodies to that earliest impact, (4) resolves the touching pairs
// via the frozen GJK/EPA narrowphase + impulse + de-pen, (5) consumes the remaining dt and repeats — copying
// engine/sim/ccd.h::StepHullWorldCCD / StepHullWorldCCDN VERBATIM. The final gBodies array is byte-identical to
// the CPU ccd::StepHullWorldCCDN reference -> the host GPU==CPU memcmp catches any divergence.
//
// THE SWEPT BROADPHASE ON THE GPU (the bit-transparency): the CPU StepHullWorldCCD builds the candidate pair
// list each substep via BuildSweptBroadphasePairs (the BP2 grid over swept-AABB centres, (i,j)-sorted). This
// shader reproduces the candidate SET by the nested for-i: for-j>i loop GATED on the SWEPT-AABB AabbOverlap:
// the grid set is a superset of every swept-AABB-overlapping pair, and the all-pairs swept gate is EXACTLY the
// swept-AABB-overlapping pairs, so any CPU-grid-extra pair is genuinely non-swept-overlapping -> its
// ConservativeAdvance returns hit=0 -> it affects NEITHER the global-min TOI NOR the resolve set. The
// per-substep min-TOI is order-invariant (a global min, lowest-pair-index tie); the resolve iterates the
// HIT pairs in (i,j)-ascending order, matching the CPU's sorted candidate order.
//
// SINGLE THREAD ([numthreads(1,1,1)]): the world step is order-dependent (the world Gauss-Seidel resolve + the
// in-place de-pen + the sequential substeps), so the GPU mirror MUST be single-thread (the broad_hull_step.comp
// convention). A CCD tick is the HEAVIEST single-thread dispatch in the suite (swept broadphase + N TOI loops
// each an embedded Gjk + a resolve, x maxSubsteps), so the host CHUNKS at 1 TICK/DISPATCH from the start with a
// ComputeToComputeBarrier between (TDR-safe by construction; the body buffer carries the exact state across the
// barrier so the chunked result is BIT-IDENTICAL to one big dispatch).
//
// INTEGER WIDTH (the determinism crux): fxmul/fxdiv/FxISqrt + the FxDot/FxCross/FxRotate/FxNormalize/quaternion
// Q16.16 products + the GJK sub-distance + the EPA face math + the Support queries all use int64_t. DXC -spirv
// compiles int64 (the broad_hull_step.comp pattern); glslc CANNOT parse int64_t in HLSL, so this shader is
// VULKAN-SPIR-V-ONLY (NOT in the Metal hf_gen_msl list). The Metal --ccd-step runs the CPU StepHullWorldCCDN
// -> byte-identical by construction, while the Vulkan side carries the GPU==CPU proof.
//
// stepEnabled=0 -> write the input bodies back UNCHANGED (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..2):
//   b0 gBodies : the body array (N x FxBody), READ+WRITE (stepped in place over K ticks).
//   b1 gHulls  : the FxHull array (N x {kMaxHullVerts x int3 verts + uint count}), READ.
//   b2 gParams : the step config + counts, READ.

#define HF_FPX_FRAC 16          // MUST match fpx.h::kFrac
#define HF_FPX_ONE  (1 << HF_FPX_FRAC)
#define HF_FPX_HALF (HF_FPX_ONE / 2)         // kHalf
#define HF_GJK_EDGE_EPS (HF_FPX_ONE / 256)   // MUST match convex.h::kEdgeEps == kOne/256
#define HF_EPA_TOL (HF_FPX_ONE / 256)        // MUST match gjk.h::kEpaTol == kOne/256
#define HF_FLAG_DYNAMIC 1u                   // MUST match fpx.h::kFlagDynamic
#define HF_GJK_MAX_VERTS 20                  // MUST match gjk.h::kMaxHullVerts
#define HF_GJK_MAX_ITER 32                   // MUST match gjk.h::kGjkMaxIter
#define HF_EPA_MAX_ITER 48                   // MUST match gjk.h::kEpaMaxIter
#define HF_EPA_MAX_PV 64                     // MUST match gjk.h::kMaxPolyVerts
#define HF_EPA_MAX_PF 128                    // MUST match gjk.h::kMaxPolyFaces
#define HF_MAX_BODIES 32                     // the moderate CCD-scene cap (a fast mover + a few slow hulls)
#define HF_CCD_CONTACT_EPS (HF_FPX_ONE / 64) // MUST match ccd.h::kContactEps
#define HF_CCD_TOI_MAX_ITER 32               // MUST match ccd.h::kToiMaxIter

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

// std430 FxHull mirror (engine/sim/gjk.h::FxHull): flat int arrays (vx,vy,vz) + count.
struct FxHull {
    int  vx[HF_GJK_MAX_VERTS];
    int  vy[HF_GJK_MAX_VERTS];
    int  vz[HF_GJK_MAX_VERTS];
    uint count;
};

// The step params (== the host CcdStepConfig + counts), packed for std430.
struct StepParams {
    int4 c0;   // x=bodyCount, y=ticks, z=stepEnabled, w=solveIters
    int4 c1;   // x=posIters, y=restitution, z=slop, w=beta
    int4 c2;   // x=linDamp, y=angDamp, z=dt, w=maxSubsteps
    int4 c3;   // x=gravity.x, y=gravity.y, z=gravity.z, w=_pad
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxBody>     gBodies : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FxHull>     gHulls  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<StepParams> gParams : register(u2);

// ===== VERBATIM Q16.16 toolbox (== broad_hull_step.comp / ccd_toi.comp / convex.h / fpx.h) =====
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
int3 FxNeg(int3 v) { return int3(-v.x, -v.y, -v.z); }
int3 FxScale(int3 v, int s) { return int3(fxmul(v.x, s), fxmul(v.y, s), fxmul(v.z, s)); }
int3 FxSub(int3 a, int3 b) { return int3(a.x - b.x, a.y - b.y, a.z - b.z); }
int3 FxAdd(int3 a, int3 b) { return int3(a.x + b.x, a.y + b.y, a.z + b.z); }
int64_t FxISqrt64(int64_t v) {
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
int FxLength(int3 v) {
    int64_t s = (int64_t)v.x * (int64_t)v.x + (int64_t)v.y * (int64_t)v.y + (int64_t)v.z * (int64_t)v.z;
    return (int)FxISqrt64(s);
}
int FxLength4(int4 v) {
    int64_t s = (int64_t)v.x * (int64_t)v.x + (int64_t)v.y * (int64_t)v.y
              + (int64_t)v.z * (int64_t)v.z + (int64_t)v.w * (int64_t)v.w;
    return (int)FxISqrt64(s);
}
int3 FxNormalize(int3 v) {
    int len = FxLength(v);
    if (len == 0) return int3(0, HF_FPX_ONE, 0);
    return int3(fxdiv(v.x, len), fxdiv(v.y, len), fxdiv(v.z, len));
}
int3 FxRotate(int4 q, int3 v) {
    int3 u = int3(q.x, q.y, q.z);
    int3 c1 = int3(fxmul(u.y, v.z) - fxmul(u.z, v.y),
                   fxmul(u.z, v.x) - fxmul(u.x, v.z),
                   fxmul(u.x, v.y) - fxmul(u.y, v.x));
    int3 t = int3(c1.x + fxmul(q.w, v.x), c1.y + fxmul(q.w, v.y), c1.z + fxmul(q.w, v.z));
    int3 c2 = int3(fxmul(u.y, t.z) - fxmul(u.z, t.y),
                   fxmul(u.z, t.x) - fxmul(u.x, t.z),
                   fxmul(u.x, t.y) - fxmul(u.y, t.x));
    return int3(v.x + 2 * c2.x, v.y + 2 * c2.y, v.z + 2 * c2.z);
}
int3 Mat3MulVec(int M[9], int3 v) {
    int3 r0 = int3(M[0], M[1], M[2]);
    int3 r1 = int3(M[3], M[4], M[5]);
    int3 r2 = int3(M[6], M[7], M[8]);
    return int3(FxDot(r0, v), FxDot(r1, v), FxDot(r2, v));
}

// ===== Quaternion integrate (== fpx.h::FxQuatMul/FxQuatNormalize/IntegrateBodyFull) =====
int4 FxQuatMul(int4 a, int4 b) {
    int4 r;
    r.w = fxmul(a.w, b.w) - fxmul(a.x, b.x) - fxmul(a.y, b.y) - fxmul(a.z, b.z);
    r.x = fxmul(a.w, b.x) + fxmul(a.x, b.w) + fxmul(a.y, b.z) - fxmul(a.z, b.y);
    r.y = fxmul(a.w, b.y) - fxmul(a.x, b.z) + fxmul(a.y, b.w) + fxmul(a.z, b.x);
    r.z = fxmul(a.w, b.z) + fxmul(a.x, b.y) - fxmul(a.y, b.x) + fxmul(a.z, b.w);
    return r;
}
int4 FxQuatNormalize(int4 q) {
    int len = FxLength4(q);
    if (len == 0) return int4(0, 0, 0, HF_FPX_ONE);
    return int4(fxdiv(q.x, len), fxdiv(q.y, len), fxdiv(q.z, len), fxdiv(q.w, len));
}
// IntegrateBodyFull mutating a body in place (== fpx.h::IntegrateBodyFull): pos from vel (dynamic only) +
// orient from angVel by `dt`, with the given gravity applied to vel.
void IntegrateBodyFull(inout FxBody b, int3 gravity, int dt) {
    if (b.flags & HF_FLAG_DYNAMIC) {
        b.vx += fxmul(gravity.x, dt);
        b.vy += fxmul(gravity.y, dt);
        b.vz += fxmul(gravity.z, dt);
        b.px += fxmul(b.vx, dt);
        b.py += fxmul(b.vy, dt);
        b.pz += fxmul(b.vz, dt);
    }
    int4 q = int4(b.ox, b.oy, b.oz, b.ow);
    int4 omega = int4(b.ax, b.ay, b.az, 0);
    int4 dq = FxQuatMul(omega, q);
    q.x += fxmul(fxmul(dq.x, HF_FPX_HALF), dt);
    q.y += fxmul(fxmul(dq.y, HF_FPX_HALF), dt);
    q.z += fxmul(fxmul(dq.z, HF_FPX_HALF), dt);
    q.w += fxmul(fxmul(dq.w, HF_FPX_HALF), dt);
    q = FxQuatNormalize(q);
    b.ox = q.x; b.oy = q.y; b.oz = q.z; b.ow = q.w;
}
// A POSE-ONLY integrate (== ccd.h ConservativeAdvance's copy advance / SweptHullAabb's end pose): ZERO gravity,
// pos from vel (dynamic only) + orient from angVel by `dt`. Operates on loose pos/vel/orient/angVel.
void IntegratePoseZeroG(inout int3 pos, int3 vel, inout int4 orient, int3 angVel, uint flags, int dt) {
    if ((flags & HF_FLAG_DYNAMIC) != 0u) {
        pos.x += fxmul(vel.x, dt);
        pos.y += fxmul(vel.y, dt);
        pos.z += fxmul(vel.z, dt);
    }
    int4 omega = int4(angVel.x, angVel.y, angVel.z, 0);
    int4 dq = FxQuatMul(omega, orient);
    orient.x += fxmul(fxmul(dq.x, HF_FPX_HALF), dt);
    orient.y += fxmul(fxmul(dq.y, HF_FPX_HALF), dt);
    orient.z += fxmul(fxmul(dq.z, HF_FPX_HALF), dt);
    orient.w += fxmul(fxmul(dq.w, HF_FPX_HALF), dt);
    orient = FxQuatNormalize(orient);
}

// ===== FxHullInvInertiaBody / WorldInvInertia (== gjk.h / convex.h, broad_hull_step.comp) =====
int3 HullInvInertiaBody(FxHull hull, int invMass) {
    if (invMass == 0) return int3(0, 0, 0);
    int hx = 0, hy = 0, hz = 0;
    for (uint i = 0u; i < hull.count; ++i) {
        int ax = abs(hull.vx[i]), ay = abs(hull.vy[i]), az = abs(hull.vz[i]);
        if (ax > hx) hx = ax;
        if (ay > hy) hy = ay;
        if (az > hz) hz = az;
    }
    int hx2 = fxmul(hx, hx), hy2 = fxmul(hy, hy), hz2 = fxmul(hz, hz);
    int three = 3 * invMass;
    return int3(fxdiv(three, hy2 + hz2), fxdiv(three, hx2 + hz2), fxdiv(three, hx2 + hy2));
}
void WorldInvInertia(int4 q, int3 invI, out int M[9]) {
    int3 ax0 = FxRotate(q, int3(HF_FPX_ONE, 0, 0));
    int3 ax1 = FxRotate(q, int3(0, HF_FPX_ONE, 0));
    int3 ax2 = FxRotate(q, int3(0, 0, HF_FPX_ONE));
    [unroll] for (int z = 0; z < 9; ++z) M[z] = 0;
    int3 ax[3] = { ax0, ax1, ax2 };
    int d[3] = { invI.x, invI.y, invI.z };
    for (int kk = 0; kk < 3; ++kk) {
        int3 a = ax[kk];
        int dk = d[kk];
        int da0 = fxmul(dk, a.x), da1 = fxmul(dk, a.y), da2 = fxmul(dk, a.z);
        M[0] += fxmul(da0, a.x); M[1] += fxmul(da0, a.y); M[2] += fxmul(da0, a.z);
        M[3] += fxmul(da1, a.x); M[4] += fxmul(da1, a.y); M[5] += fxmul(da1, a.z);
        M[6] += fxmul(da2, a.x); M[7] += fxmul(da2, a.y); M[8] += fxmul(da2, a.z);
    }
}

// ===== Support (VERBATIM gjk.h::SupportLocal / Support) =====
int3 SupportLocal(FxHull hull, int3 dir) {
    if (hull.count == 0u) return int3(0, 0, 0);
    uint best = 0u;
    int3 v0 = int3(hull.vx[0], hull.vy[0], hull.vz[0]);
    int bestDot = FxDot(v0, dir);
    for (uint i = 1u; i < hull.count; ++i) {
        int3 vi = int3(hull.vx[i], hull.vy[i], hull.vz[i]);
        int d = FxDot(vi, dir);
        if (d > bestDot) { bestDot = d; best = i; }
    }
    return int3(hull.vx[best], hull.vy[best], hull.vz[best]);
}
int3 Support(FxHull hull, int4 orient, int3 pos, int3 dir) {
    int4 conj = int4(-orient.x, -orient.y, -orient.z, orient.w);
    int3 localDir = FxRotate(conj, dir);
    int3 localV = SupportLocal(hull, localDir);
    int3 worldV = FxRotate(orient, localV);
    return int3(worldV.x + pos.x, worldV.y + pos.y, worldV.z + pos.z);
}
int3 SupportMinkowski(FxHull hA, int4 oA, int3 pA, FxHull hB, int4 oB, int3 pB, int3 dir) {
    int3 sa = Support(hA, oA, pA, dir);
    int3 sb = Support(hB, oB, pB, FxNeg(dir));
    return int3(sa.x - sb.x, sa.y - sb.y, sa.z - sb.z);
}

// ===== The Johnson sub-distance (VERBATIM gjk.h::DoSimplex2/3/4) =====
struct Sub { int3 dir; uint keep[4]; int w[4]; uint size; bool containsOrigin; };

Sub DoSimplex2(int3 a, int3 b) {
    Sub r; r.containsOrigin = false; r.dir = int3(0,0,0);
    r.keep[0]=0;r.keep[1]=0;r.keep[2]=0;r.keep[3]=0; r.w[0]=0;r.w[1]=0;r.w[2]=0;r.w[3]=0; r.size=0;
    int3 ab = int3(b.x-a.x, b.y-a.y, b.z-a.z);
    int3 ao = FxNeg(a);
    int abLen = FxLength(ab);
    if (abLen < HF_GJK_EDGE_EPS) { r.size=1; r.keep[0]=0; r.w[0]=HF_FPX_ONE; r.dir=ao; return r; }
    int d1 = FxDot(ao, ab);
    if (d1 <= 0) { r.size=1; r.keep[0]=0; r.w[0]=HF_FPX_ONE; r.dir=ao; return r; }
    int d2 = FxDot(ab, ab);
    if (d1 >= d2) { r.size=1; r.keep[0]=1; r.w[0]=HF_FPX_ONE; r.dir=FxNeg(b); return r; }
    int t = fxdiv(d1, d2);
    int3 closest = int3(a.x + fxmul(ab.x,t), a.y + fxmul(ab.y,t), a.z + fxmul(ab.z,t));
    r.size=2; r.keep[0]=0; r.keep[1]=1; r.w[0]=HF_FPX_ONE - t; r.w[1]=t; r.dir=FxNeg(closest);
    return r;
}
Sub DoSimplex3(int3 a, int3 b, int3 c) {
    Sub r; r.containsOrigin = false; r.dir = int3(0,0,0);
    r.keep[0]=0;r.keep[1]=0;r.keep[2]=0;r.keep[3]=0; r.w[0]=0;r.w[1]=0;r.w[2]=0;r.w[3]=0; r.size=0;
    int3 ab = int3(b.x-a.x, b.y-a.y, b.z-a.z);
    int3 ac = int3(c.x-a.x, c.y-a.y, c.z-a.z);
    int3 ao = FxNeg(a);
    int d1 = FxDot(ab, ao);
    int d2 = FxDot(ac, ao);
    if (d1 <= 0 && d2 <= 0) { r.size=1; r.keep[0]=0; r.w[0]=HF_FPX_ONE; r.dir=ao; return r; }
    int3 bo = FxNeg(b);
    int d3 = FxDot(ab, bo);
    int d4 = FxDot(ac, bo);
    if (d3 >= 0 && d4 <= d3) { r.size=1; r.keep[0]=1; r.w[0]=HF_FPX_ONE; r.dir=bo; return r; }
    int vc = fxmul(d1, d4) - fxmul(d3, d2);
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        int denom = d1 - d3;
        int t = (denom > HF_GJK_EDGE_EPS || denom < -HF_GJK_EDGE_EPS) ? fxdiv(d1, denom) : 0;
        int3 closest = int3(a.x + fxmul(ab.x,t), a.y + fxmul(ab.y,t), a.z + fxmul(ab.z,t));
        r.size=2; r.keep[0]=0; r.keep[1]=1; r.w[0]=HF_FPX_ONE - t; r.w[1]=t; r.dir=FxNeg(closest); return r;
    }
    int3 co = FxNeg(c);
    int d5 = FxDot(ab, co);
    int d6 = FxDot(ac, co);
    if (d6 >= 0 && d5 <= d6) { r.size=1; r.keep[0]=2; r.w[0]=HF_FPX_ONE; r.dir=co; return r; }
    int vb = fxmul(d5, d2) - fxmul(d1, d6);
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        int denom = d2 - d6;
        int t = (denom > HF_GJK_EDGE_EPS || denom < -HF_GJK_EDGE_EPS) ? fxdiv(d2, denom) : 0;
        int3 closest = int3(a.x + fxmul(ac.x,t), a.y + fxmul(ac.y,t), a.z + fxmul(ac.z,t));
        r.size=2; r.keep[0]=0; r.keep[1]=2; r.w[0]=HF_FPX_ONE - t; r.w[1]=t; r.dir=FxNeg(closest); return r;
    }
    int va = fxmul(d3, d6) - fxmul(d5, d4);
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        int denom = (d4 - d3) + (d5 - d6);
        int t = (denom > HF_GJK_EDGE_EPS || denom < -HF_GJK_EDGE_EPS) ? fxdiv(d4 - d3, denom) : 0;
        int3 bc = int3(c.x-b.x, c.y-b.y, c.z-b.z);
        int3 closest = int3(b.x + fxmul(bc.x,t), b.y + fxmul(bc.y,t), b.z + fxmul(bc.z,t));
        r.size=2; r.keep[0]=1; r.keep[1]=2; r.w[0]=HF_FPX_ONE - t; r.w[1]=t; r.dir=FxNeg(closest); return r;
    }
    int denom = va + vb + vc;
    int u, v, w;
    if (denom > HF_GJK_EDGE_EPS || denom < -HF_GJK_EDGE_EPS) {
        v = fxdiv(vb, denom);
        w = fxdiv(vc, denom);
        u = HF_FPX_ONE - v - w;
        r.w[0]=u; r.w[1]=v; r.w[2]=w;
    } else {
        r.w[0]=HF_FPX_ONE; r.w[1]=0; r.w[2]=0;
    }
    int3 closest = int3(fxmul(a.x,r.w[0]) + fxmul(b.x,r.w[1]) + fxmul(c.x,r.w[2]),
                        fxmul(a.y,r.w[0]) + fxmul(b.y,r.w[1]) + fxmul(c.y,r.w[2]),
                        fxmul(a.z,r.w[0]) + fxmul(b.z,r.w[1]) + fxmul(c.z,r.w[2]));
    r.size=3; r.keep[0]=0; r.keep[1]=1; r.keep[2]=2; r.dir=FxNeg(closest);
    return r;
}
Sub DoSimplex4(int3 a, int3 b, int3 c, int3 d) {
    int3 ao = FxNeg(a);
    int3 ab = int3(b.x-a.x, b.y-a.y, b.z-a.z);
    int3 ac = int3(c.x-a.x, c.y-a.y, c.z-a.z);
    int3 ad = int3(d.x-a.x, d.y-a.y, d.z-a.z);
    int3 nABC = FxCross(ab, ac);
    int3 nACD = FxCross(ac, ad);
    int3 nADB = FxCross(ad, ab);
    int sdABC = FxDot(nABC, ad);
    int sdACD = FxDot(nACD, ab);
    int sdADB = FxDot(nADB, ac);
    int soABC = FxDot(nABC, ao);
    int soACD = FxDot(nACD, ao);
    int soADB = FxDot(nADB, ao);
    bool oABC = (sdABC > 0) ? (soABC < 0) : ((sdABC < 0) ? (soABC > 0) : (soABC != 0));
    if (oABC) { return DoSimplex3(a, b, c); }
    bool oACD = (sdACD > 0) ? (soACD < 0) : ((sdACD < 0) ? (soACD > 0) : (soACD != 0));
    if (oACD) {
        Sub t = DoSimplex3(a, c, d);
        for (uint i = 0; i < t.size; ++i) { uint loc = t.keep[i]; t.keep[i] = (loc==0u)?0u:((loc==1u)?2u:3u); }
        return t;
    }
    bool oADB = (sdADB > 0) ? (soADB < 0) : ((sdADB < 0) ? (soADB > 0) : (soADB != 0));
    if (oADB) {
        Sub t = DoSimplex3(a, d, b);
        for (uint i = 0; i < t.size; ++i) { uint loc = t.keep[i]; t.keep[i] = (loc==0u)?0u:((loc==1u)?3u:1u); }
        return t;
    }
    Sub r; r.containsOrigin = true; r.dir = int3(0,0,0);
    r.size=4; r.keep[0]=0; r.keep[1]=1; r.keep[2]=2; r.keep[3]=3; r.w[0]=0;r.w[1]=0;r.w[2]=0;r.w[3]=0;
    return r;
}

// ===== GJK terminal-simplex result (the EPA seed) for HullContact =====
struct GjkOut {
    uint overlap;
    int3 pts[4]; int3 csoA[4]; int3 csoB[4];
    uint count;
};
GjkOut Gjk(FxHull hA, int4 oA, int3 pA, FxHull hB, int4 oB, int3 pB) {
    GjkOut res; res.overlap = 0u; res.count = 0u;
    [unroll] for (int z = 0; z < 4; ++z) { res.pts[z]=int3(0,0,0); res.csoA[z]=int3(0,0,0); res.csoB[z]=int3(0,0,0); }

    int3 dir = int3(pB.x - pA.x, pB.y - pA.y, pB.z - pA.z);
    if (dir.x == 0 && dir.y == 0 && dir.z == 0) dir = int3(HF_FPX_ONE, 0, 0);

    int3 sp[4], swA[4], swB[4];
    uint n = 0u;
    {
        int3 a = SupportMinkowski(hA, oA, pA, hB, oB, pB, dir);
        sp[0] = a;
        swA[0] = Support(hA, oA, pA, dir);
        swB[0] = Support(hB, oB, pB, FxNeg(dir));
        n = 1u;
    }
    int3 closest = sp[0];
    const int kProgressEps = HF_GJK_EDGE_EPS;

    bool overlap = false;
    uint iter = 0u;
    for (; iter < (uint)HF_GJK_MAX_ITER; ++iter) {
        if (FxLength(closest) <= kProgressEps) { overlap = true; break; }
        dir = FxNeg(closest);
        int3 sdir = FxNormalize(dir);
        int3 a  = SupportMinkowski(hA, oA, pA, hB, oB, pB, sdir);
        int3 wA = Support(hA, oA, pA, sdir);
        int3 wB = Support(hB, oB, pB, FxNeg(sdir));
        bool dup = false;
        for (uint i = 0u; i < n; ++i)
            if (a.x == sp[i].x && a.y == sp[i].y && a.z == sp[i].z) { dup = true; break; }
        if (dup) { overlap = false; break; }
        int advance = FxDot(a, sdir) - FxDot(closest, sdir);
        if (advance <= kProgressEps) { overlap = false; break; }
        for (uint i = n; i > 0u; --i) { sp[i] = sp[i-1u]; swA[i] = swA[i-1u]; swB[i] = swB[i-1u]; }
        sp[0] = a; swA[0] = wA; swB[0] = wB; ++n;
        Sub sub;
        if (n == 2u)      sub = DoSimplex2(sp[0], sp[1]);
        else if (n == 3u) sub = DoSimplex3(sp[0], sp[1], sp[2]);
        else              sub = DoSimplex4(sp[0], sp[1], sp[2], sp[3]);
        if (sub.containsOrigin) { overlap = true; break; }
        int3 nsp[4], nswA[4], nswB[4]; int nw[4] = {0,0,0,0};
        for (uint i = 0u; i < sub.size; ++i) {
            uint k = sub.keep[i];
            nsp[i] = sp[k]; nswA[i] = swA[k]; nswB[i] = swB[k]; nw[i] = sub.w[i];
        }
        int3 newClosest = int3(0, 0, 0);
        for (uint i = 0u; i < sub.size; ++i) {
            sp[i] = nsp[i]; swA[i] = nswA[i]; swB[i] = nswB[i];
            newClosest = int3(newClosest.x + fxmul(nsp[i].x, nw[i]),
                              newClosest.y + fxmul(nsp[i].y, nw[i]),
                              newClosest.z + fxmul(nsp[i].z, nw[i]));
        }
        n = sub.size;
        closest = newClosest;
    }
    res.overlap = overlap ? 1u : 0u;
    res.count = n;
    for (uint i = 0u; i < n && i < 4u; ++i) { res.pts[i] = sp[i]; res.csoA[i] = swA[i]; res.csoB[i] = swB[i]; }
    return res;
}

// ===== The trimmed GJK that returns just overlap + the closest CSO point (== ccd_toi.comp GjkSep, for the TOI
// loop). Byte-identical simplex evolution to Gjk; CCD reads only overlap + separation. =====
int3 GjkSep(FxHull hA, int4 oA, int3 pA, FxHull hB, int4 oB, int3 pB, out bool overlapOut) {
    int3 dir = int3(pB.x - pA.x, pB.y - pA.y, pB.z - pA.z);
    if (dir.x == 0 && dir.y == 0 && dir.z == 0) dir = int3(HF_FPX_ONE, 0, 0);
    int3 sp[4]; int3 swA[4]; int3 swB[4];
    uint n = 0u;
    int  wts[4] = {0, 0, 0, 0};
    uint keepCount = 1u;
    {
        int3 a = SupportMinkowski(hA, oA, pA, hB, oB, pB, dir);
        sp[0] = a;
        swA[0] = Support(hA, oA, pA, dir);
        swB[0] = Support(hB, oB, pB, FxNeg(dir));
        n = 1u;
        wts[0] = HF_FPX_ONE;
    }
    int3 closest = sp[0];
    const int kProgressEps = HF_GJK_EDGE_EPS;
    bool overlap = false;
    uint iter = 0u;
    for (; iter < (uint)HF_GJK_MAX_ITER; ++iter) {
        if (FxLength(closest) <= kProgressEps) { overlap = true; break; }
        dir = FxNeg(closest);
        int3 sdir = FxNormalize(dir);
        int3 a  = SupportMinkowski(hA, oA, pA, hB, oB, pB, sdir);
        int3 wA = Support(hA, oA, pA, sdir);
        int3 wB = Support(hB, oB, pB, FxNeg(sdir));
        bool dup = false;
        for (uint i = 0u; i < n; ++i)
            if (a.x == sp[i].x && a.y == sp[i].y && a.z == sp[i].z) { dup = true; break; }
        if (dup) { overlap = false; break; }
        int advance = FxDot(a, sdir) - FxDot(closest, sdir);
        if (advance <= kProgressEps) { overlap = false; break; }
        for (uint i2 = n; i2 > 0u; --i2) { sp[i2] = sp[i2-1u]; swA[i2] = swA[i2-1u]; swB[i2] = swB[i2-1u]; }
        sp[0] = a; swA[0] = wA; swB[0] = wB; ++n;
        Sub sub;
        if (n == 2u)      sub = DoSimplex2(sp[0], sp[1]);
        else if (n == 3u) sub = DoSimplex3(sp[0], sp[1], sp[2]);
        else              sub = DoSimplex4(sp[0], sp[1], sp[2], sp[3]);
        if (sub.containsOrigin) {
            overlap = true; keepCount = sub.size;
            for (uint i3 = 0u; i3 < 4u; ++i3) wts[i3] = sub.w[i3];
            break;
        }
        int3 nsp[4]; int3 nswA[4]; int3 nswB[4]; int nw[4] = {0,0,0,0};
        for (uint i4 = 0u; i4 < sub.size; ++i4) {
            uint k = sub.keep[i4];
            nsp[i4] = sp[k]; nswA[i4] = swA[k]; nswB[i4] = swB[k]; nw[i4] = sub.w[i4];
        }
        int3 newClosest = int3(0, 0, 0);
        for (uint i5 = 0u; i5 < sub.size; ++i5) {
            sp[i5] = nsp[i5]; swA[i5] = nswA[i5]; swB[i5] = nswB[i5]; wts[i5] = nw[i5];
            newClosest = int3(newClosest.x + fxmul(nsp[i5].x, nw[i5]),
                              newClosest.y + fxmul(nsp[i5].y, nw[i5]),
                              newClosest.z + fxmul(nsp[i5].z, nw[i5]));
        }
        n = sub.size;
        keepCount = sub.size;
        closest = newClosest;
    }
    if (!overlap) {
        if (keepCount == 0u) { keepCount = (n > 0u ? 1u : 0u); if (keepCount == 1u) wts[0] = HF_FPX_ONE; }
    }
    uint kc = (keepCount == 0u ? n : keepCount);
    if (kc > 4u) kc = 4u;
    int wsum = 0; for (uint i = 0u; i < kc; ++i) wsum += wts[i];
    if (wsum == 0 && kc > 0u) { wts[0] = HF_FPX_ONE; }
    int3 closestCso = int3(0, 0, 0);
    for (uint i6 = 0u; i6 < kc; ++i6) {
        closestCso = int3(closestCso.x + fxmul(sp[i6].x, wts[i6]),
                          closestCso.y + fxmul(sp[i6].y, wts[i6]),
                          closestCso.z + fxmul(sp[i6].z, wts[i6]));
    }
    overlapOut = overlap;
    return closestCso;
}

// ===== The EPA polytope state (fixed-size, thread-local — single-thread dispatch) =====
static int3 gVerts[HF_EPA_MAX_PV];
static int3 gVertsA[HF_EPA_MAX_PV];
static int3 gVertsB[HF_EPA_MAX_PV];
static uint gFaceA[HF_EPA_MAX_PF];
static uint gFaceB[HF_EPA_MAX_PF];
static uint gFaceC[HF_EPA_MAX_PF];
static int3 gFaceN[HF_EPA_MAX_PF];
static int  gFaceD[HF_EPA_MAX_PF];
static int3 gInterior;
static uint gVertCount;
static uint gFaceCount;

bool EpaAddFace(uint a, uint b, uint c) {
    if (gFaceCount >= (uint)HF_EPA_MAX_PF) return false;
    int3 ab = FxSub(gVerts[b], gVerts[a]);
    int3 ac = FxSub(gVerts[c], gVerts[a]);
    int3 raw = FxCross(ab, ac);
    if (FxLength(raw) < HF_GJK_EDGE_EPS) return false;
    int3 n = FxNormalize(raw);
    uint fa, fb, fc;
    if (FxDot(n, FxSub(gVerts[a], gInterior)) < 0) { fa = a; fb = c; fc = b; n = FxNeg(n); }
    else                                           { fa = a; fb = b; fc = c; }
    int d = FxDot(n, gVerts[a]);
    gFaceA[gFaceCount] = fa; gFaceB[gFaceCount] = fb; gFaceC[gFaceCount] = fc;
    gFaceN[gFaceCount] = n;  gFaceD[gFaceCount] = d;
    ++gFaceCount;
    return true;
}

struct Manifold { uint count; int3 pt; int depth; int3 normal; };
struct EpaOut { int depth; int3 normal; int3 cA; int3 cB; };
EpaOut RunEpa(FxHull hA, int4 oA, int3 pA, FxHull hB, int4 oB, int3 pB, GjkOut g) {
    EpaOut res; res.depth = 0; res.normal = int3(0, HF_FPX_ONE, 0); res.cA = int3(0,0,0); res.cB = int3(0,0,0);
    [unroll] for (uint zi = 0u; zi < (uint)HF_EPA_MAX_PV; ++zi) {
        gVerts[zi] = int3(0,0,0); gVertsA[zi] = int3(0,0,0); gVertsB[zi] = int3(0,0,0);
    }
    [unroll] for (uint zf = 0u; zf < (uint)HF_EPA_MAX_PF; ++zf) {
        gFaceA[zf] = 0u; gFaceB[zf] = 0u; gFaceC[zf] = 0u; gFaceN[zf] = int3(0,0,0); gFaceD[zf] = 0;
    }
    gInterior = int3(0,0,0);
    gVertCount = 0u;
    gFaceCount = 0u;
    uint sc = g.count; if (sc > 4u) sc = 4u;
    for (uint i = 0u; i < sc; ++i) { gVerts[i] = g.pts[i]; gVertsA[i] = g.csoA[i]; gVertsB[i] = g.csoB[i]; }
    gVertCount = sc;

    int3 kProbe[10];
    kProbe[0] = int3(HF_FPX_ONE, 0, 0); kProbe[1] = int3(0, HF_FPX_ONE, 0); kProbe[2] = int3(0, 0, HF_FPX_ONE);
    kProbe[3] = int3(-HF_FPX_ONE, 0, 0); kProbe[4] = int3(0, -HF_FPX_ONE, 0); kProbe[5] = int3(0, 0, -HF_FPX_ONE);
    kProbe[6] = int3(HF_FPX_ONE, HF_FPX_ONE, HF_FPX_ONE);
    kProbe[7] = int3(-HF_FPX_ONE, HF_FPX_ONE, HF_FPX_ONE);
    kProbe[8] = int3(HF_FPX_ONE, -HF_FPX_ONE, HF_FPX_ONE);
    kProbe[9] = int3(HF_FPX_ONE, HF_FPX_ONE, -HF_FPX_ONE);

    if (gVertCount < 2u) {
        for (uint p = 0u; p < 10u && gVertCount < 2u; ++p) {
            int3 dir = FxNormalize(kProbe[p]);
            int3 wA = Support(hA, oA, pA, dir);
            int3 wB = Support(hB, oB, pB, FxNeg(dir));
            int3 sp = FxSub(wA, wB);
            bool dup = false;
            for (uint v = 0u; v < gVertCount; ++v)
                if (sp.x == gVerts[v].x && sp.y == gVerts[v].y && sp.z == gVerts[v].z) { dup = true; break; }
            if (dup) continue;
            gVerts[gVertCount] = sp; gVertsA[gVertCount] = wA; gVertsB[gVertCount] = wB; ++gVertCount;
        }
    }
    if (gVertCount == 2u) {
        int3 e = FxSub(gVerts[1], gVerts[0]);
        int eLen = FxLength(e);
        bool added = false;
        for (uint p = 0u; p < 10u && !added; ++p) {
            int3 dir = kProbe[p];
            if (eLen >= HF_GJK_EDGE_EPS) {
                int proj = fxdiv(FxDot(dir, e), eLen);
                int3 eUnit = FxNormalize(e);
                dir = FxSub(dir, FxScale(eUnit, proj));
            }
            if (FxLength(dir) < HF_GJK_EDGE_EPS) continue;
            dir = FxNormalize(dir);
            int3 wA = Support(hA, oA, pA, dir);
            int3 wB = Support(hB, oB, pB, FxNeg(dir));
            int3 sp = FxSub(wA, wB);
            bool dup = false;
            for (uint v = 0u; v < gVertCount; ++v)
                if (sp.x == gVerts[v].x && sp.y == gVerts[v].y && sp.z == gVerts[v].z) { dup = true; break; }
            if (dup) continue;
            int3 nrm = FxCross(FxSub(gVerts[1], gVerts[0]), FxSub(sp, gVerts[0]));
            if (FxLength(nrm) < HF_GJK_EDGE_EPS) continue;
            gVerts[2] = sp; gVertsA[2] = wA; gVertsB[2] = wB; gVertCount = 3u;
            added = true;
        }
    }
    if (gVertCount == 3u) {
        int3 ab = FxSub(gVerts[1], gVerts[0]);
        int3 ac = FxSub(gVerts[2], gVerts[0]);
        int3 n = FxCross(ab, ac);
        int3 nUnit = (FxLength(n) < HF_GJK_EDGE_EPS) ? int3(0, HF_FPX_ONE, 0) : FxNormalize(n);
        int side = FxDot(nUnit, FxNeg(gVerts[0]));
        int3 cands[12];
        cands[0] = (side >= 0) ? nUnit : FxNeg(nUnit);
        cands[1] = FxNeg(cands[0]);
        for (uint p = 0u; p < 10u; ++p) cands[2u + p] = FxNormalize(kProbe[p]);
        for (uint c = 0u; c < 12u && gVertCount == 3u; ++c) {
            int3 dir = cands[c];
            int3 wA = Support(hA, oA, pA, dir);
            int3 wB = Support(hB, oB, pB, FxNeg(dir));
            int3 sp = FxSub(wA, wB);
            bool dup = false;
            for (uint v = 0u; v < gVertCount; ++v)
                if (sp.x == gVerts[v].x && sp.y == gVerts[v].y && sp.z == gVerts[v].z) { dup = true; break; }
            if (dup) continue;
            int vol = FxDot(n, FxSub(sp, gVerts[0]));
            int avol = (vol < 0) ? -vol : vol;
            if (avol < HF_GJK_EDGE_EPS) continue;
            gVerts[3] = sp; gVertsA[3] = wA; gVertsB[3] = wB; gVertCount = 4u;
        }
    }

    if (gVertCount < 4u) {
        res.depth = 0; res.normal = int3(0, HF_FPX_ONE, 0);
        res.cA = (gVertCount > 0u) ? gVertsA[0] : int3(0,0,0);
        res.cB = (gVertCount > 0u) ? gVertsB[0] : int3(0,0,0);
        return res;
    }

    gInterior = FxScale(FxAdd(FxAdd(gVerts[0], gVerts[1]), FxAdd(gVerts[2], gVerts[3])), HF_FPX_ONE / 4);

    gFaceCount = 0u;
    EpaAddFace(0u, 1u, 2u);
    EpaAddFace(0u, 1u, 3u);
    EpaAddFace(0u, 2u, 3u);
    EpaAddFace(1u, 2u, 3u);

    if (gFaceCount == 0u) {
        res.depth = 0; res.normal = int3(0, HF_FPX_ONE, 0);
        res.cA = gVertsA[0]; res.cB = gVertsB[0];
        return res;
    }

    uint closestFace = 0u;
    uint iter = 0u;
    for (; iter < (uint)HF_EPA_MAX_ITER; ++iter) {
        closestFace = 0u;
        int minDist = gFaceD[0];
        for (uint f = 1u; f < gFaceCount; ++f) {
            if (gFaceD[f] < minDist) { minDist = gFaceD[f]; closestFace = f; }
        }
        int3 cfn = gFaceN[closestFace];
        int  cfd = gFaceD[closestFace];
        int3 dir = cfn;
        int3 wA = Support(hA, oA, pA, dir);
        int3 wB = Support(hB, oB, pB, FxNeg(dir));
        int3 sp = FxSub(wA, wB);
        int proj = FxDot(sp, dir);
        if (proj <= cfd + HF_EPA_TOL) break;
        bool dup = false;
        for (uint v = 0u; v < gVertCount; ++v)
            if (sp.x == gVerts[v].x && sp.y == gVerts[v].y && sp.z == gVerts[v].z) { dup = true; break; }
        if (dup) break;
        if (gVertCount >= (uint)HF_EPA_MAX_PV) break;
        uint nv = gVertCount;
        gVerts[nv] = sp; gVertsA[nv] = wA; gVertsB[nv] = wB; ++gVertCount;

        uint horizU[HF_EPA_MAX_PF * 3];
        uint horizV[HF_EPA_MAX_PF * 3];
        uint horizCount = 0u;
        bool removed[HF_EPA_MAX_PF];
        for (uint f = 0u; f < gFaceCount; ++f) removed[f] = false;
        for (uint f = 0u; f < gFaceCount; ++f) {
            int3 toNew = FxSub(sp, gVerts[gFaceA[f]]);
            if (FxDot(gFaceN[f], toNew) > 0) {
                removed[f] = true;
                uint ea0 = gFaceA[f], eb0 = gFaceB[f];
                uint ea1 = gFaceB[f], eb1 = gFaceC[f];
                uint ea2 = gFaceC[f], eb2 = gFaceA[f];
                {
                    bool cancelled = false;
                    for (uint e = 0u; e < horizCount; ++e) {
                        if (horizU[e] == eb0 && horizV[e] == ea0) {
                            horizU[e] = horizU[horizCount-1u]; horizV[e] = horizV[horizCount-1u];
                            --horizCount; cancelled = true; break;
                        }
                    }
                    if (!cancelled && horizCount < (uint)(HF_EPA_MAX_PF*3)) {
                        horizU[horizCount] = ea0; horizV[horizCount] = eb0; ++horizCount;
                    }
                }
                {
                    bool cancelled = false;
                    for (uint e = 0u; e < horizCount; ++e) {
                        if (horizU[e] == eb1 && horizV[e] == ea1) {
                            horizU[e] = horizU[horizCount-1u]; horizV[e] = horizV[horizCount-1u];
                            --horizCount; cancelled = true; break;
                        }
                    }
                    if (!cancelled && horizCount < (uint)(HF_EPA_MAX_PF*3)) {
                        horizU[horizCount] = ea1; horizV[horizCount] = eb1; ++horizCount;
                    }
                }
                {
                    bool cancelled = false;
                    for (uint e = 0u; e < horizCount; ++e) {
                        if (horizU[e] == eb2 && horizV[e] == ea2) {
                            horizU[e] = horizU[horizCount-1u]; horizV[e] = horizV[horizCount-1u];
                            --horizCount; cancelled = true; break;
                        }
                    }
                    if (!cancelled && horizCount < (uint)(HF_EPA_MAX_PF*3)) {
                        horizU[horizCount] = ea2; horizV[horizCount] = eb2; ++horizCount;
                    }
                }
            }
        }
        uint w = 0u;
        for (uint f = 0u; f < gFaceCount; ++f) {
            if (!removed[f]) {
                if (w != f) {
                    gFaceA[w] = gFaceA[f]; gFaceB[w] = gFaceB[f]; gFaceC[w] = gFaceC[f];
                    gFaceN[w] = gFaceN[f]; gFaceD[w] = gFaceD[f];
                }
                ++w;
            }
        }
        gFaceCount = w;
        for (uint e = 0u; e < horizCount; ++e) EpaAddFace(horizU[e], horizV[e], nv);
        if (gFaceCount == 0u) {
            EpaAddFace(0u, 1u, nv);
            if (gFaceCount == 0u) { ++iter; break; }
        }
    }

    closestFace = 0u;
    {
        int minDist = gFaceD[0];
        for (uint f = 1u; f < gFaceCount; ++f) {
            if (gFaceD[f] < minDist) { minDist = gFaceD[f]; closestFace = f; }
        }
    }
    uint cfa = gFaceA[closestFace], cfb = gFaceB[closestFace], cfc = gFaceC[closestFace];
    int3 cfn = gFaceN[closestFace];
    int  cfd = gFaceD[closestFace];
    int3 pAv = gVerts[cfa];
    int3 pBv = gVerts[cfb];
    int3 pCv = gVerts[cfc];
    int3 foot = FxScale(cfn, cfd);
    int3 v0 = FxSub(pBv, pAv);
    int3 v1 = FxSub(pCv, pAv);
    int3 v2 = FxSub(foot, pAv);
    int d00 = FxDot(v0, v0);
    int d01 = FxDot(v0, v1);
    int d11 = FxDot(v1, v1);
    int d20 = FxDot(v2, v0);
    int d21 = FxDot(v2, v1);
    int denom = fxmul(d00, d11) - fxmul(d01, d01);
    int bu, bv, bw;
    if (denom > HF_GJK_EDGE_EPS || denom < -HF_GJK_EDGE_EPS) {
        bv = fxdiv(fxmul(d11, d20) - fxmul(d01, d21), denom);
        bw = fxdiv(fxmul(d00, d21) - fxmul(d01, d20), denom);
        bu = HF_FPX_ONE - bv - bw;
    } else { bu = HF_FPX_ONE; bv = 0; bw = 0; }
    int3 cA = FxAdd(FxAdd(FxScale(gVertsA[cfa], bu), FxScale(gVertsA[cfb], bv)), FxScale(gVertsA[cfc], bw));
    int3 cB = FxAdd(FxAdd(FxScale(gVertsB[cfa], bu), FxScale(gVertsB[cfb], bv)), FxScale(gVertsB[cfc], bw));
    res.depth = cfd;
    res.normal = cfn;
    res.cA = cA;
    res.cB = cB;
    return res;
}

Manifold HullContact(FxHull hA, int4 oA, int3 pA, FxHull hB, int4 oB, int3 pB) {
    Manifold m; m.count = 0u; m.pt = int3(0,0,0); m.depth = 0; m.normal = int3(0,0,0);
    GjkOut g = Gjk(hA, oA, pA, hB, oB, pB);
    if (g.overlap == 0u) return m;
    EpaOut e = RunEpa(hA, oA, pA, hB, oB, pB, g);
    m.count = 1u;
    m.normal = e.normal;
    m.pt = int3((e.cA.x + e.cB.x) / 2, (e.cA.y + e.cB.y) / 2, (e.cA.z + e.cB.z) / 2);
    m.depth = e.depth;
    return m;
}

void SolveManifoldImpulse(inout FxBody bA, inout FxBody bB, int invIaW[9], int invIbW[9], Manifold m,
                          int restitution) {
    if (m.count == 0u) return;
    int3 nrm = m.normal;
    int3 ab = int3(bB.px - bA.px, bB.py - bA.py, bB.pz - bA.pz);
    if (FxDot(nrm, ab) < 0) nrm = int3(-nrm.x, -nrm.y, -nrm.z);
    int invMassA = bA.invMass;
    int invMassB = bB.invMass;
    int3 p = m.pt;
    int3 posA2 = int3(bA.px, bA.py, bA.pz);
    int3 posB2 = int3(bB.px, bB.py, bB.pz);
    int3 rA = int3(p.x - posA2.x, p.y - posA2.y, p.z - posA2.z);
    int3 rB = int3(p.x - posB2.x, p.y - posB2.y, p.z - posB2.z);
    int3 wA = int3(bA.ax, bA.ay, bA.az);
    int3 wB = int3(bB.ax, bB.ay, bB.az);
    int3 velA = int3(bA.vx, bA.vy, bA.vz);
    int3 velB = int3(bB.vx, bB.vy, bB.vz);
    int3 vpA = velA + FxCross(wA, rA);
    int3 vpB = velB + FxCross(wB, rB);
    int3 dvp = int3(vpB.x - vpA.x, vpB.y - vpA.y, vpB.z - vpA.z);
    int vn = FxDot(dvp, nrm);
    if (vn >= 0) return;
    int3 raxn = FxCross(rA, nrm);
    int3 rbxn = FxCross(rB, nrm);
    int angA = FxDot(nrm, FxCross(Mat3MulVec(invIaW, raxn), rA));
    int angB = FxDot(nrm, FxCross(Mat3MulVec(invIbW, rbxn), rB));
    int k = invMassA + invMassB + angA + angB;
    if (k <= 0) return;
    int jn = fxdiv(-fxmul(HF_FPX_ONE + restitution, vn), k);
    if (jn < 0) jn = 0;
    int3 J = int3(fxmul(nrm.x, jn), fxmul(nrm.y, jn), fxmul(nrm.z, jn));
    int3 dvA = int3(fxmul(J.x, invMassA), fxmul(J.y, invMassA), fxmul(J.z, invMassA));
    int3 dwA = Mat3MulVec(invIaW, FxCross(rA, J));
    bA.vx -= dvA.x; bA.vy -= dvA.y; bA.vz -= dvA.z;
    bA.ax -= dwA.x; bA.ay -= dwA.y; bA.az -= dwA.z;
    int3 dvB = int3(fxmul(J.x, invMassB), fxmul(J.y, invMassB), fxmul(J.z, invMassB));
    int3 dwB = Mat3MulVec(invIbW, FxCross(rB, J));
    bB.vx += dvB.x; bB.vy += dvB.y; bB.vz += dvB.z;
    bB.ax += dwB.x; bB.ay += dwB.y; bB.az += dwB.z;
}

// BuildHullAabb (== broad.h::BuildHullAabb): the hull's tight world AABB.
void BuildHullAabb(FxHull h, int4 orient, int3 pos, out int3 outLo, out int3 outHi) {
    int3 sPX = Support(h, orient, pos, int3( HF_FPX_ONE, 0, 0));
    int3 sNX = Support(h, orient, pos, int3(-HF_FPX_ONE, 0, 0));
    int3 sPY = Support(h, orient, pos, int3(0,  HF_FPX_ONE, 0));
    int3 sNY = Support(h, orient, pos, int3(0, -HF_FPX_ONE, 0));
    int3 sPZ = Support(h, orient, pos, int3(0, 0,  HF_FPX_ONE));
    int3 sNZ = Support(h, orient, pos, int3(0, 0, -HF_FPX_ONE));
    outLo = int3(sNX.x, sNY.y, sNZ.z);
    outHi = int3(sPX.x, sPY.y, sPZ.z);
}

// BodyMaxRadius (VERBATIM ccd.h::BodyMaxRadius) — the body bounding radius for ClosingSpeedBound.
int BodyMaxRadius(FxHull hull, int4 orient, int3 pos) {
    int best = 0;
    for (uint i = 0u; i < hull.count; ++i) {
        int3 lv = int3(hull.vx[i], hull.vy[i], hull.vz[i]);
        int3 worldV = FxRotate(orient, lv);
        worldV = int3(worldV.x + pos.x, worldV.y + pos.y, worldV.z + pos.z);
        int3 rel = int3(worldV.x - pos.x, worldV.y - pos.y, worldV.z - pos.z);
        int r = FxLength(rel);
        if (r > best) best = r;
    }
    return best;
}
int ClosingSpeedBound(int3 velA, int3 angVelA, int rMaxA,
                      int3 velB, int3 angVelB, int rMaxB, int3 n) {
    int3 relVel = int3(velB.x - velA.x, velB.y - velA.y, velB.z - velA.z);
    int linClose = FxDot(relVel, n);
    int angA = fxmul(FxLength(angVelA), rMaxA);
    int angB = fxmul(FxLength(angVelB), rMaxB);
    return linClose + angA + angB;
}

// ConservativeAdvance (VERBATIM ccd.h::ConservativeAdvance over body i,j at their CURRENT pose) -> {toi, hit}.
// Returns toi in toiOut; hit in the return value. dt = remainingDt.
bool ConservativeAdvance(FxHull hA, FxBody bA, FxHull hB, FxBody bB, int dt, out int toiOut) {
    int3 posA = int3(bA.px, bA.py, bA.pz);
    int3 velA = int3(bA.vx, bA.vy, bA.vz);
    int4 oriA = int4(bA.ox, bA.oy, bA.oz, bA.ow);
    int3 angA = int3(bA.ax, bA.ay, bA.az);
    uint flagsA = bA.flags;
    int3 posB = int3(bB.px, bB.py, bB.pz);
    int3 velB = int3(bB.vx, bB.vy, bB.vz);
    int4 oriB = int4(bB.ox, bB.oy, bB.oz, bB.ow);
    int3 angB = int3(bB.ax, bB.ay, bB.az);
    uint flagsB = bB.flags;

    int rMaxA = BodyMaxRadius(hA, oriA, posA);
    int rMaxB = BodyMaxRadius(hB, oriB, posB);

    int t = 0;
    bool hit = false;
    bool done = false;
    for (uint iter = 0u; iter < (uint)HF_CCD_TOI_MAX_ITER && !done; ++iter) {
        bool overlap;
        int3 sep = GjkSep(hA, oriA, posA, hB, oriB, posB, overlap);
        if (overlap) { t = t; hit = true; done = true; break; }
        int gap = FxLength(sep);
        if (gap <= HF_CCD_CONTACT_EPS) { hit = true; done = true; break; }
        int3 nn = FxNormalize(sep);
        int bound = ClosingSpeedBound(velA, angA, rMaxA, velB, angB, rMaxB, nn);
        if (bound <= 0) { t = dt; hit = false; done = true; break; }
        int adv = fxdiv(gap, bound);
        t += adv;
        if (t >= dt) { t = dt; hit = false; done = true; break; }
        IntegratePoseZeroG(posA, velA, oriA, angA, flagsA, adv);
        IntegratePoseZeroG(posB, velB, oriB, angB, flagsB, adv);
    }
    if (!done) { hit = false; }
    toiOut = t;
    return hit;
}

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // single thread runs the WHOLE world step

    int bodyCount   = gParams[0].c0.x;
    int ticks       = gParams[0].c0.y;
    int stepEnabled = gParams[0].c0.z;
    int solveIters  = gParams[0].c0.w;
    int posIters    = gParams[0].c1.x;
    int restitution = gParams[0].c1.y;
    int slop        = gParams[0].c1.z;
    int beta        = gParams[0].c1.w;
    int linDamp     = gParams[0].c2.x;
    int angDamp     = gParams[0].c2.y;
    int dt          = gParams[0].c2.z;
    int maxSubsteps = gParams[0].c2.w;
    int3 gravity    = int3(gParams[0].c3.x, gParams[0].c3.y, gParams[0].c3.z);

    int n = bodyCount;
    if (n > HF_MAX_BODIES) n = HF_MAX_BODIES;

    FxBody body[HF_MAX_BODIES];
    FxHull hull[HF_MAX_BODIES];
    for (int li = 0; li < n; ++li) { body[li] = gBodies[li]; hull[li] = gHulls[li]; }

    if (stepEnabled == 0) {
        for (int wi0 = 0; wi0 < n; ++wi0) gBodies[wi0] = body[wi0];
        return;
    }

    // ===== StepHullWorldCCDN: `ticks` x StepHullWorldCCD (== ccd.h::StepHullWorldCCD VERBATIM) =====
    for (int tk = 0; tk < ticks; ++tk) {
        int remainingDt = dt;
        for (int sub = 0; sub < maxSubsteps && remainingDt > 0; ++sub) {
            // (1) swept broadphase: snapshot each body's swept AABB (start pose UNION end pose advanced by
            // remainingDt with ZERO gravity — == ccd.h::SweptHullAabb). The all-pairs gate over these AABBs
            // reproduces the CPU's grid candidate SET (a non-swept-overlapping pair is a ConservativeAdvance
            // no-op, so the gate is bit-transparent for the min-TOI + resolve).
            int swLo[HF_MAX_BODIES * 3];
            int swHi[HF_MAX_BODIES * 3];
            for (int ai = 0; ai < n; ++ai) {
                int4 oa = int4(body[ai].ox, body[ai].oy, body[ai].oz, body[ai].ow);
                int3 pa = int3(body[ai].px, body[ai].py, body[ai].pz);
                int3 lo0, hi0;
                BuildHullAabb(hull[ai], oa, pa, lo0, hi0);
                int3 pe = pa; int4 oe = oa;
                int3 ve = int3(body[ai].vx, body[ai].vy, body[ai].vz);
                int3 ae = int3(body[ai].ax, body[ai].ay, body[ai].az);
                IntegratePoseZeroG(pe, ve, oe, ae, body[ai].flags, remainingDt);
                int3 lo1, hi1;
                BuildHullAabb(hull[ai], oe, pe, lo1, hi1);
                swLo[ai*3+0] = min(lo0.x, lo1.x); swLo[ai*3+1] = min(lo0.y, lo1.y); swLo[ai*3+2] = min(lo0.z, lo1.z);
                swHi[ai*3+0] = max(hi0.x, hi1.x); swHi[ai*3+1] = max(hi0.y, hi1.y); swHi[ai*3+2] = max(hi0.z, hi1.z);
            }

            // (2) per-pair TOI over the swept-gated candidates -> the GLOBAL-MIN TOI (strict-less keeps the
            // lowest-(i,j) tie). Record pairHit[i*n+j] from THIS pre-advance scan (the CPU stores pairHit BEFORE
            // advancing, then resolves those same pairs at the advanced pose).
            bool pairHit[HF_MAX_BODIES * HF_MAX_BODIES];
            for (int z0 = 0; z0 < n * n; ++z0) pairHit[z0] = false;
            int minToi = remainingDt;
            bool anyHit = false;
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    bool ov = swLo[i*3+0] <= swHi[j*3+0] && swLo[j*3+0] <= swHi[i*3+0] &&
                              swLo[i*3+1] <= swHi[j*3+1] && swLo[j*3+1] <= swHi[i*3+1] &&
                              swLo[i*3+2] <= swHi[j*3+2] && swLo[j*3+2] <= swHi[i*3+2];
                    if (!ov) continue;
                    int toi;
                    bool hit = ConservativeAdvance(hull[i], body[i], hull[j], body[j], remainingDt, toi);
                    if (hit) {
                        pairHit[i*n+j] = true;
                        if (!anyHit || toi < minToi) minToi = toi;
                        anyHit = true;
                    }
                }
            }

            // (3) advance ALL dynamic bodies by minToi (a hit) or the full remainingDt (no impact this tick) +
            // per-substep velocity retention (== StepHullWorldBP pass 1, the tick's gravity over the substep).
            int advStep = anyHit ? minToi : remainingDt;
            for (int i1 = 0; i1 < n; ++i1) {
                if ((body[i1].invMass != 0) && (body[i1].flags & HF_FLAG_DYNAMIC)) {
                    IntegrateBodyFull(body[i1], gravity, advStep);
                    if (linDamp != HF_FPX_ONE) {
                        body[i1].vx = fxmul(body[i1].vx, linDamp);
                        body[i1].vy = fxmul(body[i1].vy, linDamp);
                        body[i1].vz = fxmul(body[i1].vz, linDamp);
                    }
                    if (angDamp != HF_FPX_ONE) {
                        body[i1].ax = fxmul(body[i1].ax, angDamp);
                        body[i1].ay = fxmul(body[i1].ay, angDamp);
                        body[i1].az = fxmul(body[i1].az, angDamp);
                    }
                }
            }

            if (!anyHit) { remainingDt = 0; break; }   // no impact this tick -> the full step was consumed.

            // (4) resolve at minToi: per-body world inverse inertias once (post-advance orient), then the frozen
            // solve over every TOUCHING (hit) candidate pair, in (i,j)-ascending order (== ResolveTouchingPairCCD
            // = the StepHullWorldBP solve + de-pen passes over the single pair).
            int invIW[HF_MAX_BODIES][9];
            for (int i2 = 0; i2 < n; ++i2) {
                int4 q = int4(body[i2].ox, body[i2].oy, body[i2].oz, body[i2].ow);
                int3 invIb = HullInvInertiaBody(hull[i2], body[i2].invMass);
                int M[9];
                WorldInvInertia(q, invIb, M);
                [unroll] for (int z = 0; z < 9; ++z) invIW[i2][z] = M[z];
            }
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    if (!pairHit[i*n+j]) continue;
                    int iaW[9]; int ibW[9];
                    [unroll] for (int z = 0; z < 9; ++z) { iaW[z] = invIW[i][z]; ibW[z] = invIW[j][z]; }
                    // (4a) impulse solve — solveIters Gauss-Seidel sweeps over THIS pair.
                    for (int sweep = 0; sweep < solveIters; ++sweep) {
                        if (body[i].invMass == 0 && body[j].invMass == 0) break;
                        int4 oi = int4(body[i].ox, body[i].oy, body[i].oz, body[i].ow);
                        int3 pi = int3(body[i].px, body[i].py, body[i].pz);
                        int4 oj = int4(body[j].ox, body[j].oy, body[j].oz, body[j].ow);
                        int3 pj = int3(body[j].px, body[j].py, body[j].pz);
                        Manifold m = HullContact(hull[i], oi, pi, hull[j], oj, pj);
                        if (m.count == 0u) continue;
                        FxBody bi = body[i]; FxBody bj = body[j];
                        SolveManifoldImpulse(bi, bj, iaW, ibW, m, restitution);
                        body[i] = bi; body[j] = bj;
                    }
                    // (4b) position de-penetration — posIters sweeps over THIS pair.
                    for (int pit = 0; pit < posIters; ++pit) {
                        int invSum = body[i].invMass + body[j].invMass;
                        if (invSum == 0) break;
                        int4 oi = int4(body[i].ox, body[i].oy, body[i].oz, body[i].ow);
                        int3 pi = int3(body[i].px, body[i].py, body[i].pz);
                        int4 oj = int4(body[j].ox, body[j].oy, body[j].oz, body[j].ow);
                        int3 pj = int3(body[j].px, body[j].py, body[j].pz);
                        Manifold m = HullContact(hull[i], oi, pi, hull[j], oj, pj);
                        if (m.count == 0u) continue;
                        int3 nrm = m.normal;
                        int3 ab = int3(body[j].px - body[i].px, body[j].py - body[i].py, body[j].pz - body[i].pz);
                        if (FxDot(nrm, ab) < 0) nrm = int3(-nrm.x, -nrm.y, -nrm.z);
                        int excess = m.depth - slop;
                        if (excess <= 0) continue;
                        int corrected = fxmul(excess, beta);
                        int wi = fxdiv(body[i].invMass, invSum);
                        int wj = HF_FPX_ONE - wi;
                        int3 ci = int3(fxmul(nrm.x, fxmul(corrected, wi)), fxmul(nrm.y, fxmul(corrected, wi)),
                                       fxmul(nrm.z, fxmul(corrected, wi)));
                        int3 cj = int3(fxmul(nrm.x, fxmul(corrected, wj)), fxmul(nrm.y, fxmul(corrected, wj)),
                                       fxmul(nrm.z, fxmul(corrected, wj)));
                        body[i].px -= ci.x; body[i].py -= ci.y; body[i].pz -= ci.z;
                        body[j].px += cj.x; body[j].py += cj.y; body[j].pz += cj.z;
                    }
                }
            }

            // (5) consume the substep + loop.
            remainingDt -= minToi;
        }
    }

    for (int wi = 0; wi < n; ++wi) gBodies[wi] = body[wi];
}
