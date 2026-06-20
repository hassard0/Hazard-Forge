// Slice FC4 — Deterministic Contact Friction THE FRICTION-LOCKED WORLD STEP compute pass (the 4th slice of
// FLAGSHIP #20: DETERMINISTIC TANGENTIAL CONTACT FRICTION, hf::sim::fric — THE MONEY-PHYSICS BEAT). ONE
// THREAD runs the WHOLE StepFrictionWorldN over the SMALL body set (all-pairs over the few boxes) —
// predict-integrate -> all-pairs narrowphase -> world Gauss-Seidel impulse (the FC3 normal + cone-clamped
// TANGENT friction solve, swapped in for CX4's normal-only SolveManifoldImpulse) -> POSITION de-penetration,
// K ticks — copying engine/sim/fric.h::StepFrictionWorld / StepFrictionWorldN VERBATIM (the SAME fixed body
// order, the SAME fixed i<j pair order, the SAME world-level Gauss-Seidel sweeps, the SAME face-preferred
// BoxSatStable, the SAME A->B basis, the SAME cone-clamped friction impulse, the SAME split-by-invMass linear
// de-pen with slop+beta, the SAME per-tick damping, the SAME int64 ops). The final gBodies array is
// byte-identical to the CPU reference -> the host GPU==CPU memcmp catches any divergence. The ONLY swap vs
// convex_step.comp (CX4) is the impulse pass: BuildFrictionPoints + the FC3 SolveFrictionImpulse (normal +
// t1/t2 Coulomb-cone tangent) instead of BuildManifold + SolveManifoldImpulse — so a box grips/slides on a
// ramp and a stack stands at angDamp = kOne (friction holds it).
//
// SINGLE THREAD ([numthreads(1,1,1)]): the world step is order-dependent (the world Gauss-Seidel + the
// in-place de-pen — later pairs see earlier updates), so the GPU mirror MUST be single-thread (the
// convex_step.comp / fpx_solve.comp convention). The body set is small (a ramp/floor + a few boxes).
//
// INTEGER WIDTH (the determinism crux, the FPX3/CX1/FC3 lesson): fxmul/fxdiv/FxISqrt + the FxDot/FxCross/
// FxMat3MulVec/quaternion Q16.16 products use int64_t. DXC -spirv compiles int64 (the Int64 capability, the
// convex_step.comp/fric_solve.comp pattern); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse
// int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal
// hf_gen_msl list). The Metal --fric-ramp/--fric-stack showcase runs the CPU fric::StepFrictionWorldN over
// the same world -> byte-identical to this GPU result BY CONSTRUCTION, while the Vulkan side carries the
// GPU==CPU bit-identity proof.
//
// stepEnabled=0 -> write the input bodies back UNCHANGED (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..2):
//   b0 gBodies : the body array (N x FxBody), READ+WRITE (stepped in place over K ticks).
//   b1 gBoxes  : the box half-extents (N x FxBox, packed int4 {hx,hy,hz,_pad}), READ.
//   b2 gParams : the step config + counts + mu, READ.

#define HF_FPX_FRAC 16   // MUST match fpx.h::kFrac
#define HF_FPX_ONE  (1 << HF_FPX_FRAC)
#define HF_FPX_HALF (HF_FPX_ONE / 2)           // kHalf
#define HF_CX_EDGE_EPS (HF_FPX_ONE / 256)      // MUST match convex.h::kEdgeEps == kOne/256
#define HF_CX_FACE_PREF_EPS (HF_FPX_ONE / 64)  // MUST match convex.h::kFacePrefEps == kOne/64
#define HF_FLAG_DYNAMIC 1u                     // MUST match fpx.h::kFlagDynamic
#define HF_MAX_BODIES 16                        // the small-scene cap (ramp/floor + a few boxes)

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

// std430 box half-extents (engine/sim/convex.h::FxBox) packed as int4 {hx,hy,hz,_pad}.
struct BoxGpu { int4 h; };

// The step params (== the host FrictionStepConfig + counts), packed for std430.
struct StepParams {
    int4 c0;   // x=bodyCount, y=ticks, z=stepEnabled, w=solveIters
    int4 c1;   // x=posIters, y=restitution, z=slop, w=beta
    int4 c2;   // x=linDamp, y=angDamp, z=dt, w=mu
    int4 c3;   // x=gravity.x, y=gravity.y, z=gravity.z, w=_pad
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxBody>     gBodies : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<BoxGpu>     gBoxes  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<StepParams> gParams : register(u2);

// ===== VERBATIM Q16.16 toolbox (== convex_step.comp / fric_solve.comp / convex.h / fric.h) =====
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_FPX_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_FPX_FRAC) / (int64_t)b); }

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
int FxLength(int3 v) {
    int64_t sx = (int64_t)v.x * (int64_t)v.x;
    int64_t sy = (int64_t)v.y * (int64_t)v.y;
    int64_t sz = (int64_t)v.z * (int64_t)v.z;
    return (int)FxISqrt(sx + sy + sz);
}
int3 FxNormalize(int3 v) {
    int len = FxLength(v);
    if (len == 0) return int3(0, HF_FPX_ONE, 0);
    return int3(fxdiv(v.x, len), fxdiv(v.y, len), fxdiv(v.z, len));
}
int FxDot(int3 a, int3 b) {
    int64_t d = (int64_t)a.x * (int64_t)b.x + (int64_t)a.y * (int64_t)b.y + (int64_t)a.z * (int64_t)b.z;
    return (int)(d >> HF_FPX_FRAC);
}
int3 FxCross(int3 a, int3 b) {
    return int3(fxmul(a.y, b.z) - fxmul(a.z, b.y),
                fxmul(a.z, b.x) - fxmul(a.x, b.z),
                fxmul(a.x, b.y) - fxmul(a.y, b.x));
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
int absI(int v) { return v < 0 ? -v : v; }
int ProjectedRadius(int3 L, int3 ax0, int3 ax1, int3 ax2, int3 h) {
    int d0 = absI(FxDot(L, ax0));
    int d1 = absI(FxDot(L, ax1));
    int d2 = absI(FxDot(L, ax2));
    return fxmul(d0, h.x) + fxmul(d1, h.y) + fxmul(d2, h.z);
}
int FxAt(int3 v, uint i) { return (i == 0) ? v.x : ((i == 1) ? v.y : v.z); }
int3 Mat3MulVec(int M[9], int3 v) {
    int3 r0 = int3(M[0], M[1], M[2]);
    int3 r1 = int3(M[3], M[4], M[5]);
    int3 r2 = int3(M[6], M[7], M[8]);
    return int3(FxDot(r0, v), FxDot(r1, v), FxDot(r2, v));
}
int3 BoxInvInertiaBody(int3 h, int invMass) {
    if (invMass == 0) return int3(0, 0, 0);
    int hx2 = fxmul(h.x, h.x), hy2 = fxmul(h.y, h.y), hz2 = fxmul(h.z, h.z);
    int three = 3 * invMass;
    return int3(fxdiv(three, hy2 + hz2), fxdiv(three, hx2 + hz2), fxdiv(three, hx2 + hy2));
}
void WorldInvInertia(int3 ax0, int3 ax1, int3 ax2, int3 invI, out int M[9]) {
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

// ===== VERBATIM fric.h::MakeTangentBasis (FC1 Gram-Schmidt) (== fric_solve.comp) =====
uint LeastAlignedAxis(int3 n) {
    int a0 = absI(n.x), a1 = absI(n.y), a2 = absI(n.z);
    uint best = 0u;
    int bestVal = a0;
    if (a1 < bestVal) { bestVal = a1; best = 1u; }   // strict-< -> lowest index keeps a tie
    if (a2 < bestVal) { bestVal = a2; best = 2u; }
    return best;
}
int3 CardinalAxis(uint i) {
    return int3((i == 0u) ? HF_FPX_ONE : 0, (i == 1u) ? HF_FPX_ONE : 0, (i == 2u) ? HF_FPX_ONE : 0);
}
void MakeTangentBasis(int3 n, out int3 t1, out int3 t2) {
    uint mi = LeastAlignedAxis(n);
    int3 e = CardinalAxis(mi);
    int eDotN = FxDot(e, n);
    int3 r = int3(e.x - fxmul(eDotN, n.x), e.y - fxmul(eDotN, n.y), e.z - fxmul(eDotN, n.z));
    t1 = FxNormalize(r);
    t2 = FxCross(n, t1);
}

// ===== Quaternion integrate (== fpx.h::IntegrateBodyFull / convex_step.comp) =====
int4 FxQuatMul(int4 a, int4 b) {
    int4 r;
    r.w = fxmul(a.w, b.w) - fxmul(a.x, b.x) - fxmul(a.y, b.y) - fxmul(a.z, b.z);
    r.x = fxmul(a.w, b.x) + fxmul(a.x, b.w) + fxmul(a.y, b.z) - fxmul(a.z, b.y);
    r.y = fxmul(a.w, b.y) - fxmul(a.x, b.z) + fxmul(a.y, b.w) + fxmul(a.z, b.x);
    r.z = fxmul(a.w, b.z) + fxmul(a.x, b.y) - fxmul(a.y, b.x) + fxmul(a.z, b.w);
    return r;
}
int4 FxQuatNormalize(int4 q) {
    int64_t sx = (int64_t)q.x * (int64_t)q.x;
    int64_t sy = (int64_t)q.y * (int64_t)q.y;
    int64_t sz = (int64_t)q.z * (int64_t)q.z;
    int64_t sw = (int64_t)q.w * (int64_t)q.w;
    int len = (int)FxISqrt(sx + sy + sz + sw);
    if (len == 0) return int4(0, 0, 0, HF_FPX_ONE);
    return int4(fxdiv(q.x, len), fxdiv(q.y, len), fxdiv(q.z, len), fxdiv(q.w, len));
}
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

// ===== The SAT result (== convex.h::SatResult) =====
struct Sat { bool overlap; uint axisIndex; int penetration; int3 axis; };

// BoxSat (== convex.h::BoxSat, VERBATIM the convex_step.comp body).
Sat BoxSat(FxBody bA, int3 hA, FxBody bB, int3 hB) {
    Sat outR; outR.overlap = false; outR.axisIndex = 0u; outR.penetration = 0; outR.axis = int3(0,0,0);
    int4 qA = int4(bA.ox, bA.oy, bA.oz, bA.ow);
    int4 qB = int4(bB.ox, bB.oy, bB.oz, bB.ow);
    int3 axA0 = FxRotate(qA, int3(HF_FPX_ONE, 0, 0));
    int3 axA1 = FxRotate(qA, int3(0, HF_FPX_ONE, 0));
    int3 axA2 = FxRotate(qA, int3(0, 0, HF_FPX_ONE));
    int3 axB0 = FxRotate(qB, int3(HF_FPX_ONE, 0, 0));
    int3 axB1 = FxRotate(qB, int3(0, HF_FPX_ONE, 0));
    int3 axB2 = FxRotate(qB, int3(0, 0, HF_FPX_ONE));
    int3 axA[3] = { axA0, axA1, axA2 };
    int3 axB[3] = { axB0, axB1, axB2 };
    int3 t = int3(bB.px - bA.px, bB.py - bA.py, bB.pz - bA.pz);

    bool found = false; int minPen = 0; uint minIndex = 0u; int3 minAxis = int3(0,0,0);
    bool separated = false;
    for (int phase = 0; phase < 15 && !separated; ++phase) {
        int3 rawL; bool skipDeg = false;
        if (phase < 3) { rawL = axA[phase]; }
        else if (phase < 6) { rawL = axB[phase - 3]; }
        else { int e = phase - 6; int i = e / 3, j = e % 3; rawL = FxCross(axA[i], axB[j]); skipDeg = true; }
        if (skipDeg) { int rawLen = FxLength(rawL); if (rawLen < HF_CX_EDGE_EPS) continue; }
        int3 L = FxNormalize(rawL);
        int rA = ProjectedRadius(L, axA0, axA1, axA2, hA);
        int rB = ProjectedRadius(L, axB0, axB1, axB2, hB);
        int dotLt = FxDot(L, t);
        int s = absI(dotLt);
        int sum = rA + rB;
        if (s > sum) { separated = true; continue; }
        int pen = sum - s;
        if (!found || pen < minPen) {
            found = true; minPen = pen; minIndex = (uint)phase;
            minAxis = (dotLt < 0) ? int3(-L.x, -L.y, -L.z) : L;
        }
    }
    if (separated) { outR.overlap = false; return outR; }
    outR.overlap = true; outR.axisIndex = minIndex; outR.penetration = minPen; outR.axis = minAxis;
    return outR;
}

// BoxSatStable (== convex.h::BoxSatStable): BoxSat + the face-preference post-filter.
Sat BoxSatStable(FxBody bA, int3 hA, FxBody bB, int3 hB) {
    Sat base = BoxSat(bA, hA, bB, hB);
    if (!base.overlap || base.axisIndex < 6u) return base;
    int4 qA = int4(bA.ox, bA.oy, bA.oz, bA.ow);
    int4 qB = int4(bB.ox, bB.oy, bB.oz, bB.ow);
    int3 axA0 = FxRotate(qA, int3(HF_FPX_ONE, 0, 0));
    int3 axA1 = FxRotate(qA, int3(0, HF_FPX_ONE, 0));
    int3 axA2 = FxRotate(qA, int3(0, 0, HF_FPX_ONE));
    int3 axB0 = FxRotate(qB, int3(HF_FPX_ONE, 0, 0));
    int3 axB1 = FxRotate(qB, int3(0, HF_FPX_ONE, 0));
    int3 axB2 = FxRotate(qB, int3(0, 0, HF_FPX_ONE));
    int3 axA[3] = { axA0, axA1, axA2 };
    int3 axB[3] = { axB0, axB1, axB2 };
    int3 t = int3(bB.px - bA.px, bB.py - bA.py, bB.pz - bA.pz);
    bool found = false; int minPen = 0; uint minIndex = 0u; int3 minAxis = int3(0,0,0);
    for (uint fi = 0u; fi < 6u; ++fi) {
        int3 rawL = (fi < 3u) ? axA[fi] : axB[fi - 3u];
        int3 L = FxNormalize(rawL);
        int rA = ProjectedRadius(L, axA0, axA1, axA2, hA);
        int rB = ProjectedRadius(L, axB0, axB1, axB2, hB);
        int dotLt = FxDot(L, t);
        int s = absI(dotLt);
        int sum = rA + rB;
        if (s > sum) continue;
        int pen = sum - s;
        if (!found || pen < minPen) {
            found = true; minPen = pen; minIndex = fi;
            minAxis = (dotLt < 0) ? int3(-L.x, -L.y, -L.z) : L;
        }
    }
    if (found && minPen <= base.penetration + HF_CX_FACE_PREF_EPS) {
        Sat r; r.overlap = true; r.axisIndex = minIndex; r.penetration = minPen; r.axis = minAxis;
        return r;
    }
    return base;
}

// ===== The manifold (== convex.h::ContactManifold) =====
struct Manifold { uint count; int3 pts[4]; int depth[4]; int3 normal; };

// BuildManifold (== convex.h::BuildManifold, VERBATIM the convex_step.comp body).
Manifold BuildManifold(FxBody bA, int3 hA, FxBody bB, int3 hB, Sat sat) {
    Manifold m; m.count = 0u; m.normal = int3(0,0,0);
    [unroll] for (int mi = 0; mi < 4; ++mi) { m.pts[mi] = int3(0,0,0); m.depth[mi] = 0; }
    if (!sat.overlap) return m;

    int4 qA = int4(bA.ox, bA.oy, bA.oz, bA.ow);
    int4 qB = int4(bB.ox, bB.oy, bB.oz, bB.ow);
    int3 axA0 = FxRotate(qA, int3(HF_FPX_ONE, 0, 0));
    int3 axA1 = FxRotate(qA, int3(0, HF_FPX_ONE, 0));
    int3 axA2 = FxRotate(qA, int3(0, 0, HF_FPX_ONE));
    int3 axB0 = FxRotate(qB, int3(HF_FPX_ONE, 0, 0));
    int3 axB1 = FxRotate(qB, int3(0, HF_FPX_ONE, 0));
    int3 axB2 = FxRotate(qB, int3(0, 0, HF_FPX_ONE));
    int3 axA[3] = { axA0, axA1, axA2 };
    int3 axB[3] = { axB0, axB1, axB2 };
    int3 posA = int3(bA.px, bA.py, bA.pz);
    int3 posB = int3(bB.px, bB.py, bB.pz);

    if (sat.axisIndex >= 6u) {
        uint ee = sat.axisIndex - 6u;
        uint i = ee / 3u, j = ee % 3u;
        int hAi = FxAt(hA, i);
        int hBj = FxAt(hB, j);
        int3 dA = int3(fxmul(axA[i].x, hAi), fxmul(axA[i].y, hAi), fxmul(axA[i].z, hAi));
        int3 dB = int3(fxmul(axB[j].x, hBj), fxmul(axB[j].y, hBj), fxmul(axB[j].z, hBj));
        int3 s0 = int3(posA.x - dA.x, posA.y - dA.y, posA.z - dA.z);
        int3 s1 = int3(posB.x - dB.x, posB.y - dB.y, posB.z - dB.z);
        int3 u = int3(dA.x * 2, dA.y * 2, dA.z * 2);
        int3 v = int3(dB.x * 2, dB.y * 2, dB.z * 2);
        int3 w = int3(s0.x - s1.x, s0.y - s1.y, s0.z - s1.z);
        int a = FxDot(u, u);
        int b = FxDot(u, v);
        int c = FxDot(v, v);
        int d = FxDot(u, w);
        int e = FxDot(v, w);
        int denom = fxmul(a, c) - fxmul(b, b);
        int sParam;
        if (denom > HF_CX_EDGE_EPS) {
            sParam = fxdiv(fxmul(b, e) - fxmul(c, d), denom);
            if (sParam < 0) sParam = 0; else if (sParam > HF_FPX_ONE) sParam = HF_FPX_ONE;
        } else { sParam = HF_FPX_HALF; }
        int tnum = fxmul(b, sParam) + e;
        int tdenom = c;
        int tt;
        if (tdenom > HF_CX_EDGE_EPS) {
            tt = fxdiv(tnum, tdenom);
            if (tt < 0) tt = 0; else if (tt > HF_FPX_ONE) tt = HF_FPX_ONE;
        } else { tt = HF_FPX_HALF; }
        if (a > HF_CX_EDGE_EPS) {
            int s2 = fxdiv(fxmul(b, tt) - d, a);
            if (s2 < 0) s2 = 0; else if (s2 > HF_FPX_ONE) s2 = HF_FPX_ONE;
            sParam = s2;
        }
        int3 cA = int3(s0.x + fxmul(u.x, sParam), s0.y + fxmul(u.y, sParam), s0.z + fxmul(u.z, sParam));
        int3 cB = int3(s1.x + fxmul(v.x, tt), s1.y + fxmul(v.y, tt), s1.z + fxmul(v.z, tt));
        m.count = 1u;
        m.pts[0] = int3((cA.x + cB.x) / 2, (cA.y + cB.y) / 2, (cA.z + cB.z) / 2);
        m.depth[0] = sat.penetration;
        m.normal = sat.axis;
        return m;
    }

    bool refIsA = (sat.axisIndex < 3u);
    uint refIdx = refIsA ? sat.axisIndex : (sat.axisIndex - 3u);
    int3 refAxes[3];
    int3 incAxes[3];
    for (int rk = 0; rk < 3; ++rk) {
        refAxes[rk] = refIsA ? axA[rk] : axB[rk];
        incAxes[rk] = refIsA ? axB[rk] : axA[rk];
    }
    int3 refPos = refIsA ? posA : posB;
    int3 incPos = refIsA ? posB : posA;
    int3 refHv  = refIsA ? hA : hB;
    int3 incHv  = refIsA ? hB : hA;

    int3 tRefToInc = int3(incPos.x - refPos.x, incPos.y - refPos.y, incPos.z - refPos.z);
    int3 n = refAxes[refIdx];
    if (FxDot(n, tRefToInc) < 0) n = int3(-n.x, -n.y, -n.z);
    int Href = FxAt(refHv, refIdx);

    uint ui = (refIdx == 0u) ? 1u : 0u;
    uint vi = (refIdx == 2u) ? 1u : 2u;
    if (refIdx == 1u) { ui = 0u; vi = 2u; }
    int3 u = refAxes[ui];
    int3 v = refAxes[vi];
    int hu = FxAt(refHv, ui);
    int hv = FxAt(refHv, vi);
    int3 faceCenter = int3(refPos.x + fxmul(n.x, Href), refPos.y + fxmul(n.y, Href),
                           refPos.z + fxmul(n.z, Href));

    uint bestK = 0u; bool bestNeg = false; int bestDot = 0; bool firstK = true;
    for (uint k = 0u; k < 3u; ++k) {
        for (int sgn = 0; sgn < 2; ++sgn) {
            int3 cand = (sgn == 0) ? incAxes[k] : int3(-incAxes[k].x, -incAxes[k].y, -incAxes[k].z);
            int dt = FxDot(cand, n);
            if (firstK || dt < bestDot) { bestDot = dt; bestK = k; bestNeg = (sgn == 1); firstK = false; }
        }
    }
    int3 incN = bestNeg ? int3(-incAxes[bestK].x, -incAxes[bestK].y, -incAxes[bestK].z) : incAxes[bestK];
    int incHk = FxAt(incHv, bestK);
    int3 incFaceCenter = int3(incPos.x + fxmul(incN.x, incHk), incPos.y + fxmul(incN.y, incHk),
                              incPos.z + fxmul(incN.z, incHk));
    uint iui = (bestK == 0u) ? 1u : 0u;
    uint ivi = (bestK == 2u) ? 1u : 2u;
    if (bestK == 1u) { iui = 0u; ivi = 2u; }
    int3 iu = incAxes[iui];
    int3 iv = incAxes[ivi];
    int ihu = FxAt(incHv, iui);
    int ihv = FxAt(incHv, ivi);

    int su[4] = { 1, -1, -1, 1 };
    int sv[4] = { 1, 1, -1, -1 };
    int3 poly[8];
    int polyN = 4;
    for (int k = 0; k < 4; ++k) {
        poly[k] = int3(incFaceCenter.x + su[k] * fxmul(iu.x, ihu) + sv[k] * fxmul(iv.x, ihv),
                       incFaceCenter.y + su[k] * fxmul(iu.y, ihu) + sv[k] * fxmul(iv.y, ihv),
                       incFaceCenter.z + su[k] * fxmul(iu.z, ihu) + sv[k] * fxmul(iv.z, ihv));
    }

    int3 planeA[4] = { u, u, v, v };
    int planeH[4]  = { hu, hu, hv, hv };
    int planeS[4]  = { 1, -1, 1, -1 };
    for (int pl = 0; pl < 4; ++pl) {
        int3 outV[8];
        int outN = 0;
        if (polyN == 0) break;
        int3 prev = poly[polyN - 1];
        int3 relP = int3(prev.x - faceCenter.x, prev.y - faceCenter.y, prev.z - faceCenter.z);
        int fprev = planeH[pl] - planeS[pl] * FxDot(planeA[pl], relP);
        for (int k = 0; k < polyN; ++k) {
            int3 cur = poly[k];
            int3 relC = int3(cur.x - faceCenter.x, cur.y - faceCenter.y, cur.z - faceCenter.z);
            int fcur = planeH[pl] - planeS[pl] * FxDot(planeA[pl], relC);
            bool curIn = (fcur >= 0);
            bool prevIn = (fprev >= 0);
            if (curIn) {
                if (!prevIn) {
                    int denomP = fprev - fcur;
                    int tp = (denomP != 0) ? fxdiv(fprev, denomP) : 0;
                    outV[outN++] = int3(prev.x + fxmul(cur.x - prev.x, tp),
                                       prev.y + fxmul(cur.y - prev.y, tp),
                                       prev.z + fxmul(cur.z - prev.z, tp));
                }
                outV[outN++] = cur;
            } else if (prevIn) {
                int denomP = fprev - fcur;
                int tp = (denomP != 0) ? fxdiv(fprev, denomP) : 0;
                outV[outN++] = int3(prev.x + fxmul(cur.x - prev.x, tp),
                                   prev.y + fxmul(cur.y - prev.y, tp),
                                   prev.z + fxmul(cur.z - prev.z, tp));
            }
            prev = cur; fprev = fcur;
        }
        polyN = outN;
        for (int k = 0; k < polyN; ++k) poly[k] = outV[k];
    }

    int3 candPts[8];
    int  candDepth[8];
    int  candN = 0;
    for (int k = 0; k < polyN; ++k) {
        int3 rel = int3(faceCenter.x - poly[k].x, faceCenter.y - poly[k].y, faceCenter.z - poly[k].z);
        int d = FxDot(n, rel);
        if (d >= 0) { candPts[candN] = poly[k]; candDepth[candN] = d; candN++; }
    }

    m.normal = n;
    if (candN == 0) {
        m.count = 1u;
        int3 fb = (polyN > 0) ? poly[0] : incFaceCenter;
        m.pts[0] = fb;
        m.depth[0] = sat.penetration;
    } else {
        int deepest = 0;
        for (int k = 1; k < candN; ++k) if (candDepth[k] > candDepth[deepest]) deepest = k;
        m.pts[0] = candPts[deepest]; m.depth[0] = candDepth[deepest];
        int cnt = 1;
        for (int k = 0; k < candN && cnt < 4; ++k) {
            if (k == deepest) continue;
            m.pts[cnt] = candPts[k]; m.depth[cnt] = candDepth[k]; cnt++;
        }
        m.count = (uint)cnt;
    }
    return m;
}

// SolveFrictionImpulse over a pair (== fric.h::SolveFrictionImpulse, VERBATIM the fric_solve.comp body but
// over a built manifold + a precomputed A->B basis): one inner sweep (iters=1), the world-level Gauss-Seidel
// is the outer loop. The NORMAL part reproduces convex_step.comp's SolveManifoldImpulse EXACTLY; the TANGENT
// part is the cone-clamped friction NEW PHYSICS. Mutates bA/bB vel+angVel in place.
void SolveFrictionImpulse(inout FxBody bA, inout FxBody bB, int invIaW[9], int invIbW[9],
                          Manifold m, int restitution, int mu) {
    if (m.count == 0u) return;
    // The A->B normal sign-correction (the BuildFrictionPoints rule, applied ONCE per pair).
    int3 nAB = m.normal;
    int3 ab = int3(bB.px - bA.px, bB.py - bA.py, bB.pz - bA.pz);
    if (FxDot(nAB, ab) < 0) nAB = int3(-nAB.x, -nAB.y, -nAB.z);
    int3 t1, t2;
    MakeTangentBasis(nAB, t1, t2);   // the FC1 tangent basis, computed ONCE.

    int invMassA = bA.invMass;
    int invMassB = bB.invMass;

    for (uint pi = 0; pi < m.count; ++pi) {
        int3 ppt = m.pts[pi];
        int3 posA2 = int3(bA.px, bA.py, bA.pz);
        int3 posB2 = int3(bB.px, bB.py, bB.pz);
        int3 rA = int3(ppt.x - posA2.x, ppt.y - posA2.y, ppt.z - posA2.z);
        int3 rB = int3(ppt.x - posB2.x, ppt.y - posB2.y, ppt.z - posB2.z);

        // ---- NORMAL impulse (the SolveManifoldImpulse form reproduced VERBATIM) ----
        int3 wA0 = int3(bA.ax, bA.ay, bA.az);
        int3 wB0 = int3(bB.ax, bB.ay, bB.az);
        int3 velA0 = int3(bA.vx, bA.vy, bA.vz);
        int3 velB0 = int3(bB.vx, bB.vy, bB.vz);
        int3 vpA0 = velA0 + FxCross(wA0, rA);
        int3 vpB0 = velB0 + FxCross(wB0, rB);
        int vn = FxDot(int3(vpB0.x - vpA0.x, vpB0.y - vpA0.y, vpB0.z - vpA0.z), nAB);
        int jn = 0;
        if (vn < 0) {
            int3 raxn = FxCross(rA, nAB);
            int3 rbxn = FxCross(rB, nAB);
            int angA = FxDot(nAB, FxCross(Mat3MulVec(invIaW, raxn), rA));
            int angB = FxDot(nAB, FxCross(Mat3MulVec(invIbW, rbxn), rB));
            int kn = invMassA + invMassB + angA + angB;
            if (kn > 0) {
                jn = fxdiv(-fxmul(HF_FPX_ONE + restitution, vn), kn);
                if (jn < 0) jn = 0;
                int3 J = int3(fxmul(nAB.x, jn), fxmul(nAB.y, jn), fxmul(nAB.z, jn));
                int3 dvA = int3(fxmul(J.x, invMassA), fxmul(J.y, invMassA), fxmul(J.z, invMassA));
                int3 dwA = Mat3MulVec(invIaW, FxCross(rA, J));
                bA.vx -= dvA.x; bA.vy -= dvA.y; bA.vz -= dvA.z;
                bA.ax -= dwA.x; bA.ay -= dwA.y; bA.az -= dwA.z;
                int3 dvB = int3(fxmul(J.x, invMassB), fxmul(J.y, invMassB), fxmul(J.z, invMassB));
                int3 dwB = Mat3MulVec(invIbW, FxCross(rB, J));
                bB.vx += dvB.x; bB.vy += dvB.y; bB.vz += dvB.z;
                bB.ax += dwB.x; bB.ay += dwB.y; bB.az += dwB.z;
            }
        }

        // ---- TANGENT friction impulses (THE NEW PHYSICS), t1 then t2, cone-clamped to +-mu*jn ----
        int coneLo = -fxmul(mu, jn);
        int coneHi =  fxmul(mu, jn);
        int3 tangents[2] = { t1, t2 };
        [unroll] for (int ti = 0; ti < 2; ++ti) {
            int3 tdir = tangents[ti];
            int3 wA = int3(bA.ax, bA.ay, bA.az);
            int3 wB = int3(bB.ax, bB.ay, bB.az);
            int3 velA = int3(bA.vx, bA.vy, bA.vz);
            int3 velB = int3(bB.vx, bB.vy, bB.vz);
            int3 vpA = velA + FxCross(wA, rA);
            int3 vpB = velB + FxCross(wB, rB);
            int vt = FxDot(int3(vpB.x - vpA.x, vpB.y - vpA.y, vpB.z - vpA.z), tdir);
            int3 raxt = FxCross(rA, tdir);
            int3 rbxt = FxCross(rB, tdir);
            int angA = FxDot(tdir, FxCross(Mat3MulVec(invIaW, raxt), rA));
            int angB = FxDot(tdir, FxCross(Mat3MulVec(invIbW, rbxt), rB));
            int kt = invMassA + invMassB + angA + angB;
            if (kt <= 0) continue;
            int jt = fxdiv(-vt, kt);
            if (jt < coneLo) jt = coneLo;
            else if (jt > coneHi) jt = coneHi;   // CLAMP to the Coulomb cone +-mu*jn
            int3 Jt = int3(fxmul(tdir.x, jt), fxmul(tdir.y, jt), fxmul(tdir.z, jt));
            int3 dvA = int3(fxmul(Jt.x, invMassA), fxmul(Jt.y, invMassA), fxmul(Jt.z, invMassA));
            int3 dwA = Mat3MulVec(invIaW, FxCross(rA, Jt));
            bA.vx -= dvA.x; bA.vy -= dvA.y; bA.vz -= dvA.z;
            bA.ax -= dwA.x; bA.ay -= dwA.y; bA.az -= dwA.z;
            int3 dvB = int3(fxmul(Jt.x, invMassB), fxmul(Jt.y, invMassB), fxmul(Jt.z, invMassB));
            int3 dwB = Mat3MulVec(invIbW, FxCross(rB, Jt));
            bB.vx += dvB.x; bB.vy += dvB.y; bB.vz += dvB.z;
            bB.ax += dwB.x; bB.ay += dwB.y; bB.az += dwB.z;
        }
    }
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
    int mu          = gParams[0].c2.w;
    int3 gravity    = int3(gParams[0].c3.x, gParams[0].c3.y, gParams[0].c3.z);

    int n = bodyCount;
    if (n > HF_MAX_BODIES) n = HF_MAX_BODIES;

    FxBody body[HF_MAX_BODIES];
    int3   half[HF_MAX_BODIES];
    for (int li = 0; li < n; ++li) { body[li] = gBodies[li]; half[li] = gBoxes[li].h.xyz; }

    if (stepEnabled == 0) {
        for (int wi0 = 0; wi0 < n; ++wi0) gBodies[wi0] = body[wi0];
        return;
    }

    // ===== StepFrictionWorldN: `ticks` x StepFrictionWorld (== fric.h, VERBATIM) =====
    for (int tk = 0; tk < ticks; ++tk) {
        // (1) predict-integrate dynamic bodies + per-tick damping.
        for (int i1 = 0; i1 < n; ++i1) {
            if ((body[i1].invMass != 0) && (body[i1].flags & HF_FLAG_DYNAMIC)) {
                IntegrateBodyFull(body[i1], gravity, dt);
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

        // (2) world inverse inertias, recomputed once per tick from the post-integrate orient.
        int invIW[HF_MAX_BODIES][9];
        for (int i2 = 0; i2 < n; ++i2) {
            int4 q = int4(body[i2].ox, body[i2].oy, body[i2].oz, body[i2].ow);
            int3 ax0 = FxRotate(q, int3(HF_FPX_ONE, 0, 0));
            int3 ax1 = FxRotate(q, int3(0, HF_FPX_ONE, 0));
            int3 ax2 = FxRotate(q, int3(0, 0, HF_FPX_ONE));
            int3 invIb = BoxInvInertiaBody(half[i2], body[i2].invMass);
            int M[9];
            WorldInvInertia(ax0, ax1, ax2, invIb, M);
            [unroll] for (int z = 0; z < 9; ++z) invIW[i2][z] = M[z];
        }

        // (3) impulse solve — world Gauss-Seidel over the all-pairs list, fixed i<j order. The ONLY swap vs
        // convex_step.comp: BuildManifold -> SolveFrictionImpulse (normal + cone-clamped tangent) per pair.
        for (int sweep = 0; sweep < solveIters; ++sweep) {
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    if (body[i].invMass == 0 && body[j].invMass == 0) continue;
                    Sat sat = BoxSatStable(body[i], half[i], body[j], half[j]);
                    if (!sat.overlap) continue;
                    Manifold m = BuildManifold(body[i], half[i], body[j], half[j], sat);
                    if (m.count == 0u) continue;
                    int iaW[9]; int ibW[9];
                    [unroll] for (int z = 0; z < 9; ++z) { iaW[z] = invIW[i][z]; ibW[z] = invIW[j][z]; }
                    FxBody bi = body[i]; FxBody bj = body[j];
                    SolveFrictionImpulse(bi, bj, iaW, ibW, m, restitution, mu);
                    body[i] = bi; body[j] = bj;
                }
            }
        }

        // (4) position de-penetration — posIters sweeps, fixed i<j order, split-by-invMass linear push.
        for (int pit = 0; pit < posIters; ++pit) {
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    int invSum = body[i].invMass + body[j].invMass;
                    if (invSum == 0) continue;
                    Sat sat = BoxSatStable(body[i], half[i], body[j], half[j]);
                    if (!sat.overlap) continue;
                    int3 nrm = sat.axis;
                    int3 ab = int3(body[j].px - body[i].px, body[j].py - body[i].py, body[j].pz - body[i].pz);
                    if (FxDot(nrm, ab) < 0) nrm = int3(-nrm.x, -nrm.y, -nrm.z);
                    int excess = sat.penetration - slop;
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
    }

    // Write the final body world back.
    for (int wi = 0; wi < n; ++wi) gBodies[wi] = body[wi];
}
