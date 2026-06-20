// Slice PS4 — Deterministic Persistent Contacts SLEEPING ISLANDS (THE NEW PHYSICS) compute pass (the 4th
// slice of FLAGSHIP #21: DETERMINISTIC WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, hf::sim::persist). ONE
// THREAD runs the WHOLE StepWarmSleepWorldN over the SMALL body set, K ticks — copying engine/sim/persist.h::
// StepWarmSleepWorld / StepWarmSleepWorldN VERBATIM (the SAME PS3 warm step PLUS: the per-body integer
// KineticEnergy = FxLength(vel)+FxLength(angVel), the fixed wake/sleep HYSTERESIS quietTicks update, the
// all-pairs BoxSatStable contact-adjacency + the fixed-order union-find-free ISLAND wakefulness propagation,
// the FREEZE of asleep bodies (vel/angVel zeroed → exactly zero residual), and the SKIP of integrate + solve +
// de-pen for sleeping bodies/islands; the sleeping pairs keep their prior cache entries). The final gBodies
// array AND the per-body gSleep states are byte-identical to the CPU reference → the host GPU==CPU memcmp
// catches any divergence.
//
// THE NEW PHYSICS (the PS4 headline): a warm-started tower rests + goes to SLEEP (exactly zero residual — an
// asleep body is frozen at zero velocity and skipped by integrate), then a thrown box WAKES the struck body →
// the island propagation wakes the WHOLE contact-connected island deterministically. A STATIC body is never an
// island-waker (the floor doesn't keep the tower awake). INTEGER bit-exact, every order PINNED.
//
// SINGLE THREAD ([numthreads(1,1,1)]): the world step is order-dependent (the cache + sleep state carry across
// ticks, the in-place de-pen + later pairs see earlier updates), so the GPU mirror MUST be single-thread (the
// persist_warm.comp / fric_step.comp convention). The body set is small (a floor + a few slabs).
//
// INTEGER WIDTH (the determinism crux): fxmul/fxdiv/FxISqrt + the Q16.16 products + the FxLength KE sqrt use
// int64_t. DXC -spirv compiles int64; glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64 in HLSL,
// so this shader is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal hf_gen_msl list). The
// Metal --persist-sleep / --persist-wake showcase runs the CPU persist::StepWarmSleepWorldN over the same
// world -> byte-identical to this GPU result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU proof.
//
// stepEnabled=0 -> write the input bodies + cleared sleep states back (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..3):
//   b0 gBodies : the body array (N x FxBody), READ+WRITE (stepped in place over K ticks).
//   b1 gBoxes  : the box half-extents (N x FxBox, packed int4 {hx,hy,hz,_pad}), READ.
//   b2 gParams : the step config + counts + mu + the sleep thresholds, READ.
//   b3 gSleep  : the per-body SleepState {energy, quietTicks, asleep}, WRITE (final states).

#define HF_FPX_FRAC 16
#define HF_FPX_ONE  (1 << HF_FPX_FRAC)
#define HF_FPX_HALF (HF_FPX_ONE / 2)
#define HF_CX_EDGE_EPS (HF_FPX_ONE / 256)
#define HF_CX_FACE_PREF_EPS (HF_FPX_ONE / 64)
#define HF_FLAG_DYNAMIC 1u
#define HF_MAX_BODIES 16
#define HF_MAX_PAIRS  ((HF_MAX_BODIES * (HF_MAX_BODIES - 1)) / 2)   // 120
#define HF_MAX_CACHE  (HF_MAX_PAIRS * 4)                            // 480 (4 points/pair)

struct FxBody {
    int  px, py, pz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
    int  radius;
    int  ox, oy, oz, ow;
    int  ax, ay, az;
};
struct BoxGpu { int4 h; };
struct StepParams {
    int4 c0;   // x=bodyCount, y=ticks, z=stepEnabled, w=solveIters
    int4 c1;   // x=posIters, y=restitution, z=slop, w=beta
    int4 c2;   // x=linDamp, y=angDamp, z=dt, w=mu
    int4 c3;   // x=gravity.x, y=gravity.y, z=gravity.z, w=_pad
    int4 c4;   // x=sleepThreshold, y=wakeThreshold, z=sleepDelay, w=_pad
};
// The per-body sleep state mirror (== persist.h::SleepState: energy, quietTicks, asleep).
struct SleepGpu { int energy; uint quietTicks; uint asleep; uint _pad; };

[[vk::binding(0, 0)]] RWStructuredBuffer<FxBody>     gBodies : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<BoxGpu>     gBoxes  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<StepParams> gParams : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<SleepGpu>   gSleep  : register(u3);

// ===== VERBATIM Q16.16 toolbox (== persist_warm.comp / fric_step.comp / convex.h / persist.h) =====
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

// ===== VERBATIM fric.h::MakeTangentBasis (FC1 Gram-Schmidt) =====
uint LeastAlignedAxis(int3 n) {
    int a0 = absI(n.x), a1 = absI(n.y), a2 = absI(n.z);
    uint best = 0u;
    int bestVal = a0;
    if (a1 < bestVal) { bestVal = a1; best = 1u; }
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

// ===== Quaternion integrate (== fpx.h::IntegrateBodyFull) =====
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

// ===== The SAT result + BoxSat / BoxSatStable (== convex.h, VERBATIM the persist_warm.comp body) =====
struct Sat { bool overlap; uint axisIndex; int penetration; int3 axis; };
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

// ===== The manifold + BuildManifold (== convex.h::BuildManifold, VERBATIM the persist_warm.comp body) =====
struct Manifold { uint count; int3 pts[4]; int depth[4]; int3 normal; };
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

// ===== The PS1 ContactKey (== persist.h::MakeContactKey / ContactKeysEqual, PURE int32) =====
struct ContactKey { uint bodyA; uint bodyB; uint axisIndex; uint featureIndex; };
ContactKey MakeContactKey(uint bodyAIdx, uint bodyBIdx, uint axisIndex, uint pointIndex) {
    ContactKey k;
    if (bodyAIdx <= bodyBIdx) { k.bodyA = bodyAIdx; k.bodyB = bodyBIdx; }
    else                      { k.bodyA = bodyBIdx; k.bodyB = bodyAIdx; }
    k.axisIndex = axisIndex;
    k.featureIndex = pointIndex;
    return k;
}
bool ContactKeysEqual(ContactKey a, ContactKey b) {
    return a.bodyA == b.bodyA && a.bodyB == b.bodyB &&
           a.axisIndex == b.axisIndex && a.featureIndex == b.featureIndex;
}

// ===== The keyed friction manifold (== persist.h::KeyedFrictionManifold / fric::FrictionManifold) =====
struct Keyed {
    uint  count;
    int3  pt[4];
    int3  n;
    int3  t1, t2;
    int   jn[4];
    int   jt1[4], jt2[4];
    ContactKey keys[4];
};

// BuildKeyedManifold (== persist.h::BuildKeyedManifold).
Keyed BuildKeyedManifold(uint iIdx, uint jIdx, FxBody bA, int3 hA, FxBody bB, int3 hB) {
    Keyed kd;
    kd.count = 0u;
    [unroll] for (int z = 0; z < 4; ++z) {
        kd.pt[z] = int3(0,0,0); kd.jn[z] = 0; kd.jt1[z] = 0; kd.jt2[z] = 0;
        ContactKey zk; zk.bodyA = 0u; zk.bodyB = 0u; zk.axisIndex = 0u; zk.featureIndex = 0u; kd.keys[z] = zk;
    }
    kd.n = int3(0,0,0); kd.t1 = int3(0,0,0); kd.t2 = int3(0,0,0);
    Sat sat = BoxSatStable(bA, hA, bB, hB);
    if (!sat.overlap) return kd;
    Manifold m = BuildManifold(bA, hA, bB, hB, sat);
    if (m.count == 0u) return kd;
    int3 nAB = m.normal;
    int3 ab = int3(bB.px - bA.px, bB.py - bA.py, bB.pz - bA.pz);
    if (FxDot(nAB, ab) < 0) nAB = int3(-nAB.x, -nAB.y, -nAB.z);
    int3 t1, t2; MakeTangentBasis(nAB, t1, t2);
    kd.count = m.count; kd.n = nAB; kd.t1 = t1; kd.t2 = t2;
    for (uint i = 0u; i < m.count; ++i) {
        kd.pt[i] = m.pts[i];
        kd.keys[i] = MakeContactKey(iIdx, jIdx, sat.axisIndex, i);
    }
    return kd;
}

// SolveFrictionWarm (== persist.h::SolveFrictionWarm): prime once + accumulated cone sweeps. Mutates bA/bB.
void SolveFrictionWarm(inout FxBody bA, inout FxBody bB, int invIaW[9], int invIbW[9],
                       inout Keyed kd, int restitution, int mu, int iters) {
    if (kd.count == 0u) return;
    int invMassA = bA.invMass;
    int invMassB = bB.invMass;

    for (uint pp = 0u; pp < kd.count; ++pp) {
        int3 p = kd.pt[pp];
        int3 rA = int3(p.x - bA.px, p.y - bA.py, p.z - bA.pz);
        int3 rB = int3(p.x - bB.px, p.y - bB.py, p.z - bB.pz);
        int3 J = int3(fxmul(kd.n.x, kd.jn[pp]) + fxmul(kd.t1.x, kd.jt1[pp]) + fxmul(kd.t2.x, kd.jt2[pp]),
                      fxmul(kd.n.y, kd.jn[pp]) + fxmul(kd.t1.y, kd.jt1[pp]) + fxmul(kd.t2.y, kd.jt2[pp]),
                      fxmul(kd.n.z, kd.jn[pp]) + fxmul(kd.t1.z, kd.jt1[pp]) + fxmul(kd.t2.z, kd.jt2[pp]));
        int3 dvA = int3(fxmul(J.x, invMassA), fxmul(J.y, invMassA), fxmul(J.z, invMassA));
        int3 dwA = Mat3MulVec(invIaW, FxCross(rA, J));
        bA.vx -= dvA.x; bA.vy -= dvA.y; bA.vz -= dvA.z;
        bA.ax -= dwA.x; bA.ay -= dwA.y; bA.az -= dwA.z;
        int3 dvB = int3(fxmul(J.x, invMassB), fxmul(J.y, invMassB), fxmul(J.z, invMassB));
        int3 dwB = Mat3MulVec(invIbW, FxCross(rB, J));
        bB.vx += dvB.x; bB.vy += dvB.y; bB.vz += dvB.z;
        bB.ax += dwB.x; bB.ay += dwB.y; bB.az += dwB.z;
    }

    for (int it = 0; it < iters; ++it) {
        for (uint pi = 0u; pi < kd.count; ++pi) {
            int3 p = kd.pt[pi];
            int3 n = kd.n;
            int3 rA = int3(p.x - bA.px, p.y - bA.py, p.z - bA.pz);
            int3 rB = int3(p.x - bB.px, p.y - bB.py, p.z - bB.pz);

            {
                int3 vpA = int3(bA.vx, bA.vy, bA.vz) + FxCross(int3(bA.ax, bA.ay, bA.az), rA);
                int3 vpB = int3(bB.vx, bB.vy, bB.vz) + FxCross(int3(bB.ax, bB.ay, bB.az), rB);
                int vn = FxDot(int3(vpB.x - vpA.x, vpB.y - vpA.y, vpB.z - vpA.z), n);
                int3 raxn = FxCross(rA, n);
                int3 rbxn = FxCross(rB, n);
                int angA = FxDot(n, FxCross(Mat3MulVec(invIaW, raxn), rA));
                int angB = FxDot(n, FxCross(Mat3MulVec(invIbW, rbxn), rB));
                int kn = invMassA + invMassB + angA + angB;
                if (kn > 0) {
                    int djn = fxdiv(-fxmul(HF_FPX_ONE + restitution, vn), kn);
                    int newTotal = kd.jn[pi] + djn;
                    if (newTotal < 0) newTotal = 0;
                    int applied = newTotal - kd.jn[pi];
                    int3 J = int3(fxmul(n.x, applied), fxmul(n.y, applied), fxmul(n.z, applied));
                    int3 dvA = int3(fxmul(J.x, invMassA), fxmul(J.y, invMassA), fxmul(J.z, invMassA));
                    int3 dwA = Mat3MulVec(invIaW, FxCross(rA, J));
                    bA.vx -= dvA.x; bA.vy -= dvA.y; bA.vz -= dvA.z;
                    bA.ax -= dwA.x; bA.ay -= dwA.y; bA.az -= dwA.z;
                    int3 dvB = int3(fxmul(J.x, invMassB), fxmul(J.y, invMassB), fxmul(J.z, invMassB));
                    int3 dwB = Mat3MulVec(invIbW, FxCross(rB, J));
                    bB.vx += dvB.x; bB.vy += dvB.y; bB.vz += dvB.z;
                    bB.ax += dwB.x; bB.ay += dwB.y; bB.az += dwB.z;
                    kd.jn[pi] = newTotal;
                }
            }

            int coneLo = -fxmul(mu, kd.jn[pi]);
            int coneHi =  fxmul(mu, kd.jn[pi]);
            int3 tangents[2] = { kd.t1, kd.t2 };
            for (int ti = 0; ti < 2; ++ti) {
                int3 t = tangents[ti];
                int3 vpA = int3(bA.vx, bA.vy, bA.vz) + FxCross(int3(bA.ax, bA.ay, bA.az), rA);
                int3 vpB = int3(bB.vx, bB.vy, bB.vz) + FxCross(int3(bB.ax, bB.ay, bB.az), rB);
                int vt = FxDot(int3(vpB.x - vpA.x, vpB.y - vpA.y, vpB.z - vpA.z), t);
                int3 raxt = FxCross(rA, t);
                int3 rbxt = FxCross(rB, t);
                int angA = FxDot(t, FxCross(Mat3MulVec(invIaW, raxt), rA));
                int angB = FxDot(t, FxCross(Mat3MulVec(invIbW, rbxt), rB));
                int kt = invMassA + invMassB + angA + angB;
                if (kt <= 0) continue;
                int djt = fxdiv(-vt, kt);
                int prev = (ti == 0) ? kd.jt1[pi] : kd.jt2[pi];
                int newTotal = prev + djt;
                if (newTotal < coneLo) newTotal = coneLo;
                else if (newTotal > coneHi) newTotal = coneHi;
                int applied = newTotal - prev;
                int3 Jt = int3(fxmul(t.x, applied), fxmul(t.y, applied), fxmul(t.z, applied));
                int3 dvA = int3(fxmul(Jt.x, invMassA), fxmul(Jt.y, invMassA), fxmul(Jt.z, invMassA));
                int3 dwA = Mat3MulVec(invIaW, FxCross(rA, Jt));
                bA.vx -= dvA.x; bA.vy -= dvA.y; bA.vz -= dvA.z;
                bA.ax -= dwA.x; bA.ay -= dwA.y; bA.az -= dwA.z;
                int3 dvB = int3(fxmul(Jt.x, invMassB), fxmul(Jt.y, invMassB), fxmul(Jt.z, invMassB));
                int3 dwB = Mat3MulVec(invIbW, FxCross(rB, Jt));
                bB.vx += dvB.x; bB.vy += dvB.y; bB.vz += dvB.z;
                bB.ax += dwB.x; bB.ay += dwB.y; bB.az += dwB.z;
                if (ti == 0) kd.jt1[pi] = newTotal; else kd.jt2[pi] = newTotal;
            }
        }
    }
}

// KineticEnergy (== persist.h::KineticEnergy): FxLength(vel) + FxLength(angVel).
int KineticEnergy(FxBody b) {
    return FxLength(int3(b.vx, b.vy, b.vz)) + FxLength(int3(b.ax, b.ay, b.az));
}
bool IsDyn(FxBody b) { return (b.invMass != 0) && (b.flags & HF_FLAG_DYNAMIC); }

[numthreads(1, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x != 0u) return;   // single thread runs the WHOLE warm-started sleeping world step

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
    int sleepThr    = gParams[0].c4.x;
    int wakeThr     = gParams[0].c4.y;
    uint sleepDelay = (uint)gParams[0].c4.z;

    int n = bodyCount;
    if (n > HF_MAX_BODIES) n = HF_MAX_BODIES;

    FxBody body[HF_MAX_BODIES];
    int3   half[HF_MAX_BODIES];
    for (int li = 0; li < n; ++li) { body[li] = gBodies[li]; half[li] = gBoxes[li].h.xyz; }

    // The per-body sleep state (== persist.h::SleepState), carried across ticks. SEEDED from gSleep (the host
    // passes the carried-in state — zeroed for a fresh settle, the settled state for the post-throw wake scene).
    int  sEnergy[HF_MAX_BODIES];
    uint sQuiet[HF_MAX_BODIES];
    uint sAsleep[HF_MAX_BODIES];
    for (int si = 0; si < n; ++si) {
        sEnergy[si] = gSleep[si].energy; sQuiet[si] = gSleep[si].quietTicks; sAsleep[si] = gSleep[si].asleep;
    }

    if (stepEnabled == 0) {
        for (int wi0 = 0; wi0 < n; ++wi0) {
            gBodies[wi0] = body[wi0];
            SleepGpu sg; sg.energy = 0; sg.quietTicks = 0u; sg.asleep = 0u; sg._pad = 0u; gSleep[wi0] = sg;
        }
        return;
    }

    ContactKey cacheKey[HF_MAX_CACHE];
    int        cacheJn[HF_MAX_CACHE];
    int        cacheJt1[HF_MAX_CACHE];
    int        cacheJt2[HF_MAX_CACHE];
    int        cacheCount = 0;

    // ===== StepWarmSleepWorldN: `ticks` x StepWarmSleepWorld (== persist.h, VERBATIM) =====
    for (int tk = 0; tk < ticks; ++tk) {
        // (1) Per-body KineticEnergy (PRE-integrate) + the hysteresis quietTicks update.
        for (int e1 = 0; e1 < n; ++e1) {
            int ke = KineticEnergy(body[e1]);
            sEnergy[e1] = ke;
            if (ke > wakeThr) { sQuiet[e1] = 0u; sAsleep[e1] = 0u; }
            else if (ke < sleepThr) { if (sQuiet[e1] != 0xFFFFFFFFu) sQuiet[e1] += 1u; }
            // else: in the band -> hold quietTicks (no-flicker).
        }

        // (2) The contact adjacency (all-pairs BoxSatStable overlap, FIXED i<j) + island wakefulness propagation.
        bool adj[HF_MAX_BODIES][HF_MAX_BODIES];
        for (int ai = 0; ai < n; ++ai) for (int aj = 0; aj < n; ++aj) adj[ai][aj] = false;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (body[i].invMass == 0 && body[j].invMass == 0) continue;  // static-static
                Sat sat = BoxSatStable(body[i], half[i], body[j], half[j]);
                if (sat.overlap) { adj[i][j] = true; adj[j][i] = true; }
            }
        }
        // Seed: a dynamic body is awake iff its own quietTicks < sleepDelay. Statics are inert (never awake).
        bool awake[HF_MAX_BODIES];
        for (int wi = 0; wi < n; ++wi)
            awake[wi] = (IsDyn(body[wi]) && sQuiet[wi] < sleepDelay);
        // Propagate wakefulness to a fixed point — a fixed bound of n passes in FIXED body order.
        for (int pass = 0; pass < n; ++pass) {
            bool changed = false;
            for (int i = 0; i < n; ++i) {
                if (!IsDyn(body[i]) || awake[i]) continue;
                for (int j = 0; j < n; ++j) {
                    if (adj[i][j] && awake[j]) { awake[i] = true; changed = true; break; }
                }
            }
            if (!changed) break;
        }
        // The final per-body asleep flag: dynamic -> asleep iff NOT awake; static -> always inert (asleep).
        for (int fi2 = 0; fi2 < n; ++fi2)
            sAsleep[fi2] = IsDyn(body[fi2]) ? (awake[fi2] ? 0u : 1u) : 1u;

        // (2b) FREEZE every asleep dynamic body: zero vel + angVel (exactly zero residual).
        for (int fz = 0; fz < n; ++fz) {
            if (IsDyn(body[fz]) && sAsleep[fz] != 0u) {
                body[fz].vx = 0; body[fz].vy = 0; body[fz].vz = 0;
                body[fz].ax = 0; body[fz].ay = 0; body[fz].az = 0;
            }
        }

        // (3) Predict-integrate ONLY AWAKE dynamic bodies + per-tick damping. Asleep bodies stay exactly put.
        for (int i1 = 0; i1 < n; ++i1) {
            if (IsDyn(body[i1]) && sAsleep[i1] == 0u) {
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

        // (= PS3 step 2) world inverse inertias.
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

        ContactKey nextKey[HF_MAX_CACHE];
        int        nextJn[HF_MAX_CACHE];
        int        nextJt1[HF_MAX_CACHE];
        int        nextJt2[HF_MAX_CACHE];
        int        nextCount = 0;

        // (5a) Carry over the cache entries of pairs NOT active this tick (sleeping/inactive) — their
        // warm-start persists untouched. A pair (a,b) is active iff at least one is an awake dynamic body.
        for (int ce0 = 0; ce0 < cacheCount; ++ce0) {
            uint a = cacheKey[ce0].bodyA, b = cacheKey[ce0].bodyB;
            bool activeA = (a < (uint)n) && IsDyn(body[a]) && sAsleep[a] == 0u;
            bool activeB = (b < (uint)n) && IsDyn(body[b]) && sAsleep[b] == 0u;
            bool wasActive = activeA || activeB;
            if (!wasActive && nextCount < HF_MAX_CACHE) {
                nextKey[nextCount] = cacheKey[ce0];
                nextJn[nextCount]  = cacheJn[ce0];
                nextJt1[nextCount] = cacheJt1[ce0];
                nextJt2[nextCount] = cacheJt2[ce0];
                ++nextCount;
            }
        }

        // (4a) Impulse solve over the ACTIVE pairs only — per pair BuildKeyedManifold -> MatchCache (seed from
        // last tick) -> SolveFrictionWarm. FIXED i<j order, in place.
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (body[i].invMass == 0 && body[j].invMass == 0) continue;
                bool ai = IsDyn(body[i]) && sAsleep[i] == 0u;
                bool aj = IsDyn(body[j]) && sAsleep[j] == 0u;
                if (!(ai || aj)) continue;   // fully-asleep / static-only pair -> skip
                Keyed kd = BuildKeyedManifold((uint)i, (uint)j, body[i], half[i], body[j], half[j]);
                if (kd.count == 0u) continue;
                for (uint pi = 0u; pi < kd.count; ++pi) {
                    for (int e = 0; e < cacheCount; ++e) {
                        if (ContactKeysEqual(cacheKey[e], kd.keys[pi])) {
                            kd.jn[pi] = cacheJn[e]; kd.jt1[pi] = cacheJt1[e]; kd.jt2[pi] = cacheJt2[e];
                            break;
                        }
                    }
                }
                int iaW[9]; int ibW[9];
                [unroll] for (int z = 0; z < 9; ++z) { iaW[z] = invIW[i][z]; ibW[z] = invIW[j][z]; }
                FxBody bi = body[i]; FxBody bj = body[j];
                SolveFrictionWarm(bi, bj, iaW, ibW, kd, restitution, mu, solveIters);
                body[i] = bi; body[j] = bj;
                for (uint pk = 0u; pk < kd.count; ++pk) {
                    if (nextCount < HF_MAX_CACHE) {
                        nextKey[nextCount] = kd.keys[pk];
                        nextJn[nextCount]  = kd.jn[pk];
                        nextJt1[nextCount] = kd.jt1[pk];
                        nextJt2[nextCount] = kd.jt2[pk];
                        ++nextCount;
                    }
                }
            }
        }

        // (4b) position de-penetration over the ACTIVE pairs only — posIters sweeps, FIXED i<j order, asleep
        // bodies are NOT pushed (their correction share is zeroed so only the awake partner moves).
        for (int pit = 0; pit < posIters; ++pit) {
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    int invSum = body[i].invMass + body[j].invMass;
                    if (invSum == 0) continue;
                    bool ai = IsDyn(body[i]) && sAsleep[i] == 0u;
                    bool aj = IsDyn(body[j]) && sAsleep[j] == 0u;
                    if (!(ai || aj)) continue;   // fully-asleep / static-only -> skip
                    Sat sat = BoxSatStable(body[i], half[i], body[j], half[j]);
                    if (!sat.overlap) continue;
                    int3 nrm = sat.axis;
                    int3 ab = int3(body[j].px - body[i].px, body[j].py - body[i].py, body[j].pz - body[i].pz);
                    if (FxDot(nrm, ab) < 0) nrm = int3(-nrm.x, -nrm.y, -nrm.z);
                    int excess = sat.penetration - slop;
                    if (excess <= 0) continue;
                    int corrected = fxmul(excess, beta);
                    bool moveI = ai;
                    bool moveJ = aj;
                    int wi = fxdiv(body[i].invMass, invSum);
                    int wj = HF_FPX_ONE - wi;
                    if (!moveI && moveJ) { wi = 0; wj = HF_FPX_ONE; }
                    else if (moveI && !moveJ) { wi = HF_FPX_ONE; wj = 0; }
                    else if (!moveI && !moveJ) continue;
                    int3 ci = int3(fxmul(nrm.x, fxmul(corrected, wi)), fxmul(nrm.y, fxmul(corrected, wi)),
                                   fxmul(nrm.z, fxmul(corrected, wi)));
                    int3 cj = int3(fxmul(nrm.x, fxmul(corrected, wj)), fxmul(nrm.y, fxmul(corrected, wj)),
                                   fxmul(nrm.z, fxmul(corrected, wj)));
                    if (moveI) { body[i].px -= ci.x; body[i].py -= ci.y; body[i].pz -= ci.z; }
                    if (moveJ) { body[j].px += cj.x; body[j].py += cj.y; body[j].pz += cj.z; }
                }
            }
        }

        // (5b) swap the rebuilt cache in for next tick.
        for (int ce = 0; ce < nextCount; ++ce) {
            cacheKey[ce] = nextKey[ce]; cacheJn[ce] = nextJn[ce];
            cacheJt1[ce] = nextJt1[ce]; cacheJt2[ce] = nextJt2[ce];
        }
        cacheCount = nextCount;
    }

    for (int wi = 0; wi < n; ++wi) {
        gBodies[wi] = body[wi];
        SleepGpu sg; sg.energy = sEnergy[wi]; sg.quietTicks = sQuiet[wi]; sg.asleep = sAsleep[wi]; sg._pad = 0u;
        gSleep[wi] = sg;
    }
}
