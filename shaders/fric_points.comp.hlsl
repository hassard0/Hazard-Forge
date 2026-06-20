// Slice FC2 — Deterministic Contact Friction THE FRICTION-AUGMENTED MANIFOLD POINT compute pass (the 2nd
// slice of FLAGSHIP #20: DETERMINISTIC TANGENTIAL CONTACT FRICTION, hf::sim::fric). ONE THREAD PER BOX PAIR
// (per-pair INDEPENDENT — each thread reads its pair's two bodies+boxes, runs BoxSatStable (the CX4
// face-preference SAT) then BuildManifold (the CX2 contact manifold), pairs every contact point with the FC1
// tangent basis MakeTangentBasis(nAB) + zeroed impulse accumulators, and writes its OWN FrictionManifoldGpu
// slot; race-free, NO atomics). The SAT + manifold + basis logic is copied VERBATIM from
// engine/sim/convex.h::BoxSat/BoxSatStable + BuildManifold + ClosestPointsOnSegments and
// engine/sim/fric.h::MakeTangentBasis (the SAME FxRotate box-axes, FxNormalize/FxDot/FxCross int64 ops, the
// SAME fixed 15-axis order + edge-cross skip + strict-< min-pen tie-break + the kFacePrefEps face-preference
// post-filter, the SAME reference/incident face pick + clip order + reduction, the SAME LeastAlignedAxis
// argmin + project-off-n -> FxNormalize -> FxCross(n,t1)) so the GPU FrictionManifoldGpu[] is byte-identical
// to the CPU fric::BuildFrictionPoints -> the host GPU==CPU memcmp catches any divergence.
//
// INTEGER WIDTH (the determinism crux, the FPX3/CX1 lesson): fxmul/fxdiv/FxISqrt + the FxDot/FxCross Q16.16
// products use int64_t. DXC -spirv compiles int64 (the Int64 capability, the convex_manifold.comp pattern);
// glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is
// VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal hf_gen_msl list). The Metal --fric-points
// showcase runs the CPU fric::BuildFrictionPoints over the same pairs -> byte-identical to this GPU result BY
// CONSTRUCTION, while the Vulkan side carries the GPU==CPU bit-identity proof.
//
// pointsEnabled=0 -> write a cleared FrictionManifoldGpu (count=0) for every pair (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..2):
//   b0 gPairs     : the box-pair array (FxBody A + FxBox A + FxBody B + FxBox B per pair), READ.
//   b1 gManifolds : the FrictionManifoldGpu array (count + 4 FrictionPoint) per pair, WRITE.
//   b2 gParams    : { pairCount, pointsEnabled, _, _ }, READ.

#define HF_FPX_FRAC 16   // MUST match fpx.h::kFrac
#define HF_FPX_ONE  (1 << HF_FPX_FRAC)
#define HF_FPX_HALF (HF_FPX_ONE / 2)           // kHalf
#define HF_CX_EDGE_EPS (HF_FPX_ONE / 256)      // MUST match convex.h::kEdgeEps == kOne/256
#define HF_CX_FACE_PREF_EPS (HF_FPX_ONE / 64)  // MUST match convex.h::kFacePrefEps == kOne/64

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

// std430 SatPair mirror (engine/sim/convex.h::SatPair): FxBody A (16) + FxBox A (3) + FxBody B (16) +
// FxBox B (3) = 38 x int32 (152 bytes).
struct SatPair {
    FxBody bodyA;
    int    hAx, hAy, hAz;
    FxBody bodyB;
    int    hBx, hBy, hBz;
};

// std430 FrictionPoint mirror (engine/sim/fric.h::FrictionPoint, the host packs it into THIS 15 x int32 /
// 60-byte form for the memcmp): point.xyz, normal.xyz, t1.xyz, t2.xyz, normalImpulse, tangentImpulse1,
// tangentImpulse2 (all Q16.16).
struct FricPointGpu {
    int px, py, pz;
    int nx, ny, nz;
    int t1x, t1y, t1z;
    int t2x, t2y, t2z;
    int ni, ti1, ti2;
};

// std430 FrictionManifoldGpu mirror (engine/sim/fric.h::FrictionManifold): count (uint32) + 4 FricPointGpu
// = 1 + 60 = 61 x int32 (244 bytes).
struct FricManifoldGpu {
    uint count;
    FricPointGpu pts[4];
};

struct FricPointsParams {
    int4 cfg;   // x=pairCount, y=pointsEnabled, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<SatPair>         gPairs     : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FricManifoldGpu> gManifolds : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FricPointsParams> gParams   : register(u2);

// ===== VERBATIM Q16.16 toolbox (== convex_manifold.comp / convex.h / fric.h) =====
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
// FxAt(v, i): component accessor (== convex.h::FxAt).
int FxAt(int3 v, uint i) { return (i == 0) ? v.x : ((i == 1) ? v.y : v.z); }

// ===== VERBATIM fric.h::MakeTangentBasis (the FC1 Gram-Schmidt: least-aligned cardinal axis -> project off
// n -> FxNormalize -> FxCross(n, t1)) =====
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
    int eDotN = FxDot(e, n);                       // the n-component of e
    int3 r = int3(e.x - fxmul(eDotN, n.x),
                  e.y - fxmul(eDotN, n.y),
                  e.z - fxmul(eDotN, n.z));        // project e off n
    t1 = FxNormalize(r);                           // unit tangent (int64 FxISqrt/fxdiv)
    t2 = FxCross(n, t1);                           // already unit (n, t1 orthonormal)
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int pairCount     = gParams[0].cfg.x;
    int pointsEnabled = gParams[0].cfg.y;
    int idx = (int)gid.x;
    if (idx >= pairCount) return;

    // Cleared FrictionManifoldGpu (count=0, every field zero) — the disabled no-op + the separated default.
    FricManifoldGpu mz;
    mz.count = 0u;
    {
        FricPointGpu z;
        z.px = 0; z.py = 0; z.pz = 0; z.nx = 0; z.ny = 0; z.nz = 0;
        z.t1x = 0; z.t1y = 0; z.t1z = 0; z.t2x = 0; z.t2y = 0; z.t2z = 0;
        z.ni = 0; z.ti1 = 0; z.ti2 = 0;
        mz.pts[0] = z; mz.pts[1] = z; mz.pts[2] = z; mz.pts[3] = z;
    }

    if (pointsEnabled == 0) { gManifolds[idx] = mz; return; }

    SatPair p = gPairs[idx];

    // ---- The 3 world face axes of each box (VERBATIM BoxAxes) ----
    int4 qA = int4(p.bodyA.ox, p.bodyA.oy, p.bodyA.oz, p.bodyA.ow);
    int4 qB = int4(p.bodyB.ox, p.bodyB.oy, p.bodyB.oz, p.bodyB.ow);
    int3 axA0 = FxRotate(qA, int3(HF_FPX_ONE, 0, 0));
    int3 axA1 = FxRotate(qA, int3(0, HF_FPX_ONE, 0));
    int3 axA2 = FxRotate(qA, int3(0, 0, HF_FPX_ONE));
    int3 axB0 = FxRotate(qB, int3(HF_FPX_ONE, 0, 0));
    int3 axB1 = FxRotate(qB, int3(0, HF_FPX_ONE, 0));
    int3 axB2 = FxRotate(qB, int3(0, 0, HF_FPX_ONE));
    int3 axA[3] = { axA0, axA1, axA2 };
    int3 axB[3] = { axB0, axB1, axB2 };
    int3 hA = int3(p.hAx, p.hAy, p.hAz);
    int3 hB = int3(p.hBx, p.hBy, p.hBz);
    int3 posA = int3(p.bodyA.px, p.bodyA.py, p.bodyA.pz);
    int3 posB = int3(p.bodyB.px, p.bodyB.py, p.bodyB.pz);
    int3 t  = int3(posB.x - posA.x, posB.y - posA.y, posB.z - posA.z);

    // ---- BoxSat (VERBATIM convex_sat.comp / convex.h::BoxSat) ----
    bool found = false;
    int  minPen = 0;
    uint minIndex = 0u;
    int3 minAxis = int3(0, 0, 0);
    bool separated = false;
    for (int phase = 0; phase < 15 && !separated; ++phase) {
        int3 rawL;
        bool skipDeg = false;
        if (phase < 3) { rawL = axA[phase]; }
        else if (phase < 6) { rawL = axB[phase - 3]; }
        else {
            int e = phase - 6;
            int i = e / 3, j = e % 3;
            rawL = FxCross(axA[i], axB[j]);
            skipDeg = true;
        }
        if (skipDeg) {
            int rawLen = FxLength(rawL);
            if (rawLen < HF_CX_EDGE_EPS) continue;
        }
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

    if (separated) { gManifolds[idx] = mz; return; }   // overlap=false -> empty manifold

    // ===== BoxSatStable face-preference post-filter (VERBATIM convex.h::BoxSatStable) =====
    // If BoxSat picked an EDGE axis (minIndex >= 6) but a FACE axis penetrates within kFacePrefEps, prefer
    // the (clean, drift-free) FACE result. Re-run ONLY the 6 face axes in the FIXED order (A's 3 then B's 3,
    // lowest index wins ties).
    if (minIndex >= 6u) {
        bool ffound = false;
        int  fMinPen = 0;
        uint fMinIndex = 0u;
        int3 fMinAxis = int3(0, 0, 0);
        for (int fp = 0; fp < 6; ++fp) {
            int3 rawL = (fp < 3) ? axA[fp] : axB[fp - 3];
            int3 L = FxNormalize(rawL);
            int rA = ProjectedRadius(L, axA0, axA1, axA2, hA);
            int rB = ProjectedRadius(L, axB0, axB1, axB2, hB);
            int dotLt = FxDot(L, t);
            int s = absI(dotLt);
            int sum = rA + rB;
            if (s > sum) continue;                 // (shouldn't happen for an overlapping pair, but be safe)
            int pen = sum - s;
            if (!ffound || pen < fMinPen) {
                ffound = true; fMinPen = pen; fMinIndex = (uint)fp;
                fMinAxis = (dotLt < 0) ? int3(-L.x, -L.y, -L.z) : L;
            }
        }
        if (ffound && fMinPen <= minPen + HF_CX_FACE_PREF_EPS) {
            minPen = fMinPen; minIndex = fMinIndex; minAxis = fMinAxis;
        }
    }

    // ===== BuildManifold (VERBATIM convex.h::BuildManifold) — fills mPoints[0..mCount) + mDepths + mNormal ==
    uint mCount = 0u;
    int3 mPoints[4] = { int3(0,0,0), int3(0,0,0), int3(0,0,0), int3(0,0,0) };
    int  mDepths[4] = { 0, 0, 0, 0 };
    int3 mNormal = int3(0, 0, 0);

    // ---- EDGE-EDGE case (minIndex >= 6): the single closest-point contact ----
    if (minIndex >= 6u) {
        uint ee = minIndex - 6u;
        uint i = ee / 3u, j = ee % 3u;
        int hAi = FxAt(hA, i);
        int hBj = FxAt(hB, j);
        int3 dA = int3(fxmul(axA[i].x, hAi), fxmul(axA[i].y, hAi), fxmul(axA[i].z, hAi));
        int3 dB = int3(fxmul(axB[j].x, hBj), fxmul(axB[j].y, hBj), fxmul(axB[j].z, hBj));
        // ClosestPointsOnSegments (VERBATIM)
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
        mCount = 1u;
        mPoints[0] = int3((cA.x + cB.x) / 2, (cA.y + cB.y) / 2, (cA.z + cB.z) / 2);
        mDepths[0] = minPen;
        mNormal = minAxis;
    } else {
        // ---- FACE case (minIndex 0..5): reference/incident face clip ----
        bool refIsA = (minIndex < 3u);
        uint refIdx = refIsA ? minIndex : (minIndex - 3u);
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

        // Incident face: most anti-parallel signed axis (min FxDot), tie lowest index then + before -.
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

        // Sutherland-Hodgman against the 4 side planes in the FIXED order (+u,-u,+v,-v).
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

        // Keep penetrating clipped vertices (d = FxDot(n, faceCenter - vertex) >= 0), in clip order.
        int3 candPts[8];
        int  candDepth[8];
        int  candN = 0;
        for (int k = 0; k < polyN; ++k) {
            int3 rel = int3(faceCenter.x - poly[k].x, faceCenter.y - poly[k].y, faceCenter.z - poly[k].z);
            int d = FxDot(n, rel);
            if (d >= 0) { candPts[candN] = poly[k]; candDepth[candN] = d; candN++; }
        }

        mNormal = n;
        if (candN == 0) {
            mCount = 1u;
            int3 fb = (polyN > 0) ? poly[0] : incFaceCenter;
            mPoints[0] = fb;
            mDepths[0] = minPen;
        } else {
            // Reduce to <=4: deepest (max depth, tie lowest clip-order index) + up to 3 more in clip order.
            int deepest = 0;
            for (int k = 1; k < candN; ++k) if (candDepth[k] > candDepth[deepest]) deepest = k;
            int3 outPts[4];
            int  outDep[4];
            outPts[0] = candPts[deepest]; outDep[0] = candDepth[deepest];
            int cnt = 1;
            for (int k = 0; k < candN && cnt < 4; ++k) {
                if (k == deepest) continue;
                outPts[cnt] = candPts[k]; outDep[cnt] = candDepth[k]; cnt++;
            }
            mCount = (uint)cnt;
            for (int k = 0; k < cnt; ++k) { mPoints[k] = outPts[k]; mDepths[k] = outDep[k]; }
        }
    }

    if (mCount == 0u) { gManifolds[idx] = mz; return; }

    // ===== FC2: the A->B normal sign-correction (ONCE) + the FC1 tangent basis + zeroed accumulators =====
    int3 nAB = mNormal;
    if (FxDot(nAB, t) < 0) nAB = int3(-nAB.x, -nAB.y, -nAB.z);   // t == posB - posA (the A->B separation)
    int3 t1, t2;
    MakeTangentBasis(nAB, t1, t2);

    FricManifoldGpu fm = mz;
    fm.count = mCount;
    for (uint i = 0u; i < mCount; ++i) {
        FricPointGpu fp;
        fp.px = mPoints[i].x; fp.py = mPoints[i].y; fp.pz = mPoints[i].z;
        fp.nx = nAB.x; fp.ny = nAB.y; fp.nz = nAB.z;
        fp.t1x = t1.x; fp.t1y = t1.y; fp.t1z = t1.z;
        fp.t2x = t2.x; fp.t2y = t2.y; fp.t2z = t2.z;
        fp.ni = 0; fp.ti1 = 0; fp.ti2 = 0;   // accumulators ZEROED at build
        fm.pts[i] = fp;
    }
    gManifolds[idx] = fm;
}
