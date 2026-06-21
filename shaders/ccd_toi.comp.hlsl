// Slice CD1 — Deterministic Integer CCD: THE TIME-OF-IMPACT PRIMITIVE compute pass (conservative advancement).
// ONE THREAD PER CANDIDATE PAIR (per-pair INDEPENDENT — each thread reads its pair's two hulls + moving bodies
// + dt and writes its OWN FxToi slot; race-free, NO atomics) runs the full conservative-advancement TOI loop,
// copying engine/sim/ccd.h::ConservativeAdvance's body VERBATIM (the SAME embedded gjk::Gjk simplex evolution,
// the SAME BodyMaxRadius / ClosingSpeedBound, the SAME fxdiv advance step + IntegrateBodyFull sub-step with
// ZERO gravity, the SAME kContactEps / kToiMaxIter termination) so the GPU FxToi is byte-identical to the CPU
// ccd::ConservativeAdvance -> the host GPU==CPU memcmp catches any divergence.
//
// INTEGER WIDTH (the determinism crux, the GJ2/CX1/FPX3 lesson): fxmul/fxdiv + FxDot/FxCross/FxLength(FxISqrt)/
// FxNormalize/FxRotate/FxQuatMul/FxQuatNormalize all use int64_t. DXC -spirv compiles int64 (the Int64
// capability, the gjk_distance.comp/fpx_orient.comp pattern); glslc (the Metal HLSL->SPIR-V->MSL frontend)
// CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the
// Metal hf_gen_msl list). The Metal --ccd-toi showcase runs the CPU ccd::ConservativeAdvance over the same
// pairs -> byte-identical to this GPU result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU proof.
//
// pairEnabled=0 -> write a cleared FxToi for every pair (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..3):
//   b0 gHulls   : the FxHull array (kMaxHullVerts x int3 verts + uint count per hull), READ.
//   b1 gPairs   : the CcdPair array (hullIndexA, bodyA, hullIndexB, bodyB, dt) per pair, READ.
//   b2 gOut     : the FxToiGpu array per pair, WRITE.
//   b3 gParams  : { pairCount, pairEnabled, _, _ }, READ.

#define HF_FPX_FRAC 16          // MUST match fpx.h::kFrac
#define HF_GJK_MAX_VERTS 20     // MUST match gjk.h::kMaxHullVerts
#define HF_GJK_MAX_ITER 32      // MUST match gjk.h::kGjkMaxIter
#define HF_GJK_ONE (1 << HF_FPX_FRAC)
#define HF_GJK_EDGE_EPS (HF_GJK_ONE / 256)   // MUST match convex.h::kEdgeEps
#define HF_CCD_CONTACT_EPS (HF_GJK_ONE / 64) // MUST match ccd.h::kContactEps
#define HF_CCD_MAX_ITER 32                   // MUST match ccd.h::kToiMaxIter

// std430 FxHull mirror (engine/sim/gjk.h::FxHull): flat int arrays (vx,vy,vz) + count (matches gjk_distance).
struct FxHull {
    int  vx[HF_GJK_MAX_VERTS];
    int  vy[HF_GJK_MAX_VERTS];
    int  vz[HF_GJK_MAX_VERTS];
    uint count;
};

// std430 FxBody mirror (engine/sim/fpx.h::FxBody): 16 x int32 (64 bytes). CCD reads pos/vel/orient/angVel/flags.
struct FxBody {
    int  px, py, pz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
    int  radius;
    int  ox, oy, oz, ow;
    int  ax, ay, az;
};

// std430 CcdPair mirror: hull index A + body A + hull index B + body B + dt (+pad to 16-align).
struct CcdPair {
    uint   hullIndexA;
    uint   _pad0, _pad1, _pad2;
    FxBody bodyA;
    uint   hullIndexB;
    uint   _pad3, _pad4, _pad5;
    FxBody bodyB;
    int    dt;
    int    _pad6, _pad7, _pad8;
};

// std430 FxToi mirror (engine/sim/ccd.h::FxToi): { int toi; uint hit; uint iterations; } (+pad to 16).
struct FxToiGpu {
    int  toi;
    uint hit;
    uint iterations;
    uint _pad0;
};

struct Params {
    int4 cfg;   // x=pairCount, y=pairEnabled, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxHull>   gHulls  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<CcdPair>  gPairs  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FxToiGpu> gOut    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<Params>   gParams : register(u3);

// ===== VERBATIM Q16.16 helpers (the int64 ops the CPU gjk.h / convex.h / fpx.h use) ======================
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
// floor(sqrt(v)) for v>=0 (int64 binary digit-by-digit — VERBATIM fpx.h::FxISqrt).
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
    if (len == 0) return int3(0, HF_GJK_ONE, 0);
    return int3(fxdiv(v.x, len), fxdiv(v.y, len), fxdiv(v.z, len));
}
// VERBATIM fpx.h::FxRotate — rotate v by the unit quaternion q (q.xyzw).
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
// VERBATIM fpx.h::FxQuatMul — the Hamilton product a*b (q = xyzw).
int4 FxQuatMul(int4 a, int4 b) {
    int4 r;
    r.w = fxmul(a.w, b.w) - fxmul(a.x, b.x) - fxmul(a.y, b.y) - fxmul(a.z, b.z);
    r.x = fxmul(a.w, b.x) + fxmul(a.x, b.w) + fxmul(a.y, b.z) - fxmul(a.z, b.y);
    r.y = fxmul(a.w, b.y) - fxmul(a.x, b.z) + fxmul(a.y, b.w) + fxmul(a.z, b.x);
    r.z = fxmul(a.w, b.z) + fxmul(a.x, b.y) - fxmul(a.y, b.x) + fxmul(a.z, b.w);
    return r;
}
// VERBATIM fpx.h::FxQuatNormalize.
int4 FxQuatNormalize(int4 q) {
    int len = FxLength4(q);
    if (len == 0) return int4(0, 0, 0, HF_GJK_ONE);
    return int4(fxdiv(q.x, len), fxdiv(q.y, len), fxdiv(q.z, len), fxdiv(q.w, len));
}

// ===== Support (VERBATIM gjk.h::SupportLocal / Support / SupportMinkowski) ================================
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

// ===== The Johnson sub-distance (VERBATIM gjk.h::DoSimplex2/3/4 — copied from gjk_distance.comp) ==========
struct Sub { int3 dir; uint keep[4]; int w[4]; uint size; bool containsOrigin; };

Sub DoSimplex2(int3 a, int3 b) {
    Sub r; r.containsOrigin = false; r.dir = int3(0,0,0);
    r.keep[0]=0;r.keep[1]=0;r.keep[2]=0;r.keep[3]=0; r.w[0]=0;r.w[1]=0;r.w[2]=0;r.w[3]=0; r.size=0;
    int3 ab = int3(b.x-a.x, b.y-a.y, b.z-a.z);
    int3 ao = FxNeg(a);
    int abLen = FxLength(ab);
    if (abLen < HF_GJK_EDGE_EPS) { r.size=1; r.keep[0]=0; r.w[0]=HF_GJK_ONE; r.dir=ao; return r; }
    int d1 = FxDot(ao, ab);
    if (d1 <= 0) { r.size=1; r.keep[0]=0; r.w[0]=HF_GJK_ONE; r.dir=ao; return r; }
    int d2 = FxDot(ab, ab);
    if (d1 >= d2) { r.size=1; r.keep[0]=1; r.w[0]=HF_GJK_ONE; r.dir=FxNeg(b); return r; }
    int t = fxdiv(d1, d2);
    int3 closest = int3(a.x + fxmul(ab.x,t), a.y + fxmul(ab.y,t), a.z + fxmul(ab.z,t));
    r.size=2; r.keep[0]=0; r.keep[1]=1; r.w[0]=HF_GJK_ONE - t; r.w[1]=t; r.dir=FxNeg(closest);
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
    if (d1 <= 0 && d2 <= 0) { r.size=1; r.keep[0]=0; r.w[0]=HF_GJK_ONE; r.dir=ao; return r; }
    int3 bo = FxNeg(b);
    int d3 = FxDot(ab, bo);
    int d4 = FxDot(ac, bo);
    if (d3 >= 0 && d4 <= d3) { r.size=1; r.keep[0]=1; r.w[0]=HF_GJK_ONE; r.dir=bo; return r; }
    int vc = fxmul(d1, d4) - fxmul(d3, d2);
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        int denom = d1 - d3;
        int t = (denom > HF_GJK_EDGE_EPS || denom < -HF_GJK_EDGE_EPS) ? fxdiv(d1, denom) : 0;
        int3 closest = int3(a.x + fxmul(ab.x,t), a.y + fxmul(ab.y,t), a.z + fxmul(ab.z,t));
        r.size=2; r.keep[0]=0; r.keep[1]=1; r.w[0]=HF_GJK_ONE - t; r.w[1]=t; r.dir=FxNeg(closest); return r;
    }
    int3 co = FxNeg(c);
    int d5 = FxDot(ab, co);
    int d6 = FxDot(ac, co);
    if (d6 >= 0 && d5 <= d6) { r.size=1; r.keep[0]=2; r.w[0]=HF_GJK_ONE; r.dir=co; return r; }
    int vb = fxmul(d5, d2) - fxmul(d1, d6);
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        int denom = d2 - d6;
        int t = (denom > HF_GJK_EDGE_EPS || denom < -HF_GJK_EDGE_EPS) ? fxdiv(d2, denom) : 0;
        int3 closest = int3(a.x + fxmul(ac.x,t), a.y + fxmul(ac.y,t), a.z + fxmul(ac.z,t));
        r.size=2; r.keep[0]=0; r.keep[1]=2; r.w[0]=HF_GJK_ONE - t; r.w[1]=t; r.dir=FxNeg(closest); return r;
    }
    int va = fxmul(d3, d6) - fxmul(d5, d4);
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        int denom = (d4 - d3) + (d5 - d6);
        int t = (denom > HF_GJK_EDGE_EPS || denom < -HF_GJK_EDGE_EPS) ? fxdiv(d4 - d3, denom) : 0;
        int3 bc = int3(c.x-b.x, c.y-b.y, c.z-b.z);
        int3 closest = int3(b.x + fxmul(bc.x,t), b.y + fxmul(bc.y,t), b.z + fxmul(bc.z,t));
        r.size=2; r.keep[0]=1; r.keep[1]=2; r.w[0]=HF_GJK_ONE - t; r.w[1]=t; r.dir=FxNeg(closest); return r;
    }
    int denom = va + vb + vc;
    int u, v, w;
    if (denom > HF_GJK_EDGE_EPS || denom < -HF_GJK_EDGE_EPS) {
        v = fxdiv(vb, denom);
        w = fxdiv(vc, denom);
        u = HF_GJK_ONE - v - w;
        r.w[0]=u; r.w[1]=v; r.w[2]=w;
    } else {
        r.w[0]=HF_GJK_ONE; r.w[1]=0; r.w[2]=0;
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

// ===== Gjk (VERBATIM gjk.h::Gjk — copied from gjk_distance.comp; CCD needs only overlap + separation) =====
// Returns overlap (out param) + the separation vector (origin -> closest CSO point); FxLength(separation) is
// the gap. The witness/simplex recovery the full GjkResult carries is NOT needed by CCD, so this trimmed
// variant returns just the two fields the conservative-advance loop reads — but the SIMPLEX EVOLUTION is
// byte-identical to gjk_distance.comp / gjk.h::Gjk.
int3 GjkSep(FxHull hA, int4 oA, int3 pA, FxHull hB, int4 oB, int3 pB, out bool overlapOut) {
    int3 dir = int3(pB.x - pA.x, pB.y - pA.y, pB.z - pA.z);
    if (dir.x == 0 && dir.y == 0 && dir.z == 0) dir = int3(HF_GJK_ONE, 0, 0);

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
        wts[0] = HF_GJK_ONE;
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

    // The separation = the closest CSO point = the barycentric combine of the surviving CSO points (VERBATIM
    // gjk.h::Gjk tail; the witness points are not needed by CCD).
    if (!overlap) {
        if (keepCount == 0u) { keepCount = (n > 0u ? 1u : 0u); if (keepCount == 1u) wts[0] = HF_GJK_ONE; }
    }
    uint kc = (keepCount == 0u ? n : keepCount);
    if (kc > 4u) kc = 4u;
    int wsum = 0; for (uint i = 0u; i < kc; ++i) wsum += wts[i];
    if (wsum == 0 && kc > 0u) { wts[0] = HF_GJK_ONE; }
    int3 closestCso = int3(0, 0, 0);
    for (uint i6 = 0u; i6 < kc; ++i6) {
        closestCso = int3(closestCso.x + fxmul(sp[i6].x, wts[i6]),
                          closestCso.y + fxmul(sp[i6].y, wts[i6]),
                          closestCso.z + fxmul(sp[i6].z, wts[i6]));
    }
    overlapOut = overlap;
    return closestCso;
}

// ===== IntegrateBodyFull (VERBATIM fpx.h::IntegrateBodyFull + IntegrateOrientation) ======================
// Advances pos from vel (dynamic only) + orient from angVel by `dt`. ZERO gravity passed (gx=gy=gz=0). The CCD
// loop integrates COPIES of both bodies forward by `advance`. pos/vel/orient/angVel/flags are mutated in place.
void IntegrateBodyFull(inout int3 pos, inout int3 vel, inout int4 orient, int3 angVel, uint flags, int dt) {
    if ((flags & 1u) != 0u) {
        // ZERO gravity: vel += 0. pos += vel*dt.
        pos.x += fxmul(vel.x, dt);
        pos.y += fxmul(vel.y, dt);
        pos.z += fxmul(vel.z, dt);
    }
    // IntegrateOrientation: dq = omega (x) orient; orient += 0.5*dq*dt; renormalize.
    int4 omega = int4(angVel.x, angVel.y, angVel.z, 0);
    int4 dq = FxQuatMul(omega, orient);
    const int kHalf = HF_GJK_ONE / 2;
    orient.x += fxmul(fxmul(dq.x, kHalf), dt);
    orient.y += fxmul(fxmul(dq.y, kHalf), dt);
    orient.z += fxmul(fxmul(dq.z, kHalf), dt);
    orient.w += fxmul(fxmul(dq.w, kHalf), dt);
    orient = FxQuatNormalize(orient);
}

// ===== BodyMaxRadius (VERBATIM ccd.h::BodyMaxRadius) =====================================================
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

// ===== ClosingSpeedBound (VERBATIM ccd.h::ClosingSpeedBound) =============================================
int ClosingSpeedBound(int3 velA, int3 angVelA, int rMaxA,
                      int3 velB, int3 angVelB, int rMaxB, int3 n) {
    int3 relVel = int3(velB.x - velA.x, velB.y - velA.y, velB.z - velA.z);
    int linClose = FxDot(relVel, n);
    int angA = fxmul(FxLength(angVelA), rMaxA);
    int angB = fxmul(FxLength(angVelB), rMaxB);
    return linClose + angA + angB;
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int pairCount   = gParams[0].cfg.x;
    int pairEnabled = gParams[0].cfg.y;
    int idx = (int)gid.x;
    if (idx >= pairCount) return;

    FxToiGpu zero; zero.toi = 0; zero.hit = 0u; zero.iterations = 0u; zero._pad0 = 0u;
    if (pairEnabled == 0) { gOut[idx] = zero; return; }

    CcdPair pr = gPairs[idx];
    FxHull hA = gHulls[pr.hullIndexA];
    FxHull hB = gHulls[pr.hullIndexB];

    // Working COPIES of both bodies (the conservative-advance loop integrates these forward).
    int3 posA = int3(pr.bodyA.px, pr.bodyA.py, pr.bodyA.pz);
    int3 velA = int3(pr.bodyA.vx, pr.bodyA.vy, pr.bodyA.vz);
    int4 oriA = int4(pr.bodyA.ox, pr.bodyA.oy, pr.bodyA.oz, pr.bodyA.ow);
    int3 angA = int3(pr.bodyA.ax, pr.bodyA.ay, pr.bodyA.az);
    uint flagsA = pr.bodyA.flags;
    int3 posB = int3(pr.bodyB.px, pr.bodyB.py, pr.bodyB.pz);
    int3 velB = int3(pr.bodyB.vx, pr.bodyB.vy, pr.bodyB.vz);
    int4 oriB = int4(pr.bodyB.ox, pr.bodyB.oy, pr.bodyB.oz, pr.bodyB.ow);
    int3 angB = int3(pr.bodyB.ax, pr.bodyB.ay, pr.bodyB.az);
    uint flagsB = pr.bodyB.flags;
    int dt = pr.dt;

    // The body bounding radii (pose-invariant, computed once from the START pose).
    int rMaxA = BodyMaxRadius(hA, oriA, posA);
    int rMaxB = BodyMaxRadius(hB, oriB, posB);

    FxToiGpu outToi = zero;
    int t = 0;
    bool done = false;
    for (uint iter = 0u; iter < (uint)HF_CCD_MAX_ITER && !done; ++iter) {
        outToi.iterations = iter;
        bool overlap;
        int3 sep = GjkSep(hA, oriA, posA, hB, oriB, posB, overlap);
        if (overlap) { outToi.toi = t; outToi.hit = 1u; done = true; break; }
        int gap = FxLength(sep);
        if (gap <= HF_CCD_CONTACT_EPS) { outToi.toi = t; outToi.hit = 1u; done = true; break; }
        int3 n = FxNormalize(sep);
        int bound = ClosingSpeedBound(velA, angA, rMaxA, velB, angB, rMaxB, n);
        if (bound <= 0) { outToi.toi = dt; outToi.hit = 0u; done = true; break; }
        int advance = fxdiv(gap, bound);
        t += advance;
        if (t >= dt) { outToi.toi = dt; outToi.hit = 0u; done = true; break; }
        IntegrateBodyFull(posA, velA, oriA, angA, flagsA, advance);
        IntegrateBodyFull(posB, velB, oriB, angB, flagsB, advance);
    }
    if (!done) { outToi.toi = t; outToi.hit = 0u; outToi.iterations = (uint)HF_CCD_MAX_ITER; }

    gOut[idx] = outToi;
}
