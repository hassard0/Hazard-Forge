// Slice GJ2 — General Convex-Hull Contacts: THE GJK ALGORITHM (overlap + closest distance) compute pass.
// ONE THREAD PER HULL-PAIR (per-pair INDEPENDENT — each thread reads its pair's two hulls + bodies and writes
// its OWN GjkResult slot; race-free, NO atomics) runs the full GJK simplex evolution, copying
// engine/sim/gjk.h::Gjk's body VERBATIM (the SAME fixed initial direction, the SAME normalized-search-dir
// support queries, the SAME integer Johnson sub-distance with the SAME tie-breaks, the SAME duplicate-support
// + near-origin-overlap termination) so the GPU GjkResult is byte-identical to the CPU gjk::Gjk -> the host
// GPU==CPU memcmp catches any divergence.
//
// INTEGER WIDTH (the determinism crux, the GJ1/CX1 lesson): fxmul/fxdiv + FxDot/FxCross/FxLength(FxISqrt)/
// FxNormalize/FxRotate all use int64_t. DXC -spirv compiles int64 (the Int64 capability, the gjk_support.comp/
// convex_sat.comp pattern); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so
// this shader is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal hf_gen_msl list). The Metal
// --gjk-distance showcase runs the CPU gjk::Gjk over the same pairs -> byte-identical to this GPU result BY
// CONSTRUCTION, while the Vulkan side carries the GPU==CPU bit-identity proof.
//
// pairEnabled=0 -> write a cleared GjkResult for every pair (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..3):
//   b0 gHulls   : the FxHull array (kMaxHullVerts x int3 verts + uint count per hull), READ.
//   b1 gPairs   : the GjkPair array (hullIndexA, bodyA, hullIndexB, bodyB) per pair, READ.
//   b2 gOut     : the GjkResultGpu array per pair, WRITE.
//   b3 gParams  : { pairCount, pairEnabled, _, _ }, READ.

#define HF_FPX_FRAC 16          // MUST match fpx.h::kFrac
#define HF_GJK_MAX_VERTS 20     // MUST match gjk.h::kMaxHullVerts
#define HF_GJK_MAX_ITER 32      // MUST match gjk.h::kGjkMaxIter
#define HF_GJK_ONE (1 << HF_FPX_FRAC)
#define HF_GJK_EDGE_EPS (HF_GJK_ONE / 256)   // MUST match convex.h::kEdgeEps

// std430 FxHull mirror (engine/sim/gjk.h::FxHull): flat int arrays (vx,vy,vz) + count (matches gjk_support).
struct FxHull {
    int  vx[HF_GJK_MAX_VERTS];
    int  vy[HF_GJK_MAX_VERTS];
    int  vz[HF_GJK_MAX_VERTS];
    uint count;
};

// std430 FxBody mirror (engine/sim/fpx.h::FxBody): 16 x int32 (64 bytes). GJK reads pos + orient.
struct FxBody {
    int  px, py, pz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
    int  radius;
    int  ox, oy, oz, ow;
    int  ax, ay, az;
};

// std430 GjkPair mirror: hull index A + body A + hull index B + body B.
struct GjkPair {
    uint   hullIndexA;
    uint   _pad0, _pad1, _pad2;
    FxBody bodyA;
    uint   hullIndexB;
    uint   _pad3, _pad4, _pad5;
    FxBody bodyB;
};

// std430 Simplex mirror (engine/sim/gjk.h::Simplex): pts[4], csoA[4], csoB[4] (each int3), count.
// Stored flat as 12 int3 -> 36 ints + count (+3 pad to keep the next member 16-aligned -> we keep it tight to
// match the host std430 packing of the struct: the host Simplex is {FxVec3 pts[4]; FxVec3 csoA[4];
// FxVec3 csoB[4]; uint count;} = 12*3 + 1 = 37 ints, NO trailing pad on the host (it is memcmp'd as-is via the
// packed GjkResultGpu below, NOT this nested struct directly).

// std430 GjkResult mirror — the host packs the result into THIS fixed layout (see the showcase packer). It is
// the memcmp target: overlap, separation(3), closestA(3), closestB(3), simplex {pts 12 + csoA 12 + csoB 12 +
// count}, witnessFeature, iterations.
struct GjkResultGpu {
    uint overlap;
    int  sepx, sepy, sepz;
    int  cax, cay, caz;
    int  cbx, cby, cbz;
    int  spx[4]; int spy[4]; int spz[4];   // simplex pts
    int  sax[4]; int say[4]; int saz[4];   // simplex csoA
    int  sbx[4]; int sby[4]; int sbz[4];   // simplex csoB
    uint simplexCount;
    uint witnessFeature;
    uint iterations;
};

struct Params {
    int4 cfg;   // x=pairCount, y=pairEnabled, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxHull>       gHulls  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<GjkPair>      gPairs  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<GjkResultGpu> gOut    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<Params>       gParams : register(u3);

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
int3 FxScale(int3 v, int s) { return int3(fxmul(v.x, s), fxmul(v.y, s), fxmul(v.z, s)); }
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

// ===== The Johnson sub-distance (VERBATIM gjk.h::DoSimplex2/3/4) ==========================================
// SubResult: dir, keep[4], w[4], size, containsOrigin (packed into out params).
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
    // outside(sRef, sOrigin)
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

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int pairCount   = gParams[0].cfg.x;
    int pairEnabled = gParams[0].cfg.y;
    int idx = (int)gid.x;
    if (idx >= pairCount) return;

    GjkResultGpu zero = (GjkResultGpu)0;
    if (pairEnabled == 0) { gOut[idx] = zero; return; }

    GjkPair pr = gPairs[idx];
    FxHull hA = gHulls[pr.hullIndexA];
    FxHull hB = gHulls[pr.hullIndexB];
    int4 oA = int4(pr.bodyA.ox, pr.bodyA.oy, pr.bodyA.oz, pr.bodyA.ow);
    int3 pA = int3(pr.bodyA.px, pr.bodyA.py, pr.bodyA.pz);
    int4 oB = int4(pr.bodyB.ox, pr.bodyB.oy, pr.bodyB.oz, pr.bodyB.ow);
    int3 pB = int3(pr.bodyB.px, pr.bodyB.py, pr.bodyB.pz);

    // ----- VERBATIM gjk::Gjk -----
    int3 dir = int3(pB.x - pA.x, pB.y - pA.y, pB.z - pA.z);
    if (dir.x == 0 && dir.y == 0 && dir.z == 0) dir = int3(HF_GJK_ONE, 0, 0);

    int3 sp[4], swA[4], swB[4];
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

        for (uint i = n; i > 0u; --i) { sp[i] = sp[i-1u]; swA[i] = swA[i-1u]; swB[i] = swB[i-1u]; }
        sp[0] = a; swA[0] = wA; swB[0] = wB; ++n;

        Sub sub;
        if (n == 2u)      sub = DoSimplex2(sp[0], sp[1]);
        else if (n == 3u) sub = DoSimplex3(sp[0], sp[1], sp[2]);
        else              sub = DoSimplex4(sp[0], sp[1], sp[2], sp[3]);

        if (sub.containsOrigin) {
            overlap = true; keepCount = sub.size;
            for (uint i = 0u; i < 4u; ++i) wts[i] = sub.w[i];
            break;
        }

        int3 nsp[4], nswA[4], nswB[4]; int nw[4] = {0,0,0,0};
        for (uint i = 0u; i < sub.size; ++i) {
            uint k = sub.keep[i];
            nsp[i] = sp[k]; nswA[i] = swA[k]; nswB[i] = swB[k]; nw[i] = sub.w[i];
        }
        int3 newClosest = int3(0, 0, 0);
        for (uint i = 0u; i < sub.size; ++i) {
            sp[i] = nsp[i]; swA[i] = nswA[i]; swB[i] = nswB[i]; wts[i] = nw[i];
            newClosest = int3(newClosest.x + fxmul(nsp[i].x, nw[i]),
                              newClosest.y + fxmul(nsp[i].y, nw[i]),
                              newClosest.z + fxmul(nsp[i].z, nw[i]));
        }
        n = sub.size;
        keepCount = sub.size;
        closest = newClosest;
    }

    // Witness recovery (VERBATIM gjk::Gjk tail).
    if (!overlap) {
        if (keepCount == 0u) { keepCount = (n > 0u ? 1u : 0u); if (keepCount == 1u) wts[0] = HF_GJK_ONE; }
    }
    int3 cA = int3(0,0,0), cB = int3(0,0,0), closestCso = int3(0,0,0);
    uint kc = (keepCount == 0u ? n : keepCount);
    if (kc > 4u) kc = 4u;
    int wsum = 0; for (uint i = 0u; i < kc; ++i) wsum += wts[i];
    if (wsum == 0 && kc > 0u) { wts[0] = HF_GJK_ONE; }
    for (uint i = 0u; i < kc; ++i) {
        cA = int3(cA.x + fxmul(swA[i].x, wts[i]), cA.y + fxmul(swA[i].y, wts[i]), cA.z + fxmul(swA[i].z, wts[i]));
        cB = int3(cB.x + fxmul(swB[i].x, wts[i]), cB.y + fxmul(swB[i].y, wts[i]), cB.z + fxmul(swB[i].z, wts[i]));
        closestCso = int3(closestCso.x + fxmul(sp[i].x, wts[i]),
                          closestCso.y + fxmul(sp[i].y, wts[i]),
                          closestCso.z + fxmul(sp[i].z, wts[i]));
    }

    GjkResultGpu r = zero;
    r.overlap = overlap ? 1u : 0u;
    r.sepx = closestCso.x; r.sepy = closestCso.y; r.sepz = closestCso.z;
    r.cax = cA.x; r.cay = cA.y; r.caz = cA.z;
    r.cbx = cB.x; r.cby = cB.y; r.cbz = cB.z;
    for (uint i = 0u; i < 4u; ++i) {
        if (i < n) {
            r.spx[i] = sp[i].x;  r.spy[i] = sp[i].y;  r.spz[i] = sp[i].z;
            r.sax[i] = swA[i].x; r.say[i] = swA[i].y; r.saz[i] = swA[i].z;
            r.sbx[i] = swB[i].x; r.sby[i] = swB[i].y; r.sbz[i] = swB[i].z;
        } else {
            r.spx[i] = 0; r.spy[i] = 0; r.spz[i] = 0;
            r.sax[i] = 0; r.say[i] = 0; r.saz[i] = 0;
            r.sbx[i] = 0; r.sby[i] = 0; r.sbz[i] = 0;
        }
    }
    r.simplexCount = n;
    r.witnessFeature = n;
    r.iterations = iter;
    gOut[idx] = r;
}
