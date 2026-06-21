// Slice MF3 — Hull Narrowphase Hardening: THE MULTI-POINT MANIFOLD GPU SHADER (the int64 GPU==CPU beat, the 3rd
// slice of FLAGSHIP #25: DETERMINISTIC HULL NARROWPHASE HARDENING, hf::sim::manifold). ONE THREAD PER PAIR
// (per-pair INDEPENDENT — each thread reads its pair's two hulls + bodies and writes its OWN ManifoldGpu slot;
// race-free, NO atomics) runs the FULL narrowphase->manifold chain: gjk::Gjk -> gjk::Epa -> MF2's
// HullManifoldFromEpa (the Sutherland-Hodgman hull-face clip), copying engine/sim/manifold.h::HullContactMulti's
// call chain VERBATIM (the SAME int64 FxDot/FxCross/fxdiv/fxmul ops, the SAME fixed orders, the SAME strict-
// integer tie-breaks) so the GPU convex::ContactManifold is byte-identical to the CPU HullContactMulti -> the
// host GPU==CPU memcmp catches any divergence. The GJK body == gjk_distance.comp; the EPA body == gjk_epa.comp;
// the clip == convex_manifold.comp's idiom GENERALIZED to the MF1 polygon-face tables (BuildCanonicalFaces +
// FaceNormalWorld + SupportFace/IncidentFace + ClipFaceAgainstFace) — exactly as MF2 generalized
// convex::BuildManifold in C++.
//
// INTEGER WIDTH (the determinism crux, the GJ2/GJ3/CX2 lesson): fxmul/fxdiv/FxISqrt + the FxDot/FxCross/FxLength/
// FxNormalize/FxRotate Q16.16 products + the GJK sub-distance + the EPA face math + the SH clip all use int64_t.
// DXC -spirv compiles int64 (the Int64 capability, the gjk_epa.comp / convex_manifold.comp pattern); glslc (the
// Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY (in the
// Vulkan compile list, NOT in the Metal hf_gen_msl list). The Metal --mf3-manifold showcase runs the CPU
// manifold::HullContactMulti over the same pairs -> byte-identical to this GPU result BY CONSTRUCTION, while the
// Vulkan side carries the GPU==CPU bit-identity proof.
//
// pairEnabled=0 -> write a cleared ManifoldGpu (count=0) for every pair (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..2):
//   b0 gPairs     : the HullPair array (FxHull A + FxBody A + FxHull B + FxBody B per pair), READ.
//   b1 gManifolds : the ManifoldGpu array (count, 4 points, 4 depths, normal) per pair, WRITE.
//   b2 gParams    : { pairCount, pairEnabled, _, _ }, READ.

#define HF_FPX_FRAC 16          // MUST match fpx.h::kFrac
#define HF_FPX_ONE  (1 << HF_FPX_FRAC)
#define HF_GJK_MAX_VERTS 20     // MUST match gjk.h::kMaxHullVerts
#define HF_GJK_MAX_ITER 32      // MUST match gjk.h::kGjkMaxIter
#define HF_EPA_MAX_ITER 48      // MUST match gjk.h::kEpaMaxIter
#define HF_EPA_MAX_PV 64        // MUST match gjk.h::kMaxPolyVerts
#define HF_EPA_MAX_PF 128       // MUST match gjk.h::kMaxPolyFaces
#define HF_GJK_ONE (1 << HF_FPX_FRAC)
#define HF_GJK_EDGE_EPS (HF_GJK_ONE / 256)   // MUST match convex.h::kEdgeEps
#define HF_EPA_TOL (HF_GJK_ONE / 256)        // MUST match gjk.h::kEpaTol
#define HF_MF_MAX_FACES 8       // MUST match manifold.h::kMaxHullFaces
#define HF_MF_MAX_FACE_VERTS 4  // MUST match manifold.h::kMaxFaceVerts
#define HF_MF_MAX_CLIP_VERTS 8  // MUST match manifold.h::kMaxClipVerts

// std430 FxHull mirror (engine/sim/gjk.h::FxHull): flat int arrays (vx,vy,vz) + count (== gjk_epa FxHull).
struct FxHull {
    int  vx[HF_GJK_MAX_VERTS];
    int  vy[HF_GJK_MAX_VERTS];
    int  vz[HF_GJK_MAX_VERTS];
    uint count;
};

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

// std430 HullPair mirror: FxHull A (61 ints, padded) + FxBody A (16) + FxHull B + FxBody B. The host packer
// matches THIS layout (FxHull is 61 ints; std430 pads the trailing scalar of the int-array struct, but the host
// uses the SAME HullGpu mirror -> the struct sizes agree). gjk-settle's HullGpu is the precedent.
struct HullPair {
    FxHull hullA;
    FxBody bodyA;
    FxHull hullB;
    FxBody bodyB;
};

// std430 ManifoldGpu mirror (the host packs convex::ContactManifold into THIS 20 x int32 / 80-byte form for the
// memcmp): count (uint32), 4 points (x,y,z each Q16.16), 4 depths (Q16.16), normal.xyz (Q16.16). == the
// convex_manifold.comp ManifoldGpu.
struct ManifoldGpu {
    uint count;
    int  p0x, p0y, p0z;
    int  p1x, p1y, p1z;
    int  p2x, p2y, p2z;
    int  p3x, p3y, p3z;
    int  d0, d1, d2, d3;
    int  nx, ny, nz;
};

struct Params {
    int4 cfg;   // x=pairCount, y=pairEnabled, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<HullPair>    gPairs     : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<ManifoldGpu> gManifolds : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params>      gParams    : register(u2);

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

// ===== The Johnson sub-distance (VERBATIM gjk.h::DoSimplex2/3/4 == gjk_distance.comp) ======================
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

// ===== The EPA polytope state (fixed-size, thread-local — VERBATIM gjk_epa.comp) ==========================
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

// ===== The MF1 polygon-face tables (VERBATIM manifold.h::BuildCanonicalFaces / IsOctaHull) ================
// A face table: per-face vertex indices + per-face vert count + face count. Fixed-size, std430-shaped.
struct HullFaces {
    uint vertIdx[HF_MF_MAX_FACES][HF_MF_MAX_FACE_VERTS];
    uint vertCount[HF_MF_MAX_FACES];
    uint faceCount;
};

bool IsOctaHull(FxHull hull) {
    if (hull.count != 6u) return false;
    for (uint i = 0u; i < 6u; ++i) {
        int zeros = (hull.vx[i] == 0 ? 1 : 0) + (hull.vy[i] == 0 ? 1 : 0) + (hull.vz[i] == 0 ? 1 : 0);
        if (zeros < 2) return false;
    }
    return true;
}

HullFaces BuildCanonicalFaces(FxHull hull) {
    HullFaces faces;
    faces.faceCount = 0u;
    for (uint fi = 0u; fi < (uint)HF_MF_MAX_FACES; ++fi) {
        faces.vertCount[fi] = 0u;
        for (uint vi = 0u; vi < (uint)HF_MF_MAX_FACE_VERTS; ++vi) faces.vertIdx[fi][vi] = 0u;
    }
    if (hull.count == 4u) {
        // tetra: 4 tri faces.
        uint t[4][3] = { {0u,1u,2u}, {0u,1u,3u}, {0u,2u,3u}, {1u,2u,3u} };
        for (uint f = 0u; f < 4u; ++f) {
            faces.vertIdx[f][0] = t[f][0]; faces.vertIdx[f][1] = t[f][1]; faces.vertIdx[f][2] = t[f][2];
            faces.vertCount[f] = 3u;
        }
        faces.faceCount = 4u;
    } else if (hull.count == 8u) {
        // box: 6 QUAD faces.
        uint q[6][4] = { {0u,1u,3u,2u}, {4u,5u,7u,6u}, {0u,1u,5u,4u},
                         {2u,3u,7u,6u}, {0u,2u,6u,4u}, {1u,3u,7u,5u} };
        for (uint f = 0u; f < 6u; ++f) {
            faces.vertIdx[f][0] = q[f][0]; faces.vertIdx[f][1] = q[f][1];
            faces.vertIdx[f][2] = q[f][2]; faces.vertIdx[f][3] = q[f][3];
            faces.vertCount[f] = 4u;
        }
        faces.faceCount = 6u;
    } else if (hull.count == 6u) {
        if (IsOctaHull(hull)) {
            // octa: 8 tri faces.
            uint t[8][3] = { {0u,2u,4u}, {0u,4u,3u}, {0u,3u,5u}, {0u,5u,2u},
                             {1u,4u,2u}, {1u,3u,4u}, {1u,5u,3u}, {1u,2u,5u} };
            for (uint f = 0u; f < 8u; ++f) {
                faces.vertIdx[f][0] = t[f][0]; faces.vertIdx[f][1] = t[f][1]; faces.vertIdx[f][2] = t[f][2];
                faces.vertCount[f] = 3u;
            }
            faces.faceCount = 8u;
        } else {
            // wedge: 2 tri caps + 3 quad sides.
            faces.vertIdx[0][0] = 0u; faces.vertIdx[0][1] = 2u; faces.vertIdx[0][2] = 4u; faces.vertCount[0] = 3u;
            faces.vertIdx[1][0] = 1u; faces.vertIdx[1][1] = 5u; faces.vertIdx[1][2] = 3u; faces.vertCount[1] = 3u;
            faces.vertIdx[2][0] = 0u; faces.vertIdx[2][1] = 1u; faces.vertIdx[2][2] = 3u; faces.vertIdx[2][3] = 2u; faces.vertCount[2] = 4u;
            faces.vertIdx[3][0] = 0u; faces.vertIdx[3][1] = 4u; faces.vertIdx[3][2] = 5u; faces.vertIdx[3][3] = 1u; faces.vertCount[3] = 4u;
            faces.vertIdx[4][0] = 2u; faces.vertIdx[4][1] = 3u; faces.vertIdx[4][2] = 5u; faces.vertIdx[4][3] = 4u; faces.vertCount[4] = 4u;
            faces.faceCount = 5u;
        }
    }
    return faces;
}

// Hull local vert i -> world (VERBATIM the FxAdd(FxRotate(orient, v), pos) idiom).
int3 HullVertWorld(FxHull hull, int4 orient, int3 pos, uint i) {
    int3 lv = int3(hull.vx[i], hull.vy[i], hull.vz[i]);
    return FxAdd(FxRotate(orient, lv), pos);
}

int3 HullCentroidWorld(FxHull hull, int4 orient, int3 pos) {
    if (hull.count == 0u) return pos;
    int64_t sx = 0, sy = 0, sz = 0;
    for (uint i = 0u; i < hull.count; ++i) {
        int3 w = HullVertWorld(hull, orient, pos, i);
        sx += (int64_t)w.x; sy += (int64_t)w.y; sz += (int64_t)w.z;
    }
    return int3((int)(sx / (int64_t)hull.count), (int)(sy / (int64_t)hull.count),
                (int)(sz / (int64_t)hull.count));
}

int3 FaceCentroidWorld(FxHull hull, HullFaces faces, int4 orient, int3 pos, uint f) {
    uint vc = faces.vertCount[f];
    if (vc == 0u) return pos;
    int64_t sx = 0, sy = 0, sz = 0;
    for (uint k = 0u; k < vc; ++k) {
        int3 w = HullVertWorld(hull, orient, pos, faces.vertIdx[f][k]);
        sx += (int64_t)w.x; sy += (int64_t)w.y; sz += (int64_t)w.z;
    }
    return int3((int)(sx / (int64_t)vc), (int)(sy / (int64_t)vc), (int)(sz / (int64_t)vc));
}

int3 FaceNormalWorld(FxHull hull, HullFaces faces, int4 orient, int3 pos, uint f) {
    uint vc = faces.vertCount[f];
    if (vc < 3u) return int3(0, 0, 0);
    int3 v0 = HullVertWorld(hull, orient, pos, faces.vertIdx[f][0]);
    int3 v1 = HullVertWorld(hull, orient, pos, faces.vertIdx[f][1]);
    int3 v2 = HullVertWorld(hull, orient, pos, faces.vertIdx[f][2]);
    int3 n = FxCross(FxSub(v1, v0), FxSub(v2, v0));
    int3 hc = HullCentroidWorld(hull, orient, pos);
    int3 fc = FaceCentroidWorld(hull, faces, orient, pos, f);
    if (FxDot(n, FxSub(fc, hc)) < 0) n = FxNeg(n);
    return n;
}

// SupportFace: the face whose OUTWARD world normal is MOST PARALLEL to dir; FIXED order, STRICT-greater ->
// ties keep the LOWEST index. Returns HF_MF_MAX_FACES sentinel for an empty table (== manifold.h).
uint SupportFace(FxHull hull, HullFaces faces, int4 orient, int3 pos, int3 dir) {
    if (faces.faceCount == 0u) return (uint)HF_MF_MAX_FACES;
    uint best = 0u;
    int bestDot = FxDot(FaceNormalWorld(hull, faces, orient, pos, 0u), dir);
    for (uint f = 1u; f < faces.faceCount; ++f) {
        int d = FxDot(FaceNormalWorld(hull, faces, orient, pos, f), dir);
        if (d > bestDot) { bestDot = d; best = f; }
    }
    return best;
}

uint IncidentFace(FxHull hull, HullFaces faces, int4 orient, int3 pos, int3 refNormal) {
    return SupportFace(hull, faces, orient, pos, FxNeg(refNormal));
}

// ClipFaceAgainstFace: the deterministic Sutherland-Hodgman clip of the INCIDENT polygon against the REFERENCE
// face's side planes (VERBATIM manifold.h::ClipFaceAgainstFace). Writes outPts/outN.
void ClipFaceAgainstFace(FxHull refHull, int4 refO, int3 refP, HullFaces refFaces, uint refFace,
                         FxHull incHull, int4 incO, int3 incP, HullFaces incFaces, uint incFace,
                         out int3 outPts[HF_MF_MAX_CLIP_VERTS], out int outN) {
    outN = 0;
    for (int z = 0; z < HF_MF_MAX_CLIP_VERTS; ++z) outPts[z] = int3(0,0,0);
    if (refFace >= refFaces.faceCount || incFace >= incFaces.faceCount) return;
    uint refVc = refFaces.vertCount[refFace];
    uint incVc = incFaces.vertCount[incFace];
    if (refVc < 3u || incVc < 3u) return;

    int3 refV[HF_MF_MAX_FACE_VERTS];
    for (uint k = 0u; k < refVc; ++k)
        refV[k] = HullVertWorld(refHull, refO, refP, refFaces.vertIdx[refFace][k]);
    int3 refN = FaceNormalWorld(refHull, refFaces, refO, refP, refFace);
    int3 refC = FaceCentroidWorld(refHull, refFaces, refO, refP, refFace);

    int3 poly[HF_MF_MAX_CLIP_VERTS];
    int polyN = (int)incVc;
    for (uint k = 0u; k < incVc; ++k)
        poly[k] = HullVertWorld(incHull, incO, incP, incFaces.vertIdx[incFace][k]);

    for (uint e = 0u; e < refVc; ++e) {
        int3 a = refV[e];
        int3 b = refV[(e + 1u) % refVc];
        int3 edge = FxSub(b, a);
        int3 sideN = FxCross(refN, edge);
        if (FxDot(sideN, FxSub(refC, a)) < 0) sideN = FxNeg(sideN);

        int3 outV[HF_MF_MAX_CLIP_VERTS];
        int outNloc = 0;
        if (polyN == 0) break;
        int3 prev = poly[polyN - 1];
        int fprev = FxDot(sideN, FxSub(prev, a));
        for (int k = 0; k < polyN; ++k) {
            int3 cur = poly[k];
            int fcur = FxDot(sideN, FxSub(cur, a));
            bool curIn  = (fcur >= 0);
            bool prevIn = (fprev >= 0);
            if (curIn) {
                if (!prevIn && outNloc < (int)HF_MF_MAX_CLIP_VERTS) {
                    int denom = fprev - fcur;
                    int tp = (denom != 0) ? fxdiv(fprev, denom) : 0;
                    outV[outNloc++] = int3(prev.x + fxmul(cur.x - prev.x, tp),
                                           prev.y + fxmul(cur.y - prev.y, tp),
                                           prev.z + fxmul(cur.z - prev.z, tp));
                }
                if (outNloc < (int)HF_MF_MAX_CLIP_VERTS) outV[outNloc++] = cur;
            } else if (prevIn && outNloc < (int)HF_MF_MAX_CLIP_VERTS) {
                int denom = fprev - fcur;
                int tp = (denom != 0) ? fxdiv(fprev, denom) : 0;
                outV[outNloc++] = int3(prev.x + fxmul(cur.x - prev.x, tp),
                                       prev.y + fxmul(cur.y - prev.y, tp),
                                       prev.z + fxmul(cur.z - prev.z, tp));
            }
            prev = cur; fprev = fcur;
        }
        polyN = outNloc;
        for (int k = 0; k < polyN; ++k) poly[k] = outV[k];
    }

    outN = polyN;
    for (int k = 0; k < polyN; ++k) outPts[k] = poly[k];
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int pairCount   = gParams[0].cfg.x;
    int pairEnabled = gParams[0].cfg.y;
    int idx = (int)gid.x;
    if (idx >= pairCount) return;

    ManifoldGpu mz;
    mz.count = 0u;
    mz.p0x = 0; mz.p0y = 0; mz.p0z = 0; mz.p1x = 0; mz.p1y = 0; mz.p1z = 0;
    mz.p2x = 0; mz.p2y = 0; mz.p2z = 0; mz.p3x = 0; mz.p3y = 0; mz.p3z = 0;
    mz.d0 = 0; mz.d1 = 0; mz.d2 = 0; mz.d3 = 0; mz.nx = 0; mz.ny = 0; mz.nz = 0;

    if (pairEnabled == 0) { gManifolds[idx] = mz; return; }

    HullPair pr = gPairs[idx];
    FxHull hA = pr.hullA;
    FxHull hB = pr.hullB;
    int4 oA = int4(pr.bodyA.ox, pr.bodyA.oy, pr.bodyA.oz, pr.bodyA.ow);
    int3 pA = int3(pr.bodyA.px, pr.bodyA.py, pr.bodyA.pz);
    int4 oB = int4(pr.bodyB.ox, pr.bodyB.oy, pr.bodyB.oz, pr.bodyB.ow);
    int3 pB = int3(pr.bodyB.px, pr.bodyB.py, pr.bodyB.pz);

    // ===== gjk::Gjk (VERBATIM gjk_distance.comp) — produce the terminal simplex + overlap flag. =====
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
    uint giter = 0u;
    for (; giter < (uint)HF_GJK_MAX_ITER; ++giter) {
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

    // Separated -> empty manifold (count 0), the gjk::HullContact / HullContactMulti contract.
    if (!overlap) { gManifolds[idx] = mz; return; }

    // ===== gjk::Epa (VERBATIM gjk_epa.comp) — seed from the terminal simplex, expand, recover depth/normal. =====
    gVertCount = 0u;
    gFaceCount = 0u;
    uint sc = n; if (sc > 4u) sc = 4u;
    for (uint i = 0u; i < sc; ++i) {
        gVerts[i]  = sp[i];
        gVertsA[i] = swA[i];
        gVertsB[i] = swB[i];
    }
    gVertCount = sc;

    int3 kProbe[10];
    kProbe[0] = int3(HF_GJK_ONE, 0, 0); kProbe[1] = int3(0, HF_GJK_ONE, 0); kProbe[2] = int3(0, 0, HF_GJK_ONE);
    kProbe[3] = int3(-HF_GJK_ONE, 0, 0); kProbe[4] = int3(0, -HF_GJK_ONE, 0); kProbe[5] = int3(0, 0, -HF_GJK_ONE);
    kProbe[6] = int3(HF_GJK_ONE, HF_GJK_ONE, HF_GJK_ONE);
    kProbe[7] = int3(-HF_GJK_ONE, HF_GJK_ONE, HF_GJK_ONE);
    kProbe[8] = int3(HF_GJK_ONE, -HF_GJK_ONE, HF_GJK_ONE);
    kProbe[9] = int3(HF_GJK_ONE, HF_GJK_ONE, -HF_GJK_ONE);

    if (gVertCount < 2u) {
        for (uint p = 0u; p < 10u && gVertCount < 2u; ++p) {
            int3 pdir = FxNormalize(kProbe[p]);
            int3 wA = Support(hA, oA, pA, pdir);
            int3 wB = Support(hB, oB, pB, FxNeg(pdir));
            int3 spv = FxSub(wA, wB);
            bool dup = false;
            for (uint v = 0u; v < gVertCount; ++v)
                if (spv.x == gVerts[v].x && spv.y == gVerts[v].y && spv.z == gVerts[v].z) { dup = true; break; }
            if (dup) continue;
            gVerts[gVertCount] = spv; gVertsA[gVertCount] = wA; gVertsB[gVertCount] = wB; ++gVertCount;
        }
    }
    if (gVertCount == 2u) {
        int3 e = FxSub(gVerts[1], gVerts[0]);
        int eLen = FxLength(e);
        bool added = false;
        for (uint p = 0u; p < 10u && !added; ++p) {
            int3 pdir = kProbe[p];
            if (eLen >= HF_GJK_EDGE_EPS) {
                int proj = fxdiv(FxDot(pdir, e), eLen);
                int3 eUnit = FxNormalize(e);
                pdir = FxSub(pdir, FxScale(eUnit, proj));
            }
            if (FxLength(pdir) < HF_GJK_EDGE_EPS) continue;
            pdir = FxNormalize(pdir);
            int3 wA = Support(hA, oA, pA, pdir);
            int3 wB = Support(hB, oB, pB, FxNeg(pdir));
            int3 spv = FxSub(wA, wB);
            bool dup = false;
            for (uint v = 0u; v < gVertCount; ++v)
                if (spv.x == gVerts[v].x && spv.y == gVerts[v].y && spv.z == gVerts[v].z) { dup = true; break; }
            if (dup) continue;
            int3 nrm = FxCross(FxSub(gVerts[1], gVerts[0]), FxSub(spv, gVerts[0]));
            if (FxLength(nrm) < HF_GJK_EDGE_EPS) continue;
            gVerts[2] = spv; gVertsA[2] = wA; gVertsB[2] = wB; gVertCount = 3u;
            added = true;
        }
    }
    if (gVertCount == 3u) {
        int3 ab = FxSub(gVerts[1], gVerts[0]);
        int3 ac = FxSub(gVerts[2], gVerts[0]);
        int3 nrm = FxCross(ab, ac);
        int3 nUnit = (FxLength(nrm) < HF_GJK_EDGE_EPS) ? int3(0, HF_GJK_ONE, 0) : FxNormalize(nrm);
        int side = FxDot(nUnit, FxNeg(gVerts[0]));
        int3 cands[12];
        cands[0] = (side >= 0) ? nUnit : FxNeg(nUnit);
        cands[1] = FxNeg(cands[0]);
        for (uint p = 0u; p < 10u; ++p) cands[2u + p] = FxNormalize(kProbe[p]);
        for (uint c = 0u; c < 12u && gVertCount == 3u; ++c) {
            int3 cdir = cands[c];
            int3 wA = Support(hA, oA, pA, cdir);
            int3 wB = Support(hB, oB, pB, FxNeg(cdir));
            int3 spv = FxSub(wA, wB);
            bool dup = false;
            for (uint v = 0u; v < gVertCount; ++v)
                if (spv.x == gVerts[v].x && spv.y == gVerts[v].y && spv.z == gVerts[v].z) { dup = true; break; }
            if (dup) continue;
            int vol = FxDot(nrm, FxSub(spv, gVerts[0]));
            int avol = (vol < 0) ? -vol : vol;
            if (avol < HF_GJK_EDGE_EPS) continue;
            gVerts[3] = spv; gVertsA[3] = wA; gVertsB[3] = wB; gVertCount = 4u;
        }
    }

    // EPA result accumulators.
    int  epaDepth = 0;
    int3 epaNormal = int3(0, HF_GJK_ONE, 0);
    int3 epaCA = int3(0,0,0);
    int3 epaCB = int3(0,0,0);

    if (gVertCount < 4u) {
        epaDepth = 0;
        epaNormal = int3(0, HF_GJK_ONE, 0);
        epaCA = (gVertCount > 0u) ? gVertsA[0] : int3(0,0,0);
        epaCB = (gVertCount > 0u) ? gVertsB[0] : int3(0,0,0);
    } else {
        gInterior = FxScale(FxAdd(FxAdd(gVerts[0], gVerts[1]), FxAdd(gVerts[2], gVerts[3])), HF_GJK_ONE / 4);
        gFaceCount = 0u;
        EpaAddFace(0u, 1u, 2u);
        EpaAddFace(0u, 1u, 3u);
        EpaAddFace(0u, 2u, 3u);
        EpaAddFace(1u, 2u, 3u);

        if (gFaceCount == 0u) {
            epaDepth = 0;
            epaNormal = int3(0, HF_GJK_ONE, 0);
            epaCA = gVertsA[0];
            epaCB = gVertsB[0];
        } else {
            uint closestFace = 0u;
            uint eiter = 0u;
            for (; eiter < (uint)HF_EPA_MAX_ITER; ++eiter) {
                closestFace = 0u;
                int minDist = gFaceD[0];
                for (uint f = 1u; f < gFaceCount; ++f) {
                    if (gFaceD[f] < minDist) { minDist = gFaceD[f]; closestFace = f; }
                }
                int3 cfn = gFaceN[closestFace];
                int  cfd = gFaceD[closestFace];

                int3 edir = cfn;
                int3 wA = Support(hA, oA, pA, edir);
                int3 wB = Support(hB, oB, pB, FxNeg(edir));
                int3 spv = FxSub(wA, wB);

                int proj = FxDot(spv, edir);
                if (proj <= cfd + HF_EPA_TOL) break;

                bool dup = false;
                for (uint v = 0u; v < gVertCount; ++v)
                    if (spv.x == gVerts[v].x && spv.y == gVerts[v].y && spv.z == gVerts[v].z) { dup = true; break; }
                if (dup) break;
                if (gVertCount >= (uint)HF_EPA_MAX_PV) break;

                uint nv = gVertCount;
                gVerts[nv] = spv; gVertsA[nv] = wA; gVertsB[nv] = wB; ++gVertCount;

                uint horizU[HF_EPA_MAX_PF * 3];
                uint horizV[HF_EPA_MAX_PF * 3];
                uint horizCount = 0u;
                bool removed[HF_EPA_MAX_PF];
                for (uint f = 0u; f < gFaceCount; ++f) removed[f] = false;
                for (uint f = 0u; f < gFaceCount; ++f) {
                    int3 toNew = FxSub(spv, gVerts[gFaceA[f]]);
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
                uint wc = 0u;
                for (uint f = 0u; f < gFaceCount; ++f) {
                    if (!removed[f]) {
                        if (wc != f) {
                            gFaceA[wc] = gFaceA[f]; gFaceB[wc] = gFaceB[f]; gFaceC[wc] = gFaceC[f];
                            gFaceN[wc] = gFaceN[f]; gFaceD[wc] = gFaceD[f];
                        }
                        ++wc;
                    }
                }
                gFaceCount = wc;
                for (uint e = 0u; e < horizCount; ++e) {
                    EpaAddFace(horizU[e], horizV[e], nv);
                }
                if (gFaceCount == 0u) {
                    EpaAddFace(0u, 1u, nv);
                    if (gFaceCount == 0u) { ++eiter; break; }
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
                bu = HF_GJK_ONE - bv - bw;
            } else { bu = HF_GJK_ONE; bv = 0; bw = 0; }

            int3 cA = FxAdd(FxAdd(FxScale(gVertsA[cfa], bu), FxScale(gVertsA[cfb], bv)), FxScale(gVertsA[cfc], bw));
            int3 cB = FxAdd(FxAdd(FxScale(gVertsB[cfa], bu), FxScale(gVertsB[cfb], bv)), FxScale(gVertsB[cfc], bw));

            epaDepth = cfd;
            epaNormal = cfn;
            epaCA = cA;
            epaCB = cB;
        }
    }

    // ===== HullManifoldFromEpa (VERBATIM manifold.h) — the multi-point clip. =====
    int3 nrm = epaNormal;   // UNIT, A->B

    // The EPA-witness midpoint floor (the gjk::HullContact single-point behavior).
    int3 witnessMid = int3((epaCA.x + epaCB.x) / 2, (epaCA.y + epaCB.y) / 2, (epaCA.z + epaCB.z) / 2);

    HullFaces facesA = BuildCanonicalFaces(hA);
    HullFaces facesB = BuildCanonicalFaces(hB);

    ManifoldGpu m = mz;
    m.nx = nrm.x; m.ny = nrm.y; m.nz = nrm.z;

    // fallbackOnePoint helper inlined where needed.
    bool didFallback = false;
    if (facesA.faceCount == 0u || facesB.faceCount == 0u) {
        didFallback = true;
    }

    uint sfA = 0u, sfB = 0u;
    int3 nfA = int3(0,0,0), nfB = int3(0,0,0);
    int alignA = 0, alignB = 0;
    int3 negN = FxNeg(nrm);
    bool refIsA = true;

    if (!didFallback) {
        sfA = SupportFace(hA, facesA, oA, pA, nrm);
        nfA = FaceNormalWorld(hA, facesA, oA, pA, sfA);
        alignA = FxDot(nfA, nrm);
        sfB = SupportFace(hB, facesB, oB, pB, negN);
        nfB = FaceNormalWorld(hB, facesB, oB, pB, sfB);
        alignB = FxDot(nfB, negN);
        refIsA = (alignA >= alignB);
    }

    // Reference/incident selection (only meaningful when !didFallback). NOTE: DXC's ?: cannot select whole
    // struct values (FxHull/HullFaces), so the ref/inc hulls + face tables are chosen via explicit if/else
    // (the convex_manifold.comp element-wise-select precedent). The VALUES are identical to the CPU ternary.
    FxHull refHull, incHull;
    int4   refO, incO;
    int3   refP, incP;
    HullFaces refFaces, incFaces;
    uint   refFace;
    int3   refN;
    if (refIsA) {
        refHull = hA; refO = oA; refP = pA; refFaces = facesA; refFace = sfA; refN = nfA;
        incHull = hB; incO = oB; incP = pB; incFaces = facesB;
    } else {
        refHull = hB; refO = oB; refP = pB; refFaces = facesB; refFace = sfB; refN = nfB;
        incHull = hA; incO = oA; incP = pA; incFaces = facesA;
    }
    uint incFace = didFallback ? 0u : IncidentFace(incHull, incFaces, incO, incP, refN);

    if (!didFallback && (refFace >= refFaces.faceCount || incFace >= incFaces.faceCount)) didFallback = true;

    int candN = 0;
    int3 candPts[HF_MF_MAX_CLIP_VERTS];
    int  candDepth[HF_MF_MAX_CLIP_VERTS];

    if (!didFallback) {
        int3 refC = FaceCentroidWorld(refHull, refFaces, refO, refP, refFace);
        int3 clipped[HF_MF_MAX_CLIP_VERTS];
        int clipN = 0;
        ClipFaceAgainstFace(refHull, refO, refP, refFaces, refFace,
                            incHull, incO, incP, incFaces, incFace, clipped, clipN);
        if (clipN == 0) {
            didFallback = true;
        } else {
            for (int k = 0; k < clipN; ++k) {
                int3 rel = int3(refC.x - clipped[k].x, refC.y - clipped[k].y, refC.z - clipped[k].z);
                int d = FxDot(refN, rel);
                if (d >= 0) { candPts[candN] = clipped[k]; candDepth[candN] = d; ++candN; }
            }
            if (candN == 0) didFallback = true;
        }
    }

    if (didFallback) {
        // The single-point floor: NEVER count 0 for an overlapping pair (== manifold.h fallbackOnePoint).
        m.count = 1u;
        m.p0x = witnessMid.x; m.p0y = witnessMid.y; m.p0z = witnessMid.z;
        m.d0 = epaDepth;
        m.nx = nrm.x; m.ny = nrm.y; m.nz = nrm.z;
        gManifolds[idx] = m;
        return;
    }

    // Reduce to <=4: ALWAYS keep the DEEPEST (max depth, tie -> lowest clip-order index), then up to 3 MORE in
    // clip order. (== manifold.h HullManifoldFromEpa reduce.)
    int deepest = 0;
    for (int k = 1; k < candN; ++k) if (candDepth[k] > candDepth[deepest]) deepest = k;
    int3 outPts[4];
    int  outDep[4];
    outPts[0] = candPts[deepest]; outDep[0] = candDepth[deepest];
    int cnt = 1;
    for (int k = 0; k < candN && cnt < 4; ++k) {
        if (k == deepest) continue;
        outPts[cnt] = candPts[k]; outDep[cnt] = candDepth[k]; ++cnt;
    }
    m.count = (uint)cnt;
    m.p0x = outPts[0].x; m.p0y = outPts[0].y; m.p0z = outPts[0].z; m.d0 = outDep[0];
    if (cnt > 1) { m.p1x = outPts[1].x; m.p1y = outPts[1].y; m.p1z = outPts[1].z; m.d1 = outDep[1]; }
    if (cnt > 2) { m.p2x = outPts[2].x; m.p2y = outPts[2].y; m.p2z = outPts[2].z; m.d2 = outDep[2]; }
    if (cnt > 3) { m.p3x = outPts[3].x; m.p3y = outPts[3].y; m.p3z = outPts[3].z; m.d3 = outDep[3]; }
    m.nx = nrm.x; m.ny = nrm.y; m.nz = nrm.z;
    gManifolds[idx] = m;
}
