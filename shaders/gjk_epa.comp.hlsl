// Slice GJ3 — General Convex-Hull Contacts: THE EPA ALGORITHM (penetration depth + contact normal, THE CRUX)
// compute pass. ONE THREAD PER OVERLAPPING HULL-PAIR (per-pair INDEPENDENT — each thread reads its pair's two
// hulls + bodies + the precomputed terminal GJK simplex and writes its OWN EpaResult slot; race-free, NO
// atomics) runs the full EPA polytope expansion, copying engine/sim/gjk.h::Epa's body VERBATIM (the SAME
// seed-to-tetra, the SAME OUTWARD-winding face build, the SAME min-dist lowest-index closest-face selection,
// the SAME visible-face removal + edge-cancel horizon + fixed-order re-triangulation, the SAME convergence +
// duplicate + cap termination, the SAME barycentric contact recovery) so the GPU EpaResult is byte-identical
// to the CPU gjk::Epa -> the host GPU==CPU memcmp catches any divergence.
//
// INTEGER WIDTH (the determinism crux, the GJ1/GJ2 lesson): fxmul/fxdiv + FxDot/FxCross/FxLength(FxISqrt)/
// FxNormalize/FxRotate all use int64_t. DXC -spirv compiles int64 (the Int64 capability, the gjk_distance.comp
// pattern); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is
// VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal hf_gen_msl list). The Metal --gjk-epa
// showcase runs the CPU gjk::Epa over the same pairs -> byte-identical to this GPU result BY CONSTRUCTION,
// while the Vulkan side carries the GPU==CPU bit-identity proof.
//
// pairEnabled=0 -> write a cleared EpaResult for every pair (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..3):
//   b0 gHulls   : the FxHull array (kMaxHullVerts x int3 verts + uint count per hull), READ.
//   b1 gPairs   : the EpaPair array (hullIndexA, bodyA, hullIndexB, bodyB, terminal Simplex) per pair, READ.
//   b2 gOut     : the EpaResultGpu array per pair, WRITE.
//   b3 gParams  : { pairCount, pairEnabled, _, _ }, READ.

#define HF_FPX_FRAC 16          // MUST match fpx.h::kFrac
#define HF_GJK_MAX_VERTS 20     // MUST match gjk.h::kMaxHullVerts
#define HF_EPA_MAX_ITER 48      // MUST match gjk.h::kEpaMaxIter
#define HF_EPA_MAX_PV 64        // MUST match gjk.h::kMaxPolyVerts
#define HF_EPA_MAX_PF 128       // MUST match gjk.h::kMaxPolyFaces
#define HF_GJK_ONE (1 << HF_FPX_FRAC)
#define HF_GJK_EDGE_EPS (HF_GJK_ONE / 256)   // MUST match convex.h::kEdgeEps
#define HF_EPA_TOL (HF_GJK_ONE / 256)        // MUST match gjk.h::kEpaTol

// std430 FxHull mirror (engine/sim/gjk.h::FxHull): flat int arrays (vx,vy,vz) + count.
struct FxHull {
    int  vx[HF_GJK_MAX_VERTS];
    int  vy[HF_GJK_MAX_VERTS];
    int  vz[HF_GJK_MAX_VERTS];
    uint count;
};

// std430 FxBody mirror (engine/sim/fpx.h::FxBody): 16 x int32 (64 bytes). EPA reads pos + orient.
struct FxBody {
    int  px, py, pz;
    int  vx, vy, vz;
    int  invMass;
    uint flags;
    int  radius;
    int  ox, oy, oz, ow;
    int  ax, ay, az;
};

// std430 Simplex mirror (engine/sim/gjk.h::Simplex): pts[4], csoA[4], csoB[4] (each int3), count, packed flat.
struct SimplexGpu {
    int  px[4]; int py[4]; int pz[4];     // pts
    int  ax[4]; int ay[4]; int az[4];     // csoA
    int  bx[4]; int by[4]; int bz[4];     // csoB
    uint count; uint _pad0, _pad1, _pad2;
};

// std430 EpaPair mirror: hull index A + body A + hull index B + body B + the terminal GJK simplex (the seed).
struct EpaPair {
    uint   hullIndexA; uint _pa0, _pa1, _pa2;
    FxBody bodyA;
    uint   hullIndexB; uint _pb0, _pb1, _pb2;
    FxBody bodyB;
    SimplexGpu seed;
};

// std430 EpaResult mirror — the memcmp target (matches the host packer): depth, normal(3), contactA(3),
// contactB(3), featureFaceId, iterations, valid.
struct EpaResultGpu {
    int  depth;
    int  nx, ny, nz;
    int  cax, cay, caz;
    int  cbx, cby, cbz;
    uint featureFaceId;
    uint iterations;
    uint valid;
};

struct Params {
    int4 cfg;   // x=pairCount, y=pairEnabled, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxHull>       gHulls  : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<EpaPair>      gPairs  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<EpaResultGpu> gOut    : register(u2);
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

// ===== Support (VERBATIM gjk.h::SupportLocal / Support) ===================================================
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

// ===== The EPA polytope state (fixed-size, thread-local — the GPU has no dynamic alloc) ===================
// Parallel verts / vertsA / vertsB + faces (a,b,c indices, unit normal, dist). vertCount / faceCount.
static int3 gVerts[HF_EPA_MAX_PV];
static int3 gVertsA[HF_EPA_MAX_PV];
static int3 gVertsB[HF_EPA_MAX_PV];
static uint gFaceA[HF_EPA_MAX_PF];
static uint gFaceB[HF_EPA_MAX_PF];
static uint gFaceC[HF_EPA_MAX_PF];
static int3 gFaceN[HF_EPA_MAX_PF];
static int  gFaceD[HF_EPA_MAX_PF];
static int3 gInterior;   // the FIXED interior reference (seed tetra centroid).
static uint gVertCount;
static uint gFaceCount;

// EpaAddFace (VERBATIM gjk.h::EpaAddFace): build a face a,b,c oriented AWAY from gInterior; degenerate -> skip.
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

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int pairCount   = gParams[0].cfg.x;
    int pairEnabled = gParams[0].cfg.y;
    int idx = (int)gid.x;
    if (idx >= pairCount) return;

    EpaResultGpu zero = (EpaResultGpu)0;
    if (pairEnabled == 0) { gOut[idx] = zero; return; }

    EpaPair pr = gPairs[idx];
    FxHull hA = gHulls[pr.hullIndexA];
    FxHull hB = gHulls[pr.hullIndexB];
    int4 oA = int4(pr.bodyA.ox, pr.bodyA.oy, pr.bodyA.oz, pr.bodyA.ow);
    int3 pA = int3(pr.bodyA.px, pr.bodyA.py, pr.bodyA.pz);
    int4 oB = int4(pr.bodyB.ox, pr.bodyB.oy, pr.bodyB.oz, pr.bodyB.ow);
    int3 pB = int3(pr.bodyB.px, pr.bodyB.py, pr.bodyB.pz);

    // ----- VERBATIM gjk::Epa -----
    gVertCount = 0u;
    gFaceCount = 0u;

    // SEED: copy the terminal simplex CSO points + witnesses.
    uint sc = pr.seed.count; if (sc > 4u) sc = 4u;
    for (uint i = 0u; i < sc; ++i) {
        gVerts[i]  = int3(pr.seed.px[i], pr.seed.py[i], pr.seed.pz[i]);
        gVertsA[i] = int3(pr.seed.ax[i], pr.seed.ay[i], pr.seed.az[i]);
        gVertsB[i] = int3(pr.seed.bx[i], pr.seed.by[i], pr.seed.bz[i]);
    }
    gVertCount = sc;

    // Seed expansion to a tetra (the non-tetra path — VERBATIM gjk::Epa). The FIXED candidate direction set.
    int3 kProbe[10];
    kProbe[0] = int3(HF_GJK_ONE, 0, 0); kProbe[1] = int3(0, HF_GJK_ONE, 0); kProbe[2] = int3(0, 0, HF_GJK_ONE);
    kProbe[3] = int3(-HF_GJK_ONE, 0, 0); kProbe[4] = int3(0, -HF_GJK_ONE, 0); kProbe[5] = int3(0, 0, -HF_GJK_ONE);
    kProbe[6] = int3(HF_GJK_ONE, HF_GJK_ONE, HF_GJK_ONE);
    kProbe[7] = int3(-HF_GJK_ONE, HF_GJK_ONE, HF_GJK_ONE);
    kProbe[8] = int3(HF_GJK_ONE, -HF_GJK_ONE, HF_GJK_ONE);
    kProbe[9] = int3(HF_GJK_ONE, HF_GJK_ONE, -HF_GJK_ONE);

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
        int3 nUnit = (FxLength(n) < HF_GJK_EDGE_EPS) ? int3(0, HF_GJK_ONE, 0) : FxNormalize(n);
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

    EpaResultGpu r = zero;

    if (gVertCount < 4u) {
        r.valid = 1u; r.depth = 0;
        r.nx = 0; r.ny = HF_GJK_ONE; r.nz = 0;
        int3 ca = (gVertCount > 0u) ? gVertsA[0] : int3(0,0,0);
        int3 cb = (gVertCount > 0u) ? gVertsB[0] : int3(0,0,0);
        r.cax = ca.x; r.cay = ca.y; r.caz = ca.z;
        r.cbx = cb.x; r.cby = cb.y; r.cbz = cb.z;
        r.featureFaceId = 0u; r.iterations = 0u;
        gOut[idx] = r; return;
    }

    // The FIXED interior reference = the seed tetra centroid.
    gInterior = FxScale(FxAdd(FxAdd(gVerts[0], gVerts[1]), FxAdd(gVerts[2], gVerts[3])), HF_GJK_ONE / 4);

    // Build the initial tetra's 4 faces.
    gFaceCount = 0u;
    EpaAddFace(0u, 1u, 2u);
    EpaAddFace(0u, 1u, 3u);
    EpaAddFace(0u, 2u, 3u);
    EpaAddFace(1u, 2u, 3u);

    if (gFaceCount == 0u) {
        r.valid = 1u; r.depth = 0;
        r.nx = 0; r.ny = HF_GJK_ONE; r.nz = 0;
        r.cax = gVertsA[0].x; r.cay = gVertsA[0].y; r.caz = gVertsA[0].z;
        r.cbx = gVertsB[0].x; r.cby = gVertsB[0].y; r.cbz = gVertsB[0].z;
        r.featureFaceId = 0u; r.iterations = 0u;
        gOut[idx] = r; return;
    }

    // The EPA expansion loop.
    uint closestFace = 0u;
    uint iter = 0u;
    for (; iter < (uint)HF_EPA_MAX_ITER; ++iter) {
        // closest face: min dist, lowest index.
        closestFace = 0u;
        int minDist = gFaceD[0];
        for (uint f = 1u; f < gFaceCount; ++f) {
            if (gFaceD[f] < minDist) { minDist = gFaceD[f]; closestFace = f; }
        }
        uint cfa = gFaceA[closestFace], cfb = gFaceB[closestFace], cfc = gFaceC[closestFace];
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

        // Horizon edges (ordered (u,v) pairs) + edge-cancel.
        uint horizU[HF_EPA_MAX_PF * 3];
        uint horizV[HF_EPA_MAX_PF * 3];
        uint horizCount = 0u;

        // visible-face pass (FIXED index order): collect edges + mark removed.
        bool removed[HF_EPA_MAX_PF];
        for (uint f = 0u; f < gFaceCount; ++f) removed[f] = false;
        for (uint f = 0u; f < gFaceCount; ++f) {
            int3 toNew = FxSub(sp, gVerts[gFaceA[f]]);
            if (FxDot(gFaceN[f], toNew) > 0) {
                removed[f] = true;
                // add the 3 edges with the edge-cancel rule.
                uint ea0 = gFaceA[f], eb0 = gFaceB[f];
                uint ea1 = gFaceB[f], eb1 = gFaceC[f];
                uint ea2 = gFaceC[f], eb2 = gFaceA[f];
                // edge 0
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
                // edge 1
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
                // edge 2
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

        // compact out removed faces (keep survivors in original relative order).
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

        // re-triangulate: connect nv to each horizon edge in FIXED order.
        for (uint e = 0u; e < horizCount; ++e) {
            EpaAddFace(horizU[e], horizV[e], nv);
        }

        if (gFaceCount == 0u) {
            EpaAddFace(0u, 1u, nv);
            if (gFaceCount == 0u) { ++iter; break; }
        }
    }

    // Re-find the closest face.
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

    r.valid = 1u;
    r.depth = cfd;
    r.nx = cfn.x; r.ny = cfn.y; r.nz = cfn.z;
    r.cax = cA.x; r.cay = cA.y; r.caz = cA.z;
    r.cbx = cB.x; r.cby = cB.y; r.cbz = cB.z;
    r.featureFaceId = (cfa & 0x3FFu) | ((cfb & 0x3FFu) << 10) | ((cfc & 0x3FFu) << 20);
    r.iterations = iter;
    gOut[idx] = r;
}
