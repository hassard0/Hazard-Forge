// Slice MF4 — Hull Narrowphase Hardening: THE HARDENED HULL WORLD STEP (the new-physics money beat, the HEADLINE
// of FLAGSHIP #25: hf::sim::manifold). The GPU mirror of engine/sim/manifold.h::StepHullWorldHardened/N — ONE
// THREAD ([numthreads(1,1,1)]) runs the WHOLE StepHullWorldHardenedN over the SMALL hull set: K ticks of
// predict-integrate -> all-pairs narrowphase -> world Gauss-Seidel impulse -> POSITION de-penetration, copying
// manifold.h::StepHullWorldHardened VERBATIM. StepHullWorldHardened IS gjk::StepHullWorld (== hull_step.comp)
// with EXACTLY TWO swaps and NOTHING else:
//   (step 2 inertia) HullInvInertiaBody (diagonal) -> HullInertiaBodyFull (the FULL signed-tetra tensor) +
//                    WorldInvInertia (diagonal outer-product) -> WorldInvInertiaFull (R·M·Rᵀ);
//   (steps 3+4 contact) HullContact (single-point) -> HullContactMulti (the MF2/MF3 multi-point manifold).
// So a box dropped FLAT (tilted) onto a static box SETTLES TO REST where the single-point hull_step leaves it
// teetering. The final gBodies array is byte-identical to the CPU manifold::StepHullWorldHardenedN -> the host
// GPU==CPU memcmp catches any divergence.
//
// CHUNKED 1 TICK / DISPATCH (the documented Windows ~2s TDR rule, [[hazard-forge-gpu-tdr-chunking]]): the
// hardened step is HEAVIER than the single-point step (the full inertia + the multi-point clip per pair), so the
// host dispatches it ONE tick per dispatch over the SAME body buffer (a ComputeToComputeBarrier between ticks) —
// the buffer carries the exact Gauss-Seidel state, so the result is BIT-IDENTICAL to one big dispatch; 1 tick is
// far under the TDR window by construction (TDR impossible). ticks here is the per-dispatch tick count (== 1).
//
// SINGLE THREAD: the world step is order-dependent (the world Gauss-Seidel + the in-place de-pen), so the GPU
// mirror MUST be single-thread (the hull_step.comp / convex_step.comp convention). The hull set is small.
//
// INTEGER WIDTH (the GJ4/MF3/CX4 lesson): fxmul/fxdiv/FxISqrt + the FxDot/FxCross/FxRotate/FxNormalize/quaternion
// Q16.16 products + the GJK sub-distance + the EPA face math + the SH clip + the full-inertia covariance + the
// 3x3 symmetric inverse all use int64_t. DXC -spirv compiles int64 (the Int64 capability, the hull_step.comp /
// hull_manifold.comp pattern); glslc CANNOT parse int64 in HLSL, so this shader is VULKAN-SPIR-V-ONLY (NOT in
// hf_gen_msl). The Metal --mf4-stack runs the CPU manifold::StepHullWorldHardenedN over the same world ->
// byte-identical to this GPU result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU memcmp proof.
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
#define HF_MAX_BODIES 8                      // the small-scene cap (floor + a few hulls)
#define HF_MF_MAX_FACES 8                    // MUST match manifold.h::kMaxHullFaces
#define HF_MF_MAX_FACE_VERTS 4               // MUST match manifold.h::kMaxFaceVerts
#define HF_MF_MAX_CLIP_VERTS 8               // MUST match manifold.h::kMaxClipVerts

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

// The step params (== the host ConvexStepConfig + counts), packed for std430 (== hull_step.comp StepParams).
struct StepParams {
    int4 c0;   // x=bodyCount, y=ticks, z=stepEnabled, w=solveIters
    int4 c1;   // x=posIters, y=restitution, z=slop, w=beta
    int4 c2;   // x=linDamp, y=angDamp, z=dt, w=_pad
    int4 c3;   // x=gravity.x, y=gravity.y, z=gravity.z, w=_pad
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxBody>     gBodies : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<FxHull>     gHulls  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<StepParams> gParams : register(u2);

// ===== VERBATIM Q16.16 toolbox (== hull_step.comp / hull_manifold.comp / convex.h / fpx.h) =====
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

// ===== Quaternion integrate (== fpx.h::FxQuatMul/FxQuatNormalize/IntegrateBodyFull == hull_step.comp) =====
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
    int len = (int)FxISqrt64(sx + sy + sz + sw);
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

// ===== SWAP 1: THE FULL INERTIA (== manifold.h::FxHullInertiaBodyFull/FxMat3SymInverse/WorldInvInertiaFull) ===
// HullInvInertiaBodyDiag (== gjk.h::FxHullInvInertiaBody): the diagonal floor (fallback for an empty face table).
int3 HullInvInertiaBodyDiag(FxHull hull, int invMass) {
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

// FxMat3SymInverse (== manifold.h::FxMat3SymInverse): the integer symmetric-3x3 inverse (adjugate/det, fixed
// order). Degenerate determinant -> the zero matrix (caller falls back). Writes 9-element row-major out[9].
void FxMat3SymInverse(int M[9], out int outM[9]) {
    int a = M[0], b = M[1], c = M[2];
    int d = M[3], e = M[4], f = M[5];
    int g = M[6], h = M[7], i = M[8];
    int A =  (fxmul(e, i) - fxmul(f, h));
    int B = -(fxmul(d, i) - fxmul(f, g));
    int C =  (fxmul(d, h) - fxmul(e, g));
    int D = -(fxmul(b, i) - fxmul(c, h));
    int E =  (fxmul(a, i) - fxmul(c, g));
    int F = -(fxmul(a, h) - fxmul(b, g));
    int G =  (fxmul(b, f) - fxmul(c, e));
    int H = -(fxmul(a, f) - fxmul(c, d));
    int I =  (fxmul(a, e) - fxmul(b, d));
    int det = fxmul(a, A) + fxmul(b, B) + fxmul(c, C);
    [unroll] for (int z = 0; z < 9; ++z) outM[z] = 0;
    if (det == 0) return;
    outM[0] = fxdiv(A, det); outM[1] = fxdiv(D, det); outM[2] = fxdiv(G, det);
    outM[3] = fxdiv(B, det); outM[4] = fxdiv(E, det); outM[5] = fxdiv(H, det);
    outM[6] = fxdiv(C, det); outM[7] = fxdiv(F, det); outM[8] = fxdiv(I, det);
}

// The MF1 polygon-face tables (forward-declared here; defined below for the manifold). HullInertiaBodyFull needs
// the face table, so it is defined AFTER BuildCanonicalFaces (see the manifold section). To keep the dependency
// order simple we re-declare the HullFaces struct + BuildCanonicalFaces here (== hull_manifold.comp), then
// HullInertiaBodyFull uses it.
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
        uint t[4][3] = { {0u,1u,2u}, {0u,1u,3u}, {0u,2u,3u}, {1u,2u,3u} };
        for (uint f = 0u; f < 4u; ++f) {
            faces.vertIdx[f][0] = t[f][0]; faces.vertIdx[f][1] = t[f][1]; faces.vertIdx[f][2] = t[f][2];
            faces.vertCount[f] = 3u;
        }
        faces.faceCount = 4u;
    } else if (hull.count == 8u) {
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
            uint t[8][3] = { {0u,2u,4u}, {0u,4u,3u}, {0u,3u,5u}, {0u,5u,2u},
                             {1u,4u,2u}, {1u,3u,4u}, {1u,5u,3u}, {1u,2u,5u} };
            for (uint f = 0u; f < 8u; ++f) {
                faces.vertIdx[f][0] = t[f][0]; faces.vertIdx[f][1] = t[f][1]; faces.vertIdx[f][2] = t[f][2];
                faces.vertCount[f] = 3u;
            }
            faces.faceCount = 8u;
        } else {
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

// HullInertiaBodyFull (== manifold.h::FxHullInertiaBodyFull): the FULL symmetric body-space INVERSE inertia as a
// 9-element row-major matrix. Signed-tetra covariance (tetra fan from the local origin over each face), int64
// accumulation, deferred /120 + density scaling, then FxMat3SymInverse. Static -> zero; degenerate -> diagonal.
void HullInertiaBodyFull(FxHull hull, HullFaces faces, int invMass, out int outI[9]) {
    [unroll] for (int z0 = 0; z0 < 9; ++z0) outI[z0] = 0;
    if (invMass == 0) return;   // static -> zero matrix

    bool degenerate = (faces.faceCount == 0u);

    int64_t accVol = 0;
    int64_t accXX = 0, accYY = 0, accZZ = 0;
    int64_t accXY = 0, accXZ = 0, accYZ = 0;

    if (!degenerate) {
        for (uint f = 0u; f < faces.faceCount; ++f) {
            uint vc = faces.vertCount[f];
            if (vc < 3u) continue;
            int3 v0 = int3(hull.vx[faces.vertIdx[f][0]], hull.vy[faces.vertIdx[f][0]], hull.vz[faces.vertIdx[f][0]]);
            for (uint k = 1u; k + 1u < vc; ++k) {
                int3 a = v0;
                int3 b = int3(hull.vx[faces.vertIdx[f][k]], hull.vy[faces.vertIdx[f][k]], hull.vz[faces.vertIdx[f][k]]);
                int3 c = int3(hull.vx[faces.vertIdx[f][k+1u]], hull.vy[faces.vertIdx[f][k+1u]], hull.vz[faces.vertIdx[f][k+1u]]);
                int det = FxDot(a, FxCross(b, c));
                accVol += (int64_t)det;
                // S_ij covariance sums.
                int axx = a.x, bxx = b.x, cxx = c.x;
                int ayy = a.y, byy = b.y, cyy = c.y;
                int azz = a.z, bzz = b.z, czz = c.z;
                int Sxx = 2 * (fxmul(axx,axx) + fxmul(bxx,bxx) + fxmul(cxx,cxx))
                        + (fxmul(axx,bxx) + fxmul(axx,bxx)) + (fxmul(axx,cxx) + fxmul(axx,cxx)) + (fxmul(bxx,cxx) + fxmul(bxx,cxx));
                int Syy = 2 * (fxmul(ayy,ayy) + fxmul(byy,byy) + fxmul(cyy,cyy))
                        + (fxmul(ayy,byy) + fxmul(ayy,byy)) + (fxmul(ayy,cyy) + fxmul(ayy,cyy)) + (fxmul(byy,cyy) + fxmul(byy,cyy));
                int Szz = 2 * (fxmul(azz,azz) + fxmul(bzz,bzz) + fxmul(czz,czz))
                        + (fxmul(azz,bzz) + fxmul(azz,bzz)) + (fxmul(azz,czz) + fxmul(azz,czz)) + (fxmul(bzz,czz) + fxmul(bzz,czz));
                int Sxy = 2 * (fxmul(axx,ayy) + fxmul(bxx,byy) + fxmul(cxx,cyy))
                        + (fxmul(axx,byy) + fxmul(ayy,bxx)) + (fxmul(axx,cyy) + fxmul(ayy,cxx)) + (fxmul(bxx,cyy) + fxmul(byy,cxx));
                int Sxz = 2 * (fxmul(axx,azz) + fxmul(bxx,bzz) + fxmul(cxx,czz))
                        + (fxmul(axx,bzz) + fxmul(azz,bxx)) + (fxmul(axx,czz) + fxmul(azz,cxx)) + (fxmul(bxx,czz) + fxmul(bzz,cxx));
                int Syz = 2 * (fxmul(ayy,azz) + fxmul(byy,bzz) + fxmul(cyy,czz))
                        + (fxmul(ayy,bzz) + fxmul(azz,byy)) + (fxmul(ayy,czz) + fxmul(azz,cyy)) + (fxmul(byy,czz) + fxmul(bzz,cyy));
                accXX += (int64_t)fxmul(det, Sxx);
                accYY += (int64_t)fxmul(det, Syy);
                accZZ += (int64_t)fxmul(det, Szz);
                accXY += (int64_t)fxmul(det, Sxy);
                accXZ += (int64_t)fxmul(det, Sxz);
                accYZ += (int64_t)fxmul(det, Syz);
            }
        }
        if (accVol <= 0) degenerate = true;
    }

    if (degenerate) {
        int3 d = HullInvInertiaBodyDiag(hull, invMass);
        outI[0] = d.x; outI[4] = d.y; outI[8] = d.z;
        return;
    }

    int mass = fxdiv(HF_FPX_ONE, invMass);
    int vol  = (int)(accVol / 6);
    if (vol == 0) {
        int3 d = HullInvInertiaBodyDiag(hull, invMass);
        outI[0] = d.x; outI[4] = d.y; outI[8] = d.z;
        return;
    }
    int density = fxdiv(mass, vol);

    int mXX = (int)(accXX / 120), mYY = (int)(accYY / 120), mZZ = (int)(accZZ / 120);
    int mXY = (int)(accXY / 120), mXZ = (int)(accXZ / 120), mYZ = (int)(accYZ / 120);

    int I[9];
    I[0] = fxmul(density, mYY + mZZ);
    I[4] = fxmul(density, mXX + mZZ);
    I[8] = fxmul(density, mXX + mYY);
    I[1] = -fxmul(density, mXY); I[3] = I[1];
    I[2] = -fxmul(density, mXZ); I[6] = I[2];
    I[5] = -fxmul(density, mYZ); I[7] = I[5];

    int invI[9];
    FxMat3SymInverse(I, invI);
    if (invI[0] == 0 && invI[4] == 0 && invI[8] == 0) {
        int3 d = HullInvInertiaBodyDiag(hull, invMass);
        outI[0] = d.x; outI[4] = d.y; outI[8] = d.z;
        return;
    }
    [unroll] for (int z1 = 0; z1 < 9; ++z1) outI[z1] = invI[z1];
}

// WorldInvInertiaFull (== manifold.h::WorldInvInertiaFull): R · M · Rᵀ, R columns = BoxAxes(body).
void WorldInvInertiaFull(int4 q, int M[9], out int outM[9]) {
    int3 ax0 = FxRotate(q, int3(HF_FPX_ONE, 0, 0));
    int3 ax1 = FxRotate(q, int3(0, HF_FPX_ONE, 0));
    int3 ax2 = FxRotate(q, int3(0, 0, HF_FPX_ONE));
    int3 Rrow0 = int3(ax0.x, ax1.x, ax2.x);
    int3 Rrow1 = int3(ax0.y, ax1.y, ax2.y);
    int3 Rrow2 = int3(ax0.z, ax1.z, ax2.z);
    // RM = R·M (column-by-column).
    int RM[9];
    for (int c = 0; c < 3; ++c) {
        int3 mc = int3(M[0 + c], M[3 + c], M[6 + c]);
        RM[0 + c] = FxDot(Rrow0, mc);
        RM[3 + c] = FxDot(Rrow1, mc);
        RM[6 + c] = FxDot(Rrow2, mc);
    }
    int3 RMrow0 = int3(RM[0], RM[1], RM[2]);
    int3 RMrow1 = int3(RM[3], RM[4], RM[5]);
    int3 RMrow2 = int3(RM[6], RM[7], RM[8]);
    int3 axc[3] = { ax0, ax1, ax2 };   // Rᵀ column c == ax[c]
    for (int c2 = 0; c2 < 3; ++c2) {
        int3 tc = axc[c2];
        outM[0 + c2] = FxDot(RMrow0, tc);
        outM[3 + c2] = FxDot(RMrow1, tc);
        outM[6 + c2] = FxDot(RMrow2, tc);
    }
}

// ===== Support (VERBATIM gjk.h::SupportLocal / Support / SupportMinkowski) =====
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

// ===== The Johnson sub-distance (VERBATIM gjk.h::DoSimplex2/3/4 == hull_step.comp) =====
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

// ===== GJK terminal-simplex result (the EPA seed) =====
struct GjkOut { uint overlap; int3 pts[4]; int3 csoA[4]; int3 csoB[4]; uint count; };

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
        for (uint i2 = n; i2 > 0u; --i2) { sp[i2] = sp[i2-1u]; swA[i2] = swA[i2-1u]; swB[i2] = swB[i2-1u]; }
        sp[0] = a; swA[0] = wA; swB[0] = wB; ++n;
        Sub sub;
        if (n == 2u)      sub = DoSimplex2(sp[0], sp[1]);
        else if (n == 3u) sub = DoSimplex3(sp[0], sp[1], sp[2]);
        else              sub = DoSimplex4(sp[0], sp[1], sp[2], sp[3]);
        if (sub.containsOrigin) { overlap = true; break; }
        int3 nsp[4], nswA[4], nswB[4]; int nw[4] = {0,0,0,0};
        for (uint i3 = 0u; i3 < sub.size; ++i3) {
            uint k = sub.keep[i3];
            nsp[i3] = sp[k]; nswA[i3] = swA[k]; nswB[i3] = swB[k]; nw[i3] = sub.w[i3];
        }
        int3 newClosest = int3(0, 0, 0);
        for (uint i4 = 0u; i4 < sub.size; ++i4) {
            sp[i4] = nsp[i4]; swA[i4] = nswA[i4]; swB[i4] = nswB[i4];
            newClosest = int3(newClosest.x + fxmul(nsp[i4].x, nw[i4]),
                              newClosest.y + fxmul(nsp[i4].y, nw[i4]),
                              newClosest.z + fxmul(nsp[i4].z, nw[i4]));
        }
        n = sub.size;
        closest = newClosest;
    }
    res.overlap = overlap ? 1u : 0u;
    res.count = n;
    for (uint i5 = 0u; i5 < n && i5 < 4u; ++i5) { res.pts[i5] = sp[i5]; res.csoA[i5] = swA[i5]; res.csoB[i5] = swB[i5]; }
    return res;
}

// ===== The EPA polytope state (thread-local — single-thread dispatch; == hull_step.comp) =====
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
        for (uint f0 = 0u; f0 < gFaceCount; ++f0) removed[f0] = false;
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

// ===== The MF1 face helpers (VERBATIM hull_manifold.comp) =====
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

// ===== SWAP 2: HullContactMulti (== manifold.h::HullContactMulti) — the multi-point manifold (count 1-4). =====
struct Manifold { uint count; int3 pts[4]; int depths[4]; int3 normal; };

Manifold HullContactMulti(FxHull hA, int4 oA, int3 pA, FxHull hB, int4 oB, int3 pB) {
    Manifold m;
    m.count = 0u; m.normal = int3(0,0,0);
    [unroll] for (int z = 0; z < 4; ++z) { m.pts[z] = int3(0,0,0); m.depths[z] = 0; }

    GjkOut g = Gjk(hA, oA, pA, hB, oB, pB);
    if (g.overlap == 0u) return m;   // separated -> count 0
    EpaOut e = RunEpa(hA, oA, pA, hB, oB, pB, g);

    int3 nrm = e.normal;   // UNIT, A->B
    int3 witnessMid = int3((e.cA.x + e.cB.x) / 2, (e.cA.y + e.cB.y) / 2, (e.cA.z + e.cB.z) / 2);

    HullFaces facesA = BuildCanonicalFaces(hA);
    HullFaces facesB = BuildCanonicalFaces(hB);

    bool didFallback = (facesA.faceCount == 0u || facesB.faceCount == 0u);

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
        m.count = 1u;
        m.pts[0] = witnessMid;
        m.depths[0] = e.depth;
        m.normal = nrm;
        return m;
    }

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
    for (int z = 0; z < cnt; ++z) { m.pts[z] = outPts[z]; m.depths[z] = outDep[z]; }
    m.normal = nrm;
    return m;
}

// ===== SolveManifoldImpulse over the multi-point manifold (== convex.h::SolveManifoldImpulse, ONE inner sweep,
// looping 0..count). =====
void SolveManifoldImpulse(inout FxBody bA, inout FxBody bB, int invIaW[9], int invIbW[9], Manifold m,
                          int restitution) {
    if (m.count == 0u) return;
    int3 nrm = m.normal;
    int3 ab = int3(bB.px - bA.px, bB.py - bA.py, bB.pz - bA.pz);
    if (FxDot(nrm, ab) < 0) nrm = int3(-nrm.x, -nrm.y, -nrm.z);
    int invMassA = bA.invMass;
    int invMassB = bB.invMass;
    for (uint pi = 0u; pi < m.count; ++pi) {
        int3 p = m.pts[pi];
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
        if (vn >= 0) continue;
        int3 raxn = FxCross(rA, nrm);
        int3 rbxn = FxCross(rB, nrm);
        int angA = FxDot(nrm, FxCross(Mat3MulVec(invIaW, raxn), rA));
        int angB = FxDot(nrm, FxCross(Mat3MulVec(invIbW, rbxn), rB));
        int k = invMassA + invMassB + angA + angB;
        if (k <= 0) continue;
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

    // ===== StepHullWorldHardenedN: `ticks` x StepHullWorldHardened (== manifold.h, VERBATIM). =====
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

        // (2 — SWAP) world inverse inertias, once per tick, FULL tensor.
        int invIW[HF_MAX_BODIES][9];
        for (int i2 = 0; i2 < n; ++i2) {
            int4 q = int4(body[i2].ox, body[i2].oy, body[i2].oz, body[i2].ow);
            HullFaces faces = BuildCanonicalFaces(hull[i2]);
            int Ibody[9];
            HullInertiaBodyFull(hull[i2], faces, body[i2].invMass, Ibody);
            int M[9];
            WorldInvInertiaFull(q, Ibody, M);
            [unroll] for (int z = 0; z < 9; ++z) invIW[i2][z] = M[z];
        }

        // (3 — SWAP) impulse solve — world Gauss-Seidel over the all-pairs list, HullContactMulti.
        for (int sweep = 0; sweep < solveIters; ++sweep) {
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    if (body[i].invMass == 0 && body[j].invMass == 0) continue;
                    int4 oi = int4(body[i].ox, body[i].oy, body[i].oz, body[i].ow);
                    int3 pi = int3(body[i].px, body[i].py, body[i].pz);
                    int4 oj = int4(body[j].ox, body[j].oy, body[j].oz, body[j].ow);
                    int3 pj = int3(body[j].px, body[j].py, body[j].pz);
                    Manifold m = HullContactMulti(hull[i], oi, pi, hull[j], oj, pj);
                    if (m.count == 0u) continue;
                    int iaW[9]; int ibW[9];
                    [unroll] for (int z = 0; z < 9; ++z) { iaW[z] = invIW[i][z]; ibW[z] = invIW[j][z]; }
                    FxBody bi = body[i]; FxBody bj = body[j];
                    SolveManifoldImpulse(bi, bj, iaW, ibW, m, restitution);
                    body[i] = bi; body[j] = bj;
                }
            }
        }

        // (4 — SWAP) position de-penetration — posIters sweeps, HullContactMulti, m.depths[0]+m.normal.
        for (int pit = 0; pit < posIters; ++pit) {
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    int invSum = body[i].invMass + body[j].invMass;
                    if (invSum == 0) continue;
                    int4 oi = int4(body[i].ox, body[i].oy, body[i].oz, body[i].ow);
                    int3 pi = int3(body[i].px, body[i].py, body[i].pz);
                    int4 oj = int4(body[j].ox, body[j].oy, body[j].oz, body[j].ow);
                    int3 pj = int3(body[j].px, body[j].py, body[j].pz);
                    Manifold m = HullContactMulti(hull[i], oi, pi, hull[j], oj, pj);
                    if (m.count == 0u) continue;
                    int3 nrm = m.normal;
                    int3 ab = int3(body[j].px - body[i].px, body[j].py - body[i].py, body[j].pz - body[i].pz);
                    if (FxDot(nrm, ab) < 0) nrm = int3(-nrm.x, -nrm.y, -nrm.z);
                    int excess = m.depths[0] - slop;
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

    for (int wi = 0; wi < n; ++wi) gBodies[wi] = body[wi];
}
