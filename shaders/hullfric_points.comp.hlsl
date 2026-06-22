// Slice HF1 — Hull Friction + Joints: THE TAGGED FRICTION MANIFOLD ON THE EPA NORMAL compute pass (the friction
// BEACHHEAD of FLAGSHIP #30: HULL FRICTION + HULL JOINTS, hf::sim::hullfric). ONE THREAD PER HULL PAIR (per-pair
// INDEPENDENT, 64 threads/group, NO atomics) reads its pair's HOST-BUILT warmhull keyed manifold (the int64-derived
// convex::ContactManifold positions/depths/normal + the WH1 HullContactKey per point, computed host-side by
// manifold::HullContactMulti + warmhull::BuildKeyedHullManifold) then APPENDS the friction data: SIGN-CORRECT the
// EPA normal A->B once (== warmhull::SolveHullManifoldWarm), build the integer tangent basis (t1,t2) via the FC1
// fixed Gram-Schmidt (fric::MakeTangentBasis copied VERBATIM) + the basis-axis index (fric::LeastAlignedAxis), and
// zero the per-point impulse accumulators, writing its OWN gManifolds[pair] HullFrictionManifoldGpu. The basis math
// copies engine/sim/hullfric.h::BuildHullFrictionManifold's tangent-basis body VERBATIM (the SAME FxNormalize/FxDot/
// FxCross, the SAME least-aligned-axis tie-break) so the GPU friction manifold is byte-identical to the CPU
// hullfric::BuildHullFrictionManifold -> the host GPU==CPU memcmp catches any divergence.
//
// INTEGER WIDTH (the int64 reality, the WH2/FC1 lesson): the MANIFOLD this wraps is built HOST-SIDE by the int64
// manifold::HullContactMulti (GJK/EPA + the SH clip), AND the tangent basis FxNormalize/FxDot/FxCross are int64
// (world-scale products overflow int32). DXC -spirv compiles int64; glslc (the Metal HLSL->SPIR-V->MSL frontend)
// CANNOT parse int64_t in HLSL, so this shader is VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal
// hf_gen_msl list — the warmhull_cache.comp convention). The Metal --hf1-points showcase runs the CPU hullfric::
// BuildAllHullFrictionManifolds over the same pairs -> byte-identical to this GPU result BY CONSTRUCTION, while the
// Vulkan side carries the GPU==CPU bit-identity proof.
//
// Buffers (storage, bound at compute bindings 0..2):
//   b0 gInputs    : the host-built warmhull keyed manifold per pair (count + 4 ManifoldPoint {pos.xyz, depth} +
//                   normal.xyz + 4 HullContactKey), READ. positions/depths/normal/keys are the int64 host result.
//   b1 gManifolds : the HullFrictionManifoldGpu array per pair (count + normal + 4 pts + 4 depths + t1 + t2 +
//                   basisAxis + 4 keys + 4 (point + 3 accumulators)), WRITE.
//   b2 gParams    : { pairCount, _, _, _ }, READ.

#define HF_FPX_FRAC 16          // MUST match fpx.h::kFrac
#define HF_FPX_ONE  (1 << HF_FPX_FRAC)

// std430 HullContactKey mirror (engine/sim/warmhull.h::HullContactKey): 4 x uint32 (16 bytes).
struct HullContactKey {
    uint bodyA;
    uint bodyB;
    uint refFaceId;
    uint incVertId;
};

// std430 ManifoldPoint mirror (a convex::ContactManifold point packed: pos.xyz + depth): 4 x int32 (16 bytes).
struct ManifoldPoint {
    int px, py, pz;
    int depth;
};

// std430 input keyed manifold (the host-built convex::ContactManifold + the parallel keys) — IDENTICAL layout to
// warmhull_cache.comp's KeyedHullInputGpu: count (uint) + 4 ManifoldPoint (64) + normal.xyz (12) + pad (4) + 4
// HullContactKey (64) = 148 bytes.
struct KeyedHullInputGpu {
    uint count;
    ManifoldPoint pts[4];
    int nx, ny, nz;
    int _pad0;
    HullContactKey keys[4];
};

// std430 HullFrictionPoint mirror (engine/sim/hullfric.h::HullFrictionPoint): point.xyz + 3 accumulators = 6 x int32
// (24 bytes).
struct HullFrictionPointGpu {
    int px, py, pz;
    int normalImpulse;
    int tangentImpulse1;
    int tangentImpulse2;
};

// std430 output friction manifold (engine/sim/hullfric.h::HullFrictionManifold), packed for the memcmp:
//   count (uint, 4) + normal.xyz (12) + 4 ManifoldPoint pts (64, point+depth) + t1.xyz (12) + t2.xyz (12) +
//   basisAxis (4) + 4 HullContactKey (64) + 4 HullFrictionPoint (96) = 268 bytes.
// NOTE: the host packs the SAME field order — the memcmp compares this byte image.
struct HullFrictionManifoldGpu {
    uint count;
    int nx, ny, nz;
    ManifoldPoint pts[4];          // point.xyz + depth (the geometry, == warmhull manifold)
    int t1x, t1y, t1z;
    int t2x, t2y, t2z;
    int basisAxis;
    HullContactKey keys[4];
    HullFrictionPointGpu fpts[4];
};

struct HullfricParams {
    int4 cfg;   // x=pairCount, y/z/w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<KeyedHullInputGpu>        gInputs    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<HullFrictionManifoldGpu>  gManifolds : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<HullfricParams>           gParams    : register(u2);

// ===== VERBATIM Q16.16 toolbox (== warmhull_warm.comp / fric.h / convex.h / fpx.h) =====
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

// VERBATIM fric.h::LeastAlignedAxis — the FIXED argmin cardinal-axis (lowest-index tie-break on |n.axis|).
int LeastAlignedAxis(int3 n) {
    int a0 = abs(n.x), a1 = abs(n.y), a2 = abs(n.z);
    int best = 0;
    int bestVal = a0;
    if (a1 < bestVal) { bestVal = a1; best = 1; }   // strict-< -> lowest index keeps a tie
    if (a2 < bestVal) { bestVal = a2; best = 2; }
    return best;
}

// VERBATIM fric.h::CardinalAxis — the i-th cardinal axis e_i (kOne on component i).
int3 CardinalAxis(int i) {
    return int3((i == 0) ? HF_FPX_ONE : 0, (i == 1) ? HF_FPX_ONE : 0, (i == 2) ? HF_FPX_ONE : 0);
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int pairCount = gParams[0].cfg.x;
    int idx = (int)gid.x;
    if (idx >= pairCount) return;

    // Cleared HullFrictionManifoldGpu (count=0, every field zero) — the empty/separated default.
    HullFrictionManifoldGpu mz;
    mz.count = 0u;
    mz.nx = 0; mz.ny = 0; mz.nz = 0;
    {
        ManifoldPoint pz; pz.px = 0; pz.py = 0; pz.pz = 0; pz.depth = 0;
        HullContactKey kz; kz.bodyA = 0u; kz.bodyB = 0u; kz.refFaceId = 0u; kz.incVertId = 0u;
        HullFrictionPointGpu fz; fz.px = 0; fz.py = 0; fz.pz = 0;
        fz.normalImpulse = 0; fz.tangentImpulse1 = 0; fz.tangentImpulse2 = 0;
        [unroll] for (uint z = 0u; z < 4u; ++z) { mz.pts[z] = pz; mz.keys[z] = kz; mz.fpts[z] = fz; }
    }
    mz.t1x = 0; mz.t1y = 0; mz.t1z = 0;
    mz.t2x = 0; mz.t2y = 0; mz.t2z = 0;
    mz.basisAxis = 0;

    KeyedHullInputGpu inp = gInputs[idx];
    if (inp.count == 0u) { gManifolds[idx] = mz; return; }

    HullFrictionManifoldGpu km = mz;
    km.count = inp.count;

    // (3) sign-correct the EPA normal A->B once. The host already sign-corrects via (bodyB.pos - bodyA.pos); the
    // host packs the ALREADY-sign-corrected normal into gInputs.n? NO — gInputs carries the RAW warmhull manifold
    // normal (BuildKeyedHullManifold's manifold.normal). The host BuildHullFrictionManifold flips it by the body
    // positions. So the host ALSO packs the sign-corrected normal in the input here: the showcase packs the
    // sign-corrected A->B normal into inp.n directly (the SAME bits the CPU computes), so the GPU consumes it as-is.
    int3 n = int3(inp.nx, inp.ny, inp.nz);

    // (4) the FC1 tangent basis — fric::MakeTangentBasis(n) copied VERBATIM (the fixed integer Gram-Schmidt).
    int mi = LeastAlignedAxis(n);
    int3 e = CardinalAxis(mi);
    int eDotN = FxDot(e, n);
    int3 r = int3(e.x - fxmul(eDotN, n.x), e.y - fxmul(eDotN, n.y), e.z - fxmul(eDotN, n.z));
    int3 t1 = FxNormalize(r);
    int3 t2 = FxCross(n, t1);   // (n, t1) orthonormal -> n x t1 already unit (no second normalize)

    km.nx = n.x; km.ny = n.y; km.nz = n.z;
    km.t1x = t1.x; km.t1y = t1.y; km.t1z = t1.z;
    km.t2x = t2.x; km.t2y = t2.y; km.t2z = t2.z;
    km.basisAxis = mi;

    // (5) copy the manifold geometry + keys; zero every point's accumulators (the cold-start contract).
    [unroll] for (uint i = 0u; i < 4u; ++i) {
        km.pts[i] = inp.pts[i];
        km.keys[i] = inp.keys[i];
        km.fpts[i].px = inp.pts[i].px;
        km.fpts[i].py = inp.pts[i].py;
        km.fpts[i].pz = inp.pts[i].pz;
        km.fpts[i].normalImpulse = 0;
        km.fpts[i].tangentImpulse1 = 0;
        km.fpts[i].tangentImpulse2 = 0;
    }

    gManifolds[idx] = km;
}
