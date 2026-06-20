// Slice FC1 — Deterministic Contact Friction: THE TANGENT BASIS compute pass (the BEACHHEAD of FLAGSHIP #20:
// DETERMINISTIC TANGENTIAL CONTACT FRICTION, hf::sim::fric). ONE THREAD PER INPUT NORMAL (per-normal
// INDEPENDENT — each thread reads its normal and writes its OWN TangentBasis slot; race-free, NO atomics)
// runs the fixed integer Gram-Schmidt tangent-basis build, copying engine/sim/fric.h::MakeTangentBasis's
// body VERBATIM (the SAME LeastAlignedAxis argmin + lowest-index tie-break, the SAME project-off-n ->
// FxNormalize -> FxCross(n,t1), the SAME FxDot/FxCross/FxNormalize int64 ops) so the GPU TangentBasis[] is
// byte-identical to the CPU MakeTangentBasis -> the host GPU==CPU memcmp catches any divergence.
//
// INTEGER WIDTH (the determinism crux, the FPX3/CX1 lesson): fxmul/fxdiv/FxISqrt + the FxDot/FxCross Q16.16
// products use int64_t. DXC -spirv compiles int64 (the Int64 capability, the convex_sat.comp / fpx_solve.comp
// pattern); glslc (the Metal HLSL->SPIR-V->MSL frontend) CANNOT parse int64_t in HLSL, so this shader is
// VULKAN-SPIR-V-ONLY (in the Vulkan compile list, NOT in the Metal hf_gen_msl list). The Metal --fric-basis
// showcase runs the CPU fric::MakeTangentBasis over the same normals -> byte-identical to this GPU result BY
// CONSTRUCTION, while the Vulkan side carries the GPU==CPU bit-identity proof.
//
// basisEnabled=0 -> write a cleared TangentBasis (all zero) for every normal (the disabled-path no-op).
//
// Buffers (storage, bound at compute bindings 0..2):
//   b0 gNormals : the input unit-normal array (FxVec3 = 3 x int32 per normal), READ.
//   b1 gBases   : the output TangentBasis array (t1.xyz, t2.xyz = 6 x int32 per normal), WRITE.
//   b2 gParams  : { normalCount, basisEnabled, _, _ }, READ.

#define HF_FPX_FRAC 16   // MUST match fpx.h::kFrac
#define HF_FPX_ONE  (1 << HF_FPX_FRAC)

// std430 FxVec3 mirror (engine/sim/fpx.h::FxVec3): 3 x int32 (12 bytes) — x,y,z Q16.16.
struct FxVec3 {
    int x, y, z;
};

// std430 TangentBasis mirror (engine/sim/fric.h::TangentBasis): t1.xyz + t2.xyz = 6 x int32 (24 bytes).
struct TangentBasis {
    int t1x, t1y, t1z;
    int t2x, t2y, t2z;
};

struct FricParams {
    int4 cfg;   // x=normalCount, y=basisEnabled, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<FxVec3>      gNormals : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<TangentBasis> gBases  : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<FricParams>  gParams  : register(u2);

// VERBATIM fpx.h::fxmul — (a*b) >> kFrac with an int64 intermediate (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_FPX_FRAC);
}

// VERBATIM fpx.h::fxdiv — (a << kFrac) / b in Q16.16 (int64 shift + truncating divide; guard b==0).
int fxdiv(int a, int b) {
    if (b == 0) return 0;
    return (int)(((int64_t)a << HF_FPX_FRAC) / (int64_t)b);
}

// VERBATIM fpx.h::FxISqrt — floor(sqrt) of a non-negative int64 (binary digit-by-digit).
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

// VERBATIM fpx.h::FxLength — sqrt(x^2+y^2+z^2) in Q16.16 (sum of int64 squares -> Q32.32 -> floor-sqrt).
int FxLength(int3 v) {
    int64_t sx = (int64_t)v.x * (int64_t)v.x;
    int64_t sy = (int64_t)v.y * (int64_t)v.y;
    int64_t sz = (int64_t)v.z * (int64_t)v.z;
    return (int)FxISqrt(sx + sy + sz);
}

// VERBATIM fpx.h::FxNormalize — unit vector via FxLength (int64); len==0 -> the fixed (0,1,0) fallback.
int3 FxNormalize(int3 v) {
    int len = FxLength(v);
    if (len == 0) return int3(0, HF_FPX_ONE, 0);
    return int3(fxdiv(v.x, len), fxdiv(v.y, len), fxdiv(v.z, len));
}

// VERBATIM convex.h::FxDot — (ax*bx + ay*by + az*bz) >> kFrac, int64 intermediate.
int FxDot(int3 a, int3 b) {
    int64_t d = (int64_t)a.x * (int64_t)b.x + (int64_t)a.y * (int64_t)b.y + (int64_t)a.z * (int64_t)b.z;
    return (int)(d >> HF_FPX_FRAC);
}

// VERBATIM convex.h::FxCross — the Q16.16 cross product.
int3 FxCross(int3 a, int3 b) {
    return int3(fxmul(a.y, b.z) - fxmul(a.z, b.y),
                fxmul(a.z, b.x) - fxmul(a.x, b.z),
                fxmul(a.x, b.y) - fxmul(a.y, b.x));
}

int absI(int v) { return v < 0 ? -v : v; }

// VERBATIM fric.h::LeastAlignedAxis — the index of the cardinal axis LEAST aligned with n (smallest |n.axis|),
// FIXED lowest-index tie-break (strict-< on the running minimum). |n.axis| == |FxDot(n, e_axis)|.
uint LeastAlignedAxis(int3 n) {
    int a0 = absI(n.x), a1 = absI(n.y), a2 = absI(n.z);
    uint best = 0u;
    int bestVal = a0;
    if (a1 < bestVal) { bestVal = a1; best = 1u; }   // strict-< -> lowest index keeps a tie
    if (a2 < bestVal) { bestVal = a2; best = 2u; }
    return best;
}

// VERBATIM fric.h::CardinalAxis — e_i (kOne on component i, 0 elsewhere).
int3 CardinalAxis(uint i) {
    return int3((i == 0u) ? HF_FPX_ONE : 0,
                (i == 1u) ? HF_FPX_ONE : 0,
                (i == 2u) ? HF_FPX_ONE : 0);
}

// VERBATIM fric.h::MakeTangentBasis — the fixed integer Gram-Schmidt: least-aligned cardinal axis ->
// project off n -> FxNormalize -> FxCross(n, t1). Returns (t1, t2).
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
    int normalCount  = gParams[0].cfg.x;
    int basisEnabled = gParams[0].cfg.y;
    int idx = (int)gid.x;
    if (idx >= normalCount) return;

    // Disabled -> write a cleared basis (the byte-identical no-op).
    if (basisEnabled == 0) {
        TangentBasis z;
        z.t1x = 0; z.t1y = 0; z.t1z = 0; z.t2x = 0; z.t2y = 0; z.t2z = 0;
        gBases[idx] = z;
        return;
    }

    FxVec3 nv = gNormals[idx];
    int3 n = int3(nv.x, nv.y, nv.z);
    int3 t1, t2;
    MakeTangentBasis(n, t1, t2);

    TangentBasis b;
    b.t1x = t1.x; b.t1y = t1.y; b.t1z = t1.z;
    b.t2x = t2.x; b.t2y = t2.y; b.t2z = t2.z;
    gBases[idx] = b;
}
