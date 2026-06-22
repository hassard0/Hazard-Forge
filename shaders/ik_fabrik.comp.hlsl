// Slice IK3 — Deterministic IK Control-Rig: FABRIK N-BONE CHAIN compute (the 3rd slice of FLAGSHIP #32).
// ONE thread per CHAIN/TARGET runs the WHOLE engine/anim/ik.h::FabrikSolve for `iters` and writes the K
// solved joint positions (3 ints each) to gOut. Each thread starts from the SAME fixed rest layout (the
// chain laid out straight along +X with fixed per-segment lengths from gParams) and reaches its target
// (from gTargets). Pure FxNormalize (int64 sqrt) + integer position updates — NO acos/sin/cos. The
// per-chain solves are independent -> order-independent, race-free, NO atomics.
//
// THE int64 REALITY (VULKAN-SPIR-V-ONLY): FxNormalize/FxLength use int64 (FxISqrt), so this shader is in
// the Vulkan compile list ONLY (DXC compiles int64; glslc cannot lower it to MSL) — NOT in the Metal
// hf_gen_msl list. The Metal --ik3-fabrik showcase runs the CPU FabrikSolve over the same targets ->
// byte-identical to this GPU result by construction (the IK2/FPX3 split); the Vulkan side carries the
// GPU==CPU memcmp proof. The FabrikSolve body here is copied VERBATIM from ik.h.
//
// A short chain (K<=8) x modest iters (~10) over a single dispatch is LIGHT (no TDR watchdog risk — a few
// thousand int64-sqrt ops per thread), so the whole solve runs in ONE dispatch (NO chunking needed).
//
// enabled==0 -> every thread writes 0 (the disabled-path no-op; gOut stays the cleared all-zero upload).
//
// Buffers (storage, bound at compute bindings 0..2):
//   b0 gTargets : 3 int Q16.16 per chain {tx,ty,tz} (the end-effector goal), READ.
//   b1 gOut     : K*3 int Q16.16 per chain {pos[0..K-1].xyz}, WRITE.
//   b2 gParams  : { chainCount, enabled, boneCount(K), iters; rootX, rootY, rootZ, _; len0,len1,...,len6 (K-1) }.

#define HF_IK_FRAC  16
#define HF_IK_ONE   65536       // 1.0 in Q16.16
#define HF_IK_MAXB  8           // kIkMaxBones

struct Params {
    int4 cfg;    // x=chainCount, y=enabled, z=boneCount(K), w=iters
    int4 root;   // x,y,z=root pos (Q16.16), w=_
    int4 len0;   // len[0..3]
    int4 len1;   // len[4..6], w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<int>    gTargets : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<int>    gOut     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<Params> gParams  : register(u2);

// ---- VERBATIM fpx.h int64 substrate -----------------------------------------------------------------
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_IK_FRAC); }
int fxdiv(int a, int b) { if (b == 0) return 0; return (int)(((int64_t)a << HF_IK_FRAC) / (int64_t)b); }

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
int FxLength3(int vx, int vy, int vz) {
    int64_t sx = (int64_t)vx * (int64_t)vx;
    int64_t sy = (int64_t)vy * (int64_t)vy;
    int64_t sz = (int64_t)vz * (int64_t)vz;
    return (int)FxISqrt(sx + sy + sz);
}
// FxNormalize3 -> the unit vector (or the (0,kOne,0) fallback for a zero vector). VERBATIM fpx.h.
void FxNormalize3(int vx, int vy, int vz, out int ox, out int oy, out int oz) {
    int len = FxLength3(vx, vy, vz);
    if (len == 0) { ox = 0; oy = HF_IK_ONE; oz = 0; return; }
    ox = fxdiv(vx, len); oy = fxdiv(vy, len); oz = fxdiv(vz, len);
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int chainCount = gParams[0].cfg.x;
    int enabled    = gParams[0].cfg.y;
    int K          = gParams[0].cfg.z;
    int iters      = gParams[0].cfg.w;
    int rootX = gParams[0].root.x, rootY = gParams[0].root.y, rootZ = gParams[0].root.z;

    int t = (int)gid.x;
    if (t >= chainCount) return;

    if (enabled == 0) {
        for (int k = 0; k < K * 3; ++k) gOut[(t * HF_IK_MAXB * 3) + k] = 0;
        return;
    }

    // Per-segment lengths (K-1 of them) from gParams (len0.xyzw, len1.xyz).
    int len[HF_IK_MAXB - 1];
    len[0] = gParams[0].len0.x; len[1] = gParams[0].len0.y;
    len[2] = gParams[0].len0.z; len[3] = gParams[0].len0.w;
    len[4] = gParams[0].len1.x; len[5] = gParams[0].len1.y; len[6] = gParams[0].len1.z;

    // The target for this chain.
    int tx = gTargets[t * 3 + 0];
    int ty = gTargets[t * 3 + 1];
    int tz = gTargets[t * 3 + 2];

    // The rest layout: lay the chain out straight along +X from root, cumulative segment lengths. This is
    // IDENTICAL to the CPU showcase's rest initialization (so GPU==CPU starts from the same state).
    int px[HF_IK_MAXB], py[HF_IK_MAXB], pz[HF_IK_MAXB];
    px[0] = rootX; py[0] = rootY; pz[0] = rootZ;
    for (int i = 1; i < K; ++i) {
        px[i] = px[i - 1] + len[i - 1];
        py[i] = py[i - 1];
        pz[i] = pz[i - 1];
    }

    // === FabrikSolve (VERBATIM engine/anim/ik.h) ===
    for (int it = 0; it < iters; ++it) {
        // BACKWARD: pin pos[K-1] = target, reach back toward the root.
        px[K - 1] = tx; py[K - 1] = ty; pz[K - 1] = tz;
        for (int i = K - 2; i >= 0; --i) {
            int dx = px[i] - px[i + 1];
            int dy = py[i] - py[i + 1];
            int dz = pz[i] - pz[i + 1];
            int ux, uy, uz; FxNormalize3(dx, dy, dz, ux, uy, uz);
            px[i] = px[i + 1] + fxmul(len[i], ux);
            py[i] = py[i + 1] + fxmul(len[i], uy);
            pz[i] = pz[i + 1] + fxmul(len[i], uz);
        }
        // FORWARD: pin pos[0] = root, reach forward toward the end-effector.
        px[0] = rootX; py[0] = rootY; pz[0] = rootZ;
        for (int i = 1; i < K; ++i) {
            int dx = px[i] - px[i - 1];
            int dy = py[i] - py[i - 1];
            int dz = pz[i] - pz[i - 1];
            int ux, uy, uz; FxNormalize3(dx, dy, dz, ux, uy, uz);
            px[i] = px[i - 1] + fxmul(len[i - 1], ux);
            py[i] = py[i - 1] + fxmul(len[i - 1], uy);
            pz[i] = pz[i - 1] + fxmul(len[i - 1], uz);
        }
    }

    // Write the K solved joint positions (3 ints each) into this chain's slot.
    int base = t * HF_IK_MAXB * 3;
    for (int i = 0; i < K; ++i) {
        gOut[base + i * 3 + 0] = px[i];
        gOut[base + i * 3 + 1] = py[i];
        gOut[base + i * 3 + 2] = pz[i];
    }
}
