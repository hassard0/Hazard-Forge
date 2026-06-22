// Slice IK1 — Deterministic IK Control-Rig: THE FIXED-POINT ANGLE LUT sweep compute (the BEACHHEAD of
// FLAGSHIP #32). ONE thread per sweep SAMPLE. Each thread reads its host-baked Q16.16 angle LUTs (acos /
// sin / cos, uploaded as read-only SSBOs — the SAME tables engine/anim/ik.h builds in double + snaps once)
// and evaluates the PURE-INTEGER lookups (the int32 bin index + the deterministic fixed-point lerp copied
// VERBATIM from engine/anim/ik.h::FxAcosLut / FxSinLut / FxCosLut / FxReducePhaseBin), writing the three
// Q16.16 results to gOut. The per-sample writes are independent -> order-independent, race-free, NO atomics.
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it runs
// the SAME pure-int32 index + lerp the header runs over the SAME uploaded tables. A divergence vs the
// header is exactly what the host's GPU==CPU memcmp catches.
//
// THE INT32 CRUX (MSL-NATIVE on BOTH backends): acos/sin/cos index math + lerp are PURE INT32 (the acos
// power-of-two index trick; the sin/cos modulo-reduce + nested base-256 frac; lerp diffs bounded so
// diff*frac fits int32) — NO int64. So this shader is in hf_gen_msl (a TRUE GPU pass on Vulkan AND Metal,
// the strongest cross-vendor proof, like fpx_pair_count / fract_classify). atan2 (which needs the int64
// fxdiv ratio) is the CPU/Vulkan part — engine/anim/ik.h::FxAtan2Lut — and is NOT in this sweep.
//
// enabled==0 -> every thread writes 0 (the disabled-path no-op; gOut stays the cleared all-zero upload).
//
// Buffers (storage, bound at compute bindings 0..4; on Metal these land at buffer(0..4)):
//   b0 gAcos   : the acos LUT (257 int Q16.16 entries), READ.
//   b1 gSin    : the sin  LUT (257 int Q16.16 entries), READ.
//   b2 gCos    : the cos  LUT (257 int Q16.16 entries), READ.
//   b3 gOut    : 3 int Q16.16 per sample {acos, sin, cos}, WRITE.
//   b4 gParams : { sampleCount, enabled, _, _ }, READ.

#define HF_IK_THREADS 64

// Q16.16 constants (MUST match engine/anim/ik.h exactly).
static const int HF_IK_KFRAC   = 16;
static const int HF_IK_KONE     = 65536;       // 1.0 in Q16.16
static const int HF_IK_TWOPI    = 411775;      // round(2*pi * 65536)
static const int HF_IK_ACOSBINS = 1024;
static const int HF_IK_TRIGBINS = 256;

struct Params {
    int4 cfg;   // x=sampleCount, y=enabled, z=_, w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<int>    gAcos   : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<int>    gSin    : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<int>    gCos    : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<int>    gOut    : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<Params> gParams : register(u4);

// FxAcosLut — VERBATIM engine/anim/ik.h. x a Q16.16 cosine in [-kOne, kOne] -> angle Q16.16 radians [0,pi].
int FxAcosLut(int x) {
    if (x <= -HF_IK_KONE) return gAcos[0];
    if (x >=  HF_IK_KONE) return gAcos[HF_IK_ACOSBINS];
    int scaled = (x + HF_IK_KONE) * HF_IK_ACOSBINS;   // int32-safe (<= 2^25)
    int i = scaled >> 17;                              // 2*kOne == 1<<17
    if (i >= HF_IK_ACOSBINS) i = HF_IK_ACOSBINS - 1;
    int frac = (scaled & 0x1FFFF) >> 1;                // Q16.16 fraction
    int a = gAcos[i];
    int b = gAcos[i + 1];
    return a + (((b - a) * frac) >> HF_IK_KFRAC);
}

// FxReducePhaseBin — VERBATIM engine/anim/ik.h. theta Q16.16 radians -> bin index + Q16.16 sub-bin frac
// over kTrigBins bins of [0, 2pi). Pure int32 (modulo-reduce + nested base-256 frac).
void FxReducePhaseBin(int theta, int bins, out int outI, out int outFrac) {
    int th = theta % HF_IK_TWOPI;
    if (th < 0) th += HF_IK_TWOPI;
    int prod = th * bins;                              // int32-safe (<= kTwoPi*256)
    int i = prod / HF_IK_TWOPI;
    if (i >= bins) i = bins - 1;
    int rem = prod - i * HF_IK_TWOPI;
    int hiNum = rem << 8;                              // int32-safe
    int hi    = hiNum / HF_IK_TWOPI;
    int rem2  = hiNum - hi * HF_IK_TWOPI;
    int lo    = (rem2 << 8) / HF_IK_TWOPI;
    outI    = i;
    outFrac = (hi << 8) | lo;
}

int FxSinLut(int theta) {
    int i, frac;
    FxReducePhaseBin(theta, HF_IK_TRIGBINS, i, frac);
    int a = gSin[i];
    int b = gSin[i + 1];
    return a + (((b - a) * frac) >> HF_IK_KFRAC);
}

int FxCosLut(int theta) {
    int i, frac;
    FxReducePhaseBin(theta, HF_IK_TRIGBINS, i, frac);
    int a = gCos[i];
    int b = gCos[i + 1];
    return a + (((b - a) * frac) >> HF_IK_KFRAC);
}

[numthreads(HF_IK_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int sampleCount = gParams[0].cfg.x;
    int enabled     = gParams[0].cfg.y;
    int t = (int)gid.x;
    if (t >= sampleCount) return;

    if (enabled == 0) {
        gOut[t * 3 + 0] = 0;
        gOut[t * 3 + 1] = 0;
        gOut[t * 3 + 2] = 0;
        return;
    }

    // The deterministic sweep input for sample t (matches the host exactly — PURE INT32, NO int64, NO
    // float). The sweep is sized so the host caps sampleCount so that kTwoPi*(N-1) stays well under
    // INT32_MAX (N=512 -> kTwoPi*511 ~= 2.1e8 < 2.1e9), keeping the whole shader MSL-native:
    //   acos input xcos = -kOne + (2*kOne)*t/(N-1)   (sweeps the cosine domain [-1, 1]).
    //   trig input theta = kTwoPi*t/(N-1)            (sweeps [0, 2pi)).
    int denom = sampleCount > 1 ? (sampleCount - 1) : 1;
    int xcos  = -HF_IK_KONE + ((2 * HF_IK_KONE) * t) / denom;   // int32-safe (capped N)
    int theta = (HF_IK_TWOPI * t) / denom;                      // int32-safe (capped N)

    gOut[t * 3 + 0] = FxAcosLut(xcos);
    gOut[t * 3 + 1] = FxSinLut(theta);
    gOut[t * 3 + 2] = FxCosLut(theta);
}
