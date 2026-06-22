// Slice IK2 — Deterministic IK Control-Rig: THE TWO-BONE LAW-OF-COSINES LIMB SOLVE compute (the 2nd slice
// of FLAGSHIP #32). ONE thread per TARGET solves engine/anim/ik.h::TwoBoneSolve and writes the two LOCAL
// bone FxQuats (8 ints: qUpper.xyzw, qLower.xyzw) to gOut. Each thread reads the host-baked Q16.16 angle
// LUTs (acos/sin/cos, uploaded as read-only SSBOs — the SAME tables engine/anim/ik.h builds + snaps once)
// and runs the law-of-cosines solve copied VERBATIM from ik.h. The per-target writes are independent ->
// order-independent, race-free, NO atomics.
//
// THE int64 REALITY (VULKAN-SPIR-V-ONLY): TwoBoneSolve uses FxLength/fxdiv (int64) for the reach `d` + the
// cosine ratios, so this shader is in the Vulkan compile list ONLY (DXC compiles int64; glslc cannot lower
// it to MSL) — NOT in the Metal hf_gen_msl list. The Metal --ik2-twobone showcase runs the CPU
// TwoBoneSolve over the same targets -> byte-identical to this GPU result by construction (the Vulkan side
// carries the GPU==CPU memcmp proof). UNLIKE IK1's ik_angle.comp (pure int32, MSL-native): the law-of-
// cosines reach + ratios force int64. The acos/sin/cos LUT lookups themselves are int32 (copied VERBATIM
// from ik.h), but the surrounding solve is int64.
//
// enabled==0 -> every thread writes 0 (the disabled-path no-op; gOut stays the cleared all-zero upload).
//
// Buffers (storage, bound at compute bindings 0..5):
//   b0 gAcos    : the acos LUT (kAcosBins+1 int Q16.16 entries), READ.
//   b1 gSin     : the sin  LUT (kTrigBins+1 int Q16.16 entries), READ.
//   b2 gCos     : the cos  LUT (kTrigBins+1 int Q16.16 entries), READ.
//   b3 gTargets : 3 int Q16.16 per target {tx,ty,tz} (root + pole + lengths come from gParams), READ.
//   b4 gOut     : 8 int Q16.16 per target {qU.xyzw, qL.xyzw}, WRITE.
//   b5 gParams  : { targetCount, enabled, lenUpper, lenLower; rootX, rootY, rootZ, _; poleX, poleY, poleZ, _ }.

#define HF_IK_FRAC     16
#define HF_IK_ONE      65536       // 1.0 in Q16.16
#define HF_IK_TWOPI    411775      // round(2*pi * 65536)
#define HF_IK_PI       205887      // round(pi   * 65536)
#define HF_IK_ACOSBINS 1024
#define HF_IK_TRIGBINS 256

struct Params {
    int4 cfg;    // x=targetCount, y=enabled, z=lenUpper, w=lenLower
    int4 root;   // x,y,z=root pos (Q16.16), w=_
    int4 pole;   // x,y,z=pole (Q16.16), w=_
};

[[vk::binding(0, 0)]] RWStructuredBuffer<int>    gAcos    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<int>    gSin     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<int>    gCos     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<int>    gTargets : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<int>    gOut     : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<Params> gParams  : register(u5);

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
// FxNormalize3 -> writes the unit vector (or the (0,kOne,0) fallback for a zero vector) into ox/oy/oz.
void FxNormalize3(int vx, int vy, int vz, out int ox, out int oy, out int oz) {
    int len = FxLength3(vx, vy, vz);
    if (len == 0) { ox = 0; oy = HF_IK_ONE; oz = 0; return; }
    ox = fxdiv(vx, len); oy = fxdiv(vy, len); oz = fxdiv(vz, len);
}
// FxCross -> a×b.
void FxCross3(int ax, int ay, int az, int bx, int by, int bz, out int ox, out int oy, out int oz) {
    ox = fxmul(ay, bz) - fxmul(az, by);
    oy = fxmul(az, bx) - fxmul(ax, bz);
    oz = fxmul(ax, by) - fxmul(ay, bx);
}

// ---- VERBATIM ik.h angle LUTs (int32 index + lerp over the uploaded tables) -------------------------
int FxAcosLut(int x) {
    if (x <= -HF_IK_ONE) return gAcos[0];
    if (x >=  HF_IK_ONE) return gAcos[HF_IK_ACOSBINS];
    int scaled = (x + HF_IK_ONE) * HF_IK_ACOSBINS;
    int i = scaled >> 17;
    if (i >= HF_IK_ACOSBINS) i = HF_IK_ACOSBINS - 1;
    int frac = (scaled & 0x1FFFF) >> 1;
    int a = gAcos[i];
    int b = gAcos[i + 1];
    return a + (((b - a) * frac) >> HF_IK_FRAC);
}
void FxReducePhaseBin(int theta, int bins, out int outI, out int outFrac) {
    int th = theta % HF_IK_TWOPI;
    if (th < 0) th += HF_IK_TWOPI;
    int prod = th * bins;
    int i = prod / HF_IK_TWOPI;
    if (i >= bins) i = bins - 1;
    int rem = prod - i * HF_IK_TWOPI;
    int hiNum = rem << 8;
    int hi    = hiNum / HF_IK_TWOPI;
    int rem2  = hiNum - hi * HF_IK_TWOPI;
    int lo    = (rem2 << 8) / HF_IK_TWOPI;
    outI = i; outFrac = (hi << 8) | lo;
}
int FxSinLut(int theta) {
    int i, frac; FxReducePhaseBin(theta, HF_IK_TRIGBINS, i, frac);
    int a = gSin[i]; int b = gSin[i + 1];
    return a + (((b - a) * frac) >> HF_IK_FRAC);
}
int FxCosLut(int theta) {
    int i, frac; FxReducePhaseBin(theta, HF_IK_TRIGBINS, i, frac);
    int a = gCos[i]; int b = gCos[i + 1];
    return a + (((b - a) * frac) >> HF_IK_FRAC);
}

// QuatFromAxisAngle(n, theta) -> (n·sin(theta/2), cos(theta/2)). The FxQuat w-last layout. VERBATIM ik.h.
void QuatFromAxisAngle(int nx, int ny, int nz, int theta,
                       out int qx, out int qy, out int qz, out int qw) {
    int half = theta >> 1;
    int s = FxSinLut(half);
    int c = FxCosLut(half);
    qx = fxmul(nx, s); qy = fxmul(ny, s); qz = fxmul(nz, s); qw = c;
}

// FxQuatNormalize -> the unit quaternion (the |q|≈kOne contract). VERBATIM fpx.h.
void FxQuatNormalize(int x, int y, int z, int w, out int ox, out int oy, out int oz, out int ow) {
    int64_t sx = (int64_t)x * (int64_t)x;
    int64_t sy = (int64_t)y * (int64_t)y;
    int64_t sz = (int64_t)z * (int64_t)z;
    int64_t sw = (int64_t)w * (int64_t)w;
    int len = (int)FxISqrt(sx + sy + sz + sw);
    if (len == 0) { ox = 0; oy = 0; oz = 0; ow = HF_IK_ONE; return; }
    ox = fxdiv(x, len); oy = fxdiv(y, len); oz = fxdiv(z, len); ow = fxdiv(w, len);
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int targetCount = gParams[0].cfg.x;
    int enabled     = gParams[0].cfg.y;
    int lenUpper    = gParams[0].cfg.z;
    int lenLower    = gParams[0].cfg.w;
    int rootX = gParams[0].root.x, rootY = gParams[0].root.y, rootZ = gParams[0].root.z;
    int poleX = gParams[0].pole.x, poleY = gParams[0].pole.y, poleZ = gParams[0].pole.z;

    int t = (int)gid.x;
    if (t >= targetCount) return;

    if (enabled == 0) {
        for (int k = 0; k < 8; ++k) gOut[t * 8 + k] = 0;
        return;
    }

    // The target for this thread.
    int tx = gTargets[t * 3 + 0];
    int ty = gTargets[t * 3 + 1];
    int tz = gTargets[t * 3 + 2];

    // === TwoBoneSolve (VERBATIM engine/anim/ik.h) ===
    int ex = tx - rootX, ey = ty - rootY, ez = tz - rootZ;
    int d = FxLength3(ex, ey, ez);

    int dMin = (lenUpper > lenLower) ? (lenUpper - lenLower) : (lenLower - lenUpper);
    int dMax = lenUpper + lenLower;
    if (d < dMin) d = dMin;
    if (d > dMax) d = dMax;

    int64_t lU64 = (int64_t)lenUpper;
    int64_t lL64 = (int64_t)lenLower;
    int64_t d64  = (int64_t)d;
    int lU2 = (int)((lU64 * lU64) >> HF_IK_FRAC);
    int lL2 = (int)((lL64 * lL64) >> HF_IK_FRAC);
    int d2  = (int)((d64 * d64) >> HF_IK_FRAC);
    int twoLUd  = (int)((((int64_t)2 * lU64) * d64)  >> HF_IK_FRAC);
    int twoLUlL = (int)((((int64_t)2 * lU64) * lL64) >> HF_IK_FRAC);

    int cosRoot  = fxdiv(lU2 + d2 - lL2, twoLUd);
    int cosElbow = fxdiv(lU2 + lL2 - d2, twoLUlL);
    if (cosRoot  >  HF_IK_ONE) cosRoot  =  HF_IK_ONE;
    if (cosRoot  < -HF_IK_ONE) cosRoot  = -HF_IK_ONE;
    if (cosElbow >  HF_IK_ONE) cosElbow =  HF_IK_ONE;
    if (cosElbow < -HF_IK_ONE) cosElbow = -HF_IK_ONE;

    int aRoot  = FxAcosLut(cosRoot);
    int aElbow = FxAcosLut(cosElbow);

    int axisX, axisY, axisZ;
    FxNormalize3(ex, ey, ez, axisX, axisY, axisZ);

    int crX, crY, crZ;
    FxCross3(axisX, axisY, axisZ, poleX, poleY, poleZ, crX, crY, crZ);
    if (FxLength3(crX, crY, crZ) == 0) {
        FxCross3(axisX, axisY, axisZ, 0, HF_IK_ONE, 0, crX, crY, crZ);
        if (FxLength3(crX, crY, crZ) == 0)
            FxCross3(axisX, axisY, axisZ, HF_IK_ONE, 0, 0, crX, crY, crZ);
    }
    int nX, nY, nZ;
    FxNormalize3(crX, crY, crZ, nX, nY, nZ);

    int qUx, qUy, qUz, qUw;
    QuatFromAxisAngle(nX, nY, nZ, aRoot, qUx, qUy, qUz, qUw);
    int nqUx, nqUy, nqUz, nqUw;
    FxQuatNormalize(qUx, qUy, qUz, qUw, nqUx, nqUy, nqUz, nqUw);

    // qLower = axis-angle(n, aElbow - pi) == axis-angle(-n, pi - aElbow) — the NEGATED-axis + POSITIVE-angle
    // form (pi - aElbow >= 0) so the LUT half-angle is non-negative. VERBATIM ik.h.
    int qLx, qLy, qLz, qLw;
    QuatFromAxisAngle(-nX, -nY, -nZ, HF_IK_PI - aElbow, qLx, qLy, qLz, qLw);
    int nqLx, nqLy, nqLz, nqLw;
    FxQuatNormalize(qLx, qLy, qLz, qLw, nqLx, nqLy, nqLz, nqLw);

    gOut[t * 8 + 0] = nqUx; gOut[t * 8 + 1] = nqUy; gOut[t * 8 + 2] = nqUz; gOut[t * 8 + 3] = nqUw;
    gOut[t * 8 + 4] = nqLx; gOut[t * 8 + 5] = nqLy; gOut[t * 8 + 6] = nqLz; gOut[t * 8 + 7] = nqLw;
}
