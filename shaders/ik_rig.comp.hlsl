// Slice IK4 — Deterministic IK Control-Rig: IK ON THE SKELETON (the FK-pose -> IK-corrected palette bridge,
// the 4th slice of FLAGSHIP #32). ONE thread per TARGET runs the WHOLE engine/anim/ik.h::SolveRigToTargets
// for ONE rig chain toward that thread's world target — FK the base local pose to world joint positions,
// FABRIK the chain's world positions toward the target, convert the corrected world positions back to
// corrected LOCAL joint rotations (LookAtRotation + FxQuatMul + QConj), and write the corrected per-joint
// LOCAL FxQuats (kIkMaxJoints * 4 ints) into gOut. Each thread reads the SAME base pose (parents + localT +
// localR, from gBase) + the SAME chain joint indices (gParams) + the host-baked acos/sin/cos LUTs; only the
// target (gTargets) varies per thread. The per-target writes are independent -> order-independent, race-free,
// NO atomics.
//
// THE int64 REALITY (VULKAN-SPIR-V-ONLY): the FK (FxRotate/FxQuatMul), the FABRIK solve (FxNormalize/
// FxLength), and the rotation-convert (LookAtRotation/FxQuatMul/FxQuatNormalize) all use int64, so this
// shader is in the Vulkan compile list ONLY (DXC compiles int64; glslc cannot lower it to MSL) — NOT in the
// Metal hf_gen_msl list. The Metal --ik4-rig showcase runs the CPU SolveRigToTargets over the same targets
// -> byte-identical to this GPU result by construction (the Vulkan side carries the GPU==CPU memcmp proof).
// The bodies below are copied VERBATIM from engine/anim/ik.h (+ fpx.h quat helpers).
//
// enabled==0 -> every thread writes 0 (the disabled-path no-op; gOut stays the cleared all-zero upload).
//
// Buffers (storage, bound at compute bindings 0..6):
//   b0 gAcos    : the acos LUT (kAcosBins+1 int Q16.16 entries), READ.
//   b1 gSin     : the sin  LUT (kTrigBins+1 int Q16.16 entries), READ.
//   b2 gCos     : the cos  LUT (kTrigBins+1 int Q16.16 entries), READ.
//   b3 gTargets : 3 int Q16.16 per target {tx,ty,tz} (the per-thread end-effector goal), READ.
//   b4 gBase    : one BaseJoint per skeleton joint {parent, _pad0, _pad1, _pad2; tx,ty,tz,_; rx,ry,rz,rw}, READ.
//   b5 gOut     : kIkMaxJoints*4 int Q16.16 per target {localR[j].xyzw}, WRITE.
//   b6 gParams  : { targetCount, enabled, jointCount, chainCount; iters,_,_,_; chain0[4]; chain1[4] }.

#define HF_IK_FRAC     16
#define HF_IK_ONE      65536       // 1.0 in Q16.16
#define HF_IK_TWOPI    411775      // round(2*pi * 65536)
#define HF_IK_PI       205887      // round(pi   * 65536)
#define HF_IK_ACOSBINS 1024
#define HF_IK_TRIGBINS 256
#define HF_IK_MAXJ     16          // kIkMaxJoints
#define HF_IK_MAXB     8           // kIkMaxBones

// One skeleton joint's base data, std430 (48 bytes, NO padding holes): parent + 3 pad, localT.xyz + pad,
// localR.xyzw. The host packs the IkBasePose into this flat array.
struct BaseJoint {
    int4 par;    // x=parent, yzw=_
    int4 t;      // x,y,z=localT (Q16.16), w=_
    int4 r;      // x,y,z,w=localR (Q16.16)
};

struct Params {
    int4 cfg;     // x=targetCount, y=enabled, z=jointCount, w=chainCount
    int4 misc;    // x=iters, yzw=_
    int4 chain0;  // chain joint indices [0..3]
    int4 chain1;  // chain joint indices [4..7]
};

[[vk::binding(0, 0)]] RWStructuredBuffer<int>       gAcos    : register(u0);
[[vk::binding(1, 0)]] RWStructuredBuffer<int>       gSin     : register(u1);
[[vk::binding(2, 0)]] RWStructuredBuffer<int>       gCos     : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<int>       gTargets : register(u3);
[[vk::binding(4, 0)]] RWStructuredBuffer<BaseJoint> gBase    : register(u4);
[[vk::binding(5, 0)]] RWStructuredBuffer<int>       gOut     : register(u5);
[[vk::binding(6, 0)]] RWStructuredBuffer<Params>    gParams  : register(u6);

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
void FxNormalize3(int vx, int vy, int vz, out int ox, out int oy, out int oz) {
    int len = FxLength3(vx, vy, vz);
    if (len == 0) { ox = 0; oy = HF_IK_ONE; oz = 0; return; }
    ox = fxdiv(vx, len); oy = fxdiv(vy, len); oz = fxdiv(vz, len);
}
void FxCross3(int ax, int ay, int az, int bx, int by, int bz, out int ox, out int oy, out int oz) {
    ox = fxmul(ay, bz) - fxmul(az, by);
    oy = fxmul(az, bx) - fxmul(ax, bz);
    oz = fxmul(ax, by) - fxmul(ay, bx);
}
int FxDotV3(int ax, int ay, int az, int bx, int by, int bz) {
    return fxmul(ax, bx) + fxmul(ay, by) + fxmul(az, bz);
}

// ---- VERBATIM fpx.h quaternion helpers --------------------------------------------------------------
// FxQuatMul(a,b) -> Hamilton product, w-last layout.
void FxQuatMul(int ax, int ay, int az, int aw, int bx, int by, int bz, int bw,
               out int ox, out int oy, out int oz, out int ow) {
    ow = fxmul(aw, bw) - fxmul(ax, bx) - fxmul(ay, by) - fxmul(az, bz);
    ox = fxmul(aw, bx) + fxmul(ax, bw) + fxmul(ay, bz) - fxmul(az, by);
    oy = fxmul(aw, by) - fxmul(ax, bz) + fxmul(ay, bw) + fxmul(az, bx);
    oz = fxmul(aw, bz) + fxmul(ax, by) - fxmul(ay, bx) + fxmul(az, bw);
}
void FxQuatNormalize(int x, int y, int z, int w, out int ox, out int oy, out int oz, out int ow) {
    int64_t sx = (int64_t)x * (int64_t)x;
    int64_t sy = (int64_t)y * (int64_t)y;
    int64_t sz = (int64_t)z * (int64_t)z;
    int64_t sw = (int64_t)w * (int64_t)w;
    int len = (int)FxISqrt(sx + sy + sz + sw);
    if (len == 0) { ox = 0; oy = 0; oz = 0; ow = HF_IK_ONE; return; }
    ox = fxdiv(x, len); oy = fxdiv(y, len); oz = fxdiv(z, len); ow = fxdiv(w, len);
}
// FxRotate(q, v) -> v' = v + 2*cross(u, cross(u,v) + q.w*v), u=q.xyz. VERBATIM fpx.h.
void FxRotate3(int qx, int qy, int qz, int qw, int vx, int vy, int vz,
               out int ox, out int oy, out int oz) {
    int c1x, c1y, c1z; FxCross3(qx, qy, qz, vx, vy, vz, c1x, c1y, c1z);
    int tx = c1x + fxmul(qw, vx);
    int ty = c1y + fxmul(qw, vy);
    int tz = c1z + fxmul(qw, vz);
    int c2x, c2y, c2z; FxCross3(qx, qy, qz, tx, ty, tz, c2x, c2y, c2z);
    ox = vx + 2 * c2x; oy = vy + 2 * c2y; oz = vz + 2 * c2z;
}

// ---- VERBATIM ik.h angle LUTs -----------------------------------------------------------------------
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
void QuatFromAxisAngle(int nx, int ny, int nz, int theta,
                       out int qx, out int qy, out int qz, out int qw) {
    int half = theta >> 1;
    int s = FxSinLut(half);
    int c = FxCosLut(half);
    qx = fxmul(nx, s); qy = fxmul(ny, s); qz = fxmul(nz, s); qw = c;
}

// LookAtRotation(fwd, to) -> the shortest-arc unit quaternion. VERBATIM ik.h.
void LookAtRotation(int fx_, int fy, int fz, int tx, int ty, int tz,
                    out int qx, out int qy, out int qz, out int qw) {
    int eps = 64;
    int c = FxDotV3(fx_, fy, fz, tx, ty, tz);
    if (c >  HF_IK_ONE) c =  HF_IK_ONE;
    if (c < -HF_IK_ONE) c = -HF_IK_ONE;
    if (c >= HF_IK_ONE - eps) { qx = 0; qy = 0; qz = 0; qw = HF_IK_ONE; return; }
    int angle = FxAcosLut(c);
    int crx, cry, crz; FxCross3(fx_, fy, fz, tx, ty, tz, crx, cry, crz);
    if (c <= -HF_IK_ONE + eps || FxLength3(crx, cry, crz) == 0) {
        FxCross3(fx_, fy, fz, 0, HF_IK_ONE, 0, crx, cry, crz);
        if (FxLength3(crx, cry, crz) == 0) FxCross3(fx_, fy, fz, HF_IK_ONE, 0, 0, crx, cry, crz);
        int ax, ay, az; FxNormalize3(crx, cry, crz, ax, ay, az);
        int rx, ry, rz, rw; QuatFromAxisAngle(ax, ay, az, HF_IK_PI, rx, ry, rz, rw);
        FxQuatNormalize(rx, ry, rz, rw, qx, qy, qz, qw);
        return;
    }
    int ax, ay, az; FxNormalize3(crx, cry, crz, ax, ay, az);
    int rx, ry, rz, rw; QuatFromAxisAngle(ax, ay, az, angle, rx, ry, rz, rw);
    FxQuatNormalize(rx, ry, rz, rw, qx, qy, qz, qw);
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int targetCount = gParams[0].cfg.x;
    int enabled     = gParams[0].cfg.y;
    int jointCount  = gParams[0].cfg.z;
    int chainCount  = gParams[0].cfg.w;
    int iters       = gParams[0].misc.x;

    int t = (int)gid.x;
    if (t >= targetCount) return;

    if (enabled == 0) {
        for (int k = 0; k < HF_IK_MAXJ * 4; ++k) gOut[t * HF_IK_MAXJ * 4 + k] = 0;
        return;
    }

    int base = t * HF_IK_MAXJ * 4;

    // chain joint indices.
    int chain[HF_IK_MAXB];
    chain[0] = gParams[0].chain0.x; chain[1] = gParams[0].chain0.y;
    chain[2] = gParams[0].chain0.z; chain[3] = gParams[0].chain0.w;
    chain[4] = gParams[0].chain1.x; chain[5] = gParams[0].chain1.y;
    chain[6] = gParams[0].chain1.z; chain[7] = gParams[0].chain1.w;

    // The per-joint base local pose + parents (from gBase) into local arrays.
    int parents[HF_IK_MAXJ];
    int lrx[HF_IK_MAXJ], lry[HF_IK_MAXJ], lrz[HF_IK_MAXJ], lrw[HF_IK_MAXJ];
    int ltx[HF_IK_MAXJ], lty[HF_IK_MAXJ], ltz[HF_IK_MAXJ];
    for (int j = 0; j < jointCount; ++j) {
        parents[j] = gBase[j].par.x;
        ltx[j] = gBase[j].t.x; lty[j] = gBase[j].t.y; ltz[j] = gBase[j].t.z;
        lrx[j] = gBase[j].r.x; lry[j] = gBase[j].r.y; lrz[j] = gBase[j].r.z; lrw[j] = gBase[j].r.w;
    }

    // === (1) FK: forward-accumulate world rotations + positions (VERBATIM FkWorldPositions). ===
    int grx[HF_IK_MAXJ], gry[HF_IK_MAXJ], grz[HF_IK_MAXJ], grw[HF_IK_MAXJ];
    int gpx[HF_IK_MAXJ], gpy[HF_IK_MAXJ], gpz[HF_IK_MAXJ];
    for (int j2 = 0; j2 < jointCount; ++j2) {
        int p = parents[j2];
        if (p < 0) {
            grx[j2] = lrx[j2]; gry[j2] = lry[j2]; grz[j2] = lrz[j2]; grw[j2] = lrw[j2];
            gpx[j2] = ltx[j2]; gpy[j2] = lty[j2]; gpz[j2] = ltz[j2];
        } else {
            FxQuatMul(grx[p], gry[p], grz[p], grw[p], lrx[j2], lry[j2], lrz[j2], lrw[j2],
                      grx[j2], gry[j2], grz[j2], grw[j2]);
            int twx, twy, twz;
            FxRotate3(grx[p], gry[p], grz[p], grw[p], ltx[j2], lty[j2], ltz[j2], twx, twy, twz);
            gpx[j2] = gpx[p] + twx; gpy[j2] = gpy[p] + twy; gpz[j2] = gpz[p] + twz;
        }
    }

    // Seed the output with the EXACT base local rotations (un-IK'd joints stay byte-identical).
    for (int j3 = 0; j3 < jointCount; ++j3) {
        gOut[base + j3 * 4 + 0] = lrx[j3];
        gOut[base + j3 * 4 + 1] = lry[j3];
        gOut[base + j3 * 4 + 2] = lrz[j3];
        gOut[base + j3 * 4 + 3] = lrw[j3];
    }

    if (chainCount < 2) return;

    // === (2) FABRIK over the chain world positions toward the target. ===
    int tx = gTargets[t * 3 + 0];
    int ty = gTargets[t * 3 + 1];
    int tz = gTargets[t * 3 + 2];

    int cpx[HF_IK_MAXB], cpy[HF_IK_MAXB], cpz[HF_IK_MAXB];
    int clen[HF_IK_MAXB - 1];
    for (int i = 0; i < chainCount; ++i) {
        int jc = chain[i];
        cpx[i] = gpx[jc]; cpy[i] = gpy[jc]; cpz[i] = gpz[jc];
    }
    for (int i2 = 0; i2 + 1 < chainCount; ++i2) {
        clen[i2] = FxLength3(cpx[i2 + 1] - cpx[i2], cpy[i2 + 1] - cpy[i2], cpz[i2 + 1] - cpz[i2]);
    }
    int rx0 = cpx[0], ry0 = cpy[0], rz0 = cpz[0];   // the chain root (fixed)
    for (int it = 0; it < iters; ++it) {
        // BACKWARD: pin pos[K-1] = target, reach back toward the root.
        cpx[chainCount - 1] = tx; cpy[chainCount - 1] = ty; cpz[chainCount - 1] = tz;
        for (int i = chainCount - 2; i >= 0; --i) {
            int ux, uy, uz;
            FxNormalize3(cpx[i] - cpx[i + 1], cpy[i] - cpy[i + 1], cpz[i] - cpz[i + 1], ux, uy, uz);
            cpx[i] = cpx[i + 1] + fxmul(clen[i], ux);
            cpy[i] = cpy[i + 1] + fxmul(clen[i], uy);
            cpz[i] = cpz[i + 1] + fxmul(clen[i], uz);
        }
        // FORWARD: pin pos[0] = root, reach forward toward the end-effector.
        cpx[0] = rx0; cpy[0] = ry0; cpz[0] = rz0;
        for (int i = 1; i < chainCount; ++i) {
            int ux, uy, uz;
            FxNormalize3(cpx[i] - cpx[i - 1], cpy[i] - cpy[i - 1], cpz[i] - cpz[i - 1], ux, uy, uz);
            cpx[i] = cpx[i - 1] + fxmul(clen[i - 1], ux);
            cpy[i] = cpy[i - 1] + fxmul(clen[i - 1], uy);
            cpz[i] = cpz[i - 1] + fxmul(clen[i - 1], uz);
        }
    }

    // === (3) corrected world positions -> corrected LOCAL rotations on the chain joints. ===
    // CRUX (VERBATIM ik.h): the local rotation is expressed in the CORRECTED PARENT world frame — chain joint
    // 0's parent is outside the chain (base FK rot), every later chain joint's parent is the PREVIOUS chain
    // joint (its corrected world rotation). Track prevCorrWorldRot forward.
    int pcwx = 0, pcwy = 0, pcwz = 0, pcww = HF_IK_ONE;   // corrected world rotation of the prev chain joint
    for (int i = 0; i + 1 < chainCount; ++i) {
        int c  = chain[i];
        int cn = chain[i + 1];
        int bdx, bdy, bdz;
        FxNormalize3(gpx[cn] - gpx[c], gpy[cn] - gpy[c], gpz[cn] - gpz[c], bdx, bdy, bdz);
        int cdx, cdy, cdz;
        FxNormalize3(cpx[i + 1] - cpx[i], cpy[i + 1] - cpy[i], cpz[i + 1] - cpz[i], cdx, cdy, cdz);
        int dqx, dqy, dqz, dqw;
        LookAtRotation(bdx, bdy, bdz, cdx, cdy, cdz, dqx, dqy, dqz, dqw);
        // corrWorldRot = normalize(deltaWorld * fkRot[c]).
        int cwx, cwy, cwz, cww;
        FxQuatMul(dqx, dqy, dqz, dqw, grx[c], gry[c], grz[c], grw[c], cwx, cwy, cwz, cww);
        FxQuatNormalize(cwx, cwy, cwz, cww, cwx, cwy, cwz, cww);
        // the corrected parent world frame: i==0 -> base FK rot of c's skeleton parent; else prevCorrWorldRot.
        int pwx, pwy, pwz, pww;
        if (i == 0) {
            int p = parents[c];
            if (p >= 0) { pwx = grx[p]; pwy = gry[p]; pwz = grz[p]; pww = grw[p]; }
            else        { pwx = 0; pwy = 0; pwz = 0; pww = HF_IK_ONE; }
        } else {
            pwx = pcwx; pwy = pcwy; pwz = pcwz; pww = pcww;
        }
        // localR = normalize(QConj(parentWorldRot) * corrWorldRot). QConj: negate xyz.
        int lx, ly, lz, lw;
        FxQuatMul(-pwx, -pwy, -pwz, pww, cwx, cwy, cwz, cww, lx, ly, lz, lw);
        FxQuatNormalize(lx, ly, lz, lw, lx, ly, lz, lw);
        gOut[base + c * 4 + 0] = lx;
        gOut[base + c * 4 + 1] = ly;
        gOut[base + c * 4 + 2] = lz;
        gOut[base + c * 4 + 3] = lw;
        pcwx = cwx; pcwy = cwy; pcwz = cwz; pcww = cww;
    }
}
