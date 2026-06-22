// Slice GI4 — Deterministic Lumen-class GI: INTEGER MULTI-BOUNCE FEEDBACK (light that bounces, FLAGSHIP #29).
// ONE thread per PROBE. Each thread loops the 16 baked Fibonacci-sphere directions; for each direction it
// opens a RayQuery over the TLAS (gTlas, binding 5) draining EVERY candidate procedural primitive (the GI1
// gi_probe_trace.comp contract VERBATIM — NO commit, NO tMax shrink), runs the FROZEN fx Q16.16
// IntersectSphere/IntersectAabb on each candidate, folds into the running closest by the total order
// (t, primIndex). If it HIT, a SECOND RayQuery (the SHADOW ray) sets occluded; and — the ONE GI4 addition —
// the hit's shade adds an INDIRECT term = the GI3 FxInterpolateIrradiance of the bound PREVIOUS SH buffer
// (gPrevSH, binding 4) at the hit point for the hit normal (the trilinear 8-corner blend + FxSHEvaluate,
// copied from gi_interp.comp / gi.h VERBATIM). The shade is the inline ShadeHitGI (the ShadeHitShadowed
// direct diffuse + albedo*indirect), the result unpacked to a GiRadiance and written to gRadiance[probe*16 +
// dir] — bit-identical to the CPU gi::ShadeHitGI / BounceProbes one-iteration re-trace (memcmp).
//
// THE HOST drives the K-iteration loop: dispatch gi_bounce.comp reading SH_{k-1} (gPrevSH), then
// gi_sh_encode.comp to make SH_k. When gPrevSH is all-zero (iteration 1, the host binds a zeroed SH buffer),
// FxInterpolateIrradiance reconstructs {0,0,0} so the indirect term is a literal +0 -> gi_bounce.comp is
// BYTE-IDENTICAL to gi_probe_trace.comp (the no-op contract: K==1 == GI1+GI2 single-bounce).
//
// THE DETERMINISM CONTRACT (do NOT violate): the HW BVH is ONLY a candidate-AABB GENERATOR. We NEVER read the
// query's float t / hit attributes; correctness is OWNED by OUR fx math + (t,primIndex) min (closest) / the
// boolean OR (shadow). The driver's float AABB-overlap only WIDENS the candidate set -> the HW radiance
// buffer is byte-identical to the no-cull CPU reference, vendor-independent.
//
// VULKAN-SPIR-V-ONLY: HLSL inline RayQuery (DXC -> SPIR-V RayQueryKHR, which glslc/spirv-cross can't lower to
// MSL) AND int64 fx math (the SH blend + reconstruction Q32.32/Q48.48 accumulators) -> compiled by DXC
// (cs_6_5 + SPV_KHR_ray_query, the :csrq stage) on the Vulkan path ONLY; NOT in the Metal hf_gen_msl list. On
// Metal the --gi4-bounce showcase runs the CPU gi::BounceProbes (the SAME bit-exact reference) -> byte-
// identical by construction. The int64 + RayQuery + the candidate drain are the gi_probe_trace.comp +
// gi_interp.comp conventions fused.

#define HF_GI_THREADS 64
#define HF_RT_FRAC 16             // MUST match rtrace.h::kFrac (fpx.h::kFrac)
static const int HF_RT_ONE = 1 << HF_RT_FRAC;       // 1.0 in Q16.16
static const int HF_RT_NOHIT = 0x7FFFFFFF;          // kRtNoHit (INT32_MAX)
static const uint HF_RT_MISS = 0xFFFFFFFFu;         // kRtMiss
static const uint HF_RT_AABB_TAG = 0x800000u;       // custom-index bit 23 -> the prim is an AABB
static const int HF_GI_RAYS = 16;                   // kGiRaysPerProbe
static const int HF_RT_SHADOW_EPS  = HF_RT_ONE / 256;   // kRtShadowEps (anti-acne surface offset)
static const int HF_RT_SHADOW_MINT = HF_RT_ONE / 1024;  // kRtShadowMinT (self-hit guard)

// The 16 baked Fibonacci-sphere directions — IDENTICAL Q16.16 bits to gi.h::kGiProbeDirs.
static const int3 HF_GI_DIRS[16] = {
    int3(   22806,       0,   61440 ),
    int3(  -28171,   25807,   53248 ),
    int3(    4161,  -47409,   45056 ),
    int3(   32968,   43001,   36864 ),
    int3(  -58030,  -10265,   28672 ),
    int3(   52527,  -33413,   20480 ),
    int3(  -16712,   62167,   12288 ),
    int3(  -30147,  -58046,    4096 ),
    int3(   61439,   22437,   -4096 ),
    int3(  -59504,   24562,  -12288 ),
    int3(   26386,  -56385,  -20480 ),
    int3(   17637,   56230,  -28672 ),
    int3(  -46881,  -27169,  -36864 ),
    int3(   46481,  -10219,  -45056 ),
    int3(  -21973,   31254,  -53248 ),
    int3(   -2931,  -22616,  -61440 ),
};

// std430 mirrors of the rtrace.h POD types. fx == int (Q16.16). IDENTICAL to gi_probe_trace.comp.
struct GpuSphere { int cx, cy, cz; int radius; uint primIndex; uint _pad0, _pad1, _pad2; };  // 32 B
struct GpuAabb   { int lox, loy, loz; int hix, hiy, hiz; uint primIndex; uint _pad0; };       // 32 B

// std430 FxProbeSH mirror (gi.h::FxProbeSH): 9*3 = 27 Q16.16 coeff ints + 1 pad int = 28 ints (112 B).
struct FxProbeSH { int c[28]; };

// GI params (the probe grid + the light + background). std430. IDENTICAL to gi_probe_trace.comp's GiParams.
struct GiParams {
    int4 origin;    // x,y,z = grid.origin (Q16.16), w unused
    int4 grid;      // x = spacing (Q16.16), y = nx, z = ny, w = nz
    int4 light;     // x,y,z = lightDir unit (Q16.16), w unused
    uint4 counts;   // x = sphereCount, y = aabbCount, z = background (RGBA8), w = probeCount
};

// One radiance element (std430 16 B = gi.h::GiRadiance).
struct GiRad { int r, g, b, _pad; };

[[vk::binding(0, 0)]] StructuredBuffer<GpuSphere>   gSpheres  : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<GpuAabb>     gAabbs    : register(t1);
[[vk::binding(2, 0)]] StructuredBuffer<GiParams>    gParams   : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<GiRad>     gRadiance : register(u3);
[[vk::binding(4, 0)]] StructuredBuffer<FxProbeSH>   gPrevSH   : register(t3);  // SH_{k-1} (READ; zeroed @ k=1)
[[vk::binding(5, 0)]] RaytracingAccelerationStructure gTlas   : register(t4);

// The cosine-lobe reconstruction band factors B_l = 4*A_l (Q16.16), IDENTICAL to gi.h::kGiSHReconB*.
static const int64_t kReconB0 = 823550;   // 4*pi    in Q16.16
static const int64_t kReconB1 = 549033;   // 8*pi/3  in Q16.16
static const int64_t kReconB2 = 205887;   // pi      in Q16.16

// VERBATIM rtrace.h::fxmul / fxdiv / FxISqrt (int64, the gi_probe_trace.comp math copied EXACTLY).
int fxmul(int a, int b) { return (int)(((int64_t)a * (int64_t)b) >> HF_RT_FRAC); }
int fxdiv(int a, int b) { return (int)(((int64_t)a << HF_RT_FRAC) / (int64_t)b); }
int64_t fxisqrt(int64_t v) {
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
int fxdot(int3 a, int3 b) { return fxmul(a.x, b.x) + fxmul(a.y, b.y) + fxmul(a.z, b.z); }
int3 fxnormalize(int3 v) {
    int64_t sx = (int64_t)v.x * (int64_t)v.x;
    int64_t sy = (int64_t)v.y * (int64_t)v.y;
    int64_t sz = (int64_t)v.z * (int64_t)v.z;
    int len = (int)fxisqrt(sx + sy + sz);
    if (len == 0) return v;
    return int3(fxdiv(v.x, len), fxdiv(v.y, len), fxdiv(v.z, len));
}

struct Hit { int t; uint primIndex; int3 pos; int3 normal; };
Hit MissHit() { Hit h; h.t = HF_RT_NOHIT; h.primIndex = HF_RT_MISS; h.pos = int3(0,0,0); h.normal = int3(0,0,0); return h; }

// VERBATIM rtrace.h::IntersectSphere.
bool IntersectSphere(int3 ro, int3 rd, int3 center, int radius, uint primIndex, out Hit outHit) {
    outHit = MissHit();
    int3 oc = ro - center;
    int a = fxdot(rd, rd);
    if (a <= 0) return false;
    int half_b = fxdot(oc, rd);
    int c = fxdot(oc, oc) - fxmul(radius, radius);
    int64_t hb = (int64_t)half_b * (int64_t)half_b;
    int64_t ac = (int64_t)a * (int64_t)c;
    int64_t disc = hb - ac;
    if (disc < 0) return false;
    int sq = (int)fxisqrt(disc);
    int tNear = fxdiv(-half_b - sq, a);
    int tFar  = fxdiv(-half_b + sq, a);
    int t;
    if (tNear >= 0)      t = tNear;
    else if (tFar >= 0)  t = tFar;
    else                 return false;
    outHit.t = t;
    outHit.primIndex = primIndex;
    outHit.pos = ro + int3(fxmul(rd.x, t), fxmul(rd.y, t), fxmul(rd.z, t));
    outHit.normal = fxnormalize(outHit.pos - center);
    return true;
}

// VERBATIM rtrace.h::IntersectAabb.
bool IntersectAabb(int3 ro, int3 rd, int3 lo, int3 hi, uint primIndex, out Hit outHit) {
    outHit = MissHit();
    int o[3]  = { ro.x, ro.y, ro.z };
    int d[3]  = { rd.x, rd.y, rd.z };
    int lov[3] = { lo.x, lo.y, lo.z };
    int hiv[3] = { hi.x, hi.y, hi.z };

    int tEnter = (int)0x80000000;
    int tExit  = HF_RT_NOHIT;
    int enterAxis = 0;
    int enterSign = 0;

    [unroll] for (int ax = 0; ax < 3; ++ax) {
        if (d[ax] == 0) {
            if (o[ax] < lov[ax] || o[ax] > hiv[ax]) return false;
            continue;
        }
        int t1 = fxdiv(lov[ax] - o[ax], d[ax]);
        int t2 = fxdiv(hiv[ax] - o[ax], d[ax]);
        int tNear, tFar, nearSign;
        if (t1 <= t2) { tNear = t1; tFar = t2; nearSign = -1; }
        else          { tNear = t2; tFar = t1; nearSign = +1; }
        if (tNear > tEnter) { tEnter = tNear; enterAxis = ax; enterSign = nearSign; }
        if (tFar  < tExit)  { tExit = tFar; }
        if (tEnter > tExit) return false;
    }
    if (tExit < 0) return false;

    int t;
    int3 n = int3(0, 0, 0);
    if (tEnter >= 0) t = tEnter; else t = tExit;
    if (enterAxis == 0) n.x = enterSign * HF_RT_ONE;
    else if (enterAxis == 1) n.y = enterSign * HF_RT_ONE;
    else n.z = enterSign * HF_RT_ONE;

    outHit.t = t;
    outHit.primIndex = primIndex;
    outHit.pos = ro + int3(fxmul(rd.x, t), fxmul(rd.y, t), fxmul(rd.z, t));
    outHit.normal = n;
    return true;
}

// VERBATIM rtrace.h::AlbedoFor (Q16.16 albedo by primIndex).
int3 AlbedoFor(uint primIndex) {
    uint i = primIndex % 6u;
    if (i == 0u) return int3(HF_RT_ONE * 78 / 100, HF_RT_ONE * 30 / 100, HF_RT_ONE * 26 / 100);
    if (i == 1u) return int3(HF_RT_ONE * 28 / 100, HF_RT_ONE * 52 / 100, HF_RT_ONE * 80 / 100);
    if (i == 2u) return int3(HF_RT_ONE * 36 / 100, HF_RT_ONE * 70 / 100, HF_RT_ONE * 38 / 100);
    if (i == 3u) return int3(HF_RT_ONE * 82 / 100, HF_RT_ONE * 70 / 100, HF_RT_ONE * 28 / 100);
    if (i == 4u) return int3(HF_RT_ONE * 64 / 100, HF_RT_ONE * 40 / 100, HF_RT_ONE * 72 / 100);
    return int3(HF_RT_ONE * 60 / 100, HF_RT_ONE * 60 / 100, HF_RT_ONE * 62 / 100);
}

// VERBATIM gi.h::UnpackRadiance per-channel (RGBA8 byte -> Q16.16, integer divide).
int UnpackChannel(int byteVal) {
    if (byteVal < 0) byteVal = 0;
    if (byteVal > 255) byteVal = 255;
    return (int)(((int64_t)byteVal * (int64_t)HF_RT_ONE) / 255);
}

// ===== FxInterpolateIrradiance — the GI3 trilinear 8-corner SH blend + cosine-lobe reconstruction =====
// VERBATIM gi.h::FxInterpolateIrradiance (FxNearestProbes -> FxInterpolateSH -> FxSHEvaluate), int64 blend +
// reconstruction. `point`/`normal` are Q16.16; reads gPrevSH (probeCount). Returns the indirect GiRadiance.
int3 FxInterpolateIrradiance(int3 wpt, int3 nrm, int probeCount, int nx, int ny, int nz,
                             int ox, int oy, int oz, int spacing) {
    // ----- FxNearestProbes: floor-cell + frac + 8-corner index + partition-of-unity weights. -----
    int base[3]; int frac[3];
    int pv[3] = { wpt.x, wpt.y, wpt.z };
    int orig[3] = { ox, oy, oz };
    int dim[3] = { nx, ny, nz };
    if (spacing <= 0 || nx <= 0 || ny <= 0 || nz <= 0) return int3(0, 0, 0);
    [unroll] for (int ax = 0; ax < 3; ++ax) {
        if (dim[ax] <= 1) { base[ax] = 0; frac[ax] = 0; }
        else {
            int g = (int)((((int64_t)(pv[ax] - orig[ax])) << HF_RT_FRAC) / (int64_t)spacing);
            int b = g >> HF_RT_FRAC;                       // floor toward -inf
            if (b < 0) b = 0;
            if (b > dim[ax] - 2) b = dim[ax] - 2;
            int fr = (int)((int64_t)g - ((int64_t)b << HF_RT_FRAC));
            if (fr < 0) fr = 0;
            if (fr > HF_RT_ONE) fr = HF_RT_ONE;
            base[ax] = b; frac[ax] = fr;
        }
    }
    int wlo[3] = { HF_RT_ONE - frac[0], HF_RT_ONE - frac[1], HF_RT_ONE - frac[2] };
    int whi[3] = { frac[0], frac[1], frac[2] };

    int idx[8]; int wgt[8];
    int accumW = 0;
    [unroll] for (int c = 0; c < 8; ++c) {
        int sx = (c & 1), sy = ((c >> 1) & 1), sz = ((c >> 2) & 1);
        int cx = base[0] + sx; if (cx > nx - 1) cx = nx - 1;
        int cy = base[1] + sy; if (cy > ny - 1) cy = ny - 1;
        int cz = base[2] + sz; if (cz > nz - 1) cz = nz - 1;
        idx[c] = cx + cy * nx + cz * (nx * ny);            // ProbeFlatIndex (cx-major)
        int wx = sx ? whi[0] : wlo[0];
        int wy = sy ? whi[1] : wlo[1];
        int wz = sz ? whi[2] : wlo[2];
        if (c < 7) {
            int64_t p = (int64_t)wx * (int64_t)wy * (int64_t)wz;   // Q48.48
            int wc = (int)(p >> (2 * HF_RT_FRAC));                  // FLOOR narrow -> Q16.16
            wgt[c] = wc;
            accumW += wc;
        } else {
            wgt[c] = HF_RT_ONE - accumW;                  // exact leftover -> Σ w == kOne
        }
    }

    // ----- FxInterpolateSH: int64 8-corner blend, narrow ONCE. -----
    int blended[27];
    if (probeCount <= 0) {
        [unroll] for (int z0 = 0; z0 < 27; ++z0) blended[z0] = 0;
    } else {
        [unroll] for (int k = 0; k < 27; ++k) {
            int64_t acc = 0;
            [unroll] for (int c2 = 0; c2 < 8; ++c2)
                acc += (int64_t)wgt[c2] * (int64_t)gPrevSH[idx[c2]].c[k];   // Q32.32 — WIDE
            blended[k] = (int)(acc >> HF_RT_FRAC);                          // narrow ONCE -> Q16.16
        }
    }

    // ----- FxSHEvaluate: the integer cosine-lobe irradiance reconstruction. -----
    int x = nrm.x, y = nrm.y, z = nrm.z;
    const int kY00f = 18487, kY1f = 32021, kY2af = 71601, kY2bf = 20670, kY2cf = 35801;
    int Y[9];
    Y[0] = kY00f;
    Y[1] = fxmul(kY1f, y);
    Y[2] = fxmul(kY1f, z);
    Y[3] = fxmul(kY1f, x);
    Y[4] = fxmul(kY2af, fxmul(x, y));
    Y[5] = fxmul(kY2af, fxmul(y, z));
    Y[6] = fxmul(kY2bf, fxmul(fxmul(3 * HF_RT_ONE, z), z) - HF_RT_ONE);
    Y[7] = fxmul(kY2af, fxmul(x, z));
    Y[8] = fxmul(kY2cf, fxmul(x, x) - fxmul(y, y));

    int64_t B[9] = { kReconB0, kReconB1, kReconB1, kReconB1,
                     kReconB2, kReconB2, kReconB2, kReconB2, kReconB2 };

    int outc[3];
    [unroll] for (int ch = 0; ch < 3; ++ch) {
        int64_t acc = 0;   // Q48.48-ish (coeff*basis*B, three Q16.16 -> Q48.48)
        [unroll] for (int j = 0; j < 9; ++j)
            acc += (int64_t)blended[j * 3 + ch] * (int64_t)Y[j] * B[j];   // WIDE triple product
        int v = (int)(acc >> (2 * HF_RT_FRAC));   // narrow ONCE: Q48.48 >> 32 -> Q16.16
        outc[ch] = (v < 0) ? 0 : v;               // clamp >= 0
    }
    return int3(outc[0], outc[1], outc[2]);
}

// VERBATIM gi.h::ShadeHitGI — the direct shadowed shade + albedo*indirect, unpacked to a GiRadiance. A MISS
// -> UnpackRadiance(background). When indirect == {0,0,0} this is byte-identical to the unpacked
// gi_probe_trace.comp ShadeHitShadowed (the no-op contract).
GiRad ShadeHitGI(Hit hit, int3 lightDir, uint background, bool occluded, int3 indirect) {
    GiRad o;
    if (hit.primIndex == HF_RT_MISS) {
        o.r = UnpackChannel((int)((background >>  0) & 0xFFu));
        o.g = UnpackChannel((int)((background >>  8) & 0xFFu));
        o.b = UnpackChannel((int)((background >> 16) & 0xFFu));
        o._pad = 0;
        return o;
    }
    int ndl = fxdot(hit.normal, lightDir);
    if (ndl < 0) ndl = 0;
    int ambient = HF_RT_ONE * 18 / 100;
    int lambert = occluded ? 0 : fxmul(HF_RT_ONE - ambient, ndl);
    int diffuse = ambient + lambert;
    int3 alb = AlbedoFor(hit.primIndex);
    int albc[3] = { alb.x, alb.y, alb.z };
    int indc[3] = { indirect.x, indirect.y, indirect.z };
    int outc[3];
    [unroll] for (int i = 0; i < 3; ++i) {
        int lit = fxmul(albc[i], diffuse) + fxmul(albc[i], indc[i]);   // direct + indirect, Q16.16
        int byteVal = (int)(((int64_t)lit * 255) >> HF_RT_FRAC);       // [0,255], clamped in UnpackChannel
        outc[i] = UnpackChannel(byteVal);
    }
    o.r = outc[0]; o.g = outc[1]; o.b = outc[2]; o._pad = 0;
    return o;
}

float FxToFloat(int v) { return (float)v / (float)HF_RT_ONE; }

// Fold a candidate hit into the running closest by the total order (t, primIndex). VERBATIM TraceClosest.
void Consider(inout Hit best, Hit h) {
    if (h.primIndex == HF_RT_MISS) return;
    if (best.primIndex == HF_RT_MISS || h.t < best.t ||
        (h.t == best.t && h.primIndex < best.primIndex)) best = h;
}

// TraceClosest over gTlas (the candidate-drain): VERBATIM gi_probe_trace.comp.
Hit TraceClosestHW(int3 ro, int3 rd) {
    float3 fro = float3(FxToFloat(ro.x), FxToFloat(ro.y), FxToFloat(ro.z));
    float3 frd = float3(FxToFloat(rd.x), FxToFloat(rd.y), FxToFloat(rd.z));
    RayDesc ray;
    ray.Origin = fro; ray.Direction = frd; ray.TMin = 0.0f; ray.TMax = 1.0e30f;
    Hit best = MissHit();
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint tag = q.CandidateInstanceID();
            Hit h;
            if (tag & HF_RT_AABB_TAG) {
                uint idx = tag & ~HF_RT_AABB_TAG;
                GpuAabb b = gAabbs[idx];
                if (IntersectAabb(ro, rd, int3(b.lox, b.loy, b.loz), int3(b.hix, b.hiy, b.hiz),
                                  b.primIndex, h)) Consider(best, h);
            } else {
                GpuSphere s = gSpheres[tag];
                if (IntersectSphere(ro, rd, int3(s.cx, s.cy, s.cz), s.radius, s.primIndex, h))
                    Consider(best, h);
            }
        }
    }
    return best;
}

// TraceAnyHit over gTlas (the shadow query): VERBATIM gi_probe_trace.comp.
bool TraceAnyHitHW(int3 ro, int3 rd, int minT) {
    float3 fro = float3(FxToFloat(ro.x), FxToFloat(ro.y), FxToFloat(ro.z));
    float3 frd = float3(FxToFloat(rd.x), FxToFloat(rd.y), FxToFloat(rd.z));
    RayDesc ray;
    ray.Origin = fro; ray.Direction = frd; ray.TMin = 0.0f; ray.TMax = 1.0e30f;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint tag = q.CandidateInstanceID();
            Hit h;
            bool hit = false;
            if (tag & HF_RT_AABB_TAG) {
                uint idx = tag & ~HF_RT_AABB_TAG;
                GpuAabb b = gAabbs[idx];
                hit = IntersectAabb(ro, rd, int3(b.lox, b.loy, b.loz), int3(b.hix, b.hiy, b.hiz),
                                    b.primIndex, h);
            } else {
                GpuSphere s = gSpheres[tag];
                hit = IntersectSphere(ro, rd, int3(s.cx, s.cy, s.cz), s.radius, s.primIndex, h);
            }
            if (hit && h.t > minT) return true;
        }
    }
    return false;
}

[numthreads(HF_GI_THREADS, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    GiParams p = gParams[0];
    uint probeCount = p.counts.w;
    uint probe = gid.x;
    if (probe >= probeCount) return;

    int spacing = p.grid.x;
    int nx = p.grid.y, ny = p.grid.z, nz = p.grid.w;
    int3 origin = int3(p.origin.x, p.origin.y, p.origin.z);
    int3 lightDir = int3(p.light.x, p.light.y, p.light.z);
    uint background = p.counts.z;

    // VERBATIM gi.h::ProbePos(grid, linearIndex) (cx-major decode -> origin + (ix,iy,iz)*spacing).
    int ix = (int)(probe % (uint)nx);
    int iy = (int)((probe / (uint)nx) % (uint)ny);
    int iz = (int)(probe / (uint)(nx * ny));
    int3 ppos = int3(origin.x + fxmul(ix * HF_RT_ONE, spacing),
                     origin.y + fxmul(iy * HF_RT_ONE, spacing),
                     origin.z + fxmul(iz * HF_RT_ONE, spacing));

    [loop] for (int d = 0; d < HF_GI_RAYS; ++d) {
        int3 rd = HF_GI_DIRS[d];
        Hit hit = TraceClosestHW(ppos, rd);
        bool occluded = false;
        int3 indirect = int3(0, 0, 0);
        if (hit.primIndex != HF_RT_MISS) {
            int3 so = hit.pos + int3(fxmul(hit.normal.x, HF_RT_SHADOW_EPS),
                                     fxmul(hit.normal.y, HF_RT_SHADOW_EPS),
                                     fxmul(hit.normal.z, HF_RT_SHADOW_EPS));
            occluded = TraceAnyHitHW(so, lightDir, HF_RT_SHADOW_MINT);
            // The GI4 indirect: the GI3-interpolated irradiance of the PREVIOUS SH at the hit point.
            indirect = FxInterpolateIrradiance(hit.pos, hit.normal, (int)probeCount, nx, ny, nz,
                                               origin.x, origin.y, origin.z, spacing);
        }
        gRadiance[probe * (uint)HF_GI_RAYS + (uint)d] = ShadeHitGI(hit, lightDir, background, occluded, indirect);
    }
}
