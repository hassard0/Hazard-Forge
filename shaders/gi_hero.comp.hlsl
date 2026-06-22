// Slice GI6 — Deterministic Lumen-class GI: THE LIT GI HERO CAPSTONE (the csrq stage; the FLAGSHIP #29
// MONEY-SHOT). ONE thread per pixel (gid.x = px, gid.y = py). The body is COPIED VERBATIM from
// shaders/rt_hero.comp.hlsl (RT6 — the primary ray + RT3 shadow + RT4 one-bounce reflection + the integer
// blend + the graded SkyGradient miss) with ONE addition: at each primary hit, a GI INDIRECT term =
// albedo(hit) ⊙ FxInterpolateIrradianceOcc(grid, gSH, gMoments, hit.pos, hit.normal, occStrength) ×
// giStrength is added to the RT6 shade on the unpacked 0..255 RGBA8 channels. The GI integer math (the
// trilinear+Chebyshev SH blend + FxSHEvaluate) is COPIED VERBATIM from shaders/gi_occ.comp.hlsl so the HW
// image is bit-identical to the CPU gi::RenderSceneGI, proven memcmp HW==CPU. STRICT INTEGER — the capstone
// stays byte-identical cross-vendor + HW==CPU. giStrength == 0 -> a literal +0 indirect -> byte-identical
// to the no-GI RT6 hero (the falsifiable no-op contract).
//
// THE DETERMINISM CONTRACT (inherited from RT2/RT6 + GI2/GI3/GI5): the HW BVH is ONLY a candidate-AABB
// GENERATOR (we never read the query's float t / hit attributes; correctness is OWNED by OUR fx math), and
// the GI indirect is a pure-integer lookup over the host-baked gSH + gMoments buffers -> the whole hero
// image is bit-exact cross-backend, regardless of BVH order.
//
// VULKAN-SPIR-V-ONLY: HLSL inline RayQuery (DXC -> SPIR-V RayQueryKHR, which glslc/spirv-cross can't lower
// to MSL) AND int64 fx math (the SH blend + Chebyshev) -> compiled by DXC (cs_6_5 + SPV_KHR_ray_query) on
// the Vulkan path ONLY; NOT in the Metal hf_gen_msl list. On Metal the --gi6-hero showcase runs the CPU
// gi::RenderSceneGI (the SAME bit-exact reference) -> byte-identical by construction.

#define HF_RT_THREADS_X 8
#define HF_RT_THREADS_Y 8
#define HF_RT_FRAC 16            // MUST match rtrace.h::kFrac (fpx.h::kFrac) / gi.h::kFrac
static const int HF_RT_ONE = 1 << HF_RT_FRAC;       // 1.0 in Q16.16
static const int HF_RT_NOHIT = 0x7FFFFFFF;          // kRtNoHit (INT32_MAX)
static const uint HF_RT_MISS = 0xFFFFFFFFu;         // kRtMiss
static const uint HF_RT_AABB_TAG = 0x800000u;       // custom-index bit 23 -> the prim is an AABB

// RT3 shadow constants (MUST match rtrace.h::kRtShadowEps / kRtShadowMinT).
static const int HF_RT_SHADOW_EPS  = HF_RT_ONE / 256;    // surface offset along the normal (anti-acne)
static const int HF_RT_SHADOW_MINT = HF_RT_ONE / 1024;   // ignore shadow hits closer than this (self-hit)

// RT4 reflection constants (MUST match rtrace.h::kRtReflEps / kRtMirrorSpherePrim).
static const int  HF_RT_REFL_EPS    = HF_RT_ONE / 256;   // reflection-ray surface offset along the normal
static const uint HF_RT_MIRROR_PRIM = 6u;                // the designated reflective mirror-sphere primIndex

// RT6 sky endpoint colors (MUST match rtrace.h::kRtSkyHorizon* / kRtSkyZenith*). Per-channel integer lerp.
static const int HF_RT_SKY_HORIZON_R = 120, HF_RT_SKY_HORIZON_G = 110, HF_RT_SKY_HORIZON_B = 140;  // horizon
static const int HF_RT_SKY_ZENITH_R  =  34, HF_RT_SKY_ZENITH_G  =  40, HF_RT_SKY_ZENITH_B  =  56;  // zenith

// GI cosine-lobe reconstruction band factors B_l = 4*A_l (Q16.16), IDENTICAL to gi.h::kGiSHReconB*.
static const int64_t kReconB0 = 823550;   // 4*pi    in Q16.16
static const int64_t kReconB1 = 549033;   // 8*pi/3  in Q16.16
static const int64_t kReconB2 = 205887;   // pi      in Q16.16
// The GI5 integer variance floor (gi.h::kGiVarFloor == kOne/4096 == 16).
static const int kVarFloor = HF_RT_ONE / 4096;

// std430 mirrors of the rtrace.h POD types. fx == int (Q16.16). IDENTICAL to rt_hero.comp.
struct GpuSphere { int cx, cy, cz; int radius; uint primIndex; uint _pad0, _pad1, _pad2; };  // 32 B
struct GpuAabb   { int lox, loy, loz; int hix, hiy, hiz; uint primIndex; uint _pad0; };       // 32 B

struct GpuParams {
    int4 eye;        // x,y,z = eye (Q16.16), w unused
    int4 right;      // x,y,z = right basis (Q16.16), w unused
    int4 up;         // x,y,z = up basis (Q16.16), w unused
    int4 forward;    // x,y,z = forward basis (Q16.16), w unused
    int4 light;      // x,y,z = lightDir unit (Q16.16), w unused
    int4 plane;      // x = halfW, y = halfH (Q16.16), z = width, w = height
    uint4 counts;    // x = sphereCount, y = aabbCount, z = background (UNUSED), w unused
    // GI params: gi0 = { nx, ny, nz, probeCount }; gi1 = { originX, originY, originZ, spacing };
    //            gi2 = { giStrength, occStrength, 0, 0 }.
    int4 gi0;
    int4 gi1;
    int4 gi2;
};

// std430 FxProbeSH mirror (gi.h::FxProbeSH): 9*3 = 27 Q16.16 coeff ints + 1 pad int = 28 ints (112 B).
struct FxProbeSH { int c[28]; };
// std430 FxProbeMoments mirror (gi.h::FxProbeMoments): meanDist, meanDist2 (Q16.16) == 2 ints (8 B).
struct FxProbeMoments { int meanDist; int meanDist2; };

// Bindings 0..5 are the storage buffers (SSBOs, contiguous — the RHI storageBufferCount range); the accel
// structure goes at binding 6 (== storageBufferCount, OUTSIDE the SSBO range, the gi_bounce.comp/rt_hero.comp
// convention).
[[vk::binding(0, 0)]] StructuredBuffer<GpuSphere>   gSpheres : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<GpuAabb>     gAabbs   : register(t1);
[[vk::binding(2, 0)]] StructuredBuffer<GpuParams>   gParams  : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>      gImage   : register(u3);
[[vk::binding(4, 0)]] StructuredBuffer<FxProbeSH>      gSH      : register(t4);
[[vk::binding(5, 0)]] StructuredBuffer<FxProbeMoments> gMoments : register(t5);
[[vk::binding(6, 0)]] RaytracingAccelerationStructure gTlas  : register(t6);

// VERBATIM rtrace.h::fxmul / fxdiv / FxISqrt (int64, the rt_hero.comp math copied EXACTLY).
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

// VERBATIM rtrace.h::AlbedoFor.
int3 AlbedoFor(uint primIndex) {
    uint i = primIndex % 6u;
    if (i == 0u) return int3(HF_RT_ONE * 78 / 100, HF_RT_ONE * 30 / 100, HF_RT_ONE * 26 / 100);
    if (i == 1u) return int3(HF_RT_ONE * 28 / 100, HF_RT_ONE * 52 / 100, HF_RT_ONE * 80 / 100);
    if (i == 2u) return int3(HF_RT_ONE * 36 / 100, HF_RT_ONE * 70 / 100, HF_RT_ONE * 38 / 100);
    if (i == 3u) return int3(HF_RT_ONE * 82 / 100, HF_RT_ONE * 70 / 100, HF_RT_ONE * 28 / 100);
    if (i == 4u) return int3(HF_RT_ONE * 64 / 100, HF_RT_ONE * 40 / 100, HF_RT_ONE * 72 / 100);
    return int3(HF_RT_ONE * 60 / 100, HF_RT_ONE * 60 / 100, HF_RT_ONE * 62 / 100);
}
uint PackRGBA8(int r, int g, int b, int a) {
    r = clamp(r, 0, 255); g = clamp(g, 0, 255); b = clamp(b, 0, 255); a = clamp(a, 0, 255);
    return (uint)r | ((uint)g << 8) | ((uint)b << 16) | ((uint)a << 24);
}

// RT6 — VERBATIM rtrace.h::SkyGradient.
uint SkyGradient(int3 dir) {
    int3 n = fxnormalize(dir);
    int t = (n.y + HF_RT_ONE) / 2;
    if (t < 0) t = 0;
    if (t > HF_RT_ONE) t = HF_RT_ONE;
    int cr = (int)(((int64_t)HF_RT_SKY_HORIZON_R * (int64_t)HF_RT_ONE + (int64_t)(HF_RT_SKY_ZENITH_R - HF_RT_SKY_HORIZON_R) * (int64_t)t) >> HF_RT_FRAC);
    int cg = (int)(((int64_t)HF_RT_SKY_HORIZON_G * (int64_t)HF_RT_ONE + (int64_t)(HF_RT_SKY_ZENITH_G - HF_RT_SKY_HORIZON_G) * (int64_t)t) >> HF_RT_FRAC);
    int cb = (int)(((int64_t)HF_RT_SKY_HORIZON_B * (int64_t)HF_RT_ONE + (int64_t)(HF_RT_SKY_ZENITH_B - HF_RT_SKY_HORIZON_B) * (int64_t)t) >> HF_RT_FRAC);
    return PackRGBA8(cr, cg, cb, 255);
}

// VERBATIM rtrace.h::ShadeHitShadowed.
uint ShadeHitShadowed(Hit hit, int3 lightDir, uint background, bool occluded) {
    if (hit.primIndex == HF_RT_MISS) return background;
    int ndl = fxdot(hit.normal, lightDir);
    if (ndl < 0) ndl = 0;
    int ambient = HF_RT_ONE * 18 / 100;
    int lambert = occluded ? 0 : fxmul(HF_RT_ONE - ambient, ndl);
    int diffuse = ambient + lambert;
    int3 alb = AlbedoFor(hit.primIndex);
    int qr = (int)(((int64_t)fxmul(alb.x, diffuse) * 255) >> HF_RT_FRAC);
    int qg = (int)(((int64_t)fxmul(alb.y, diffuse) * 255) >> HF_RT_FRAC);
    int qb = (int)(((int64_t)fxmul(alb.z, diffuse) * 255) >> HF_RT_FRAC);
    return PackRGBA8(qr, qg, qb, 255);
}

// VERBATIM rtrace.h::FxReflect.
int3 FxReflect(int3 d, int3 n) {
    int dn = fxdot(d, n);
    int s = 2 * dn;
    return d - int3(fxmul(n.x, s), fxmul(n.y, s), fxmul(n.z, s));
}

// VERBATIM rtrace.h::ReflectivityFor.
int ReflectivityFor(uint primIndex) {
    if (primIndex == 0u) return HF_RT_ONE * 55 / 100;
    if (primIndex == HF_RT_MIRROR_PRIM) return HF_RT_ONE * 75 / 100;
    return 0;
}

float FxToFloat(int v) { return (float)v / (float)HF_RT_ONE; }

void Consider(inout Hit best, Hit h) {
    if (h.primIndex == HF_RT_MISS) return;
    if (best.primIndex == HF_RT_MISS || h.t < best.t ||
        (h.t == best.t && h.primIndex < best.primIndex)) best = h;
}

Hit TraceClosestHw(int3 ro, int3 rd) {
    float3 fro = float3(FxToFloat(ro.x), FxToFloat(ro.y), FxToFloat(ro.z));
    float3 frd = float3(FxToFloat(rd.x), FxToFloat(rd.y), FxToFloat(rd.z));
    RayDesc ray;
    ray.Origin = fro;
    ray.Direction = frd;
    ray.TMin = 0.0f;
    ray.TMax = 1.0e30f;

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

bool TraceAnyHitHw(int3 sro, int3 srd) {
    float3 sfro = float3(FxToFloat(sro.x), FxToFloat(sro.y), FxToFloat(sro.z));
    float3 sfrd = float3(FxToFloat(srd.x), FxToFloat(srd.y), FxToFloat(srd.z));
    RayDesc sray;
    sray.Origin = sfro;
    sray.Direction = sfrd;
    sray.TMin = 0.0f;
    sray.TMax = 1.0e30f;

    bool occluded = false;
    RayQuery<RAY_FLAG_NONE> sq;
    sq.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, sray);
    while (sq.Proceed()) {
        if (sq.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint tag = sq.CandidateInstanceID();
            Hit h;
            bool hitOk;
            if (tag & HF_RT_AABB_TAG) {
                uint idx = tag & ~HF_RT_AABB_TAG;
                GpuAabb b = gAabbs[idx];
                hitOk = IntersectAabb(sro, srd, int3(b.lox, b.loy, b.loz),
                                      int3(b.hix, b.hiy, b.hiz), b.primIndex, h);
            } else {
                GpuSphere s = gSpheres[tag];
                hitOk = IntersectSphere(sro, srd, int3(s.cx, s.cy, s.cz), s.radius, s.primIndex, h);
            }
            if (hitOk && h.t > HF_RT_SHADOW_MINT) { occluded = true; sq.Abort(); }
        }
    }
    return occluded;
}

// ===== GI integer math (COPIED VERBATIM from gi_occ.comp.hlsl) =======================================
// FxChebyshevVisibility — the integer variance-shadow visibility, IDENTICAL to gi.h::FxChebyshevVisibility.
int FxChebyshevVisibility(FxProbeMoments m, int queryDist) {
    if (queryDist <= m.meanDist) return HF_RT_ONE;
    int variance = m.meanDist2 - fxmul(m.meanDist, m.meanDist);
    if (variance < kVarFloor) variance = kVarFloor;
    int dd = queryDist - m.meanDist;
    int dd2 = fxmul(dd, dd);
    int denom = variance + dd2;
    int vis = fxdiv(variance, denom);
    if (vis < 0) vis = 0;
    if (vis > HF_RT_ONE) vis = HF_RT_ONE;
    return vis;
}
// GiFxLerp — a + (b-a)*t, IDENTICAL to gi.h::GiFxLerp.
int GiFxLerp(int a, int b, int t) { return a + fxmul(b - a, t); }
// ProbePos integer decode (gi.h::ProbePos by linear index).
void ProbePosI(int linearIndex, int nx, int ny, int ox, int oy, int oz, int spacing,
               out int px, out int py, out int pz) {
    int ix = linearIndex % nx;
    int iy = (linearIndex / nx) % ny;
    int iz = linearIndex / (nx * ny);
    px = ox + fxmul(ix * HF_RT_ONE, spacing);
    py = oy + fxmul(iy * HF_RT_ONE, spacing);
    pz = oz + fxmul(iz * HF_RT_ONE, spacing);
}

// FxInterpolateIrradianceOcc — the GI3 trilinear blend + GI5 Chebyshev occlusion weighting +
// FxInterpolateSH + FxSHEvaluate, COPIED VERBATIM from gi_occ.comp.hlsl. Returns int3 GiRadiance (clamped
// >= 0 per channel). `pt`/`nm` are the Q16.16 hit position/normal; grid params come from gParams.
int3 GiIndirect(int3 pt, int3 nm, int nx, int ny, int nz, int ox, int oy, int oz, int spacing,
                int probeCount, int occStrength) {
    // ===== FxNearestProbes — floor-cell + frac + 8-corner index + partition-of-unity weights =====
    int base[3]; int frac[3];
    int pv[3] = { pt.x, pt.y, pt.z };
    int orig[3] = { ox, oy, oz };
    int dim[3] = { nx, ny, nz };
    [unroll] for (int ax = 0; ax < 3; ++ax) {
        if (dim[ax] <= 1) { base[ax] = 0; frac[ax] = 0; }
        else {
            int g = (int)((((int64_t)(pv[ax] - orig[ax])) << HF_RT_FRAC) / (int64_t)spacing);
            int b = g >> HF_RT_FRAC;
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
        idx[c] = cx + cy * nx + cz * (nx * ny);
        int wx = sx ? whi[0] : wlo[0];
        int wy = sy ? whi[1] : wlo[1];
        int wz = sz ? whi[2] : wlo[2];
        if (c < 7) {
            int64_t pp = (int64_t)wx * (int64_t)wy * (int64_t)wz;
            int wc = (int)(pp >> (2 * HF_RT_FRAC));
            wgt[c] = wc;
            accumW += wc;
        } else {
            wgt[c] = HF_RT_ONE - accumW;
        }
    }

    // ===== GI5 — Chebyshev occlusion weighting + re-normalize =====
    bool haveMoments = (probeCount > 0);
    int scaled[8];
    int64_t sumW = 0;
    [unroll] for (int c = 0; c < 8; ++c) {
        int w = wgt[c];
        if (haveMoments && occStrength != 0) {
            int cpx, cpy, cpz;
            ProbePosI(idx[c], nx, ny, ox, oy, oz, spacing, cpx, cpy, cpz);
            int dx = pt.x - cpx;
            int dy = pt.y - cpy;
            int dz = pt.z - cpz;
            int64_t ss = (int64_t)dx * (int64_t)dx + (int64_t)dy * (int64_t)dy + (int64_t)dz * (int64_t)dz;
            int distToCorner = (int)fxisqrt(ss);
            int vis = FxChebyshevVisibility(gMoments[idx[c]], distToCorner);
            int factor = GiFxLerp(HF_RT_ONE, vis, occStrength);
            w = fxmul(w, factor);
        }
        scaled[c] = w;
        sumW += (int64_t)w;
    }
    int normW[8];
    [unroll] for (int c = 0; c < 8; ++c) normW[c] = wgt[c];
    if (sumW > 0) {
        int accum = 0;
        [unroll] for (int c = 0; c < 7; ++c) {
            int w = fxdiv(scaled[c], (int)sumW);
            normW[c] = w;
            accum += w;
        }
        normW[7] = HF_RT_ONE - accum;
    }

    // ===== FxInterpolateSH — int64 8-corner blend, narrow ONCE =====
    int blended[27];
    if (probeCount <= 0) {
        [unroll] for (int z0 = 0; z0 < 27; ++z0) blended[z0] = 0;
    } else {
        [unroll] for (int k = 0; k < 27; ++k) {
            int64_t acc = 0;
            [unroll] for (int c2 = 0; c2 < 8; ++c2)
                acc += (int64_t)normW[c2] * (int64_t)gSH[idx[c2]].c[k];
            blended[k] = (int)(acc >> HF_RT_FRAC);
        }
    }

    // ===== FxSHEvaluate — the integer cosine-lobe irradiance reconstruction =====
    int x = nm.x, y = nm.y, z = nm.z;
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
        int64_t acc = 0;
        [unroll] for (int j = 0; j < 9; ++j)
            acc += (int64_t)blended[j * 3 + ch] * (int64_t)Y[j] * B[j];
        int v = (int)(acc >> (2 * HF_RT_FRAC));
        outc[ch] = (v < 0) ? 0 : v;
    }
    return int3(outc[0], outc[1], outc[2]);
}

[numthreads(HF_RT_THREADS_X, HF_RT_THREADS_Y, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    GpuParams p = gParams[0];
    int width  = p.plane.z;
    int height = p.plane.w;
    int px = (int)gid.x;
    int py = (int)gid.y;
    if (px >= width || py >= height) return;

    int3 eye     = int3(p.eye.x, p.eye.y, p.eye.z);
    int3 right   = int3(p.right.x, p.right.y, p.right.z);
    int3 up      = int3(p.up.x, p.up.y, p.up.z);
    int3 forward = int3(p.forward.x, p.forward.y, p.forward.z);
    int halfW = p.plane.x;
    int halfH = p.plane.y;
    int3 lightDir = int3(p.light.x, p.light.y, p.light.z);
    uint background = p.counts.z;

    // GI params.
    int nx = p.gi0.x, ny = p.gi0.y, nz = p.gi0.z, probeCount = p.gi0.w;
    int ox = p.gi1.x, oy = p.gi1.y, oz = p.gi1.z, spacing = p.gi1.w;
    int giStrength = p.gi2.x, occStrength = p.gi2.y;

    // VERBATIM rtrace.h::PrimaryRay.
    int64_t sxNum = (int64_t)(2 * (int64_t)px + 1) << HF_RT_FRAC;
    int sx = (int)(sxNum / (int64_t)(2 * (int64_t)width));
    int64_t syNum = (int64_t)(2 * (int64_t)py + 1) << HF_RT_FRAC;
    int sy = (int)(syNum / (int64_t)(2 * (int64_t)height));
    int ndcX = (sx * 2) - HF_RT_ONE;
    int ndcY = HF_RT_ONE - (sy * 2);
    int ox2 = fxmul(ndcX, halfW);
    int oy2 = fxmul(ndcY, halfH);
    int3 rd = forward;
    rd += int3(fxmul(right.x, ox2), fxmul(right.y, ox2), fxmul(right.z, ox2));
    rd += int3(fxmul(up.x, oy2), fxmul(up.y, oy2), fxmul(up.z, oy2));
    int3 ro = eye;

    // ===== PRIMARY RAY (closest-hit drain) =====
    Hit best = TraceClosestHw(ro, rd);

    // ===== PRIMARY MISS -> the GRADED SKY (no surface -> no indirect) =====
    if (best.primIndex == HF_RT_MISS) {
        gImage[(uint)(py * width + px)] = SkyGradient(rd);
        return;
    }

    // ===== PRIMARY SHADOW RAY -> the RT3 shadowed color cS =====
    int3 sro = best.pos + int3(fxmul(best.normal.x, HF_RT_SHADOW_EPS),
                               fxmul(best.normal.y, HF_RT_SHADOW_EPS),
                               fxmul(best.normal.z, HF_RT_SHADOW_EPS));
    bool occluded = TraceAnyHitHw(sro, lightDir);
    uint cS = ShadeHitShadowed(best, lightDir, background, occluded);

    // ===== RT4 REFLECTION (one bounce + integer blend; reflection miss -> graded sky) =====
    uint baseColor = cS;
    int refl = ReflectivityFor(best.primIndex);
    if (refl > 0) {
        int3 rro = best.pos + int3(fxmul(best.normal.x, HF_RT_REFL_EPS),
                                   fxmul(best.normal.y, HF_RT_REFL_EPS),
                                   fxmul(best.normal.z, HF_RT_REFL_EPS));
        int3 rrd = FxReflect(rd, best.normal);
        Hit rHit = TraceClosestHw(rro, rrd);

        uint cR;
        if (rHit.primIndex == HF_RT_MISS) {
            cR = SkyGradient(rrd);
        } else {
            int3 rsro = rHit.pos + int3(fxmul(rHit.normal.x, HF_RT_SHADOW_EPS),
                                        fxmul(rHit.normal.y, HF_RT_SHADOW_EPS),
                                        fxmul(rHit.normal.z, HF_RT_SHADOW_EPS));
            bool rOccluded = TraceAnyHitHw(rsro, lightDir);
            cR = ShadeHitShadowed(rHit, lightDir, background, rOccluded);
        }

        int sr = (int)( cS        & 0xFFu), rr = (int)( cR        & 0xFFu);
        int sg = (int)((cS >> 8)  & 0xFFu), rg = (int)((cR >> 8)  & 0xFFu);
        int sb = (int)((cS >> 16) & 0xFFu), rb = (int)((cR >> 16) & 0xFFu);
        int cr = (int)(((int64_t)sr * (int64_t)(HF_RT_ONE - refl) + (int64_t)rr * (int64_t)refl) >> HF_RT_FRAC);
        int cg = (int)(((int64_t)sg * (int64_t)(HF_RT_ONE - refl) + (int64_t)rg * (int64_t)refl) >> HF_RT_FRAC);
        int cb = (int)(((int64_t)sb * (int64_t)(HF_RT_ONE - refl) + (int64_t)rb * (int64_t)refl) >> HF_RT_FRAC);
        baseColor = PackRGBA8(cr, cg, cb, 255);
    }

    // ===== THE GI INDIRECT: albedo ⊙ irradiance(hit) × giStrength, added per channel (clamped). =====
    int3 irr = GiIndirect(best.pos, best.normal, nx, ny, nz, ox, oy, oz, spacing, probeCount, occStrength);
    int3 alb = AlbedoFor(best.primIndex);
    int gainR = (int)(((int64_t)fxmul(alb.x, fxmul(irr.x, giStrength)) * 255) >> HF_RT_FRAC);
    int gainG = (int)(((int64_t)fxmul(alb.y, fxmul(irr.y, giStrength)) * 255) >> HF_RT_FRAC);
    int gainB = (int)(((int64_t)fxmul(alb.z, fxmul(irr.z, giStrength)) * 255) >> HF_RT_FRAC);
    int baseR = (int)( baseColor        & 0xFFu);
    int baseG = (int)((baseColor >> 8)  & 0xFFu);
    int baseB = (int)((baseColor >> 16) & 0xFFu);
    uint outColor = PackRGBA8(baseR + gainR, baseG + gainG, baseB + gainB, 255);

    gImage[(uint)(py * width + px)] = outColor;
}
