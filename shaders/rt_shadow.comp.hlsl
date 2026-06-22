// Slice RT3 — Hardware Ray Tracing: DETERMINISTIC RT HARD SHADOWS (the csrq stage). ONE thread per pixel
// (gid.x = px, gid.y = py). The PRIMARY ray body is COPIED VERBATIM from shaders/rt_query.comp.hlsl (RT2):
// each thread generates its integer primary ray (the frozen rtrace.h::PrimaryRay math), opens a RayQuery
// over the TLAS (gTlas, binding 4) with a FLOAT ray, DRAINS EVERY candidate procedural primitive via
// Proceed() WITHOUT committing or shrinking tMax, runs the FROZEN fx Q16.16 IntersectSphere/IntersectAabb
// on each candidate, and folds into the running closest by the total order (t, primIndex). THEN, if the
// primary hit a surface, it casts a SECOND ray query — the SHADOW ray from hit.pos + normal*kRtShadowEps
// toward scene.lightDir — and runs OUR fx intersection on each candidate, setting occluded if ANY yields
// t > kRtShadowMinT (an order-independent boolean OR — the shadow query MAY early-out without breaking
// determinism). It writes ShadeHitShadowed (the RT2 integer Lambert with the diffuse GATED by occlusion).
// Bit-identical to the CPU rtrace::RenderSceneShadowed, proven memcmp HW==CPU.
//
// THE DETERMINISM CONTRACT (do NOT violate; inherited from RT2): the HW BVH is ONLY a candidate-AABB
// GENERATOR. We NEVER read the query's float t / hit attributes; correctness is OWNED by OUR fx math. The
// driver's float AABB-overlap (and the float traversal ray) only WIDEN the candidate set; every BLAS AABB
// is host-inflated by kRtAabbMargin so the float overlap is a strict SUPERSET of every true fx hit. The
// any-hit occlusion result is a boolean OR over the candidates' fx hits -> order-independent, so the HW
// any-hit == the SW brute-force any-hit == the CPU, bit-exact, regardless of the vendor's BVH order/layout.
//
// VULKAN-SPIR-V-ONLY: HLSL inline RayQuery (DXC -> SPIR-V RayQueryKHR, which glslc/spirv-cross can't lower
// to MSL) AND int64 fx math -> compiled by DXC (cs_6_5 + SPV_KHR_ray_query) on the Vulkan path ONLY; NOT
// in the Metal hf_gen_msl list. On Metal the --rt3-shadow showcase runs the CPU rtrace::RenderSceneShadowed
// (the SAME bit-exact reference) -> byte-identical by construction.

#define HF_RT_THREADS_X 8
#define HF_RT_THREADS_Y 8
#define HF_RT_FRAC 16            // MUST match rtrace.h::kFrac (fpx.h::kFrac)
static const int HF_RT_ONE = 1 << HF_RT_FRAC;       // 1.0 in Q16.16
static const int HF_RT_NOHIT = 0x7FFFFFFF;          // kRtNoHit (INT32_MAX)
static const uint HF_RT_MISS = 0xFFFFFFFFu;         // kRtMiss
static const uint HF_RT_AABB_TAG = 0x800000u;       // custom-index bit 23 -> the prim is an AABB

// RT3 shadow constants (MUST match rtrace.h::kRtShadowEps / kRtShadowMinT).
static const int HF_RT_SHADOW_EPS  = HF_RT_ONE / 256;    // surface offset along the normal (anti-acne)
static const int HF_RT_SHADOW_MINT = HF_RT_ONE / 1024;   // ignore shadow hits closer than this (self-hit)

// std430 mirrors of the rtrace.h POD types. fx == int (Q16.16). IDENTICAL to rt_query.comp.
struct GpuSphere { int cx, cy, cz; int radius; uint primIndex; uint _pad0, _pad1, _pad2; };  // 32 B
struct GpuAabb   { int lox, loy, loz; int hix, hiy, hiz; uint primIndex; uint _pad0; };       // 32 B

struct GpuParams {
    int4 eye;        // x,y,z = eye (Q16.16), w unused
    int4 right;      // x,y,z = right basis (Q16.16), w unused
    int4 up;         // x,y,z = up basis (Q16.16), w unused
    int4 forward;    // x,y,z = forward basis (Q16.16), w unused
    int4 light;      // x,y,z = lightDir unit (Q16.16), w unused
    int4 plane;      // x = halfW, y = halfH (Q16.16), z = width, w = height
    uint4 counts;    // x = sphereCount, y = aabbCount, z = background (RGBA8), w unused
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuSphere>   gSpheres : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<GpuAabb>     gAabbs   : register(t1);
[[vk::binding(2, 0)]] StructuredBuffer<GpuParams>   gParams  : register(t2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>      gImage   : register(u3);
[[vk::binding(4, 0)]] RaytracingAccelerationStructure gTlas  : register(t4);

// VERBATIM rtrace.h::fxmul / fxdiv / FxISqrt (int64, the rt_query.comp math copied EXACTLY).
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

// VERBATIM rtrace.h::ShadeHitShadowed — the integer Lambert with the DIFFUSE term gated by occlusion.
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

// Convert a Q16.16 fx scalar to float (the ONLY float: the driver's BVH traversal ray; correctness is
// re-checked in fx). Exact for the bounded scene magnitudes.
float FxToFloat(int v) { return (float)v / (float)HF_RT_ONE; }

// Fold a candidate hit into the running closest by the total order (t, primIndex). VERBATIM the
// rtrace.h::TraceClosest tie-break.
void Consider(inout Hit best, Hit h) {
    if (h.primIndex == HF_RT_MISS) return;
    if (best.primIndex == HF_RT_MISS || h.t < best.t ||
        (h.t == best.t && h.primIndex < best.primIndex)) best = h;
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

    // VERBATIM rtrace.h::PrimaryRay (integer NDC -> image-plane offset -> dir).
    int64_t sxNum = (int64_t)(2 * (int64_t)px + 1) << HF_RT_FRAC;
    int sx = (int)(sxNum / (int64_t)(2 * (int64_t)width));
    int64_t syNum = (int64_t)(2 * (int64_t)py + 1) << HF_RT_FRAC;
    int sy = (int)(syNum / (int64_t)(2 * (int64_t)height));
    int ndcX = (sx * 2) - HF_RT_ONE;
    int ndcY = HF_RT_ONE - (sy * 2);
    int ox = fxmul(ndcX, halfW);
    int oy = fxmul(ndcY, halfH);
    int3 rd = forward;
    rd += int3(fxmul(right.x, ox), fxmul(right.y, ox), fxmul(right.z, ox));
    rd += int3(fxmul(up.x, oy), fxmul(up.y, oy), fxmul(up.z, oy));
    int3 ro = eye;

    // ===== PRIMARY RAY (COPIED VERBATIM from rt_query.comp) =====
    float3 fro = float3(FxToFloat(ro.x), FxToFloat(ro.y), FxToFloat(ro.z));
    float3 frd = float3(FxToFloat(rd.x), FxToFloat(rd.y), FxToFloat(rd.z));

    RayDesc ray;
    ray.Origin = fro;
    ray.Direction = frd;       // NOT normalized — t is in units of |dir|, matching the fx ray
    ray.TMin = 0.0f;
    ray.TMax = 1.0e30f;        // large; we NEVER shrink it (drain every overlapped leaf)

    Hit best = MissHit();

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint tag = q.CandidateInstanceID();   // the instanceCustomIndex we set host-side
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
            // Deliberately do NOT q.CommitProceduralPrimitiveHit — keep draining every candidate.
        }
    }

    // ===== SHADOW RAY (RT3) — an any-hit occlusion test over the SAME accel structure =====
    bool occluded = false;
    if (best.primIndex != HF_RT_MISS) {
        // Shadow-ray origin = best.pos offset along the surface normal by kRtShadowEps (anti-acne);
        // direction = the directional light direction. PURE INTEGER (matches rtrace::RenderSceneShadowed).
        int3 sro = best.pos + int3(fxmul(best.normal.x, HF_RT_SHADOW_EPS),
                                   fxmul(best.normal.y, HF_RT_SHADOW_EPS),
                                   fxmul(best.normal.z, HF_RT_SHADOW_EPS));
        int3 srd = lightDir;

        float3 sfro = float3(FxToFloat(sro.x), FxToFloat(sro.y), FxToFloat(sro.z));
        float3 sfrd = float3(FxToFloat(srd.x), FxToFloat(srd.y), FxToFloat(srd.z));

        RayDesc sray;
        sray.Origin = sfro;
        sray.Direction = sfrd;     // NOT normalized — t is in units of |dir| (the directional light dir)
        sray.TMin = 0.0f;
        sray.TMax = 1.0e30f;       // directional light -> no far bound; we never shrink it

        // The any-hit query: drain candidates running OUR fx intersection; occluded = ANY fx hit with
        // t > kRtShadowMinT. This is an order-independent boolean OR, so we MAY early-out the moment one
        // occluder is found — the SW reference scans all primitives, both yield the SAME boolean.
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
    }

    gImage[(uint)(py * width + px)] = ShadeHitShadowed(best, lightDir, p.counts.z, occluded);
}
