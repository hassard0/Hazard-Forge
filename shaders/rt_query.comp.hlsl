// Slice RT2 — Hardware Ray Tracing: DETERMINISTIC HW INLINE RAY QUERY (the moat proof). ONE thread per
// pixel (gid.x = px, gid.y = py). Each thread generates its integer primary ray (the SAME frozen
// rtrace.h::PrimaryRay math as rt_trace.comp), opens a RayQuery over the TLAS (gTlas, binding 4) with a
// FLOAT ray (FxToFloat of the fx ray — the ONLY float, fed to the driver's BVH traversal), DRAINS EVERY
// candidate procedural primitive via Proceed() WITHOUT ever committing or shrinking tMax, runs the FROZEN
// fx Q16.16 IntersectSphere/IntersectAabb on each candidate, folds into a running closest by the total
// order (t, primIndex), and writes the integer Lambert RGBA8 — bit-identical to rt_trace.comp (SW GPU,
// brute-force no-cull) and to the CPU rtrace::RenderScene.
//
// THE DETERMINISM CONTRACT (do NOT violate): the HW BVH is ONLY a candidate-AABB GENERATOR. We NEVER read
// the query's float t / hit attributes; correctness is OWNED by OUR fx math + (t,primIndex) min. The
// driver's float AABB-overlap (and the float traversal ray) only WIDEN the candidate set — extra
// candidates are free (our min discards them). Every BLAS AABB is INFLATED by kRtAabbMargin (host-side) so
// the float overlap is a strict SUPERSET of every true fx hit -> the HW path can never SKIP a hit the
// no-cull SW reference finds (the candidate-completeness oracle). This is what makes the HW image
// BYTE-IDENTICAL to the SW reference, independent of the vendor's BVH order/layout.
//
// CANDIDATE -> PRIMITIVE: each TLAS instance carries instanceCustomIndex = a TAGGED index: bit 31 set ->
// an AABB at gAabbs[idx]; clear -> a sphere at gSpheres[idx] (idx = bits 0..30). q.CandidateInstanceIndex
// is NOT used; q.CandidateInstanceID() (the SPIR-V InstanceCustomIndex) gives the tag. The fx primIndex
// for the (t,primIndex) total order is the primitive's OWN stored primIndex (global), NOT the tag.
//
// VULKAN-SPIR-V-ONLY: HLSL inline RayQuery (DXC -> SPIR-V RayQueryKHR, which glslc/spirv-cross can't lower
// to MSL) AND int64 fx math -> compiled by DXC (cs_6_5 + SPV_KHR_ray_query) on the Vulkan path ONLY; NOT
// in the Metal hf_gen_msl list. On Metal the --rt2-query showcase runs the CPU rtrace::RenderScene (the
// SAME bit-exact reference) -> byte-identical by construction. The int64 + RayQuery convention is the
// fpx_integrate.comp / rt_trace.comp Vulkan-only convention extended to HW ray tracing.

#define HF_RT_THREADS_X 8
#define HF_RT_THREADS_Y 8
#define HF_RT_FRAC 16            // MUST match rtrace.h::kFrac (fpx.h::kFrac)
static const int HF_RT_ONE = 1 << HF_RT_FRAC;       // 1.0 in Q16.16
static const int HF_RT_NOHIT = 0x7FFFFFFF;          // kRtNoHit (INT32_MAX)
static const uint HF_RT_MISS = 0xFFFFFFFFu;         // kRtMiss
static const uint HF_RT_AABB_TAG = 0x800000u;       // custom-index bit 23 -> the prim is an AABB
                                                    // (instanceCustomIndex is a 24-BIT field; bit 31 would
                                                    // be masked off by the driver — bit 23 is the top usable
                                                    // bit, leaving 23 bits of index, ample for the scene)

// std430 mirrors of the rtrace.h POD types. fx == int (Q16.16). IDENTICAL to rt_trace.comp.
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

// VERBATIM rtrace.h::fxmul / fxdiv / FxISqrt (int64, the rt_trace.comp math copied EXACTLY).
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
// VERBATIM rtrace.h::ShadeHitInt.
uint ShadeHitInt(Hit hit, int3 lightDir, uint background) {
    if (hit.primIndex == HF_RT_MISS) return background;
    int ndl = fxdot(hit.normal, lightDir);
    if (ndl < 0) ndl = 0;
    int ambient = HF_RT_ONE * 18 / 100;
    int diffuse = ambient + fxmul(HF_RT_ONE - ambient, ndl);
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

    // The FLOAT ray for the driver's traversal (the ONLY float — origin/dir only WIDEN the candidate set).
    float3 fro = float3(FxToFloat(ro.x), FxToFloat(ro.y), FxToFloat(ro.z));
    float3 frd = float3(FxToFloat(rd.x), FxToFloat(rd.y), FxToFloat(rd.z));

    RayDesc ray;
    ray.Origin = fro;
    ray.Direction = frd;       // NOT normalized — t is in units of |dir|, matching the fx ray (our min
                               // uses OUR fx t; the float t is never read)
    ray.TMin = 0.0f;
    ray.TMax = 1.0e30f;        // large; we NEVER shrink it (drain every overlapped leaf)

    Hit best = MissHit();

    // RAY_FLAG_NONE: do NOT cull; yield every candidate procedural primitive. We never call CommitProcedural
    // (so the driver keeps the query open over ALL overlapped leaves) and never call CommitNonOpaqueTriangle.
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
            // Deliberately do NOT q.CommitProceduralPrimitiveHit — we keep draining every candidate.
        }
    }

    gImage[(uint)(py * width + px)] = ShadeHitInt(best, lightDir, p.counts.z);
}
