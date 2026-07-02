// Slice RT7 — Hardware Ray Tracing: the REAL MULTI-INSTANCE TLAS kernel (Track-R R9). ONE thread per
// pixel. The scene is ONE sphere-cluster BLAS instanced N times with per-instance transforms (translate +
// a 90-degree rotation, exact in float AND in Q16.16). The candidate mapping is TWO-LEVEL — the point of
// the slice: q.CandidateInstanceID() (the instanceCustomIndex = TlasInstance.instanceId = the cluster k)
// and q.CandidatePrimitiveIndex() (the AABB slot within the BLAS = the local sphere j) select the
// HOST-PRETRANSFORMED world-space sphere gPrims[k * primsPerInstance + j]. The intersection itself runs on
// that world-space fx sphere (VERBATIM the frozen rtrace.h math) so correctness NEVER touches the driver's
// float instance transform — the transform only steers candidate generation (the rt_query.comp
// determinism contract, extended to a two-level structure).
//
// THE DETERMINISM CONTRACT (rt_query.comp:10-16, unchanged): the HW BVH is ONLY a candidate GENERATOR. We
// NEVER read the query's float t / hit attributes; the fx Q16.16 math + the (t,primIndex) total-order min
// own correctness. Every BLAS AABB is inflated by kRtAabbMargin host-side; the RT7 instance transforms are
// 90-degree rotations + exactly-float-representable translations, so the driver's instance-local ray is
// EXACT and the float overlap stays a strict SUPERSET of every true fx hit.
//
// OUTPUTS: gImage (the integer Lambert RGBA8, byte-identical to the CPU rtrace::RenderScene over the same
// pre-transformed world spheres) + gInstId (the WINNING candidate's instance id per pixel, HF_RT_MISS on a
// miss — the instance-id-correctness proof read back and memcmp'd against primIndex/primsPerInstance).
//
// VULKAN-SPIR-V-ONLY (HLSL RayQuery + int64 -> DXC cs_6_5 + SPV_KHR_ray_query; NOT in the Metal MSL-gen
// list). The Metal twin is the hand-authored shaders/rt_instanced.metal (instance_acceleration_structure).

#define HF_RT_THREADS_X 8
#define HF_RT_THREADS_Y 8
#define HF_RT_FRAC 16            // MUST match rtrace.h::kFrac (fpx.h::kFrac)
static const int HF_RT_ONE = 1 << HF_RT_FRAC;       // 1.0 in Q16.16
static const int HF_RT_NOHIT = 0x7FFFFFFF;          // kRtNoHit (INT32_MAX)
static const uint HF_RT_MISS = 0xFFFFFFFFu;         // kRtMiss

// A world-space (host-pretransformed) sphere record. 32 B std430 — the rt_query.comp GpuSphere layout.
struct GpuSphere { int cx, cy, cz; int radius; uint primIndex; uint _pad0, _pad1, _pad2; };

struct GpuParams {
    int4 eye;        // x,y,z = eye (Q16.16), w unused
    int4 right;      // x,y,z = right basis (Q16.16), w unused
    int4 up;         // x,y,z = up basis (Q16.16), w unused
    int4 forward;    // x,y,z = forward basis (Q16.16), w unused
    int4 light;      // x,y,z = lightDir unit (Q16.16), w unused
    int4 plane;      // x = halfW, y = halfH (Q16.16), z = width, w = height
    uint4 counts;    // x = primsPerInstance (K), y = instanceCount, z = background (RGBA8), w unused
};

[[vk::binding(0, 0)]] StructuredBuffer<GpuSphere>     gPrims  : register(t0);
[[vk::binding(1, 0)]] StructuredBuffer<GpuParams>     gParams : register(t1);
[[vk::binding(2, 0)]] RWStructuredBuffer<uint>        gImage  : register(u2);
[[vk::binding(3, 0)]] RWStructuredBuffer<uint>        gInstId : register(u3);
[[vk::binding(4, 0)]] RaytracingAccelerationStructure gTlas   : register(t4);

// VERBATIM rtrace.h::fxmul / fxdiv / FxISqrt (int64 — the rt_query.comp math copied EXACTLY).
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

// VERBATIM rtrace.h::IntersectSphere (== rt_query.comp).
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

// VERBATIM rtrace.h::AlbedoFor / PackRGBA8 / ShadeHitInt (== rt_query.comp).
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

float FxToFloat(int v) { return (float)v / (float)HF_RT_ONE; }

// The (t,primIndex) total-order fold (VERBATIM rt_query.comp::Consider), EXTENDED to also carry the
// winning candidate's INSTANCE id. primIndex is globally unique across instances (k*K+j), so the winning
// instance is a deterministic function of the winning primIndex — order-independent.
void ConsiderInst(inout Hit best, inout uint bestInst, Hit h, uint inst) {
    if (h.primIndex == HF_RT_MISS) return;
    if (best.primIndex == HF_RT_MISS || h.t < best.t ||
        (h.t == best.t && h.primIndex < best.primIndex)) { best = h; bestInst = inst; }
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
    uint primsPerInstance = p.counts.x;

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

    // The FLOAT ray for the driver's traversal (the ONLY float — it only steers candidate generation).
    RayDesc ray;
    ray.Origin = float3(FxToFloat(ro.x), FxToFloat(ro.y), FxToFloat(ro.z));
    ray.Direction = float3(FxToFloat(rd.x), FxToFloat(rd.y), FxToFloat(rd.z));  // NOT normalized
    ray.TMin = 0.0f;
    ray.TMax = 1.0e30f;        // large; we NEVER shrink it (drain every overlapped leaf)

    Hit best = MissHit();
    uint bestInst = HF_RT_MISS;

    // Drain EVERY candidate procedural primitive across ALL instances; never commit.
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(gTlas, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            uint inst = q.CandidateInstanceID();       // instanceCustomIndex = TlasInstance.instanceId = k
            uint prim = q.CandidatePrimitiveIndex();   // the AABB slot within the BLAS = local sphere j
            GpuSphere s = gPrims[inst * primsPerInstance + prim];
            Hit h;
            if (IntersectSphere(ro, rd, int3(s.cx, s.cy, s.cz), s.radius, s.primIndex, h))
                ConsiderInst(best, bestInst, h, inst);
            // Deliberately do NOT CommitProceduralPrimitiveHit — keep draining every candidate.
        }
    }

    uint idx = (uint)(py * width + px);
    gImage[idx]  = ShadeHitInt(best, lightDir, p.counts.z);
    gInstId[idx] = (best.primIndex == HF_RT_MISS) ? HF_RT_MISS : bestInst;
}
