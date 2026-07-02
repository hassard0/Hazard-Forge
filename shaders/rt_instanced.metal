// Slice RT7 — Hardware Ray Tracing: the REAL MULTI-INSTANCE TLAS kernel on METAL (Track-R R9). The Metal
// twin of shaders/rt_instanced.comp.hlsl. ONE thread per pixel over an INSTANCE acceleration structure
// (the true two-level TLAS metal_accel.mm now builds for N >= 2 instances — the S1 "degenerate
// single-instance" caveat closed): open an intersection_query<instancing>, DRAIN every candidate
// bounding-box primitive WITHOUT committing, map the candidate TWO-LEVEL —
// get_candidate_user_instance_id() (the MTLAccelerationStructureUserIDInstanceDescriptor userID =
// TlasInstance.instanceId = the cluster k) and get_candidate_primitive_id() (the AABB slot within the
// child BLAS = the local sphere j) — into the HOST-PRETRANSFORMED world-space sphere
// gPrims[k * primsPerInstance + j], run the FROZEN fx Q16.16 IntersectSphere on it, fold by the
// (t,primIndex) total order, and write the integer Lambert RGBA8 + the winning instance id — bit-identical
// to the CPU rtrace::RenderScene over the same world spheres and to the Vulkan rt_instanced.comp image.
//
// THE DETERMINISM CONTRACT (rt_query.metal:10-15, unchanged): the HW BVH is ONLY a candidate GENERATOR; we
// NEVER read the query's float distance; correctness is OWNED by the fx math + (t,primIndex) min. The RT7
// instance transforms are 90-degree rotations + exactly-float-representable translations, so the driver's
// instance-local traversal ray is EXACT and the margin-inflated float overlap stays a strict SUPERSET of
// every true fx hit. int64 via `long` (native in MSL). Metal 2.4 (MTLLanguageVersion2_4 pinned host-side).
//
// FALLBACK NOTE FOR THE MAC CONTROLLER: if get_candidate_user_instance_id() proves fiddly, switching to
// get_candidate_instance_id() (the descriptor-array index) is behaviorally identical for the --rt7-
// instanced showcase (it passes instanceId == the array index) — pair that with the plain
// MTLAccelerationStructureInstanceDescriptor in metal_accel.mm's instanced path.

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

constant int  HF_RT_FRAC   = 16;
constant int  HF_RT_ONE    = 1 << 16;            // 1.0 in Q16.16
constant int  HF_RT_NOHIT  = 0x7FFFFFFF;         // kRtNoHit
constant uint HF_RT_MISS   = 0xFFFFFFFFu;        // kRtMiss

// A world-space (host-pretransformed) sphere record — 32 B, the rt_instanced.comp.hlsl GpuSphere layout.
struct GpuSphere {
    int  cx, cy, cz;      // world-space center (Q16.16)
    int  radius;          // Q16.16
    uint primIndex;       // GLOBAL primitive index (k * primsPerInstance + j)
    uint _pad0, _pad1, _pad2;
};

struct GpuParams {
    int eye[4];      // xyz = eye (Q16.16)
    int right[4];    // xyz = right basis
    int up[4];       // xyz = up basis
    int forward[4];  // xyz = forward basis
    int light[4];    // xyz = lightDir unit
    int plane[4];    // x=halfW, y=halfH, z=width, w=height
    uint counts[4];  // x=primsPerInstance (K), y=instanceCount, z=background (RGBA8), w=unused
};

// ---- VERBATIM rt_query.metal fx math (int64 via long) ----
static int fxmul(int a, int b) { return (int)(((long)a * (long)b) >> HF_RT_FRAC); }
static int fxdiv(int a, int b) { return (int)(((long)a << HF_RT_FRAC) / (long)b); }
static long fxisqrt(long v) {
    if (v <= 0) return 0;
    long bit = (long)1 << 62;
    while (bit > v) bit >>= 2;
    long res = 0;
    while (bit != 0) {
        if (v >= res + bit) { v -= res + bit; res = (res >> 1) + bit; }
        else { res >>= 1; }
        bit >>= 2;
    }
    return res;
}
static int fxdot(int3 a, int3 b) { return fxmul(a.x, b.x) + fxmul(a.y, b.y) + fxmul(a.z, b.z); }
static int3 fxnormalize(int3 v) {
    long sx = (long)v.x * (long)v.x;
    long sy = (long)v.y * (long)v.y;
    long sz = (long)v.z * (long)v.z;
    int len = (int)fxisqrt(sx + sy + sz);
    if (len == 0) return v;
    return int3(fxdiv(v.x, len), fxdiv(v.y, len), fxdiv(v.z, len));
}

struct Hit { int t; uint primIndex; int3 pos; int3 normal; };
static Hit MissHit() { Hit h; h.t = HF_RT_NOHIT; h.primIndex = HF_RT_MISS; h.pos = int3(0); h.normal = int3(0); return h; }

// VERBATIM rt_query.metal::IntersectSphere.
static bool IntersectSphere(int3 ro, int3 rd, int3 center, int radius, uint primIndex, thread Hit& outHit) {
    outHit = MissHit();
    int3 oc = ro - center;
    int a = fxdot(rd, rd);
    if (a <= 0) return false;
    int half_b = fxdot(oc, rd);
    int c = fxdot(oc, oc) - fxmul(radius, radius);
    long hb = (long)half_b * (long)half_b;
    long ac = (long)a * (long)c;
    long disc = hb - ac;
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

// VERBATIM rt_query.metal::AlbedoFor / PackRGBA8 / ShadeHitInt.
static int3 AlbedoFor(uint primIndex) {
    uint i = primIndex % 6u;
    if (i == 0u) return int3(HF_RT_ONE * 78 / 100, HF_RT_ONE * 30 / 100, HF_RT_ONE * 26 / 100);
    if (i == 1u) return int3(HF_RT_ONE * 28 / 100, HF_RT_ONE * 52 / 100, HF_RT_ONE * 80 / 100);
    if (i == 2u) return int3(HF_RT_ONE * 36 / 100, HF_RT_ONE * 70 / 100, HF_RT_ONE * 38 / 100);
    if (i == 3u) return int3(HF_RT_ONE * 82 / 100, HF_RT_ONE * 70 / 100, HF_RT_ONE * 28 / 100);
    if (i == 4u) return int3(HF_RT_ONE * 64 / 100, HF_RT_ONE * 40 / 100, HF_RT_ONE * 72 / 100);
    return int3(HF_RT_ONE * 60 / 100, HF_RT_ONE * 60 / 100, HF_RT_ONE * 62 / 100);
}
static uint PackRGBA8(int r, int g, int b, int a) {
    r = clamp(r, 0, 255); g = clamp(g, 0, 255); b = clamp(b, 0, 255); a = clamp(a, 0, 255);
    return (uint)r | ((uint)g << 8) | ((uint)b << 16) | ((uint)a << 24);
}
static uint ShadeHitInt(thread const Hit& hit, int3 lightDir, uint background) {
    if (hit.primIndex == HF_RT_MISS) return background;
    int ndl = fxdot(hit.normal, lightDir);
    if (ndl < 0) ndl = 0;
    int ambient = HF_RT_ONE * 18 / 100;
    int diffuse = ambient + fxmul(HF_RT_ONE - ambient, ndl);
    int3 alb = AlbedoFor(hit.primIndex);
    int qr = (int)(((long)fxmul(alb.x, diffuse) * 255) >> HF_RT_FRAC);
    int qg = (int)(((long)fxmul(alb.y, diffuse) * 255) >> HF_RT_FRAC);
    int qb = (int)(((long)fxmul(alb.z, diffuse) * 255) >> HF_RT_FRAC);
    return PackRGBA8(qr, qg, qb, 255);
}
static float FxToFloat(int v) { return (float)v / (float)HF_RT_ONE; }

// The (t,primIndex) total-order fold (VERBATIM rt_query.metal::Consider), EXTENDED to carry the winning
// candidate's INSTANCE id. primIndex is globally unique across instances (k*K+j), so the winning instance
// is a deterministic function of the winning primIndex — order-independent.
static void ConsiderInst(thread Hit& best, thread uint& bestInst, thread const Hit& h, uint inst) {
    if (h.primIndex == HF_RT_MISS) return;
    if (best.primIndex == HF_RT_MISS || h.t < best.t ||
        (h.t == best.t && h.primIndex < best.primIndex)) { best = h; bestInst = inst; }
}

kernel void rt_instanced_main(
    device const GpuSphere* gPrims   [[buffer(0)]],
    device const GpuParams& p        [[buffer(1)]],
    device uint*            gImage   [[buffer(2)]],
    device uint*            gInstId  [[buffer(3)]],
    instance_acceleration_structure accel [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    int width  = p.plane[2];
    int height = p.plane[3];
    int px = (int)gid.x;
    int py = (int)gid.y;
    if (px >= width || py >= height) return;

    int3 eye     = int3(p.eye[0], p.eye[1], p.eye[2]);
    int3 right   = int3(p.right[0], p.right[1], p.right[2]);
    int3 up      = int3(p.up[0], p.up[1], p.up[2]);
    int3 forward = int3(p.forward[0], p.forward[1], p.forward[2]);
    int halfW = p.plane[0];
    int halfH = p.plane[1];
    int3 lightDir = int3(p.light[0], p.light[1], p.light[2]);
    uint primsPerInstance = p.counts[0];

    // VERBATIM rtrace.h::PrimaryRay (integer NDC -> image-plane offset -> dir).
    long sxNum = (long)(2 * (long)px + 1) << HF_RT_FRAC;
    int sx = (int)(sxNum / (long)(2 * (long)width));
    long syNum = (long)(2 * (long)py + 1) << HF_RT_FRAC;
    int sy = (int)(syNum / (long)(2 * (long)height));
    int ndcX = (sx * 2) - HF_RT_ONE;
    int ndcY = HF_RT_ONE - (sy * 2);
    int ox = fxmul(ndcX, halfW);
    int oy = fxmul(ndcY, halfH);
    int3 rd = forward;
    rd += int3(fxmul(right.x, ox), fxmul(right.y, ox), fxmul(right.z, ox));
    rd += int3(fxmul(up.x, oy), fxmul(up.y, oy), fxmul(up.z, oy));
    int3 ro = eye;

    // The FLOAT ray for the driver's traversal (the ONLY float — it only steers candidate generation).
    ray r;
    r.origin = float3(FxToFloat(ro.x), FxToFloat(ro.y), FxToFloat(ro.z));
    r.direction = float3(FxToFloat(rd.x), FxToFloat(rd.y), FxToFloat(rd.z));  // NOT normalized
    r.min_distance = 0.0f;
    r.max_distance = 1.0e30f;   // large; we NEVER narrow it (drain every overlapped box)

    Hit best = MissHit();
    uint bestInst = HF_RT_MISS;

    // The instanced query: the `instancing` tag + the instance AS + the all-pass 0xFF mask (matching the
    // Vulkan TraceRayInline mask and the instance descriptors' mask=0xFF).
    intersection_query<instancing> q;
    q.reset(r, accel, 0xFFu);
    while (q.next()) {
        if (q.get_candidate_intersection_type() == intersection_type::bounding_box) {
            uint inst = q.get_candidate_user_instance_id();  // descriptor userID = TlasInstance.instanceId
            uint prim = q.get_candidate_primitive_id();      // the AABB slot within the child BLAS
            GpuSphere s = gPrims[inst * primsPerInstance + prim];
            Hit h;
            if (IntersectSphere(ro, rd, int3(s.cx, s.cy, s.cz), s.radius, s.primIndex, h))
                ConsiderInst(best, bestInst, h, inst);
            // Deliberately do NOT commit_bounding_box_intersection — keep draining every candidate.
        }
    }

    uint idx = (uint)(py * width + px);
    gImage[idx]  = ShadeHitInt(best, lightDir, p.counts[2]);
    gInstId[idx] = (best.primIndex == HF_RT_MISS) ? HF_RT_MISS : bestInst;
}
