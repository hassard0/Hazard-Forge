// Slice RT2b — Hardware Ray Tracing: METAL HW RAY QUERY (cross-platform HW-RT parity). The Apple-Metal twin
// of the Vulkan rt_query.comp.hlsl — proving Metal-HW == Metal-CPU == Vulkan-HW byte-identical. ONE thread per
// pixel: generate the integer primary ray (the FROZEN rtrace.h::PrimaryRay fx math), open a
// metal::raytracing::intersection_query over the primitive acceleration structure (margin-inflated bounding
// boxes), DRAIN EVERY candidate bounding-box primitive via next() WITHOUT ever committing or narrowing the ray
// interval, run the FROZEN fx Q16.16 IntersectSphere/IntersectAabb on each candidate, fold into a running
// closest by the total order (t, primIndex), and write the integer Lambert RGBA8 — bit-identical to the CPU
// rtrace::RenderScene and the Vulkan HW image.
//
// THE DETERMINISM CONTRACT (do NOT violate): the HW BVH is ONLY a candidate-AABB GENERATOR. We NEVER read the
// query's float distance; correctness is OWNED by OUR fx math + (t,primIndex) min. The driver's float
// AABB-overlap (and the float traversal ray) only WIDEN the candidate set. Every bounding box is INFLATED by
// kRtAabbMargin host-side so the float overlap is a strict SUPERSET of every true fx hit. int64 via `long`
// (native in MSL). This is the FIRST hand-authored .metal in the repo (all others are spirv-cross-gen'd);
// metal::raytracing::intersection_query needs Metal 2.4 (set MTLLanguageVersion2_4 host-side).
//
// CANDIDATE -> PRIMITIVE: one primitive acceleration structure with N bounding boxes (sphere bounds + box
// AABBs). get_candidate_primitive_id() indexes a parallel gPrims[] info buffer {kind, data, primIndex}.

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

constant int  HF_RT_FRAC   = 16;
constant int  HF_RT_ONE    = 1 << 16;            // 1.0 in Q16.16
constant int  HF_RT_NOHIT  = 0x7FFFFFFF;         // kRtNoHit
constant uint HF_RT_MISS   = 0xFFFFFFFFu;        // kRtMiss

// A primitive info record (parallel to the bounding-box buffer; indexed by primitive_id). kind 0 = sphere,
// 1 = aabb. For a sphere: c[0..2]=center, p0=radius. For an aabb: c=lo, p0..p2 via hi[]. primIndex = the
// global (t,primIndex) tie-break / shade key.
struct GpuPrim {
    int  kind;
    int  cx, cy, cz;      // sphere center / aabb lo
    int  hx, hy, hz;      // aabb hi (unused for sphere)
    int  radius;          // sphere radius (unused for aabb)
    uint primIndex;
    uint _pad0, _pad1, _pad2;   // pad to 48 B (12 * 4)
};

struct GpuParams {
    int eye[4];      // xyz = eye (Q16.16)
    int right[4];    // xyz = right basis
    int up[4];       // xyz = up basis
    int forward[4];  // xyz = forward basis
    int light[4];    // xyz = lightDir unit
    int plane[4];    // x=halfW, y=halfH, z=width, w=height
    uint counts[4];  // x=primCount, y=unused, z=background (RGBA8), w=unused
};

// ---- VERBATIM rtrace.h fx math (int64 via long) ----
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

static bool IntersectAabb(int3 ro, int3 rd, int3 lo, int3 hi, uint primIndex, thread Hit& outHit) {
    outHit = MissHit();
    int o[3]  = { ro.x, ro.y, ro.z };
    int d[3]  = { rd.x, rd.y, rd.z };
    int lov[3] = { lo.x, lo.y, lo.z };
    int hiv[3] = { hi.x, hi.y, hi.z };
    int tEnter = (int)0x80000000;
    int tExit  = HF_RT_NOHIT;
    int enterAxis = 0;
    int enterSign = 0;
    for (int ax = 0; ax < 3; ++ax) {
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
    int3 n = int3(0);
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

static void Consider(thread Hit& best, thread const Hit& h) {
    if (h.primIndex == HF_RT_MISS) return;
    if (best.primIndex == HF_RT_MISS || h.t < best.t ||
        (h.t == best.t && h.primIndex < best.primIndex)) best = h;
}

kernel void rt_query_main(
    device const GpuPrim*   gPrims   [[buffer(0)]],
    device const GpuParams& p        [[buffer(1)]],
    device uint*            gImage    [[buffer(2)]],
    primitive_acceleration_structure accel [[buffer(3)]],
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

    // The FLOAT ray for the driver's traversal (the ONLY float — origin/dir only WIDEN the candidate set).
    ray r;
    r.origin = float3(FxToFloat(ro.x), FxToFloat(ro.y), FxToFloat(ro.z));
    r.direction = float3(FxToFloat(rd.x), FxToFloat(rd.y), FxToFloat(rd.z));  // NOT normalized (t in |dir|)
    r.min_distance = 0.0f;
    r.max_distance = 1.0e30f;   // large; we NEVER narrow it (drain every overlapped box)

    Hit best = MissHit();

    intersection_query<> q;
    q.reset(r, accel);
    while (q.next()) {
        if (q.get_candidate_intersection_type() == intersection_type::bounding_box) {
            uint pid = q.get_candidate_primitive_id();
            GpuPrim pr = gPrims[pid];
            Hit h;
            if (pr.kind == 0) {
                if (IntersectSphere(ro, rd, int3(pr.cx, pr.cy, pr.cz), pr.radius, pr.primIndex, h))
                    Consider(best, h);
            } else {
                if (IntersectAabb(ro, rd, int3(pr.cx, pr.cy, pr.cz), int3(pr.hx, pr.hy, pr.hz),
                                  pr.primIndex, h)) Consider(best, h);
            }
            // Deliberately do NOT commit_bounding_box_intersection — keep draining every candidate.
        }
    }

    gImage[(uint)(py * width + px)] = ShadeHitInt(best, lightDir, p.counts[2]);
}
