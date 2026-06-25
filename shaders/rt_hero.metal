// Slice METAL-RT S4 — Hardware Ray Tracing: THE LIT HERO CAPSTONE (the rt_hero.comp.hlsl Metal twin).
// The Apple-Metal native MSL kernel proving Metal-HW == CPU rtrace::RenderSceneHero byte-identical.
// ONE thread per pixel. The body is COPIED VERBATIM from shaders/rt_reflect.metal (RT4/S3) — the primary
// closest-hit + the primary shadow any-hit + the integer shadowed shade cS + the RT4 one-bounce mirror
// reflection + the integer blend — with the ONE change that every MISS color (the PRIMARY miss AND the
// reflection-ray miss) is now the integer SkyGradient(missDir) (a graded horizon->zenith sky) instead of
// the flat background param. The SkyGradient math is the EXACT integer lerp of rtrace.h::SkyGradient
// (translated verbatim from rt_hero.comp.hlsl:175-184) so the HW hero image is bit-identical to the CPU
// rtrace::RenderSceneHero, proven memcmp HW==CPU. STRICT INTEGER — the capstone stays byte-identical
// cross-vendor + HW==CPU. ONE bounce (the reflected hit's own reflectivity ignored).
//
// THE DETERMINISM CONTRACT (do NOT violate; inherited from rt_query.metal/rt_shadow.metal/rt_reflect.metal):
// the HW BVH is ONLY a candidate-AABB GENERATOR. We NEVER read the query's float distance/attributes;
// correctness is OWNED by OUR fx math + the (t,primIndex) min (primary AND reflection) / the boolean OR
// (shadow). The sky gradient is a pure integer function of the (integer-normalized) ray direction -> the
// whole hero image is bit-exact cross-backend, regardless of the vendor's BVH order. Every bounding box is
// INFLATED by kRtAabbMargin host-side so the float overlap is a strict SUPERSET of every true fx hit.
// int64 via `long` (native in MSL).
//
// This is the FOURTH hand-authored .metal in the repo (the rt_query.metal / rt_shadow.metal / rt_reflect.metal
// twin; all others are spirv-cross-gen'd); metal::raytracing::intersection_query needs Metal 2.4 (set
// MTLLanguageVersion2_4 host-side). NO HLSL / NO SPIR-V.
//
// CANDIDATE -> PRIMITIVE: one primitive acceleration structure with N bounding boxes (sphere bounds + box
// AABBs). get_candidate_primitive_id() indexes a parallel gPrims[] info buffer {kind, data, primIndex} —
// IDENTICAL to rt_reflect.metal's binding (so the showcase uploads the SAME HwPrim/HwParams buffers).

#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

constant int  HF_RT_FRAC   = 16;
constant int  HF_RT_ONE    = 1 << 16;            // 1.0 in Q16.16
constant int  HF_RT_NOHIT  = 0x7FFFFFFF;         // kRtNoHit
constant uint HF_RT_MISS   = 0xFFFFFFFFu;        // kRtMiss

// RT3 shadow constants (MUST match rtrace.h::kRtShadowEps / kRtShadowMinT and rt_reflect.metal:43-44).
constant int HF_RT_SHADOW_EPS  = HF_RT_ONE / 256;    // surface offset along the normal (anti-acne)
constant int HF_RT_SHADOW_MINT = HF_RT_ONE / 1024;   // ignore shadow hits closer than this (self-hit guard)

// RT4 reflection constants (MUST match rtrace.h::kRtReflEps / kRtMirrorSpherePrim and rt_reflect.metal:47-48).
constant int  HF_RT_REFL_EPS    = HF_RT_ONE / 256;   // reflection-ray surface offset along the normal
constant uint HF_RT_MIRROR_PRIM = 6u;                // the designated reflective mirror-sphere primIndex

// RT6 sky endpoint colors (MUST match rtrace.h::kRtSkyHorizon* / kRtSkyZenith* and rt_hero.comp.hlsl:37-38).
constant int HF_RT_SKY_HORIZON_R = 120, HF_RT_SKY_HORIZON_G = 110, HF_RT_SKY_HORIZON_B = 140;  // horizon
constant int HF_RT_SKY_ZENITH_R  =  34, HF_RT_SKY_ZENITH_G  =  40, HF_RT_SKY_ZENITH_B  =  56;  // zenith

// A primitive info record (parallel to the bounding-box buffer; indexed by primitive_id). kind 0 = sphere,
// 1 = aabb. For a sphere: c[0..2]=center, p0=radius. For an aabb: c=lo, p0..p2 via hi[]. primIndex = the
// global (t,primIndex) tie-break / shade key. IDENTICAL to rt_reflect.metal::GpuPrim / the host HwPrim.
struct GpuPrim {
    int  kind;
    int  cx, cy, cz;      // sphere center / aabb lo
    int  hx, hy, hz;      // aabb hi (unused for sphere)
    int  radius;          // sphere radius (unused for aabb)
    uint primIndex;
    uint _pad0, _pad1, _pad2;   // pad to 48 B (12 * 4)
};

// IDENTICAL to rt_reflect.metal::GpuParams / the host HwParams. light[] = the directional light dir (unit,
// Q16.16) — the shadow ray direction AND the Lambert light dir. counts[2] = background (RGBA8, UNUSED in RT6
// — misses use the sky gradient; kept for layout parity with the shared HwParams).
struct GpuParams {
    int eye[4];      // xyz = eye (Q16.16)
    int right[4];    // xyz = right basis
    int up[4];       // xyz = up basis
    int forward[4];  // xyz = forward basis
    int light[4];    // xyz = lightDir unit
    int plane[4];    // x=halfW, y=halfH, z=width, w=height
    uint counts[4];  // x=primCount, y=unused, z=background (RGBA8, UNUSED in RT6), w=unused
};

// ---- VERBATIM rtrace.h fx math (int64 via long) — copied from rt_reflect.metal:74-97. ----
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

// VERBATIM rtrace.h::IntersectSphere (copied from rt_reflect.metal:102-126).
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

// VERBATIM rtrace.h::IntersectAabb (copied from rt_reflect.metal:128-165).
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

// VERBATIM rtrace.h::AlbedoFor (copied from rt_reflect.metal:168-176).
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

// RT6 — VERBATIM rtrace.h::SkyGradient (translated from rt_hero.comp.hlsl:175-184): integer horizon->zenith
// lerp by the (integer-normalized) dir.y. t = (n.y + kOne)/2 clamped to [0,kOne]; each channel =
// horizon + ((zenith-horizon)*t) >> kFrac. Returns RGBA8. The EXACT integer math of the CPU oracle so the
// HW hero image is byte-identical at BOTH miss sites (the primary miss AND the reflection-ray miss).
static uint SkyGradient(int3 dir) {
    int3 n = fxnormalize(dir);
    int t = (n.y + HF_RT_ONE) / 2;
    if (t < 0) t = 0;
    if (t > HF_RT_ONE) t = HF_RT_ONE;
    int cr = (int)(((long)HF_RT_SKY_HORIZON_R * (long)HF_RT_ONE + (long)(HF_RT_SKY_ZENITH_R - HF_RT_SKY_HORIZON_R) * (long)t) >> HF_RT_FRAC);
    int cg = (int)(((long)HF_RT_SKY_HORIZON_G * (long)HF_RT_ONE + (long)(HF_RT_SKY_ZENITH_G - HF_RT_SKY_HORIZON_G) * (long)t) >> HF_RT_FRAC);
    int cb = (int)(((long)HF_RT_SKY_HORIZON_B * (long)HF_RT_ONE + (long)(HF_RT_SKY_ZENITH_B - HF_RT_SKY_HORIZON_B) * (long)t) >> HF_RT_FRAC);
    return PackRGBA8(cr, cg, cb, 255);
}

// VERBATIM rtrace.h::ShadeHitShadowed (copied from rt_reflect.metal:184-196) — the integer Lambert with the
// DIFFUSE term GATED by occlusion. occluded -> the pixel keeps only the ambient floor. NOTE: in RT6 a MISS
// never reaches ShadeHitShadowed (the caller substitutes SkyGradient for misses), so the `background`
// argument is never returned. Kept VERBATIM for byte-identity with the RT4 body.
static uint ShadeHitShadowed(thread const Hit& hit, int3 lightDir, uint background, bool occluded) {
    if (hit.primIndex == HF_RT_MISS) return background;
    int ndl = fxdot(hit.normal, lightDir);
    if (ndl < 0) ndl = 0;
    int ambient = HF_RT_ONE * 18 / 100;
    int lambert = occluded ? 0 : fxmul(HF_RT_ONE - ambient, ndl);   // the GATED diffuse term
    int diffuse = ambient + lambert;
    int3 alb = AlbedoFor(hit.primIndex);
    int qr = (int)(((long)fxmul(alb.x, diffuse) * 255) >> HF_RT_FRAC);
    int qg = (int)(((long)fxmul(alb.y, diffuse) * 255) >> HF_RT_FRAC);
    int qb = (int)(((long)fxmul(alb.z, diffuse) * 255) >> HF_RT_FRAC);
    return PackRGBA8(qr, qg, qb, 255);
}
static float FxToFloat(int v) { return (float)v / (float)HF_RT_ONE; }

// VERBATIM rtrace.h::FxReflect (copied from rt_reflect.metal:201-205) — r = d - 2*(d·n)*n (n unit;
// |r|=|d| so the reflected ray's t units stay consistent). PURE INTEGER, deterministic.
static int3 FxReflect(int3 d, int3 n) {
    int dn = fxdot(d, n);
    int s = 2 * dn;
    return d - int3(fxmul(n.x, s), fxmul(n.y, s), fxmul(n.z, s));
}

// VERBATIM rtrace.h::ReflectivityFor (copied from rt_reflect.metal:210-214) — the fixed per-prim integer
// reflectivity in [0,kOne]. The ground (primIndex 0) is 0.55; the designated mirror sphere is 0.75; every
// other primitive is 0 (matte -> byte-identical to the RT3 shadowed shade).
static int ReflectivityFor(uint primIndex) {
    if (primIndex == 0u) return HF_RT_ONE * 55 / 100;
    if (primIndex == HF_RT_MIRROR_PRIM) return HF_RT_ONE * 75 / 100;
    return 0;
}

// Fold a candidate hit into the running closest by the total order (t, primIndex). VERBATIM the
// rtrace.h::TraceClosest tie-break (copied from rt_reflect.metal:218-222).
static void Consider(thread Hit& best, thread const Hit& h) {
    if (h.primIndex == HF_RT_MISS) return;
    if (best.primIndex == HF_RT_MISS || h.t < best.t ||
        (h.t == best.t && h.primIndex < best.primIndex)) best = h;
}

// A closest-hit drain over the accel structure — VERBATIM rt_reflect.metal::TraceClosestHw:228-258 (shared
// by the primary AND the reflection ray; the HW BVH is ONLY a candidate generator, correctness in fx). Each
// call declares its OWN intersection_query<> instance (function-local), so the primary and the
// reflected-primary do not share state.
static Hit TraceClosestHw(int3 ro, int3 rd,
                          device const GpuPrim* gPrims,
                          primitive_acceleration_structure accel) {
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
    return best;
}

// An any-hit OCCLUSION drain over the accel structure — VERBATIM rt_reflect.metal::TraceAnyHitHw:264-293
// (shared by the primary shadow AND the reflected shadow). occluded iff ANY candidate's fx hit has
// t > kRtShadowMinT (an order-independent boolean OR — MAY early-out).
static bool TraceAnyHitHw(int3 sro, int3 srd,
                          device const GpuPrim* gPrims,
                          primitive_acceleration_structure accel) {
    // The FLOAT traversal ray (the ONLY float — origin/dir only WIDEN the candidate set).
    ray sr;
    sr.origin = float3(FxToFloat(sro.x), FxToFloat(sro.y), FxToFloat(sro.z));
    sr.direction = float3(FxToFloat(srd.x), FxToFloat(srd.y), FxToFloat(srd.z));  // NOT normalized
    sr.min_distance = 0.0f;
    sr.max_distance = 1.0e30f;   // directional light -> no far bound; we never narrow it

    bool occluded = false;
    intersection_query<> sq;
    sq.reset(sr, accel);
    while (sq.next()) {
        if (sq.get_candidate_intersection_type() == intersection_type::bounding_box) {
            uint pid = sq.get_candidate_primitive_id();
            GpuPrim pr = gPrims[pid];
            Hit h;
            bool hitOk;
            if (pr.kind == 0) {
                hitOk = IntersectSphere(sro, srd, int3(pr.cx, pr.cy, pr.cz), pr.radius, pr.primIndex, h);
            } else {
                hitOk = IntersectAabb(sro, srd, int3(pr.cx, pr.cy, pr.cz), int3(pr.hx, pr.hy, pr.hz),
                                      pr.primIndex, h);
            }
            if (hitOk && h.t > HF_RT_SHADOW_MINT) { occluded = true; sq.abort(); }
        }
    }
    return occluded;
}

kernel void rt_hero_main(
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
    uint background = p.counts[2];   // UNUSED in RT6 (misses use SkyGradient); kept for layout parity.

    // VERBATIM rtrace.h::PrimaryRay (integer NDC -> image-plane offset -> dir) — rt_reflect.metal:318-329.
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

    // ===== PRIMARY RAY (closest-hit drain) =====
    Hit best = TraceClosestHw(ro, rd, gPrims, accel);

    // ===== PRIMARY MISS -> the GRADED SKY (SkyGradient of the primary dir) =====
    // RT6 change vs rt_reflect.metal: the primary miss is the sky gradient, NOT the flat background.
    // Matches rt_hero.comp.hlsl:333-337 / rtrace.h::RenderSceneHero:644-647.
    if (best.primIndex == HF_RT_MISS) {
        gImage[(uint)(py * width + px)] = SkyGradient(rd);
        return;
    }

    // ===== PRIMARY SHADOW RAY (any-hit OR) -> the RT3 shadowed color cS =====
    int3 sro = best.pos + int3(fxmul(best.normal.x, HF_RT_SHADOW_EPS),
                               fxmul(best.normal.y, HF_RT_SHADOW_EPS),
                               fxmul(best.normal.z, HF_RT_SHADOW_EPS));
    bool occluded = TraceAnyHitHw(sro, lightDir, gPrims, accel);
    uint cS = ShadeHitShadowed(best, lightDir, background, occluded);

    // ===== RT4 REFLECTION — if the primary surface is reflective, one bounce + the integer blend, but a
    // reflection MISS uses the GRADED SKY (SkyGradient of the reflection dir) =====
    // Translated VERBATIM from rt_hero.comp.hlsl:348-375 (matches rtrace::RenderSceneHero:657-690). The ONLY
    // delta vs rt_reflect.metal is the reflected-miss color (SkyGradient(rrd) instead of background).
    uint outColor = cS;
    int refl = ReflectivityFor(best.primIndex);
    if (refl > 0) {
        // The reflection ray: origin offset along the normal (anti-acne), dir = the mirror reflect of the
        // primary ray dir about the surface normal.
        int3 rro = best.pos + int3(fxmul(best.normal.x, HF_RT_REFL_EPS),
                                   fxmul(best.normal.y, HF_RT_REFL_EPS),
                                   fxmul(best.normal.z, HF_RT_REFL_EPS));
        int3 rrd = FxReflect(rd, best.normal);
        Hit rHit = TraceClosestHw(rro, rrd, gPrims, accel);

        // The reflected color cR: the GRADED SKY (SkyGradient of the reflection dir) on a MISS, else the RT3
        // shadowed shade of the reflected hit (its OWN shadow ray — one bounce).
        uint cR;
        if (rHit.primIndex == HF_RT_MISS) {
            cR = SkyGradient(rrd);
        } else {
            int3 rsro = rHit.pos + int3(fxmul(rHit.normal.x, HF_RT_SHADOW_EPS),
                                        fxmul(rHit.normal.y, HF_RT_SHADOW_EPS),
                                        fxmul(rHit.normal.z, HF_RT_SHADOW_EPS));
            bool rOccluded = TraceAnyHitHw(rsro, lightDir, gPrims, accel);
            cR = ShadeHitShadowed(rHit, lightDir, background, rOccluded);
        }

        // The per-channel integer blend on the UNPACKED 0..255 RGBA8 channels.
        // c = (s*(kOne-refl) + r*refl) >> kFrac  (s,r plain 0..255; refl Q16.16).
        int sr = (int)( cS        & 0xFFu), rr = (int)( cR        & 0xFFu);
        int sg = (int)((cS >> 8)  & 0xFFu), rg = (int)((cR >> 8)  & 0xFFu);
        int sb = (int)((cS >> 16) & 0xFFu), rb = (int)((cR >> 16) & 0xFFu);
        int cr = (int)(((long)sr * (long)(HF_RT_ONE - refl) + (long)rr * (long)refl) >> HF_RT_FRAC);
        int cg = (int)(((long)sg * (long)(HF_RT_ONE - refl) + (long)rg * (long)refl) >> HF_RT_FRAC);
        int cb = (int)(((long)sb * (long)(HF_RT_ONE - refl) + (long)rb * (long)refl) >> HF_RT_FRAC);
        outColor = PackRGBA8(cr, cg, cb, 255);
    }

    gImage[(uint)(py * width + px)] = outColor;
}
