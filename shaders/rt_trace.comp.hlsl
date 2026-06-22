// Slice RT1 — Hardware Ray Tracing: the DETERMINISTIC Q16.16 SOFTWARE-REFERENCE RAY TRACER, GPU TWIN (the
// BEACHHEAD of FLAGSHIP #28). ONE thread per pixel (gid.x = px, gid.y = py). Each thread generates its
// integer primary ray, BRUTE-FORCES every primitive (NO acceleration-structure cull), keeps the running
// closest by the total order (t, primIndex), and writes the integer Lambert RGBA8 to gImage[py*w+px] —
// the TraceClosest/ShadeHitInt/PrimaryRay math copied VERBATIM from engine/render/rtrace.h. Each pixel is
// INDEPENDENT (the per-ray min is a scalar register reduction), so the per-thread write is
// order-independent — race-free, NO atomics, bit-identical GPU==CPU + cross-backend (the FPX1/VT1/MC1
// integer-replay argument applied to a ray tracer).
//
// WHY BIT-IDENTICAL to the CPU reference (the make-or-break): the GPU does ZERO floating point — it
// consumes the host-snapped Q16.16 int scene (spheres/aabbs/camera SSBOs) and runs the SAME pure-integer
// fxmul ((int64)a*b >> 16, an ARITHMETIC right shift on int64) + fxdiv (((int64)a << 16) / b, integer
// divide) + the FxISqrt integer binary digit loop the header runs. A divergence here vs
// engine/render/rtrace.h is exactly what the host's GPU==CPU memcmp catches.
//
// INTEGER WIDTH (the determinism crux): fxmul/fxdiv/FxISqrt use int64_t — IDENTICAL to rtrace.h. HLSL SM6
// supports int64_t (the SAME pattern shaders/fpx_integrate.comp / swraster.comp use — DXC -spirv with the
// Int64 capability). Because of int64, this shader is VULKAN-SPIR-V-ONLY (glslc, the Metal HLSL->SPIR-V
// frontend, cannot parse int64_t) — NOT in the Metal hf_gen_msl list; on Metal the --rt1-trace showcase
// runs the CPU rtrace::RenderScene reference (byte-identical by construction). See
// metal_headless/CMakeLists.txt for the Vulkan-only comment block (the fpx_integrate.comp convention).
//
// SEAM DISCIPLINE: ABOVE the RHI seam; the only vk mention is the [[vk::binding]] decorations (same as
// fpx_integrate.comp / swraster.comp), not backend CODE symbols.

#define HF_RT_THREADS_X 8
#define HF_RT_THREADS_Y 8
#define HF_RT_FRAC 16            // MUST match rtrace.h::kFrac (fpx.h::kFrac)
static const int HF_RT_ONE = 1 << HF_RT_FRAC;       // 1.0 in Q16.16
static const int HF_RT_NOHIT = 0x7FFFFFFF;          // kRtNoHit (INT32_MAX)
static const uint HF_RT_MISS = 0xFFFFFFFFu;         // kRtMiss

// std430 mirrors of the rtrace.h POD types. fx == int (Q16.16).
struct GpuSphere { int cx, cy, cz; int radius; uint primIndex; uint _pad0, _pad1, _pad2; };  // 32 B
struct GpuAabb   { int lox, loy, loz; int hix, hiy, hiz; uint primIndex; uint _pad0; };       // 32 B

// Params (std430). Camera basis + image size + light + counts + background.
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

// VERBATIM rtrace.h::fxmul — (a*b) >> kFrac with an int64 intermediate (arithmetic shift).
int fxmul(int a, int b) {
    return (int)(((int64_t)a * (int64_t)b) >> HF_RT_FRAC);
}
// VERBATIM rtrace.h::fxdiv — ((int64)a << kFrac) / b (integer divide).
int fxdiv(int a, int b) {
    return (int)(((int64_t)a << HF_RT_FRAC) / (int64_t)b);
}
// VERBATIM rtrace.h/fpx.h::FxISqrt — floor(sqrt(v)) on a non-negative int64 (integer binary digit loop).
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

// VERBATIM rtrace.h::FxNormalize — unit Q16.16 direction via the int64 Q32.32 sum-of-squares sqrt.
int3 fxnormalize(int3 v) {
    int64_t sx = (int64_t)v.x * (int64_t)v.x;
    int64_t sy = (int64_t)v.y * (int64_t)v.y;
    int64_t sz = (int64_t)v.z * (int64_t)v.z;
    int len = (int)fxisqrt(sx + sy + sz);
    if (len == 0) return v;
    return int3(fxdiv(v.x, len), fxdiv(v.y, len), fxdiv(v.z, len));
}

// A hit record (mirrors rtrace.h::RtHit).
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

// VERBATIM rtrace.h::TraceClosest — brute-force, total-order (t, primIndex) min.
Hit TraceClosest(int3 ro, int3 rd, uint sphereCount, uint aabbCount) {
    Hit best = MissHit();
    for (uint si = 0; si < sphereCount; ++si) {
        GpuSphere s = gSpheres[si];
        Hit h;
        if (IntersectSphere(ro, rd, int3(s.cx, s.cy, s.cz), s.radius, s.primIndex, h)) {
            if (best.primIndex == HF_RT_MISS || h.t < best.t ||
                (h.t == best.t && h.primIndex < best.primIndex)) best = h;
        }
    }
    for (uint bi = 0; bi < aabbCount; ++bi) {
        GpuAabb b = gAabbs[bi];
        Hit h;
        if (IntersectAabb(ro, rd, int3(b.lox, b.loy, b.loz), int3(b.hix, b.hiy, b.hiz), b.primIndex, h)) {
            if (best.primIndex == HF_RT_MISS || h.t < best.t ||
                (h.t == best.t && h.primIndex < best.primIndex)) best = h;
        }
    }
    return best;
}

// VERBATIM rtrace.h::AlbedoFor (the 6-tint palette, integer Q16.16).
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
    int3 dir = forward;
    dir += int3(fxmul(right.x, ox), fxmul(right.y, ox), fxmul(right.z, ox));
    dir += int3(fxmul(up.x, oy), fxmul(up.y, oy), fxmul(up.z, oy));

    Hit hit = TraceClosest(eye, dir, p.counts.x, p.counts.y);
    gImage[(uint)(py * width + px)] = ShadeHitInt(hit, lightDir, p.counts.z);
}
