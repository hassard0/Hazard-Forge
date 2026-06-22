#pragma once
// Slice RT1 — Hardware Ray Tracing: the DETERMINISTIC Q16.16 SOFTWARE REFERENCE RAY TRACER (the BEACHHEAD
// of FLAGSHIP #28: HARDWARE RAY TRACING, DETERMINISTICALLY RECONCILED, hf::render::rtrace). Pure CPU
// (header-only, NO device, NO backend symbols, NO <cmath> on the bit-exact path). Namespace
// hf::render::rtrace. The STRUCTURAL TWIN of the VT1/MC1/FPX1 integer beachhead (render/vt.h, render/mc.h,
// sim/fpx.h): a pure-integer per-pixel ray trace proven GPU==CPU BIT-EXACT, with a cross-backend
// BIT-IDENTICAL integer golden.
//
// WHAT THIS IS: the GOLDEN + the no-HW fallback. The closest-hit search BRUTE-FORCES every primitive with
// NO acceleration-structure cull — deliberately: in RT2 the HW inline-ray-query path (AABB-gated by the
// driver's BVH) will be memcmp'd against THIS no-cull reference, so an insufficient AABB margin that makes
// the driver skip a true hit FAILS LOUDLY (the candidate-completeness oracle). RT1 ships only the
// reference half of that proof; RT2 adds the HW half.
//
// THE CROSS-BACKEND CRUX (the make-or-break for GPU==CPU, like fpx.h's host-snap / swraster.h's integer
// ScreenVerts): every ray/primitive intersection + shade is PURE-INTEGER Q16.16 — fpx::fxmul ((int64)a*b
// >> 16, an ARITHMETIC right shift on int64, deterministic + identical on every compiler/vendor),
// fxdiv (((int64)a << kFrac) / b, integer division — deterministic, identical DXC<->CPU) for the AABB
// slab test, and fpx::FxISqrt (the integer binary digit loop, == render/mc.h::ISqrt) for the sphere
// discriminant root + the shade-normal normalize. NEVER any float, std::sqrt, or <cmath> on the bit-exact
// path. Each pixel/ray is INDEPENDENT (the closest-hit min is a sequential scalar reduction in registers,
// one thread per ray -> NO atomics), so the GPU twin shaders/rt_trace.comp.hlsl writes each pixel
// race-free and two runs are byte-identical.
//
// THE FIXED-POINT FORMAT: fx = int32_t in Q16.16 (kFrac=16, kOne=1<<16) — the fpx.h format reused
// verbatim. Positions are bounded to +-32768 world units; all products use an int64 intermediate -> NO
// overflow within that bound. The closest-hit key is the TOTAL ORDER (t, primIndex): smaller t wins;
// equal t -> smaller primIndex wins (a deterministic tie-break) -> order-independent ⇒ identical
// regardless of primitive storage order, vendor, or thread scheduling.
//
// REUSE MAP (file:line): fpx::fx/kFrac/kOne/fxmul (engine/sim/fpx.h:45-56) + fpx::FxVec3/FxAdd/FxSub/
// FxScale (58-72) + fpx::FxISqrt (78-89), all #included READ-ONLY (sim header FROZEN). The fxdiv helper is
// DEFINED HERE (the spec forbids editing fpx.h for a tiny render-only helper). render/swraster.h's
// PackSw/UnpackSw (70-83) is the packed-key precedent (here packing is OPTIONAL — the per-ray min is a
// scalar register reduction, so we store the full RtHit). NO sim header / sim shader / sim golden touched
// (RT is render-only).
//
// SEAM DISCIPLINE: ZERO backend (vk*/MTL*/mtl::) symbols. NO GPU, NO new RHI here. Mentions of "GPU"/RT2
// are doc-only.

#include <cstdint>
#include <span>
#include <vector>

#include "sim/fpx.h"  // fx / kFrac / kOne / fxmul / FxVec3 / FxAdd / FxSub / FxScale / FxISqrt (READ-ONLY)

namespace hf::render::rtrace {

// ----- Reuse the Q16.16 fixed-point scalar + vector from sim/fpx.h (frozen) -------------------------
using fx = hf::sim::fpx::fx;
using FxVec3 = hf::sim::fpx::FxVec3;
inline constexpr int kFrac = hf::sim::fpx::kFrac;   // 16 fractional bits
inline constexpr fx  kOne  = hf::sim::fpx::kOne;    // 1.0 in Q16.16 (65536)

using hf::sim::fpx::FxAdd;
using hf::sim::fpx::FxSub;
using hf::sim::fpx::FxScale;
using hf::sim::fpx::fxmul;
using hf::sim::fpx::FxISqrt;

// ----- The render-only integer division helper (DEFINED HERE — fpx.h is frozen) ---------------------
// fxdiv(a,b) = a / b in Q16.16 = ((int64)a << kFrac) / b — a single int64 widen + an integer divide
// (deterministic, truncates toward ZERO identically on every compiler/vendor + DXC; the slab test only
// ever divides by a NON-ZERO ray-dir component, guarded by the caller). NOT in fpx.h: the spec forbids
// editing the frozen sim header for a render-only helper, so it lives in rtrace.h.
inline fx fxdiv(fx a, fx b) {
    return (fx)(((int64_t)a << kFrac) / (int64_t)b);
}

// fxdot(a,b) = a.x*b.x + a.y*b.y + a.z*b.z in Q16.16 (int64-intermediate sum of fxmuls — the fpx FxDot
// idiom; pure integer).
inline fx FxDot(const FxVec3& a, const FxVec3& b) {
    return fxmul(a.x, b.x) + fxmul(a.y, b.y) + fxmul(a.z, b.z);
}

// RtNormalize(v) -> a unit-length (Q16.16) direction. len = floor-sqrt of the int64 Q32.32 sum-of-squares
// (== fpx::FxLength); if len==0 returns v unchanged (the caller guards the degenerate case). Each axis is
// fxdiv'd by the length (pure integer). Deterministic, NO float. NAMED RtNormalize (not FxNormalize) to
// avoid an ADL collision with fpx::FxNormalize (the fx vector type is shared via the `using` alias).
inline FxVec3 RtNormalize(const FxVec3& v) {
    int64_t sx = (int64_t)v.x * (int64_t)v.x;
    int64_t sy = (int64_t)v.y * (int64_t)v.y;
    int64_t sz = (int64_t)v.z * (int64_t)v.z;
    fx len = (fx)FxISqrt(sx + sy + sz);
    if (len == 0) return v;
    return FxVec3{fxdiv(v.x, len), fxdiv(v.y, len), fxdiv(v.z, len)};
}

// ----- Primitives (Q16.16 only) ---------------------------------------------------------------------
// A sphere primitive. primIndex is the GLOBAL primitive index (the (t,primIndex) tie-break key + the
// ShadeHitInt albedo tint key).
struct RtSphere {
    FxVec3   center;
    fx       radius = 0;
    uint32_t primIndex = 0;
};

// An axis-aligned box primitive (lo <= hi per axis). primIndex is the GLOBAL primitive index.
struct RtAabb {
    FxVec3   lo;
    FxVec3   hi;
    uint32_t primIndex = 0;
};

// ----- The ray ---------------------------------------------------------------------------------------
// origin + dir (Q16.16). dir is NOT required unit-length — t is in units of |dir|, consistent within a
// ray, so the per-ray closest-hit min is valid. The scene/ray builders reject dir==0 (degenerate).
struct RtRay {
    FxVec3 origin;
    FxVec3 dir;
};

// ----- The hit record + the miss sentinel ------------------------------------------------------------
inline constexpr uint32_t kRtMiss   = 0xFFFFFFFFu;  // primIndex of a miss
inline constexpr fx        kRtNoHit = 0x7FFFFFFF;   // t of a miss (INT32_MAX, the largest possible t key)

struct RtHit {
    fx       t = kRtNoHit;          // ray parameter (Q16.16, units of |dir|); kRtNoHit on miss
    uint32_t primIndex = kRtMiss;   // global primitive index; kRtMiss on miss
    FxVec3   pos;                   // world hit position (Q16.16)
    FxVec3   normal;                // unit surface normal at the hit (Q16.16)
};

inline RtHit RtMissHit() { return RtHit{kRtNoHit, kRtMiss, FxVec3{}, FxVec3{}}; }

// ----- The scene (SoA spans of each primitive + a fixed directional light + background) --------------
struct RtScene {
    std::span<const RtSphere> spheres;
    std::span<const RtAabb>   aabbs;
    FxVec3                    lightDir;       // UNIT directional light direction TOWARD the surface->light
    uint32_t                  background = 0; // RGBA8 packed (0xAABBGGRR), the miss color
};

// ----- The integer camera basis (NO float) -----------------------------------------------------------
// A pinhole camera: eye + an orthonormal basis (right/up/forward, all Q16.16 unit vectors) + the
// half-extents of the image plane at unit forward distance (Q16.16). Primary rays are generated by
// integer NDC -> a fxmul'd image-plane offset (no float, no tan() — the half-extents bake the FOV).
struct RtCamera {
    FxVec3 eye;
    FxVec3 right;     // +x of the image plane (unit)
    FxVec3 up;        // +y of the image plane (unit)
    FxVec3 forward;   // view direction (unit)
    fx     halfW = 0; // image-plane half-width  at forward distance 1 (Q16.16) — bakes the horizontal FOV
    fx     halfH = 0; // image-plane half-height at forward distance 1 (Q16.16) — bakes the vertical FOV
};

// ===== Intersection ==================================================================================

// IntersectSphere — fx ray/sphere quadratic. With oc = origin - center, the quadratic in t is
//   a = dir.dir,  b = 2*(oc.dir),  c = oc.oc - r^2,  disc = b^2 - 4ac.
// disc < 0 -> miss. Else the nearest NON-NEGATIVE root t = (-b - sqrt(disc)) / (2a) (and the far root if
// the near is behind the origin). sqrt(disc) via FxISqrt on the int64 Q32.32 discriminant. Writes
// t/pos/normal; returns whether it hit (with t >= 0).
inline bool IntersectSphere(const RtRay& ray, const RtSphere& s, RtHit& out) {
    FxVec3 oc = FxSub(ray.origin, s.center);
    fx a = FxDot(ray.dir, ray.dir);
    if (a <= 0) return false;            // degenerate dir (guarded at build; defensive)
    fx half_b = FxDot(oc, ray.dir);      // = b/2 (avoids the 2x; disc' = half_b^2 - a*c)
    fx c = FxDot(oc, oc) - fxmul(s.radius, s.radius);
    // disc' = half_b^2 - a*c, formed in int64 Q32.32 so FxISqrt yields the Q16.16 root.
    int64_t hb = (int64_t)half_b * (int64_t)half_b;     // Q32.32
    int64_t ac = (int64_t)a * (int64_t)c;               // Q32.32
    int64_t disc = hb - ac;
    if (disc < 0) return false;
    fx sq = (fx)FxISqrt(disc);           // sqrt(disc') in Q16.16
    // t = (-half_b -+ sq) / a, pick the nearest non-negative root.
    fx tNear = fxdiv(-half_b - sq, a);
    fx tFar  = fxdiv(-half_b + sq, a);
    fx t;
    if (tNear >= 0)      t = tNear;
    else if (tFar >= 0)  t = tFar;
    else                 return false;   // both roots behind the origin
    out.t = t;
    out.primIndex = s.primIndex;
    out.pos = FxAdd(ray.origin, FxScale(ray.dir, t));
    out.normal = RtNormalize(FxSub(out.pos, s.center));
    return true;
}

// IntersectAabb — fx slab test (the integer ray/AABB). For each axis the entry/exit t are
// (lo-origin)/dir and (hi-origin)/dir (fxdiv); axes with dir==0 only hit if origin is within [lo,hi] on
// that axis. tEnter = max of per-axis near, tExit = min of per-axis far; a hit needs tEnter <= tExit AND
// tExit >= 0. The hit t is tEnter if >= 0 (ray starts outside) else tExit (ray starts inside). The face
// normal is the axis whose slab produced the chosen entry, signed against the ray.
inline bool IntersectAabb(const RtRay& ray, const RtAabb& box, RtHit& out) {
    fx o[3]  = {ray.origin.x, ray.origin.y, ray.origin.z};
    fx d[3]  = {ray.dir.x,    ray.dir.y,    ray.dir.z};
    fx lo[3] = {box.lo.x, box.lo.y, box.lo.z};
    fx hi[3] = {box.hi.x, box.hi.y, box.hi.z};

    fx tEnter = (fx)0x80000000;   // -inf-ish (INT32_MIN)
    fx tExit  = kRtNoHit;         // +inf-ish (INT32_MAX)
    int enterAxis = 0;
    int enterSign = 0;            // -1 if the +face (dir<0 hit the hi slab first), +1 if the -face

    for (int ax = 0; ax < 3; ++ax) {
        if (d[ax] == 0) {
            // Ray parallel to this slab: miss unless the origin is inside it.
            if (o[ax] < lo[ax] || o[ax] > hi[ax]) return false;
            continue;
        }
        fx t1 = fxdiv(lo[ax] - o[ax], d[ax]);   // t to the lo plane
        fx t2 = fxdiv(hi[ax] - o[ax], d[ax]);   // t to the hi plane
        // near = closer plane, far = farther plane; track which face the NEAR plane is.
        fx tNear, tFar;
        int nearSign;   // the normal sign along this axis at the near plane
        if (t1 <= t2) { tNear = t1; tFar = t2; nearSign = -1; }  // entered through the lo (−) face
        else          { tNear = t2; tFar = t1; nearSign = +1; }  // entered through the hi (+) face
        if (tNear > tEnter) { tEnter = tNear; enterAxis = ax; enterSign = nearSign; }
        if (tFar  < tExit)  { tExit = tFar; }
        if (tEnter > tExit) return false;       // slabs disjoint -> miss
    }
    if (tExit < 0) return false;                // box entirely behind the origin

    fx t;
    FxVec3 n{0, 0, 0};
    if (tEnter >= 0) {
        t = tEnter;
        if (enterAxis == 0) n.x = (fx)(enterSign * kOne);
        else if (enterAxis == 1) n.y = (fx)(enterSign * kOne);
        else n.z = (fx)(enterSign * kOne);
    } else {
        // Origin inside the box: the hit is the EXIT; the normal is left as the entry face's (a degenerate
        // inside-hit; the showcase rays start outside, so this is the defensive branch).
        t = tExit;
        if (enterAxis == 0) n.x = (fx)(enterSign * kOne);
        else if (enterAxis == 1) n.y = (fx)(enterSign * kOne);
        else n.z = (fx)(enterSign * kOne);
    }
    out.t = t;
    out.primIndex = box.primIndex;
    out.pos = FxAdd(ray.origin, FxScale(ray.dir, t));
    out.normal = n;
    return true;
}

// ===== Closest-hit (brute-force, NO cull, total-order (t,primIndex) min) =============================
// TraceClosest iterates EVERY primitive (no acceleration structure), keeping the running closest by the
// total order (t, primIndex): a candidate replaces the best if its t is strictly smaller, OR its t is
// equal and its primIndex is smaller. Order-independent ⇒ identical regardless of primitive storage
// order, vendor, or thread scheduling (the RT2 candidate-completeness oracle rests on this).
inline RtHit TraceClosest(const RtRay& ray, const RtScene& scene) {
    RtHit best = RtMissHit();
    auto consider = [&](const RtHit& h) {
        if (h.primIndex == kRtMiss) return;
        if (best.primIndex == kRtMiss ||
            h.t < best.t ||
            (h.t == best.t && h.primIndex < best.primIndex)) {
            best = h;
        }
    };
    for (const RtSphere& s : scene.spheres) {
        RtHit h;
        if (IntersectSphere(ray, s, h)) consider(h);
    }
    for (const RtAabb& b : scene.aabbs) {
        RtHit h;
        if (IntersectAabb(ray, b, h)) consider(h);
    }
    return best;
}

// ===== Integer shade =================================================================================
// A small fixed albedo palette keyed by primIndex (Q16.16 r,g,b in [0,kOne]). Deterministic + integer.
inline FxVec3 AlbedoFor(uint32_t primIndex) {
    // 6 warm/cool tints; modulo so any primIndex maps deterministically.
    static const FxVec3 kPalette[6] = {
        {(fx)(kOne * 78 / 100), (fx)(kOne * 30 / 100), (fx)(kOne * 26 / 100)},  // 0: warm red
        {(fx)(kOne * 28 / 100), (fx)(kOne * 52 / 100), (fx)(kOne * 80 / 100)},  // 1: cool blue
        {(fx)(kOne * 36 / 100), (fx)(kOne * 70 / 100), (fx)(kOne * 38 / 100)},  // 2: green
        {(fx)(kOne * 82 / 100), (fx)(kOne * 70 / 100), (fx)(kOne * 28 / 100)},  // 3: amber
        {(fx)(kOne * 64 / 100), (fx)(kOne * 40 / 100), (fx)(kOne * 72 / 100)},  // 4: violet
        {(fx)(kOne * 60 / 100), (fx)(kOne * 60 / 100), (fx)(kOne * 62 / 100)},  // 5: grey
    };
    return kPalette[primIndex % 6u];
}

// Pack 4 bytes (already clamped to [0,255]) into RGBA8 (0xAABBGGRR — R low byte, A high byte). Matches
// the showcase image writers' little-endian RGBA8 expectation.
inline uint32_t PackRGBA8(int32_t r, int32_t g, int32_t b, int32_t a) {
    auto cl = [](int32_t v) -> uint32_t { return (uint32_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); };
    return cl(r) | (cl(g) << 8) | (cl(b) << 16) | (cl(a) << 24);
}

// ShadeHitInt — integer Lambert. shade = albedo * (ambient + (1-ambient)*max(dot(n, L), 0)), quantized to
// RGBA8. L is scene.lightDir (TOWARD the light). A MISS -> scene.background. PURE INTEGER (fxmul + a
// fixed ambient + a (val * 255) >> kFrac quantize). Deterministic + identical cross-backend.
inline uint32_t ShadeHitInt(const RtHit& hit, const RtScene& scene) {
    if (hit.primIndex == kRtMiss) return scene.background;
    fx ndl = FxDot(hit.normal, scene.lightDir);
    if (ndl < 0) ndl = 0;                          // clamp max(dot,0)
    const fx ambient = (fx)(kOne * 18 / 100);      // 0.18 ambient floor
    fx diffuse = ambient + fxmul(kOne - ambient, ndl);   // [ambient, 1]
    FxVec3 alb = AlbedoFor(hit.primIndex);
    // quantize: (albedo * diffuse) in Q16.16 -> [0,255] via (v * 255) >> kFrac.
    auto q = [&](fx ch) -> int32_t {
        fx lit = fxmul(ch, diffuse);                       // Q16.16 in [0,1]
        return (int32_t)(((int64_t)lit * 255) >> kFrac);   // [0,255]
    };
    return PackRGBA8(q(alb.x), q(alb.y), q(alb.z), 255);
}

// ===== Primary-ray gen + render ======================================================================
// PrimaryRay — the integer primary ray for pixel (px,py) of a width x height image. NDC in Q16.16:
//   ndcX = (2*(px + 0.5)/width  - 1) * halfW   ->  the image-plane x offset (Q16.16)
//   ndcY = (1 - 2*(py + 0.5)/height) * halfH   ->  the image-plane y offset (Q16.16, y DOWN in image)
// computed as integer ratios scaled to Q16.16 (no float), then dir = forward + right*ndcX + up*ndcY.
inline RtRay PrimaryRay(const RtCamera& cam, uint32_t px, uint32_t py, uint32_t width, uint32_t height) {
    // (px + 0.5) / width in Q16.16 = ((2*px + 1) << kFrac) / (2*width). Then map to [-1,1]*half.
    int64_t sxNum = ((int64_t)(2 * (int64_t)px + 1) << kFrac);
    fx sx = (fx)(sxNum / (int64_t)(2 * (int64_t)width));      // (px+0.5)/width in Q16.16, [0,1]
    int64_t syNum = ((int64_t)(2 * (int64_t)py + 1) << kFrac);
    fx sy = (fx)(syNum / (int64_t)(2 * (int64_t)height));     // (py+0.5)/height in Q16.16, [0,1]
    fx ndcX = (sx * 2) - kOne;                  // [-1,1]
    fx ndcY = kOne - (sy * 2);                  // [-1,1], y up
    fx ox = fxmul(ndcX, cam.halfW);             // image-plane x offset
    fx oy = fxmul(ndcY, cam.halfH);             // image-plane y offset
    FxVec3 dir = cam.forward;
    dir = FxAdd(dir, FxScale(cam.right, ox));
    dir = FxAdd(dir, FxScale(cam.up, oy));
    return RtRay{cam.eye, dir};
}

// RenderScene — integer primary-ray gen -> TraceClosest -> ShadeHitInt for every pixel, writing RGBA8 to
// outRGBA8 (row-major, top row first, size == width*height). Pure CPU, deterministic. The GPU twin
// shaders/rt_trace.comp.hlsl copies this math VERBATIM (one thread per pixel) so the GPU readback is
// bit-identical, proven memcmp. Returns the number of pixels that HIT a primitive (for the proof line).
inline uint32_t RenderScene(const RtScene& scene, const RtCamera& cam, uint32_t width, uint32_t height,
                            std::span<uint32_t> outRGBA8) {
    uint32_t hits = 0;
    for (uint32_t py = 0; py < height; ++py) {
        for (uint32_t px = 0; px < width; ++px) {
            RtRay ray = PrimaryRay(cam, px, py, width, height);
            RtHit hit = TraceClosest(ray, scene);
            if (hit.primIndex != kRtMiss) ++hits;
            outRGBA8[(size_t)py * width + px] = ShadeHitInt(hit, scene);
        }
    }
    return hits;
}

// ===== The fixed RT1 showcase scene + camera =========================================================
// A small analytic scene: a large GROUND AABB + a few spheres + a couple of boxes, lit by a fixed
// directional light, viewed from a fixed integer pinhole camera. The primitive set + camera are PINNED
// (the cross-vendor golden rests on it). The owning storage lives in the returned struct (the spans in
// `scene` point into it — keep the RtScene1 alive while tracing).
struct RtScene1 {
    std::vector<RtSphere> spheres;
    std::vector<RtAabb>   aabbs;
    RtScene               scene;
    RtCamera              camera;
};

inline fx F(int wholeNum, int wholeDen) {  // a Q16.16 fraction wholeNum/wholeDen
    return (fx)(((int64_t)wholeNum << kFrac) / wholeDen);
}

inline RtScene1 BuildRt1Scene() {
    RtScene1 r;

    // --- primitives (Q16.16) ---------------------------------------------------------------------
    // A large ground slab AABB (primIndex 0) centered near the origin, thin in Y.
    r.aabbs.push_back(RtAabb{
        FxVec3{F(-20, 1), F(-3, 1), F(-20, 1)},   // lo
        FxVec3{F(20, 1),  F(-1, 1), F(20, 1)},    // hi
        /*primIndex*/ 0});
    // An upright box (primIndex 1) sitting on the ground, off to the left.
    r.aabbs.push_back(RtAabb{
        FxVec3{F(-5, 2), F(-1, 1), F(-1, 1)},     // lo
        FxVec3{F(-1, 2), F(3, 2),  F(3, 2)},      // hi
        /*primIndex*/ 1});

    // Three spheres at various depths (primIndices 2,3,4).
    r.spheres.push_back(RtSphere{FxVec3{F(0, 1), F(0, 1), F(1, 1)},  F(3, 2), /*primIndex*/ 2});
    r.spheres.push_back(RtSphere{FxVec3{F(5, 2), F(-1, 4), F(5, 2)}, F(5, 4), /*primIndex*/ 3});
    r.spheres.push_back(RtSphere{FxVec3{F(-7, 2), F(1, 2), F(7, 2)}, F(1, 1), /*primIndex*/ 4});

    // --- the scene wrapper -----------------------------------------------------------------------
    r.scene.spheres = std::span<const RtSphere>(r.spheres);
    r.scene.aabbs   = std::span<const RtAabb>(r.aabbs);
    // Directional light from the upper-front-right, normalized to a Q16.16 unit vector.
    r.scene.lightDir = RtNormalize(FxVec3{F(4, 10), F(8, 10), F(-3, 10)});
    // Background: a cool dim sky-grey.
    r.scene.background = PackRGBA8(34, 40, 56, 255);

    // --- the integer pinhole camera (pinned) -----------------------------------------------------
    // Eye behind/above the origin looking toward +Z. Orthonormal integer basis (the world is axis-aligned
    // so the basis IS the world axes — exactly unit, no normalization drift). HalfW/HalfH bake a ~53deg
    // horizontal FOV (the image plane is +-0.7 at unit forward for a 16:9-ish aspect baked into halfH).
    r.camera.eye     = FxVec3{F(0, 1), F(2, 1), F(-9, 1)};
    r.camera.right   = FxVec3{kOne, 0, 0};      // +X
    r.camera.up      = FxVec3{0, kOne, 0};       // +Y
    r.camera.forward = FxVec3{0, 0, kOne};       // +Z
    r.camera.halfW   = F(7, 10);                 // 0.70
    r.camera.halfH   = F(7, 10);                 // 0.70 (square; the image aspect handles the ratio)
    return r;
}

// ===== Slice RT3 — DETERMINISTIC RT HARD SHADOWS (APPEND-ONLY; RT1's code above is BYTE-FROZEN) ========
// After the primary TraceClosest finds the surface point, cast a SECOND ray (a SHADOW ray) from
// hit.pos + normal*kRtShadowEps (the offset kills self-shadow acne) toward scene.lightDir. The point is
// in shadow iff ANY primitive is intersected with t > kRtShadowMinT (a directional light -> no far
// bound). "Any hit" is an order-independent boolean OR over OUR fx intersections, so the HW any-hit
// (drain candidates, OR our fx hits) == the SW brute-force any-hit == the CPU, bit-exact, regardless of
// BVH traversal order. The shade gates the diffuse term: ambient + (occluded ? 0 : diffuse), so shadowed
// pixels keep only ambient (visibly darker) -> the integer shadowed image is strict-zero cross-vendor.
// PURE INTEGER Q16.16 — the same frozen rtrace:: fx helpers, NO float on the bit-exact path.

inline constexpr fx kRtShadowEps  = kOne / 256;   // surface offset along the normal (anti-acne)
inline constexpr fx kRtShadowMinT = kOne / 1024;  // ignore shadow hits closer than this (self-hit guard)

// TraceAnyHit — the OCCLUSION test. Brute-force EVERY primitive; return true on the FIRST primitive whose
// IntersectSphere/IntersectAabb yields a hit with t > minT. The result is a boolean OR over the fx hits —
// ORDER-INDEPENDENT (presence of an occluder, NOT an order-dependent min), so it CAN early-out on the
// first qualifying hit WITHOUT breaking determinism: the HW any-hit (which may early-out on its first
// candidate) and the SW brute-force (which may scan all) yield the SAME boolean. NO closest-hit search.
inline bool TraceAnyHit(const RtRay& ray, const RtScene& scene, fx minT) {
    for (const RtSphere& s : scene.spheres) {
        RtHit h;
        if (IntersectSphere(ray, s, h) && h.t > minT) return true;
    }
    for (const RtAabb& b : scene.aabbs) {
        RtHit h;
        if (IntersectAabb(ray, b, h) && h.t > minT) return true;
    }
    return false;
}

// ShadeHitShadowed — the ShadeHitInt body with the DIFFUSE term gated by occlusion. A MISS -> background.
// Else diffuse = ambient + (occluded ? 0 : fxmul(1-ambient, max(dot(n,L),0))) — albedo-tinted + quantized
// identically to ShadeHitInt. When NOT occluded this is BYTE-IDENTICAL to ShadeHitInt; when occluded the
// pixel keeps only the ambient floor (strictly darker). PURE INTEGER, deterministic, cross-backend exact.
inline uint32_t ShadeHitShadowed(const RtHit& hit, const RtScene& scene, bool occluded) {
    if (hit.primIndex == kRtMiss) return scene.background;
    fx ndl = FxDot(hit.normal, scene.lightDir);
    if (ndl < 0) ndl = 0;                          // clamp max(dot,0)
    const fx ambient = (fx)(kOne * 18 / 100);      // 0.18 ambient floor (== ShadeHitInt)
    fx lambert = occluded ? 0 : fxmul(kOne - ambient, ndl);  // the GATED diffuse term
    fx diffuse = ambient + lambert;                // [ambient, 1]; occluded -> exactly ambient
    FxVec3 alb = AlbedoFor(hit.primIndex);
    auto q = [&](fx ch) -> int32_t {
        fx lit = fxmul(ch, diffuse);                       // Q16.16 in [0,1]
        return (int32_t)(((int64_t)lit * 255) >> kFrac);   // [0,255]
    };
    return PackRGBA8(q(alb.x), q(alb.y), q(alb.z), 255);
}

// RenderSceneShadowed — per pixel: PrimaryRay -> TraceClosest; if it HIT, build the shadow ray
// {hit.pos + normal*kRtShadowEps, lightDir} and occluded = TraceAnyHit(shadowRay, scene, kRtShadowMinT);
// out = ShadeHitShadowed(hit, scene, occluded). A MISS shades to background (occluded irrelevant). Pure
// CPU, deterministic. The GPU twin shaders/rt_shadow.comp copies this math VERBATIM (one thread per pixel)
// so the HW readback is bit-identical, proven memcmp. Returns the number of SHADOWED pixels (a primary HIT
// whose surface was occluded — the real-occlusion proof line).
inline uint32_t RenderSceneShadowed(const RtScene& scene, const RtCamera& cam, uint32_t width,
                                    uint32_t height, std::span<uint32_t> outRGBA8) {
    uint32_t shadowed = 0;
    for (uint32_t py = 0; py < height; ++py) {
        for (uint32_t px = 0; px < width; ++px) {
            RtRay ray = PrimaryRay(cam, px, py, width, height);
            RtHit hit = TraceClosest(ray, scene);
            bool occluded = false;
            if (hit.primIndex != kRtMiss) {
                // Shadow-ray origin = hit.pos offset along the surface normal by kRtShadowEps (anti-acne);
                // direction = the directional light direction (TOWARD the light).
                RtRay shadowRay;
                shadowRay.origin = FxAdd(hit.pos, FxScale(hit.normal, kRtShadowEps));
                shadowRay.dir = scene.lightDir;
                occluded = TraceAnyHit(shadowRay, scene, kRtShadowMinT);
                if (occluded) ++shadowed;
            }
            outRGBA8[(size_t)py * width + px] = ShadeHitShadowed(hit, scene, occluded);
        }
    }
    return shadowed;
}

}  // namespace hf::render::rtrace
