#pragma once
// Slice SC5 — FOLIAGE SCATTER AT SCALE (docs/GAP_CLOSING_ROADMAP.md Tier 2). Pure CPU (header-only,
// no device, no backend symbols), namespace hf::render::sc5 — the same composition-header pattern as
// render/sc3_stack.h: pcg.h / foliage.h / fpx.h are byte-UNTOUCHED, SC5 composes their public entry
// points into the "field of grass" hero at 10k+ instances.
//
// WHY: the pieces existed only in isolation — the PCG scatter pipeline (PCG1-6: jittered grid ->
// density mask -> transforms -> overlap prune), the foliage wind field + LOD (FO1-FO4: integer
// multi-gust bend via the committed kFoliageWind16 LUT; integer distance-LOD buckets), and the
// instanced render path (proven at the low-thousands in --fo5/--fo6). No showcase composed them at
// SCALE: SC5 scatters 12,123 plants (11,103 drawn after LOD cull — 10k+ through the instanced path),
// applies the deterministic wind bend at a fixed frame, buckets by camera distance, and renders the
// field through THREE per-LOD instanced draws (near=full sphere, mid=low-poly sphere, far=cube slab).
//
// THE FO-A GAP, CLOSED HERE (integer-bend -> float-matrix, HOST-SIDE, NO transcendentals): foliage.h's
// FO5/FO6 render bridges build the wind-lean quaternion with std::sin/std::cos — correct, but libm is
// NOT pinned across compilers (MSVC can route vectorized sinf through its own SVML-style libm), so the
// resulting transform bytes were never cross-COMPILER pinnable. SC5 does the conversion with the SAME
// committed integer LUT the wind itself uses (kFoliageWind16, linear-interpolated by the phase
// accumulator's next 8 bits) and only then crosses to float: every float op left in the transform
// build is +,-,*,/ or sqrt — all IEEE-754 correctly-rounded, bit-identical on every conforming
// compiler at the SSE2 baseline (no FMA contraction: MSVC x64 default /arch has no FMA, clang x86-64
// default likewise). RESULT: the 10k-instance TRANSFORM BUFFER digest is pinned identical MSVC + clang
// (kSc5XformDigest* below, verified locally against LLVM clang++). HONEST CAVEAT: on arm64 (Apple
// clang) -ffp-contract may fuse multiply-adds, so the FLOAT transform digest is an x64 pin; the
// INTEGER plant digest (kSc5PlantDigest*, over the bit-exact FO2-FO4 {pos,orient,scale,bend,lod}) is
// the cross-PLATFORM pin and is asserted on Metal too.
//
// THE PINNED ARTIFACTS (one source of truth for the test + the Vulkan/Metal handlers):
//   * kSc5ExpectedPlants = 12123 surviving instances after the overlap prune (a pure function of the
//     seed) with per-LOD counts 1312 near / 5181 mid / 4610 far / 1020 culled -> 11103 DRAWN (>= 10k).
//   * kSc5PlantDigestF120/F121 — FNV-1a-64 over the little-endian integer plant records at wind
//     frames 120/121. DIFFERENT by construction: the wind is LIVE (load-bearing), not baked-in decor.
//   * kSc5XformDigestF120/F121 — FNV-1a-64 over the concatenated per-LOD instance transform buffers
//     (the exact bytes the instanced draws consume). Same two-frame liveness property.
//
// CAPACITY NOTE (the SC5 scale audit): the instanced-lit path has NO fixed instance ceiling — the
// instance buffer is created at (count * sizeof(scene::InstanceData)) from the vector (64 B/instance;
// 11,103 drawn = ~710 KB, far under any pool). render/gpu_driven.h (BuildBatch) was considered and
// REJECTED for this composition: it emits one MDI command per OBJECT (instanceCount fixed at 1), so
// 10k foliage clones would mean 10k commands + 10k 96-byte per-draw records — the hardware-instanced
// path (3 draws total, one per LOD bucket) is the clean fit for 10k copies of 3 meshes. No capacity
// constant needed a fix.
//
// Shared by THREE call sites (the sc3_stack.h discipline):
//   1. tests/sc5_foliage_test.cpp — the pure-CPU 10k pipeline pins (counts, both digests, both
//      frames, determinism, the PCG5 shuffle-invariance re-asserted at 10k, the empty no-op).
//   2. samples/hello_triangle/main.cpp (--sc5-foliage-shot, Vulkan BMP).
//   3. metal_headless/visual_test.mm (--sc5-foliage-shot, Metal PNG).

#include <cstdint>
#include <cstring>
#include <vector>

#include "foliage/foliage.h"   // FO1-FO4 wind/placement/LOD + kFoliageWind16 (byte-untouched, composed)
#include "math/math.h"          // Mat4 / Quat / Vec3 / FromTRS / Normalize (the float render crossing)
#include "net/session.h"        // net::DigestBytes (FNV-1a-64, the pinned-golden currency)
#include "pcg/pcg.h"            // PcgGraph / PcgStream (byte-untouched, composed)
#include "sim/fpx.h"            // fx / kOne / fxmul / FxToFloat (read-only)

namespace hf::render::sc5 {

using hf::sim::fpx::fx;
using hf::sim::fpx::kOne;
using hf::sim::fpx::fxmul;

// ----- The canonical SC5 scene knobs (ONE source of truth for test + Vulkan + Metal) -----------------
// A 136x136 jittered grid (18,496 candidate cells) over a 40x40 XZ patch centred at origin, a GENTLE
// radial density mask (radius 64 — never zero inside the patch, thins toward the rim), random yaw +
// scale [0.6, 1.4], overlap prune at footprint 0.08 -> 12,123 surviving plants. Wind: the FO6 3-gust
// field. LOD: bucketed from the camera's ground position (0,0,22 — the low camera stands AT the
// field's south edge) with nearR=12 / farR=40 so all four buckets populate (1312/5181/4610/1020)
// and the far corners honestly cull.
struct Sc5Config {
    hf::pcg::PcgStream       stream;
    hf::foliage::FoliageField field;
    hf::foliage::WindField    wind;
    hf::pcg::FxVec3           lodCam;      // the FO4 LOD reference point (the camera's ground XZ)
    fx                        nearR = 0;
    fx                        farR  = 0;
    float                     baseScale = 1.0f;   // render-only: world size of a scale-1.0 plant
    fx                        leanGain  = kOne;   // render-only Q16.16 wind-lean amplification (integer!)
    float                     heightMul = 1.0f;   // render-only taller-than-wide blade stretch
};

inline Sc5Config Sc5DefaultConfig() {
    namespace pcg = hf::pcg;
    namespace fol = hf::foliage;
    Sc5Config c;
    c.stream = pcg::PcgStream{4242u, 0x5C5F011Au};

    c.field.graph.area.min = pcg::FxVec3{-20 * pcg::kOne, 0, -20 * pcg::kOne};  // 40x40 XZ patch
    c.field.graph.area.max = pcg::FxVec3{ 20 * pcg::kOne, 0,  20 * pcg::kOne};
    c.field.graph.cellsX = 136; c.field.graph.cellsZ = 136;   // 18,496 candidate cells (the scale story)
    c.field.graph.useMask = true;                             // gentle radial thinning toward the rim
    c.field.graph.mask.type   = pcg::PcgMaskType::Radial;
    c.field.graph.mask.center = pcg::FxVec3{0, 0, 0};
    c.field.graph.mask.radius = 64 * pcg::kOne;               // weight >= ~0.56 everywhere in the patch
    c.field.graph.density     = pcg::kOne;
    c.field.graph.transform.randomYaw = true;
    c.field.graph.transform.scaleLo   = pcg::kOne * 6 / 10;   // scale [0.6, 1.4]
    c.field.graph.transform.scaleHi   = pcg::kOne * 14 / 10;
    c.field.graph.prune       = true;
    c.field.graph.pruneRadius = pcg::kOne * 8 / 100;          // footprint 0.08 -> 12,123 survive

    c.wind.gustCount = 3;                                     // the FO6 hero wind (bend reads at scale)
    c.wind.master    = pcg::kOne;
    c.wind.gusts[0]  = fol::Gust{ 0x01000000, 0x00400000, 0x02000000, pcg::kOne / 4 };
    c.wind.gusts[1]  = fol::Gust{ 0x00300000, 0x01200000, 0x03000000, pcg::kOne / 7 };
    c.wind.gusts[2]  = fol::Gust{ 0x00800000, 0x00900000, 0x01800000, pcg::kOne / 10 };

    c.lodCam = pcg::FxVec3{0, 0, 22 * pcg::kOne};             // the low camera's ground position
    c.nearR  = pcg::kOne * 12;
    c.farR   = pcg::kOne * 40;

    c.baseScale = 0.35f;                                      // small dense blades
    c.leanGain  = 3 * kOne;                                   // 3.0 in Q16.16 (integer gain — see lean)
    c.heightMul = 3.0f;                                       // tall blade so the lean READS
    return c;
}

// The canonical showcase wind frames: kSc5Frame is THE shot; kSc5FrameB is the liveness control (two
// different fixed frames MUST produce different pinned digests — the wind is load-bearing).
inline constexpr uint32_t kSc5Frame  = 120u;
inline constexpr uint32_t kSc5FrameB = 121u;

// ----- The pinned artifacts (verified MSVC + LLVM clang++ locally; see the header banner) -------------
inline constexpr uint32_t kSc5ExpectedPlants   = 12123u;  // survivors after the overlap prune
inline constexpr uint32_t kSc5ExpectedNear     = 1312u;   // LOD0 (full mesh)
inline constexpr uint32_t kSc5ExpectedMid      = 5181u;   // LOD1 (low-poly)
inline constexpr uint32_t kSc5ExpectedFar      = 4610u;   // LOD2 (slab, 0.7x)
inline constexpr uint32_t kSc5ExpectedCulled   = 1020u;   // LOD3 (beyond farR)
inline constexpr uint32_t kSc5ExpectedDrawn    = 11103u;  // near+mid+far (>= 10k through the draw)
inline constexpr uint64_t kSc5PlantDigestF120  = 0xd8a9963b269d83e3ull;  // integer, cross-PLATFORM pin
inline constexpr uint64_t kSc5PlantDigestF121  = 0x90d1014a01b35dc3ull;
inline constexpr uint64_t kSc5XformDigestF120  = 0x855823fa7da53bf4ull;  // float bytes, x64 MSVC+clang pin
inline constexpr uint64_t kSc5XformDigestF121  = 0xe0c31bedc680f68aull;

// ----- Sc5BuildPlants: the bit-exact FO2 -> FO3 -> FO4 integer pipeline at a fixed wind frame ---------
// PlaceFoliage (PCG scatter->mask->transform->prune, a pure function of the seed) -> ApplyWind (the
// FO1 LUT wind at `frame`) -> AssignLods (integer XZ distance vs nearR/farR). Pure integer end to end.
inline std::vector<hf::foliage::FoliageInstance> Sc5BuildPlants(const Sc5Config& c, uint32_t frame) {
    std::vector<hf::foliage::FoliageInstance> plants = hf::foliage::PlaceFoliage(c.field, c.stream);
    hf::foliage::ApplyWind(plants, c.wind, frame);
    hf::foliage::AssignLods(plants, c.lodCam, c.nearR, c.farR);
    return plants;
}

// ----- Sc5PlantDigest: FNV-1a-64 over the little-endian INTEGER plant records -------------------------
// {pos.xyz, orient.xyzw, scale, bend, lod} per plant, in placement order — every field is int32/uint32
// (the bit-exact FO2-FO4 data), so this digest is the CROSS-PLATFORM pin (asserted on Metal too).
inline uint64_t Sc5PlantDigest(const std::vector<hf::foliage::FoliageInstance>& v) {
    std::vector<unsigned char> s;
    s.reserve(v.size() * 40);
    auto putU32 = [&s](uint32_t x) {
        s.push_back((unsigned char)(x & 0xFFu));
        s.push_back((unsigned char)((x >> 8) & 0xFFu));
        s.push_back((unsigned char)((x >> 16) & 0xFFu));
        s.push_back((unsigned char)((x >> 24) & 0xFFu));
    };
    for (const hf::foliage::FoliageInstance& p : v) {
        putU32((uint32_t)p.base.pos.x);    putU32((uint32_t)p.base.pos.y);    putU32((uint32_t)p.base.pos.z);
        putU32((uint32_t)p.base.orient.x); putU32((uint32_t)p.base.orient.y);
        putU32((uint32_t)p.base.orient.z); putU32((uint32_t)p.base.orient.w);
        putU32((uint32_t)p.base.scale);    putU32((uint32_t)p.bend);          putU32(p.lod);
    }
    return net::DigestBytes(s.data(), s.size());
}

// ----- The LUT-based lean (the FO-A gap fix: integer bend -> sin/cos with NO transcendentals) ----------
// kSc5PhasePerFx = round(2^16 / (2*pi)) maps a Q16.16 radian angle onto the kFoliageWind16 phase
// convention (one full turn == 2^32). Sc5SinQ15 indexes the committed 256-entry table by the phase's
// top 8 bits and linearly interpolates by the NEXT 8 bits (pure int32 lerp; C++20 arithmetic >> on the
// signed delta) — ~Q15 out, smooth to ~0.09 degree steps. Both are pure integer; the ONLY float ops in
// the lean are two exact int->float conversions and two exact divides by 32768.
inline constexpr int32_t kSc5PhasePerFx = 10430;   // round(2^16 / (2*pi))

inline int32_t Sc5SinQ15(uint32_t phase) {
    const uint32_t idx  = (phase >> 24) & 255u;
    const int32_t  a    = hf::foliage::kFoliageWind16[idx];
    const int32_t  b    = hf::foliage::kFoliageWind16[(idx + 1u) & 255u];
    const int32_t  frac = (int32_t)((phase >> 16) & 255u);
    return a + (((b - a) * frac) >> 8);
}

// sin/cos of HALF the (gain-scaled) bend angle — exactly what the lean quaternion needs. The gain is a
// Q16.16 INTEGER (fxmul) so the amplification itself stays bit-exact; negative bends wrap the uint32
// phase, which IS the LUT's periodicity. cos(x) == sin(x + quarter turn) == phase + 0x40000000.
inline void Sc5LeanSinCos(fx bend, fx gain, float& sinHalf, float& cosHalf) {
    const fx half = fxmul(bend, gain) / 2;
    const uint32_t phase = (uint32_t)((int64_t)half * (int64_t)kSc5PhasePerFx);
    sinHalf = (float)Sc5SinQ15(phase) / 32768.0f;
    cosHalf = (float)Sc5SinQ15(phase + 0x40000000u) / 32768.0f;
}

// ----- Sc5RenderSet: the per-LOD-bucket instance transform buffers (the draw structure) ---------------
// lod[0]=near (full mesh), lod[1]=mid (low-poly), lod[2]=far (slab, FoliageLodScale 0.7x); LOD3 culled.
// counts[] tallies ALL four buckets. One hardware-instanced draw per non-empty bucket = <= 3 draws for
// the whole 10k+ field.
struct Sc5RenderSet {
    std::vector<math::Mat4> lod[3];
    uint32_t counts[4] = {0, 0, 0, 0};
    uint32_t Drawn() const { return counts[0] + counts[1] + counts[2]; }
};

// ----- Sc5BuildRenderSet: the ONE float crossing (render-only, the FO5/FO6 visresolve-bar precedent) ---
// Per non-culled plant: t = FxToFloat(pos) lifted by half the blade height (the blade RESTS ON the
// ground; meshes are unit-diameter, so world height == the Y scale); yaw = the placement quaternion
// renormalized in float; lean = the LUT-based half-angle quat about +X (Sc5LeanSinCos — NO libm);
// q = lean*yaw (Hamilton product, spelled out — the FO5 convention); scale = FxToFloat(base.scale) *
// baseScale * FoliageLodScale(lod), stretched heightMul-tall in local Y. Every float op is IEEE
// +,-,*,/ or sqrt -> the output bytes are the x64 cross-compiler pin (see banner). Deterministic
// bucket order: plants are walked in placement order, appended to their bucket.
inline Sc5RenderSet Sc5BuildRenderSet(const std::vector<hf::foliage::FoliageInstance>& plants,
                                      const Sc5Config& c) {
    namespace fpx = hf::sim::fpx;
    Sc5RenderSet out;
    for (const hf::foliage::FoliageInstance& inst : plants) {
        ++out.counts[inst.lod & 3u];
        if (inst.lod >= 3u) continue;   // LOD3 = culled (beyond farR) — never rendered
        const math::Quat yaw = math::Normalize(math::Quat{
            fpx::FxToFloat(inst.base.orient.x), fpx::FxToFloat(inst.base.orient.y),
            fpx::FxToFloat(inst.base.orient.z), fpx::FxToFloat(inst.base.orient.w)});
        float hs, hc;
        Sc5LeanSinCos(inst.bend, c.leanGain, hs, hc);
        const math::Quat lean{hs, 0.0f, 0.0f, hc};   // rotate about +X by the gain-scaled bend
        const math::Quat q{
            lean.w * yaw.x + lean.x * yaw.w + lean.y * yaw.z - lean.z * yaw.y,
            lean.w * yaw.y - lean.x * yaw.z + lean.y * yaw.w + lean.z * yaw.x,
            lean.w * yaw.z + lean.x * yaw.y - lean.y * yaw.x + lean.z * yaw.w,
            lean.w * yaw.w - lean.x * yaw.x - lean.y * yaw.y - lean.z * yaw.z};
        const float s  = fpx::FxToFloat(inst.base.scale) * c.baseScale *
                         hf::foliage::FoliageLodScale(inst.lod);
        const float sy = s * c.heightMul;
        const math::Vec3 t{fpx::FxToFloat(inst.base.pos.x),
                           fpx::FxToFloat(inst.base.pos.y) + 0.5f * sy,
                           fpx::FxToFloat(inst.base.pos.z)};
        out.lod[inst.lod].push_back(math::FromTRS(t, q, math::Vec3{s, sy, s}));
    }
    return out;
}

// ----- Sc5TransformDigest: FNV-1a-64 over the concatenated per-LOD instance transform buffers ---------
// EXACTLY the bytes the three instanced draws consume (lod0 | lod1 | lod2, 64 B per instance, raw
// little-endian float bytes on every supported target). The deterministic render artifact: pinned
// identical MSVC + clang on x64 (kSc5XformDigest*); an x64 pin, not asserted on arm64 (banner caveat).
inline uint64_t Sc5TransformDigest(const Sc5RenderSet& rs) {
    std::vector<unsigned char> bytes;
    size_t total = 0;
    for (int l = 0; l < 3; ++l) total += rs.lod[l].size() * sizeof(math::Mat4);
    bytes.reserve(total);
    for (int l = 0; l < 3; ++l) {
        const size_t n = rs.lod[l].size() * sizeof(math::Mat4);
        const size_t off = bytes.size();
        bytes.resize(off + n);
        if (n) std::memcpy(bytes.data() + off, rs.lod[l].data(), n);
    }
    return net::DigestBytes(bytes.data(), bytes.size());
}

}  // namespace hf::render::sc5
