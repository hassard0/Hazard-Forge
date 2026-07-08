#pragma once
// Slice LA1 — LANDSCAPE AUTHORING (heightmap brush sculpting + splat-layer painting + spline-carved
// roads), hf::terrain::author. Pure CPU, header-only, PURE INTEGER on every authored byte — NO device,
// NO backend symbols, NO new RHI, NO new shader, NO float on any state-mutating path.
//
// THE GAP THIS CLOSES: the engine's terrain is PROCEDURAL-ONLY (heightmap.h float field + procterrain.h
// integer fBm + terrain_stream.h LOD tiles) — there is no way to AUTHOR a landscape. UE5's Landscape
// ships brush sculpting, paint layers and spline roads; LA1 builds the deterministic authoring core:
// every edit is a PURE INTEGER OP over an AuthoredTerrain (Q16.16 heights + uint8x4 splat weights), so
// the authored landscape is bit-identical on every platform/compiler BY CONSTRUCTION and every op is
// recorded/reversible (the ED5 capture-before/apply/record discipline, terrain-local history below).
//
// COORDINATES (pinned): TEXEL SPACE. Texel (x, z) sits at the Q16.16 point (x<<16, z<<16); brush
// centers/radii and road spline control points are Q16.16 texel coordinates (the spline's y channel is
// the road's Q16.16 HEIGHT). This aligns 1:1 with the heightQ grid (row-major index z*w + x) and with
// procterrain.h's SampleHeight convention (grid index i <-> coordinate i*kOne).
//
// THE BRUSH KERNEL (pinned): BrushFalloff(d, r) has a FLAT CORE — falloff == kOne for d <= r/2 (r/2 by
// arithmetic shift), then the integer smoothstep ramp s = t^2(3-2t) of t = fxdiv(r-d, r-r/2) across the
// outer half, 0 outside r. The flat core is what makes "flatten hits the target EXACTLY within the
// brush" a real, testable contract (a pure smoothstep kernel is exact only at the infinitesimal
// center). Distance d = FxISqrt(dx^2+dz^2) (int64 squares are exact Q32.32; FxISqrt is the fpx integer
// sqrt — floor, deterministic). NO float, NO <cmath>.
//
// THE SPLAT RENORM (pinned, largest-remainder): weights are uint8x4 summing to EXACTLY 255 per texel
// (the standard splat convention). SetLayerWeight(s, layer, w') pins the target layer then rescales the
// OTHER three proportionally: wi' = floor(wi * rem / oldOthers) with the shortfall (<= 2) distributed
// by LARGEST REMAINDER ri = wi*rem % oldOthers, ties broken by LOWEST layer index. oldOthers == 0
// (the target layer owned all 255) frees rem into the BASE layer (layer 0; layer 1 if the target IS 0).
// Every path re-sums to exactly 255 — asserted globally by the test.
//
// THE ROAD CARVE (pinned, the SP1 composition): CarveRoad samples the spline by ARC DISTANCE
// (EvalByDistance every kOne/2 — half a texel) into a polyline, then for every texel within
// width/2 + shoulder of the polyline (point-to-segment distance, int64 squared-distance compare,
// EARLIEST-segment tie-break — the NAV integer discipline) FLATTENS: d <= width/2 -> height = the
// nearest polyline point's y EXACTLY (the road bed) + the ROAD splat layer set to 255; d in
// (width/2, width/2+shoulder] -> height = roadY + smoothstep(fxdiv(d - width/2, shoulder)) *
// (beforeH - roadY) (the shoulder blend, monotone in d over constant terrain, continuous with the bed
// at d = width/2 and with the untouched terrain at the shoulder edge because smoothstep(kOne) == kOne
// EXACTLY). All blends read a BEFORE-SNAPSHOT of the affected rect (order-independent). HONESTY: the
// bed takes the NEAREST polyline point's height, so a road climbing a slope quantizes per texel — a
// staircase profile along the run direction (the pinned real profile; a longitudinal-smoothing pass is
// future fidelity work). One CarveRoad == ONE history command.
//
// HISTORY — THE TERRAIN-LOCAL CHOICE (documented per the enrollment notes): ED5's EditCommand is a
// FIXED-SIZE flat POD ("NO std::function / heap indirection, serializable by construction",
// edit_history.h) addressed through EditTargets' entity/node models. A brush memento is inherently
// VARIABLE-SIZE (the affected rect's before/after pixels — a full-map flatten memento is the whole
// map), so enrolling would plant the first heap-owning payload inside EditCommand and break the
// flat-POD serialize contract, and terrain texels fit neither the view-index nor the NodeId addressing
// model. LA1 therefore keeps edit_history.h byte-UNTOUCHED and ships a TERRAIN-LOCAL history that
// follows the SAME recipe (capture BEFORE -> raw op -> capture AFTER -> Record; cursor; branch-kill
// truncate on Record; LIFO Undo applies BEFORE, Redo applies AFTER; bitwise no-ops record nothing).
// Undo/redo restore the rect's pixels VERBATIM -> bit-exact by construction (the memento proof).
// MEMORY BOUND (honest): a command stores before+after height (4+4 B) and splat (4+4 B) per affected
// texel = 16 B/texel; a whole-map op on 128x128 is ~256 KiB, on 1024x1024 ~16 MiB. Brushes are
// rect-bounded ((2r+2)^2 texels); only a map-sized brush pays the map-sized memento.
//
// MESH/REBUILD SEAM (v1, documented): the authored heightfield feeds the EXISTING PT4 render bridge —
// BuildAuthoredTerrainMesh forwards heightQ to procterrain.h's BuildIntTerrainMesh (square terrains;
// the same one-float-crossing render-only path the --pt4 shot uses). v1 is a WHOLE-TILE rebuild keyed
// off the `version` counter (bumped by every applied op, undo and redo); per-rect incremental remesh is
// future work. The authored DATA stays pure integer; only the mesh bridge is float (render-only).
//
// REUSE MAP: sim/fpx.h fx/kOne/kFrac/fxmul/fxdiv/FxISqrt (read-only Q16.16 toolbox);
// terrain/procterrain.h IntHeight (the fBm seed constructor) + BuildIntTerrainMesh (the mesh seam,
// read-only); spline/spline.h Spline/ArcTable/ArcTotal/EvalByDistance (SP1, read-only — the road
// composition); net/session.h DigestBytes (the digest currency). heightmap.h / procterrain.h /
// terrain_stream.h / spline.h / edit_history.h are all byte-UNTOUCHED.

#include <cstdint>
#include <vector>

#include "net/session.h"          // hf::net::DigestBytes — the pinned-digest FNV-1a-64 currency
#include "spline/spline.h"        // SP1: Spline/ArcTable/EvalByDistance (the road carve composes this)
#include "terrain/procterrain.h"  // IntHeight (the fBm seed) + BuildIntTerrainMesh (the mesh seam)

namespace hf::terrain::author {

using hf::sim::fpx::fx;
using hf::sim::fpx::kOne;
using hf::sim::fpx::kFrac;
using hf::sim::fpx::fxmul;
using hf::sim::fpx::fxdiv;
using hf::sim::fpx::FxISqrt;
using hf::sim::fpx::FxVec3;

// The four splat layers (pinned meaning; the showcase palette): 0 = grass (the base layer every texel
// starts fully weighted to), 1 = rock, 2 = snow, 3 = ROAD (the layer CarveRoad paints).
inline constexpr int kSplatLayers = 4;
inline constexpr int kRoadLayer   = 3;

// --- The authored terrain ---------------------------------------------------------------------------

// W x H Q16.16 heights (row-major, index z*w + x) + per-texel uint8x4 splat weights summing to EXACTLY
// 255 (index (z*w + x)*4 + layer) + a version counter (bumped by every applied op / undo / redo — the
// whole-tile mesh-rebuild dirty seam). Plain data; every operation below is a free function.
struct AuthoredTerrain {
    int32_t              w = 0, h = 0;
    std::vector<fx>      heightQ;   // w*h
    std::vector<uint8_t> splat;     // w*h*4, per-texel sum == 255
    uint32_t             version = 0;
};

// Seed constructor 1: FLAT at heightQ0, all splat weight on the base (grass) layer.
inline AuthoredTerrain MakeFlatTerrain(int32_t w, int32_t h, fx heightQ0) {
    AuthoredTerrain t;
    if (w <= 0 || h <= 0) return t;
    t.w = w; t.h = h;
    t.heightQ.assign((size_t)w * (size_t)h, heightQ0);
    t.splat.assign((size_t)w * (size_t)h * kSplatLayers, 0);
    for (size_t i = 0; i < (size_t)w * (size_t)h; ++i) t.splat[i * kSplatLayers] = 255;
    return t;
}

// Seed constructor 2: from the EXISTING procterrain fBm — height(x,z) = fxmul(scaleQ, IntHeight(x,z))
// sampled at the texel coordinates x = gx*worldSizeQ/w, z = gz*worldSizeQ/h (int64 floor division, the
// GenHeightField mapping generalized to W x H). Splat starts all-grass. Pure integer; pinned digest.
inline AuthoredTerrain MakeProcTerrain(int32_t w, int32_t h, fx worldSizeQ, int octaves,
                                       uint32_t seed, fx scaleQ) {
    AuthoredTerrain t = MakeFlatTerrain(w, h, 0);
    if (t.w == 0) return t;
    for (int32_t gz = 0; gz < h; ++gz) {
        const fx z = (fx)(((int64_t)gz * (int64_t)worldSizeQ) / (int64_t)h);
        for (int32_t gx = 0; gx < w; ++gx) {
            const fx x = (fx)(((int64_t)gx * (int64_t)worldSizeQ) / (int64_t)w);
            t.heightQ[(size_t)gz * (size_t)w + (size_t)gx] =
                fxmul(scaleQ, IntHeight(x, z, octaves, seed));
        }
    }
    return t;
}

// The terrain's deterministic fingerprint: FNV-1a-64 over hand-LE bytes of {w, h, every height, every
// splat byte} (field-serialized — layout/padding-free, so the pin is cross-platform/compiler). The
// version counter is EXCLUDED on purpose: undo restores the authored PIXELS bit-exactly, not the
// monotonic rebuild counter (the digest is the state identity the memento proof compares).
inline uint64_t DigestTerrain(const AuthoredTerrain& t) {
    std::vector<uint8_t> b;
    b.reserve(8 + t.heightQ.size() * 4 + t.splat.size());
    auto putU32 = [&b](uint32_t v) {
        b.push_back((uint8_t)v); b.push_back((uint8_t)(v >> 8));
        b.push_back((uint8_t)(v >> 16)); b.push_back((uint8_t)(v >> 24));
    };
    putU32((uint32_t)t.w);
    putU32((uint32_t)t.h);
    for (fx v : t.heightQ) putU32((uint32_t)v);
    b.insert(b.end(), t.splat.begin(), t.splat.end());
    return hf::net::DigestBytes(b.data(), b.size());
}

// --- The brush kernel (pinned) ------------------------------------------------------------------------

// The integer smoothstep s = t^2(3-2t) in Q16.16 (the procterrain fade shape). SmoothQ(0) == 0 and
// SmoothQ(kOne) == kOne EXACTLY (fxmul(kOne,kOne) == kOne) — the exactness the flatten/shoulder
// endpoint contracts stand on.
inline fx SmoothQ(fx t) { return fxmul(fxmul(t, t), 3 * kOne - 2 * t); }

// BrushFalloff(d, r): kOne for d <= r/2 (the FLAT CORE, r/2 = r>>1), the smoothstep ramp of
// t = fxdiv(r-d, r - r/2) across the outer half, 0 outside r (or for a degenerate radius). Monotone
// non-increasing in d. Pure integer.
inline fx BrushFalloff(fx d, fx r) {
    if (r <= 0 || d > r) return 0;
    const fx core = r >> 1;
    if (d <= core) return kOne;
    return SmoothQ(fxdiv(r - d, r - core));
}

// The affected texel rect [x0,x1) x [z0,z1) (clamped to the terrain; empty when x1<=x0 or z1<=z0).
// This is the MEMENTO BOUND: recorded commands snapshot exactly this rect.
struct TexRect {
    int32_t x0 = 0, z0 = 0, x1 = 0, z1 = 0;
    bool Empty() const { return x1 <= x0 || z1 <= z0; }
};

// BrushRect: the texels a brush at (cx, cz) with radius r can touch — the circle's bounding square
// (floor(c-r) .. floor(c+r)+1), clamped. Pure function of the parameters (the Recorded wrappers call
// it BEFORE the op to capture the before-pixels).
inline TexRect BrushRect(const AuthoredTerrain& t, fx cx, fx cz, fx r) {
    TexRect rc;
    if (t.w <= 0 || t.h <= 0 || r <= 0) return rc;
    int32_t x0 = (int32_t)((cx - r) >> kFrac);          // arithmetic shift == floor
    int32_t z0 = (int32_t)((cz - r) >> kFrac);
    int32_t x1 = (int32_t)((cx + r) >> kFrac) + 1;
    int32_t z1 = (int32_t)((cz + r) >> kFrac) + 1;
    if (x0 < 0) x0 = 0; if (z0 < 0) z0 = 0;
    if (x1 > t.w) x1 = t.w; if (z1 > t.h) z1 = t.h;
    rc.x0 = x0; rc.z0 = z0; rc.x1 = x1; rc.z1 = z1;
    return rc;
}

// Texel-to-brush distance (Q16.16): FxISqrt over the exact int64 squared distance (floor sqrt).
inline fx TexelDist(int32_t x, int32_t z, fx cx, fx cz) {
    const int64_t dx = (int64_t)(((fx)x) << kFrac) - (int64_t)cx;
    const int64_t dz = (int64_t)(((fx)z) << kFrac) - (int64_t)cz;
    return (fx)FxISqrt(dx * dx + dz * dz);
}

// --- Sculpt ops (pure; the Recorded wrappers below enroll them in the history) -------------------------

// RAISE: heightQ += fxmul(strengthQ, falloff). strengthQ may be negative (a lower brush). Bumps version
// (even when the rect is empty the op COUNTS as applied for the seam; a no-change op records nothing).
inline void ApplyRaiseBrush(AuthoredTerrain& t, fx cx, fx cz, fx radiusQ, fx strengthQ) {
    const TexRect rc = BrushRect(t, cx, cz, radiusQ);
    for (int32_t z = rc.z0; z < rc.z1; ++z)
        for (int32_t x = rc.x0; x < rc.x1; ++x) {
            const fx fall = BrushFalloff(TexelDist(x, z, cx, cz), radiusQ);
            if (fall == 0) continue;
            t.heightQ[(size_t)z * (size_t)t.w + (size_t)x] += fxmul(strengthQ, fall);
        }
    ++t.version;
}

// FLATTEN: heightQ += fxmul(falloff, target - heightQ) — EXACTLY targetQ inside the flat core
// (fall == kOne -> h + (target - h) == target, fxmul(kOne, v) == v exactly), a smoothstep blend across
// the outer half, untouched outside the radius.
inline void ApplyFlattenBrush(AuthoredTerrain& t, fx cx, fx cz, fx radiusQ, fx targetQ) {
    const TexRect rc = BrushRect(t, cx, cz, radiusQ);
    for (int32_t z = rc.z0; z < rc.z1; ++z)
        for (int32_t x = rc.x0; x < rc.x1; ++x) {
            const fx fall = BrushFalloff(TexelDist(x, z, cx, cz), radiusQ);
            if (fall == 0) continue;
            fx& hh = t.heightQ[(size_t)z * (size_t)t.w + (size_t)x];
            hh += fxmul(fall, targetQ - hh);
        }
    ++t.version;
}

// SMOOTH: one falloff-blended 3x3 integer box-blur pass — newH = h + fxmul(falloff, avg9 - h) with
// avg9 = the truncating int64 /9 of the 3x3 edge-clamped neighborhood, ALL NEIGHBOR READS FROM A
// BEFORE-SNAPSHOT of the rect (+1 ring) so the result is evaluation-order independent (deterministic
// Jacobi-style pass; repeated application converges toward the local mean).
inline void ApplySmoothBrush(AuthoredTerrain& t, fx cx, fx cz, fx radiusQ) {
    const TexRect rc = BrushRect(t, cx, cz, radiusQ);
    if (!rc.Empty()) {
        // Snapshot the rect + 1-texel ring (edge-clamped reads stay inside the snapshot).
        int32_t sx0 = rc.x0 - 1, sz0 = rc.z0 - 1, sx1 = rc.x1 + 1, sz1 = rc.z1 + 1;
        if (sx0 < 0) sx0 = 0; if (sz0 < 0) sz0 = 0;
        if (sx1 > t.w) sx1 = t.w; if (sz1 > t.h) sz1 = t.h;
        const int32_t sw = sx1 - sx0, sh = sz1 - sz0;
        std::vector<fx> snap((size_t)sw * (size_t)sh);
        for (int32_t z = sz0; z < sz1; ++z)
            for (int32_t x = sx0; x < sx1; ++x)
                snap[(size_t)(z - sz0) * (size_t)sw + (size_t)(x - sx0)] =
                    t.heightQ[(size_t)z * (size_t)t.w + (size_t)x];
        auto snapAt = [&](int32_t x, int32_t z) -> fx {   // edge-clamped snapshot read
            if (x < sx0) x = sx0; else if (x >= sx1) x = sx1 - 1;
            if (z < sz0) z = sz0; else if (z >= sz1) z = sz1 - 1;
            return snap[(size_t)(z - sz0) * (size_t)sw + (size_t)(x - sx0)];
        };
        for (int32_t z = rc.z0; z < rc.z1; ++z)
            for (int32_t x = rc.x0; x < rc.x1; ++x) {
                const fx fall = BrushFalloff(TexelDist(x, z, cx, cz), radiusQ);
                if (fall == 0) continue;
                int64_t sum = 0;
                for (int32_t dz = -1; dz <= 1; ++dz)
                    for (int32_t dx = -1; dx <= 1; ++dx) sum += (int64_t)snapAt(x + dx, z + dz);
                const fx avg = (fx)(sum / 9);   // truncating division (pinned; identical MSVC/clang)
                fx& hh = t.heightQ[(size_t)z * (size_t)t.w + (size_t)x];
                hh += fxmul(fall, avg - hh);
            }
    }
    ++t.version;
}

// --- Paint op (splat weights; the pinned largest-remainder renorm) -------------------------------------

// SetLayerWeight: pin s[layer] = newW (clamped 0..255), rescale the OTHER three layers proportionally
// to sum exactly 255 - newW: wi' = floor(wi*rem/oldOthers), shortfall (<= 2) by LARGEST REMAINDER
// (ri = wi*rem % oldOthers), ties -> lowest layer index. oldOthers == 0 frees rem into the base layer
// (0; or 1 when the target IS layer 0). Postcondition: sum == 255 EXACTLY, every path.
inline void SetLayerWeight(uint8_t* s, int layer, int newW) {
    if (newW < 0) newW = 0; else if (newW > 255) newW = 255;
    const int oldOthers = 255 - (int)s[layer];
    const int rem = 255 - newW;
    int oth[3]; int k = 0;
    for (int i = 0; i < kSplatLayers; ++i)
        if (i != layer) oth[k++] = i;
    s[layer] = (uint8_t)newW;
    if (oldOthers == 0) {
        for (int i = 0; i < 3; ++i) s[oth[i]] = 0;
        s[layer == 0 ? 1 : 0] = (uint8_t)rem;   // the freed weight goes to the base layer
        return;
    }
    int given = 0, remainder[3];
    for (int i = 0; i < 3; ++i) {
        const int wi = (int)s[oth[i]];
        const int q = (wi * rem) / oldOthers;
        remainder[i] = (wi * rem) % oldOthers;
        s[oth[i]] = (uint8_t)q;
        given += q;
    }
    int short_ = rem - given;                    // 0..2 by construction
    while (short_ > 0) {                         // largest remainder, ties -> lowest index
        int best = 0;
        for (int i = 1; i < 3; ++i)
            if (remainder[i] > remainder[best]) best = i;
        s[oth[best]] = (uint8_t)(s[oth[best]] + 1);
        remainder[best] = -1;
        --short_;
    }
}

// PAINT: raise the target layer's weight by round-free floor((fxmul(strengthQ, falloff) * 255) >> 16)
// weight units (strengthQ = kOne paints the flat core to FULL in one stamp), renormalized per texel by
// SetLayerWeight. Heights untouched. layer outside [0,3] is a safe no-op (version still bumps).
inline void ApplyPaintBrush(AuthoredTerrain& t, int layer, fx cx, fx cz, fx radiusQ, fx strengthQ) {
    if (layer >= 0 && layer < kSplatLayers) {
        const TexRect rc = BrushRect(t, cx, cz, radiusQ);
        for (int32_t z = rc.z0; z < rc.z1; ++z)
            for (int32_t x = rc.x0; x < rc.x1; ++x) {
                const fx fall = BrushFalloff(TexelDist(x, z, cx, cz), radiusQ);
                if (fall == 0) continue;
                const int add = (int)(((int64_t)fxmul(strengthQ, fall) * 255) >> kFrac);
                if (add <= 0) continue;
                uint8_t* s = &t.splat[((size_t)z * (size_t)t.w + (size_t)x) * kSplatLayers];
                SetLayerWeight(s, layer, (int)s[layer] + add);
            }
    }
    ++t.version;
}

// --- The spline road (the SP1 composition) -------------------------------------------------------------

// The polyline the carve rasterizes against: the spline sampled by ARC DISTANCE every kOne/2 (half a
// texel — the EvalByDistance quantization; includes the exact endpoint). Pure integer.
inline std::vector<FxVec3> BuildRoadPolyline(const spline::Spline& sp, const spline::ArcTable& tab) {
    std::vector<FxVec3> poly;
    const fx total = spline::ArcTotal(tab);
    if (total <= 0) return poly;
    const fx step = kOne / 2;
    const int n = (int)(total / step) + 1;
    poly.reserve((size_t)n + 1);
    for (int i = 0; i < n; ++i)
        poly.push_back(spline::EvalByDistance(sp, tab, (fx)((int64_t)i * step)));
    poly.push_back(spline::EvalByDistance(sp, tab, total));   // the exact end
    return poly;
}

// CarveRect: the polyline's XZ bounds expanded by width/2 + shoulder + one texel, clamped — the road
// command's memento bound.
inline TexRect CarveRect(const AuthoredTerrain& t, const std::vector<FxVec3>& poly, fx widthQ,
                         fx shoulderQ) {
    TexRect rc;
    if (t.w <= 0 || t.h <= 0 || poly.empty() || widthQ <= 0) return rc;
    fx minX = poly[0].x, maxX = poly[0].x, minZ = poly[0].z, maxZ = poly[0].z;
    for (const FxVec3& p : poly) {
        if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
        if (p.z < minZ) minZ = p.z; if (p.z > maxZ) maxZ = p.z;
    }
    const fx pad = (widthQ >> 1) + (shoulderQ > 0 ? shoulderQ : 0) + kOne;
    int32_t x0 = (int32_t)((minX - pad) >> kFrac), z0 = (int32_t)((minZ - pad) >> kFrac);
    int32_t x1 = (int32_t)((maxX + pad) >> kFrac) + 1, z1 = (int32_t)((maxZ + pad) >> kFrac) + 1;
    if (x0 < 0) x0 = 0; if (z0 < 0) z0 = 0;
    if (x1 > t.w) x1 = t.w; if (z1 > t.h) z1 = t.h;
    rc.x0 = x0; rc.z0 = z0; rc.x1 = x1; rc.z1 = z1;
    return rc;
}

// Nearest point on the polyline to texel (px, pz) in XZ: min squared distance over segments (int64
// Q32.32 compare, STRICTLY-LESS so the EARLIEST segment wins ties — pinned), the projection parameter
// t = clamp(dot/int64-len2, 0, kOne) by deterministic floor division. Returns {distQ, roadYQ} — the
// Q16.16 distance (floor sqrt) and the y of the nearest polyline point (lerped along its segment).
struct RoadNearest {
    fx dist = 0;
    fx y    = 0;
};
inline RoadNearest NearestOnPolyline(const std::vector<FxVec3>& poly, fx px, fx pz) {
    RoadNearest out;
    int64_t bestSq = -1;
    for (size_t i = 0; i + 1 < poly.size(); ++i) {
        const FxVec3& A = poly[i];
        const FxVec3& B = poly[i + 1];
        const int64_t abx = (int64_t)B.x - A.x, abz = (int64_t)B.z - A.z;
        const int64_t apx = (int64_t)px - A.x, apz = (int64_t)pz - A.z;
        const int64_t den = abx * abx + abz * abz;                  // Q32.32
        fx tq = 0;
        if (den > 0) {
            int64_t num = abx * apx + abz * apz;                    // Q32.32
            if (num < 0) num = 0;
            int64_t tt = (num * kOne) / den;                        // Q16.16, floor
            if (tt > kOne) tt = kOne;
            tq = (fx)tt;
        }
        const fx cxq = A.x + fxmul((fx)(B.x - A.x), tq);
        const fx czq = A.z + fxmul((fx)(B.z - A.z), tq);
        const int64_t dx = (int64_t)px - cxq, dz = (int64_t)pz - czq;
        const int64_t dsq = dx * dx + dz * dz;
        if (bestSq < 0 || dsq < bestSq) {
            bestSq = dsq;
            out.y = A.y + fxmul((fx)(B.y - A.y), tq);
        }
    }
    out.dist = (fx)FxISqrt(bestSq < 0 ? 0 : bestSq);
    return out;
}

// CARVE (the polyline core; ApplyCarveRoad below is the spline-facing wrapper): for every texel of the
// carve rect with d = NearestOnPolyline.dist:
//   d <= width/2                    -> heightQ = roadY EXACTLY; splat = ROAD layer 255 (SetLayerWeight);
//                                      counts toward the returned roadTexels.
//   width/2 < d <= width/2+shoulder -> heightQ = roadY + fxmul(SmoothQ(fxdiv(d - width/2, shoulder)),
//                                      beforeH - roadY) — the monotone shoulder blend (beforeH from a
//                                      BEFORE-SNAPSHOT of the rect; smoothstep endpoints exact so the
//                                      band is continuous with both the bed and the untouched terrain).
//   d beyond the band               -> bit-untouched.
// Returns the flattened bed texel count (the stat line's roadTexels).
inline uint32_t ApplyCarveRoadPoly(AuthoredTerrain& t, const std::vector<FxVec3>& poly, fx widthQ,
                                   fx shoulderQ) {
    uint32_t roadTexels = 0;
    const TexRect rc = CarveRect(t, poly, widthQ, shoulderQ);
    if (!rc.Empty() && poly.size() >= 2) {
        const fx halfW = widthQ >> 1;
        const fx sh = shoulderQ > 0 ? shoulderQ : 0;
        // BEFORE-snapshot of the rect heights (the shoulder blend reads pre-carve terrain).
        const int32_t rw = rc.x1 - rc.x0;
        std::vector<fx> before((size_t)rw * (size_t)(rc.z1 - rc.z0));
        for (int32_t z = rc.z0; z < rc.z1; ++z)
            for (int32_t x = rc.x0; x < rc.x1; ++x)
                before[(size_t)(z - rc.z0) * (size_t)rw + (size_t)(x - rc.x0)] =
                    t.heightQ[(size_t)z * (size_t)t.w + (size_t)x];
        for (int32_t z = rc.z0; z < rc.z1; ++z)
            for (int32_t x = rc.x0; x < rc.x1; ++x) {
                const RoadNearest nr = NearestOnPolyline(poly, ((fx)x) << kFrac, ((fx)z) << kFrac);
                if (nr.dist > halfW + sh) continue;
                fx& hh = t.heightQ[(size_t)z * (size_t)t.w + (size_t)x];
                if (nr.dist <= halfW) {
                    hh = nr.y;                                            // the exactly-flat road bed
                    uint8_t* s = &t.splat[((size_t)z * (size_t)t.w + (size_t)x) * kSplatLayers];
                    SetLayerWeight(s, kRoadLayer, 255);                   // the road paint
                    ++roadTexels;
                } else if (sh > 0) {
                    const fx bh = before[(size_t)(z - rc.z0) * (size_t)rw + (size_t)(x - rc.x0)];
                    hh = nr.y + fxmul(SmoothQ(fxdiv(nr.dist - halfW, sh)), bh - nr.y);
                }
            }
    }
    ++t.version;
    return roadTexels;
}

// The spline-facing carve (ONE op == one history command via RecordedCarveRoad below).
inline uint32_t ApplyCarveRoad(AuthoredTerrain& t, const spline::Spline& sp,
                               const spline::ArcTable& tab, fx widthQ, fx shoulderQ) {
    return ApplyCarveRoadPoly(t, BuildRoadPolyline(sp, tab), widthQ, shoulderQ);
}

// --- The terrain-local history (the ED5 recipe over rect mementos) -------------------------------------

// One recorded command: the affected rect + its before/after height and splat pixels (row-major within
// the rect). Undo copies BEFORE back verbatim, redo copies AFTER — bit-exact by construction. Memory:
// 16 bytes per affected texel (see the header memory-bound note).
struct TerrainCmd {
    TexRect              rect;
    std::vector<fx>      hBefore, hAfter;   // rect texel heights
    std::vector<uint8_t> sBefore, sAfter;   // rect texel splats (x4)
};

// commands[0..cursor) are APPLIED; [cursor..size) is the redo tail (the ED5 shape).
struct TerrainHistory {
    std::vector<TerrainCmd> commands;
    std::size_t             cursor = 0;
};

namespace detail {
inline void CaptureRect(const AuthoredTerrain& t, const TexRect& rc, std::vector<fx>& hOut,
                        std::vector<uint8_t>& sOut) {
    hOut.clear(); sOut.clear();
    if (rc.Empty()) return;
    hOut.reserve((size_t)(rc.x1 - rc.x0) * (size_t)(rc.z1 - rc.z0));
    sOut.reserve(hOut.capacity() * kSplatLayers);
    for (int32_t z = rc.z0; z < rc.z1; ++z)
        for (int32_t x = rc.x0; x < rc.x1; ++x) {
            const size_t i = (size_t)z * (size_t)t.w + (size_t)x;
            hOut.push_back(t.heightQ[i]);
            for (int l = 0; l < kSplatLayers; ++l) sOut.push_back(t.splat[i * kSplatLayers + l]);
        }
}
inline void RestoreRect(AuthoredTerrain& t, const TexRect& rc, const std::vector<fx>& hIn,
                        const std::vector<uint8_t>& sIn) {
    size_t k = 0;
    for (int32_t z = rc.z0; z < rc.z1; ++z)
        for (int32_t x = rc.x0; x < rc.x1; ++x, ++k) {
            const size_t i = (size_t)z * (size_t)t.w + (size_t)x;
            t.heightQ[i] = hIn[k];
            for (int l = 0; l < kSplatLayers; ++l) t.splat[i * kSplatLayers + l] = sIn[k * kSplatLayers + l];
        }
    ++t.version;   // the mesh-rebuild seam sees undo/redo too
}
// Record with the ED5 branch-kill semantics; a bitwise no-op (before == after) records nothing.
inline void RecordCmd(TerrainHistory& hist, TerrainCmd&& c) {
    if (c.hBefore == c.hAfter && c.sBefore == c.sAfter) return;
    hist.commands.resize(hist.cursor);
    hist.commands.push_back(static_cast<TerrainCmd&&>(c));
    hist.cursor = hist.commands.size();
}
}  // namespace detail

// Undo the most recent applied command (restores its BEFORE rect verbatim). False when empty.
inline bool Undo(TerrainHistory& hist, AuthoredTerrain& t) {
    if (hist.cursor == 0) return false;
    const TerrainCmd& c = hist.commands[hist.cursor - 1];
    detail::RestoreRect(t, c.rect, c.hBefore, c.sBefore);
    --hist.cursor;
    return true;
}

// Redo the next undone command (re-applies its AFTER rect verbatim). False when no redo tail.
inline bool Redo(TerrainHistory& hist, AuthoredTerrain& t) {
    if (hist.cursor >= hist.commands.size()) return false;
    const TerrainCmd& c = hist.commands[hist.cursor];
    detail::RestoreRect(t, c.rect, c.hAfter, c.sAfter);
    ++hist.cursor;
    return true;
}

// --- Recorded wrappers (capture BEFORE -> raw op -> capture AFTER -> Record; the ED5 discipline) --------

inline void RecordedRaiseBrush(TerrainHistory& hist, AuthoredTerrain& t, fx cx, fx cz, fx radiusQ,
                               fx strengthQ) {
    TerrainCmd c;
    c.rect = BrushRect(t, cx, cz, radiusQ);
    detail::CaptureRect(t, c.rect, c.hBefore, c.sBefore);
    ApplyRaiseBrush(t, cx, cz, radiusQ, strengthQ);
    detail::CaptureRect(t, c.rect, c.hAfter, c.sAfter);
    detail::RecordCmd(hist, static_cast<TerrainCmd&&>(c));
}
inline void RecordedFlattenBrush(TerrainHistory& hist, AuthoredTerrain& t, fx cx, fx cz, fx radiusQ,
                                 fx targetQ) {
    TerrainCmd c;
    c.rect = BrushRect(t, cx, cz, radiusQ);
    detail::CaptureRect(t, c.rect, c.hBefore, c.sBefore);
    ApplyFlattenBrush(t, cx, cz, radiusQ, targetQ);
    detail::CaptureRect(t, c.rect, c.hAfter, c.sAfter);
    detail::RecordCmd(hist, static_cast<TerrainCmd&&>(c));
}
inline void RecordedSmoothBrush(TerrainHistory& hist, AuthoredTerrain& t, fx cx, fx cz, fx radiusQ) {
    TerrainCmd c;
    c.rect = BrushRect(t, cx, cz, radiusQ);
    detail::CaptureRect(t, c.rect, c.hBefore, c.sBefore);
    ApplySmoothBrush(t, cx, cz, radiusQ);
    detail::CaptureRect(t, c.rect, c.hAfter, c.sAfter);
    detail::RecordCmd(hist, static_cast<TerrainCmd&&>(c));
}
inline void RecordedPaintBrush(TerrainHistory& hist, AuthoredTerrain& t, int layer, fx cx, fx cz,
                               fx radiusQ, fx strengthQ) {
    TerrainCmd c;
    c.rect = BrushRect(t, cx, cz, radiusQ);
    detail::CaptureRect(t, c.rect, c.hBefore, c.sBefore);
    ApplyPaintBrush(t, layer, cx, cz, radiusQ, strengthQ);
    detail::CaptureRect(t, c.rect, c.hAfter, c.sAfter);
    detail::RecordCmd(hist, static_cast<TerrainCmd&&>(c));
}
// ONE spline carve == ONE history command (the polyline is built once; the rect memento covers the
// whole band). Returns the flattened bed texel count.
inline uint32_t RecordedCarveRoad(TerrainHistory& hist, AuthoredTerrain& t, const spline::Spline& sp,
                                  const spline::ArcTable& tab, fx widthQ, fx shoulderQ) {
    const std::vector<FxVec3> poly = BuildRoadPolyline(sp, tab);
    TerrainCmd c;
    c.rect = CarveRect(t, poly, widthQ, shoulderQ);
    detail::CaptureRect(t, c.rect, c.hBefore, c.sBefore);
    const uint32_t roadTexels = ApplyCarveRoadPoly(t, poly, widthQ, shoulderQ);
    detail::CaptureRect(t, c.rect, c.hAfter, c.sAfter);
    detail::RecordCmd(hist, static_cast<TerrainCmd&&>(c));
    return roadTexels;
}

// --- The mesh/rebuild seam (v1: whole-tile; the PT4 render bridge, float render-only) ------------------

// Forward the authored heightfield to the EXISTING BuildIntTerrainMesh (procterrain.h — the same
// one-float-crossing bridge the --pt4 shot renders through). SQUARE terrains only in v1 (w == h — the
// bridge's n x n contract); non-square returns an empty mesh. Whole-tile rebuild: callers re-mesh when
// `version` changed (the documented v1 seam; per-rect incremental remesh is future work).
inline TerrainMesh BuildAuthoredTerrainMesh(const AuthoredTerrain& t, float worldSize,
                                            float heightScale) {
    if (t.w != t.h || t.w < 2) return TerrainMesh{};
    return BuildIntTerrainMesh(t.heightQ, t.w, worldSize, heightScale);
}

// ===================== LA1 showcase — the shared scenario + raster (WV1 pattern) =======================
// Header-local so BOTH showcase backends (Vulkan --la1-landscape-shot / Metal --la1-landscape) run the
// IDENTICAL bytes with ZERO copy drift: a 128x128 fBm-seeded terrain, a sculpted hill (two raise
// stamps), a flattened plateau, a smoothed valley, rock/snow paint, and the SP1 spline road carved
// through it all — every op RECORDED, with the undo-all/redo-all digest round-trip proven live.

inline constexpr int32_t  kShowSize     = 128;
inline constexpr uint32_t kShowSeed     = 1234u;
inline constexpr int      kShowOctaves  = 4;
inline constexpr fx       kShowScale    = 10 * kOne;   // fBm heights span ~[0, 10) wu
inline constexpr fx       kShowWorld    = 32 * kOne;   // fBm window (32 wu over 128 texels -> rolling)

// The FIXED road spline (texel-space XZ, Q16.16; y = the road's Q16.16 height). Roughly-even spacing
// (the SP1 uniform-CR authoring discipline). Keep FIXED forever — pinned digests + the golden hash it.
inline spline::Spline MakeShowcaseRoadSpline() {
    spline::Spline s;
    s.closed = false;
    s.points = {
        FxVec3{  6 * kOne, 3 * kOne,  14 * kOne},
        FxVec3{ 34 * kOne, 4 * kOne,  30 * kOne},
        FxVec3{ 58 * kOne, 5 * kOne,  62 * kOne},
        FxVec3{ 84 * kOne, 5 * kOne,  86 * kOne},
        FxVec3{112 * kOne, 6 * kOne, 108 * kOne},
        FxVec3{122 * kOne, 6 * kOne, 122 * kOne},
    };
    return s;
}

struct LandscapeShotRun {
    AuthoredTerrain       terrain;
    TerrainHistory        hist;
    spline::Spline        road;
    uint32_t              roadTexels = 0;
    uint32_t              ops        = 0;
    uint64_t              seedDigest  = 0;   // the fBm-seeded pre-edit terrain
    uint64_t              digest      = 0;   // the fully-authored terrain (the stat line's digest)
    bool                  undoRedoOk  = false;   // undo-all -> seedDigest, redo-all -> digest (bit-exact)
    bool                  splatSum255 = true;    // every texel's 4 weights sum to exactly 255
    bool                  flattenExact = false;  // the plateau's flat-core texels hit the target exactly
};

// RunLandscapeShotScenario: the pure function both backends call. Every edit goes through the Recorded
// wrappers; the undo/redo round-trip proof runs INSIDE the scenario (both backends assert it).
inline LandscapeShotRun RunLandscapeShotScenario() {
    LandscapeShotRun run;
    run.terrain = MakeProcTerrain(kShowSize, kShowSize, kShowWorld, kShowOctaves, kShowSeed,
                                  kShowScale);
    run.seedDigest = DigestTerrain(run.terrain);

    // (1)+(2) THE HILL: two recorded raise stamps (a broad base + a sharper cap).
    RecordedRaiseBrush(run.hist, run.terrain, 34 * kOne, 88 * kOne, 20 * kOne, 9 * kOne);
    RecordedRaiseBrush(run.hist, run.terrain, 38 * kOne, 84 * kOne, 10 * kOne, 6 * kOne);
    // (3) THE PLATEAU: a recorded flatten to exactly 6 wu.
    const fx plateauTarget = 6 * kOne;
    RecordedFlattenBrush(run.hist, run.terrain, 94 * kOne, 34 * kOne, 16 * kOne, plateauTarget);
    // (4) THE SMOOTHED VALLEY: one recorded smooth pass mid-map.
    RecordedSmoothBrush(run.hist, run.terrain, 64 * kOne, 56 * kOne, 14 * kOne);
    // (5)+(6) PAINT: rock over the hill flank, snow on the cap (strength kOne = full in the flat core).
    RecordedPaintBrush(run.hist, run.terrain, 1, 34 * kOne, 88 * kOne, 22 * kOne, kOne);
    RecordedPaintBrush(run.hist, run.terrain, 2, 38 * kOne, 84 * kOne, 8 * kOne, kOne);
    // (7) sand-free: paint the plateau lightly with rock too (a second splat region for the viz).
    RecordedPaintBrush(run.hist, run.terrain, 1, 94 * kOne, 34 * kOne, 14 * kOne, kOne / 2);
    // (8) THE ROAD: the SP1 spline carved through everything (ONE recorded command).
    run.road = MakeShowcaseRoadSpline();
    const spline::ArcTable tab = spline::BuildArcTable(run.road);
    run.roadTexels = RecordedCarveRoad(run.hist, run.terrain, run.road, tab, 6 * kOne, 5 * kOne);
    run.ops = (uint32_t)run.hist.commands.size();
    run.digest = DigestTerrain(run.terrain);

    // PROOF (live, both backends): the plateau flat core hit the target exactly... the road may have
    // recarved part of it, so probe a flat-core texel OUTSIDE the road band (94,28: 6 texels from the
    // plateau center — inside radius/2=8; the road polyline passes ~40+ texels away).
    run.flattenExact = true;
    {
        const int32_t probes[][2] = {{94, 28}, {90, 34}, {98, 36}};
        for (const auto& p : probes)
            if (run.terrain.heightQ[(size_t)p[1] * kShowSize + (size_t)p[0]] != plateauTarget)
                run.flattenExact = false;
    }
    // PROOF (live): every texel's splat weights sum to exactly 255.
    for (size_t i = 0; i < (size_t)kShowSize * kShowSize && run.splatSum255; ++i) {
        int sum = 0;
        for (int l = 0; l < kSplatLayers; ++l) sum += run.terrain.splat[i * kSplatLayers + l];
        if (sum != 255) run.splatSum255 = false;
    }
    // PROOF (live): undo-all returns the SEED digest bit-exact; redo-all returns the authored digest.
    AuthoredTerrain scratch = run.terrain;
    TerrainHistory  hist2   = run.hist;
    while (Undo(hist2, scratch)) {}
    const bool undoOk = (DigestTerrain(scratch) == run.seedDigest);
    while (Redo(hist2, scratch)) {}
    const bool redoOk = (DigestTerrain(scratch) == run.digest);
    run.undoRedoOk = undoOk && redoOk;
    return run;
}

// RenderLandscapeShot: the PURE-INTEGER top-down raster both backends call — strict-zero cross-backend
// BY CONSTRUCTION. 128x128 texels at 4 px/texel -> 512x512 BGRA8. Per texel: the 4 splat layer colors
// weight-blended (sum(w_i*c_i)/255 — exact integer), SHADED RELIEF from the NW height gradient
// (brightness 256 + clamp((h - h_NW) >> 9, -120, 120), the emboss shade), CONTOUR LINES every 1 wu
// (darken where floor(h/kOne) differs from the +x or +z neighbor), road spline control points as
// orange squares.
inline void RenderLandscapeShot(const LandscapeShotRun& run, std::vector<uint8_t>& bgra,
                                uint32_t& outW, uint32_t& outH) {
    const int S = 4, W = kShowSize * S, H = kShowSize * S;
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 255);
    // Layer palette (RGB): grass, rock, snow, road asphalt.
    static const int kLayerR[4] = { 66, 128, 236, 70};
    static const int kLayerG[4] = {112, 122, 240, 72};
    static const int kLayerB[4] = { 52, 116, 244, 76};
    const AuthoredTerrain& t = run.terrain;
    auto hAt = [&](int32_t x, int32_t z) -> fx {
        if (x < 0) x = 0; else if (x >= t.w) x = t.w - 1;
        if (z < 0) z = 0; else if (z >= t.h) z = t.h - 1;
        return t.heightQ[(size_t)z * (size_t)t.w + (size_t)x];
    };
    for (int32_t z = 0; z < t.h; ++z)
        for (int32_t x = 0; x < t.w; ++x) {
            const size_t i = (size_t)z * (size_t)t.w + (size_t)x;
            const uint8_t* s = &t.splat[i * kSplatLayers];
            int r = 0, g = 0, b = 0;
            for (int l = 0; l < kSplatLayers; ++l) {
                r += (int)s[l] * kLayerR[l]; g += (int)s[l] * kLayerG[l]; b += (int)s[l] * kLayerB[l];
            }
            r /= 255; g /= 255; b /= 255;
            // Emboss shade from the NW gradient (light from the north-west).
            int dh = (int)((hAt(x, z) - hAt(x - 1, z - 1)) >> 9);
            if (dh > 120) dh = 120; else if (dh < -120) dh = -120;
            const int bright = 256 + dh;
            r = (r * bright) >> 8; g = (g * bright) >> 8; b = (b * bright) >> 8;
            // Contour lines every 1 wu (skip on the road bed — keep the carve reading flat).
            const fx hq = t.heightQ[i];
            const bool contour = ((hq >> kFrac) != (hAt(x + 1, z) >> kFrac)) ||
                                 ((hq >> kFrac) != (hAt(x, z + 1) >> kFrac));
            if (contour && s[kRoadLayer] < 128) { r = (r * 3) >> 2; g = (g * 3) >> 2; b = (b * 3) >> 2; }
            if (r > 255) r = 255; if (g > 255) g = 255; if (b > 255) b = 255;
            if (r < 0) r = 0; if (g < 0) g = 0; if (b < 0) b = 0;
            for (int pz = 0; pz < S; ++pz)
                for (int px = 0; px < S; ++px) {
                    // y-flip: texel z=0 at the bottom (the top-down map convention).
                    const size_t p = ((size_t)(H - 1 - (z * S + pz)) * (size_t)W + (size_t)(x * S + px)) * 4;
                    bgra[p + 0] = (uint8_t)b; bgra[p + 1] = (uint8_t)g;
                    bgra[p + 2] = (uint8_t)r; bgra[p + 3] = 255;
                }
        }
    // Road control points: orange squares (over the map).
    for (const FxVec3& p : run.road.points) {
        const int cx = (int)((int64_t)p.x * S >> kFrac);
        const int cy = H - 1 - (int)((int64_t)p.z * S >> kFrac);
        for (int dy = -3; dy <= 3; ++dy)
            for (int dx = -3; dx <= 3; ++dx) {
                const int ix = cx + dx, iy = cy + dy;
                if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;
                const size_t q = ((size_t)iy * (size_t)W + (size_t)ix) * 4;
                bgra[q + 0] = 58; bgra[q + 1] = 128; bgra[q + 2] = 232; bgra[q + 3] = 255;
            }
    }
}

}  // namespace hf::terrain::author
