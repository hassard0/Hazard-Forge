#pragma once
// Slice PCG1 — Seeded hash-PRNG primitive (the BEACHHEAD of FLAGSHIP #22: DETERMINISTIC PCG). The irreducible
// primitive every later PCG slice (scatter / mask / transform / placement) consumes: a DETERMINISTIC SEEDED
// INTEGER HASH-PRNG in Q16.16, pure int32 (MSL-native — NO int64 fxmul, NO trig, NO float, NO clock, NO
// <cmath>). Header-only, namespace hf::pcg. NO device, NO backend symbols, NO new RHI, NO shader.
//
// The determinism story is IDENTICAL to the particle emitter's: the hash is the SAME integer avalanche shape
// as engine/sim/particles.h::ParticleHash (a fixed uint32 wrapping xorshift/multiply mixing — defined +
// identical on every vendor/compiler) and the no-trig direction comes from the SAME engine/sim/particles.h::
// EmitDir host-Q16.16 direction table. We do NOT improvise a new hash; reusing the proven one keeps the
// cross-platform bit-exactness story unchanged. fpx.h (fx / kOne / kFrac / fxmul / FxVec3) is reused read-only.
//
// THE CROSS-BACKEND CRUX: every value PcgHash/PcgRand01/PcgRandRange/PcgUnitDir returns is a pure function of
// (seed, index[, salt]) computed with ONLY uint32 wrapping arithmetic + arithmetic shifts/masks + (for ranges)
// the int64-intermediate fxmul — deterministic + identical on every compiler/vendor. PcgRand01 takes the TOP
// 16 bits of the 32-bit hash as the Q16.16 fraction (a pure shift — NO division, NO float), so it lands in
// [0, kOne) strictly (top16 of a uint32 is in [0, 65535] = [0, kOne)). This is what makes a CPU-rendered
// PCG point-field byte-identical CPU<->Vulkan<->Metal BY CONSTRUCTION (a strict zero-differing-pixel golden).

#include <algorithm>
#include <cstdint>
#include <vector>

#include "sim/fpx.h"            // Q16.16 toolbox (read-only): fx / kOne / kFrac / fxmul / FxVec3
#include "sim/particles.h"     // ParticleHash (the avalanche shape) + EmitDir (the no-trig direction table), read-only

namespace hf::pcg {

// Pull the Q16.16 primitives from fpx (the SAME ones particles.h re-exports) so callers can `using` them.
using hf::sim::fpx::fx;
using hf::sim::fpx::kOne;
using hf::sim::fpx::kFrac;
using hf::sim::fpx::fxmul;
using hf::sim::fpx::FxVec3;

// ----- PcgHash: the deterministic seeded integer avalanche over (seed, index) ------------------------------
// The SAME integer hash SHAPE as engine/sim/particles.h::ParticleHash (verbatim ops) — a fixed uint32 wrapping
// xorshift/multiply mixing. Pure uint32 arithmetic (defined + identical on every vendor/compiler) — NO RNG, NO
// clock, NO float. `seed` plays the emitterId role, `index` the spawn-index role: distinct (seed,index) pairs
// hash to distinct, replay-stable streams.
inline uint32_t PcgHash(uint32_t seed, uint32_t index) {
    uint32_t h = seed * 2654435761u;               // Knuth multiplicative
    h ^= (index + 0x9E3779B9u + (h << 6) + (h >> 2));
    h += index * 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return h;
}

// ----- PcgRand01: a Q16.16 value in [0, kOne) from the TOP 16 bits of the hash ------------------------------
// A pure shift (NO division, NO float): the top 16 bits of the 32-bit hash become the Q16.16 fraction. The
// top16 of a uint32 is in [0, 65535] == [0, kOne), so the result is STRICTLY < kOne (a valid [0,1) sample).
inline fx PcgRand01(uint32_t seed, uint32_t index) {
    return (fx)(PcgHash(seed, index) >> 16);       // in [0, kOne)
}

// ----- PcgRandRange: lo + PcgRand01 * (hi - lo) in Q16.16 ---------------------------------------------------
// lo + fxmul(PcgRand01(...), hi - lo). Since PcgRand01 in [0, kOne) and fxmul is the int64-intermediate
// ((int64)a*b >> kFrac), the result lies in [lo, hi) (so within [lo, hi]) for hi >= lo.
inline fx PcgRandRange(uint32_t seed, uint32_t index, fx lo, fx hi) {
    return lo + fxmul(PcgRand01(seed, index), hi - lo);
}

// ----- PcgUnitDir: a deterministic ~unit direction by indexing the EmitDir table with the hash --------------
// Reuse engine/sim/particles.h::EmitDir VERBATIM (the fixed 13-entry host-Q16.16 direction table, NO trig, NO
// sqrt). The hash selects the table entry (EmitDir does the `% kEmitDirCount`), so the direction is a pure
// function of (seed, index) — deterministic + cross-vendor identical.
inline FxVec3 PcgUnitDir(uint32_t seed, uint32_t index) {
    return hf::sim::particles::EmitDir(PcgHash(seed, index));
}

// ----- PcgStream: a {seed, salt} stream so distinct PCG layers diverge --------------------------------------
// The emitterId-salt pattern from particles: a stream carries a `seed` AND a `salt`; the overloads fold `salt`
// into the hash INPUT so two streams with the same seed but different salt produce DIFFERENT sequences for the
// same index (the future scatter/mask/transform stages each get their own salt -> independent streams). The
// fold is `seed ^ salt` (then PcgHash mixes it thoroughly) — pure uint32, deterministic.
struct PcgStream {
    uint32_t seed = 0;
    uint32_t salt = 0;
};

inline uint32_t PcgHash(const PcgStream& s, uint32_t index) {
    return PcgHash(s.seed ^ s.salt, index);
}
inline fx PcgRand01(const PcgStream& s, uint32_t index) {
    return (fx)(PcgHash(s, index) >> 16);          // in [0, kOne)
}
inline fx PcgRandRange(const PcgStream& s, uint32_t index, fx lo, fx hi) {
    return lo + fxmul(PcgRand01(s, index), hi - lo);
}
inline FxVec3 PcgUnitDir(const PcgStream& s, uint32_t index) {
    return hf::sim::particles::EmitDir(PcgHash(s, index));
}

// ===== Slice PCG2 — Jittered-grid point scatter (the 2nd slice of FLAGSHIP #22) =============================
// The first real generation primitive built on PCG1's hash-PRNG: deterministic JITTERED-GRID SCATTER — one
// point per grid cell, each offset within its own cell by a seeded PcgRand01 jitter, so the lattice is broken
// WITHOUT any float/trig (the declarative replacement for the engine's hand-coded for(gx)for(gz) instance
// grids). Pure int32 — the only multiply is fxmul (int64-intermediate, CPU-side). NO float, NO trig.

// ----- PcgArea: the XZ scatter region in Q16.16 world units (a flat patch, Y = min.y) ----------------------
// The area is a flat ground patch: scatter happens in the XZ plane, every point is emitted at height Y =
// min.y. Keep it a plain struct (no behaviour) — ScatterGrid does all the work.
struct PcgArea {
    FxVec3 min;
    FxVec3 max;
};

// ----- ScatterGrid: one jittered point per cell over the area's XZ extent -----------------------------------
// Partition the area's XZ extent into cellsX x cellsZ EQUAL cells (cellW/cellD = integer extents
// (max.x-min.x)/cellsX, (max.z-min.z)/cellsZ). For each cell (cx,cz) in FIXED ascending order (cz OUTER, cx
// INNER — the order is PINNED so the output vector is deterministic), emit ONE point at the cell's min corner
// plus a per-axis in-cell jitter fxmul(PcgRand01(stream, idx*2+0), cellW) in X and
// fxmul(PcgRand01(stream, idx*2+1), cellD) in Z (idx = cz*cellsX+cx). Because PcgRand01 in [0,kOne) and the
// jitter scales it by the cell extent, every point stays STRICTLY inside its own cell (cellMin <= p <
// cellMin + cellExtent). Returns empty for cellsX<=0 || cellsZ<=0 or a degenerate area (the no-op control).
// Int32 only (the one mul is fxmul). idx*2+0 / idx*2+1 draw INDEPENDENT PcgRand01 samples (distinct indices).
inline std::vector<FxVec3> ScatterGrid(const PcgStream& stream, const PcgArea& area, int cellsX, int cellsZ) {
    std::vector<FxVec3> out;
    if (cellsX <= 0 || cellsZ <= 0) return out;            // no-op control
    const fx extX = area.max.x - area.min.x;
    const fx extZ = area.max.z - area.min.z;
    if (extX <= 0 || extZ <= 0) return out;                // degenerate area -> no-op control
    const fx cellW = extX / cellsX;                        // integer cell extents (positive divisor + numerator)
    const fx cellD = extZ / cellsZ;
    if (cellW <= 0 || cellD <= 0) return out;              // cells collapsed to zero width -> no-op control
    out.reserve((size_t)cellsX * (size_t)cellsZ);
    for (int cz = 0; cz < cellsZ; ++cz) {                  // cz OUTER (pinned order)
        for (int cx = 0; cx < cellsX; ++cx) {              // cx INNER (pinned order)
            const uint32_t idx = (uint32_t)(cz * cellsX + cx);
            const fx cellMinX = area.min.x + cellW * cx;   // the cell's min corner (X)
            const fx cellMinZ = area.min.z + cellD * cz;   // the cell's min corner (Z)
            const fx jitterX = fxmul(PcgRand01(stream, idx * 2u + 0u), cellW);  // in [0, cellW)
            const fx jitterZ = fxmul(PcgRand01(stream, idx * 2u + 1u), cellD);  // in [0, cellD)
            out.push_back(FxVec3{cellMinX + jitterX, area.min.y, cellMinZ + jitterZ});
        }
    }
    return out;
}

// ===== Slice PCG3 — Analytic density mask + importance rejection (the 3rd slice of FLAGSHIP #22) ============
// The control layer over PCG2's ScatterGrid: an analytic DENSITY MASK (a scalar field in Q16.16) the scatter
// follows. A candidate point survives in proportion to mask*density at its position (importance rejection), so
// density VARIES across the area (a falloff disc, a half-plane) instead of a uniform grid. Builds on PCG2's
// ScatterGrid (candidate positions IDENTICAL to PCG2) and adds the FIRST int64 path of the PCG arc — the radial
// mask's FxLength (the int64 integer sqrt). But PCG is a CPU-side host generator (NO GPU shader), so it runs
// CPU-on-both-backends -> still byte-identical cross-backend (a strict zero-differing-pixel golden), the same
// integer discipline as the rest of the flagship. Only the radial Eval touches int64 (FxLength); the rest int32.

// ----- PcgMaskType / PcgMask: an analytic Q16.16 weight field over the area -----------------------------------
// Eval(p) returns a Q16.16 weight clamped to [0, kOne]:
//   * Uniform   -> always kOne (the no-op control — every point passes).
//   * Radial    -> clamp(kOne - fxdiv(dist, radius), 0, kOne), dist = FxLength of (p - center) with Y zeroed
//     (the XZ distance; FxLength is the int64 path, reused from fpx.h VERBATIM). 1 at the centre, falling
//     linearly to 0 at radius, 0 beyond. radius <= 0 -> 0 everywhere (a degenerate/zero mask).
//   * HalfPlane -> kOne on the inward side of the plane through center with normal axis, 0 on the other side
//     (a hard 0/kOne step via the sign of the integer dot product (p-center)·axis). Secondary mask — kept simple.
enum class PcgMaskType { Uniform, Radial, HalfPlane };

struct PcgMask {
    PcgMaskType type   = PcgMaskType::Uniform;
    FxVec3      center;      // Radial: disc centre (XZ); HalfPlane: a point on the plane
    fx          radius = 0;  // Radial: falloff radius
    FxVec3      axis;        // HalfPlane: the inward normal (XZ)

    fx Eval(const FxVec3& p) const {
        switch (type) {
            case PcgMaskType::Uniform:
                return kOne;                                   // every point passes (the no-op control)
            case PcgMaskType::Radial: {
                if (radius <= 0) return 0;                     // degenerate/zero mask -> 0 everywhere
                // XZ distance (Y zeroed): the int64 FxLength path (reused from fpx.h verbatim).
                const FxVec3 d{ p.x - center.x, 0, p.z - center.z };
                const fx dist = hf::sim::fpx::FxLength(d);     // int64 integer sqrt, Q16.16
                const fx w = kOne - hf::sim::fpx::fxdiv(dist, radius);
                if (w <= 0)    return 0;                       // at/beyond the rim
                if (w >= kOne) return kOne;                    // at/inside the centre (dist==0)
                return w;                                      // linear falloff in (0, kOne)
            }
            case PcgMaskType::HalfPlane: {
                // Sign of the integer dot product (p - center)·axis (int64 intermediate, no overflow). >= 0 is
                // the inward side -> kOne; the other side -> 0. A hard 0/kOne step (kept simple).
                const int64_t dot = (int64_t)(p.x - center.x) * (int64_t)axis.x +
                                    (int64_t)(p.y - center.y) * (int64_t)axis.y +
                                    (int64_t)(p.z - center.z) * (int64_t)axis.z;
                return dot >= 0 ? kOne : 0;
            }
        }
        return kOne;                                           // unreachable; defensive no-op
    }
};

// ----- ScatterMasked: PCG2's ScatterGrid filtered by importance rejection against the mask -------------------
// Generate the PCG2 grid (call ScatterGrid internally so the candidate positions are IDENTICAL to PCG2), then
// KEEP each candidate (in the same fixed cell order) iff fxmul(mask.Eval(p), density) > PcgRand01(keepStream,
// idx), where keepStream is `stream` with a DISTINCT salt (stream.salt ^ 0xA11C0DEu) so the keep-decision draw
// is independent of the jitter draw, and idx is the linear cell index. Returns the surviving subset (variable
// count, FIXED order). density = kOne + a Uniform mask keeps ALL points (fxmul(kOne,kOne)=kOne > PcgRand01 in
// [0,kOne) ALWAYS -> the no-op == ScatterGrid); density = 0 keeps NONE. Only the radial Eval touches int64
// (FxLength); the rest is int32. The mask only REMOVES candidates — it never moves them.
inline std::vector<FxVec3> ScatterMasked(const PcgStream& stream, const PcgArea& area, int cellsX, int cellsZ,
                                         const PcgMask& mask, fx density) {
    std::vector<FxVec3> out;
    const std::vector<FxVec3> cand = ScatterGrid(stream, area, cellsX, cellsZ);  // PCG2 candidates (IDENTICAL)
    if (cand.empty()) return out;                              // no-op control (empty grid -> empty subset)
    const PcgStream keepStream{ stream.seed, stream.salt ^ 0xA11C0DEu };  // independent keep-decision stream
    out.reserve(cand.size());
    for (uint32_t idx = 0; idx < (uint32_t)cand.size(); ++idx) {
        const FxVec3& p = cand[idx];
        const fx weight = fxmul(mask.Eval(p), density);        // mask*density in Q16.16
        if (weight > PcgRand01(keepStream, idx)) out.push_back(p);  // importance rejection (same fixed order)
    }
    return out;
}

// ===== Slice PCG4 — Per-instance transform rules (the 4th slice of FLAGSHIP #22) ===========================
// The "rules" layer: turn the scattered POINTS (PCG2/PCG3 output) into full INSTANCES — each gets a seed-stable
// random yaw + a random uniform scale within designer ranges. This is what makes a procedural field look natural
// (rocks at varied angles and sizes) instead of identical clones. Strict int32: the yaw comes from a BAKED
// integer quaternion table (NO runtime transcendentals, NO <cmath>, NO QFromAxisAngleSnapped — that small-angle
// Taylor series diverges for the full 0..2pi range). The pattern mirrors particles.h::EmitDir: a host-constant
// Q16.16 quaternion table, bit-identical cross-vendor BY CONSTRUCTION (stored integer literals). Pure CPU,
// header-only, APPEND-ONLY (PCG1/PCG2/PCG3 + fpx.h + particles.h read-only).

// ----- kPcgYaw16: 16 baked yaw quaternions about +Y (a full turn in 22.5-degree steps) ---------------------
// A yaw quaternion about +Y for theta = k*22.5deg is {x=0, y=sin(theta/2), z=0, w=cos(theta/2)}. These are
// PRE-SNAPPED host-constant Q16.16 literals (each y^2+w^2 ~= kOne^2 to ~0.0002% — unit to tolerance). Index with
// the hash (& 15 selects an entry). kPcgYaw16[0] is identity {0,0,0,kOne}. NO runtime trig — stored integers.
inline const hf::sim::fpx::FxQuat kPcgYaw16[16] = {
    {0,     0, 0, 65536},  {0, 12785, 0, 64277},  {0, 25080, 0, 60547},  {0, 36410, 0, 54491},
    {0, 46341, 0, 46341},  {0, 54491, 0, 36410},  {0, 60547, 0, 25080},  {0, 64277, 0, 12785},
    {0, 65536, 0,     0},  {0, 64277, 0,-12785},  {0, 60547, 0,-25080},  {0, 54491, 0,-36410},
    {0, 46341, 0,-46341},  {0, 36410, 0,-54491},  {0, 25080, 0,-60547},  {0, 12785, 0,-64277},
};

// ----- PcgInstance: a placed instance = a scattered point annotated with a seed-stable orient + scale --------
struct PcgInstance {
    FxVec3                pos;     // the scattered point (UNCHANGED — the transform only annotates, never moves)
    hf::sim::fpx::FxQuat  orient;  // a baked yaw quaternion (or identity when randomYaw == false)
    fx                    scale;   // a Q16.16 uniform scale in [scaleLo, scaleHi] (== scaleLo when scaleLo == scaleHi)
};

// ----- PcgTransform: the designer's per-instance rule (random yaw on/off + a uniform scale range) ------------
// randomYaw == false -> identity orientation; scaleLo == scaleHi -> a fixed scale. Together (randomYaw=false,
// scaleLo=scaleHi=kOne) they are the NO-OP control: every instance is identity orient + unit scale at its point.
struct PcgTransform {
    bool randomYaw = true;
    fx   scaleLo   = kOne;
    fx   scaleHi   = kOne;
};

// ----- BuildInstances: annotate each scattered point with a yaw + scale drawn from the stream -----------------
// For each point i (FIXED order): orient = rule.randomYaw ? kPcgYaw16[PcgHash(yawStream,i)&15] : identity;
// scale = PcgRandRange(scaleStream, i, scaleLo, scaleHi). yawStream/scaleStream are `stream` with DISTINCT salts
// (stream.salt ^ 0x4A09 for yaw, stream.salt ^ 0x5CA1E for scale) so the yaw and scale draws are independent of
// each other AND of the PCG2 jitter / PCG3 keep draws. Emit PcgInstance{point, orient, scale}. Pure int32 (the
// table is integer literals; PcgRandRange's only mul is fxmul). NO float, NO trig. The transform only ANNOTATES
// the points (pos is copied verbatim) — it never moves them, so the no-op rule returns the input points exactly.
inline std::vector<PcgInstance> BuildInstances(const std::vector<FxVec3>& points, const PcgStream& stream,
                                               const PcgTransform& rule) {
    std::vector<PcgInstance> out;
    out.reserve(points.size());
    const PcgStream yawStream{   stream.seed, stream.salt ^ 0x4A09u };    // independent yaw-draw stream
    const PcgStream scaleStream{ stream.seed, stream.salt ^ 0x5CA1Eu };   // independent scale-draw stream
    for (uint32_t i = 0; i < (uint32_t)points.size(); ++i) {
        const hf::sim::fpx::FxQuat orient = rule.randomYaw ? kPcgYaw16[PcgHash(yawStream, i) & 15u]
                                                           : hf::sim::fpx::FxQuat{0, 0, 0, kOne};
        const fx scale = PcgRandRange(scaleStream, i, rule.scaleLo, rule.scaleHi);
        out.push_back(PcgInstance{ points[(size_t)i], orient, scale });
    }
    return out;
}

// ===== Slice PCG5 — Declarative layered pipeline + overlap-prune (the 5th slice of FLAGSHIP #22) ============
// THE DETERMINISM HEADLINE. Composes PCG2/PCG3/PCG4 into one declarative `Generate(graph, stream)` call (the
// integer-deterministic analog of a PCG graph), and adds the capstone PruneOverlaps stage: reject instances
// whose footprint spheres interpenetrate, processed in a FIXED CANONICAL ORDER (first-placed-wins, the
// fpx::SolveContacts Gauss-Seidel order-determinism discipline) so a designer gets a Poisson-like
// minimum-spacing guarantee deterministically. The headline: `Generate` is a PURE FUNCTION of the seed, so two
// netcode "peers" produce a byte-identical world from the seed alone. Strict int32 except the prune's FxLength
// (int64, CPU-both -> still byte-identical). APPEND-ONLY (PCG1-PCG4 + fpx.h + particles.h read-only).

// ----- PcgGraph: the declarative recipe (scatter -> mask -> transform -> prune) ------------------------------
struct PcgGraph {
    PcgArea      area;
    int          cellsX = 1, cellsZ = 1;     // ScatterStage
    bool         useMask = false;            // MaskStage (optional)
    PcgMask      mask;
    fx           density = kOne;
    PcgTransform transform;                  // TransformStage
    bool         prune = false;              // PruneStage (optional)
    fx           pruneRadius = 0;            // footprint sphere radius (scaled per-instance by instance.scale)
};

// ----- PruneOverlaps: greedy minimum-spacing prune in a CANONICAL order independent of input order -----------
// (1) Establish a CANONICAL order INDEPENDENT of input order (the load-bearing bit): build an index array
//     0..n-1 and STABLE-sort it by a position-derived key — primary pos.z, tie-break pos.x, final tie-break the
//     original index (pure integer compares, NO float). Positions are unique per scatter cell so the key is
//     effectively total; the explicit origIndex tie-break makes the sort fully deterministic regardless. This
//     is what makes "shuffle the input -> SAME survivors" hold.
// (2) Greedily walk the canonical order; KEEP instance i iff it does NOT overlap any ALREADY-KEPT j: overlap
//     <=> FxLength((pos_i - pos_j) with Y zeroed) < r_i + r_j, where r_i = fxmul(baseRadius, inst_i.scale)
//     (the per-instance footprint; FxLength is the int64 path, reused from fpx.h). First-placed in canonical
//     order wins. Return the survivors IN canonical order (deterministic output order).
// (3) baseRadius <= 0 -> no pair can overlap -> all kept (a clean no-op), returned in canonical order.
inline std::vector<PcgInstance> PruneOverlaps(const std::vector<PcgInstance>& in, fx baseRadius) {
    std::vector<PcgInstance> out;
    const size_t n = in.size();
    if (n == 0) return out;

    // Canonical order: stable integer sort of indices by (pos.z, pos.x, origIndex).
    std::vector<uint32_t> order(n);
    for (uint32_t i = 0; i < (uint32_t)n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&in](uint32_t a, uint32_t b) {
        if (in[a].pos.z != in[b].pos.z) return in[a].pos.z < in[b].pos.z;  // primary: pos.z
        if (in[a].pos.x != in[b].pos.x) return in[a].pos.x < in[b].pos.x;  // tie-break: pos.x
        return a < b;                                                       // final tie-break: original index
    });

    out.reserve(n);
    if (baseRadius <= 0) {                       // no-op: no pair can overlap -> all kept, in canonical order
        for (size_t k = 0; k < n; ++k) out.push_back(in[order[k]]);
        return out;
    }

    // Greedy first-placed-wins walk over the canonical order; keep iff disjoint from every already-kept.
    std::vector<fx> keptR;                        // the per-instance footprint radius of each kept survivor
    keptR.reserve(n);
    for (size_t k = 0; k < n; ++k) {
        const PcgInstance& ci = in[order[k]];
        const fx ri = fxmul(baseRadius, ci.scale);
        bool overlaps = false;
        for (size_t j = 0; j < out.size() && !overlaps; ++j) {
            const PcgInstance& cj = out[j];
            const FxVec3 d{ ci.pos.x - cj.pos.x, 0, ci.pos.z - cj.pos.z };  // Y zeroed (XZ footprint)
            const fx dist = hf::sim::fpx::FxLength(d);                       // int64 integer sqrt, Q16.16
            if (dist < ri + keptR[j]) overlaps = true;                       // interpenetration
        }
        if (!overlaps) { out.push_back(ci); keptR.push_back(ri); }
    }
    return out;
}

// ----- Generate: compose the stages into one declarative pure-of-the-seed call -------------------------------
// (1) points = useMask ? ScatterMasked(...) : ScatterGrid(...);  (2) instances = BuildInstances(points,...);
// (3) if (prune) instances = PruneOverlaps(instances, pruneRadius). cellsX<=0||cellsZ<=0 -> empty (the empty
// no-op, since ScatterGrid/ScatterMasked already short-circuit). A PURE FUNCTION of (g, stream).
inline std::vector<PcgInstance> Generate(const PcgGraph& g, const PcgStream& stream) {
    const std::vector<FxVec3> points =
        g.useMask ? ScatterMasked(stream, g.area, g.cellsX, g.cellsZ, g.mask, g.density)
                  : ScatterGrid(stream, g.area, g.cellsX, g.cellsZ);
    std::vector<PcgInstance> instances = BuildInstances(points, stream, g.transform);
    if (g.prune) instances = PruneOverlaps(instances, g.pruneRadius);
    return instances;
}

}  // namespace hf::pcg
