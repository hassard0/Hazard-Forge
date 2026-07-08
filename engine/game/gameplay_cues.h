#pragma once
// Slice GC1 — DETERMINISTIC ABILITY TARGETING SHAPES + GAMEPLAY CUES (multi-target overlap queries + a
// cosmetic-cue EVENT stream, hf::game::cues). Next-tier parity gap #8 and the CAPSTONE of the gameplay-
// framework cluster (GAS1 abilities + GT1 tags + AL1 notify-bridge). GAS1's ability system targets a SINGLE
// EntityId and has NO area/shape targeting; GT1 adds tags; neither has a cue/event layer for cosmetic
// reactions. GC1 adds BOTH, deterministically, COMPOSED onto GAS1/GT1 through THIN ADAPTERS — ability.h,
// gameplay_tags.h and verdict.h are #included READ-ONLY / BYTE-UNTOUCHED:
//   (a) TARGETING SHAPES — sphere / box (AABB) / cone overlap over a small entity world -> the deterministic,
//       ascending-id SET of affected entities (optional GT1 tag FILTER — "friendly-fire off" via a team tag).
//   (b) GAMEPLAY CUES — a deterministic EVENT STREAM emitted when abilities activate + effects land / kill
//       (impact / buff / death), each cue carrying a GT1 cue tag (its identity), a source/target, a Q16.16
//       location and a magnitude, and the world tick. CollectCues(world, tick) yields a tick's cues in
//       deterministic order.
//
// EVERYTHING IS PURE INTEGER (Q16.16 positions; the overlap tests are exact integer distance²/AABB compares;
// the cone half-angle is a PINNED host-provided cos THRESHOLD compared against an integer normalized dot via
// FxISqrt + fxdiv — NO runtime transcendental, NO acos). Like GAS1/GT1 it is a pure function of the world +
// the command stream: a peer re-derives every resolved target SET and every cue EVENT bit-for-bit, and a
// rollback corrects a mispredicted AREA activation (a different set of entities hit).
//
// THE PINNED SEMANTICS (the determinism contract — all boundary conventions are FIXED and documented):
//   * ENTITY SPATIAL STATE: GC1 carries a GC1-LOCAL parallel array of {EntityId, Q16.16 pos, Q16.16 facing,
//     team} — GAS1's GasEntity has NO position, so GC1 owns the spatial world (parallel to gas.entities by id,
//     spawn order). Entities are STATIC in GC1 (no movement system — a future slice); the positions are the
//     targeting inputs. Snapshot completeness FOLLOWS from carrying this array (dropping it -> wrong target
//     sets -> divergence, proven).
//   * SPHERE overlap: an entity is IN a SphereTarget{center, radiusQ} iff FxDist2(pos,center) <= radiusQ²
//     (both int64 Q32.32 — EXACT, NO sqrt). INCLUSIVE convention: dist² == radius² is INSIDE. radius 0 targets
//     ONLY an entity exactly AT the center (dist²==0 <= 0).
//   * BOX overlap: an entity is IN a BoxTarget{center, halfExtents} (an AXIS-ALIGNED AABB — v1; an oriented
//     OBB is a future slice, documented) iff |pos.axis - center.axis| <= halfExtents.axis on EVERY axis.
//     INCLUSIVE convention: a point exactly on a face (==) is INSIDE.
//   * CONE overlap: an entity is IN a ConeTarget{apex, dir, rangeQ, cosHalfAngleQ} iff (1) within range
//     (FxDist2(pos,apex) <= rangeQ², inclusive) AND (2) dot(normalize(pos-apex), dir) >= cosHalfAngleQ. `dir`
//     is a PINNED UNIT direction (|dir| == kOne by construction of the caller's data); the normalize is
//     FxISqrt+fxdiv (integer, deterministic). cosHalfAngleQ is a PINNED host Q16.16 cosine threshold (NOT a
//     runtime acos). INCLUSIVE at exactly cosHalfAngleQ. AT-APEX DEGENERATE (pos==apex, length 0): INCLUDED
//     (the apex is trivially within any non-negative-range cone). A target directly BEHIND (dot < cos) is
//     excluded.
//   * TAG FILTER (optional, composes GT1): a shape may carry {filterTag, filterExclude}. filterExclude=true is
//     "friendly-fire OFF" (EXCLUDE entities whose EFFECTIVE tag set query-matches the team tag);
//     filterExclude=false keeps ONLY matching entities. The effective set is GT1's (owned UNION live-effect
//     grants), so the filter binds to the GT1 lifecycle for free.
//   * RESOLVED SET ORDER: ResolveTargets returns the hit ids in ASCENDING EntityId (ents[] is spawn-order
//     ascending by construction; sorted defensively). An empty set is deterministic.
//   * AREA APPLICATION (the adapter, ability.h byte-untouched): AreaActivate gates the CASTER (unknown ability
//     / unknown caster / GT1 blocked / GT1 missing-required), then COMMITS cost+cooldown VERBATIM through
//     gas::TryActivate with a ZERO-EFFECT filtered ability (so all of GAS1's cost/cooldown gate is reused, no
//     effects applied yet), then applies the ability's kTargetOther effects to EACH resolved target (targets
//     OUTER ascending, effects INNER array order) and its kTargetSelf effects ONCE to the caster — each
//     application GT1-immunity-gated (an immune receiver nullifies that effect, immuneCount++). A SINGLE-target
//     shape (kShapeSingle) reduces to the GAS1/GT1 single-target behaviour (the compat pin). Cost/cooldown are
//     paid ONCE regardless of hit count; an empty set still commits (cost paid, self effects applied).
//   * CUES: on a successful AreaActivate, an IMPACT cue is emitted at EACH hit target that took a kTargetOther
//     effect (magnitude = base-health LOST, >=0), a DEATH cue at any target whose base health crossed to 0, and
//     a SELF cue at the caster if a kTargetSelf effect landed. StepCues also emits a DEATH cue for any entity
//     whose base health reaches 0 DURING the gas step (a periodic-DoT kill). Cues append to cueLog in emission
//     order (targets ascending) -> a deterministic stream; CollectCues filters by tick.
//   * STEP: StepCues applies this tick's AREA commands (fixed array order) BEFORE gas::StepAbilities (the GAS1
//     commands-before-step contract). Activation cues are stamped with the pre-step world tick (== the
//     command's tick); step-death cues with the step's tick.
//
// HONEST SCOPE / CAVEATS: GC1 pins the cue EVENT STREAM — the deterministic contract a presentation layer
// consumes; it does NOT render VFX/audio (that is the downstream presentation layer, e.g. the VR1 particle /
// AL1 notify bridge — GC1 says clearly it is the EVENT layer, not the cosmetics). Box targeting is an
// axis-aligned AABB (OBB is future). The cone cosHalfAngle is a PINNED host threshold compared against an
// integer normalized dot — NOT a runtime acos; `dir` is assumed unit. Entities are STATIC (no movement).
// The cue CATALOG pinned here is activation + death; periodic/expiry cosmetic cues beyond death are a
// downstream extension. Targeting positions live in a GC1-LOCAL array (GAS1 has no position field).
//
// PURE CPU INTEGER (the strictest determinism tier). Header-only, namespace hf::game::cues. #includes
// game/gameplay_tags.h READ-ONLY (which pulls game/ability.h + game/verdict.h read-only — GasWorld / the GT1
// tag layer / EntityId / DigestFnv / the fpx Q16.16 toolbox + FxDist2). NO render RHI, NO new shader, NO new
// compute. No goldens committed by this slice.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "game/gameplay_tags.h"   // read-only: the GT1 tag layer (TagRegistry/TagContainer/TagRules/
                                  // EffectiveTags/HasTagQuery) + transitively the whole GAS1 core + verdict

namespace hf::game {
namespace cues {

// Reuse the composed layers' vocabulary verbatim (NO new primitives).
namespace gas     = hf::game::gas;
namespace tags    = hf::game::tags;
namespace fpx     = hf::sim::fpx;
namespace verdict = hf::game::verdict;
using gas::fx;
using gas::kOne;
using gas::kFrac;
using fpx::FxVec3;
using verdict::EntityId;
using verdict::kNoEntity;

// =================================================================================================
// 1. THE INTEGER TARGETING GEOMETRY (pure integer, NO sqrt-based normalize except FxISqrt/fxdiv; NO trig)
// =================================================================================================

// ----- SphereTarget / BoxTarget / ConeTarget: the three deterministic overlap shapes ------------------------
struct SphereTarget {
    FxVec3 center;         // Q16.16 world center
    fx     radiusQ = 0;    // Q16.16 radius (>= 0); 0 targets only an entity AT the center
};
struct BoxTarget {
    FxVec3 center;         // Q16.16 AABB center
    FxVec3 halfExtents;    // Q16.16 half-extents (axis-aligned v1; OBB is future)
};
struct ConeTarget {
    FxVec3 apex;           // Q16.16 cone apex
    FxVec3 dir;            // Q16.16 UNIT direction (|dir| == kOne by construction)
    fx     rangeQ        = 0;   // Q16.16 range (inclusive)
    fx     cosHalfAngleQ = 0;   // Q16.16 PINNED cos(half-angle) threshold (inclusive; NOT a runtime acos)
};

// InSphere: FxDist2 <= radius² (both int64 Q32.32) — EXACT, inclusive at the boundary, NO sqrt.
inline bool InSphere(const SphereTarget& s, const FxVec3& p) {
    const int64_t d2 = verdict::FxDist2(p, s.center);
    const int64_t r2 = (int64_t)s.radiusQ * (int64_t)s.radiusQ;
    return d2 <= r2;                                   // INCLUSIVE (dist²==radius² -> inside)
}

// InBox: axis-aligned AABB membership, inclusive on faces (|delta| <= half on every axis).
inline bool InBox(const BoxTarget& b, const FxVec3& p) {
    const FxVec3 d = fpx::FxSub(p, b.center);
    const fx ax = d.x < 0 ? (fx)(-d.x) : d.x;
    const fx ay = d.y < 0 ? (fx)(-d.y) : d.y;
    const fx az = d.z < 0 ? (fx)(-d.z) : d.z;
    return ax <= b.halfExtents.x && ay <= b.halfExtents.y && az <= b.halfExtents.z;  // INCLUSIVE
}

// InCone: within range (inclusive, exact int64) AND dot(normalize(p-apex), dir) >= cosHalfAngle (inclusive).
// The normalize is FxISqrt(len)+fxdiv (integer, deterministic); `dir` is assumed unit. At-apex (len 0) is
// INCLUDED (pinned). NO runtime transcendental.
inline bool InCone(const ConeTarget& c, const FxVec3& p) {
    const int64_t d2 = verdict::FxDist2(p, c.apex);
    const int64_t r2 = (int64_t)c.rangeQ * (int64_t)c.rangeQ;
    if (d2 > r2) return false;                          // outside range
    const FxVec3 tt = fpx::FxSub(p, c.apex);
    const fx len = fpx::FxLength(tt);
    if (len == 0) return true;                          // AT-APEX degenerate -> INCLUDED (pinned)
    const fx nx = fpx::fxdiv(tt.x, len);               // unit-ish components (Q16.16, deterministic rounding)
    const fx ny = fpx::fxdiv(tt.y, len);
    const fx nz = fpx::fxdiv(tt.z, len);
    const fx dotN = (fx)(fpx::fxmul(nx, c.dir.x) + fpx::fxmul(ny, c.dir.y) + fpx::fxmul(nz, c.dir.z));
    return dotN >= c.cosHalfAngleQ;                     // INCLUSIVE at exactly cosHalfAngle
}

// ----- TargetShape: a tagged union of the three shapes + a single-target compat form + optional tag filter ----
inline constexpr uint32_t kShapeSingle = 0u;   // one explicit target id (the GAS1/GT1 single-target compat form)
inline constexpr uint32_t kShapeSphere = 1u;
inline constexpr uint32_t kShapeBox    = 2u;
inline constexpr uint32_t kShapeCone   = 3u;

struct TargetShape {
    uint32_t     kind = kShapeSingle;
    EntityId     single = kNoEntity;   // for kShapeSingle
    SphereTarget sphere;
    BoxTarget    box;
    ConeTarget   cone;
    // Optional GT1 tag filter (composes the effective-tag set). kNoTag -> no filter.
    tags::TagId  filterTag     = tags::kNoTag;
    bool         filterExclude = false;   // true = friendly-fire OFF (exclude matching); false = only matching
};

// Convenience shape constructors (builder-ish; the tag filter is set separately).
inline TargetShape MakeSphereShape(const FxVec3& center, fx radiusQ) {
    TargetShape s; s.kind = kShapeSphere; s.sphere = SphereTarget{center, radiusQ}; return s;
}
inline TargetShape MakeBoxShape(const FxVec3& center, const FxVec3& halfExtents) {
    TargetShape s; s.kind = kShapeBox; s.box = BoxTarget{center, halfExtents}; return s;
}
inline TargetShape MakeConeShape(const FxVec3& apex, const FxVec3& dir, fx rangeQ, fx cosHalfAngleQ) {
    TargetShape s; s.kind = kShapeCone; s.cone = ConeTarget{apex, dir, rangeQ, cosHalfAngleQ}; return s;
}
inline TargetShape MakeSingleShape(EntityId target) {
    TargetShape s; s.kind = kShapeSingle; s.single = target; return s;
}

// =================================================================================================
// 2. THE SPATIAL ENTITY WORLD + THE CUE EVENT STREAM
// =================================================================================================

// ----- CueEntity: the GC1-local spatial record (parallel to gas.entities by id, spawn order) ----------------
struct CueEntity {
    EntityId id = kNoEntity;
    FxVec3   pos;              // Q16.16 world position (the targeting input)
    FxVec3   facing{kOne, 0, 0};  // Q16.16 unit facing (carried; cone casts pass their own dir)
    uint32_t team = 0;         // team id (data; team membership is expressed via an owned GT1 team tag)
};

// ----- CueEvent: one deterministic cosmetic-cue event (the presentation-layer contract) ---------------------
struct CueEvent {
    tags::TagId cueTag       = tags::kNoTag;   // the cue identity (an interned "Cue.*" tag)
    EntityId    sourceEntity = kNoEntity;      // the caster (kNoEntity for a source-less step death)
    EntityId    targetEntity = kNoEntity;      // the affected entity
    FxVec3      location;                      // Q16.16 world location (the target's position)
    fx          magnitude    = 0;              // Q16.16 (e.g. damage dealt); 0 if n/a
    uint32_t    tick         = 0;              // the world tick the cue was emitted at
};

// ----- CueRules / CueAbility: the STATIC authored cue table (per-ability impact/self cue + a global death cue)
struct CueAbility {
    uint32_t    abilityId = 0;
    tags::TagId impactCue = tags::kNoTag;   // emitted at each hit target (kNoTag = none)
    tags::TagId selfCue   = tags::kNoTag;   // emitted at the caster on a landed self-effect (kNoTag = none)
};
struct CueRules {
    std::vector<CueAbility> abilities;
    tags::TagId             deathCue = tags::kNoTag;   // emitted when a target's base health crosses to 0

    const CueAbility* Find(uint32_t abilityId) const {
        for (const CueAbility& a : abilities) if (a.abilityId == abilityId) return &a;
        return nullptr;
    }
};

// ----- CuesWorld: the composed GAS1+GT1 world + the GC1 spatial array + the accumulated cue log -------------
// A value COPY is a complete snapshot (tw is a copyable TaggedWorld; ents/cueLog are plain vectors). The
// registries/rules are STATIC authored assets (shared by every peer), NOT snapshot state.
struct CuesWorld {
    tags::TaggedWorld     tw;        // the GAS1 attributes/effects + GT1 owned tags (composed read-only)
    std::vector<CueEntity> ents;     // GC1-LOCAL spatial state, parallel to tw.gas.entities (same ids, order)
    std::vector<CueEvent>  cueLog;   // the accumulated cue event stream (append order = deterministic)
};

inline int FindEnt(const CuesWorld& w, EntityId id) {
    for (size_t i = 0; i < w.ents.size(); ++i) if (w.ents[i].id == id) return (int)i;
    return -1;
}
inline FxVec3 PosOf(const CuesWorld& w, EntityId id) {
    const int i = FindEnt(w, id);
    return i >= 0 ? w.ents[(size_t)i].pos : FxVec3{0, 0, 0};
}

// SpawnCueEntity: GAS1+GT1 spawn (attributes + owned tags) + a parallel GC1 spatial record (same id).
inline EntityId SpawnCueEntity(CuesWorld& w, const fx (&bases)[gas::kAttrCount],
                               const tags::TagContainer& owned, const FxVec3& pos, uint32_t team,
                               const FxVec3& facing = FxVec3{kOne, 0, 0}) {
    const EntityId id = tags::SpawnTagged(w.tw, bases, owned);
    w.ents.push_back(CueEntity{id, pos, facing, team});
    return id;
}

// EmitCue: append one cue to the log (the ONLY cue-log mutation site — a fixed field order).
inline void EmitCue(CuesWorld& w, tags::TagId cueTag, EntityId source, EntityId target, const FxVec3& loc,
                    fx magnitude, uint32_t tick) {
    w.cueLog.push_back(CueEvent{cueTag, source, target, loc, magnitude, tick});
}

// CollectCues: the tick's cues in deterministic order (a filtered view over the append-ordered log).
inline std::vector<CueEvent> CollectCues(const CuesWorld& w, uint32_t tick) {
    std::vector<CueEvent> out;
    for (const CueEvent& c : w.cueLog) if (c.tick == tick) out.push_back(c);
    return out;
}

// =================================================================================================
// 3. TARGET RESOLUTION (shape overlap + optional GT1 tag filter -> ascending-id set)
// =================================================================================================

// ResolveTargets: the entities inside `shape` (optionally tag-filtered), in ASCENDING EntityId order.
inline std::vector<EntityId> ResolveTargets(const CuesWorld& w, const tags::TagRules& tagRules,
                                            const tags::TagRegistry& reg, const TargetShape& shape) {
    std::vector<EntityId> out;
    if (shape.kind == kShapeSingle) {
        if (shape.single != kNoEntity && gas::FindEntity(w.tw.gas, shape.single) >= 0) out.push_back(shape.single);
        return out;   // (the compat form: exactly the explicit target, if live)
    }
    for (size_t i = 0; i < w.ents.size(); ++i) {
        const CueEntity& e = w.ents[i];
        bool inside = false;
        if      (shape.kind == kShapeSphere) inside = InSphere(shape.sphere, e.pos);
        else if (shape.kind == kShapeBox)    inside = InBox(shape.box, e.pos);
        else if (shape.kind == kShapeCone)   inside = InCone(shape.cone, e.pos);
        if (!inside) continue;
        if (shape.filterTag != tags::kNoTag) {
            const tags::TagContainer eff = tags::EffectiveTags(w.tw, tagRules, e.id);
            const bool has = tags::HasTagQuery(reg, eff, shape.filterTag);
            if (shape.filterExclude &&  has) continue;   // friendly-fire OFF: skip matching (team) entities
            if (!shape.filterExclude && !has) continue;   // only-matching: skip non-matching entities
        }
        out.push_back(e.id);
    }
    std::sort(out.begin(), out.end());   // ents[] is already ascending by construction; sort defensively
    return out;
}

// =================================================================================================
// 4. AREA ABILITY APPLICATION (the adapter — ability.h/gameplay_tags.h byte-untouched) + cues
// =================================================================================================

// AreaResult: the gate OUTCOME (mirrors the GT1 vocabulary), the resolved hit set, and the application tallies.
inline constexpr uint32_t kAreaOk                 = 0u;
inline constexpr uint32_t kAreaUnknownAbility     = 1u;
inline constexpr uint32_t kAreaUnknownCaster      = 2u;
inline constexpr uint32_t kAreaOnCooldown         = 4u;   // (3 reserved for parity with GT1 unknown-target)
inline constexpr uint32_t kAreaUnaffordable       = 5u;
inline constexpr uint32_t kAreaMissingRequiredTag = 6u;
inline constexpr uint32_t kAreaBlockedByTag       = 7u;
inline constexpr uint32_t kAreaImmune             = 8u;   // committed, but EVERY landed effect was nullified

struct AreaResult {
    uint32_t              outcome        = kAreaOk;
    std::vector<EntityId> hits;                       // the resolved target set (ascending id)
    uint32_t              effectsApplied = 0;         // effect applications that actually landed
    uint32_t              immuneCount    = 0;         // effect applications nullified by receiver immunity
    uint32_t              cuesEmitted    = 0;         // cue events emitted by this activation
};

// ImmuneTo: does `receiver`'s GT1 effective set query-match ANY of the effect's immunityTags? (GT1's model.)
inline bool ImmuneTo(const tags::TaggedWorld& tw, const tags::TagRules& tagRules, const tags::TagRegistry& reg,
                     EntityId receiver, uint32_t effectId) {
    const tags::EffectTags* et = tagRules.FindEffect(effectId);
    if (!et || et->immunityTags.empty()) return false;
    const tags::TagContainer eff = tags::EffectiveTags(tw, tagRules, receiver);
    return tags::HasAnyQuery(reg, eff, et->immunityTags);
}

// AreaActivate: gate the caster (unknown/blocked/required), COMMIT cost+cooldown VERBATIM via gas (zero-effect
// filtered ability), resolve the target set, apply kTargetOther effects per hit + kTargetSelf effects once,
// each GT1-immunity-gated, and emit the cue stream. ability.h/gameplay_tags.h are NOT modified.
inline AreaResult AreaActivate(CuesWorld& w, const gas::AbilityKit& kit, const tags::TagRules& tagRules,
                               const CueRules& cueRules, const tags::TagRegistry& reg, EntityId caster,
                               uint32_t abilityId, const TargetShape& shape) {
    AreaResult res;
    // (1) the ability (fixed linear lookup).
    const gas::AbilityDef* ab = nullptr;
    for (const gas::AbilityDef& a : kit.abilities) if (a.abilityId == abilityId) { ab = &a; break; }
    if (!ab) { res.outcome = kAreaUnknownAbility; return res; }
    // (2) the caster.
    const int ci = gas::FindEntity(w.tw.gas, caster);
    if (ci < 0) { res.outcome = kAreaUnknownCaster; return res; }
    // (3)/(4) the GT1 caster tag gates on the effective set (a failure mutates nothing; blocked BEFORE missing).
    const tags::TagContainer casterEff = tags::EffectiveTags(w.tw, tagRules, caster);
    const tags::AbilityTags* at = tagRules.FindAbility(abilityId);
    if (at) {
        if (tags::HasAnyQuery(reg, casterEff, at->blockedTags))   { res.outcome = kAreaBlockedByTag;       return res; }
        if (!tags::HasAllQuery(reg, casterEff, at->requiredTags)) { res.outcome = kAreaMissingRequiredTag; return res; }
    }
    // (5) COMMIT cost+cooldown VERBATIM via gas (a ZERO-EFFECT filtered ability -> reuses the gas cost/cooldown
    // gate; needsTarget is false with no kTargetOther effect so kNoEntity target is fine; no effects applied).
    gas::AbilityDef costOnly = *ab;
    costOnly.effects.clear();
    gas::AbilityKit costKit;
    costKit.abilities.push_back(costOnly);
    const gas::ActivateResult cr = gas::TryActivate(w.tw.gas, costKit, caster, abilityId, kNoEntity);
    if (cr == gas::kActivateOnCooldown)   { res.outcome = kAreaOnCooldown;   return res; }
    if (cr == gas::kActivateUnaffordable) { res.outcome = kAreaUnaffordable; return res; }
    // cr == kActivateOk: cost paid from BASE, cooldown started. (Unknown-ability/caster handled above.)

    const uint32_t nowTick = w.tw.gas.tick;   // the pre-step world tick (== the command's tick in StepCues)
    res.hits = ResolveTargets(w, tagRules, reg, shape);
    const CueAbility* ca = cueRules.Find(abilityId);

    // Apply kTargetOther effects to each hit (targets OUTER ascending, effects INNER array order).
    for (EntityId tgt : res.hits) {
        const int ti = gas::FindEntity(w.tw.gas, tgt);
        if (ti < 0) continue;
        const fx before = w.tw.gas.entities[(size_t)ti].attrs.base[gas::kAttrHealth];
        bool anyLanded = false;
        for (const gas::EffectDef& e : ab->effects) {
            if (e.targetMode != gas::kTargetOther) continue;
            if (ImmuneTo(w.tw, tagRules, reg, tgt, e.effectId)) { ++res.immuneCount; continue; }
            gas::ApplyEffectTo(w.tw.gas, w.tw.gas.entities[(size_t)ti], e);
            ++res.effectsApplied; anyLanded = true;
        }
        const fx after = w.tw.gas.entities[(size_t)ti].attrs.base[gas::kAttrHealth];
        if (anyLanded && ca && ca->impactCue != tags::kNoTag) {
            fx dmg = (fx)(before - after); if (dmg < 0) dmg = 0;
            EmitCue(w, ca->impactCue, caster, tgt, PosOf(w, tgt), dmg, nowTick); ++res.cuesEmitted;
        }
        if (before > 0 && after == 0 && cueRules.deathCue != tags::kNoTag) {
            EmitCue(w, cueRules.deathCue, caster, tgt, PosOf(w, tgt), 0, nowTick); ++res.cuesEmitted;
        }
    }
    // Apply kTargetSelf effects ONCE to the caster (immune-filtered).
    bool selfLanded = false;
    for (const gas::EffectDef& e : ab->effects) {
        if (e.targetMode != gas::kTargetSelf) continue;
        if (ImmuneTo(w.tw, tagRules, reg, caster, e.effectId)) { ++res.immuneCount; continue; }
        gas::ApplyEffectTo(w.tw.gas, w.tw.gas.entities[(size_t)ci], e);
        ++res.effectsApplied; selfLanded = true;
    }
    if (selfLanded && ca && ca->selfCue != tags::kNoTag) {
        EmitCue(w, ca->selfCue, caster, caster, PosOf(w, caster), 0, nowTick); ++res.cuesEmitted;
    }
    // Committed-but-fully-nullified (an immune wall) is distinct from a clean OK.
    if (res.effectsApplied == 0 && res.immuneCount > 0) res.outcome = kAreaImmune;
    return res;
}

// =================================================================================================
// 5. THE COMMAND BUS + THE COMPOSED CUE TICK (commands BEFORE the gas step) + LOCKSTEP/ROLLBACK
// =================================================================================================

struct AreaCommand {
    uint32_t    tick      = 0;
    EntityId    caster    = kNoEntity;
    uint32_t    abilityId = 0;
    TargetShape shape;
};

inline uint32_t ApplyAreaCommands(CuesWorld& w, const gas::AbilityKit& kit, const tags::TagRules& tagRules,
                                  const CueRules& cueRules, const tags::TagRegistry& reg,
                                  const std::vector<AreaCommand>& cmds, uint32_t tick,
                                  std::vector<AreaResult>* outResults = nullptr) {
    uint32_t committed = 0;
    for (const AreaCommand& c : cmds) {
        if (c.tick != tick) continue;
        const AreaResult r = AreaActivate(w, kit, tagRules, cueRules, reg, c.caster, c.abilityId, c.shape);
        if (r.outcome == kAreaOk || r.outcome == kAreaImmune) ++committed;
        if (outResults) outResults->push_back(r);
    }
    return committed;
}

// StepCues: ONE composed tick — area commands (fixed array order) BEFORE gas::StepAbilities. Also emits a
// DEATH cue for any entity whose base health reaches 0 DURING the step (a periodic-DoT kill). Returns the
// gas periodic-firing count.
inline uint32_t StepCues(CuesWorld& w, const gas::AbilityKit& kit, const tags::TagRules& tagRules,
                         const CueRules& cueRules, const tags::TagRegistry& reg,
                         const std::vector<AreaCommand>& cmds, uint32_t tick) {
    ApplyAreaCommands(w, kit, tagRules, cueRules, reg, cmds, tick);
    // Capture pre-step base health (in gas.entities index order) for step-death detection.
    std::vector<fx> hpBefore(w.tw.gas.entities.size());
    for (size_t i = 0; i < w.tw.gas.entities.size(); ++i)
        hpBefore[i] = w.tw.gas.entities[i].attrs.base[gas::kAttrHealth];
    const uint32_t periodic = gas::StepAbilities(w.tw.gas);
    if (cueRules.deathCue != tags::kNoTag) {
        for (size_t i = 0; i < w.tw.gas.entities.size(); ++i) {
            const fx after = w.tw.gas.entities[i].attrs.base[gas::kAttrHealth];
            if (hpBefore[i] > 0 && after == 0) {
                const EntityId id = w.tw.gas.entities[i].id;
                EmitCue(w, cueRules.deathCue, kNoEntity, id, PosOf(w, id), 0, tick);
            }
        }
    }
    return periodic;
}

// ----- Equality + digest (the lockstep/rollback currency; the cue stream is part of the contract) -----------
inline bool CueEventsEqual(const CueEvent& a, const CueEvent& b) {
    return a.cueTag == b.cueTag && a.sourceEntity == b.sourceEntity && a.targetEntity == b.targetEntity &&
           a.location.x == b.location.x && a.location.y == b.location.y && a.location.z == b.location.z &&
           a.magnitude == b.magnitude && a.tick == b.tick;
}

inline bool CuesStatesEqual(const CuesWorld& a, const CuesWorld& b) {
    if (!tags::TaggedStatesEqual(a.tw, b.tw)) return false;
    if (a.ents.size() != b.ents.size()) return false;
    for (size_t i = 0; i < a.ents.size(); ++i) {
        const CueEntity& x = a.ents[i];
        const CueEntity& y = b.ents[i];
        if (x.id != y.id || x.team != y.team) return false;
        if (x.pos.x != y.pos.x || x.pos.y != y.pos.y || x.pos.z != y.pos.z) return false;
        if (x.facing.x != y.facing.x || x.facing.y != y.facing.y || x.facing.z != y.facing.z) return false;
    }
    if (a.cueLog.size() != b.cueLog.size()) return false;
    for (size_t i = 0; i < a.cueLog.size(); ++i)
        if (!CueEventsEqual(a.cueLog[i], b.cueLog[i])) return false;
    return true;
}

inline uint64_t DigestCuesWorld(const CuesWorld& w) {
    verdict::DigestFnv d;
    const uint64_t g = tags::DigestTaggedWorld(w.tw);
    d.mix32((uint32_t)(g & 0xFFFFFFFFull));
    d.mix32((uint32_t)(g >> 32));
    d.sep();
    d.mix32((uint32_t)w.ents.size());
    for (const CueEntity& e : w.ents) {
        d.mix32((uint32_t)e.id);
        d.mix32((uint32_t)e.pos.x); d.mix32((uint32_t)e.pos.y); d.mix32((uint32_t)e.pos.z);
        d.mix32((uint32_t)e.facing.x); d.mix32((uint32_t)e.facing.y); d.mix32((uint32_t)e.facing.z);
        d.mix32(e.team);
    }
    d.sep();
    d.mix32((uint32_t)w.cueLog.size());
    for (const CueEvent& c : w.cueLog) {
        d.mix32(c.cueTag);
        d.mix32((uint32_t)c.sourceEntity);
        d.mix32((uint32_t)c.targetEntity);
        d.mix32((uint32_t)c.location.x); d.mix32((uint32_t)c.location.y); d.mix32((uint32_t)c.location.z);
        d.mix32((uint32_t)c.magnitude);
        d.mix32(c.tick);
    }
    return d.h;
}

inline std::string DigestHex(uint64_t h) { return gas::DigestHex(h); }

// RunCuesLockstep: two peers cloned from world0 (a value copy — the complete snapshot), both fed ONLY the same
// command stream for `ticks` StepCues ticks. *outIdentical = whether the two finals are byte-identical (target
// sets + cue streams re-derived bit-for-bit). Returns the authority world.
inline CuesWorld RunCuesLockstep(const CuesWorld& world0, const gas::AbilityKit& kit,
                                 const tags::TagRules& tagRules, const CueRules& cueRules,
                                 const tags::TagRegistry& reg, const std::vector<AreaCommand>& commands,
                                 uint32_t ticks, bool* outIdentical = nullptr) {
    CuesWorld authority = world0;
    CuesWorld replica   = world0;
    for (uint32_t t = 0; t < ticks; ++t) {
        StepCues(authority, kit, tagRules, cueRules, reg, commands, t);
        StepCues(replica,   kit, tagRules, cueRules, reg, commands, t);
    }
    if (outIdentical) *outIdentical = CuesStatesEqual(authority, replica);
    return authority;
}

// RunCuesRollback: advance 0..rollbackAt (auth), snapshot (copy), speculate <=3 ticks (mispredict), ROLLBACK,
// re-sim rollbackAt..ticks (auth). *outCorrectedEqAuthority = corrected == straight-lockstep; *outMispredict-
// Diverged = the speculation actually diverged (the non-vacuous control).
inline CuesWorld RunCuesRollback(const CuesWorld& world0, const gas::AbilityKit& kit,
                                 const tags::TagRules& tagRules, const CueRules& cueRules,
                                 const tags::TagRegistry& reg, const std::vector<AreaCommand>& authStream,
                                 const std::vector<AreaCommand>& mispredictStream, uint32_t ticks,
                                 uint32_t rollbackAt, bool* outCorrectedEqAuthority = nullptr,
                                 bool* outMispredictDiverged = nullptr) {
    CuesWorld w = world0;
    for (uint32_t t = 0; t < rollbackAt; ++t) StepCues(w, kit, tagRules, cueRules, reg, authStream, t);
    const CuesWorld snap = w;                                    // the complete restore point (a copy)
    uint32_t specTicks = ticks - rollbackAt;
    if (specTicks > 3u) specTicks = 3u;
    for (uint32_t s = 0; s < specTicks; ++s)
        StepCues(w, kit, tagRules, cueRules, reg, mispredictStream, rollbackAt + s);   // the misprediction
    const CuesWorld specWorld = w;
    w = snap;                                                    // ROLLBACK
    for (uint32_t t = rollbackAt; t < ticks; ++t) StepCues(w, kit, tagRules, cueRules, reg, authStream, t);

    if (outCorrectedEqAuthority || outMispredictDiverged) {
        CuesWorld authAtSpec = world0;
        for (uint32_t t = 0; t < rollbackAt + specTicks; ++t)
            StepCues(authAtSpec, kit, tagRules, cueRules, reg, authStream, t);
        if (outMispredictDiverged) *outMispredictDiverged = !CuesStatesEqual(specWorld, authAtSpec);
        if (outCorrectedEqAuthority) {
            const CuesWorld authFinal = RunCuesLockstep(world0, kit, tagRules, cueRules, reg, authStream, ticks);
            *outCorrectedEqAuthority = CuesStatesEqual(w, authFinal);
        }
    }
    return w;
}

// =================================================================================================
// 6. THE SHOWCASE FIXTURE — THE CUE BATTLE (FIXED forever; the test pins the registry/rules/kit digests, the
// per-tick health + cue-stream trace digest, and the exact finals; the --gc1-cues shots run THIS EXACT
// scenario on both backends — byte-identical viz BY CONSTRUCTION)
// =================================================================================================

// The pinned ability ids (a GC1-local kit; GAS1's MakeCoreKit / GT1's MakeSkirmishKit stay untouched).
inline constexpr uint32_t kAbFireball = 1u;   // sphere/box AoE: instant -30 health + a -2/tick burn (3 ticks)
inline constexpr uint32_t kAbShield   = 2u;   // self: +15 armor for 8 ticks
inline constexpr uint32_t kAbScorch   = 3u;   // cone: instant -25 health

inline constexpr uint32_t kEffFireHit  = 10u;
inline constexpr uint32_t kEffFireBurn = 11u;
inline constexpr uint32_t kEffShield   = 20u;
inline constexpr uint32_t kEffScorch   = 30u;

// The cue + team tag registry (ids pinned by authoring order). See the accessors below.
inline tags::TagRegistry MakeCuesRegistry() {
    tags::TagRegistry r;
    r.Intern("Team.Ally");         // "Team"(0), "Team.Ally"(1)
    r.Intern("Team.Enemy");        // "Team.Enemy"(2)
    r.Intern("Cue.Impact.Fire");   // "Cue"(3), "Cue.Impact"(4), "Cue.Impact.Fire"(5)
    r.Intern("Cue.Buff.Shield");   // "Cue.Buff"(6), "Cue.Buff.Shield"(7)
    r.Intern("Cue.Death");         // "Cue.Death"(8)
    return r;
}
inline tags::TagId TagTeamAlly(const tags::TagRegistry& r)   { return r.Find("Team.Ally"); }      // 1
inline tags::TagId TagTeamEnemy(const tags::TagRegistry& r)  { return r.Find("Team.Enemy"); }     // 2
inline tags::TagId CueImpactFire(const tags::TagRegistry& r) { return r.Find("Cue.Impact.Fire"); }// 5
inline tags::TagId CueBuffShield(const tags::TagRegistry& r) { return r.Find("Cue.Buff.Shield"); }// 7
inline tags::TagId CueDeath(const tags::TagRegistry& r)      { return r.Find("Cue.Death"); }      // 8

// MakeCuesKit: the ability kit (authored through KitBuilder — the "authored, not hardcoded" discipline).
// Fireball cd 0 so it can fire on both the sphere and the box cast; costs isolate the area behaviour lightly.
inline gas::AbilityKit MakeCuesKit() {
    gas::KitBuilder b;
    b.Ability(kAbFireball, gas::kAttrMana, 15 * kOne, 0u)
     .Effect(kEffFireHit,  gas::kAttrHealth, gas::kOpAdd, -30 * kOne, gas::kDurInstant, 0u,  gas::kStackIgnore,  1u, 0u, gas::kTargetOther)
     .Effect(kEffFireBurn, gas::kAttrHealth, gas::kOpAdd, -2 * kOne,  gas::kDurTicks,   3u,  gas::kStackRefresh, 1u, 1u, gas::kTargetOther);
    b.Ability(kAbShield, gas::kAttrMana, 10 * kOne, 0u)
     .Effect(kEffShield, gas::kAttrArmor, gas::kOpAdd, 15 * kOne, gas::kDurTicks, 8u, gas::kStackRefresh, 1u, 0u, gas::kTargetSelf);
    b.Ability(kAbScorch, gas::kAttrMana, 12 * kOne, 0u)
     .Effect(kEffScorch, gas::kAttrHealth, gas::kOpAdd, -25 * kOne, gas::kDurInstant, 0u, gas::kStackIgnore, 1u, 0u, gas::kTargetOther);
    return b.Build();
}

// MakeCuesTagRules: no ability blocking here (the FILTER, not TagRules, does friendly-fire) — an empty ruleset
// (EffectiveTags then returns owned team tags for the filter). A NON-empty container would gate; GC1's battle
// leans on the shape tag FILTER, so the rules are intentionally empty (the effective set = owned team tags).
inline tags::TagRules MakeCuesTagRules() {
    tags::TagRules rules;   // (deliberately empty — the demo gates via the shape tag filter, not blocked/required)
    return rules;
}

// MakeCueRules: the authored cue table — Fireball/Scorch fire Cue.Impact.Fire per hit; Shield a Cue.Buff.Shield
// on the caster; a global Cue.Death when a target dies.
inline CueRules MakeCueRules(const tags::TagRegistry& reg) {
    CueRules cr;
    cr.abilities.push_back(CueAbility{kAbFireball, CueImpactFire(reg), tags::kNoTag});
    cr.abilities.push_back(CueAbility{kAbShield,   tags::kNoTag,       CueBuffShield(reg)});
    cr.abilities.push_back(CueAbility{kAbScorch,   CueImpactFire(reg), tags::kNoTag});
    cr.deathCue = CueDeath(reg);
    return cr;
}

// The battlefield: 6 entities (health 50/50, mana 100/100). Allies carry the Team.Ally owned tag; enemies
// Team.Enemy. Positions in Q16.16 (integer world units). e1 is the caster (ally).
inline constexpr uint32_t kBattleEnts  = 6u;
inline constexpr uint32_t kBattleTicks = 10u;
inline constexpr fx kBattleBases[gas::kAttrCount] = {
    50 * kOne, 50 * kOne, 100 * kOne, 100 * kOne, 4 * kOne, 10 * kOne, 5 * kOne, 0,
};

// The THREE canonical shapes (shared by the script AND the viz panels; friendly-fire OFF via Team.Ally).
inline SphereTarget BattleSphere() { return SphereTarget{FxVec3{2 * kOne, 0, 0}, 2 * kOne}; }
inline BoxTarget    BattleBox()    { return BoxTarget{FxVec3{2 * kOne, 0, 0}, FxVec3{3 * kOne / 2, 3 * kOne / 2, 3 * kOne / 2}}; }
inline ConeTarget   BattleCone()   { return ConeTarget{FxVec3{0, 0, 0}, FxVec3{kOne, 0, 0}, 9 * kOne, 32768}; } // cos60

inline TargetShape SphereCast(const tags::TagRegistry& reg) {
    TargetShape s = MakeSphereShape(BattleSphere().center, BattleSphere().radiusQ);
    s.filterTag = TagTeamAlly(reg); s.filterExclude = true; return s;   // friendly-fire OFF
}
inline TargetShape BoxCast(const tags::TagRegistry& reg) {
    TargetShape s = MakeBoxShape(BattleBox().center, BattleBox().halfExtents);
    s.filterTag = TagTeamAlly(reg); s.filterExclude = true; return s;
}
inline TargetShape ConeCast(const tags::TagRegistry& reg) {
    const ConeTarget c = BattleCone();
    TargetShape s = MakeConeShape(c.apex, c.dir, c.rangeQ, c.cosHalfAngleQ);
    s.filterTag = TagTeamAlly(reg); s.filterExclude = true; return s;
}

// MakeBattleWorld: e1 ally caster (0,0); e2 enemy (2,0); e3 enemy (3,1); e4 enemy (2,-1); e5 ally (1,0);
// e6 enemy (8,0). Spawn order pinned (ascending id).
inline CuesWorld MakeBattleWorld(const tags::TagRegistry& reg) {
    CuesWorld w;
    tags::TagContainer ally;  ally.Add(TagTeamAlly(reg));
    tags::TagContainer enemy; enemy.Add(TagTeamEnemy(reg));
    SpawnCueEntity(w, kBattleBases, ally,  FxVec3{0,          0,        0}, 0u);   // e1 caster (ally)
    SpawnCueEntity(w, kBattleBases, enemy, FxVec3{2 * kOne,   0,        0}, 1u);   // e2 enemy
    SpawnCueEntity(w, kBattleBases, enemy, FxVec3{3 * kOne,   1 * kOne, 0}, 1u);   // e3 enemy
    SpawnCueEntity(w, kBattleBases, enemy, FxVec3{2 * kOne,  -1 * kOne, 0}, 1u);   // e4 enemy
    SpawnCueEntity(w, kBattleBases, ally,  FxVec3{1 * kOne,   0,        0}, 0u);   // e5 ally
    SpawnCueEntity(w, kBattleBases, enemy, FxVec3{8 * kOne,   0,        0}, 1u);   // e6 enemy (far)
    return w;
}

// MakeBattleStream: the FIXED 10-tick script. t1 sphere fireball, t2 self shield, t3 box fireball (kills),
// t6 cone scorch (far enemy). Friendly-fire OFF excludes the two allies (incl. the caster on the sphere
// boundary) every cast.
inline std::vector<AreaCommand> MakeBattleStream(const tags::TagRegistry& reg) {
    const EntityId C = 1u;
    return {
        AreaCommand{1u, C, kAbFireball, SphereCast(reg)},   // hits e2,e3,e4 (allies + far e6 excluded)
        AreaCommand{2u, C, kAbShield,   MakeSingleShape(C)},// self buff (no area targets; self cue)
        AreaCommand{3u, C, kAbFireball, BoxCast(reg)},      // hits e2,e3,e4 again -> deaths + death cues
        AreaCommand{6u, C, kAbScorch,   ConeCast(reg)},     // cone reaches e6 (8,0) -> impact
    };
}

// ----- BattleTickSample / BattleRun / RunCuesBattle: the shared per-tick trace (test + BOTH showcases) ------
struct BattleTickSample {
    fx       health[kBattleEnts];   // current health per entity AFTER this tick's step (index = spawn order)
    uint8_t  alive[kBattleEnts];    // base health > 0 after the step
    uint32_t cuesThisTick;          // cues emitted at this tick
};
struct BattleRun {
    CuesWorld                    finalWorld;
    std::vector<BattleTickSample> samples;        // kBattleTicks entries
    uint64_t                     traceDigest = 0; // per-tick health + cue-stream fold (pinned)
    uint32_t                     shapesCast  = 0;  // area/single casts issued
    uint32_t                     targetsHit  = 0;  // total resolved hits across casts
    uint32_t                     cuesTotal   = 0;  // total cue events
    uint32_t                     deaths      = 0;  // Cue.Death events
};

inline BattleRun RunCuesBattle() {
    const tags::TagRegistry reg   = MakeCuesRegistry();
    const gas::AbilityKit   kit   = MakeCuesKit();
    const tags::TagRules    trul  = MakeCuesTagRules();
    const CueRules          cr    = MakeCueRules(reg);
    const std::vector<AreaCommand> stream = MakeBattleStream(reg);
    const tags::TagId deathTag = CueDeath(reg);

    BattleRun run;
    CuesWorld w = MakeBattleWorld(reg);
    verdict::DigestFnv trace;
    for (uint32_t t = 0; t < kBattleTicks; ++t) {
        // Commands BEFORE the step (array order) — tally casts/hits + fold the per-cast outcome.
        for (const AreaCommand& c : stream) {
            if (c.tick != t) continue;
            const AreaResult r = AreaActivate(w, kit, trul, cr, reg, c.caster, c.abilityId, c.shape);
            ++run.shapesCast;
            run.targetsHit += (uint32_t)r.hits.size();
            trace.mix32(r.outcome);
            trace.mix32((uint32_t)r.hits.size());
            trace.mix32(r.effectsApplied);
            for (EntityId h : r.hits) trace.mix32((uint32_t)h);
        }
        trace.sep();
        // The step (commands were applied inline above to tally r; pass an EMPTY stream so they are NOT
        // double-applied — StepCues here runs only the gas step + any DoT-death cues for this tick).
        (void)StepCues(w, kit, trul, cr, reg, std::vector<AreaCommand>{}, t);
        // Sample AFTER the step.
        BattleTickSample s{};
        for (uint32_t i = 0; i < kBattleEnts; ++i) {
            if ((size_t)i < w.tw.gas.entities.size()) {
                const gas::GasEntity& e = w.tw.gas.entities[(size_t)i];
                s.health[i] = e.attrs.current[gas::kAttrHealth];
                s.alive[i]  = (e.attrs.base[gas::kAttrHealth] > 0) ? 1 : 0;
                trace.mix32((uint32_t)e.attrs.current[gas::kAttrHealth]);
            }
        }
        // Fold this tick's cue events (the stream is the contract) + count them.
        uint32_t stamped = 0;
        for (const CueEvent& ce : w.cueLog) {
            if (ce.tick != t) continue;
            trace.mix32(ce.cueTag);
            trace.mix32((uint32_t)ce.targetEntity);
            trace.mix32((uint32_t)ce.magnitude);
            ++stamped;
        }
        trace.sep();
        s.cuesThisTick = stamped;
        run.samples.push_back(s);
    }
    run.cuesTotal = (uint32_t)w.cueLog.size();
    for (const CueEvent& ce : w.cueLog) if (ce.cueTag == deathTag) ++run.deaths;
    run.traceDigest = trace.h;
    run.finalWorld  = w;
    return run;
}

// =================================================================================================
// 7. THE SHOWCASE VIZ (strict-integer, NO shader) — shared VERBATIM by --gc1-cues-shot (Vulkan) and
// --gc1-cues (Metal) so the pixels are byte-identical cross-backend BY CONSTRUCTION.
// A battlefield map (entities as team-colored dots; the three targeting SHAPES shaded — sphere circle / box
// rect / cone wedge — with the hit entities ringed + cue markers at impact points), a per-entity HEALTH
// timeline grid, and a CUE-STREAM ribbon. Every coordinate is an integer map of the Q16.16 world.
// =================================================================================================

inline constexpr int kGc1ImgW   = 620;
inline constexpr int kGc1ImgH   = 360;
inline constexpr int kGc1MapX0  = 24;    // battlefield origin x (world x=0)
inline constexpr int kGc1MapY0  = 96;    // battlefield origin y (world y=0; +y is UP -> screen up)
inline constexpr int kGc1Scale  = 40;    // pixels per world unit

struct Gc1VizStats {
    uint32_t entities   = 0;
    uint32_t shapes     = 0;   // shapes/casts drawn
    uint32_t targetsHit = 0;
    uint32_t cues       = 0;
    uint32_t ticks      = 0;
    uint32_t width      = 0;
    uint32_t height     = 0;
    uint64_t pixDigest  = 0;   // FNV over the RGBA8 pixels (the cross-backend strict-zero proof)
    uint64_t battleDigest = 0; // DigestCuesWorld(final) (the runtime determinism digest)
};

// Map a Q16.16 world position to battlefield pixels (y inverted; +y world -> up on screen).
inline int Gc1MapX(fx wx) { return kGc1MapX0 + (int)(((int64_t)wx * kGc1Scale) / (int64_t)kOne); }
inline int Gc1MapY(fx wy) { return kGc1MapY0 - (int)(((int64_t)wy * kGc1Scale) / (int64_t)kOne); }

inline void RenderGc1CuesViz(std::vector<uint8_t>& out, Gc1VizStats& stats) {
    const int W = kGc1ImgW, H = kGc1ImgH;
    out.assign((std::size_t)W * H * 4u, 0);
    auto px = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        uint8_t* d = &out[((std::size_t)y * W + x) * 4u];
        d[0] = r; d[1] = g; d[2] = b; d[3] = 255;
    };
    auto fillRect = [&](int x0, int y0, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y < y0 + h; ++y) for (int x = x0; x < x0 + w; ++x) px(x, y, r, g, b);
    };
    auto ring = [&](int cx, int cy, int rad, uint8_t r, uint8_t g, uint8_t b) {
        const int r2i = rad * rad, r2o = (rad + 1) * (rad + 1);
        for (int dy = -rad - 1; dy <= rad + 1; ++dy)
            for (int dx = -rad - 1; dx <= rad + 1; ++dx) {
                const int dd = dx * dx + dy * dy;
                if (dd >= r2i && dd <= r2o) px(cx + dx, cy + dy, r, g, b);
            }
    };

    // Background.
    fillRect(0, 0, W, H, 14, 15, 20);

    const tags::TagRegistry reg = MakeCuesRegistry();
    const BattleRun run = RunCuesBattle();

    // ---- The battlefield map: shade the three shape footprints, then draw entities + hit rings + cue marks.
    const int mapX = kGc1MapX0 - 8, mapY = 20, mapW = 9 * kGc1Scale + 20, mapH = 150;
    fillRect(mapX, mapY, mapW, mapH, 22, 24, 30);

    const SphereTarget sph = BattleSphere();
    const BoxTarget    bx  = BattleBox();
    const ConeTarget   cn  = BattleCone();
    // Per-pixel world membership shading over the map band (deterministic integer inverse map).
    for (int sy = mapY; sy < mapY + mapH; ++sy) {
        for (int sx = mapX; sx < mapX + mapW; ++sx) {
            const fx wx = (fx)(((int64_t)(sx - kGc1MapX0) * (int64_t)kOne) / (int64_t)kGc1Scale);
            const fx wy = (fx)(((int64_t)(kGc1MapY0 - sy) * (int64_t)kOne) / (int64_t)kGc1Scale);
            const FxVec3 wp{wx, wy, 0};
            uint8_t r = 0, g = 0, b = 0; bool any = false;
            if (InCone(cn, wp))   { r = 60;  g = 40;  b = 12; any = true; }  // cone wedge (amber)
            if (InBox(bx, wp))    { r = 16;  g = 42;  b = 52; any = true; }  // box (teal)
            if (InSphere(sph, wp)){ r = 52;  g = 20;  b = 20; any = true; }  // sphere (red)
            if (any) {
                uint8_t* d = &out[((std::size_t)sy * W + sx) * 4u];
                // additive-ish tint over the map bg (saturating).
                int nr = d[0] + r, ng = d[1] + g, nb = d[2] + b;
                d[0] = (uint8_t)(nr > 255 ? 255 : nr);
                d[1] = (uint8_t)(ng > 255 ? 255 : ng);
                d[2] = (uint8_t)(nb > 255 ? 255 : nb);
            }
        }
    }
    // Shape outlines: sphere circle + box rectangle + cone apex marker.
    {
        const int scx = Gc1MapX(sph.center.x), scy = Gc1MapY(sph.center.y);
        const int srad = (int)(((int64_t)sph.radiusQ * kGc1Scale) / (int64_t)kOne);
        ring(scx, scy, srad, 210, 90, 90);
        const int bx0 = Gc1MapX((fx)(bx.center.x - bx.halfExtents.x));
        const int bx1 = Gc1MapX((fx)(bx.center.x + bx.halfExtents.x));
        const int by0 = Gc1MapY((fx)(bx.center.y + bx.halfExtents.y));
        const int by1 = Gc1MapY((fx)(bx.center.y - bx.halfExtents.y));
        for (int x = bx0; x <= bx1; ++x) { px(x, by0, 90, 200, 220); px(x, by1, 90, 200, 220); }
        for (int y = by0; y <= by1; ++y) { px(bx0, y, 90, 200, 220); px(bx1, y, 90, 200, 220); }
        fillRect(Gc1MapX(cn.apex.x) - 2, Gc1MapY(cn.apex.y) - 2, 4, 4, 230, 180, 60);
    }

    // The entities (recompute their spatial state from the fixture) + hit rings for the sphere cast.
    const CuesWorld w0 = MakeBattleWorld(reg);
    const std::vector<EntityId> sphereHits = ResolveTargets(w0, MakeCuesTagRules(), reg, SphereCast(reg));
    for (const CueEntity& e : w0.ents) {
        const int cx = Gc1MapX(e.pos.x), cy = Gc1MapY(e.pos.y);
        uint8_t r = (e.team == 0) ? 90 : 220, g = (e.team == 0) ? 200 : 70, b = (e.team == 0) ? 120 : 60;
        fillRect(cx - 3, cy - 3, 7, 7, r, g, b);
        bool hit = false; for (EntityId h : sphereHits) if (h == e.id) hit = true;
        if (hit) ring(cx, cy, 6, 250, 230, 90);   // gold hit ring
    }
    // Cue markers (impact/death) at their locations (small diamonds).
    for (const CueEvent& ce : run.finalWorld.cueLog) {
        const int cx = Gc1MapX(ce.location.x), cy = Gc1MapY(ce.location.y);
        const bool death = (ce.cueTag == CueDeath(reg));
        const uint8_t r = death ? 240 : 250, g = death ? 60 : 160, b = death ? 200 : 40;
        for (int dy = -2; dy <= 2; ++dy) { const int span = 2 - (dy < 0 ? -dy : dy);
            for (int dx = -span; dx <= span; ++dx) px(cx + dx, cy + dy, r, g, b); }
    }

    // ---- Per-entity HEALTH timeline grid (rows = entities, cols = ticks). ----------------------------------
    const int gx0 = 24, gy0 = 196, colW = 34, rowH = 20, rowGap = 4;
    for (uint32_t i = 0; i < kBattleEnts; ++i) {
        const int ry = gy0 + (int)i * (rowH + rowGap);
        fillRect(gx0 - 2, ry - 2, (int)kBattleTicks * colW + 4, rowH + 4, 24, 26, 32);
        for (uint32_t t = 0; t < kBattleTicks; ++t) {
            const BattleTickSample& s = run.samples[t];
            int hp = (int)(s.health[i] >> kFrac); if (hp < 0) hp = 0; if (hp > 50) hp = 50;
            int hw = (hp * (colW - 4)) / 50;
            const uint8_t r = s.alive[i] ? 200 : 60, g = s.alive[i] ? 70 : 30, b = 60;
            fillRect(gx0 + (int)t * colW + 2, ry + 2, hw, rowH - 4, r, g, b);
        }
    }

    // ---- CUE-STREAM ribbon (a column per cue in log order, colored by cue tag). ----------------------------
    const int rx0 = 24, ry0 = 340, cw = 14;
    for (size_t ci = 0; ci < run.finalWorld.cueLog.size() && (int)(rx0 + (int)ci * cw) < W - cw; ++ci) {
        const CueEvent& ce = run.finalWorld.cueLog[ci];
        uint8_t r = 120, g = 120, b = 120;
        if (ce.cueTag == CueImpactFire(reg)) { r = 240; g = 150; b = 40; }
        else if (ce.cueTag == CueBuffShield(reg)) { r = 90; g = 180; b = 240; }
        else if (ce.cueTag == CueDeath(reg)) { r = 230; g = 50; b = 200; }
        fillRect(rx0 + (int)ci * cw, ry0, cw - 2, 12, r, g, b);
    }

    // ---- Stats. --------------------------------------------------------------------------------------------
    stats.entities     = kBattleEnts;
    stats.shapes       = run.shapesCast;
    stats.targetsHit   = run.targetsHit;
    stats.cues         = run.cuesTotal;
    stats.ticks        = kBattleTicks;
    stats.width        = (uint32_t)W;
    stats.height       = (uint32_t)H;
    stats.battleDigest = DigestCuesWorld(run.finalWorld);
    verdict::DigestFnv pd;
    for (uint8_t byte : out) pd.mix32((uint32_t)byte);
    stats.pixDigest = pd.h;
}

}  // namespace cues
}  // namespace hf::game
