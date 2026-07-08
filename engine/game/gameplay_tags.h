#pragma once
// Slice GT1 — A DETERMINISTIC GAMEPLAY-TAG LAYER (hierarchical tag containers gating abilities/effects,
// hf::game::tags). The #1 next-tier parity gap on the gameplay-framework pillar: the shipped GAS1 ability
// system (game/ability.h — attributes/effects/cooldowns) has ZERO gameplay tags, and a gameplay-tag system
// is the foundational layer that makes an ability system expressive (UE5's GameplayTags: requirement /
// blocking / immunity gating). GT1 adds it deterministically and COMPOSES it onto GAS1 through a THIN
// ADAPTER — game/ability.h is #included READ-ONLY / BYTE-UNTOUCHED (no additive edits): the tag gates run
// AROUND gas::TryActivate, and immunity is expressed by handing gas::TryActivate a per-activation FILTERED
// ability (the immune effects removed) so ALL of GAS1's cost/cooldown/apply logic is reused verbatim.
//
// EVERYTHING IS PURE INTEGER (tags are interned to stable uint32 ids; containers are sorted id vectors; NO
// float, NO wall clock, NO strings on the hot path — the registry is a STATIC authored asset, like the GAS1
// kit). UE5's GameplayTags carry runtime FName hashing + tag-net-index maps that vary by cook order; this
// one is a pure function of the authored tag table: a peer re-derives every tag-gated ability outcome
// bit-for-bit from the input stream, and a rollback corrects a mispredicted activation that was actually
// tag-blocked.
//
// THE PINNED SEMANTICS (the determinism contract):
//   * INTERNING: a gameplay tag is a hierarchical DOTTED name ("Ability.Fire.Fireball", "State.Stunned").
//     TagRegistry::Intern(name) returns a stable TagId (the insertion index) and is idempotent. Interning a
//     deep tag AUTO-INTERNS every dotted PREFIX shortest->longest ("Ability.Fire.Fireball" also interns
//     "Ability" then "Ability.Fire"), so ids are a deterministic function of the authoring ORDER. Each id
//     stores its ANCESTOR chain (all segment-prefix ids incl. self, in prefix order) — the hierarchy key.
//   * HIERARCHY / QUERY MATCHING (the standard GameplayTag rule): a container holding "A.B.C" QUERY-matches
//     "A" and "A.B" (owning a specific tag satisfies queries for its parents). Implemented on the interned
//     representation: HasTagQuery(reg, owned, q) is true iff some owned tag o has q in reg.ancestors[o] (i.e.
//     owned holds q OR a DESCENDANT of q). Matching is on SEGMENT boundaries — "A.BB" does NOT match query
//     "A.B" (ancestors of "A.BB" = {"A","A.BB"}, which excludes "A.B"), the raw-string-prefix false positive
//     is structurally impossible. HasTagExact is plain id membership (no hierarchy).
//   * CONTAINER: TagContainer is a SORTED-ASCENDING, DEDUPED id vector (deterministic order -> a stable
//     digest). Add/Remove keep the invariant; HasTagExact/HasAnyExact/HasAllExact are exact set ops;
//     HasTagQuery/HasAnyQuery/HasAllQuery are the parent-prefix-aware forms.
//   * EFFECTIVE TAGS (the lifecycle-bound derivation, the crux): an entity's EFFECTIVE tag set is DERIVED
//     each query = its OWNED (persistent/innate) tags UNION the grantsTags of every LIVE active effect on
//     it (looked up by def.effectId in the tag rules). This binds effect-granted tags to the GAS1 effect
//     lifecycle for FREE: a stun effect grants "State.Stunned" while it is in the entity's active[] and the
//     grant VANISHES the tick GAS1 removes the expired effect — no separate bookkeeping, so the snapshot
//     completeness follows from the GAS1 active[] + the owned container alone.
//   * GATING (the adapter, in PINNED order): TryActivateTagged runs (1) unknown-ability, (2) unknown-caster,
//     then the TAG GATES on the caster's EFFECTIVE set: (3) blockedTags — blocked if ANY present (query
//     match) else kTagBlockedByTag; (4) requiredTags — activation needs ALL (query match) else
//     kTagMissingRequiredTag; a tag-gate failure mutates NOTHING. BLOCKED is PINNED BEFORE MISSING-REQUIRED
//     (a stunned unit is blocked regardless of what else it lacks). Then it delegates the remaining GAS1 gates
//     (unknown-target / cooldown / cost) + the apply to gas::TryActivate via a FILTERED single-ability kit:
//     each effect whose RECEIVER (target for kTargetOther, caster for kTargetSelf) has ANY of the effect's
//     immunityTags in its effective set is REMOVED (immuneCount++), so the effect is nullified while the
//     activation still commits cost+cooldown (the UE5 immunity model: immunity blocks the EFFECT, not the
//     ability). On gas success, the ability's grantedTags are unioned into the caster's OWNED set.
//   * OUTCOME (FIXED numbering — the proofs pin these): kTagOk / kTagUnknownAbility / kTagUnknownCaster /
//     kTagUnknownTarget / kTagOnCooldown / kTagUnaffordable mirror gas::ActivateResult 0..5; kTagMissing-
//     RequiredTag / kTagBlockedByTag are the hard tag-gate failures; kTagImmune is the COMMITTED outcome
//     when the activation succeeded but >=1 effect was nullified by receiver immunity (distinct from kTagOk).
//   * STEP: StepTagged applies this tick's commands (fixed array order) BEFORE gas::StepAbilities — the GAS1
//     commands-before-step contract. No tag-specific step logic is needed: effect-granted tags follow the
//     GAS1 effect lifecycle through the EFFECTIVE derivation.
//
// HONEST SCOPE / CAVEATS: ability.grantedTags are added to the caster's OWNED set on a successful activation
// and PERSIST (GAS1 has no ability-active duration to hang a removal on — the lifecycle-bound grants are the
// EFFECT grantsTags, which is what the stun/empower proofs use); immunity is per-(effect,receiver) and does
// NOT refund the ability's cost/cooldown (the ability committed); the TagRegistry/TagRules are a STATIC
// asset shared by every peer (interned ids match because the authoring order matches — NOT snapshot state).
//
// PURE CPU INTEGER. Header-only, namespace hf::game::tags. #includes game/ability.h READ-ONLY (which itself
// pulls game/verdict.h read-only — the EntityId vocabulary + the DigestFnv discipline). NO render RHI, NO
// new shader, NO new compute.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "game/ability.h"   // read-only: the whole GAS1 core (GasWorld/AbilityKit/TryActivate/StepAbilities)

namespace hf::game {
namespace tags {

// Reuse the GAS layer's vocabulary verbatim (NO new primitives).
namespace gas = hf::game::gas;
using gas::fx;
using gas::kOne;
using gas::kFrac;
using verdict::EntityId;
using verdict::kNoEntity;

// ----- TagId + the registry (interning + the hierarchy ancestor chains) -------------------------------------
using TagId = uint32_t;
inline constexpr TagId kNoTag = 0xFFFFFFFFu;   // the sentinel (an absent/unknown tag)

// Split a dotted name into its segment PREFIXES, shortest -> longest ("A.B.C" -> "A","A.B","A.B.C").
inline std::vector<std::string> SegmentPrefixes(const std::string& name) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= name.size()) {
        size_t dot = name.find('.', start);
        if (dot == std::string::npos) { out.push_back(name); break; }
        out.push_back(name.substr(0, dot));
        start = dot + 1;
    }
    return out;
}

// TagRegistry: the deterministic string<->id table. Ids are insertion indices; Intern is idempotent and
// auto-interns every dotted prefix shortest->longest, so ids are a pure function of the authoring order.
// ancestors[id] = the segment-prefix ids of that tag (incl. self), in prefix order — the hierarchy key.
struct TagRegistry {
    std::vector<std::string>          names;        // id -> dotted name
    std::vector<std::vector<TagId>>   ancestors;    // id -> {prefix ids incl. self}, prefix order

    TagId Find(const std::string& name) const {
        for (size_t i = 0; i < names.size(); ++i)
            if (names[i] == name) return (TagId)i;
        return kNoTag;
    }

    // InternSingle: intern ONE exact name (no prefix expansion). Assumes all SHORTER prefixes already exist
    // (the caller — Intern — interns shortest-first), so ancestors resolve by Find.
    TagId InternSingle(const std::string& name) {
        const TagId existing = Find(name);
        if (existing != kNoTag) return existing;
        const TagId id = (TagId)names.size();
        names.push_back(name);
        std::vector<TagId> anc;
        const std::vector<std::string> prefixes = SegmentPrefixes(name);
        for (const std::string& p : prefixes) {
            if (p == name) anc.push_back(id);          // self (this id, not yet Find-able mid-construction)
            else           anc.push_back(Find(p));     // a strict ancestor — interned earlier (shortest-first)
        }
        ancestors.push_back(anc);
        return id;
    }

    // Intern: intern `name` AND every dotted prefix, shortest -> longest. Returns the id of the full name.
    TagId Intern(const std::string& name) {
        const std::vector<std::string> prefixes = SegmentPrefixes(name);
        TagId last = kNoTag;
        for (const std::string& p : prefixes) last = InternSingle(p);
        return last;
    }

    uint32_t size() const { return (uint32_t)names.size(); }
};

// ----- TagContainer: a sorted-ascending, deduped id set (the deterministic tag set) -------------------------
struct TagContainer {
    std::vector<TagId> ids;   // sorted ascending, unique

    void Add(TagId t) {
        if (t == kNoTag) return;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == t) return;                    // already present
            if (ids[i] > t) { ids.insert(ids.begin() + (std::ptrdiff_t)i, t); return; }
        }
        ids.push_back(t);                               // largest so far
    }
    void Remove(TagId t) {
        for (size_t i = 0; i < ids.size(); ++i)
            if (ids[i] == t) { ids.erase(ids.begin() + (std::ptrdiff_t)i); return; }
    }
    bool HasTagExact(TagId t) const {
        for (TagId x : ids) if (x == t) return true;
        return false;
    }
    void Union(const TagContainer& o) { for (TagId t : o.ids) Add(t); }
    bool empty() const { return ids.empty(); }
    size_t size() const { return ids.size(); }
};

// ----- The hierarchy-aware query predicates (the parent-prefix matching) ------------------------------------
// HasTagQuery: true iff `owned` holds `query` OR any DESCENDANT of `query` (owning "A.B.C" matches query "A").
inline bool HasTagQuery(const TagRegistry& reg, const TagContainer& owned, TagId query) {
    if (query == kNoTag) return false;
    for (TagId o : owned.ids) {
        if (o >= reg.ancestors.size()) continue;
        for (TagId a : reg.ancestors[o]) if (a == query) return true;
    }
    return false;
}
inline bool HasAnyQuery(const TagRegistry& reg, const TagContainer& owned, const TagContainer& queries) {
    for (TagId q : queries.ids) if (HasTagQuery(reg, owned, q)) return true;
    return false;
}
inline bool HasAllQuery(const TagRegistry& reg, const TagContainer& owned, const TagContainer& queries) {
    for (TagId q : queries.ids) if (!HasTagQuery(reg, owned, q)) return false;
    return true;
}
// Exact (no hierarchy) set ops between two containers.
inline bool HasAnyExact(const TagContainer& a, const TagContainer& b) {
    for (TagId t : b.ids) if (a.HasTagExact(t)) return true;
    return false;
}
inline bool HasAllExact(const TagContainer& a, const TagContainer& b) {
    for (TagId t : b.ids) if (!a.HasTagExact(t)) return false;
    return true;
}

inline uint64_t DigestContainer(verdict::DigestFnv& d, const TagContainer& c) {
    d.mix32((uint32_t)c.ids.size());
    for (TagId t : c.ids) d.mix32(t);
    d.sep();
    return d.h;
}

// ----- The tag rules (the STATIC authored gating asset, parallel to the GAS1 kit) ---------------------------
// AbilityTags: per-ability tag gates. requiredTags (activation needs ALL, query), blockedTags (blocked if
// ANY present, query), grantedTags (unioned into the caster's OWNED set on a successful activation).
struct AbilityTags {
    uint32_t     abilityId = 0;
    TagContainer requiredTags;
    TagContainer blockedTags;
    TagContainer grantedTags;
};
// EffectTags: per-effect tag data. immunityTags (the effect is nullified if the RECEIVER has ANY, query),
// grantsTags (added to the receiver's EFFECTIVE set while the effect is a live active effect).
struct EffectTags {
    uint32_t     effectId = 0;
    TagContainer immunityTags;
    TagContainer grantsTags;
};
struct TagRules {
    std::vector<AbilityTags> abilities;
    std::vector<EffectTags>  effects;

    const AbilityTags* FindAbility(uint32_t abilityId) const {
        for (const AbilityTags& a : abilities) if (a.abilityId == abilityId) return &a;
        return nullptr;
    }
    const EffectTags* FindEffect(uint32_t effectId) const {
        for (const EffectTags& e : effects) if (e.effectId == effectId) return &e;
        return nullptr;
    }
};

inline uint64_t DigestRules(const TagRules& rules) {
    verdict::DigestFnv d;
    d.mix32((uint32_t)rules.abilities.size());
    for (const AbilityTags& a : rules.abilities) {
        d.mix32(a.abilityId);
        DigestContainer(d, a.requiredTags);
        DigestContainer(d, a.blockedTags);
        DigestContainer(d, a.grantedTags);
        d.sep();
    }
    d.mix32((uint32_t)rules.effects.size());
    for (const EffectTags& e : rules.effects) {
        d.mix32(e.effectId);
        DigestContainer(d, e.immunityTags);
        DigestContainer(d, e.grantsTags);
        d.sep();
    }
    return d.h;
}

// ----- TaggedWorld: the GAS1 world + the parallel per-entity OWNED tag containers ---------------------------
// The owned containers are the ONLY tag STATE (kept parallel to gas.entities by id, spawn order). A value
// copy IS a complete snapshot (both members are plain vectors). The registry/rules are static assets.
struct EntityTags {
    EntityId     id = kNoEntity;
    TagContainer owned;      // persistent/innate tags (immunities, committed grants)
};
struct TaggedWorld {
    gas::GasWorld           gas;
    std::vector<EntityTags> tags;    // parallel to gas.entities (same ids, spawn order)
};

inline int FindTags(const TaggedWorld& tw, EntityId id) {
    for (size_t i = 0; i < tw.tags.size(); ++i) if (tw.tags[i].id == id) return (int)i;
    return -1;
}

// SpawnTagged: GAS1 spawn + a parallel owned container (the same monotonic id).
inline EntityId SpawnTagged(TaggedWorld& tw, const fx (&bases)[gas::kAttrCount], const TagContainer& owned) {
    const EntityId id = gas::SpawnGasEntity(tw.gas, bases);
    tw.tags.push_back(EntityTags{id, owned});
    return id;
}

// EffectiveTags: the DERIVED tag set of an entity = OWNED union grantsTags(every live active effect). Fixed
// iteration order (owned sorted; active[] ascending instanceId) -> deterministic.
inline TagContainer EffectiveTags(const TaggedWorld& tw, const TagRules& rules, EntityId id) {
    TagContainer eff;
    const int ti = FindTags(tw, id);
    if (ti >= 0) eff = tw.tags[(size_t)ti].owned;
    const int gi = gas::FindEntity(tw.gas, id);
    if (gi >= 0) {
        for (const gas::ActiveEffect& ae : tw.gas.entities[(size_t)gi].active) {
            const EffectTags* et = rules.FindEffect(ae.def.effectId);
            if (et) eff.Union(et->grantsTags);
        }
    }
    return eff;
}

// ----- TaggedResult: the outcome enum (FIXED numbering — 0..5 mirror gas::ActivateResult) -------------------
enum TaggedResult : uint32_t {
    kTagOk                 = 0,
    kTagUnknownAbility     = 1,
    kTagUnknownCaster      = 2,
    kTagUnknownTarget      = 3,
    kTagOnCooldown         = 4,
    kTagUnaffordable       = 5,
    kTagMissingRequiredTag = 6,   // caster's effective set lacks a requiredTag (query) — hard fail
    kTagBlockedByTag       = 7,   // caster's effective set has a blockedTag (query) — hard fail
    kTagImmune             = 8,   // COMMITTED: activated but >=1 effect nullified by receiver immunity
};

// TryActivateTagged: the tag-aware activation adapter. Pinned gate order (see the header contract). Returns
// the outcome; *outImmuneCount (optional) receives how many effects were nullified by receiver immunity.
inline TaggedResult TryActivateTagged(TaggedWorld& tw, const gas::AbilityKit& kit, const TagRules& rules,
                                      const TagRegistry& reg, EntityId caster, uint32_t abilityId,
                                      EntityId target, uint32_t* outImmuneCount = nullptr) {
    if (outImmuneCount) *outImmuneCount = 0;
    // (1) the ability (fixed linear lookup).
    const gas::AbilityDef* ab = nullptr;
    for (const gas::AbilityDef& a : kit.abilities) if (a.abilityId == abilityId) { ab = &a; break; }
    if (!ab) return kTagUnknownAbility;
    // (2) the caster.
    if (gas::FindEntity(tw.gas, caster) < 0) return kTagUnknownCaster;
    // (3)/(4) the TAG GATES on the caster's EFFECTIVE set (a failure mutates nothing).
    const TagContainer casterEff = EffectiveTags(tw, rules, caster);
    const AbilityTags* at = rules.FindAbility(abilityId);
    if (at) {
        if (HasAnyQuery(reg, casterEff, at->blockedTags))   return kTagBlockedByTag;
        if (!HasAllQuery(reg, casterEff, at->requiredTags)) return kTagMissingRequiredTag;
    }
    // IMMUNITY: build a FILTERED single-ability kit (remove effects the RECEIVER is immune to) so gas reuses
    // its cost/cooldown/apply logic verbatim while immune effects are nullified.
    uint32_t immuneCount = 0;
    gas::AbilityDef filtered = *ab;
    filtered.effects.clear();
    for (const gas::EffectDef& e : ab->effects) {
        const EntityId receiver = (e.targetMode == gas::kTargetOther) ? target : caster;
        bool immune = false;
        if (gas::FindEntity(tw.gas, receiver) >= 0) {
            const EffectTags* et = rules.FindEffect(e.effectId);
            if (et && !et->immunityTags.empty()) {
                const TagContainer recvEff = EffectiveTags(tw, rules, receiver);
                if (HasAnyQuery(reg, recvEff, et->immunityTags)) immune = true;
            }
        }
        if (immune) ++immuneCount;
        else        filtered.effects.push_back(e);
    }
    gas::AbilityKit tempKit;
    tempKit.abilities.push_back(filtered);
    // Delegate the remaining GAS1 gates + the apply.
    const gas::ActivateResult r = gas::TryActivate(tw.gas, tempKit, caster, abilityId, target);
    switch (r) {
        case gas::kActivateUnknownAbility: return kTagUnknownAbility;   // (defensive — cannot happen)
        case gas::kActivateUnknownCaster:  return kTagUnknownCaster;    // (defensive)
        case gas::kActivateUnknownTarget:  return kTagUnknownTarget;
        case gas::kActivateOnCooldown:     return kTagOnCooldown;
        case gas::kActivateUnaffordable:   return kTagUnaffordable;
        case gas::kActivateOk: break;
    }
    // COMMITTED: union the ability's grantedTags into the caster's OWNED set (persistent — see caveats).
    if (at && !at->grantedTags.empty()) {
        const int ci = FindTags(tw, caster);
        if (ci >= 0) tw.tags[(size_t)ci].owned.Union(at->grantedTags);
    }
    if (outImmuneCount) *outImmuneCount = immuneCount;
    return (immuneCount > 0) ? kTagImmune : kTagOk;
}

// ----- The command bus + the composed tagged tick (commands BEFORE the step) --------------------------------
struct TagCommand {
    uint32_t tick      = 0;
    EntityId caster    = kNoEntity;
    uint32_t abilityId = 0;
    EntityId target    = kNoEntity;
};

inline uint32_t ApplyTaggedCommands(TaggedWorld& tw, const gas::AbilityKit& kit, const TagRules& rules,
                                    const TagRegistry& reg, const std::vector<TagCommand>& cmds, uint32_t tick,
                                    std::vector<TaggedResult>* outResults = nullptr) {
    uint32_t ok = 0;
    for (const TagCommand& c : cmds) {
        if (c.tick != tick) continue;
        const TaggedResult r = TryActivateTagged(tw, kit, rules, reg, c.caster, c.abilityId, c.target);
        if (r == kTagOk || r == kTagImmune) ++ok;   // a committed activation (immune still committed)
        if (outResults) outResults->push_back(r);
    }
    return ok;
}

inline uint32_t StepTagged(TaggedWorld& tw, const gas::AbilityKit& kit, const TagRules& rules,
                           const TagRegistry& reg, const std::vector<TagCommand>& cmds, uint32_t tick) {
    ApplyTaggedCommands(tw, kit, rules, reg, cmds, tick);
    return gas::StepAbilities(tw.gas);
}

// ----- Equality + digest (the lockstep/rollback currency; owned tags are the added state) -------------------
inline bool TaggedStatesEqual(const TaggedWorld& a, const TaggedWorld& b) {
    if (!gas::GasStatesEqual(a.gas, b.gas)) return false;
    if (a.tags.size() != b.tags.size()) return false;
    for (size_t i = 0; i < a.tags.size(); ++i) {
        if (a.tags[i].id != b.tags[i].id) return false;
        if (a.tags[i].owned.ids != b.tags[i].owned.ids) return false;
    }
    return true;
}

inline uint64_t DigestTaggedWorld(const TaggedWorld& tw) {
    verdict::DigestFnv d;
    const uint64_t g = gas::DigestGasWorld(tw.gas);
    d.mix32((uint32_t)(g & 0xFFFFFFFFull));
    d.mix32((uint32_t)(g >> 32));
    d.sep();
    d.mix32((uint32_t)tw.tags.size());
    for (const EntityTags& e : tw.tags) {
        d.mix32((uint32_t)e.id);
        d.mix32((uint32_t)e.owned.ids.size());
        for (TagId t : e.owned.ids) d.mix32(t);
        d.sep();
    }
    return d.h;
}

inline std::string DigestHex(uint64_t h) { return gas::DigestHex(h); }

// ===== LOCKSTEP + ROLLBACK (the GAS1 command+snapshot mold; TaggedWorld is COPYABLE — a copy IS the
// snapshot) ==================================================================================================
inline TaggedWorld RunTaggedLockstep(const TaggedWorld& world0, const gas::AbilityKit& kit,
                                     const TagRules& rules, const TagRegistry& reg,
                                     const std::vector<TagCommand>& commands, uint32_t ticks,
                                     bool* outIdentical = nullptr) {
    TaggedWorld authority = world0;
    TaggedWorld replica   = world0;
    for (uint32_t t = 0; t < ticks; ++t) {
        StepTagged(authority, kit, rules, reg, commands, t);
        StepTagged(replica,   kit, rules, reg, commands, t);
    }
    if (outIdentical) *outIdentical = TaggedStatesEqual(authority, replica);
    return authority;
}

inline TaggedWorld RunTaggedRollback(const TaggedWorld& world0, const gas::AbilityKit& kit,
                                     const TagRules& rules, const TagRegistry& reg,
                                     const std::vector<TagCommand>& authStream,
                                     const std::vector<TagCommand>& mispredictStream,
                                     uint32_t ticks, uint32_t rollbackAt,
                                     bool* outCorrectedEqAuthority = nullptr,
                                     bool* outMispredictDiverged = nullptr) {
    TaggedWorld w = world0;
    for (uint32_t t = 0; t < rollbackAt; ++t) StepTagged(w, kit, rules, reg, authStream, t);
    const TaggedWorld snap = w;                                  // the complete restore point (a copy)
    uint32_t specTicks = ticks - rollbackAt;
    if (specTicks > 3u) specTicks = 3u;
    for (uint32_t s = 0; s < specTicks; ++s)
        StepTagged(w, kit, rules, reg, mispredictStream, rollbackAt + s);   // the misprediction
    const TaggedWorld specWorld = w;
    w = snap;                                                    // ROLLBACK
    for (uint32_t t = rollbackAt; t < ticks; ++t) StepTagged(w, kit, rules, reg, authStream, t);

    if (outCorrectedEqAuthority || outMispredictDiverged) {
        TaggedWorld authAtSpec = world0;
        for (uint32_t t = 0; t < rollbackAt + specTicks; ++t) StepTagged(authAtSpec, kit, rules, reg, authStream, t);
        if (outMispredictDiverged) *outMispredictDiverged = !TaggedStatesEqual(specWorld, authAtSpec);
        if (outCorrectedEqAuthority) {
            const TaggedWorld authFinal = RunTaggedLockstep(world0, kit, rules, reg, authStream, ticks);
            *outCorrectedEqAuthority = TaggedStatesEqual(w, authFinal);
        }
    }
    return w;
}

// ===== THE SHOWCASE FIXTURE — THE TAG SKIRMISH (FIXED forever; the test pins the registry/rules digests, the
// per-tick tag-state + attribute trace digest, and the exact finals; the --gt1-tags-shot showcases run THIS
// EXACT scenario on both backends — byte-identical viz BY CONSTRUCTION) ======================================

// The pinned ability ids (the GT skirmish kit — a GT1-local kit, GAS1's MakeCoreKit stays untouched).
inline constexpr uint32_t kAbStun    = 1u;   // grants State.Stunned to the target for 3 ticks
inline constexpr uint32_t kAbFire    = 2u;   // fire damage + burn to the target (nullified by Immune.Fire)
inline constexpr uint32_t kAbEmpower = 3u;   // grants State.Empowered to self for 4 ticks
inline constexpr uint32_t kAbSmite   = 4u;   // BIG damage — REQUIRES State.Empowered

inline constexpr uint32_t kEffStunAura = 10u; // grants State.Stunned
inline constexpr uint32_t kEffFireHit  = 11u; // instant -20 health (immunity Immune.Fire)
inline constexpr uint32_t kEffFireBurn = 12u; // -2/tick DoT (immunity Immune.Fire; grants State.Burning)
inline constexpr uint32_t kEffEmpower  = 13u; // grants State.Empowered
inline constexpr uint32_t kEffSmiteHit = 14u; // instant -30 health

// MakeTagRegistry: the FIXED authored tag table (ids are pinned by the authoring order). See the pinned ids
// in the accessors below.
inline TagRegistry MakeTagRegistry() {
    TagRegistry r;
    r.Intern("State.Stunned");          // "State"(0), "State.Stunned"(1)
    r.Intern("State.Empowered");        // "State.Empowered"(2)
    r.Intern("State.Burning");          // "State.Burning"(3)
    r.Intern("Immune.Fire");            // "Immune"(4), "Immune.Fire"(5)
    r.Intern("Ability.Fire.Fireball");  // "Ability"(6), "Ability.Fire"(7), "Ability.Fire.Fireball"(8)
    return r;
}
// The pinned ids (idempotent Intern of the canonical registry).
inline TagId TagStunned(const TagRegistry& r)   { return r.Find("State.Stunned"); }    // 1
inline TagId TagEmpowered(const TagRegistry& r) { return r.Find("State.Empowered"); }  // 2
inline TagId TagBurning(const TagRegistry& r)   { return r.Find("State.Burning"); }    // 3
inline TagId TagImmuneFire(const TagRegistry& r){ return r.Find("Immune.Fire"); }      // 5

// MakeSkirmishKit: the GAS1 ability kit (cost 0 / cd 0 to isolate the TAG behaviour; the tag rules carry the
// gating). Authored through the KitBuilder (the "authored, not hardcoded" discipline).
inline gas::AbilityKit MakeSkirmishKit() {
    gas::KitBuilder b;
    // Stun: a 3-tick aura effect on the target (attr is harmless — armor +0 — its PRESENCE grants the tag).
    b.Ability(kAbStun, gas::kAttrMana, 0, 0u)
     .Effect(kEffStunAura, gas::kAttrArmor, gas::kOpAdd, 0, gas::kDurTicks, 3u, gas::kStackRefresh, 1u, 0u, gas::kTargetOther);
    // Fire: instant -20 health + a -2/tick burn (3 ticks), both on the target.
    b.Ability(kAbFire, gas::kAttrMana, 0, 0u)
     .Effect(kEffFireHit,  gas::kAttrHealth, gas::kOpAdd, -20 * kOne, gas::kDurInstant, 0u, gas::kStackIgnore,  1u, 0u, gas::kTargetOther)
     .Effect(kEffFireBurn, gas::kAttrHealth, gas::kOpAdd, -2 * kOne,  gas::kDurTicks,   3u, gas::kStackRefresh, 1u, 1u, gas::kTargetOther);
    // Empower: a 4-tick self aura (grants State.Empowered).
    b.Ability(kAbEmpower, gas::kAttrMana, 0, 0u)
     .Effect(kEffEmpower, gas::kAttrArmor, gas::kOpAdd, 0, gas::kDurTicks, 4u, gas::kStackRefresh, 1u, 0u, gas::kTargetSelf);
    // Smite: instant -30 health on the target — REQUIRES State.Empowered (the tag rules).
    b.Ability(kAbSmite, gas::kAttrMana, 0, 0u)
     .Effect(kEffSmiteHit, gas::kAttrHealth, gas::kOpAdd, -30 * kOne, gas::kDurInstant, 0u, gas::kStackIgnore, 1u, 0u, gas::kTargetOther);
    return b.Build();
}

// MakeSkirmishRules: the tag GATES (all four mechanics). blockedTags State.Stunned on every ability;
// requiredTags State.Empowered on Smite; immunity Immune.Fire on both fire effects; grants on the auras.
inline TagRules MakeSkirmishRules(const TagRegistry& reg) {
    const TagId stunned   = TagStunned(reg);
    const TagId empowered = TagEmpowered(reg);
    const TagId burning   = TagBurning(reg);
    const TagId immuneF   = TagImmuneFire(reg);
    TagRules rules;
    auto stun = [&](uint32_t id) { AbilityTags a; a.abilityId = id; a.blockedTags.Add(stunned); return a; };
    rules.abilities.push_back(stun(kAbStun));
    rules.abilities.push_back(stun(kAbFire));
    rules.abilities.push_back(stun(kAbEmpower));
    { AbilityTags a; a.abilityId = kAbSmite; a.blockedTags.Add(stunned); a.requiredTags.Add(empowered);
      rules.abilities.push_back(a); }
    { EffectTags e; e.effectId = kEffStunAura; e.grantsTags.Add(stunned);   rules.effects.push_back(e); }
    { EffectTags e; e.effectId = kEffFireHit;  e.immunityTags.Add(immuneF); rules.effects.push_back(e); }
    { EffectTags e; e.effectId = kEffFireBurn; e.immunityTags.Add(immuneF); e.grantsTags.Add(burning);
      rules.effects.push_back(e); }
    { EffectTags e; e.effectId = kEffEmpower;  e.grantsTags.Add(empowered); rules.effects.push_back(e); }
    return rules;
}

// The skirmish base attributes (both duelists): health 100/100, mana 100/100, moveSpeed 4, ap 10, armor 5.
inline constexpr fx kSkirmishBases[gas::kAttrCount] = {
    100 * kOne, 100 * kOne, 100 * kOne, 100 * kOne, 4 * kOne, 10 * kOne, 5 * kOne, 0,
};
inline constexpr uint32_t kSkirmishTicks = 14u;

// MakeSkirmishWorld: Hero = id 1 (no innate tags), Villain = id 2 (innate Immune.Fire).
inline TaggedWorld MakeSkirmishWorld(const TagRegistry& reg) {
    TaggedWorld tw;
    TagContainer none;
    SpawnTagged(tw, kSkirmishBases, none);                 // Hero = 1
    TagContainer villainInnate; villainInnate.Add(TagImmuneFire(reg));
    SpawnTagged(tw, kSkirmishBases, villainInnate);        // Villain = 2 (Immune.Fire)
    return tw;
}

// MakeSkirmishStream: THE FIXED 14-tick script exercising ALL four tag mechanics + the precedence rule.
inline std::vector<TagCommand> MakeSkirmishStream() {
    const EntityId H = 1u, V = 2u;
    return {
        TagCommand{0u, V, kAbStun,    H},   // ok: Villain stuns Hero (State.Stunned on Hero, 3 ticks)
        TagCommand{1u, H, kAbFire,    V},   // FAIL kTagBlockedByTag (Hero stunned)
        TagCommand{2u, H, kAbSmite,   V},   // FAIL kTagBlockedByTag (blocked BEFORE missing-required — precedence)
        TagCommand{3u, H, kAbFire,    V},   // kTagImmune: Hero free now; Villain Immune.Fire nullifies BOTH fire effects
        TagCommand{4u, H, kAbSmite,   V},   // FAIL kTagMissingRequiredTag (Hero not Empowered)
        TagCommand{5u, H, kAbEmpower, H},   // ok: Hero self-grants State.Empowered (4 ticks)
        TagCommand{6u, H, kAbSmite,   V},   // ok: Empowered -> Smite lands (-30 -> Villain 70)
        TagCommand{7u, V, kAbFire,    H},   // ok: Hero NOT immune -> -20 + burn (grants State.Burning)
    };
}

// ----- SkirmishTickSample / SkirmishRun / RunSkirmish: the shared per-tick trace (test + BOTH showcases) ----
struct SkirmishTickSample {
    fx       health[2];     // current health of Hero, Villain AFTER this tick's step
    fx       mana[2];
    uint8_t  stunned[2];    // effective State.Stunned after the step
    uint8_t  empowered[2];  // effective State.Empowered
    uint8_t  burning[2];    // effective State.Burning
    uint8_t  immune[2];     // effective Immune.Fire (innate — constant here, but sampled)
    uint8_t  castOk[2];     // this tick: a committed activation (ok)
    uint8_t  castImmune[2]; // this tick: committed but immunity nullified an effect
    uint8_t  castBlocked[2];// this tick: kTagBlockedByTag
    uint8_t  castMissReq[2];// this tick: kTagMissingRequiredTag
    uint32_t tagCount[2];   // effective tag count
};
struct SkirmishRun {
    TaggedWorld                      finalWorld;
    std::vector<SkirmishTickSample>  samples;          // kSkirmishTicks entries
    uint64_t                         traceDigest = 0;  // per-tick tag-state + attributes + outcomes (pinned)
    uint32_t                         activationsOk = 0;
    uint32_t                         activationsBlocked = 0;
    uint32_t                         activationsMissReq = 0;
    uint32_t                         activationsImmune = 0;
    uint32_t                         effectsApplied = 0;   // effect applications on committed casts + periodic
};

inline SkirmishRun RunSkirmish() {
    const TagRegistry reg = MakeTagRegistry();
    const gas::AbilityKit kit = MakeSkirmishKit();
    const TagRules rules = MakeSkirmishRules(reg);
    const std::vector<TagCommand> stream = MakeSkirmishStream();
    const TagId tStun = TagStunned(reg), tEmp = TagEmpowered(reg), tBurn = TagBurning(reg),
                tImm = TagImmuneFire(reg);
    SkirmishRun run;
    TaggedWorld tw = MakeSkirmishWorld(reg);
    verdict::DigestFnv trace;
    for (uint32_t t = 0; t < kSkirmishTicks; ++t) {
        SkirmishTickSample s{};
        // Commands BEFORE the step (array order) — record per-caster outcome + effect-application count.
        for (const TagCommand& c : stream) {
            if (c.tick != t) continue;
            uint32_t imm = 0;
            const TaggedResult r = TryActivateTagged(tw, kit, rules, reg, c.caster, c.abilityId, c.target, &imm);
            const int who = (c.caster == 2u) ? 1 : 0;
            if (r == kTagOk || r == kTagImmune) {
                ++run.activationsOk;
                if (r == kTagImmune) { s.castImmune[who] = 1; ++run.activationsImmune; }
                else                   s.castOk[who] = 1;
                // Applied effects = the ability's effects MINUS the immune-nullified ones.
                for (const gas::AbilityDef& a : kit.abilities)
                    if (a.abilityId == c.abilityId) run.effectsApplied += (uint32_t)a.effects.size() - imm;
            } else if (r == kTagBlockedByTag)       { s.castBlocked[who] = 1; ++run.activationsBlocked; }
            else if (r == kTagMissingRequiredTag)   { s.castMissReq[who] = 1; ++run.activationsMissReq; }
            trace.mix32((uint32_t)r);
            trace.mix32(imm);
        }
        trace.sep();
        // The step.
        const uint32_t periodic = gas::StepAbilities(tw.gas);
        run.effectsApplied += periodic;
        // Sample AFTER the step (Hero = entities[0], Villain = entities[1] — spawn order pinned).
        for (int i = 0; i < 2; ++i) {
            const gas::GasEntity& e = tw.gas.entities[(size_t)i];
            const TagContainer eff = EffectiveTags(tw, rules, e.id);
            s.health[i] = e.attrs.current[gas::kAttrHealth];
            s.mana[i]   = e.attrs.current[gas::kAttrMana];
            s.stunned[i]   = HasTagQuery(reg, eff, tStun) ? 1 : 0;
            s.empowered[i] = HasTagQuery(reg, eff, tEmp)  ? 1 : 0;
            s.burning[i]   = HasTagQuery(reg, eff, tBurn) ? 1 : 0;
            s.immune[i]    = HasTagQuery(reg, eff, tImm)  ? 1 : 0;
            s.tagCount[i]  = (uint32_t)eff.size();
            for (TagId tg : eff.ids) trace.mix32(tg);
            trace.mix32((uint32_t)e.attrs.current[gas::kAttrHealth]);
            trace.sep();
        }
        run.samples.push_back(s);
    }
    run.traceDigest = trace.h;
    run.finalWorld  = tw;
    return run;
}

}  // namespace tags
}  // namespace hf::game
