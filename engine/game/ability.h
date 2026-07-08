#pragma once
// Slice GAS1 — A DETERMINISTIC GAMEPLAY ABILITY SYSTEM (attributes + effects + cooldowns — the GAS-class
// core, hf::game::gas). Flagship #1 of the fresh parity++ audit: UE5's Gameplay Ability System is the
// gameplay-framework pillar HF lacked — verdict.h carries only a Health counter + kCmdAbility doing a raw
// hp delta. GAS1 builds the real thing: ATTRIBUTE SETS (a fixed-enum Q16.16 array per entity with the GAS
// BASE-vs-CURRENT split), GAMEPLAY EFFECTS (add/multiply/override modifiers with instant/N-tick/infinite
// duration, stack/refresh/ignore stacking, optional periodic re-application — damage-over-time), ABILITIES
// (cost + cooldown + an effect list, deterministic failure reasons), and the command+snapshot LOCKSTEP mold.
//
// EVERYTHING IS PURE INTEGER (Q16.16 / uint32, NO float, NO wall clock — durations/cooldowns/periods are
// TICKS). UE5's GAS carries float attributes and wall-clock timers and cannot replay deterministically; this
// one is a pure function of (kit, world0, command stream): a peer re-derives every damage number, buff stack,
// and cooldown bit-for-bit from the input stream, and a rollback corrects a mispredicted activation exactly.
//
// THE PINNED SEMANTICS (all order-sensitive rules are FIXED and documented — the determinism contract):
//   * BASE vs CURRENT: base[] is the persistent value (instant effects + periodic firings + costs mutate
//     BASE); current[] is DERIVED every recompute: base folded with the live non-periodic duration effects.
//   * THE FOLD ORDER (RecomputeCurrents): current = base, then ALL kOpAdd effects in ASCENDING effect-
//     instance id (each contributes magnitude * stacks), then ALL kOpMultiply in ascending instance id
//     (current = fxmul(current, magnitude) applied once PER STACK), then kOpOverride in ascending instance
//     id (so the LAST instance — the highest id — wins). Application order does NOT matter; the fold order
//     is pinned. After the fold: current health clamps to [0, current maxHealth], current mana to
//     [0, current maxMana].
//   * INSTANT effects apply the op to BASE once at activation (the GAS instant-GE model). Base health/mana
//     re-clamp after every base mutation ([0, base max]).
//   * PERIODIC effects (periodTicks > 0) do NOT contribute to the current fold; instead every periodTicks
//     steps they fire magnitude * stacks as an ADD to BASE (the GAS periodic-executes-as-instant model; the
//     op field of a periodic effect is ignored — periodic is always an add, documented). periodCountdown
//     starts at periodTicks, so a period-1 duration-3 DoT fires on exactly the 3 steps after application.
//   * STACKING (matched by def.effectId on the same target — the stacking identity key):
//       kStackStack:   stacks = min(stacks + 1, maxStacks) AND remainingTicks resets to durationTicks (a
//                      re-application at the cap still refreshes the duration; magnitude never changes).
//       kStackRefresh: remainingTicks resets to durationTicks; stacks/magnitude UNCHANGED.
//       kStackIgnore:  a re-application while active is a deterministic NO-OP.
//   * TryActivate: unknown-ability / unknown-caster / unknown-target / on-cooldown / unaffordable are
//     DETERMINISTIC failure enums (a failed activation mutates NOTHING). Affordability is checked against
//     CURRENT[costAttr]; the cost is PAID from BASE[costAttr] (the GAS cost convention).
//   * StepAbilities(world) — ONE deterministic tick over entities in ASCENDING entities[] (spawn) order,
//     per entity in this PINNED phase order: (1) cooldowns decrement, (2) periodic effects fire (ascending
//     instance id), (3) durations decrement + expired effects removed (ascending order preserved),
//     (4) RecomputeCurrents. Then ++world.tick (ONCE, world-level).
//   * COMMANDS BEFORE STEP (the verdict/convex contract): ApplyGasCommands applies this tick's activation
//     requests in FIXED array order BEFORE StepAbilities.
//
// THE AUTHORING PROOF (the PA1 pattern, builder-API variant): the ability KIT (Fireball / Shield / Haste)
// is authored EXCLUSIVELY through the KitBuilder fluent API — no hand-rolled struct literals — and the
// kit's canonical serialization digest is PINNED by the test (any drift in the authoring calls or the
// builder is caught). GAS1 deliberately uses the builder-API option (effects are DATA, not per-tick
// programs); binding flow-graph outputs to effect magnitudes is a future authoring slice.
//
// HONEST SCOPE / CAVEATS: no death/despawn rule (health clamps at 0 and the entity keeps acting); armor is
// a buffable attribute but there is NO damage-mitigation formula (damage is a direct health change);
// periodic effects are always base ADDs; kAttrHealthRegen is carried data-only (no regen system in GAS1);
// GasWorld entities are GAS1-local (id-compatible with verdict::EntityId but not bound to a VerdictWorld —
// the ECS bridge is a future slice).
//
// PURE CPU INTEGER (the strictest determinism tier). Header-only, namespace hf::game::gas. #includes
// game/verdict.h READ-ONLY (the EntityId vocabulary + the DigestFnv digest discipline + the fpx Q16.16
// toolbox — verdict.h is NEVER modified). NO new render RHI, NO new shader, NO new compute.

#include <cstdint>
#include <cstdio>    // std::snprintf (the 16-hex digest string)
#include <vector>

#include "game/verdict.h"   // read-only: EntityId/kNoEntity + verdict::DigestFnv + fpx aliases (NOT modified)

namespace hf::game {
namespace gas {

// Reuse the layer's vocabulary verbatim (NO new fixed-point primitives).
namespace fpx = hf::sim::fpx;
using fpx::fx;
using fpx::kOne;
using fpx::kFrac;
using fpx::fxmul;
using verdict::EntityId;
using verdict::kNoEntity;

// ----- The attribute set: a FIXED enum (the wire/asset contract — numbering never changes) -----------------
// ~8 gameplay attributes, all Q16.16. kAttrHealth/kAttrMana are CLAMPED against their max twins (base and
// current independently); the rest are free-range. kAttrHealthRegen is data-only in GAS1 (no regen system).
enum Attr : uint32_t {
    kAttrHealth      = 0,   // clamped to [0, maxHealth]
    kAttrMaxHealth   = 1,
    kAttrMana        = 2,   // clamped to [0, maxMana]
    kAttrMaxMana     = 3,
    kAttrMoveSpeed   = 4,
    kAttrAttackPower = 5,
    kAttrArmor       = 6,
    kAttrHealthRegen = 7,   // carried data-only (no regen system in GAS1 — honest scope)
};
inline constexpr uint32_t kAttrCount = 8u;

// ----- AttributeSet: the GAS base-vs-current split (both Q16.16) -------------------------------------------
// base[] is the persistent value (instant effects / periodic firings / costs mutate it); current[] is the
// DERIVED value (base folded with the live duration modifiers — RecomputeCurrents). Zero-initialized.
struct AttributeSet {
    fx base[kAttrCount]    = {0, 0, 0, 0, 0, 0, 0, 0};
    fx current[kAttrCount] = {0, 0, 0, 0, 0, 0, 0, 0};
};

// ----- The effect vocabulary (FIXED numbering — the wire/asset contract) -----------------------------------
inline constexpr uint32_t kOpAdd      = 0u;   // current/base += magnitude (per stack)
inline constexpr uint32_t kOpMultiply = 1u;   // current/base = fxmul(current/base, magnitude) (per stack)
inline constexpr uint32_t kOpOverride = 2u;   // current/base = magnitude (last instance wins in the fold)

inline constexpr uint32_t kDurInstant  = 0u;  // applied ONCE to BASE at activation (never an ActiveEffect)
inline constexpr uint32_t kDurTicks    = 1u;  // lives durationTicks steps as an ActiveEffect
inline constexpr uint32_t kDurInfinite = 2u;  // lives until (a future) explicit removal — never expires

inline constexpr uint32_t kStackStack   = 0u; // stacks+1 up to maxStacks; duration refreshes (even at cap)
inline constexpr uint32_t kStackRefresh = 1u; // duration refreshes; stacks/magnitude unchanged
inline constexpr uint32_t kStackIgnore  = 2u; // re-application while active is a deterministic no-op

inline constexpr uint32_t kTargetSelf  = 0u;  // the effect applies to the CASTER
inline constexpr uint32_t kTargetOther = 1u;  // the effect applies to the activation TARGET

// ----- EffectDef: ONE gameplay-effect definition (all-integer POD — the authored asset unit) ---------------
// effectId is the STACKING IDENTITY (unique within a kit): a re-application matches an ActiveEffect on the
// same target with the same effectId. periodTicks > 0 makes the effect PERIODIC (fires magnitude * stacks
// as a base ADD every periodTicks steps; it then does NOT contribute to the current fold — the pinned
// periodic model above).
struct EffectDef {
    uint32_t effectId      = 0;           // the stacking identity key (unique within the kit)
    uint32_t attr          = kAttrHealth; // the target attribute (Attr)
    uint32_t op            = kOpAdd;      // kOpAdd / kOpMultiply / kOpOverride (ignored when periodic)
    fx       magnitude     = 0;           // Q16.16
    uint32_t durKind       = kDurInstant; // kDurInstant / kDurTicks / kDurInfinite
    uint32_t durationTicks = 0;           // for kDurTicks
    uint32_t stackPolicy   = kStackIgnore;// kStackStack / kStackRefresh / kStackIgnore
    uint32_t maxStacks     = 1;           // for kStackStack (>= 1)
    uint32_t periodTicks   = 0;           // 0 = none; else the periodic base-ADD cadence (the DoT)
    uint32_t targetMode    = kTargetOther;// kTargetSelf / kTargetOther
};

// ----- AbilityDef / AbilityKit: {id, cost, cooldown, effects} — the authored ability asset ------------------
struct AbilityDef {
    uint32_t abilityId     = 0;           // unique within the kit, != 0
    uint32_t costAttr      = kAttrMana;   // the attribute the cost is paid from (checked vs CURRENT, paid from BASE)
    fx       costAmount    = 0;           // Q16.16 (>= 0)
    uint32_t cooldownTicks = 0;           // ticks until re-activation (0 = none)
    std::vector<EffectDef> effects;       // applied in ARRAY ORDER on successful activation
};

struct AbilityKit {
    std::vector<AbilityDef> abilities;    // looked up by abilityId (linear, fixed order)
};

// ----- KitBuilder: the fluent authoring API (THE PA1 "authored, not hardcoded" seam) ------------------------
// The kit is constructed EXCLUSIVELY through Ability()/Effect() calls; the test pins DigestKit of the
// result, so any drift in the authoring calls (or the builder) is caught. Effect() appends to the MOST
// RECENT Ability() (an Effect() before any Ability() is a deterministic no-op — the "no edge" discipline).
struct KitBuilder {
    AbilityKit kit;

    KitBuilder& Ability(uint32_t abilityId, uint32_t costAttr, fx costAmount, uint32_t cooldownTicks) {
        AbilityDef a;
        a.abilityId     = abilityId;
        a.costAttr      = costAttr;
        a.costAmount    = costAmount;
        a.cooldownTicks = cooldownTicks;
        kit.abilities.push_back(a);
        return *this;
    }
    KitBuilder& Effect(uint32_t effectId, uint32_t attr, uint32_t op, fx magnitude, uint32_t durKind,
                       uint32_t durationTicks, uint32_t stackPolicy, uint32_t maxStacks,
                       uint32_t periodTicks, uint32_t targetMode) {
        if (kit.abilities.empty()) return *this;   // no ability begun -> deterministic no-op
        EffectDef e;
        e.effectId      = effectId;
        e.attr          = attr;
        e.op            = op;
        e.magnitude     = magnitude;
        e.durKind       = durKind;
        e.durationTicks = durationTicks;
        e.stackPolicy   = stackPolicy;
        e.maxStacks     = maxStacks;
        e.periodTicks   = periodTicks;
        e.targetMode    = targetMode;
        kit.abilities.back().effects.push_back(e);
        return *this;
    }
    AbilityKit Build() { return kit; }
};

// ----- DigestKit: the pinned "authored, not hardcoded" fingerprint (verdict::DigestFnv discipline) ---------
// A canonical field-order fold over every ability + effect (a separator after each ability so the pools
// cannot alias). Same 64-bit FNV-1a constants as the engine's other digest sites (via verdict::DigestFnv).
inline uint64_t DigestKit(const AbilityKit& kit) {
    verdict::DigestFnv d;
    d.mix32((uint32_t)kit.abilities.size());
    for (const AbilityDef& a : kit.abilities) {
        d.mix32(a.abilityId);
        d.mix32(a.costAttr);
        d.mix32((uint32_t)a.costAmount);
        d.mix32(a.cooldownTicks);
        d.mix32((uint32_t)a.effects.size());
        for (const EffectDef& e : a.effects) {
            d.mix32(e.effectId);
            d.mix32(e.attr);
            d.mix32(e.op);
            d.mix32((uint32_t)e.magnitude);
            d.mix32(e.durKind);
            d.mix32(e.durationTicks);
            d.mix32(e.stackPolicy);
            d.mix32(e.maxStacks);
            d.mix32(e.periodTicks);
            d.mix32(e.targetMode);
        }
        d.sep();
    }
    return d.h;
}

// ----- ActiveEffect: one LIVE duration effect on an entity ---------------------------------------------------
// instanceId is a WORLD-monotonic allocation (nextInstanceId++, NEVER recycled) — THE fold-order key. The
// def is COPIED in (the snapshot is self-contained; the kit stays a static asset). The active vector is in
// ascending-instanceId order BY CONSTRUCTION (monotonic ids, push_back, order-preserving erase).
struct ActiveEffect {
    uint32_t  instanceId      = 0;   // world-monotonic (the pinned fold-order key)
    EffectDef def;                   // the definition, copied at application
    uint32_t  remainingTicks  = 0;   // for kDurTicks (unused for kDurInfinite)
    uint32_t  stacks          = 1;
    uint32_t  periodCountdown = 0;   // for periodic effects (starts at periodTicks)
};

// ----- Cooldown: one ability's per-entity cooldown counter ---------------------------------------------------
// Entries are appended on FIRST activation of that ability by that entity (fixed order thereafter);
// remaining == 0 means ready. Decremented once per StepAbilities.
struct Cooldown {
    uint32_t abilityId = 0;
    uint32_t remaining = 0;
};

// ----- GasEntity / GasWorld: the deterministic ability world -------------------------------------------------
// Entities live in entities[] in SPAWN ORDER (ascending id by construction — the pinned iteration order,
// the verdict order[] convention). Ids are monotonic from 1 and NEVER recycled. GasWorld is plain-vector
// COPYABLE — a copy IS a snapshot (the lockstep/rollback harnesses use value copies).
struct GasEntity {
    EntityId                  id = kNoEntity;
    AttributeSet              attrs;
    std::vector<ActiveEffect> active;      // ascending instanceId by construction
    std::vector<Cooldown>     cooldowns;   // appended on first use, fixed order
};

struct GasWorld {
    std::vector<GasEntity> entities;       // spawn order (ascending id) — the pinned iteration sequence
    EntityId               nextEntityId   = 1u;   // monotonic, NEVER recycled (the verdict convention)
    uint32_t               nextInstanceId = 1u;   // monotonic effect-instance allocator (the fold-order key)
    uint32_t               tick           = 0;    // the world clock
};

// ----- FindEntity(world, id) -> index or -1 (fixed linear scan — deterministic, no hash order) --------------
inline int FindEntity(const GasWorld& world, EntityId id) {
    for (size_t i = 0; i < world.entities.size(); ++i)
        if (world.entities[i].id == id) return (int)i;
    return -1;
}

// ----- ClampBases / RecomputeCurrents: the derived-attribute machinery --------------------------------------
// ClampBases: base health to [0, base maxHealth], base mana to [0, base maxMana] — after EVERY base
// mutation (instant effects, periodic firings, costs). Pure integer clamps.
inline void ClampBases(AttributeSet& a) {
    if (a.base[kAttrHealth] < 0) a.base[kAttrHealth] = 0;
    if (a.base[kAttrHealth] > a.base[kAttrMaxHealth]) a.base[kAttrHealth] = a.base[kAttrMaxHealth];
    if (a.base[kAttrMana] < 0) a.base[kAttrMana] = 0;
    if (a.base[kAttrMana] > a.base[kAttrMaxMana]) a.base[kAttrMana] = a.base[kAttrMaxMana];
}

// RecomputeCurrents: THE PINNED FOLD (the header contract): current = base; ALL adds ascending instance id
// (magnitude * stacks, an int64 product narrowed back — stacks is tiny); ALL multiplies ascending instance
// id (fxmul once PER STACK); overrides ascending (LAST instance wins). Periodic effects are EXCLUDED from
// the fold (they fire as base adds in StepAbilities). Then the current health/mana clamps. Pure integer.
inline void RecomputeCurrents(GasEntity& e) {
    for (uint32_t a = 0; a < kAttrCount; ++a) e.attrs.current[a] = e.attrs.base[a];
    // Pass 1: ALL kOpAdd, ascending instanceId (the active vector order).
    for (const ActiveEffect& fx_ : e.active) {
        if (fx_.def.periodTicks != 0) continue;                       // periodic -> not in the fold
        if (fx_.def.op != kOpAdd || fx_.def.attr >= kAttrCount) continue;
        e.attrs.current[fx_.def.attr] =
            (fx)((int64_t)e.attrs.current[fx_.def.attr] + (int64_t)fx_.def.magnitude * (int64_t)fx_.stacks);
    }
    // Pass 2: ALL kOpMultiply, ascending instanceId, fxmul once per stack.
    for (const ActiveEffect& fx_ : e.active) {
        if (fx_.def.periodTicks != 0) continue;
        if (fx_.def.op != kOpMultiply || fx_.def.attr >= kAttrCount) continue;
        for (uint32_t s = 0; s < fx_.stacks; ++s)
            e.attrs.current[fx_.def.attr] = fxmul(e.attrs.current[fx_.def.attr], fx_.def.magnitude);
    }
    // Pass 3: kOpOverride, ascending instanceId -> the LAST (highest-id) instance wins.
    for (const ActiveEffect& fx_ : e.active) {
        if (fx_.def.periodTicks != 0) continue;
        if (fx_.def.op != kOpOverride || fx_.def.attr >= kAttrCount) continue;
        e.attrs.current[fx_.def.attr] = fx_.def.magnitude;
    }
    // The current clamps (against the FOLDED current maxima).
    if (e.attrs.current[kAttrHealth] < 0) e.attrs.current[kAttrHealth] = 0;
    if (e.attrs.current[kAttrHealth] > e.attrs.current[kAttrMaxHealth])
        e.attrs.current[kAttrHealth] = e.attrs.current[kAttrMaxHealth];
    if (e.attrs.current[kAttrMana] < 0) e.attrs.current[kAttrMana] = 0;
    if (e.attrs.current[kAttrMana] > e.attrs.current[kAttrMaxMana])
        e.attrs.current[kAttrMana] = e.attrs.current[kAttrMaxMana];
}

// ----- SpawnGasEntity(world, bases) -> EntityId: deterministic monotonic spawn ------------------------------
// Allocates nextEntityId++ (NEVER recycled), appends to entities[] (spawn order = ascending id), seeds the
// base attributes, clamps, and derives currents (== bases at zero effects — the identity).
inline EntityId SpawnGasEntity(GasWorld& world, const fx (&bases)[kAttrCount]) {
    GasEntity e;
    e.id = world.nextEntityId++;
    for (uint32_t a = 0; a < kAttrCount; ++a) e.attrs.base[a] = bases[a];
    ClampBases(e.attrs);
    RecomputeCurrents(e);
    world.entities.push_back(e);
    return e.id;
}

// ----- ApplyInstantToBase: one instant op on a BASE attribute (+ the base clamps) ---------------------------
inline void ApplyInstantToBase(GasEntity& e, uint32_t attr, uint32_t op, fx magnitude) {
    if (attr >= kAttrCount) return;                          // out-of-range -> deterministic no-op
    if (op == kOpAdd)           e.attrs.base[attr] = (fx)((int64_t)e.attrs.base[attr] + (int64_t)magnitude);
    else if (op == kOpMultiply) e.attrs.base[attr] = fxmul(e.attrs.base[attr], magnitude);
    else if (op == kOpOverride) e.attrs.base[attr] = magnitude;
    ClampBases(e.attrs);
}

// ----- ActivateResult: the DETERMINISTIC failure reasons (FIXED numbering — the proofs pin these) ------------
enum ActivateResult : uint32_t {
    kActivateOk             = 0,
    kActivateUnknownAbility = 1,   // abilityId not in the kit
    kActivateUnknownCaster  = 2,   // caster id not a live entity
    kActivateUnknownTarget  = 3,   // an effect targets kTargetOther but the target id is not live
    kActivateOnCooldown     = 4,   // the caster's cooldown counter for this ability is > 0
    kActivateUnaffordable   = 5,   // current[costAttr] < costAmount
};

// ----- ApplyEffectTo: apply ONE EffectDef to a target entity (instant vs duration + the stacking rules) -----
// Instant -> ApplyInstantToBase. Duration/infinite -> match an ActiveEffect by def.effectId (the stacking
// identity): kStackStack bumps stacks to the cap AND refreshes duration; kStackRefresh refreshes duration
// only; kStackIgnore no-ops. No match -> a NEW instance (instanceId = world.nextInstanceId++, stacks 1,
// periodCountdown = periodTicks). Currents recomputed after every mutation.
inline void ApplyEffectTo(GasWorld& world, GasEntity& target, const EffectDef& def) {
    if (def.durKind == kDurInstant) {
        ApplyInstantToBase(target, def.attr, def.op, def.magnitude);
        RecomputeCurrents(target);
        return;
    }
    // A duration/infinite effect: stacking match by effectId (fixed order scan).
    for (ActiveEffect& ae : target.active) {
        if (ae.def.effectId != def.effectId) continue;
        if (def.stackPolicy == kStackStack) {
            if (ae.stacks < def.maxStacks) ++ae.stacks;      // capped at maxStacks (pinned)
            ae.remainingTicks = def.durationTicks;           // stacking refreshes duration (even at cap)
        } else if (def.stackPolicy == kStackRefresh) {
            ae.remainingTicks = def.durationTicks;           // duration only; stacks/magnitude unchanged
        }
        // kStackIgnore: deterministic no-op.
        RecomputeCurrents(target);
        return;
    }
    ActiveEffect ae;
    ae.instanceId      = world.nextInstanceId++;             // monotonic — the fold-order key
    ae.def             = def;
    ae.remainingTicks  = def.durationTicks;
    ae.stacks          = 1;
    ae.periodCountdown = def.periodTicks;
    target.active.push_back(ae);                              // ascending instanceId by construction
    RecomputeCurrents(target);
}

// ----- TryActivate(world, kit, caster, abilityId, target) -> ActivateResult ---------------------------------
// THE GAS activation gate, in PINNED check order: unknown-ability -> unknown-caster -> unknown-target (only
// if some effect targets kTargetOther) -> on-cooldown -> unaffordable. A failure mutates NOTHING. On
// success: the cost is paid from BASE[costAttr] (affordability was checked vs CURRENT), the cooldown
// counter starts at cooldownTicks, and the effects apply in ARRAY ORDER (self/other per targetMode).
inline ActivateResult TryActivate(GasWorld& world, const AbilityKit& kit, EntityId caster,
                                  uint32_t abilityId, EntityId target) {
    // (1) the ability (fixed linear lookup).
    const AbilityDef* ab = nullptr;
    for (const AbilityDef& a : kit.abilities)
        if (a.abilityId == abilityId) { ab = &a; break; }
    if (!ab) return kActivateUnknownAbility;
    // (2) the caster.
    const int ci = FindEntity(world, caster);
    if (ci < 0) return kActivateUnknownCaster;
    // (3) the target — required only when some effect targets kTargetOther.
    bool needsTarget = false;
    for (const EffectDef& e : ab->effects)
        if (e.targetMode == kTargetOther) needsTarget = true;
    int ti = -1;
    if (needsTarget) {
        ti = FindEntity(world, target);
        if (ti < 0) return kActivateUnknownTarget;
    }
    // (4) the cooldown gate.
    GasEntity& c = world.entities[(size_t)ci];
    Cooldown* cd = nullptr;
    for (Cooldown& k : c.cooldowns)
        if (k.abilityId == abilityId) { cd = &k; break; }
    if (cd && cd->remaining > 0) return kActivateOnCooldown;
    // (5) the cost gate (checked vs CURRENT).
    if (ab->costAttr < kAttrCount && c.attrs.current[ab->costAttr] < ab->costAmount)
        return kActivateUnaffordable;

    // COMMIT: pay the cost from BASE, start the cooldown, apply the effects in array order.
    if (ab->costAttr < kAttrCount && ab->costAmount != 0) {
        c.attrs.base[ab->costAttr] = (fx)((int64_t)c.attrs.base[ab->costAttr] - (int64_t)ab->costAmount);
        ClampBases(c.attrs);
        RecomputeCurrents(c);
    }
    if (cd) cd->remaining = ab->cooldownTicks;
    else    c.cooldowns.push_back(Cooldown{abilityId, ab->cooldownTicks});
    for (const EffectDef& e : ab->effects) {
        // Re-resolve indices each iteration (an instant effect cannot move entities, but the discipline is
        // cheap and future-proof against vector churn).
        const int si = FindEntity(world, caster);
        const int oi = (e.targetMode == kTargetOther) ? FindEntity(world, target) : si;
        if (si < 0 || oi < 0) continue;                       // defensive (deterministic no-op)
        ApplyEffectTo(world, world.entities[(size_t)oi], e);
    }
    return kActivateOk;
}

// ----- StepAbilities(world) -> the count of periodic firings this tick --------------------------------------
// ONE deterministic GAS tick. Entities in ASCENDING entities[] (spawn) order; per entity the PINNED phase
// order: (1) cooldowns decrement; (2) periodic effects fire (ascending instanceId): --periodCountdown, at 0
// fire magnitude * stacks as a BASE ADD + reset countdown = periodTicks; (3) durations decrement (kDurTicks
// only) + effects at 0 removed (order-preserving erase — ascending instanceId maintained); (4)
// RecomputeCurrents. Then ++world.tick ONCE. Returns the periodic-firing count (a derived report value —
// NOT world state). Pure integer, fixed order -> bit-identical on every peer/platform.
inline uint32_t StepAbilities(GasWorld& world) {
    uint32_t periodicFired = 0;
    for (size_t i = 0; i < world.entities.size(); ++i) {
        GasEntity& e = world.entities[i];
        // (1) cooldowns.
        for (Cooldown& cd : e.cooldowns)
            if (cd.remaining > 0) --cd.remaining;
        // (2) periodic firings (ascending instanceId — the vector order).
        bool baseChanged = false;
        for (ActiveEffect& ae : e.active) {
            if (ae.def.periodTicks == 0) continue;
            if (ae.periodCountdown > 0) --ae.periodCountdown;
            if (ae.periodCountdown == 0) {
                if (ae.def.attr < kAttrCount) {
                    e.attrs.base[ae.def.attr] =
                        (fx)((int64_t)e.attrs.base[ae.def.attr] +
                             (int64_t)ae.def.magnitude * (int64_t)ae.stacks);   // ALWAYS an add (pinned)
                    baseChanged = true;
                }
                ae.periodCountdown = ae.def.periodTicks;
                ++periodicFired;
            }
        }
        if (baseChanged) ClampBases(e.attrs);
        // (3) duration expiry (kDurTicks only; kDurInfinite never expires). Order-preserving erase.
        for (size_t a = 0; a < e.active.size(); /* in-body */) {
            ActiveEffect& ae = e.active[a];
            if (ae.def.durKind == kDurTicks) {
                if (ae.remainingTicks > 0) --ae.remainingTicks;
                if (ae.remainingTicks == 0) {
                    e.active.erase(e.active.begin() + (std::ptrdiff_t)a);   // expired (ascending kept)
                    continue;
                }
            }
            ++a;
        }
        // (4) the derived currents.
        RecomputeCurrents(e);
    }
    ++world.tick;
    return periodicFired;
}

// ----- GasCommand + ApplyGasCommands: the activation-request input bus (commands BEFORE the step) ------------
// The wire verb: on tick `tick`, `caster` requests `abilityId` at `target`. ApplyGasCommands applies, in
// FIXED array order, every command whose .tick == tick (TryActivate each — a failure is a deterministic
// no-op with a pinned reason). Returns the successful-activation count; outResults (optional) receives one
// ActivateResult per MATCHED command in array order.
struct GasCommand {
    uint32_t tick      = 0;
    EntityId caster    = kNoEntity;
    uint32_t abilityId = 0;
    EntityId target    = kNoEntity;
};

inline uint32_t ApplyGasCommands(GasWorld& world, const AbilityKit& kit, const std::vector<GasCommand>& cmds,
                                 uint32_t tick, std::vector<ActivateResult>* outResults = nullptr) {
    uint32_t ok = 0;
    for (size_t c = 0; c < cmds.size(); ++c) {
        if (cmds[c].tick != tick) continue;
        const ActivateResult r = TryActivate(world, kit, cmds[c].caster, cmds[c].abilityId, cmds[c].target);
        if (r == kActivateOk) ++ok;
        if (outResults) outResults->push_back(r);
    }
    return ok;
}

// ----- StepGas: ONE composed tick — commands BEFORE the step (the verdict/convex contract) -------------------
inline uint32_t StepGas(GasWorld& world, const AbilityKit& kit, const std::vector<GasCommand>& cmds,
                        uint32_t tick) {
    ApplyGasCommands(world, kit, cmds, tick);
    return StepAbilities(world);
}

// ----- Equality + digest: the byte-comparable currency (the lockstep/rollback proofs) ------------------------
inline bool EffectDefsEqual(const EffectDef& a, const EffectDef& b) {
    return a.effectId == b.effectId && a.attr == b.attr && a.op == b.op && a.magnitude == b.magnitude &&
           a.durKind == b.durKind && a.durationTicks == b.durationTicks && a.stackPolicy == b.stackPolicy &&
           a.maxStacks == b.maxStacks && a.periodTicks == b.periodTicks && a.targetMode == b.targetMode;
}

inline bool GasStatesEqual(const GasWorld& a, const GasWorld& b) {
    if (a.tick != b.tick || a.nextEntityId != b.nextEntityId || a.nextInstanceId != b.nextInstanceId)
        return false;
    if (a.entities.size() != b.entities.size()) return false;
    for (size_t i = 0; i < a.entities.size(); ++i) {
        const GasEntity& x = a.entities[i];
        const GasEntity& y = b.entities[i];
        if (x.id != y.id) return false;
        for (uint32_t at = 0; at < kAttrCount; ++at) {
            if (x.attrs.base[at] != y.attrs.base[at]) return false;
            if (x.attrs.current[at] != y.attrs.current[at]) return false;
        }
        if (x.active.size() != y.active.size()) return false;
        for (size_t e = 0; e < x.active.size(); ++e) {
            const ActiveEffect& p = x.active[e];
            const ActiveEffect& q = y.active[e];
            if (p.instanceId != q.instanceId || p.remainingTicks != q.remainingTicks ||
                p.stacks != q.stacks || p.periodCountdown != q.periodCountdown) return false;
            if (!EffectDefsEqual(p.def, q.def)) return false;
        }
        if (x.cooldowns.size() != y.cooldowns.size()) return false;
        for (size_t k = 0; k < x.cooldowns.size(); ++k)
            if (x.cooldowns[k].abilityId != y.cooldowns[k].abilityId ||
                x.cooldowns[k].remaining != y.cooldowns[k].remaining) return false;
    }
    return true;
}

// DigestGasWorld: the canonical fold over the WHOLE determinism-relevant state (the verdict::DigestFnv
// discipline — fixed field order, separators between pools). Two GasStatesEqual worlds hash identically.
inline uint64_t DigestGasWorld(const GasWorld& world) {
    verdict::DigestFnv d;
    d.mix32(world.tick);
    d.mix32((uint32_t)world.nextEntityId);
    d.mix32(world.nextInstanceId);
    d.mix32((uint32_t)world.entities.size());
    d.sep();
    for (const GasEntity& e : world.entities) {
        d.mix32((uint32_t)e.id);
        for (uint32_t a = 0; a < kAttrCount; ++a) d.mix32((uint32_t)e.attrs.base[a]);
        for (uint32_t a = 0; a < kAttrCount; ++a) d.mix32((uint32_t)e.attrs.current[a]);
        d.sep();
        d.mix32((uint32_t)e.active.size());
        for (const ActiveEffect& ae : e.active) {
            d.mix32(ae.instanceId);
            d.mix32(ae.def.effectId);
            d.mix32(ae.def.attr);
            d.mix32(ae.def.op);
            d.mix32((uint32_t)ae.def.magnitude);
            d.mix32(ae.def.durKind);
            d.mix32(ae.def.durationTicks);
            d.mix32(ae.def.stackPolicy);
            d.mix32(ae.def.maxStacks);
            d.mix32(ae.def.periodTicks);
            d.mix32(ae.def.targetMode);
            d.mix32(ae.remainingTicks);
            d.mix32(ae.stacks);
            d.mix32(ae.periodCountdown);
        }
        d.sep();
        d.mix32((uint32_t)e.cooldowns.size());
        for (const Cooldown& cd : e.cooldowns) { d.mix32(cd.abilityId); d.mix32(cd.remaining); }
        d.sep();
    }
    return d.h;
}

// DigestHex: the 16-hex-digit lowercase string of a 64-bit digest (the DX5 formatting convention).
inline std::string DigestHex(uint64_t h) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return std::string(buf);
}

// ===== LOCKSTEP + ROLLBACK (the established command+snapshot mold; GasWorld is COPYABLE — a copy IS the
// snapshot, so the harness is the VD5 control flow WITHOUT the non-copyable clone dance) ====================

// RunGasLockstep: two peers cloned from world0 (a value copy — the complete snapshot), BOTH fed ONLY the
// SAME command stream for `ticks` StepGas ticks. Sets *outIdentical to whether the two final worlds are
// byte-identical (GasStatesEqual — THE lockstep proof: a peer re-derives every damage number, buff stack,
// and cooldown bit-for-bit from inputs alone). Returns the converged authority world.
inline GasWorld RunGasLockstep(const GasWorld& world0, const AbilityKit& kit,
                               const std::vector<GasCommand>& commands, uint32_t ticks,
                               bool* outIdentical = nullptr) {
    GasWorld authority = world0;
    GasWorld replica   = world0;
    for (uint32_t t = 0; t < ticks; ++t) {
        StepGas(authority, kit, commands, t);
        StepGas(replica,   kit, commands, t);
    }
    if (outIdentical) *outIdentical = GasStatesEqual(authority, replica);
    return authority;
}

// RunGasRollback: (1) advance 0..rollbackAt with the authoritative stream; (2) snapshot (a value copy);
// (2b) speculate <= 3 ticks with the MISPREDICTED stream (the divergent client prediction — e.g. a
// mispredicted activation); (3) ROLLBACK (restore the copy) + re-sim rollbackAt..ticks with the CORRECT
// stream. Sets *outCorrectedEqAuthority to corrected == straight-lockstep-authority (bit-exact recovery)
// and *outMispredictDiverged to whether the speculation ACTUALLY diverged from the authority at that tick
// (the non-vacuous control). Returns the corrected world.
inline GasWorld RunGasRollback(const GasWorld& world0, const AbilityKit& kit,
                               const std::vector<GasCommand>& authStream,
                               const std::vector<GasCommand>& mispredictStream,
                               uint32_t ticks, uint32_t rollbackAt,
                               bool* outCorrectedEqAuthority = nullptr,
                               bool* outMispredictDiverged = nullptr) {
    GasWorld w = world0;
    for (uint32_t t = 0; t < rollbackAt; ++t)
        StepGas(w, kit, authStream, t);
    const GasWorld snap = w;                                   // (2) the complete restore point (a copy)
    uint32_t specTicks = ticks - rollbackAt;
    if (specTicks > 3u) specTicks = 3u;
    for (uint32_t s = 0; s < specTicks; ++s)
        StepGas(w, kit, mispredictStream, rollbackAt + s);     // (2b) the misprediction
    const GasWorld specWorld = w;                              // the diverged pre-rollback world
    w = snap;                                                  // (3) ROLLBACK
    for (uint32_t t = rollbackAt; t < ticks; ++t)
        StepGas(w, kit, authStream, t);

    if (outCorrectedEqAuthority || outMispredictDiverged) {
        GasWorld authAtSpec = world0;
        for (uint32_t t = 0; t < rollbackAt + specTicks; ++t)
            StepGas(authAtSpec, kit, authStream, t);
        if (outMispredictDiverged) *outMispredictDiverged = !GasStatesEqual(specWorld, authAtSpec);
        if (outCorrectedEqAuthority) {
            const GasWorld authFinal = RunGasLockstep(world0, kit, authStream, ticks);
            *outCorrectedEqAuthority = GasStatesEqual(w, authFinal);
        }
    }
    return w;
}

// ===== THE SHOWCASE FIXTURE — the CORE KIT + THE 60-TICK DUEL (FIXED forever; the test pins the kit digest,
// the duel attribute-trace digest, and the exact final attribute integers; the --gas1-duel showcases run
// THIS EXACT scenario on both backends — byte-identical viz BY CONSTRUCTION) ================================

// The pinned ability ids + effect ids (the kit vocabulary).
inline constexpr uint32_t kAbilityFireball = 1u;   // mana 20.0, cd 6: instant -30.0 health + a 3-tick -2.0/tick burn
inline constexpr uint32_t kAbilityShield   = 2u;   // mana 10.0, cd 8: +15.0 armor for 10 ticks, kStackRefresh
inline constexpr uint32_t kAbilityHaste    = 3u;   // mana  5.0, cd 1: moveSpeed x1.25 for 20 ticks, kStackStack max 3

inline constexpr uint32_t kEffFireballHit  = 10u;
inline constexpr uint32_t kEffFireballBurn = 11u;
inline constexpr uint32_t kEffShieldArmor  = 20u;
inline constexpr uint32_t kEffHasteSpeed   = 30u;

// MakeCoreKit: the ability kit, authored EXCLUSIVELY through the KitBuilder (the PA1 "authored, not
// hardcoded" proof — the test pins DigestKit of this result). Q16.16 magnitudes; x1.25 == 81920.
inline AbilityKit MakeCoreKit() {
    KitBuilder b;
    b.Ability(kAbilityFireball, kAttrMana, 20 * kOne, 6u)
     .Effect(kEffFireballHit,  kAttrHealth,    kOpAdd,      -30 * kOne, kDurInstant, 0u,  kStackIgnore,  1u, 0u, kTargetOther)
     .Effect(kEffFireballBurn, kAttrHealth,    kOpAdd,      -2 * kOne,  kDurTicks,   3u,  kStackRefresh, 1u, 1u, kTargetOther)
     .Ability(kAbilityShield, kAttrMana, 10 * kOne, 8u)
     .Effect(kEffShieldArmor,  kAttrArmor,     kOpAdd,      15 * kOne,  kDurTicks,   10u, kStackRefresh, 1u, 0u, kTargetSelf)
     .Ability(kAbilityHaste, kAttrMana, 5 * kOne, 1u)
     .Effect(kEffHasteSpeed,   kAttrMoveSpeed, kOpMultiply, (fx)81920,  kDurTicks,   20u, kStackStack,   3u, 0u, kTargetSelf);
    return b.Build();
}

// The duel base attributes (both duelists identical): health 200/200, mana 100/100, moveSpeed 4.0,
// attackPower 10.0, armor 5.0, regen 0.
inline constexpr fx kDuelBases[kAttrCount] = {
    200 * kOne, 200 * kOne, 100 * kOne, 100 * kOne, 4 * kOne, 10 * kOne, 5 * kOne, 0,
};

inline constexpr uint32_t kDuelTicks = 60u;

// MakeDuelWorld: two duelists, A = id 1, B = id 2 (spawn order pinned).
inline GasWorld MakeDuelWorld() {
    GasWorld w;
    SpawnGasEntity(w, kDuelBases);   // A = 1
    SpawnGasEntity(w, kDuelBases);   // B = 2
    return w;
}

// MakeDuelStream: THE FIXED 60-tick duel script. Deliberately exercises the whole surface: fireballs both
// ways (instant + burn DoT), a shield mid-fight, haste stacked to the cap, a COOLDOWN rejection (t7 — the
// t2 fireball's 6-tick cooldown still has 1 tick left) and TWO UNAFFORDABLE rejections (t32/t44 — A is down
// to 10.0 mana after the t30 shield). The rejections are part of the pinned trace (deterministic failures).
inline std::vector<GasCommand> MakeDuelStream() {
    const EntityId A = 1u, B = 2u;
    return {
        GasCommand{2u,  A, kAbilityFireball, B},   // ok: -30 + burn on B
        GasCommand{3u,  B, kAbilityShield,   B},   // ok: +15 armor on B for 10 ticks
        GasCommand{4u,  B, kAbilityFireball, A},   // ok: -30 + burn on A
        GasCommand{5u,  A, kAbilityHaste,    A},   // ok: haste stack 1
        GasCommand{6u,  A, kAbilityHaste,    A},   // ok: haste stack 2
        GasCommand{7u,  A, kAbilityFireball, B},   // FAIL kActivateOnCooldown (1 tick left)
        GasCommand{8u,  A, kAbilityFireball, B},   // ok: the exact ready tick
        GasCommand{9u,  A, kAbilityHaste,    A},   // ok: haste stack 3 (the cap)
        GasCommand{10u, A, kAbilityHaste,    A},   // ok activation; stacks CAPPED at 3 (duration refreshes)
        GasCommand{12u, B, kAbilityFireball, A},   // ok
        GasCommand{15u, B, kAbilityHaste,    B},   // ok
        GasCommand{20u, A, kAbilityFireball, B},   // ok (A mana 100-80=20 -> exactly affordable)
        GasCommand{26u, B, kAbilityFireball, A},   // ok
        GasCommand{30u, A, kAbilityShield,   A},   // ok (A mana 20-10=10)
        GasCommand{32u, A, kAbilityFireball, B},   // FAIL kActivateUnaffordable (10.0 < 20.0)
        GasCommand{40u, B, kAbilityFireball, A},   // ok (B mana 25-20=5)
        GasCommand{44u, A, kAbilityFireball, B},   // FAIL kActivateUnaffordable (still 10.0)
    };
}

// ----- DuelTickSample / DuelRun / RunDuelScenario: the shared per-tick trace (test + BOTH showcases run
// THIS EXACT code — the strip-chart pixels and the pinned digests derive from one implementation) -----------
struct DuelTickSample {
    fx       health[2];        // current health of A, B AFTER this tick's step
    fx       mana[2];          // current mana
    fx       armor[2];         // current armor
    fx       moveSpeed[2];     // current move speed
    uint8_t  castOk[2];        // this tick: entity issued a SUCCESSFUL activation
    uint8_t  castFail[2];      // this tick: entity issued a FAILED activation (pinned reason in the trace digest)
    uint8_t  burnActive[2];    // a periodic (DoT) effect is live on the entity after the step
    uint8_t  cdActive[2];      // any ability cooldown counter > 0 after the step
    uint32_t periodicFired;    // world-wide periodic firings this tick
};

struct DuelRun {
    GasWorld                    finalWorld;
    std::vector<DuelTickSample> samples;         // kDuelTicks entries
    uint64_t                    traceDigest = 0; // the per-tick attribute+result fold (pinned)
    uint32_t                    activationsOk = 0;
    uint32_t                    activationsFailed = 0;
    uint32_t                    effectsApplied = 0;   // effect applications on ok casts + periodic firings
    uint32_t                    periodicTotal = 0;
};

inline DuelRun RunDuelScenario() {
    const AbilityKit kit = MakeCoreKit();
    const std::vector<GasCommand> stream = MakeDuelStream();
    DuelRun run;
    GasWorld w = MakeDuelWorld();
    verdict::DigestFnv trace;
    for (uint32_t t = 0; t < kDuelTicks; ++t) {
        DuelTickSample s{};
        // Commands BEFORE the step (array order) — record per-caster ok/fail + the effect-application count.
        for (const GasCommand& c : stream) {
            if (c.tick != t) continue;
            const ActivateResult r = TryActivate(w, kit, c.caster, c.abilityId, c.target);
            const int who = (c.caster == 2u) ? 1 : 0;
            if (r == kActivateOk) {
                s.castOk[who] = 1;
                ++run.activationsOk;
                for (const AbilityDef& a : kit.abilities)
                    if (a.abilityId == c.abilityId) run.effectsApplied += (uint32_t)a.effects.size();
            } else {
                s.castFail[who] = 1;
                ++run.activationsFailed;
            }
            trace.mix32((uint32_t)r);   // the failure REASONS are part of the pinned trace
        }
        trace.sep();
        // The step.
        s.periodicFired = StepAbilities(w);
        run.periodicTotal += s.periodicFired;
        run.effectsApplied += s.periodicFired;
        // Sample AFTER the step (entities[0] = A, entities[1] = B — spawn order pinned).
        for (int i = 0; i < 2; ++i) {
            const GasEntity& e = w.entities[(size_t)i];
            s.health[i]    = e.attrs.current[kAttrHealth];
            s.mana[i]      = e.attrs.current[kAttrMana];
            s.armor[i]     = e.attrs.current[kAttrArmor];
            s.moveSpeed[i] = e.attrs.current[kAttrMoveSpeed];
            for (const ActiveEffect& ae : e.active)
                if (ae.def.periodTicks != 0) s.burnActive[i] = 1;
            for (const Cooldown& cd : e.cooldowns)
                if (cd.remaining > 0) s.cdActive[i] = 1;
            for (uint32_t a = 0; a < kAttrCount; ++a) trace.mix32((uint32_t)e.attrs.current[a]);
        }
        trace.sep();
        run.samples.push_back(s);
    }
    run.traceDigest = trace.h;
    run.finalWorld  = w;
    return run;
}

}  // namespace gas
}  // namespace hf::game
