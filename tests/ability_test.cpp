// Slice GAS1 — A DETERMINISTIC GAMEPLAY ABILITY SYSTEM (attributes + effects + cooldowns, hf::game::gas).
// ability.h is a NEW additive sibling that #includes game/verdict.h READ-ONLY (the EntityId vocabulary +
// the DigestFnv discipline). PURE CPU INTEGER (Q16.16 / uint32 — no float, no wall clock).
//
// What this test PINS (the spec's proofs):
//   (a) THE FOLD ORDER: base -> ALL adds (ascending instance id) -> ALL multiplies -> overrides
//       (last-instance wins) — application order does NOT matter; exact pinned currents.
//   (b) STACKING: kStackStack caps at maxStacks (pinned current at the cap); kStackRefresh resets
//       duration NOT magnitude/stacks; kStackIgnore no-ops.
//   (c) DoT: a periodic effect fires exactly every periodTicks; total damage exact; expiry at the exact tick.
//   (d) COOLDOWN + COST: pinned deterministic failure reasons (on-cooldown / unaffordable / unknown-*);
//       activation succeeds at the exact ready tick; the cost is deducted exactly.
//   (e) THE SCENARIO: the FIXED 60-tick two-entity duel — the full attribute-trace digest + the exact
//       final attribute integers + the authored-kit digest (the PA1 "authored, not hardcoded" pin).
//   (f) IDENTITY: an entity with zero effects — currents == bases bit-exact every tick.
//   (g) LOCKSTEP: a peer re-derives the duel bit-for-bit from the command stream alone; rollback corrects
//       a GENUINELY-diverged misprediction (with the non-diverging control); snapshot completeness
//       (omitting cooldowns or active effects diverges — the controls).
//   (h) The pinned digests are IDENTICAL under MSVC and local clang (the cross-compiler anchor).
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "game/ability.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace gas = hf::game::gas;
using gas::fx;
using gas::kOne;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// A neutral single-entity world (the fold/stacking/DoT scaffolding).
static gas::GasWorld OneEntityWorld() {
    gas::GasWorld w;
    gas::SpawnGasEntity(w, gas::kDuelBases);   // id 1: health 200/200, mana 100/100, speed 4, ap 10, armor 5
    return w;
}

// A duration-effect literal helper (direct ApplyEffectTo — the fold proofs bypass abilities on purpose).
static gas::EffectDef Dur(uint32_t effectId, uint32_t attr, uint32_t op, fx mag, uint32_t ticks,
                          uint32_t stackPolicy = gas::kStackIgnore, uint32_t maxStacks = 1,
                          uint32_t period = 0) {
    gas::EffectDef e;
    e.effectId      = effectId;
    e.attr          = attr;
    e.op            = op;
    e.magnitude     = mag;
    e.durKind       = gas::kDurTicks;
    e.durationTicks = ticks;
    e.stackPolicy   = stackPolicy;
    e.maxStacks     = maxStacks;
    e.periodTicks   = period;
    e.targetMode    = gas::kTargetSelf;
    return e;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- (a) THE FOLD ORDER: adds before multiplies regardless of application order; override last wins ----
    {
        // Apply MULTIPLY x2 FIRST, then ADD +5, on attackPower base 10 -> the pinned fold is
        // (10 + 5) * 2 = 30 (NOT 10*2+5=25 — the fold order beats the application order).
        gas::GasWorld w1 = OneEntityWorld();
        gas::ApplyEffectTo(w1, w1.entities[0], Dur(1u, gas::kAttrAttackPower, gas::kOpMultiply, 2 * kOne, 50u));
        gas::ApplyEffectTo(w1, w1.entities[0], Dur(2u, gas::kAttrAttackPower, gas::kOpAdd, 5 * kOne, 50u));
        check(w1.entities[0].attrs.current[gas::kAttrAttackPower] == 30 * kOne,
              "fold: multiply-then-add applications fold as (base+adds)*multiplies == 30.0");
        // The reverse application order yields the SAME current (the fold order is pinned, not the order).
        gas::GasWorld w2 = OneEntityWorld();
        gas::ApplyEffectTo(w2, w2.entities[0], Dur(2u, gas::kAttrAttackPower, gas::kOpAdd, 5 * kOne, 50u));
        gas::ApplyEffectTo(w2, w2.entities[0], Dur(1u, gas::kAttrAttackPower, gas::kOpMultiply, 2 * kOne, 50u));
        check(w2.entities[0].attrs.current[gas::kAttrAttackPower] == 30 * kOne,
              "fold: add-then-multiply applications fold identically == 30.0");
        // Override: the LAST instance (highest instance id == most recent application) wins.
        gas::ApplyEffectTo(w2, w2.entities[0], Dur(3u, gas::kAttrAttackPower, gas::kOpOverride, 7 * kOne, 50u));
        gas::ApplyEffectTo(w2, w2.entities[0], Dur(4u, gas::kAttrAttackPower, gas::kOpOverride, 9 * kOne, 50u));
        check(w2.entities[0].attrs.current[gas::kAttrAttackPower] == 9 * kOne,
              "fold: override last-instance-wins == 9.0");
        // Base is UNTOUCHED by duration modifiers (the base-vs-current split).
        check(w2.entities[0].attrs.base[gas::kAttrAttackPower] == 10 * kOne,
              "fold: base attackPower untouched by duration modifiers");
    }

    // ---- (b) STACKING: kStackStack caps; kStackRefresh resets duration not magnitude; kStackIgnore no-ops ----
    {
        // kStackStack: x1.25 moveSpeed, max 3 stacks; 4 applications cap at 3 -> 4.0 * 1.25^3 == 7.8125.
        gas::GasWorld w = OneEntityWorld();
        const gas::EffectDef haste = Dur(1u, gas::kAttrMoveSpeed, gas::kOpMultiply, (fx)81920, 20u,
                                         gas::kStackStack, 3u);
        for (int i = 0; i < 4; ++i) gas::ApplyEffectTo(w, w.entities[0], haste);
        check(w.entities[0].active.size() == 1 && w.entities[0].active[0].stacks == 3u,
              "stacking: kStackStack caps at maxStacks == 3");
        check(w.entities[0].attrs.current[gas::kAttrMoveSpeed] == (fx)512000,
              "stacking: 3-stack x1.25 fold == 7.8125 (512000) exact");
        // kStackRefresh: duration resets, stacks/magnitude unchanged.
        gas::GasWorld r = OneEntityWorld();
        const gas::EffectDef shield = Dur(2u, gas::kAttrArmor, gas::kOpAdd, 15 * kOne, 10u,
                                          gas::kStackRefresh, 1u);
        gas::ApplyEffectTo(r, r.entities[0], shield);
        gas::StepAbilities(r);
        gas::StepAbilities(r);
        check(r.entities[0].active[0].remainingTicks == 8u, "stacking: refresh control decayed to 8");
        gas::ApplyEffectTo(r, r.entities[0], shield);   // the refresh
        check(r.entities[0].active[0].remainingTicks == 10u && r.entities[0].active[0].stacks == 1u,
              "stacking: kStackRefresh resets duration to 10, stacks stay 1");
        check(r.entities[0].attrs.current[gas::kAttrArmor] == 20 * kOne,
              "stacking: refresh does NOT change magnitude (armor stays 5+15 == 20.0)");
        // kStackIgnore: a re-application while active is a no-op (duration NOT reset).
        gas::GasWorld g = OneEntityWorld();
        const gas::EffectDef ig = Dur(3u, gas::kAttrArmor, gas::kOpAdd, 15 * kOne, 10u, gas::kStackIgnore, 1u);
        gas::ApplyEffectTo(g, g.entities[0], ig);
        gas::StepAbilities(g);
        const uint32_t remBefore = g.entities[0].active[0].remainingTicks;
        gas::ApplyEffectTo(g, g.entities[0], ig);
        check(g.entities[0].active.size() == 1 && g.entities[0].active[0].remainingTicks == remBefore,
              "stacking: kStackIgnore re-application is a no-op (duration unchanged)");
    }

    // ---- (c) DoT: exact period firing, exact total, exact expiry tick ----
    {
        // The burn: -2.0 health per tick, period 1, duration 3 -> fires on exactly the 3 steps after
        // application; total -6.0; removed on the 3rd step.
        gas::GasWorld w = OneEntityWorld();
        gas::ApplyEffectTo(w, w.entities[0],
                           Dur(1u, gas::kAttrHealth, gas::kOpAdd, -2 * kOne, 3u, gas::kStackRefresh, 1u, 1u));
        check(gas::StepAbilities(w) == 1u, "dot: fires on step 1");
        check(w.entities[0].attrs.base[gas::kAttrHealth] == 198 * kOne, "dot: -2.0 after step 1");
        check(gas::StepAbilities(w) == 1u, "dot: fires on step 2");
        check(gas::StepAbilities(w) == 1u, "dot: fires on step 3");
        check(w.entities[0].attrs.base[gas::kAttrHealth] == 194 * kOne, "dot: total exactly -6.0");
        check(w.entities[0].active.empty(), "dot: expired at the exact tick (step 3)");
        check(gas::StepAbilities(w) == 0u, "dot: no firing after expiry");
        check(w.entities[0].attrs.base[gas::kAttrHealth] == 194 * kOne, "dot: no further damage");
        // A period-2 duration-4 DoT fires on steps 2 and 4 only (2 firings, -4.0 total, expiry step 4).
        gas::GasWorld p = OneEntityWorld();
        gas::ApplyEffectTo(p, p.entities[0],
                           Dur(2u, gas::kAttrHealth, gas::kOpAdd, -2 * kOne, 4u, gas::kStackRefresh, 1u, 2u));
        check(gas::StepAbilities(p) == 0u, "dot: period-2 silent on step 1");
        check(gas::StepAbilities(p) == 1u, "dot: period-2 fires on step 2");
        check(gas::StepAbilities(p) == 0u, "dot: period-2 silent on step 3");
        check(gas::StepAbilities(p) == 1u, "dot: period-2 fires on step 4");
        check(p.entities[0].attrs.base[gas::kAttrHealth] == 196 * kOne, "dot: period-2 total exactly -4.0");
        check(p.entities[0].active.empty(), "dot: period-2 expired at step 4");
        // A periodic effect does NOT contribute to the current fold (currents == bases while burning).
        gas::GasWorld f = OneEntityWorld();
        gas::ApplyEffectTo(f, f.entities[0],
                           Dur(3u, gas::kAttrHealth, gas::kOpAdd, -2 * kOne, 3u, gas::kStackRefresh, 1u, 1u));
        check(f.entities[0].attrs.current[gas::kAttrHealth] == f.entities[0].attrs.base[gas::kAttrHealth],
              "dot: a periodic effect is excluded from the current fold");
    }

    // ---- (d) COOLDOWN + COST: pinned failure reasons; the exact ready tick; exact cost deduction ----
    {
        const gas::AbilityKit kit = gas::MakeCoreKit();
        gas::GasWorld w;
        gas::SpawnGasEntity(w, gas::kDuelBases);   // caster 1
        gas::SpawnGasEntity(w, gas::kDuelBases);   // target 2
        // Unknown-* reasons (nothing mutates).
        check(gas::TryActivate(w, kit, 1u, 999u, 2u) == gas::kActivateUnknownAbility,
              "gate: unknown ability -> kActivateUnknownAbility");
        check(gas::TryActivate(w, kit, 77u, gas::kAbilityFireball, 2u) == gas::kActivateUnknownCaster,
              "gate: unknown caster -> kActivateUnknownCaster");
        check(gas::TryActivate(w, kit, 1u, gas::kAbilityFireball, 77u) == gas::kActivateUnknownTarget,
              "gate: unknown target -> kActivateUnknownTarget");
        // Activation ok: cost deducted EXACTLY (mana 100 -> 80), cooldown started (6).
        check(gas::TryActivate(w, kit, 1u, gas::kAbilityFireball, 2u) == gas::kActivateOk,
              "gate: first fireball activates");
        check(w.entities[0].attrs.base[gas::kAttrMana] == 80 * kOne, "gate: cost deducted exactly (80.0)");
        check(w.entities[1].attrs.base[gas::kAttrHealth] == 170 * kOne, "gate: instant -30.0 landed");
        // On cooldown for exactly 6 steps; ready on the 6th.
        for (int s = 0; s < 5; ++s) {
            gas::StepAbilities(w);
            check(gas::TryActivate(w, kit, 1u, gas::kAbilityFireball, 2u) == gas::kActivateOnCooldown,
                  "gate: on cooldown during the 5 waiting ticks");
        }
        gas::StepAbilities(w);   // the 6th decrement -> ready
        check(gas::TryActivate(w, kit, 1u, gas::kAbilityFireball, 2u) == gas::kActivateOk,
              "gate: activates at the EXACT ready tick");
        // Unaffordable: drain the caster's mana base below the cost.
        w.entities[0].attrs.base[gas::kAttrMana] = 19 * kOne;
        gas::RecomputeCurrents(w.entities[0]);
        gas::StepAbilities(w); gas::StepAbilities(w); gas::StepAbilities(w);
        gas::StepAbilities(w); gas::StepAbilities(w); gas::StepAbilities(w);   // clear the cooldown
        check(gas::TryActivate(w, kit, 1u, gas::kAbilityFireball, 2u) == gas::kActivateUnaffordable,
              "gate: 19.0 mana < 20.0 cost -> kActivateUnaffordable");
    }

    // ---- (e) THE SCENARIO: the FIXED 60-tick duel — kit digest + trace digest + exact finals ----
    {
        // The authored-kit pin (the PA1 "authored, not hardcoded" proof — the kit is built EXCLUSIVELY
        // through the KitBuilder; any drift in the authoring calls moves this digest).
        const uint64_t kitDigest = gas::DigestKit(gas::MakeCoreKit());
        const uint64_t kPinnedKitDigest = 0xa092a05e3a6b1a50ull;  // PINNED on first run (MSVC == clang)
        std::printf("gas1: kit digest %s\n", gas::DigestHex(kitDigest).c_str());
        check(kitDigest == kPinnedKitDigest, "duel: authored kit digest matches the pinned value");

        const gas::DuelRun run = gas::RunDuelScenario();
        std::printf("gas1: duel trace digest %s, ok %u, failed %u, periodic %u, effectsApplied %u\n",
                    gas::DigestHex(run.traceDigest).c_str(), run.activationsOk, run.activationsFailed,
                    run.periodicTotal, run.effectsApplied);
        // The activation/effect tallies (the scripted duel structure).
        check(run.activationsOk == 14u, "duel: exactly 14 successful activations");
        check(run.activationsFailed == 3u, "duel: exactly 3 deterministic rejections (1 cooldown + 2 cost)");
        check(run.periodicTotal == 21u, "duel: 7 burns x 3 ticks == 21 periodic firings");
        check(run.effectsApplied == 42u, "duel: 21 activation effects + 21 periodic firings");
        // The exact final attribute integers (Q16.16).
        const gas::GasEntity& A = run.finalWorld.entities[0];
        const gas::GasEntity& B = run.finalWorld.entities[1];
        check(A.attrs.base[gas::kAttrHealth] == 56 * kOne, "duel: A final health == 56.0 exact");
        check(B.attrs.base[gas::kAttrHealth] == 92 * kOne, "duel: B final health == 92.0 exact");
        check(A.attrs.base[gas::kAttrMana] == 10 * kOne, "duel: A final mana == 10.0 exact");
        check(B.attrs.base[gas::kAttrMana] == 5 * kOne, "duel: B final mana == 5.0 exact");
        // All buffs expired + all cooldowns idle by tick 60 -> currents == bases (the fold identity).
        check(A.active.empty() && B.active.empty(), "duel: all effects expired by tick 60");
        for (uint32_t a = 0; a < gas::kAttrCount; ++a) {
            check(A.attrs.current[a] == A.attrs.base[a], "duel: A currents == bases at the end");
            check(B.attrs.current[a] == B.attrs.base[a], "duel: B currents == bases at the end");
        }
        // Mid-fight texture (the samples): haste peaks at 3 stacks (7.8125 speed) and armor peaks at 20.0.
        check(run.samples[10].moveSpeed[0] == (fx)512000, "duel: A move speed == 7.8125 at tick 10 (3 stacks)");
        check(run.samples[4].armor[1] == 20 * kOne, "duel: B armor == 20.0 while shielded");
        check(run.samples[2].burnActive[1] == 1, "duel: B burning after the t2 fireball");
        // The trace digest (attributes + activation results, every tick) — THE duel pin.
        const uint64_t kPinnedTraceDigest = 0xf74f7e4198440670ull;  // PINNED on first run (MSVC == clang)
        check(run.traceDigest == kPinnedTraceDigest, "duel: attribute-trace digest matches the pinned value");
        // The final-world digest is reproducible (two full runs byte-identical).
        const gas::DuelRun run2 = gas::RunDuelScenario();
        check(gas::GasStatesEqual(run.finalWorld, run2.finalWorld) && run2.traceDigest == run.traceDigest,
              "duel: two runs are byte-identical");
    }

    // ---- (f) IDENTITY: zero effects -> currents == bases bit-exact every tick ----
    {
        gas::GasWorld w = OneEntityWorld();
        for (uint32_t t = 0; t < 20u; ++t) {
            gas::StepAbilities(w);
            for (uint32_t a = 0; a < gas::kAttrCount; ++a)
                check(w.entities[0].attrs.current[a] == w.entities[0].attrs.base[a],
                      "identity: zero-effect currents == bases every tick");
        }
    }

    // ---- (g) LOCKSTEP + ROLLBACK + SNAPSHOT COMPLETENESS ----
    {
        const gas::AbilityKit kit = gas::MakeCoreKit();
        const std::vector<gas::GasCommand> auth = gas::MakeDuelStream();
        const gas::GasWorld world0 = gas::MakeDuelWorld();

        // LOCKSTEP: a peer fed ONLY the command stream re-derives the duel bit-for-bit.
        bool identical = false;
        const gas::GasWorld authority = gas::RunGasLockstep(world0, kit, auth, gas::kDuelTicks, &identical);
        check(identical, "lockstep: authority == replica from inputs alone (bit-identical)");
        const gas::DuelRun run = gas::RunDuelScenario();
        check(gas::GasStatesEqual(authority, run.finalWorld),
              "lockstep: the lockstep authority == the duel-scenario final world");

        // ROLLBACK: a GENUINELY-diverged misprediction (t12 fireball mispredicted as a haste) corrected.
        std::vector<gas::GasCommand> mispredict = auth;
        for (gas::GasCommand& c : mispredict)
            if (c.tick == 12u) { c.abilityId = gas::kAbilityHaste; c.target = c.caster; }
        bool corrected = false, diverged = false;
        const gas::GasWorld rolled = gas::RunGasRollback(world0, kit, auth, mispredict, gas::kDuelTicks, 10u,
                                                         &corrected, &diverged);
        check(diverged, "rollback: the misprediction ACTUALLY diverged (non-vacuous)");
        check(corrected, "rollback: corrected == the straight lockstep authority (bit-exact recovery)");
        check(gas::GasStatesEqual(rolled, authority), "rollback: returned world == authority");
        // The non-diverging CONTROL: mispredict == auth -> diverged false, corrected still true.
        bool corrected2 = false, diverged2 = true;
        (void)gas::RunGasRollback(world0, kit, auth, auth, gas::kDuelTicks, 10u, &corrected2, &diverged2);
        check(!diverged2 && corrected2, "rollback: identical streams -> no divergence, still corrected");

        // SNAPSHOT COMPLETENESS: a snapshot that OMITS cooldowns (or active effects) diverges — the control.
        // The snapshot point is tick 4 (after steps t0..t3): the t2 fireball's cooldown still has 4 ticks
        // (it GATES the t7 rejection) and B's burn still has one firing left — both are LOAD-BEARING state.
        gas::GasWorld mid = world0;
        for (uint32_t t = 0; t < 4u; ++t) gas::StepGas(mid, kit, auth, t);
        const gas::GasWorld snapFull = mid;   // the complete snapshot (a value copy)
        // Resume the FULL snapshot -> must equal the authority.
        gas::GasWorld resumeFull = snapFull;
        for (uint32_t t = 4u; t < gas::kDuelTicks; ++t) gas::StepGas(resumeFull, kit, auth, t);
        check(gas::GasStatesEqual(resumeFull, authority), "snapshot: the COMPLETE snapshot resumes bit-exact");
        // Omit the COOLDOWNS -> the t7 fireball is NOT rejected -> a real gameplay divergence. (The t7-vs-t8
        // hit shift converges in TOTALS by the end, so the gameplay divergence is asserted MID-FLIGHT: B's
        // health differs right after the early hit; the final worlds still differ structurally — the
        // cooldown bookkeeping re-grows in a different order.)
        gas::GasWorld noCds = snapFull;
        for (gas::GasEntity& e : noCds.entities) e.cooldowns.clear();
        gas::GasWorld authTrack = snapFull;
        bool midDiverged = false;
        for (uint32_t t = 4u; t < gas::kDuelTicks; ++t) {
            gas::StepGas(noCds, kit, auth, t);
            gas::StepGas(authTrack, kit, auth, t);
            if (noCds.entities[1].attrs.base[gas::kAttrHealth] !=
                authTrack.entities[1].attrs.base[gas::kAttrHealth]) midDiverged = true;
        }
        check(midDiverged, "snapshot: without cooldowns the t7 fireball lands early (mid-flight divergence)");
        check(!gas::GasStatesEqual(noCds, authority), "snapshot: OMITTING cooldowns diverges (the control)");
        // Omit the ACTIVE EFFECTS -> the burn's remaining firing vanishes -> divergence.
        gas::GasWorld noFx = snapFull;
        for (gas::GasEntity& e : noFx.entities) { e.active.clear(); gas::RecomputeCurrents(e); }
        for (uint32_t t = 4u; t < gas::kDuelTicks; ++t) gas::StepGas(noFx, kit, auth, t);
        check(!gas::GasStatesEqual(noFx, authority), "snapshot: OMITTING active effects diverges (the control)");
    }

    if (g_fail == 0) std::printf("ability_test: ALL PASS\n");
    else             std::printf("ability_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
