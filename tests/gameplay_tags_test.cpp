// Slice GT1 — A DETERMINISTIC GAMEPLAY-TAG LAYER (hierarchical tag containers gating abilities/effects,
// hf::game::tags). gameplay_tags.h is a NEW additive header #including game/ability.h READ-ONLY / BYTE-
// UNTOUCHED (the tag gates COMPOSE onto GAS1 through a thin adapter). PURE CPU INTEGER (tags interned to
// stable uint32 ids; no float, no wall clock).
//
// What this test PINS (the spec's proofs):
//   (a) INTERN / HIERARCHY: registry ids stable+pinned (auto-interned prefixes); "A.B.C" query-matches "A"
//       and "A.B" but NOT "A.B.D" or "A.BB" (SEGMENT-boundary matching — the "A.BB" false-positive guard);
//       HasTagExact vs HasTagQuery distinction.
//   (b) CONTAINER: Add/Remove/HasAny/HasAll (exact + query); dedup; the sorted-order digest.
//   (c) GATING: requiredTags gate activation (pass/fail + reason); blockedTags block; effect immunityTags
//       nullify application; effect grantsTags add to the target and are removed on expiry (the exact tick).
//   (d) THE SCENARIO: the FIXED 14-tick tag skirmish (stun/immune/empower) — the full tag-state + attribute
//       trace digest + the exact finals; the stunned unit's activation fails at the exact tick and recovers
//       when the stun expires; blocked-before-missing-required precedence.
//   (e) LOCKSTEP: a peer re-derives the tag-gated outcomes bit-for-bit; rollback corrects a genuinely-
//       diverged mispredicted activation; snapshot completeness (omitting the owned tag container diverges).
//   (f) The pinned digests are IDENTICAL under MSVC and local clang (the cross-compiler anchor).
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "game/gameplay_tags.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace tags = hf::game::tags;
namespace gas  = hf::game::gas;
using tags::fx;
using tags::kOne;
using tags::TagId;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- (a) INTERN / HIERARCHY: stable pinned ids; segment-boundary query matching; exact vs query ----
    {
        const tags::TagRegistry reg = tags::MakeTagRegistry();
        // The pinned ids (insertion order, prefixes auto-interned shortest-first).
        check(reg.Find("State")                == 0u, "intern: State == id 0");
        check(reg.Find("State.Stunned")        == 1u, "intern: State.Stunned == id 1");
        check(reg.Find("State.Empowered")      == 2u, "intern: State.Empowered == id 2");
        check(reg.Find("State.Burning")        == 3u, "intern: State.Burning == id 3");
        check(reg.Find("Immune")               == 4u, "intern: Immune == id 4");
        check(reg.Find("Immune.Fire")          == 5u, "intern: Immune.Fire == id 5");
        check(reg.Find("Ability")              == 6u, "intern: Ability == id 6 (auto-interned prefix)");
        check(reg.Find("Ability.Fire")         == 7u, "intern: Ability.Fire == id 7 (auto-interned prefix)");
        check(reg.Find("Ability.Fire.Fireball")== 8u, "intern: Ability.Fire.Fireball == id 8");
        check(reg.size() == 9u, "intern: exactly 9 tags interned");
        tags::TagRegistry regCopy = reg;   // Intern mutates -> a mutable copy for the idempotence check
        check(regCopy.Intern("State.Stunned") == 1u && regCopy.size() == 9u,
              "intern: idempotent (re-intern returns the same id, no new tag)");

        // Hierarchy: a container holding the SPECIFIC tag query-matches its PARENTS.
        tags::TagContainer c;
        c.Add(reg.Find("Ability.Fire.Fireball"));
        check(tags::HasTagQuery(reg, c, reg.Find("Ability")),               "query: Fireball matches Ability");
        check(tags::HasTagQuery(reg, c, reg.Find("Ability.Fire")),          "query: Fireball matches Ability.Fire");
        check(tags::HasTagQuery(reg, c, reg.Find("Ability.Fire.Fireball")), "query: Fireball matches itself");
        check(!tags::HasTagQuery(reg, c, reg.Find("State")),                "query: Fireball does NOT match State");
        // Exact vs query: the container does NOT EXACTLY hold the parents.
        check(c.HasTagExact(reg.Find("Ability.Fire.Fireball")),  "exact: holds Fireball exactly");
        check(!c.HasTagExact(reg.Find("Ability")),               "exact: does NOT hold Ability exactly (query-only)");
        check(!c.HasTagExact(reg.Find("Ability.Fire")),          "exact: does NOT hold Ability.Fire exactly");

        // THE SEGMENT-BOUNDARY FALSE-POSITIVE GUARD: raw-string prefixes must NOT match.
        tags::TagRegistry r2;
        const TagId abc = r2.Intern("A.B.C");    // interns A, A.B, A.B.C
        const TagId abb = r2.Intern("A.BB");     // interns A.BB (A already exists)
        const TagId abd = r2.Intern("A.B.D");    // interns A.B.D (A.B already exists)
        tags::TagContainer holdsABC; holdsABC.Add(abc);
        check(tags::HasTagQuery(r2, holdsABC, r2.Find("A")),   "guard: A.B.C matches query A");
        check(tags::HasTagQuery(r2, holdsABC, r2.Find("A.B")), "guard: A.B.C matches query A.B");
        check(!tags::HasTagQuery(r2, holdsABC, abd),           "guard: A.B.C does NOT match A.B.D (sibling)");
        check(!tags::HasTagQuery(r2, holdsABC, abb),           "guard: A.B.C does NOT match A.BB (segment boundary)");
        // The reverse direction is the crux: owning "A.BB", a raw prefix of "A.B", must NOT match query "A.B".
        tags::TagContainer holdsABB; holdsABB.Add(abb);
        check(tags::HasTagQuery(r2, holdsABB, r2.Find("A")),   "guard: A.BB matches query A");
        check(!tags::HasTagQuery(r2, holdsABB, r2.Find("A.B")),"guard: A.BB does NOT match query A.B (\"A.B\" is a raw prefix, NOT a segment ancestor)");
    }

    // ---- (b) CONTAINER: Add/Remove/HasAny/HasAll; dedup; sorted order digest ----
    {
        const tags::TagRegistry reg = tags::MakeTagRegistry();
        tags::TagContainer c;
        const TagId stun = reg.Find("State.Stunned"), emp = reg.Find("State.Empowered"),
                    burn = reg.Find("State.Burning"), imm = reg.Find("Immune.Fire");
        c.Add(burn); c.Add(stun); c.Add(emp); c.Add(stun);   // stun added twice -> dedup
        check(c.size() == 3u, "container: dedup (3 unique of 4 adds)");
        // Sorted ascending by construction (ids 1,2,3).
        check(c.ids[0] == 1u && c.ids[1] == 2u && c.ids[2] == 3u, "container: sorted ascending order");
        check(c.HasTagExact(stun) && c.HasTagExact(emp) && c.HasTagExact(burn), "container: HasTagExact members");
        check(!c.HasTagExact(imm), "container: non-member absent");
        c.Remove(emp);
        check(c.size() == 2u && !c.HasTagExact(emp) && c.HasTagExact(stun), "container: Remove drops exactly one");
        c.Remove(imm);   // removing an absent tag is a no-op
        check(c.size() == 2u, "container: Remove of absent tag is a no-op");
        // HasAny / HasAll (exact).
        tags::TagContainer q1; q1.Add(imm); q1.Add(stun);       // {Immune.Fire, Stunned}
        check(tags::HasAnyExact(c, q1),  "container: HasAnyExact (Stunned present)");
        check(!tags::HasAllExact(c, q1), "container: !HasAllExact (Immune.Fire absent)");
        tags::TagContainer q2; q2.Add(stun); q2.Add(burn);      // {Stunned, Burning} both in c
        check(tags::HasAllExact(c, q2), "container: HasAllExact (both present)");
        // Deterministic order digest: two independently-built equal sets hash identically.
        tags::TagContainer d1; d1.Add(stun); d1.Add(burn);
        tags::TagContainer d2; d2.Add(burn); d2.Add(stun);      // reversed insertion order
        check(d1.ids == d2.ids, "container: insertion order does NOT affect the sorted representation");
    }

    // ---- (c) GATING: required/blocked gate activation; immunity nullifies; grants add + expire ----
    {
        const tags::TagRegistry reg = tags::MakeTagRegistry();
        const gas::AbilityKit kit = tags::MakeSkirmishKit();
        const tags::TagRules rules = tags::MakeSkirmishRules(reg);

        // requiredTags: Smite REQUIRES State.Empowered. Without it -> kTagMissingRequiredTag (mutates nothing).
        {
            tags::TaggedWorld tw = tags::MakeSkirmishWorld(reg);   // Hero 1 (no tags), Villain 2 (Immune.Fire)
            const gas::fx vHp0 = tw.gas.entities[1].attrs.base[gas::kAttrHealth];
            check(tags::TryActivateTagged(tw, kit, rules, reg, 1u, tags::kAbSmite, 2u)
                      == tags::kTagMissingRequiredTag, "gating: Smite without Empowered -> kTagMissingRequiredTag");
            check(tw.gas.entities[1].attrs.base[gas::kAttrHealth] == vHp0, "gating: missing-required mutates nothing");
            // Grant State.Empowered (via the empower effect) -> Smite now PASSES.
            check(tags::TryActivateTagged(tw, kit, rules, reg, 1u, tags::kAbEmpower, 1u) == tags::kTagOk,
                  "gating: Empower activates");
            check(tags::HasTagQuery(reg, tags::EffectiveTags(tw, rules, 1u), reg.Find("State.Empowered")),
                  "gating: Hero effectively Empowered after the empower effect");
            check(tags::TryActivateTagged(tw, kit, rules, reg, 1u, tags::kAbSmite, 2u) == tags::kTagOk,
                  "gating: Smite WITH Empowered -> kTagOk");
            check(tw.gas.entities[1].attrs.base[gas::kAttrHealth] == vHp0 - 30 * kOne,
                  "gating: Smite landed -30.0 exactly");
        }
        // blockedTags: State.Stunned blocks activation.
        {
            tags::TaggedWorld tw = tags::MakeSkirmishWorld(reg);
            // Villain stuns Hero -> Hero effectively Stunned.
            check(tags::TryActivateTagged(tw, kit, rules, reg, 2u, tags::kAbStun, 1u) == tags::kTagOk,
                  "gating: Stun activates");
            check(tags::HasTagQuery(reg, tags::EffectiveTags(tw, rules, 1u), reg.Find("State.Stunned")),
                  "gating: Hero effectively Stunned");
            check(tags::TryActivateTagged(tw, kit, rules, reg, 1u, tags::kAbFire, 2u) == tags::kTagBlockedByTag,
                  "gating: a stunned Hero's ability is kTagBlockedByTag");
        }
        // immunity: Immune.Fire nullifies BOTH fire effects on the immune Villain (no damage).
        {
            tags::TaggedWorld tw = tags::MakeSkirmishWorld(reg);
            const gas::fx vHp0 = tw.gas.entities[1].attrs.base[gas::kAttrHealth];
            uint32_t imm = 0;
            const tags::TaggedResult r = tags::TryActivateTagged(tw, kit, rules, reg, 1u, tags::kAbFire, 2u, &imm);
            check(r == tags::kTagImmune, "gating: Fire on an Immune.Fire target -> kTagImmune (committed)");
            check(imm == 2u, "gating: BOTH fire effects nullified (immuneCount == 2)");
            check(tw.gas.entities[1].attrs.base[gas::kAttrHealth] == vHp0, "gating: the immune target took NO damage");
            // A NON-immune target (Hero) takes the fire damage + gains State.Burning.
            const gas::fx hHp0 = tw.gas.entities[0].attrs.base[gas::kAttrHealth];
            uint32_t imm2 = 0;
            check(tags::TryActivateTagged(tw, kit, rules, reg, 2u, tags::kAbFire, 1u, &imm2) == tags::kTagOk,
                  "gating: Fire on a non-immune target -> kTagOk");
            check(imm2 == 0u, "gating: no effects nullified on the non-immune target");
            check(tw.gas.entities[0].attrs.base[gas::kAttrHealth] == hHp0 - 20 * kOne, "gating: non-immune -20.0 hit landed");
            check(tags::HasTagQuery(reg, tags::EffectiveTags(tw, rules, 1u), reg.Find("State.Burning")),
                  "gating: the fire's burn effect grants State.Burning");
        }
        // grants add + EXPIRE on the exact tick: the empower effect (dur 4) grants State.Empowered for 4
        // steps, then the grant vanishes the tick GAS1 removes the expired effect.
        {
            tags::TaggedWorld tw = tags::MakeSkirmishWorld(reg);
            const TagId emp = reg.Find("State.Empowered");
            check(tags::TryActivateTagged(tw, kit, rules, reg, 1u, tags::kAbEmpower, 1u) == tags::kTagOk,
                  "grants: empower activates");
            for (int step = 1; step <= 4; ++step) {
                check(tags::HasTagQuery(reg, tags::EffectiveTags(tw, rules, 1u), emp),
                      "grants: Empowered present while the effect is live");
                gas::StepAbilities(tw.gas);
            }
            check(!tags::HasTagQuery(reg, tags::EffectiveTags(tw, rules, 1u), emp),
                  "grants: Empowered REMOVED on the exact expiry tick (step 4)");
        }
    }

    // ---- (d) THE SCENARIO: the FIXED 14-tick tag skirmish ----
    {
        const tags::TagRegistry reg = tags::MakeTagRegistry();
        const uint64_t regDigest = [&]{
            hf::game::verdict::DigestFnv d; d.mix32(reg.size());
            for (uint32_t i = 0; i < reg.size(); ++i) { for (TagId a : reg.ancestors[i]) d.mix32(a); d.sep(); }
            return d.h;
        }();
        const uint64_t rulesDigest = tags::DigestRules(tags::MakeSkirmishRules(reg));
        const uint64_t kitDigest   = gas::DigestKit(tags::MakeSkirmishKit());
        std::printf("gt1: registry digest %s\n", tags::DigestHex(regDigest).c_str());
        std::printf("gt1: rules digest %s\n", tags::DigestHex(rulesDigest).c_str());
        std::printf("gt1: kit digest %s\n", tags::DigestHex(kitDigest).c_str());

        const tags::SkirmishRun run = tags::RunSkirmish();
        std::printf("gt1: skirmish trace digest %s, ok %u, blocked %u, missReq %u, immune %u, effectsApplied %u\n",
                    tags::DigestHex(run.traceDigest).c_str(), run.activationsOk, run.activationsBlocked,
                    run.activationsMissReq, run.activationsImmune, run.effectsApplied);
        std::printf("gt1: final world digest %s\n", tags::DigestHex(tags::DigestTaggedWorld(run.finalWorld)).c_str());

        // The outcome tallies (the scripted skirmish structure).
        check(run.activationsOk == 5u,      "skirmish: 5 committed activations (incl. the immune one)");
        check(run.activationsBlocked == 2u, "skirmish: 2 tag-blocked activations (t1, t2)");
        check(run.activationsMissReq == 1u, "skirmish: 1 missing-required rejection (t4)");
        check(run.activationsImmune == 1u,  "skirmish: 1 immune (nullified) activation (t3)");
        check(run.effectsApplied == 8u,     "skirmish: 8 effect applications (5 committed-effect + 3 periodic)");

        // The exact final integers.
        const gas::GasEntity& H = run.finalWorld.gas.entities[0];
        const gas::GasEntity& V = run.finalWorld.gas.entities[1];
        check(H.attrs.base[gas::kAttrHealth] == 74 * kOne, "skirmish: Hero final health == 74.0 (80 - 6 burn)");
        check(V.attrs.base[gas::kAttrHealth] == 70 * kOne, "skirmish: Villain final health == 70.0 (only the smite; fire immune)");
        // Villain keeps its innate immunity; Hero owns no persistent tags (empower was effect-granted).
        check(run.finalWorld.tags[1].owned.HasTagExact(reg.Find("Immune.Fire")), "skirmish: Villain retains innate Immune.Fire");
        check(run.finalWorld.tags[0].owned.empty(), "skirmish: Hero owns no persistent tags (effect-granted expired)");

        // The stunned unit's activation fails at the exact tick and recovers when the stun expires.
        check(run.samples[1].stunned[0] == 1 && run.samples[1].castBlocked[0] == 1,
              "skirmish: Hero stunned + blocked at t1");
        check(run.samples[2].castBlocked[0] == 1, "skirmish: Hero still blocked at t2 (blocked-before-missing-required)");
        check(run.samples[2].stunned[0] == 0, "skirmish: the stun effect expired during step t2");
        check(run.samples[3].castImmune[0] == 1, "skirmish: Hero's t3 fire is immune-nullified on the immune Villain");
        check(run.samples[3].health[1] == 100 * kOne, "skirmish: Villain still full health after the immune fire (t3)");
        check(run.samples[4].castMissReq[0] == 1, "skirmish: Hero's t4 smite fails (not yet Empowered)");
        check(run.samples[6].empowered[0] == 1 && run.samples[6].castOk[0] == 1,
              "skirmish: Hero Empowered + Smite lands at t6");
        check(run.samples[6].health[1] == 70 * kOne, "skirmish: Villain to 70 after the t6 smite");
        check(run.samples[7].burning[0] == 1, "skirmish: Hero Burning after the t7 fire (non-immune)");

        // The pinned digests (PINNED on first run — MSVC == clang).
        const uint64_t kPinnedRegDigest    = 0x6f1894cac0e3da86ull;
        const uint64_t kPinnedRulesDigest  = 0xe29d927a9af23da5ull;
        const uint64_t kPinnedKitDigest    = 0xf2eadd15ad66f011ull;
        const uint64_t kPinnedTraceDigest  = 0x807d1bc4f5e9c28full;
        const uint64_t kPinnedFinalDigest  = 0x3042b61b3766a817ull;
        check(regDigest   == kPinnedRegDigest,   "skirmish: registry digest matches the pinned value");
        check(rulesDigest == kPinnedRulesDigest, "skirmish: rules digest matches the pinned value");
        check(kitDigest   == kPinnedKitDigest,   "skirmish: kit digest matches the pinned value");
        check(run.traceDigest == kPinnedTraceDigest, "skirmish: trace digest matches the pinned value");
        check(tags::DigestTaggedWorld(run.finalWorld) == kPinnedFinalDigest, "skirmish: final-world digest matches the pinned value");

        // Two full runs byte-identical.
        const tags::SkirmishRun run2 = tags::RunSkirmish();
        check(tags::TaggedStatesEqual(run.finalWorld, run2.finalWorld) && run2.traceDigest == run.traceDigest,
              "skirmish: two runs are byte-identical");
    }

    // ---- (e) LOCKSTEP + ROLLBACK + SNAPSHOT COMPLETENESS ----
    {
        const tags::TagRegistry reg = tags::MakeTagRegistry();
        const gas::AbilityKit kit = tags::MakeSkirmishKit();
        const tags::TagRules rules = tags::MakeSkirmishRules(reg);
        const std::vector<tags::TagCommand> auth = tags::MakeSkirmishStream();
        const tags::TaggedWorld world0 = tags::MakeSkirmishWorld(reg);

        // LOCKSTEP: a peer fed ONLY the command stream re-derives the tag-gated skirmish bit-for-bit.
        bool identical = false;
        const tags::TaggedWorld authority =
            tags::RunTaggedLockstep(world0, kit, rules, reg, auth, tags::kSkirmishTicks, &identical);
        check(identical, "lockstep: authority == replica from inputs alone (bit-identical)");
        const tags::SkirmishRun run = tags::RunSkirmish();
        check(tags::TaggedStatesEqual(authority, run.finalWorld),
              "lockstep: the lockstep authority == the skirmish-scenario final world");

        // ROLLBACK: a GENUINELY-diverged tag-relevant misprediction — the t6 Smite mispredicted as a Fire
        // (which the immune Villain nullifies) -> Villain never takes the smite -> divergence; corrected.
        std::vector<tags::TagCommand> mispredict = auth;
        for (tags::TagCommand& c : mispredict)
            if (c.tick == 6u) c.abilityId = tags::kAbFire;
        bool corrected = false, diverged = false;
        const tags::TaggedWorld rolled = tags::RunTaggedRollback(world0, kit, rules, reg, auth, mispredict,
                                                                 tags::kSkirmishTicks, 6u, &corrected, &diverged);
        check(diverged, "rollback: the misprediction ACTUALLY diverged (non-vacuous)");
        check(corrected, "rollback: corrected == the straight lockstep authority (bit-exact recovery)");
        check(tags::TaggedStatesEqual(rolled, authority), "rollback: returned world == authority");
        // The non-diverging CONTROL.
        bool corrected2 = false, diverged2 = true;
        (void)tags::RunTaggedRollback(world0, kit, rules, reg, auth, auth, tags::kSkirmishTicks, 6u, &corrected2, &diverged2);
        check(!diverged2 && corrected2, "rollback: identical streams -> no divergence, still corrected");

        // SNAPSHOT COMPLETENESS: the tag container is LOAD-BEARING. Snapshot at tick 3 (after steps t0..t2):
        // Villain's innate Immune.Fire GATES the t3 fire immunity. Dropping the OWNED tags on restore makes
        // the Villain take fire damage -> a real, structural divergence (the control).
        tags::TaggedWorld mid = world0;
        for (uint32_t t = 0; t < 3u; ++t) tags::StepTagged(mid, kit, rules, reg, auth, t);
        const tags::TaggedWorld snapFull = mid;   // the complete snapshot (a value copy)
        // Resume the FULL snapshot -> must equal the authority.
        tags::TaggedWorld resumeFull = snapFull;
        for (uint32_t t = 3u; t < tags::kSkirmishTicks; ++t) tags::StepTagged(resumeFull, kit, rules, reg, auth, t);
        check(tags::TaggedStatesEqual(resumeFull, authority), "snapshot: the COMPLETE snapshot resumes bit-exact");
        // Omit the OWNED TAGS -> the Villain loses its Immune.Fire -> the t3 fire lands -> divergence.
        tags::TaggedWorld noTags = snapFull;
        for (tags::EntityTags& e : noTags.tags) e.owned.ids.clear();
        tags::TaggedWorld authTrack = snapFull;
        bool midDiverged = false;
        for (uint32_t t = 3u; t < tags::kSkirmishTicks; ++t) {
            tags::StepTagged(noTags, kit, rules, reg, auth, t);
            tags::StepTagged(authTrack, kit, rules, reg, auth, t);
            if (noTags.gas.entities[1].attrs.base[gas::kAttrHealth] !=
                authTrack.gas.entities[1].attrs.base[gas::kAttrHealth]) midDiverged = true;
        }
        check(midDiverged, "snapshot: without the owned tags the immune Villain takes fire damage (mid-flight divergence)");
        check(!tags::TaggedStatesEqual(noTags, authority), "snapshot: OMITTING the owned tag container diverges (the control)");
    }

    if (g_fail == 0) std::printf("gameplay_tags_test: ALL PASS\n");
    else             std::printf("gameplay_tags_test: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
