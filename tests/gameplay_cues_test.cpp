// Slice GC1 — DETERMINISTIC ABILITY TARGETING SHAPES + GAMEPLAY CUES (hf::game::cues). gameplay_cues.h is a
// NEW additive header #including game/gameplay_tags.h READ-ONLY / BYTE-UNTOUCHED (which pulls game/ability.h +
// game/verdict.h read-only). PURE CPU INTEGER (Q16.16 positions; exact integer overlap tests; the cone
// half-angle is a PINNED host cos threshold vs an integer normalized dot — NO runtime transcendental).
//
// What this test PINS (the spec's proofs):
//   (a) SPHERE: in / out / boundary (dist²==radius² INCLUSIVE) / radius-0 (only the center point).
//   (b) BOX: inside / outside / face-boundary (|delta|==half INCLUSIVE), axis-aligned AABB.
//   (c) CONE: within/outside range + angle; the cosHalfAngle boundary (dot==cos INCLUSIVE); at-apex
//       degenerate (INCLUDED); a target directly behind is excluded.
//   (d) TAG FILTER: friendly-fire OFF excludes team-tagged entities (incl. the caster on the sphere boundary).
//   (e) AREA APPLY: a sphere fireball damages the N enemies inside (tag-gated) and NOT the allies/outside;
//       per-entity health trace + the cue stream.
//   (f) CUES: the cue event stream for the battle (order + tags + ticks); impact/buff/death cues at the exact
//       ticks; CollectCues filters by tick.
//   (g) LOCKSTEP: a peer re-derives the target sets + cue stream bit-for-bit; rollback corrects a mispredicted
//       area hit; snapshot completeness (dropping entity positions -> wrong targets -> divergence).
//   (h) COMPAT: a single-target shape behaves exactly as the GAS1/GT1 single-target path.
//   The pinned digests are IDENTICAL under MSVC and local clang (the cross-compiler anchor).
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "game/gameplay_cues.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace cues = hf::game::cues;
namespace tags = hf::game::tags;
namespace gas  = hf::game::gas;
using cues::fx;
using cues::kOne;
using cues::FxVec3;
using cues::EntityId;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- (a) SPHERE: in / out / boundary (inclusive) / radius-0 --------------------------------------------
    {
        const cues::SphereTarget s{FxVec3{2 * kOne, 0, 0}, 2 * kOne};
        check( cues::InSphere(s, FxVec3{2 * kOne, 0, 0}),          "sphere: center is IN");
        check( cues::InSphere(s, FxVec3{3 * kOne, 0, 0}),          "sphere: 1 unit away (< r) is IN");
        check( cues::InSphere(s, FxVec3{0, 0, 0}),                 "sphere: EXACTLY on boundary (dist==r) INCLUSIVE");
        check( cues::InSphere(s, FxVec3{4 * kOne, 0, 0}),          "sphere: opposite boundary INCLUSIVE");
        check(!cues::InSphere(s, FxVec3{4 * kOne + 1, 0, 0}),      "sphere: 1 unit past boundary is OUT");
        check(!cues::InSphere(s, FxVec3{8 * kOne, 0, 0}),          "sphere: far is OUT");
        const cues::SphereTarget z{FxVec3{2 * kOne, 0, 0}, 0};
        check( cues::InSphere(z, FxVec3{2 * kOne, 0, 0}),          "sphere r=0: ONLY the exact center is IN");
        check(!cues::InSphere(z, FxVec3{2 * kOne + 1, 0, 0}),      "sphere r=0: one unit off-center is OUT");
    }

    // ---- (b) BOX: inside / outside / face-boundary (inclusive), axis-aligned ------------------------------
    {
        const fx h = 3 * kOne / 2;   // 1.5
        const cues::BoxTarget b{FxVec3{2 * kOne, 0, 0}, FxVec3{h, h, h}};
        check( cues::InBox(b, FxVec3{2 * kOne, 0, 0}),                    "box: center is IN");
        check( cues::InBox(b, FxVec3{2 * kOne + h, 0, 0}),               "box: +x face (|dx|==half) INCLUSIVE");
        check( cues::InBox(b, FxVec3{2 * kOne, h, 0}),                    "box: +y face INCLUSIVE");
        check( cues::InBox(b, FxVec3{2 * kOne - h, -h, 0}),              "box: corner INCLUSIVE");
        check(!cues::InBox(b, FxVec3{2 * kOne + h + 1, 0, 0}),           "box: 1 unit past +x face is OUT");
        check(!cues::InBox(b, FxVec3{2 * kOne, 0, h + 1}),               "box: past +z face is OUT (3D AABB)");
        check(!cues::InBox(b, FxVec3{8 * kOne, 0, 0}),                    "box: far is OUT");
    }

    // ---- (c) CONE: range + angle; cos boundary (inclusive); at-apex; behind -------------------------------
    {
        // A hemisphere cone (cosHalfAngle == 0 -> 90 deg half-angle), dir +X, range 9.
        const cues::ConeTarget hemi{FxVec3{0, 0, 0}, FxVec3{kOne, 0, 0}, 9 * kOne, 0};
        check( cues::InCone(hemi, FxVec3{kOne, 0, 0}),        "cone(hemi): on-axis, in range -> IN (dot==kOne)");
        check( cues::InCone(hemi, FxVec3{0, kOne, 0}),        "cone(hemi): at 90deg (dot==0==cos) INCLUSIVE");
        check( cues::InCone(hemi, FxVec3{0, 0, 0}),           "cone(hemi): AT-APEX degenerate -> INCLUDED");
        check(!cues::InCone(hemi, FxVec3{-kOne, 0, 0}),       "cone(hemi): directly BEHIND (dot<0) -> OUT");
        check(!cues::InCone(hemi, FxVec3{10 * kOne, 0, 0}),   "cone(hemi): on-axis but OUT of range -> OUT");
        // A 60-degree half-angle cone (cos60 == 0.5 == 32768).
        const cues::ConeTarget c60{FxVec3{0, 0, 0}, FxVec3{kOne, 0, 0}, 9 * kOne, 32768};
        check( cues::InCone(c60, FxVec3{kOne, 0, 0}),         "cone(60): on-axis (dot==kOne>=cos) -> IN");
        check(!cues::InCone(c60, FxVec3{0, kOne, 0}),         "cone(60): at 90deg (dot==0<cos60) -> OUT");
        check( cues::InCone(c60, FxVec3{4 * kOne, kOne, 0}),  "cone(60): shallow off-axis, in range -> IN");
    }

    // ---- (d) TAG FILTER: friendly-fire OFF excludes team-tagged (incl. the boundary caster) ---------------
    {
        const tags::TagRegistry reg = cues::MakeCuesRegistry();
        const cues::CuesWorld w = cues::MakeBattleWorld(reg);
        // Sphere over (2,0) r2: geometrically catches e1(0,0 boundary), e2, e3, e4, e5(ally) — but NOT e6(far).
        cues::TargetShape noFilter = cues::MakeSphereShape(cues::BattleSphere().center, cues::BattleSphere().radiusQ);
        const std::vector<EntityId> allHits = cues::ResolveTargets(w, cues::MakeCuesTagRules(), reg, noFilter);
        check(allHits.size() == 5u, "filter: no filter -> 5 in the sphere (e1..e5; e6 far excluded)");
        check(allHits.front() == 1u && allHits.back() == 5u, "filter: hits are ASCENDING id (e1..e5)");
        // Friendly-fire OFF (exclude Team.Ally): drops e1 (boundary, ally) and e5 (ally).
        const std::vector<EntityId> ffHits = cues::ResolveTargets(w, cues::MakeCuesTagRules(), reg, cues::SphereCast(reg));
        check(ffHits.size() == 3u, "filter: friendly-fire OFF -> only the 3 enemies (e2,e3,e4)");
        check(ffHits[0] == 2u && ffHits[1] == 3u && ffHits[2] == 4u, "filter: exactly {e2,e3,e4} ascending");
    }

    // ---- (e) AREA APPLY: sphere fireball damages the enemies inside, not allies/outside -------------------
    {
        const tags::TagRegistry reg = cues::MakeCuesRegistry();
        const gas::AbilityKit   kit = cues::MakeCuesKit();
        const tags::TagRules    trul = cues::MakeCuesTagRules();
        const cues::CueRules    cr  = cues::MakeCueRules(reg);
        cues::CuesWorld w = cues::MakeBattleWorld(reg);
        const cues::AreaResult r = cues::AreaActivate(w, kit, trul, cr, reg, 1u, cues::kAbFireball, cues::SphereCast(reg));
        check(r.outcome == cues::kAreaOk, "area: sphere fireball activates OK");
        check(r.hits.size() == 3u, "area: 3 enemies hit (e2,e3,e4)");
        check(r.effectsApplied == 6u, "area: 2 effects x 3 targets applied (hit + burn each)");
        // e2,e3,e4 took the instant -30 to BASE (50 -> 20); e1,e5 (allies) + e6 (far) UNCHANGED at 50.
        auto baseHp = [&](EntityId id) {
            const int i = gas::FindEntity(w.tw.gas, id);
            return (int)(w.tw.gas.entities[(size_t)i].attrs.base[gas::kAttrHealth] >> cues::kFrac);
        };
        check(baseHp(2u) == 20 && baseHp(3u) == 20 && baseHp(4u) == 20, "area: enemies e2/e3/e4 at 20 hp (-30)");
        check(baseHp(1u) == 50 && baseHp(5u) == 50, "area: allies e1/e5 UNTOUCHED (friendly-fire off)");
        check(baseHp(6u) == 50, "area: far enemy e6 UNTOUCHED (outside the sphere)");
        // One impact cue per hit (3), no death yet, no self cue.
        check(w.cueLog.size() == 3u, "area: 3 impact cues (one per hit)");
        for (const cues::CueEvent& ce : w.cueLog)
            check(ce.cueTag == cues::CueImpactFire(reg) && ce.magnitude == 30 * kOne,
                  "area: cue is Cue.Impact.Fire, magnitude 30");
        // Mana paid ONCE (100 - 15 = 85), not per-target.
        const int ci = gas::FindEntity(w.tw.gas, 1u);
        check((int)(w.tw.gas.entities[(size_t)ci].attrs.base[gas::kAttrMana] >> cues::kFrac) == 85,
              "area: caster mana paid ONCE (100->85), not per target");
    }

    // ---- (h) COMPAT: a single-target shape == the GAS1/GT1 single-target path -----------------------------
    {
        const tags::TagRegistry reg = cues::MakeCuesRegistry();
        const gas::AbilityKit   kit = cues::MakeCuesKit();
        const tags::TagRules    trul = cues::MakeCuesTagRules();
        const cues::CueRules    cr  = cues::MakeCueRules(reg);
        cues::CuesWorld w = cues::MakeBattleWorld(reg);
        // Single-target fireball at e2 only.
        const cues::AreaResult r = cues::AreaActivate(w, kit, trul, cr, reg, 1u, cues::kAbFireball, cues::MakeSingleShape(2u));
        check(r.outcome == cues::kAreaOk && r.hits.size() == 1u && r.hits[0] == 2u, "compat: single-target hits only e2");
        const int i2 = gas::FindEntity(w.tw.gas, 2u);
        check((int)(w.tw.gas.entities[(size_t)i2].attrs.base[gas::kAttrHealth] >> cues::kFrac) == 20,
              "compat: e2 took -30 (single-target)");
        const int i3 = gas::FindEntity(w.tw.gas, 3u);
        check((int)(w.tw.gas.entities[(size_t)i3].attrs.base[gas::kAttrHealth] >> cues::kFrac) == 50,
              "compat: e3 untouched (single-target)");
    }

    // ---- (f) CUES: the battle cue stream — order, tags, ticks; impact/buff/death at the exact ticks -------
    {
        const tags::TagRegistry reg = cues::MakeCuesRegistry();
        const cues::BattleRun run = cues::RunCuesBattle();
        // The impact cues at t1 (3), the shield self cue at t2 (1), the box-fireball impacts + deaths at t3.
        const std::vector<cues::CueEvent> t1 = cues::CollectCues(run.finalWorld, 1u);
        check(t1.size() == 3u, "cues: 3 impact cues at t1 (the sphere fireball)");
        for (const cues::CueEvent& ce : t1)
            check(ce.cueTag == cues::CueImpactFire(reg) && ce.tick == 1u, "cues: t1 cues are Cue.Impact.Fire");
        const std::vector<cues::CueEvent> t2 = cues::CollectCues(run.finalWorld, 2u);
        check(t2.size() == 1u && t2[0].cueTag == cues::CueBuffShield(reg) && t2[0].targetEntity == 1u,
              "cues: t2 -> one Cue.Buff.Shield on the caster");
        // t3 box fireball hits the 3 enemies; -30 on their remaining hp kills them -> death cues fire.
        const std::vector<cues::CueEvent> t3 = cues::CollectCues(run.finalWorld, 3u);
        uint32_t t3deaths = 0, t3impacts = 0;
        for (const cues::CueEvent& ce : t3) {
            if (ce.cueTag == cues::CueDeath(reg)) ++t3deaths;
            if (ce.cueTag == cues::CueImpactFire(reg)) ++t3impacts;
        }
        check(t3impacts == 3u && t3deaths == 3u, "cues: t3 -> 3 impacts + 3 deaths (the box fireball kills)");
        // The far enemy e6 is hit by the cone at t6.
        const std::vector<cues::CueEvent> t6 = cues::CollectCues(run.finalWorld, 6u);
        bool coneHitE6 = false;
        for (const cues::CueEvent& ce : t6)
            if (ce.cueTag == cues::CueImpactFire(reg) && ce.targetEntity == 6u) coneHitE6 = true;
        check(coneHitE6, "cues: t6 cone scorch impacts the far enemy e6");
        check(run.deaths >= 3u, "cues: at least the 3 box-fireball deaths occurred");
    }

    // ---- (f2) PINNED DIGESTS (MSVC == clang; captured on first run) ---------------------------------------
    {
        const tags::TagRegistry reg = cues::MakeCuesRegistry();
        const uint64_t kitDigest = gas::DigestKit(cues::MakeCuesKit());
        const cues::BattleRun run = cues::RunCuesBattle();
        const uint64_t battleDigest = cues::DigestCuesWorld(run.finalWorld);
        std::printf("gc1: registry tags=%u kit=%s\n", reg.size(), gas::DigestHex(kitDigest).c_str());
        std::printf("gc1: battle trace=%s final=%s (shapes=%u hits=%u cues=%u deaths=%u)\n",
                    cues::DigestHex(run.traceDigest).c_str(), cues::DigestHex(battleDigest).c_str(),
                    run.shapesCast, run.targetsHit, run.cuesTotal, run.deaths);
        const uint64_t kPinnedKit    = 0x234557dd3a5f7627ull;  // PINNED on first run (MSVC == clang)
        const uint64_t kPinnedTrace  = 0x15ccdaedad5caf1cull;  // PINNED on first run (MSVC == clang)
        const uint64_t kPinnedBattle = 0xf80ce1458fe4f432ull;  // PINNED on first run (MSVC == clang)
        check(kitDigest    == kPinnedKit,    "digest: kit matches the pinned value");
        check(run.traceDigest == kPinnedTrace, "digest: battle trace matches the pinned value");
        check(battleDigest == kPinnedBattle, "digest: final battle state matches the pinned value");
    }

    // ---- (g) LOCKSTEP: a peer re-derives the target sets + cue stream bit-for-bit -------------------------
    {
        const tags::TagRegistry reg = cues::MakeCuesRegistry();
        const gas::AbilityKit   kit = cues::MakeCuesKit();
        const tags::TagRules    trul = cues::MakeCuesTagRules();
        const cues::CueRules    cr  = cues::MakeCueRules(reg);
        const std::vector<cues::AreaCommand> stream = cues::MakeBattleStream(reg);
        bool identical = false;
        const cues::CuesWorld authority = cues::RunCuesLockstep(cues::MakeBattleWorld(reg), kit, trul, cr, reg,
                                                                stream, cues::kBattleTicks, &identical);
        check(identical, "lockstep: two peers fed only the command stream are byte-identical");
        // The lockstep authority equals the RunCuesBattle final (the inline-apply path == StepCues path).
        const cues::BattleRun run = cues::RunCuesBattle();
        check(cues::CuesStatesEqual(authority, run.finalWorld), "lockstep: authority == the battle final");

        // ROLLBACK: mispredict the t3 box fireball as a self-shield (no enemies hit -> the deaths never fire),
        // then correct. The misprediction MUST genuinely diverge and the correction MUST be bit-exact.
        std::vector<cues::AreaCommand> mispredict = stream;
        for (cues::AreaCommand& c : mispredict)
            if (c.tick == 3u) { c.abilityId = cues::kAbShield; c.shape = cues::MakeSingleShape(1u); }
        bool corrected = false, diverged = false;
        (void)cues::RunCuesRollback(cues::MakeBattleWorld(reg), kit, trul, cr, reg, stream, mispredict,
                                    cues::kBattleTicks, 3u, &corrected, &diverged);
        check(diverged, "rollback: the mispredicted area activation genuinely diverged (non-vacuous)");
        check(corrected, "rollback: the corrected re-sim is bit-exact to the authority");

        // SNAPSHOT COMPLETENESS: dropping the entity POSITIONS (a wrong-targets divergence) must NOT match.
        cues::CuesWorld broken = cues::MakeBattleWorld(reg);
        for (cues::CueEntity& e : broken.ents) e.pos = FxVec3{0, 0, 0};   // collapse all to the origin
        bool brokenIdentical = false;
        const cues::CuesWorld brokenAuth = cues::RunCuesLockstep(broken, kit, trul, cr, reg, stream,
                                                                 cues::kBattleTicks, &brokenIdentical);
        check(brokenIdentical, "rollback: the broken world is still internally deterministic");
        check(!cues::CuesStatesEqual(brokenAuth, authority),
              "snapshot: dropping positions changes the resolved targets -> DIVERGES (completeness control)");
    }

    if (g_fail == 0) std::printf("gameplay_cues_test: ALL PASS\n");
    else             std::printf("gameplay_cues_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
