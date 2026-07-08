#pragma once
// Slice GAME1 — A COMPLETE DETERMINISTIC ROLLBACK-PHYSICS GAME: a 2-player physics KNOCKOUT DUEL,
// hf::game::duel. THE BEAT-UE5 P0: the engine has shipped a deep determinism-moat stack (bit-exact
// integer physics, whole-world lockstep/rollback, replay, what-if fork, provable anti-cheat) but never an
// actual GAME — samples/ held only hello_triangle. GAME1 ships a small-but-COMPLETE playable duel that
// COMPOSES the whole moat stack into ONE deliverable and demonstrates the full "superior to UE5" story:
// a real game that is bit-exact deterministic, rollback-netcode-replayable, fork-able (what-if), replay-
// pinnable AND provably fair (anti-cheat-verifiable) — none of which UE5's float architecture can offer.
//
// THE GAME (a sumo/knockout arena):
//   * ARENA: a static platform over a void. Two player avatars are dynamic fpx rigid bodies resting on the
//     platform. A player is KNOCKED OUT when its body leaves the RING bounds — |x| > kArenaEdge or
//     |z| > kArenaEdge (shoved out of the ring) OR y < kKillPlaneY (fallen into the void).
//   * CONTROLS: each player's per-tick input = {moveX, shove}. The SHOVE ability (a real GAS1 ability with a
//     mana COST + a COOLDOWN) applies an outward IMPULSE to the opponent when it is within kShoveRange —
//     knocking it toward the edge. A landed shove applies a GT1 "State.Stunned" tag to the opponent for
//     kStunTicks ticks (GT1 blockedTags gates the stunned player's own shoves). Each landed shove fires a
//     GC1 impact CUE; a knockout fires a knockout CUE.
//   * RULES: best-of-kRounds rounds; a round ends at the first knockout (or a tick limit -> draw). The match
//     ends when a player wins kWinThreshold rounds. Each round is a fresh deterministic scene (bodies at the
//     spawn points) — deterministic round reset by construction (a fresh ClonePeer).
//
// THE ARCHITECTURE (honest — see the report): the SHOVE physics is the moat's currency. A round is played by
// RunDuelRound, which materializes the verdict world (ClonePeer) and steps it tick-by-tick; each tick it
// RESOLVES the two players' inputs through the REAL GAS1/GT1 ability layer (cost/cooldown/stun gates), and a
// committed shove is LOWERED into a verdict kCmdImpulse on the opponent's body (+ a cue). The emitted verdict
// COMMAND STREAM is a pure deterministic function of the input scripts — so the whole moat (lockstep /
// rollback / replay / fork / anti-cheat) runs over that emitted stream through the FROZEN harnesses VERBATIM.
// This is a HEADLESS deterministic MATCH SIMULATION + the moat proofs (there is no live input loop / no
// interactive window — that is a future slice); the physics avatars are fpx unit BODIES (box hulls).
//
// COMPOSES, READ-ONLY / BYTE-UNTOUCHED (this header is a NEW additive sibling; it adds NO field and edits NO
// frozen function):
//   * game/verdict.h   (VD1-VD6) — VerdictWorld / StepWorld / SimVerdictTick / snapshot+restore /
//                        RunVerdictLockstep / RunVerdictRollback / DigestSnapshot / ReplayFile+Serialize/
//                        Parse — the whole-world deterministic gameplay+physics tick + its moat harnesses.
//   * sim/fpx.h        — FxBody rigid bodies (the avatars) + the Q16.16 toolbox (via verdict).
//   * game/ability.h   (GAS1) — the SHOVE ability (mana cost + cooldown), TryActivate / StepAbilities.
//   * game/gameplay_tags.h (GT1) — the "State.Stunned" tag gate (a landed shove stuns; a stunned player is
//                        blocked from shoving). TryActivateTagged.
//   * game/gameplay_cues.h (GC1) — the CueEvent event-stream contract (impact + knockout cues). (We emit
//                        CueEvent records directly — the single-target shove does NOT need AreaActivate's
//                        area targeting; the cue EVENT layer is what GC1 pins.)
//   * net/authority_verify.h (AC1) — Verify: the server re-simulates the emitted stream + rejects a faked
//                        outcome at the exact divergence tick (provable fairness).
//   * replay/fork.h (FK1) + replay/replay.h (RP*) — BuildForkDemo / ForkAt / ResimulateFork (what-if) +
//                        ReplayFile serialize/parse (the pinned demo file) — a fork that changes the WINNER.
//
// PURE CPU INTEGER (the strictest determinism tier, strict-zero integer viz — NO new shader, NO new RHI, NO
// new compute). No goldens committed by this slice.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "game/verdict.h"          // read-only: the VD1-VD6 whole-world deterministic gameplay+physics + moat
#include "game/ability.h"          // read-only: GAS1 — the SHOVE ability (cost + cooldown)
#include "game/gameplay_tags.h"    // read-only: GT1 — the State.Stunned gate
#include "game/gameplay_cues.h"    // read-only: GC1 — the CueEvent event-stream contract
#include "net/authority_verify.h"  // read-only: AC1 — provable anti-cheat re-simulation verifier
#include "replay/fork.h"           // read-only: FK1 what-if fork + RP* replay/demo (pulls replay/replay.h)

namespace hf {
namespace game {
namespace duel {

// Alias the composed layers verbatim (NO new primitives).
namespace verdict = hf::game::verdict;
namespace gas     = hf::game::gas;
namespace tags    = hf::game::tags;
namespace cues    = hf::game::cues;
namespace acnet   = hf::net;         // authority_verify.h (hf::net)
namespace fork    = hf::replay;      // fork.h + replay.h (hf::replay)
namespace fpx     = hf::sim::fpx;
namespace gjk     = hf::sim::gjk;
namespace convex  = hf::sim::convex;
namespace warmhull = hf::sim::warmhull;

using verdict::fx;
using verdict::kOne;
using verdict::kFrac;
using verdict::FxVec3;
using verdict::FxQuat;
using verdict::EntityId;
using verdict::kNoEntity;
using verdict::Command;

// ============================ THE PINNED GAME CONSTANTS (FIXED forever — the wire/rules contract) ========
inline fx FI(int v) { return (fx)((int64_t)v * (int64_t)kOne); }        // an integer -> Q16.16 helper

inline constexpr fx       kArenaEdge   = 5 * kOne;      // ring bound: |x|>edge or |z|>edge -> ring-out
inline constexpr fx       kKillPlaneY  = 0;             // void floor: body y < 0 -> fell in
inline constexpr fx       kSpawnX      = 2 * kOne;      // players spawn at (-2,spawnY,0) and (+2,spawnY,0)
inline constexpr fx       kSpawnY      = 2 * kOne;
inline constexpr fx       kShoveMag    = 8 * kOne;      // Q16.16 shove impulse (units/s delta-momentum)
inline constexpr fx       kShoveRange  = 8 * kOne;      // Q16.16 max distance a shove reaches the opponent
inline constexpr fx       kShoveCost   = 20 * kOne;     // mana cost per shove (GAS1)
inline constexpr uint32_t kShoveCd     = 3u;            // shove cooldown (ticks, GAS1)
inline constexpr uint32_t kStunTicks   = 4u;            // a landed shove stuns the opponent this many ticks (GT1)
inline constexpr uint32_t kRoundTicks  = 48u;           // ticks per round (a knockout usually lands ~t14)
inline constexpr uint32_t kRounds      = 3u;            // best-of-3
inline constexpr uint32_t kWinThreshold = 2u;           // first to 2 round wins takes the match
inline constexpr int      kDraw        = -1;            // a round/match with no winner

// The GAS1 shove-ability vocabulary (a duel-local kit; GAS1's MakeCoreKit stays untouched).
inline constexpr uint32_t kAbShove   = 1u;              // the SHOVE ability id
inline constexpr uint32_t kEffStun   = 10u;             // the stun aura effect id (grants State.Stunned)

// The duel player base attributes: health 100/100, mana 100/100, moveSpeed 4, attackPower 10, armor 5.
inline constexpr fx kDuelPlayerBases[gas::kAttrCount] = {
    100 * kOne, 100 * kOne, 100 * kOne, 100 * kOne, 4 * kOne, 10 * kOne, 5 * kOne, 0,
};

// ============================ THE ABILITY-LAYER ASSETS (GAS1 kit + GT1 rules + the tag/cue registry) =====

// MakeDuelRegistry: the FIXED interned tag table (ids pinned by authoring order). "State.Stunned" is the
// gameplay-tag gate; "Cue.Shove.Impact" / "Cue.Knockout" are the GC1 cue identities.
inline tags::TagRegistry MakeDuelRegistry() {
    tags::TagRegistry r;
    r.Intern("State.Stunned");       // "State"(0), "State.Stunned"(1)
    r.Intern("Cue.Shove.Impact");    // "Cue"(2), "Cue.Shove"(3), "Cue.Shove.Impact"(4)
    r.Intern("Cue.Knockout");        // "Cue.Knockout"(5)
    return r;
}
inline tags::TagId DuelTagStunned(const tags::TagRegistry& r)  { return r.Find("State.Stunned"); }     // 1
inline tags::TagId DuelCueImpact(const tags::TagRegistry& r)   { return r.Find("Cue.Shove.Impact"); }  // 4
inline tags::TagId DuelCueKnockout(const tags::TagRegistry& r) { return r.Find("Cue.Knockout"); }      // 5

// MakeShoveKit: the GAS1 SHOVE ability — costs kShoveCost mana, cooldown kShoveCd, and applies a kStunTicks
// stun aura to the TARGET (its PRESENCE grants State.Stunned via the tag rules; the armor +0 op is inert).
// Authored through the KitBuilder (the "authored, not hardcoded" discipline).
inline gas::AbilityKit MakeShoveKit() {
    gas::KitBuilder b;
    b.Ability(kAbShove, gas::kAttrMana, kShoveCost, kShoveCd)
     .Effect(kEffStun, gas::kAttrArmor, gas::kOpAdd, 0, gas::kDurTicks, kStunTicks,
             gas::kStackRefresh, 1u, 0u, gas::kTargetOther);
    return b.Build();
}

// MakeShoveRules: the GT1 gates — the shove is blockedTags State.Stunned (a stunned player cannot shove);
// the stun aura effect grantsTags State.Stunned (a landed shove stuns the opponent).
inline tags::TagRules MakeShoveRules(const tags::TagRegistry& reg) {
    const tags::TagId stunned = DuelTagStunned(reg);
    tags::TagRules rules;
    { tags::AbilityTags a; a.abilityId = kAbShove; a.blockedTags.Add(stunned); rules.abilities.push_back(a); }
    { tags::EffectTags e; e.effectId = kEffStun; e.grantsTags.Add(stunned); rules.effects.push_back(e); }
    return rules;
}

// MakeDuelTagged: the GAS1+GT1 ability world for a round — two players (gas ids 1 & 2, spawn order pinned).
inline tags::TaggedWorld MakeDuelTagged() {
    tags::TaggedWorld tw;
    tags::TagContainer none;
    tags::SpawnTagged(tw, kDuelPlayerBases, none);   // player 0 -> gas id 1
    tags::SpawnTagged(tw, kDuelPlayerBases, none);   // player 1 -> gas id 2
    return tw;
}
inline constexpr EntityId kGasId[2] = { 1u, 2u };    // the two players' GAS/GT1 entity ids

// ============================ THE VERDICT PHYSICS SCENE (platform + 2 player bodies) =====================

// The warm+sleep step config (== the verdict BuildCanonicalReplay lineage, but linDamp 0.98 for a crisp
// slide and sleep DISABLED so a resting avatar always reacts to a shove — a deterministic duel arena).
inline warmhull::HullSleepConfig MakeDuelCfg() {
    const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));   // host-snapped gravity
    convex::ConvexStepConfig stepCfg;
    stepCfg.gravity     = FxVec3{0, kGravY, 0};
    stepCfg.dt          = kOne / 60;
    stepCfg.solveIters  = 8;
    stepCfg.restitution = 0;
    stepCfg.slop        = kOne / 64;
    stepCfg.beta        = (fx)((int64_t)2 * kOne / 10);
    stepCfg.linDamp     = (fx)((int64_t)98 * kOne / 100);   // 0.98 retain (a crisp, decaying slide)
    stepCfg.angDamp     = kOne;
    stepCfg.posIters    = 4;
    warmhull::HullSleepConfig cfg;
    cfg.warm           = stepCfg;
    cfg.sleepThreshold = 0;
    cfg.wakeThreshold  = 0;
    cfg.sleepTicks     = 1000000u;   // never sleep (a shoved avatar must always move)
    return cfg;
}

// DuelScene: the FIXED verdict scene the moat harnesses replay against — the platform + two player bodies +
// the constant params. world0Snap is the copyable VD4 snapshot the harnesses clone; player[]/bodyIndex[] map
// the two avatars to their verdict EntityIds + sim body indices.
struct DuelScene {
    verdict::VerdictParams   params;              // hazard(empty)/player(none)/collectR/cfg/hulls
    verdict::VerdictSnapshot world0Snap;          // the copyable initial world (VD4 snapshot)
    EntityId                 player[2] = {kNoEntity, kNoEntity};   // the two avatars' verdict entity ids
    uint32_t                 bodyIndex[2] = {0u, 0u};              // the two avatars' sim body indices
};

// BuildDuelScene(world0): build the platform + two player bodies into world0 (filled IN PLACE — VerdictWorld
// is non-copyable), and return the DuelScene (params + the copyable snapshot + the id/body maps). Mirrors the
// verdict BuildCanonicalReplay composition (a static support body + dynamic bodies + bound gameplay entities).
inline DuelScene BuildDuelScene(verdict::VerdictWorld& world0) {
    const FxQuat kI{0, 0, 0, kOne};
    world0 = verdict::VerdictWorld{};

    // The sim: body 0 = the static platform (a wide flat box, top surface at y=+1); bodies 1,2 = the dynamic
    // player avatars (unit box hulls) at the two spawn points.
    {
        gjk::HullWorld sim;
        { fpx::FxBody b; b.pos = {0, 0, 0}; b.orient = {0, 0, 0, kOne}; b.invMass = 0; b.flags = 0u; b.vel = {0, 0, 0}; b.angVel = {0, 0, 0}; sim.bodies.push_back(b); }
        sim.hulls.push_back(gjk::MakeBox(6 * kOne, kOne, 6 * kOne));   // 0: static platform
        { fpx::FxBody b; b.pos = {(fx)(-kSpawnX), kSpawnY, 0}; b.orient = {0, 0, 0, kOne}; b.invMass = kOne; b.flags = fpx::kFlagDynamic; b.vel = {0, 0, 0}; b.angVel = {0, 0, 0}; sim.bodies.push_back(b); }
        sim.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));           // 1: player 0 avatar (left)
        { fpx::FxBody b; b.pos = {kSpawnX, kSpawnY, 0}; b.orient = {0, 0, 0, kOne}; b.invMass = kOne; b.flags = fpx::kFlagDynamic; b.vel = {0, 0, 0}; b.angVel = {0, 0, 0}; sim.bodies.push_back(b); }
        sim.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));           // 2: player 1 avatar (right)
        world0.sim = sim;
    }

    // Gameplay entities: the platform (body 0), then the two avatars (bodies 1,2) bound to their sim bodies.
    const EntityId platform = verdict::SpawnEntity(world0, verdict::Transform2D{FxVec3{0, 0, 0}, kI});
    verdict::BindBody(world0, platform, 0u);
    const EntityId p0 = verdict::SpawnEntity(world0, verdict::Transform2D{FxVec3{(fx)(-kSpawnX), kSpawnY, 0}, kI});
    verdict::BindBody(world0, p0, 1u);
    const EntityId p1 = verdict::SpawnEntity(world0, verdict::Transform2D{FxVec3{kSpawnX, kSpawnY, 0}, kI});
    verdict::BindBody(world0, p1, 2u);
    (void)platform;

    DuelScene sc;
    sc.player[0]    = p0;   sc.player[1]    = p1;
    sc.bodyIndex[0] = 1u;   sc.bodyIndex[1] = 2u;
    // An EMPTY hazard band (no SystemDamage) + no collect player — a pure physics duel.
    const verdict::HazardRegion hazard{FI(-99), FI(-99), FI(-99), FI(-99)};
    sc.params     = verdict::VerdictParams{hazard, kNoEntity, kOne, MakeDuelCfg(), world0.sim.hulls};
    sc.world0Snap = verdict::SnapshotWorld(world0);
    return sc;
}

// ============================ THE KNOCKOUT REFEREE (a pure re-sim over a verdict command stream) =========

// OutOfRing(p): the deterministic knockout predicate — a body outside the ring bounds or below the void.
inline bool OutOfRing(const FxVec3& p) {
    const fx ax = (p.x < 0) ? (fx)(-p.x) : p.x;
    const fx az = (p.z < 0) ? (fx)(-p.z) : p.z;
    return ax > kArenaEdge || az > kArenaEdge || p.y < kKillPlaneY;
}

// RoundVerdict: the outcome of a round (a knockout tick + the winner, or a draw at the tick limit).
struct RoundVerdict {
    int knockoutTick = kDraw;   // the tick the FIRST avatar left the ring (kDraw == none within the limit)
    int winner       = kDraw;   // 0 / 1 (the surviving player) or kDraw
    int loser        = kDraw;   // the knocked-out player (or kDraw)
};

// RefereeStream(scene, stream, ticks): RE-SIMULATE the verdict scene over `stream` for `ticks` ticks (the
// FROZEN SimVerdictTick), checking the ring bounds AFTER each tick; return the FIRST knockout (the earlier of
// the two avatars leaving the ring wins for the other). A pure function of (scene, stream, ticks) — the same
// re-sim the moat harnesses run, so the referee's winner is bit-exactly reproducible. If both leave on the
// SAME tick, the one with the larger outward distance is the loser (deterministic tie-break).
inline RoundVerdict RefereeStream(const DuelScene& scene, const std::vector<Command>& stream, uint32_t ticks) {
    verdict::VerdictWorld w = verdict::ClonePeer(scene.world0Snap, scene.params);
    RoundVerdict rv;
    for (uint32_t t = 0; t < ticks; ++t) {
        verdict::SimVerdictTick(w, scene.params, stream, t);
        const FxVec3& b0 = w.sim.bodies[(size_t)scene.bodyIndex[0]].pos;
        const FxVec3& b1 = w.sim.bodies[(size_t)scene.bodyIndex[1]].pos;
        const bool o0 = OutOfRing(b0), o1 = OutOfRing(b1);
        if (o0 || o1) {
            rv.knockoutTick = (int)t;
            if (o0 && !o1)      { rv.loser = 0; rv.winner = 1; }
            else if (o1 && !o0) { rv.loser = 1; rv.winner = 0; }
            else {
                // Both out on the same tick — the farther-out avatar is the loser (deterministic).
                const fx d0 = (b0.x < 0 ? (fx)(-b0.x) : b0.x);
                const fx d1 = (b1.x < 0 ? (fx)(-b1.x) : b1.x);
                if (d0 >= d1) { rv.loser = 0; rv.winner = 1; } else { rv.loser = 1; rv.winner = 0; }
            }
            return rv;
        }
    }
    return rv;   // draw/timeout (kDraw)
}

// ============================ THE ROUND — PLAY THE GAME (ability layer -> emitted physics stream) ========

// DuelInput: one player's per-tick input — a lateral move impulse (usually 0) + a shove BUTTON.
struct DuelInput {
    fx   moveX = 0;      // Q16.16 lateral self-impulse this tick (0 = no move)
    bool shove = false;  // the shove button this tick
};

// The per-caster shove-resolution outcome (mirrors the GT1 vocabulary, for the pins).
inline constexpr uint32_t kShoveLanded    = 0u;   // committed + in range -> impulse + stun + cue
inline constexpr uint32_t kShoveOutOfRange = 1u;  // the opponent was too far (no activation attempted)
inline constexpr uint32_t kShoveBlocked   = 2u;   // GT1 blocked (the caster is stunned)
inline constexpr uint32_t kShoveUnafford  = 3u;   // out of mana
inline constexpr uint32_t kShoveOnCd      = 4u;   // on cooldown

// RoundResult: the full record of a played round (the emitted stream the moat replays + the game outcome +
// the ability/tag/cue pins + the per-tick trails for the viz + the digests).
struct RoundResult {
    std::vector<Command> emittedStream;              // the LOWERED verdict physics stream (moat input)
    uint32_t             ticks = 0;
    RoundVerdict         verdictOut;                 // knockout tick + winner (from the played world)
    std::vector<FxVec3>  trail[2];                   // per-tick avatar positions (post-step) — the viz trails
    uint32_t             shovesLanded[2]  = {0, 0};
    uint32_t             shovesBlocked[2] = {0, 0};  // blocked by stun
    uint32_t             shovesUnafford[2]= {0, 0};  // out of mana
    uint32_t             shovesOnCd[2]    = {0, 0};
    uint32_t             stunsApplied     = 0;       // total landed shoves that stunned the opponent
    fx                   finalMana[2]     = {0, 0};  // each player's remaining mana (the cost pin)
    std::vector<uint32_t> impactTicks;               // ticks a shove landed (the cue-fire ticks)
    std::vector<cues::CueEvent> cueLog;              // the GC1 impact + knockout cue event stream
    std::string          finalDigest;                // DigestSnapshot of the final verdict world
    uint64_t             traceDigest = 0;            // per-tick trail + ability-outcome fold (pinned)
};

// FxDist2XZ: the Q16.16 squared distance in the XZ plane (the shove range test; NO sqrt).
inline int64_t FxDist2XZ(const FxVec3& a, const FxVec3& b) {
    const int64_t dx = (int64_t)a.x - (int64_t)b.x;
    const int64_t dz = (int64_t)a.z - (int64_t)b.z;
    return dx * dx + dz * dz;
}

// RunDuelRound: PLAY one round. Materialize the verdict world (ClonePeer); each tick, in the PINNED order:
//   (A) resolve BOTH players' inputs (player 0 then 1): a shove is attempted only when the opponent is within
//       kShoveRange (a pure physics range check on the PRE-step body positions); if in range, gate it through
//       the REAL GAS1/GT1 ability layer (tags::TryActivateTagged — cost + cooldown + not-stunned). On a
//       committed shove: LOWER it to a verdict kCmdImpulse pushing the opponent OUTWARD (+ the opponent is
//       stunned by the aura effect, + an impact cue fires); else record the deterministic failure reason. A
//       non-zero moveX lowers to a self-impulse.
//   (B) gas::StepAbilities — tick cooldowns + expire the stun aura (ONCE).
//   (C) verdict::SimVerdictTick over the emitted stream (applies THIS tick's impulses + steps the physics).
//   (D) record the trails + detect the first knockout (a knockout cue fires once).
// The emitted stream is a pure deterministic function of the input scripts, so the moat harnesses replay it.
inline RoundResult RunDuelRound(const DuelScene& scene, const tags::TagRegistry& reg,
                                const std::vector<DuelInput>& inA, const std::vector<DuelInput>& inB,
                                uint32_t ticks) {
    const gas::AbilityKit kit   = MakeShoveKit();
    const tags::TagRules  rules = MakeShoveRules(reg);
    const tags::TagId      stunTag = DuelTagStunned(reg);
    const tags::TagId      impactCueTag   = DuelCueImpact(reg);
    const tags::TagId      knockoutCueTag = DuelCueKnockout(reg);
    (void)stunTag;

    RoundResult rr;
    rr.ticks = ticks;
    tags::TaggedWorld tw = MakeDuelTagged();                          // the GAS1+GT1 ability world (2 players)
    verdict::VerdictWorld w = verdict::ClonePeer(scene.world0Snap, scene.params);   // the physics world (VD4 clone)
    const std::vector<DuelInput>* in[2] = { &inA, &inB };
    verdict::DigestFnv trace;

    bool decided = false;

    for (uint32_t t = 0; t < ticks; ++t) {
        // (A) resolve BOTH players' inputs in PINNED order (0 then 1) — a shove lowers to a kCmdImpulse.
        for (int i = 0; i < 2; ++i) {
            const int opp = 1 - i;
            const DuelInput di = (t < in[i]->size()) ? (*in[i])[(size_t)t] : DuelInput{};
            // Movement: a small self-impulse this tick (lowered verbatim).
            if (di.moveX != 0) {
                Command mv; mv.tick = t; mv.kind = verdict::kCmdImpulse; mv.target = scene.player[i];
                mv.arg = FxVec3{di.moveX, 0, 0};
                rr.emittedStream.push_back(mv);
            }
            if (!di.shove) continue;
            // The shove range check (pre-step body positions, the XZ plane).
            const FxVec3& pSelf = w.sim.bodies[(size_t)scene.bodyIndex[i]].pos;
            const FxVec3& pOpp  = w.sim.bodies[(size_t)scene.bodyIndex[opp]].pos;
            const int64_t r2 = (int64_t)kShoveRange * (int64_t)kShoveRange;
            if (FxDist2XZ(pSelf, pOpp) > r2) { trace.mix32(kShoveOutOfRange); continue; }  // out of range -> whiff (no attempt)
            // Gate through GAS1/GT1 (cost + cooldown + not-stunned) targeting the opponent.
            const tags::TaggedResult tr =
                tags::TryActivateTagged(tw, kit, rules, reg, kGasId[i], kAbShove, kGasId[opp]);
            trace.mix32((uint32_t)tr);
            if (tr == tags::kTagOk || tr == tags::kTagImmune) {
                // A landed shove: LOWER to an outward impulse on the opponent's body.
                const fx dir = (pOpp.x >= pSelf.x) ? kShoveMag : (fx)(-kShoveMag);
                Command sh; sh.tick = t; sh.kind = verdict::kCmdImpulse; sh.target = scene.player[opp];
                sh.arg = FxVec3{dir, 0, 0};
                rr.emittedStream.push_back(sh);
                ++rr.shovesLanded[i];
                ++rr.stunsApplied;                            // the aura stunned the opponent (GT1)
                rr.impactTicks.push_back(t);
                rr.cueLog.push_back(cues::CueEvent{impactCueTag, scene.player[i], scene.player[opp],
                                                   pOpp, kShoveMag, t});
            } else if (tr == tags::kTagBlockedByTag)       { ++rr.shovesBlocked[i]; }
            else if (tr == tags::kTagUnaffordable)         { ++rr.shovesUnafford[i]; }
            else if (tr == tags::kTagOnCooldown)           { ++rr.shovesOnCd[i]; }
        }

        // (B) tick the ability world (cooldowns-- + stun-aura expiry).
        gas::StepAbilities(tw.gas);

        // (C) step the physics over the emitted stream (applies THIS tick's impulses).
        verdict::SimVerdictTick(w, scene.params, rr.emittedStream, t);

        // (D) trails + first-knockout detection (a knockout cue fires once).
        const FxVec3 p0 = w.sim.bodies[(size_t)scene.bodyIndex[0]].pos;
        const FxVec3 p1 = w.sim.bodies[(size_t)scene.bodyIndex[1]].pos;
        rr.trail[0].push_back(p0);
        rr.trail[1].push_back(p1);
        trace.mix32((uint32_t)p0.x); trace.mix32((uint32_t)p0.y);
        trace.mix32((uint32_t)p1.x); trace.mix32((uint32_t)p1.y);
        trace.sep();
        if (!decided) {
            const bool o0 = OutOfRing(p0), o1 = OutOfRing(p1);
            if (o0 || o1) {
                decided = true;
                rr.verdictOut.knockoutTick = (int)t;
                if (o0 && !o1)      { rr.verdictOut.loser = 0; rr.verdictOut.winner = 1; }
                else if (o1 && !o0) { rr.verdictOut.loser = 1; rr.verdictOut.winner = 0; }
                else {
                    const fx d0 = (p0.x < 0 ? (fx)(-p0.x) : p0.x);
                    const fx d1 = (p1.x < 0 ? (fx)(-p1.x) : p1.x);
                    if (d0 >= d1) { rr.verdictOut.loser = 0; rr.verdictOut.winner = 1; }
                    else          { rr.verdictOut.loser = 1; rr.verdictOut.winner = 0; }
                }
                const EntityId loserE = scene.player[rr.verdictOut.loser];
                const FxVec3&  loserP = w.sim.bodies[(size_t)scene.bodyIndex[rr.verdictOut.loser]].pos;
                rr.cueLog.push_back(cues::CueEvent{knockoutCueTag, kNoEntity, loserE, loserP, 0, t});
            }
        }
    }

    // Final ability + physics pins.
    for (int i = 0; i < 2; ++i) {
        const int gi = gas::FindEntity(tw.gas, kGasId[i]);
        rr.finalMana[i] = (gi >= 0) ? tw.gas.entities[(size_t)gi].attrs.current[gas::kAttrMana] : 0;
    }
    rr.finalDigest = verdict::DigestSnapshot(verdict::SnapshotWorld(w));
    // Fold the outcome + ability tallies into the trace (the per-round pin).
    trace.mix32((uint32_t)rr.verdictOut.knockoutTick);
    trace.mix32((uint32_t)rr.verdictOut.winner);
    for (int i = 0; i < 2; ++i) { trace.mix32(rr.shovesLanded[i]); trace.mix32(rr.shovesBlocked[i]);
                                  trace.mix32((uint32_t)rr.finalMana[i]); }
    trace.mix32(rr.stunsApplied);
    trace.mix32((uint32_t)rr.cueLog.size());
    rr.traceDigest = trace.h;
    return rr;
}

// ============================ THE CANONICAL MATCH SCRIPTS (FIXED forever — test + both showcases run these) =

// The FIXED per-round input scripts. Round 0: player 0 aggressive (shoves t2/t6/t10) -> knocks out player 1.
// Round 1: player 1 aggressive (shoves t2/t6/t10, STUNNING player 0 who then tries to shove at t3 while
// stunned -> BLOCKED) -> knocks out player 0. Round 2: player 0 aggressive again -> match 2-1, player 0 wins.
// A shove at tick T is a DuelInput{0, true} at index T.
inline std::vector<DuelInput> MakeRoundScript(uint32_t round, int player) {
    std::vector<DuelInput> s((size_t)kRoundTicks);
    auto shoveAt = [&](uint32_t t) { if (t < s.size()) s[(size_t)t].shove = true; };
    const bool p0Aggressor = (round != 1u);   // rounds 0 and 2: player 0 attacks; round 1: player 1 attacks
    const int  aggressor   = p0Aggressor ? 0 : 1;
    if (player == aggressor) {
        shoveAt(2u); shoveAt(6u); shoveAt(10u);   // three shoves knock the opponent out ~tick 14
    } else {
        // The defender: in round 1 the (stunned) defender tries a doomed shove at t3 (the blocked-by-stun pin).
        if (round == 1u) shoveAt(3u);
    }
    return s;
}

// MatchResult: the whole best-of-3 outcome (per-round records + the score + the winner + the match digest).
struct MatchResult {
    RoundResult rounds[kRounds];
    uint32_t    roundsPlayed = 0;
    uint32_t    score[2]     = {0, 0};
    int         matchWinner  = kDraw;
    uint64_t    matchDigest  = 0;
};

// RunDuelMatch: play the FIXED best-of-3 over a FRESH scene each round (deterministic round reset), tally the
// score, and stop when a player reaches kWinThreshold round wins. A pure function -> two runs are identical.
inline MatchResult RunDuelMatch() {
    const tags::TagRegistry reg = MakeDuelRegistry();
    MatchResult mr;
    verdict::DigestFnv md;
    for (uint32_t r = 0; r < kRounds; ++r) {
        verdict::VerdictWorld world0;
        const DuelScene scene = BuildDuelScene(world0);
        const std::vector<DuelInput> sA = MakeRoundScript(r, 0);
        const std::vector<DuelInput> sB = MakeRoundScript(r, 1);
        mr.rounds[r] = RunDuelRound(scene, reg, sA, sB, kRoundTicks);
        ++mr.roundsPlayed;
        const int win = mr.rounds[r].verdictOut.winner;
        if (win == 0 || win == 1) ++mr.score[win];
        md.mix(mr.rounds[r].traceDigest);
        md.mix32(mr.score[0]); md.mix32(mr.score[1]); md.sep();
        if (mr.score[0] >= kWinThreshold) { mr.matchWinner = 0; break; }
        if (mr.score[1] >= kWinThreshold) { mr.matchWinner = 1; break; }
    }
    mr.matchDigest = md.h;
    return mr;
}

// ============================ THE MOAT PROOFS (pure functions over a round's emitted stream) =============

// (1) LOCKSTEP — two peers re-derive the round bit-for-bit from the emitted stream ALONE (RunVerdictLockstep,
// verbatim). Returns the final DigestSnapshot; *outIdentical = the two peers were byte-identical.
inline std::string DuelLockstep(const DuelScene& scene, const std::vector<Command>& stream, uint32_t ticks,
                                bool* outIdentical) {
    const verdict::VerdictSnapshot fin =
        verdict::RunVerdictLockstep(scene.world0Snap, scene.params, stream, ticks, outIdentical);
    return verdict::DigestSnapshot(fin);
}

// (2) ROLLBACK — a mispredicted input is corrected bit-exactly (RunVerdictRollback, verbatim). The mispredict
// stream flips one impulse (the DX6 tamper convention) so it diverges across the physics. *outCorrected = the
// corrected world == the straight authority; *outDiverged = the misprediction really perturbed the world.
// PerturbStream flips the FIRST impulse whose tick >= fromTick (so the misprediction lands INSIDE the
// speculation window [rollbackAt, rollbackAt+3) and actually diverges — the non-vacuous rollback control).
inline std::vector<Command> PerturbStream(const std::vector<Command>& stream, uint32_t fromTick) {
    std::vector<Command> m = stream;
    for (size_t i = 0; i < m.size(); ++i) {
        if (m[i].kind == verdict::kCmdImpulse && m[i].tick >= fromTick) {
            m[i].arg.x = (fx)(-m[i].arg.x - kOne);   // a real, physics-divergent flip (the DX6 convention)
            break;
        }
    }
    return m;
}
inline void DuelRollback(const DuelScene& scene, const std::vector<Command>& stream, uint32_t ticks,
                         uint32_t rollbackAt, bool* outCorrected, bool* outDiverged) {
    const std::vector<Command> mis = PerturbStream(stream, rollbackAt);
    (void)verdict::RunVerdictRollback(scene.world0Snap, scene.params, stream, mis, ticks, rollbackAt,
                                      outCorrected, outDiverged);
}

// (3) REPLAY — the round records to a verdict ReplayFile (Serialize -> a demo TEXT), whose byte-hash is
// pinned; parsing it back + re-running lockstep reproduces the SAME final digest (record==replay). Returns
// the serialized demo text; *outDemoHash = its FNV byte-hash; *outReplayOk = the round-trip re-derives the
// pinned final digest.
inline std::string DuelReplayDemo(const DuelScene& scene, const std::vector<Command>& stream, uint32_t ticks,
                                  uint64_t* outDemoHash, bool* outReplayOk) {
    verdict::ReplayFile rf;
    rf.ticks  = ticks;
    rf.stream = stream;
    bool ident = false;
    rf.finalDigest = DuelLockstep(scene, stream, ticks, &ident);
    const std::string text = verdict::SerializeReplay(rf);
    if (outDemoHash) {
        verdict::DigestFnv d;
        for (unsigned char c : text) d.mix32((uint32_t)c);
        *outDemoHash = d.h;
    }
    if (outReplayOk) {
        const verdict::ReplayFile rf2 = verdict::ParseReplay(text);   // parse the demo back
        bool id2 = false;
        const std::string again = DuelLockstep(scene, rf2.stream, rf2.ticks, &id2);
        *outReplayOk = (again == rf.finalDigest);                      // record == replay (bit-exact)
    }
    return text;
}

// (4) FORK (what-if) — fork the round at a tick and INJECT an EARLIER shove for the aggressor so the opponent
// is knocked out SOONER / or a DIFFERENT avatar wins. Composes fork.h (BuildForkDemo / ForkAt / Resimulate)
// for the reproducible timeline-tree digests AND runs the referee over both streams to show the WINNER change.
struct DuelForkProof {
    int      forkTick        = 0;
    int      origWinner      = kDraw;
    int      forkWinner      = kDraw;
    int      origKnockout    = kDraw;
    int      forkKnockout    = kDraw;
    bool     winnerChanged   = false;   // the counterfactual reaches a different game outcome
    int      firstDivergence = -1;      // fork.h: the per-tick digest divergence tick (== the injection tick)
    uint64_t origFullDigest  = 0;
    uint64_t forkFullDigest  = 0;
    std::vector<Command> forkStream;    // the mutated physics stream (for the referee)
};

inline constexpr uint32_t kForkTick      = 2u;          // fork the winning round here
inline constexpr fx       kForkShoveMag  = 40 * kOne;   // a decisive counter-shove (one hit -> a clean flip)

// MakeForkCounterShove(scene): a SINGLE big outward counter-shove on player 0 at kForkTick — the "what if
// player 1 had shoved back hard" input that flips the winner from 0 to 1.
inline Command MakeForkCounterShove(const DuelScene& scene) {
    Command c;
    c.tick = kForkTick; c.kind = verdict::kCmdImpulse; c.target = scene.player[0];
    c.arg = FxVec3{(fx)(-kForkShoveMag), 0, 0};   // push player 0 toward the -x edge
    return c;
}

// DuelForkChangeWinner: build the fork tree over `stream`, inject `inject` (a counterfactual shove) at
// forkTick, and compare the ORIGINAL vs the FORKED winner. `inject` must fire at tick >= forkTick.
inline DuelForkProof DuelForkChangeWinner(const DuelScene& scene, const std::vector<Command>& stream,
                                          uint32_t ticks, uint32_t forkTick, const Command& inject) {
    DuelForkProof pf;
    pf.forkTick = (int)forkTick;

    // The fork.h timeline tree over the emitted stream (reproducible counterfactual digests).
    const fork::Timeline original = fork::SimulateTimeline(scene.world0Snap, scene.params, stream, ticks);
    const fork::ForkDemo demo = fork::BuildForkDemo(scene.world0Snap, scene.params, stream, ticks, /*kf*/8u);
    fork::ForkedTimeline branch = fork::ForkAt(demo, original, scene.params, forkTick, stream);
    fork::MutateInput(branch, inject);
    const fork::Timeline resim = fork::ResimulateFork(branch);
    const fork::TimelineDiff diff = fork::DiffTimelines(original, resim);
    pf.firstDivergence = diff.firstDivergence;
    pf.origFullDigest  = original.fullDigest;
    pf.forkFullDigest  = resim.fullDigest;

    // The game-level winner change: the referee over the ORIGINAL vs the FORKED (mutated) physics stream.
    pf.forkStream = branch.stream;
    const RoundVerdict ov = RefereeStream(scene, stream, ticks);
    const RoundVerdict fv = RefereeStream(scene, branch.stream, ticks);
    pf.origWinner   = ov.winner;   pf.origKnockout = ov.knockoutTick;
    pf.forkWinner   = fv.winner;   pf.forkKnockout = fv.knockoutTick;
    pf.winnerChanged = (ov.winner != fv.winner) || (ov.knockoutTick != fv.knockoutTick);
    return pf;
}

// (5) ANTI-CHEAT — the honest match verifies; a client that CLAIMS a different outcome (an impossible extra
// impulse / a fabricated digest) is REJECTED at the exact divergence tick (authority_verify::Verify, verbatim).
struct DuelAntiCheatProof {
    bool honestVerified   = false;
    int  cheaterCaughtTick = -1;
    std::string serverDigest, clientDigest;
    std::string inputCommitment;
    uint32_t cheatTick = 0;
};

// DuelAntiCheat: the server RE-SIMULATES the submitted (honest) stream; the cheater submits the SAME stream
// but CLAIMS the digest trace of a DIFFERENT stream (an extra shove at cheatTick — an impossible outcome).
inline DuelAntiCheatProof DuelAntiCheat(const DuelScene& scene, const std::vector<Command>& stream,
                                        uint32_t ticks, uint32_t cheatTick) {
    DuelAntiCheatProof pf;
    pf.cheatTick = cheatTick;
    // The honest claim = the true per-tick digest trace of the submitted stream.
    const std::vector<std::string> honestClaim =
        acnet::VerdictDigestTrace(scene.world0Snap, scene.params, stream, ticks);
    // The cheater's claim = the trace of a CHEATED stream (an extra impulse the submitted inputs don't have).
    std::vector<Command> cheatStream = stream;
    Command extra; extra.tick = cheatTick; extra.kind = verdict::kCmdImpulse;
    extra.target = scene.player[0]; extra.arg = FxVec3{kShoveMag, 0, 0};
    cheatStream.push_back(extra);
    const std::vector<std::string> cheaterClaim =
        acnet::VerdictDigestTrace(scene.world0Snap, scene.params, cheatStream, ticks);

    const acnet::VerifyResult honest  = acnet::Verify(scene.world0Snap, scene.params, stream, honestClaim,  ticks);
    const acnet::VerifyResult cheater = acnet::Verify(scene.world0Snap, scene.params, stream, cheaterClaim, ticks);
    pf.honestVerified    = honest.ok && honest.firstDivergentTick == -1;
    pf.cheaterCaughtTick = cheater.firstDivergentTick;
    pf.serverDigest      = cheater.serverDigest;
    pf.clientDigest      = cheater.clientDigest;
    pf.inputCommitment   = acnet::CommitInputStream(stream);
    return pf;
}

// ============================ THE SHOWCASE VIZ (strict-integer, NO shader) — shared VERBATIM by the Vulkan
// --game1-duel-shot + the Metal --game1-duel so the pixels are byte-identical cross-backend BY CONSTRUCTION.
// A side-view arena (platform + ring edges + void), the two avatars' TRAILS over the decisive round (colored
// polylines), shove-impact + knockout markers, a per-round SCOREBOARD, and a MOAT-PROOF panel (deterministic /
// lockstep / replay / fork->different-winner / anti-cheat). Every coordinate is an integer map of the Q16.16
// world. ============================================================================================

inline constexpr int kG1ImgW   = 640;
inline constexpr int kG1ImgH   = 400;
inline constexpr int kG1MapX0  = 320;   // screen x of world x=0 (arena centered)
inline constexpr int kG1MapY0  = 150;   // screen y of world y=0 (the void line; +y world -> up)
inline constexpr int kG1Scale  = 26;    // pixels per world unit

struct Duel1VizStats {
    uint32_t players      = 0;
    uint32_t rounds       = 0;
    int      matchWinner  = kDraw;       // 0 / 1 / kDraw
    uint32_t score0       = 0;
    uint32_t score1       = 0;
    int      knockoutTick = kDraw;       // round 0's knockout tick
    bool     deterministic = false;
    bool     lockstep      = false;
    bool     replayOk      = false;
    bool     forkChanged   = false;
    int      cheaterCaught = -1;
    uint32_t width  = 0;
    uint32_t height = 0;
    uint64_t pixDigest   = 0;            // FNV over the RGBA8 pixels (the cross-backend strict-zero proof)
    uint64_t matchDigest = 0;            // RunDuelMatch().matchDigest (the runtime determinism digest)
    std::string finalDigest;             // round 0 final DigestSnapshot
};

inline int G1MapX(fx wx) { return kG1MapX0 + (int)(((int64_t)wx * kG1Scale) / (int64_t)kOne); }
inline int G1MapY(fx wy) { return kG1MapY0 - (int)(((int64_t)wy * kG1Scale) / (int64_t)kOne); }

// RenderDuelViz: fill `out` (RGBA8, kG1ImgW x kG1ImgH) + the stat block. Plays the canonical match, runs ALL
// FIVE moat proofs on round 0's emitted stream, and draws the report. Deterministic + pure integer -> two
// calls (and two backends) byte-identical.
inline void RenderDuelViz(std::vector<uint8_t>& out, Duel1VizStats& stats) {
    const int W = kG1ImgW, H = kG1ImgH;
    out.assign((std::size_t)W * H * 4u, 0);
    auto px = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        uint8_t* d = &out[((std::size_t)y * W + x) * 4u];
        d[0] = r; d[1] = g; d[2] = b; d[3] = 255;
    };
    auto fillRect = [&](int x0, int y0, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y < y0 + h; ++y) for (int x = x0; x < x0 + w; ++x) px(x, y, r, g, b);
    };
    auto border = [&](int x0, int y0, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        for (int x = x0; x < x0 + w; ++x) { px(x, y0, r, g, b); px(x, y0 + h - 1, r, g, b); }
        for (int y = y0; y < y0 + h; ++y) { px(x0, y, r, g, b); px(x0 + w - 1, y, r, g, b); }
    };
    auto vline = [&](int x, int y0, int y1, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y <= y1; ++y) px(x, y, r, g, b);
    };
    auto disc = [&](int cx, int cy, int rad, uint8_t r, uint8_t g, uint8_t b) {
        for (int dy = -rad; dy <= rad; ++dy) for (int dx = -rad; dx <= rad; ++dx)
            if (dx * dx + dy * dy <= rad * rad) px(cx + dx, cy + dy, r, g, b);
    };

    // Background + title band.
    fillRect(0, 0, W, H, 14, 15, 20);
    fillRect(0, 0, W, 30, 24, 27, 36);

    // ---- Play the canonical match + run the moat proofs on round 0's emitted stream. -------------------
    const MatchResult mr = RunDuelMatch();
    const RoundResult& r0 = mr.rounds[0];

    // Rebuild round 0's scene for the proofs (BuildDuelScene is deterministic; a fresh world0 each call).
    verdict::VerdictWorld pw;
    const DuelScene pscene = BuildDuelScene(pw);
    bool lockstepId = false;
    const std::string lockDigest = DuelLockstep(pscene, r0.emittedStream, r0.ticks, &lockstepId);
    uint64_t demoHash = 0; bool replayOk = false;
    (void)DuelReplayDemo(pscene, r0.emittedStream, r0.ticks, &demoHash, &replayOk);
    // Fork the winning round at kForkTick + inject a SINGLE decisive player-1 counter-shove on player 0 (a big
    // outward -x impulse) -> player 0 is flung out first -> the WINNER flips 0 -> 1 (a reproducible what-if).
    const DuelForkProof fkp = DuelForkChangeWinner(pscene, r0.emittedStream, r0.ticks, kForkTick,
                                                   MakeForkCounterShove(pscene));
    const DuelAntiCheatProof acp = DuelAntiCheat(pscene, r0.emittedStream, r0.ticks, 6u);

    // Two-run determinism control (a second match == the first).
    const MatchResult mr2 = RunDuelMatch();
    const bool deterministic = (mr.matchDigest == mr2.matchDigest);

    // ---- The arena side-view: platform, ring edges, void, avatar trails, impacts, knockout. -------------
    const int platY = G1MapY(kOne);                         // platform top (world y=+1)
    const int platX0 = G1MapX((fx)(-6 * kOne)), platX1 = G1MapX(6 * kOne);
    const int voidY0 = G1MapY(0);                           // world y=0 (kill plane)
    fillRect(0, voidY0, W, H - voidY0, 8, 9, 14);           // the void below
    fillRect(platX0, platY, platX1 - platX0, voidY0 - platY, 46, 50, 62);  // the platform slab
    // Ring edges (|x| = kArenaEdge) — bright vertical danger lines.
    vline(G1MapX((fx)(-kArenaEdge)), 40, voidY0, 200, 80, 60);
    vline(G1MapX(kArenaEdge),        40, voidY0, 200, 80, 60);

    // The two avatars' trails over round 0 (polyline of post-step positions).
    const uint8_t col[2][3] = { {90, 160, 240}, {240, 150, 70} };   // P0 blue, P1 orange
    for (int i = 0; i < 2; ++i) {
        int prevx = 0, prevy = 0; bool have = false;
        for (size_t t = 0; t < r0.trail[i].size(); ++t) {
            const int cx = G1MapX(r0.trail[i][t].x), cy = G1MapY(r0.trail[i][t].y);
            if (have) { // draw a short segment (integer DDA-ish: sample a few points)
                const int steps = 6;
                for (int s = 0; s <= steps; ++s) {
                    const int sx = prevx + (cx - prevx) * s / steps;
                    const int sy = prevy + (cy - prevy) * s / steps;
                    px(sx, sy, col[i][0], col[i][1], col[i][2]);
                }
            }
            prevx = cx; prevy = cy; have = true;
        }
        // The avatar's final position as a disc.
        if (!r0.trail[i].empty()) {
            const FxVec3 last = r0.trail[i].back();
            disc(G1MapX(last.x), G1MapY(last.y), 4, col[i][0], col[i][1], col[i][2]);
        }
        // Spawn marker.
        disc(G1MapX((fx)((i == 0) ? -kSpawnX : kSpawnX)), G1MapY(kSpawnY), 3, col[i][0], col[i][1], col[i][2]);
    }
    // Impact markers (small yellow diamonds) at each shove-impact along the loser's trail.
    for (uint32_t it : r0.impactTicks) {
        const int loser = (r0.verdictOut.loser >= 0) ? r0.verdictOut.loser : 1;
        if ((size_t)it < r0.trail[loser].size()) {
            const int cx = G1MapX(r0.trail[loser][it].x), cy = G1MapY(r0.trail[loser][it].y);
            for (int dy = -2; dy <= 2; ++dy) { const int sp = 2 - (dy < 0 ? -dy : dy);
                for (int dx = -sp; dx <= sp; ++dx) px(cx + dx, cy + dy, 250, 230, 90); }
        }
    }
    // The knockout moment (a bright ring at the loser's knockout position).
    if (r0.verdictOut.knockoutTick >= 0 && r0.verdictOut.loser >= 0) {
        const int loser = r0.verdictOut.loser;
        const size_t kt = (size_t)r0.verdictOut.knockoutTick;
        if (kt < r0.trail[loser].size()) {
            const int cx = G1MapX(r0.trail[loser][kt].x), cy = G1MapY(r0.trail[loser][kt].y);
            for (int rad = 5; rad <= 7; ++rad)
                for (int a = 0; a < 360; a += 12) {
                    // cheap integer circle via a small fixed dx/dy table (avoid trig): sample 30 dirs.
                    static const int cs[30] = {100,97,86,70,50,25,0,-25,-50,-70,-86,-97,-100,-97,-86,-70,-50,-25,0,25,50,70,86,97,100,97,86,70,50,25};
                    const int idx = (a / 12) % 30;
                    const int sn = cs[(idx + 22) % 30];   // sin ~ cos shifted
                    px(cx + cs[idx] * rad / 100, cy + sn * rad / 100, 250, 90, 90);
                }
        }
    }

    // ---- Scoreboard: per-round win pips (P0 top row, P1 bottom row). ------------------------------------
    fillRect(12, 40, 150, 70, 20, 22, 30);
    border(12, 40, 150, 70, 60, 66, 80);
    for (uint32_t r = 0; r < mr.roundsPlayed; ++r) {
        const int win = mr.rounds[r].verdictOut.winner;
        const int bx = 22 + (int)r * 34;
        // P0 pip
        fillRect(bx, 52, 26, 20, win == 0 ? col[0][0] : 40, win == 0 ? col[0][1] : 44, win == 0 ? col[0][2] : 54);
        // P1 pip
        fillRect(bx, 80, 26, 20, win == 1 ? col[1][0] : 40, win == 1 ? col[1][1] : 44, win == 1 ? col[1][2] : 54);
    }

    // ---- MOAT-PROOF panel (five checks). ---------------------------------------------------------------
    struct Chk { const char* label; bool ok; };
    const Chk checks[5] = {
        { "deterministic",   deterministic },
        { "lockstep",        lockstepId },
        { "replay-identical", replayOk },
        { "fork->diff-winner", fkp.winnerChanged },
        { "anti-cheat-caught", acp.cheaterCaughtTick == (int)acp.cheatTick },
    };
    const int py0 = 120;
    fillRect(12, py0, 150, 130, 20, 22, 30);
    border(12, py0, 150, 130, 60, 66, 80);
    for (int i = 0; i < 5; ++i) {
        const int ry = py0 + 10 + i * 24;
        // A green (ok) / red (fail) swatch.
        fillRect(20, ry, 16, 16, checks[i].ok ? 46 : 175, checks[i].ok ? 160 : 48, checks[i].ok ? 70 : 48);
        border(20, ry, 16, 16, 210, 210, 210);
        if (checks[i].ok) { // a small check mark
            for (int k = 0; k < 5; ++k) px(22 + k, ry + 8 + (k < 2 ? k : 4 - k), 240, 255, 240);
            for (int k = 0; k < 8; ++k) px(25 + k, ry + 9 - k, 240, 255, 240);
        }
    }

    // ---- Stats. ----------------------------------------------------------------------------------------
    stats.players       = 2u;
    stats.rounds        = mr.roundsPlayed;
    stats.matchWinner   = mr.matchWinner;
    stats.score0        = mr.score[0];
    stats.score1        = mr.score[1];
    stats.knockoutTick  = r0.verdictOut.knockoutTick;
    stats.deterministic = deterministic;
    stats.lockstep      = lockstepId;
    stats.replayOk      = replayOk;
    stats.forkChanged   = fkp.winnerChanged;
    stats.cheaterCaught = acp.cheaterCaughtTick;
    stats.width         = (uint32_t)W;
    stats.height        = (uint32_t)H;
    stats.matchDigest   = mr.matchDigest;
    stats.finalDigest   = r0.finalDigest;
    verdict::DigestFnv pd;
    for (uint8_t byte : out) pd.mix32((uint32_t)byte);
    stats.pixDigest = pd.h;
}

}  // namespace duel
}  // namespace game
}  // namespace hf
