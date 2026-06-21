// Slice VD1 — Deterministic Gameplay / Netcode: THE ENTITY WORLD + THE INPUT-COMMAND BUS (the beachhead of
// FLAGSHIP #27, hf::game::verdict). The PURE-CPU integer entity world + the unified tick-ordered command bus that
// generalizes convex::ConvexCommand + lowers cleanly to the frozen sim command contract. verdict.h is a NEW
// additive sibling that #includes ecs/ecs.h + sim/warmhull.h READ-ONLY.
//
// What this test PINS (the determinism contract + the spec's four proofs):
//   * SpawnEntity allocates MONOTONIC EntityIds (start 1) + appends to order[]; component get/has round-trips.
//   * DespawnEntity removes from order[] + frees the ecs handle, but the next spawn does NOT reuse the id — the
//     PINNED-IDENTITY contract: a spawn->despawn->spawn re-derives the SAME next EntityId regardless of the
//     ecs::Registry free-list recycle (proof 2).
//   * ApplyCommands is TICK-FILTERED + FIXED-ORDER + DEAD/OUT-OF-RANGE-TARGET-NO-OP (proof 3).
//   * LowerToHullCommands == a hand-written convex::ConvexCommand stream (proof 4).
//   * A fixed spawn/despawn/command script yields a BYTE-IDENTICAL world summary on two runs (proof 1).
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests.
#include "game/verdict.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace verdict  = hf::game::verdict;
namespace convex   = hf::sim::convex;
namespace fpx      = hf::sim::fpx;
using fpx::fx;
using fpx::kOne;
using fpx::FxVec3;
using fpx::FxQuat;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static FxVec3 V(int x, int y, int z) {
    return FxVec3{(fx)((int64_t)x * kOne), (fx)((int64_t)y * kOne), (fx)((int64_t)z * kOne)};
}

// The deterministic spawn/despawn/command SCRIPT used by the two-run byte-equality proof + the showcase.
// Builds a small world: spawn 3 entities, fire move/ability over a few ticks, despawn one, then spawn one more
// (which must NOT reuse the despawned id). Returns the converged VerdictWorld.
static verdict::VerdictWorld RunScript() {
    verdict::VerdictWorld w;
    // Spawn three entities with distinct positions (the pinned ids 1,2,3 in spawn order).
    const verdict::EntityId a = verdict::SpawnEntity(w, verdict::Transform2D{V(-3, 0, 0), FxQuat{0, 0, 0, kOne}},
                                                     verdict::Health{100}, verdict::BodyRef{verdict::kNoBody});
    const verdict::EntityId b = verdict::SpawnEntity(w, verdict::Transform2D{V(0, 0, 0), FxQuat{0, 0, 0, kOne}},
                                                     verdict::Health{100}, verdict::BodyRef{verdict::kNoBody});
    const verdict::EntityId c = verdict::SpawnEntity(w, verdict::Transform2D{V(3, 0, 0), FxQuat{0, 0, 0, kOne}},
                                                     verdict::Health{100}, verdict::BodyRef{verdict::kNoBody});

    // A fixed tick-ordered command stream (the bus). Out-of-array order on purpose for some ticks to exercise the
    // tick filter; ApplyCommands per tick re-scans the whole stream in array order.
    const std::vector<verdict::Command> stream = {
        verdict::Command{0u, verdict::kCmdMove,    a, V(1, 0, 0)},
        verdict::Command{0u, verdict::kCmdMove,    b, V(0, 1, 0)},
        verdict::Command{1u, verdict::kCmdAbility, c, V(-25, 0, 0)},   // hp += -25 (arg.x>>kFrac)
        verdict::Command{1u, verdict::kCmdMove,    a, V(0, 0, 2)},
        verdict::Command{2u, verdict::kCmdDespawn, b, FxVec3{}},        // despawn b
        verdict::Command{3u, verdict::kCmdSpawn,   verdict::kNoEntity, V(5, 5, 0)},  // spawn a 4th (id must be 4)
        verdict::Command{4u, verdict::kCmdMove,    b, V(9, 9, 9)},      // DEAD target -> no-op
        verdict::Command{4u, verdict::kCmdAbility, a, V(10, 0, 0)},     // a.hp += 10
    };
    for (uint32_t t = 0; t <= 4; ++t) {
        verdict::ApplyCommands(w, stream, t);
        ++w.tick;
    }
    (void)a; (void)b; (void)c;
    return w;
}

int main() {
    HF_TEST_MAIN_INIT();

    // ---- SpawnEntity: monotonic ids + order[] + component round-trip ----
    {
        verdict::VerdictWorld w;
        const verdict::EntityId id1 = verdict::SpawnEntity(w, verdict::Transform2D{V(1, 2, 3), FxQuat{0, 0, 0, kOne}});
        const verdict::EntityId id2 = verdict::SpawnEntity(w, verdict::Transform2D{V(4, 5, 6), FxQuat{0, 0, 0, kOne}});
        check(id1 == 1u, "SpawnEntity: first id is the monotonic 1");
        check(id2 == 2u, "SpawnEntity: second id is the monotonic 2");
        check(w.nextId == 3u, "SpawnEntity: nextId advanced to 3");
        check(w.order.size() == 2 && w.order[0] == 1u && w.order[1] == 2u, "SpawnEntity: order[] is spawn order");
        check(verdict::IsLive(w, id1) && verdict::IsLive(w, id2), "SpawnEntity: both entities live");
        // Component get/has round-trip via the handle.
        const ecs::Entity e1 = w.handle[id1];
        check(w.reg.has<verdict::Transform2D>(e1), "SpawnEntity: Transform2D present");
        check(w.reg.get<verdict::Transform2D>(e1).pos.x == V(1, 2, 3).x, "SpawnEntity: Transform2D.pos round-trips");
        check(w.reg.has<verdict::Health>(e1) && w.reg.has<verdict::BodyRef>(e1), "SpawnEntity: Health+BodyRef present");
    }

    // ---- DespawnEntity: removes from order[] + the id is NEVER recycled (the pinned-identity contract, proof 2) ----
    {
        verdict::VerdictWorld w;
        const verdict::EntityId a = verdict::SpawnEntity(w, verdict::Transform2D{V(0, 0, 0), FxQuat{0, 0, 0, kOne}});
        const verdict::EntityId b = verdict::SpawnEntity(w, verdict::Transform2D{V(0, 0, 0), FxQuat{0, 0, 0, kOne}});
        check(a == 1u && b == 2u, "churn: ids 1,2 allocated");
        verdict::DespawnEntity(w, a);
        check(!verdict::IsLive(w, a), "churn: a despawned -> not live");
        check(w.order.size() == 1 && w.order[0] == b, "churn: order[] = [b] after despawn");
        check(w.nextId == 3u, "churn: nextId UNCHANGED by despawn (3)");
        // The next spawn must allocate id 3 — NOT recycle id 1, regardless of the ecs free-list recycling its INDEX.
        const verdict::EntityId c = verdict::SpawnEntity(w, verdict::Transform2D{V(0, 0, 0), FxQuat{0, 0, 0, kOne}});
        check(c == 3u, "churn: spawn->despawn->spawn re-derives the SAME monotonic next id (3, NOT recycled 1)");
        check(c != a, "churn: the new id is NOT the despawned id (pinned identity)");
        // The underlying ecs index MAY recycle (free-list), but the EntityId did not — that is the contract.
        std::printf("vd1-world: id alloc churn-independent (nextId:%u)\n", w.nextId);
    }

    // ---- ApplyCommands: tick-filtered + fixed-order + dead-target no-op (proof 3) ----
    {
        verdict::VerdictWorld w;
        const verdict::EntityId a = verdict::SpawnEntity(w, verdict::Transform2D{V(0, 0, 0), FxQuat{0, 0, 0, kOne}});
        const std::vector<verdict::Command> stream = {
            verdict::Command{0u, verdict::kCmdMove, a, V(1, 0, 0)},   // tick 0 fires
            verdict::Command{5u, verdict::kCmdMove, a, V(0, 100, 0)}, // tick 5 — must NOT fire on tick 0
        };
        verdict::ApplyCommands(w, stream, 0u);   // only the tick-0 command applies
        const ecs::Entity ea = w.handle[a];
        check(w.reg.get<verdict::Transform2D>(ea).pos.x == V(1, 0, 0).x, "apply: tick-0 move applied");
        check(w.reg.get<verdict::Transform2D>(ea).pos.y == 0, "apply: tick-5 move NOT applied at tick 0 (tick filter)");

        // Fixed array order: two moves at the same tick accumulate in array order (commutative add, but order is
        // pinned regardless).
        verdict::VerdictWorld w2;
        const verdict::EntityId p = verdict::SpawnEntity(w2, verdict::Transform2D{V(0, 0, 0), FxQuat{0, 0, 0, kOne}});
        const std::vector<verdict::Command> s2 = {
            verdict::Command{0u, verdict::kCmdMove, p, V(2, 0, 0)},
            verdict::Command{0u, verdict::kCmdMove, p, V(3, 0, 0)},
        };
        verdict::ApplyCommands(w2, s2, 0u);
        check(w2.reg.get<verdict::Transform2D>(w2.handle[p]).pos.x == V(5, 0, 0).x, "apply: fixed-order accumulate");

        // Dead/out-of-range target -> deterministic no-op (the world summary is unchanged by an out-of-range cmd).
        const verdict::VerdictMeasure before = verdict::MeasureVerdict(w2);
        const std::vector<verdict::Command> dead = {
            verdict::Command{0u, verdict::kCmdMove,    9999u, V(1, 1, 1)},  // unknown EntityId
            verdict::Command{0u, verdict::kCmdAbility, 9999u, V(50, 0, 0)},
            verdict::Command{0u, verdict::kCmdDespawn, 9999u, FxVec3{}},
        };
        verdict::ApplyCommands(w2, dead, 0u);
        const verdict::VerdictMeasure after = verdict::MeasureVerdict(w2);
        check(verdict::VerdictMeasuresEqual(before, after), "apply: dead/out-of-range target is a no-op (world unchanged)");
        std::printf("vd1-world: ApplyCommands {tickFiltered, fixedOrder, deadTargetNoOp} OK\n");
    }

    // ---- LowerToHullCommands == a hand-written ConvexCommand stream (proof 4) ----
    {
        verdict::VerdictWorld w;
        // Two entities BOUND to sim body indices 7 and 11.
        const verdict::EntityId a = verdict::SpawnEntity(w, verdict::Transform2D{V(0, 0, 0), FxQuat{0, 0, 0, kOne}},
                                                         verdict::Health{0}, verdict::BodyRef{7u});
        const verdict::EntityId b = verdict::SpawnEntity(w, verdict::Transform2D{V(0, 0, 0), FxQuat{0, 0, 0, kOne}},
                                                         verdict::Health{0}, verdict::BodyRef{11u});
        const std::vector<verdict::Command> stream = {
            verdict::Command{3u, verdict::kCmdImpulse,   a, V(2, 0, 0)},      // -> AddImpulse body 7
            verdict::Command{3u, verdict::kCmdMove,      b, V(9, 9, 9)},      // gameplay -> NOT lowered
            verdict::Command{3u, verdict::kCmdSetAngVel, b, V(0, 0, 1)},      // -> SetAngVel body 11
            verdict::Command{4u, verdict::kCmdImpulse,   a, V(5, 0, 0)},      // wrong tick -> not in tick-3 lowering
            verdict::Command{3u, verdict::kCmdImpulse,   9999u, V(1, 0, 0)},  // dead target -> skipped
        };
        const std::vector<convex::ConvexCommand> lowered = verdict::LowerToHullCommands(w, stream, 3u);
        const std::vector<convex::ConvexCommand> hand = {
            convex::ConvexCommand{3u, convex::kConvexCmdAddImpulse, 7u,  V(2, 0, 0)},
            convex::ConvexCommand{3u, convex::kConvexCmdSetAngVel,  11u, V(0, 0, 1)},
        };
        bool eq = lowered.size() == hand.size();
        if (eq) eq = std::memcmp(lowered.data(), hand.data(), hand.size() * sizeof(convex::ConvexCommand)) == 0;
        check(eq, "lower: LowerToHullCommands == hand-written ConvexCommand stream (byte-equal)");
        std::printf("vd1-world: LowerToHullCommands == hand-written stream (cmds:%u)\n", (unsigned)lowered.size());
        (void)a; (void)b;
    }

    // ---- Two-run byte-equality of the whole world over a fixed script (proof 1) ----
    {
        const verdict::VerdictWorld w1 = RunScript();
        const verdict::VerdictWorld w2 = RunScript();
        const verdict::VerdictMeasure m1 = verdict::MeasureVerdict(w1);
        const verdict::VerdictMeasure m2 = verdict::MeasureVerdict(w2);
        check(verdict::VerdictMeasuresEqual(m1, m2), "two-run: the fixed script yields a byte-identical world");
        // The script: 3 spawns + 1 despawn + 1 spawn = 3 live entities, nextId 5 (ids 1..4 allocated).
        check(m1.entities == 3u, "two-run: 3 entities live after the script");
        check(m1.nextId == 5u, "two-run: nextId is 5 (ids 1,2,3,4 allocated; despawn did not recycle)");
        // Print order[] as the spec line.
        std::printf("vd1-world: {entities:%u, order:[", m1.entities);
        for (size_t i = 0; i < w1.order.size(); ++i)
            std::printf("%s%u", i ? "," : "", w1.order[i]);
        std::printf("], nextId:%u} two-run BYTE-IDENTICAL\n", m1.nextId);
    }

    // =============================================================================================
    // Slice VD2 — THE SYSTEM SCHEDULE + THE GAMEPLAY TICK. OrderedView/ForEachOrdered iterate in
    // order[] sequence (the determinism contract — NOT the raw reg.view smallest-pool/insertion order
    // that renumbers under churn); the systems are pure integer (Q16.16, distance² compare, NO sqrt);
    // the schedule order is FIXED (swapping two systems changes the result).
    // =============================================================================================

    // ---- OrderedView yields order[] sequence AND differs from a churned raw reg.view (the contract) ----
    {
        // Build a CHURNED world: spawn 5 movers, despawn an EARLY one. The ecs pool's swap-pop moves the
        // LAST dense element into the freed slot, so the raw reg.view<>'s dense (insertion) iteration order
        // is RENUMBERED and does NOT match order[] (which keeps spawn order of the survivors). OrderedView
        // MUST follow order[] regardless — the determinism contract.
        verdict::VerdictWorld w;
        const verdict::EntityId a = verdict::SpawnMover(w, verdict::Transform2D{V(0, 0, 0), FxQuat{0,0,0,kOne}},
                                                        verdict::Health{10}, verdict::Velocity2D{V(1,0,0)});  // 1
        const verdict::EntityId b = verdict::SpawnMover(w, verdict::Transform2D{V(1, 0, 0), FxQuat{0,0,0,kOne}},
                                                        verdict::Health{10}, verdict::Velocity2D{V(1,0,0)});  // 2
        verdict::SpawnMover(w, verdict::Transform2D{V(2, 0, 0), FxQuat{0,0,0,kOne}},
                            verdict::Health{10}, verdict::Velocity2D{V(1,0,0)});  // 3
        verdict::SpawnMover(w, verdict::Transform2D{V(3, 0, 0), FxQuat{0,0,0,kOne}},
                            verdict::Health{10}, verdict::Velocity2D{V(1,0,0)});  // 4
        verdict::SpawnMover(w, verdict::Transform2D{V(4, 0, 0), FxQuat{0,0,0,kOne}},
                            verdict::Health{10}, verdict::Velocity2D{V(1,0,0)});  // 5
        verdict::DespawnEntity(w, b);                         // despawn id 2 -> pool swap-pops id 5 into its slot
        (void)a;

        // OrderedView is EXACTLY order[]-filtered (here every live entity has Transform2D+Velocity2D).
        const std::vector<verdict::EntityId> ov = verdict::OrderedView<verdict::Transform2D, verdict::Velocity2D>(w);
        check(ov.size() == w.order.size(), "OrderedView: size == live count");
        bool matchesOrder = (ov.size() == w.order.size());
        for (size_t i = 0; matchesOrder && i < ov.size(); ++i) matchesOrder = (ov[i] == w.order[i]);
        check(matchesOrder, "OrderedView: yields the EXACT order[] sequence");
        check(ov.size() == 4 && ov[0] == 1u && ov[1] == 3u && ov[2] == 4u && ov[3] == 5u,
              "OrderedView: order[] is [1,3,4,5] after despawning id 2");

        // The raw reg.view<> visits entities in dense (insertion, index-recycled) order — here the
        // recycled slot puts id 4 in id 2's old dense position, so the raw view order is [1,4,3] which
        // DIFFERS from order[] = [1,3,4]. Falsify the contract against it.
        std::vector<verdict::EntityId> raw;
        for (auto&& [e, xf, vel] : w.reg.view<verdict::Transform2D, verdict::Velocity2D>()) {
            (void)xf; (void)vel;
            // map the ecs handle back to its EntityId via the handle map (linear — tiny scene)
            for (auto& kv : w.handle) if (kv.second == e) { raw.push_back(kv.first); break; }
        }
        check(raw.size() == ov.size(), "OrderedView: raw view sees the same entity SET");
        bool rawDiffers = (raw.size() == ov.size());
        if (rawDiffers) { rawDiffers = false; for (size_t i = 0; i < raw.size(); ++i) if (raw[i] != ov[i]) rawDiffers = true; }
        check(rawDiffers, "OrderedView: raw reg.view order DIFFERS from order[] under churn (the contract)");
        std::printf("vd2-tick: OrderedView order-stable (order[] != reg.view churn)\n");
    }

    // ---- SystemMovement: exact Q16.16 move rule ----
    {
        verdict::VerdictWorld w;
        const verdict::EntityId p = verdict::SpawnMover(w, verdict::Transform2D{V(2, 3, 0), FxQuat{0,0,0,kOne}},
                                                        verdict::Health{0}, verdict::Velocity2D{V(1, -1, 0)});
        const FxVec3 start = w.reg.get<verdict::Transform2D>(w.handle[p]).pos;
        verdict::SystemMovement(w);
        const FxVec3 end = w.reg.get<verdict::Transform2D>(w.handle[p]).pos;
        check(end.x == V(3, 2, 0).x && end.y == V(3, 2, 0).y, "SystemMovement: pos = FxAdd(pos, vel) exact Q16.16");
        std::printf("vd2-tick: move rule {start:(%d,%d), end:(%d,%d)} exact\n",
                    (int)(start.x >> fpx::kFrac), (int)(start.y >> fpx::kFrac),
                    (int)(end.x >> fpx::kFrac),   (int)(end.y >> fpx::kFrac));
    }

    // ---- SystemCollect: removes the overlapping pickup + bumps score by value; leaves non-overlapping ----
    {
        verdict::VerdictWorld w;
        // Player at origin with a Score; one pickup ON the player (overlap), one far away (no overlap).
        const verdict::EntityId player = verdict::SpawnEntity(w, verdict::Transform2D{V(0,0,0), FxQuat{0,0,0,kOne}},
                                                              verdict::Health{0}, verdict::BodyRef{verdict::kNoBody});
        w.reg.add<verdict::Score>(w.handle[player], verdict::Score{0});
        const verdict::EntityId near = verdict::SpawnEntity(w, verdict::Transform2D{V(0,0,0), FxQuat{0,0,0,kOne}});
        w.reg.add<verdict::Pickup>(w.handle[near], verdict::Pickup{7});
        const verdict::EntityId far = verdict::SpawnEntity(w, verdict::Transform2D{V(5,5,0), FxQuat{0,0,0,kOne}});
        w.reg.add<verdict::Pickup>(w.handle[far], verdict::Pickup{99});

        verdict::SystemCollect(w, player, kOne);   // radius 1.0 in Q16.16
        check(!verdict::IsLive(w, near), "SystemCollect: overlapping pickup despawned");
        check(verdict::IsLive(w, far), "SystemCollect: non-overlapping pickup left alive");
        check(w.reg.get<verdict::Score>(w.handle[player]).points == 7, "SystemCollect: score bumped by pickup.value (7)");
    }

    // ---- SystemDamage: deterministic integer hazard decay ----
    {
        verdict::VerdictWorld w;
        const verdict::HazardRegion hz{V(-1,-1,0).x, V(-1,-1,0).y, V(1,1,0).x, V(1,1,0).y};
        const verdict::EntityId inH  = verdict::SpawnEntity(w, verdict::Transform2D{V(0,0,0), FxQuat{0,0,0,kOne}},
                                                            verdict::Health{5}, verdict::BodyRef{verdict::kNoBody});
        const verdict::EntityId outH = verdict::SpawnEntity(w, verdict::Transform2D{V(5,5,0), FxQuat{0,0,0,kOne}},
                                                            verdict::Health{5}, verdict::BodyRef{verdict::kNoBody});
        verdict::SystemDamage(w, hz);
        check(w.reg.get<verdict::Health>(w.handle[inH]).hp == 4, "SystemDamage: -1 hp inside the hazard");
        check(w.reg.get<verdict::Health>(w.handle[outH]).hp == 5, "SystemDamage: untouched outside the hazard");
        // Decay clamps at 0 — apply enough ticks.
        for (int i = 0; i < 10; ++i) verdict::SystemDamage(w, hz);
        check(w.reg.get<verdict::Health>(w.handle[inH]).hp == 0, "SystemDamage: clamps at 0 (deterministic)");
    }

    // ---- StepGameplay deterministic over a fixed script (two runs byte-equal) ----
    auto RunGameplay = []() {
        verdict::VerdictWorld w;
        const verdict::HazardRegion hz{V(-1,-1,0).x, V(-2,-2,0).y, V(1,1,0).x, V(2,2,0).y};
        // Player at (-3,0) moving +X each tick toward two pickups at (0,0) and (2,0); carries a Score.
        const verdict::EntityId player = verdict::SpawnMover(w, verdict::Transform2D{V(-3,0,0), FxQuat{0,0,0,kOne}},
                                                             verdict::Health{20}, verdict::Velocity2D{V(1,0,0)});
        w.reg.add<verdict::Score>(w.handle[player], verdict::Score{0});
        const verdict::EntityId pk0 = verdict::SpawnEntity(w, verdict::Transform2D{V(0,0,0), FxQuat{0,0,0,kOne}});
        w.reg.add<verdict::Pickup>(w.handle[pk0], verdict::Pickup{5});
        const verdict::EntityId pk1 = verdict::SpawnEntity(w, verdict::Transform2D{V(2,0,0), FxQuat{0,0,0,kOne}});
        w.reg.add<verdict::Pickup>(w.handle[pk1], verdict::Pickup{8});
        (void)pk0; (void)pk1;
        const std::vector<verdict::Command> stream = {};   // movement is driven by Velocity2D
        for (uint32_t t = 0; t < 6; ++t)
            verdict::StepGameplay(w, stream, t, hz, player, kOne);
        return std::pair<verdict::VerdictWorld, verdict::EntityId>(std::move(w), player);
    };
    {
        auto r1 = RunGameplay();
        auto r2 = RunGameplay();
        const verdict::VerdictMeasure m1 = verdict::MeasureVerdict(r1.first);
        const verdict::VerdictMeasure m2 = verdict::MeasureVerdict(r2.first);
        check(verdict::VerdictMeasuresEqual(m1, m2), "StepGameplay: fixed script two runs BYTE-IDENTICAL");
        const int32_t score = r1.first.reg.get<verdict::Score>(r1.first.handle[r1.second]).points;
        check(score == 13, "StepGameplay: both pickups collected (score 5+8=13)");
        std::printf("vd2-tick: {ticks:6, entities:%u, score:%d} two-run BYTE-IDENTICAL\n", m1.entities, score);
    }

    // ---- The schedule order is FIXED: swapping two systems changes the result ----
    {
        // A player crosses a pickup that sits exactly where the player ARRIVES after one move. With the
        // fixed schedule (move THEN collect) the player reaches the pickup and collects it. If collect ran
        // BEFORE move, the player would still be at the start and NOT overlap -> no collect. Proving order
        // matters and is pinned.
        verdict::HazardRegion hz{V(-9,-9,0).x, V(-9,-9,0).y, V(-9,-9,0).x, V(-9,-9,0).y};  // empty hazard
        auto build = [&](verdict::VerdictWorld& w) {
            const verdict::EntityId player = verdict::SpawnMover(w, verdict::Transform2D{V(0,0,0), FxQuat{0,0,0,kOne}},
                                                                 verdict::Health{0}, verdict::Velocity2D{V(1,0,0)});
            w.reg.add<verdict::Score>(w.handle[player], verdict::Score{0});
            const verdict::EntityId pk = verdict::SpawnEntity(w, verdict::Transform2D{V(1,0,0), FxQuat{0,0,0,kOne}});
            w.reg.add<verdict::Pickup>(w.handle[pk], verdict::Pickup{4});
            return player;
        };
        // Radius 0.5 (Q16.16): a pickup at distance 0 collects, at distance 1.0 does NOT (the swap case).
        const fx kR = kOne / 2;
        // Fixed schedule: move (0->1) then collect (player AT pickup, dist 0) -> collected.
        verdict::VerdictWorld wFixed;
        const verdict::EntityId pF = build(wFixed);
        verdict::SystemMovement(wFixed);
        verdict::SystemCollect(wFixed, pF, kR);
        const int32_t fixedScore = wFixed.reg.get<verdict::Score>(wFixed.handle[pF]).points;
        // Swapped: collect (player still at 0, pickup at 1 -> dist 1.0 > 0.5, no overlap) then move -> NOT collected.
        verdict::VerdictWorld wSwap;
        const verdict::EntityId pS = build(wSwap);
        verdict::SystemCollect(wSwap, pS, kR);
        verdict::SystemMovement(wSwap);
        const int32_t swapScore = wSwap.reg.get<verdict::Score>(wSwap.handle[pS]).points;
        check(fixedScore == 4 && swapScore == 0,
              "schedule: swapping move/collect changes the result (order is pinned)");
    }

    if (g_fail == 0) std::printf("verdict_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
