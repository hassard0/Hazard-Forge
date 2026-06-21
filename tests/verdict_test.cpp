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

    // =============================================================================================
    // Slice VD3 — COMPOSING THE PHYSICS SUBSYSTEM — ONE WORLD TICK. Pure CPU integer. The embedded sim
    // is the FROZEN warmhull warm+sleep hull world; StepWorld composes the gameplay systems + the sim
    // via a deterministic bidirectional BodyRef sync. The make-or-break contract: the embedded sim is
    // NEVER perturbed (byte-identical to a STANDALONE StepWarmSleepHullWorldN over the same scene).
    // =============================================================================================
    namespace warmhull = hf::sim::warmhull;
    namespace gjk      = hf::sim::gjk;
    {
        // A shared deterministic warm+sleep config (the WH4 lineage; angDamp OFF).
        auto makeCfg = []() {
            const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
            convex::ConvexStepConfig c;
            c.gravity = FxVec3{0, kGravY, 0};
            c.dt = kOne / 60; c.solveIters = 8; c.restitution = 0; c.slop = kOne / 64;
            c.beta = (fx)((int64_t)2 * kOne / 10); c.linDamp = (fx)((int64_t)95 * kOne / 100);
            c.angDamp = kOne; c.posIters = 4;
            warmhull::HullSleepConfig sc;
            sc.warm = c; sc.sleepThreshold = kOne; sc.wakeThreshold = (fx)(2 * (int)kOne); sc.sleepTicks = 30;
            return sc;
        };
        const warmhull::HullSleepConfig kCfg = makeCfg();
        const verdict::HazardRegion kHazard{V(-9,-9,0).x, V(-9,-9,0).y, V(-9,-9,0).x, V(-9,-9,0).y}; // empty
        const fx kCollectR = kOne;
        const uint32_t kTicks = 24u;

        // The composed scene: a static support box (body 0) + a dynamic "player" hull body (body 1)
        // above it. A player entity bound to body 1; a static-support entity bound to body 0; a couple
        // of gameplay-only pickups. A kCmdImpulse nudges the player body; gameplay collects a pickup.
        auto buildSimScene = []() {
            gjk::HullWorld sim;
            { fpx::FxBody b; b.pos={0,0,0}; b.orient={0,0,0,kOne}; b.invMass=0; b.flags=0u; b.vel={0,0,0}; b.angVel={0,0,0}; sim.bodies.push_back(b); }
            sim.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));            // 0 static support
            { fpx::FxBody b; b.pos={0, (fx)((int64_t)3*kOne), 0}; b.orient={0,0,0,kOne}; b.invMass=kOne; b.flags=fpx::kFlagDynamic; b.vel={0,0,0}; b.angVel={0,0,0}; sim.bodies.push_back(b); }
            sim.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));            // 1 dynamic player body
            return sim;
        };
        const std::vector<verdict::Command> kStream = {
            verdict::Command{0u, verdict::kCmdImpulse, 2u /*player entity id*/, V(1,0,0)},
        };
        auto runWorld = [&](verdict::VerdictWorld& w, verdict::EntityId& outPlayer) {
            w.sim = buildSimScene();
            // entity 1 = static support bound to body 0; entity 2 = player bound to body 1.
            const verdict::EntityId support = verdict::SpawnEntity(w, verdict::Transform2D{V(0,0,0), FxQuat{0,0,0,kOne}});
            verdict::BindBody(w, support, 0u);
            // The player is bound to body 1 + driven by the kCmdImpulse (a SIM verb) — NOT by a Velocity2D,
            // so SyncComponentsToBodies does NOT overwrite the body's sim-evolved velocity (the body is driven
            // purely by the impulse + the frozen sim -> the embedded sim matches the standalone reference).
            const verdict::EntityId player = verdict::SpawnEntity(w, verdict::Transform2D{V(0,3,0), FxQuat{0,0,0,kOne}});
            w.reg.add<verdict::Score>(w.handle[player], verdict::Score{0});
            verdict::BindBody(w, player, 1u);
            // gameplay-only pickups (unbound) near the player's start so a collect fires.
            const verdict::EntityId pk0 = verdict::SpawnEntity(w, verdict::Transform2D{V(0,3,0), FxQuat{0,0,0,kOne}});
            w.reg.add<verdict::Pickup>(w.handle[pk0], verdict::Pickup{5});
            (void)support; (void)pk0;
            outPlayer = player;
            verdict::StepWorldN(w, kStream, 0u, kHazard, player, kCollectR, kCfg, kTicks);
        };

        // ---- PROOF (1) the composed world is deterministic (whole world + sim TRIPLE two-run equal) ----
        verdict::VerdictWorld w1, w2; verdict::EntityId p1, p2;
        runWorld(w1, p1); runWorld(w2, p2);
        {
            const verdict::VerdictMeasure m1 = verdict::MeasureVerdict(w1);
            const verdict::VerdictMeasure m2 = verdict::MeasureVerdict(w2);
            const bool simEq = warmhull::WarmHullStatesEqual(w1.sim.bodies, w1.cache, w1.sleep,
                                                             w2.sim.bodies, w2.cache, w2.sleep);
            check(verdict::VerdictMeasuresEqual(m1, m2) && simEq,
                  "vd3: StepWorld whole-world + sim TRIPLE two-run BYTE-IDENTICAL");
            std::printf("vd3-world: {entities:%u, bodies:%zu, ticks:%u} two-run BYTE-IDENTICAL\n",
                        m1.entities, w1.sim.bodies.size(), kTicks);
        }

        // ---- PROOF (2) Verdict did NOT perturb the frozen sim: embedded == STANDALONE ----
        {
            // The standalone path: build the SAME sim scene + feed the SAME lowered impulse stream
            // through gjk::ApplyHullCommands BEFORE each StepWarmSleepHullWorld (NO sync, NO gameplay).
            gjk::HullWorld sa = buildSimScene();
            warmhull::HullCache saCache;
            std::vector<warmhull::HullSleepState> saSleep;
            // The lowered impulse: entity 2 (player) is bound to body 1 -> bodyId 1, kConvexCmdAddImpulse.
            const std::vector<convex::ConvexCommand> saStream = {
                convex::ConvexCommand{0u, convex::kConvexCmdAddImpulse, 1u, V(1,0,0)},
            };
            for (uint32_t t = 0; t < kTicks; ++t) {
                gjk::ApplyHullCommands(sa, saStream, t);
                warmhull::StepWarmSleepHullWorld(sa, saCache, saSleep, kCfg);
            }
            const bool unperturbed = warmhull::WarmHullStatesEqual(w1.sim.bodies, w1.cache, w1.sleep,
                                                                   sa.bodies, saCache, saSleep);
            check(unperturbed, "vd3: embedded sim == standalone StepWarmSleepHullWorldN (bodies BIT-EXACT)");
            std::printf("vd3-world: embedded sim == standalone (bodies BIT-EXACT)\n");
        }

        // ---- PROOF (3) the syncs are pure functions + a body-bound entity tracks its body ----
        {
            // Calling each sync twice on the same world is byte-equal (no hidden state).
            verdict::VerdictWorld wp; verdict::EntityId pp; runWorld(wp, pp);
            const std::vector<fpx::FxBody> bodiesBefore = wp.sim.bodies;
            verdict::SyncComponentsToBodies(wp);
            const std::vector<fpx::FxBody> afterP2B1 = wp.sim.bodies;
            verdict::SyncComponentsToBodies(wp);
            const std::vector<fpx::FxBody> afterP2B2 = wp.sim.bodies;
            const bool p2bPure = gjk::HullBodiesEqual(afterP2B1, afterP2B2);
            // b2p twice: capture the player Transform2D after each call.
            verdict::SyncBodiesToComponents(wp);
            const verdict::Transform2D xf1 = wp.reg.get<verdict::Transform2D>(wp.handle[pp]);
            verdict::SyncBodiesToComponents(wp);
            const verdict::Transform2D xf2 = wp.reg.get<verdict::Transform2D>(wp.handle[pp]);
            const bool b2pPure = (xf1.pos.x == xf2.pos.x && xf1.pos.y == xf2.pos.y && xf1.pos.z == xf2.pos.z);
            check(p2bPure && b2pPure, "vd3: sync {p2b, b2p} two-call BYTE-EQUAL (pure functions)");
            // A body-bound entity's Transform2D tracks its bound body's settled pos.
            const fpx::FxBody& body1 = wp.sim.bodies[1];
            const bool tracks = (xf2.pos.x == body1.pos.x && xf2.pos.y == body1.pos.y && xf2.pos.z == body1.pos.z);
            check(tracks, "vd3: body-bound entity Transform2D tracks its bound body pos");
            std::printf("vd3-world: sync pure {p2b, b2p} two-call BYTE-EQUAL\n");
            (void)bodiesBefore;
        }

        // ---- An unbound (kNoBody) entity is a sync no-op (gameplay-only) ----
        {
            verdict::VerdictWorld w;
            w.sim = buildSimScene();
            const verdict::EntityId go = verdict::SpawnMover(w, verdict::Transform2D{V(7,7,0), FxQuat{0,0,0,kOne}},
                                                             verdict::Health{0}, verdict::Velocity2D{V(2,2,0)});
            // unbound BodyRef (SpawnMover leaves it kNoBody) — the sync must not touch any body.
            const std::vector<fpx::FxBody> before = w.sim.bodies;
            verdict::SyncComponentsToBodies(w);
            check(gjk::HullBodiesEqual(before, w.sim.bodies), "vd3: unbound entity is a SyncComponentsToBodies no-op");
            // And SyncBodiesToComponents must not move the unbound entity's transform.
            const verdict::Transform2D bxf = w.reg.get<verdict::Transform2D>(w.handle[go]);
            verdict::SyncBodiesToComponents(w);
            const verdict::Transform2D axf = w.reg.get<verdict::Transform2D>(w.handle[go]);
            check(bxf.pos.x == axf.pos.x && bxf.pos.y == axf.pos.y, "vd3: unbound entity is a SyncBodiesToComponents no-op");
        }

        // ---- The schedule order is pinned: swapping the sync halves changes the result ----
        {
            // Run the canonical schedule (push BEFORE step, read-back AFTER step) for ONE tick, vs the
            // swapped schedule (read-back BEFORE step, push AFTER step). With a nonzero gameplay velocity
            // driving the body, the canonical path integrates the pushed velocity this tick (body moves);
            // the swapped path reads the pre-step pose into the transform + pushes too late.
            auto build = [&](verdict::VerdictWorld& w, verdict::EntityId& player) {
                w.sim = buildSimScene();
                player = verdict::SpawnMover(w, verdict::Transform2D{V(0,3,0), FxQuat{0,0,0,kOne}},
                                             verdict::Health{0}, verdict::Velocity2D{V(3,0,0)});  // a fast +X drive
                w.reg.add<verdict::Score>(w.handle[player], verdict::Score{0});
                verdict::BindBody(w, player, 1u);
            };
            const std::vector<verdict::Command> empty = {};
            // canonical
            verdict::VerdictWorld wc; verdict::EntityId pc; build(wc, pc);
            verdict::SyncComponentsToBodies(wc);
            warmhull::StepWarmSleepHullWorld(wc.sim, wc.cache, wc.sleep, kCfg);
            verdict::SyncBodiesToComponents(wc);
            const verdict::Transform2D xc = wc.reg.get<verdict::Transform2D>(wc.handle[pc]);
            // swapped (read-back first, push after the step)
            verdict::VerdictWorld ws; verdict::EntityId ps; build(ws, ps);
            verdict::SyncBodiesToComponents(ws);
            warmhull::StepWarmSleepHullWorld(ws.sim, ws.cache, ws.sleep, kCfg);
            verdict::SyncComponentsToBodies(ws);
            const verdict::Transform2D xs = ws.reg.get<verdict::Transform2D>(ws.handle[ps]);
            check(!(xc.pos.x == xs.pos.x && xc.pos.y == xs.pos.y),
                  "vd3: swapping the sync halves changes the result (schedule pinned)");
            (void)empty;
        }
    }

    // =============================================================================================
    // Slice VD4 — THE HETEROGENEOUS SNAPSHOT/RESTORE + EQUALITY. Pure CPU integer. SnapshotWorld
    // captures the ENTIRE heterogeneous world (entities + EVERY component pool in order[] sequence +
    // the embedded sim TRIPLE bodies/cache/sleep); RestoreWorld REBUILDS it (handle churn irrelevant);
    // VerdictStatesEqual compares the whole thing order[]-keyed. THE CRUX: completeness — advance ->
    // snapshot -> advance (diverge) -> restore -> re-advance is byte-identical to a never-diverged
    // reference (nothing escapes the snapshot), and a restore that OMITS the sim TRIPLE DIVERGES.
    // =============================================================================================
    {
        // The same deterministic warm+sleep config + composed scene as VD3 (re-declared — the VD3
        // block's locals are out of scope here).
        auto makeCfg = []() {
            const fx kGravY = (fx)(-9.8 * (double)kOne + (-9.8 < 0 ? -0.5 : 0.5));
            convex::ConvexStepConfig c;
            c.gravity = FxVec3{0, kGravY, 0};
            c.dt = kOne / 60; c.solveIters = 8; c.restitution = 0; c.slop = kOne / 64;
            c.beta = (fx)((int64_t)2 * kOne / 10); c.linDamp = (fx)((int64_t)95 * kOne / 100);
            c.angDamp = kOne; c.posIters = 4;
            warmhull::HullSleepConfig sc;
            sc.warm = c; sc.sleepThreshold = kOne; sc.wakeThreshold = (fx)(2 * (int)kOne); sc.sleepTicks = 30;
            return sc;
        };
        const warmhull::HullSleepConfig kCfg = makeCfg();
        const verdict::HazardRegion kHazard{V(-9,-9,0).x, V(-9,-9,0).y, V(-9,-9,0).x, V(-9,-9,0).y}; // empty
        const fx kCollectR = kOne;

        auto buildSimScene = []() {
            gjk::HullWorld sim;
            { fpx::FxBody b; b.pos={0,0,0}; b.orient={0,0,0,kOne}; b.invMass=0; b.flags=0u; b.vel={0,0,0}; b.angVel={0,0,0}; sim.bodies.push_back(b); }
            sim.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));            // 0 static support
            { fpx::FxBody b; b.pos={0, (fx)((int64_t)3*kOne), 0}; b.orient={0,0,0,kOne}; b.invMass=kOne; b.flags=fpx::kFlagDynamic; b.vel={0,0,0}; b.angVel={0,0,0}; sim.bodies.push_back(b); }
            sim.hulls.push_back(gjk::MakeBox(kOne, kOne, kOne));            // 1 dynamic player body
            return sim;
        };
        // A command stream that nudges the player body (a sim verb) — exercises the sim TRIPLE evolution.
        const std::vector<verdict::Command> kStream = {
            verdict::Command{0u, verdict::kCmdImpulse, 2u /*player entity id*/, V(1,0,0)},
        };
        // Build the gameplay+physics world: a static-support entity (body 0), a player (body 1, Score),
        // a couple of pickups (one near the player so a collect fires + despawns it), and a mover with a
        // Velocity2D (so EVERY component type — Transform2D/Health/BodyRef/Velocity2D/Pickup/Score — is
        // present in the snapshot).
        auto buildWorld = [&](verdict::VerdictWorld& w, verdict::EntityId& outPlayer) {
            w = verdict::VerdictWorld{};
            w.sim = buildSimScene();
            const verdict::EntityId support = verdict::SpawnEntity(w, verdict::Transform2D{V(0,0,0), FxQuat{0,0,0,kOne}});
            verdict::BindBody(w, support, 0u);
            const verdict::EntityId player = verdict::SpawnEntity(w, verdict::Transform2D{V(0,3,0), FxQuat{0,0,0,kOne}},
                                                                  verdict::Health{77}, verdict::BodyRef{verdict::kNoBody});
            w.reg.add<verdict::Score>(w.handle[player], verdict::Score{0});
            verdict::BindBody(w, player, 1u);
            const verdict::EntityId pk0 = verdict::SpawnEntity(w, verdict::Transform2D{V(0,3,0), FxQuat{0,0,0,kOne}});
            w.reg.add<verdict::Pickup>(w.handle[pk0], verdict::Pickup{5});
            const verdict::EntityId mover = verdict::SpawnMover(w, verdict::Transform2D{V(4,4,0), FxQuat{0,0,0,kOne}},
                                                               verdict::Health{10}, verdict::Velocity2D{V(0,0,0)});
            (void)support; (void)pk0; (void)mover;
            outPlayer = player;
        };

        const uint32_t kN = 6u;   // advance to the snapshot point
        const uint32_t kM = 8u;   // diverge / re-advance length

        // ---- PROOF (1) snapshot->restore round-trip is BIT-EXACT over the whole world ----
        {
            verdict::VerdictWorld w; verdict::EntityId player; buildWorld(w, player);
            verdict::StepWorldN(w, kStream, 0u, kHazard, player, kCollectR, kCfg, kN);
            const verdict::VerdictSnapshot snap = verdict::SnapshotWorld(w);
            // Restore INTO A FRESH world (the ecs handles WILL differ — the contract is order[]-keyed).
            verdict::VerdictWorld r;
            verdict::RestoreWorld(r, snap);
            const bool roundTrip = verdict::VerdictStatesEqual(w, r);
            check(roundTrip, "vd4: snapshot->restore round-trip BIT-EXACT");
            // And the restored snapshot equals the original snapshot (snapshot-level equality).
            check(verdict::VerdictSnapshotsEqual(snap, verdict::SnapshotWorld(r)),
                  "vd4: restored snapshot == original snapshot");
            if (roundTrip) std::printf("vd4-snap: snapshot->restore round-trip BIT-EXACT\n");
        }

        // ---- PROOF (2) THE COMPLETENESS CRUX: advance N -> snapshot -> advance M (diverge) ->
        // restore -> re-advance M == a reference that advanced N+M straight (nothing escapes) ----
        {
            // The reference: advance N+M straight through with the same inputs.
            verdict::VerdictWorld ref; verdict::EntityId refPlayer; buildWorld(ref, refPlayer);
            verdict::StepWorldN(ref, kStream, 0u, kHazard, refPlayer, kCollectR, kCfg, kN + kM);

            // The rollback path: advance N, snapshot, advance M with a DIVERGING stream, restore, re-advance.
            verdict::VerdictWorld w; verdict::EntityId player; buildWorld(w, player);
            verdict::StepWorldN(w, kStream, 0u, kHazard, player, kCollectR, kCfg, kN);
            const verdict::VerdictSnapshot snap = verdict::SnapshotWorld(w);
            // A diverging mispredicted stream advances M ticks -> diverged. A kCmdImpulse on the player's
            // BOUND body perturbs the sim TRIPLE (a kCmdMove on the player's Transform2D would be clobbered
            // by SyncBodiesToComponents each tick — the body-bound transform is owned by the sim).
            const std::vector<verdict::Command> kBad = {
                verdict::Command{kN, verdict::kCmdImpulse, player, V(-3,0,0)},
            };
            verdict::StepWorldN(w, kBad, kN, kHazard, player, kCollectR, kCfg, kM);
            const bool diverged = !verdict::VerdictStatesEqual(w, ref);   // the mispredict really diverged
            // Restore the snapshot + re-advance the CORRECT stream for M ticks.
            verdict::RestoreWorld(w, snap);
            verdict::StepWorldN(w, kStream, kN, kHazard, player, kCollectR, kCfg, kM);
            const bool complete = verdict::VerdictStatesEqual(w, ref);
            check(diverged, "vd4: mispredicted stream genuinely diverged before restore");
            check(complete, "vd4: completeness — restore+re-advance == reference BIT-EXACT");
            if (complete)
                std::printf("vd4-snap: completeness {advance:%u, restore@:%u, readvance:%u} == reference BIT-EXACT\n",
                            kN + kM, kN, kM);
        }

        // ---- PROOF (3) the necessity of the sim third: a restore that OMITS the WarmHullSnapshot
        // (bodies+cache+sleep) DIVERGES; including it matches (the WH5 TRIPLE lesson at the world level) ----
        {
            verdict::VerdictWorld ref; verdict::EntityId refPlayer; buildWorld(ref, refPlayer);
            verdict::StepWorldN(ref, kStream, 0u, kHazard, refPlayer, kCollectR, kCfg, kN + kM);

            verdict::VerdictWorld w; verdict::EntityId player; buildWorld(w, player);
            verdict::StepWorldN(w, kStream, 0u, kHazard, player, kCollectR, kCfg, kN);
            const verdict::VerdictSnapshot snap = verdict::SnapshotWorld(w);
            // diverge M ticks
            const std::vector<verdict::Command> kBad = {
                verdict::Command{kN, verdict::kCmdImpulse, player, V(-3,0,0)},
            };
            verdict::StepWorldN(w, kBad, kN, kHazard, player, kCollectR, kCfg, kM);

            // (a) An OMITTING restore: restore everything EXCEPT the sim TRIPLE (leave the diverged sim
            // in place) — the rolled-back peer resumes with stale bodies/cache/sleep -> MUST diverge. We
            // build a snapshot whose simSnap is the WRONG (diverged) sim TRIPLE, restore w with it, and
            // re-advance (w already carries the diverged sim; RestoreWarmHull rewrites it with the diverged
            // capture — a no-op equivalent, modelling "the sim third was NOT rolled back").
            {
                verdict::VerdictSnapshot noSim = snap;
                noSim.simSnap = verdict::warmhull::SnapshotWarmHull(w.sim, w.cache, w.sleep, w.tick); // the WRONG (diverged) sim
                verdict::RestoreWorld(w, noSim);   // entities/components from snap, but sim = diverged
                verdict::StepWorldN(w, kStream, kN, kHazard, player, kCollectR, kCfg, kM);
                const bool omitDiverges = !verdict::VerdictStatesEqual(w, ref);
                check(omitDiverges, "vd4: sim-third OMITTED restore DIVERGES (the TRIPLE is necessary)");
            }
            // (b) The full restore (sim TRIPLE included) MATCHES.
            verdict::RestoreWorld(w, snap);
            verdict::StepWorldN(w, kStream, kN, kHazard, player, kCollectR, kCfg, kM);
            const bool includeMatches = verdict::VerdictStatesEqual(w, ref);
            check(includeMatches, "vd4: sim-third INCLUDED restore == reference (the TRIPLE captured)");
            if (includeMatches)
                std::printf("vd4-snap: sim-third necessary (omit -> diverge, include -> ==)\n");
        }

        // ---- VerdictStatesEqual compares in order[] order (order-stable, handle-independent) ----
        {
            // Two worlds built the SAME way are equal; a despawn (changing order[]) makes them unequal.
            verdict::VerdictWorld a; verdict::EntityId pa; buildWorld(a, pa);
            verdict::VerdictWorld b; verdict::EntityId pb; buildWorld(b, pb);
            check(verdict::VerdictStatesEqual(a, b), "vd4: identically-built worlds are VerdictStatesEqual");
            // Restore a from b's snapshot — handles differ, equality must still hold (order[]-keyed).
            const verdict::VerdictSnapshot sb = verdict::SnapshotWorld(b);
            verdict::RestoreWorld(a, sb);
            check(verdict::VerdictStatesEqual(a, b), "vd4: restore re-derives id->value mapping despite handle churn");
            // A despawn changes order[] -> NOT equal.
            verdict::DespawnEntity(b, pb);
            check(!verdict::VerdictStatesEqual(a, b), "vd4: a despawn (order[] change) breaks VerdictStatesEqual");
        }

        // ---- A despawn-then-restore preserves the retired-id contract (nextId survives) ----
        {
            verdict::VerdictWorld w; verdict::EntityId player; buildWorld(w, player);
            verdict::StepWorldN(w, kStream, 0u, kHazard, player, kCollectR, kCfg, kN);
            // Despawn the player, then snapshot + restore. nextId must be unchanged; the next spawn must
            // NOT reuse the retired id.
            const verdict::EntityId nextBefore = w.nextId;
            verdict::DespawnEntity(w, player);
            const verdict::VerdictSnapshot snap = verdict::SnapshotWorld(w);
            verdict::VerdictWorld r; verdict::RestoreWorld(r, snap);
            check(r.nextId == nextBefore, "vd4: restore preserves nextId (the retired-id contract)");
            check(!verdict::IsLive(r, player), "vd4: a despawned entity stays absent after restore");
            const verdict::EntityId reborn = verdict::SpawnEntity(r, verdict::Transform2D{V(1,1,0), FxQuat{0,0,0,kOne}});
            check(reborn == nextBefore && reborn != player,
                  "vd4: next spawn after restore does NOT reuse the retired id");
        }
    }

    if (g_fail == 0) std::printf("verdict_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
