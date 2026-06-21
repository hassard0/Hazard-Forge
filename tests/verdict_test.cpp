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

    if (g_fail == 0) std::printf("verdict_test: ALL PASS\n");
    return g_fail == 0 ? 0 : 1;
}
