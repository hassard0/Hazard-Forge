// Unit test for fpx collision EVENTS (engine/sim/fpx.h, issue #44) — the additive event API for sample
// authors who drive the raw deterministic core `StepWorld` directly.
//
// Proves: (1) CollectContacts emits the right deterministic events; (2) the new 5-arg StepWorld overload is
// PHYSICS-IDENTICAL to the byte-unchanged 4-arg StepWorld (so every fpx golden is invariant); (3) events
// are consistent with CountResidualOverlaps. Pure hf_core (fpx.h is pure integer + math.h only), so this is
// standalone-clang-compilable: clang++ -std=c++20 -I engine -I tests tests/fpx_events_test.cpp

#include "sim/fpx.h"

#include <cstdint>
#include <cstdio>
#include <vector>
#include "test_main.h"

using namespace hf::sim::fpx;

static int g_fail = 0;
static void check(bool c, const char* what) {
    if (!c) { std::printf("FAIL: %s\n", what); ++g_fail; }
    else    { std::printf("PASS: %s\n", what); }
}

// Two overlapping unit-radius spheres: A at origin, B at x=0.5 -> centres 0.5 apart, radii sum 2.0,
// penetration 1.5. One pair (0,1). Dynamic so StepWorld would resolve them, but CollectContacts reads the
// instantaneous overlap.
static FxWorld TwoOverlap() {
    FxWorld w;
    w.gravity = FxVec3{0, 0, 0};
    w.groundY = -1000 * kOne;        // far below — no floor interaction
    FxBody a; a.pos = FxVec3{0, 0, 0};        a.radius = kOne; a.invMass = kOne; a.flags = kFlagDynamic;
    FxBody b; b.pos = FxVec3{kOne / 2, 0, 0}; b.radius = kOne; b.invMass = kOne; b.flags = kFlagDynamic;
    w.bodies = {a, b};
    return w;
}

int main() {
    HF_TEST_MAIN_INIT();
    const std::vector<FxPair> pairs = { FxPair{0, 1} };

    // (1) CollectContacts on the overlapping config -> exactly one event, exact integer fields.
    {
        FxWorld w = TwoOverlap();
        std::vector<FxHitEvent> ev = CollectContacts(w, std::span<const FxPair>(pairs));
        check(ev.size() == 1, "fpx-events: one overlapping pair -> one event");
        if (ev.size() == 1) {
            check(ev[0].a == 0 && ev[0].b == 1, "fpx-events: event names the contacting bodies (a<b)");
            check(ev[0].depth == (2 * kOne - kOne / 2), "fpx-events: depth == residual penetration (1.5 in Q16.16)");
            check(ev[0].point.x == kOne / 4 && ev[0].point.y == 0 && ev[0].point.z == 0,
                  "fpx-events: contact point == inter-centre midpoint (0.25,0,0)");
        }
    }

    // (2) The 5-arg StepWorld overload is PHYSICS-IDENTICAL to the 4-arg (the golden-invariance guarantee).
    {
        FxWorld w4 = TwoOverlap();
        FxWorld w5 = TwoOverlap();
        StepWorld(w4, std::span<const FxPair>(pairs), kOne / 60, 8);                       // 4-arg (frozen)
        std::vector<FxHitEvent> ev;
        StepWorld(w5, std::span<const FxPair>(pairs), kOne / 60, 8, ev);                   // 5-arg (additive)
        bool same = w4.bodies.size() == w5.bodies.size();
        for (size_t i = 0; i < w4.bodies.size() && same; ++i) {
            const FxBody& A = w4.bodies[i]; const FxBody& B = w5.bodies[i];
            same = A.pos.x == B.pos.x && A.pos.y == B.pos.y && A.pos.z == B.pos.z &&
                   A.vel.x == B.vel.x && A.vel.y == B.vel.y && A.vel.z == B.vel.z;
        }
        check(same, "fpx-events: 5-arg StepWorld is byte-identical physics to the 4-arg (goldens invariant)");

        // (3) events are consistent with the existing CountResidualOverlaps reader.
        check(ev.size() == CountResidualOverlaps(w5, std::span<const FxPair>(pairs)),
              "fpx-events: events.size() == CountResidualOverlaps (consistent readers)");
    }

    // (4) A non-overlapping pair -> zero events.
    {
        FxWorld w; w.gravity = FxVec3{0, 0, 0}; w.groundY = -1000 * kOne;
        FxBody a; a.pos = FxVec3{0, 0, 0};         a.radius = kOne / 4; a.invMass = kOne; a.flags = kFlagDynamic;
        FxBody b; b.pos = FxVec3{10 * kOne, 0, 0}; b.radius = kOne / 4; b.invMass = kOne; b.flags = kFlagDynamic;
        w.bodies = {a, b};
        std::vector<FxHitEvent> ev = CollectContacts(w, std::span<const FxPair>(pairs));
        check(ev.empty(), "fpx-events: far-apart bodies -> no events");
    }

    // (5) determinism: CollectContacts is a pure function -> two calls are identical.
    {
        FxWorld w = TwoOverlap();
        auto e1 = CollectContacts(w, std::span<const FxPair>(pairs));
        auto e2 = CollectContacts(w, std::span<const FxPair>(pairs));
        bool eq = e1.size() == e2.size();
        for (size_t i = 0; i < e1.size() && eq; ++i)
            eq = e1[i].a == e2[i].a && e1[i].b == e2[i].b && e1[i].depth == e2[i].depth &&
                 e1[i].point.x == e2[i].point.x;
        check(eq, "fpx-events: CollectContacts is deterministic (re-call identical)");
    }

    if (g_fail == 0) std::printf("fpx_events_test: ALL PASS\n");
    else             std::printf("fpx_events_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
