// Unit tests for the pool-based ECS (engine/ecs/ecs.h): entity lifecycle (generation invalidates
// stale handles), add/get/remove components, and single- and multi-type view iteration.
#include "ecs/ecs.h"

#include <cstdio>
#include <set>

using namespace hf::ecs;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

struct Position { float x, y, z; };
struct Velocity { float dx, dy, dz; };
struct Tag { int id; };

int main() {
    // --- Entity create / valid / destroy ------------------------------------
    {
        Registry r;
        Entity a = r.create();
        Entity b = r.create();
        check(r.valid(a), "created entity a is valid");
        check(r.valid(b), "created entity b is valid");
        check(a != b, "distinct entities differ");
        check(r.aliveCount() == 2, "two alive");

        r.destroy(a);
        check(!r.valid(a), "destroyed entity a is invalid");
        check(r.valid(b), "b still valid after a destroyed");
        check(r.aliveCount() == 1, "one alive after destroy");
    }

    // --- Generation invalidates stale handles -------------------------------
    {
        Registry r;
        Entity a = r.create();
        r.destroy(a);
        Entity recycled = r.create();  // should reuse a's index, bumped generation
        check(recycled.index == a.index, "index recycled");
        check(recycled.generation != a.generation, "generation bumped on recycle");
        check(!r.valid(a), "stale handle (old generation) is invalid");
        check(r.valid(recycled), "recycled handle is valid");
    }

    // --- add / get / has / remove -------------------------------------------
    {
        Registry r;
        Entity e = r.create();
        check(!r.has<Position>(e), "no component before add");
        r.add<Position>(e, {1.0f, 2.0f, 3.0f});
        check(r.has<Position>(e), "has after add");
        Position& p = r.get<Position>(e);
        check(p.x == 1.0f && p.y == 2.0f && p.z == 3.0f, "get returns stored values");

        // Mutate through the reference.
        p.x = 9.0f;
        check(r.get<Position>(e).x == 9.0f, "component is mutable via reference");

        r.remove<Position>(e);
        check(!r.has<Position>(e), "no component after remove");
    }

    // --- destroy drops the entity's components ------------------------------
    {
        Registry r;
        Entity e = r.create();
        r.add<Position>(e, {0, 0, 0});
        r.destroy(e);
        // Recycle the slot; the new entity must NOT inherit the old component.
        Entity e2 = r.create();
        check(e2.index == e.index, "slot recycled");
        check(!r.has<Position>(e2), "recycled entity has no stale component");
    }

    // --- single-type view yields exactly the matching entities --------------
    {
        Registry r;
        Entity a = r.create();
        Entity b = r.create();
        Entity c = r.create();
        r.add<Position>(a, {0, 0, 0});
        r.add<Position>(c, {0, 0, 0});
        // b has no Position.

        std::set<uint32_t> seen;
        int count = 0;
        for (auto [e, pos] : r.view<Position>()) {
            (void)pos;
            seen.insert(e.index);
            ++count;
        }
        check(count == 2, "single-type view visits exactly 2 entities");
        check(seen.count(a.index) && seen.count(c.index), "view visited a and c");
        check(!seen.count(b.index), "view skipped b (no Position)");
    }

    // --- two-type view yields entities holding BOTH -------------------------
    {
        Registry r;
        Entity a = r.create();  // Position + Velocity
        Entity b = r.create();  // Position only
        Entity c = r.create();  // Position + Velocity
        Entity d = r.create();  // Velocity only
        r.add<Position>(a, {1, 0, 0}); r.add<Velocity>(a, {1, 1, 1});
        r.add<Position>(b, {2, 0, 0});
        r.add<Position>(c, {3, 0, 0}); r.add<Velocity>(c, {2, 2, 2});
        r.add<Velocity>(d, {3, 3, 3});

        std::set<uint32_t> seen;
        int count = 0;
        for (auto [e, pos, vel] : r.view<Position, Velocity>()) {
            // Integrate so we also confirm both refs are live & writable.
            pos.x += vel.dx;
            seen.insert(e.index);
            ++count;
        }
        check(count == 2, "two-type view visits exactly 2 entities (a and c)");
        check(seen.count(a.index) && seen.count(c.index), "two-type view visited a and c");
        check(!seen.count(b.index) && !seen.count(d.index), "two-type view skipped b and d");
        check(r.get<Position>(a).x == 2.0f, "view mutation applied to a");
        check(r.get<Position>(c).x == 5.0f, "view mutation applied to c");
    }

    // --- view picks the smallest pool as driver (rare component) ------------
    {
        Registry r;
        // Many entities with Position, exactly one also with Tag.
        Entity tagged{};
        for (int i = 0; i < 100; ++i) {
            Entity e = r.create();
            r.add<Position>(e, {(float)i, 0, 0});
            if (i == 42) { r.add<Tag>(e, {7}); tagged = e; }
        }
        int count = 0;
        Entity found{};
        for (auto [e, pos, tag] : r.view<Position, Tag>()) {
            (void)pos;
            check(tag.id == 7, "tag value correct");
            found = e;
            ++count;
        }
        check(count == 1, "Position+Tag view yields exactly the one tagged entity");
        check(found == tagged, "view yielded the correct tagged entity");
    }

    if (g_fail == 0) { std::printf("ecs_test OK\n"); return 0; }
    std::printf("ecs_test: %d failures\n", g_fail);
    return 1;
}
