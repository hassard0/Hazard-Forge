// Slice W unit test: the immediate-mode debug-draw collector. Pure C++ (no GPU). Validates that
// each primitive decomposes to the expected number of LINE_LIST vertices (2 per segment) and that a
// Line's emitted endpoints match its inputs. Counts are load-bearing: the showcase + the LINE_LIST
// draw rely on this decomposition.
#include "debug/debug_draw.h"
#include "debug/debug_emitters.h"
#include "math/math.h"
#include "physics/world.h"
#include "physics/body.h"

#include <cmath>
#include <cstdio>

using namespace hf;
using debug::DebugDraw;
using math::Vec3;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool feq(float a, float b) { return std::fabs(a - b) < 1e-5f; }

int main() {
    // --- 1. A single Line = 1 segment = 2 vertices; endpoints match the inputs, color carried. ---
    {
        DebugDraw dd;
        dd.Line({1, 2, 3}, {4, 5, 6}, {0.1f, 0.2f, 0.3f});
        check(dd.VertexCount() == 2, "Line => 2 vertices");
        const auto& v = dd.Vertices();
        check(feq(v[0].pos[0], 1) && feq(v[0].pos[1], 2) && feq(v[0].pos[2], 3),
              "Line start endpoint matches input");
        check(feq(v[1].pos[0], 4) && feq(v[1].pos[1], 5) && feq(v[1].pos[2], 6),
              "Line end endpoint matches input");
        check(feq(v[0].color[0], 0.1f) && feq(v[0].color[1], 0.2f) && feq(v[0].color[2], 0.3f),
              "Line carries per-vertex color");
        dd.Clear();
        check(dd.VertexCount() == 0 && dd.Empty(), "Clear empties the vertex list");
    }

    // --- 2. Box = 12 edges = 24 vertices, and the bounds appear among the emitted positions. ---
    {
        DebugDraw dd;
        dd.Box({-1, -2, -3}, {1, 2, 3}, {1, 1, 1});
        check(dd.VertexCount() == 24, "Box => 12 edges => 24 vertices");
        // Every emitted coordinate must lie on the box extents.
        bool onExtents = true, sawMinCorner = false, sawMaxCorner = false;
        for (const auto& vert : dd.Vertices()) {
            bool xok = feq(vert.pos[0], -1) || feq(vert.pos[0], 1);
            bool yok = feq(vert.pos[1], -2) || feq(vert.pos[1], 2);
            bool zok = feq(vert.pos[2], -3) || feq(vert.pos[2], 3);
            onExtents = onExtents && xok && yok && zok;
            if (feq(vert.pos[0], -1) && feq(vert.pos[1], -2) && feq(vert.pos[2], -3)) sawMinCorner = true;
            if (feq(vert.pos[0], 1) && feq(vert.pos[1], 2) && feq(vert.pos[2], 3)) sawMaxCorner = true;
        }
        check(onExtents, "Box vertices all lie on the box extents");
        check(sawMinCorner && sawMaxCorner, "Box includes both min and max corners");
    }

    // --- 3. Obb = 12 edges = 24 vertices (identity transform reduces to a Box about the origin). ---
    {
        DebugDraw dd;
        dd.Obb(math::Mat4::Identity(), {1, 1, 1}, {1, 0, 0});
        check(dd.VertexCount() == 24, "Obb => 12 edges => 24 vertices");
    }

    // --- 4. WireSphere(segments) = 3 circles * segments chords * 2 verts. ---
    {
        DebugDraw dd;
        dd.WireSphere({0, 0, 0}, 1.0f, {1, 1, 1}, 16);
        check(dd.VertexCount() == (size_t)(3 * 16 * 2), "WireSphere(16) => 3*16*2 = 96 vertices");
        DebugDraw dd2;
        dd2.WireSphere({0, 0, 0}, 1.0f, {1, 1, 1}, 8);
        check(dd2.VertexCount() == (size_t)(3 * 8 * 2), "WireSphere(8) => 3*8*2 = 48 vertices");
    }

    // --- 5. Grid: lines per axis = 2*floor(half/step)+1; total verts = 2*(2n+1)*2. ---
    {
        DebugDraw dd;
        dd.Grid(10.0f, 1.0f, {0.5f, 0.5f, 0.5f});
        // n = floor(10/1) = 10 -> 21 lines/axis -> 42 lines -> 84 vertices.
        check(dd.VertexCount() == 84, "Grid(10,1) => 42 lines => 84 vertices");
        DebugDraw dd2;
        dd2.Grid(4.0f, 2.0f, {0.5f, 0.5f, 0.5f});
        // n = 2 -> 5 lines/axis -> 10 lines -> 20 vertices.
        check(dd2.VertexCount() == 20, "Grid(4,2) => 10 lines => 20 vertices");
    }

    // --- 6. Axes = 3 segments = 6 vertices; the three basis axes radiate from the origin. ---
    {
        DebugDraw dd;
        dd.Axes(math::Mat4::Identity(), 2.0f);
        check(dd.VertexCount() == 6, "Axes => 3 segments => 6 vertices");
        const auto& v = dd.Vertices();
        check(feq(v[1].pos[0], 2) && feq(v[1].pos[1], 0) && feq(v[1].pos[2], 0), "Axes X tip at +X*len");
        check(feq(v[3].pos[0], 0) && feq(v[3].pos[1], 2) && feq(v[3].pos[2], 0), "Axes Y tip at +Y*len");
        check(feq(v[5].pos[0], 0) && feq(v[5].pos[1], 0) && feq(v[5].pos[2], 2), "Axes Z tip at +Z*len");
    }

    // --- 7. Ray = 1 segment = 2 vertices; the far end is origin + normalize(dir)*len. ---
    {
        DebugDraw dd;
        dd.Ray({0, 0, 0}, {0, 3, 0}, 5.0f, {1, 1, 0});
        check(dd.VertexCount() == 2, "Ray => 2 vertices");
        const auto& v = dd.Vertices();
        check(feq(v[1].pos[1], 5.0f), "Ray far end at origin + normalize(dir)*len");
    }

    // --- 8. Emitter: AabbWorld about a translated unit box draws a Box (24 verts) hugging it. ---
    {
        DebugDraw dd;
        math::Mat4 model = math::Mat4::Translate({10, 0, 0});
        debug::AabbWorld(dd, {-0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}, model, {0, 1, 0});
        check(dd.VertexCount() == 24, "AabbWorld => Box => 24 vertices");
        bool centeredAt10 = false;
        for (const auto& vert : dd.Vertices())
            if (feq(vert.pos[0], 9.5f) || feq(vert.pos[0], 10.5f)) centeredAt10 = true;
        check(centeredAt10, "AabbWorld AABB is translated to the model position");
    }

    // --- 9. Emitter: PhysicsContacts on a resting sphere produces a ground-contact cross+normal. ---
    {
        DebugDraw dd;
        physics::World w;
        // Sphere resting exactly on the plane (pen ~ 0): nudge it slightly into the plane so a
        // ground contact is detected deterministically.
        physics::RigidBody s = physics::MakeDynamicSphere({0, 0.49f, 0}, 0.5f);
        w.bodies.push_back(s);
        debug::PhysicsContacts(dd, w, {1, 1, 0}, {0, 1, 1});
        // One ground contact => 3 cross segments + 1 normal segment => 8 vertices.
        check(dd.VertexCount() == 8, "PhysicsContacts: one ground contact => 4 segments => 8 verts");
    }

    if (g_fail == 0) std::printf("debug_draw_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
