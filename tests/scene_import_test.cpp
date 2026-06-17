// Slice V — pure scene-graph hierarchy-composition tests (no device, no glTF file).
//
// Exercises asset::ComposeWorld + asset::WalkHierarchy on hand-built SceneNode arrays and asserts
// that composed leaf world transforms equal the hand-checked parent*local products. This is the
// load-bearing correctness proof for the truck import: the same wheels mesh must land at two
// different world positions via two different parent transforms.

#include "asset/gltf_loader.h"
#include "math/math.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
using namespace hf::math;
using hf::asset::SceneNode;
using hf::asset::ComposeWorld;
using hf::asset::WalkHierarchy;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static bool approx(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

// Multiply world * (x,y,z,1) and return the transformed point (column-major).
static void xformPoint(const Mat4& w, float x, float y, float z, float out[3]) {
    out[0] = w.m[0]*x + w.m[4]*y + w.m[8]*z  + w.m[12];
    out[1] = w.m[1]*x + w.m[5]*y + w.m[9]*z  + w.m[13];
    out[2] = w.m[2]*x + w.m[6]*y + w.m[10]*z + w.m[14];
}

int main() {
    HF_TEST_MAIN_INIT();
    // ---- ComposeWorld == parent * local --------------------------------------------------------
    {
        Mat4 parent = Mat4::Translate({10.0f, 0.0f, 0.0f});
        Mat4 local  = Mat4::Translate({0.0f, 5.0f, 0.0f});
        Mat4 world  = ComposeWorld(parent, local);
        Mat4 expect = parent * local;
        for (int k = 0; k < 16; ++k)
            check(approx(world.m[k], expect.m[k]), "ComposeWorld == parent*local");
        // Translation is additive for pure translates.
        check(approx(world.m[12], 10.0f) && approx(world.m[13], 5.0f) && approx(world.m[14], 0.0f),
              "translate compose sums offsets");
    }

    // ---- Order matters: parent ROTATE then child TRANSLATE rotates the child's offset ----------
    // parent = RotateY(90deg). A child translated +X (1,0,0) should land at world ~ (0,0,-1)
    // (RH RotateY(90) maps +X -> -Z), proving world = parent*local (not local*parent).
    {
        Mat4 parent = Mat4::RotateY(1.5707963267948966f);   // 90 degrees
        Mat4 local  = Mat4::Translate({1.0f, 0.0f, 0.0f});
        Mat4 world  = ComposeWorld(parent, local);
        // The child's origin in world space:
        float p[3];
        xformPoint(world, 0, 0, 0, p);
        check(approx(p[0], 0.0f) && approx(p[1], 0.0f) && approx(p[2], -1.0f, 1e-3f),
              "parent*local: rotate-then-translate places child origin at (0,0,-1)");
    }

    // ---- WalkHierarchy: a milk-truck-shaped hierarchy with one mesh at two positions -----------
    // Mirror the CesiumMilkTruck topology: a body node (node 0) with two child "axle" nodes
    // (node 1 at +X, node 2 at -X), each carrying a "wheels" leaf (nodes 3,4) that is the SAME
    // conceptual mesh. The two leaves must compose to two distinct world positions.
    {
        std::vector<SceneNode> nodes(5);
        nodes[0].local = Mat4::Translate({0.0f, 1.0f, 0.0f});   // body, lifted up
        nodes[0].children = {1, 2};
        nodes[1].local = Mat4::Translate({2.0f, 0.0f, 0.0f});   // front axle (+X)
        nodes[1].children = {3};
        nodes[2].local = Mat4::Translate({-2.0f, 0.0f, 0.0f});  // back axle (-X)
        nodes[2].children = {4};
        nodes[3].local = Mat4::Identity();                       // front wheels leaf
        nodes[4].local = Mat4::Identity();                       // back wheels leaf

        std::vector<int> roots = {0};
        std::vector<Mat4> worlds(5, Mat4::Identity());
        std::vector<bool> seen(5, false);
        int visitCount = 0;
        WalkHierarchy(nodes, roots, [&](int idx, const Mat4& w) {
            worlds[idx] = w;
            seen[idx] = true;
            ++visitCount;
        });

        check(visitCount == 5, "WalkHierarchy visits every reachable node exactly once");
        for (int i = 0; i < 5; ++i) check(seen[i], "node visited");

        // Body world == its own local (root).
        check(approx(worlds[0].m[13], 1.0f), "body world Y == 1");

        // Front wheels leaf: world = body * frontAxle * identity -> position (2, 1, 0).
        float fw[3]; xformPoint(worlds[3], 0, 0, 0, fw);
        check(approx(fw[0], 2.0f) && approx(fw[1], 1.0f) && approx(fw[2], 0.0f),
              "front wheels compose to (2,1,0)");

        // Back wheels leaf: world = body * backAxle * identity -> position (-2, 1, 0).
        float bw[3]; xformPoint(worlds[4], 0, 0, 0, bw);
        check(approx(bw[0], -2.0f) && approx(bw[1], 1.0f) && approx(bw[2], 0.0f),
              "back wheels compose to (-2,1,0)");

        // The two wheel sets are at DIFFERENT positions (the whole point of hierarchy import).
        check(!approx(fw[0], bw[0]), "the two wheel sets land at distinct world X (front != back)");
    }

    // ---- WalkHierarchy is robust against a cycle (must not loop forever) ------------------------
    {
        std::vector<SceneNode> nodes(2);
        nodes[0].children = {1};
        nodes[1].children = {0};   // cycle back to 0
        int visits = 0;
        WalkHierarchy(nodes, {0}, [&](int, const Mat4&) { ++visits; });
        check(visits == 2, "cycle guarded: each node visited once");
    }

    if (g_fail == 0) std::printf("scene_import_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
