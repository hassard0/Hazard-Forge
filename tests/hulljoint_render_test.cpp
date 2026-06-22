// Slice HF6 — Hull Friction + Joints: LIT 3D CAPSTONE (the money-shot completing FLAGSHIP #30,
// hf::sim::hulljoint). HF6 appends a render-only FLOAT delegate JointedHullToRenderInstances(world) over the
// bit-exact friction+joint settled world (the WH6/CX6/FC6/PS6 capstone mold). This pure-CPU test pins the
// render-bridge contracts the showcase builds on. NO GPU, NO shader, NO RHI. #includes hulljoint.h (which
// pulls in hullfric/joint/warmhull/gjk/fpx READ-ONLY/BYTE-FROZEN).
//
// What this test PINS:
//   * PROVENANCE: two JointedHullToRenderInstances calls on the bit-exact world are BYTE-EQUAL
//     (gjk::HullRenderMeshEqual) — the render is a PURE FUNCTION of the world.
//   * INSTANCE/BODY COUNT: every body is meshed — the soup is non-empty AND a per-body triangle count
//     consistent with the body count (one canonical hull mesh per body).
//   * THE RENDER IS A PURE READ: the (bodies, cache) PAIR is JointedHullStatesEqual before vs after a
//     JointedHullToRenderInstances call (the render does NOT mutate the bit-exact integer sim).
//   * THE SETTLED SCENE has the expected body count (the HF4/HF5 chain+door+ramp scene).
//
// Pure C++ (hf_core), ASan-eligible like the other sim tests.
#include "sim/hulljoint.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace hulljoint = hf::sim::hulljoint;
namespace hullfric  = hf::sim::hullfric;
namespace joint     = hf::sim::joint;
namespace convex    = hf::sim::convex;
namespace gjk       = hf::sim::gjk;
namespace fpx       = hf::sim::fpx;
using gjk::fx;
using gjk::kOne;
using gjk::FxVec3;
using gjk::FxHull;
using fpx::FxBody;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}
static fx F(double v) { return (fx)(v * 65536.0); }

// ----- The PINNED HF4/HF5 scene (== the hf4-joint / --hf5-net / --hf6-hull scene): a CHAIN of 3 dynamic
// hulls hung by ball joints from a STATIC anchor + a HINGED hull DOOR (a ball joint at the pivot + an
// angular-limit hinge to a static frame) + a hull resting on a static ground hull with friction. All boxes
// are 2x2x2 (half-extent 1).
static constexpr double kLinkGap = 2.2;

static hulljoint::JointedHullWorld BuildScene() {
    hulljoint::JointedHullWorld w;
    const FxHull boxH = gjk::MakeBox(kOne, kOne, kOne);
    auto mkStatic = [&](double x, double y, double z) {
        FxBody b; b.pos = {F(x), F(y), F(z)}; b.orient = {0, 0, 0, kOne};
        b.vel = {0,0,0}; b.angVel = {0,0,0}; b.invMass = 0; b.flags = 0u; b.radius = 0; return b;
    };
    auto mkDyn = [&](double x, double y, double z) {
        FxBody b; b.pos = {F(x), F(y), F(z)}; b.orient = {0, 0, 0, kOne};
        b.vel = {0,0,0}; b.angVel = {0,0,0}; b.invMass = kOne; b.flags = fpx::kFlagDynamic; b.radius = 0; return b;
    };
    const double aY = 6.0;
    w.hulls.bodies = {
        mkStatic(-4.0, aY, 0.0),                 // 0 chain anchor (static)
        mkDyn(-4.0, aY - kLinkGap, 0.0),         // 1 link0
        mkDyn(-4.0, aY - 2 * kLinkGap, 0.0),     // 2 link1
        mkDyn(-4.0, aY - 3 * kLinkGap, 0.0),     // 3 link2
        mkStatic(2.0, 0.0, 0.0),                 // 4 ground (static)
        mkDyn(2.0, 2.0 - 0.0625, 0.0),           // 5 friction-resting hull
        mkStatic(4.0, aY, 0.0),                  // 6 hinge frame (static)
        mkDyn(4.0, aY - kLinkGap, 0.0),          // 7 hinged door
    };
    w.hulls.hulls = {boxH, boxH, boxH, boxH, boxH, boxH, boxH, boxH};
    const FxVec3 bottomA{0, F(-kLinkGap / 2.0), 0};
    const FxVec3 topA{0, F(kLinkGap / 2.0), 0};
    auto mkBall = [&](uint32_t a, uint32_t b) {
        joint::FxJoint j; j.bodyA = a; j.bodyB = b; j.anchorA = bottomA; j.anchorB = topA;
        j.kind = joint::kJointBall; j.limit = 0; return j;
    };
    w.joints = { mkBall(0, 1), mkBall(1, 2), mkBall(2, 3), mkBall(6, 7) };
    joint::FxAngularLimit hinge;
    hinge.bodyA = 6; hinge.bodyB = 7; hinge.axis = {0, 0, kOne};
    hinge.cosHalfLimit = kOne; hinge.sinHalfLimit = 0; hinge.kind = joint::kAngularHinge;
    w.limits = { hinge };
    return w;
}

static hulljoint::JointedHullStepConfig StepCfg() {
    hulljoint::JointedHullStepConfig cfg;
    cfg.fric.mu = (fx)((int64_t)6 * kOne / 10);   // 0.6 Coulomb cone
    cfg.fric.solveIters = 12;
    cfg.fric.posIters   = 4;
    cfg.jointIters      = 8;
    return cfg;
}

static constexpr uint32_t kTicks = 240u;

int main() {
    HF_TEST_MAIN_INIT();

    // === Build + settle the bit-exact friction+joint scene (the input the render bridge consumes). ===
    hulljoint::JointedHullWorld world = BuildScene();
    const uint32_t kBodies = (uint32_t)world.hulls.bodies.size();
    check(kBodies == 8u, "settled scene has the expected body count (8: anchor+3 links+ground+hull+frame+door)");
    hulljoint::StepJointedHullWorldN(world, StepCfg(), kTicks);

    // The snapshot of the PAIR BEFORE the render (the pure-read baseline).
    const hulljoint::JointedHullSnapshot before = hulljoint::SnapshotJointedHull(world, (int32_t)kTicks);

    // === PROOF (1) PROVENANCE: two JointedHullToRenderInstances calls are BYTE-EQUAL. ===
    const gjk::HullRenderMesh soup = hulljoint::JointedHullToRenderInstances(world);
    {
        const gjk::HullRenderMesh rebuild = hulljoint::JointedHullToRenderInstances(world);
        check(gjk::HullRenderMeshEqual(soup, rebuild),
              "provenance: two JointedHullToRenderInstances calls BYTE-EQUAL (render is a pure function)");
    }

    // === PROOF (2) INSTANCE/BODY COUNT: every body is meshed — the soup is non-empty + tri count is a
    // positive multiple of the body count (one canonical-hull mesh per body, FIXED-order). Each canonical
    // box hull meshes to the same triangle count, so soup.triangles == kBodies * (tris-per-box). ===
    check(soup.triangles > 0u, "render mesh is non-empty (the settled bodies meshed)");
    check(soup.verts.size() == (size_t)soup.triangles * 3u, "render mesh vert count == 3 * triangle count");
    check((soup.triangles % kBodies) == 0u,
          "triangle count is a multiple of the body count (one mesh per body)");

    // === PROOF (3) THE RENDER IS A PURE READ: the (bodies, cache) PAIR is byte-identical before vs after
    // the render call (the render did NOT mutate the bit-exact integer sim). ===
    const hulljoint::JointedHullSnapshot after = hulljoint::SnapshotJointedHull(world, (int32_t)kTicks);
    check(hulljoint::JointedHullStatesEqual(before, after),
          "render is a pure read (world (bodies,cache) PAIR JointedHullStatesEqual before/after)");

    // A second render call must STILL leave the world unmutated (idempotent pure read).
    (void)hulljoint::JointedHullToRenderInstances(world);
    const hulljoint::JointedHullSnapshot after2 = hulljoint::SnapshotJointedHull(world, (int32_t)kTicks);
    check(hulljoint::JointedHullStatesEqual(before, after2),
          "render is a pure read after a second call (world PAIR still unmutated)");

    if (g_fail == 0) std::printf("hulljoint_render_test: OK (%u bodies, %u tris)\n", kBodies, soup.triangles);
    return g_fail == 0 ? 0 : 1;
}
