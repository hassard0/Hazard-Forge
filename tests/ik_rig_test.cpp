// Slice IK4 — Deterministic IK Control-Rig: IK ON THE SKELETON (the FK-pose -> IK-corrected palette bridge,
// the 4th slice of FLAGSHIP #32). The pure-CPU contract test for engine/anim/ik.h::SolveRigToTargets /
// FkWorldPositions / IkPoseToPalette (the bridge the GPU shaders/ik_rig.comp copies VERBATIM + proves
// bit-identical). Pure C++ (header-only, no device, no backend symbols). Namespace hf::anim::ik.
//
// What this test PINS (the contracts the GPU solve + later IK slices build on):
//   * determinism: SolveRigToTargets is a pure function -> two runs byte-identical.
//   * foot-plant: FK the CORRECTED pose -> the chain end-effector at the world target within a documented
//     band over a reachable sweep (the FABRIK iterative + LookAtRotation LSB residual — NOT zero).
//   * un-IK'd-joints invariant: every joint NOT in the rig chain has a local rotation + translation
//     byte-identical to the base pose (EXACT — the falsifiable invariant).
//   * IK-did-something: the chain joints' local rotations DIFFER from the base when the target != the rest
//     end-effector (the IK actually moved the limb).
//   * IkPoseToPalette determinism: two calls byte-identical (the render-only float read-back is a pure
//     deterministic function of the bit-exact corrected pose).
//
// The HONEST within-band caveat: the foot-plant residual is bounded + deterministic (the FABRIK iterative
// Gauss-Seidel residual + the FxNormalize / LookAtRotation LSB — NOT zero). The un-IK'd-joints invariant is
// EXACT. The headline is DETERMINISM + cross-platform bit-identity, NOT analytic reach.
#include "anim/ik.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace ik  = hf::anim::ik;
namespace fpx = hf::sim::fpx;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static ik::fx F(double v) { return (ik::fx)std::llround(v * (double)ik::kOne); }

// The fixed hand-built TEST SKELETON: a 5-joint humanoid leg lying straight down -Y:
//   0 pelvis (root, parent -1)  at world origin
//   1 hip    (parent 0)         0.2 down
//   2 knee   (parent 1)         0.5 down
//   3 ankle  (parent 2)         0.5 down
//   4 foot   (parent 3)         0.2 down (the end-effector)
// The IK chain is hip->knee->ankle->foot (joints 1,2,3,4, count 4). Base local rotations are identity, so
// the rest FK lays the leg straight down. The target sweep moves the foot to plant it at various positions.
static const int kJointCount = 5;
static const int kParents[kJointCount] = {-1, 0, 1, 2, 3};
static const double kSegDown[kJointCount] = {0.0, 0.2, 0.5, 0.5, 0.2};  // local -Y translation per joint

static ik::IkBasePose BuildBasePose() {
    ik::IkBasePose b{};
    b.count = kJointCount;
    for (int j = 0; j < kJointCount; ++j) {
        b.localT[j] = ik::FxVec3{0, (ik::fx)(-F(kSegDown[j])), 0};
        b.localR[j] = ik::FxQuat{0, 0, 0, ik::kOne};   // identity
    }
    return b;
}

static ik::IkRig BuildRig(const ik::FxVec3& target) {
    ik::IkRig rig{};
    rig.count = 4;
    rig.joint[0] = 1; rig.joint[1] = 2; rig.joint[2] = 3; rig.joint[3] = 4;
    rig.target = target;
    rig.iters = 12;
    return rig;
}

// A small anim::Skeleton for the IkPoseToPalette test (identity inverseBind, the rest TRS).
static anim::Skeleton BuildSkeleton() {
    anim::Skeleton sk;
    sk.joints.resize(kJointCount);
    for (int j = 0; j < kJointCount; ++j) {
        sk.joints[(size_t)j].parent = kParents[j];
        sk.joints[(size_t)j].t = math::Vec3{0.0f, -(float)kSegDown[j], 0.0f};
        sk.joints[(size_t)j].r = math::Quat{0, 0, 0, 1};
        sk.joints[(size_t)j].s = math::Vec3{1, 1, 1};
        sk.joints[(size_t)j].inverseBind = math::Mat4::Identity();
    }
    return sk;
}

static bool PoseEqual(const ik::IkPose& a, const ik::IkPose& b) {
    if (a.count != b.count) return false;
    for (int j = 0; j < a.count; ++j) {
        if (a.localR[j].x != b.localR[j].x || a.localR[j].y != b.localR[j].y ||
            a.localR[j].z != b.localR[j].z || a.localR[j].w != b.localR[j].w) return false;
        if (a.localT[j].x != b.localT[j].x || a.localT[j].y != b.localT[j].y ||
            a.localT[j].z != b.localT[j].z) return false;
    }
    return true;
}

static bool InChain(const ik::IkRig& rig, int j) {
    for (int i = 0; i < rig.count; ++i) if (rig.joint[i] == j) return true;
    return false;
}

int main() {
    HF_TEST_MAIN_INIT();
    const ik::IkBasePose base = BuildBasePose();

    // The rest end-effector world position (foot straight down): -(0.2+0.5+0.5+0.2) = -1.4 on Y.
    const ik::IkFkWorld restFk = ik::FkWorldPositions(kParents, base);

    // ===== determinism: SolveRigToTargets is a pure function (two runs byte-identical) =====
    {
        const ik::IkRig rig = BuildRig(ik::FxVec3{F(0.6), F(-1.0), 0});
        const ik::IkPose p1 = ik::SolveRigToTargets(kParents, base, rig);
        const ik::IkPose p2 = ik::SolveRigToTargets(kParents, base, rig);
        check(PoseEqual(p1, p2), "SolveRigToTargets is deterministic (two runs byte-identical)");
    }

    // ===== foot-plant: FK the corrected pose -> end-effector at the world target within band =====
    {
        // The documented foot-plant band: the FABRIK iterative residual + the LookAtRotation/FxNormalize LSB
        // propagated through the rotation-convert + the re-FK. 0.03 world units over the reachable sweep.
        const ik::fx kFootPlantBand = F(0.03);
        ik::fx maxResidual = 0;
        const int N = 24;
        // The chain reach is 0.5+0.5+0.2 = 1.2 from the hip (joint 1, at world Y=-0.2). Sweep reachable
        // targets in front of + below the hip (a planted-foot arc), all within reach of the hip.
        for (int t = 0; t < N; ++t) {
            const double frac = (double)t / (double)(N - 1);
            const double ang   = -0.6 + 1.2 * frac;     // [-0.6, 0.6] rad
            const double reach = 0.5 + 0.6 * frac;       // [0.5, 1.1] (< 1.2 hip reach -> reachable)
            // target relative to the hip world pos (0,-0.2,0): forward (+X) and down.
            const double hx = reach * std::sin(ang);
            const double hy = -0.2 - reach * std::cos(ang);
            const ik::IkRig rig = BuildRig(ik::FxVec3{F(hx), F(hy), 0});
            const ik::IkPose p = ik::SolveRigToTargets(kParents, base, rig);
            const ik::fx res = ik::IkFootPlantResidual(kParents, p, rig);
            if (res > maxResidual) maxResidual = res;
        }
        std::printf("ik_rig_test: reachable-sweep maxFootPlantResidual=%d LSB (band=%d)\n",
                    maxResidual, kFootPlantBand);
        check(maxResidual <= kFootPlantBand, "foot-plant: end-effector within the documented residual band");
    }

    // ===== un-IK'd-joints invariant: joints NOT in the chain byte-identical to the base =====
    {
        const ik::IkRig rig = BuildRig(ik::FxVec3{F(0.7), F(-0.9), 0});
        const ik::IkPose p = ik::SolveRigToTargets(kParents, base, rig);
        bool allBaseEqual = true;
        for (int j = 0; j < base.count; ++j) {
            if (InChain(rig, j)) continue;        // chain joints are corrected (except the end-effector)
            const bool rEq = (p.localR[j].x == base.localR[j].x && p.localR[j].y == base.localR[j].y &&
                              p.localR[j].z == base.localR[j].z && p.localR[j].w == base.localR[j].w);
            const bool tEq = (p.localT[j].x == base.localT[j].x && p.localT[j].y == base.localT[j].y &&
                              p.localT[j].z == base.localT[j].z);
            if (!rEq || !tEq) allBaseEqual = false;
        }
        check(allBaseEqual, "un-IK'd joints == base pose EXACTLY (untouched joints byte-identical)");
        // The end-effector joint (last in the chain) has no child bone -> its local rotation stays base.
        const int ee = rig.joint[rig.count - 1];
        const bool eeEq = (p.localR[ee].x == base.localR[ee].x && p.localR[ee].y == base.localR[ee].y &&
                           p.localR[ee].z == base.localR[ee].z && p.localR[ee].w == base.localR[ee].w);
        check(eeEq, "chain end-effector joint keeps its base local rotation (no child bone to aim)");
        // ALL translations are untouched by IK (IK corrects rotations only).
        bool allTEq = true;
        for (int j = 0; j < base.count; ++j)
            if (p.localT[j].x != base.localT[j].x || p.localT[j].y != base.localT[j].y ||
                p.localT[j].z != base.localT[j].z) allTEq = false;
        check(allTEq, "IK leaves all local translations byte-identical to the base");
    }

    // ===== IK-did-something: chain joints differ from base when target != rest end-effector =====
    {
        // A target well away from the rest foot (-1.4 Y straight down) -> the chain MUST bend.
        const ik::IkRig rig = BuildRig(ik::FxVec3{F(0.8), F(-0.8), 0});
        const ik::IkPose p = ik::SolveRigToTargets(kParents, base, rig);
        int movedCount = 0;
        for (int i = 0; i + 1 < rig.count; ++i) {   // the chain joints that drive a bone (not the end-effector)
            const int j = rig.joint[i];
            const bool same = (p.localR[j].x == base.localR[j].x && p.localR[j].y == base.localR[j].y &&
                               p.localR[j].z == base.localR[j].z && p.localR[j].w == base.localR[j].w);
            if (!same) ++movedCount;
        }
        std::printf("ik_rig_test: IK moved %d/%d chain-bone joints (target off the rest pose)\n",
                    movedCount, rig.count - 1);
        check(movedCount > 0, "IK actually moved the chain (corrected rotations differ from base)");
        (void)restFk;
    }

    // ===== IkPoseToPalette determinism: two calls byte-identical =====
    {
        const anim::Skeleton sk = BuildSkeleton();
        const ik::IkRig rig = BuildRig(ik::FxVec3{F(0.5), F(-1.0), 0});
        const ik::IkPose p = ik::SolveRigToTargets(kParents, base, rig);
        const std::vector<math::Mat4> pal1 = ik::IkPoseToPalette(sk, p);
        const std::vector<math::Mat4> pal2 = ik::IkPoseToPalette(sk, p);
        bool eq = (pal1.size() == pal2.size());
        if (eq) for (size_t j = 0; j < pal1.size(); ++j)
            for (int e = 0; e < 16; ++e) if (pal1[j].m[e] != pal2[j].m[e]) { eq = false; break; }
        check(eq, "IkPoseToPalette is deterministic (two calls byte-identical)");
        check(pal1.size() == (size_t)kJointCount, "IkPoseToPalette returns one Mat4 per joint");
    }

    if (g_fail == 0) std::printf("ik_rig_test: ALL PASS\n");
    else std::printf("ik_rig_test: %d FAIL\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
