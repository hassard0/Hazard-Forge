// Slice IK6 — Deterministic IK Control-Rig: LIT 3D SKINNED RENDER CAPSTONE (the money-shot, the 6th and FINAL
// slice of FLAGSHIP #32). The pure-CPU contract test for the engine/anim/ik.h render-capstone bridge appended
// in IK6 (IkToPalette / WriteCorrectedSubChain) + the IK4 IkPoseToPalette/SolveRigToTargets provenance the
// --ik6-render-shot / --ik6-render showcases rest on. Pure C++ (header-only, no device, no backend symbols).
// Namespace hf::anim::ik.
//
// What this test PINS (the contracts the lit_skinned render money-shot builds on):
//   * IkToPalette provenance: two calls over the SAME corrected IkPose are BYTE-IDENTICAL (the rendered
//     palette is a PURE function of the bit-exact corrected pose — the FLOAT read-back is deterministic).
//   * IkToPalette == IkPoseToPalette (the thin alias is the IK4 read-back verbatim).
//   * the corrected palette's LEG joints derive from the bit-exact SolveRigToTargets (the palette entry ==
//     the read-back of the corrected pose at those joints — provenance from the integer IK leg).
//   * the un-IK'd joints' palette entries == the BASE pose's (the IK4 un-IK'd-joints invariant at the palette
//     level — only the chain bones move; a joint with no corrected ancestor is byte-identical to base).
//   * the leg end-effector reaches the foot-target within a documented band (the IK4 within-band foot-plant).
//   * WriteCorrectedSubChain writes back ONLY the chain bones [firstBone, count-1) (the virtual root +
//     end-effector are untouched).
//   * compose-the-moat: the IK-corrected pose seeds a deterministic ragdoll qTarget set (active.h
//     WriteClipTargets over a clip carrying the corrected pose — anim(IK) -> ragdoll, bit-reproducible).
//
// PURE CPU: SolveRigToTargets is the frozen IK4 integer solve (GPU==CPU already proven by ik_rig_test); IK6
// adds only the FLOAT palette read-back + the write-back bridge. The headline is DETERMINISM + provenance.
#include "anim/ik.h"
#include "anim/animation.h"
#include "anim/skeleton.h"
#include "sim/active.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;
namespace ik     = hf::anim::ik;
namespace anim   = hf::anim;
namespace active = hf::sim::active;
namespace joint  = hf::sim::joint;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

static ik::fx F(double v) { return (ik::fx)std::llround(v * (double)ik::kOne); }

// The fixed hand-built TEST SKELETON: a 5-joint humanoid leg lying straight down -Y (== the IK4/IK5 config):
//   0 pelvis (root, -1) | 1 hip (0) | 2 knee (1) | 3 ankle (2) | 4 foot (3, the end-effector).
// The IK chain is hip->knee->ankle->foot (joints 1,2,3,4, count 4). The base local rotations are identity.
static const int kJointCount = 5;
static const int kParents[kJointCount] = {-1, 0, 1, 2, 3};
static const double kSegDown[kJointCount] = {0.0, 0.2, 0.5, 0.5, 0.2};
static const int kChainCount = 4;
static const int kChainIdx[kChainCount] = {1, 2, 3, 4};

static ik::IkBasePose BuildBasePose() {
    ik::IkBasePose b{};
    b.count = kJointCount;
    for (int j = 0; j < kJointCount; ++j) {
        b.localT[j] = ik::FxVec3{0, (ik::fx)(-F(kSegDown[j])), 0};
        b.localR[j] = ik::FxQuat{0, 0, 0, ik::kOne};   // identity
    }
    return b;
}

// A matching anim::Skeleton (the same topology + rest TRS, inverseBind = the bind global inverse) so
// IkToPalette / PaletteFromLocalPose have a real skeleton to read back through.
static anim::Skeleton BuildSkeleton() {
    anim::Skeleton s;
    for (int j = 0; j < kJointCount; ++j) {
        anim::Joint jt;
        jt.parent = kParents[j];
        jt.t = math::Vec3{0.0f, -(float)kSegDown[j], 0.0f};
        jt.r = math::Quat{0, 0, 0, 1};
        jt.s = math::Vec3{1, 1, 1};
        s.joints.push_back(jt);
    }
    // inverseBind = inverse of the bind global (the rest pose) — a single forward pass.
    std::vector<math::Mat4> g((size_t)kJointCount);
    for (int j = 0; j < kJointCount; ++j) {
        const math::Mat4 local = math::FromTRS(s.joints[(size_t)j].t, s.joints[(size_t)j].r, s.joints[(size_t)j].s);
        const int p = s.joints[(size_t)j].parent;
        g[(size_t)j] = (p >= 0) ? (g[(size_t)p] * local) : local;
        s.joints[(size_t)j].inverseBind = g[(size_t)j].Inverse();
    }
    return s;
}

static ik::IkRig BuildRig() {
    ik::IkRig rig{};
    rig.count = kChainCount;
    for (int i = 0; i < kChainCount; ++i) rig.joint[i] = kChainIdx[i];
    // A planted-foot target forward + below the hip, within the chain reach (0.5+0.5+0.2 = 1.2 from the hip).
    rig.target = ik::FxVec3{F(0.4), -F(1.0), 0};
    rig.iters = 16;
    return rig;
}

int main() {
    HF_TEST_MAIN_INIT();

    const ik::IkBasePose base = BuildBasePose();
    const anim::Skeleton  skel = BuildSkeleton();
    const ik::IkRig       rig  = BuildRig();
    const ik::IkPose      corrected = ik::SolveRigToTargets(kParents, base, rig);

    // ===== IkToPalette provenance: two calls over the SAME corrected pose are BYTE-IDENTICAL =====
    {
        const std::vector<math::Mat4> p1 = ik::IkToPalette(skel, corrected);
        const std::vector<math::Mat4> p2 = ik::IkToPalette(skel, corrected);
        bool eq = (p1.size() == p2.size() && p1.size() == (size_t)kJointCount);
        for (size_t k = 0; k < p1.size() && eq; ++k)
            if (std::memcmp(p1[k].m, p2[k].m, sizeof(float) * 16) != 0) eq = false;
        check(eq, "IkToPalette is a pure function (two calls byte-identical)");
    }

    // ===== IkToPalette == IkPoseToPalette (the thin alias is the IK4 read-back VERBATIM) =====
    {
        const std::vector<math::Mat4> a = ik::IkToPalette(skel, corrected);
        const std::vector<math::Mat4> b = ik::IkPoseToPalette(skel, corrected);
        bool eq = (a.size() == b.size());
        for (size_t k = 0; k < a.size() && eq; ++k)
            if (std::memcmp(a[k].m, b[k].m, sizeof(float) * 16) != 0) eq = false;
        check(eq, "IkToPalette == IkPoseToPalette (the thin alias)");
    }

    // ===== the corrected palette's LEG joints derive from the bit-exact SolveRigToTargets =====
    // Rebuild the palette from a SECOND independent solve over the SAME inputs (the bit-exact integer solve is
    // deterministic) and confirm the chain joints' palette entries match — provenance from the integer IK leg.
    {
        const ik::IkPose corrected2 = ik::SolveRigToTargets(kParents, base, rig);
        const std::vector<math::Mat4> palA = ik::IkToPalette(skel, corrected);
        const std::vector<math::Mat4> palB = ik::IkToPalette(skel, corrected2);
        bool legEq = true, legPosed = false;
        const std::vector<math::Mat4> basePalette =
            ik::IkPoseToPalette(skel, [&]{ ik::IkPose bp{}; bp.count = base.count;
                for (int j = 0; j < base.count; ++j) { bp.localT[j]=base.localT[j]; bp.localR[j]=base.localR[j]; }
                return bp; }());
        for (int i = 0; i + 1 < kChainCount; ++i) {
            const int c = kChainIdx[i];
            if (std::memcmp(palA[(size_t)c].m, palB[(size_t)c].m, sizeof(float)*16) != 0) legEq = false;
            if (std::memcmp(palA[(size_t)c].m, basePalette[(size_t)c].m, sizeof(float)*16) != 0) legPosed = true;
        }
        check(legEq,    "corrected leg palette is bit-exact across two solves (derives from the integer IK)");
        check(legPosed, "corrected leg palette != base palette (the IK posed the leg)");
    }

    // ===== the un-IK'd joints' palette entries == the BASE pose's (the IK4 invariant at the palette level) ==
    // Joint 0 (pelvis) is NOT on the chain AND has no corrected ancestor -> its world transform (hence palette
    // entry) is byte-identical to the base pose's. (The end-effector joint 4 keeps its base LOCAL rotation but
    // its world transform changes via its corrected ancestors, so it is NOT compared here — the invariant is
    // about un-IK'd joints OUTSIDE the chain, the IK4 contract.)
    {
        ik::IkPose basePose{}; basePose.count = base.count;
        for (int j = 0; j < base.count; ++j) { basePose.localT[j]=base.localT[j]; basePose.localR[j]=base.localR[j]; }
        const std::vector<math::Mat4> basePalette = ik::IkToPalette(skel, basePose);
        const std::vector<math::Mat4> corrPalette = ik::IkToPalette(skel, corrected);
        check(std::memcmp(basePalette[0].m, corrPalette[0].m, sizeof(float)*16) == 0,
              "un-IK'd joint (pelvis) palette == base (the IK did not leak beyond the chain)");
        // and the un-IK'd joints' LOCAL rotations are byte-identical (the strict IK4 local invariant).
        bool localEq = true;
        for (int j = 0; j < base.count; ++j) {
            bool onChain = false;
            for (int i = 0; i + 1 < kChainCount; ++i) if (kChainIdx[i] == j) onChain = true;
            if (onChain) continue;
            const ik::FxQuat& a = corrected.localR[j]; const ik::FxQuat& b = base.localR[j];
            if (a.x!=b.x || a.y!=b.y || a.z!=b.z || a.w!=b.w) localEq = false;
        }
        check(localEq, "un-IK'd joints keep their base LOCAL rotation EXACTLY (the IK4 invariant)");
    }

    // ===== the leg end-effector reaches the foot-target within a documented band =====
    {
        const ik::fx res = ik::IkFootPlantResidual(kParents, corrected, rig);
        // band: the chain reach is 1.2; the FABRIK + rotation-convert residual is bounded well under 25%.
        const ik::fx band = F(0.30);
        check(res >= 0 && res <= band, "foot-plant residual within band (leg reaches the target)");
    }

    // ===== WriteCorrectedSubChain writes back ONLY the chain bones [firstBone, count-1) =====
    {
        // a 5-entry full local-rotation array seeded with a SENTINEL (so we see exactly which entries change).
        ik::FxQuat full[kJointCount];
        const ik::FxQuat sentinel{F(0.1), F(0.2), F(0.3), F(0.4)};
        for (int j = 0; j < kJointCount; ++j) full[j] = sentinel;
        // map sub-chain index -> full joint index = the chain joints; firstBone=1 (skip the virtual root).
        ik::WriteCorrectedSubChain(corrected, kChainIdx, kChainCount, 1, full);
        // expected: full[kChainIdx[i]] overwritten for i in [1, kChainCount-1); all others == sentinel.
        bool ok = true;
        for (int j = 0; j < kJointCount; ++j) {
            bool written = false;
            for (int i = 1; i + 1 < kChainCount; ++i) if (kChainIdx[i] == j) written = true;
            const bool isSentinel = (full[j].x == sentinel.x && full[j].y == sentinel.y &&
                                     full[j].z == sentinel.z && full[j].w == sentinel.w);
            if (written && isSentinel) ok = false;       // a written bone must NOT still be the sentinel
            if (!written && !isSentinel) ok = false;      // an untouched joint must STILL be the sentinel
            if (written) {
                // and it must be the corrected sub-chain rotation.
                int subI = -1; for (int i = 0; i < kChainCount; ++i) if (kChainIdx[i] == j) subI = i;
                const ik::FxQuat& w = corrected.localR[subI];
                if (full[j].x!=w.x || full[j].y!=w.y || full[j].z!=w.z || full[j].w!=w.w) ok = false;
            }
        }
        check(ok, "WriteCorrectedSubChain writes ONLY the chain bones [1, count-1) (root + end-effector kept)");
    }

    // ===== compose-the-moat: the IK-corrected pose seeds a deterministic ragdoll qTarget set =====
    // Build a clip carrying the corrected pose (snapped back to float local rotations), bind a ragdoll, and
    // confirm two WriteClipTargets passes produce byte-identical qTargets (anim(IK) -> ragdoll, bit-exact).
    {
        // the corrected pose as float JointPoses (the read-back the showcase writes back into the Fox pose).
        std::vector<anim::JointPose> pose((size_t)kJointCount);
        for (int j = 0; j < kJointCount; ++j) {
            pose[(size_t)j].t = math::Vec3{0.0f, -(float)kSegDown[j], 0.0f};
            const ik::FxQuat& q = corrected.localR[j];
            pose[(size_t)j].r = math::Normalize(math::Quat{
                (float)q.x / (float)ik::kOne, (float)q.y / (float)ik::kOne,
                (float)q.z / (float)ik::kOne, (float)q.w / (float)ik::kOne});
            pose[(size_t)j].s = math::Vec3{1, 1, 1};
        }
        anim::Animation clip; clip.name = "ik6_corrected"; clip.duration = 0.0f;
        for (int j = 0; j < kJointCount; ++j) {
            anim::Channel c; c.jointIndex = j; c.path = anim::Channel::Path::Rotation;
            c.interp = anim::Channel::Interp::Step; c.times = {0.0f};
            const math::Quat& r = pose[(size_t)j].r;
            c.values = {r.x, r.y, r.z, r.w};
            clip.channels.push_back(std::move(c));
        }
        joint::RagdollConfig rcfg;
        rcfg.worldScale = joint::kOne; rcfg.boneRadius = joint::kOne; rcfg.invMass = joint::kOne;
        rcfg.coneCos = -joint::kOne; rcfg.coneSin = 0;
        rcfg.gravity = joint::FxVec3{0, 0, 0}; rcfg.groundY = 0; rcfg.rootStatic = true;
        active::ActiveRagdoll a1 = active::ActiveFromSkeleton(skel, rcfg, joint::kOne);
        active::ActiveRagdoll a2 = active::ActiveFromSkeleton(skel, rcfg, joint::kOne);
        active::WriteClipTargets(a1, skel, clip, 0.0f);
        active::WriteClipTargets(a2, skel, clip, 0.0f);
        bool qDet = (a1.drives.size() == a2.drives.size() && !a1.drives.empty());
        for (size_t e = 0; e < a1.drives.size() && qDet; ++e) {
            const auto &x = a1.drives[e].qTarget, &y = a2.drives[e].qTarget;
            if (x.x!=y.x || x.y!=y.y || x.z!=y.z || x.w!=y.w) qDet = false;
        }
        check(qDet, "IK-corrected pose seeds a deterministic ragdoll qTarget set (anim(IK) -> ragdoll)");
    }

    if (g_fail == 0) std::printf("ik_render_test: ALL CHECKS PASSED\n");
    else             std::printf("ik_render_test: %d CHECK(S) FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
