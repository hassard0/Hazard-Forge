#pragma once
// Slice AN2 — DETERMINISTIC ANIMATION RETARGETING (play one skeleton's clip on a differently-
// proportioned skeleton; the parity++ audit's AN2 gap + the animation-pillar continuation after AN1
// blend spaces). The anim stack ships skeletons + clips + sampling (animation.h), blend spaces (AN1)
// and motion matching (MM1) — but NOTHING that maps a clip authored for ONE skeleton onto ANOTHER
// with different bone lengths/rest orientations. UE5 authors this as the IK Rig / IK Retargeter; AN2
// builds the ROTATION-retargeting core deterministically. Namespace hf::anim::retarget, header-only,
// pure CPU, NO device / backend symbols, NO RNG, NO clock, NO new shader. engine/anim/animation.h +
// skeleton.h are #included READ-ONLY (byte-untouched); motion_match.h is #included READ-ONLY for the
// anim family's Q16.16 currency (mm::fx/kOne/kFrac/fxmul/QuantizeFx/FxV3 + mm::detail::Fnv1a64Word) —
// the same integer bits AN1/MM1 pin.
//
// THE FLOAT / INTEGER SPLIT (the MM1/AN1 discipline, documented honestly): the ONE float->integer
// boundary is QuantizeFx — the source clip's FLOAT pose (animation.h SampleLocalPose) and both
// skeletons' FLOAT bind rotations/translations are quantized to Q16.16 ONCE. EVERYTHING downstream —
// the quaternion algebra, the bind-delta composition, the root-scale, the forward-kinematics used for
// the model-space digests + the stick-figure viz — is INTEGER (int32 state, int64 intermediates), so
// two peers / two compilers agree bit-for-bit BY CONSTRUCTION (the MM1 guarantee; no FMA / rounding
// sensitivity past the boundary). The fixtures use exact binary-fraction keyframe times/values sampled
// AT keyframe times (frac 0 -> animation.h Slerp returns the key EXACTLY, its Normalize sees a unit
// input -> sqrt(1)==1), so the QuantizeFx inputs are themselves cross-compiler exact.
//
// INTEGER QUATERNIONS (Q16.16): FxQuat + FxQuatMul (Hamilton product, int64 intermediates, arithmetic
// >> kFrac) + FxQuatConj. INVERSE == CONJUGATE (documented v1 assumption: all ROTATION quaternions are
// UNIT — bind poses and sampled rotations are; for a unit quat q^-1 == conj(q) == (-x,-y,-z,w) EXACTLY,
// no sqrt / division). This is what makes the bind-delta collapse EXACT: for a unit q with norm^2 ==
// kOne in fixed point (the exact-unit fixtures), q (x) conj(q) == (0,0,0,kOne) the identity quaternion
// (the vector part cancels; w == a^2+b^2+c^2+d^2 == kOne), and identity (x) p == p EXACTLY (fxmul by
// kOne is a no-op). A true normalize (integer sqrt LUT) is a future refinement for non-unit inputs;
// v1 does not need it and stays division-free.
//
// COMPOSITION ORDER (PINNED — the determinism contract): for each MAPPED target bone the retargeted
// LOCAL rotation is
//     R_target = D (x) sanim,   with the bind-delta D = tbind (x) conj(sbind),
// i.e. FxQuatMul(FxQuatMul(tbind, FxQuatConj(sbind)), sanim). tbind / sbind are the TARGET / SOURCE
// bind LOCAL rotations; sanim is the SOURCE sampled LOCAL rotation. The bind delta re-expresses the
// source's absolute local rotation in the target's rest frame, so DIFFERENT rest orientations retarget
// correctly: at rest (sanim == sbind) R_target collapses to tbind (target rest maps to target rest);
// self-retarget (tbind == sbind) collapses D to identity so R_target == sanim (the SOURCE motion,
// reproduced BIT-EXACT). Quaternion product is Hamilton (a (x) b) == "apply b, then a".
//
// TRANSLATION (preserve TARGET proportions, the standard rule): every MAPPED non-root bone's local
// translation is taken from the TARGET BIND pose (the target's own bone lengths — NOT the source's,
// NOT the clip's; clip translation on non-root bones is intentionally dropped, the "rotation-only
// retarget" contract). ONLY the ROOT bone's translation is retargeted, SCALED by the height ratio
// (standard root-motion scale so a taller target strides proportionally further):
//     t_root_target = t_root_sourceAnim * heightRatio   (per component, truncating fxmul).
// HEIGHT RATIO derivation (PINNED choice): heightRatio == targetHeight / sourceHeight in Q16.16
// (((tgtH << kFrac) / srcH), truncating), where HEIGHT is the bind-pose model-space VERTICAL distance
// |modelY(footBone) - modelY(rootBone)| along the designated root->foot leg chain (a "leg-length"
// ratio — the most robust of the common choices for locomotion clips; hip-height is the same number
// here). Self-retarget -> ratio == kOne -> root translation passes through EXACTLY.
//
// BONE MAP (name-matched v1, documented): skeleton.h's Joint carries NO name (byte-untouched), so the
// bone NAMES are supplied as a PARALLEL std::vector<std::string> alongside each skeleton (glTF node
// names, engine bone ids — the caller's provenance). BuildRetargetMap maps each TARGET bone to a
// SOURCE bone by CASE-EXACT name match; an UNMATCHED target bone gets source index -1 and HOLDS ITS
// BIND pose (rotation == tbind, translation == target bind t) — the pinned unmapped convention.
// Explicit {targetIdx, sourceIdx} overrides are applied AFTER name matching (they win). HONEST v1
// LIMITS: this is a NAME-matched, per-bone rotation+root-translation retarget — NO twist-bone
// redistribution, NO IK end-effector preservation (foot/hand planting), NO chain re-proportioning
// beyond the root scale; those are future slices. On EXTREME proportion/rest-orientation differences
// the bind-delta is exact algebra but the visual result inherits the source's joint angles verbatim
// (the retarget does not re-solve limb reach) — the tests pin the ACTUAL behavior, not an idealized one.
//
// SHOWCASE (--an2-retarget-shot, both backends): the AN1 header-shared-scenario pattern — a side-by-
// side stick-figure viz (SOURCE skeleton posed by the clip on the left; the differently-proportioned
// TARGET playing the RETARGETED motion on the right), bones drawn as line segments between model-space
// joint positions over a few keyframes, from the ONE pure-integer scenario both backends run (strict-
// zero cross-backend BY CONSTRUCTION; NO new shader, NO lit render).

#include <cstdint>
#include <string>
#include <vector>

#include "anim/animation.h"     // READ-ONLY: Animation/Channel/JointPose/SampleLocalPose — the source pose
#include "anim/motion_match.h"  // READ-ONLY: mm::fx/kOne/kFrac/fxmul/QuantizeFx/FxV3 + Fnv1a64Word (the Q16.16 currency)
#include "anim/skeleton.h"      // READ-ONLY: Skeleton/Joint — the bone hierarchy + bind pose

namespace hf::anim {
namespace retarget {

using mm::fx;
using mm::FxV3;
using mm::fxmul;
using mm::kFrac;
using mm::kOne;
using mm::QuantizeFx;   // THE float->integer boundary (llround, round-half-away — the MM1 discipline)

// ===================== Q16.16 quaternion algebra (integer, division-free) =============================

// A Q16.16 quaternion (x, y, z, w); identity == (0, 0, 0, kOne). Rotation quats are UNIT (see banner).
struct FxQuat {
    fx x = 0, y = 0, z = 0, w = kOne;
};

// The identity rotation.
inline FxQuat FxQuatIdentity() { return FxQuat{0, 0, 0, kOne}; }

// Quantize a float quaternion (the ONE boundary — component-wise llround).
inline FxQuat FxQuatFromFloat(const math::Quat& q) {
    return FxQuat{QuantizeFx(q.x), QuantizeFx(q.y), QuantizeFx(q.z), QuantizeFx(q.w)};
}

// CONJUGATE == INVERSE for a UNIT quaternion (banner): (-x, -y, -z, w). Pure integer, exact.
inline FxQuat FxQuatConj(const FxQuat& q) { return FxQuat{-q.x, -q.y, -q.z, q.w}; }

// FxQuatMul: the Hamilton product a (x) b (int64 intermediates, arithmetic >> kFrac in fxmul). Convention:
// (a (x) b) applied to a vector rotates by b FIRST, then a. This is the pinned composition primitive.
inline FxQuat FxQuatMul(const FxQuat& a, const FxQuat& b) {
    FxQuat r;
    r.w = fxmul(a.w, b.w) - fxmul(a.x, b.x) - fxmul(a.y, b.y) - fxmul(a.z, b.z);
    r.x = fxmul(a.w, b.x) + fxmul(a.x, b.w) + fxmul(a.y, b.z) - fxmul(a.z, b.y);
    r.y = fxmul(a.w, b.y) - fxmul(a.x, b.z) + fxmul(a.y, b.w) + fxmul(a.z, b.x);
    r.z = fxmul(a.w, b.z) + fxmul(a.x, b.y) - fxmul(a.y, b.x) + fxmul(a.z, b.w);
    return r;
}

// FxQuatRotate: rotate the vector v by the UNIT quaternion q — v' = v + 2*(w*(qv x v) + qv x (qv x v)),
// qv == (x, y, z). All integer (fxmul); rotating by identity returns v EXACTLY (qv == 0). The forward-
// kinematics primitive (child offsets rotated into the parent's world frame).
inline FxV3 FxQuatRotate(const FxQuat& q, const FxV3& v) {
    // uv = qv x v
    const fx uvx = fxmul(q.y, v.z) - fxmul(q.z, v.y);
    const fx uvy = fxmul(q.z, v.x) - fxmul(q.x, v.z);
    const fx uvz = fxmul(q.x, v.y) - fxmul(q.y, v.x);
    // uuv = qv x uv
    const fx uuvx = fxmul(q.y, uvz) - fxmul(q.z, uvy);
    const fx uuvy = fxmul(q.z, uvx) - fxmul(q.x, uvz);
    const fx uuvz = fxmul(q.x, uvy) - fxmul(q.y, uvx);
    FxV3 out;
    out.x = v.x + 2 * (fxmul(q.w, uvx) + uuvx);
    out.y = v.y + 2 * (fxmul(q.w, uvy) + uuvy);
    out.z = v.z + 2 * (fxmul(q.w, uvz) + uuvz);
    return out;
}

// ===================== The retargeted pose (fixed-point local TRS) ====================================

// One target joint's retargeted LOCAL transform in Q16.16 (rotation + translation; scale is UNIFORM 1
// in v1 — bone lengths live in the bind translations, documented). The pinned digest currency.
struct FxJointPose {
    FxV3   t;                    // local translation (Q16.16)
    FxQuat r = FxQuatIdentity(); // local rotation (Q16.16 unit quat)
};

// QuantizePose: a FLOAT anim::JointPose stream -> the Q16.16 FxJointPose stream (rotation + translation;
// scale dropped — v1 uniform-scale). The reusable float boundary for comparing a source pose bit-for-bit.
inline std::vector<FxJointPose> QuantizePose(const std::vector<JointPose>& pose) {
    std::vector<FxJointPose> out(pose.size());
    for (size_t j = 0; j < pose.size(); ++j) {
        out[j].t = FxV3{QuantizeFx(pose[j].t.x), QuantizeFx(pose[j].t.y), QuantizeFx(pose[j].t.z)};
        out[j].r = FxQuatFromFloat(pose[j].r);
    }
    return out;
}

// BindPose: a skeleton's bind (rest) local TRS as a fixed-point pose (rotation + translation).
inline std::vector<FxJointPose> BindPose(const Skeleton& sk) {
    std::vector<FxJointPose> out(sk.joints.size());
    for (size_t j = 0; j < sk.joints.size(); ++j) {
        out[j].t = FxV3{QuantizeFx(sk.joints[j].t.x), QuantizeFx(sk.joints[j].t.y),
                        QuantizeFx(sk.joints[j].t.z)};
        out[j].r = FxQuatFromFloat(sk.joints[j].r);
    }
    return out;
}

// ===================== Forward kinematics (integer, rigid) ============================================

// A joint's model-space (world) rigid transform: position + orientation (no scale — banner). The FK
// output the model-space digests + the stick-figure viz read.
struct FxJointModel {
    FxV3   pos;
    FxQuat rot = FxQuatIdentity();
};

// ForwardKinematics: walk the (topologically sorted) hierarchy — worldPos[j] = worldPos[parent] +
// rotate(worldRot[parent], local.t); worldRot[j] = worldRot[parent] (x) local.r. Pure integer; a parent
// always precedes its children (the anim contract), so one forward pass suffices.
inline std::vector<FxJointModel> ForwardKinematics(const Skeleton& sk,
                                                   const std::vector<FxJointPose>& local) {
    const size_t n = sk.joints.size();
    std::vector<FxJointModel> out(n);
    for (size_t j = 0; j < n && j < local.size(); ++j) {
        const int parent = sk.joints[j].parent;
        if (parent >= 0 && (size_t)parent < j) {
            out[j].rot = FxQuatMul(out[(size_t)parent].rot, local[j].r);
            const FxV3 off = FxQuatRotate(out[(size_t)parent].rot, local[j].t);
            out[j].pos = FxV3{out[(size_t)parent].pos.x + off.x, out[(size_t)parent].pos.y + off.y,
                              out[(size_t)parent].pos.z + off.z};
        } else {
            out[j].rot = local[j].r;
            out[j].pos = local[j].t;
        }
    }
    return out;
}

// ===================== The bone map (name-matched v1) =================================================

// An explicit bone-map override: force target bone `targetIdx` to sample source bone `sourceIdx`
// (applied AFTER name matching — overrides win). sourceIdx == -1 forces "unmapped" (hold bind).
struct RetargetOverride {
    int32_t targetIdx = -1;
    int32_t sourceIdx = -1;
};

// The built retarget map: the two skeletons (stored by value — small), the per-target-bone source
// index (or -1 unmapped), the root designation, and the Q16.16 height ratio (banner). `mappedCount`
// is a convenience (number of target bones with a source).
struct RetargetMap {
    Skeleton             source;
    Skeleton             target;
    std::vector<int32_t> targetToSource;   // size == target.joints; -1 == unmapped (hold bind)
    int32_t              rootTarget = 0;
    int32_t              rootSource = 0;
    fx                   heightRatio = kOne;
    int32_t              mappedCount = 0;
};

namespace detail {

// NameIndex: the FIRST index whose name matches `want` case-exact, or -1. (Duplicate names resolve to
// the first — the deterministic convention; well-formed skeletons have unique bone names.)
inline int32_t NameIndex(const std::vector<std::string>& names, const std::string& want) {
    for (size_t i = 0; i < names.size(); ++i)
        if (names[i] == want) return (int32_t)i;
    return -1;
}

// BindModelY: the model-space Y of joint `idx` in the skeleton's BIND pose (the FK Y coordinate). Used
// only for the height-ratio derivation; identity-bind leg chains make this a clean translation sum.
inline fx BindModelY(const Skeleton& sk, int32_t idx) {
    if (idx < 0 || (size_t)idx >= sk.joints.size()) return 0;
    const std::vector<FxJointModel> g = ForwardKinematics(sk, BindPose(sk));
    return g[(size_t)idx].pos.y;
}

}  // namespace detail

// BuildRetargetMap: map TARGET bones to SOURCE bones by CASE-EXACT name (banner), then apply overrides.
// `rootName` designates the root bone (must exist in BOTH); `footName` designates the leg-chain tip for
// the height ratio (must exist in BOTH — falls back to ratio kOne if missing/degenerate). Unmatched
// target bones -> -1 (hold bind).
inline RetargetMap BuildRetargetMap(const Skeleton& source, const std::vector<std::string>& sourceNames,
                                    const Skeleton& target, const std::vector<std::string>& targetNames,
                                    const std::string& rootName, const std::string& footName,
                                    const std::vector<RetargetOverride>& overrides = {}) {
    RetargetMap m;
    m.source = source;
    m.target = target;
    m.targetToSource.assign(target.joints.size(), -1);
    // (1) Name match: each target bone -> the source bone of the same name.
    for (size_t tj = 0; tj < targetNames.size() && tj < target.joints.size(); ++tj)
        m.targetToSource[tj] = detail::NameIndex(sourceNames, targetNames[tj]);
    // (2) Overrides win (bounds-checked).
    for (const RetargetOverride& o : overrides)
        if (o.targetIdx >= 0 && (size_t)o.targetIdx < m.targetToSource.size())
            m.targetToSource[(size_t)o.targetIdx] = o.sourceIdx;
    // (3) Roots (name-resolved; default 0 if absent).
    const int32_t rt = detail::NameIndex(targetNames, rootName);
    const int32_t rs = detail::NameIndex(sourceNames, rootName);
    m.rootTarget = rt >= 0 ? rt : 0;
    m.rootSource = rs >= 0 ? rs : 0;
    // (4) Height ratio == targetHeight / sourceHeight (leg-length, banner). Guard a degenerate source.
    const int32_t footT = detail::NameIndex(targetNames, footName);
    const int32_t footS = detail::NameIndex(sourceNames, footName);
    fx srcH = 0, tgtH = 0;
    if (footS >= 0) {
        const fx d = detail::BindModelY(source, footS) - detail::BindModelY(source, m.rootSource);
        srcH = d < 0 ? -d : d;
    }
    if (footT >= 0) {
        const fx d = detail::BindModelY(target, footT) - detail::BindModelY(target, m.rootTarget);
        tgtH = d < 0 ? -d : d;
    }
    m.heightRatio = (srcH > 0) ? (fx)((((int64_t)tgtH) << kFrac) / srcH) : kOne;
    // (5) mappedCount.
    for (int32_t s : m.targetToSource) if (s >= 0) ++m.mappedCount;
    return m;
}

// ===================== The retarget core =============================================================

// Retarget: map a SOURCE local pose (float — animation.h SampleLocalPose output) onto the TARGET
// skeleton, producing the target's Q16.16 LOCAL pose. Per the banner:
//   * MAPPED bone: R = (tbind (x) conj(sbind)) (x) sanim (the PINNED bind-delta composition); translation
//     from the TARGET bind, EXCEPT the ROOT whose translation is source-anim * heightRatio.
//   * UNMAPPED bone: holds its TARGET BIND (rotation tbind, translation target bind t).
// `sourcePose` is indexed by SOURCE joint. The result is indexed by TARGET joint.
inline std::vector<FxJointPose> Retarget(const RetargetMap& m,
                                         const std::vector<JointPose>& sourcePose) {
    const std::vector<FxJointPose> sBind = BindPose(m.source);
    const std::vector<FxJointPose> tBind = BindPose(m.target);
    const std::vector<FxJointPose> sPoseQ = QuantizePose(sourcePose);
    const size_t nT = m.target.joints.size();
    std::vector<FxJointPose> out(nT);
    for (size_t tj = 0; tj < nT; ++tj) {
        const int32_t s = (tj < m.targetToSource.size()) ? m.targetToSource[tj] : -1;
        if (s < 0 || (size_t)s >= sPoseQ.size()) {
            out[tj] = tBind[tj];   // UNMAPPED: hold the target bind (pinned)
            continue;
        }
        // ROTATION: the bind-delta composition (pinned order).
        const FxQuat D = FxQuatMul(tBind[tj].r, FxQuatConj(sBind[(size_t)s].r));
        out[tj].r = FxQuatMul(D, sPoseQ[(size_t)s].r);
        // TRANSLATION: target bind everywhere; root scaled by the height ratio.
        if ((int32_t)tj == m.rootTarget) {
            const FxV3 sr = sPoseQ[(size_t)s].t;
            out[tj].t = FxV3{fxmul(sr.x, m.heightRatio), fxmul(sr.y, m.heightRatio),
                             fxmul(sr.z, m.heightRatio)};
        } else {
            out[tj].t = tBind[tj].t;
        }
    }
    return out;
}

// ===================== Digests (the pinned-golden currency, FNV-1a-64) ================================

// DigestFxPose: fold a fixed-point LOCAL pose (t.xyz + r.xyzw per joint) through the MM1 FNV primitive.
inline uint64_t DigestFxPose(const std::vector<FxJointPose>& pose) {
    uint64_t h = 14695981039346656037ull;
    for (const FxJointPose& p : pose) {
        h = mm::detail::Fnv1a64Word(h, (uint32_t)p.t.x);
        h = mm::detail::Fnv1a64Word(h, (uint32_t)p.t.y);
        h = mm::detail::Fnv1a64Word(h, (uint32_t)p.t.z);
        h = mm::detail::Fnv1a64Word(h, (uint32_t)p.r.x);
        h = mm::detail::Fnv1a64Word(h, (uint32_t)p.r.y);
        h = mm::detail::Fnv1a64Word(h, (uint32_t)p.r.z);
        h = mm::detail::Fnv1a64Word(h, (uint32_t)p.r.w);
    }
    return h;
}

// DigestModel: fold a model-space FK result (pos.xyz per joint) — the "palette" digest currency.
inline uint64_t DigestModel(const std::vector<FxJointModel>& g) {
    uint64_t h = 14695981039346656037ull;
    for (const FxJointModel& j : g) {
        h = mm::detail::Fnv1a64Word(h, (uint32_t)j.pos.x);
        h = mm::detail::Fnv1a64Word(h, (uint32_t)j.pos.y);
        h = mm::detail::Fnv1a64Word(h, (uint32_t)j.pos.z);
    }
    return h;
}

// FxPoseEqual: field-exact equality (the identity / unmapped proofs — no tolerance).
inline bool FxPoseEqual(const std::vector<FxJointPose>& a, const std::vector<FxJointPose>& b) {
    if (a.size() != b.size()) return false;
    for (size_t j = 0; j < a.size(); ++j) {
        if (a[j].t.x != b[j].t.x || a[j].t.y != b[j].t.y || a[j].t.z != b[j].t.z) return false;
        if (a[j].r.x != b[j].r.x || a[j].r.y != b[j].r.y || a[j].r.z != b[j].r.z ||
            a[j].r.w != b[j].r.w)
            return false;
    }
    return true;
}

// ===================== The shared showcase scenario (--an2-retarget-shot, both backends) =============
// The AN1 header-shared-scenario pattern: RunRetargetShotScenario + RenderRetargetShot are the ONE
// implementation both backends call — strict-zero cross-backend BY CONSTRUCTION (pure-CPU integer
// scenario; the raster consumes only integers).

// The fixture skeletons share the SAME bone NAMES; the SOURCE has normal proportions, the TARGET has
// LONG legs + SHORT arms. 7 bones. All bind rotations identity (the showcase isolates the proportion
// remap; the bind-DELTA math is exercised by retarget_test). Every value is a binary fraction.
//   0 root  1 spine(0)  2 head(1)  3 armUpperL(1)  4 armLowerL(3)  5 legUpperL(0)  6 legLowerL(5)
inline std::vector<std::string> ShowcaseBoneNames() {
    return {"root", "spine", "head", "armUpperL", "armLowerL", "legUpperL", "legLowerL"};
}

inline Skeleton MakeSourceSkeleton() {
    Skeleton sk;
    Joint root;                                            // 0 root
    Joint spine;     spine.parent = 0;     spine.t = math::Vec3{0.0f, 1.0f, 0.0f};      // 1
    Joint head;      head.parent = 1;      head.t = math::Vec3{0.0f, 0.5f, 0.0f};       // 2
    Joint armU;      armU.parent = 1;      armU.t = math::Vec3{0.5f, 0.0f, 0.0f};       // 3
    Joint armL;      armL.parent = 3;      armL.t = math::Vec3{0.5f, 0.0f, 0.0f};       // 4
    Joint legU;      legU.parent = 0;      legU.t = math::Vec3{0.25f, -1.0f, 0.0f};     // 5
    Joint legL;      legL.parent = 5;      legL.t = math::Vec3{0.0f, -1.0f, 0.0f};      // 6
    sk.joints = {root, spine, head, armU, armL, legU, legL};
    return sk;
}

inline Skeleton MakeTargetSkeleton() {
    Skeleton sk;
    Joint root;                                            // 0 root
    Joint spine;     spine.parent = 0;     spine.t = math::Vec3{0.0f, 1.0f, 0.0f};      // 1 (same spine)
    Joint head;      head.parent = 1;      head.t = math::Vec3{0.0f, 0.5f, 0.0f};       // 2
    Joint armU;      armU.parent = 1;      armU.t = math::Vec3{0.25f, 0.0f, 0.0f};      // 3 SHORT arm
    Joint armL;      armL.parent = 3;      armL.t = math::Vec3{0.25f, 0.0f, 0.0f};      // 4 SHORT arm
    Joint legU;      legU.parent = 0;      legU.t = math::Vec3{0.25f, -2.0f, 0.0f};     // 5 LONG leg
    Joint legL;      legL.parent = 5;      legL.t = math::Vec3{0.0f, -2.0f, 0.0f};      // 6 LONG leg
    sk.joints = {root, spine, head, armU, armL, legU, legL};
    return sk;
}

// The retarget clip: root strides +z (locomotion — the root-scale exercise) and the legs/arms swing
// between identity and the exact-unit quat qSwing == (0.5, 0.5, 0.5, 0.5) (a STYLIZED swing — the only
// exact binary-fraction unit quats are the axis-aligned 0/180-deg turns and the (+-1/2,+-1/2,+-1/2,
// +-1/2) family; a real leg swing about X has no binary-fraction unit quat, documented). Keys at exact
// integer times 0..3, sampled AT those times -> Slerp returns the key EXACTLY (frac 0, unit input).
inline Animation MakeRetargetClip() {
    Animation a;
    a.name = "an2_walk";
    a.duration = 3.0f;
    // root: stride +z (0 -> 1.5 over the clip; the target scales this by the height ratio).
    Channel root;
    root.jointIndex = 0; root.path = Channel::Path::Translation; root.interp = Channel::Interp::Linear;
    root.times = {0.0f, 1.0f, 2.0f, 3.0f};
    root.values = {0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.5f,  0.0f, 0.0f, 1.0f,  0.0f, 0.0f, 1.5f};
    a.channels.push_back(root);
    // legUpperL (5): swing identity / qSwing.
    Channel legU;
    legU.jointIndex = 5; legU.path = Channel::Path::Rotation; legU.interp = Channel::Interp::Linear;
    legU.times = {0.0f, 1.0f, 2.0f, 3.0f};
    legU.values = {0.0f, 0.0f, 0.0f, 1.0f,   0.5f, 0.5f, 0.5f, 0.5f,
                   0.0f, 0.0f, 0.0f, 1.0f,   0.5f, 0.5f, 0.5f, 0.5f};
    a.channels.push_back(legU);
    // armUpperL (3): swing in ANTIPHASE (qSwing / identity).
    Channel armU;
    armU.jointIndex = 3; armU.path = Channel::Path::Rotation; armU.interp = Channel::Interp::Linear;
    armU.times = {0.0f, 1.0f, 2.0f, 3.0f};
    armU.values = {0.5f, 0.5f, 0.5f, 0.5f,   0.0f, 0.0f, 0.0f, 1.0f,
                   0.5f, 0.5f, 0.5f, 0.5f,   0.0f, 0.0f, 0.0f, 1.0f};
    a.channels.push_back(armU);
    return a;
}

inline constexpr int kShotFrames = 4;   // sample the clip at integer times 0..3 (exact keyframes)

// One recorded frame of the shot: the SOURCE model-space joints (posed by the clip) and the TARGET
// model-space joints (playing the retargeted motion) — the stick-figure viz reads these.
struct RetargetShotFrame {
    std::vector<FxJointModel> sourceModel;
    std::vector<FxJointModel> targetModel;
};

struct RetargetShotRun {
    RetargetMap                    map;
    std::vector<RetargetShotFrame> frames;
    int32_t                        srcBones = 0;
    int32_t                        tgtBones = 0;
    int32_t                        mapped = 0;
    fx                             heightRatio = kOne;
    uint64_t                       sourceDigest = 0;   // FNV over the source model-space frames
    uint64_t                       targetDigest = 0;   // FNV over the target model-space frames
    uint64_t                       localDigest = 0;    // FNV over the retargeted LOCAL poses
    uint64_t                       digest = 0;         // the combined two-run comparison currency
};

// RunRetargetShotScenario: the pure function both backends call — build the name-matched map, then for
// each keyframe sample the SOURCE clip, FK the source, retarget onto the TARGET, FK the target.
inline RetargetShotRun RunRetargetShotScenario() {
    RetargetShotRun run;
    const Skeleton src = MakeSourceSkeleton();
    const Skeleton tgt = MakeTargetSkeleton();
    const std::vector<std::string> names = ShowcaseBoneNames();
    run.map = BuildRetargetMap(src, names, tgt, names, "root", "legLowerL");
    run.srcBones = (int32_t)src.joints.size();
    run.tgtBones = (int32_t)tgt.joints.size();
    run.mapped = run.map.mappedCount;
    run.heightRatio = run.map.heightRatio;
    const Animation clip = MakeRetargetClip();
    uint64_t hs = 14695981039346656037ull, ht = hs, hl = hs;
    run.frames.reserve((size_t)kShotFrames);
    for (int f = 0; f < kShotFrames; ++f) {
        const float t = (float)f;
        const std::vector<JointPose> sPose = SampleLocalPose(src, clip, t);
        const std::vector<FxJointPose> sPoseQ = QuantizePose(sPose);
        const std::vector<FxJointPose> tPose = Retarget(run.map, sPose);
        RetargetShotFrame fr;
        fr.sourceModel = ForwardKinematics(src, sPoseQ);
        fr.targetModel = ForwardKinematics(tgt, tPose);
        // Fold the model-space positions + the retargeted local pose into the digests.
        const uint64_t ds = DigestModel(fr.sourceModel);
        const uint64_t dt = DigestModel(fr.targetModel);
        const uint64_t dl = DigestFxPose(tPose);
        hs = mm::detail::Fnv1a64Word(hs, (uint32_t)(ds & 0xffffffffu));
        hs = mm::detail::Fnv1a64Word(hs, (uint32_t)(ds >> 32));
        ht = mm::detail::Fnv1a64Word(ht, (uint32_t)(dt & 0xffffffffu));
        ht = mm::detail::Fnv1a64Word(ht, (uint32_t)(dt >> 32));
        hl = mm::detail::Fnv1a64Word(hl, (uint32_t)(dl & 0xffffffffu));
        hl = mm::detail::Fnv1a64Word(hl, (uint32_t)(dl >> 32));
        run.frames.push_back(std::move(fr));
    }
    run.sourceDigest = hs;
    run.targetDigest = ht;
    run.localDigest = hl;
    uint64_t h = 14695981039346656037ull;
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hs & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hs >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(ht & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(ht >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hl & 0xffffffffu));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)(hl >> 32));
    h = mm::detail::Fnv1a64Word(h, (uint32_t)run.heightRatio);
    run.digest = h;
    return run;
}

// RenderRetargetShot: the PURE-INTEGER raster both backends call — strict-zero cross-backend BY
// CONSTRUCTION. A side-by-side FRONT view: SOURCE stick figure (left) posed by the clip; TARGET stick
// figure (right, LONG legs / SHORT arms) playing the retargeted motion. Each keyframe is overlaid with
// a small per-frame screen shear (so the progression reads) and per-frame brightness. Bones are line
// segments from each joint's model-space position to its parent's. 520x420 BGRA8, integer DDA only.
inline void RenderRetargetShot(const RetargetShotRun& run, std::vector<uint8_t>& bgra, uint32_t& outW,
                               uint32_t& outH) {
    const int W = 520, H = 420;
    outW = (uint32_t)W; outH = (uint32_t)H;
    bgra.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {   // deep slate ground
        bgra[p * 4 + 0] = 24; bgra[p * 4 + 1] = 18; bgra[p * 4 + 2] = 14; bgra[p * 4 + 3] = 255;
    }
    auto putPx = [&](int ix, int iy, uint8_t r, uint8_t g, uint8_t b) {
        if (ix < 0 || ix >= W || iy < 0 || iy >= H) return;
        uint8_t* dst = &bgra[((size_t)iy * W + ix) * 4];
        dst[0] = b; dst[1] = g; dst[2] = r; dst[3] = 255;
    };
    auto line = [&](int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b) {
        const int dx = x1 - x0, dy = y1 - y0;
        int steps = dx >= 0 ? dx : -dx;
        const int ady = dy >= 0 ? dy : -dy;
        if (ady > steps) steps = ady;
        if (steps == 0) { putPx(x0, y0, r, g, b); return; }
        for (int i = 0; i <= steps; ++i) {   // integer DDA (rounded interpolation — the AN1 raster)
            const int px = x0 + (int)(((int64_t)dx * i * 2 + steps) / (2 * (int64_t)steps));
            const int py = y0 + (int)(((int64_t)dy * i * 2 + steps) / (2 * (int64_t)steps));
            putPx(px, py, r, g, b);
        }
    };
    auto disc = [&](int cx, int cy, int rr, uint8_t r, uint8_t g, uint8_t b) {
        for (int dy = -rr; dy <= rr; ++dy)
            for (int dx = -rr; dx <= rr; ++dx)
                if (dx * dx + dy * dy <= rr * rr) putPx(cx + dx, cy + dy, r, g, b);
    };
    // Vertical divider between the two panels.
    for (int y = 0; y < H; ++y) putPx(260, y, 70, 78, 90);
    // Model-space (x, y) -> screen. Left panel centered x=130, right x=390; y=0 (root) at py=150; 60
    // px/unit; +z sheared 5 px/unit to the right so the stride reads; +f a small extra shear.
    const int kScale = 60, kBaseY = 150;
    auto project = [&](const FxV3& p, int centerX, int frame, int& sx, int& sy) {
        sx = centerX + (int)(((int64_t)p.x * kScale) >> kFrac) +
             (int)(((int64_t)p.z * 5) >> kFrac) + frame * 4;
        sy = kBaseY - (int)(((int64_t)p.y * kScale) >> kFrac);
    };
    // Per-frame brightness (older frames dimmer).
    auto shade = [&](int frame, int nFrames, uint8_t base) -> uint8_t {
        const int lo = 90;
        const int v = lo + (base - lo) * (frame + 1) / (nFrames > 0 ? nFrames : 1);
        return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    };
    const int nF = (int)run.frames.size();
    // Draw bones per skeleton (need the parent array — source uses run.map.source, target run.map.target).
    auto drawSkel = [&](const Skeleton& sk, const std::vector<FxJointModel>& g, int centerX, int frame,
                        uint8_t br, uint8_t bg, uint8_t bb) {
        const uint8_t r = shade(frame, nF, br), gg = shade(frame, nF, bg), b = shade(frame, nF, bb);
        for (size_t j = 0; j < g.size() && j < sk.joints.size(); ++j) {
            int sx, sy; project(g[j].pos, centerX, frame, sx, sy);
            const int parent = sk.joints[j].parent;
            if (parent >= 0 && (size_t)parent < g.size()) {
                int px, py; project(g[(size_t)parent].pos, centerX, frame, px, py);
                line(px, py, sx, sy, r, gg, b);
            }
            disc(sx, sy, 2, r, gg, b);
        }
    };
    for (int f = 0; f < nF; ++f) {
        drawSkel(run.map.source, run.frames[(size_t)f].sourceModel, 130, f, 120, 200, 232);  // cyan source
        drawSkel(run.map.target, run.frames[(size_t)f].targetModel, 390, f, 232, 168, 96);   // amber target
    }
}

}  // namespace retarget
}  // namespace hf::anim
