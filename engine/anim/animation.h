#pragma once
#include "anim/skeleton.h"
#include "math/math.h"
#include <string>
#include <vector>

namespace hf::anim {

// One animation channel: a stream of keyframes targeting one TRS component of one joint.
//   * jointIndex  — index into Skeleton::joints (already remapped from the glTF node index).
//   * path        — which local-transform component this channel drives.
//   * times       — keyframe timestamps in seconds (sorted ascending).
//   * values      — packed keyframe values: 3 floats/key for Translation/Scale, 4 floats/key
//                   (xyzw quaternion) for Rotation. values.size() == times.size() * stride.
//   * interp      — Linear (lerp T/S, slerp R) or Step (hold the previous key).
struct Channel {
    enum class Path { Translation, Rotation, Scale };
    enum class Interp { Linear, Step };
    int   jointIndex = -1;
    Path  path = Path::Translation;
    Interp interp = Interp::Linear;
    std::vector<float> times;
    std::vector<float> values;
};

struct Animation {
    std::string name;                 // glTF animation name (e.g. "Survey"/"Walk"/"Run")
    float duration = 0.0f;            // seconds (max keyframe time across channels)
    std::vector<Channel> channels;
};

// A joint's LOCAL transform as decomposed TRS (one per joint, in skeleton order). This is the
// intermediate "local pose" produced by sampling a clip, BEFORE the hierarchy walk. Keeping the
// pose decomposed (rather than as a Mat4) is what lets us blend two clips per-component: lerp the
// translations/scales and slerp the rotations, then compose to matrices once at the end.
struct JointPose {
    math::Vec3 t{0, 0, 0};
    math::Quat r{0, 0, 0, 1};
    math::Vec3 s{1, 1, 1};
};

// Sample `animation` at `time` (seconds) against `skeleton`'s rest pose into a LOCAL pose: one local
// TRS per joint, in skeleton order. Each joint starts from its rest TRS; every channel overrides the
// component it targets. Joints with no channel keep their rest TRS. `time` is clamped to [0, duration].
std::vector<JointPose> SampleLocalPose(const Skeleton& skeleton, const Animation& animation,
                                       float time);

// Compose a local pose into the joint-matrix palette: walk the (topologically sorted) hierarchy to
// build each joint's global (model-space) transform, then palette[j] = global[j] * inverseBind[j].
std::vector<math::Mat4> PaletteFromLocalPose(const Skeleton& skeleton,
                                             const std::vector<JointPose>& pose);

// Per-joint blend of two local poses at `weight` in [0,1]: t/s are lerped, r is Slerp'd
// (shortest-arc, normalized). weight 0 -> pose `a`, weight 1 -> pose `b`. If the inputs differ in
// length, the shorter joint count is used.
std::vector<JointPose> BlendLocalPoses(const std::vector<JointPose>& a,
                                       const std::vector<JointPose>& b, float weight);

// Sample `animation` at `time` (seconds) against `skeleton`'s rest pose, returning the joint-matrix
// palette: palette[j] = global[j] * inverseBind[j], where global[j] is joint j's model-space
// transform after applying the sampled local TRS and walking the (topologically sorted) hierarchy.
// `time` is clamped to [0, duration]. The returned vector has one Mat4 per joint, in skeleton order.
std::vector<math::Mat4> SampleAnimation(const Skeleton& skeleton, const Animation& animation,
                                        float time);

// Sample two clips (animA at timeA, animB at timeB) into local poses, blend them per-joint at
// `weight` in [0,1], and compose the blended pose into a joint-matrix palette. weight 0 -> pure
// animA, weight 1 -> pure animB. Both clips are sampled against the same `skeleton`.
std::vector<math::Mat4> BlendAnimations(const Skeleton& skeleton,
                                        const Animation& animA, float timeA,
                                        const Animation& animB, float timeB, float weight);

} // namespace hf::anim
