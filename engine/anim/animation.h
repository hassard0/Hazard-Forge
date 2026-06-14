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

// Sample `animation` at `time` (seconds) against `skeleton`'s rest pose, returning the joint-matrix
// palette: palette[j] = global[j] * inverseBind[j], where global[j] is joint j's model-space
// transform after applying the sampled local TRS and walking the (topologically sorted) hierarchy.
// `time` is clamped to [0, duration]. The returned vector has one Mat4 per joint, in skeleton order.
std::vector<math::Mat4> SampleAnimation(const Skeleton& skeleton, const Animation& animation,
                                        float time);

} // namespace hf::anim
