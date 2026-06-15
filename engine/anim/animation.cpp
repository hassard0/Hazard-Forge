#include "anim/animation.h"

#include <algorithm>

namespace hf::anim {

namespace {

// Find the keyframe interval [i, i+1] bracketing `time` in a sorted `times` array, and the
// interpolation fraction in [0,1]. Clamps to the first/last key outside the range. Returns the
// lower index `i0`; `i1` is i0 for the single-key / clamped-end case.
struct KeyLerp { size_t i0; size_t i1; float frac; };

KeyLerp FindKey(const std::vector<float>& times, float time) {
    const size_t n = times.size();
    if (n == 0) return {0, 0, 0.0f};
    if (n == 1 || time <= times.front()) return {0, 0, 0.0f};
    if (time >= times.back())            return {n - 1, n - 1, 0.0f};
    // Binary search for the first key strictly greater than `time`.
    size_t hi = static_cast<size_t>(
        std::upper_bound(times.begin(), times.end(), time) - times.begin());
    size_t i0 = hi - 1;
    size_t i1 = hi;
    float span = times[i1] - times[i0];
    float frac = (span > 0.0f) ? (time - times[i0]) / span : 0.0f;
    return {i0, i1, frac};
}

} // namespace

std::vector<JointPose> SampleLocalPose(const Skeleton& skeleton, const Animation& animation,
                                       float time) {
    const size_t jointCount = skeleton.joints.size();

    // Start from each joint's REST local TRS, then let channels override components.
    std::vector<JointPose> pose(jointCount);
    for (size_t j = 0; j < jointCount; ++j) {
        pose[j].t = skeleton.joints[j].t;
        pose[j].r = skeleton.joints[j].r;
        pose[j].s = skeleton.joints[j].s;
    }

    // Clamp the sample time into the clip's range (no looping here; the showcase samples a fixed t).
    float t = std::clamp(time, 0.0f, animation.duration);

    for (const Channel& ch : animation.channels) {
        if (ch.jointIndex < 0 || static_cast<size_t>(ch.jointIndex) >= jointCount) continue;
        if (ch.times.empty()) continue;
        KeyLerp k = FindKey(ch.times, t);
        const bool step = (ch.interp == Channel::Interp::Step);
        const float frac = step ? 0.0f : k.frac;

        switch (ch.path) {
            case Channel::Path::Translation: {
                const float* a = &ch.values[k.i0 * 3];
                const float* b = &ch.values[k.i1 * 3];
                pose[ch.jointIndex].t = math::Vec3{
                    a[0] + (b[0] - a[0]) * frac,
                    a[1] + (b[1] - a[1]) * frac,
                    a[2] + (b[2] - a[2]) * frac};
                break;
            }
            case Channel::Path::Scale: {
                const float* a = &ch.values[k.i0 * 3];
                const float* b = &ch.values[k.i1 * 3];
                pose[ch.jointIndex].s = math::Vec3{
                    a[0] + (b[0] - a[0]) * frac,
                    a[1] + (b[1] - a[1]) * frac,
                    a[2] + (b[2] - a[2]) * frac};
                break;
            }
            case Channel::Path::Rotation: {
                const float* a = &ch.values[k.i0 * 4];
                const float* b = &ch.values[k.i1 * 4];
                math::Quat qa{a[0], a[1], a[2], a[3]};
                math::Quat qb{b[0], b[1], b[2], b[3]};
                pose[ch.jointIndex].r = step ? qa : math::Slerp(qa, qb, frac);
                break;
            }
        }
    }
    return pose;
}

std::vector<math::Mat4> PaletteFromLocalPose(const Skeleton& skeleton,
                                             const std::vector<JointPose>& pose) {
    const size_t jointCount = skeleton.joints.size();

    // Single forward pass: joints are topologically sorted (parent before child), so the parent's
    // global transform is already computed when we reach a child.
    std::vector<math::Mat4> global(jointCount);
    for (size_t j = 0; j < jointCount; ++j) {
        math::Mat4 local = math::FromTRS(pose[j].t, pose[j].r, pose[j].s);
        int parent = skeleton.joints[j].parent;
        global[j] = (parent >= 0) ? (global[parent] * local) : local;
    }

    // palette[j] = global[j] * inverseBind[j].
    std::vector<math::Mat4> palette(jointCount);
    for (size_t j = 0; j < jointCount; ++j)
        palette[j] = global[j] * skeleton.joints[j].inverseBind;
    return palette;
}

std::vector<JointPose> BlendLocalPoses(const std::vector<JointPose>& a,
                                       const std::vector<JointPose>& b, float weight) {
    const float w = std::clamp(weight, 0.0f, 1.0f);
    const size_t n = std::min(a.size(), b.size());
    std::vector<JointPose> out(n);
    for (size_t j = 0; j < n; ++j) {
        out[j].t = a[j].t + (b[j].t - a[j].t) * w;   // lerp translation
        out[j].s = a[j].s + (b[j].s - a[j].s) * w;   // lerp scale
        out[j].r = math::Slerp(a[j].r, b[j].r, w);   // shortest-arc, normalized
    }
    return out;
}

std::vector<math::Mat4> SampleAnimation(const Skeleton& skeleton, const Animation& animation,
                                        float time) {
    return PaletteFromLocalPose(skeleton, SampleLocalPose(skeleton, animation, time));
}

std::vector<math::Mat4> BlendAnimations(const Skeleton& skeleton,
                                        const Animation& animA, float timeA,
                                        const Animation& animB, float timeB, float weight) {
    std::vector<JointPose> poseA = SampleLocalPose(skeleton, animA, timeA);
    std::vector<JointPose> poseB = SampleLocalPose(skeleton, animB, timeB);
    return PaletteFromLocalPose(skeleton, BlendLocalPoses(poseA, poseB, weight));
}

} // namespace hf::anim
