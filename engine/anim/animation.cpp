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

std::vector<math::Mat4> SampleAnimation(const Skeleton& skeleton, const Animation& animation,
                                        float time) {
    const size_t jointCount = skeleton.joints.size();

    // Start from each joint's REST local TRS, then let channels override components.
    std::vector<math::Vec3> locT(jointCount);
    std::vector<math::Quat> locR(jointCount);
    std::vector<math::Vec3> locS(jointCount);
    for (size_t j = 0; j < jointCount; ++j) {
        locT[j] = skeleton.joints[j].t;
        locR[j] = skeleton.joints[j].r;
        locS[j] = skeleton.joints[j].s;
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
                locT[ch.jointIndex] = math::Vec3{
                    a[0] + (b[0] - a[0]) * frac,
                    a[1] + (b[1] - a[1]) * frac,
                    a[2] + (b[2] - a[2]) * frac};
                break;
            }
            case Channel::Path::Scale: {
                const float* a = &ch.values[k.i0 * 3];
                const float* b = &ch.values[k.i1 * 3];
                locS[ch.jointIndex] = math::Vec3{
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
                locR[ch.jointIndex] = step ? qa : math::Slerp(qa, qb, frac);
                break;
            }
        }
    }

    // Single forward pass: joints are topologically sorted (parent before child), so the parent's
    // global transform is already computed when we reach a child.
    std::vector<math::Mat4> global(jointCount);
    for (size_t j = 0; j < jointCount; ++j) {
        math::Mat4 local = math::FromTRS(locT[j], locR[j], locS[j]);
        int parent = skeleton.joints[j].parent;
        global[j] = (parent >= 0) ? (global[parent] * local) : local;
    }

    // palette[j] = global[j] * inverseBind[j].
    std::vector<math::Mat4> palette(jointCount);
    for (size_t j = 0; j < jointCount; ++j)
        palette[j] = global[j] * skeleton.joints[j].inverseBind;
    return palette;
}

} // namespace hf::anim
