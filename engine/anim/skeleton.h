#pragma once
#include "math/math.h"
#include <vector>

namespace hf::anim {

// One joint of a skeleton. `parent` is the index of the parent joint in the same Skeleton::joints
// array, or -1 for a root. `inverseBind` maps a vertex from model space into this joint's bind-pose
// local space (glTF's inverse-bind-matrix accessor). `t/r/s` is the joint's REST local transform
// (the node's TRS at bind time); an animation channel overrides whichever component it targets.
//
// Joints are stored in a TOPOLOGICALLY SORTED order (every parent appears before its children), so
// SampleAnimation can compute global transforms in a single forward pass.
struct Joint {
    int       parent = -1;
    math::Mat4 inverseBind = math::Mat4::Identity();
    math::Vec3 t{0, 0, 0};
    math::Quat r{0, 0, 0, 1};
    math::Vec3 s{1, 1, 1};
};

struct Skeleton {
    std::vector<Joint> joints;
};

} // namespace hf::anim
