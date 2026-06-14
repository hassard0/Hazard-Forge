#pragma once
#include "math/math.h"
namespace hf::scene {
struct Transform {
    math::Vec3 position{0,0,0};
    math::Vec3 eulerRadians{0,0,0};  // x=pitch, y=yaw, z=roll
    math::Vec3 scale{1,1,1};
    math::Mat4 Matrix() const {
        using namespace math;
        Mat4 r = Mat4::RotateZ(eulerRadians.z) * Mat4::RotateY(eulerRadians.y) * Mat4::RotateX(eulerRadians.x);
        return Mat4::Translate(position) * r * Mat4::Scale(scale);
    }
};
} // namespace
