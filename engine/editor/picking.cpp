// Slice AB — editor ray-cast picking implementation. Pure C++ (math + runtime::Camera).
#include "editor/picking.h"

namespace hf::editor {

using math::Mat4;
using math::Vec3;

Ray ScreenRayThroughCamera(const runtime::Camera& cam, float ndcX, float ndcY) {
    Mat4 invVP = cam.ViewProj().Inverse();
    // Vulkan clip depth range is [0,1]: z=0 is the near plane, z=1 the far plane. Unproject both,
    // applying the perspective divide, then build the ray from the near point through the far point.
    float wNear = 0.0f, wFar = 0.0f;
    Vec3 nearW = math::MulPointDivide(invVP, Vec3{ndcX, ndcY, 0.0f}, wNear);
    Vec3 farW  = math::MulPointDivide(invVP, Vec3{ndcX, ndcY, 1.0f}, wFar);
    return math::MakeRay(nearW, farW);
}

PickResult PickNearest(const Ray& ray, std::span<const PickAabb> objects) {
    PickResult best;
    best.index = -1;
    best.t = 1e30f;
    for (int i = 0; i < (int)objects.size(); ++i) {
        float t = 0.0f;
        if (math::RayAabb(ray, objects[i].box, t) && t < best.t) {
            best.t = t;
            best.index = i;
        }
    }
    if (best.index < 0) best.t = 0.0f;
    return best;
}

}  // namespace hf::editor
