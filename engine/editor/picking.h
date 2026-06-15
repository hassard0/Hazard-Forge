#pragma once
// Slice AB — editor ray-cast picking. PURE C++ (engine/math + runtime::Camera only); NO rhi/SDL/
// backend symbols, so it compiles into hf_core (ASan-scoped, unit-tested) and is window-free.
//
// The viewport hands a screen point (in normalized device coordinates, x,y in [-1,1]) and a camera;
// ScreenRayThroughCamera unprojects it into a world-space ray via the inverse view-proj. PickNearest
// then slab-tests that ray against a span of world AABBs and returns the nearest hit. Both are fully
// deterministic and testable without a GUI (see tests/editor_test.cpp + --pick-test).

#include <span>

#include "math/math.h"
#include "runtime/camera.h"

namespace hf::editor {

using math::Aabb;
using math::Ray;

// Unproject a screen point in NDC (ndcX, ndcY in [-1,1], +x right, +y up in NDC) through `cam` to a
// world-space ray. Uses (cam.Proj()*cam.View()).Inverse(): the near plane point (clip z=0, Vulkan
// depth range) and far plane point (clip z=1) are unprojected to world space and the ray goes from
// near to far. Self-consistent with cam.ViewProj() (a world point projected by VP unprojects back
// through VP-inverse), so it lands on objects the camera actually sees.
Ray ScreenRayThroughCamera(const runtime::Camera& cam, float ndcX, float ndcY);

// Convert an absolute cursor position in FRAMEBUFFER PIXELS (origin top-left, +x right, +y DOWN —
// the windowing convention) to normalized device coordinates (ndcX,ndcY in [-1,1], +x right, +y UP —
// what ScreenRayThroughCamera expects). The Y axis is flipped. Pure math, shared between the live
// viewport (HAL cursor px) and the unit test so both exercise the identical mapping.
struct Ndc { float x, y; };
inline Ndc PixelToNdc(float px, float py, float width, float height) {
    float nx = (width  > 0.0f) ? (px / width)  * 2.0f - 1.0f : 0.0f;
    float ny = (height > 0.0f) ? (py / height) * 2.0f - 1.0f : 0.0f;
    return Ndc{nx, -ny};  // flip Y: pixel +y is down, NDC +y is up
}

// One pickable object: its world-space AABB. (The caller fits this from the object's posed mesh
// bounds; the picker stays geometry-agnostic.)
struct PickAabb {
    Aabb box;
};

// Result of a pick: `index` into the span (or -1 on a miss), and the ray parameter `t` of the hit.
struct PickResult {
    int index = -1;
    float t = 0.0f;
};

// Nearest ray/AABB hit among `objects` (slab test). Returns index<0 if the ray misses them all.
PickResult PickNearest(const Ray& ray, std::span<const PickAabb> objects);

}  // namespace hf::editor
