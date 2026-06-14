#pragma once
#include "rhi/rhi.h"
namespace hf::scene {
// pos(12) + color(12) + uv(8) + normal(12) + tangent(12) = stride 56.
// tangent is the world-space-able object-space tangent (dP/du) at location 4; the lit vertex
// shader transforms it to world space and the fragment shader builds the TBN basis from it
// (B = cross(N, T) * handedness; handedness is +1 for all engine-authored meshes).
struct Vertex { float pos[3]; float color[3]; float uv[2]; float normal[3]; float tangent[3]; };
inline rhi::VertexLayout MeshVertexLayout() {
    rhi::VertexLayout l;
    l.stride = sizeof(Vertex);
    l.attributes = {
        {0, rhi::Format::RGB32_Float, 0},
        {1, rhi::Format::RGB32_Float, 12},
        {2, rhi::Format::RG32_Float,  24},
        {3, rhi::Format::RGB32_Float, 32},
        {4, rhi::Format::RGB32_Float, 44},
    };
    return l;
}
} // namespace
