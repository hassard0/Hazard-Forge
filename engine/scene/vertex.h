#pragma once
#include "rhi/rhi.h"
namespace hf::scene {
struct Vertex { float pos[3]; float color[3]; float uv[2]; float normal[3]; };
inline rhi::VertexLayout MeshVertexLayout() {
    rhi::VertexLayout l;
    l.stride = sizeof(Vertex);
    l.attributes = {
        {0, rhi::Format::RGB32_Float, 0},
        {1, rhi::Format::RGB32_Float, 12},
        {2, rhi::Format::RG32_Float,  24},
        {3, rhi::Format::RGB32_Float, 32},
    };
    return l;
}
} // namespace
