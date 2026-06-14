#pragma once
#include <memory>
#include <cstdint>
#include "rhi/rhi.h"
namespace hf::scene {
class Mesh {
public:
    Mesh(std::unique_ptr<rhi::IBuffer> v, std::unique_ptr<rhi::IBuffer> i, uint32_t indexCount)
        : vertices_(std::move(v)), indices_(std::move(i)), indexCount_(indexCount) {}
    rhi::IBuffer& vertices() const { return *vertices_; }
    rhi::IBuffer& indices() const { return *indices_; }
    uint32_t indexCount() const { return indexCount_; }
    static Mesh Cube(rhi::IRHIDevice& device);
    static Mesh Plane(rhi::IRHIDevice& device);
private:
    std::unique_ptr<rhi::IBuffer> vertices_, indices_;
    uint32_t indexCount_;
};
} // namespace
