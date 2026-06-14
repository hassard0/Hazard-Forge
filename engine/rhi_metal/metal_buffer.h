#pragma once
#import <Metal/Metal.h>
#include "rhi/rhi.h"

namespace hf::rhi::mtl {

// Shared-storage MTLBuffer (CPU+GPU visible on Apple Silicon). Used for vertex/index/uniform.
class MetalBuffer final : public IBuffer {
public:
    MetalBuffer(id<MTLDevice> device, const BufferDesc& desc);
    ~MetalBuffer() override;

    id<MTLBuffer> handle() const { return buffer_; }
    uint64_t size() const { return size_; }

private:
    id<MTLBuffer> buffer_ = nil;
    uint64_t      size_   = 0;
};

} // namespace hf::rhi::mtl
