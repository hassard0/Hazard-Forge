#include "rhi_metal/metal_buffer.h"
#include "rhi_metal/metal_common.h"
#include <cstring>

namespace hf::rhi::mtl {

MetalBuffer::MetalBuffer(id<MTLDevice> device, const BufferDesc& desc) : size_(desc.size) {
    // Shared storage: directly host-writable on Apple Silicon (unified memory). BufferUsage
    // (Vertex/Index/Uniform) doesn't change the MTLBuffer — Metal binds by call site, not usage.
    if (desc.initialData) {
        buffer_ = [device newBufferWithBytes:desc.initialData
                                      length:desc.size
                                     options:MTLResourceStorageModeShared];
    } else {
        buffer_ = [device newBufferWithLength:desc.size
                                      options:MTLResourceStorageModeShared];
    }
    if (!buffer_) Fail("newBuffer failed");
}

MetalBuffer::~MetalBuffer() {
    // ARC releases buffer_.
}

} // namespace hf::rhi::mtl
