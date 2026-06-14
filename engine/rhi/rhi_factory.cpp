#include "rhi/rhi_factory.h"
#include "rhi_vulkan/vulkan_device.h"
#include <stdexcept>

#ifdef __APPLE__
namespace hf::rhi::mtl {
// Defined in engine/rhi_metal/metal_device_windowed.mm (Obj-C++). Forward-declared so this plain
// .cpp never includes Metal/QuartzCore headers — keeps the RHI seam free of Obj-C types.
std::unique_ptr<IRHIDevice> CreateMetalDevice(hf::hal::Window& window);
} // namespace hf::rhi::mtl
#endif

namespace hf::rhi {

std::unique_ptr<IRHIDevice> CreateDevice(Backend backend, hf::hal::Window& window) {
    switch (backend) {
        case Backend::Vulkan:
            return std::make_unique<vk::VulkanDevice>(window);
        case Backend::Metal:
#ifdef __APPLE__
            return mtl::CreateMetalDevice(window);
#else
            throw std::runtime_error("Metal backend not available on this platform");
#endif
        default:
            throw std::runtime_error("Unsupported RHI backend");
    }
}

} // namespace hf::rhi
