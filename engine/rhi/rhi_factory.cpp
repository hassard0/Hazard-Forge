#include "rhi/rhi_factory.h"
#include "rhi_vulkan/vulkan_device.h"
#include <stdexcept>

namespace hf::rhi {

std::unique_ptr<IRHIDevice> CreateDevice(Backend backend, hf::hal::Window& window) {
    switch (backend) {
        case Backend::Vulkan:
            return std::make_unique<vk::VulkanDevice>(window);
        default:
            throw std::runtime_error("Unsupported RHI backend");
    }
}

} // namespace hf::rhi
