#pragma once
#include <vulkan/vulkan.h>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanShaderModule final : public IShaderModule {
public:
    VulkanShaderModule(VkDevice device, std::span<const uint32_t> spirv);
    ~VulkanShaderModule() override;
    VkShaderModule handle() const { return module_; }

private:
    VkDevice device_;
    VkShaderModule module_ = VK_NULL_HANDLE;
};

} // namespace hf::rhi::vk
