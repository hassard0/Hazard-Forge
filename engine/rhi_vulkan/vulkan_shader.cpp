#include "rhi_vulkan/vulkan_shader.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanShaderModule::VulkanShaderModule(VkDevice device, std::span<const uint32_t> spirv)
    : device_(device) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = spirv.size() * sizeof(uint32_t);
    ci.pCode = spirv.data();
    Check(vkCreateShaderModule(device_, &ci, nullptr, &module_), "vkCreateShaderModule");
}

VulkanShaderModule::~VulkanShaderModule() {
    if (module_) vkDestroyShaderModule(device_, module_, nullptr);
}

} // namespace hf::rhi::vk
