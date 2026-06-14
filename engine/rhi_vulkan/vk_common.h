#pragma once
#include <vulkan/vulkan.h>
#include "rhi/rhi.h"
#include <stdexcept>
#include <string>

namespace hf::rhi::vk {

inline void Check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) {
        throw std::runtime_error(std::string("Vulkan error in ") + what +
                                 " (VkResult=" + std::to_string((int)r) + ")");
    }
}

inline VkFormat ToVk(Format f) {
    switch (f) {
        case Format::RGBA8_UNorm: return VK_FORMAT_R8G8B8A8_UNORM;
        case Format::BGRA8_UNorm: return VK_FORMAT_B8G8R8A8_UNORM;
        case Format::RG32_Float:  return VK_FORMAT_R32G32_SFLOAT;
        case Format::RGB32_Float: return VK_FORMAT_R32G32B32_SFLOAT;
        default:                  return VK_FORMAT_UNDEFINED;
    }
}

inline Format FromVk(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: return Format::RGBA8_UNorm;
        case VK_FORMAT_B8G8R8A8_UNORM: return Format::BGRA8_UNorm;
        default:                       return Format::Undefined;
    }
}

} // namespace hf::rhi::vk
