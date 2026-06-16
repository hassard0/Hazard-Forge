#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <span>
#include "rhi/rhi.h"

namespace hf::rhi::vk {

class VulkanDevice;

// Slice BZ — a BINDLESS texture set: ONE descriptor set (the device's bindless set layout, set 4) whose
// binding 0 is a runtime/partially-bound sampled-image ARRAY filled with every scene texture's view IN
// ORDER (index i -> textures[i]) and whose binding 1 is a shared LINEAR/REPEAT sampler (identical to the
// device default sampler the per-material bound path uses, so the SAME texel is fetched -> byte-identical
// render). Allocated with VARIABLE_DESCRIPTOR_COUNT (the array sized to the texture count) and written
// ONCE here; the command buffer binds it ONCE via BindBindlessTextures. lit_bindless.frag samples
// gTextures[NonUniformResourceIndex(texIndex)] where texIndex is the per-draw push constant.
class VulkanBindlessTextureSet final : public IBindlessTextureSet {
public:
    VulkanBindlessTextureSet(VulkanDevice& device, std::span<ITexture* const> textures);
    ~VulkanBindlessTextureSet() override;

    VkDescriptorSet descriptorSet() const { return set_; }
    uint32_t count() const { return count_; }

private:
    VulkanDevice& device_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;  // dedicated pool (UPDATE_AFTER_BIND-capable) for this set
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    uint32_t count_ = 0;
};

} // namespace hf::rhi::vk
