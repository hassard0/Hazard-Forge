#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"
#include "rhi_vulkan/vulkan_sampled.h"

namespace hf::rhi::vk {

class VulkanDevice;

// An offscreen color image (COLOR_ATTACHMENT | SAMPLED, swapchain format) plus its own depth
// image (D32). The color image has a material descriptor set so a later pass can sample it
// (ISampledVk). Tracks the color image's current layout for correct transitions.
//
// In depthOnly mode (shadow map): no color image; the depth image gets usage
// DEPTH_STENCIL_ATTACHMENT | SAMPLED and is exposed via depthView(). The frame set (set 0)
// samples it directly, so no per-RT descriptor set is allocated.
class VulkanRenderTarget final : public IRenderTarget, public ISampledVk {
public:
    // colorFormat == Format::Undefined selects the swapchain color format (the historical
    // behavior, byte-for-byte). Format::RGBA16_Float makes a half-float HDR color image (Slice U).
    VulkanRenderTarget(VulkanDevice& device, uint32_t width, uint32_t height,
                       bool depthOnly = false, Format colorFormat = Format::Undefined);
    ~VulkanRenderTarget() override;

    uint32_t width() const override { return width_; }
    uint32_t height() const override { return height_; }

    bool depthOnly() const { return depthOnly_; }

    VkImage colorImage() const { return colorImage_; }
    VkImageView colorView() const { return colorView_; }
    VkImageView depthView() const { return depthView_; }
    VkImage depthImage() const { return depthImage_; }

    // Attachment formats (for Slice AU secondary-command-buffer dynamic-rendering inheritance).
    // colorVkFormat() is UNDEFINED for a depth-only RT (no color image); depth is always D32.
    VkFormat colorVkFormat() const { return colorFormat_; }
    VkFormat depthVkFormat() const { return VK_FORMAT_D32_SFLOAT; }

    VkDescriptorSet descriptorSet() const { return set_; }
    VkDescriptorSet vkDescriptorSet() const override { return set_; }
    VkImageView vkImageView() const override { return colorView_; }  // Slice BZ: bindless array source

    // Slice AK — a descriptor set on the dedicated ENVIRONMENT set layout (set 3, binding 11 = this
    // RT's color view, binding 12 = the env sampler) so a baked probe atlas RT can be bound at the
    // env slot via BindReflectionProbe, exactly like an HDR env VulkanTexture. Allocated lazily the
    // first time it's requested (most RTs never need it). Not valid for depth-only RTs.
    VkDescriptorSet environmentSet();

    // Repoint this RT's material-set second slot (binding 3) at another image view, so a fullscreen
    // pass can sample THIS RT (binding 0) and `secondView` (binding 3) from the one set. Used by the
    // bloom composite to sample the HDR scene + the bloom result together (Slice U). Cached: a repeat
    // with the same view issues no descriptor update.
    void attachSecondaryColor(VkImageView secondView);

    // Layout bookkeeping for the color image (driven by Begin/EndRenderTargetFrame).
    VkImageLayout colorLayout() const { return colorLayout_; }
    void setColorLayout(VkImageLayout l) { colorLayout_ = l; }

    // Layout bookkeeping for the depth image (driven by Begin/EndShadowPass in depthOnly mode).
    VkImageLayout depthLayout() const { return depthLayout_; }
    void setDepthLayout(VkImageLayout l) { depthLayout_ = l; }

private:
    VulkanDevice& device_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    bool     depthOnly_ = false;

    VkFormat        colorFormat_ = VK_FORMAT_UNDEFINED;  // color attachment format (UNDEFINED if depthOnly)
    VkImage         colorImage_ = VK_NULL_HANDLE;
    VmaAllocation   colorAlloc_ = VK_NULL_HANDLE;
    VkImageView     colorView_  = VK_NULL_HANDLE;
    VkImageLayout   colorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage         depthImage_ = VK_NULL_HANDLE;
    VmaAllocation   depthAlloc_ = VK_NULL_HANDLE;
    VkImageView     depthView_  = VK_NULL_HANDLE;
    VkImageLayout   depthLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkImageView     secondaryView_ = VK_NULL_HANDLE;  // current binding-3 view (cache for attachSecondaryColor)
    VkDescriptorSet environmentSet_ = VK_NULL_HANDLE; // lazy set 3 (probe atlas at env slot), Slice AK
};

} // namespace hf::rhi::vk
