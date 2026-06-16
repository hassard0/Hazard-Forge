#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"
#include "rhi_vulkan/vulkan_sampled.h"

namespace hf::rhi::vk {

class VulkanDevice;

// Slice DD — a sampleable CUBEMAP render target: ONE color image with 6 array layers + the
// VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT flag (so it can be viewed as a cube), `size`x`size` per face,
// usage COLOR_ATTACHMENT | SAMPLED | TRANSFER_SRC (the last for the per-face readback that drives the
// capture-correctness proof). The probe capture renders into each face through a SINGLE-LAYER 2D
// attachment view (faceView(i), layer i) with FaceView/FaceProj; the reflection pass samples the
// whole cube through cubeView() bound at the environment slot (set 3, binding 11/12) via
// BindCubemapProbe. A SHARED single-layer 2D depth image is re-cleared per face (the faces never read
// each other's depth). Tracks the color image's layout for correct capture-write -> sample-read
// transitions. Mirrors VulkanRenderTarget; the cube-specific bits (array image, per-face views, cube
// view) live ONLY here in the backend dir — no backend symbol leaks above the seam.
class VulkanCubemapTarget final : public ICubemapTarget, public ISampledVk {
public:
    VulkanCubemapTarget(VulkanDevice& device, uint32_t size, Format colorFormat);
    ~VulkanCubemapTarget() override;

    uint32_t size() const override { return size_; }

    VkImage colorImage() const { return colorImage_; }
    VkImageView faceView(uint32_t face) const { return faceViews_[face]; }
    VkImageView cubeView() const { return cubeView_; }
    VkImage depthImage() const { return depthImage_; }
    VkImageView depthView() const { return depthView_; }
    VkFormat colorVkFormat() const { return colorFormat_; }

    // The cube binds at the environment slot (set 3, binding 11 = cube view, 12 = env sampler) so
    // BindCubemapProbe routes it through the existing env set layout exactly like a probe atlas RT.
    VkDescriptorSet environmentSet() const { return environmentSet_; }
    // ISampledVk: a cube can't be sampled by a plain Texture2D material set, so the material set is
    // unused; expose the cube view for completeness (bindless never sources a cube).
    VkDescriptorSet vkDescriptorSet() const override { return VK_NULL_HANDLE; }
    VkImageView vkImageView() const override { return cubeView_; }

    VkImageLayout colorLayout() const { return colorLayout_; }
    void setColorLayout(VkImageLayout l) { colorLayout_ = l; }

private:
    VulkanDevice& device_;
    uint32_t size_ = 0;

    VkFormat        colorFormat_ = VK_FORMAT_UNDEFINED;
    VkImage         colorImage_  = VK_NULL_HANDLE;
    VmaAllocation   colorAlloc_  = VK_NULL_HANDLE;
    VkImageView     faceViews_[6]{};                  // per-face single-layer 2D attachment views
    VkImageView     cubeView_    = VK_NULL_HANDLE;    // cube sampling view (VK_IMAGE_VIEW_TYPE_CUBE)
    VkImageLayout   colorLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage         depthImage_ = VK_NULL_HANDLE;
    VmaAllocation   depthAlloc_ = VK_NULL_HANDLE;
    VkImageView     depthView_  = VK_NULL_HANDLE;

    VkDescriptorSet environmentSet_ = VK_NULL_HANDLE; // set 3 (cube at env slot, binding 11/12)
};

} // namespace hf::rhi::vk
