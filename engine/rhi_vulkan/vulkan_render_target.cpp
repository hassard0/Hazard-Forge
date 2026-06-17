#include "rhi_vulkan/vulkan_render_target.h"
#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vk_common.h"

namespace hf::rhi::vk {

VulkanRenderTarget::VulkanRenderTarget(VulkanDevice& device, uint32_t width, uint32_t height,
                                       bool depthOnly, Format colorFormatReq)
    : device_(device), width_(width), height_(height), depthOnly_(depthOnly) {
    // Color image format: Format::Undefined keeps the historical swapchain format (so every
    // existing call site renders into the same image as before). An explicit format (e.g.
    // RGBA16_Float for the HDR bloom chain, Slice U) overrides it. Usage: render target + sampled
    // (so a later pass can read it). Skipped entirely for a depth-only shadow map.
    const VkFormat colorFormat = (colorFormatReq == Format::Undefined)
                                     ? device_.swapchainFormat()
                                     : ToVk(colorFormatReq);
    colorFormat_ = depthOnly ? VK_FORMAT_UNDEFINED : colorFormat;
    if (!depthOnly_) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = colorFormat;
        ici.extent = {width_, height_, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        // TRANSFER_SRC (Slice DD): lets a color RT be read back to host memory via ReadRenderTarget
        // (the capture-correctness proof reads the face-0 direct render). A usage flag only — it does
        // not affect rendering, so existing render targets / goldens are byte-for-byte unchanged.
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        Check(vmaCreateImage(device_.allocator(), &ici, &aci, &colorImage_, &colorAlloc_, nullptr),
              "vmaCreateImage(rt color)");

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = colorImage_;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = colorFormat;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        Check(vkCreateImageView(device_.device(), &vci, nullptr, &colorView_),
              "vkCreateImageView(rt color)");
    }

    // Depth image: D32, the RT owns it (matches the depth format the pipelines expect).
    {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_D32_SFLOAT;
        ici.extent = {width_, height_, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        // Shadow map needs SAMPLED so the lit pass can read it via the frame set.
        ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                    (depthOnly_ ? VK_IMAGE_USAGE_SAMPLED_BIT : 0);
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        Check(vmaCreateImage(device_.allocator(), &ici, &aci, &depthImage_, &depthAlloc_, nullptr),
              "vmaCreateImage(rt depth)");

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = depthImage_;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_D32_SFLOAT;
        vci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        Check(vkCreateImageView(device_.device(), &vci, nullptr, &depthView_),
              "vkCreateImageView(rt depth)");
    }

    // Depth-only shadow map: no color image and no per-RT descriptor set. The per-frame set
    // (set 0) samples depthView() directly via VulkanDevice::SetShadowMap.
    if (depthOnly_) return;

    // Slice DW/DX seam-subtlety (additive): an INTEGER color format (R32_Uint visibility buffer) is
    // bound as a SAMPLED uint texture and TEXEL-FETCHED (OpImageFetch, no sampler) by the DX deferred
    // resolve pass — never SAMPLED-as-float. DW gated the per-RT material set OFF entirely (set_ stayed
    // NULL) because it only ReadRenderTarget'd the integer RT; DX needs to BindTexture(*this) so the
    // resolve frag can texel-fetch it. The material set's binding 0 is a SEPARATE SAMPLED_IMAGE (NOT a
    // combined image+sampler) and binding 1 is a SEPARATE SAMPLER, so writing the integer image at
    // binding 0 alongside the default sampler at binding 1 is VALIDATION-CLEAN: the only hazard DW
    // avoided is a shader that COMBINES a sampler with an integer image, and the resolve frag does
    // exactly the opposite (a bare integer texelFetch — the sampler descriptor is never used). We
    // therefore allocate + populate the set exactly like the float path below (binding 0 = this image,
    // 1 = default sampler, 3/4 = default normal/sampler), with NO float-only assumptions. Existing
    // float RTs are byte-for-byte unchanged (integerColor is false for every one of them). VERIFIED
    // validation-clean under the Khronos layer (the --visresolve-shot gate). NO new RHI symbol.
    const bool integerColor = (colorFormatReq == Format::R32_Uint);
    (void)integerColor;

    // Descriptor set on the material set layout (set 1) so the post pass (and the DX resolve) can
    // BindTexture(*this). Same two-write pattern as VulkanTexture: binding 0 sampled image, binding 1
    // sampler. For an integer image the sampler at binding 1 is a harmless unused descriptor (the
    // resolve frag texel-fetches binding 0 only).
    VkDescriptorSetLayout layout = device_.materialSetLayout();
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = device_.descriptorPool();
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &layout;
    Check(vkAllocateDescriptorSets(device_.device(), &dai, &set_),
          "vkAllocateDescriptorSets(rt)");

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = colorView_;
    // The set is read while the color image is in SHADER_READ_ONLY_OPTIMAL (after the RT pass).
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = device_.defaultSampler();

    // Material set layout now has 4 bindings (base 0/1 + normal-map 3/4). The post pass only samples
    // the color image (binding 0/1), but populate binding 3/4 with the device's flat default normal
    // so the set is fully written (no incomplete-descriptor validation warnings when bound).
    VkDescriptorImageInfo normalImageInfo{};
    normalImageInfo.imageView = device_.defaultNormalView();
    normalImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo normalSamplerInfo{};
    normalSamplerInfo.sampler = device_.defaultSampler();

    VkWriteDescriptorSet writes[4]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[0].pImageInfo = &imageInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[1].pImageInfo = &samplerInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = set_;
    writes[2].dstBinding = 3;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[2].pImageInfo = &normalImageInfo;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = set_;
    writes[3].dstBinding = 4;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[3].pImageInfo = &normalSamplerInfo;

    vkUpdateDescriptorSets(device_.device(), 4, writes, 0, nullptr);
}

void VulkanRenderTarget::attachSecondaryColor(VkImageView secondView) {
    if (secondaryView_ == secondView) return;  // already pointed there
    secondaryView_ = secondView;
    VkDescriptorImageInfo info{};
    info.imageView = secondView;
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = set_;
    w.dstBinding = 3;  // the material set's second sampled-image slot (binding 3/4)
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w.pImageInfo = &info;
    vkUpdateDescriptorSets(device_.device(), 1, &w, 0, nullptr);
}

VkDescriptorSet VulkanRenderTarget::environmentSet() {
    // Slice AK — lazily allocate a set on the dedicated environment layout pointing at this RT's color
    // image (binding 11) + the env sampler (binding 12), so a baked probe atlas RT binds at the env
    // slot just like an HDR env texture. Reuses the existing env set layout + sampler — no new layout.
    if (environmentSet_ != VK_NULL_HANDLE) return environmentSet_;
    VkDescriptorSetLayout layout = device_.environmentSetLayout();
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = device_.descriptorPool();
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &layout;
    Check(vkAllocateDescriptorSets(device_.device(), &dai, &environmentSet_),
          "vkAllocateDescriptorSets(rt env)");

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = colorView_;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = device_.environmentSampler();

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = environmentSet_;
    writes[0].dstBinding = 11;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[0].pImageInfo = &imageInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = environmentSet_;
    writes[1].dstBinding = 12;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[1].pImageInfo = &samplerInfo;
    vkUpdateDescriptorSets(device_.device(), 2, writes, 0, nullptr);
    return environmentSet_;
}

VulkanRenderTarget::~VulkanRenderTarget() {
    if (environmentSet_)
        vkFreeDescriptorSets(device_.device(), device_.descriptorPool(), 1, &environmentSet_);
    if (set_) vkFreeDescriptorSets(device_.device(), device_.descriptorPool(), 1, &set_);
    if (colorView_) vkDestroyImageView(device_.device(), colorView_, nullptr);
    if (depthView_) vkDestroyImageView(device_.device(), depthView_, nullptr);
    if (colorImage_) vmaDestroyImage(device_.allocator(), colorImage_, colorAlloc_);
    if (depthImage_) vmaDestroyImage(device_.allocator(), depthImage_, depthAlloc_);
}

} // namespace hf::rhi::vk
