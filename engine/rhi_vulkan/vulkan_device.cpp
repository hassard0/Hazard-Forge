#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vk_common.h"
#include "rhi_vulkan/vulkan_command_buffer.h"
#include "rhi_vulkan/vulkan_shader.h"
#include "rhi_vulkan/vulkan_pipeline.h"
#include "rhi_vulkan/vulkan_compute_pipeline.h"
#include "rhi_vulkan/vulkan_buffer.h"
#include "rhi_vulkan/vulkan_texture.h"
#include "rhi_vulkan/vulkan_render_target.h"
#include "rhi_vulkan/vulkan_pbr_material.h"
#include <cstring>

#define VMA_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace hf::rhi::vk {

namespace {
void TransitionImage(VkCommandBuffer cmd, VkImage image,
                     VkImageLayout oldLayout, VkImageLayout newLayout,
                     VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                     VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                     VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = srcStage;  b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStage;  b.dstAccessMask = dstAccess;
    b.oldLayout = oldLayout;    b.newLayout = newLayout;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &di);
}
} // namespace

VulkanDevice::VulkanDevice(hf::hal::Window& window) : window_(window) {
    // --- Instance ---
    vkb::InstanceBuilder ib;
    ib.set_app_name("Hazard Forge")
      .require_api_version(1, 3, 0)
#ifndef NDEBUG
      .request_validation_layers(true)
      .use_default_debug_messenger()
#endif
      ;
    for (const char* ext : window_.RequiredVulkanInstanceExtensions()) {
        ib.enable_extension(ext);
    }
    auto instRet = ib.build();
    if (!instRet) throw std::runtime_error("Instance build failed: " + instRet.error().message());
    vkbInstance_ = instRet.value();
    instance_ = vkbInstance_.instance;

    // --- Surface ---
    surface_ = window_.CreateVulkanSurface(instance_);

    // --- Physical device (require dynamic rendering + sync2) ---
    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;

    vkb::PhysicalDeviceSelector selector{vkbInstance_};
    auto physRet = selector
        .set_surface(surface_)
        .set_minimum_version(1, 3)
        .set_required_features_13(f13)
        .select();
    if (!physRet) throw std::runtime_error("GPU select failed: " + physRet.error().message());

    // VK_KHR_push_descriptor lets the compute path bind its storage buffer(s) inline via
    // vkCmdPushDescriptorSetKHR — no descriptor pool/set lifetime to manage per frame.
    vkb::PhysicalDevice physForDevice = physRet.value();
    physForDevice.enable_extension_if_present(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

    vkb::DeviceBuilder db{physForDevice};
    auto devRet = db.build();
    if (!devRet) throw std::runtime_error("Device build failed: " + devRet.error().message());
    vkbDevice_ = devRet.value();
    device_ = vkbDevice_.device;
    physical_ = physRet.value().physical_device;

    graphicsQueue_ = vkbDevice_.get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily_ = vkbDevice_.get_queue_index(vkb::QueueType::graphics).value();

    // Load the push-descriptor entry point (used by the compute path to bind storage buffers).
    vkCmdPushDescriptorSetKHR_ = reinterpret_cast<PFN_vkCmdPushDescriptorSetKHR>(
        vkGetDeviceProcAddr(device_, "vkCmdPushDescriptorSetKHR"));

    // --- VMA ---
    VmaAllocatorCreateInfo aci{};
    aci.physicalDevice = physical_;
    aci.device = device_;
    aci.instance = instance_;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    Check(vmaCreateAllocator(&aci, &allocator_), "vmaCreateAllocator");

    // --- Swapchain (owns the depth attachment; needs the VMA allocator) ---
    swapchain_ = std::make_unique<VulkanSwapchain>(
        device_, vkbDevice_, allocator_,
        window_.FramebufferWidth(), window_.FramebufferHeight());

    CreateTextureResources();
    CreateSyncObjects();
    recorder_ = std::make_unique<VulkanCommandBuffer>(*this);

    // Dedicated command pool/buffer/fence + recorder for the offscreen render-target pass.
    {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = graphicsQueueFamily_;
        Check(vkCreateCommandPool(device_, &pci, nullptr, &rtPool_), "vkCreateCommandPool(rt)");

        VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbi.commandPool = rtPool_;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        Check(vkAllocateCommandBuffers(device_, &cbi, &rtCmd_), "vkAllocateCommandBuffers(rt)");

        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};  // unsignaled
        Check(vkCreateFence(device_, &fi, nullptr, &rtFence_), "vkCreateFence(rt)");

        rtRecorder_ = std::make_unique<VulkanCommandBuffer>(*this);
    }
}

VulkanDevice::~VulkanDevice() {
    if (device_) vkDeviceWaitIdle(device_);
    rtRecorder_.reset();
    if (rtFence_) vkDestroyFence(device_, rtFence_, nullptr);
    if (rtPool_) vkDestroyCommandPool(device_, rtPool_, nullptr);  // frees rtCmd_
    recorder_.reset();
    DestroySyncObjects();
    swapchain_.reset();
    DestroyTextureResources();
    if (allocator_) vmaDestroyAllocator(allocator_);
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (device_) vkb::destroy_device(vkbDevice_);
    if (instance_) vkb::destroy_instance(vkbInstance_);
}

void VulkanDevice::CreateSyncObjects() {
    for (auto& fr : frames_) {
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = graphicsQueueFamily_;
        Check(vkCreateCommandPool(device_, &pci, nullptr, &fr.pool), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbi.commandPool = fr.pool;
        cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbi.commandBufferCount = 1;
        Check(vkAllocateCommandBuffers(device_, &cbi, &fr.cmd), "vkAllocateCommandBuffers");

        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        Check(vkCreateSemaphore(device_, &si, nullptr, &fr.imageAvailable), "sem");
        Check(vkCreateSemaphore(device_, &si, nullptr, &fr.renderFinished), "sem");

        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        Check(vkCreateFence(device_, &fi, nullptr, &fr.inFlight), "fence");
    }
}

void VulkanDevice::DestroySyncObjects() {
    for (auto& fr : frames_) {
        if (fr.inFlight) vkDestroyFence(device_, fr.inFlight, nullptr);
        if (fr.imageAvailable) vkDestroySemaphore(device_, fr.imageAvailable, nullptr);
        if (fr.renderFinished) vkDestroySemaphore(device_, fr.renderFinished, nullptr);
        if (fr.pool) vkDestroyCommandPool(device_, fr.pool, nullptr);
        fr = FrameSync{};
    }
}

void VulkanDevice::CreateTextureResources() {
    // Default sampler: linear filtering, repeat addressing.
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    Check(vkCreateSampler(device_, &sci, nullptr, &defaultSampler_), "vkCreateSampler");

    // Shadow sampler: linear, clamp-to-edge (so out-of-bounds UVs read the border depth, not
    // wrapped geometry). No comparison — the shader does the manual depth compare + PCF.
    VkSamplerCreateInfo ssci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ssci.magFilter = VK_FILTER_LINEAR;
    ssci.minFilter = VK_FILTER_LINEAR;
    ssci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ssci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ssci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    ssci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    Check(vkCreateSampler(device_, &ssci, nullptr, &shadowSampler_), "vkCreateSampler(shadow)");

    // --- Default flat normal map: a 1x1 RGBA8 image encoding the tangent-space normal (0,0,1) as
    // (128,128,255). Bound at material set binding 3 when a renderable has no normal map, so the lit
    // shader always samples a valid normal map (perturbation = identity). ---
    {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {1, 1, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        Check(vmaCreateImage(allocator_, &ici, &aci, &defaultNormalImage_, &defaultNormalAlloc_,
                             nullptr), "vmaCreateImage(defaultNormal)");
        const uint8_t flatNormal[4] = {128, 128, 255, 255};
        UploadToImage(defaultNormalImage_, 1, 1, flatNormal, sizeof(flatNormal));

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = defaultNormalImage_;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        Check(vkCreateImageView(device_, &vci, nullptr, &defaultNormalView_),
              "vkCreateImageView(defaultNormal)");
    }

    // Material set layout (used at set 1), fragment stage. (DXC emits Texture2D/SamplerState as two
    // separate descriptors.) Four bindings now that materials carry a normal map:
    //   binding 0 = base-color sampled image   (gTex,        set1 b0)
    //   binding 1 = base-color sampler         (gSmp,        set1 b1)
    //   binding 3 = normal-map sampled image   (gNormalMap,  set1 b3)
    //   binding 4 = normal-map sampler         (gNormalSmp,  set1 b4)
    // (Bindings 3/4 match the lit.frag HLSL [[vk::binding(3,1)]]/[[vk::binding(4,1)]] so spirv-cross
    // maps them to Metal texture(3)/sampler(4). Binding 2 is intentionally unused on Vulkan.)
    VkDescriptorSetLayoutBinding bindings[4]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[2].binding = 3;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[3].binding = 4;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 4;
    lci.pBindings = bindings;
    Check(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &texturedSetLayout_),
          "vkCreateDescriptorSetLayout");

    // --- Full-PBR material set layout (set 1 for the lit-PBR pipeline only). A SEPARATE, WIDER
    // layout so the existing 2-texture material set / lit pipeline stays byte-for-byte unchanged.
    // Ten bindings (five sampled images + five samplers), fragment stage. The binding numbers are
    // chosen so spirv-cross --msl-decoration-binding maps each to the flat fragment index the PBR
    // backend binds (base 0/1, normal 3/4, metalRough 5/6, emissive 7/8, occlusion 9/10):
    //   b0  base-color image            b1  base-color sampler
    //   b3  normal-map image            b4  normal-map sampler
    //   b5  metallic-roughness image    b6  metallic-roughness sampler
    //   b7  emissive image              b8  emissive sampler
    //   b9  occlusion image             b10 occlusion sampler
    // (Binding 2 stays unused, matching the existing material set's gap.)
    {
        const uint32_t imgBindings[5] = {0, 3, 5, 7, 9};
        const uint32_t smpBindings[5] = {1, 4, 6, 8, 10};
        VkDescriptorSetLayoutBinding pbrBindings[10]{};
        for (int k = 0; k < 5; ++k) {
            pbrBindings[k * 2].binding = imgBindings[k];
            pbrBindings[k * 2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            pbrBindings[k * 2].descriptorCount = 1;
            pbrBindings[k * 2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            pbrBindings[k * 2 + 1].binding = smpBindings[k];
            pbrBindings[k * 2 + 1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            pbrBindings[k * 2 + 1].descriptorCount = 1;
            pbrBindings[k * 2 + 1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo plci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        plci.bindingCount = 10;
        plci.pBindings = pbrBindings;
        Check(vkCreateDescriptorSetLayout(device_, &plci, nullptr, &pbrMaterialSetLayout_),
              "vkCreateDescriptorSetLayout(pbr)");
    }

    // Pool: allow individual set frees (VulkanTexture frees its set in its dtor). Includes
    // a UNIFORM_BUFFER capacity for the per-frame UBO sets (set 0).
    // SAMPLED_IMAGE/SAMPLER counts cover material sets (set 1) plus the per-frame sets' shadow-map
    // image + sampler bindings (set 0, bindings 1+2): +kFramesInFlight each.
    // Each material set now binds TWO sampled images + TWO samplers (base-color + normal map), so
    // double the per-set image/sampler capacity vs. the old single-texture material set.
    VkDescriptorPoolSize poolSizes[3]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    poolSizes[0].descriptorCount = 256 + kFramesInFlight;  // +full-PBR sets (5 images each)
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLER;
    poolSizes[1].descriptorCount = 256 + kFramesInFlight;  // +full-PBR sets (5 samplers each)
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[2].descriptorCount = kFramesInFlight + 16;  // margin
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets = 64;
    pci.poolSizeCount = 3;
    pci.pPoolSizes = poolSizes;
    Check(vkCreateDescriptorPool(device_, &pci, nullptr, &descriptorPool_),
          "vkCreateDescriptorPool");

    // --- Per-frame set layout (set 0): binding 0 = uniform buffer (vertex+fragment), binding 1 =
    // shadow-map sampled image (fragment), binding 2 = shadow sampler (fragment). The shadow
    // bindings are populated by SetShadowMap; the lit pipeline samples them for shadowing. ---
    VkDescriptorSetLayoutBinding frameBindings[3]{};
    frameBindings[0].binding = 0;
    frameBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    frameBindings[0].descriptorCount = 1;
    frameBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    frameBindings[1].binding = 1;
    frameBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    frameBindings[1].descriptorCount = 1;
    frameBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    frameBindings[2].binding = 2;
    frameBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    frameBindings[2].descriptorCount = 1;
    frameBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo fci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    fci.bindingCount = 3;
    fci.pBindings = frameBindings;
    Check(vkCreateDescriptorSetLayout(device_, &fci, nullptr, &frameSetLayout_),
          "vkCreateDescriptorSetLayout(frame)");

    // One host-visible MAPPED uniform buffer + descriptor set per frame-in-flight; each set
    // is pre-updated to point at its persistent buffer (SetFrameUniforms only writes memory).
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = kFrameUboSize;
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        // HOST_COHERENT is required (not preferred): guarantees host-visible mapped memory, so
        // uboMapped_ is always non-null and SetFrameUniforms needs no flush. Do not relax this
        // without also handling a possibly-null mapped pointer.
        aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VmaAllocationInfo info{};
        Check(vmaCreateBuffer(allocator_, &bci, &aci, &uboBuffer_[i], &uboAlloc_[i], &info),
              "vmaCreateBuffer(ubo)");
        uboMapped_[i] = info.pMappedData;

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = descriptorPool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &frameSetLayout_;
        Check(vkAllocateDescriptorSets(device_, &dai, &frameSet_[i]),
              "vkAllocateDescriptorSets(frame)");

        VkDescriptorBufferInfo bufInfo{uboBuffer_[i], 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = frameSet_[i];
        write.dstBinding = 0;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &bufInfo;
        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    }

    // --- Joint-palette set layout (set 2): binding 0 = uniform buffer (vertex stage), the
    // JointPalette { float4x4 joints[64]; } skinning matrix array. One host-visible mapped UBO +
    // descriptor set per frame-in-flight, pre-wired to its buffer (SetJointPalette only memcpys). ---
    VkDescriptorSetLayoutBinding jointBinding{};
    jointBinding.binding = 0;
    jointBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    jointBinding.descriptorCount = 1;
    jointBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo jci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    jci.bindingCount = 1;
    jci.pBindings = &jointBinding;
    Check(vkCreateDescriptorSetLayout(device_, &jci, nullptr, &jointSetLayout_),
          "vkCreateDescriptorSetLayout(joint)");

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = kJointPaletteSize;
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VmaAllocationInfo info{};
        Check(vmaCreateBuffer(allocator_, &bci, &aci, &jointBuffer_[i], &jointAlloc_[i], &info),
              "vmaCreateBuffer(joint)");
        jointMapped_[i] = info.pMappedData;

        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = descriptorPool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &jointSetLayout_;
        Check(vkAllocateDescriptorSets(device_, &dai, &jointSet_[i]),
              "vkAllocateDescriptorSets(joint)");

        VkDescriptorBufferInfo bufInfo{jointBuffer_[i], 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = jointSet_[i];
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.pBufferInfo = &bufInfo;
        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    }
}

VkDescriptorSet VulkanDevice::pbrMaterialSet(ITexture& base, ITexture& metalRough,
                                             ITexture& normalMap, ITexture& emissive,
                                             ITexture& occlusion) {
    auto it = pbrMaterials_.find(&base);
    if (it == pbrMaterials_.end()) {
        auto mat = std::make_unique<VulkanPbrMaterial>(*this, base, metalRough, normalMap,
                                                       emissive, occlusion);
        it = pbrMaterials_.emplace(&base, std::move(mat)).first;
    }
    return it->second->descriptorSet();
}

void VulkanDevice::DestroyTextureResources() {
    // Free the cached PBR material descriptor sets before the pool that owns them is destroyed.
    pbrMaterials_.clear();

    // Per-frame UBOs + layout. Frame sets are freed when the pool below is destroyed.
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (uboBuffer_[i]) vmaDestroyBuffer(allocator_, uboBuffer_[i], uboAlloc_[i]);
        uboBuffer_[i] = VK_NULL_HANDLE;
        uboAlloc_[i] = VK_NULL_HANDLE;
        uboMapped_[i] = nullptr;
        frameSet_[i] = VK_NULL_HANDLE;
    }
    if (frameSetLayout_) vkDestroyDescriptorSetLayout(device_, frameSetLayout_, nullptr);
    frameSetLayout_ = VK_NULL_HANDLE;

    // Joint-palette UBOs + layout. Joint sets are freed when the pool below is destroyed.
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (jointBuffer_[i]) vmaDestroyBuffer(allocator_, jointBuffer_[i], jointAlloc_[i]);
        jointBuffer_[i] = VK_NULL_HANDLE;
        jointAlloc_[i] = VK_NULL_HANDLE;
        jointMapped_[i] = nullptr;
        jointSet_[i] = VK_NULL_HANDLE;
    }
    if (jointSetLayout_) vkDestroyDescriptorSetLayout(device_, jointSetLayout_, nullptr);
    jointSetLayout_ = VK_NULL_HANDLE;

    if (defaultNormalView_) vkDestroyImageView(device_, defaultNormalView_, nullptr);
    if (defaultNormalImage_) vmaDestroyImage(allocator_, defaultNormalImage_, defaultNormalAlloc_);
    defaultNormalView_ = VK_NULL_HANDLE;
    defaultNormalImage_ = VK_NULL_HANDLE;
    defaultNormalAlloc_ = VK_NULL_HANDLE;

    if (descriptorPool_) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    if (texturedSetLayout_) vkDestroyDescriptorSetLayout(device_, texturedSetLayout_, nullptr);
    if (pbrMaterialSetLayout_) vkDestroyDescriptorSetLayout(device_, pbrMaterialSetLayout_, nullptr);
    if (defaultSampler_) vkDestroySampler(device_, defaultSampler_, nullptr);
    if (shadowSampler_) vkDestroySampler(device_, shadowSampler_, nullptr);
    descriptorPool_ = VK_NULL_HANDLE;
    texturedSetLayout_ = VK_NULL_HANDLE;
    pbrMaterialSetLayout_ = VK_NULL_HANDLE;
    defaultSampler_ = VK_NULL_HANDLE;
    shadowSampler_ = VK_NULL_HANDLE;
}

void VulkanDevice::UploadToImage(VkImage image, uint32_t w, uint32_t h,
                                 const void* data, uint64_t size) {
    // 1. Host-visible staging buffer.
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    VmaAllocationInfo info{};
    Check(vmaCreateBuffer(allocator_, &bci, &aci, &staging, &stagingAlloc, &info),
          "vmaCreateBuffer(staging)");
    std::memcpy(info.pMappedData, data, size);
    Check(vmaFlushAllocation(allocator_, stagingAlloc, 0, size), "vmaFlushAllocation(staging)");

    // 2. Transient one-time command buffer.
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = graphicsQueueFamily_;
    VkCommandPool pool = VK_NULL_HANDLE;
    Check(vkCreateCommandPool(device_, &cpci, nullptr, &pool), "vkCreateCommandPool(transient)");

    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    Check(vkAllocateCommandBuffers(device_, &cbi, &cmd), "vkAllocateCommandBuffers(transient)");

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer(transient)");

    // 3. UNDEFINED -> TRANSFER_DST_OPTIMAL.
    TransitionImage(cmd, image,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // 4. Copy buffer -> image.
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(cmd, staging, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &region);

    // 5. TRANSFER_DST -> SHADER_READ_ONLY_OPTIMAL.
    TransitionImage(cmd, image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    Check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer(transient)");

    // 6. Submit (sync2) and wait.
    VkCommandBufferSubmitInfo cmdSub{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdSub.commandBuffer = cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdSub;
    Check(vkQueueSubmit2(graphicsQueue_, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit2(upload)");
    Check(vkQueueWaitIdle(graphicsQueue_), "vkQueueWaitIdle(upload)");

    // 7. Cleanup.
    vkDestroyCommandPool(device_, pool, nullptr);
    vmaDestroyBuffer(allocator_, staging, stagingAlloc);
}

void VulkanDevice::SetFrameUniforms(const void* data, uint32_t size) {
    // Writes the UBO for the frame currently being recorded (frameIndex_). currentFrameSet()
    // returns frameSet_[frameIndex_], whose descriptor points at uboBuffer_[frameIndex_] — the
    // same buffer mapped at uboMapped_[frameIndex_]. EndFrame advances frameIndex_, so calling
    // this after BeginFrame and before the draw targets the correct double-buffered UBO.
    if (size > kFrameUboSize) throw std::runtime_error("SetFrameUniforms: size exceeds UBO");
    std::memcpy(uboMapped_[frameIndex_], data, size);
    // Allocation is HOST_COHERENT; flush is a no-op safeguard against non-coherent fallback.
    Check(vmaFlushAllocation(allocator_, uboAlloc_[frameIndex_], 0, size),
          "vmaFlushAllocation(ubo)");
}

void VulkanDevice::SetJointPalette(const void* data, size_t size) {
    // Writes the joint-palette UBO for the frame currently being recorded (frameIndex_), mirroring
    // SetFrameUniforms. currentJointPaletteSet() returns jointSet_[frameIndex_], whose descriptor
    // points at jointBuffer_[frameIndex_] (mapped at jointMapped_[frameIndex_]).
    if (size > kJointPaletteSize) throw std::runtime_error("SetJointPalette: size exceeds UBO");
    std::memcpy(jointMapped_[frameIndex_], data, size);
    Check(vmaFlushAllocation(allocator_, jointAlloc_[frameIndex_], 0, size),
          "vmaFlushAllocation(joint)");
}

void VulkanDevice::WaitIdle() { if (device_) vkDeviceWaitIdle(device_); }

bool VulkanDevice::GetCapturedPixels(std::vector<uint8_t>& out, uint32_t& w, uint32_t& h) {
    if (capturedBGRA_.empty()) return false;
    out = std::move(capturedBGRA_);
    w = capW_;
    h = capH_;
    capturedBGRA_.clear();
    capW_ = capH_ = 0;
    return true;
}

// --- Resource creation ---
std::unique_ptr<IShaderModule> VulkanDevice::CreateShaderModule(const ShaderModuleDesc& d) {
    return std::make_unique<VulkanShaderModule>(device_, d.spirv);
}
std::unique_ptr<IPipeline> VulkanDevice::CreateGraphicsPipeline(const GraphicsPipelineDesc& d) {
    return std::make_unique<VulkanPipeline>(*this, d);
}
std::unique_ptr<IComputePipeline> VulkanDevice::CreateComputePipeline(const ComputePipelineDesc& d) {
    return std::make_unique<VulkanComputePipeline>(*this, d);
}
std::unique_ptr<IBuffer> VulkanDevice::CreateBuffer(const BufferDesc& d) {
    return std::make_unique<VulkanBuffer>(allocator_, d);
}
std::unique_ptr<ITexture> VulkanDevice::CreateTexture(const TextureDesc& d) {
    return std::make_unique<VulkanTexture>(*this, d);
}
std::unique_ptr<IRenderTarget> VulkanDevice::CreateRenderTarget(uint32_t w, uint32_t h) {
    return std::make_unique<VulkanRenderTarget>(*this, w, h);
}
std::unique_ptr<IRenderTarget> VulkanDevice::CreateShadowMap(uint32_t size) {
    return std::make_unique<VulkanRenderTarget>(*this, size, size, /*depthOnly=*/true);
}

// Point each per-frame set's shadow bindings (set 0, binding 1 = sampled depth, binding 2 =
// shadow sampler) at this shadow map. Call once after CreateShadowMap.
void VulkanDevice::SetShadowMap(IRenderTarget& smBase) {
    auto& sm = static_cast<VulkanRenderTarget&>(smBase);
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = sm.depthView();
        // Sampled while the shadow map is in SHADER_READ_ONLY (after EndShadowPass).
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo samplerInfo{};
        samplerInfo.sampler = shadowSampler_;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = frameSet_[i];
        writes[0].dstBinding = 1;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        writes[0].pImageInfo = &imageInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = frameSet_[i];
        writes[1].dstBinding = 2;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        writes[1].pImageInfo = &samplerInfo;

        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    }
}

// --- Shadow (depth-only) pass — reuses the dedicated rt command buffer/fence/recorder. ---
FrameContext VulkanDevice::BeginShadowPass(IRenderTarget& smBase) {
    auto& sm = static_cast<VulkanRenderTarget&>(smBase);
    shadowInFlight_ = &sm;

    Check(vkResetCommandBuffer(rtCmd_, 0), "vkResetCommandBuffer(shadow)");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(vkBeginCommandBuffer(rtCmd_, &bi), "vkBeginCommandBuffer(shadow)");

    // Transition the shadow depth from its current layout -> DEPTH_ATTACHMENT_OPTIMAL (cleared).
    TransitionImage(rtCmd_, sm.depthImage(),
                    sm.depthLayout(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);
    sm.setDepthLayout(VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    // Depth-only recorder: null color view -> BeginRenderPass emits no color attachment.
    rtRecorder_->Begin(rtCmd_, VK_NULL_HANDLE, sm.depthView(), {sm.width(), sm.height()});
    return FrameContext{rtRecorder_.get()};
}

void VulkanDevice::EndShadowPass(const FrameContext& frame) {
    if (!frame.cmd || !shadowInFlight_) return;
    VulkanRenderTarget& sm = *shadowInFlight_;

    // The sample already called EndRenderPass() (vkCmdEndRendering). Transition the depth image
    // DEPTH_ATTACHMENT -> SHADER_READ_ONLY so the lit pass can sample it.
    TransitionImage(rtCmd_, sm.depthImage(),
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);
    sm.setDepthLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    Check(vkEndCommandBuffer(rtCmd_), "vkEndCommandBuffer(shadow)");

    VkCommandBufferSubmitInfo cmdSub{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdSub.commandBuffer = rtCmd_;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdSub;
    Check(vkResetFences(device_, 1, &rtFence_), "vkResetFences(shadow)");
    Check(vkQueueSubmit2(graphicsQueue_, 1, &submit, rtFence_), "vkQueueSubmit2(shadow)");
    Check(vkWaitForFences(device_, 1, &rtFence_, VK_TRUE, UINT64_MAX), "vkWaitForFences(shadow)");

    shadowInFlight_ = nullptr;
}

// --- Offscreen render-target pass ---
FrameContext VulkanDevice::BeginRenderTargetFrame(IRenderTarget& rtBase) {
    auto& rt = static_cast<VulkanRenderTarget&>(rtBase);
    rtInFlight_ = &rt;

    Check(vkResetCommandBuffer(rtCmd_, 0), "vkResetCommandBuffer(rt)");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(vkBeginCommandBuffer(rtCmd_, &bi), "vkBeginCommandBuffer(rt)");

    // Transition the RT color from its current layout -> COLOR_ATTACHMENT_OPTIMAL.
    TransitionImage(rtCmd_, rt.colorImage(),
                    rt.colorLayout(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    rt.setColorLayout(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Transition the RT depth UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL (cleared each frame).
    TransitionImage(rtCmd_, rt.depthImage(),
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);

    rtRecorder_->Begin(rtCmd_, rt.colorView(), rt.depthView(),
                       {rt.width(), rt.height()});
    return FrameContext{rtRecorder_.get()};
}

void VulkanDevice::EndRenderTargetFrame(const FrameContext& frame) {
    if (!frame.cmd || !rtInFlight_) return;
    VulkanRenderTarget& rt = *rtInFlight_;

    // The sample already called EndRenderPass() (vkCmdEndRendering). Transition the color image
    // COLOR_ATTACHMENT -> SHADER_READ_ONLY so the later swapchain pass can sample it.
    TransitionImage(rtCmd_, rt.colorImage(),
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    rt.setColorLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    Check(vkEndCommandBuffer(rtCmd_), "vkEndCommandBuffer(rt)");

    // Submit with no semaphores; signal rtFence_ and wait so the swapchain pass sees the result.
    VkCommandBufferSubmitInfo cmdSub{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdSub.commandBuffer = rtCmd_;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdSub;
    Check(vkResetFences(device_, 1, &rtFence_), "vkResetFences(rt)");
    Check(vkQueueSubmit2(graphicsQueue_, 1, &submit, rtFence_), "vkQueueSubmit2(rt)");
    Check(vkWaitForFences(device_, 1, &rtFence_, VK_TRUE, UINT64_MAX), "vkWaitForFences(rt)");

    rtInFlight_ = nullptr;
}

// --- Frame loop ---
FrameContext VulkanDevice::BeginFrame() {
    FrameSync& fr = frames_[frameIndex_];
    vkWaitForFences(device_, 1, &fr.inFlight, VK_TRUE, UINT64_MAX);

    VkResult acq = vkAcquireNextImageKHR(device_, swapchain_->handle(), UINT64_MAX,
                                         fr.imageAvailable, VK_NULL_HANDLE, &acquiredImage_);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchain_->Recreate(window_.FramebufferWidth(), window_.FramebufferHeight());
        return FrameContext{nullptr};
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) Check(acq, "vkAcquireNextImageKHR");

    vkResetFences(device_, 1, &fr.inFlight);
    vkResetCommandBuffer(fr.cmd, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Check(vkBeginCommandBuffer(fr.cmd, &bi), "vkBeginCommandBuffer");

    // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
    TransitionImage(fr.cmd, swapchain_->image(acquiredImage_),
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    // UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL (no transition back: depth isn't presented)
    TransitionImage(fr.cmd, swapchain_->depthImage(),
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT);

    recorder_->Begin(fr.cmd, swapchain_->view(acquiredImage_),
                     swapchain_->depthView(), swapchain_->extent());
    return FrameContext{recorder_.get()};
}

void VulkanDevice::EndFrame(const FrameContext& frame) {
    if (!frame.cmd) return;  // skipped frame
    FrameSync& fr = frames_[frameIndex_];

    // --- Headless capture branch: copy the rendered image back to host memory, skip present. ---
    if (captureArmed_) {
        VkExtent2D extent = swapchain_->extent();
        VkImage srcImage = swapchain_->image(acquiredImage_);

        // COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL
        TransitionImage(fr.cmd, srcImage,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

        // Host-visible MAPPED staging buffer (TRANSFER_DST), HOST_COHERENT required.
        const VkDeviceSize bufSize =
            (VkDeviceSize)extent.width * extent.height * 4;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bufSize;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocation stagingAlloc = VK_NULL_HANDLE;
        VmaAllocationInfo stagingInfo{};
        Check(vmaCreateBuffer(allocator_, &bci, &aci, &staging, &stagingAlloc, &stagingInfo),
              "vmaCreateBuffer(capture)");

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(fr.cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging, 1, &region);

        Check(vkEndCommandBuffer(fr.cmd), "vkEndCommandBuffer(capture)");

        // Submit: wait on imageAvailable (so the copy follows acquire/render), signal the
        // in-flight fence. Do NOT signal renderFinished — present is skipped, so nobody waits
        // on it (avoids a signaled-but-unwaited semaphore).
        VkSemaphoreSubmitInfo waitSem{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        waitSem.semaphore = fr.imageAvailable;
        waitSem.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkCommandBufferSubmitInfo cmdSub{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cmdSub.commandBuffer = fr.cmd;
        VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit.waitSemaphoreInfoCount = 1;
        submit.pWaitSemaphoreInfos = &waitSem;
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &cmdSub;
        Check(vkQueueSubmit2(graphicsQueue_, 1, &submit, fr.inFlight), "vkQueueSubmit2(capture)");
        Check(vkQueueWaitIdle(graphicsQueue_), "vkQueueWaitIdle(capture)");

        // Map staging and copy out (top-row-first BGRA8, tightly packed).
        capW_ = extent.width;
        capH_ = extent.height;
        capturedBGRA_.resize((size_t)capW_ * capH_ * 4);
        std::memcpy(capturedBGRA_.data(), stagingInfo.pMappedData, capturedBGRA_.size());

        vmaDestroyBuffer(allocator_, staging, stagingAlloc);

        // Skip present. Advance frame index (consistent with the normal path) and disarm.
        frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
        captureArmed_ = false;
        return;
    }

    // COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC
    TransitionImage(fr.cmd, swapchain_->image(acquiredImage_),
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);

    Check(vkEndCommandBuffer(fr.cmd), "vkEndCommandBuffer");

    VkSemaphoreSubmitInfo waitSem{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    waitSem.semaphore = fr.imageAvailable;
    waitSem.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphoreSubmitInfo signalSem{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signalSem.semaphore = fr.renderFinished;
    signalSem.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
    VkCommandBufferSubmitInfo cmdSub{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdSub.commandBuffer = fr.cmd;

    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.waitSemaphoreInfoCount = 1;
    submit.pWaitSemaphoreInfos = &waitSem;
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdSub;
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signalSem;
    Check(vkQueueSubmit2(graphicsQueue_, 1, &submit, fr.inFlight), "vkQueueSubmit2");

    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &fr.renderFinished;
    VkSwapchainKHR sc = swapchain_->handle();
    present.swapchainCount = 1;
    present.pSwapchains = &sc;
    present.pImageIndices = &acquiredImage_;
    VkResult pr = vkQueuePresentKHR(graphicsQueue_, &present);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
        swapchain_->Recreate(window_.FramebufferWidth(), window_.FramebufferHeight());
    } else {
        Check(pr, "vkQueuePresentKHR");
    }

    frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
}

} // namespace hf::rhi::vk
