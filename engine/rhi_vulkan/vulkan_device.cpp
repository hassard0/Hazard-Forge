#include "rhi_vulkan/vulkan_device.h"
#include "rhi_vulkan/vk_common.h"
#include "rhi_vulkan/vulkan_command_buffer.h"
#include "rhi_vulkan/vulkan_shader.h"
#include "rhi_vulkan/vulkan_pipeline.h"
#include "rhi_vulkan/vulkan_buffer.h"

#define VMA_IMPLEMENTATION
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace hf::rhi::vk {

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

    vkb::DeviceBuilder db{physRet.value()};
    auto devRet = db.build();
    if (!devRet) throw std::runtime_error("Device build failed: " + devRet.error().message());
    vkbDevice_ = devRet.value();
    device_ = vkbDevice_.device;
    physical_ = physRet.value().physical_device;

    graphicsQueue_ = vkbDevice_.get_queue(vkb::QueueType::graphics).value();
    graphicsQueueFamily_ = vkbDevice_.get_queue_index(vkb::QueueType::graphics).value();

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

    CreateSyncObjects();
    recorder_ = std::make_unique<VulkanCommandBuffer>(*this);
}

VulkanDevice::~VulkanDevice() {
    if (device_) vkDeviceWaitIdle(device_);
    recorder_.reset();
    DestroySyncObjects();
    swapchain_.reset();
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

void VulkanDevice::WaitIdle() { if (device_) vkDeviceWaitIdle(device_); }

// --- Resource creation ---
std::unique_ptr<IShaderModule> VulkanDevice::CreateShaderModule(const ShaderModuleDesc& d) {
    return std::make_unique<VulkanShaderModule>(device_, d.spirv);
}
std::unique_ptr<IPipeline> VulkanDevice::CreateGraphicsPipeline(const GraphicsPipelineDesc& d) {
    return std::make_unique<VulkanPipeline>(device_, d);
}
std::unique_ptr<IBuffer> VulkanDevice::CreateBuffer(const BufferDesc& d) {
    return std::make_unique<VulkanBuffer>(allocator_, d);
}

// --- Frame loop ---
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
