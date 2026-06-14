#pragma once
#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"
#include "rhi_vulkan/vulkan_swapchain.h"
#include "hal/window.h"
#include <map>
#include <memory>

namespace hf::rhi::vk {

// Frames-in-flight; double-buffered.
constexpr uint32_t kFramesInFlight = 2;

class VulkanDevice final : public IRHIDevice {
public:
    explicit VulkanDevice(hf::hal::Window& window);
    ~VulkanDevice() override;

    ISwapchain& Swapchain() override { return *swapchain_; }

    std::unique_ptr<IShaderModule> CreateShaderModule(const ShaderModuleDesc&) override;
    std::unique_ptr<IPipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc&) override;
    std::unique_ptr<IComputePipeline> CreateComputePipeline(const ComputePipelineDesc&) override;
    std::unique_ptr<IBuffer> CreateBuffer(const BufferDesc&) override;
    std::unique_ptr<ITexture> CreateTexture(const TextureDesc&) override;
    std::unique_ptr<IRenderTarget> CreateRenderTarget(uint32_t width, uint32_t height) override;
    std::unique_ptr<IRenderTarget> CreateShadowMap(uint32_t size) override;

    FrameContext BeginRenderTargetFrame(IRenderTarget& rt) override;
    void EndRenderTargetFrame(const FrameContext&) override;

    FrameContext BeginShadowPass(IRenderTarget& shadowMap) override;
    void EndShadowPass(const FrameContext&) override;
    void SetShadowMap(IRenderTarget& shadowMap) override;

    FrameContext BeginFrame() override;
    void EndFrame(const FrameContext&) override;
    void SetFrameUniforms(const void* data, uint32_t size) override;
    void SetJointPalette(const void* data, size_t size) override;
    void WaitIdle() override;
    void CaptureNextFrame() override { captureArmed_ = true; }
    bool GetCapturedPixels(std::vector<uint8_t>& outBGRA,
                           uint32_t& width, uint32_t& height) override;

    // Accessors used by sibling Vulkan objects.
    VkDevice device() const { return device_; }
    VmaAllocator allocator() const { return allocator_; }
    VkSampler defaultSampler() const { return defaultSampler_; }
    // 1x1 flat tangent-space normal (0.5,0.5,1)->(0,0,1): the default normal map bound at material
    // set binding 3/4 so every material set is complete even without an explicit normal map.
    VkImageView defaultNormalView() const { return defaultNormalView_; }
    VkDescriptorSetLayout texturedSetLayout() const { return texturedSetLayout_; }
    // The material set layout logically lives at set 1 (set index decided by the pipeline
    // layout array); the layout object is identical to the Slice C textured set layout.
    VkDescriptorSetLayout materialSetLayout() const { return texturedSetLayout_; }
    // The wider full-PBR material set layout (set 1 for the lit-PBR pipeline only): 5 sampled
    // images + 5 samplers (base/normal/metalRough/emissive/occlusion). Separate from the 2-texture
    // material set so the existing lit pipeline is unaffected.
    VkDescriptorSetLayout pbrMaterialSetLayout() const { return pbrMaterialSetLayout_; }
    VkDescriptorPool descriptorPool() const { return descriptorPool_; }
    // Per-frame UBO set (set 0). currentFrameSet() returns the set for the frame being recorded.
    VkDescriptorSetLayout frameSetLayout() const { return frameSetLayout_; }
    VkDescriptorSet currentFrameSet() const { return frameSet_[frameIndex_]; }
    // Joint-palette UBO set (set 2). currentJointPaletteSet() returns the set for the recording frame.
    VkDescriptorSetLayout jointPaletteSetLayout() const { return jointSetLayout_; }
    VkDescriptorSet currentJointPaletteSet() const { return jointSet_[frameIndex_]; }
    // Swapchain color format — render targets match it so the lit pipeline renders in unchanged.
    VkFormat swapchainFormat() const { return swapchain_->vkFormat(); }

    // vkCmdPushDescriptorSetKHR loader (VK_KHR_push_descriptor). Used by the compute command path
    // to bind storage buffers inline. Null if the extension was unavailable.
    PFN_vkCmdPushDescriptorSetKHR pushDescriptorFn() const { return vkCmdPushDescriptorSetKHR_; }

    // Stage host pixel data into a device-local image (synchronous; see vulkan_texture).
    void UploadToImage(VkImage image, uint32_t w, uint32_t h,
                       const void* data, uint64_t size);

    // Return (building + caching on first use) the descriptor set for a full-PBR material — the
    // wider set 1 pointing at the five textures' views. Keyed on the base-texture pointer (a
    // material binds a fixed 5-texture set), so the command-buffer BindMaterialPBR path re-binds an
    // already-built set without re-issuing descriptor writes each frame. Lives until device teardown.
    VkDescriptorSet pbrMaterialSet(ITexture& base, ITexture& metalRough, ITexture& normalMap,
                                   ITexture& emissive, ITexture& occlusion);

private:
    void CreateSyncObjects();
    void DestroySyncObjects();
    void CreateTextureResources();
    void DestroyTextureResources();

    hf::hal::Window& window_;

    vkb::Instance vkbInstance_{};
    vkb::Device   vkbDevice_{};
    VkSurfaceKHR  surface_ = VK_NULL_HANDLE;
    VkInstance    instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice      device_ = VK_NULL_HANDLE;
    VkQueue       graphicsQueue_ = VK_NULL_HANDLE;
    uint32_t      graphicsQueueFamily_ = 0;
    VmaAllocator  allocator_ = VK_NULL_HANDLE;

    // Loaded from the device (VK_KHR_push_descriptor); used to bind compute storage buffers inline.
    PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR_ = nullptr;

    // Texture support: one default sampler + a shared image+sampler set layout (set 1) + pool.
    VkSampler             defaultSampler_    = VK_NULL_HANDLE;
    VkSampler             shadowSampler_     = VK_NULL_HANDLE;  // clamp-to-edge linear (shadow map)
    VkDescriptorSetLayout texturedSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout pbrMaterialSetLayout_ = VK_NULL_HANDLE;  // wider set 1 for lit-PBR
    VkDescriptorPool      descriptorPool_    = VK_NULL_HANDLE;

    // 1x1 flat tangent-space normal map (RGBA 128,128,255,255 -> (0,0,1) after decode), used as the
    // default normal map when a material has none, so the lit shader's TBN perturbation is a no-op.
    VkImage         defaultNormalImage_ = VK_NULL_HANDLE;
    VmaAllocation   defaultNormalAlloc_ = VK_NULL_HANDLE;
    VkImageView     defaultNormalView_  = VK_NULL_HANDLE;

    // Per-frame uniform buffers (set 0): one host-visible mapped UBO + descriptor set per
    // frame-in-flight, so the CPU never writes a UBO the GPU is still reading.
    static constexpr uint32_t kFrameUboSize = 512;  // >= sizeof(FrameData) (288B w/ lightViewProj)
    VkDescriptorSetLayout frameSetLayout_ = VK_NULL_HANDLE;
    VkBuffer        uboBuffer_[kFramesInFlight]{};
    VmaAllocation   uboAlloc_[kFramesInFlight]{};
    void*           uboMapped_[kFramesInFlight]{};
    VkDescriptorSet frameSet_[kFramesInFlight]{};

    // Joint-palette UBO (set 2): JointPalette { float4x4 joints[64]; } = 4096 B. One host-visible
    // mapped UBO + descriptor set per frame-in-flight, mirroring the per-frame UBO above. Written by
    // SetJointPalette; auto-bound at set 2 on BindPipeline when the pipeline declares usesJointPalette.
    static constexpr uint32_t kJointPaletteSize = 64 * 64;  // 64 mat4 (64B each) = 4096 B
    VkDescriptorSetLayout jointSetLayout_ = VK_NULL_HANDLE;
    VkBuffer        jointBuffer_[kFramesInFlight]{};
    VmaAllocation   jointAlloc_[kFramesInFlight]{};
    void*           jointMapped_[kFramesInFlight]{};
    VkDescriptorSet jointSet_[kFramesInFlight]{};

    // Cache of built full-PBR material descriptor sets, keyed on the base-texture pointer. Owns the
    // VulkanPbrMaterial objects for the device's lifetime (freed on teardown before the pool dies).
    std::map<ITexture*, std::unique_ptr<class VulkanPbrMaterial>> pbrMaterials_;

    std::unique_ptr<VulkanSwapchain> swapchain_;

    // Per-frame-in-flight sync + command resources.
    struct FrameSync {
        VkCommandPool   pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd  = VK_NULL_HANDLE;
        VkSemaphore     imageAvailable = VK_NULL_HANDLE;
        VkSemaphore     renderFinished = VK_NULL_HANDLE;
        VkFence         inFlight = VK_NULL_HANDLE;
    };
    FrameSync frames_[kFramesInFlight];
    uint32_t  frameIndex_ = 0;
    uint32_t  acquiredImage_ = 0;

    // Headless capture state. When captureArmed_ is set, EndFrame copies the rendered
    // swapchain image back to host memory (capturedBGRA_) instead of presenting.
    bool                 captureArmed_ = false;
    std::vector<uint8_t> capturedBGRA_;
    uint32_t             capW_ = 0, capH_ = 0;

    std::unique_ptr<class VulkanCommandBuffer> recorder_;

    // Dedicated command resources for the offscreen render-target pass. Independent of the
    // per-frame swapchain sync: BeginRenderTargetFrame records here, EndRenderTargetFrame
    // submits + waits on rtFence_ before the swapchain pass samples the RT.
    VkCommandPool   rtPool_  = VK_NULL_HANDLE;
    VkCommandBuffer rtCmd_   = VK_NULL_HANDLE;
    VkFence         rtFence_ = VK_NULL_HANDLE;
    std::unique_ptr<class VulkanCommandBuffer> rtRecorder_;
    class VulkanRenderTarget* rtInFlight_ = nullptr;  // RT being recorded between Begin/End

    // The shadow (depth-only) pass reuses the dedicated rt command buffer/fence/recorder: the
    // shadow pass runs, waits on rtFence_, then the RT pass runs — they never overlap.
    class VulkanRenderTarget* shadowInFlight_ = nullptr;  // shadow map recorded between Begin/End
};

} // namespace hf::rhi::vk
