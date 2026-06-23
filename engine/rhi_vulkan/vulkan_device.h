#pragma once
#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vk_mem_alloc.h>
#include "rhi/rhi.h"
#include "rhi_vulkan/vulkan_swapchain.h"
#include "hal/window.h"
#include <map>
#include <memory>
#include <utility>
#include <vector>

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
    std::unique_ptr<IBindlessTextureSet> CreateBindlessTextureSet(
        std::span<ITexture* const> textures) override;
    std::unique_ptr<IRenderTarget> CreateRenderTarget(uint32_t width, uint32_t height) override;
    std::unique_ptr<IRenderTarget> CreateRenderTarget(uint32_t width, uint32_t height,
                                                      Format colorFormat) override;
    std::unique_ptr<IRenderTarget> CreateShadowMap(uint32_t size) override;

    // Slice DD — runtime cubemap-capture reflection probe.
    std::unique_ptr<ICubemapTarget> CreateCubemapTarget(uint32_t size, Format colorFormat) override;
    FrameContext BeginCubemapFace(ICubemapTarget& cube, uint32_t face) override;
    void EndCubemapFace(const FrameContext&) override;
    bool ReadCubemapFace(ICubemapTarget& cube, uint32_t face, std::vector<uint8_t>& outBGRA,
                         uint32_t& width, uint32_t& height) override;
    bool ReadRenderTarget(IRenderTarget& rt, std::vector<uint8_t>& outBGRA,
                          uint32_t& width, uint32_t& height) override;

    FrameContext BeginRenderTargetFrame(IRenderTarget& rt) override;
    void EndRenderTargetFrame(const FrameContext&) override;

    FrameContext BeginShadowPass(IRenderTarget& shadowMap) override;
    void EndShadowPass(const FrameContext&) override;
    void SetShadowMap(IRenderTarget& shadowMap) override;

    ICommandBuffer* CreateSecondaryCommandBuffer(uint32_t threadIndex) override;

    FrameContext BeginFrame() override;
    void EndFrame(const FrameContext&) override;
    void SetFrameUniforms(const void* data, uint32_t size) override;
    void SetJointPalette(const void* data, size_t size) override;
    void WaitIdle() override;
    void CaptureNextFrame() override { captureArmed_ = true; }
    bool GetCapturedPixels(std::vector<uint8_t>& outBGRA,
                           uint32_t& width, uint32_t& height) override;
    void ReadBuffer(IBuffer& buffer, void* dst, size_t size, size_t offset) override;

    // Slice RT2 — the Vulkan hardware ray-query backend behind the RT1 accel-struct seam. CreateBlas builds
    // a bottom-level accel structure from AABB-procedural geometry (each AABB inflated by kRtAabbMargin so
    // the driver's FLOAT BVH overlap is a strict SUPERSET of every true fx hit); CreateTlas builds a
    // top-level accel structure of those instances (instanceCustomIndex = instanceId). Both build
    // synchronously (one-shot cmd buffer + vkQueueWaitIdle, the staging-upload pattern). On a device
    // WITHOUT RT support (no rayQuery/accelerationStructure feature+extension) they return nullptr and
    // SupportsHardwareRayQuery() is false -> the showcase falls back to the SW path.
    std::unique_ptr<IAccelStructure> CreateBlas(const BlasDesc&) override;
    std::unique_ptr<IAccelStructure> CreateTlas(const TlasDesc&) override;
    bool SupportsHardwareRayQuery() const override { return hwRayQuery_; }

    // Accessors used by sibling Vulkan objects.
    VkDevice device() const { return device_; }
    VkPhysicalDevice physicalDevice() const { return physical_; }
    VmaAllocator allocator() const { return allocator_; }
    VkQueue graphicsQueue() const { return graphicsQueue_; }
    uint32_t graphicsQueueFamily() const { return graphicsQueueFamily_; }
    VkCommandPool rtCommandPool() const { return rtPool_; }

    // Slice RT2 — the acceleration-structure entry points loaded from the device (null if RT unsupported).
    PFN_vkCreateAccelerationStructureKHR        createAccelStructFn() const { return vkCreateAccelerationStructureKHR_; }
    PFN_vkDestroyAccelerationStructureKHR       destroyAccelStructFn() const { return vkDestroyAccelerationStructureKHR_; }
    PFN_vkGetAccelerationStructureBuildSizesKHR getAccelStructBuildSizesFn() const { return vkGetAccelerationStructureBuildSizesKHR_; }
    PFN_vkCmdBuildAccelerationStructuresKHR     cmdBuildAccelStructsFn() const { return vkCmdBuildAccelerationStructuresKHR_; }
    PFN_vkGetAccelerationStructureDeviceAddressKHR getAccelStructDeviceAddressFn() const { return vkGetAccelerationStructureDeviceAddressKHR_; }
    VkSampler defaultSampler() const { return defaultSampler_; }
    // Slice CX: the clamp-to-edge depth sampler the lit pass uses for the shadow map, exposed so the
    // froxel inject compute can sample the SAME map via BindShadowMapCompute.
    VkSampler shadowSampler() const { return shadowSampler_; }
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

    // Stage N mip levels into a device-local image (synchronous). mipData[i] points at tightly-packed
    // pixels for mip i (dimensions max(1,w>>i) x max(1,h>>i)); mipBytes[i] is its byte count. Used by
    // the HDR environment texture (Slice R). bytesPerPixel selects the element size (8 for RGBA16F).
    void UploadToImageMips(VkImage image, uint32_t w, uint32_t h, uint32_t mipLevels,
                           const void* const* mipData, const uint64_t* mipBytes);

    // HDR environment sampling support (Slice R): the trilinear sampler (repeat-U / clamp-V) used by
    // the equirect env map, and the dedicated environment descriptor set layout (set 3: binding 0 =
    // sampled image, binding 1 = sampler). Separate from the material/frame/joint layouts so the
    // existing pipelines' set 0/1/2 layouts are byte-for-byte unchanged.
    VkSampler environmentSampler() const { return environmentSampler_; }
    VkDescriptorSetLayout environmentSetLayout() const { return environmentSetLayout_; }

    // Clustered-lighting set layout (set 3, fragment stage, Slice AG): THREE STORAGE_BUFFER bindings
    // (0=clusters, 1=lightIndices, 2=lights). Created with PUSH_DESCRIPTOR so the three buffers are
    // bound inline via BindLightClusters (no pool sizing — mirrors the compute SSBO path). Separate
    // from the material/frame/joint/env layouts so the existing set 0/1/2 layouts (and the
    // golden-locked pipelines) are byte-for-byte unchanged.
    VkDescriptorSetLayout clusterSetLayout() const { return clusterSetLayout_; }

    // Per-draw set layout (set 2, VERTEX stage, Slice BM): ONE STORAGE_BUFFER binding (0) holding the
    // PerDraw[ ] array (model mat4 + material) the multi-draw-indirect vertex shader reads as
    // PerDraw[gl_DrawID]. PUSH_DESCRIPTOR so the buffer is bound inline via BindPerDrawData (no pool —
    // mirrors the cluster/compute SSBO path). Separate from the material/frame/joint/env/cluster
    // layouts so existing pipelines are byte-for-byte unchanged.
    VkDescriptorSetLayout perDrawSetLayout() const { return perDrawSetLayout_; }

    // Bindless texture set layout (set 4, FRAGMENT stage, Slice BZ): binding 0 = an UNBOUNDED runtime
    // sampled-image ARRAY (PARTIALLY_BOUND + VARIABLE_DESCRIPTOR_COUNT, max kBindlessMaxTextures) holding
    // every scene texture; binding 1 = a shared sampler. Created with UPDATE_AFTER_BIND so the array is
    // written after allocation. Separate from the material/frame/joint/env/cluster/perDraw layouts so
    // existing pipelines are byte-for-byte unchanged. The CreateBindlessTextureSet path allocates a set
    // from this layout + the dedicated update-after-bind pool.
    VkDescriptorSetLayout bindlessSetLayout() const { return bindlessSetLayout_; }
    // Upper bound on the bindless array's descriptor count (the layout's array size). The showcase scene
    // has a handful of textures; this is the variable-count ceiling.
    static constexpr uint32_t kBindlessMaxTextures = 256;

    // RT-graphics accel set layout (a DEDICATED set, FRAGMENT stage, Issue #34): ONE
    // ACCELERATION_STRUCTURE_KHR binding at the slot the desc requests, so a ps_6_5 RayQuery fragment
    // shader can trace rays. PUSH_DESCRIPTOR so the TLAS is bound inline via BindAccelStructure at
    // VK_PIPELINE_BIND_POINT_GRAPHICS (no pool — mirrors the compute accel + cluster SSBO paths). Separate
    // from every other set layout so existing pipelines are byte-for-byte unchanged. The binding slot is
    // baked at create (the RT graphics frag declares it at binding `slot`). Lazily created (depends on the
    // rayQuery device feature; nullptr if RT is unavailable).
    VkDescriptorSetLayout accelGraphicsSetLayout(uint32_t slot);

    // Return (building + caching on first use) the descriptor set for a full-PBR material — the
    // wider set 1 pointing at the five textures' views. Keyed on the base-texture pointer (a
    // material binds a fixed 5-texture set), so the command-buffer BindMaterialPBR path re-binds an
    // already-built set without re-issuing descriptor writes each frame. Lives until device teardown.
    VkDescriptorSet pbrMaterialSet(ITexture& base, ITexture& metalRough, ITexture& normalMap,
                                   ITexture& emissive, ITexture& occlusion);

    // Return (building + caching on first use) the 2-texture material set (texturedSetLayout_, set 1)
    // for a (base-color view, normal-map view) PAIR: binding 0/1 = base color + default sampler,
    // binding 3/4 = normal map + default sampler. Keyed on BOTH views so each distinct (base, normal)
    // combination owns its OWN immutable set, written exactly once here and never mutated again.
    // BindMaterial routes through this so a base texture shared by renderables with DIFFERENT normal
    // maps (the default scene: checker+normalmap on the plane/cubes, checker+flat_normal on the
    // spheres) no longer re-points one set IN PLACE after it was already bound into the recording
    // command buffer — which the validation layer flagged as updating a bound set without
    // UPDATE_AFTER_BIND (VUID-vkCmdBindDescriptorSets-commandBuffer-recording). Descriptor contents
    // are byte-identical to the old in-place-updated set, so the render is unchanged. Lives until
    // device teardown.
    VkDescriptorSet materialSet(VkImageView baseView, VkImageView normalView);

private:
    // Slice DD — copy a color image's (mip 0) array layer back to host memory at `bpp` bytes/pixel
    // (the image must be in SHADER_READ_ONLY). Shared by ReadCubemapFace + ReadRenderTarget.
    bool readImageLayer(VkImage image, uint32_t layer, uint32_t w, uint32_t h, uint32_t bpp,
                        VkImageLayout currentLayout, std::vector<uint8_t>& out);
    void CreateSyncObjects();
    void DestroySyncObjects();
    void CreateRenderFinishedSemaphores();   // one present-wait semaphore per swapchain image
    void DestroyRenderFinishedSemaphores();
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

    // Slice RT2 — hardware ray query. hwRayQuery_ is true iff BOTH the rayQuery + accelerationStructure
    // features were advertised AND both extensions enabled (queried at device creation). The 5 accel-struct
    // entry points are loaded only when supported (else null -> CreateBlas/CreateTlas return nullptr).
    bool hwRayQuery_ = false;
    PFN_vkCreateAccelerationStructureKHR           vkCreateAccelerationStructureKHR_ = nullptr;
    PFN_vkDestroyAccelerationStructureKHR          vkDestroyAccelerationStructureKHR_ = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR    vkGetAccelerationStructureBuildSizesKHR_ = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR        vkCmdBuildAccelerationStructuresKHR_ = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR_ = nullptr;

    // Texture support: one default sampler + a shared image+sampler set layout (set 1) + pool.
    VkSampler             defaultSampler_    = VK_NULL_HANDLE;
    VkSampler             shadowSampler_     = VK_NULL_HANDLE;  // clamp-to-edge linear (shadow map)
    VkSampler             environmentSampler_ = VK_NULL_HANDLE; // trilinear repeat-U/clamp-V (HDR env)
    VkDescriptorSetLayout texturedSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout pbrMaterialSetLayout_ = VK_NULL_HANDLE;  // wider set 1 for lit-PBR
    VkDescriptorSetLayout environmentSetLayout_ = VK_NULL_HANDLE;  // dedicated set 3 for HDR IBL
    VkDescriptorSetLayout clusterSetLayout_ = VK_NULL_HANDLE;      // dedicated set 3 for clustered lights
    VkDescriptorSetLayout perDrawSetLayout_ = VK_NULL_HANDLE;      // dedicated set 2 for MDI per-draw (Slice BM)
    VkDescriptorSetLayout bindlessSetLayout_ = VK_NULL_HANDLE;     // dedicated set 4 for bindless textures (Slice BZ)
    VkDescriptorSetLayout accelGraphicsSetLayout_ = VK_NULL_HANDLE; // dedicated set for RT-graphics accel (Issue #34)
    uint32_t              accelGraphicsSlot_ = 0;                  // the accel binding slot baked into the layout
    VkDescriptorPool      descriptorPool_    = VK_NULL_HANDLE;

    // 1x1 flat tangent-space normal map (RGBA 128,128,255,255 -> (0,0,1) after decode), used as the
    // default normal map when a material has none, so the lit shader's TBN perturbation is a no-op.
    VkImage         defaultNormalImage_ = VK_NULL_HANDLE;
    VmaAllocation   defaultNormalAlloc_ = VK_NULL_HANDLE;
    VkImageView     defaultNormalView_  = VK_NULL_HANDLE;

    // Per-frame uniform buffers (set 0): one host-visible mapped UBO + descriptor set per
    // frame-in-flight, so the CPU never writes a UBO the GPU is still reading.
    // 1024B (was 512). Bumped for the CSM FrameData layout (Slice AD): 4 cascade mat4 (256B) on top
    // of the ~352B base layout + split/atlas vec4s overflows 512. Additive — existing shaders read
    // fewer bytes, so their layout (and the 15 goldens) are byte-for-byte unchanged.
    static constexpr uint32_t kFrameUboSize = 1024;  // >= sizeof(FrameData) and the CSM layout (464B)
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

    // Cache of immutable 2-texture material sets, keyed on (base-color view, normal-map view). Each
    // distinct pair owns one set on texturedSetLayout_, written once in materialSet() and never
    // mutated, so BindMaterial never updates a descriptor set that is still bound in a recording
    // command buffer. Freed when the descriptor pool is destroyed (the sets are pool-owned).
    std::map<std::pair<VkImageView, VkImageView>, VkDescriptorSet> materialSets_;

    std::unique_ptr<VulkanSwapchain> swapchain_;

    // Per-frame-in-flight sync + command resources. imageAvailable is per-frame-in-flight: its reuse
    // is gated by waiting inFlight before the acquire that signals it again. renderFinished is NOT
    // here — see renderFinished_ below.
    struct FrameSync {
        VkCommandPool   pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd  = VK_NULL_HANDLE;
        VkSemaphore     imageAvailable = VK_NULL_HANDLE;
        VkFence         inFlight = VK_NULL_HANDLE;
    };
    FrameSync frames_[kFramesInFlight];
    uint32_t  frameIndex_ = 0;
    uint32_t  acquiredImage_ = 0;

    // Present-wait (render-finished) semaphores, sized PER SWAPCHAIN IMAGE rather than per
    // frame-in-flight. A present-wait semaphore is signalled by the queue submit that renders into
    // image i and waited by the present of image i; its completion is NOT observable through the
    // per-frame inFlight fence (present is a separate queue operation). With only kFramesInFlight
    // semaphores, frame N+kFramesInFlight could re-submit a renderFinished semaphore whose previous
    // present-wait had not yet completed (vkAcquireNextImageKHR can hand back image indices in any
    // order), which the core-validation layer flags as semaphore reuse. One semaphore per swapchain
    // image, indexed by the ACQUIRED image, makes each present-wait signal/wait pair unique to that
    // image and never re-submitted while pending. Sized in CreateSyncObjects from the swapchain image
    // count; rebuilt on swapchain Recreate.
    std::vector<VkSemaphore> renderFinished_;

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

    // Slice DD — the cubemap-face capture pass reuses the SAME dedicated rt command buffer/fence/
    // recorder (the face passes run serially, one Begin/EndCubemapFace at a time, never overlapping
    // the shadow/rt passes). The in-flight cube + the face being recorded between Begin/End.
    class VulkanCubemapTarget* cubeInFlight_ = nullptr;
    uint32_t cubeFaceInFlight_ = 0;

    // --- Multithreaded recording (Slice AU) ----------------------------------
    // One command pool + secondary command buffer + recorder PER WORKER THREAD (pools are not
    // thread-safe, so each worker owns its own). Created lazily on first use and reused every frame;
    // grown if a frame asks for more workers than seen before. The primary recorder records the
    // current render pass's INHERITANCE (color/depth formats, sample count, extent, hasColor) into
    // mtInherit_ when BeginRenderPass(clear, expectsSecondaries=true) opens the pass, so
    // CreateSecondaryCommandBuffer begins each secondary with the matching
    // VkCommandBufferInheritanceRenderingInfo (the AT validation gate needs this).
    struct MtWorker {
        VkCommandPool   pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd  = VK_NULL_HANDLE;
        std::unique_ptr<class VulkanCommandBuffer> recorder;
    };
    std::vector<MtWorker> mtWorkers_;
    struct MtInheritance {
        VkFormat   colorFormat = VK_FORMAT_UNDEFINED;
        VkFormat   depthFormat = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{};
        bool       hasColor = true;
    } mtInherit_;
    void EnsureMtWorker(uint32_t threadIndex);

public:
    // Called by the primary VulkanCommandBuffer when it opens a render pass that expects secondaries,
    // so CreateSecondaryCommandBuffer can build matching dynamic-rendering inheritance info. Backend-
    // internal (takes vk* types) — NOT part of the abstract RHI seam.
    void SetSecondaryInheritance(VkFormat colorFormat, VkFormat depthFormat, VkExtent2D extent,
                                 bool hasColor);
};

} // namespace hf::rhi::vk
