#include "rhi_metal/metal_device.h"
#include "rhi_metal/metal_command_buffer.h"
#include "rhi_metal/metal_shader.h"
#include "rhi_metal/metal_pipeline.h"
#include "rhi_metal/metal_compute_pipeline.h"
#include "rhi_metal/metal_buffer.h"
#include "rhi_metal/metal_texture.h"
#include "rhi_metal/metal_render_target.h"
#include "rhi_metal/metal_common.h"
#import <QuartzCore/CAMetalLayer.h>
#include <cstring>

// NOTE: the windowed constructor MetalDevice(Window&) and the CreateMetalDevice(Window&) factory
// live in metal_device_windowed.mm — they depend on hal/window.h (SDL) and must NOT be linked into
// the headless (SSH / CI) target, which never references a Window.

namespace hf::rhi::mtl {

MetalDevice::MetalDevice(uint32_t width, uint32_t height) {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) Fail("MTLCreateSystemDefaultDevice returned nil");

    queue_ = [device_ newCommandQueue];
    if (!queue_) Fail("newCommandQueue failed");

    // Headless: offscreen swapchain (owns its own BGRA8 color texture, no CAMetalLayer).
    swapchain_ = std::make_unique<MetalSwapchain>(device_, width, height);

    Init();
}

void MetalDevice::Init() {
    inFlight_ = dispatch_semaphore_create(kFramesInFlight);
    CreateFrameResources();
    recorder_ = std::make_unique<MetalCommandBuffer>(*this);
    rtRecorder_ = std::make_unique<MetalCommandBuffer>(*this);
}

MetalDevice::~MetalDevice() {
    WaitIdle();
    recorder_.reset();
    rtRecorder_.reset();
    swapchain_.reset();
    // ARC releases device_/queue_/uboBuffer_. inFlight_ is an ObjC object under ARC too.
}

void MetalDevice::CreateFrameResources() {
    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        uboBuffer_[i] = [device_ newBufferWithLength:kFrameUboSize
                                             options:MTLResourceStorageModeShared];
        if (!uboBuffer_[i]) Fail("UBO newBuffer failed");
        jointBuffer_[i] = [device_ newBufferWithLength:kJointPaletteSize
                                               options:MTLResourceStorageModeShared];
        if (!jointBuffer_[i]) Fail("joint-palette newBuffer failed");
    }
}

std::unique_ptr<IShaderModule> MetalDevice::CreateShaderModule(const ShaderModuleDesc&) {
    Fail("Metal cannot consume SPIR-V; use CreateShaderModuleMSL (the sample does under __APPLE__)");
}

std::unique_ptr<IShaderModule>
MetalDevice::CreateShaderModuleMSL(const std::string& source, const std::string& entryPoint) {
    return std::make_unique<MetalShaderModule>(device_, source, entryPoint);
}

std::unique_ptr<IPipeline> MetalDevice::CreateGraphicsPipeline(const GraphicsPipelineDesc& d) {
    return std::make_unique<MetalPipeline>(*this, d);
}

std::unique_ptr<IComputePipeline> MetalDevice::CreateComputePipeline(const ComputePipelineDesc& d) {
    return std::make_unique<MetalComputePipeline>(*this, d);
}

std::unique_ptr<IBuffer> MetalDevice::CreateBuffer(const BufferDesc& d) {
    return std::make_unique<MetalBuffer>(device_, d);
}

std::unique_ptr<ITexture> MetalDevice::CreateTexture(const TextureDesc& d) {
    return std::make_unique<MetalTexture>(*this, d);
}

// --- Offscreen render targets (implemented). Mirrors the Vulkan RT pass: render the scene into
// an offscreen color+depth, then commit+wait so a later fullscreen post pass can sample the color
// image. Metal tracks hazards automatically, so there is no explicit layout-transition bookkeeping
// like Vulkan's — commit+waitUntilCompleted is the simplest correct fence for this headless,
// one-shot path. -----------------------------------------------------------------------------

std::unique_ptr<IRenderTarget> MetalDevice::CreateRenderTarget(uint32_t width, uint32_t height) {
    return std::make_unique<MetalRenderTarget>(*this, width, height);
}
std::unique_ptr<IRenderTarget> MetalDevice::CreateRenderTarget(uint32_t width, uint32_t height,
                                                               Format colorFormat) {
    return std::make_unique<MetalRenderTarget>(*this, width, height, /*depthOnly=*/false,
                                               colorFormat);
}

FrameContext MetalDevice::BeginRenderTargetFrame(IRenderTarget& rtBase) {
    auto& rt = static_cast<MetalRenderTarget&>(rtBase);

    // Fresh command buffer for the offscreen pass; the rtRecorder targets the RT's color+depth.
    rtCmd_ = [queue_ commandBuffer];
    if (!rtCmd_) Fail("BeginRenderTargetFrame: commandBuffer failed");

    rtRecorder_->Begin(rtCmd_, rt.colorTexture(), rt.depthTexture(),
                       rt.width(), rt.height());
    return FrameContext{rtRecorder_.get()};
}

void MetalDevice::EndRenderTargetFrame(const FrameContext& frame) {
    if (!frame.cmd || !rtCmd_) return;
    // The caller already issued EndRenderPass() (endEncoding). Commit + wait so the RT color
    // texture is fully written before the swapchain/post pass samples it.
    [rtCmd_ commit];
    [rtCmd_ waitUntilCompleted];
    rtCmd_ = nil;
}

// --- Directional shadow mapping. Mirrors the Vulkan backend: a depth-only sampleable shadow map
// (Depth32Float, RenderTarget|ShaderRead), an offscreen depth-only pass from the light that
// commit+waits so the depth is fully written before the lit pass samples it, and a SetShadowMap
// that records which map the lit pass binds. Metal tracks hazards automatically, so there is no
// explicit layout bookkeeping like Vulkan's. ---------------------------------------------------

std::unique_ptr<IRenderTarget> MetalDevice::CreateShadowMap(uint32_t size) {
    return std::make_unique<MetalRenderTarget>(*this, size, size, /*depthOnly=*/true);
}

// Record which shadow map the lit pass samples. MetalCommandBuffer::BindPipeline binds this map's
// depth texture + clamp-to-edge sampler to the lit fragment shader's shadow slots (texture/sampler
// index 1). Call once after CreateShadowMap.
void MetalDevice::SetShadowMap(IRenderTarget& smBase) {
    shadowMap_ = static_cast<MetalRenderTarget*>(&smBase);
}

// Begin a depth-only pass from the light into the shadow map: a fresh command buffer + an encoder
// with NO color attachment, depth store=store, clear depth=1.0. The depth-only MetalPipeline (built
// with desc.depthOnly) has no fragment function and writes only depth; a modest depth bias on the
// encoder fights shadow acne.
FrameContext MetalDevice::BeginShadowPass(IRenderTarget& smBase) {
    auto& sm = static_cast<MetalRenderTarget&>(smBase);
    shadowCmd_ = [queue_ commandBuffer];
    if (!shadowCmd_) Fail("BeginShadowPass: commandBuffer failed");

    // Depth-only: no color texture (nil) -> MetalCommandBuffer::BeginRenderPass skips the color
    // attachment and stores depth. The recorder samples sm.depthTexture() as the depth attachment.
    rtRecorder_->Begin(shadowCmd_, /*colorTex=*/nil, sm.depthTexture(),
                       sm.width(), sm.height());
    return FrameContext{rtRecorder_.get()};
}

void MetalDevice::EndShadowPass(const FrameContext& frame) {
    if (!frame.cmd || !shadowCmd_) return;
    // The caller already issued EndRenderPass() (endEncoding). Commit + wait so the shadow depth
    // is fully written before the lit pass samples it (Metal's equivalent of the Vulkan fence +
    // DEPTH_ATTACHMENT -> SHADER_READ_ONLY transition).
    [shadowCmd_ commit];
    [shadowCmd_ waitUntilCompleted];
    shadowCmd_ = nil;
}

void MetalDevice::SetFrameUniforms(const void* data, uint32_t size) {
    if (size > kFrameUboSize) Fail("SetFrameUniforms: size exceeds UBO");
    // Shared-storage buffer: contents() is host-visible; the in-flight semaphore guarantees the
    // GPU is no longer reading uboBuffer_[frameIndex_] (see BeginFrame).
    std::memcpy([uboBuffer_[frameIndex_] contents], data, size);
}

void MetalDevice::SetJointPalette(const void* data, size_t size) {
    if (size > kJointPaletteSize) Fail("SetJointPalette: size exceeds UBO");
    // Shared-storage buffer; the in-flight semaphore guarantees the GPU is no longer reading this
    // frame's palette buffer (same pacing as SetFrameUniforms).
    std::memcpy([jointBuffer_[frameIndex_] contents], data, size);
}

void MetalDevice::WaitIdle() {
    // Submit an empty command buffer and wait for completion to drain the queue.
    if (!queue_) return;
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    [cb commit];
    [cb waitUntilCompleted];
}

bool MetalDevice::GetCapturedPixels(std::vector<uint8_t>& out, uint32_t& w, uint32_t& h) {
    if (capturedBGRA_.empty()) return false;
    out = std::move(capturedBGRA_);
    w = capW_;
    h = capH_;
    capturedBGRA_.clear();
    capW_ = capH_ = 0;
    return true;
}

void MetalDevice::ReadBuffer(IBuffer& buffer, void* dst, size_t size, size_t offset) {
    // Slice AR — read back a shared-storage MTLBuffer's bytes (CPU-visible via .contents on Apple
    // Silicon). Call after the GPU work that wrote it has completed (the showcase waits for the frame
    // before reading the GPU-cull instanceCount).
    auto& b = static_cast<MetalBuffer&>(buffer);
    const uint8_t* src = static_cast<const uint8_t*>([b.handle() contents]);
    if (src) std::memcpy(dst, src + offset, size);
}

void MetalDevice::CaptureFromTexture(id<MTLTexture> colorTex) {
    capW_ = (uint32_t)colorTex.width;
    capH_ = (uint32_t)colorTex.height;
    capturedBGRA_.resize((size_t)capW_ * capH_ * 4);
    const NSUInteger bytesPerRow = (NSUInteger)capW_ * 4;
    MTLRegion region = MTLRegionMake2D(0, 0, capW_, capH_);
    // BGRA8Unorm color texture -> tightly-packed BGRA8, top row first (matches Vulkan capture).
    [colorTex getBytes:capturedBGRA_.data()
           bytesPerRow:bytesPerRow
            fromRegion:region
           mipmapLevel:0];
}

FrameContext MetalDevice::BeginFrame() {
    // CPU/GPU pacing: don't get more than kFramesInFlight frames ahead.
    dispatch_semaphore_wait(inFlight_, DISPATCH_TIME_FOREVER);

    if (swapchain_->headless()) {
        // Headless: render straight into the offscreen color texture (no drawable to acquire).
        currentDrawable_ = nil;
        currentColorTex_ = swapchain_->offscreenColor();
    } else {
        currentDrawable_ = swapchain_->AcquireNext();
        if (!currentDrawable_) {
            // No drawable (e.g. zero-sized / occluded). Release the in-flight slot and skip.
            dispatch_semaphore_signal(inFlight_);
            return FrameContext{nullptr};
        }
        currentColorTex_ = currentDrawable_.texture;
    }

    currentCmd_ = [queue_ commandBuffer];

    recorder_->Begin(currentCmd_, currentColorTex_, swapchain_->depthTexture(),
                     swapchain_->width(), swapchain_->height());
    return FrameContext{recorder_.get()};
}

void MetalDevice::EndFrame(const FrameContext& frame) {
    if (!frame.cmd) return;  // skipped frame

    // --- Headless / capture branch: commit + wait, read the color texture back, no present. ---
    // Headless always takes this branch (there is nothing to present); a windowed device only
    // takes it when capture was armed.
    if (swapchain_->headless() || captureArmed_) {
        [currentCmd_ commit];
        [currentCmd_ waitUntilCompleted];

        if (captureArmed_) {
            // On Apple Silicon the color texture is shared/managed (framebufferOnly=NO for the
            // CAMetalLayer drawable, MTLStorageModeShared for the offscreen texture), so getBytes
            // reads it back directly. On a discrete-GPU Mac a drawable may be private — then this
            // needs a blit into a shared staging texture first. We target Apple Silicon.
            CaptureFromTexture(currentColorTex_);
        }

        dispatch_semaphore_signal(inFlight_);
        frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
        currentDrawable_ = nil;
        currentCmd_ = nil;
        currentColorTex_ = nil;
        captureArmed_ = false;
        return;
    }

    // Normal windowed path: present the drawable, signal the in-flight semaphore on GPU completion.
    [currentCmd_ presentDrawable:currentDrawable_];

    dispatch_semaphore_t sem = inFlight_;
    [currentCmd_ addCompletedHandler:^(id<MTLCommandBuffer>) {
        dispatch_semaphore_signal(sem);
    }];
    [currentCmd_ commit];

    frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
    currentDrawable_ = nil;
    currentCmd_ = nil;
    currentColorTex_ = nil;
}

// Headless factory (no window). Lets plain-C++ callers build an offscreen Metal device without
// touching Metal/Obj-C headers; declared extern in metal_offscreen.h.
std::unique_ptr<IRHIDevice> CreateMetalDeviceHeadless(uint32_t width, uint32_t height) {
    return std::make_unique<MetalDevice>(width, height);
}

} // namespace hf::rhi::mtl
