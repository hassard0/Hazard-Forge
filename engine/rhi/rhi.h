#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

// Forward declarations of HAL + handle types — NO vulkan headers in this file.
struct VkInstance_T;  using VkInstance = VkInstance_T*;
struct VkSurfaceKHR_T; using VkSurfaceKHR = VkSurfaceKHR_T*;

namespace hf::hal { class Window; }

namespace hf::rhi {

enum class Backend { Vulkan, Metal };

enum class Format {
    Undefined,
    RGBA8_UNorm,
    BGRA8_UNorm,
    RG32_Float,    // vec2 vertex attribute
    RGB32_Float,   // vec3 vertex attribute
    RGBA32_Float,  // vec4 vertex attribute (skinned joints/weights)
    RGBA16_Float,  // half4 sampled HDR texture (equirect environment map; Slice R)
    D32_Float,     // depth attachment
};

struct ClearColor { float r = 0, g = 0, b = 0, a = 1; };

// --- Resource descriptors ----------------------------------------------------

struct VertexAttribute {
    uint32_t location;   // shader input location
    Format   format;     // attribute element format
    uint32_t offset;     // byte offset within a vertex
};

struct VertexLayout {
    uint32_t stride = 0;                       // bytes per vertex
    std::vector<VertexAttribute> attributes;   // per-attribute layout
};

struct ShaderModuleDesc {
    std::span<const uint32_t> spirv;  // SPIR-V words
};

struct GraphicsPipelineDesc {
    class IShaderModule* vertex = nullptr;
    class IShaderModule* fragment = nullptr;
    VertexLayout vertexLayout;
    // Depth-WRITE control. Independent of depthTest. When depthTest is on, this decides whether
    // passing fragments also WRITE the depth buffer. Default true reproduces the historical
    // behavior (depth-tested pipelines also wrote depth), so EVERY existing pipeline is byte-for-byte
    // unchanged. Set false for the translucent pass (Slice T): glass still depth-TESTS against the
    // opaque scene (so opaque geometry in front occludes it) but does NOT write depth, so overlapping
    // translucent surfaces blend correctly and never self-occlude. Ignored when depthTest is false.
    bool depthWrite = true;
    // Optional SECOND, per-instance vertex stream (binding 1, input rate = per-instance). When its
    // `attributes` is non-empty the pipeline declares a second vertex binding fed by
    // BindInstanceBuffer; binding 0 stays the per-vertex `vertexLayout`. Empty (the default) means
    // no instancing — identical to all existing pipelines. The instanced lit pipeline puts a mat4
    // per-instance transform here (4x RGBA32_Float at locations 7-10, stride 64).
    VertexLayout instanceLayout;
    Format colorFormat = Format::BGRA8_UNorm;  // must match swapchain format
    bool depthTest = true;
    Format depthFormat = Format::D32_Float;
    uint32_t pushConstantSize = 0;   // bytes, vertex stage
    bool usesFrameUniforms = false;  // when true, layout includes the per-frame UBO set (set 0)
    bool usesTexture = false;        // when true, layout includes the material set (set 1)
    bool usesJointPalette = false;   // when true, layout includes the skinning joint-palette set (set 2)
    bool fullscreen = false;         // when true: no vertex input (fullscreen triangle from SV_VertexID)
    bool fragmentPushConstants = false;  // when true: the push-constant range is visible to the
                                     // FRAGMENT stage (in addition to vertex), so a fullscreen
                                     // fragment pass (e.g. the bloom chain, Slice U) can read its
                                     // per-pass params from a push constant. Default false keeps
                                     // the range vertex-only, so every existing pipeline's layout
                                     // is byte-for-byte unchanged.
    bool depthOnly = false;          // when true: no color attachment; depth write + bias (shadow pass)
    bool pointList = false;          // when true: POINT_LIST topology (GPU particles drawn as points)
    bool lineList = false;           // when true: LINE_LIST topology (immediate-mode debug-draw —
                                     // grids, AABB/OBB wireframes, gizmos, normals; Slice W). Each
                                     // pair of vertices is one line segment. Default false keeps the
                                     // existing point/triangle selection so every golden-locked
                                     // pipeline is byte-for-byte unchanged. Mutually exclusive with
                                     // pointList (line wins if both are somehow set).
    bool additiveBlend = false;      // when true: additive color blend (glowing particles over scene)
    bool alphaBlend = false;         // when true: standard src_alpha/one_minus_src_alpha blend (UI/ImGui)
    bool cullNone = false;           // when true: no back-face culling (ImGui draws CW-wound quads)
    bool pbrMaterial = false;        // when true: set 1 is the WIDER full-PBR material set (5 textures:
                                     // base/metalRough/normal/emissive/occlusion) instead of the 2-texture
                                     // material set. Only the lit-PBR pipeline sets this; all other
                                     // material pipelines keep the unchanged 2-texture set.
    bool usesEnvironment = false;    // when true: the layout includes a DEDICATED environment set
                                     // (set 3) — one sampled HDR image + sampler — for image-based
                                     // lighting (Slice R). Bound via BindEnvironment. Existing
                                     // set 0/1/2 layouts are unchanged, so golden-locked pipelines
                                     // (which leave this false) are byte-for-byte unaffected.
};

// Storage = read-write SSBO usable by a compute shader (and bindable as a vertex stream so a
// graphics pass can draw the compute-written data without a copy).
enum class BufferUsage { Vertex, Index, Uniform, Storage };

// Compute pipeline: a single compute shader that reads+writes storageBufferCount storage buffers
// bound at successive bindings (0..N-1), driven by an optional push-constant block (e.g. dt/time).
struct ComputePipelineDesc {
    class IShaderModule* compute = nullptr;
    uint32_t storageBufferCount = 1;  // SSBOs bound at binding 0..count-1
    uint32_t pushConstantSize = 0;    // bytes, compute stage (params like dt/time/count)
};

struct TextureDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    Format format = Format::RGBA8_UNorm;
    const void* data = nullptr;   // tightly packed pixels (mip 0 when mipLevels==1)
    uint64_t dataSize = 0;        // bytes
    // --- Optional N-mip 2D texture (Slice R: HDR environment with a CPU-prefiltered mip chain). ---
    // When mipLevels > 1 the backend creates an N-mip sampled image and uploads each mip from
    // mipData[i] (tightly packed, dimensions max(1, width>>i) x max(1, height>>i)). When mipLevels
    // == 1 (the default) the existing single-mip data/dataSize path is used verbatim, so RGBA8
    // textures are byte-for-byte unchanged. The mip sampler uses trilinear filtering (linear mipmap).
    uint32_t mipLevels = 1;
    const void* const* mipData = nullptr;  // [mipLevels] tightly-packed per-mip pixel pointers
    bool environment = false;       // when true: address U = repeat (equirect longitude wraps),
                                    // address V = clamp (poles); used for the HDR equirect env map.
};

struct BufferDesc {
    uint64_t size = 0;          // bytes
    const void* initialData = nullptr;  // optional upload-on-create
    BufferUsage usage = BufferUsage::Vertex;
};

// --- Resource interfaces -----------------------------------------------------

class IShaderModule    { public: virtual ~IShaderModule() = default; };
class IPipeline        { public: virtual ~IPipeline() = default; };
class IComputePipeline { public: virtual ~IComputePipeline() = default; };
class IBuffer          { public: virtual ~IBuffer() = default; };
class ITexture         { public: virtual ~ITexture() = default; };

// A sampleable offscreen color image (+ its own depth) you render into. Inheriting ITexture
// lets the post pass bind it via the existing ICommandBuffer::BindTexture.
class IRenderTarget : public ITexture {
public:
    virtual uint32_t width() const = 0;
    virtual uint32_t height() const = 0;
};

// Records draw commands for one frame's swapchain image.
class ICommandBuffer {
public:
    virtual ~ICommandBuffer() = default;
    virtual void BeginRenderPass(const ClearColor& clear) = 0;
    virtual void BindPipeline(IPipeline& pipeline) = 0;
    virtual void BindVertexBuffer(IBuffer& buffer) = 0;
    // Bind a per-instance vertex stream to binding 1 (paired with a pipeline whose instanceLayout is
    // non-empty). Default no-op so non-instancing backends/passes are unaffected; both shipping
    // backends override it. Must be called before DrawIndexedInstanced.
    virtual void BindInstanceBuffer(IBuffer& /*buffer*/) {}
    virtual void BindIndexBuffer(IBuffer& buffer) = 0;
    virtual void BindTexture(ITexture& texture) = 0;
    // Bind TWO sampled images at once into a single material set: `primary` at the base slot
    // (binding 0/1) and `secondary` at the second material slot (binding 3/4) — the same two slots
    // BindMaterial uses for base+normal. The HDR bloom composite (Slice U) uses this to sample the
    // HDR scene (primary) and the bloom result (secondary) in one fullscreen pass. Both arguments
    // may be render targets. Default forwards to BindTexture(primary) so passes/backends that do not
    // implement the second slot still bind the primary image.
    virtual void BindTexturePair(ITexture& primary, ITexture& /*secondary*/) { BindTexture(primary); }
    // Bind a material: base-color texture at the material slot AND a tangent-space normal map at the
    // second material slot (material set binding 2/3 on Vulkan; Metal texture(3)/sampler(4)). The lit
    // pass uses this so a normal map can perturb the shading normal. Default forwards to BindTexture
    // (base only) for backends/passes that do not implement the normal-map slot.
    virtual void BindMaterial(ITexture& base, ITexture& normalMap) { (void)normalMap; BindTexture(base); }
    // Bind a full glTF metallic-roughness PBR material: five textures bound as one material set —
    // base-color, metallic-roughness (G=roughness, B=metallic), tangent-space normal, emissive, and
    // ambient-occlusion (R). The lit-PBR pass uses this. Default forwards to BindMaterial(base,
    // normal) so passes/backends without the wider set still bind the base + normal.
    virtual void BindMaterialPBR(ITexture& base, ITexture& metalRough, ITexture& normalMap,
                                 ITexture& emissive, ITexture& occlusion) {
        (void)metalRough; (void)emissive; (void)occlusion;
        BindMaterial(base, normalMap);
    }
    // Bind an HDR environment map (equirectangular, N-mip) at the dedicated environment set/slot
    // (set 3 on the Vulkan backend; flat fragment texture/sampler slots 11/12 on the other), for
    // image-based lighting in the sky_hdr + lit_pbr_ibl passes (Slice R). Pair with a pipeline whose
    // usesEnvironment is true. Default no-op so passes/backends without IBL are unaffected; both
    // shipping backends override it.
    virtual void BindEnvironment(ITexture& /*env*/) {}
    virtual void Draw(uint32_t vertexCount, uint32_t firstVertex = 0) = 0;
    // `vertexOffset` is added to every index before vertex fetch (ImGui draws share one combined
    // vertex+index buffer per cmd-list and offset into it). Defaults to 0 for the existing scene draws.
    virtual void DrawIndexed(uint32_t indexCount, uint32_t firstIndex = 0,
                             int32_t vertexOffset = 0) = 0;
    // Instanced indexed draw: draws `instanceCount` copies of the indexed geometry, each fed a
    // distinct per-instance attribute set from the buffer bound via BindInstanceBuffer (binding 1).
    // gl_InstanceIndex/[[instance_id]] selects the per-instance record. Default forwards to a single
    // DrawIndexed (instanceCount ignored) so backends that do not implement instancing still link;
    // both shipping backends override it.
    virtual void DrawIndexedInstanced(uint32_t indexCount, uint32_t instanceCount,
                                      uint32_t firstIndex = 0, int32_t vertexOffset = 0,
                                      uint32_t firstInstance = 0) {
        (void)instanceCount; (void)firstInstance;
        DrawIndexed(indexCount, firstIndex, vertexOffset);
    }
    virtual void PushConstants(const void* data, uint32_t size) = 0;
    // Override the render pass's full-extent scissor for the following draw(s). ImGui needs a
    // per-draw scissor (clip rect). Coordinates are in framebuffer pixels (top-left origin).
    virtual void SetScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) = 0;
    virtual void EndRenderPass() = 0;

    // --- Compute recording (must be OUTSIDE a render pass) -------------------
    // Bind a compute pipeline; subsequent BindStorageBuffer / ComputePushConstants / DispatchCompute
    // target it. Default no-op so backends that lack compute do not need to override.
    virtual void BindComputePipeline(IComputePipeline& /*pipeline*/) {}
    // Bind a Storage buffer at the compute pipeline's binding `index` (0-based).
    virtual void BindStorageBuffer(IBuffer& /*buffer*/, uint32_t /*index*/ = 0) {}
    // Push constants for the bound compute pipeline (compute stage).
    virtual void ComputePushConstants(const void* /*data*/, uint32_t /*size*/) {}
    // Dispatch `groupsX*groupsY*groupsZ` workgroups of the bound compute pipeline.
    virtual void DispatchCompute(uint32_t /*groupsX*/, uint32_t groupsY = 1, uint32_t groupsZ = 1) {}
    // Barrier so a later vertex stage reads the storage buffer the compute stage just wrote.
    virtual void ComputeToVertexBarrier() {}
};

class ISwapchain {
public:
    virtual ~ISwapchain() = default;
    virtual Format ColorFormat() const = 0;
    // Recreate after a window resize (or out-of-date acquire/present).
    virtual void Recreate(uint32_t width, uint32_t height) = 0;
};

// Per-frame handle returned by BeginFrame; passed back to EndFrame.
struct FrameContext {
    ICommandBuffer* cmd = nullptr;  // null if the frame was skipped (e.g. minimized)
};

class IRHIDevice {
public:
    virtual ~IRHIDevice() = default;

    virtual ISwapchain& Swapchain() = 0;

    virtual std::unique_ptr<IShaderModule> CreateShaderModule(const ShaderModuleDesc&) = 0;
    virtual std::unique_ptr<IPipeline> CreateGraphicsPipeline(const GraphicsPipelineDesc&) = 0;
    // Compute pipeline for GPU-driven work (e.g. a particle simulation). Backends without compute
    // support return nullptr by default.
    virtual std::unique_ptr<IComputePipeline> CreateComputePipeline(const ComputePipelineDesc&) {
        return nullptr;
    }
    virtual std::unique_ptr<IBuffer> CreateBuffer(const BufferDesc&) = 0;
    virtual std::unique_ptr<ITexture> CreateTexture(const TextureDesc&) = 0;

    // Offscreen render target: a sampleable color image (swapchain format) + its own depth.
    virtual std::unique_ptr<IRenderTarget> CreateRenderTarget(uint32_t width, uint32_t height) = 0;

    // Offscreen render target with an EXPLICIT color format (Slice U: an HDR RGBA16_Float target
    // for the bloom chain). Format::Undefined selects the swapchain color format, so this is
    // identical to the 2-arg overload above (which delegates here with Undefined) — existing call
    // sites are byte-for-byte unchanged. RGBA16_Float yields a renderable + sampleable half-float
    // HDR color image (COLOR_ATTACHMENT | SAMPLED); the depth image stays D32.
    virtual std::unique_ptr<IRenderTarget> CreateRenderTarget(uint32_t width, uint32_t height,
                                                              Format colorFormat) = 0;

    // Depth-only sampleable shadow map (size x size, D32, DEPTH_STENCIL_ATTACHMENT | SAMPLED).
    // Rendered into via Begin/EndShadowPass; sampled by the lit pass via the per-frame set (set 0).
    virtual std::unique_ptr<IRenderTarget> CreateShadowMap(uint32_t size) = 0;

    // Begin recording a depth-only pass from the light into the shadow map (dynamic rendering,
    // no color attachment). Returns a FrameContext whose cmd records depth-only draws.
    virtual FrameContext BeginShadowPass(IRenderTarget& shadowMap) = 0;
    // End recording, submit, and transition the shadow map depth to SHADER_READ_ONLY so the lit
    // pass can sample it. Blocks until the shadow pass completes.
    virtual void EndShadowPass(const FrameContext&) = 0;

    // Point the per-frame sets' shadow-map binding (set 0, bindings 1+2) at this shadow map.
    // Call once after CreateShadowMap; the lit pipeline then samples it every frame.
    virtual void SetShadowMap(IRenderTarget& shadowMap) = 0;

    // Begin recording the scene into the render target's color+depth (dynamic rendering).
    // Returns a FrameContext whose cmd records into the RT; pair with EndRenderTargetFrame.
    virtual FrameContext BeginRenderTargetFrame(IRenderTarget& rt) = 0;
    // End recording, submit, and transition the RT color to SHADER_READ_ONLY so a later pass
    // can sample it. Blocks until the RT pass completes (so the swapchain pass sees the result).
    virtual void EndRenderTargetFrame(const FrameContext&) = 0;

    // Acquire next swapchain image + begin command recording.
    // Returns FrameContext{nullptr} when the frame must be skipped this tick.
    virtual FrameContext BeginFrame() = 0;
    // Submit recorded commands and present.
    virtual void EndFrame(const FrameContext&) = 0;

    // Copy per-frame uniform data into the current frame-in-flight's UBO. Call after
    // BeginFrame (so the current frame index is set) and before recording draws; the
    // matching per-frame descriptor set (set 0) is auto-bound on BindPipeline.
    virtual void SetFrameUniforms(const void* data, uint32_t size) = 0;

    // Copy the joint-matrix palette into the current frame-in-flight's palette UBO. Call after
    // BeginFrame (so the current frame index is set) and before recording skinned draws; the
    // matching joint-palette descriptor set (set 2) is auto-bound on BindPipeline when the bound
    // pipeline declared usesJointPalette. `data` is a tightly-packed array of column-major
    // float4x4 (<= 64 matrices = 4096 bytes). Default no-op so non-skinning backends/passes are
    // unaffected; both shipping backends override it.
    virtual void SetJointPalette(const void* /*data*/, size_t /*size*/) {}

    // Block until the GPU is idle (call before destroying GPU resources).
    virtual void WaitIdle() = 0;

    // Headless capture: arm before BeginFrame; after EndFrame, retrieve via GetCapturedPixels.
    virtual void CaptureNextFrame() = 0;
    // Returns the last captured frame as tightly-packed BGRA8 (top row first); false if none.
    virtual bool GetCapturedPixels(std::vector<uint8_t>& outBGRA,
                                   uint32_t& width, uint32_t& height) = 0;
};

} // namespace hf::rhi
