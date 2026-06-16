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

// Backend-agnostic resource ACCESS STATE for the render graph's automatic-barrier solver (Slice AS).
// The graph tracks each render-target image's state and asks the command buffer to transition it via
// ResourceBarrier; the BACKEND maps each state pair to its synchronization primitive. The names
// describe GPU USAGE, not a layout: the Vulkan backend maps them to (VkImageLayout, srcStage/access,
// dstStage/access) inside a vkCmdPipelineBarrier2; Metal's default tracked-hazard model makes the
// transition a no-op. No backend types leak here — this is the pure interface seam.
enum class ResourceState {
    Undefined,    // initial / contents-don't-matter
    ColorTarget,  // color attachment write
    DepthWrite,   // depth attachment write
    DepthRead,    // depth read (reserved)
    ShaderRead,   // sampled in a shader
    Present,      // handed to the presentation engine (terminal)
};

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
    bool usesPerDrawData = false;    // when true (Slice BM): the layout includes a DEDICATED per-draw
                                     // set placed at index 2 (after frame set 0 + material set 1) —
                                     // ONE VERTEX-stage storage buffer holding the PerDraw[ ] array
                                     // (model mat4 + material) the multi-draw-indirect vertex shader
                                     // (lit_mdi.vert) reads as PerDraw[gl_DrawID]. Bound via
                                     // BindPerDrawData. Only the MDI lit pipeline sets this; it pairs
                                     // with usesFrameUniforms + usesTexture so the shared lit fragment
                                     // still samples its base material at set 1. Mirrors the cluster
                                     // set's push-descriptor mechanism but is a VERTEX-stage SSBO.
                                     // Existing pipelines (which leave this false) are byte-for-byte
                                     // unchanged. Mutually exclusive with usesJointPalette/usesEnvironment/
                                     // usesLightClusters (those also claim set 2/3).
    bool usesBindlessTextures = false; // when true (Slice BZ): the layout includes a DEDICATED bindless
                                     // texture set placed at index 4 — ONE unbounded fragment-stage
                                     // sampled-image ARRAY (+ a shared sampler) holding every scene
                                     // texture, which lit_bindless.frag samples as
                                     // gTextures[NonUniformResourceIndex(texIndex)]. Bound via
                                     // BindBindlessTextures. Pairs with usesFrameUniforms + usesTexture
                                     // (set 1 still carries the normal map); sets 2/3 are filled with
                                     // placeholders so the bindless set sits at index 4. Only the
                                     // bindless lit pipeline sets this; existing pipelines (which leave
                                     // it false) are byte-for-byte unchanged.
    bool usesLightClusters = false;  // when true: the layout includes a DEDICATED light-cluster set
                                     // (set 3) — THREE fragment-stage STORAGE buffers (clusters /
                                     // lightIndices / lights) — for clustered Forward+ shading
                                     // (Slice AG). Bound via BindLightClusters. Mirrors usesEnvironment
                                     // (also set 3): sets 1/2 are padded with placeholders so the
                                     // cluster set sits at index 3. Existing set 0/1/2 layouts are
                                     // unchanged, so golden-locked pipelines (which leave this false)
                                     // are byte-for-byte unaffected. Mutually exclusive with
                                     // usesEnvironment (the clustered-lit shader does its own sky IBL).
};

// Storage = read-write SSBO usable by a compute shader (and bindable as a vertex stream so a
// graphics pass can draw the compute-written data without a copy).
// Indirect (Slice AR) = a buffer whose contents are read by the GPU as draw arguments
// (DrawIndexedIndirect). On Vulkan it requests VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT (and is also a
// storage buffer + transfer-dst, so a compute shader can WRITE the args and the host can READ the
// resulting count back); on Metal an indirect-args buffer is just a shared MTLBuffer (no special
// usage), so the flag is a no-op there. This is the minimal additive change: one new enum value.
enum class BufferUsage { Vertex, Index, Uniform, Storage, Indirect };

// Compute pipeline: a single compute shader that reads+writes storageBufferCount storage buffers
// bound at successive bindings (0..N-1), driven by an optional push-constant block (e.g. dt/time).
struct ComputePipelineDesc {
    class IShaderModule* compute = nullptr;
    uint32_t storageBufferCount = 1;  // SSBOs bound at binding 0..count-1
    uint32_t pushConstantSize = 0;    // bytes, compute stage (params like dt/time/count)
    // The shader's [numthreads(X,1,1)] local-workgroup width. Vulkan bakes this into the SPIR-V, so
    // the dispatch only needs the GROUP count; Metal's dispatchThreadgroups takes an explicit
    // threadsPerThreadgroup, so the backend reads this to size the threadgroup. Default 64 matches
    // the GPU-particle kernel; the GPU-cull kernel (Slice AR) sets 1024 (one workgroup, ordered
    // prefix-sum compaction over <=1024 instances). Additive: existing pipelines keep 64.
    uint32_t threadsPerGroupX = 64;
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

// Opaque handle for a BINDLESS texture set (Slice BZ): a single large sampled-image ARRAY holding all
// the scene textures (index i -> the i-th texture passed to CreateBindlessTextureSet), bound ONCE and
// indexed per-draw by a `texIndex` push constant. Vulkan: a descriptor set with a runtime/partially-
// bound sampled-image array + a shared sampler (VK_EXT_descriptor_indexing). Metal: a no-op/fallback
// (the Metal golden renders via the per-material bound path). The vk*/MTL* details live ONLY in the
// backend dirs; the seam exposes only this abstract handle.
class IBindlessTextureSet { public: virtual ~IBindlessTextureSet() = default; };

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
    // Open the render pass declaring that its draws will arrive via SECONDARY command buffers
    // recorded on worker threads (Slice AU — multithreaded recording), NOT inline on this primary.
    // The backend sets the matching "contents = secondary" flag so the validation layer doesn't flag
    // mismatched contents (Vulkan: VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT on
    // vkCmdBeginRendering; Metal: the parallel encoder is opened instead of a normal one). When
    // `expectsSecondaries` is false this is byte-for-byte identical to BeginRenderPass(clear), so the
    // single-threaded path is unchanged. After this, the worker threads record into secondaries from
    // CreateSecondaryCommandBuffer, and the primary replays them via ExecuteSecondaries before
    // EndRenderPass. Default forwards to the inline BeginRenderPass so backends without the parallel
    // path still link; both shipping backends override it.
    virtual void BeginRenderPass(const ClearColor& clear, bool expectsSecondaries) {
        (void)expectsSecondaries;
        BeginRenderPass(clear);
    }
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
    // Bind a baked reflection+irradiance PROBE atlas (Slice AK) at the SAME dedicated environment
    // set/slot BindEnvironment uses (Vulkan set 3 binding 11/12; Metal fragment texture(11)/
    // sampler(12)), for LOCAL cubemap GI in the lit_probe pass. The argument is the RGBA16F render
    // target the probe was baked into (a single atlas holding both the reflection and irradiance
    // blocks). Pair with a pipeline whose usesEnvironment is true. Unlike BindEnvironment, this
    // accepts a render target as the source. Default no-op so passes/backends without probes are
    // unaffected; both shipping backends override it.
    virtual void BindReflectionProbe(ITexture& /*probeAtlas*/) {}
    // Bind the THREE clustered-lighting storage buffers (Slice AG) at the dedicated cluster set
    // (set 3 on Vulkan, bindings 0/1/2; flat fragment buffer slots on Metal), readable by the
    // clustered-lit fragment shader: `clusters` = per-cluster {offset,count}, `lightIndices` = flat
    // light-index array, `lights` = per-light {posRadius, color}. Pair with a pipeline whose
    // usesLightClusters is true. Default no-op so passes/backends without clustered lighting are
    // unaffected; both shipping backends override it.
    virtual void BindLightClusters(IBuffer& /*clusters*/, IBuffer& /*lightIndices*/,
                                   IBuffer& /*lights*/) {}
    // Bind the per-draw STORAGE buffer (Slice BM) at the dedicated per-draw set (set 2 on Vulkan),
    // readable by the multi-draw-indirect VERTEX shader (lit_mdi.vert) as PerDraw[gl_DrawID]: the
    // packed array of {model mat4, float4 material} laid out by render::mdi::BuildBatch, one record
    // per object. Pair with a pipeline whose usesPerDrawData is true; call before
    // DrawIndexedMultiIndirect. Default no-op so passes/backends without MDI are unaffected; both
    // shipping backends override it (Vulkan: push-descriptor SSBO; Metal: a bound vertex buffer slot).
    virtual void BindPerDrawData(IBuffer& /*perDraw*/) {}
    // Bind the BINDLESS texture array (Slice BZ) ONCE at its dedicated set (set 4 on Vulkan), so the
    // following draws can sample any scene texture by INDEX — gTextures[NonUniformResourceIndex(texIndex)]
    // in lit_bindless.frag — with the per-draw `texIndex` arriving via the existing PushConstants (the
    // bindless vertex shader's push constant). The handle comes from CreateBindlessTextureSet. Pair with
    // a pipeline whose usesBindlessTextures is true; call before the bindless draws. Default no-op so
    // passes/backends without bindless are unaffected (Metal no-ops/falls back to the bound path). The
    // vk* descriptor-array bind lives ONLY in the backend dir.
    virtual void BindBindlessTextures(IBindlessTextureSet& /*set*/) {}
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
    // GPU-DRIVEN indexed draw (Slice AR): the draw arguments are READ FROM `argsBuffer` (created with
    // BufferUsage::Indirect) at byte `offset`, NOT passed from the CPU. The buffer holds the standard
    // 5x u32 record {indexCount, instanceCount, firstIndex, vertexOffset, firstInstance} — identical
    // for VkDrawIndexedIndirectCommand and MTLDrawIndexedPrimitivesIndirectArguments. A compute
    // shader writes `instanceCount` (the GPU-decided survivor count) so the number of instances is
    // never round-tripped to the CPU. Bind the per-instance stream (the compute-compacted survivors)
    // via BindInstanceBuffer and the index buffer via BindIndexBuffer first, exactly like
    // DrawIndexedInstanced. Default no-op so backends/passes without indirect draw still link; both
    // shipping backends override it. The vk*/MTL* indirect-draw calls live ONLY in the backend dirs.
    virtual void DrawIndexedIndirect(IBuffer& /*argsBuffer*/, size_t /*offset*/ = 0) {}
    // GPU-DRIVEN MULTI-draw (Slice BM): issue `drawCount` indexed draws in ONE call, each reading its
    // own {indexCount, instanceCount, firstIndex, vertexOffset, firstInstance} record from `argsBuffer`
    // (created with BufferUsage::Indirect) at successive byte offsets `i*stride` (i in [0,drawCount)).
    // The records are the standard 5x u32 (VkDrawIndexedIndirectCommand layout), so the per-draw model
    // matrix + material live in a SEPARATE storage buffer indexed by the per-draw index `gl_DrawID`
    // (SPIR-V DrawIndex). This batches a many-distinct-object scene into a single draw call. Vulkan:
    // vkCmdDrawIndexedIndirect(argsBuffer, 0, drawCount, stride). Metal: MAY no-op/fallback (the Metal
    // golden renders the identical scene via its per-object path — the image is backend-identical, and
    // an MTLIndirectCommandBuffer is optional, not required). Default no-op so backends/passes without
    // MDI still link. Bind the index buffer + the per-vertex stream first, like DrawIndexedIndirect.
    // The vk*/MTL* multi-indirect call lives ONLY in the backend dirs.
    virtual void DrawIndexedMultiIndirect(IBuffer& /*argsBuffer*/, uint32_t /*drawCount*/,
                                          uint32_t /*stride*/) {}
    virtual void PushConstants(const void* data, uint32_t size) = 0;
    // Override the render pass's full-extent scissor for the following draw(s). ImGui needs a
    // per-draw scissor (clip rect). Coordinates are in framebuffer pixels (top-left origin).
    virtual void SetScissor(int32_t x, int32_t y, uint32_t width, uint32_t height) = 0;
    // Restrict the following draw(s) to a sub-rect of the current attachment, by setting BOTH the
    // viewport (so NDC maps into the sub-rect) AND the scissor (so nothing draws outside it). The
    // cascaded-shadow pass (Slice AD) uses this to render each cascade into one tile of a shadow
    // ATLAS: BeginRenderPass clears the whole atlas once, then per cascade SetViewport selects the
    // tile and the casters are drawn with that cascade's lightViewProj. Default no-op so existing
    // passes/backends are byte-for-byte unaffected; both shipping backends override it.
    virtual void SetViewport(int32_t /*x*/, int32_t /*y*/, uint32_t /*width*/, uint32_t /*height*/) {}

    // --- Multithreaded command recording (Slice AU) --------------------------
    // PRIMARY-only: replay the given SECONDARY command buffers (each recorded on a worker thread by
    // ExecuteSecondaries' siblings) into this primary IN THE GIVEN ORDER, inside the render pass
    // opened with BeginRenderPass(clear, /*expectsSecondaries=*/true). The order is the worker-index
    // order, so the final command stream's draw order matches single-threaded recording — that is
    // what makes 1-worker and N-worker renders byte-identical. Vulkan: vkCmdExecuteCommands with the
    // array in order; Metal: closing the parallel encoder commits its sub-encoders in creation order
    // (creation order == worker index), so this is the no-op marker that the parallel encoder ends.
    // The `secondaries` pointers come from CreateSecondaryCommandBuffer. Default no-op so backends
    // without the parallel path still link; both shipping backends override it. All vk*/MTL* calls
    // live inside the backend dirs.
    virtual void ExecuteSecondaries(std::span<ICommandBuffer* const> /*secondaries*/) {}

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

    // --- Render-graph automatic barriers (Slice AS) --------------------------
    // Transition a render-target image from one resource STATE to another. The render graph's barrier
    // solver computes the minimal set of (from->to) transitions and calls this BEFORE each pass that
    // needs the new state (and for the swapchain's terminal ->Present). The BACKEND emits the actual
    // synchronization: Vulkan maps the state pair to a vkCmdPipelineBarrier2 (VkImageMemoryBarrier2
    // with the matching layout + stage/access masks); Metal's default tracked-hazard model auto-orders
    // cross-encoder dependencies, so its implementation is a documented no-op. Default no-op so
    // backends/passes that don't need it still link (must NOT be called inside an open render pass).
    virtual void ResourceBarrier(IRenderTarget& /*resource*/, ResourceState /*from*/,
                                 ResourceState /*to*/) {}
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

    // Create a BINDLESS texture set (Slice BZ): one large sampled-image ARRAY filled with `textures`
    // IN ORDER (index i -> textures[i]) + a shared sampler, bound ONCE via
    // ICommandBuffer::BindBindlessTextures and indexed per-draw by a `texIndex` push constant. Vulkan:
    // a descriptor set with VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT + VARIABLE_DESCRIPTOR_COUNT over
    // runtimeDescriptorArray (VK_EXT_descriptor_indexing), updated once with the textures' views. Metal:
    // returns nullptr (the Metal golden uses the per-material bound path). Default returns nullptr so
    // backends without bindless still link; only the Vulkan backend overrides it.
    virtual std::unique_ptr<IBindlessTextureSet> CreateBindlessTextureSet(
        std::span<ITexture* const> /*textures*/) { return nullptr; }

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

    // Multithreaded recording (Slice AU): vend a SECONDARY command buffer for worker thread
    // `threadIndex` to record draws into, inheriting the CURRENTLY-OPEN render pass's attachment
    // formats/sample-count. The returned recorder is begun (ready for Bind*/Draw*), owned by the
    // device, and valid until the primary's ExecuteSecondaries replays it this frame. Vulkan: a
    // secondary VkCommandBuffer from a PER-THREAD VkCommandPool (pools are not thread-safe — one pool
    // per worker thread, created up front), begun with VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_
    // BIT + VkCommandBufferInheritanceRenderingInfo (dynamic-rendering inheritance: color/depth
    // formats + sample count). Metal: a sub-encoder from the open MTLParallelRenderCommandEncoder
    // (creation order == threadIndex == commit order). Each distinct threadIndex returns a DISJOINT
    // recorder so workers never touch shared mutable state. Default returns nullptr so backends
    // without the parallel path still link; both shipping backends override it. No vk*/MTL* leaks
    // above the seam. Call only between BeginRenderPass(clear,true) and ExecuteSecondaries.
    virtual ICommandBuffer* CreateSecondaryCommandBuffer(uint32_t /*threadIndex*/) { return nullptr; }

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

    // Read `size` bytes back from a GPU buffer (created with BufferUsage::Storage or ::Indirect) at
    // byte `offset` into `dst`. Call AFTER the work that wrote the buffer has completed (the
    // GPU-cull showcase reads the indirect-args instanceCount this way for the exact-count proof,
    // after the capture frame finished). Default no-op so backends without readback still link; both
    // shipping backends override it (Vulkan: host-visible mapped storage; Metal: shared MTLBuffer
    // .contents). NO vk*/MTL* leaks above the seam — the mapping lives inside the backend.
    virtual void ReadBuffer(IBuffer& /*buffer*/, void* /*dst*/, size_t /*size*/,
                            size_t /*offset*/ = 0) {}
};

} // namespace hf::rhi
