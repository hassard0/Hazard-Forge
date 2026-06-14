# Hazard Forge — Slice C: Textures (Sampling, Staging, Descriptors)

**Date:** 2026-06-14
**Status:** Self-approved (autonomous session)
**Branch:** `slice-c-textures`

## Goal

Give the engine **GPU textures**: a sampled image on the spinning cube. This forces the three
pieces of real material plumbing that Slices A/B deliberately skipped:
1. **Staging upload** — host data → device-local GPU image via a transfer command + layout
   transitions (`UNDEFINED → TRANSFER_DST → SHADER_READ_ONLY`).
2. **Samplers** — a default linear/repeat sampler.
3. **Descriptor sets** — set layout (combined image sampler), pool, allocation, update, bind.

**Definition of Done:** the cube renders **textured** (a procedural checkerboard/UV-grid image
sampled across its faces, modulated by vertex color), spinning, depth-correct, zero validation
errors, screenshot-verified.

## Bold decisions (architect-of-record)

1. **Procedural texture, not an image file.** Generate an RGBA8 checkerboard + UV-gradient in
   code at startup. This exercises the *entire* upload/sample/descriptor path identically to a
   loaded image, but with zero asset-management surface. Image-file decode (`stb_image`) and the
   asset pipeline are their own later slice — don't conflate "can sample a texture" with "has an
   asset system." Bold = ship the GPU capability without the asset detour.
2. **Texture owns its descriptor set.** Each `VulkanTexture` allocates one descriptor set from a
   device pool and pre-updates it (image view + the device's default sampler) at creation.
   `BindTexture` just binds that pre-baked set. Simple, no per-frame descriptor churn.
3. **Device owns the canonical textured-set layout + pool + default sampler.** Both pipeline
   creation (when `usesTexture`) and texture descriptor allocation reference the *same*
   `VkDescriptorSetLayout` so they're guaranteed compatible.
4. **Staging is synchronous (`vkQueueWaitIdle` after the copy).** Fine for load-time uploads in
   this slice; an async transfer queue is a later optimization. Bold = don't build the async
   transfer machinery before there's a frame budget reason to.

## RHI seam extensions (additive)

- `struct TextureDesc { uint32_t width, height; Format format; const void* data; uint64_t dataSize; }`
- `class ITexture { virtual ~ITexture() = default; };`
- `IRHIDevice::CreateTexture(const TextureDesc&) -> std::unique_ptr<ITexture>`
- `GraphicsPipelineDesc` gains `bool usesTexture = false;` (when true, the pipeline layout
  includes the device's textured-set layout at set 0).
- `ICommandBuffer::BindTexture(ITexture&)` — binds the texture's descriptor set at set 0 using the
  currently bound pipeline's layout.

Hard rule still holds: no `vk*` in `engine/rhi/`.

## Vulkan implementation (`engine/rhi_vulkan/`)

- **Device** gains: `VkSampler defaultSampler_` (linear, repeat), `VkDescriptorSetLayout
  texturedSetLayout_` (set 0, binding 0, `COMBINED_IMAGE_SAMPLER`, fragment stage),
  `VkDescriptorPool descriptorPool_` (sized for N textures). Created in ctor, destroyed in dtor
  (before device). Accessors for pipeline + texture to use.
- **Staging helper** on device: `UploadToImage(VkImage, width, height, data, size)` — create
  host-visible staging buffer (VMA), memcpy, one-time command buffer from a transient pool:
  barrier `UNDEFINED→TRANSFER_DST`, `vkCmdCopyBufferToImage`, barrier
  `TRANSFER_DST→SHADER_READ_ONLY_OPTIMAL`, submit, `vkQueueWaitIdle`, free staging.
- **`VulkanTexture`** (new file): VMA device-local `VkImage` (`SAMPLED|TRANSFER_DST`, RGBA8),
  `VkImageView`, allocate+update a descriptor set from the device pool (binding 0 = this view +
  device default sampler). Destroys image/view in dtor (descriptor set freed with pool).
- **Pipeline**: when `desc.usesTexture`, include `device.texturedSetLayout()` in the
  `VkPipelineLayoutCreateInfo.pSetLayouts`. Push-constant range (MVP) coexists. Pipeline needs
  access to the device — pass it (or the layout) into pipeline creation.
- **CommandBuffer** `BindTexture`: `vkCmdBindDescriptorSets(GRAPHICS, boundLayout_, set=0, 1,
  &textureSet, 0, nullptr)`.

## Shaders (`shaders/cube_tex.*.hlsl`)

Vertex gains `float2 uv` input (location 2) → passes uv through. Fragment:
`[[vk::binding(0,0)]] Texture2D tex; [[vk::binding(0,0)]] SamplerState smp;` (combined sampler),
`return tex.Sample(smp, uv) * float4(vertexColor, 1.0);`. Cube vertices gain UVs (per-face 0..1).

## Cube geometry change

Switch from 8 shared vertices to **24 vertices** (4 per face) so each face has proper UVs
(0,0)-(1,1) and a flat-ish per-face tint. 36 indices (6 faces × 2 triangles). Keeps CCW outward.

## Testing

- `rhi_smoke` extended: also `CreateTexture` of a tiny 2×2 image and destroy — exercises the
  staging/descriptor path headlessly. Still must pass.
- `math_test` unchanged, still passes.
- Visual: screenshot of the textured spinning cube (controller).

## Out of scope
Image-file decode / asset pipeline (later slice), mipmaps, anisotropy, multiple textures/materials,
normal maps, async transfer queue. One procedural texture, one sampler, one descriptor set per
texture.
