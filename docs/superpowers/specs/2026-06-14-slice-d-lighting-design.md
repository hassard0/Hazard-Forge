# Hazard Forge — Slice D: Real-Time Lighting (Blinn-Phong + Uniform Buffers)

**Date:** 2026-06-14
**Status:** Self-approved (autonomous session)
**Branch:** `slice-d-lighting`

## Goal

Light the cube. Go from flat texture sampling to **Blinn-Phong shading** with a directional light
+ ambient + specular, so faces visibly brighten toward the light and darken away from it. This
forces the engine's last missing piece of standard material plumbing: **uniform buffers** updated
per frame, and a **frequency-based descriptor model** (per-frame set vs per-material set).

**Definition of Done:** the textured cube renders **lit** — clear directional shading (lit faces
bright, back faces dark), a specular highlight, spinning, depth-correct, zero validation errors,
screenshot-verified (the difference from Slice C must be obvious: shading gradient across faces).

## Bold decisions (architect-of-record)

1. **Frequency-based descriptor sets now.** set 0 = **per-frame** (camera+light UBO), set 1 =
   **per-material** (texture+sampler). This is the standard real-engine layout and the right time
   to adopt it (textures from Slice C move from set 0 → set 1). Bold = refactor the binding model
   to the correct shape now rather than accrete hacks.
2. **Per-frame-in-flight uniform buffers.** The device owns `kFramesInFlight` host-visible mapped
   UBOs, each with its own pre-baked descriptor set, cycled by frame index — so the CPU never
   writes a UBO the GPU is still reading. This is the correct double-buffered UBO pattern.
3. **Model matrix stays a push constant; everything per-frame goes in the UBO.** Push = `mat4
   model` (64B). UBO = `viewProj` + light + camera (≤128B). Shader computes `clip = viewProj *
   model * pos`. Keeps push constants small and per-object, UBO per-frame. (Defers per-object
   material/instance data — that's Slice E.)
4. **Normals via `(float3x3)model`.** Cubes use uniform scale, so the model's upper-3×3 transforms
   normals correctly without a separate normal matrix. Bold = don't build the inverse-transpose
   normal-matrix machinery before non-uniform scale actually exists.

## RHI seam extensions (additive)

- `BufferUsage`: add `Uniform`.
- `GraphicsPipelineDesc`: add `bool usesFrameUniforms = false;` (when true, pipeline layout includes
  the device per-frame set layout at set 0). `usesTexture` now maps the material set to **set 1**.
- `IRHIDevice::SetFrameUniforms(const void* data, uint32_t size)` — copies into the current
  frame-in-flight's UBO; the matching descriptor set (set 0) is auto-bound for subsequent draws.
- `ICommandBuffer::BindTexture` now binds the material set at **set index 1** (was 0).

Hard rule holds: no `vk*` in `engine/rhi/`.

## Vulkan implementation

- **Device** gains: per-frame `VkBuffer uboBuffer_[kFramesInFlight]` (host-visible mapped, VMA) +
  `void* uboMapped_[kFramesInFlight]`; `VkDescriptorSetLayout frameSetLayout_` (set 0, binding 0 =
  `UNIFORM_BUFFER`, vertex+fragment stage); `VkDescriptorSet frameSet_[kFramesInFlight]` (one per
  buffer, pre-updated to point at its UBO). Material set layout from Slice C becomes **set 1**.
  Pool sizing adds `UNIFORM_BUFFER` capacity. Created in ctor, destroyed in dtor (order: after
  swapchain reset, before allocator).
- `SetFrameUniforms`: `memcpy(uboMapped_[frameIndex_], data, size)`; record the current frame set
  so the recorder binds it. Simplest: in `BeginFrame`, hand the recorder the current
  `frameSet_[frameIndex_]`; the recorder binds set 0 on `BindPipeline` **iff** the pipeline has a
  frame set (pipeline carries a `bool hasFrameSet`). `SetFrameUniforms` just writes the memory
  (the set already points at that persistent buffer, so no re-bind needed beyond the per-pipeline
  bind).
- **Pipeline**: build `pSetLayouts` from {frameSetLayout (set0) if usesFrameUniforms,
  materialSetLayout (set1) if usesTexture}. Note set indices: if usesFrameUniforms && usesTexture,
  layouts array = [frameSetLayout, materialSetLayout] (sets 0 and 1). If only one, it occupies set 0
  — but for this slice the cube uses BOTH, so the array is [frame, material]. Carry `hasFrameSet` on
  the pipeline so the recorder knows to bind set 0.
- **CommandBuffer**: on `BindPipeline`, if pipeline `hasFrameSet`, `vkCmdBindDescriptorSets(set 0,
  1, &frameSet_)`. `BindTexture` binds at `firstSet = 1`.
- **Buffer**: `BufferUsage::Uniform` → `VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT` (used internally for the
  device UBOs; the host-visible mapped path already exists).

## Shaders (`shaders/lit.*.hlsl`)

Vertex input gains `float3 normal` (location 3). UBO at set 0 binding 0:
```
struct FrameData { float4x4 viewProj; float4 lightDir; float4 lightColor; float4 viewPos; };
```
Vertex: `world = mul(model, float4(pos,1)); clip = mul(viewProj, world);` pass world pos,
`worldNormal = normalize(mul((float3x3)model, normal))`, uv, color. Model comes via push constant.
Fragment (set 1 texture): Blinn-Phong —
`N=normalize(worldNormal); L=normalize(-lightDir.xyz); diff=max(dot(N,L),0);
V=normalize(viewPos.xyz-worldPos); H=normalize(L+V); spec=pow(max(dot(N,H),0),32);
ambient=0.15; rgb = tex.rgb*color*( (ambient+diff)*lightColor.rgb ) + spec*lightColor.rgb*0.4;`

## Cube geometry change

24 vertices now include a per-face **normal** (±X/±Y/±Z). Vertex =
`{pos[3], color[3], uv[2], normal[3]}` (stride 44). Layout adds location 3 = `RGB32_Float` @ 32.

## Sample

Each frame: build `viewProj = proj*view` (camera orbiting or fixed at {2.5,2.5,4}); set
`FrameData{viewProj, lightDir=(-0.5,-1,-0.3), lightColor=(1,1,1,1), viewPos}` via
`SetFrameUniforms`. Spin model = `RotateY(t)*RotateX(t*0.5)`, push it. Light is fixed in world
space so shading sweeps across faces as the cube spins (clearly visible). Title "Hazard Forge —
Lit Cube".

## Testing

- `rhi_smoke` extended: device creation now also builds per-frame UBOs + frame sets (exercised on
  construct/teardown). Add a `SetFrameUniforms` call with a dummy 112-byte struct to exercise the
  copy path. Still must pass.
- `math_test` unchanged.
- Visual: screenshot showing directional shading + specular (controller).

## Out of scope
Multiple objects / instancing (Slice E), point/spot lights, shadow maps, normal mapping, PBR,
inverse-transpose normal matrix (uniform scale only). One directional light, one lit cube.
