# Hazard Forge — Slice B: 3D Core (Spinning Cube)

**Date:** 2026-06-14
**Status:** Self-approved (autonomous session; architect-of-record decisions documented for later review)
**Branch:** `slice-b-3d-cube`

## Goal

Turn the engine from a flat 2D triangle into a real **3D renderer**: a spinning, depth-correct
cube rendered with a model-view-projection transform and a camera. This is the capability leap
that makes Hazard Forge an actual 3D engine.

**Definition of Done:** `hello_cube` (renamed/extended sample) shows a solid, spinning 3D cube
with correct depth occlusion (back faces hidden by front faces), perspective, and a camera
looking at it. Resizes correctly (aspect ratio updates, depth buffer recreates). Zero Vulkan
validation errors. Screenshot-verified.

## Bold decisions (made as architect-of-record)

1. **Own math library (`engine/math/`), not glm.** An engine must own its math to SIMD-optimize
   later (the spec's ARM64/NEON + AVX2 plan) and to control layout/ABI. Cost: we write
   vec3/vec4/mat4/quat + transforms ourselves. Worth it — this is core engine surface, not a
   place to take a dependency. Right-handed, column-major, depth range [0,1] (Vulkan
   convention), Y-flip handled in projection.
2. **MVP via push constants, not descriptor sets.** A single `mat4` (64 bytes, within the
   128-byte guaranteed push-constant limit) pushed per draw. Defers the entire
   descriptor-pool/set-layout machinery to Slice C (textures), where it's actually needed.
   Bold = ship the 3D leap without the heavyweight binding model yet.
3. **Index buffers** added to the RHI now (cubes need them) — extends `BufferDesc` with a usage
   enum and adds `BindIndexBuffer` + `DrawIndexed`.
4. **Depth buffer owned by the device**, sized to the swapchain, recreated on resize, attached
   via dynamic rendering (`VK_FORMAT_D32_SFLOAT`). The RHI gains a depth concept without
   exposing Vulkan.
5. **Time-driven animation in the sample** via `std::chrono` (real wall clock in the compiled
   app — fine; this is the engine, not a workflow script).

## RHI seam extensions (the contract — additive, no breaking changes to Slice A shape)

- `Format`: add `D32_Float`.
- `BufferUsage { Vertex, Index }`; `BufferDesc` gains `BufferUsage usage`.
- `GraphicsPipelineDesc` gains: `bool depthTest = true`, `Format depthFormat = D32_Float`,
  `uint32_t pushConstantSize = 0`.
- `ICommandBuffer` gains: `BindIndexBuffer(IBuffer&)`, `DrawIndexed(uint32_t indexCount)`,
  `PushConstants(const void* data, uint32_t size)`. `BeginRenderPass` now also clears depth.
- `IRHIDevice`: unchanged signature; internally manages the depth image + recreation on resize.

The hard rule still holds: no `vk*` past `engine/rhi/`.

## Math API (`engine/math/math.h` — header-mostly)

`hf::math`: `Vec3`, `Vec4`, `Mat4` (column-major, `float m[16]`), operators, `Mat4::Identity()`,
`Mat4::Perspective(fovYRadians, aspect, near, far)` (Vulkan [0,1] depth, Y-flipped),
`Mat4::LookAt(eye, center, up)`, `Mat4::RotateY(rad)`, `Mat4::RotateX(rad)`,
`Mat4::Translate(Vec3)`, `operator*(Mat4,Mat4)`. Right-handed.

## Vulkan backend deltas (`engine/rhi_vulkan/`)

- Device owns a depth `VkImage`/`VkImageView` (VMA `GPU_ONLY`), created at swapchain size,
  destroyed+recreated in swapchain `Recreate`.
- BeginFrame: transition depth to `DEPTH_ATTACHMENT_OPTIMAL`; command buffer's render-pass
  attaches it with clear (depth=1.0).
- Pipeline: `VkPipelineDepthStencilStateCreateInfo` (test+write, `COMPARE_OP_LESS`),
  depth format in `VkPipelineRenderingCreateInfo.depthAttachmentFormat`, and a
  `VkPushConstantRange` (vertex stage, size from desc) in the pipeline layout.
- Buffer: honor `usage` → add `VK_BUFFER_USAGE_INDEX_BUFFER_BIT` when Index.
- CommandBuffer: `vkCmdBindIndexBuffer` (UINT32), `vkCmdDrawIndexed`, `vkCmdPushConstants`.
- Backface culling stays off for now (cube is solid; depth test does the occlusion) — actually
  enable `CULL_MODE_BACK_BIT` with CCW front faces since we author a proper winding; verify.

## Shaders (`shaders/cube.*.hlsl`)

Vertex: input `float3 pos`, `float3 color`; push constant `float4x4 mvp`; output
`mul(mvp, float4(pos,1))`. Fragment: pass-through vertex color. Cube = 8 corner positions, 36
indices, distinct face/vertex colors so 3D structure is legible.

## Testing

- `rhi_smoke` still passes (device now also creates a depth buffer — exercises that path).
- New `math_test` (CTest, headless, no GPU): asserts identity, perspective/lookat known values,
  matrix multiply associativity, a transformed point. Pure math, fast, deterministic.
- Visual: screenshot of the spinning cube (controller captures it).

## Out of scope for Slice B
Textures/samplers/descriptor sets (Slice C), lighting (Slice D), multiple objects/scene graph,
normals, MSAA. Single cube, vertex-colored, push-constant MVP.
