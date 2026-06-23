# Issue #34 — RT ray query in a GRAPHICS (fragment) pipeline — design

Expose the ray-tracing acceleration structure as a bindable resource to GRAPHICS pipelines so a FRAGMENT
shader can trace rays (scene-object RT reflections from a raster pass). Today RT works only in COMPUTE.

## Why this is contained + deterministic-suite-safe (confirmed by scout)
The RT moat is STRICT-INTEGER: the HW BVH is only a margin-inflated candidate generator; the shader drains
every candidate via `q.Proceed()` WITHOUT committing/reading the driver float `t`, re-runs the frozen `fx`
`IntersectSphere`/`IntersectAabb`, and folds by a `(t, primIndex)` total order → HW image is BYTE-IDENTICAL
to the CPU `rtrace::` reference, cross-vendor. RT goldens are strict-zero integer goldens. On Metal,
`SupportsHardwareRayQuery()==false` and every RT showcase already runs the CPU `rtrace::` reference to bake
its golden. So #34 ships **Vulkan-only HW** + **Metal CPU-reference**, the new golden is strict-zero, and the
suite stays green with NO Metal RT dependency and NO float bar.

The graphics path is missing ONLY a descriptor wire-up — three proven precedents to copy:
- Compute accel binding: `ComputePipelineDesc::accelStructureBinding` (rhi.h:193) → `vulkan_compute_pipeline.cpp:46-53`
  reserves a `VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR` binding → `VulkanCommandBuffer::BindAccelStructure`
  (vulkan_command_buffer.cpp:385-405) pushes via `vkCmdPushDescriptorSetKHR` at `VK_PIPELINE_BIND_POINT_COMPUTE`.
- Graphics fragment-stage push-descriptor: `BindLightClusters` (vulkan_command_buffer.cpp:250-271) pushes
  graphics-stage SSBOs via `pushDescriptorFn()(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, boundLayout_, …)`.
- Bindless set-4 append: `vulkan_pipeline.cpp:190-191` appends an extra set without perturbing sets 0-3.
- `rayQuery` is enabled DEVICE-WIDE (vulkan_device.cpp:155-164); SPIR-V `RayQueryKHR` + `SPV_KHR_ray_query`
  work in a `ps_6_5` fragment entry with no extra capability (DXC emits identical SPIR-V as `cs_6_5`).

## Implementation plan
1. **rhi.h:** add `int accelStructureBinding = -1;` to `GraphicsPipelineDesc` (mirror the compute field +
   comment). Use a DEDICATED graphics set (e.g. set 5, following the bindless set-4 precedent) so existing
   sets 0-4 layouts are untouched and EVERY existing golden stays byte-identical.
2. **vulkan_pipeline.cpp:** when `desc.accelStructureBinding >= 0`, append a set layout with ONE
   `ACCELERATION_STRUCTURE_KHR` binding at `VK_SHADER_STAGE_FRAGMENT_BIT` + the `PUSH_DESCRIPTOR` flag
   (model on the set-4 bindless append at 190-191 + the device set-layout helpers).
3. **vulkan_command_buffer.cpp:** add a graphics-path accel push (either a new method or branch the existing
   `BindAccelStructure` on whether a graphics vs compute pipeline is bound) that pushes the TLAS at
   `VK_PIPELINE_BIND_POINT_GRAPHICS` against `boundLayout_` (the graphics layout member, distinct from the
   compute `boundComputeLayout_`) — a ~6-line copy of the compute version (394-405) with the bind point +
   layout swapped, exactly as `BindLightClusters` does.
4. **shaders/rt_reflect_graphics.frag.hlsl (NEW):** a fullscreen pass (reuse the `post.vert`/fullscreen-tri
   vertex path). Port the BODY of `shaders/rt_reflect.comp.hlsl` into a `[shader("pixel")]` `ps_6_5` entry:
   copy the candidate-drain + `fx` intersection + `(t,primIndex)` fold + integer reflectivity blend VERBATIM
   (so it stays byte-identical to `rtrace::RenderSceneReflected`); the ONLY edits — compute `px,py` from
   `SV_Position` instead of `SV_DispatchThreadID`, and `return PackRGBA8(...)` as `SV_Target` instead of
   `gImage[]` write. Add to the DXC `ps_6_5 + SPV_KHR_ray_query` build list; do NOT add to the Metal MSL list.
5. **Showcase `--rt-reflect-graphics-shot` (Vulkan, hello_triangle/main.cpp):** build the SAME RT2/RT4 accel
   structure + scene as `--rt4-reflect-shot`, but drive it through a GRAPHICS raster pass (fullscreen
   triangle, the new frag shader, accel structure bound to the graphics pipeline) instead of a compute
   dispatch; read back + `memcmp` HW == CPU `rtrace::RenderSceneReflected` (the identical proof shape as
   `--rt4-reflect-shot`). Print the proof lines (HW==CPU bit-exact, determinism two-run).
   **Metal `--rt-reflect-graphics` (visual_test.mm):** run the CPU `rtrace::RenderSceneReflected` reference
   (one-line copy of the `--rt4-reflect` Metal case). Byte-identical by construction.
6. **Golden:** strict-zero integer `tests/golden/metal/rt_reflect_graphics.png`, baked from the CPU reference
   like every RT golden. Register in `scripts/verify.ps1` `$Goldens` + add `--rt-reflect-graphics-shot` to
   `$vkShots`. (Controller bakes the Metal golden — implementer must NOT commit it.)

## Determinism / golden-invariance
The new accel set is set 5 (or another unused index) → sets 0-4 untouched → all existing goldens byte-
identical. The new frag shader's math is the rt_reflect.comp body verbatim → HW==CPU==golden. The existing
RT shaders/showcases/goldens are BYTE-FROZEN (do not edit them).

## Constraints (HARD)
- Do NOT modify existing RT shaders, the RT compute path, or any existing golden. rhi.h change is additive
  (a new optional field). Use a NEW descriptor set index so existing set layouts are unchanged.
- Branch `fix-issue-34`, commit there, do NOT merge, do NOT touch goldens under `tests/golden/metal/`
  (controller bakes), but DO register the new golden name in verify.ps1.
- Build Vulkan: `cmd /c "<vcvars64> && cmake --build build/windows-msvc-release --target hello_triangle"`
  (vcvars `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`).
  Run `--rt-reflect-graphics-shot` on Windows, confirm exit 0 + the HW==CPU memcmp proof line passes (this is
  THE proof the graphics-pipeline accel binding works AND is deterministic). Build+run any rt ctest.
- Vulkan validation layers must be clean on the new graphics-RT path (the `$vkShots` entry runs it under
  validation) — confirm no validation errors when rendering the shot.

## #35 (Metal HW RT) — DEFERRED (separate note)
Large 4-6 slice effort: a whole Metal `MTLAccelerationStructure` BLAS/TLAS build + bind path (none exists in
engine/rhi_metal/), PLUS re-writing every RT kernel in MSL via `metal::raytracing::intersection_query` with
bit-exact int64 `fx` parity (the HLSL RayQuery kernels are SPIR-V-only). The hardest part is PROVING the MSL
int64 fx math is bit-identical to the CPU/HLSL reference so the strict-zero goldens don't move. Does NOT
block #34. Flag to the user for buy-in as its own flagship sequence. (A `shaders/rt_query.metal` stub hints
at prior exploration — read it before scoping.)

---
## ATTEMPT 1 DIAGNOSIS (2026-06-23) — PARKED, not merged

An implementer built the full feature; the controller gate caught a real bug; the attempt was DISCARDED
(no orphaned branch, master clean). What was learned:

**The CORE #34 capability WORKS.** The RHI plumbing (GraphicsPipelineDesc.accelStructureBinding, the
vulkan_pipeline graphics accel set layout, the vulkan_command_buffer graphics-bind-point BindAccelStructure,
the ps_6_5 rt_reflect_graphics.frag) all compiled and ran: the TLAS traverses from a FRAGMENT shader and
AABB primitives ray-trace correctly (the ground + 2 boxes rendered + shadowed in the graphics-RT image).
So a fragment shader CAN ray-query the accel structure — the issue's core ask is feasible exactly as designed.

**THE BUG: set-3 binding 13 (the FIRST cluster binding) reads WRONG data in the graphics push-descriptor
path; bindings 14 and 15 work.** The showcase bound its 3 scene SSBOs (spheres/aabbs/params) via
`BindLightClusters` to set-3 bindings 13/14/15. Spheres (at 13) rendered as background (garbage geometry:
probed s.cx read small while tag was a valid index and primIndex varied). PROVEN by a swap test: swapping to
aabbs@13 / spheres@14 made the AABBs (ground+boxes) vanish and the SPHERES render correctly — so it is
BINDING 13 (the first `writes[0]` in the cluster `vkCmdPushDescriptorSetKHR`), independent of the buffer.

**Likely root cause (for the next attempt to verify):** an off-by-one / first-write issue in the graphics
cluster push-descriptor, OR the placeholder set-1/set-2 layouts that `usesLightClusters` pushes (materialSetLayout
at set 1, a placeholder at set 2) shift the effective descriptor for the first cluster binding. WORTH CHECKING
whether the shipped clustered-lighting slice (AG) actually exercises binding 13 in practice, or only 14/15 —
binding 13 may have a latent bug never hit before.

**FIX OPTIONS (next attempt):**
1. Fix the RHI graphics cluster push so binding 13 reads correctly (the principled fix — also de-risks the
   clustered-lighting feature). Start by dumping the descriptor writes / validating the set-3 layout vs the
   placeholder sets 1/2 for the usesLightClusters graphics pipeline.
2. AVOID binding 13: bind only 2 cluster SSBOs (spheres@14, aabbs@15, both known-good) and move the small
   `params` struct into the frame UBO (set 0, usesFrameUniforms — proven working in this path); OR pack
   spheres+aabbs into one buffer (raw int/ByteAddressBuffer, manual unpack since GpuSphere/GpuAabb have
   different primIndex offsets) at binding 14 + params at 15.

Option 2 is the faster path to a green gate; option 1 is the better engine fix. The Mac bake + verify.ps1
registration steps from the original spec still apply once spheres render correctly on Vulkan.

---
## ATTEMPT 2 PLAN — avoid binding 13 entirely (DO THIS)

Re-implement the feature. The RHI plumbing from attempt 1 was CORRECT — redo it exactly per the
"Implementation plan" section above (rhi.h `GraphicsPipelineDesc::accelStructureBinding`, the vulkan_pipeline
graphics accel set, the vulkan_command_buffer graphics-bind-point `BindAccelStructure`, the ps_6_5
`rt_reflect_graphics.frag` ported verbatim from rt_reflect.comp). The ONLY change vs attempt 1 is HOW the
scene data reaches the fragment shader — attempt 1's `gParams` at cluster binding 13 read garbage.

**Bind the scene data avoiding binding 13:**
- `gSpheres` → cluster binding **14**, `gAabbs` → cluster binding **15** (both PROVEN working). In the host,
  call `BindLightClusters(dummy, spheresBuf, aabbsBuf)` so the first arg (the broken binding 13) gets a throwaway
  buffer and spheres/aabbs land on 14/15. In the shader: `[[vk::binding(14,3)]] StructuredBuffer<GpuSphere> gSpheres`,
  `[[vk::binding(15,3)]] StructuredBuffer<GpuAabb> gAabbs`.
- `gParams` → the **frame UBO at set 0 binding 0** (the `usesFrameUniforms` path, PROVEN working — boxes were
  positioned correctly from it in attempt 1). The showcase already uploads a DUMMY FrameData via
  `SetFrameUniforms`; instead upload the RT params bytes (eye/right/up/forward/light/plane/counts, ~112 B, well
  under kFrameUboSize 1024). In the shader declare a custom cbuffer matching those bytes:
  `[[vk::binding(0,0)]] cbuffer RtFrame { int4 eye; int4 right; int4 up; int4 forward; int4 light; int4 plane; uint4 counts; };`
  and read params from it instead of `gParams[0]`. (The pipeline's set-0 layout is just a UBO at binding 0; the
  shader interprets the bytes — no RHI change needed.)

This uses only proven-good bindings (14, 15, frame UBO) and never reads cluster binding 13.

**Completion criteria (do NOT stop early):** build Windows, run `--rt-reflect-graphics-shot`, and CONFIRM the
`HW(GRAPHICS)==CPU image BYTE-EQUAL` proof line PRINTS (exit 0) with the SPHERES rendering — i.e. the whole
scene (ground + boxes + 16 spheres + reflections) matches the CPU reference. If spheres are still missing, the
binding is still wrong — fix it before committing. Only commit once the proof passes. Then it's mergeable and
the controller bakes the Metal golden.
