# Slice I: Shadow Mapping (Directional) — Design + Plan

**Status:** Self-approved (autonomous). subagent-driven. Builds on Slices A–H on `master`.

## Goal
The directional light casts **real shadows**: a depth pass from the light renders the scene into a
shadow map; the lit pass samples it (with PCF) to darken shadowed fragments. The cubes cast shadows
on the ground and each other. Verifiable headlessly via `--shot` (shadows visible).

## Bold decision
Use the Slice G render-target machinery for a **depth-only sampleable shadow map**. Manual depth
compare + 3×3 PCF in the shader (no comparison sampler — simpler, portable). Single directional
light, single 2048² map. Light-space matrix via a new `Mat4::Ortho`.

## The descriptor change (the crux)
The shadow map is **per-frame** data, so it belongs in the **frame set (set 0)** alongside the UBO.
`frameSetLayout_` grows from 1 binding to 3:
- binding 0: `UNIFORM_BUFFER` (existing FrameData UBO)
- binding 1: `SAMPLED_IMAGE` (shadow depth)
- binding 2: `SAMPLER` (shadow sampler — a plain clamp-to-border/edge linear sampler)
Each `frameSet_[i]` is updated to point at the shadow map's depth view + a shadow sampler when
`SetShadowMap` is called. Material set (set 1) unchanged.

UBO grows: add `float4x4 lightViewProj` to FrameData → 224 + 64 = **288 bytes**. So **bump
`kFrameUboSize` 256 → 512** (engine one-liner; future-proofs UBO growth).

## Tasks

### I1: math — `Mat4::Ortho`
engine/math/math.h: add
```cpp
static Mat4 Ortho(float l, float r, float b, float t, float zn, float zf) {
    Mat4 m;  // zeros; column-major, Vulkan [0,1] depth, Y-flip to match Perspective
    m.m[0]  = 2.0f / (r - l);
    m.m[5]  = -2.0f / (t - b);                 // Y-flip (matches Perspective's -1/tan)
    m.m[10] = -1.0f / (zf - zn);
    m.m[12] = -(r + l) / (r - l);
    m.m[13] = (t + b) / (t - b);               // sign paired with the Y-flip
    m.m[14] = -zn / (zf - zn);
    m.m[15] = 1.0f;
    return m;
}
```
math_test: assert Ortho maps center of the box to ~0 in x/y and zn→0,zf→1 in z for a sample point. Commit `feat(math): orthographic projection`.

### I2: RHI seam (rhi.h, additive)
- `IRHIDevice::CreateShadowMap(uint32_t size) -> std::unique_ptr<IRenderTarget>` (depth-only, sampleable).
- `IRHIDevice::BeginShadowPass(IRenderTarget& sm) -> FrameContext` / `EndShadowPass(const FrameContext&)`.
- `IRHIDevice::SetShadowMap(IRenderTarget& sm)` — point the per-frame sets' shadow binding at this map.
- `GraphicsPipelineDesc::depthOnly = false` (when true: no color attachment, depth write + bias, vertex stage only).

### I3: Vulkan
- **Shadow map** = a `VulkanRenderTarget` variant or a flag: a `D32` image with usage
  `DEPTH_STENCIL_ATTACHMENT | SAMPLED`, a depth view, and a descriptor set is NOT needed on the RT
  itself (the frame set samples it). Expose `depthView()` + track layout. Add a `bool depthOnly`
  ctor path to VulkanRenderTarget (no color image). `CreateShadowMap` builds one at size×size.
- **frameSetLayout_**: add bindings 1 (SAMPLED_IMAGE, fragment) + 2 (SAMPLER, fragment). Create a
  `shadowSampler_` (linear, addressMode CLAMP_TO_EDGE, optionally compareEnable false). Pool sizes:
  add SAMPLED_IMAGE + SAMPLER counts for the frame sets (≥ kFramesInFlight each).
- `SetShadowMap(rt)`: for each frameSet_[i], `vkUpdateDescriptorSets` binding1 = rt depth view
  (layout SHADER_READ_ONLY) + binding2 = shadowSampler_.
- `BeginShadowPass(sm)`: use the existing dedicated rt command buffer/pool/fence (or a second). Begin
  one-time cmd; transition sm depth UNDEFINED→DEPTH_ATTACHMENT; begin dynamic rendering with ONLY a
  depth attachment (no color: colorAttachmentCount=0), loadOp CLEAR depth=1.0, set viewport/scissor
  to size×size; return a recorder bound to it (depthView, no colorView).
- `EndShadowPass`: end rendering, transition sm depth DEPTH_ATTACHMENT→SHADER_READ_ONLY, submit +
  fence-wait. (Reuse rtFence_ pattern.)
- **Depth-only pipeline:** when `desc.depthOnly`: rendering create info has 0 color attachments,
  `depthAttachmentFormat = D32`; no fragment shader (or a trivial one); enable depth bias
  (`rasterizationState.depthBiasEnable = VK_TRUE`, constant 1.25, slope 1.75) to fight acne; cull
  front faces OR keep back-cull (keep back-cull, rely on bias). Vertex input = the scene vertex
  layout; the shadow vertex shader outputs only clip = lightViewProj*model*pos.
- The command buffer BeginRenderPass must support a depth-only pass (colorView_ may be null → skip
  the color attachment). Handle null colorView_ in BeginRenderPass.

### I4: shaders
- `shaders/shadow.vert.hlsl`: input pos(loc0) [+ unused others ok]; push constant model; cbuffer
  Frame at set0 b0 (need lightViewProj); `clip = mul(lightViewProj, mul(model, float4(pos,1)))`.
  (Depth-only: no fragment shader needed — but DXC/pipeline may require one; include a trivial
  `shadow.frag.hlsl` returning nothing/void or a dummy. Simplest: a frag that writes nothing — use
  an empty `void main(){}`. If the pipeline requires a fragment stage, provide a no-op.)
- Update `lit.vert.hlsl`/`lit.frag.hlsl` FrameData to add `float4x4 lightViewProj;` (after the point
  light arrays — keep C++/HLSL layout in sync; total 288B).
- `lit.frag.hlsl`: add shadow sampling. Declare at set0: `[[vk::binding(1,0)]] Texture2D gShadow;`
  `[[vk::binding(2,0)]] SamplerState gShadowSmp;`. Compute:
  ```
  float4 lp = mul(f.lightViewProj, float4(i.wpos, 1.0));
  float3 proj = lp.xyz / lp.w;
  float2 smUV = proj.xy * 0.5 + 0.5;   // [0,1]; note Vulkan Y already handled by Ortho Y-flip → verify, may need (proj.y* -0.5+0.5)
  float curDepth = proj.z;
  float shadow = 1.0;
  if (smUV.x>=0 && smUV.x<=1 && smUV.y>=0 && smUV.y<=1 && curDepth<=1.0) {
      float bias = 0.0025;
      float s = 0.0; float texel = 1.0/2048.0;
      for (int x=-1;x<=1;x++) for (int y=-1;y<=1;y++) {
          float d = gShadow.Sample(gShadowSmp, smUV + float2(x,y)*texel).r;
          s += (curDepth - bias > d) ? 0.0 : 1.0;
      }
      shadow = s/9.0;
  }
  ```
  Apply `shadow` to the directional diffuse+spec (NOT ambient, NOT point lights):
  `rgb = tex*ambient*f.lightColor.rgb + shadow*(tex*diff*f.lightColor.rgb + spec*f.lightColor.rgb*0.4);`
  then add point lights as before.
  > The shadow-map UV Y-orientation and depth convention are the #1 risk — the controller will
  > screenshot; if shadows are inverted/offset, adjust smUV.y flip and bias.
- samples CMakeLists: compile shadow.vert/shadow.frag.

### I5: sample
- `Mat4 lightViewProj`: build an ortho box covering the scene from the light. lightDir = (-0.5,-1,-0.3)
  normalized; eye = sceneCenter - lightDir*12; `LookAt(eye, sceneCenter, {0,1,0})`;
  `Ortho(-8,8,-8,8, 1, 25)`. `lightVP = ortho * lookAt`. Put into FrameData.lightViewProj.
- Create the shadow map once: `auto shadowMap = device->CreateShadowMap(2048);` and
  `device->SetShadowMap(*shadowMap);` after creation.
- Per frame (both --shot and interactive), BEFORE the scene RT pass: a shadow pass —
  `auto sc = device->BeginShadowPass(*shadowMap); device->SetFrameUniforms(&fd,sizeof(fd));`
  (fd already has lightViewProj) `sc.cmd->BeginRenderPass(clear)`; for each renderable: bind the
  SHADOW pipeline (a depthOnly pipeline using shadow.vert), PushConstants(model), bind vbuf/ibuf,
  DrawIndexed (no texture bind needed). `sc.cmd->EndRenderPass(); device->EndShadowPass(sc);`
  Then the existing scene→RT pass (lit pipeline now samples the shadow map) then post pass.
  > drawScene currently binds the lit pipeline + textures. For the shadow pass, add a
  > `drawDepthOnly(cmd)` that binds the shadow pipeline and draws geometry only.
- Create the shadow pipeline: `depthOnly=true, usesFrameUniforms=true` (needs lightViewProj from set0
  UBO), `usesTexture=false`, `pushConstantSize=64`, vertexLayout = scene layout, depthTest=true.
- Title "Hazard Forge — Shadows".
- rhi_smoke: `auto sm = device->CreateShadowMap(256); device->SetShadowMap(*sm);` then drop.

## Verify
- Build clean; ctest (rhi_smoke + math_test) pass.
- `--shot "$env:TEMP\hf_i.bmp"` exit 0, BMP ~3.6MB. (Controller views it: cubes cast shadows on the
  ground/each other from the directional light.)
- grep: no vk in engine/rhi/ or engine/scene/.

## Commit
`feat(rhi+vulkan): directional shadow mapping (depth pass + PCF)`

## Risks (flag for the controller's screenshot)
- Shadow UV Y-flip + depth convention (Ortho Y-flip vs Vulkan): shadows may be offset/inverted →
  adjust smUV.y and bias. - Acne/peter-panning: tune depthBias + the 0.0025 shader bias. - Null
  colorView_ handling in BeginRenderPass for the depth-only pass. - kFrameUboSize bump to 512.
