# Slice U — HDR Bloom (design)

Date: 2026-06-14
Branch: `slice-bloom`

## Goal
Add a true HDR bloom post-effect: render the IBL helmet showcase into an HDR (RGBA16F)
render target, run a threshold → progressive-downsample → tent-upsample bloom chain on
half-res HDR RTs, then composite the bloom additively into the HDR scene and apply the
existing tonemap/exposure/grade pipeline while writing the LDR swapchain. The existing LDR
post path, all existing pipelines/shaders, and all existing goldens stay byte-identical.

## Hard constraints
- RHI seam stays clean: `grep -rnE "vk[A-Z]|MTL|Metal" engine/{rhi,scene,math,physics,asset,render}` == 12 (baseline). Zero new tokens.
- Existing `CreateRenderTarget(w,h)` 2-arg overload keeps swapchain-format behavior byte-identical.
- No changes to `post.frag`, the LDR RT, or any committed golden.

## RHI changes
1. `IRHIDevice::CreateRenderTarget(w, h, Format colorFormat)` — new 3-arg overload.
   The existing 2-arg overload delegates with `colorFormat = <swapchain format>` sentinel
   (`Format::Undefined` → backend substitutes the swapchain/BGRA8 format, preserving exact
   prior behavior). When `RGBA16_Float`, the color image is created
   `VK_FORMAT_R16G16B16A16_SFLOAT` / `MTLPixelFormatRGBA16Float` with COLOR_ATTACHMENT|SAMPLED.
   Depth stays D32. Both backends.
2. `GraphicsPipelineDesc.fragmentPushConstants` (bool, default false). When true the Vulkan
   push-constant range covers VERTEX|FRAGMENT so a fullscreen fragment shader can read push
   constants; pipelines leave it false (unchanged). The bound pipeline reports its push-constant
   stage flags so `ICommandBuffer::PushConstants` pushes to the right stage(s). Default path
   (false) is vertex-only — byte-identical to today.

   - Render passes are format-agnostic at record time in both backends (Vulkan dynamic rendering;
     Metal reads the attachment's own pixel format), so only the RT constructor + pipeline
     `colorFormat` need the HDR format.

## Shaders (HLSL → SPIR-V via DXC; MSL via glslc→spirv-cross with HF_MSL_GEN guards)
All fullscreen (`post.vert` reused), `usesTexture=true`, `fragmentPushConstants=true`.
Push-constant block (fragment): `{ float2 texel; float threshold; float knee; float strength; float intensity; }`.

- `bloom_prefilter.frag.hlsl` — bright-pass with a soft knee around luma≈1.0 (Karis/COD curve),
  keeps HDR energy above threshold, writes half-res HDR.
- `bloom_downsample.frag.hlsl` — 13-tap Jimenez/COD dual-filter downsample (partial Karis average
  on the brightest group at the first mip to tame fireflies), builds the mip chain.
- `bloom_upsample.frag.hlsl` — 3×3 tent filter; additively blended onto the finer mip
  (RHI additiveBlend) so each coarser mip accumulates up the chain.
- `bloom_composite.frag.hlsl` — sample HDR scene + final bloom mip (× strength), apply the SAME
  exposure(1.7)/ACES/gamma/ColorGrade/grain/vignette as `post.frag`, write LDR swapchain.

## Bloom chain
- Scene RT: full-res RGBA16F. Bloom mips: 5 levels, each `CreateRenderTarget(w>>i, h>>i, RGBA16F)`
  starting at half-res (i=1..5). Prefilter scene→mip1. Downsample mip(k)→mip(k+1). Upsample
  additively mip(k+1)→mip(k) down to mip1. Composite scene+mip1→swapchain.
- Params: threshold 1.0, knee 0.6, bloom strength 0.06 (composite mix), upsample additive.
  Tuned so the HDR sun (radiance ~75k) and the emissive cyan gauge (>1) bloom with a soft halo
  while the rest of the frame stays sharp (no global milky haze).

## Showcase + verification
- Vulkan `hello_triangle --bloom-shot <bmp>`: clone of `--ibl-shot` but scene/sky pipelines use
  `colorFormat = RGBA16_Float`, render into the HDR RT, then the bloom chain + composite replace
  the LDR post pass. Capture → BMP → PNG → visual inspection (sun + gauge bloom, rest sharp).
- Metal `visual_test --bloom <png>`: same showcase; new golden `tests/golden/metal/bloom.png`
  (two runs DIFF 0.0000). Existing 7 goldens unchanged.

## Non-goals / deviations
- No compute; all passes are fullscreen graphics (matches existing post infrastructure).
- Bloom chain is driven imperatively via Begin/EndRenderTargetFrame in sequence (the RenderGraph
  maps one external write per pass to SceneColor scaffolding; the many-RT linear chain is clearer
  imperatively, and the final swapchain pass still uses the graph's capture-retry arm path).
