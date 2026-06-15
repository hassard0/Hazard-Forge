# Slice R — HDR environment IBL (design)

Date: 2026-06-14
Branch: `slice-hdr-ibl`

## Goal
Replace the procedural `SkyColor()` image-based lighting with a real HDR equirectangular
environment map: the helmet's metal reflects the actual captured sky/sun/terrain, and the skybox
shows the real HDR sky. Scope is **equirectangular 2D + mip-LOD prefilter approximation** — NO
cubemaps, NO render-to-cube. Existing pipelines/shaders/goldens stay byte-unchanged; this slice adds
SEPARATE `sky_hdr` + `lit_pbr_ibl` shaders + pipelines + a new golden.

## RHI changes (both backends; existing RGBA8 single-mip path untouched)
1. **`Format::RGBA16_Float`** added to `rhi.h` (Vulkan `VK_FORMAT_R16G16B16A16_SFLOAT`, Metal
   `MTLPixelFormatRGBA16Float`). Added to ToVk / ToMetalPixelFormat.
2. **`TextureDesc` gains `uint32_t mipLevels = 1` and `const void* const* mipData = nullptr`** — an
   optional array of per-mip tightly-packed pixel pointers. When `mipLevels > 1` the backend creates
   an N-mip 2D image and uploads each mip from `mipData[i]`. When `mipLevels == 1` (default), the
   existing single-mip `data`/`dataSize` path is used verbatim — RGBA8 textures are byte-unchanged.
   The HDR sampler uses linear min/mag + linear mipmap (trilinear), addressU = repeat (equirect wraps
   in longitude), addressV = clamp (poles).
3. **Environment binding via a DEDICATED set that does not perturb set 0/1/2.**
   `GraphicsPipelineDesc.usesEnvironment` (default false). On Vulkan the env image+sampler live at a
   dedicated **set 3** (binding 0 = sampled image, binding 1 = sampler), declared in HLSL as
   `[[vk::binding(0,3)]]` / `[[vk::binding(1,3)]]`. The pipeline-layout array only appends set 3 when
   `usesEnvironment`; the lit-PBR pipeline that uses IBL declares set 1 (PBR material) + set 3 (env),
   leaving the set-1 PBR material layout identical. On Metal the env binds at flat fragment
   `texture(11)/sampler(12)` (past the PBR material's 0..10), chosen so `--msl-decoration-binding`
   maps set-3 bindings there.
   `ICommandBuffer::BindEnvironment(ITexture& env)` binds the env set/texture. Default no-op.
4. The env texture exposes its descriptor set (Vulkan) / texture+sampler (Metal) via a small
   dedicated path. Vulkan: the HDR texture owns a set-3 descriptor set built from a new
   `environmentSetLayout_`. Metal: BindEnvironment binds texture(11)/sampler(12) directly.

## Environment loader (`engine/asset/env_loader.{h,cpp}`, in hf_core)
`LoadHdrEnvironment(device, path) -> EnvironmentMap { unique_ptr<ITexture> equirect; int mipLevels; int width; int height; }`.
- Read file bytes, decode with `stbi_loadf` (linear float RGB, HDR range > 1). STBI_NO_STDIO is set
  in gltf_loader's TU, but env_loader is its OWN TU and reads the file into a buffer then calls
  `stbi_loadf_from_memory`.
- Build the mip chain on the CPU by repeated 2x2 box-downsample in float (each coarser mip = a
  blurrier environment → approximates roughness-prefiltered specular). `mipLevels = floor(log2(max(w,h)))+1`.
- Convert each mip's float RGB → RGBA16F (A=1) with a `float32 -> float16` helper, upload all mips.

## Shaders (NEW files; lit_pbr.frag / sky.frag / lit.frag are golden-locked, untouched)
- `shaders/sky_hdr.frag.hlsl`: like sky.frag but samples the equirect env at LOD 0 for the
  background. `dir → uv: u = atan2(dir.z, dir.x)/(2π)+0.5, v = acos(clamp(dir.y,-1,1))/π`. Outputs
  linear HDR radiance (the post pass does exposure + ACES); do NOT pre-tonemap.
- `shaders/lit_pbr_ibl.frag.hlsl`: a copy of lit_pbr.frag where the IBL term samples the env map
  instead of `SkyColor`. specular = `env.SampleLevel(equirectUV(reflect(-V,N)), roughness*maxLod) * F`;
  diffuse irradiance = `env.SampleLevel(equirectUV(N), maxLod-1) * albedo * (1-metallic)`. Direct
  lights / shadow / normal-map / emissive / occlusion identical to lit_pbr.frag.
  `maxLod` is passed via `FrameData.skyParams.z` (an unused spare component — the existing layout is
  unchanged, only a previously-zero slot is populated for the IBL pass).

## Showcase + verification
- Vulkan `hello_triangle --ibl-shot <path>`: HDR equirect skybox (`sky_hdr`) + DamagedHelmet shaded
  by `lit_pbr_ibl` (+ ground), fixed camera/light, one frame → BMP. New golden only; existing
  scene/pipelines untouched.
- CMake `HF_ENV_PATH` define → `assets/env/env.hdr` (both targets). Compile `sky_hdr.frag` +
  `lit_pbr_ibl.frag` (Vulkan SHADERS list; Metal gen list).
- Metal `visual_test --ibl <out.png>` mirrors the showcase; new golden `tests/golden/metal/ibl_helmet.png`.

## Invariants
- Seam grep (`vk[A-Z]|MTL|Metal` in rhi/scene/math/asset/render) stays at 12.
- `ctest --preset windows-msvc-debug` stays green (8/8).
- Existing goldens (scene_shadow/skinning/pbr_helmet/instanced) byte-unchanged.
