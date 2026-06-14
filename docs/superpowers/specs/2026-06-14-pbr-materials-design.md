# Slice P — Full PBR glTF material import (design)

Date: 2026-06-14
Branch: `slice-pbr-materials`

## Goal

Import the **full glTF metallic-roughness material** (base color + metallic-roughness +
tangent-space normal + emissive + ambient occlusion textures) from a `.glb` and render it through a
new lit-PBR pipeline, without disturbing the existing golden-locked `lit` pipeline / scene. Showcase
with `DamagedHelmet.glb` (Khronos sample asset, CC-BY): 1 mesh / 1 primitive, 5 JPEG textures, no
authored TANGENT.

## What already exists (baseline)

* `lit.frag.hlsl` already does Cook-Torrance metallic-roughness + procedural sky IBL + directional
  PCF shadows + point lights + tangent-space normal mapping. Material set 1 holds base-color
  (binding 0/1) + normal map (binding 3/4). `BindMaterial(base, normal)` drives it.
* `gltf_loader.cpp` already decodes the embedded base-color image (buffer_view → stb_image RGBA8),
  reads metallic/roughness factors, computes tangents (Lengyel + Gram-Schmidt) when TANGENT is
  absent, and recentres geometry to origin.
* Vulkan material set layout lives in `VulkanDevice::CreateTextureResources`; `VulkanTexture` owns
  one descriptor set. Metal binds textures by flat index from `metal_common.h`.

## Decisions

### 1. Loader → full PBR material

Add `LoadPbrGltfModel(device, path) -> PbrModel`:

```
struct PbrModel {
    scene::Mesh mesh;
    std::unique_ptr<ITexture> baseColor, metalRough, normalMap, emissive, occlusion;
    float metallicFactor, roughnessFactor;
    float emissiveFactor[3];
};
```

* Reuse `BuildMesh` (recentre to origin, compute tangents from POSITION/UV/NORMAL — already handles
  the missing-TANGENT case the helmet needs).
* Decode each present texture via the existing embedded-image → stb_image path (generalised
  `DecodeImageTexture` helper, factored out of `LoadBaseColorTexture`).
* For any ABSENT texture, supply a 1×1 fallback so the shader always binds 5 textures:
  * baseColor → white `(255,255,255,255)`
  * metalRough → `(255,255,0,255)` — neutral: shader reads G(rough)=1, B(metallic)=0 (both then
    scaled by factors). R/A unused.
  * normalMap → flat `(128,128,255,255)` → decodes to `(0,0,1)` (no perturbation)
  * emissive → black `(0,0,0,255)` (adds nothing)
  * occlusion → white `(255,255,255,255)` (R=1 → no occlusion)
* Factors default per glTF spec: metallic=1, roughness=1, emissive=(0,0,0). Read the glTF metallic /
  roughness factors and `emissive_factor`.

### 2. RHI material binding for 5 textures

* New `GraphicsPipelineDesc::pbrMaterial` flag: when set, the pipeline declares the **PBR material
  set** (a *separate, wider* set-1 layout) instead of the existing 2-texture material layout, so the
  existing lit/normal-map pipelines are byte-for-byte unaffected.
* New `ICommandBuffer::BindMaterialPBR(base, metalRough, normal, emissive, occlusion)` (default
  forwards to `BindMaterial(base, normal)` for backends/passes that don't implement it). Existing
  `BindMaterial` stays.
* **Vulkan:** new `pbrMaterialSetLayout_` with 10 bindings (5 sampled-image + 5 sampler), and a new
  `VulkanPbrMaterial` RHI texture-set object that owns all five image views + one descriptor set and
  writes them. `BindMaterialPBR` binds that single set at the material set index. Bindings chosen so
  `--msl-decoration-binding` maps them to the Metal indices below:
  * base       image b0  / sampler b1
  * normal     image b3  / sampler b4   (matches existing convention)
  * metalRough image b5  / sampler b6
  * emissive   image b7  / sampler b8
  * occlusion  image b9  / sampler b10
* **Metal:** add binding constants `kFragMetalRoughTex=5`, `kFragMetalRoughSmp=6`,
  `kFragEmissiveTex=7`, `kFragEmissiveSmp=8`, `kFragOcclusionTex=9`, `kFragOcclusionSmp=10` and
  implement `BindMaterialPBR` (binds the 5 image+sampler pairs at those indices). The Metal pipeline
  has no fixed descriptor layout, so no `pbrMaterial`-specific pipeline change is needed there.

### 3. Shader — `lit_pbr.frag.hlsl` (separate file, leaves `lit.frag.hlsl` untouched)

Copy of `lit.frag` + the extra material textures. Per glTF metallic-roughness packing:

* baseColor (sRGB) `× vertex color` → albedo
* metalRough: `metallic = B × metallicFactor`, `roughness = G × roughnessFactor` (linear)
* normal map: tangent-space TBN perturb (reuse existing approach)
* occlusion: `R` multiplies the ambient/IBL term only (not direct lights — matches glTF AO intent)
* emissive: `emissiveTexture.rgb × emissiveFactor`, ADDED after lighting

Vertex shader reuses `lit.vert.hlsl`. The metalRough/emissive/occlusion textures bind to the new
HLSL `[[vk::binding(...,1)]]` slots (5/6, 7/8, 9/10). MSL generated via the existing
glslc→spirv-cross toolchain (`HF_MSL_GEN` guards), entry `pbr_fragment`.

### 4. Showcase + verification (NEW golden; existing goldens & main scene untouched)

* Vulkan: `hello_triangle.exe --pbr-shot <path>` — minimal showcase (ground plane + skybox + the
  DamagedHelmet with full PBR, lit + shadowed, fixed camera/light), one frame → 32-bit BMP. Helmet
  placed with explicit model matrix: rotate −90° about X (asset is Z-up), uniform scale + translate
  to sit on the plane.
* New CMake define `HF_HELMET_MODEL_PATH` (both targets) → `assets/models/DamagedHelmet.glb`.
* Metal: same showcase behind `--pbr` in `visual_test.mm`; new golden
  `tests/golden/metal/pbr_helmet.png` IF the Mac is reachable (else "Metal golden pending").

## Seam

No new `vk*`/`MTL`/`Metal` tokens in `engine/rhi`, `engine/scene`, `engine/math`, `engine/asset`,
`engine/render` (master baseline 12, must stay 12). The loader/`rhi.h` additions avoid those tokens.

## DON'T

Do not modify the existing lit/scene pipeline, its material set layout, or any existing golden.
PBR is an additive, separate pipeline + a separate wider material set + a new golden.
