# Metal golden-image test

## `debug_viz.png` — immediate-mode debug visualization (Slice W)

`debug_viz.png` is a SEPARATE golden produced by `visual_test --debug`: the SAME settled physics
sphere-pyramid scene as `--physics` (ground checkerboard + procedural sky + the 30-body pile, lit +
shadowed), with an immediate-mode **debug-draw overlay** rendered on top — a ground grid, a colored
wireframe **AABB** hugging each settled body, a per-body **wire sphere**, a directional-**light
arrow**, and **physics contact** markers (crosses + normal stubs). The overlay is built CPU-side by
the pure-C++ `hf::debug::DebugDraw` collector (each shape decomposes to LINE_LIST segments) and drawn
as ONE non-indexed `Draw` through a new debug-line pipeline.

New RHI: `GraphicsPipelineDesc.lineList` (default false) → `VK_PRIMITIVE_TOPOLOGY_LINE_LIST` on
Vulkan / `MTLPrimitiveTypeLine` selected at draw time on Metal. The debug pipeline is
`lineList=true, usesFrameUniforms=true, depthTest=true, depthWrite=false`, so the lines are correctly
**occluded** by opaque geometry in front but never fight the depth buffer; it draws AFTER the opaque
spheres in the scene pass. The line MSL is generated from the shared HLSL (`shaders/debug_line.{vert,
frag}.hlsl`) by the build toolchain (`debug_line.vert.gen.metal` entry `debug_line_vertex`,
`debug_line.frag.gen.metal` entry `debug_line_fragment`). `lineList` defaults false so every
pre-existing pipeline is byte-for-byte unchanged: all nine other goldens still diff `0.0000`.
Deterministic — two `--debug` runs diff `0.0000`. This golden is INDEPENDENT of all the others.

Re-bake / validate the same way as the others:

```sh
./build-metal/visual_test --debug /tmp/dbg.png
~/mac-remote-rig/compare.sh tests/golden/metal/debug_viz.png /tmp/dbg.png 0.0   # -> DIFF 0.0000
```

---

## `bloom.png` — HDR bloom (Slice U)

`bloom.png` is a SEPARATE golden produced by `visual_test --bloom`: the same HDR-IBL helmet scene as
`--ibl`, but rendered into an **HDR `RGBA16_Float` render target** so highlights keep values >1, then
run through a true bloom chain — a soft-knee **threshold** bright-pass, a 5-level progressively
half-res **downsample** mip chain (13-tap COD/Jimenez dual filter), a 3×3 tent-filter
**upsample/combine** back up, and a **composite** that adds the bloom and applies the *same*
exposure/ACES/grade/grain/vignette as `post.frag`, writing the LDR swapchain. The HDR sun and the
helmet's emissive cyan gauge bloom (soft halo); the rest of the frame stays sharp.

New RHI: `IRHIDevice::CreateRenderTarget(w, h, Format)` (the 2-arg overload delegates with
`Format::Undefined` → swapchain format, byte-for-byte unchanged); `GraphicsPipelineDesc.fragmentPushConstants`
(widens the push-constant range to VERTEX|FRAGMENT so the fullscreen bloom passes read per-pass params
in the fragment stage — default false leaves every existing pipeline's layout unchanged); and
`ICommandBuffer::BindTexturePair(primary, secondary)` (binds two sampled images into one material set —
base slot + the second/normal slot — so the composite samples HDR scene + bloom together). On Metal the
RT uses `MTLPixelFormatRGBA16Float`; the bloom MSL is generated from the shared HLSL (`bloom_*.frag.hlsl`)
and the fragment push constant binds at `kFbPushConst=1`. Deterministic — two `--bloom` runs diff
`0.0000`. This golden is INDEPENDENT of all the others, which are unchanged (all still diff `0.0000`).

Re-bake / validate the same way as the others:

```sh
./build-metal/visual_test --bloom /tmp/bloom.png
~/mac-remote-rig/compare.sh tests/golden/metal/bloom.png /tmp/bloom.png 0.0   # -> DIFF 0.0000
```

---

## `ibl_helmet.png` — HDR environment IBL (Slice R)

`ibl_helmet.png` is a SEPARATE golden produced by `visual_test --ibl`: a minimal HDR
image-based-lighting showcase (HDR **equirectangular** skybox + ground checkerboard plane + the
DamagedHelmet shaded so its metal reflects the **real captured sky/sun/terrain**, lit + shadowed,
fixed camera/light). The environment is `assets/env/env.hdr` (a CC0 clear-sky 1k equirect), decoded
with `stbi_loadf` and uploaded as an **N-mip `RGBA16_Float`** texture: each coarser mip is a CPU 2×2
box-downsample (progressively blurrier → approximates a roughness-prefiltered specular environment).
The new shaders `shaders/sky_hdr.frag.hlsl` (background, LOD 0) and `shaders/lit_pbr_ibl.frag.hlsl`
(specular = env at `roughness*maxLod`, diffuse irradiance = env at a very high mip) sample the env
via the equirect mapping `u = atan2(z,x)/2π+0.5, v = acos(y)/π`. The env binds on a DEDICATED slot
(`usesEnvironment` → flat fragment `texture(11)/sampler(12)`, `metal_common.h kFragEnvTex/kFragEnvSmp`)
via `ICommandBuffer::BindEnvironment`, so the existing set 0/1/2 layouts and the golden-locked
`lit`/`sky`/`lit_pbr` pipelines are byte-for-byte unchanged. The frame is deterministic — two runs
diff `0.0000`. This golden is INDEPENDENT of `scene_shadow.png` / `skinning.png` / `pbr_helmet.png` /
`instanced.png`, which are all unchanged (still diff `0.0000`).

Re-bake / validate the same way as the others:

```sh
./build-metal/visual_test --ibl /tmp/ibl.png
~/mac-remote-rig/compare.sh tests/golden/metal/ibl_helmet.png /tmp/ibl.png 0.0   # -> DIFF 0.0000
```

---

## `instanced.png` — GPU instanced rendering (Slice Q)

`instanced.png` is a SEPARATE golden produced by `visual_test --instanced`: a minimal GPU-instancing
showcase (ground checkerboard plane + procedural sky + a **12×12 = 144 field of spheres** drawn in a
single `ICommandBuffer::DrawIndexedInstanced` call, lit + shadowed, fixed elevated camera/light).
Each instance is placed by its own per-instance model matrix supplied through a SECOND, per-instance
vertex stream (RHI binding 1): four `RGBA32_Float` attributes at locations 7–10 carry the columns of
a column-major `float4x4` (`scene::InstanceTransformLayout`, stride 64), bound via
`BindInstanceBuffer`. The transforms come from the deterministic `scene::BuildInstanceGrid` (NO RNG —
a pure function of the instance index: grid x/z + a per-index sin height bob + a per-index Y spin), so
the frame is golden-stable — two runs diff to `0.0000`. The instanced MSL is generated from the
shared HLSL (`shaders/lit_instanced.vert.hlsl` + the unchanged `lit.frag`; the field's shadows use
`shaders/shadow_instanced.vert.hlsl`) by the build toolchain (`lit_instanced.vert.gen.metal`, entry
`instanced_vertex`); the per-instance attributes bind at Metal vertex buffer slot `kVbInstance = 4`
(`metal_common.h`), past vertex(0)/frameUbo(1)/pushConst(2)/jointPalette(3), with the vertex
descriptor's `kVbInstance` layout set to `MTLVertexStepFunctionPerInstance`. This golden is
INDEPENDENT of `scene_shadow.png` / `skinning.png` / `pbr_helmet.png`, which are unchanged (all still
diff `0.0000`) because instancing is an additive separate pipeline + vertex binding.

Re-bake / validate the same way as the others:

```sh
./build-metal/visual_test --instanced /tmp/inst.png
~/mac-remote-rig/compare.sh tests/golden/metal/instanced.png /tmp/inst.png 0.0   # -> DIFF 0.0000
```

---

## `pbr_helmet.png` — full glTF PBR material (Slice P)

`pbr_helmet.png` is a SEPARATE golden produced by `visual_test --pbr`: a minimal full-PBR showcase
(ground checkerboard plane + procedural sky + the **DamagedHelmet** glTF model, lit + shadowed,
fixed camera/light). The helmet renders the complete glTF metallic-roughness material set — base
color (sRGB), metallic-roughness (G=roughness, B=metallic), tangent-space normal map, emissive, and
ambient occlusion — through a dedicated **lit-PBR pipeline** (`shaders/lit_pbr.frag.hlsl`, shared
`lit.vert`) and a wider 5-texture material set bound via `ICommandBuffer::BindMaterialPBR`. The
material textures are decoded from the embedded `.glb` images by the shared
`asset::LoadPbrGltfModel`; tangents are computed from POSITION/UV/NORMAL (the asset has no TANGENT).
The MSL is generated from the same HLSL by the build toolchain (`lit_pbr.frag.gen.metal`, entry
`pbr_fragment`); the PBR map indices come from `metal_common.h` (`kFragMetalRoughTex=5`,
`kFragEmissiveTex=7`, `kFragOcclusionTex=9`, + their samplers). Deterministic — two runs diff to
`0.0000`. This golden is INDEPENDENT of `scene_shadow.png` / `skinning.png`, which are unchanged
(both still diff `0.0000`) because the PBR path is an additive separate pipeline + material set.

Re-bake / validate the same way as the others:

```sh
./build-metal/visual_test --pbr /tmp/pbr.png
~/mac-remote-rig/compare.sh tests/golden/metal/pbr_helmet.png /tmp/pbr.png 0.0   # -> DIFF 0.0000
```

---

`scene_shadow.png` is the golden reference produced by the **real** Metal RHI backend running
headless on an Apple M4 (`metal_headless/visual_test`), rendering the full scene (ground
checkerboard plane + a 3×3 grid mixing 3 shiny metal spheres on the main diagonal with 6 matte
dielectric cubes — the per-material PBR showcase, matching the Vulkan `hello_triangle` scene)
through the complete frame pipeline:
**directional shadow pass → scene into an offscreen render target → fullscreen post (FXAA + glow +
ACES tonemap + cinematic grade + film grain + vignette)**. Deterministic — fixed camera, fixed
light, static transforms, offscreen `MTLTexture` target, byte-exact readback; two runs diff to
`0.0000`.

It exercises the full Metal parity surface through the `IRHIDevice` / `ICommandBuffer` seam — the
same calls the Vulkan `hello_triangle` sample makes: `CreateShadowMap` / `BeginShadowPass` /
`EndShadowPass`, `CreateRenderTarget` / `BeginRenderTargetFrame` / `EndRenderTargetFrame`, the
`fullscreen` post pipeline, per-frame UBO, push-constant model matrix **plus per-material
metallic/roughness** (80-byte vertex push constant: `{ float4x4 model; float4 material; }`), and
PCF shadow sampling.

> **Re-bake (per-material PBR):** this golden was re-baked when per-object metallic/roughness
> landed. The lit shaders now read metallic/roughness from the push constant (passed vertex→fragment
> as a flat `nointerpolation` interpolant) instead of fixed values, so the metal spheres render
> dark/specular-reflective (no diffuse, no env map yet) next to the bright dielectric cubes — a
> legitimate material-driven change. Two runs diff to `0.0000`; Vulkan renders the same variety.

> **Re-bake (procedural IBL):** re-baked again when lightweight image-based lighting landed
> (`lit.frag.hlsl` gained a `SkyColor()` that replicates `sky.frag.hlsl`'s gradient + sun glow, plus
> an environment specular reflection term: roughness-aware Fresnel-Schlick over `reflect(-V, N)`,
> roughness-blurred toward the up-sky, multiplied by F, additive to the direct lights, with a small
> sky-tinted ambient diffuse for dielectrics). The metal spheres now **reflect the procedural sky**
> (shiny) instead of rendering dark, and dielectrics gain a subtle Fresnel sky sheen — a legitimate
> shading change (DIFF 3.4854 vs the previous golden). One shared HLSL change hits both backends;
> two Metal runs diff to `0.0000`, and Vulkan renders the same reflective metals.

> **Re-bake (glTF model):** re-baked again when real glTF model loading landed
> (`engine/asset/gltf_loader.cpp`, header-only `cgltf`). The centre cell of the 3×3 grid is now a
> **real 3D model loaded from `assets/models/Duck.glb`** (Khronos glTF-Sample-Assets, CC0) instead of
> a procedural sphere, rendered through the *same* `scene::Vertex` layout and lit PBR pipeline as a
> fixed metallic material (metallic=1.0, roughness=0.2) so it reflects the procedural sky via IBL —
> polished chrome. Geometry only (POSITION/NORMAL/TEXCOORD_0 + indices); glTF materials/textures
> deferred. One shared loader feeds both backends; two Metal runs diff to `0.0000`, and Vulkan
> renders the same model identically.

> **Re-bake (GPU particles):** re-baked again when the compute-shader GPU particle system landed
> (`shaders/particles.comp.hlsl` + `particle.vert/frag.hlsl`). A compute kernel animates a 50k-particle
> storage buffer each frame (gravity + swirl fountain, deterministic respawn from a stable per-index
> seed), and the particles are drawn as **additive points** over the scene. `visual_test` advances the
> sim a fixed 100 steps at a fixed `dt` before the captured frame, so the fountain is golden-stable —
> two Metal runs diff to `0.0000`. The compute MSL is generated from the shared HLSL exactly like the
> graphics shaders (HLSL → SPIR-V → `kernel` MSL via spirv-cross); Vulkan renders the same fountain.

> **Re-bake (tangent-space normal mapping):** re-baked again when normal mapping landed. The lit
> shader (`shaders/lit.{vert,frag}.hlsl`) now builds a TBN basis from a per-vertex tangent (new
> `scene::Vertex.tangent` at location 4, stride 56) and the interpolated world normal, samples a
> tangent-space normal map, and perturbs the shading normal used by **all** the PBR/IBL/shadow
> lighting. The material descriptor set gained a second sampled image + sampler for the normal map
> (Vulkan set 1 binding 3/4; generated MSL maps it to `gNormalMap [[texture(3)]]` /
> `gNormalSmp [[sampler(4)]]` via `--msl-decoration-binding`, matching `metal_common.h`'s
> `kFragNormalTex`/`kFragNormalSmp`). A procedural domed-tile normal map bumps the dielectric cubes
> + ground plane; the metal spheres and the duck keep a flat (0,0,1) normal. Legitimate shading
> change: **DIFF 0.3283** vs the previous golden. One shared HLSL change hits both backends; two
> Metal runs diff to `0.0000`, and Vulkan renders the same bumps.

## `transparency.png` — alpha-blended transparency (Slice T)

`transparency.png` is a SEPARATE golden produced by `visual_test --transparency`: a minimal
translucent-glass showcase (ground checkerboard plane + procedural sky + three **opaque** lit cubes —
two behind the glass, one in front as an occluder — then **four overlapping tinted glass spheres** at
different depths, lit + shadowed, fixed head-on camera/light). The glass is drawn in a SORTED
(back-to-front by distance from the camera eye) **alpha-blended** pass using a SEPARATE transparent
pipeline (`shaders/transparent.vert.hlsl` + `shaders/transparent.frag.hlsl`): the fragment shader is
self-contained (no material textures) — a directional half-Lambert diffuse + a Blinn-Phong specular
highlight + a procedural `SkyColor` reflection, with a **Fresnel-style alpha** that rises at grazing
angles so edges read as more opaque (`alpha = lerp(baseAlpha, 1, pow(1-N·V, 5))`). Per-object tint +
base alpha arrive via the push constant (`{ float4x4 model; float4 tintAlpha }`, 80 bytes).

The pipeline runs `alphaBlend=true, depthTest=true, depthWrite=false, cullNone=true` (double-sided
glass). The new RHI flag **`GraphicsPipelineDesc.depthWrite`** (default `true`) lets the glass
depth-TEST against the opaque scene (so opaque geometry in front correctly occludes it — see the
front cube) while NOT writing depth (so overlapping glass blends correctly and never self-occludes).
`depthWrite` defaults true, so every pre-existing pipeline is byte-for-byte unchanged: the five
goldens above (`scene_shadow`/`skinning`/`pbr_helmet`/`instanced`/`ibl_helmet`) and `physics` all
still diff `0.0000`. The scene is deterministic (fixed camera, deterministic CPU sort) — two
`--transparency` runs diff `0.0000`. This golden is INDEPENDENT of all the others.

Re-bake / validate the same way as the others:

```sh
./build-metal/visual_test --transparency /tmp/glass.png
~/mac-remote-rig/compare.sh tests/golden/metal/transparency.png /tmp/glass.png 0.0   # -> DIFF 0.0000
```

## Shaders are generated, not hand-written

The Metal shaders are **generated from the shared HLSL sources** at build time — there is no
hand-written MSL to drift from the canonical shaders. The `metal_headless` build runs, for each
shader: HLSL → SPIR-V (`glslc -x hlsl`) → MSL (`spirv-cross --msl --msl-decoration-binding`),
emitting `*.gen.metal`, which `visual_test` compiles at runtime via `newLibraryWithSource:`. The
`--msl-decoration-binding` flag maps each resource's SPIR-V binding directly to its Metal
`[[buffer/texture/sampler(n)]]` index; the engine's Metal binding constants
(`engine/rhi_metal/metal_common.h`) are chosen to match.

> **Re-bake history:** this golden was re-baked from the unified pipeline when the toolchain
> landed. The previous golden had been produced from hand-written MSL that had silently *drifted*
> from the HLSL (it used `ambient 0.15` vs the HLSL's `0.12`, and its post pass omitted the FXAA,
> glow, and film grain that the canonical `post.frag.hlsl` — and the Vulkan backend — apply). The
> generated MSL faithfully reproduces the canonical HLSL, so this golden now matches what Vulkan
> renders semantically. Eliminating exactly that kind of drift is the point of the toolchain.

## How it is produced

```sh
# On the Mac (Command Line Tools only — runtime MSL compile, no Xcode/metal CLI):
source ~/mac-remote-rig/env.sh
cd ~/hazard-forge
cmake -S metal_headless -B build-metal -G Ninja
cmake --build build-metal          # also generates *.gen.metal from the HLSL
./build-metal/visual_test /tmp/out.png
```

## Validate a future run (visual regression)

```sh
./build-metal/visual_test /tmp/out.png
~/mac-remote-rig/compare.sh tests/golden/metal/scene_shadow.png /tmp/out.png 0.0
# -> DIFF 0.0000 (threshold 0.0), exit 0 on a match.
```

A non-zero DIFF means the Metal render changed — investigate before re-baking this golden.

## NDC-Y convention (handled CPU-side)

`math::Perspective` / `math::Ortho` bake the Vulkan clip-space Y-flip (+Y down). Metal NDC is +Y up.
Rather than flip in the shader (which would diverge from the shared HLSL), `visual_test` flips the
projection's and ortho's Y row on the **CPU** before composing view-proj / lightViewProj, so the
shared HLSL→MSL needs no Metal-specific clip flip. The only remaining Metal-specific adjustments are
two *texture-origin* V-flips (Metal stores row 0 = top): the fullscreen-post sample UV and the
shadow-map sample UV — both guarded by `#ifdef HF_MSL_GEN` so the Vulkan SPIR-V is byte-identical.
The shadow render and the lit-pass sampling both derive from the same CPU-flipped `lightViewProj`,
so they stay self-consistent.
