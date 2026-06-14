# Metal golden-image test

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
