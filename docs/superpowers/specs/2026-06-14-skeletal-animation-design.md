# Slice O — Skeletal Animation (glTF skins + GPU skinning) — Design

> Autonomous-session spec (delegate self-approval per the standing user directive: bold decisions,
> answer my own design questions, document every call here for later review). Verifiable on this
> Windows+Vulkan box headlessly; Mac/Metal parity coordinated after.

**Goal:** Turn the engine from a static renderer into one that renders a *rigged, animated* glTF
character. Load a glTF skin (`JOINTS_0`/`WEIGHTS_0` + inverse-bind matrices + node hierarchy) and one
animation, sample it at a fixed time into a joint-matrix palette, and skin the mesh on the GPU in a
skinned lit-shader variant. Deterministic → golden-verifiable on both backends.

**Asset:** `assets/models/Fox.glb` (CC0, PixelMannen, via Khronos glTF-Sample-Assets). 1 mesh, 1 skin,
24 joints (+ inverse-bind matrices), 26 nodes, animations `Survey`/`Walk`/`Run`, 1 base-color texture.
Primitive attrs: `POSITION, TEXCOORD_0, JOINTS_0, WEIGHTS_0` — **no NORMAL/TANGENT**, so the loader
must compute smooth normals from the indexed positions (flat tangent (1,0,0) is fine; the fox uses no
normal map).

## Design decisions (locked)

1. **Separate skinned vertex layout — do NOT bloat the static `scene::Vertex`.**
   New `scene::SkinnedVertex { float pos[3]; float color[3]; float uv[2]; float normal[3];
   float tangent[3]; float joints[4]; float weights[4]; }` (stride 88). Joint *indices* stored as
   floats (GPU reads them as floats and casts to int) to avoid an integer vertex format. New
   `scene::SkinnedMeshVertexLayout()` adds locations 5 (joints) + 6 (weights) as `RGBA32_Float`.

2. **New RHI vertex format `Format::RGBA32_Float`** (vec4). Map: Vulkan
   `VK_FORMAT_R32G32B32A32_SFLOAT`, Metal `MTLVertexFormatFloat4`. Pure-additive enum change.

3. **Joint palette via a dedicated palette UBO — mirror the existing per-frame UBO pattern exactly.**
   - RHI: `GraphicsPipelineDesc.usesJointPalette` (when true the pipeline declares descriptor **set 2**
     = a single uniform buffer `JointPalette { float4x4 joints[64]; }`).
   - RHI: `IRHIDevice::SetJointPalette(const void* data, size_t size)` — memcpy into the current
     frame's palette UBO (double-buffered, one per frame-in-flight), analogous to `SetFrameUniforms`.
   - On `BindPipeline`, when `usesJointPalette` is true, auto-bind set 2 (the current frame's palette
     UBO), exactly like set 0 auto-binds for `usesFrameUniforms`.
   - Palette UBO sized for **64 joints** (64×64 B = 4096 B). Fox uses 24; the rest are identity/unused.
   This keeps the seam clean (no SSBO-in-vertex-stage plumbing) and supports one skinned object/frame,
   which is all the showcase needs.

4. **Skinned lit pipeline variant.** New `shaders/lit_skinned.vert.hlsl`: reads `JointPalette` (set 2),
   builds `skinMat = w.x*J[idx.x] + w.y*J[idx.y] + w.z*J[idx.z] + w.w*J[idx.w]`, transforms position by
   `skinMat` then the push-constant model matrix; transforms normal/tangent by `skinMat`'s upper 3×3
   (no non-uniform-scale handling needed for Fox). Fragment stage reuses the existing `lit.frag.hlsl`
   unchanged (PBR + IBL + shadow PCF + base-color texture). The HLSL→SPIR-V→MSL toolchain generates the
   MSL exactly as for the other shaders (`HF_MSL_GEN` guards already handle push-constant/UV conventions).

5. **Animation sampling (engine math, no GPU).** New `engine/anim/`:
   - `skeleton.h` — `Skeleton { std::vector<Joint> joints; }` where `Joint { int parent; Mat4 inverseBind;
     Vec3 t; Quat r; Vec3 s; /*local rest TRS*/ }`, plus the joint→node and node→joint maps needed to
     apply channels. Store joints in a topologically-sorted order (parents before children) so a single
     forward pass computes global transforms.
   - `animation.h` — `Animation { float duration; std::vector<Channel> channels; }`,
     `Channel { int jointIndex; Path {Translation,Rotation,Scale}; std::vector<float> times;
     std::vector<float> values; Interp {Linear,Step}; }`.
   - `Quat` added to `engine/math/math.h` (x,y,z,w) with `Normalize`, `Slerp` (nlerp acceptable —
     document the choice), and `Mat4 FromTRS(Vec3 t, Quat r, Vec3 s)`.
   - `SampleAnimation(const Skeleton&, const Animation&, float time) -> std::vector<Mat4> palette`:
     per channel, find the keyframe interval (clamp/loop at duration), interpolate (lerp T/S, slerp R),
     compose local TRS, forward-walk the hierarchy for global transforms, then
     `palette[j] = global[j] * inverseBind[j]`. Unit-tested with hand-checked values (identity rest pose
     → palette of identities; a single-joint 90° Z rotation → known matrix).

6. **glTF loader extension.** New `LoadSkinnedGltfModel(device, path) -> SkinnedModel { scene::Mesh mesh
   (skinned-vertex buffers); std::unique_ptr<ITexture> baseColor; float metallic, roughness;
   anim::Skeleton skeleton; std::vector<anim::Animation> animations; }`. Reuses the existing base-color
   decode path. Reads `JOINTS_0` (u8/u16 → float) + `WEIGHTS_0` (normalize so they sum to 1). Computes
   smooth normals from indexed positions when `NORMAL` is absent. Recentre is **NOT** applied to skinned
   meshes (the skin's inverse-bind matrices define the bind-space origin; recentring would desync them) —
   instead the showcase places the fox with an explicit model matrix (uniform scale + translate +
   ground-align using the asset's bbox).

7. **Showcase + verification — add a NEW golden, do NOT re-bake existing goldens.**
   - The main scene golden (`tests/golden/metal/scene_shadow.png`) and the Vulkan scene stay untouched.
   - Add a dedicated capture path: `hello_triangle.exe --skinning-shot <path>` renders a minimal
     skinning showcase (ground plane + skybox + the Fox skinned at a FIXED animation — `Survey`, time
     `0.5s` — lit + shadowed) and captures one frame. Deterministic (fixed camera/light/time).
   - New Vulkan/Windows golden review: I (controller) eyeball the captured PNG to confirm the fox is a
     correctly-posed fox (skinning correctness is a *first-time* property the golden cannot self-verify).
   - New Metal golden `tests/golden/metal/skinning.png`: `metal_headless/visual_test` gains the same
     showcase (a `--skinning` arg or a second entry) at the identical fixed time; two Metal runs diff to
     `0.0000`; Vulkan renders the same pose.
   - Wire `HF_FOX_MODEL_PATH` define (both CMake targets) → `assets/models/Fox.glb`.

## RHI seam additions (summary)
- `Format::RGBA32_Float`
- `GraphicsPipelineDesc.usesJointPalette` (bool, default false)
- `IRHIDevice::SetJointPalette(const void* data, size_t size)` (default no-op in the base; implemented
  in both backends)
- Descriptor **set 2** = `JointPalette` UBO, auto-bound on `BindPipeline` when `usesJointPalette`.
Hard rule unchanged: no `vk*`/`MTL*` symbols above the backend dirs. `kFrameUboSize` untouched (palette
is its own buffer).

## Out of scope (YAGNI)
Animation blending/transitions, multiple simultaneous skinned objects, CPU skinning fallback, morph
targets, non-uniform-scale normal correction (inverse-transpose), animation playback UI. One fox, one
fixed pose, GPU-skinned, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` stays green (existing 6/6) + new `anim_test` (math/sampling).
2. `--skinning-shot` on Windows/Vulkan produces a recognizable posed fox (controller visual review).
3. Metal: `visual_test` skinning showcase → new golden `tests/golden/metal/skinning.png`, two runs
   `DIFF 0.0000`; existing goldens unchanged.
4. Seam grep clean. ASan preset still builds (skinning sources land in `hf_core` where backend-agnostic).
