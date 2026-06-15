# Slice V — Full glTF scene-graph import (design)

Date: 2026-06-14
Branch: `slice-scene-import`

## Goal
Generalize the single-primitive PBR loader (`LoadPbrGltfModel`, Slice P) into a full
scene-graph importer that walks a glTF node hierarchy, composes world transforms
(parent * local), emits one renderable instance per primitive of every referenced mesh,
and dedups/shares materials. Reuse the existing lit-PBR pipeline + `BindMaterialPBR`
unchanged. Prove correctness with the CesiumMilkTruck (the same wheel mesh placed at two
positions via different parent transforms).

## Non-goals
- No new backend code (pure C++ in `hf_core`, abstract RHI + cgltf/stb only).
- No new shaders/pipelines; reuse Slice P's lit-PBR pipeline + ground/shadow/sky/post.
- Existing `LoadPbrGltfModel` behavior and all existing goldens stay byte-identical.

## Decisions

### 1. Per-primitive decode helper `BuildPrimitive`
Refactor the geometry+tangent decode out of `BuildMesh` into a reusable internal helper
`BuildPrimitive(device, prim, recentre) -> scene::Mesh` operating on an arbitrary
`cgltf_primitive` (not hard-coded to mesh0/prim0). It reads POSITION/NORMAL/TEXCOORD_0,
widens indices to u32, and computes tangents (authored TANGENT preferred, else Lengyel
from POSITION/UV/NORMAL, else default (1,0,0)). `recentre=true` preserves the legacy
single-mesh behavior (recentre on bbox centre); scene import uses `recentre=false` so node
world transforms place geometry in its authored model space — recentring is done once at
scene level instead.

`BuildMesh` (legacy) is kept as a thin wrapper: `BuildPrimitive(device, mesh0.prim0, /*recentre=*/true)`.
`LoadPbrGltfModel` is unchanged (still mesh0/prim0, recentred).

### 2. Material decode + dedup
`DecodeMaterial(device, mat) -> PbrMaterial` decodes the 5 textures + factors exactly as
`LoadPbrGltfModel` does today, with the same neutral fallbacks. A null material yields a
sensible default (white base, neutral metalRough = rough 1 / metal 0, flat normal, black
emissive, white occlusion, metallic 0 / roughness 1).

`PbrMaterial` owns its 5 `std::unique_ptr<rhi::ITexture>` + factors. The scene stores an
owning `std::vector<std::unique_ptr<PbrMaterial>>` (the material table). Materials are
deduped by `cgltf_material*` (and a single shared default) via a cache map, so the truck's
4 glTF materials decode once and are shared across the primitives that use them.
`SceneInstance` holds a NON-owning `const PbrMaterial*` into that table.

### 3. Scene-graph traversal
`LoadGltfScene(device, path) -> GltfScene`:
- `GltfScene { std::vector<std::unique_ptr<scene::Mesh>> meshStorage; std::vector<std::unique_ptr<PbrMaterial>> materialStorage; std::vector<SceneInstance> instances; float bbMin[3], bbMax[3]; }`
- `SceneInstance { const scene::Mesh* mesh; const PbrMaterial* material; math::Mat4 worldTransform; }`

Traversal: start from the default scene (`data->scene`, fallback scene 0) root nodes,
depth-first. `world = parentWorld * local`, where `local` is the node `matrix` (16 floats,
column-major, copied straight into `Mat4`) if present, else `FromTRS(T,R,S)`. For every
node that references a mesh, emit one `SceneInstance` per primitive: the primitive's
`scene::Mesh` (built once per (mesh,prim) and cached so the shared wheels mesh uploads
once) + its deduped material + the node's world transform. The same mesh under two parents
therefore yields instances at two different worlds (front/back wheels) — the core proof.

Geometry is built with `recentre=false`. Each unique (meshIndex, primIndex) is uploaded
once and cached; multiple nodes referencing the same mesh reuse the same `scene::Mesh*`.

### 4. Scene-level fit (no geometry mutation)
`SceneFitTransform(scene, targetSize, groundY) -> Mat4` computes the scene-wide AABB by
transforming each primitive's local bbox corners by its instance world transform, then
returns a uniform-scale + translate that centres the scene on the ground plane (min-Y to
`groundY`, XZ centre to origin) and fits it to `targetSize`. This is a pure helper; it does
NOT mutate vertices. The showcase pre-multiplies it onto each instance world.

### 5. Showcase + verification (NEW golden; existing untouched)
- Vulkan: `hello_triangle --scene-shot <path>` loads CesiumMilkTruck, applies the fit
  transform + an orientation fix so the truck stands on its wheels, renders ground + sky +
  lit-PBR + shadow exactly like `--pbr-shot` but iterating instances (PushConstants(world),
  BindMaterialPBR(material 5 textures), bind VB/IB, DrawIndexed). BMP -> PNG -> visual check.
- The asset's root node "Yup2Zup" rotates Y-up authored content into Z-up. Since the engine
  is Y-up, that root rotation lays the truck on its side; the showcase applies a counter
  orientation (rotate to put +Z-up content back to +Y-up) determined empirically from the
  capture so the truck stands on its wheels.
- `HF_TRUCK_MODEL_PATH` CMake define (both targets) -> CesiumMilkTruck.glb.
- Metal: same showcase added to `visual_test.mm` (`--scene`); new golden
  `tests/golden/metal/scene_import.png` if the Mac is reachable, else "Metal golden pending".

### 6. Unit test (pure C++ / hf_core / ASan)
`tests/scene_import_test.cpp`: factor the hierarchy composition into a pure function
`ComposeWorld(parentWorld, localTransform)` and a node-array walker operating on plain
node structs (no device). Build a small parent-translate -> child-translate hierarchy and
assert the leaf world transform equals the hand-checked parent*local product (translation
sums; off-axis to catch order bugs). Registered in `tests/CMakeLists.txt`.

## Seam
Loader stays pure C++ in `hf_core`. No `vk*`/`MTL`/`Metal` tokens added to the guarded dirs;
seam grep stays at the 12 baseline.
