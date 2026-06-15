add_executable(hello_triangle main.cpp)
# Vulkan/Windows sample links the editor shell (Dear ImGui through the RHI); the editor is gated to
# non-Apple in the root CMakeLists, so this link only happens on the Vulkan path.
target_link_libraries(hello_triangle PRIVATE hf_engine hf_editor)
target_compile_definitions(hello_triangle PRIVATE HF_HAS_EDITOR=1)

set(SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/shaders")
# The ui.vert/frag shaders are Vulkan-only (editor) — added only to the Vulkan sample's compile list,
# never to the metal_headless gen list, so the Metal shader generation is untouched.
hf_compile_shaders(
  TARGET hello_triangle
  OUTPUT_DIR "${SHADER_OUT}"
  SHADERS
    "shaders/lit.vert.hlsl:vs"
    "shaders/lit.frag.hlsl:ps"
    "shaders/lit_csm.frag.hlsl:ps"
    "shaders/shadow_csm.vert.hlsl:vs"
    # Spot light + perspective spot-light shadow (Slice AE): cone-cut lit fragment + reuse the CSM
    # depth-only caster (spotViewProj in the push constant).
    "shaders/lit_spot.frag.hlsl:ps"
    # Omnidirectional point-light shadow (Slice AF): 6-face cube atlas lit fragment + reuse the CSM
    # depth-only caster (per-face view-proj in the push constant).
    "shaders/lit_point.frag.hlsl:ps"
    # Clustered / Forward+ lit fragment (Slice AG): per-fragment cluster index + iterate that
    # cluster's point lights from three fragment-stage storage buffers (set 3).
    "shaders/lit_clustered.frag.hlsl:ps"
    "shaders/lit_pbr.frag.hlsl:ps"
    "shaders/lit_pbr_ibl.frag.hlsl:ps"
    "shaders/sky_hdr.frag.hlsl:ps"
    "shaders/lit_skinned.vert.hlsl:vs"
    "shaders/shadow_skinned.vert.hlsl:vs"
    "shaders/lit_instanced.vert.hlsl:vs"
    "shaders/shadow_instanced.vert.hlsl:vs"
    "shaders/transparent.vert.hlsl:vs"
    "shaders/transparent.frag.hlsl:ps"
    "shaders/shadow.vert.hlsl:vs"
    "shaders/shadow.frag.hlsl:ps"
    "shaders/post.vert.hlsl:vs"
    "shaders/post.frag.hlsl:ps"
    "shaders/bloom_prefilter.frag.hlsl:ps"
    "shaders/bloom_downsample.frag.hlsl:ps"
    "shaders/bloom_upsample.frag.hlsl:ps"
    "shaders/bloom_composite.frag.hlsl:ps"
    "shaders/sky.vert.hlsl:vs"
    "shaders/sky.frag.hlsl:ps"
    "shaders/particles.comp.hlsl:cs"
    "shaders/particle.vert.hlsl:vs"
    "shaders/particle.frag.hlsl:ps"
    "shaders/ui.vert.hlsl:vs"
    "shaders/ui.frag.hlsl:ps"
    # Immediate-mode debug-draw line pipeline (Slice W): world-space line verts projected by
    # FrameData.viewProj, flat per-vertex color. LINE_LIST topology.
    "shaders/debug_line.vert.hlsl:vs"
    "shaders/debug_line.frag.hlsl:ps"
    # SSAO chain (Slice Y): view-space normal+linear-depth g-buffer prepass (static + instanced
    # vertex variants + shared fragment), hemisphere-kernel AO pass, box blur, and composite/apply.
    "shaders/gbuffer.vert.hlsl:vs"
    "shaders/gbuffer_instanced.vert.hlsl:vs"
    "shaders/gbuffer.frag.hlsl:ps"
    "shaders/ssao.frag.hlsl:ps"
    "shaders/ssao_blur.frag.hlsl:ps"
    "shaders/ssao_composite.frag.hlsl:ps"
    # Screen-space reflections (Slice AH): reuses the SSAO g-buffer (view normal + linear depth);
    # ssr.frag ray-marches the depth buffer for on-screen reflections, ssr_composite blends +
    # tonemaps. Existing gbuffer/ssao/bloom shaders + their goldens untouched.
    "shaders/ssr.frag.hlsl:ps"
    "shaders/ssr_composite.frag.hlsl:ps"
    # Volumetric fog / light shafts (Slice AJ): reuses the directional shadow map + SSAO g-buffer;
    # volumetric.frag ray-marches the view ray sampling the shadow map per step for in-scattering,
    # volumetric_composite adds it over the scene + tonemaps. Existing shaders + goldens untouched.
    "shaders/volumetric.frag.hlsl:ps"
    "shaders/volumetric_composite.frag.hlsl:ps"
    # Reflection + irradiance probes (Slice AK): bake the local room into a cube atlas (reflection
    # block) + cosine-convolve an irradiance block, then shade with lit_probe sampling that atlas as
    # LOCAL GI. Reuses lit.vert + the CSM depth-only caster's vertex (probe-face VP in push const) and
    # post.vert for the blit/convolution fullscreen passes. Existing shaders + goldens untouched.
    "shaders/probe_bake.vert.hlsl:vs"
    "shaders/probe_bake.frag.hlsl:ps"
    "shaders/lit_probe.frag.hlsl:ps"
    "shaders/probe_blit.frag.hlsl:ps"
    "shaders/probe_irradiance.frag.hlsl:ps"
)
target_compile_definitions(hello_triangle PRIVATE
  "HF_SHADER_DIR=\"${SHADER_OUT}\"")
# Absolute path to the glTF showcase model, baked at configure time (repo asset).
target_compile_definitions(hello_triangle PRIVATE
  "HF_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/assets/models/Duck.glb\"")
# Absolute path to the skinned-animation showcase model (Fox.glb: 24-joint skin + animations).
target_compile_definitions(hello_triangle PRIVATE
  "HF_FOX_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/assets/models/Fox.glb\"")
# Absolute path to the full-PBR showcase model (DamagedHelmet.glb: 5-texture metallic-roughness PBR).
target_compile_definitions(hello_triangle PRIVATE
  "HF_HELMET_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/assets/models/DamagedHelmet.glb\"")
# Absolute path to the scene-graph import showcase model (CesiumMilkTruck.glb: node hierarchy +
# multiple meshes/materials; the wheels mesh is placed at two positions via two parent transforms).
target_compile_definitions(hello_triangle PRIVATE
  "HF_TRUCK_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/assets/models/CesiumMilkTruck.glb\"")
# Absolute path to the default data-driven scene description (repo asset).
target_compile_definitions(hello_triangle PRIVATE
  "HF_SCENE_PATH=\"${CMAKE_SOURCE_DIR}/assets/scenes/default.json\"")
# Absolute path to the HDR equirectangular environment map (Slice R: kloofendal clear sky, 1k).
target_compile_definitions(hello_triangle PRIVATE
  "HF_ENV_PATH=\"${CMAKE_SOURCE_DIR}/assets/env/env.hdr\"")
