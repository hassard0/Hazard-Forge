# Standalone, conan-free, SDL-free headless Metal target for bringing up the Metal RHI
# backend on a real Mac over SSH (no window server). Drives the REAL Metal RHI classes
# (MetalDevice/MetalSwapchain/MetalCommandBuffer/MetalPipeline/MetalTexture/MetalBuffer) in
# headless/offscreen mode, renders the full Slice-F scene (ground plane + 3x3 lit cube grid),
# reads the offscreen MTLTexture back, and writes a PNG.
#
#   cmake -S metal_headless -B build-metal -G Ninja
#   cmake --build build-metal
#   ./build-metal/visual_test out.png
cmake_minimum_required(VERSION 3.20)
project(hf_metal_headless LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Repo root is one level up from this dir.
get_filename_component(HF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# ---------------------------------------------------------------------------
# Unified shader toolchain: GENERATE the Metal MSL from the SHARED HLSL sources
# (the same HLSL the Vulkan/DXC backend compiles) so there is no hand-written
# MSL to drift. Pipeline per shader: HLSL --glslc--> SPIR-V --spirv-cross--> MSL.
#
#   * glslc compiles HLSL with -DHF_MSL_GEN so the vertex shaders declare the
#     model matrix as an explicit cbuffer (glslang ignores [[vk::push_constant]]);
#     the bindings are chosen so that --msl-decoration-binding yields the engine's
#     flat Metal indices (see engine/rhi_metal/metal_common.h).
#   * spirv-cross --msl --msl-decoration-binding maps SPIR-V binding -> MSL index,
#     and --rename-entry-point gives stable entry names the loader expects.
# ---------------------------------------------------------------------------
find_program(GLSLC_EXECUTABLE NAMES glslc REQUIRED)
find_program(SPIRV_CROSS_EXECUTABLE NAMES spirv-cross REQUIRED)

set(HF_SHADER_SRC "${HF_ROOT}/shaders")
set(HF_GEN_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen_shaders")
file(MAKE_DIRECTORY "${HF_GEN_SHADER_DIR}")

set(HF_GENERATED_MSL "")
# hf_gen_msl(<basename> <hlsl-file> <vert|frag> <new-entry-point>)
function(hf_gen_msl base src stage entry)
  set(spv "${HF_GEN_SHADER_DIR}/${base}.spv")
  set(msl "${HF_GEN_SHADER_DIR}/${base}.gen.metal")
  if(stage STREQUAL "vert")
    set(glslc_stage "vert")
    set(sc_stage "vert")
  elseif(stage STREQUAL "comp")
    set(glslc_stage "compute")  # glslc compute stage flag
    set(sc_stage "comp")        # spirv-cross --rename-entry-point execution model
  else()
    set(glslc_stage "frag")
    set(sc_stage "frag")
  endif()
  add_custom_command(
    OUTPUT "${msl}"
    COMMAND "${GLSLC_EXECUTABLE}" -DHF_MSL_GEN -fshader-stage=${glslc_stage}
            -x hlsl -fentry-point=main "${HF_SHADER_SRC}/${src}" -o "${spv}"
    COMMAND "${SPIRV_CROSS_EXECUTABLE}" --msl --msl-decoration-binding
            --rename-entry-point main ${entry} ${sc_stage}
            "${spv}" --output "${msl}"
    DEPENDS "${HF_SHADER_SRC}/${src}"
    COMMENT "Gen MSL ${src} -> ${base}.gen.metal"
    VERBATIM)
  set(HF_GENERATED_MSL ${HF_GENERATED_MSL} "${msl}" PARENT_SCOPE)
endfunction()

hf_gen_msl(lit.vert    lit.vert.hlsl    vert vertex_main)
hf_gen_msl(lit.frag    lit.frag.hlsl    frag fragment_main)
# Cascaded shadow maps (Slice AD): CSM lit fragment (atlas cascade selection + PCF) + the depth-only
# CSM shadow caster (cascade VP in the push constant). Reuse lit.vert. Same HLSL->SPIR-V->MSL chain.
hf_gen_msl(lit_csm.frag    lit_csm.frag.hlsl    frag csm_fragment)
hf_gen_msl(shadow_csm.vert shadow_csm.vert.hlsl vert csm_shadow_vertex)
# Spot light + perspective spot-light shadow (Slice AE): cone-cut lit fragment. Reuses lit.vert +
# the CSM depth-only shadow caster (shadow_csm.vert: spotViewProj in the push constant).
hf_gen_msl(lit_spot.frag   lit_spot.frag.hlsl   frag spot_fragment)
# Omnidirectional point-light shadow (Slice AF): 6-face cube atlas lit fragment. Reuses lit.vert +
# the CSM depth-only shadow caster (shadow_csm.vert: per-face view-proj in the push constant).
hf_gen_msl(lit_point.frag  lit_point.frag.hlsl  frag point_fragment)
# Clustered / Forward+ lit fragment (Slice AG): per-fragment cluster index + iterate that cluster's
# point lights from three fragment-stage storage buffers (space3 -> flat Metal fragment buffers).
# Reuses lit.vert. Same HLSL->SPIR-V->MSL toolchain.
hf_gen_msl(lit_clustered.frag  lit_clustered.frag.hlsl  frag clustered_fragment)
# Full-PBR fragment shader (Slice P): reuses lit.vert; samples the 5-texture glTF material set.
hf_gen_msl(lit_pbr.frag lit_pbr.frag.hlsl frag pbr_fragment)
# HDR environment IBL (Slice R): equirect skybox + full-PBR-with-HDR-IBL fragment shaders. Same
# HLSL->SPIR-V->MSL toolchain; the env map binds at flat fragment texture(11)/sampler(12).
hf_gen_msl(sky_hdr.frag      sky_hdr.frag.hlsl      frag sky_hdr_fragment)
hf_gen_msl(lit_pbr_ibl.frag  lit_pbr_ibl.frag.hlsl  frag pbr_ibl_fragment)
# Skinned-animation variants (Slice O): GPU skinning vertex shader (reuses lit.frag) + skinned
# depth-only shadow vertex shader. Same HLSL->SPIR-V->MSL toolchain as the others.
hf_gen_msl(lit_skinned.vert    lit_skinned.vert.hlsl    vert skinned_vertex)
hf_gen_msl(shadow_skinned.vert shadow_skinned.vert.hlsl vert skinned_shadow_vertex)
# Instanced variants (Slice Q): per-instance model-matrix vertex stream (binding 1). The lit
# variant reuses lit.frag; the shadow variant is depth-only. Same HLSL->SPIR-V->MSL toolchain.
hf_gen_msl(lit_instanced.vert    lit_instanced.vert.hlsl    vert instanced_vertex)
hf_gen_msl(shadow_instanced.vert shadow_instanced.vert.hlsl vert instanced_shadow_vertex)
# Transparent ("glass") pass (Slice T): a dedicated vertex shader (model + per-object tint/alpha push
# constant) and a self-contained glassy fragment shader (no material textures). Same HLSL->SPIR-V->MSL
# toolchain as the others.
hf_gen_msl(transparent.vert transparent.vert.hlsl vert transparent_vertex)
hf_gen_msl(transparent.frag transparent.frag.hlsl frag transparent_fragment)
hf_gen_msl(post.vert   post.vert.hlsl   vert post_vertex)
hf_gen_msl(post.frag   post.frag.hlsl   frag post_fragment)
# HDR bloom chain (Slice U): bright-pass, downsample, tent upsample, composite/tonemap. All
# fullscreen fragment shaders reusing post.vert; each reads its per-pass params from a fragment
# push constant (a HF_MSL_GEN cbuffer at the bloom push-const slot here).
hf_gen_msl(bloom_prefilter.frag  bloom_prefilter.frag.hlsl  frag bloom_prefilter_fragment)
hf_gen_msl(bloom_downsample.frag bloom_downsample.frag.hlsl frag bloom_downsample_fragment)
hf_gen_msl(bloom_upsample.frag   bloom_upsample.frag.hlsl   frag bloom_upsample_fragment)
hf_gen_msl(bloom_composite.frag  bloom_composite.frag.hlsl  frag bloom_composite_fragment)
hf_gen_msl(shadow.vert shadow.vert.hlsl vert shadow_vertex)
hf_gen_msl(sky.vert    sky.vert.hlsl    vert sky_vertex)
hf_gen_msl(sky.frag    sky.frag.hlsl    frag sky_fragment)
# GPU particle system: compute kernel + point-sprite vertex/fragment.
hf_gen_msl(particles.comp particles.comp.hlsl comp particles_main)
hf_gen_msl(particle.vert   particle.vert.hlsl  vert particle_vertex)
hf_gen_msl(particle.frag   particle.frag.hlsl  frag particle_fragment)
# Immediate-mode debug-draw line pipeline (Slice W): world-space line verts projected by
# FrameData.viewProj, flat per-vertex color. LINE_LIST topology. Same HLSL->SPIR-V->MSL toolchain.
hf_gen_msl(debug_line.vert debug_line.vert.hlsl vert debug_line_vertex)
hf_gen_msl(debug_line.frag debug_line.frag.hlsl frag debug_line_fragment)
# SSAO chain (Slice Y): view-space normal+linear-depth g-buffer prepass (static + instanced vertex
# variants + shared fragment), hemisphere-kernel AO pass, box blur, composite/apply. Same
# HLSL->SPIR-V->MSL toolchain as the others.
hf_gen_msl(gbuffer.vert           gbuffer.vert.hlsl           vert gbuffer_vertex)
hf_gen_msl(gbuffer_instanced.vert gbuffer_instanced.vert.hlsl vert gbuffer_instanced_vertex)
hf_gen_msl(gbuffer.frag           gbuffer.frag.hlsl           frag gbuffer_fragment)
hf_gen_msl(ssao.frag              ssao.frag.hlsl              frag ssao_fragment)
hf_gen_msl(ssao_blur.frag         ssao_blur.frag.hlsl         frag ssao_blur_fragment)
hf_gen_msl(ssao_composite.frag    ssao_composite.frag.hlsl    frag ssao_composite_fragment)

add_custom_target(hf_gen_shaders DEPENDS ${HF_GENERATED_MSL})

add_executable(visual_test
    visual_test.mm
    # The REAL Metal RHI backend, built headless (no hal/window.h needed for the offscreen path).
    ${HF_ROOT}/engine/rhi_metal/metal_device.mm
    ${HF_ROOT}/engine/rhi_metal/metal_swapchain.mm
    ${HF_ROOT}/engine/rhi_metal/metal_command_buffer.mm
    ${HF_ROOT}/engine/rhi_metal/metal_pipeline.mm
    ${HF_ROOT}/engine/rhi_metal/metal_compute_pipeline.mm
    ${HF_ROOT}/engine/rhi_metal/metal_texture.mm
    ${HF_ROOT}/engine/rhi_metal/metal_render_target.mm
    ${HF_ROOT}/engine/rhi_metal/metal_shader.mm
    ${HF_ROOT}/engine/rhi_metal/metal_shader_load.mm
    ${HF_ROOT}/engine/rhi_metal/metal_buffer.mm
    # Declarative render graph (backend-agnostic; depends only on rhi/ + scene/ + std).
    ${HF_ROOT}/engine/render/render_graph.cpp
    # Scene layer (plain C++): builds Mesh::Cube / Mesh::Plane through the RHI.
    ${HF_ROOT}/engine/scene/mesh.cpp
    # Data-driven scene load/dump (plain C++, vendored single-header JSON parser).
    ${HF_ROOT}/engine/scene/scene_io.cpp
    # glTF loader (plain C++, header-only cgltf): loads a real model into a scene::Mesh.
    ${HF_ROOT}/engine/asset/gltf_loader.cpp
    # HDR environment loader (Slice R): decodes env.hdr, builds the CPU mip chain, uploads RGBA16F.
    ${HF_ROOT}/engine/asset/env_loader.cpp
    # Skeletal-animation sampling (plain C++): Quat/Slerp/FromTRS live in math.h; this is the
    # SampleAnimation palette builder (Slice O).
    ${HF_ROOT}/engine/anim/animation.cpp
    # Impulse-based rigid-body world (Slice S): pure C++ physics core (no Metal/RHI deps).
    ${HF_ROOT}/engine/physics/world.cpp
    # Immediate-mode debug-draw collector + engine-object emitters (Slice W): pure C++ (math +
    # scene + physics), no Metal/RHI deps. Drives the LINE_LIST debug overlay.
    ${HF_ROOT}/engine/debug/debug_draw.cpp
    ${HF_ROOT}/engine/debug/debug_emitters.cpp
    # Editor transform-gizmo emit (Slice AB): pure C++ (math + scene::Transform + debug::DebugDraw),
    # no Metal/RHI deps. Drives the --gizmo showcase's translate-gizmo line overlay.
    ${HF_ROOT}/engine/editor/gizmo.cpp
)

# The generated MSL must exist before the executable's shaders are loaded at runtime; make the
# build depend on the generation target so it always runs first.
add_dependencies(visual_test hf_gen_shaders)

# Compile all Obj-C++ with ARC.
target_compile_options(visual_test PRIVATE -fobjc-arc)

# Include roots so #include "rhi/rhi.h", "rhi_metal/...", "scene/...", "asset/...", "math/..."
# and the header-only "cgltf/cgltf.h" resolve.
target_include_directories(visual_test PRIVATE
    ${HF_ROOT}/engine
    ${HF_ROOT}/third_party
)

# Path to the GENERATED MSL shaders (HLSL->SPIR-V->MSL), baked in as a compile definition.
target_compile_definitions(visual_test PRIVATE
    HF_SHADER_DIR="${HF_ROOT}/shaders"
    HF_GEN_SHADER_DIR="${HF_GEN_SHADER_DIR}"
    HF_MODEL_PATH="${HF_ROOT}/assets/models/Duck.glb"
    HF_FOX_MODEL_PATH="${HF_ROOT}/assets/models/Fox.glb"
    HF_HELMET_MODEL_PATH="${HF_ROOT}/assets/models/DamagedHelmet.glb"
    HF_TRUCK_MODEL_PATH="${HF_ROOT}/assets/models/CesiumMilkTruck.glb"
    HF_SCENE_PATH="${HF_ROOT}/assets/scenes/default.json"
    HF_ENV_PATH="${HF_ROOT}/assets/env/env.hdr"
)

target_link_libraries(visual_test PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework CoreGraphics"
    "-framework ImageIO"
    "-framework CoreServices"
    "-framework UniformTypeIdentifiers"
)
