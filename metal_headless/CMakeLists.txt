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
hf_gen_msl(post.vert   post.vert.hlsl   vert post_vertex)
hf_gen_msl(post.frag   post.frag.hlsl   frag post_fragment)
hf_gen_msl(shadow.vert shadow.vert.hlsl vert shadow_vertex)
hf_gen_msl(sky.vert    sky.vert.hlsl    vert sky_vertex)
hf_gen_msl(sky.frag    sky.frag.hlsl    frag sky_fragment)
# GPU particle system: compute kernel + point-sprite vertex/fragment.
hf_gen_msl(particles.comp particles.comp.hlsl comp particles_main)
hf_gen_msl(particle.vert   particle.vert.hlsl  vert particle_vertex)
hf_gen_msl(particle.frag   particle.frag.hlsl  frag particle_fragment)

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
    # glTF loader (plain C++, header-only cgltf): loads a real model into a scene::Mesh.
    ${HF_ROOT}/engine/asset/gltf_loader.cpp
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
