# Slice AN — standalone, conan-free, SDL-free WINDOWED Metal target: a native Cocoa/AppKit NSWindow
# with a CAMetalLayer content view, driving the REAL Metal RHI backend in WINDOWED (present) mode +
# the same backend-agnostic runtime camera/loop the Windows Vulkan `--fly` viewport uses. This is the
# macOS interactive viewport. It deliberately mirrors the sibling metal_headless/ target (same
# HLSL->SPIR-V->MSL shader generation, same RHI source list) so the rendered scene matches the
# committed Metal scene golden; only the present target (window CAMetalLayer vs offscreen texture)
# and the live camera differ.
#
#   cmake -S mac_window -B build-mac-window -G Ninja
#   cmake --build build-mac-window
#   ./build-mac-window/mac_window          # opens the window; WASD move, RIGHT-drag look, ESC quits
#
# HONESTY: building over SSH proves COMPILE + LINK only. The window opening / rendering / flying is
# NOT verifiable headlessly (no display session) — run the binary on the Mac's physical display.
cmake_minimum_required(VERSION 3.20)
project(hf_mac_window LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Repo root is one level up from this dir.
get_filename_component(HF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# ---------------------------------------------------------------------------
# Generate the Metal MSL from the SHARED HLSL — IDENTICAL chain + entry points to metal_headless, so
# the shaders are byte-for-byte the same as the headless golden path (no hand-written MSL, no drift).
# Only the subset the windowed scene viewport uses is generated here.
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

# The Slice-F scene pipeline set the windowed viewport uses (mirrors RunSceneShowcase).
hf_gen_msl(lit.vert     lit.vert.hlsl     vert vertex_main)
hf_gen_msl(lit.frag     lit.frag.hlsl     frag fragment_main)
hf_gen_msl(lit_pbr.frag lit_pbr.frag.hlsl frag pbr_fragment)
hf_gen_msl(shadow.vert  shadow.vert.hlsl  vert shadow_vertex)
hf_gen_msl(sky.vert     sky.vert.hlsl     vert sky_vertex)
hf_gen_msl(sky.frag     sky.frag.hlsl     frag sky_fragment)
hf_gen_msl(post.vert    post.vert.hlsl    vert post_vertex)
hf_gen_msl(post.frag    post.frag.hlsl    frag post_fragment)

add_custom_target(hf_gen_shaders_macwin DEPENDS ${HF_GENERATED_MSL})

add_executable(mac_window
    main.mm
    # The REAL Metal RHI backend, WINDOWED. metal_device_windowed.mm compiles WITHOUT
    # HF_HAVE_SDL_WINDOW here, so it provides the SDL-free MetalDevice(void* layer,...) ctor +
    # CreateMetalDeviceWindowedLayer factory (NOT the SDL Window& ctor).
    ${HF_ROOT}/engine/rhi_metal/metal_device.mm
    ${HF_ROOT}/engine/rhi_metal/metal_device_windowed.mm
    ${HF_ROOT}/engine/rhi_metal/metal_swapchain.mm
    ${HF_ROOT}/engine/rhi_metal/metal_command_buffer.mm
    ${HF_ROOT}/engine/rhi_metal/metal_pipeline.mm
    ${HF_ROOT}/engine/rhi_metal/metal_compute_pipeline.mm
    ${HF_ROOT}/engine/rhi_metal/metal_texture.mm
    ${HF_ROOT}/engine/rhi_metal/metal_render_target.mm
    ${HF_ROOT}/engine/rhi_metal/metal_shader.mm
    ${HF_ROOT}/engine/rhi_metal/metal_shader_load.mm
    ${HF_ROOT}/engine/rhi_metal/metal_buffer.mm
    # Backend-agnostic engine pieces the viewport drives (same set metal_headless links, + runtime).
    ${HF_ROOT}/engine/render/render_graph.cpp
    ${HF_ROOT}/engine/scene/mesh.cpp
    ${HF_ROOT}/engine/scene/scene_io.cpp
    ${HF_ROOT}/engine/asset/gltf_loader.cpp
    ${HF_ROOT}/engine/asset/env_loader.cpp
    ${HF_ROOT}/engine/anim/animation.cpp
    ${HF_ROOT}/engine/physics/world.cpp
    ${HF_ROOT}/engine/debug/debug_draw.cpp
    ${HF_ROOT}/engine/debug/debug_emitters.cpp
    ${HF_ROOT}/engine/editor/gizmo.cpp
    # Runtime: backend-agnostic Camera / FixedTimestep / FlyCameraController (the --fly loop core).
    ${HF_ROOT}/engine/runtime/camera.cpp
    ${HF_ROOT}/engine/runtime/clock.cpp
    ${HF_ROOT}/engine/runtime/fly_camera_controller.cpp
)

add_dependencies(mac_window hf_gen_shaders_macwin)

# Compile all Obj-C++ with ARC.
target_compile_options(mac_window PRIVATE -fobjc-arc)

target_include_directories(mac_window PRIVATE
    ${HF_ROOT}/engine
    ${HF_ROOT}/third_party
)

target_compile_definitions(mac_window PRIVATE
    HF_SHADER_DIR="${HF_ROOT}/shaders"
    HF_GEN_SHADER_DIR="${HF_GEN_SHADER_DIR}"
    HF_MODEL_PATH="${HF_ROOT}/assets/models/Duck.glb"
    HF_FOX_MODEL_PATH="${HF_ROOT}/assets/models/Fox.glb"
    HF_HELMET_MODEL_PATH="${HF_ROOT}/assets/models/DamagedHelmet.glb"
    HF_TRUCK_MODEL_PATH="${HF_ROOT}/assets/models/CesiumMilkTruck.glb"
    HF_SCENE_PATH="${HF_ROOT}/assets/scenes/default.json"
    HF_ENV_PATH="${HF_ROOT}/assets/env/env.hdr"
)

target_link_libraries(mac_window PRIVATE
    "-framework Metal"
    "-framework MetalKit"
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework CoreGraphics"
    "-framework Cocoa"
    "-framework AppKit"
)
