# Backend selection: Metal on Apple, Vulkan everywhere else. The RHI seam (engine/rhi/*) and the
# scene/hal/math layers are backend-agnostic and always compiled.
set(HF_ENGINE_COMMON_SOURCES
  hal/window.cpp
  rhi/rhi_factory.cpp
  runtime/camera.cpp
  runtime/clock.cpp
  runtime/fly_camera_controller.cpp
  runtime/hot_reload.cpp
  render/render_graph.cpp
  scene/mesh.cpp
  scene/scene_io.cpp
  scene/commands.cpp
  asset/gltf_loader.cpp
  asset/env_loader.cpp
  anim/animation.cpp
  physics/world.cpp
  debug/debug_draw.cpp
  debug/debug_emitters.cpp
  editor/picking.cpp
  editor/gizmo.cpp
  editor/introspect.cpp
)

if(APPLE)
  # Obj-C++ Metal backend. Vulkan sources are intentionally excluded on Apple.
  set(HF_ENGINE_BACKEND_SOURCES
    rhi_metal/metal_device.mm
    rhi_metal/metal_device_windowed.mm
    rhi_metal/metal_swapchain.mm
    rhi_metal/metal_shader.mm
    rhi_metal/metal_pipeline.mm
    rhi_metal/metal_compute_pipeline.mm
    rhi_metal/metal_buffer.mm
    rhi_metal/metal_command_buffer.mm
    rhi_metal/metal_texture.mm
    rhi_metal/metal_render_target.mm
    rhi_metal/metal_shader_load.mm
  )
else()
  set(HF_ENGINE_BACKEND_SOURCES
    rhi_vulkan/vulkan_device.cpp
    rhi_vulkan/vulkan_swapchain.cpp
    rhi_vulkan/vulkan_shader.cpp
    rhi_vulkan/vulkan_pipeline.cpp
    rhi_vulkan/vulkan_compute_pipeline.cpp
    rhi_vulkan/vulkan_buffer.cpp
    rhi_vulkan/vulkan_command_buffer.cpp
    rhi_vulkan/vulkan_texture.cpp
    rhi_vulkan/vulkan_pbr_material.cpp
    rhi_vulkan/vulkan_render_target.cpp
  )
endif()

add_library(hf_engine STATIC
  ${HF_ENGINE_COMMON_SOURCES}
  ${HF_ENGINE_BACKEND_SOURCES}
)
# NOTE: hf_engine is intentionally NOT sanitized. It links ASan-hostile third-party binaries
# (vk-bootstrap, SDL) built without ASan, and is consumed by the non-sanitized sample/editor;
# instrumenting it would cause MSVC ASan container-annotation (annotate_string/vector) link
# mismatches. ASan is scoped to the pure-C++ core (hf_core, below) + the pure test executables.

target_include_directories(hf_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# hf_engine always links SDL (it owns the SDL HAL window). HF_HAVE_SDL_WINDOW gates the
# SDL-dependent windowed Metal ctor MetalDevice(hal::Window&) + CreateMetalDevice(Window&) so the
# SDL-free standalone mac_window/ target (which compiles the SAME metal_device_windowed.mm) does NOT
# pull in hal/window.h / SDL — it uses the MetalDevice(void* layer,...) ctor instead.
target_compile_definitions(hf_engine PRIVATE HF_HAVE_SDL_WINDOW=1)
# Header-only third-party (cgltf): CGLTF_IMPLEMENTATION is defined in exactly one TU
# (engine/asset/gltf_loader.cpp).
target_include_directories(hf_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

# ---------------------------------------------------------------------------------------------------
# hf_core: the backend-agnostic engine core (render graph + scene/mesh + scene_io + commands +
# gltf loader). Depends ONLY on the abstract RHI interface header and the C++ stdlib — NO Vulkan/
# SDL/Metal binaries. This is the AddressSanitizer-friendly slice: the pure-C++ unit tests link
# hf_core (not hf_engine) so they can be built+run under MSVC /fsanitize=address without dragging
# in the ASan-incompatible third-party libraries. The same .cpp are also compiled (non-ASan) inside
# hf_engine for the live Vulkan/Metal builds; this duplicate compilation of a handful of small TUs
# is deliberate and keeps the sanitizer scoping clean.
add_library(hf_core STATIC
  runtime/camera.cpp
  runtime/clock.cpp
  runtime/fly_camera_controller.cpp
  runtime/hot_reload.cpp
  render/render_graph.cpp
  scene/mesh.cpp
  scene/scene_io.cpp
  scene/commands.cpp
  asset/gltf_loader.cpp
  asset/env_loader.cpp
  anim/animation.cpp
  physics/world.cpp
  debug/debug_draw.cpp
  debug/debug_emitters.cpp
  editor/picking.cpp
  editor/gizmo.cpp
  editor/introspect.cpp
)
target_include_directories(hf_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
)
hf_apply_sanitizer(hf_core)

if(APPLE)
  # ARC for all Obj-C++ TUs in this target.
  set_source_files_properties(${HF_ENGINE_BACKEND_SOURCES}
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_link_libraries(hf_engine PUBLIC sdl::sdl)
  target_link_libraries(hf_engine PUBLIC
    "-framework Metal"
    "-framework QuartzCore"
    "-framework Foundation"
    "-framework Cocoa"
  )
else()
  target_link_libraries(hf_engine PUBLIC
    sdl::sdl
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    vulkan-headers::vulkan-headers
    Vulkan::Loader
  )
endif()
