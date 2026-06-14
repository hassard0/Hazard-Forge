# Backend selection: Metal on Apple, Vulkan everywhere else. The RHI seam (engine/rhi/*) and the
# scene/hal/math layers are backend-agnostic and always compiled.
set(HF_ENGINE_COMMON_SOURCES
  hal/window.cpp
  rhi/rhi_factory.cpp
  scene/mesh.cpp
  asset/gltf_loader.cpp
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
    rhi_vulkan/vulkan_render_target.cpp
  )
endif()

add_library(hf_engine STATIC
  ${HF_ENGINE_COMMON_SOURCES}
  ${HF_ENGINE_BACKEND_SOURCES}
)
target_include_directories(hf_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Header-only third-party (cgltf): CGLTF_IMPLEMENTATION is defined in exactly one TU
# (engine/asset/gltf_loader.cpp).
target_include_directories(hf_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

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
