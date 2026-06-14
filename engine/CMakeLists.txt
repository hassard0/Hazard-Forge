add_library(hf_engine STATIC
  hal/window.cpp
  rhi/rhi_factory.cpp
)
target_include_directories(hf_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(hf_engine PUBLIC
  sdl::sdl
  vk-bootstrap::vk-bootstrap
  GPUOpen::VulkanMemoryAllocator
  vulkan-headers::vulkan-headers
  Vulkan::Loader
)
