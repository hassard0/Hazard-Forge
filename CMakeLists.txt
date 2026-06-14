cmake_minimum_required(VERSION 3.25)
# Obj-C++ (Metal backend) is only enabled on Apple; elsewhere this stays a pure C++ project so
# the Windows/Vulkan build is byte-for-byte unaffected.
if(APPLE)
  project(HazardForge LANGUAGES CXX OBJCXX)
else()
  project(HazardForge LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive-)
endif()

find_package(SDL3 REQUIRED)
# Vulkan dependencies are only needed for the Vulkan backend (non-Apple).
if(NOT APPLE)
  find_package(vk-bootstrap REQUIRED)
  find_package(VulkanMemoryAllocator REQUIRED)
  find_package(VulkanHeaders REQUIRED)
  find_package(VulkanLoader REQUIRED)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
# CompileShaders.cmake requires DXC (HLSL->SPIR-V) for the Vulkan path. The Metal path uses
# hand-written MSL compiled at runtime, so it is not included on Apple.
if(NOT APPLE)
  include(CompileShaders)
endif()

enable_testing()

add_subdirectory(engine)
add_subdirectory(samples/hello_triangle)
add_subdirectory(tests)
