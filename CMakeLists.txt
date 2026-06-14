cmake_minimum_required(VERSION 3.25)
project(HazardForge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive-)
endif()

find_package(SDL3 REQUIRED)
find_package(vk-bootstrap REQUIRED)
find_package(VulkanMemoryAllocator REQUIRED)
find_package(VulkanHeaders REQUIRED)
find_package(VulkanLoader REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

enable_testing()

add_subdirectory(engine)
add_subdirectory(samples/hello_triangle)
add_subdirectory(tests)
