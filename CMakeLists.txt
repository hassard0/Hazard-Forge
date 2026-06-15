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

# ---------------------------------------------------------------------------------------------------
# HF_SANITIZE: opt-in sanitizer build. off (default) | address | undefined.
#   - The DEFAULT build is byte-for-byte unaffected (this is fully opt-in via the windows-msvc-asan
#     preset / -DHF_SANITIZE=address).
#   - On MSVC, address -> /fsanitize=address (UBSan is not supported by MSVC, so 'undefined' is a
#     no-op there with a warning; it maps to -fsanitize=undefined on Clang/GCC for portability).
#   - Applied selectively via hf_apply_sanitizer(<target>) so we can scope it to the engine + the
#     pure-C++ test targets (which don't drag in ASan-hostile third-party binaries).
# ---------------------------------------------------------------------------------------------------
set(HF_SANITIZE "off" CACHE STRING "Sanitizer: off | address | undefined")
set_property(CACHE HF_SANITIZE PROPERTY STRINGS off address undefined)

function(hf_apply_sanitizer target)
  if(HF_SANITIZE STREQUAL "off")
    return()
  endif()
  if(HF_SANITIZE STREQUAL "address")
    if(MSVC)
      target_compile_options(${target} PRIVATE /fsanitize=address)
      # MSVC links the ASan runtime automatically from the compile flag; nothing extra to link.
      # Edit-and-continue / incremental + ASan are incompatible: keep debug info but no /INCREMENTAL.
      target_link_options(${target} PRIVATE /INCREMENTAL:NO)
    else()
      target_compile_options(${target} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=address)
    endif()
  elseif(HF_SANITIZE STREQUAL "undefined")
    if(MSVC)
      message(WARNING "HF_SANITIZE=undefined: MSVC has no UBSan; building without a sanitizer.")
    else()
      target_compile_options(${target} PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
      target_link_options(${target} PRIVATE -fsanitize=undefined)
    endif()
  endif()
endfunction()

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
# Editor shell (Dear ImGui through the RHI) is Windows/Vulkan-only — never added on Apple, so the
# metal_headless path is unaffected.
if(NOT APPLE)
  add_subdirectory(engine/editor)
endif()
# Material-graph codegen host tool (Slice AV): turns assets/materials/*.mat.json into a generated
# HLSL fragment at build time. Pure-CPU (links hf_core); built on every platform.
add_subdirectory(tools/material_codegen)
add_subdirectory(samples/hello_triangle)
add_subdirectory(tests)
