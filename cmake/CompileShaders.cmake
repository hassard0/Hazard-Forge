# hf_compile_shaders(TARGET <tgt> OUTPUT_DIR <dir> SHADERS <file:stage> ...)
# Each SHADERS entry is "<relative-hlsl-path>:<stage>" where stage is vs|ps.
#
# Find a DXC that has SPIR-V codegen. Search order (highest priority first):
#   1. VULKAN_SDK env var (LunarG SDK)
#   2. WinGet Microsoft.DirectX.ShaderCompiler portable install
#   3. System PATH (last resort — Windows SDK dxc lacks -spirv)
find_program(DXC_EXECUTABLE NAMES dxc
  HINTS
    "$ENV{VULKAN_SDK}/Bin"
    "$ENV{LOCALAPPDATA}/Microsoft/WinGet/Packages/Microsoft.DirectX.ShaderCompiler_Microsoft.Winget.Source_8wekyb3d8bbwe/bin/x64"
  REQUIRED)

function(hf_compile_shaders)
  set(oneValue TARGET OUTPUT_DIR)
  set(multiValue SHADERS)
  cmake_parse_arguments(ARG "" "${oneValue}" "${multiValue}" ${ARGN})

  set(spv_outputs "")
  foreach(entry ${ARG_SHADERS})
    string(REPLACE ":" ";" parts "${entry}")
    list(GET parts 0 src)
    list(GET parts 1 stage)
    if(stage STREQUAL "vs")
      set(profile "vs_6_0")
    elseif(stage STREQUAL "ps")
      set(profile "ps_6_0")
    else()
      message(FATAL_ERROR "Unknown shader stage: ${stage}")
    endif()

    get_filename_component(name "${src}" NAME)
    set(out "${ARG_OUTPUT_DIR}/${name}.spv")
    add_custom_command(
      OUTPUT "${out}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${ARG_OUTPUT_DIR}"
      COMMAND "${DXC_EXECUTABLE}" -spirv -T ${profile} -E main
              -fspv-target-env=vulkan1.3
              "${CMAKE_SOURCE_DIR}/${src}" -Fo "${out}"
      DEPENDS "${CMAKE_SOURCE_DIR}/${src}"
      COMMENT "DXC ${src} -> ${out}"
      VERBATIM)
    list(APPEND spv_outputs "${out}")
  endforeach()

  add_custom_target(${ARG_TARGET}_shaders DEPENDS ${spv_outputs})
  add_dependencies(${ARG_TARGET} ${ARG_TARGET}_shaders)
endfunction()
