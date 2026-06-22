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
    set(extra_dxc_flags "")   # reset per-shader (only csrq sets ray-query SPIR-V extension flags)
    if(stage STREQUAL "vs")
      set(profile "vs_6_0")
    elseif(stage STREQUAL "ps")
      set(profile "ps_6_0")
    elseif(stage STREQUAL "cs")
      set(profile "cs_6_0")  # compute shader (GPU particle simulation)
    elseif(stage STREQUAL "csrq")
      # Slice RT2 — an inline-RAY-QUERY compute shader (shaders/rt_query.comp). HLSL RayQuery requires
      # Shader Model 6.5; DXC lowers it to SPIR-V RayQueryKHR via the SPV_KHR_ray_query +
      # SPV_KHR_acceleration_structure extensions. Additive — only rt_query.comp uses csrq; every existing
      # :cs entry keeps cs_6_0 (byte-for-byte unchanged SPIR-V).
      set(profile "cs_6_5")
      # DXC auto-enables SPV_KHR_acceleration_structure as a dependency of ray-query; passing it explicitly
      # is rejected by some DXC builds ("unknown SPIR-V extension"). Allowing it via -fspv-extension is the
      # portable form: SPV_KHR_ray_query named + the rest auto-pulled. If a DXC build still needs the dep
      # named, -fspv-extension=KHR (allow ALL KHR extensions) is the fallback.
      set(extra_dxc_flags -fspv-extension=KHR)
    else()
      message(FATAL_ERROR "Unknown shader stage: ${stage}")
    endif()

    get_filename_component(name "${src}" NAME)
    set(out "${ARG_OUTPUT_DIR}/${name}.spv")
    add_custom_command(
      OUTPUT "${out}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${ARG_OUTPUT_DIR}"
      COMMAND "${DXC_EXECUTABLE}" -spirv -T ${profile} -E main
              -fspv-target-env=vulkan1.3 ${extra_dxc_flags}
              "${CMAKE_SOURCE_DIR}/${src}" -Fo "${out}"
      DEPENDS "${CMAKE_SOURCE_DIR}/${src}"
      COMMENT "DXC ${src} -> ${out}"
      VERBATIM)
    list(APPEND spv_outputs "${out}")
  endforeach()

  add_custom_target(${ARG_TARGET}_shaders DEPENDS ${spv_outputs})
  add_dependencies(${ARG_TARGET} ${ARG_TARGET}_shaders)
endfunction()
