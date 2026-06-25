# Hazard Forge editor shell: Dear ImGui rendered THROUGH the engine RHI. Windows/Vulkan-only — this
# target is NOT added on Apple, so the metal_headless path is completely unaffected.
#
# Builds:
#   - third_party Dear ImGui core (docking branch), configured for 32-bit indices via hf_imconfig.h
#   - the engine-side ImGui renderer (RHI-only) + the editor panels (ECS-driven)
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui")

add_library(hf_imgui STATIC
  "${IMGUI_DIR}/imgui.cpp"
  "${IMGUI_DIR}/imgui_draw.cpp"
  "${IMGUI_DIR}/imgui_tables.cpp"
  "${IMGUI_DIR}/imgui_widgets.cpp"
)
target_include_directories(hf_imgui PUBLIC "${IMGUI_DIR}")
# Force 32-bit ImDrawIdx so index buffers bind straight through the RHI (UINT32). Docking features
# come from the docking-branch sources we vendored.
target_compile_definitions(hf_imgui PUBLIC
  "IMGUI_USER_CONFIG=\"${CMAKE_CURRENT_SOURCE_DIR}/hf_imconfig.h\"")
# ImGui's own sources are third-party; relax the strict warnings the engine builds with.
if(MSVC)
  target_compile_options(hf_imgui PRIVATE /W0)
endif()

add_library(hf_editor STATIC
  imgui_renderer.cpp
  editor_panels.cpp
  flow_editor_panels.cpp
  profiler_view_panels.cpp
)
target_link_libraries(hf_editor PUBLIC hf_engine hf_imgui)
target_include_directories(hf_editor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
