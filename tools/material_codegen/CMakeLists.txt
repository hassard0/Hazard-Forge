# material_codegen: tiny host tool that turns a *.mat.json graph into a generated HLSL fragment.
# Links hf_core (pure CPU — material graph + codegen + loader; no Vulkan/SDL/Metal), so it builds on
# every platform. The build wires a custom command (see samples/hello_triangle/CMakeLists.txt) that
# runs this over assets/materials/showcase.mat.json to (re)generate shaders/generated/mat_showcase.frag.hlsl.
add_executable(material_codegen main.cpp)
target_link_libraries(material_codegen PRIVATE hf_core)
# hf_core receives the opt-in sanitizer (no-op unless HF_SANITIZE != off); this host tool links it,
# so it must be instrumented to match or the MSVC ASan container-annotation (annotate_string/vector)
# link check fails. Apply the same sanitizer to keep the windows-msvc-asan build linking.
hf_apply_sanitizer(material_codegen)
