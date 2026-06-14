add_executable(hello_triangle main.cpp)
target_link_libraries(hello_triangle PRIVATE hf_engine)

set(SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/shaders")
hf_compile_shaders(
  TARGET hello_triangle
  OUTPUT_DIR "${SHADER_OUT}"
  SHADERS
    "shaders/lit.vert.hlsl:vs"
    "shaders/lit.frag.hlsl:ps"
    "shaders/shadow.vert.hlsl:vs"
    "shaders/shadow.frag.hlsl:ps"
    "shaders/post.vert.hlsl:vs"
    "shaders/post.frag.hlsl:ps"
    "shaders/sky.vert.hlsl:vs"
    "shaders/sky.frag.hlsl:ps"
)
target_compile_definitions(hello_triangle PRIVATE
  "HF_SHADER_DIR=\"${SHADER_OUT}\"")
# Absolute path to the glTF showcase model, baked at configure time (repo asset).
target_compile_definitions(hello_triangle PRIVATE
  "HF_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/assets/models/Duck.glb\"")
