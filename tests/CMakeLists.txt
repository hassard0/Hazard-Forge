add_executable(rhi_smoke rhi_smoke.cpp)
target_link_libraries(rhi_smoke PRIVATE hf_engine)
add_test(NAME rhi_smoke COMMAND rhi_smoke)
