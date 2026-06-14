add_executable(rhi_smoke rhi_smoke.cpp)
target_link_libraries(rhi_smoke PRIVATE hf_engine)
add_test(NAME rhi_smoke COMMAND rhi_smoke)

add_executable(math_test math_test.cpp)
target_link_libraries(math_test PRIVATE hf_engine)
add_test(NAME math_test COMMAND math_test)

add_executable(render_graph_test render_graph_test.cpp)
target_link_libraries(render_graph_test PRIVATE hf_engine)
add_test(NAME render_graph_test COMMAND render_graph_test)
