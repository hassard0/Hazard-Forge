add_executable(rhi_smoke rhi_smoke.cpp)
target_link_libraries(rhi_smoke PRIVATE hf_engine)
add_test(NAME rhi_smoke COMMAND rhi_smoke)

add_executable(math_test math_test.cpp)
target_link_libraries(math_test PRIVATE hf_engine)
add_test(NAME math_test COMMAND math_test)

add_executable(render_graph_test render_graph_test.cpp)
target_link_libraries(render_graph_test PRIVATE hf_engine)
add_test(NAME render_graph_test COMMAND render_graph_test)

add_executable(ecs_test ecs_test.cpp)
target_link_libraries(ecs_test PRIVATE hf_engine)
add_test(NAME ecs_test COMMAND ecs_test)

add_executable(scene_io_test scene_io_test.cpp)
target_link_libraries(scene_io_test PRIVATE hf_engine)
add_test(NAME scene_io_test COMMAND scene_io_test)
