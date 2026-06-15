# Two tiers of test target:
#
#   * PURE tests (math/ecs/render_graph/scene_io/commands): link hf_core (backend-agnostic, no
#     Vulkan/SDL) and receive the opt-in sanitizer (no-op unless HF_SANITIZE != off). These are the
#     AddressSanitizer-checked targets — they build+run clean under MSVC /fsanitize=address.
#       - math_test / ecs_test are header-only (math/math.h, ecs/ecs.h) but still link hf_core for
#         uniformity (it carries the include dirs and costs nothing).
#
#   * rhi_smoke: drives the LIVE Vulkan device, so it must link the full hf_engine (vk-bootstrap +
#     SDL). Those third-party binaries are not ASan-instrumented, so rhi_smoke is intentionally
#     NOT sanitized (mixing would cause MSVC ASan container-annotation link mismatches). It still
#     runs under the asan preset's ctest as an ordinary (uninstrumented) test.

# Pure, sanitizer-eligible tests.
function(hf_add_pure_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE hf_core)
  hf_apply_sanitizer(${name})
  add_test(NAME ${name} COMMAND ${name})
endfunction()

hf_add_pure_test(math_test)
hf_add_pure_test(render_graph_test)
# Render-graph AUTOMATIC BARRIER solver (Slice AS): the resource-state tracker + topo-ordered
# transition solver. Asserts the emitted barrier plan (shadow->shaderRead, sceneColor->shaderRead,
# swapchain->present) matches a hand-checked sequence, no-op (from==to) coalescing drops redundant
# transitions, and freshly imported targets start Undefined. Pure logic — the Vulkan
# synchronization-validation layer is the live device oracle; this pins the solver. ASan-eligible.
hf_add_pure_test(render_graph_barriers_test)
hf_add_pure_test(ecs_test)
hf_add_pure_test(scene_io_test)
hf_add_pure_test(commands_test)
# Skeletal-animation math + sampling (Quat/Slerp/FromTRS + SampleAnimation). Pure C++ (hf_core),
# ASan-eligible like the others.
hf_add_pure_test(anim_test)
# Instanced-rendering layout + grid builder (Slice Q). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(instance_grid_test)
# Impulse-based rigid-body world (Slice S). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(physics_test)
# Scene-graph hierarchy composition (Slice V): parent*local world-transform walk. Pure C++, ASan.
hf_add_pure_test(scene_import_test)
# Immediate-mode debug-draw collector + emitters (Slice W): primitive->line-vertex counts +
# endpoint correctness. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(debug_draw_test)
# Interactive-runtime core (Slice AA): Camera yaw/pitch->basis/view, FlyCameraController move/look
# deltas, FixedTimestep accumulator step counts. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(runtime_test)
# Editor selection/gizmo/picking (Slice AB): Mat4::Inverse, ScreenRayThroughCamera, PickNearest,
# PickGizmoAxis, ApplyDrag (translate/rotate/scale), PlayState, select->translate->DumpScene
# round-trip. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(editor_test)
# Cascaded shadow maps (Slice AD): frustum-split scheme (N increasing splits in (near,far]) +
# per-cascade ortho fit (all 8 slice corners land inside the cascade clip volume). Pure C++
# (render/csm.h, header-only), ASan-eligible.
hf_add_pure_test(csm_test)
# Spot light + perspective spot-light shadow (Slice AE): spotViewProj projection (on-axis center
# projects to clip center + [0,1] depth; off-axis projects outside the frustum) + cone attenuation
# smoothstep (1 on-axis, 0 past outer). Pure C++ (render/spot.h, header-only), ASan-eligible.
hf_add_pure_test(spot_test)
# Omnidirectional point-light shadows (Slice AF): 6-face cube view-proj (on-axis center projects to
# clip center + [0,1] depth; behind-a-face projects outside) + dominant-axis face selection +
# 3x2 tile mapping. Pure C++ (render/point_shadow.h, header-only), ASan-eligible.
hf_add_pure_test(point_shadow_test)
# Clustered / Forward+ lighting (Slice AG): exponential z-slice round-trip (SliceZ <-> SliceForViewZ),
# cluster-index from a view-space position (point lands inside the cluster AABB it maps to), light
# sphere -> cluster AABB assignment (hits home cluster, misses far ones, monotone in radius), and
# BuildClusters consistency (offsets == prefix sum, sum(count) == lightIndices.size()). Pure C++
# (render/clustered.h, header-only), ASan-eligible.
hf_add_pure_test(clustered_test)
# Screen-space reflections (Slice AH): view<->screen projection round-trip (ViewToScreenUV and
# ReconstructViewPos are mutual inverses, with the Vulkan/Metal yFlip sign) + the view-space
# reflection ray (ReflectView matches HLSL reflect; down ray off a floor goes up). Pure C++
# (render/ssr.h, header-only), ASan-eligible.
hf_add_pure_test(ssr_test)

# Volumetric fog / light shafts (Slice AJ): Henyey-Greenstein phase (g=0 isotropic 1/4pi, forward
# peak > back for g>0, integrates over the sphere to ~1) + camera-basis world-ray reconstruction
# (centered pixel == camFwd, unit camFwd projection) + Beer-Lambert transmittance. Pure C++
# (render/volumetric.h, header-only), ASan-eligible.
hf_add_pure_test(volumetric_test)

# Reflection + irradiance probes (Slice AK): the 6-face cube projection + dominant-axis face
# selection (reused from point_shadow) + the COMBINED reflection/irradiance atlas-tile mapping the
# lit_probe shader mirrors (on-axis dir -> face center; face round-trip; tile UVs inside their
# block). Pure C++ (render/probe.h, header-only), ASan-eligible.
hf_add_pure_test(probe_test)

# Machine-readable engine-state introspection (Slice AL): DescribeEngine builds a deterministic JSON
# document (engine/features/showcases/commands + scene entities + camera/lights + stats). Asserts the
# JSON parses, the entity/transform/camera/light values are present, backends == ["vulkan","metal"],
# the commands manifest includes set_transform, and the output is byte-identical across runs. Pure
# C++ (hf_core), ASan-eligible.
hf_add_pure_test(introspect_test)

# Live editor: cursor-px->NDC->ray (PixelToNdc round-trip + ScreenRayThroughCamera), click-ray
# picks the expected entity among world AABBs, ApplyDrag axis delta (prevRay/curRay), and
# FileWatcher.Poll change detection via an injected fake-stat clock (Slice AM). The live mouse
# interaction in --fly is manual-only; this exercises the deterministic logic it drives. Pure C++
# (hf_core), ASan-eligible.
hf_add_pure_test(live_editor_test)

# Temporal anti-aliasing (Slice AP): Halton(2,3) radical-inverse determinism + [0,1) range, the
# sub-pixel NDC Jitter offset (bounded by one pixel, deterministic, frame-dependent), the
# neighborhood clamp (per-channel AABB, exactly what taa_resolve.frag does), and the resolve blend
# (lerp(clampedHistory, current, alpha): alpha=1 first frame, ~0.1 steady state). Pure C++
# (render/taa.h, header-only), ASan-eligible.
hf_add_pure_test(taa_test)

# Frustum culling (Slice AQ): Gribb-Hartmann plane extraction from the column-major Vulkan-clip
# view-proj (hand-checked near/far/side planes), SphereOutside/Inside + AabbOutside p/n-vertex hand
# cases, and a cull-partition cross-checked against an independent brute-force NDC-clip reference
# over random spheres (the render-invariance contract: never false-cull a visible point). Pure C++
# (render/frustum.h, header-only), ASan-eligible.
hf_add_pure_test(frustum_test)

# GPU-driven culling (Slice AR): the CPU MIRROR of shaders/cull.comp.hlsl's ordered single-workgroup
# prefix-sum compaction + frustum-sphere test (render/gpu_cull.h). Asserts the compacted survivor
# stream + count match an independent brute-force reference IN SOURCE-INDEX ORDER over a 1024-instance
# grid and 200 random frustums (the determinism contract that makes the gpu_cull.png golden stable —
# an unordered atomicAdd append would pass a count check but reorder the stream). Pure C++ (hf_core),
# ASan-eligible.
hf_add_pure_test(gpu_cull_test)

# Live-device smoke test: full engine, never sanitized.
add_executable(rhi_smoke rhi_smoke.cpp)
target_link_libraries(rhi_smoke PRIVATE hf_engine)
add_test(NAME rhi_smoke COMMAND rhi_smoke)
