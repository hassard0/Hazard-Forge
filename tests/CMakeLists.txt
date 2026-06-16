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
# Animation state machine / cross-fade blend (Slice BL): the parameter-driven FSM over the existing
# SampleAnimation/BlendAnimations. Asserts transition conditions (speed>0.3 fires at 0.5 not 0.1;
# first-satisfied in fixed order), the cross-fade timing (BlendWeight ramps 0->1 linearly over
# duration; IsTransitioning until transitionTime>=duration then current=target), any-state (from==-1)
# transitions, scripted-timeline determinism (two runs -> identical {current,transitioning,blend}),
# and Evaluate endpoints (blend==0 palette == SampleAnimation(from); transitioning palette ==
# BlendAnimations(from,to,weight)). Pure CPU (hf_core), ZERO backend symbols, ASan-eligible.
hf_add_pure_test(anim_fsm_test)
# Instanced-rendering layout + grid builder (Slice Q). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(instance_grid_test)
# Impulse-based rigid-body world (Slice S). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(physics_test)
# Procedural terrain / heightmap (Slice BF): the deterministic Height(x,z) field + BuildTerrain mesh
# generator. Asserts Height determinism + hand-anchored sample values (vs an independent in-test
# reimplementation of the locked formula), mesh structure/counts/index-range/corner positions/
# displaced Y, finite-difference normals (unit length, +Y up, flat->+Y, slope tilt), and BuildTerrain
# determinism (two calls -> bit-identical buffers). Pure CPU (hf_core), no RHI/backend symbols,
# ASan-eligible.
hf_add_pure_test(terrain_test)
# Terrain-streaming LOD integration (Slice BJ): the TileGrid layout + BuildTerrainTile world-origin mesh
# (tile coverage + shared-edge seamlessness at equal LOD), the discrete LOD distance bands (LodFor) +
# hysteresis (LodNext doesn't coarsen within the margin after refining), per-LOD mesh vertex/index
# counts, and residency+LOD DETERMINISM (a scripted camera path replays bit-identical per-frame
# {resident ids + LODs}) + LOD refine-on-approach rebuild. Pure CPU (hf_core), no RHI/backend symbols,
# ASan-eligible.
hf_add_pure_test(terrain_stream_test)
# Distance-based scene/asset streaming (Slice BD): StreamingWorld::Update residency policy. Asserts
# radius residency + the hysteresis band (band cell keeps prior state both ways), the per-frame load
# BUDGET throttle with NEAREST-first ordering (all resident after ceil(M/K) frames), no-thrash when the
# camera oscillates in the band, and determinism (a scripted camera path replays bit-identical
# per-frame ResidentCellIds). Pure CPU (hf_core), no RHI/backend symbols, ASan-eligible.
hf_add_pure_test(streaming_test)
# Deterministic software audio mixer (Slice BB): integer/fixed-point Render (Q15 gains, int32
# accumulate, int16 hard-clamp) + the 44-byte PCM WAV writer. Asserts determinism (Render twice ->
# bit-identical), mix math + clamp, the linear pan law (endpoint-exact), ADSR Q15 at known offsets,
# and the WAV header fields. Pure CPU (hf_core), no RHI/backend symbols, ASan-eligible.
hf_add_pure_test(audio_test)
# CPU particle / VFX emitter system (Slice CC): the authorable emitter (spawn rate, lifetime, init
# velocity + spread, gravity, drag, size/color over life) simulated at a fixed timestep with a
# fixed-seed integer LCG for the spread jitter (engine/vfx/particles.{h,cpp}) — DISTINCT from the
# fixed gpu-particles compute fountain. Asserts spawn rate (~R*T over T sec, capped at maxParticles),
# lifetime/retirement (age >= lifetime), gravity+drag integration (hand-checked semi-implicit Euler;
# no-gravity/no-drag == linear), the SizeAt/ColorAt lifetime curves (endpoints + midpoint),
# DETERMINISM (same seed+steps -> bit-identical particle state two runs), and BuildBillboards (6
# camera-facing verts/particle spanning cameraRight/cameraUp, centered on pos, sized by SizeAt,
# colored by ColorAt). Pure CPU (hf_core), no RHI/backend symbols, ASan-eligible.
hf_add_pure_test(vfx_test)
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
# Docked editor panel DATA model (Slice BT): BuildPanelData (engine/editor/editor_panel_data.{h,cpp})
# computes what the Hierarchy/Inspector/Stats panels would display from the live registry, ImGui-free
# + backend-free, so the docked editor's data is asserted headlessly (data, not pixels). Asserts the
# hierarchy rows (ids/labels), the inspector for the selected entity (Transform/Material/Mesh), the
# stats counts, selection follow + out-of-range clamp + empty-scene "none", determinism, and the fixed
# code-driven dock layout VALUE. The --editor-shot capture golden-verifies the rendered chrome; this
# pins the data behind it. Pure C++ (hf_core), no RHI/backend/ImGui symbols, ASan-eligible.
hf_add_pure_test(editor_panels_test)
# Docked editor LIVE-EDIT ops + scene_io round-trip (Slice BX): ApplyTransformEdit / ApplyMaterialEdit
# (engine/editor/edit_ops.{h,cpp}) mutate the selected entity's TransformC/MaterialC on the live
# registry — the WRITE path behind the BT read-only inspector. Asserts SET-vs-ADD transform semantics
# (absolute replace vs relative delta; set wins same-field; others unchanged), out-of-range/component-
# less safe no-ops, material SET (metallic/roughness/baseColor pointer), BuildPanelData reflection (the
# read path sees the write), the scene_io round-trip (edits -> DumpScene JSON carries them -> LoadScene
# into a fresh registry reloads them), and edit-sequence determinism (twice -> identical dump). The
# --editor-edit-shot capture golden-verifies the rendered EDITED scene; this pins the ops behind it.
# Pure C++ (hf_core), no RHI/backend/ImGui symbols, ASan-eligible.
hf_add_pure_test(editor_edit_test)
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

# Screen-space global illumination (Slice BP): the cosine-weighted hemisphere kernel (K unit dirs in
# the hemisphere of the normal, deterministic, +Z hand-checked, average ~ normal), the SHARED SSR
# view<->screen reconstruction round-trip (guards the reuse), the AccumulateIndirect mean estimator
# (all-miss -> 0), and a color-bleed mini-model (a red emitter in some hemisphere dirs -> reddish
# indirect). Pure C++ (render/ssgi.h, header-only; reuses render/ssr.h), ASan-eligible.
hf_add_pure_test(ssgi_test)

# SSGI spatial denoise (Slice BR): the edge-stopping BilateralWeight tap weight (1 at center, decays
# with spatial distance, ~0 across a large depth diff / opposing normals, monotone in each term), a
# CPU bilateral-blur mini-model (variance reduction + mean preservation on a uniform surface, and
# EDGE PRESERVATION across a depth/normal seam), and determinism. Pure C++ (render/ssgi.h), ASan-eligible.
hf_add_pure_test(ssgi_denoise_test)

# Depth of field (Slice CG): the thin-lens circle-of-confusion (CoC == 0 at the focal plane, grows
# monotonically with |depth - focalDist| on the near + far sides, clamped to maxCoCpx, robust/finite at
# depth -> 0 and depth -> focalLength) + the scatter-as-gather BlurWeight (a tap contributes iff its CoC
# covers the center; a focal CoC~0 neighbor doesn't blur the center) + determinism. Pure C++
# (render/dof.h, header-only), ASan-eligible.
hf_add_pure_test(dof_test)

# Temporal SSGI accumulation (Slice BV): the per-frame jittered hemisphere kernel
# (HemisphereDirJittered = the base SSGI kernel + a FIXED per-frame golden-angle azimuth rotation;
# frame 0 == HemisphereDir; frame>0 dirs unit-length, in-hemisphere, and DIFFER from frame 0 so
# accumulation adds new samples), the fixed N=8 accumulation count, a variance-reduction mini-model
# (averaging N frames of an unbiased noisy estimator -> ~1/N variance, same mean), and determinism.
# Pure C++ (render/ssgi.h, header-only), ASan-eligible.
hf_add_pure_test(ssgi_temporal_test)

# Water rendering (Slice CF): the Gerstner displacement (single-wave closed form at t=0; sum of N ==
# sum of individuals), the ANALYTIC surface normal (unit length; flat zero-amplitude -> +Y; a known
# wave tilts the expected direction; cross-checked against a FINITE DIFFERENCE of Displace within
# tolerance, proving the derivative is right), Schlick fresnel (head-on==f0, grazing==1, monotone),
# depth-based RefractTint (depth 0 -> shallow, large -> ~deep, monotone), and DETERMINISM (same
# (x,z,t) -> bit-identical Displace/Normal). Pure CPU (render/water.h, header-only), ASan-eligible.
hf_add_pure_test(water_test)

# Volumetric clouds (Slice CH): the deterministic value-noise (integer-lattice sine-fract hash +
# trilinear interp; bit-deterministic, range [0,1], spans low/high), FBM (deterministic + bounded in
# [0,1]), the slab Density field (0 below slabBottom / above slabTop; non-zero inside a covered region;
# the height gradient makes mid-slab denser than the edges), COVERAGE (raising the threshold reduces
# total density; Density == DensityCoverage at the default), Beer-Lambert (Beer(0)==1, Beer(large)->0,
# monotone, Beer(1)==1/e), Henyey-Greenstein (peaks forward for g>0, hand-checked closed-form values,
# g=0 isotropic), and DETERMINISM (same (p,t) -> bit-identical Density). Pure CPU (render/clouds.h,
# header-only), ASan-eligible.
hf_add_pure_test(clouds_test)

# Cloud shadows on the ground (Slice CK): CloudShadow(worldPos, sunDir, t, slabBottom, slabTop, steps)
# marches from a surface point TOWARD the sun (-sunDir) through the cloud slab, accumulating optical
# depth from clouds::Density, and returns Beer(opticalDepth) in [0,1] (1 = full sun, 0 = fully
# shadowed). Verifies range [0,1], FULL SUN when the sun ray never enters the slab / passes only clear
# sky, SHADOWED (< 1) under a dense column, MONOTONE (denser column -> smaller factor),
# Beer-CONSISTENCY (== Beer(marched OD)), and DETERMINISM (same (worldPos,sunDir,t) -> bit-identical).
# Pure CPU (render/clouds.h, header-only), ASan-eligible. The lit_cloudshadow.frag variant mirrors this.
hf_add_pure_test(cloud_shadows_test)

# Screen-space projected decals (Slice BH): the decal-box local<->world TRS round-trip
# (center -> local origin; center+R*halfExtents -> local corner (0.5,0.5,0.5), incl. a rotated box),
# InsideUnitBox (boundary-inclusive), the top-down DecalUV (local.xz+0.5), and the EdgeFade falloff
# (1 at center, 0 at a face, monotonic). Pure C++ (render/decal.h, header-only), ASan-eligible.
hf_add_pure_test(decal_test)

# Data-driven post-process stack (Slice BN): the JSON config loader (ordered effect list + params,
# empty=pass-through, order preserved, unknown kind rejected), the ColorGrade lift/gamma/gain curve
# (identity + a known grade), FilmGrain determinism (pure fn of the integer pixel coord, bounded,
# zero-intensity no-op), ChromaticAberration (zero strength / center -> no aberration), and the
# stack-apply CPU mirror (== composing the per-effect evaluators in order). Pure C++
# (render/post_stack.h + post_stack.cpp), ASan-eligible.
hf_add_pure_test(post_stack_test)

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

# GPU multi-draw-indirect batching (Slice BM): the pure-CPU BUILDER (render/mdi.h) that lays out, for
# a scene of N distinct objects, the N-command indirect-args buffer (5x u32 == VkDrawIndexedIndirectCommand,
# instanceCount=1/firstInstance=0) + the parallel per-draw storage buffer (mat4 model + float4 material)
# the one vkCmdDrawIndexedIndirect(drawCount=N) consumes, indexed by gl_DrawID. Asserts the command
# layout (each command i == object i's index slice; drawCount==N), per-draw packing (PerDraw[i] round-
# trips object i's matrix+material), POD byte layouts (20/80 bytes, std430-friendly), and DETERMINISM
# (two builds -> bit-identical command + per-draw buffers). The render-invariance proof (MDI image ==
# per-object image) lives in --mdi-shot; this pins the CPU layout the GPU path reads. Pure C++ (hf_core),
# ASan-eligible.
hf_add_pure_test(mdi_test)

# Bindless textures (Slice BZ): the pure-CPU texture-index TABLE / interner (render/bindless.h) that
# assigns each UNIQUE scene texture a stable index in one large descriptor array and maps each material
# to its index. Asserts INTERNING (Intern returns stable indices, dedups identical textures, distinct
# textures -> distinct indices, first-insertion order), MATERIAL COVERAGE (BuildTable over a known
# material set covers every texture; each material resolves to the right index; shared textures collapse
# to one index), and DETERMINISM (two builds -> identical textures array + per-material indices). The
# render-invariance proof (bindless image == bound image) lives in --bindless-shot; this pins the CPU
# index table the GPU descriptor array is filled from. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(bindless_test)

# Fully-GPU-driven pass (Slice CB): the pure-CPU COMBINED builder (render/gpu_driven.h) that composes the
# BM multi-draw-indirect command + per-draw layout with the BZ bindless texture-index table into ONE
# batch. Asserts the COMBINED LAYOUT (N MDI commands + N GpuDrivenPerDraw, each carrying model+material+
# texIndex where texIndex == bindless::Intern of that object's texture; drawCount==N; 96-byte std430
# per-draw record), INDEX CONSISTENCY (distinct textures -> distinct texIndex; shared textures -> same
# texIndex via dedup; every texIndex in range), and DETERMINISM (two builds -> bit-identical commands +
# per-draw + bindless table). The render-invariance proof (GPU-driven image == per-object bound image)
# lives in --gpudriven-shot; this pins the CPU layout the fully-GPU-driven path reads. Pure C++ (hf_core),
# ASan-eligible.
hf_add_pure_test(gpu_driven_test)

# Fully-GPU-driven-CULLED pass (Slice CD): the pure-CPU cull+compact MIRROR (render/gpu_culled.h) of
# shaders/gpudriven_cull.comp.hlsl that composes the AR ordered prefix-sum compaction + frustum-sphere
# test with the CB combined per-draw (model+material+texIndex). For the FULL per-draw list + a frustum
# it frustum-culls each object's world bounding sphere (the SAME render::frustum::SphereOutside +
# render::gpu_cull::InstanceWorldSphere math the validated AR cull uses) and COMPACTS the survivors in
# SOURCE-INDEX order into the GpuDrivenPerDraw SSBO + N' MDI commands + the survivor drawCount. Asserts
# the cull+compact == an INDEPENDENT brute-force frustum.h reference IN ORDER (count + 0 false-culls),
# PER-DRAW CARRY (each survivor's per-draw carries the right model+material+texIndex; its command its
# index slice), and DETERMINISM (two builds -> bit-identical compacted per-draw + drawCount) over a
# 144-object grid + 200 random frustums. The render-invariance proof (GPU-culled image == CPU-culled
# bound image) lives in --gpucull-draw-shot; this pins the CPU layout the path reads. Pure C++ (hf_core),
# ASan-eligible.
hf_add_pure_test(gpu_culled_test)

# Hi-Z occlusion culling (Slice CJ): the pure-CPU Hi-Z math (render/hiz.h) the GPU cull compute
# (shaders/hiz_cull.comp.hlsl) mirrors. Asserts the Hi-Z BUILD (each coarser mip texel == MAX of its
# 2x2 children; dims halve to a 1x1 top; odd-dim ceil edge texels MAX their in-range child; a known
# buffer -> the expected pyramid; deterministic), the OCCLUSION TEST true-positive (a box fully behind
# a closer full-screen occluder is IsOccluded), the true-NEGATIVE safety cases (in-front / partially-
# visible / near-plane-straddle / partly-off-screen -> NOT occluded — false-culling any corrupts the
# image), the CONSERVATIVE no-false-cull parity (over 4000 random AABBs + a non-uniform Hi-Z,
# IsOccluded never culls an AABB any of whose covered mip0 texels is at/in front of its nearest depth
# — a brute-force per-texel check, 0 false-culls), and determinism. The render-invariance proof
# (occlusion-culled image == frustum-only byte-identical, occluded>0, GPU occluded==CPU) lives in
# --hiz-cull-shot; this pins the CPU math both paths read. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(hiz_test)

# Multithreaded command recording (Slice AU): the deterministic CONTIGUOUS draw-list partition
# (PartitionDraws splits [0,K) into W gap/overlap-free ranges, run-to-run identical, degenerate
# W>K/K=0/W=1/W=0) + the real worker-pool dispatch (RecordParallel marks each draw exactly once
# across W threads with no shared mutable state). The order-independent-of-W property is the
# determinism oracle the 1-vs-N byte-identical render proof rests on. Pure C++ (runtime, links
# hf_core) with REAL std::thread workers — must stay clean under MSVC /fsanitize=address.
hf_add_pure_test(parallel_record_test)

# Data-driven material / shader graph (Slice AV): graph validation (cycle detection, missing
# PBROutput, edge type-mismatch, unconnected-input defaults), CPU-interpreter parity (the per-node
# math matches hand-computed values — pins node SEMANTICS shared with the HLSL codegen), codegen
# determinism/structure (GenerateHlsl deterministic + topologically-ordered temporaries feeding the
# shared PBR core), and a .mat.json loader round-trip. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(shader_graph_test)
# Slice AZ: the new-node showcase graph the test codegens (structural asserts on the swizzle/intrinsics).
target_compile_definitions(shader_graph_test PRIVATE
  "HF_MAT3_JSON=\"${CMAKE_SOURCE_DIR}/assets/materials/showcase3.mat.json\"")
# Slice BE: the NormalMap-node material (PBROutput.normal = NormalMap(slot="normalmap")) the test
# codegens (structural asserts on the TBN transform + perturbed-normal lighting).
target_compile_definitions(shader_graph_test PRIVATE
  "HF_MATNORMAL_JSON=\"${CMAKE_SOURCE_DIR}/assets/materials/normalmap.mat.json\"")

# Live runtime material authoring (Slice AW): the reload-orchestration host logic — given a STUBBED
# compiler (success/failure, no GPU/dxc), the LiveMaterial controller swaps on success, KEEPS the old
# material + records the error on a compile/validation/parse failure, the FileWatcher fires exactly on
# an mtime change (fake-stat injected), and a malformed/invalid graph does NOT swap. The real
# dxc-subprocess compile is integration-tested via the --material-live-shot byte-equality proof;
# stubbing here keeps the unit test GPU/dxc-free + deterministic. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(runtime_material_test)

# Deterministic roll-a-ball gameplay loop (Slice AX): MakeRollGame/StepGame/ScriptedTrack over the
# impulse physics::World. Asserts determinism (ScriptedTrack twice -> bit-identical final GameState),
# the win condition (collects all 3 -> score==3, won), collection logic (player AT a pickup collects
# only it; distant collects none; at most once), and physics integration (player moves under input,
# stays grounded y~=radius, no explosion via KineticEnergy). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(roll_game_test)

# Networking / replication snapshot layer (Slice BQ): the deterministic, socket-free snapshot+delta+
# replica+interpolation core (engine/net/snapshot). Asserts snapshot serialize/deserialize round-trip
# (raw-IEEE LE, fixed field order), delta correctness (DeltaApply(prev,DeltaEncode(prev,curr))==curr for
# no-change/moved/added/removed/combined) + compression (unchanged delta < full serialize), replica==
# authority over a keyframe+delta stream through the Replicator (exact after every Receive, byte
# savings, two runs deterministic), interpolation endpoints + midpoint (lerp pos + slerp orientation
# normalized + alpha clamp), and roll-game Capture (player id 0 + uncollected pickups id 1..N at their
# positions; collected pickup drops out keeping stable ids). Pure CPU (hf_core), no RHI/backend
# symbols, ASan-eligible.
hf_add_pure_test(replication_test)

# Networking transport + client jitter-buffer / interpolation (Slice BU): the deterministic, socket-free
# SimChannel (seeded-LCG latency + packet loss + reordering) + ClientView jitter buffer over the BQ
# snapshot layer (engine/net/transport). Asserts channel determinism (same config+stream -> identical
# delivered/dropped/reordered two runs; delivered+dropped==sent; latency respected), loss/reorder
# (dropRate>0 drops some; reorderRate>0 perturbs delivery order; rates=0 = pure FIFO delay), jitter-
# buffer interpolation (clean channel render == authority at renderTick; a single dropped snapshot is
# hidden by interpolating across the gap; hold-last edge policy), and convergence/bounds (interpolated
# client stays within tolerance of the authority over a lossy stream, no unbounded drift, deterministic).
# Pure CPU (hf_core), no RHI/backend symbols, ASan-eligible.
hf_add_pure_test(net_transport_test)

# Client prediction + server reconciliation (Slice BY): the deterministic, socket-free PredictedClient
# over the AX roll-game (engine/net/prediction). Asserts prediction advances (PredictTick moves the
# predicted player ahead under local input without a snapshot), reconcile with NO misprediction
# (identical authority -> OnAuthoritative leaves the predicted now unchanged, lastMisprediction ~= 0),
# reconcile WITH a SERVER-ONLY impulse the client could not predict (before reconcile predicted differs
# from authority; after OnAuthoritative + replay predicted now == authority's TRUE corrected state,
# lastMisprediction > 0), input-buffer ack/drop + bounded growth + ordered replay, and convergence/
# determinism (full run with the server-only impulse converges to authority finalError small +
# maxMisprediction > 0; two runs byte-identical). Pure CPU (hf_core), no RHI/backend symbols,
# ASan-eligible.
hf_add_pure_test(prediction_test)

# Text / HUD layout + atlas (Slice BA): BuildFontAtlas determinism + shape (128x48, white with
# coverage alpha) and LayoutText math — 6 verts/visible glyph, quad count with spaces + newline,
# cursor advance == kGlyphPx*pxScale, atlas-cell UVs, pixel->NDC conversion, empty string => 0 quads,
# out-of-range char => blank. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(text_test)

# Material-graph introspection (Slice BI): DescribeGraphJson + ToDot over the in-memory material
# Graph. Asserts structure (nodeCount/edgeCount/output + per-node type/params/inputs/output + edge
# from/to with the right port names), unconnected PBROutput defaults (source:null + the documented
# default value), determinism (two runs byte-identical, stable node/port order), DOT well-formedness
# (starts with 'digraph', one node line per node + one edge line per edge), and GOLDEN PARITY:
# DescribeGraphJson(load(showcase3.mat.json)) byte-equals tests/golden/material/showcase3_graph.json
# (the test reads the committed golden + compares, so CI checks it too). Pure C++ (hf_core),
# ASan-eligible; ZERO backend symbols (pure CPU text over the graph).
hf_add_pure_test(graph_introspect_test)
target_compile_definitions(graph_introspect_test PRIVATE
  "HF_MAT3_JSON=\"${CMAKE_SOURCE_DIR}/assets/materials/showcase3.mat.json\""
  "HF_MAT3_GOLDEN=\"${CMAKE_SOURCE_DIR}/tests/golden/material/showcase3_graph.json\"")

# Live-device smoke test: full engine, never sanitized.
add_executable(rhi_smoke rhi_smoke.cpp)
target_link_libraries(rhi_smoke PRIVATE hf_engine)
add_test(NAME rhi_smoke COMMAND rhi_smoke)
