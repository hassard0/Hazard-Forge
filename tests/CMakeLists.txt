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
# DX2: the selective scene-query read protocol (scene::RunQueries). Pure CPU JSON over the command
# bus — determinism, field-selection + canonical order, stats counts, out-of-range error object.
hf_add_pure_test(agent_query_test)
# DX3: the declarative scene-spec authoring loop (scene::CanonicalizeScene/Text = LoadScene then
# DumpScene). Pure CPU JSON — round-trip to the canonical text, idempotence/fixed-point, two-call
# determinism, and the unknown-mesh std::runtime_error reject. ASan-eligible like the others.
hf_add_pure_test(agent_author_test)
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
# Slice CL — Clustered Light Culling (Forward+): SphereAABBIntersect (inside/touching/outside),
# ClusterViewAABB frustum tiling, AssignLights correctness + the no-false-negative brute-force parity
# (the render-invariance safety property), real-culling reduction (maxPerCluster<totalLights), and
# determinism. Pure C++ (render/cluster.h, header-only), ASan-eligible.
hf_add_pure_test(cluster_test)
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

# Per-object + camera motion blur (Slice CN): the screen-space velocity (cur-prev pixel delta from a
# prev/cur view-proj pair; EXACTLY (0,0) when the two projections are identical -> the zero-velocity
# pass-through proof), the length clamp (caps at maxBlurPx, preserves direction, zero stays zero), and
# the depth-aware gather TapWeight (an in-extent foreground/co-planar tap weight 1, a beyond-extent or
# farther-background tap 0 -> a near moving object streaks over a static background but a moving
# background does not smear a static foreground; normalized so zero velocity -> center weight 1). The
# render-invariance proof (motionblur(zeroVel) == sceneColor byte-identical) lives in --motionblur-shot;
# this pins the CPU math both the shader + the showcase read. Pure C++ (render/motion_blur.h,
# header-only), ASan-eligible.
hf_add_pure_test(motion_blur_test)

# Order-Independent Transparency (Slice CO): Weighted Blended OIT (McGuire & Bavoil) — the McGuire depth
# Weight (positive/finite for all inputs, non-increasing in depth so a near transparent surface dominates
# a far one, scales with alpha), the per-fragment Accumulate (additive accum SUM + multiplicative
# revealage PRODUCT), and the ResolveOver composite (accum.rgb/max(accum.a,eps) over the background by
# 1-revealage). The CORE proof: accumulate a fixed well-separated fragment set in canonical vs PERMUTED
# orders -> the resolved color is BIT-IDENTICAL (sum + product are commutative) — true order independence.
# The byte-identical permuted==canonical render proof lives in --oit-shot; this pins the CPU math both the
# oit_accum/oit_resolve shaders + the showcase read. Pure C++ (render/oit.h, header-only), ASan-eligible.
hf_add_pure_test(oit_test)

# Parallax Occlusion Mapping (Slice CP): the steep-parallax + binary-refine ParallaxUV march and the
# SelfShadow soft self-occlusion (render/pom.h, header-only). The CORE proof: ParallaxUV(heightScale=0)
# == baseUV EXACTLY and a straight-on view -> baseUV exactly (zero UV drift -> the heightScale=0 render
# is byte-identical to plain normal mapping). Also pins offset direction/magnitude (deeper height /
# steeper angle -> larger monotone offset), march intersection-on-surface (no overshoot), the SelfShadow
# crevice-occlusion + degenerate exits + monotone-in-heightScale, and determinism. The same pom.h math
# is mirrored VERBATIM in pom.frag.hlsl. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(pom_test)

# Ground-Truth Ambient Occlusion (Slice CR): the closed-form per-slice IntegrateArc visibility integral,
# the per-sample HorizonAngle, and the full horizon-search Visibility estimate (render/gtao.h, header-
# only). The CORE proof: IntegrateArc(full unoccluded hemisphere) == 1, Visibility over a FLAT depth
# field == 1 EXACTLY, and Visibility at radius == 0 == 1 EXACTLY (no horizon search -> the radius=0
# GTAO render is byte-identical to the no-AO scene). Also pins the known-occluder analytic match +
# monotone (closer/taller occluder -> more AO), range [0,1], IntegrateArc slice symmetry, and
# determinism. The same gtao.h math is mirrored VERBATIM in gtao.frag.hlsl. Pure C++ (hf_core),
# ASan-eligible.
hf_add_pure_test(gtao_test)

# Subsurface Scattering (Slice CZ): the skin diffusion profile weight DiffusionWeight (positive,
# peaks at distance 0, strictly decreasing, a wider sssWidthPx spreads weight), the depth-aware
# silhouette-stop DepthFalloff (1 at equal depth, -> ~0 across a depth discontinuity, monotone), and one
# separable diffusion blur BlurAxis (render/sss.h, header-only). The CORE proof: BlurAxis(sssStrength=0)
# and BlurAxis(sssWidthPx=0) return the center color EXACTLY for ANY depth/mask field (the byte-identical
# sssStrength=0 == non-SSS render proof's CPU half), plus a mask-0 pixel passes through unchanged. Also
# pins the blur behavior (a bright subsurface center spreads/softens to neighbors, a dark dip fills in,
# both normalized/energy-conserving; the depth stop blocks bleed across a silhouette) and determinism.
# The same sss.h math is mirrored VERBATIM in sss_blur.frag.hlsl. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(sss_test)

# Color Grading (Slice DB): the analytic lift/gamma/gain wheels (out = (gain*(c+lift*(1-c)))^(1/gamma)),
# the ASC-CDL slope/offset/power primary (out = max(c*slope+offset,0)^power), the luma-preserving
# Saturate (lerp(Rec.709 luma, c, saturation)), and the composed Grade (render/color_grade.h,
# header-only). The CORE proof: Grade(c, IDENTITY) == c EXACTLY (lift 0/gamma 1/gain 1/slope 1/offset 0/
# power 1/saturation 1) and each sub-op at its identity -> c EXACTLY — the CPU half of the byte-identical
# identity-grade == ungraded render proof. Also pins the monotone controls (gain brightens, lift raises
# blacks more than brights, gamma>1 brightens mids, ASC-CDL SOP hand-checks, saturation 0->greyscale &
# >1->more saturated), range/no-NaN (c=0 with power/gamma terms, the clamp>=0), and determinism. The
# same color_grade.h math is mirrored VERBATIM in color_grade.frag.hlsl. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(color_grade_test)

# Contrast-Adaptive Sharpening (Slice DF; AMD FidelityFX CAS): the adaptive negative-lobe weight
# CasWeight (amp = sqrt(clamp(min(minL,1-maxL)/max(maxL,eps),0,1)); w = amp*lerp(-1/8,-1/5,sharpness),
# forced to EXACTLY 0 at sharpness 0) and the normalized cross sharpen CasSharpen (out = (center +
# w*(up+down+left+right))/(1+4w), clamped per-channel to the neighborhood [min,max] — the no-ringing
# clamp) (render/cas.h, header-only). The CORE proof: CasSharpen(c, ..., sharpness=0) == center EXACTLY
# for ANY neighborhood and CasWeight(...,0) == 0 — the CPU half of the byte-identical sharpness=0 ==
# unsharpened render proof. Also pins the sharpening behavior (a bright/contrasty center gains contrast,
# a flat neighborhood is unchanged), the no-overshoot clamp on a hard edge, the adaptive weight (|w|
# larger in low-contrast than at a hard edge; monotone in sharpness), and determinism. The same cas.h
# math is mirrored VERBATIM in cas.frag.hlsl. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(cas_test)

# Box-Projected Cubemap Reflections (Slice DA): the slab-method ray-box EXIT t (RayBoxExitT: analytic
# exit for axis-aligned + diagonal rays, always positive) and the box-projected reflection direction
# (BoxProject, render/reflection_probe.h, header-only). The CORE proof: BoxProject(parallaxStrength=0)
# == normalize(R) EXACTLY for ANY R/P/box (the byte-identical parallaxStrength=0 == standard-reflection
# render proof's CPU half), plus the infinite-box convergence (a huge box -> ≈ normalize(R), the
# correction vanishes) and a hand-computed finite-box correction (the corrected dir leans toward the
# box-exit relative to the probe center). The same reflprobe math is mirrored VERBATIM in
# reflprobe.frag.hlsl. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(reflection_probe_test)

# Froxel Volumetric Fog (Slice CS): the exponential Z-slice mapping + its EXACT inverse
# (ViewZToSlice(SliceZ(k)) == k — the apply's depth->slice sampling correctness, an off-by-one here
# breaks the density=0==scene proof), SliceZ monotone + [zNear,zFar] coverage, height-fog density
# (falloff, baseDensity==0 -> 0 EXACTLY, non-negative), the Henyey-Greenstein phase (forward-peak,
# isotropic at g=0, finite), and the front-to-back single-scatter IntegrateStep (zero-density NO-OP ->
# T==1 & inScatter==0, constant-sigma -> exp(-sigma*L) analytic Beer-Lambert, transmittance monotone,
# near-vs-far scatter weighting), plus determinism. The same froxel.h math is mirrored VERBATIM in
# froxel_inject/integrate/apply. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(froxel_test)

# Probe-Grid Ray-Trace — DDGI beachhead (Slice DH): the world-space probe lattice + the deterministic
# Fibonacci-sphere ray set + the nearest-probe lookup + the view-space depth-field march
# (engine/render/probe_gi.h, header-only) the probe_raytrace.comp shader copies VERBATIM (so the GPU
# ray-hit SSBO is BIT-EXACT to this CPU reference). Asserts FibonacciSphere (unit-length, deterministic,
# (0,1)->(0,0,1), even full-sphere distribution: |mean|->0 + z monotone/evenly-spaced ~2/N + no dupes),
# TraceRayToDepth over a FLAT field (a ray aimed away MISSES -> kRayMiss; a ray at the plane HITS at the
# analytic depth) and a SYNTHETIC near-occluder (the ray through the occluder column hits the occluder,
# not the far plane), GetProbeGridIndex (round-trips probePos->flatIndex at every lattice point + clamps
# far-outside positions to the boundary probe), the DISABLED path (dimX/Y/Z=0 -> probeCount 0 ->
# ProbeDispatchGroups 0 -> the dispatch-0 no-op), and DETERMINISM (the full per-probe ray set is
# bit-identical across two runs). Pure C++ (hf_core; reuses render/ssr.h), no RHI/backend symbols,
# ASan-eligible.
hf_add_pure_test(probe_gi_test)

# Slice DI — DDGI Slice 2: Probe Radiance Capture. Pure CPU math for the per-probe cubemap-capture data
# layer (render/probe_capture.h, reusing render/probe_gi.h's grid + render/cubemap.h's 6-face projection):
# CaptureFaceCount (probeCount*6; 0 at dimX/Y/Z=0 -> the capture-loop skip no-op), ProbeFaceIndex (the
# probe×face flat-slot round-trip + full coverage without overlap), the FaceView/FaceProj reuse from the
# probe center (the 6 axes + the +Y/-Y up convention), the ProbeRadiance RGBA16F/6-face layout descriptor,
# and the deterministic debug-viz FaceAverage. No RHI/backend symbols, ASan-eligible.
hf_add_pure_test(probe_capture_test)

# Slice DJ — DDGI Slice 3: Probe SH-Encode. Pure CPU math for the per-probe 3rd-order real spherical-
# harmonic encode (render/probe_sh.h, reusing render/probe_gi.h's grid + Fibonacci sample set): SHBasis9
# (the 9 real-SH basis functions, polynomial in the unit direction — Y00 == 0.282095 const + the +X/+Y/+Z
# axis values hand-checked vs the documented constants, no NaN), SHEncodeAccumulate (the per-sample fma
# accumulation), SHNormalize (the solid-angle normalization + the totalWeight<=0 degenerate guard), and
# SHEvaluate (cosine-lobe irradiance reconstruction). Pins the classic SH sanity checks: a uniform
# radiance over the sphere projects to band-0 only (coeff[0] == uniform*Y00, coeff[1..8] ~= 0) + an
# isotropic reconstruction, a directional lobe reconstructs larger toward its peak than away
# (SHEvaluate(d) > SHEvaluate(-d)), zero radiance -> an EXACTLY all-zero ProbeSH (the zero-radiance ==
# zero-SH proof's CPU half), EncodeDispatchGroups (ceil(n/64); 0 at any zero dim -> the probeCount=0
# dispatch-0 no-op), and DETERMINISM (two encodes -> bit-identical SH). The same probe_sh.h math is
# mirrored VERBATIM in probe_sh_encode.comp; the GPU==CPU bit-exact SH proof lives in --probesh-shot.
# Pure C++ (hf_core; reuses render/probe_gi.h), no RHI/backend symbols, ASan-eligible.
hf_add_pure_test(probe_sh_test)

# Slice DL — DDGI Slice 4: Probe-Grid Update / Trilinear SH Interpolation. Pure CPU math for the trilinear
# probe blend (render/probe_gi.h's NearestProbes + InterpDispatchGroups, render/probe_sh.h's InterpolateSH
# + InterpolateIrradiance): the FLOOR-cell 8-corner lookup with polynomial trilinear weights (computed
# with std::fma per corner so the per-corner product is a single correctly-rounded op matching the shader's
# mad — the cross-backend bit-exactness key) and the fma-blended 27-float SH at a world position. Pins:
# partition of unity (Σw==1 over interior points, weights in [0,1]), lattice-point identity (at a probe
# position that corner has weight 1 + InterpolateSH == that probe's SH byte-identical — the round-trip),
# boundary clamp (far-outside -> in-range indices + frac saturates to a single boundary corner), dim==1
# axis degeneracy (a 1-thick axis contributes no second corner), the disabled guard (spacing<=0 / dim<=0
# -> valid=false -> InterpolateSH zero SH + InterpolateIrradiance {0,0,0}; InterpDispatchGroups 0 at any
# zero dim / probeCount / nQueries), blend linearity (8 identical SH -> that SH; monotone falloff away from
# a bright probe), and DETERMINISM (bit-identical blend). The same math is mirrored VERBATIM in
# probe_interp.comp; the GPU==CPU bit-exact blend proof + lattice-point identity live in --probeinterp-shot.
# Pure C++ (hf_core; reuses render/probe_gi.h + render/probe_sh.h), no RHI/backend symbols, ASan-eligible.
hf_add_pure_test(probe_interp_test)

# Slice DO — DDGI Visibility Slice 1: Per-Probe Distance-Moment Capture. Pure CPU math for the visibility
# data layer (render/probe_dist.h): the moment-from-distance reference MomentsFromDistance(d) == {d, d*d}
# (a single BARE multiply for the second moment so it is bit-identical to the shader's bare d*d — the
# GPU==CPU bit-exact step, given the same distance bytes; the sqrt-derived distance itself is NOT claimed
# bit-exact cross-backend), DistTexelCount == probeCount*384 (0 at any zero dim — the probeCount=0 capture-
# loop no-op), and the per-probe×face×texel flat layout p*384+f*64+v*8+u as a BIJECTION over
# [0, probeCount*384) (ProbeDistFaceIndex block start + ProbeDistTexelIndex round-trip + full coverage, no
# overlap). The same math is mirrored in probe_dist.frag (the moment output) + consumed by --probedist-shot
# (the GPU==CPU bit-exact moment proof + capture==direct render proof). Pure C++ (hf_core; reuses
# render/probe_gi.h + render/cubemap.h), no RHI/backend symbols, ASan-eligible.
hf_add_pure_test(probe_dist_test)

# DDGI GI Composite (Slice DN): the COMPOSITE-specific properties of the single-bounce DDGI indirect-diffuse
# term the lit_ddgi.frag.hlsl shader adds last — indirect = InterpolateIrradianceSH(wpos,N)*(1-metallic);
# rgb += indirect*albedo*giStrength. Pins giStrength=0 -> a literal +0.0 (the byte-identical no-op the 77
# image goldens rest on), the color-bleed DIRECTION (a red-dominant ProbeSH toward a facing normal -> a
# red-dominant indirect color), the probeCount=0 zero-indirect fallback, the metallic=1 zeroing, and
# determinism. The same math is mirrored VERBATIM in lit_ddgi.frag.hlsl (mad for every accumulation); the
# GPU==CPU bit-exact blend lives in --probeinterp-shot + the giStrength=0==no-GI BYTE-IDENTICAL render proof
# in --ddgi-shot. Pure C++ (hf_core; reuses render/probe_gi.h + render/probe_sh.h), ASan-eligible.
hf_add_pure_test(ddgi_test)

# Slice DP — DDGI Visibility Slice 2: Chebyshev Occlusion Weighting. Pure CPU math for the visibility
# leak-fix (render/probe_dist.h's DistDirToFaceUV / SampleProbeMoments / ChebyshevVisibility, mirrored
# VERBATIM in lit_ddgi_occ.frag.hlsl): the dir->(face,u,v) cube query matches DO's FaceView capture
# convention (the 6 axis dirs -> the 6 face centres; consistent with cubemap::DirToFaceUV), SampleProbeMoments
# reads back a planted texel, the Chebyshev variance-shadow closed form (dist<=mean -> 1, dist>>mean small
# var -> ≈0, var==0 -> a hard step with no 0/0 NaN), the occlusionStrength=0 identity (the all-vis==1
# renormalised weighted blend == probesh::InterpolateSH BYTE-IDENTICAL — the CPU mirror of the shader's
# occlusionStrength==0 -> VERBATIM-DN-path no-op, exact because Σ trilinear weights == 1.0 and /1.0 is exact),
# and the all-occluded -> zero-indirect fallback (no /wsum NaN). Pure C++ (hf_core; reuses render/probe_dist.h
# + render/probe_gi.h + render/probe_sh.h + render/cubemap.h), no RHI/backend symbols, ASan-eligible.
hf_add_pure_test(probe_occlusion_test)

# Slice DR — DDGI MULTI-BOUNCE (2nd light bounce). Pure CPU logic/math for the 2nd-bounce feed
# (render/probe_multibounce.h): SelectBounceSH returns SH0 at bounceCount=1 (the byte-identical no-op the
# make-or-break proof rests on — the bounce-1 capture is skipped) and SH1 at bounceCount>=2; ClampBounceCount
# clamps to [1,2]; the BounceIndirect CPU mirror of probe_bake_gi.frag's added indirect (a non-negative
# SH0-irradiance*albedo*giStrength add -> the bounce only ADDS light, SH1 irradiance >= SH0 for positive
# albedo — 2nd-bounce monotonicity); probeCount=0 -> zero indirect (the DN zero-SH fallback); and determinism.
# Pure C++ (hf_core; reuses render/probe_multibounce.h + probe_gi.h + probe_sh.h), no RHI/backend symbols,
# ASan-eligible.
hf_add_pure_test(ddgi_multibounce_test)

# Auto-Exposure (Slice CW): the histogram eye-adaptation math (render/auto_exposure.h, header-only) the
# autoexposure_histogram/reduce compute shaders mirror VERBATIM. Asserts Luminance (Rec.709: white->1,
# black->0 exactly, green->0.7152, linear), the log2-luminance LumToBin (lum<=0/below-floor -> bin 0,
# above-ceiling -> top bin, monotone) and its inverse BinToLum (monotone increasing,
# BinToLum(LumToBin(lum)) ~= lum within one bin width), the black-bin-EXCLUDED weighted AverageLuminance
# (single bin -> that bin's center luminance; two-bin -> count-weighted average; a huge black pile is
# excluded so it meters the lit content; all-black -> a finite non-zero floor), the key-value
# ExposureFromAverage (== keyValue/avg, avg==keyValue -> EV 1, monotone DECREASING so a bright scene
# darkens + a dark scene brightens, near-black -> a finite eps-floored exposure), the uniform-image exact
# EV (avg == the bin center -> exposure == keyValue/binCenter), and DETERMINISM (same histogram ->
# bit-identical average + exposure). The adaptation-off byte-identical proof lives in --autoexposure-shot;
# this pins the CPU math both the shaders + the showcase read. Pure C++ (hf_core), no RHI/backend
# symbols, ASan-eligible.
hf_add_pure_test(auto_exposure_test)

# Per-Froxel Clustered-Light Injection (Slice CV): the ADDED clustered-point-light in-scatter term
# froxel::InjectClusteredLights the inject pass (froxel_inject.comp.hlsl behind the injectLights flag)
# accumulates on top of the CS sun in-scatter — the marquee CS (froxel fog) + CL (clustered lights)
# fusion. The CORE proofs: lightCount==0 -> (0,0,0) EXACTLY (the additive lights-off identity that makes
# the inject pass reduce to CS sun-only fog -> the byte-identical lights-off proof) and density==0 ->
# (0,0,0) EXACTLY (the density=0==no-fog foundation). Also pins: a reaching light scatters POSITIVE +
# LINEAR in density, falls with distance (windowed), a light BEYOND radius -> 0 (the windowed hard cutoff
# IDENTICAL to CL's atten, incl. the d==radius boundary), color tint + HG forward-peak phase, additive
# accumulation, lightOffset slicing of the shared light-index buffer, the froxel->cluster Z map (froxel
# view-Z -> the correct cluster slice via cluster::SliceForViewZ; froxel/cluster XY tiles align 16x9),
# and determinism. The same froxel.h math is mirrored VERBATIM in froxel_inject.comp's light loop. Pure
# C++ (hf_core; reuses render/cluster.h), no RHI/backend symbols, ASan-eligible.
hf_add_pure_test(froxel_lights_test)

# Volumetric Shadows (Slice CX): the per-froxel sun-shadow VISIBILITY gate (render::froxel::SunVisibility)
# the froxel inject (froxel_inject.comp.hlsl behind the volumetricShadows flag) multiplies into the SUN
# in-scatter so the fog has shadowed gaps + visible sun light shafts, plus the CSM cascade selection
# (render::froxel::SelectCascade) mirroring lit_csm.frag's split pick. The CORE proofs: a LIT froxel
# (closer to the sun than the stored occluder) -> 1 and a DEGENERATE/out-of-cascade froxel -> 1 (the lit
# pass's shadow==1 default; with volumetricShadows=false the shader skips the gate -> visibility forced 1
# -> the byte-identical shadows-off == CV proof), a SHADOWED froxel (behind a closer occluder beyond the
# bias) -> 0, a froxel AT the occluder depth within the bias -> 1 (no self-shadow acne). Also pins: the
# light-space projection samples the froxel's OWN smUV tile + round-trips to in-[0,1] UV/depth, the cascade
# selection (depth -> expected cascade; single-cascade -> 0; vs csm::CsmSplits increasing), the
# multiplicative identity (visibility 1 leaves the sun scatter unchanged), and determinism. The same
# froxel.h SunVisibility/SelectCascade math is mirrored VERBATIM in froxel_inject.comp's volumetricShadows
# gate. Pure C++ (hf_core; reuses render/csm.h), no RHI/backend symbols, ASan-eligible.
hf_add_pure_test(vol_shadows_test)

# Screen-Space Contact Shadows (Slice CT): the screen-space depth ray-march toward the sun
# (render/contact_shadows.h, header-only). The CORE proof: RayMarchShadow(maxDist=0) == 1 and
# (steps=0) == 1 EXACTLY (no march -> no occluder -> the maxDist=0 contact-shadow render is byte-
# identical to the no-contact-shadow lit render). Also pins: a clear/empty depth field -> 1, an occluder
# within the thickness window -> < 1 (monotone: a closer/earlier occluder darkens more), an occluder
# BEYOND the thickness window -> 1 (no false shadow / haloing), a flat surface within the bias -> 1 (no
# self-occlusion acne), and determinism (incl. the maxDist=0 identity holding for any dither). The same
# contact_shadows.h march is mirrored VERBATIM in contact_shadows.frag.hlsl. Pure C++ (hf_core),
# ASan-eligible.
hf_add_pure_test(contact_shadows_test)

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

# Runtime Cubemap-Capture Reflection Probe (Slice DD): the 6 cube-face view/proj matrices (FaceView
# looks down the correct ±axis with the documented up; FaceProj's 90°-FOV square frustum projects an
# on-axis point to the face center; depth in [0,1]) + the hardware cube lookup (DirToFaceUV: 6 axes ->
# 6 faces at UV center, major-axis selection, off-axis UV in [0,1], the cube tiles all directions) +
# the +Y/-Y up-vector convention hand-check (the classic flip pitfall — captured pole faces aren't
# flipped/rotated vs how a TextureCube samples) + determinism. The same cube math is mirrored in the
# capture path + the reflection shader. Pure C++ (render/cubemap.h, header-only), ASan-eligible.
hf_add_pure_test(cubemap_test)

# Planar Reflections (Slice DE): the householder ReflectionMatrix (reflect a point across the mirror
# plane: +h above y=0 -> -h, a point ON the plane unchanged, the normal flips; INVOLUTORY R*R == I for
# several planes) + the Lengyel ObliqueNearClip (near plane == mirror plane: a point in front of the
# mirror kept (depth in [0,1]), a point behind clipped (depth<0); degenerate clip plane -> proj
# unchanged) + PlaneToView (world plane -> reflected view space, axis-aligned hand-check) + determinism.
# The same reflection math is mirrored in the reflected-camera render path + the planar.frag mirror
# shader. Pure C++ (render/planar_reflection.h, header-only), ASan-eligible.
hf_add_pure_test(planar_reflection_test)

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

# Virtual-geometry meshlet decomposition (Slice DS, Nanite-style arc beachhead): the SAME
# render/meshlet.h BuildMeshlets the --meshlet-viz showcase uses. Pins partition completeness (the
# reordered triangle multiset == the original, Sum(triCount)==T, <=128/cluster, only-last-partial),
# determinism (two builds byte-identical), the CONSERVATIVE per-cluster bounds contract (every vert in
# the AABB + radius — the contract DT/DU rely on), the degenerate/empty no-ops + non-mult-of-3 trailing-
# ignore, MortonCode10 round-trip, and hashColor determinism/spread. Pure C++ (header-only), ASan-eligible.
hf_add_pure_test(meshlet_test)

# GPU-driven culling (Slice AR): the CPU MIRROR of shaders/cull.comp.hlsl's ordered single-workgroup
# prefix-sum compaction + frustum-sphere test (render/gpu_cull.h). Asserts the compacted survivor
# stream + count match an independent brute-force reference IN SOURCE-INDEX ORDER over a 1024-instance
# grid and 200 random frustums (the determinism contract that makes the gpu_cull.png golden stable —
# an unordered atomicAdd append would pass a count check but reorder the stream). Pure C++ (hf_core),
# ASan-eligible.
hf_add_pure_test(gpu_cull_test)

# Virtual-geometry per-CLUSTER frustum cull (Slice DT): the CPU MIRROR of shaders/cluster_cull.comp.hlsl's
# ordered single-workgroup prefix-sum compaction + frustum-sphere test (render/cluster_cull.h) at the
# (instance x cluster) granularity (a DS cluster is just an object with a per-cluster bounding sphere).
# Pins BuildClusterInstances order (instance-major/cluster-minor) + conservative world radius
# (InstanceWorldSphere, never under-bounds), CullClusterInstances all-survive/none/half-cut == the
# independent frustum.h::SphereOutside subset IN SOURCE ORDER, the MdiCommand fields (indexCount==triCount*3,
# firstIndex==triOffset*3, firstInstance==source index), determinism (two builds + culls byte-identical),
# and the conservative-keep contract (a straddling sphere is KEPT). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(cluster_cull_test)

# Virtual-geometry per-CLUSTER Hi-Z OCCLUSION cull (Slice DU): the CPU MIRROR of
# shaders/cluster_hiz_cull.comp.hlsl (render/cluster_cull.h::CullClusterInstancesHiZ) — DT's frustum cull
# PLUS the Hi-Z occlusion test over the cluster's bounding-sphere AABB. Pins occlusionEnabled=false ==
# CullClusterInstances (DT) byte-identical (the disabled-path guarantee), a cluster fully behind the Hi-Z
# DROPPED / a cluster in front KEPT, conservative-keep (near-plane/off-screen -> KEPT), occlusion survivor
# count <= frustum-only, the dropped set subset of the frustum-surviving set, determinism, and MdiCommand
# field preservation. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(cluster_hiz_test)

# Virtual-geometry discrete cluster-LOD selection (Slice DV): the CPU MIRROR of
# shaders/cluster_lod_select.comp.hlsl (render/cluster_lod.h::BuildLodMeshes + SelectLod). Pins the 3
# pre-baked LODs' contiguous cluster ranges + conservative errors (LOD0 error 0, coarser = larger error +
# fewer clusters, combined buffer covered exactly), SelectLod near->LOD0 / far->coarser / forceLod0 ->
# LOD0 / distance-monotonicity / threshold boundaries (squared form) / clamp [0,2], the host-precomputed
# ProjectionScaleForScreenError == screenH/(2 tan(fovY/2)), and determinism. The squared-distance form
# (no sqrt) is the bit-exact GPU==CPU select the showcase proves. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(cluster_lod_test)

# Virtual-geometry VISIBILITY BUFFER (Slice DW): the CPU side of the (clusterID, triangleID) ID packing
# (render/visbuffer.h). Pins PackVisId/UnpackVisId round-trip over the full valid range, kTriIdBits>=7,
# packing injective (collision-free) over [0,survivorCount)x[0,128), kVisBackground never equals a valid
# packed ID, the CPU coverage reference's survivor membership == CullClusterInstances (ties DW to the
# DT/DV contract), and determinism. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(visbuffer_test)

# Virtual-geometry DEFERRED MATERIAL RESOLVE (Slice DX): the CPU mirror of the flat-shade resolve math
# (render/visresolve.h) the GPU visresolve.frag.hlsl copies verbatim. Pins FlatNormal (unit outward
# normal + winding sign-flip), LambertShade (face-on, back-facing max(0,ndotl) clamp, ambient),
# ResolveFlatShade over a known mesh (+ triOffset addressing + translation-invariant normal), ResolvePixel
# (sentinel + out-of-range clusterID -> sky, valid -> shade), EncodeBGRA8 saturate+round, and determinism.
# Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(visresolve_test)

# Nanite Software-Raster Slice 1 (Slice SW1): the CPU-reference INTEGER software rasterizer
# (render/swraster.h) the GPU compute software-rasterizer (SW2) copies verbatim + proves bit-identical.
# Pins PackSw/UnpackSw round-trip + kSwClear non-collision + nearer-key<farther-key, the TOP-LEFT fill
# rule (shared edge covered EXACTLY once — watertight), sub-pixel coverage (a sub-pixel triangle covers
# its straddled pixel), min order-independence (set vs shuffled byte-identical), degenerate zero-area ->
# no coverage, depth occlusion (nearer visId wins on overlap), empty==cleared, and determinism. Pure
# C++ (hf_core), ASan-eligible.
hf_add_pure_test(swraster_test)

# GPU Isosurface Meshing Slice 1 (Slice MC1): the per-cell MARCHING-CUBES CASE CLASSIFICATION integer
# core (render/mc.h) the GPU shaders/mc_classify.comp.hlsl copies verbatim + proves bit-identical. Pins
# the CaseIndex truth table (empty 0x00 / full 0xFF / each single-corner (1<<i) / mixed, the strict
# scalar>isovalue "inside" convention), kCornerOffset == the locked canonical MC numbering
# 0=(0,0,0)..7=(0,1,1), ClassifyCells over a known planar-boundary field (-> a known per-cell case) +
# the cellId layout, MakeSphereField's SDF shell (surface cells a non-trivial thin proper subset with
# both inside+outside cells present), the disabled (cleared) no-op model, and determinism. Pure C++
# (hf_core), ASan-eligible.
hf_add_pure_test(mc_test)

# Deterministic Rigid-Body Fracture Slice 1 (Slice FR1, the BEACHHEAD of FLAGSHIP #14): the per-sample
# NEAREST-SEED CELL classification integer core (engine/sim/fract.h) the GPU
# shaders/fract_classify.comp.hlsl copies VERBATIM + proves bit-identical. Pins the SampleIndex layout
# ((z*ny+y)*nx+x dense+monotone) + SampleCount, NearestSeed (pure int32 squared distance, on-seed d2==0,
# the STRICTLY-less lowest-index tie-break at the exact integer bisector, 3D), ClassifyFractCells over a
# 1-seed field (all cellId 0) + a 2-seed bisector split + on-seed samples, partition completeness (every
# sample assigned, all cellId in [0,M), per-cell counts sum == sampleCount), and determinism (two
# classifies byte-identical). Pure C++ (hf_core), ASan-eligible. fract.h #includes sim/fpx.h read-only.
hf_add_pure_test(fract_test)

# Deterministic Articulated-Body Ragdoll Slice 1 (Slice JT1, the BEACHHEAD of FLAGSHIP #15): the JOINT
# GRAPH + BALL-JOINT CONSTRAINT integer core (engine/sim/joint.h) the GPU
# shaders/joint_ball_solve.comp.hlsl copies VERBATIM + proves bit-identical. Reuses the fpx.h Q16.16
# toolbox (fx/fxmul/fxdiv/FxVec3/FxLength/FxNormalize/FxRotate/FxBody/IntegrateBodyFull) read-only. Pins
# WorldAnchor (identity orient -> pos+anchor; a rotated orient rotates the local anchor by orient),
# SolveBallJoint (two equal-mass bodies with offset anchors -> both move half the gap to coincidence; a
# pinned body -> only the dynamic one moves; coincident anchors -> no-op; both pinned -> no-op;
# off-centre anchors pull the anchor POINTS together), and StepJointWorld (a 1-joint chain settles with
# the gap shrinking + the dynamic body below the pinned root; a 4-link chain hangs; two runs
# byte-identical). JT1 is the POSITIONAL ball joint only (translates centres, NO lever-arm/angular
# coupling — JT2+). Pure C++ (hf_core), ASan-eligible. joint.h #includes sim/fpx.h read-only.
hf_add_pure_test(joint_test)

# Deterministic Vehicle Physics Slice 1 (Slice VH1, the BEACHHEAD of FLAGSHIP #16): the SUSPENSION SPRING
# JOINT integer core (engine/sim/vehicle.h) the GPU shaders/vehicle_spring_solve.comp.hlsl copies VERBATIM
# + proves bit-identical. Reuses the fpx.h Q16.16 toolbox (fx/fxmul/fxdiv/FxVec3/FxLength/FxNormalize/
# FxBody/IntegrateBodyFull) + joint.h (WorldAnchor/FxDot) read-only. Pins SolveSpringJoint (a body stretched
# beyond restLen pulled IN, compressed below restLen pushed OUT, at exactly restLen ~no positional
# correction, a pinned body -> only the dynamic one moves, the velocity damping reduces the relative normal
# speed, damping=0 -> no velocity change) + StepSpringWorld (a displaced body on a damped spring settles at
# ~restLen at rest; two runs byte-identical; the undamped control keeps moving). Pure C++ (hf_core),
# ASan-eligible. vehicle.h #includes sim/joint.h + sim/fpx.h read-only.
hf_add_pure_test(vehicle_test)

# Deterministic Active Ragdoll Slice 1 (Slice AC1, the BEACHHEAD of FLAGSHIP #17: DETERMINISTIC ACTIVE
# RAGDOLL / PHYSICAL-ANIMATION BLENDING): the ANGULAR POSE-DRIVE primitive integer core (engine/sim/active.h)
# the GPU shaders/active_drive_solve.comp.hlsl copies VERBATIM + proves bit-identical. Reuses the joint.h
# toolbox (SolveAngularLimit's inverse-mass nlerp apply / QConj / QNlerp) + fpx.h (FxQuatMul/FxQuatNormalize/
# IntegrateBodyFull) read-only. Pins SolveAngularDrive (stiffness kOne snaps qrel = qA⁻¹·qB to qTarget within
# an LSB band; stiffness 0 leaves qrel unchanged; a pinned body is NOT rotated; the inverse-mass split
# rotates the lighter body more; out-of-range / both-pinned skip) + DriveAngleCos (on-target kOne, off-target
# below) + StepDriveWorld (a 3-link chain drives to + HOLDS the L-pose [DriveAngleCos near kOne]; the
# stiffness-0 control hangs straight; two runs byte-identical). Pure C++ (hf_core), ASan-eligible. active.h
# #includes sim/joint.h + sim/fpx.h read-only.
hf_add_pure_test(active_test)

# Deterministic GPU Crowds Slice 1 (Slice BD1, the BEACHHEAD of FLAGSHIP #18: DETERMINISTIC GPU CROWDS —
# boids + steering + path-following): the INTEGER STEERING PRIMITIVE integer core (engine/sim/boids.h) the GPU
# shaders/boids_steer.comp.hlsl copies VERBATIM + proves bit-identical. Reuses the fpx.h Q16.16 toolbox
# (fx/kFrac/kOne/fxmul/FxVec3/FxAdd/FxSub/FxScale) read-only. Pins SteerSeek (the force points target-ward,
# zero at the target, scales with seekGain), SteerSeparation (a too-close neighbor pushes the agent away,
# beyond sepRadius contributes nothing, symmetric neighbors cancel, exactly on the radius is excluded),
# StepBoids (a tight cluster seeking a target spreads + advances [meanToTarget drops, minSep above a floor];
# the sepGain=0 control collapses; the per-axis clamp caps force/speed; two runs byte-identical). Pure C++
# (hf_core), ASan-eligible. boids.h #includes sim/fpx.h read-only.
hf_add_pure_test(boids_test)

# Deterministic Fixed-Point Physics Slice 1 (Slice FPX1, the BEACHHEAD of FLAGSHIP #6): the Q16.16
# fixed-point INTEGRATOR + integer broadphase core (engine/sim/fpx.h) the GPU
# shaders/fpx_integrate.comp.hlsl copies VERBATIM + proves bit-identical. Pins fxmul known products
# incl negatives + the int64 intermediate (no overflow within the +-32768 bound) + arithmetic-shift
# floor semantics, FxISqrt floor(sqrt) on known squares + FxLength 3-4-5, IntegrateBody/IntegrateStep
# one-step + K-step semi-implicit-Euler closed form (exact Q16.16 fall) + the ground floor clamp
# (pos.y==groundY, vel.y zeroed on contact) + static bodies never move + the disabled no-op model,
# determinism (two runs byte-identical), and BroadphaseCell/CellId/FloorDiv incl NEGATIVE coords
# (correct floor, not truncate-toward-zero). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(fpx_test)

# Hardware Ray Tracing Slice 1 (Slice RT1, the BEACHHEAD of FLAGSHIP #28: HARDWARE RAY TRACING): the
# deterministic Q16.16 SW reference ray tracer (engine/render/rtrace.h) the GPU shaders/rt_trace.comp.hlsl
# copies VERBATIM + proves bit-identical. Reuses the fpx.h Q16.16 toolbox (fx/fxmul/FxVec3/FxISqrt)
# read-only + a local render-only fxdiv. Pins IntersectSphere (a ray at the sphere center hits at the
# closed-form t==dist-radius; a ray pointed away misses), IntersectAabb (a ray into a known face hits at
# the slab-entry t with the correct face normal; a parallel-outside ray misses), miss handling +
# kRtMiss/kRtNoHit, TraceClosest nearest-by-(t,primIndex) INCLUDING an equal-t tie-break (smaller
# primIndex wins, storage-order-independent), degenerate dir==0 rejection, RenderScene two-run
# byte-identical determinism (+ a visible scene), and ShadeHitInt integer-exactness (a fixed normal under
# a fixed light -> a PINNED RGBA8; a miss -> the background). Pure C++ (hf_core), ASan-eligible.
# rtrace.h #includes sim/fpx.h read-only.
hf_add_pure_test(rt_test)

# Hardware Ray Tracing Slice 2 (Slice RT2, FLAGSHIP #28): the CPU-side invariants the HW inline-ray-query
# kernel (shaders/rt_query.comp) rests on. The HW==SW-GPU==CPU memcmp proof lives in the --rt2-query-shot
# showcase (needs a GPU); this pure-CPU test pins the deterministic-reconciliation contract: the RT2 scene's
# rtrace::RenderScene is two-run byte-identical; the AABB-MARGIN INVARIANT (every true fx hit point produced
# by the frozen intersection lies inside its float bounding AABB inflated by kRtAabbMargin = 1/64 — the
# candidate-completeness guarantee that the driver's float BVH overlap never skips a true hit); and the
# per-pixel HIT MASK + closest-prim shade match an independent brute-force no-cull reference (the oracle the
# HW path is reconciled against). Defines the RT2 scene locally with the frozen rtrace:: types (rtrace.h is
# BYTE-FROZEN, #included read-only). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(rt_query_test)

# Hardware Ray Tracing Slice 3 (Slice RT3, FLAGSHIP #28): the CPU-side invariants the HW hard-shadow kernel
# (shaders/rt_shadow.comp) rests on. The HW==CPU memcmp proof lives in the --rt3-shadow-shot showcase (needs
# a GPU); this pure-CPU test pins the deterministic shadow contract: TraceAnyHit is analytic (a point under
# a sphere toward an overhead light is occluded; an open point is not) AND order-independent (shuffling the
# scene's primitive storage yields the IDENTICAL any-hit boolean — the HW-early-out == SW-full-scan
# guarantee); ShadeHitShadowed == ShadeHitInt when NOT occluded and strictly darker (ambient-only) when
# occluded; RenderSceneShadowed is two-run byte-identical; and 0 < shadowed < hits on the RT2 scene (a real
# occlusion, not all-lit/all-dark). Defines the RT2 scene locally with the frozen rtrace:: types (rtrace.h
# is APPEND-ONLY/byte-frozen, #included read-only). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(rt_shadow_test)

# Hardware Ray Tracing Slice 4 (Slice RT4, FLAGSHIP #28): the CPU-side invariants the HW mirror-reflection
# kernel (shaders/rt_reflect.comp) rests on. The HW==CPU memcmp proof lives in the --rt4-reflect-shot showcase
# (needs a GPU); this pure-CPU test pins the deterministic reflection contract: FxReflect is analytic (an
# axis-plane mirror flips only the normal component; reflecting twice returns the original — the involution)
# AND the per-channel integer blend is EXACTLY cS at refl=0 (so a non-reflective pixel is byte-identical to
# the RT3 shadowed shade) and EXACTLY cR at refl=kOne; a no-reflective-prim scene makes RenderSceneReflected
# byte-identical to RenderSceneShadowed; on the RT2 scene (reflective ground + mirror sphere) the reflected
# image DIFFERS from the matte shadowed image at >=1 pixel (the reflection contributed) with reflective>0; and
# RenderSceneReflected is two-run byte-identical. Defines the RT2 scene locally with the frozen rtrace:: types
# (rtrace.h is APPEND-ONLY/byte-frozen, #included read-only). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(rt_reflect_test)

# Hardware Ray Tracing Slice 6 (Slice RT6, FLAGSHIP #28, the MONEY-SHOT CAPSTONE): the CPU-side invariants the
# HW lit-hero kernel (shaders/rt_hero.comp) rests on. The HW==CPU memcmp proof lives in the --rt6-hero-shot
# showcase (needs a GPU); this pure-CPU test pins the deterministic hero contract: SkyGradient is monotonic in
# dir.y (straight-up == the pinned zenith, straight-down == the pinned horizon, distinct, each channel marches
# monotonically as dir.y rises); RenderSceneHero is BYTE-IDENTICAL to RenderSceneReflected on an ENCLOSED scene
# where no ray misses (the sky is never sampled — the ONLY change is the miss color); the curated hero scene
# has shadowed>0 AND reflective>0 AND the sky is GRADED (>=2 distinct primary-miss colors); RenderSceneHero is
# two-run byte-identical (image + counts). Defines the hero scene locally with the frozen rtrace:: types
# (rtrace.h is APPEND-ONLY/byte-frozen, #included read-only). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(rt_hero_test)

# Hardware Ray Tracing Slice 5 (Slice RT5, FLAGSHIP #28): the DETERMINISM-ENVELOPE + LOCKSTEP TIE-IN — the
# pure-CPU invariants the --rt5-simrender-shot / --rt5-simrender showcases rest on (the HW==CPU memcmp lives
# in the Vulkan showcase, which needs a GPU). RT5 COMPOSES the determinism moat (fpx's deterministic Q16.16
# rigid-body sim, lockstep-replayable) with the RT4 render. This test pins the joint invariant: fpx::RunLockstep
# authority==replica BYTE-IDENTICAL (inputs-only re-sim); the shared rt5::BodiesToRtScene bridge is PURE (two
# calls byte-equal; B bodies -> B+1 prims; ground primIndex 0 reflective; each FxBody -> RtSphere{pos,radius,1+i});
# RenderSceneReflected(authScene) == RenderSceneReflected(replicaScene) (RT(authority)==RT(replica) — a deterministic
# sim renders reproducibly); and the RT render does NOT mutate the fpx world (byte-identical pre==post — a pure
# read). Uses the SAME converter (tests/rt5_simrender_scene.h) the showcases use; rtrace.h + fpx.h are byte-frozen,
# #included read-only. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(rt_simrender_test)

# Deterministic Lumen-class GI Slice 1 (Slice GI1, the BEACHHEAD of FLAGSHIP #29): the CPU-side invariants
# the HW probe-trace kernel (shaders/gi_probe_trace.comp) rests on. The HW==CPU memcmp proof lives in the
# --gi1-probe-shot showcase (needs a GPU); this pure-CPU test pins the deterministic GI contract: the 16
# baked Fibonacci-sphere directions (gi::kGiProbeDirs) are unit-length within an fx epsilon AND distinct;
# the integer ProbePos lattice round-trips; TraceProbeRays is analytic (a probe facing a lit surface sees
# radiance>0, a shadowed probe-ray is darker, an open-space probe -> sky/background); TraceProbeRays is
# two-run byte-identical; and ProbeCount==0 -> the radiance buffer is UNTOUCHED (the dispatch-0 no-op).
# Uses the pinned gi::BuildGi1Scene enclosure; gi.h #includes render/rtrace.h read-only (byte-frozen),
# rtrace.h #includes sim/fpx.h read-only. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(gi_probe_test)

# Deterministic Lumen-class GI Slice 2 (Slice GI2, THE CRUX of FLAGSHIP #29): the CPU-side validation that
# SETTLES THE INTEGER-SH CRUX before any GPU trust. The GPU SH == CPU SH memcmp proof lives in the
# --gi2-shencode-shot showcase (needs a GPU); this pure-CPU test pins the integer 3rd-order SH contract:
# zero radiance -> zero SH; a constant ambient -> only the DC (l=0) coeff dominates (SH orthogonality); a
# directional input -> band-2 coeffs NON-ZERO (NO underflow-to-zero) and in-range; the encode round-trips
# the directional SENSE (a +Y-lit probe reads brighter in +Y); FxSHEvaluate is clamped >= 0; the
# DYNAMIC-RANGE ENVELOPE on the pinned GI1 scene (no coeff / no reconstructed irradiance exceeds the < 2.0
# Q16.16 headroom, band-2 non-zero) — the int64-accumulator "stay wide, narrow once" bound; and two encodes
# are byte-identical (determinism). gi.h #includes render/rtrace.h read-only (byte-frozen), rtrace.h
# #includes sim/fpx.h read-only. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(gi_sh_test)

# Deterministic Lumen-class GI Slice 3 (Slice GI3, the CONTINUOUS IRRADIANCE FIELD of FLAGSHIP #29): the
# pure-CPU validation of the INTEGER TRILINEAR SH INTERPOLATION (engine/render/gi.h::FxNearestProbes /
# FxInterpolateSH / FxInterpolateIrradiance / InterpolateField the gi_interp.comp shader runs VERBATIM). It
# settles the integer trilinear blend before any GPU trust: the 8 corner weights sum to kOne EXACTLY
# (partition of unity, maxErr 0); the 8 corner indices are the correct cell corners; lattice-point identity
# (a query@probe reproduces that probe's FxSHEvaluate BYTE-exact); cell-center -> w == kOne/8 each;
# out-of-grid clamps to the boundary; a point nearer a brighter probe reads brighter; two runs byte-identical.
# gi.h #includes render/rtrace.h read-only (byte-frozen), rtrace.h #includes sim/fpx.h read-only. Pure C++
# (hf_core), ASan-eligible.
hf_add_pure_test(gi_interp_test)

# Deterministic Lumen-class GI Slice 4 (Slice GI4, INTEGER MULTI-BOUNCE FEEDBACK of FLAGSHIP #29): the
# pure-CPU validation of the INTEGER MULTI-BOUNCE feedback (engine/render/gi.h::ShadeHitGI / BounceProbes the
# gi_bounce.comp shader runs VERBATIM). It settles the second make-or-break beat (after GI2's crux) before any
# GPU trust: the NO-OP CONTRACT (ShadeHitGI with a zero indirect == UnpackRadiance(ShadeHitShadowed) BYTE-exact
# -> K==1 BounceProbes == the GI1+GI2 single-bounce SH EXACTLY); the MONOTONICITY guard (SH_2 +Y irradiance >=
# SH_1 component-wise AND strictly greater for >=1 probe — the integer indirect did NOT underflow to zero);
# convergence/in-range (every coeff AND maxIrr across K<=3 stays < 2.0 — the GI2 [-2,2] headroom holds under
# the bounded feedback); a colored-wall TINT (the per-channel irradiance gain is imbalanced — light bounces
# off the red wall and tints); two-run determinism; and the ProbeCount==0 no-op. gi.h #includes
# render/rtrace.h read-only (byte-frozen), rtrace.h #includes sim/fpx.h read-only. Pure C++ (hf_core),
# ASan-eligible.
hf_add_pure_test(gi_bounce_test)

# Deterministic Lumen-class GI Slice 5 (Slice GI5, INTEGER CHEBYSHEV OCCLUSION WEIGHTING — the probe-volume
# light-leak fix of FLAGSHIP #29): the pure-CPU validation of the INTEGER Chebyshev occlusion weighting
# (engine/render/gi.h::FxProbeMoments_All / FxChebyshevVisibility / FxInterpolateIrradianceOcc the
# gi_occ.comp shader runs VERBATIM). Settles the leak-fix before any GPU trust: the per-probe distance
# moments are deterministic + a VALID variance (meanDist2 >= meanDist²); the Chebyshev visibility is kOne for
# d<=meanDist, STRICTLY decreasing for d>meanDist, bounded to [0,kOne]; occStrength==0 -> the occlusion-
# weighted interp is BYTE-IDENTICAL to the unoccluded GI3 FxInterpolateIrradiance (the no-op contract); the
# re-normalized weights sum to kOne EXACTLY (partition of unity preserved); a behind-occluder query is
# ATTENUATED vs the unoccluded blend (the leak reduced); two-run determinism + the empty-query no-op. gi.h
# #includes render/rtrace.h read-only (byte-frozen), rtrace.h #includes sim/fpx.h read-only. Pure C++
# (hf_core), ASan-eligible.
hf_add_pure_test(gi_occ_test)

# Deterministic Lumen-class GI Slice 6 (Slice GI6, THE LIT GI HERO CAPSTONE — the FLAGSHIP #29 MONEY-SHOT):
# the pure-CPU validation of the lit GI hero render (engine/render/gi.h::RenderSceneGI the gi_hero.comp shader
# runs VERBATIM). Settles the money-shot before any GPU trust over the pinned BuildGi6Scene Cornell enclosure
# + the GI1->GI5 bake (BounceProbes(K=3) + FxProbeMoments_All): RenderSceneGI(giStrength=0) is BYTE-IDENTICAL
# to the no-GI hero (the indirect is a true +0, the no-op contract); the indirect is PURELY ADDITIVE
# (giStrength>0 -> every channel >= the no-GI image, never darkens); the COLOR-BLEED invariant (a red-wall-
# adjacent floor point's indirect R > G,B + a green-wall floor point's indirect G > R,B — multi-bounce
# colored GI); two-run determinism; and maxIrr of the baked field stays < 2.0 (the GI2 headroom). The GPU==CPU
# image memcmp proof lives in --gi6-hero-shot. gi.h #includes render/rtrace.h read-only (byte-frozen),
# rtrace.h #includes sim/fpx.h read-only. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(gi_hero_test)

# Deterministic Convex Rigid-Body Contacts Slice 1 (Slice CX1, the BEACHHEAD of FLAGSHIP #19): the Q16.16
# integer box-box SEPARATING-AXIS TEST core (engine/sim/convex.h) the GPU shaders/convex_sat.comp.hlsl
# copies VERBATIM + proves bit-identical. Reuses the fpx.h Q16.16 toolbox (fx/fxmul/fxdiv/FxVec3/FxRotate/
# FxNormalize/FxLength/FxISqrt) read-only. Pins FxDot/FxCross hand-computed (the right-hand basis x×y==z),
# FxMat3 identity/diagonal/MulVec/transpose (the CX3 inertia-tensor type's CX1 helpers), and BoxSat (the
# 15-axis box-box SAT): two clearly-separated boxes -> overlap=false; a box deeply inside another ->
# overlap=true with a face-axis min-pen + penetration>0; a face-face touching pair the ~0-pen boundary; an
# edge-edge skew pair picks an EDGE-CROSS axis (index >= 6); the min-pen strict-< tie-break keeps the LOWEST
# index; two runs byte-identical; MeasureSat deterministic overlap-count + mean/min penetration. Pure C++
# (hf_core), ASan-eligible. convex.h #includes sim/fpx.h read-only.
hf_add_pure_test(convex_test)

# Deterministic Contact Friction Slice 1 (Slice FC1, the BEACHHEAD of FLAGSHIP #20: DETERMINISTIC TANGENTIAL
# CONTACT FRICTION, hf::sim::fric): the integer TANGENT-BASIS core (engine/sim/fric.h) the GPU
# shaders/fric_basis.comp.hlsl copies VERBATIM + proves bit-identical. fric.h #includes sim/convex.h
# read-only (transitively fpx.h). Pins MakeTangentBasis: +z gives two unit tangents in the xy-plane mutually
# orthogonal + orthogonal to z; an oblique normal gives an orthonormal (t1,t2) (the three dot products ~0,
# the two lengths ~kOne within an integer epsilon); the least-aligned cardinal axis is the FIXED argmin
# (lowest-index tie-break, NOT the most-aligned axis); the axis-aligned normals are all degeneracy-safe;
# MeasureBasis deterministic orthonormality summary; two runs byte-identical. Pure C++ (hf_core),
# ASan-eligible.
hf_add_pure_test(fric_test)

# Deterministic Persistent Contacts Slice 1 (Slice PS1, the BEACHHEAD of FLAGSHIP #21: DETERMINISTIC
# WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, hf::sim::persist): the PURE-INT32 CONTACT FEATURE ID core
# (engine/sim/persist.h) the GPU shaders/persist_key.comp.hlsl copies VERBATIM + proves bit-identical.
# persist.h #includes sim/fric.h read-only (transitively convex + fpx). PURE INT32 (compares + shifts + xors,
# NO Q16.16 products) -> MSL-native, a TRUE GPU pass on both backends. Pins MakeContactKey order-normalization
# (MakeContactKey(i,j,..) == MakeContactKey(j,i,..) — bodyA<bodyB always); distinct features (pair/axis/clip
# corner) -> distinct keys; identical features -> equal keys + equal hash (the warm-start cache match); a
# self-pair takes no swap; ContactKeyHash deterministic + collision-light over the showcase contact set;
# MeasureKeys deterministic total/distinct summary collapsing duplicates; two runs byte-identical. Pure C++
# (hf_core), ASan-eligible.
hf_add_pure_test(persist_test)

# General Convex-Hull Contacts Slice 1 (Slice GJ1, the BEACHHEAD of FLAGSHIP #22: DETERMINISTIC GENERAL
# CONVEX-HULL CONTACTS via integer GJK + EPA, hf::sim::gjk): the integer SUPPORT-FUNCTION core
# (engine/sim/gjk.h) the GPU shaders/gjk_support.comp.hlsl copies VERBATIM + proves bit-identical. gjk.h
# #includes sim/convex.h read-only (transitively fpx.h). Pins SupportLocal (maximal vs brute force over the
# canonical tetra/box/octa/wedge hulls x a direction sweep; the strict-greater scan keeps the LOWEST tied
# index); Support (matches the hand-computed conjugate-rotate-in world support for a rotated body);
# SupportMinkowski (== Support_A(dir) - Support_B(-dir)); MeasureSupport a PURE function (two calls byte-equal).
# Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(gjk_test)

# Deterministic Integer CCD Slice 1 (Slice CD1, the BEACHHEAD of FLAGSHIP #24): the integer TIME-OF-IMPACT
# primitive (engine/sim/ccd.h) the GPU shaders/ccd_toi.comp.hlsl copies VERBATIM + proves bit-identical.
# Reuses the gjk/fpx Q16.16 toolbox read-only (via sim/broad.h). Pins ConservativeAdvance (straight-approach
# TOI within the hand-computed band + non-overlapping at the TOI pose; rotating-approach does not overshoot;
# receding/tangential -> hit=0,toi=dt; already-overlapping -> toi=0,hit=1), ClosingSpeedBound (conservative,
# the angular term mandatory), and determinism (two calls byte-equal). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(ccd_test)

# Deterministic Hull Narrowphase Hardening Slice 1 (Slice MF1, the BEACHHEAD of FLAGSHIP #25): the pure-CPU
# integer HULL FACE TOPOLOGY primitive (engine/sim/manifold.h) — per-hull POLYGON face tables
# (BuildCanonicalFaces: tetra4/box6/octa8/wedge5) promoted from gjk::HullTriIndices' canonical triangle
# groupings, outward-canonicalized FaceNormalWorld (the gjk::EpaAddFace idiom), and the SupportFace/IncidentFace
# reference/incident selectors (FxDot rank, strict-greater lowest-index tie-break) + the render-only float
# FacesToRenderInstances. manifold.h #includes sim/ccd.h read-only (gjk/broad/convex/fpx BYTE-FROZEN). Pins the
# 4 proofs: face counts {4,6,8,5}; every face outward-wound (minDot>0) + valid indices; SupportFace/IncidentFace
# battery two-run byte-equal + box +X/-X opposing-face relation; FacesToRenderInstances pure (two calls
# byte-equal). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(manifold_test)

# Warm-Started Hull Contacts Slice 1 (Slice WH1, the int32 MSL-native BEACHHEAD of FLAGSHIP #26: WARM-STARTED
# HULL CONTACTS + ROBUST DETERMINISTIC STACKING, hf::sim::warmhull): the pure-CPU integer HULL CONTACT FEATURE
# ID core (engine/sim/warmhull.h) the GPU shaders/warmhull_key.comp.hlsl copies VERBATIM + proves bit-identical.
# warmhull.h #includes sim/manifold.h read-only (gjk/convex/fpx BYTE-FROZEN). Pins THE CRUX
# (ClipFaceAgainstFaceTagged's OUTPUT POSITIONS BYTE-EQUAL to the frozen manifold::ClipFaceAgainstFace over the
# box-on-box/tetra-on-face/edge-on-face battery), distinct contacts -> distinct keys, sub-LSB nudge -> SAME key
# (the warm-start match), a sliding/flipping contact -> the key CHANGES (the honest warm-start-miss boundary),
# order-normalization (swap bodyA/bodyB + flip refIsA -> SAME key), the tag encodings never collide, and
# MeasureHullKeys determinism. The KEY is PURE INT32 (compares/shifts/xors, NO products) -> MSL-native. Pure C++
# (hf_core), ASan-eligible.
hf_add_pure_test(warmhull_test)

# Hull Friction + Joints Slice 1 (Slice HF1, the friction BEACHHEAD of FLAGSHIP #30: hf::sim::hullfric): the
# integer friction-extended keyed hull manifold (engine/sim/hullfric.h, a NEW additive sibling #including
# sim/warmhull.h + sim/gjk.h + sim/fric.h + sim/persist.h + sim/manifold.h READ-ONLY/BYTE-FROZEN) the GPU
# shaders/hullfric_points.comp copies VERBATIM. Pins MakeTangentBasis orthonormal on a spread of EPA normals,
# BuildHullFrictionManifold's points/depths/keys == the FROZEN warmhull manifold (HF1 is additive over the
# geometry — it appends the basis + zeroed accumulators), basisAxis == fric::LeastAlignedAxis(normal),
# MatchHullFrictionCache warm-seeds on a key+basis match AND cold-starts the tangents on a basis-axis FLIP (the
# crux groundwork), UpdateHullFrictionCache evict/rewrite, and two-run determinism. Pure C++ (hf_core),
# ASan-eligible.
hf_add_pure_test(hullfric_test)

# Hull Friction + Joints Slice 2 (Slice HF2, the friction SOLVER of FLAGSHIP #30: hf::sim::hullfric): the
# accumulated, warm-started, cone-clamped Coulomb friction SOLVE over the HF1 hull manifold (engine/sim/hullfric.h
# APPEND) the GPU shaders/hullfric_warm.comp copies VERBATIM — the persist::SolveFrictionWarm body with the hull
# manifold::WorldInvInertiaFull tensor (exactly how warmhull::SolveHullManifoldWarm adapted the normal-only warm
# solver). Pins SolveHullFrictionWarm convergence (residual non-increasing with iters), the warm-start lever
# (warm-primed residual < cold at equal iters), the cone invariants (accumNormal>=0, |jt|<=mu*jn), the mu=0
# frictionless control, friction biting (high-mu removes more tangential slip), and two-run determinism. Pure C++
# (hf_core), ASan-eligible.
hf_add_pure_test(hullfric_warm_test)

# Hull Friction + Joints Slice 3 (Slice HF3, the friction-locked WORLD STEP of FLAGSHIP #30: hf::sim::hullfric): the
# full per-tick friction-locked hull world step (engine/sim/hullfric.h APPEND) the GPU shaders/hullfric_step.comp
# copies VERBATIM — the warmhull::StepWarmHullWorld 5-pass tick (integrate -> all-pairs broadphase -> warm cone
# solve -> position de-pen -> orientation) with the HF2 SolveHullFrictionWarm swapped in + the per-pair tangent basis
# rebuilt each tick + the friction cache persisted across ticks. Pins StepWarmFrictionHullWorldN two-run determinism,
# the gripped-on-ramp delta (mu>0 the hull rests, mu=0 it slides off — the falsifiable friction proof), the resting
# contact's tangent impulse within the cone, the de-pen keeping the hull above the ramp (no sink-through), and the
# cache persisting/warm-starting. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(hullfric_step_test)

# Hull Friction + Joints Slice 4 (Slice HF4, HULL JOINTS COMPOSED of FLAGSHIP #30: hf::sim::hulljoint): the NEW
# engine/sim/hulljoint.h composes the body-agnostic joint.h solvers (ball / hinge / angular-limit) with the HF3 hull
# friction contacts in ONE deterministic tick (integrate -> jointIters {SolveBallJoint | SolveAngularLimit} sweeps ->
# the HF3 FULL-inertia friction-contact warm solve -> de-pen -> orient), the GPU shaders/hulljoint_step.comp copies
# VERBATIM. Pins StepJointedHullWorldN two-run determinism, a hung ball-joint chain staying connected (MaxAnchorGap
# small), a hinge holding its angular limit (SwingAngleCos>=cosHalfLimit), joints+friction composing (the cone holds
# while the chain stays connected), and THE NO-JOINTS CONTROL (empty-joint step == hullfric::StepWarmFrictionHullWorld
# byte-for-byte — the joints are purely additive). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(hulljoint_test)

# Hull Friction + Joints Slice 5 (Slice HF5, LOCKSTEP + ROLLBACK of FLAGSHIP #30: hf::sim::hulljoint): the PURE-CPU
# in-process two-peer lockstep + rollback harness APPENDED to engine/sim/hulljoint.h over the HF4
# StepJointedHullWorld (the WH5/JT5/FR5/VD5 mold). NO GPU, NO shader, NO RHI. Pins the (bodies, friction cache)
# snapshot->restore round-trip bit-exact, two-peer lockstep convergence (the WHOLE PAIR byte-equal every tick from
# inputs alone), rollback corrected==authority bit-exact, the mispredict diverging across the chain AND the door AND
# the friction-resting hull then correcting, and THE CACHE-NECESSARY proof (omit the HullFrictionCache -> diverge,
# include it -> converge — the warm-state-necessary lesson at the friction layer). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(hulljoint_net_test)

# Hull Friction + Joints Slice 6 (Slice HF6, the LIT 3D CAPSTONE of FLAGSHIP #30: hf::sim::hulljoint): the
# render-only FLOAT delegate JointedHullToRenderInstances(world) APPENDED to engine/sim/hulljoint.h (a one-line
# delegate to the FROZEN gjk::HullToRenderInstances over the settled jointed hull world — the WH6/CX6/FC6/PS6
# mold). NO GPU, NO shader, NO RHI. Pins the render-bridge contracts the showcase builds on: PROVENANCE (two
# calls byte-equal via gjk::HullRenderMeshEqual), the per-body triangle soup (non-empty, tri count a multiple of
# the body count), THE RENDER IS A PURE READ (the (bodies, friction cache) PAIR JointedHullStatesEqual before vs
# after), and the settled scene's expected body count. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(hulljoint_render_test)

# Deterministic Gameplay / Netcode Slice 1 (Slice VD1, the BEACHHEAD of FLAGSHIP #27: hf::game::verdict): the
# PURE-CPU integer entity world + the unified tick-ordered input-command bus that layers a determinism contract
# over the raw ecs::Registry (engine/game/verdict.h, a NEW additive sibling #including ecs/ecs.h + sim/warmhull.h
# READ-ONLY). Pins SpawnEntity (monotonic EntityIds start 1 + order[] spawn-order + component get/has round-trip),
# DespawnEntity (removes from order[] + frees the ecs handle but the next spawn does NOT reuse the id — the pinned-
# identity contract: spawn->despawn->spawn re-derives the SAME monotonic next id regardless of the ecs free-list
# recycle), ApplyCommands (tick-filtered + fixed array order + dead/out-of-range target deterministic no-op),
# LowerToHullCommands (the kCmdImpulse/kCmdSetAngVel subset == a hand-written convex::ConvexCommand stream,
# byte-equal, targeting the BodyRef.simBodyIndex), and the two-run byte-equality of the whole world over a fixed
# spawn/despawn/command script (the 4 spec proof lines). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(verdict_test)

# Deterministic GPU Cloth Slice 1 (Slice CL1, the BEACHHEAD of FLAGSHIP #8): the Q16.16 fixed-point
# cloth PARTICLE LATTICE INTEGRATOR + grid build core (engine/sim/cloth.h) the GPU
# shaders/cloth_integrate.comp.hlsl copies VERBATIM + proves bit-identical. Reuses the fpx.h Q16.16
# toolbox (fx/fxmul/FxVec3/FxAdd/FxSub/FxScale) read-only. Pins InitGrid (a flat W x H sheet, row-major,
# spacing/origin host-snapped, the two TOP corners PINNED, prev==pos, interior dynamic), IntegrateParticle
# one-step + K-step semi-implicit-Euler closed form (a free particle falls the EXACT Q16.16 distance —
# hand-checked integer recurrence — prev=pos each step) + the ground floor clamp (pos.y==groundY, vel.y
# zeroed on contact, stays settled) + a PINNED particle never moves + two runs byte-identical + an
# ALL-PINNED grid unchanged after K steps (the static no-op). CL1 is ONLY the integrator + grid + pin +
# ground (NO constraints/collision/render — later CL slices). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(cloth_test)

# Deterministic GPU Fluid Slice 1 (Slice FL1, the BEACHHEAD of FLAGSHIP #9): the Q16.16 fixed-point
# fluid PARTICLE POOL INTEGRATOR + dam-break block core (engine/sim/fluid.h) the GPU
# shaders/fluid_integrate.comp.hlsl copies VERBATIM + proves bit-identical. Reuses the fpx.h Q16.16
# toolbox (fx/fxmul/FxVec3/FxAdd/FxSub/FxScale) read-only. Pins InitBlock (a solid W x H x D block in a
# corner, the x-major/y-mid/z-minor traversal order, spacing/origin host-snapped, every particle dynamic,
# prev==pos, vel==0) + ParticleIndex strides, IntegrateFluidParticle one-step + K-step semi-implicit-Euler
# closed form (a free particle falls the EXACT Q16.16 distance — hand-checked integer recurrence, prev=pos
# each step) + the ground floor clamp (pos.y==groundY, vel.y zeroed on contact, stays settled) + a STATIC
# particle never moves + two runs byte-identical + a coherent dam-break fall (every particle fell, some
# pile at the ground) + a ZERO-gravity pool unchanged after K steps (the static no-op). FL1 is ONLY the
# integrator + block + ground (NO neighbours/density/constraints — later FL slices). Pure C++ (hf_core),
# ASan-eligible.
hf_add_pure_test(fluid_test)

# Deterministic GPU Granular/Sand Slice 1 (Slice GR1, the BEACHHEAD of FLAGSHIP #10): the Q16.16 GRAIN POOL
# INTEGRATOR + dropped-block core (engine/sim/grain.h) the GPU shaders/grain_integrate.comp.hlsl copies
# VERBATIM + proves bit-identical. The FL1/CL1 integer-beachhead twin with TWO deltas: a first-class `radius`
# field (48-byte std430 packing) + a RADIUS-AWARE ground rest (the grain's surface rests on the floor). Pins
# the 48-byte packing, InitGrainBlock (W*H*D, uniform radius, all dynamic, prev==pos/vel==0), the one-step +
# K-step semi-implicit-Euler closed form (hand-checked integer recurrence), the radius-aware ground rest
# (pos.y pinned to groundY + radius, vel.y zeroed, stays settled; radius 0 -> the FL1 plain-ground degenerate),
# IntegrateGrains ORDER-INDEPENDENCE (shuffled == in-order per grain, grains independent in GR1), determinism
# (two runs byte-identical + a coherent dropped-block fall), and the STATIC no-op (invMass 0 / flags bit0 pool
# UNCHANGED after K steps). GR1 is ONLY the integrator + block + ground (NO neighbours/contact/friction —
# later GR slices). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(grain_test)

# Deterministic Integer Broadphase Slice 1 (Slice BP1, the BEACHHEAD of FLAGSHIP #23): the BODY GRID + CSR CELL
# TABLE core (engine/sim/broad.h) the GPU shaders/broad_cell_{count,scan,emit}.comp.hlsl copy VERBATIM + prove
# bit-identical. The grain.h GrainGrid/GrainCellTable count->scan->emit, keyed on fpx::FxBody. broad_test pins
# BuildBodyCellTable as a TOTAL partition (each body once, ascending within cell, cellStart monotone, last==N),
# MakeBodyGrid's tight bound (cellMin/gridDim, empty->1x1x1, negative coords via FloorDiv), FlatBodyCellId in
# [0,cellCount), MeasureBodyGrid pure, and two-build determinism. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(broad_test)

# Deterministic Rigid<->Fluid Coupling Slice 1 (Slice CP1, the BEACHHEAD of FLAGSHIP #11): the UNIFIED
# COUPLED WORLD + the BODY->FLUID grid-hash QUERY core (engine/sim/couple.h) the GPU
# shaders/couple_body_{count,scan,emit}.comp.hlsl copy VERBATIM + prove bit-identical. The GR2 grain-neighbor
# twin, per BODY (not per grain): over the FL2 fluid grid + cell table it gathers the fluid particles inside
# each body's sphere, iterating the body's BodyAabb cell RANGE (not a fixed 27-cell stencil), accepting iff
# the per-axis box reject |body.pos.axis - p.pos.axis| < body.radius passes (PURE INT32, the exact radial
# sphere cull deferred to CP2). Pins BodyParticleAccept (the per-axis abs-compare), GatherBodyParticles (one
# body over a hand-laid pool -> the EXACT 3x3 gathered box, CSR offsets correct, ascending within-body order;
# a body clear of the pool -> empty; two bodies -> disjoint ascending per-body lists; the empty world / no
# particles no-op; determinism: two builds byte-identical). PLUS Slice CP2 (the BUOYANCY + DRAG fluid->body
# pass, the CRUX of FLAGSHIP #11): AccumBodyForces (a body over a hand-laid gathered list -> the EXACT Q16.16
# vel delta; buoyancy up EXACTLY proportional to the gathered count; drag damps toward the fluid avg; a static
# / 0-gather body untouched), StepCoupleBuoyancy (a body dropped above a pool settles to a FLOAT LINE above
# the bed, the buoy=0 control sinks to the bed, deterministic + body-order-independent). CP2 claims a
# deterministic fluid->body buoyancy+drag (NO fluid reaction — CP3; NO coupled step — CP4; NO lockstep — CP5;
# NO render — CP6). Pure C++ (hf_core), ASan-eligible. couple.h #includes fpx.h + fluid.h read-only.
hf_add_pure_test(couple_test)

# Deterministic Rigid<->Grain Coupling Slice 1 (Slice CG1, the BEACHHEAD of FLAGSHIP #12): the UNIFIED
# bodies+grains WORLD + the BODY->GRAIN grid-hash QUERY core (engine/sim/couple_grain.h) the GPU
# shaders/cgrain_body_{count,scan,emit}.comp.hlsl copy VERBATIM + prove bit-identical. The CP1 twin (couple.h's
# body->fluid query) with the GRAIN grid instead of the fluid grid: over the GR2 grain grid + cell table it
# gathers the grain particles inside each body's sphere, iterating the body's BodyAabb cell RANGE (not a fixed
# 27-cell stencil), accepting iff the per-axis box reject |body.pos.axis - g.pos.axis| < body.radius passes
# (PURE INT32, the exact radial sphere cull deferred to CG2/CG3). Pins BodyGrainAccept (the per-axis
# abs-compare), GatherBodyGrains (one body over a hand-laid bed -> the EXACT 3x3 gathered box, CSR offsets
# correct, ascending within-body order; a body clear of the bed -> empty; two bodies -> disjoint ascending
# per-body lists; the empty world / no grains no-op; hSearch-independence of the gathered set; determinism: two
# builds byte-identical) + CountGathered/MaxPerBody. CG1 claims ONLY the unified world + the per-body grain
# query (NO momentum exchange — CG2 support/drag, CG3 displacement, CG4 step, CG5 lockstep, CG6 render). Pure
# C++ (hf_core), ASan-eligible. couple_grain.h #includes fpx.h + grain.h read-only.
hf_add_pure_test(cgrain_test)

# Deterministic Grain<->Fluid Coupling Slice 1 (Slice GF1, the BEACHHEAD of FLAGSHIP #13): the UNIFIED TWO-POOL
# WORLD + the SHARED-GRID CROSS QUERY core (engine/sim/couple_gf.h) the GPU
# shaders/cgf_gf_{count,scan,emit}.comp.hlsl + cgf_fg_{count,scan,emit}.comp.hlsl copy VERBATIM + prove
# bit-identical. The GR2/FL2 27-cell-stencil neighbour search applied CROSS-POOL over ONE shared grid: per grain
# the FLUID particles in its 27-cell stencil (gfNeighbors), per fluid the GRAINS in its 27-cell stencil
# (fgNeighbors), each accepted iff the per-axis box reject |query.axis - target.axis| < h passes (PURE INT32, the
# exact radial cull deferred to GF2/GF3). Pins MakeCGFGrid (the union bounds over BOTH pools' cell ranges — every
# grain AND fluid cell in [0,gridDim); empty/one-empty degenerate cases), BuildCGFNeighbors (a hand-laid
# overlapping set -> the EXACT gf/fg cross lists vs a naive O(G*F) oracle, CSR offsets correct, ascending
# within-query order, NO self-pairs), the SYMMETRY CountGF==CountFG, separated pools -> 0 (the no-op), determinism
# (two builds byte-identical). GF1 claims ONLY the unified two-pool world + the two cross-pool neighbour lists (NO
# momentum exchange — GF2 buoyancy/seepage, GF3 contact, GF4 step, GF5 lockstep, GF6 render). Pure C++ (hf_core),
# ASan-eligible. couple_gf.h #includes fpx.h + grain.h + fluid.h read-only.
hf_add_pure_test(cgf_test)

# Deterministic GPU Navmesh Slice 1 (Slice NAV1, the BEACHHEAD of FLAGSHIP #7): the integer HEIGHTFIELD
# SPAN RASTERIZATION core (engine/nav/navmesh.h) the GPU shaders/nav_raster_count/scan/emit.comp.hlsl
# copy VERBATIM + prove bit-identical. Pins QuantizeCoord (world->voxel column via FloorDiv incl NEGATIVE
# coords — correct floor, not truncate-toward-zero), PointInTriXZ (the three-integer-edge cover test,
# both windings), TriColumnAabb (clamped inclusive column range incl off-grid -> empty),
# RasterizeTriangleSpans (a single tri -> the expected raw span count/offset/y-range; a ground quad fills
# every column; the exclusive prefix-sum consistency), MergeColumnSpans (overlapping->1, disjoint->2
# sorted, touching->1, empty->empty), empty input -> zero spans (the no-op the GPU disabled-path
# mirrors), and determinism (two rasterizations byte-identical). Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(nav_test)

# Virtual Shadow Maps Slice 1 (Slice VA): the virtual clipmap PAGE TABLE + PAGE-NEEDED marking math
# (render/vsm.h) the GPU shaders/vsm_mark.comp.hlsl copies verbatim. Pins PageId/UnpackPageId bijection
# over [0,pageCount()), SelectClipmapLevel's INTEGER threshold-ladder edges (at-camera->level0, far->top,
# the 2^L boundaries — the GPU==CPU DETERMINISM CRUX, no transcendental), MarkPage single-point->expected
# (level,px,py), MarkResidentPages known set (count+membership, duplicate-safe), markingEnabled-off->empty
# set, and determinism. Pure C++ (hf_core), ASan-eligible.
hf_add_pure_test(vsm_test)

# Runtime Virtual Texturing Slice 1 (Slice VT1): the VT mip page-table + page-needed FEEDBACK/MARKING
# math (render/vt.h) the GPU shaders/vt_feedback.comp.hlsl copies verbatim. Pins pagesPerSide/
# mipPageOffset (the PREFIX-SUM, NOT mip*vpps² — the VT-vs-VSM crux), PageId/UnpackPageId bijection over
# [0,pageCount()), SelectMipLevel's INTEGER threshold-ladder edges (the 2^m boundaries — the GPU==CPU
# DETERMINISM CRUX, no transcendental), VtPageId corners (0/just<1/mid/clamp), MarkFeedbackPages known
# set (count+membership, duplicate-safe), feedbackEnabled-off->empty set, and determinism. Pure C++
# (hf_core), ASan-eligible.
hf_add_pure_test(vt_test)

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
