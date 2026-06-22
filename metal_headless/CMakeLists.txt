# Standalone, conan-free, SDL-free headless Metal target for bringing up the Metal RHI
# backend on a real Mac over SSH (no window server). Drives the REAL Metal RHI classes
# (MetalDevice/MetalSwapchain/MetalCommandBuffer/MetalPipeline/MetalTexture/MetalBuffer) in
# headless/offscreen mode, renders the full Slice-F scene (ground plane + 3x3 lit cube grid),
# reads the offscreen MTLTexture back, and writes a PNG.
#
#   cmake -S metal_headless -B build-metal -G Ninja
#   cmake --build build-metal
#   ./build-metal/visual_test out.png
cmake_minimum_required(VERSION 3.20)
project(hf_metal_headless LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Repo root is one level up from this dir.
get_filename_component(HF_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# ---------------------------------------------------------------------------
# Unified shader toolchain: GENERATE the Metal MSL from the SHARED HLSL sources
# (the same HLSL the Vulkan/DXC backend compiles) so there is no hand-written
# MSL to drift. Pipeline per shader: HLSL --glslc--> SPIR-V --spirv-cross--> MSL.
#
#   * glslc compiles HLSL with -DHF_MSL_GEN so the vertex shaders declare the
#     model matrix as an explicit cbuffer (glslang ignores [[vk::push_constant]]);
#     the bindings are chosen so that --msl-decoration-binding yields the engine's
#     flat Metal indices (see engine/rhi_metal/metal_common.h).
#   * spirv-cross --msl --msl-decoration-binding maps SPIR-V binding -> MSL index,
#     and --rename-entry-point gives stable entry names the loader expects.
# ---------------------------------------------------------------------------
find_program(GLSLC_EXECUTABLE NAMES glslc REQUIRED)
find_program(SPIRV_CROSS_EXECUTABLE NAMES spirv-cross REQUIRED)

set(HF_SHADER_SRC "${HF_ROOT}/shaders")
set(HF_GEN_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/gen_shaders")
file(MAKE_DIRECTORY "${HF_GEN_SHADER_DIR}")

set(HF_GENERATED_MSL "")
# hf_gen_msl(<basename> <hlsl-file> <vert|frag> <new-entry-point> [extra spirv-cross args...])
#
# Any trailing args (ARGN, after the 4 required ones) are appended VERBATIM to the spirv-cross
# invocation for THIS shader only. Used to opt a single shader into a higher --msl-version (e.g.
# visbuffer.frag needs --msl-version 20200 for SV_PrimitiveID -> [[primitive_id]]) WITHOUT touching
# any other shader's command — every other call omits the extra args, so its spirv-cross command line
# (and thus its generated MSL) is byte-for-byte unchanged and its golden is unaffected.
function(hf_gen_msl base src stage entry)
  set(extra_sc_args ${ARGN})  # optional per-shader extra spirv-cross args (empty for all but visbuffer.frag)
  set(spv "${HF_GEN_SHADER_DIR}/${base}.spv")
  set(msl "${HF_GEN_SHADER_DIR}/${base}.gen.metal")
  if(stage STREQUAL "vert")
    set(glslc_stage "vert")
    set(sc_stage "vert")
  elseif(stage STREQUAL "comp")
    set(glslc_stage "compute")  # glslc compute stage flag
    set(sc_stage "comp")        # spirv-cross --rename-entry-point execution model
  else()
    set(glslc_stage "frag")
    set(sc_stage "frag")
  endif()
  add_custom_command(
    OUTPUT "${msl}"
    COMMAND "${GLSLC_EXECUTABLE}" -DHF_MSL_GEN -fshader-stage=${glslc_stage}
            -x hlsl -fentry-point=main "${HF_SHADER_SRC}/${src}" -o "${spv}"
    COMMAND "${SPIRV_CROSS_EXECUTABLE}" --msl --msl-decoration-binding
            --rename-entry-point main ${entry} ${sc_stage} ${extra_sc_args}
            "${spv}" --output "${msl}"
    DEPENDS "${HF_SHADER_SRC}/${src}"
    COMMENT "Gen MSL ${src} -> ${base}.gen.metal"
    VERBATIM)
  set(HF_GENERATED_MSL ${HF_GENERATED_MSL} "${msl}" PARENT_SCOPE)
endfunction()

hf_gen_msl(lit.vert    lit.vert.hlsl    vert vertex_main)
hf_gen_msl(lit.frag    lit.frag.hlsl    frag fragment_main)
# Cascaded shadow maps (Slice AD): CSM lit fragment (atlas cascade selection + PCF) + the depth-only
# CSM shadow caster (cascade VP in the push constant). Reuse lit.vert. Same HLSL->SPIR-V->MSL chain.
hf_gen_msl(lit_csm.frag    lit_csm.frag.hlsl    frag csm_fragment)
hf_gen_msl(shadow_csm.vert shadow_csm.vert.hlsl vert csm_shadow_vertex)
# Spot light + perspective spot-light shadow (Slice AE): cone-cut lit fragment. Reuses lit.vert +
# the CSM depth-only shadow caster (shadow_csm.vert: spotViewProj in the push constant).
hf_gen_msl(lit_spot.frag   lit_spot.frag.hlsl   frag spot_fragment)
# Omnidirectional point-light shadow (Slice AF): 6-face cube atlas lit fragment. Reuses lit.vert +
# the CSM depth-only shadow caster (shadow_csm.vert: per-face view-proj in the push constant).
hf_gen_msl(lit_point.frag  lit_point.frag.hlsl  frag point_fragment)
# Clustered / Forward+ lit fragment (Slice AG): per-fragment cluster index + iterate that cluster's
# point lights from three fragment-stage storage buffers (space3 -> flat Metal fragment buffers).
# Reuses lit.vert. Same HLSL->SPIR-V->MSL toolchain.
hf_gen_msl(lit_clustered.frag  lit_clustered.frag.hlsl  frag clustered_fragment)
# Clustered LIGHT-CULLING lit fragment (Slice CL): windowed hard-radius point lights from this pixel's
# cluster's fixed-slot ordered list (the SAME ClusterList layout the cluster_assign compute fills on
# Vulkan; on Metal the list is assigned CPU-side via render/cluster.h). Reuses lit.vert. Shared
# HLSL->SPIR-V->MSL toolchain -> the light field is bit-identical cross-backend.
hf_gen_msl(lit_clustered_cl.frag  lit_clustered_cl.frag.hlsl  frag clustered_lights_fragment)
# Full-PBR fragment shader (Slice P): reuses lit.vert; samples the 5-texture glTF material set.
hf_gen_msl(lit_pbr.frag lit_pbr.frag.hlsl frag pbr_fragment)
# Data-driven material graph (Slice AV): the COMMITTED build-time-generated fragment shader
# (shaders/generated/mat_showcase.frag.hlsl, codegen'd from assets/materials/showcase.mat.json on the
# Windows build). It #includes ../pbr_core.hlsli and reuses lit.vert + the existing PBR material set,
# so it flows through the SAME HLSL->SPIR-V->MSL toolchain as every other shader (no hand-written MSL).
hf_gen_msl(mat_showcase.frag generated/mat_showcase.frag.hlsl frag material_fragment)
# Live runtime material authoring (Slice AW): the SECOND material's committed build-time-generated
# fragment (shaders/generated/mat_showcase2.frag.hlsl, codegen'd from assets/materials/showcase2.mat.json).
# Metal stays BUILD-TIME (no runtime HLSL compiler in the standalone), so showcase2 flows through the
# SAME HLSL->SPIR-V->MSL toolchain here; the new mat_graph2.png golden is rendered from this MSL.
hf_gen_msl(mat_showcase2.frag generated/mat_showcase2.frag.hlsl frag material2_fragment)
# Slice AZ: the THIRD material's committed build-time-generated fragment (shaders/generated/
# mat_showcase3.frag.hlsl) — exercises the new node types (Swizzle/MakeFloat3/Power/Saturate/OneMinus).
# Same HLSL->SPIR-V->MSL toolchain; drawn alongside showcase + showcase2 in the --material-multi scene
# that renders the new mat_multi.png golden.
hf_gen_msl(mat_showcase3.frag generated/mat_showcase3.frag.hlsl frag material3_fragment)
# Slice BE: the NormalMap-node material's committed build-time-generated fragment (shaders/generated/
# mat_normalmap.frag.hlsl) — PBROutput.normal = NormalMap(slot="normalmap"), so the codegen builds the
# world normal from the tangent-space sample via the SAME Gram-Schmidt TBN lit.frag uses and shades via
# hfShadePBRN. Same HLSL->SPIR-V->MSL toolchain; renders the new mat_normal.png golden (--material-normal).
hf_gen_msl(mat_normalmap.frag generated/mat_normalmap.frag.hlsl frag material_normal_fragment)
# HDR environment IBL (Slice R): equirect skybox + full-PBR-with-HDR-IBL fragment shaders. Same
# HLSL->SPIR-V->MSL toolchain; the env map binds at flat fragment texture(11)/sampler(12).
hf_gen_msl(sky_hdr.frag      sky_hdr.frag.hlsl      frag sky_hdr_fragment)
hf_gen_msl(lit_pbr_ibl.frag  lit_pbr_ibl.frag.hlsl  frag pbr_ibl_fragment)
# Skinned-animation variants (Slice O): GPU skinning vertex shader (reuses lit.frag) + skinned
# depth-only shadow vertex shader. Same HLSL->SPIR-V->MSL toolchain as the others.
hf_gen_msl(lit_skinned.vert    lit_skinned.vert.hlsl    vert skinned_vertex)
hf_gen_msl(shadow_skinned.vert shadow_skinned.vert.hlsl vert skinned_shadow_vertex)
# Instanced variants (Slice Q): per-instance model-matrix vertex stream (binding 1). The lit
# variant reuses lit.frag; the shadow variant is depth-only. Same HLSL->SPIR-V->MSL toolchain.
hf_gen_msl(lit_instanced.vert    lit_instanced.vert.hlsl    vert instanced_vertex)
hf_gen_msl(shadow_instanced.vert shadow_instanced.vert.hlsl vert instanced_shadow_vertex)
# Transparent ("glass") pass (Slice T): a dedicated vertex shader (model + per-object tint/alpha push
# constant) and a self-contained glassy fragment shader (no material textures). Same HLSL->SPIR-V->MSL
# toolchain as the others.
hf_gen_msl(transparent.vert transparent.vert.hlsl vert transparent_vertex)
hf_gen_msl(transparent.frag transparent.frag.hlsl frag transparent_fragment)
hf_gen_msl(post.vert   post.vert.hlsl   vert post_vertex)
hf_gen_msl(post.frag   post.frag.hlsl   frag post_fragment)
# HDR bloom chain (Slice U): bright-pass, downsample, tent upsample, composite/tonemap. All
# fullscreen fragment shaders reusing post.vert; each reads its per-pass params from a fragment
# push constant (a HF_MSL_GEN cbuffer at the bloom push-const slot here).
hf_gen_msl(bloom_prefilter.frag  bloom_prefilter.frag.hlsl  frag bloom_prefilter_fragment)
hf_gen_msl(bloom_downsample.frag bloom_downsample.frag.hlsl frag bloom_downsample_fragment)
hf_gen_msl(bloom_upsample.frag   bloom_upsample.frag.hlsl   frag bloom_upsample_fragment)
hf_gen_msl(bloom_composite.frag  bloom_composite.frag.hlsl  frag bloom_composite_fragment)
hf_gen_msl(shadow.vert shadow.vert.hlsl vert shadow_vertex)
hf_gen_msl(sky.vert    sky.vert.hlsl    vert sky_vertex)
hf_gen_msl(sky.frag    sky.frag.hlsl    frag sky_fragment)
# GPU-driven culling + indirect draw (Slice AR): compute kernel frustum-culls a 1024-instance grid
# and ORDER-compacts the survivors + writes the indirect draw-args. Reuses lit_instanced.vert for the
# indirect draw. Same HLSL->SPIR-V->MSL toolchain (one workgroup, groupshared prefix sum).
hf_gen_msl(cull.comp cull.comp.hlsl comp cull_main)
# GPU particle system: compute kernel + point-sprite vertex/fragment.
hf_gen_msl(particles.comp particles.comp.hlsl comp particles_main)
hf_gen_msl(particle.vert   particle.vert.hlsl  vert particle_vertex)
hf_gen_msl(particle.frag   particle.frag.hlsl  frag particle_fragment)
# Immediate-mode debug-draw line pipeline (Slice W): world-space line verts projected by
# FrameData.viewProj, flat per-vertex color. LINE_LIST topology. Same HLSL->SPIR-V->MSL toolchain.
hf_gen_msl(debug_line.vert debug_line.vert.hlsl vert debug_line_vertex)
hf_gen_msl(debug_line.frag debug_line.frag.hlsl frag debug_line_fragment)
# Virtual-geometry meshlet viz (Slice DS): per-cluster flat-color pass (model matrix + fixed-direction
# Lambert in the vertex, pushed hashColor(cluster) forwarded to the fragment). Reuses the existing draw
# + push-constant surface (no new RHI). Same HLSL->SPIR-V->MSL toolchain; behind --meshlet-viz ONLY.
hf_gen_msl(meshlet_viz.vert meshlet_viz.vert.hlsl vert meshlet_viz_vertex)
hf_gen_msl(meshlet_viz.frag meshlet_viz.frag.hlsl frag meshlet_viz_fragment)
# Virtual-geometry VISIBILITY BUFFER (Slice DW): the BOUND visbuffer vertex (clusterID pushed per
# survivor, since glslc/spirv-cross can't lower the Vulkan MDI DrawIndex builtin — same as cluster_viz)
# + the shared visbuffer fragment (out single uint = (clusterID<<7)|(SV_PrimitiveID&0x7F) into the
# R32_Uint visibility buffer). spirv-cross lowers SV_PrimitiveID to [[primitive_id]] in the MSL fragment,
# but ONLY when targeting MSL 2.2+ (else it throws "PrimitiveId on macOS requires MSL 2.2"). So this ONE
# shader gets an extra --msl-version 20200 (passed as the optional trailing hf_gen_msl arg); EVERY other
# shader's spirv-cross command is left exactly as-is (no version flag), so all other generated MSL — and
# thus all existing goldens — are byte-for-byte unchanged. Same HLSL->SPIR-V->MSL toolchain; the only new
# RHI is the additive Format::R32_Uint. Behind --visbuffer only.
hf_gen_msl(visbuffer_bound.vert visbuffer_bound.vert.hlsl vert visbuffer_vertex)
hf_gen_msl(visbuffer.frag       visbuffer.frag.hlsl       frag visbuffer_fragment --msl-version 20200)
# Virtual-geometry DEFERRED MATERIAL RESOLVE (Slice DX): the FULLSCREEN resolve fragment that texel-fetches
# the DW R32_Uint vis-buffer (Texture2D<uint>.Load -> MSL texture.read), unpacks (clusterID,triID), looks
# up the covering triangle via the cluster-meta/index/vertex SSBOs (bound through the existing
# BindLightClusters set-3 path -> Metal fragment buffers 13/14/15), and FLAT-shades it. Like visbuffer.frag,
# the INTEGER texture.read needs MSL 2.2, so this ONE shader gets the ISOLATED --msl-version 20200 (every
# other shader's spirv-cross command is left exactly as-is, so all 86 existing goldens are byte-for-byte
# unchanged). Reuses the post.vert fullscreen triangle (already generated). NO new RHI. Behind --visresolve.
hf_gen_msl(visresolve.frag      visresolve.frag.hlsl      frag visresolve_fragment --msl-version 20200)
# Virtual Shadow Maps Slice 1 (Slice VA): vsm_mark.comp marks which virtual clipmap PAGES the visible
# receivers need — one thread per receiver world-sample runs SelectClipmapLevel (the INTEGER threshold-
# ladder, NO transcendental log2) + MarkPage (subtract/divide/floor) copied VERBATIM from render/vsm.h,
# writing resident[pageId]=1 (an order-independent integer SET — a plain write, no atomics). A pure
# integer SSBO compute (like froxel_inject.comp) -> the default MSL gen suffices (no integer texture.read
# / atomics, so NO --msl-version 20200 needed). The resident set is bit-exact GPU==CPU + cross-backend by
# construction. Behind --vsm-mark only; existing shaders + goldens untouched.
hf_gen_msl(vsm_mark.comp         vsm_mark.comp.hlsl         comp vsm_mark_main)
# Runtime Virtual Texturing Slice 1 (Slice VT1): vt_feedback.comp marks which virtual TEXTURE PAGES a
# set of sample-requests need — one thread per request reads its HOST-SNAPPED integer (mip,px,py) page
# coords (the host ran the UV->page quantization; the GPU does ZERO float) + computes pageId via PageId
# (mipPageOffset + py*pagesPerSide + px) copied VERBATIM from render/vt.h, writing feedback[pageId]=1
# (an order-independent integer SET — a plain write, no atomics). A pure integer SSBO compute (like
# vsm_mark.comp / froxel_inject.comp) -> the default MSL gen suffices (no integer texture.read /
# atomics, so NO --msl-version 20200 needed). The feedback set is bit-exact GPU==CPU + cross-backend by
# construction. Behind --vt-feedback only; existing shaders + goldens untouched.
hf_gen_msl(vt_feedback.comp      vt_feedback.comp.hlsl      comp vt_feedback_main)
# Runtime Virtual Texturing Slice 2 (Slice VT2): vt_alloc.comp is the SINGLE-THREAD physical tile-pool
# allocator ([numthreads(1,1,1)]) — one thread walks pageId 0..pageCount-1 ASCENDING maintaining nextTile,
# writing gIndirection[pageId] = (resident && nextTile<cap) ? nextTile++ : 0xFFFFFFFF (kNoTile), the DIRECT
# mirror of render/vt.h::AllocatePhysicalTiles. Pure integer SSBO compute (no atomics, no integer
# texture.read) -> the default MSL gen suffices (NO --msl-version 20200). The indirection table is bit-exact
# GPU==CPU + cross-backend by construction. Behind --vt-alloc only; existing shaders + goldens untouched.
hf_gen_msl(vt_alloc.comp         vt_alloc.comp.hlsl         comp vt_alloc_main)
# Runtime Virtual Texturing Slice 3 (Slice VT3): vt_pagegen.comp GENERATES each allocated virtual page's
# procedural CONTENT into its physical tile in the atlas — ONE thread per atlas texel (race-free parallel MAP,
# NO atomics). Each thread reads the host-built reverse table gTilePageId, UnpackPageId's the owning pageId,
# and writes PageTexel(pageId,mip,localXY) (the PURE-INTEGER per-texel generator copied VERBATIM from
# render/vt.h) into gAtlas; unallocated -> kAtlasClear; genEnabled=0 -> all kAtlasClear. Pure integer SSBO
# compute (no atomics, no integer texture.read) -> the default MSL gen suffices (NO --msl-version 20200). The
# atlas is bit-exact GPU==CPU + cross-backend by construction. Behind --vt-pagegen only; goldens untouched.
hf_gen_msl(vt_pagegen.comp       vt_pagegen.comp.hlsl       comp vt_pagegen_main)
# Runtime Virtual Texturing Slice 4 (Slice VT4): vt_sample.comp SAMPLES the virtual texture through the
# virtual->physical INDIRECTION — ONE thread per VIRTUAL texel (race-free MAP, NO atomics). Each thread computes
# its pageId, reads gIndirection[pageId] -> the physical tile, offsets into the physical atlas and reads the
# texel (NEAREST integer read; non-resident -> kVtMiss); SampleVirtualTexel copied VERBATIM from render/vt.h.
# sampleEnabled=0 -> all kVtMiss. Pure integer SSBO compute (no atomics, no integer texture.read) -> the default
# MSL gen suffices (NO --msl-version 20200). The reconstructed image is bit-exact GPU==CPU + cross-backend by
# construction. Behind --vt-sample only; goldens untouched.
hf_gen_msl(vt_sample.comp        vt_sample.comp.hlsl        comp vt_sample_main)
# Runtime Virtual Texturing Slice 5 (Slice VT5): vt_cachegen.comp is the CACHE-AWARE sibling of VT3's
# vt_pagegen.comp (UNCHANGED) — ONE thread per atlas texel (race-free MAP, NO atomics). Each thread finds its
# tile; if unallocated OR gNeedsGen[tileIndex]==0 (a cache hit) it RETURNS without writing (the atlas SSBO
# PERSISTS, so cached texels carry over byte-identical); else it writes PageTexelV(pageId,mip,localXY,
# gTileVersion[tile]) (the content-version-aware generator copied VERBATIM from render/vt.h; version 0 == VT3
# PageTexel). Pure integer SSBO compute (no atomics, no integer texture.read) -> the default MSL gen suffices
# (NO --msl-version 20200). The cached atlas is bit-exact GPU==CPU + cross-backend AND byte-identical to a
# full regen by construction. Behind --vt-cache only; goldens untouched.
hf_gen_msl(vt_cachegen.comp      vt_cachegen.comp.hlsl      comp vt_cachegen_main)
# GPU Isosurface Meshing Slice 1 (Slice MC1): mc_classify.comp classifies each voxel CELL into its 8-bit
# MARCHING-CUBES case index — ONE thread per cell gathers its 8 cube-CORNER scalars from gField at the
# canonical kCornerOffset and writes gCases[cell] = CaseIndex(corners,isovalue) (idx = Σ (corner[i] >
# isovalue) << i, copied VERBATIM from render/mc.h); an order-independent per-cell integer write (no
# atomics). classifyEnabled=0 -> writes 0. A pure integer SSBO compute (like vsm_mark.comp /
# vt_feedback.comp — no atomics, no integer texture.read) -> the default MSL gen suffices (NO
# --msl-version 20200 needed). The case set is bit-exact GPU==CPU + cross-backend by construction.
# Behind --mc-classify only; existing shaders + goldens untouched.
hf_gen_msl(mc_classify.comp      mc_classify.comp.hlsl      comp mc_classify_main)
# Deterministic Rigid-Body Fracture Slice 1 (Slice FR1, the BEACHHEAD of FLAGSHIP #14): fract_classify.comp
# partitions a source-volume integer LATTICE into N fragment CELLS — ONE thread per lattice SAMPLE
# decomposes its flat id -> (x,y,z), runs the nearest-seed loop (bestId = argmin_k dx*dx+dy*dy+dz*dz over
# the M host seeds, STRICTLY-less lowest-index tie-break) copied VERBATIM from engine/sim/fract.h, writing
# gCells[sample]=bestId (an order-independent per-sample integer write, no atomics). classifyEnabled=0 ->
# writes 0. PURE INT32 (small lattice coords -> max d2 ~3.1M fits int32, NO int64) -> the default MSL gen
# suffices (NO --msl-version 20200 needed, the mc_classify lesson), a TRUE GPU pass on both backends. The
# cell set is bit-exact GPU==CPU + cross-backend by construction. Behind --fract-cells only; existing
# shaders + goldens untouched.
hf_gen_msl(fract_classify.comp   fract_classify.comp.hlsl   comp fract_classify_main)
# Deterministic Rigid-Body Fracture Slice 2 (Slice FR2): FRAGMENT EXTRACTION — turns each non-empty FR1
# Voronoi cell into a FRAGMENT RECORD (integer centroid, AABB, boundRadiusSq/boundRadius, volume, cellId,
# invMass) via the GR2/FL2 count->scan->emit CSR over FR1's cellId[] (cells == seedCount) + a per-fragment
# reduction. fract_emit_count (ONE thread per lattice SAMPLE: InterlockedAdd into its cell's counter — the
# order-independent sum), fract_emit_scan ([numthreads(1,1,1)] exclusive prefix-sum + the CL2/MC3
# stream-compaction -> fragStart + cellToFragment + fragmentToCell + fragCount), fract_emit_emit
# (SINGLE-THREAD ascending-sample scatter into fragSamples — single-thread for the within-cell ascending
# order, the GR2/FL2 emit crux), fract_reduce (ONE thread per FRAGMENT: reduce its CSR slice in ascending
# sample order -> the FractFragment record, per-fragment disjoint, NO atomics). ALL PURE INT32 (centroid
# sums of small lattice coords, per-axis min/max, squared dists < ~3.1M fit int32, a truncating int divide,
# an int32 binary ISqrt32, a pure-integer kOne/volume invMass — NO int64, NO float, NO Q16.16 world
# scaling) -> the default MSL gen suffices (NO --msl-version 20200, the fract_classify/mc_classify lesson),
# a TRUE GPU pass on both backends. The fragment array + CSR + remap are bit-exact GPU==CPU + cross-backend
# by construction. Behind --fract-fragments only; FR1 fract_classify + every existing shader + goldens
# untouched.
hf_gen_msl(fract_emit_count.comp fract_emit_count.comp.hlsl comp fract_emit_count_main)
hf_gen_msl(fract_emit_scan.comp  fract_emit_scan.comp.hlsl  comp fract_emit_scan_main)
hf_gen_msl(fract_emit_emit.comp  fract_emit_emit.comp.hlsl  comp fract_emit_emit_main)
hf_gen_msl(fract_reduce.comp     fract_reduce.comp.hlsl     comp fract_reduce_main)
# Deterministic Fixed-Point Physics Slice 1 (Slice FPX1, the BEACHHEAD of FLAGSHIP #6): fpx_integrate.comp
# runs the Q16.16 semi-implicit-Euler INTEGRATOR — ONE thread per body runs `steps` iterations of
# (vel += gravity*dt; pos += vel*dt; ground floor-clamp) on its OWN body, the fxmul ((int64)a*b >> 16) +
# integrate + clamp copied VERBATIM from engine/sim/fpx.h::IntegrateBody, writing gBodies[i] back (each
# body independent -> race-free, NO atomics).
# It is VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd here: its int64 fxmul (int64_t in engine/sim/fpx.h,
# GENUINELY required because gravity*dt ≈ -7e11 overflows int32) is compiled fine by DXC on the Vulkan
# path, but glslc (glslang's HLSL frontend used for the Metal HLSL->SPIR-V->MSL chain) cannot parse
# int64_t in HLSL (`error: 'int64_t' : unknown variable`), so it can't lower this shader. Same Vulkan-only
# convention as swraster.comp / cluster_cull.comp (which glslc likewise can't lower): on Metal the --fpx
# showcase runs the CPU fpx::IntegrateStep (the SAME bit-exact reference the Vulkan GPU==CPU proof compares
# against) -> the Metal result is byte-identical to the Vulkan GPU result BY CONSTRUCTION, while the Vulkan
# side carries the GPU==CPU bit-identity proof. So fpx_integrate.comp gets NO hf_gen_msl line.
# Hardware Ray Tracing Slice 1 (Slice RT1, the BEACHHEAD of FLAGSHIP #28: HARDWARE RAY TRACING):
# rt_trace.comp is the GPU TWIN of the deterministic Q16.16 SW reference ray tracer — ONE thread per pixel
# generates the integer primary ray, brute-forces every primitive (NO accel-struct cull), keeps the
# closest by the total order (t, primIndex), and writes the integer Lambert RGBA8, the
# TraceClosest/ShadeHitInt/PrimaryRay math copied VERBATIM from engine/render/rtrace.h.
# It is VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd here: its int64 fxmul/fxdiv ((int64)a*b >> 16 and
# ((int64)a << 16)/b — GENUINELY required so a Q16.16 product/quotient does not overflow int32) plus the
# int64 FxISqrt digit loop are compiled fine by DXC on the Vulkan path, but glslc (glslang's HLSL frontend
# used for the Metal HLSL->SPIR-V->MSL chain) cannot parse int64_t in HLSL, so it can't lower this shader.
# Same Vulkan-only convention as fpx_integrate.comp / swraster.comp / cluster_cull.comp (which glslc
# likewise can't lower): on Metal the --rt1-trace showcase runs the CPU rtrace::RenderScene (the SAME
# bit-exact reference the Vulkan GPU==CPU proof compares against) -> the Metal result is byte-identical to
# the Vulkan GPU result BY CONSTRUCTION, while the Vulkan side carries the GPU==CPU bit-identity proof. So
# rt_trace.comp gets NO hf_gen_msl line.
# Hardware Ray Tracing Slice 2 (Slice RT2, FLAGSHIP #28): rt_query.comp is the DETERMINISTIC HW INLINE
# RAY QUERY kernel (HLSL inline RayQuery over the TLAS draining every candidate procedural primitive, then
# the FROZEN fx Q16.16 intersection + (t,primIndex) min). It is VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd
# here: (a) HLSL RayQuery lowers (via DXC) to SPIR-V RayQueryKHR (the SPV_KHR_ray_query extension), which
# glslc/spirv-cross — the Metal HLSL->SPIR-V->MSL chain — cannot lower; (b) it also uses the int64 fxmul/
# fxdiv/FxISqrt math glslc can't parse anyway. Same Vulkan-only convention as rt_trace.comp /
# fpx_integrate.comp: on Metal the --rt2-query showcase runs the CPU rtrace::RenderScene (the SAME bit-exact
# reference the Vulkan HW==SW==CPU memcmp compares against) -> the Metal result is byte-identical to the
# Vulkan HW result BY CONSTRUCTION, while the Vulkan side carries the HW-ray-query == SW == CPU proof. So
# rt_query.comp gets NO hf_gen_msl line. (Metal HARDWARE ray query — rt_query.metal over an
# MTLAccelerationStructure — is the deliberate follow-on slice RT2b.)
# Hardware Ray Tracing Slice 3 (Slice RT3, FLAGSHIP #28): rt_shadow.comp is the DETERMINISTIC RT HARD
# SHADOWS kernel — the rt_query.comp primary RayQuery body VERBATIM, THEN a SECOND RayQuery (the shadow ray)
# whose any-hit occlusion (an order-independent boolean OR over OUR fx intersections) GATES the diffuse
# shade. Like rt_query.comp it is VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd here: HLSL inline RayQuery lowers
# (via DXC) to SPIR-V RayQueryKHR (the SPV_KHR_ray_query extension), which glslc/spirv-cross — the Metal
# HLSL->SPIR-V->MSL chain — cannot lower; and it also uses the int64 fxmul/fxdiv/FxISqrt math glslc can't
# parse anyway. Same Vulkan-only convention: on Metal the --rt3-shadow showcase runs the CPU
# rtrace::RenderSceneShadowed (the SAME bit-exact reference the Vulkan HW==CPU memcmp compares against) ->
# the Metal result is byte-identical to the Vulkan HW result BY CONSTRUCTION, while the Vulkan side carries
# the HW-shadow-ray == CPU proof. So rt_shadow.comp gets NO hf_gen_msl line.
# Hardware Ray Tracing Slice 4 (Slice RT4, FLAGSHIP #28): rt_reflect.comp is the DETERMINISTIC RT MIRROR
# REFLECTIONS kernel — the rt_shadow.comp primary closest-hit + primary shadow body VERBATIM, THEN if the
# primary hit is reflective a REFLECTION RayQuery (a second closest-hit drain, the view dir mirrored about
# the fx normal) + the reflected shadow + the integer UNPACKED-channel blend (final = surface*(1-refl) +
# reflected*refl). Like rt_shadow.comp it is VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd here: HLSL inline
# RayQuery lowers (via DXC) to SPIR-V RayQueryKHR (the SPV_KHR_ray_query extension), which glslc/spirv-cross
# cannot lower; and it uses the int64 fxmul/fxdiv/FxISqrt math glslc can't parse anyway. On Metal the
# --rt4-reflect showcase runs the CPU rtrace::RenderSceneReflected (the SAME bit-exact reference the Vulkan
# HW==CPU memcmp compares against) -> the Metal result is byte-identical to the Vulkan HW result BY
# CONSTRUCTION, while the Vulkan side carries the HW-reflection == CPU proof. So rt_reflect.comp gets NO
# hf_gen_msl line.
# Hardware Ray Tracing Slice 6 (Slice RT6, FLAGSHIP #28, the MONEY-SHOT CAPSTONE): rt_hero.comp is the LIT
# HERO kernel — it COPIES the rt_reflect.comp body VERBATIM (primary + shadow + one-bounce reflection + the
# integer blend) with the ONE change that every MISS color (primary AND reflection) is the integer
# SkyGradient(missDir) (a graded horizon->zenith sky) instead of the flat background. Like rt_reflect.comp it
# is VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd here: HLSL inline RayQuery lowers (via DXC) to SPIR-V
# RayQueryKHR which glslc/spirv-cross cannot lower, plus int64 fx math glslc can't parse. On Metal the
# --rt6-hero showcase runs the CPU rtrace::RenderSceneHero (the SAME bit-exact reference the Vulkan HW==CPU
# memcmp compares against) -> the Metal result is byte-identical to the Vulkan HW result BY CONSTRUCTION. So
# rt_hero.comp gets NO hf_gen_msl line.
# Deterministic Lumen-class GI Slice 1 (Slice GI1, the BEACHHEAD of FLAGSHIP #29): gi_probe_trace.comp is the
# DETERMINISTIC INTEGER RT PROBE-TRACE kernel — ONE thread per PROBE loops the 16 baked Fibonacci-sphere
# directions, a closest-hit RayQuery (the rt_query.comp candidate-drain VERBATIM) + a shadow RayQuery + the
# inline ShadeHitShadowed, writing the unpacked GiRadiance per (probe,ray). Like rt_query.comp it is
# VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd here: HLSL inline RayQuery lowers (via DXC) to SPIR-V RayQueryKHR
# which glslc/spirv-cross cannot lower, plus int64 fx math glslc can't parse. On Metal the --gi1-probe
# showcase runs the CPU gi::TraceProbeRays (the SAME bit-exact reference the Vulkan HW==CPU memcmp compares
# against) -> the Metal result is byte-identical to the Vulkan HW result BY CONSTRUCTION, while the Vulkan
# side carries the HW probe-radiance == CPU proof. So gi_probe_trace.comp gets NO hf_gen_msl line.
# Deterministic Lumen-class GI Slice 2 (Slice GI2, THE CRUX of FLAGSHIP #29): gi_sh_encode.comp is the
# DETERMINISTIC INTEGER 3rd-order SH ENCODE kernel — ONE thread per PROBE folds its 16 GI1 GiRadiance rays
# into a 9x3 FxProbeSH via an int64 (Q32.32) accumulator (acc += (int64)rad.ch * (int64)kGiSHBasis[r][i]),
# quantized to Q16.16 EXACTLY ONCE at the final /N >> kFrac normalize — bit-identical to gi::FxSHEncodeProbe.
# It has NO RayQuery / NO accel structure (pure-integer compute over the GI1 radiance buffer), but the int64
# accumulator (a single radiance*basis Q32.32 product overflows int32) keeps it VULKAN-SPIR-V-ONLY: DXC -spirv
# compiles int64_t, but glslc (the Metal HLSL->SPIR-V->MSL frontend) cannot parse int64_t in HLSL (the
# grain_integrate.comp / fluid_integrate.comp convention). So gi_sh_encode.comp gets NO hf_gen_msl line; on
# Metal the --gi2-shencode showcase runs the CPU gi::EncodeAllProbes (byte-identical by construction), while
# the Vulkan side carries the GPU SH == CPU SH memcmp proof.
# Deterministic Lumen-class GI Slice 3 (Slice GI3, the CONTINUOUS IRRADIANCE FIELD of FLAGSHIP #29):
# gi_interp.comp is the DETERMINISTIC INTEGER TRILINEAR SH INTERPOLATION kernel — ONE thread per QUERY POINT
# runs FxInterpolateIrradiance VERBATIM (FxNearestProbes partition-of-unity trilinear weights Σw==kOne ->
# FxInterpolateSH int64 8-corner blend, quantize once -> FxSHEvaluate cosine-lobe reconstruction) —
# bit-identical to gi::InterpolateField. It has NO RayQuery / NO accel structure (pure-integer compute over
# the GI2 SH buffer), but the int64 SH-blend accumulator (a single weight*coeff Q32.32 product overflows
# int32) keeps it VULKAN-SPIR-V-ONLY: DXC -spirv compiles int64_t, but glslc (the Metal HLSL->SPIR-V->MSL
# frontend) cannot parse int64_t in HLSL (the gi_sh_encode.comp / grain_integrate.comp convention). So
# gi_interp.comp gets NO hf_gen_msl line; on Metal the --gi3-interp showcase runs the CPU
# gi::InterpolateField + GiFieldToImage (byte-identical by construction), while the Vulkan side carries the
# GPU irradiance == CPU irradiance memcmp proof.
# Deterministic Lumen-class GI Slice 4 (Slice GI4, INTEGER MULTI-BOUNCE FEEDBACK of FLAGSHIP #29):
# gi_bounce.comp is the DETERMINISTIC INTEGER MULTI-BOUNCE re-trace kernel — ONE thread per PROBE loops the 16
# baked Fibonacci-sphere directions, a closest-hit RayQuery (the gi_probe_trace.comp candidate-drain VERBATIM)
# + a shadow RayQuery + the inline ShadeHitGI, which adds an INDIRECT term = the GI3 FxInterpolateIrradiance
# of the bound PREVIOUS SH buffer (gPrevSH) at the hit point (the trilinear blend + FxSHEvaluate copied from
# gi_interp.comp). Like gi_probe_trace.comp + gi_interp.comp it is VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd
# here: HLSL inline RayQuery lowers (via DXC) to SPIR-V RayQueryKHR which glslc/spirv-cross cannot lower, plus
# the int64 SH-blend glslc can't parse. On Metal the --gi4-bounce showcase runs the CPU gi::BounceProbes (the
# SAME bit-exact reference the Vulkan HW==CPU memcmp compares against) -> the Metal result is byte-identical to
# the Vulkan HW result BY CONSTRUCTION, while the Vulkan side carries the HW final-SH == CPU proof. So
# gi_bounce.comp gets NO hf_gen_msl line.
# Deterministic Lumen-class GI Slice 5 (Slice GI5, INTEGER CHEBYSHEV OCCLUSION WEIGHTING — the probe-volume
# LIGHT-LEAK FIX of FLAGSHIP #29): gi_occ.comp is the DETERMINISTIC INTEGER OCCLUSION-WEIGHTED interpolation
# kernel — ONE thread per QUERY POINT runs FxInterpolateIrradianceOcc (the GI3 trilinear blend, but each of
# the 8 corner weights is scaled by the variance-shadow CHEBYSHEV VISIBILITY of that corner probe from the
# per-probe distance moments gMoments, the 8 weights RE-NORMALIZED to Σ == kOne). Like gi_interp.comp it is
# VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd here: the int64 SH-blend accumulator PLUS the int64 Chebyshev
# fxdiv / t² / FxISqrt glslc cannot lower to MSL. On Metal the --gi5-occlusion showcase runs the CPU
# gi::InterpolateFieldOcc (the SAME bit-exact reference the Vulkan GPU==CPU memcmp compares against) -> the
# Metal result is byte-identical to the Vulkan GPU result BY CONSTRUCTION, while the Vulkan side carries the
# GPU==CPU irradiance proof. So gi_occ.comp gets NO hf_gen_msl line.
# Deterministic Lumen-class GI Slice 6 (Slice GI6, THE LIT GI HERO CAPSTONE — the FLAGSHIP #29 MONEY-SHOT):
# gi_hero.comp is the lit GI hero kernel — ONE thread per pixel copies the RT6 rt_hero.comp body VERBATIM
# (primary + RT3 shadow + RT4 reflection + the graded SkyGradient miss) and ADDS a GI INDIRECT term at each
# primary hit (albedo ⊙ FxInterpolateIrradianceOcc × giStrength, the GI3/GI5 SH blend + FxSHEvaluate). Like
# rt_hero.comp + gi_occ.comp it is VULKAN-SPIR-V-ONLY: HLSL inline RayQuery (DXC -> SPIR-V RayQueryKHR which
# glslc/spirv-cross can't lower to MSL) PLUS the int64 SH-blend / Chebyshev that glslc cannot parse. On Metal
# the --gi6-hero showcase runs the CPU gi::RenderSceneGI (the SAME bit-exact reference the Vulkan HW==CPU
# memcmp compares against) -> the Metal image is byte-identical to the Vulkan HW image BY CONSTRUCTION, while
# the Vulkan side carries the HW==CPU hero-image proof. So gi_hero.comp gets NO hf_gen_msl line.
# Deterministic GPU Cloth Slice 1 (Slice CL1, the BEACHHEAD of FLAGSHIP #8): cloth_integrate.comp runs the
# Q16.16 semi-implicit-Euler PARTICLE LATTICE INTEGRATOR — ONE thread per particle runs `steps` iterations
# of (vel += gravity*dt; prev = pos; pos += vel*dt; ground floor-clamp) on its OWN particle, the fxmul
# ((int64)a*b >> 16) + integrate + clamp copied VERBATIM from engine/sim/cloth.h::IntegrateParticle,
# writing gParticles[i] back (each particle independent in CL1 -> race-free, NO atomics; PINNED untouched).
# It is VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd here: its int64 fxmul (GENUINELY required because
# gravity*dt over Q16.16 overflows int32, the SAME form as fpx_integrate.comp) is compiled fine by DXC on
# the Vulkan path, but glslc (glslang's HLSL frontend used for the Metal HLSL->SPIR-V->MSL chain) cannot
# parse int64_t in HLSL, so it can't lower this shader. Same Vulkan-only convention as fpx_integrate.comp /
# fpx_solve.comp / swraster.comp: on Metal the --cloth-integrate showcase runs the CPU
# cloth::IntegrateParticles (the SAME bit-exact reference the Vulkan GPU==CPU proof compares against) ->
# the Metal result is byte-identical to the Vulkan GPU result BY CONSTRUCTION, while the Vulkan side
# carries the GPU==CPU bit-identity proof. So cloth_integrate.comp gets NO hf_gen_msl line.
# Deterministic GPU Granular/Sand Slice 1 (Slice GR1, the BEACHHEAD of FLAGSHIP #10): grain_integrate.comp
# runs the Q16.16 semi-implicit-Euler GRAIN POOL INTEGRATOR — ONE thread per grain runs `steps` iterations
# of (vel += gravity*dt; prev = pos; pos += vel*dt; RADIUS-AWARE ground rest: pos.y < groundY + radius ->
# groundY + radius) on its OWN grain, the fxmul ((int64)a*b >> 16) + integrate + prev-snap + radius-aware
# clamp copied VERBATIM from engine/sim/grain.h::IntegrateGrainParticle, writing gGrains[i] back (each grain
# independent in GR1 -> race-free, NO atomics). The GrainParticle adds a first-class `radius` field vs FL1's
# FluidParticle -> 48-byte std430 packing. It is VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd here: its int64
# fxmul (GENUINELY required because gravity*dt over Q16.16 overflows int32, the SAME form as
# fluid_integrate.comp / cloth_integrate.comp / fpx_integrate.comp) is compiled fine by DXC on the Vulkan
# path, but glslc (glslang's HLSL frontend used for the Metal HLSL->SPIR-V->MSL chain) cannot parse int64_t
# in HLSL, so it can't lower this shader. Same Vulkan-only convention as fluid_integrate.comp /
# cloth_integrate.comp / fpx_integrate.comp: on Metal the --grain-integrate showcase runs the CPU
# grain::IntegrateGrains (the SAME bit-exact reference the Vulkan GPU==CPU proof compares against) -> the
# Metal result is byte-identical to the Vulkan GPU result BY CONSTRUCTION, while the Vulkan side carries the
# GPU==CPU bit-identity proof. So grain_integrate.comp gets NO hf_gen_msl line.
# Deterministic Fixed-Point Physics Slice 2 (Slice FPX2): the integer-AABB BROADPHASE candidate-pair
# generator (the MC3 count->scan->emit on bodies). fpx_pair_count.comp (ONE thread per body i:
# perBodyCount[i]=Σ AabbOverlap(BodyAabb(i),BodyAabb(j)) over j>i), fpx_pair_scan.comp (SINGLE-THREAD
# exclusive prefix-sum -> perBodyOffset), fpx_pair_emit.comp (ONE thread per body i: emit each overlapping
# (i,j) into gPairs at its DISJOINT range, NO atomics). BodyAabb + AabbOverlap copied VERBATIM from
# engine/sim/fpx.h. Unlike fpx_integrate.comp, these are PURE INT32 (integer add/sub + six compares, NO
# products, NO int64), so glslc CAN lower them and they MSL-generate NATIVELY (the default MSL gen
# suffices, NO --msl-version 20200, the mc_classify/mc_count lesson). The Metal --fpx-pairs showcase
# dispatches the GPU shaders directly (NOT a CPU fallback) -> the GPU==CPU pair-list proof runs on Metal
# too. Pair list bit-exact GPU==CPU + cross-backend by construction. Behind --fpx-pairs only; existing
# shaders + goldens untouched.
hf_gen_msl(fpx_pair_count.comp   fpx_pair_count.comp.hlsl   comp fpx_pair_count_main)
hf_gen_msl(fpx_pair_scan.comp    fpx_pair_scan.comp.hlsl    comp fpx_pair_scan_main)
hf_gen_msl(fpx_pair_emit.comp    fpx_pair_emit.comp.hlsl    comp fpx_pair_emit_main)
# Deterministic GPU Fluid Slice 2 (Slice FL2, the 2nd slice of FLAGSHIP #9): the GRID-HASH NEIGHBOR SEARCH
# (the FPX2 count->scan->emit twin, applied to fluid particles). The CELL TABLE: fluid_cell_count.comp (ONE
# thread per particle: InterlockedAdd into its cell's counter — CellOf = FloorDiv per axis at cell-size h,
# the bounded-dense-grid linearization, VERBATIM engine/sim/fluid.h), fluid_cell_scan.comp (SINGLE-THREAD
# exclusive prefix-sum -> cellStart CSR), fluid_cell_emit.comp (SINGLE-THREAD ascending-particle scatter ->
# cellParticles, single-thread because the within-cell order must be ascending particle index). The NEIGHBOR
# LIST: fluid_neighbor_count.comp (ONE thread per particle: count 27-cell-stencil candidates passing the
# per-axis |dx|<h reject — NeighborAccept VERBATIM), fluid_neighbor_scan.comp (SINGLE-THREAD exclusive
# prefix-sum -> neighborStart), fluid_neighbor_emit.comp (ONE thread per particle emits its candidates into
# its DISJOINT neighbors[] slice, NO atomics). UNLIKE FL1's int64 fluid_integrate.comp (Vulkan-only -> CPU on
# Metal), these are PURE INT32 (FloorDiv integer divide + per-axis abs-compare, NO products/int64/fxmul/sqrt,
# NO radial r<h cull — deferred to FL3's kernel=0-beyond-h), so glslc CAN lower them and they MSL-generate
# NATIVELY (the default MSL gen suffices, the fpx_pair/cloth_edge lesson). The Metal --fluid-neighbors
# showcase dispatches the GPU shaders directly (NOT a CPU fallback) -> the GPU==CPU cell-table+neighbor-list
# proof runs on Metal too. Bit-exact GPU==CPU + cross-backend by construction. Behind --fluid-neighbors only;
# FL1's fluid_integrate + existing goldens untouched.
hf_gen_msl(fluid_cell_count.comp     fluid_cell_count.comp.hlsl     comp fluid_cell_count_main)
hf_gen_msl(fluid_cell_scan.comp      fluid_cell_scan.comp.hlsl      comp fluid_cell_scan_main)
hf_gen_msl(fluid_cell_emit.comp      fluid_cell_emit.comp.hlsl      comp fluid_cell_emit_main)
hf_gen_msl(fluid_neighbor_count.comp fluid_neighbor_count.comp.hlsl comp fluid_neighbor_count_main)
hf_gen_msl(fluid_neighbor_scan.comp  fluid_neighbor_scan.comp.hlsl  comp fluid_neighbor_scan_main)
hf_gen_msl(fluid_neighbor_emit.comp  fluid_neighbor_emit.comp.hlsl  comp fluid_neighbor_emit_main)
# Deterministic GPU Granular/Sand Slice 2 (Slice GR2, the 2nd slice of FLAGSHIP #10): the GRID-HASH NEIGHBOR
# SEARCH (the FL2 fluid_cell/fluid_neighbor twin, applied to grains). The CELL TABLE: grain_cell_count.comp
# (ONE thread per grain: InterlockedAdd into its cell's counter — GrainCellOf = FloorDiv per axis at cell-size
# hSearch, the bounded-dense-grid linearization, VERBATIM engine/sim/grain.h), grain_cell_scan.comp
# (SINGLE-THREAD exclusive prefix-sum -> cellStart CSR), grain_cell_emit.comp (SINGLE-THREAD ascending-grain
# scatter -> cellGrains, single-thread because the within-cell order must be ascending grain index). The
# NEIGHBOR LIST: grain_neighbor_count.comp (ONE thread per grain: count 27-cell-stencil candidates passing the
# per-axis |dx|<hSearch reject — GrainNeighborAccept VERBATIM), grain_neighbor_scan.comp (SINGLE-THREAD
# exclusive prefix-sum -> neighborStart), grain_neighbor_emit.comp (ONE thread per grain emits its candidates
# into its DISJOINT neighbors[] slice, NO atomics). UNLIKE GR1's int64 grain_integrate.comp (Vulkan-only ->
# CPU on Metal), these are PURE INT32 (FloorDiv integer divide + per-axis abs-compare, NO
# products/int64/fxmul/sqrt, NO radial overlap cull — deferred to GR3's contact solve), so glslc CAN lower
# them and they MSL-generate NATIVELY (the default MSL gen suffices, the fluid_cell/fluid_neighbor lesson).
# The Metal --grain-neighbors showcase dispatches the GPU shaders directly (NOT a CPU fallback) -> the
# GPU==CPU cell-table+neighbor-list proof runs on Metal too. Bit-exact GPU==CPU + cross-backend by
# construction. Behind --grain-neighbors only; GR1's grain_integrate + existing goldens untouched.
hf_gen_msl(grain_cell_count.comp     grain_cell_count.comp.hlsl     comp grain_cell_count_main)
hf_gen_msl(grain_cell_scan.comp      grain_cell_scan.comp.hlsl      comp grain_cell_scan_main)
hf_gen_msl(grain_cell_emit.comp      grain_cell_emit.comp.hlsl      comp grain_cell_emit_main)
hf_gen_msl(grain_neighbor_count.comp grain_neighbor_count.comp.hlsl comp grain_neighbor_count_main)
hf_gen_msl(grain_neighbor_scan.comp  grain_neighbor_scan.comp.hlsl  comp grain_neighbor_scan_main)
hf_gen_msl(grain_neighbor_emit.comp  grain_neighbor_emit.comp.hlsl  comp grain_neighbor_emit_main)
# Deterministic GPU Crowds Slice 2 (Slice BD2, the 2nd slice of FLAGSHIP #18): the GRID-HASH NEIGHBOR LIST
# (the GR2 grain_cell/grain_neighbor twin, applied to agents). The CELL TABLE: boids_cell_count.comp (ONE
# thread per agent: InterlockedAdd into its cell's counter — BoidsCellOf = FloorDiv per axis at cell-size
# radius, the bounded-dense-grid linearization, VERBATIM engine/sim/boids.h), boids_cell_scan.comp
# (SINGLE-THREAD exclusive prefix-sum -> cellStart CSR), boids_cell_emit.comp (SINGLE-THREAD ascending-agent
# scatter into cellAgents — single-thread because the within-cell order must be ascending agent index). The
# NEIGHBOR LIST: boids_neighbor_count.comp (ONE thread per agent: count 27-cell-stencil candidates passing the
# per-axis |dx|<radius reject — BoidsNeighborAccept VERBATIM), boids_neighbor_scan.comp (SINGLE-THREAD
# exclusive prefix-sum -> neighborStart), boids_neighbor_emit.comp (ONE thread per agent emits its candidates
# into its DISJOINT neighbors[] slice, NO atomics). ALL PURE INT32 (FloorDiv + per-axis abs-compare, NO
# int64/fxmul/sqrt — the box is the perception neighborhood, the int64 dist² compare AVOIDED), so UNLIKE BD1's
# int64 boids_steer these MSL-generate NATIVELY (a TRUE GPU pass on BOTH backends, the strongest cross-vendor
# proof). ReadBuffer reads the cell table + neighbor list, PROVEN BIT-EXACT vs the CPU
# boids.h::BuildBoidsCellTable + BuildBoidsNeighborList (memcmp). Behind --boids-neighbors only; BD1's
# boids_steer (Vulkan-only, NOT here) + existing goldens untouched.
hf_gen_msl(boids_cell_count.comp     boids_cell_count.comp.hlsl     comp boids_cell_count_main)
hf_gen_msl(boids_cell_scan.comp      boids_cell_scan.comp.hlsl      comp boids_cell_scan_main)
hf_gen_msl(boids_cell_emit.comp      boids_cell_emit.comp.hlsl      comp boids_cell_emit_main)
hf_gen_msl(boids_neighbor_count.comp boids_neighbor_count.comp.hlsl comp boids_neighbor_count_main)
hf_gen_msl(boids_neighbor_scan.comp  boids_neighbor_scan.comp.hlsl  comp boids_neighbor_scan_main)
hf_gen_msl(boids_neighbor_emit.comp  boids_neighbor_emit.comp.hlsl  comp boids_neighbor_emit_main)
# Deterministic Integer Broadphase Slice 1 (Slice BP1, the BEACHHEAD of FLAGSHIP #23: DETERMINISTIC INTEGER
# BROADPHASE, hf::sim::broad): THE BODY GRID + CSR CELL TABLE (the grain_cell_*/boids_cell_*/fluid_cell_* twin,
# keyed on fpx::FxBody instead of grains/agents/particles). broad_cell_count.comp (ONE thread per body:
# InterlockedAdd into its cell's counter — BodyCellOf = FloorDiv per axis at cell-size cellSize, the
# bounded-dense-grid linearization, VERBATIM engine/sim/broad.h), broad_cell_scan.comp (SINGLE-THREAD exclusive
# prefix-sum -> cellStart CSR), broad_cell_emit.comp (SINGLE-THREAD ascending-body scatter into cellBodies —
# single-thread because the within-cell order must be ascending body index, NOT GPU-scheduling order). ALL PURE
# INT32 (FloorDiv + integer compares + an ascending-index scatter, NO int64/fxmul/sqrt) so UNLIKE the int64
# fxmul integrators these MSL-generate NATIVELY (a TRUE GPU pass on BOTH backends, the strongest cross-vendor
# proof tier). ReadBuffer reads the cell table, PROVEN BIT-EXACT vs the CPU broad.h::BuildBodyCellTable (memcmp).
# Behind --broad-cell only; NO new RHI (rides the grain_cell compute-dispatch + SSBO path); all existing
# shaders + goldens untouched.
hf_gen_msl(broad_cell_count.comp     broad_cell_count.comp.hlsl     comp broad_cell_count_main)
hf_gen_msl(broad_cell_scan.comp      broad_cell_scan.comp.hlsl      comp broad_cell_scan_main)
hf_gen_msl(broad_cell_emit.comp      broad_cell_emit.comp.hlsl      comp broad_cell_emit_main)
# Deterministic Integer Broadphase Slice 2 (Slice BP2, THE CRUX of FLAGSHIP #23): THE CANDIDATE-PAIR GENERATOR —
# the 27-cell-stencil pair cull over the BP1 grid + CSR cell table. broad_pair_count.comp (ONE thread per body i:
# scan i's 3x3x3 stencil in FIXED order, count j>i with AabbOverlap — the six-compare predicate copied VERBATIM
# from broad.h/fpx.h), broad_pair_scan.comp (SINGLE-THREAD exclusive prefix-sum -> perBodyOffset),
# broad_pair_emit.comp (SINGLE-THREAD ascending-body scatter of the canonical fpx::FxPair{i,j} into each body's
# disjoint slice — single-thread because the per-i emit ORDER must be reproduced bit-for-bit). ALL PURE INT32
# (FloorDiv + integer compares + an ascending scatter, NO int64/fxmul/sqrt) so these MSL-generate NATIVELY (a
# TRUE GPU pass on BOTH backends). ReadBuffer reads pairsOut + offsets, PROVEN BIT-EXACT vs the CPU
# broad.h::BuildBroadphasePairs (memcmp) + grid-pairs==all-pairs equivalence vs fpx::BuildPairs. Behind
# --broad-pair only; NO new RHI; all existing shaders + goldens untouched.
hf_gen_msl(broad_pair_count.comp     broad_pair_count.comp.hlsl     comp broad_pair_count_main)
hf_gen_msl(broad_pair_scan.comp      broad_pair_scan.comp.hlsl      comp broad_pair_scan_main)
hf_gen_msl(broad_pair_emit.comp      broad_pair_emit.comp.hlsl      comp broad_pair_emit_main)
# Deterministic Integer CCD Slice 2 (Slice CD2, FLAGSHIP #24): THE SWEPT-AABB BROADPHASE — the 27-cell-stencil
# candidate-pair generator over the host-precomputed SWEPT AABBs (each body's box at the tick start UNION its box
# at the integrated end pose). ccd_swept_count.comp (ONE thread per body i: scan i's 3x3x3 stencil over the
# AABB-centre cell table, count j>i with the six-compare AabbOverlap over the explicit lo/hi swept boxes),
# ccd_swept_scan.comp (SINGLE-THREAD exclusive prefix-sum -> perBodyOffset), ccd_swept_emit.comp (SINGLE-THREAD
# ascending-body scatter of fpx::FxPair{i,j} into each body's disjoint slice — the per-i emit-order DET-CRUX). The
# swept-AABB precompute (integrate end pose + 2x BuildHullAabb + union — the int64 part) is done HOST-side and the
# int32 lo/hi AABBs are fed identically to the CPU ref + these shaders. ALL PURE INT32 (FloorDiv + integer compares
# + an ascending scatter, NO int64/fxmul/sqrt) so these MSL-generate NATIVELY (a TRUE GPU pass on BOTH backends).
# ReadBuffer reads pairsOut + offsets, PROVEN BIT-EXACT vs ccd.h::BuildSweptBroadphasePairs (memcmp) + the
# swept⊇discrete superset proof. Behind --ccd-swept only; NO new RHI; all existing shaders + goldens untouched.
hf_gen_msl(ccd_swept_count.comp      ccd_swept_count.comp.hlsl      comp ccd_swept_count_main)
hf_gen_msl(ccd_swept_scan.comp       ccd_swept_scan.comp.hlsl       comp ccd_swept_scan_main)
hf_gen_msl(ccd_swept_emit.comp       ccd_swept_emit.comp.hlsl       comp ccd_swept_emit_main)
# Deterministic Persistent Contacts Slice 1 (Slice PS1, the BEACHHEAD of FLAGSHIP #21: DETERMINISTIC
# WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, hf::sim::persist): persist_key.comp is THE CONTACT FEATURE
# ID pass — ONE thread per contact point runs the order-normalized integer key (bodyA<bodyB swap) + the fixed
# (bodyA<<20)^(bodyB<<8)^(axisIndex<<4)^featureIndex packing + avalanche xor-shift hash, copying
# engine/sim/persist.h::MakeContactKey + ContactKeyHash VERBATIM, writing its OWN ContactKey + hash per
# contact. UNLIKE the int64 contact MATH (convex_*/fric_* — Vulkan-SPIR-V-only because glslc cannot parse
# int64_t) the contact KEY is PURE INT32 (compares + shifts + xors, NO Q16.16 products, NO int64, NO float),
# so the default MSL gen suffices (the fract_emit_count / boids_cell_count tier) — a TRUE GPU pass on Metal,
# NOT a CPU fallback. The Metal --persist-key runs THIS shader; ReadBuffer reads the GPU ContactKey[]+hash[],
# PROVEN BIT-EXACT vs the CPU persist.h::MakeContactKey/ContactKeyHash (the same proof the Vulkan
# --persist-key-shot runs) -> cross-backend strict zero-differing-pixel. Behind --persist-key only; every
# existing shader + golden untouched. NO new RHI; persist.h is a NEW additive sibling.
hf_gen_msl(persist_key.comp         persist_key.comp.hlsl         comp persist_key_main)
# Warm-Started Hull Contacts Slice 1 (Slice WH1, the int32 MSL-native BEACHHEAD of FLAGSHIP #26: WARM-STARTED
# HULL CONTACTS + ROBUST DETERMINISTIC STACKING, hf::sim::warmhull): warmhull_key.comp is THE HULL CONTACT
# FEATURE ID pass — ONE thread per contact point runs the order-normalized integer key (bodyA<bodyB swap + the
# refIsA/role fold into refFaceId's high bit) + the fixed (bodyA<<20)^(bodyB<<8)^(refFaceId<<4)^incVertId
# packing + avalanche xor-shift hash, copying engine/sim/warmhull.h::MakeHullContactKey + HullContactKeyHash
# VERBATIM, writing its OWN HullContactKey + hash per contact. The (refFace, incTag) provenance is produced
# HOST-SIDE by the int64 manifold + the tagged clip; the SHADER is PURE INT32 (compares + shifts + xors, NO
# Q16.16 products, NO int64, NO float), so UNLIKE the int64 hull MATH (hull_manifold / hull_step_hardened —
# Vulkan-SPIR-V-only because glslc cannot parse int64_t) the default MSL gen suffices (the persist_key tier) —
# a TRUE GPU pass on Metal, NOT a CPU fallback. The Metal --wh1-keys runs THIS shader; ReadBuffer reads the GPU
# HullContactKey[]+hash[], PROVEN BIT-EXACT vs the CPU warmhull.h::MakeHullContactKey/HullContactKeyHash (the
# same proof the Vulkan --wh1-keys-shot runs) -> cross-backend strict zero-differing-pixel. Behind --wh1-keys
# only; every existing shader + golden untouched. NO new RHI; warmhull.h is a NEW additive sibling.
hf_gen_msl(warmhull_key.comp        warmhull_key.comp.hlsl        comp warmhull_key_main)
# Hull Friction + Joints Slice 1 (Slice HF1, the friction BEACHHEAD of FLAGSHIP #30): hullfric_points.comp is THE
# TAGGED FRICTION MANIFOLD ON THE EPA NORMAL — it consumes the host-built warmhull keyed manifold + the
# sign-corrected A->B EPA normal and builds the integer tangent basis (fric::MakeTangentBasis copied VERBATIM) + the
# basis-axis index + zeroed accumulators. UNLIKE warmhull_key.comp (the KEY alone is pure int32 -> MSL-native) the
# tangent basis FxNormalize/FxDot/FxCross are int64 (world-scale products overflow int32) AND the manifold it wraps
# is int64-derived, so hullfric_points.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is
# DELIBERATELY NOT MSL-gen'd here (NO hf_gen_msl line) — the Metal --hf1-points showcase runs the CPU
# hullfric::BuildAllHullFrictionManifoldsPairs (byte-identical to the Vulkan GPU result by construction, the
# warmhull_cache.comp convention), while the Vulkan --hf1-points-shot carries the GPU==CPU memcmp proof.
# Deterministic Rigid<->Fluid Coupling Slice 1 (Slice CP1, the BEACHHEAD of FLAGSHIP #11): the BODY->FLUID
# grid-hash QUERY (the GR2 grain_neighbor twin, per BODY instead of per grain) OVER the reused FL2 fluid cell
# table (fluid_cell_{count,scan,emit} above build the cellStart/cellParticles the query iterates — no new
# cell-table passes). couple_body_count.comp (ONE thread per body: count the fluid particles in the body's
# BodyAabb cell RANGE [CellOf(pos-radius) .. CellOf(pos+radius)] passing the per-axis |body.pos.axis -
# p.pos.axis| < body.radius reject — BodyParticleAccept VERBATIM; a body spans MANY cells, NOT a fixed 27-cell
# stencil), couple_body_scan.comp (SINGLE-THREAD exclusive prefix-sum -> bodyStart CSR + sentinel),
# couple_body_emit.comp (ONE thread per body emits its gathered particle indices into its DISJOINT
# bodyParticles[] slice, NO atomics). ALL PURE INT32 (FloorDiv integer divide + per-axis abs-compare, NO
# products/int64/fxmul/sqrt, NO radial sphere cull — deferred to CP2's force), so glslc CAN lower them and they
# MSL-generate NATIVELY (the default MSL gen suffices, the grain_neighbor lesson). The Metal --couple-query
# showcase dispatches the GPU shaders directly (NOT a CPU fallback) -> the GPU==CPU cell-table+query proof runs
# on Metal too. Bit-exact GPU==CPU + cross-backend by construction. Behind --couple-query only; the FL2/GR2
# pipelines + their goldens untouched; couple.h #includes fpx.h + fluid.h read-only (the additive sibling).
hf_gen_msl(couple_body_count.comp couple_body_count.comp.hlsl comp couple_body_count_main)
hf_gen_msl(couple_body_scan.comp  couple_body_scan.comp.hlsl  comp couple_body_scan_main)
hf_gen_msl(couple_body_emit.comp  couple_body_emit.comp.hlsl  comp couple_body_emit_main)
# Deterministic Rigid<->Grain Coupling Slice 1 (Slice CG1, the BEACHHEAD of FLAGSHIP #12): the BODY->GRAIN
# grid-hash QUERY (the CP1 couple_body twin, per BODY) OVER the reused GR2 GRAIN cell table
# (grain_cell_{count,scan,emit} above build the cellStart/cellGrains the query iterates — no new cell-table
# passes). cgrain_body_count.comp (ONE thread per body: count the grains in the body's BodyAabb cell RANGE
# [CellOf(pos-radius) .. CellOf(pos+radius)] passing the per-axis |body.pos.axis - g.pos.axis| < body.radius
# reject — BodyGrainAccept VERBATIM; a body spans MANY cells, NOT a fixed 27-cell stencil), cgrain_body_scan.comp
# (SINGLE-THREAD exclusive prefix-sum -> bodyStart CSR + sentinel), cgrain_body_emit.comp (ONE thread per body
# emits its gathered grain indices into its DISJOINT bodyGrains[] slice, NO atomics). ALL PURE INT32 (FloorDiv
# integer divide + per-axis abs-compare, NO products/int64/fxmul/sqrt, NO radial sphere cull — deferred to
# CG2/CG3), so glslc CAN lower them and they MSL-generate NATIVELY (the default MSL gen suffices, the
# couple_body/grain_neighbor lesson). The Metal --cgrain-query showcase dispatches the GPU shaders directly
# (NOT a CPU fallback) -> the GPU==CPU cell-table+query proof runs on Metal too. Bit-exact GPU==CPU +
# cross-backend by construction. Behind --cgrain-query only; the GR2 pipelines + their goldens untouched;
# couple_grain.h #includes fpx.h + grain.h read-only (the additive sibling).
hf_gen_msl(cgrain_body_count.comp cgrain_body_count.comp.hlsl comp cgrain_body_count_main)
hf_gen_msl(cgrain_body_scan.comp  cgrain_body_scan.comp.hlsl  comp cgrain_body_scan_main)
hf_gen_msl(cgrain_body_emit.comp  cgrain_body_emit.comp.hlsl  comp cgrain_body_emit_main)
# Deterministic Grain<->Fluid Coupling Slice 1 (Slice GF1, the BEACHHEAD of FLAGSHIP #13): the UNIFIED TWO-POOL
# WORLD + the SHARED-GRID CROSS QUERY (the GR2/FL2 27-cell-stencil neighbour search applied CROSS-POOL over ONE
# shared grid) OVER the reused GR2 grain cell table + FL2 fluid cell table (grain_cell_* + fluid_cell_* above
# build the two cellStart/cellGrains|cellParticles the cross query iterates, sized to the SHARED grid — no new
# cell-table passes). cgf_gf_{count,scan,emit}.comp (ONE thread per grain: count/emit the FLUID particles in the
# grain's 27-cell stencil over the FLUID cell table passing the per-axis |grain.axis - fluid.axis| < h reject —
# CrossAccept VERBATIM, NO j==i self-skip since the pools are distinct) + cgf_fg_{count,scan,emit}.comp (the
# MIRROR: ONE thread per fluid over the GRAIN cell table). ALL PURE INT32 (FloorDiv integer divide + per-axis
# abs-compare, NO int64/fxmul/sqrt, NO radial cull — deferred to GF2/GF3), so glslc CAN lower them and they
# MSL-generate NATIVELY (the default MSL gen suffices, the cgrain_body/grain_neighbor lesson). The Metal
# --cgf-query showcase dispatches the GPU shaders directly (NOT a CPU fallback) -> the GPU==CPU cell-table+cross
# proof runs on Metal too. Bit-exact GPU==CPU + cross-backend by construction. Behind --cgf-query only; the
# GR2/FL2 pipelines + their goldens untouched; couple_gf.h #includes fpx.h + grain.h + fluid.h read-only (additive).
hf_gen_msl(cgf_gf_count.comp cgf_gf_count.comp.hlsl comp cgf_gf_count_main)
hf_gen_msl(cgf_gf_scan.comp  cgf_gf_scan.comp.hlsl  comp cgf_gf_scan_main)
hf_gen_msl(cgf_gf_emit.comp  cgf_gf_emit.comp.hlsl  comp cgf_gf_emit_main)
hf_gen_msl(cgf_fg_count.comp cgf_fg_count.comp.hlsl comp cgf_fg_count_main)
hf_gen_msl(cgf_fg_scan.comp  cgf_fg_scan.comp.hlsl  comp cgf_fg_scan_main)
hf_gen_msl(cgf_fg_emit.comp  cgf_fg_emit.comp.hlsl  comp cgf_fg_emit_main)
# Deterministic GPU Cloth Slice 2 (Slice CL2, the 2nd slice of FLAGSHIP #8): the DISTANCE-CONSTRAINT GRAPH
# BUILD (the FPX2 count->scan->emit on lattice particles). cloth_edge_count.comp (ONE thread per particle:
# gEdgeCount[p]=#OWNED forward edges in bounds), cloth_edge_scan.comp (SINGLE-THREAD exclusive prefix-sum ->
# gEdgeOffset), cloth_edge_emit.comp (ONE thread per particle: emit each owned edge {i,j,restLen,kind} into
# gEdges at its DISJOINT range, NO atomics). CountOwnedEdges + the fixed emit order copied VERBATIM from
# engine/sim/cloth.h::BuildConstraints. UNLIKE CL1's int64 cloth_integrate.comp (Vulkan-only -> CPU on
# Metal), these are PURE INT32 (index arithmetic + bounds compares, NO products/int64, NO sqrt — restLen is
# the host-snapped int32 per kind), so glslc CAN lower them and they MSL-generate NATIVELY (the default MSL
# gen suffices, NO --msl-version 20200, the fpx_pair/nav_raster lesson). The Metal --cloth-edges showcase
# dispatches the GPU shaders directly (NOT a CPU fallback) -> the GPU==CPU constraint-list proof runs on
# Metal too. Edge list bit-exact GPU==CPU + cross-backend by construction. Behind --cloth-edges only;
# existing shaders + goldens untouched.
hf_gen_msl(cloth_edge_count.comp cloth_edge_count.comp.hlsl comp cloth_edge_count_main)
hf_gen_msl(cloth_edge_scan.comp  cloth_edge_scan.comp.hlsl  comp cloth_edge_scan_main)
hf_gen_msl(cloth_edge_emit.comp  cloth_edge_emit.comp.hlsl  comp cloth_edge_emit_main)
# Deterministic GPU Navmesh Slice 1 (Slice NAV1, the BEACHHEAD of FLAGSHIP #7: deterministic GPU navmesh
# + pathfinding): the integer HEIGHTFIELD SPAN RASTERIZATION (the MC3 count->scan->emit on voxel
# columns). nav_raster_count.comp (ONE thread per column (cx,cz): colCount[col]=#triangles whose XZ
# projection covers the column-center), nav_raster_scan.comp (SINGLE-THREAD exclusive prefix-sum ->
# colOffset), nav_raster_emit.comp (ONE thread per column: emit each covering tri's integer y-span into
# gSpans at its DISJOINT range, NO atomics). TriColumnAabb + PointInTriXZ (three integer edge functions)
# + TriYSpan copied VERBATIM from engine/nav/navmesh.h. Like the FPX2 broadphase + UNLIKE
# fpx_integrate.comp, these are PURE INT32 (integer add/sub/compare + small edge-function products that
# fit int32, NO int64), so glslc CAN lower them and they MSL-generate NATIVELY (the default MSL gen
# suffices, NO --msl-version 20200, the mc_classify/fpx_pair lesson). The Metal --nav-raster showcase
# dispatches the GPU shaders directly (NOT a CPU fallback) -> the GPU==CPU span-list proof runs on Metal
# too. Span list bit-exact GPU==CPU + cross-backend by construction. Behind --nav-raster only; existing
# shaders + goldens untouched.
hf_gen_msl(nav_raster_count.comp nav_raster_count.comp.hlsl comp nav_raster_count_main)
hf_gen_msl(nav_raster_scan.comp  nav_raster_scan.comp.hlsl  comp nav_raster_scan_main)
hf_gen_msl(nav_raster_emit.comp  nav_raster_emit.comp.hlsl  comp nav_raster_emit_main)
# Deterministic GPU Navmesh Slice 2 (Slice NAV2, the 2nd slice of FLAGSHIP #7): the WALKABLE FILTER +
# INTEGER CHAMFER DISTANCE FIELD over the NAV1 heightfield. nav_filter.comp (ONE thread per column: walk
# the column's MERGED spans, mark each span's top walkable by integer clearance-above >= walkableHeight,
# write the per-column walkable mask + the topmost walkable surface y), nav_distance.comp (SINGLE-THREAD
# [numthreads(1,1,1)] two-pass integer chamfer: seed non-walkable+border=0/walkable=INF, forward TL->BR
# then backward BR->TL min-sweep, cardinal weight 2 / diagonal weight 3, only traverse walkable+connected
# neighbours -> geodesic). The clearance/step + chamfer math is copied VERBATIM from
# engine/nav/navmesh.h. Like the NAV1 rasterizer + UNLIKE fpx_integrate.comp, these are PURE INT32 (NO
# int64), so glslc CAN lower them and they MSL-generate NATIVELY (the default MSL gen suffices). The Metal
# --nav-distance showcase dispatches the GPU shaders directly -> the GPU==CPU walkable[]+dist[] proof runs
# on Metal too; bit-exact cross-backend by construction. Behind --nav-distance only; NAV1 + existing
# shaders + goldens untouched.
hf_gen_msl(nav_filter.comp   nav_filter.comp.hlsl   comp nav_filter_main)
hf_gen_msl(nav_distance.comp nav_distance.comp.hlsl comp nav_distance_main)
# Deterministic GPU Navmesh Slice 3 (Slice NAV3, the 3rd slice + MAKE-OR-BREAK of FLAGSHIP #7):
# nav_region.comp — the INTEGER WATERSHED REGION GENERATION over the NAV2 distance field. SINGLE-THREAD
# ([numthreads(1,1,1)], the nav_distance/nav_raster_scan serial mirror) runs the WHOLE level-descending
# fixed-order watershed (for level=maxDist..1: GROW existing regions into this level by lowest-region-id
# among connected neighbours in ascending cellId until stable; then SEED + grow any still-unassigned
# cell). The watershed is copied VERBATIM from engine/nav/navmesh.h::BuildRegions. Like the NAV1/NAV2
# shaders + UNLIKE fpx_*.comp, it is PURE INT32 (NO int64), so glslc CAN lower it and it MSL-generates
# NATIVELY (the default MSL gen suffices). The Metal --nav-region showcase dispatches the GPU shader
# directly -> the GPU==CPU region[] proof runs on Metal too; bit-exact cross-backend by construction.
# Behind --nav-region only; NAV1+NAV2 + existing shaders + goldens untouched.
hf_gen_msl(nav_region.comp   nav_region.comp.hlsl   comp nav_region_main)
# nav_contour.comp + nav_polygonize.comp — Slice NAV4 CONTOUR TRACE + INTEGER POLYGONIZATION over the
# NAV3 region partition. nav_contour: SINGLE-THREAD ([numthreads(1,1,1)], the nav_region serial mirror)
# traces each region boundary into an integer corner loop (lowest-cellId start + fixed wall-follow turn
# order) then Douglas-Peucker simplifies (perpendicular-distance SQUARED vs maxError^2, no sqrt) ->
# gContourVerts + per-region offsets/counts. nav_polygonize: SINGLE-THREAD ear-clip triangulation (clip
# the LOWEST-index valid ear) + per-edge cross-poly adjacency -> gPolys + per-region offsets/counts.
# Both copied VERBATIM from engine/nav/navmesh.h::TraceContours/SimplifyContour/BuildPolyMesh. Like the
# NAV1-NAV3 shaders + UNLIKE fpx_*.comp, they are PURE INT32 (NO int64; corner coords <=32 for the 32x32
# showcase, the documented overflow bound), so glslc CAN lower them and they MSL-generate NATIVELY. The
# Metal --nav-polymesh showcase dispatches the GPU shaders directly -> the GPU==CPU proof runs on Metal
# too; bit-exact cross-backend by construction. Behind --nav-polymesh only; NAV1-NAV3 + goldens untouched.
hf_gen_msl(nav_contour.comp     nav_contour.comp.hlsl     comp nav_contour_main)
hf_gen_msl(nav_polygonize.comp  nav_polygonize.comp.hlsl  comp nav_polygonize_main)
# nav_astar.comp — Slice NAV5 INTEGER A* PATHFINDING over the NAV4 poly adjacency graph (the FLAGSHIP
# HEADLINE). SINGLE-THREAD ([numthreads(1,1,1)], the nav_region/nav_polygonize serial mirror) runs the
# WHOLE A*: connected-components flood -> deterministic start/goal (largest component, lowest-id ties) ->
# integer A* (linear min-scan frontier popping lowest f=g+h tie-break lowest poly id, came_from reconstruct)
# -> a CORRIDOR + total cost. Edge cost + heuristic = the MANHATTAN distance between integer poly centroids
# (admissible+consistent -> optimal A*). Copied VERBATIM from engine/nav/navmesh.h::ConnectedComponents/
# SelectStartGoal/FindPath. Like the NAV1-NAV4 shaders + UNLIKE fpx_*.comp, it is PURE INT32 (NO int64;
# centroids <=32, a single edge cost <=64, the documented overflow bound), so glslc CAN lower it and it
# MSL-generates NATIVELY. The Metal --nav-path showcase dispatches the GPU shader directly -> the GPU==CPU
# corridor proof runs on Metal too; bit-exact cross-backend by construction. Behind --nav-path only;
# NAV1-NAV4 + goldens untouched.
hf_gen_msl(nav_astar.comp        nav_astar.comp.hlsl        comp nav_astar_main)
# Deterministic Fixed-Point Physics Slice 3 (Slice FPX3, the MAKE-OR-BREAK of FLAGSHIP #6): fpx_solve.comp
# is the PBD POSITIONAL collision-response solver — a SINGLE THREAD ([numthreads(1,1,1)]) runs `steps` of
# StepWorld = IntegrateStep then SolveContacts (K=solveIters Gauss-Seidel sweeps: ground then sphere-sphere
# pairs in the FIXED FPX2 order), the fxdiv/FxNormalize/ground + sphere-sphere math copied VERBATIM from
# engine/sim/fpx.h. UNLIKE the int32 FPX2 broadphase shaders above (which MSL-gen natively), this is int64
# (fxdiv/FxISqrt for the sphere-sphere normal — GENUINELY required for the Q16.16 divide + integer sqrt),
# compiled fine by DXC on the Vulkan path but glslc CANNOT parse int64_t in HLSL. Same Vulkan-only
# convention as fpx_integrate.comp / swraster.comp: on Metal the --fpx-solve showcase runs the CPU
# fpx::StepWorld (the SAME bit-exact reference the Vulkan GPU==CPU proof compares against) -> the Metal
# result is byte-identical to the Vulkan GPU result BY CONSTRUCTION, while the Vulkan side carries the
# GPU==CPU bit-identity proof. So fpx_solve.comp gets NO hf_gen_msl line.
# Deterministic Fixed-Point Physics Slice 4 (Slice FPX4): fpx_orient.comp is the integer QUATERNION
# ORIENTATION integrator — ONE thread PER BODY (per-body INDEPENDENT, 64 threads/group, NO atomics) runs
# `steps` of IntegrateBodyFull = the FPX1 translational integrate + IntegrateOrientation (dq = ω⊗q;
# orient += 0.5*dq*dt; orient = FxQuatNormalize), the FxQuatMul/FxQuatNormalize/FxISqrt math copied
# VERBATIM from engine/sim/fpx.h. Like fpx_solve.comp this is int64 (the quaternion fxmul products +
# FxQuatNormalize's fxdiv/FxISqrt — GENUINELY required for the Q16.16 quaternion math), compiled fine by
# DXC on the Vulkan path but glslc CANNOT parse int64_t in HLSL. Same Vulkan-only convention as
# fpx_solve.comp / fpx_integrate.comp / swraster.comp: on Metal the --fpx-orient showcase runs the CPU
# fpx::IntegrateBodyFull (the SAME bit-exact reference the Vulkan GPU==CPU proof compares against) -> the
# Metal result is byte-identical to the Vulkan GPU result BY CONSTRUCTION, while the Vulkan side carries
# the GPU==CPU bit-identity proof. So fpx_orient.comp gets NO hf_gen_msl line.
# GPU Isosurface Meshing Slice 2 (Slice MC2): mc_count.comp counts the triangles each voxel CELL emits —
# ONE thread per cell runs CaseIndex (VERBATIM render/mc.h), looks up triCount = gTriCount[caseIdx] from
# the host-uploaded 256-entry count table (built from render/mc.h::kTriTable), writes gCounts[cell] and
# InterlockedAdd(gTotal[0], triCount) the grand total (the autoexposure_histogram integer-atomic
# precedent — commutative -> order-independent -> deterministic). countEnabled=0 -> writes 0, no atomic.
# The integer atomic lowers on Metal with NO MSL-2.2 feature -> the default MSL gen suffices (NO
# --msl-version 20200 needed, the mc_classify lesson). Counts + total bit-exact GPU==CPU + cross-backend
# by construction. Behind --mc-count only; existing shaders + goldens untouched.
hf_gen_msl(mc_count.comp         mc_count.comp.hlsl         comp mc_count_main)
# GPU Isosurface Meshing Slice 3 (Slice MC3): the prefix-sum compaction + triangle EMISSION. mc_scan.comp
# is a SINGLE-THREAD ([numthreads(1,1,1)]) exclusive prefix-sum of the MC2 per-cell counts -> each cell's
# write offset (the VT2 vt_alloc single-thread-allocator pattern; serial dependency -> one thread ->
# GPU==CPU bit-exact). mc_emit.comp is ONE thread per cell: reclassify via CaseIndex (VERBATIM
# render/mc.h), read triOffset=gOffsets[cell], walk gTriTable emitting each triangle's 3 edge-MIDPOINT
# vertices (EdgeMidpoint = the integer SUM of two corner coords in half-units, NO float) + the identity
# index into its DISJOINT [3*triOffset,..) range (race-free, NO atomics). meshEnabled=0 -> emit nothing.
# Plain integer SSBO writes (no atomics, no integer texture.read) -> the default MSL gen suffices (NO
# --msl-version 20200 needed, the mc_classify/mc_count lesson). Vertex+index buffers bit-exact GPU==CPU +
# cross-backend by construction. Behind --mc-emit only; existing shaders + goldens untouched.
hf_gen_msl(mc_scan.comp          mc_scan.comp.hlsl          comp mc_scan_main)
hf_gen_msl(mc_emit.comp          mc_emit.comp.hlsl          comp mc_emit_main)
# GPU Isosurface Meshing Slice 4 (Slice MC4): mc_interp.comp is mc_emit.comp with EdgeInterp instead of
# EdgeMidpoint — the per-cell triangle emission with FIXED-POINT INTERPOLATED vertex placement (the vertex
# sits on the ACTUAL isosurface crossing, t = ((iso-s0)*kSub)/(s1-s0) on the 1/kSub lattice, the SAME
# truncating integer divide both sides -> bit-exact GPU==CPU + cross-backend). INT32 throughout (numerator
# fits) -> no int64; plain integer SSBO writes -> the default MSL gen suffices (NO --msl-version 20200, the
# mc_classify/mc_count/mc_emit lesson). Reuses mc_scan.comp UNCHANGED. Behind --mc-interp only; existing
# MC1-MC3 shaders + goldens untouched.
hf_gen_msl(mc_interp.comp        mc_interp.comp.hlsl        comp mc_interp_main)
# Nanite Software-Raster Slice 2 (Slice SW2): swraster.comp is the GPU COMPUTE software rasterizer (one
# thread per cluster-triangle scan-converts into a w*h depth|id visibility-buffer SSBO via InterlockedMin).
# It is VULKAN-SPIR-V-ONLY and is NOT MSL-gen'd here: its int64 integer edge math (int64_t in render/
# swraster.h, REQUIRED because the snapped-coord products exceed 32 bits) is compiled fine by DXC on the
# Vulkan path, but glslc (glslang's HLSL frontend used for the Metal HLSL->SPIR-V->MSL chain) cannot parse
# int64_t in HLSL, so it can't lower this shader. Same Vulkan-only convention as cluster_cull.comp /
# gpudriven_cull.comp / cluster_hiz_cull.comp (which glslc/spirv-cross likewise can't lower): on Metal the
# --swraster-gpu showcase renders the IDENTICAL result via the CPU bound path (render/swraster.h::
# RasterClusters — the SAME bit-exact reference the Vulkan GPU==CPU proof compares against), while the
# Vulkan side carries the GPU==CPU bit-identity proof. So swraster.comp gets NO hf_gen_msl line.
# Nanite Software-Raster RESOLVE blit (Slice SW4): swraster_resolve_blit.frag is a FULLSCREEN pass (reuses
# post.vert) that unpacks the SW software-rasterizer's packed depth|id vis-buffer SSBO into the R32_Uint
# visId render target the EXISTING visresolve.frag deferred resolve consumes (out=(packed==0xFFFFFFFF)?
# 0xFFFFFFFF:(packed&0xFFFF)), so the resolve is reused BYTE-FOR-BYTE. Reads the SW SSBO via the set-3
# BindLightClusters fragment-storage path (binding 13) + a {width,height} UBO. CRITICAL: this shader is
# PLAIN MSL — it has NO int64 and NO integer texture.read; it only WRITES an integer SV_Target (which is
# plain MSL), so it gets NO --msl-version 20200 (unlike visresolve.frag, which keeps its 20200 for the
# integer texture.read). NO new RHI. Behind --swraster-resolve.
hf_gen_msl(swraster_resolve_blit.frag swraster_resolve_blit.frag.hlsl frag swraster_resolve_blit_fragment)
# Virtual Shadow Maps Slice 2 (Slice VB): vsm_depth.frag renders each resident PHYSICAL page's casters'
# depth into its atlas tile. Paired with shadow_csm.vert (the page's light-space ortho VP in the push
# constant); unlike the depth-only CSM caster it renders into a COLOR atlas (SV_Position.z -> grayscale,
# near=bright) so the per-tile depth is ReadRenderTarget'd back as the golden. Default MSL gen suffices.
hf_gen_msl(vsm_depth.frag         vsm_depth.frag.hlsl         frag vsm_depth_fragment)
# Virtual Shadow Maps Slice 3 (Slice VC): lit_vsm.frag — the LIT-PASS VSM indirection sample (the shadowed
# scene), a SIBLING of lit_csm.frag. Per receiver pixel: VA-threshold-ladder clipmap level -> virtual page
# -> physical tile via the indirection SSBO (binding 13, BindLightClusters) -> PageWorldOrtho light-space ->
# tile-clamped 3x3 PCF over the DEPTH atlas (SetShadowMap). vsmEnabled=0 -> unshadowed byte-identical. The
# SAME HLSL->SPIR-V->MSL chain; lit_csm.frag + its goldens untouched (sibling isolation). NO new RHI.
hf_gen_msl(lit_vsm.frag           lit_vsm.frag.hlsl           frag vsm_sample_fragment)
# Text / HUD renderer (Slice BA): NDC pass-through vertex + atlas-coverage fragment (configurable
# text color in a FRAGMENT push constant -> the HF_MSL_GEN cbuffer at the flat fragment push-const
# slot, like the bloom passes). Alpha-blended screen-space overlay drawn over the scene after post.
hf_gen_msl(text.vert text.vert.hlsl vert text_vertex)
hf_gen_msl(text.frag text.frag.hlsl frag text_fragment)
# CPU particle / VFX emitter billboards (Slice CC): world-space camera-facing quad vertex (frame UBO,
# no push constant -> the FrameData cbuffer at the flat Metal frame slot) + procedural soft-sprite
# fragment modulated by the per-particle color. Additive-blended over the scene. DISTINCT from the
# gpu-particles fountain. Drives the --vfx showcase.
hf_gen_msl(vfx.vert vfx.vert.hlsl vert vfx_vertex)
hf_gen_msl(vfx.frag vfx.frag.hlsl frag vfx_fragment)
# SSAO chain (Slice Y): view-space normal+linear-depth g-buffer prepass (static + instanced vertex
# variants + shared fragment), hemisphere-kernel AO pass, box blur, composite/apply. Same
# HLSL->SPIR-V->MSL toolchain as the others.
hf_gen_msl(gbuffer.vert           gbuffer.vert.hlsl           vert gbuffer_vertex)
hf_gen_msl(gbuffer_instanced.vert gbuffer_instanced.vert.hlsl vert gbuffer_instanced_vertex)
hf_gen_msl(gbuffer.frag           gbuffer.frag.hlsl           frag gbuffer_fragment)
hf_gen_msl(ssao.frag              ssao.frag.hlsl              frag ssao_fragment)
hf_gen_msl(ssao_blur.frag         ssao_blur.frag.hlsl         frag ssao_blur_fragment)
hf_gen_msl(ssao_composite.frag    ssao_composite.frag.hlsl    frag ssao_composite_fragment)
# Screen-space reflections (Slice AH): reuses the SSAO g-buffer (view normal + linear depth); ssr.frag
# ray-marches the depth buffer for on-screen reflections, ssr_composite blends + tonemaps. Same
# HLSL->SPIR-V->MSL toolchain. Existing shaders + goldens untouched.
hf_gen_msl(ssr.frag               ssr.frag.hlsl               frag ssr_fragment)
hf_gen_msl(ssr_composite.frag     ssr_composite.frag.hlsl     frag ssr_composite_fragment)
# Depth of field (Slice CG): reuses the SSAO/SSR g-buffer (view normal + linear depth) + the resolved
# HDR scene; dof.frag computes the thin-lens circle of confusion per pixel (render/dof.h math, mirrored
# in-shader) and gathers a CoC-sized Vogel disk weighted by the scatter-as-gather BlurWeight so the
# focal plane stays sharp while fore/background blur, then tonemaps. Same scene/g-buffer pair bindings
# as SSR; existing shaders + goldens untouched.
hf_gen_msl(dof.frag               dof.frag.hlsl               frag dof_fragment)
# Per-object + camera motion blur (Slice CN): reuses the SSAO/SSR/DoF g-buffer (view normal + linear
# depth) + the resolved HDR scene; motion_blur.frag reconstructs each pixel's view position, computes the
# screen-space velocity from the prev/cur view-proj (render/motion_blur.h math, mirrored in-shader),
# clamps it to maxBlurPx and gathers N taps along it weighted by the depth-aware TapWeight, then tonemaps.
# Same scene/g-buffer pair bindings as DoF; the SAME render/motion_blur.h math makes the velocity field
# bit-identical cross-backend. Existing shaders + goldens untouched.
hf_gen_msl(motion_blur.frag       motion_blur.frag.hlsl       frag motion_blur_fragment)
# Order-Independent Transparency (Slice CO): Weighted Blended OIT (McGuire & Bavoil). oit_accum.vert
# rasterizes the transparent geometry + passes the view-space linear depth (clip.w); oit_accum.frag emits
# premultColor*Weight (additive ONE,ONE accum) and oit_revealage.frag emits alpha (multiplicative
# oitRevealageBlend dst*=(1-alpha)); oit_resolve.frag composites accum.rgb/max(accum.a,eps) + coverage,
# and water_composite lerps it over the opaque scene (== oit::ResolveOver) + tonemaps. The SAME
# render/oit.h Weight makes the result bit-identical cross-backend. Existing shaders/goldens untouched.
hf_gen_msl(oit_accum.vert         oit_accum.vert.hlsl         vert oit_accum_vertex)
hf_gen_msl(oit_accum.frag         oit_accum.frag.hlsl         frag oit_accum_fragment)
hf_gen_msl(oit_revealage.frag     oit_revealage.frag.hlsl     frag oit_revealage_fragment)
hf_gen_msl(oit_resolve.frag       oit_resolve.frag.hlsl       frag oit_resolve_fragment)
# Parallax Occlusion Mapping (Slice CP): pom.vert (lit.vert + heightScale/numSteps forwarded in the
# material push constant .zw) + pom.frag (TBN -> tangent-space view dir -> render/pom.h ParallaxUV march
# of the height texture -> displaced-uv lit shading + pom::SelfShadow). The SAME render/pom.h math
# mirrored in-shader makes the marched uv bit-identical to the Vulkan/CPU path; heightScale==0 degenerates
# to the base uv -> byte-identical to the plain normal-mapped lit render. Behind --pom only.
hf_gen_msl(pom.vert               pom.vert.hlsl               vert vertex_main)
hf_gen_msl(pom.frag               pom.frag.hlsl               frag fragment_main)
# Water rendering (Slice CF): water.vert displaces a 128x128 grid by the Gerstner field (render/water.h
# math, mirrored in-shader); water.frag fresnel-blends a SKY reflection + a depth-tinted REFRACTION of
# the scene-color RT (G-buffer depth like SSR) + a sun glint; water_composite blends + tonemaps. Same
# FrameData + scene/g-buffer pair bindings as SSR; existing shaders/goldens untouched.
hf_gen_msl(water.vert             water.vert.hlsl             vert water_vertex)
hf_gen_msl(water.frag             water.frag.hlsl             frag water_fragment)
hf_gen_msl(water_composite.frag   water_composite.frag.hlsl   frag water_composite_fragment)
# Ground-Truth Ambient Occlusion (Slice CR): gtao.frag reconstructs the view position+normal from the
# SSAO g-buffer (like ssao.frag) and runs the render/gtao.h horizon-search visibility integral
# (IntegrateArc + HorizonAngle + Visibility mirrored in-shader) with a baked per-pixel slice rotation;
# the EXISTING ssao_composite.frag multiplies the ambient term by it. The SAME render/gtao.h math
# compiled here makes the AO bit-identical to the Vulkan/CPU path; radius==0 -> AO==1 -> byte-identical
# to the no-AO render. Behind --gtao only; existing ssao/ssr shaders + goldens untouched.
hf_gen_msl(gtao.frag              gtao.frag.hlsl              frag gtao_fragment)
# Subsurface Scattering (Slice CZ): sss_mask.frag (reuses gbuffer.vert) renders the flagged subsurface
# objects into an "SSS G-buffer" RT (mask.r + view linear depth.w); sss_blur.frag is a fullscreen pass
# run TWICE (horizontal then vertical) that gathers the render/sss.h diffusion profile (DiffusionWeight x
# DepthFalloff x mask, mirrored in-shader) along the axis so subsurface pixels' lit color softens, while
# non-subsurface pixels pass through. The SAME render/sss.h math compiled here makes the diffusion
# bit-identical to the Vulkan/CPU path; sssStrength==0 (or sssWidth==0) -> the kernel collapses to the
# center tap -> byte-identical to the non-SSS lit render. Behind --sss only; existing shaders + goldens
# untouched.
hf_gen_msl(sss_mask.frag          sss_mask.frag.hlsl          frag sss_mask_fragment)
hf_gen_msl(sss_blur.frag          sss_blur.frag.hlsl          frag sss_blur_fragment)
# Color Grading (Slice DB): color_grade.frag is a fullscreen pass (reuses post.vert) that reads the
# resolved TONEMAPPED scene color and applies render/color_grade.h's Grade (analytic lift/gamma/gain +
# ASC-CDL slope/offset/power + luma-preserving saturation, mirrored in-shader) from a small GradeParams
# push constant. The SAME render/color_grade.h math compiled here makes the grade bit-identical to the
# Vulkan/CPU path; at the IDENTITY params the grade is a pure pass-through -> byte-identical to the
# ungraded tonemap render. Behind --colorgrade only; existing shaders + goldens untouched.
hf_gen_msl(color_grade.frag       color_grade.frag.hlsl       frag color_grade_fragment)
# Contrast-Adaptive Sharpening (Slice DF; AMD FidelityFX CAS): cas.frag is a NEW fullscreen pass (reuses
# post.vert) that reads the resolved TONEMAPPED SDR scene color, gathers the cross neighborhood (center +
# up/down/left/right via the texel size from GetDimensions), and applies render/cas.h's CasSharpen
# (CasWeight + CasSharpen mirrored in-shader; out = (center + w*(up+down+left+right))/(1+4w) clamped to
# the neighborhood [min,max]) from a small `sharpness` push constant. The SAME render/cas.h math compiled
# here makes the sharpen bit-identical to the Vulkan/CPU path; at sharpness 0 CasWeight==0 -> a pure
# pass-through -> byte-identical to the unsharpened tonemap render. Behind --cas only; existing shaders +
# goldens untouched.
hf_gen_msl(cas.frag               cas.frag.hlsl               frag cas_fragment)
# Screen-Space Contact Shadows (Slice CT): contact_shadows.frag reconstructs the view position from the
# SSAO/GTAO g-buffer and marches a SHORT ray TOWARD the sun through the depth field (render/
# contact_shadows.h RayMarchShadow mirrored in-shader) with a baked per-pixel dither, outputting the
# single-channel contact-shadow factor; lit_contactshadow.frag (a lit.frag variant) samples that factor
# at the pixel's screen position + multiplies the direct sun radiance by it (ambient/IBL/point lights
# untouched). The SAME render/contact_shadows.h math compiled here makes the factor bit-identical to the
# Vulkan/CPU path; maxDist==0 -> factor==1 -> byte-identical to the no-contact render. Behind
# --contactshadow only; existing lit/gtao/ssao shaders + goldens untouched.
hf_gen_msl(contact_shadows.frag   contact_shadows.frag.hlsl   frag contact_shadows_fragment)
hf_gen_msl(lit_contactshadow.frag lit_contactshadow.frag.hlsl frag contactshadow_fragment)
# DDGI GI Composite — DDGI Slice 5 (Slice DN): a SIBLING of lit.frag adding a single-bounce DDGI
# indirect-diffuse term (the visible Cornell color bleed). Per pixel it trilinearly blends the 8 nearest
# probes' 3rd-order real-SH irradiance (NearestProbes + InterpolateSH + SHEvaluate copied VERBATIM from
# engine/render/probe_gi.h + probe_sh.h, mad for every accumulation) toward the surface normal. The ProbeSH[]
# rides in a fragment storage buffer (binding 13/space3 -> Metal fragment buffer 13) bound via the existing
# usesLightClusters/BindLightClusters path. The SAME math compiled here makes the indirect term bit-identical
# to the Vulkan/CPU mirror; giStrength==0 (or probeCount=0) -> a literal +0.0 -> byte-identical to the no-GI
# render. Behind --ddgi only; existing lit shaders + goldens untouched.
hf_gen_msl(lit_ddgi.frag           lit_ddgi.frag.hlsl           frag ddgi_fragment)
# DDGI Visibility Slice 2 — Chebyshev occlusion-weighted GI composite (Slice DP): a SIBLING of lit_ddgi.frag
# that weights each of the 8 nearest probes' SH contribution by a per-probe Chebyshev (variance-shadow)
# visibility weight read from DO's per-probe distance-moment cube, killing DDGI light-leak through occluders.
# ProbeSH at set-3 binding 13 (Metal fragment buffer 13) + the distance moments at binding 14 (buffer 14),
# both via the existing usesLightClusters/BindLightClusters path. The SAME math compiled here makes the
# occlusion weight bit-identical to the Vulkan/CPU mirror; occlusionStrength==0 -> the VERBATIM DN path ->
# byte-identical to the DN render. Behind --ddgiocc only; existing lit/ddgi shaders + goldens untouched.
hf_gen_msl(lit_ddgi_occ.frag       lit_ddgi_occ.frag.hlsl       frag ddgi_occ_fragment)
# DDGI MULTI-BOUNCE capture-with-GI fragment (Slice DR): a SIBLING of probe_bake.frag (the DI probe-radiance
# capture) with the DN lit_ddgi InterpolateIrradianceSH indirect term added LAST, so the bounce-1 capture
# sees the 1st-bounce GI -> SH-encoding it yields SH1 (the 2nd bounce, brighter/more-filled). SH0 binds at
# set-3 binding 13 (Metal fragment buffer 13) via the existing usesLightClusters/BindLightClusters path; the
# grid params ride the same 208-byte lit_ddgi FrameData UBO (fragment buffer 0). The SAME math compiled here
# makes the bounce-1 indirect bit-identical to the Vulkan/CPU mirror. Behind --ddgimb only; probe_bake.frag +
# lit_ddgi.frag + their goldens untouched.
hf_gen_msl(probe_bake_gi.frag      probe_bake_gi.frag.hlsl      frag probe_bake_gi_fragment)
# Froxel Volumetric Fog (Slice CS): a true 3D view-space froxel volume. froxel_inject.comp writes
# per-froxel (scatter, extinction) from the render/froxel.h height-density + HG phase; froxel_integrate.comp
# marches each Z column front-to-back via IntegrateStep into (inScatter, transmittance); froxel_apply.frag
# samples the integrated volume at each pixel's view-depth slice (ViewZToSlice) + composites scene*T +
# inScatter. The SAME render/froxel.h math compiled here makes the fog bit-identical to the Vulkan/CPU
# path; baseDensity==0 -> T==1, inScatter==0 -> byte-identical to the no-fog render. Behind --froxelfog
# only; existing shaders + goldens untouched.
hf_gen_msl(froxel_inject.comp     froxel_inject.comp.hlsl     comp froxel_inject_main)
hf_gen_msl(froxel_integrate.comp  froxel_integrate.comp.hlsl  comp froxel_integrate_main)
hf_gen_msl(froxel_apply.frag      froxel_apply.frag.hlsl      frag froxel_apply_fragment)
# Probe-Grid Ray-Trace — DDGI beachhead (Slice DH): one thread per probe traces 16 deterministic
# Fibonacci-sphere rays against the scene's view-space depth field (a FLAT float SSBO read back from the
# G-buffer, NOT a sampled depth texture) into a flat ProbeRayHit SSBO. FibonacciSphere +
# ViewToScreenUV/ReconstructViewPos + TraceRayToDepth are copied VERBATIM from engine/render/probe_gi.h
# (+ ssr.h); the host precomputes the 16 Fibonacci dirs + uploads them as exact bits + the shader marks
# the marched position `precise`, so the GPU ray-hit SSBO is BIT-EXACT to the CPU reference on BOTH
# backends. Behind --probegi only; existing shaders + goldens untouched.
hf_gen_msl(probe_raytrace.comp    probe_raytrace.comp.hlsl    comp probe_raytrace_main)
# Probe SH-Encode — DDGI Slice 3 (Slice DJ): one thread per probe loops the FIXED host-precomputed sample
# set (per cube-texel: the 9 SHBasis9 weights + the solid-angle weight) and accumulates the captured
# radiance (read from a flat radiance SSBO) into 3rd-order real-SH coefficients, writing the per-probe
# ProbeSH SSBO. SHEncodeAccumulate/SHNormalize are copied VERBATIM from engine/render/probe_sh.h; the
# host precomputes the sample basis weights + solid-angle weights + uploads them as exact bits, so the
# GPU per-probe SH SSBO is BIT-EXACT to the CPU SH-encode on BOTH backends. Behind --probesh only;
# existing shaders + goldens untouched.
hf_gen_msl(probe_sh_encode.comp   probe_sh_encode.comp.hlsl   comp probe_sh_encode_main)
# Probe Trilinear SH-Interp — DDGI Slice 4 (Slice DL): one thread per QUERY point finds the FLOOR cell
# whose 8 corner probes bracket the query world position (NearestProbes) and trilinearly blends those 8
# probes' 3rd-order real-SH coefficients into a single per-query ProbeSH. NearestProbes (FLOOR-cell 8
# corners + polynomial weights) + InterpolateSH (the per-corner fma SH accumulation) are copied VERBATIM
# from engine/render/probe_gi.h + probe_sh.h (mad for the per-corner product, matching std::fma), so the
# GPU blended-SH SSBO is BIT-EXACT to the CPU InterpolateSH reference on BOTH backends. Behind
# --probeinterp only; existing shaders + goldens untouched.
hf_gen_msl(probe_interp.comp      probe_interp.comp.hlsl      comp probe_interp_main)
# Per-Probe Distance-Moment Capture — DDGI Visibility Slice 1 (Slice DO): the VISIBILITY data layer.
# probe_dist.vert clones probe_bake.vert (passes the world position + carries probeCentre in the push
# constant); probe_dist.frag outputs the linear world-distance d = length(worldPos - probeCentre) packed
# as the two moments float2(d, d*d) (d*d a BARE multiply, bit-identical to the CPU reference
# probedist::MomentsFromDistance). Pure geometry (no lighting), captured into the DD cube RT + read back
# into a per-probe distance-moment store (the data the next slice DP, Chebyshev occlusion, consumes to
# kill DDGI leak). Same HLSL->SPIR-V->MSL toolchain -> bit-identical to the Vulkan path; new golden
# probe_dist.png. Behind --probedist only; existing shaders + goldens untouched. NO new RHI.
hf_gen_msl(probe_dist.vert        probe_dist.vert.hlsl        vert probe_dist_vertex)
hf_gen_msl(probe_dist.frag        probe_dist.frag.hlsl        frag probe_dist_fragment)
# Auto-Exposure (Slice CW): histogram eye adaptation. autoexposure_histogram.comp reads the HDR scene
# (flat float4 SSBO), Luminance -> LumToBin -> InterlockedAdd(histogram[bin],1) into a bins-entry INTEGER
# SSBO (order-independent counts -> deterministic; Metal threadgroup atomic_uint maps from the SPIR-V
# integer atomic). autoexposure_reduce.comp reduces -> AverageLuminance (black-bin excluded) ->
# ExposureFromAverage -> writes the exposure float (gated by adaptationEnabled: false -> E0).
# tonemap_autoexp.frag is post.frag with kExposure replaced by that SSBO value (cluster binding 13). The
# SAME render/auto_exposure.h math compiled here makes the result bit-identical to the Vulkan/CPU path;
# adaptationEnabled=false -> E0 -> byte-identical to the standard tonemap. Behind --autoexposure only;
# existing shaders + goldens untouched.
hf_gen_msl(autoexposure_histogram.comp autoexposure_histogram.comp.hlsl comp autoexposure_histogram_main)
hf_gen_msl(autoexposure_reduce.comp    autoexposure_reduce.comp.hlsl    comp autoexposure_reduce_main)
hf_gen_msl(tonemap_autoexp.frag        tonemap_autoexp.frag.hlsl        frag tonemap_autoexp_fragment)
# Screen-space global illumination (Slice BP): reuses the SSAO/SSR g-buffer + lit HDR scene; ssgi.frag
# traces K cosine-hemisphere rays and marches the depth buffer (same march as ssr.frag) for indirect
# diffuse, ssgi_composite ADDS it over the scene + tonemaps. Same HLSL->SPIR-V->MSL toolchain.
hf_gen_msl(ssgi.frag              ssgi.frag.hlsl              frag ssgi_fragment)
hf_gen_msl(ssgi_composite.frag    ssgi_composite.frag.hlsl    frag ssgi_composite_fragment)
# SSGI spatial denoise (Slice BR): a depth+normal-guided bilateral blur of the SSGI indirect buffer
# (same NxN fullscreen shape as ssao_blur, edge-stopped per tap by the g-buffer) inserted before
# ssgi_composite on the --ssgi-denoise path only. Same HLSL->SPIR-V->MSL toolchain.
hf_gen_msl(ssgi_denoise.frag      ssgi_denoise.frag.hlsl      frag ssgi_denoise_fragment)
# Temporal SSGI accumulation (Slice BV): a running-mean accumulation pass that folds N=8 jittered SSGI
# indirect frames (each a golden-angle-rotated hemisphere kernel via ssgi.frag's `frame` param) into an
# HDR accumulation RT before ssgi_composite, on the --ssgi-temporal path only. Same HLSL->SPIR-V->MSL.
hf_gen_msl(ssgi_accum.frag        ssgi_accum.frag.hlsl        frag ssgi_accum_fragment)
# Volumetric fog / light shafts (Slice AJ): reuses the directional shadow map + SSAO g-buffer;
# volumetric.frag ray-marches the view ray sampling the shadow map per step for in-scattering,
# volumetric_composite adds it over the scene + tonemaps. Same HF_MSL_GEN shadow V-flip as lit.frag.
hf_gen_msl(volumetric.frag           volumetric.frag.hlsl           frag volumetric_fragment)
hf_gen_msl(volumetric_composite.frag volumetric_composite.frag.hlsl frag volumetric_composite_fragment)
# Volumetric clouds (Slice CH): a raymarched cumulus LAYER in the sky dome (DISTINCT from the fog).
# clouds.frag reconstructs the view ray, intersects the cloud slab, ray-marches render/clouds.h density
# (mirrored in-shader so noise/density is bit-identical cross-backend) with a short Beer-Lambert light
# march + a Henyey-Greenstein phase, masked to the sky background via the g-buffer depth;
# clouds_composite blends lerp(scene, cloud.rgb, cloud.a) + tonemaps. Same HLSL->SPIR-V->MSL toolchain.
hf_gen_msl(clouds.frag           clouds.frag.hlsl           frag clouds_fragment)
hf_gen_msl(clouds_composite.frag clouds_composite.frag.hlsl frag clouds_composite_fragment)
# Cloud shadows on the ground (Slice CK): a VARIANT of lit.frag that ADDITIONALLY attenuates the direct
# sun by the cloud-shadow transmittance (render/clouds.h CloudShadow, mirrored in-shader so the shadow
# field is bit-identical cross-backend). Reuses lit.vert; the push constant is the standard lit
# {model,material} extended with cloudParams (slab/time/steps), read in the FRAGMENT stage (HF_MSL_GEN
# cbuffer at the fragment push-const slot, like the bloom/clouds fragment push constants). Drives the
# --cloud-shadows showcase; lit.frag + its goldens are untouched.
hf_gen_msl(lit_cloudshadow.frag  lit_cloudshadow.frag.hlsl  frag cloudshadow_fragment)
# Reflection + irradiance probes (Slice AK): bake the local room into a cube atlas (reflection block)
# + cosine-convolve an irradiance block, then shade with lit_probe sampling that atlas as LOCAL GI.
# lit_probe reuses lit.vert; the blit/convolution passes reuse post.vert. Same HLSL->SPIR-V->MSL
# toolchain; the probe atlas binds at the env fragment texture(11)/sampler(12) like the HDR env.
hf_gen_msl(probe_bake.vert       probe_bake.vert.hlsl       vert probe_bake_vertex)
hf_gen_msl(probe_bake.frag       probe_bake.frag.hlsl       frag probe_bake_fragment)
hf_gen_msl(lit_probe.frag        lit_probe.frag.hlsl        frag probe_fragment)
hf_gen_msl(probe_blit.frag       probe_blit.frag.hlsl       frag probe_blit_fragment)
hf_gen_msl(probe_irradiance.frag probe_irradiance.frag.hlsl frag probe_irradiance_fragment)
# Box-Projected Cubemap Reflections (Slice DA): a reflective-surface variant of lit_probe.frag that
# parallax-corrects the specular reflection direction to the local probe BOX before sampling the
# reflection cubemap. Reuses lit.vert + the probe bake/blit/irradiance + post.vert passes. Same
# HLSL->SPIR-V->MSL toolchain -> bit-identical to the Vulkan path; new golden refl_probe.png.
hf_gen_msl(reflprobe.frag        reflprobe.frag.hlsl        frag reflprobe_fragment)
# Runtime cubemap-capture reflection probe (Slice DD): a reflective-surface variant of reflprobe.frag
# that samples a CAPTURED hardware TextureCube (the scene rendered into 6 cube faces from the probe
# center, via the additive cubemap-RT interface) box-projected by DA's reflprobe math. Reuses lit.vert
# + probe_bake.{vert,frag} (the per-face capture) + post.vert. Same HLSL->SPIR-V->MSL toolchain ->
# bit-identical to the Vulkan path; new golden capture_probe.png.
hf_gen_msl(captureprobe.frag     captureprobe.frag.hlsl     frag captureprobe_fragment)
# Planar reflections (Slice DE): the MIRROR-SURFACE fragment — shades the flat reflective floor with
# lit.frag's matte Cook-Torrance result, then blends lerp(matte, reflectionSample, reflectivity), the
# reflection RT being the scene rendered through the camera reflected across the mirror plane
# (render::planar) sampled at the pixel's own screen UV. reflectivity=0 -> byte-identical to the matte
# render. Reuses lit.vert + post.{vert,frag}. Same HLSL->SPIR-V->MSL toolchain -> bit-identical to the
# Vulkan path; new golden planar_reflection.png.
hf_gen_msl(planar.frag           planar.frag.hlsl           frag planar_fragment)
# Temporal anti-aliasing (Slice AP): the resolve pass — neighborhood-clamped history blend over the
# current jittered HDR scene. Reuses post.vert (fullscreen triangle). Same HLSL->SPIR-V->MSL toolchain;
# existing shaders + goldens untouched.
hf_gen_msl(taa_resolve.frag      taa_resolve.frag.hlsl      frag taa_resolve_fragment)
# Screen-space projected decals (Slice BH): reuses the SSAO/SSR g-buffer (view normal + linear depth)
# + the lit HDR scene. decal.frag reconstructs the world position (view-space recon like SSR, then
# view->world via invView), projects it into the decal box, and alpha-blends a procedural decal over
# the scene + tonemaps. Reuses post.vert. Same HLSL->SPIR-V->MSL toolchain; existing shaders+goldens
# untouched.
hf_gen_msl(decal.frag            decal.frag.hlsl            frag decal_fragment)
# Data-driven post-process stack (Slice BN): the configurable ORDERED composite chain (tonemap/
# colorgrade/chromatic/vignette/grain applied in order from a push-constant effect list). A SEPARATE
# final pass from post.frag; reuses post.vert. Same HLSL->SPIR-V->MSL toolchain; shared math with
# engine/render/post_stack.h. Existing shaders + goldens untouched.
hf_gen_msl(post_stack.frag       post_stack.frag.hlsl       frag post_stack_fragment)
# Docked editor UI (Slice BT): the Dear ImGui pipeline shaders (ImDrawVert -> ortho clip via a vertex
# push constant; font-atlas sample * vertex color). Same HLSL->SPIR-V->MSL toolchain as everything
# else; the ImGuiRenderer (RHI-only) consumes these MSL modules on the Metal --editor path exactly as
# it consumes the SPIR-V on Vulkan.
hf_gen_msl(ui.vert ui.vert.hlsl vert ui_vertex)
hf_gen_msl(ui.frag ui.frag.hlsl frag ui_fragment)

add_custom_target(hf_gen_shaders DEPENDS ${HF_GENERATED_MSL})

add_executable(visual_test
    visual_test.mm
    # The REAL Metal RHI backend, built headless (no hal/window.h needed for the offscreen path).
    ${HF_ROOT}/engine/rhi_metal/metal_device.mm
    ${HF_ROOT}/engine/rhi_metal/metal_swapchain.mm
    ${HF_ROOT}/engine/rhi_metal/metal_command_buffer.mm
    ${HF_ROOT}/engine/rhi_metal/metal_pipeline.mm
    ${HF_ROOT}/engine/rhi_metal/metal_compute_pipeline.mm
    ${HF_ROOT}/engine/rhi_metal/metal_texture.mm
    ${HF_ROOT}/engine/rhi_metal/metal_render_target.mm
    ${HF_ROOT}/engine/rhi_metal/metal_cubemap_target.mm
    ${HF_ROOT}/engine/rhi_metal/metal_shader.mm
    ${HF_ROOT}/engine/rhi_metal/metal_shader_load.mm
    ${HF_ROOT}/engine/rhi_metal/metal_buffer.mm
    # Deterministic parallel command-recording partition + worker pool (Slice AU). Pure C++
    # (std::thread); drives the --mt showcase's multithreaded scene-pass recording.
    ${HF_ROOT}/engine/runtime/parallel_record.cpp
    # Declarative render graph (backend-agnostic; depends only on rhi/ + scene/ + std).
    ${HF_ROOT}/engine/render/render_graph.cpp
    # Scene layer (plain C++): builds Mesh::Cube / Mesh::Plane through the RHI.
    ${HF_ROOT}/engine/scene/mesh.cpp
    # Data-driven scene load/dump (plain C++, vendored single-header JSON parser).
    ${HF_ROOT}/engine/scene/scene_io.cpp
    # Distance-based scene/asset streaming (Slice BD, pure C++): the --stream showcase.
    ${HF_ROOT}/engine/scene/streaming.cpp
    # glTF loader (plain C++, header-only cgltf): loads a real model into a scene::Mesh.
    ${HF_ROOT}/engine/asset/gltf_loader.cpp
    # HDR environment loader (Slice R): decodes env.hdr, builds the CPU mip chain, uploads RGBA16F.
    ${HF_ROOT}/engine/asset/env_loader.cpp
    # Skeletal-animation sampling (plain C++): Quat/Slerp/FromTRS live in math.h; this is the
    # SampleAnimation palette builder (Slice O).
    ${HF_ROOT}/engine/anim/animation.cpp
    # Animation state machine / cross-fade (Slice BL): the SAME pure-C++ StateMachine TU the Windows
    # engine compiles, so the FSM-produced joint palette is BIT-IDENTICAL across backends (the
    # cross-backend golden contract). No Metal/RHI deps. Drives the --anim-fsm showcase.
    ${HF_ROOT}/engine/anim/state_machine.cpp
    # Impulse-based rigid-body world (Slice S): pure C++ physics core (no Metal/RHI deps).
    ${HF_ROOT}/engine/physics/world.cpp
    # Procedural terrain / heightmap (Slice BF): the SAME pure-C++ Height + BuildTerrain TU the
    # Windows engine compiles, so the generated terrain mesh is BIT-IDENTICAL across backends (the
    # cross-backend golden contract). No Metal/RHI deps. Drives the --terrain showcase.
    ${HF_ROOT}/engine/terrain/heightmap.cpp
    # Terrain-streaming LOD integration (Slice BJ): the SAME pure-C++ TileGrid + residency/LOD TU the
    # Windows engine compiles, so the per-tile meshes are BIT-IDENTICAL across backends. No Metal/RHI
    # deps. Drives the --terrain-stream showcase.
    ${HF_ROOT}/engine/terrain/terrain_stream.cpp
    # Deterministic roll-a-ball gameplay loop (Slice AX): pure C++ (math + physics, no Metal/RHI
    # deps). Drives the --game showcase's scripted playthrough + capture-step state.
    ${HF_ROOT}/engine/game/roll_game.cpp
    # Networking / replication snapshot layer (Slice BQ): the SAME pure-C++ snapshot.cpp the Windows
    # engine compiles, so the serialized snapshot/delta BYTES are bit-identical cross-backend. No
    # Metal/RHI deps. Drives the --net showcase's authority->replica stream + replica-scene render.
    ${HF_ROOT}/engine/net/snapshot.cpp
    # Networking TRANSPORT + client jitter-buffer / interpolation (Slice BU): the SAME pure-C++
    # transport.cpp the Windows engine compiles, so the seeded channel's drop/reorder decisions + the
    # client interpolation are BIT-IDENTICAL cross-backend (the netsim stat line matches Vulkan). No
    # Metal/RHI deps. Drives the --netsim showcase's authority->lossy-channel->client-interp render.
    ${HF_ROOT}/engine/net/transport.cpp
    # Client PREDICTION + server RECONCILIATION (Slice BY): the SAME pure-C++ prediction.cpp the Windows
    # engine compiles, so the predicted sim + rewind/replay reconciliation are BIT-IDENTICAL cross-backend
    # (the netpredict stat line matches Vulkan). No Metal/RHI deps. Drives the --netpredict showcase's
    # authority(+server-only impulse)->delayed-channel->predicted-client render.
    ${HF_ROOT}/engine/net/prediction.cpp
    # Text / HUD layout + baked font atlas (Slice BA): pure C++ (no Metal/RHI deps). Drives the
    # --hud + --game-hud showcases' screen-space alpha-blended text overlay.
    ${HF_ROOT}/engine/ui/text.cpp
    # CPU particle / VFX emitter system (Slice CC): the SAME pure-C++ particles.cpp the Windows engine
    # compiles, so the simulated particle state + camera-facing billboards are BIT-IDENTICAL
    # cross-backend (the vfx stat line matches Vulkan). No Metal/RHI deps. Drives the --vfx showcase's
    # additive-blended billboard fountain. DISTINCT from the gpu-particles compute fountain.
    ${HF_ROOT}/engine/vfx/particles.cpp
    # Immediate-mode debug-draw collector + engine-object emitters (Slice W): pure C++ (math +
    # scene + physics), no Metal/RHI deps. Drives the LINE_LIST debug overlay.
    ${HF_ROOT}/engine/debug/debug_draw.cpp
    ${HF_ROOT}/engine/debug/debug_emitters.cpp
    # Editor transform-gizmo emit (Slice AB): pure C++ (math + scene::Transform + debug::DebugDraw),
    # no Metal/RHI deps. Drives the --gizmo showcase's translate-gizmo line overlay.
    ${HF_ROOT}/engine/editor/gizmo.cpp
    # Docked editor UI (Slice BT): the panel DATA model (pure C++, ImGui-free), the ECS-driven docked
    # panels (ImGui), the RHI-only ImGui renderer, and the vendored Dear ImGui core. All
    # backend-agnostic (the renderer uses ONLY the rhi:: seam); the --editor showcase draws the docked
    # editor chrome over the scene exactly like the Vulkan --editor-shot path.
    ${HF_ROOT}/engine/editor/editor_panel_data.cpp
    ${HF_ROOT}/engine/editor/edit_ops.cpp
    ${HF_ROOT}/engine/editor/editor_panels.cpp
    ${HF_ROOT}/engine/editor/imgui_renderer.cpp
    ${HF_ROOT}/third_party/imgui/imgui.cpp
    ${HF_ROOT}/third_party/imgui/imgui_draw.cpp
    ${HF_ROOT}/third_party/imgui/imgui_tables.cpp
    ${HF_ROOT}/third_party/imgui/imgui_widgets.cpp
)

# The generated MSL must exist before the executable's shaders are loaded at runtime; make the
# build depend on the generation target so it always runs first.
add_dependencies(visual_test hf_gen_shaders)

# Compile all Obj-C++ with ARC.
target_compile_options(visual_test PRIVATE -fobjc-arc)

# Include roots so #include "rhi/rhi.h", "rhi_metal/...", "scene/...", "asset/...", "math/..."
# and the header-only "cgltf/cgltf.h" resolve.
target_include_directories(visual_test PRIVATE
    ${HF_ROOT}/engine
    ${HF_ROOT}/third_party
    ${HF_ROOT}/third_party/imgui
    # Slice RT5: the SHARED sim->RT bridge tests/rt5_simrender_scene.h (used by the Vulkan + Metal showcases
    # AND rt_simrender_test) lives under tests/ to keep rtrace.h sim-free + fpx.h render-free; on the include
    # path so --rt5-simrender (visual_test.mm) can #include it.
    ${HF_ROOT}/tests
)

# Docked editor UI (Slice BT): force 32-bit ImDrawIdx (matches the engine's RHI index binding) via the
# SAME hf_imconfig.h the Windows build uses, so ImGui's draw data binds straight through the Metal RHI.
target_compile_definitions(visual_test PRIVATE
    "IMGUI_USER_CONFIG=\"${HF_ROOT}/engine/editor/hf_imconfig.h\"")

# Path to the GENERATED MSL shaders (HLSL->SPIR-V->MSL), baked in as a compile definition.
target_compile_definitions(visual_test PRIVATE
    HF_SHADER_DIR="${HF_ROOT}/shaders"
    HF_GEN_SHADER_DIR="${HF_GEN_SHADER_DIR}"
    HF_MODEL_PATH="${HF_ROOT}/assets/models/Duck.glb"
    HF_FOX_MODEL_PATH="${HF_ROOT}/assets/models/Fox.glb"
    HF_HELMET_MODEL_PATH="${HF_ROOT}/assets/models/DamagedHelmet.glb"
    HF_TRUCK_MODEL_PATH="${HF_ROOT}/assets/models/CesiumMilkTruck.glb"
    HF_SCENE_PATH="${HF_ROOT}/assets/scenes/default.json"
    HF_ENV_PATH="${HF_ROOT}/assets/env/env.hdr"
)

target_link_libraries(visual_test PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework QuartzCore"
    "-framework CoreGraphics"
    "-framework ImageIO"
    "-framework CoreServices"
    "-framework UniformTypeIdentifiers"
)
