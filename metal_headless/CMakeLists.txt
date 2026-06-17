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
# hf_gen_msl(<basename> <hlsl-file> <vert|frag> <new-entry-point>)
function(hf_gen_msl base src stage entry)
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
            --rename-entry-point main ${entry} ${sc_stage}
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
