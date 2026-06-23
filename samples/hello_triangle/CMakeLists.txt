add_executable(hello_triangle main.cpp)
# Vulkan/Windows sample links the editor shell (Dear ImGui through the RHI); the editor is gated to
# non-Apple in the root CMakeLists, so this link only happens on the Vulkan path.
target_link_libraries(hello_triangle PRIVATE hf_engine hf_editor)
target_compile_definitions(hello_triangle PRIVATE HF_HAS_EDITOR=1)
# Slice RT5: the sim->RT bridge (tests/rt5_simrender_scene.h) is SHARED by the Vulkan + Metal showcases AND
# the rt_simrender_test, so the converter is ONE source of truth. It lives under tests/ (NOT engine/) to keep
# rtrace.h sim-free + fpx.h render-free; add that dir to the include path so main.cpp can #include it.
target_include_directories(hello_triangle PRIVATE "${CMAKE_SOURCE_DIR}/tests")

# main() is one monolithic function with ~190 inline `--*-shot` showcase blocks; their locals are all
# reserved together in main()'s stack frame, which grew past the default 1 MB Windows stack under the
# debug /RTC1 stack-frame checks (release is fine), crashing EVERY showcase at main() entry with
# STATUS_STACK_OVERFLOW (0xC00000FD). Reserve a larger stack so the debug build runs. Pure build-config
# (no logic / RHI / shader change); harmless on release.
if (MSVC)
    target_link_options(hello_triangle PRIVATE /STACK:16777216)
    # main.cpp is one ~42k-line translation unit with ~190 inline `--*-shot` showcase blocks; the COFF
    # object's section count crossed the default 65535 limit (fatal error C1128) once the CL4
    # --cloth-collide-shot block was added. /bigobj raises the section cap. Pure build-config (no logic /
    # RHI / shader change); standard remedy for a large single TU.
    target_compile_options(hello_triangle PRIVATE /bigobj)
endif()

# --- Material graph (Slice AV): regenerate the showcase fragment shader from its *.mat.json at build
# time via the material_codegen host tool. The JSON is the source of truth; the generated HLSL is
# committed (shaders/generated/) for diff transparency + so the standalone Metal build has it without
# running the tool. Re-runs whenever the JSON, the tool, or the shared PBR core changes. -------------
set(MAT_JSON "${CMAKE_SOURCE_DIR}/assets/materials/showcase.mat.json")
set(MAT_GEN_HLSL "${CMAKE_SOURCE_DIR}/shaders/generated/mat_showcase.frag.hlsl")
add_custom_command(
  OUTPUT "${MAT_GEN_HLSL}"
  COMMAND material_codegen "${MAT_JSON}" "${MAT_GEN_HLSL}"
  DEPENDS material_codegen "${MAT_JSON}"
          "${CMAKE_SOURCE_DIR}/shaders/pbr_core.hlsli"
  COMMENT "material_codegen showcase.mat.json -> mat_showcase.frag.hlsl"
  VERBATIM)
# Slice AW: a SECOND material (different node mix) regenerated the same way; its committed HLSL is
# what Metal renders (Metal stays build-time). On Vulkan it is rendered via the RUNTIME compile path.
set(MAT2_JSON "${CMAKE_SOURCE_DIR}/assets/materials/showcase2.mat.json")
set(MAT2_GEN_HLSL "${CMAKE_SOURCE_DIR}/shaders/generated/mat_showcase2.frag.hlsl")
add_custom_command(
  OUTPUT "${MAT2_GEN_HLSL}"
  COMMAND material_codegen "${MAT2_JSON}" "${MAT2_GEN_HLSL}"
  DEPENDS material_codegen "${MAT2_JSON}"
          "${CMAKE_SOURCE_DIR}/shaders/pbr_core.hlsli"
  COMMENT "material_codegen showcase2.mat.json -> mat_showcase2.frag.hlsl"
  VERBATIM)
# Slice AZ: a THIRD material exercising the new node types (Swizzle/MakeFloat3/Power/Saturate/
# OneMinus). The multi-material showcase draws all three in one frame. Same build-time codegen path.
set(MAT3_JSON "${CMAKE_SOURCE_DIR}/assets/materials/showcase3.mat.json")
set(MAT3_GEN_HLSL "${CMAKE_SOURCE_DIR}/shaders/generated/mat_showcase3.frag.hlsl")
add_custom_command(
  OUTPUT "${MAT3_GEN_HLSL}"
  COMMAND material_codegen "${MAT3_JSON}" "${MAT3_GEN_HLSL}"
  DEPENDS material_codegen "${MAT3_JSON}"
          "${CMAKE_SOURCE_DIR}/shaders/pbr_core.hlsli"
  COMMENT "material_codegen showcase3.mat.json -> mat_showcase3.frag.hlsl"
  VERBATIM)
add_custom_target(hello_triangle_matgen DEPENDS "${MAT_GEN_HLSL}" "${MAT2_GEN_HLSL}" "${MAT3_GEN_HLSL}")
# Slice BE: the NormalMap-node material (PBROutput.normal = NormalMap(slot="normalmap")). Same
# build-time codegen path; its committed generated HLSL flows through the same DXC->SPIR-V (and
# glslc->MSL) toolchain. The --material-normal-shot showcase renders a sphere shaded by it.
set(MATNORMAL_JSON "${CMAKE_SOURCE_DIR}/assets/materials/normalmap.mat.json")
set(MATNORMAL_GEN_HLSL "${CMAKE_SOURCE_DIR}/shaders/generated/mat_normalmap.frag.hlsl")
add_custom_command(
  OUTPUT "${MATNORMAL_GEN_HLSL}"
  COMMAND material_codegen "${MATNORMAL_JSON}" "${MATNORMAL_GEN_HLSL}"
  DEPENDS material_codegen "${MATNORMAL_JSON}"
          "${CMAKE_SOURCE_DIR}/shaders/pbr_core.hlsli"
  COMMENT "material_codegen normalmap.mat.json -> mat_normalmap.frag.hlsl"
  VERBATIM)
add_custom_target(hello_triangle_matgen_normal DEPENDS "${MATNORMAL_GEN_HLSL}")
add_dependencies(hello_triangle hello_triangle_matgen hello_triangle_matgen_normal)

set(SHADER_OUT "${CMAKE_CURRENT_BINARY_DIR}/shaders")
# The ui.vert/frag shaders are Vulkan-only (editor) — added only to the Vulkan sample's compile list,
# never to the metal_headless gen list, so the Metal shader generation is untouched.
hf_compile_shaders(
  TARGET hello_triangle
  OUTPUT_DIR "${SHADER_OUT}"
  SHADERS
    "shaders/lit.vert.hlsl:vs"
    "shaders/lit.frag.hlsl:ps"
    "shaders/lit_csm.frag.hlsl:ps"
    "shaders/shadow_csm.vert.hlsl:vs"
    # Spot light + perspective spot-light shadow (Slice AE): cone-cut lit fragment + reuse the CSM
    # depth-only caster (spotViewProj in the push constant).
    "shaders/lit_spot.frag.hlsl:ps"
    # Omnidirectional point-light shadow (Slice AF): 6-face cube atlas lit fragment + reuse the CSM
    # depth-only caster (per-face view-proj in the push constant).
    "shaders/lit_point.frag.hlsl:ps"
    # Clustered / Forward+ lit fragment (Slice AG): per-fragment cluster index + iterate that
    # cluster's point lights from three fragment-stage storage buffers (set 3).
    "shaders/lit_clustered.frag.hlsl:ps"
    "shaders/lit_pbr.frag.hlsl:ps"
    "shaders/lit_pbr_ibl.frag.hlsl:ps"
    # Data-driven material graph (Slice AV): the build-time-generated fragment from
    # assets/materials/showcase.mat.json (regenerated by the hello_triangle_matgen custom command
    # above). Reuses lit.vert + the existing PBR descriptor layout; shares pbr_core.hlsli.
    "shaders/generated/mat_showcase.frag.hlsl:ps"
    # Slice AW: the SECOND material's build-time-generated fragment. On Vulkan the live/dry-run path
    # RUNTIME-compiles showcase2.mat.json; this build-time .spv is kept for parity/diagnostics + so the
    # generated HLSL is exercised by the same DXC the runtime path uses.
    "shaders/generated/mat_showcase2.frag.hlsl:ps"
    # Slice AZ: the THIRD material's build-time-generated fragment (new node types). Drawn alongside
    # showcase + showcase2 in the --material-multi-shot multi-material scene; also runtime-compiled
    # via --material-live-shot to prove the new nodes survive the dxc runtime path.
    "shaders/generated/mat_showcase3.frag.hlsl:ps"
    # Slice BE: the NormalMap-node material's build-time-generated fragment (a tangent-space normal
    # map perturbs the shading normal via the same TBN as lit.frag). Drawn by --material-normal-shot;
    # also runtime-compiled via --material-live-shot to prove the NormalMap node survives the dxc path.
    "shaders/generated/mat_normalmap.frag.hlsl:ps"
    "shaders/sky_hdr.frag.hlsl:ps"
    "shaders/lit_skinned.vert.hlsl:vs"
    "shaders/shadow_skinned.vert.hlsl:vs"
    "shaders/lit_instanced.vert.hlsl:vs"
    "shaders/shadow_instanced.vert.hlsl:vs"
    "shaders/transparent.vert.hlsl:vs"
    "shaders/transparent.frag.hlsl:ps"
    "shaders/shadow.vert.hlsl:vs"
    "shaders/shadow.frag.hlsl:ps"
    "shaders/post.vert.hlsl:vs"
    "shaders/post.frag.hlsl:ps"
    "shaders/bloom_prefilter.frag.hlsl:ps"
    "shaders/bloom_downsample.frag.hlsl:ps"
    "shaders/bloom_upsample.frag.hlsl:ps"
    "shaders/bloom_composite.frag.hlsl:ps"
    "shaders/sky.vert.hlsl:vs"
    "shaders/sky.frag.hlsl:ps"
    # Animated procedural sky (issue #5): the time-channel demo. Reads FrameData.skyParams.z = time to
    # drift a cloud band; shares HFSkyColor (procedural_sky.hlsli, issue #4). Behind --sky-animated-shot.
    "shaders/sky_animated.frag.hlsl:ps"
    "shaders/particles.comp.hlsl:cs"
    "shaders/particle.vert.hlsl:vs"
    "shaders/particle.frag.hlsl:ps"
    "shaders/ui.vert.hlsl:vs"
    "shaders/ui.frag.hlsl:ps"
    # Text / HUD renderer (Slice BA): NDC pass-through vertex + atlas-coverage fragment (configurable
    # text color in a fragment push constant). Alpha-blended screen-space overlay drawn after post.
    "shaders/text.vert.hlsl:vs"
    "shaders/text.frag.hlsl:ps"
    # CPU particle / VFX emitter billboards (Slice CC): world-space camera-facing quad verts (built
    # CPU-side by vfx::BuildBillboards) projected by FrameData.viewProj, carrying a per-vertex color +
    # corner UV; the fragment computes a procedural soft round sprite (like the decal) modulated by the
    # particle color, additively blended over the lit scene. DISTINCT from the gpu-particles fountain.
    "shaders/vfx.vert.hlsl:vs"
    "shaders/vfx.frag.hlsl:ps"
    # Immediate-mode debug-draw line pipeline (Slice W): world-space line verts projected by
    # FrameData.viewProj, flat per-vertex color. LINE_LIST topology.
    "shaders/debug_line.vert.hlsl:vs"
    "shaders/debug_line.frag.hlsl:ps"
    # Virtual-geometry meshlet viz (Slice DS): per-cluster flat-color pass. The vertex shader applies
    # the per-draw model matrix + a fixed-direction Lambert and forwards the pushed hashColor(cluster)
    # color; the fragment outputs it. Behind --meshlet-viz ONLY; reuses the existing draw + push-
    # constant surface (no new RHI), no existing shader/pipeline/golden touched.
    "shaders/meshlet_viz.vert.hlsl:vs"
    "shaders/meshlet_viz.frag.hlsl:ps"
    # SSAO chain (Slice Y): view-space normal+linear-depth g-buffer prepass (static + instanced
    # vertex variants + shared fragment), hemisphere-kernel AO pass, box blur, and composite/apply.
    "shaders/gbuffer.vert.hlsl:vs"
    "shaders/gbuffer_instanced.vert.hlsl:vs"
    "shaders/gbuffer.frag.hlsl:ps"
    "shaders/ssao.frag.hlsl:ps"
    "shaders/ssao_blur.frag.hlsl:ps"
    "shaders/ssao_composite.frag.hlsl:ps"
    # Screen-space reflections (Slice AH): reuses the SSAO g-buffer (view normal + linear depth);
    # ssr.frag ray-marches the depth buffer for on-screen reflections, ssr_composite blends +
    # tonemaps. Existing gbuffer/ssao/bloom shaders + their goldens untouched.
    "shaders/ssr.frag.hlsl:ps"
    "shaders/ssr_composite.frag.hlsl:ps"
    # Depth of field (Slice CG): reuses the SSAO/SSR g-buffer (view normal + linear depth) + the resolved
    # HDR scene. dof.frag computes the thin-lens circle of confusion per pixel (render/dof.h math,
    # mirrored in-shader) and gathers a Vogel-spiral disk weighted by the scatter-as-gather BlurWeight so
    # the focal plane stays sharp while fore/background blur, then tonemaps to the swapchain. Fullscreen
    # (reuses post.vert), behind --dof-shot ONLY; existing post/ssr/ssgi shaders + goldens untouched.
    "shaders/dof.frag.hlsl:ps"
    # Per-object + camera motion blur (Slice CN): reuses the SSAO/SSR/DoF g-buffer (view normal + linear
    # depth) + the resolved HDR scene. motion_blur.frag reconstructs each pixel's view position, computes
    # the screen-space velocity from the prev/cur view-proj (render/motion_blur.h math, mirrored in-shader),
    # clamps it to maxBlurPx and gathers N taps along it weighted by the depth-aware TapWeight so moving
    # content streaks while static content stays sharp, then tonemaps. Fullscreen (reuses post.vert), behind
    # --motionblur-shot ONLY; with zero motion the gather short-circuits to a byte-identical pass-through.
    # Existing post/ssr/ssgi/dof shaders + goldens untouched.
    "shaders/motion_blur.frag.hlsl:ps"
    # Order-Independent Transparency (Slice CO): Weighted Blended OIT (McGuire & Bavoil). oit_accum.vert
    # rasterizes the transparent geometry and passes the VIEW-space linear depth; oit_accum.frag writes the
    # premultiplied-color*Weight (additive ONE,ONE accum target, RGBA16F) and oit_revealage.frag writes the
    # alpha that multiplies the revealage target (oitRevealageBlend: dst*=(1-srcAlpha)); oit_resolve.frag
    # (fullscreen, reuses post.vert) reads accum+revealage and composites oit::ResolveOver over the opaque
    # scene. render/oit.h math, mirrored in-shader. Behind --oit-shot ONLY; transparent set rendered in two
    # permuted draw orders -> byte-identical resolve (order-independence proof). Existing shaders/goldens untouched.
    "shaders/oit_accum.vert.hlsl:vs"
    "shaders/oit_accum.frag.hlsl:ps"
    "shaders/oit_revealage.frag.hlsl:ps"
    "shaders/oit_resolve.frag.hlsl:ps"
    # Parallax Occlusion Mapping (Slice CP): pom.vert mirrors lit.vert but forwards heightScale+numSteps
    # in the material push constant .zw; pom.frag builds the orthonormal TBN, rotates the view dir into
    # tangent space, ray-marches the HEIGHT texture (steep parallax + binary refine, render/pom.h
    # ParallaxUV mirrored in-shader) to a parallax-offset uv, samples albedo+normal there, applies the
    # soft pom::SelfShadow toward the sun, and lights it with the SAME directional-sun + shadow + IBL
    # model as lit.frag. Behind --pom-shot ONLY; with heightScale==0 the march/self-shadow degenerate
    # to the base uv / 1.0 -> byte-identical to the plain normal-mapped lit render. Height bound via the
    # existing BindMaterialPBR slot (t5). Existing lit/gbuffer/oit shaders + goldens untouched.
    "shaders/pom.vert.hlsl:vs"
    "shaders/pom.frag.hlsl:ps"
    # Water rendering (Slice CF): water.vert displaces a flat NxN grid by the summed Gerstner waves at a
    # FIXED time (render/water.h math, mirrored in-shader); water.frag recomputes the analytic wave
    # normal and shades a fresnel blend of a SKY reflection (reusing SkyColor) and a REFRACTED scene
    # (sampling the HDR scene-color RT + G-buffer like SSR, tinted by depth) + a sun specular glint;
    # water_composite blends lerp(scene, water.rgb, water.a) + tonemaps. Existing shaders + goldens
    # untouched.
    "shaders/water.vert.hlsl:vs"
    "shaders/water.frag.hlsl:ps"
    "shaders/water_composite.frag.hlsl:ps"
    # Ground-Truth Ambient Occlusion (Slice CR): reuses the SSAO/SSR g-buffer (view normal + linear
    # depth). gtao.frag reconstructs the view position + normal (like ssao.frag), runs the render/gtao.h
    # horizon-search visibility integral (IntegrateArc + HorizonAngle + Visibility mirrored in-shader)
    # with a baked per-pixel slice rotation, and outputs the AO factor; the EXISTING ssao_composite.frag
    # MULTIPLIES the ambient term by it (reused unchanged). Fullscreen (reuses post.vert), behind
    # --gtao-shot ONLY; with radius==0 the horizon search raises no horizon -> AO==1 -> byte-identical to
    # the no-AO render. Existing ssao/ssr shaders + their goldens untouched.
    "shaders/gtao.frag.hlsl:ps"
    # Subsurface Scattering (Slice CZ): screen-space separable SSS (Jimenez 2015). sss_mask.frag (reuses
    # gbuffer.vert) renders ONLY the flagged subsurface objects into an "SSS G-buffer" RT carrying the
    # subsurface MASK in .r + the view-space linear depth in .w. sss_blur.frag is a fullscreen pass run
    # TWICE (horizontal then vertical, a uniform selects the axis; reuses post.vert) that gathers the
    # render/sss.h diffusion profile (DiffusionWeight x DepthFalloff x mask, mirrored in-shader) along the
    # axis so the lit color of subsurface pixels softens (the translucent terminator glow); non-subsurface
    # (mask 0) pixels pass through. Reuses the scene-color RT + the mask/depth G-buffer pair via
    # BindTexturePair (no new RHI seam), behind --sss-shot ONLY; with sssStrength==0 (or sssWidth==0) the
    # kernel collapses to the center tap -> byte-identical to the non-SSS lit render. Existing
    # post/ssr/ssgi/dof/gtao shaders + goldens untouched.
    "shaders/sss_mask.frag.hlsl:ps"
    "shaders/sss_blur.frag.hlsl:ps"
    # Color Grading (Slice DB): analytic lift/gamma/gain + ASC-CDL slope/offset/power + saturation,
    # applied POST-tonemap. color_grade.frag is a fullscreen pass (reuses post.vert) that reads the
    # resolved TONEMAPPED scene color (the engine's post.frag ACES+gamma LDR result rendered into an
    # intermediate RT) and applies render/color_grade.h's Grade (LiftGammaGain + ASC_CDL + Saturate,
    # mirrored in-shader) from a small GradeParams push constant, for a cinematic look. Behind
    # --colorgrade-shot ONLY; at the IDENTITY params (lift 0/gamma 1/gain 1/slope 1/offset 0/power 1/
    # saturation 1) the grade is a pure pass-through -> byte-identical to the ungraded tonemap render.
    # Existing post/ssr/ssgi/dof/gtao/sss shaders + goldens untouched.
    "shaders/color_grade.frag.hlsl:ps"
    # Contrast-Adaptive Sharpening (Slice DF; AMD FidelityFX CAS): cas.frag is a NEW fullscreen pass
    # (reuses post.vert) that reads the resolved TONEMAPPED SDR scene color (the engine's post.frag LDR
    # result rendered into an intermediate RT), gathers the cross neighborhood (center + up/down/left/
    # right via the texel size from GetDimensions), and applies render/cas.h's CasSharpen (CasWeight +
    # CasSharpen mirrored in-shader: out = (center + w*(up+down+left+right))/(1+4w) clamped to the
    # neighborhood [min,max], the no-ringing clamp) from a small `sharpness` push constant — an adaptive
    # edge sharpen. Behind --cas-shot ONLY; at sharpness 0 CasWeight==0 -> a pure pass-through ->
    # byte-identical to the unsharpened tonemap render. Existing post/ssr/ssgi/dof/gtao/sss/colorgrade
    # shaders + goldens untouched.
    "shaders/cas.frag.hlsl:ps"
    # Screen-Space Contact Shadows (Slice CT): reuses the SSAO/SSR/GTAO g-buffer (view normal + linear
    # depth). contact_shadows.frag reconstructs the view position (like gtao.frag), marches a SHORT ray
    # TOWARD the sun through the depth field (render/contact_shadows.h RayMarchShadow mirrored in-shader)
    # with a baked per-pixel dither, and outputs the single-channel contact-shadow factor; the NEW
    # lit_contactshadow.frag (a lit.frag VARIANT) samples that factor at the pixel's screen position and
    # MULTIPLIES the direct sun radiance by it (ambient/IBL/point lights untouched, like the CSM term).
    # Fullscreen factor pass (reuses post.vert), behind --contactshadow-shot ONLY; with maxDist==0 the
    # march takes no steps -> factor==1 -> byte-identical to the no-contact render. Existing lit/gtao/ssao
    # shaders + their goldens untouched.
    "shaders/contact_shadows.frag.hlsl:ps"
    "shaders/lit_contactshadow.frag.hlsl:ps"
    # DDGI GI Composite — DDGI Slice 5 (Slice DN): a SIBLING of lit.frag that adds a single-bounce DDGI
    # indirect-diffuse term as the LAST contribution. Per pixel it trilinearly blends the 8 nearest probes'
    # 3rd-order real-SH irradiance (NearestProbes + InterpolateSH + SHEvaluate copied VERBATIM from
    # engine/render/probe_gi.h + probe_sh.h, mad for every accumulation) toward the surface normal and adds
    # indirect*albedo*(1-metallic)*giStrength -> colored Cornell color bleed. The ProbeSH[] rides in a
    # fragment-stage storage buffer bound via the EXISTING usesLightClusters/BindLightClusters path (binding
    # 13/space3, single SSBO via dummies) — NO new RHI. giStrength==0 (or probeCount=0) -> the indirect term
    # is a literal +0.0 -> byte-identical to the no-GI render. Used ONLY by --ddgi-shot; lit.frag + its
    # goldens are untouched.
    "shaders/lit_ddgi.frag.hlsl:ps"
    # DDGI Visibility Slice 2 — Chebyshev occlusion-weighted GI composite (Slice DP): a SIBLING of
    # lit_ddgi.frag that weights each of the 8 nearest probes' SH contribution by a per-probe Chebyshev
    # (variance-shadow) visibility weight read from DO's per-probe distance-moment cube (DistDirToFaceUV +
    # SampleProbeMoments copied VERBATIM from engine/render/probe_dist.h), so a probe geometrically occluded
    # from a surface stops bleeding indirect light through the occluder (the DDGI light-leak fix). ProbeSH
    # binds at set-3 binding 13 (slot 0, as DN) + the distance moments at the next free slot binding 14 via
    # the EXISTING BindLightClusters path — NO new RHI. occlusionStrength==0 branches to the VERBATIM DN path
    # -> frame A byte-identical to the DN lit_ddgi render. Used ONLY by --ddgiocc-shot; lit_ddgi.frag +
    # lit.frag + their goldens untouched.
    "shaders/lit_ddgi_occ.frag.hlsl:ps"
    # DDGI MULTI-BOUNCE capture-with-GI fragment (Slice DR): a SIBLING of probe_bake.frag.hlsl (the DI
    # probe-radiance capture) with the DN lit_ddgi InterpolateIrradianceSH indirect term added LAST, so
    # the bounce-1 capture sees the 1st-bounce GI -> SH-encoding it yields SH1 (the 2nd bounce). SH0 binds
    # at set-3 binding 13 via the EXISTING usesLightClusters/BindLightClusters path; the grid params ride
    # the same 208-byte lit_ddgi FrameData UBO. Used ONLY by the bounce-1 capture in --ddgimb-shot;
    # probe_bake.frag + lit_ddgi.frag + lit.frag + their goldens stay byte-identical (sibling isolation).
    "shaders/probe_bake_gi.frag.hlsl:ps"
    # Screen-space global illumination (Slice BP): reuses the SSAO/SSR g-buffer (view normal + linear
    # depth) + the lit HDR scene. ssgi.frag traces K cosine-hemisphere rays per pixel and marches them
    # (the SAME view-space march + binary-search as ssr.frag), sampling the lit scene as indirect
    # diffuse radiance; ssgi_composite ADDS indirect*albedo over the scene + tonemaps. Existing
    # gbuffer/ssao/ssr/bloom shaders + their goldens untouched.
    "shaders/ssgi.frag.hlsl:ps"
    "shaders/ssgi_composite.frag.hlsl:ps"
    # SSGI spatial denoise (Slice BR): a depth+normal-guided BILATERAL blur of the SSGI indirect buffer
    # (same NxN fullscreen shape as ssao_blur, but each tap weighted by the G-buffer edge-stop) — smooths
    # the floor color-bleed pool while keeping the bleed crisp at edges. Inserted on the SSGI path before
    # ssgi_composite, behind --ssgi-denoise-shot ONLY; the raw --ssgi-shot path + ssgi.png stay unchanged.
    "shaders/ssgi_denoise.frag.hlsl:ps"
    # Temporal SSGI accumulation (Slice BV): a running-mean accumulation pass that folds N=8 jittered
    # SSGI indirect frames (each a golden-angle-rotated hemisphere kernel, via ssgi.frag's `frame` param)
    # into an HDR accumulation RT, then composites the mean like BP. Inserted on the --ssgi-temporal-shot
    # path ONLY; the raw --ssgi-shot + spatial --ssgi-denoise-shot paths + their goldens stay unchanged.
    "shaders/ssgi_accum.frag.hlsl:ps"
    # Volumetric fog / light shafts (Slice AJ): reuses the directional shadow map + SSAO g-buffer;
    # volumetric.frag ray-marches the view ray sampling the shadow map per step for in-scattering,
    # volumetric_composite adds it over the scene + tonemaps. Existing shaders + goldens untouched.
    "shaders/volumetric.frag.hlsl:ps"
    "shaders/volumetric_composite.frag.hlsl:ps"
    # Volumetric clouds (Slice CH): a raymarched cumulus LAYER in the sky dome. clouds.frag
    # reconstructs the view ray (like sky.frag), intersects the cloud SLAB, ray-marches density
    # (render/clouds.h math, mirrored in-shader) with a short secondary light-march (Beer-Lambert) +
    # a Henyey-Greenstein phase, masked to the sky background via the SSAO g-buffer depth;
    # clouds_composite blends lerp(scene, cloud.rgb, cloud.a) + tonemaps. DISTINCT from volumetric fog.
    # Existing sky/scene/volumetric shaders + goldens untouched (new path behind --clouds-shot).
    "shaders/clouds.frag.hlsl:ps"
    "shaders/clouds_composite.frag.hlsl:ps"
    # Cloud shadows on the ground (Slice CK): a VARIANT of lit.frag that ADDITIONALLY attenuates the
    # direct sun by the cloud-shadow transmittance (render/clouds.h CloudShadow, mirrored in-shader) —
    # the surface samples the SAME CH cloud field along the ray toward the sun, casting dappled cloud
    # shadows on the scene. Reuses lit.vert (push constant extended with the cloud slab/time/steps).
    # Used ONLY by --cloud-shadows-shot; lit.frag + its goldens are byte-identical (new path).
    "shaders/lit_cloudshadow.frag.hlsl:ps"
    # Reflection + irradiance probes (Slice AK): bake the local room into a cube atlas (reflection
    # block) + cosine-convolve an irradiance block, then shade with lit_probe sampling that atlas as
    # LOCAL GI. Reuses lit.vert + the CSM depth-only caster's vertex (probe-face VP in push const) and
    # post.vert for the blit/convolution fullscreen passes. Existing shaders + goldens untouched.
    "shaders/probe_bake.vert.hlsl:vs"
    "shaders/probe_bake.frag.hlsl:ps"
    "shaders/lit_probe.frag.hlsl:ps"
    "shaders/probe_blit.frag.hlsl:ps"
    "shaders/probe_irradiance.frag.hlsl:ps"
    # Box-Projected Cubemap Reflections (Slice DA): a reflective-surface variant of lit_probe.frag that
    # parallax-corrects the specular reflection direction to the local probe BOX (render/reflection_probe.h
    # ray-box exit + box-project, copied verbatim) before sampling the EXISTING reflection cubemap, so the
    # reflected room walls line up with the geometry. parallaxStrength=0 -> byte-identical to the plain
    # probe reflection. Reuses the probe bake/blit/irradiance pipeline + post.vert. Existing shaders +
    # goldens untouched.
    "shaders/reflprobe.frag.hlsl:ps"
    # Runtime cubemap-capture reflection probe (Slice DD): the reflective surface samples a CAPTURED
    # hardware TextureCube (the scene rendered into 6 cube faces from the probe center, via the new
    # additive cubemap-RT RHI interface) box-projected by DA's reflprobe math. The dynamic counterpart
    # to DA's static env reflection. Reuses probe_bake.{vert,frag} for the capture + post.vert.
    # Existing shaders + goldens untouched.
    "shaders/captureprobe.frag.hlsl:ps"
    # Planar reflections (Slice DE): the MIRROR-SURFACE fragment. Shades the flat reflective floor with
    # lit.frag's matte Cook-Torrance result, then blends lerp(matte, reflectionSample, reflectivity) —
    # the reflection RT being the scene rendered through the camera reflected across the mirror plane
    # (render::planar's ReflectionMatrix + oblique near-clip + winding flip), sampled at the pixel's own
    # screen UV. reflectivity=0 -> byte-identical to the matte render (the reflection RT is never read).
    # Reuses lit.vert (unchanged). Existing shaders + goldens untouched.
    "shaders/planar.frag.hlsl:ps"
    # Temporal anti-aliasing (Slice AP): the resolve pass — neighborhood-clamped history blend over the
    # current jittered HDR scene. Reuses post.vert (fullscreen triangle) + the existing RT/RGBA16F path;
    # the only new shader for the slice. Existing post/bloom/ssao/ssr shaders + goldens untouched.
    "shaders/taa_resolve.frag.hlsl:ps"
    # GPU-driven culling + indirect draw (Slice AR): a compute kernel frustum-culls a 1024-instance
    # grid and ORDER-compacts the survivors + writes the indirect draw-args; reuses lit_instanced.vert
    # for the indirect draw. Existing shaders + goldens untouched.
    "shaders/cull.comp.hlsl:cs"
    # Fully-GPU-driven-CULLED pass (Slice CD): a compute kernel frustum-culls the FULL per-draw list and
    # ORDER-compacts the survivors into the GpuDrivenPerDraw SSBO + the MDI command buffer + writes the
    # survivor count; the host reads the count back and issues ONE DrawIndexedMultiIndirect(count) over
    # lit_gpudriven.{vert,frag}. Composes AR (cull+compact) + BM (MDI) + BZ (bindless) + CB (combined
    # per-draw). Existing shaders + goldens untouched.
    "shaders/gpudriven_cull.comp.hlsl:cs"
    # Virtual-geometry per-CLUSTER frustum cull (Slice DT): a NEAR-COPY of gpudriven_cull.comp with the
    # (instance x cluster) record as the unit (a DS cluster is just an object with a per-cluster bounding
    # sphere). One thread/cluster-instance: SphereOutside (copied VERBATIM) -> ORDERED single-workgroup
    # prefix-sum compaction into the MdiCommand buffer + the compacted (model+color) per-draw SSBO +
    # survivor count. cluster_viz.vert is the MDI variant of meshlet_viz.vert (reads PerDraw[gl_DrawID]
    # model+color instead of a push constant); the shared meshlet_viz.frag is the flat-color pass. CPU
    # mirror: render/cluster_cull.h. NO new RHI; existing shaders + goldens untouched.
    "shaders/cluster_cull.comp.hlsl:cs"
    "shaders/cluster_viz.vert.hlsl:vs"
    # Virtual-geometry VISIBILITY BUFFER (Slice DW): the same DT/DV survivor cluster MDI draw, but the
    # fragment output is a SINGLE uint (clusterID<<7 | SV_PrimitiveID) into an R32_Uint render target
    # instead of a color — the Nanite-style (clusterID, triangleID) visibility buffer. visbuffer.vert is
    # a thin cluster_viz.vert variant (forwards the flat gl_DrawID clusterID, no Lambert/color);
    # visbuffer.frag packs the (clusterID, SV_PrimitiveID) pair. FIRST SV_PrimitiveID use in the repo
    # (Vulkan DXC -> SPIR-V PrimitiveId). NOT in the Metal MSL-gen list (DrawIndex); the Metal showcase
    # renders via the per-cluster bound path. CPU mirror: render/visbuffer.h. The only new RHI is the
    # additive Format::R32_Uint; existing shaders + goldens untouched.
    "shaders/visbuffer.vert.hlsl:vs"
    "shaders/visbuffer.frag.hlsl:ps"
    # Virtual-geometry DEFERRED MATERIAL RESOLVE (Slice DX): a FULLSCREEN pass (reuses post.vert) that
    # texel-fetches the DW R32_Uint vis-buffer (Texture2D<uint>.Load, no sampler), unpacks (clusterID,
    # triID), looks up the covering triangle via the cluster-meta/index/vertex SSBOs (bound through the
    # existing BindLightClusters set-3 path), computes a FLAT geometric normal + Lambert shade, and
    # outputs the lit image — the Nanite decoupled geometry->material resolve. VERBATIM CPU mirror:
    # render/visresolve.h. NO new RHI (the integer vis-buffer rides BindTexture; the SSBOs ride
    # BindLightClusters). On Metal the integer texture.read needs MSL 2.2 (the isolated --msl-version
    # 20200 in metal_headless/CMakeLists.txt, like visbuffer.frag).
    "shaders/visresolve.frag.hlsl:ps"
    # Virtual Shadow Maps Slice 1 (Slice VA): vsm_mark.comp marks which virtual clipmap PAGES the visible
    # receivers need — one thread per receiver world-sample, SelectClipmapLevel (the INTEGER threshold-
    # ladder, NO transcendental log2) + MarkPage (subtract/divide/floor) copied VERBATIM from
    # render/vsm.h, output resident[pageId]=1 (order-independent integer SET). markingEnabled push flag
    # false -> writes nothing. Pure compute over an integer SSBO + ReadBuffer (the froxel/cluster surface)
    # -> bit-exact GPU==CPU resident set, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/vsm_mark.comp.hlsl:cs"
    # Runtime Virtual Texturing Slice 1 (Slice VT1): vt_feedback.comp marks which virtual TEXTURE PAGES a
    # set of sample-requests need — one thread per request reads its HOST-SNAPPED integer (mip,px,py)
    # page coords (the host ran the UV->page quantization, GPU does ZERO float) + computes pageId via
    # PageId (mipPageOffset + py*pagesPerSide + px) copied VERBATIM from render/vt.h, output
    # feedback[pageId]=1 (order-independent integer SET). feedbackEnabled push flag false -> writes
    # nothing. Pure integer compute over an SSBO + ReadBuffer (the vsm_mark/froxel surface) -> bit-exact
    # GPU==CPU feedback set, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/vt_feedback.comp.hlsl:cs"
    # Runtime Virtual Texturing Slice 2 (Slice VT2): vt_alloc.comp is the SINGLE-THREAD physical tile-pool
    # allocator — one thread walks pageId 0..pageCount-1 ASCENDING maintaining nextTile, writing
    # gIndirection[pageId] = (resident && nextTile<cap) ? nextTile++ : 0xFFFFFFFF (kNoTile) — the DIRECT
    # mirror of render/vt.h::AllocatePhysicalTiles (and the VSM-VB AllocatePhysicalPages allocator). The
    # serial scan is required because allocation is inherently sequential (nextTile depends on prior
    # assignments); one thread is correct at this scale. allocEnabled=0 -> all kNoTile (no-op). Pure integer
    # SSBO compute + ReadBuffer -> the GPU indirection table is bit-exact GPU==CPU, proven memcmp. NO new
    # RHI; existing shaders + goldens untouched.
    "shaders/vt_alloc.comp.hlsl:cs"
    # Runtime Virtual Texturing Slice 3 (Slice VT3): vt_pagegen.comp GENERATES each allocated virtual page's
    # procedural CONTENT into its physical tile in the atlas — ONE thread per atlas texel (race-free parallel
    # MAP, NO atomics). Each thread finds its tile, reads the host-built reverse table gTilePageId to recover
    # the owning pageId, UnpackPageId's it -> (mip,px,py) (verbatim render/vt.h), and writes
    # PageTexel(pageId,mip,localXY) (the PURE-INTEGER per-texel generator copied VERBATIM from render/vt.h)
    # into gAtlas; unallocated tiles -> kAtlasClear. genEnabled=0 -> all kAtlasClear (no-op). Pure integer
    # SSBO compute + ReadBuffer -> the physical atlas is bit-exact GPU==CPU + cross-backend, proven memcmp.
    # NO new RHI; existing shaders + goldens untouched.
    "shaders/vt_pagegen.comp.hlsl:cs"
    # Runtime Virtual Texturing Slice 4 (Slice VT4): vt_sample.comp SAMPLES the virtual texture through the
    # virtual->physical INDIRECTION — ONE thread per VIRTUAL texel (race-free parallel MAP, NO atomics). Each
    # thread computes its pageId, reads gIndirection[pageId] -> the physical tile, offsets into the physical
    # atlas (PhysicalTileOrigin) and reads the texel (NEAREST integer read; non-resident -> kVtMiss);
    # SampleVirtualTexel copied VERBATIM from render/vt.h. sampleEnabled=0 -> all kVtMiss (no-op). Pure integer
    # SSBO compute + ReadBuffer -> the reconstructed virtual image is bit-exact GPU==CPU + cross-backend.
    # NO new RHI; existing shaders + goldens untouched.
    "shaders/vt_sample.comp.hlsl:cs"
    # Runtime Virtual Texturing Slice 5 (Slice VT5): vt_cachegen.comp is the CACHE-AWARE sibling of VT3's
    # vt_pagegen.comp (which is UNCHANGED) — ONE thread per atlas texel (race-free MAP, NO atomics). Each
    # thread finds its tile; if the tile is unallocated OR gNeedsGen[tileIndex]==0 (a cache hit) it RETURNS
    # without writing (the atlas SSBO PERSISTS across passes, so cached texels carry over byte-identical);
    # else it UnpackPageId's the owning pageId and writes PageTexelV(pageId,mip,localXY,gTileVersion[tile])
    # (the content-version-aware generator copied VERBATIM from render/vt.h; version 0 == VT3 PageTexel).
    # Pure integer SSBO compute + ReadBuffer -> the cached atlas is bit-exact GPU==CPU + cross-backend AND
    # byte-identical to a full regen, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/vt_cachegen.comp.hlsl:cs"
    # Nanite Software-Raster Slice 2 (Slice SW2): swraster.comp is the GPU COMPUTE software rasterizer —
    # ONE thread per cluster-triangle scan-converts the triangle into a w*h depth|id visibility-buffer SSBO
    # via InterlockedMin, running the int64 integer edge math + top-left fill rule + flat min-depth copied
    # VERBATIM from render/swraster.h over the HOST-SNAPPED integer ScreenVerts (ZERO GPU FP). The
    # commutative atomic-min => the GPU vis-buffer is bit-identical to the CPU swraster.h::RasterClusters
    # reference at EVERY pixel, full-frame, on any vendor. RWStructuredBuffer<uint> + InterlockedMin (the
    # existing InterlockedAdd family) + ReadBuffer -> NO new RHI; existing shaders + goldens untouched.
    "shaders/swraster.comp.hlsl:cs"
    # GPU Isosurface Meshing Slice 1 (Slice MC1): mc_classify.comp classifies each voxel CELL into its
    # 8-bit MARCHING-CUBES case index — ONE thread per cell decomposes its flat id -> (cx,cy,cz), gathers
    # the 8 cube-CORNER scalars from gField at the canonical kCornerOffset, runs CaseIndex (idx = Σ
    # (corner[i] > isovalue) << i) copied VERBATIM from render/mc.h, and writes gCases[cell]=caseIdx (an
    # order-independent per-cell integer write — race-free, NO atomics). classifyEnabled=0 -> writes 0.
    # Pure integer compute over an SSBO + ReadBuffer (the vsm_mark/vt_feedback surface) -> bit-exact
    # GPU==CPU case set + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/mc_classify.comp.hlsl:cs"
    # Deterministic Rigid-Body Fracture Slice 1 (Slice FR1, the BEACHHEAD of FLAGSHIP #14):
    # fract_classify.comp partitions a source-volume integer LATTICE into N fragment CELLS — ONE thread
    # per lattice SAMPLE decomposes its flat id -> (x,y,z), runs the nearest-seed loop (bestId = argmin_k
    # dx*dx+dy*dy+dz*dz over the M host seeds, STRICTLY-less lowest-index tie-break) copied VERBATIM from
    # engine/sim/fract.h::NearestSeed, and writes gCells[sample]=bestId (an order-independent per-sample
    # integer write — race-free, NO atomics). classifyEnabled=0 -> writes 0. PURE INT32 (small lattice
    # coords -> max d2 ~3.1M fits int32, NO int64) over an SSBO + ReadBuffer -> bit-exact GPU==CPU cell
    # set + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/fract_classify.comp.hlsl:cs"
    # Deterministic IK Control-Rig Slice 1 (Slice IK1, the BEACHHEAD of FLAGSHIP #32): ik_angle.comp
    # evaluates the host-baked Q16.16 ANGLE LUTs (acos/sin/cos) over a fixed sweep — one thread per sample
    # reads the uploaded tables + runs the PURE-INT32 bin index + fixed-point lerp copied VERBATIM from
    # engine/anim/ik.h, writing 3 Q16.16 results per sample. PURE INT32 -> MSL-native (in hf_gen_msl).
    # ReadBuffer reads the sweep, PROVEN BIT-EXACT vs the CPU ik.h lookups (memcmp, no tol). Behind
    # --ik1-angle-shot only; existing shaders + goldens untouched.
    "shaders/ik_angle.comp.hlsl:cs"
    # Deterministic IK Control-Rig Slice 2 (Slice IK2, the 2nd slice of FLAGSHIP #32): ik_twobone.comp
    # solves the law-of-cosines two-bone limb IK — ONE thread per target reads the host-baked acos/sin/cos
    # Q16.16 LUTs + runs engine/anim/ik.h::TwoBoneSolve VERBATIM, writing the two LOCAL bone FxQuats
    # (8 ints {qUpper.xyzw, qLower.xyzw}) per target. UNLIKE IK1's ik_angle.comp (pure int32, MSL-native),
    # the law-of-cosines reach d (FxLength) + the cosine ratios (fxdiv) are int64, so ik_twobone.comp is
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal
    # --ik2-twobone runs the CPU TwoBoneSolve (byte-identical by construction). ReadBuffer reads the quats,
    # PROVEN BIT-EXACT vs the CPU ik.h TwoBoneSolve (memcmp, no tol). Behind --ik2-twobone-shot only;
    # existing shaders + goldens untouched.
    "shaders/ik_twobone.comp.hlsl:cs"
    # Deterministic IK Control-Rig Slice 3 (Slice IK3, the 3rd slice of FLAGSHIP #32): ik_fabrik.comp
    # solves the FABRIK n-bone chain IK — ONE thread per chain/target runs engine/anim/ik.h::FabrikSolve
    # VERBATIM (forward+backward reaching for `iters`), writing the K solved joint positions (3 ints each)
    # per chain. LIKE ik_twobone.comp (and UNLIKE IK1's MSL-native ik_angle.comp), FxNormalize/FxLength are
    # int64, so ik_fabrik.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal
    # hf_gen_msl list; the Metal --ik3-fabrik runs the CPU FabrikSolve (byte-identical by construction).
    # ReadBuffer reads the positions, PROVEN BIT-EXACT vs the CPU ik.h FabrikSolve (memcmp, no tol). Behind
    # --ik3-fabrik-shot only; existing shaders + goldens untouched.
    "shaders/ik_fabrik.comp.hlsl:cs"
    # Deterministic IK Control-Rig Slice 4 (Slice IK4, the 4th slice of FLAGSHIP #32): ik_rig.comp is the
    # FK-pose -> IK-corrected palette BRIDGE — ONE thread per target runs the WHOLE engine/anim/ik.h::
    # SolveRigToTargets VERBATIM (FK the base local pose to world joint positions, FABRIK the chain's world
    # positions toward the target, convert the corrected world positions back to corrected LOCAL joint
    # rotations via LookAtRotation+FxQuatMul+QConj), writing the corrected per-joint LOCAL FxQuats
    # (kIkMaxJoints*4 ints) per target. LIKE ik_fabrik.comp / ik_twobone.comp (and UNLIKE IK1's MSL-native
    # ik_angle.comp), the FK/FABRIK/quat math is int64, so ik_rig.comp is VULKAN-SPIR-V-ONLY (DXC compiles
    # int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal --ik4-rig runs the CPU
    # SolveRigToTargets (byte-identical by construction). ReadBuffer reads the corrected pose, PROVEN
    # BIT-EXACT vs the CPU ik.h SolveRigToTargets (memcmp, no tol). Behind --ik4-rig-shot only; existing
    # shaders + goldens untouched.
    "shaders/ik_rig.comp.hlsl:cs"
    # Deterministic Rigid-Body Fracture Slice 2 (Slice FR2): FRAGMENT EXTRACTION — the GR2/FL2
    # count->scan->emit CSR over FR1's cellId[] (cells == seedCount) + a per-fragment reduction. PURE
    # INT32 -> MSL-native, a TRUE GPU pass on both backends. fract_emit_count (ONE thread per sample:
    # InterlockedAdd into its cell's counter), fract_emit_scan ([numthreads(1,1,1)] exclusive prefix-sum
    # + the CL2/MC3 stream-compaction -> fragStart/cellToFragment/fragmentToCell/fragCount),
    # fract_emit_emit (SINGLE-THREAD ascending-sample scatter into fragSamples — single-thread for the
    # within-cell ascending order), fract_reduce (ONE thread per FRAGMENT: reduce its CSR slice ->
    # centroid/AABB/boundRadiusSq/boundRadius/volume/cellId/invMass, per-fragment disjoint, NO atomics).
    # Bit-exact GPU==CPU vs fract.h::ExtractFragments (memcmp, NO tol). NO new RHI; FR1 shaders untouched.
    "shaders/fract_emit_count.comp.hlsl:cs"
    "shaders/fract_emit_scan.comp.hlsl:cs"
    "shaders/fract_emit_emit.comp.hlsl:cs"
    "shaders/fract_reduce.comp.hlsl:cs"
    # Deterministic Rigid-Body Fracture Slice 3 (Slice FR3): BONDED-CLUSTER BREAK — THE NEW PHYSICS.
    # fract_break.comp runs the int64 Jacobi load-diffusion break VERBATIM from fract.h::ApplyImpactBreak:
    # inject the impact load at one fragment, K Jacobi iters (read-start load / write a SEPARATE delta
    # buffer / apply — race-free, the FL4/GR3 discipline) accumulate per-bond loadAccum, then SEVER iff
    # loadAccum > fxmul(kBreakThreshold, faceArea<<kFrac). Because the load + transmit + threshold use
    # int64, fract_break.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal
    # hf_gen_msl list; the Metal --fract-break runs the CPU ApplyImpactBreak (byte-identical by
    # construction, the grain_contact_dp.comp/cloth_solve.comp convention). The bond graph is host-built
    # (BuildFractBonds, pure int32) + uploaded; the GPU-proven pass is this int64 break (memcmp the per-bond
    # {loadAccum, severed} vs the CPU). NO new RHI; FR1/FR2 shaders + goldens untouched (FR3 is additive).
    "shaders/fract_break.comp.hlsl:cs"
    # Deterministic Fixed-Point Physics Slice 1 (Slice FPX1, the BEACHHEAD of FLAGSHIP #6):
    # fpx_integrate.comp runs the Q16.16 semi-implicit-Euler INTEGRATOR — ONE thread per body runs
    # `steps` iterations of (vel += gravity*dt; pos += vel*dt; ground floor-clamp) on its OWN body, the
    # fxmul ((int64)a*b >> 16, the swraster int64 pattern) + integrate + clamp copied VERBATIM from
    # engine/sim/fpx.h::IntegrateBody, and writes gBodies[i] back (each body independent -> race-free,
    # NO atomics). integrateEnabled=0 -> write the input back UNCHANGED. Pure integer SSBO compute +
    # ReadBuffer -> bit-exact GPU==CPU body array + cross-backend, proven memcmp. NO new RHI; existing
    # shaders + goldens untouched.
    "shaders/fpx_integrate.comp.hlsl:cs"
    # Hardware Ray Tracing Slice 1 (Slice RT1, the BEACHHEAD of FLAGSHIP #28: HARDWARE RAY TRACING): the GPU
    # TWIN of the deterministic Q16.16 SW reference ray tracer. rt_trace.comp runs ONE thread per pixel —
    # generates the integer primary ray, brute-forces every primitive (NO accel-struct cull), keeps the
    # closest by the total order (t, primIndex), and writes the integer Lambert RGBA8, the
    # TraceClosest/ShadeHitInt/PrimaryRay math (incl. int64 fxmul/fxdiv + the FxISqrt digit loop) copied
    # VERBATIM from engine/render/rtrace.h. Pure integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU
    # image + cross-backend, proven memcmp. It is VULKAN-SPIR-V-ONLY (int64 -> DXC compiles it; glslc
    # cannot) -> NOT in the Metal hf_gen_msl list (see metal_headless/CMakeLists.txt); on Metal the
    # --rt1-trace showcase runs the CPU rtrace::RenderScene (byte-identical by construction). NO new RHI on
    # the trace path (the accel-struct RHI seam is additive + INERT in RT1); existing shaders/goldens untouched.
    "shaders/rt_trace.comp.hlsl:cs"
    # Hardware Ray Tracing Slice 2 (Slice RT2, FLAGSHIP #28): rt_query.comp is the DETERMINISTIC HW INLINE
    # RAY QUERY kernel — ONE thread per pixel opens a RayQuery over the TLAS (binding 4), DRAINS every
    # candidate procedural primitive via Proceed() (NO commit, NO tMax shrink), runs the FROZEN fx Q16.16
    # IntersectSphere/IntersectAabb on each candidate, folds into the closest by the total order
    # (t, primIndex), and writes the integer Lambert RGBA8 — bit-identical to rt_trace.comp (SW GPU) and the
    # CPU rtrace::RenderScene. The HW BVH is ONLY a candidate-AABB generator (margin-inflated -> a strict
    # superset); OUR fx math owns the closest hit -> vendor/BVH-order-independent, hence byte-identical to
    # the SW no-cull reference. The :csrq stage compiles it cs_6_5 + SPV_KHR_ray_query (HLSL RayQuery -> DXC
    # SPIR-V RayQueryKHR). It is VULKAN-SPIR-V-ONLY (RayQuery + int64 -> glslc/spirv-cross can't lower to
    # MSL) -> NOT in the Metal hf_gen_msl list; on Metal --rt2-query runs the CPU rtrace::RenderScene
    # (byte-identical by construction). The Vulkan side carries the HW==SW-GPU==CPU memcmp proof.
    "shaders/rt_query.comp.hlsl:csrq"
    # Hardware Ray Tracing Slice 3 (Slice RT3, FLAGSHIP #28): rt_shadow.comp is the DETERMINISTIC RT HARD
    # SHADOWS kernel — ONE thread per pixel runs the rt_query.comp PRIMARY RayQuery body VERBATIM (closest
    # hit), then casts a SECOND RayQuery (the SHADOW ray from hit.pos + normal*kRtShadowEps toward the light)
    # draining candidates + running OUR fx IntersectSphere/IntersectAabb, setting occluded if ANY yields
    # t > kRtShadowMinT (an order-independent boolean OR — the shadow query MAY early-out), and writes
    # ShadeHitShadowed (the integer Lambert with the diffuse GATED by occlusion) — bit-identical to the CPU
    # rtrace::RenderSceneShadowed. The :csrq stage compiles it cs_6_5 + SPV_KHR_ray_query (HLSL RayQuery ->
    # DXC SPIR-V RayQueryKHR). It is VULKAN-SPIR-V-ONLY (RayQuery + int64 -> glslc/spirv-cross can't lower to
    # MSL) -> NOT in the Metal hf_gen_msl list; on Metal --rt3-shadow runs the CPU rtrace::RenderSceneShadowed
    # (byte-identical by construction). The Vulkan side carries the HW==CPU memcmp proof.
    "shaders/rt_shadow.comp.hlsl:csrq"
    # Hardware Ray Tracing Slice 4 (Slice RT4, FLAGSHIP #28): rt_reflect.comp is the DETERMINISTIC RT MIRROR
    # REFLECTIONS kernel — ONE thread per pixel runs the rt_shadow.comp PRIMARY closest-hit + PRIMARY shadow
    # body VERBATIM (the shadowed color cS), then if the primary hit is REFLECTIVE casts a REFLECTION ray (the
    # view dir mirrored about the fx normal, a SECOND closest-hit RayQuery drain), shades the reflected hit with
    # its OWN shadow ray (the RT3 shadowed color cR; a reflection MISS -> background), and BLENDS cS/cR on the
    # UNPACKED RGBA8 channels by the integer reflectivity (final = surface*(1-refl) + reflected*refl) — bit-
    # identical to the CPU rtrace::RenderSceneReflected. The :csrq stage compiles it cs_6_5 + SPV_KHR_ray_query
    # (HLSL RayQuery -> DXC SPIR-V RayQueryKHR). It is VULKAN-SPIR-V-ONLY (RayQuery + int64 -> glslc/spirv-cross
    # can't lower to MSL) -> NOT in the Metal hf_gen_msl list; on Metal --rt4-reflect runs the CPU
    # rtrace::RenderSceneReflected (byte-identical by construction). The Vulkan side carries the HW==CPU memcmp.
    "shaders/rt_reflect.comp.hlsl:csrq"
    # Issue #34: rt_reflect_graphics.frag is the GRAPHICS (fragment) twin of rt_reflect.comp — a ps_6_5
    # RayQuery fragment shader (fullscreen pass, reuses post.vert) that traces the SAME DETERMINISTIC RT
    # MIRROR REFLECTIONS body VERBATIM from a RASTER pass, proving a fragment shader can ray-query the
    # accel structure (the TLAS bound to the GRAPHICS pipeline via BindAccelStructure). ATTEMPT-2 data
    # binding: spheres/aabbs at cluster bindings 14/15, params via the frame UBO (NEVER the broken
    # graphics cluster binding 13). The :psrq stage compiles it ps_6_5 + SPV_KHR_ray_query (HLSL RayQuery
    # -> DXC SPIR-V RayQueryKHR). VULKAN-SPIR-V-ONLY (RayQuery + int64) -> NOT in the Metal MSL list; on
    # Metal --rt-reflect-graphics runs the CPU rtrace::RenderSceneReflected (byte-identical by construction).
    "shaders/rt_reflect_graphics.frag.hlsl:psrq"
    # Hardware Ray Tracing Slice 6 (Slice RT6, FLAGSHIP #28, the MONEY-SHOT CAPSTONE): rt_hero.comp is the LIT
    # HERO kernel — it COPIES the rt_reflect.comp body VERBATIM (primary closest-hit + primary shadow + the
    # one-bounce reflection + reflected shadow + the integer blend) with the ONE change that every MISS color
    # (the primary miss AND the reflection-ray miss) is the integer SkyGradient(missDir) (a graded
    # horizon->zenith sky) instead of the flat background — the EXACT integer lerp of rtrace.h::SkyGradient, so
    # the HW hero image is bit-identical to the CPU rtrace::RenderSceneHero. The :csrq stage compiles it
    # cs_6_5 + SPV_KHR_ray_query (HLSL RayQuery -> DXC SPIR-V RayQueryKHR). It is VULKAN-SPIR-V-ONLY (RayQuery +
    # int64 -> glslc/spirv-cross can't lower to MSL) -> NOT in the Metal hf_gen_msl list; on Metal --rt6-hero
    # runs the CPU rtrace::RenderSceneHero (byte-identical by construction). The Vulkan side carries HW==CPU.
    "shaders/rt_hero.comp.hlsl:csrq"
    # Deterministic Lumen-class GI Slice 1 (Slice GI1, the BEACHHEAD of FLAGSHIP #29): gi_probe_trace.comp is
    # the DETERMINISTIC INTEGER RT PROBE-TRACE kernel — ONE thread per PROBE loops the 16 baked Fibonacci-sphere
    # directions, opens a RayQuery over the TLAS (binding 4) draining every candidate procedural primitive (the
    # rt_query.comp contract VERBATIM), runs the FROZEN fx Q16.16 IntersectSphere/IntersectAabb + (t,primIndex)
    # min for the closest hit, casts a SECOND RayQuery (the shadow ray) for occlusion, shades with the inline
    # ShadeHitShadowed (the integer Lambert with the diffuse GATED by the shadow), and writes the unpacked
    # GiRadiance to gRadiance[probe*16 + dir] — bit-identical to the CPU gi::TraceProbeRays. The :csrq stage
    # compiles it cs_6_5 + SPV_KHR_ray_query (HLSL RayQuery -> DXC SPIR-V RayQueryKHR). It is VULKAN-SPIR-V-ONLY
    # (RayQuery + int64 -> glslc/spirv-cross can't lower to MSL) -> NOT in the Metal hf_gen_msl list; on Metal
    # --gi1-probe runs the CPU gi::TraceProbeRays (byte-identical by construction). The Vulkan side carries the
    # HW==CPU radiance memcmp proof.
    "shaders/gi_probe_trace.comp.hlsl:csrq"
    # Deterministic Lumen-class GI Slice 2 (Slice GI2, THE CRUX of FLAGSHIP #29): gi_sh_encode.comp is the
    # DETERMINISTIC INTEGER 3rd-order SH ENCODE kernel — ONE thread per PROBE folds its 16 GI1 GiRadiance rays
    # into a 9x3 FxProbeSH via an int64 (Q32.32) accumulator (acc += (int64)rad.ch * (int64)kGiSHBasis[r][i]),
    # quantized to Q16.16 EXACTLY ONCE at the final /N >> kFrac normalize — bit-identical to the CPU
    # gi::FxSHEncodeProbe / EncodeAllProbes (the GPU SH SSBO == CPU SH memcmp proof). NO RayQuery / NO accel
    # structure (pure-integer compute over the GI1 radiance buffer), so the :cs stage compiles it cs_6_x; but
    # the int64 accumulator (a single radiance*basis Q32.32 product overflows int32) makes it VULKAN-SPIR-V-ONLY
    # (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; on Metal the --gi2-shencode
    # showcase runs the CPU gi::EncodeAllProbes (byte-identical by construction, the grain_integrate.comp
    # convention). Pure integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU SH array + cross-backend.
    "shaders/gi_sh_encode.comp.hlsl:cs"
    # Deterministic Lumen-class GI Slice 3 (Slice GI3, the CONTINUOUS IRRADIANCE FIELD of FLAGSHIP #29):
    # gi_interp.comp is the DETERMINISTIC INTEGER TRILINEAR SH INTERPOLATION kernel — ONE thread per QUERY
    # POINT runs FxInterpolateIrradiance VERBATIM: FxNearestProbes (floor-cell + 8-corner partition-of-unity
    # trilinear weights, each axis wlo+whi == kOne EXACTLY, the triple products narrowed >> 2*kFrac with the
    # LAST corner taking the exact leftover so Σw == kOne) -> FxInterpolateSH (the int64 8-corner SH blend,
    # quantize ONCE >> kFrac) -> FxSHEvaluate (the int64 cosine-lobe reconstruction) — bit-identical to the
    # CPU gi::InterpolateField (the GPU irradiance == CPU irradiance memcmp proof). NO RayQuery / NO accel
    # structure (pure-integer compute over the GI2 SH buffer), so the :cs stage compiles it cs_6_x; but the
    # int64 SH-blend accumulator (a single weight*coeff Q32.32 product overflows int32) makes it
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; on Metal the
    # --gi3-interp showcase runs the CPU gi::InterpolateField (byte-identical by construction, the
    # gi_sh_encode.comp convention). Pure integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU irradiance.
    "shaders/gi_interp.comp.hlsl:cs"
    # Deterministic Lumen-class GI Slice 4 (Slice GI4, INTEGER MULTI-BOUNCE FEEDBACK of FLAGSHIP #29):
    # gi_bounce.comp is the DETERMINISTIC INTEGER MULTI-BOUNCE re-trace kernel — ONE thread per PROBE loops the
    # 16 baked Fibonacci-sphere directions, a closest-hit RayQuery (the gi_probe_trace.comp candidate-drain
    # VERBATIM) + a shadow RayQuery, and — the ONE GI4 addition — adds an INDIRECT term to each hit's shade:
    # the GI3 FxInterpolateIrradiance of the bound PREVIOUS SH buffer (gPrevSH, binding 4) at the hit point
    # (the trilinear 8-corner blend + FxSHEvaluate copied from gi_interp.comp VERBATIM). The host drives the
    # K<=3 iteration loop (dispatch gi_bounce.comp reading SH_{k-1}, then gi_sh_encode.comp to make SH_k); a
    # zeroed gPrevSH at k=1 -> the indirect is a literal +0 -> byte-identical to gi_probe_trace.comp (the no-op
    # contract). The result is bit-identical to the CPU gi::BounceProbes one-iteration re-trace. The :csrq
    # stage compiles it cs_6_5 + SPV_KHR_ray_query (HLSL RayQuery -> DXC SPIR-V RayQueryKHR). It is
    # VULKAN-SPIR-V-ONLY (RayQuery + int64 SH-blend -> glslc/spirv-cross can't lower to MSL) -> NOT in the
    # Metal hf_gen_msl list; on Metal --gi4-bounce runs the CPU gi::BounceProbes (byte-identical by
    # construction). The Vulkan side carries the HW==CPU final-SH memcmp proof.
    "shaders/gi_bounce.comp.hlsl:csrq"
    # Deterministic Lumen-class GI Slice 5 (Slice GI5, INTEGER CHEBYSHEV OCCLUSION WEIGHTING — the probe-volume
    # LIGHT-LEAK FIX of FLAGSHIP #29): gi_occ.comp is the DETERMINISTIC INTEGER OCCLUSION-WEIGHTED interpolation
    # kernel — ONE thread per QUERY POINT runs FxInterpolateIrradianceOcc VERBATIM: the GI3 trilinear blend, but
    # each of the 8 corner weights is scaled by the variance-shadow CHEBYSHEV VISIBILITY of that corner probe
    # (from the per-probe distance moments gMoments — gi::FxProbeMoments, the RT-trace-t by-product the CPU
    # FxProbeMoments_All computes + uploads, NO GPU distance pass) and the 8 weights are RE-NORMALIZED to Σ ==
    # kOne -> FxInterpolateSH (int64 8-corner blend) -> FxSHEvaluate — bit-identical to the CPU
    # gi::InterpolateFieldOcc (the GPU==CPU irradiance memcmp proof; occStrength==0 -> byte-identical to the GI3
    # unoccluded blend, the no-op contract). NO RayQuery / NO accel structure (pure-integer compute over the GI2
    # SH + GI5 moments buffers), so the :cs stage compiles it cs_6_x; but the int64 SH-blend + the int64
    # Chebyshev fxdiv/t² make it VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal
    # hf_gen_msl list; on Metal the --gi5-occlusion showcase runs the CPU gi::InterpolateFieldOcc (byte-identical
    # by construction, the gi_interp.comp convention). Pure integer SSBO compute + ReadBuffer -> bit-exact.
    "shaders/gi_occ.comp.hlsl:cs"
    # Deterministic Lumen-class GI Slice 6 (Slice GI6, THE LIT GI HERO CAPSTONE — the FLAGSHIP #29 MONEY-SHOT):
    # gi_hero.comp is the lit GI hero kernel — ONE thread per pixel copies the RT6 rt_hero.comp body VERBATIM
    # (primary closest-hit + RT3 hard shadow + RT4 one-bounce reflection + the integer blend + the graded
    # SkyGradient miss) and ADDS a GI INDIRECT term at each primary hit: albedo(hit) ⊙
    # FxInterpolateIrradianceOcc(grid, gSH, gMoments, hit.pos, hit.normal, occStrength) × giStrength (the
    # GI3/GI5 trilinear+Chebyshev SH blend + FxSHEvaluate copied from gi_occ.comp VERBATIM, reading the bound
    # gSH + gMoments buffers). Bit-identical to the CPU gi::RenderSceneGI (the HW==CPU image memcmp proof;
    # giStrength==0 -> a literal +0 indirect -> byte-identical to the no-GI RT6 hero, the no-op contract). The
    # :csrq stage compiles it cs_6_5 + SPV_KHR_ray_query (HLSL RayQuery -> DXC SPIR-V RayQueryKHR). It is
    # VULKAN-SPIR-V-ONLY (RayQuery + the int64 SH-blend/Chebyshev -> glslc/spirv-cross can't lower to MSL) ->
    # NOT in the Metal hf_gen_msl list; on Metal --gi6-hero runs the CPU gi::RenderSceneGI (byte-identical by
    # construction). The Vulkan side carries the HW==CPU hero-image memcmp proof.
    "shaders/gi_hero.comp.hlsl:csrq"
    # Deterministic GPU Crowds Slice 1 (Slice BD1, the BEACHHEAD of FLAGSHIP #18: deterministic GPU crowds —
    # boids + steering + path-following): boids_steer.comp runs the Q16.16 Reynolds STEERING tick — ONE thread
    # per AGENT reads the FROZEN input buffer (gAgentsIn), computes SteerSeek (proportional pull toward the
    # shared target) + a brute-force all-pairs SteerSeparation (accumulate raw integer (a.pos-o.pos) deltas for
    # neighbors within sepRadius², skip self by index), per-axis clamps the force, integrates velocity
    # (per-axis clamped) + position, and writes agent i into gAgentsOut — the seek + separation + clamps +
    # integrate copied VERBATIM from engine/sim/boids.h::StepBoids. JACOBI/PING-PONG (read gAgentsIn / write
    # gAgentsOut -> race-free, NO atomics, order-independent; the host ping-pongs the buffers between steps).
    # Like fpx_integrate.comp / grain_integrate.comp this is int64 (the gain*offset + force*dt + vel*dt fxmul
    # over Q16.16 overflows int32; the separation d²<sepRadius² compare is also int64) -> VULKAN-SPIR-V-ONLY
    # (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal --boids-steer showcase
    # runs the CPU boids::StepBoids (byte-identical by construction, the fpx_integrate.comp convention). Pure
    # integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU agent array + cross-backend, proven memcmp. NO
    # new RHI; existing shaders + goldens untouched.
    "shaders/boids_steer.comp.hlsl:cs"
    # Deterministic GPU Crowds Slice 2 (Slice BD2, the 2nd slice of FLAGSHIP #18): the GRID-HASH NEIGHBOR LIST
    # meshed by SIX pure-INT32 count->scan->emit passes (the GR2 grain_cell/grain_neighbor twin, applied to
    # agents). The CELL TABLE: boids_cell_count.comp (ONE thread per agent: InterlockedAdd into its cell's
    # counter — the cell id is BoidsCellOf = FloorDiv per axis at cell-size radius, the bounded-dense-grid
    # linearization, VERBATIM engine/sim/boids.h), boids_cell_scan.comp (SINGLE-THREAD [numthreads(1,1,1)]
    # exclusive prefix-sum -> cellStart CSR + sentinel), boids_cell_emit.comp (SINGLE-THREAD ascending-agent
    # scatter into cellAgents — single-thread because the within-cell order must be ascending agent index, NOT
    # GPU-scheduling order). The NEIGHBOR LIST: boids_neighbor_count.comp (ONE thread per agent: count
    # candidates in its 27-cell stencil passing the per-axis |dx|<radius reject — BoidsNeighborAccept VERBATIM),
    # boids_neighbor_scan.comp (SINGLE-THREAD exclusive prefix-sum -> neighborStart), boids_neighbor_emit.comp
    # (ONE thread per agent emits its candidates into its DISJOINT neighbors[] slice, NO atomics — the
    # per-agent-disjoint pattern). ALL PURE INT32 (FloorDiv integer divide + per-axis abs-compare, NO
    # int64/fxmul/sqrt — the box is the perception neighborhood, the int64 dist² compare AVOIDED to stay
    # MSL-native), so UNLIKE BD1's boids_steer these MSL-GENERATE NATIVELY (they ARE in the Metal hf_gen_msl
    # list) and Metal dispatches the GPU passes — a TRUE GPU pass on BOTH backends, the strongest cross-vendor
    # proof. ReadBuffer reads the cell table + neighbor list, PROVEN BIT-EXACT vs the CPU
    # boids.h::BuildBoidsCellTable + BuildBoidsNeighborList (memcmp). NO new RHI; BD1's boids_steer + goldens
    # untouched.
    "shaders/boids_cell_count.comp.hlsl:cs"
    "shaders/boids_cell_scan.comp.hlsl:cs"
    "shaders/boids_cell_emit.comp.hlsl:cs"
    "shaders/boids_neighbor_count.comp.hlsl:cs"
    "shaders/boids_neighbor_scan.comp.hlsl:cs"
    "shaders/boids_neighbor_emit.comp.hlsl:cs"
    # Deterministic GPU Crowds Slice 3 (Slice BD3, the 3rd slice of FLAGSHIP #18): boids_flock.comp runs THE
    # FULL FLOCK STEP — ONE thread per AGENT accumulates the THREE Reynolds rules over agent i's BD2 NEIGHBOR
    # LIST (separation Σ(pos_i-pos_j)*sepGain + alignment ((Σvel_j)/count - vel_i)*alignGain + cohesion
    # ((Σpos_j)/count - pos_i)*cohGain; the means are INTEGER DIVIDES by the neighbor count, count==0 guard),
    # adds the optional seek + gravity, per-axis clamps the force, integrates velocity (per-axis clamped) +
    # position, and writes agent i into gAgentsOut — the 3 rules + means + clamps + integrate copied VERBATIM
    # from engine/sim/boids.h::AccumFlock + StepFlock. JACOBI/PING-PONG (read gAgentsIn + the neighbor list
    # built over THAT frozen snapshot / write gAgentsOut -> race-free, NO atomics; the host rebuilds the BD2
    # neighbor list each tick on the new positions). Like boids_steer.comp this is int64 (the gain*delta +
    # force*dt + vel*dt fxmul over Q16.16 overflows int32) -> VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc
    # cannot) -> NOT in the Metal hf_gen_msl list; the Metal --boids-flock showcase runs the CPU
    # boids::StepFlock (byte-identical by construction), while the MSL-native BD2 grid passes stay a true GPU
    # pass on both backends. Pure integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU + cross-backend,
    # proven memcmp. NO new RHI; BD1/BD2 shaders + goldens untouched (BD3 additive).
    "shaders/boids_flock.comp.hlsl:cs"
    # Deterministic GPU Cloth Slice 1 (Slice CL1, the BEACHHEAD of FLAGSHIP #8: deterministic GPU cloth):
    # cloth_integrate.comp runs the Q16.16 semi-implicit-Euler PARTICLE LATTICE INTEGRATOR — ONE thread per
    # particle runs `steps` iterations of (vel += gravity*dt; prev = pos; pos += vel*dt; ground floor-clamp)
    # on its OWN particle, the fxmul ((int64)a*b >> 16, the swraster/fpx int64 pattern) + integrate + prev-
    # snap + clamp copied VERBATIM from engine/sim/cloth.h::IntegrateParticle, and writes gParticles[i] back
    # (each particle independent in CL1 -> race-free, NO atomics; PINNED particles untouched). integrateEnabled
    # =0 -> write the input back UNCHANGED. Like fpx_integrate.comp this is int64 (gravity*dt over Q16.16
    # overflows int32) -> VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl
    # list; the Metal --cloth-integrate showcase runs the CPU cloth::IntegrateParticles (byte-identical by
    # construction, the fpx_integrate.comp convention). Pure integer SSBO compute + ReadBuffer -> bit-exact
    # GPU==CPU particle array + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/cloth_integrate.comp.hlsl:cs"
    # Deterministic GPU Fluid Slice 1 (Slice FL1, the BEACHHEAD of FLAGSHIP #9: deterministic GPU fluid):
    # fluid_integrate.comp runs the Q16.16 semi-implicit-Euler PARTICLE POOL INTEGRATOR — ONE thread per
    # particle runs `steps` iterations of (vel += gravity*dt; prev = pos; pos += vel*dt; ground floor-clamp)
    # on its OWN particle, the fxmul ((int64)a*b >> 16, the swraster/fpx int64 pattern) + integrate + prev-
    # snap + clamp copied VERBATIM from engine/sim/fluid.h::IntegrateFluidParticle, and writes gParticles[i]
    # back (each particle independent in FL1 -> race-free, NO atomics). integrateEnabled=0 -> write the input
    # back UNCHANGED. Like cloth_integrate.comp / fpx_integrate.comp this is int64 (gravity*dt over Q16.16
    # overflows int32) -> VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl
    # list; the Metal --fluid-integrate showcase runs the CPU fluid::IntegrateFluid (byte-identical by
    # construction, the cloth_integrate.comp convention). Pure integer SSBO compute + ReadBuffer -> bit-exact
    # GPU==CPU particle array + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/fluid_integrate.comp.hlsl:cs"
    # Deterministic GPU Granular/Sand Slice 1 (Slice GR1, the BEACHHEAD of FLAGSHIP #10: deterministic GPU
    # granular/sand): grain_integrate.comp runs the Q16.16 semi-implicit-Euler GRAIN POOL INTEGRATOR — ONE
    # thread per grain runs `steps` iterations of (vel += gravity*dt; prev = pos; pos += vel*dt; RADIUS-AWARE
    # ground rest: pos.y < groundY + radius -> groundY + radius) on its OWN grain, the fxmul ((int64)a*b >> 16,
    # the swraster/fpx int64 pattern) + integrate + prev-snap + radius-aware clamp copied VERBATIM from
    # engine/sim/grain.h::IntegrateGrainParticle, and writes gGrains[i] back (each grain independent in GR1 ->
    # race-free, NO atomics). The GrainParticle adds a first-class `radius` field vs FL1's FluidParticle ->
    # 48-byte std430 packing. integrateEnabled=0 -> write the input back UNCHANGED. Like fluid_integrate.comp /
    # cloth_integrate.comp / fpx_integrate.comp this is int64 (gravity*dt over Q16.16 overflows int32) ->
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal
    # --grain-integrate showcase runs the CPU grain::IntegrateGrains (byte-identical by construction, the
    # fluid_integrate.comp convention). Pure integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU grain
    # array + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/grain_integrate.comp.hlsl:cs"
    # Deterministic GPU Particles Slice 1 (Slice PT1, the BEACHHEAD of FLAGSHIP #19: deterministic GPU
    # particles): particles_integrate.comp runs the Q16.16 semi-implicit-Euler PARTICLE POOL INTEGRATOR — ONE
    # thread per particle slot runs (vel += gravity*dt; vel -= vel*dragK; pos += vel*dt; age += dt; age >=
    # lifetime -> clear ALIVE) on its OWN slot, the fxmul ((int64)a*b >> 16, the swraster/fpx int64 pattern) +
    # integrate + drag + age copied VERBATIM from engine/sim/particles.h::IntegrateParticle, and writes
    # gParticles[i] back (each particle independent in PT1 -> race-free, NO atomics). The EMIT + RECYCLE are
    # HOST-side (single-thread, between integrate dispatches — the deterministic free-list); the GPU dispatch is
    # ONLY the per-particle integrate. The FxParticle packs pos/vel/age/lifetime/seed/flags -> 48-byte std430.
    # integrateEnabled=0 -> write the input back UNCHANGED. Like grain_integrate.comp / fluid_integrate.comp /
    # fpx_integrate.comp this is int64 (gravity*dt over Q16.16 overflows int32) -> VULKAN-SPIR-V-ONLY (DXC
    # compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal --pt1-emit showcase runs the
    # CPU particles::StepEmitIntegrate (byte-identical by construction, the grain_integrate.comp convention).
    # Pure integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU particle array + cross-backend, proven
    # memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/particles_integrate.comp.hlsl:cs"
    # Deterministic GPU Granular/Sand Slice 2 (Slice GR2, the 2nd slice of FLAGSHIP #10): the GRID-HASH
    # NEIGHBOR SEARCH meshed by SIX pure-INT32 count->scan->emit passes (the FL2 fluid_cell/fluid_neighbor
    # twin, applied to grains). The CELL TABLE: grain_cell_count.comp (ONE thread per grain: InterlockedAdd
    # into its cell's counter — the cell id is GrainCellOf = FloorDiv per axis at cell-size hSearch, the
    # bounded-dense-grid linearization, VERBATIM engine/sim/grain.h), grain_cell_scan.comp (SINGLE-THREAD
    # [numthreads(1,1,1)] exclusive prefix-sum -> cellStart CSR + sentinel), grain_cell_emit.comp
    # (SINGLE-THREAD ascending-grain scatter into cellGrains — single-thread because the within-cell order
    # must be ascending grain index, NOT GPU-scheduling order). The NEIGHBOR LIST: grain_neighbor_count.comp
    # (ONE thread per grain: count candidates in its 27-cell stencil passing the per-axis |dx|<hSearch reject
    # — GrainNeighborAccept VERBATIM), grain_neighbor_scan.comp (SINGLE-THREAD exclusive prefix-sum ->
    # neighborStart), grain_neighbor_emit.comp (ONE thread per grain emits its candidates into its DISJOINT
    # neighbors[] slice, NO atomics — the per-grain-disjoint pattern). ALL PURE INT32 (FloorDiv integer divide
    # + per-axis abs-compare, NO int64/fxmul/sqrt, NO radial overlap cull — deferred to GR3's contact solve),
    # so UNLIKE grain_integrate.comp these MSL-GENERATE NATIVELY (they ARE in the Metal hf_gen_msl list) and
    # Metal dispatches the GPU passes. ReadBuffer reads the cell table + neighbor list, PROVEN BIT-EXACT vs
    # the CPU grain.h::BuildGrainCellTable + BuildGrainNeighborList (memcmp). NO new RHI; GR1's
    # grain_integrate + goldens untouched.
    "shaders/grain_cell_count.comp.hlsl:cs"
    "shaders/grain_cell_scan.comp.hlsl:cs"
    "shaders/grain_cell_emit.comp.hlsl:cs"
    "shaders/grain_neighbor_count.comp.hlsl:cs"
    "shaders/grain_neighbor_scan.comp.hlsl:cs"
    "shaders/grain_neighbor_emit.comp.hlsl:cs"
    # Deterministic Integer Broadphase Slice 1 (Slice BP1, the BEACHHEAD of FLAGSHIP #23): THE BODY GRID + CSR
    # CELL TABLE meshed by THREE pure-INT32 count->scan->emit passes (the GR2 grain_cell_* twin, keyed on
    # fpx::FxBody). broad_cell_count.comp (ONE thread per body: InterlockedAdd into its cell's counter —
    # BodyCellOf = FloorDiv per axis at cell-size cellSize, the bounded-dense-grid linearization, VERBATIM
    # engine/sim/broad.h), broad_cell_scan.comp (SINGLE-THREAD [numthreads(1,1,1)] exclusive prefix-sum ->
    # cellStart CSR + sentinel), broad_cell_emit.comp (SINGLE-THREAD ascending-body scatter into cellBodies —
    # single-thread because the within-cell order must be ascending body index, NOT GPU-scheduling order). ALL
    # PURE INT32 (FloorDiv + integer compares + an ascending-index scatter, NO int64/fxmul/sqrt) so these
    # MSL-GENERATE NATIVELY (they ARE in the Metal hf_gen_msl list) and Metal dispatches the GPU passes.
    # ReadBuffer reads cellStart + cellBodies, PROVEN BIT-EXACT vs the CPU broad.h::BuildBodyCellTable (memcmp).
    # NO new RHI; all existing shaders + goldens untouched.
    "shaders/broad_cell_count.comp.hlsl:cs"
    "shaders/broad_cell_scan.comp.hlsl:cs"
    "shaders/broad_cell_emit.comp.hlsl:cs"
    # Deterministic Integer Broadphase Slice 2 (Slice BP2, THE CRUX of FLAGSHIP #23): THE CANDIDATE-PAIR
    # GENERATOR — the 27-cell-stencil pair cull over the BP1 grid + CSR cell table, meshed by THREE pure-INT32
    # count->scan->emit passes. broad_pair_count.comp (ONE thread per body i: scan i's 3x3x3 stencil in FIXED
    # order, count j>i with the six-compare AabbOverlap copied VERBATIM from broad.h/fpx.h), broad_pair_scan.comp
    # (SINGLE-THREAD exclusive prefix-sum -> perBodyOffset + sentinel), broad_pair_emit.comp (SINGLE-THREAD
    # ascending-body scatter of the canonical fpx::FxPair{i,j} into each body's disjoint slice — single-thread
    # because the per-i emit ORDER must be reproduced bit-for-bit). ALL PURE INT32 (FloorDiv + integer compares +
    # an ascending scatter, NO int64/fxmul/sqrt) so these MSL-GENERATE NATIVELY (in hf_gen_msl) and Metal
    # dispatches the GPU passes. ReadBuffer reads pairsOut + offsets, PROVEN BIT-EXACT vs the CPU
    # broad.h::BuildBroadphasePairs (memcmp) + grid-pairs==all-pairs equivalence vs fpx::BuildPairs. NO new RHI.
    "shaders/broad_pair_count.comp.hlsl:cs"
    "shaders/broad_pair_scan.comp.hlsl:cs"
    "shaders/broad_pair_emit.comp.hlsl:cs"
    # Deterministic Rigid<->Fluid Coupling Slice 1 (Slice CP1, the BEACHHEAD of FLAGSHIP #11): the BODY->FLUID
    # grid-hash QUERY meshed by THREE pure-INT32 count->scan->emit passes (the GR2 grain_neighbor twin, per
    # BODY instead of per grain) OVER the reused FL2 fluid cell table (fluid_cell_{count,scan,emit} build the
    # cellStart/cellParticles the query iterates — no new cell-table passes). couple_body_count.comp (ONE
    # thread per body: count the fluid particles in the body's BodyAabb cell RANGE [CellOf(pos-radius) ..
    # CellOf(pos+radius)] passing the per-axis |body.pos.axis - p.pos.axis| < body.radius reject —
    # BodyParticleAccept VERBATIM; a body spans MANY cells, NOT a fixed 27-cell stencil), couple_body_scan.comp
    # (SINGLE-THREAD [numthreads(1,1,1)] exclusive prefix-sum -> bodyStart CSR + sentinel), couple_body_emit.comp
    # (ONE thread per body emits its gathered particle indices into its DISJOINT bodyParticles[] slice, NO
    # atomics — the per-body-disjoint pattern). ALL PURE INT32 (FloorDiv integer divide + per-axis abs-compare,
    # NO int64/fxmul/sqrt, NO radial sphere cull — deferred to CP2's force), so these MSL-GENERATE NATIVELY
    # (they ARE in the Metal hf_gen_msl list) and Metal dispatches the GPU passes (a true GPU pass on both
    # backends, the GR2 precedent). ReadBuffer reads cellStart + cellParticles + bodyStart + bodyParticles,
    # PROVEN BIT-EXACT vs the CPU couple.h::BuildCellTable + GatherBodyParticles (memcmp). NO momentum exchange
    # (CP2-CP4), NO new RHI; couple.h #includes fpx.h + fluid.h read-only (the additive sibling).
    "shaders/couple_body_count.comp.hlsl:cs"
    "shaders/couple_body_scan.comp.hlsl:cs"
    "shaders/couple_body_emit.comp.hlsl:cs"
    # Deterministic Rigid<->Grain Coupling Slice 1 (Slice CG1, the BEACHHEAD of FLAGSHIP #12): the BODY->GRAIN
    # grid-hash QUERY meshed by THREE pure-INT32 count->scan->emit passes (the CP1 couple_body twin, over the
    # reused GR2 GRAIN cell table — grain_cell_{count,scan,emit} above build the cellStart/cellGrains the query
    # iterates — no new cell-table passes). cgrain_body_count.comp (ONE thread per body: count the grains in the
    # body's BodyAabb cell RANGE [CellOf(pos-radius) .. CellOf(pos+radius)] passing the per-axis |body.pos.axis -
    # g.pos.axis| < body.radius reject — BodyGrainAccept VERBATIM; a body spans MANY cells, NOT a fixed 27-cell
    # stencil), cgrain_body_scan.comp (SINGLE-THREAD [numthreads(1,1,1)] exclusive prefix-sum -> bodyStart CSR +
    # sentinel), cgrain_body_emit.comp (ONE thread per body emits its gathered grain indices into its DISJOINT
    # bodyGrains[] slice, NO atomics — the per-body-disjoint pattern). ALL PURE INT32 (FloorDiv integer divide +
    # per-axis abs-compare, NO int64/fxmul/sqrt, NO radial sphere cull — deferred to CG2/CG3), so these
    # MSL-GENERATE NATIVELY (they ARE in the Metal hf_gen_msl list) and Metal dispatches the GPU passes (a true
    # GPU pass on both backends, the CP1/GR2 precedent). ReadBuffer reads cellStart + cellGrains + bodyStart +
    # bodyGrains, PROVEN BIT-EXACT vs the CPU couple_grain.h::BuildGrainCellTable + GatherBodyGrains (memcmp). NO
    # momentum exchange (CG2-CG4), NO new RHI; couple_grain.h #includes fpx.h + grain.h read-only (additive).
    "shaders/cgrain_body_count.comp.hlsl:cs"
    "shaders/cgrain_body_scan.comp.hlsl:cs"
    "shaders/cgrain_body_emit.comp.hlsl:cs"
    # Deterministic Grain<->Fluid Coupling Slice 1 (Slice GF1, the BEACHHEAD of FLAGSHIP #13): the UNIFIED
    # TWO-POOL WORLD + the SHARED-GRID CROSS QUERY meshed by SIX pure-INT32 count->scan->emit passes (the GR2/FL2
    # 27-cell-stencil neighbour search applied CROSS-POOL over ONE shared grid). The two cell tables are the
    # REUSED GR2 grain_cell_{count,scan,emit} + FL2 fluid_cell_{count,scan,emit} passes sized to the SHARED grid
    # (no new cell-table passes). The two CROSS lists: cgf_gf_{count,scan,emit}.comp (ONE thread per grain: count
    # the FLUID particles in the grain's 27-cell stencil over the FLUID cell table passing the per-axis
    # |grain.axis - fluid.axis| < h reject — CrossAccept VERBATIM, NO j==i self-skip since the pools are distinct;
    # SINGLE-THREAD exclusive prefix-sum -> gfStart CSR + sentinel; ONE thread per grain emits its FLUID indices
    # into its DISJOINT gfNeighbors[] slice, NO atomics) + cgf_fg_{count,scan,emit}.comp (the MIRROR: ONE thread
    # per fluid over the GRAIN cell table -> fgStart/fgNeighbors). ALL PURE INT32 (FloorDiv integer divide +
    # per-axis abs-compare, NO int64/fxmul/sqrt, NO radial cull — deferred to GF2/GF3), so these MSL-GENERATE
    # NATIVELY (they ARE in the Metal hf_gen_msl list) and Metal dispatches the GPU passes (a true GPU pass on
    # both backends, the CG1/CP1/GR2/FL2 precedent). ReadBuffer reads grainCellTable + fluidCellTable + gfStart +
    # gfNeighbors + fgStart + fgNeighbors, PROVEN BIT-EXACT vs the CPU couple_gf.h::BuildCGFNeighbors (memcmp). NO
    # momentum exchange (GF2-GF6), NO new RHI; couple_gf.h #includes fpx.h + grain.h + fluid.h read-only (additive).
    "shaders/cgf_gf_count.comp.hlsl:cs"
    "shaders/cgf_gf_scan.comp.hlsl:cs"
    "shaders/cgf_gf_emit.comp.hlsl:cs"
    "shaders/cgf_fg_count.comp.hlsl:cs"
    "shaders/cgf_fg_scan.comp.hlsl:cs"
    "shaders/cgf_fg_emit.comp.hlsl:cs"
    # Deterministic Grain<->Fluid Coupling Slice 2 (Slice GF2, the BUOYANCY + DRAG fluid->grain pass — the CRUX of
    # FLAGSHIP #13, the FIRST momentum exchange). ONE new int64 shader: cgf_buoyancy.comp — ONE THREAD PER GRAIN
    # (multi-thread OVER grains, each thread sums its OWN GF1 gfNeighbors fluid list, the FIXED ascending GF1 emit
    # order, into a buoyancy (∝ the fluid count, opposing gravity up=-FxNormalize(gravity)) + drag (toward the
    # static fluid's mean velocity) impulse, then writes the grain's new vel; AccumGrainBuoyancy copied VERBATIM
    # from engine/sim/couple_gf.h (linear only — grains are points, no torque; the host driver runs IntegrateGrains
    # AFTER). PER-GRAIN DISJOINT -> NO race, NO atomics, [numthreads(64,1,1)] MULTI-THREAD, NO single-thread TDR
    # (the FL4/GR3 Jacobi pattern, cleaner than CG2's per-body reduction). The K-step loop is HOST-driven: each
    # step re-runs the GF1 cross-query passes (cgf_gf/cgf_fg_{count,scan,emit}, int32 MSL-native) from the grains'
    # CURRENT positions -> cgf_buoyancy -> the host integrate (IntegrateGrains), a ComputeToComputeBarrier between
    # sub-passes; the fluid is held STATIC (the reaction is GF3). Because the buoyancy/drag fxmul/fxdiv + the
    # vFluidAvg int64 sum + FxNormalize (FxISqrt) use int64, cgf_buoyancy.comp is VULKAN-SPIR-V-ONLY (DXC compiles
    # int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal --cgf-buoyancy showcase runs the CPU
    # cgf::StepCGFBuoyancy (byte-identical by construction, the cgrain_support.comp convention) — the Vulkan side
    # carries the GPU==CPU proof. ReadBuffer reads the grain array, PROVEN BIT-EXACT vs the CPU
    # couple_gf.h::StepCGFBuoyancySteps (memcmp — the make-or-break). NO new RHI; GF1's cgf_gf/cgf_fg
    # {count,scan,emit} shaders + their goldens UNCHANGED (GF2 is additive; the GF1 cross-query passes stay int32 MSL-native).
    "shaders/cgf_buoyancy.comp.hlsl:cs"
    # Deterministic Grain<->Fluid Coupling Slice 3 (Slice GF3, the CONTACT REACTION / DISPLACEMENT grain->fluid
    # pass — the Newton's-3rd-law HALF of GF2, the 3rd slice of FLAGSHIP #13). ONE new int64 shader:
    # cgf_displace.comp — ONE THREAD PER FLUID PARTICLE (multi-thread OVER fluid particles, each thread serially
    # iterates its GF1 fgNeighbors grain list, the FIXED ascending GF1 emit order). For each grain that CONTAINS
    # the fluid particle (dist < g.radius), it (1) accumulates the surface-snap push (snap the fluid particle to
    # g.radius along the outward normal — LITERALLY grain.h::CollideGrainSphere with the GRAIN as the sphere, fluid
    # particles are points) into a per-thread Jacobi dp + (2) applies the equal-opposite drag-reaction velocity
    # impulse (toward the grain velocity), then writes pos += dp + the updated vel; ApplyGrainsToFluid copied
    # VERBATIM from engine/sim/couple_gf.h. Per-fluid DISJOINT -> NO race, NO atomics, [numthreads(64,1,1)]
    # MULTI-THREAD, NO TDR (the GR3/CG3/CP3 win). Because the FxLength/FxNormalize (FxISqrt) + the drag-reaction
    # fxmul use int64, cgf_displace.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the
    # Metal hf_gen_msl list; the Metal --cgf-displace showcase runs the CPU cgf::ApplyGrainsToFluid (byte-identical
    # by construction). The GPU fluid array is PROVEN BIT-EXACT vs the CPU reference (memcmp — the make-or-break).
    # NO new RHI; GF1/GF2 cgf code + shaders + their goldens UNCHANGED (GF3 is additive; the GF1 cross-query passes
    # stay int32 MSL-native).
    "shaders/cgf_displace.comp.hlsl:cs"
    # Deterministic Rigid<->Grain Coupling Slice 2 (Slice CG2, the CONTACT SUPPORT + DRAG grain->body pass — the
    # CRUX of FLAGSHIP #12, the FIRST momentum exchange). ONE new int64 shader: cgrain_support.comp — ONE THREAD
    # PER BODY (multi-thread OVER bodies, NOT per grain), each thread SERIALLY sums its body's CG1 gathered grain
    # list (bodyGrains[bodyStart[i]..bodyStart[i+1]), the FIXED ascending CG1 emit order) into a contact-support
    # (Σ pen·n over the OVERLAPPING grains — the exact radial cull deferred from CG1's box happens here — the bed
    # pushes the body out/up along each contact normal, ∝ penetration) + drag (toward the static grains' mean
    # velocity) impulse, then writes the body's new vel; AccumBodyGrainForces copied VERBATIM from
    # engine/sim/couple_grain.h (linear only — a sphere body has no support torque; the host driver runs
    # IntegrateBody + ResolveGround AFTER). The body count is TINY so each body's inner loop (LONGER than CP2's —
    # a bed supports through many simultaneous contacts) is still far below the ~2s watchdog -> NO single-thread
    # [numthreads(1,1,1)] / TDR ceiling (the CL3/FPX3 limit does not apply; if body counts scale, an integer
    # atomic-add reduction is needed — OUT of scope CG1-CG6). The K-step loop is HOST-driven: each step re-runs the
    # CG1 query passes (cgrain_body_{count,scan,emit}, int32 MSL-native) from the bodies' CURRENT positions ->
    # cgrain_support -> the host integrate (IntegrateBody + ResolveGround), a ComputeToComputeBarrier between
    # sub-passes; the grains are held STATIC (the reaction is CG3). Because the support/drag fxmul/fxdiv + the
    # vGrainAvg int64 sum + FxNormalize (FxISqrt) use int64, cgrain_support.comp is VULKAN-SPIR-V-ONLY (DXC
    # compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal --cgrain-support showcase runs
    # the CPU cgrain::StepCGrainSupport (byte-identical by construction, the couple_buoyancy.comp convention) —
    # the Vulkan side carries the GPU==CPU proof. ReadBuffer reads the body array, PROVEN BIT-EXACT vs the CPU
    # couple_grain.h::StepCGrainSupportSteps (memcmp — the make-or-break). NO new RHI; CG1's cgrain_body_
    # {count,scan,emit} shaders + their goldens UNCHANGED (CG2 is additive; the CG1 query passes stay int32 MSL-native).
    "shaders/cgrain_support.comp.hlsl:cs"
    # Deterministic Rigid<->Grain Coupling Slice 3 (Slice CG3, the GRAIN REACTION / DISPLACEMENT body->grain pass
    # — the Newton's-3rd-law HALF of CG2, the 3rd slice of FLAGSHIP #12). ONE new int64 shader:
    # cgrain_displace.comp — ONE THREAD PER GRAIN (multi-thread OVER grains, each thread serially iterates the
    # tiny body list, fixed order). For each DYNAMIC body that CONTAINS the grain (dist < b.radius + g.radius),
    # it (1) accumulates the surface-snap push (snap the grain centre to b.radius+g.radius along the outward
    # normal — LITERALLY grain.h::CollideGrainSphere(g, GrainSphereFromBody(b))) into a per-thread Jacobi dp +
    # (2) applies the equal-opposite drag-reaction velocity impulse (toward the body velocity), then writes
    # pos += dp + the updated vel; ApplyBodyToGrains copied VERBATIM from engine/sim/couple_grain.h. Per-grain
    # DISJOINT -> NO race, NO atomics, [numthreads(64,1,1)] MULTI-THREAD, NO TDR (the GR3/CP3 win). Because the
    # FxLength/FxNormalize (FxISqrt) + the drag-reaction fxmul use int64, cgrain_displace.comp is
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal
    # --cgrain-displace showcase runs the CPU cgrain::ApplyBodyToGrains (byte-identical by construction). The
    # GPU grain array is PROVEN BIT-EXACT vs the CPU reference (memcmp — the make-or-break). NO new RHI; CG1/CG2
    # cgrain code + shaders + their goldens UNCHANGED (CG3 is additive; the CG1 query passes stay int32 MSL-native).
    "shaders/cgrain_displace.comp.hlsl:cs"
    # Deterministic Rigid<->Fluid Coupling Slice 2 (Slice CP2, the BUOYANCY + DRAG fluid->body pass — the CRUX
    # of FLAGSHIP #11, the FIRST momentum exchange). ONE new int64 shader: couple_buoyancy.comp — ONE THREAD
    # PER BODY (multi-thread OVER bodies, NOT per particle), each thread SERIALLY sums its body's CP1 gathered
    # list (bodyParticles[bodyStart[i]..bodyStart[i+1]), the FIXED ascending CP1 emit order) into a buoyant
    # (∝ the gathered count = the displaced volume, along up = -normalize(gravity)) + drag (toward the static
    # fluid's mean velocity) impulse, then writes the body's new vel; AccumBodyForces copied VERBATIM from
    # engine/sim/couple.h (linear only — a sphere body has no buoyancy torque; the host driver runs IntegrateBody
    # + ResolveGround AFTER). The body count is TINY so each body's short inner loop is far below the ~2s
    # watchdog -> NO single-thread [numthreads(1,1,1)] / TDR ceiling (the CL3/FPX3 limit does not apply). The
    # K-step loop is HOST-driven: each step re-runs the CP1 query passes (couple_body_{count,scan,emit}, int32
    # MSL-native) from the bodies' CURRENT positions -> couple_buoyancy -> the host integrate (IntegrateBody +
    # ResolveGround), a ComputeToComputeBarrier between sub-passes; the fluid is held STATIC (the reaction is
    # CP3). Because the buoyancy/drag fxmul/fxdiv + the vFluidAvg int64 sum + FxNormalize (FxISqrt) use int64,
    # couple_buoyancy.comp is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal
    # hf_gen_msl list; the Metal --couple-buoyancy showcase runs the CPU couple::StepCoupleBuoyancy (byte-
    # identical by construction, the grain_contact_dp.comp/fluid_dp.comp convention) — the Vulkan side carries
    # the GPU==CPU proof. ReadBuffer reads the body array, PROVEN BIT-EXACT vs the CPU couple.h::
    # StepCoupleBuoyancySteps (memcmp — the make-or-break). NO new RHI; CP1's couple_body_{count,scan,emit}
    # shaders + their goldens UNCHANGED (CP2 is additive; the CP1 query passes stay int32 MSL-native).
    "shaders/couple_buoyancy.comp.hlsl:cs"
    # Deterministic Rigid<->Fluid Coupling Slice 3 (Slice CP3, the FLUID REACTION / DISPLACEMENT body->fluid
    # pass — the Newton's-3rd-law HALF of CP2, the 3rd slice of FLAGSHIP #11). ONE new int64 shader:
    # couple_displace.comp — ONE THREAD PER FLUID PARTICLE (multi-thread OVER particles, the mirror of CP2's
    # per-body dispatch), each thread serially iterates the TINY body list (fixed order) and for each DYNAMIC
    # body that CONTAINS it (dist < b.radius) accumulates the surface-snap push into a LOCAL Jacobi accumulator
    # (reads the iteration-start body state read-only, writes ONLY its own particle -> per-particle disjoint,
    # NO atomics, NO TDR) + applies the equal-opposite DRAG-REACTION velocity impulse, then writes pos += accum
    # + the updated vel; ApplyBodyToFluid copied VERBATIM from engine/sim/couple.h (the body parts the fluid +
    # drags it along — a cavity/wake). The Vulkan --couple-displace-shot does the GPU pass then memcmp's the
    # fluid array vs the CPU couple.h::ApplyBodyToFluid reference (the make-or-break). Because the FxLength/
    # FxNormalize (FxISqrt) + the drag-reaction fxmul use int64, couple_displace.comp is VULKAN-SPIR-V-ONLY (DXC
    # compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal --couple-displace showcase
    # runs the CPU couple::ApplyBodyToFluid (byte-identical by construction, the couple_buoyancy.comp/
    # grain_collide.comp convention) — the Vulkan side carries the GPU==CPU proof. NO new RHI; CP1's couple_body_
    # {count,scan,emit} + CP2's couple_buoyancy shaders + their goldens UNCHANGED (CP3 is additive; the CP1
    # query passes stay int32 MSL-native).
    "shaders/couple_displace.comp.hlsl:cs"
    # Deterministic GPU Granular/Sand Slice 3 (Slice GR3, the FRICTIONLESS CONTACT PROJECTION of FLAGSHIP
    # #10, the FL4 fluid_dp/fluid_apply/fluid_collide twin): the JACOBI contact solve. The K-step JACOBI loop
    # is HOST-driven (predict[GR1] -> GR2 neighbour passes[host-rebuilt from the predicted positions] ->
    # {grain_contact_dp -> grain_contact_apply}×iters -> grain_collide per step, a ComputeToComputeBarrier
    # between sub-passes). grain_contact_dp.comp: ONE thread per grain computes Δp_i = Σ_{j, pen>0}
    # (w_i/(w_i+w_j))·pen·unit(p_i−p_j) into a SEPARATE dp[] buffer (the Jacobi double-buffer); only pen>0
    # contributes (the exact radial overlap cull GR2 deferred); grain_contact_apply.comp: ONE thread per grain
    # applies p_i += dp_i; grain_collide.comp: ONE thread per grain derives vel = (pos−prev)/dt then projects
    # RADIUS-AWARE out of the ground (pos.y >= groundY + radius) + the static spheres (centre -> sphereR +
    # grainR). ALL per-grain INDEPENDENT (reads iteration-start state read-only, writes only its own grain) ->
    # MULTI-THREAD [numthreads(64,1,1)], NO atomics, NO single-thread, NO TDR ceiling (the FL4 Jacobi win over
    # the cloth's order-dependent Gauss-Seidel). SolveGrainContact + the collide math copied VERBATIM from
    # engine/sim/grain.h. Because the contact pen·n fxmul/fxdiv + FxLength/FxNormalize (FxISqrt) + the velocity
    # fxdiv use int64, grain_contact_dp.comp + grain_collide.comp are VULKAN-SPIR-V-ONLY (DXC compiles int64;
    # glslc cannot) -> NOT in the Metal hf_gen_msl list; grain_contact_apply.comp is a plain int32 pos+=dp add
    # (it COULD MSL-generate) but is kept Vulkan-only alongside its two int64 siblings (the simplest correct
    # split — the whole solve trio is one Vulkan-only group), the Metal --grain-contact showcase running the
    # CPU grain::StepGrainContact (byte-identical by construction). ReadBuffer reads gGrains, PROVEN BIT-EXACT
    # vs the CPU grain.h::StepGrainContactSteps (memcmp — the make-or-break). NO new RHI; GR1's grain_integrate
    # + GR2's grain_cell/grain_neighbor shaders + their goldens untouched.
    "shaders/grain_contact_dp.comp.hlsl:cs"
    "shaders/grain_contact_apply.comp.hlsl:cs"
    "shaders/grain_collide.comp.hlsl:cs"
    # Deterministic GPU Granular/Sand Slice 4 (Slice GR4, the TANGENTIAL COULOMB FRICTION — the angle-of-repose
    # SIGNATURE of FLAGSHIP #10, the grain_contact_dp twin with the friction clamp). ONE new int64 shader:
    # grain_friction.comp — ONE thread per grain computes the JACOBI tangential-friction Δp_i = Σ_{j, pen>0}
    # −share·corr where corr is the tangential relative displacement Δx_t = Δx_rel − (Δx_rel·n)·n clamped to the
    # Coulomb cone fmax = fxmul(μ, pen) (STATIC t<=fmax cancels all slip, KINETIC t>fmax scales by fxdiv(fmax,t)),
    # into a SEPARATE dp[] buffer (the Jacobi double-buffer). The K-step JACOBI loop is HOST-driven: each step =
    # predict[GR1] -> GR2 neighbour passes -> {grain_contact_dp -> grain_contact_apply (NORMAL) -> grain_friction
    # -> grain_contact_apply (TANGENTIAL)}×iters -> grain_collide, a ComputeToComputeBarrier between sub-passes.
    # Per-grain INDEPENDENT (reads iteration-start state read-only, writes only its own dp) -> MULTI-THREAD
    # [numthreads(64,1,1)], NO atomics, NO single-thread, NO TDR ceiling (the FL4/GR3 Jacobi win). REUSES
    # grain_contact_apply.comp + grain_collide.comp VERBATIM (the apply + velocity/collide passes are shared with
    # GR3 — NOT duplicated). SolveGrainFriction copied VERBATIM from engine/sim/grain.h. Because the friction
    # Δx_t/cone fxmul/fxdiv + FxLength (FxISqrt) use int64, grain_friction.comp is VULKAN-SPIR-V-ONLY (DXC
    # compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list, the Metal --grain-friction showcase
    # running the CPU grain::StepGrainFriction (byte-identical by construction). ReadBuffer reads gGrains, PROVEN
    # BIT-EXACT vs the CPU grain.h::StepGrainFrictionSteps (memcmp — the make-or-break). NO new RHI; GR1's
    # grain_integrate + GR2's grain_cell/grain_neighbor + GR3's grain_contact_dp/grain_contact_apply/grain_collide
    # shaders + their goldens untouched.
    "shaders/grain_friction.comp.hlsl:cs"
    # Deterministic GPU Fluid Slice 2 (Slice FL2, the 2nd slice of FLAGSHIP #9): the GRID-HASH NEIGHBOR
    # SEARCH meshed by SIX pure-INT32 count->scan->emit passes (the FPX2 fpx_pair / CL2 cloth_edge twin).
    # The CELL TABLE: fluid_cell_count.comp (ONE thread per particle: InterlockedAdd into its cell's counter
    # — the cell id is CellOf = FloorDiv per axis at cell-size h, the bounded-dense-grid linearization,
    # VERBATIM engine/sim/fluid.h), fluid_cell_scan.comp (SINGLE-THREAD [numthreads(1,1,1)] exclusive
    # prefix-sum -> cellStart CSR + sentinel), fluid_cell_emit.comp (SINGLE-THREAD ascending-particle scatter
    # into cellParticles — single-thread because the within-cell order must be ascending particle index,
    # NOT GPU-scheduling order). The NEIGHBOR LIST: fluid_neighbor_count.comp (ONE thread per particle:
    # count candidates in its 27-cell stencil passing the per-axis |dx|<h reject — NeighborAccept VERBATIM),
    # fluid_neighbor_scan.comp (SINGLE-THREAD exclusive prefix-sum -> neighborStart), fluid_neighbor_emit.comp
    # (ONE thread per particle emits its candidates into its DISJOINT neighbors[] slice, NO atomics — the
    # FPX2 per-particle-disjoint pattern). ALL PURE INT32 (FloorDiv integer divide + per-axis abs-compare,
    # NO int64/fxmul/sqrt, NO radial r<h cull — deferred to FL3's kernel=0-beyond-h), so UNLIKE
    # fluid_integrate.comp these MSL-GENERATE NATIVELY (they ARE in the Metal hf_gen_msl list) and Metal
    # dispatches the GPU passes. ReadBuffer reads the cell table + neighbor list, PROVEN BIT-EXACT vs the
    # CPU fluid.h::BuildCellTable + BuildNeighborList (memcmp). NO new RHI; FL1's fluid_integrate + goldens
    # untouched.
    "shaders/fluid_cell_count.comp.hlsl:cs"
    "shaders/fluid_cell_scan.comp.hlsl:cs"
    "shaders/fluid_cell_emit.comp.hlsl:cs"
    "shaders/fluid_neighbor_count.comp.hlsl:cs"
    "shaders/fluid_neighbor_scan.comp.hlsl:cs"
    "shaders/fluid_neighbor_emit.comp.hlsl:cs"
    # Deterministic GPU Fluid Slice 3 (Slice FL3, the MAKE-OR-BREAK of FLAGSHIP #9): the PBF DENSITY + λ.
    # fluid_density.comp: ONE thread per particle gathers ρ_i = W[0] + Σ_{j∈neighbors} W[bin(r_ij²)] over the
    # FL2 neighbour list via a HOST-SNAPPED Q16.16 kernel LUT (W), the RadiusSq (int64 dx²) + BinOf
    # ((r²*bins)/h² int64 divide) + gather copied VERBATIM from engine/sim/fluid.h::ComputeDensity.
    # fluid_lambda.comp: ONE thread per particle computes C_i = ρ_i/ρ0 − 1 then λ_i = −C_i/(Σ|∇C_i|²+ε)
    # (the unilateral clamp C_i<0 -> λ_i=0), the fxdiv/fxmul/FxNormalize (via the int64 FxISqrt, NO float
    # sqrt) + gradW gather copied VERBATIM from engine/sim/fluid.h::ComputeLambda. Per-particle INDEPENDENT
    # (a gather over the fixed FL2 list) -> multi-thread, NO atomics, NO single-thread/TDR (the FL1 pattern).
    # Because r² (dx² over Q16.16) + the λ fxdiv/FxISqrt use int64, these two are VULKAN-SPIR-V-ONLY (DXC
    # compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal --fluid-density showcase
    # runs the CPU fluid::ComputeDensity + ComputeLambda (byte-identical by construction, the
    # fluid_integrate.comp/cloth_solve.comp convention). ReadBuffer reads ρ + λ, PROVEN BIT-EXACT vs the CPU
    # reference (memcmp — the make-or-break). NO new RHI; FL1+FL2 shaders + goldens untouched.
    "shaders/fluid_density.comp.hlsl:cs"
    "shaders/fluid_lambda.comp.hlsl:cs"
    # Deterministic GPU Fluid Slice 4 (Slice FL4, the incompressibility solver of FLAGSHIP #9): the PBF
    # DENSITY-CONSTRAINT SOLVE. The K-step JACOBI PBF loop is HOST-driven (predict[FL1] -> neighbours[host
    # FL2] -> {fluid_density -> fluid_lambda -> fluid_dp -> fluid_apply}×iters -> fluid_collide per step, a
    # ComputeToComputeBarrier between sub-passes), reusing the FL3 fluid_density/fluid_lambda passes per
    # iteration. fluid_dp.comp: ONE thread per particle computes Δp_i = (1/ρ0) Σ_{j∈neighbors} (λ_i+λ_j)
    # gradW[bin]·unit(p_j−p_i) into a SEPARATE dp[] buffer (the Jacobi double-buffer); fluid_apply.comp: ONE
    # thread per particle applies p_i += dp_i; fluid_collide.comp: ONE thread per particle derives vel =
    # (pos−prev)/dt then projects out of the ground + the static spheres. ALL per-particle INDEPENDENT (reads
    # iteration-start state read-only, writes only its own particle) -> MULTI-THREAD, NO atomics, NO
    # single-thread, NO TDR ceiling (the Jacobi design win over the cloth's order-dependent Gauss-Seidel).
    # SolveDensityConstraint + the collide math copied VERBATIM from engine/sim/fluid.h. Because the λ-scale
    # fxmul + the gradW fxdiv + FxISqrt + the velocity fxdiv use int64, these three are VULKAN-SPIR-V-ONLY
    # (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal --fluid-solve showcase
    # runs the CPU fluid::StepFluid (byte-identical by construction). ReadBuffer reads gParticles, PROVEN
    # BIT-EXACT vs the CPU fluid.h::StepFluid (memcmp — the make-or-break). NO new RHI; FL1-FL3 shaders +
    # goldens untouched.
    "shaders/fluid_dp.comp.hlsl:cs"
    "shaders/fluid_apply.comp.hlsl:cs"
    "shaders/fluid_collide.comp.hlsl:cs"
    # Deterministic GPU Cloth Slice 2 (Slice CL2, the 2nd slice of FLAGSHIP #8): the DISTANCE-CONSTRAINT
    # GRAPH BUILD (the FPX2/MC3 count->scan->emit on lattice particles). cloth_edge_count.comp: ONE thread
    # per particle counts its OWNED forward edges in bounds (right/down structural, the two cell diagonals
    # shear, 2-away right/down bend) -> gEdgeCount[p]. cloth_edge_scan.comp: SINGLE-THREAD ([numthreads(1,1,1)])
    # exclusive prefix-sum -> gEdgeOffset. cloth_edge_emit.comp: ONE thread per particle emits each owned edge
    # {i,j,restLen,kind} into gEdges at its DISJOINT [offset[p],..) range (race-free, NO atomics; restLen the
    # host-snapped int32 per kind). CountOwnedEdges + the fixed emit order copied VERBATIM from
    # engine/sim/cloth.h::BuildConstraints — PURE INT32 (index arithmetic + bounds compares, NO products, NO
    # int64, NO sqrt in-shader; the restLen FxLength is computed HOST-side), so unlike cloth_integrate.comp's
    # int64 fxmul these MSL-generate NATIVELY on Metal. Pure integer SSBO compute + compute->compute barriers
    # + ReadBuffer -> bit-exact GPU==CPU constraint list + cross-backend, proven memcmp. NO new RHI; existing
    # shaders + goldens untouched.
    "shaders/cloth_edge_count.comp.hlsl:cs"
    "shaders/cloth_edge_scan.comp.hlsl:cs"
    "shaders/cloth_edge_emit.comp.hlsl:cs"
    # Deterministic GPU Cloth Slice 3 (Slice CL3, the MAKE-OR-BREAK of FLAGSHIP #8): the PBD DISTANCE-
    # CONSTRAINT SOLVER. cloth_solve.comp runs `steps` iterations of StepCloth = IntegrateParticles (CL1)
    # then `iters` Gauss-Seidel constraint passes (each iterating ALL CL2 constraints in the FIXED emit
    # order applying SolveDistanceConstraint = the verbatim fpx.h::ResolvePair projection generalized from a
    # sphere-pair to a constraint EDGE: pen = FxLength(pos[j]-pos[i]) - restLen; split by inverse-mass shares
    # via fxdiv; pinned (invMass 0) never move) then a ground floor-clamp — copied VERBATIM from
    # engine/sim/cloth.h::StepCloth, so the pinned corners hold + the lattice DRAPES cohesively (unlike CL1's
    # free-fall). The constraint projection is SEQUENTIAL (Gauss-Seidel, order-dependent), so this is a SINGLE
    # THREAD ([numthreads(1,1,1)], the fpx_solve.comp mirror) -> bit-exact GPU==CPU + cross-backend, NO
    # atomics. solveEnabled=0 -> particles UNCHANGED. Like fpx_solve.comp / cloth_integrate.comp this is int64
    # (fxdiv/FxISqrt) -> VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl
    # list; the Metal --cloth-solve showcase runs the CPU cloth::StepCloth (byte-identical by construction,
    # the fpx_solve.comp convention). Pure integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU particle
    # array + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/cloth_solve.comp.hlsl:cs"
    # Deterministic Articulated-Body Ragdoll Slice 1 (Slice JT1, the BEACHHEAD of FLAGSHIP #15): the
    # JOINT GRAPH + BALL-JOINT CONSTRAINT. joint_ball_solve.comp runs `steps` iterations of StepJointWorld
    # = IntegrateBodyFull each body (the FPX4 6-DOF integrate) then `iters` Gauss-Seidel ball passes (each
    # iterating ALL joints in the FIXED order applying SolveBallJoint = cloth::SolveDistanceConstraint with
    # restLen 0 over the WORLD anchors [pa = a.pos + FxRotate(a.orient, anchorA), pb = b.pos +
    # FxRotate(b.orient, anchorB)], the correction applied to the body CENTRES — the JT1 translation-only
    # split, NO lever-arm/angular coupling [JT2]) then a ground floor-clamp — copied VERBATIM from
    # engine/sim/joint.h::StepJointWorld, so the pinned root holds + the ball joints keep the links
    # connected -> the chain HANGS / SWINGS (unlike independent free-fall). The ball projection is
    # SEQUENTIAL (Gauss-Seidel, order-dependent), so this is a SINGLE THREAD ([numthreads(1,1,1)], the
    # cloth_solve.comp/fpx_solve.comp mirror) -> bit-exact GPU==CPU + cross-backend, NO atomics.
    # solveEnabled=0 -> bodies UNCHANGED. Like cloth_solve.comp / fpx_solve.comp this is int64 (FxRotate
    # fxmul/fxdiv/FxISqrt) -> VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal
    # hf_gen_msl list; the Metal --joint-ball showcase runs the CPU joint::StepJointWorld (byte-identical
    # by construction, the cloth_solve.comp convention). Pure integer SSBO compute + ReadBuffer -> bit-exact
    # GPU==CPU body array + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/joint_ball_solve.comp.hlsl:cs"
    # Deterministic Articulated-Body Ragdoll Slice 2 (Slice JT2, THE NEW-PHYSICS BEAT of FLAGSHIP #15): the
    # ANGULAR LIMIT (hinge + cone). joint_angular_solve.comp runs `steps` iterations of StepArticulated =
    # IntegrateBodyFull each body (the FPX4 6-DOF integrate) then `iters` Gauss-Seidel passes EACH doing
    # {all SolveBallJoint (the JT1 positional ball) then all SolveAngularLimit (the JT2 swing-twist + host-cos
    # cone clamp + nlerp inverse-mass apply, projecting qrel = qA⁻¹·qB back into the cone/hinge)} then a
    # ground floor-clamp — copied VERBATIM from engine/sim/joint.h::StepArticulated/SolveAngularLimit. The
    # cone limit is supplied as HOST cos/sin constants (cosHalfLimit/sinHalfLimit) so the SIM does ZERO
    # runtime acos/sin; the apply is NLERP (component lerp + FxQuatNormalize), NOT slerp. The whole step is
    # SEQUENTIAL (Gauss-Seidel, order-dependent) -> a SINGLE THREAD ([numthreads(1,1,1)], the
    # joint_ball_solve.comp mirror) -> bit-exact GPU==CPU + cross-backend, NO atomics. solveEnabled=0 ->
    # bodies UNCHANGED. Like joint_ball_solve.comp this is int64 (the quaternion fxmul/fxdiv/FxISqrt) ->
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal
    # --joint-hinge showcase runs the CPU joint::StepArticulated (byte-identical by construction). NO new RHI;
    # existing shaders + goldens untouched (joint_ball_solve.comp + JT1's FxJoint/SolveBallJoint frozen).
    "shaders/joint_angular_solve.comp.hlsl:cs"
    # Deterministic Vehicle Physics Slice 1 (Slice VH1, the BEACHHEAD of FLAGSHIP #16): the SUSPENSION
    # SPRING JOINT. vehicle_spring_solve.comp runs `steps` iterations of StepSpringWorld = IntegrateBodyFull
    # each body (the FPX4 6-DOF integrate) then `iters` Gauss-Seidel spring passes (each iterating ALL
    # springs in the FIXED order applying SolveSpringJoint = cloth::SolveDistanceConstraint GENERALIZED to
    # restLen != 0 + a stiffness scale + normal-velocity damping over the WORLD anchors [pa = a.pos +
    # FxRotate(a.orient, anchorA), pb = b.pos + FxRotate(b.orient, anchorB)]; pen = len - restLen; corr =
    # fxmul(pen, stiffness); the inverse-mass split moves the body centres; vRelN = FxDot(b.vel-a.vel, n);
    # the damper subtracts fxmul(vRelN, damping) along n) then a ground floor-clamp — copied VERBATIM from
    # engine/sim/vehicle.h::StepSpringWorld, so a pinned anchor holds + the spring holds the body at ~restLen
    # + the damper kills the oscillation -> the body BOBS then SETTLES at rest (a chassis<->wheel suspension).
    # The spring projection is SEQUENTIAL (Gauss-Seidel, order-dependent), so this is a SINGLE THREAD
    # ([numthreads(1,1,1)], the joint_ball_solve.comp mirror) -> bit-exact GPU==CPU + cross-backend, NO
    # atomics. solveEnabled=0 -> bodies UNCHANGED. Like joint_ball_solve.comp this is int64 (FxRotate/FxDot
    # fxmul/fxdiv/FxISqrt) -> VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal
    # hf_gen_msl list; the Metal --vehicle-spring showcase runs the CPU vehicle::StepSpringWorld
    # (byte-identical by construction, the joint_ball_solve.comp convention). Pure integer SSBO compute +
    # ReadBuffer -> bit-exact GPU==CPU body array + cross-backend, proven memcmp. NO new RHI; existing shaders
    # + goldens untouched (vehicle.h is the additive sibling; #includes joint.h + fpx.h read-only).
    "shaders/vehicle_spring_solve.comp.hlsl:cs"
    # Deterministic Active Ragdoll Slice 1 (Slice AC1, the BEACHHEAD of FLAGSHIP #17: DETERMINISTIC ACTIVE
    # RAGDOLL / PHYSICAL-ANIMATION BLENDING, hf::sim::active): the ANGULAR POSE-DRIVE primitive.
    # active_drive_solve.comp runs `steps` iterations of StepDriveWorld = IntegrateBodyFull each body (the
    # FPX4 6-DOF integrate) then `iters` Gauss-Seidel passes EACH doing {all SolveBallJoint (the JT1
    # positional ball) then all SolveAngularLimit (the JT2 cone/hinge) then all SolveAngularDrive (the AC1
    # NEW PRIMITIVE: drive qrel = qA⁻¹·qB toward a target quaternion by a stiffness fraction — SolveAngular
    # Limit's inverse-mass nlerp apply with the cone clamp replaced by an nlerp-toward-target + a shortest-
    # arc sign fix)} then a ground floor-clamp — copied VERBATIM from engine/sim/active.h::StepDriveWorld/
    # SolveAngularDrive (which reuse joint.h + fpx.h). The drive strength is a stiffness in [0,kOne] (the VH1
    # spring soft-constraint pattern lifted to orientation); stiffness 0 -> no drive, kOne -> snap to target.
    # The whole step is SEQUENTIAL (Gauss-Seidel, order-dependent) -> a SINGLE THREAD ([numthreads(1,1,1)],
    # the joint_angular_solve.comp mirror) -> bit-exact GPU==CPU + cross-backend, NO atomics. solveEnabled=0
    # -> bodies UNCHANGED. Like joint_angular_solve.comp this is int64 (the quaternion fxmul/fxdiv/FxISqrt) ->
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal
    # --active-drive showcase runs the CPU active::StepDriveWorld (byte-identical by construction). NO new RHI;
    # existing shaders + goldens untouched (active.h is the additive sibling; #includes joint.h + fpx.h
    # read-only). The ONLY new shader of the whole flagship.
    "shaders/active_drive_solve.comp.hlsl:cs"
    # CL4 COLLISION. cloth_collide.comp FOLDS the CL3 PBD step (integrate + Gauss-Seidel constraint passes +
    # ground floor-clamp, copied VERBATIM from cloth_solve.comp/cloth.h) AND per-step COLLISION (project each
    # non-pinned particle out of a small STATIC set of SPHERE colliders along the outward normal, then a
    # ground plane clamp — the int32 AABB reject + int64 FxLength/FxNormalize push to the surface copied
    # VERBATIM from engine/sim/cloth.h::CollideParticleSphere) into ONE single-thread ([numthreads(1,1,1)])
    # `steps` loop so the cloth DRAPES over the rigid sphere (the SphereCollider reuses fpx::FxBody pos+radius,
    # the SAME Q16.16 units). The Gauss-Seidel order-dependence makes single-thread necessary for bit-
    # exactness; the per-particle collision within it is order-independent but runs in the same thread.
    # collideEnabled=0 -> particles UNCHANGED; sphereCount=0 -> byte-identical to cloth_solve (the zero-
    # collider equivalence). Like cloth_solve.comp / fpx_solve.comp this is int64 (fxdiv/FxISqrt) ->
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) -> NOT in the Metal hf_gen_msl list; the Metal
    # --cloth-collide showcase runs the CPU cloth::StepClothCollide (byte-identical by construction). Pure
    # integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU particle array + cross-backend, proven memcmp.
    # NO new RHI; CL1-CL3 shaders + goldens untouched.
    "shaders/cloth_collide.comp.hlsl:cs"
    # Deterministic Fixed-Point Physics Slice 2 (Slice FPX2): the integer-AABB BROADPHASE candidate-pair
    # generator (the MC3 count->scan->emit on bodies). fpx_pair_count.comp: ONE thread per body i scans j>i
    # and writes perBodyCount[i]=Σ AabbOverlap(BodyAabb(i),BodyAabb(j)) (each unordered pair counted once,
    # i<j). fpx_pair_scan.comp: SINGLE-THREAD ([numthreads(1,1,1)]) exclusive prefix-sum -> perBodyOffset.
    # fpx_pair_emit.comp: ONE thread per body i re-scans j>i and emits each overlapping (i,j) into gPairs at
    # its DISJOINT [offset[i],..) range (race-free, NO atomics). enabled=0 -> count 0 / emit nothing.
    # BodyAabb + AabbOverlap copied VERBATIM from engine/sim/fpx.h — PURE INT32 (integer add/sub + six
    # compares, NO products, NO int64), so these MSL-generate NATIVELY on Metal (unlike fpx_integrate.comp's
    # int64 fxmul). Pure integer SSBO compute + compute->compute barriers + ReadBuffer -> bit-exact GPU==CPU
    # pair list + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/fpx_pair_count.comp.hlsl:cs"
    "shaders/fpx_pair_scan.comp.hlsl:cs"
    "shaders/fpx_pair_emit.comp.hlsl:cs"
    # Deterministic GPU Navmesh Slice 1 (Slice NAV1, the BEACHHEAD of FLAGSHIP #7: deterministic GPU
    # navmesh + pathfinding): the integer HEIGHTFIELD SPAN RASTERIZATION (the MC3 count->scan->emit on
    # voxel columns). nav_raster_count.comp: ONE thread per column (cx,cz) scans every host-snapped voxel
    # triangle and writes colCount[col] = #triangles whose XZ projection covers the column-center (raw
    # span count). nav_raster_scan.comp: SINGLE-THREAD ([numthreads(1,1,1)]) exclusive prefix-sum ->
    # colOffset. nav_raster_emit.comp: ONE thread per column re-scans the triangles + emits each covering
    # tri's integer y-span {ymin,ymax,area} into gSpans at its DISJOINT [offset[col],..) range (race-free,
    # NO atomics). enabled=0 -> count 0 / emit nothing. TriColumnAabb + PointInTriXZ (three integer edge
    # functions) + TriYSpan copied VERBATIM from engine/nav/navmesh.h — PURE INT32 (the voxel coords are
    # small enough the edge-function products fit int32; the CPU reference widens to int64 to be
    # overflow-safe but is bit-identical for these bounded coords), so these MSL-generate NATIVELY on Metal
    # (unlike fpx_integrate.comp's int64 fxmul). Pure integer SSBO compute + compute->compute barriers +
    # ReadBuffer -> bit-exact GPU==CPU span list + cross-backend, proven memcmp. NO new RHI; existing
    # shaders + goldens untouched.
    "shaders/nav_raster_count.comp.hlsl:cs"
    "shaders/nav_raster_scan.comp.hlsl:cs"
    "shaders/nav_raster_emit.comp.hlsl:cs"
    # Deterministic GPU Navmesh Slice 2 (Slice NAV2, the 2nd slice of FLAGSHIP #7): the WALKABLE FILTER +
    # INTEGER CHAMFER DISTANCE FIELD over the NAV1 heightfield. nav_filter.comp: ONE thread per column —
    # walk that column's MERGED spans (top->bottom), mark each span's top walkable iff the integer
    # clearance ABOVE it (gap to the next solid span / the heightfield top) >= walkableHeight, write the
    # per-column walkable mask + the topmost walkable surface y. nav_distance.comp: SINGLE-THREAD
    # ([numthreads(1,1,1)], the nav_raster_scan/mc_scan serial mirror) two-pass integer chamfer (seed
    # non-walkable+border=0/walkable=INF; forward TL->BR then backward BR->TL min-sweep; cardinal weight 2
    # / diagonal weight 3, NO sqrt; only traverse walkable+connected neighbours via the abs(surfaceY)
    # max-step test -> geodesic). The clearance/step + chamfer math is copied VERBATIM from
    # engine/nav/navmesh.h — PURE INT32 (NO int64), so these MSL-generate NATIVELY on Metal. Pure integer
    # SSBO compute + a compute->compute barrier + ReadBuffer -> bit-exact GPU==CPU walkable[]+dist[] +
    # cross-backend, proven memcmp. NO new RHI; NAV1 shaders + goldens untouched.
    "shaders/nav_filter.comp.hlsl:cs"
    "shaders/nav_distance.comp.hlsl:cs"
    # Deterministic GPU Navmesh Slice 3 (Slice NAV3, the 3rd slice + MAKE-OR-BREAK of FLAGSHIP #7): the
    # INTEGER WATERSHED REGION GENERATION over the NAV2 distance field. nav_region.comp: SINGLE-THREAD
    # ([numthreads(1,1,1)], the nav_distance/nav_raster_scan/mc_scan serial mirror) runs the WHOLE
    # level-descending fixed-order watershed (for level=maxDist..1: GROW existing regions into this
    # level's cells by lowest-region-id among connected neighbours, in ascending cellId until stable;
    # then SEED any still-unassigned walkable cell at this level in ascending cellId + grow it) ->
    # gRegion[] (0=none, ids dense from 1). The watershed is copied VERBATIM from
    # engine/nav/navmesh.h::BuildRegions — PURE INT32 (NO int64), so it MSL-generates NATIVELY on Metal.
    # Pure integer SSBO compute + a compute->compute barrier + ReadBuffer -> bit-exact GPU==CPU region[]
    # + cross-backend, proven memcmp. NO new RHI; NAV1+NAV2 shaders + goldens untouched.
    "shaders/nav_region.comp.hlsl:cs"
    # Deterministic GPU Navmesh Slice 4 (Slice NAV4, the 4th slice of FLAGSHIP #7): CONTOUR TRACE +
    # INTEGER POLYGONIZATION over the NAV3 region partition. nav_contour.comp: SINGLE-THREAD
    # ([numthreads(1,1,1)], the nav_region serial mirror) traces each region's boundary into an integer
    # corner loop (lowest-cellId start, fixed wall-follow turn order) then Douglas-Peucker simplifies it
    # (perpendicular-distance SQUARED vs maxError^2, explicit fixed-order stack, no sqrt) -> gContourVerts
    # + per-region offsets/counts. nav_polygonize.comp: SINGLE-THREAD ear-clip triangulation (clip the
    # LOWEST-index valid ear, integer Cross2 orientation + point-in-triangle) + per-edge cross-poly
    # adjacency -> gPolys (8 uints/poly) + per-region offsets/counts. Both copied VERBATIM from
    # engine/nav/navmesh.h::TraceContours/SimplifyContour/BuildPolyMesh — PURE INT32 (NO int64; corner
    # coords <=32 for the 32x32 showcase, the documented overflow bound), so they MSL-generate NATIVELY
    # on Metal. Pure integer SSBO compute + compute->compute barriers + ReadBuffer -> bit-exact GPU==CPU
    # (contour verts + poly indices + adjacency) + cross-backend, proven memcmp. NO new RHI; NAV1-NAV3
    # shaders + goldens untouched.
    "shaders/nav_contour.comp.hlsl:cs"
    "shaders/nav_polygonize.comp.hlsl:cs"
    # Deterministic GPU Navmesh Slice 5 (Slice NAV5, the 5th slice + HEADLINE of FLAGSHIP #7): INTEGER
    # A* PATHFINDING over the NAV4 poly adjacency graph. nav_astar.comp: SINGLE-THREAD ([numthreads(1,1,1)],
    # the nav_region/nav_polygonize serial mirror) runs the WHOLE A* — connected-components flood ->
    # deterministic start/goal (largest component, lowest-id ties; start=lowest id, goal=farthest by
    # Manhattan centroid distance) -> integer A* (g/f arrays, linear min-scan frontier popping lowest f=g+h
    # tie-break lowest poly id, came_from reconstruct) -> a CORRIDOR (poly-id sequence) + total cost. Edge
    # cost + heuristic = the MANHATTAN distance between integer poly CENTROIDS (admissible+consistent metric
    # -> optimal A*). Copied VERBATIM from engine/nav/navmesh.h::ConnectedComponents/SelectStartGoal/FindPath
    # — PURE INT32 (NO int64; centroids <=32, a single edge cost <=64, the documented overflow bound), so it
    # MSL-generates NATIVELY on Metal. Pure integer SSBO compute + a compute->compute barrier + ReadBuffer ->
    # bit-exact GPU==CPU corridor + cost + start/goal + cross-backend, proven memcmp. NO new RHI; NAV1-NAV4
    # shaders + goldens untouched.
    "shaders/nav_astar.comp.hlsl:cs"
    # Deterministic Fixed-Point Physics Slice 3 (Slice FPX3, the MAKE-OR-BREAK of FLAGSHIP #6):
    # fpx_solve.comp is the PBD POSITIONAL collision-response solver — a SINGLE THREAD ([numthreads(1,1,1)],
    # the mc_scan/vt_alloc allocator pattern) runs `steps` of StepWorld = IntegrateStep then SolveContacts
    # (K=solveIters Gauss-Seidel sweeps: ALL ground contacts then ALL sphere-sphere pair contacts in the
    # FIXED FPX2 pair order). The fxdiv/FxNormalize/ground + sphere-sphere math is copied VERBATIM from
    # engine/sim/fpx.h; contact resolution is inherently sequential -> one thread -> bit-exact GPU==CPU +
    # cross-backend, NO atomics. solveEnabled=0 -> bodies UNCHANGED. UNLIKE the int32 FPX2 broadphase, this
    # is int64 (fxdiv/FxISqrt for the sphere-sphere normal), so it is VULKAN-SPIR-V-ONLY (DXC compiles
    # int64; glslc cannot) and is NOT MSL-gen'd — the Metal --fpx-solve showcase runs the CPU StepWorld
    # instead (byte-identical by construction). Pure integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU
    # body array + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/fpx_solve.comp.hlsl:cs"
    # Deterministic Fixed-Point Physics Slice 4 (Slice FPX4): fpx_orient.comp is the integer QUATERNION
    # ORIENTATION integrator — ONE thread PER BODY (per-body INDEPENDENT, unlike FPX3's single-thread
    # serial solver; 64 threads/group, NO atomics) runs `steps` of IntegrateBodyFull = the FPX1
    # translational integrate + IntegrateOrientation (dq = ω⊗q; orient += 0.5*dq*dt; orient = normalize),
    # the FxQuatMul/FxQuatNormalize/FxISqrt/fxdiv/fxmul math copied VERBATIM from engine/sim/fpx.h, writing
    # gBodies[i] back. enabled=0 -> bodies UNCHANGED. Like fpx_solve.comp this is int64 (the quaternion
    # fxmul products + FxQuatNormalize's fxdiv/FxISqrt), so it is VULKAN-SPIR-V-ONLY (DXC compiles int64;
    # glslc cannot) and is NOT MSL-gen'd — the Metal --fpx-orient showcase runs the CPU IntegrateBodyFull
    # instead (byte-identical by construction). Pure integer SSBO compute + ReadBuffer -> bit-exact
    # GPU==CPU body array + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/fpx_orient.comp.hlsl:cs"
    # Deterministic Convex Rigid-Body Contacts Slice 1 (Slice CX1, the BEACHHEAD of FLAGSHIP #19):
    # convex_sat.comp is the box-box SEPARATING-AXIS TEST — ONE THREAD PER BOX PAIR (per-pair INDEPENDENT,
    # 64 threads/group, NO atomics) runs the 15-axis SAT (3 A-face normals + 3 B-face normals + 9 edge-edge
    # crosses in a FIXED order), copying engine/sim/convex.h::BoxSat's body VERBATIM (the SAME FxRotate
    # box-axes + FxNormalize/FxDot/FxCross int64 ops + the near-zero edge-cross skip + the strict-< min-pen
    # lowest-index tie-break), writing its OWN gResults[pair] SatResult (overlap, axisIndex, penetration,
    # axis-toward-B). satEnabled=0 -> cleared results. Like fpx_solve.comp / fpx_orient.comp this is int64
    # (the FxDot/FxCross/FxNormalize/FxISqrt Q16.16 products), so it is VULKAN-SPIR-V-ONLY (DXC compiles
    # int64; glslc cannot) and is NOT MSL-gen'd — the Metal --convex-sat showcase runs the CPU BoxSat
    # instead (byte-identical by construction). Pure integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU
    # SatResult array + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/convex_sat.comp.hlsl:cs"
    # General Convex-Hull Contacts Slice 1 (Slice GJ1, the BEACHHEAD of FLAGSHIP #22): gjk_support.comp is the
    # convex-hull SUPPORT FUNCTION — ONE THREAD PER (hull, direction) QUERY (per-query INDEPENDENT, 64
    # threads/group, NO atomics) runs the world-space support, copying engine/sim/gjk.h::Support's body
    # VERBATIM (the SAME conjugate-rotate the direction into local, the SAME FIXED strict-greater lowest-index
    # argmax SupportLocal, the SAME FxRotate out + pos), writing its OWN gOut[query] support vertex + extent.
    # queryEnabled=0 -> cleared support. Like convex_sat.comp this is int64 (the FxDot/FxRotate Q16.16
    # products), so it is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the
    # Metal --gjk-support showcase runs the CPU gjk::Support instead (byte-identical by construction). Pure
    # integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU support array + cross-backend, proven memcmp.
    # NO new RHI; existing shaders + goldens untouched.
    "shaders/gjk_support.comp.hlsl:cs"
    # General Convex-Hull Contacts Slice 2 (Slice GJ2): gjk_distance.comp is the GJK ALGORITHM pass — ONE
    # THREAD PER HULL-PAIR (per-pair INDEPENDENT, 64 threads/group, NO atomics) runs the full GJK simplex
    # evolution (overlap + closest distance + witness points), copying gjk.h::Gjk VERBATIM. int64 (the
    # sub-distance fxdiv + FxDot/FxCross/FxLength/FxNormalize/FxRotate products) -> VULKAN-SPIR-V-ONLY (DXC
    # compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --gjk-distance showcase runs the CPU
    # gjk::Gjk instead (byte-identical by construction). Pure integer SSBO compute + ReadBuffer -> bit-exact
    # GPU==CPU GjkResult array + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/gjk_distance.comp.hlsl:cs"
    # General Convex-Hull Contacts Slice 3 (Slice GJ3, THE CRUX): gjk_epa.comp is the EPA ALGORITHM pass —
    # ONE THREAD PER OVERLAPPING HULL-PAIR (per-pair INDEPENDENT, 64 threads/group, NO atomics) runs the full
    # EPA polytope expansion (penetration depth + contact normal A->B + contact points), seeded from the
    # precomputed terminal GJK simplex, copying gjk.h::Epa VERBATIM (the seed-to-tetra, the OUTWARD-winding
    # face build, the min-dist lowest-index closest-face selection, the visible-face removal + edge-cancel
    # horizon + fixed-order re-triangulation, the barycentric contact recovery). int64 (the FxNormalize/FxDot/
    # FxCross/fxdiv products) -> VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd —
    # the Metal --gjk-epa showcase runs the CPU gjk::Epa instead (byte-identical by construction). Pure integer
    # SSBO compute + ReadBuffer -> bit-exact GPU==CPU EpaResult array + cross-backend, proven memcmp. NO new
    # RHI; existing shaders + goldens untouched.
    "shaders/gjk_epa.comp.hlsl:cs"
    # Hull Narrowphase Hardening Slice 3 (Slice MF3, the 3rd slice of FLAGSHIP #25: DETERMINISTIC HULL
    # NARROWPHASE HARDENING, hf::sim::manifold): hull_manifold.comp is THE MULTI-POINT MANIFOLD GPU generator —
    # ONE THREAD PER PAIR (per-pair INDEPENDENT, 64 threads/group, NO atomics) runs gjk::Gjk -> gjk::Epa -> MF2's
    # HullManifoldFromEpa (the Sutherland-Hodgman hull-face clip), copying manifold.h::HullContactMulti's call
    # chain VERBATIM (the SAME int64 FxDot/FxCross/fxdiv/fxmul ops, the SAME fixed orders + strict-integer
    # tie-breaks), writing its OWN convex::ContactManifold slot (count 1-4; separated -> count 0). Like
    # gjk_epa.comp / convex_manifold.comp this is int64 (the GJK sub-distance + the EPA face math + the SH clip
    # Q16.16 products), so it is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the
    # Metal --mf3-manifold showcase runs the CPU manifold::HullContactMulti instead (byte-identical by
    # construction, the gjk_epa.comp/convex_manifold.comp split). Per-pair BOUNDED work (GJK <= kGjkMaxIter, EPA
    # <= kEpaMaxIter, clip <= 8 verts) so a modest battery is TDR-safe. Pure integer SSBO compute + ReadBuffer ->
    # bit-exact GPU==CPU ContactManifold array + cross-backend, proven memcmp. NO new RHI.
    "shaders/hull_manifold.comp.hlsl:cs"
    # Deterministic Integer CCD Slice 1 (Slice CD1, the BEACHHEAD of FLAGSHIP #24): ccd_toi.comp is the
    # TIME-OF-IMPACT pass — ONE THREAD PER CANDIDATE PAIR (per-pair INDEPENDENT, 64 threads/group, NO atomics)
    # runs the full conservative-advancement TOI loop (an embedded gjk::Gjk closest-distance query + the
    # fxdiv(gap, closingSpeedBound) advance + IntegrateBodyFull sub-step with ZERO gravity, until the gap closes
    # or t>=dt or the fixed iteration bound), copying ccd.h::ConservativeAdvance VERBATIM. int64 (the GJK
    # sub-distance fxdiv + FxDot/FxCross/FxLength/FxNormalize/FxRotate/FxQuat products) -> VULKAN-SPIR-V-ONLY
    # (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --ccd-toi showcase runs the CPU
    # ccd::ConservativeAdvance instead (byte-identical by construction). Pure integer SSBO compute + ReadBuffer
    # -> bit-exact GPU==CPU FxToi array + cross-backend, proven memcmp. NO new RHI; existing shaders + goldens
    # untouched.
    "shaders/ccd_toi.comp.hlsl:cs"
    # Deterministic Integer CCD Slice 2 (Slice CD2, FLAGSHIP #24): THE SWEPT-AABB BROADPHASE — the 27-cell-stencil
    # candidate-pair generator over the host-precomputed SWEPT AABBs (each body's box at the tick start UNION its
    # box at the integrated end pose), so a FAST mover's sweep overlaps obstacles the discrete (instantaneous-AABB)
    # broadphase MISSES. The swept-AABB precompute (integrate end pose + 2x BuildHullAabb + union — the int64 part)
    # is done HOST-side; these THREE shaders take the int32 lo/hi swept AABBs + the BP1 grid as SSBO input and do
    # the PURE INT32 grid+pair work: ccd_swept_count.comp (ONE thread per body i: scan i's 3x3x3 stencil over the
    # AABB-centre cell table, count j>i with the six-compare AabbOverlap over lo/hi), ccd_swept_scan.comp
    # (SINGLE-THREAD exclusive prefix-sum -> perBodyOffset + sentinel), ccd_swept_emit.comp (SINGLE-THREAD
    # ascending-body scatter of fpx::FxPair{i,j} into each body's disjoint slice — the per-i emit-order DET-CRUX).
    # ALL PURE INT32 (FloorDiv + integer compares + an ascending scatter, NO int64/fxmul/sqrt) so these MSL-GENERATE
    # NATIVELY (in hf_gen_msl) and Metal dispatches the GPU passes. ReadBuffer reads pairsOut + offsets, PROVEN
    # BIT-EXACT vs the CPU ccd.h::BuildSweptBroadphasePairs (memcmp) + the swept⊇discrete superset proof. NO new RHI.
    "shaders/ccd_swept_count.comp.hlsl:cs"
    "shaders/ccd_swept_scan.comp.hlsl:cs"
    "shaders/ccd_swept_emit.comp.hlsl:cs"
    # Deterministic Convex Rigid-Body Contacts Slice 2 (Slice CX2): convex_manifold.comp is the box-box
    # CONTACT MANIFOLD pass — ONE THREAD PER BOX PAIR (per-pair INDEPENDENT, 64 threads/group, NO atomics)
    # runs the 15-axis BoxSat then BuildManifold (the reference/incident face clip via Sutherland-Hodgman in
    # a FIXED corner + plane order + the deepest+clip-order reduction, OR the edge-edge single closest-point
    # contact), copying engine/sim/convex.h::BuildManifold + ClosestPointsOnSegments VERBATIM, writing its
    # OWN gManifolds[pair] ManifoldGpu (count, 4 points, 4 depths, normal). manifoldEnabled=0 -> cleared.
    # Like convex_sat.comp this is int64 (the FxDot/FxCross/FxNormalize/fxdiv Q16.16 products), so it is
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --convex-manifold
    # showcase runs the CPU BuildManifold instead (byte-identical by construction). Pure integer SSBO compute
    # + ReadBuffer -> bit-exact GPU==CPU ManifoldGpu array + cross-backend, proven memcmp. NO new RHI.
    "shaders/convex_manifold.comp.hlsl:cs"
    # Deterministic Convex Rigid-Body Contacts Slice 3 (Slice CX3, THE NEW-PHYSICS BEAT): convex_solve.comp
    # is the ANGULAR CONTACT IMPULSE pass — ONE THREAD PER BOX PAIR (per-pair INDEPENDENT, 64 threads/group,
    # NO atomics) runs BoxSat -> BuildManifold -> the box inertia tensors (FxBoxInvInertiaBody +
    # WorldInvInertia) -> SolveManifoldImpulse (the FIXED Gauss-Seidel iters×points double loop with the
    # A->B normal sign-correction), copying engine/sim/convex.h::ResolveContactPair VERBATIM, writing its OWN
    # two resolved bodies gResults[pair*2 + 0/1] (the impulse drives BOTH linear vel AND angVel — the first
    # contact-driven spin in the engine). solveEnabled=0 -> bodies UNCHANGED. Like convex_manifold.comp this
    # is int64 (the inertia fxdiv + the FxDot/FxCross/FxMat3MulVec Q16.16 products), so it is VULKAN-SPIR-V-
    # ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --convex-tumble runs the CPU
    # ResolveContactPair instead (byte-identical by construction). Pure integer SSBO compute + ReadBuffer ->
    # bit-exact GPU==CPU resolved-body array + cross-backend, proven memcmp. NO new RHI. VELOCITY-ONLY.
    "shaders/convex_solve.comp.hlsl:cs"
    # Deterministic Convex Rigid-Body Contacts Slice 4 (Slice CX4): convex_step.comp is THE FULL CONVEX STEP
    # — ONE THREAD ([numthreads(1,1,1)]) runs the WHOLE StepConvexWorldN over the small body set (all-pairs
    # over the few boxes): K ticks of predict-integrate -> all-pairs narrowphase -> world Gauss-Seidel
    # impulse solve -> POSITION de-penetration (the NEW bit vs CX3's velocity-only), copying
    # engine/sim/convex.h::StepConvexWorld/StepConvexWorldN VERBATIM (the SAME fixed body+pair orders, the
    # SAME face-preferred BoxSatStable, the SAME split-by-invMass linear de-pen with slop+beta, the SAME
    # per-tick damping). stepEnabled=0 -> bodies UNCHANGED. Like convex_solve.comp this is int64 (the
    # inertia fxdiv + the FxDot/FxCross/FxMat3MulVec/quaternion Q16.16 products), so it is VULKAN-SPIR-V-ONLY
    # (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --convex-stack runs the CPU
    # StepConvexWorldN instead (byte-identical by construction). Single-thread because the world step is
    # order-dependent (the Gauss-Seidel + in-place de-pen). Pure integer SSBO compute + ReadBuffer ->
    # bit-exact GPU==CPU final body world + cross-backend, proven memcmp. NO new RHI. THE SETTLING STACK.
    "shaders/convex_step.comp.hlsl:cs"
    # General Convex-Hull Contacts Slice 4 (Slice GJ4, THE NEW-PHYSICS BEAT): hull_step.comp is THE HULL WORLD
    # STEP — ONE THREAD ([numthreads(1,1,1)]) runs the WHOLE StepHullWorldN over the small hull set (all-pairs
    # over the few hulls): K ticks of predict-integrate -> all-pairs GJK/EPA narrowphase (HullContact =
    # Gjk->Epa->ContactManifold) -> world Gauss-Seidel impulse -> POSITION de-penetration, copying
    # engine/sim/gjk.h::StepHullWorld/StepHullWorldN VERBATIM (the StepConvexWorld 5-pass shell with the ONLY
    # swap being BoxSatStable -> HullContact, the SAME fixed body+pair orders, the SAME split-by-invMass linear
    # de-pen with slop+beta, the SAME per-tick damping). So arbitrary convex polyhedra (tetra/octa/wedge/box)
    # integrate, collide, and SETTLE (a tetra resting on its triangular FACE — physics the box-only SAT can't
    # represent). stepEnabled=0 -> bodies UNCHANGED. Like convex_step.comp / gjk_epa.comp this is int64 (the
    # GJK/EPA support + the inertia fxdiv + the FxDot/FxCross/FxRotate/quaternion Q16.16 products), so it is
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --gjk-settle runs
    # the CPU StepHullWorldN instead (byte-identical by construction). Single-thread because the world step is
    # order-dependent (the Gauss-Seidel + in-place de-pen). Pure integer SSBO compute + ReadBuffer -> bit-exact
    # GPU==CPU final body world + cross-backend, proven memcmp. NO new RHI. THE SETTLING HULL WORLD.
    "shaders/hull_step.comp.hlsl:cs"
    # Hull Narrowphase Hardening Slice 4 (Slice MF4, THE NEW-PHYSICS MONEY BEAT — the HEADLINE of FLAGSHIP #25):
    # hull_step_hardened.comp is THE HARDENED HULL WORLD STEP — ONE THREAD ([numthreads(1,1,1)]) runs the WHOLE
    # StepHullWorldHardenedN over the small hull set: K ticks of predict-integrate -> all-pairs narrowphase ->
    # world Gauss-Seidel impulse -> POSITION de-penetration, copying engine/sim/manifold.h::StepHullWorldHardened
    # VERBATIM. It IS the gjk::StepHullWorld shell (== hull_step.comp) with EXACTLY TWO swaps and NOTHING else:
    # (step 2 inertia) the diagonal FxHullInvInertiaBody+WorldInvInertia -> the FULL signed-tetra
    # FxHullInertiaBodyFull+WorldInvInertiaFull (R·M·Rᵀ); (steps 3+4 contact) the single-point HullContact ->
    # the MF2/MF3 multi-point HullContactMulti. So a box dropped FLAT (tilted) onto a static box SETTLES TO REST
    # where the single-point hull_step leaves it TEETERING. The GPU final body world is memcmp'd BIT-EXACT vs the
    # CPU manifold::StepHullWorldHardenedN. CHUNKED 1 tick/dispatch (TDR-safe — heavier than the single-point
    # step). Like hull_step.comp/hull_manifold.comp this is int64 (the GJK/EPA + the inertia covariance + the
    # 3x3 symmetric inverse + the SH clip), so it is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is
    # NOT MSL-gen'd — the Metal --mf4-stack runs the CPU StepHullWorldHardenedN (byte-identical by construction).
    # Single-thread because the world step is order-dependent. NO new RHI. THE RESTACKED-STABILITY STEP.
    "shaders/hull_step_hardened.comp.hlsl:cs"
    # Deterministic Integer Broadphase Slice 3 (Slice BP3, THE SCALING BEAT): broad_convex_step.comp is THE
    # BROADPHASE-DRIVEN BOX WORLD STEP — ONE THREAD ([numthreads(1,1,1)]) runs the WHOLE StepConvexWorldBPN over
    # a LARGE body set (a static floor + 256+ dynamic boxes): K ticks of predict-integrate -> per-tick
    # re-broadphase (the candidate pairs, AABB-culled, visited in (i,j)-ASCENDING order) -> world Gauss-Seidel
    # impulse over the candidates -> POSITION de-penetration over the candidates, copying
    # engine/sim/broad.h::StepConvexWorldBP/StepConvexWorldBPN (the convex_step shell with the all-pairs
    # solve/de-pen loops GATED on the fpx::AabbOverlap broadphase predicate -> the SAME sorted candidate set as
    # the CPU grid+sort, BP2 equivalence). The make-or-break: the GPU final body world is BIT-EXACT vs the CPU
    # broad::StepConvexWorldBPN, which is itself BYTE-IDENTICAL to the all-pairs convex::StepConvexWorldN (the
    # SCALE proof — the broadphase is bit-transparent). stepEnabled=0 -> bodies UNCHANGED. Like convex_step.comp
    # this is int64 (the inertia fxdiv + the FxDot/FxCross/FxMat3MulVec/quaternion Q16.16 products), so it is
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --broad-convex runs
    # the CPU StepConvexWorldBPN instead (byte-identical by construction). Single-thread because the world step
    # is order-dependent (the Gauss-Seidel + in-place de-pen). The host CHUNKS the K ticks into bounded
    # tick-batches with a ComputeToComputeBarrier between (the GJ4 TDR fix; bit-identical to one big dispatch).
    # Pure integer SSBO compute + ReadBuffer. NO new RHI. THE BROADPHASE-DRIVEN SETTLING PILE.
    "shaders/broad_convex_step.comp.hlsl:cs"
    # Deterministic Integer Broadphase Slice 4 (Slice BP4, THE NEW-CAPABILITY BEAT): broad_hull_step.comp is THE
    # BROADPHASE-DRIVEN HULL WORLD STEP — the direct HULL TWIN of broad_convex_step (the GJ4 hull_step shell with
    # the all-pairs solve/de-pen loops GATED on the hull-AABB broadphase predicate). ONE THREAD
    # ([numthreads(1,1,1)]) runs the WHOLE StepHullWorldBPN over a MODERATE mixed-hull set (a static floor +
    # ~48-64 dynamic tetra/octa/wedge/box): K ticks of predict-integrate -> per-tick re-broadphase (the
    # hull-AABB candidate pairs, BuildHullAabb = the min/max over six Support(+/-X/Y/Z) queries, visited in
    # (i,j)-ASCENDING order via a cube-proxy AABB gate that reproduces the CPU's sorted candidate list as a
    # superset whose extras are HullContact no-ops) -> world Gauss-Seidel GJK/EPA impulse over the candidates ->
    # POSITION de-penetration over the candidates, copying engine/sim/broad.h::StepHullWorldBP/StepHullWorldBPN.
    # The make-or-break: the GPU final body world is BIT-EXACT vs the CPU broad::StepHullWorldBPN, which is itself
    # BYTE-IDENTICAL to the all-pairs gjk::StepHullWorldN (the SCALE proof — the broadphase is bit-transparent
    # for hulls too). stepEnabled=0 -> bodies UNCHANGED. Like hull_step.comp this is int64 (the GJK/EPA support +
    # the inertia fxdiv + the FxDot/FxCross/FxRotate/quaternion products + the Support-based hull AABB), so it is
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --broad-hull runs
    # the CPU StepHullWorldBPN instead (byte-identical by construction). Single-thread because the world step is
    # order-dependent (the Gauss-Seidel + in-place de-pen). The host CHUNKS the K ticks at 1 TICK/DISPATCH from
    # the start (the hull step is heavier than the box step; a 1-tick dispatch is provably << the ~2s TDR
    # watchdog) with a ComputeToComputeBarrier between (bit-identical to one big dispatch). Pure integer SSBO
    # compute + ReadBuffer. NO new RHI. THE BROADPHASE-DRIVEN SETTLING MIXED-HULL PILE.
    "shaders/broad_hull_step.comp.hlsl:cs"
    # Deterministic Integer CCD Slice 3 (Slice CD3, THE MECHANISM BEAT): ccd_step.comp is THE SUBSTEPPED CCD
    # WORLD STEP — ONE THREAD ([numthreads(1,1,1)]) runs the WHOLE StepHullWorldCCDN over a MODERATE scene (a
    # fast mover aimed at a thin wall + a few slow settling hulls): each tick a bounded substep loop that
    # swept-broadphases the candidates (all-pairs swept-AABB gate reproducing the CPU grid set), per-pair
    # conservative-advancement TOI -> the GLOBAL-MIN TOI, advances ALL dynamic bodies to that earliest impact,
    # then resolves the touching pairs via the frozen GJK/EPA narrowphase + impulse + de-pen, consumes the
    # remaining dt and repeats — so a fast body STOPS at the wall instead of tunneling through it. Copies
    # engine/sim/ccd.h::StepHullWorldCCD/StepHullWorldCCDN VERBATIM; the GPU final body world is BIT-EXACT vs the
    # CPU ccd::StepHullWorldCCDN. Like broad_hull_step.comp this is int64 (the embedded Gjk TOI loop + the GJK/EPA
    # resolve + the Support-based swept AABB), so it is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and
    # is NOT MSL-gen'd — the Metal --ccd-step runs the CPU StepHullWorldCCDN instead (byte-identical by
    # construction). Single-thread because the substepped world step is order-dependent. The host CHUNKS the K
    # ticks at 1 TICK/DISPATCH from the start (the heaviest single-thread dispatch in the suite; a 1-tick dispatch
    # is provably << the ~2s TDR watchdog) with a ComputeToComputeBarrier between (bit-identical to one big
    # dispatch). Pure integer SSBO compute + ReadBuffer. NO new RHI. THE NO-TUNNEL SUBSTEPPED CCD WORLD STEP.
    "shaders/ccd_step.comp.hlsl:cs"
    # Deterministic Contact Friction Slice 1 (Slice FC1, the BEACHHEAD of FLAGSHIP #20: DETERMINISTIC
    # TANGENTIAL CONTACT FRICTION, hf::sim::fric): fric_basis.comp is THE TANGENT BASIS pass — ONE THREAD PER
    # INPUT NORMAL (per-normal INDEPENDENT, 64 threads/group, NO atomics) runs the fixed integer Gram-Schmidt
    # (least-aligned cardinal axis -> project off n -> FxNormalize -> FxCross(n,t1)), copying
    # engine/sim/fric.h::MakeTangentBasis's body VERBATIM (the SAME LeastAlignedAxis argmin + lowest-index
    # tie-break + the SAME FxDot/FxCross/FxNormalize int64 ops), writing its OWN gBases[normal] TangentBasis
    # (t1.xyz, t2.xyz). basisEnabled=0 -> cleared bases. Like convex_sat.comp / fpx_solve.comp this is int64
    # (the FxDot/FxCross/FxNormalize/FxISqrt Q16.16 products), so it is VULKAN-SPIR-V-ONLY (DXC compiles
    # int64; glslc cannot) and is NOT MSL-gen'd — the Metal --fric-basis showcase runs the CPU
    # MakeTangentBasis instead (byte-identical by construction). Pure integer SSBO compute + ReadBuffer ->
    # bit-exact GPU==CPU TangentBasis array + cross-backend, proven memcmp. NO new RHI.
    "shaders/fric_basis.comp.hlsl:cs"
    # Deterministic Contact Friction Slice 2 (Slice FC2): fric_points.comp is THE FRICTION-AUGMENTED MANIFOLD
    # POINT pass — ONE THREAD PER BOX PAIR (per-pair INDEPENDENT, 64 threads/group, NO atomics) runs
    # BoxSatStable (the CX4 face-preference SAT) then BuildManifold (the CX2 contact manifold), pairs every
    # contact point with the FC1 tangent basis MakeTangentBasis(nAB) + the A->B normal sign-correction (ONCE)
    # + ZEROED impulse accumulators, writing its OWN gManifolds[pair] FrictionManifoldGpu (count + 4
    # FrictionPoint). pointsEnabled=0 -> cleared. Copies engine/sim/convex.h::BoxSatStable/BuildManifold +
    # engine/sim/fric.h::BuildFrictionPoints/MakeTangentBasis VERBATIM. Like convex_manifold.comp /
    # fric_basis.comp this is int64 (the clip fxdiv + the FxDot/FxCross/FxNormalize Q16.16 products), so it is
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --fric-points
    # showcase runs the CPU BuildFrictionPoints instead (byte-identical by construction). Pure integer SSBO
    # compute + ReadBuffer -> bit-exact GPU==CPU FrictionManifold array + cross-backend, proven memcmp. NO new RHI.
    "shaders/fric_points.comp.hlsl:cs"
    # Deterministic Contact Friction Slice 3 (Slice FC3): fric_solve.comp is THE CONE-CLAMPED TANGENT-IMPULSE
    # SOLVER pass — ONE THREAD PER BOX PAIR (per-pair INDEPENDENT, 64 threads/group, NO atomics) runs
    # BoxSatStable (the CX4 face-preference SAT) then BuildManifold (the CX2 manifold) then the FC1 tangent
    # basis then the box inertia tensors then the combined NORMAL + cone-clamped TANGENT Gauss-Seidel impulse
    # solve (the normal part reproduces convex_solve.comp's SolveManifoldImpulse VERBATIM so mu=0 == CX3; the
    # tangent part clamps jt to +-mu*jn — the Coulomb cone — applied to linear+angular velocity), writing its
    # OWN two resolved bodies. solveEnabled=0 -> bodies UNCHANGED. Copies engine/sim/convex.h::BoxSatStable/
    # BuildManifold/FxBoxInvInertiaBody/WorldInvInertia + engine/sim/fric.h::MakeTangentBasis/
    # SolveFrictionImpulse/ResolveContactFriction VERBATIM. Like convex_solve.comp / fric_points.comp this is
    # int64 (the inertia fxdiv + the FxDot/FxCross/FxMat3MulVec Q16.16 products), so it is VULKAN-SPIR-V-ONLY
    # (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --fric-solve showcase runs the CPU
    # ResolveContactFriction instead (byte-identical by construction). Pure integer SSBO compute + ReadBuffer
    # -> bit-exact GPU==CPU resolved bodies + cross-backend, proven memcmp. NO new RHI.
    "shaders/fric_solve.comp.hlsl:cs"
    # Deterministic Contact Friction Slice 4 (Slice FC4): fric_step.comp is THE FRICTION-LOCKED WORLD STEP
    # (the money-physics beat) — ONE thread ([numthreads(1,1,1)]) runs the WHOLE StepFrictionWorldN over the
    # small body set (predict-integrate -> all-pairs narrowphase -> world Gauss-Seidel impulse with the FC3
    # normal + cone-clamped TANGENT friction solve swapped in for CX4's normal-only SolveManifoldImpulse ->
    # POSITION de-penetration, K ticks), copying engine/sim/fric.h::StepFrictionWorld/StepFrictionWorldN
    # VERBATIM (the ONLY swap vs convex_step.comp is the impulse pass). So a box GRIPS/SLIDES on a ramp and a
    # stack stands at angDamp=kOne (friction holds it). stepEnabled=0 -> bodies UNCHANGED. GPU final body world
    # memcmp'd BIT-EXACT vs the CPU fric::StepFrictionWorldN. Like convex_step.comp / fric_solve.comp this is
    # int64 (the inertia fxdiv + the FxDot/FxCross/FxMat3MulVec/quaternion Q16.16 products), so it is
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --fric-ramp /
    # --fric-stack showcase runs the CPU StepFrictionWorldN instead (byte-identical by construction). Pure
    # integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU + cross-backend, proven memcmp. NO new RHI.
    "shaders/fric_step.comp.hlsl:cs"
    # Deterministic Persistent Contacts Slice 1 (Slice PS1, the BEACHHEAD of FLAGSHIP #21: DETERMINISTIC
    # WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, hf::sim::persist): persist_key.comp is THE CONTACT
    # FEATURE ID pass — ONE THREAD PER CONTACT POINT (per-contact INDEPENDENT, 64 threads/group, NO atomics)
    # runs the order-normalized integer key (bodyA<bodyB swap) + the fixed (bodyA<<20)^(bodyB<<8)^(axisIndex
    # <<4)^featureIndex packing + avalanche xor-shift hash, copying engine/sim/persist.h::MakeContactKey +
    # ContactKeyHash VERBATIM, writing its OWN gKeys[contact] ContactKey + gHashes[contact] hash.
    # keyEnabled=0 -> cleared key + hash 0. UNLIKE convex_*/fric_* (int64 -> Vulkan-only) the contact KEY is
    # PURE INT32 (compares + shifts + xors, NO Q16.16 products, NO int64, NO float), so this shader IS MSL-gen'd
    # (in the metal_headless hf_gen_msl list — the fract_emit_count / boids_cell_count tier) — a TRUE GPU pass on
    # BOTH backends with strict zero-differing-pixel cross-vendor (the Metal --persist-key runs THIS shader, NOT
    # a CPU reference). The Vulkan --persist-key-shot still memcmp's GPU ContactKey[]+hash[] BIT-EXACT vs the CPU
    # MakeContactKey/ContactKeyHash. Pure integer SSBO compute + ReadBuffer. NO new RHI; existing shaders +
    # goldens untouched. persist.h is a NEW additive sibling (#includes sim/fric.h read-only).
    "shaders/persist_key.comp.hlsl:cs"
    # Deterministic Persistent Contacts Slice 2 (Slice PS2): persist_cache.comp is THE PERSISTENT MANIFOLD
    # CACHE pass — ONE THREAD PER BOX PAIR (per-pair INDEPENDENT, 64 threads/group, NO atomics) RECOMPUTES the
    # keyed friction manifold (BoxSatStable -> BuildManifold -> the FC1 tangent basis + the PS1 ContactKey per
    # point, copying engine/sim/convex.h + engine/sim/fric.h::BuildFrictionPoints + engine/sim/persist.h::
    # MakeContactKey VERBATIM) then MATCHES the persistent cache (a small SSBO scanned in FIXED order, copying
    # persist.h::MatchCache VERBATIM) so a matched point INHERITS its prior normal+tangent impulses, writing its
    # OWN gManifolds[pair] KeyedManifoldGpu (count + 4 FricPoint + 4 ContactKey). cacheEnabled=0 -> cleared.
    # UNLIKE PS1's persist_key.comp (the KEY alone is pure int32 -> MSL-native) the MANIFOLD this caches is int64
    # (the clip fxdiv + the FxDot/FxCross/FxNormalize Q16.16 products), so this shader is VULKAN-SPIR-V-ONLY
    # (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --persist-cache showcase runs the CPU
    # persist::BuildKeyedManifold + MatchCache instead (byte-identical by construction, the fric_points.comp
    # convention). Pure integer SSBO compute + ReadBuffer -> bit-exact GPU==CPU inherited-impulse KeyedManifold
    # array + cross-backend, proven memcmp. NO new RHI; PS1 + existing shaders + goldens untouched.
    "shaders/persist_cache.comp.hlsl:cs"
    # Deterministic Persistent Contacts Slice 3 (Slice PS3): persist_warm.comp is THE WARM-STARTED CONE SOLVER
    # — ONE thread ([numthreads(1,1,1)]) runs the WHOLE StepWarmWorldN over the small body set (predict-integrate
    # -> all-pairs BuildKeyedManifold + MatchCache (seed the accumulators from a per-tick persistent CACHE keyed
    # by the PS1 ContactKey) + the ACCUMULATED SolveFrictionWarm (prime once + solveIters accumulated cone sweeps
    # applying only the DELTA) -> POSITION de-penetration -> rebuild the cache for next tick, K ticks), copying
    # engine/sim/persist.h::StepWarmWorld/StepWarmWorldN + SolveFrictionWarm VERBATIM. The warm cache persists
    # across ticks so a resting stack converges tighter at fixed iterations. stepEnabled=0 -> bodies UNCHANGED.
    # GPU final body world memcmp'd BIT-EXACT vs the CPU persist::StepWarmWorldN. Like fric_step.comp this is
    # int64 (the inertia fxdiv + the FxDot/FxCross/FxMat3MulVec/quaternion Q16.16 products), so it is
    # VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --persist-warm
    # showcase runs the CPU StepWarmWorldN instead (byte-identical by construction). Pure integer SSBO compute +
    # ReadBuffer -> bit-exact GPU==CPU + cross-backend, proven memcmp. NO new RHI; PS1/PS2 + existing shaders +
    # goldens untouched. persist.h is APPEND-only.
    "shaders/persist_warm.comp.hlsl:cs"
    # Deterministic Persistent Contacts Slice 4 (Slice PS4): persist_sleep.comp is SLEEPING ISLANDS (THE NEW
    # PHYSICS) — ONE thread ([numthreads(1,1,1)]) runs the WHOLE StepWarmSleepWorldN over the small body set: the
    # PS3 warm step PLUS the per-body integer KineticEnergy = FxLength(vel)+FxLength(angVel), the fixed wake/sleep
    # HYSTERESIS quietTicks update, the all-pairs BoxSatStable contact-adjacency + the fixed-order union-find-free
    # ISLAND wakefulness propagation, the FREEZE of asleep bodies (vel/angVel zeroed -> exactly zero residual),
    # and the SKIP of integrate + solve + de-pen for sleeping bodies/islands (the sleeping pairs keep their prior
    # cache entries), K ticks, copying engine/sim/persist.h::StepWarmSleepWorld/StepWarmSleepWorldN VERBATIM.
    # stepEnabled=0 -> bodies UNCHANGED + cleared sleep states. GPU final body world + per-body sleep states
    # memcmp'd BIT-EXACT vs the CPU persist::StepWarmSleepWorldN. Like persist_warm.comp this is int64 (the
    # inertia/quaternion Q16.16 products + the FxLength KE sqrt), so it is VULKAN-SPIR-V-ONLY (DXC compiles int64;
    # glslc cannot) and is NOT MSL-gen'd — the Metal --persist-sleep / --persist-wake showcase runs the CPU
    # StepWarmSleepWorldN instead (byte-identical by construction). Pure integer SSBO compute (4 buffers: bodies,
    # boxes, params, sleep) + ReadBuffer -> bit-exact GPU==CPU + cross-backend, proven memcmp. NO new RHI;
    # PS1/PS2/PS3 + existing shaders + goldens untouched. persist.h is APPEND-only.
    "shaders/persist_sleep.comp.hlsl:cs"
    # Warm-Started Hull Contacts Slice 1 (Slice WH1, the int32 MSL-native BEACHHEAD of FLAGSHIP #26: WARM-
    # STARTED HULL CONTACTS + ROBUST DETERMINISTIC STACKING, hf::sim::warmhull): warmhull_key.comp is THE HULL
    # CONTACT FEATURE ID pass — ONE THREAD PER CONTACT POINT (per-contact INDEPENDENT, 64 threads/group, NO
    # atomics) runs the order-normalized integer key (bodyA<bodyB swap + the refIsA/role fold into refFaceId's
    # high bit) + the fixed (bodyA<<20)^(bodyB<<8)^(refFaceId<<4)^incVertId packing + avalanche xor-shift hash,
    # copying engine/sim/warmhull.h::MakeHullContactKey + HullContactKeyHash VERBATIM, writing its OWN
    # gKeys[contact] HullContactKey + gHashes[contact] hash. keyEnabled=0 -> cleared key + hash 0. The
    # (refFace, incTag) provenance the key packs is produced HOST-SIDE by the int64 manifold + the tagged clip
    # (ClipFaceAgainstFaceTagged, byte-identical to manifold::ClipFaceAgainstFace); the SHADER itself is PURE
    # INT32 (compares + shifts + xors, NO Q16.16 products, NO int64, NO float) — so UNLIKE the int64 hull MATH
    # (hull_manifold / hull_step_hardened, Vulkan-only) this shader IS MSL-gen'd (in the metal_headless
    # hf_gen_msl list — the persist_key tier) — a TRUE GPU pass on BOTH backends with strict zero-differing-pixel
    # cross-vendor (the Metal --wh1-keys runs THIS shader, NOT a CPU reference). The Vulkan --wh1-keys-shot
    # memcmp's GPU HullContactKey[]+hash[] BIT-EXACT vs the CPU MakeHullContactKey/HullContactKeyHash. Pure
    # integer SSBO compute + ReadBuffer. NO new RHI; existing shaders + goldens untouched. warmhull.h is a NEW
    # additive sibling (#includes sim/manifold.h read-only).
    "shaders/warmhull_key.comp.hlsl:cs"
    # Warm-Started Hull Contacts Slice 2 (Slice WH2): warmhull_cache.comp is THE KEYED MANIFOLD + PERSISTENT
    # CACHE pass — ONE THREAD PER HULL PAIR (per-pair INDEPENDENT, 64 threads/group, NO atomics) reads its
    # HOST-BUILT keyed manifold (the int64-derived convex::ContactManifold positions/depths/normal + the WH1
    # HullContactKey per point, computed host-side by manifold::HullContactMulti + warmhull::BuildKeyedHullManifold)
    # then MATCHES the persistent cache (a small SSBO scanned in FIXED order, copying engine/sim/warmhull.h::
    # MatchHullCache + HullContactKeysEqual VERBATIM) so a matched point INHERITS its prior accumulated normal
    # impulse, writing its OWN gManifolds[pair] KeyedHullManifoldGpu (count + 4 ManifoldPoint + normal + 4 key +
    # 4 normalImpulse). cacheEnabled=0 -> cleared. UNLIKE WH1's warmhull_key.comp (the KEY alone is pure int32 ->
    # MSL-native) the MANIFOLD this caches is int64 (manifold::HullContactMulti's GJK/EPA + the SH clip Q16.16
    # products run HOST-SIDE), so this shader is VULKAN-SPIR-V-ONLY (DXC compiles int64 data layouts; glslc cannot)
    # and is NOT MSL-gen'd — the Metal --wh2-cache showcase runs the CPU warmhull::BuildKeyedHullManifold +
    # MatchHullCache instead (byte-identical by construction, the persist_cache.comp convention). Pure integer
    # SSBO compute + ReadBuffer -> bit-exact GPU==CPU inherited-impulse keyed manifold array + cross-backend,
    # proven memcmp. NO new RHI; WH1 + existing shaders + goldens untouched. warmhull.h is APPEND-only.
    "shaders/warmhull_cache.comp.hlsl:cs"
    # Hull Friction + Joints Slice 1 (Slice HF1): hullfric_points.comp is THE TAGGED FRICTION MANIFOLD ON THE EPA
    # NORMAL pass — ONE THREAD PER HULL PAIR (per-pair INDEPENDENT, 64 threads/group, NO atomics) reads its
    # HOST-BUILT warmhull keyed manifold (the int64-derived convex::ContactManifold positions/depths/normal + the WH1
    # HullContactKey per point + the sign-corrected A->B EPA normal, computed host-side by manifold::HullContactMulti
    # + warmhull::BuildKeyedHullManifold) then APPENDS the friction data: the integer tangent basis (t1,t2) via the
    # FC1 fixed Gram-Schmidt (fric::MakeTangentBasis copied VERBATIM: project the least-aligned cardinal axis off n +
    # FxNormalize + n x t1) + the basis-axis index (fric::LeastAlignedAxis) + zeroed per-point accumulators, writing
    # its OWN gManifolds[pair] HullFrictionManifoldGpu. The basis FxNormalize/FxDot/FxCross are int64 (world-scale
    # products overflow int32), so this shader is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT
    # MSL-gen'd — the Metal --hf1-points showcase runs the CPU hullfric::BuildAllHullFrictionManifoldsPairs instead
    # (byte-identical by construction, the warmhull_cache.comp convention), while the Vulkan side carries the GPU==CPU
    # memcmp proof. NO new RHI; warmhull.h/gjk.h/fric.h/persist.h + existing shaders + goldens UNTOUCHED. hullfric.h
    # is a brand-new additive header.
    "shaders/hullfric_points.comp.hlsl:cs"
    # Hull Friction + Joints Slice 2 (Slice HF2): hullfric_warm.comp is THE WARM CONE SOLVER — ONE THREAD
    # ([numthreads(1,1,1)]) runs the WHOLE hullfric.h::StepHullFrictionSolveOnly over the small pair set: per pair
    # in the FIXED i<j order, warm-seed the HOST-BUILT keyed friction manifold from the cache, then run
    # SolveHullFrictionWarm (the accumulated, warm-started, cone-clamped Coulomb friction solve — the
    # persist::SolveFrictionWarm body with the hull manifold::WorldInvInertiaFull tensor: prime the seed once, then
    # cfg.iters Gauss-Seidel sweeps doing the accumulated normal block clamped >= 0 + the two cone-clamped tangent
    # blocks clamped to +/-mu*jn) mutating the bodies IN PLACE so later pairs see earlier updates. HF2 solves a
    # FIXED manifold set in place — NO integrate, NO de-pen (that is HF3). The host builds the keyed friction
    # manifold + the per-body FULL world inverse-inertia tensor (the int64 narrowphase + inertia run host-side); the
    # shader does ONLY the solve. The warm-solve delta math (fxdiv/fxmul/FxDot Q16.16 products) is int64, so this
    # shader is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --hf2-warm
    # showcase runs the CPU hullfric::StepHullFrictionSolveOnly instead (byte-identical by construction, the
    # warmhull_warm.comp convention). The GPU final gBodies + gManifolds (the solved accumulators) memcmp-equals the
    # CPU reference. NO new RHI; hullfric.h is APPEND-only (HF1 byte-frozen); warmhull.h/gjk.h/fric.h/persist.h/
    # manifold.h + existing shaders + goldens UNTOUCHED.
    "shaders/hullfric_warm.comp.hlsl:cs"
    # Hull Friction + Joints Slice 3 (Slice HF3): hullfric_step.comp is THE FRICTION-LOCKED HULL WORLD STEP — ONE
    # THREAD ([numthreads(1,1,1)]) runs the WHOLE hullfric.h::StepWarmFrictionHullWorldN over the small scene: K
    # ticks of the warmhull 5-pass tick — predict-integrate every dynamic body + damping -> FULL world inverse
    # inertia per body -> all-pairs i<j broadphase (build the HF1 friction manifold: the keyed hull manifold + the
    # per-pair tangent basis on the EPA normal + the basis-axis flip field), MatchHullFrictionCache (key + basisAxis)
    # warm-seed, cfg.solveIters sweeps of SolveHullFrictionWarm (the normal-only solve SWAPPED for the cone solver),
    # rewrite the persistent FRICTION cache -> position de-penetration along the A->B normal -> orientation (already
    # in integrate). CHUNKED 1 tick / dispatch (the Windows ~2s TDR rule) over the SAME body + cache buffers. The
    # whole chain (GJK/EPA + SH clip + full inertia + the cone solve) is int64, so this shader is VULKAN-SPIR-V-ONLY
    # (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the Metal --hf3-step runs the CPU
    # hullfric::StepWarmFrictionHullWorldN instead (byte-identical by construction, the warmhull_sleep.comp
    # convention). The GPU final gBodies + gCache memcmp-equals the CPU reference. NO new RHI; hullfric.h is
    # APPEND-only (HF1+HF2 byte-frozen); warmhull.h/gjk.h/fric.h/persist.h/manifold.h + existing shaders + goldens
    # UNTOUCHED.
    "shaders/hullfric_step.comp.hlsl:cs"
    # Hull Friction + Joints Slice 4 (Slice HF4): hulljoint_step.comp is HULL JOINTS COMPOSED — ONE THREAD
    # ([numthreads(1,1,1)]) runs the WHOLE hulljoint.h::StepJointedHullWorldN over the small scene: K ticks of the
    # HF3 friction tick with the joint constraint passes inserted in the joint::StepArticulatedContacts position —
    # predict-integrate -> jointIters Gauss-Seidel sweeps EACH {all SolveBallJoint, then all SolveAngularLimit} (the
    # joint.h bodies copied VERBATIM) -> FULL inertia -> the HF1 friction manifold + cache warm-seed + the cone solve
    # -> position de-pen -> orientation. CHUNKED 1 tick / dispatch (the Windows ~2s TDR rule) over the SAME body +
    # cache buffers. The whole chain (GJK/EPA + SH clip + full inertia + the cone solve + the joint FxRotate/
    # FxQuatMul/FxQuatNormalize) is int64, so this shader is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and
    # is NOT MSL-gen'd — the Metal --hf4-joint runs the CPU hulljoint::StepJointedHullWorldN instead (byte-identical
    # by construction, the hullfric_step.comp convention). The GPU final gBodies + gCache memcmp-equals the CPU
    # reference. NO new RHI; hulljoint.h is a NEW additive header; hullfric.h/joint.h/warmhull.h/gjk.h/fpx.h +
    # existing shaders + goldens UNTOUCHED.
    "shaders/hulljoint_step.comp.hlsl:cs"
    # Warm-Started Hull Contacts Slice 3 (Slice WH3): warmhull_warm.comp is THE ACCUMULATED WARM-STARTED SOLVER
    # — ONE THREAD ([numthreads(1,1,1)]) runs the WHOLE warmhull.h::StepWarmHullWorldN over the small hull set:
    # K ticks of predict-integrate -> FULL inertia -> build-keyed-manifold (the WH1 tagged-clip key derivation
    # ported VERBATIM) -> MATCH the persistent cache (warm-seed each point's accumulated normal impulse) -> the
    # ACCUMULATED warm normal Gauss-Seidel solve (SolveHullManifoldWarm: prime the seed, then DELTA-apply each
    # sweep clamping the ACCUMULATED total >= 0) -> REWRITE the cache (UpdateHullCache, evict absent keys) ->
    # POSITION de-penetration, copying StepWarmHullWorld VERBATIM. It IS hull_step_hardened.comp with EXACTLY
    # step (3) — the impulse solve — replaced by the warm-started accumulated solve over the cache; steps (1)
    # integrate, (2) full inertia, (4) de-pen UNCHANGED. The cache persists ACROSS ticks in gCache (a flat
    # CachedHullContactGpu list + a live count in gParams.c2.w), READ+WRITE, surviving the chunked
    # 1-tick/dispatch boundaries exactly like the body buffer. Chunked 1 tick/dispatch (TDR-safe — HEAVIER than
    # the hardened step). The whole chain is int64 (GJK/EPA + the SH clip + the full inertia + the accumulated
    # solve), so this shader is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot) and is NOT MSL-gen'd — the
    # Metal --wh3-warm runs the CPU StepWarmHullWorldN instead (byte-identical by construction). The GPU final
    # gBodies+gCache memcmp-equals the CPU reference. NO new RHI; warmhull.h is APPEND-only; WH1/WH2 + all other
    # shaders + goldens UNCHANGED.
    "shaders/warmhull_warm.comp.hlsl:cs"
    # Warm-Started Hull Contacts Slice 4 (Slice WH4): warmhull_sleep.comp is SLEEPING ISLANDS -> THE STABLE
    # STACK (the new-physics HEADLINE of FLAGSHIP #26) — ONE THREAD ([numthreads(1,1,1)]) runs the WHOLE
    # warmhull.h::StepWarmSleepHullWorldN over the small tower: K ticks of (1) per-body KineticEnergyHull + the
    # hysteresis lowEnergyTicks update -> (2) the all-pairs HullContactMulti adjacency + the island wakefulness
    # propagation -> (2b) FREEZE every asleep body (zero vel+angVel — exactly zero residual) -> (3) integrate
    # ONLY awake bodies -> (4) the warm accumulated solve + de-penetration over ACTIVE pairs only -> (5) rebuild
    # the cache (sleeping pairs keep their prior entries; active pairs store the fresh accumulators), copying
    # StepWarmSleepHullWorld VERBATIM. It IS warmhull_warm.comp (== StepWarmHullWorld) with the persist.h PS4
    # sleeping-island logic (per-body sleep state in gSleep, READ+WRITE, carried across the chunked
    # 1-tick/dispatch boundaries) layered on. THE HEADLINE: a settled island freezes at EXACTLY zero residual
    # before the within-band residual torque can topple it -> a deterministic STABLE STACK the frozen
    # StepHullWorldHardenedN topples at the same budget. Chunked 1 tick/dispatch (TDR-safe — HEAVIER than WH3, a
    # multi-body tower). The whole chain is int64 (GJK/EPA + the SH clip + the full inertia + the accumulated
    # solve + the FxLength energy sqrt), so this shader is VULKAN-SPIR-V-ONLY (DXC compiles int64; glslc cannot)
    # and is NOT MSL-gen'd — the Metal --wh4-stack runs the CPU StepWarmSleepHullWorldN instead (byte-identical
    # by construction). The GPU final gBodies + gSleep + gCache memcmp-equals the CPU reference. NO new RHI;
    # warmhull.h is APPEND-only; WH1/WH2/WH3 + all other shaders + goldens UNCHANGED.
    "shaders/warmhull_sleep.comp.hlsl:cs"
    # GPU Isosurface Meshing Slice 2 (Slice MC2): mc_count.comp counts the triangles each voxel CELL emits
    # — ONE thread per cell gathers its 8 cube-CORNER scalars, runs CaseIndex (copied VERBATIM from
    # render/mc.h), looks up triCount = gTriCount[caseIdx] from the host-uploaded 256-entry count table
    # (built from render/mc.h::kTriTable), writes gCounts[cell]=triCount (order-independent per-cell write)
    # and InterlockedAdd(gTotal[0], triCount) the grand total (the autoexposure_histogram integer-atomic
    # precedent — commutative -> order-independent -> deterministic). countEnabled=0 -> writes 0, no atomic.
    # Pure integer compute over Storage SSBOs + ReadBuffer -> bit-exact GPU==CPU counts + total +
    # cross-backend, proven memcmp. NO new RHI; existing shaders + goldens untouched.
    "shaders/mc_count.comp.hlsl:cs"
    # GPU Isosurface Meshing Slice 3 (Slice MC3): the prefix-sum compaction + triangle EMISSION — the
    # geometry-generating heart of Marching Cubes. mc_scan.comp is a SINGLE-THREAD ([numthreads(1,1,1)])
    # exclusive prefix-sum of the MC2 per-cell counts -> each cell's write offset (the VT2 vt_alloc
    # single-thread-allocator pattern, GPU==CPU bit-exact). mc_emit.comp is ONE thread per cell: it
    # reclassifies via CaseIndex (VERBATIM render/mc.h), reads triOffset=gOffsets[cell], and walks
    # gTriTable emitting each triangle's 3 edge-MIDPOINT vertices (EdgeMidpoint = the integer SUM of two
    # corner coords in half-units, NO float) + the identity index at its DISJOINT range (race-free, NO
    # atomics). meshEnabled=0 -> emit nothing. Pure integer SSBO compute + compute->compute barrier +
    # ReadBuffer -> bit-exact GPU==CPU vertex+index buffers + cross-backend, proven memcmp. NO new RHI;
    # existing shaders + goldens untouched.
    "shaders/mc_scan.comp.hlsl:cs"
    "shaders/mc_emit.comp.hlsl:cs"
    # GPU Isosurface Meshing Slice 4 (Slice MC4): mc_interp.comp is mc_emit.comp with EdgeInterp instead
    # of EdgeMidpoint — the per-cell triangle emission with FIXED-POINT INTERPOLATED vertex placement (the
    # vertex sits on the ACTUAL isosurface crossing along the edge, t = ((iso-s0)*kSub)/(s1-s0) on the
    # 1/kSub lattice, the SAME truncating integer divide both sides -> bit-exact GPU==CPU + cross-backend;
    # int32 throughout (numerator fits) so NO int64/glslc issue). Reuses mc_scan.comp UNCHANGED. NO new
    # RHI; existing MC1-MC3 shaders + goldens untouched.
    "shaders/mc_interp.comp.hlsl:cs"
    # Nanite Software-Raster RESOLVE blit (Slice SW4): swraster_resolve_blit.frag is a FULLSCREEN pass
    # (reuses post.vert) that unpacks the SW2 software rasterizer's packed depth|id visibility-buffer SSBO
    # into the R32_Uint visId render target the EXISTING visresolve.frag deferred resolve consumes, so the
    # resolve is reused BYTE-FOR-BYTE. Per pixel: idx=y*width+x; out=(packed==0xFFFFFFFF)?0xFFFFFFFF:
    # (packed&0xFFFF) — strip the HIGH-16 depth, keep the LOW-16 visId=(clusterID<<7)|triID. Plain uint32
    # (NO int64, NO integer texture.read — it WRITES an integer SV_Target). Reads the SW SSBO via the
    # BindLightClusters set-3 fragment-storage path (binding 13) + a {width,height} UBO. NO new RHI;
    # existing shaders + goldens untouched. Behind --swraster-resolve-shot.
    "shaders/swraster_resolve_blit.frag.hlsl:ps"
    # Virtual Shadow Maps Slice 2 (Slice VB): vsm_depth.frag renders each resident PHYSICAL page's
    # casters' depth into its atlas tile (paired with shadow_csm.vert: the page's light-space ortho VP in
    # the push constant). Unlike the depth-only CSM caster it renders into a COLOR atlas — SV_Position.z ->
    # grayscale (near=bright) — so the per-tile depth is ReadRenderTarget'd back as the golden. NO new RHI
    # (reuses CreateShadowMap-sized CreateRenderTarget + SetViewport + the integer AllocatePhysicalPages).
    "shaders/vsm_depth.frag.hlsl:ps"
    # Virtual Shadow Maps Slice 3 (Slice VC): lit_vsm.frag is the LIT-PASS VSM indirection sample (the
    # shadowed scene) — a SIBLING of lit_csm.frag. Per receiver pixel it picks the clipmap level (the VA
    # threshold-ladder, verbatim), projects to the virtual page, looks up the physical tile in the
    # indirection SSBO ([[vk::binding(13,3)]], bound via usesLightClusters/BindLightClusters), projects into
    # that page's PageWorldOrtho light-space, and tile-clamped 3x3-PCF-samples the DEPTH atlas (gShadow via
    # SetShadowMap) -> a VSM-shadowed scene. vsmEnabled=0 -> lerp(1,x,0)==1 -> unshadowed BYTE-IDENTICAL.
    # NO new RHI. lit.frag/lit_csm.frag + their goldens untouched (sibling isolation).
    "shaders/lit_vsm.frag.hlsl:ps"
    # Virtual-geometry per-CLUSTER Hi-Z OCCLUSION cull (Slice DU): DT's cluster_cull.comp PLUS CJ's
    # hiz_cull.comp occlusion test. After the frustum test, a frustum-surviving cluster-instance is ALSO
    # tested against a Hi-Z max-depth pyramid (built CPU-side from a depth pre-pass + uploaded; render/hiz.h)
    # and dropped iff its bounding-sphere AABB is fully hidden behind closer geometry; SphereOutside +
    # IsOccluded are copied VERBATIM; SAME ordered single-workgroup prefix-sum compaction. CONSERVATIVE ->
    # occlusion-culled == frustum-only byte-identical. CPU mirror: render/cluster_cull.h::CullClusterInstancesHiZ.
    # NO new RHI; existing shaders + goldens untouched.
    "shaders/cluster_hiz_cull.comp.hlsl:cs"
    # Virtual-geometry discrete cluster-LOD selection (Slice DV): ONE thread/instance computes the squared
    # view distance + SelectLod (copied VERBATIM from render/cluster_lod.h) to pick one of 3 pre-baked LODs
    # by projected screen-space error, then ORDER-compacts the selected LOD's clusters into the MdiCommand
    # buffer + the compacted (model+hashColor) per-draw SSBO + writes the per-instance selected-LOD int (the
    # GPU==CPU bit-exact proof) + the total command count. reuses cluster_viz.vert + meshlet_viz.frag for the
    # draw. CPU mirror: render/cluster_lod.h. NOT in the Metal MSL-gen list (Metal renders the selected set
    # via the CPU bound path; Vulkan carries the GPU==CPU select proof). NO new RHI; goldens untouched.
    "shaders/cluster_lod_select.comp.hlsl:cs"
    # Hi-Z OCCLUSION cull (Slice CJ): extends gpudriven_cull with a hierarchical-Z occlusion test —
    # after the frustum test, an object FULLY HIDDEN behind closer geometry (its screen-rect nearest
    # depth farther than the Hi-Z max everywhere it covers) is dropped from the compacted survivors.
    # The Hi-Z pyramid is built CPU-side (render/hiz.h) from a depth pre-pass + uploaded; the shader
    # runs the SAME IsOccluded math. Render-invariant: occlusion-culled == frustum-only byte-identical.
    "shaders/hiz_cull.comp.hlsl:cs"
    # Clustered Light Culling (Slice CL): cluster_assign.comp assigns each point light to the clusters
    # its hard-radius view-sphere intersects (the SAME ClusterViewAABB + SphereAABBIntersect math as
    # engine/render/cluster.h), writing a DETERMINISTIC ORDERED per-cluster light list. lit_clustered_cl
    # is the clustered lit variant (windowed hard-radius point lights on top of the default sun+IBL),
    # asserted BYTE-IDENTICAL to a brute-force all-lights render. Distinct from Slice AG's clustered
    # path; separate golden (clustered_lights.png); existing shaders + goldens untouched.
    "shaders/cluster_assign.comp.hlsl:cs"
    "shaders/lit_clustered_cl.frag.hlsl:ps"
    # GPU multi-draw-indirect batching (Slice BM): the MDI lit vertex shader reads the per-draw model
    # matrix + material from a storage buffer indexed by gl_DrawID (SPIR-V DrawIndex). VULKAN-ONLY (the
    # DrawIndex builtin needs shaderDrawParameters); shares the existing lit.frag. Not in the Metal MSL
    # gen list (glslc can't lower DrawIndex; the Metal golden uses the per-object path).
    "shaders/lit_mdi.vert.hlsl:vs"
    # Bindless textures (Slice BZ): the bindless lit vertex (lit.vert + a uint texIndex push constant
    # passed flat to the fragment) + fragment (lit.frag with the base-color source swapped for an
    # unbounded gTextures[] array sampled via NonUniformResourceIndex(texIndex), space4). VULKAN-ONLY
    # (DXC emits SPV_EXT_descriptor_indexing / runtimeDescriptorArray); the Metal golden renders the
    # identical scene via the per-material bound path. Not in the Metal MSL gen list.
    "shaders/lit_bindless.vert.hlsl:vs"
    "shaders/lit_bindless.frag.hlsl:ps"
    # Fully-GPU-driven pass (Slice CB, MDI + bindless capstone): the GPU-driven lit vertex reads the
    # per-draw model+material+texIndex from a storage buffer indexed by gl_DrawID (SPIR-V DrawIndex, BM)
    # and passes texIndex flat; the fragment samples the base color from the unbounded gTextures[] array
    # via NonUniformResourceIndex(texIndex) (SPV_EXT_descriptor_indexing, BZ). ONE DrawIndexedMultiIndirect
    # + ONE BindBindlessTextures draws the whole multi-material scene. VULKAN-ONLY (DXC emits BOTH
    # DrawParameters AND descriptor-indexing); the Metal golden renders the identical scene via the
    # per-object bound path. Not in the Metal MSL gen list (glslc can't lower DrawIndex).
    "shaders/lit_gpudriven.vert.hlsl:vs"
    "shaders/lit_gpudriven.frag.hlsl:ps"
    # Screen-space projected decals (Slice BH): reuses the SSAO/SSR g-buffer (view normal + linear
    # depth) + the lit HDR scene. decal.frag reconstructs the world position (view-space recon like
    # SSR, then view->world via invView), projects it into the decal box, and alpha-blends the decal
    # texture over the scene + tonemaps. Existing gbuffer/ssao/ssr/bloom shaders + goldens untouched.
    "shaders/decal.frag.hlsl:ps"
    # Data-driven post-process stack (Slice BN): the configurable ORDERED composite chain. A SEPARATE
    # final pass from post.frag (the default --shot post path is byte-identical + untouched). Reuses
    # post.vert (fullscreen triangle); applies the configured effects (tonemap/colorgrade/chromatic/
    # vignette/grain) in order from a push-constant effect list. Shared math with engine/render/post_stack.h.
    "shaders/post_stack.frag.hlsl:ps"
    # Froxel Volumetric Fog (Slice CS): a true 3D view-space froxel volume (inject -> integrate ->
    # depth-composited apply). froxel_inject.comp writes per-froxel (scatter, extinction) from the
    # render/froxel.h height-density + HG phase; froxel_integrate.comp marches each Z column front-to-back
    # via IntegrateStep into (inScatter, transmittance); froxel_apply.frag (fullscreen, reuses post.vert)
    # samples the integrated volume at each pixel's view-depth slice (ViewZToSlice) and composites
    # scene*T + inScatter. The volume is a FLAT SSBO (no new RHI seam). With baseDensity==0 the apply is a
    # pure pass-through (T==1, inScatter==0) -> byte-identical to the no-fog scene. New golden
    # froxel_fog.png; existing shaders + goldens untouched (incl. the 2D light-shaft volumetric.png).
    "shaders/froxel_inject.comp.hlsl:cs"
    "shaders/froxel_integrate.comp.hlsl:cs"
    "shaders/froxel_apply.frag.hlsl:ps"
    # Probe-Grid Ray-Trace — DDGI beachhead (Slice DH): one thread per probe traces 16 deterministic
    # Fibonacci-sphere rays against the scene's view-space depth field (a FLAT float SSBO read back from
    # the G-buffer, NOT a sampled depth texture) into a flat ProbeRayHit SSBO. FibonacciSphere +
    # ViewToScreenUV/ReconstructViewPos + TraceRayToDepth are copied VERBATIM from engine/render/probe_gi.h
    # (+ ssr.h), so the GPU ray-hit SSBO is BIT-EXACT to the CPU reference over the same depth field. Pure
    # compute + storage buffers (no new RHI seam). The probeCount=0 dispatch-0 leaves the SSBO untouched.
    "shaders/probe_raytrace.comp.hlsl:cs"
    # Probe SH-Encode — DDGI Slice 3 (Slice DJ): one thread per probe loops the FIXED host-precomputed
    # sample set (per cube-texel: the 9 SHBasis9 weights + the solid-angle weight, computed once on the
    # host), reads the captured radiance at each sample from the flat radiance store, and accumulates it
    # into 3rd-order real-SH coefficients (SHEncodeAccumulate/SHNormalize copied VERBATIM from
    # engine/render/probe_sh.h), writing the per-probe ProbeSH SSBO. Because the sample basis weights +
    # solid-angle weights are uploaded as exact bits the GPU + CPU read identically, the GPU SH SSBO is
    # BIT-EXACT to the CPU SH-encode. Pure compute + storage buffers (no new RHI seam). The probeCount=0
    # dispatch-0 leaves the SH SSBO untouched == the cleared upload.
    "shaders/probe_sh_encode.comp.hlsl:cs"
    # Probe Trilinear SH-Interp — DDGI Slice 4 (Slice DL): one thread per QUERY point trilinearly blends
    # the 8 surrounding probes' SH at the query's world position into a single ProbeSH. NearestProbes (the
    # FLOOR-cell 8 corners + polynomial weights) + InterpolateSH (the per-corner fma SH accumulation) are
    # copied VERBATIM from engine/render/probe_gi.h + probe_sh.h (mad for the per-corner product), so the
    # GPU blended-SH SSBO is BIT-EXACT to the CPU InterpolateSH reference. Pure compute + storage buffers
    # (no new RHI seam). The probeCount=0 dispatch-0 leaves the output SSBO untouched == the cleared upload.
    "shaders/probe_interp.comp.hlsl:cs"
    # Per-Probe Distance-Moment Capture — DDGI Visibility Slice 1 (Slice DO): the VISIBILITY data layer.
    # probe_dist.vert is a probe_bake.vert clone that passes the world position + carries probeCentre in the
    # push constant; probe_dist.frag outputs the linear world-distance d = length(worldPos - probeCentre)
    # packed as the two moments float2(d, d*d) (d*d a BARE multiply so it is bit-identical to the CPU
    # reference probedist::MomentsFromDistance). Pure geometry (no lighting), captured into the DD cube RT +
    # read back into a per-probe distance-moment store (the data the next slice DP, Chebyshev occlusion,
    # consumes to kill DDGI leak). Behind --probedist-shot ONLY; lit.frag + every existing pipeline + golden
    # stay byte-identical. NO new RHI (reuses DD's ICubemapTarget / Begin-EndCubemapFace / ReadCubemapFace).
    "shaders/probe_dist.vert.hlsl:vs"
    "shaders/probe_dist.frag.hlsl:ps"
    # Auto-Exposure (Slice CW): histogram eye adaptation. autoexposure_histogram.comp reads the HDR scene
    # color (flat float4 SSBO), computes the Rec.709 Luminance, maps it to a log2-luminance bin via
    # render/auto_exposure.h::LumToBin, and InterlockedAdd(histogram[bin],1) into a bins-entry INTEGER SSBO
    # (order-independent counts -> deterministic). autoexposure_reduce.comp reduces the histogram ->
    # AverageLuminance (black-bin excluded) -> ExposureFromAverage (key-value) -> writes the single exposure
    # float (gated by adaptationEnabled: when false it writes the fixed E0). tonemap_autoexp.frag is the
    # post.frag tonemap with the fixed kExposure replaced by that exposure SSBO value (read via the cluster
    # binding 13/space3 — no new RHI seam). With adaptationEnabled=false the exposure is E0 -> byte-identical
    # to the standard fixed-exposure tonemap. New golden auto_exposure.png; existing post.frag + goldens
    # untouched (a NEW path behind --autoexposure-shot).
    "shaders/autoexposure_histogram.comp.hlsl:cs"
    "shaders/autoexposure_reduce.comp.hlsl:cs"
    "shaders/tonemap_autoexp.frag.hlsl:ps"
)
target_compile_definitions(hello_triangle PRIVATE
  "HF_SHADER_DIR=\"${SHADER_OUT}\"")
# Slice AW: absolute paths to the material graphs the live/runtime path loads + recompiles at runtime.
target_compile_definitions(hello_triangle PRIVATE
  "HF_MAT_JSON=\"${MAT_JSON}\""
  "HF_MAT2_JSON=\"${MAT2_JSON}\""
  "HF_MAT3_JSON=\"${MAT3_JSON}\""
  "HF_MATNORMAL_JSON=\"${MATNORMAL_JSON}\"")
# Absolute path to the glTF showcase model, baked at configure time (repo asset).
target_compile_definitions(hello_triangle PRIVATE
  "HF_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/assets/models/Duck.glb\"")
# Absolute path to the skinned-animation showcase model (Fox.glb: 24-joint skin + animations).
target_compile_definitions(hello_triangle PRIVATE
  "HF_FOX_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/assets/models/Fox.glb\"")
# Absolute path to the full-PBR showcase model (DamagedHelmet.glb: 5-texture metallic-roughness PBR).
target_compile_definitions(hello_triangle PRIVATE
  "HF_HELMET_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/assets/models/DamagedHelmet.glb\"")
# Absolute path to the scene-graph import showcase model (CesiumMilkTruck.glb: node hierarchy +
# multiple meshes/materials; the wheels mesh is placed at two positions via two parent transforms).
target_compile_definitions(hello_triangle PRIVATE
  "HF_TRUCK_MODEL_PATH=\"${CMAKE_SOURCE_DIR}/assets/models/CesiumMilkTruck.glb\"")
# Absolute path to the default data-driven scene description (repo asset).
target_compile_definitions(hello_triangle PRIVATE
  "HF_SCENE_PATH=\"${CMAKE_SOURCE_DIR}/assets/scenes/default.json\"")
# Absolute path to the HDR equirectangular environment map (Slice R: kloofendal clear sky, 1k).
target_compile_definitions(hello_triangle PRIVATE
  "HF_ENV_PATH=\"${CMAKE_SOURCE_DIR}/assets/env/env.hdr\"")
