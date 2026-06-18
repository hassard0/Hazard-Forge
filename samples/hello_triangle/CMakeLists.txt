add_executable(hello_triangle main.cpp)
# Vulkan/Windows sample links the editor shell (Dear ImGui through the RHI); the editor is gated to
# non-Apple in the root CMakeLists, so this link only happens on the Vulkan path.
target_link_libraries(hello_triangle PRIVATE hf_engine hf_editor)
target_compile_definitions(hello_triangle PRIVATE HF_HAS_EDITOR=1)

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
    # Deterministic Fixed-Point Physics Slice 1 (Slice FPX1, the BEACHHEAD of FLAGSHIP #6):
    # fpx_integrate.comp runs the Q16.16 semi-implicit-Euler INTEGRATOR — ONE thread per body runs
    # `steps` iterations of (vel += gravity*dt; pos += vel*dt; ground floor-clamp) on its OWN body, the
    # fxmul ((int64)a*b >> 16, the swraster int64 pattern) + integrate + clamp copied VERBATIM from
    # engine/sim/fpx.h::IntegrateBody, and writes gBodies[i] back (each body independent -> race-free,
    # NO atomics). integrateEnabled=0 -> write the input back UNCHANGED. Pure integer SSBO compute +
    # ReadBuffer -> bit-exact GPU==CPU body array + cross-backend, proven memcmp. NO new RHI; existing
    # shaders + goldens untouched.
    "shaders/fpx_integrate.comp.hlsl:cs"
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
