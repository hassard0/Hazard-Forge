<#
.SYNOPSIS
    Full cross-platform verification for Hazard Forge: Windows/Vulkan + Mac/Metal in one command.

.DESCRIPTION
    Runs the complete verification gate and exits non-zero on ANY failure:

      1. Windows / Vulkan
         - conan install (cppstd=17 + Ninja generator), cmake configure, build, ctest.
         - Plus the Slice-AL introspection JSON golden: an EXACT byte match of the live
           --introspect output for the default scene vs tests/golden/introspect/default_scene.json.
           Backend-agnostic (pure hf_core) so it is verified ONLY here -- the Mac is not needed.
         - All steps run inside a VS BuildTools x64 dev shell so cl/ninja resolve.

      2. Mac / Metal (headless, over SSH on the LAN)
         - tar the repo (excluding build dirs + .git + stray PNGs, KEEPING the tracked goldens),
           scp it to the Mac, extract, configure+build the metal_headless target ONCE, then for
           EACH committed Metal golden run visual_test with its showcase flag and compare
           the output to the matching golden with threshold 0.0 (every pair must be DIFF 0.0000).
           A per-golden table is printed; the Mac portion passes only if ALL 44 diff 0.0000.

    Idempotent and re-runnable: build dirs are reused; the Mac staging dir is recreated each run.

.PREREQUISITES
      Windows:
        - Visual Studio 2022 BuildTools with the C++ x64 toolset at:
            C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1
        - conan 2.x and cmake 3.25+ on PATH.
        - A conan profile that can build the deps (cppstd is overridden to 17 by this script).
      Mac (one-time setup, already done on the bench Mac):
        - Passwordless SSH as ianhassard@192.168.4.215 using ~/.ssh/id_ed25519 (LAN-only).
        - ~/mac-remote-rig/{env.sh,compare.sh} present (Metal/MoltenVK env + PNG diff tool).
        - Xcode Command Line Tools + Homebrew glslc/spirv-cross/cmake/ninja/python3.

.NOTES
    Run from anywhere; the script locates the repo root as its own parent's parent.
    Does NOT re-bake any golden. Rendering behavior is never modified.
#>

[CmdletBinding()]
param(
    # Skip a platform if you only want to verify one side (both run by default).
    [switch]$SkipWindows,
    [switch]$SkipMac
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Config ----------------------------------------------------------------------------------------
$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$VsDevShell = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1'

$MacUser    = 'ianhassard'
$MacHost    = '192.168.4.215'
$SshKey     = "$env:USERPROFILE\.ssh\id_ed25519"
$MacStage   = '~/hf-verify'                       # remote staging dir (recreated each run)
$TarName    = 'hf-verify.tar.gz'

# The committed Metal goldens, each produced by a distinct visual_test invocation. Name = the
# golden basename under tests/golden/metal/; Flag = the argv passed to visual_test BEFORE the output
# path (empty for the default Slice-F scene). The flags are the REAL ones parsed in
# metal_headless/visual_test.mm main() - confirmed there, not guessed. Every pair must diff 0.0000.
$Goldens = @(
    @{ Name = 'scene_shadow';  Flag = '' }                       # default visual_test <out>
    @{ Name = 'skinning';      Flag = '--skinning' }             # Slice O
    @{ Name = 'pbr_helmet';    Flag = '--pbr' }                  # Slice P
    @{ Name = 'mat_graph';     Flag = '--material' }             # Slice AV (data-driven material graph)
    @{ Name = 'mat_graph2';    Flag = '--material2' }            # Slice AW (second material; build-time codegen)
    @{ Name = 'mat_multi';     Flag = '--material-multi' }       # Slice AZ (three distinct graph materials in one frame)
    @{ Name = 'mat_normal';    Flag = '--material-normal' }      # Slice BE (NormalMap node: tangent-space normal map)
    @{ Name = 'instanced';     Flag = '--instanced' }            # Slice Q
    @{ Name = 'ibl_helmet';    Flag = '--ibl' }                  # Slice R
    @{ Name = 'physics';       Flag = '--physics' }              # Slice S
    @{ Name = 'transparency';  Flag = '--transparency' }         # Slice T
    @{ Name = 'bloom';         Flag = '--bloom' }                # Slice U
    @{ Name = 'scene_import';  Flag = '--scene' }                # Slice V
    @{ Name = 'debug_viz';     Flag = '--debug' }                # Slice W
    @{ Name = 'anim_blend';    Flag = '--blend' }                # Slice X
    @{ Name = 'anim_fsm';      Flag = '--anim-fsm' }             # Slice BL (animation state machine + cross-fade)
    @{ Name = 'ssao';          Flag = '--ssao' }                 # Slice Y
    @{ Name = 'capstone';      Flag = '--capstone' }             # Slice Z
    @{ Name = 'camera_pose';   Flag = '--camera 0.2,-0.1,0,3,10' } # Slice AA (scripted pose)
    @{ Name = 'gizmo';         Flag = '--gizmo 2' }              # Slice AB (select obj 2)
    @{ Name = 'csm';           Flag = '--csm' }                  # Slice AD (cascaded shadows)
    @{ Name = 'spot';          Flag = '--spot' }                 # Slice AE (spot-light shadows)
    @{ Name = 'point_shadow';  Flag = '--point-shadow' }         # Slice AF (omnidirectional point shadows)
    @{ Name = 'clustered';     Flag = '--clustered' }            # Slice AG (clustered / Forward+ lighting)
    @{ Name = 'clustered_lights'; Flag = '--clustered-lights' }  # Slice CL (clustered light culling: clustered==brute-force byte-identical)
    @{ Name = 'ssr';           Flag = '--ssr' }                  # Slice AH (screen-space reflections)
    @{ Name = 'dof';           Flag = '--dof' }                  # Slice CG (depth of field: thin-lens CoC depth gather)
    @{ Name = 'motion_blur';   Flag = '--motionblur' }           # Slice CN (per-object + camera motion blur: velocity-gather streak; zero motion byte-identical pass-through)
    @{ Name = 'oit';           Flag = '--oit' }                  # Slice CO (order-independent transparency: Weighted Blended OIT; permuted draw order resolves byte-identically)
    @{ Name = 'pom';           Flag = '--pom' }                  # Slice CP (parallax occlusion mapping: tangent-space height-field march; heightScale=0 byte-identical to plain normal mapping)
    @{ Name = 'gtao';          Flag = '--gtao' }                 # Slice CR (ground-truth ambient occlusion: horizon-search cosine-weighted visibility integral; radius=0 byte-identical to the no-AO scene)
    @{ Name = 'sss';           Flag = '--sss' }                  # Slice CZ (subsurface scattering: screen-space separable depth-aware diffusion; sssStrength=0 byte-identical to the non-SSS lit render)
    @{ Name = 'color_grade';   Flag = '--colorgrade' }           # Slice DB (analytic color grading: lift/gamma/gain + ASC-CDL slope/offset/power + luma-preserving saturation applied post-tonemap; the identity grade is byte-identical to the ungraded tonemap render)
    @{ Name = 'cas';           Flag = '--cas' }                  # Slice DF (contrast-adaptive sharpening / AMD FidelityFX CAS: an adaptive 3x3-cross edge sharpen on the tonemapped SDR result, clamped to the neighborhood min/max so it never rings; the sharpness=0 render is byte-identical to the unsharpened tonemap render)
    @{ Name = 'refl_probe';    Flag = '--reflprobe' }            # Slice DA (box-projected cubemap reflections: the specular reflection direction parallax-corrected to a local probe box so reflected walls line up with the room geometry; parallaxStrength=0 byte-identical to the standard infinite-cubemap reflection)
    @{ Name = 'capture_probe'; Flag = '--captureprobe' }         # Slice DD (runtime cubemap-capture reflection probe: the scene is rendered into 6 cube faces from the probe center, then a reflective sphere/floor samples the captured cube box-projected; a captured cube face is byte-identical to the scene rendered directly with that face's view/proj)
    @{ Name = 'planar_reflection'; Flag = '--planar' }           # Slice DE (planar reflections: the scene is rendered a second time through the camera reflected across the mirror plane — householder reflection + Lengyel oblique near-clip + flipped winding — into a 2D RT, then the mirror floor samples it at its own screen UV; reflectivity=0 byte-identical to the matte non-reflective render)
    @{ Name = 'froxel_fog';    Flag = '--froxelfog' }            # Slice CS (froxel volumetric fog: inject->integrate->depth-composited apply; baseDensity=0 byte-identical to the no-fog scene)
    @{ Name = 'probegi';       Flag = '--probegi' }              # Slice DH (DDGI beachhead probe-grid ray-trace: 256 probes x 16 Fibonacci-sphere rays vs the view-space depth field as a FLAT SSBO; the GPU ray-hit SSBO is BIT-EXACT to the CPU TraceRayToDepth reference + probeCount=0 -> the SSBO is byte-identical to the cleared upload; debug-viz = ray-hits as colored points over the lit scene)
    @{ Name = 'probe_capture'; Flag = '--probecapture' }         # Slice DI (DDGI probe radiance capture: a small 2x2x2 = 8-probe grid, each probe's lit scene captured into the 6 cube faces from its center — looping the single cubemap RT one probe at a time — and read back into a per-probe radiance store; a captured face is byte-identical to the scene rendered directly with that face's view/proj + probeCount=0 leaves the store byte-identical to its cleared value; debug-viz = per-probe average-radiance swatch spheres at the probe positions over the colored room)
    @{ Name = 'probe_sh';      Flag = '--probesh' }              # Slice DJ (DDGI probe SH-encode: each captured probe cubemap encoded into 3rd-order real spherical harmonics — 9 coeffs/RGB channel — by a pure compute pass over a fixed host-precomputed sample set; the sample dirs + SHBasis9 weights + solid-angle weights are uploaded as exact bits both the GPU encode + CPU reference read, so the GPU per-probe SH SSBO is BIT-EXACT to the CPU SH-encode + zero-radiance encodes to zero-SH + probeCount=0 leaves the SH SSBO byte-identical to its cleared value; debug-viz = per-probe swatch spheres colored by the SH-reconstructed irradiance over the colored room)
    @{ Name = 'probeinterp';   Flag = '--probeinterp' }          # Slice DL (DDGI probe trilinear SH-interp: the per-probe SH irradiance records trilinearly blended at an arbitrary world position — for each query point, the FLOOR cell whose 8 corner probes bracket it is found and those 8 probes' 3rd-order real-SH coefficients are blended by polynomial trilinear weights — by a pure compute pass, one thread per query point; NearestProbes + InterpolateSH are copied VERBATIM into the shader (mad for the per-corner product) so the GPU blended-SH SSBO is BIT-EXACT to the CPU InterpolateSH reference + a query at a probe position blends to that probe's SH byte-identical + probeCount=0 leaves the output SSBO byte-identical to its cleared value; debug-viz = a dense grid of swatch spheres colored by the SH-reconstructed irradiance of the interpolated field over the colored room)
    @{ Name = 'ddgi';          Flag = '--ddgi' }                  # Slice DN (DDGI GI composite — the VISIBLE payoff: a single-bounce indirect-diffuse term added to a sibling lit-pass variant (lit_ddgi.frag) so the Cornell box shows colored indirect light bleed; per pixel the 8 nearest probes' 3rd-order real-SH irradiance is trilinearly blended toward the surface normal (NearestProbes + InterpolateSH + SHEvaluate copied VERBATIM into the shader, mad for every accumulation) and added as indirect*albedo*(1-metallic)*giStrength; the ProbeSH[] rides in a fragment storage buffer bound via the existing usesLightClusters/BindLightClusters path (no new RHI); INTERNALLY the giStrength=0 render == the probeCount=0 render BYTE-IDENTICAL (the indirect term is a literal +0.0 == the no-GI render through the same pipeline) + frame B (giStrength>0) differs from frame A + two frame-B runs byte-identical; golden = frame B, the visible color bleed; lit.frag + all its goldens untouched via sibling-shader isolation)
    @{ Name = 'probe_dist';    Flag = '--probedist' }            # Slice DO (DDGI per-probe distance-moment capture — the visibility data layer: the same 2x2x2 = 8-probe grid, each probe renders the scene GEOMETRY from its center into the 6 cube faces writing the linear world-distance d = length(worldPos - probeCentre) packed as the two moments float2(d, d*d) — looping the single cubemap RT one probe at a time — and read back into a per-probe distance-moment store; the second moment d*d is a BARE multiply so, given the same read-back distance bytes, the GPU moment is BIT-EXACT to the CPU MomentsFromDistance reference (the GPU==CPU proof is on the moment-from-distance step; the sqrt distance itself is render-equivalence per backend), a captured distance face is BYTE-IDENTICAL to the scene rendered directly with that face's view/proj, and probeCount=0 leaves the store byte-identical to its cleared value; produces NO visible lighting change — the next slice consumes it for Chebyshev occlusion; debug-viz = per-probe swatch spheres colored by the probe's mean captured distance, near warm / far cool, over the colored room)
    @{ Name = 'ddgi_mb';       Flag = '--ddgimb' }               # Slice DR (DDGI multi-bounce — the 2nd light bounce: single-bounce DDGI captures only DIRECT light into the probes so indirect light bounces once; multi-bounce RE-captures each probe with a sibling capture-with-GI shader (probe_bake_gi.frag) that adds the DN single-bounce indirect term sampling SH0 -> SH-encode -> SH1, and the composite binds SH(bounceCount-1): SH0 at bounceCount=1 (the DN single-bounce path, BYTE-IDENTICAL), SH1 at bounceCount>=2 (the brighter 2nd bounce); two ProbeSH[] SSBOs ride the existing usesLightClusters/BindLightClusters path (no new RHI); INTERNALLY frame A (bounceCount=1) is byte-identical to the DN render (bounce-1 skipped) + frame B (bounceCount=2) is measurably brighter than A + the bounce-1 SH1 SSBO is BIT-EXACT to a CPU SH-encode of the read-back bounce-1 radiance + two frame-B runs byte-identical; golden = frame B, the 2nd bounce; probe_bake.frag + the DN lit_ddgi shader + their goldens untouched via sibling-shader isolation)
    @{ Name = 'ddgi_occ';      Flag = '--ddgiocc' }              # Slice DP (DDGI Chebyshev occlusion weighting — the VISIBLE leak-fix: a sibling lit-pass variant weights each of the 8 nearest probes' SH contribution by a per-probe Chebyshev (variance-shadow) visibility weight read from DO's per-probe distance-moment cube, so a probe geometrically occluded from a surface stops bleeding indirect light through the occluder; occlusionStrength=0 branches to the VERBATIM DN composite path and is BYTE-IDENTICAL to the DN lit_ddgi render — the make-safe no-op — while occlusionStrength=1 shows the leak through the occluder panel GONE/attenuated; golden = the leak-fixed frame; the DN lit_ddgi shader + its goldens untouched via sibling-shader isolation)
    @{ Name = 'froxel_lights'; Flag = '--froxellights' }         # Slice CV (per-froxel clustered-light injection: the 96 clustered point lights scatter through the fog as colored volumetric shafts; injectLights=false byte-identical to the CS sun-only fog + density=0 byte-identical to the no-fog scene)
    @{ Name = 'vol_shadows';   Flag = '--volshadows' }           # Slice CX (volumetric shadows: the froxel inject samples the sun's CSM shadow map per froxel and gates the sun in-scatter -> dark fog volumes + foggy sun shafts; volumetricShadows=false byte-identical to the CV froxel-lights render + density=0 byte-identical to the no-fog scene)
    @{ Name = 'contact_shadows'; Flag = '--contactshadow' }      # Slice CT (screen-space contact shadows: short depth ray-march toward the sun fills the fine contact occlusion the CSM misses; maxDist=0 byte-identical to the no-contact scene)
    @{ Name = 'auto_exposure'; Flag = '--autoexposure' }         # Slice CW (auto-exposure histogram eye adaptation: integer-histogram luminance metering -> key-value exposure -> tonemap applies it; adaptationEnabled=false uses the fixed E0 and is byte-identical to the standard fixed-exposure tonemap)
    @{ Name = 'water';         Flag = '--water' }                # Slice CF (Gerstner water reflect/refract + sun glint)
    @{ Name = 'ssgi';          Flag = '--ssgi' }                 # Slice BP (screen-space global illumination)
    @{ Name = 'ssgi_denoise';  Flag = '--ssgi-denoise' }         # Slice BR (SSGI bilateral spatial denoise)
    @{ Name = 'ssgi_temporal'; Flag = '--ssgi-temporal' }        # Slice BV (temporal SSGI fixed-N accumulation)
    @{ Name = 'volumetric';    Flag = '--volumetric' }          # Slice AJ (volumetric fog / light shafts)
    @{ Name = 'probe';         Flag = '--probe' }                # Slice AK (reflection + irradiance probes)
    @{ Name = 'taa';           Flag = '--taa' }                  # Slice AP (temporal anti-aliasing)
    @{ Name = 'cull';          Flag = '--cull' }                 # Slice AQ (frustum-culling visualization)
    @{ Name = 'meshlet_viz';   Flag = '--meshlet-viz' }          # Slice DS (virtual-geometry meshlet/cluster decomposition: SphereGeometry(48,32) partitioned by a pure-CPU integer-deterministic Morton sort into <=128-tri clusters, each drawn as an index sub-range with its deterministic per-cluster hash color -> coherent flat-colored spatial cluster patches; cross-backend-identical by construction — same verts, same projection, color = pure integer hash; the BEACHHEAD of the Nanite-style virtual-geometry arc)
    @{ Name = 'gpu_cull';      Flag = '--gpu-cull' }             # Slice AR (GPU-driven culling + indirect draw)
    @{ Name = 'mdi';           Flag = '--mdi' }                  # Slice BM (GPU multi-draw-indirect; Metal renders the identical scene per-object)
    @{ Name = 'bindless';      Flag = '--bindless' }            # Slice BZ (bindless textures; Metal renders the identical multi-texture scene per-material bound)
    @{ Name = 'gpudriven';     Flag = '--gpudriven' }           # Slice CB (fully-GPU-driven pass: MDI + bindless; Metal renders the identical multi-material scene per-object bound)
    @{ Name = 'gpucull_draw';  Flag = '--gpucull-draw' }        # Slice CD (fully-GPU-driven-CULLED pass: compute-cull -> MDI + bindless; Metal renders the identical CPU-frustum-culled survivor subset per-object bound)
    @{ Name = 'cluster_cull';  Flag = '--cluster-cull' }        # Slice DT (virtual-geometry per-CLUSTER frustum cull -> indirect cluster draw: an instance grid of DS-clustered spheres culled at (instance x cluster) granularity on the GPU via compute-cull -> MDI; Metal renders the identical CPU-frustum-culled surviving cluster-instances per-cluster bound; gpu-culled==cpu-culled byte-identical, gpu count==cpu frustum.h)
    @{ Name = 'hiz_cull';      Flag = '--hiz-cull' }            # Slice CJ (Hi-Z occlusion cull: depth pre-pass -> Hi-Z pyramid -> frustum+occlusion cull; Metal renders the identical CPU-occlusion-culled visible survivor subset per-object bound; occlusion-culled==frustum-only)
    @{ Name = 'cluster_hiz';   Flag = '--cluster-hiz' }         # Slice DU (virtual-geometry per-CLUSTER Hi-Z occlusion cull: DT's per-cluster frustum cull PLUS CJ's Hi-Z occlusion test; an occluder wall hides a back row of DS-clustered spheres; Metal renders the identical CPU-occlusion-culled visible cluster-instance subset per-cluster bound; occlusion-culled==frustum-only byte-identical, survivor count drops, gpu count==cpu SurvivorClusterCountHiZ)
    @{ Name = 'cluster_lod';   Flag = '--cluster-lod' }         # Slice DV (virtual-geometry discrete cluster-LOD selection by screen-space error: a row of instances marching away each picks one of 3 pre-baked LOD tessellations by projected screen-space error; near LOD0 fine, far LOD2 coarse; the squared-distance SelectLod is bit-exact GPU==CPU on Vulkan; Metal renders the identical CPU-selected-LOD clusters per-cluster bound; forceLod0==full-detail byte-identical, LOD varies, distance-monotonic)
    @{ Name = 'visbuffer';     Flag = '--visbuffer' }           # Slice DW (virtual-geometry VISIBILITY BUFFER: the DT survivor cluster MDI draw rasterizes (clusterID<<7 | SV_PrimitiveID) into an R32_Uint render target, read back bit-exact; the image golden is a CPU-coloring of the read-back IDs (bg->clear, else hashColor(clusterID)) — identical both backends by construction; visbuffer self-consistent + coverage == CPU survivors EXACT, two renders BYTE-IDENTICAL, GPU==CPU interior coverage EXACT; the only new RHI is the additive Format::R32_Uint)
    @{ Name = 'visresolve';    Flag = '--visresolve' }          # Slice DX (virtual-geometry DEFERRED MATERIAL RESOLVE: a fullscreen pass texel-fetches the DW R32_Uint vis-buffer per pixel, looks up the covering triangle via the cluster-meta/index/vertex SSBOs, computes a FLAT geometric normal + Lambert shade (verbatim render/visresolve.h, std::fma/mad), and outputs the LIT image — Nanite's decoupled geometry->material resolve; the spheres render flat-shaded/faceted via the vis-buffer; ID-provenance bit-exact + GPU==CPU resolve-math bit-exact (interior pixels) + two-run determinism + a resolve-vs-forward SMOKE bound (< eps, explicitly NOT byte-identical — cross-vendor rasterizer divergence); the integer vis-buffer rides BindTexture (sampled uint texture, texel-fetched no sampler) + the SSBOs ride BindLightClusters — NO new RHI; the resolve frag gets the isolated --msl-version 20200 for the integer texture.read)
    @{ Name = 'swraster';      Flag = '--swraster' }            # Slice SW1 (Nanite SOFTWARE-RASTER: a PURE-CPU deterministic integer/fixed-point software rasterizer scan-converts a small fixed set of cluster triangles into a packed depth|id vis-buffer — the same (clusterID<<7|triID) identity as DW's HW vis-buffer — front-most-per-pixel via a serial min over a 32-bit depth|id key (the CPU mirror of SW2's InterlockedMin); the ONE FP step ProjectToScreenVert is host-only, coverage is pure integer edge functions + a TOP-LEFT fill rule (watertight); the golden is a CPU-colored vis-buffer (bg->clear, else hashColor(visId>>7)) — identical both backends by construction; proofs: bit-exact hash + shuffled-order BYTE-IDENTICAL + a sub-pixel triangle COVERED + empty==cleared; PURE CPU, NO GPU, NO new RHI)
    @{ Name = 'swraster_gpu';  Flag = '--swraster-gpu' }        # Slice SW2 (Nanite SOFTWARE-RASTER GPU: the GPU COMPUTE software rasterizer — shaders/swraster.comp, one thread per cluster-triangle, scan-converts a clustered SphereGeometry's triangles into a w*h depth|id vis-buffer SSBO via InterlockedMin (MSL atomic_min), running the int64 integer edge math + top-left fill rule + flat min-depth copied VERBATIM from render/swraster.h over the HOST-SNAPPED integer ScreenVerts (ZERO GPU FP); ReadBuffer reads the integer vis-buffer, PROVEN BIT-IDENTICAL vs the CPU swraster.h::RasterClusters reference (memcmp, full-frame — the make-or-break), two dispatches byte-identical (atomic-min commutative), a sub-pixel triangle COVERED, triCount=0 -> all-kSwClear; the image golden is a CPU-coloring of the read-back integer vis-buffer (bg->clear, else hashColor(visId>>kTriIdBits)) — identical both backends by construction; RWStructuredBuffer<uint> + InterlockedMin + ReadBuffer, NO new RHI)
    @{ Name = 'swraster_resolve'; Flag = '--swraster-resolve' }  # Slice SW4 (Nanite SOFTWARE-RASTER vis-buffer -> DEFERRED MATERIAL RESOLVE: drive the EXISTING visresolve.frag deferred resolve from the SOFTWARE visibility buffer — swraster.comp fills the SW depth|id SSBO over the SW2 sphere scene (~769 tris, 512x512), a fullscreen unpack blit (swraster_resolve_blit.frag) converts it into the R32_Uint visId RT the resolve reads (out=(packed==kSwClear)?kVisBackground:(packed&0xFFFF) — strip the HIGH-16 depth, keep the LOW-16 visId=(clusterID<<7)|triID; the sentinels collide so sky passes through), then the UNCHANGED visresolve.frag flat-shades it into a Lambert-lit BGRA8 image; proofs: blit GPU==CPU unpack BIT-IDENTICAL full-frame, SW-resolved RGB == CPU vg::ResolvePixel expectation @interior, GPU==CPU shade @interior EXACT, two renders BYTE-IDENTICAL, disabled-path all-sky no-op, a coherent shaded mesh (shaded>0, not uniform); the blit rides BindLightClusters (the SW SSBO at set-3 binding 13) + the resolve rides BindTexture (the R32_Uint RT) + BindLightClusters (the cluster SSBOs) — NO new RHI; on Metal the CPU swraster.h::RasterClusters reference fills the SSBO (swraster.comp is int64/Vulkan-only) -> the SAME blit + resolve, identical by construction; the blit frag is plain MSL (writes an integer SV_Target — NO --msl-version 20200))
    @{ Name = 'vsm_pages';     Flag = '--vsm-mark' }            # Slice VA (Virtual Shadow Maps page table + PAGE-NEEDED marking: a fixed directional CLIPMAP + a fixed ground-grid receiver point-set feed vsm_mark.comp — one thread per receiver runs the INTEGER threshold-ladder SelectClipmapLevel (NO transcendental log2) + MarkPage (subtract/divide/floor), verbatim render/vsm.h -> resident[pageId]=1; ReadBuffer reads the integer resident set, PROVEN BIT-EXACT vs the CPU vsm::MarkResidentPages reference (memcmp, no FP tol — the GPU==CPU determinism crux), markingEnabled=false -> empty set, two runs byte-identical; the image golden is a top-down concentric-clipmap debug-viz CPU-colored from the read-back integer set (resident page -> hashColor(pageId), else dark) — identical both backends by construction; pure compute over a Storage SSBO + ReadBuffer — NO rendering, NO new RHI)
    @{ Name = 'vsm_atlas';     Flag = '--vsm-render' }          # Slice VB (Virtual Shadow Maps PHYSICAL-PAGE depth render: allocate PHYSICAL atlas tiles to VA's resident pages — a deterministic integer virtual->physical indirection table (vsm::AllocatePhysicalPages, ascending-pageId priority) — then render each resident page's casters' depth into its tile via the CSM SetViewport atlas loop with that page's clipmap-level ortho (vsm::PageWorldOrtho); the indirection is two-run byte-identical AND == the CPU AllocatePhysicalPages reference (memcmp), the colorized depth atlas is two-run byte-identical, markingEnabled=false -> empty atlas; the golden is the colorized depth atlas (vsm_depth.frag SV_Position.z -> grayscale, near=bright, ReadRenderTarget'd from the COLOR atlas, non-resident tiles the clear color, resident tiles per-page caster-depth silhouettes); reuses CreateRenderTarget + SetViewport + ReadRenderTarget + the pure-CPU allocator — NO new RHI)
    @{ Name = 'vsm_shadow';    Flag = '--vsm-sample' }          # Slice VC (Virtual Shadow Maps LIT-PASS indirection sample — the SHADOWED scene: per receiver pixel lit_vsm.frag (a sibling of lit_csm.frag) picks the clipmap level via the VA integer threshold-ladder, projects to the virtual page, looks up the physical tile in the indirection SSBO (bound via usesLightClusters/BindLightClusters), projects into that page's PageWorldOrtho light-space, and tile-clamped-3x3-PCF-samples the VSM DEPTH atlas (bound via SetShadowMap) -> a VSM-shadowed image; the make-or-break NO-OP: shadow = lerp(1, shadow, vsmEnabled) -> vsmEnabled=0 == the unshadowed lit BYTE-IDENTICAL; reuses BindLightClusters + SetShadowMap + the FrameData UBO + the lit-pass pipeline — NO new RHI)
    @{ Name = 'vsm_cache';     Flag = '--vsm-cache' }           # Slice VD (Virtual Shadow Maps PER-PAGE CACHING — the FINAL VSM slice: a per-page CONTENT KEY (vsm::PageContentKey, a fixed FNV hash of the page's clipmap level + world region + the casters whose bounds overlap it) + a per-page CACHE (vsm::VsmPageCache) skip re-rendering physical pages whose content is unchanged — UE5's VSM key performance property; a HIT page's tile carries over from the previous atlas via a CPU tile-copy (a readback memcpy, NO new RHI) so the cached atlas is BYTE-IDENTICAL to the fully-re-rendered atlas; Pass 1 populate (all miss) -> atlasA, Pass 2 same scene (all HIT, 0 re-renders) cached==fresh BYTE-IDENTICAL + hits==residentPages (the make-or-break), Pass 3 move ONE caster (cached==full BYTE-IDENTICAL + a minimal miss set), two-run determinism, prints {residentPages,hits,misses}; the golden is a cache-status viz — resident tiles CPU-tinted CACHED=green / RE-RENDERED=red from the integer cache state, identical both backends by construction; pure host-side cache over VB's existing render + readback — NO new RHI) (Virtual Shadow Maps LIT-PASS indirection sample — the SHADOWED scene: per receiver pixel lit_vsm.frag (a sibling of lit_csm.frag) picks the clipmap level via the VA integer threshold-ladder, projects to the virtual page, looks up the physical tile in the indirection SSBO (bound via usesLightClusters/BindLightClusters), projects into that page's PageWorldOrtho light-space, and tile-clamped-3x3-PCF-samples the VSM DEPTH atlas (bound via SetShadowMap) -> a VSM-shadowed image; the make-or-break NO-OP: shadow = lerp(1, shadow, vsmEnabled) -> vsmEnabled=0 == the unshadowed lit BYTE-IDENTICAL; PROOFS: vsmEnabled=0==unshadowed byte-identical + frame B != A (shadows active) + page-lookup GPU==CPU bit-exact + two-run determinism + a VSM-vs-CSM-1-level smoke; the golden is frame B, the VSM-shadowed scene; reuses BindLightClusters (indirection) + SetShadowMap (atlas) + the FrameData UBO + the lit-pass pipeline; lit.frag/lit_csm.frag + their goldens untouched — NO new RHI)
    @{ Name = 'vt_feedback';   Flag = '--vt-feedback' }         # Slice VT1 (Runtime Virtual Texturing page table + PAGE-NEEDED FEEDBACK marking, the beachhead of FLAGSHIP #4 — UE5's literal RVT: a fixed virtual texture (a MIP PYRAMID, mipLevels=4/pageSize=128/vpps0=16 -> 340 pages) + a fixed deterministic (UV,mip) sample-request set feed vt_feedback.comp — the host HOST-SNAPS each request to integer (mip,px,py) page coords (vt::SnapRequest, the UV->page quantization done CPU-side so the GPU does ZERO float), one thread per request computes pageId via PageId (verbatim render/vt.h) -> feedback[pageId]=1; ReadBuffer reads the integer feedback set, PROVEN BIT-EXACT vs the CPU vt::MarkFeedbackPages reference (memcmp, no FP tol — the GPU==CPU determinism crux), feedbackEnabled=false -> empty set, two dispatches byte-identical; the image golden is a per-mip page-grid debug-viz CPU-colored from the read-back integer set (resident page -> hashColor(pageId), else dark) — identical both backends by construction; pure compute over a Storage SSBO + ReadBuffer — NO rendering, NO new RHI)
    @{ Name = 'vt_atlas';      Flag = '--vt-pagegen' }          # Slice VT3 (Runtime Virtual Texturing procedural PAGE GENERATION into the PHYSICAL ATLAS, the 3rd RVT slice — the analog of VSM Slice VB's physical-page depth render, writing procedural texture COLOR (PageTexel) instead of caster depth: VT2's indirection -> a host reverse table tilePageId -> vt_pagegen.comp (ONE thread per atlas texel, a race-free MAP, NO atomics) generates each allocated page's content into its physical tile via PageTexel(pageId,mip,localXY) (the PURE-INTEGER per-texel generator copied VERBATIM from render/vt.h: a per-page hash base color modulated by an 8x8 checkerboard + a coarse gradient, every channel floored into [0x40,0xFF] so no texel collides with kAtlasClear 0xFF101010) over a finite VtTilePool{tilesPerSide=16} (256 tiles > resident(212) so 212 allocated + 44 free/dark), unallocated -> kAtlasClear; the 1024x1024 atlas SSBO is ReadBuffer'd, PROVEN BIT-EXACT vs the CPU vt::GeneratePhysicalAtlas reference (memcmp, no FP tol — the GPU per-texel generator reproduces the CPU atlas), every allocated tile fully written + the rest kAtlasClear, genEnabled=false -> all kAtlasClear, two dispatches byte-identical; the image golden is the atlas decoded directly from the RGBA8-packed uints — identical both backends by construction; pure compute over a Storage SSBO + ReadBuffer — NO rendering, NO new RHI)
    @{ Name = 'vt_alloc';      Flag = '--vt-alloc' }            # Slice VT2 (Runtime Virtual Texturing PHYSICAL TILE-POOL ALLOCATION + virtual->physical INDIRECTION table, the 2nd RVT slice — the DIRECT analog of VSM Slice VB's AllocatePhysicalPages: VT1's resident feedback set -> a SINGLE-THREAD allocator (vt_alloc.comp, [numthreads(1,1,1)]: one thread walks pageId ascending maintaining nextTile -> gIndirection[pageId] = (resident && nextTile<cap) ? nextTile++ : 0xFFFFFFFF) over a finite VtTilePool{tilesPerSide=12} (144 tiles < resident -> overflow exercised); ReadBuffer reads the indirection table, PROVEN BIT-EXACT vs the CPU vt::AllocatePhysicalTiles reference (memcmp, no FP tol — the GPU allocator reproduces the CPU table), allocated=min(resident,capacity) with unique/in-range/ascending-priority tile indices, allocEnabled=false -> all kNoTile, two dispatches byte-identical; the image golden is a per-mip page-grid debug-viz CPU-colored from the read-back integer indirection (allocated -> hashColor(tileIndex), overflow-resident -> a dim color, non-resident -> dark) — identical both backends by construction; pure compute over a Storage SSBO + ReadBuffer — NO rendering, NO new RHI)
    @{ Name = 'vt_sample';     Flag = '--vt-sample' }            # Slice VT4 (Runtime Virtual Texturing material-pass SAMPLE through the virtual->physical INDIRECTION, the 4th RVT slice — the round-trip that makes RVT a usable texture: reuses VT3's atlas + VT2's indirection (the SAME VtTexture 4mip/128/16vpps0 + 576 requests + VtTilePool{tilesPerSide=16} + VtAtlasDims{64}) and dispatches vt_sample.comp — ONE thread per VIRTUAL texel (a race-free MAP, NO atomics) — reconstructing the mip-0 virtual image (16 pages/side × 64-px = 1024×1024) NEAREST through the indirection: each virtual texel computes its pageId -> gIndirection[pageId] -> the physical tile -> reads the atlas texel (SampleVirtualTexel copied VERBATIM from render/vt.h; non-resident page -> kVtMiss 0xFFFF00FF magenta); kept INTEGER nearest (NOT bilinear) so the sample is cross-backend BIT-IDENTICAL. ReadBuffer reads the RGBA8-packed virtual image, PROVEN BIT-EXACT vs the CPU vt::ReconstructVirtualImage reference (memcmp, no FP tol — the GPU sample reproduces the CPU image), round-trip self-consistency (every resident virtual texel == PageTexel(pageId,mip,lx,ly), every non-resident == kVtMiss), page-lookup @interior GPU==CPU, sampleEnabled=false -> all kVtMiss, two dispatches byte-identical; the image golden is the reconstructed virtual image decoded directly from the RGBA8-packed uints (resident pages textured at their virtual-UV location, non-resident pages magenta — visually DISTINCT from VT3's physical-atlas) — identical both backends by construction; pure compute over a Storage SSBO + ReadBuffer — NO rendering, NO new RHI)
    @{ Name = 'vt_cache';      Flag = '--vt-cache' }             # Slice VT5 (see above)
    @{ Name = 'mc_classify';   Flag = '--mc-classify' }         # Slice MC1 (GPU Isosurface Meshing per-cell MARCHING-CUBES CASE CLASSIFICATION, the beachhead of FLAGSHIP #5: a fixed VoxelField (33³ corners -> 32³ cells) filled with a deterministic INTEGER sphere SDF (radius ~12 cells, quantized int32 fixed-point, isovalue 0) feeds shaders/mc_classify.comp — ONE thread per cell gathers its 8 cube-CORNER scalars at the canonical kCornerOffset and writes gCases[cell]=CaseIndex(corners,isovalue) (idx = Σ (corner[i] > isovalue) << i, copied VERBATIM from render/mc.h; an order-independent per-cell integer write, NO atomics); ReadBuffer reads the case-index field, PROVEN BIT-EXACT vs the CPU render/mc.h::ClassifyCells reference (memcmp, NO tolerance — the GPU==CPU crux), classifyEnabled=false -> all-zero no-op, two dispatches byte-identical, hand-checked known cells (empty=0x00/full=0xFF/corner0=0x01); the image golden is the case-index field CPU-colored as a grid of 32 Z-slices (each (nx-1)x(ny-1), hashColor(caseIndex), empty/full dark; tiled 8x4) — identical both backends by construction (PURE INTEGER); pure compute over a Storage SSBO + ReadBuffer — NO rendering, NO new RHI) (Runtime Virtual Texturing per-page CACHING across frames, the 5th and FINAL RVT slice — the direct analog of VSM Slice VD's PageContentKey/VsmPageCache: reuses VT3/VT4's scene (the SAME VtTexture 4mip/128/16vpps0 + 576 requests + VtTilePool{tilesPerSide=16} + VtAtlasDims{64}) and a host per-page FNV content key vt::PageContentKey(pageId,mip,contentVersion) + per-tile vt::VtPageCache builds per-tile needsGen flags, then dispatches vt_cachegen.comp — ONE thread per atlas texel (a race-free MAP, NO atomics) — over a PERSISTENT atlas SSBO (NOT re-cleared): gNeedsGen[tile]==0 (cached/unallocated) -> RETURN without writing so the texel persists byte-identical, else write PageTexelV(pageId,mip,localXY,gTileVersion[tile]) (the content-version-aware generator copied VERBATIM from render/vt.h; version 0 reproduces VT3 PageTexel exactly so VT3/VT4 goldens are unchanged). Three passes: Pass1 populate (all allocated needsGen=1), Pass2 all-cached (needsGen=0, writes nothing -> atlas BYTE-IDENTICAL to Pass1, misses2==0), Pass3 bump ONE page's contentVersion (only that tile needsGen=1). PROVEN: all-cached==fresh BYTE-IDENTICAL; invalidation minimal (misses3==1) + transparent (the partial cached regen == a full uncached regen, BYTE-IDENTICAL); the Pass3 GPU atlas == the CPU vt::GenerateCachedAtlas reference (memcmp, no FP tol); two full 3-pass runs byte-identical; the image golden is a cache-status viz of the Pass3 state (allocated tiles GREEN cached / RED the 1 regenerated, dark unallocated) — identical both backends by construction; pure compute over a Storage SSBO + ReadBuffer — NO rendering, NO new RHI)
    @{ Name = 'mc_count';      Flag = '--mc-count' }            # Slice MC2 (GPU Isosurface Meshing per-cell MARCHING-CUBES TRIANGLE COUNT, the 2nd slice of FLAGSHIP #5: the SAME MC1 sphere field (33³ corners -> 32³ cells, radius 12, iso 0) feeds shaders/mc_count.comp — ONE thread per cell runs CaseIndex (verbatim render/mc.h), looks up triCount = gTriCount[caseIdx] from the host-uploaded 256-entry count table (built from the canonical render/mc.h::kTriTable), writes gCounts[cell]=triCount and InterlockedAdd(gTotal[0],triCount) the grand total (the autoexposure_histogram integer-atomic precedent — commutative -> order-independent -> deterministic); countEnabled=0 -> writes 0, no atomic. ReadBuffer reads the per-cell counts + the atomic total, PROVEN BIT-EXACT vs the CPU render/mc.h::CountCells + TotalTriangles reference (memcmp counts, == total — the GPU==CPU + atomic-determinism crux), tri-table self-consistency (kTriCount[c] == derived-from-kTriTable for all 256, empty/full=0), countEnabled=false -> zero counts + zero total no-op, two dispatches byte-identical; the image golden is the per-cell count field CPU-colored as 32 Z-slices via a fixed count->color ramp (0=dark, 1..5 distinct hues; tiled 8x4) — identical both backends by construction (PURE INTEGER); pure compute over Storage SSBOs + InterlockedAdd + ReadBuffer — NO rendering, NO new RHI)
    @{ Name = 'mt';            Flag = '--mt' }                   # Slice AU (multithreaded recording; Metal N=4 parallel encoder)
    @{ Name = 'game';          Flag = '--game' }                # Slice AX (playable roll-a-ball game sample)
    @{ Name = 'net';           Flag = '--net' }                 # Slice BQ (replication; replica reconstructs + renders the scene)
    @{ Name = 'netsim';        Flag = '--netsim' }              # Slice BU (transport sim: lossy/laggy channel + client jitter-buffer interpolation)
    @{ Name = 'netpredict';    Flag = '--netpredict' }          # Slice BY (client prediction + server reconciliation: rewind+replay corrects a server-only misprediction)
    @{ Name = 'hud';           Flag = '--hud' }                 # Slice BA (text / HUD overlay)
    @{ Name = 'game_hud';      Flag = '--game-hud' }            # Slice BA (game scene + live SCORE HUD)
    @{ Name = 'stream';        Flag = '--stream' }             # Slice BD (scene/asset streaming; resident cell subset)
    @{ Name = 'terrain';       Flag = '--terrain' }            # Slice BF (procedural terrain / heightmap)
    @{ Name = 'terrain_stream'; Flag = '--terrain-stream' }    # Slice BJ (terrain streaming + per-tile LOD)
    @{ Name = 'decal';         Flag = '--decal' }             # Slice BH (screen-space projected decals)
    @{ Name = 'poststack';     Flag = '--poststack' }         # Slice BN (data-driven post-process stack)
    @{ Name = 'editor';        Flag = '--editor' }            # Slice BT (docked editor UI: Hierarchy/Inspector/Stats/Viewport)
    @{ Name = 'editor_edit';   Flag = '--editor-edit' }      # Slice BX (editor live-edit: edited scene + scene_io round-trip)
    @{ Name = 'vfx';           Flag = '--vfx' }              # Slice CC (CPU particle / VFX emitter: additive billboard fountain)
    @{ Name = 'clouds';        Flag = '--clouds' }           # Slice CH (volumetric clouds: raymarched sunlit cumulus layer in the sky dome)
    @{ Name = 'cloud_shadows'; Flag = '--cloud-shadows' }    # Slice CK (cloud shadows on the ground: the cloud field attenuates the direct sun on the lit scene as dappled shadows)
)

$winResult = 'SKIP'
$macResult = 'SKIP'
$script:macGoldenResults = @()   # per-golden [{ Name; Diff; Ok }] filled by Invoke-MacVerify

function Write-Section($t) { Write-Host ""; Write-Host "==== $t ====" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------------------------------
# Windows / Vulkan
# ---------------------------------------------------------------------------------------------------
function Invoke-WindowsVerify {
    Write-Section "WINDOWS / VULKAN"

    if (-not (Test-Path $VsDevShell)) {
        throw "VS BuildTools dev shell not found at: $VsDevShell"
    }

    # Build a single child-shell script: enter the VS dev shell, then run the full pipeline.
    # `$LASTEXITCODE` is checked after each native step so the first failure aborts with non-zero.
    # NOTE: do NOT set `$ErrorActionPreference='Stop' here. Launch-VsDevShell shells out to
    # vswhere and can emit a benign native-command warning to stderr; under 'Stop' that aborts
    # the shell. We gate every step on `$LASTEXITCODE explicitly instead, which is exact.
    $inner = @"
`$ErrorActionPreference = 'Continue'
& '$VsDevShell' -Arch amd64 2>`$null | Out-Null
Set-Location '$RepoRoot'

Write-Host '--- conan install ---'
conan install . -of build/windows-msvc-debug -s build_type=Debug -s compiler.cppstd=17 ``
    -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing
if (`$LASTEXITCODE -ne 0) { exit 11 }

# Conan appends each build's generated preset to CMakeUserPresets.json. If an ASan build was also
# configured (build/windows-msvc-asan), both conan presets are named 'conan-debug' and CMake
# refuses to read presets ('Duplicate preset'). Pin CMakeUserPresets.json to the debug include only
# so the verify run is deterministic regardless of what else has been built. NOTE: written WITHOUT a
# BOM -- conan's preset reader does json.loads() and chokes on a UTF-8 BOM.
`$pinned = @'
{
    "version": 4,
    "vendor": { "conan": {} },
    "include": [ "build/windows-msvc-debug/build/Debug/generators/CMakePresets.json" ]
}
'@
[System.IO.File]::WriteAllText((Join-Path (Get-Location) 'CMakeUserPresets.json'), `$pinned, (New-Object System.Text.UTF8Encoding(`$false)))

Write-Host '--- cmake configure ---'
cmake --preset windows-msvc-debug
if (`$LASTEXITCODE -ne 0) { exit 12 }

Write-Host '--- cmake build ---'
cmake --build --preset windows-msvc-debug
if (`$LASTEXITCODE -ne 0) { exit 13 }

Write-Host '--- ctest ---'
ctest --preset windows-msvc-debug
if (`$LASTEXITCODE -ne 0) { exit 14 }

# --- JSON introspection golden (Slice AL): an EXACT byte-for-byte match of the live --introspect
# output for the default scene against the committed text golden. This is the agent-OBSERVE artifact
# (editor::DescribeEngine). It is backend-AGNOSTIC (pure hf_core, no vk*/Metal symbols), so unlike the
# 26 IMAGE goldens it does NOT need the Mac: the bytes are identical on Vulkan and Metal. We therefore
# verify it once, here, on the Windows/Vulkan build. ---
Write-Host '--- introspection JSON golden ---'
`$introExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$introGolden = 'tests/golden/introspect/default_scene.json'
`$introLive = Join-Path `$env:TEMP 'hf_introspect_live.json'
& `$introExe --introspect `$introLive 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host 'introspect run failed'; exit 15 }
# Compare RAW bytes (the program writes LF-only, no BOM); any difference is a failure.
`$gBytes = [System.IO.File]::ReadAllBytes((Resolve-Path `$introGolden).Path)
`$lBytes = [System.IO.File]::ReadAllBytes(`$introLive)
`$introOk = (`$gBytes.Length -eq `$lBytes.Length)
if (`$introOk) { for (`$bi = 0; `$bi -lt `$gBytes.Length; `$bi++) { if (`$gBytes[`$bi] -ne `$lBytes[`$bi]) { `$introOk = `$false; break } } }
if (-not `$introOk) {
    Write-Host 'introspection JSON golden MISMATCH (tests/golden/introspect/default_scene.json)'
    exit 16
}
Write-Host 'introspection JSON golden: exact match'

# --- Audio mixer WAV golden (Slice BB): an EXACT byte-for-byte match of a fresh --audio-render of the
# fixed deterministic audio scene against the committed tests/golden/audio/scene.wav. The mixer is
# INTEGER / fixed-point end to end (Q15 gains, int32 accumulate, int16 hard-clamp), so the rendered
# WAV is bit-identical run-to-run AND across compilers (MSVC vs Apple clang) -- we verify it once here
# on the Windows build (the gate). Pure CPU (hf_core), no vk*/Metal symbols, like the JSON golden. ---
Write-Host '--- audio mixer WAV golden ---'
`$audExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$audGolden = 'tests/golden/audio/scene.wav'
`$audLive = Join-Path `$env:TEMP 'hf_audio_live.wav'
& `$audExe --audio-render `$audLive 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host 'audio-render run failed'; exit 24 }
`$agBytes = [System.IO.File]::ReadAllBytes((Resolve-Path `$audGolden).Path)
`$alBytes = [System.IO.File]::ReadAllBytes(`$audLive)
`$audOk = (`$agBytes.Length -eq `$alBytes.Length)
if (`$audOk) { for (`$ai = 0; `$ai -lt `$agBytes.Length; `$ai++) { if (`$agBytes[`$ai] -ne `$alBytes[`$ai]) { `$audOk = `$false; break } } }
if (-not `$audOk) {
    Write-Host 'audio WAV golden MISMATCH (tests/golden/audio/scene.wav)'
    exit 25
}
Write-Host 'audio WAV golden: exact match'

# --- Material-graph introspection JSON golden (Slice BI): an EXACT byte-for-byte match of a fresh
# --material-introspect dump of assets/materials/showcase3.mat.json against the committed
# tests/golden/material/showcase3_graph.json. DescribeGraphJson is pure CPU (hf_core, no vk*/Metal
# symbols) and deterministic by construction, so the text is bit-identical run-to-run AND across
# backends/compilers -- we verify it once here on the Windows build (the gate), like the JSON + WAV
# goldens. No Metal round-trip needed. ---
Write-Host '--- material-graph introspection JSON golden ---'
`$matExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$matGolden = 'tests/golden/material/showcase3_graph.json'
`$matSrc = 'assets/materials/showcase3.mat.json'
`$matLive = Join-Path `$env:TEMP 'hf_matintrospect_live.json'
& `$matExe --material-introspect `$matSrc `$matLive 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host 'material-introspect run failed'; exit 26 }
`$mgBytes = [System.IO.File]::ReadAllBytes((Resolve-Path `$matGolden).Path)
`$mlBytes = [System.IO.File]::ReadAllBytes(`$matLive)
`$matOk = (`$mgBytes.Length -eq `$mlBytes.Length)
if (`$matOk) { for (`$mi = 0; `$mi -lt `$mgBytes.Length; `$mi++) { if (`$mgBytes[`$mi] -ne `$mlBytes[`$mi]) { `$matOk = `$false; break } } }
if (-not `$matOk) {
    Write-Host 'material-graph introspection JSON golden MISMATCH (tests/golden/material/showcase3_graph.json)'
    exit 27
}
Write-Host 'material-graph introspection JSON golden: exact match'

# --- Vulkan validation gate (Slice AT): run representative showcases under the Khronos validation
# layer (synchronization + core validation) and FAIL on any real validation error. This is the
# permanent oracle that keeps the engine Vulkan-validation-CLEAN: Slice AS activated the layer and
# Slice AT fixed the two latent core-validation bugs (GPU-particle descriptor invalidation +
# swapchain semaphore reuse), so any regression that re-introduces a hazard surfaces here.
#
# The layer is provided by the conan 'vulkan-validationlayers' package (see conanfile.py); it is NOT
# installed system-wide on this box, so we must point VK_LAYER_PATH at the package's bin dir (the dir
# holding VkLayer_khronos_validation.json) or the layer loads as a no-op and the gate is blind. We
# locate it by globbing the conan2 cache for the layer manifest; if VK_LAYER_PATH is already set in
# the environment we honor that instead.
Write-Host '--- Vulkan validation gate ---'
`$vkExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$layerDir = `$env:VK_LAYER_PATH
if (-not `$layerDir -or -not (Test-Path (Join-Path `$layerDir 'VkLayer_khronos_validation.json'))) {
    `$manifest = Get-ChildItem -Path (Join-Path `$env:USERPROFILE '.conan2\p') -Recurse -Filter 'VkLayer_khronos_validation.json' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (`$manifest) { `$layerDir = `$manifest.Directory.FullName }
}
if (-not `$layerDir -or -not (Test-Path (Join-Path `$layerDir 'VkLayer_khronos_validation.json'))) {
    Write-Host 'Vulkan validation layer not found (conan vulkan-validationlayers missing?) - cannot run the validation gate'
    exit 17
}
Write-Host ('validation layer dir: ' + `$layerDir)
`$env:VK_LAYER_PATH = `$layerDir
`$env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
`$env:VK_VALIDATION_FEATURE_ENABLE = 'VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION'
# Representative showcases: --shot (GPU particles + shared-base/varied-normal materials, where the
# UPDATE_AFTER_BIND bug lived) and --csm-shot (cascaded shadow atlas, a heavy multi-pass graph path).
# The GI/DDGI showcases (--ddgi-shot ... --probeinterp-shot) were ADDED in slice DQ: they run the
# lit_ddgi pass which binds the per-frame dummy shadow map, and were never validation-gated before —
# they emitted 8x VUID-vkCmdDraw-None-09600 (sampled-image layout) until DQ added the missing
# SHADER_READ_ONLY transition. Gating them here keeps the GI path validation-clean permanently.
`$vkShots = @(@('--shot'), @('--csm-shot'), @('--mt-shot'), @('--mdi-shot'), @('--bindless-shot'), @('--gpudriven-shot'), @('--gpucull-draw-shot'), @('--cluster-cull-shot'), @('--hiz-cull-shot'), @('--cluster-hiz-shot'), @('--cluster-lod-shot'), @('--visbuffer-shot'), @('--visresolve-shot'), @('--swraster-shot'), @('--swraster-gpu-shot'), @('--swraster-resolve-shot'), @('--vsm-mark-shot'), @('--vsm-render-shot'), @('--vsm-sample-shot'), @('--vsm-cache-shot'), @('--vt-feedback-shot'), @('--vt-alloc-shot'), @('--vt-pagegen-shot'), @('--vt-sample-shot'), @('--vt-cache-shot'), @('--mc-classify-shot'), @('--mc-count-shot'), @('--ddgi-shot'), @('--ddgimb-shot'), @('--ddgiocc-shot'), @('--probedist-shot'), @('--probegi-shot'), @('--probecapture-shot'), @('--probesh-shot'), @('--probeinterp-shot'), @('--meshlet-viz'))
`$vkErrors = 0
foreach (`$shot in `$vkShots) {
    `$shotArgs = `$shot + @((Join-Path `$env:TEMP ('hf_validate_' + (`$shot[0] -replace '-','') + '.png')))
    `$vlog = & `$vkExe @shotArgs 2>&1
    # A REAL validation error is a 'VUID-<name>' token, a 'SYNC-HAZARD-*', an 'UNASSIGNED-*', or any
    # '[ERROR'-tagged line. The benign duplicate-limit notice mentions the bare word 'VUID' (no
    # hyphen) inside a [WARNING] line, so we match the hyphenated token form to avoid a false positive.
    `$bad = `$vlog | Select-String -Pattern 'VUID-|SYNC-HAZARD|UNASSIGNED-|\[ERROR'
    if (`$bad) {
        Write-Host ('validation FAIL on ' + (`$shot -join ' ') + ':')
        `$bad | ForEach-Object { Write-Host ('  ' + `$_) }
        `$vkErrors += `$bad.Count
    } else {
        Write-Host ('validation clean: ' + (`$shot -join ' '))
    }
}
if (`$vkErrors -ne 0) { Write-Host ('Vulkan validation gate FAILED (' + `$vkErrors + ' error line(s))'); exit 18 }
Write-Host 'Vulkan validation gate: CLEAN (zero VUID / SYNC-HAZARD / UNASSIGNED across showcases)'

# --- Multithreaded-recording determinism oracle (Slice AU): the SAME draw-heavy scene recorded with
# 1 worker vs N workers must be BYTE-IDENTICAL. Partition + in-order secondary execution guarantee
# the draw order is independent of worker count; this asserts it on the LIVE Vulkan render. Render
# with --workers 1 and --workers 4, then compare the captured BMPs byte-for-byte. (The unit test
# parallel_record_test pins the partition; this pins the end-to-end render.) ---
Write-Host '--- multithreaded-recording 1-vs-N determinism ---'
`$mtExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$mt1 = Join-Path `$env:TEMP 'hf_mt_w1.bmp'
`$mt4 = Join-Path `$env:TEMP 'hf_mt_w4.bmp'
& `$mtExe --mt-shot `$mt1 --workers 1 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host 'mt-shot --workers 1 failed'; exit 19 }
& `$mtExe --mt-shot `$mt4 --workers 4 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host 'mt-shot --workers 4 failed'; exit 19 }
`$mt1Bytes = [System.IO.File]::ReadAllBytes(`$mt1)
`$mt4Bytes = [System.IO.File]::ReadAllBytes(`$mt4)
`$mtOk = (`$mt1Bytes.Length -eq `$mt4Bytes.Length)
if (`$mtOk) { for (`$mi = 0; `$mi -lt `$mt1Bytes.Length; `$mi++) { if (`$mt1Bytes[`$mi] -ne `$mt4Bytes[`$mi]) { `$mtOk = `$false; break } } }
if (-not `$mtOk) { Write-Host 'multithreaded-recording MISMATCH (--workers 1 != --workers 4)'; exit 20 }
Write-Host 'multithreaded-recording: --workers 1 == --workers 4 (byte-identical render)'

# --- Live runtime material authoring (Slice AW): the runtime==build-time proof. --material-live-shot
# renders the showcase material via the RUNTIME path (in-process codegen -> dxc SUBPROCESS -> SPIR-V
# -> pipeline). Because that subprocess is the SAME dxc + SAME flags the build used for
# mat_showcase.frag.hlsl, the runtime SPIR-V is byte-identical to the build-time SPIR-V, so the live
# image MUST be byte-identical to --material-shot (the committed-HLSL build-time path). Assert the two
# captured BMPs are byte-for-byte equal. Also run the headless live A->B hot-swap dry-run + assert it
# passes (the swap happened, deterministic, no crash). Both run under the validation gate above. ---
Write-Host '--- live material authoring: runtime == build-time ---'
`$lmExe = 'build/windows-msvc-debug/samples/hello_triangle/hello_triangle.exe'
`$lmBuild = Join-Path `$env:TEMP 'hf_material_build.bmp'
`$lmLive  = Join-Path `$env:TEMP 'hf_material_live.bmp'
& `$lmExe --material-shot `$lmBuild 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host '--material-shot failed'; exit 21 }
& `$lmExe --material-live-shot `$lmLive 2>`$null | Out-Null
if (`$LASTEXITCODE -ne 0) { Write-Host '--material-live-shot failed'; exit 21 }
`$bBytes = [System.IO.File]::ReadAllBytes(`$lmBuild)
`$vBytes = [System.IO.File]::ReadAllBytes(`$lmLive)
`$lmOk = (`$bBytes.Length -eq `$vBytes.Length)
if (`$lmOk) { for (`$li = 0; `$li -lt `$bBytes.Length; `$li++) { if (`$bBytes[`$li] -ne `$vBytes[`$li]) { `$lmOk = `$false; break } } }
if (-not `$lmOk) { Write-Host 'runtime != build-time: --material-live-shot != --material-shot (SPIR-V drift?)'; exit 22 }
Write-Host 'live material authoring: --material-live-shot == --material-shot (byte-identical; runtime==build-time)'

Write-Host '--- live material authoring: hot-swap dry-run ---'
`$dry = & `$lmExe --material-hotswap-dry-run 2>&1
`$dry | ForEach-Object { Write-Host `$_ }
`$dryOk = (`$LASTEXITCODE -eq 0) -and (`$dry | Select-String -SimpleMatch 'hotswap-dry-run: PASS')
if (-not `$dryOk) { Write-Host 'hot-swap dry-run FAILED'; exit 23 }
Write-Host 'live material authoring: hot-swap dry-run PASS'

exit 0
"@

    # Run the inner pipeline in a fresh powershell so the dev-shell env is isolated per run.
    # Invoke via a temp -File (not -Command) and merge the child's stderr into stdout so a benign
    # native warning (e.g. vswhere) cannot bubble up as a terminating ErrorRecord under the outer
    # $ErrorActionPreference='Stop'. We rely solely on the child's exit code for pass/fail.
    $innerFile = Join-Path $env:TEMP 'hf-verify-win.ps1'
    Set-Content -LiteralPath $innerFile -Value $inner -Encoding UTF8
    $logFile = Join-Path $env:TEMP 'hf-verify-win.log'

    # Locally relax error handling and route the child's stderr to a file so a benign native
    # warning cannot terminate the outer 'Stop' scope. We trust only the child exit code.
    $savedEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $innerFile > $logFile 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $savedEAP

    if (Test-Path $logFile) { Get-Content $logFile | ForEach-Object { Write-Host $_ } }
    if ($code -ne 0) {
        $script:winResult = 'FAIL'
        Write-Host "Windows verification FAILED (stage exit code $code)" -ForegroundColor Red
        return
    }
    $script:winResult = 'PASS'
    Write-Host "Windows verification PASSED" -ForegroundColor Green
}

# ---------------------------------------------------------------------------------------------------
# Mac / Metal
# ---------------------------------------------------------------------------------------------------
function Invoke-MacVerify {
    Write-Section "MAC / METAL"

    if (-not (Test-Path $SshKey)) { throw "SSH key not found at: $SshKey" }

    $ssh = @('ssh', '-i', $SshKey, '-o', 'StrictHostKeyChecking=accept-new', "$MacUser@$MacHost")
    $scp = @('scp', '-i', $SshKey, '-o', 'StrictHostKeyChecking=accept-new', '-q')

    # 1) tar the repo. Keep the tracked golden; drop build dirs, .git, conan user presets, and the
    #    stray top-level bring-up PNGs (NOT the golden under tests/golden/). Uses git-bash tar via WSL?
    #    No - we use the bsdtar shipped with Windows 10/11 (tar.exe on PATH).
    $tarPath = Join-Path $env:TEMP $TarName
    if (Test-Path $tarPath) { Remove-Item $tarPath -Force }

    Write-Host "--- tar repo (keeping golden) ---"
    Push-Location $RepoRoot
    try {
        # bsdtar: exclude build artifacts + .git + known stray PNGs. The golden under
        # tests/golden/ is intentionally NOT matched by these patterns.
        & tar `
            --exclude='./build' `
            --exclude='./build-metal' `
            --exclude='./.git' `
            --exclude='./CMakeUserPresets.json' `
            --exclude='./hf_editor.png' `
            --exclude='./hf_nmap_crop.png' `
            --exclude='./hf_nmap_cube.png' `
            --exclude='./hf_nmap_cubeface.png' `
            --exclude='./hf_nmap_metal_cube.png' `
            -czf $tarPath .
        if ($LASTEXITCODE -ne 0) { throw "tar failed" }
    } finally { Pop-Location }

    # Sanity: EVERY golden MUST be in the archive or its compare on the Mac can't run.
    $archiveList = (& tar -tzf $tarPath)
    foreach ($g in $Goldens) {
        $needle = "golden/metal/$($g.Name).png"
        if (-not ($archiveList | Select-String -SimpleMatch $needle)) {
            throw "golden '$($g.Name).png' missing from archive - refusing to continue"
        }
    }

    # 2) recreate the remote staging dir + copy + extract (idempotent).
    Write-Host "--- scp + extract on Mac ---"
    & $ssh[0] $ssh[1..($ssh.Count-1)] "rm -rf $MacStage && mkdir -p $MacStage"
    if ($LASTEXITCODE -ne 0) { throw "remote mkdir failed" }
    & $scp[0] $scp[1..($scp.Count-1)] $tarPath "${MacUser}@${MacHost}:$MacStage/"
    if ($LASTEXITCODE -ne 0) { throw "scp failed" }

    # 3) extract + build ONCE + loop ALL goldens. To avoid the login shell being zsh and to dodge
    #    PowerShell here-string backtick-escaping fragility, the per-golden loop is generated as a
    #    standalone bash script, scp'd to the Mac, and run with an explicit `bash`. For each
    #    (flag -> golden) pair it renders visual_test <flag> /tmp/hf_<name>.png and compares to
    #    tests/golden/metal/<name>.png at threshold 0.0, emitting a machine-parseable
    #    "RESULT <name> <diff> <PASS|FAIL>" line we parse on the Windows side. The loop does NOT
    #    abort on a single failure (no `set -e` inside it): a drifted golden must still yield the full
    #    table so a reviewer sees exactly which one(s) changed. Build failures abort hard.
    #    compare.sh prints "DIFF <value>" and exits 0 only when DIFF <= threshold (0.0 here).
    #
    #    The PAIRS data is "<name>|<flag>" lines (flag empty for the default Slice-F scene). LF-only
    #    line endings are required so bash reads them cleanly.
    $pairLines = ($Goldens | ForEach-Object { "$($_.Name)|$($_.Flag)" }) -join "`n"

    # NOTE: this bash body is a single-quoted PS here-string, so NOTHING in it is expanded/escaped by
    # PowerShell. The PAIRS block is injected by string-replacing the @@PAIRS@@ token afterwards.
    $bashBody = @'
#!/usr/bin/env bash
set -e
source ~/mac-remote-rig/env.sh
# Run from the staging dir (this script was scp'd into it).
cd "$(cd "$(dirname "$0")" && pwd)"
tar -xzf "$TARBALL"
cmake -S metal_headless -B build-metal -G Ninja >/dev/null
cmake --build build-metal >/dev/null
set +e
read -r -d '' PAIRS <<'PAIRS_EOF'
@@PAIRS@@
PAIRS_EOF
while IFS='|' read -r name flag; do
    [ -z "$name" ] && continue
    out="/tmp/hf_${name}.png"
    golden="tests/golden/metal/${name}.png"
    if [ -z "$flag" ]; then
        ./build-metal/visual_test "$out" >/dev/null 2>&1
    else
        # flag may be multi-token (e.g. "--camera 0.2,-0.1,0,3,10") -> intentional word-split.
        ./build-metal/visual_test $flag "$out" >/dev/null 2>&1
    fi
    if [ $? -ne 0 ]; then
        echo "RESULT $name RENDER_FAIL FAIL"
        continue
    fi
    diffline=$(~/mac-remote-rig/compare.sh "$golden" "$out" 0.0 2>&1)
    crc=$?
    # compare.sh prints e.g. "DIFF 0.0000 (threshold 0.0)" -> keep ONLY the numeric value (field 2).
    diffval=$(echo "$diffline" | sed -n 's/^DIFF \([^ ]*\).*/\1/p' | head -1)
    [ -z "$diffval" ] && diffval=NO-DIFF
    if [ $crc -eq 0 ]; then
        echo "RESULT $name $diffval PASS"
    else
        echo "RESULT $name $diffval FAIL"
    fi
done <<< "$PAIRS"
'@

    $bashScript = $bashBody.Replace('@@PAIRS@@', $pairLines)
    # Write LF-only, no BOM (bash chokes on CRLF and a UTF-8 BOM).
    $bashScript = $bashScript -replace "`r`n", "`n"
    $localSh = Join-Path $env:TEMP 'hf-verify-mac.sh'
    [System.IO.File]::WriteAllText($localSh, $bashScript, (New-Object System.Text.UTF8Encoding($false)))

    & $scp[0] $scp[1..($scp.Count-1)] $localSh "${MacUser}@${MacHost}:$MacStage/run-goldens.sh"
    if ($LASTEXITCODE -ne 0) { throw "scp of golden-runner script failed" }

    $out = & $ssh[0] $ssh[1..($ssh.Count-1)] "STAGE='$MacStage' TARBALL='$TarName' bash $MacStage/run-goldens.sh" 2>&1
    $code = $LASTEXITCODE
    $out | ForEach-Object { Write-Host $_ }

    # Parse the RESULT lines into the per-golden table.
    $parsed = @{}
    foreach ($line in $out) {
        $s = [string]$line
        if ($s -match '^RESULT\s+(\S+)\s+(\S+)\s+(PASS|FAIL)\s*$') {
            $parsed[$matches[1]] = @{ Diff = $matches[2]; Ok = ($matches[3] -eq 'PASS') }
        }
    }

    $results = @()
    $allOk = $true
    foreach ($g in $Goldens) {
        if ($parsed.ContainsKey($g.Name)) {
            $r = $parsed[$g.Name]
            $results += @{ Name = $g.Name; Diff = $r.Diff; Ok = $r.Ok }
            if (-not $r.Ok) { $allOk = $false }
        } else {
            # No RESULT line emitted for this golden - treat as a failure (build/loop aborted early).
            $results += @{ Name = $g.Name; Diff = 'NO-RESULT'; Ok = $false }
            $allOk = $false
        }
    }
    $script:macGoldenResults = $results

    if ($code -ne 0 -or -not $allOk) {
        $script:macResult = 'FAIL'
        $bad = ($results | Where-Object { -not $_.Ok } | ForEach-Object { $_.Name }) -join ', '
        Write-Host "Mac verification FAILED (remote exit $code; non-0.0000 goldens: $bad)" -ForegroundColor Red
        return
    }
    $script:macResult = 'PASS'
    Write-Host ("Mac verification PASSED (all {0} goldens DIFF 0.0000)" -f $Goldens.Count) -ForegroundColor Green
}

# ---------------------------------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------------------------------
$failed = $false

if (-not $SkipWindows) {
    try { Invoke-WindowsVerify } catch { $script:winResult = 'FAIL'; Write-Host "Windows verify error: $_" -ForegroundColor Red }
}
if (-not $SkipMac) {
    try { Invoke-MacVerify } catch { $script:macResult = 'FAIL'; Write-Host "Mac verify error: $_" -ForegroundColor Red }
}

Write-Section "SUMMARY"
function Show($label, $r) {
    $color = if ($r -eq 'PASS') { 'Green' } elseif ($r -eq 'SKIP') { 'Yellow' } else { 'Red' }
    Write-Host ("  {0,-22} {1}" -f $label, $r) -ForegroundColor $color
}
Show 'Windows / Vulkan (ctest)' $winResult
Show ("Mac / Metal ({0} goldens)" -f $Goldens.Count) $macResult

# Per-golden Metal table (only when the Mac portion ran).
if ($script:macGoldenResults -and $script:macGoldenResults.Count -gt 0) {
    Write-Host ""
    Write-Host "  Metal goldens (threshold 0.0 - every diff must be 0.0000):" -ForegroundColor Cyan
    Write-Host ("    {0,-16} {1,-12} {2}" -f 'golden', 'DIFF', 'result')
    Write-Host ("    {0,-16} {1,-12} {2}" -f '------', '----', '------')
    $passCount = 0
    foreach ($r in $script:macGoldenResults) {
        $tag = if ($r.Ok) { 'PASS' } else { 'FAIL' }
        if ($r.Ok) { $passCount++ }
        $color = if ($r.Ok) { 'Green' } else { 'Red' }
        Write-Host ("    {0,-16} {1,-12} {2}" -f $r.Name, $r.Diff, $tag) -ForegroundColor $color
    }
    Write-Host ("    {0} / {1} goldens at DIFF 0.0000" -f $passCount, $script:macGoldenResults.Count)
}

if ($winResult -eq 'FAIL' -or $macResult -eq 'FAIL') {
    Write-Host ""
    Write-Host "VERIFY: FAIL" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "VERIFY: PASS" -ForegroundColor Green
exit 0
