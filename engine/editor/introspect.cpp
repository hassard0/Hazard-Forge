// Hazard Forge — machine-readable engine-state introspection (see introspect.h).
//
// Emits ONE deterministic, pretty-printed JSON document describing the whole live engine + scene.
// The float format reuses scene_io's "%g on the double-promoted float" so authored values round-trip
// and two runs are byte-identical. No vk*/Metal/rhi symbols: SceneResources is used only to reverse
// mesh/texture POINTERS back to their registered NAMES (never dereferenced) — the same opaque-pointer
// contract scene_io and commands rely on.
//
// The `features`, `showcases`, and `commands` arrays are a CONSTANT, curated manifest of the shipped
// engine (they describe what the engine IS, not per-run state), so they are identical on every run
// and on every backend. Keeping them here, next to the schema, is the single source of truth an agent
// reads to discover the engine's surface.
#include "editor/introspect.h"

#include "scene/components.h"

#include <cstddef>
#include <cstdio>
#include <iterator>
#include <sstream>
#include <string>

namespace hf::editor {
namespace {

// --- Deterministic JSON emission ----------------------------------------------------------------
// A tiny indent-tracking writer over std::ostringstream. Everything is emitted in a FIXED order with
// a FIXED float format, so the output is byte-identical run-to-run. We hand-write (rather than build a
// DOM) precisely to keep the key order + whitespace stable for the text golden.

// Append a float the same way scene_io does: %g on the double-promoted value. Stable + lossless
// enough to round-trip authored values; no locale dependence (%g uses '.' in the C locale).
void AppendFloat(std::ostream& os, float f) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(f));
    os << buf;
}

void AppendVec3(std::ostream& os, const math::Vec3& v) {
    os << "[";
    AppendFloat(os, v.x); os << ", ";
    AppendFloat(os, v.y); os << ", ";
    AppendFloat(os, v.z);
    os << "]";
}

// Emit a JSON string with the minimal escaping our values ever need (mesh/texture names, descs,
// feature labels — all simple ASCII, but escape quotes/backslashes/control just in case).
void AppendString(std::ostream& os, const std::string& s) {
    os << '"';
    for (char c : s) {
        switch (c) {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n";  break;
            case '\t': os << "\\t";  break;
            case '\r': os << "\\r";  break;
            default:   os << c;       break;
        }
    }
    os << '"';
}

std::string Indent(int depth) { return std::string(static_cast<size_t>(depth) * 2, ' '); }

// --- The shipped-engine manifest (constant; describes what the engine IS). ----------------------

struct Showcase { const char* flag; const char* desc; };

// The headless render commands an agent can discover + run (the --*-shot Vulkan flags; the Metal
// visual_test flags mirror these). Curated to match samples/hello_triangle/main.cpp +
// metal_headless/visual_test.mm. An agent reads this to know what it can RENDER.
const Showcase kShowcases[] = {
    {"--shot",              "Default lit + shadowed scene (ground + grid + duck)."},
    {"--pbr-shot",          "Full PBR showcase (DamagedHelmet, metallic-roughness)."},
    {"--material-shot",     "Data-driven material graph (sphere shaded by showcase.mat.json)."},
    {"--material-live-shot","Live runtime material authoring (runtime codegen+dxc compile, no rebuild)."},
    {"--material-multi-shot","Multi-material scene (three spheres, each a distinct graph material)."},
    {"--material-normal-shot","Tangent-space normal-mapped material graph (NormalMap node perturbs the shading normal)."},
    {"--material-introspect","Dump a material graph as deterministic JSON (or DOT with --dot): nodes, params, typed ports, edges, resolved PBROutput."},
    {"--ibl-shot",          "HDR image-based lighting (equirect skybox reflections)."},
    {"--scene-shot",        "glTF scene-graph import (CesiumMilkTruck node hierarchy)."},
    {"--bloom-shot",        "HDR bloom post-process."},
    {"--ssao-shot",         "Screen-space ambient occlusion."},
    {"--skinning-shot",     "Skeletal animation (GPU skinning)."},
    {"--blend-shot",        "Animation blending between two clips."},
    {"--anim-fsm-shot",     "Animation state machine (parameter-driven states + cross-fade blend)."},
    {"--instanced-shot",    "Hardware-instanced grid."},
    {"--physics-shot",      "Rigid-body physics (impulse solver)."},
    {"--terrain-shot",      "Procedural terrain / heightmap (deterministic NxN displaced grid, lit + shadowed)."},
    {"--game-shot",         "Playable game sample (deterministic roll-a-ball collect-the-pickups)."},
    {"--net-shot",          "State replication (deterministic snapshot+delta; replica reconstructs the scene)."},
    {"--netsim-shot",       "Simulated network transport + client jitter-buffer interpolation (seeded latency/loss/reorder; smoothing hides the lossy channel)."},
    {"--netpredict-shot",   "Client prediction + server reconciliation (local-player prediction over a delayed authoritative sim with a scripted server-only impulse; rewind+replay corrects the misprediction)."},
    {"--stream-shot",       "Scene/asset streaming (distance-based cell residency + per-frame budget)."},
    {"--terrain-stream-shot","Terrain streaming with per-tile LOD (distance-banded tile residency + LOD selection)."},
    {"--hud-shot",          "Screen-space text / HUD overlay (baked 8x8 font, alpha-blended quads)."},
    {"--game-hud-shot",     "Game scene with a live SCORE HUD overlay (own golden; game.png unchanged)."},
    {"--transparency-shot", "Sorted alpha-blended transparency."},
    {"--debug-shot",        "Immediate-mode debug-line visualization."},
    {"--capstone-shot",     "Capstone scene combining the major features."},
    {"--csm-shot",          "Cascaded shadow maps (directional)."},
    {"--spot-shot",         "Spot light with perspective shadow."},
    {"--point-shadow-shot", "Omnidirectional point-light shadow (cube map)."},
    {"--clustered-shot",    "Clustered / Forward+ many-light shading."},
    {"--clustered-lights-shot", "Clustered light culling (Forward+): compute assigns 96 point lights to a 3D cluster grid by hard-radius sphere/AABB; clustered shade == brute-force all-lights byte-identical, 1<maxPerCluster<96, gpu assign total==cpu cluster::AssignLights reference."},
    {"--ssr-shot",          "Screen-space reflections."},
    {"--dof-shot",          "Depth of field (thin-lens circle-of-confusion depth gather; sharp focal plane, blurred fore/background)."},
    {"--motionblur-shot",   "Per-object + camera motion blur (velocity-gather post pass; moving content streaks along its screen-space motion vector while static content stays sharp; zero-motion render byte-identical to the un-blurred scene)."},
    {"--oit-shot",          "Order-independent transparency (Weighted Blended OIT: overlapping transparent layers composited without sorting; the transparent set rendered in two PERMUTED draw orders resolves BYTE-IDENTICALLY, proving order independence)."},
    {"--pom-shot",          "Parallax occlusion mapping (steep-parallax + binary-refine height-field march in tangent space gives a flat surface per-pixel depth + groove self-shadowing; the heightScale=0 render is BYTE-IDENTICAL to the plain normal-mapped render, proving zero UV drift)."},
    {"--gtao-shot",         "Ground-truth ambient occlusion (Jimenez et al. 2016: per-pixel horizon-search over the G-buffer depth field along several slice directions, analytic cosine-weighted visibility integral; ambient term darkened in contacts/concavities; the radius=0 render is BYTE-IDENTICAL to the no-AO scene, proving the integral normalizes to a true identity at zero occlusion)."},
    {"--sss-shot",          "Subsurface scattering (Jimenez et al. 2015 screen-space separable SSS: a horizontal + a vertical depth-aware diffusion pass blur the lit color of flagged subsurface materials so light bleeds under the surface — the soft translucent glow wrapping the terminator — while non-subsurface pixels pass through; the sssStrength=0 render is BYTE-IDENTICAL to the non-SSS lit render, proving the separable kernel collapses to the center tap at zero strength)."},
    {"--colorgrade-shot",   "Analytic color grading (post-tonemap: the lift/gamma/gain wheels (out = (gain*(c+lift*(1-c)))^(1/gamma)), the ASC-CDL slope/offset/power primary (out = max(c*slope+offset,0)^power), and a luma-preserving saturation control, composed for a cinematic teal-shadows / warm-highlights look on the tonemapped LDR result; fully analytic, no 3D-LUT; the identity-grade render (lift 0, gamma 1, gain 1, slope 1, offset 0, power 1, saturation 1) is BYTE-IDENTICAL to the ungraded tonemap render, proving the grade is a pure pass-through with no rounding/clamp bias at identity)."},
    {"--cas-shot",          "Contrast-adaptive sharpening (AMD FidelityFX CAS: a fullscreen adaptive 3x3-cross edge sharpen on the tonemapped SDR result — out = (center + w*(up+down+left+right))/(1+4w) with a negative adaptive lobe w driven by the neighborhood luminance min/max (more sharpening in low-contrast regions, less near a hard edge), then clamped per-channel to the neighborhood [min,max] so a sharpen never overshoots past the brightest/darkest neighbor — crisper detail with no ringing halos; the sharpness=0 render is BYTE-IDENTICAL to the unsharpened tonemap render, proving the sharpen is a pure pass-through with no bias/clamp drift at zero)."},
    {"--froxelfog-shot",    "Froxel volumetric fog (a true 3D view-space froxel volume: a compute pass injects per-froxel sun in-scatter + extinction, a second integrates front-to-back into per-froxel in-scatter + transmittance, a fullscreen apply samples it at each pixel's view-depth slice and composites scene*T + in-scatter; the baseDensity=0 render is BYTE-IDENTICAL to the no-fog scene, proving the inject->integrate->apply chain is a pure pass-through at zero density)."},
    {"--probegi-shot",      "DDGI beachhead — probe-grid ray-trace (the first slice of the dynamic diffuse GI arc: a world-space lattice of irradiance probes each trace 16 deterministic Fibonacci-sphere rays against the scene view-space depth field, supplied as a FLAT float SSBO read back from the G-buffer rather than a sampled depth texture, into a flat ray-hit SSBO via a pure compute pass; the GPU ray-hit SSBO is BIT-EXACT to a CPU TraceRayToDepth reference over the same depth field, and a probeCount=0 dispatch leaves the SSBO byte-identical to the cleared upload, proving the ray-trace data layer is deterministic and the disabled path a clean no-op)."},
    {"--froxellights-shot", "Per-froxel clustered-light injection (the CS+CL fusion: the 96 clustered point lights scatter through the froxel fog as colored volumetric shafts — the froxel inject pass maps each froxel to its cluster and ADDS each cluster light's windowed-attenuation * HG-phase * density in-scatter on top of the sun in-scatter; the injectLights=false render is BYTE-IDENTICAL to the CS sun-only froxel fog, proving the point-light term is purely additive, and the density=0 render is BYTE-IDENTICAL to the no-fog scene)."},
    {"--volshadows-shot",   "Volumetric shadows (sun light shafts through the fog: the froxel inject samples the sun's CSM shadow map per froxel and MULTIPLIES the sun in-scatter by the visibility so the fog has dark shadowed volumes behind geometry + bright foggy sun shafts between them; the volumetricShadows=false render forces visibility to 1 and is BYTE-IDENTICAL to the CV per-froxel clustered-light render, proving the shadow gate is a clean multiplicative identity when off, and the density=0 render is BYTE-IDENTICAL to the no-fog scene)."},
    {"--contactshadow-shot","Screen-space contact shadows (a short per-pixel depth ray-march toward the sun over the G-buffer adds the fine-scale contact occlusion the cascaded shadow map is too coarse to capture — tight dark lines at the bases/creases of small objects in contact — darkening ONLY the direct sun; the maxDist=0 render is BYTE-IDENTICAL to the no-contact-shadow scene, proving the march + sun-apply is a pure pass-through when disabled)."},
    {"--autoexposure-shot", "Auto-exposure (histogram eye adaptation: a compute pass builds a luminance histogram of the HDR scene with INTEGER atomics, a reduce pass computes the average log-luminance + the key-value target exposure, and a tonemap variant applies it so a bright scene darkens + a dark scene brightens like the eye; the adaptationEnabled=false render uses the fixed reference exposure E0 and is BYTE-IDENTICAL to the standard fixed-exposure tonemap, proving the histogram->reduce->apply chain is a pure pass-through when disabled)."},
    {"--water-shot",        "Water rendering (Gerstner waves, fresnel sky-reflect / scene-refract, sun glint)."},
    {"--ssgi-shot",         "Screen-space global illumination (color-bleed indirect diffuse)."},
    {"--ssgi-denoise-shot", "SSGI bilateral spatial denoise (edge-preserving smoothing of the indirect-diffuse buffer)."},
    {"--ssgi-temporal-shot","Temporal SSGI accumulation (fixed-N golden-angle-jittered indirect-diffuse averaging)."},
    {"--decal-shot",        "Screen-space projected decals (top-down decal box on the ground)."},
    {"--poststack-shot",    "Data-driven post-process stack (ordered tonemap/grade/chromatic/vignette/grain chain)."},
    {"--vfx-shot",          "CPU particle / VFX emitter (authorable fountain: spawn rate, lifetime curves, gravity/drag, camera-facing additive billboards)."},
    {"--volumetric-shot",   "Volumetric fog / light shafts."},
    {"--clouds-shot",       "Volumetric clouds (raymarched sunlit cumulus layer in the sky dome)."},
    {"--cloud-shadows-shot","Cloud shadows on the ground (the cloud field casts dappled shadows on the lit scene by attenuating the direct sun)."},
    {"--probe-shot",        "Reflection + irradiance probes (local cubemap GI)."},
    {"--reflprobe-shot",    "Box-projected cubemap reflections (local reflection probe: the specular reflection direction is parallax-corrected to a local box volume — the reflection ray is intersected with the probe box and the cubemap sampled toward the box-exit relative to the probe center — so reflected walls line up with the room geometry instead of appearing infinitely distant; the parallaxStrength=0 render is BYTE-IDENTICAL to the standard infinite-cubemap reflection, proving the box-projection is a pure identity at the infinite-box limit)."},
    {"--captureprobe-shot", "Runtime cubemap-capture reflection probe (the dynamic counterpart to the box-projected reflection: the scene is RENDERED into the 6 faces of a real captured cubemap from a fixed probe center, then a reflective sphere/floor samples the captured cube — box-projected — so it mirrors the actual surrounding objects rather than a static env; a captured cube face is BYTE-IDENTICAL to the scene rendered directly with that face's view/proj, proving the capture is a faithful scene render)."},
    {"--probecapture-shot", "DDGI probe radiance capture (the second slice of the dynamic diffuse GI arc: for a small world-space grid of irradiance probes, the lit scene is RENDERED into the 6 cube faces from each probe center — looping the single cubemap render target one probe at a time — and read back into a per-probe radiance store, the raw radiance the SH-encode slice will convolve into irradiance; a captured face is BYTE-IDENTICAL to the scene rendered directly with that face's view/proj, and a probeCount=0 grid leaves the radiance store byte-identical to its cleared value, proving the capture is a faithful scene render and the disabled path a clean skip-loop no-op)."},
    {"--probesh-shot",      "DDGI probe SH-encode (the third slice of the dynamic diffuse GI arc: each captured probe cubemap is encoded into 3rd-order real spherical harmonics — 9 coefficients per RGB channel — by a pure compute pass that loops a fixed host-precomputed sample set and accumulates the captured radiance into the SH basis; the sample directions + their SHBasis9 weights + solid-angle weights are host-precomputed and uploaded as exact float32 bits both the GPU encode and the CPU reference read, so the GPU per-probe SH SSBO is BIT-EXACT to a CPU SH-encode over the same read-back radiance, a zero-radiance probe encodes to an all-zero SH, and a probeCount=0 grid leaves the SH SSBO byte-identical to its cleared value — proving the SH-irradiance layer is deterministic and the disabled path a clean dispatch-0 no-op; the per-probe swatches are colored by the SH-reconstructed irradiance)."},
    {"--probeinterp-shot",  "DDGI probe trilinear SH-interpolation (the fourth slice of the dynamic diffuse GI arc: the per-probe SH irradiance records are trilinearly blended at an arbitrary world position — for a query point, the FLOOR cell whose 8 corner probes bracket it is found and those 8 probes' 3rd-order real-SH coefficients are blended by polynomial trilinear weights — by a pure compute pass, one thread per query point; the NearestProbes 8-corner lookup + the per-corner fma SH accumulation are copied VERBATIM into the shader (mad for the per-corner product) so the GPU blended-SH SSBO is BIT-EXACT to a CPU InterpolateSH reference over the same uploaded probes + query points, a query exactly at a probe position blends to that probe's SH byte-identical (the lattice-point round-trip), and a probeCount=0 grid leaves the output SSBO byte-identical to its cleared value — proving the interpolation layer is deterministic and the disabled path a clean dispatch-0 no-op; the golden renders a dense grid of swatch spheres colored by the SH-reconstructed irradiance, a smooth interpolated color field between the probes)."},
    {"--probedist-shot",    "DDGI per-probe distance-moment capture (the first slice of the DDGI VISIBILITY sub-arc — the data layer that makes the Chebyshev occlusion test possible, killing DDGI light-leak through walls: for the same small world-space grid of irradiance probes, each probe renders the scene GEOMETRY from its centre into the 6 cube faces — looping the single cubemap render target one probe at a time — writing the linear WORLD-DISTANCE d = length(worldPos - probeCentre) packed as the two moments float2(d, d*d) instead of radiance, then reads the faces back into a per-probe distance-moment store; the second moment d*d is a BARE multiply so, GIVEN the same read-back distance bytes, the GPU-written moment float2(d, d*d) is BIT-EXACT to the CPU MomentsFromDistance(d) reference (the GPU==CPU proof is on the moment-from-distance step — the sqrt-derived distance itself is render-equivalence per backend, not claimed bit-identical), a captured distance face is BYTE-IDENTICAL to the scene rendered directly with that face's view/proj through the same distance shader, and a probeCount=0 grid leaves the moment store byte-identical to its cleared value — proving the visibility capture is a faithful geometric render and the disabled path a clean skip-loop no-op; produces NO visible lighting change yet (the next slice consumes it for Chebyshev occlusion weighting); debug-viz = per-probe swatch spheres coloured by the probe's mean captured distance, near warm / far cool, over the colored room)."},
    {"--ddgi-shot",         "DDGI global illumination (the fifth and final slice of the dynamic diffuse GI arc — the VISIBLE payoff: a single-bounce indirect-diffuse term is added to a sibling lit-pass variant so a Cornell box shows colored indirect light bleed — the red wall bleeds reddish onto the floor, the green wall bleeds green — ≈ UE4-parity local GI; per pixel the 8 nearest irradiance probes are trilinearly blended into a single 3rd-order real-SH irradiance record (NearestProbes + InterpolateSH + SHEvaluate copied VERBATIM into the shader, mad for every accumulation) reconstructed toward the surface normal and added as indirect*albedo*(1-metallic)*giStrength; the ProbeSH[] rides in a fragment-stage storage buffer bound via the existing cluster path (no new RHI); the indirect term is added LAST so giStrength=0 (or a probeCount=0 grid) makes it a literal +0.0 and the render is BYTE-IDENTICAL to the no-GI baseline through the same pipeline — proving the disabled path a clean no-op that leaves all the existing lit goldens untouched, while the enabled frame shows coherent recognizable Cornell color bleed)."},
    {"--ddgimb-shot",       "DDGI multi-bounce / 2nd light bounce (the eighth slice of the dynamic diffuse GI arc — the next GI-quality leap: single-bounce DDGI (the DH–DN arc) captures only DIRECT light into the probes so indirect light bounces exactly ONCE, whereas multi-bounce feeds the 1st-bounce GI BACK into a second probe capture so a second capture sees the indirect light on the surfaces and the composite gets a brighter, more filled-in 2nd bounce — the step from UE4-style DDGI toward Lumen's infinite-bounce; done as a FIXED 2-iteration pass in one frame (deterministic, golden-able), not a temporal feedback loop: bounce 0 = the existing direct-light capture -> SH0, bounce 1 (only when bounceCount>=2) re-captures each probe with a NEW capture-with-GI fragment (a sibling of the direct-capture shader that adds the DN single-bounce indirect term — InterpolateIrradianceSH(wpos,N)*(1-metallic)*albedo*giStrength, sampling SH0) -> SH-encode -> SH1, and the composite binds SH(bounceCount-1); two SH[] storage buffers ride the EXISTING cluster path (no new RHI); INTERNALLY the bounceCount=1 render SKIPS the bounce-1 capture so it is BYTE-IDENTICAL (SHA) to the DN single-bounce render — the make-safe no-op — while the bounceCount=2 frame is measurably BRIGHTER (the bounced wall light itself bounces), the bounce-1 SH1 SSBO is BIT-EXACT to a CPU SH-encode of the read-back bounce-1 radiance, and two runs are byte-identical; golden = the brighter 2nd-bounce frame; the direct-capture shader + the DN lit_ddgi shader + their goldens untouched via sibling-shader isolation)."},
    {"--ddgiocc-shot",      "DDGI Chebyshev occlusion weighting (the second slice of the DDGI VISIBILITY sub-arc — the VISIBLE leak-fix: a sibling lit-pass variant weights each of the 8 nearest irradiance probes' SH contribution by a per-probe Chebyshev (variance-shadow-style) VISIBILITY weight read from the per-probe distance-moment cube, so a surface NO LONGER receives indirect light from probes geometrically OCCLUDED from it — a probe behind a wall stops bleeding light through it, the #1 quality gap of plain irradiance-probe GI and the real DDGI (Majercik et al. 2019) fix; for each corner probe the probe->point direction + linear distance index the distance cube (mean=m[0], variance=m[1]-mean²), cheb = dist<=mean ? 1 : variance/(variance+(dist-mean)²) is the Chebyshev upper bound in [0,1], vis = lerp(1, cheb, occlusionStrength), and the per-probe trilinear weight is scaled by vis, renormalised, and the SH blended with mad; the distance moments ride a second fragment-stage storage buffer at the next free cluster-set slot via the existing bind path (no new RHI); INTERNALLY the occlusionStrength=0 render BRANCHES to the VERBATIM DN composite path so it is BYTE-IDENTICAL (SHA) to the DN lit_ddgi render — the make-safe no-op — while the occlusionStrength=1 frame shows the leak through the occluder GONE/attenuated and differs from frame A, the visibility-math CPU mirror matches the closed form, and two runs are byte-identical; golden = the leak-fixed frame, the occluder no longer bleeds; the DN lit_ddgi shader + its goldens untouched via sibling-shader isolation)."},
    {"--planar-shot",       "Planar reflections (flat mirror-plane scene reflection: the scene is RENDERED a second time through the camera REFLECTED across the mirror plane — householder reflection + Lengyel oblique near-clip at the mirror plane + flipped front-face winding — into a 2D reflection target, then the mirror floor samples that target at its own screen-space position and blends lerp(matte, reflection, reflectivity); the reflectivity=0 render is BYTE-IDENTICAL to the matte non-reflective render, proving the reflection blend is a pure pass-through when off)."},
    {"--taa-shot",          "Temporal anti-aliasing (jittered accumulation)."},
    {"--cull-shot",         "Frustum culling visualization (overview camera + kept/culled bounds)."},
    {"--meshlet-viz",       "Virtual-geometry meshlet/cluster decomposition (the BEACHHEAD of the Nanite-style virtual-geometry arc: a sphere's index buffer is partitioned by a pure-CPU integer-deterministic Morton-sort into clusters of up to 128 triangles, each with a conservative per-cluster AABB + bounding sphere; the reordered index buffer is uploaded once and each cluster drawn as an index sub-range with its deterministic per-cluster hash color, so the sphere shows as coherent flat-colored SPATIAL cluster patches — the cluster data structure the whole DT/DU/DV arc builds on; the partition is COMPLETE (Sum(triCount)==T, every triangle covered exactly once) and two builds are BYTE-IDENTICAL; no new RHI (existing draw + push-constant surface), no GPU math (the color is a pure CPU integer hash))."},
    {"--gpu-cull-shot",     "GPU-driven culling + indirect draw (compute compaction, GPU draw count)."},
    {"--mdi-shot",          "GPU multi-draw-indirect batching (144 objects in 1 draw; MDI==per-draw byte-identical)."},
    {"--bindless-shot",     "Bindless textures (multi-texture scene via 1 descriptor array; bindless==bound byte-identical)."},
    {"--gpudriven-shot",    "Fully-GPU-driven pass (100-object multi-material scene via 1 MDI call + 1 bindless bind; gpu-driven==bound byte-identical)."},
    {"--gpucull-draw-shot", "Fully-GPU-driven-culled pass (compute frustum-cull -> compacted MDI + bindless; GPU decides AND draws; gpu-culled==CPU-culled byte-identical, gpu drawn==cpuRef)."},
    {"--cluster-cull-shot", "Virtual-geometry per-CLUSTER frustum cull -> indirect cluster draw (Slice DT: an instance grid of DS-clustered spheres is decomposed into (instance x cluster) records each with a per-cluster world bounding sphere; a compute shader frustum-culls + ORDER-compacts the survivors into the MDI command buffer + the compacted per-draw SSBO + writes the survivor count, then ONE DrawIndexedMultiIndirect renders exactly the in-frustum cluster-instances; the per-cluster analogue of the per-OBJECT GPU cull — GPU draw-count == CPU SurvivorClusterCount frustum.h reference EXACT, GPU-culled image == CPU-CullClusterInstances image byte-identical, all-in-frustum == full unculled draw byte-identical, two-run deterministic; no new RHI)."},
    {"--hiz-cull-shot",     "Hi-Z occlusion culling (depth pre-pass -> Hi-Z max-depth pyramid -> compute frustum+occlusion cull; objects fully hidden behind a near occluder are dropped; occlusion-culled==frustum-only byte-identical, occluded>0, gpu occluded==cpu hiz reference)."},
    {"--cluster-hiz-shot",  "Virtual-geometry per-CLUSTER Hi-Z occlusion cull -> indirect cluster draw (Slice DU: DT's per-cluster frustum cull PLUS CJ's Hi-Z occlusion test at cluster granularity; an occluder wall in front of a back row of DS-clustered spheres + non-occluded spheres; a depth pre-pass -> Hi-Z max-depth pyramid -> a compute shader frustum-culls THEN drops any cluster-instance whose bounding-sphere AABB is fully hidden behind the occluder, ORDER-compacts the survivors into the MDI buffer + writes the survivor count, then ONE DrawIndexedMultiIndirect renders exactly the visible cluster-instances; conservative -> occlusion-culled==frustum-only byte-identical, occlusion-disabled==DT byte-identical, survivor count DROPS, GPU count==CPU SurvivorClusterCountHiZ + GPU-culled==CPU-CullClusterInstancesHiZ byte-identical, two-run deterministic; no new RHI)."},
    {"--cluster-lod-shot",  "Virtual-geometry discrete cluster-LOD selection by projected screen-space error (Slice DV, the FINAL slice of the virtual-geometry CORE: a row of instances at INCREASING distance each picks one of 3 pre-baked discrete LOD tessellations — each DS-cluster-decomposed — by its projected screen-space error; near instances draw LOD0 (full detail, many fine clusters), far instances draw a coarser LOD (fewer coarse clusters) — a visible detail falloff; a compute shader computes the squared view distance + SelectLod (a host-precomputed projScale + std::fma + squared-distance comparison, NO sqrt -> the selected-LOD integer is bit-exact CPU==GPU) and ORDER-compacts the selected LOD's clusters into the MDI command buffer + the compacted per-draw SSBO + writes the per-instance selected-LOD int + the total command count, then ONE DrawIndexedMultiIndirect renders the selected-LOD clusters; the Nanite LOD primitive kept deterministic via pre-baked LODs (no runtime simplifier) — GPU per-instance selected-LOD == CPU SelectLod BIT-EXACT (memcmp), forceLod0/errorScale=0 == the full-detail render BYTE-IDENTICAL (the disabled path), the selected set spans >=2 distinct LODs, distance-monotonic (farther never finer), two-run deterministic; no new RHI)."},
    {"--visbuffer-shot",    "Virtual-geometry VISIBILITY BUFFER (Slice DW, the Nanite-style rendering beachhead: the same DT survivor cluster MDI draw, but the fragment output is a SINGLE uint32 (clusterID<<7 | SV_PrimitiveID) into an R32_Uint render target — the (clusterID, triangleID) visibility buffer that decouples geometry rasterization from material shading; NO lighting/resolve; the integer RT is ReadRenderTarget'd back into uint32[w*h] and proven bit-exact — visbuffer self-consistent + coverage == CPU CullClusterInstances survivor set EXACT, two renders BYTE-IDENTICAL, GPU==CPU interior coverage EXACT; the image golden is a CPU-coloring of the read-back IDs (bg->clear, else hashColor(clusterID)); the only new RHI is the additive Format::R32_Uint — existing goldens untouched)."},
    {"--mt-shot",           "Multithreaded command recording (per-thread secondaries, 1-vs-N identical)."},
    {"--editor-shot",       "Docked editor UI (Scene Hierarchy / Inspector / Stats panels around a central scene Viewport, fixed selected entity)."},
    {"--editor-edit-shot",  "Editor live-edit: apply a fixed transform+material edit to the selected entity, render the EDITED docked scene, round-trip through scene_io (DumpScene/LoadScene)."},
    {"--gizmo-shot",        "Editor selection + translate gizmo overlay."},
    {"--camera-shot",       "Scripted-pose interactive-runtime capture."},
    {"--audio-render",      "Deterministic audio mixer (integer/fixed-point voices -> 16-bit PCM WAV)."},
};

// The shipped feature/capability list (stable order). An agent reads this to know what the engine
// can DO.
const char* kFeatures[] = {
    "pbr-metallic-roughness",
    "image-based-lighting",
    "cascaded-shadow-maps",
    "spot-light-shadows",
    "point-light-shadows",
    "clustered-lighting",
    "clustered-light-culling",
    "screen-space-reflections",
    "depth-of-field",
    "motion-blur",
    "order-independent-transparency",
    "parallax-occlusion-mapping",
    "ground-truth-ambient-occlusion",
    "subsurface-scattering",
    "color-grading",
    "contrast-adaptive-sharpening",
    "froxel-volumetric-fog",
    "froxel-light-injection",
    "volumetric-shadows",
    "contact-shadows",
    "auto-exposure",
    "screen-space-global-illumination",
    "water-rendering",
    "decals",
    "post-process-stack",
    "volumetric-fog",
    "volumetric-clouds",
    "cloud-shadows",
    "ddgi-probe-raytrace",
    "ddgi-probe-capture",
    "ddgi-probe-sh-encode",
    "ddgi-probe-interp",
    "ddgi-probe-distance",
    "ddgi-global-illumination",
    "ddgi-probe-occlusion",
    "ddgi-multi-bounce",
    "reflection-irradiance-probes",
    "reflection-probe",
    "capture-reflection-probe",
    "planar-reflections",
    "temporal-anti-aliasing",
    "frustum-culling",
    "gpu-driven-culling",
    "gpu-multi-draw-indirect",
    "bindless-textures",
    "gpu-driven-rendering",
    "gpu-driven-culling-draw",
    "hiz-occlusion-culling",
    "multithreaded-recording",
    "bloom",
    "ssao",
    "transparency",
    "hardware-instancing",
    "gpu-particles",
    "particle-vfx",
    "gltf-import",
    "skeletal-animation",
    "animation-blending",
    "animation-state-machine",
    "rigid-body-physics",
    "procedural-terrain",
    "debug-visualization",
    "interactive-runtime",
    "editor-selection-gizmos",
    "docked-editor",
    "editor-live-edit",
    "automatic-barriers",
    "material-graph",
    "material-graph-introspection",
    "live-material-authoring",
    "gameplay-sample",
    "hud-text",
    "audio-mixer",
    "scene-streaming",
    "terrain-streaming-lod",
    "state-replication",
    "network-transport-sim",
    "client-prediction",
    "virtual-geometry-meshlets",
    "virtual-geometry-cluster-cull",
    "virtual-geometry-cluster-hiz",
    "virtual-geometry-cluster-lod",
    "virtual-geometry-visbuffer",
};

// One scriptable command verb (the commands.cpp ops) + its argument shape. An agent reads this to
// know how to MUTATE the scene. `args` is emitted as a JSON object of name -> type-hint strings.
struct Command { const char* op; const char* argsJson; };

const Command kCommands[] = {
    {"dump",          "{}"},
    {"list",          "{}"},
    {"set_transform", "{\"entity\": \"int\", \"position\": \"[x,y,z]?\", \"euler\": \"[x,y,z]?\", \"scale\": \"[x,y,z]?\"}"},
    {"set_material",  "{\"entity\": \"int\", \"metallic\": \"float?\", \"roughness\": \"float?\", \"baseColor\": \"string|null?\", \"normalMap\": \"string|null?\"}"},
    {"add",           "{\"mesh\": \"string\", \"baseColor\": \"string|null?\", \"normalMap\": \"string|null?\", \"metallic\": \"float?\", \"roughness\": \"float?\", \"position\": \"[x,y,z]?\", \"euler\": \"[x,y,z]?\", \"scale\": \"[x,y,z]?\"}"},
    {"remove",        "{\"entity\": \"int\"}"},
    {"capture",       "{\"path\": \"string\"}"},
    {"save_scene",    "{\"path\": \"string\"}"},
    {"introspect",    "{\"path\": \"string?\"}"},
};

}  // namespace

std::string DescribeEngine(ecs::Registry& reg, const scene::SceneResources& resources,
                           const EngineState& extra) {
    using scene::MaterialC;
    using scene::MeshC;
    using scene::TransformC;

    std::ostringstream os;

    os << "{\n";

    // --- engine ----------------------------------------------------------------------------------
    os << Indent(1) << "\"engine\": {\n";
    os << Indent(2) << "\"name\": \"Hazard Forge\",\n";
    os << Indent(2) << "\"version\": \"0.1.0\",\n";
    os << Indent(2) << "\"backends\": [\"vulkan\", \"metal\"],\n";
    os << Indent(2) << "\"activeBackend\": ";
    if (extra.backend.empty()) os << "null"; else AppendString(os, extra.backend);
    os << ",\n";
    os << Indent(2) << "\"features\": [\n";
    for (size_t i = 0; i < std::size(kFeatures); ++i) {
        os << Indent(3); AppendString(os, kFeatures[i]);
        os << (i + 1 < std::size(kFeatures) ? ",\n" : "\n");
    }
    os << Indent(2) << "]\n";
    os << Indent(1) << "},\n";

    // --- showcases -------------------------------------------------------------------------------
    os << Indent(1) << "\"showcases\": [\n";
    for (size_t i = 0; i < std::size(kShowcases); ++i) {
        os << Indent(2) << "{ \"flag\": ";
        AppendString(os, kShowcases[i].flag);
        os << ", \"desc\": ";
        AppendString(os, kShowcases[i].desc);
        os << " }";
        os << (i + 1 < std::size(kShowcases) ? ",\n" : "\n");
    }
    os << Indent(1) << "],\n";

    // --- commands --------------------------------------------------------------------------------
    os << Indent(1) << "\"commands\": [\n";
    for (size_t i = 0; i < std::size(kCommands); ++i) {
        os << Indent(2) << "{ \"op\": ";
        AppendString(os, kCommands[i].op);
        os << ", \"args\": " << kCommands[i].argsJson << " }";
        os << (i + 1 < std::size(kCommands) ? ",\n" : "\n");
    }
    os << Indent(1) << "],\n";

    // --- scene.entities (view<Transform,Mesh,Material> order — the addressing order) -------------
    int entityCount = 0;
    os << Indent(1) << "\"scene\": {\n";
    os << Indent(2) << "\"entities\": [\n";
    {
        std::ostringstream ents;
        bool first = true;
        for (auto [e, tc, mc, mat] : reg.view<TransformC, MeshC, MaterialC>()) {
            if (!first) ents << ",\n";
            first = false;
            ++entityCount;

            std::string meshName = mc.mesh ? resources.NameOfMesh(mc.mesh) : std::string();
            std::string baseName = mat.base ? resources.NameOfTexture(mat.base) : std::string();
            std::string normalName = mat.normal ? resources.NameOfTexture(mat.normal) : std::string();

            ents << Indent(3) << "{\n";
            ents << Indent(4) << "\"id\": " << e.index << ",\n";
            ents << Indent(4) << "\"generation\": " << e.generation << ",\n";
            ents << Indent(4) << "\"components\": {\n";
            ents << Indent(5) << "\"transform\": {\n";
            ents << Indent(6) << "\"position\": "; AppendVec3(ents, tc.t.position); ents << ",\n";
            ents << Indent(6) << "\"euler\": ";    AppendVec3(ents, tc.t.eulerRadians); ents << ",\n";
            ents << Indent(6) << "\"scale\": ";    AppendVec3(ents, tc.t.scale); ents << "\n";
            ents << Indent(5) << "},\n";
            ents << Indent(5) << "\"mesh\": "; AppendString(ents, meshName); ents << ",\n";
            ents << Indent(5) << "\"material\": {\n";
            ents << Indent(6) << "\"metallic\": ";  AppendFloat(ents, mat.metallic);  ents << ",\n";
            ents << Indent(6) << "\"roughness\": "; AppendFloat(ents, mat.roughness); ents << ",\n";
            ents << Indent(6) << "\"baseColor\": ";
            if (baseName.empty()) ents << "null"; else AppendString(ents, baseName);
            ents << ",\n";
            ents << Indent(6) << "\"normalMap\": ";
            if (normalName.empty()) ents << "null"; else AppendString(ents, normalName);
            ents << "\n";
            ents << Indent(5) << "}\n";
            ents << Indent(4) << "}\n";
            ents << Indent(3) << "}";
        }
        if (!first) ents << "\n";
        os << ents.str();
    }
    os << Indent(2) << "]\n";
    os << Indent(1) << "},\n";

    // --- camera (omitted when absent) ------------------------------------------------------------
    if (extra.hasCamera) {
        os << Indent(1) << "\"camera\": {\n";
        os << Indent(2) << "\"position\": "; AppendVec3(os, extra.camera.position); os << ",\n";
        os << Indent(2) << "\"yaw\": ";    AppendFloat(os, extra.camera.yaw);    os << ",\n";
        os << Indent(2) << "\"pitch\": ";  AppendFloat(os, extra.camera.pitch);  os << ",\n";
        os << Indent(2) << "\"fovDeg\": "; AppendFloat(os, extra.camera.fovDeg); os << "\n";
        os << Indent(1) << "},\n";
    }

    // --- lights ----------------------------------------------------------------------------------
    os << Indent(1) << "\"lights\": {\n";
    os << Indent(2) << "\"directional\": ";
    if (extra.hasDirectional) {
        os << "{ \"dir\": "; AppendVec3(os, extra.directional.dir);
        os << ", \"color\": "; AppendVec3(os, extra.directional.color);
        os << " }";
    } else {
        os << "null";
    }
    os << ",\n";

    os << Indent(2) << "\"points\": [";
    if (extra.points.empty()) {
        os << "],\n";
    } else {
        os << "\n";
        for (size_t i = 0; i < extra.points.size(); ++i) {
            const LightPoint& p = extra.points[i];
            os << Indent(3) << "{ \"pos\": "; AppendVec3(os, p.pos);
            os << ", \"color\": "; AppendVec3(os, p.color);
            os << ", \"radius\": "; AppendFloat(os, p.radius);
            os << ", \"intensity\": "; AppendFloat(os, p.intensity);
            os << " }";
            os << (i + 1 < extra.points.size() ? ",\n" : "\n");
        }
        os << Indent(2) << "],\n";
    }

    os << Indent(2) << "\"spots\": [";
    if (extra.spots.empty()) {
        os << "]\n";
    } else {
        os << "\n";
        for (size_t i = 0; i < extra.spots.size(); ++i) {
            const LightSpot& s = extra.spots[i];
            os << Indent(3) << "{ \"pos\": "; AppendVec3(os, s.pos);
            os << ", \"dir\": "; AppendVec3(os, s.dir);
            os << ", \"color\": "; AppendVec3(os, s.color);
            os << ", \"range\": "; AppendFloat(os, s.range);
            os << ", \"innerDeg\": "; AppendFloat(os, s.innerDeg);
            os << ", \"outerDeg\": "; AppendFloat(os, s.outerDeg);
            os << " }";
            os << (i + 1 < extra.spots.size() ? ",\n" : "\n");
        }
        os << Indent(2) << "]\n";
    }
    os << Indent(1) << "},\n";

    // --- stats -----------------------------------------------------------------------------------
    // Per-component counts via the view (each component pool is private to the registry; the view of
    // a single type yields every entity holding it).
    int transformCount = 0, meshCount = 0, materialCount = 0;
    for (auto [e, c] : reg.view<TransformC>()) { (void)e; (void)c; ++transformCount; }
    for (auto [e, c] : reg.view<MeshC>())      { (void)e; (void)c; ++meshCount; }
    for (auto [e, c] : reg.view<MaterialC>())  { (void)e; (void)c; ++materialCount; }

    os << Indent(1) << "\"stats\": {\n";
    os << Indent(2) << "\"entityCount\": " << entityCount << ",\n";
    os << Indent(2) << "\"aliveEntities\": " << reg.aliveCount() << ",\n";
    os << Indent(2) << "\"transformCount\": " << transformCount << ",\n";
    os << Indent(2) << "\"meshCount\": " << meshCount << ",\n";
    os << Indent(2) << "\"materialCount\": " << materialCount << ",\n";
    os << Indent(2) << "\"pointLightCount\": " << extra.points.size() << ",\n";
    os << Indent(2) << "\"spotLightCount\": " << extra.spots.size() << "\n";
    os << Indent(1) << "}\n";

    os << "}\n";
    return os.str();
}

}  // namespace hf::editor
