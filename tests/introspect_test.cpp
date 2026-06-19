// Unit test for the machine-readable engine-state introspection (engine/editor/introspect.{h,cpp},
// Slice AL). Build a known Registry (2 entities with transforms + materials) + a known EngineState
// (camera + directional + 2 point lights), call DescribeEngine, and assert:
//   * the JSON parses (round-trips through the vendored json.h),
//   * the scene entity count + a known entity's transform values are present,
//   * the camera fields + light values are present,
//   * the engine.backends array is exactly ["vulkan","metal"],
//   * the commands manifest includes "set_transform" (and the new "introspect"),
//   * two calls on identical inputs are BYTE-IDENTICAL (determinism).
//
// Pure C++ (hf_core), ASan-eligible like the other pure tests. Resources are opaque pointers (mapped
// to names), so the test uses distinct non-null stand-in Mesh*/ITexture* addresses — never dereferenced.
#include "editor/introspect.h"
#include "scene/components.h"
#include "scene/scene_io.h"
#include "ecs/ecs.h"

#include "json/json.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include "test_main.h"  // HF_TEST_MAIN_INIT(): headless crash-dialog suppression

using namespace hf;

static int g_fail = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("FAIL: %s\n", what); ++g_fail; }
}

// --- Minimal typed accessors over json.h (mirrors scene_io's helpers) ---------------------------
static const json_value_s* MemberOf(const json_object_s* obj, const char* key) {
    for (const json_object_element_s* el = obj->start; el; el = el->next)
        if (el->name && std::strcmp(el->name->string, key) == 0) return el->value;
    return nullptr;
}
static const json_object_s* AsObject(const json_value_s* v) {
    return (v && v->type == json_type_object) ? static_cast<const json_object_s*>(v->payload) : nullptr;
}
static const json_array_s* AsArray(const json_value_s* v) {
    return (v && v->type == json_type_array) ? static_cast<const json_array_s*>(v->payload) : nullptr;
}
static std::string AsString(const json_value_s* v) {
    if (!v || v->type != json_type_string) return {};
    const json_string_s* s = static_cast<const json_string_s*>(v->payload);
    return std::string(s->string, s->string_size);
}
static double AsNumber(const json_value_s* v, double fallback = 0.0) {
    if (!v || v->type != json_type_number) return fallback;
    return std::strtod(static_cast<const json_number_s*>(v->payload)->number, nullptr);
}
static bool approx(double a, double b) { return std::fabs(a - b) < 1e-4; }

int main() {
    HF_TEST_MAIN_INIT();
    // --- Build a known scene ---------------------------------------------------------------------
    auto* meshCube   = reinterpret_cast<scene::Mesh*>(0x1000);
    auto* meshSphere = reinterpret_cast<scene::Mesh*>(0x2000);
    auto* texChecker = reinterpret_cast<rhi::ITexture*>(0x3000);
    auto* texFlat    = reinterpret_cast<rhi::ITexture*>(0x4000);

    scene::SceneResources res;
    res.AddMesh("cube", meshCube);
    res.AddMesh("sphere", meshSphere);
    res.AddTexture("checker", texChecker);
    res.AddTexture("flat_normal", texFlat);

    ecs::Registry reg;
    {  // entity 0: cube at (1,2,3), scale 2, dielectric.
        ecs::Entity e = reg.create();
        scene::Transform t;
        t.position = {1, 2, 3};
        t.eulerRadians = {0, 0.5f, 0};
        t.scale = {2, 2, 2};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshCube});
        reg.add<scene::MaterialC>(e, {texChecker, texFlat, 0.0f, 0.5f});
    }
    {  // entity 1: sphere, metallic, no normal map.
        ecs::Entity e = reg.create();
        scene::Transform t;
        t.position = {-1, 0, 0};
        reg.add<scene::TransformC>(e, {t});
        reg.add<scene::MeshC>(e, {meshSphere});
        reg.add<scene::MaterialC>(e, {texChecker, nullptr, 1.0f, 0.15f});
    }

    // --- Known non-ECS state ---------------------------------------------------------------------
    editor::EngineState state;
    state.backend = "vulkan";
    state.hasCamera = true;
    state.camera.position = {4, 5, 6};
    state.camera.yaw = 0.25f;
    state.camera.pitch = -0.3f;
    state.camera.fovDeg = 60.0f;
    state.hasDirectional = true;
    state.directional.dir = {-0.5f, -1.0f, -0.3f};
    state.directional.color = {0.95f, 0.93f, 0.85f};
    state.points.push_back({{3, 1, 0}, {1, 0.25f, 0.2f}, 3.2f, 1.0f});
    state.points.push_back({{-3, 1, 0}, {0.2f, 1.0f, 0.35f}, 3.2f, 1.0f});

    std::string json = editor::DescribeEngine(reg, res, state);

    // --- Determinism: a second identical call must be byte-identical. ----------------------------
    std::string json2 = editor::DescribeEngine(reg, res, state);
    check(json == json2, "DescribeEngine is deterministic (byte-identical across runs)");

    // --- Parse the JSON (must be well-formed). ---------------------------------------------------
    json_parse_result_s err{};
    json_value_s* root = json_parse_ex(json.data(), json.size(), json_parse_flags_default,
                                       nullptr, nullptr, &err);
    check(root != nullptr, "DescribeEngine output parses as JSON");
    const json_object_s* top = AsObject(root);
    check(top != nullptr, "top-level value is an object");

    if (top) {
        // engine.name + backends == ["vulkan","metal"].
        const json_object_s* engine = AsObject(MemberOf(top, "engine"));
        check(engine != nullptr, "has engine object");
        if (engine) {
            check(AsString(MemberOf(engine, "name")) == "Hazard Forge", "engine.name is Hazard Forge");
            check(AsString(MemberOf(engine, "activeBackend")) == "vulkan", "engine.activeBackend is vulkan");
            const json_array_s* backends = AsArray(MemberOf(engine, "backends"));
            check(backends != nullptr && backends->length == 2, "engine.backends has 2 entries");
            if (backends && backends->length == 2) {
                const json_array_element_s* b0 = backends->start;
                const json_array_element_s* b1 = b0->next;
                check(AsString(b0->value) == "vulkan", "backends[0] == vulkan");
                check(AsString(b1->value) == "metal", "backends[1] == metal");
            }
            const json_array_s* features = AsArray(MemberOf(engine, "features"));
            check(features != nullptr && features->length > 10, "engine.features is a non-trivial list");
            // Slice AP: the temporal-anti-aliasing capability is advertised in the feature manifest.
            bool sawTaaFeature = false;
            // Slice AQ: the frustum-culling capability is advertised in the feature manifest.
            bool sawCullFeature = false;
            // Slice AR: the gpu-driven-culling capability is advertised in the feature manifest.
            bool sawGpuCullFeature = false;
            // Slice BM: the gpu-multi-draw-indirect capability is advertised in the feature manifest.
            bool sawMdiFeature = false;
            // Slice BZ: the bindless-textures capability is advertised in the feature manifest.
            bool sawBindlessFeature = false;
            // Slice CB: the gpu-driven-rendering capability is advertised in the feature manifest.
            bool sawGpuDrivenFeature = false;
            // Slice CD: the gpu-driven-culling-draw capability is advertised in the feature manifest.
            bool sawGpuCullDrawFeature = false;
            // Slice CJ: the hiz-occlusion-culling capability is advertised in the feature manifest.
            bool sawHizFeature = false;
            // Slice CL: the clustered-light-culling capability is advertised in the feature manifest.
            bool sawClusteredLightCullingFeature = false;
            // Slice AS: the automatic-barriers (render-graph resource-state tracking) capability.
            bool sawBarriersFeature = false;
            // Slice AU: the multithreaded-recording capability is advertised in the feature manifest.
            bool sawMtFeature = false;
            // Slice AV: the data-driven material-graph capability is advertised in the feature manifest.
            bool sawMatGraphFeature = false;
            // Slice AW: the live-runtime material-authoring capability is advertised.
            bool sawLiveMatFeature = false;
            // Slice AX: the playable gameplay-sample capability is advertised.
            bool sawGameplayFeature = false;
            // Slice BA: the text / HUD capability is advertised.
            bool sawHudTextFeature = false;
            // Slice BB: the deterministic audio-mixer capability is advertised.
            bool sawAudioFeature = false;
            // Slice BD: the scene/asset-streaming capability is advertised.
            bool sawStreamingFeature = false;
            // Slice BF: the procedural-terrain capability is advertised.
            bool sawTerrainFeature = false;
            // Slice BH: the screen-space projected-decals capability is advertised.
            bool sawDecalFeature = false;
            // Slice BI: the material-graph-introspection capability is advertised.
            bool sawMatIntrospectFeature = false;
            // Slice BJ: the terrain-streaming-LOD capability is advertised.
            bool sawTerrainStreamFeature = false;
            // Slice BL: the animation-state-machine capability is advertised.
            bool sawAnimFsmFeature = false;
            // Slice BN: the data-driven post-process-stack capability is advertised.
            bool sawPostStackFeature = false;
            // Slice BP: the screen-space global-illumination capability is advertised.
            bool sawSsgiFeature = false;
            // Slice CG: the depth-of-field capability is advertised in the feature manifest.
            bool sawDofFeature = false;
            // Slice CN: the motion-blur capability is advertised in the feature manifest.
            bool sawMotionBlurFeature = false;
            // Slice CO: the order-independent-transparency capability is advertised in the feature manifest.
            bool sawOitFeature = false;
            // Slice CP: the parallax-occlusion-mapping capability is advertised in the feature manifest.
            bool sawPomFeature = false;
            // Slice CR: the ground-truth-ambient-occlusion capability is advertised in the feature manifest.
            bool sawGtaoFeature = false;
            // Slice CZ: the subsurface-scattering capability is advertised in the feature manifest.
            bool sawSssFeature = false;
            // Slice DB: the color-grading capability is advertised in the feature manifest.
            bool sawColorGradeFeature = false;
            // Slice DF: the contrast-adaptive-sharpening capability is advertised in the feature manifest.
            bool sawCasFeature = false;
            // Slice DA: the reflection-probe (box-projected reflections) capability is advertised.
            bool sawReflProbeFeature = false;
            // Slice DD: the capture-reflection-probe (runtime cubemap-capture) capability is advertised.
            bool sawCaptureProbeFeature = false;
            // Slice DE: the planar-reflections capability is advertised in the feature manifest.
            bool sawPlanarFeature = false;
            // Slice CS: the froxel-volumetric-fog capability is advertised in the feature manifest.
            bool sawFroxelFogFeature = false;
            // Slice CV: the froxel-light-injection capability is advertised in the feature manifest.
            bool sawFroxelLightsFeature = false;
            // Slice CX: the volumetric-shadows capability is advertised in the feature manifest.
            bool sawVolShadowsFeature = false;
            // Slice CT: the contact-shadows capability is advertised in the feature manifest.
            bool sawContactShadowsFeature = false;
            // Slice CW: the auto-exposure capability is advertised in the feature manifest.
            bool sawAutoExposureFeature = false;
            // Slice CF: the water-rendering capability is advertised in the feature manifest.
            bool sawWaterFeature = false;
            // Slice BQ: the state-replication capability is advertised in the feature manifest.
            bool sawReplicationFeature = false;
            // Slice BU: the network-transport-sim capability is advertised in the feature manifest.
            bool sawNetsimFeature = false;
            // Slice BY: the client-prediction capability is advertised in the feature manifest.
            bool sawPredictionFeature = false;
            // Slice BT: the docked-editor capability is advertised in the feature manifest.
            bool sawDockedEditorFeature = false;
            // Slice CC: the CPU particle / VFX emitter capability is advertised in the feature manifest.
            bool sawVfxFeature = false;
            // Slice CH: the volumetric-clouds capability is advertised in the feature manifest.
            bool sawCloudsFeature = false;
            // Slice CK: the cloud-shadows capability is advertised in the feature manifest.
            bool sawCloudShadowsFeature = false;
            // Slice DH: the ddgi-probe-raytrace capability is advertised in the feature manifest.
            bool sawProbeGiFeature = false;
            // Slice DI: the ddgi-probe-capture capability is advertised in the feature manifest.
            bool sawProbeCaptureFeature = false;
            // Slice DJ: the ddgi-probe-sh-encode capability is advertised in the feature manifest.
            bool sawProbeShFeature = false;
            // Slice DL: the ddgi-probe-interp capability is advertised in the feature manifest.
            bool sawProbeInterpFeature = false;
            // Slice DO: the ddgi-probe-distance capability is advertised in the feature manifest.
            bool sawProbeDistFeature = false;
            // Slice DN: the ddgi-global-illumination capability is advertised in the feature manifest.
            bool sawDdgiFeature = false;
            // Slice DP: the ddgi-probe-occlusion capability is advertised in the feature manifest.
            bool sawProbeOcclusionFeature = false;
            // Slice DR: the ddgi-multi-bounce capability is advertised in the feature manifest.
            bool sawMultiBounceFeature = false;
            // Slice DS: the virtual-geometry-meshlets capability is advertised in the feature manifest.
            bool sawMeshletFeature = false;
            // Slice DT: the virtual-geometry-cluster-cull capability is advertised in the feature manifest.
            bool sawClusterCullFeature = false;
            // Slice DU: the virtual-geometry-cluster-hiz capability is advertised in the feature manifest.
            bool sawClusterHizFeature = false;
            // Slice DV: the virtual-geometry-cluster-lod capability is advertised in the feature manifest.
            bool sawClusterLodFeature = false;
            // Slice DW: the virtual-geometry-visbuffer capability is advertised in the feature manifest.
            bool sawVisbufferFeature = false;
            // Slice DX: the virtual-geometry-visresolve capability is advertised in the feature manifest.
            bool sawVisresolveFeature = false;
            // Slice VA: the virtual-shadow-maps-marking capability is advertised in the feature manifest.
            bool sawVsmMarkFeature = false;
            // Slice VB: the virtual-shadow-maps-render capability is advertised in the feature manifest.
            bool sawVsmRenderFeature = false;
            // Slice VC: the virtual-shadow-maps-sample capability is advertised in the feature manifest.
            bool sawVsmSampleFeature = false;
            // Slice VD: the virtual-shadow-maps-cache capability is advertised in the feature manifest.
            bool sawVsmCacheFeature = false;
            // Slice MC1: the gpu-isosurface-meshing-classify capability is advertised in the manifest.
            bool sawMcClassifyFeature = false;
            // Slice MC2: the gpu-isosurface-meshing-count capability is advertised in the manifest.
            bool sawMcCountFeature = false;
            // Slice MC3: the gpu-isosurface-meshing-emit capability is advertised in the manifest.
            bool sawMcEmitFeature = false;
            // Slice MC4: the gpu-isosurface-meshing-interp capability is advertised in the manifest.
            bool sawMcInterpFeature = false;
            // Slice MC5: the gpu-isosurface-meshing-render capability is advertised in the manifest.
            bool sawMcRenderFeature = false;
            // Slice MC6: the gpu-isosurface-meshing-normals capability is advertised in the manifest.
            bool sawMcNormalsFeature = false;
            // Slice FPX1: the deterministic-fixedpoint-physics-integrate capability is advertised.
            bool sawFpxIntegrateFeature = false;
            // Slice FPX2: the deterministic-fixedpoint-physics-broadphase capability is advertised.
            bool sawFpxBroadphaseFeature = false;
            // Slice FPX3: the deterministic-fixedpoint-physics-solve capability is advertised.
            bool sawFpxSolveFeature = false;
            // Slice FPX4: the deterministic-fixedpoint-physics-orient capability is advertised.
            bool sawFpxOrientFeature = false;
            // Slice FPX5: the deterministic-fixedpoint-physics-lockstep capability is advertised.
            bool sawFpxLockstepFeature = false;
            // Slice FPX6: the deterministic-fixedpoint-physics-render capability is advertised.
            bool sawFpxRenderFeature = false;
            // Slice NAV1: the deterministic-navmesh-rasterization capability is advertised.
            bool sawNavRasterFeature = false;
            // Slice NAV2: the deterministic-navmesh-distancefield capability is advertised.
            bool sawNavDistanceFeature = false;
            // Slice NAV3: the deterministic-navmesh-regions capability is advertised.
            bool sawNavRegionFeature = false;
            // Slice NAV4: the deterministic-navmesh-polymesh capability is advertised.
            bool sawNavPolymeshFeature = false;
            // Slice NAV5: the deterministic-navmesh-pathfinding capability is advertised.
            bool sawNavPathFeature = false;
            // Slice NAV6: the deterministic-navmesh-render capability is advertised.
            bool sawNavRenderFeature = false;
            // Slice CL1: the deterministic-cloth-integrate capability is advertised.
            bool sawClothIntegrateFeature = false;
            // Slice CL2: the deterministic-cloth-constraints capability is advertised.
            bool sawClothConstraintsFeature = false;
            // Slice CL3: the deterministic-cloth-solve capability is advertised.
            bool sawClothSolveFeature = false;
            // Slice CL4: the deterministic-cloth-collide capability is advertised.
            bool sawClothCollideFeature = false;
            // Slice CL5: the deterministic-cloth-lockstep capability is advertised.
            bool sawClothLockstepFeature = false;
            // Slice CL6: the deterministic-cloth-render capability is advertised.
            bool sawClothRenderFeature = false;
            // Slice FL1: the deterministic-fluid-integrate capability is advertised.
            bool sawFluidIntegrateFeature = false;
            // Slice FL2: the deterministic-fluid-neighbors capability is advertised.
            bool sawFluidNeighborsFeature = false;
            // Slice FL3: the deterministic-fluid-density capability is advertised.
            bool sawFluidDensityFeature = false;
            // Slice FL4: the deterministic-fluid-solve capability is advertised.
            bool sawFluidSolveFeature = false;
            // Slice FL5: the deterministic-fluid-lockstep capability is advertised.
            bool sawFluidLockstepFeature = false;
            // Slice FL6: the deterministic-fluid-render capability is advertised.
            bool sawFluidRenderFeature = false;
            // Slice GR1: the deterministic-grain-integrate capability is advertised.
            bool sawGrainIntegrateFeature = false;
            // Slice GR2: the deterministic-grain-neighbors capability is advertised.
            bool sawGrainNeighborsFeature = false;
            // Slice GR3: the deterministic-grain-contact capability is advertised.
            bool sawGrainContactFeature = false;
            // Slice GR4: the deterministic-grain-friction capability is advertised.
            bool sawGrainFrictionFeature = false;
            // Slice GR5: the deterministic-grain-lockstep capability is advertised.
            bool sawGrainLockstepFeature = false;
            // Slice GR6: the deterministic-grain-render capability is advertised.
            bool sawGrainRenderFeature = false;
            // Slice CP1: the deterministic-couple-query capability is advertised.
            bool sawCoupleQueryFeature = false;
            // Slice CP2: the deterministic-couple-buoyancy capability is advertised.
            bool sawCoupleBuoyancyFeature = false;
            // Slice CP3: the deterministic-couple-displace capability is advertised.
            bool sawCoupleDisplaceFeature = false;
            // Slice CP4: the deterministic-couple-step capability is advertised.
            bool sawCoupleStepFeature = false;
            // Slice CP5: the deterministic-couple-lockstep capability is advertised.
            bool sawCoupleLockstepFeature = false;
            // Slice CP6: the deterministic-couple-render capability is advertised.
            bool sawCoupleRenderFeature = false;
            // Slice CG1: the deterministic-cgrain-query capability is advertised.
            bool sawCgrainQueryFeature = false;
            // Slice CG2: the deterministic-cgrain-support capability is advertised.
            bool sawCgrainSupportFeature = false;
            // Slice CG3: the deterministic-cgrain-displace capability is advertised.
            bool sawCgrainDisplaceFeature = false;
            // Slice CG4: the deterministic-cgrain-step capability is advertised.
            bool sawCgrainStepFeature = false;
            // Slice CG5: the deterministic-cgrain-lockstep capability is advertised.
            bool sawCgrainLockstepFeature = false;
            // Slice CG6: the deterministic-cgrain-render capability is advertised.
            bool sawCgrainRenderFeature = false;
            // Slice GF1: the deterministic-cgf-query capability is advertised.
            bool sawCgfQueryFeature = false;
            // Slice GF2: the deterministic-cgf-buoyancy capability is advertised.
            bool sawCgfBuoyancyFeature = false;
            // Slice GF3: the deterministic-cgf-displace capability is advertised.
            bool sawCgfDisplaceFeature = false;
            // Slice GF4: the deterministic-cgf-step capability is advertised.
            bool sawCgfStepFeature = false;
            // Slice GF5: the deterministic-cgf-lockstep capability is advertised.
            bool sawCgfLockstepFeature = false;
            // Slice GF6: the deterministic-cgf-render capability is advertised.
            bool sawCgfRenderFeature = false;
            // Slice FR1: the deterministic-fract-cells capability is advertised.
            bool sawFractCellsFeature = false;
            // Slice FR2: the deterministic-fract-fragments capability is advertised.
            bool sawFractFragmentsFeature = false;
            // Slice FR3: the deterministic-fract-break capability is advertised.
            bool sawFractBreakFeature = false;
            // Slice FR4: the deterministic-fract-step capability is advertised.
            bool sawFractStepFeature = false;
            // Slice VT1: the runtime-virtual-texturing-feedback capability is advertised in the manifest.
            bool sawVtFeedbackFeature = false;
            // Slice VT2: the runtime-virtual-texturing-allocate capability is advertised in the manifest.
            bool sawVtAllocFeature = false;
            // Slice VT3: the runtime-virtual-texturing-pagegen capability is advertised in the manifest.
            bool sawVtPagegenFeature = false;
            // Slice VT4: the runtime-virtual-texturing-sample capability is advertised in the manifest.
            bool sawVtSampleFeature = false;
            // Slice VT5: the runtime-virtual-texturing-cache capability is advertised in the manifest.
            bool sawVtCacheFeature = false;
            // Slice SW1: the nanite-software-raster capability is advertised in the feature manifest.
            bool sawSwRasterFeature = false;
            // Slice SW2: the nanite-software-raster-gpu capability is advertised in the feature manifest.
            bool sawSwRasterGpuFeature = false;
            // Slice SW4: the nanite-software-raster-resolve capability is advertised in the feature manifest.
            bool sawSwRasterResolveFeature = false;
            // Slice BX: the editor-live-edit capability is advertised in the feature manifest.
            bool sawEditorLiveEditFeature = false;
            if (features)
                for (const json_array_element_s* el = features->start; el; el = el->next) {
                    if (AsString(el->value) == "temporal-anti-aliasing") sawTaaFeature = true;
                    if (AsString(el->value) == "frustum-culling") sawCullFeature = true;
                    if (AsString(el->value) == "gpu-driven-culling") sawGpuCullFeature = true;
                    if (AsString(el->value) == "gpu-multi-draw-indirect") sawMdiFeature = true;
                    if (AsString(el->value) == "bindless-textures") sawBindlessFeature = true;
                    if (AsString(el->value) == "gpu-driven-rendering") sawGpuDrivenFeature = true;
                    if (AsString(el->value) == "gpu-driven-culling-draw") sawGpuCullDrawFeature = true;
                    if (AsString(el->value) == "hiz-occlusion-culling") sawHizFeature = true;
                    if (AsString(el->value) == "clustered-light-culling") sawClusteredLightCullingFeature = true;
                    if (AsString(el->value) == "automatic-barriers") sawBarriersFeature = true;
                    if (AsString(el->value) == "multithreaded-recording") sawMtFeature = true;
                    if (AsString(el->value) == "material-graph") sawMatGraphFeature = true;
                    if (AsString(el->value) == "live-material-authoring") sawLiveMatFeature = true;
                    if (AsString(el->value) == "gameplay-sample") sawGameplayFeature = true;
                    if (AsString(el->value) == "hud-text") sawHudTextFeature = true;
                    if (AsString(el->value) == "audio-mixer") sawAudioFeature = true;
                    if (AsString(el->value) == "scene-streaming") sawStreamingFeature = true;
                    if (AsString(el->value) == "procedural-terrain") sawTerrainFeature = true;
                    if (AsString(el->value) == "decals") sawDecalFeature = true;
                    if (AsString(el->value) == "material-graph-introspection") sawMatIntrospectFeature = true;
                    if (AsString(el->value) == "terrain-streaming-lod") sawTerrainStreamFeature = true;
                    if (AsString(el->value) == "animation-state-machine") sawAnimFsmFeature = true;
                    if (AsString(el->value) == "post-process-stack") sawPostStackFeature = true;
                    if (AsString(el->value) == "screen-space-global-illumination") sawSsgiFeature = true;
                    if (AsString(el->value) == "depth-of-field") sawDofFeature = true;
                    if (AsString(el->value) == "motion-blur") sawMotionBlurFeature = true;
                    if (AsString(el->value) == "order-independent-transparency") sawOitFeature = true;
                    if (AsString(el->value) == "parallax-occlusion-mapping") sawPomFeature = true;
                    if (AsString(el->value) == "ground-truth-ambient-occlusion") sawGtaoFeature = true;
                    if (AsString(el->value) == "subsurface-scattering") sawSssFeature = true;
                    if (AsString(el->value) == "color-grading") sawColorGradeFeature = true;
                    if (AsString(el->value) == "contrast-adaptive-sharpening") sawCasFeature = true;
                    if (AsString(el->value) == "reflection-probe") sawReflProbeFeature = true;
                    if (AsString(el->value) == "capture-reflection-probe") sawCaptureProbeFeature = true;
                    if (AsString(el->value) == "planar-reflections") sawPlanarFeature = true;
                    if (AsString(el->value) == "froxel-volumetric-fog") sawFroxelFogFeature = true;
                    if (AsString(el->value) == "froxel-light-injection") sawFroxelLightsFeature = true;
                    if (AsString(el->value) == "volumetric-shadows") sawVolShadowsFeature = true;
                    if (AsString(el->value) == "contact-shadows") sawContactShadowsFeature = true;
                    if (AsString(el->value) == "auto-exposure") sawAutoExposureFeature = true;
                    if (AsString(el->value) == "water-rendering") sawWaterFeature = true;
                    if (AsString(el->value) == "state-replication") sawReplicationFeature = true;
                    if (AsString(el->value) == "network-transport-sim") sawNetsimFeature = true;
                    if (AsString(el->value) == "client-prediction") sawPredictionFeature = true;
                    if (AsString(el->value) == "docked-editor") sawDockedEditorFeature = true;
                    if (AsString(el->value) == "editor-live-edit") sawEditorLiveEditFeature = true;
                    if (AsString(el->value) == "particle-vfx") sawVfxFeature = true;
                    if (AsString(el->value) == "volumetric-clouds") sawCloudsFeature = true;
                    if (AsString(el->value) == "cloud-shadows") sawCloudShadowsFeature = true;
                    if (AsString(el->value) == "ddgi-probe-raytrace") sawProbeGiFeature = true;
                    if (AsString(el->value) == "ddgi-probe-capture") sawProbeCaptureFeature = true;
                    if (AsString(el->value) == "ddgi-probe-sh-encode") sawProbeShFeature = true;
                    if (AsString(el->value) == "ddgi-probe-interp") sawProbeInterpFeature = true;
                    if (AsString(el->value) == "ddgi-probe-distance") sawProbeDistFeature = true;
                    if (AsString(el->value) == "ddgi-global-illumination") sawDdgiFeature = true;
                    if (AsString(el->value) == "ddgi-probe-occlusion") sawProbeOcclusionFeature = true;
                    if (AsString(el->value) == "ddgi-multi-bounce") sawMultiBounceFeature = true;
                    if (AsString(el->value) == "virtual-geometry-meshlets") sawMeshletFeature = true;
                    if (AsString(el->value) == "virtual-geometry-cluster-cull") sawClusterCullFeature = true;
                    if (AsString(el->value) == "virtual-geometry-cluster-hiz") sawClusterHizFeature = true;
                    if (AsString(el->value) == "virtual-geometry-cluster-lod") sawClusterLodFeature = true;
                    if (AsString(el->value) == "virtual-geometry-visbuffer") sawVisbufferFeature = true;
                    if (AsString(el->value) == "virtual-geometry-visresolve") sawVisresolveFeature = true;
                    if (AsString(el->value) == "virtual-shadow-maps-marking") sawVsmMarkFeature = true;
                    if (AsString(el->value) == "virtual-shadow-maps-render") sawVsmRenderFeature = true;
                    if (AsString(el->value) == "virtual-shadow-maps-sample") sawVsmSampleFeature = true;
                    if (AsString(el->value) == "virtual-shadow-maps-cache") sawVsmCacheFeature = true;
                    if (AsString(el->value) == "runtime-virtual-texturing-feedback") sawVtFeedbackFeature = true;
                    if (AsString(el->value) == "runtime-virtual-texturing-allocate") sawVtAllocFeature = true;
                    if (AsString(el->value) == "runtime-virtual-texturing-pagegen") sawVtPagegenFeature = true;
                    if (AsString(el->value) == "runtime-virtual-texturing-sample") sawVtSampleFeature = true;
                    if (AsString(el->value) == "runtime-virtual-texturing-cache") sawVtCacheFeature = true;
                    if (AsString(el->value) == "gpu-isosurface-meshing-classify") sawMcClassifyFeature = true;
                    if (AsString(el->value) == "gpu-isosurface-meshing-count") sawMcCountFeature = true;
                    if (AsString(el->value) == "gpu-isosurface-meshing-emit") sawMcEmitFeature = true;
                    if (AsString(el->value) == "gpu-isosurface-meshing-interp") sawMcInterpFeature = true;
                    if (AsString(el->value) == "gpu-isosurface-meshing-render") sawMcRenderFeature = true;
                    if (AsString(el->value) == "gpu-isosurface-meshing-normals") sawMcNormalsFeature = true;
                    if (AsString(el->value) == "deterministic-fixedpoint-physics-integrate") sawFpxIntegrateFeature = true;
                    if (AsString(el->value) == "deterministic-fixedpoint-physics-broadphase") sawFpxBroadphaseFeature = true;
                    if (AsString(el->value) == "deterministic-fixedpoint-physics-solve") sawFpxSolveFeature = true;
                    if (AsString(el->value) == "deterministic-fixedpoint-physics-orient") sawFpxOrientFeature = true;
                    if (AsString(el->value) == "deterministic-fixedpoint-physics-lockstep") sawFpxLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-fixedpoint-physics-render") sawFpxRenderFeature = true;
                    if (AsString(el->value) == "deterministic-navmesh-rasterization") sawNavRasterFeature = true;
                    if (AsString(el->value) == "deterministic-navmesh-distancefield") sawNavDistanceFeature = true;
                    if (AsString(el->value) == "deterministic-navmesh-regions") sawNavRegionFeature = true;
                    if (AsString(el->value) == "deterministic-navmesh-polymesh") sawNavPolymeshFeature = true;
                    if (AsString(el->value) == "deterministic-navmesh-pathfinding") sawNavPathFeature = true;
                    if (AsString(el->value) == "deterministic-navmesh-render") sawNavRenderFeature = true;
                    if (AsString(el->value) == "deterministic-cloth-integrate") sawClothIntegrateFeature = true;
                    if (AsString(el->value) == "deterministic-cloth-constraints") sawClothConstraintsFeature = true;
                    if (AsString(el->value) == "deterministic-cloth-solve") sawClothSolveFeature = true;
                    if (AsString(el->value) == "deterministic-cloth-collide") sawClothCollideFeature = true;
                    if (AsString(el->value) == "deterministic-cloth-lockstep") sawClothLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-cloth-render") sawClothRenderFeature = true;
                    if (AsString(el->value) == "deterministic-fluid-integrate") sawFluidIntegrateFeature = true;
                    if (AsString(el->value) == "deterministic-fluid-neighbors") sawFluidNeighborsFeature = true;
                    if (AsString(el->value) == "deterministic-fluid-density") sawFluidDensityFeature = true;
                    if (AsString(el->value) == "deterministic-fluid-solve") sawFluidSolveFeature = true;
                    if (AsString(el->value) == "deterministic-fluid-lockstep") sawFluidLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-fluid-render") sawFluidRenderFeature = true;
                    if (AsString(el->value) == "deterministic-grain-integrate") sawGrainIntegrateFeature = true;
                    if (AsString(el->value) == "deterministic-grain-neighbors") sawGrainNeighborsFeature = true;
                    if (AsString(el->value) == "deterministic-grain-contact") sawGrainContactFeature = true;
                    if (AsString(el->value) == "deterministic-grain-friction") sawGrainFrictionFeature = true;
                    if (AsString(el->value) == "deterministic-grain-lockstep") sawGrainLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-grain-render") sawGrainRenderFeature = true;
                    if (AsString(el->value) == "deterministic-couple-query") sawCoupleQueryFeature = true;
                    if (AsString(el->value) == "deterministic-couple-buoyancy") sawCoupleBuoyancyFeature = true;
                    if (AsString(el->value) == "deterministic-couple-displace") sawCoupleDisplaceFeature = true;
                    if (AsString(el->value) == "deterministic-couple-step") sawCoupleStepFeature = true;
                    if (AsString(el->value) == "deterministic-couple-lockstep") sawCoupleLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-couple-render") sawCoupleRenderFeature = true;
                    if (AsString(el->value) == "deterministic-cgrain-query") sawCgrainQueryFeature = true;
                    if (AsString(el->value) == "deterministic-cgrain-support") sawCgrainSupportFeature = true;
                    if (AsString(el->value) == "deterministic-cgrain-displace") sawCgrainDisplaceFeature = true;
                    if (AsString(el->value) == "deterministic-cgrain-step") sawCgrainStepFeature = true;
                    if (AsString(el->value) == "deterministic-cgrain-lockstep") sawCgrainLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-cgrain-render") sawCgrainRenderFeature = true;
                    if (AsString(el->value) == "deterministic-cgf-query") sawCgfQueryFeature = true;
                    if (AsString(el->value) == "deterministic-cgf-buoyancy") sawCgfBuoyancyFeature = true;
                    if (AsString(el->value) == "deterministic-cgf-displace") sawCgfDisplaceFeature = true;
                    if (AsString(el->value) == "deterministic-cgf-step") sawCgfStepFeature = true;
                    if (AsString(el->value) == "deterministic-cgf-lockstep") sawCgfLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-cgf-render") sawCgfRenderFeature = true;
                    if (AsString(el->value) == "deterministic-fract-cells") sawFractCellsFeature = true;
                    if (AsString(el->value) == "deterministic-fract-fragments") sawFractFragmentsFeature = true;
                    if (AsString(el->value) == "deterministic-fract-break") sawFractBreakFeature = true;
                    if (AsString(el->value) == "deterministic-fract-step") sawFractStepFeature = true;
                    if (AsString(el->value) == "nanite-software-raster") sawSwRasterFeature = true;
                    if (AsString(el->value) == "nanite-software-raster-gpu") sawSwRasterGpuFeature = true;
                    if (AsString(el->value) == "nanite-software-raster-resolve") sawSwRasterResolveFeature = true;
                }
            check(sawTaaFeature, "engine.features includes temporal-anti-aliasing");
            check(sawCullFeature, "engine.features includes frustum-culling");
            check(sawGpuCullFeature, "engine.features includes gpu-driven-culling");
            check(sawMdiFeature, "engine.features includes gpu-multi-draw-indirect");
            check(sawBindlessFeature, "engine.features includes bindless-textures");
            check(sawGpuDrivenFeature, "engine.features includes gpu-driven-rendering");
            check(sawGpuCullDrawFeature, "engine.features includes gpu-driven-culling-draw");
            check(sawHizFeature, "engine.features includes hiz-occlusion-culling");
            check(sawClusteredLightCullingFeature, "engine.features includes clustered-light-culling");
            check(sawBarriersFeature, "engine.features includes automatic-barriers");
            check(sawMtFeature, "engine.features includes multithreaded-recording");
            check(sawMatGraphFeature, "engine.features includes material-graph");
            check(sawLiveMatFeature, "engine.features includes live-material-authoring");
            check(sawGameplayFeature, "engine.features includes gameplay-sample");
            check(sawHudTextFeature, "engine.features includes hud-text");
            check(sawAudioFeature, "engine.features includes audio-mixer");
            check(sawStreamingFeature, "engine.features includes scene-streaming");
            check(sawTerrainFeature, "engine.features includes procedural-terrain");
            check(sawDecalFeature, "engine.features includes decals");
            check(sawMatIntrospectFeature, "engine.features includes material-graph-introspection");
            check(sawTerrainStreamFeature, "engine.features includes terrain-streaming-lod");
            check(sawAnimFsmFeature, "engine.features includes animation-state-machine");
            check(sawPostStackFeature, "engine.features includes post-process-stack");
            check(sawSsgiFeature, "engine.features includes screen-space-global-illumination");
            check(sawDofFeature, "engine.features includes depth-of-field");
            check(sawMotionBlurFeature, "engine.features includes motion-blur");
            check(sawOitFeature, "engine.features includes order-independent-transparency");
            check(sawPomFeature, "engine.features includes parallax-occlusion-mapping");
            check(sawGtaoFeature, "engine.features includes ground-truth-ambient-occlusion");
            check(sawSssFeature, "engine.features includes subsurface-scattering");
            check(sawColorGradeFeature, "engine.features includes color-grading");
            check(sawCasFeature, "engine.features includes contrast-adaptive-sharpening");
        check(sawReflProbeFeature, "engine.features includes reflection-probe");
            check(sawCaptureProbeFeature, "engine.features includes capture-reflection-probe");
            check(sawPlanarFeature, "engine.features includes planar-reflections");
            check(sawFroxelFogFeature, "engine.features includes froxel-volumetric-fog");
            check(sawFroxelLightsFeature, "engine.features includes froxel-light-injection");
            check(sawVolShadowsFeature, "engine.features includes volumetric-shadows");
            check(sawContactShadowsFeature, "engine.features includes contact-shadows");
            check(sawAutoExposureFeature, "engine.features includes auto-exposure");
            check(sawWaterFeature, "engine.features includes water-rendering");
            check(sawReplicationFeature, "engine.features includes state-replication");
            check(sawNetsimFeature, "engine.features includes network-transport-sim");
            check(sawPredictionFeature, "engine.features includes client-prediction");
            check(sawDockedEditorFeature, "engine.features includes docked-editor");
            check(sawEditorLiveEditFeature, "engine.features includes editor-live-edit");
            check(sawVfxFeature, "engine.features includes particle-vfx");
            check(sawCloudsFeature, "engine.features includes volumetric-clouds");
            check(sawCloudShadowsFeature, "engine.features includes cloud-shadows");
            check(sawProbeGiFeature, "engine.features includes ddgi-probe-raytrace");
            check(sawProbeCaptureFeature, "engine.features includes ddgi-probe-capture");
            check(sawProbeShFeature, "engine.features includes ddgi-probe-sh-encode");
            check(sawProbeInterpFeature, "engine.features includes ddgi-probe-interp");
            check(sawProbeDistFeature, "engine.features includes ddgi-probe-distance");
            check(sawDdgiFeature, "engine.features includes ddgi-global-illumination");
            check(sawProbeOcclusionFeature, "engine.features includes ddgi-probe-occlusion");
            check(sawMultiBounceFeature, "engine.features includes ddgi-multi-bounce");
            check(sawMeshletFeature, "engine.features includes virtual-geometry-meshlets");
            check(sawClusterCullFeature, "engine.features includes virtual-geometry-cluster-cull");
            check(sawClusterHizFeature, "engine.features includes virtual-geometry-cluster-hiz");
            check(sawClusterLodFeature, "engine.features includes virtual-geometry-cluster-lod");
            check(sawVisbufferFeature, "engine.features includes virtual-geometry-visbuffer");
            check(sawVisresolveFeature, "engine.features includes virtual-geometry-visresolve");
            check(sawVsmMarkFeature, "engine.features includes virtual-shadow-maps-marking");
            check(sawVsmRenderFeature, "engine.features includes virtual-shadow-maps-render");
            check(sawVsmSampleFeature, "engine.features includes virtual-shadow-maps-sample");
            check(sawVsmCacheFeature, "engine.features includes virtual-shadow-maps-cache");
            check(sawVtFeedbackFeature, "engine.features includes runtime-virtual-texturing-feedback");
            check(sawVtAllocFeature, "engine.features includes runtime-virtual-texturing-allocate");
            check(sawVtPagegenFeature, "engine.features includes runtime-virtual-texturing-pagegen");
            check(sawVtSampleFeature, "engine.features includes runtime-virtual-texturing-sample");
            check(sawVtCacheFeature, "engine.features includes runtime-virtual-texturing-cache");
            check(sawMcClassifyFeature, "engine.features includes gpu-isosurface-meshing-classify");
            check(sawMcCountFeature, "engine.features includes gpu-isosurface-meshing-count");
            check(sawMcEmitFeature, "engine.features includes gpu-isosurface-meshing-emit");
            check(sawMcInterpFeature, "engine.features includes gpu-isosurface-meshing-interp");
            check(sawMcRenderFeature, "engine.features includes gpu-isosurface-meshing-render");
            check(sawMcNormalsFeature, "engine.features includes gpu-isosurface-meshing-normals");
            check(sawFpxIntegrateFeature, "engine.features includes deterministic-fixedpoint-physics-integrate");
            check(sawFpxBroadphaseFeature, "engine.features includes deterministic-fixedpoint-physics-broadphase");
            check(sawFpxSolveFeature, "engine.features includes deterministic-fixedpoint-physics-solve");
            check(sawFpxOrientFeature, "engine.features includes deterministic-fixedpoint-physics-orient");
            check(sawFpxLockstepFeature, "engine.features includes deterministic-fixedpoint-physics-lockstep");
            check(sawFpxRenderFeature, "engine.features includes deterministic-fixedpoint-physics-render");
            check(sawNavRasterFeature, "engine.features includes deterministic-navmesh-rasterization");
            check(sawNavDistanceFeature, "engine.features includes deterministic-navmesh-distancefield");
            check(sawNavRegionFeature, "engine.features includes deterministic-navmesh-regions");
            check(sawNavPolymeshFeature, "engine.features includes deterministic-navmesh-polymesh");
            check(sawNavPathFeature, "engine.features includes deterministic-navmesh-pathfinding");
            check(sawNavRenderFeature, "engine.features includes deterministic-navmesh-render");
            check(sawClothIntegrateFeature, "engine.features includes deterministic-cloth-integrate");
            check(sawClothConstraintsFeature, "engine.features includes deterministic-cloth-constraints");
            check(sawClothSolveFeature, "engine.features includes deterministic-cloth-solve");
            check(sawClothCollideFeature, "engine.features includes deterministic-cloth-collide");
            check(sawClothLockstepFeature, "engine.features includes deterministic-cloth-lockstep");
            check(sawClothRenderFeature, "engine.features includes deterministic-cloth-render");
            check(sawFluidIntegrateFeature, "engine.features includes deterministic-fluid-integrate");
            check(sawFluidNeighborsFeature, "engine.features includes deterministic-fluid-neighbors");
            check(sawFluidDensityFeature, "engine.features includes deterministic-fluid-density");
        check(sawFluidSolveFeature, "engine.features includes deterministic-fluid-solve");
        check(sawFluidLockstepFeature, "engine.features includes deterministic-fluid-lockstep");
        check(sawFluidRenderFeature, "engine.features includes deterministic-fluid-render");
        check(sawGrainIntegrateFeature, "engine.features includes deterministic-grain-integrate");
        check(sawGrainNeighborsFeature, "engine.features includes deterministic-grain-neighbors");
        check(sawGrainContactFeature, "engine.features includes deterministic-grain-contact");
        check(sawGrainFrictionFeature, "engine.features includes deterministic-grain-friction");
        check(sawGrainLockstepFeature, "engine.features includes deterministic-grain-lockstep");
        check(sawGrainRenderFeature, "engine.features includes deterministic-grain-render");
        check(sawCoupleQueryFeature, "engine.features includes deterministic-couple-query");
        check(sawCoupleBuoyancyFeature, "engine.features includes deterministic-couple-buoyancy");
        check(sawCoupleDisplaceFeature, "engine.features includes deterministic-couple-displace");
        check(sawCoupleStepFeature, "engine.features includes deterministic-couple-step");
        check(sawCoupleLockstepFeature, "engine.features includes deterministic-couple-lockstep");
        check(sawCoupleRenderFeature, "engine.features includes deterministic-couple-render");
        check(sawCgrainQueryFeature, "engine.features includes deterministic-cgrain-query");
        check(sawCgrainSupportFeature, "engine.features includes deterministic-cgrain-support");
        check(sawCgrainDisplaceFeature, "engine.features includes deterministic-cgrain-displace");
        check(sawCgrainStepFeature, "engine.features includes deterministic-cgrain-step");
        check(sawCgrainLockstepFeature, "engine.features includes deterministic-cgrain-lockstep");
        check(sawCgrainRenderFeature, "engine.features includes deterministic-cgrain-render");
        check(sawCgfQueryFeature, "engine.features includes deterministic-cgf-query");
        check(sawCgfBuoyancyFeature, "engine.features includes deterministic-cgf-buoyancy");
        check(sawCgfDisplaceFeature, "engine.features includes deterministic-cgf-displace");
        check(sawCgfStepFeature, "engine.features includes deterministic-cgf-step");
        check(sawCgfLockstepFeature, "engine.features includes deterministic-cgf-lockstep");
        check(sawCgfRenderFeature, "engine.features includes deterministic-cgf-render");
        check(sawFractCellsFeature, "engine.features includes deterministic-fract-cells");
        check(sawFractFragmentsFeature, "engine.features includes deterministic-fract-fragments");
        check(sawFractBreakFeature, "engine.features includes deterministic-fract-break");
        check(sawFractStepFeature, "engine.features includes deterministic-fract-step");
            check(sawSwRasterFeature, "engine.features includes nanite-software-raster");
            check(sawSwRasterGpuFeature, "engine.features includes nanite-software-raster-gpu");
            check(sawSwRasterResolveFeature, "engine.features includes nanite-software-raster-resolve");
        }

        // commands manifest includes set_transform + introspect.
        const json_array_s* commands = AsArray(MemberOf(top, "commands"));
        check(commands != nullptr, "has commands array");
        bool sawSetTransform = false, sawIntrospect = false;
        if (commands) {
            for (const json_array_element_s* el = commands->start; el; el = el->next) {
                const json_object_s* c = AsObject(el->value);
                if (!c) continue;
                std::string op = AsString(MemberOf(c, "op"));
                if (op == "set_transform") sawSetTransform = true;
                if (op == "introspect")    sawIntrospect = true;
            }
        }
        check(sawSetTransform, "commands manifest includes set_transform");
        check(sawIntrospect, "commands manifest includes introspect");

        // showcases present + non-trivial.
        const json_array_s* showcases = AsArray(MemberOf(top, "showcases"));
        check(showcases != nullptr && showcases->length > 10, "showcases is a non-trivial list");
        // Slice AP: the --taa-shot showcase flag is listed in the showcase manifest.
        bool sawTaaShot = false;
        // Slice AQ: the --cull-shot showcase flag is listed in the showcase manifest.
        bool sawCullShot = false;
        // Slice AR: the --gpu-cull-shot showcase flag is listed in the showcase manifest.
        bool sawGpuCullShot = false;
        // Slice BM: the --mdi-shot showcase flag is listed in the showcase manifest.
        bool sawMdiShot = false;
        // Slice BZ: the --bindless-shot showcase flag is listed in the showcase manifest.
        bool sawBindlessShot = false;
        // Slice CB: the --gpudriven-shot showcase flag is listed in the showcase manifest.
        bool sawGpuDrivenShot = false;
        // Slice CD: the --gpucull-draw-shot showcase flag is listed in the showcase manifest.
        bool sawGpuCullDrawShot = false;
        // Slice CJ: the --hiz-cull-shot showcase flag is listed in the showcase manifest.
        bool sawHizCullShot = false;
        // Slice CL: the --clustered-lights-shot showcase flag is listed in the showcase manifest.
        bool sawClusteredLightsShot = false;
        // Slice AU: the --mt-shot showcase flag is listed in the showcase manifest.
        bool sawMtShot = false;
        // Slice AV: the --material-shot showcase flag is listed in the showcase manifest.
        bool sawMaterialShot = false;
        // Slice AW: the --material-live-shot showcase flag is listed in the showcase manifest.
        bool sawMaterialLiveShot = false;
        // Slice AZ: the --material-multi-shot showcase flag is listed in the showcase manifest.
        bool sawMaterialMultiShot = false;
        // Slice BE: the --material-normal-shot showcase flag is listed in the showcase manifest.
        bool sawMaterialNormalShot = false;
        // Slice AX: the --game-shot showcase flag is listed in the showcase manifest.
        bool sawGameShot = false;
        // Slice BA: the --hud-shot + --game-hud-shot showcase flags are listed.
        bool sawHudShot = false, sawGameHudShot = false;
        // Slice BB: the --audio-render showcase flag is listed in the showcase manifest.
        bool sawAudioRender = false;
        // Slice BD: the --stream-shot showcase flag is listed in the showcase manifest.
        bool sawStreamShot = false;
        // Slice BJ: the --terrain-stream-shot showcase flag is listed in the showcase manifest.
        bool sawTerrainStreamShot = false;
        // Slice BF: the --terrain-shot showcase flag is listed in the showcase manifest.
        bool sawTerrainShot = false;
        // Slice BH: the --decal-shot showcase flag is listed in the showcase manifest.
        bool sawDecalShot = false;
        // Slice BI: the --material-introspect showcase flag is listed in the showcase manifest.
        bool sawMaterialIntrospectShot = false;
        // Slice BL: the --anim-fsm-shot showcase flag is listed in the showcase manifest.
        bool sawAnimFsmShot = false;
        // Slice BN: the --poststack-shot showcase flag is listed in the showcase manifest.
        bool sawPostStackShot = false;
        // Slice BP: the --ssgi-shot showcase flag is listed in the showcase manifest.
        bool sawSsgiShot = false;
        // Slice CG: the --dof-shot showcase flag is listed in the showcase manifest.
        bool sawDofShot = false;
        // Slice CN: the --motionblur-shot showcase flag is listed in the showcase manifest.
        bool sawMotionBlurShot = false;
        // Slice CO: the --oit-shot showcase flag is listed in the showcase manifest.
        bool sawOitShot = false;
        // Slice CP: the --pom-shot showcase flag is listed in the showcase manifest.
        bool sawPomShot = false;
        // Slice CR: the --gtao-shot showcase flag is listed in the showcase manifest.
        bool sawGtaoShot = false;
        // Slice CZ: the --sss-shot showcase flag is listed in the showcase manifest.
        bool sawSssShot = false;
        // Slice DB: the --colorgrade-shot showcase flag is listed in the showcase manifest.
        bool sawColorGradeShot = false;
        // Slice DF: the --cas-shot showcase flag is listed in the showcase manifest.
        bool sawCasShot = false;
        // Slice DA: the --reflprobe-shot showcase flag is listed in the showcase manifest.
        bool sawReflProbeShot = false;
        // Slice DD: the --captureprobe-shot showcase flag is listed in the showcase manifest.
        bool sawCaptureProbeShot = false;
        // Slice DE: the --planar-shot showcase flag is listed in the showcase manifest.
        bool sawPlanarShot = false;
        // Slice CS: the --froxelfog-shot showcase flag is listed in the showcase manifest.
        bool sawFroxelFogShot = false;
        // Slice DH: the --probegi-shot showcase flag is listed in the showcase manifest.
        bool sawProbeGiShot = false;
        // Slice DI: the --probecapture-shot showcase flag is listed in the showcase manifest.
        bool sawProbeCaptureShot = false;
        // Slice DJ: the --probesh-shot showcase flag is listed in the showcase manifest.
        bool sawProbeShShot = false;
        // Slice DL: the --probeinterp-shot showcase flag is listed in the showcase manifest.
        bool sawProbeInterpShot = false;
        // Slice DO: the --probedist-shot showcase flag is listed in the showcase manifest.
        bool sawProbeDistShot = false;
        // Slice DN: the --ddgi-shot showcase flag is listed in the showcase manifest.
        bool sawDdgiShot = false;
        // Slice DR: the --ddgimb-shot showcase flag is listed in the showcase manifest.
        bool sawDdgiMbShot = false;
        // Slice DS: the --meshlet-viz showcase flag is listed in the showcase manifest.
        bool sawMeshletVizShot = false;
        // Slice DT: the --cluster-cull-shot showcase flag is listed in the showcase manifest.
        bool sawClusterCullShot = false;
        // Slice DU: the --cluster-hiz-shot showcase flag is listed in the showcase manifest.
        bool sawClusterHizShot = false;
        // Slice DV: the --cluster-lod-shot showcase flag is listed in the showcase manifest.
        bool sawClusterLodShot = false;
        // Slice DW: the --visbuffer-shot showcase flag is listed in the showcase manifest.
        bool sawVisbufferShot = false;
        // Slice DX: the --visresolve-shot showcase flag is listed in the showcase manifest.
        bool sawVisresolveShot = false;
        // Slice SW1: the --swraster-shot showcase flag is listed in the showcase manifest.
        bool sawSwRasterShot = false;
        // Slice SW2: the --swraster-gpu-shot showcase flag is listed in the showcase manifest.
        bool sawSwRasterGpuShot = false;
        // Slice SW4: the --swraster-resolve-shot showcase flag is listed in the showcase manifest.
        bool sawSwRasterResolveShot = false;
        // Slice VA: the --vsm-mark-shot showcase flag is listed in the showcase manifest.
        bool sawVsmMarkShot = false;
        // Slice VB: the --vsm-render-shot showcase flag is listed in the showcase manifest.
        bool sawVsmRenderShot = false;
        // Slice VC: the --vsm-sample-shot showcase flag is listed in the showcase manifest.
        bool sawVsmSampleShot = false;
        // Slice VD: the --vsm-cache-shot showcase flag is listed in the showcase manifest.
        bool sawVsmCacheShot = false;
        // Slice VT1: the --vt-feedback-shot showcase flag is listed in the showcase manifest.
        bool sawVtFeedbackShot = false;
        // Slice VT2: the --vt-alloc-shot showcase flag is listed in the showcase manifest.
        bool sawVtAllocShot = false;
        // Slice VT3: the --vt-pagegen-shot showcase flag is listed in the showcase manifest.
        bool sawVtPagegenShot = false;
        // Slice VT4: the --vt-sample-shot showcase flag is listed in the showcase manifest.
        bool sawVtSampleShot = false;
        // Slice VT5: the --vt-cache-shot showcase flag is listed in the showcase manifest.
        bool sawVtCacheShot = false;
        // Slice MC1: the --mc-classify-shot showcase flag is listed in the showcase manifest.
        bool sawMcClassifyShot = false;
        // Slice MC2: the --mc-count-shot showcase flag is listed in the showcase manifest.
        bool sawMcCountShot = false;
        // Slice MC3: the --mc-emit-shot showcase flag is listed in the showcase manifest.
        bool sawMcEmitShot = false;
        // Slice MC4: the --mc-interp-shot showcase flag is listed in the showcase manifest.
        bool sawMcInterpShot = false;
        // Slice MC5: the --mc-render-shot showcase flag is listed in the showcase manifest.
        bool sawMcRenderShot = false;
        // Slice MC6: the --mc-normals-shot showcase flag is listed in the showcase manifest.
        bool sawMcNormalsShot = false;
        // Slice FPX1: the --fpx-shot showcase flag is listed in the showcase manifest.
        bool sawFpxShot = false;
        // Slice FL1: the --fluid-integrate-shot showcase flag is listed in the showcase manifest.
        bool sawFluidIntegrateShot = false;
        // Slice GR1: the --grain-integrate-shot showcase flag is listed in the showcase manifest.
        bool sawGrainIntegrateShot = false;
        // Slice GR2: the --grain-neighbors-shot showcase flag is listed in the showcase manifest.
        bool sawGrainNeighborsShot = false;
        // Slice GR3: the --grain-contact-shot showcase flag is listed in the showcase manifest.
        bool sawGrainContactShot = false;
        // Slice GR4: the --grain-friction-shot showcase flag is listed in the showcase manifest.
        bool sawGrainFrictionShot = false;
        // Slice GR5: the --grain-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawGrainLockstepShot = false;
        // Slice GR6: the --grain-render-shot showcase flag is listed in the showcase manifest.
        bool sawGrainRenderShot = false;
        // Slice CP1: the --couple-query-shot showcase flag is listed in the showcase manifest.
        bool sawCoupleQueryShot = false;
        // Slice CP2: the --couple-buoyancy-shot showcase flag is listed in the showcase manifest.
        bool sawCoupleBuoyancyShot = false;
        // Slice CP3: the --couple-displace-shot showcase flag is listed in the showcase manifest.
        bool sawCoupleDisplaceShot = false;
        // Slice CP4: the --couple-step-shot showcase flag is listed in the showcase manifest.
        bool sawCoupleStepShot = false;
        // Slice CP5: the --couple-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawCoupleLockstepShot = false;
        // Slice CP6: the --couple-render-shot showcase flag is listed in the showcase manifest.
        bool sawCoupleRenderShot = false;
        // Slice CG1: the --cgrain-query-shot showcase flag is listed in the showcase manifest.
        bool sawCgrainQueryShot = false;
        // Slice CG2: the --cgrain-support-shot showcase flag is listed in the showcase manifest.
        bool sawCgrainSupportShot = false;
        // Slice CG3: the --cgrain-displace-shot showcase flag is listed in the showcase manifest.
        bool sawCgrainDisplaceShot = false;
        // Slice CG4: the --cgrain-step-shot showcase flag is listed in the showcase manifest.
        bool sawCgrainStepShot = false;
        // Slice CG5: the --cgrain-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawCgrainLockstepShot = false;
        // Slice CG6: the --cgrain-render-shot showcase flag is listed in the showcase manifest.
        bool sawCgrainRenderShot = false;
        // Slice GF1: the --cgf-query-shot showcase flag is listed in the showcase manifest.
        bool sawCgfQueryShot = false;
        // Slice GF2: the --cgf-buoyancy-shot showcase flag is listed in the showcase manifest.
        bool sawCgfBuoyancyShot = false;
        // Slice GF3: the --cgf-displace-shot showcase flag is listed in the showcase manifest.
        bool sawCgfDisplaceShot = false;
        // Slice GF4: the --cgf-step-shot showcase flag is listed in the showcase manifest.
        bool sawCgfStepShot = false;
        // Slice GF6: the --cgf-render-shot showcase flag is listed in the showcase manifest.
        bool sawCgfRenderShot = false;
        // Slice FR1: the --fract-cells-shot showcase flag is listed in the showcase manifest.
        bool sawFractCellsShot = false;
        // Slice FR2: the --fract-fragments-shot showcase flag is listed in the showcase manifest.
        bool sawFractFragmentsShot = false;
        // Slice FR3: the --fract-break-shot showcase flag is listed in the showcase manifest.
        bool sawFractBreakShot = false;
        // Slice FR4: the --fract-step-shot showcase flag is listed in the showcase manifest.
        bool sawFractStepShot = false;
        // Slice FL2: the --fluid-neighbors-shot showcase flag is listed in the showcase manifest.
        bool sawFluidNeighborsShot = false;
        // Slice FL3: the --fluid-density-shot showcase flag is listed in the showcase manifest.
        bool sawFluidDensityShot = false;
        // Slice FL4: the --fluid-solve-shot showcase flag is listed in the showcase manifest.
        bool sawFluidSolveShot = false;
        // Slice FL5: the --fluid-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawFluidLockstepShot = false;
        // Slice FL6: the --fluid-render-shot showcase flag is listed in the showcase manifest.
        bool sawFluidRenderShot = false;
        // Slice CL1: the --cloth-integrate-shot showcase flag is listed in the showcase manifest.
        bool sawClothIntegrateShot = false;
        // Slice CL2: the --cloth-edges-shot showcase flag is listed in the showcase manifest.
        bool sawClothEdgesShot = false;
        // Slice CL3: the --cloth-solve-shot showcase flag is listed in the showcase manifest.
        bool sawClothSolveShot = false;
        // Slice CL4: the --cloth-collide-shot showcase flag is listed in the showcase manifest.
        bool sawClothCollideShot = false;
        // Slice CL5: the --cloth-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawClothLockstepShot = false;
        // Slice CL6: the --cloth-render-shot showcase flag is listed in the showcase manifest.
        bool sawClothRenderShot = false;
        // Slice FPX2: the --fpx-pairs-shot showcase flag is listed in the showcase manifest.
        bool sawFpxPairsShot = false;
        // Slice FPX3: the --fpx-solve-shot showcase flag is listed in the showcase manifest.
        bool sawFpxSolveShot = false;
        // Slice FPX4: the --fpx-orient-shot showcase flag is listed in the showcase manifest.
        bool sawFpxOrientShot = false;
        // Slice FPX5: the --fpx-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawFpxLockstepShot = false;
        // Slice FPX6: the --fpx-render-shot showcase flag is listed in the showcase manifest.
        bool sawFpxRenderShot = false;
        // Slice NAV1: the --nav-raster-shot showcase flag is listed in the showcase manifest.
        bool sawNavRasterShot = false;
        // Slice NAV2: the --nav-distance-shot showcase flag is listed in the showcase manifest.
        bool sawNavDistanceShot = false;
        // Slice NAV3: the --nav-region-shot showcase flag is listed in the showcase manifest.
        bool sawNavRegionShot = false;
        // Slice NAV4: the --nav-polymesh-shot showcase flag is listed in the showcase manifest.
        bool sawNavPolymeshShot = false;
        // Slice NAV5: the --nav-path-shot showcase flag is listed in the showcase manifest.
        bool sawNavPathShot = false;
        // Slice NAV6: the --nav-render-shot showcase flag is listed in the showcase manifest.
        bool sawNavRenderShot = false;
        // Slice CV: the --froxellights-shot showcase flag is listed in the showcase manifest.
        bool sawFroxelLightsShot = false;
        // Slice CX: the --volshadows-shot showcase flag is listed in the showcase manifest.
        bool sawVolShadowsShot = false;
        // Slice CT: the --contactshadow-shot showcase flag is listed in the showcase manifest.
        bool sawContactShadowShot = false;
        // Slice CW: the --autoexposure-shot showcase flag is listed in the showcase manifest.
        bool sawAutoExposureShot = false;
        // Slice CF: the --water-shot showcase flag is listed in the showcase manifest.
        bool sawWaterShot = false;
        // Slice BR: the --ssgi-denoise-shot showcase flag is listed in the showcase manifest.
        bool sawSsgiDenoiseShot = false;
        // Slice BV: the --ssgi-temporal-shot showcase flag is listed in the showcase manifest.
        bool sawSsgiTemporalShot = false;
        // Slice BQ: the --net-shot showcase flag is listed in the showcase manifest.
        bool sawNetShot = false;
        // Slice BU: the --netsim-shot showcase flag is listed in the showcase manifest.
        bool sawNetsimShot = false;
        // Slice BY: the --netpredict-shot showcase flag is listed in the showcase manifest.
        bool sawNetpredictShot = false;
        // Slice BT: the --editor-shot showcase flag is listed in the showcase manifest.
        bool sawEditorShot = false;
        // Slice BX: the --editor-edit-shot showcase flag is listed in the showcase manifest.
        bool sawEditorEditShot = false;
        // Slice CC: the --vfx-shot showcase flag is listed in the showcase manifest.
        bool sawVfxShot = false;
        // Slice CH: the --clouds-shot showcase flag is listed in the showcase manifest.
        bool sawCloudsShot = false;
        // Slice CK: the --cloud-shadows-shot showcase flag is listed in the showcase manifest.
        bool sawCloudShadowsShot = false;
        if (showcases)
            for (const json_array_element_s* el = showcases->start; el; el = el->next) {
                const json_object_s* s = AsObject(el->value);
                if (s && AsString(MemberOf(s, "flag")) == "--taa-shot") sawTaaShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cull-shot") sawCullShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gpu-cull-shot") sawGpuCullShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mdi-shot") sawMdiShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--bindless-shot") sawBindlessShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gpudriven-shot") sawGpuDrivenShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gpucull-draw-shot") sawGpuCullDrawShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--hiz-cull-shot") sawHizCullShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--clustered-lights-shot") sawClusteredLightsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mt-shot") sawMtShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--material-shot") sawMaterialShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--material-live-shot") sawMaterialLiveShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--material-multi-shot") sawMaterialMultiShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--material-normal-shot") sawMaterialNormalShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--game-shot") sawGameShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--hud-shot") sawHudShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--game-hud-shot") sawGameHudShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--audio-render") sawAudioRender = true;
                if (s && AsString(MemberOf(s, "flag")) == "--stream-shot") sawStreamShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--terrain-stream-shot") sawTerrainStreamShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--terrain-shot") sawTerrainShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--decal-shot") sawDecalShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--material-introspect") sawMaterialIntrospectShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--anim-fsm-shot") sawAnimFsmShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--poststack-shot") sawPostStackShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ssgi-shot") sawSsgiShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--dof-shot") sawDofShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--motionblur-shot") sawMotionBlurShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--oit-shot") sawOitShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--pom-shot") sawPomShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gtao-shot") sawGtaoShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--sss-shot") sawSssShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--colorgrade-shot") sawColorGradeShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cas-shot") sawCasShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--reflprobe-shot") sawReflProbeShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--captureprobe-shot") sawCaptureProbeShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--planar-shot") sawPlanarShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--froxelfog-shot") sawFroxelFogShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--probegi-shot") sawProbeGiShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--probecapture-shot") sawProbeCaptureShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--probesh-shot") sawProbeShShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--probeinterp-shot") sawProbeInterpShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--probedist-shot") sawProbeDistShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ddgi-shot") sawDdgiShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ddgimb-shot") sawDdgiMbShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--meshlet-viz") sawMeshletVizShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cluster-cull-shot") sawClusterCullShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cluster-hiz-shot") sawClusterHizShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cluster-lod-shot") sawClusterLodShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--visbuffer-shot") sawVisbufferShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--visresolve-shot") sawVisresolveShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--swraster-shot") sawSwRasterShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--swraster-gpu-shot") sawSwRasterGpuShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--swraster-resolve-shot") sawSwRasterResolveShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vsm-mark-shot") sawVsmMarkShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vsm-render-shot") sawVsmRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vsm-sample-shot") sawVsmSampleShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vsm-cache-shot") sawVsmCacheShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vt-feedback-shot") sawVtFeedbackShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vt-alloc-shot") sawVtAllocShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vt-pagegen-shot") sawVtPagegenShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vt-sample-shot") sawVtSampleShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vt-cache-shot") sawVtCacheShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mc-classify-shot") sawMcClassifyShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mc-count-shot") sawMcCountShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mc-emit-shot") sawMcEmitShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mc-interp-shot") sawMcInterpShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mc-render-shot") sawMcRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mc-normals-shot") sawMcNormalsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fpx-shot") sawFpxShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fluid-integrate-shot") sawFluidIntegrateShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--grain-integrate-shot") sawGrainIntegrateShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--grain-neighbors-shot") sawGrainNeighborsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--couple-query-shot") sawCoupleQueryShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--couple-buoyancy-shot") sawCoupleBuoyancyShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--couple-displace-shot") sawCoupleDisplaceShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--couple-step-shot") sawCoupleStepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--couple-lockstep-shot") sawCoupleLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--couple-render-shot") sawCoupleRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cgrain-query-shot") sawCgrainQueryShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cgrain-support-shot") sawCgrainSupportShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cgrain-displace-shot") sawCgrainDisplaceShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cgrain-step-shot") sawCgrainStepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cgrain-lockstep-shot") sawCgrainLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cgrain-render-shot") sawCgrainRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cgf-query-shot") sawCgfQueryShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cgf-buoyancy-shot") sawCgfBuoyancyShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cgf-displace-shot") sawCgfDisplaceShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cgf-step-shot") sawCgfStepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cgf-render-shot") sawCgfRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fract-cells-shot") sawFractCellsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fract-fragments-shot") sawFractFragmentsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fract-break-shot") sawFractBreakShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fract-step-shot") sawFractStepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--grain-contact-shot") sawGrainContactShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--grain-friction-shot") sawGrainFrictionShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--grain-lockstep-shot") sawGrainLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--grain-render-shot") sawGrainRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fluid-neighbors-shot") sawFluidNeighborsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fluid-density-shot") sawFluidDensityShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fluid-solve-shot") sawFluidSolveShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fluid-lockstep-shot") sawFluidLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fluid-render-shot") sawFluidRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cloth-integrate-shot") sawClothIntegrateShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cloth-edges-shot") sawClothEdgesShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cloth-solve-shot") sawClothSolveShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cloth-collide-shot") sawClothCollideShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cloth-lockstep-shot") sawClothLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cloth-render-shot") sawClothRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fpx-pairs-shot") sawFpxPairsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fpx-solve-shot") sawFpxSolveShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fpx-orient-shot") sawFpxOrientShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fpx-lockstep-shot") sawFpxLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fpx-render-shot") sawFpxRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--nav-raster-shot") sawNavRasterShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--nav-distance-shot") sawNavDistanceShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--nav-region-shot") sawNavRegionShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--nav-polymesh-shot") sawNavPolymeshShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--nav-path-shot") sawNavPathShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--nav-render-shot") sawNavRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--froxellights-shot") sawFroxelLightsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--volshadows-shot") sawVolShadowsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--contactshadow-shot") sawContactShadowShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--autoexposure-shot") sawAutoExposureShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--water-shot") sawWaterShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ssgi-denoise-shot") sawSsgiDenoiseShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ssgi-temporal-shot") sawSsgiTemporalShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--net-shot") sawNetShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--netsim-shot") sawNetsimShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--netpredict-shot") sawNetpredictShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--editor-shot") sawEditorShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--editor-edit-shot") sawEditorEditShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vfx-shot") sawVfxShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--clouds-shot") sawCloudsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--cloud-shadows-shot") sawCloudShadowsShot = true;
            }
        check(sawTaaShot, "showcases manifest includes --taa-shot");
        check(sawCullShot, "showcases manifest includes --cull-shot");
        check(sawGpuCullShot, "showcases manifest includes --gpu-cull-shot");
        check(sawMdiShot, "showcases manifest includes --mdi-shot");
        check(sawBindlessShot, "showcases manifest includes --bindless-shot");
        check(sawGpuDrivenShot, "showcases manifest includes --gpudriven-shot");
        check(sawGpuCullDrawShot, "showcases manifest includes --gpucull-draw-shot");
        check(sawHizCullShot, "showcases manifest includes --hiz-cull-shot");
        check(sawClusteredLightsShot, "showcases manifest includes --clustered-lights-shot");
        check(sawMtShot, "showcases manifest includes --mt-shot");
        check(sawMaterialShot, "showcases manifest includes --material-shot");
        check(sawMaterialLiveShot, "showcases manifest includes --material-live-shot");
        check(sawMaterialMultiShot, "showcases manifest includes --material-multi-shot");
        check(sawMaterialNormalShot, "showcases manifest includes --material-normal-shot");
        check(sawGameShot, "showcases manifest includes --game-shot");
        check(sawHudShot, "showcases manifest includes --hud-shot");
        check(sawGameHudShot, "showcases manifest includes --game-hud-shot");
        check(sawAudioRender, "showcases manifest includes --audio-render");
        check(sawStreamShot, "showcases manifest includes --stream-shot");
        check(sawTerrainStreamShot, "showcases manifest includes --terrain-stream-shot");
        check(sawTerrainShot, "showcases manifest includes --terrain-shot");
        check(sawDecalShot, "showcases manifest includes --decal-shot");
        check(sawMaterialIntrospectShot, "showcases manifest includes --material-introspect");
        check(sawAnimFsmShot, "showcases manifest includes --anim-fsm-shot");
        check(sawPostStackShot, "showcases manifest includes --poststack-shot");
        check(sawSsgiShot, "showcases manifest includes --ssgi-shot");
        check(sawDofShot, "showcases manifest includes --dof-shot");
        check(sawMotionBlurShot, "showcases manifest includes --motionblur-shot");
        check(sawOitShot, "showcases manifest includes --oit-shot");
        check(sawPomShot, "showcases manifest includes --pom-shot");
        check(sawGtaoShot, "showcases manifest includes --gtao-shot");
        check(sawSssShot, "showcases manifest includes --sss-shot");
        check(sawColorGradeShot, "showcases manifest includes --colorgrade-shot");
        check(sawCasShot, "showcases manifest includes --cas-shot");
        check(sawReflProbeShot, "showcases manifest includes --reflprobe-shot");
        check(sawCaptureProbeShot, "showcases manifest includes --captureprobe-shot");
        check(sawPlanarShot, "showcases manifest includes --planar-shot");
        check(sawFroxelFogShot, "showcases manifest includes --froxelfog-shot");
        check(sawProbeGiShot, "showcases manifest includes --probegi-shot");
        check(sawProbeCaptureShot, "showcases manifest includes --probecapture-shot");
        check(sawProbeShShot, "showcases manifest includes --probesh-shot");
        check(sawProbeInterpShot, "showcases manifest includes --probeinterp-shot");
        check(sawProbeDistShot, "showcases manifest includes --probedist-shot");
        check(sawDdgiShot, "showcases manifest includes --ddgi-shot");
        check(sawDdgiMbShot, "showcases manifest includes --ddgimb-shot");
        check(sawMeshletVizShot, "showcases manifest includes --meshlet-viz");
        check(sawClusterCullShot, "showcases manifest includes --cluster-cull-shot");
        check(sawClusterHizShot, "showcases manifest includes --cluster-hiz-shot");
        check(sawClusterLodShot, "showcases manifest includes --cluster-lod-shot");
        check(sawVisbufferShot, "showcases manifest includes --visbuffer-shot");
        check(sawVisresolveShot, "showcases manifest includes --visresolve-shot");
        check(sawSwRasterShot, "showcases manifest includes --swraster-shot");
        check(sawSwRasterGpuShot, "showcases manifest includes --swraster-gpu-shot");
        check(sawSwRasterResolveShot, "showcases manifest includes --swraster-resolve-shot");
        check(sawVsmMarkShot, "showcases manifest includes --vsm-mark-shot");
        check(sawVsmRenderShot, "showcases manifest includes --vsm-render-shot");
        check(sawVsmSampleShot, "showcases manifest includes --vsm-sample-shot");
        check(sawVsmCacheShot, "showcases manifest includes --vsm-cache-shot");
        check(sawVtFeedbackShot, "showcases manifest includes --vt-feedback-shot");
        check(sawVtAllocShot, "showcases manifest includes --vt-alloc-shot");
        check(sawVtPagegenShot, "showcases manifest includes --vt-pagegen-shot");
        check(sawVtSampleShot, "showcases manifest includes --vt-sample-shot");
        check(sawVtCacheShot, "showcases manifest includes --vt-cache-shot");
        check(sawMcClassifyShot, "showcases manifest includes --mc-classify-shot");
        check(sawMcCountShot, "showcases manifest includes --mc-count-shot");
        check(sawMcEmitShot, "showcases manifest includes --mc-emit-shot");
        check(sawMcInterpShot, "showcases manifest includes --mc-interp-shot");
        check(sawMcRenderShot, "showcases manifest includes --mc-render-shot");
        check(sawMcNormalsShot, "showcases manifest includes --mc-normals-shot");
        check(sawFpxShot, "showcases manifest includes --fpx-shot");
        check(sawFluidIntegrateShot, "showcases manifest includes --fluid-integrate-shot");
        check(sawGrainIntegrateShot, "showcases manifest includes --grain-integrate-shot");
        check(sawGrainNeighborsShot, "showcases manifest includes --grain-neighbors-shot");
        check(sawCoupleQueryShot, "showcases manifest includes --couple-query-shot");
        check(sawCoupleBuoyancyShot, "showcases manifest includes --couple-buoyancy-shot");
        check(sawCoupleDisplaceShot, "showcases manifest includes --couple-displace-shot");
        check(sawCoupleStepShot, "showcases manifest includes --couple-step-shot");
        check(sawCoupleLockstepShot, "showcases manifest includes --couple-lockstep-shot");
        check(sawCoupleRenderShot, "showcases manifest includes --couple-render-shot");
        check(sawCgrainQueryShot, "showcases manifest includes --cgrain-query-shot");
        check(sawCgrainSupportShot, "showcases manifest includes --cgrain-support-shot");
        check(sawCgrainDisplaceShot, "showcases manifest includes --cgrain-displace-shot");
        check(sawCgrainStepShot, "showcases manifest includes --cgrain-step-shot");
        check(sawCgrainLockstepShot, "showcases manifest includes --cgrain-lockstep-shot");
        check(sawCgrainRenderShot, "showcases manifest includes --cgrain-render-shot");
        check(sawCgfQueryShot, "showcases manifest includes --cgf-query-shot");
        check(sawCgfBuoyancyShot, "showcases manifest includes --cgf-buoyancy-shot");
        check(sawCgfDisplaceShot, "showcases manifest includes --cgf-displace-shot");
        check(sawCgfStepShot, "showcases manifest includes --cgf-step-shot");
        check(sawCgfRenderShot, "showcases manifest includes --cgf-render-shot");
        check(sawFractCellsShot, "showcases manifest includes --fract-cells-shot");
        check(sawFractFragmentsShot, "showcases manifest includes --fract-fragments-shot");
        check(sawFractBreakShot, "showcases manifest includes --fract-break-shot");
        check(sawFractStepShot, "showcases manifest includes --fract-step-shot");
        check(sawGrainContactShot, "showcases manifest includes --grain-contact-shot");
        check(sawGrainFrictionShot, "showcases manifest includes --grain-friction-shot");
        check(sawGrainLockstepShot, "showcases manifest includes --grain-lockstep-shot");
        check(sawGrainRenderShot, "showcases manifest includes --grain-render-shot");
        check(sawFluidNeighborsShot, "showcases manifest includes --fluid-neighbors-shot");
        check(sawFluidDensityShot, "showcases manifest includes --fluid-density-shot");
        check(sawFluidSolveShot, "showcases manifest includes --fluid-solve-shot");
        check(sawFluidLockstepShot, "showcases manifest includes --fluid-lockstep-shot");
        check(sawFluidRenderShot, "showcases manifest includes --fluid-render-shot");
        check(sawClothIntegrateShot, "showcases manifest includes --cloth-integrate-shot");
        check(sawClothEdgesShot, "showcases manifest includes --cloth-edges-shot");
        check(sawClothSolveShot, "showcases manifest includes --cloth-solve-shot");
        check(sawClothCollideShot, "showcases manifest includes --cloth-collide-shot");
        check(sawClothLockstepShot, "showcases manifest includes --cloth-lockstep-shot");
        check(sawClothRenderShot, "showcases manifest includes --cloth-render-shot");
        check(sawFpxPairsShot, "showcases manifest includes --fpx-pairs-shot");
        check(sawFpxSolveShot, "showcases manifest includes --fpx-solve-shot");
        check(sawFpxOrientShot, "showcases manifest includes --fpx-orient-shot");
        check(sawFpxLockstepShot, "showcases manifest includes --fpx-lockstep-shot");
        check(sawFpxRenderShot, "showcases manifest includes --fpx-render-shot");
        check(sawNavRasterShot, "showcases manifest includes --nav-raster-shot");
        check(sawNavDistanceShot, "showcases manifest includes --nav-distance-shot");
        check(sawNavRegionShot, "showcases manifest includes --nav-region-shot");
        check(sawNavPolymeshShot, "showcases manifest includes --nav-polymesh-shot");
        check(sawNavPathShot, "showcases manifest includes --nav-path-shot");
        check(sawNavRenderShot, "showcases manifest includes --nav-render-shot");
        check(sawFroxelLightsShot, "showcases manifest includes --froxellights-shot");
        check(sawVolShadowsShot, "showcases manifest includes --volshadows-shot");
        check(sawContactShadowShot, "showcases manifest includes --contactshadow-shot");
        check(sawAutoExposureShot, "showcases manifest includes --autoexposure-shot");
        check(sawWaterShot, "showcases manifest includes --water-shot");
        check(sawSsgiDenoiseShot, "showcases manifest includes --ssgi-denoise-shot");
        check(sawSsgiTemporalShot, "showcases manifest includes --ssgi-temporal-shot");
        check(sawNetShot, "showcases manifest includes --net-shot");
        check(sawNetsimShot, "showcases manifest includes --netsim-shot");
        check(sawNetpredictShot, "showcases manifest includes --netpredict-shot");
        check(sawEditorShot, "showcases manifest includes --editor-shot");
        check(sawEditorEditShot, "showcases manifest includes --editor-edit-shot");
        check(sawVfxShot, "showcases manifest includes --vfx-shot");
        check(sawCloudsShot, "showcases manifest includes --clouds-shot");
        check(sawCloudShadowsShot, "showcases manifest includes --cloud-shadows-shot");

        // scene.entities: count == 2 + entity 0's transform values.
        const json_object_s* scene = AsObject(MemberOf(top, "scene"));
        const json_array_s* ents = scene ? AsArray(MemberOf(scene, "entities")) : nullptr;
        check(ents != nullptr && ents->length == 2, "scene has exactly 2 entities");
        if (ents && ents->length == 2) {
            const json_object_s* e0 = AsObject(ents->start->value);
            const json_object_s* comps = e0 ? AsObject(MemberOf(e0, "components")) : nullptr;
            const json_object_s* xform = comps ? AsObject(MemberOf(comps, "transform")) : nullptr;
            const json_array_s* pos = xform ? AsArray(MemberOf(xform, "position")) : nullptr;
            check(pos != nullptr && pos->length == 3, "entity 0 has a 3-vec position");
            if (pos && pos->length == 3) {
                const json_array_element_s* px = pos->start;
                const json_array_element_s* py = px->next;
                const json_array_element_s* pz = py->next;
                check(approx(AsNumber(px->value), 1) && approx(AsNumber(py->value), 2) &&
                      approx(AsNumber(pz->value), 3), "entity 0 position == [1,2,3]");
            }
            check(AsString(MemberOf(comps, "mesh")) == "cube", "entity 0 mesh name == cube");
            const json_object_s* mat = comps ? AsObject(MemberOf(comps, "material")) : nullptr;
            check(mat && approx(AsNumber(MemberOf(mat, "metallic")), 0.0), "entity 0 metallic == 0");
            check(mat && AsString(MemberOf(mat, "baseColor")) == "checker", "entity 0 baseColor == checker");
        }

        // camera fields.
        const json_object_s* cam = AsObject(MemberOf(top, "camera"));
        check(cam != nullptr, "has camera object");
        if (cam) {
            check(approx(AsNumber(MemberOf(cam, "fovDeg")), 60.0), "camera.fovDeg == 60");
            check(approx(AsNumber(MemberOf(cam, "yaw")), 0.25), "camera.yaw == 0.25");
            const json_array_s* cpos = AsArray(MemberOf(cam, "position"));
            check(cpos != nullptr && cpos->length == 3, "camera.position is a 3-vec");
        }

        // lights: directional present + 2 points.
        const json_object_s* lights = AsObject(MemberOf(top, "lights"));
        check(lights != nullptr, "has lights object");
        if (lights) {
            const json_object_s* dir = AsObject(MemberOf(lights, "directional"));
            check(dir != nullptr, "lights.directional present");
            const json_array_s* pts = AsArray(MemberOf(lights, "points"));
            check(pts != nullptr && pts->length == 2, "lights.points has 2 entries");
            if (pts && pts->length == 2) {
                const json_object_s* p0 = AsObject(pts->start->value);
                check(p0 && approx(AsNumber(MemberOf(p0, "radius")), 3.2), "point 0 radius == 3.2");
            }
        }

        // stats.
        const json_object_s* stats = AsObject(MemberOf(top, "stats"));
        check(stats != nullptr, "has stats object");
        if (stats) {
            check((int)AsNumber(MemberOf(stats, "entityCount")) == 2, "stats.entityCount == 2");
            check((int)AsNumber(MemberOf(stats, "pointLightCount")) == 2, "stats.pointLightCount == 2");
        }
    }

    if (root) std::free(root);

    if (g_fail == 0) { std::printf("introspect_test OK\n"); return 0; }
    std::printf("introspect_test: %d failures\n", g_fail);
    return 1;
}
