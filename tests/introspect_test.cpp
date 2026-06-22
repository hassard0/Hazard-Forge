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
            // Slice FR5: the deterministic-fract-lockstep capability is advertised.
            bool sawFractLockstepFeature = false;
            // Slice FR6: the deterministic-fract-render capability is advertised.
            bool sawFractRenderFeature = false;
            // Slice JT1: the deterministic-joint-ball capability is advertised.
            bool sawJointBallFeature = false;
            // Slice JT2: the deterministic-joint-hinge capability is advertised.
            bool sawJointHingeFeature = false;
            // Slice JT3: the deterministic-joint-step capability is advertised.
            bool sawJointStepFeature = false;
            // Slice JT4: the deterministic-joint-ragdoll capability is advertised.
            bool sawJointRagdollFeature = false;
            // Slice JT5: the deterministic-joint-lockstep capability is advertised.
            bool sawJointLockstepFeature = false;
            // Slice JT6: the deterministic-joint-render capability is advertised.
            bool sawJointRenderFeature = false;
            // Slice VH1: the deterministic-vehicle-spring capability is advertised.
            bool sawVehicleSpringFeature = false;
            // Slice VH2: the deterministic-vehicle-rig capability is advertised.
            bool sawVehicleRigFeature = false;
            // Slice VH3: the deterministic-vehicle-drive capability is advertised.
            bool sawVehicleDriveFeature = false;
            // Slice VH4: the deterministic-vehicle-traction capability is advertised.
            bool sawVehicleTractionFeature = false;
            // Slice VH5: the deterministic-vehicle-lockstep capability is advertised.
            bool sawVehicleLockstepFeature = false;
            // Slice VH6: the deterministic-vehicle-render capability is advertised.
            bool sawVehicleRenderFeature = false;
            // Slice AC1: the deterministic-active-drive capability is advertised.
            bool sawActiveDriveFeature = false;
            // Slice AC2: the deterministic-active-blend capability is advertised.
            bool sawActiveBlendFeature = false;
            // Slice AC3: the deterministic-active-step capability is advertised.
            bool sawActiveStepFeature = false;
            // Slice AC4: the deterministic-active-recover capability is advertised.
            bool sawActiveRecoverFeature = false;
            // Slice AC5: the deterministic-active-lockstep capability is advertised.
            bool sawActiveLockstepFeature = false;
            // Slice AC6: the deterministic-active-render capability is advertised.
            bool sawActiveRenderFeature = false;
            // Slice BD1: the deterministic-boids-steer capability is advertised.
            bool sawBoidsSteerFeature = false;
            // Slice BD2: the deterministic-boids-neighbors capability is advertised.
            bool sawBoidsNeighborsFeature = false;
            // Slice BD3: the deterministic-boids-flock capability is advertised.
            bool sawBoidsFlockFeature = false;
            // Slice BD4: the deterministic-boids-path capability is advertised.
            bool sawBoidsPathFeature = false;
            // Slice BD5: the deterministic-boids-lockstep capability is advertised.
            bool sawBoidsLockstepFeature = false;
            // Slice BD6: the deterministic-boids-render capability is advertised.
            bool sawBoidsRenderFeature = false;
            // Slice CX1: the deterministic-convex-sat capability is advertised.
            bool sawConvexSatFeature = false;
            // Slice CX2: the deterministic-convex-manifold capability is advertised.
            bool sawConvexManifoldFeature = false;
            // Slice CX3: the deterministic-convex-impulse capability is advertised.
            bool sawConvexImpulseFeature = false;
            // Slice CX4: the deterministic-convex-step capability is advertised.
            bool sawConvexStepFeature = false;
            // Slice CX5: the deterministic-convex-lockstep capability is advertised.
            bool sawConvexLockstepFeature = false;
            // Slice CX6: the deterministic-convex-render capability is advertised.
            bool sawConvexRenderFeature = false;
            // Slice FC1: the deterministic-friction-basis capability is advertised.
            bool sawFrictionBasisFeature = false;
            // Slice FC2: the deterministic-friction-points capability is advertised.
            bool sawFrictionPointsFeature = false;
            // Slice FC3: the deterministic-friction-solve capability is advertised.
            bool sawFrictionSolveFeature = false;
            // Slice FC4: the deterministic-friction-step capability is advertised.
            bool sawFrictionStepFeature = false;
            // Slice FC5: the deterministic-friction-lockstep capability is advertised.
            bool sawFrictionLockstepFeature = false;
            // Slice FC6: the deterministic-friction-render capability is advertised.
            bool sawFrictionRenderFeature = false;
            // Slice PS1: the deterministic-persist-key capability is advertised.
            bool sawPersistKeyFeature = false;
            // Slice PS2: the deterministic-persist-cache capability is advertised.
            bool sawPersistCacheFeature = false;
            // Slice PS3: the deterministic-persist-warm capability is advertised.
            bool sawPersistWarmFeature = false;
            // Slice PS4: the deterministic-persist-sleep capability is advertised.
            bool sawPersistSleepFeature = false;
            // Slice PS5: the deterministic-persist-lockstep capability is advertised.
            bool sawPersistLockstepFeature = false;
            // Slice PS6: the deterministic-persist-render capability is advertised.
            bool sawPersistRenderFeature = false;
            // Slice WH1: the warmhull-contact-key capability is advertised.
            bool sawWarmhullKeyFeature = false;
            // Slice WH2: the warmhull-cache capability is advertised.
            bool sawWarmhullCacheFeature = false;
            // Slice WH3: the warmhull-solve capability is advertised.
            bool sawWarmhullSolveFeature = false;
            // Slice WH4: the warmhull-sleep capability is advertised.
            bool sawWarmhullSleepFeature = false;
            // Slice WH5: the warmhull-lockstep capability is advertised.
            bool sawWarmhullLockstepFeature = false;
            // Slice WH6: the warmhull-render capability is advertised.
            bool sawWarmhullRenderFeature = false;
            // Slice GJ1: the deterministic-hull-support capability is advertised.
            bool sawHullSupportFeature = false;
            // Slice GJ2: the deterministic-gjk-distance capability is advertised.
            bool sawGjkDistanceFeature = false;
            // Slice GJ3: the deterministic-gjk-epa capability is advertised.
            bool sawGjkEpaFeature = false;
            // Slice CD1: the deterministic-ccd-toi capability is advertised.
            bool sawCcdToiFeature = false;
            // Slice CD2: the deterministic-ccd-swept capability is advertised.
            bool sawCcdSweptFeature = false;
            // Slice CD3: the deterministic-ccd-step capability is advertised.
            bool sawCcdStepFeature = false;
            // Slice CD4: the deterministic-ccd-bullet capability is advertised.
            bool sawCcdBulletFeature = false;
            // Slice CD5: the deterministic-ccd-lockstep capability is advertised.
            bool sawCcdLockstepFeature = false;
            // Slice CD6: the deterministic-ccd-render capability is advertised.
            bool sawCcdRenderFeature = false;
            // Slice MF1: the manifold-hull-faces capability is advertised.
            bool sawManifoldHullFacesFeature = false;
            // Slice MF2: the manifold-face-clip capability is advertised.
            bool sawManifoldFaceClipFeature = false;
            // Slice MF3: the manifold-gpu capability is advertised.
            bool sawManifoldGpuFeature = false;
            // Slice MF4: the manifold-hardened-step capability is advertised.
            bool sawManifoldHardenedStepFeature = false;
            // Slice MF5: the manifold-lockstep capability is advertised.
            bool sawManifoldLockstepFeature = false;
            // Slice MF6: the manifold-render capability is advertised.
            bool sawManifoldRenderFeature = false;
            // Slice VD1: the verdict-world capability is advertised.
            bool sawVerdictWorldFeature = false;
            // Slice VD2: the verdict-systems capability is advertised.
            bool sawVerdictSystemsFeature = false;
            // Slice VD3: the verdict-world-step capability is advertised.
            bool sawVerdictWorldStepFeature = false;
            // Slice VD4: the verdict-snapshot capability is advertised.
            bool sawVerdictSnapshotFeature = false;
            // Slice VD5: the verdict-lockstep capability is advertised.
            bool sawVerdictLockstepFeature = false;
            // Slice VD6: the verdict-render capability is advertised.
            bool sawVerdictRenderFeature = false;
            // Slice RT1: the rt1-trace capability is advertised (FLAGSHIP #28 beachhead).
            bool sawRt1TraceFeature = false;
            // Slice RT2: the rt2-query capability is advertised (HW inline ray query).
            bool sawRt2QueryFeature = false;
            // Slice RT2b: the rt2-query-hw capability is advertised (Metal HW ray query).
            bool sawRt2QueryHwFeature = false;
            // Slice RT3: the rt3-shadow capability is advertised (deterministic RT hard shadows).
            bool sawRt3ShadowFeature = false;
            // Slice RT4: the rt4-reflect capability is advertised (deterministic RT mirror reflections).
            bool sawRt4ReflectFeature = false;
            // Slice RT5: the rt5-simrender capability is advertised (determinism-envelope + lockstep tie-in).
            bool sawRt5SimrenderFeature = false;
            // Slice RT6: the rt6-hero capability is advertised (the lit hero capstone — the FLAGSHIP #28 money-shot).
            bool sawRt6HeroFeature = false;
            // Slice GI1: the gi1-probe capability is advertised (the integer RT probe trace — the FLAGSHIP #29 beachhead).
            bool sawGi1ProbeFeature = false;
            // Slice GI2: the gi2-shencode capability is advertised (the integer SH encode — the FLAGSHIP #29 crux).
            bool sawGi2ShencodeFeature = false;
            // Slice GI3: the gi3-interp capability is advertised (the integer trilinear SH interpolation — FLAGSHIP #29).
            bool sawGi3InterpFeature = false;
            // Slice GI4: the gi4-bounce capability is advertised (the integer multi-bounce feedback — FLAGSHIP #29).
            bool sawGi4BounceFeature = false;
            // Slice GI5: the gi5-occlusion capability is advertised (the integer Chebyshev occlusion leak-fix — FLAGSHIP #29).
            bool sawGi5OcclusionFeature = false;
            // Slice GI6: the gi6-hero capability is advertised (the lit GI hero capstone — the FLAGSHIP #29 money-shot).
            bool sawGi6HeroFeature = false;
            // Slice HF1: the hf1-points capability is advertised (the hull-friction tagged-manifold beachhead, FLAGSHIP #30).
            bool sawHf1PointsFeature = false;
            // Slice HF2: the hf2-warm capability is advertised (the hull-friction warm cone solver, FLAGSHIP #30).
            bool sawHf2WarmFeature = false;
            // Slice HF3: the hf3-step capability is advertised (the friction-locked hull world step, FLAGSHIP #30).
            bool sawHf3StepFeature = false;
            // Slice HF4: the hf4-joint capability is advertised (hull joints composed with friction, FLAGSHIP #30).
            bool sawHf4JointFeature = false;
            // Slice HF5: the hf5-net capability is advertised (friction+joint hull lockstep+rollback, FLAGSHIP #30).
            bool sawHf5NetFeature = false;
            // Slice HF6: the hf6-hull capability is advertised (friction+joint hull LIT 3D capstone, FLAGSHIP #30).
            bool sawHf6HullFeature = false;
            // Slice IK1: the ik1-angle capability is advertised (the fixed-point angle LUT, FLAGSHIP #32).
            bool sawIk1AngleFeature = false;
            // Slice IK2: the ik2-twobone capability is advertised (the two-bone law-of-cosines limb solve).
            bool sawIk2TwoboneFeature = false;
            // Slice IK3: the ik3-fabrik capability is advertised (the FABRIK n-bone chain + look-at solve).
            bool sawIk3FabrikFeature = false;
            // Slice IK4: the ik4-rig capability is advertised (the FK-pose -> IK-corrected palette bridge).
            bool sawIk4RigFeature = false;
            // Slice GJ4: the deterministic-hull-step capability is advertised.
            bool sawHullStepFeature = false;
            // Slice GJ5: the deterministic-hull-lockstep capability is advertised.
            bool sawHullLockstepFeature = false;
            // Slice GJ6: the deterministic-hull-render capability is advertised.
            bool sawHullRenderFeature = false;
            // Slice BP1: the deterministic-broadphase-grid capability is advertised.
            bool sawBroadphaseGridFeature = false;
            // Slice BP2: the deterministic-broadphase-pairs capability is advertised.
            bool sawBroadphasePairsFeature = false;
            // Slice BP3: the deterministic-broadphase-convex-step capability is advertised.
            bool sawBroadphaseConvexStepFeature = false;
            // Slice BP4: the deterministic-broadphase-hull-step capability is advertised.
            bool sawBroadphaseHullStepFeature = false;
            // Slice BP5: the deterministic-broadphase-lockstep capability is advertised.
            bool sawBroadphaseLockstepFeature = false;
            // Slice BP6: the deterministic-broadphase-render capability is advertised.
            bool sawBroadphaseRenderFeature = false;
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
                    if (AsString(el->value) == "deterministic-fract-lockstep") sawFractLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-fract-render") sawFractRenderFeature = true;
                    if (AsString(el->value) == "deterministic-joint-ball") sawJointBallFeature = true;
                    if (AsString(el->value) == "deterministic-joint-hinge") sawJointHingeFeature = true;
                    if (AsString(el->value) == "deterministic-joint-step") sawJointStepFeature = true;
                    if (AsString(el->value) == "deterministic-joint-ragdoll") sawJointRagdollFeature = true;
                    if (AsString(el->value) == "deterministic-joint-lockstep") sawJointLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-joint-render") sawJointRenderFeature = true;
                    if (AsString(el->value) == "deterministic-vehicle-spring") sawVehicleSpringFeature = true;
                    if (AsString(el->value) == "deterministic-vehicle-rig") sawVehicleRigFeature = true;
                    if (AsString(el->value) == "deterministic-vehicle-drive") sawVehicleDriveFeature = true;
                    if (AsString(el->value) == "deterministic-vehicle-traction") sawVehicleTractionFeature = true;
                    if (AsString(el->value) == "deterministic-vehicle-lockstep") sawVehicleLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-vehicle-render") sawVehicleRenderFeature = true;
                    if (AsString(el->value) == "deterministic-active-drive") sawActiveDriveFeature = true;
                    if (AsString(el->value) == "deterministic-active-blend") sawActiveBlendFeature = true;
                    if (AsString(el->value) == "deterministic-active-step") sawActiveStepFeature = true;
                    if (AsString(el->value) == "deterministic-active-recover") sawActiveRecoverFeature = true;
                    if (AsString(el->value) == "deterministic-active-lockstep") sawActiveLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-active-render") sawActiveRenderFeature = true;
                    if (AsString(el->value) == "deterministic-boids-steer") sawBoidsSteerFeature = true;
                    if (AsString(el->value) == "deterministic-boids-neighbors") sawBoidsNeighborsFeature = true;
                    if (AsString(el->value) == "deterministic-boids-flock") sawBoidsFlockFeature = true;
                    if (AsString(el->value) == "deterministic-boids-path") sawBoidsPathFeature = true;
                    if (AsString(el->value) == "deterministic-boids-lockstep") sawBoidsLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-boids-render") sawBoidsRenderFeature = true;
                    if (AsString(el->value) == "deterministic-convex-sat") sawConvexSatFeature = true;
                    if (AsString(el->value) == "deterministic-convex-manifold") sawConvexManifoldFeature = true;
                    if (AsString(el->value) == "deterministic-convex-impulse") sawConvexImpulseFeature = true;
                    if (AsString(el->value) == "deterministic-convex-step") sawConvexStepFeature = true;
                    if (AsString(el->value) == "deterministic-convex-lockstep") sawConvexLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-convex-render") sawConvexRenderFeature = true;
                    if (AsString(el->value) == "deterministic-friction-basis") sawFrictionBasisFeature = true;
                    if (AsString(el->value) == "deterministic-friction-points") sawFrictionPointsFeature = true;
                    if (AsString(el->value) == "deterministic-friction-solve") sawFrictionSolveFeature = true;
                    if (AsString(el->value) == "deterministic-friction-step") sawFrictionStepFeature = true;
                    if (AsString(el->value) == "deterministic-friction-lockstep") sawFrictionLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-friction-render") sawFrictionRenderFeature = true;
                    if (AsString(el->value) == "deterministic-persist-key") sawPersistKeyFeature = true;
                    if (AsString(el->value) == "deterministic-persist-cache") sawPersistCacheFeature = true;
                    if (AsString(el->value) == "deterministic-persist-warm") sawPersistWarmFeature = true;
                    if (AsString(el->value) == "deterministic-persist-sleep") sawPersistSleepFeature = true;
                    if (AsString(el->value) == "deterministic-persist-lockstep") sawPersistLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-persist-render") sawPersistRenderFeature = true;
                    if (AsString(el->value) == "warmhull-contact-key") sawWarmhullKeyFeature = true;
                    if (AsString(el->value) == "warmhull-cache") sawWarmhullCacheFeature = true;
                    if (AsString(el->value) == "warmhull-solve") sawWarmhullSolveFeature = true;
                    if (AsString(el->value) == "warmhull-sleep") sawWarmhullSleepFeature = true;
                    if (AsString(el->value) == "warmhull-lockstep") sawWarmhullLockstepFeature = true;
                    if (AsString(el->value) == "warmhull-render") sawWarmhullRenderFeature = true;
                    if (AsString(el->value) == "deterministic-hull-support") sawHullSupportFeature = true;
                    if (AsString(el->value) == "deterministic-gjk-distance") sawGjkDistanceFeature = true;
                    if (AsString(el->value) == "deterministic-gjk-epa") sawGjkEpaFeature = true;
                    if (AsString(el->value) == "deterministic-ccd-toi") sawCcdToiFeature = true;
                    if (AsString(el->value) == "deterministic-ccd-swept") sawCcdSweptFeature = true;
                    if (AsString(el->value) == "deterministic-ccd-step") sawCcdStepFeature = true;
                    if (AsString(el->value) == "deterministic-ccd-bullet") sawCcdBulletFeature = true;
                    if (AsString(el->value) == "deterministic-ccd-lockstep") sawCcdLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-ccd-render") sawCcdRenderFeature = true;
                    if (AsString(el->value) == "manifold-hull-faces") sawManifoldHullFacesFeature = true;
                    if (AsString(el->value) == "manifold-face-clip") sawManifoldFaceClipFeature = true;
                    if (AsString(el->value) == "manifold-gpu") sawManifoldGpuFeature = true;
                    if (AsString(el->value) == "manifold-hardened-step") sawManifoldHardenedStepFeature = true;
                    if (AsString(el->value) == "manifold-lockstep") sawManifoldLockstepFeature = true;
                    if (AsString(el->value) == "manifold-render") sawManifoldRenderFeature = true;
                    if (AsString(el->value) == "verdict-world") sawVerdictWorldFeature = true;
                    if (AsString(el->value) == "verdict-systems") sawVerdictSystemsFeature = true;
                    if (AsString(el->value) == "verdict-world-step") sawVerdictWorldStepFeature = true;
                    if (AsString(el->value) == "verdict-snapshot") sawVerdictSnapshotFeature = true;
                    if (AsString(el->value) == "verdict-lockstep") sawVerdictLockstepFeature = true;
                    if (AsString(el->value) == "verdict-render") sawVerdictRenderFeature = true;
                    if (AsString(el->value) == "rt1-trace") sawRt1TraceFeature = true;
                    if (AsString(el->value) == "rt2-query") sawRt2QueryFeature = true;
                    if (AsString(el->value) == "rt2-query-hw") sawRt2QueryHwFeature = true;
                    if (AsString(el->value) == "rt3-shadow") sawRt3ShadowFeature = true;
                    if (AsString(el->value) == "rt4-reflect") sawRt4ReflectFeature = true;
                    if (AsString(el->value) == "rt5-simrender") sawRt5SimrenderFeature = true;
                    if (AsString(el->value) == "rt6-hero") sawRt6HeroFeature = true;
                    if (AsString(el->value) == "gi1-probe") sawGi1ProbeFeature = true;
                    if (AsString(el->value) == "gi2-shencode") sawGi2ShencodeFeature = true;
                    if (AsString(el->value) == "gi3-interp") sawGi3InterpFeature = true;
                    if (AsString(el->value) == "gi4-bounce") sawGi4BounceFeature = true;
                    if (AsString(el->value) == "gi5-occlusion") sawGi5OcclusionFeature = true;
                    if (AsString(el->value) == "gi6-hero") sawGi6HeroFeature = true;
                    if (AsString(el->value) == "hf1-points") sawHf1PointsFeature = true;
                    if (AsString(el->value) == "hf2-warm") sawHf2WarmFeature = true;
                    if (AsString(el->value) == "hf3-step") sawHf3StepFeature = true;
                    if (AsString(el->value) == "hf4-joint") sawHf4JointFeature = true;
                    if (AsString(el->value) == "hf5-net") sawHf5NetFeature = true;
                    if (AsString(el->value) == "hf6-hull") sawHf6HullFeature = true;
                    if (AsString(el->value) == "ik1-angle") sawIk1AngleFeature = true;
                    if (AsString(el->value) == "ik2-twobone") sawIk2TwoboneFeature = true;
                    if (AsString(el->value) == "ik3-fabrik") sawIk3FabrikFeature = true;
                    if (AsString(el->value) == "ik4-rig") sawIk4RigFeature = true;
                    if (AsString(el->value) == "deterministic-hull-step") sawHullStepFeature = true;
                    if (AsString(el->value) == "deterministic-hull-lockstep") sawHullLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-hull-render") sawHullRenderFeature = true;
                    if (AsString(el->value) == "deterministic-broadphase-grid") sawBroadphaseGridFeature = true;
                    if (AsString(el->value) == "deterministic-broadphase-pairs") sawBroadphasePairsFeature = true;
                    if (AsString(el->value) == "deterministic-broadphase-convex-step") sawBroadphaseConvexStepFeature = true;
                    if (AsString(el->value) == "deterministic-broadphase-hull-step") sawBroadphaseHullStepFeature = true;
                    if (AsString(el->value) == "deterministic-broadphase-lockstep") sawBroadphaseLockstepFeature = true;
                    if (AsString(el->value) == "deterministic-broadphase-render") sawBroadphaseRenderFeature = true;
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
        check(sawFractLockstepFeature, "engine.features includes deterministic-fract-lockstep");
        check(sawFractRenderFeature, "engine.features includes deterministic-fract-render");
        check(sawJointBallFeature, "engine.features includes deterministic-joint-ball");
        check(sawJointHingeFeature, "engine.features includes deterministic-joint-hinge");
        check(sawJointStepFeature, "engine.features includes deterministic-joint-step");
        check(sawJointRagdollFeature, "engine.features includes deterministic-joint-ragdoll");
        check(sawJointLockstepFeature, "engine.features includes deterministic-joint-lockstep");
        check(sawJointRenderFeature, "engine.features includes deterministic-joint-render");
        check(sawVehicleSpringFeature, "engine.features includes deterministic-vehicle-spring");
        check(sawVehicleRigFeature, "engine.features includes deterministic-vehicle-rig");
        check(sawVehicleDriveFeature, "engine.features includes deterministic-vehicle-drive");
        check(sawVehicleTractionFeature, "engine.features includes deterministic-vehicle-traction");
        check(sawVehicleLockstepFeature, "engine.features includes deterministic-vehicle-lockstep");
        check(sawVehicleRenderFeature, "engine.features includes deterministic-vehicle-render");
        check(sawActiveDriveFeature, "engine.features includes deterministic-active-drive");
        check(sawActiveBlendFeature, "engine.features includes deterministic-active-blend");
        check(sawActiveStepFeature, "engine.features includes deterministic-active-step");
        check(sawActiveRecoverFeature, "engine.features includes deterministic-active-recover");
        check(sawActiveLockstepFeature, "engine.features includes deterministic-active-lockstep");
        check(sawActiveRenderFeature, "engine.features includes deterministic-active-render");
        check(sawBoidsSteerFeature, "engine.features includes deterministic-boids-steer");
        check(sawBoidsNeighborsFeature, "engine.features includes deterministic-boids-neighbors");
        check(sawBoidsFlockFeature, "engine.features includes deterministic-boids-flock");
        check(sawBoidsPathFeature, "engine.features includes deterministic-boids-path");
        check(sawBoidsLockstepFeature, "engine.features includes deterministic-boids-lockstep");
        check(sawBoidsRenderFeature, "engine.features includes deterministic-boids-render");
        check(sawConvexSatFeature, "engine.features includes deterministic-convex-sat");
        check(sawConvexManifoldFeature, "engine.features includes deterministic-convex-manifold");
        check(sawConvexImpulseFeature, "engine.features includes deterministic-convex-impulse");
        check(sawConvexStepFeature, "engine.features includes deterministic-convex-step");
        check(sawConvexLockstepFeature, "engine.features includes deterministic-convex-lockstep");
        check(sawConvexRenderFeature, "engine.features includes deterministic-convex-render");
        check(sawFrictionBasisFeature, "engine.features includes deterministic-friction-basis");
        check(sawFrictionPointsFeature, "engine.features includes deterministic-friction-points");
        check(sawFrictionSolveFeature, "engine.features includes deterministic-friction-solve");
        check(sawFrictionStepFeature, "engine.features includes deterministic-friction-step");
        check(sawFrictionLockstepFeature, "engine.features includes deterministic-friction-lockstep");
        check(sawFrictionRenderFeature, "engine.features includes deterministic-friction-render");
        check(sawPersistKeyFeature, "engine.features includes deterministic-persist-key");
        check(sawPersistCacheFeature, "engine.features includes deterministic-persist-cache");
        check(sawPersistWarmFeature, "engine.features includes deterministic-persist-warm");
        check(sawPersistSleepFeature, "engine.features includes deterministic-persist-sleep");
        check(sawPersistLockstepFeature, "engine.features includes deterministic-persist-lockstep");
        check(sawPersistRenderFeature, "engine.features includes deterministic-persist-render");
        check(sawWarmhullKeyFeature, "engine.features includes warmhull-contact-key");
        check(sawWarmhullCacheFeature, "engine.features includes warmhull-cache");
        check(sawWarmhullSolveFeature, "engine.features includes warmhull-solve");
        check(sawWarmhullSleepFeature, "engine.features includes warmhull-sleep");
        check(sawWarmhullLockstepFeature, "engine.features includes warmhull-lockstep");
        check(sawWarmhullRenderFeature, "engine.features includes warmhull-render");
            check(sawHullSupportFeature, "engine.features includes deterministic-hull-support");
            check(sawGjkDistanceFeature, "engine.features includes deterministic-gjk-distance");
            check(sawGjkEpaFeature, "engine.features includes deterministic-gjk-epa");
            check(sawCcdToiFeature, "engine.features includes deterministic-ccd-toi");
            check(sawCcdSweptFeature, "engine.features includes deterministic-ccd-swept");
            check(sawCcdStepFeature, "engine.features includes deterministic-ccd-step");
            check(sawCcdBulletFeature, "engine.features includes deterministic-ccd-bullet");
            check(sawCcdLockstepFeature, "engine.features includes deterministic-ccd-lockstep");
            check(sawCcdRenderFeature, "engine.features includes deterministic-ccd-render");
            check(sawManifoldHullFacesFeature, "engine.features includes manifold-hull-faces");
            check(sawManifoldFaceClipFeature, "engine.features includes manifold-face-clip");
            check(sawManifoldGpuFeature, "engine.features includes manifold-gpu");
            check(sawManifoldHardenedStepFeature, "engine.features includes manifold-hardened-step");
            check(sawManifoldLockstepFeature, "engine.features includes manifold-lockstep");
            check(sawManifoldRenderFeature, "engine.features includes manifold-render");
            check(sawVerdictWorldFeature, "engine.features includes verdict-world");
            check(sawVerdictSystemsFeature, "engine.features includes verdict-systems");
            check(sawVerdictWorldStepFeature, "engine.features includes verdict-world-step");
            check(sawVerdictSnapshotFeature, "engine.features includes verdict-snapshot");
            check(sawVerdictLockstepFeature, "engine.features includes verdict-lockstep");
            check(sawVerdictRenderFeature, "engine.features includes verdict-render");
            check(sawRt1TraceFeature, "engine.features includes rt1-trace");
            check(sawRt2QueryFeature, "engine.features includes rt2-query");
            check(sawRt2QueryHwFeature, "engine.features includes rt2-query-hw");
            check(sawRt3ShadowFeature, "engine.features includes rt3-shadow");
            check(sawRt4ReflectFeature, "engine.features includes rt4-reflect");
            check(sawRt5SimrenderFeature, "engine.features includes rt5-simrender");
            check(sawRt6HeroFeature, "engine.features includes rt6-hero");
            check(sawGi1ProbeFeature, "engine.features includes gi1-probe");
            check(sawGi2ShencodeFeature, "engine.features includes gi2-shencode");
            check(sawGi3InterpFeature, "engine.features includes gi3-interp");
            check(sawGi4BounceFeature, "engine.features includes gi4-bounce");
            check(sawGi5OcclusionFeature, "engine.features includes gi5-occlusion");
            check(sawGi6HeroFeature, "engine.features includes gi6-hero");
            check(sawHf1PointsFeature, "engine.features includes hf1-points");
            check(sawHf2WarmFeature, "engine.features includes hf2-warm");
            check(sawHf3StepFeature, "engine.features includes hf3-step");
            check(sawHf4JointFeature, "engine.features includes hf4-joint");
            check(sawHf5NetFeature, "engine.features includes hf5-net");
            check(sawHf6HullFeature, "engine.features includes hf6-hull");
            check(sawIk1AngleFeature, "engine.features includes ik1-angle");
            check(sawIk2TwoboneFeature, "engine.features includes ik2-twobone");
            check(sawIk3FabrikFeature, "engine.features includes ik3-fabrik");
            check(sawIk4RigFeature, "engine.features includes ik4-rig");
            check(sawHullStepFeature, "engine.features includes deterministic-hull-step");
            check(sawHullLockstepFeature, "engine.features includes deterministic-hull-lockstep");
            check(sawHullRenderFeature, "engine.features includes deterministic-hull-render");
            check(sawBroadphaseGridFeature, "engine.features includes deterministic-broadphase-grid");
            check(sawBroadphasePairsFeature, "engine.features includes deterministic-broadphase-pairs");
            check(sawBroadphaseConvexStepFeature, "engine.features includes deterministic-broadphase-convex-step");
            check(sawBroadphaseHullStepFeature, "engine.features includes deterministic-broadphase-hull-step");
            check(sawBroadphaseLockstepFeature, "engine.features includes deterministic-broadphase-lockstep");
            check(sawBroadphaseRenderFeature, "engine.features includes deterministic-broadphase-render");
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

        // --- Slice DX1: the versioned Agent-SDK contract (the additive trailing "agentApi" key). ---
        const json_object_s* agentApi = AsObject(MemberOf(top, "agentApi"));
        check(agentApi != nullptr, "has agentApi object");
        if (agentApi) {
            // schemaVersion == 1 (SHAPE-ONLY version).
            check((int)AsNumber(MemberOf(agentApi, "schemaVersion")) == 1, "agentApi.schemaVersion == 1");

            // capabilities == exactly the 5, in order.
            const json_array_s* caps = AsArray(MemberOf(agentApi, "capabilities"));
            check(caps != nullptr && caps->length == 5, "agentApi.capabilities has 5 entries");
            if (caps && caps->length == 5) {
                const char* expected[5] = {"observe", "mutate", "author", "replay", "hot-reload"};
                const json_array_element_s* el = caps->start;
                bool order = true;
                for (int ci = 0; ci < 5 && el; ++ci, el = el->next)
                    if (AsString(el->value) != expected[ci]) order = false;
                check(order, "agentApi.capabilities is exactly [observe,mutate,author,replay,hot-reload] in order");
            }

            // commands: every entry has a non-empty op, a capability in the set, and an args array.
            const json_array_s* acmds = AsArray(MemberOf(agentApi, "commands"));
            check(acmds != nullptr && acmds->length > 0, "agentApi.commands is a non-empty array");
            if (acmds) {
                bool allOk = true;
                for (const json_array_element_s* e = acmds->start; e; e = e->next) {
                    const json_object_s* c = AsObject(e->value);
                    if (!c) { allOk = false; continue; }
                    std::string op = AsString(MemberOf(c, "op"));
                    std::string cap = AsString(MemberOf(c, "capability"));
                    const json_array_s* args = AsArray(MemberOf(c, "args"));
                    bool capInSet = (cap == "observe" || cap == "mutate" || cap == "author" ||
                                     cap == "replay" || cap == "hot-reload");
                    if (op.empty() || !capInSet || args == nullptr) allOk = false;
                    // Each arg (when present) is an object with a name + type.
                    if (args)
                        for (const json_array_element_s* ae = args->start; ae; ae = ae->next) {
                            const json_object_s* a = AsObject(ae->value);
                            if (!a) { allOk = false; continue; }
                            if (AsString(MemberOf(a, "name")).empty() ||
                                AsString(MemberOf(a, "type")).empty())
                                allOk = false;
                        }
                }
                check(allOk, "every agentApi.command has a non-empty op, a valid capability, and an args array");
            }

            // contentHash present + non-empty.
            std::string contentHash = AsString(MemberOf(agentApi, "contentHash"));
            check(!contentHash.empty(), "agentApi.contentHash present + non-empty");
        }

        // --- DescribeAgentApi() standalone == the folded "agentApi" sub-object, byte-for-byte. ------
        // The standalone document is the SAME object text (base=1) plus a trailing newline. Locate the
        // folded value in the full document and compare the object substrings exactly.
        {
            std::string standalone = editor::DescribeAgentApi();
            // Standalone is byte-stable across two calls (contentHash deterministic).
            std::string standalone2 = editor::DescribeAgentApi();
            check(standalone == standalone2, "DescribeAgentApi() is deterministic (byte-identical across runs)");

            // Strip the single trailing newline from the standalone to get the bare object text.
            std::string bareStandalone = standalone;
            if (!bareStandalone.empty() && bareStandalone.back() == '\n') bareStandalone.pop_back();

            // Extract the folded value: from the '{' after "\"agentApi\": " to the matching close brace
            // at the document's depth-1 indent ("  }"). Since the block is the LAST key, the object runs
            // to just before the final "}\n".
            const std::string keyMarker = "\"agentApi\": {";
            size_t kp = json.find(keyMarker);
            check(kp != std::string::npos, "full document contains the agentApi key");
            if (kp != std::string::npos) {
                size_t objStart = kp + std::string("\"agentApi\": ").size();  // points at '{'
                // The folded object ends at "\n  }" (the depth-1 close), which is the LAST "  }" before
                // the document's final "}\n". Find the last occurrence of "\n  }" in the document.
                size_t objClose = json.rfind("\n  }");
                check(objClose != std::string::npos, "found agentApi object close brace");
                if (objClose != std::string::npos) {
                    // The folded object text spans [objStart, objClose + len("\n  }")).
                    size_t objEnd = objClose + std::string("\n  }").size();
                    std::string folded = json.substr(objStart, objEnd - objStart);
                    check(folded == bareStandalone,
                          "DescribeAgentApi() == the folded agentApi sub-object (byte-for-byte)");
                }
            }
        }

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
        // Slice BP1: the --broad-cell-shot showcase flag is listed in the showcase manifest.
        bool sawBroadCellShot = false;
        // Slice BP2: the --broad-pair-shot showcase flag is listed in the showcase manifest.
        bool sawBroadPairShot = false;
        // Slice BP4: the --broad-hull-shot showcase flag is listed in the showcase manifest.
        bool sawBroadHullShot = false;
        // Slice BP5: the --broad-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawBroadLockstepShot = false;
        // Slice BP6: the --broad-render-shot showcase flag is listed in the showcase manifest.
        bool sawBroadRenderShot = false;
        // Slice CD6: the --ccd-render-shot showcase flag is listed in the showcase manifest.
        bool sawCcdRenderShot = false;
        // Slice MF1: the --mf1-faces-shot showcase flag is listed in the showcase manifest.
        bool sawMf1FacesShot = false;
        // Slice MF2: the --mf2-clip-shot showcase flag is listed in the showcase manifest.
        bool sawMf2ClipShot = false;
        // Slice MF3: the --mf3-manifold-shot showcase flag is listed in the showcase manifest.
        bool sawMf3ManifoldShot = false;
        // Slice MF4: the --mf4-stack-shot showcase flag is listed in the showcase manifest.
        bool sawMf4StackShot = false;
        // Slice MF5: the --mf5-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawMf5LockstepShot = false;
        // Slice VD1: the --vd1-world-shot showcase flag is listed in the showcase manifest.
        bool sawVd1WorldShot = false;
        // Slice VD2: the --vd2-tick-shot showcase flag is listed in the showcase manifest.
        bool sawVd2TickShot = false;
        // Slice VD3: the --vd3-world-shot showcase flag is listed in the showcase manifest.
        bool sawVd3WorldShot = false;
        // Slice VD4: the --vd4-snap-shot showcase flag is listed in the showcase manifest.
        bool sawVd4SnapShot = false;
        // Slice VD5: the --vd5-net-shot showcase flag is listed in the showcase manifest.
        bool sawVd5NetShot = false;
        // Slice VD6: the --vd6-game-shot showcase flag is listed in the showcase manifest.
        bool sawVd6GameShot = false;
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
        // Slice FR5: the --fract-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawFractLockstepShot = false;
        // Slice FR6: the --fract-render-shot showcase flag is listed in the showcase manifest.
        bool sawFractRenderShot = false;
        // Slice JT1: the --joint-ball-shot showcase flag is listed in the showcase manifest.
        bool sawJointBallShot = false;
        // Slice JT2: the --joint-hinge-shot showcase flag is listed in the showcase manifest.
        bool sawJointHingeShot = false;
        // Slice JT3: the --joint-step-shot showcase flag is listed in the showcase manifest.
        bool sawJointStepShot = false;
        // Slice JT4: the --joint-ragdoll-shot showcase flag is listed in the showcase manifest.
        bool sawJointRagdollShot = false;
        // Slice JT5: the --joint-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawJointLockstepShot = false;
        // Slice JT6: the --joint-render-shot showcase flag is listed in the showcase manifest.
        bool sawJointRenderShot = false;
        // Slice VH1: the --vehicle-spring-shot showcase flag is listed in the showcase manifest.
        bool sawVehicleSpringShot = false;
        // Slice VH2: the --vehicle-rig-shot showcase flag is listed in the showcase manifest.
        bool sawVehicleRigShot = false;
        // Slice VH3: the --vehicle-step-shot showcase flag is listed in the showcase manifest.
        bool sawVehicleStepShot = false;
        // Slice VH4: the --vehicle-traction-shot showcase flag is listed in the showcase manifest.
        bool sawVehicleTractionShot = false;
        // Slice VH5: the --vehicle-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawVehicleLockstepShot = false;
        // Slice VH6: the --vehicle-render-shot showcase flag is listed in the showcase manifest.
        bool sawVehicleRenderShot = false;
        // Slice AC1: the --active-drive-shot showcase flag is listed in the showcase manifest.
        bool sawActiveDriveShot = false;
        // Slice AC2: the --active-blend-shot showcase flag is listed in the showcase manifest.
        bool sawActiveBlendShot = false;
        // Slice AC3: the --active-step-shot showcase flag is listed in the showcase manifest.
        bool sawActiveStepShot = false;
        // Slice AC4: the --active-recover-shot showcase flag is listed in the showcase manifest.
        bool sawActiveRecoverShot = false;
        // Slice AC5: the --active-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawActiveLockstepShot = false;
        // Slice AC6: the --active-render-shot showcase flag is listed in the showcase manifest.
        bool sawActiveRenderShot = false;
        // Slice BD1: the --boids-steer-shot showcase flag is listed in the showcase manifest.
        bool sawBoidsSteerShot = false;
        // Slice BD2: the --boids-neighbors-shot showcase flag is listed in the showcase manifest.
        bool sawBoidsNeighborsShot = false;
        // Slice BD3: the --boids-flock-shot showcase flag is listed in the showcase manifest.
        bool sawBoidsFlockShot = false;
        // Slice BD4: the --boids-path-shot showcase flag is listed in the showcase manifest.
        bool sawBoidsPathShot = false;
        // Slice BD5: the --boids-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawBoidsLockstepShot = false;
        // Slice BD6: the --boids-render-shot showcase flag is listed in the showcase manifest.
        bool sawBoidsRenderShot = false;
        // Slice CX1: the --convex-sat-shot showcase flag is listed in the showcase manifest.
        bool sawConvexSatShot = false;
        // Slice CX2: the --convex-manifold-shot showcase flag is listed in the showcase manifest.
        bool sawConvexManifoldShot = false;
        // Slice CX3: the --convex-tumble-shot showcase flag is listed in the showcase manifest.
        bool sawConvexTumbleShot = false;
        // Slice CX4: the --convex-stack-shot showcase flag is listed in the showcase manifest.
        bool sawConvexStackShot = false;
        // Slice CX5: the --convex-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawConvexLockstepShot = false;
        // Slice CX6: the --convex-render-shot showcase flag is listed in the showcase manifest.
        bool sawConvexRenderShot = false;
        // Slice FC1: the --fric-basis-shot showcase flag is listed in the showcase manifest.
        bool sawFricBasisShot = false;
        // Slice FC2: the --fric-points-shot showcase flag is listed in the showcase manifest.
        bool sawFricPointsShot = false;
        // Slice FC3: the --fric-solve-shot showcase flag is listed in the showcase manifest.
        bool sawFricSolveShot = false;
        // Slice FC4: the --fric-ramp-shot showcase flag is listed in the showcase manifest.
        bool sawFricRampShot = false;
        // Slice PS1: the --persist-key-shot showcase flag is listed in the showcase manifest.
        bool sawPersistKeyShot = false;
        // Slice WH1: the --wh1-keys-shot showcase flag is listed in the showcase manifest.
        bool sawWh1KeysShot = false;
        // Slice WH2: the --wh2-cache-shot showcase flag is listed in the showcase manifest.
        bool sawWh2CacheShot = false;
        // Slice HF1: the --hf1-points-shot showcase flag is listed in the showcase manifest.
        bool sawHf1PointsShot = false;
        // Slice HF2: the --hf2-warm-shot showcase flag is listed in the showcase manifest.
        bool sawHf2WarmShot = false;
        // Slice HF3: the --hf3-step-shot showcase flag is listed in the showcase manifest.
        bool sawHf3StepShot = false;
        // Slice HF4: the --hf4-joint-shot showcase flag is listed in the showcase manifest.
        bool sawHf4JointShot = false;
        // Slice HF5: the --hf5-net-shot showcase flag is listed in the showcase manifest.
        bool sawHf5NetShot = false;
        // Slice HF6: the --hf6-hull-shot showcase flag is listed in the showcase manifest.
        bool sawHf6HullShot = false;
        // Slice IK1: the --ik1-angle-shot showcase flag is listed in the showcase manifest.
        bool sawIk1AngleShot = false;
        // Slice IK2: the --ik2-twobone-shot showcase flag is listed in the showcase manifest.
        bool sawIk2TwoboneShot = false;
        // Slice IK3: the --ik3-fabrik-shot showcase flag is listed in the showcase manifest.
        bool sawIk3FabrikShot = false;
        // Slice WH3: the --wh3-warm-shot showcase flag is listed in the showcase manifest.
        bool sawWh3WarmShot = false;
        // Slice WH4: the --wh4-stack-shot showcase flag is listed in the showcase manifest.
        bool sawWh4StackShot = false;
        // Slice WH5: the --wh5-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawWh5LockstepShot = false;
        // Slice WH6: the --wh6-render-shot showcase flag is listed in the showcase manifest.
        bool sawWh6RenderShot = false;
        // Slice PS2: the --persist-cache-shot showcase flag is listed in the showcase manifest.
        bool sawPersistCacheShot = false;
        // Slice PS3: the --persist-warm-shot showcase flag is listed in the showcase manifest.
        bool sawPersistWarmShot = false;
        // Slice PS4: the --persist-sleep-shot showcase flag is listed in the showcase manifest.
        bool sawPersistSleepShot = false;
        // Slice PS5: the --persist-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawPersistLockstepShot = false;
        // Slice PS6: the --persist-render-shot showcase flag is listed in the showcase manifest.
        bool sawPersistRenderShot = false;
        // Slice GJ1: the --gjk-support-shot showcase flag is listed in the showcase manifest.
        bool sawGjkSupportShot = false;
        // Slice GJ2: the --gjk-distance-shot showcase flag is listed in the showcase manifest.
        bool sawGjkDistanceShot = false;
        // Slice GJ3: the --gjk-epa-shot showcase flag is listed in the showcase manifest.
        bool sawGjkEpaShot = false;
        // Slice CD1: the --ccd-toi-shot showcase flag is listed in the showcase manifest.
        bool sawCcdToiShot = false;
        // Slice CD2: the --ccd-swept-shot showcase flag is listed in the showcase manifest.
        bool sawCcdSweptShot = false;
        // Slice CD3: the --ccd-step-shot showcase flag is listed in the showcase manifest.
        bool sawCcdStepShot = false;
        // Slice CD4: the --ccd-bullet-shot showcase flag is listed in the showcase manifest.
        bool sawCcdBulletShot = false;
        // Slice CD5: the --ccd-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawCcdLockstepShot = false;
        // Slice GJ4: the --gjk-settle-shot showcase flag is listed in the showcase manifest.
        bool sawGjkSettleShot = false;
        // Slice GJ5: the --gjk-lockstep-shot showcase flag is listed in the showcase manifest.
        bool sawGjkLockstepShot = false;
        // Slice GJ6: the --gjk-render-shot showcase flag is listed in the showcase manifest.
        bool sawGjkRenderShot = false;
        // Slice RT1: the --rt1-trace-shot showcase flag is listed in the showcase manifest.
        bool sawRt1TraceShot = false;
        // Slice RT2: the --rt2-query-shot showcase flag is listed in the showcase manifest.
        bool sawRt2QueryShot = false;
        // Slice RT3: the --rt3-shadow-shot showcase flag is listed in the showcase manifest.
        bool sawRt3ShadowShot = false;
        // Slice RT4: the --rt4-reflect-shot showcase flag is listed in the showcase manifest.
        bool sawRt4ReflectShot = false;
        // Slice RT5: the --rt5-simrender-shot showcase flag is listed in the showcase manifest.
        bool sawRt5SimrenderShot = false;
        // Slice RT6: the --rt6-hero-shot showcase flag is listed in the showcase manifest.
        bool sawRt6HeroShot = false;
        // Slice GI1: the --gi1-probe-shot showcase flag is listed in the showcase manifest.
        bool sawGi1ProbeShot = false;
        // Slice GI2: the --gi2-shencode-shot showcase flag is listed in the showcase manifest.
        bool sawGi2ShencodeShot = false;
        // Slice GI3: the --gi3-interp-shot showcase flag is listed in the showcase manifest.
        bool sawGi3InterpShot = false;
        // Slice GI4: the --gi4-bounce-shot showcase flag is listed in the showcase manifest.
        bool sawGi4BounceShot = false;
        // Slice GI5: the --gi5-occlusion-shot showcase flag is listed in the showcase manifest.
        bool sawGi5OcclusionShot = false;
        // Slice GI6: the --gi6-hero-shot showcase flag is listed in the showcase manifest.
        bool sawGi6HeroShot = false;
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
                if (s && AsString(MemberOf(s, "flag")) == "--broad-cell-shot") sawBroadCellShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--broad-pair-shot") sawBroadPairShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--broad-hull-shot") sawBroadHullShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--broad-lockstep-shot") sawBroadLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--broad-render-shot") sawBroadRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ccd-render-shot") sawCcdRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mf1-faces-shot") sawMf1FacesShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mf2-clip-shot") sawMf2ClipShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mf3-manifold-shot") sawMf3ManifoldShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mf4-stack-shot") sawMf4StackShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--mf5-lockstep-shot") sawMf5LockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vd1-world-shot") sawVd1WorldShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vd2-tick-shot") sawVd2TickShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vd3-world-shot") sawVd3WorldShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vd4-snap-shot") sawVd4SnapShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vd5-net-shot") sawVd5NetShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vd6-game-shot") sawVd6GameShot = true;
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
                if (s && AsString(MemberOf(s, "flag")) == "--fract-lockstep-shot") sawFractLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fract-render-shot") sawFractRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--joint-ball-shot") sawJointBallShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--joint-hinge-shot") sawJointHingeShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--joint-step-shot") sawJointStepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--joint-ragdoll-shot") sawJointRagdollShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--joint-lockstep-shot") sawJointLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--joint-render-shot") sawJointRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vehicle-spring-shot") sawVehicleSpringShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vehicle-rig-shot") sawVehicleRigShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vehicle-step-shot") sawVehicleStepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vehicle-traction-shot") sawVehicleTractionShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vehicle-lockstep-shot") sawVehicleLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--vehicle-render-shot") sawVehicleRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--active-drive-shot") sawActiveDriveShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--active-blend-shot") sawActiveBlendShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--active-step-shot") sawActiveStepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--active-recover-shot") sawActiveRecoverShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--active-lockstep-shot") sawActiveLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--active-render-shot") sawActiveRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--boids-steer-shot") sawBoidsSteerShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--boids-neighbors-shot") sawBoidsNeighborsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--boids-flock-shot") sawBoidsFlockShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--boids-path-shot") sawBoidsPathShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--boids-lockstep-shot") sawBoidsLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--boids-render-shot") sawBoidsRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--convex-sat-shot") sawConvexSatShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--convex-manifold-shot") sawConvexManifoldShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--convex-tumble-shot") sawConvexTumbleShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--convex-stack-shot") sawConvexStackShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--convex-lockstep-shot") sawConvexLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--convex-render-shot") sawConvexRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fric-basis-shot") sawFricBasisShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--persist-key-shot") sawPersistKeyShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--wh1-keys-shot") sawWh1KeysShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--wh2-cache-shot") sawWh2CacheShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--hf1-points-shot") sawHf1PointsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--hf2-warm-shot") sawHf2WarmShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--hf3-step-shot") sawHf3StepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--hf4-joint-shot") sawHf4JointShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--hf5-net-shot") sawHf5NetShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--hf6-hull-shot") sawHf6HullShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ik1-angle-shot") sawIk1AngleShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ik2-twobone-shot") sawIk2TwoboneShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ik3-fabrik-shot") sawIk3FabrikShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--wh3-warm-shot") sawWh3WarmShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--wh4-stack-shot") sawWh4StackShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--wh5-lockstep-shot") sawWh5LockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--wh6-render-shot") sawWh6RenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--persist-cache-shot") sawPersistCacheShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--persist-warm-shot") sawPersistWarmShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--persist-sleep-shot") sawPersistSleepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--persist-lockstep-shot") sawPersistLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--persist-render-shot") sawPersistRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gjk-support-shot") sawGjkSupportShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gjk-distance-shot") sawGjkDistanceShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gjk-epa-shot") sawGjkEpaShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ccd-toi-shot") sawCcdToiShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ccd-swept-shot") sawCcdSweptShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ccd-step-shot") sawCcdStepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ccd-bullet-shot") sawCcdBulletShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--ccd-lockstep-shot") sawCcdLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gjk-settle-shot") sawGjkSettleShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gjk-lockstep-shot") sawGjkLockstepShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gjk-render-shot") sawGjkRenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--rt1-trace-shot") sawRt1TraceShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--rt2-query-shot") sawRt2QueryShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--rt3-shadow-shot") sawRt3ShadowShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--rt4-reflect-shot") sawRt4ReflectShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--rt5-simrender-shot") sawRt5SimrenderShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--rt6-hero-shot") sawRt6HeroShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gi1-probe-shot") sawGi1ProbeShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gi2-shencode-shot") sawGi2ShencodeShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gi3-interp-shot") sawGi3InterpShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gi4-bounce-shot") sawGi4BounceShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gi5-occlusion-shot") sawGi5OcclusionShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--gi6-hero-shot") sawGi6HeroShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fric-points-shot") sawFricPointsShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fric-solve-shot") sawFricSolveShot = true;
                if (s && AsString(MemberOf(s, "flag")) == "--fric-ramp-shot") sawFricRampShot = true;
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
        check(sawBroadCellShot, "showcases manifest includes --broad-cell-shot");
        check(sawBroadPairShot, "showcases manifest includes --broad-pair-shot");
        check(sawBroadHullShot, "showcases manifest includes --broad-hull-shot");
        check(sawBroadLockstepShot, "showcases manifest includes --broad-lockstep-shot");
        check(sawBroadRenderShot, "showcases manifest includes --broad-render-shot");
        check(sawCcdRenderShot, "showcases manifest includes --ccd-render-shot");
        check(sawMf1FacesShot, "showcases manifest includes --mf1-faces-shot");
        check(sawMf2ClipShot, "showcases manifest includes --mf2-clip-shot");
        check(sawMf3ManifoldShot, "showcases manifest includes --mf3-manifold-shot");
        check(sawMf4StackShot, "showcases manifest includes --mf4-stack-shot");
        check(sawMf5LockstepShot, "showcases manifest includes --mf5-lockstep-shot");
        check(sawVd1WorldShot, "showcases manifest includes --vd1-world-shot");
        check(sawVd2TickShot, "showcases manifest includes --vd2-tick-shot");
        check(sawVd3WorldShot, "showcases manifest includes --vd3-world-shot");
        check(sawVd4SnapShot, "showcases manifest includes --vd4-snap-shot");
        check(sawVd5NetShot, "showcases manifest includes --vd5-net-shot");
        check(sawVd6GameShot, "showcases manifest includes --vd6-game-shot");
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
        check(sawFractLockstepShot, "showcases manifest includes --fract-lockstep-shot");
        check(sawFractRenderShot, "showcases manifest includes --fract-render-shot");
        check(sawJointBallShot, "showcases manifest includes --joint-ball-shot");
        check(sawJointHingeShot, "showcases manifest includes --joint-hinge-shot");
        check(sawJointStepShot, "showcases manifest includes --joint-step-shot");
        check(sawJointRagdollShot, "showcases manifest includes --joint-ragdoll-shot");
        check(sawJointLockstepShot, "showcases manifest includes --joint-lockstep-shot");
        check(sawJointRenderShot, "showcases manifest includes --joint-render-shot");
        check(sawVehicleSpringShot, "showcases manifest includes --vehicle-spring-shot");
        check(sawVehicleRigShot, "showcases manifest includes --vehicle-rig-shot");
        check(sawVehicleStepShot, "showcases manifest includes --vehicle-step-shot");
        check(sawVehicleTractionShot, "showcases manifest includes --vehicle-traction-shot");
        check(sawVehicleLockstepShot, "showcases manifest includes --vehicle-lockstep-shot");
        check(sawVehicleRenderShot, "showcases manifest includes --vehicle-render-shot");
        check(sawActiveDriveShot, "showcases manifest includes --active-drive-shot");
        check(sawActiveBlendShot, "showcases manifest includes --active-blend-shot");
        check(sawActiveStepShot, "showcases manifest includes --active-step-shot");
        check(sawActiveRecoverShot, "showcases manifest includes --active-recover-shot");
        check(sawActiveLockstepShot, "showcases manifest includes --active-lockstep-shot");
        check(sawActiveRenderShot, "showcases manifest includes --active-render-shot");
        check(sawBoidsSteerShot, "showcases manifest includes --boids-steer-shot");
        check(sawBoidsNeighborsShot, "showcases manifest includes --boids-neighbors-shot");
        check(sawBoidsFlockShot, "showcases manifest includes --boids-flock-shot");
        check(sawBoidsPathShot, "showcases manifest includes --boids-path-shot");
        check(sawBoidsLockstepShot, "showcases manifest includes --boids-lockstep-shot");
        check(sawBoidsRenderShot, "showcases manifest includes --boids-render-shot");
        check(sawConvexSatShot, "showcases manifest includes --convex-sat-shot");
        check(sawConvexManifoldShot, "showcases manifest includes --convex-manifold-shot");
        check(sawConvexTumbleShot, "showcases manifest includes --convex-tumble-shot");
        check(sawConvexStackShot, "showcases manifest includes --convex-stack-shot");
        check(sawConvexLockstepShot, "showcases manifest includes --convex-lockstep-shot");
        check(sawConvexRenderShot, "showcases manifest includes --convex-render-shot");
        check(sawFricBasisShot, "showcases manifest includes --fric-basis-shot");
        check(sawPersistKeyShot, "showcases manifest includes --persist-key-shot");
        check(sawWh1KeysShot, "showcases manifest includes --wh1-keys-shot");
        check(sawWh2CacheShot, "showcases manifest includes --wh2-cache-shot");
        check(sawHf1PointsShot, "showcases manifest includes --hf1-points-shot");
        check(sawHf2WarmShot, "showcases manifest includes --hf2-warm-shot");
        check(sawHf3StepShot, "showcases manifest includes --hf3-step-shot");
        check(sawHf4JointShot, "showcases manifest includes --hf4-joint-shot");
        check(sawHf5NetShot, "showcases manifest includes --hf5-net-shot");
        check(sawHf6HullShot, "showcases manifest includes --hf6-hull-shot");
        check(sawIk1AngleShot, "showcases manifest includes --ik1-angle-shot");
        check(sawIk2TwoboneShot, "showcases manifest includes --ik2-twobone-shot");
        check(sawIk3FabrikShot, "showcases manifest includes --ik3-fabrik-shot");
        check(sawWh3WarmShot, "showcases manifest includes --wh3-warm-shot");
        check(sawWh4StackShot, "showcases manifest includes --wh4-stack-shot");
        check(sawWh5LockstepShot, "showcases manifest includes --wh5-lockstep-shot");
        check(sawWh6RenderShot, "showcases manifest includes --wh6-render-shot");
        check(sawPersistCacheShot, "showcases manifest includes --persist-cache-shot");
        check(sawPersistWarmShot, "showcases manifest includes --persist-warm-shot");
        check(sawPersistSleepShot, "showcases manifest includes --persist-sleep-shot");
        check(sawPersistLockstepShot, "showcases manifest includes --persist-lockstep-shot");
        check(sawPersistRenderShot, "showcases manifest includes --persist-render-shot");
        check(sawGjkSupportShot, "showcases manifest includes --gjk-support-shot");
        check(sawGjkDistanceShot, "showcases manifest includes --gjk-distance-shot");
        check(sawGjkEpaShot, "showcases manifest includes --gjk-epa-shot");
        check(sawCcdToiShot, "showcases manifest includes --ccd-toi-shot");
        check(sawCcdSweptShot, "showcases manifest includes --ccd-swept-shot");
        check(sawCcdStepShot, "showcases manifest includes --ccd-step-shot");
        check(sawCcdBulletShot, "showcases manifest includes --ccd-bullet-shot");
        check(sawCcdLockstepShot, "showcases manifest includes --ccd-lockstep-shot");
        check(sawGjkSettleShot, "showcases manifest includes --gjk-settle-shot");
        check(sawGjkLockstepShot, "showcases manifest includes --gjk-lockstep-shot");
        check(sawGjkRenderShot, "showcases manifest includes --gjk-render-shot");
        check(sawRt1TraceShot, "showcases manifest includes --rt1-trace-shot");
        check(sawRt2QueryShot, "showcases manifest includes --rt2-query-shot");
        check(sawRt3ShadowShot, "showcases manifest includes --rt3-shadow-shot");
        check(sawRt4ReflectShot, "showcases manifest includes --rt4-reflect-shot");
        check(sawRt5SimrenderShot, "showcases manifest includes --rt5-simrender-shot");
        check(sawRt6HeroShot, "showcases manifest includes --rt6-hero-shot");
        check(sawGi1ProbeShot, "showcases manifest includes --gi1-probe-shot");
        check(sawGi2ShencodeShot, "showcases manifest includes --gi2-shencode-shot");
        check(sawGi3InterpShot, "showcases manifest includes --gi3-interp-shot");
        check(sawGi4BounceShot, "showcases manifest includes --gi4-bounce-shot");
        check(sawGi5OcclusionShot, "showcases manifest includes --gi5-occlusion-shot");
        check(sawGi6HeroShot, "showcases manifest includes --gi6-hero-shot");
        check(sawFricPointsShot, "showcases manifest includes --fric-points-shot");
        check(sawFricSolveShot, "showcases manifest includes --fric-solve-shot");
        check(sawFricRampShot, "showcases manifest includes --fric-ramp-shot");
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
