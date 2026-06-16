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
    {"--ssr-shot",          "Screen-space reflections."},
    {"--ssgi-shot",         "Screen-space global illumination (color-bleed indirect diffuse)."},
    {"--ssgi-denoise-shot", "SSGI bilateral spatial denoise (edge-preserving smoothing of the indirect-diffuse buffer)."},
    {"--decal-shot",        "Screen-space projected decals (top-down decal box on the ground)."},
    {"--poststack-shot",    "Data-driven post-process stack (ordered tonemap/grade/chromatic/vignette/grain chain)."},
    {"--volumetric-shot",   "Volumetric fog / light shafts."},
    {"--probe-shot",        "Reflection + irradiance probes (local cubemap GI)."},
    {"--taa-shot",          "Temporal anti-aliasing (jittered accumulation)."},
    {"--cull-shot",         "Frustum culling visualization (overview camera + kept/culled bounds)."},
    {"--gpu-cull-shot",     "GPU-driven culling + indirect draw (compute compaction, GPU draw count)."},
    {"--mdi-shot",          "GPU multi-draw-indirect batching (144 objects in 1 draw; MDI==per-draw byte-identical)."},
    {"--mt-shot",           "Multithreaded command recording (per-thread secondaries, 1-vs-N identical)."},
    {"--editor-shot",       "Docked editor UI (Scene Hierarchy / Inspector / Stats panels around a central scene Viewport, fixed selected entity)."},
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
    "screen-space-reflections",
    "screen-space-global-illumination",
    "decals",
    "post-process-stack",
    "volumetric-fog",
    "reflection-irradiance-probes",
    "temporal-anti-aliasing",
    "frustum-culling",
    "gpu-driven-culling",
    "gpu-multi-draw-indirect",
    "multithreaded-recording",
    "bloom",
    "ssao",
    "transparency",
    "hardware-instancing",
    "gpu-particles",
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
