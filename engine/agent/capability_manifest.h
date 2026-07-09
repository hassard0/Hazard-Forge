#pragma once
// Hazard Forge — machine-readable CAPABILITY MANIFEST (Slice AX2, FLAGSHIP #31 "THE AGENT EXPERIENCE").
//
// PRODUCTIZES the engine's uniquely-winnable 2nd axis: HF is an engine an AI agent can develop
// END-TO-END. Every capability ships behind a HEADLESS showcase FLAG and is proven by a byte-identical
// GOLDEN — so an agent DISCOVERS what the engine can do, DRIVES it headlessly, and VERIFIES its own
// change against a machine-checkable oracle, with NO human and NO GUI in the loop. UE5 has no
// equivalent: a GUI-bound editor + non-deterministic float sim you cannot golden-verify + no headless
// authored-scene->verify loop. This header packages that advantage as data.
//
// BuildCapabilityManifest() emits ONE deterministic, pretty-printed JSON document enumerating a
// REPRESENTATIVE (not exhaustive) slice of the engine's capability surface — dozens of the major
// capability families, each grouped, with, per capability:
//   { name, flag, golden, moat, description }
//     * flag   — the headless showcase flag that EXERCISES the capability (samples/hello_triangle),
//     * golden — the committed artifact (in scripts/verify.ps1) whose byte-compare VERIFIES it,
//     * moat   — the structural property that makes it agent-verifiable / beyond-UE5 (see kMoat*),
// plus a "verifyContract" block that states the agent-dev LOOP as machine-readable data (the oracle),
// and a contentHash (FNV-1a) over the volatile content so an agent can cheaply detect drift.
//
// SOURCE OF TRUTH: a canonical IN-CODE registry (the kGroups table below) — NOT parsed from verify.ps1
// at runtime. This keeps the generator self-contained + deterministic + backend-agnostic (identical
// bytes on Vulkan and Metal, byte-for-byte across runs and compilers). The companion
// capability_manifest_test cross-checks that EVERY cited golden name actually appears in
// scripts/verify.ps1 (the "manifest doesn't lie" proof) — the registry is representative, but honest.
//
// Pure: depends ONLY on the C++ stdlib (header-only). No ECS/Registry/SceneResources, no vk*/Metal/rhi.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace hf::agent {

// --- The moat-property vocabulary (a fixed, closed set; each capability names exactly one). ---------
// These are the STRUCTURAL reasons a capability is agent-verifiable and/or beyond UE5's reach.
inline constexpr const char* kMoatDeterministic     = "deterministic";        // strict byte-identical golden
inline constexpr const char* kMoatLockstepRollback  = "lockstep-rollback";    // bit-exact replay + rollback netcode
inline constexpr const char* kMoatCrossPlatform     = "cross-platform-bitexact"; // identical bytes CPU/Vulkan/Metal
inline constexpr const char* kMoatReproducible      = "reproducible";         // seed/inputs -> pinned digest
inline constexpr const char* kMoatHeadlessGolden    = "headless-golden";      // headless flag -> committed image/text golden

// --- The in-code registry (the single source of truth). ------------------------------------------

struct Capability {
    const char* name;    // stable machine id (kebab-case)
    const char* flag;    // the headless showcase flag that exercises it (samples/hello_triangle)
    const char* golden;  // the committed golden (present in scripts/verify.ps1) that verifies it
    const char* moat;    // one of the kMoat* constants
    const char* desc;    // one-line human description
};

struct CapGroup {
    const char* group;         // stable machine id for the family
    const char* description;   // one-line family description
    const Capability* caps;
    std::size_t count;
};

// -- deterministic-simulation: the moat crown jewels (bit-exact, lockstep-replayable physics UE5's
//    float Chaos cannot do). Each is proven by a strict golden + a lockstep headline. ---------------
inline constexpr Capability kSimCaps[] = {
    {"fixed-point-rigid-physics", "--fpx-render",    "fpx_render",    kMoatLockstepRollback,
     "Q16.16 fixed-point rigid-body solver (bit-identical CPU/Vulkan/Metal)."},
    {"fixed-point-physics-lockstep", "--fpx-lockstep", "fpx_lockstep", kMoatLockstepRollback,
     "Lockstep + rollback proof: two peers re-derive the pile bit-for-bit from inputs."},
    {"deterministic-cloth",       "--cloth-render",  "cloth_render",  kMoatLockstepRollback,
     "Position-based deterministic cloth (fixed-point PBD, lockstep-replayable)."},
    {"deterministic-fluid",       "--fluid-render",  "fluid_render",  kMoatLockstepRollback,
     "Position-based fluid (PBF) with a bit-exact density-constraint solve."},
    {"deterministic-granular",    "--grain-render",  "grain_render",  kMoatLockstepRollback,
     "Dry Coulomb-friction granular / sand (angle-of-repose), lockstep-replayable."},
    {"rigid-fluid-coupling",      "--couple-render", "couple_render", kMoatLockstepRollback,
     "Two-way rigid<->fluid coupling: a body buoyed/dragged by + displacing the fluid."},
    {"rigid-grain-coupling",      "--cgrain-render", "cgrain_render", kMoatLockstepRollback,
     "Two-way rigid<->grain coupling: a body sinks into + is supported by the sand bed."},
    {"grain-fluid-coupling",      "--cgf-render",    "cgf_render",    kMoatLockstepRollback,
     "Two-way grain<->fluid coupling (wet sand / mud / slurry)."},
    {"deterministic-fracture",    "--fract-render",  "fract_render",  kMoatLockstepRollback,
     "Rigid-body fracture/destruction, bit-exact + rollback-replayable (vs float Chaos)."},
    {"articulated-ragdoll",       "--joint-ragdoll", "joint_ragdoll", kMoatLockstepRollback,
     "Skeleton->ragdoll ball/cone joints; a bit-exact articulated mechanism."},
    {"vehicle-dynamics",          "--vehicle-render","vehicle_render",kMoatLockstepRollback,
     "Deterministic sprung-wheel vehicle rig (traction + drive), lockstep-replayable."},
    {"continuous-collision",      "--ccd-render",    "ccd_render",    kMoatLockstepRollback,
     "Deterministic continuous collision / time-of-impact (bullet-through-paper)."},
    {"convex-collision",          "--convex-render", "convex_render", kMoatLockstepRollback,
     "Deterministic convex SAT + manifold + impulse solver."},
};

// -- rendering: the standard AAA pipeline features, each behind a headless capture -> strict/float
//    golden. These prove the render surface is real + headless-verifiable. --------------------------
inline constexpr Capability kRenderCaps[] = {
    {"physically-based-rendering","--pbr",           "pbr_helmet",    kMoatHeadlessGolden,
     "Metallic-roughness PBR (glTF DamagedHelmet showcase)."},
    {"image-based-lighting",      "--ibl",           "ibl_helmet",    kMoatHeadlessGolden,
     "HDR image-based lighting (prefiltered specular + irradiance)."},
    {"bloom",                     "--bloom",         "bloom",         kMoatHeadlessGolden,
     "HDR threshold + separable-blur bloom."},
    {"screen-space-ambient-occlusion","--ssao",      "ssao",          kMoatHeadlessGolden,
     "Screen-space ambient occlusion."},
    {"screen-space-reflections",  "--ssr",           "ssr",           kMoatHeadlessGolden,
     "Screen-space reflections."},
    {"temporal-anti-aliasing",    "--taa",           "taa",           kMoatHeadlessGolden,
     "Temporal anti-aliasing (Halton jitter + history resolve)."},
    {"temporal-super-resolution", "--us2-tsr",       "us2_tsr",       kMoatHeadlessGolden,
     "Temporal super-resolution upscaling."},
    {"dynamic-diffuse-gi",        "--gi6-hero",      "gi6_hero",      kMoatDeterministic,
     "DDGI-style dynamic diffuse global illumination (integer probe trace)."},
    {"virtual-shadow-maps",       "--vsm-render",    "vsm_atlas",     kMoatHeadlessGolden,
     "Clipmap virtual shadow maps."},
    {"point-shadows",             "--point-shadow",  "point_shadow",  kMoatHeadlessGolden,
     "Omnidirectional point-light shadow cubemaps."},
    {"volumetric-shadows",        "--volshadows",    "vol_shadows",   kMoatHeadlessGolden,
     "Froxel volumetric shadows."},
    {"contact-shadows",           "--contactshadow", "contact_shadows",kMoatHeadlessGolden,
     "Screen-space contact shadows."},
    {"skinned-animation",         "--skinning",      "skinning",      kMoatHeadlessGolden,
     "GPU skeletal skinning (joint-palette skinned mesh)."},
    {"ray-traced-shadows",        "--rt3-shadow",    "rt3_shadow",    kMoatDeterministic,
     "Deterministic integer ray-traced hard shadows."},
    {"ray-traced-hero",           "--rt6-hero",      "rt6_hero",      kMoatDeterministic,
     "Lit ray-traced hero capstone (deterministic RT pipeline)."},
    {"gpu-isosurface-meshing",    "--mc-render",     "mc_render",     kMoatDeterministic,
     "GPU Marching-Cubes isosurface meshing (voxel/SDF -> bit-exact mesh)."},
};

// -- gameplay-ai: the AI + navigation + character pillars, each deterministic + lockstep where the
//    sim moat applies. -------------------------------------------------------------------------------
inline constexpr Capability kAiCaps[] = {
    {"behavior-tree-ai",          "--ai6-render",    "ai_render",     kMoatLockstepRollback,
     "Deterministic blackboard + decision-tree NPC agent (lit 3D render)."},
    {"ai-lockstep",               "--ai5-lockstep",  "ai5_lockstep",  kMoatLockstepRollback,
     "NPC AI lockstep + rollback (bit-identical replayable AI, vs float Detour)."},
    {"navmesh-pathfinding",       "--nav-path",      "nav_path",      kMoatDeterministic,
     "Recast/Detour-style navmesh + integer A* pathfinding."},
    {"navmesh-render",            "--nav-render",    "nav_render",    kMoatHeadlessGolden,
     "Navmesh generation pipeline capstone (lit 3D render)."},
    {"crowd-simulation",          "--cr1-crowd",     "cr1_crowd",     kMoatDeterministic,
     "Deterministic crowd simulation at 10k+ agents (O(N) grid separation)."},
    {"boids-flocking",            "--boids-render",  "boids_render",  kMoatLockstepRollback,
     "Deterministic boids flocking (neighbour grid), lockstep-replayable."},
    {"inverse-kinematics",        "--ik6-render",    "ik6_render",    kMoatDeterministic,
     "Deterministic IK control-rig (two-bone + FABRIK), fixed-point LUTs."},
    {"ik-lockstep",               "--ik5-lockstep",  "ik5_lockstep",  kMoatLockstepRollback,
     "IK-driven lockstep + rollback headline."},
};

// -- netcode: the deterministic-lockstep netcode substrate the whole sim moat rests on. ------------
inline constexpr Capability kNetCaps[] = {
    {"deterministic-lockstep-match","--vd5-net",     "vd5_net",       kMoatLockstepRollback,
     "Gameplay-world lockstep: a peer re-simulates the whole match bit-for-bit."},
    {"friction-joint-lockstep",   "--hf5-net",       "hf5_net",       kMoatLockstepRollback,
     "Friction + joint hull lockstep + rollback."},
    {"network-transport-sim",     "--netsim",        "netsim",        kMoatReproducible,
     "Deterministic network-transport simulation (loss/latency/jitter)."},
    {"client-prediction",         "--netpredict",    "netpredict",    kMoatReproducible,
     "Client-side prediction + reconciliation."},
    {"state-replication",         "--net",           "net",           kMoatReproducible,
     "Snapshot state-replication / delta encoding."},
};

// -- asset-authoring & procedural: import + the deterministic procedural generators (seed->pinned). --
inline constexpr Capability kAssetCaps[] = {
    {"gltf-scene-import",         "--scene",         "scene_import",  kMoatHeadlessGolden,
     "glTF scene-graph import (nodes/meshes/materials)."},
    {"animation-blend",           "--blend",         "anim_blend",    kMoatHeadlessGolden,
     "Skeletal animation cross-fade blending."},
    {"animation-state-machine",   "--anim-fsm",      "anim_fsm",      kMoatHeadlessGolden,
     "Parameter-driven animation state machine."},
    {"procedural-terrain",        "--pt6-hero",      "pt6_hero",      kMoatReproducible,
     "Procedural terrain (height + hydraulic/thermal erosion), seed-reproducible."},
    {"procedural-content-gen",    "--pcg6-field",    "pcg6_field",    kMoatReproducible,
     "Procedural content generation (seed -> byte-identical field digest)."},
    {"wave-function-collapse",    "--wfc6-render",   "wfc6_render",   kMoatReproducible,
     "Wave-function-collapse tiling generation, seed-reproducible."},
    {"dynamic-weather",           "--we6-hero",      "we6_hero",      kMoatReproducible,
     "Deterministic dynamic weather (clouds/precip/time-of-day)."},
    {"foliage-scatter",           "--fo6-hero",      "fo6_hero",      kMoatReproducible,
     "Deterministic foliage scatter + wind sway on terrain."},
};

// -- agent-native tooling: the DISCOVER + DRIVE + VERIFY surface an agent codes against (the axis
//    this whole manifest productizes). Each cites its TEXT golden (all present in verify.ps1). -------
inline constexpr Capability kAgentCaps[] = {
    {"engine-introspection",      "--introspect",    "default_scene.json", kMoatDeterministic,
     "Machine-readable full engine+scene state dump (DescribeEngine)."},
    {"agent-sdk-contract",        "--agent-api",     "agent_api.json", kMoatDeterministic,
     "Versioned Agent-SDK contract (capabilities + command schemas + contentHash)."},
    {"scene-query",               "--agent-query",   "query_responses.json", kMoatDeterministic,
     "Selective scene-query read protocol (request/response over the command bus)."},
    {"scene-authoring",           "--author-scene",  "authored_scene.json", kMoatDeterministic,
     "Declarative scene-spec authoring (LoadScene -> canonical DumpScene)."},
    {"headless-hot-reload",       "--hot-reload",    "reload_trace.json", kMoatDeterministic,
     "Deterministic headless scene hot-reload (reload == cold-load, no residue)."},
    {"record-replay",             "--replay-verify", "canonical.replay", kMoatReproducible,
     "Record -> replay -> assert-determinism harness (tamper-detecting digest)."},
    {"determinism-stress",        "--determinism-stress", "stress_report.json", kMoatReproducible,
     "Rollback determinism-stress fuzzer (every snapshot point recovers authority)."},
    {"material-graph-introspection","--material-introspect","showcase3_graph.json", kMoatDeterministic,
     "Pure-CPU material-graph introspection (deterministic JSON / DOT)."},
    {"capability-manifest",       "--capability-manifest", "capability_manifest.json", kMoatDeterministic,
     "THIS manifest: the machine-readable capability-discovery + agent-verify contract."},
};

inline constexpr CapGroup kGroups[] = {
    {"deterministic-simulation",
     "Bit-exact, lockstep-replayable physics/sim (cross-platform) — the beyond-UE5 moat.",
     kSimCaps, std::size(kSimCaps)},
    {"rendering",
     "The AAA render pipeline, each feature behind a headless capture + committed golden.",
     kRenderCaps, std::size(kRenderCaps)},
    {"gameplay-ai",
     "AI, navigation, crowds and character rigs — deterministic + lockstep where the moat applies.",
     kAiCaps, std::size(kAiCaps)},
    {"netcode",
     "The deterministic-lockstep netcode substrate (send inputs, re-simulate bit-for-bit).",
     kNetCaps, std::size(kNetCaps)},
    {"asset-authoring",
     "Asset import, animation and seed-reproducible procedural generation.",
     kAssetCaps, std::size(kAssetCaps)},
    {"agent-native-tooling",
     "The discover/drive/verify surface an AI agent codes against — HF's unique axis.",
     kAgentCaps, std::size(kAgentCaps)},
};

// --- Deterministic JSON emission (self-contained; mirrors editor/introspect.cpp's discipline). -----
namespace detail {

inline std::string Indent(int depth) { return std::string(static_cast<std::size_t>(depth) * 2, ' '); }

inline void AppendString(std::ostream& os, const char* s) {
    os << '"';
    for (const char* p = s; *p; ++p) {
        switch (*p) {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n";  break;
            case '\t': os << "\\t";  break;
            case '\r': os << "\\r";  break;
            default:   os << *p;      break;
        }
    }
    os << '"';
}

// FNV-1a 64-bit over the canonical concatenation of the VOLATILE content (group ids + every capability
// name/flag/golden/moat, in emit order), with a separator byte after each string so concatenation is
// unambiguous. The same fixed-hash discipline the agent-api contentHash uses. Deterministic + backend-
// agnostic — identical inputs always hash identically, cross-run + cross-compiler.
inline void FnvBytes(uint64_t& h, const char* s) {
    for (const char* p = s; *p; ++p) {
        h ^= static_cast<uint64_t>(static_cast<unsigned char>(*p));
        h *= 1099511628211ull;
    }
    h ^= 0x1Full;
    h *= 1099511628211ull;
}

inline std::string ContentHash() {
    uint64_t h = 1469598103934665603ull;
    for (const CapGroup& g : kGroups) {
        FnvBytes(h, g.group);
        for (std::size_t i = 0; i < g.count; ++i) {
            FnvBytes(h, g.caps[i].name);
            FnvBytes(h, g.caps[i].flag);
            FnvBytes(h, g.caps[i].golden);
            FnvBytes(h, g.caps[i].moat);
        }
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return std::string(buf);
}

}  // namespace detail

// --- Public API ----------------------------------------------------------------------------------

// Total number of capabilities across all groups.
inline std::size_t CapabilityCount() {
    std::size_t n = 0;
    for (const CapGroup& g : kGroups) n += g.count;
    return n;
}

// Number of capability groups.
inline std::size_t CapabilityGroupCount() { return std::size(kGroups); }

// Every golden name cited by the manifest, in emit order (for the "manifest doesn't lie" cross-check).
inline std::vector<std::string> CitedGoldenNames() {
    std::vector<std::string> out;
    for (const CapGroup& g : kGroups)
        for (std::size_t i = 0; i < g.count; ++i)
            out.emplace_back(g.caps[i].golden);
    return out;
}

// Every showcase flag cited by the manifest, in emit order.
inline std::vector<std::string> CitedFlags() {
    std::vector<std::string> out;
    for (const CapGroup& g : kGroups)
        for (std::size_t i = 0; i < g.count; ++i)
            out.emplace_back(g.caps[i].flag);
    return out;
}

// The 16-hex-digit content digest over the volatile content (drift detection).
inline std::string ContentHash() { return detail::ContentHash(); }

// Build the complete, deterministic, pretty-printed capability-manifest JSON document.
inline std::string BuildCapabilityManifest() {
    using detail::AppendString;
    using detail::Indent;
    std::ostringstream os;

    os << "{\n";
    os << Indent(1) << "\"schemaVersion\": 1,\n";
    os << Indent(1) << "\"engine\": \"Hazard Forge\",\n";
    os << Indent(1) << "\"thesis\": \"An AI agent can develop Hazard Forge end-to-end: discover "
                       "capabilities here, drive each behind a headless flag, and verify its own "
                       "change against a byte-identical golden oracle — no human, no GUI in the loop.\",\n";

    // The agent-dev LOOP as a machine-readable CONTRACT (the oracle an agent codes against).
    os << Indent(1) << "\"verifyContract\": {\n";
    os << Indent(2) << "\"loop\": [\"author\", \"build\", \"run-flag\", \"byte-compare-golden\"],\n";
    os << Indent(2) << "\"oracle\": \"a byte-identical golden (image or text) is the machine-checkable "
                       "pass/fail; the digest is the correctness signal an agent reads without a human.\",\n";
    os << Indent(2) << "\"driver\": \"samples/hello_triangle (headless showcase flags) + ctest "
                       "(pure-CPU deterministic tests)\",\n";
    os << Indent(2) << "\"harness\": \"scripts/verify.ps1\",\n";
    os << Indent(2) << "\"goldenDir\": \"tests/golden\",\n";
    os << Indent(2) << "\"howToVerify\": \"to add/verify capability X: run its flag F, byte-compare "
                       "its golden G; equal bytes == PASS, any diff == FAIL.\",\n";
    os << Indent(2) << "\"moatVsUe5\": \"UE5 is structurally agent-hostile: a GUI-bound editor (no "
                       "headless authored-scene->verify loop) and non-deterministic float simulation "
                       "(no byte-identical golden oracle). HF is headless + deterministic by "
                       "construction.\"\n";
    os << Indent(1) << "},\n";

    // The moat-property vocabulary (the closed set each capability's `moat` draws from).
    os << Indent(1) << "\"moatProperties\": [";
    const char* moats[] = {kMoatDeterministic, kMoatLockstepRollback, kMoatCrossPlatform,
                           kMoatReproducible, kMoatHeadlessGolden};
    for (std::size_t i = 0; i < std::size(moats); ++i) {
        if (i) os << ", ";
        AppendString(os, moats[i]);
    }
    os << "],\n";

    // The grouped capability registry.
    os << Indent(1) << "\"groups\": [\n";
    for (std::size_t gi = 0; gi < std::size(kGroups); ++gi) {
        const CapGroup& g = kGroups[gi];
        os << Indent(2) << "{\n";
        os << Indent(3) << "\"group\": ";     AppendString(os, g.group);        os << ",\n";
        os << Indent(3) << "\"description\": "; AppendString(os, g.description); os << ",\n";
        os << Indent(3) << "\"capabilities\": [\n";
        for (std::size_t i = 0; i < g.count; ++i) {
            const Capability& c = g.caps[i];
            os << Indent(4) << "{ \"name\": ";  AppendString(os, c.name);
            os << ", \"flag\": ";               AppendString(os, c.flag);
            os << ", \"golden\": ";             AppendString(os, c.golden);
            os << ", \"moat\": ";               AppendString(os, c.moat);
            os << ", \"description\": ";        AppendString(os, c.desc);
            os << " }";
            os << (i + 1 < g.count ? ",\n" : "\n");
        }
        os << Indent(3) << "]\n";
        os << Indent(2) << "}";
        os << (gi + 1 < std::size(kGroups) ? ",\n" : "\n");
    }
    os << Indent(1) << "],\n";

    os << Indent(1) << "\"groupCount\": " << CapabilityGroupCount() << ",\n";
    os << Indent(1) << "\"capabilityCount\": " << CapabilityCount() << ",\n";
    os << Indent(1) << "\"contentHash\": ";
    AppendString(os, detail::ContentHash().c_str());
    os << "\n";
    os << "}\n";
    return os.str();
}

}  // namespace hf::agent
