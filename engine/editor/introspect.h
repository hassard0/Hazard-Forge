#pragma once
// Hazard Forge — machine-readable engine-state introspection (Slice AL).
//
// The agent-facing OBSERVE side of the engine. An AI agent can already MUTATE the engine headlessly
// (scene_io LoadScene/DumpScene + the scriptable commands interface); this layer adds the structured
// "describe the whole live state" call so an agent can QUERY the engine in one shot: what engine is
// this, which backends + features ship, what can it render (showcases), how do I mutate it (the
// command verbs), and what is in the scene RIGHT NOW (entities + components, camera, lights, stats).
//
// DescribeEngine() returns ONE deterministic, pretty-printed JSON document with a STABLE key order
// and a FIXED float format (the same %g-on-double helper scene_io uses), so two runs on identical
// inputs are byte-identical — the output is committed as a text golden and diffed exactly.
//
// Backend-agnostic by construction: this lives in hf_core and touches ONLY the ECS Registry,
// SceneResources (opaque named pointers, never dereferenced — same contract as scene_io/commands),
// math, and json. NO vk*/Metal/rhi rendering symbols. An agent therefore gets the SAME introspection
// regardless of the live Vulkan/Metal backend; `EngineState::backend` is the only field that names
// the live backend and is purely informational.
//
// The non-ECS world bits (camera, lights, the live backend name) cannot live in the Registry, so the
// caller (the sample/showcase) fills an EngineState with whatever it has; absent pieces are simply
// omitted from the dump. This keeps DescribeEngine from having to reach into every showcase.

#include "ecs/ecs.h"
#include "math/math.h"
#include "scene/scene_io.h"

#include <string>
#include <vector>

namespace hf::editor {

// --- The non-ECS state the caller supplies (everything the Registry cannot hold). ----------------

struct CameraState {
    math::Vec3 position{0, 0, 0};
    float yaw = 0.0f;     // radians
    float pitch = 0.0f;   // radians
    float fovDeg = 60.0f; // vertical FOV in degrees
};

struct LightDir {
    math::Vec3 dir{0, -1, 0};
    math::Vec3 color{1, 1, 1};
};

struct LightPoint {
    math::Vec3 pos{0, 0, 0};
    math::Vec3 color{1, 1, 1};
    float radius = 1.0f;
    float intensity = 1.0f;
};

struct LightSpot {
    math::Vec3 pos{0, 0, 0};
    math::Vec3 dir{0, -1, 0};
    math::Vec3 color{1, 1, 1};
    float range = 10.0f;
    float innerDeg = 20.0f;
    float outerDeg = 30.0f;
};

// The caller-filled, non-ECS portion of the engine state. Anything left default/empty is OMITTED
// from the JSON (so a scene-only caller still produces a valid, smaller document).
struct EngineState {
    bool hasCamera = false;
    CameraState camera{};

    bool hasDirectional = false;
    LightDir directional{};

    std::vector<LightPoint> points;
    std::vector<LightSpot> spots;

    std::string backend;  // "vulkan" | "metal" — the LIVE backend of this process (informational).
};

// Build the complete, deterministic, pretty-printed engine-state JSON document. See introspect.h
// header comment + docs/superpowers/specs/2026-06-15-introspection-design.md for the schema.
//   reg        : the live ECS scene (entities with TransformC/MeshC/MaterialC).
//   resources  : the named GPU resources (used ONLY to reverse mesh/texture pointers to names).
//   extra      : the non-ECS bits (camera, lights, active backend) the caller fills.
std::string DescribeEngine(ecs::Registry& reg, const scene::SceneResources& resources,
                           const EngineState& extra);

// --- Slice DX1: the versioned Agent-SDK contract (FLAGSHIP #31, THE AGENT EXPERIENCE). -----------
//
// DescribeAgentApi() builds the ADDITIVE top-level "agentApi" block that DescribeEngine folds in as
// its trailing key, AND is exposed standalone via the `--agent-api` flag (the lightweight negotiation
// call). It is the contract an agent codes against:
//
//   "agentApi": {
//     "schemaVersion": 1,                                      // SHAPE-ONLY version (see below)
//     "capabilities": ["observe","mutate","author","replay","hot-reload"],
//     "commands": [ {"op":"...","capability":"...","args":[{"name":"...","type":"..."}]}, ... ],
//     "contentHash": "<hex>"                                   // digest over the VOLATILE content
//   }
//
// THE KEY DESIGN CALL: `schemaVersion` governs the SHAPE ONLY — the set/structure of keys and the
// shape of a command entry. It bumps ONLY when that structure changes, NOT when a showcase/feature/
// command is added. The volatile content (the showcase flags + feature names + command ops) is
// summarized by `contentHash`, a deterministic 64-bit FNV-1a over the canonical concatenation of
// those strings in their existing emit order. An agent pinned to schemaVersion:1 codes against the
// fixed shape and watches contentHash to cheaply detect content drift — future slices append a
// showcase -> contentHash changes + the golden rebakes -> schemaVersion HOLDS at 1. This is what
// makes it a CONTRACT rather than a dump.
//
// Pure (no Registry/SceneResources) — reads only the static showcase/feature/command tables — so it
// is deterministic + backend-agnostic (identical bytes on Vulkan and Metal). Returns the "agentApi"
// object as a STANDALONE top-level JSON document (no leading/trailing keys); DescribeEngine emits the
// IDENTICAL bytes (re-indented one level deeper) as its "agentApi" value.
std::string DescribeAgentApi();

// Counts over the static manifests, for the DX1 proof lines (showcases:<S>, features:<F>,
// commands:<C>, capabilities). Pure, deterministic, no allocation of the JSON.
size_t AgentApiShowcaseCount();
size_t AgentApiFeatureCount();
size_t AgentApiCommandCount();
size_t AgentApiCapabilityCount();

}  // namespace hf::editor
