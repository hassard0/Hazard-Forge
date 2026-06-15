# Slice AL — Engine-State Introspection (design)

Date: 2026-06-15
Branch: `slice-introspect`
Status: design + implementation

## Goal

Complete the AGENT-facing editor. The engine can already be MUTATED headlessly by an AI agent
(scene_io `LoadScene`/`DumpScene`, the scriptable `commands` interface). This slice adds the
structured OBSERVE side: a single, deterministic, machine-readable JSON document describing the
WHOLE live engine + scene state, so an agent can query "what is this engine, what can it render,
how do I mutate it, and what is in the scene right now?" in one call.

ADDITIVE only. No rendering / RHI / shader change. The 22 image goldens are untouched. Pure C++ in
`hf_core` (depends on scene / ecs / math / json only — zero backend symbols).

## New module: `engine/editor/introspect.{h,cpp}` (hf_core)

```cpp
namespace hf::editor {

struct LightDir   { math::Vec3 dir;   math::Vec3 color; };
struct LightPoint { math::Vec3 pos;   math::Vec3 color; float radius; float intensity; };
struct LightSpot  { math::Vec3 pos;   math::Vec3 dir;   math::Vec3 color;
                    float range; float innerDeg; float outerDeg; };

struct CameraState { math::Vec3 position; float yaw; float pitch; float fovDeg; };

// The non-ECS "world" bits the Registry cannot hold (camera, lights, backend name). The caller
// (sample / showcase) fills what it has; empty optionals are simply omitted from the dump.
struct EngineState {
    bool                    hasCamera = false;
    CameraState             camera{};
    bool                    hasDirectional = false;
    LightDir                directional{};
    std::vector<LightPoint> points;
    std::vector<LightSpot>  spots;
    std::string             backend;  // "vulkan" | "metal" — the LIVE backend, informational only.
};

std::string DescribeEngine(ecs::Registry& reg, const scene::SceneResources& resources,
                           const EngineState& extra);

}  // namespace hf::editor
```

`DescribeEngine` returns a pretty-printed JSON document with a STABLE key order and a FIXED float
format (`%g` on the double-promoted float, same helper scene_io uses), so two runs on identical
inputs are BYTE-IDENTICAL and the output can be committed as a text golden.

## JSON schema (stable key order)

```jsonc
{
  "engine": {
    "name": "Hazard Forge",
    "version": "<string>",
    "backends": ["vulkan", "metal"],   // the engine SHIPS both; constant.
    "activeBackend": "vulkan",          // the live backend of THIS process (from EngineState).
    "features": [ ... ]                 // shipped capability list (PBR, IBL, CSM, ... editor).
  },
  "showcases": [                        // headless render commands an agent can discover + run.
    { "flag": "--shot",        "desc": "..." },
    { "flag": "--pbr-shot",    "desc": "..." }, ...
  ],
  "commands": [                         // scriptable MUTATE ops (the commands.cpp verbs).
    { "op": "dump",          "args": {} },
    { "op": "set_transform", "args": { "entity": "int", "position": "[x,y,z]?",
                                       "euler": "[x,y,z]?", "scale": "[x,y,z]?" } }, ...
  ],
  "scene": {
    "entities": [
      { "id": 0, "generation": 0,
        "components": {
          "transform": { "position": [..], "euler": [..], "scale": [..] },
          "mesh": "<name>",
          "material": { "metallic": f, "roughness": f,
                        "baseColor": "<name>"|null, "normalMap": "<name>"|null } } },
      ...
    ]
  },
  "camera": { "position": [x,y,z], "yaw": f, "pitch": f, "fovDeg": f },   // omitted if absent
  "lights": {
    "directional": { "dir": [x,y,z], "color": [r,g,b] },                  // omitted if absent
    "points": [ { "pos": [..], "color": [..], "radius": f, "intensity": f } ],
    "spots":  [ { "pos": [..], "dir": [..], "color": [..],
                  "range": f, "innerDeg": f, "outerDeg": f } ]
  },
  "stats": {
    "entityCount": N,                   // entities with Transform+Mesh+Material (view order)
    "aliveEntities": N,                 // registry alive count
    "transformCount": N, "meshCount": N, "materialCount": N,
    "pointLightCount": N, "spotLightCount": N
  }
}
```

Entity `id`/`generation` are the ECS `Entity` handle fields. Entities are emitted in
view<Transform,Mesh,Material> order — the SAME order DumpScene / `--dump-scene` / the command
interface use to address entities, so an agent that reads `scene.entities[k]` can mutate it with
`{"op":"set_transform","entity":k,...}`.

The `features`, `showcases`, and `commands` arrays are a CONSTANT, curated manifest baked into
introspect.cpp (documented there). They describe the shipped engine, not per-run state, so they
are identical on every run and on every backend.

## Wiring

1. **commands.cpp** — add an `introspect` op:
   `{"op":"introspect","path":"out.json"}` writes the dump to a file (or to stdout if `path`
   absent/empty). To keep commands.cpp free of camera/light state, the op builds an `EngineState`
   with `backend=""` and no camera/lights and dumps scene+manifest — i.e. the scene-only view an
   agent can request mid-script. (The full camera/light dump is the `--introspect` sample entry.)
   The op is listed in the `commands` manifest.

2. **samples/hello_triangle/main.cpp** — add `--introspect [outpath]`:
   build the default scene Registry (same `LoadScene` path as `--dump-scene`), fill a
   representative `EngineState` (the showcase camera: eye {4.5,4,6.5}, the 60° fov; the directional
   key light dir (-0.5,-1,-0.3) color (0.95,0.93,0.85); the 3 point lights at their t=0 phase),
   set `backend="vulkan"`, call `DescribeEngine`, write to `outpath` or stdout. Headless, no GPU,
   deterministic.

## Verification (NO image golden — JSON is the artifact)

- **Unit test** `tests/introspect_test.cpp` (hf_core / ASan): build a known 2-entity Registry +
  a known EngineState (camera + directional + 2 point lights); call `DescribeEngine`; assert the
  JSON parses through json.h, has the expected entityCount, the known transform values, the camera
  fields, the light values, `backends == ["vulkan","metal"]`, the `commands` manifest includes
  `set_transform`, and that two calls produce a BYTE-IDENTICAL string (determinism).

- **Text golden** `tests/golden/introspect/default_scene.json`: the exact `--introspect` output for
  the default scene, committed. A check compares the live `--introspect` output to it byte-for-byte.
  Added to `scripts/verify.ps1` as a Windows-side exact-match step. It does NOT need the Mac: the
  document is pure C++ / backend-agnostic, so the bytes are identical on Vulkan and Metal (the only
  per-process variable, `activeBackend`, is set to the literal `"vulkan"` by the sample entry; the
  golden is the Windows/Vulkan run). Documented here.

## Backend-identical guarantee

`DescribeEngine` lives in hf_core and touches only `Registry` / `SceneResources` / `EngineState` /
`math` — all backend-agnostic. No `vk*` / `MTL*` / `mtl::` / `Backend::Metal` symbols are added.
An agent therefore receives the SAME introspection JSON regardless of whether the engine is running
its Vulkan or Metal backend; `activeBackend` is the only field that reflects the live backend and is
purely informational.
