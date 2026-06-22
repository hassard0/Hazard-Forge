#pragma once
// Hazard Forge — agentic command interface (scriptable scene driver).
//
// This is the command channel that closes the agentic-dev loop. The scene is already authorable as
// JSON (scene_io LoadScene) and inspectable (DumpScene / --dump-scene); this layer adds the missing
// verb: MUTATE + CAPTURE. An AI agent (or CI) hands the engine a JSON array of commands and the
// engine applies them headlessly — querying state, mutating the live ECS, and capturing renders —
// so a scene can be driven entirely from a script.
//
// Backend-agnostic by construction: this layer touches the ECS + SceneResources only as opaque,
// named values (exactly like scene_io). The one rendering verb, "capture", is delegated to a
// std::function the caller supplies, so engine/scene/commands.cpp pulls in NO vk*/Metal/rhi
// rendering symbols. (rhi::ITexture* appears only as an opaque pointer via SceneResources, never
// dereferenced — same contract scene_io already relies on.)
//
// Entity addressing — commands refer to an entity by its VIEW-ORDER INDEX: its position when the
// scene is iterated by view<TransformC, MeshC, MaterialC>() (creation order). This is the SAME
// index DumpScene emits, --dump-scene prints, and the editor hierarchy shows, so "entity 9" means
// the 10th object in the dumped scene array. Indices are resolved fresh per command, so after a
// "remove" the indices of later entities shift down (just like the dumped array would).
//
// Command schema — a JSON array of objects, each with an "op":
//   {"op":"dump"}                          -> print DumpScene(...) JSON to stdout.
//   {"op":"list"}                          -> print a terse "index: mesh  pos=[..]" line per entity.
//   {"op":"set_transform","entity":N,      -> mutate entity N's TransformC (any subset of fields).
//        "position":[x,y,z],"euler":[..],"scale":[..]}
//   {"op":"set_material","entity":N,       -> mutate entity N's MaterialC (any subset).
//        "metallic":f,"roughness":f,"baseColor":"name"|null,"normalMap":"name"|null}
//   {"op":"add","mesh":"name",...}         -> create an entity (same fields as a scene_io object);
//                                             prints the new entity's view-order index.
//   {"op":"remove","entity":N}             -> destroy entity N.
//   {"op":"capture","path":"out.bmp"}      -> render the current scene to a file via the callback.
//   {"op":"save_scene","path":"x.json"}    -> write DumpScene(...) to a file.
//   {"op":"introspect","path":"x.json"}    -> write the full machine-readable engine-state JSON
//                                             (editor::DescribeEngine) to a file (or stdout if no
//                                             path): engine/features/showcases/commands + scene +
//                                             stats. The agent-facing OBSERVE call.
//   {"op":"query","select":"entity",       -> print the SELECTIVE JSON read of one entity (the
//        "entity":N,"fields":[..]?}            transform/material/mesh object DumpScene emits for it,
//                                             optionally narrowed to "fields"; addressed by view-order
//                                             index). The read half of the request/response loop.
//   {"op":"query","select":"stats"}        -> print {entities, meshes, textures} resource counts.
//   {"op":"query","select":"entities"}     -> print a terse ordered [{index, mesh}] list.
//
// A baseColor/normalMap of JSON null (or "") clears the texture; an unknown texture name is an
// error, as is an unknown mesh name on "add" or an out-of-range entity index.
//
// query op (DX2) — a DETERMINISTIC, index-addressed, field-selectable JSON read of the scene:
//   select:"entity" — the requested entity's components. "fields" (optional) is any subset of
//     ["transform","material","mesh"]; ALL three are emitted (in that FIXED canonical order) when
//     "fields" is omitted, REGARDLESS of the request array's order. An out-of-range / negative
//     "entity" yields a deterministic {"error":"out-of-range","entity":N,"count":C} response (no
//     crash, no abort — the bus's bad-command-keeps-going contract). Unknown field names are listed
//     once under an "unknownFields" array (the recognized fields still emit).
//   select:"stats" — {"entities":E,"meshes":M,"textures":T}: the live entity count and the named
//     mesh/texture resource counts.
//   select:"entities" — a terse ordered list [{"index":i,"mesh":"name"}, ...] of every entity.
// The component shapes mirror DumpScene (scene_io's %g floats, 2-space indent) so responses match
// the house JSON style and are backend-identical.

#include "ecs/ecs.h"
#include "scene/scene_io.h"

#include <functional>

namespace hf::scene {

// The capture callback: render the current scene and write it to `path` (e.g. a BMP). Returns true
// on success. The sample supplies this, wiring the existing offscreen --shot render path; keeping it
// here as a std::function is what lets commands.cpp stay free of any rendering backend symbols.
using CaptureFn = std::function<bool(const char* path)>;

// Parse the JSON command array at `commandsJsonPath` and apply each command in order to `reg` +
// `resources`. "dump"/"list"/"add" write to stdout; "capture" invokes `capture` (may be null — then
// a capture command is reported as skipped). Returns true if every command applied successfully;
// throws std::runtime_error on a missing/malformed command file (a bad individual command is
// reported to stderr and makes the result false, but does not abort the remaining commands).
bool RunCommands(ecs::Registry& reg, SceneResources& resources, const char* commandsJsonPath,
                 const CaptureFn& capture);

// Apply commands already held in memory (the JSON array text). Same semantics as the path overload;
// used by the unit test (no file needed). Throws std::runtime_error on a malformed array.
bool RunCommandsFromText(ecs::Registry& reg, SceneResources& resources, const char* commandsJson,
                         const CaptureFn& capture);

// Run a JSON array of "query" objects (DX2 — the read half of the request/response SDK loop) against
// the live scene and return a DETERMINISTIC, pretty-printed JSON array of {"request":<query obj>,
// "response":<result>} pairs (scene_io's %g floats, 2-space indent). Each element's "request" is the
// verbatim query object; its "response" is the entity/stats/entities read (or the deterministic
// {"error":"out-of-range",...} for a bad index). File-free + side-effect-free (no stdout, no mutate)
// — the clean byte-golden artifact AND the unit-test entry. Throws std::runtime_error on a malformed
// top-level array. The same per-query response is what the live "query" op prints to stdout.
std::string RunQueries(ecs::Registry& reg, SceneResources& resources, const char* queriesJson);

}  // namespace hf::scene
