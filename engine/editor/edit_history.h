#pragma once
// Hazard Forge — DETERMINISTIC UNDO/REDO COMMAND STACK for the scene editor (Slice ED5, pure CPU).
//
// Every editor edit is already a PURE OP (edit_ops.h ApplyTransformEdit/ApplyMaterialEdit; the
// flow_edit_ops.h graph ops). THIS module wraps them in a recorded, reversible, REPLAYABLE command
// stack: each command is a flat POD carrying the op's target plus the BEFORE payload (captured at
// record time) and the AFTER payload — the classic memento-command hybrid, with NO std::function /
// heap indirection, so the history is SERIALIZABLE BY CONSTRUCTION. Undo applies the BEFORE payload
// through the SAME pure ops; Redo re-applies AFTER; both are therefore exactly as deterministic as
// the ops themselves, and undo N + redo N restores BIT-IDENTICAL state (proven by DumpScene
// byte-compares in tests/edit_history_test.cpp and the --ed5-dry-run).
//
// THE MOAT APPLIED TO AUTHORING: SerializeHistory/DeserializeHistory turn the live edit session into
// a portable byte artifact (hand-LE bytes, the codebase discipline; DigestHistory = the FNV-1a-64
// DigestBytes currency) and ReplayHistory re-applies it onto a fresh scene, reproducing the edited
// state byte-for-byte. UE5's transaction system (FTransaction) is neither serializable nor
// replay-stable; this one is both.
//
// ENROLLMENT DESIGN (how op families join the stack):
//   1. Add an EditCmdKind tag + flat POD payload fields to EditCommand (fixed-size; before+after).
//   2. Add the target pointer to EditTargets (each command family applies against one target model;
//      a command whose target pointer is null is a safe no-op returning false).
//   3. Add a RecordedXxx wrapper that captures BEFORE, calls the EXISTING raw op, captures AFTER,
//      and Records (skipping bitwise no-ops). The raw ops stay byte-untouched — call sites opt in
//      by moving to the wrapper.
//   4. Extend ApplyCommand's switch (forward = AFTER payload, backward = BEFORE payload) and the
//      Serialize/Deserialize per-kind field lists (append-only; bump kHistoryVersion on layout
//      change).
// Slice ED5 enrolls the scene ops (Transform/Material) FULLY plus the FLOW family (add/connect/
// disconnect/delete) as the extensibility proof. The seq/widget op families (seq_edit_ops.h,
// widget_edit_ops.h) follow the same recipe and are documented enrollment work, not wired here.
// Slice ED6 enrolls ENTITY CREATION (the asset browser's click-to-place, edit_ops2.h) via this
// exact recipe: EntityCreate appended to the kind enum, a meshPtr field appended to EditCommand,
// and per-kind serialize/deserialize cases appended (existing kinds' bytes untouched, so the ED5
// pinned digest still holds and kHistoryVersion stays 1).
//
// Scope/limits (documented, LOCKED):
//   * MaterialC.baseColor is an OPAQUE POINTER (the scene_io contract): the LIVE command stores it
//     as an opaque 64-bit VALUE, never dereferenced — in-process undo/redo/replay is exact. For the
//     SERIALIZED artifact, pass a SceneResources to Serialize/Deserialize/DigestHistory and the
//     pointer is resolved to its registered NAME (the exact DumpScene reverse-mapping), making the
//     artifact PROCESS-PORTABLE: two sessions recording the same logical edits serialize to
//     byte-identical files regardless of pointer values, and deserializing against another
//     process's SceneResources re-binds the names to that process's pointers. Without a resources
//     table the raw 64-bit values are written (an in-process-only artifact; the header flags say
//     which encoding a file carries, and a named artifact REQUIRES resources to deserialize).
//   * Scene commands address entities by VIEW-ORDER INDEX (the edit_ops addressing). The index
//     stays stable under the ONE lifetime op enrolled (ED6 EntityCreate) because it only ever
//     APPENDS: a spawn lands at the END of the drawable view (every existing index unchanged),
//     and its undo destroys the entity only while it is STILL the last drawable (guarded by
//     ApplyDestroyLastEntity; guaranteed by LIFO undo — every later command was undone first),
//     which pops the last dense slot of each pool without a swap, restoring the surviving view
//     order bit-identically. Arbitrary mid-list entity DELETION is NOT enrolled (that WOULD shift
//     every later view index and invalidate recorded targets); a general delete family would need
//     its own re-insertion payload, like FlowDelete's.
//   * FlowDelete captures the severed inbound references inline (kFlowMaxCutRefs). A victim with
//     more inbound references than fit CANNOT be recorded exactly; RecordedDeleteFlowNode then
//     still performs the delete but CLEARS the history (a deterministic, documented invalidation —
//     never a corrupt undo stack). Editor graphs are far below the cap.
//
// Pure CPU: ECS registry + flow graph + math + the DigestBytes currency. ZERO vk*/Metal/rhi symbols
// beyond the opaque-pointer VALUES edit_ops already carries. Header-only; lives with edit_ops.h.

#include <cstdint>
#include <cstring>
#include <vector>

#include "ecs/ecs.h"
#include "editor/edit_ops.h"        // the scene ops (ApplyTransformEdit/ApplyMaterialEdit)
#include "editor/edit_ops2.h"       // the ED6 entity-creation op (ApplyCreateEntity + undo twin)
#include "editor/flow_edit_ops.h"   // the flow ops (AddFlowNode/ConnectFlow/DeleteFlowNode)
#include "net/session.h"            // hf::net::DigestBytes — the pinned-digest FNV-1a-64 currency
#include "scene/components.h"       // TransformC/MaterialC (the state the scene commands snapshot)
#include "scene/scene_io.h"         // SceneResources — the name<->pointer map for PORTABLE artifacts

namespace hf::editor {

// --- The command envelope -------------------------------------------------------------------------

// FIXED numbering = the wire contract (serialized; never renumber, append only).
enum class EditCmdKind : uint32_t {
    None           = 0,
    Transform      = 1,   // scene: full TransformC before/after (target = view-order index)
    Material       = 2,   // scene: MaterialC editable fields before/after (target = view-order index)
    FlowAdd        = 3,   // flow: AddFlowNode(kind, constArg) (target = the new NodeId)
    FlowConnect    = 4,   // flow: one input slot rewired (target = `to`; before/after refs)
    FlowDisconnect = 5,   // flow: one input slot reset to the sentinel (same payload as FlowConnect)
    FlowDelete     = 6,   // flow: DeleteFlowNode (target = victim; node fields + severed inbound refs)
    EntityCreate   = 7,   // scene: ED6 click-to-place spawn (target = the new entity's view index,
                          // always the END of the drawable view; payload = meshPtr + xAfter + mAfter)
};

// Full TransformC snapshot (9 floats, no padding). Captured bitwise; undo/redo re-applies it as an
// ABSOLUTE SET through ApplyTransformEdit, so restoration is float-exact.
struct XformState {
    float px = 0, py = 0, pz = 0;   // position
    float ex = 0, ey = 0, ez = 0;   // eulerRadians
    float sx = 1, sy = 1, sz = 1;   // scale
};

// MaterialC's EDITABLE fields (metallic/roughness/baseColor — the MaterialEdit surface). baseColor
// is the opaque pointer as a 64-bit value (see the header note). 4+4+8 = 16 bytes, no padding.
struct MatState {
    float    metallic = 0.0f;
    float    roughness = 0.5f;
    uint64_t baseColor = 0;
};

// One severed inbound reference recorded by a FlowDelete (node `node`'s input `slot` pointed at the
// victim and was cut to the sentinel by DeleteFlowNode's remap).
struct FlowCutRef {
    uint32_t node = 0;
    uint32_t slot = 0;
};
inline constexpr uint32_t kFlowMaxCutRefs = 12;   // editor graphs are tiny; see the header note

// The flat POD command: {kind, target, BEFORE payload, AFTER payload} per op family. Only the
// fields of the tagged kind are meaningful (the rest stay zero-initialized so serialization and
// bit-compares are stable).
struct EditCommand {
    uint32_t kind = 0;      // EditCmdKind
    int32_t  target = 0;    // scene: view-order entity index; flow: NodeId (new/to/victim)
    // --- Transform payload ---
    XformState xBefore{}, xAfter{};
    // --- Material payload ---
    MatState mBefore{}, mAfter{};
    // --- Flow payloads ---
    uint32_t flowKind = 0;       // FlowAdd/FlowDelete: the node kind
    int32_t  flowConstArg = 0;   // FlowAdd/FlowDelete: the node constArg (flow::Reg)
    uint32_t flowSlot = 0;       // FlowConnect/Disconnect: the input slot (0..2)
    uint32_t flowRefBefore = 0;  // FlowConnect/Disconnect: the slot's ref before the op
    uint32_t flowRefAfter = 0;   // FlowConnect/Disconnect: the slot's ref after the op
    uint32_t flowA = 0, flowB = 0, flowC = 0;      // FlowDelete: the victim's original input refs
    uint32_t flowCutCount = 0;                     // FlowDelete: severed inbound reference count
    FlowCutRef flowCut[kFlowMaxCutRefs] = {};      // FlowDelete: the severed references
    // --- EntityCreate payload (ED6; appended — earlier kinds' layout/serialization untouched) ---
    uint64_t meshPtr = 0;   // the spawned entity's scene::Mesh* as an opaque 64-bit value (the
                            // MatState::baseColor discipline: never dereferenced; serialized as a
                            // registered NAME under the named encoding). Spawn transform/material
                            // ride in xAfter/mAfter (spawns carry no normal map — DefaultSpawnMaterial).
};

// The recorded history: commands[0..cursor) are APPLIED; [cursor..size) is the redo tail.
struct EditHistory {
    std::vector<EditCommand> commands;
    std::size_t cursor = 0;
};

// Record a command: truncate the redo tail (the standard branch-kill semantics — pinned by the
// interleave test), append, advance the cursor.
inline void Record(EditHistory& h, const EditCommand& c) {
    h.commands.resize(h.cursor);
    h.commands.push_back(c);
    h.cursor = h.commands.size();
}

// The models commands apply against. Each command family names ONE target pointer; a command whose
// target is null is a safe no-op (returns false). New op families enroll by adding a pointer here.
struct EditTargets {
    ecs::Registry* registry = nullptr;   // Transform/Material commands
    flow::Graph*   flowGraph = nullptr;  // Flow* commands
};

// --- Payload capture helpers ----------------------------------------------------------------------

inline XformState CaptureXform(const scene::Transform& t) {
    XformState s;
    s.px = t.position.x;     s.py = t.position.y;     s.pz = t.position.z;
    s.ex = t.eulerRadians.x; s.ey = t.eulerRadians.y; s.ez = t.eulerRadians.z;
    s.sx = t.scale.x;        s.sy = t.scale.y;        s.sz = t.scale.z;
    return s;
}
inline MatState CaptureMat(const scene::MaterialC& m) {
    MatState s;
    s.metallic = m.metallic;
    s.roughness = m.roughness;
    s.baseColor = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(m.base));
    return s;
}
// Bitwise payload equality (both structs are padding-free by construction; a bitwise compare is the
// conservative "did the op actually change anything" test — NaN/-0.0 safe).
inline bool BitEqual(const XformState& a, const XformState& b) {
    return std::memcmp(&a, &b, sizeof(XformState)) == 0;
}
inline bool BitEqual(const MatState& a, const MatState& b) {
    return std::memcmp(&a, &b, sizeof(MatState)) == 0;
}

// --- Command application (the undo/redo core) ------------------------------------------------------

// Apply one command against the targets: forward=true applies the AFTER payload (redo/replay),
// forward=false the BEFORE payload (undo). Scene payloads re-apply through the SAME pure ops as
// ABSOLUTE SETS of every captured field (fields the original edit left untouched carry identical
// before/after values, so the set is exact). Returns false when the command's target model is
// missing or the flow op reports a malformed target (never mutates on false for flow; the scene ops
// are safe no-ops by contract).
inline bool ApplyCommand(const EditCommand& c, const EditTargets& t, bool forward) {
    switch (static_cast<EditCmdKind>(c.kind)) {
        case EditCmdKind::Transform: {
            if (!t.registry) return false;
            const XformState& s = forward ? c.xAfter : c.xBefore;
            TransformEdit e;
            e.setPosition = true; e.position = {s.px, s.py, s.pz};
            e.setEuler = true;    e.euler    = {s.ex, s.ey, s.ez};
            e.setScale = true;    e.scale    = {s.sx, s.sy, s.sz};
            ApplyTransformEdit(*t.registry, c.target, e);
            return true;
        }
        case EditCmdKind::Material: {
            if (!t.registry) return false;
            const MatState& s = forward ? c.mAfter : c.mBefore;
            MaterialEdit e;
            e.setMetallic = true;  e.metallic = s.metallic;
            e.setRoughness = true; e.roughness = s.roughness;
            e.setBaseColor = true;
            e.baseColor = reinterpret_cast<rhi::ITexture*>(static_cast<uintptr_t>(s.baseColor));
            ApplyMaterialEdit(*t.registry, c.target, e);
            return true;
        }
        case EditCmdKind::FlowAdd: {
            if (!t.flowGraph) return false;
            if (forward) {
                // Re-appends at the same id given the same graph state (the append-only contract).
                const flow::NodeId id =
                    AddFlowNode(*t.flowGraph, c.flowKind, static_cast<flow::Reg>(c.flowConstArg));
                return id == static_cast<flow::NodeId>(c.target);
            }
            // Undo: the added node is the LAST node again (LIFO undo — every later command that
            // referenced it has already been undone), so DeleteFlowNode's remap is the identity on
            // the survivors and the pre-add graph is restored bit-exactly.
            return DeleteFlowNode(*t.flowGraph, static_cast<flow::NodeId>(c.target));
        }
        case EditCmdKind::FlowConnect:
        case EditCmdKind::FlowDisconnect: {
            if (!t.flowGraph) return false;
            // ConnectFlow SETS the slot, so restoring the recorded ref (which may be the sentinel)
            // is exact in both directions; Disconnect is just a Connect whose after-ref is the
            // sentinel (`to`'s own id).
            const uint32_t ref = forward ? c.flowRefAfter : c.flowRefBefore;
            return ConnectFlow(*t.flowGraph, static_cast<flow::NodeId>(ref),
                               static_cast<flow::NodeId>(c.target), c.flowSlot);
        }
        case EditCmdKind::FlowDelete: {
            if (!t.flowGraph) return false;
            const flow::NodeId v = static_cast<flow::NodeId>(c.target);
            if (forward) return DeleteFlowNode(*t.flowGraph, v);
            // Undo: re-insert the victim at its original index, shift the survivors' references
            // back up (the exact inverse of DeleteFlowNode's remap — a post-delete ref r >= v was
            // originally r + 1; the "no edge" self-sentinel shifts WITH its node so it stays a
            // self-sentinel), then restore the severed inbound references from the recorded cut
            // list. Well-formed graphs round-trip bit-exactly (an out-of-range ref DeleteFlowNode
            // clamped to the sentinel is restored AS the sentinel — documented in flow_edit_ops.h).
            flow::Graph& g = *t.flowGraph;
            if (static_cast<std::size_t>(v) > g.nodes.size()) return false;
            flow::Graph out;
            out.nodes.reserve(g.nodes.size() + 1);
            for (std::size_t i = 0; i <= g.nodes.size(); ++i) {
                if (i == static_cast<std::size_t>(v)) {
                    flow::Node nd;
                    nd.kind = c.flowKind;
                    nd.a = c.flowA; nd.b = c.flowB; nd.c = c.flowC;
                    nd.constArg = static_cast<flow::Reg>(c.flowConstArg);
                    out.nodes.push_back(nd);
                    continue;
                }
                const std::size_t src = (i > static_cast<std::size_t>(v)) ? i - 1 : i;
                flow::Node nd = g.nodes[src];
                auto unmap = [&](flow::NodeId r) -> flow::NodeId {
                    return (r >= v) ? static_cast<flow::NodeId>(r + 1) : r;
                };
                nd.a = unmap(nd.a); nd.b = unmap(nd.b); nd.c = unmap(nd.c);
                out.nodes.push_back(nd);
            }
            const uint32_t cuts = (c.flowCutCount <= kFlowMaxCutRefs) ? c.flowCutCount
                                                                      : kFlowMaxCutRefs;
            for (uint32_t k = 0; k < cuts; ++k) {
                const FlowCutRef& cr = c.flowCut[k];
                if (static_cast<std::size_t>(cr.node) >= out.nodes.size() || cr.slot > 2u) continue;
                flow::Node& nd = out.nodes[static_cast<std::size_t>(cr.node)];
                if (cr.slot == 0u) nd.a = v;
                else if (cr.slot == 1u) nd.b = v;
                else nd.c = v;
            }
            g = out;
            return true;
        }
        case EditCmdKind::EntityCreate: {
            if (!t.registry) return false;
            if (forward) {
                // Redo/replay: re-create the entity from the recorded payload. It appends at the
                // END of the drawable view, which under LIFO redo is exactly the recorded target
                // index again (mirrors FlowAdd's re-append contract).
                scene::Transform tr;
                tr.position     = {c.xAfter.px, c.xAfter.py, c.xAfter.pz};
                tr.eulerRadians = {c.xAfter.ex, c.xAfter.ey, c.xAfter.ez};
                tr.scale        = {c.xAfter.sx, c.xAfter.sy, c.xAfter.sz};
                scene::MaterialC m;
                m.base = reinterpret_cast<rhi::ITexture*>(
                    static_cast<uintptr_t>(c.mAfter.baseColor));
                m.normal = nullptr;   // ED6 spawns carry no normal map (DefaultSpawnMaterial)
                m.metallic = c.mAfter.metallic;
                m.roughness = c.mAfter.roughness;
                const int idx = ApplyCreateEntityRaw(
                    *t.registry,
                    reinterpret_cast<scene::Mesh*>(static_cast<uintptr_t>(c.meshPtr)), tr, m);
                return idx == c.target;
            }
            // Undo: the spawned entity is the LAST drawable again (LIFO — every later command was
            // undone first); destroying the last drawable pops each pool's last dense slot without
            // a swap, so the surviving view order (and every earlier command's view-index target)
            // is restored bit-identically. The guard inside refuses any non-last index.
            return ApplyDestroyLastEntity(*t.registry, c.target);
        }
        default:
            return false;
    }
}

// Undo the most recent applied command (applies its BEFORE payload through the same ops). Returns
// false with the history untouched when there is nothing to undo or the target model is missing.
inline bool Undo(EditHistory& h, const EditTargets& t) {
    if (h.cursor == 0) return false;
    if (!ApplyCommand(h.commands[h.cursor - 1], t, /*forward=*/false)) return false;
    --h.cursor;
    return true;
}

// Redo the next undone command (re-applies its AFTER payload). Returns false with the history
// untouched when there is no redo tail or the target model is missing.
inline bool Redo(EditHistory& h, const EditTargets& t) {
    if (h.cursor >= h.commands.size()) return false;
    if (!ApplyCommand(h.commands[h.cursor], t, /*forward=*/true)) return false;
    ++h.cursor;
    return true;
}

// --- Recorded wrappers (capture BEFORE -> raw op -> capture AFTER -> Record) -----------------------

// The recorded twin of ApplyTransformEdit. Captures the full TransformC before/after; records only
// when the op actually changed the component bitwise (a no-op edit records nothing). The raw op's
// safe-no-op contract on a bad index is preserved (nothing recorded).
inline void RecordedApplyTransformEdit(EditHistory& h, ecs::Registry& registry, int entity,
                                       const TransformEdit& edit) {
    ecs::Entity e = EntityAtViewIndex(registry, entity);
    if (e == ecs::kNullEntity || !registry.has<scene::TransformC>(e)) {
        ApplyTransformEdit(registry, entity, edit);   // the raw op's safe no-op path
        return;
    }
    EditCommand c;
    c.kind = static_cast<uint32_t>(EditCmdKind::Transform);
    c.target = entity;
    c.xBefore = CaptureXform(registry.get<scene::TransformC>(e).t);
    ApplyTransformEdit(registry, entity, edit);
    c.xAfter = CaptureXform(registry.get<scene::TransformC>(e).t);
    if (!BitEqual(c.xBefore, c.xAfter)) Record(h, c);
}

// The recorded twin of ApplyMaterialEdit (same discipline).
inline void RecordedApplyMaterialEdit(EditHistory& h, ecs::Registry& registry, int entity,
                                      const MaterialEdit& edit) {
    ecs::Entity e = EntityAtViewIndex(registry, entity);
    if (e == ecs::kNullEntity || !registry.has<scene::MaterialC>(e)) {
        ApplyMaterialEdit(registry, entity, edit);
        return;
    }
    EditCommand c;
    c.kind = static_cast<uint32_t>(EditCmdKind::Material);
    c.target = entity;
    c.mBefore = CaptureMat(registry.get<scene::MaterialC>(e));
    ApplyMaterialEdit(registry, entity, edit);
    c.mAfter = CaptureMat(registry.get<scene::MaterialC>(e));
    if (!BitEqual(c.mBefore, c.mAfter)) Record(h, c);
}

// Record a gizmo-style DIRECT transform manipulation as one command: the caller captured the
// TransformC before the manipulation (e.g. at gizmo grab) and calls this after it (e.g. at release).
// Records nothing when the state did not change bitwise. This is how a continuous drag becomes ONE
// undo step (the inspector's per-commit wrappers above are the per-op route).
inline void RecordTransformState(EditHistory& h, int entity, const XformState& before,
                                 const XformState& after) {
    if (BitEqual(before, after)) return;
    EditCommand c;
    c.kind = static_cast<uint32_t>(EditCmdKind::Transform);
    c.target = entity;
    c.xBefore = before;
    c.xAfter = after;
    Record(h, c);
}

// The recorded twin of ApplyCreateEntity (Slice ED6 — the asset browser's click-to-place). Runs
// the raw op, then captures the CREATED components as the AFTER payload (there is no BEFORE — the
// entity did not exist; undo is the destroy of the appended entity). An unknown mesh name is the
// raw op's safe no-op (-1) and records nothing.
inline int RecordedApplyCreateEntity(EditHistory& h, ecs::Registry& registry,
                                     const scene::SceneResources& resources,
                                     const std::string& meshName) {
    const int index = ApplyCreateEntity(registry, resources, meshName);
    if (index < 0) return index;
    ecs::Entity e = EntityAtViewIndex(registry, index);
    EditCommand c;
    c.kind = static_cast<uint32_t>(EditCmdKind::EntityCreate);
    c.target = index;
    c.xAfter = CaptureXform(registry.get<scene::TransformC>(e).t);
    c.mAfter = CaptureMat(registry.get<scene::MaterialC>(e));
    c.meshPtr =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(registry.get<scene::MeshC>(e).mesh));
    Record(h, c);
    return index;
}

// The recorded twin of AddFlowNode.
inline flow::NodeId RecordedAddFlowNode(EditHistory& h, flow::Graph& g, uint32_t kind,
                                        flow::Reg constArg = 0) {
    const flow::NodeId id = AddFlowNode(g, kind, constArg);
    EditCommand c;
    c.kind = static_cast<uint32_t>(EditCmdKind::FlowAdd);
    c.target = static_cast<int32_t>(id);
    c.flowKind = kind;
    c.flowConstArg = static_cast<int32_t>(constArg);
    Record(h, c);
    return id;
}

// The recorded twin of ConnectFlow. Records only a successful, state-changing rewire.
inline bool RecordedConnectFlow(EditHistory& h, flow::Graph& g, flow::NodeId from, flow::NodeId to,
                                uint32_t slot) {
    if (static_cast<std::size_t>(to) >= g.nodes.size() || slot > 2u) return false;
    const flow::Node& nd = g.nodes[static_cast<std::size_t>(to)];
    const flow::NodeId before = (slot == 0u) ? nd.a : (slot == 1u) ? nd.b : nd.c;
    if (!ConnectFlow(g, from, to, slot)) return false;
    if (before == from) return true;   // bitwise no-op: nothing to record
    EditCommand c;
    c.kind = static_cast<uint32_t>(EditCmdKind::FlowConnect);
    c.target = static_cast<int32_t>(to);
    c.flowSlot = slot;
    c.flowRefBefore = before;
    c.flowRefAfter = from;
    Record(h, c);
    return true;
}

// The recorded twin of DisconnectFlow (a Connect whose after-ref is the "no edge" sentinel).
inline bool RecordedDisconnectFlow(EditHistory& h, flow::Graph& g, flow::NodeId to, uint32_t slot) {
    if (static_cast<std::size_t>(to) >= g.nodes.size() || slot > 2u) return false;
    const flow::Node& nd = g.nodes[static_cast<std::size_t>(to)];
    const flow::NodeId before = (slot == 0u) ? nd.a : (slot == 1u) ? nd.b : nd.c;
    if (!DisconnectFlow(g, to, slot)) return false;
    if (before == to) return true;     // was already the sentinel
    EditCommand c;
    c.kind = static_cast<uint32_t>(EditCmdKind::FlowDisconnect);
    c.target = static_cast<int32_t>(to);
    c.flowSlot = slot;
    c.flowRefBefore = before;
    c.flowRefAfter = to;
    Record(h, c);
    return true;
}

// The recorded twin of DeleteFlowNode. Captures the victim node + every severed inbound reference
// so undo can rebuild the exact pre-delete graph. When the inbound references exceed
// kFlowMaxCutRefs the delete still happens but the HISTORY IS CLEARED (the documented deterministic
// invalidation — see the header note); the return value mirrors DeleteFlowNode's.
inline bool RecordedDeleteFlowNode(EditHistory& h, flow::Graph& g, flow::NodeId victim) {
    if (static_cast<std::size_t>(victim) >= g.nodes.size()) return false;
    EditCommand c;
    c.kind = static_cast<uint32_t>(EditCmdKind::FlowDelete);
    c.target = static_cast<int32_t>(victim);
    const flow::Node& vn = g.nodes[static_cast<std::size_t>(victim)];
    c.flowKind = vn.kind;
    c.flowConstArg = static_cast<int32_t>(vn.constArg);
    c.flowA = vn.a; c.flowB = vn.b; c.flowC = vn.c;
    bool overflow = false;
    for (std::size_t i = 0; i < g.nodes.size() && !overflow; ++i) {
        if (static_cast<flow::NodeId>(i) == victim) continue;
        const flow::Node& nd = g.nodes[i];
        const flow::NodeId refs[3] = {nd.a, nd.b, nd.c};
        for (uint32_t slot = 0; slot < 3u; ++slot) {
            if (refs[slot] != victim) continue;
            if (c.flowCutCount >= kFlowMaxCutRefs) { overflow = true; break; }
            c.flowCut[c.flowCutCount].node = static_cast<uint32_t>(i);
            c.flowCut[c.flowCutCount].slot = slot;
            ++c.flowCutCount;
        }
    }
    if (!DeleteFlowNode(g, victim)) return false;
    if (overflow) {
        h.commands.clear();
        h.cursor = 0;
        return true;
    }
    Record(h, c);
    return true;
}

// --- Serialization (hand-LE bytes) + digest + replay -----------------------------------------------

inline constexpr uint32_t kHistoryMagic = 0x48454648u;   // 'HFEH' little-endian ("HFEH")
inline constexpr uint32_t kHistoryVersion = 1u;
// Header flag bit0: material baseColor payloads are encoded as REGISTERED NAMES (length-prefixed,
// resolved through a SceneResources) instead of raw 64-bit pointer values — the process-portable
// artifact encoding. All other flag bits must be zero (a file with unknown bits is rejected).
inline constexpr uint32_t kHistoryFlagNamedTextures = 1u << 0;

namespace detail {
inline void PutU32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(static_cast<uint8_t>(v));
    b.push_back(static_cast<uint8_t>(v >> 8));
    b.push_back(static_cast<uint8_t>(v >> 16));
    b.push_back(static_cast<uint8_t>(v >> 24));
}
inline void PutU64(std::vector<uint8_t>& b, uint64_t v) {
    PutU32(b, static_cast<uint32_t>(v));
    PutU32(b, static_cast<uint32_t>(v >> 32));
}
inline void PutF32(std::vector<uint8_t>& b, float f) {
    uint32_t v;
    std::memcpy(&v, &f, sizeof(v));   // the float's exact bit pattern (no text round-trip)
    PutU32(b, v);
}
inline void PutXform(std::vector<uint8_t>& b, const XformState& s) {
    PutF32(b, s.px); PutF32(b, s.py); PutF32(b, s.pz);
    PutF32(b, s.ex); PutF32(b, s.ey); PutF32(b, s.ez);
    PutF32(b, s.sx); PutF32(b, s.sy); PutF32(b, s.sz);
}
// Material payload writer: with a resources table the baseColor pointer is reversed to its
// registered NAME (u32 length + bytes; empty for null/unregistered — matching DumpScene's mapping),
// making the bytes pointer-free; without one the raw 64-bit value is written (in-process artifact).
inline void PutMat(std::vector<uint8_t>& b, const MatState& s, const scene::SceneResources* res) {
    PutF32(b, s.metallic);
    PutF32(b, s.roughness);
    if (res) {
        const std::string name = res->NameOfTexture(
            reinterpret_cast<const rhi::ITexture*>(static_cast<uintptr_t>(s.baseColor)));
        PutU32(b, static_cast<uint32_t>(name.size()));
        for (char ch : name) b.push_back(static_cast<uint8_t>(ch));
    } else {
        PutU64(b, s.baseColor);
    }
}
// Mesh-pointer writer (ED6 EntityCreate; the PutMat baseColor discipline): with a resources table
// the mesh pointer is reversed to its registered NAME (u32 length + bytes; empty for
// null/unregistered), pointer-free; without one the raw 64-bit value is written.
inline void PutMeshPtr(std::vector<uint8_t>& b, uint64_t meshPtr,
                       const scene::SceneResources* res) {
    if (res) {
        const std::string name = res->NameOfMesh(
            reinterpret_cast<const scene::Mesh*>(static_cast<uintptr_t>(meshPtr)));
        PutU32(b, static_cast<uint32_t>(name.size()));
        for (char ch : name) b.push_back(static_cast<uint8_t>(ch));
    } else {
        PutU64(b, meshPtr);
    }
}
// Bounds-checked LE readers: each returns false on underrun (the cursor is left unspecified).
struct Reader {
    const uint8_t* p = nullptr;
    std::size_t n = 0;
    std::size_t at = 0;
    bool U32(uint32_t& v) {
        if (at + 4 > n) return false;
        v = static_cast<uint32_t>(p[at]) | (static_cast<uint32_t>(p[at + 1]) << 8) |
            (static_cast<uint32_t>(p[at + 2]) << 16) | (static_cast<uint32_t>(p[at + 3]) << 24);
        at += 4;
        return true;
    }
    bool U64(uint64_t& v) {
        uint32_t lo, hi;
        if (!U32(lo) || !U32(hi)) return false;
        v = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
        return true;
    }
    bool F32(float& f) {
        uint32_t v;
        if (!U32(v)) return false;
        std::memcpy(&f, &v, sizeof(f));
        return true;
    }
    bool Xform(XformState& s) {
        return F32(s.px) && F32(s.py) && F32(s.pz) && F32(s.ex) && F32(s.ey) && F32(s.ez) &&
               F32(s.sx) && F32(s.sy) && F32(s.sz);
    }
    // Material payload reader (the PutMat mirror): named encoding resolves the name back to the
    // caller's registered pointer (empty name -> null; an UNKNOWN name is rejected — the artifact
    // references a resource this process did not register, so a silent null would corrupt replay).
    bool Mat(MatState& s, const scene::SceneResources* res, bool named) {
        if (!F32(s.metallic) || !F32(s.roughness)) return false;
        if (!named) return U64(s.baseColor);
        uint32_t len = 0;
        if (!U32(len) || at + len > n) return false;
        std::string name(reinterpret_cast<const char*>(p + at), len);
        at += len;
        if (name.empty()) { s.baseColor = 0; return true; }
        if (!res) return false;
        rhi::ITexture* tex = res->FindTexture(name);
        if (!tex) return false;
        s.baseColor = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tex));
        return true;
    }
    // Mesh-pointer reader (the PutMeshPtr mirror): named encoding resolves the name back to the
    // caller's registered mesh pointer (empty name -> null; an unknown name is rejected — the
    // Mat reader's discipline).
    bool MeshPtr(uint64_t& v, const scene::SceneResources* res, bool named) {
        if (!named) return U64(v);
        uint32_t len = 0;
        if (!U32(len) || at + len > n) return false;
        std::string name(reinterpret_cast<const char*>(p + at), len);
        at += len;
        if (name.empty()) { v = 0; return true; }
        if (!res) return false;
        scene::Mesh* mesh = res->FindMesh(name);
        if (!mesh) return false;
        v = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mesh));
        return true;
    }
};
}  // namespace detail

// Serialize the WHOLE history (commands + cursor) to hand-LE bytes: the edit session as a portable,
// digestable, replayable artifact. Per-kind field lists only (no struct memcpy -> no padding bytes,
// no compiler-layout dependence): the bytes are identical for identical histories on every
// platform/compiler, so DigestHistory is a cross-platform pin. With `res` the material baseColor
// payloads are written as registered NAMES (kHistoryFlagNamedTextures) — the bytes are then
// POINTER-FREE, so the same logical session serializes identically in any process (the truly
// portable artifact); without it the raw pointer values are written (in-process only).
inline std::vector<uint8_t> SerializeHistory(const EditHistory& h,
                                             const scene::SceneResources* res = nullptr) {
    std::vector<uint8_t> b;
    detail::PutU32(b, kHistoryMagic);
    detail::PutU32(b, kHistoryVersion);
    detail::PutU32(b, res ? kHistoryFlagNamedTextures : 0u);
    detail::PutU64(b, static_cast<uint64_t>(h.commands.size()));
    detail::PutU64(b, static_cast<uint64_t>(h.cursor));
    for (const EditCommand& c : h.commands) {
        detail::PutU32(b, c.kind);
        detail::PutU32(b, static_cast<uint32_t>(c.target));
        switch (static_cast<EditCmdKind>(c.kind)) {
            case EditCmdKind::Transform:
                detail::PutXform(b, c.xBefore);
                detail::PutXform(b, c.xAfter);
                break;
            case EditCmdKind::Material:
                detail::PutMat(b, c.mBefore, res);
                detail::PutMat(b, c.mAfter, res);
                break;
            case EditCmdKind::FlowAdd:
                detail::PutU32(b, c.flowKind);
                detail::PutU32(b, static_cast<uint32_t>(c.flowConstArg));
                break;
            case EditCmdKind::FlowConnect:
            case EditCmdKind::FlowDisconnect:
                detail::PutU32(b, c.flowSlot);
                detail::PutU32(b, c.flowRefBefore);
                detail::PutU32(b, c.flowRefAfter);
                break;
            case EditCmdKind::FlowDelete:
                detail::PutU32(b, c.flowKind);
                detail::PutU32(b, static_cast<uint32_t>(c.flowConstArg));
                detail::PutU32(b, c.flowA);
                detail::PutU32(b, c.flowB);
                detail::PutU32(b, c.flowC);
                detail::PutU32(b, c.flowCutCount);
                for (uint32_t k = 0; k < c.flowCutCount && k < kFlowMaxCutRefs; ++k) {
                    detail::PutU32(b, c.flowCut[k].node);
                    detail::PutU32(b, c.flowCut[k].slot);
                }
                break;
            case EditCmdKind::EntityCreate:
                detail::PutMeshPtr(b, c.meshPtr, res);
                detail::PutXform(b, c.xAfter);
                detail::PutMat(b, c.mAfter, res);
                break;
            default:
                break;   // unknown kinds carry no payload (forward-compat: they never serialize)
        }
    }
    return b;
}

// Deserialize a SerializeHistory artifact. Strict: the magic, version, flags, every kind tag, the
// cursor bound, the cut-ref cap, and the EXACT byte length must all validate, or false is returned
// and `out` is left empty. A named-texture artifact (kHistoryFlagNamedTextures) REQUIRES `res` —
// its material names are re-bound to THIS process's registered pointers (an unknown name is
// rejected). A deserialize->serialize round-trip under the same encoding is byte-identical (same
// digest).
inline bool DeserializeHistory(const uint8_t* data, std::size_t size, EditHistory& out,
                               const scene::SceneResources* res = nullptr) {
    out.commands.clear();
    out.cursor = 0;
    detail::Reader r{data, size, 0};
    uint32_t magic = 0, version = 0, flags = 0;
    uint64_t count = 0, cursor = 0;
    if (!r.U32(magic) || magic != kHistoryMagic) return false;
    if (!r.U32(version) || version != kHistoryVersion) return false;
    if (!r.U32(flags) || (flags & ~kHistoryFlagNamedTextures) != 0) return false;
    const bool named = (flags & kHistoryFlagNamedTextures) != 0;
    if (named && !res) return false;   // a portable artifact needs the name->pointer table
    if (!r.U64(count) || !r.U64(cursor) || cursor > count) return false;
    std::vector<EditCommand> cmds;
    cmds.reserve(static_cast<std::size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        EditCommand c;
        uint32_t target = 0;
        if (!r.U32(c.kind) || !r.U32(target)) return false;
        c.target = static_cast<int32_t>(target);
        switch (static_cast<EditCmdKind>(c.kind)) {
            case EditCmdKind::Transform:
                if (!r.Xform(c.xBefore) || !r.Xform(c.xAfter)) return false;
                break;
            case EditCmdKind::Material:
                if (!r.Mat(c.mBefore, res, named) || !r.Mat(c.mAfter, res, named)) return false;
                break;
            case EditCmdKind::FlowAdd: {
                uint32_t ca = 0;
                if (!r.U32(c.flowKind) || !r.U32(ca)) return false;
                c.flowConstArg = static_cast<int32_t>(ca);
                break;
            }
            case EditCmdKind::FlowConnect:
            case EditCmdKind::FlowDisconnect:
                if (!r.U32(c.flowSlot) || !r.U32(c.flowRefBefore) || !r.U32(c.flowRefAfter))
                    return false;
                break;
            case EditCmdKind::FlowDelete: {
                uint32_t ca = 0;
                if (!r.U32(c.flowKind) || !r.U32(ca) || !r.U32(c.flowA) || !r.U32(c.flowB) ||
                    !r.U32(c.flowC) || !r.U32(c.flowCutCount))
                    return false;
                c.flowConstArg = static_cast<int32_t>(ca);
                if (c.flowCutCount > kFlowMaxCutRefs) return false;
                for (uint32_t k = 0; k < c.flowCutCount; ++k)
                    if (!r.U32(c.flowCut[k].node) || !r.U32(c.flowCut[k].slot)) return false;
                break;
            }
            case EditCmdKind::EntityCreate:
                if (!r.MeshPtr(c.meshPtr, res, named) || !r.Xform(c.xAfter) ||
                    !r.Mat(c.mAfter, res, named))
                    return false;
                break;
            default:
                return false;   // an unknown kind tag is a malformed artifact
        }
        cmds.push_back(c);
    }
    if (r.at != size) return false;   // trailing bytes are a malformed artifact
    out.commands = std::move(cmds);
    out.cursor = static_cast<std::size_t>(cursor);
    return true;
}

// The history's deterministic fingerprint: FNV-1a-64 over the serialized bytes (the codebase's
// pinned-golden DigestBytes currency). Identical histories -> identical digest on every
// platform/compiler (the serialization is hand-LE, layout-free). With `res` (the named encoding)
// the digest is additionally PROCESS-STABLE — independent of texture pointer values.
inline uint64_t DigestHistory(const EditHistory& h, const scene::SceneResources* res = nullptr) {
    const std::vector<uint8_t> bytes = SerializeHistory(h, res);
    return net::DigestBytes(bytes.data(), bytes.size());
}

// Replay the APPLIED prefix (commands[0..cursor)) forward onto fresh targets: given the same
// starting state the session was recorded from, the replayed models finish bit-identical to the
// live session's (the DumpScene byte-compare proof). Returns false if any command fails to apply
// (missing target model / malformed flow command); commands before the failure remain applied.
inline bool ReplayHistory(const EditHistory& h, const EditTargets& t) {
    for (std::size_t i = 0; i < h.cursor; ++i)
        if (!ApplyCommand(h.commands[i], t, /*forward=*/true)) return false;
    return true;
}

}  // namespace hf::editor
