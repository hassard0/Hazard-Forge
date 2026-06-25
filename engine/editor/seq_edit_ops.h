#pragma once
// Hazard Forge — cinematic-sequencer TIMELINE EDIT OPERATIONS (pure CPU, ImGui-free, backend-free).
//
// Issue #25 (Cinematic Sequencer / timeline), the editor WRITE path: deterministic, programmatic
// keyframe-mutation ops the timeline editor's add/move/delete actions call. Mirrors flow_edit_ops.{h}
// (the node-graph editor's write path) but for a hf::seq::ScalarTrack instead of a flow::Graph. Each op
// MUTATES the track deterministically and PRESERVES THE INVARIANT seq.h's SampleScalar relies on: `times`
// stays STRICTLY ASCENDING (sorted, no duplicate times) with values.size() == times.size(). A keyframe
// authored at a time that already exists OVERWRITES that key's value (no duplicate time is ever created),
// keeping the binary-search sampler well-defined. AddKeyframe / MoveKeyframe keep the arrays sorted by
// insertion-at-the-right-slot; DeleteKeyframe erases a pair. No <algorithm> (hand integer loops, the seq.h
// discipline) so this compiles STANDALONE with clang exactly like seq.h / seq_editor_data.h.
//
// Header-only + SELF-CONTAINED (only seq/seq.h, which pulls <cstddef>/<cstdint>/<vector> + sim/fpx.h +
// net/session.h). NO ImGui / rhi / float / hash containers. The view-digest after an edit is deterministic +
// CHANGES (proven by the golden test).

#include "seq/seq.h"   // hf::seq::ScalarTrack / fx (Q16.16)

namespace hf::editor {

using hf::seq::fx;
using hf::seq::ScalarTrack;

// ---- AddKeyframe: insert (time, value) keeping `times` strictly ascending (the ScalarTrack invariant). --
// Hand integer scan for the first index whose time is >= `time` (the sorted insertion slot). If that slot's
// time EQUALS `time`, OVERWRITE its value (no duplicate-time key is ever created — the strictly-ascending
// invariant). Otherwise insert the pair before that slot. Returns the index the key now lives at. A
// well-formed track (parallel arrays) stays well-formed.
inline std::size_t AddKeyframe(ScalarTrack& tr, fx time, fx value) {
    std::size_t slot = tr.times.size();                 // default: append at the end
    for (std::size_t i = 0; i < tr.times.size(); ++i) {
        if (tr.times[i] == time) {                      // exact-time hit -> overwrite the value
            tr.values[i] = value;
            return i;
        }
        if (tr.times[i] > time) { slot = i; break; }    // first key after `time` -> insert before it
    }
    tr.times.insert(tr.times.begin() + static_cast<std::ptrdiff_t>(slot), time);
    tr.values.insert(tr.values.begin() + static_cast<std::ptrdiff_t>(slot), value);
    return slot;
}

// ---- DeleteKeyframe: erase the (time,value) pair at `keyIndex`. Returns false if out of range. ----------
// Keeps the arrays parallel + the remaining keys strictly ascending (erasing one element preserves order).
inline bool DeleteKeyframe(ScalarTrack& tr, std::size_t keyIndex) {
    if (keyIndex >= tr.times.size()) return false;
    tr.times.erase(tr.times.begin() + static_cast<std::ptrdiff_t>(keyIndex));
    tr.values.erase(tr.values.begin() + static_cast<std::ptrdiff_t>(keyIndex));
    return true;
}

// ---- MoveKeyframe: retime + revalue the key at `keyIndex` to (newTime, newValue), re-sorting if needed. -
// Returns false if `keyIndex` is out of range. Implemented as delete-then-add so the strictly-ascending
// invariant is RE-ESTABLISHED for free (AddKeyframe inserts at the correct slot, and OVERWRITES if newTime
// collides with another existing key — a drag onto a neighbor merges deterministically rather than creating
// a duplicate time). The returned index is the key's NEW position after re-sorting.
inline bool MoveKeyframe(ScalarTrack& tr, std::size_t keyIndex, fx newTime, fx newValue) {
    if (keyIndex >= tr.times.size()) return false;
    DeleteKeyframe(tr, keyIndex);                       // remove from the old slot
    AddKeyframe(tr, newTime, newValue);                 // re-insert at the sorted slot (or overwrite)
    return true;
}

}  // namespace hf::editor
