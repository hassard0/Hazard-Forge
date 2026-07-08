#pragma once
// Slice FK1 — WHAT-IF FORK REPLAY (counterfactual timelines), hf::replay.
//
// THE STRATEGIC MOAT (moat play #2 — the PUREST "UE5 literally cannot do this"): because the whole Hazard
// Forge simulation is BIT-EXACT deterministic (VD1-VD6 whole-world gameplay/physics lockstep from an input
// stream ALONE) and rollback/seek-capable (replay.h RP1-6: record -> encode -> Seek -> scrub -> verify),
// you can Seek a recorded replay to ANY tick, MUTATE one input, and RE-SIMULATE a COUNTERFACTUAL timeline
// that is ITSELF perfectly reproducible — a "what-if" replay. Fork the SAME recording at different ticks /
// with different mutations and you get a small TIMELINE TREE (the original + N counterfactuals), each
// branch bit-identical run-to-run + cross-platform, sharing a byte-identical prefix and diverging causally.
// UE5's float sim has NO reproducible re-derivation and NO inverse, so a bit-exact counterfactual is
// STRUCTURALLY impossible (the best it can do is store interpolated transforms) — it is disqualified from
// the category, not merely behind.
//
// THE SUBSTRATE IS ALREADY SHIPPED — FK1 only PACKAGES the fork, composing three FROZEN pieces READ-ONLY
// (this header adds NO field and edits NO existing function):
//   * replay.h RP4 Seek — reconstruct the EXACT world state at forkTick by restoring the nearest keyframe
//     and replaying the tail (Seek itself reuses net::CatchUp). ForkAt's rewind IS Seek, verbatim.
//   * net::CatchUp (NS6) — restore a snapshot + replay a command tail forward. The counterfactual re-sim
//     (from the forkTick state, over the MUTATED stream) IS CatchUp, verbatim — so both the original
//     timeline and every forked branch are produced by the SAME frozen replay body (no hand-rolled loop).
//   * verdict.h VD1-VD6 (ClonePeer / SimVerdictTick / SnapshotWorld / DigestSnapshot) — the WHOLE-WORLD
//     deterministic tick + its order[]-keyed fingerprint. One verdict tick is the CatchUp/Seek StepFn.
//
// THE BRIDGE (the one structural subtlety): the World that flows through replay's Seek/CatchUp must be
// COPYABLE, but verdict::VerdictWorld holds an ecs::Registry (unique_ptr pools) and is NON-COPYABLE. So the
// flowing "World" is the COPYABLE verdict::VerdictSnapshot (the VD4 heterogeneous snapshot), and the StepFn
// materializes a peer from it (verdict::ClonePeer — the determinism-faithful non-copyable clone), runs ONE
// SimVerdictTick, and snapshots back. This makes replay's generic <World,Input> machinery drive the whole
// gameplay+physics world with ZERO changes to replay.h / session.h / verdict.h.
//
// HONESTY (the boundaries, stated up front — see the report):
//   * The counterfactual is only as rich as the recorded COMMAND model: what is mutable is the verdict
//     Command stream {tick,kind,target,arg} (spawn/despawn/move/ability + the lowered sim impulse/angvel).
//     A "what-if" is expressed as INJECTING a command (append; ApplyCommands' fixed array order lets a
//     later same-tick command override) or REMOVING a tick's commands (RemoveInputsAt) — not by rewriting
//     engine constants or scene geometry.
//   * Each forked tick RE-CLONES the world from a snapshot (ClonePeer) because VerdictWorld is
//     non-copyable — correct + bit-exact, but heavier than a native in-place step (fine at slice scale).
//   * The "timeline tree" here is the original + 2 counterfactual branches (not arbitrary depth); ForkAt
//     is general (any tick, any mutation), so deeper trees compose, but FK1 pins a 2-branch tree.
//
// PURE CPU INTEGER (strict determinism tier). NO new render RHI, NO new shader, NO new compute. replay.h /
// session.h / verdict.h are #included READ-ONLY / BYTE-UNCHANGED; fork.h is a brand-new additive sibling.

#include <cstdint>
#include <cstdlib>     // std::strtoull (16-hex DigestSnapshot -> the uint64 digest currency)
#include <string>
#include <vector>

#include "replay/replay.h"   // read-only: RP4 Seek / SeekResult / Demo<World,Input> (which pulls net/session.h)
#include "net/session.h"     // read-only: NS6 CatchUp / JoinSnapshot / InputRing / DigestBytes
#include "game/verdict.h"    // read-only: the VD1-VD6 whole-world deterministic sim — VerdictSnapshot /
                             // ClonePeer / SimVerdictTick / SnapshotWorld / DigestSnapshot / VerdictParams /
                             // Command / BuildCanonicalReplay (the sim the fork re-derives)

namespace hf {
namespace replay {

// Alias the frozen gameplay/netcode namespaces locally (READ-ONLY compose).
namespace verdict = hf::game::verdict;

// The World that flows through replay's Seek/CatchUp is the COPYABLE VD4 snapshot; the Input is one verdict
// Command. (VerdictWorld itself is non-copyable — the StepFn below materializes a peer from the snapshot.)
using ForkWorld = verdict::VerdictSnapshot;
using ForkInput = verdict::Command;
using ForkDemo  = Demo<ForkWorld, ForkInput>;   // the replay demo the fork Seeks into

// ----- ForkHexToU64: parse verdict::DigestSnapshot's 16-hex string into the uint64 digest currency --------
// DigestSnapshot returns a lowercase 16-hex string (an exact bijection with uint64). We compare / fold
// digests as uint64 (compact, the FNV currency). A malformed/empty string parses to 0 deterministically
// (never a crash). Pure integer, no float.
inline uint64_t ForkHexToU64(const std::string& hex) {
    if (hex.empty()) return 0ull;
    return (uint64_t)std::strtoull(hex.c_str(), nullptr, 16);
}

// ----- The StepFn bridge: ONE verdict world tick over a COPYABLE snapshot (the Seek/CatchUp transition) ---
// Matches net::Advance / net::CatchUp's step signature step(World&, const std::vector<Input>&, uint32_t):
// materialize a peer from the snapshot (ClonePeer — the VD4 determinism-faithful non-copyable clone; hulls
// seeded from params), run ONE SimVerdictTick over this tick's commands, and snapshot back. `params` must
// outlive the call (it does — the fork holds it). Deterministic of (snap, cmds, tick, params) alone.
struct VerdictStep {
    const verdict::VerdictParams* params;
    void operator()(ForkWorld& snap, const std::vector<ForkInput>& cmds, uint32_t tick) const {
        verdict::VerdictWorld w = verdict::ClonePeer(snap, *params);  // the VD4 clone (NOT a copy)
        verdict::SimVerdictTick(w, *params, cmds, tick);              // the FROZEN VD5 whole-world tick, verbatim
        snap = verdict::SnapshotWorld(w);                            // snapshot back (copyable)
    }
};

// Same transition, but ALSO records the AFTER-tick digest into an external trace — so net::CatchUp produces
// a per-tick digest trace as a pure SIDE EFFECT of the frozen replay body (no re-implemented replay loop).
struct RecordingVerdictStep {
    const verdict::VerdictParams* params;
    std::vector<uint64_t>*        trace;
    void operator()(ForkWorld& snap, const std::vector<ForkInput>& cmds, uint32_t tick) const {
        verdict::VerdictWorld w = verdict::ClonePeer(snap, *params);
        verdict::SimVerdictTick(w, *params, cmds, tick);
        snap = verdict::SnapshotWorld(w);
        trace->push_back(ForkHexToU64(verdict::DigestSnapshot(snap)));  // the outcome digest AFTER tick `tick`
    }
};

// The DigestFn replay::Seek requires (uint64 over a world snapshot) — verdict::DigestSnapshot re-cast.
struct VerdictDigest {
    uint64_t operator()(const ForkWorld& snap) const { return ForkHexToU64(verdict::DigestSnapshot(snap)); }
};

// ----- FoldTimelineDigest: a whole-timeline fingerprint (FNV-1a-64 over the per-tick digest sequence) -----
// Folds ticks then each per-tick digest LSB-first (endianness-independent) into one uint64 — the pinnable
// identity of an ENTIRE timeline (two timelines are byte-equal iff this matches). Same FNV constants as the
// engine's other digest sites.
inline uint64_t FoldTimelineDigest(const std::vector<uint64_t>& digests, uint32_t ticks) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { for (int b = 0; b < 8; ++b) { h ^= (uint64_t)(uint8_t)(v >> (b * 8)); h *= 1099511628211ull; } };
    mix((uint64_t)ticks);
    for (std::size_t i = 0; i < digests.size(); ++i) mix(digests[i]);
    return h;
}

// ----- Timeline: a fully-realized timeline — its per-tick digest trace + final world + fingerprint --------
struct Timeline {
    uint32_t              ticks = 0;
    std::vector<uint64_t> digests;      // per-tick outcome digest AFTER each tick (length == ticks)
    ForkWorld             finalWorld;   // the VerdictSnapshot after `ticks` ticks
    uint64_t              fullDigest = 0;  // FoldTimelineDigest(digests, ticks) — the whole-timeline identity
};

// ----- SimulateTimeline: run the ORIGINAL (or a fresh) timeline from tick 0 over `stream` for `ticks` -----
// This IS net::CatchUp (NS6) seeded at tick 0 with a RECORDING step, so the per-tick digest trace + the
// final world fall out of the FROZEN replay body. Deterministic of (w0Snap, params, stream, ticks) alone.
inline Timeline SimulateTimeline(const ForkWorld& w0Snap, const verdict::VerdictParams& params,
                                 const std::vector<ForkInput>& stream, uint32_t ticks) {
    net::InputRing<ForkInput> ring;
    for (std::size_t i = 0; i < stream.size(); ++i) ring.AddInput(stream[i].tick, stream[i]);
    Timeline tl;
    tl.ticks = ticks;
    tl.digests.reserve((std::size_t)ticks);
    RecordingVerdictStep step{&params, &tl.digests};
    net::JoinSnapshot<ForkWorld> snap{0u, w0Snap};
    tl.finalWorld = net::CatchUp(snap, ticks, ring, step);   // the frozen NS6 replay, verbatim
    tl.fullDigest = FoldTimelineDigest(tl.digests, ticks);
    return tl;
}

// ----- BuildForkDemo: assemble a replay Demo<VerdictSnapshot,Command> the fork Seeks into ------------------
// Populates the DECODED Demo directly (we never serialize a VerdictSnapshot to bytes — Seek reads
// initial/keyframes/ring/header.tickCount+keyframeInterval only): the initial snapshot, the per-tick input
// ring (stream grouped by tick), and a keyframe table (a VerdictSnapshot AS OF tick T at T%interval==0, the
// RP3 seek substrate) so RP4 Seek genuinely restores the nearest keyframe rather than always falling back to
// tick 0. keyframe@0 == the initial world (captured BEFORE stepping tick 0 — the RP3 convention).
inline ForkDemo BuildForkDemo(const ForkWorld& w0Snap, const verdict::VerdictParams& params,
                              const std::vector<ForkInput>& stream, uint32_t ticks, uint32_t keyframeInterval) {
    ForkDemo demo;
    for (int i = 0; i < 8; ++i) demo.header.magic[i] = kDemoMagic[i];
    demo.header.version          = kDemoVersion;
    demo.header.tickCount        = ticks;
    demo.header.keyframeInterval = keyframeInterval;
    demo.initial = w0Snap;
    for (std::size_t i = 0; i < stream.size(); ++i) demo.ring.AddInput(stream[i].tick, stream[i]);

    // Walk the session one tick at a time, capturing a keyframe every `keyframeInterval` ticks BEFORE the step.
    verdict::VerdictWorld w = verdict::ClonePeer(w0Snap, params);
    for (uint32_t t = 0; t < ticks; ++t) {
        if (keyframeInterval > 0 && (t % keyframeInterval) == 0) {
            typename ForkDemo::DecodedKeyframe dk;
            dk.tick  = t;                          // world AS OF tick t (BEFORE stepping t)
            dk.world = verdict::SnapshotWorld(w);
            demo.keyframes.push_back(std::move(dk));
        }
        verdict::SimVerdictTick(w, params, demo.ring.At(t), t);
    }
    return demo;
}

// ----- ForkedTimeline: a rewound-and-diverged branch — the counterfactual state + its (mutable) stream ----
struct ForkedTimeline {
    uint32_t                forkTick = 0;            // where this branch rewound to (shares [0, forkTick])
    ForkWorld               baseWorld;               // world AS OF forkTick (reconstructed by RP4 Seek)
    std::vector<uint64_t>   prefixDigests;           // the original digests [0, forkTick) — the SHARED history
    std::vector<ForkInput>  stream;                  // the MUTABLE forked command stream (== original at fork)
    verdict::VerdictParams  params;                  // the constant scene knobs (copied)
    uint32_t                ticks = 0;               // total timeline length
    uint32_t                seekKeyframeTick = 0;    // RP4 bookkeeping: which keyframe Seek restored
    uint32_t                seekReplayedTicks = 0;   // RP4 bookkeeping: replay-tail length (seek cost)
    bool                    mutated = false;         // has any input been injected/removed?
    uint32_t                firstMutationTick = 0;   // the earliest mutated tick (== ticks when unmutated)
};

// ----- ForkAt: rewind a recorded demo to forkTick (RP4 Seek) + open a NEW divergeable timeline ------------
// (1) Clamp forkTick to the demo length. (2) Seek(demo, forkTick) — restore the nearest keyframe <= forkTick
// + replay the tail forward — to reconstruct the EXACT world state at forkTick (bit-identical to live-at-
// forkTick; this IS RP4, which itself reuses net::CatchUp). (3) Copy the original digests [0, forkTick) as
// the SHARED prefix, and seed the forked stream from the original (so an un-mutated fork == the original —
// the null fork). The branch is ready to MutateInput + ResimulateFork.
inline ForkedTimeline ForkAt(const ForkDemo& demo, const Timeline& original,
                             const verdict::VerdictParams& params, uint32_t forkTick,
                             const std::vector<ForkInput>& originalStream) {
    if (forkTick > demo.header.tickCount) forkTick = demo.header.tickCount;
    const VerdictStep   step{&params};
    const VerdictDigest digest{};
    const SeekResult<ForkWorld, ForkInput> sr = Seek(demo, forkTick, step, digest);  // RP4 Seek, verbatim

    ForkedTimeline fk;
    fk.forkTick          = forkTick;
    fk.baseWorld         = sr.world;
    fk.seekKeyframeTick  = sr.keyframeTick;
    fk.seekReplayedTicks = sr.replayedTicks;
    fk.prefixDigests.assign(original.digests.begin(),
                            original.digests.begin() + (std::ptrdiff_t)forkTick);
    fk.stream            = originalStream;   // start identical (the null fork re-derives the original)
    fk.params            = params;
    fk.ticks             = demo.header.tickCount;
    fk.firstMutationTick = fk.ticks;         // no mutation yet
    return fk;
}

// ----- MutateInput: inject a counterfactual command into the forked stream (the "what-if") ----------------
// The command fires at cmd.tick, which MUST be >= forkTick (a mutation cannot rewrite the SHARED history —
// the causal guard; a tick < forkTick is rejected, returns false). Injection APPENDS the command; because
// ApplyCommands applies a tick's commands in FIXED ARRAY ORDER, a later same-tick command overrides an
// earlier one, so "inject" also serves "replace/augment". Records the earliest mutation tick. Deterministic.
inline bool MutateInput(ForkedTimeline& fk, const ForkInput& cmd) {
    if (cmd.tick < fk.forkTick) return false;   // causal guard — cannot mutate the shared prefix
    fk.stream.push_back(cmd);
    fk.mutated = true;
    if (cmd.tick < fk.firstMutationTick) fk.firstMutationTick = cmd.tick;
    return true;
}

// ----- RemoveInputsAt: drop every command at `tick` (the REPLACE-style mutation helper) -------------------
// For "what if the player had done NOTHING (or something else) at tick T": RemoveInputsAt(fk, T) then
// optionally MutateInput a replacement. tick must be >= forkTick (the causal guard). Returns the count removed.
inline uint32_t RemoveInputsAt(ForkedTimeline& fk, uint32_t tick) {
    if (tick < fk.forkTick) return 0;
    uint32_t removed = 0;
    std::vector<ForkInput> keep;
    keep.reserve(fk.stream.size());
    for (std::size_t i = 0; i < fk.stream.size(); ++i) {
        if (fk.stream[i].tick == tick) ++removed;
        else keep.push_back(fk.stream[i]);
    }
    fk.stream.swap(keep);
    if (removed) { fk.mutated = true; if (tick < fk.firstMutationTick) fk.firstMutationTick = tick; }
    return removed;
}

// ----- ResimulateFork: re-simulate the counterfactual from the Seek'd forkTick state over the MUTATED stream
// The SHARED prefix [0, forkTick) is byte-identical to the original (copied at ForkAt) — we do NOT re-run it.
// The SUFFIX [forkTick, ticks) is produced by net::CatchUp (NS6), seeded with JoinSnapshot{forkTick,
// baseWorld} over the MUTATED ring — the frozen replay body, verbatim, recording the per-tick digest as it
// goes. Concatenated = the full counterfactual timeline. It is itself bit-exact + reproducible (two calls ==).
inline Timeline ResimulateFork(const ForkedTimeline& fk) {
    net::InputRing<ForkInput> ring;
    for (std::size_t i = 0; i < fk.stream.size(); ++i) ring.AddInput(fk.stream[i].tick, fk.stream[i]);

    Timeline tl;
    tl.ticks   = fk.ticks;
    tl.digests = fk.prefixDigests;   // [0, forkTick) — the SHARED history, byte-identical to the original
    tl.digests.reserve((std::size_t)fk.ticks);

    std::vector<uint64_t> suffix;
    suffix.reserve((std::size_t)(fk.ticks - fk.forkTick));
    RecordingVerdictStep step{&fk.params, &suffix};
    net::JoinSnapshot<ForkWorld> snap{fk.forkTick, fk.baseWorld};
    tl.finalWorld = net::CatchUp(snap, fk.ticks, ring, step);   // the counterfactual re-sim (NS6, verbatim)
    for (std::size_t i = 0; i < suffix.size(); ++i) tl.digests.push_back(suffix[i]);
    tl.fullDigest = FoldTimelineDigest(tl.digests, tl.ticks);
    return tl;
}

// ----- TimelineDiff: where two timelines start to differ (the causal divergence proof) --------------------
struct TimelineDiff {
    int      firstDivergence = -1;   // first tick where per-tick digests differ; -1 if identical over the common length
    bool     identical       = false;  // fully identical (same length + same fold + no per-tick divergence)
    uint32_t ticks           = 0;    // the common length compared
    uint64_t origFullDigest  = 0;
    uint64_t forkFullDigest  = 0;
};

// DiffTimelines: scan the two per-tick digest traces for the FIRST tick they differ (the causal
// divergence — for a mutation at tick M this is exactly M, never earlier, because [0, M) apply identical
// commands from an identical prefix). identical iff no per-tick divergence AND same length AND same fold.
inline TimelineDiff DiffTimelines(const Timeline& a, const Timeline& b) {
    TimelineDiff d;
    d.ticks          = a.ticks < b.ticks ? a.ticks : b.ticks;
    d.origFullDigest = a.fullDigest;
    d.forkFullDigest = b.fullDigest;
    d.firstDivergence = -1;
    for (uint32_t t = 0; t < d.ticks; ++t)
        if (a.digests[(std::size_t)t] != b.digests[(std::size_t)t]) { d.firstDivergence = (int)t; break; }
    d.identical = (d.firstDivergence == -1) && (a.ticks == b.ticks) && (a.fullDigest == b.fullDigest);
    return d;
}

// ----- Outcome: a compact, gameplay-observable summary of a timeline's FINAL world (does the story change?)
struct Outcome {
    uint32_t entityCount = 0;   // final live entity count (order.size())
    verdict::EntityId nextId = 1u;  // the monotonic id allocator (a spawn counterfactual moves it)
    int32_t  playerHealth = 0;  // the player's final Health (an ability counterfactual moves it)
    uint64_t finalDigest  = 0;  // DigestSnapshot(finalWorld) — the whole-world final fingerprint
};

// OutcomeOf: read the final-world outcome from a timeline (a pure read of the VD4 snapshot fields).
inline Outcome OutcomeOf(const Timeline& tl, verdict::EntityId player) {
    Outcome o;
    o.entityCount = (uint32_t)tl.finalWorld.order.size();
    o.nextId      = tl.finalWorld.nextId;
    for (std::size_t i = 0; i < tl.finalWorld.healths.size(); ++i)
        if (tl.finalWorld.healths[i].id == player) o.playerHealth = tl.finalWorld.healths[i].value.hp;
    o.finalDigest = ForkHexToU64(verdict::DigestSnapshot(tl.finalWorld));
    return o;
}

inline bool OutcomesEqual(const Outcome& a, const Outcome& b) {
    return a.entityCount == b.entityCount && a.nextId == b.nextId &&
           a.playerHealth == b.playerHealth && a.finalDigest == b.finalDigest;
}

// =================================================================================================
// THE CANONICAL FORK SCENARIO — the shared HEADLINE PROOF (the showcase + the test both build it).
// Record the FROZEN verdict::BuildCanonicalReplay match (the composed gameplay+physics scene DX5/DX6/AC1 use,
// so EVERY component type + the embedded sim participate in the digests). Then fork it at kFk1ForkTick with
// TWO different single-input mutations -> a 3-node timeline tree:
//   * ORIGINAL  — the recorded match, replayed.
//   * BRANCH A  — "what if the player had been shoved at the fork tick" (a kCmdImpulse -> the PHYSICS
//                 diverges: the player body + the stack settle differently).
//   * BRANCH B  — "what if the player had triggered an ability at the fork tick" (a kCmdAbility -> the
//                 GAMEPLAY diverges: the player's Health jumps).
// Both branches share ticks [0, kFk1ForkTick) byte-for-byte with the original + each other, diverge at
// EXACTLY kFk1ForkTick, and reach DIFFERENT final outcomes — each perfectly reproducible.
// =================================================================================================

inline constexpr uint32_t kFk1ForkTick          = 8u;   // fork at a keyframe boundary (RP4 restores the keyframe)
inline constexpr uint32_t kFk1KeyframeInterval  = 8u;   // demo keyframes at 0, 8, 16 (the seek substrate)

struct Fk1Scenario {
    verdict::VerdictParams  params;
    ForkWorld               w0Snap;
    std::vector<ForkInput>  originalStream;
    uint32_t                ticks  = 0;
    verdict::EntityId       player = verdict::kNoEntity;
    ForkDemo                demo;         // the recorded replay (with keyframes) the forks Seek into
    Timeline                original;     // the recorded match, replayed (the tree root)
    ForkedTimeline          forkA;        // the PHYSICS counterfactual (kCmdImpulse)
    ForkedTimeline          forkB;        // the GAMEPLAY counterfactual (kCmdAbility)
    Timeline                resimA;       // branch A's realized counterfactual timeline
    Timeline                resimB;       // branch B's realized counterfactual timeline
    TimelineDiff            diffA;        // original vs branch A
    TimelineDiff            diffB;        // original vs branch B
    Outcome                 outOriginal, outA, outB;   // the three final outcomes
};

// BuildFk1Scenario(world0): assemble the canonical fork scenario over the FROZEN canonical replay world.
// world0 is filled IN PLACE (VerdictWorld is non-copyable). Returns the fully-realized 3-node tree.
inline Fk1Scenario BuildFk1Scenario(verdict::VerdictWorld& world0) {
    const verdict::CanonicalReplay cr = verdict::BuildCanonicalReplay(world0);
    Fk1Scenario sc;
    sc.params         = cr.params;
    sc.ticks          = cr.ticks;
    sc.player         = cr.player;
    sc.w0Snap         = verdict::SnapshotWorld(world0);
    sc.originalStream = cr.stream;

    // Record the match (demo with keyframes) + replay the original timeline (the tree root).
    sc.demo     = BuildForkDemo(sc.w0Snap, sc.params, sc.originalStream, sc.ticks, kFk1KeyframeInterval);
    sc.original = SimulateTimeline(sc.w0Snap, sc.params, sc.originalStream, sc.ticks);

    // Branch A — the PHYSICS counterfactual: rewind to kFk1ForkTick (RP4 Seek), inject a shove on the player.
    sc.forkA = ForkAt(sc.demo, sc.original, sc.params, kFk1ForkTick, sc.originalStream);
    verdict::Command shove;
    shove.tick   = kFk1ForkTick;
    shove.kind   = verdict::kCmdImpulse;
    shove.target = sc.player;
    shove.arg    = verdict::FxVec3{(verdict::fx)((int64_t)3 * (int64_t)verdict::kOne), 0, 0};  // +x delta-vel
    MutateInput(sc.forkA, shove);
    sc.resimA = ResimulateFork(sc.forkA);

    // Branch B — the GAMEPLAY counterfactual: rewind to kFk1ForkTick, trigger an ability (+25 Health).
    sc.forkB = ForkAt(sc.demo, sc.original, sc.params, kFk1ForkTick, sc.originalStream);
    verdict::Command ability;
    ability.tick   = kFk1ForkTick;
    ability.kind   = verdict::kCmdAbility;
    ability.target = sc.player;
    ability.arg    = verdict::FxVec3{(verdict::fx)((int64_t)25 * (int64_t)verdict::kOne), 0, 0};  // +25 hp (arg.x>>16)
    MutateInput(sc.forkB, ability);
    sc.resimB = ResimulateFork(sc.forkB);

    // The diffs + outcomes (the tree's proofs).
    sc.diffA       = DiffTimelines(sc.original, sc.resimA);
    sc.diffB       = DiffTimelines(sc.original, sc.resimB);
    sc.outOriginal = OutcomeOf(sc.original, sc.player);
    sc.outA        = OutcomeOf(sc.resimA,   sc.player);
    sc.outB        = OutcomeOf(sc.resimB,   sc.player);
    return sc;
}

// =================================================================================================
// THE SHOWCASE VIZ — a strict-integer TIMELINE-TREE image (NO shader, NO float, NO <cmath>).
// Three horizontal tracks (ORIGINAL + branch A + branch B) over the ticks; each tick a cell tinted by that
// track's per-tick digest bytes (a visual fingerprint). The SHARED prefix [0, forkTick) is highlighted
// identical (a bright top rule spanning the three tracks); a FORK-POINT column is marked at forkTick; each
// branch's DIVERGENCE column is marked; and a right-side PANEL shows the original-final vs the two
// fork-final outcomes (swatches tinted by finalDigest, health bars). Pure integer pixel writes into an RGBA8
// buffer, shared VERBATIM by the Vulkan --fk1-fork-shot + Metal --fk1-fork so the pixels are byte-identical
// cross-backend BY CONSTRUCTION (the ac1/sq2 precedent).
// =================================================================================================

struct Fk1VizStats {
    uint32_t    ticks            = 0;   // timeline length
    uint32_t    forkTick         = 0;   // the rewind/fork tick
    uint32_t    branches         = 0;   // counterfactual branches drawn (2)
    int         firstDivergenceA = -1;  // original vs branch A first-divergent tick
    int         firstDivergenceB = -1;  // original vs branch B first-divergent tick
    bool        outcomeChanged   = false;  // original-final != BOTH fork-finals
    uint32_t    width            = 0;
    uint32_t    height           = 0;
    uint64_t    pixDigest        = 0;   // net::DigestBytes over the RGBA8 pixels (the strict-zero proof)
    uint64_t    origFullDigest   = 0;   // the three whole-timeline fingerprints (pinned)
    uint64_t    forkAFullDigest  = 0;
    uint64_t    forkBFullDigest  = 0;
};

// Fixed viz geometry (all integer).
inline constexpr int kFk1ImgW    = 600;
inline constexpr int kFk1ImgH    = 320;
inline constexpr int kFk1StripX  = 96;    // per-tick cell strip left edge
inline constexpr int kFk1StripW  = 372;   // strip pixel width (spans all ticks)
inline constexpr int kFk1Trk0Y   = 52;    // ORIGINAL track top
inline constexpr int kFk1Trk1Y   = 128;   // branch A track top
inline constexpr int kFk1Trk2Y   = 204;   // branch B track top
inline constexpr int kFk1TrkH    = 48;    // track cell height
inline constexpr int kFk1PanelX  = 484;   // outcome panel left edge
inline constexpr int kFk1PanelW  = 104;   // outcome panel width

// RenderFk1ForkViz: fill `out` (RGBA8, kFk1ImgW x kFk1ImgH) + the stat block. Builds the canonical fork
// scenario, resimulates the tree, and draws the timeline-tree report. Deterministic + pure integer -> two
// calls (and two backends) byte-identical.
inline void RenderFk1ForkViz(std::vector<uint8_t>& out, Fk1VizStats& stats) {
    const int W = kFk1ImgW, H = kFk1ImgH;
    out.assign((std::size_t)W * H * 4u, 0);

    auto px = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        uint8_t* d = &out[((std::size_t)y * W + x) * 4u];
        d[0] = r; d[1] = g; d[2] = b; d[3] = 255;
    };
    auto fillRect = [&](int x0, int y0, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y < y0 + h; ++y)
            for (int x = x0; x < x0 + w; ++x) px(x, y, r, g, b);
    };
    auto vline = [&](int x, int y0, int y1, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y <= y1; ++y) px(x, y, r, g, b);
    };
    auto hline = [&](int x0, int x1, int y, uint8_t r, uint8_t g, uint8_t b) {
        for (int x = x0; x <= x1; ++x) px(x, y, r, g, b);
    };
    auto border = [&](int x0, int y0, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        for (int x = x0; x < x0 + w; ++x) { px(x, y0, r, g, b); px(x, y0 + h - 1, r, g, b); }
        for (int y = y0; y < y0 + h; ++y) { px(x0, y, r, g, b); px(x0 + w - 1, y, r, g, b); }
    };

    // Background + title band.
    fillRect(0, 0, W, H, 15, 17, 23);
    fillRect(0, 0, W, 36, 24, 27, 36);
    fillRect(kFk1StripX, 8, kFk1StripW, 3, 66, 72, 88);   // a strip-aligned title rule

    // ---- Build + realize the canonical fork tree. ---------------------------------------------------
    verdict::VerdictWorld world0;
    const Fk1Scenario sc = BuildFk1Scenario(world0);
    const uint32_t T  = sc.ticks;
    const uint32_t FK = sc.forkA.forkTick;   // both branches rewound to kFk1ForkTick

    // Cell X mapping (deterministic integer division): tick t -> [x0, x1).
    auto cellX0 = [&](uint32_t t) { return kFk1StripX + (int)(((int64_t)t * kFk1StripW) / (int64_t)(T ? T : 1)); };

    // ---- The three tracks: ORIGINAL, branch A, branch B (each tinted per-tick by its digest). --------
    struct Track { int y; const Timeline* tl; const TimelineDiff* diff; uint8_t br, bg, bb; };
    const Track tracks[3] = {
        { kFk1Trk0Y, &sc.original, nullptr,   90,  120, 200 },   // ORIGINAL — cool blue accent
        { kFk1Trk1Y, &sc.resimA,   &sc.diffA, 210, 120, 60  },   // branch A (physics) — warm orange
        { kFk1Trk2Y, &sc.resimB,   &sc.diffB, 90,  190, 120 },   // branch B (gameplay) — green
    };
    for (int ti = 0; ti < 3; ++ti) {
        const Track& tr = tracks[ti];
        const int y0 = tr.y;
        // Accent swatch (identifies the track).
        fillRect(kFk1StripX - 84, y0, 72, kFk1TrkH, tr.br, tr.bg, tr.bb);
        border(kFk1StripX - 84, y0, 72, kFk1TrkH, 230, 230, 230);
        // Track background.
        fillRect(kFk1StripX, y0, kFk1StripW, kFk1TrkH, 28, 31, 40);
        // Per-tick digest cells.
        for (uint32_t t = 0; t < T; ++t) {
            const int x0 = cellX0(t), x1 = cellX0(t + 1);
            const int cw = (x1 - x0 > 1) ? (x1 - x0 - 1) : 1;
            const uint64_t d = (t < tr.tl->digests.size()) ? tr.tl->digests[(std::size_t)t] : 0ull;
            const uint8_t r = (uint8_t)(d & 0xFF);
            const uint8_t g = (uint8_t)((d >> 21) & 0xFF);
            const uint8_t b = (uint8_t)((d >> 42) & 0xFF);
            fillRect(x0, y0 + 6, cw, kFk1TrkH - 12, r, g, b);
        }
        border(kFk1StripX, y0, kFk1StripW, kFk1TrkH, 70, 76, 92);
        // Divergence marker (branches only): a bright column at the first-divergent tick.
        if (tr.diff && tr.diff->firstDivergence >= 0) {
            const int mx = cellX0((uint32_t)tr.diff->firstDivergence);
            vline(mx, y0 - 6, y0 + kFk1TrkH + 6, 250, 220, 60);
            vline(mx + 1, y0 - 6, y0 + kFk1TrkH + 6, 250, 220, 60);
        }
    }

    // ---- The SHARED-PREFIX highlight: a bright rule spanning the three tracks over [0, forkTick). ----
    {
        const int px0 = cellX0(0), px1 = cellX0(FK);
        hline(px0, px1, kFk1Trk0Y - 12, 120, 220, 255);
        hline(px0, px1, kFk1Trk0Y - 11, 120, 220, 255);
        // small identity ticks down each track's shared region.
        for (int ti = 0; ti < 3; ++ti)
            border(px0, tracks[ti].y - 2, px1 - px0, kFk1TrkH + 4, 60, 130, 170);
    }

    // ---- The FORK-POINT column: a full-height marker where the branches rewind to. ------------------
    {
        const int fx = cellX0(FK);
        vline(fx, kFk1Trk0Y - 14, kFk1Trk2Y + kFk1TrkH + 8, 255, 255, 255);
        vline(fx + 1, kFk1Trk0Y - 14, kFk1Trk2Y + kFk1TrkH + 8, 200, 200, 210);
        // a down-arrow head above the tracks.
        for (int dy = 0; dy < 6; ++dy)
            for (int dx = -dy; dx <= dy; ++dx) px(fx + dx, kFk1Trk0Y - 14 + dy, 255, 255, 255);
    }

    // ---- The OUTCOME PANEL (right): original-final vs the two fork-finals (swatch + health bar). -----
    {
        fillRect(kFk1PanelX, 44, kFk1PanelW, H - 56, 22, 24, 32);
        border(kFk1PanelX, 44, kFk1PanelW, H - 56, 70, 76, 92);
        struct Row { int y; const Outcome* o; uint8_t br, bg, bb; };
        const Row rows[3] = {
            { 56,  &sc.outOriginal, 90,  120, 200 },
            { 152, &sc.outA,        210, 120, 60  },
            { 248, &sc.outB,        90,  190, 120 },
        };
        for (int ri = 0; ri < 3; ++ri) {
            const Row& rw = rows[ri];
            // A finalDigest swatch.
            const uint64_t d = rw.o->finalDigest;
            fillRect(kFk1PanelX + 8, rw.y, 40, 40,
                     (uint8_t)(d & 0xFF), (uint8_t)((d >> 21) & 0xFF), (uint8_t)((d >> 42) & 0xFF));
            border(kFk1PanelX + 8, rw.y, 40, 40, rw.br, rw.bg, rw.bb);
            // A player-health bar (clamped to the panel width; scaled /2 so 100hp ~ 50px).
            int hp = rw.o->playerHealth; if (hp < 0) hp = 0; if (hp > 2 * (kFk1PanelW - 60)) hp = 2 * (kFk1PanelW - 60);
            fillRect(kFk1PanelX + 54, rw.y + 6, hp / 2, 12, rw.br, rw.bg, rw.bb);
            // An entity-count tick row (one pip per live entity, capped).
            uint32_t ec = rw.o->entityCount; if (ec > (uint32_t)(kFk1PanelW - 60)) ec = (uint32_t)(kFk1PanelW - 60);
            for (uint32_t k = 0; k < ec; ++k) px(kFk1PanelX + 54 + (int)k, rw.y + 26, 210, 210, 220);
        }
    }

    // ---- Stats. --------------------------------------------------------------------------------------
    stats.ticks            = T;
    stats.forkTick         = FK;
    stats.branches         = 2u;
    stats.firstDivergenceA = sc.diffA.firstDivergence;
    stats.firstDivergenceB = sc.diffB.firstDivergence;
    stats.outcomeChanged   = !OutcomesEqual(sc.outOriginal, sc.outA) && !OutcomesEqual(sc.outOriginal, sc.outB);
    stats.width            = (uint32_t)W;
    stats.height           = (uint32_t)H;
    stats.pixDigest        = net::DigestBytes(out.data(), out.size());
    stats.origFullDigest   = sc.original.fullDigest;
    stats.forkAFullDigest  = sc.resimA.fullDigest;
    stats.forkBFullDigest  = sc.resimB.fullDigest;
}

}  // namespace replay
}  // namespace hf
