#pragma once
// Slice MM1 — DETERMINISTIC MOTION MATCHING (Track-S slice S4 of docs/SUPERIORITY_ROADMAP.md; the
// long-queued flagship #33): a pose FEATURE DATABASE extracted from animation clips + an INTEGER
// nearest-neighbor search that picks the best next frame given the character's trajectory intent —
// the data-driven locomotion technique UE5 markets as Motion Matching, made BIT-EXACT and
// lockstep/rollback-replayable (UE5's is float / non-replayable; ours is a pure integer function of
// (database, input stream), so two peers re-derive the entire animation-selection sequence
// bit-for-bit). Namespace hf::anim::mm, header-only, pure CPU, NO device / backend symbols, NO RNG,
// NO clock. engine/anim/animation.h + skeleton.h are #included READ-ONLY (byte-untouched) — the
// existing float clip/sampling stack is the POSE SOURCE; everything downstream of the one documented
// quantize boundary is integer.
//
// THE ONE FLOAT->INTEGER BOUNDARY (documented, the FPX host-snap discipline): BuildMotionDatabase
// samples each clip's float pose (anim::SampleLocalPose + the hierarchy walk) on a FIXED tick grid
// (tick/fps seconds) and quantizes every sampled joint POSITION to Q16.16 ONCE (QuantizeFx =
// llround(v * 65536) — round-half-away, deterministic IEEE double). ALL derived features (velocities,
// trajectory offsets) are then computed in INTEGER from the quantized positions, and the runtime
// (query build, cost, argmin, controller, lockstep) is integer end-to-end. The float sampling itself
// is deterministic per platform; for the CROSS-COMPILER digest pins the fixture clips use exact
// binary-fraction keyframe times/values (all lerps exact in float, immune to FMA contraction).
//
// THE FEATURE VECTOR (kFeatureDim = 20, all Q16.16, the standard motion-matching set):
//   [0..1]   root velocity (x, z)                    — (rootQ(t+1) - rootQ(t)) * fps
//   [2..7]   future trajectory offsets (x, z) at t + {10, 20, 30} frames, ROOT-RELATIVE
//            (rootQ(t+d) - rootQ(t))
//   [8..13]  left/right foot positions (x, y, z), ROOT-RELATIVE (footQ - rootQ)
//   [14..19] left/right foot velocities (x, y, z)     — (footRel(t+1) - footRel(t)) * fps
// HONEST v1 LIMITS (documented, not hidden): features are root-POSITION-relative but NOT
// root-facing-relative (no orientation removal — clips are assumed authored in a canonical facing;
// yaw-invariant matching is a future refinement). Frames within kTrajOffsets.back() ticks of a
// clip's end are NOT database entries (the lookahead margin is TRUNCATED rather than loop-wrapped —
// the cursor loops over the database range only). Root height / hip features are omitted (planar
// locomotion v1).
//
// THE DATABASE: flat POD arrays {clip, tick, feature[20]} + per-clip ranges + the weight vector,
// built by BuildMotionDatabase (deterministic: fixed sample grid, fixed clip order, quantize once);
// DigestMotionDatabase is the pinned FNV-1a-64 golden currency (field-wise, layout-independent).
//
// THE QUERY (MatchPose): integer WEIGHTED L1 cost over ALL database frames — v1 is a documented
// BRUTE-FORCE linear scan (int64 accumulate; at showcase scale, tens-to-thousands of frames, this is
// microseconds; acceleration structures — kd-trees/VP-trees — are a future refinement, and any such
// structure must reproduce this scan's argmin bit-for-bit to be admissible). The argmin uses the
// STRICT (cost, index) total order: strictly-lower cost wins; equal cost keeps the LOWER index (the
// deterministic tie-break — duplicate feature rows resolve to the first occurrence). Fixed default
// weights (Q16.16): rootVel 2.0, trajectory 1.0, footPos 1.0, footVel 0.25 — velocity intent
// dominates, foot continuity disambiguates, foot velocity breaks pose mirrors. cost units: raw
// Q16.16*Q16.16 int64 products, NOT down-shifted (cost == 0 iff every weighted dim matches exactly).
//
// THE CONTROLLER (StepMotionMatch) — one call per fixed tick:
//   1. ADVANCE: cursor = (cursor + 1) mod clipFrames (the current clip plays forward, looping over
//      its database range).
//   2. if (tick % searchInterval == 0): build the query (desired velocity + trajectory extrapolated
//      from the input; foot pos/vel features COPIED from the current database frame — the pose IS
//      database-resident, the standard MM trick) and MatchPose over the whole database. Adopt the
//      winner. When the input matches the current clip the current frame self-matches at cost 0 and
//      the argmin keeps it (continuation is free); a changed intent makes the velocity/trajectory
//      terms dominate and the search jumps to the other clip's best-foot-matching frame.
//   3. tick += 1. Selections (clip switches) can therefore ONLY change on search ticks.
// THE TRANSITION SCHEME (v1, documented): HARD SWITCH — the cursor teleports to the matched frame
// with NO cross-fade (visible popping at switches is the honest v1 behavior; the foot-feature terms
// minimize it by choosing the closest-pose frame). A fixed-length cross-fade via the existing float
// anim::BlendAnimations is a RENDER-side smoothing a caller can layer on top (blend the palettes of
// the previous and current selections for N frames) WITHOUT touching this integer selection state —
// keeping the blend out of MMState is what keeps the lockstep snapshot minimal and integer.
//
// LOCKSTEP + ROLLBACK (the CL5/FPX5/PT5 command+snapshot mold): commands are desired-velocity changes
// (the player stick, kMMCmdSetVelocity); the held input is PART OF MMState (snapshot completeness —
// omitting it, or the cursor, diverges: the tests prove it). MMState is a fixed-size POD (7 int32
// words) — the snapshot is a struct copy; RunMMLockstep re-derives the entire selection + cursor
// sequence from (database, initial state, command stream) alone, and RunMMRollback restores a
// snapshot and re-simulates to correct a misprediction. THE HEADLINE: replayable motion-matched
// locomotion — a peer reproduces every animation-selection decision bit-for-bit.
//
// THE FIXTURE ASSETS (MakeMMTestRig / MakeMMIdleClip / MakeMMWalkClip — the PA1
// MakePulsingFountainEffect precedent: shared deterministic fixtures so the test and BOTH showcase
// harnesses use identical bits): a 3-joint rig (root + two feet) with two 2-second clips at 32
// samples/sec — an IDLE loop (root static, feet bobbing +-1/16) and a WALK loop (root advancing +z at
// exactly 1.5 units/s, feet swinging +-3/8 in antiphase). All keyframe times/values are exact binary
// fractions -> the float lerp inside anim::SampleLocalPose is EXACT (no rounding, no FMA sensitivity)
// -> the quantized database is bit-identical across MSVC/clang/architectures BY CONSTRUCTION.
// (The repo's checked-in test clips are trivial single-key fixtures, so MM1 synthesizes its own.)
//
// BOUNDS: every clip/cursor/frame access is range-checked (deterministic no-op / reset on
// out-of-range). Cost accumulate: |int32 diff| <= 2^32, weight <= 2^18, 20 dims -> < 2^55, safely
// inside int64.

#include <cmath>
#include <cstdint>
#include <vector>

#include "anim/animation.h"   // READ-ONLY: Animation/Channel/JointPose/SampleLocalPose — the pose source
#include "anim/skeleton.h"    // READ-ONLY: Skeleton/Joint — the joint hierarchy
#include "math/math.h"        // READ-ONLY: Vec3/Quat/Mat4/FromTRS — the float hierarchy walk

namespace hf::anim {
namespace mm {

// ----- Q16.16 fixed point (self-contained; the anim family's first integer header) ------------------
using fx = int32_t;
inline constexpr int kFrac = 16;
inline constexpr fx  kOne  = 1 << kFrac;

// fxmul: Q16.16 multiply (int64 intermediate, arithmetic >> — truncates toward -inf; C++20-defined).
inline fx fxmul(fx a, fx b) { return (fx)(((int64_t)a * (int64_t)b) >> kFrac); }

// QuantizeFx: THE one float->integer boundary — round-half-away-from-zero via llround on the IEEE
// double product (deterministic on every conforming platform/compiler).
inline fx QuantizeFx(float v) { return (fx)std::llround((double)v * 65536.0); }

struct FxV3 { fx x = 0, y = 0, z = 0; };

// ----- The feature vector layout (see the header banner) ---------------------------------------------
inline constexpr int kNumTraj = 3;
inline constexpr int kTrajOffsets[kNumTraj] = {10, 20, 30};   // future ticks (frames at cfg.fps)
inline constexpr int kFeatureDim = 20;
// Named base indices into feature[] (documented layout — tests hand-address through these).
inline constexpr int kFRootVel  = 0;    // [0..1]   root velocity x,z
inline constexpr int kFTraj     = 2;    // [2..7]   trajectory offsets x,z at +10/+20/+30
inline constexpr int kFFootPos  = 8;    // [8..13]  left(x,y,z) then right(x,y,z) foot positions
inline constexpr int kFFootVel  = 14;   // [14..19] left then right foot velocities

// ----- MMConfig: the database-build knobs + fixed cost weights (all part of the pinned digest) -------
struct MMConfig {
    int fps       = 32;        // database sample rate, ticks/second (32 = exact binary tick dt)
    int rootJoint = 0;         // the root joint index (trajectory source)
    int leftFoot  = 1;         // left-foot joint index
    int rightFoot = 2;         // right-foot joint index
    fx  wRootVel  = 2 * kOne;  // root-velocity weight (intent dominates)
    fx  wTraj     = kOne;      // trajectory-offset weight
    fx  wFootPos  = kOne;      // foot-position weight (pose continuity)
    fx  wFootVel  = kOne / 4;  // foot-velocity weight (breaks pose mirrors)
};

// One database frame: which clip, which sample tick, and its 20-dim Q16.16 feature row. POD
// (int32 words only — memcmp/byte-compare safe, no padding).
struct MotionFrame {
    int32_t clip = 0;                // index into the clip list passed to BuildMotionDatabase
    int32_t tick = 0;                // the sample tick within that clip (time = tick / cfg.fps)
    fx      feature[kFeatureDim] = {};
};

// Per-clip database range: frames [firstFrame, firstFrame + frameCount) belong to this clip and are
// consecutive ascending ticks 0..frameCount-1 (the cursor loops over exactly this range).
struct MotionClipRange {
    int32_t firstFrame = 0;
    int32_t frameCount = 0;
};

struct MotionDatabase {
    MMConfig                     cfg;
    std::vector<MotionClipRange> clips;
    std::vector<MotionFrame>     frames;
    fx                           weight[kFeatureDim] = {};   // per-dim expansion of cfg's 4 weights
};

// FrameIndex: the flat frame index of (clip, cursor), or -1 out-of-range (bounds-checked).
inline int FrameIndex(const MotionDatabase& db, int clip, int cursor) {
    if (clip < 0 || (size_t)clip >= db.clips.size()) return -1;
    const MotionClipRange& r = db.clips[(size_t)clip];
    if (cursor < 0 || cursor >= r.frameCount) return -1;
    return r.firstFrame + cursor;
}

// ----- The database BUILD (deterministic: fixed grid, fixed order, quantize once) --------------------
namespace detail {

// The global (model-space) joint transforms for a local pose — PaletteFromLocalPose's forward pass
// WITHOUT the inverse-bind multiply (we need true joint world positions; palette entries fold in the
// bind pose). Skeleton joints are topologically sorted (parent before child) by the anim contract.
inline std::vector<math::Mat4> GlobalsFromLocalPose(const Skeleton& sk,
                                                    const std::vector<JointPose>& pose) {
    const size_t n = sk.joints.size();
    std::vector<math::Mat4> global(n);
    for (size_t j = 0; j < n && j < pose.size(); ++j) {
        const math::Mat4 local = math::FromTRS(pose[j].t, pose[j].r, pose[j].s);
        const int parent = sk.joints[j].parent;
        global[j] = (parent >= 0 && (size_t)parent < j) ? (global[(size_t)parent] * local) : local;
    }
    return global;
}

// Sample the three tracked joints' world positions at `tick` and quantize (THE boundary). A joint
// index out of range deterministically yields the origin.
struct JointsQ { FxV3 root, lf, rf; };

inline JointsQ SampleJointsQ(const Skeleton& sk, const Animation& a, const MMConfig& cfg, int tick) {
    const float t = (float)tick / (float)cfg.fps;
    const std::vector<math::Mat4> g = GlobalsFromLocalPose(sk, SampleLocalPose(sk, a, t));
    auto posQ = [&](int j) -> FxV3 {
        if (j < 0 || (size_t)j >= g.size()) return FxV3{};
        const math::Mat4& m = g[(size_t)j];
        return FxV3{QuantizeFx(m.m[12]), QuantizeFx(m.m[13]), QuantizeFx(m.m[14])};
    };
    return JointsQ{posQ(cfg.rootJoint), posQ(cfg.leftFoot), posQ(cfg.rightFoot)};
}

}  // namespace detail

// BuildMotionDatabase: sample every clip on the fixed tick grid, quantize ONCE, derive all features
// in INTEGER. A clip contributes frames for ticks [0, totalTicks - maxTrajOffset] (the lookahead
// margin is truncated — see the banner); a clip too short for the lookahead contributes none.
inline MotionDatabase BuildMotionDatabase(const Skeleton& sk, const std::vector<Animation>& clips,
                                          const MMConfig& cfg = MMConfig{}) {
    MotionDatabase db;
    db.cfg = cfg;
    // Expand the 4 weights into the per-dim vector (fixed layout — see the banner).
    for (int k = 0; k < 2; ++k)              db.weight[kFRootVel + k] = cfg.wRootVel;
    for (int k = 0; k < 2 * kNumTraj; ++k)   db.weight[kFTraj + k]    = cfg.wTraj;
    for (int k = 0; k < 6; ++k)              db.weight[kFFootPos + k] = cfg.wFootPos;
    for (int k = 0; k < 6; ++k)              db.weight[kFFootVel + k] = cfg.wFootVel;

    const int maxOff = kTrajOffsets[kNumTraj - 1];
    for (size_t c = 0; c < clips.size(); ++c) {
        const Animation& a = clips[c];
        const int totalTicks = (int)std::llround((double)a.duration * (double)cfg.fps);
        int dbCount = totalTicks - maxOff + 1;             // ticks 0 .. totalTicks - maxOff
        if (dbCount < 0) dbCount = 0;
        MotionClipRange r;
        r.firstFrame = (int32_t)db.frames.size();
        r.frameCount = dbCount;
        db.clips.push_back(r);
        for (int t = 0; t < dbCount; ++t) {
            const detail::JointsQ j0 = detail::SampleJointsQ(sk, a, cfg, t);
            const detail::JointsQ j1 = detail::SampleJointsQ(sk, a, cfg, t + 1);
            MotionFrame f;
            f.clip = (int32_t)c;
            f.tick = t;
            // Root velocity (x,z): integer diff of quantized positions * fps (exact int multiply).
            f.feature[kFRootVel + 0] = (fx)((int64_t)(j1.root.x - j0.root.x) * cfg.fps);
            f.feature[kFRootVel + 1] = (fx)((int64_t)(j1.root.z - j0.root.z) * cfg.fps);
            // Future trajectory offsets (x,z), root-relative.
            for (int k = 0; k < kNumTraj; ++k) {
                const detail::JointsQ jf = detail::SampleJointsQ(sk, a, cfg, t + kTrajOffsets[k]);
                f.feature[kFTraj + 2 * k + 0] = jf.root.x - j0.root.x;
                f.feature[kFTraj + 2 * k + 1] = jf.root.z - j0.root.z;
            }
            // Foot positions, root-relative (left then right).
            const FxV3 lf0{j0.lf.x - j0.root.x, j0.lf.y - j0.root.y, j0.lf.z - j0.root.z};
            const FxV3 rf0{j0.rf.x - j0.root.x, j0.rf.y - j0.root.y, j0.rf.z - j0.root.z};
            const FxV3 lf1{j1.lf.x - j1.root.x, j1.lf.y - j1.root.y, j1.lf.z - j1.root.z};
            const FxV3 rf1{j1.rf.x - j1.root.x, j1.rf.y - j1.root.y, j1.rf.z - j1.root.z};
            f.feature[kFFootPos + 0] = lf0.x; f.feature[kFFootPos + 1] = lf0.y;
            f.feature[kFFootPos + 2] = lf0.z;
            f.feature[kFFootPos + 3] = rf0.x; f.feature[kFFootPos + 4] = rf0.y;
            f.feature[kFFootPos + 5] = rf0.z;
            // Foot velocities: diff of root-relative positions * fps.
            f.feature[kFFootVel + 0] = (fx)((int64_t)(lf1.x - lf0.x) * cfg.fps);
            f.feature[kFFootVel + 1] = (fx)((int64_t)(lf1.y - lf0.y) * cfg.fps);
            f.feature[kFFootVel + 2] = (fx)((int64_t)(lf1.z - lf0.z) * cfg.fps);
            f.feature[kFFootVel + 3] = (fx)((int64_t)(rf1.x - rf0.x) * cfg.fps);
            f.feature[kFFootVel + 4] = (fx)((int64_t)(rf1.y - rf0.y) * cfg.fps);
            f.feature[kFFootVel + 5] = (fx)((int64_t)(rf1.z - rf0.z) * cfg.fps);
            db.frames.push_back(f);
        }
    }
    return db;
}

// ----- The pinned digest (FNV-1a-64, field-wise int32 words — layout/endianness-independent) ---------
namespace detail {
inline uint64_t Fnv1a64Word(uint64_t h, uint32_t w) {
    for (int b = 0; b < 4; ++b) {
        h ^= (w >> (8 * b)) & 0xffu;
        h *= 1099511628211ull;
    }
    return h;
}
}  // namespace detail

inline uint64_t DigestMotionDatabase(const MotionDatabase& db) {
    uint64_t h = 14695981039346656037ull;
    h = detail::Fnv1a64Word(h, (uint32_t)db.cfg.fps);
    h = detail::Fnv1a64Word(h, (uint32_t)db.cfg.rootJoint);
    h = detail::Fnv1a64Word(h, (uint32_t)db.cfg.leftFoot);
    h = detail::Fnv1a64Word(h, (uint32_t)db.cfg.rightFoot);
    for (int k = 0; k < kFeatureDim; ++k) h = detail::Fnv1a64Word(h, (uint32_t)db.weight[k]);
    for (const MotionClipRange& r : db.clips) {
        h = detail::Fnv1a64Word(h, (uint32_t)r.firstFrame);
        h = detail::Fnv1a64Word(h, (uint32_t)r.frameCount);
    }
    for (const MotionFrame& f : db.frames) {
        h = detail::Fnv1a64Word(h, (uint32_t)f.clip);
        h = detail::Fnv1a64Word(h, (uint32_t)f.tick);
        for (int k = 0; k < kFeatureDim; ++k) h = detail::Fnv1a64Word(h, (uint32_t)f.feature[k]);
    }
    return h;
}

// ----- The QUERY: integer weighted-L1 brute-force argmin with the (cost, index) tie-break ------------
struct MatchResult {
    int32_t frameIndex = -1;   // flat index into db.frames (-1 = empty database)
    int32_t clip = -1;
    int32_t tick = 0;
    int64_t cost = INT64_MAX;
};

// MatchCost: Sum_k |q[k] - f[k]| * weight[k], int64 (raw Q16.16*Q16.16 products, NOT down-shifted;
// 0 iff every weighted dim matches exactly).
inline int64_t MatchCost(const MotionDatabase& db, const MotionFrame& f, const fx* q) {
    int64_t cost = 0;
    for (int k = 0; k < kFeatureDim; ++k) {
        int64_t d = (int64_t)q[k] - (int64_t)f.feature[k];
        if (d < 0) d = -d;
        cost += d * (int64_t)db.weight[k];
    }
    return cost;
}

// MatchPose: the v1 BRUTE-FORCE linear scan (documented in the banner). Strict '<' keeps the earlier
// frame on ties -> the (cost, index) total order (lower index wins).
inline MatchResult MatchPose(const MotionDatabase& db, const fx* q) {
    MatchResult best;
    for (size_t i = 0; i < db.frames.size(); ++i) {
        const int64_t c = MatchCost(db, db.frames[i], q);
        if (c < best.cost) {
            best.cost = c;
            best.frameIndex = (int32_t)i;
            best.clip = db.frames[i].clip;
            best.tick = db.frames[i].tick;
        }
    }
    return best;
}

// ----- The CONTROLLER: MMState (POD, snapshotable) + StepMotionMatch ---------------------------------
struct MMInput { fx velX = 0; fx velZ = 0; };   // the desired planar velocity (the player stick)

// MMState is the COMPLETE simulation state (7 int32 words, no padding): snapshot = struct copy;
// omitting ANY field (the tests prove cursor and input) diverges a replay.
struct MMState {
    int32_t  clip = 0;             // current clip
    int32_t  cursor = 0;           // current frame WITHIN the clip's database range
    uint32_t tick = 0;             // the controller tick counter (drives the search cadence)
    MMInput  input;                // the HELD desired velocity (commands change it; part of state)
    uint32_t searches = 0;         // diagnostics: searches run
    uint32_t switches = 0;         // diagnostics: CLIP switches adopted
    int32_t  lastSwitchTick = -1;  // diagnostics: tick of the most recent clip switch (-1 = none)
};

struct MMParams {
    int searchInterval = 10;       // run MatchPose every N ticks (N >= 1)
};

// BuildQuery: desired velocity + trajectory EXTRAPOLATED from the input (offset_d = vel * d/fps,
// truncating fxmul) + foot pos/vel features COPIED from the current database frame (the pose is
// database-resident — the standard MM query construction). q must have kFeatureDim slots.
inline void BuildQuery(const MotionDatabase& db, const MMState& st, fx* q) {
    q[kFRootVel + 0] = st.input.velX;
    q[kFRootVel + 1] = st.input.velZ;
    for (int k = 0; k < kNumTraj; ++k) {
        // d/fps in Q16.16 (exact when fps divides d * 2^16 — true for the binary fps convention).
        const fx dt = (fx)(((int64_t)kTrajOffsets[k] << kFrac) / db.cfg.fps);
        q[kFTraj + 2 * k + 0] = fxmul(st.input.velX, dt);
        q[kFTraj + 2 * k + 1] = fxmul(st.input.velZ, dt);
    }
    const int cur = FrameIndex(db, st.clip, st.cursor);
    for (int k = kFFootPos; k < kFeatureDim; ++k)
        q[k] = (cur >= 0) ? db.frames[(size_t)cur].feature[k] : 0;
}

// StepMotionMatch: one controller tick — ADVANCE, then SEARCH on the interval, then tick++ (the
// banner's exact semantics; clip switches can only happen on ticks where tick % searchInterval == 0).
inline void StepMotionMatch(MMState& st, const MotionDatabase& db, const MMParams& prm) {
    // Deterministic bounds repair (a corrupt/foreign state resets to the database origin).
    if (st.clip < 0 || (size_t)st.clip >= db.clips.size()) { st.clip = 0; st.cursor = 0; }
    if (!db.clips.empty()) {
        const MotionClipRange& r0 = db.clips[(size_t)st.clip];
        if (st.cursor < 0 || st.cursor >= r0.frameCount) st.cursor = 0;
        // 1) ADVANCE the cursor (loop over the clip's database range).
        if (r0.frameCount > 0) st.cursor = (st.cursor + 1) % r0.frameCount;
        // 2) SEARCH on the interval.
        const int n = prm.searchInterval > 0 ? prm.searchInterval : 1;
        if ((st.tick % (uint32_t)n) == 0u && !db.frames.empty()) {
            fx q[kFeatureDim];
            BuildQuery(db, st, q);
            const MatchResult m = MatchPose(db, q);
            st.searches += 1u;
            const int cur = FrameIndex(db, st.clip, st.cursor);
            if (m.frameIndex >= 0 && m.frameIndex != cur) {
                if (m.clip != st.clip) {
                    st.switches += 1u;
                    st.lastSwitchTick = (int32_t)st.tick;
                }
                st.clip = m.clip;
                st.cursor = m.frameIndex - db.clips[(size_t)m.clip].firstFrame;
            }
        }
    }
    // 3) advance the tick counter.
    st.tick += 1u;
}

// MMRootVelocity: the CURRENT frame's root-velocity feature (x,z) — the root-motion drive a caller
// integrates to move the character (pos += vel * 1/fps per tick). Zero when out of range.
inline void MMRootVelocity(const MotionDatabase& db, const MMState& st, fx& velX, fx& velZ) {
    const int cur = FrameIndex(db, st.clip, st.cursor);
    velX = (cur >= 0) ? db.frames[(size_t)cur].feature[kFRootVel + 0] : 0;
    velZ = (cur >= 0) ? db.frames[(size_t)cur].feature[kFRootVel + 1] : 0;
}

// MMStateDigest: the FNV-1a-64 field-wise state digest (the lockstep comparison currency).
inline uint64_t MMStateDigest(const MMState& st) {
    uint64_t h = 14695981039346656037ull;
    h = detail::Fnv1a64Word(h, (uint32_t)st.clip);
    h = detail::Fnv1a64Word(h, (uint32_t)st.cursor);
    h = detail::Fnv1a64Word(h, st.tick);
    h = detail::Fnv1a64Word(h, (uint32_t)st.input.velX);
    h = detail::Fnv1a64Word(h, (uint32_t)st.input.velZ);
    h = detail::Fnv1a64Word(h, st.searches);
    h = detail::Fnv1a64Word(h, st.switches);
    h = detail::Fnv1a64Word(h, (uint32_t)st.lastSwitchTick);
    return h;
}

// DigestMMTrace: the pinned digest of a per-tick (clip, cursor) selection trace — THE selection
// sequence made a golden.
struct MMTraceEntry { int32_t clip = 0; int32_t cursor = 0; };

inline uint64_t DigestMMTrace(const std::vector<MMTraceEntry>& trace) {
    uint64_t h = 14695981039346656037ull;
    for (const MMTraceEntry& e : trace) {
        h = detail::Fnv1a64Word(h, (uint32_t)e.clip);
        h = detail::Fnv1a64Word(h, (uint32_t)e.cursor);
    }
    return h;
}

// ----- The lockstep command stream (the CL5/FPX5 mold): desired-velocity changes ---------------------
inline constexpr uint32_t kMMCmdSetVelocity = 0u;

struct MMCommand {
    uint32_t tick = 0;                  // the tick this input applies on
    uint32_t kind = kMMCmdSetVelocity;
    fx       velX = 0;                  // the new held desired velocity
    fx       velZ = 0;
};

inline void ApplyMMCommand(MMState& st, const MMCommand& c) {
    if (c.kind == kMMCmdSetVelocity) {
        st.input.velX = c.velX;
        st.input.velZ = c.velZ;
    }
}

// SimMMTick: apply this tick's commands (ARRAY ORDER — the deterministic input-order contract) then
// StepMotionMatch once.
inline void SimMMTick(MMState& st, const MotionDatabase& db, const MMParams& prm,
                      const std::vector<MMCommand>& stream, uint32_t tick) {
    for (const MMCommand& c : stream)
        if (c.tick == tick) ApplyMMCommand(st, c);
    StepMotionMatch(st, db, prm);
}

// RunMMLockstep: THE peer entry point — `ticks` SimMMTicks from a COPY of `init` fed the command
// stream ALONE (inputs, not state). authority == replica BIT-EXACT by determinism. Optionally records
// the per-tick (clip, cursor) selection trace.
inline MMState RunMMLockstep(const MotionDatabase& db, const MMParams& prm, const MMState& init,
                             const std::vector<MMCommand>& stream, int ticks,
                             std::vector<MMTraceEntry>* trace = nullptr) {
    MMState st = init;
    if (trace) { trace->clear(); trace->reserve((size_t)(ticks > 0 ? ticks : 0)); }
    for (int t = 0; t < ticks; ++t) {
        SimMMTick(st, db, prm, stream, (uint32_t)t);
        if (trace) trace->push_back(MMTraceEntry{st.clip, st.cursor});
    }
    return st;
}

// RunMMRollback: the rollback harness (the hair/cloth RunRollback twin). Advance 0..mispredictTick
// with the authoritative stream, SNAPSHOT (a struct copy — MMState is POD), speculatively advance a
// few ticks with the MISPREDICTED stream, then RESTORE + re-simulate mispredictTick..ticks with the
// correct stream. The proof asserts result == RunMMLockstep(init, authStream, ticks) AND that the
// full mispredicted run DIFFERED (the divergence was real).
inline MMState RunMMRollback(const MotionDatabase& db, const MMParams& prm, const MMState& init,
                             const std::vector<MMCommand>& authStream,
                             const std::vector<MMCommand>& mispredictStream,
                             int ticks, int mispredictTick) {
    MMState st = init;
    for (int t = 0; t < mispredictTick; ++t)
        SimMMTick(st, db, prm, authStream, (uint32_t)t);
    const MMState snap = st;                                  // SNAPSHOT (POD copy)
    int specTicks = ticks - mispredictTick;
    if (specTicks > 3) specTicks = 3;                         // bounded speculation (the CL5 shape)
    for (int s = 0; s < specTicks; ++s)
        SimMMTick(st, db, prm, mispredictStream, (uint32_t)(mispredictTick + s));
    st = snap;                                                // ROLLBACK (restore)
    for (int t = mispredictTick; t < ticks; ++t)
        SimMMTick(st, db, prm, authStream, (uint32_t)t);
    return st;
}

// ----- The deterministic fixture assets (test + BOTH showcases use these identical bits) -------------
// A 3-joint rig: root (0), left foot (1, rest local x = -0.25), right foot (2, rest local x = +0.25).
// Identity inverse-binds, identity rest rotations — the hierarchy walk is translation-only, so every
// float op in sampling is EXACT on binary-fraction keyframes (the cross-compiler guarantee).
inline Skeleton MakeMMTestRig() {
    Skeleton sk;
    Joint root;                                   // parent -1, identity rest
    Joint lf; lf.parent = 0; lf.t = math::Vec3{-0.25f, 0.0f, 0.0f};
    Joint rf; rf.parent = 0; rf.t = math::Vec3{+0.25f, 0.0f, 0.0f};
    sk.joints = {root, lf, rf};
    return sk;
}

// The IDLE loop: 2.0 s. Root static at the origin; feet bob vertically +-1/16 in antiphase (small
// nonzero foot velocities so every database frame is feature-distinct). All binary fractions.
inline Animation MakeMMIdleClip() {
    Animation a;
    a.name = "mm_idle";
    a.duration = 2.0f;
    Channel root;
    root.jointIndex = 0; root.path = Channel::Path::Translation;
    root.times = {0.0f}; root.values = {0.0f, 0.0f, 0.0f};
    a.channels.push_back(root);
    Channel lf;
    lf.jointIndex = 1; lf.path = Channel::Path::Translation;
    lf.times = {0.0f, 1.0f, 2.0f};
    lf.values = {-0.25f, 0.0f,    0.0f,
                 -0.25f, 0.0625f, 0.0f,
                 -0.25f, 0.0f,    0.0f};
    a.channels.push_back(lf);
    Channel rf;
    rf.jointIndex = 2; rf.path = Channel::Path::Translation;
    rf.times = {0.0f, 1.0f, 2.0f};
    rf.values = {0.25f, 0.0625f, 0.0f,
                 0.25f, 0.0f,    0.0f,
                 0.25f, 0.0625f, 0.0f};
    a.channels.push_back(rf);
    return a;
}

// The WALK loop: 2.0 s. Root advances +z at exactly 1.5 units/s (0 -> 3.0); feet swing +-0.375 in z
// in antiphase with a 0.5 s half-period. All binary fractions.
inline Animation MakeMMWalkClip() {
    Animation a;
    a.name = "mm_walk";
    a.duration = 2.0f;
    Channel root;
    root.jointIndex = 0; root.path = Channel::Path::Translation;
    root.times = {0.0f, 2.0f};
    root.values = {0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 3.0f};
    a.channels.push_back(root);
    Channel lf;
    lf.jointIndex = 1; lf.path = Channel::Path::Translation;
    lf.times = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    lf.values = {-0.25f, 0.0f, -0.375f,
                 -0.25f, 0.0f, +0.375f,
                 -0.25f, 0.0f, -0.375f,
                 -0.25f, 0.0f, +0.375f,
                 -0.25f, 0.0f, -0.375f};
    a.channels.push_back(lf);
    Channel rf;
    rf.jointIndex = 2; rf.path = Channel::Path::Translation;
    rf.times = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    rf.values = {0.25f, 0.0f, +0.375f,
                 0.25f, 0.0f, -0.375f,
                 0.25f, 0.0f, +0.375f,
                 0.25f, 0.0f, -0.375f,
                 0.25f, 0.0f, +0.375f};
    a.channels.push_back(rf);
    return a;
}

}  // namespace mm
}  // namespace hf::anim
