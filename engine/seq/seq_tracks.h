#pragma once
// Slice SQ2 — DETERMINISTIC SEQUENCER SEMANTIC TRACK TYPES (camera-cut + audio + material/param +
// sub-sequence nesting). The parity gap over seq.h's S1–S6 runtime: that header ships a deterministic
// timeline substrate (scalar / event / transform tracks + easing LUTs + lockstep + Seek/scrub, goldens
// seq_test / seq_render / seq_editor) but it lacks the SEMANTIC track types a real cinematics system
// needs — camera cuts, audio one-shots, material/parameter animation, and shot-within-scene sub-sequence
// nesting. SQ2 ADDS them as a NEW header COMPOSING seq.h READ-ONLY.
//
// WHY A NEW HEADER (not additive seq.h edits): seq.h models each track kind as its OWN distinct struct
// type (ScalarTrack / EventTrack / TransformTrack) — there is NO unified TrackKind enum + dispatch that
// would grow (the audio_graph.h Kind pattern lives elsewhere). seq.h is also the self-contained,
// <cmath>-free, standalone-clang bit-exact core whose S1–S6 sections are effectively frozen (every
// section header says "APPEND-ONLY — prior slices untouched + golden-invariant"). So the idiomatic move
// is the seq_render.h / seq_editor_data.h precedent: a SEPARATE header that #includes seq.h and reuses
// its primitives verbatim, leaving seq.h BYTE-UNTOUCHED (seq_test / seq_render / seq_editor stay
// bit-identical). This header adds NO new include beyond seq.h (which already pulls fpx.h / session.h /
// flow.h). It stays PURE INTEGER Q16.16, NO <cmath>/<algorithm>/float/clock/RNG — so it inherits seq.h's
// cheap standalone-clang cross-platform proof.
//
// THE FOUR SEMANTIC TRACKS (all deterministic, all pure-integer VALUE queries):
//   1. CameraCutTrack — piecewise-constant ActiveCamera(t): the last cut at-or-before t. A cut switches
//      WHICH camera the sequence views through. HONESTY: this is a value-producing query returning a
//      cameraId; it does NOT mutate a live runtime::Camera (that is float; no live render here). A
//      transform track (seq.h S4) can itself animate the chosen camera's pose — composition, not coupling.
//   2. AudioTrack — SampleAudioCues(prevTick,curTick): fires one-shots whose tick is in the HALF-OPEN
//      window [prev, cur) — the EXACT seq.h S3 SampleEvents / AL1 event-window convention (cited below), so
//      a tick boundary fires EXACTLY ONCE across consecutive sweeps and the union of consecutive windows ==
//      one big window (the scrub-equality lemma). A fired cue carries {soundId, gain}; it COMPOSES with the
//      AU2 audio graph (engine/audio/audio_graph.h) by RETRIGGERING an envelope / stepping a note exactly
//      as RenderAu1Showcase's host schedule does (`st.st[n].envT = 0`) — demonstrated in seq_tracks_test
//      (which includes audio_graph.h) so the CORE header stays light + standalone-clang. HONESTY: the cue
//      is a value/event; wiring it to a bus is the caller's job (shown in the test).
//   3. MaterialParamTrack — SampleMaterialParam(t): a named (targetMaterialId, paramId) driven by a seq.h
//      ScalarTrack (SampleScalar + easing, VERBATIM). Drives e.g. emissive intensity / dissolve amount.
//      HONESTY: value-producing query; it does NOT mutate a live hf::material graph (no live render).
//   4. SubSequenceTrack — nested Timelines. Evaluating the parent at tick T evaluates each sub at the
//      LOCAL time SubLocalTime(sub, T) = clamp( fxmul(T - startTick, playRate), clampStart, clampEnd ).
//      PLAY-RATE + CLAMP CONVENTION (pinned): localRaw = (T - startTick) * playRate computed as the Q16.16
//      fxmul product (a (int64)a*b>>16 ARITHMETIC shift — truncates toward -inf, so a fractional playRate
//      can ALIAS on integer ticks; that truncation IS the deterministic convention). playRate = kOne is
//      1x, 2*kOne is DOUBLE SPEED. Before startTick (T < startTick) localRaw < 0 -> clamps to clampStart
//      (the sub holds its first frame); past the sub's end -> clamps to clampEnd (holds its last frame).
//      Nesting depth >= 2 (a sub may hold its own SubSequenceTrack; a kMaxSubDepth guard bounds recursion).
//
// THE MOAT: UE5 Sequencer's camera cuts / audio / material tracks / sub-sequences all evaluate on FLOAT
// playback timing — two machines diverge in the low bits and cannot bit-exactly agree on "which camera,
// which cue fired, what emissive value, which nested frame" at a given time. SQ2's are INTEGER ticks,
// identically, replayably, SCRUB-ably: SeekCine(N) == PlayCine to N (the load-bearing determinism claim,
// pinned in seq_tracks_test) — for BOTH the instantaneous state AND the fired-cue trace.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "seq/seq.h"   // S1–S5 core: fx/kOne/fxmul, ScalarTrack/SampleScalar (+easing), Sequence/
                        // SampleSequence, EventTrack/SampleEvents (the half-open window), DigestTrack
                        // (net::DigestBytes). seq.h stays BYTE-UNTOUCHED.

namespace hf::seq {

// ================================ 1. CAMERA-CUT TRACK ==============================================
// A cut = {tick, cameraId}. The track's cuts are STRICTLY ASCENDING by tick (sorted, no dupes) — the
// ScalarTrack/EventTrack invariant. ActiveCamera is PIECEWISE-CONSTANT: the camera of the LAST cut at or
// before t (a cut exactly ON a tick becomes active AT that tick, the `<=` convention). Before the first
// cut -> the caller's `defaultCam`. Deterministic integer scan (NO <algorithm>).
struct CameraCut {
    fx       tick     = 0;   // Q16.16 seconds (STRICTLY ASCENDING across the track)
    uint32_t cameraId = 0;   // which camera becomes active at/after this tick
};

struct CameraCutTrack {
    std::vector<CameraCut> cuts;   // ascending by tick; empty -> ActiveCamera always returns defaultCam
};

// ActiveCamera(tr, t, defaultCam): the last cut with tick <= t (cuts ascending). Empty / all-after-t ->
// defaultCam. Single cut c -> defaultCam for t < c.tick, c.cameraId for all t >= c.tick (pinned).
inline uint32_t ActiveCamera(const CameraCutTrack& tr, fx t, uint32_t defaultCam = 0) {
    uint32_t cam = defaultCam;
    for (const CameraCut& c : tr.cuts) {
        if (c.tick <= t) cam = c.cameraId;   // at/before t -> candidate (ascending scan keeps the LAST)
        else             break;               // first cut after t -> none remain (sorted)
    }
    return cam;
}

// ================================ 2. AUDIO TRACK ==================================================
// A cue = {tick, soundId, gain}. Cues are STRICTLY ASCENDING by tick. SampleAudioCues fires every cue in
// the HALF-OPEN window [tPrev, t) — the EXACT seq.h S3 SampleEvents convention (a tick boundary fires
// EXACTLY ONCE across consecutive sweeps; the AL1 / sampler-window standard). Ascending emission order.
// The composition lemma this buys (used by the scrub proof): the union of the consecutive per-tick
// windows [i*dt,(i+1)*dt) over i in [0,N) equals the single window [0, N*dt) — same fired SET, same order.
struct AudioCue {
    fx       tick    = 0;   // Q16.16 seconds (STRICTLY ASCENDING across the track)
    uint32_t soundId = 0;   // which one-shot / sound to trigger (routes to an AU2 audio_graph node)
    fx       gain    = 0;   // Q16.16 trigger gain (the caller maps this to a graph gain / velocity)
};

struct AudioTrack {
    std::vector<AudioCue> cues;   // ascending by tick
};

// SampleAudioCues(tr, tPrev, t): fire every cue whose tick is in [tPrev, t). Guard: t <= tPrev -> empty
// (no negative/empty window fires — the SampleEvents guard). Ascending index scan == ascending time.
inline std::vector<AudioCue> SampleAudioCues(const AudioTrack& tr, fx tPrev, fx t) {
    std::vector<AudioCue> out;
    if (t <= tPrev) return out;                    // empty/negative window fires nothing
    for (const AudioCue& q : tr.cues) {
        if (q.tick <  tPrev) continue;              // before the window -> skip
        if (q.tick >= t)     break;                 // at/after the window end -> done (sorted)
        out.push_back(q);
    }
    return out;
}

// ================================ 3. MATERIAL / PARAM TRACK =======================================
// A named material parameter animated by a seq.h ScalarTrack (SampleScalar + easing, reused VERBATIM).
// targetMaterialId/paramId identify WHICH parameter (e.g. an emissive-intensity or dissolve-amount slot);
// SampleMaterialParam returns its Q16.16 value at t. Multiple tracks are independent. HONEST: this is a
// value-producing query — it does NOT mutate a live material graph (that would need the float material
// system + a live render); the value is the deterministic thing SQ2 pins.
struct MaterialParamTrack {
    uint32_t    targetMaterialId = 0;   // which material this drives
    uint32_t    paramId          = 0;   // which named parameter on that material
    ScalarTrack curve;                  // the animated value (seq.h scalar keyframes + easing)
};

// SampleMaterialParam(tr, t): the seq.h scalar sample of the param curve (== that key AT a keyframe;
// eased between). Pinned equal to SampleScalar(tr.curve, t) by construction.
inline fx SampleMaterialParam(const MaterialParamTrack& tr, fx t) {
    return SampleScalar(tr.curve, t);
}

// ================================ 4. SUB-SEQUENCE NESTING =========================================
// A Timeline is the SQ2 composite: the seq.h scalar value-bus (Sequence) + a camera-cut track + an audio
// track + N material tracks + a sub-sequence track (each sub a non-owning pointer to another Timeline).
// Forward-declared so SubSequence can hold a Timeline*.
struct Timeline;

// SubSequence — a nested shot: at parent tick T evaluate `sub` at SubLocalTime(T). `sub` is a NON-OWNING
// pointer (the pointee must outlive; the fixtures use function-static storage so addresses are stable).
struct SubSequence {
    fx              startTick  = 0;      // parent tick at which this sub begins (sub-local t = 0 there)
    const Timeline* sub        = nullptr;// the nested timeline (non-owning; nullptr -> a no-op eval)
    fx              playRate   = kOne;   // Q16.16 rate: kOne == 1x, 2*kOne == double speed
    fx              clampStart = 0;      // sub local-time clamp lower (usually the sub's range start)
    fx              clampEnd   = 0;      // sub local-time clamp upper (usually the sub's range end)
};

struct SubSequenceTrack {
    std::vector<SubSequence> subs;
};

struct Timeline {
    Sequence                        scalars;   // seq.h S2 multi-track scalar value-bus (e.g. camera dolly)
    CameraCutTrack                  camera;     // which camera is active over the timeline
    AudioTrack                      audio;      // one-shot cues (fired over windows, not at an instant)
    std::vector<MaterialParamTrack> materials;  // named material-parameter animations
    SubSequenceTrack                subseq;      // nested shots
    uint32_t                        defaultCamera = 0;  // ActiveCamera before the first cut
};

// Recursion bound (a defensive guard against a malformed sub cycle; the fixtures nest to depth 2).
inline constexpr int kMaxSubDepth = 8;

// SubLocalTime(s, parentT): localRaw = (parentT - startTick) * playRate as the Q16.16 fxmul product
// (an (int64)a*b>>16 ARITHMETIC shift — truncates toward -inf; a fractional playRate can alias on integer
// ticks, which IS the deterministic convention), then CLAMPED to [clampStart, clampEnd]. parentT <
// startTick -> negative localRaw -> clampStart (hold first frame); past the end -> clampEnd (hold last).
inline fx SubLocalTime(const SubSequence& s, fx parentT) {
    const fx delta = parentT - s.startTick;      // may be negative (before the sub starts)
    fx local = fxmul(delta, s.playRate);          // Q16.16 * Q16.16 -> Q16.16 (truncating, deterministic)
    if (local < s.clampStart) local = s.clampStart;
    if (local > s.clampEnd)   local = s.clampEnd;
    return local;
}

// SubDepth(tl): the nesting depth of a Timeline's sub-sequences (0 = no subs; 1 = one level; ...). A pure
// recursive max over the sub tree, bounded by kMaxSubDepth (so a malformed cycle terminates deterministically).
inline int SubDepth(const Timeline& tl, int depth = 0) {
    if (depth >= kMaxSubDepth) return depth;
    int best = 0;
    for (const SubSequence& s : tl.subseq.subs) {
        const int d = (s.sub != nullptr) ? (1 + SubDepth(*s.sub, depth + 1)) : 1;
        if (d > best) best = d;
    }
    return best;
}

// ---- The instantaneous evaluation of a Timeline at a tick (camera + scalars + materials + subs) --------
// A recursive VALUE: the active camera, the sampled scalar bus, each material-param value, each sub's
// clamped local time, and each sub's own (recursive) evaluation. AUDIO is NOT here — cues fire over
// WINDOWS during playback (SampleAudioCues), not at an instant. Same (tl, t) -> byte-identical eval.
struct TimelineEval {
    fx                        time         = 0;   // the tick this was evaluated at
    uint32_t                  activeCamera = 0;   // ActiveCamera(camera, time, defaultCamera)
    std::vector<fx>           scalarBus;          // SampleSequence(scalars, time) (one fx per channel)
    std::vector<fx>           materialValues;     // one fx per material track
    std::vector<fx>           subLocal;           // clamped local time fed to each sub
    std::vector<TimelineEval> subs;               // recursive sub evaluations (depth-2+)
};

inline TimelineEval EvalTimeline(const Timeline& tl, fx t, int depth = 0) {
    TimelineEval e;
    e.time         = t;
    e.activeCamera = ActiveCamera(tl.camera, t, tl.defaultCamera);
    e.scalarBus    = SampleSequence(tl.scalars, t);
    e.materialValues.reserve(tl.materials.size());
    for (const MaterialParamTrack& m : tl.materials) e.materialValues.push_back(SampleMaterialParam(m, t));
    if (depth < kMaxSubDepth) {
        e.subLocal.reserve(tl.subseq.subs.size());
        e.subs.reserve(tl.subseq.subs.size());
        for (const SubSequence& s : tl.subseq.subs) {
            const fx local = SubLocalTime(s, t);
            e.subLocal.push_back(local);
            if (s.sub != nullptr) e.subs.push_back(EvalTimeline(*s.sub, local, depth + 1));
            else                  e.subs.push_back(TimelineEval{});
        }
    }
    return e;
}

// SerializeEval / DigestEval: HAND field-by-field serialization of a TimelineEval into a contiguous fx
// buffer (NEVER DigestBytes the struct — the vectors carry heap pointers; the seq.h S4/S5 discipline),
// then DigestTrack (net::DigestBytes). Sizes are embedded so a structural change (a sub added/removed)
// changes the digest. Recursive over the sub tree.
inline void SerializeEval(const TimelineEval& e, std::vector<fx>& buf) {
    buf.push_back(e.time);
    buf.push_back(static_cast<fx>(e.activeCamera));
    buf.push_back(static_cast<fx>(e.scalarBus.size()));
    for (fx v : e.scalarBus)      buf.push_back(v);
    buf.push_back(static_cast<fx>(e.materialValues.size()));
    for (fx v : e.materialValues) buf.push_back(v);
    buf.push_back(static_cast<fx>(e.subLocal.size()));
    for (fx v : e.subLocal)       buf.push_back(v);
    buf.push_back(static_cast<fx>(e.subs.size()));
    for (const TimelineEval& c : e.subs) SerializeEval(c, buf);
}
inline uint64_t DigestEval(const TimelineEval& e) {
    std::vector<fx> buf;
    SerializeEval(e, buf);
    return DigestTrack(buf);
}

// ================================ 5. PLAYBACK + SCRUB (THE MOAT) ==================================
// CineTrace — the complete state after driving the cinematic: the instantaneous eval at the final tick +
// the accumulated fired-cue trace. DigestCine pins both (hand field-by-field; cues by {tick,soundId,gain}).
struct CineTrace {
    TimelineEval          instant;   // EvalTimeline(tl, tEnd)
    std::vector<AudioCue> fired;     // cues fired over [0, tEnd), in ascending order
};

inline uint64_t DigestCine(const CineTrace& c) {
    std::vector<fx> buf;
    SerializeEval(c.instant, buf);
    buf.push_back(static_cast<fx>(c.fired.size()));
    for (const AudioCue& q : c.fired) {
        buf.push_back(q.tick);
        buf.push_back(static_cast<fx>(q.soundId));
        buf.push_back(q.gain);
    }
    return DigestTrack(buf);
}

// PlayCine(tl, dt, nTicks): PLAY forward tick by tick. Each tick i fires the half-open window
// [i*dt, (i+1)*dt) (int64 products, the seq.h SampleEventSweep pattern), accumulating cues; the final
// instantaneous state is EvalTimeline at tEnd = nTicks*dt. This is the "played to N" reference.
inline CineTrace PlayCine(const Timeline& tl, fx dt, uint32_t nTicks) {
    CineTrace c;
    for (uint32_t i = 0; i < nTicks; ++i) {
        const fx a = static_cast<fx>(static_cast<int64_t>(i)       * static_cast<int64_t>(dt));
        const fx b = static_cast<fx>(static_cast<int64_t>(i + 1u)  * static_cast<int64_t>(dt));
        const std::vector<AudioCue> fired = SampleAudioCues(tl.audio, a, b);
        for (const AudioCue& q : fired) c.fired.push_back(q);
    }
    const fx tEnd = static_cast<fx>(static_cast<int64_t>(nTicks) * static_cast<int64_t>(dt));
    c.instant = EvalTimeline(tl, tEnd);
    return c;
}

// SeekCine(tl, dt, nTicks): SEEK directly to tick N. The instantaneous state is EvalTimeline at tEnd
// (stateless — a pure function of time). The fired-cue trace is the SINGLE half-open window [0, tEnd) —
// which, by the SampleAudioCues composition lemma (== seq.h SampleEvents), equals the UNION of PlayCine's
// per-tick windows. THE LOAD-BEARING CLAIM: DigestCine(SeekCine(N)) == DigestCine(PlayCine to N) — the
// deterministic scrub UE5's float playback timing cannot do. Direction-independent (reverse seek yields
// the same state + the same [0,tEnd) cue set). Pinned in seq_tracks_test.
inline CineTrace SeekCine(const Timeline& tl, fx dt, uint32_t nTicks) {
    CineTrace c;
    const fx tEnd = static_cast<fx>(static_cast<int64_t>(nTicks) * static_cast<int64_t>(dt));
    c.fired   = SampleAudioCues(tl.audio, 0, tEnd);   // ONE window == the union of the per-tick windows
    c.instant = EvalTimeline(tl, tEnd);
    return c;
}

// ================================ 6. THE FIXED CINEMATIC FIXTURE ==================================
// A 2-shot cinematic over [0, 4s]:
//   * CAMERA-CUT: cam 0 for shot A [0,2s), cam 1 for shot B [2s,4s] (cut at t=2s).
//   * SCALARS: a 2-channel value-bus — a per-shot "dolly" ramp (ch0) + a "focus" pulse (ch1) — the
//     transform-animated-camera stand-in (a real transform track would drive the chosen camera's pose).
//   * MATERIAL: an emissive-intensity fade 0 -> 1 -> 0 over [0,2s,4s] (EaseInOutSine) on material 7 param 1.
//   * AUDIO: an intro cue at 0.5s (sound 100) + a STINGER at the cut t=2s (sound 200).
//   * SUB-SEQUENCE: a looping AMBIENT animation starting at 0.5s at DOUBLE speed (playRate 2x), clamped to
//     [0,1s]; that ambient sub itself holds a nested FLICKER sub (depth-2 nesting).
// The sub Timelines are function-static so their addresses are stable for the non-owning pointers.

// The depth-2 leaf: a fast FLICKER — one scalar channel oscillating, no further subs.
inline const Timeline& Sq2FlickerSub() {
    static const Timeline tl = [] {
        Timeline t;
        ScalarTrack flick;
        flick.times  = {0, kOne / 4, kOne / 2};
        flick.values = {0, kOne, 0};
        flick.easing = Easing::Linear;
        t.scalars.tracks = {flick};
        t.defaultCamera  = 5;                  // a distinct "flicker cam" so nesting shows in the eval
        return t;
    }();
    return tl;
}

// The depth-1 ambient sub: a gentle 1s-loop scalar + its own camera cut, holding the flicker leaf as a sub.
inline const Timeline& Sq2AmbientSub() {
    static const Timeline tl = [] {
        Timeline t;
        ScalarTrack amb;
        amb.times  = {0, kOne / 2, kOne};
        amb.values = {kOne / 4, kOne, kOne / 4};
        amb.easing = Easing::EaseInOutSine;
        t.scalars.tracks = {amb};
        t.camera.cuts = { CameraCut{0, 3}, CameraCut{kOne / 2, 4} };  // ambient sub switches cam 3->4
        t.defaultCamera = 3;
        // The depth-2 nested flicker: starts at 0.25s of the ambient local time, 1x, clamped to [0,0.5s].
        t.subseq.subs = { SubSequence{ kOne / 4, &Sq2FlickerSub(), kOne, 0, kOne / 2 } };
        return t;
    }();
    return tl;
}

// MakeSq2Cinematic(): the FIXED master timeline. Keep FIXED forever — seq_tracks_test pins its digests.
inline Timeline MakeSq2Cinematic() {
    Timeline tl;

    // Camera cuts: shot A cam 0, shot B cam 1 (cut exactly at t = 2s).
    tl.camera.cuts = { CameraCut{0, 0}, CameraCut{2 * kOne, 1} };
    tl.defaultCamera = 0;

    // Scalars: ch0 dolly ramp (0 -> 1 -> 2 over the two shots), ch1 focus pulse (0 -> 1 -> 0).
    ScalarTrack dolly;
    dolly.times  = {0, 2 * kOne, 4 * kOne};
    dolly.values = {0, kOne, 2 * kOne};
    dolly.easing = Easing::Linear;
    ScalarTrack focus;
    focus.times  = {0, 2 * kOne, 4 * kOne};
    focus.values = {0, kOne, 0};
    focus.easing = Easing::EaseInQuad;
    tl.scalars.tracks = {dolly, focus};

    // Material: emissive fade up then down (EaseInOutSine) on material 7, param 1.
    MaterialParamTrack emissive;
    emissive.targetMaterialId = 7;
    emissive.paramId          = 1;
    emissive.curve.times  = {0, 2 * kOne, 4 * kOne};
    emissive.curve.values = {0, kOne, 0};
    emissive.curve.easing = Easing::EaseInOutSine;
    tl.materials = {emissive};

    // Audio: intro cue at 0.5s, stinger at the cut (2s).
    tl.audio.cues = {
        AudioCue{kOne / 2, 100, kOne / 2},   // intro chime, half gain
        AudioCue{2 * kOne, 200, kOne},       // the cut stinger, full gain
    };

    // Sub-sequence: the ambient loop starts at 0.5s, DOUBLE speed, clamped to its [0,1s] range.
    tl.subseq.subs = { SubSequence{ kOne / 2, &Sq2AmbientSub(), 2 * kOne, 0, kOne } };

    return tl;
}

// Fixed drive parameters for the cinematic (30 fps over 4s -> 120 ticks; a scrub playhead at tick 75).
inline constexpr fx       kSq2Dt        = kOne / 30;   // 30 fps tick
inline constexpr uint32_t kSq2Ticks     = 120u;        // 4 seconds
inline constexpr uint32_t kSq2ScrubTick = 75u;         // 2.5s — inside shot B (the pinned scrub position)

// ================================ 7. THE SHOWCASE VIZ (strict-integer, NO shader) =================
// A multi-track editor-strip viz of the cinematic timeline, rendered by PURE INTEGER pixel writes into an
// RGBA8 buffer — shared VERBATIM by the Vulkan --sq2-cinematic-shot and the Metal --sq2-cinematic so the
// pixels are byte-identical cross-backend BY CONSTRUCTION (the pt1/we1 precedent). Rows: the camera-cut
// track (colored blocks per active camera), the audio track (cue tick-markers), the material track (a
// value curve polyline), the sub-sequence track (a nested bar + its depth-2 inner bar), a PLAYHEAD line,
// plus a small "viewport" cell whose framing marker + color track the active camera / dolly across the cut.
// NO float, NO <cmath> — every coordinate is an integer map of the Q16.16 runtime.

struct Sq2VizStats {
    uint32_t tracks   = 0;   // number of semantic tracks drawn (camera+audio+material+subseq = 4)
    uint32_t cuts     = 0;   // camera cuts
    uint32_t cues     = 0;   // audio cues
    int      subDepth = 0;   // sub-sequence nesting depth
    uint32_t ticks    = 0;   // ticks driven
    uint32_t width    = 0;
    uint32_t height   = 0;
    uint64_t pixDigest  = 0; // net::DigestBytes over the RGBA8 pixels (the cross-backend strict-zero proof)
    uint64_t evalDigest = 0; // DigestCine at the scrub tick (the runtime determinism digest)
};

// Fixed viz geometry (all integer).
inline constexpr int kSq2ImgW    = 560;
inline constexpr int kSq2ImgH    = 260;
inline constexpr int kSq2StripX0 = 40;    // strip left (t == tMin)
inline constexpr int kSq2StripW  = 400;   // strip width (spans [0, 4s])
inline constexpr int kSq2RowH    = 42;    // per-track row height
inline constexpr int kSq2RowGap  = 8;
inline constexpr int kSq2Row0Y   = 20;
inline constexpr int kSq2ViewX0  = 456;   // viewport cell left
inline constexpr int kSq2ViewW   = 96;

// MapTX: map a Q16.16 time t in [0, tMax] to a strip X (the seq_editor MapTimeToX integer form, inlined).
inline int Sq2MapTX(fx t, fx tMax) {
    if (tMax <= 0) return kSq2StripX0;
    if (t < 0)     t = 0;
    if (t > tMax)  t = tMax;
    const int64_t num = static_cast<int64_t>(t) * static_cast<int64_t>(kSq2StripW);
    const int64_t px  = (num + static_cast<int64_t>(tMax) / 2) / static_cast<int64_t>(tMax);
    return kSq2StripX0 + static_cast<int>(px);
}

// RenderSq2CinematicViz: fill `out` (RGBA8, kSq2ImgW x kSq2ImgH) and the stat block. Deterministic +
// pure integer -> two calls (and two backends) byte-identical.
inline void RenderSq2CinematicViz(std::vector<uint8_t>& out, Sq2VizStats& stats) {
    const int W = kSq2ImgW, H = kSq2ImgH;
    out.assign(static_cast<std::size_t>(W) * H * 4u, 0);

    auto px = [&](int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x < 0 || x >= W || y < 0 || y >= H) return;
        uint8_t* d = &out[(static_cast<std::size_t>(y) * W + x) * 4u];
        d[0] = r; d[1] = g; d[2] = b; d[3] = 255;
    };
    auto fillRect = [&](int x0, int y0, int w, int h, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y < y0 + h; ++y)
            for (int x = x0; x < x0 + w; ++x) px(x, y, r, g, b);
    };
    auto vline = [&](int x, int y0, int y1, uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y <= y1; ++y) px(x, y, r, g, b);
    };

    // Background: dark editor gray.
    fillRect(0, 0, W, H, 18, 20, 26);

    const Timeline tl = MakeSq2Cinematic();
    const fx tMax = static_cast<fx>(static_cast<int64_t>(kSq2Ticks) * static_cast<int64_t>(kSq2Dt));  // 4s
    const int rowY[4] = {
        kSq2Row0Y,
        kSq2Row0Y + 1 * (kSq2RowH + kSq2RowGap),
        kSq2Row0Y + 2 * (kSq2RowH + kSq2RowGap),
        kSq2Row0Y + 3 * (kSq2RowH + kSq2RowGap),
    };

    // Per-camera color palette (deterministic; cam id -> RGB).
    auto camColor = [](uint32_t cam, uint8_t& r, uint8_t& g, uint8_t& b) {
        switch (cam % 6u) {
            case 0: r = 46;  g = 160; b = 160; break;  // teal
            case 1: r = 220; g = 130; b = 50;  break;  // orange
            case 2: r = 120; g = 90;  b = 200; break;  // purple
            case 3: r = 90;  g = 180; b = 90;  break;  // green
            case 4: r = 200; g = 80;  b = 120; break;  // magenta
            default: r = 150; g = 150; b = 150; break; // gray
        }
    };

    // ---- Row 0: CAMERA-CUT track — colored blocks per active-camera segment. ------------------------
    // Sample the active camera per pixel column across the strip; color the row band by that camera.
    {
        const int y0 = rowY[0];
        fillRect(kSq2StripX0, y0, kSq2StripW, kSq2RowH, 30, 32, 40);  // lane bg
        for (int c = 0; c <= kSq2StripW; ++c) {
            const fx t = static_cast<fx>((static_cast<int64_t>(c) * static_cast<int64_t>(tMax)) / kSq2StripW);
            const uint32_t cam = ActiveCamera(tl.camera, t, tl.defaultCamera);
            uint8_t r, g, b; camColor(cam, r, g, b);
            for (int y = y0 + 4; y < y0 + kSq2RowH - 4; ++y) px(kSq2StripX0 + c, y, r, g, b);
        }
        // Cut boundary markers (bright vertical lines at each cut tick).
        for (const CameraCut& cut : tl.camera.cuts)
            vline(Sq2MapTX(cut.tick, tMax), y0, y0 + kSq2RowH - 1, 240, 240, 120);
    }

    // ---- Row 1: AUDIO track — cue tick-markers (vertical ticks + a small diamond head). --------------
    {
        const int y0 = rowY[1];
        fillRect(kSq2StripX0, y0, kSq2StripW, kSq2RowH, 30, 32, 40);
        for (const AudioCue& q : tl.audio.cues) {
            const int x = Sq2MapTX(q.tick, tMax);
            // gain -> marker height (Q16.16 gain in [0,kOne] folded into the row band).
            int hgt = static_cast<int>((static_cast<int64_t>(q.gain) * (kSq2RowH - 10)) / kOne);
            if (hgt < 2) hgt = 2;
            vline(x, y0 + kSq2RowH - 4 - hgt, y0 + kSq2RowH - 4, 90, 200, 240);
            // diamond head
            const int hy = y0 + kSq2RowH - 4 - hgt;
            for (int dy = -3; dy <= 3; ++dy) {
                const int span = 3 - (dy < 0 ? -dy : dy);
                for (int dx = -span; dx <= span; ++dx) px(x + dx, hy + dy, 240, 230, 90);
            }
        }
    }

    // ---- Row 2: MATERIAL track — the emissive value curve as an integer polyline. --------------------
    {
        const int y0 = rowY[2];
        fillRect(kSq2StripX0, y0, kSq2StripW, kSq2RowH, 30, 32, 40);
        const MaterialParamTrack& m = tl.materials[0];
        // value range from the curve keys.
        fx vMin = m.curve.values.empty() ? 0 : m.curve.values[0];
        fx vMax = vMin;
        for (fx v : m.curve.values) { if (v < vMin) vMin = v; if (v > vMax) vMax = v; }
        const int bandTop = y0 + 4, bandH = kSq2RowH - 8;
        auto valY = [&](fx v) -> int {
            if (vMax <= vMin) return bandTop + bandH / 2;
            if (v < vMin) v = vMin; if (v > vMax) v = vMax;
            const int64_t span = static_cast<int64_t>(vMax) - static_cast<int64_t>(vMin);
            const int64_t off  = ((static_cast<int64_t>(vMax) - static_cast<int64_t>(v)) * bandH + span / 2) / span;
            return bandTop + static_cast<int>(off);
        };
        int prevX = -1, prevY = -1;
        for (int c = 0; c <= kSq2StripW; ++c) {
            const fx t = static_cast<fx>((static_cast<int64_t>(c) * static_cast<int64_t>(tMax)) / kSq2StripW);
            const int x = kSq2StripX0 + c;
            const int y = valY(SampleMaterialParam(m, t));
            px(x, y, 240, 120, 160);
            if (prevX >= 0) {
                const int lo = prevY < y ? prevY : y, hi = prevY < y ? y : prevY;  // fill vertical gap
                for (int yy = lo; yy <= hi; ++yy) px(x, yy, 240, 120, 160);
            }
            prevX = x; prevY = y;
        }
    }

    // ---- Row 3: SUB-SEQUENCE track — a nested bar (the sub's active span) + its depth-2 inner bar. ---
    {
        const int y0 = rowY[3];
        fillRect(kSq2StripX0, y0, kSq2StripW, kSq2RowH, 30, 32, 40);
        for (const SubSequence& s : tl.subseq.subs) {
            // The sub's parent-time span: from startTick until local time reaches clampEnd, i.e.
            // duration = clampEnd/playRate (parent seconds). Pure integer via fxdiv-like folding.
            const fx dur = (s.playRate != 0) ? fxdiv(s.clampEnd - s.clampStart, s.playRate) : 0;
            const int bx0 = Sq2MapTX(s.startTick, tMax);
            const int bx1 = Sq2MapTX(s.startTick + dur, tMax);
            fillRect(bx0, y0 + 6, (bx1 - bx0 > 1 ? bx1 - bx0 : 1), kSq2RowH - 20, 80, 150, 110);
            // depth-2 inner bar: the nested flicker sub, mapped through the parent's rate.
            if (s.sub != nullptr && !s.sub->subseq.subs.empty()) {
                const SubSequence& inner = s.sub->subseq.subs[0];
                // inner starts at parent time: startTick + inner.startTick/playRate.
                const fx innerStartParent = s.startTick + ((s.playRate != 0) ? fxdiv(inner.startTick, s.playRate) : 0);
                const fx innerDurLocal    = inner.clampEnd - inner.clampStart;
                const fx innerDurParent   = (s.playRate != 0) ? fxdiv(innerDurLocal, s.playRate) : 0;
                const int ix0 = Sq2MapTX(innerStartParent, tMax);
                const int ix1 = Sq2MapTX(innerStartParent + innerDurParent, tMax);
                fillRect(ix0, y0 + 16, (ix1 - ix0 > 1 ? ix1 - ix0 : 1), kSq2RowH - 30, 200, 200, 90);
            }
        }
    }

    // ---- PLAYHEAD: a vertical line at the scrub tick across all rows. --------------------------------
    const fx scrubT = static_cast<fx>(static_cast<int64_t>(kSq2ScrubTick) * static_cast<int64_t>(kSq2Dt));
    const int phX = Sq2MapTX(scrubT, tMax);
    vline(phX, kSq2Row0Y - 4, rowY[3] + kSq2RowH + 2, 240, 80, 80);

    // ---- VIEWPORT cell: the active camera's framing at the scrub tick. -------------------------------
    // A box tinted by the active camera color; a framing marker whose X tracks the dolly value (ch0).
    {
        const TimelineEval e = EvalTimeline(tl, scrubT);
        uint8_t r, g, b; camColor(e.activeCamera, r, g, b);
        const int vy0 = kSq2Row0Y, vh = rowY[3] + kSq2RowH - kSq2Row0Y;
        fillRect(kSq2ViewX0, vy0, kSq2ViewW, vh, 24, 26, 32);
        // camera-tinted border
        for (int x = kSq2ViewX0; x < kSq2ViewX0 + kSq2ViewW; ++x) { px(x, vy0, r, g, b); px(x, vy0 + vh - 1, r, g, b); }
        for (int y = vy0; y < vy0 + vh; ++y) { px(kSq2ViewX0, y, r, g, b); px(kSq2ViewX0 + kSq2ViewW - 1, y, r, g, b); }
        // dolly value (ch0, ~[0,2]) -> marker X within the cell interior.
        const fx dolly = e.scalarBus.empty() ? 0 : e.scalarBus[0];
        int64_t mx = (static_cast<int64_t>(dolly) * (kSq2ViewW - 20)) / (2 * kOne);
        if (mx < 0) mx = 0; if (mx > kSq2ViewW - 20) mx = kSq2ViewW - 20;
        const int cx = kSq2ViewX0 + 10 + static_cast<int>(mx);
        const int cy = vy0 + vh / 2;
        fillRect(cx - 6, cy - 6, 12, 12, r, g, b);
        // focus pulse (ch1) -> a small brightness cross above/below the marker.
        vline(cx, cy - 14, cy - 8, 230, 230, 230);
        vline(cx, cy + 8,  cy + 14, 230, 230, 230);
    }

    // ---- Stats. --------------------------------------------------------------------------------------
    stats.tracks     = 4;
    stats.cuts       = static_cast<uint32_t>(tl.camera.cuts.size());
    stats.cues       = static_cast<uint32_t>(tl.audio.cues.size());
    stats.subDepth   = SubDepth(tl);
    stats.ticks      = kSq2Ticks;
    stats.width      = static_cast<uint32_t>(W);
    stats.height     = static_cast<uint32_t>(H);
    stats.pixDigest  = hf::net::DigestBytes(out.data(), out.size());
    stats.evalDigest = DigestCine(SeekCine(tl, kSq2Dt, kSq2ScrubTick));
}

}  // namespace hf::seq
