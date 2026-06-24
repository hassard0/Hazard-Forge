// Slice RP1 — Demo-file format + RECORD (flagship #28 REPLAY/DEMO, 1st/6 — beachhead).
//
// A demo file does NOT store state-over-time; it stores the CAUSAL SEED of a deterministic session —
// (seed, the initial-world snapshot, the per-tick INPUT stream, the per-tick digest trace) — so a later
// slice re-derives the byte-identical world by re-running the EXISTING deterministic Step. RP1 establishes
// the on-disk FORMAT + the RECORDER and PINS the demo file's hash so the byte layout is proven identical
// on Windows/MSVC and Mac/clang.
//
// SELF-CONTAINED ON PURPOSE: this header includes ONLY <cstdint>/<cstddef>/<vector> and "net/session.h"
// (itself self-contained) so replay_test.cpp compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/replay_test.cpp` on the Mac — the cheap cross-platform
// proof. NO fpx / RHI / GPU / <functional> / <cmath> / <fstream> / float. Pure-CPU INTEGER.
//
// The serialization discipline mirrors audio/wav.cpp VERBATIM: every multi-byte field is hand-serialized
// little-endian; we NEVER memcpy a host struct (which would embed host endianness/padding and break the
// Mac hash). The driver's World/Input serializers (supplied as template callables) MUST be hand-LE too.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "net/session.h"

namespace hf::replay {

// --- Little-endian byte appenders (mirror audio/wav.cpp:11-26 discipline; PutU64 = the 8-byte PutU32) ---
inline void PutU32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(static_cast<uint8_t>(v & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}
inline void PutU64(std::vector<uint8_t>& b, uint64_t v) {
    b.push_back(static_cast<uint8_t>(v & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 32) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 40) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 48) & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 56) & 0xFF));
}
// Append `n` bytes from `p` in address order (the caller has already laid them out LE).
inline void PutBytes(std::vector<uint8_t>& b, const void* p, std::size_t n) {
    const auto* s = static_cast<const uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i) b.push_back(s[i]);
}

// --- Matching readers for the header round-trip test (pure of side effects, LE) ---------------------
inline uint32_t GetU32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t GetU64(const uint8_t* p) {
    return static_cast<uint64_t>(p[0]) | (static_cast<uint64_t>(p[1]) << 8) |
           (static_cast<uint64_t>(p[2]) << 16) | (static_cast<uint64_t>(p[3]) << 24) |
           (static_cast<uint64_t>(p[4]) << 32) | (static_cast<uint64_t>(p[5]) << 40) |
           (static_cast<uint64_t>(p[6]) << 48) | (static_cast<uint64_t>(p[7]) << 56);
}

// --- DemoHeader — the fixed-layout file header (serialized field-by-field LE, NEVER memcpy'd) -------
struct DemoHeader {
    char     magic[8];          // "HFDEMO\0\0" — 8 bytes, exact
    uint32_t version;           // = kDemoVersion (start at 1)
    uint32_t seed;              // the session seed (RP1: a fixed constant for the ToyA demo)
    uint32_t tickCount;         // number of recorded ticks
    uint32_t keyframeInterval;  // ticks between keyframes (RP3); 0 = no keyframes. A HEADER FIELD on purpose.
    uint32_t worldByteLen;      // length of the initial-snapshot blob that follows
    uint32_t inputByteLen;      // length of the serialized input stream that follows
    uint32_t keyframeByteLen;   // RP3: length of the keyframe section that follows the inputs (0 = none)
};
inline constexpr uint32_t kDemoVersion = 1;
inline constexpr char     kDemoMagic[8] = {'H', 'F', 'D', 'E', 'M', 'O', '\0', '\0'};

// --- Recorder<World,Input> — captures one deterministic session's causal seed ----------------------
template <class World, class Input>
struct Recorder {
    uint32_t              seed = 0;
    World                 initial{};      // the world snapshot AS OF tick 0
    net::InputRing<Input> ring;           // the per-tick input stream (reuse net::InputRing)
    std::vector<uint64_t> digestTrace;    // per-tick digest trace (reuse net::DigestTrace)
    uint32_t              tickCount = 0;
    uint32_t              keyframeInterval = 0;  // RP3: ticks between keyframes (0 = none captured)

    // RP3: a full driver-serialized world snapshot AS OF `tick` (= world after [0,tick) ticks applied).
    struct Keyframe { uint32_t tick; std::vector<uint8_t> worldBytes; };
    std::vector<Keyframe> keyframes;      // captured at ticks 0, K, 2K, ... (driver-serialized bytes)
};

// Record `ticks` of a session: capture seed, the initial world, the input ring, the per-tick digest trace,
// AND (RP3) a keyframe every `keyframeInterval` ticks. Walk the session ONE tick at a time (we need the
// intermediate worlds for keyframes) in ONE consistent loop: capture a keyframe whenever T % interval == 0
// BEFORE stepping tick T (so keyframe@0 == initial world, keyframe@K == world after K ticks), then Advance,
// then record digest AFTER. digestTrace stays the per-tick-AFTER trace (length == ticks), bit-identical to
// net::DigestTrace. When keyframeInterval == 0, capture NO keyframes. Deterministic of
// (seed, initial, ring, ticks, keyframeInterval, step, digest, serWorld) alone.
template <class World, class Input, class StepFn, class DigestFn, class SerWorldFn>
Recorder<World, Input> RecordSession(uint32_t seed, World initial, const net::InputRing<Input>& ring,
                                     uint32_t ticks, uint32_t keyframeInterval,
                                     StepFn step, DigestFn digest, SerWorldFn serWorld) {
    Recorder<World, Input> rec;
    rec.seed             = seed;
    rec.initial          = initial;
    rec.ring             = ring;
    rec.tickCount        = ticks;
    rec.keyframeInterval = keyframeInterval;

    net::Session<World, Input> s;
    s.world = initial;
    s.ring  = ring;     // a COPY — the caller's ring is untouched (matches RunLockstep / DigestTrace)
    s.tick  = 0;
    rec.digestTrace.reserve(static_cast<std::size_t>(ticks));
    for (uint32_t T = 0; T < ticks; ++T) {
        if (keyframeInterval > 0 && (T % keyframeInterval) == 0) {
            typename Recorder<World, Input>::Keyframe kf;
            kf.tick       = T;                 // world AS OF tick T (BEFORE stepping T)
            kf.worldBytes = serWorld(s.world);
            rec.keyframes.push_back(std::move(kf));
        }
        net::Advance(s, step);                 // NS1 deterministic transition
        rec.digestTrace.push_back(digest(s.world));  // digest AFTER the tick
    }
    return rec;
}

// --- EncodeDemo — serialize a Recorder to a byte buffer, hand-LE, deterministic, byte-exact ---------
// Layout (ALL fields LE, in this exact order):
//   magic(8) | version(u32) | seed(u32) | tickCount(u32) | keyframeInterval(u32) | worldByteLen(u32) |
//   inputByteLen(u32) | keyframeByteLen(u32) | <worldBytes> | <inputBytes> | <keyframeSection>
// Header is now 8 + 7*4 = 36 bytes. The keyframe section (only when keyframeByteLen > 0) is:
//   keyframeCount(u32), then per keyframe: tick(u32), kfWorldByteLen(u32), <kfWorldBytes>.
// `serWorld(const World&)` and `serInputRing(const net::InputRing<Input>&)` MUST themselves be hand-LE
// (field-by-field, no struct memcpy) so the bytes are identical on MSVC + Apple clang.
template <class World, class Input, class SerWorldFn, class SerInputFn>
std::vector<uint8_t> EncodeDemo(const Recorder<World, Input>& rec, SerWorldFn serWorld,
                                SerInputFn serInputRing) {
    const std::vector<uint8_t> worldBytes = serWorld(rec.initial);
    const std::vector<uint8_t> inputBytes = serInputRing(rec.ring);

    // Build the keyframe section first so we know its total byte length (0 when there are no keyframes).
    std::vector<uint8_t> kfSection;
    if (!rec.keyframes.empty()) {
        PutU32(kfSection, static_cast<uint32_t>(rec.keyframes.size()));    // keyframeCount
        for (const auto& kf : rec.keyframes) {
            PutU32(kfSection, kf.tick);                                    // tick
            PutU32(kfSection, static_cast<uint32_t>(kf.worldBytes.size())); // kfWorldByteLen
            PutBytes(kfSection, kf.worldBytes.data(), kf.worldBytes.size()); // <kfWorldBytes>
        }
    }
    const uint32_t keyframeByteLen = static_cast<uint32_t>(kfSection.size());

    std::vector<uint8_t> out;
    PutBytes(out, kDemoMagic, 8);                                          // magic(8)
    PutU32(out, kDemoVersion);                                             // version
    PutU32(out, rec.seed);                                                 // seed
    PutU32(out, rec.tickCount);                                            // tickCount
    PutU32(out, rec.keyframeInterval);                                     // keyframeInterval (RP3)
    PutU32(out, static_cast<uint32_t>(worldBytes.size()));                 // worldByteLen
    PutU32(out, static_cast<uint32_t>(inputBytes.size()));                 // inputByteLen
    PutU32(out, keyframeByteLen);                                          // keyframeByteLen (RP3)
    PutBytes(out, worldBytes.data(), worldBytes.size());                   // <worldBytes>
    PutBytes(out, inputBytes.data(), inputBytes.size());                   // <inputBytes>
    PutBytes(out, kfSection.data(), kfSection.size());                     // <keyframeSection>
    return out;
}

// ============================ RP2: PLAYBACK — DecodeDemo + Replay ====================================
// The inverse of RP1's RECORD half: decode a demo's bytes back into its CAUSAL SEED (header + initial-
// world snapshot + input ring), then REPLAY it by re-running the EXISTING deterministic Step over the
// decoded ring — proving the replayed world is BIT-IDENTICAL to the live session (re-derived, NOT
// interpolated). The format is FROZEN (RP1's pinned file hash stays valid); RP2 adds Demo/DecodeDemo/
// ReplayResult/Replay BELOW the RP1 code, append-only. Pure-CPU INTEGER, header stays self-contained.

// --- Demo<World,Input> — the decoded demo (the inverse of EncodeDemo's product) ---------------------
template <class World, class Input>
struct Demo {
    DemoHeader            header{};   // fields read back out of the file (magic/version/seed/tickCount/...)
    World                 initial{};  // the deserialized initial-world snapshot
    net::InputRing<Input> ring;       // the deserialized per-tick input stream

    // RP3: the decoded keyframe table — each is a full world snapshot AS OF `tick` (the seek substrate).
    struct DecodedKeyframe { uint32_t tick; World world; };
    std::vector<DecodedKeyframe> keyframes;
};

// --- DecodeDemo — parse demo bytes back into a Demo<World,Input>, hand-LE (GetU32 + driver deser) ----
// Layout is EncodeDemo's exact inverse: magic(8) then 7 u32 fields (version, seed, tickCount,
// keyframeInterval, worldByteLen, inputByteLen, keyframeByteLen) = 8 + 7*4 = 36-byte header, then the
// world blob at offset 36, the input blob at 36 + worldByteLen, and the keyframe section at
// 36 + worldByteLen + inputByteLen (parsed only when keyframeByteLen > 0). (Re-derived by COUNTING
// EncodeDemo's 7 PutU32 calls after the 8-byte magic — do NOT trust comment arithmetic.) The driver
// supplies the inverse serializers: deserWorld(const uint8_t* p, uint32_t len) -> World and
// deserInputRing(const uint8_t* p, uint32_t len) -> net::InputRing<Input>. Validates magic == kDemoMagic
// and version == kDemoVersion (RP2 only feeds valid demos; the corruption path is RP6).
template <class World, class Input, class DeserWorldFn, class DeserInputFn>
Demo<World, Input> DecodeDemo(const std::vector<uint8_t>& bytes, DeserWorldFn deserWorld,
                              DeserInputFn deserInputRing) {
    Demo<World, Input> demo;
    const uint8_t* p = bytes.data();

    // magic(8) — read back in PutBytes (address) order.
    for (int i = 0; i < 8; ++i) demo.header.magic[i] = static_cast<char>(p[i]);
    demo.header.version          = GetU32(p + 8);    // offset  8
    demo.header.seed             = GetU32(p + 12);   // offset 12
    demo.header.tickCount        = GetU32(p + 16);   // offset 16
    demo.header.keyframeInterval = GetU32(p + 20);   // offset 20
    demo.header.worldByteLen     = GetU32(p + 24);   // offset 24
    demo.header.inputByteLen     = GetU32(p + 28);   // offset 28
    demo.header.keyframeByteLen  = GetU32(p + 32);   // offset 32
    // header is 8 + 7*4 = 36 bytes; world blob @ 36, input blob @ 36+worldByteLen, kf section @ 36+world+input.

    // Validate magic == kDemoMagic and version == kDemoVersion (keep it simple — RP6 owns corruption).
    bool magicOk = true;
    for (int i = 0; i < 8; ++i)
        if (demo.header.magic[i] != kDemoMagic[i]) magicOk = false;
    if (!magicOk || demo.header.version != kDemoVersion) return demo;  // empty/default Demo on mismatch

    const uint32_t worldOff = 36u;
    const uint32_t inputOff = 36u + demo.header.worldByteLen;
    demo.initial = deserWorld(p + worldOff, demo.header.worldByteLen);
    demo.ring    = deserInputRing(p + inputOff, demo.header.inputByteLen);

    // RP3: parse the keyframe section (only present when keyframeByteLen > 0).
    if (demo.header.keyframeByteLen > 0) {
        const uint8_t* kp = p + inputOff + demo.header.inputByteLen;  // section base @ 36+world+input
        std::size_t off = 0;
        const uint32_t keyframeCount = GetU32(kp + off); off += 4;
        demo.keyframes.reserve(static_cast<std::size_t>(keyframeCount));
        for (uint32_t i = 0; i < keyframeCount; ++i) {
            typename Demo<World, Input>::DecodedKeyframe dk;
            dk.tick                     = GetU32(kp + off); off += 4;
            const uint32_t kfWorldLen   = GetU32(kp + off); off += 4;
            dk.world                    = deserWorld(kp + off, kfWorldLen); off += kfWorldLen;
            demo.keyframes.push_back(std::move(dk));
        }
    }
    return demo;
}

// --- ReplayResult — the playback product: the final digest + the re-derived per-tick trace ----------
template <class World, class Input>
struct ReplayResult {
    uint64_t              finalDigest = 0;
    std::vector<uint64_t> trace;          // digest(world) after each of header.tickCount ticks
};

// --- Replay — restore the initial snapshot and Advance for tickCount ticks over the decoded ring ----
// Playback IS the CatchUp/RunLockstep body seeded from the initial world: net::Session s; s.world =
// demo.initial; s.ring = demo.ring; s.tick = 0; then loop demo.header.tickCount times calling net::Advance
// (the NS1 transition) and pushing digest(s.world) into trace; finalDigest = digest(s.world). The result
// is BIT-IDENTICAL to a live net::RunLockstep / net::DigestTrace (re-derived, not interpolated).
template <class World, class Input, class StepFn, class DigestFn>
ReplayResult<World, Input> Replay(const Demo<World, Input>& demo, StepFn step, DigestFn digest) {
    net::Session<World, Input> s;
    s.world = demo.initial;
    s.ring  = demo.ring;
    s.tick  = 0;
    ReplayResult<World, Input> r;
    r.trace.reserve(static_cast<std::size_t>(demo.header.tickCount));
    for (uint32_t t = 0; t < demo.header.tickCount; ++t) {
        net::Advance(s, step);             // NS1 deterministic transition
        r.trace.push_back(digest(s.world)); // record the digest AFTER the tick
    }
    r.finalDigest = digest(s.world);
    return r;
}

// ============================ RP4: SEEK to an arbitrary tick =========================================
// The headline timeline capability built on RP3's keyframes: SEEK(N) — jump to ANY tick N by restoring
// the NEAREST keyframe at-or-before N and replaying the input tail [keyframeTick, N) forward. Seeking is
// EXACT re-derivation (NOT interpolation): Seek(demo, N) is byte-identical to having run the live session
// for N ticks. The win over replaying-from-0 is COST — from the nearest keyframe you replay at most
// keyframeInterval-1 ticks instead of N. This IS net::CatchUp (restore a confirmed snapshot, replay the
// tail) — the SAME primitive that proves late-join == bit-identical at NS6 — so Seek REUSES it verbatim
// rather than hand-rolling the replay loop. Append-only below RP3; header stays self-contained.

// --- SeekResult — the sought world + its digest + the seek bookkeeping (which keyframe, how far) -----
template <class World, class Input>
struct SeekResult {
    World    world{};            // the world AS OF toTick (bit-identical to live-at-toTick)
    uint64_t digest = 0;         // digest(world)
    uint32_t keyframeTick = 0;   // the keyframe we restored from (the nearest <= toTick; 0 if none)
    uint32_t replayedTicks = 0;  // how many ticks we replayed forward (toTick - keyframeTick) — seek cost
};

// --- NearestKeyframeAtOrBefore — index of the LAST decoded keyframe with tick <= toTick, or -1 if none.
// demo.keyframes is sorted ascending by tick (DecodeDemo preserves EncodeDemo's 0,K,2K,... order), so a
// linear scan keeping the last qualifying index is the simplest deterministic form. Returns -1 when the
// table is empty or every keyframe.tick > toTick (then Seek falls back to the initial world at tick 0).
template <class World, class Input>
inline int NearestKeyframeAtOrBefore(const Demo<World, Input>& demo, uint32_t toTick) {
    int best = -1;
    for (std::size_t i = 0; i < demo.keyframes.size(); ++i)
        if (demo.keyframes[i].tick <= toTick) best = static_cast<int>(i);
    return best;
}

// --- Seek — restore the nearest keyframe at-or-before toTick, then CatchUp forward to toTick. ---------
// 1. Clamp toTick to header.tickCount (seeking past the end returns the FINAL world).
// 2. Pick the nearest keyframe at-or-before toTick (the LAST keyframes[i] with tick <= toTick). If none
//    (empty table / keyframeInterval 0 / all > toTick), fall back to base = demo.initial, keyframeTick = 0.
// 3. Restore + replay forward = net::CatchUp: build a JoinSnapshot{tick=keyframeTick, world=base} and
//    CatchUp(snap, toTick, demo.ring, step) — steps every tick in [keyframeTick, toTick) over the ring.
// 4. digest = digest(world); replayedTicks = toTick - keyframeTick. Bit-identical to live-at-toTick.
template <class World, class Input, class StepFn, class DigestFn>
SeekResult<World, Input> Seek(const Demo<World, Input>& demo, uint32_t toTick, StepFn step, DigestFn digest) {
    // (1) Clamp past-the-end to the final tick — seeking beyond the demo returns the final world.
    if (toTick > demo.header.tickCount) toTick = demo.header.tickCount;

    // (2) Nearest keyframe at-or-before toTick; fall back to the initial world at tick 0 if none.
    World    base         = demo.initial;
    uint32_t keyframeTick = 0;
    const int kf = NearestKeyframeAtOrBefore(demo, toTick);
    if (kf >= 0) {
        base         = demo.keyframes[static_cast<std::size_t>(kf)].world;
        keyframeTick = demo.keyframes[static_cast<std::size_t>(kf)].tick;
    }

    // (3) Restore + replay the tail forward — this IS net::CatchUp (NS6), reused verbatim.
    net::JoinSnapshot<World> snap{ keyframeTick, base };
    SeekResult<World, Input> r;
    r.world         = net::CatchUp(snap, toTick, demo.ring, step);
    r.keyframeTick  = keyframeTick;
    r.replayedTicks = toTick - keyframeTick;  // toTick >= keyframeTick always (keyframeTick <= toTick)
    r.digest        = digest(r.world);        // (4)
    return r;
}

// ============================ RP5: SCRUB + variable-speed playback ===================================
// The user-facing TIMELINE built on RP4's Seek: a Player with a CURRENT position you can SCRUB forward
// and backward at VARIABLE SPEED (1x/2x/0.5x/reverse). The determinism rule: speed changes only WHEN you
// observe the world, never WHAT the world computes. Forward scrubbing steps from the CACHED world (cheap —
// net::CatchUp from currentTick, sharing Seek's exact replay body); backward scrubbing RE-SEEKS (the sim
// has no inverse) so the PATH taken to reach tick N never perturbs the world AT tick N (path-independence /
// no drift). Append-only below RP4; header stays self-contained (int64_t lives in <cstdint>). Pure-CPU
// INTEGER — variable "speed" is an integer tick STRIDE (the caller chooses when to call ScrubBy), no
// float/clock/RNG.

// --- Player<World,Input> — a timeline cursor over a decoded demo (caches the world AT currentTick). ----
template <class World, class Input>
struct Player {
    const Demo<World, Input>* demo = nullptr;  // the decoded demo (non-owning; demo outlives the player)
    uint32_t currentTick = 0;                  // the cursor position (0 .. demo->header.tickCount)
    World    world{};                          // the world AS OF currentTick (cached)
    uint64_t digest = 0;                       // digest(world) at currentTick
};

// --- MakePlayer — construct a Player positioned at tick 0 (the initial world). -----------------------
template <class World, class Input, class DigestFn>
Player<World, Input> MakePlayer(const Demo<World, Input>& demo, DigestFn digest) {
    Player<World, Input> p;
    p.demo        = &demo;
    p.currentTick = 0;
    p.world       = demo.initial;
    p.digest      = digest(p.world);
    return p;
}

// --- ScrubTo — move the cursor to an ABSOLUTE target tick (forward or backward), update cache. --------
// Clamp target to [0, demo->header.tickCount]. FORWARD (target >= currentTick): step from the CACHED
// p.world over [currentTick, target) via net::CatchUp seeded with JoinSnapshot{currentTick, p.world} — so
// forward-scrub shares Seek's EXACT replay body (the make-or-break: bit-identical to Seek(*demo, target)).
// BACKWARD (target < currentTick): the sim has no inverse, so RE-SEEK — Seek(*demo, target) (restore the
// nearest keyframe <= target + replay forward) and take its world. Updates p.currentTick/world/digest.
template <class World, class Input, class StepFn, class DigestFn>
void ScrubTo(Player<World, Input>& p, uint32_t target, StepFn step, DigestFn digest) {
    if (target > p.demo->header.tickCount) target = p.demo->header.tickCount;  // clamp past-the-end
    if (target >= p.currentTick) {
        // FORWARD: step from the cached world — this IS net::CatchUp (Seek's tail body), seeded here.
        net::JoinSnapshot<World> snap{ p.currentTick, p.world };
        p.world = net::CatchUp(snap, target, p.demo->ring, step);
    } else {
        // BACKWARD: re-Seek (no inverse) — restore nearest keyframe + replay forward to target.
        const SeekResult<World, Input> r = Seek(*p.demo, target, step, digest);
        p.world = r.world;
    }
    p.currentTick = target;
    p.digest      = digest(p.world);
}

// --- ScrubBy — relative scrub by a SIGNED delta (the variable-speed primitive). ----------------------
// A 2x-forward player calls ScrubBy(+2) per UI frame; 0.5x calls ScrubBy(+1) every other frame; reverse
// calls a negative delta. Implemented as ScrubTo with a saturating clamp at 0 and tickCount, computed in
// int64 BEFORE the uint32 cast (so negative deltas / over-scroll at the ends saturate, never wrap = no-op).
template <class World, class Input, class StepFn, class DigestFn>
void ScrubBy(Player<World, Input>& p, int64_t delta, StepFn step, DigestFn digest) {
    int64_t target = static_cast<int64_t>(p.currentTick) + delta;
    if (target < 0) target = 0;
    const int64_t hi = static_cast<int64_t>(p.demo->header.tickCount);
    if (target > hi) target = hi;
    ScrubTo(p, static_cast<uint32_t>(target), step, digest);
}

}  // namespace hf::replay
