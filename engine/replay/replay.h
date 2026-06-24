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
    uint32_t keyframeInterval;  // RP1: 0 (no keyframes yet — RP3 introduces them). A HEADER FIELD on purpose.
    uint32_t worldByteLen;      // length of the initial-snapshot blob that follows
    uint32_t inputByteLen;      // length of the serialized input stream that follows
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
};

// Record `ticks` of a session: capture seed, the initial world, the input ring, and the per-tick digest
// trace (via net::DigestTrace — the SAME confirmed-state checksum stream a peer emits). Deterministic of
// (seed, initial, ring, ticks, step, digest) alone.
template <class World, class Input, class StepFn, class DigestFn>
Recorder<World, Input> RecordSession(uint32_t seed, World initial, const net::InputRing<Input>& ring,
                                     uint32_t ticks, StepFn step, DigestFn digest) {
    Recorder<World, Input> rec;
    rec.seed        = seed;
    rec.initial     = initial;
    rec.ring        = ring;
    rec.tickCount   = ticks;
    rec.digestTrace = net::DigestTrace<World, Input>(initial, ring, ticks, step, digest);
    return rec;
}

// --- EncodeDemo — serialize a Recorder to a byte buffer, hand-LE, deterministic, byte-exact ---------
// Layout (ALL fields LE, in this exact order):
//   magic(8) | version(u32) | seed(u32) | tickCount(u32) | keyframeInterval(u32) | worldByteLen(u32) |
//   inputByteLen(u32) | <worldBytes> | <inputBytes>
// `serWorld(const World&)` and `serInputRing(const net::InputRing<Input>&)` MUST themselves be hand-LE
// (field-by-field, no struct memcpy) so the bytes are identical on MSVC + Apple clang.
template <class World, class Input, class SerWorldFn, class SerInputFn>
std::vector<uint8_t> EncodeDemo(const Recorder<World, Input>& rec, SerWorldFn serWorld,
                                SerInputFn serInputRing) {
    const std::vector<uint8_t> worldBytes = serWorld(rec.initial);
    const std::vector<uint8_t> inputBytes = serInputRing(rec.ring);

    std::vector<uint8_t> out;
    PutBytes(out, kDemoMagic, 8);                                          // magic(8)
    PutU32(out, kDemoVersion);                                             // version
    PutU32(out, rec.seed);                                                 // seed
    PutU32(out, rec.tickCount);                                            // tickCount
    PutU32(out, 0u);                                                       // keyframeInterval (RP1: 0)
    PutU32(out, static_cast<uint32_t>(worldBytes.size()));                 // worldByteLen
    PutU32(out, static_cast<uint32_t>(inputBytes.size()));                 // inputByteLen
    PutBytes(out, worldBytes.data(), worldBytes.size());                   // <worldBytes>
    PutBytes(out, inputBytes.data(), inputBytes.size());                   // <inputBytes>
    return out;
}

}  // namespace hf::replay
