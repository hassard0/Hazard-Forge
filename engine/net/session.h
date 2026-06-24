// Generic deterministic SESSION core (Slice NS1, flagship #24 NETCODE, beachhead).
//
// The irreducible rollback-netcode primitive: a header-only, transport-agnostic, templated
// Session<World,Input> that drives N ticks of a caller-supplied deterministic Step from a per-tick
// input ring and exposes an FNV-1a-64 state DIGEST. This generalizes the copy-pasted FPX5-style
// RunLockstep (the per-sim lockstep harnesses) into ONE parameterized engine: two peers fed the
// same inputs re-derive a bit-identical world EVERY tick (the lockstep invariant) and a hard-pinned
// final digest. Pure-CPU INTEGER — no float, no <cmath>, no clock/RNG, no GPU.
//
// SELF-CONTAINED ON PURPOSE: this header includes ONLY <cstdint>/<vector>/<cstddef> (NO fpx / mixer
// / RHI) so it compiles STANDALONE with `clang++ -std=c++20 -I engine -I tests tests/session_test.cpp`
// on the Mac (like audio/dsp.h) — the cheap cross-platform proof. World/Input are template params;
// the deterministic Step/Digest are supplied by the caller as template callables (NO <functional>).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hf::net {

// --- DigestBytes: FNV-1a-64 over n raw bytes (the generic state-digest currency) --------------------
// Reuses the engine-wide FNV-1a-64 constants (offset basis 1469598103934665603 / prime 1099511628211 —
// the SAME FNV as audio/dsp.h::DigestBuffer / ai.h DigestBlackboard). Hashing byte-by-byte keeps the
// digest stable; two equal byte spans hash IDENTICALLY, a single changed byte changes the digest. The
// per-sim harnesses only memcmp; a PINNED uint64_t golden is NS1's new contribution.
inline uint64_t DigestBytes(const void* data, std::size_t n) {
    const auto* p = static_cast<const unsigned char*>(data);
    uint64_t h = 1469598103934665603ull;           // the FNV-1a 64-bit offset basis
    for (std::size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ull;                      // the FNV-1a 64-bit prime
    }
    return h;
}

// --- InputRing: the per-tick input store, queried by tick index -------------------------------------
// A simple growable structure (std::vector<std::vector<Input>> indexed by tick). Inputs at the SAME
// tick are kept in INSERTION ORDER (the deterministic in-array application order — the SimTick
// contract). At(tick) returns an empty vector for a tick with no inputs (deterministic).
template <class Input>
struct InputRing {
    std::vector<std::vector<Input>> byTick;   // byTick[t] = inputs applied on tick t (insertion order)

    // Append an input for `tick`, growing to cover the tick. Same-tick inputs stay in insertion order.
    void AddInput(uint32_t tick, const Input& in) {
        if (static_cast<std::size_t>(tick) >= byTick.size())
            byTick.resize(static_cast<std::size_t>(tick) + 1);
        byTick[static_cast<std::size_t>(tick)].push_back(in);
    }

    // Inputs applied on `tick` — an empty vector for a tick with no inputs (deterministic).
    const std::vector<Input>& At(uint32_t tick) const {
        static const std::vector<Input> kEmpty{};
        if (static_cast<std::size_t>(tick) >= byTick.size()) return kEmpty;
        return byTick[static_cast<std::size_t>(tick)];
    }
};

// --- Session: the generic deterministic state + its input ring + the next tick ----------------------
template <class World, class Input>
struct Session {
    World            world;     // the deterministic state
    InputRing<Input> ring;      // inputs by tick
    uint32_t         tick = 0;  // the next tick to step
};

// Advance one tick: apply the deterministic transition step(world, inputs-this-tick, tick), then ++tick.
// `step` is the SimTick / ApplyCommand+integrate analog — pure of its inputs (no hidden global state).
template <class World, class Input, class StepFn>
void Advance(Session<World, Input>& s, StepFn step) {
    step(s.world, s.ring.At(s.tick), s.tick);
    ++s.tick;
}

// One-shot reference: run `ticks` advances from `init` over a COPY of `ring`, return the final digest.
// The generalized fpx-style RunLockstep — deterministic of (init, ring, ticks, step, digest) alone, so
// two peers / two calls produce the IDENTICAL uint64_t.
template <class World, class Input, class StepFn, class DigestFn>
uint64_t RunLockstep(World init, const InputRing<Input>& ring, uint32_t ticks,
                     StepFn step, DigestFn digest) {
    Session<World, Input> s;
    s.world = init;
    s.ring  = ring;     // a COPY — the caller's ring is untouched
    s.tick  = 0;
    for (uint32_t t = 0; t < ticks; ++t) Advance(s, step);
    return digest(s.world);
}

}  // namespace hf::net
