// Slice PROFILE-S6 — Live ScopedZone capstone (issue #31, the FLAGSHIP #31 capstone).
//
// S1–S5 built + PROVED the deterministic, scrub-seekable capture with the structure-vs-timing split on
// synthetic/scripted captures. S6 is the LIVE CAPSTONE: a `ScopedZone` RAII helper that, in a REAL frame,
// emits the structural enter/exit events via the SAME S1 emitters AND measures wall-clock time into the
// NON-golden timing overlay — proving the split holds with REAL measured timing, not just synthetic. The
// structural digest of a live-captured frame is byte-identical to the scripted MakeShowcaseCapture()
// expectation (structure golden, deterministic), while the timing slots carry genuine, non-deterministic
// nanosecond measurements.
//
// ============================ THE ISOLATION (the seq_render.h precedent) ============================
// The ONE non-deterministic crossing — reading the clock — is ISOLATED in THIS sibling header (which is
// the ONLY place `<chrono>` is allowed in the profiler), so the bit-exact `engine/profile/profile.h`
// stays `<chrono>`-free + standalone-clang-pure. `ScopedZone` writes ONLY into the timing overlay
// (`timings[]`); it NEVER touches the structural column (`events[]` is built by the same S1
// `EmitEnter`/`EmitExit` used by the scripted path). The structural digest path never reads `timings[]`.
// This is the honest proof: real timing in, structure still golden out.
// ====================================================================================================
//
// Header-only. This header is NOT on the bit-exact standalone-clang TIMING path — it is a live helper
// (timing is non-deterministic by nature). It compiles on MSVC + clang (chrono is portable). It does NOT
// modify profile.h (S1–S5 untouched — all pinned digests stay).

#pragma once

#include "profile/profile.h"

#include <chrono>   // the ONE non-deterministic crossing — isolated HERE, nowhere in profile.h

namespace hf::profile {

// --- ScopedZone — RAII scope timing (the one clock crossing) ----------------------------------------
// On construction: capture the enter event's index (BEFORE EmitEnter so it points AT the enter event) +
// the start time, then EmitEnter(c, nameId) (the S1 structural emitter — structure built EXACTLY as the
// scripted path). On destruction: EmitExit(c, nameId) (the recorded nameId), then write the measured
// duration into the ENTER event's timing slot. Structure (events) is the SAME S1 stream → identical
// structural digest; timing is written ONLY to the overlay → NEVER affects the structural digest.
class ScopedZone {
public:
    ScopedZone(Capture& c, uint32_t nameId)
        : c_(&c),
          enterIdx_(static_cast<uint32_t>(c.events.size())),
          start_(std::chrono::steady_clock::now()) {
        EmitEnter(c, nameId);
    }
    ~ScopedZone() {
        EmitExit(*c_, c_->events[enterIdx_].nameId);
        const auto end = std::chrono::steady_clock::now();
        const auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        // The standard "scope duration lives on the enter marker" convention. Clamp (monotonic clock so
        // ns >= 0, but be defensive). WRITES ONLY to timings[] — the structural column is untouched.
        c_->timings[enterIdx_].cpuNanos = static_cast<uint64_t>(ns < 0 ? 0 : ns);
    }
    ScopedZone(const ScopedZone&)            = delete;
    ScopedZone& operator=(const ScopedZone&) = delete;

private:
    Capture*                              c_;
    uint32_t                              enterIdx_;
    std::chrono::steady_clock::time_point start_;
};

// --- ScopedFrame — RAII frame bracket (EmitFrameBegin / EmitFrameEnd) -------------------------------
// No timing on the frame markers in v1 — the frame duration is the sum of its zones, which the overlay
// already carries. Keep v1 simple. Structure via the S1 frame emitters ONLY.
class ScopedFrame {
public:
    ScopedFrame(Capture& c, uint32_t frameNumber) : c_(&c) { EmitFrameBegin(c, frameNumber); }
    ~ScopedFrame() { EmitFrameEnd(*c_); }
    ScopedFrame(const ScopedFrame&)            = delete;
    ScopedFrame& operator=(const ScopedFrame&) = delete;

private:
    Capture* c_;
};

// --- BuildLiveShowcase — build the showcase capture via RAII, with real measurable work --------------
// Build a capture whose STRUCTURE matches MakeShowcaseCapture() EXACTLY (Enter Frame; Enter Shadow;
// Draw(Shadow,2); Exit Shadow; Enter Lit; Draw(Lit,5); Exit Lit; Exit Frame), but via ScopedZone RAII +
// a volatile-accumulating busy loop of `work` iterations INSIDE each zone so steady_clock measures a real,
// nonzero, work-dependent duration. The names are interned in the SAME first-seen order as
// MakeShowcaseCapture — "Frame"/"Shadow"/"Lit" → ids 0/1/2 — so the structural digest matches.
// `seedNames` is accepted to mirror the live-engine seam (a pre-seeded NameTable); the names are
// re-interned into the fresh capture in the canonical order regardless (interning is idempotent), so the
// structural digest is identical for ANY seed that contains those names first (or is empty).
inline Capture BuildLiveShowcase(const NameTable& seedNames, uint64_t work) {
    (void)seedNames;  // the live-engine seam (a pre-seeded name table); canonical re-intern below is authoritative

    Capture c;
    const char kFrame[]  = { 'F', 'r', 'a', 'm', 'e' };
    const char kShadow[] = { 'S', 'h', 'a', 'd', 'o', 'w' };
    const char kLit[]    = { 'L', 'i', 't' };
    const uint32_t frame  = Intern(c.names, kFrame,  sizeof(kFrame));   // id 0 (matches MakeShowcaseCapture)
    const uint32_t shadow = Intern(c.names, kShadow, sizeof(kShadow));  // id 1
    const uint32_t lit    = Intern(c.names, kLit,    sizeof(kLit));     // id 2

    // A volatile-accumulating busy loop so steady_clock measures a real, nonzero, work-dependent duration.
    volatile uint64_t sink = 0;
    auto busy = [&sink, work]() {
        for (uint64_t i = 0; i < work; ++i) {
            sink = sink + i * 2654435761ull + 1ull;   // volatile write → the optimizer can't elide the loop
        }
    };

    {
        ScopedZone zFrame(c, frame);   // Enter Frame  (Exit Frame on scope close)
        busy();
        {
            ScopedZone zShadow(c, shadow);   // Enter Shadow  (Exit Shadow on scope close)
            busy();
            EmitDraw(c, shadow, 2);          // Draw(Shadow, 2) — inside the Shadow scope
        }
        {
            ScopedZone zLit(c, lit);   // Enter Lit  (Exit Lit on scope close)
            busy();
            EmitDraw(c, lit, 5);       // Draw(Lit, 5) — inside the Lit scope
        }
    }
    (void)sink;  // consume the accumulator
    return c;
}

}  // namespace hf::profile
