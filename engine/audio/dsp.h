#pragma once
// Slice DSP1 — block buffer + stateful wavetable oscillator NODE (Issue #26, flagship #23 DETERMINISTIC
// AUDIO, beachhead). Header-only, namespace hf::audio::dsp. Pure C++ INTEGER / FIXED-POINT (Q15 int16
// PCM, uint32 phase accumulator): NO <cmath>, NO float/double, NO clock/RNG, NO RHI/backend/GPU symbols.
//
// The crux this slice pins: a graph node persists its 32-bit phase as state and advances it ACROSS
// RenderBlock calls, so rendering one big buffer is BYTE-FOR-BYTE identical to rendering it as N blocks.
// The "buffer-hash golden" (DigestBuffer) is the cross-platform proof — the SAME CPU test produces the
// IDENTICAL 64-bit digest on Windows/MSVC and Mac/clang (NO image, NO render-bake).
//
// The sine table + kSineShift + the phase-increment formula + the square convention are COPIED VERBATIM
// from engine/audio/mixer.cpp (kSineTable / kSineN=256 / kSineShift=24, inc = (freqHz<<32)/sampleRate,
// square = (phase & 0x80000000u) ? -32767 : +32767) so the audio story is bit-identical to the mixer and
// a future unification is trivial. We copy (rather than #include) because mixer.cpp keeps them in an
// anonymous namespace; the committed integer values here MUST match the mixer's exactly.
#include <cstdint>
#include <vector>

namespace hf::audio::dsp {

// --- Sine wavetable (COPIED VERBATIM from mixer.cpp; must match exactly) -----------------------------
// kSineTable[i] == round(32767 * sin(2*pi*i / kSineN)), i in [0, 256). The phase is a 32-bit accumulator
// spanning one full cycle as 2^32; the table index is the top 8 bits (phase >> 24).
inline constexpr int kSineN = 256;
inline constexpr int kSineShift = 24;   // 32-bit phase -> top 8 bits index a 256-entry table
inline constexpr int16_t kSineTable[kSineN] = {
         0,    804,   1608,   2410,   3212,   4011,   4808,   5602,
      6393,   7179,   7962,   8739,   9512,  10278,  11039,  11793,
     12539,  13279,  14010,  14732,  15446,  16151,  16846,  17530,
     18204,  18868,  19519,  20159,  20787,  21403,  22005,  22594,
     23170,  23731,  24279,  24811,  25329,  25832,  26319,  26790,
     27245,  27683,  28105,  28510,  28898,  29268,  29621,  29956,
     30273,  30571,  30852,  31113,  31356,  31580,  31785,  31971,
     32137,  32285,  32412,  32521,  32609,  32678,  32728,  32757,
     32767,  32757,  32728,  32678,  32609,  32521,  32412,  32285,
     32137,  31971,  31785,  31580,  31356,  31113,  30852,  30571,
     30273,  29956,  29621,  29268,  28898,  28510,  28105,  27683,
     27245,  26790,  26319,  25832,  25329,  24811,  24279,  23731,
     23170,  22594,  22005,  21403,  20787,  20159,  19519,  18868,
     18204,  17530,  16846,  16151,  15446,  14732,  14010,  13279,
     12539,  11793,  11039,  10278,   9512,   8739,   7962,   7179,
      6393,   5602,   4808,   4011,   3212,   2410,   1608,    804,
         0,   -804,  -1608,  -2410,  -3212,  -4011,  -4808,  -5602,
     -6393,  -7179,  -7962,  -8739,  -9512, -10278, -11039, -11793,
    -12539, -13279, -14010, -14732, -15446, -16151, -16846, -17530,
    -18204, -18868, -19519, -20159, -20787, -21403, -22005, -22594,
    -23170, -23731, -24279, -24811, -25329, -25832, -26319, -26790,
    -27245, -27683, -28105, -28510, -28898, -29268, -29621, -29956,
    -30273, -30571, -30852, -31113, -31356, -31580, -31785, -31971,
    -32137, -32285, -32412, -32521, -32609, -32678, -32728, -32757,
    -32767, -32757, -32728, -32678, -32609, -32521, -32412, -32285,
    -32137, -31971, -31785, -31580, -31356, -31113, -30852, -30571,
    -30273, -29956, -29621, -29268, -28898, -28510, -28105, -27683,
    -27245, -26790, -26319, -25832, -25329, -24811, -24279, -23731,
    -23170, -22594, -22005, -21403, -20787, -20159, -19519, -18868,
    -18204, -17530, -16846, -16151, -15446, -14732, -14010, -13279,
    -12539, -11793, -11039, -10278,  -9512,  -8739,  -7962,  -7179,
     -6393,  -5602,  -4808,  -4011,  -3212,  -2410,  -1608,   -804,
};

inline constexpr int16_t kSquareAmp = 32767;   // full-scale square amplitude (mixer convention)

// Q15 fixed-point multiply: (a * b) >> 15 with an int64 intermediate (avoids overflow before the shift).
// Copied verbatim from mixer.cpp; provided for parity with the mixer's helper set.
inline int32_t MulQ15(int32_t a, int32_t b) {
    return static_cast<int32_t>((static_cast<int64_t>(a) * b) >> 15);
}

inline int16_t ClampI16(int32_t v) {
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return static_cast<int16_t>(v);
}

// --- DigestBuffer: FNV-1a-64 over the PCM buffer (the per-slice golden currency) ---------------------
// Reuses the engine-wide FNV-1a-64 constants (offset basis 1469598103934665603 / prime 1099511628211 —
// the SAME FNV used by ai.h DigestBlackboard / verdict DigestSnapshot). Hashes each int16 as its two
// little-endian bytes, so the digest is endianness-independent + stable. Two equal buffers hash
// IDENTICALLY; a single changed sample changes the digest.
inline uint64_t DigestBuffer(const std::vector<int16_t>& buf) {
    uint64_t h = 1469598103934665603ull;          // the FNV-1a 64-bit offset basis
    for (const int16_t s : buf) {
        const uint16_t u = static_cast<uint16_t>(s);
        h ^= static_cast<uint64_t>(static_cast<uint8_t>(u & 0xFFu));        // low byte (LSB-first)
        h *= 1099511628211ull;                                             // the FNV-1a 64-bit prime
        h ^= static_cast<uint64_t>(static_cast<uint8_t>((u >> 8) & 0xFFu)); // high byte
        h *= 1099511628211ull;
    }
    return h;
}

// --- Oscillator node ---------------------------------------------------------------------------------
enum class Wave { Sine, Saw, Square };

// A STATEFUL oscillator graph node. `phase` is a 32-bit accumulator that carries ACROSS RenderBlock
// calls — that is the whole point of the slice (block-boundary continuity).
struct OscNode {
    Wave     wave   = Wave::Sine;
    uint32_t freqHz = 440;
    uint32_t phase  = 0;
};

// Render `frames` samples, APPENDING int16 PCM to `outAppend` (so N calls build one continuous stream).
// `osc.phase` is read FROM and written BACK TO `osc` so it persists across calls. The phase increment is
// the mixer's exact formula: inc = (uint32_t)(((uint64_t)freqHz << 32) / (uint32_t)sampleRate).
inline void RenderBlock(OscNode& osc, int sampleRate, int frames, std::vector<int16_t>& outAppend) {
    if (frames <= 0 || sampleRate <= 0) return;
    const uint32_t inc = static_cast<uint32_t>(
        (static_cast<uint64_t>(osc.freqHz) << 32) / static_cast<uint32_t>(sampleRate));
    uint32_t phase = osc.phase;   // pull into a local; write back at the end
    outAppend.reserve(outAppend.size() + static_cast<size_t>(frames));
    for (int i = 0; i < frames; ++i) {
        int16_t sample;
        switch (osc.wave) {
            case Wave::Sine:
                sample = kSineTable[(phase >> kSineShift) & (kSineN - 1)];
                break;
            case Wave::Saw:
                // Pure-integer ramp: the top 16 bits of the phase span 0..65535 over a cycle; subtract
                // the midpoint 32768 to ramp -32768..+32767 (the signed int16 sawtooth).
                sample = static_cast<int16_t>(
                    static_cast<int32_t>(phase >> 16) - 32768);
                break;
            case Wave::Square:
                // Top bit of the phase splits the cycle: first half (+), second half (-).
                sample = (phase & 0x80000000u) ? -kSquareAmp : kSquareAmp;
                break;
            default:
                sample = 0;
                break;
        }
        outAppend.push_back(sample);
        phase += inc;
    }
    osc.phase = phase;
}

// Convenience: make a FRESH OscNode and render `totalFrames` in ONE RenderBlock — the "one big buffer"
// reference the block-boundary test compares the N-block render against.
inline std::vector<int16_t> RenderOsc(Wave wave, uint32_t freqHz, int sampleRate, int totalFrames) {
    OscNode osc;
    osc.wave = wave;
    osc.freqHz = freqHz;
    osc.phase = 0;
    std::vector<int16_t> out;
    RenderBlock(osc, sampleRate, totalFrames, out);
    return out;
}

}  // namespace hf::audio::dsp
