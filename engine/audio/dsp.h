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

// =====================================================================================================
// Slice DSP2 — Integer ADSR envelope NODE (Issue #26, flagship #23 DETERMINISTIC AUDIO, 2nd slice).
// APPEND-ONLY: the DSP1 osc code above is untouched. A streaming envelope node that shapes a source
// block sample-by-sample, carrying its elapsed-sample state (`t`) ACROSS ApplyEnvBlock calls — exactly
// like DSP1's phase. Pure-CPU integer (Q15). NO <cmath>, NO float.
//
// The linear-ADSR math (Adsr fields + EnvelopeAt + MulQ15/ClampI16) is COPIED VERBATIM from
// engine/audio/mixer.{h,cpp} (mixer.h Adsr / mixer.cpp:62-70 MulQ15+ClampI16 [already present above] /
// mixer.cpp:95-122 EnvelopeAt) so the envelope is bit-identical to the mixer. We copy (not #include) to
// keep dsp.h self-contained / single-translation-unit compilable (Mac clang one-file build).
// -----------------------------------------------------------------------------------------------------

// Piecewise-linear ADSR envelope. attack/decay/release in SAMPLES; sustainLevel in Q15 (0..32767).
// Mirrors mixer.h's Adsr fields exactly.
struct Adsr {
    int attack = 0;            // samples
    int decay = 0;             // samples
    int sustainLevel = 32767;  // Q15
    int release = 0;           // samples
};

// Evaluate the ADSR envelope at sample offset `t` within a voice of length `durSample`. Returns Q15 in
// [0, 32767]. COPIED VERBATIM from mixer.cpp:95-122 (release-takes-precedence linear ADSR; the
// int64-intermediate ramps); bit-identical to the mixer.
inline int EnvelopeAt(const Adsr& env, int t, int durSample) {
    if (t < 0 || t > durSample) return 0;

    const int a = env.attack;
    const int d = env.decay;
    const int s = env.sustainLevel;
    const int r = env.release;
    const int releaseStart = durSample - r;   // sample offset where the release ramp begins

    // Release region takes precedence (it runs against the END of the voice).
    if (r > 0 && t >= releaseStart) {
        // Level at the start of release is the sustain level; ramp sustain -> 0 over r samples.
        const int into = t - releaseStart;             // 0..r
        // s * (r - into) / r, integer.
        return static_cast<int>((static_cast<int64_t>(s) * (r - into)) / r);
    }
    // Attack: 0 -> 32767 over [0, a).
    if (a > 0 && t < a) {
        return static_cast<int>((static_cast<int64_t>(32767) * t) / a);
    }
    // Decay: 32767 -> s over [a, a+d).
    if (d > 0 && t < a + d) {
        const int into = t - a;                         // 0..d
        return static_cast<int>(32767 + (static_cast<int64_t>(s - 32767) * into) / d);
    }
    // Sustain: hold s.
    return s;
}

// A STATEFUL envelope graph node. `t` is the elapsed-sample counter that carries ACROSS ApplyEnvBlock
// calls — that is the whole point of the slice (block-boundary continuity for the envelope stage).
struct EnvNode {
    Adsr     env{};
    int      durSample = 0;
    uint32_t t         = 0;
    bool     enabled   = true;
};

// Shape `src` sample-by-sample, APPENDING the result to `outAppend` (so N calls build one continuous
// stream). If enabled, each sample is scaled by the envelope at the node's current `t`
// (ClampI16(MulQ15(s, level))); if disabled, the sample is appended verbatim (bypass no-op). `node.t`
// advances once per sample and is written back, so the envelope stage persists across calls.
inline void ApplyEnvBlock(EnvNode& node, const std::vector<int16_t>& src,
                          std::vector<int16_t>& outAppend) {
    outAppend.reserve(outAppend.size() + src.size());
    for (const int16_t s : src) {
        if (node.enabled) {
            const int level = EnvelopeAt(node.env, static_cast<int>(node.t), node.durSample);
            outAppend.push_back(ClampI16(MulQ15(s, level)));
        } else {
            outAppend.push_back(s);
        }
        ++node.t;
    }
}

// =====================================================================================================
// Slice DSP3 — Fixed-point biquad FILTER NODE (Issue #26, flagship #23 DETERMINISTIC AUDIO, 3rd slice,
// THE CRUX). APPEND-ONLY: the DSP1 osc + DSP2 env code above is untouched. A Q14 (16384 == 1.0)
// Direct-Form-I biquad with COMMITTED integer coefficient presets (host-precomputed OFFLINE — NO runtime
// cos/tan), int64-intermediate accumulation, and a delay line carried ACROSS FilterBlock calls — exactly
// like DSP1's phase and DSP2's `t`. Pure-CPU integer. NO <cmath>, NO float at runtime.
//
// Q14 (not Q15) because the normalized a1 coefficient can reach ~±2.0, which overflows a Q15 int16 range;
// Q14 has the headroom (a1 fits comfortably in int32, the accumulate is int64). Direct Form I, a0=1:
//   y[n] = (b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]) >> 14   (then ClampI16)
// The feedback taps y[n-1]/y[n-2] are the actual EMITTED (post-clamp) int16 samples, which keeps the
// fixed-point loop bounded.
// -----------------------------------------------------------------------------------------------------

// Q14 biquad coefficients (16384 == 1.0). a0 normalized to 1 (not stored). int32 so a1 (~±2.0) fits.
struct BiquadCoeffs {
    int32_t b0, b1, b2, a1, a2;
};

// --- Committed presets (host-baked OFFLINE via the RBJ audio-EQ cookbook; the kSineTable LUT discipline:
//     COMMITTED int Q14 literals, NO runtime cos/tan). The LPF/HPF were computed with a throwaway python
//     calc (w0=2pi*fc/fs, alpha=sin(w0)/(2Q); LPF b0=(1-cos w0)/2,b1=1-cos w0,b2=(1-cos w0)/2; HPF
//     b0=(1+cos w0)/2,b1=-(1+cos w0),b2=(1+cos w0)/2; a0=1+alpha,a1=-2cos w0,a2=1-alpha; divide all by a0,
//     x16384 round-to-nearest) at fc=2000, fs=48000, Q=0.707.

// Passthrough — the additive-identity no-op (EXACT): y = (16384*x) >> 14 = x byte-identical. Make-or-break.
inline constexpr BiquadCoeffs kBiquadPassthrough = {16384, 0, 0, 0, 0};

// Lowpass, RBJ LPF fc=2000 fs=48000 Q=0.707, quantized to Q14 round-to-nearest.
// STABLE (poles inside the unit circle): |a2| = 11314 < 16384, and |a1| = 26754 < 16384 + a2 = 27698.
inline constexpr BiquadCoeffs kBiquadLowpass2k = {236, 472, 236, -26754, 11314};

// Highpass, RBJ HPF fc=2000 fs=48000 Q=0.707, quantized to Q14 round-to-nearest. Same denominator (a1,a2)
// as the LPF. STABLE: |a2| = 11314 < 16384, and |a1| = 26754 < 16384 + a2 = 27698.
inline constexpr BiquadCoeffs kBiquadHighpass2k = {13613, -27226, 13613, -26754, 11314};

// A STATEFUL biquad filter graph node. The four int16 delay-line taps (x1/x2 past inputs, y1/y2 past
// EMITTED outputs) carry ACROSS FilterBlock calls — that is the whole point of the slice (cross-block
// state continuity for the filter stage).
struct BiquadNode {
    BiquadCoeffs c;
    int16_t x1 = 0, x2 = 0, y1 = 0, y2 = 0;
};

// Filter `src` sample-by-sample, APPENDING the result to `outAppend` (so N calls build one continuous
// stream). Direct Form I, int64 accumulate, arithmetic >>14, ClampI16, then shift the delay line. The
// state (x1/x2/y1/y2) persists in `bq` across calls. NO <cmath>, NO float.
inline void FilterBlock(BiquadNode& bq, const std::vector<int16_t>& src,
                        std::vector<int16_t>& outAppend) {
    outAppend.reserve(outAppend.size() + src.size());
    for (const int16_t x : src) {
        const int64_t acc =
            static_cast<int64_t>(bq.c.b0) * x +
            static_cast<int64_t>(bq.c.b1) * bq.x1 +
            static_cast<int64_t>(bq.c.b2) * bq.x2 -
            static_cast<int64_t>(bq.c.a1) * bq.y1 -
            static_cast<int64_t>(bq.c.a2) * bq.y2;
        const int16_t y = ClampI16(static_cast<int32_t>(acc >> 14));  // arithmetic shift on signed int64
        outAppend.push_back(y);
        bq.x2 = bq.x1; bq.x1 = x;   // shift the input delay line
        bq.y2 = bq.y1; bq.y1 = y;   // shift the (emitted) output delay line
    }
}

// =====================================================================================================
// Slice DSP4 — Declarative node-graph PATCH (Issue #26, flagship #23 DETERMINISTIC AUDIO, 4th slice, THE
// MetaSounds-CLASS HEADLINE). APPEND-ONLY: the DSP1 osc + DSP2 env + DSP3 filter code above is untouched.
// A whole synth voice is a FLAT node array with INTEGER-INDEX wiring (the pcg::PcgGraph / ai decision-tree
// / nav-poly precedent — indices, NOT pointers), evaluated per block in fixed topological order. Each
// node reuses a DSP1/2/3 stage verbatim (or the new int32-accumulate-then-clamp Mix). Node states (osc
// phase, env t, filt delay line) mutate IN the patch so they carry across EvaluateBlock calls — so the
// graph produces a byte-identical buffer whether rendered as one big buffer or block-by-block.
//
// The discipline: in0/in1 reference LOWER node indices (a fixed topo order, validated defensively — an
// out-of-range / non-strictly-lower input is treated as a silent/zero input, never read out of bounds).
// Pure-CPU integer. NO <cmath>, NO float. Self-contained (only <cstdint>/<vector>; no mixer/fpx include).
// -----------------------------------------------------------------------------------------------------

enum class NodeType { Osc, Env, Filter, Mix };

// One node of a Patch. The `type` selects which embedded stage runs; the others are inert. in0/in1 are
// integer indices of input nodes (MUST be < this node's own index). gain0/gain1 are Q15 mix gains.
struct DspNode {
    NodeType   type = NodeType::Osc;
    int        in0 = -1;
    int        in1 = -1;
    OscNode    osc{};                       // type==Osc    (a source — ignores in0/in1)
    EnvNode    env{};                       // type==Env    (shapes in0's block)
    BiquadNode filt{};                      // type==Filter (filters in0's block)
    int32_t    gain0 = 32767;               // type==Mix    (Q15 gain on in0)
    int32_t    gain1 = 32767;               // type==Mix    (Q15 gain on in1)
};

// A declarative DSP graph: a flat node array (topo-ordered: a node's inputs are lower indices) plus the
// output node index (the block whose samples are the patch output; default to the last node if -1).
struct Patch {
    std::vector<DspNode> nodes;
    int outNode = -1;
};

// Evaluate `frames` of the patch, APPENDING the output node's block to `outAppend`. Each node is rendered
// once in ascending index order into a per-node scratch block; node states mutate IN `p.nodes` so they
// carry across calls (block-by-block == one big buffer). Defensive: an input index that is not a valid,
// strictly-lower node yields a silent (empty/zero) contribution — never an OOB read.
inline void EvaluateBlock(Patch& p, int sampleRate, int frames, std::vector<int16_t>& outAppend) {
    const int n = static_cast<int>(p.nodes.size());
    if (n <= 0 || frames <= 0) return;

    std::vector<std::vector<int16_t>> blk(static_cast<size_t>(n));

    // A valid input index must be in [0, i) — a lower node already evaluated this block.
    auto validIn = [&](int idx, int self) -> bool { return idx >= 0 && idx < self; };

    for (int i = 0; i < n; ++i) {
        DspNode& nd = p.nodes[i];
        switch (nd.type) {
            case NodeType::Osc:
                RenderBlock(nd.osc, sampleRate, frames, blk[static_cast<size_t>(i)]);
                break;
            case NodeType::Env: {
                static const std::vector<int16_t> kEmpty;
                const std::vector<int16_t>& src =
                    validIn(nd.in0, i) ? blk[static_cast<size_t>(nd.in0)] : kEmpty;
                ApplyEnvBlock(nd.env, src, blk[static_cast<size_t>(i)]);
                break;
            }
            case NodeType::Filter: {
                static const std::vector<int16_t> kEmpty;
                const std::vector<int16_t>& src =
                    validIn(nd.in0, i) ? blk[static_cast<size_t>(nd.in0)] : kEmpty;
                FilterBlock(nd.filt, src, blk[static_cast<size_t>(i)]);
                break;
            }
            case NodeType::Mix: {
                const bool ok0 = validIn(nd.in0, i);
                const bool ok1 = validIn(nd.in1, i);
                const std::vector<int16_t>* a = ok0 ? &blk[static_cast<size_t>(nd.in0)] : nullptr;
                const std::vector<int16_t>* b = ok1 ? &blk[static_cast<size_t>(nd.in1)] : nullptr;
                std::vector<int16_t>& dst = blk[static_cast<size_t>(i)];
                dst.reserve(static_cast<size_t>(frames));
                for (int f = 0; f < frames; ++f) {
                    const int32_t s0 = (a && static_cast<size_t>(f) < a->size())
                        ? MulQ15((*a)[static_cast<size_t>(f)], nd.gain0) : 0;
                    const int32_t s1 = (b && static_cast<size_t>(f) < b->size())
                        ? MulQ15((*b)[static_cast<size_t>(f)], nd.gain1) : 0;
                    dst.push_back(ClampI16(s0 + s1));
                }
                break;
            }
            default:
                break;
        }
    }

    const int outIdx = (p.outNode < 0) ? (n - 1) : p.outNode;
    if (outIdx >= 0 && outIdx < n) {
        const std::vector<int16_t>& out = blk[static_cast<size_t>(outIdx)];
        outAppend.insert(outAppend.end(), out.begin(), out.end());
    }
}

// Convenience: evaluate `totalFrames` in ONE EvaluateBlock — the "one big buffer" reference the
// block-boundary test compares the N-block render against.
inline std::vector<int16_t> RenderPatch(Patch& p, int sampleRate, int totalFrames) {
    std::vector<int16_t> out;
    EvaluateBlock(p, sampleRate, totalFrames, out);
    return out;
}

}  // namespace hf::audio::dsp
