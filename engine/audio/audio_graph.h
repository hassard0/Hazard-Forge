// Slice AU1 — DETERMINISTIC PROCEDURAL AUDIO GRAPH + 3D SPATIALIZATION (Track-S S10, the
// MetaSounds-class layer). Header-only, namespace hf::audio::graph. Pure C++ INTEGER / FIXED-POINT
// (Q15 int16 PCM, Q16.16 positions): NO <cmath>, NO float/double anywhere in the sample path, NO
// clock/RNG, NO RHI/backend/GPU symbols. Audio has NO GPU path — cross-platform bit-exactness IS the
// product: the same graph renders the IDENTICAL int16 samples on every machine/compiler (MSVC ==
// clang), a category a float audio DSP (UE5 MetaSounds) structurally cannot enter.
//
// WHAT THIS ADDS over the existing primitives (all READ-ONLY composed, never modified):
//   * engine/audio/mixer.h  — Q15 gains / int32 accumulate / hard clamp / linear constant-sum pan law
//                             (the pan + mix DISCIPLINE is replicated bit-exactly here; mixer.{h,cpp}
//                             stay byte-frozen).
//   * engine/audio/dsp.h    — the stateful wavetable OscNode (32-bit phase accumulator), the linear
//                             integer Adsr/EnvelopeAt, MulQ15/ClampI16, IsqrtU (all #included and
//                             reused VERBATIM — the graph is a wiring layer, not a fork).
//   * engine/flow/flow.h    — the deterministic node-graph VM DISCIPLINE mirrored (NOT its types):
//                             NodeId == index, the "own-id == no edge" sentinel, EdgeMask-gated
//                             in-degrees, Kahn TopoOrder with the LOWEST-id-first ascending-scan
//                             tie-break (ONE canonical order), deterministic cycle rejection
//                             (ok=false, never UB/hang), disconnected inputs read SILENCE (the
//                             flow.h "no edge reads 0" convention).
//
// THE MODEL: a Graph is a flat node array wired by input-slot indices; MakeGraphState topo-orders it
// ONCE (rejecting cycles) and allocates per-node state (osc phase / env t / delay rings — the
// dsp::Node flat-state pattern); RenderBlock(g, state, sampleRate, frames, out) renders `frames`
// stereo frames per call, each node evaluated EXACTLY ONCE per block in the canonical order into a
// per-node scratch block, the kOut sink's stereo block appended interleaved (L,R,...). All node state
// carries ACROSS RenderBlock calls, so one big render is BYTE-IDENTICAL to N block renders (the DSP1
// invariant, re-proved here at graph level). The output block feeds the EXISTING wav.h encoder (the
// showcase) and composes bit-exactly with the EXISTING mixer path (audio_graph_test proof (e)).
//
// CHANNELS: nodes are MONO sources/processors until a kPan / kSpatial splits to STEREO; kGain / kMix /
// kDelay / kAdsr are channel-count-preserving (they process each channel of their input); kPan /
// kSpatial read their input's channel 0 (mono-in convention, documented); kOut emits stereo (a mono
// input is DUPLICATED to both channels — the exact-identity convention, not center-pan halving).
// A node's channel-ness is derived from the static topology, so it never changes between blocks.
//
// 3D SPATIALIZATION (the new capability, kSpatial): SpatializeParams{listenerPos, listenerRight,
// emitterPos} in Q16.16 -> (1) integer inverse-square DISTANCE attenuation with a near clamp,
//       gainDist = min(kQ15One, refDist^2 * kQ15One / max(dist^2, near^2))       [exact integer form:
//       all squared distances are int64 Q32.32 (per-axis deltas clamped to +-2^30 first, so the
//       3-axis sum cannot overflow); refDist/nearDist are clamped to [1, 2^23] so refDist^2 * kQ15One
//       <= 2^61 fits int64; the divide is a truncating int64 divide; kQ15One = 32768 = EXACT 1.0]
//   and (2) AZIMUTH pan from the dot product of the normalized emitter direction with listenerRight:
//       pan = clamp((dot / max(dist,1)) >> 1, -32768, 32767)                      [dot is int64
//       Q32.32 (right components clamped to +-65536 = a unit vector), dist = dsp::IsqrtU(dist2) is
//       Q16.16, the truncating divide yields the Q16.16 normalized lateral component in [-1,1],
//       >> 1 converts Q16.16 [-65536,65536] -> Q15 [-32768,32768], then clamp]
//   feeding the mixer's EXACT linear constant-sum pan law (panL = (32768-pan)>>1, panR =
//   (32768+pan)>>1, each clamped to 32767 — copied from mixer.cpp so full-left/right/center are
//   integer-exact and dead-center is L==R EXACTLY). ALL hand-int64 (documented above) — no fpx.h
//   include, keeping this header standalone-clang compilable like dsp.h/flow.h.
//
// DIGEST: DigestAudioBlock = hf::net::DigestBytes over the rendered int16 samples (the pinned-golden
// currency; on the little-endian platforms we ship this equals dsp::DigestBuffer bit-for-bit, and the
// test asserts that equivalence).
//
// HONEST integer-DSP character (pinned in audio_graph_test, not smoothed): MulQ15 truncates (floor
// shift), so a Q15 feedback delay DECAYS TO EXACT SILENCE in finitely many echoes (a 16383-amplitude
// impulse at feedback 16384 is byte-zero from echo 14 on); a gain change between blocks steps
// discontinuously (zipper), by design — parameters are block-constant.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "audio/dsp.h"     // READ-ONLY primitives: OscNode/RenderBlock, Adsr/EnvelopeAt, MulQ15/ClampI16, IsqrtU
#include "net/session.h"   // hf::net::DigestBytes — the pinned-golden FNV-1a-64 currency

namespace hf::audio::graph {

using NodeId = uint32_t;   // a node's id == its index into Graph::nodes (the flow.h discipline)

inline constexpr int32_t kQ15One    = 32768;     // EXACT Q15 1.0 — MulQ15(x, kQ15One) == x bit-exactly
inline constexpr int     kMaxInputs = 4;         // fixed input slots per node (kMix uses all four)
inline constexpr int32_t kMaxDelay  = 1 << 20;   // delay-ring capacity bound (samples)

// FIXED enum numbering = the wire contract (never renumber) — the flow.h Kind discipline.
enum Kind : uint32_t {
    kOsc     = 0,   // 0 inputs: dsp wavetable oscillator (Sine/Saw/Square; stateful 32-bit phase)
    kGain    = 1,   // 1 input : per-channel MulQ15 by gainQ15 (kQ15One == bit-exact identity)
    kMix     = 2,   // N inputs: per-channel int32 accumulate over all connected slots, hard clamp
    kDelay   = 3,   // 1 input : feedback comb — y[n] = clamp(x[n] + MulQ15(y[n-D], feedbackQ15))
    kAdsr    = 4,   // 1 input : dsp::EnvelopeAt(env, t, durSample) Q15 scale per FRAME (stateful t)
    kPan     = 5,   // 1 input : mono ch0 -> stereo via the mixer's linear constant-sum pan law
    kSpatial = 6,   // 1 input : mono ch0 -> stereo via 3D distance attenuation + azimuth pan
    kOut     = 7,   // 1 input : the sink — its stereo block is the graph's rendered output
};

// ---- EdgeMask: which input slots are REAL data edges for a kind (the flow.h determinism crux) ------
// bit i == slot in[i]. kOsc is a source (no edges); kMix reads all four slots; every other kind reads
// exactly slot 0. TopoOrder counts ONLY masked, real (in-range, non-self) inputs.
inline uint32_t EdgeMask(uint32_t kind) {
    switch (kind) {
        case kOsc:  return 0b0000u;
        case kMix:  return 0b1111u;
        case kGain: case kDelay: case kAdsr: case kPan: case kSpatial: case kOut: return 0b0001u;
        default:    return 0b0000u;   // unknown kind: no edges (deterministic no-op source of silence)
    }
}

// 3D spatialization parameters (all positions/vectors Q16.16 fixed-point world units).
// listenerRight must be (approximately) unit length; its components are defensively clamped to
// +-65536. Coordinates are clamped per-axis to +-2^30 (+-16384 world units) before squaring.
// refDist = the unity-gain reference distance; nearDist = the inverse-square denominator clamp
// (both clamped to [1, 2^23] = (0, 128] world units — the documented int64-overflow bound).
struct SpatializeParams {
    int32_t listenerPos[3]   = {0, 0, 0};
    int32_t listenerRight[3] = {65536, 0, 0};   // unit +X
    int32_t emitterPos[3]    = {0, 0, 0};
    int32_t refDist          = 65536;           // Q16.16 (1.0)
    int32_t nearDist         = 65536;           // Q16.16 (1.0)
};

// One node. Unused input slots are set to the node's OWN id (the flow.h "no edge" sentinel); an
// out-of-range or self input is NEVER a dependency and reads SILENCE. Only the fields the kind uses
// are live; the rest are inert (the dsp.h DspNode fat-struct pattern).
struct Node {
    uint32_t kind = kOsc;
    NodeId   in[kMaxInputs] = {0, 0, 0, 0};

    dsp::Wave wave   = dsp::Wave::Sine;   // kOsc
    uint32_t  freqHz = 440;               // kOsc

    int32_t gainQ15 = kQ15One;            // kGain (Q15; kQ15One == exact identity)

    int32_t delaySamples = 1;             // kDelay ring length D (clamped to [1, kMaxDelay])
    int32_t feedbackQ15  = 0;             // kDelay feedback (Q15)

    dsp::Adsr env{};                      // kAdsr
    int       durSample = 0;              // kAdsr envelope span (samples)

    int32_t panQ15 = 0;                   // kPan: -32768 full-left .. 0 center .. +32767 full-right

    SpatializeParams spatial{};           // kSpatial
};

struct Graph {
    std::vector<Node> nodes;   // nodes[i] has NodeId i
};

// ---- "No edge" convention (identical to flow.h::IsRealEdge) ----------------------------------------
inline bool IsRealEdge(const Graph& g, NodeId self, NodeId in) {
    return in != self && static_cast<std::size_t>(in) < g.nodes.size();
}

// CountEdges: the number of REAL (EdgeMask-gated, in-range, non-self) input edges — the stat-line /
// introspection count.
inline uint32_t CountEdges(const Graph& g) {
    uint32_t edges = 0;
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
        const Node& nd = g.nodes[i];
        const uint32_t mask = EdgeMask(nd.kind);
        for (int s = 0; s < kMaxInputs; ++s)
            if ((mask & (1u << s)) && IsRealEdge(g, static_cast<NodeId>(i), nd.in[s])) ++edges;
    }
    return edges;
}

// ---- TopoOrder: Kahn's algorithm producing the ONE CANONICAL order (mirrors flow.h::TopoOrder) -----
// indeg[i] counts node i's EdgeMask-gated real input edges; repeatedly emit the LOWEST not-yet-emitted
// NodeId with indeg==0 via an ASCENDING SCAN (lowest-id-first tie-break — NEVER insertion/hash order).
// Returns true + the full order; returns false on a CYCLE (outOrder cleared) — a DETERMINISTIC
// rejection, never UB, never a hang (the outer loop is bounded by nodes.size()).
inline bool TopoOrder(const Graph& g, std::vector<NodeId>& outOrder) {
    outOrder.clear();
    const std::size_t n = g.nodes.size();
    if (n == 0) return true;

    std::vector<uint32_t> indeg(n, 0u);
    for (std::size_t i = 0; i < n; ++i) {
        const Node& nd = g.nodes[i];
        const NodeId self = static_cast<NodeId>(i);
        const uint32_t mask = EdgeMask(nd.kind);
        for (int s = 0; s < kMaxInputs; ++s)
            if ((mask & (1u << s)) && IsRealEdge(g, self, nd.in[s])) ++indeg[i];
    }

    std::vector<uint8_t> emitted(n, 0u);
    outOrder.reserve(n);
    for (std::size_t round = 0; round < n; ++round) {
        NodeId pick = 0;
        bool found = false;
        for (std::size_t i = 0; i < n; ++i) {   // ascending scan -> lowest-id-first (the canon)
            if (!emitted[i] && indeg[i] == 0u) { pick = static_cast<NodeId>(i); found = true; break; }
        }
        if (!found) { outOrder.clear(); return false; }   // CYCLE -> deterministic reject

        emitted[pick] = 1u;
        outOrder.push_back(pick);

        for (std::size_t j = 0; j < n; ++j) {
            if (emitted[j]) continue;
            const Node& dep = g.nodes[j];
            const NodeId self = static_cast<NodeId>(j);
            const uint32_t mask = EdgeMask(dep.kind);
            for (int s = 0; s < kMaxInputs; ++s)
                if ((mask & (1u << s)) && IsRealEdge(g, self, dep.in[s]) && dep.in[s] == pick)
                    --indeg[j];
        }
    }
    return true;
}

// ---- Per-node persistent state (the dsp::Node flat-state pattern) ----------------------------------
// Only the slots the node's kind uses are live. Delay rings hold past EMITTED (post-clamp) outputs
// (the dsp.h biquad discipline — keeps the fixed-point feedback loop bounded), one ring per channel.
struct NodeState {
    uint32_t             phase   = 0;    // kOsc: the 32-bit phase accumulator (dsp::OscNode::phase)
    uint32_t             envT    = 0;    // kAdsr: elapsed frames within the envelope
    std::vector<int16_t> ringL;          // kDelay channel-0 ring (size == clamped delaySamples)
    std::vector<int16_t> ringR;          // kDelay channel-1 ring
    int32_t              ringPos = 0;    // kDelay ring cursor (shared by both channels)
};

// GraphState: the topo order (computed ONCE; ok=false == cycle rejected) + per-node state.
struct GraphState {
    std::vector<NodeState> st;
    std::vector<NodeId>    order;
    bool                   ok = false;
};

// MakeGraphState: topo-order the graph (deterministic cycle rejection -> ok=false) and allocate the
// per-node state (delay rings zero-filled at their clamped capacity). Bounds: delaySamples is clamped
// to [1, kMaxDelay] here, so RenderBlock's ring indexing is in-range by construction.
inline GraphState MakeGraphState(const Graph& g) {
    GraphState s;
    s.ok = TopoOrder(g, s.order);
    s.st.assign(g.nodes.size(), NodeState{});
    if (!s.ok) return s;
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
        const Node& nd = g.nodes[i];
        if (nd.kind == kDelay) {
            int32_t d = nd.delaySamples;
            if (d < 1) d = 1;
            if (d > kMaxDelay) d = kMaxDelay;
            s.st[i].ringL.assign(static_cast<std::size_t>(d), 0);
            s.st[i].ringR.assign(static_cast<std::size_t>(d), 0);
        }
    }
    return s;
}

// ---- Spatialization coefficients (pure integer; exposed for the pinned unit tests) -----------------
// See the header comment for the exact integer forms. Stateless — computed once per rendered block
// (parameters are block-constant, like every node parameter).
struct SpatialCoeffs {
    int32_t distQ16;       // Q16.16 listener->emitter distance (IsqrtU of the clamped dist^2)
    int32_t gainDistQ15;   // Q15 distance gain in [0, kQ15One] (kQ15One == exact unity)
    int32_t panQ15;        // Q15 azimuth pan in [-32768, 32767]
    int32_t panL, panR;    // the mixer's linear constant-sum pan gains (Q15, endpoint-exact)
};

inline SpatialCoeffs ComputeSpatialCoeffs(const SpatializeParams& p) {
    // Per-axis deltas, clamped to +-2^30 so dx^2+dy^2+dz^2 <= 3*2^60 < 2^62 can never overflow int64.
    const int64_t kAxisBound = static_cast<int64_t>(1) << 30;
    auto clampAxis = [&](int64_t v) -> int64_t {
        if (v >  kAxisBound) return  kAxisBound;
        if (v < -kAxisBound) return -kAxisBound;
        return v;
    };
    const int64_t dx = clampAxis(static_cast<int64_t>(p.emitterPos[0]) - p.listenerPos[0]);
    const int64_t dy = clampAxis(static_cast<int64_t>(p.emitterPos[1]) - p.listenerPos[1]);
    const int64_t dz = clampAxis(static_cast<int64_t>(p.emitterPos[2]) - p.listenerPos[2]);
    const int64_t dist2 = dx * dx + dy * dy + dz * dz;          // Q32.32
    const int32_t dist  = dsp::IsqrtU(dist2);                   // Q16.16 (floor sqrt)

    // Distance gain: min(kQ15One, refDist^2 * kQ15One / max(dist^2, near^2)). refDist/nearDist are
    // clamped to [1, 2^23] so ref2 * kQ15One <= 2^61 fits int64 (the documented bound).
    const int64_t kDistBound = static_cast<int64_t>(1) << 23;
    auto clampDist = [&](int64_t v) -> int64_t {
        if (v < 1) return 1;
        if (v > kDistBound) return kDistBound;
        return v;
    };
    const int64_t rd    = clampDist(p.refDist);
    const int64_t nd    = clampDist(p.nearDist);
    const int64_t ref2  = rd * rd;
    const int64_t near2 = nd * nd;
    const int64_t den   = (dist2 > near2) ? dist2 : near2;      // >= 1 by the nearDist clamp
    int64_t gd = (ref2 * kQ15One) / den;                        // truncating int64 divide
    if (gd > kQ15One) gd = kQ15One;
    if (gd < 0)       gd = 0;

    // Azimuth: dot(emitterDir, listenerRight) / |emitterDir| -> the normalized lateral component.
    // Right components clamped to +-65536 (a unit vector) -> |dot| <= 3 * 2^30 * 2^16 < 2^48, safe.
    const int64_t kUnit = 65536;
    auto clampUnit = [&](int64_t v) -> int64_t {
        if (v >  kUnit) return  kUnit;
        if (v < -kUnit) return -kUnit;
        return v;
    };
    const int64_t dot = dx * clampUnit(p.listenerRight[0]) +
                        dy * clampUnit(p.listenerRight[1]) +
                        dz * clampUnit(p.listenerRight[2]);      // Q32.32
    const int64_t dd  = (dist > 1) ? dist : 1;
    int64_t pan = (dot / dd) >> 1;   // Q16.16 normalized component -> Q15 (truncating divide; the
                                     // int64 divide truncates toward zero, symmetrically L/R)
    if (pan >  32767) pan =  32767;
    if (pan < -32768) pan = -32768;

    // The mixer's EXACT linear constant-sum pan law (mixer.cpp): endpoint-exact, dead-center L==R.
    int32_t panL = (32768 - static_cast<int32_t>(pan)) >> 1;
    int32_t panR = (32768 + static_cast<int32_t>(pan)) >> 1;
    if (panL > 32767) panL = 32767;
    if (panR > 32767) panR = 32767;

    return SpatialCoeffs{dist, static_cast<int32_t>(gd), static_cast<int32_t>(pan), panL, panR};
}

// PanGains: the mixer's linear constant-sum pan law for a plain kPan node (same integer form).
inline void PanGains(int32_t panQ15, int32_t& outL, int32_t& outR) {
    if (panQ15 >  32767) panQ15 =  32767;
    if (panQ15 < -32768) panQ15 = -32768;
    int32_t l = (32768 - panQ15) >> 1;
    int32_t r = (32768 + panQ15) >> 1;
    if (l > 32767) l = 32767;
    if (r > 32767) r = 32767;
    outL = l;
    outR = r;
}

// ---- RenderBlock: evaluate the graph for `frames` frames, appending interleaved stereo -------------
// Each node renders EXACTLY ONCE per call, in the canonical topo order, into a per-node scratch block
// (channel-count derived from its input); node state mutates IN `s` so it carries across calls
// (one big render == N block renders, byte-identical). The LOWEST-id kOut node's stereo block is
// appended (L,R,L,R,...); no kOut -> silence. A cycle-rejected state (ok=false) appends silence and
// returns false — deterministic, never UB. Pure integer.
inline bool RenderBlock(const Graph& g, GraphState& s, int sampleRate, int frames,
                        std::vector<int16_t>& outAppend) {
    if (frames <= 0) return s.ok;
    const std::size_t n = g.nodes.size();
    if (!s.ok || s.st.size() != n || sampleRate <= 0) {
        outAppend.insert(outAppend.end(), static_cast<std::size_t>(frames) * 2u, static_cast<int16_t>(0));
        return false;
    }

    // Per-node scratch blocks: ch[0]/ch[1] sized `frames` when live; channels 1 (mono) or 2 (stereo).
    struct Block {
        std::vector<int16_t> ch0, ch1;
        uint32_t channels = 1;
    };
    std::vector<Block> blk(n);

    // Read input slot `slot` of node `self` at frame f, channel c — the "no edge reads SILENCE"
    // convention: a missing/self/out-of-range input reads 0; a mono input read at channel 1 reads
    // its channel 0 (mono is the same signal on both ears).
    auto inBlock = [&](NodeId self, const Node& nd, int slot) -> const Block* {
        const NodeId in = nd.in[slot];
        return IsRealEdge(g, self, in) ? &blk[static_cast<std::size_t>(in)] : nullptr;
    };
    auto readCh = [&](const Block* b, int f, uint32_t c) -> int32_t {
        if (b == nullptr) return 0;
        const std::vector<int16_t>& v = (c == 1u && b->channels == 2u) ? b->ch1 : b->ch0;
        return (static_cast<std::size_t>(f) < v.size()) ? v[static_cast<std::size_t>(f)] : 0;
    };

    for (const NodeId id : s.order) {
        const std::size_t si = static_cast<std::size_t>(id);
        const Node& nd = g.nodes[si];
        NodeState& ns = s.st[si];
        Block& out = blk[si];

        switch (nd.kind) {
            case kOsc: {
                // The dsp DSP1 oscillator VERBATIM: pull the persistent phase into a dsp::OscNode,
                // render the block, write the phase back (block-boundary continuity).
                dsp::OscNode osc;
                osc.wave = nd.wave;
                osc.freqHz = nd.freqHz;
                osc.phase = ns.phase;
                out.ch0.reserve(static_cast<std::size_t>(frames));
                dsp::RenderBlock(osc, sampleRate, frames, out.ch0);
                ns.phase = osc.phase;
                out.channels = 1;
                break;
            }
            case kGain: {
                const Block* a = inBlock(id, nd, 0);
                out.channels = a ? a->channels : 1u;
                out.ch0.reserve(static_cast<std::size_t>(frames));
                if (out.channels == 2u) out.ch1.reserve(static_cast<std::size_t>(frames));
                for (int f = 0; f < frames; ++f) {
                    out.ch0.push_back(dsp::ClampI16(dsp::MulQ15(readCh(a, f, 0), nd.gainQ15)));
                    if (out.channels == 2u)
                        out.ch1.push_back(dsp::ClampI16(dsp::MulQ15(readCh(a, f, 1), nd.gainQ15)));
                }
                break;
            }
            case kMix: {
                const Block* srcs[kMaxInputs];
                uint32_t chans = 1;
                for (int sl = 0; sl < kMaxInputs; ++sl) {
                    srcs[sl] = inBlock(id, nd, sl);
                    if (srcs[sl] != nullptr && srcs[sl]->channels == 2u) chans = 2;
                }
                out.channels = chans;
                out.ch0.reserve(static_cast<std::size_t>(frames));
                if (chans == 2u) out.ch1.reserve(static_cast<std::size_t>(frames));
                for (int f = 0; f < frames; ++f) {
                    int32_t accL = 0, accR = 0;   // the mixer.h int32-accumulate discipline
                    for (int sl = 0; sl < kMaxInputs; ++sl) {
                        accL += readCh(srcs[sl], f, 0);
                        if (chans == 2u) accR += readCh(srcs[sl], f, 1);
                    }
                    out.ch0.push_back(dsp::ClampI16(accL));   // ONE hard clamp after the sum
                    if (chans == 2u) out.ch1.push_back(dsp::ClampI16(accR));
                }
                break;
            }
            case kDelay: {
                const Block* a = inBlock(id, nd, 0);
                out.channels = a ? a->channels : 1u;
                out.ch0.reserve(static_cast<std::size_t>(frames));
                if (out.channels == 2u) out.ch1.reserve(static_cast<std::size_t>(frames));
                const int32_t D = static_cast<int32_t>(ns.ringL.size());   // clamped >= 1 at MakeGraphState
                for (int f = 0; f < frames; ++f) {
                    // Feedback comb over EMITTED (post-clamp) outputs: y[n] = clamp(x[n] + fb*y[n-D]).
                    // ring[ringPos] holds y[n-D] (written D frames ago); overwrite it with y[n].
                    const int16_t yl = dsp::ClampI16(
                        readCh(a, f, 0) + dsp::MulQ15(ns.ringL[static_cast<std::size_t>(ns.ringPos)],
                                                      nd.feedbackQ15));
                    ns.ringL[static_cast<std::size_t>(ns.ringPos)] = yl;
                    out.ch0.push_back(yl);
                    if (out.channels == 2u) {
                        const int16_t yr = dsp::ClampI16(
                            readCh(a, f, 1) + dsp::MulQ15(ns.ringR[static_cast<std::size_t>(ns.ringPos)],
                                                          nd.feedbackQ15));
                        ns.ringR[static_cast<std::size_t>(ns.ringPos)] = yr;
                        out.ch1.push_back(yr);
                    }
                    ns.ringPos = (ns.ringPos + 1) % D;
                }
                break;
            }
            case kAdsr: {
                const Block* a = inBlock(id, nd, 0);
                out.channels = a ? a->channels : 1u;
                out.ch0.reserve(static_cast<std::size_t>(frames));
                if (out.channels == 2u) out.ch1.reserve(static_cast<std::size_t>(frames));
                for (int f = 0; f < frames; ++f) {
                    // ONE envelope evaluation per FRAME (t advances per frame, not per channel-sample),
                    // applied to every channel — the mixer's per-frame envelope discipline.
                    const int level = dsp::EnvelopeAt(nd.env, static_cast<int>(ns.envT), nd.durSample);
                    out.ch0.push_back(dsp::ClampI16(dsp::MulQ15(readCh(a, f, 0), level)));
                    if (out.channels == 2u)
                        out.ch1.push_back(dsp::ClampI16(dsp::MulQ15(readCh(a, f, 1), level)));
                    ++ns.envT;
                }
                break;
            }
            case kPan: {
                const Block* a = inBlock(id, nd, 0);
                int32_t panL, panR;
                PanGains(nd.panQ15, panL, panR);
                out.channels = 2;
                out.ch0.reserve(static_cast<std::size_t>(frames));
                out.ch1.reserve(static_cast<std::size_t>(frames));
                for (int f = 0; f < frames; ++f) {
                    const int32_t x = readCh(a, f, 0);   // mono-in convention (channel 0)
                    out.ch0.push_back(dsp::ClampI16(dsp::MulQ15(x, panL)));
                    out.ch1.push_back(dsp::ClampI16(dsp::MulQ15(x, panR)));
                }
                break;
            }
            case kSpatial: {
                const Block* a = inBlock(id, nd, 0);
                const SpatialCoeffs sc = ComputeSpatialCoeffs(nd.spatial);
                out.channels = 2;
                out.ch0.reserve(static_cast<std::size_t>(frames));
                out.ch1.reserve(static_cast<std::size_t>(frames));
                for (int f = 0; f < frames; ++f) {
                    const int32_t x = readCh(a, f, 0);   // mono-in convention (channel 0)
                    const int32_t base = dsp::MulQ15(x, sc.gainDistQ15);   // distance attenuation first
                    out.ch0.push_back(dsp::ClampI16(dsp::MulQ15(base, sc.panL)));
                    out.ch1.push_back(dsp::ClampI16(dsp::MulQ15(base, sc.panR)));
                }
                break;
            }
            case kOut: {
                const Block* a = inBlock(id, nd, 0);
                out.channels = 2;
                out.ch0.reserve(static_cast<std::size_t>(frames));
                out.ch1.reserve(static_cast<std::size_t>(frames));
                for (int f = 0; f < frames; ++f) {
                    // A stereo input passes through; a mono input is DUPLICATED to both channels
                    // (bit-exact identity, not center-pan halving).
                    out.ch0.push_back(static_cast<int16_t>(readCh(a, f, 0)));
                    out.ch1.push_back(static_cast<int16_t>(readCh(a, f, 1)));
                }
                break;
            }
            default:
                // Unknown kind: a deterministic silent mono source (never UB).
                out.ch0.assign(static_cast<std::size_t>(frames), 0);
                out.channels = 1;
                break;
        }
    }

    // Append the LOWEST-id kOut node's stereo block interleaved; no kOut -> silence (deterministic).
    for (std::size_t i = 0; i < n; ++i) {
        if (g.nodes[i].kind == kOut) {
            const Block& ob = blk[i];
            outAppend.reserve(outAppend.size() + static_cast<std::size_t>(frames) * 2u);
            for (int f = 0; f < frames; ++f) {
                outAppend.push_back(ob.ch0[static_cast<std::size_t>(f)]);
                outAppend.push_back(ob.ch1[static_cast<std::size_t>(f)]);
            }
            return true;
        }
    }
    outAppend.insert(outAppend.end(), static_cast<std::size_t>(frames) * 2u, static_cast<int16_t>(0));
    return true;
}

// Convenience: fresh state + ONE RenderBlock over totalFrames — the "one big buffer" reference the
// block-boundary test compares the N-block render against. Returns an empty buffer's worth of silence
// via RenderBlock's own conventions on any rejected graph.
inline std::vector<int16_t> RenderGraph(const Graph& g, int sampleRate, int totalFrames) {
    GraphState s = MakeGraphState(g);
    std::vector<int16_t> out;
    RenderBlock(g, s, sampleRate, totalFrames, out);
    return out;
}

// ---- DigestAudioBlock: net::DigestBytes over the rendered int16 samples (the golden currency) ------
// On the little-endian platforms we ship (x64 / arm64), the raw sample memory IS the LSB-first byte
// stream, so this equals dsp::DigestBuffer bit-for-bit (asserted in audio_graph_test — the two
// digest currencies agree; no fork).
inline uint64_t DigestAudioBlock(const std::vector<int16_t>& samples) {
    return hf::net::DigestBytes(samples.data(), samples.size() * sizeof(int16_t));
}

// ====================================================================================================
// Fixtures (deterministic integer literals — the flow.h MakeShowcase* precedent). FIXED forever: the
// audio_graph_test pins MakeShowcaseGraph()'s digest and the AU1 showcase scene's digest.
// ====================================================================================================

// MakeShowcaseGraph: the FIXED pinned-digest chain osc -> adsr -> gain -> delay -> pan -> out.
// Unused input slots = the node's OWN id (the "no edge" sentinel).
inline Graph MakeShowcaseGraph() {
    Graph g;
    g.nodes.resize(6);

    // n0: 440 Hz sine source.
    g.nodes[0].kind = kOsc;
    g.nodes[0].in[0] = 0; g.nodes[0].in[1] = 0; g.nodes[0].in[2] = 0; g.nodes[0].in[3] = 0;
    g.nodes[0].wave = dsp::Wave::Sine;
    g.nodes[0].freqHz = 440;
    // n1: ADSR over n0 (attack 480, decay 960, sustain ~0.6, release 2400 over a 9600-sample note).
    g.nodes[1].kind = kAdsr;
    g.nodes[1].in[0] = 0; g.nodes[1].in[1] = 1; g.nodes[1].in[2] = 1; g.nodes[1].in[3] = 1;
    g.nodes[1].env = dsp::Adsr{480, 960, 19660, 2400};
    g.nodes[1].durSample = 9600;
    // n2: gain 0.75 (Q15 24576).
    g.nodes[2].kind = kGain;
    g.nodes[2].in[0] = 1; g.nodes[2].in[1] = 2; g.nodes[2].in[2] = 2; g.nodes[2].in[3] = 2;
    g.nodes[2].gainQ15 = 24576;
    // n3: delay 1200 samples, feedback 0.5 (Q15 16384) — the echo tail.
    g.nodes[3].kind = kDelay;
    g.nodes[3].in[0] = 2; g.nodes[3].in[1] = 3; g.nodes[3].in[2] = 3; g.nodes[3].in[3] = 3;
    g.nodes[3].delaySamples = 1200;
    g.nodes[3].feedbackQ15 = 16384;
    // n4: pan right of center (+8000).
    g.nodes[4].kind = kPan;
    g.nodes[4].in[0] = 3; g.nodes[4].in[1] = 4; g.nodes[4].in[2] = 4; g.nodes[4].in[3] = 4;
    g.nodes[4].panQ15 = 8000;
    // n5: the sink.
    g.nodes[5].kind = kOut;
    g.nodes[5].in[0] = 4; g.nodes[5].in[1] = 5; g.nodes[5].in[2] = 5; g.nodes[5].in[3] = 5;

    return g;
}

// MakeCyclicGraph: two gains reading each other — a REAL cycle for the deterministic-rejection test.
inline Graph MakeCyclicGraph() {
    Graph g;
    g.nodes.resize(2);
    g.nodes[0].kind = kGain;
    g.nodes[0].in[0] = 1; g.nodes[0].in[1] = 0; g.nodes[0].in[2] = 0; g.nodes[0].in[3] = 0;
    g.nodes[1].kind = kGain;
    g.nodes[1].in[0] = 0; g.nodes[1].in[1] = 1; g.nodes[1].in[2] = 1; g.nodes[1].in[3] = 1;
    return g;
}

// ====================================================================================================
// The AU1 showcase SCENE (shared by the Vulkan-side --au1-graph-shot and the Metal-side --au1-graph
// showcases so the WAV bytes are IDENTICAL cross-backend BY CONSTRUCTION — the water::ShowcaseWaves
// pattern). ~2 s at 48 kHz: two spatialized emitters — a left-behind low square PULSE (retriggered
// every 0.5 s) + a right-front sine ARPEGGIO (C5 E5 G5 C6, retriggered every 0.25 s) — mixed, run
// through a feedback delay tail, out. The host drives the note schedule BETWEEN blocks (block-constant
// parameter changes + envelope retriggers — the MetaSounds-style control seam), all integer.
// ====================================================================================================

struct Au1ShowcaseStats {
    uint32_t nodes   = 0;
    uint32_t edges   = 0;
    int      frames  = 0;
    uint64_t digest  = 0;
};

// MakeAu1SceneGraph: the FIXED 11-node scene graph (see the schedule in RenderAu1Showcase).
inline Graph MakeAu1SceneGraph() {
    Graph g;
    g.nodes.resize(11);
    auto self = [&](NodeId i) {
        g.nodes[i].in[0] = i; g.nodes[i].in[1] = i; g.nodes[i].in[2] = i; g.nodes[i].in[3] = i;
    };

    // ---- Emitter A: the low square pulse, LEFT-BEHIND of the listener --------------------------------
    // n0: 110 Hz square.
    self(0);
    g.nodes[0].kind = kOsc;
    g.nodes[0].wave = dsp::Wave::Square;
    g.nodes[0].freqHz = 110;
    // n1: the pulse envelope (retriggered by the host every 24000 samples).
    self(1);
    g.nodes[1].kind = kAdsr;
    g.nodes[1].in[0] = 0;
    g.nodes[1].env = dsp::Adsr{480, 2400, 8000, 4800};
    g.nodes[1].durSample = 12000;
    // n2: gain 0.61 (Q15 20000) — keeps the square polite under the mix.
    self(2);
    g.nodes[2].kind = kGain;
    g.nodes[2].in[0] = 1;
    g.nodes[2].gainQ15 = 20000;
    // n3: spatialize LEFT-BEHIND: emitter (-2, 0, -2), listener at origin facing +Z, right = +X.
    self(3);
    g.nodes[3].kind = kSpatial;
    g.nodes[3].in[0] = 2;
    g.nodes[3].spatial.emitterPos[0] = -131072;   // -2.0
    g.nodes[3].spatial.emitterPos[1] = 0;
    g.nodes[3].spatial.emitterPos[2] = -131072;   // -2.0
    g.nodes[3].spatial.refDist = 131072;          // 2.0 (dist sqrt(8) ~ 2.83 -> gain 4/8 = 0.5)
    // listenerPos/listenerRight/nearDist: the defaults (origin, +X, 1.0).

    // ---- Emitter B: the sine arpeggio, RIGHT-FRONT of the listener -----------------------------------
    // n4: sine (the host steps freqHz through the arp between blocks; 523 is the fixed first note).
    self(4);
    g.nodes[4].kind = kOsc;
    g.nodes[4].wave = dsp::Wave::Sine;
    g.nodes[4].freqHz = 523;
    // n5: the pluck envelope (retriggered by the host every 12000 samples).
    self(5);
    g.nodes[5].kind = kAdsr;
    g.nodes[5].in[0] = 4;
    g.nodes[5].env = dsp::Adsr{240, 1440, 18021, 3360};
    g.nodes[5].durSample = 12000;
    // n6: gain ~0.49 (Q15 16000).
    self(6);
    g.nodes[6].kind = kGain;
    g.nodes[6].in[0] = 5;
    g.nodes[6].gainQ15 = 16000;
    // n7: spatialize RIGHT-FRONT: emitter (1, 0, 2).
    self(7);
    g.nodes[7].kind = kSpatial;
    g.nodes[7].in[0] = 6;
    g.nodes[7].spatial.emitterPos[0] = 65536;     // +1.0
    g.nodes[7].spatial.emitterPos[1] = 0;
    g.nodes[7].spatial.emitterPos[2] = 131072;    // +2.0
    g.nodes[7].spatial.refDist = 131072;          // 2.0 (dist sqrt(5) ~ 2.24 -> gain 4/5 = 0.8)

    // ---- Bus: mix -> delay tail -> out ----------------------------------------------------------------
    // n8: mix the two stereo emitters.
    self(8);
    g.nodes[8].kind = kMix;
    g.nodes[8].in[0] = 3;
    g.nodes[8].in[1] = 7;
    // n9: the delay tail: 9600 samples (0.2 s), feedback ~0.40 (Q15 13000) — a stereo comb.
    self(9);
    g.nodes[9].kind = kDelay;
    g.nodes[9].in[0] = 8;
    g.nodes[9].delaySamples = 9600;
    g.nodes[9].feedbackQ15 = 13000;
    // n10: the sink.
    self(10);
    g.nodes[10].kind = kOut;
    g.nodes[10].in[0] = 9;

    return g;
}

// RenderAu1Showcase: render the FIXED ~2 s scene (96000 frames at 48 kHz, 200 blocks of 480) into
// `outInterleaved` (L,R,...), returning the stat-line numbers. The host schedule (all integer, all
// block-aligned): every 24000 samples retrigger emitter A's envelope; every 12000 samples step
// emitter B's oscillator to the next arpeggio note (C5 E5 G5 C6) and retrigger its envelope.
// Deterministic of nothing but the code — two calls (and two platforms) are byte-identical.
inline Au1ShowcaseStats RenderAu1Showcase(std::vector<int16_t>& outInterleaved) {
    const int kSR     = 48000;
    const int kBlock  = 480;      // 10 ms
    const int kBlocks = 200;      // 96000 frames == 2 s
    const uint32_t kArp[4] = {523, 659, 784, 1047};   // C5 E5 G5 C6

    Graph g = MakeAu1SceneGraph();
    GraphState st = MakeGraphState(g);

    outInterleaved.clear();
    outInterleaved.reserve(static_cast<std::size_t>(kBlock) * kBlocks * 2u);
    for (int b = 0; b < kBlocks; ++b) {
        const int sample = b * kBlock;
        if (sample % 24000 == 0) st.st[1].envT = 0;                    // retrigger pulse env (n1)
        if (sample % 12000 == 0) {
            g.nodes[4].freqHz = kArp[(sample / 12000) % 4];            // step the arp note (n4)
            st.st[5].envT = 0;                                          // retrigger pluck env (n5)
        }
        RenderBlock(g, st, kSR, kBlock, outInterleaved);
    }

    Au1ShowcaseStats stats;
    stats.nodes  = static_cast<uint32_t>(g.nodes.size());
    stats.edges  = CountEdges(g);
    stats.frames = kBlock * kBlocks;
    stats.digest = DigestAudioBlock(outInterleaved);
    return stats;
}

}  // namespace hf::audio::graph
