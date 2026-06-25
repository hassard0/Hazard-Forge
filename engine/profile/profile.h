// Slice PROFILE-S1 — Capture event model + interned name table (issue #31, beachhead).
//
// The beachhead of the DETERMINISTIC SCRUB-FRIENDLY PROFILER CAPTURE (issue #31). A profiler measures
// TIME (non-deterministic) — so this is NOT "a deterministic profiler". The moat is a CAPTURE whose
// STRUCTURE is deterministic + replayable + scrub-seekable, with TIMING as a NON-GOLDEN overlay.
//
// ============================ THE LOAD-BEARING INVARIANT (banner it) =================================
// The StructuralDigest covers the event stream's LOGICAL CONTENT ONLY (event kinds, interned names,
// structural payloads). The parallel `timings` overlay (cpu/gpu nanoseconds) is STORED but is NEVER fed
// to StructuralDigest — so the same workload yields the byte-identical structural digest on a fast
// machine and a slow machine. The S1 golden PROVES this: filling `timings` with arbitrary nonzero values
// leaves StructuralDigest UNCHANGED. (S6's live ScopedZone overwrites a timing slot; the structural path
// never reads it.)
// ====================================================================================================
//
// SELF-CONTAINED ON PURPOSE: this header includes ONLY <cstddef>/<cstdint>/<vector> and "net/session.h"
// (itself self-contained) so profile_test.cpp compiles STANDALONE with
// `clang++ -std=c++20 -I engine -I tests tests/profile_test.cpp` on the Mac — the cheap cross-platform
// proof. NO <string> (names are byte-strings — std::vector<uint8_t>), NO replay.h / render_graph.h
// (inline the LE appenders below), NO <cmath> / float / clock / RNG / <random> / <unordered_*> / <map> /
// <functional> / std::hash / <algorithm>. Pure-CPU INTEGER on the bit-exact path.
//
// This is ONE growing header — every later slice (S2–S5) APPENDS a section below S1; do NOT modify S1's
// symbols once pinned. The serialization mirrors the wav.cpp / replay.h discipline VERBATIM: every
// multi-byte field is hand-serialized little-endian, byte-by-byte — NEVER a struct memcpy (which would
// embed host endianness/padding and break the Mac digest).

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "net/session.h"

namespace hf::profile {

// --- Little-endian byte appenders (self-contained — mirror replay.h:29-49, NEVER a struct memcpy) ----
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

// --- Event kinds (FROZEN values — S2+ uses them; never renumber) ------------------------------------
enum class EvKind : uint32_t {
    ScopeEnter = 0, ScopeExit = 1, DrawCall = 2, FrameBegin = 3, FrameEnd = 4,
};

// --- Interned name table: names as raw byte-strings in FIRST-SEEN order ------------------------------
// (No <string>, no pointers in the digest — the chunk_diff.h content-addressing move.)
struct NameTable {
    std::vector<std::vector<uint8_t>> names;   // names[id] = the interned bytes for nameId `id`
};

// Intern `[p, p+n)`: linear-scan for an existing identical byte-string; return its id, else append +
// return the new id. Deterministic (first-seen order). Linear scan is fine — name counts are small; NO
// unordered.
inline uint32_t Intern(NameTable& t, const void* p, std::size_t n) {
    const auto* s = static_cast<const uint8_t*>(p);
    for (std::size_t id = 0; id < t.names.size(); ++id) {
        const std::vector<uint8_t>& cand = t.names[id];
        if (cand.size() != n) continue;
        bool same = true;
        for (std::size_t i = 0; i < n; ++i) { if (cand[i] != s[i]) { same = false; break; } }
        if (same) return static_cast<uint32_t>(id);
    }
    std::vector<uint8_t> bytes(s, s + n);
    t.names.push_back(bytes);
    return static_cast<uint32_t>(t.names.size() - 1);
}

// --- A capture event — ONE structural record, NO timing here ----------------------------------------
struct CaptureEvent {
    EvKind   kind   = EvKind::ScopeEnter;
    uint32_t nameId = 0;   // index into NameTable (ScopeEnter/Draw); 0 for frame markers
    uint32_t a      = 0;   // kind-specific structural payload (e.g. DrawCall drawCount)
    uint32_t b      = 0;   // kind-specific structural payload (reserved; e.g. depth)
};

// --- The NON-golden timing overlay — NEVER fed to StructuralDigest ------------------------------------
struct TimingSample { uint64_t cpuNanos = 0; uint64_t gpuNanos = 0; };

// --- A capture: the structural column + the parallel (same-index) timing overlay column --------------
struct Capture {
    NameTable                  names;
    std::vector<CaptureEvent>  events;    // the structural column
    std::vector<TimingSample>  timings;   // the overlay column — parallel to events (same index)
};

// --- Emitters: push a structural event AND a parallel zero TimingSample so events.size()==timings.size()
inline void EmitEnter(Capture& c, uint32_t nameId) {
    c.events.push_back(CaptureEvent{ EvKind::ScopeEnter, nameId, 0, 0 });
    c.timings.push_back(TimingSample{});
}
inline void EmitExit(Capture& c, uint32_t nameId) {
    c.events.push_back(CaptureEvent{ EvKind::ScopeExit, nameId, 0, 0 });
    c.timings.push_back(TimingSample{});
}
inline void EmitDraw(Capture& c, uint32_t nameId, uint32_t drawCount) {
    c.events.push_back(CaptureEvent{ EvKind::DrawCall, nameId, drawCount, 0 });
    c.timings.push_back(TimingSample{});
}

// --- EncodeStructural: hand-LE the GOLDEN bytes (the timings vector is NOT serialized — the moat) -----
// FIXED order: magic "HFCAP1\0\0" (8) + version(1) + nameCount + per name [len + bytes] + eventCount +
// per event [kind, nameId, a, b]. (S5's full EncodeCapture serializes timings in a SEPARATE section.)
inline std::vector<uint8_t> EncodeStructural(const Capture& c) {
    std::vector<uint8_t> b;
    const char magic[8] = { 'H', 'F', 'C', 'A', 'P', '1', '\0', '\0' };
    PutBytes(b, magic, 8);
    PutU32(b, 1u);   // version
    PutU32(b, static_cast<uint32_t>(c.names.names.size()));
    for (std::size_t i = 0; i < c.names.names.size(); ++i) {
        const std::vector<uint8_t>& nm = c.names.names[i];
        PutU32(b, static_cast<uint32_t>(nm.size()));
        PutBytes(b, nm.data(), nm.size());
    }
    PutU32(b, static_cast<uint32_t>(c.events.size()));
    for (std::size_t i = 0; i < c.events.size(); ++i) {
        const CaptureEvent& e = c.events[i];
        PutU32(b, static_cast<uint32_t>(e.kind));
        PutU32(b, e.nameId);
        PutU32(b, e.a);
        PutU32(b, e.b);
    }
    return b;
}

// --- StructuralDigest: FNV-1a-64 over the structural encoding (TIMING EXCLUDED by construction) -------
inline uint64_t StructuralDigest(const Capture& c) {
    const std::vector<uint8_t> enc = EncodeStructural(c);
    return net::DigestBytes(enc.data(), enc.size());
}

// --- The deterministic FIXED showcase fixture (pinned forever by the golden) -------------------------
// intern "Frame","Shadow","Lit"; emit Enter Frame; Enter Shadow; Draw(Shadow,2); Exit Shadow; Enter Lit;
// Draw(Lit,5); Exit Lit; Exit Frame. timings all-zero.
inline Capture MakeShowcaseCapture() {
    Capture c;
    const char kFrame[]  = { 'F', 'r', 'a', 'm', 'e' };
    const char kShadow[] = { 'S', 'h', 'a', 'd', 'o', 'w' };
    const char kLit[]    = { 'L', 'i', 't' };
    const uint32_t frame  = Intern(c.names, kFrame,  sizeof(kFrame));
    const uint32_t shadow = Intern(c.names, kShadow, sizeof(kShadow));
    const uint32_t lit    = Intern(c.names, kLit,    sizeof(kLit));

    EmitEnter(c, frame);
    EmitEnter(c, shadow);
    EmitDraw (c, shadow, 2);
    EmitExit (c, shadow);
    EmitEnter(c, lit);
    EmitDraw (c, lit, 5);
    EmitExit (c, lit);
    EmitExit (c, frame);
    return c;
}

// ============================ SLICE PROFILE-S2 — hierarchical scope/zone TREE ========================
// APPEND-ONLY below S1 (do NOT modify S1 — the structural digest 0xedc7791443141dfd stays pinned). S2
// reconstructs the scope/zone TREE from the flat Enter/Exit event stream (the hierarchical "track" view a
// profiler UI shows) and aggregates draw counts bottom-up. Pure-integer + deterministic: the tree has its
// own pinned digest; an unbalanced Enter/Exit stream resolves to a deterministic canonical tree (never UB).
// NO new include. NO recursion — every traversal uses an explicit integer work-stack (std::vector<uint32_t>).
// The tree is built from `events` ONLY — `timings` is NEVER read.

// --- A scope-tree node — INDICES, never pointers (the chunk_diff.h discipline) ----------------------
struct ScopeNode {
    uint32_t nameId           = 0;          // the interned scope name (the synthetic ROOT uses kRootName)
    uint32_t parent           = 0;          // index into nodes[]; the root is its own parent (index 0)
    uint32_t firstChild       = 0;          // index of the first child, or kNoNode
    uint32_t nextSibling      = 0;          // index of the next sibling, or kNoNode
    uint32_t selfDrawCount    = 0;          // draws emitted DIRECTLY in this scope (not in children)
    uint32_t subtreeDrawCount = 0;          // selfDrawCount + sum of all descendants' subtreeDrawCount
};
constexpr uint32_t kNoNode   = 0xFFFFFFFFu; // the "no node" sentinel for firstChild/nextSibling
constexpr uint32_t kRootName = 0xFFFFFFFFu; // the synthetic root's reserved nameId

struct ScopeTree {
    std::vector<ScopeNode> nodes;           // nodes[0] is the synthetic ROOT (parent of all top-level scopes)
    bool                   balanced = true; // false if the event stream had unbalanced Enter/Exit
};

// --- BuildScopeTree — one integer open-scope-stack pass (NO recursion, NO <stack>) -------------------
inline ScopeTree BuildScopeTree(const Capture& c) {
    ScopeTree t;
    t.nodes.push_back(ScopeNode{ kRootName, 0u, kNoNode, kNoNode, 0u, 0u });  // root at index 0, parent self
    std::vector<uint32_t> stack;                  // the integer open-scope stack (node indices)
    stack.push_back(0u);                          // the root is always open

    for (std::size_t i = 0; i < c.events.size(); ++i) {
        const CaptureEvent& ev = c.events[i];
        if (ev.kind == EvKind::ScopeEnter) {
            const uint32_t parent   = stack.back();
            const uint32_t newIndex = static_cast<uint32_t>(t.nodes.size());
            t.nodes.push_back(ScopeNode{ ev.nameId, parent, kNoNode, kNoNode, 0u, 0u });
            // Link at the BACK of the parent's child chain (emission order).
            if (t.nodes[parent].firstChild == kNoNode) {
                t.nodes[parent].firstChild = newIndex;
            } else {
                uint32_t sib = t.nodes[parent].firstChild;
                while (t.nodes[sib].nextSibling != kNoNode) sib = t.nodes[sib].nextSibling;
                t.nodes[sib].nextSibling = newIndex;
            }
            stack.push_back(newIndex);
        } else if (ev.kind == EvKind::ScopeExit) {
            if (stack.size() > 1) stack.pop_back();     // close the current scope
            else                  t.balanced = false;   // a stray exit with no matching enter (ignore it)
        } else if (ev.kind == EvKind::DrawCall) {
            t.nodes[stack.back()].selfDrawCount += ev.a; // draws land on the currently-open scope
        }
        // FrameBegin / FrameEnd are ignored by S2 (S3 handles frames).
    }
    if (stack.size() > 1) t.balanced = false;   // open scopes never exited -> unbalanced (truncation)

    // Aggregate subtreeDrawCount bottom-up: a child always has a HIGHER index than its parent, so a single
    // reverse pass suffices (integer-exact, no recursion).
    for (std::size_t k = t.nodes.size(); k-- > 0; ) {
        t.nodes[k].subtreeDrawCount += t.nodes[k].selfDrawCount;
        if (k != 0) t.nodes[t.nodes[k].parent].subtreeDrawCount += t.nodes[k].subtreeDrawCount;
    }
    return t;
}

// --- DigestTree — hand-LE over the nodes in PRE-ORDER (deterministic, NO recursion) -----------------
// Pre-order (root, then each child subtree in firstChild->nextSibling order) via an explicit integer work-
// stack. Per node: PutU32(nameId), selfDrawCount, subtreeDrawCount, childCount. balanced is emitted once at
// the front. Indices are NEVER serialized — pre-order + childCount captures the shape, layout-stable.
inline uint64_t DigestTree(const ScopeTree& t) {
    std::vector<uint8_t> buf;
    PutU32(buf, static_cast<uint32_t>(t.balanced));
    if (t.nodes.empty()) return net::DigestBytes(buf.data(), buf.size());

    std::vector<uint32_t> work;        // the integer work-stack (node indices, LIFO)
    work.push_back(0u);                // start at the root
    while (!work.empty()) {
        const uint32_t n = work.back();
        work.pop_back();
        const ScopeNode& node = t.nodes[n];
        // Count children via the sibling chain.
        uint32_t childCount = 0u;
        for (uint32_t ch = node.firstChild; ch != kNoNode; ch = t.nodes[ch].nextSibling) ++childCount;
        PutU32(buf, node.nameId);
        PutU32(buf, node.selfDrawCount);
        PutU32(buf, node.subtreeDrawCount);
        PutU32(buf, childCount);
        // Push children in REVERSE so they pop in firstChild->nextSibling (emission) order — pre-order.
        std::vector<uint32_t> kids;
        for (uint32_t ch = node.firstChild; ch != kNoNode; ch = t.nodes[ch].nextSibling) kids.push_back(ch);
        for (std::size_t i = kids.size(); i-- > 0; ) work.push_back(kids[i]);
    }
    return net::DigestBytes(buf.data(), buf.size());
}

// ============================ SLICE PROFILE-S3 — frame boundaries + multi-frame TIMELINE =============
// APPEND-ONLY below S2 (do NOT modify S1/S2 — 0xedc7791443141dfd and 0xb41eb67a1d13443e stay pinned).
// S3 adds FrameBegin/FrameEnd markers that bracket each frame's events, plus a TIMELINE: a per-frame
// index where every frame carries its OWN structural sub-digest (the digest of just that frame's events).
// CRUCIAL: a frame's digest depends on ONLY that frame's events — NO cross-frame state, NO timings — which
// is the position-independence that makes S5's seek-to-frame exact. NO new include. NO recursion (integer
// walk). The frame index is built from `events` ONLY — `timings` is NEVER read.

// --- Frame-marker emitters: push the marker + a parallel zero TimingSample (events.size()==timings.size())
inline void EmitFrameBegin(Capture& c, uint32_t frameNumber) {
    c.events.push_back(CaptureEvent{ EvKind::FrameBegin, 0u, frameNumber, 0u });
    c.timings.push_back(TimingSample{});
}
inline void EmitFrameEnd(Capture& c) {
    c.events.push_back(CaptureEvent{ EvKind::FrameEnd, 0u, 0u, 0u });
    c.timings.push_back(TimingSample{});
}

// --- EncodeFrameEvents: hand-LE a SLICE [first, first+count) of the event stream (NO names, NO absolute
// position): PutU32(count), then per event PutU32((uint32_t)kind), PutU32(nameId), PutU32(a), PutU32(b).
// Position-independent by construction — two frames with the identical event RECORDS encode identically.
inline std::vector<uint8_t> EncodeFrameEvents(const Capture& c, uint32_t first, uint32_t count) {
    std::vector<uint8_t> b;
    PutU32(b, count);
    for (uint32_t k = 0; k < count; ++k) {
        const CaptureEvent& e = c.events[static_cast<std::size_t>(first) + k];
        PutU32(b, static_cast<uint32_t>(e.kind));
        PutU32(b, e.nameId);
        // A FrameBegin's `a` is the FRAME NUMBER — pure timeline metadata, NOT structural workload. Encoding
        // it as 0 here is what makes two identical-workload frames (frame 0 vs frame 1) hash IDENTICALLY,
        // the position-independence the spec's per-frame-reproducibility property demands (the frame number
        // is preserved separately in FrameIndex.frameNumber + serialized by DigestTimeline). Every OTHER
        // event encodes its `a` verbatim.
        PutU32(b, (e.kind == EvKind::FrameBegin) ? 0u : e.a);
        PutU32(b, e.b);
    }
    return b;
}
inline uint64_t FrameStructuralDigest(const Capture& c, uint32_t first, uint32_t count) {
    const std::vector<uint8_t> b = EncodeFrameEvents(c, first, count);
    return net::DigestBytes(b.data(), b.size());
}

// --- The frame index (the timeline) — one FrameIndex per complete frame ------------------------------
struct FrameIndex {
    uint32_t frameNumber      = 0;   // the FrameBegin's `a`
    uint32_t firstEvent       = 0;   // index of this frame's FrameBegin in c.events
    uint32_t eventCount       = 0;   // number of events in [FrameBegin .. FrameEnd] inclusive
    uint64_t structuralDigest = 0;   // FrameStructuralDigest over this frame's events (the timeline cell)
};

// --- BuildFrameIndex — split the event stream on FrameBegin/FrameEnd markers (integer walk, NO recursion).
// On FrameBegin record firstEvent/frameNumber + open a frame; on FrameEnd close it. An unclosed final frame
// runs to the end of the stream (deterministic, never UB). Events outside any frame are skipped.
inline std::vector<FrameIndex> BuildFrameIndex(const Capture& c) {
    std::vector<FrameIndex> frames;
    bool     open       = false;
    uint32_t firstEvent = 0u;
    uint32_t frameNumber = 0u;
    for (std::size_t i = 0; i < c.events.size(); ++i) {
        const CaptureEvent& ev = c.events[i];
        if (ev.kind == EvKind::FrameBegin) {
            open        = true;
            firstEvent  = static_cast<uint32_t>(i);
            frameNumber = ev.a;
        } else if (ev.kind == EvKind::FrameEnd) {
            if (open) {
                const uint32_t eventCount = static_cast<uint32_t>(i) - firstEvent + 1u;
                FrameIndex fi;
                fi.frameNumber      = frameNumber;
                fi.firstEvent       = firstEvent;
                fi.eventCount       = eventCount;
                fi.structuralDigest = FrameStructuralDigest(c, firstEvent, eventCount);
                frames.push_back(fi);
                open = false;
            }
            // A stray FrameEnd with no open frame is ignored (deterministic).
        }
    }
    if (open) {   // an unclosed trailing frame -> close it at the end of the stream
        const uint32_t eventCount = static_cast<uint32_t>(c.events.size()) - firstEvent;
        FrameIndex fi;
        fi.frameNumber      = frameNumber;
        fi.firstEvent       = firstEvent;
        fi.eventCount       = eventCount;
        fi.structuralDigest = FrameStructuralDigest(c, firstEvent, eventCount);
        frames.push_back(fi);
    }
    return frames;
}

// --- DigestTimeline — hand-LE over the frame index: PutU32(frameCount), then per frame PutU32(frameNumber),
// PutU32(eventCount), PutU64(structuralDigest). Then net::DigestBytes.
inline uint64_t DigestTimeline(const std::vector<FrameIndex>& frames) {
    std::vector<uint8_t> b;
    PutU32(b, static_cast<uint32_t>(frames.size()));
    for (std::size_t i = 0; i < frames.size(); ++i) {
        PutU32(b, frames[i].frameNumber);
        PutU32(b, frames[i].eventCount);
        PutU64(b, frames[i].structuralDigest);
    }
    return net::DigestBytes(b.data(), b.size());
}

// --- The deterministic FIXED timeline fixture (pinned forever by the golden) -------------------------
// intern "Frame","Shadow","Lit","Post"; build 4 frames. Frame 0 and frame 1 are the IDENTICAL workload
// (different frame number) so their structuralDigest MUST match.
inline Capture MakeTimelineCapture() {
    Capture c;
    const char kFrame[]  = { 'F', 'r', 'a', 'm', 'e' };
    const char kShadow[] = { 'S', 'h', 'a', 'd', 'o', 'w' };
    const char kLit[]    = { 'L', 'i', 't' };
    const char kPost[]   = { 'P', 'o', 's', 't' };
    (void)Intern(c.names, kFrame, sizeof(kFrame));       // "Frame" interned (id 0) for stable parity with S1
    const uint32_t shadow = Intern(c.names, kShadow, sizeof(kShadow));
    const uint32_t lit    = Intern(c.names, kLit,    sizeof(kLit));
    const uint32_t post   = Intern(c.names, kPost,   sizeof(kPost));

    // frame 0: FrameBegin(0); Enter Shadow; Draw(Shadow,2); Exit Shadow; FrameEnd
    EmitFrameBegin(c, 0u);
    EmitEnter(c, shadow);
    EmitDraw (c, shadow, 2);
    EmitExit (c, shadow);
    EmitFrameEnd(c);
    // frame 1: IDENTICAL workload, FrameBegin(1)
    EmitFrameBegin(c, 1u);
    EmitEnter(c, shadow);
    EmitDraw (c, shadow, 2);
    EmitExit (c, shadow);
    EmitFrameEnd(c);
    // frame 2: FrameBegin(2); Enter Lit; Draw(Lit,5); Exit Lit; FrameEnd
    EmitFrameBegin(c, 2u);
    EmitEnter(c, lit);
    EmitDraw (c, lit, 5);
    EmitExit (c, lit);
    EmitFrameEnd(c);
    // frame 3: FrameBegin(3); Enter Post; Draw(Post,1); Exit Post; FrameEnd
    EmitFrameBegin(c, 3u);
    EmitEnter(c, post);
    EmitDraw (c, post, 1);
    EmitExit (c, post);
    EmitFrameEnd(c);
    return c;
}

}  // namespace hf::profile
