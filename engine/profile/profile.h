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

// ============================ SLICE PROFILE-S4 — draw-call / GPU-pass inspection =====================
// APPEND-ONLY below S3 (do NOT modify S1/S2/S3 — 0xedc7791443141dfd / 0xb41eb67a1d13443e /
// 0xc68ff46e1ab25f37 stay pinned). S4 adds the RENDER-STRUCTURE inspection: a deterministic record of the
// frame's render passes + their draw calls (pass order, draw counts, instance counts, pipeline ids), with
// its OWN pinned structural digest. This is the deterministic STRUCTURE of the render — NOT GPU timing
// (that is the S6 non-golden overlay).
//
// THE CLANG-PURITY BOUNDARY: the real render structure lives in render::RenderGraph (render_graph.h), which
// pulls rhi.h + <functional> + <string> → NOT standalone-clang-compilable. So profile.h NEVER #includes it.
// Instead S4 defines a plain-POD RenderStructInput (vectors of integers) that the live engine (S6) fills
// from RenderGraph::LastOrder() + per-pass draw enumeration — the exact replay.h serWorld injected boundary.
// S4 is STANDALONE: it builds its own RenderStructure and does NOT mutate S1's Capture (EncodeStructural is
// untouched). NO new include. NO timing in the render digest — counts + ids only.

// --- The render-structure records (integers — structure, NOT timing) --------------------------------
struct DrawRecord {
    uint32_t passId        = 0;   // index of the owning pass in execution order
    uint32_t pipelineId    = 0;   // the bound pipeline/PSO id (a stable engine-side id, not a pointer)
    uint32_t drawCount     = 0;   // 1 for a normal draw; N for an MDI draw that collapses N objects
    uint32_t instanceCount = 0;   // instances in this draw
    uint32_t indexCount    = 0;   // indices drawn (0 for non-indexed)
};
struct PassRecord {
    uint32_t nameId         = 0;   // the interned pass name id
    uint32_t firstDraw      = 0;   // index of this pass's first DrawRecord in RenderStructure.draws
    uint32_t drawCount      = 0;   // number of draws in this pass
    uint32_t totalInstances = 0;   // sum of instanceCount over this pass's draws
};
struct RenderStructure { std::vector<PassRecord> passes; std::vector<DrawRecord> draws; };

// --- The injected input (a plain POD — the live engine fills this from RenderGraph) ------------------
// One pass as the live caller describes it: a name id + its draws (in submission order). The caller builds
// `passes` in RenderGraph::LastOrder() execution order. profile.h NEVER #includes render_graph.h — this POD
// is the boundary (the replay.h serWorld pattern).
struct RenderPassInput { uint32_t nameId = 0; std::vector<DrawRecord> draws; };
struct RenderStructInput { std::vector<RenderPassInput> passes; };  // in execution (topo) order

// --- IngestRenderStructure — POD → the deterministic RenderStructure ---------------------------------
// For each pass IN ORDER (passId == index — execution order is the canonical key, no sorting): record the
// pass's first draw index + accumulate its draws + totalInstances.
inline RenderStructure IngestRenderStructure(const RenderStructInput& in) {
    RenderStructure rs;
    for (std::size_t p = 0; p < in.passes.size(); ++p) {
        const RenderPassInput& pin = in.passes[p];
        const uint32_t passId    = static_cast<uint32_t>(p);
        const uint32_t firstDraw = static_cast<uint32_t>(rs.draws.size());
        uint32_t       totalInstances = 0u;
        for (std::size_t d = 0; d < pin.draws.size(); ++d) {
            const DrawRecord& dr = pin.draws[d];
            rs.draws.push_back(DrawRecord{ passId, dr.pipelineId, dr.drawCount, dr.instanceCount, dr.indexCount });
            totalInstances += dr.instanceCount;
        }
        rs.passes.push_back(PassRecord{ pin.nameId, firstDraw,
                                        static_cast<uint32_t>(pin.draws.size()), totalInstances });
    }
    return rs;
}

// --- DigestRenderStructure — the pinned structural digest (counts + ids, NO timing) -----------------
// Hand-LE in passId/draw order: PutU32(passCount), then per pass nameId/firstDraw/drawCount/totalInstances;
// then PutU32(drawTotal), then per draw passId/pipelineId/drawCount/instanceCount/indexCount; net::DigestBytes.
inline uint64_t DigestRenderStructure(const RenderStructure& rs) {
    std::vector<uint8_t> b;
    PutU32(b, static_cast<uint32_t>(rs.passes.size()));
    for (std::size_t i = 0; i < rs.passes.size(); ++i) {
        const PassRecord& p = rs.passes[i];
        PutU32(b, p.nameId);
        PutU32(b, p.firstDraw);
        PutU32(b, p.drawCount);
        PutU32(b, p.totalInstances);
    }
    PutU32(b, static_cast<uint32_t>(rs.draws.size()));
    for (std::size_t i = 0; i < rs.draws.size(); ++i) {
        const DrawRecord& d = rs.draws[i];
        PutU32(b, d.passId);
        PutU32(b, d.pipelineId);
        PutU32(b, d.drawCount);
        PutU32(b, d.instanceCount);
        PutU32(b, d.indexCount);
    }
    return net::DigestBytes(b.data(), b.size());
}

// --- The deterministic FIXED showcase fixture (pinned forever by the golden) -------------------------
// A fixed 3-pass frame (nameIds 10/11/12 representing Shadow/Lit/Composite — fixed ints, the test documents
// the mapping): pass 0 Shadow 1 draw, pass 1 Lit 1 MDI draw (drawCount 64 = the inspection headline),
// pass 2 Composite 1 draw. Keep FIXED — the golden pins its DigestRenderStructure.
inline RenderStructInput MakeShowcaseRenderInput() {
    RenderStructInput in;
    RenderPassInput shadow;
    shadow.nameId = 10u;
    shadow.draws.push_back(DrawRecord{ 0u, 100u, 1u, 1u, 36u });
    in.passes.push_back(shadow);

    RenderPassInput lit;
    lit.nameId = 11u;
    lit.draws.push_back(DrawRecord{ 0u, 200u, 64u, 64u, 1536u });   // MDI collapses 64 objects
    in.passes.push_back(lit);

    RenderPassInput composite;
    composite.nameId = 12u;
    composite.draws.push_back(DrawRecord{ 0u, 300u, 1u, 1u, 3u });
    in.passes.push_back(composite);
    return in;
}

// ============================ SLICE PROFILE-S5 — THE SCRUB: serializable .capture + seek ============
// APPEND-ONLY below S4 (do NOT modify S1/S2/S3/S4 — 0xedc7791443141dfd / 0xb41eb67a1d13443e /
// 0xc68ff46e1ab25f37 / 0x9b75187d6a4c3bf1 stay pinned). S5 is THE HEADLINE: a serializable `.capture`
// artifact whose STRUCTURE is byte-stable, plus a SCRUB — seek to frame N reproduces the BIT-IDENTICAL
// structural state a from-frame-0 playback reaches at N — built on net::CatchUp (the SAME primitive seq's
// SCRUB==SEEK used). The `.capture` puts the STRUCTURAL bytes FIRST (verbatim S1 EncodeStructural) and the
// TIMING overlay LAST in a SEPARATE length-prefixed section, so the structural digest covers ONLY the
// structural section bytes: corrupting a timing byte leaves the structural digest UNCHANGED; corrupting a
// structural byte diverges at the exact frame. NO new include (net/session.h present — CatchUp/JoinSnapshot/
// DesyncDetector reused read-only). NO recursion, NO <string>/<cmath>/clock/RNG/<unordered_*>/<map>/<algorithm>.

// --- Little-endian byte readers (the inverse of PutU32/PutU64 — pure of side effects, LE) ------------
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

// --- The .capture file header (fixed-layout, serialized field-by-field LE, NEVER memcpy'd) ----------
// File magic "HFCAPF1\0" (8) is DISTINCT from S1's structural-section magic "HFCAP1\0\0".
struct CaptureHeader {
    uint32_t version          = 1;
    uint32_t frameCount       = 0;   // BuildFrameIndex(c).size()
    uint32_t nameCount        = 0;
    uint32_t eventCount       = 0;
    uint32_t structuralByteLen = 0;  // == EncodeStructural(c).size()  (the structural section length)
    uint32_t timingByteLen    = 0;   // == eventCount * 16  (cpuNanos u64 + gpuNanos u64 per event)
    uint32_t keyframeInterval = 0;   // frames between seek keyframes (>=1)
};
constexpr std::size_t kCaptureHeaderLen = 8 /*magic*/ + 7 * 4 /*u32 fields*/;  // = 36
inline constexpr char kCaptureMagic[8] = { 'H', 'F', 'C', 'A', 'P', 'F', '1', '\0' };

// --- EncodeCapture: [magic+header][structuralSection = EncodeStructural(c) VERBATIM][timingSection] ---
// The structural section IS S1's EncodeStructural output, so CaptureStructuralDigest == StructuralDigest(c).
// The timing section starts at offset kCaptureHeaderLen + structuralByteLen, PROVABLY outside the structural
// digest's byte range — per event PutU64(cpuNanos), PutU64(gpuNanos).
inline std::vector<uint8_t> EncodeCapture(const Capture& c, uint32_t keyframeInterval = 1) {
    const std::vector<uint8_t>    structural = EncodeStructural(c);   // S1's encoding, verbatim
    const std::vector<FrameIndex> frames     = BuildFrameIndex(c);
    if (keyframeInterval < 1u) keyframeInterval = 1u;

    CaptureHeader h;
    h.version           = 1u;
    h.frameCount        = static_cast<uint32_t>(frames.size());
    h.nameCount         = static_cast<uint32_t>(c.names.names.size());
    h.eventCount        = static_cast<uint32_t>(c.events.size());
    h.structuralByteLen = static_cast<uint32_t>(structural.size());
    h.timingByteLen     = static_cast<uint32_t>(c.timings.size()) * 16u;
    h.keyframeInterval  = keyframeInterval;

    std::vector<uint8_t> out;
    PutBytes(out, kCaptureMagic, 8);                   // file magic(8)
    PutU32(out, h.version);
    PutU32(out, h.frameCount);
    PutU32(out, h.nameCount);
    PutU32(out, h.eventCount);
    PutU32(out, h.structuralByteLen);
    PutU32(out, h.timingByteLen);
    PutU32(out, h.keyframeInterval);
    PutBytes(out, structural.data(), structural.size());   // STRUCTURAL section (verbatim S1 encoding)
    for (std::size_t i = 0; i < c.timings.size(); ++i) {   // TIMING section (separate, last)
        PutU64(out, c.timings[i].cpuNanos);
        PutU64(out, c.timings[i].gpuNanos);
    }
    return out;
}

// --- DecodeStructural: the inverse of S1's EncodeStructural over bytes[off, off+len) -----------------
// Parses magic "HFCAP1\0\0" + version + nameCount + per name [len, bytes] + eventCount + per event
// [kind, nameId, a, b] into `out.names` + `out.events` (timings are restored separately by DecodeCapture).
// Returns false on truncation / bad magic. NO recursion — a flat offset walk.
inline bool DecodeStructural(const std::vector<uint8_t>& bytes, std::size_t off, std::size_t len,
                             Capture& out) {
    if (off + len > bytes.size()) return false;
    const uint8_t* base = bytes.data() + off;
    std::size_t p = 0;
    if (len < 12u) return false;                                 // magic(8) + version(4)
    const char structMagic[8] = { 'H', 'F', 'C', 'A', 'P', '1', '\0', '\0' };
    for (int i = 0; i < 8; ++i) if (base[i] != static_cast<uint8_t>(structMagic[i])) return false;
    p += 8;
    const uint32_t version = GetU32(base + p); p += 4;
    if (version != 1u) return false;

    if (p + 4 > len) return false;
    const uint32_t nameCount = GetU32(base + p); p += 4;
    out.names.names.clear();
    out.names.names.reserve(static_cast<std::size_t>(nameCount));
    for (uint32_t i = 0; i < nameCount; ++i) {
        if (p + 4 > len) return false;
        const uint32_t nlen = GetU32(base + p); p += 4;
        if (p + nlen > len) return false;
        std::vector<uint8_t> nm(base + p, base + p + nlen);
        out.names.names.push_back(nm);
        p += nlen;
    }
    if (p + 4 > len) return false;
    const uint32_t eventCount = GetU32(base + p); p += 4;
    out.events.clear();
    out.events.reserve(static_cast<std::size_t>(eventCount));
    for (uint32_t i = 0; i < eventCount; ++i) {
        if (p + 16 > len) return false;
        CaptureEvent e;
        e.kind   = static_cast<EvKind>(GetU32(base + p)); p += 4;
        e.nameId = GetU32(base + p); p += 4;
        e.a      = GetU32(base + p); p += 4;
        e.b      = GetU32(base + p); p += 4;
        out.events.push_back(e);
    }
    return true;
}

// --- DecodeCapture: the inverse of EncodeCapture — header + structural section + timing section -------
// Returns false on truncation / bad file magic / bad version / structural-parse failure.
inline bool DecodeCapture(const std::vector<uint8_t>& bytes, Capture& out) {
    if (bytes.size() < kCaptureHeaderLen) return false;
    const uint8_t* p = bytes.data();
    for (int i = 0; i < 8; ++i) if (p[i] != static_cast<uint8_t>(kCaptureMagic[i])) return false;
    CaptureHeader h;
    h.version           = GetU32(p + 8);
    h.frameCount        = GetU32(p + 12);
    h.nameCount         = GetU32(p + 16);
    h.eventCount        = GetU32(p + 20);
    h.structuralByteLen = GetU32(p + 24);
    h.timingByteLen     = GetU32(p + 28);
    h.keyframeInterval  = GetU32(p + 32);
    if (h.version != 1u) return false;

    const std::size_t structOff = kCaptureHeaderLen;
    const std::size_t timingOff = structOff + h.structuralByteLen;
    if (timingOff + h.timingByteLen > bytes.size()) return false;
    if (h.timingByteLen != h.eventCount * 16u) return false;

    Capture c;
    if (!DecodeStructural(bytes, structOff, h.structuralByteLen, c)) return false;
    if (c.events.size() != h.eventCount) return false;

    c.timings.clear();
    c.timings.reserve(static_cast<std::size_t>(h.eventCount));
    for (uint32_t i = 0; i < h.eventCount; ++i) {
        const std::size_t to = timingOff + static_cast<std::size_t>(i) * 16u;
        TimingSample ts;
        ts.cpuNanos = GetU64(bytes.data() + to);
        ts.gpuNanos = GetU64(bytes.data() + to + 8);
        c.timings.push_back(ts);
    }
    out = c;
    return true;
}

// --- CaptureStructuralDigest: DigestBytes over ONLY the structural section bytes ---------------------
// [kCaptureHeaderLen .. kCaptureHeaderLen + structuralByteLen) — equals StructuralDigest(c) by construction
// (the section IS the S1 encoding). Timing lives at >= kCaptureHeaderLen + structuralByteLen, PROVABLY
// excluded by byte offset.
inline uint64_t CaptureStructuralDigest(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < kCaptureHeaderLen + 4u) return net::DigestBytes(nullptr, 0);
    const uint32_t structuralByteLen = GetU32(bytes.data() + 24);
    const std::size_t off = kCaptureHeaderLen;
    if (off + structuralByteLen > bytes.size()) return net::DigestBytes(nullptr, 0);
    return net::DigestBytes(bytes.data() + off, structuralByteLen);
}

// --- The scrub playback world: the current frame + a running fold of every frame's structural digest --
// Flat + value-copyable so net::Session's snapshot is complete by construction (the seq.h discipline).
struct CaptureWorld { uint32_t currentFrame = 0; uint64_t acc = 0; };

// --- Mix: an FNV-step fold (acc * FNV-prime) ^ digest — deterministic, integer ----------------------
inline uint64_t Mix(uint64_t acc, uint64_t digest) { return (acc * 1099511628211ull) ^ digest; }

// --- DigestCaptureWorld: hand-LE (currentFrame, acc) -> DigestBytes (the per-frame replay checksum) --
inline uint64_t DigestCaptureWorld(const CaptureWorld& w) {
    std::vector<uint8_t> b;
    PutU32(b, w.currentFrame);
    PutU64(b, w.acc);
    return net::DigestBytes(b.data(), b.size());
}

// --- SeekToFrame: restore the nearest keyframe <= N and replay forward to N via net::CatchUp ----------
// A thin wrapper over net::CatchUp(JoinSnapshot{keyframeFrame, keyframeWorld}, toFrame, tail, step) — the
// structural state at toFrame is BIT-IDENTICAL to a from-0 playback. The composition IS the point.
template <class StepFn>
inline CaptureWorld SeekToFrame(const std::vector<FrameIndex>& /*frames*/, uint32_t toFrame,
                                const CaptureWorld& keyframeWorld, uint32_t keyframeFrame,
                                const hf::net::InputRing<uint32_t>& tail, StepFn step) {
    const hf::net::JoinSnapshot<CaptureWorld> snap{ keyframeFrame, keyframeWorld };
    return hf::net::CatchUp(snap, toFrame, tail, step);
}

// --- VerifyCapture — structural integrity via net::DesyncDetector (the replay.h VerifyReplay pattern) -
// Replay the decoded capture's BuildFrameIndex per-frame digests as the "local" trace; compare against the
// expected per-frame digest vector (the authority). A structural corruption diverges at the EXACT frame; a
// timing corruption does NOT (the per-frame digest is over structural events only).
struct VerifyResult { bool ok = true; uint32_t firstBadFrame = 0; };
inline VerifyResult VerifyCapture(const Capture& decoded, const std::vector<uint64_t>& expectedFrameDigests) {
    const std::vector<FrameIndex> frames = BuildFrameIndex(decoded);
    const std::size_t n = frames.size() < expectedFrameDigests.size()
                              ? frames.size() : expectedFrameDigests.size();
    net::DesyncDetector d;
    for (std::size_t t = 0; t < n; ++t)
        net::RecordLocal(d, static_cast<uint32_t>(t), frames[t].structuralDigest);  // local = the decoded capture
    for (std::size_t t = 0; t < n; ++t)
        net::IngestRemote(d, net::ChecksumPacket{ static_cast<uint32_t>(t), expectedFrameDigests[t] });  // expected
    VerifyResult r;
    r.ok            = !d.desynced;
    r.firstBadFrame = d.desyncTick;
    return r;
}

// --- The deterministic FIXED timeline fixture WITH nonzero timings (so the timing section is corruptible)
// MakeTimelineCapture() but timings[i] = { (i+1)*1000, (i+1)*7 } (nonzero, FIXED). The structural content is
// identical to MakeTimelineCapture so StructuralDigest is UNCHANGED (timings never enter the structural path).
inline Capture MakeTimelineCaptureTimed() {
    Capture c = MakeTimelineCapture();
    for (std::size_t i = 0; i < c.timings.size(); ++i) {
        c.timings[i].cpuNanos = (static_cast<uint64_t>(i) + 1ull) * 1000ull;
        c.timings[i].gpuNanos = (static_cast<uint64_t>(i) + 1ull) * 7ull;
    }
    return c;
}

}  // namespace hf::profile
