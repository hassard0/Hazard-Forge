// Slice AU — deterministic contiguous draw-list partition + parallel worker-pool recorder.
//
// The whole multithreaded-recording determinism oracle rests on PartitionDraws being a pure,
// contiguous, gap/overlap-free, run-to-run-identical split of [0,K) into W ranges. These tests pin
// exactly that, plus the degenerate cases (W>K, K=0, W=1, W=0) and that RecordParallel actually
// drives every range concurrently with no shared mutable state (each worker writes only its own
// disjoint slots — clean under ASan with real threads).

#include "runtime/parallel_record.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

using hf::runtime::DrawRange;
using hf::runtime::PartitionDraws;
using hf::runtime::RecordParallel;

namespace {

int g_failures = 0;
#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                              \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
            ++g_failures;                                                           \
        }                                                                            \
    } while (0)

// Assert: ranges are contiguous, non-overlapping, cover exactly [0,K), and there are exactly
// max(W,1) of them. Returns the covered count for extra sanity.
void CheckCoversExactly(const std::vector<DrawRange>& r, uint32_t K, uint32_t W) {
    const uint32_t expectCount = (W == 0) ? 1u : W;
    CHECK(r.size() == expectCount);
    uint32_t cursor = 0;
    for (const auto& seg : r) {
        CHECK(seg.begin == cursor);   // contiguous: each starts where the previous ended
        CHECK(seg.end >= seg.begin);  // never inverted (empty allowed)
        cursor = seg.end;
    }
    CHECK(cursor == K);  // covers exactly [0,K)
}

// The "differ by at most one" balance property — no worker is starved while another hoards.
void CheckBalanced(const std::vector<DrawRange>& r) {
    uint32_t lo = 0xFFFFFFFFu, hi = 0;
    for (const auto& seg : r) {
        uint32_t len = seg.end - seg.begin;
        if (len < lo) lo = len;
        if (len > hi) hi = len;
    }
    CHECK(hi - lo <= 1);
}

void TestEvenSplit() {
    // 100 draws / 4 workers -> 25 each, contiguous.
    auto r = PartitionDraws(100, 4);
    CheckCoversExactly(r, 100, 4);
    CheckBalanced(r);
    CHECK(r[0].begin == 0 && r[0].end == 25);
    CHECK(r[1].begin == 25 && r[1].end == 50);
    CHECK(r[2].begin == 50 && r[2].end == 75);
    CHECK(r[3].begin == 75 && r[3].end == 100);
}

void TestUnevenSplit() {
    // 10 / 3: first (10%3==1) worker gets the extra -> [0,4)[4,7)[7,10).
    auto r = PartitionDraws(10, 3);
    CheckCoversExactly(r, 10, 3);
    CheckBalanced(r);
    CHECK(r[0].end - r[0].begin == 4);
    CHECK(r[1].end - r[1].begin == 3);
    CHECK(r[2].end - r[2].begin == 3);

    // 17 / 5: remainder 2 -> first two workers get 4, rest get 3.
    auto r2 = PartitionDraws(17, 5);
    CheckCoversExactly(r2, 17, 5);
    CheckBalanced(r2);
    CHECK(r2[0].end - r2[0].begin == 4);
    CHECK(r2[1].end - r2[1].begin == 4);
    CHECK(r2[2].end - r2[2].begin == 3);
}

void TestDegenerate() {
    // W == 1: one range covering everything (the single-threaded path).
    auto w1 = PartitionDraws(42, 1);
    CheckCoversExactly(w1, 42, 1);
    CHECK(w1[0].begin == 0 && w1[0].end == 42);

    // W == 0 treated as 1.
    auto w0 = PartitionDraws(42, 0);
    CheckCoversExactly(w0, 42, 0);
    CHECK(w0.size() == 1);
    CHECK(w0[0].begin == 0 && w0[0].end == 42);

    // K == 0: all-empty ranges, still exactly W of them.
    auto k0 = PartitionDraws(0, 4);
    CheckCoversExactly(k0, 0, 4);
    for (const auto& seg : k0) CHECK(seg.begin == seg.end);

    // W > K: surplus workers get empty ranges; coverage still exact.
    auto wGtK = PartitionDraws(3, 8);
    CheckCoversExactly(wGtK, 3, 8);
    CheckBalanced(wGtK);  // 0/1 lengths -> max diff 1
    // First 3 workers own one draw each, the rest are empty.
    CHECK(wGtK[0].begin == 0 && wGtK[0].end == 1);
    CHECK(wGtK[1].begin == 1 && wGtK[1].end == 2);
    CHECK(wGtK[2].begin == 2 && wGtK[2].end == 3);
    CHECK(wGtK[3].begin == 3 && wGtK[3].end == 3);
    CHECK(wGtK[7].begin == 3 && wGtK[7].end == 3);

    // K == 1, W == 1.
    auto one = PartitionDraws(1, 1);
    CheckCoversExactly(one, 1, 1);
    CHECK(one[0].begin == 0 && one[0].end == 1);
}

void TestDeterministicRunToRun() {
    // Identical inputs -> byte-identical partition every call. This is the determinism the
    // 1-vs-N byte-identical render proof depends on.
    for (uint32_t K : {0u, 1u, 7u, 64u, 1000u}) {
        for (uint32_t W : {1u, 2u, 3u, 4u, 8u, 16u}) {
            auto a = PartitionDraws(K, W);
            auto b = PartitionDraws(K, W);
            CHECK(a.size() == b.size());
            for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                CHECK(a[i].begin == b[i].begin && a[i].end == b[i].end);
            }
            CheckCoversExactly(a, K, W);
        }
    }
}

// The KEY equivalence the render oracle exploits: the SET of draw indices visited, IN ORDER, is
// independent of worker count. Concatenating the ranges in worker-index order reproduces 0,1,..,K-1.
void TestOrderIndependentOfWorkerCount() {
    const uint32_t K = 257;
    std::vector<uint32_t> ref;
    for (uint32_t i = 0; i < K; ++i) ref.push_back(i);
    for (uint32_t W : {1u, 2u, 3u, 5u, 13u, 64u, 300u}) {
        auto r = PartitionDraws(K, W);
        std::vector<uint32_t> seq;
        for (const auto& seg : r)
            for (uint32_t i = seg.begin; i < seg.end; ++i) seq.push_back(i);
        CHECK(seq == ref);
    }
}

// RecordParallel must invoke fn for EVERY worker (including empty ranges) exactly once, with the
// partition's ranges, and the real threading must be race-free: each worker writes only its own
// disjoint slots. We mark each draw index by exactly one thread and assert full, single-cover.
void TestRecordParallelDispatch() {
    for (uint32_t W : {1u, 2u, 4u, 7u}) {
        const uint32_t K = 1000;
        std::vector<std::atomic<int>> hits(K);
        for (auto& a : hits) a.store(0);
        std::atomic<uint32_t> invocations{0};
        auto expected = PartitionDraws(K, W);

        std::vector<std::pair<uint32_t, uint32_t>> seen(expected.size(), {0xFFFFFFFFu, 0u});
        RecordParallel(K, W, [&](uint32_t t, uint32_t b, uint32_t e) {
            invocations.fetch_add(1, std::memory_order_relaxed);
            seen[t] = {b, e};
            for (uint32_t i = b; i < e; ++i) hits[i].fetch_add(1, std::memory_order_relaxed);
        });

        CHECK(invocations.load() == expected.size());
        for (uint32_t i = 0; i < K; ++i) CHECK(hits[i].load() == 1);  // each draw recorded once
        for (size_t t = 0; t < expected.size(); ++t) {
            CHECK(seen[t].first == expected[t].begin);
            CHECK(seen[t].second == expected[t].end);
        }
    }
}

// W==1 (and W==0) run fn(0,0,K) on the calling thread — the historical single-threaded path.
void TestRecordParallelSingleThread() {
    for (uint32_t W : {0u, 1u}) {
        uint32_t calls = 0, gotBegin = 99, gotEnd = 99, gotT = 99;
        RecordParallel(50, W, [&](uint32_t t, uint32_t b, uint32_t e) {
            ++calls; gotT = t; gotBegin = b; gotEnd = e;
        });
        CHECK(calls == 1);
        CHECK(gotT == 0 && gotBegin == 0 && gotEnd == 50);
    }
}

}  // namespace

int main() {
    TestEvenSplit();
    TestUnevenSplit();
    TestDegenerate();
    TestDeterministicRunToRun();
    TestOrderIndependentOfWorkerCount();
    TestRecordParallelDispatch();
    TestRecordParallelSingleThread();

    if (g_failures == 0) {
        std::printf("parallel_record_test: ALL PASS\n");
        return 0;
    }
    std::printf("parallel_record_test: %d FAILURE(S)\n", g_failures);
    return 1;
}
