#include "runtime/parallel_record.h"

#include <thread>

namespace hf::runtime {

std::vector<DrawRange> PartitionDraws(uint32_t drawCount, uint32_t workerCount) {
    const uint32_t W = (workerCount == 0) ? 1u : workerCount;
    std::vector<DrawRange> ranges(W);

    // Contiguous block split: base = floor(K/W), the first (K%W) workers get one extra item. This
    // keeps the ranges contiguous in worker-index order (a prefix-stable block per worker) AND
    // balanced (lengths differ by at most one). It is a pure function of (drawCount, workerCount):
    // no atomics, no ordering ambiguity, identical run-to-run.
    const uint32_t base = drawCount / W;
    const uint32_t rem  = drawCount % W;
    uint32_t cursor = 0;
    for (uint32_t k = 0; k < W; ++k) {
        const uint32_t len = base + (k < rem ? 1u : 0u);
        ranges[k].begin = cursor;
        ranges[k].end   = cursor + len;
        cursor += len;
    }
    // cursor == drawCount by construction (base*W + rem == drawCount).
    return ranges;
}

void RecordParallel(uint32_t drawCount, uint32_t workerCount,
                    const std::function<void(uint32_t, uint32_t, uint32_t)>& fn) {
    const uint32_t W = (workerCount == 0) ? 1u : workerCount;

    // Single-threaded fast path: run fn on the CALLING thread, no thread spawned. This is the exact
    // historical behavior the existing goldens depend on (workerCount==1 must be byte-identical).
    if (W == 1) {
        fn(0, 0, drawCount);
        return;
    }

    const std::vector<DrawRange> ranges = PartitionDraws(drawCount, W);

    // Spawn W-1 workers for ranges [1..W); record range[0] on the calling thread so we use exactly
    // W threads total. Each worker touches ONLY its own disjoint range + whatever disjoint per-thread
    // resources fn closes over — no shared mutable state, so this is data-race free. The join below
    // is a hard barrier before the caller proceeds to ExecuteSecondaries.
    std::vector<std::thread> workers;
    workers.reserve(W - 1);
    for (uint32_t k = 1; k < W; ++k) {
        const DrawRange r = ranges[k];
        workers.emplace_back([&fn, k, r] { fn(k, r.begin, r.end); });
    }
    fn(0, ranges[0].begin, ranges[0].end);
    for (auto& t : workers) t.join();
}

}  // namespace hf::runtime
