#pragma once
#include <cstdint>
#include <functional>
#include <vector>

// Deterministic, contiguous partition of a draw list across N worker threads, plus a tiny fixed
// worker pool that records the partition in parallel (Slice AU — multithreaded command recording).
//
// This module is PURE above-seam logic: it knows nothing about Vulkan/Metal. It only decides WHICH
// contiguous index range each worker owns and dispatches a callback per range. The callback is what
// records draws into a backend secondary command buffer (the backend symbols live behind the RHI
// seam). The whole determinism guarantee rests here: the partition + execution order are pure
// functions of (drawCount, workerCount), so 1-worker and N-worker renders are byte-identical.
namespace hf::runtime {

// A half-open contiguous index range [begin, end) of the draw list assigned to one worker.
struct DrawRange {
    uint32_t begin = 0;  // first draw index (inclusive)
    uint32_t end   = 0;  // one past the last draw index (exclusive); end>=begin, empty if ==
};

// Split [0, drawCount) into exactly `workerCount` CONTIGUOUS, NON-OVERLAPPING ranges that together
// cover [0, drawCount) with no gaps. The split is deterministic (run-to-run identical) and a pure
// function of its arguments — NO atomics, NO work-stealing. The first (drawCount % workerCount)
// workers get one extra item so the ranges differ by at most one and stay contiguous in index order
// (worker k owns a prefix-stable block). Degenerate cases are well-defined:
//   * workerCount == 0 is treated as 1 (a single range covering everything).
//   * workerCount  > drawCount yields empty ranges for the surplus workers (begin==end), still
//     contiguous and exactly covering [0, drawCount).
//   * drawCount == 0 yields all-empty ranges.
// Always returns exactly max(workerCount,1) ranges so the caller can index ranges[k] for every
// worker it spawns.
std::vector<DrawRange> PartitionDraws(uint32_t drawCount, uint32_t workerCount);

// Record `drawCount` draws across `workerCount` worker threads. Computes the contiguous partition,
// dispatches fn(threadIndex, beginDraw, endDraw) to each worker, and JOINS (a hard barrier) before
// returning. fn for worker k is invoked with the [begin,end) of ranges[k]; empty ranges are still
// invoked (so the backend can begin+end an empty secondary, keeping the secondary count == worker
// count). Workers share NO mutable state except their disjoint index ranges and whatever disjoint
// per-thread resources fn closes over (e.g. one secondary command buffer per thread index) — so the
// real threading is data-race free. workerCount==1 (or 0) runs fn(0, 0, drawCount) ON THE CALLING
// THREAD with no thread spawned, so the single-threaded path is exactly the historical behavior.
void RecordParallel(uint32_t drawCount, uint32_t workerCount,
                    const std::function<void(uint32_t threadIndex, uint32_t beginDraw,
                                             uint32_t endDraw)>& fn);

}  // namespace hf::runtime
