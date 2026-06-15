# Slice AU — Multithreaded Command Recording (Phase 3 perf #4) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. Builds on the AT validation gate (the new
> secondary-buffer path must stay validation-clean).

**Goal:** Record a draw-heavy pass's draw list across N worker threads in parallel, then assemble the
frame. On Vulkan: per-thread `VkCommandPool` → each worker records a SECONDARY command buffer over its
draw sub-range → the primary executes them with `vkCmdExecuteCommands`. On Metal:
`MTLParallelRenderCommandEncoder` vends N sub-encoders from one render pass, each driven by a worker.
This is a hallmark high-performance-engine capability. Correctness is proven by an oracle that can't be
faked: rendering the SAME scene with **1 worker vs N workers is BYTE-IDENTICAL**.

## Why this is verifiable (determinism oracle)

Draw output must be independent of how the work is split across threads. Guarantee it by construction:
- Partition the pass's draw list into **N contiguous, deterministic index ranges** (worker `k` records
  draws `[k*span, (k+1)*span)`), NOT a work-stealing/atomic queue (whose assignment order is
  nondeterministic).
- Execute the secondaries / commit the sub-encoders **in worker-index order**, so the final command
  stream has the SAME draw order as single-threaded recording.
- Therefore: 1-worker and N-worker renders are bit-identical, and existing single-threaded scenes are
  unchanged. The proof: `--mt-shot --workers 1` and `--mt-shot --workers 4` produce byte-identical
  captures (Vulkan SHA match; Metal two N=4 runs DIFF 0.0000 AND N=1==N=4). All existing 25 goldens stay
  byte-identical (they keep their current single-threaded path, or route through N=1).

## Design decisions (locked)

1. **RHI seam: secondary/parallel recording (additive).** Add a backend-agnostic interface for
   thread-parallel recording. Concretely (adapt names to the existing RHI style):
   - `IRHIDevice::CreateSecondaryCommandBuffer(threadIndex)` → an `ICommandBuffer` a worker thread can
     record draws into, inside the current render pass's inheritance context. Vulkan: a secondary
     `VkCommandBuffer` from a per-thread `VkCommandPool` (pools are NOT thread-safe — one pool per worker
     thread, created up front), begun with `VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT` +
     `VkCommandBufferInheritanceRenderingInfo` (dynamic rendering inheritance: color/depth formats,
     sample count). Metal: a sub-encoder from `MTLParallelRenderCommandEncoder`.
   - `ICommandBuffer::ExecuteSecondaries(span<ICommandBuffer*>)` (primary only) → Vulkan
     `vkCmdExecuteCommands` in order; Metal: closing the parallel encoder commits the sub-encoders in
     creation order (so creation order = worker index).
   - The pass's normal Begin* opens the render pass; with multithreading the primary does NOT record
     draws directly — it creates the parallel context, workers record secondaries, primary executes them,
     then End*. Single-threaded path (N=1) must remain EXACTLY the current behavior (ideally literally the
     same code with one worker) so existing goldens can't move.
   - ALL `vk*`/`MTL*` types/calls stay INSIDE the backend dirs. The seam exposes only abstract
     `ICommandBuffer`/thread-index/format-inheritance in backend-agnostic terms.

2. **Thread pool (engine side, deterministic).** A small fixed-size worker pool in `engine/runtime/` (or
   reuse if one exists). `RecordParallel(drawCount, workerCount, fn)` splits `[0,drawCount)` into
   `workerCount` contiguous ranges, dispatches `fn(threadIndex, begin, end)` to each worker, joins. No
   atomics in the partition (ranges are precomputed). The join is a hard barrier before
   ExecuteSecondaries. Determinism: ranges + execution order are pure functions of (drawCount,
   workerCount). Unit-tested (`tests/parallel_record_test.cpp`): the partition of K items into W workers
   is contiguous, covers exactly `[0,K)` with no gaps/overlaps, and is identical run-to-run; degenerate
   cases (W>K, K=0, W=1).

3. **Showcase `--mt-shot <out> [--workers N]` (Vulkan) / `--mt` (Metal).** A draw-heavy scene — reuse the
   instanced/many-object capstone or a grid of distinct draws (enough draws that splitting is meaningful;
   use NON-instanced distinct draws so each is a separate recorded command, exercising real parallel
   recording, not one instanced call). Records the scene pass across N=4 workers via the parallel path,
   captures one frame. Prints `mt: {workers: N, draws: D, secondaries: N}`. New golden
   `tests/golden/metal/mt.png` (Metal N=4, two runs DIFF 0.0000). The determinism proof: the showcase (or
   the verify script) also renders with `--workers 1` and asserts byte-identical to `--workers 4`.

4. **Existing scenes unchanged.** All other showcases keep their current single-threaded recording (or
   pass through the parallel path with workerCount=1, which must be byte-identical). `git diff master
   --stat -- tests/golden/metal` shows ONLY `mt.png` added; the other 25 byte-identical.

5. **Stay validation-clean (AT gate).** The new secondary-buffer path must pass the AT Vulkan-validation
   smoke: secondaries need correct inheritance info + the primary's `vkCmdBeginRendering` must use
   `VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT` (or the dynamic-rendering equivalent flag) so the
   layer doesn't flag mismatched contents. Run `--mt-shot` under validation → ZERO errors. Per-thread
   pools avoid the pool-thread-safety VUIDs.

6. **Introspect.** Add exactly `multithreaded-recording` (features) + `--mt-shot` (showcases). One-pattern
   rebake, no other drift.

## RHI seam additions (summary)
- `CreateSecondaryCommandBuffer(threadIndex)` + `ExecuteSecondaries(span)` (or equivalent), plus the
  render-pass "expects secondaries" flag on Begin* — additive, default no-op/single-threaded fallback,
  both backends implement. All `vk*`/`MTL*` inside backend dirs; comments referencing the flags tolerated.
- Engine `RecordParallel` + worker pool are pure above-seam logic (no backend symbols). Seam grep: no new
  real backend CODE symbols above the seam (audit any tick-up = comments only).

## Out of scope (YAGNI)
Parallel recording of MULTIPLE passes concurrently / async queues, persistent secondary caching across
frames, GPU-driven / draw-indirect batching beyond AR, job-graph/task-system generality, work-stealing,
lock-free queues, recording the shadow pass in parallel (main scene pass only this slice). Fixed worker
pool, contiguous deterministic partition, in-order execution, 1-vs-N byte-identical.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 26) + new `parallel_record_test` (deterministic
   contiguous partition + degenerate cases). Clean under `windows-msvc-asan` (the worker pool is real
   threading — ASan/TSan-style data-race freedom matters; ensure no shared mutable state across workers
   except disjoint command buffers/ranges).
2. **Determinism oracle:** `--mt-shot --workers 1` capture == `--mt-shot --workers 4` capture
   (byte-identical, SHA match) on Vulkan; on Metal N=1==N=4 and two N=4 runs DIFF 0.0000.
3. `--mt-shot` on Windows/Vulkan: controller visual review — the draw-heavy scene renders correctly
   (identical to its single-threaded form); stat line sane.
4. Metal: `visual_test --mt` → new golden `tests/golden/metal/mt.png`; two runs DIFF 0.0000.
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `mt.png` added; other
   25 byte-identical.
6. **Validation-clean:** `--mt-shot` under the AT Vulkan-validation gate → ZERO errors (correct secondary
   inheritance + contents flag).
7. Introspect JSON rebaked exactly `+multithreaded-recording` + `--mt-shot`; introspect test updated.
8. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `mt` image
   golden in the Mac round-trip loop AND (ideally) the 1-vs-N determinism assertion on Windows.
