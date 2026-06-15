# Slice AS — Render-Graph Automatic Barriers / Resource-State Tracking (Phase 3 perf #3) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan (synchronization-validation oracle) AND Apple M4+Metal
> (render-invariance). Builds on the existing `engine/render/render_graph.{h,cpp}`.

**Goal:** Make the render graph OWN inter-pass GPU synchronization. Today the graph header explicitly
states it "does NOT generate barriers — the RHI's Begin*/End* pass methods still own all of that"
(hand-placed transitions). This slice adds a **resource-state tracker + barrier solver** to the graph: each
resource carries a current access state, each pass declares the state it needs per read/write, and the
graph computes the minimal transition set and emits it through a new additive RHI `ResourceBarrier`
primitive. Correctness is proven by an oracle that is impossible to fake: the **Vulkan synchronization
validation layer** must report ZERO sync hazards across every showcase, while all 25 image goldens stay
BYTE-IDENTICAL.

## Why this is verifiable (oracle + invariance)

1. **Synchronization-validation oracle (Vulkan).** Enable `VK_LAYER_KHRONOS_validation` with the
   **synchronization2 / sync validation** feature (`VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION`).
   A missing or wrong barrier → a `SYNC-HAZARD-*` error. Run every `--*-shot` showcase with sync
   validation ON and require **zero** sync-hazard messages. This is a real GPU-correctness oracle, not a
   visual guess. (If the engine already runs standard validation, ADD the sync-validation feature flag.)
2. **Render-invariance.** Correct synchronization does not change pixels — all 25 image goldens
   (`tests/golden/metal/*.png`) + the introspect scene stay byte-identical. A barrier bug that corrupts an
   image would ALSO show as a golden diff, double-covering it.
3. **Unit-tested solver.** The barrier solver is pure logic (`engine/render/render_graph` resource-state
   model) and is unit-tested independently of any device.

## Design decisions (locked)

1. **Resource-state model.** Add `enum class RgResourceState { Undefined, ColorTarget, DepthWrite,
   DepthRead, ShaderRead, IndirectArgs, TransferSrc, TransferDst, Present };` to the render graph. Each
   `Resource` gains a `currentState` (initialized: external imports start `Undefined`/their imported
   state; document the initial state per `RgResourceKind`). The pass API gains the state each read/write
   resource must be in for that pass — either an explicit per-access state on `AddPass` (preferred: extend
   the reads/writes to `{RgResource, RgResourceState}` pairs) OR inferred from the pass's scaffolding kind
   + access role (ShadowMap-write→DepthWrite, then a reader needs ShaderRead; SceneColor-write→ColorTarget,
   reader→ShaderRead; Swapchain-write→ColorTarget then Present). Pick the cleaner design and document it;
   keep the existing `AddPass` working (an overload or default that infers, so current call sites compile).

2. **Barrier solver (pure logic, the intellectual core).** In topo order, for each pass and each
   resource it accesses: if `currentState != requiredState`, record a transition
   `Barrier{resource, from=currentState, to=requiredState}` and set `currentState = requiredState`. Output
   an ordered `std::vector<Barrier>` interleaved with the passes (the barriers that must be emitted
   BEFORE each pass). Coalesce redundant transitions (same from==to → no barrier). Expose the computed
   plan via a `LastBarriers()` accessor for tests/inspection (mirrors the existing `LastOrder()`).
   Unit-tested: given a hand-built pass DAG + resource accesses, the emitted barrier list matches a
   hand-checked expected sequence (incl. the no-op coalescing and the shadow→shaderRead,
   sceneColor→shaderRead, swapchain→present cases).

3. **RHI primitive (ONE additive seam method).** `IRHICommandBuffer::ResourceBarrier(IRenderTarget&
   resource, RgResourceState from, RgResourceState to)` — WAIT: the RHI must not depend on render-graph
   types. Instead define a backend-agnostic `rhi::ResourceState` enum IN `engine/rhi/rhi.h` and map the
   graph's `RgResourceState` to it (or just USE `rhi::ResourceState` directly in the graph). Method:
   `virtual void ResourceBarrier(IRenderTarget& resource, rhi::ResourceState from, rhi::ResourceState to)
   {}` (default no-op; both backends implement). Vulkan impl: `vkCmdPipelineBarrier2` with an
   `VkImageMemoryBarrier2` mapping the state pair → (srcStage/srcAccess/oldLayout, dstStage/dstAccess/
   newLayout). Metal impl: Metal's default `MTLResourceHazardTrackingModeTracked` auto-tracks
   intra-encoder hazards, so most transitions are no-ops; where a cross-encoder hazard exists Metal
   handles it at encoder boundaries — implement as a no-op or a lightweight `memoryBarrierWithScope:` only
   if a real hazard is found (document the Metal semantics honestly). The `vk*`/`MTL*` calls live ONLY in
   the backend dirs.

4. **Cutover (de-risked, incremental).** The graph emits its computed barriers for the **inter-pass**
   transitions (the shadow-map→shader-read and scene-color→shader-read and swapchain→present transitions
   that Begin*/End* currently hardcode). Where the graph now owns a transition, REMOVE the now-redundant
   hardcoded transition from the corresponding Begin*/End* so synchronization isn't double-applied — BUT
   only remove one once sync-validation confirms the graph's barrier fully covers it. If any transition is
   safer left in Begin/End for this slice, LEAVE it and document why (incremental is fine; the headline is
   the graph CAN and DOES generate barriers, proven by the oracle). Never leave a GAP (sync error) — the
   validation layer is the gate.

5. **No new image golden (invisible, render-invariant slice).** This is a correctness/perf slice with no
   visual change — correctly, there is NO new `.png` golden. The proof is: sync-validation-clean on all
   showcases + 25 existing image goldens byte-identical + the solver unit test. Document this explicitly
   (a slice whose correctness is proven by an oracle + invariance, not a new picture).

6. **Introspect.** Add exactly `automatic-barriers` to the `features` list (no new showcase flag — there's
   no new visual). One-line introspect-golden delta; introspect test updated.

## RHI seam additions (summary)
- `rhi::ResourceState` enum (backend-agnostic) + `IRHICommandBuffer::ResourceBarrier(IRenderTarget&,
  ResourceState, ResourceState)` — additive, default no-op, both backends implement. All
  `vkCmdPipelineBarrier2`/Metal hazard calls stay INSIDE backend dirs.
- Render-graph state model/solver is pure logic above the RHI (no backend symbols). New/modified
  non-backend files add ZERO real backend types (comments referencing layout names are OK). Seam grep:
  no NEW real backend code symbols above the seam (the benign comment baseline may tick up by prose only —
  audit and confirm all additions are comments or `rhi::ResourceState` interface, not `vk*`/`MTL*` code).

## Out of scope (YAGNI)
Aliasing/transient memory reuse, async-compute queue scheduling, split barriers / events, automatic
render-pass merging, buffer-state tracking for the SSBO/indirect buffers (this slice tracks RENDER-TARGET
image states; the compute/indirect buffers from AR keep their current explicit barriers — note this),
multi-queue ownership transfers. One linear topo-ordered barrier solver over render-target resources,
Vulkan-validated, render-invariant.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 25) + new `render_graph_barriers_test` (solver:
   hand-checked barrier sequences incl. coalescing + shadow/sceneColor/swapchain transitions). Clean under
   `windows-msvc-asan`.
2. **Synchronization-validation oracle:** every `--*-shot` showcase run with `VK_LAYER_KHRONOS_validation`
   + sync-validation feature ON reports ZERO `SYNC-HAZARD-*` errors (capture/report the validation output;
   any sync error is a BLOCKER — STOP, do not merge). Confirm validation is actually active (a deliberate
   removed barrier should trigger an error — sanity-check the oracle once, then restore).
3. **Render-invariance:** `git diff master --stat -- tests/golden/metal` is EMPTY (no image golden added or
   changed — all 25 byte-identical). Re-run a representative set on Vulkan headless to confirm unchanged
   output; Metal goldens unchanged (spot-check several on the M4 → DIFF 0.0000).
4. Introspect JSON rebaked with exactly `+automatic-barriers` (features); no showcase added; no other
   drift; introspect test updated.
5. Seam grep: no new real backend code symbols above the seam (audit any tick-up = comments/`rhi::`
   interface only). `scripts/verify.ps1` unchanged re: goldens (no new image golden) — but if it lists
   showcases, leave it consistent.
6. Honest reporting: if the Metal side is largely no-op due to default hazard tracking, SAY so; the
   cross-backend story is "graph computes transitions; Vulkan emits explicit barriers (validated); Metal's
   tracked hazard model covers it" — do not fabricate Metal barriers that do nothing.
