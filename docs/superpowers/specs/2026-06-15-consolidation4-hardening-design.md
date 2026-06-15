# Slice AT — Consolidation #4 / Validation Hardening — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Consolidation pass (DUE: 4 feature slices AP–AS since AO). Cross-platform re-verify + fix the latent
> validation bugs the Slice-AS sync-validation oracle exposed. Verifiable on Windows+Vulkan (validation
> oracle) AND Apple M4+Metal (golden re-verify).

**Goal:** Two jobs. (1) **Re-verify** the entire A–AS engine cross-platform (Windows ctest debug+asan,
all 25 Metal image goldens DIFF 0.0000, introspect JSON golden byte-exact). (2) **Harden:** Slice AS
activated the Vulkan validation layer and found the engine is sync-hazard-clean but NOT fully
validation-clean — two pre-existing CORE-validation bugs remain. Fix them so the engine is **fully
Vulkan-validation-CLEAN** (zero errors of ANY class across all 25 showcases), and wire validation as a
permanent gate so it can't silently regress. This is render-invariant: fixing descriptor/semaphore
correctness changes NO pixels — all goldens stay byte-identical.

## The two latent bugs to fix (found by the now-active oracle, confirmed pre-existing on master)

1. **GPU-particle push-descriptor `UPDATE_AFTER_BIND` invalidation** (fires on `--shot`, `--camera-shot`).
   The GPU-particle pass updates a descriptor after it has been bound in a way the validation layer flags
   as invalidated. Inspect the particle pass's descriptor binding/update order. Fix options (pick the
   minimal correct one, document it): update the descriptor BEFORE binding; use a per-frame descriptor set
   with correct lifetime; or stop relying on UPDATE_AFTER_BIND semantics where the binding model doesn't
   guarantee it. The fix must NOT change the rendered particles (render-invariant).

2. **Multi-frame swapchain semaphore-reuse CORE-validation error.** An acquire/present semaphore is reused
   across frames-in-flight before its prior signal has been waited. Fix: size the acquire/present
   semaphores per-frame-in-flight (or per-swapchain-image as appropriate) so no semaphore is re-submitted
   while still pending. Standard Vulkan frames-in-flight semaphore hygiene. Must stay render-invariant and
   not deadlock/stall headless capture.

Both are Windows/Vulkan-only concerns (Metal path unaffected). After the fixes, EVERY `--*-shot` showcase
run under full validation (sync + core) must report ZERO validation errors.

## Design decisions (locked)

1. **Render-invariance is mandatory.** These are correctness fixes, not behavior changes. After fixing,
   re-run the Windows headless captures and confirm the images are unchanged; the 25 Metal goldens stay
   byte-identical (`git diff master --stat -- tests/golden/metal` EMPTY). If any golden would change, the
   fix is wrong — STOP.

2. **Validation as a permanent gate.** Add a `--validate` / env path (or reuse what AS wired) so the
   engine can run with `VK_LAYER_KHRONOS_validation` + sync-validation + core validation enabled, failing
   loudly on any error. Document the conan `vulkan-validationlayers` dependency + `VK_LAYER_PATH` setup AS
   discovered (the layer was NOT installed on this box → the oracle was silently inactive before AS). Wire
   a validation smoke into `scripts/verify.ps1` (run a representative showcase, e.g. `--shot` + `--csm-shot`,
   under validation; grep the output for any `VUID`/`SYNC-HAZARD`/`UNASSIGNED` error and fail if found).
   Keep verify.ps1 pure-ASCII.

3. **No new goldens, no new features.** This is hardening — no `--*-shot` added, no introspect features
   added (introspect JSON byte-unchanged). The deliverable is "validation-clean + everything re-verified",
   not a new picture. The proof is the validation oracle + unchanged goldens + green ctest.

4. **Full cross-platform re-verification matrix** (the consolidation half), all must pass:
   - `ctest --preset windows-msvc-debug` 26/26.
   - `ctest` under `windows-msvc-asan` 26/26 (zero memory errors).
   - All 25 Metal image goldens: two M4 runs each (or the existing per-golden visual_test) → DIFF 0.0000.
   - Introspect JSON golden byte-exact vs live `--introspect`.
   - Seam grep: no new real backend code symbols above the seam (audit any tick-up = comments only).
   - All `--*-shot` showcases under full Vulkan validation → ZERO errors.

## Out of scope (YAGNI)
Any new rendering feature, multithreaded recording (that's the NEXT slice), Metal validation tooling
(Metal's API-validation layer is a separate concern; this slice is Vulkan validation-clean + golden
re-verify), performance profiling/optimization beyond the two correctness fixes.

## Verification gate
1. The two CORE-validation bugs FIXED: `--shot` and `--camera-shot` (and all other showcases) run under
   `VK_LAYER_KHRONOS_validation` (sync + core) with ZERO validation errors. Capture/report the clean
   validation output. Confirm the specific prior errors (GPU-particle UPDATE_AFTER_BIND, swapchain
   semaphore-reuse) are gone.
2. **Render-invariance:** `git diff master --stat -- tests/golden/metal` EMPTY; Windows headless captures
   visually unchanged; introspect JSON byte-unchanged.
3. `ctest` debug 26/26 + asan 26/26.
4. All 25 Metal goldens DIFF 0.0000 on the M4 (re-verify, no rebake).
5. `scripts/verify.ps1` gains a Vulkan-validation smoke gate (pure-ASCII); document the validation-layer
   conan dependency + VK_LAYER_PATH so the oracle stays active.
6. Seam grep clean (no new code symbols). Commit message documents: the two fixes + the full re-verify
   matrix result. If EITHER bug cannot be fixed render-invariantly, STOP and report (do not paper over a
   validation error by disabling the layer or rebaking a golden).
