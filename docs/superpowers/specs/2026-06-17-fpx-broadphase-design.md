# Slice FPX2 — Deterministic Fixed-Point Physics: INTEGER BROADPHASE PAIR GENERATION (Phase 11 #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The 2nd FPX slice (after FPX1's
> integrator): generate the DETERMINISTIC candidate collision-pair list — for each pair of bodies whose integer AABBs
> overlap, emit `(i, j)` with `i < j`, as a sorted compacted list. This is the prerequisite for FPX3's collision
> response (narrowphase + impulses run per candidate pair). Proven GPU==CPU BIT-EXACT (the pair list AND its count),
> integer, cross-backend BIT-IDENTICAL. NO new RHI. Kept INT32 (AABB overlap = integer compares, no int64) so it
> MSL-generates natively on Metal (unlike FPX1's int64 fxmul, which is Vulkan-only). Namespace `hf::sim::fpx`. Branch:
> `slice-fpx-broadphase`. See [[hazard-forge-fpx-roadmap]].

**Goal:** Extend `engine/sim/fpx.h` with the integer AABB + overlap test + the deterministic pair builder
(`FxAabb`, `BodyAabb`, `AabbOverlap`, `CountPairs`, `BuildPairs` CPU reference) + `shaders/fpx_pair_count.comp.hlsl`
(per-body overlap count) + `shaders/fpx_pair_emit.comp.hlsl` (per-body emit at the prefix-sum offset) + a
`--fpx-pairs-shot` (Vulkan) / `--fpx-pairs` (Metal) showcase that builds the pair list for a clustered scene, reads it
back, proves it BIT-EXACT vs the CPU reference, and bakes an integer pair-matrix golden. Make-safe: header additions +
2 NEW shaders + NEW showcase + NEW golden; FPX1 + the existing float `engine/physics/` UNCHANGED.

## The integer core (extends fpx.h)
- **`struct FxAabb { FxVec3 lo, hi; };`** — an integer (Q16.16) axis-aligned bounding box. A body's AABB =
  `BodyAabb(body, radius)` = `{ pos - radius, pos + radius }` (radius a per-body or global `fx`; FPX1's `FxBody` gets
  an `fx radius` field OR a global `kRadius` — add a `radius` to `FxBody` as a defaulted field so FPX1's showcase/test
  bodies are unchanged in behavior; document). Pure integer add/sub.
- **`bool AabbOverlap(const FxAabb& a, const FxAabb& b)`** = `a.lo.x <= b.hi.x && b.lo.x <= a.hi.x && (y) && (z)` —
  six integer compares, NO int64, NO products → bit-identical cross-vendor by construction (the strongest form; even
  simpler than FPX1's int64 fxmul). This is the deterministic broadphase predicate.
- **`uint32_t CountPairs(const FxWorld&, fx radius, std::span<uint32_t> perBodyCountOut)`** — the CPU reference count:
  `perBodyCountOut[i]` = number of `j > i` with `AabbOverlap(BodyAabb(i), BodyAabb(j))`; returns the total. (Ordered
  `j > i` so each unordered pair is counted ONCE, deterministically, with a canonical `i < j` orientation — the
  gpu_culled.h source-order discipline.)
- **`struct FxPair { uint32_t i, j; };`** + **`void BuildPairs(const FxWorld&, fx radius, std::vector<uint32_t>&
  perBodyOffset, std::vector<FxPair>& pairsOut)`** — the full CPU mesher: `CountPairs` → exclusive prefix-sum →
  `perBodyOffset` → for each `i`, emit each overlapping `(i, j>i)` at `pairsOut[perBodyOffset[i] + local++]`. The pair
  list is grouped by `i` (ascending) then `j` (ascending) → fully deterministic, the exact list the GPU memcmp's
  against.

## Reuse map (file:line)
- **FPX1 (the inputs):** `engine/sim/fpx.h` — `FxVec3`, `FxBody`, `FxWorld`, `FloorDiv` (the int32 fixed-point types).
- **The count→prefix-sum→emit compaction pattern (copy MC3 / VT2 verbatim):** `shaders/mc_count.comp.hlsl` (per-cell
  count → the per-body count analog), `shaders/mc_scan.comp.hlsl` (single-thread `[numthreads(1,1,1)]` exclusive
  prefix-sum → reuse the SHAPE for `fpx_pair`), `shaders/mc_emit.comp.hlsl` (per-cell emit at the offset → the
  per-body pair emit analog); the CPU mirror `engine/render/gpu_culled.h:78,107` (`CullAndCompact`, source-order
  deterministic). The fixed-capacity-output + ReadBuffer-count pattern (`gpudriven_cull.comp` + `gpu_culled.h:64-70`).
- **The compute + readback surface (NO new RHI):** `BufferUsage::Storage` (`rhi.h:166`), compute (`rhi.h:412-426`),
  compute→compute barrier (`rhi.h:434`, between count→scan→emit), `ReadBuffer` (`rhi.h:616`).
- **The integer-set debug-viz golden:** the `vt_alloc`/`mc_count` CPU-colored-from-integer template; `meshlet.h:79`
  `hashColor`.
- **Showcase + registration:** the FPX1 `--fpx-shot`/`--fpx` showcase + introspect + verify.ps1 shapes.

## Design decisions (locked)

1. **Three compute passes (the MC3 pattern), all bit-exact, ALL INT32 (Metal-native — no int64).** (a)
   `fpx_pair_count.comp.hlsl` — one thread per body `i`, scans `j > i`, `perBodyCount[i] = Σ AabbOverlap(...)`. (b)
   `fpx_pair_scan.comp.hlsl` — `[numthreads(1,1,1)]` SINGLE-THREAD exclusive prefix-sum of `perBodyCount` →
   `perBodyOffset` (the mc_scan / VT2 single-thread-allocator pattern; or REUSE mc_scan.comp if its binding layout
   fits — prefer a dedicated fpx_pair_scan for clarity). (c) `fpx_pair_emit.comp.hlsl` — one thread per body `i`,
   scans `j > i`, emits each overlapping `(i,j)` into `gPairs` at `perBodyOffset[i] + local++` (disjoint per-body
   ranges → race-free, NO atomics). `gBodies`(int32 FxBody) / `perBodyCount` / `perBodyOffset` / `gPairs` (FxPair) /
   `gParams{bodyCount, radius, enabled}`. AabbOverlap copied VERBATIM from fpx.h. `enabled=0` → count 0 / emit
   nothing. Register all in BOTH compile lists — **int32 only → no `--msl-version 20200`, MSL-gens on Metal natively
   (the broadphase is NOT Vulkan-only, unlike fpx_integrate.comp).**
2. **Output capacity.** The host runs `CountPairs` first (or computes the total via the CPU reference — simplest +
   deterministic), preallocates `gPairs` to exactly `totalPairs`, then dispatches count→scan→emit. (A worst-case
   `N*(N-1)/2` preallocation is also fine for the showcase's small N.)
3. **Showcase `--fpx-pairs-shot <out>` (Vulkan, main.cpp) AND `--fpx-pairs` (Metal, visual_test.mm — WIRE BOTH;
   confirm visual_test.mm + `#include "sim/fpx.h"`).** A deterministic CLUSTERED `FxWorld` (so the pair set is
   non-trivial): e.g. a tight grid / a settled pile from `IntegrateStep` where adjacent bodies' AABBs overlap (radius
   chosen so a known subset of neighbors overlap, NOT all-pairs and NOT none). Build the pair list (count→scan→emit),
   `ReadBuffer` `gPairs` + the count. CPU-run `BuildPairs`. Golden = an integer PAIR-MATRIX viz: an `N×N` grid (scaled
   to a reasonable image, e.g. 64×64 bodies → an 8px/cell 512×512), pixel `(i,j)` lit `hashColor(i*N+j)` (or a flat
   color) iff `(i,j)` (or `(j,i)`) is a candidate pair, the diagonal a marker, else dark → `tests/golden/metal/
   fpx_pairs.png` (the symmetric collision-candidate adjacency; CPU-colored from the integer pair read-back →
   identical both backends by construction).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU pair list bit-exact (make-or-break):** `perBodyCount`, `perBodyOffset`, AND the `gPairs` list
     (`totalPairs` entries) all equal the CPU `BuildPairs` reference byte-for-byte (`ReadBuffer` memcmp). Print `fpx-pairs
     GPU==CPU: <P> pairs BIT-EXACT (count+scan+emit)`.
   - **(2) pair-set correctness:** every emitted `(i,j)` has `i < j` and `AabbOverlap(BodyAabb(i), BodyAabb(j))` true;
     the list is grouped-by-i ascending; no duplicates; `totalPairs == Σ perBodyCount`. Print `fpx-pairs: {bodies:<N>,
     pairs:<P>, max-pairs/body:<m>}`.
   - **(3) disabled-path no-op:** `enabled=false` → count all-zero, `gPairs` untouched (cleared). Print `fpx-pairs
     disabled: zero pairs (no-op)`.
   - **(4) determinism:** two full (count+scan+emit) runs → byte-identical pair list. Print `fpx-pairs determinism:
     two runs BYTE-IDENTICAL`.
   - **(5) known case:** a hand-checked tiny sub-config (e.g. two bodies overlapping → exactly 1 pair `(a,b)`; two
     bodies apart → 0 pairs). Print `fpx-pairs known: overlap→1 apart→0 OK`.
   - **Golden discipline: ONLY `tests/golden/metal/fpx_pairs.png`; do NOT commit it — the CONTROLLER bakes on the
     Mac.** Existing 106 image goldens UNTOUCHED.
5. **Determinism / cross-backend.** AabbOverlap is six int32 compares (no int64, no products); the count/emit are
   per-body independent over `j>i` in a fixed order; the prefix-sum is a single-thread serial scan; the golden is
   CPU-colored from the integer pair read-back → bit-identical Vulkan/Metal AND the GPU==CPU memcmp holds. Run under
   the Vulkan sync-validation gate → the count→scan→emit barriers SYNC-HAZARD-free.
6. **Tests `tests/fpx_test.cpp` additions (pure CPU):** `AabbOverlap` (overlapping / touching / apart, incl negative
   coords); `BodyAabb` (pos±radius); `CountPairs`/`BuildPairs` on a known scene → known count + known pair list
   (grouped-by-i, `i<j`, no dups); prefix-sum offsets correct; `enabled`-off → empty; determinism; the
   `totalPairs==Σ perBodyCount` invariant. Clean under `windows-msvc-asan`.
7. **Introspect.** Add exactly `deterministic-fixedpoint-physics-broadphase` (features) + `--fpx-pairs-shot`
   (showcases).

## RHI seam additions (summary)
- **None.** `BufferUsage::Storage` + compute (count + single-thread scan + emit) + compute→compute barrier +
  `ReadBuffer` — the MC3/VT2 precedent. ZERO above-seam backend symbols. `rhi.h` + `rhi_factory` (baseline 2) +
  backend dirs UNCHANGED. `engine/physics/` UNTOUCHED. Report the seam.

## Out of scope (YAGNI — FPX3+)
Narrowphase + collision RESPONSE (FPX3 — this slice only generates candidate PAIRS, no contact resolution), a true
spatial-hash grid (the `j>i` AABB scan is O(N²) but deterministic + bit-exact, fine at this N; a grid is a perf
optimization, not correctness — defer), orientation/box colliders (FPX4), the lockstep proof (FPX5). ONE deterministic
integer-AABB broadphase producing the bit-exact candidate-pair list with the GPU==CPU proof + pair-set correctness +
disabled no-op + determinism + a known case and the integer pair-matrix golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 92) + the new `fpx_test` broadphase cases. Clean under
   `windows-msvc-asan`.
2. **proofs + visual:** `--fpx-pairs-shot` on Vulkan: a coherent symmetric pair-matrix viz (the clustered scene's
   collision candidates); all 5 proof lines. Run under the Vulkan-validation gate → ZERO VUID in the OUTPUT.
3. Metal: `visual_test --fpx-pairs` → new golden `tests/golden/metal/fpx_pairs.png`; two runs DIFF 0.0000 (gate on
   compare.sh EXIT CODE). The GPU==CPU + determinism proofs also pass on Metal (int32 — MSL-gens natively, NOT
   Vulkan-only). **Confirm visual_test.mm in the diff; confirm fpx_pair_*.comp MSL-generate (int32, no MSL-2.2).**
   Integer golden → a strict cross-backend pixel compare must show ZERO differing pixels.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `fpx_pairs.png` added; the other 106
   byte-identical (FPX1 `fpx.png` untouched). `git diff master --stat -- tests/golden` = ONLY `fpx_pairs.png` (metal)
   + the introspect json.
5. Introspect JSON rebaked exactly `+deterministic-fixedpoint-physics-broadphase` + `--fpx-pairs-shot`; introspect
   test updated.
6. Seam grep clean (`rhi.h` UNCHANGED — no new RHI). `scripts/verify.ps1` updated: `fpx_pairs` golden in the Mac loop
   + `--fpx-pairs-shot` in `$vkShots`.
