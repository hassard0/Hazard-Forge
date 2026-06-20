# Slice PS6 — Deterministic Persistent Contacts: THE LIT 3D RENDER CAPSTONE (the money-shot) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SIXTH and FINAL slice of FLAGSHIP #21
> (DETERMINISTIC WARM-STARTED CONTACT CACHING + SLEEPING ISLANDS, `hf::sim::persist`) — THE LIT 3D RENDER
> CAPSTONE. PS1-PS5 built the contact key, the persistent cache, the warm-started solver, sleeping islands, and
> the lockstep+rollback netcode. PS6 is the money-shot: it takes the bit-exact warm+sleep converged world (a
> `convex::ConvexWorld`) and draws it as a LIT 3D INSTANCED scene — a warm-started tower at dead rest (asleep),
> toppled by a wake-impulse, the settled/toppled stack rendered as oriented matte boxes under directional light.
> The render is the ONE FLOAT crossing of the whole flagship (outside the bit-exact integer loop), so its bar is
> the FLOAT visresolve cross-vendor in-band metric (the documented ~20-55 float baseline), NOT strict-zero. PURE
> REUSE: the world IS a `ConvexWorld`, so PS6 delegates VERBATIM to the FROZEN CX6 render bridge
> `convex::ConvexToRenderInstances` (the FC6 precedent, fric.h:657) + the existing instanced-lit pipeline. ZERO
> new render shader, ZERO new RHI. PS1-PS5's `persist.h` code + `convex.h`/`fric.h`/`fpx.h` are BYTE-FROZEN (PS6
> is additive). Branch: `slice-ps6`. See [[hazard-forge-persist-roadmap]], [[hazard-forge-docs-style]].

**Goal:** Extend `engine/sim/persist.h` (additive — PS1-PS5 + convex.h/fric.h byte-unchanged) with the render
delegate `PersistToRenderInstances(const ConvexWorld& world)` → a one-line call to the frozen
`convex::ConvexToRenderInstances` (dynamic bodies → warm matte amber oriented boxes, statics → cool-grey floor;
render-only, OUTSIDE the integer loop; a PURE FUNCTION of the bit-exact world — two calls byte-equal). Add
`--persist-render-shot <out>` (Vulkan) / `--persist-render` (Metal) — both build the PS4 warm+sleep tower + the
PS5 command stream, run `RunPersistLockstep` to the converged authority world, and draw it LIT 3D INSTANCED.
Bake the float golden `persist_render`. **NO new shader, NO new RHI.**

## Design call: the render is the ONE float crossing; the sim stays bit-exact

The bit-exact, lockstep-replayable warm+sleep world is PS1-PS5's deterministic Q16.16 integer result. PS6 adds
ONLY a render bridge: it maps that frozen world to FLOAT model matrices for display. The sim is NOT mutated; the
provenance contract (two `PersistToRenderInstances` calls on the same world produce byte-equal matrices) is the
proof that the render is a pure function of the deterministic sim.
- **`PersistToRenderInstances(const ConvexWorld& world)` → `convex::ConvexRenderInstances`** — a one-line
  delegate to the frozen `convex::ConvexToRenderInstances(world)` (the FC6 idiom VERBATIM, fric.h:657-659).
  Provided in `persist::` for namespace symmetry + so the showcase/test call it from `persist::`; the actual
  matrices come from the byte-unchanged CX6 helper. Each body → an oriented CUBE
  (`translate(FxToFloat(pos))·rotate(normalize(orient))·scale(2·halfExtents)`); the output is SPLIT (static
  floor cool-grey vs dynamic boxes warm matte amber). **MATTE (metallic 0 / roughness 1)** to DODGE the
  documented GF6/FR6/JT6 IRIDESCENCE TRAP (a metallic/low-roughness material reads as blue iridescence — stay
  matte). The showcase passes the warm+sleep converged world (`authority.world` from `RunPersistLockstep`).
- **The scene = the PS5 lockstep scene rendered in 3D.** Build the PS4 warm+sleep tower (a static floor + 3
  dynamic slabs) + the PS5 fixed command stream (early nudges → the tower settles + SLEEPS → a wake-impulse at
  tick 160 wakes the island + topples it). Run `RunPersistLockstep(world0, cache0, sleep0, cfg, authStream,
  ticks)` to the converged AUTHORITY world; draw `PersistToRenderInstances(authority.world)` as two colored
  instanced CUBE draws (floor + dynamic boxes) under the existing directional-lit instanced pipeline. The
  money-shot: a toppled-from-rest tower, lit, in 3D — the visible payoff of the deterministic warm+sleep moat.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **PS5 `engine/sim/persist.h` (read it; APPEND only after `RunPersistRollback`, before the namespace close):**
  `RunPersistLockstep`, `PersistState`, `ConvexWorld`. PS1-PS5 byte-frozen.
- **convex.h CX6 render bridge (read-only — REUSE, do NOT redefine):** `convex::ConvexToRenderInstances`
  (convex.h:~1156+), `convex::ConvexRenderInstances` (floor[] + boxes[] of `math::Mat4`). PS6's delegate is the
  one-liner `return convex::ConvexToRenderInstances(world);` (the fric.h:657 `FrictionToRenderInstances` twin).
- **The showcase precedent:** FC6's `--fric-render-shot` (Vulkan, `samples/hello_triangle/main.cpp`) /
  `--fric-render` (Metal, `metal_headless/visual_test.mm`) — the lit 3D instanced render of a settled box world
  via the instanced-lit pipeline + the float visresolve cross-vendor bar. Mirror for `--persist-render`. The
  PS5 `--persist-lockstep-shot` scene builder + command stream are RIGHT ABOVE — reuse them verbatim to build
  the world, then render the converged authority world instead of the 2D side-view.
- **Registration:** `scripts/verify.ps1` (add `persist_render` to the Mac golden loop EXACTLY like the existing
  `fric_render`/`convex_render` entries — a plain `@{ Name = 'persist_render'; Flag = '--persist-render' }`, NO
  special threshold; the committed golden is the Mac bake, verify re-renders on the Mac + compares vs it at
  0.0000 same-backend — + add `--persist-render-shot` to `$vkShots`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden — do NOT**), append to
  `tests/persist_test.cpp`.

## Design decisions (locked)
1. **APPEND to `engine/sim/persist.h`** (PS1-PS5 byte-frozen): exactly `PersistToRenderInstances` (the one-line
   CX6 delegate). **NO new shader, NO new RHI** (the seam incl. shaders — EMPTY; render reuses the instanced-lit
   pipeline). Render-only float, OUTSIDE the bit-exact loop — the integer sim is NOT touched.
2. **Showcase `--persist-render-shot <out>` (Vulkan) AND `--persist-render` (Metal) — WIRE BOTH** (standalone
   arg-parse). BOTH build the PS4 warm+sleep tower + the PS5 command stream, run `RunPersistLockstep` to the
   converged authority world, and draw `PersistToRenderInstances(authority.world)` LIT 3D INSTANCED (floor +
   dynamic boxes, matte). Golden = `tests/golden/metal/persist_render.png` (Mac-baked by the CONTROLLER — DO NOT
   commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) provenance:** two `PersistToRenderInstances(authority.world)` calls produce BYTE-EQUAL matrix sets
     (the render is a pure function of the bit-exact sim). Print `persist-render: {floor:<F>, boxes:<B>}
     provenance two-calls BYTE-EQUAL`; assert.
   - **(2) non-trivial scene:** the rendered world is the woken/toppled tower (assert the instance counts match
     the scene: floor:1 + boxes:3, and the converged world is non-trivial — e.g. at least one dynamic box's
     orientation is visibly tilted / the tower toppled). Print `persist-render: {dynamic:<B>, toppled:true}`.
   - **(3) determinism:** the Vulkan render path runs cleanly + writes the image (exit 0); the Metal side bakes
     two runs DIFF 0.0000 (the render is deterministic per-backend — same world, same pipeline).
   - **Golden discipline: ONLY `tests/golden/metal/persist_render.png`; do NOT commit it.** Existing 202 image
     goldens UNTOUCHED.
4. **Cross-backend bar:** the COMMITTED golden is the Mac-Metal bake; verify.ps1 re-renders on the Mac +
   compares vs it at 0.0000 (same-backend determinism — the Metal two-run DIFF 0.0000 IS the gate, exactly like
   every other render golden). SEPARATELY, the CONTROLLER measures the Windows-Vulkan vs Mac-Metal cross-vendor
   visresolve mean as a DIAGNOSTIC — a FLOAT render is in-band (~20-55, the FR6/JT6/CX6 lineage), NOT strict-zero
   cross-vendor (that's expected for float; the integer determinism headline is PS1-PS5's, not PS6's).
5. **Tests — APPEND to `tests/persist_test.cpp` (pure CPU):** `PersistToRenderInstances` over the converged
   warm+sleep world → the provenance contract (two calls byte-equal — compare the `math::Mat4` arrays
   element-by-element); the instance split is correct (floor count = the static count, boxes count = the
   dynamic count); a render of the asleep settled world vs the woken toppled world differ (the render reflects
   the sim state). Clean under `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-persist-render` (features) + `--persist-render-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None — and NO new shader.** Render reuses the existing instanced-lit pipeline. `rhi.h` + backend dirs
  UNCHANGED. `engine/sim/convex.h` + `fric.h` + `fpx.h` + **PS1-PS5's persist.h code + ALL persist shaders
  (persist_key/cache/warm/sleep.comp)** + all other sim headers + `engine/nav/` + `engine/anim/` +
  `engine/physics/` + ALL existing shaders UNCHANGED. `persist.h` APPEND-only (only the
  `PersistToRenderInstances` delegate). Report the seam empty (only the persist.h APPEND + the
  showcase/test/introspect are new/changed; NO shaders/ change at all).

## Out of scope (YAGNI)
Real-time interactive replay UI. Soft shadows / SSAO / post on the capstone (the existing instanced-lit pipeline
as-is). PS6 claims ONLY: the bit-exact warm+sleep converged world renders as a coherent LIT 3D scene (the
toppled-from-rest tower), the render is a PURE FUNCTION of the deterministic sim (provenance byte-equal), the
Metal bake is per-backend deterministic (two runs 0.0000) + cross-vendor in-band (float visresolve). NOTE: boxes
only; the same within-band warm/sleep caveats as PS3/PS4. This is the FINAL slice → flagship #21 COMPLETE.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 107 incl. PS1-PS5's `persist_test` + the appended PS6
   cases). Clean under `windows-msvc-asan` (build+run `persist_test` + `introspect_test`).
2. **proofs + visual:** `--persist-render-shot` on Vulkan: the proofs + exit 0, under the Vulkan-validation gate
   → ZERO VUID. **VERIFY the image shows a coherent LIT 3D toppled-from-rest tower (oriented matte boxes, a
   floor, directional light) — NO iridescence (matte), NO garbage/NaN.**
3. Metal: `visual_test --persist-render` → new golden `tests/golden/metal/persist_render.png`; two runs DIFF
   0.0000. **Confirm `visual_test.mm` in the diff; confirm NO shader added.** Cross-vendor = FLOAT visresolve
   in-band (~20-55, NOT strict-zero).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `persist_render.png` added; the
   other 202 byte-identical. `git diff master --stat -- tests/golden` = ONLY `persist_render.png` (metal) + the
   introspect json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-persist-render` + `--persist-render-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/convex.h`/`fric.h`/`fpx.h` + **PS1-PS5's persist.h code + ALL
   persist shaders** + ALL other sim headers + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing
   shaders byte-unchanged; **NO shaders/ change at all**). `scripts/verify.ps1` updated: `persist_render` in the
   Mac loop (as a float/visresolve golden) + `--persist-render-shot` in `$vkShots`.
