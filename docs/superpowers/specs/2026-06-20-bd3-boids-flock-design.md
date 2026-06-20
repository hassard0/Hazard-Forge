# Slice BD3 — Deterministic GPU Crowds: THE FULL FLOCK STEP (SEPARATION + ALIGNMENT + COHESION) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The THIRD slice of FLAGSHIP #18 (DETERMINISTIC GPU
> CROWDS, `hf::sim::boids`). BD1 built the steering primitive (brute-force seek + separation); BD2 built the
> grid-hash neighbor list. BD3 is the FULL FLOCK: over the BD2 neighbor list, accumulate the THREE Reynolds rules
> — SEPARATION (steer away from crowding) + ALIGNMENT (steer toward the neighbors' mean heading) + COHESION (steer
> toward the neighbors' mean position) — and integrate one deterministic tick. A real flock emerges. INTEGER-bit-
> exact. The flock-step shader is int64-Vulkan-only (the integrate `fxmul`, the BD1/sim-integrator lesson) over
> the MSL-native BD2 neighbor list. **NO new RHI.** Branch: `slice-bd3`. See [[hazard-forge-boids-roadmap]].

**Goal:** Extend `engine/sim/boids.h` (additive — BD1/BD2 byte-frozen) with `FlockConfig` (BD1 BoidsConfig +
`alignGain`/`cohGain`) + `AccumFlock` (the 3 Reynolds rules over an agent's BD2 neighbor list) + `StepFlock`
(rebuild the BD2 grid+neighbors on the current positions → per-agent AccumFlock + integrate) + `StepFlockSteps` +
a flock measure. Add the new shader `shaders/boids_flock.comp.hlsl` (int64, Vulkan-only) + `--boids-flock-shot`
(Vulkan) / `--boids-flock` (Metal). Bake the integer golden `boids_flock`. **NO new RHI.**

## Design call: the 3 Reynolds rules over the BD2 neighbor list, integrated per tick (grid rebuilt each tick)
Each tick `StepFlock`:
1. **Rebuild the BD2 neighbor engine on the CURRENT positions** — `MakeBoidsGrid(agents, perceptionRadius)` →
   `BuildBoidsCellTable` → `BuildBoidsNeighborList` (the BD2 functions, REUSED). The flock moves, so the grid is
   rebuilt every tick (the grain/fluid per-tick-rebuild precedent). **The grid is FIXED-SIZE for the showcase**:
   the scene is a bounded flock (cohesion keeps it together) in a known region, so the showcase sizes the grid
   ONCE generously to contain the flock for all K ticks (constant `gridDim` → constant GPU buffer sizes → the
   per-tick rebuild just re-buckets). Document this; a fully-dynamic re-AABB grid is a BD-future.
2. **`AccumFlock(agent i, neighbors, agents, cfg)`** — over agent i's BD2 neighbor list (the indices in
   `neighborStart[i]..neighborStart[i+1]`):
   - **Separation** `sep` = Σ `FxSub(pos_i, pos_j)` (away from each neighbor — the BD1 rule, now over the grid
     list). Scaled by `sepGain`.
   - **Alignment** `align` = (Σ `vel_j`) / count − `vel_i` (steer toward the neighbors' mean velocity). The mean is
     an INTEGER divide by the neighbor count (deterministic). Scaled by `alignGain`.
   - **Cohesion** `coh` = (Σ `pos_j`) / count − `pos_i` (steer toward the neighbors' mean position). Integer divide.
     Scaled by `cohGain`.
   - `force = FxAdd(FxAdd(FxScale(sep, sepGain), FxScale(align, alignGain)), FxScale(coh, cohGain))`. (Zero
     neighbors → only the seek/gravity terms, no div-by-zero — guard count==0.)
3. **Integrate** (the BD1 `StepBoids` body REUSED): add the optional seek-to-target + gravity, per-axis clamp
   `force` to `maxForce`, `vel += force*dt` clamp to `maxSpeed`, `pos += vel*dt`. **JACOBI** (frozen-snapshot
   input — every agent reads the previous step's positions/velocities; the BD1 ping-pong discipline) → GPU
   race-free + order-independent + bit-exact GPU==CPU.

The sums (Σ pos_j, Σ vel_j) are integer ADDS; the means are integer DIVIDES by the count; the integrate uses
`fxmul` (int64). So **`boids_flock.comp` is int64-Vulkan-only** (NOT in hf_gen_msl; Metal runs the CPU `StepFlock`)
— BUT it consumes the MSL-native BD2 neighbor list (the BD2 grid passes stay a true GPU pass both backends; only
the flock-integrate pass is the int64 split). Document this honestly.

`FlockConfig` = BD1 `BoidsConfig` + `fx alignGain, cohGain, perceptionRadius`. `MeasureFlock(agents, cfg)` → the
mean speed, the flock's bounding-box diagonal (cohesion pulls it IN), the mean heading-alignment (how aligned the
velocities are — the "they flock" stat) — deterministic integer/L1 metrics.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **BD1/BD2 (this branch's `boids.h`, read-only — build on, DON'T modify):** `Agent`, `BoidsConfig`, `ClampAxisVec`,
  `SteerSeek`, `StepBoids`'s integrate body (the clamp+integrate to REUSE), `BoidsGrid`/`MakeBoidsGrid`/
  `BuildBoidsCellTable`/`BuildBoidsNeighborList`/`BoidsNeighborList` (the BD2 engine — REUSED each tick). DO NOT
  modify the BD1/BD2 functions. BD3 APPENDS.
- **fpx integer ops (`engine/sim/fpx.h`, read-only):** `FxVec3`/`FxAdd`/`FxSub`/`FxScale`/`fxmul`, the integer
  divide for the means. **DO NOT modify fpx.h.**
- **The int64-Vulkan-only flock shader + the per-tick-grid-rebuild GPU driver:** study `--grain-*-shot` (how a sim
  rebuilds the grid each tick on the GPU + drives the int64 integrate shader after the MSL-native grid passes +
  memcmps) and BD1's `--boids-steer-shot` (the int64 integrate + ping-pong). `boids_flock.comp` is the BD2
  grid-passes (MSL-native, REUSED per tick) + the new int64 flock-integrate pass. The GPU showcase per tick:
  clear+run `boids_cell_*` + `boids_neighbor_*` (rebuild the list) → `boids_flock.comp` (step) → next tick.
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), `tests/boids_test.cpp`.

## Design decisions (locked)
1. **boids.h additions:** `FlockConfig`, `AccumFlock` (the 3 Reynolds rules over the BD2 neighbor list, count==0
   guard), `StepFlock` (rebuild grid+neighbors → JACOBI per-agent AccumFlock + integrate), `StepFlockSteps`,
   `MeasureFlock`. Pure integer, fixed op order, JACOBI frozen-snapshot. **NEW shader** `boids_flock.comp.hlsl`
   (int64, Vulkan-only, one thread per agent: read the BD2 neighbor list + the frozen agent buffer, the 3 rules +
   integrate, write the output buffer — ping-pong). NOT in hf_gen_msl; Metal runs CPU `StepFlock`.
2. **Showcase `--boids-flock-shot <out>` (Vulkan) AND `--boids-flock` (Metal) — WIRE BOTH** (standalone arg-parse).
   The SCENE: a flock (~256 agents) started spread/random-but-deterministic in a bounded region with small initial
   velocities; settle K `StepFlock` steps → a COHERENT FLOCK emerges (the agents cluster [cohesion], align their
   headings [alignment], and keep spacing [separation] — a recognizable swarm, NOT collapsed to a point, NOT
   scattered/exploded). Vulkan: per tick the GPU `boids_cell_*`/`boids_neighbor_*` (MSL-native) rebuild + the int64
   `boids_flock.comp` step → **memcmp the GPU agent array vs the CPU `StepFlock`** (NO tolerance). Metal: the CPU
   reference. Render a PURE-INTEGER 2D top-down view (each agent a dot at `pos>>kFrac`, a short velocity tick
   showing the aligned heading). Golden = `tests/golden/metal/boids_flock.png` (Mac-baked by the CONTROLLER — DO
   NOT commit).
3. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU agent array after K steps == the CPU `StepFlock` byte-for-byte. Print
     `boids-flock: {agents:<N>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `boids-flock determinism: two runs BYTE-IDENTICAL`.
   - **(3) the flock emerged:** after settling, the flock's bounding diagonal SHRANK from the start (cohesion
     pulled it together) AND the mean heading-alignment ROSE (alignment aligned the velocities) AND the min
     separation stayed above a floor (separation kept spacing). Print `boids-flock emerged: {diagShrank:true,
     aligned:<A>, minSep:<S>, flocked:true}`; assert the three.
   - **(4) the rules matter:** an `alignGain=cohGain=0` control (separation only) does NOT cohere/align (the
     flock stays loose/unaligned) — the three rules together make the flock. Print `boids-flock control:
     {sepOnly:notFlocked}`.
   - **Golden discipline: ONLY `tests/golden/metal/boids_flock.png`; do NOT commit it. BD1 `boids_steer.png` + BD2
     `boids_neighbors.png` byte-identical.** Existing 179 image goldens UNTOUCHED.
4. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal CPU-ref == golden, ZERO differing pixels.
5. **Tests `tests/boids_test.cpp` additions (pure CPU):** `AccumFlock` — separation points away from a near
   neighbor, alignment toward the neighbors' mean velocity, cohesion toward their mean position, zero-neighbors no
   crash; `StepFlock` — a spread flock coheres (diagonal shrinks) + aligns (heading-alignment rises) while keeping
   min-separation above a floor; a sep-only control stays loose; two runs byte-identical. Clean under
   `windows-msvc-asan`.
6. **Introspect.** Add exactly `deterministic-boids-flock` (features) + `--boids-flock-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the BD2 compute + SSBO + dispatch + read-back path. `rhi.h` + backend dirs UNCHANGED. `engine/
  sim/fpx.h` + `grain.h` + `fluid.h` + `cloth.h` + `joint.h` + `couple*.h` + `fract.h` + `vehicle.h` + `active.h` +
  `engine/nav/` + `engine/anim/` + `engine/physics/` + all EXISTING shaders (incl `boids_steer.comp` +
  `boids_cell_*`/`boids_neighbor_*`) UNCHANGED. The ONLY new shader is `boids_flock.comp.hlsl` (int64, Vulkan-only).
  BD1/BD2 `boids.h` code UNCHANGED (BD3 additive). Report the seam empty (only `boids.h` extended + the 1 new
  shader + showcase/test/introspect).

## Out of scope (YAGNI — later BD slices)
Path-following the A* corridor (BD4 — BD3 is free flocking, no goal beyond the optional seek target), lockstep (BD5),
the lit 3D render (BD6 — BD3's render is the 2D top-down diagnostic). A fully-dynamic re-AABB grid (BD3 uses a fixed
generously-sized grid; cohesion keeps the flock inside it), RVO/ORCA avoidance, 3D volumetric flocking. BD3 claims
ONLY: a deterministic 3-rule Reynolds flock (separation + alignment + cohesion over the grid neighbor list) that
coheres into a recognizable swarm, bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs.
NOTE (honest): boids are POINTS with steering FORCES (soft separation, not hard contact); the means are integer
divides (a deterministic proxy for the centroid, no float); the integrate is int64-Vulkan-only (the BD1/sim-
integrator lesson) over the MSL-native BD2 list; the showcase grid is fixed-size (cohesion-bounded).

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 104 + the new `boids_test` flock cases). Clean under
   `windows-msvc-asan` (build+run `boids_test` + `introspect_test`).
2. **proofs + visual:** `--boids-flock-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   a coherent flock — a clustered, heading-aligned swarm (not collapsed to a point, not scattered/exploded).**
   Re-run `--boids-steer-shot` + `--boids-neighbors-shot` → still bit-exact (BD1/BD2 render-invariance).
3. Metal: `visual_test --boids-flock` → new golden `tests/golden/metal/boids_flock.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; confirm `boids_flock.comp` NOT in `hf_gen_msl` (int64, Metal runs the
   CPU StepFlock) while the BD2 grid passes STAY in hf_gen_msl.** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `boids_flock.png` added; the other
   179 byte-identical. `git diff master --stat -- tests/golden` = ONLY `boids_flock.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-boids-flock` + `--boids-flock-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` + the frozen sim/nav/anim/physics headers + ALL existing shaders incl `boids_steer`/
   `boids_cell_*`/`boids_neighbor_*` byte-unchanged; ONLY `boids.h` extended additively + the 1 new shader +
   showcase/test/introspect). `scripts/verify.ps1` updated: `boids_flock` golden + `--boids-flock-shot` in
   `$vkShots`. **The ONLY new shader is `boids_flock.comp.hlsl` (int64, NOT in hf_gen_msl).**
