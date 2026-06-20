# Slice BD1 — Deterministic GPU Crowds: THE STEERING PRIMITIVE — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice (the BEACHHEAD) of FLAGSHIP #18
> (DETERMINISTIC GPU CROWDS — BOIDS + STEERING + PATH-FOLLOWING, `hf::sim::boids`). The headline: 10,000 agents
> flocking + path-following in LOCKSTEP, bit-identical Vulkan/Metal, rollback-replayable (UE5's Mass/AI is float).
> BD1 adds the one genuinely-new primitive a crowd needs: a deterministic integer STEERING force — `SteerSeek`
> (toward a target) + `SteerSeparation` (away from too-close neighbors) accumulated into a per-agent velocity, the
> Reynolds steering model in Q16.16. INTEGER-bit-exact. New shader `boids_steer.comp` (the implementer determines
> int32-MSL-native vs int64-Vulkan-only and DOCUMENTS which — see §"the int32 goal"). Branch: `slice-bd1`. See
> [[hazard-forge-boids-roadmap]].

**Goal:** Create `engine/sim/boids.h` (header-only, namespace `hf::sim::boids`, `#include "sim/fpx.h"` read-only
ONLY) with `Agent` + `BoidsConfig` + `SteerSeek` + `SteerSeparation` (brute-force all-pairs in BD1 — BD2 adds the
grid) + `StepBoids` + `StepBoidsSteps` + a measure. Add the new shader `shaders/boids_steer.comp.hlsl` +
`--boids-steer-shot` (Vulkan) / `--boids-steer` (Metal). Bake the integer golden `boids_steer`. **NO new RHI.**

## Design call: integer Reynolds steering — seek + separation accumulated into velocity
A boid is a point with a position + velocity (the `fpx::FxBody` cousin, no orientation/mass). Each tick it
computes a STEERING force (the Reynolds model) and integrates:
- **`SteerSeek(agent, target, cfg)`** — pull toward `target`: `desired = target − agent.pos`; the agent accelerates
  toward `desired`. To stay MSL-native int32 where possible, BD1 uses a **proportional (un-normalized) seek**: the
  force is `FxScale(desired, seekGain)` (a Q16.16 gain scaling the raw offset), NOT a `FxNormalize(desired) *
  maxSpeed` (which would need the int64 `FxISqrt`). The seek is then magnitude-clamped (see below). (Un-normalized
  seek is the standard "arrive" behaviour — pulls harder when far, eases in when close — and it is integer-cheap.)
- **`SteerSeparation(agent, others, cfg)`** — push away from neighbors within `sepRadius`: for each OTHER agent `o`
  with `dist² < sepRadius²` (the squared-distance compare; see the int note), accumulate `FxSub(agent.pos, o.pos)`
  (the away-direction, optionally inverse-weighted) into a running sum. The accumulation is integer ADDS (no
  per-pair normalize — the scout's "accumulate raw integer deltas" rule). Scaled by `sepGain`.
- **`StepBoids(agents, cfg, dt)`** — for each agent (FIXED index order): `force = FxAdd(FxScale(SteerSeek, seekGain),
  FxScale(SteerSeparation, sepGain))`; magnitude-clamp `force` to `maxForce` (a deterministic clamp — see below);
  `agent.vel = FxAdd(agent.vel, FxScale(force, dt))`; magnitude-clamp `vel` to `maxSpeed`; `agent.pos =
  FxAdd(agent.pos, FxScale(agent.vel, dt))`. Fixed op order → two-run bit-identical AND bit-exact GPU==CPU.

**The magnitude clamp:** clamping a vector to a max magnitude needs `|v|` (an `FxISqrt`, int64). To keep BD1 lean,
clamp **per-axis** (`FxClampCone`-style on each component to `±maxForce`/`±maxSpeed`) OR use a squared-magnitude
compare + a conditional scale. Per-axis clamp is int32 (no sqrt) and deterministic — use it (document that it is an
axis-box clamp, not a radial clamp — the honest BD-caveat shape; the flock still steers correctly). If a radial
clamp is wanted, that is the int64 path — DON'T, keep BD1 int32.

**THE int32 GOAL (the strongest cross-vendor proof):** keep `boids_steer.comp` PURE INT32 so it MSL-generates
NATIVELY (a TRUE GPU pass on BOTH backends = strict zero-differing-pixel, the GR2/NAV2/FPX2 bar) — NO
`FxNormalize`/`FxISqrt`/`fxmul`-of-large-values on the per-agent path. The ONLY int64 risk is the `dist² <
sepRadius²` separation compare (for Q16.16 positions in a large world, `dx²` overflows int32). MITIGATION: keep the
BD1 scene SMALL (positions within ~a few world units so `dx²` fits int32), OR do the radius compare in int64 while
keeping the rest int32 (then the shader is int64 → Vulkan-only + Metal-CPU-ref). **The implementer MUST choose,
keep it MSL-native if feasible, and DOCUMENT honestly which** (int32-MSL-native strict-GPU-both-backends, OR
int64-Vulkan-only + Metal-CPU-ref byte-identical-by-construction). Lean int32.

`BoidsConfig { fx seekGain, sepGain, sepRadius, maxForce, maxSpeed; FxVec3 target; FxVec3 gravity(=0); }` (host-fixed
Q16.16 tuning). `MeasureBoids(agents, cfg)` → the mean speed, the converged-toward-target distance, the min
pair-separation (the "agents didn't collapse" stat) — deterministic Q16.16.

## Reuse map (file:line — the implementer MUST ground these before coding)
- **fpx Q16.16 toolbox (`engine/sim/fpx.h`, read-only):** `FxVec3`, `FxAdd`, `FxSub`, `FxScale`, `fxmul`,
  `FxDot` (for the squared-distance / squared-magnitude), `kOne`, `kFrac`, the per-axis clamp idiom (grep
  `FxClampCone` in vehicle.h for the scalar clamp shape — re-implement a small `ClampAxis` in boids.h, do NOT
  modify vehicle.h). The `FxBody` integrate pattern (`IntegrateBody`) for the pos/vel update shape. **DO NOT modify
  fpx.h.**
- **The new-shader showcase precedent (`samples/hello_triangle/main.cpp`):** study `--grain-integrate-shot` /
  `--fpx-shot` (the per-agent/per-body compute + the GPU==CPU memcmp + the standalone arg-parse) and (for the
  MSL-native int32 strict-GPU pattern) `--nav-raster-shot` / `--grain-neighbors-shot` (an int32 shader that's in
  `hf_gen_msl` — a true GPU pass both backends). Mirror these.
- **The int32-MSL-native vs int64-Vulkan-only shader wiring:** how `grain_integrate.comp` (int64, NOT in
  hf_gen_msl) vs `grain_cell_count.comp` (int32, IN hf_gen_msl) are registered (CMake DXC SPIR-V + the hf_gen_msl
  list). `boids_steer.comp` follows whichever its int-ness dictates (lean int32 → IN hf_gen_msl).
- **Registration:** `scripts/verify.ps1`, `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller
  rebakes the JSON golden — do NOT**), a NEW `tests/boids_test.cpp` (+ CMake wiring, the active_test/vehicle_test
  pattern).

## Design decisions (locked)
1. **NEW `engine/sim/boids.h`** (header-only, namespace `hf::sim::boids`, `#include "sim/fpx.h"` read-only):
   `struct Agent { FxVec3 pos; FxVec3 vel; }`, `struct BoidsConfig { ... }`, `SteerSeek`, `SteerSeparation`
   (brute-force all-pairs), `StepBoids`/`StepBoidsSteps`, `MeasureBoids`, a small `ClampAxis` helper. Pure integer,
   fixed op order.
2. **NEW `shaders/boids_steer.comp.hlsl`** (the per-agent steer+integrate — ONE thread per agent, brute-force inner
   loop over all agents for separation; copies `StepBoids`'s body VERBATIM). int32 MSL-native if feasible (→ IN
   hf_gen_msl, a true GPU pass both backends), else int64 Vulkan-only (→ NOT in hf_gen_msl, Metal runs the CPU
   `StepBoids`). **DOCUMENT which in the shader header + the spec-completion report.**
3. **Showcase `--boids-steer-shot <out>` (Vulkan) AND `--boids-steer` (Metal) — WIRE BOTH** (standalone arg-parse).
   The SCENE: a small cluster of agents (~32-64) started near each other with a shared seek target offset to one
   side; settle K `StepBoids` steps → the agents STREAM toward the target while SEPARATING (they spread out into a
   loose flock heading to the target, NOT collapsing to a point, NOT exploding). Vulkan: the GPU `boids_steer.comp`
   per step → **memcmp the GPU agent array vs the CPU `StepBoids`** (NO tolerance — the make-or-break). Metal: the
   GPU pass (if int32) OR the CPU reference (if int64). Render a PURE-INTEGER 2D top-down view (each agent a dot at
   its `pos>>kFrac`, optionally a short velocity tick; the target marked). Golden = `tests/golden/metal/
   boids_steer.png` (Mac-baked by the CONTROLLER — DO NOT commit).
4. **PROOFS (fail loudly; exact lines):**
   - **(1) GPU==CPU bit-exact:** the GPU agent array after K steps == the CPU `StepBoids` byte-for-byte. Print
     `boids-steer: {agents:<N>, steps:<K>} GPU==CPU BIT-EXACT`.
   - **(2) determinism:** two runs → identical. Print `boids-steer determinism: two runs BYTE-IDENTICAL`.
   - **(3) seek + separate:** after settling, the flock's mean distance to the target DROPPED (they sought it) AND
     the min pair-separation is ABOVE a floor (separation kept them apart — they didn't collapse). Print
     `boids-steer behavior: {meanToTarget:<D>, minSep:<S>, soughtAndSeparated:true}`; assert both.
   - **(4) control:** a `sepGain = 0` control lets the agents collapse toward the target (min-sep → ~0) — the
     separation does the spreading. Print `boids-steer control: {noSep:collapsed}`.
   - **Golden discipline: ONLY `tests/golden/metal/boids_steer.png`; do NOT commit it.** Existing 177 image goldens
     UNTOUCHED.
5. **Cross-backend bar (INTEGER, strict):** Vulkan GPU == Metal (GPU if int32 / CPU-ref if int64) == golden, ZERO
   differing pixels.
6. **Tests `tests/boids_test.cpp` (NEW, pure CPU):** `SteerSeek` pulls toward the target (the force points
   target-ward); `SteerSeparation` pushes apart two close agents (the force points away from the neighbor), zero
   for agents beyond `sepRadius`; `StepBoids` — a cluster seeking a target spreads + advances (meanToTarget drops,
   minSep above a floor); a `sepGain=0` run collapses; the per-axis clamp caps force/speed; two runs byte-identical.
   Clean under `windows-msvc-asan`. Wire the new test into CMake.
7. **Introspect.** Add exactly `deterministic-boids-steer` (features) + `--boids-steer-shot` (showcases) in
   `engine/editor/introspect.cpp` + update `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the
   controller does that.**

## RHI seam additions (summary)
- **None.** Reuse the existing compute + SSBO + dispatch + read-back path (the grain/fpx surface). `rhi.h` + backend
  dirs UNCHANGED. `engine/sim/fpx.h` + `joint.h` + `grain.h` + `fluid.h` + `cloth.h` + `couple*.h` + `fract.h` +
  `vehicle.h` + `active.h` + `engine/nav/` + `engine/anim/` + `engine/physics/` + all EXISTING shaders UNCHANGED.
  The ONLY new shader is `boids_steer.comp.hlsl`. `boids.h` is a NEW additive sibling (`#include`s fpx.h read-only).
  Report the seam empty (only `boids.h` + the new shader + the showcase/test/introspect are new/changed).

## Out of scope (YAGNI — later BD slices)
The grid-hash neighbor list (BD2 — BD1 is brute-force all-pairs), alignment + cohesion (BD3 — BD1 is seek +
separation only), path-following the A* corridor (BD4), lockstep/rollback (BD5), the lit 3D instanced render (BD6 —
BD1's render is the 2D top-down diagnostic). Orientation/inertia, RVO/ORCA avoidance, 3D volumetric flocking. BD1
claims ONLY: a deterministic integer steering primitive (seek + separation) that streams a small flock toward a
target while keeping them apart, bit-identical CPU↔Vulkan↔Metal, with the integer golden + the four proofs. NOTE
(honest, the GR4 caveat shape): boids are POINTS with steering FORCES (a soft separation push, NOT a hard
non-penetration contact — agents can briefly overlap); the per-axis clamp is an axis-box, not a radial magnitude
clamp (deterministic + integer-cheap, the honest simplification). Determinism + cross-platform bit-identity is the
headline.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 103 + the NEW `boids_test`). Clean under `windows-msvc-asan`
   (build+run `boids_test` + `introspect_test`).
2. **proofs + visual:** `--boids-steer-shot` on Vulkan: the 4 proofs + exit 0, under the Vulkan-validation gate →
   ZERO VUID (set BOTH `VK_LAYER_PATH` + `VK_INSTANCE_LAYERS`; confirm the layer LOADED). **VERIFY the image shows
   a loose flock streaming toward the target (spread out, not collapsed to a point, not exploded/scrambled).**
3. Metal: `visual_test --boids-steer` → new golden `tests/golden/metal/boids_steer.png`; two runs DIFF 0.0000.
   **Confirm `visual_test.mm` in the diff; DOCUMENT whether `boids_steer.comp` is in `hf_gen_msl` (int32, a true
   Metal GPU pass) or NOT (int64, Metal runs the CPU StepBoids).** Cross-vendor STRICT ZERO.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` = ONLY `boids_steer.png` added; the other
   177 byte-identical. `git diff master --stat -- tests/golden` = ONLY `boids_steer.png` (metal) + the introspect
   json (controller rebake, post-gate).
5. Introspect: exactly `+deterministic-boids-steer` + `--boids-steer-shot` added; introspect test updated.
6. Seam grep clean (`rhi.h` UNCHANGED; `engine/sim/fpx.h`/`joint.h`/`grain.h`/`fluid.h`/`cloth.h`/`couple*.h`/
   `fract.h`/`vehicle.h`/`active.h` + `engine/nav/` + `engine/anim/` + `engine/physics/` + ALL existing shaders
   byte-unchanged). `scripts/verify.ps1` updated: `boids_steer` golden in the Mac loop + `--boids-steer-shot` in
   `$vkShots`. **The ONLY new shader is `boids_steer.comp.hlsl`; document its hf_gen_msl membership.**
