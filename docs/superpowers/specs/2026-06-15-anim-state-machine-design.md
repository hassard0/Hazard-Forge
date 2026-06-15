# Slice BL — Animation State Machine / Blend Tree (Phase 4 #13) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. A real
> gameplay-animation system on top of the existing skeletal animation (Slice O) + blending.

**Goal:** Drive a skinned character with a parameter-driven animation state machine: named states (each an
animation clip), transitions between them gated by parameter conditions, and cross-fading during a
transition. Builds on the existing `engine/anim` (`SampleAnimation`, `BlendAnimations`) + the Fox model
(which has `Survey`/`Walk`/`Run` clips → idle/walk/run states). A scripted parameter timeline drives the
FSM deterministically; `--anim-fsm-shot` captures a fixed mid-transition frame; golden-verified + the FSM
state asserted.

## Why scripted params (determinism)

The FSM output is a pure function of (state graph, parameter timeline, dt sequence). With a fixed SCRIPTED
parameter timeline + fixed timestep (no input/RNG/clock), the current state, transition progress, blend
weight, and final joint palette at every frame are deterministic ⇒ the captured frame + asserted state are
bit-stable. The cross-fade weight is `transitionTime / duration` (a deterministic ratio).

## Design decisions (locked)

1. **State machine (engine/anim/state_machine.{h,cpp}, pure CPU, no backend symbols).** Namespace
   `hf::anim`. Reuses `Skeleton`, `Animation`, `SampleAnimation`, `BlendAnimations`.
   - `struct AnimState { std::string name; int animationIndex; bool loop; float speed; };` (references an
     animation in a caller-provided `std::vector<Animation>`).
   - `struct Transition { int from; int to; int paramIndex; enum class Cmp { Greater, Less } cmp;
     float threshold; float duration; };` (parameter-threshold-gated; `from == -1` = "any state").
   - Parameters: a small named `float` set (`SetParam(name, value)` / index lookup). (Bool/trigger params
     are just 0/1 floats — keep it to floats; document.)
   - `class StateMachine`: owns states + transitions + params + runtime `{current, stateTime,
     transitioningTo (-1 if none), transitionTime}`.
     - `void Update(float dt)`: advance `stateTime += dt * states[current].speed`. If NOT transitioning,
       evaluate outgoing transitions from `current` (and any `from==-1`) IN ORDER; the FIRST whose
       condition holds (`param cmp threshold`) begins a transition (`transitioningTo`, `transitionTime=0`).
       If transitioning, `transitionTime += dt`; when `transitionTime >= duration`, complete
       (`current = transitioningTo`, reset `transitioningTo=-1`, carry/blend `stateTime` as documented).
       Deterministic, fixed evaluation order.
     - `std::vector<math::Mat4> Evaluate(const Skeleton&, const std::vector<Animation>&) const`: if not
       transitioning → `SampleAnimation(skel, anims[current.animationIndex], stateTime)`; if transitioning
       → `BlendAnimations(skel, anims[from], anims[to], stateTimeFrom, stateTimeTo,
       weight=transitionTime/duration)` (match the existing BlendAnimations signature — inspect it). The
       blend weight 0→1 cross-fades from→to.
     - Accessors for tests/showcase: `CurrentStateName()`, `IsTransitioning()`, `BlendWeight()`,
       `TransitioningToName()`.

2. **Showcase `--anim-fsm-shot <out>` (Vulkan) / `--anim-fsm` (Metal).** Load the Fox (the existing
   skinning showcase asset) + its `Survey`/`Walk`/`Run` animations as states idle/walk/run. Build an FSM:
   idle --(speed>0.3)--> walk --(speed>0.7)--> run, with reverse transitions (run--(speed<0.7)-->walk,
   etc.), each with a fixed cross-fade duration. Drive a SCRIPTED `speed` timeline over a fixed number of
   fixed-dt steps so the character is, at the FIXED capture step, MID cross-fade (e.g. walk→run at
   blend≈0.5 — chosen + documented). Render the skinned Fox at that blended pose, lit + shadowed. Print
   `anim-fsm: {state:<from>-><to>, blend:0.50, speed:<v>, step:S}` (deterministic). New golden
   `tests/golden/metal/anim_fsm.png` (Metal two runs DIFF 0.0000). Existing 37 image goldens UNTOUCHED
   (including the existing `anim_blend` 2-clip golden — this is a NEW showcase, not a change to it).

3. **Render reuses the existing skinned/lit/shadowed pipeline (Slice O).** The FSM produces a joint
   palette exactly like `SampleAnimation`/`BlendAnimations` already do; the GPU skinning path is
   unchanged. NO new RHI, NO new shader.

4. **Tests `tests/anim_fsm_test.cpp` (pure CPU, no GPU):**
   - **Transition condition:** a `speed>0.3` transition fires when `speed=0.5`, NOT when `speed=0.1`;
     fixed evaluation order picks the first satisfied transition.
   - **Transition timing + cross-fade:** after beginning a transition, `BlendWeight()` ramps
     `0 → 1` linearly over `duration`; the FSM is `IsTransitioning()` until `transitionTime >= duration`,
     then `current` becomes the target and `IsTransitioning()` is false.
   - **Any-state transition** (`from==-1`) fires from any current state when its condition holds.
   - **Determinism:** the same scripted (param,dt) sequence yields identical
     `{current, transitioning, blend}` per step across two runs.
   - **Evaluate sanity:** at blend=0 the palette equals `SampleAnimation(from)`; at blend=1 it equals
     `SampleAnimation(to)` (within the blend's documented convention).
   - Clean under `windows-msvc-asan`.

5. **Introspect.** Add exactly `animation-state-machine` (features) + `--anim-fsm-shot` (showcases).
   One-pattern rebake.

## RHI seam additions (summary)
- **None.** Pure CPU on top of the existing anim + skinning. New files (`engine/anim/state_machine.{h,cpp}`,
  `tests/anim_fsm_test.cpp`) add ZERO backend symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
Blend SPACES (2D directional blend), additive layers/masks, IK, root motion, event/notify callbacks,
a visual FSM editor, multiple simultaneous transitions / transition interruption blending, sync groups,
mirroring. One linear state graph, parameter-threshold transitions, single cross-fade, golden + asserted.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 37) + new `anim_fsm_test` (conditions, timing,
   cross-fade, any-state, determinism, evaluate endpoints). Clean under `windows-msvc-asan`.
2. `--anim-fsm-shot` on Windows/Vulkan: controller visual review — a recognizable posed/animated Fox at
   the blended pose, lit + shadowed, coherent; the `anim-fsm: {...}` state line is deterministic (two runs
   → byte-identical capture). Run under the AT Vulkan-validation gate → ZERO errors.
3. Metal: `visual_test --anim-fsm` → new golden `tests/golden/metal/anim_fsm.png`; two runs DIFF 0.0000;
   state line matches Vulkan.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `anim_fsm.png` added;
   the other 37 byte-identical (incl. `anim_blend.png`).
5. Introspect JSON rebaked exactly `+animation-state-machine` + `--anim-fsm-shot`; introspect test
   updated; no other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new `anim_fsm`
   image golden in the Mac round-trip loop.
