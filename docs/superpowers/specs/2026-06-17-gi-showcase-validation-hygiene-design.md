# Slice DQ — GI-Showcase Vulkan-Validation Hygiene (agentic-operability) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> An AGENTIC-OPERABILITY / quality-pillar hardening: the engine's mission requires **Vulkan-validation-CLEAN**.
> The DDGI showcases (`--ddgi-shot`, `--ddgiocc-shot`, `--probedist-shot`, and any probe showcase that runs the
> lit pass) currently emit **8× `VUID-vkCmdDraw-None-09600`** (a SAMPLED_IMAGE / image-layout core-validation
> error on a dummy/unused sampled image — most likely the dummy shadow map or a probe atlas bound to the lit
> pipeline but not transitioned to `SHADER_READ_ONLY_OPTIMAL` before the lit draw). This was introduced during
> the GI arc and **slipped through every consolidation** because `verify.ps1`'s validation gate runs only a FIXED
> list of 8 non-GI showcases. This slice (a) FIXES the VUID at its shared source as a **pixel no-op** (all goldens
> byte-identical) and (b) HARDENS the gate by adding the GI showcases to it so it can never regress. NO new RHI,
> NO golden change. Branch: slice-dq-givalidation.

**Goal:** Make the GI showcases Vulkan-validation-CLEAN (zero `VUID-vkCmdDraw-None-09600`, zero new VUID/
SYNC-HAZARD/UNASSIGNED) WITHOUT changing a single pixel of any golden, and add them to the permanent validation
gate. The VUID is a layout/tracking bug on a sampled image the lit pass binds but the GI showcase path never
transitions; fixing the layout (or properly clearing+transitioning the dummy) removes the error while the sampled
VALUES stay identical, so every existing golden is byte-for-byte unchanged.

## Investigation first (the implementer MUST do this before editing)
1. **Reproduce + localize.** Build `windows-msvc-debug`, run `--ddgi-shot` under the validation layer (set
   `VK_LAYER_PATH` to the conan `VkLayer_khronos_validation.json` dir + `VK_INSTANCE_LAYERS=
   VK_LAYER_KHRONOS_validation`, as `scripts/verify.ps1`'s validation gate does). Capture the FULL `VUID-vkCmdDraw-
   None-09600` messages — the validation layer names the **descriptor binding / image** at fault. Identify WHICH
   sampled image (the dummy shadow-map atlas? a probe/reflection atlas? the env cubemap?) is bound to the lit
   pipeline in the GI showcase path but left in the wrong layout (UNDEFINED/GENERAL instead of
   SHADER_READ_ONLY_OPTIMAL) at the lit draw.
2. **Find the shared setup.** Locate where the GI showcase(s) create + bind that image and where (if anywhere) the
   non-GI lit showcases transition it (the non-GI `--shot`/`--csm-shot` are validation-clean, so they DO transition
   it — diff the GI path against them to find the missing transition/clear). Confirm whether the image is
   meaningfully SAMPLED (affects lighting) or a pure dummy (bound to satisfy the descriptor layout but its sampled
   value doesn't change the GI showcase's lit result).

## Design decisions (locked)
1. **Fix at the shared source, as a pixel NO-OP.** Add the missing image-layout transition (or a defined
   clear+transition) so the sampled image is in `SHADER_READ_ONLY_OPTIMAL` at the lit draw in the GI showcase
   path — mirroring exactly what the validation-clean non-GI lit showcases already do. The fix MUST NOT change the
   sampled VALUES: 
   - If the image is a true dummy whose content the GI lit result does not depend on, transitioning the layout is
     a pure no-op for pixels.
   - If it IS sampled (e.g. a dummy shadow map that the lit pass reads), clear it to the SAME defined value that
     produces the CURRENT lit result (e.g. the "fully lit / no shadow" depth the non-GI path uses), so the golden
     is unchanged. **The make-safe gate is: ALL existing goldens stay byte-identical (verify on both backends).**
   - Prefer reusing the existing transition/clear helper the non-GI path uses (do NOT invent a new RHI entry point;
     this is showcase/setup code above the seam). If the proper fix genuinely requires an RHI change, STOP and
     report (it shouldn't — the non-GI showcases already do this with the existing surface).
2. **NO new RHI, NO new golden, NO shader change.** This is showcase/host-setup + maybe a render-graph barrier
   ordering fix, all above the seam. rhi.h + rhi_factory (dispatch baseline 2) + backend dirs UNCHANGED. The lit
   shaders + all pipelines UNCHANGED. No introspect change (not a user-facing feature).
3. **Harden the gate.** In `scripts/verify.ps1`, add the GI showcases to the `$vkShots` validation-gate list
   (the array around line 296 that currently holds `--shot`,`--csm-shot`,`--mt-shot`,`--mdi-shot`,`--bindless-shot`,
   `--gpudriven-shot`,`--gpucull-draw-shot`,`--hiz-cull-shot`): add `--ddgi-shot`, `--ddgiocc-shot`,
   `--probedist-shot`, `--probegi-shot`, `--probecapture-shot`, `--probesh-shot`, `--probeinterp-shot` (every GI
   showcase that records a draw/dispatch — confirm the exact flag names parsed in main()). After the fix, the gate
   must report `validation clean` for ALL of them (zero VUID/SYNC-HAZARD/UNASSIGNED). Document that this closes the
   gap where the GI showcases were never validation-gated.

## RHI seam additions (summary)
- **None.** Showcase/host-setup layout-transition fix using the EXISTING barrier/transition surface the non-GI lit
  showcases already use; the gate change is a verify.ps1 list edit. rhi.h + backend dirs UNCHANGED, dispatch
  baseline 2. Report the seam.

## Out of scope (YAGNI)
Any rendering/behavior change, any new RHI, refactoring the showcase shadow-map plumbing beyond the minimal
layout-correctness fix, fixing validation in non-GI showcases (they're already clean). ONE minimal pixel-no-op
layout fix that makes the GI showcases validation-clean + the gate hardened to keep them clean.

## Verification gate
1. `ctest --preset windows-msvc-debug` still 79? (note: DP merged → 80 tests on master now; the count is whatever
   master has — this slice adds NO test; just confirm ctest stays green at master's count) + clean under
   `windows-msvc-asan`.
2. **VUID GONE:** under the validation layer, `--ddgi-shot`, `--ddgiocc-shot`, `--probedist-shot` (+ the other GI
   showcases) emit ZERO `VUID-vkCmdDraw-None-09600` and zero other VUID/SYNC-HAZARD/UNASSIGNED. Show the before
   (8 lines) → after (0 lines) for at least `--ddgi-shot` and `--ddgiocc-shot`.
3. **Render-invariance (CRITICAL — the make-safe gate):** `git diff master --stat -- tests/golden` is EMPTY (NO
   golden changes at all — this is a pure layout/validation fix). Spot-check a broad set still byte-identical on
   Vulkan: `--ddgi-shot`, `--ddgiocc-shot`, `--probedist-shot`, `--probegi-shot`, `--probesh-shot` captures
   byte-identical to a pre-fix capture (the fix did not move a pixel). **If ANY golden changes, STOP and report —
   the fix touched sampled values and needs the clear-to-matching-value approach instead.**
4. **Gate hardened:** `scripts/verify.ps1`'s `$vkShots` now includes the GI showcases; running the Windows
   validation gate reports `validation clean` for all of them.
5. Metal: no golden change → no Mac re-bake strictly required, but confirm the Mac `metal_headless` still BUILDS
   (the fix is Vulkan-showcase host code or backend-agnostic barrier ordering; if it touches only Vulkan-showcase
   setup, Metal is unaffected). If the fix is in shared render-graph barrier code, run the full Mac golden loop to
   confirm DIFF 0.0000 unchanged.
6. Seam grep clean (rhi.h UNCHANGED — no new RHI). No introspect change (introspect golden UNCHANGED).

## Verification gate summary
GI showcases validation-CLEAN (0× VUID-09600); `git diff master -- tests/golden` EMPTY (every golden byte-
identical — pure pixel no-op); verify.ps1 $vkShots hardened to include the GI showcases; ctest green at master's
count + asan; seam baseline 2 + rhi.h unchanged; Metal still builds. NO new RHI, NO golden, NO introspect, NO
shader change.
