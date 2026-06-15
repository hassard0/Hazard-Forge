# Slice AP — Temporal Anti-Aliasing (TAA) — Design

> Autonomous-session spec (standing user directive: bold decisions, self-approve, document every call
> here for later review). Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF
> 0.0000. Phase 2 finale.

**Goal:** Add temporal anti-aliasing — accumulate several sub-pixel-jittered frames into a smooth,
alias-free image. Each frame jitters the projection by a sub-pixel Halton offset; the resolve pass
blends the current frame with reprojected history, neighborhood-clamped to suppress ghosting. The
showcase renders a FIXED number of accumulation frames over a static scene/camera, so two runs are
bit-identical (deterministic golden).

## Determinism contract (the crux)

TAA is temporal: the result depends on the accumulation history. To stay golden-verifiable:
- The `--taa-shot` showcase runs a **FIXED count of N=8 accumulation frames** over a **static scene +
  static camera + static lights**, advancing only the jitter index 0..7, then captures frame 8.
- The Halton(2,3) jitter sequence is **deterministic** (pure function of the frame index, no time, no
  RNG). Same index → same offset on both backends.
- Static camera ⇒ motion vectors are **zero** ⇒ history reprojection samples the same UV (identity).
  The plumbing computes the reprojected UV from the previous-frame view-proj stored in the frame UBO
  (real, exercised), but evaluates to identity for this static shot. A future moving-camera shot would
  exercise the non-trivial reprojection without any seam change.
- First frame (index 0, empty history): resolve outputs the current frame unblended (blend weight 1.0)
  so accumulation has a defined start. History texture is cleared once before frame 0.

Result: a single aliased frame's hard edges resolve into smooth anti-aliased edges. Controller eyeballs
the captured PNG to confirm visibly smoother silhouettes than `--shot` (a first-time correctness
property the golden can't self-verify); thereafter two Metal runs diff to 0.0000.

## Design decisions (locked)

1. **Pure-CPU math header `engine/render/taa.h`** (header-only, NO device/backend symbols — same
   pattern as `engine/render/ssr.h`). Namespace `hf::render::taa`. Contents:
   - `float Halton(int base, int index)` — the radical-inverse Halton low-discrepancy sequence
     (index is 1-based internally; `Halton(2,i)` / `Halton(3,i)` give the X/Y bases).
   - `math::Vec2 Jitter(int frameIndex, int width, int height)` — returns the **NDC-space** sub-pixel
     offset: `((Halton(2,frameIndex+1)-0.5)*2/width, (Halton(3,frameIndex+1)-0.5)*2/height)`. This is
     added to the projection matrix's `m[2][0]`/`m[2][1]` (clip-space XY translation per unit W) — a
     standard jittered-projection injection, applied AFTER the base projection is built so it does not
     disturb depth.
   - `math::Vec3 ClipHistoryToNeighborhood(const math::Vec3& history, const math::Vec3& boxMin,
     const math::Vec3& boxMax)` — clip the history color toward the current 3x3 neighborhood AABB
     (per-channel clamp is acceptable and is what the shader does; document that we use the simple AABB
     clamp, not the full clip-toward-center variant — YAGNI for a static shot). Used by the unit test to
     mirror the shader exactly.
   - `math::Vec3 ResolveBlend(const math::Vec3& current, const math::Vec3& clampedHistory,
     float alpha)` — exponential blend `lerp(clampedHistory, current, alpha)`; `alpha=0.1` steady
     state, `alpha=1.0` on the first frame.

2. **Jittered projection — engine side, no new RHI.** The per-frame UBO already carries the view-proj.
   `main.cpp` (Vulkan) and `visual_test.mm` (Metal) build the jittered proj by adding `Jitter(...)`
   into the projection before composing view-proj, per accumulation frame. NO RHI seam change for the
   jitter — it rides the existing `SetFrameUniforms` path. Also stash the **previous** frame's
   (unjittered) view-proj in the frame UBO for reprojection (new field `prevViewProj` in the frame UBO
   struct — pure additive; bump `kFrameUboSize` only if the struct overflows the current 1024 B, which
   it should not — verify and document the byte count).

3. **History buffer + resolve pass — reuse the existing post/RT plumbing (Slice G + Y).** The scene
   renders into the existing HDR RT (`RGBA16_Float`). Add ONE persistent history texture
   (`RGBA16_Float`, same size) and a `taa_resolve` fullscreen pass that reads (current HDR RT, history)
   and writes the resolved image, which is then (a) copied/used as the next frame's history and (b) fed
   into the existing tonemap/post chain for the final captured frame. Sequencing:
   - Render scene (jittered) → currentHDR.
   - `taa_resolve(currentHDR, history) → resolved` with neighborhood clamp + blend.
   - history := resolved (ping-pong two `RGBA16_Float` textures to avoid read=write hazard; standard).
   - On the final frame, run the existing post chain on `resolved` and capture.
   This reuses `Format::RGBA16_Float`, the fullscreen-triangle `post.vert`, and the RT bind path. The
   only new shader is `shaders/taa_resolve.frag.hlsl`.

4. **`taa_resolve.frag.hlsl`** — samples the current HDR color, the 3x3 neighborhood (building the
   min/max AABB over the 8 neighbors + center), the reprojected history (using `prevViewProj` to map
   the current UV back to the previous frame's UV — identity for the static shot), clamps history to the
   neighborhood AABB, and blends `lerp(clampedHistory, current, alpha)`. `alpha` and the
   first-frame-flag arrive via a small push constant or a reused frame-UBO field (pick the path that
   matches how the other post passes receive parameters — inspect bloom/ssao; document the choice).
   HLSL→SPIR-V→MSL through the existing toolchain; `HF_MSL_GEN` guards already handle the
   UV/push-constant conventions.

5. **Showcase wiring.**
   - Windows/Vulkan: `hello_triangle.exe --taa-shot <out>` renders the N=8 accumulation loop over the
     default lit+shadowed scene (same scene as `--shot`) and captures frame 8.
   - Metal: `metal_headless/visual_test` gains a `--taa` showcase / second entry rendering the identical
     8-frame loop at the identical jitter sequence; two runs DIFF 0.0000.
   - New golden `tests/golden/metal/taa.png`. Existing 22 goldens + the introspect JSON golden UNTOUCHED
     — EXCEPT: add `temporal-anti-aliasing` to the introspect `features` list and a `--taa-shot` entry
     to `showcases`, which means `tests/golden/introspect/default_scene.json` IS intentionally rebaked
     (one feature + one showcase added). Update that golden deliberately and note it in the verify gate;
     do NOT touch any image golden.

## RHI seam additions (summary)
- **None required for jitter** (rides `SetFrameUniforms`).
- Frame UBO struct gains `prevViewProj` (additive; confirm it fits in `kFrameUboSize`=1024, else bump
  and document). History textures use the existing `RGBA16_Float` + RT path. New fullscreen pass uses
  existing post infrastructure.
- Hard rule unchanged: NO `vk*`/`MTL*`/`Backend::Metal`/`mtl::` symbols above the backend dirs. New
  files (`engine/render/taa.h`, `shaders/taa_resolve.frag.hlsl`, `tests/taa_test.cpp`) add ZERO backend
  types. Seam grep must stay at the benign baseline (no NEW real backend symbols above the seam).

## Out of scope (YAGNI)
Motion-vector G-buffer for dynamic objects, velocity-based history rejection beyond neighborhood clamp,
variance clipping / YCoCg clamp, sharpening pass, dynamic resolution, jitter on a moving camera (the
reprojection plumbing is present but exercised only as identity here). One static scene, 8 jittered
frames, neighborhood-clamped accumulation, golden-verified on both backends.

## Verification gate
1. `ctest --preset windows-msvc-debug` stays green (existing 22) + new `taa_test` (Halton determinism,
   jitter offset bounds, neighborhood clamp = per-channel AABB, resolve blend lerp).
2. `--taa-shot` on Windows/Vulkan produces a recognizably anti-aliased version of the default scene
   (controller visual review: edges smoother than `--shot`).
3. Metal: `visual_test` TAA showcase → new golden `tests/golden/metal/taa.png`; two runs DIFF 0.0000;
   existing 22 image goldens unchanged.
4. `tests/golden/introspect/default_scene.json` rebaked with exactly `temporal-anti-aliasing` (features)
   + `--taa-shot` (showcases) added; introspect test updated; no other JSON drift.
5. Seam grep clean (no new backend symbols above the seam). ASan preset still builds; `taa_test` runs
   clean under `windows-msvc-asan`.
6. `scripts/verify.ps1` updated to include the new `taa` image golden in the Mac round-trip loop.
