# Slice DD — Runtime Cubemap-Capture Reflection Probe (Phase 4 #50) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The dynamic counterpart
> to DA (which reused a STATIC env cubemap): a runtime probe that RENDERS the actual scene into a 6-face
> cubemap, so reflective surfaces mirror the real geometry. Reuses DA's box-projection + the existing scene
> render path. Deterministic, with a clean capture-correctness proof.

**Goal:** A runtime reflection probe — from a fixed probe position, render the scene into the 6 faces of a
cubemap (each face = the scene viewed through a 90° FOV down a +X/-X/+Y/-Y/+Z/-Z axis), then a reflective
surface samples the captured cubemap (box-projected, reusing DA) so it reflects the actual scene objects. A
`--captureprobe-shot` showcase shows a reflective sphere/floor reflecting the surrounding scene (the colored
objects appear in the reflection); golden-verified on both backends. The existing reflection/IBL path + its
goldens are UNTOUCHED (this is a NEW path behind the showcase flag). The pass carries its proof: a captured
cubemap FACE is BYTE-IDENTICAL to the scene rendered directly with that face's view/projection (the capture
is a faithful scene render) — asserted internally, fail loudly on any diff.

## Why this is render-safe (the capture-correctness proof)

The probe capture renders the scene 6 times, once per cube face, with a fixed 90°-FOV view/proj per axis. So
the `+X` face of the captured cubemap MUST equal the scene rendered standalone with the `+X` view/proj — the
capture pass and a direct render use the IDENTICAL scene + shaders + camera math. The showcase INTERNALLY
renders the scene directly with one face's view/proj and asserts it is BYTE-IDENTICAL (SHA) to that face read
back from the captured cubemap — proving the capture is a correct, deterministic scene render (no face
orientation bug, no projection skew, no RT format mismatch). Deterministic (fixed scene/probe → fixed cubemap
→ fixed reflection). Two runs byte-identical. (Same internal-assert discipline as the other slices; here the
proof is a capture==direct-render identity rather than a disabled-path no-op.)

## Reuse map (builds on proven pieces)

- **DA (`engine/render/reflection_probe.h`, `shaders/reflprobe.frag.hlsl`)** — the box-projection
  `BoxProject` + the reflective-surface shader. DD swaps the STATIC env cubemap for the freshly CAPTURED
  cubemap; the box-projection + sampling are unchanged.
- **The scene render path** — the capture renders the SAME scene (objects + lighting) the main pass renders,
  just 6× into cube faces with per-face view/proj. Reuse the lit pass.
- **The cube-face view math** — the 6 standard cube-face view matrices (look down each ±axis, the standard
  up-vectors) + a 90° FOV perspective; pure-CPU shared (engine/render/cubemap.h), unit-tested.

## Design decisions (locked)

1. **Cube-face math (engine/render/cubemap.h, header-only pure CPU, no backend symbols).** Namespace
   `hf::render::cubemap`. Mirrors `reflection_probe.h`/`gtao.h` (shared with the capture code + the unit test).
   - `math::Mat4 FaceView(int face, const math::Vec3& probeCenter)` — the view matrix for cube face `face`
     (0=+X,1=-X,2=+Y,3=-Y,4=+Z,5=-Z) looking from `probeCenter` down that axis with the STANDARD cubemap
     up-vector per face (document the convention — match the GPU cubemap-sampling convention so the captured
     faces line up with how `reflprobe.frag` samples them). `math::Mat4 FaceProj(float zNear, float zFar)` —
     the 90°-FOV (square, aspect 1) perspective for a face. Document the handedness/depth convention to MATCH
     the engine's main projection.
   - `math::Vec3 DirToFaceUV(const math::Vec3& dir, int& outFace, math::Vec2& outUV)` — the cubemap lookup:
     given a sample direction, the major axis selects the face + the UV (the standard cube mapping); used to
     unit-test that `FaceView`/`FaceProj` + `DirToFaceUV` are mutually consistent (a direction down +X maps to
     face 0 center, etc.).
   - Pure, deterministic. Document the full convention so capture + sampling agree.

2. **Probe capture (engine + a NEW additive RHI interface IF cubemap RTs are missing).** From the probe
   center, render the scene into the 6 faces of a color cubemap (+ a depth target per face or shared). REUSE
   the existing lit scene-render pass with the per-face `FaceView`/`FaceProj`. The cubemap COLOR render target
   + per-face render almost certainly needs an ADDITIVE RHI interface (create a cubemap render target +
   begin-render-pass into face `i` + a barrier so the captured cubemap is sampleable in the reflection pass).
   Add it as a PURE INTERFACE in `engine/rhi/rhi.h` (e.g. `CreateCubemapTarget(size, format)`,
   `BeginCubemapFace(target, face)`, and the cubemap as a sampleable texture) with the impls INSIDE the
   backend dirs (`engine/rhi_vulkan/`, `engine/rhi_metal/`, `metal_headless/`) — NEVER a backend code symbol
   above the seam. Document + report exactly what was added. Existing pipelines/paths that don't use it are
   byte-for-byte unchanged (the existing goldens MUST stay byte-identical — verify). HLSL→SPIR-V→MSL via the
   existing toolchain.

3. **Reflection pass (reuse DA's `reflprobe.frag.hlsl` OR a `captureprobe.frag.hlsl` variant).** The
   reflective surface computes `R`, box-projects it (DA), and samples the CAPTURED cubemap (instead of the
   static env). Bind the captured cubemap as the reflection source. The box-projection + blend are DA's.

4. **Showcase `--captureprobe-shot <out>` (Vulkan) / `--captureprobe` (Metal).** A reflective sphere/floor
   surrounded by distinct colored objects; the probe captures the scene into the cubemap; the reflective
   surface shows the surrounding objects in its reflection (the real scene, not a static env). Fixed camera/
   probe/scene. Print `capture-probe: {faces:6, cubeSize:N}` (deterministic). INTERNALLY render the scene
   directly with face-0's view/proj and assert BYTE-IDENTICAL (SHA) to the captured cubemap's face 0 — fail
   loudly on any diff (the capture-correctness proof). New golden `tests/golden/metal/capture_probe.png` (Metal
   two runs DIFF 0.0000, gate on the compare.sh EXIT CODE). Existing 70 image goldens UNTOUCHED.

5. **Determinism.** Fixed scene/probe/camera; the 6 face renders are deterministic; the captured cubemap +
   the reflection are deterministic. Two runs byte-identical.

6. **Tests `tests/cubemap_test.cpp` (pure CPU, no GPU):**
   - **Face-view/proj sanity:** `FaceView(face, center)` looks down the correct ±axis with the documented up;
     a point on the +X axis from the center projects to ~the center of face 0; the 6 faces tile all
     directions.
   - **Direction↔face consistency:** `DirToFaceUV(dir)` selects the major-axis face + the right UV; round-trip
     with `FaceView`/`FaceProj` (a direction down a face's axis → that face, UV ≈ center); the 6 standard
     axes map to the 6 faces.
   - **Up-vector convention:** the per-face up-vectors match the cubemap sampling convention (so captured
     faces aren't flipped/rotated vs how the shader samples — hand-checked for at least +Y/-Y which are the
     usual flip pitfalls).
   - **Determinism:** same inputs → same matrices.
   - Clean under `windows-msvc-asan`.

7. **Introspect.** Add exactly `capture-reflection-probe` (features) + `--captureprobe-shot` (showcases).

## RHI seam additions (summary)
- **Expected: an ADDITIVE cubemap-render-target interface** (the engine likely has no cubemap RT). Put it in
  `engine/rhi/rhi.h` as a PURE interface (cubemap target create + per-face begin-pass + sampleable-cubemap
  bind) with impls ONLY in the backend dirs; existing pipelines that don't use it are byte-for-byte unchanged.
  Report exactly what was added. New non-backend files (`engine/render/cubemap.h`, the capture-probe shader,
  `tests/cubemap_test.cpp`) add ZERO above-seam backend code symbols. Seam grep above-seam stays at baseline
  (2) — the cubemap-RT backend code lives entirely in the backend dirs. Report the seam result + the exact
  rhi.h additions.

## Out of scope (YAGNI)
Per-frame dynamic re-capture (capture ONCE for the showcase — a fixed snapshot; dynamic per-frame is a future
optimization), roughness prefiltering / mip convolution of the captured cubemap (sharp reflection only — note
future), multiple probes / blending, probe relighting, capturing the reflective surface itself into its own
probe (avoid recursion — the captured scene excludes the reflective surface or uses a fixed order; document),
HDR-cubemap tonemapping nuances beyond the existing pipeline. One single static 6-face scene capture into a
cubemap + box-projected reflection, with a capture==direct-render byte-identical proof + a cube-face unit test,
golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 70) + new `cubemap_test` (face-view/proj sanity,
   direction↔face consistency, up-vector convention, determinism). Clean under `windows-msvc-asan`.
2. **Capture-correctness proof + visual:** `--captureprobe-shot` on Vulkan: the reflective surface reflects
   the surrounding scene objects (a recognizable real-scene reflection), coherent; the INTERNAL face-0
   direct-render is BYTE-IDENTICAL (SHA) to the captured cubemap's face 0; the `capture-probe: {...}` line is
   deterministic (two runs → byte-identical capture). Run under the AT Vulkan-validation gate → ZERO errors
   (the capture render passes + the capture→sample barrier must be SYNC-HAZARD-free).
3. Metal: `visual_test --captureprobe` → new golden `tests/golden/metal/capture_probe.png`; two runs DIFF
   0.0000 (gate on the compare.sh EXIT CODE).
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `capture_probe.png` added;
   the other 70 byte-identical (the additive cubemap-RT interface must not perturb existing rendering — verify
   a broad set of existing proofs + the existing goldens stay byte-identical).
5. Introspect JSON rebaked exactly `+capture-reflection-probe` + `--captureprobe-shot`; introspect test
   updated; no other drift.
6. Seam grep clean (above-seam == baseline 2; report the rhi.h additive cubemap-RT interface + confirm its
   backend code is only in the backend dirs). `scripts/verify.ps1` updated to include the new `capture_probe`
   image golden in the Mac round-trip loop. (`captureprobe.frag.hlsl` / the capture path are NEW — no re-bake
   of other Metal goldens needed; gate the new golden on the compare.sh EXIT CODE.)
