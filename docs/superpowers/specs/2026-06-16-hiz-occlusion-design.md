# Slice CJ — Hi-Z Occlusion Culling (Phase 4 #35) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The AAA-completion of
> the GPU-driven culling pipeline: frustum (CD) + OCCLUSION. Render-invariant by construction.

**Goal:** Cull objects that are fully hidden behind closer geometry (not just off-screen). A depth pre-pass
renders all objects' depth; a Hi-Z (hierarchical-Z) max-depth pyramid is built; the cull compute pass tests
each object's screen-space bounding rectangle's NEAREST depth against the Hi-Z's farthest depth in that
region — if the object's nearest point is farther than everything already drawn there, it is fully occluded
and culled. CONSERVATIVE (never culls a visible object). Proven byte-identical to a frustum-only render
(occlusion removes only fully-hidden objects that contribute zero pixels) + the occluded count > 0 + the
GPU occluded count matches a CPU Hi-Z reference.

## Reuse map (builds on proven pieces)

- **CD (`engine/render/gpu_culled.h`, `shaders/gpudriven_cull.comp.hlsl`, `frustum.h`)** — the per-draw
  cull+compact compute + the CPU reference. This slice ADDS an occlusion test to the same compute pass.
- **The G-buffer / depth (SSR/SSAO/SSGI)** — the depth buffer the Hi-Z is built from.
- **BM/BZ/CB** — the per-draw + MDI + bindless render of the survivors (optional — the showcase can render
  per-object; the focus is the OCCLUSION CULL correctness, not the draw path).

## Design decisions (locked)

1. **Hi-Z math (engine/render/hiz.h, pure CPU, no backend symbols).** Namespace `hf::render::hiz`. Mirrors
   `frustum.h`/`gpu_culled.h` (shared with the shader/test).
   - `BuildHiZ(const float* depth, int w, int h, std::vector<HiZMip>& mips)` — build a mip pyramid where
     mip 0 = the depth buffer and each coarser mip texel = the **MAX (farthest)** of its 2x2 children
     (conservative: a coarse texel's value is the farthest surface in that block, so "if your nearest is
     beyond this, something nearer covers the whole block"). Document the depth convention (view-linear or
     NDC; pick + document; MAX = farthest in that convention).
   - `bool IsOccluded(const math::Vec3& aabbMin, const math::Vec3& aabbMax, const math::Mat4& viewProj,
     int screenW, int screenH, span<const HiZMip>)` — project the world AABB's 8 corners to screen →
     screen-space rect + the object's NEAREST screen depth; select the Hi-Z mip whose texel size covers the
     rect (so the test reads ~1-4 texels); the object is OCCLUDED iff its nearest depth is FARTHER than the
     Hi-Z MAX depth across the covered texels (i.e. everything in that region is in front of the object's
     nearest point). CONSERVATIVE: any uncertainty (rect spans more texels than the mip, partially
     on-screen, near-plane straddle) → NOT occluded (keep). Document the mip-selection + the conservative
     fallbacks. Unit-tested.

2. **Occlusion test in the cull compute (extend `shaders/gpudriven_cull.comp.hlsl` or a new
   `hiz_cull.comp.hlsl`).** After the existing frustum test, an object that passed frustum is ALSO tested
   with the Hi-Z occlusion test (same math as `hiz::IsOccluded`, sampling the Hi-Z texture mips); occluded
   objects are dropped from the compacted survivor list. The Hi-Z texture is built by a sequence of
   downsample passes (mip i+1 = max of mip i's 2x2) from the depth pre-pass — OR built on the CPU and
   uploaded (simpler + deterministic; pick + document, but the GPU-downsample is the "real" path — if the
   engine's mip/compute plumbing supports it use it, else CPU-build the Hi-Z and upload). Deterministic.

3. **Showcase `--hiz-cull-shot <out>` (Vulkan) / `--hiz-cull` (Metal).** A scene with a BIG OCCLUDER (a
   large wall/box near the camera) and many objects BEHIND it (fully hidden → occluded) plus objects
   beside/in-front (visible). Pipeline: depth pre-pass (all objects) → build Hi-Z → cull (frustum +
   occlusion) → render the survivors (+ the occluder + the visible objects), lit + shadowed, fixed camera.
   Print `hiz-cull: {total:N, frustumKept:K, occluded:O, drawn:D, cpuOccluded:O}` where `O > 0` (real
   occlusion happened) AND the GPU `occluded` == the CPU `hiz` reference `cpuOccluded` (assert; fail on
   mismatch) AND the rendered image is BYTE-IDENTICAL to the SAME scene with occlusion culling DISABLED
   (frustum-only) — because the occluded objects were fully hidden, they contribute zero pixels (INTERNALLY
   render the frustum-only version + assert byte-identical; fail loudly on any diff = a false-cull bug).
   New golden `tests/golden/metal/hiz_cull.png` (Metal two runs DIFF 0.0000). Existing 55 image goldens
   UNTOUCHED.

4. **Determinism.** Fixed scene/camera, deterministic Hi-Z build (pure max-reduction), conservative test
   (no RNG). Two runs byte-identical.

5. **Tests `tests/hiz_test.cpp` (pure CPU, no GPU):**
   - **Hi-Z build:** each coarser mip texel == the MAX of its 2x2 children; mip dimensions halve; a known
     depth buffer produces the expected pyramid.
   - **Occlusion test — true positive:** an object entirely BEHIND a closer full-screen occluder (its
     nearest depth farther than the Hi-Z everywhere it covers) → `IsOccluded == true`.
   - **Occlusion test — true negatives (never false-cull):** an object IN FRONT of the occluder, an object
     PARTIALLY visible (nearer than the Hi-Z somewhere it covers), an object straddling the near plane, an
     object partly off-screen → `IsOccluded == false`. (These are the safety cases — false-culling any of
     them would corrupt the image.)
   - **Conservative reference parity:** over a set of random AABBs + a known depth field, `IsOccluded`
     never reports occluded for an AABB that has ANY texel where its nearest depth is in front of the Hi-Z
     (a brute-force per-texel check) — i.e. 0 false-culls.
   - **Determinism:** same inputs → same result.
   - Clean under `windows-msvc-asan`.

## RHI seam additions (summary)
- Possibly a Hi-Z build (mip-downsample) helper IF the GPU path is used — additive pure-interface in
  `rhi.h` with the impl inside the backend dirs; OR build the Hi-Z on the CPU + upload via the existing
  texture path (no new seam). PREFER the no-new-seam CPU-build-and-upload (or reuse any existing mip/
  compute infra) — document the choice. New non-backend files (`engine/render/hiz.h`,
  `shaders/hiz_cull.comp.hlsl`, `tests/hiz_test.cpp`) add ZERO above-seam backend code symbols. Seam grep
  stays at baseline (2).

## Out of scope (YAGNI)
Two-phase occlusion culling (last-frame Hi-Z + re-test), temporal Hi-Z reuse, GPU mip-downsample if the
CPU build is simpler (note it), occlusion of shadow-pass geometry, per-cluster/meshlet occlusion, software
rasterized occluders. One single-frame depth-prepass → Hi-Z → conservative occlusion cull (+ the existing
frustum cull), byte-identical to frustum-only, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 55) + new `hiz_test` (Hi-Z build, true-positive
   occlusion, true-NEGATIVE safety cases, conservative no-false-cull parity, determinism). Clean under
   `windows-msvc-asan`.
2. **Occlusion proof (render-invariance + count):** `--hiz-cull-shot` on Vulkan renders with frustum +
   Hi-Z occlusion culling; the captured image is BYTE-IDENTICAL (SHA) to the SAME scene with occlusion
   culling DISABLED (frustum-only); `occluded > 0` (a real occluder hid objects) and the GPU `occluded`
   count == the CPU `hiz` reference. Prints `hiz-cull: {total:N, frustumKept:K, occluded:O, drawn:D,
   cpuOccluded:O}`. Two runs identical.
3. `--hiz-cull-shot` visual review (controller): the visible scene (occluder wall + the objects in front /
   beside it) renders correctly; the hidden objects are absent (but they were hidden anyway, so the image
   looks like a normal scene). Run under the AT Vulkan-validation gate → ZERO errors.
4. Metal: `visual_test --hiz-cull` → new golden `tests/golden/metal/hiz_cull.png`; two runs DIFF 0.0000.
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `hiz_cull.png` added;
   the other 55 byte-identical.
6. Introspect JSON rebaked exactly `+hiz-occlusion-culling` + `--hiz-cull-shot`; introspect test updated; no
   other drift.
7. Seam grep clean (no new above-seam code symbols). `scripts/verify.ps1` updated to include the new
   `hiz_cull` image golden in the Mac round-trip loop.
