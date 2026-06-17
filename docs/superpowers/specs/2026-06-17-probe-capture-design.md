# Slice DI — DDGI Slice 2: Probe Radiance Capture (Phase 5 #2) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. The second slice of the
> DDGI GI flagship arc (after DH probe ray-trace): each probe CAPTURES the scene's lit radiance into a cubemap
> (the data the SH-encode slice will convolve into irradiance). Reuses DD's cubemap render targets + cubemap.h
> + DH's probe grid. Deterministic, with a capture==direct-render byte-identical proof + a probeCount=0 no-op.

**Goal:** For a (small) grid of probes, render the lit scene into the 6 faces of a cubemap from each probe
center (one probe at a time, reusing the single DD `ICubemapTarget`), reading each probe's captured faces back
into a host/SSBO radiance store. A `--probecapture-shot` showcase visualizes the captured probe cubemap(s) and
proves a captured face is BYTE-IDENTICAL to the scene rendered directly with that face's view/projection (the
DD capture-correctness pattern), proves the probeCount=0 skip-loop no-op, and writes a deterministic golden.
No SH/irradiance/visible-GI yet (those are slices DJ/DK/DL) — this slice is the verified radiance-capture
layer. To keep render counts tractable the capture grid is SMALL (see §1) — it is a separate, smaller grid
from DH's 256-probe ray-trace grid (the probe_gi.h `ProbeGrid` is parameterized; DH's golden is untouched).

## Reuse map (builds on proven pieces)

- **DD (`engine/render/cubemap.h` `FaceView`/`FaceProj`/`FaceViewProj` + the cubemap-RT RHI: `ICubemapTarget`,
  `CreateCubemapTarget`, `BeginCubemapFace`/`EndCubemapFace`, `ReadCubemapFace`)** — the per-face render +
  readback. DI loops probes through the SAME single cubemap RT (no cubemap-array RHI).
- **DH (`engine/render/probe_gi.h` `ProbeGrid`/`probePos`/`probeCount`/`ProbeDispatchGroups`)** — the probe
  lattice + the probeCount=0 disabled-path idiom (here: skip the capture loop).
- **The lit scene pass** — each face renders the SAME lit scene DD's `--captureprobe-shot` renders, from the
  probe center with the per-face camera. Reuse it.

## Design decisions (locked)

1. **Capture-grid sizing + the radiance store (engine/render/probe_capture.h, header-only pure CPU, no backend
   symbols).** Namespace `hf::render::probecap`. Small math header (the heavy lifting is the showcase capture
   loop + DD's RHI). To keep render counts sane, the DI capture grid is small — e.g. `ProbeGrid{dimX=2,dimY=2,
   dimZ=2}` = **8 probes** (8×6 = 48 face renders, vs 256×6=1536 for the full grid). Document that DDGI in
   production captures incrementally (a few probes/frame); this slice captures the small grid in one shot for
   determinism.
   - `int CaptureFaceCount(const ProbeGrid& grid)` = `grid.probeCount() * 6` (the total face renders; 0 when
     probeCount==0 → the skip-loop no-op).
   - `struct ProbeRadiance { ... }` — the per-probe captured-radiance record (e.g. a fixed low-res face store
     or the SH-input sample set the next slice consumes; for THIS slice keep it the raw read-back face data
     descriptor — document the layout: face size, RGBA16F, 6 faces). The actual face pixels live in the
     read-back buffer; this header just describes the indexing (`ProbeFaceIndex(probe, face)` etc.).
   - Any small shared math (e.g. averaging a face to a single radiance for the debug-viz, or the
     probe→capture-slot index) goes here, unit-tested. Keep it minimal + deterministic.

2. **Capture loop (showcase-side, reusing DD's cubemap RT).** For each probe p in the (small) grid: for face
   f in 0..5: `BeginCubemapFace(cubeRT, f)` → render the lit scene from `probePos(p)` with
   `cubemap::FaceView(f, probeCenter)` + `cubemap::FaceProj(zNear,zFar)` → `EndCubemapFace`. After the 6 faces,
   `ReadCubemapFace(cubeRT, f, ...)` each face back into the per-probe radiance store. Loop the next probe
   through the SAME `cubeRT` (single cubemap RT, one-at-a-time — no cubemap-array RHI). Avoid recursion (the
   captured scene excludes any reflective/probe-debug surfaces; document). PREFER no new RHI (DD's cubemap-RT
   interface already supports create + per-face begin/end + readback); if a genuinely-missing capability
   appears, add an ADDITIVE pure-interface in rhi.h with backend-dir impls (document + report) — but PREFER
   none. `probeCount == 0` → the loop body never runs → the radiance store stays at its cleared value (the
   no-op).

3. **Showcase `--probecapture-shot <out>` (Vulkan) / `--probecapture` (Metal).** A scene with distinct colored
   geometry surrounding the small probe grid; capture each probe's cubemap; visualize the captured radiance —
   e.g. a cross/lat-long unwrap of probe-0's 6 faces, or the per-probe average radiance rendered as colored
   spheres at the probe positions. Fixed camera/scene/grid. Print `probe-capture: {probes:8, faces:48,
   cubeSize:N}` (deterministic). INTERNAL proofs (fail loudly):
   - **Capture-correctness (the DD pattern):** render the scene DIRECTLY with `FaceView(0, probe0Center)`/
     `FaceProj` and assert BYTE-IDENTICAL (SHA) to `ReadCubemapFace(probe0's cube, face 0)` — the capture is a
     faithful scene render. Print `probe-capture: face-0 == direct render: BYTE-IDENTICAL`.
   - **probeCount=0 no-op:** re-run with `grid.dimX=0` → the capture loop is skipped → the radiance store is
     byte-identical to its cleared value. Print `probe-capture probeCount=0: store UNTOUCHED == cleared`.
   - **Determinism:** two runs byte-identical.
   New golden `tests/golden/metal/probe_capture.png` (Metal two runs DIFF 0.0000, gate on the compare.sh EXIT
   CODE). Existing 74 image goldens UNTOUCHED.

4. **Determinism.** Fixed scene/grid/camera, deterministic per-face renders. Two runs byte-identical;
   capture==direct byte-identical; cross-backend the GOLDEN is DIFF 0.0000 (the captured-radiance viz is
   probe-anchored world data, NOT screen-space, so unlike DH's ray-viz it IS cross-backend-stable — but VERIFY:
   if any face uses a screen-space-dependent term, the viz may be a Metal-stable golden only; document the
   result honestly like DH). **Cross-backend FP note (from DH):** any host→GPU transcendental or multiply-add
   in the capture math must follow the DH discipline (host-precompute transcendentals + explicit fma) if a
   GPU==CPU bit-exact assert is used; the face renders themselves go through the shared lit pipeline (already
   cross-backend-stable, as DD proved capture==direct on both backends).

5. **Tests `tests/probe_capture_test.cpp` (pure CPU, no GPU):**
   - **CaptureFaceCount:** `= probeCount*6`; `0` at probeCount==0 (dimX/Y/Z=0).
   - **Probe→face indexing:** `ProbeFaceIndex(p,f)` round-trips / covers all probe×face slots without overlap.
   - **FaceView/FaceProj reuse:** the 6 `cubemap::FaceView(f, center)` look down the 6 axes from the probe
     center (re-assert the DD convention for the probe-centered case — at least +Y/-Y).
   - **Any radiance-store math** (averaging / slot mapping) is deterministic + correct on hand cases.
   - **Determinism:** same inputs → same result.
   - Clean under `windows-msvc-asan`.

6. **Introspect.** Add exactly `ddgi-probe-capture` (features) + `--probecapture-shot` (showcases).

## RHI seam additions (summary)
- **Prefer none** — reuses DD's cubemap-RT interface (`ICubemapTarget`/`CreateCubemapTarget`/`BeginCubemapFace`/
  `EndCubemapFace`/`ReadCubemapFace`) + the lit pass. The single cube RT is looped over probes (no cubemap-array
  RHI). New non-backend files (`engine/render/probe_capture.h`, any new viz shader, `tests/probe_capture_test.cpp`)
  add ZERO above-seam backend code symbols. If a genuinely-missing capability appears, additive pure-interface
  in rhi.h with backend-dir impls (document + report). Seam grep stays at baseline (2). Report the seam result.

## Out of scope (YAGNI)
SH encoding / irradiance convolution (slice DJ), probe relighting / neighbor gather (slice DK), the GI
composite (slice DL), per-frame incremental capture, cubemap-array RTs (loop the single cube RT), roughness
prefiltering, the full 256-probe capture (small grid here for tractable render counts; production scales it).
One small-grid per-probe radiance capture into the DD cubemap RT with a capture==direct byte-identical proof +
a probeCount=0 no-op proof + a captured-radiance-viz golden.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 74) + new `probe_capture_test` (CaptureFaceCount,
   probe→face indexing, FaceView/FaceProj reuse, radiance-store math, determinism). Clean under
   `windows-msvc-asan`.
2. **Capture-correctness + no-op proofs + visual:** `--probecapture-shot` on Vulkan: the captured-radiance viz
   is coherent (the probe cubemaps show the surrounding colored scene); `probe-capture: face-0 == direct
   render: BYTE-IDENTICAL` + `probe-capture probeCount=0: store UNTOUCHED == cleared` + two-run byte-identical;
   the `probe-capture: {...}` line is deterministic. Run under the AT Vulkan-validation gate → ZERO errors (the
   6×N face render passes + the capture→readback barriers SYNC-HAZARD-free).
3. Metal: `visual_test --probecapture` → new golden `tests/golden/metal/probe_capture.png`; two runs DIFF
   0.0000 (gate on the compare.sh EXIT CODE). The capture==direct + no-op proofs also pass on Metal.
4. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY `probe_capture.png` added;
   the other 74 byte-identical (incl. DH's `probegi.png` + DD's `capture_probe.png`).
5. Introspect JSON rebaked exactly `+ddgi-probe-capture` + `--probecapture-shot`; introspect test updated; no
   other drift.
6. Seam grep clean (report any rhi.h additive interface; PREFER none). `scripts/verify.ps1` updated to include
   the new `probe_capture` image golden in the Mac round-trip loop (gate on compare.sh EXIT CODE).
