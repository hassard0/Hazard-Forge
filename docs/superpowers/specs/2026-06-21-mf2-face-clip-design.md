# Slice MF2 — Hull Narrowphase Hardening: SUTHERLAND–HODGMAN FACE CLIP → THE MULTI-POINT MANIFOLD — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The SECOND slice of
> FLAGSHIP #25 (DETERMINISTIC HULL NARROWPHASE HARDENING, `hf::sim::manifold`). MF1 built the per-hull POLYGON
> FACE topology (`FxHullFaces`/`BuildCanonicalFaces`/`FaceNormalWorld`/`SupportFace`/`IncidentFace`). MF2 uses it
> to GENERATE the multi-point contact manifold that fixes gjk's documented single-point teeter: given two
> overlapping hulls + the EPA contact normal, pick the REFERENCE face (most parallel to the normal) and the
> INCIDENT face (most anti-parallel on the other hull), CLIP the incident polygon against the reference face's
> side planes by deterministic Sutherland–Hodgman, and keep the clipped points below the reference plane as the
> 1-4 contact points. This is the EXACT idiom `convex::BuildManifold` (convex.h:352-399) already runs for BOXES —
> MF2 GENERALIZES it from box quads (2 axes → 4 fixed side planes) to arbitrary convex polygon faces (MF1's
> `FxHullFaces` → one side plane per face edge). The output is a `convex::ContactManifold` VERBATIM (the frozen
> solver `convex::SolveManifoldImpulse` already loops `0..count`, convex.h:662 — so a count-4 manifold needs ZERO
> solver change). MF2 generates the manifold ON THE CPU and proves it is a deterministic PURE FUNCTION (two runs
> byte-equal); the GPU bit-identity is the SEPARATE MF3 slice. APPEND to `engine/sim/manifold.h` (MF1 +
> gjk/broad/ccd/convex/fpx/etc BYTE-FROZEN). Branch: `slice-mf2`. See [[hazard-forge-manifold-roadmap]],
> [[hazard-forge-gjk-roadmap]], [[hazard-forge-docs-style]], [[hazard-forge-metal-showcase-gate]].

**Goal:** Extend `engine/sim/manifold.h` (additive — MF1 byte-unchanged) with `ClipFaceAgainstFace(refHull,
refBody, refFaces, refFace, incHull, incBody, incFaces, incFace)` → up to `kMaxManifoldPts` (4) clipped contact
points + per-point penetration depths, by deterministic Sutherland–Hodgman of the incident polygon against the
reference face's side planes (FIXED edge order, strict-integer tie-breaks) + `HullManifoldFromEpa(hullA, bodyA,
hullB, bodyB, epa)` → a `convex::ContactManifold` (count 1-4) that selects ref/incident faces from the EPA normal
and clips. Add the showcase `--mf2-clip-shot <out>` (Vulkan) / `--mf2-clip` (Metal) — both build a deterministic
box-resting-flat-on-box scene (+ a tetra-on-face + an edge-on-face contact), run `HullManifoldFromEpa`, render the
hulls LIT 3D with the contact points marked (the multi-point manifold made VISIBLE). Bake the float golden
`mf2_clip`. **NO new shader, NO new RHI** (render reuses the instanced-lit pipeline; MF2 is pure-CPU manifold
generation — no GPU compute, no TDR/VUID risk).

## Design call: generalize the frozen box-clip to polygon faces; CPU-generate, prove purity (MF3 proves the GPU)

`convex::BuildManifold` (convex.h:352-399) is the canonical deterministic SAT face-clip — for boxes. Its whole
discipline is exactly what MF2 needs and reuses:
- **reference / incident faces** (convex.h:357-362): MF2 picks the reference face = `SupportFace(hullRef, dir=n)`
  (MF1 — most parallel to the EPA normal `n`) and the incident face = `IncidentFace(hullInc, refNormal=n)` (MF1 —
  most anti-parallel). Which hull is "reference": the one whose `SupportFace(n)` normal is MOST parallel to `n`
  (i.e. the better-aligned face owner), tie-break A. `n` is SIGNED from A→B (the EPA / `convex::SatResult.axis`
  convention, convex.h:289).
- **the clip** (convex.h:363-366): Sutherland–Hodgman the incident polygon against the reference face's SIDE
  planes. A box ref face has 4 side planes from its 2 axes; a GENERAL polygon face has one side plane PER EDGE:
  for ref-face edge `(v[k] → v[k+1])` the inward side-plane normal is `FxCross(faceNormal, edgeDir)` (oriented
  toward the face interior), the plane passes through `v[k]`. Clip in FIXED edge order `k = 0..vertCount-1`; the
  inside test is a STRICT integer sign (`FxDot(sideNormal, p - v[k]) >= 0` = inside — on-plane counts as inside,
  NO tolerance band); crossing edges emit `fxdiv` intersection points in a PINNED iteration order. THE CRUX (see
  below) is that this tie-break is bit-deterministic.
- **keep + reduce** (convex.h:366-369): keep clipped vertices with depth `d = FxDot(n, refFaceCenter - vertex) >=
  0` (below/inside the ref face); the contact point is the vertex itself (the documented convex.h:282-284
  simplification — NOT projected); reduce to ≤4 by ALWAYS keeping the DEEPEST (max `d`, tie → lowest clip-order
  index) then up to 3 more in clip order. `normal = n`. **Inherit the convex.h:280-284 honest simplification
  VERBATIM** (deepest + first-3-in-clip-order, not area-maximizing — the documented deferred refinement).
- **edge-edge fallback:** if either selected face is degenerate for the contact (the EPA normal is an edge axis,
  or the clip yields 0 points), fall back to the single closest-point contact (`count=1`) — reuse
  `convex::ClosestPointsOnSegments` (convex.h:307) over the two nearest edges, OR simply emit the EPA witness
  midpoint as the 1-point manifold (the gjk::HullContact behavior, the safe deterministic floor). count never 0
  for an overlapping pair.

`HullManifoldFromEpa(hullA, bodyA, hullB, bodyB, epa)` runs `gjk::Gjk` → `gjk::Epa` is ALREADY done by the
caller (the showcase/test calls `gjk::Epa` for the seed `normal`); MF2 takes the `gjk::EpaResult` and does the
ref/incident selection + clip. Output: `convex::ContactManifold`.

> NOTE: MF2 generates the manifold ON THE CPU (a pure integer function) and RENDERS the result — NO GPU compute
> dispatch, so NO TDR/VUID risk from this slice. The MF3 slice lifts `HullContactMulti` onto the GPU and proves
> GPU==CPU bit-identity; that is WHY MF2 (two-run CPU purity) and MF3 (cross-backend identity) are separate
> slices — the multi-point manifold's variable-length point set is the flagship's hardest determinism surface and
> earns its own proof beat.

## Reuse map (file:line)
- **MF1 `engine/sim/manifold.h` (APPEND after `FacesToRenderInstances`):** `FxHullFaces`, `BuildCanonicalFaces`,
  `FaceNormalWorld`, `FaceCentroidWorld`, `SupportFace`, `IncidentFace`, `kMaxFaceVerts`. MF1 byte-frozen.
- **convex.h (read-only — the idiom to GENERALIZE, do NOT edit):** `convex::BuildManifold` (convex.h:352-399 —
  the box face-clip whose ref/incident/clip/keep/reduce logic MF2 mirrors for polygon faces), the documented
  honest simplification (convex.h:280-284), `convex::ContactManifold` (convex.h:292 — the OUTPUT type, reused
  verbatim), `convex::ClosestPointsOnSegments` (convex.h:307 — the edge-edge fallback), `convex::FxDot`/`FxCross`/
  `FxAt`, `convex::SolveManifoldImpulse` (convex.h:651 — the CONSUMER, loops `0..count`, UNCHANGED).
- **gjk.h (read-only):** `gjk::Gjk` (gjk.h:457), `gjk::Epa` (gjk.h:743) + `gjk::EpaResult` (the seed contact
  normal + witnesses), `gjk::HullContact` (gjk.h:1155 — the single-point floor / fallback reference),
  `gjk::HullWorld`/`FxHull`/the canonical builders, `gjk::HullToRenderInstances` (the render base).
- **The showcase precedent:** MF1 `--mf1-faces` (the four-hull lit render + the proofs) and CD4/GJ4 `--ccd-bullet`/
  hull-step showcases (a contact scene rendered with the contact made visible). Mirror for `--mf2-clip`: render
  the resting-box scene + mark the manifold points.
- **Registration:** `scripts/verify.ps1` (append `mf2_clip` to the Mac golden loop + `--mf2-clip-shot` to
  `$vkShots`), `engine/editor/introspect.cpp` + `tests/introspect_test.cpp` (**controller rebakes the JSON
  golden — do NOT touch it**), append to `tests/manifold_test.cpp`. NO shader → nothing for `hf_gen_msl`.

## Design decisions (locked)
1. **APPEND to `engine/sim/manifold.h`** (MF1 byte-frozen): `ClipFaceAgainstFace`, `HullManifoldFromEpa`, and any
   small clip-buffer helper (a fixed-size scratch polygon, `kMaxClipVerts` ≥ 8 — a quad clipped by 4 planes stays
   ≤ 8). **gjk.h/convex.h/ALL other sim headers BYTE-UNCHANGED.** `kMaxManifoldPts = 4` (== `convex::ContactManifold`'s
   4). Output is `convex::ContactManifold` (no new manifold struct).
2. **Showcase `--mf2-clip-shot <out>` (Vulkan) AND `--mf2-clip` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--mf2-clip` BEFORE reporting DONE — the recurring omitted-Metal-showcase failure).**
   BOTH build the SAME deterministic scene (a box resting FLAT on a static box → the count-4 face manifold; plus
   a tetra-on-face → count-3 and an edge-on-face → count-2 contact), run `gjk::Gjk`+`gjk::Epa`+`HullManifoldFromEpa`
   per pair, and render the hulls LIT 3D with each manifold CONTACT POINT marked (a small marker/instance at each
   `m.points[0..count]`) so the multi-point manifold is VISIBLE. Golden = `tests/golden/metal/mf2_clip.png`
   (Mac-baked by the CONTROLLER — DO NOT commit).
3. **PROOFS (fail loudly; exact stdout lines, asserted in the showcase AND the test):**
   - **(1) the teeter fix:** `mf2-clip: {boxOnBox:4, tetraOnFace:3, edgeOnFace:2} OK` — `HullManifoldFromEpa` on
     the flat box-on-box returns `count==4`; the tetra-face-down-on-box returns `count==3`; the edge-on-face
     returns `count==2`. Assert each (the single-point `gjk::HullContact` would give 1 for all three).
   - **(2) points valid + below the ref face:** `mf2-clip depths: all points below ref face (minDepth:<v> >= 0)`
     — every kept point's depth `d >= 0`; and every point lies within the reference face's clipped region
     (assert the min depth ≥ 0 and count ≤ 4).
   - **(3) determinism / purity:** `mf2-clip: {pointSum:<S>} two-run BYTE-EQUAL` — `HullManifoldFromEpa` over the
     fixed battery is byte-equal across two runs (a `memcmp`/sum of the `ContactManifold` POD — THE crux guard).
   - **(4) consistency:** `mf2-clip normal: == EPA normal (dot:<v> > 0)` — the manifold normal agrees with the
     EPA seed normal (FxDot > 0; the contact normal is the reference-face normal, signed A→B).
   - **Golden discipline: ONLY `tests/golden/metal/mf2_clip.png`; do NOT commit it.** Existing 222 goldens
     UNTOUCHED (MF2 adds no compute, edits no frozen file).
4. **Cross-backend bar.** The NUMERIC proofs are pure integer → strict and backend-independent (a `memcmp` of the
   manifold). The golden IMAGE is the float render: committed = Mac-Metal bake; verify.ps1 re-renders on the Mac +
   compares at `0.0000` (same-backend determinism — the gate). The CONTROLLER measures Windows-Vulkan vs
   Mac-Metal cross-vendor visresolve as a DIAGNOSTIC (in-band ~20-55, the render lineage), NOT strict-zero.
5. **Tests — APPEND to `tests/manifold_test.cpp` (pure CPU):** `HullManifoldFromEpa` counts {4,3,2} for the three
   canonical contacts; every point depth ≥ 0; count ≤ 4; the normal agrees with the EPA normal; two-run
   byte-equality of the manifold over the battery; a degenerate/edge case falls back to count==1 (never 0 for an
   overlapping pair). Clean under `windows-msvc-asan` (separate build + test).
6. **Introspect.** Add EXACTLY `manifold-face-clip` (features) + `--mf2-clip-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None — and NO new shader.** MF2 is pure-CPU manifold generation; the render reuses the existing instanced-lit
  pipeline (+ the existing marker/instance draw for the contact points — the debug-marker idiom, NO new pipeline).
  `engine/sim/manifold.h` APPEND-only (MF1 frozen); gjk.h/convex.h/broad.h/ccd.h/fpx.h + ALL other sim headers +
  ALL existing shaders UNCHANGED. Report the seam: manifold.h APPEND-only; NO rhi.h change, NO shader change, NO
  frozen-file edit.

## Out of scope (YAGNI — later slices)
The manifold GPU shader + GPU==CPU bit-identity (MF3). The full inertia tensor + the hardened restacked step
(MF4) — MF2 GENERATES the manifold but does NOT yet feed it into a stepped world (the frozen `gjk::StepHullWorld`
still uses the single-point `HullContact`; MF4 wires the hardened step). Lockstep (MF5), the settled-stack
capstone (MF6). The AREA-MAXIMIZING 4-point reduction (MF2 inherits the convex.h:280-284 deepest+clip-order cut —
the documented deferred refinement). A general quickhull face builder (canonical hulls only, via MF1). MF2 claims
ONLY: given two overlapping canonical hulls + the EPA normal, a deterministic 1-4 point contact manifold (the
flat-face case = 4 points, the teeter fixed), generated as a pure integer function (two runs byte-equal), with the
float render golden + the four proofs.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "manifold|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test — never chain `--build && ctest`).
2. **proofs + visual:** `--mf2-clip-shot` on Vulkan: the 4 proof lines + exit 0 under the conan validation layer
   → ZERO VUID. VERIFY the image shows the box resting FLAT on the box with FOUR contact markers at the face
   corners (the multi-point manifold), no garbage/NaN/iridescence.
3. Metal: `visual_test --mf2-clip` → `tests/golden/metal/mf2_clip.png`; two runs DIFF 0.0000. **Confirm
   `--mf2-clip` is wired in `visual_test.mm` (grep it) BEFORE the Mac bake** — NO shader added. Cross-vendor =
   FLOAT visresolve in-band (~20-55).
4. **Render-invariance:** ONLY `mf2_clip.png` added; the other 222 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+manifold-face-clip` + `--mf2-clip-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + MF1 manifold.h code + gjk.h/convex.h/broad.h/ccd.h/fpx.h + ALL other sim headers +
   ALL existing shaders byte-unchanged; manifold.h APPEND-only; NO shader change). `mf2_clip` in the Mac loop +
   `--mf2-clip-shot` in `$vkShots`.
