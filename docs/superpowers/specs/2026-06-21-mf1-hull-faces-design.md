# Slice MF1 — Hull Narrowphase Hardening: HULL FACE TOPOLOGY (the new primitive) — Design

> Autonomous-session spec. Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal. The FIRST slice of
> FLAGSHIP #25 (DETERMINISTIC HULL NARROWPHASE HARDENING, `hf::sim::manifold`). This flagship closes the two
> documented limits of the frozen GJK/EPA hull narrowphase: (1) `gjk::HullContact` (gjk.h:1155) hardcodes a
> SINGLE-POINT manifold (`m.count = 1u`, gjk.h:1161) → a face-resting hull TEETERS on one point; (2)
> `gjk::FxHullInvInertiaBody` (gjk.h:1127) is an AABB-DIAGONAL inertia approximation → wrong rotational dynamics
> for tetra/octa/wedge. The whole flagship is STRICTLY ADDITIVE: `gjk.h` is NEVER edited — `manifold.h` appends
> hardened ALTERNATIVES (`HullContactMulti`, `FxHullInertiaBodyFull`) and a sibling step `StepHullWorldHardened`,
> leaving the frozen `gjk::StepHullWorld`/`broad::StepHullWorldBP` (and the 221 existing goldens) untouched.
> MF1 builds the missing PRIMITIVE both of those need: per-hull POLYGON FACE topology. The manifold clip (MF2),
> the GPU manifold (MF3), and the full inertia + restacked-stability tick (MF4) all consume MF1's faces.
> APPEND to a NEW sibling header `engine/sim/manifold.h` (`#include "sim/ccd.h"` — gjk/broad/convex/fpx/etc all
> BYTE-FROZEN). Branch: `slice-mf1`. See [[hazard-forge-manifold-roadmap]], [[hazard-forge-gjk-roadmap]],
> [[hazard-forge-docs-style]], [[hazard-forge-metal-showcase-gate]].

**Goal:** Create `engine/sim/manifold.h` (namespace `hf::sim::manifold`) with the hull FACE primitive:
`FxHullFaces` (a fixed-size, std430-packable polygon-face table) + `BuildCanonicalFaces(const gjk::FxHull&)`
(the deterministic polygon faces of the four canonical hulls, by MERGING the coplanar triangles that
`gjk::HullTriIndices` (gjk.h:1502) already emits into polygon faces) + `FaceNormalWorld` / `FaceCentroidWorld` +
`SupportFace(hull, faces, body, dir)` (the face whose world normal is most parallel to `dir` — the reference-face
selector) + `IncidentFace(hull, faces, body, refNormal)` (most anti-parallel — the incident-face selector). Add
the showcase `--mf1-faces-shot <out>` (Vulkan) / `--mf1-faces` (Metal) — both render the four canonical hulls with
each FACE flat-tinted a distinct color (the face decomposition made VISIBLE), via a render-only float helper
`FacesToRenderInstances`. Bake the float golden `mf1_faces`. **NO new shader, NO new RHI** (render reuses the
existing instanced-lit pipeline). `gjk.h` and all other headers UNCHANGED; `manifold.h` is a brand-new sibling.

## Design call: faces are the missing primitive; derive them from the proven render triangulation

The frozen narrowphase is vertices-only (`gjk::FxHull` is `verts[]` + `count`; the comment at gjk.h:53 notes
"Faces are NOT needed for GJ1 — faces arrive in a later slice"). That later slice is MF1. The canonical-hull
FACE sets are ALREADY encoded — as triangle index lists with documented per-face groupings — in
`gjk::HullTriIndices` (gjk.h:1502-1538). MF1 promotes those to first-class POLYGON faces:
- **tetra (count 4):** 4 triangular faces `{0,1,2} {0,1,3} {0,2,3} {1,2,3}` (gjk.h:1507).
- **box (count 8):** 6 QUAD faces — the six quads at gjk.h:1510-1512: `-x{0,1,3,2} +x{4,5,7,6} -y{0,1,5,4}
  +y{2,3,7,6} -z{0,2,6,4} +z{1,3,7,5}`.
- **count 6:** OCTA vs WEDGE disambiguated by the SAME structural test gjk.h:1517-1521 uses (an octa vertex is an
  axis pole → exactly two zero coordinates; a wedge vertex is not) — a PURE function of the verts. **octa:** 8
  triangular faces (gjk.h:1525-1526). **wedge:** 5 faces = 2 triangular caps `{0,2,4} {1,5,3}` + 3 quad sides
  `{0,1,3,2} {0,4,5,1} {2,3,5,4}` (gjk.h:1530-1534).
- Any other count → 0 faces (the canonical hulls only — the documented YAGNI, gjk.h:1537; a general quickhull
  face builder is OUT OF SCOPE).

**Each face's outward normal** is the `FxCross` of two of its edges, oriented OUTWARD by flipping it to disagree
with `(faceCentroid − hullCentroid)` (the integer idiom `gjk::EpaAddFace` uses, gjk.h:727, and that
`HullToRenderMesh` uses render-side, gjk.h:1562). So the source winding in the face table does NOT matter — the
normal is canonicalized outward, exactly as the render soup already does. `SupportFace`/`IncidentFace` then rank
faces by `FxDot(worldNormal, dir)` in FIXED index order with a STRICT-greater lowest-index tie-break (the
`gjk::SupportLocal` rule, gjk.h:85) — deterministic, pure integer (the int64 `FxDot`/`FxCross`).

> NOTE: MF1 is a PRIMITIVE slice — geometry-table construction + face selection, all on the CPU. There is NO GPU
> compute dispatch (so no TDR/VUID risk from this slice); the only GPU work is the normal lit RENDER of the
> showcase (a draw, not a heavy compute dispatch). The numeric proofs are pure-CPU 0px-by-construction; the golden
> IMAGE is the one float crossing (render-only), in-band visresolve like every render capstone.

## Reuse map (file:line)
- **gjk.h (read-only — REUSE verbatim, do NOT edit):** `gjk::FxHull` (gjk.h:55), `gjk::HullTriIndices`
  (gjk.h:1502 — the canonical face groupings to promote), the octa/wedge structural test (gjk.h:1517-1521),
  `gjk::SupportLocal`'s strict-greater lowest-index tie-break (gjk.h:85), `gjk::HullToRenderInstances`/
  `HullToRenderMesh` (gjk.h:1558+, the render-bridge precedent + the outward-normal-against-centroid idiom),
  `gjk::HullWorld`, `gjk::FromInt`/the canonical builders (`MakeBox`/`MakeOcta`/`MakeTetra`/`MakeWedge`).
- **convex.h (read-only):** `convex::FxDot` (the Q16.16 int64 dot), `convex::FxCross` (the int64 cross for the
  face normal), `convex::FxVec3`/`FxSub`/`FxAdd`.
- **The render showcase precedent:** GJ6/BP6/CD6 `--gjk-render`/`--broad-render`/`--ccd-render` (the lit
  instanced 3D draw of a `gjk::HullWorld` + the float visresolve-bar + the two-call provenance proof). MF1's
  `FacesToRenderInstances` is the same shape, but tints per FACE (a fixed palette indexed by face number) instead
  of per-hull-type — render-only float, OUTSIDE any bit-exact loop.
- **Registration:** `scripts/verify.ps1` (append `mf1_faces` to the Mac golden loop as a plain entry like
  `gjk_render` — no special threshold — + `--mf1-faces-shot` to `$vkShots`), `engine/editor/introspect.cpp` +
  `tests/introspect_test.cpp` (**controller rebakes the JSON golden — do NOT touch it**), a NEW
  `tests/manifold_test.cpp` (registered in the test CMakeLists alongside `ccd_test`). NO shader → nothing for
  `hf_gen_msl`.

## Design decisions (locked)
1. **NEW header `engine/sim/manifold.h`** (`namespace hf::sim::manifold`, `#include "sim/ccd.h"`): `FxHullFaces`,
   `BuildCanonicalFaces`, `FaceNormalWorld`, `FaceCentroidWorld`, `SupportFace`, `IncidentFace`, and the
   render-only `FacesToRenderInstances`. **gjk.h and ALL other sim headers BYTE-UNCHANGED.** `FxHullFaces` is
   std430-packable (fixed `kMaxHullFaces`×(`kMaxFaceVerts` `uint` indices + a `uint` vert count) + a `uint` face
   count) so MF2/MF3 can hand it to a shader later. Pick `kMaxHullFaces = 8` (octa) and `kMaxFaceVerts = 4`
   (box/wedge quads) — the canonical ceiling; document them like `gjk::kMaxHullVerts` (gjk.h:50).
2. **Showcase `--mf1-faces-shot <out>` (Vulkan) AND `--mf1-faces` (Metal) — WIRE BOTH (grep your own
   `visual_test.mm` for `--mf1-faces` BEFORE reporting DONE — the recurring omitted-Metal-showcase failure).**
   BOTH build a deterministic scene of the four canonical hulls (tetra/box/octa/wedge, fixed poses, e.g. in a
   row), `BuildCanonicalFaces` each, and render via `FacesToRenderInstances` LIT 3D (matte, directional light) —
   every face a distinct flat color from a fixed palette so the decomposition is unmistakable (box = 6 quads,
   octa = 8 tris, etc.). Golden = `tests/golden/metal/mf1_faces.png` (Mac-baked by the CONTROLLER — DO NOT
   commit it).
3. **PROOFS (fail loudly; exact stdout lines, asserted in the showcase AND the test):**
   - **(1) face counts:** `mf1-faces: {tetra:4, box:6, octa:8, wedge:5} OK` — `BuildCanonicalFaces` returns
     exactly these face counts; assert each.
   - **(2) outward winding:** `mf1-faces winding: all faces outward (minDot:<v> > 0)` — for every face of every
     canonical hull, `FxDot(faceNormalWorld, faceCentroidWorld − hullCentroidWorld) > 0`; assert the min > 0.
   - **(3) selection purity + correctness:** `mf1-faces support: {sum:<S>} two-run BYTE-EQUAL` — `SupportFace`/
     `IncidentFace` over a FIXED (hull, dir) battery are byte-equal across two runs (a `memcmp`/sum), AND for a
     box `SupportFace(+X)` returns the +X face and `IncidentFace` of `+X`'s normal returns the −X face (assert
     the opposing-face relation).
   - **(4) render provenance:** `mf1-faces render: {hulls:4, tris:<T>} two-call BYTE-EQUAL` —
     `FacesToRenderInstances` is a pure function (two calls byte-equal).
   - **Golden discipline: ONLY `tests/golden/metal/mf1_faces.png`; do NOT commit it.** Existing 221 goldens
     UNTOUCHED (MF1 adds no compute, edits no frozen file → they are byte-reproducible by construction).
4. **Cross-backend bar.** The NUMERIC proofs (1)-(4) are pure integer / pure-function → strict and backend-
   independent. The golden IMAGE is the float render: the COMMITTED golden is the Mac-Metal bake; verify.ps1
   re-renders on the Mac + compares vs it at `0.0000` (same-backend determinism — the gate). The CONTROLLER
   measures Windows-Vulkan vs Mac-Metal cross-vendor visresolve as a DIAGNOSTIC — a float render is in-band
   (~20-55, the GJ6/BP6/CD6 lineage), NOT strict-zero cross-vendor.
5. **Tests — NEW `tests/manifold_test.cpp` (pure CPU):** `BuildCanonicalFaces` face counts {4,6,8,5}; every face
   outward-wound; every face's vertex indices are valid (`< hull.count`) and within the per-face `vertCount`;
   `SupportFace(+X)` / `IncidentFace` opposing-face relation for a box; two-run byte-equality of the
   support battery and of `FacesToRenderInstances`. Clean under `windows-msvc-asan` (separate build + test).
6. **Introspect.** Add EXACTLY `manifold-hull-faces` (features) + `--mf1-faces-shot` (showcases) + update
   `tests/introspect_test.cpp`. **Do NOT rebake the JSON golden — the controller does.**

## RHI seam additions (summary)
- **None — and NO new shader.** The render reuses the existing instanced-lit pipeline (the GJ6/BP6/CD6 idiom);
  MF1 adds no compute dispatch at all. `engine/sim/manifold.h` is a brand-new APPEND-only sibling header;
  `gjk.h`/`broad.h`/`ccd.h`/`convex.h`/`fpx.h` + ALL other sim headers + ALL existing shaders UNCHANGED. Report
  the seam: NEW header `manifold.h` only; NO rhi.h change, NO shader change, NO frozen-file edit.

## Out of scope (YAGNI — later slices / flagships)
The multi-point manifold CLIP (MF2), the manifold GPU shader (MF3), the full inertia tensor + hardened step
(MF4), lockstep (MF5), the capstone render of a SETTLED hardened stack (MF6). A general quickhull face builder
for arbitrary point sets (MF1 covers the four canonical hulls only, mirroring `gjk::HullTriIndices`' documented
YAGNI). Hull JOINTS (flagship #26). MF1 claims ONLY: each canonical hull has a deterministic, outward-wound
POLYGON face table, and the reference/incident face selectors are pure deterministic integer functions — with
the float face-colored render golden + the four proofs.

## Verification gate (controller)
1. `ctest --preset windows-msvc-debug -R "manifold|introspect"` green. Clean under `windows-msvc-asan` (SEPARATE
   build + test — never chain `--build && ctest`).
2. **proofs + visual:** `--mf1-faces-shot` on Vulkan: the 4 proof lines + exit 0 under the conan validation layer
   → ZERO VUID. VERIFY the image shows the four canonical hulls with each FACE distinctly flat-colored under
   directional light (box = 6 quad faces, octa = 8 tri faces, tetra = 4, wedge = 5), no garbage/NaN/iridescence.
3. Metal: `visual_test --mf1-faces` → `tests/golden/metal/mf1_faces.png`; two runs DIFF 0.0000. **Confirm
   `--mf1-faces` is wired in `visual_test.mm` (grep it) BEFORE the Mac bake** — NO shader added. Cross-vendor =
   FLOAT visresolve in-band (~20-55).
4. **Render-invariance:** ONLY `mf1_faces.png` added; the other 221 byte-identical (+ controller introspect
   rebake).
5. Introspect: exactly `+manifold-hull-faces` + `--mf1-faces-shot`; `tests/introspect_test.cpp` updated.
6. Seam grep clean (`rhi.h` + `gjk.h`/`broad.h`/`ccd.h`/`convex.h`/`fpx.h` + ALL other sim headers + ALL existing
   shaders byte-unchanged; `manifold.h` is a NEW sibling, `#include "sim/ccd.h"`; NO shader change).
   `mf1_faces` in the Mac loop + `--mf1-faces-shot` in `$vkShots`.
