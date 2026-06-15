# Slice BI — Material-Graph Introspection (JSON + DOT) — Phase 4 #11 — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows (deterministic text golden); pure CPU, no graphics. Completes the
> agentic material story: AUTHOR (AV/AW .mat.json) -> RENDER (AV/AZ) -> **INSPECT** (this slice).

**Goal:** Dump a material graph as machine-readable, deterministic JSON (and a Graphviz DOT string for
visualization) so an agent — or a future visual node editor — can introspect any material's structure:
its nodes (id, type, params, ports), the edges between them, and the resolved PBROutput. Mirrors the
engine-state introspection (Slice AL) but for material graphs. Golden-verified as a byte-exact text
artifact (the audio/JSON-golden pattern, no image).

## Design decisions (locked)

1. **Graph introspection (engine/material/graph_introspect.{h,cpp}, pure CPU, no backend/graphics
   symbols).** Namespace `hf::material`. Reuses the existing `Graph`/`Node`/`Edge` model + loader.
   - `std::string DescribeGraphJson(const Graph&, const std::string& name)` -> deterministic pretty JSON:
     `{ material: name, nodeCount, edgeCount, output: <PBROutput node id>, nodes: [ { id, type,
     params:{...}, inputs:[ { port, type, source: "<nodeId.outPort>" | null, default: <value if
     unconnected> } ], output:{ type } } ], edges: [ { from: "<nodeId.outPort>", to: "<nodeId.inPort>" }
     ] }`. Nodes emitted in a STABLE order (by id); ports in declaration order; deterministic number
     formatting (same convention as the engine introspect JSON / `tests/golden/introspect`). LF newlines
     (pin via the existing `.gitattributes` if needed).
   - `std::string ToDot(const Graph&)` -> a Graphviz `digraph { ... }`: one `node [label="type\n#id"]`
     per node, one edge per connection (`n<from> -> n<to> [label="outPort->inPort"]`), PBROutput
     highlighted. Deterministic (stable node/edge order).
   - Both are pure string builders over the in-memory `Graph`; no I/O beyond the caller writing the
     returned strings.

2. **Showcase / CLI `--material-introspect <mat.json> [out.json]`.** Loads the graph via the existing
   `material_loader`, prints `DescribeGraphJson` to `out.json` (or stdout), and optionally a `.dot`
   sidecar. Also a `--material-introspect-dot <mat.json> [out.dot]` OR a `--dot` flag — pick one form,
   document it. Deterministic. Print a one-line summary `mat-introspect: {material:<name>, nodes:N,
   edges:E}`.

3. **Golden.** New text-golden category `tests/golden/material/`: `showcase3_graph.json` = the byte-exact
   `DescribeGraphJson(load("assets/materials/showcase3.mat.json"))` (showcase3 is the richest existing
   graph — Swizzle/MakeFloat3/etc). Checked on Windows (byte-exact, like the introspect JSON / audio
   WAV). No image golden, no Metal round-trip needed (pure CPU, deterministic by construction — note the
   cross-platform text guarantee). Existing 36 image goldens + audio + introspect goldens UNTOUCHED.

4. **Tests `tests/graph_introspect_test.cpp` (pure CPU):**
   - **Structure:** for a hand-built small graph, the JSON reports the correct nodeCount/edgeCount, the
     PBROutput id, each node's type/params, and each edge `from`/`to` with the right port names.
   - **Unconnected defaults:** an unconnected PBROutput input is reported with its `default` value and
     `source: null`.
   - **Determinism:** `DescribeGraphJson` on the same graph twice -> byte-identical; node/port order
     stable.
   - **DOT well-formedness:** `ToDot` starts with `digraph`, contains one node line per node + one edge
     line per edge, and is deterministic.
   - **Golden parity:** `DescribeGraphJson(load(showcase3.mat.json))` byte-equals the committed
     `tests/golden/material/showcase3_graph.json` (the test reads the golden file and compares — so the
     golden is also CI-checked, not just by verify.ps1).
   - Clean under `windows-msvc-asan`.

5. **Introspect (engine-state).** Add exactly `material-graph-introspection` (features) +
   `--material-introspect` (showcases) to the ENGINE introspect (introspect.cpp + its golden). One-pattern
   rebake, no other drift.

## RHI seam additions (summary)
- **None.** Pure CPU text generation over the material graph. New files (`engine/material/graph_introspect.*`,
  `tests/graph_introspect_test.cpp`, the JSON golden) add ZERO backend symbols. Seam grep stays at
  baseline (2).

## Out of scope (YAGNI)
A live/interactive node editor UI, round-trip JSON->Graph re-serialization (the .mat.json loader already
parses authoring JSON; this is a richer INTROSPECTION dump, not the authoring format), graph diffing,
rendering the DOT to an image, introspecting compiled SPIR-V, per-node preview thumbnails. One
deterministic JSON + DOT dump, golden-checked.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 35) + new `graph_introspect_test` (structure,
   defaults, determinism, DOT well-formedness, golden parity). Clean under `windows-msvc-asan`.
2. `--material-introspect assets/materials/showcase3.mat.json out.json` produces JSON byte-equal to
   `tests/golden/material/showcase3_graph.json`; the summary line `mat-introspect: {...}` is deterministic
   (two runs identical).
3. `--material-introspect` (dot form) emits a well-formed `digraph` for showcase3.
4. **Render-invariance:** no image golden touched — `git diff master --stat -- tests/golden/metal` EMPTY;
   the only new golden is `tests/golden/material/showcase3_graph.json`.
5. Engine introspect JSON rebaked exactly `+material-graph-introspection` + `--material-introspect`;
   introspect test updated; no other drift.
6. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to byte-check the new
   `showcase3_graph.json` golden on Windows (pure-ASCII; OPTIONALLY fix the cosmetic "30 goldens"
   summary string while editing).
