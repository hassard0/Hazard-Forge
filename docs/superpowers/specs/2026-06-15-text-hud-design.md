# Slice BA — Text / HUD Renderer (Phase 4 #5) — Design

> Autonomous-session spec (standing directive: bold decisions, self-approve, document every call).
> Verifiable headlessly on Windows+Vulkan AND Apple M4+Metal at golden DIFF 0.0000. Fills a foundational
> gap: the engine has NO text rendering (AX flagged "no text renderer yet"). Every game UI, debug overlay,
> and editor label needs this.

**Goal:** Render screen-space text. A deterministic procedural bitmap font (a fixed 8x8 glyph atlas baked
in code — NO external asset) is uploaded as a texture; a string is laid out into a batch of screen-space
textured, alpha-blended quads; a UI/overlay pass draws them OVER the scene. A `--hud-shot` showcase draws
a scene plus a HUD string, golden-verified on both backends. The AX game's score is then drawn with it.

## Why a procedural baked font (determinism + no asset dep)

A font atlas loaded from a file adds an asset dependency and decode nondeterminism risk. Instead BAKE a
fixed monospace **8x8 bitmap font in code** — a `static const` table of glyph bitmaps for the printable
ASCII range (0x20..0x7E), one `uint8_t[8]` (8 rows x 8 bits) per glyph. Generate the atlas texture from
this table at startup (deterministic, identical on both backends). Same table + same layout math ⇒ same
pixels ⇒ goldens match. (A TTF/SDF loader is a future slice — OUT OF SCOPE.)

## Design decisions (locked)

1. **Font + layout (engine/ui/text.{h,cpp}, pure CPU, no backend symbols).** Namespace `hf::ui`.
   - `static const uint8_t kFont8x8[95][8]` — the baked glyph bitmaps for ASCII 0x20..0x7E. Provide a
     complete, legible monospace set (a well-known public-domain 8x8 font table is fine; if reproducing
     one is bulky, define the digits 0-9, A-Z, a-z, space, colon, and common punctuation at minimum and
     render unknown glyphs as a blank — document the covered set; the showcase strings must use only
     covered glyphs).
   - `BuildFontAtlas(uint8_t* rgbaOut, int atlasW, int atlasH)` — rasterizes the glyph table into a
     single-channel-expanded-to-RGBA atlas (e.g. 16x6 glyph grid, each cell 8x8 → 128x48), deterministic.
   - `struct TextVertex { float posPx[2]; float uv[2]; };`
   - `LayoutText(const std::string& s, float originX, float originY, float pxScale, int screenW,
     int screenH, std::vector<TextVertex>& outVerts) -> int quadCount` — produces 6 verts/glyph (two
     tris) in NDC (convert pixel positions → NDC using screenW/H), advancing the cursor by the scaled
     glyph width; newline handling; UVs index the atlas cell for each char. Pure math, unit-tested.

2. **UI/overlay render pass (reuse existing infra).** The text quads are drawn AFTER the scene/post into
   the final target, alpha-blended (text alpha from the atlas; transparent background). Add a small
   `text.vert.hlsl` + `text.frag.hlsl` (the vert passes NDC pos + uv straight through; the frag samples
   the atlas, uses its coverage as alpha, outputs a configurable text color). This needs an
   ALPHA-BLENDED pipeline variant — inspect how the transparency slice / debug-line overlay sets blend
   state and reuse that pattern. The atlas is a normal sampled texture (existing texture path). HLSL→
   SPIR-V→MSL via the existing toolchain (HF_MSL_GEN conventions).

3. **Showcase `--hud-shot <out>` (Vulkan) / `--hud` (Metal).** Render the standard lit+shadowed scene,
   then overlay a deterministic HUD: e.g. top-left `"HAZARD FORGE"` and `"SCORE: 0"` and a small FPS-less
   stat line (fixed text, NO clock — deterministic). Fixed positions/scale/color. New golden
   `tests/golden/metal/hud.png` (Metal two runs DIFF 0.0000). Existing 30 image goldens UNTOUCHED.

4. **Wire into the AX game (small, optional-but-nice).** `--game-shot` draws `"SCORE: N"` (N from the
   deterministic GameState at the capture step) as a HUD overlay. NOTE: this CHANGES the `game.png`
   golden (text now overlaid) → it must be RE-BAKED deliberately (a legitimate visual change, not a
   regression). If re-baking game.png, do it on the M4 like any golden and call it out explicitly in the
   VERIFY report; the controller will visually re-confirm the game scene + score text. IF you prefer to
   keep game.png frozen this slice, instead add the score HUD only behind a NEW `--game-hud-shot` with its
   own golden and leave game.png untouched — PICK ONE and document it. (Default recommendation: add
   `--game-hud-shot` + new golden `game_hud.png`, leave `game.png` byte-identical — simpler invariance.)

5. **Tests `tests/text_test.cpp` (pure CPU, no GPU):** atlas build is deterministic (same bytes each
   run); `LayoutText` produces 6 verts/glyph, correct quad count for a known string (incl. spaces +
   newline), cursor advance matches pxScale, UVs map to the right atlas cells for sample chars, NDC
   conversion correct at known screen coords; empty string → 0 quads. Clean under `windows-msvc-asan`.

## RHI seam additions (summary)
- **None** beyond reusing the existing alpha-blend pipeline + texture + fullscreen-overlay paths. If an
  alpha-blended UI pipeline variant needs a new pure-interface flag, make it additive in `rhi.h` with
  backend impls inside the backend dirs. New files (`engine/ui/text.{h,cpp}`, `shaders/text.*.hlsl`,
  `tests/text_test.cpp`) add ZERO backend code symbols. Seam grep stays at baseline (2).

## Out of scope (YAGNI)
TTF/SDF fonts, kerning/proportional fonts, Unicode beyond ASCII, rich text/markup, text wrapping
layout engine, an immediate-mode GUI, input-driven UI widgets, scrolling, a font asset pipeline. One
baked 8x8 monospace font, screen-space alpha quads, a fixed HUD string, golden-verified.

## Verification gate
1. `ctest --preset windows-msvc-debug` green (existing 30) + new `text_test` (atlas determinism + layout
   math). Clean under `windows-msvc-asan`.
2. Build compiles `text.vert.hlsl`/`text.frag.hlsl` through the normal pipeline.
3. `--hud-shot` on Windows/Vulkan: controller visual review — the HUD text is legible and correctly
   positioned over the scene. Run under the AT Vulkan-validation gate → ZERO errors (the alpha-blend UI
   pass must be sync-correct).
4. Metal: `visual_test --hud` → new golden `tests/golden/metal/hud.png`; two runs DIFF 0.0000.
5. **Render-invariance:** `git diff master --stat -- tests/golden/metal` shows ONLY the new HUD golden(s)
   added (`hud.png`, and `game_hud.png` if you took the default `--game-hud-shot` route) — `game.png` and
   the other existing goldens byte-identical. (If you instead re-baked `game.png`, it appears as MODIFIED
   and you MUST call that out + the controller re-confirms it visually.)
6. Introspect JSON rebaked exactly `+hud-text` (features) + `--hud-shot` (showcases) [+ `--game-hud-shot`
   if taken]; introspect test updated; no other drift.
7. Seam grep clean (no new code symbols). `scripts/verify.ps1` updated to include the new HUD golden(s)
   in the Mac round-trip loop.
