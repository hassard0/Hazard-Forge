# Hazard Forge reference asset library — MANIFEST

The complete curated reference asset library for the engine's samples. Two tiers:

1. **Committed in-repo** (`assets/reference/*.glb`) — small, self-contained, **CC0** assets that
   every sample loads with zero network. See `ATTRIBUTION.md`.
2. **On-demand heavy heroes** (this manifest) — the Sponza/Bistro-class scenes and the large CC0
   PBR heroes. Fetched into `assets/reference/_downloaded/` (gitignored) by
   `fetch_reference_assets.sh` (POSIX) / `fetch_reference_assets.ps1` (Windows). Kept out of git
   so the repo stays small.

After fetching, render any of them through the engine with:

```
hello_triangle --reference-shot <path-to.glb-or.gltf> out.bmp
```

Each `--reference-shot` run prints the imported instance/mesh/material count (the load-proof) and
writes the BMP.

---

## Tier 1 — committed in-repo (CC0, auto-available)

| File | © | License | Bytes |
|------|---|---------|-------|
| `BoxVertexColors.glb` | 2023, Public | CC0 1.0 | ~1.9 KB |
| `TextureCoordinateTest.glb` | 2017, Analytical Graphics, Inc. | CC0 1.0 | ~14 KB |
| `TextureLinearInterpolationTest.glb` | 2017, Public | CC0 1.0 | ~13 KB |

---

## Tier 2 — on-demand heavy heroes (downloader, gitignored)

### A. Auto-downloadable (stable raw URLs — the fetch scripts pull these directly)

These are the large **CC0** PBR hero models from the Khronos glTF Sample Assets, each a
single self-contained `.glb`. `fetch_reference_assets.sh --heroes` downloads them.

| Asset | License | Single-file URL |
|-------|---------|-----------------|
| `Avocado.glb` (~8.1 MB) | **CC0 1.0** | https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Avocado/glTF-Binary/Avocado.glb |
| `WaterBottle.glb` (~9.0 MB) | **CC0 1.0** | https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/WaterBottle/glTF-Binary/WaterBottle.glb |
| `BoomBox.glb` (~10.6 MB) | **CC0 1.0** | https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/BoomBox/glTF-Binary/BoomBox.glb |
| `Lantern.glb` (~9.6 MB) | **CC0 1.0** | https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Lantern/glTF-Binary/Lantern.glb |
| `BarramundiFish.glb` (~12.5 MB) | **CC0 1.0** | https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/BarramundiFish/glTF-Binary/BarramundiFish.glb |
| `Corset.glb` (~13.5 MB) | **CC0 1.0** | https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Corset/glTF-Binary/Corset.glb |

### B. Sponza (the canonical hero scene)

The Khronos PBR Sponza is a multi-file `.gltf` + `.bin` + texture directory.
`fetch_reference_assets.sh --sponza` pulls the whole directory. **NOTE the license:** the Khronos
PBR Sponza geometry/textures derive from Crytek's Sponza and carry the
**CryEngine Limited License Agreement** (© 2016 Crytek) — NOT CC0. Attribution + the CryEngine
terms apply. (The original 2010 Sponza is the most-cited test scene in graphics; there is no
single universally-CC0 "Sponza".)

| Asset | License | Source (directory root) |
|-------|---------|------------------------|
| Sponza (PBR) | CryEngine Limited License (© Crytek) | https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Sponza/glTF |

### C. EULA / browser-gated heroes (manual download — landing pages only)

These require accepting a license on the vendor site (no stable raw URL); the fetch script prints
the landing page and the post-download conversion hint rather than blindly scraping.

| Asset | License | Landing page |
|-------|---------|--------------|
| **Intel NewSponza** (Main + curtains/ivy/candles add-ons; Blender/USD/glTF/FBX) | **CC-BY 4.0** (Intel — attribution required) | https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html |
| **Amazon Lumberyard Bistro** (interior/exterior, ~4M tris; FBX/Falcor → community glTF) | **CC-BY 4.0** (via NVIDIA ORCA) | https://developer.nvidia.com/orca/amazon-lumberyard-bistro |
| **McGuire Computer Graphics Archive** (Sponza, San Miguel, Rungholt, etc.; OBJ+PNG) | **CC-BY 3.0** — cite "Morgan McGuire, Computer Graphics Archive, July 2017" | https://casual-effects.com/data/ |

Community glTF conversions of Bistro exist (e.g. `GLTF-Assets/Bistro`, `DGriffin91/bevy_bistro_scene`)
and inherit the upstream CC-BY 4.0; the fetch script lists them as a convenience but they are
third-party mirrors.

---

## License summary

- **CC0 1.0 Universal** (public domain, no attribution required): the three committed assets +
  all Tier-2A heroes.
- **CC-BY 4.0** (attribution required): Intel NewSponza, Amazon Lumberyard Bistro.
- **CC-BY 3.0** (attribution required): McGuire Computer Graphics Archive.
- **CryEngine Limited License** (© Crytek): Khronos PBR Sponza.

Always preserve the attribution for any non-CC0 asset shipped in a build.
