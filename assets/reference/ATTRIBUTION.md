# Reference asset attribution (in-repo, committed)

These small, self-contained `.glb` files are committed directly into the repository as the
built-in reference assets every sample can rely on. **Every committed file here is CC0 1.0
Universal (public domain)** — verified against each model's `README.md` in
[KhronosGroup/glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets) before
committing. The heavier hero scenes (Sponza, Bistro, NewSponza, the McGuire archive, and the
larger Khronos CC0 PBR heroes) are NOT committed — they are listed in `MANIFEST.md` and pulled
on demand by `fetch_reference_assets.sh` / `fetch_reference_assets.ps1` into `_downloaded/`
(gitignored).

| File | Source | Author / © | License | Notes |
|------|--------|-----------|---------|-------|
| `BoxVertexColors.glb` | Khronos glTF-Sample-Assets / Models/BoxVertexColors | © 2023, Public | **CC0 1.0 Universal** | Vertex-colored cube; no external textures. The minimal load smoke asset. |
| `TextureCoordinateTest.glb` | Khronos glTF-Sample-Assets / Models/TextureCoordinateTest | © 2017, Analytical Graphics, Inc. | **CC0 1.0 Universal** | Textured quad grid (embedded base-color image) — exercises the texture-decode + UV path. |
| `TextureLinearInterpolationTest.glb` | Khronos glTF-Sample-Assets / Models/TextureLinearInterpolationTest | © 2017, Public | **CC0 1.0 Universal** | Textured planes with an embedded image; exercises sampler filtering. |

## Why these three (the git-size call)

The classic textured CC0 PBR "hero" models from Khronos (Avocado, WaterBottle, BoomBox,
Lantern, Corset, BarramundiFish) are all **8–14 MB** each (high-res embedded PBR texture sets),
and Sponza/Bistro-class scenes are tens to hundreds of MB. Committing those would bloat the git
history permanently. Instead we commit a few **genuinely CC0, kilobyte-scale** assets in-repo
(so every sample has a real, self-contained model with zero network), and ship the heavy heroes
through `MANIFEST.md` + the on-demand `fetch_reference_assets` scripts (downloaded into the
gitignored `_downloaded/`). The library is therefore complete *and* honest about git weight.

## A note on the existing `assets/models/` heroes

`assets/models/Duck.glb` (SCEA Shared Source, © Sony), `CesiumMilkTruck.glb` (CC-BY, Cesium) and
`DamagedHelmet.glb` (CC-BY) predate this library and are NOT CC0, so they are intentionally not
mirrored here. This `assets/reference/` library is the CC0-clean set.

CC0 1.0 Universal full text: https://creativecommons.org/publicdomain/zero/1.0/legalcode
