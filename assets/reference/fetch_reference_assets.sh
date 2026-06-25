#!/usr/bin/env bash
# Hazard Forge — on-demand reference-asset downloader (issue #18).
#
# Pulls the HEAVY hero scenes listed in MANIFEST.md into assets/reference/_downloaded/
# (gitignored, so a fetch never bloats the repo). The small CC0 assets are already committed
# in-repo (assets/reference/*.glb) and need no download.
#
# Usage:
#   ./fetch_reference_assets.sh            # default: fetch the CC0 hero .glb set (Tier 2A)
#   ./fetch_reference_assets.sh --heroes   # same as default
#   ./fetch_reference_assets.sh --sponza   # also fetch the Khronos PBR Sponza directory (CryEngine license)
#   ./fetch_reference_assets.sh --all      # heroes + sponza
#   ./fetch_reference_assets.sh --list     # print the manual/EULA-gated heroes (NewSponza, Bistro, McGuire)
#
# After fetching, render any asset through the engine:
#   hello_triangle --reference-shot assets/reference/_downloaded/Avocado.glb out.bmp
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/_downloaded"
mkdir -p "$OUT"

RAW="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"

# name|url  — the Tier-2A CC0 single-file heroes (stable raw URLs).
HEROES=(
  "Avocado.glb|$RAW/Avocado/glTF-Binary/Avocado.glb"
  "WaterBottle.glb|$RAW/WaterBottle/glTF-Binary/WaterBottle.glb"
  "BoomBox.glb|$RAW/BoomBox/glTF-Binary/BoomBox.glb"
  "Lantern.glb|$RAW/Lantern/glTF-Binary/Lantern.glb"
  "BarramundiFish.glb|$RAW/BarramundiFish/glTF-Binary/BarramundiFish.glb"
  "Corset.glb|$RAW/Corset/glTF-Binary/Corset.glb"
)

fetch_one() {
  local name="$1" url="$2"
  echo "  -> $name"
  curl -fSL --retry 3 -o "$OUT/$name" "$url"
}

fetch_heroes() {
  echo "[fetch] Tier-2A CC0 hero .glb set -> $OUT"
  for entry in "${HEROES[@]}"; do
    fetch_one "${entry%%|*}" "${entry#*|}"
  done
}

fetch_sponza() {
  echo "[fetch] Khronos PBR Sponza (CryEngine Limited License, (c) Crytek) -> $OUT/Sponza"
  mkdir -p "$OUT/Sponza"
  local base="$RAW/Sponza/glTF"
  # The Sponza glTF is multi-file: the .gltf + its single .bin + the texture set. We grab the
  # .gltf, parse its buffer/image "uri" fields, and pull each referenced file alongside it.
  curl -fSL --retry 3 -o "$OUT/Sponza/Sponza.gltf" "$base/Sponza.gltf"
  echo "  parsing referenced buffers + textures..."
  # Extract every "uri": "<file>" value (buffers + images), de-dup, fetch each.
  grep -oE '"uri"[[:space:]]*:[[:space:]]*"[^"]+"' "$OUT/Sponza/Sponza.gltf" \
    | sed -E 's/.*"uri"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/' \
    | sort -u \
    | while read -r uri; do
        [ -z "$uri" ] && continue
        echo "    -> $uri"
        mkdir -p "$OUT/Sponza/$(dirname "$uri")"
        curl -fSL --retry 3 -o "$OUT/Sponza/$uri" "$base/$uri"
      done
  echo "  Sponza ready: $OUT/Sponza/Sponza.gltf"
}

list_manual() {
  cat <<'EOF'
[manual] EULA / browser-gated heroes (accept the license on the vendor site, then drop the
         download into assets/reference/_downloaded/ and point --reference-shot at the .glb/.gltf):

  Intel NewSponza (CC-BY 4.0, attribution required)
    https://www.intel.com/content/www/us/en/developer/topic-technology/graphics-research/samples.html

  Amazon Lumberyard Bistro (CC-BY 4.0 via NVIDIA ORCA; FBX/Falcor -> convert to glTF)
    https://developer.nvidia.com/orca/amazon-lumberyard-bistro
    community glTF mirror: https://github.com/DGriffin91/bevy_bistro_scene

  McGuire Computer Graphics Archive (CC-BY 3.0; cite "Morgan McGuire, Computer Graphics Archive, July 2017")
    https://casual-effects.com/data/

  See MANIFEST.md for the full license table.
EOF
}

MODE="${1:---heroes}"
case "$MODE" in
  --heroes|"") fetch_heroes ;;
  --sponza)    fetch_sponza ;;
  --all)       fetch_heroes; fetch_sponza ;;
  --list)      list_manual ;;
  *) echo "unknown option: $MODE (use --heroes | --sponza | --all | --list)"; exit 2 ;;
esac

echo "[done] downloaded assets are in $OUT (gitignored)."
list_manual
