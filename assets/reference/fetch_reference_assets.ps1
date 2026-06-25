<#
  Hazard Forge - on-demand reference-asset downloader (issue #18, Windows/PowerShell twin of
  fetch_reference_assets.sh).

  Pulls the HEAVY hero scenes from MANIFEST.md into assets/reference/_downloaded/ (gitignored).
  The small CC0 assets are already committed in-repo and need no download.

  Usage:
    .\fetch_reference_assets.ps1            # default: fetch the CC0 hero .glb set (Tier 2A)
    .\fetch_reference_assets.ps1 -Heroes    # same as default
    .\fetch_reference_assets.ps1 -Sponza    # also fetch the Khronos PBR Sponza directory (CryEngine license)
    .\fetch_reference_assets.ps1 -All       # heroes + sponza
    .\fetch_reference_assets.ps1 -List      # print the manual/EULA-gated heroes

  After fetching, render any asset through the engine:
    hello_triangle --reference-shot assets/reference/_downloaded/Avocado.glb out.bmp
#>
[CmdletBinding()]
param(
  [switch]$Heroes,
  [switch]$Sponza,
  [switch]$All,
  [switch]$List
)
$ErrorActionPreference = 'Stop'

$Dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Out = Join-Path $Dir '_downloaded'
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$Raw = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models'

$HeroSet = @(
  @{ name = 'Avocado.glb';        url = "$Raw/Avocado/glTF-Binary/Avocado.glb" }
  @{ name = 'WaterBottle.glb';    url = "$Raw/WaterBottle/glTF-Binary/WaterBottle.glb" }
  @{ name = 'BoomBox.glb';        url = "$Raw/BoomBox/glTF-Binary/BoomBox.glb" }
  @{ name = 'Lantern.glb';        url = "$Raw/Lantern/glTF-Binary/Lantern.glb" }
  @{ name = 'BarramundiFish.glb'; url = "$Raw/BarramundiFish/glTF-Binary/BarramundiFish.glb" }
  @{ name = 'Corset.glb';         url = "$Raw/Corset/glTF-Binary/Corset.glb" }
)

function Fetch-Heroes {
  Write-Host "[fetch] Tier-2A CC0 hero .glb set -> $Out"
  foreach ($h in $HeroSet) {
    Write-Host "  -> $($h.name)"
    Invoke-WebRequest -Uri $h.url -OutFile (Join-Path $Out $h.name) -UseBasicParsing
  }
}

function Fetch-Sponza {
  Write-Host "[fetch] Khronos PBR Sponza (CryEngine Limited License, (c) Crytek) -> $Out\Sponza"
  $spDir = Join-Path $Out 'Sponza'
  New-Item -ItemType Directory -Force -Path $spDir | Out-Null
  $base = "$Raw/Sponza/glTF"
  $gltfPath = Join-Path $spDir 'Sponza.gltf'
  Invoke-WebRequest -Uri "$base/Sponza.gltf" -OutFile $gltfPath -UseBasicParsing
  Write-Host '  parsing referenced buffers + textures...'
  $text = Get-Content -Raw $gltfPath
  $uris = [regex]::Matches($text, '"uri"\s*:\s*"([^"]+)"') |
          ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
  foreach ($uri in $uris) {
    if ([string]::IsNullOrWhiteSpace($uri)) { continue }
    Write-Host "    -> $uri"
    $dest = Join-Path $spDir $uri
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
    Invoke-WebRequest -Uri "$base/$uri" -OutFile $dest -UseBasicParsing
  }
  Write-Host "  Sponza ready: $gltfPath"
}

function List-Manual {
  Write-Host @'
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
'@
}

if ($List)        { List-Manual; return }
if ($All)         { Fetch-Heroes; Fetch-Sponza }
elseif ($Sponza)  { Fetch-Sponza }
else              { Fetch-Heroes }   # default / -Heroes

Write-Host "[done] downloaded assets are in $Out (gitignored)."
List-Manual
