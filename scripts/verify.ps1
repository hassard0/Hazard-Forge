<#
.SYNOPSIS
    Full cross-platform verification for Hazard Forge: Windows/Vulkan + Mac/Metal in one command.

.DESCRIPTION
    Runs the complete verification gate and exits non-zero on ANY failure:

      1. Windows / Vulkan
         - conan install (cppstd=17 + Ninja generator), cmake configure, build, ctest.
         - All steps run inside a VS BuildTools x64 dev shell so cl/ninja resolve.

      2. Mac / Metal (headless, over SSH on the LAN)
         - tar the repo (excluding build dirs + .git + stray PNGs, KEEPING the tracked golden),
           scp it to the Mac, extract, configure+build the metal_headless target, run visual_test,
           and compare the output to the committed golden with threshold 0.0 (must be DIFF 0.0000).

    Idempotent and re-runnable: build dirs are reused; the Mac staging dir is recreated each run.

.PREREQUISITES
      Windows:
        - Visual Studio 2022 BuildTools with the C++ x64 toolset at:
            C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1
        - conan 2.x and cmake 3.25+ on PATH.
        - A conan profile that can build the deps (cppstd is overridden to 17 by this script).
      Mac (one-time setup, already done on the bench Mac):
        - Passwordless SSH as ianhassard@192.168.4.215 using ~/.ssh/id_ed25519 (LAN-only).
        - ~/mac-remote-rig/{env.sh,compare.sh} present (Metal/MoltenVK env + PNG diff tool).
        - Xcode Command Line Tools + Homebrew glslc/spirv-cross/cmake/ninja/python3.

.NOTES
    Run from anywhere; the script locates the repo root as its own parent's parent.
    Does NOT re-bake any golden. Rendering behavior is never modified.
#>

[CmdletBinding()]
param(
    # Skip a platform if you only want to verify one side (both run by default).
    [switch]$SkipWindows,
    [switch]$SkipMac
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- Config ----------------------------------------------------------------------------------------
$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$VsDevShell = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1'

$MacUser    = 'ianhassard'
$MacHost    = '192.168.4.215'
$SshKey     = "$env:USERPROFILE\.ssh\id_ed25519"
$MacStage   = '~/hf-verify'                       # remote staging dir (recreated each run)
$Golden     = 'tests/golden/metal/scene_shadow.png'
$TarName    = 'hf-verify.tar.gz'

$winResult = 'SKIP'
$macResult = 'SKIP'

function Write-Section($t) { Write-Host ""; Write-Host "==== $t ====" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------------------------------
# Windows / Vulkan
# ---------------------------------------------------------------------------------------------------
function Invoke-WindowsVerify {
    Write-Section "WINDOWS / VULKAN"

    if (-not (Test-Path $VsDevShell)) {
        throw "VS BuildTools dev shell not found at: $VsDevShell"
    }

    # Build a single child-shell script: enter the VS dev shell, then run the full pipeline.
    # `$LASTEXITCODE` is checked after each native step so the first failure aborts with non-zero.
    # NOTE: do NOT set `$ErrorActionPreference='Stop' here. Launch-VsDevShell shells out to
    # vswhere and can emit a benign native-command warning to stderr; under 'Stop' that aborts
    # the shell. We gate every step on `$LASTEXITCODE explicitly instead, which is exact.
    $inner = @"
`$ErrorActionPreference = 'Continue'
& '$VsDevShell' -Arch amd64 2>`$null | Out-Null
Set-Location '$RepoRoot'

Write-Host '--- conan install ---'
conan install . -of build/windows-msvc-debug -s build_type=Debug -s compiler.cppstd=17 ``
    -c tools.cmake.cmaketoolchain:generator=Ninja --build=missing
if (`$LASTEXITCODE -ne 0) { exit 11 }

# Conan appends each build's generated preset to CMakeUserPresets.json. If an ASan build was also
# configured (build/windows-msvc-asan), both conan presets are named 'conan-debug' and CMake
# refuses to read presets ('Duplicate preset'). Pin CMakeUserPresets.json to the debug include only
# so the verify run is deterministic regardless of what else has been built. NOTE: written WITHOUT a
# BOM -- conan's preset reader does json.loads() and chokes on a UTF-8 BOM.
`$pinned = @'
{
    "version": 4,
    "vendor": { "conan": {} },
    "include": [ "build/windows-msvc-debug/build/Debug/generators/CMakePresets.json" ]
}
'@
[System.IO.File]::WriteAllText((Join-Path (Get-Location) 'CMakeUserPresets.json'), `$pinned, (New-Object System.Text.UTF8Encoding(`$false)))

Write-Host '--- cmake configure ---'
cmake --preset windows-msvc-debug
if (`$LASTEXITCODE -ne 0) { exit 12 }

Write-Host '--- cmake build ---'
cmake --build --preset windows-msvc-debug
if (`$LASTEXITCODE -ne 0) { exit 13 }

Write-Host '--- ctest ---'
ctest --preset windows-msvc-debug
if (`$LASTEXITCODE -ne 0) { exit 14 }

exit 0
"@

    # Run the inner pipeline in a fresh powershell so the dev-shell env is isolated per run.
    # Invoke via a temp -File (not -Command) and merge the child's stderr into stdout so a benign
    # native warning (e.g. vswhere) cannot bubble up as a terminating ErrorRecord under the outer
    # $ErrorActionPreference='Stop'. We rely solely on the child's exit code for pass/fail.
    $innerFile = Join-Path $env:TEMP 'hf-verify-win.ps1'
    Set-Content -LiteralPath $innerFile -Value $inner -Encoding UTF8
    $logFile = Join-Path $env:TEMP 'hf-verify-win.log'

    # Locally relax error handling and route the child's stderr to a file so a benign native
    # warning cannot terminate the outer 'Stop' scope. We trust only the child exit code.
    $savedEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $innerFile > $logFile 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $savedEAP

    if (Test-Path $logFile) { Get-Content $logFile | ForEach-Object { Write-Host $_ } }
    if ($code -ne 0) {
        $script:winResult = 'FAIL'
        Write-Host "Windows verification FAILED (stage exit code $code)" -ForegroundColor Red
        return
    }
    $script:winResult = 'PASS'
    Write-Host "Windows verification PASSED" -ForegroundColor Green
}

# ---------------------------------------------------------------------------------------------------
# Mac / Metal
# ---------------------------------------------------------------------------------------------------
function Invoke-MacVerify {
    Write-Section "MAC / METAL"

    if (-not (Test-Path $SshKey)) { throw "SSH key not found at: $SshKey" }

    $ssh = @('ssh', '-i', $SshKey, '-o', 'StrictHostKeyChecking=accept-new', "$MacUser@$MacHost")
    $scp = @('scp', '-i', $SshKey, '-o', 'StrictHostKeyChecking=accept-new', '-q')

    # 1) tar the repo. Keep the tracked golden; drop build dirs, .git, conan user presets, and the
    #    stray top-level bring-up PNGs (NOT the golden under tests/golden/). Uses git-bash tar via WSL?
    #    No - we use the bsdtar shipped with Windows 10/11 (tar.exe on PATH).
    $tarPath = Join-Path $env:TEMP $TarName
    if (Test-Path $tarPath) { Remove-Item $tarPath -Force }

    Write-Host "--- tar repo (keeping golden) ---"
    Push-Location $RepoRoot
    try {
        # bsdtar: exclude build artifacts + .git + known stray PNGs. The golden under
        # tests/golden/ is intentionally NOT matched by these patterns.
        & tar `
            --exclude='./build' `
            --exclude='./build-metal' `
            --exclude='./.git' `
            --exclude='./CMakeUserPresets.json' `
            --exclude='./hf_editor.png' `
            --exclude='./hf_nmap_crop.png' `
            --exclude='./hf_nmap_cube.png' `
            --exclude='./hf_nmap_cubeface.png' `
            --exclude='./hf_nmap_metal_cube.png' `
            -czf $tarPath .
        if ($LASTEXITCODE -ne 0) { throw "tar failed" }
    } finally { Pop-Location }

    # Sanity: the golden MUST be in the archive or the compare on the Mac can't run.
    $hasGolden = (& tar -tzf $tarPath) | Select-String -SimpleMatch 'golden/metal/scene_shadow.png'
    if (-not $hasGolden) { throw "golden PNG missing from archive - refusing to continue" }

    # 2) recreate the remote staging dir + copy + extract (idempotent).
    Write-Host "--- scp + extract on Mac ---"
    & $ssh[0] $ssh[1..($ssh.Count-1)] "rm -rf $MacStage && mkdir -p $MacStage"
    if ($LASTEXITCODE -ne 0) { throw "remote mkdir failed" }
    & $scp[0] $scp[1..($scp.Count-1)] $tarPath "${MacUser}@${MacHost}:$MacStage/"
    if ($LASTEXITCODE -ne 0) { throw "scp failed" }

    # 3) extract + build + run + compare, all in one remote shell. The remote script echoes a
    #    final DIFF line; we both stream it and parse it. `set -e` aborts on the first failure.
    $remote = @"
set -e
source ~/mac-remote-rig/env.sh
cd $MacStage
tar -xzf $TarName
cmake -S metal_headless -B build-metal -G Ninja >/dev/null
cmake --build build-metal >/dev/null
./build-metal/visual_test /tmp/verify.png
~/mac-remote-rig/compare.sh $Golden /tmp/verify.png 0.0
"@

    $out = & $ssh[0] $ssh[1..($ssh.Count-1)] $remote 2>&1
    $code = $LASTEXITCODE
    $out | ForEach-Object { Write-Host $_ }

    # compare.sh exits 0 only when DIFF <= threshold (0.0 here), and prints "DIFF 0.0000".
    $diffOk = ($out | Select-String -SimpleMatch 'DIFF 0.0000')
    if ($code -ne 0 -or -not $diffOk) {
        $script:macResult = 'FAIL'
        Write-Host "Mac verification FAILED (exit $code; golden DIFF not 0.0000)" -ForegroundColor Red
        return
    }
    $script:macResult = 'PASS'
    Write-Host "Mac verification PASSED (golden DIFF 0.0000)" -ForegroundColor Green
}

# ---------------------------------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------------------------------
$failed = $false

if (-not $SkipWindows) {
    try { Invoke-WindowsVerify } catch { $script:winResult = 'FAIL'; Write-Host "Windows verify error: $_" -ForegroundColor Red }
}
if (-not $SkipMac) {
    try { Invoke-MacVerify } catch { $script:macResult = 'FAIL'; Write-Host "Mac verify error: $_" -ForegroundColor Red }
}

Write-Section "SUMMARY"
function Show($label, $r) {
    $color = if ($r -eq 'PASS') { 'Green' } elseif ($r -eq 'SKIP') { 'Yellow' } else { 'Red' }
    Write-Host ("  {0,-22} {1}" -f $label, $r) -ForegroundColor $color
}
Show 'Windows / Vulkan (ctest)' $winResult
Show 'Mac / Metal (golden 0.0)' $macResult

if ($winResult -eq 'FAIL' -or $macResult -eq 'FAIL') {
    Write-Host ""
    Write-Host "VERIFY: FAIL" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "VERIFY: PASS" -ForegroundColor Green
exit 0
