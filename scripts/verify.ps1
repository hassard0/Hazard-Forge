<#
.SYNOPSIS
    Full cross-platform verification for Hazard Forge: Windows/Vulkan + Mac/Metal in one command.

.DESCRIPTION
    Runs the complete verification gate and exits non-zero on ANY failure:

      1. Windows / Vulkan
         - conan install (cppstd=17 + Ninja generator), cmake configure, build, ctest.
         - All steps run inside a VS BuildTools x64 dev shell so cl/ninja resolve.

      2. Mac / Metal (headless, over SSH on the LAN)
         - tar the repo (excluding build dirs + .git + stray PNGs, KEEPING the tracked goldens),
           scp it to the Mac, extract, configure+build the metal_headless target ONCE, then for
           EACH of the 19 committed Metal goldens run visual_test with its showcase flag and compare
           the output to the matching golden with threshold 0.0 (every pair must be DIFF 0.0000).
           A per-golden table is printed; the Mac portion passes only if ALL 19 diff 0.0000.

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
$TarName    = 'hf-verify.tar.gz'

# The 19 committed Metal goldens, each produced by a distinct visual_test invocation. Name = the
# golden basename under tests/golden/metal/; Flag = the argv passed to visual_test BEFORE the output
# path (empty for the default Slice-F scene). The flags are the REAL ones parsed in
# metal_headless/visual_test.mm main() - confirmed there, not guessed. Every pair must diff 0.0000.
$Goldens = @(
    @{ Name = 'scene_shadow';  Flag = '' }                       # default visual_test <out>
    @{ Name = 'skinning';      Flag = '--skinning' }             # Slice O
    @{ Name = 'pbr_helmet';    Flag = '--pbr' }                  # Slice P
    @{ Name = 'instanced';     Flag = '--instanced' }            # Slice Q
    @{ Name = 'ibl_helmet';    Flag = '--ibl' }                  # Slice R
    @{ Name = 'physics';       Flag = '--physics' }              # Slice S
    @{ Name = 'transparency';  Flag = '--transparency' }         # Slice T
    @{ Name = 'bloom';         Flag = '--bloom' }                # Slice U
    @{ Name = 'scene_import';  Flag = '--scene' }                # Slice V
    @{ Name = 'debug_viz';     Flag = '--debug' }                # Slice W
    @{ Name = 'anim_blend';    Flag = '--blend' }                # Slice X
    @{ Name = 'ssao';          Flag = '--ssao' }                 # Slice Y
    @{ Name = 'capstone';      Flag = '--capstone' }             # Slice Z
    @{ Name = 'camera_pose';   Flag = '--camera 0.2,-0.1,0,3,10' } # Slice AA (scripted pose)
    @{ Name = 'gizmo';         Flag = '--gizmo 2' }              # Slice AB (select obj 2)
    @{ Name = 'csm';           Flag = '--csm' }                  # Slice AD (cascaded shadows)
    @{ Name = 'spot';          Flag = '--spot' }                 # Slice AE (spot-light shadows)
    @{ Name = 'point_shadow';  Flag = '--point-shadow' }         # Slice AF (omnidirectional point shadows)
    @{ Name = 'clustered';     Flag = '--clustered' }            # Slice AG (clustered / Forward+ lighting)
)

$winResult = 'SKIP'
$macResult = 'SKIP'
$script:macGoldenResults = @()   # per-golden [{ Name; Diff; Ok }] filled by Invoke-MacVerify

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

    # Sanity: EVERY golden MUST be in the archive or its compare on the Mac can't run.
    $archiveList = (& tar -tzf $tarPath)
    foreach ($g in $Goldens) {
        $needle = "golden/metal/$($g.Name).png"
        if (-not ($archiveList | Select-String -SimpleMatch $needle)) {
            throw "golden '$($g.Name).png' missing from archive - refusing to continue"
        }
    }

    # 2) recreate the remote staging dir + copy + extract (idempotent).
    Write-Host "--- scp + extract on Mac ---"
    & $ssh[0] $ssh[1..($ssh.Count-1)] "rm -rf $MacStage && mkdir -p $MacStage"
    if ($LASTEXITCODE -ne 0) { throw "remote mkdir failed" }
    & $scp[0] $scp[1..($scp.Count-1)] $tarPath "${MacUser}@${MacHost}:$MacStage/"
    if ($LASTEXITCODE -ne 0) { throw "scp failed" }

    # 3) extract + build ONCE + loop ALL 19 goldens. To avoid the login shell being zsh and to dodge
    #    PowerShell here-string backtick-escaping fragility, the per-golden loop is generated as a
    #    standalone bash script, scp'd to the Mac, and run with an explicit `bash`. For each
    #    (flag -> golden) pair it renders visual_test <flag> /tmp/hf_<name>.png and compares to
    #    tests/golden/metal/<name>.png at threshold 0.0, emitting a machine-parseable
    #    "RESULT <name> <diff> <PASS|FAIL>" line we parse on the Windows side. The loop does NOT
    #    abort on a single failure (no `set -e` inside it): a drifted golden must still yield the full
    #    table so a reviewer sees exactly which one(s) changed. Build failures abort hard.
    #    compare.sh prints "DIFF <value>" and exits 0 only when DIFF <= threshold (0.0 here).
    #
    #    The PAIRS data is "<name>|<flag>" lines (flag empty for the default Slice-F scene). LF-only
    #    line endings are required so bash reads them cleanly.
    $pairLines = ($Goldens | ForEach-Object { "$($_.Name)|$($_.Flag)" }) -join "`n"

    # NOTE: this bash body is a single-quoted PS here-string, so NOTHING in it is expanded/escaped by
    # PowerShell. The PAIRS block is injected by string-replacing the @@PAIRS@@ token afterwards.
    $bashBody = @'
#!/usr/bin/env bash
set -e
source ~/mac-remote-rig/env.sh
# Run from the staging dir (this script was scp'd into it).
cd "$(cd "$(dirname "$0")" && pwd)"
tar -xzf "$TARBALL"
cmake -S metal_headless -B build-metal -G Ninja >/dev/null
cmake --build build-metal >/dev/null
set +e
read -r -d '' PAIRS <<'PAIRS_EOF'
@@PAIRS@@
PAIRS_EOF
while IFS='|' read -r name flag; do
    [ -z "$name" ] && continue
    out="/tmp/hf_${name}.png"
    golden="tests/golden/metal/${name}.png"
    if [ -z "$flag" ]; then
        ./build-metal/visual_test "$out" >/dev/null 2>&1
    else
        # flag may be multi-token (e.g. "--camera 0.2,-0.1,0,3,10") -> intentional word-split.
        ./build-metal/visual_test $flag "$out" >/dev/null 2>&1
    fi
    if [ $? -ne 0 ]; then
        echo "RESULT $name RENDER_FAIL FAIL"
        continue
    fi
    diffline=$(~/mac-remote-rig/compare.sh "$golden" "$out" 0.0 2>&1)
    crc=$?
    # compare.sh prints e.g. "DIFF 0.0000 (threshold 0.0)" -> keep ONLY the numeric value (field 2).
    diffval=$(echo "$diffline" | sed -n 's/^DIFF \([^ ]*\).*/\1/p' | head -1)
    [ -z "$diffval" ] && diffval=NO-DIFF
    if [ $crc -eq 0 ]; then
        echo "RESULT $name $diffval PASS"
    else
        echo "RESULT $name $diffval FAIL"
    fi
done <<< "$PAIRS"
'@

    $bashScript = $bashBody.Replace('@@PAIRS@@', $pairLines)
    # Write LF-only, no BOM (bash chokes on CRLF and a UTF-8 BOM).
    $bashScript = $bashScript -replace "`r`n", "`n"
    $localSh = Join-Path $env:TEMP 'hf-verify-mac.sh'
    [System.IO.File]::WriteAllText($localSh, $bashScript, (New-Object System.Text.UTF8Encoding($false)))

    & $scp[0] $scp[1..($scp.Count-1)] $localSh "${MacUser}@${MacHost}:$MacStage/run-goldens.sh"
    if ($LASTEXITCODE -ne 0) { throw "scp of golden-runner script failed" }

    $out = & $ssh[0] $ssh[1..($ssh.Count-1)] "STAGE='$MacStage' TARBALL='$TarName' bash $MacStage/run-goldens.sh" 2>&1
    $code = $LASTEXITCODE
    $out | ForEach-Object { Write-Host $_ }

    # Parse the RESULT lines into the per-golden table.
    $parsed = @{}
    foreach ($line in $out) {
        $s = [string]$line
        if ($s -match '^RESULT\s+(\S+)\s+(\S+)\s+(PASS|FAIL)\s*$') {
            $parsed[$matches[1]] = @{ Diff = $matches[2]; Ok = ($matches[3] -eq 'PASS') }
        }
    }

    $results = @()
    $allOk = $true
    foreach ($g in $Goldens) {
        if ($parsed.ContainsKey($g.Name)) {
            $r = $parsed[$g.Name]
            $results += @{ Name = $g.Name; Diff = $r.Diff; Ok = $r.Ok }
            if (-not $r.Ok) { $allOk = $false }
        } else {
            # No RESULT line emitted for this golden - treat as a failure (build/loop aborted early).
            $results += @{ Name = $g.Name; Diff = 'NO-RESULT'; Ok = $false }
            $allOk = $false
        }
    }
    $script:macGoldenResults = $results

    if ($code -ne 0 -or -not $allOk) {
        $script:macResult = 'FAIL'
        $bad = ($results | Where-Object { -not $_.Ok } | ForEach-Object { $_.Name }) -join ', '
        Write-Host "Mac verification FAILED (remote exit $code; non-0.0000 goldens: $bad)" -ForegroundColor Red
        return
    }
    $script:macResult = 'PASS'
    Write-Host "Mac verification PASSED (all 19 goldens DIFF 0.0000)" -ForegroundColor Green
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
Show 'Mac / Metal (19 goldens)'  $macResult

# Per-golden Metal table (only when the Mac portion ran).
if ($script:macGoldenResults -and $script:macGoldenResults.Count -gt 0) {
    Write-Host ""
    Write-Host "  Metal goldens (threshold 0.0 - every diff must be 0.0000):" -ForegroundColor Cyan
    Write-Host ("    {0,-16} {1,-12} {2}" -f 'golden', 'DIFF', 'result')
    Write-Host ("    {0,-16} {1,-12} {2}" -f '------', '----', '------')
    $passCount = 0
    foreach ($r in $script:macGoldenResults) {
        $tag = if ($r.Ok) { 'PASS' } else { 'FAIL' }
        if ($r.Ok) { $passCount++ }
        $color = if ($r.Ok) { 'Green' } else { 'Red' }
        Write-Host ("    {0,-16} {1,-12} {2}" -f $r.Name, $r.Diff, $tag) -ForegroundColor $color
    }
    Write-Host ("    {0} / {1} goldens at DIFF 0.0000" -f $passCount, $script:macGoldenResults.Count)
}

if ($winResult -eq 'FAIL' -or $macResult -eq 'FAIL') {
    Write-Host ""
    Write-Host "VERIFY: FAIL" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "VERIFY: PASS" -ForegroundColor Green
exit 0
