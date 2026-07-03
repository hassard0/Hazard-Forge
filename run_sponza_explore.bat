@echo off
REM Opens an INTERACTIVE window and flies a camera through the real Khronos PBR Sponza scene.
REM This is a LIVE window (not a headless render) -> no image is written; you drive the camera.
setlocal
cd /d "%~dp0"

set EXE=build\windows-msvc-release\samples\hello_triangle\hello_triangle.exe

if not exist "%EXE%" (
  echo [x] The release build is missing: %EXE%
  echo     Build it first, then re-run this script.
  pause ^& exit /b 1
)
if not exist "assets\reference\_downloaded\Sponza\Sponza.gltf" (
  echo [x] The Sponza asset is not fetched.
  echo     Run:  powershell -ExecutionPolicy Bypass -File assets\reference\fetch_reference_assets.ps1 --sponza
  pause ^& exit /b 1
)

echo Sponza fly-through -- controls:
echo    WASD          move
echo    hold RIGHT mouse  look around
echo    mouse wheel   adjust move speed (if bound)
echo    ESC           quit
echo.
echo Launching (a window will open)...
"%EXE%" --sponza-explore
endlocal
