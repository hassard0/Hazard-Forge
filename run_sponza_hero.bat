@echo off
REM Renders the real Khronos PBR Sponza hero scene and opens the image.
REM This is a HEADLESS render (no window pops up) -> it writes an image file, then opens it.
setlocal
cd /d "%~dp0"

set EXE=build\windows-msvc-release\samples\hello_triangle\hello_triangle.exe
set OUT=%TEMP%\sponza_hero.bmp

if not exist "%EXE%" (
  echo [x] The release build is missing: %EXE%
  echo     Build it first, then re-run this script.
  pause & exit /b 1
)
if not exist "assets\reference\_downloaded\Sponza\Sponza.gltf" (
  echo [x] The Sponza asset is not fetched.
  echo     Run:  powershell -ExecutionPolicy Bypass -File assets\reference\fetch_reference_assets.ps1 --sponza
  pause & exit /b 1
)

echo Rendering the real Sponza (headless, ~4 seconds)...
"%EXE%" --sc1-hero-shot "%OUT%"
if errorlevel 1 (
  echo [x] Render failed - see the message above.
  pause & exit /b 1
)

echo Done. Opening %OUT% ...
start "" "%OUT%"
endlocal
