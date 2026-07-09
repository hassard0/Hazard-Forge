@echo off
REM Slice GDW1 -- opens an INTERACTIVE window and runs the GAME1 deterministic rollback-physics DUEL
REM live: a 2-player local (hotseat) sumo/knockout match at a fixed deterministic timestep, rendered
REM LIT 3D. This is a LIVE window (verify-by-launch, like --sponza-explore) -> no image is written.
REM The match-deciding round is RECORDED to duel_play_match.demo.json (replay it with the -replay flag).
setlocal
cd /d "%~dp0"

set EXE=build\windows-msvc-release\samples\hello_triangle\hello_triangle.exe

if not exist "%EXE%" (
  echo [x] The release build is missing: %EXE%
  echo     Build it first, then re-run this script.
  pause ^& exit /b 1
)

echo Hazard Forge -- DETERMINISTIC ROLLBACK-PHYSICS DUEL (GAME1, playable):
echo    shove your opponent past the ring edge (^|x^|^>5) or into the void to win a round; best-of-3.
echo.
echo  Player 1:  A / D        move left / right
echo             Space or F   SHOVE
echo  Player 2:  Left / Right move left / right
echo             Enter or Shift  SHOVE
echo    R              restart the match
echo    ESC            quit
echo.
echo Launching (a window will open)...
"%EXE%" --duel-play
endlocal
