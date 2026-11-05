@echo off
rem Build the Windows version of Equestria Atlas: dist\Equestria-Atlas-windows.zip
rem Needs Python 3.11 or newer from https://www.python.org (tick "Add python.exe to PATH" when installing).
rem Double-click this file, or run it from a Command Prompt. It takes a few minutes the first time.
setlocal
cd /d "%~dp0\.."
where py >nul 2>nul && (set PYLAUNCH=py -3) || (set PYLAUNCH=python)
if not exist .venv-win\Scripts\python.exe (
    echo Creating a Python environment...
    %PYLAUNCH% -m venv .venv-win || goto :fail
)
set PY=.venv-win\Scripts\python.exe
echo Installing the libraries (numpy, scipy, pillow, pygame, pyinstaller)...
%PY% -m pip install -q --upgrade pip
%PY% -m pip install -q -r packaging\requirements.txt || goto :fail
echo Building the game...
%PY% -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging\equestria_atlas.spec || goto :fail
echo Checking the build (loads every place)...
"dist\Equestria Atlas\Equestria Atlas.exe" --selftest
type "%APPDATA%\Equestria Atlas\selftest.txt"
copy /y "packaging\READ ME FIRST.txt" "dist\Equestria Atlas\" >nul
echo Zipping...
if exist dist\Equestria-Atlas-windows.zip del dist\Equestria-Atlas-windows.zip
powershell -NoProfile -Command "Compress-Archive -Path 'dist\Equestria Atlas' -DestinationPath 'dist\Equestria-Atlas-windows.zip'" || goto :fail
echo.
echo Done: dist\Equestria-Atlas-windows.zip  (share this file)
pause
exit /b 0
:fail
echo.
echo Build FAILED - see the messages above.
pause
exit /b 1
